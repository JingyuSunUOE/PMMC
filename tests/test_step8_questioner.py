from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pmmc.compiler.opportunities import local_opportunity_id
from pmmc.compiler.questioner import (
    QuestionerCandidateGate,
    parse_questioner_output,
    questioner_output_parser,
)
from pmmc.compiler.questioner_config import load_questioner_config
from pmmc.compiler.questioner_native import (
    NativeSupportError,
    allocate_opportunities,
    information_need_key,
    native_response_schema,
    native_static_codes,
    native_support_catalog,
    questioner_verifier_v7_output_parser,
    resolve_v36_draft,
)
from pmmc.compiler.questioner_prompt import render_questioner_verifier_request
from pmmc.compiler.questioner_runner import QuestionerCompilationGraph
from pmmc.compiler.units import create_compilation_unit_builder
from pmmc.infrastructure.database.questioner_mappers import (
    questioner_answer_claim_from_row,
    questioner_answer_claim_to_row,
)
from pmmc.infrastructure.database.questioner_repositories import _candidate_family_dedupe_version
from pmmc.models.execution import ModelOutputContractError
from pmmc.models.tokenization import HeuristicTokenCounter
from pmmc.schemas import (
    AppStage,
    CandidateGateStatus,
    CandidateRejectionCode,
    DatasetName,
    ExperimentJob,
    MemoryUnit,
    ModelCallTrace,
    QuestionOpportunityType,
    SessionRecord,
    SpeakerRole,
    TraceStatus,
)
from pmmc.schemas.questioner import (
    QuestionerAnswerableDraftV36,
    QuestionerControlDraftV36,
    QuestionerNativeSupport,
    QuestionerVerificationDimensionsV6,
)

ROOT = Path(__file__).resolve().parents[1]


def build(
    text="I bought a Yamaha P45 piano. I practice piano daily.",
    speaker=SpeakerRole.USER,
    *,
    memory_updates=None,
    run_id="run-native",
    policy="allow",
):
    config = load_questioner_config(ROOT / "configs/compiler/questioner-v3.6.yaml")
    config = type(config).model_validate({**config.model_dump(), "memory_raw_image_policy": policy})
    memory = MemoryUnit(
        memory_id="memory-native",
        dataset=DatasetName.MEMLENS,
        conversation_id="conversation-native",
        session_id="session-native",
        turn_id="turn-native",
        timestamp_order=0,
        speaker=speaker,
        text=text,
        available_after_session="session-native",
    )
    if memory_updates:
        memory = MemoryUnit.model_validate({**memory.model_dump(), **memory_updates})
    session = SessionRecord(
        session_id="session-native",
        dataset=DatasetName.MEMLENS,
        conversation_id="conversation-native",
        session_order=0,
        source_session_id="source-native",
        source_date_iso="2026-01-01",
    )
    built = create_compilation_unit_builder(
        config=config,
        dataset=DatasetName.MEMLENS,
        token_counter=HeuristicTokenCounter("test", "native-test-v1"),
    ).build(run_id=run_id, snapshot_id="snapshot-native", memories=[memory], sessions=[session])[0]
    return config, built


@pytest.mark.parametrize("text", ['{"candidates":[', '{"candidates":[],"control_candidates":[]}'])
def test_native_length_stop_is_never_an_accepted_generation(text):
    config, _ = build()
    with pytest.raises(ModelOutputContractError, match="questioner_output_truncated"):
        questioner_output_parser(config)(SimpleNamespace(text=text, finish_reason="length"))


@pytest.mark.parametrize("text,code", [
    ('{"candidates":[', "native_json_required"),
    ('[]', "native_envelope_invalid"),
    ('{"candidates":[]}', "native_envelope_invalid"),
    ('{"candidates":[],"control_candidates":null}', "native_envelope_invalid"),
])
def test_native_invalid_envelope_has_auditable_contract_type(text, code):
    config, _ = build()
    with pytest.raises(ModelOutputContractError, match=code):
        questioner_output_parser(config)(SimpleNamespace(text=text, finish_reason="stop"))


def test_native_empty_envelope_remains_a_valid_observed_zero():
    config, _ = build()
    value = questioner_output_parser(config)(SimpleNamespace(
        text='{"candidates":[],"control_candidates":[]}', finish_reason="stop",
    )).value
    assert value["raw_candidate_count"] == 0
    assert value["candidates"] == []


@pytest.mark.parametrize("error", [ValueError("implementation_bug"), MemoryError("private")])
def test_native_parser_implementation_error_is_not_scientific_rejection(monkeypatch, error):
    import pmmc.compiler.questioner as module

    config, _ = build()

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, "parse_questioner_output", broken)
    with pytest.raises(type(error)) as captured:
        questioner_output_parser(config)(SimpleNamespace(text="{}", finish_reason="stop"))
    assert captured.value is error
    assert not isinstance(captured.value, ModelOutputContractError)


@pytest.mark.parametrize("code", ["questioner_output_truncated", "native_json_required"])
@pytest.mark.parametrize("archived", [False, True])
def test_native_final_generation_failure_is_not_a_legitimate_empty_answer(code, capsys, archived):
    from unittest.mock import AsyncMock

    from pmmc.compiler.questioner_runner import questioner_model_call_id
    from pmmc.schemas import QuestionerParseStatus

    config, built = build()
    job = ExperimentJob(
        job_id="job-native-invalid", run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id, compiler_unit_id=built.unit.compiler_unit_id,
        stage=AppStage.COMPILE, attempt_count=2, max_attempts=2,
    )
    trace = ModelCallTrace(
        model_call_id=questioner_model_call_id(run_id=job.run_id,
            compiler_unit_id=built.unit.compiler_unit_id, attempt_number=2),
        run_id=job.run_id, snapshot_id=job.snapshot_id, job_id=job.job_id,
        compiler_unit_id=built.unit.compiler_unit_id, stage=AppStage.QUESTIONER,
        model_alias=config.compiler_model_alias, prompt_hash="a" * 64,
        status=TraceStatus.FAILED, output_asset_id="raw-invalid",
        parse_error=code, error_type="ModelOutputContractError",
        metadata={"output_contract_error_code": code},
    )

    traces = {trace.model_call_id: trace} if archived else {}

    async def existing(call_id):
        return traces.get(call_id)

    async def reject(*args, **kwargs):
        traces[trace.model_call_id] = trace
        raise ModelOutputContractError(code)

    execution = SimpleNamespace(generate_audited=AsyncMock(
        side_effect=AssertionError("new call") if archived else reject,
    ))
    seed, final = AsyncMock(), AsyncMock()
    graph = QuestionerCompilationGraph(
        execution=execution, gateway=object(), config=config,
        load_successful_call=AsyncMock(return_value=None), load_model_call=existing,
        before_gate=AsyncMock(), persist_v3_seed=seed, persist_result=final,
    )
    result = asyncio.run(graph.process(built, job))
    assert execution.generate_audited.await_count == (0 if archived else 1)
    assert result.candidates == ()
    assert result.generation.parse_status is QuestionerParseStatus.FAILED
    metadata = result.generation.metadata
    assert metadata["provider_item_count_status"] == "unavailable_invalid_envelope"
    assert metadata["declared_empty"] is False
    assert metadata["scientific_acceptance"] is False
    assert metadata["terminal_outcome"] == "generation_contract_failure"
    assert metadata["model_output_truncated"] == (code == "questioner_output_truncated")
    counts = result.stage_result.metadata["questioner_funnel_v1"]["counts"]
    assert counts["provider_items"] is None and counts["dedup_eligible"] == 0
    assert seed.await_count == final.await_count == 1
    assert code in capsys.readouterr().out


def draft(
    built,
    *,
    question="Which piano model did I buy?",
    answer="A Yamaha P45 piano.",
    span="T001",
    answer_type="entity",
    subject="user",
    assertion="fact",
):
    opportunity = next(
        item
        for item in built.opportunities.opportunities
        if item.opportunity_type is QuestionOpportunityType.LOCAL_FACT
    )
    return QuestionerAnswerableDraftV36(
        candidate_kind="answerable",
        opportunity_id=local_opportunity_id(opportunity),
        generated_question=question,
        answer_type=answer_type,
        answer_claims=[
            dict(
                claim_kind="text",
                support_alias=span,
                claim_text=answer,
                subject=subject,
                assertion=assertion,
            )
        ],
        confidence=0.9,
    )


def resolve(built, candidate=None):
    return resolve_v36_draft(
        candidate or draft(built), opportunity_set=built.opportunities, memories=built.memories
    )


def verdict(candidate, **updates):
    payload = dict(
        candidate_alias="Q01",
        dimensions={name: "true" for name in QuestionerVerificationDimensionsV6.model_fields},
        answer_question_closed=True,
        same_entity_attribute="not_applicable",
        explicit_count_set="not_applicable",
        claims=[
            dict(
                claim_index=i,
                support_alias=claim.support_alias,
                span_exists=True,
                claim_entailed=True,
                speaker_subject_aligned=True,
            )
            for i, claim in enumerate(candidate.exact_claims)
        ],
    )
    payload.update(updates)
    return {"result": payload}


def test_native_versions_and_no_paraphrase_requirement():
    config, built = build()
    assert config.is_v36 and config.is_v3 and config.compiler_schema_version == "0.9.0"
    candidate = draft(built)
    parsed = parse_questioner_output(
        json.dumps({"candidates": [candidate.model_dump(mode="json")], "control_candidates": []}),
        parser_version=config.parser_version,
    )
    assert parsed.v36_candidates == (candidate,)
    assert parsed.raw_candidate_count == 1 and parsed.invalid_candidate_count == 0
    assert resolve(built).internal.paraphrases == []
    assert (
        _candidate_family_dedupe_version(
            compiler_schema_version="0.9.0", questioner_prompt_version=config.prompt_version
        )
        == "qbank-family-dedupe-v5"
    )


@pytest.mark.parametrize("changes", [
    {}, {"reasoning_type": "multi_hop"}, {"reasoning_type": "temporal"},
    {"reasoning_type": "update"}, {"reasoning_type": "conflict"},
    {"answer_type": "count"}, {"image_set_scope": True},
    {"required_modalities": ["image"]},
])
def test_native_verifier_schema_and_parser_share_required_dimensions(changes):
    import jsonschema

    from pmmc.compiler.questioner_native import verifier_required_dimensions
    from pmmc.schemas.questioner import QuestionerVerifierCandidateV36

    config, built = build()
    base = resolve(built).provider_for_verifier
    candidate = QuestionerVerifierCandidateV36.model_validate({**base.model_dump(), **changes})
    request = render_questioner_verifier_request(
        request_id="v7-contract", model_alias=config.compiler_model_alias,
        unit_id=built.unit.compiler_unit_id, cutoff_session_id="session-native",
        memories=list(built.memories), sessions=list(built.sessions),
        opportunity_set=built.opportunities, candidates=[("Q01", candidate)],
        final_round=False, max_output_tokens=1024,
        questioner_prompt_version=config.prompt_version,
        questioner_renderer_version=config.renderer_version,
        verifier_prompt_version=config.verifier_prompt_version,
    )
    schema = request.response_format["json_schema"]["schema"]
    payload = json.loads(request.messages[1].parts[0].text)
    required, closure = verifier_required_dimensions(candidate)
    assert payload["required_boolean_dimensions"] == list(required)
    assert payload["required_boolean_closure_fields"] == list(closure)
    parser = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=candidate)
    good = verdict(candidate, **dict.fromkeys(closure, "true"))
    jsonschema.validate(good, schema)
    parser(SimpleNamespace(text=json.dumps(good)))
    for group, names in (("dimensions", required), (None, closure)):
        for name in names:
            bad = json.loads(json.dumps(good))
            target = bad["result"][group] if group else bad["result"]
            target[name] = "not_applicable"
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(bad, schema)
            with pytest.raises(
                ModelOutputContractError, match="verifier_required_dimension_missing",
            ):
                parser(SimpleNamespace(text=json.dumps(bad)))
            target[name] = "false"
            jsonschema.validate(bad, schema)
            parser(SimpleNamespace(text=json.dumps(bad)))


def test_native_malformed_schema_is_classified_but_parser_bug_is_not(monkeypatch):
    import pmmc.compiler.questioner as legacy

    _, built = build()
    candidate = resolve(built).provider_for_verifier
    parser = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=candidate)
    for text in ("not JSON", '{"result": {}}'):
        with pytest.raises(ModelOutputContractError, match="verifier_output_schema_invalid"):
            parser(SimpleNamespace(text=text))
    def bug(**kwargs):
        raise ValueError("implementation bug")
    monkeypatch.setattr(legacy, "questioner_verifier_v6_output_parser", bug)
    parser = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=candidate)
    with pytest.raises(ValueError) as exc:
        parser(SimpleNamespace(text=json.dumps(verdict(candidate))))
    assert not isinstance(exc.value, ModelOutputContractError)


def test_native_long_span_and_multiple_spans_from_same_memory(monkeypatch):
    import pmmc.compiler.questioner as legacy

    _, built = build(
        "I bought a Yamaha P45 piano " + "for practice " * 30 + ". I practice piano daily."
    )
    candidate = draft(built)
    candidate = candidate.model_copy(
        update={
            "answer_claims": [
                *candidate.answer_claims,
                candidate.answer_claims[0].model_copy(
                    update={"support_alias": "T002", "claim_text": "I practice piano daily."}
                ),
            ]
        }
    )
    monkeypatch.setattr(
        legacy, "resolve_v33_draft", lambda *_a, **_k: pytest.fail("legacy conversion")
    )
    resolved = resolve(built, candidate)
    assert len(resolved.claims[0].support_quote.split()) > 32
    assert len(resolved.claims) == 2
    assert len(resolved.internal.expected_evidence) == 1
    assert resolved.internal.hop_count == 1
    assert len({claim.support.support_id for claim in resolved.claims}) == 2


def test_native_ranges_and_owning_memory_checked():
    _, built = build("I practice piano daily. I practice piano daily.")
    supports = native_support_catalog(built.opportunities, built.memories)
    assert supports["T002"].start_char > supports["T001"].end_char
    with pytest.raises(ValidationError):
        QuestionerNativeSupport(**{**supports["T001"].model_dump(), "end_char": 1})
    broken = replace(built.opportunities, sentence_span_aliases={"T001": ("M01", "not present")})
    with pytest.raises(NativeSupportError, match="span_range_mismatch"):
        native_support_catalog(broken, built.memories)


def test_no_controls_without_refusal_opportunity_and_dynamic_alias_scope():
    _, built = build()
    schema = native_response_schema(built.opportunities)
    assert schema["properties"]["control_candidates"]["maxItems"] == 0
    branch = schema["properties"]["candidates"]["items"]["anyOf"][0]
    assert "paraphrases" not in branch["properties"]
    allowed = branch["properties"]["answer_claims"]["items"]["anyOf"][0]
    assert allowed["properties"]["support_alias"]["enum"] == ["T001", "T002"]
    control = QuestionerControlDraftV36(
        candidate_kind="refusal",
        opportunity_id=draft(built).opportunity_id,
        generated_question="What is unknown?",
        answer_claims=[],
        confidence=0.5,
    )
    with pytest.raises(NativeSupportError, match="control_without_refusal_opportunity"):
        resolve(built, control)


def test_verifier_sees_exact_span_not_neighbor():
    config, built = build(
        "Looking forward to it! Indoor plants are a meaningful upgrade.",
        speaker=SpeakerRole.ASSISTANT,
    )
    resolved = resolve(
        built, draft(built, answer="Indoor plants are a meaningful upgrade.", subject="assistant")
    )
    request = render_questioner_verifier_request(
        request_id="verify-native",
        model_alias=config.compiler_model_alias,
        unit_id="unit",
        cutoff_session_id="session-native",
        memories=list(built.memories),
        sessions=list(built.sessions),
        opportunity_set=built.opportunities,
        candidates=[("Q01", resolved.provider_for_verifier)],
        final_round=False,
        max_output_tokens=1024,
        questioner_prompt_version=config.prompt_version,
        questioner_renderer_version=config.renderer_version,
        verifier_prompt_version=config.verifier_prompt_version,
    )
    payload = json.loads(request.messages[1].parts[0].text)
    assert payload["exact_support"][0]["exact_text"] == "Looking forward to it!"
    assert len(payload["exact_support"]) == 1
    assert "T002" not in request.messages[1].parts[0].text
    assert CandidateRejectionCode.EVIDENCE_NOT_ENTAILED in native_static_codes(resolved.internal)


@pytest.mark.parametrize("dimension", ["span_exists", "claim_entailed", "speaker_subject_aligned"])
def test_each_claim_dimension_is_binding(dimension):
    _, built = build()
    provider = resolve(built).provider_for_verifier
    raw = verdict(provider)
    raw["result"]["claims"][0][dimension] = False
    parsed = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=provider)(
        SimpleNamespace(text=json.dumps(raw))
    )
    assert parsed.value["result"]["dimensions"]["answer_explicitly_entailed"] is False
    assert parsed.value["_parser_metadata"]["support_assessment"]["claims"][0][dimension] is False


def test_verifier_rejects_wrong_claim_alias_and_missing_claim():
    _, built = build()
    provider = resolve(built).provider_for_verifier
    parser = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=provider)
    for claims in (
        [],
        [
            dict(
                claim_index=0,
                support_alias="T002",
                span_exists=True,
                claim_entailed=True,
                speaker_subject_aligned=True,
            )
        ],
    ):
        with pytest.raises(NativeSupportError, match="verifier_claim_scope_mismatch"):
            parser(SimpleNamespace(text=json.dumps(verdict(provider, claims=claims))))


@pytest.mark.parametrize(
    ("text", "question", "answer"),
    [
        (
            "I am fascinated by astronomy.",
            "Why am I interested in astronomy?",
            "I am fascinated by astronomy.",
        ),
        ("I bought the full-sized one.", "Which piano model did I buy?", "the full-sized one"),
        (
            "That's the spirit.",
            "What do I think about the piano suggestions?",
            "That's the spirit.",
        ),
    ],
)
def test_known_closure_failures(text, question, answer):
    _, built = build(text)
    result = resolve(built, draft(built, question=question, answer=answer))
    assert CandidateRejectionCode.QUESTION_ANSWER_CLOSURE_FAILED in native_static_codes(
        result.internal
    )


def test_assistant_state_cannot_be_user_fact_but_advice_is_valid():
    _, built = build("I recommend a Yamaha P45 piano.", speaker=SpeakerRole.ASSISTANT)
    assert CandidateRejectionCode.EVIDENCE_NOT_ENTAILED in native_static_codes(
        resolve(built).internal
    )
    advice = draft(
        built,
        question="Which piano did the assistant recommend?",
        answer="A Yamaha P45 piano.",
        subject="assistant",
        assertion="advice",
    )
    assert native_static_codes(resolve(built, advice).internal) == []


def test_explicit_user_plan_is_allowed():
    config, built = build("I plan to buy a Yamaha P45 piano.")
    resolved = resolve(
        built,
        draft(
            built,
            question="Which piano do I plan to buy?",
            answer="I plan to buy a Yamaha P45 piano.",
            assertion="plan",
        ),
    )
    gated = QuestionerCandidateGate(config).evaluate(
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        unit=built.unit,
        generation_id="generation",
        model_call_id="call",
        drafts=[resolved.internal],
        memories=list(built.memories),
        opportunity_set=built.opportunities,
    )
    assert gated[0].gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING


def test_count_and_temporal_require_specific_closure():
    _, built = build()
    base = resolve(built).provider_for_verifier
    for field, changes in (
        ("explicit_count_set", {"answer_type": "count"}),
        ("same_entity_attribute", {"reasoning_type": "update"}),
    ):
        from pmmc.schemas.questioner import QuestionerVerifierCandidateV36

        provider = QuestionerVerifierCandidateV36.model_validate({**base.model_dump(), **changes})
        parser = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=provider)
        with pytest.raises(NativeSupportError, match="verifier_required_dimension_missing"):
            parser(SimpleNamespace(text=json.dumps(verdict(provider))))
        raw = verdict(provider, **{field: "false"})
        parsed = parser(SimpleNamespace(text=json.dumps(raw)))
        assert parsed.value["result"]["dimensions"]["answer_explicitly_entailed"] is False


def test_type_allocation_and_information_need_are_query_blind():
    _, built = build()
    template = built.opportunities.opportunities[0]
    kinds = [QuestionOpportunityType.LOCAL_FACT] * 10 + [
        QuestionOpportunityType.UPDATE_PAIR,
        QuestionOpportunityType.TEMPORAL_PAIR,
        QuestionOpportunityType.MULTI_EVIDENCE,
        QuestionOpportunityType.VISUAL_SEARCH,
    ]
    opportunities = tuple(
        template.model_copy(
            update={
                "opportunity_id": f"o-{index}",
                "ordinal": index,
                "opportunity_type": kind,
            }
        )
        for index, kind in enumerate(kinds)
    )
    selected = allocate_opportunities(replace(built.opportunities, opportunities=opportunities))
    assert len(selected) == 8
    assert {item.opportunity_type for item in selected} >= set(kinds[-4:])
    assert information_need_key("Will learning bass make me calmer?", "boolean") == (
        information_need_key("Might learning bass make me calmer?", "boolean")
    )
    assert information_need_key("Will learning bass make me calmer?", "boolean") != (
        information_need_key("Will learning bass not make me calmer?", "boolean")
    )


@pytest.mark.parametrize("entailed", [True, False])
def test_actual_graph_dispatch_persists_native_support_verifier_and_funnel(entailed):
    config, built = build()
    candidate = draft(built)
    generated = {
        "candidates": [
            candidate.model_dump(mode="json"),
            {"invalid": True},
            draft(built, span="T999").model_dump(mode="json"),
        ],
        "control_candidates": [],
    }
    requests, seeds, results = [], [], []

    class Execution:
        async def generate_audited(self, gateway, request, context, *, output_parser):
            requests.append(request)
            if request.metadata["task"] == "pmmc_questioner":
                raw = generated
            else:
                raw = verdict(resolve(built).provider_for_verifier)
                raw["result"]["claims"][0]["claim_entailed"] = entailed
            parsed = output_parser(SimpleNamespace(text=json.dumps(raw)))
            trace = ModelCallTrace(
                model_call_id=context.model_call_id,
                run_id=context.run_id,
                snapshot_id=context.snapshot_id,
                conversation_id=context.conversation_id,
                job_id=context.job_id,
                compiler_unit_id=context.compiler_unit_id,
                stage=AppStage.QUESTIONER,
                model_alias=config.compiler_model_alias,
                status=TraceStatus.SUCCESS,
                parsed_output=parsed.value,
                prompt_hash="native-prompt-hash",
                output_asset_id="native-output",
            )
            return SimpleNamespace(trace=trace)

    async def nothing(*args):
        return None

    async def primaries(*args):
        return {}

    async def seed(*args):
        seeds.append(args)

    async def persist(result):
        results.append(result)

    graph = QuestionerCompilationGraph(
        execution=Execution(),
        gateway=object(),
        config=config,
        load_successful_call=nothing,
        load_model_call=nothing,
        before_gate=primaries,
        persist_v3_seed=seed,
        persist_result=persist,
    )
    job = ExperimentJob(
        job_id="job-native",
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        compiler_unit_id=built.unit.compiler_unit_id,
        stage=AppStage.COMPILE,
        attempt_count=1,
        max_attempts=2,
    )
    result = asyncio.run(graph.process(built, job))
    assert results == [result] and len(seeds) == 1
    assert [request.metadata["prompt_version"] for request in requests] == [
        "pmmc-questioner-v3.6",
        "pmmc-questioner-verifier-v7",
    ]
    assert result.candidates[0].gate_result.status is (
        CandidateGateStatus.ELIGIBLE_FOR_PLANNING if entailed else CandidateGateStatus.REJECTED
    )
    if entailed:
        assert result.candidates[0].gate_result.metadata["family_budget_rank"] == 1
    claim = result.answer_claims[0]
    assert claim.metadata["span_exists"] is True
    assert claim.metadata["claim_entailed"] == "not_evaluated"
    assert claim.metadata["native_support"]["speaker"] == "user"
    assert questioner_answer_claim_from_row(questioner_answer_claim_to_row(claim)) == claim
    verification = result.verifications[0].metadata["support_assessment"]
    assert verification["claims"][0]["claim_entailed"] is entailed
    counts = result.stage_result.metadata["questioner_funnel_v1"]["counts"]
    assert counts == {
        "provider_items": 3,
        "schema_native": 2,
        "alias_resolved": 1,
        "static_pass": 1,
        "verifier_pass": int(entailed),
        "dedup_eligible": int(entailed),
    }
    alias_failure = result.stage_result.metadata["questioner_funnel_v1"]["items"][2]
    assert alias_failure["opportunity_type"] == "local_fact"
    assert alias_failure["first_failure_reason"] == "support_alias_unknown"


def test_native_json_is_not_silently_repaired():
    with pytest.raises(ValueError, match="native_json_required"):
        parse_questioner_output(
            '{"candidates": [], "control_candidates": []',
            parser_version="questioner-json-parser-v3.6",
        )


def test_audited_native_graph_retains_invalid_verifier_trace_and_rejects(tmp_path):
    from pmmc.models.campaign_audit import campaign_model_scope
    from pmmc.models.fake import DeterministicFakeGateway
    from test_step8_model_journal import fixture

    service, audit, gateway, _, _, writer, _ = fixture(tmp_path)
    config, built = build()
    entry = service.registry.get("deterministic-fake").model_copy(
        update={"alias": config.compiler_model_alias},
    )
    service.registry.models.append(entry)
    traces = {}
    requests = []
    results = []

    async def generate(request):
        requests.append(request)
        raw = {"candidates": [draft(built).model_dump(mode="json")], "control_candidates": []}
        if request.metadata["task"] == "pmmc_questioner_verifier":
            raw = verdict(resolve(built).provider_for_verifier)
            raw["result"]["dimensions"]["user_facing"] = "not_applicable"
        response = await DeterministicFakeGateway().generate(request)
        return response.model_copy(update={"text": json.dumps(raw), "raw_response": raw})

    async def add(trace: ModelCallTrace) -> ModelCallTrace:
        assert trace.model_call_id not in traces or traces[trace.model_call_id] == trace
        traces[trace.model_call_id] = trace
        return trace

    async def load(call_id):
        return traces.get(call_id)

    async def nothing(*args):
        return None

    async def primaries(*args):
        return {}

    async def persist(result):
        results.append(result)

    gateway.generate.side_effect = generate
    writer.add.side_effect = add
    graph = QuestionerCompilationGraph(
        execution=service, gateway=gateway, config=config,
        load_successful_call=nothing, load_model_call=load, before_gate=primaries,
        persist_v3_seed=nothing, persist_result=persist,
    )
    job = ExperimentJob(
        job_id="job-native", run_id=built.unit.run_id, snapshot_id=built.unit.snapshot_id,
        compiler_unit_id=built.unit.compiler_unit_id, stage=AppStage.COMPILE,
        attempt_count=1, max_attempts=2,
    )
    with campaign_model_scope(audit):
        result = asyncio.run(graph.process(built, job))
    assert results == [result]
    assert result.candidates[0].gate_result.status is CandidateGateStatus.REJECTED
    failed = [trace for trace in traces.values() if trace.status is TraceStatus.FAILED]
    assert len(failed) == 1
    assert failed[0].output_asset_id and failed[0].parse_error
    assert failed[0].metadata["output_contract_error_code"] == "verifier_required_dimension_missing"
    assert result.verifications[0].metadata["provider_output_fallback"] is True
    assert len(requests) == 2  # No whole-unit retry after an audited invalid verifier response.


def test_same_owner_text_and_caption_are_not_duplicate_evidence_or_extra_hops():
    _, built = build(memory_updates={"caption": "A Yamaha piano in the studio."})
    catalog = native_support_catalog(built.opportunities, built.memories)
    caption_alias = next(
        alias for alias, support in catalog.items() if support.source_field == "caption"
    )
    candidate = draft(built)
    candidate = candidate.model_copy(
        update={
            "answer_claims": [
                *candidate.answer_claims,
                candidate.answer_claims[0].model_copy(
                    update={"support_alias": caption_alias, "claim_text": "A Yamaha piano."}
                ),
            ]
        }
    )
    resolved = resolve(built, candidate)
    assert len(resolved.internal.expected_evidence) == 1
    assert resolved.internal.hop_count == 1
    opportunity = next(
        item
        for item in built.opportunities.opportunities
        if local_opportunity_id(item) == candidate.opportunity_id
    )
    multiple = opportunity.model_copy(update={"minimum_evidence_groups": 2})
    bad_set = replace(built.opportunities, opportunities=(multiple,))
    with pytest.raises(NativeSupportError, match="independent_support_groups_missing"):
        resolve_v36_draft(candidate, opportunity_set=bad_set, memories=built.memories)


def test_image_ownership_is_checked_separately():
    _, built = build(memory_updates={"image_ids": ["photo-one"]})
    supports = native_support_catalog(built.opportunities, built.memories)
    assert supports["I01"].image_id == "photo-one"
    evidence = dict(built.opportunities.alias_to_evidence)
    evidence["I01"] = evidence["I01"].model_copy(
        update={"metadata": {"owning_memory_id": "foreign"}}
    )
    with pytest.raises(NativeSupportError, match="image_owner_mismatch"):
        native_support_catalog(
            replace(built.opportunities, alias_to_evidence=evidence), built.memories
        )


def test_image_set_opportunity_requires_complete_scan_and_list_closure():
    _, built = build(memory_updates={"image_ids": ["photo-one", "photo-two"]})
    opportunity = next(
        item for item in built.opportunities.opportunities if item.metadata.get("image_set_scope")
    )
    assert opportunity.evidence_aliases == ["I01", "I02"]
    refs = [
        ref
        for ref in built.opportunities.evidence_refs
        if ref.opportunity_id == opportunity.opportunity_id
    ]
    assert [ref.alias for ref in refs] == opportunity.evidence_aliases
    candidate = QuestionerAnswerableDraftV36(
        candidate_kind="answerable",
        opportunity_id=local_opportunity_id(opportunity),
        generated_question="Which photographs show a piano?",
        answer_type="list",
        confidence=0.9,
        answer_claims=[
            dict(
                claim_kind="visual",
                support_alias=alias,
                claim_text=f"The {color} piano photograph.",
                subject="other",
                assertion="fact",
            )
            for alias, color in zip(opportunity.evidence_aliases, ("red", "blue"), strict=True)
        ],
    )
    provider = resolve(built, candidate).provider_for_verifier
    assert provider.image_set_scope
    with pytest.raises(NativeSupportError, match="image_set_scan_incomplete"):
        resolve(built, candidate.model_copy(update={"answer_claims": candidate.answer_claims[:1]}))
    parser = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=provider)
    with pytest.raises(NativeSupportError, match="verifier_required_dimension_missing"):
        parser(SimpleNamespace(text=json.dumps(verdict(provider))))
    parsed = parser(SimpleNamespace(text=json.dumps(verdict(provider, explicit_count_set="true"))))
    assert parsed.value["result"]["dimensions"]["answer_explicitly_entailed"]


def test_step8_workflow_pins_manifest_versions_and_run_scoped_units():
    from pmmc.compiler.workflow import (
        questioner_uses_run_scoped_units,
        questioner_workflow_manifest_versions,
    )
    from pmmc.schemas import RunManifest

    config, built = build()
    other_config, other = build(run_id="run-native-other")
    versions = questioner_workflow_manifest_versions(config)
    assert versions == {"schema_version": "0.15.0", "compiler_schema_version": "0.9.0"}
    manifest = RunManifest(
        **versions,
        run_id="step8",
        experiment_name="step8-test",
        dataset="memlens",
        dataset_version="test",
        method="pmmc_compiler",
        operator_set_version="pmmc-operators-v7",
        operator_registry_hash="registry",
        compiler_config_hash="compiler",
    )
    assert manifest.schema_version == "0.15.0"
    assert questioner_uses_run_scoped_units(config)
    assert questioner_uses_run_scoped_units(other_config)
    assert built.unit.compiler_unit_id != other.unit.compiler_unit_id
    assert built.unit.input_hash == other.unit.input_hash
    old = load_questioner_config(ROOT / "configs/compiler/questioner-v3.5.yaml")
    assert questioner_workflow_manifest_versions(old) == {"compiler_schema_version": "0.8.0"}


def test_step8_method_protocol_identity_is_not_a_legacy_freeze():
    from pmmc.compiler.planner_config import load_planner_config
    from pmmc.compiler.workflow import ProgramWorkflowIdentity, _validate_step8_program_identity

    config, _ = build()
    planner = load_planner_config(ROOT / "configs/compiler/planner-v1.5.yaml")
    identity = ProgramWorkflowIdentity(
        questioner_freeze_or_audit_hash="compiler-subprotocol-content-hash",
        questioner_identity_kind="step8_method_protocol",
        planner=planner,
        memory_text_model_alias="text",
        memory_multimodal_model_alias="multimodal",
    )
    _validate_step8_program_identity(config, identity)
    old = load_questioner_config(ROOT / "configs/compiler/questioner-v3.5.yaml")
    with pytest.raises(ValueError, match="requires Questioner v3.6"):
        _validate_step8_program_identity(old, identity)
    legacy_planner = load_planner_config(ROOT / "configs/compiler/planner-v1.4.yaml")
    with pytest.raises(ValueError, match="Planner v1.5"):
        _validate_step8_program_identity(config, replace(identity, planner=legacy_planner))


def test_no_raw_policy_removes_opportunities_and_images_but_preserves_captions():
    from pmmc.compiler.questioner_config import questioner_config_hash
    from pmmc.schemas import ModelContentKind

    updates = {"image_ids": ["photo-one"], "caption": "A Yamaha piano in the studio."}
    allow_config, allowed = build(memory_updates=updates)
    config, built = build(memory_updates=updates, policy="forbid")
    assert questioner_config_hash(allow_config) != questioner_config_hash(config)
    assert allowed.unit.compiler_unit_id != built.unit.compiler_unit_id
    assert built.memories[0].image_ids == ["photo-one"]
    assert built.unit.metadata["input_image_count"] == 0
    assert built.opportunities.image_aliases == {}
    assert all(item.required_modalities == ["text"] for item in built.opportunities.opportunities)
    assert all(
        part.kind is not ModelContentKind.IMAGE
        for message in built.request.messages
        for part in message.parts
    )
    assert "A Yamaha piano in the studio." in built.request.messages[1].parts[0].text
    catalog = native_support_catalog(built.opportunities, built.memories)
    caption = next(alias for alias, support in catalog.items() if support.source_field == "caption")
    assert resolve(built, draft(built, span=caption)).claims[0].support.source_field == "caption"
    candidate = draft(allowed)
    visual = {
        **candidate.answer_claims[0].model_dump(),
        "claim_kind": "visual",
        "support_alias": "I01",
    }
    visual_candidate = QuestionerAnswerableDraftV36.model_validate(
        {**candidate.model_dump(), "answer_claims": [visual]}
    )
    with pytest.raises(NativeSupportError, match="raw_image_forbidden"):
        resolve_v36_draft(
            visual_candidate,
            opportunity_set=allowed.opportunities,
            memories=allowed.memories,
            memory_raw_image_policy="forbid",
        )
    old = load_questioner_config(ROOT / "configs/compiler/questioner-v3.5.yaml")
    assert "memory_raw_image_policy" not in old.model_dump()
    with pytest.raises(ValidationError, match="requires Questioner v3.6"):
        type(old).model_validate({**old.model_dump(), "memory_raw_image_policy": "forbid"})


def test_native_family_merges_modal_variants_in_actual_gate():
    config, built = build("Learning bass makes me calmer before performances.")
    candidates = [
        resolve(
            built,
            draft(
                built,
                question=f"{modal} learning bass make me calmer before performances?",
                answer="Learning bass makes me calmer before performances.",
                answer_type="boolean",
            ),
        ).internal
        for modal in ("Will", "Might")
    ]
    gated = QuestionerCandidateGate(config).evaluate(
        run_id=built.unit.run_id,
        snapshot_id=built.unit.snapshot_id,
        unit=built.unit,
        generation_id="generation-family",
        model_call_id="family-call",
        drafts=candidates,
        memories=list(built.memories),
        opportunity_set=built.opportunities,
    )
    assert gated[0].gate_result.status is CandidateGateStatus.ELIGIBLE_FOR_PLANNING
    assert gated[1].gate_result.rejection_codes == [CandidateRejectionCode.DUPLICATE]
    assert gated[1].duplicate_link.primary_qbank_id == gated[0].candidate.qbank_id


def test_historical_parser_still_requires_historical_shape():
    _, built = build()
    raw = json.dumps(
        {"candidates": [draft(built).model_dump(mode="json")], "control_candidates": []}
    )
    old = parse_questioner_output(raw, parser_version="questioner-json-parser-v3.5")
    assert old.invalid_candidate_count == 1
    config = load_questioner_config(ROOT / "configs/compiler/questioner-v3.5.yaml")
    assert config.max_paraphrases == 3 and config.compiler_schema_version == "0.8.0"
    new_config, _ = build()
    assert (
        questioner_output_parser(new_config)(SimpleNamespace(text=raw)).value["raw_candidate_count"]
        == 1
    )
