from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from jsonschema import validate
from pydantic import ValidationError

from pmmc.baselines.answering import BaselineAnswerService
from pmmc.baselines.context import ContextBudgeter
from pmmc.baselines.rag_config import RagBaselineConfig
from pmmc.baselines.rag_context import RagContextBudgeter, RankedWindow
from pmmc.models.tokenization import HeuristicTokenCounter
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.answering import (
    InvalidRuntimeAnswer,
    SharedAnswerGenerationService,
    answer_format_retry_request,
    make_typed_answer_context,
    map_typed_image_items,
    parse_typed_answer,
    prepare_typed_answer_request,
    validate_typed_answer_input,
)
from pmmc.runtime.grounding import (
    AuditedGroundingService,
    GroundingEngineeringError,
    build_grounding_request,
    configured_grounding_service,
)
from pmmc.schemas import (
    AnswerGenerationInput,
    BaselineMethod,
    ExperimentJob,
    GenerationRequest,
    GenerationResult,
    ImageRef,
    MemoryUnit,
    MemoryWindow,
    ModelCallTrace,
    ModelContentPart,
    ModelMessage,
    PreparedAnswerInput,
    ProgramExecutionEvidenceRef,
    QueryAnswerVariant,
    QueryInput,
    RetrievalTrace,
    SessionRecord,
    TokenUsage,
)
from pmmc.schemas.answer_output import (
    TYPED_ANSWER_CONTRACT_VERSION,
    AnswerContractFailure,
    AnswerContractV3,
    AnswerEvidenceSource,
    AnswerKind,
    AnswerSourceImage,
    StructuredAnswerOutput,
    TypedAnswerResult,
    answer_response_schema,
    typed_answer_response_schema,
)


def query():
    return QueryInput(
        query_id="query",
        dataset="memlens",
        conversation_id="conversation",
        raw_query="Which photos show the trip?",
        available_after_session="session",
        metadata={
            "gold": "GOLD_SENTINEL",
            "expected_answer": "EXPECTED_SENTINEL",
            "category": "CATEGORY_SENTINEL",
        },
    )


def sources():
    return [
        AnswerEvidenceSource(
            evidence_alias=f"E{n:02d}",
            evidence_id=f"private-source-{n}",
            text=f"SOURCE_{n}: The trip photo was taken in May.",
            images=[
                AnswerSourceImage(
                    image_id=f"private-image-{n}",
                    image=ImageRef(
                        image_id=f"private-image-{n}",
                        metadata={"gold": "IMAGE_GOLD_SENTINEL"},
                    ),
                )
            ],
        )
        for n in (1, 2, 3)
    ]


def base_request():
    return GenerationRequest(
        request_id="request",
        model_alias="answer-qwen3.5-9b",
        max_output_tokens=1024,
        messages=[ModelMessage(role="user", parts=[ModelContentPart(kind="text", text="unused")])],
        metadata={
            "answer_contract_version": TYPED_ANSWER_CONTRACT_VERSION,
            "gold": "REQUEST_GOLD_SENTINEL",
        },
    )


def prepared(method="pmmc"):
    request = prepare_typed_answer_request(
        base_request(), query=query(), evidence_sources=sources()
    )
    return AnswerGenerationInput(
        run_id="run",
        snapshot_id="snapshot",
        query=query(),
        method_id=method,
        answer_variant="primary",
        request=request,
        evidence_ids_in_prompt=[s.evidence_id for s in sources()],
        image_ids_in_prompt=[
            p.image.image_id for m in request.messages for p in m.parts if p.image
        ],
        source_lineage={"execution_id": "execution"},
        metadata=dict(request.metadata),
    )


def output(kind="text", text="May", selected=(), aliases=("E01",)):
    abstain = kind == "abstention"
    return {
        "answer_kind": kind,
        "answer_text": None if selected or abstain else text,
        "selected_image_aliases": list(selected),
        "supported": not abstain,
        "used_evidence_aliases": [] if abstain else list(aliases),
        "abstain_reason": "insufficient_evidence" if abstain else None,
    }


def assessment(supported=True):
    return {
        "supported": supported,
        "query_closure": "pass",
        "entity_consistency": "pass",
        "time_consistency": "not_applicable",
        "set_consistency": "not_applicable",
        "source_support": "pass" if supported else "fail",
    }


class FakeGroundingCounter:
    alias = "grounding-qwen3.5-9b"
    revision = "synthetic-grounding-test-counter"
    estimated = False

    def count_request(self, request):
        return 100

    def count_text(self, text):
        return 100

    def count_image(self, *, width=None, height=None):
        return 100


class FakeExecution:
    def __init__(self, responses, *, finish="stop", tokens=50):
        self.responses = iter(responses)
        self.finish = finish
        self.tokens = tokens
        self.requests = []
        self.traces = []

    async def generate_audited(self, gateway, request, context, *, typed_schema_handoff=False):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        text = response if isinstance(response, str) else json.dumps(response)
        result = GenerationResult(
            request_id=request.request_id,
            model_alias=request.model_alias,
            actual_model_id="9b-commit",
            text=text,
            finish_reason=self.finish,
            usage=TokenUsage(
                input_tokens=100, output_tokens=self.tokens, total_tokens=100 + self.tokens
            ),
        )
        trace = ModelCallTrace(
            model_call_id=context.model_call_id,
            run_id=context.run_id,
            query_id=context.query_id,
            snapshot_id=context.snapshot_id,
            stage=context.stage,
            model_alias=request.model_alias,
            actual_model_id=result.actual_model_id,
            input_tokens=100,
            output_tokens=self.tokens,
            cost_usd=0.01,
            latency_ms=10,
            parent_model_call_id=context.parent_model_call_id,
            parsed_output={"text": text, "finish_reason": self.finish},
            request_config={"request_metadata": request.metadata},
        )
        self.traces.append(trace)
        return SimpleNamespace(result=result, trace=trace)


def service(responses, **kwargs):
    execution = FakeExecution(responses, **kwargs)
    grounding = AuditedGroundingService(
        execution=execution, gateway=object(), counter=FakeGroundingCounter()
    )
    return SharedAnswerGenerationService(
        execution=execution, gateway=object(), grounding=grounding
    ), execution


@pytest.mark.parametrize(
    "kind,text,images",
    [
        ("text", "May", []),
        ("number", "-12.50", []),
        ("boolean", "false", []),
        ("image", None, ["I02"]),
        ("image_set", None, ["I03", "I01"]),
        ("ordered_image_list", None, ["I03", "I01"]),
        ("abstention", None, []),
    ],
)
def test_all_typed_kinds_native_schema_and_deterministic_mapping(kind, text, images):
    context = make_typed_answer_context(sources())
    value = output(kind, text, images)
    validate(value, typed_answer_response_schema(context))
    parsed = parse_typed_answer(json.dumps(value), context=context, finish_reason="stop")
    items = map_typed_image_items(parsed, context)
    expected = sorted(images) if kind == "image_set" else images
    assert [i.image_alias for i in items] == expected
    assert [i.image_id for i in items] == [f"private-image-{int(i[1:])}" for i in expected]
    assert all(
        i.public_media_ref.startswith("media:") and "private" not in i.public_media_ref
        for i in items
    )


@pytest.mark.parametrize(
    "update",
    [
        {"answer_kind": "number", "answer_text": "1,200"},
        {"answer_kind": "number", "answer_text": "NaN"},
        {"answer_kind": "boolean", "answer_text": "yes"},
        {"supported": "true"},
        {"selected_image_aliases": ["I01"]},
        {"answer_kind": "image", "answer_text": None, "selected_image_aliases": []},
        {"answer_kind": "image", "answer_text": None, "selected_image_aliases": ["I01", "I02"]},
        {"answer_kind": "image_set", "answer_text": None, "selected_image_aliases": ["I01", "I01"]},
        {
            "answer_kind": "image",
            "answer_text": None,
            "selected_image_aliases": ["private-image-1"],
        },
        {"answer_kind": "image", "answer_text": None, "selected_image_aliases": ["E01"]},
        {"used_evidence_aliases": ["I01"]},
        {"used_evidence_aliases": ["E01", "E01"]},
        {"answer_text": None},
        {"abstain_reason": "unsupported_by_evidence"},
    ],
)
def test_native_schema_server_rejects_cross_field_and_type_errors(update):
    with pytest.raises(ValidationError):
        AnswerContractV3.model_validate({**output(), **update})


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("answer_text", "private-source-1", "internal_id_leak"),
        ("answer_text", "private-image-1", "internal_id_leak"),
        ("answer_text", "E01", "evidence_alias_leak"),
        ("answer_text", "I01", "evidence_alias_leak"),
        ("answer_text", "<think>May</think>", "thinking_tag_leak"),
        ("answer_text", "https://private.example/image", "url_leak"),
        ("used_evidence_aliases", ["E99"], "unknown_evidence_alias"),
        ("used_evidence_aliases", [" E01 "], "schema_invalid"),
    ],
)
def test_runtime_subset_and_public_leak_gates(field, value, code):
    with pytest.raises(InvalidRuntimeAnswer) as caught:
        parse_typed_answer(
            json.dumps({**output(), field: value}),
            context=make_typed_answer_context(sources()),
            finish_reason="stop",
        )
    assert AnswerContractFailure(code) in caught.value.failures
    assert "private" not in str(caught.value)


def test_unknown_image_alias_is_not_salvaged_or_converted_to_text():
    with pytest.raises(InvalidRuntimeAnswer) as caught:
        parse_typed_answer(
            json.dumps(output("image", None, ["I99"])),
            context=make_typed_answer_context(sources()),
            finish_reason="stop",
        )
    assert caught.value.failures == (AnswerContractFailure.UNKNOWN_IMAGE_ALIAS,)


def test_selected_images_and_evidence_aliases_are_not_same_semantic_field():
    parsed = parse_typed_answer(
        json.dumps(output("image", None, ["I02"], ["E01"])),
        context=make_typed_answer_context(sources()),
        finish_reason="stop",
    )
    assert parsed.selected_image_aliases == ["I02"]
    assert parsed.used_evidence_aliases == ["E01"]


def test_explicit_request_adapter_enforces_budget_and_source_boundary():
    value = prepared()
    validate_typed_answer_input(value)
    serialized = value.request.model_dump_json()
    assert "SENTINEL" not in serialized
    assert value.request.max_output_tokens == 512
    assert value.request.thinking_mode.value == "disabled"
    assert value.request.response_format["json_schema"]["name"] == "typed_answer_contract_v3"
    retry = answer_format_retry_request(value.request)
    assert len(retry.messages) == len(value.request.messages) + 1
    assert "six required" in retry.messages[-1].parts[0].text
    wrong = base_request().model_copy(
        update={"metadata": {"answer_contract_version": "strict-answer-contract-v2"}}
    )
    with pytest.raises(ValueError, match="explicit"):
        prepare_typed_answer_request(wrong, query=query(), evidence_sources=sources())


@pytest.mark.parametrize("supported", [True, False])
def test_gated_result_retains_ungated_answer_and_separate_costs(supported):
    runner, execution = service(
        [output("image_set", None, ["I01", "I02"], ["E01", "E02"]), assessment(supported)]
    )
    result = asyncio.run(runner.answer(prepared(), model_call_id="call"))
    assert isinstance(result, TypedAnswerResult)
    assert result.terminal_state == ("answered" if supported else "abstained")
    assert result.output.supported is supported
    assert result.ungated_output.supported
    assert result.ungated_image_items
    assert bool(result.image_items) is supported
    assert [c.role for c in result.calls] == ["answer", "answer_grounding"]
    assert sum(c.cost_usd for c in result.calls) == 0.02
    assert result.grounding.assessment_source == "model_assessed"
    assert execution.traces[1].parent_model_call_id == "call"
    assert "private" not in json.dumps(result.public_payload())


def test_failed_grounding_can_request_pmmc_fallback_but_not_baseline_fallback():
    runner, _ = service([output(), assessment(False)])
    result = asyncio.run(
        runner.answer(prepared(), model_call_id="call", on_grounding_failure="fallback")
    )
    assert result.terminal_state == "fallback_required"
    runner, execution = service([])
    with pytest.raises(ValueError, match="only PMMC"):
        asyncio.run(
            runner.answer(
                prepared("native_multimodal_rag"),
                model_call_id="call",
                on_grounding_failure="fallback",
            )
        )
    assert not execution.requests


def test_legal_abstention_is_one_call_no_grounding_no_retry_no_exception():
    runner, execution = service([output("abstention")])
    runner.grounding = None
    result = asyncio.run(runner.answer(prepared(), model_call_id="call"))
    assert result.terminal_state == "abstained"
    assert result.grounding is None
    assert len(execution.requests) == 1


def test_one_billed_format_retry_uses_same_model_then_grounds():
    runner, execution = service(["NOT JSON", output(), assessment()])
    result = asyncio.run(runner.answer(prepared(), model_call_id="call"))
    assert [c.role for c in result.calls] == ["answer", "answer_format_retry", "answer_grounding"]
    assert execution.requests[0].model_alias == execution.requests[1].model_alias
    assert execution.requests[1].max_output_tokens == 512
    assert len(result.format_failure_codes) == 1


@pytest.mark.parametrize("finish,tokens", [("length", 50), ("stop", 513), (None, 50)])
def test_truncation_output_limit_and_unknown_finish_are_engineering_errors(finish, tokens):
    runner, execution = service(
        [output("abstention"), output("abstention")], finish=finish, tokens=tokens
    )
    with pytest.raises(InvalidRuntimeAnswer):
        asyncio.run(runner.answer(prepared(), model_call_id="call"))
    assert len(execution.requests) == 2


def test_invalid_verifier_output_and_transport_never_become_scientific_abstention():
    runner, execution = service([output(), "BAD", "BAD"])
    with pytest.raises(GroundingEngineeringError) as caught:
        asyncio.run(runner.answer(prepared(), model_call_id="call"))
    assert caught.value.code == "grounding_format_invalid"
    assert len(execution.requests) == 3
    runner, execution = service([output(), TimeoutError("provider unavailable")])
    with pytest.raises(TimeoutError):
        asyncio.run(runner.answer(prepared(), model_call_id="call"))
    assert len(execution.requests) == 2


def test_grounding_request_contains_only_actual_query_answer_and_cited_sources():
    value = prepared()
    request = build_grounding_request(
        value, AnswerContractV3.model_validate(output()), answer_model_call_id="call"
    )
    serialized = request.model_dump_json()
    assert "SOURCE_1" in serialized and "SOURCE_2" not in serialized
    assert "SOURCE_3" not in serialized and "SENTINEL" not in serialized
    text = "\n".join(p.text for m in request.messages for p in m.parts if p.text)
    assert "private-source" not in text and "private-image" not in text
    assert request.model_alias == "grounding-qwen3.5-9b"
    changed = value.model_copy(
        update={
            "metadata": {
                **value.metadata,
                "typed_answer_context": make_typed_answer_context(sources()[:1]).model_dump(
                    mode="json"
                ),
            }
        }
    )
    with pytest.raises(ValueError):
        build_grounding_request(
            changed, AnswerContractV3.model_validate(output()), answer_model_call_id="call"
        )


def test_later_grounding_phase_recovers_journaled_answer_without_generating_again():
    runner, execution = service([output()])
    runner.grounding = None
    with pytest.raises(GroundingEngineeringError, match="requires_grounding"):
        asyncio.run(runner.answer(prepared(), model_call_id="call"))
    trace = execution.traces[0]
    second, execution2 = service([assessment()])
    result = asyncio.run(
        second.answer_typed(prepared(), model_call_id="call", recovered_trace=trace)
    )
    assert result.output.supported and result.answer_model_call_id == "call"
    assert len(execution2.requests) == 1
    assert execution2.requests[0].model_alias == "grounding-qwen3.5-9b"


def test_factory_uses_existing_alias_gateway_and_missing_alias_returns_none(monkeypatch):
    factory = Mock(return_value=object())
    monkeypatch.setattr("pmmc.models.factory.create_generation_gateway", factory)
    execution = SimpleNamespace(registry=SimpleNamespace(get=Mock(return_value=object())))
    result = configured_grounding_service(
        settings="settings", execution=execution, image_url_resolver="resolver"
    )
    assert isinstance(result, AuditedGroundingService)
    factory.assert_called_once_with(
        "settings", execution.registry, "grounding-qwen3.5-9b", image_url_resolver="resolver"
    )
    execution.registry.get.side_effect = KeyError("absent")
    assert configured_grounding_service(settings="settings", execution=execution) is None


def memory():
    return MemoryUnit(
        memory_id="memory",
        dataset="memlens",
        conversation_id="conversation",
        session_id="session",
        turn_id="turn",
        timestamp_order=0,
        text="The trip was in May.",
        caption="The trip photo.",
        image_refs=[ImageRef(image_id="photo")],
        available_after_session="session",
    )


def sessions():
    return [
        SessionRecord(
            session_id="session",
            dataset="memlens",
            conversation_id="conversation",
            session_order=0,
            source_session_id="original",
            source_date_iso="2026-05-01",
        )
    ]


@pytest.mark.parametrize(
    "method,no_raw",
    [
        *((method, False) for method in BaselineMethod),
        (BaselineMethod.NATIVE_MULTIMODAL_RAG, True),
    ],
)
@pytest.mark.parametrize("split_phases", [False, True])
def test_all_five_baselines_use_native_adapter_and_uniform_grounding(method, no_raw, split_phases):
    counter = HeuristicTokenCounter()
    actual_query = query().model_copy(
        update={"query_image_refs": [ImageRef(image_id="query-photo")]}
    )
    if method in {BaselineMethod.FULL_CONTEXT_CAPTIONS, BaselineMethod.FULL_CONTEXT_RAW_IMAGES}:
        assembled = ContextBudgeter(counter).assemble(
            run_id="run",
            snapshot_id="snapshot",
            instruction="legacy",
            query=actual_query,
            memories=[memory()],
            sessions=sessions(),
            method=method,
            answer_contract_version=TYPED_ANSWER_CONTRACT_VERSION,
        )
        prepared_input = PreparedAnswerInput(
            method=method,
            query=actual_query,
            request=assembled.request,
            context_assembly=assembled.trace,
            evidence_ids_in_prompt=["memory"],
            image_ids_in_prompt=[
                p.image.image_id for m in assembled.request.messages for p in m.parts if p.image
            ],
            metadata=assembled.request.metadata,
        )
    else:
        trace = RetrievalTrace(
            retrieval_trace_id="retrieval",
            retrieval_batch_id="batch",
            query_id="query",
            run_id="run",
            method=method.value,
            retriever_name="test",
            rank=1,
            retrieved_id="window",
            retrieved_type="memory",
        )
        window = MemoryWindow(
            window_id="window",
            corpus_id="corpus",
            snapshot_id="snapshot",
            dataset="memlens",
            conversation_id="conversation",
            session_id="session",
            window_ordinal=0,
            memory_ids=["memory"],
            start_timestamp_order=0,
            end_timestamp_order=0,
            text_caption_token_count=5,
            available_after_session="session",
        )
        prepared_input = RagContextBudgeter(counter).assemble(
            run_id="run",
            snapshot_id="snapshot",
            query=actual_query,
            method=method,
            ranked_windows=[RankedWindow(trace=trace, window=window, memories=(memory(),))],
            sessions=sessions(),
            instruction="legacy",
            config=RagBaselineConfig(
                answer_contract_version=TYPED_ANSWER_CONTRACT_VERSION,
                memory_raw_image_policy="forbid" if no_raw else "allow",
            ),
            answer_contract_version=TYPED_ANSWER_CONTRACT_VERSION,
        )
    execution = FakeExecution([output("image", None, ["I01"]), assessment()])
    baseline = BaselineAnswerService(
        execution,
        object(),
        grounding=AuditedGroundingService(
            execution=execution, gateway=object(), counter=FakeGroundingCounter()
        ),
    )
    job = ExperimentJob(
        job_id="job",
        run_id="run",
        snapshot_id="snapshot",
        query_id="query",
        stage="answer",
        variant="primary",
    )
    if split_phases:
        from pmmc.schemas.answer_output import TypedAnswerCandidate

        baseline.grounding = None
        candidate = asyncio.run(baseline.answer(prepared_input, job, phase="generate"))
        assert isinstance(candidate, TypedAnswerCandidate)
        assert len(execution.requests) == 1
        assert candidate.state == "generated_awaiting_finalization"
        baseline.gateway = None
        baseline.grounding = AuditedGroundingService(
            execution=execution, gateway=object(), counter=FakeGroundingCounter()
        )
        result = asyncio.run(
            baseline.answer(
                prepared_input,
                job,
                phase="ground",
                recovered_trace=execution.traces[0],
            )
        )
        assert execution.requests[-1].model_alias == "grounding-qwen3.5-9b"
    else:
        result = asyncio.run(baseline.answer(prepared_input, job))
    assert result.output.answer_kind is AnswerKind.IMAGE
    assert result.image_items[0].image_id == "photo"
    assert len(execution.requests) == 2
    assert prepared_input.request.max_output_tokens == 512
    assert prepared_input.context_assembly.reserved_output_tokens == 512
    assert "SENTINEL" not in prepared_input.request.model_dump_json()
    if no_raw:
        for request in execution.requests:
            assert [p.image.image_id for m in request.messages for p in m.parts if p.image] == [
                "query-photo",
            ]


def test_deferred_retry_journal_retains_parse_diagnostics_and_all_costs():
    from pmmc.schemas.answer_output import TypedAnswerCandidate

    first, generation = service(["not-json", output()])
    candidate = asyncio.run(
        first.answer_typed(prepared(), model_call_id="call", defer_grounding=True)
    )
    assert isinstance(candidate, TypedAnswerCandidate)
    assert candidate.format_failure_codes and len(candidate.calls) == 2
    second, grounding = service([assessment()])
    second.gateway = None
    result = asyncio.run(
        second.answer_typed(
            prepared(),
            model_call_id=candidate.answer_model_call_id,
            allow_generation=False,
            recovered_trace=generation.traces[-1],
            prior_attempt_failures=candidate.format_failure_codes,
        )
    )
    assert result.format_failure_codes == candidate.format_failure_codes
    assert result.calls[:2] == candidate.calls
    assert len(result.calls) == 3 and len(grounding.requests) == 1
    assert grounding.requests[0].model_alias == "grounding-qwen3.5-9b"


def test_ground_only_never_repairs_invalid_or_missing_answer_journal():
    first, generation = service([output()])
    asyncio.run(first.answer_typed(prepared(), model_call_id="call", defer_grounding=True))
    second, execution = service([])
    with pytest.raises(ValueError, match="requires a generated answer journal"):
        asyncio.run(second.answer_typed(prepared(), model_call_id="call", allow_generation=False))
    trace = generation.traces[0].model_copy(
        update={"parsed_output": {"text": "bad", "finish_reason": "stop"}}
    )
    with pytest.raises(InvalidRuntimeAnswer):
        asyncio.run(
            second.answer_typed(
                prepared(), model_call_id="call", allow_generation=False, recovered_trace=trace
            )
        )
    assert not execution.requests


@pytest.mark.parametrize("allow_memory_raw_images", [True, False])
def test_existing_pmmc_provider_is_the_entry_point_for_typed_format(allow_memory_raw_images):
    class Memories:
        async def iter_visible(self, *args):
            yield memory()

    images = SimpleNamespace(get_many=AsyncMock(return_value=[]))
    provider = PmmcEvidenceAnswerInputProvider(
        memories=Memories(),
        images=images,
        counter=HeuristicTokenCounter(),
        instruction="legacy",
        answer_contract_version=TYPED_ANSWER_CONTRACT_VERSION,
        allow_memory_raw_images=allow_memory_raw_images,
    )
    ref = ProgramExecutionEvidenceRef(
        evidence_ref_id="evidence-ref",
        execution_id="execution",
        snapshot_id="snapshot",
        step_id=0,
        rank=1,
        evidence_id="memory",
        evidence_type="memory",
        memory_id="memory",
    )
    result = SimpleNamespace(
        trace=SimpleNamespace(
            query_id="query", run_id="run", execution_id="execution", plan_id="plan"
        ),
        evidence=[ref],
    )
    value = asyncio.run(
        provider.prepare(
            run_id="run",
            snapshot_id="snapshot",
            query=query().model_copy(
                update={"query_image_refs": [ImageRef(image_id="query-photo")]}
            ),
            result=result,
            variant=QueryAnswerVariant.PMMC_R0,
        )
    )
    context = validate_typed_answer_input(value)
    assert context.image_bindings[0].image_alias == "I01"
    assert value.request.response_format["json_schema"]["name"] == "typed_answer_contract_v3"
    assert value.request.max_output_tokens == 512
    pixels = [p.image.image_id for m in value.request.messages for p in m.parts if p.image]
    assert "query-photo" in pixels
    assert ("photo" in pixels) is allow_memory_raw_images
    assert context.image_bindings[0].image_id == "photo"
    if not allow_memory_raw_images:
        assert all(image.image is None for source in context.sources for image in source.images)
        assert any(image.caption for source in context.sources for image in source.images)


def test_historical_v2_payload_and_schema_are_unchanged():
    original = {
        "answer": "May",
        "supported": True,
        "used_evidence_aliases": ["E01"],
        "abstain_reason": None,
    }
    assert StructuredAnswerOutput.model_validate(original).model_dump(mode="json") == original
    validate(original, answer_response_schema(["E01"]))
    with pytest.raises(ValidationError):
        StructuredAnswerOutput.model_validate(output())
