"""Synthetic write-time treatment tests; no queries, gold, images or models required."""

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jsonschema
import pytest
from pydantic import ValidationError

from pmmc.compiler import questioner_prompt
from pmmc.compiler.config import OPERATOR_SET_V7
from pmmc.compiler.opportunities import local_opportunity_id
from pmmc.compiler.questioner_config import (
    EVIDENCE_FIRST_VERSION,
    QuestionerConfig,
    load_questioner_config,
    questioner_config_hash,
)
from pmmc.compiler.questioner_native import (
    NativeSupportError,
    allocate_opportunities,
    native_response_schema,
    questioner_verifier_v7_output_parser,
    resolve_v36_draft,
)
from pmmc.compiler.questioner_prompt import (
    render_questioner_request,
    render_questioner_verifier_request,
)
from pmmc.compiler.questioner_runner import (
    PreparedQuestionerInputError,
    QuestionerCompilationGraph,
)
from pmmc.compiler.units import (
    FULL_REQUEST_BUDGET_VERSION,
    QuestionerInputTooLarge,
    create_compilation_unit_builder,
)
from pmmc.models.tokenization import HeuristicTokenCounter
from pmmc.schemas import AppStage, DatasetName, ExperimentJob, ModelCallTrace
from pmmc.schemas.questioner import QuestionerAnswerableDraftV36
from test_questioner_v3 import _memory, _session
from test_step8_questioner import build as native_fixture
from test_step8_questioner import verdict

CONFIGS = Path(__file__).resolve().parents[1] / "configs/compiler"


@pytest.fixture
def config():
    return load_questioner_config(CONFIGS / "questioner-release-w.yaml")


def units(config, memories, *, counter=None):
    dataset = memories[0].dataset
    sessions = [_session(s, i, dataset=dataset)
                for i, s in enumerate(sorted({m.session_id for m in memories}))]
    return create_compilation_unit_builder(
        config=config, dataset=dataset, token_counter=counter or HeuristicTokenCounter(),
    ).build(run_id="run-w", snapshot_id="snapshot-w", sessions=sessions, memories=memories)


def payload(built):
    return json.loads(built.request.messages[1].parts[0].text)


def candidate(row, *, answer_type=None, support=None):
    return QuestionerAnswerableDraftV36(
        candidate_kind="answerable", opportunity_id=row["opportunity_id"],
        generated_question="What was my recorded commute time?",
        answer_type=answer_type or row["expected_answer_types"][0],
        answer_claims=[
            dict(claim_kind="visual" if alias.startswith("I") else "text", support_alias=alias,
                 claim_text="Forty minutes.", subject="user", assertion="fact")
            for alias in (support or row["support_aliases"])
        ],
        confidence=0.9,
    )


def resolve(built, item):
    return resolve_v36_draft(item, opportunity_set=built.opportunities, memories=built.memories)


def process_job(config, built, execution, *, load):
    graph = QuestionerCompilationGraph(
        execution=execution, gateway=object(), config=config,
        load_successful_call=load, load_model_call=load, before_gate=AsyncMock(return_value={}),
        persist_v3_seed=AsyncMock(), persist_result=AsyncMock(),
    )
    job = ExperimentJob(
        job_id="w-job", run_id=built.unit.run_id, snapshot_id=built.unit.snapshot_id,
        compiler_unit_id=built.unit.compiler_unit_id, stage=AppStage.COMPILE,
        attempt_count=1, max_attempts=2,
    )
    return asyncio.run(graph.process(built, job))


def test_w_is_an_isolated_opt_in_and_native_comparison_does_not_change(config, monkeypatch):
    old, native = native_fixture()
    assert old == load_questioner_config(CONFIGS / "questioner-v3.6.yaml")
    assert questioner_config_hash(old) == (
        "fc5cd26f759781ea3edf83c5d554c58408b8b237d4ee62bc8d246c11cdcf83d6"
    )
    assert hashlib.sha256(native.request.to_json().encode()).hexdigest() == (
        "e898aefd98a68eeff6235761a33d9f053af9e9a2e832496f6a3975b95b9d7cb9"
    )
    assert config.model_dump() == {
        **old.model_dump(), "opportunity_builder_version": EVIDENCE_FIRST_VERSION,
    }
    assert config.is_v36 and config.evidence_first and not old.evidence_first
    assert questioner_config_hash(config) != questioner_config_hash(old)
    memory = _memory("m", "s", 0, "I bought a travel pass.")
    w, off = (units(c, [memory])[0] for c in (config, old))
    assert w.unit.compiler_unit_id != off.unit.compiler_unit_id
    assert w.unit.metadata["builder_identity"]["request_contract_hash"] != (
        off.unit.metadata["builder_identity"]["request_contract_hash"]
    )
    assert w.opportunities.opportunity_set.content_hash != (
        off.opportunities.opportunity_set.content_hash
    )
    assert w.request.metadata["allocation_version"] == EVIDENCE_FIRST_VERSION
    assert off.request.metadata["allocation_version"] == "questioner-query-blind-type-budget-v1"
    monkeypatch.setattr(questioner_prompt, "EVIDENCE_FIRST_INSTRUCTION", "changed instruction")
    assert units(config, [memory])[0].unit.compiler_unit_id != w.unit.compiler_unit_id
    assert units(old, [memory])[0].request.to_json() == off.request.to_json()
    older = load_questioner_config(CONFIGS / "questioner-v3.5.yaml")
    with pytest.raises(ValidationError, match="require native"):
        QuestionerConfig.model_validate({
            **older.model_dump(), "opportunity_builder_version": EVIDENCE_FIRST_VERSION,
        })
    with pytest.raises(ValidationError, match="versions are frozen"):
        QuestionerConfig.model_validate({**config.model_dump(), "opportunity_builder_version": "W"})


def test_w_allocates_distinct_needs_from_exact_support_before_the_catalog_cap(config):
    text = " ".join([
        *["I prefer taking the train."] * 30,
        *[f"I visited station {i}." for i in range(29)],
    ])
    memories = [_memory("m0", "s", 0, text), _memory("m1", "s", 1, "I booked Oxford.")]
    built = units(config, memories)[0]
    rows = payload(built)["opportunities"]
    assert len(built.opportunities.opportunities) == config.max_opportunities_per_unit
    assert len(rows) == config.question_budget_max == 8
    assert sum(row["opportunity_type"] == "preference" for row in rows) == 1
    selected = allocate_opportunities(built.opportunities)
    assert allocate_opportunities(built.opportunities, budget=24) == selected
    assert [row["opportunity_id"] for row in rows] == [local_opportunity_id(i) for i in selected]
    assert selected[:2] == allocate_opportunities(built.opportunities, budget=2)
    assert len({item.metadata["information_need_key"] for item in selected}) == 8
    bound = {s["support_alias"]: s for s in payload(built)["exact_support"]}
    assert {s["owning_memory"] for s in bound.values()} == {"M01", "M02"}
    assert any(s["exact_text"] == "I booked Oxford." for s in bound.values())
    assert sum(s["exact_text"] == "I prefer taking the train." for s in bound.values()) == 1
    assert built.request.to_json() == units(config, list(reversed(memories)))[0].request.to_json()
    unallocated = next(i for i in built.opportunities.opportunities if i not in selected)
    item = candidate(rows[0]).model_copy(
        update={"opportunity_id": local_opportunity_id(unallocated)}
    )
    with pytest.raises(NativeSupportError, match="opportunity_not_allocated"):
        resolve(built, item)


def test_w_render_schema_and_resolution_share_closed_support_and_answer_types(config):
    built = units(config, [
        _memory("m0", "s0", 0, "My Oxford commute took 20 minutes in 2024.", image_id="img-a"),
        _memory("m1", "s1", 1, "My Oxford commute now takes 40 minutes in 2025. "
                "I prefer the train.", image_id="img-b"),
    ])[-1]
    packet = payload(built)
    rows = {r["opportunity_type"]: r for r in packet["opportunities"]}
    assert all(set(row["operator_path"]) <= set(OPERATOR_SET_V7) for row in rows.values())
    update = rows["update_pair"]
    support = {s["support_alias"]: s for s in packet["exact_support"]}
    assert {support[a]["evidence_group"] for a in update["support_aliases"]} == {"G01", "G02"}
    assert update["operator_path"] == ["retrieve_text_by_embedding", "expand_temporal_neighbors"]
    assert not {"multi_evidence", "temporal_pair"} & rows.keys()  # same pair, not new needs
    assert rows["visual_detail"]["operator_path"][-1] == "inspect_raw_image"
    assert rows["visual_search"]["expected_answer_types"] == ["list"]
    assert set(rows["visual_search"]["support_aliases"]) == {"I01", "I02"}
    schema = built.request.response_format["json_schema"]["schema"]

    good = candidate(update)
    jsonschema.validate(dict(candidates=[good.model_dump(mode="json")], control_candidates=[]),
                        schema)
    assert resolve(built, good).internal.hop_count == 2
    for bad, error in [
        (candidate(update, support=rows["preference"]["support_aliases"]),
         "support_outside_opportunity"),
        (candidate(update, answer_type="count"), "answer_type_outside_opportunity"),
    ]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(dict(candidates=[bad.model_dump(mode="json")],
                                     control_candidates=[]), schema)
        with pytest.raises(NativeSupportError, match=error):
            resolve(built, bad)
    with pytest.raises(NativeSupportError, match="independent_support_groups_missing"):
        resolve(built, candidate(update, support=update["support_aliases"][:1]))
    with pytest.raises(NativeSupportError, match="image_text_support_incomplete"):
        resolve(built, candidate(rows["image_text_binding"], support=["I02"]))
    with pytest.raises(NativeSupportError, match="image_set_scan_incomplete"):
        resolve(built, candidate(rows["visual_search"], support=["I01"]))


def test_w_can_underfill_without_query_metadata_or_raw_images(config):
    original = _memory("m0", "s", 0, "Thanks! What time is it?")
    clean = units(config, [original])[0]
    tainted = original.model_copy(update={"metadata": {
        "query": "private-query", "gold": "private-gold", "judge": "private-verdict",
        "old_failures": ["private-list"],
    }})
    assert clean.request.to_json() == units(config, [tainted])[0].request.to_json()
    assert not allocate_opportunities(clean.opportunities)
    assert payload(clean)["exact_support"] == []
    assert clean.request.response_format["json_schema"]["schema"]["properties"]["candidates"][
        "maxItems"
    ] == 0
    forbidden = QuestionerConfig.model_validate({
        **config.model_dump(), "memory_raw_image_policy": "forbid",
    })
    memory = _memory("m", "s", 0, "I chose Oxford.", image_id="photo")
    packet = payload(units(forbidden, [memory])[0])
    assert {s["source_field"] for s in packet["exact_support"]} == {"text", "caption"}
    assert memory.image_ids == ["photo"]
    assert not any(r["opportunity_type"].startswith(("image", "visual"))
                   for r in packet["opportunities"])


def test_w_does_not_invent_independence_or_image_ownership(config):
    gallery = DatasetName.MEMGALLERY
    memories = [
        _memory("a", "s0", 0, "Oxford took 30 minutes.", dataset=gallery,
                source_round_id="r1", image_id="shared"),
        _memory("b", "s0", 1, "Oxford now takes 40 minutes.", dataset=gallery,
                source_round_id="r1", image_id="shared"),
    ]
    packet = payload(units(config, memories)[0])
    types = {r["opportunity_type"] for r in packet["opportunities"]}
    assert not types & {"visual_search", "temporal_pair", "update_pair", "multi_evidence"}
    assert {s["evidence_group"] for s in packet["exact_support"]} == {"G01"}
    both = memories[0].model_copy(update={
        "image_ids": ["one", "two"], "image_refs": [], "caption": "I like red.",
    })
    other = _memory("b", "s0", 1, "Thanks!", dataset=gallery, source_round_id="r1")
    packet = payload(units(config, [both, other])[0])
    types = {r["opportunity_type"] for r in packet["opportunities"]}
    assert {"visual_search", "visual_detail"} <= types
    assert not types & {"image_text_binding", "preference"}
    unrelated = memories[0].model_copy(update={"caption": None})
    unrelated_rows = payload(units(config, [unrelated, other])[0])["opportunities"]
    assert {r["opportunity_type"] for r in unrelated_rows} == {"local_fact", "visual_detail"}


def verifier_request(config, built, resolved):
    return render_questioner_verifier_request(
        request_id="verify", model_alias=config.compiler_model_alias,
        unit_id=built.unit.compiler_unit_id, cutoff_session_id=built.unit.available_after_session,
        memories=list(built.memories), sessions=list(built.sessions),
        opportunity_set=built.opportunities, candidates=[("Q01", resolved)], final_round=False,
        max_output_tokens=config.reserved_output_tokens,
        questioner_prompt_version=config.prompt_version,
        questioner_renderer_version=config.renderer_version,
        verifier_prompt_version=config.verifier_prompt_version,
    )


def test_w_exact_alias_is_not_proof_of_entailment(config):
    built = units(config, [_memory("m", "s", 0, "My commute takes twenty minutes.")])[0]
    row = payload(built)["opportunities"][0]
    resolved = resolve(built, candidate(row)).provider_for_verifier
    request = verifier_request(config, built, resolved)
    packet = json.loads(request.messages[1].parts[0].text)
    assert request.metadata["prompt_version"] == "pmmc-questioner-verifier-v7"
    assert packet["claims"][0]["claim_text"] == "Forty minutes."
    assert packet["exact_support"][0]["exact_text"] == "My commute takes twenty minutes."
    raw = verdict(resolved)
    raw["result"]["claims"][0]["claim_entailed"] = False
    verdict_parser = questioner_verifier_v7_output_parser(expected_alias="Q01", candidate=resolved)
    assert not verdict_parser(SimpleNamespace(text=json.dumps(raw))).value[
        "result"
    ]["dimensions"]["answer_explicitly_entailed"]


@pytest.fixture
def w_scan(config):
    # Two owners for one physical image, plus one independent image.
    built = units(config, [
        _memory("a", "s", 0, "This photo shows red keys.", image_id="red"),
        _memory("b", "s", 1, "Another copy of the photo.", image_id="red"),
        _memory("c", "s", 2, "This photo shows blue keys.", image_id="blue"),
    ])[0]
    scan = next(i for i in built.opportunities.opportunities if i.metadata.get("image_set_scope"))
    assert scan.metadata["support_aliases"] == ["I01", "I03"]
    draft = QuestionerAnswerableDraftV36(
        candidate_kind="answerable", opportunity_id=local_opportunity_id(scan),
        generated_question="Which photographs show a piano?", answer_type="list", confidence=0.9,
        answer_claims=[
            dict(claim_kind="visual", support_alias=a, subject="other", assertion="fact",
                 claim_text=f"The {color} piano photograph.")
            for a, color in zip(scan.evidence_aliases, ("red", "blue"), strict=True)
        ],
    )
    return built, scan, draft


@pytest.mark.parametrize(
    "defect", ["native", "truncated", "extra_owner", "swap_owner", "wrong_type"]
)
def test_invalid_scan_fails_before_dispatch(config, w_scan, defect):
    built, scan, draft = w_scan
    meta = dict(scan.metadata)
    changed = {}
    if defect == "native":
        meta.pop("support_aliases")
    elif defect == "truncated":
        meta.update(support_aliases=["I01"], image_set_size=1)
        changed.update(evidence_aliases=["I01"], evidence_group_ids=scan.evidence_group_ids[:1])
    elif defect in {"extra_owner", "swap_owner"}:
        aliases = ["I01", "I02", "I03"] if defect == "extra_owner" else ["I02", "I03"]
        meta.update(support_aliases=aliases, image_set_size=len(aliases))
        changed.update(evidence_aliases=meta["support_aliases"],
                       evidence_group_ids=[built.opportunities.alias_to_group[a]
                                           for a in meta["support_aliases"]])
    else:
        meta["expected_answer_types"] = ["entity"]
    corrupted = scan.model_copy(update={**changed, "metadata": meta})
    catalog = replace(built.opportunities, opportunities=tuple(
        corrupted if i == scan else i for i in built.opportunities.opportunities
    ))
    bad = replace(built, opportunities=catalog)
    load = AsyncMock(return_value=None)
    for operation in [
        lambda: allocate_opportunities(catalog), lambda: native_response_schema(catalog),
        lambda: resolve(bad, draft), lambda: process_job(config, bad, object(), load=load),
    ]:
        with pytest.raises(ValueError, match="evidence_first_"):
            operation()
    load.assert_not_awaited()


def test_graph_refuses_cross_treatment_inputs_before_replay(config, w_scan):
    w, _, _ = w_scan
    old, off = native_fixture()
    load = AsyncMock(return_value=None)
    for cfg, bad in [
        (config, off), (old, w), (config, replace(w, opportunities=off.opportunities)),
        (config, replace(w, unit=off.unit)), (config, replace(w, request=w.request.model_copy(
            update={"messages": off.request.messages}
        ))),
    ]:
        with pytest.raises(PreparedQuestionerInputError):
            process_job(cfg, bad, object(), load=load)
    load.assert_not_awaited()


def test_w_stale_request_contract_cannot_replay_under_same_config(config, w_scan):
    built, _, _ = w_scan
    identity = {**built.unit.metadata["builder_identity"], "request_contract_hash": "0" * 64}
    stale = built.unit.model_copy(update={
        "metadata": {**built.unit.metadata, "builder_identity": identity},
    })
    load = AsyncMock(return_value=None)
    with pytest.raises(PreparedQuestionerInputError, match="evidence_first_unit_contract_mismatch"):
        process_job(config, replace(built, unit=stale), object(), load=load)
    load.assert_not_awaited()


def test_w_graph_verifies_only_complete_bound_physical_image_scans(config, w_scan):
    built, _, draft = w_scan
    provider = resolve(built, draft).provider_for_verifier
    partial = draft.model_copy(update={"answer_claims": draft.answer_claims[:1]})
    duplicate_owner = draft.model_copy(update={"answer_claims": [
        draft.answer_claims[0].model_copy(update={"support_alias": "I02"}), draft.answer_claims[1],
    ]})
    with pytest.raises(NativeSupportError, match="image_set_scan_incomplete"):
        verifier_request(config, built, provider.model_copy(update={
            "exact_claims": provider.exact_claims[:1],
        }))
    with pytest.raises(NativeSupportError, match="verifier_candidate_mismatch"):
        verifier_request(config, built, provider.model_copy(update={"image_set_scope": False}))

    requests = []
    async def generate(_gateway, request, context, *, output_parser):
        requests.append(request)
        if request.metadata["task"] == "pmmc_questioner":
            raw = {"control_candidates": [], "candidates": [
                item.model_dump(mode="json") for item in (draft, partial, duplicate_owner)
            ]}
        else:
            packet = json.loads(request.messages[1].parts[0].text)
            assert [s["support_alias"] for s in packet["exact_support"]] == ["I01", "I03"]
            raw = verdict(provider, explicit_count_set="true")
        trace = ModelCallTrace(
            model_call_id=context.model_call_id, run_id=context.run_id,
            snapshot_id=context.snapshot_id, compiler_unit_id=context.compiler_unit_id,
            job_id=context.job_id, stage=AppStage.QUESTIONER,
            model_alias=config.compiler_model_alias, status="success",
            parsed_output=output_parser(SimpleNamespace(text=json.dumps(raw))).value,
            prompt_hash="fixture-prompt", output_asset_id="fixture-output",
        )
        return SimpleNamespace(trace=trace)

    result = process_job(config, built, SimpleNamespace(generate_audited=generate),
                         load=AsyncMock(return_value=None))
    assert [r.metadata["task"] for r in requests] == ["pmmc_questioner", "pmmc_questioner_verifier"]
    funnel = result.stage_result.metadata["questioner_funnel_v1"]
    assert funnel["counts"]["dedup_eligible"] == 1
    assert [i["first_failure_reason"] for i in funnel["items"][1:]] == [
        "image_set_scan_incomplete", "support_outside_opportunity",
    ]


@pytest.mark.parametrize("dataset", [DatasetName.MEMLENS, DatasetName.MEMGALLERY])
def test_budget_probes_count_the_same_complete_w_request_as_generation(config, dataset):
    requests = []

    class Counter(HeuristicTokenCounter):
        def count_request(self, request):
            requests.append(request.to_json())
            assert request.metadata["allocation_version"] == EVIDENCE_FIRST_VERSION
            packet = json.loads(request.messages[1].parts[0].text)
            assert packet["allocation_policy"]["version"] == EVIDENCE_FIRST_VERSION
            assert request.response_format["json_schema"]["schema"]["properties"]["candidates"]
            return super().count_request(request)

    memories = [_memory(f"m{i}", "s", i, f"I visited station {i}.", dataset=dataset,
                        source_round_id=f"r{i // 2}") for i in range(4)]
    counter = Counter()
    for built in units(config, memories, counter=counter):
        assert built.request.to_json() in requests
        assert built.input_tokens == counter.count_request(built.request) <= config.max_input_tokens
        assert built.unit.metadata["builder_identity"]["budget_policy_version"] == (
            FULL_REQUEST_BUDGET_VERSION
        )
        rebuilt = render_questioner_request(
            request_id=built.request.request_id, model_alias=config.compiler_model_alias,
            unit_id=built.unit.compiler_unit_id,
            cutoff_session_id=built.unit.available_after_session,
            memories=list(built.memories), sessions=list(built.sessions),
            max_output_tokens=config.reserved_output_tokens, prompt_version=config.prompt_version,
            renderer_version=config.renderer_version, opportunity_set=built.opportunities,
        )
        assert rebuilt.to_json() == built.request.to_json()

    class Over(Counter):
        def count_request(self, request):
            super().count_request(request)
            return config.max_input_tokens + 1

    with pytest.raises(QuestionerInputTooLarge):
        units(config, memories, counter=Over())
