from __future__ import annotations

import asyncio
from dataclasses import replace
from itertools import product

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.runtime.query_executor import OnlineQueryExecutor
from pmmc.runtime.relevance_grid import (
    evidence_reference_hash,
    project_runtime_relevance_grid,
    relevance_answer_input_provider,
    resolve_relevance_grid_answers,
    select_execution_evidence,
)
from pmmc.schemas import (
    AnswerGenerationInput,
    EvidenceType,
    QueryAnswerVariant,
    QueryEvidenceRelevanceItem,
)
from test_step8_query_backbone_context import SOURCE, TARGET, harness
from test_step8_runtime import CONFIG, PhasedBackend, hit, typed_answer


def score(result, reference, cosine, *, image=False):
    payload = {
        "reference": reference.evidence_ref_id,
        "cosine": cosine,
        "channel": "multimodal" if image else "text",
    }
    return QueryEvidenceRelevanceItem(
        relevance_item_id=canonical_content_hash(payload),
        relevance_assessment_id="assessment",
        query_executor_run_id="query-run-1", query_id="query-1",
        execution_id=result.trace.execution_id, ordinal=reference.rank - 1,
        evidence_id_hash=canonical_content_hash(
            reference.image_id if image else f"memory-text:{reference.memory_id}"
        ),
        evidence_group_hash=canonical_content_hash(reference.metadata["evidence_group_id"]),
        modality="image" if image else "text", cosine_similarity=cosine,
        normalized_score=(cosine + 1) / 2, threshold=0.3, relevant=cosine >= 0.3,
        content_hash=canonical_content_hash(payload),
        metadata={"score_transform": "raw-cosine-v1"},
    )


class GridBackend(PhasedBackend):
    relevance_grid_enabled = True

    def __init__(self):
        super().__init__([hit(round_id=r) for r in range(4)])
        self.prepared_subsets = []

    async def execute_memory_program(self, query, frozen):
        result = await super().execute_memory_program(query, frozen)
        reference = result.evidence[0]
        return replace(result, evidence=tuple(
            reference.model_copy(update={
                "evidence_ref_id": f"reference-{name}", "evidence_id": name,
                "memory_id": name, "rank": index + 1,
                "metadata": {"evidence_group_id": f"group-{name}"},
            }) for index, name in enumerate(("low", "high"))
        ))

    async def relevant_runtime_evidence_groups(self, **kwargs):
        relevance = await super().relevant_runtime_evidence_groups(**kwargs)
        result = kwargs["result"]
        return replace(relevance, items=tuple(
            score(result, item, value)
            for item, value in zip(result.evidence, (0.15, 0.45), strict=True)
        ))

    async def prepare_typed_pmmc_answer(self, **kwargs):
        prepared = await super().prepare_typed_pmmc_answer(**kwargs)
        result = kwargs["result"]
        subset = tuple(item.memory_id for item in result.evidence)
        self.prepared_subsets.append(subset)
        return prepared.model_copy(update={"source_lineage": {
            "plan_id": result.trace.plan_id, "evidence_subset_hash": canonical_content_hash(subset),
        }})

    def typed_request_identity(self, request):
        return request.source_lineage["evidence_subset_hash"]


async def fixture():
    backend = GridBackend()
    executor = OnlineQueryExecutor(backend=backend, config=CONFIG)
    state = await executor.execute(run_id="query-run-1", query_id="query-1", phase="match")
    frozen = next(iter(state["frozen_by_cell"].values()))
    result = next(iter(state["execution_by_cell"].values()))
    items = (await backend.relevant_runtime_evidence_groups(
        query=state["query"], frozen=frozen, result=result,
    )).items
    return backend, executor, state, frozen, result, items


def test_real_grid_is_full_cartesian_and_filters_active_context_without_extra_cells():
    async def scenario():
        backend, executor, state, _frozen, _result, _items = await fixture()
        assert len(state["cell_results"]) == 48
        assert backend.executed == ["plan-q1"]
        assert backend.prepared_subsets == [("low", "high"), ("high",)]
        main = [fact for fact in state["cell_results"] if fact.cell.treatment == "pmmc"]
        for fact in main:
            points = fact.metadata["relevance_grid"]
            assert len(points) == 25
            assert len({(p["text_threshold"], p["multimodal_threshold"]) for p in points}) == 25
            assert sum(p["status"] == "relevance_rejected" for p in points) == 5
            assert fact.sufficiency.independent_group_count == 1
            assert fact.sufficiency.answer_support is None
        assert not backend.events
        await executor.execute(run_id="query-run-1", query_id="query-1", phase="answer")
        assert len(backend.events) == 2
        terminal = await executor.execute(run_id="query-run-1", query_id="query-1", phase="select")
        assert len(backend.events) == 4
        assert len(terminal["typed_answers"]) == 2
        for fact in terminal["cell_results"]:
            if fact.answer_source_cell_id is not None:
                assert fact.answer_source_cell_id in {f.cell_id for f in terminal["cell_results"]}
            points = fact.metadata["relevance_grid"]
            assert sum(p["status"] == "completed" for p in points) == 20
            for point in points:
                if point["status"] == "completed":
                    assert point["typed_answer_id"] in terminal["typed_answers"]
                    assert point["grounding_call_ids"]
        # Filtering never rewrites the executed evidence or its original lineage.
        assert len(next(iter(state["execution_by_cell"].values())).evidence) == 2

    asyncio.run(scenario())


def project(frozen, result, items):
    return project_runtime_relevance_grid(
        run_id="query-run-1", cell_id="cell", binding=frozen.binding,
        requirement=frozen.requirement, result=result, items=items,
        thresholds=CONFIG.relevance_cosine_grid,
    )


def test_missing_scores_do_not_fabricate_grid_answers_or_support():
    _backend, _executor, _state, frozen, result, _items = asyncio.run(fixture())
    points = project(frozen, result, ())
    assert all(not value.sufficiency.sufficient for value in points)
    assert all(value.point["answer_support"] is None for value in points)
    assert all(value.point["status"] == "incomplete" for value in points)
    assert all(value.point["error_type"] == "MissingEvidenceRelevanceScore" for value in points)


def test_multimodal_thresholds_are_independent_and_raw_image_survives_only_when_relevant():
    _backend, _executor, _state, frozen, result, items = asyncio.run(fixture())
    image = result.evidence[1].model_copy(update={
        "memory_id": None, "image_id": "picture", "evidence_type": EvidenceType.RAW_IMAGE,
    })
    # Pydantic model_copy is intentionally not a validating constructor.
    image = type(image).model_validate(image.model_dump())
    result = replace(result, evidence=(result.evidence[0], image))
    items = (items[0], score(result, image, 0.25, image=True))
    points = {(p.point["text_threshold"], p.point["multimodal_threshold"]): p
              for p in project(frozen, result, items)}
    assert len(points[0.1, 0.1].result.evidence) == 2
    assert len(points[0.1, 0.3].result.evidence) == 1
    assert points[0.3, 0.1].result.evidence == (image,)
    assert not points[0.3, 0.3].result.evidence
    assert points[0.1, 0.1].sufficiency.raw_image_reached
    assert not points[0.1, 0.3].sufficiency.raw_image_reached


def test_multi_group_role_gate_and_score_scope_are_not_bypassed():
    _backend, _executor, _state, frozen, result, items = asyncio.run(fixture())
    requirement = frozen.requirement.model_copy(update={
        "minimum_evidence_groups": 2,
        "metadata": {"required_semantic_roles": ["temporal_neighbor"]},
    })
    assert not any(p.sufficiency.sufficient for p in project(
        replace(frozen, requirement=requirement), result, items,
    ))
    with pytest.raises(ValueError, match="scoped raw-cosine"):
        project(frozen, result, (items[0].model_copy(update={"query_id": "another-query"}),))
    with pytest.raises(ValueError, match="scoped raw-cosine"):
        project(frozen, result, (items[0].model_copy(update={"metadata": {}}),))
    with pytest.raises(ValueError, match="escaped"):
        select_execution_evidence(result, ["unknown-reference"])
    ref_hash = evidence_reference_hash(result.evidence[0])
    with pytest.raises(ValueError, match="escaped"):
        select_execution_evidence(result, [ref_hash, ref_hash])


def test_grid_grounding_failure_and_missing_answer_are_distinct_from_relevance_rejection():
    points = [
        {"prepared_key": "a", "status": "pending_answer"},
        {"prepared_key": "b", "status": "pending_answer"},
        {"prepared_key": None, "status": "relevance_rejected"},
    ]
    resolved = resolve_relevance_grid_answers(points, {"a": typed_answer("a", abstain=True)}, {})
    assert resolved[0]["status"] == "completed"
    assert resolved[0]["terminal_state"] == "abstained"
    assert resolved[0]["answer_support"] is False
    assert resolved[1]["status"] == "incomplete"
    assert resolved[1]["error_type"] == "MissingGridAnswer"
    assert resolved[2] == points[2]


def test_grid_consumer_rebuilds_untrimmed_subset_and_excludes_unscored_images(monkeypatch):
    case = harness(monkeypatch, "openai")
    execution = case.state.execution
    main_id = case.state.checkpoint["cells"][0]["cell_id"]
    refs = [evidence_reference_hash(ref) for ref in execution.evidence[:2]]
    points = [{
        "protocol": "runtime-relevance-context-grid-v1",
        "query_executor_run_id": SOURCE, "query_id": case.query.query_id,
        "source_cell_id": main_id, "execution_id": execution.trace.execution_id,
        "grid_point_id": f"grid-{text}-{image}", "text_threshold": text,
        "multimodal_threshold": image, "selected_reference_hashes": refs,
        "evidence_subset_hash": canonical_content_hash(refs),
        "prepared_key": main_id,
    } for text, image in product(CONFIG.relevance_cosine_grid, repeat=2)]
    for cell in case.state.checkpoint["cells"]:
        cell["metadata"]["relevance_grid"] = points
    before = canonical_content_hash(case.state.checkpoint)
    asyncio.run(case.backend.reuse_matching_phase(TARGET, case.query.query_id))
    saved = case.persisted.call_args.args[3]
    rebuilt = AnswerGenerationInput.model_validate(saved["prepared"][main_id])
    # The source provider fitted only memory-0; target gets the actual two-ref subset.
    assert rebuilt.evidence_ids_in_prompt == ["memory-0", "memory-1"]
    assert rebuilt.image_ids_in_prompt == ["query-photo"]
    for cell in saved["cells"]:
        assert len(cell["metadata"]["relevance_grid"]) == 25
        assert all(p["query_executor_run_id"] == TARGET
                   for p in cell["metadata"]["relevance_grid"])
    assert canonical_content_hash(case.state.checkpoint) == before


def test_projected_provider_does_not_mutate_shared_memory_reader(monkeypatch):
    case = harness(monkeypatch, "openai")
    original = case.backend.answer_inputs
    filtered = relevance_answer_input_provider(original, case.state.execution)
    assert filtered is not original and filtered.memories is not original.memories
    async def scenario():
        prepared = await filtered.prepare(
            run_id=TARGET, snapshot_id="snapshot", query=case.query,
            result=case.state.execution, variant=QueryAnswerVariant.PMMC_B8_R2,
            source_execution_run_id=SOURCE,
        )
        assert prepared.image_ids_in_prompt == ["query-photo"]
        untouched = [memory async for memory in original.memories.iter_visible(
            "snapshot", "conversation", "session",
        )]
        assert all(memory.image_refs for memory in untouched)
    asyncio.run(scenario())


def test_failed_grid_context_does_not_retry_once_per_threshold_point():
    class FailingGrid(GridBackend):
        async def generate_deferred_typed_answer(self, request):
            self.events.append(("answer", self.typed_request_identity(request)))
            raise TimeoutError("do not leak provider details")
    backend = FailingGrid()
    state = asyncio.run(OnlineQueryExecutor(backend=backend, config=CONFIG).execute(
        run_id="query-run-1", query_id="query-1",
    ))
    assert len(backend.events) == 2
    points = state["cell_results"][0].metadata["relevance_grid"]
    assert {p.get("error_type") for p in points if p["sufficient"]} == {"TimeoutError"}
    assert all(p["status"] == "incomplete" for p in points if p["sufficient"])


def test_phased_grounding_failure_does_not_retry_per_grid_point_or_cell():
    class FailingGrounding(GridBackend):
        async def ground_prepared_typed_answer(self, prepared, reference):
            self.events.append(("ground", self.typed_request_identity(prepared)))
            raise TimeoutError("provider details remain private")
    async def scenario():
        backend = FailingGrounding()
        executor = OnlineQueryExecutor(backend=backend, config=CONFIG)
        for phase in ("match", "answer", "select"):
            state = await executor.execute(run_id="query-run-1", query_id="query-1", phase=phase)
        assert sum(event[0] == "ground" for event in backend.events) == 2
        for fact in state["cell_results"]:
            assert all(point["status"] == "incomplete"
                       for point in fact.metadata["relevance_grid"] if point["sufficient"])
    asyncio.run(scenario())
