from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy.dialects.postgresql import dialect

from pmmc.experiments.manifest import run_config_hash
from pmmc.infrastructure.database.query_executor_models import QueryExecutorRunModel
from pmmc.infrastructure.database.query_executor_repositories import (
    QueryExecutorPersistenceConflictError,
    _idempotent_identity_conflicts,
    _idempotent_insert_statement,
    _query_stage_job_identity_conflicts,
)
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.runtime.qbank_matcher import RuntimeQBankChannelHit
from pmmc.runtime.query_embeddings import required_query_embedding_channels
from pmmc.runtime.query_executor import (
    FrozenProgramBinding,
    OnlineQueryExecutor,
    QueryExecutorGraphState,
    RuntimeEvidenceRelevance,
)
from pmmc.runtime.query_executor_config import load_query_executor_config
from pmmc.runtime.query_runner import (
    QueryExecutorBatchRunner,
    _active_query_worker_count,
    _runtime_run_registration_conflicts,
    _safe_query_job_error,
)
from pmmc.runtime.query_workflow import (
    required_coverage_query_vector_pairs,
    safe_query_executor_error_diagnostics,
)
from pmmc.schemas import (
    AnswerGenerationInput,
    DatasetName,
    EvidenceType,
    FinalAnswerRecord,
    ImageRef,
    MemoryProgram,
    Modality,
    OperatorControlAction,
    OperatorExecutionPurpose,
    PlannerCapability,
    PlannerRequiredModality,
    PlannerRequirementContract,
    PmmcFallbackRunLink,
    ProgramExecutionEvidenceRef,
    ProgramExecutionStepTrace,
    ProgramExecutionTrace,
    ProgramStep,
    QBankMatchFusedCandidate,
    QBankRerankerAssessment,
    QBankRouteClass,
    QBankRuntimeChannel,
    QueryAnswerVariant,
    QueryExecutorRunManifest,
    QueryExecutorRunState,
    QueryExecutorRunStatus,
    QueryExecutorStage,
    QueryExecutorStageJob,
    QueryFallbackReason,
    QueryInput,
    QueryProgramBinding,
    QueryRouteDecision,
    QueryRouteOutcome,
    RunState,
    RunStatus,
    TraceStatus,
)
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.base import utc_now
from pmmc.schemas.query_executor import ExperimentCell

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_query_executor_config(ROOT / "configs/runtime/query-executor-v1.yaml")
CONFIG_V2 = load_query_executor_config(ROOT / "configs/runtime/query-executor-v2.yaml")


def test_step7_query_executor_insert_uses_core_table_for_metadata_column() -> None:
    manifest = QueryExecutorRunManifest(
        schema_version="0.1.0",
        query_executor_run_id="query-run-1",
        source_compiler_run_id="compiler-run-1",
        routing_projection_id="projection-1",
        snapshot_id="snapshot-1",
        dataset=DatasetName.MEMLENS,
        config_hash="config-hash",
        selected_query_count=1,
        pilot_projection=True,
        content_hash="content-hash",
        metadata={
            "answer_prompt_version": "answer-v1",
            "answer_prompt_hash": "a" * 64,
        },
    )
    row = manifest.model_dump(mode="python")
    row.update(
        status="created",
        matched_count=0,
        executed_count=0,
        pmmc_answer_count=0,
        fallback_answer_count=0,
        selected_answer_count=0,
        failed_count=0,
    )

    statement = _idempotent_insert_statement(
        QueryExecutorRunModel,
        row,
        index_elements=["query_executor_run_id"],
    )
    rendered = str(statement.compile(dialect=dialect()))

    assert statement.table is QueryExecutorRunModel.__table__
    assert "INSERT INTO pmmc_runtime.query_executor_runs" in rendered
    assert "metadata" in statement.compile(dialect=dialect()).params


def test_step7_stage_job_idempotency_ignores_mutable_scheduler_state() -> None:
    job = QueryExecutorStageJob(
        query_stage_job_id="query-stage-job-1",
        query_executor_run_id="query-run-1",
        snapshot_id="snapshot-1",
        query_id="query-1",
        stage=QueryExecutorStage.SELECT,
        content_hash="content-hash",
    )
    expected = job.model_dump(mode="python")
    actual = {
        **expected,
        "status": "running",
        "attempt_count": 1,
        "claimed_by": "query-executor-00",
        "available_at": utc_now(),
        "started_at": utc_now(),
    }

    assert _query_stage_job_identity_conflicts(actual, expected) == ()
    assert _query_stage_job_identity_conflicts(
        {**actual, "content_hash": "different"}, expected
    ) == ("content_hash",)


def test_step7_query_persistence_diagnostic_is_structural_only() -> None:
    error = QueryExecutorPersistenceConflictError(
        "private row detail",
        operation="add_query_executor_stage_jobs",
        error_code="query_stage_job_identity_conflict",
    )

    assert safe_query_executor_error_diagnostics(error) == {
        "error_type": "QueryExecutorPersistenceConflictError",
        "persistence_operation": "add_query_executor_stage_jobs",
        "error_code": "query_stage_job_identity_conflict",
    }
    assert _safe_query_job_error(error) == {
        "error_type": "QueryExecutorPersistenceConflictError",
        "persistence_operation": "add_query_executor_stage_jobs",
        "error_code": "query_stage_job_identity_conflict",
    }


def test_step7_idempotent_facts_preserve_first_insert_audit_time() -> None:
    first = PmmcFallbackRunLink(
        query_executor_run_id="query-run-1",
        fallback_run_id="fallback-run-1",
        snapshot_id="snapshot-1",
        content_hash="content-hash",
    )
    repeated = first.model_copy(
        update={"created_at": first.created_at + timedelta(seconds=1)}
    )

    assert _idempotent_identity_conflicts(
        first.model_dump(mode="python"), repeated.model_dump(mode="python")
    ) == ()
    assert _idempotent_identity_conflicts(
        first.model_dump(mode="python"),
        repeated.model_copy(update={"content_hash": "different"}).model_dump(
            mode="python"
        ),
    ) == ("content_hash",)


def test_step7_runner_reuses_prepared_terminal_state() -> None:
    manifest = QueryExecutorRunManifest(
        schema_version="0.1.0",
        query_executor_run_id="query-run-1",
        source_compiler_run_id="compiler-run-1",
        routing_projection_id="projection-1",
        snapshot_id="snapshot-1",
        dataset=DatasetName.MEMLENS,
        config_hash="config-hash",
        selected_query_count=1,
        pilot_projection=True,
        content_hash="content-hash",
        metadata={
            "answer_prompt_version": "answer-v1",
            "answer_prompt_hash": "a" * 64,
        },
    )
    state = QueryExecutorRunState(
        query_executor_run_id=manifest.query_executor_run_id,
        status=QueryExecutorRunStatus.COMPLETED,
        selected_count=1,
        fallback_answer_count=1,
        selected_answer_count=1,
        completed_at=utc_now(),
    )
    runner = QueryExecutorBatchRunner(
        sessions=object(),  # type: ignore[arg-type]
        executor=object(),  # type: ignore[arg-type]
        concurrency=1,
    )

    result = asyncio.run(
        runner.run(manifest, ["query-1"], prepared_state=state)
    )

    assert result.succeeded_count == 1
    assert result.resumed_count == 1


def test_step7_runtime_run_replay_ignores_only_first_insert_audit_fields() -> None:
    manifest = QueryExecutorRunManifest(
        schema_version="0.1.0",
        query_executor_run_id="query-run-replay",
        source_compiler_run_id="compiler-run-1",
        routing_projection_id="projection-1",
        snapshot_id="snapshot-1",
        dataset=DatasetName.MEMLENS,
        config_hash="config-hash",
        selected_query_count=1,
        pilot_projection=True,
        content_hash="content-hash",
        metadata={
            "answer_prompt_version": "answer-v1",
            "answer_prompt_hash": "a" * 64,
        },
    )
    runner = QueryExecutorBatchRunner(
        sessions=object(),  # type: ignore[arg-type]
        executor=object(),  # type: ignore[arg-type]
        concurrency=1,
    )
    first, _assignments = runner._runtime_run_registration(manifest, ["query-1"])
    replay = first.model_copy(
        update={
            "created_at": first.created_at + timedelta(seconds=1),
        }
    )
    replay = replay.model_copy(update={"config_hash": run_config_hash(replay)})

    assert _runtime_run_registration_conflicts(first, replay) == ()
    changed = replay.model_copy(update={"selection_hash": "different-selection"})
    changed = changed.model_copy(update={"config_hash": run_config_hash(changed)})
    assert _runtime_run_registration_conflicts(first, changed) == ("selection_hash",)


def test_step7_prepare_reuses_cross_process_generic_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = QueryExecutorRunManifest(
        schema_version="0.1.0",
        query_executor_run_id="query-run-replay",
        source_compiler_run_id="compiler-run-1",
        routing_projection_id="projection-1",
        snapshot_id="snapshot-1",
        dataset=DatasetName.MEMLENS,
        config_hash="config-hash",
        selected_query_count=1,
        pilot_projection=True,
        content_hash="content-hash",
        metadata={
            "answer_prompt_version": "answer-v1",
            "answer_prompt_hash": "a" * 64,
        },
    )
    runner = QueryExecutorBatchRunner(
        sessions=object(),  # type: ignore[arg-type]
        executor=object(),  # type: ignore[arg-type]
        concurrency=1,
    )
    existing, _assignments = runner._runtime_run_registration(manifest, ["query-1"])
    existing = existing.model_copy(
        update={"created_at": existing.created_at - timedelta(seconds=1)}
    )
    existing = existing.model_copy(update={"config_hash": run_config_hash(existing)})

    class FakeRuns:
        register_calls = 0

        async def get_manifest(self, _run_id: str):  # type: ignore[no-untyped-def]
            return existing

        async def get_state(self, _run_id: str):  # type: ignore[no-untyped-def]
            return RunState(
                run_id=manifest.query_executor_run_id,
                status=RunStatus.RUNNING,
                selected_count=1,
                started_at=utc_now(),
            )

        async def register(self, *_args: object):
            self.register_calls += 1

    class FakeQueryExecutor:
        add_run_calls = 0

        async def add_run(self, _manifest: QueryExecutorRunManifest) -> None:
            self.add_run_calls += 1

        async def add_stage_jobs(self, _jobs: object) -> None:
            return None

        async def get_state(self, _run_id: str) -> QueryExecutorRunState:
            return QueryExecutorRunState(
                query_executor_run_id=manifest.query_executor_run_id,
                status=QueryExecutorRunStatus.COMPLETED,
                selected_count=1,
                fallback_answer_count=1,
                selected_answer_count=1,
                completed_at=utc_now(),
            )

    fake_runs = FakeRuns()
    fake_query_executor = FakeQueryExecutor()

    class FakeUnitOfWork:
        def __init__(self, *_args: object) -> None:
            self.runs = fake_runs
            self.query_executor = fake_query_executor

        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("pmmc.runtime.query_runner.ExperimentUnitOfWork", FakeUnitOfWork)

    state = asyncio.run(runner.prepare(manifest, ["query-1"]))

    assert state.status is QueryExecutorRunStatus.COMPLETED
    assert fake_runs.register_calls == 0
    assert fake_query_executor.add_run_calls == 1


def test_step7_runner_caps_workers_to_query_denominator() -> None:
    assert _active_query_worker_count(32, 12) == 12
    assert _active_query_worker_count(8, 12) == 8
    with pytest.raises(ValueError):
        _active_query_worker_count(32, 0)


def test_step7_query_embedding_denominator_is_modality_aware() -> None:
    text_query = QueryInput(
        query_id="query-text",
        dataset=DatasetName.MEMLENS,
        conversation_id="conversation-1",
        raw_query="What was discussed?",
        available_after_session="session-2",
    )
    visual_query = QueryInput(
        query_id="query-visual",
        dataset=DatasetName.MEMLENS,
        conversation_id="conversation-1",
        raw_query="What is shown in the image?",
        query_image_refs=[ImageRef(image_id="query-image-1")],
        available_after_session="session-2",
    )

    assert required_query_embedding_channels(text_query) == {
        QBankRuntimeChannel.TEXT
    }
    assert required_query_embedding_channels(visual_query) == {
        QBankRuntimeChannel.TEXT,
        QBankRuntimeChannel.MULTIMODAL,
    }
    assert required_coverage_query_vector_pairs([text_query, visual_query]) == {
        ("query-text", "text_semantic"),
        ("query-visual", "text_semantic"),
        ("query-visual", "multimodal_semantic"),
    }


def _hit(
    qbank_id: str,
    channel: QBankRuntimeChannel,
    *,
    score: float = 1.0,
) -> RuntimeQBankChannelHit:
    return RuntimeQBankChannelHit(
        qbank_id=qbank_id,
        plan_id=f"plan-{qbank_id}",
        route_class=QBankRouteClass.ANSWERABLE,
        generation_rank=1,
        accepted_round=0,
        required_modalities=(Modality.TEXT,),
        channel=channel,
        raw_score=score,
    )


def _answer(
    *,
    run_id: str,
    query_id: str,
    variant: QueryAnswerVariant,
) -> FinalAnswerRecord:
    return FinalAnswerRecord(
        answer_id=f"answer-{run_id}-{variant.value}",
        snapshot_id="snapshot-1",
        query_id=query_id,
        run_id=run_id,
        answer_variant=variant.value,
        method="pmmc" if variant.value.startswith("pmmc_") else "native_multimodal_rag",
        answer_model_alias="answer-qwen3.5-9b",
        final_answer="A grounded answer.",
    )


@dataclass
class _FakeBackend:
    hits: list[RuntimeQBankChannelHit]
    execution_mode: str = "success"
    answer_fails: bool = False
    fallback_fails: bool = False
    persisted: list[QueryExecutorGraphState] = field(default_factory=list)
    fallback_run_id: str = "fallback-run-1"

    # This fixture executes the v1 graph; later protocol stages must fail closed.
    @property
    def source_projection_hash(self) -> str:
        raise AssertionError("v1 fixture cannot execute typed projection stages")

    async def reuse_matching_phase(self, run_id: str, query_id: str) -> None:
        raise AssertionError("v1 fixture cannot reuse typed matching phases")

    async def prepare_typed_pmmc_answer(
        self, *, query: QueryInput, result: ProgramExecutionResult, cell: ExperimentCell
    ) -> AnswerGenerationInput:
        raise AssertionError("v1 fixture cannot prepare typed answers")

    def typed_request_identity(self, prepared: AnswerGenerationInput) -> str:
        raise AssertionError("v1 fixture cannot identify typed requests")

    async def generate_deferred_typed_answer(self, prepared: AnswerGenerationInput) -> dict:
        raise AssertionError("v1 fixture cannot defer typed answers")

    async def ground_prepared_typed_answer(
        self, prepared: AnswerGenerationInput, reference: dict
    ) -> TypedAnswerResult:
        raise AssertionError("v1 fixture cannot ground typed answers")

    async def persist_matching_phase(
        self, state: QueryExecutorGraphState, prepared: dict
    ) -> None:
        raise AssertionError("v1 fixture cannot persist typed matching phases")

    async def load_runtime_phase(self, run_id: str, query_id: str, phase: str) -> dict:
        raise AssertionError("v1 fixture cannot load typed runtime phases")

    async def persist_runtime_phase(
        self, run_id: str, query_id: str, phase: str, payload: dict
    ) -> None:
        raise AssertionError("v1 fixture cannot persist typed runtime phases")

    async def generate_typed_pmmc_answer(
        self, *, query: QueryInput, frozen: FrozenProgramBinding,
        result: ProgramExecutionResult, cell: ExperimentCell,
    ) -> TypedAnswerResult:
        raise AssertionError("v1 fixture cannot generate typed answers")

    async def run_typed_native_mm_counterfactual(
        self, run_id: str, query: QueryInput
    ) -> TypedAnswerResult:
        raise AssertionError("v1 fixture cannot generate typed fallback answers")

    async def rerank_qbank_candidates(
        self, *, run_id: str, query: QueryInput,
        candidates: tuple[QBankMatchFusedCandidate, ...],
    ) -> dict[str, QBankRerankerAssessment]:
        raise AssertionError("v1 fixture cannot rerank candidates")

    async def relevant_runtime_evidence_groups(
        self, *, query: QueryInput, frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
    ) -> RuntimeEvidenceRelevance:
        raise AssertionError("v1 fixture cannot assess reranked evidence")

    async def load_query(self, run_id: str, query_id: str) -> QueryInput:
        del run_id
        return QueryInput(
            query_id=query_id,
            dataset=DatasetName.MEMLENS,
            conversation_id="conversation-1",
            raw_query="What color was the item?",
            available_after_session="session-2",
        )

    async def load_verified_projection(self, run_id: str, query: QueryInput) -> str:
        del run_id, query
        return "projection-1"

    async def ensure_precomputed_query_embeddings(
        self, run_id: str, query: QueryInput
    ) -> None:
        del run_id, query

    async def load_channel_hits(
        self, run_id: str, query: QueryInput
    ) -> list[RuntimeQBankChannelHit]:
        del run_id, query
        return self.hits

    async def bind_frozen_program(
        self,
        *,
        run_id: str,
        query: QueryInput,
        projection_id: str,
        route_decision: QueryRouteDecision,
        candidate: QBankMatchFusedCandidate,
        round_id: int,
        counterfactual: bool,
    ) -> FrozenProgramBinding:
        binding = QueryProgramBinding(
            query_program_binding_id=f"binding-{round_id}",
            query_executor_run_id=run_id,
            query_id=query.query_id,
            route_decision_id=route_decision.route_decision_id,
            routing_projection_id=projection_id,
            qbank_id=candidate.qbank_id,
            plan_id=candidate.plan_id,
            round_id=round_id,
            program_hash=f"program-hash-{round_id}",
            operator_set_version="pmmc-operators-v5",
            requirement_contract_hash=f"requirement-hash-{round_id}",
            counterfactual=counterfactual,
            content_hash=f"binding-hash-{round_id}",
        )
        requirement = PlannerRequirementContract(
            requirement_contract_id=f"requirement-{round_id}",
            run_id="compiler-run-1",
            compiler_unit_id="unit-1",
            qbank_id=candidate.qbank_id,
            required_modalities=[PlannerRequiredModality.TEXT],
            minimum_evidence_groups=1,
            required_capabilities=[
                PlannerCapability.TEXT,
                PlannerCapability.EVIDENCE_GROUPS,
            ],
            allowed_capabilities=[
                PlannerCapability.TEXT,
                PlannerCapability.EVIDENCE_GROUPS,
                PlannerCapability.VERIFICATION,
                PlannerCapability.CONTROL,
            ],
            fallback_required=True,
            content_hash=f"requirement-hash-{round_id}",
        )
        program = MemoryProgram(
            plan_id=candidate.plan_id,
            qbank_id=candidate.qbank_id,
            round_id=round_id,
            operator_sequence=[
                ProgramStep(
                    step_id=0,
                    op_name="retrieve_text_by_embedding",
                    args={"query_vector_role": "question_text_query", "top_k": 10},
                    expected_output_type="memory_ids",
                )
            ],
            required_modalities=[Modality.TEXT],
            planner_rationale_short="Retrieve relevant text evidence.",
            planner_confidence=1.0,
            program_hash=f"program-hash-{round_id}",
        )
        return FrozenProgramBinding(
            binding=binding,
            program=program,
            requirement=requirement,
            predicted_question="What color was the item?",
            collection_bindings={"text_memory": "collection-1"},
            query_vectors={"question_text_query": [1.0, 0.0]},
            source_program_hash=f"program-hash-{round_id}",
        )

    async def execute_memory_program(
        self, query: QueryInput, frozen: FrozenProgramBinding
    ) -> ProgramExecutionResult:
        if self.execution_mode == "raise":
            raise RuntimeError("safe execution failure")
        now = utc_now()
        trace = ProgramExecutionTrace(
            execution_id=f"execution-{frozen.binding.round_id}",
            run_id=frozen.binding.query_executor_run_id,
            source_compiler_run_id="compiler-run-1",
            snapshot_id="snapshot-1",
            plan_id=frozen.binding.plan_id,
            query_id=query.query_id,
            purpose=OperatorExecutionPurpose.QUERY_TIME,
            status=TraceStatus.SUCCESS,
            bindings_hash=f"bindings-{frozen.binding.round_id}",
            control_action=OperatorControlAction.CONTINUE,
            started_at=now,
            completed_at=now,
            elapsed_ms=1.0,
        )
        step = ProgramExecutionStepTrace(
            execution_id=trace.execution_id,
            query_id=query.query_id,
            run_id=trace.run_id,
            snapshot_id=trace.snapshot_id,
            plan_id=trace.plan_id,
            step_id=0,
            op_name="retrieve_text_by_embedding",
            retrieved_memory_ids=(
                [] if self.execution_mode == "empty" else ["memory-1"]
            ),
        )
        evidence: tuple[ProgramExecutionEvidenceRef, ...] = ()
        if self.execution_mode != "empty":
            evidence = (
                ProgramExecutionEvidenceRef(
                    evidence_ref_id=f"evidence-{frozen.binding.round_id}",
                    execution_id=trace.execution_id,
                    step_id=0,
                    snapshot_id=trace.snapshot_id,
                    evidence_id="memory-1",
                    evidence_type=EvidenceType.MEMORY,
                    memory_id="memory-1",
                    rank=1,
                    metadata={"evidence_group_id": "group-1"},
                ),
            )
        return ProgramExecutionResult(trace=trace, steps=(step,), evidence=evidence)

    async def generate_pmmc_answer(
        self,
        *,
        query: QueryInput,
        frozen: FrozenProgramBinding,
        result: ProgramExecutionResult,
        variant: QueryAnswerVariant,
    ) -> FinalAnswerRecord:
        del frozen, result
        if self.answer_fails:
            raise RuntimeError("safe answer failure")
        return _answer(run_id="query-run-1", query_id=query.query_id, variant=variant)

    async def run_native_mm_counterfactual(
        self, run_id: str, query: QueryInput
    ) -> FinalAnswerRecord:
        del run_id
        if self.fallback_fails:
            raise RuntimeError("safe fallback failure")
        return _answer(
            run_id=self.fallback_run_id,
            query_id=query.query_id,
            variant=QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL,
        )

    async def persist_terminal(self, state: QueryExecutorGraphState) -> None:
        self.persisted.append(state)


def _execute(backend: _FakeBackend) -> QueryExecutorGraphState:
    return asyncio.run(
        OnlineQueryExecutor(backend=backend, config=CONFIG).execute(
            run_id="query-run-1",
            query_id="query-1",
        )
    )


def test_online_executor_selects_grounded_pmmc_and_runs_counterfactual_fallback() -> None:
    backend = _FakeBackend(
        hits=[
            _hit("qbank-1", QBankRuntimeChannel.LEXICAL),
            _hit("qbank-1", QBankRuntimeChannel.TEXT),
        ]
    )

    state = _execute(backend)

    assert state["route_decision"].outcome is QueryRouteOutcome.PROGRAM
    assert state["sufficiency_by_round"][2].sufficient
    assert state["fallback_decision"].counterfactual_completed
    assert not state["fallback_decision"].active_fallback
    assert state["answer_selection"].active_variant is QueryAnswerVariant.PMMC_R2
    assert state["active_answer"].answer_variant == QueryAnswerVariant.ACTIVE_FINAL.value
    assert len(backend.persisted) == 1


def test_step7b_answer_variants_are_zero_cost_reuse_projections() -> None:
    backend = _FakeBackend(
        hits=[
            _hit("qbank-1", QBankRuntimeChannel.LEXICAL),
            _hit("qbank-1", QBankRuntimeChannel.TEXT),
        ]
    )
    state = _execute(backend)

    projected = asyncio.run(
        OnlineQueryExecutor(backend=backend, config=CONFIG_V2)._select_active_answer(
            state
        )
    )

    aliases = {
        answer.answer_variant: answer
        for answer in projected["evaluation_alias_answers"]
    }
    assert set(aliases) == {
        QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL.value,
        QueryAnswerVariant.NO_FALLBACK_COUNTERFACTUAL.value,
        QueryAnswerVariant.ACTIVE_FINAL_PROVISIONAL.value,
    }
    assert all(answer.model_call_id is None for answer in aliases.values())
    assert aliases[QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL.value].method == (
        "native_multimodal_rag"
    )
    assert projected["active_answer"].answer_variant == (
        QueryAnswerVariant.ACTIVE_FINAL_PROVISIONAL.value
    )
    assert {
        link.target_variant for link in projected["answer_reuse_links"]
    } >= set(aliases)


@pytest.mark.parametrize(
    ("hits", "expected_reason"),
    [
        ([], QueryFallbackReason.NO_QBANK_CANDIDATE),
        (
            [_hit("qbank-1", QBankRuntimeChannel.LEXICAL)],
            QueryFallbackReason.LOW_MATCH_CONFIDENCE,
        ),
        (
            [
                _hit("qbank-1", QBankRuntimeChannel.LEXICAL),
                _hit("qbank-1", QBankRuntimeChannel.TEXT),
                _hit("qbank-2", QBankRuntimeChannel.LEXICAL),
                _hit("qbank-2", QBankRuntimeChannel.TEXT),
            ],
            QueryFallbackReason.AMBIGUOUS_MATCH,
        ),
    ],
)
def test_online_executor_routes_match_gate_failures_to_native_mm(
    hits: list[RuntimeQBankChannelHit],
    expected_reason: QueryFallbackReason,
) -> None:
    state = _execute(_FakeBackend(hits=hits))

    assert state["route_decision"].outcome is QueryRouteOutcome.FALLBACK
    assert state["fallback_decision"].reason is expected_reason
    assert state["answer_selection"].active_variant is QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL
    assert not state["answer_selection"].terminal_failure


@pytest.mark.parametrize(
    ("execution_mode", "answer_fails", "expected_reason"),
    [
        ("empty", False, QueryFallbackReason.EMPTY_EVIDENCE),
        ("raise", False, QueryFallbackReason.PROGRAM_EXECUTION_FAILED),
        ("success", True, QueryFallbackReason.ANSWER_GENERATION_FAILED),
    ],
)
def test_online_executor_routes_program_failures_to_native_mm(
    execution_mode: str,
    answer_fails: bool,
    expected_reason: QueryFallbackReason,
) -> None:
    backend = _FakeBackend(
        hits=[
            _hit("qbank-1", QBankRuntimeChannel.LEXICAL),
            _hit("qbank-1", QBankRuntimeChannel.TEXT),
        ],
        execution_mode=execution_mode,
        answer_fails=answer_fails,
    )
    state = _execute(backend)

    assert state["route_decision"].outcome is QueryRouteOutcome.PROGRAM
    assert state["fallback_decision"].active_fallback
    assert state["fallback_decision"].reason is expected_reason
    assert state["answer_selection"].active_variant is QueryAnswerVariant.NATIVE_MM_COUNTERFACTUAL
    assert not state["answer_selection"].terminal_failure


def test_online_executor_persists_explicit_terminal_failure_when_fallback_fails() -> None:
    backend = _FakeBackend(hits=[], fallback_fails=True)

    state = _execute(backend)

    assert state["fallback_decision"].reason is QueryFallbackReason.NO_QBANK_CANDIDATE
    assert not state["fallback_decision"].counterfactual_completed
    assert state["fallback_decision"].error_type == "RuntimeError"
    assert state["answer_selection"].terminal_failure
    assert state["answer_selection"].selected_answer_id is None
    assert len(backend.persisted) == 1
