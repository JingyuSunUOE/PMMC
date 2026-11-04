"""Transactional repositories for experiment and evaluation facts."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeVar, cast

from sqlalchemy import and_, func, insert, or_, select, tuple_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, aliased

from pmmc.experiments.manifest import run_config_hash
from pmmc.infrastructure.database.baseline_models import (
    ContextAssemblyModel,
    RetrievalCorpusModel,
    RetrievalDocumentModel,
)
from pmmc.infrastructure.database.errors import (
    InvalidStateTransitionError,
    PersistenceConflictError,
)
from pmmc.infrastructure.database.experiment_mappers import (
    artifact_from_row,
    artifact_to_row,
    bad_case_label_to_row,
    embedding_collection_from_row,
    embedding_collection_to_row,
    embedding_from_row,
    embedding_to_row,
    final_answer_from_parts,
    final_answer_to_rows,
    job_attempt_to_row,
    job_from_row,
    job_to_row,
    judge_score_to_row,
    metric_score_to_row,
    model_call_from_row,
    model_call_to_row,
    retrieval_batch_from_row,
    retrieval_batch_to_row,
    retrieval_component_from_row,
    retrieval_component_to_row,
    retrieval_evaluation_to_rows,
    retrieval_query_embedding_from_row,
    retrieval_query_embedding_to_row,
    retrieval_trace_from_row,
    retrieval_trace_to_row,
    run_query_to_row,
    run_stage_summary_to_row,
    run_state_from_row,
    run_state_to_row,
)
from pmmc.infrastructure.database.experiment_models import (
    BadCaseLabelModel,
    EmbeddingCollectionModel,
    EmbeddingModel,
    EvaluationArtifactModel,
    EvaluationModelCallModel,
    ExperimentJobModel,
    FinalAnswerEvidenceIdModel,
    FinalAnswerImageIdModel,
    FinalAnswerModel,
    JobAttemptModel,
    JudgeScoreModel,
    MetricScoreModel,
    RetrievalBatchModel,
    RetrievalBatchQueryEmbeddingModel,
    RetrievalEvaluationMatchModel,
    RetrievalEvaluationModel,
    RetrievalTraceComponentModel,
    RetrievalTraceModel,
    RunQueryModel,
    RunStageSummaryModel,
    RunStateModel,
    RuntimeArtifactModel,
    RuntimeModelCallModel,
)
from pmmc.infrastructure.database.mappers import run_manifest_to_row
from pmmc.infrastructure.database.models import (
    BenchmarkSessionModel,
    RunManifestModel,
    RunSnapshotLinkModel,
)
from pmmc.schemas import (
    AppStage,
    ArtifactState,
    BadCaseLabel,
    BaselineMethod,
    CandidateGateStatus,
    CompilationStage,
    DataScope,
    DistanceMetric,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    EmbeddingRecord,
    EmbeddingSearchHit,
    EmbeddingSourceType,
    ExperimentArtifact,
    ExperimentJob,
    FinalAnswerRecord,
    JobAttempt,
    JobStatus,
    JudgeScore,
    MetricScore,
    ModelCallTrace,
    PipelineStageSummary,
    RankingScore,
    RetrievalBatchTrace,
    RetrievalEvaluation,
    RetrievalQueryEmbeddingRef,
    RetrievalScoreComponent,
    RetrievalTrace,
    RetrievalTraceComponent,
    RunManifest,
    RunQueryAssignment,
    RunState,
    RunStatus,
    ScoreDirection,
    TraceStatus,
    WindowSearchHit,
)
from pmmc.schemas.base import utc_now

if TYPE_CHECKING:
    from pmmc.infrastructure.database.protocols import (
        CompilerRunRepository,
        CompilerStageJobRepository,
        CompilerUnitRepository,
        ContextAssemblyRepository,
        DoubterRepository,
        MemoryOperatorIndexRepository,
        MemoryProgramRepository,
        PlannerRepository,
        ProgramDryRunRepository,
        ProgramExecutionRepository,
        ProgramQueryEmbeddingRepository,
        QBankCompilerRepository,
        QBankRoutingRepository,
        QBankSemanticMatchRepository,
        StructuredOutputDiagnosticRepository,
    )
    from pmmc.memory.operators import OperatorDataRepository

ModelT = TypeVar("ModelT", bound=DeclarativeBase)


def _model_mapping(value: DeclarativeBase) -> dict[str, Any]:
    return {
        prop.columns[0].name: getattr(value, prop.key) for prop in value.__mapper__.column_attrs
    }


def _comparable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    return value


async def _insert_idempotent(
    session: AsyncSession,
    model: type[ModelT],
    row: dict[str, Any],
    *identity: Any,
) -> ModelT:
    existing = await session.get(model, identity[0] if len(identity) == 1 else identity)
    if existing is not None:
        actual = _model_mapping(existing)
        if any(_comparable(actual[key]) != _comparable(value) for key, value in row.items()):
            raise PersistenceConflictError(
                f"{model.__tablename__} stable ID already exists with different content"
            )
        return existing
    try:
        async with session.begin_nested():
            await session.execute(insert(cast(Any, model.__table__)).values(row))
    except IntegrityError:
        existing = await session.get(model, identity[0] if len(identity) == 1 else identity)
        if existing is None:
            raise
        actual = _model_mapping(existing)
        if any(_comparable(actual[key]) != _comparable(value) for key, value in row.items()):
            raise PersistenceConflictError(
                f"{model.__tablename__} stable ID already exists with different content"
            ) from None
        return existing
    inserted = await session.get(model, identity[0] if len(identity) == 1 else identity)
    if inserted is None:
        raise RuntimeError(f"failed to read inserted {model.__tablename__} row")
    return inserted


async def _insert_many_idempotent(
    session: AsyncSession,
    model: type[ModelT],
    rows: list[dict[str, Any]],
    *,
    identity_key: str,
    alternate_keys: tuple[str, ...] = (),
) -> None:
    """Bulk insert after validating stable and alternate identities."""

    if not rows:
        return
    identities = [row[identity_key] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate {identity_key} values in one write batch")
    alternate_values = [tuple(row[key] for key in alternate_keys) for row in rows]
    if alternate_keys and len(alternate_values) != len(set(alternate_values)):
        raise ValueError("duplicate alternate identities in one write batch")

    async def load_existing() -> list[ModelT]:
        predicates = [getattr(model, identity_key).in_(identities)]
        if alternate_keys:
            predicates.append(
                tuple_(*(getattr(model, key) for key in alternate_keys)).in_(alternate_values)
            )
        return list((await session.execute(select(model).where(or_(*predicates)))).scalars().all())

    def validate_existing(existing: list[ModelT]) -> list[dict[str, Any]]:
        by_identity = {getattr(value, identity_key): value for value in existing}
        by_alternate = {
            tuple(getattr(value, key) for key in alternate_keys): value for value in existing
        }
        missing: list[dict[str, Any]] = []
        for row in rows:
            stored = by_identity.get(row[identity_key])
            alternate = tuple(row[key] for key in alternate_keys)
            alternate_match = by_alternate.get(alternate) if alternate_keys else None
            if stored is None and alternate_match is not None:
                raise PersistenceConflictError(
                    f"{model.__tablename__} alternate identity belongs to a different stable ID"
                )
            if stored is None:
                missing.append(row)
                continue
            actual = _model_mapping(stored)
            if any(_comparable(actual[key]) != _comparable(value) for key, value in row.items()):
                raise PersistenceConflictError(
                    f"{model.__tablename__} stable ID already exists with different content"
                )
        return missing

    missing = validate_existing(await load_existing())
    if not missing:
        return
    try:
        async with session.begin_nested():
            await session.execute(insert(cast(Any, model.__table__)).values(missing))
    except IntegrityError:
        # A concurrent exact replay is valid; re-read and compare before deciding.
        remaining = validate_existing(await load_existing())
        if remaining:
            raise


class PostgresRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(
        self, manifest: RunManifest, assignments: list[RunQueryAssignment]
    ) -> RunState:
        if not manifest.dataset_snapshot_ids:
            raise ValueError("run registration requires at least one snapshot")
        if manifest.selection_hash is None:
            raise ValueError("run registration requires manifest.selection_hash")
        if manifest.config_hash != run_config_hash(manifest):
            raise ValueError("run registration requires a valid config_hash")
        if [item.ordinal for item in assignments] != list(range(len(assignments))):
            raise ValueError("run query ordinals must be contiguous and ordered from 0")
        for item in assignments:
            if item.run_id != manifest.run_id or item.selection_hash != manifest.selection_hash:
                raise ValueError("run query assignment does not match manifest")
            if item.snapshot_id not in manifest.dataset_snapshot_ids:
                raise ValueError("run query snapshot is not linked by manifest")

        await _insert_idempotent(
            self.session, RunManifestModel, run_manifest_to_row(manifest), manifest.run_id
        )
        stored_state = await self.session.get(RunStateModel, manifest.run_id)
        if stored_state is None:
            state = RunState(run_id=manifest.run_id, selected_count=len(assignments))
            await _insert_idempotent(
                self.session, RunStateModel, run_state_to_row(state), manifest.run_id
            )
        else:
            state = run_state_from_row(_model_mapping(stored_state))
            if state.selected_count != len(assignments):
                raise PersistenceConflictError(
                    "registered run state has a different selected query count"
                )
        for snapshot_id in manifest.dataset_snapshot_ids:
            await _insert_idempotent(
                self.session,
                RunSnapshotLinkModel,
                {"run_id": manifest.run_id, "snapshot_id": snapshot_id},
                manifest.run_id,
                snapshot_id,
            )
        for assignment in assignments:
            await _insert_idempotent(
                self.session,
                RunQueryModel,
                run_query_to_row(assignment),
                assignment.run_id,
                assignment.snapshot_id,
                assignment.query_id,
            )
        return state

    async def get_manifest(self, run_id: str) -> RunManifest:
        value = await self.session.get(RunManifestModel, run_id)
        if value is None:
            raise KeyError(f"run does not exist: {run_id}")
        return RunManifest.model_validate(value.manifest_json)

    async def add_stage_summary(
        self,
        run_id: str,
        summary: PipelineStageSummary,
    ) -> PipelineStageSummary:
        await _insert_idempotent(
            self.session,
            RunStageSummaryModel,
            run_stage_summary_to_row(run_id, summary),
            run_id,
            summary.stage,
        )
        return summary

    async def get_state(self, run_id: str) -> RunState:
        value = await self.session.get(RunStateModel, run_id)
        if value is None:
            raise KeyError(f"run does not exist: {run_id}")
        return run_state_from_row(_model_mapping(value))

    async def transition(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_type: str | None = None,
        error_summary: str | None = None,
        error_artifact_id: str | None = None,
    ) -> RunState:
        stored = await self.session.scalar(
            select(RunStateModel).where(RunStateModel.run_id == run_id).with_for_update()
        )
        if stored is None:
            raise KeyError(f"run does not exist: {run_id}")
        current = run_state_from_row(_model_mapping(stored))
        allowed = {
            RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
            RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED},
        }
        if status not in allowed.get(current.status, set()):
            raise InvalidStateTransitionError(
                f"cannot transition run from {current.status.value} to {status.value}"
            )
        if status is RunStatus.FAILED and error_type is None:
            raise ValueError("failed runs require error_type")
        now = utc_now()
        if status is RunStatus.COMPLETED:
            incomplete = await self.session.scalar(
                select(ExperimentJobModel.job_id)
                .where(
                    ExperimentJobModel.run_id == run_id,
                    ExperimentJobModel.status != JobStatus.SUCCEEDED.value,
                )
                .limit(1)
            )
            covered_queries = await self.session.scalar(
                select(func.count(func.distinct(ExperimentJobModel.query_id))).where(
                    ExperimentJobModel.run_id == run_id,
                    ExperimentJobModel.query_id.is_not(None),
                    ExperimentJobModel.status == JobStatus.SUCCEEDED.value,
                )
            )
            if incomplete is not None or covered_queries != current.selected_count:
                raise InvalidStateTransitionError(
                    "completed runs require successful jobs for every selected query"
                )
            await self._assert_runtime_completion_integrity(run_id, current.selected_count)
        values: dict[str, Any] = {
            "status": status.value,
            "heartbeat_at": now,
            "error_type": error_type,
            "error_summary": error_summary,
            "error_artifact_id": error_artifact_id,
        }
        if status is RunStatus.RUNNING:
            values["started_at"] = now
        if status is RunStatus.COMPLETED:
            values["succeeded_count"] = current.selected_count
            values["failed_count"] = 0
        if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            values["completed_at"] = now
        await self.session.execute(
            update(RunStateModel).where(RunStateModel.run_id == run_id).values(**values)
        )
        await self.session.flush()
        return await self.get_state(run_id)

    async def _assert_runtime_completion_integrity(
        self,
        run_id: str,
        selected_count: int,
    ) -> None:
        assignments = int(
            await self.session.scalar(
                select(func.count())
                .select_from(RunQueryModel)
                .where(RunQueryModel.run_id == run_id)
            )
            or 0
        )
        if assignments != selected_count:
            raise InvalidStateTransitionError("run query denominator differs from selected_count")
        await self._assert_compiler_completion_integrity(run_id, selected_count)
        non_ready_artifact = await self.session.scalar(
            select(RuntimeArtifactModel.artifact_id)
            .where(
                RuntimeArtifactModel.run_id == run_id,
                RuntimeArtifactModel.state != ArtifactState.READY.value,
            )
            .limit(1)
        )
        if non_ready_artifact is not None:
            raise InvalidStateTransitionError("completed runs require ready runtime artifacts")
        invalid_tokens = await self.session.scalar(
            select(RuntimeModelCallModel.model_call_id)
            .where(
                RuntimeModelCallModel.run_id == run_id,
                RuntimeModelCallModel.input_tokens.is_not(None),
                RuntimeModelCallModel.output_tokens.is_not(None),
                RuntimeModelCallModel.total_tokens.is_not(None),
                RuntimeModelCallModel.total_tokens
                != RuntimeModelCallModel.input_tokens + RuntimeModelCallModel.output_tokens,
            )
            .limit(1)
        )
        if invalid_tokens is not None:
            raise InvalidStateTransitionError("completed run has inconsistent token totals")
        incomplete_collection = await self.session.scalar(
            select(EmbeddingCollectionModel.collection_id)
            .where(
                EmbeddingCollectionModel.run_id == run_id,
                EmbeddingCollectionModel.status != EmbeddingCollectionStatus.READY.value,
            )
            .limit(1)
        )
        if incomplete_collection is not None:
            raise InvalidStateTransitionError("completed runs require ready embedding collections")

        batch_rows = (
            await self.session.execute(
                select(
                    RetrievalBatchModel.retrieval_batch_id,
                    RetrievalBatchModel.returned_count,
                    RetrievalBatchModel.requested_top_k,
                    func.count(RetrievalTraceModel.retrieval_trace_id),
                    func.min(RetrievalTraceModel.rank),
                    func.max(RetrievalTraceModel.rank),
                )
                .outerjoin(
                    RetrievalTraceModel,
                    RetrievalTraceModel.retrieval_batch_id
                    == RetrievalBatchModel.retrieval_batch_id,
                )
                .where(RetrievalBatchModel.run_id == run_id)
                .group_by(RetrievalBatchModel.retrieval_batch_id)
            )
        ).all()
        manifest = await self.session.get(RunManifestModel, run_id)
        for batch_id, returned, requested, count, minimum, maximum in batch_rows:
            expected_min = 1 if returned else None
            expected_max = returned if returned else None
            if count != returned or minimum != expected_min or maximum != expected_max:
                raise InvalidStateTransitionError(
                    f"retrieval batch ranks are incomplete: {batch_id}"
                )
            if (
                manifest is not None
                and manifest.retrieval_topk_max is not None
                and (
                    requested > manifest.retrieval_topk_max
                    or (manifest.paper_mode and requested != manifest.retrieval_topk_max)
                )
            ):
                raise InvalidStateTransitionError(
                    f"retrieval batch violates manifest max K: {batch_id}"
                )

        if manifest is not None and manifest.method in {
            BaselineMethod.FULL_CONTEXT_CAPTIONS.value,
            BaselineMethod.FULL_CONTEXT_RAW_IMAGES.value,
        }:
            answer_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(FinalAnswerModel)
                    .where(
                        FinalAnswerModel.run_id == run_id,
                        FinalAnswerModel.answer_variant == "primary",
                        FinalAnswerModel.method == manifest.method,
                    )
                )
                or 0
            )
            context_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(ContextAssemblyModel)
                    .where(
                        ContextAssemblyModel.run_id == run_id,
                        ContextAssemblyModel.method == manifest.method,
                    )
                )
                or 0
            )
            call_query_count = int(
                await self.session.scalar(
                    select(func.count(func.distinct(RuntimeModelCallModel.query_id))).where(
                        RuntimeModelCallModel.run_id == run_id,
                        RuntimeModelCallModel.stage == "answer",
                        RuntimeModelCallModel.status == TraceStatus.SUCCESS.value,
                        RuntimeModelCallModel.context_assembly_id.is_not(None),
                    )
                )
                or 0
            )
            retrieval_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(RetrievalBatchModel)
                    .where(RetrievalBatchModel.run_id == run_id)
                )
                or 0
            )
            embedding_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(EmbeddingCollectionModel)
                    .where(EmbeddingCollectionModel.run_id == run_id)
                )
                or 0
            )
            if (
                RunManifest.model_validate(manifest.manifest_json).metadata.get(
                    "answer_contract_version"
                )
                == "typed-answer-contract-v3"
            ):
                typed = await self.assert_typed_answer_terminals(run_id, manifest.method)
                answer_count = typed["answered_count"] + typed["abstained_count"]
            if answer_count != selected_count:
                raise InvalidStateTransitionError(
                    "full-context runs require one primary answer per selected query"
                )
            if context_count != selected_count or call_query_count != selected_count:
                raise InvalidStateTransitionError(
                    "full-context runs require context and successful answer call coverage"
                )
            if retrieval_count or embedding_count:
                raise InvalidStateTransitionError(
                    "full-context runs cannot persist retrieval or embedding facts"
                )
        if manifest is not None and manifest.method in {
            BaselineMethod.CAPTION_TEXT_RAG.value,
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG.value,
            BaselineMethod.NATIVE_MULTIMODAL_RAG.value,
        }:
            retrieval_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(RetrievalBatchModel)
                    .where(
                        RetrievalBatchModel.run_id == run_id,
                        RetrievalBatchModel.method == manifest.method,
                        RetrievalBatchModel.query_embedding_collection_id.is_not(None),
                    )
                )
                or 0
            )
            context_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(ContextAssemblyModel)
                    .where(
                        ContextAssemblyModel.run_id == run_id,
                        ContextAssemblyModel.method == manifest.method,
                        ContextAssemblyModel.retrieval_batch_id.is_not(None),
                    )
                )
                or 0
            )
            answer_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(FinalAnswerModel)
                    .where(
                        FinalAnswerModel.run_id == run_id,
                        FinalAnswerModel.method == manifest.method,
                        FinalAnswerModel.answer_variant == "primary",
                    )
                )
                or 0
            )
            answer_call_queries = int(
                await self.session.scalar(
                    select(func.count(func.distinct(RuntimeModelCallModel.query_id))).where(
                        RuntimeModelCallModel.run_id == run_id,
                        RuntimeModelCallModel.stage == AppStage.ANSWER.value,
                        RuntimeModelCallModel.status == TraceStatus.SUCCESS.value,
                        RuntimeModelCallModel.context_assembly_id.is_not(None),
                    )
                )
                or 0
            )
            retrieve_jobs = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(ExperimentJobModel)
                    .where(
                        ExperimentJobModel.run_id == run_id,
                        ExperimentJobModel.stage == AppStage.RETRIEVE.value,
                        ExperimentJobModel.status == JobStatus.SUCCEEDED.value,
                    )
                )
                or 0
            )
            answer_jobs = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(ExperimentJobModel)
                    .where(
                        ExperimentJobModel.run_id == run_id,
                        ExperimentJobModel.stage == AppStage.ANSWER.value,
                        ExperimentJobModel.status == JobStatus.SUCCEEDED.value,
                    )
                )
                or 0
            )
            if retrieval_count != selected_count:
                raise InvalidStateTransitionError(
                    "RAG runs require one complete retrieval batch per selected query"
                )
            terminal_count = answer_count
            if (
                RunManifest.model_validate(manifest.manifest_json).metadata.get(
                    "answer_contract_version"
                )
                == "typed-answer-contract-v3"
            ):
                terminals = await self.assert_typed_answer_terminals(run_id, manifest.method)
                terminal_count = terminals["answered_count"] + terminals["abstained_count"]
            if (
                RunManifest.model_validate(manifest.manifest_json).metadata.get(
                    "answer_contract_version"
                )
                == "strict-answer-contract-v2"
            ):
                terminals = await self.assert_rag_answer_terminals(run_id, manifest.method)
                terminal_count = terminals["answered_count"] + terminals["abstained_count"]
            if context_count != selected_count or terminal_count != selected_count:
                raise InvalidStateTransitionError(
                    "RAG runs require one context and primary answer terminal per selected query"
                )
            if answer_call_queries != selected_count:
                raise InvalidStateTransitionError(
                    "RAG runs require a successful audited answer call per selected query"
                )
            if retrieve_jobs != selected_count or answer_jobs != selected_count:
                raise InvalidStateTransitionError(
                    "RAG runs require successful retrieve and answer jobs for every query"
                )

    async def assert_typed_answer_terminals(self, run_id: str, method: str) -> dict[str, int]:
        from pmmc.infrastructure.database.query_executor_models import QueryRuntimeTypedAnswerModel
        from pmmc.schemas.answer_output import TypedAnswerResult

        rows = (
            await self.session.scalars(
                select(QueryRuntimeTypedAnswerModel).where(
                    QueryRuntimeTypedAnswerModel.query_executor_run_id == run_id,
                )
            )
        ).all()
        expected = set(
            (
                await self.session.scalars(
                    select(RunQueryModel.query_id).where(
                        RunQueryModel.run_id == run_id,
                    )
                )
            ).all()
        )
        seen: set[str] = set()
        counts = {"answered_count": 0, "abstained_count": 0}
        required_calls: set[str] = set()
        for row in rows:
            try:
                result = TypedAnswerResult.model_validate(row.payload["metadata"]["typed_result"])
            except (ValueError, KeyError, TypeError) as exc:
                raise InvalidStateTransitionError("typed_answer_payload_invalid") from exc
            if (
                result.run_id != run_id
                or result.query_id not in expected
                or result.method != method
                or result.answer_variant != "primary"
                or result.query_id in seen
                or result.terminal_state not in {"answered", "abstained"}
            ):
                raise InvalidStateTransitionError("typed_answer_terminal_lineage_invalid")
            seen.add(result.query_id)
            counts[
                "answered_count" if result.terminal_state == "answered" else "abstained_count"
            ] += 1
            required_calls.update(call.model_call_id for call in result.calls)
        if seen != expected:
            raise InvalidStateTransitionError("typed_answer_terminal_denominator_incomplete")
        actual = set(
            (
                await self.session.scalars(
                    select(RuntimeModelCallModel.model_call_id).where(
                        RuntimeModelCallModel.run_id == run_id,
                        RuntimeModelCallModel.model_call_id.in_(required_calls),
                        RuntimeModelCallModel.status == TraceStatus.SUCCESS.value,
                    )
                )
            ).all()
        )
        if actual != required_calls:
            raise InvalidStateTransitionError("typed_answer_audited_call_lineage_incomplete")
        return counts

    async def assert_rag_answer_terminals(self, run_id: str, method: str) -> dict[str, int]:
        """Validate strict answer/refusal coverage without N+1 model-call reads."""
        from pmmc.infrastructure.database.query_executor_models import StructuredAnswerOutputModel
        from pmmc.runtime.answering import validate_abstention_fact
        from pmmc.schemas.answer_output import StructuredAnswerFact

        selected = set(
            (
                await self.session.scalars(
                    select(RunQueryModel.query_id).where(RunQueryModel.run_id == run_id)
                )
            ).all()
        )
        answers = (
            await self.session.execute(
                select(
                    FinalAnswerModel.query_id,
                    FinalAnswerModel.method,
                ).where(
                    FinalAnswerModel.run_id == run_id,
                    FinalAnswerModel.answer_variant == "primary",
                )
            )
        ).all()
        # Outer joins retain malformed lineage so it fails the gate, not disappears.
        rows = (
            await self.session.execute(
                select(
                    StructuredAnswerOutputModel,
                    RuntimeModelCallModel,
                    ContextAssemblyModel,
                )
                .outerjoin(
                    RuntimeModelCallModel,
                    RuntimeModelCallModel.model_call_id
                    == StructuredAnswerOutputModel.model_call_id,
                )
                .outerjoin(
                    ContextAssemblyModel,
                    ContextAssemblyModel.context_assembly_id
                    == RuntimeModelCallModel.context_assembly_id,
                )
                .where(
                    StructuredAnswerOutputModel.run_id == run_id,
                    StructuredAnswerOutputModel.answer_variant == "primary",
                    StructuredAnswerOutputModel.supported.is_(False),
                    StructuredAnswerOutputModel.parse_status == "accepted",
                )
            )
        ).all()
        answered = Counter(query_id for query_id, _ in answers)
        abstained: Counter[str] = Counter()
        invalid = sum(answer_method != method for _, answer_method in answers)
        for row, call, context in rows:
            abstained[row.query_id] += 1
            try:
                fact = StructuredAnswerFact.model_validate(_model_mapping(row))
                validate_abstention_fact(fact)
                if (
                    fact.metadata.get("method") != method
                    or "source_run_id" in fact.metadata
                    or call is None
                    or call.run_id != run_id
                    or call.query_id != fact.query_id
                    or call.stage != AppStage.ANSWER.value
                    or call.status != TraceStatus.SUCCESS.value
                    or context is None
                    or context.run_id != run_id
                    or context.query_id != fact.query_id
                    or context.method != method
                    or context.retrieval_batch_id is None
                ):
                    raise ValueError("invalid baseline abstention lineage")
            except ValueError:
                invalid += 1
        counts = {
            "expected_count": len(selected),
            "answered_count": sum(answered.values()),
            "abstained_count": sum(abstained.values()),
            "missing_count": sum(answered[q] + abstained[q] == 0 for q in selected),
            "duplicate_or_overlap_count": sum(answered[q] + abstained[q] > 1 for q in selected),
            "unassigned_count": len((answered.keys() | abstained.keys()) - selected),
            "invalid_count": invalid,
        }
        rejected = any(
            counts[key]
            for key in (
                "missing_count",
                "duplicate_or_overlap_count",
                "unassigned_count",
                "invalid_count",
            )
        )
        print(
            "RAG_ANSWER_TERMINAL_VERIFY "
            + json.dumps(
                {
                    **counts,
                    "state": "rejected" if rejected else "ready",
                    "policy": "independent-answer-abstention-v1",
                    "abstention_is_answer_success": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        if rejected:
            raise InvalidStateTransitionError("rag_answer_terminal_integrity_failed")
        return counts

    async def _assert_compiler_completion_integrity(
        self,
        run_id: str,
        selected_count: int,
    ) -> None:
        from pmmc.infrastructure.database.compiler_models import (
            CompilerRunModel,
            CompilerUnitMemoryModel,
            CompilerUnitModel,
            CompilerUnitStageResultModel,
        )
        from pmmc.infrastructure.database.compiler_repositories import (
            QUESTIONER_STAGE_ACCOUNTING_SCHEMA_VERSIONS,
            compiler_questioner_stage_expected_counts,
            compiler_unit_fact_counts,
            compiler_v04_stage_expected_counts,
        )

        compiler = await self.session.get(CompilerRunModel, run_id)
        if compiler is None:
            return
        if selected_count != 0:
            raise InvalidStateTransitionError(
                "compiler runs must keep the benchmark query denominator at zero"
            )
        units = list(
            (
                await self.session.execute(
                    select(CompilerUnitModel)
                    .where(CompilerUnitModel.run_id == run_id)
                    .order_by(CompilerUnitModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        expected_ordinals = list(range(compiler.unit_count))
        if (
            len(units) != compiler.unit_count
            or [value.ordinal for value in units] != expected_ordinals
        ):
            raise InvalidStateTransitionError(
                "compiler unit denominator is incomplete or unordered"
            )
        required_stages = {
            "questions": {"questioner"},
            "programs": {"questioner", "planner", "dry_run"},
            "verified_qbank": {
                "questioner",
                "planner",
                "dry_run",
                "doubter",
                "decision",
            },
        }[compiler.target_stage]
        for unit in units:
            members = list(
                (
                    await self.session.execute(
                        select(CompilerUnitMemoryModel)
                        .where(CompilerUnitMemoryModel.compiler_unit_id == unit.compiler_unit_id)
                        .order_by(CompilerUnitMemoryModel.ordinal)
                    )
                )
                .scalars()
                .all()
            )
            member_ordinals = [value.ordinal for value in members]
            if (
                not members
                or member_ordinals != list(range(len(members)))
                or not any(value.role == "anchor" for value in members)
            ):
                raise InvalidStateTransitionError(
                    f"compiler unit membership is incomplete: {unit.compiler_unit_id}"
                )
            compile_jobs = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(ExperimentJobModel)
                    .where(
                        ExperimentJobModel.run_id == run_id,
                        ExperimentJobModel.compiler_unit_id == unit.compiler_unit_id,
                        ExperimentJobModel.stage == AppStage.COMPILE.value,
                        ExperimentJobModel.status == JobStatus.SUCCEEDED.value,
                    )
                )
                or 0
            )
            if compile_jobs != 1:
                raise InvalidStateTransitionError(
                    f"compiler unit requires one succeeded job: {unit.compiler_unit_id}"
                )
            succeeded_results = list(
                (
                    await self.session.scalars(
                        select(CompilerUnitStageResultModel).where(
                            CompilerUnitStageResultModel.compiler_unit_id == unit.compiler_unit_id,
                            CompilerUnitStageResultModel.status == TraceStatus.SUCCESS.value,
                        )
                    )
                ).all()
            )
            succeeded_stages = {result.stage for result in succeeded_results}
            if not required_stages.issubset(succeeded_stages):
                raise InvalidStateTransitionError(
                    f"compiler target stages are incomplete: {unit.compiler_unit_id}"
                )
            fact_counts = await compiler_unit_fact_counts(
                self.session,
                run_id=run_id,
                compiler_unit_id=unit.compiler_unit_id,
            )
            for result in succeeded_results:
                if result.stage not in required_stages:
                    continue
                if (
                    result.stage == CompilationStage.QUESTIONER.value
                    and compiler.compiler_schema_version
                    in QUESTIONER_STAGE_ACCOUNTING_SCHEMA_VERSIONS
                ):
                    expected_counts = await compiler_questioner_stage_expected_counts(
                        self.session,
                        run_id=run_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        compiler_schema_version=compiler.compiler_schema_version,
                    )
                    if expected_counts is None:
                        raise InvalidStateTransitionError(
                            "compiler unit requires exactly one successful Questioner "
                            f"generation: {unit.compiler_unit_id}"
                        )
                    actual_counts = (
                        result.produced_count,
                        result.accepted_count,
                        result.rejected_count,
                    )
                    if actual_counts != expected_counts:
                        raise InvalidStateTransitionError(
                            "Questioner stage accounting differs from generation/gate "
                            f"facts: {unit.compiler_unit_id}; "
                            f"actual={actual_counts}, expected={expected_counts}"
                        )
                elif result.stage in {
                    CompilationStage.PLANNER.value,
                    CompilationStage.DRY_RUN.value,
                } and compiler.compiler_schema_version in {
                    "0.4.0",
                    "0.5.0",
                    "0.6.0",
                    "0.7.0",
                    "0.8.0",
                    "0.9.0",
                }:
                    expected_stage_counts = await compiler_v04_stage_expected_counts(
                        self.session,
                        run_id=run_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        stage=result.stage,
                        fact_counts=fact_counts,
                    )
                    actual_counts = (
                        result.produced_count,
                        result.accepted_count,
                        result.rejected_count,
                    )
                    if expected_stage_counts is None or actual_counts != expected_stage_counts:
                        raise InvalidStateTransitionError(
                            f"{result.stage} stage accounting differs from atomic facts: "
                            f"{unit.compiler_unit_id}; "
                            f"actual={actual_counts}, expected={expected_stage_counts}"
                        )
                elif result.produced_count != fact_counts[result.stage]:
                    raise InvalidStateTransitionError(
                        "compiler stage counts differ from persisted facts: "
                        f"{unit.compiler_unit_id}/{result.stage}"
                    )
            if compiler.compiler_schema_version in {
                "0.3.0",
                "0.4.0",
                "0.5.0",
                "0.6.0",
                "0.7.0",
                "0.8.0",
            }:
                from pmmc.infrastructure.database.questioner_repositories import (
                    questioner_v3_unit_issues,
                )

                v3_issues = await questioner_v3_unit_issues(
                    self.session,
                    run_id=run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                )
                if v3_issues:
                    raise InvalidStateTransitionError(
                        f"Questioner v3 unit is incomplete: {unit.compiler_unit_id}: {v3_issues[0]}"
                    )


class PostgresJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, job: ExperimentJob) -> ExperimentJob:
        value = await _insert_idempotent(
            self.session, ExperimentJobModel, job_to_row(job), job.job_id
        )
        return job_from_row(_model_mapping(value))

    async def get(self, job_id: str) -> ExperimentJob | None:
        value = await self.session.get(ExperimentJobModel, job_id)
        return job_from_row(_model_mapping(value)) if value is not None else None

    async def status_counts(
        self, run_id: str, *, stage: AppStage | None = None
    ) -> dict[JobStatus, int]:
        statement = select(ExperimentJobModel.status, func.count()).where(
            ExperimentJobModel.run_id == run_id
        )
        if stage is not None:
            statement = statement.where(ExperimentJobModel.stage == stage.value)
        rows = await self.session.execute(statement.group_by(ExperimentJobModel.status))
        return {JobStatus(status): int(count) for status, count in rows.all()}

    async def claim(
        self,
        worker_id: str,
        lease_seconds: int,
        *,
        run_id: str | None = None,
        stage: AppStage | None = None,
    ) -> ExperimentJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        exhausted_conditions = [
            RunStateModel.status.in_([RunStatus.CREATED.value, RunStatus.RUNNING.value]),
            ExperimentJobModel.status == JobStatus.RUNNING.value,
            ExperimentJobModel.lease_expires_at < now,
            ExperimentJobModel.attempt_count >= ExperimentJobModel.max_attempts,
        ]
        candidate_conditions = [
            RunStateModel.status.in_([RunStatus.CREATED.value, RunStatus.RUNNING.value]),
            ExperimentJobModel.available_at <= now,
            ExperimentJobModel.attempt_count < ExperimentJobModel.max_attempts,
            or_(
                ExperimentJobModel.status == JobStatus.PENDING.value,
                and_(
                    ExperimentJobModel.status == JobStatus.RUNNING.value,
                    ExperimentJobModel.lease_expires_at < now,
                ),
            ),
        ]
        if run_id is not None:
            exhausted_conditions.append(ExperimentJobModel.run_id == run_id)
            candidate_conditions.append(ExperimentJobModel.run_id == run_id)
        if stage is not None:
            exhausted_conditions.append(ExperimentJobModel.stage == stage.value)
            candidate_conditions.append(ExperimentJobModel.stage == stage.value)

        exhausted = (
            (
                await self.session.execute(
                    select(ExperimentJobModel)
                    .join(RunStateModel, RunStateModel.run_id == ExperimentJobModel.run_id)
                    .where(*exhausted_conditions)
                    .with_for_update(skip_locked=True, of=ExperimentJobModel)
                )
            )
            .scalars()
            .all()
        )
        for expired in exhausted:
            expired.status = JobStatus.FAILED.value
            expired.claimed_by = None
            expired.lease_expires_at = None
            expired.finished_at = now
            expired.error_type = "LeaseExpired"
            expired.error_summary = "job lease expired after the final attempt"
            await self._fail_expired_attempt(expired, now)

        candidate = await self.session.scalar(
            select(ExperimentJobModel)
            .join(RunStateModel, RunStateModel.run_id == ExperimentJobModel.run_id)
            .where(*candidate_conditions)
            .order_by(
                ExperimentJobModel.priority.desc(),
                ExperimentJobModel.available_at,
                ExperimentJobModel.job_id,
            )
            .with_for_update(skip_locked=True, of=ExperimentJobModel)
            .limit(1)
        )
        if candidate is None:
            return None
        if candidate.status == JobStatus.RUNNING.value:
            await self._fail_expired_attempt(candidate, now)
        attempt_number = candidate.attempt_count + 1
        lease = now + timedelta(seconds=lease_seconds)
        candidate.status = JobStatus.RUNNING.value
        candidate.claimed_by = worker_id
        candidate.lease_expires_at = lease
        candidate.attempt_count = attempt_number
        candidate.started_at = candidate.started_at or now
        candidate.finished_at = None
        candidate.error_type = None
        candidate.error_summary = None
        candidate.error_artifact_id = None
        attempt = JobAttempt(
            attempt_id=f"{candidate.job_id}:attempt:{attempt_number}",
            job_id=candidate.job_id,
            attempt_number=attempt_number,
            worker_id=worker_id,
        )
        await _insert_idempotent(
            self.session,
            JobAttemptModel,
            job_attempt_to_row(attempt),
            attempt.attempt_id,
        )
        await self.session.flush()
        return job_from_row(_model_mapping(candidate))

    async def _fail_expired_attempt(
        self,
        job: ExperimentJobModel,
        finished_at: datetime,
    ) -> None:
        await self.session.execute(
            update(JobAttemptModel)
            .where(
                JobAttemptModel.job_id == job.job_id,
                JobAttemptModel.attempt_number == job.attempt_count,
                JobAttemptModel.status == JobStatus.RUNNING.value,
            )
            .values(
                status=JobStatus.FAILED.value,
                finished_at=finished_at,
                error_type="LeaseExpired",
                error_summary="worker lease expired before completion",
            )
        )

    async def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        lease_seconds: int,
        *,
        expected_attempt_count: int | None = None,
    ) -> ExperimentJob:
        """Renew the claimed generation; an omitted fence is legacy compatibility only."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if expected_attempt_count is not None and expected_attempt_count < 1:
            raise ValueError("expected_attempt_count must be positive")
        now = utc_now()
        conditions = [
            ExperimentJobModel.job_id == job_id,
            ExperimentJobModel.status == JobStatus.RUNNING.value,
            ExperimentJobModel.claimed_by == worker_id,
            ExperimentJobModel.lease_expires_at > now,
        ]
        if expected_attempt_count is not None:
            conditions.append(ExperimentJobModel.attempt_count == expected_attempt_count)
        result = await self.session.execute(
            update(ExperimentJobModel)
            .where(*conditions)
            .values(lease_expires_at=now + timedelta(seconds=lease_seconds))
            .returning(ExperimentJobModel)
            .execution_options(populate_existing=True)
        )
        value = result.scalar_one_or_none()
        if value is None:
            raise InvalidStateTransitionError(
                "job heartbeat requires the active lease owner and claimed attempt"
            )
        return job_from_row(_model_mapping(value))

    async def finish(
        self,
        job_id: str,
        worker_id: str,
        *,
        success: bool,
        error_type: str | None = None,
        error_summary: str | None = None,
        error_artifact_id: str | None = None,
        retry_delay_seconds: float = 0,
        retryable: bool = True,
        expected_attempt_count: int | None = None,
    ) -> ExperimentJob:
        """Fence under a row lock held through the caller's scientific-fact transaction."""
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if expected_attempt_count is not None and expected_attempt_count < 1:
            raise ValueError("expected_attempt_count must be positive")
        value = await self.session.scalar(
            select(ExperimentJobModel)
            .where(ExperimentJobModel.job_id == job_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        now = utc_now()
        if (
            value is None
            or value.status != JobStatus.RUNNING.value
            or value.claimed_by != worker_id
            or value.lease_expires_at is None
            or value.lease_expires_at <= now
            or (
                expected_attempt_count is not None
                and value.attempt_count != expected_attempt_count
            )
        ):
            raise InvalidStateTransitionError(
                "only the active lease owner and claimed attempt can finish a job"
            )
        attempt_id = f"{job_id}:attempt:{value.attempt_count}"
        if success:
            value.status = JobStatus.SUCCEEDED.value
            value.finished_at = now
            attempt_status = JobStatus.SUCCEEDED
        else:
            if error_type is None:
                raise ValueError("failed jobs require error_type")
            attempt_status = JobStatus.FAILED
            if retryable and value.attempt_count < value.max_attempts:
                value.status = JobStatus.PENDING.value
                value.available_at = now + timedelta(seconds=retry_delay_seconds)
                value.finished_at = None
            else:
                value.status = JobStatus.FAILED.value
                value.finished_at = now
            value.error_type = error_type
            value.error_summary = error_summary
            value.error_artifact_id = error_artifact_id
        value.claimed_by = None
        value.lease_expires_at = None
        await self.session.execute(
            update(JobAttemptModel)
            .where(JobAttemptModel.attempt_id == attempt_id)
            .values(
                status=attempt_status.value,
                finished_at=now,
                error_type=error_type,
                error_summary=error_summary,
                error_artifact_id=error_artifact_id,
            )
        )
        await self.session.flush()
        return job_from_row(_model_mapping(value))


class PostgresArtifactRepository:
    def __init__(self, session: AsyncSession, scope: DataScope) -> None:
        self.session = session
        self.scope = scope
        self.model = RuntimeArtifactModel if scope is DataScope.RUNTIME else EvaluationArtifactModel

    async def add(self, artifact: ExperimentArtifact) -> ExperimentArtifact:
        if artifact.data_scope is not self.scope:
            raise ValueError("artifact scope does not match repository scope")
        value = await _insert_idempotent(
            self.session, self.model, artifact_to_row(artifact), artifact.artifact_id
        )
        return artifact_from_row(_model_mapping(value), self.scope)

    async def get(self, artifact_id: str) -> ExperimentArtifact | None:
        value = await self.session.get(self.model, artifact_id)
        if value is None:
            return None
        return artifact_from_row(_model_mapping(value), self.scope)

    async def retry_failed(self, artifact_id: str) -> ExperimentArtifact:
        result = await self.session.execute(
            update(self.model)
            .where(
                self.model.artifact_id == artifact_id,
                self.model.state == ArtifactState.FAILED.value,
            )
            .values(state=ArtifactState.PENDING.value, error_type=None)
            .returning(self.model)
        )
        value = result.scalar_one_or_none()
        if value is None:
            raise InvalidStateTransitionError("only failed artifacts can be retried")
        return artifact_from_row(_model_mapping(value), self.scope)

    async def mark_ready(
        self, artifact_id: str, *, storage_uri: str, size_bytes: int
    ) -> ExperimentArtifact:
        result = await self.session.execute(
            update(self.model)
            .where(
                self.model.artifact_id == artifact_id,
                self.model.state == ArtifactState.PENDING.value,
            )
            .values(
                state=ArtifactState.READY.value,
                storage_uri=storage_uri,
                size_bytes=size_bytes,
                ready_at=utc_now(),
                error_type=None,
            )
            .returning(self.model)
        )
        value = result.scalar_one_or_none()
        if value is None:
            current = cast(Any, await self.session.get(self.model, artifact_id))
            if current is None or current.state != ArtifactState.READY.value:
                raise InvalidStateTransitionError("artifact cannot transition to ready")
            if current.storage_uri != storage_uri or current.size_bytes != size_bytes:
                raise PersistenceConflictError(
                    "ready artifact already exists with different storage metadata"
                )
            value = current
        return artifact_from_row(_model_mapping(value), self.scope)

    async def mark_failed(self, artifact_id: str, error_type: str) -> ExperimentArtifact:
        result = await self.session.execute(
            update(self.model)
            .where(
                self.model.artifact_id == artifact_id,
                self.model.state == ArtifactState.PENDING.value,
            )
            .values(state=ArtifactState.FAILED.value, error_type=error_type)
            .returning(self.model)
        )
        value = result.scalar_one_or_none()
        if value is None:
            current = cast(Any, await self.session.get(self.model, artifact_id))
            if current is None or current.state != ArtifactState.FAILED.value:
                raise InvalidStateTransitionError("artifact cannot transition to failed")
            if current.error_type != error_type:
                raise PersistenceConflictError(
                    "failed artifact already exists with a different error type"
                )
            value = current
        return artifact_from_row(_model_mapping(value), self.scope)


class PostgresEmbeddingRepository:
    def __init__(self, session: AsyncSession, batch_size: int = 1000) -> None:
        if batch_size <= 0:
            raise ValueError("embedding write batch_size must be positive")
        self.session = session
        self.batch_size = batch_size

    async def add_collection(self, value: EmbeddingCollection) -> EmbeddingCollection:
        if value.status is not EmbeddingCollectionStatus.BUILDING:
            raise InvalidStateTransitionError("new embedding collections must start building")
        if value.corpus_id is not None:
            corpus = await self.session.get(RetrievalCorpusModel, value.corpus_id)
            if corpus is None or corpus.status != "ready":
                raise InvalidStateTransitionError(
                    "embedding collections require a ready retrieval corpus"
                )
            if corpus.snapshot_id != value.snapshot_id:
                raise ValueError("embedding collection and corpus snapshots must match")
        stored = await _insert_idempotent(
            self.session,
            EmbeddingCollectionModel,
            embedding_collection_to_row(value),
            value.collection_id,
        )
        return EmbeddingCollection.model_validate(_model_mapping(stored))

    async def add_records(self, records: list[EmbeddingRecord]) -> None:
        collection_ids = {record.collection_id for record in records}
        collections = {
            value.collection_id: value
            for value in (
                (
                    await self.session.execute(
                        select(EmbeddingCollectionModel).where(
                            EmbeddingCollectionModel.collection_id.in_(collection_ids)
                        )
                    )
                )
                .scalars()
                .all()
            )
        }
        missing = collection_ids - set(collections)
        if missing:
            raise KeyError(f"embedding collection does not exist: {sorted(missing)[0]}")
        for collection_id in collection_ids:
            group = [record for record in records if record.collection_id == collection_id]
            collection = collections[collection_id]
            document_sources = {
                EmbeddingSourceType.RETRIEVAL_DOCUMENT_TEXT_CAPTION.value,
                EmbeddingSourceType.RETRIEVAL_DOCUMENT_TEXT.value,
                EmbeddingSourceType.RETRIEVAL_DOCUMENT_IMAGE.value,
                EmbeddingSourceType.RETRIEVAL_DOCUMENT_MULTIMODAL.value,
                EmbeddingSourceType.RETRIEVAL_DOCUMENT_CONTRASTIVE.value,
            }
            if collection.source_type in document_sources:
                document_ids = {
                    record.retrieval_document_id
                    for record in group
                    if record.retrieval_document_id is not None
                }
                if len(document_ids) != len(group):
                    raise ValueError(
                        "retrieval-document embedding records require document lineage"
                    )
                documents = {
                    document.document_id: document
                    for document in (
                        (
                            await self.session.execute(
                                select(RetrievalDocumentModel).where(
                                    RetrievalDocumentModel.document_id.in_(document_ids)
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                }
                for record in group:
                    document = documents.get(record.retrieval_document_id or "")
                    if document is None or document.corpus_id != collection.corpus_id:
                        raise ValueError(
                            "embedding record document does not belong to collection corpus"
                        )
            elif any(record.retrieval_document_id is not None for record in group):
                raise ValueError(
                    "non-document embedding collections cannot link retrieval documents"
                )
            if collection.status == EmbeddingCollectionStatus.FAILED.value:
                raise InvalidStateTransitionError("failed embedding collections are immutable")
            if collection.status == EmbeddingCollectionStatus.READY.value:
                for record in group:
                    stored = await self.session.get(EmbeddingModel, record.embedding_id)
                    if stored is None:
                        stored = await self.session.scalar(
                            select(EmbeddingModel).where(
                                EmbeddingModel.collection_id == record.collection_id,
                                EmbeddingModel.source_id == record.source_id,
                            )
                        )
                    if stored is None:
                        raise PersistenceConflictError(
                            "ready embedding collection cannot accept new records"
                        )
                    row = embedding_to_row(record)
                    actual = _model_mapping(stored)
                    if any(
                        _comparable(actual[key]) != _comparable(value) for key, value in row.items()
                    ):
                        raise PersistenceConflictError(
                            "ready embedding record replay has different content"
                        )
                continue
            for start in range(0, len(group), self.batch_size):
                chunk = group[start : start + self.batch_size]
                await _insert_many_idempotent(
                    self.session,
                    EmbeddingModel,
                    [embedding_to_row(record) for record in chunk],
                    identity_key="embedding_id",
                    alternate_keys=("collection_id", "source_id"),
                )

    async def search_exact(
        self,
        collection_id: str,
        query_vector: list[float],
        *,
        corpus_id: str,
        top_k: int,
        conversation_id: str | None = None,
        cutoff_session_id: str | None = None,
    ) -> list[EmbeddingSearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if cutoff_session_id is not None and conversation_id is None:
            raise ValueError("cutoff_session_id requires conversation_id")
        if not query_vector or any(not isinstance(value, (int, float)) for value in query_vector):
            raise ValueError("query_vector must contain finite numeric values")
        if any(not math.isfinite(float(value)) for value in query_vector):
            raise ValueError("query_vector must contain finite numeric values")
        collection = await self.session.get(EmbeddingCollectionModel, collection_id)
        if collection is None or collection.status != EmbeddingCollectionStatus.READY.value:
            raise KeyError(f"ready embedding collection does not exist: {collection_id}")
        if collection.corpus_id != corpus_id:
            raise ValueError("embedding collection does not belong to requested corpus")
        if len(query_vector) != collection.dimension:
            raise ValueError("query vector dimension does not match collection")

        if collection.distance_metric == "cosine":
            order_expression = EmbeddingModel.embedding.cosine_distance(query_vector)
            raw_expression = 1.0 - order_expression
            score_type = "cosine_similarity"
            direction = ScoreDirection.HIGHER_IS_BETTER
        elif collection.distance_metric == "inner_product":
            order_expression = EmbeddingModel.embedding.max_inner_product(query_vector)
            raw_expression = -order_expression
            score_type = "inner_product"
            direction = ScoreDirection.HIGHER_IS_BETTER
        else:
            order_expression = EmbeddingModel.embedding.l2_distance(query_vector)
            raw_expression = order_expression
            score_type = "l2_distance"
            direction = ScoreDirection.LOWER_IS_BETTER

        statement = (
            select(
                EmbeddingModel.embedding_id,
                EmbeddingModel.collection_id,
                EmbeddingModel.retrieval_document_id,
                raw_expression.label("raw_score"),
            )
            .join(
                RetrievalDocumentModel,
                RetrievalDocumentModel.document_id == EmbeddingModel.retrieval_document_id,
            )
            .where(
                EmbeddingModel.collection_id == collection_id,
                RetrievalDocumentModel.corpus_id == corpus_id,
            )
        )
        if conversation_id is not None:
            statement = statement.where(RetrievalDocumentModel.conversation_id == conversation_id)
        if cutoff_session_id is not None:
            cutoff = await self.session.scalar(
                select(BenchmarkSessionModel).where(
                    BenchmarkSessionModel.snapshot_id == collection.snapshot_id,
                    BenchmarkSessionModel.conversation_id == conversation_id,
                    BenchmarkSessionModel.session_id == cutoff_session_id,
                )
            )
            if cutoff is None:
                raise KeyError(f"cutoff session does not exist: {cutoff_session_id}")
            if conversation_id is not None and cutoff.conversation_id != conversation_id:
                raise ValueError("cutoff session does not belong to requested conversation")
            available_session = aliased(BenchmarkSessionModel)
            statement = statement.join(
                available_session,
                and_(
                    available_session.snapshot_id == RetrievalDocumentModel.snapshot_id,
                    available_session.session_id == RetrievalDocumentModel.available_after_session,
                    available_session.conversation_id == RetrievalDocumentModel.conversation_id,
                ),
            ).where(
                RetrievalDocumentModel.conversation_id == cutoff.conversation_id,
                available_session.session_order <= cutoff.session_order,
            )
        rows = (
            await self.session.execute(
                statement.order_by(
                    order_expression.asc(),
                    RetrievalDocumentModel.document_id,
                ).limit(top_k)
            )
        ).all()
        return [
            EmbeddingSearchHit(
                embedding_id=row.embedding_id,
                collection_id=row.collection_id,
                retrieval_document_id=row.retrieval_document_id,
                rank=rank,
                score=RankingScore(
                    score_type=score_type,
                    raw_value=float(row.raw_score),
                    direction=direction,
                ),
                distance_metric=collection.distance_metric,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def search_windows_exact(
        self,
        collection_id: str,
        query_vector: list[float],
        *,
        corpus_id: str,
        top_k: int,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> list[WindowSearchHit]:
        """Exact-scan eligible documents and aggregate them to unique memory windows."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query_vector or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in query_vector
        ):
            raise ValueError("query_vector must contain finite numeric values")
        collection = await self.session.get(EmbeddingCollectionModel, collection_id)
        if collection is None or collection.status != EmbeddingCollectionStatus.READY.value:
            raise KeyError(f"ready embedding collection does not exist: {collection_id}")
        if collection.corpus_id != corpus_id:
            raise ValueError("embedding collection does not belong to requested corpus")
        if len(query_vector) != collection.dimension:
            raise ValueError("query vector dimension does not match collection")

        if collection.distance_metric == "cosine":
            distance = EmbeddingModel.embedding.cosine_distance(query_vector)
            raw_expression = 1.0 - distance
            score_type = "cosine_similarity"
            direction = ScoreDirection.HIGHER_IS_BETTER
        elif collection.distance_metric == "inner_product":
            distance = EmbeddingModel.embedding.max_inner_product(query_vector)
            raw_expression = -distance
            score_type = "inner_product"
            direction = ScoreDirection.HIGHER_IS_BETTER
        else:
            distance = EmbeddingModel.embedding.l2_distance(query_vector)
            raw_expression = distance
            score_type = "l2_distance"
            direction = ScoreDirection.LOWER_IS_BETTER

        cutoff = await self.session.scalar(
            select(BenchmarkSessionModel).where(
                BenchmarkSessionModel.snapshot_id == collection.snapshot_id,
                BenchmarkSessionModel.conversation_id == conversation_id,
                BenchmarkSessionModel.session_id == cutoff_session_id,
            )
        )
        if cutoff is None:
            raise KeyError(f"cutoff session does not exist: {cutoff_session_id}")
        available_session = aliased(BenchmarkSessionModel)
        statement = (
            select(
                EmbeddingModel.embedding_id,
                RetrievalDocumentModel.document_id,
                RetrievalDocumentModel.parent_window_id,
                RetrievalDocumentModel.representation,
                raw_expression.label("raw_score"),
            )
            .join(
                RetrievalDocumentModel,
                RetrievalDocumentModel.document_id == EmbeddingModel.retrieval_document_id,
            )
            .join(
                available_session,
                and_(
                    available_session.snapshot_id == RetrievalDocumentModel.snapshot_id,
                    available_session.conversation_id == RetrievalDocumentModel.conversation_id,
                    available_session.session_id == RetrievalDocumentModel.available_after_session,
                ),
            )
            .where(
                EmbeddingModel.collection_id == collection_id,
                RetrievalDocumentModel.corpus_id == corpus_id,
                RetrievalDocumentModel.conversation_id == conversation_id,
                available_session.session_order <= cutoff.session_order,
            )
        )
        rows = (await self.session.execute(statement)).all()
        by_window: dict[str, list[RetrievalScoreComponent]] = {}
        for row in rows:
            by_window.setdefault(row.parent_window_id, []).append(
                RetrievalScoreComponent(
                    retrieval_document_id=row.document_id,
                    embedding_collection_id=collection_id,
                    representation=row.representation,
                    score=RankingScore(
                        score_type=score_type,
                        raw_value=float(row.raw_score),
                        direction=direction,
                    ),
                    metadata={"embedding_id": row.embedding_id},
                )
            )

        reverse = direction is ScoreDirection.HIGHER_IS_BETTER

        def component_order(item: RetrievalScoreComponent) -> tuple[float, str]:
            value = item.score.raw_value
            return ((-value if reverse else value), item.retrieval_document_id)

        aggregated: list[tuple[str, RetrievalScoreComponent, list[RetrievalScoreComponent]]] = []
        for window_id, components in by_window.items():
            ordered = sorted(components, key=component_order)
            aggregated.append((window_id, ordered[0], ordered))
        aggregated.sort(
            key=lambda item: (
                -item[1].score.raw_value if reverse else item[1].score.raw_value,
                item[0],
            )
        )
        return [
            WindowSearchHit(
                window_id=window_id,
                rank=rank,
                score=winner.score,
                winning_document_id=winner.retrieval_document_id,
                components=components,
            )
            for rank, (window_id, winner, components) in enumerate(aggregated[:top_k], start=1)
        ]

    async def search_qbank_exact(
        self,
        collection_id: str,
        query_vector: list[float],
        *,
        compiler_run_id: str,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
        question_budget: int,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Exact-scan one frozen QBank channel and aggregate paraphrases by parent."""

        from pmmc.infrastructure.database.compiler_models import (
            QBankParaphraseModel,
            QBankQuestionModel,
        )
        from pmmc.infrastructure.database.questioner_models import (
            QuestionerCandidateGateResultModel,
        )

        if question_budget <= 0 or top_k <= 0:
            raise ValueError("QBank search limits must be positive")
        if not query_vector or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in query_vector
        ):
            raise ValueError("query_vector must contain finite numeric values")
        collection = await self.session.get(EmbeddingCollectionModel, collection_id)
        if collection is None or collection.status != EmbeddingCollectionStatus.READY.value:
            raise KeyError(f"ready embedding collection does not exist: {collection_id}")
        if (
            collection.snapshot_id != snapshot_id
            or collection.run_id != compiler_run_id
            or collection.source_type
            not in {
                EmbeddingSourceType.QBANK_QUESTION.value,
                EmbeddingSourceType.QBANK_PARAPHRASE.value,
            }
        ):
            raise ValueError("embedding collection is not part of the requested QBank")
        if collection.distance_metric != DistanceMetric.COSINE.value:
            raise ValueError("QBank-v1 requires cosine embedding collections")
        if len(query_vector) != collection.dimension:
            raise ValueError("query vector dimension does not match collection")

        cutoff_order = await self.session.scalar(
            select(BenchmarkSessionModel.session_order).where(
                BenchmarkSessionModel.snapshot_id == snapshot_id,
                BenchmarkSessionModel.conversation_id == conversation_id,
                BenchmarkSessionModel.session_id == cutoff_session_id,
            )
        )
        if cutoff_order is None:
            raise KeyError(f"cutoff session does not exist: {cutoff_session_id}")
        available = aliased(BenchmarkSessionModel)
        distance = EmbeddingModel.embedding.cosine_distance(query_vector)
        score = (1.0 - distance).label("raw_score")
        if collection.source_type == EmbeddingSourceType.QBANK_QUESTION.value:
            statement = select(QBankQuestionModel.qbank_id, score).join(
                EmbeddingModel,
                EmbeddingModel.qbank_id == QBankQuestionModel.qbank_id,
            )
        else:
            statement = (
                select(QBankParaphraseModel.qbank_id, score)
                .join(
                    EmbeddingModel,
                    EmbeddingModel.paraphrase_id == QBankParaphraseModel.paraphrase_id,
                )
                .join(
                    QBankQuestionModel,
                    QBankQuestionModel.qbank_id == QBankParaphraseModel.qbank_id,
                )
            )
        statement = (
            statement.join(
                QuestionerCandidateGateResultModel,
                and_(
                    QuestionerCandidateGateResultModel.qbank_id == QBankQuestionModel.qbank_id,
                    QuestionerCandidateGateResultModel.run_id == QBankQuestionModel.run_id,
                ),
            )
            .join(
                available,
                and_(
                    available.snapshot_id == QBankQuestionModel.snapshot_id,
                    available.conversation_id == QBankQuestionModel.conversation_id,
                    available.session_id == QBankQuestionModel.created_after_session,
                ),
            )
            .where(
                EmbeddingModel.collection_id == collection_id,
                QBankQuestionModel.run_id == compiler_run_id,
                QBankQuestionModel.snapshot_id == snapshot_id,
                QBankQuestionModel.conversation_id == conversation_id,
                QBankQuestionModel.generation_rank <= question_budget,
                QuestionerCandidateGateResultModel.status
                == CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value,
                available.session_order <= cutoff_order,
            )
        )
        rows = (await self.session.execute(statement)).all()
        by_question: dict[str, float] = {}
        for row in rows:
            raw = float(row.raw_score)
            previous = by_question.get(str(row.qbank_id))
            if previous is None or raw > previous:
                by_question[str(row.qbank_id)] = raw
        ranked = sorted(by_question.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    async def get_collection(self, collection_id: str) -> EmbeddingCollection | None:
        value = await self.session.get(EmbeddingCollectionModel, collection_id)
        return embedding_collection_from_row(_model_mapping(value)) if value is not None else None

    async def get_record(self, embedding_id: str) -> EmbeddingRecord | None:
        value = await self.session.get(EmbeddingModel, embedding_id)
        return embedding_from_row(_model_mapping(value)) if value is not None else None

    async def get_records(self, embedding_ids: list[str]) -> dict[str, EmbeddingRecord]:
        if not embedding_ids:
            return {}
        values = (
            (
                await self.session.execute(
                    select(EmbeddingModel).where(EmbeddingModel.embedding_id.in_(embedding_ids))
                )
            )
            .scalars()
            .all()
        )
        return {value.embedding_id: embedding_from_row(_model_mapping(value)) for value in values}

    async def count_records(self, collection_id: str) -> int:
        return int(
            await self.session.scalar(
                select(func.count())
                .select_from(EmbeddingModel)
                .where(EmbeddingModel.collection_id == collection_id)
            )
            or 0
        )

    async def get_query_records(self, collection_id: str, query_id: str) -> list[EmbeddingRecord]:
        values = (
            (
                await self.session.execute(
                    select(EmbeddingModel)
                    .where(
                        EmbeddingModel.collection_id == collection_id,
                        EmbeddingModel.metadata_json["query_id"].astext == query_id,
                    )
                    .order_by(EmbeddingModel.source_id)
                )
            )
            .scalars()
            .all()
        )
        return [embedding_from_row(_model_mapping(value)) for value in values]

    async def mark_collection_ready(
        self,
        collection_id: str,
        *,
        index_type: str | None = None,
        index_params: dict[str, Any] | None = None,
    ) -> EmbeddingCollection:
        result = await self.session.execute(
            update(EmbeddingCollectionModel)
            .where(
                EmbeddingCollectionModel.collection_id == collection_id,
                EmbeddingCollectionModel.status == EmbeddingCollectionStatus.BUILDING.value,
            )
            .values(
                status=EmbeddingCollectionStatus.READY.value,
                index_type=index_type,
                index_params=index_params or {},
            )
            .returning(EmbeddingCollectionModel)
        )
        value = result.scalar_one_or_none()
        if value is None:
            current = await self.session.get(EmbeddingCollectionModel, collection_id)
            if current is None or current.status != EmbeddingCollectionStatus.READY.value:
                raise InvalidStateTransitionError(
                    "only building embedding collections can become ready"
                )
            if current.index_type != index_type or current.index_params != (index_params or {}):
                raise PersistenceConflictError(
                    "ready embedding collection has different index metadata"
                )
            value = current
        return EmbeddingCollection.model_validate(_model_mapping(value))

    async def mark_collection_failed(self, collection_id: str) -> EmbeddingCollection:
        result = await self.session.execute(
            update(EmbeddingCollectionModel)
            .where(
                EmbeddingCollectionModel.collection_id == collection_id,
                EmbeddingCollectionModel.status == EmbeddingCollectionStatus.BUILDING.value,
            )
            .values(status=EmbeddingCollectionStatus.FAILED.value)
            .returning(EmbeddingCollectionModel)
        )
        value = result.scalar_one_or_none()
        if value is None:
            current = await self.session.get(EmbeddingCollectionModel, collection_id)
            if current is None or current.status != EmbeddingCollectionStatus.FAILED.value:
                raise InvalidStateTransitionError(
                    "only building embedding collections can become failed"
                )
            value = current
        return EmbeddingCollection.model_validate(_model_mapping(value))


class PostgresModelCallRepository:
    def __init__(self, session: AsyncSession, scope: DataScope) -> None:
        self.session = session
        self.scope = scope
        self.model = (
            RuntimeModelCallModel if scope is DataScope.RUNTIME else EvaluationModelCallModel
        )

    async def add(self, trace: ModelCallTrace) -> ModelCallTrace:
        if trace.data_scope is not self.scope:
            raise ValueError("model-call scope does not match repository scope")
        if self.scope is DataScope.RUNTIME and trace.compiler_unit_id is not None:
            # Imported COMPILE rows retain their projected primary keys. Do not
            # create a second parent or rewrite a durable provider response.
            jobs = (await self.session.scalars(
                select(ExperimentJobModel).where(
                    ExperimentJobModel.run_id == trace.run_id,
                    ExperimentJobModel.snapshot_id == trace.snapshot_id,
                    ExperimentJobModel.compiler_unit_id == trace.compiler_unit_id,
                    ExperimentJobModel.stage == AppStage.COMPILE.value,
                    ExperimentJobModel.variant == "questions",
                )
            )).all()
            if len(jobs) == 1 and jobs[0].job_id != trace.job_id:
                trace = bind_imported_compiler_call(trace, _model_mapping(jobs[0]))
        await _insert_idempotent(
            self.session, self.model, model_call_to_row(trace), trace.model_call_id
        )
        return trace

    async def get(self, model_call_id: str) -> ModelCallTrace | None:
        value = await self.session.get(self.model, model_call_id)
        if value is None:
            return None
        return model_call_from_row(_model_mapping(value), self.scope)

    async def get_successful_for_job(self, job_id: str) -> ModelCallTrace | None:
        if self.scope is not DataScope.RUNTIME:
            raise ValueError("experiment job lookup is available only for runtime calls")
        model = RuntimeModelCallModel
        value = await self.session.scalar(
            select(model)
            .where(
                model.job_id == job_id,
                model.status == TraceStatus.SUCCESS.value,
            )
            .order_by(model.created_at.desc(), model.model_call_id.desc())
            .limit(1)
        )
        if value is None:
            return None
        return model_call_from_row(_model_mapping(value), self.scope)


def bind_imported_compiler_call(trace: ModelCallTrace, job: Mapping[str, Any]) -> ModelCallTrace:
    """Project only a native parent reference onto a proven adopted COMPILE job."""
    from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id

    native_id = stable_compiler_id(
        "compile-job", [trace.run_id, trace.compiler_unit_id, "questions"]
    )
    metadata = job.get("metadata", {})
    origin = metadata.get("partial_prefix_import") or metadata.get("upstream_import") or {}
    source_id = origin.get("source_primary_key", {}).get("job_id")
    expected = "q-adopt-" + canonical_content_hash(
        [trace.run_id, "experiment_jobs", source_id]
    )[:40]
    if (
        trace.data_scope is not DataScope.RUNTIME
        or trace.compiler_unit_id is None
        or trace.stage not in {AppStage.QUESTIONER, AppStage.PLANNER, AppStage.DOUBTER}
        or trace.job_id != native_id
        or job.get("job_id") != expected
        or job.get("run_id") != trace.run_id
        or job.get("snapshot_id") != trace.snapshot_id
        or job.get("compiler_unit_id") != trace.compiler_unit_id
        or job.get("stage") != "compile"
        or job.get("variant") != "questions"
        or job.get("status") != "succeeded"
        or origin.get("version") not in {
            "step8-partial-prefix-adoption-v1", "step8-questioner-fact-adoption-v1"
        }
        or not source_id or not origin.get("proof_hash") or not origin.get("source_row_hash")
        or "compiler_job_binding" in trace.metadata
    ):
        raise PersistenceConflictError("compiler_call_adopted_job_binding_unproven")
    return trace.model_copy(update={
        "job_id": job["job_id"],
        "metadata": {**trace.metadata, "compiler_job_binding": {
            "version": "compiler-call-adopted-parent-v1",
            "original_job_id": trace.job_id,
            "original_trace_hash": canonical_content_hash(trace.model_dump(mode="json")),
            "adoption_proof_hash": origin["proof_hash"],
            "parent_row_hash": canonical_content_hash(dict(job)),
        }},
    })


class PostgresRetrievalRepository:
    def __init__(self, session: AsyncSession, batch_size: int = 1000) -> None:
        if batch_size <= 0:
            raise ValueError("retrieval write batch_size must be positive")
        self.session = session
        self.batch_size = batch_size

    async def add_batch(
        self,
        batch: RetrievalBatchTrace,
        traces: list[RetrievalTrace],
        *,
        components: list[RetrievalTraceComponent] | None = None,
        query_embeddings: list[RetrievalQueryEmbeddingRef] | None = None,
    ) -> RetrievalBatchTrace:
        components = components or []
        query_embeddings = query_embeddings or []
        if batch.returned_count != len(traces):
            raise ValueError("returned_count must equal retrieval trace count")
        if [item.rank for item in traces] != list(range(1, len(traces) + 1)):
            raise ValueError("retrieval ranks must be contiguous and ordered from 1")
        linked_document_ids = {
            trace.retrieval_document_id
            for trace in traces
            if trace.retrieval_document_id is not None
        }
        corpus = None
        if batch.corpus_id is not None:
            corpus = await self.session.get(RetrievalCorpusModel, batch.corpus_id)
            if corpus is None or corpus.status != "ready":
                raise InvalidStateTransitionError("retrieval batches require a ready corpus")
            if corpus.snapshot_id != batch.snapshot_id:
                raise ValueError("retrieval batch and corpus snapshots must match")
        collection = None
        if batch.embedding_collection_id is not None:
            collection = await self.session.get(
                EmbeddingCollectionModel, batch.embedding_collection_id
            )
            if collection is None or collection.status != EmbeddingCollectionStatus.READY.value:
                raise InvalidStateTransitionError(
                    "retrieval batches require a ready embedding collection"
                )
            if collection.snapshot_id != batch.snapshot_id:
                raise ValueError("retrieval batch and embedding collection snapshots must match")
            if collection.corpus_id is not None and collection.corpus_id != batch.corpus_id:
                raise ValueError(
                    "retrieval batch corpus must match its embedding collection corpus"
                )
        query_collection = None
        if batch.query_embedding_collection_id is not None:
            query_collection = await self.session.get(
                EmbeddingCollectionModel, batch.query_embedding_collection_id
            )
            if query_collection is None or query_collection.status not in {
                EmbeddingCollectionStatus.BUILDING.value,
                EmbeddingCollectionStatus.READY.value,
            }:
                raise InvalidStateTransitionError(
                    "retrieval batches require a building or ready query collection"
                )
            if query_collection.snapshot_id != batch.snapshot_id:
                raise ValueError("retrieval batch and query collection snapshots must match")
            if query_collection.run_id != batch.run_id:
                raise ValueError("query embedding collection must belong to retrieval run")
        if bool(query_embeddings) != bool(batch.query_embedding_collection_id):
            raise ValueError(
                "query embedding references and query_embedding_collection_id must coexist"
            )
        if linked_document_ids:
            if batch.corpus_id is None:
                raise ValueError("retrieval-document traces require batch corpus_id")
            documents = (
                (
                    await self.session.execute(
                        select(RetrievalDocumentModel).where(
                            RetrievalDocumentModel.document_id.in_(linked_document_ids)
                        )
                    )
                )
                .scalars()
                .all()
            )
            if len(documents) != len(linked_document_ids) or any(
                document.corpus_id != batch.corpus_id for document in documents
            ):
                raise ValueError("retrieval trace document does not belong to batch corpus")
        trace_by_id = {trace.retrieval_trace_id: trace for trace in traces}
        components_by_trace: dict[str, list[RetrievalTraceComponent]] = {}
        for component in components:
            if component.retrieval_trace_id not in trace_by_id:
                raise ValueError("retrieval component does not belong to this batch")
            components_by_trace.setdefault(component.retrieval_trace_id, []).append(component)
        for trace in traces:
            linked = components_by_trace.get(trace.retrieval_trace_id, [])
            if trace.retrieval_window_id is not None:
                if not linked:
                    raise ValueError("window retrieval traces require score components")
                if sum(item.is_winner for item in linked) != 1:
                    raise ValueError(
                        "window retrieval trace requires exactly one winning component"
                    )
                winner = next(item for item in linked if item.is_winner)
                if winner.retrieval_document_id != trace.retrieval_document_id:
                    raise ValueError("winning component must match trace retrieval_document_id")
                if [item.ordinal for item in linked] != list(range(len(linked))):
                    raise ValueError("component ordinals must be contiguous from zero")
            elif linked:
                raise ValueError("document retrieval traces cannot have window components")
        if query_embeddings:
            if [item.ordinal for item in query_embeddings] != list(range(len(query_embeddings))):
                raise ValueError("query embedding ordinals must be contiguous from zero")
            if sum(item.used_for_search for item in query_embeddings) != 1:
                raise ValueError("retrieval batch requires exactly one effective query vector")
            if any(
                item.retrieval_batch_id != batch.retrieval_batch_id for item in query_embeddings
            ):
                raise ValueError("query embedding reference does not match retrieval batch")
        for trace in traces:
            if (
                trace.retrieval_batch_id != batch.retrieval_batch_id
                or trace.run_id != batch.run_id
                or trace.query_id != batch.query_id
                or trace.method != batch.method
                or trace.retriever_name != batch.retriever_name
            ):
                raise ValueError("retrieval trace does not match its batch")
        await _insert_idempotent(
            self.session,
            RetrievalBatchModel,
            retrieval_batch_to_row(batch),
            batch.retrieval_batch_id,
        )
        for start in range(0, len(traces), self.batch_size):
            chunk = traces[start : start + self.batch_size]
            await _insert_many_idempotent(
                self.session,
                RetrievalTraceModel,
                [retrieval_trace_to_row(trace) for trace in chunk],
                identity_key="retrieval_trace_id",
                alternate_keys=("retrieval_batch_id", "rank"),
            )
        if query_embeddings:
            for query_embedding in query_embeddings:
                row = retrieval_query_embedding_to_row(query_embedding, batch=batch)
                await _insert_idempotent(
                    self.session,
                    RetrievalBatchQueryEmbeddingModel,
                    row,
                    row["retrieval_batch_id"],
                    row["embedding_id"],
                )
        for trace_id, linked in components_by_trace.items():
            trace = trace_by_id[trace_id]
            for component in linked:
                row = retrieval_component_to_row(
                    component, trace=trace, snapshot_id=batch.snapshot_id
                )
                await _insert_idempotent(
                    self.session,
                    RetrievalTraceComponentModel,
                    row,
                    row["retrieval_trace_id"],
                    row["retrieval_document_id"],
                )
        return batch

    async def get_batch(
        self, retrieval_batch_id: str
    ) -> (
        tuple[
            RetrievalBatchTrace,
            list[RetrievalTrace],
            list[RetrievalTraceComponent],
            list[RetrievalQueryEmbeddingRef],
        ]
        | None
    ):
        parent = await self.session.get(RetrievalBatchModel, retrieval_batch_id)
        if parent is None:
            return None
        traces = list(
            (
                await self.session.execute(
                    select(RetrievalTraceModel)
                    .where(RetrievalTraceModel.retrieval_batch_id == retrieval_batch_id)
                    .order_by(RetrievalTraceModel.rank)
                )
            ).scalars()
        )
        components = (
            list(
                (
                    await self.session.execute(
                        select(RetrievalTraceComponentModel)
                        .where(
                            RetrievalTraceComponentModel.retrieval_trace_id.in_(
                                [item.retrieval_trace_id for item in traces]
                            )
                        )
                        .order_by(
                            RetrievalTraceComponentModel.retrieval_trace_id,
                            RetrievalTraceComponentModel.ordinal,
                        )
                    )
                ).scalars()
            )
            if traces
            else []
        )
        refs = list(
            (
                await self.session.execute(
                    select(RetrievalBatchQueryEmbeddingModel)
                    .where(
                        RetrievalBatchQueryEmbeddingModel.retrieval_batch_id == retrieval_batch_id
                    )
                    .order_by(RetrievalBatchQueryEmbeddingModel.ordinal)
                )
            ).scalars()
        )
        return (
            retrieval_batch_from_row(_model_mapping(parent)),
            [retrieval_trace_from_row(_model_mapping(item)) for item in traces],
            [retrieval_component_from_row(_model_mapping(item)) for item in components],
            [retrieval_query_embedding_from_row(_model_mapping(item)) for item in refs],
        )


class PostgresAnswerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, answer: FinalAnswerRecord) -> FinalAnswerRecord:
        rows = final_answer_to_rows(answer)
        await _insert_idempotent(
            self.session, FinalAnswerModel, rows["answer"][0], answer.answer_id
        )
        for row in rows["evidence"]:
            await _insert_idempotent(
                self.session,
                FinalAnswerEvidenceIdModel,
                row,
                row["answer_id"],
                row["evidence_id"],
            )
        for row in rows["images"]:
            await _insert_idempotent(
                self.session,
                FinalAnswerImageIdModel,
                row,
                row["answer_id"],
                row["image_id"],
            )
        return answer

    async def get(self, answer_id: str) -> FinalAnswerRecord | None:
        value = await self.session.get(FinalAnswerModel, answer_id)
        if value is None:
            return None
        evidence = list(
            (
                await self.session.scalars(
                    select(FinalAnswerEvidenceIdModel.evidence_id)
                    .where(FinalAnswerEvidenceIdModel.answer_id == answer_id)
                    .order_by(FinalAnswerEvidenceIdModel.ordinal)
                )
            ).all()
        )
        images = list(
            (
                await self.session.scalars(
                    select(FinalAnswerImageIdModel.image_id)
                    .where(FinalAnswerImageIdModel.answer_id == answer_id)
                    .order_by(FinalAnswerImageIdModel.ordinal)
                )
            ).all()
        )
        return final_answer_from_parts(_model_mapping(value), evidence, images)

    async def get_for_query(
        self,
        run_id: str,
        query_id: str,
        variant: str = "primary",
    ) -> FinalAnswerRecord | None:
        answer_id = await self.session.scalar(
            select(FinalAnswerModel.answer_id).where(
                FinalAnswerModel.run_id == run_id,
                FinalAnswerModel.query_id == query_id,
                FinalAnswerModel.answer_variant == variant,
            )
        )
        return await self.get(answer_id) if answer_id is not None else None


class PostgresEvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_retrieval(self, value: RetrievalEvaluation) -> RetrievalEvaluation:
        rows = retrieval_evaluation_to_rows(value)
        await _insert_idempotent(
            self.session,
            RetrievalEvaluationModel,
            rows["evaluation"][0],
            value.retrieval_evaluation_id,
        )
        for row in rows["matches"]:
            await _insert_idempotent(
                self.session,
                RetrievalEvaluationMatchModel,
                row,
                row["retrieval_evaluation_id"],
                row["gold_evidence_id"],
            )
        return value

    async def add_metric(self, value: MetricScore) -> MetricScore:
        await _insert_idempotent(
            self.session, MetricScoreModel, metric_score_to_row(value), value.metric_score_id
        )
        return value

    async def add_judge(self, value: JudgeScore) -> JudgeScore:
        await _insert_idempotent(
            self.session, JudgeScoreModel, judge_score_to_row(value), value.judge_score_id
        )
        return value

    async def add_bad_case(self, value: BadCaseLabel) -> BadCaseLabel:
        await _insert_idempotent(
            self.session,
            BadCaseLabelModel,
            bad_case_label_to_row(value),
            value.bad_case_label_id,
        )
        return value


class ExperimentUnitOfWork:
    """Own one async transaction and its scope-safe repositories."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        scope: DataScope = DataScope.RUNTIME,
        write_batch_size: int = 1000,
    ) -> None:
        if write_batch_size <= 0:
            raise ValueError("experiment write_batch_size must be positive")
        self.sessions = sessions
        self.scope = scope
        self.write_batch_size = write_batch_size
        self.session: AsyncSession | None = None
        self.artifacts: PostgresArtifactRepository
        self.model_calls: PostgresModelCallRepository
        self.runs: PostgresRunRepository | None
        self.jobs: PostgresJobRepository | None
        self.embeddings: PostgresEmbeddingRepository | None
        self.retrieval: PostgresRetrievalRepository | None
        self.answers: PostgresAnswerRepository | None
        self.evaluation: PostgresEvaluationRepository | None
        self.contexts: ContextAssemblyRepository | None
        self.compiler_runs: CompilerRunRepository | None
        self.compiler_units: CompilerUnitRepository | None
        self.qbank_compiler: QBankCompilerRepository | None
        self.memory_programs: MemoryProgramRepository | None
        self.program_executions: ProgramExecutionRepository | None
        self.compiler_stage_jobs: CompilerStageJobRepository | None
        self.planner: PlannerRepository | None
        self.memory_operator_indexes: MemoryOperatorIndexRepository | None
        self.program_query_embeddings: ProgramQueryEmbeddingRepository | None
        self.program_dry_runs: ProgramDryRunRepository | None
        self.doubter: DoubterRepository | None
        self.qbank_routing: QBankRoutingRepository | None
        self.operator_data: OperatorDataRepository | None
        self.questioner: Any | None
        self.qbank_indexes: Any | None
        self.qbank_coverage: Any | None
        self.structured_output_diagnostics: StructuredOutputDiagnosticRepository
        self.qbank_semantic: QBankSemanticMatchRepository | None
        self.query_executor: Any | None

    async def __aenter__(self) -> ExperimentUnitOfWork:
        self.session = self.sessions()
        await self.session.begin()
        self.artifacts = PostgresArtifactRepository(self.session, self.scope)
        self.model_calls = PostgresModelCallRepository(self.session, self.scope)
        if self.scope is DataScope.RUNTIME:
            from pmmc.infrastructure.database.baseline_repositories import (
                PostgresContextAssemblyRepository,
            )
            from pmmc.infrastructure.database.compiler_repositories import (
                PostgresCompilerRunRepository,
                PostgresCompilerUnitRepository,
                PostgresMemoryProgramRepository,
                PostgresOperatorDataRepository,
                PostgresProgramExecutionRepository,
                PostgresQBankCompilerRepository,
                PostgresQBankRoutingRepository,
            )
            from pmmc.infrastructure.database.doubter_repositories import (
                PostgresDoubterRepository,
            )
            from pmmc.infrastructure.database.planner_repositories import (
                PostgresCompilerStageJobRepository,
                PostgresMemoryOperatorIndexRepository,
                PostgresPlannerRepository,
                PostgresProgramDryRunRepository,
                PostgresProgramQueryEmbeddingRepository,
            )
            from pmmc.infrastructure.database.query_executor_repositories import (
                PostgresQueryExecutorRepository,
            )
            from pmmc.infrastructure.database.questioner_repositories import (
                PostgresQBankIndexRepository,
                PostgresQuestionerRepository,
            )
            from pmmc.infrastructure.database.semantic_repositories import (
                PostgresStructuredOutputDiagnosticRepository,
            )

            self.runs = PostgresRunRepository(self.session)
            self.jobs = PostgresJobRepository(self.session)
            self.embeddings = PostgresEmbeddingRepository(self.session, self.write_batch_size)
            self.retrieval = PostgresRetrievalRepository(self.session, self.write_batch_size)
            self.answers = PostgresAnswerRepository(self.session)
            self.contexts = PostgresContextAssemblyRepository(self.session)
            self.compiler_runs = PostgresCompilerRunRepository(self.session)
            self.compiler_units = PostgresCompilerUnitRepository(self.session)
            self.qbank_compiler = PostgresQBankCompilerRepository(self.session)
            self.memory_programs = PostgresMemoryProgramRepository(self.session)
            self.program_executions = PostgresProgramExecutionRepository(self.session)
            self.compiler_stage_jobs = PostgresCompilerStageJobRepository(self.session)
            self.planner = PostgresPlannerRepository(self.session)
            self.memory_operator_indexes = PostgresMemoryOperatorIndexRepository(self.session)
            self.program_query_embeddings = PostgresProgramQueryEmbeddingRepository(self.session)
            self.program_dry_runs = PostgresProgramDryRunRepository(self.session)
            self.doubter = PostgresDoubterRepository(self.session)
            self.qbank_routing = PostgresQBankRoutingRepository(self.session)
            self.operator_data = PostgresOperatorDataRepository(self.session)
            self.questioner = PostgresQuestionerRepository(self.session)
            self.qbank_indexes = PostgresQBankIndexRepository(self.session)
            self.qbank_coverage = None
            self.structured_output_diagnostics = PostgresStructuredOutputDiagnosticRepository(
                self.session, evaluation_scope=False
            )
            self.qbank_semantic = None
            self.query_executor = PostgresQueryExecutorRepository(self.session)
            self.evaluation = None
        else:
            self.runs = None
            self.jobs = None
            self.embeddings = None
            self.retrieval = None
            self.answers = None
            self.contexts = None
            self.compiler_runs = None
            self.compiler_units = None
            self.qbank_compiler = None
            self.memory_programs = None
            self.program_executions = None
            self.compiler_stage_jobs = None
            self.planner = None
            self.memory_operator_indexes = None
            self.program_query_embeddings = None
            self.program_dry_runs = None
            self.doubter = None
            self.qbank_routing = None
            self.operator_data = None
            self.questioner = None
            self.qbank_indexes = None
            self.query_executor = None
            from pmmc.infrastructure.database.questioner_repositories import (
                PostgresQBankCoverageRepository,
            )
            from pmmc.infrastructure.database.semantic_repositories import (
                PostgresQBankSemanticMatchRepository,
                PostgresStructuredOutputDiagnosticRepository,
            )

            self.qbank_coverage = PostgresQBankCoverageRepository(self.session)
            self.structured_output_diagnostics = PostgresStructuredOutputDiagnosticRepository(
                self.session, evaluation_scope=True
            )
            self.qbank_semantic = PostgresQBankSemanticMatchRepository(self.session)
            self.evaluation = PostgresEvaluationRepository(self.session)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.session is None:
            return
        try:
            if exc_type is None:
                await self.session.commit()
            else:
                await self.session.rollback()
        finally:
            await self.session.close()
