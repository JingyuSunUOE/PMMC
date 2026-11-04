"""Recoverable Postgres repositories for Step 7 query execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, NoReturn, TypeVar

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.infrastructure.database.errors import (
    InvalidStateTransitionError,
    PersistenceConflictError,
)
from pmmc.infrastructure.database.experiment_mappers import model_call_from_row
from pmmc.infrastructure.database.experiment_models import EmbeddingModel, RuntimeModelCallModel
from pmmc.infrastructure.database.query_executor_models import (
    AnswerFormatDiagnosticModel,
    AnswerReuseLinkModel,
    PmmcFallbackQueryLinkModel,
    PmmcFallbackRunLinkModel,
    QBankLexicalTermFactModel,
    QBankMatchBatchModel,
    QBankMatchChannelCandidateModel,
    QBankMatchFusedCandidateModel,
    QBankRerankerAssessmentModel,
    QueryAnswerSelectionModel,
    QueryEmbeddingBatchModel,
    QueryEmbeddingModel,
    QueryEvidenceRelevanceAssessmentModel,
    QueryEvidenceRelevanceItemModel,
    QueryEvidenceSufficiencyAssessmentModel,
    QueryExecutionReuseLinkModel,
    QueryExecutorFreezeManifestModel,
    QueryExecutorRunModel,
    QueryExecutorStageJobModel,
    QueryExperimentCellResultModel,
    QueryFallbackDecisionModel,
    QueryProgramBindingModel,
    QueryRouteDecisionModel,
    QueryRuntimeEmbeddingCacheModel,
    QueryRuntimePhaseStateModel,
    QueryRuntimeTypedAnswerModel,
    StructuredAnswerOutputModel,
)
from pmmc.schemas import (
    AnswerFormatDiagnostic,
    AnswerReuseLink,
    DataScope,
    ModelCallTrace,
    PmmcFallbackQueryLink,
    PmmcFallbackRunLink,
    QBankLexicalTermFact,
    QBankMatchBatch,
    QBankMatchChannelCandidate,
    QBankMatchFusedCandidate,
    QBankRerankerAssessment,
    QueryAnswerSelection,
    QueryEmbeddingBatch,
    QueryEmbeddingFact,
    QueryEvidenceRelevanceAssessment,
    QueryEvidenceRelevanceItem,
    QueryEvidenceSufficiencyAssessment,
    QueryExecutionReuseLink,
    QueryExecutorFreezeManifest,
    QueryExecutorRunManifest,
    QueryExecutorRunState,
    QueryExecutorRunStatus,
    QueryExecutorStageJob,
    QueryFallbackDecision,
    QueryProgramBinding,
    QueryRouteDecision,
    StructuredAnswerFact,
)
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.base import utc_now
from pmmc.schemas.persistence import JobStatus
from pmmc.schemas.query_executor import (
    QueryExperimentCellResult,
    QueryRuntimeTypedAnswerFact,
    runtime_fact_payload,
)

ModelT = TypeVar("ModelT", bound=DeclarativeBase)

_QUERY_STAGE_JOB_IDENTITY_FIELDS = (
    "query_stage_job_id",
    "query_executor_run_id",
    "snapshot_id",
    "query_id",
    "stage",
    "round_id",
    "max_attempts",
    "content_hash",
    "metadata",
)
_IDEMPOTENT_AUDIT_FIELDS = frozenset({"created_at"})


class QueryExecutorPersistenceConflictError(PersistenceConflictError):
    """Persistence conflict carrying only allow-listed structural diagnostics."""

    def __init__(self, message: str, *, operation: str, error_code: str) -> None:
        super().__init__(message)
        self.persistence_operation = operation
        self.error_code = error_code


def _row(value: object) -> dict[str, Any]:
    return value.model_dump(mode="python")  # type: ignore[attr-defined]


def _mapping(value: DeclarativeBase) -> dict[str, Any]:
    return {
        prop.columns[0].name: getattr(value, prop.key) for prop in value.__mapper__.column_attrs
    }


def _idempotent_insert_statement(
    model: type[ModelT],
    row: Mapping[str, Any],
    *,
    index_elements: Sequence[str],
) -> Insert:
    """Build Core DML so the SQL ``metadata`` column cannot hit ORM metadata."""

    return (
        insert(model.__table__)  # type: ignore[attr-defined,arg-type]
        .values(dict(row))
        .on_conflict_do_nothing(index_elements=list(index_elements))
    )


def _query_stage_job_identity_conflicts(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return immutable field names that differ; scheduler state is intentionally ignored."""

    return tuple(
        name for name in _QUERY_STAGE_JOB_IDENTITY_FIELDS if actual.get(name) != expected.get(name)
    )


def _idempotent_identity_conflicts(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[str, ...]:
    """Compare immutable fact content while preserving the first insert audit time."""

    return tuple(
        name
        for name, expected_value in expected.items()
        if name not in _IDEMPOTENT_AUDIT_FIELDS and actual.get(name) != expected_value
    )


def _conflict(message: str, *, operation: str, error_code: str) -> NoReturn:
    raise QueryExecutorPersistenceConflictError(
        message,
        operation=operation,
        error_code=error_code,
    )


async def _add_idempotent(
    session: AsyncSession,
    model: type[ModelT],
    value: object,
    *,
    primary_key: str,
) -> ModelT:
    row = _row(value)
    return await _add_idempotent_row(session, model, row, primary_key=primary_key)


async def _add_idempotent_row(
    session: AsyncSession,
    model: type[ModelT],
    row: Mapping[str, Any],
    *,
    primary_key: str,
) -> ModelT:
    key = row[primary_key]
    try:
        await session.execute(
            _idempotent_insert_statement(model, row, index_elements=[primary_key])
        )
    except SQLAlchemyError as exc:
        # Driver diagnostics can omit table/constraint names for trigger errors.
        # Keep a code-owned operation without exposing the SQL or row values.
        exc.persistence_operation = f"add_{model.__tablename__}"  # type: ignore[attr-defined]
        raise
    stored = await session.get(model, key)
    if stored is None:
        _conflict(
            f"failed to persist {model.__tablename__}",
            operation=f"add_{model.__tablename__}",
            error_code="row_missing_after_idempotent_insert",
        )
    actual = _mapping(stored)
    if _idempotent_identity_conflicts(actual, row):
        _conflict(
            f"existing {model.__tablename__} row conflicts with immutable input",
            operation=f"add_{model.__tablename__}",
            error_code="immutable_row_identity_conflict",
        )
    return stored


class PostgresQueryExecutorRepository:
    """All query facts are written in caller-owned short transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_runtime_embedding_cache(self, cache_id: str) -> dict | None:
        row = await self.session.get(QueryRuntimeEmbeddingCacheModel, cache_id)
        return _mapping(row) if row is not None else None

    async def add_runtime_embedding_cache(self, payload: dict) -> None:
        await _add_idempotent_row(
            self.session, QueryRuntimeEmbeddingCacheModel, payload, primary_key="cache_id"
        )

    async def add_phase_state(self, run_id: str, query_id: str, phase: str, payload: dict) -> None:
        if phase not in {"match", "answer"}:
            raise ValueError("unsupported runtime checkpoint phase")
        clean = runtime_fact_payload(payload)
        await _add_idempotent_row(
            self.session,
            QueryRuntimePhaseStateModel,
            {
                "phase_state_id": stable_compiler_id(
                    "runtime-query-phase", [run_id, query_id, phase]
                ),
                "query_executor_run_id": run_id,
                "query_id": query_id,
                "phase": phase,
                "payload": clean,
                "content_hash": canonical_content_hash(clean),
                "created_at": utc_now(),
            },
            primary_key="phase_state_id",
        )

    async def get_phase_state(self, run_id: str, query_id: str, phase: str) -> dict | None:
        row = await self.session.scalar(
            select(QueryRuntimePhaseStateModel).where(
                QueryRuntimePhaseStateModel.query_executor_run_id == run_id,
                QueryRuntimePhaseStateModel.query_id == query_id,
                QueryRuntimePhaseStateModel.phase == phase,
            )
        )
        return dict(row.payload) if row is not None else None

    async def add_typed_answer(self, value: QueryRuntimeTypedAnswerFact) -> None:
        payload = runtime_fact_payload(value)
        await _add_idempotent_row(
            self.session,
            QueryRuntimeTypedAnswerModel,
            {
                **{
                    key: getattr(value, key)
                    for key in (
                        "typed_answer_id",
                        "query_executor_run_id",
                        "query_id",
                        "answer_identity_hash",
                        "answer_model_call_id",
                        "terminal_state",
                        "content_hash",
                        "created_at",
                    )
                },
                "payload": payload,
            },
            primary_key="typed_answer_id",
        )

    async def get_typed_answer(
        self,
        run_id: str,
        query_id: str,
        answer_identity_hash: str,
    ) -> QueryRuntimeTypedAnswerFact | None:
        row = await self.session.scalar(
            select(QueryRuntimeTypedAnswerModel).where(
                QueryRuntimeTypedAnswerModel.query_executor_run_id == run_id,
                QueryRuntimeTypedAnswerModel.query_id == query_id,
                QueryRuntimeTypedAnswerModel.answer_identity_hash == answer_identity_hash,
            )
        )
        return (
            QueryRuntimeTypedAnswerFact.model_validate(
                {
                    **row.payload,
                    "created_at": row.created_at,
                }
            )
            if row is not None
            else None
        )

    async def add_native_typed_answer(self, result: TypedAnswerResult) -> None:
        payload = runtime_fact_payload(result)
        fact = QueryRuntimeTypedAnswerFact(
            typed_answer_id=result.typed_answer_id,
            query_executor_run_id=result.run_id,
            query_id=result.query_id,
            answer_identity_hash=canonical_content_hash(
                [result.run_id, result.query_id, result.answer_variant]
            ),
            answer_model_call_id=result.answer_model_call_id,
            terminal_state=result.terminal_state,
            output=result.output.model_dump(mode="json"),
            ungated_output=result.ungated_output.model_dump(mode="json"),
            image_items=[item.model_dump(mode="json") for item in result.image_items],
            grounding=result.grounding.model_dump(mode="json") if result.grounding else None,
            content_hash=canonical_content_hash(payload),
            metadata={"typed_result": payload},
        )
        await self.add_typed_answer(fact)

    async def get_typed_answer_for_variant(
        self,
        run_id: str,
        query_id: str,
        variant: str = "primary",
    ) -> TypedAnswerResult | None:
        rows = await self.session.scalars(
            select(QueryRuntimeTypedAnswerModel).where(
                QueryRuntimeTypedAnswerModel.query_executor_run_id == run_id,
                QueryRuntimeTypedAnswerModel.query_id == query_id,
            )
        )
        results = [
            TypedAnswerResult.model_validate(row.payload["metadata"]["typed_result"])
            for row in rows
            if row.payload["metadata"]["typed_result"]["answer_variant"] == variant
        ]
        if len(results) > 1:
            raise ValueError("typed answer variant has conflicting identities")
        return results[0] if results else None

    async def add_cell_result(self, value: QueryExperimentCellResult) -> None:
        payload = runtime_fact_payload(value)
        await _add_idempotent_row(
            self.session,
            QueryExperimentCellResultModel,
            {
                **{
                    key: getattr(value, key)
                    for key in (
                        "cell_result_id",
                        "query_executor_run_id",
                        "query_id",
                        "cell_id",
                        "terminal_state",
                        "selected_answer_id",
                        "content_hash",
                        "created_at",
                    )
                },
                "treatment": value.cell.treatment,
                "question_budget": value.cell.question_budget,
                "revision_depth": value.cell.revision_depth,
                "payload": payload,
            },
            primary_key="cell_result_id",
        )

    async def list_cell_results(self, run_id: str) -> list[QueryExperimentCellResult]:
        rows = await self.session.scalars(
            select(QueryExperimentCellResultModel)
            .where(
                QueryExperimentCellResultModel.query_executor_run_id == run_id,
            )
            .order_by(
                QueryExperimentCellResultModel.query_id, QueryExperimentCellResultModel.cell_id
            )
        )
        return [
            QueryExperimentCellResult.model_validate(
                {
                    **row.payload,
                    "created_at": row.created_at,
                }
            )
            for row in rows
        ]

    async def get_reranker_calls(
        self,
        run_id: str,
        query_id: str,
        call_ids: Sequence[str],
    ) -> dict[str, ModelCallTrace]:
        if not call_ids:
            return {}
        rows = (
            await self.session.scalars(
                select(RuntimeModelCallModel).where(
                    RuntimeModelCallModel.run_id == run_id,
                    RuntimeModelCallModel.query_id == query_id,
                    RuntimeModelCallModel.model_call_id.in_(call_ids),
                )
            )
        ).all()
        return {
            row.model_call_id: model_call_from_row(_mapping(row), DataScope.RUNTIME) for row in rows
        }

    async def add_run(self, value: QueryExecutorRunManifest) -> Mapping[str, Any]:
        row = _row(value)
        row.update(
            {
                "status": "created",
                "matched_count": 0,
                "executed_count": 0,
                "pmmc_answer_count": 0,
                "fallback_answer_count": 0,
                "selected_answer_count": 0,
                "failed_count": 0,
            }
        )
        await self.session.execute(
            _idempotent_insert_statement(
                QueryExecutorRunModel,
                row,
                index_elements=["query_executor_run_id"],
            )
        )
        stored = await self.session.get(QueryExecutorRunModel, value.query_executor_run_id)
        if stored is None or stored.content_hash != value.content_hash:
            _conflict(
                "query executor run identity conflict",
                operation="add_query_executor_run",
                error_code="query_executor_run_identity_conflict",
            )
        return _mapping(stored)

    async def get_run(self, run_id: str) -> Mapping[str, Any] | None:
        stored = await self.session.get(QueryExecutorRunModel, run_id)
        return _mapping(stored) if stored is not None else None

    async def get_state(self, run_id: str) -> QueryExecutorRunState:
        stored = await self.session.get(QueryExecutorRunModel, run_id)
        if stored is None:
            raise KeyError(f"query executor run does not exist: {run_id}")
        return QueryExecutorRunState(
            query_executor_run_id=stored.query_executor_run_id,
            status=QueryExecutorRunStatus(stored.status),
            selected_count=stored.selected_query_count,
            matched_count=stored.matched_count,
            executed_count=stored.executed_count,
            pmmc_answer_count=stored.pmmc_answer_count,
            fallback_answer_count=stored.fallback_answer_count,
            selected_answer_count=stored.selected_answer_count,
            failed_count=stored.failed_count,
            started_at=stored.started_at,
            heartbeat_at=stored.heartbeat_at,
            completed_at=stored.completed_at,
            error_type=stored.error_type,
            error_summary=stored.error_summary,
            metadata=dict(stored.metadata_json),
        )

    async def transition_run(
        self,
        run_id: str,
        *,
        status: str,
        counts: Mapping[str, int] | None = None,
        now: datetime,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> None:
        values: dict[str, object] = {
            "status": status,
            "heartbeat_at": now,
            "error_type": error_type,
            "error_summary": error_summary,
        }
        if status == "running":
            values["started_at"] = now
        if status in {"completed", "failed", "cancelled"}:
            values["completed_at"] = now
        values.update(dict(counts or {}))
        result = await self.session.execute(
            update(QueryExecutorRunModel)
            .where(
                QueryExecutorRunModel.query_executor_run_id == run_id,
                QueryExecutorRunModel.status.not_in(("completed", "cancelled")),
            )
            .values(**values)
        )
        if getattr(result, "rowcount", 0) != 1:
            raise InvalidStateTransitionError("query executor run transition rejected")

    async def add_stage_jobs(self, values: Sequence[QueryExecutorStageJob]) -> None:
        for value in values:
            row = _row(value)
            await self.session.execute(
                _idempotent_insert_statement(
                    QueryExecutorStageJobModel,
                    row,
                    index_elements=["query_stage_job_id"],
                )
            )
            stored = await self.session.get(
                QueryExecutorStageJobModel,
                value.query_stage_job_id,
            )
            if stored is None:
                _conflict(
                    "failed to persist query_executor_stage_jobs",
                    operation="add_query_executor_stage_jobs",
                    error_code="query_stage_job_missing_after_insert",
                )
            conflicts = _query_stage_job_identity_conflicts(_mapping(stored), row)
            if conflicts:
                _conflict(
                    "existing query_executor_stage_jobs row conflicts with immutable input",
                    operation="add_query_executor_stage_jobs",
                    error_code="query_stage_job_identity_conflict",
                )

    async def claim_stage_job(
        self,
        *,
        run_id: str,
        stage: str,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> QueryExecutorStageJob | None:
        row = await self.session.scalar(
            select(QueryExecutorStageJobModel)
            .where(
                QueryExecutorStageJobModel.query_executor_run_id == run_id,
                QueryExecutorStageJobModel.stage == stage,
                or_(
                    QueryExecutorStageJobModel.status == JobStatus.PENDING.value,
                    (
                        (QueryExecutorStageJobModel.status == JobStatus.RUNNING.value)
                        & (QueryExecutorStageJobModel.lease_expires_at < now)
                    ),
                ),
                QueryExecutorStageJobModel.available_at <= now,
            )
            .order_by(QueryExecutorStageJobModel.query_id, QueryExecutorStageJobModel.round_id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if row is None:
            return None
        row.status = JobStatus.RUNNING.value
        row.claimed_by = worker_id
        row.lease_expires_at = now + timedelta(seconds=lease_seconds)
        row.started_at = row.started_at or now
        row.attempt_count += 1
        return QueryExecutorStageJob.model_validate(_mapping(row))

    async def finish_stage_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        success: bool,
        now: datetime,
        error_type: str | None = None,
        error_summary: str | None = None,
        retryable: bool = True,
        attempt_count: int | None = None,
    ) -> QueryExecutorStageJob:
        row = await self.session.scalar(
            select(QueryExecutorStageJobModel)
            .where(QueryExecutorStageJobModel.query_stage_job_id == job_id)
            .with_for_update()
        )
        if row is None:
            raise KeyError(f"query executor stage job does not exist: {job_id}")
        if row.status != JobStatus.RUNNING.value or row.claimed_by != worker_id:
            raise InvalidStateTransitionError("only the active query worker may finish a job")
        if attempt_count is not None and (
            row.attempt_count != attempt_count
            or row.lease_expires_at is None
            or row.lease_expires_at <= now
        ):
            raise InvalidStateTransitionError("query attempt lease expired or was superseded")
        if success:
            row.status = JobStatus.SUCCEEDED.value
            row.finished_at = now
        elif retryable and row.attempt_count < row.max_attempts:
            if error_type is None:
                raise ValueError("failed query jobs require error_type")
            row.status = JobStatus.PENDING.value
            row.available_at = now
            row.finished_at = None
        else:
            if error_type is None:
                raise ValueError("failed query jobs require error_type")
            row.status = JobStatus.FAILED.value
            row.finished_at = now
        row.claimed_by = None
        row.lease_expires_at = None
        row.error_type = error_type
        row.error_summary = error_summary
        await self.session.flush()
        return QueryExecutorStageJob.model_validate(_mapping(row))

    async def heartbeat_stage_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
        attempt_count: int | None = None,
    ) -> QueryExecutorStageJob:
        """Renew only a lease still owned by the active query worker."""

        result = await self.session.execute(
            update(QueryExecutorStageJobModel)
            .where(
                QueryExecutorStageJobModel.query_stage_job_id == job_id,
                QueryExecutorStageJobModel.status == JobStatus.RUNNING.value,
                QueryExecutorStageJobModel.claimed_by == worker_id,
                *(
                    [
                        QueryExecutorStageJobModel.attempt_count == attempt_count,
                        QueryExecutorStageJobModel.lease_expires_at > now,
                    ]
                    if attempt_count is not None
                    else []
                ),
            )
            .values(lease_expires_at=now + timedelta(seconds=lease_seconds))
        )
        if getattr(result, "rowcount", 0) != 1:
            raise InvalidStateTransitionError("query stage job lease is not owned")
        stored = await self.session.get(QueryExecutorStageJobModel, job_id)
        assert stored is not None
        await self.session.execute(
            update(QueryExecutorRunModel)
            .where(
                QueryExecutorRunModel.query_executor_run_id == stored.query_executor_run_id,
                QueryExecutorRunModel.status == "running",
            )
            .values(heartbeat_at=now)
        )
        return QueryExecutorStageJob.model_validate(_mapping(stored))

    async def assert_attempt_lease(
        self,
        *,
        job_id: str,
        worker_id: str,
        attempt_count: int,
        now: datetime,
    ) -> None:
        """Lock the claim until this transaction commits, fencing stale writers."""
        row = await self.session.scalar(
            select(QueryExecutorStageJobModel)
            .where(
                QueryExecutorStageJobModel.query_stage_job_id == job_id,
            )
            .with_for_update()
        )
        if (
            row is None
            or row.status != JobStatus.RUNNING.value
            or row.claimed_by != worker_id
            or row.attempt_count != attempt_count
            or row.lease_expires_at is None
            or row.lease_expires_at <= now
        ):
            raise InvalidStateTransitionError("query attempt no longer owns a live lease")

    async def stage_status_counts(self, run_id: str, stage: str) -> dict[str, int]:
        rows = await self.session.execute(
            select(QueryExecutorStageJobModel.status, func.count())
            .where(
                QueryExecutorStageJobModel.query_executor_run_id == run_id,
                QueryExecutorStageJobModel.stage == stage,
            )
            .group_by(QueryExecutorStageJobModel.status)
        )
        return {str(status): int(count) for status, count in rows}

    async def add_embedding_batch(self, value: QueryEmbeddingBatch) -> None:
        await _add_idempotent(
            self.session,
            QueryEmbeddingBatchModel,
            value,
            primary_key="query_embedding_batch_id",
        )

    async def add_embedding(self, value: QueryEmbeddingFact) -> None:
        await _add_idempotent(
            self.session,
            QueryEmbeddingModel,
            value,
            primary_key="query_embedding_id",
        )

    async def get_query_embedding_vectors(
        self,
        *,
        run_id: str,
        query_id: str,
    ) -> dict[str, tuple[QueryEmbeddingFact, list[float]]]:
        """Return complete persisted query vectors keyed by runtime channel."""

        rows = (
            await self.session.execute(
                select(QueryEmbeddingModel, EmbeddingModel)
                .join(
                    EmbeddingModel,
                    EmbeddingModel.embedding_id == QueryEmbeddingModel.embedding_id,
                )
                .where(
                    QueryEmbeddingModel.query_executor_run_id == run_id,
                    QueryEmbeddingModel.query_id == query_id,
                )
                .order_by(QueryEmbeddingModel.channel)
            )
        ).all()
        result: dict[str, tuple[QueryEmbeddingFact, list[float]]] = {}
        for fact_row, embedding_row in rows:
            fact = QueryEmbeddingFact.model_validate(_mapping(fact_row))
            vector = list(embedding_row.embedding)
            if len(vector) != fact.vector_dimension:
                _conflict(
                    "query embedding dimension differs from its immutable fact",
                    operation="load_query_embeddings",
                    error_code="query_embedding_dimension_conflict",
                )
            if fact.channel.value in result:
                _conflict(
                    "query has more than one embedding for the same runtime channel",
                    operation="load_query_embeddings",
                    error_code="duplicate_query_embedding_channel",
                )
            result[fact.channel.value] = (fact, vector)
        return result

    async def list_query_embedding_facts(
        self,
        *,
        run_id: str,
        channel: str,
        query_ids: Sequence[str],
    ) -> dict[str, QueryEmbeddingFact]:
        if channel not in {"text", "multimodal"}:
            raise ValueError("query embedding channel must be text or multimodal")
        if not query_ids:
            return {}
        rows = list(
            (
                await self.session.scalars(
                    select(QueryEmbeddingModel)
                    .where(
                        QueryEmbeddingModel.query_executor_run_id == run_id,
                        QueryEmbeddingModel.channel == channel,
                        QueryEmbeddingModel.query_id.in_(query_ids),
                    )
                    .order_by(QueryEmbeddingModel.query_id)
                )
            ).all()
        )
        return {row.query_id: QueryEmbeddingFact.model_validate(_mapping(row)) for row in rows}

    async def terminal_fact_counts(self, run_id: str) -> dict[str, int]:
        """Compute run counters from terminal facts instead of inferred job success."""

        run = await self.session.get(QueryExecutorRunModel, run_id)
        if run is not None and run.schema_version == "0.3.0":
            cells = await self.list_cell_results(run_id)
            active = [
                fact
                for fact in cells
                if fact.cell.treatment not in {"no_fallback", "forced_top1_counterfactual"}
                and fact.cell.question_budget == run.question_budget
                and fact.cell.revision_depth == run.active_round
            ]
            return {
                "matched_count": len(active),
                "executed_count": sum(fact.execution_id is not None for fact in active),
                "pmmc_answer_count": sum(fact.pmmc_answer_id is not None for fact in active),
                "fallback_answer_count": sum(
                    fact.fallback_answer_id is not None for fact in active
                ),
                # A legal abstention is a completed selection, never answer success.
                "selected_answer_count": sum(
                    fact.selected_answer_id is not None
                    and fact.terminal_state in {"answered", "abstained"} for fact in active
                ),
                "answered_count": sum(fact.terminal_state == "answered" for fact in active),
                "abstained_count": sum(fact.terminal_state == "abstained" for fact in active),
                "failed_count": sum(
                    fact.terminal_state not in {"answered", "abstained"} for fact in active
                ),
                "structured_answer_count": sum(
                    fact.selected_answer_id is not None for fact in active
                ),
                "structured_fallback_answer_count": sum(
                    fact.active_fallback and fact.terminal_state == "answered" for fact in active
                ),
                "fallback_abstention_count": sum(
                    fact.active_fallback and fact.terminal_state == "abstained" for fact in active
                ),
                "answer_format_rejected_attempt_count": 0,
                "cell_result_count": len(cells),
                "technical_failure_count": sum(
                    fact.terminal_state == "technical_failure" for fact in active
                ),
            }

        async def count(model: type[DeclarativeBase]) -> int:
            value: int = (
                await self.session.execute(
                    select(func.count())
                    .select_from(model)
                    .where(
                        model.query_executor_run_id == run_id  # type: ignore[attr-defined]
                    )
                )
            ).scalar_one()
            return value

        pmmc_answers = await self.session.scalar(
            select(func.count(func.distinct(QueryAnswerSelectionModel.query_id))).where(
                QueryAnswerSelectionModel.query_executor_run_id == run_id,
                QueryAnswerSelectionModel.pmmc_answer_id.is_not(None),
            )
        )
        executed = await self.session.scalar(
            select(func.count(func.distinct(QueryProgramBindingModel.query_id))).where(
                QueryProgramBindingModel.query_executor_run_id == run_id
            )
        )
        selected_answers = await self.session.scalar(
            select(func.count())
            .select_from(QueryAnswerSelectionModel)
            .where(
                QueryAnswerSelectionModel.query_executor_run_id == run_id,
                QueryAnswerSelectionModel.terminal_failure.is_(False),
            )
        )
        terminal_failures = await self.session.scalar(
            select(func.count())
            .select_from(QueryAnswerSelectionModel)
            .where(
                QueryAnswerSelectionModel.query_executor_run_id == run_id,
                QueryAnswerSelectionModel.terminal_failure.is_(True),
            )
        )
        structured_answers = await self.session.scalar(
            select(func.count())
            .select_from(StructuredAnswerOutputModel)
            .where(
                StructuredAnswerOutputModel.run_id == run_id,
                StructuredAnswerOutputModel.parse_status == "accepted",
                StructuredAnswerOutputModel.supported.is_(True),
            )
        )
        structured_fallback_answers = await self.session.scalar(
            select(func.count())
            .select_from(StructuredAnswerOutputModel)
            .where(
                StructuredAnswerOutputModel.run_id == run_id,
                StructuredAnswerOutputModel.answer_variant == "native_mm_counterfactual",
                StructuredAnswerOutputModel.parse_status == "accepted",
                StructuredAnswerOutputModel.supported.is_(True),
            )
        )
        fallback_abstentions = await self.session.scalar(
            select(func.count())
            .select_from(StructuredAnswerOutputModel)
            .where(
                StructuredAnswerOutputModel.run_id == run_id,
                StructuredAnswerOutputModel.answer_variant == "native_mm_counterfactual",
                StructuredAnswerOutputModel.parse_status == "accepted",
                StructuredAnswerOutputModel.supported.is_(False),
            )
        )
        rejected_answer_attempts = await self.session.scalar(
            select(func.count())
            .select_from(AnswerFormatDiagnosticModel)
            .where(
                AnswerFormatDiagnosticModel.run_id == run_id,
                AnswerFormatDiagnosticModel.accepted.is_(False),
            )
        )
        return {
            "matched_count": await count(QueryRouteDecisionModel),
            "executed_count": int(executed or 0),
            "pmmc_answer_count": int(pmmc_answers or 0),
            "fallback_answer_count": await count(QueryFallbackDecisionModel),
            "selected_answer_count": int(selected_answers or 0),
            "failed_count": int(terminal_failures or 0),
            "structured_answer_count": int(structured_answers or 0),
            "structured_fallback_answer_count": int(structured_fallback_answers or 0),
            "fallback_abstention_count": int(fallback_abstentions or 0),
            "answer_format_rejected_attempt_count": int(rejected_answer_attempts or 0),
        }

    async def add_match(
        self,
        batch: QBankMatchBatch,
        channels: Sequence[QBankMatchChannelCandidate],
        fused: Sequence[QBankMatchFusedCandidate],
    ) -> None:
        await _add_idempotent(
            self.session, QBankMatchBatchModel, batch, primary_key="match_batch_id"
        )
        for channel_value in channels:
            await _add_idempotent(
                self.session,
                QBankMatchChannelCandidateModel,
                channel_value,
                primary_key="channel_candidate_id",
            )
        for fused_value in fused:
            await _add_idempotent(
                self.session,
                QBankMatchFusedCandidateModel,
                fused_value,
                primary_key="fused_candidate_id",
            )

    async def add_lexical_terms(self, values: Sequence[QBankLexicalTermFact]) -> None:
        for value in values:
            await _add_idempotent(
                self.session,
                QBankLexicalTermFactModel,
                value,
                primary_key="lexical_term_fact_id",
            )

    async def add_reranker_assessments(self, values: Sequence[QBankRerankerAssessment]) -> None:
        for value in values:
            row = {
                "reranker_assessment_id": value.reranker_assessment_id,
                "query_executor_run_id": value.query_executor_run_id,
                "query_id": value.query_id,
                "qbank_id": value.qbank_id,
                "model_call_id": value.model_call_id,
                "same_information_need": value.dimensions.same_information_need.value,
                "candidate_can_answer_query": (value.dimensions.candidate_can_answer_query.value),
                "answer_type_compatible": value.dimensions.answer_type_compatible.value,
                "modality_compatible": value.dimensions.modality_compatible.value,
                "match_level": value.effective_match_level.value,
                "reranker_score": value.reranker_score,
                "provider_schema_conformant": value.provider_schema_conformant,
                "content_hash": value.content_hash,
                "created_at": value.created_at,
                "metadata": {
                    **value.metadata,
                    "prompt_version": value.prompt_version,
                    "provider_match_level": value.provider_match_level.value,
                },
            }
            stored = await _add_idempotent_row(
                self.session,
                QBankRerankerAssessmentModel,
                row,
                primary_key="reranker_assessment_id",
            )
            del stored

    async def add_route(self, value: QueryRouteDecision) -> None:
        await _add_idempotent(
            self.session, QueryRouteDecisionModel, value, primary_key="route_decision_id"
        )

    async def add_binding(self, value: QueryProgramBinding) -> None:
        await _add_idempotent(
            self.session,
            QueryProgramBindingModel,
            value,
            primary_key="query_program_binding_id",
        )

    async def add_sufficiency(self, value: QueryEvidenceSufficiencyAssessment) -> None:
        await _add_idempotent(
            self.session,
            QueryEvidenceSufficiencyAssessmentModel,
            value,
            primary_key="sufficiency_assessment_id",
        )

    async def add_relevance(
        self,
        assessment: QueryEvidenceRelevanceAssessment,
        items: Sequence[QueryEvidenceRelevanceItem],
    ) -> None:
        await _add_idempotent(
            self.session,
            QueryEvidenceRelevanceAssessmentModel,
            assessment,
            primary_key="relevance_assessment_id",
        )
        for item in items:
            await _add_idempotent(
                self.session,
                QueryEvidenceRelevanceItemModel,
                item,
                primary_key="relevance_item_id",
            )

    async def add_execution_reuse(self, value: QueryExecutionReuseLink) -> None:
        await _add_idempotent(
            self.session,
            QueryExecutionReuseLinkModel,
            value,
            primary_key="execution_reuse_link_id",
        )

    async def add_structured_answer(self, value: StructuredAnswerFact) -> None:
        row = value.model_dump(mode="python")
        row["abstain_reason"] = (
            value.abstain_reason.value if value.abstain_reason is not None else None
        )
        await _add_idempotent_row(
            self.session,
            StructuredAnswerOutputModel,
            row,
            primary_key="structured_answer_output_id",
        )

    async def get_abstention(
        self,
        run_id: str,
        query_id: str,
        answer_variant: str = "primary",
    ) -> StructuredAnswerFact | None:
        from pmmc.infrastructure.database.experiment_models import RuntimeModelCallModel
        from pmmc.runtime.answering import validate_abstention_fact

        rows = (
            await self.session.scalars(
                select(StructuredAnswerOutputModel).where(
                    StructuredAnswerOutputModel.run_id == run_id,
                    StructuredAnswerOutputModel.query_id == query_id,
                    StructuredAnswerOutputModel.answer_variant == answer_variant,
                    StructuredAnswerOutputModel.supported.is_(False),
                    StructuredAnswerOutputModel.parse_status == "accepted",
                )
            )
        ).all()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("answer has multiple abstention terminals")
        fact = StructuredAnswerFact.model_validate(_mapping(rows[0]))
        validate_abstention_fact(fact)
        call = await self.session.get(RuntimeModelCallModel, fact.model_call_id)
        source = fact.metadata.get("source_run_id", run_id)
        if (
            call is None
            or call.run_id != source
            or call.query_id != query_id
            or call.stage != "answer"
            or call.status != "success"
            or (
                fact.metadata.get("method") == "native_multimodal_rag"
                and call.context_assembly_id is None
            )
        ):
            raise ValueError("answer abstention model-call lineage mismatch")
        return fact

    async def abstention_count(self, run_id: str, answer_variant: str = "primary") -> int:
        return int(
            await self.session.scalar(
                select(func.count())
                .select_from(StructuredAnswerOutputModel)
                .where(
                    StructuredAnswerOutputModel.run_id == run_id,
                    StructuredAnswerOutputModel.answer_variant == answer_variant,
                    StructuredAnswerOutputModel.parse_status == "accepted",
                    StructuredAnswerOutputModel.supported.is_(False),
                )
            )
            or 0
        )

    async def add_answer_diagnostic(self, value: AnswerFormatDiagnostic) -> None:
        row = value.model_dump(mode="python")
        row["failure_codes"] = [item.value for item in value.failure_codes]
        await _add_idempotent_row(
            self.session,
            AnswerFormatDiagnosticModel,
            row,
            primary_key="diagnostic_id",
        )

    async def add_answer_reuse(self, value: AnswerReuseLink) -> None:
        await _add_idempotent(
            self.session,
            AnswerReuseLinkModel,
            value,
            primary_key="answer_reuse_link_id",
        )

    async def add_fallback(self, value: QueryFallbackDecision) -> None:
        await _add_idempotent(
            self.session,
            QueryFallbackDecisionModel,
            value,
            primary_key="fallback_decision_id",
        )

    async def add_selection(self, value: QueryAnswerSelection) -> None:
        await _add_idempotent(
            self.session,
            QueryAnswerSelectionModel,
            value,
            primary_key="answer_selection_id",
        )

    async def add_fallback_run_link(self, value: PmmcFallbackRunLink) -> None:
        await _add_idempotent(
            self.session,
            PmmcFallbackRunLinkModel,
            value,
            primary_key="query_executor_run_id",
        )

    async def add_fallback_query_link(self, value: PmmcFallbackQueryLink) -> None:
        row = _row(value)
        await self.session.execute(
            _idempotent_insert_statement(
                PmmcFallbackQueryLinkModel,
                row,
                index_elements=["query_executor_run_id", "query_id"],
            )
        )
        stored = await self.session.scalar(
            select(PmmcFallbackQueryLinkModel).where(
                PmmcFallbackQueryLinkModel.query_executor_run_id == value.query_executor_run_id,
                PmmcFallbackQueryLinkModel.query_id == value.query_id,
            )
        )
        if stored is None:
            _conflict(
                "failed to persist pmmc_fallback_query_links",
                operation="add_pmmc_fallback_query_links",
                error_code="fallback_query_link_missing_after_insert",
            )
        actual = _mapping(stored)
        if any(actual.get(name) != expected for name, expected in row.items()):
            _conflict(
                "existing pmmc_fallback_query_links row conflicts with immutable input",
                operation="add_pmmc_fallback_query_links",
                error_code="fallback_query_link_identity_conflict",
            )

    async def add_freeze_manifest(self, value: QueryExecutorFreezeManifest) -> None:
        await _add_idempotent(
            self.session,
            QueryExecutorFreezeManifestModel,
            value,
            primary_key="freeze_manifest_id",
        )


__all__ = ["PostgresQueryExecutorRepository"]
