"""Async Postgres repositories for Planner v1 and program dry-run facts."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pmmc.infrastructure.database.compiler_models import (
    CompilerStageJobModel,
    MemoryOperatorIndexManifestModel,
    PlannerBatchProgramModel,
    PlannerCandidateDraftModel,
    PlannerCandidateOutcomeModel,
    PlannerGenerationBatchModel,
    PlannerRequirementContractModel,
    ProgramDryRunAssessmentModel,
    ProgramDryRunEvidenceMatchModel,
    ProgramQueryEmbeddingBatchModel,
    ProgramQueryEmbeddingModel,
    ProgramValidationIssueModel,
    ProgramValidationResultModel,
)
from pmmc.infrastructure.database.compiler_repositories import (
    _insert_idempotent,
    _model_mapping,
)
from pmmc.infrastructure.database.errors import InvalidStateTransitionError
from pmmc.infrastructure.database.planner_mappers import (
    compiler_stage_job_from_row,
    compiler_stage_job_to_row,
    dry_run_assessment_from_rows,
    dry_run_assessment_to_rows,
    memory_index_from_row,
    memory_index_to_row,
    planner_batch_program_to_row,
    planner_candidate_draft_from_row,
    planner_candidate_draft_to_row,
    planner_candidate_outcome_from_row,
    planner_candidate_outcome_to_row,
    planner_generation_batch_from_row,
    planner_generation_batch_to_row,
    planner_requirement_contract_from_row,
    planner_requirement_contract_to_row,
    query_embedding_batch_from_row,
    query_embedding_batch_to_row,
    query_embedding_from_row,
    query_embedding_to_row,
    validation_result_from_rows,
    validation_result_to_rows,
)
from pmmc.schemas import (
    CompilerStageJob,
    CompilerStageJobStatus,
    CompilerStageKind,
    MemoryOperatorIndexManifest,
    PlannerBatchProgram,
    PlannerCandidateDraftFact,
    PlannerCandidateOutcome,
    PlannerGenerationBatch,
    PlannerRequirementContract,
    ProgramDryRunAssessment,
    ProgramEvidenceMatch,
    ProgramQueryEmbedding,
    ProgramQueryEmbeddingBatch,
    ProgramValidationKind,
    ProgramValidationResultFact,
)
from pmmc.schemas.base import utc_now


class PostgresCompilerStageJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, value: CompilerStageJob) -> CompilerStageJob:
        stored = await _insert_idempotent(
            self.session,
            CompilerStageJobModel,
            compiler_stage_job_to_row(value),
            value.stage_job_id,
        )
        return compiler_stage_job_from_row(_model_mapping(stored))

    async def get(self, stage_job_id: str) -> CompilerStageJob | None:
        stored = await self.session.get(CompilerStageJobModel, stage_job_id)
        return (
            compiler_stage_job_from_row(_model_mapping(stored))
            if stored is not None
            else None
        )

    async def list_for_run(
        self, run_id: str, stage: CompilerStageKind | None = None
    ) -> list[CompilerStageJob]:
        statement = select(CompilerStageJobModel).where(
            CompilerStageJobModel.run_id == run_id
        )
        if stage is not None:
            statement = statement.where(CompilerStageJobModel.stage == stage.value)
        rows = (
            (
                await self.session.execute(
                    statement.order_by(
                        CompilerStageJobModel.stage,
                        CompilerStageJobModel.compiler_unit_id,
                        CompilerStageJobModel.round_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [compiler_stage_job_from_row(_model_mapping(item)) for item in rows]

    async def claim(
        self,
        *,
        run_id: str,
        stage: CompilerStageKind,
        worker_id: str,
        lease_seconds: int,
    ) -> CompilerStageJob | None:
        if lease_seconds <= 0:
            raise ValueError("compiler stage lease_seconds must be positive")
        now = utc_now()
        row = await self.session.scalar(
            select(CompilerStageJobModel)
            .where(
                CompilerStageJobModel.run_id == run_id,
                CompilerStageJobModel.stage == stage.value,
                CompilerStageJobModel.attempt_count
                < CompilerStageJobModel.max_attempts,
                or_(
                    CompilerStageJobModel.status == CompilerStageJobStatus.PENDING.value,
                    CompilerStageJobModel.status == CompilerStageJobStatus.FAILED.value,
                    (
                        (CompilerStageJobModel.status == CompilerStageJobStatus.RUNNING.value)
                        & (CompilerStageJobModel.lease_expires_at < now)
                    ),
                ),
                CompilerStageJobModel.available_at <= now,
            )
            .order_by(
                CompilerStageJobModel.compiler_unit_id,
                CompilerStageJobModel.round_id,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if row is None:
            return None
        await self.session.execute(
            update(CompilerStageJobModel)
            .where(CompilerStageJobModel.stage_job_id == row.stage_job_id)
            .values(
                status=CompilerStageJobStatus.RUNNING.value,
                attempt_count=row.attempt_count + 1,
                claimed_by=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                started_at=row.started_at or now,
                finished_at=None,
                error_type=None,
                error_summary=None,
            )
        )
        await self.session.flush()
        await self.session.refresh(row)
        return compiler_stage_job_from_row(_model_mapping(row))

    async def heartbeat(
        self, stage_job_id: str, *, worker_id: str, lease_seconds: int
    ) -> CompilerStageJob:
        now = utc_now()
        result = await self.session.execute(
            update(CompilerStageJobModel)
            .where(
                CompilerStageJobModel.stage_job_id == stage_job_id,
                CompilerStageJobModel.status == CompilerStageJobStatus.RUNNING.value,
                CompilerStageJobModel.claimed_by == worker_id,
            )
            .values(lease_expires_at=now + timedelta(seconds=lease_seconds))
        )
        if getattr(result, "rowcount", 0) != 1:
            raise InvalidStateTransitionError("compiler stage job lease is not owned")
        stored = await self.session.get(CompilerStageJobModel, stage_job_id)
        assert stored is not None
        return compiler_stage_job_from_row(_model_mapping(stored))

    async def finish(
        self,
        stage_job_id: str,
        *,
        worker_id: str,
        succeeded: bool,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> CompilerStageJob:
        if not succeeded and not error_type:
            raise ValueError("failed compiler stage job requires error_type")
        now = utc_now()
        result = await self.session.execute(
            update(CompilerStageJobModel)
            .where(
                CompilerStageJobModel.stage_job_id == stage_job_id,
                CompilerStageJobModel.status == CompilerStageJobStatus.RUNNING.value,
                CompilerStageJobModel.claimed_by == worker_id,
            )
            .values(
                status=(
                    CompilerStageJobStatus.SUCCEEDED.value
                    if succeeded
                    else CompilerStageJobStatus.FAILED.value
                ),
                claimed_by=None,
                lease_expires_at=None,
                finished_at=now,
                error_type=error_type,
                error_summary=error_summary,
            )
        )
        if getattr(result, "rowcount", 0) != 1:
            raise InvalidStateTransitionError("compiler stage job cannot be finished")
        stored = await self.session.get(CompilerStageJobModel, stage_job_id)
        assert stored is not None
        return compiler_stage_job_from_row(_model_mapping(stored))


class PostgresPlannerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_batch(self, value: PlannerGenerationBatch) -> PlannerGenerationBatch:
        stored = await _insert_idempotent(
            self.session,
            PlannerGenerationBatchModel,
            planner_generation_batch_to_row(value),
            value.planner_batch_id,
        )
        return planner_generation_batch_from_row(_model_mapping(stored))

    async def add_requirement_contract(
        self, value: PlannerRequirementContract
    ) -> PlannerRequirementContract:
        stored = await _insert_idempotent(
            self.session,
            PlannerRequirementContractModel,
            planner_requirement_contract_to_row(value),
            value.requirement_contract_id,
        )
        return planner_requirement_contract_from_row(_model_mapping(stored))

    async def get_requirement_contract(
        self, qbank_id: str
    ) -> PlannerRequirementContract | None:
        stored = await self.session.scalar(
            select(PlannerRequirementContractModel).where(
                PlannerRequirementContractModel.qbank_id == qbank_id
            )
        )
        return (
            planner_requirement_contract_from_row(_model_mapping(stored))
            if stored is not None
            else None
        )

    async def list_requirement_contracts(
        self, run_id: str
    ) -> list[PlannerRequirementContract]:
        stored = list(
            (
                await self.session.scalars(
                    select(PlannerRequirementContractModel)
                    .where(PlannerRequirementContractModel.run_id == run_id)
                    .order_by(PlannerRequirementContractModel.qbank_id)
                )
            ).all()
        )
        return [
            planner_requirement_contract_from_row(_model_mapping(item))
            for item in stored
        ]

    async def add_candidate_draft(
        self, value: PlannerCandidateDraftFact
    ) -> PlannerCandidateDraftFact:
        stored = await _insert_idempotent(
            self.session,
            PlannerCandidateDraftModel,
            planner_candidate_draft_to_row(value),
            value.planner_candidate_draft_id,
        )
        return planner_candidate_draft_from_row(_model_mapping(stored))

    async def get_batch(
        self, compiler_unit_id: str, batch_ordinal: int
    ) -> PlannerGenerationBatch | None:
        stored = await self.session.scalar(
            select(PlannerGenerationBatchModel).where(
                PlannerGenerationBatchModel.compiler_unit_id == compiler_unit_id,
                PlannerGenerationBatchModel.batch_ordinal == batch_ordinal,
            )
        )
        return (
            planner_generation_batch_from_row(_model_mapping(stored))
            if stored is not None
            else None
        )

    async def add_program_link(self, value: PlannerBatchProgram) -> None:
        await _insert_idempotent(
            self.session,
            PlannerBatchProgramModel,
            planner_batch_program_to_row(value),
            value.planner_batch_id,
            value.plan_id,
        )

    async def add_outcome(self, value: PlannerCandidateOutcome) -> PlannerCandidateOutcome:
        stored = await _insert_idempotent(
            self.session,
            PlannerCandidateOutcomeModel,
            planner_candidate_outcome_to_row(value),
            value.planner_outcome_id,
        )
        return planner_candidate_outcome_from_row(_model_mapping(stored))

    async def get_outcome(self, qbank_id: str) -> PlannerCandidateOutcome | None:
        stored = await self.session.scalar(
            select(PlannerCandidateOutcomeModel).where(
                PlannerCandidateOutcomeModel.qbank_id == qbank_id
            )
        )
        return (
            planner_candidate_outcome_from_row(_model_mapping(stored))
            if stored is not None
            else None
        )

    async def list_outcomes(self, run_id: str) -> list[PlannerCandidateOutcome]:
        rows = (
            (
                await self.session.execute(
                    select(PlannerCandidateOutcomeModel)
                    .where(PlannerCandidateOutcomeModel.run_id == run_id)
                    .order_by(PlannerCandidateOutcomeModel.qbank_id)
                )
            )
            .scalars()
            .all()
        )
        return [planner_candidate_outcome_from_row(_model_mapping(item)) for item in rows]

    async def add_validation(
        self, value: ProgramValidationResultFact
    ) -> ProgramValidationResultFact:
        result_row, issue_rows = validation_result_to_rows(value)
        stored = await _insert_idempotent(
            self.session,
            ProgramValidationResultModel,
            result_row,
            value.validation_result_id,
        )
        for row in issue_rows:
            await _insert_idempotent(
                self.session,
                ProgramValidationIssueModel,
                row,
                row["validation_issue_id"],
            )
        issues = (
            (
                await self.session.execute(
                    select(ProgramValidationIssueModel)
                    .where(
                        ProgramValidationIssueModel.validation_result_id
                        == value.validation_result_id
                    )
                    .order_by(ProgramValidationIssueModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return validation_result_from_rows(
            _model_mapping(stored), [_model_mapping(item) for item in issues]
        )

    async def get_validation(
        self, plan_id: str, kind: ProgramValidationKind
    ) -> ProgramValidationResultFact | None:
        stored = await self.session.scalar(
            select(ProgramValidationResultModel).where(
                ProgramValidationResultModel.plan_id == plan_id,
                ProgramValidationResultModel.validation_kind == kind.value,
            )
        )
        if stored is None:
            return None
        issues = (
            (
                await self.session.execute(
                    select(ProgramValidationIssueModel)
                    .where(
                        ProgramValidationIssueModel.validation_result_id
                        == stored.validation_result_id
                    )
                    .order_by(ProgramValidationIssueModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return validation_result_from_rows(
            _model_mapping(stored), [_model_mapping(item) for item in issues]
        )


class PostgresMemoryOperatorIndexRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self, value: MemoryOperatorIndexManifest
    ) -> MemoryOperatorIndexManifest:
        stored = await _insert_idempotent(
            self.session,
            MemoryOperatorIndexManifestModel,
            memory_index_to_row(value),
            value.memory_index_id,
        )
        return memory_index_from_row(_model_mapping(stored))

    async def get_for_run(self, run_id: str) -> MemoryOperatorIndexManifest | None:
        stored = await self.session.scalar(
            select(MemoryOperatorIndexManifestModel).where(
                MemoryOperatorIndexManifestModel.run_id == run_id
            )
        )
        return memory_index_from_row(_model_mapping(stored)) if stored else None

    async def mark_ready(self, memory_index_id: str) -> MemoryOperatorIndexManifest:
        stored = await self.session.get(MemoryOperatorIndexManifestModel, memory_index_id)
        if stored is None:
            raise KeyError(f"memory operator index does not exist: {memory_index_id}")
        if stored.status == "failed":
            raise InvalidStateTransitionError("failed memory index cannot become ready")
        if stored.status != "ready":
            stored.status = "ready"
            await self.session.flush()
        return memory_index_from_row(_model_mapping(stored))


class PostgresProgramQueryEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_batch(
        self,
        batch: ProgramQueryEmbeddingBatch,
        values: list[ProgramQueryEmbedding],
    ) -> ProgramQueryEmbeddingBatch:
        if {item.qbank_id for item in values} != set(batch.qbank_ids):
            raise ValueError("program query embedding batch must be committed atomically")
        if any(
            item.run_id != batch.run_id
            or item.query_embedding_batch_id != batch.query_embedding_batch_id
            or item.channel != batch.channel
            for item in values
        ):
            raise ValueError("program query embeddings conflict with batch identity")
        stored = await _insert_idempotent(
            self.session,
            ProgramQueryEmbeddingBatchModel,
            query_embedding_batch_to_row(batch),
            batch.query_embedding_batch_id,
        )
        for value in values:
            await _insert_idempotent(
                self.session,
                ProgramQueryEmbeddingModel,
                query_embedding_to_row(value),
                value.program_query_embedding_id,
            )
        return query_embedding_batch_from_row(_model_mapping(stored))

    async def list_for_run(self, run_id: str) -> list[ProgramQueryEmbedding]:
        rows = (
            (
                await self.session.execute(
                    select(ProgramQueryEmbeddingModel)
                    .where(ProgramQueryEmbeddingModel.run_id == run_id)
                    .order_by(
                        ProgramQueryEmbeddingModel.channel,
                        ProgramQueryEmbeddingModel.qbank_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [query_embedding_from_row(_model_mapping(item)) for item in rows]


class PostgresProgramDryRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        value: ProgramDryRunAssessment,
        matches: list[ProgramEvidenceMatch],
    ) -> ProgramDryRunAssessment:
        row, match_rows = dry_run_assessment_to_rows(value, matches)
        stored = await _insert_idempotent(
            self.session,
            ProgramDryRunAssessmentModel,
            row,
            value.assessment_id,
        )
        for match_row in match_rows:
            await _insert_idempotent(
                self.session,
                ProgramDryRunEvidenceMatchModel,
                match_row,
                match_row["match_id"],
            )
        return ProgramDryRunAssessment.model_validate(_model_mapping(stored))

    async def get(
        self, plan_id: str
    ) -> tuple[ProgramDryRunAssessment, tuple[ProgramEvidenceMatch, ...]] | None:
        stored = await self.session.scalar(
            select(ProgramDryRunAssessmentModel).where(
                ProgramDryRunAssessmentModel.plan_id == plan_id
            )
        )
        if stored is None:
            return None
        matches = (
            (
                await self.session.execute(
                    select(ProgramDryRunEvidenceMatchModel)
                    .where(
                        ProgramDryRunEvidenceMatchModel.assessment_id
                        == stored.assessment_id
                    )
                    .order_by(ProgramDryRunEvidenceMatchModel.match_id)
                )
            )
            .scalars()
            .all()
        )
        return dry_run_assessment_from_rows(
            _model_mapping(stored), [_model_mapping(item) for item in matches]
        )

    async def list_for_run(self, run_id: str) -> list[ProgramDryRunAssessment]:
        rows = (
            (
                await self.session.execute(
                    select(ProgramDryRunAssessmentModel)
                    .where(ProgramDryRunAssessmentModel.run_id == run_id)
                    .order_by(ProgramDryRunAssessmentModel.plan_id)
                )
            )
            .scalars()
            .all()
        )
        return [ProgramDryRunAssessment.model_validate(_model_mapping(item)) for item in rows]
