"""Recoverable Postgres repositories for Step 6D."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pmmc.infrastructure.database.compiler_repositories import (
    _insert_idempotent,
    _model_mapping,
)
from pmmc.infrastructure.database.doubter_mappers import (
    assessment_from_rows,
    assessment_to_rows,
    candidate_job_from_row,
    candidate_job_to_row,
    r0_from_row,
    r0_to_row,
    revision_from_row,
    revision_to_row,
    round_member_from_row,
    round_member_to_row,
    round_projection_from_row,
    round_projection_to_row,
    routing_member_from_row,
    routing_member_to_row,
    routing_projection_from_row,
    routing_projection_to_row,
    stage_component_from_row,
    stage_component_to_row,
)
from pmmc.infrastructure.database.doubter_models import (
    CandidateStageJobModel,
    DoubterAssessmentDimensionModel,
    DoubterAssessmentModel,
    DoubterDirectiveModel,
    PlannerRevisionLineageModel,
    PlannerRevisionPatchModel,
    QBankRoundProjectionManifestModel,
    QBankRoundProjectionMemberModel,
    R0InputManifestModel,
    RoutingProjectionManifestModel,
    RoutingProjectionMemberModel,
    StageComponentManifestModel,
)
from pmmc.infrastructure.database.errors import InvalidStateTransitionError
from pmmc.schemas import (
    CandidateStageJob,
    CandidateStageJobStatus,
    CandidateStageKind,
    DoubterAssessment,
    PlannerRevisionLineage,
    PlannerRevisionPatch,
    QBankRoundProjectionManifest,
    QBankRoundProjectionMember,
    R0InputManifest,
    RoutingProjectionManifest,
    RoutingProjectionMember,
    StageComponentManifest,
    StageComponentStatus,
)
from pmmc.schemas.base import utc_now


class PostgresDoubterRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_r0_manifest(self, value: R0InputManifest) -> R0InputManifest:
        stored = await _insert_idempotent(
            self.session,
            R0InputManifestModel,
            r0_to_row(value),
            value.r0_manifest_id,
        )
        return r0_from_row(_model_mapping(stored))

    async def get_r0_manifest(self, run_id: str) -> R0InputManifest | None:
        stored = await self.session.scalar(
            select(R0InputManifestModel).where(R0InputManifestModel.run_id == run_id)
        )
        return r0_from_row(_model_mapping(stored)) if stored is not None else None

    async def add_candidate_job(self, value: CandidateStageJob) -> CandidateStageJob:
        stored = await _insert_idempotent(
            self.session,
            CandidateStageJobModel,
            candidate_job_to_row(value),
            value.candidate_stage_job_id,
        )
        return candidate_job_from_row(_model_mapping(stored))

    async def get_candidate_job(self, job_id: str) -> CandidateStageJob | None:
        stored = await self.session.get(CandidateStageJobModel, job_id)
        return candidate_job_from_row(_model_mapping(stored)) if stored is not None else None

    async def list_candidate_jobs(
        self,
        run_id: str,
        *,
        stage: CandidateStageKind | None = None,
        round_id: int | None = None,
    ) -> list[CandidateStageJob]:
        statement = select(CandidateStageJobModel).where(CandidateStageJobModel.run_id == run_id)
        if stage is not None:
            statement = statement.where(CandidateStageJobModel.stage == stage.value)
        if round_id is not None:
            statement = statement.where(CandidateStageJobModel.round_id == round_id)
        rows = (
            (
                await self.session.execute(
                    statement.order_by(
                        CandidateStageJobModel.round_id,
                        CandidateStageJobModel.qbank_id,
                        CandidateStageJobModel.stage,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [candidate_job_from_row(_model_mapping(item)) for item in rows]

    async def claim_candidate_job(
        self,
        *,
        run_id: str,
        stage: CandidateStageKind,
        round_id: int,
        worker_id: str,
        lease_seconds: int,
        qbank_ids: set[str] | None = None,
    ) -> CandidateStageJob | None:
        if qbank_ids is not None and not qbank_ids:
            return None
        now = utc_now()
        statement = select(CandidateStageJobModel).where(
                CandidateStageJobModel.run_id == run_id,
                CandidateStageJobModel.stage == stage.value,
                CandidateStageJobModel.round_id == round_id,
                CandidateStageJobModel.attempt_count < CandidateStageJobModel.max_attempts,
                or_(
                    CandidateStageJobModel.status == CandidateStageJobStatus.PENDING.value,
                    (CandidateStageJobModel.status == CandidateStageJobStatus.RUNNING.value)
                    & (CandidateStageJobModel.lease_expires_at < now),
                    CandidateStageJobModel.status == CandidateStageJobStatus.FAILED.value,
                ),
            )
        if qbank_ids is not None:
            statement = statement.where(CandidateStageJobModel.qbank_id.in_(qbank_ids))
        stored = await self.session.scalar(
            statement
            .order_by(CandidateStageJobModel.qbank_id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if stored is None:
            return None
        stored.status = CandidateStageJobStatus.RUNNING.value
        stored.claimed_by = worker_id
        stored.lease_expires_at = now + timedelta(seconds=lease_seconds)
        stored.started_at = stored.started_at or now
        stored.finished_at = None
        stored.error_type = None
        stored.error_summary = None
        stored.attempt_count += 1
        await self.session.flush()
        return candidate_job_from_row(_model_mapping(stored))

    async def finish_candidate_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        succeeded: bool,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> CandidateStageJob:
        now = utc_now()
        result = await self.session.execute(
            update(CandidateStageJobModel)
            .where(
                CandidateStageJobModel.candidate_stage_job_id == job_id,
                CandidateStageJobModel.status == CandidateStageJobStatus.RUNNING.value,
                CandidateStageJobModel.claimed_by == worker_id,
            )
            .values(
                status=(
                    CandidateStageJobStatus.SUCCEEDED.value
                    if succeeded
                    else CandidateStageJobStatus.FAILED.value
                ),
                claimed_by=None,
                lease_expires_at=None,
                finished_at=now,
                error_type=None if succeeded else (error_type or "UnknownError"),
                error_summary=None if succeeded else error_summary,
            )
        )
        if getattr(result, "rowcount", 0) != 1:
            raise InvalidStateTransitionError("candidate stage job cannot be finished")
        stored = await self.session.get(CandidateStageJobModel, job_id)
        assert stored is not None
        return candidate_job_from_row(_model_mapping(stored))

    async def cancel_candidate_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        terminal_reason: str,
    ) -> CandidateStageJob:
        now = utc_now()
        stored = await self.session.get(CandidateStageJobModel, job_id)
        if (
            stored is None
            or stored.status != CandidateStageJobStatus.RUNNING.value
            or stored.claimed_by != worker_id
        ):
            raise InvalidStateTransitionError("candidate stage job cannot be cancelled")
        metadata = dict(stored.metadata_json or {})
        metadata["terminal_reason"] = terminal_reason
        stored.status = CandidateStageJobStatus.CANCELLED.value
        stored.claimed_by = None
        stored.lease_expires_at = None
        stored.finished_at = now
        stored.error_type = None
        stored.error_summary = None
        stored.metadata_json = metadata
        await self.session.flush()
        return candidate_job_from_row(_model_mapping(stored))

    async def add_assessment(self, value: DoubterAssessment) -> DoubterAssessment:
        row, dimensions, directives = assessment_to_rows(value)
        stored = await _insert_idempotent(
            self.session,
            DoubterAssessmentModel,
            row,
            value.assessment_id,
        )
        for dimension in dimensions:
            await _insert_idempotent(
                self.session,
                DoubterAssessmentDimensionModel,
                dimension,
                dimension["assessment_id"],
                dimension["dimension"],
            )
        for directive in directives:
            await _insert_idempotent(
                self.session,
                DoubterDirectiveModel,
                directive,
                directive["directive_id"],
            )
        return await self._assessment_from_model(stored)

    async def _assessment_from_model(self, stored: DoubterAssessmentModel) -> DoubterAssessment:
        dimensions = (
            (
                await self.session.execute(
                    select(DoubterAssessmentDimensionModel)
                    .where(DoubterAssessmentDimensionModel.assessment_id == stored.assessment_id)
                    .order_by(DoubterAssessmentDimensionModel.dimension)
                )
            )
            .scalars()
            .all()
        )
        directives = (
            (
                await self.session.execute(
                    select(DoubterDirectiveModel)
                    .where(DoubterDirectiveModel.feedback_id == stored.feedback_id)
                    .order_by(DoubterDirectiveModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return assessment_from_rows(
            _model_mapping(stored),
            [_model_mapping(item) for item in dimensions],
            [_model_mapping(item) for item in directives],
        )

    async def get_assessment(
        self, run_id: str, qbank_id: str, round_id: int
    ) -> DoubterAssessment | None:
        stored = await self.session.scalar(
            select(DoubterAssessmentModel).where(
                DoubterAssessmentModel.run_id == run_id,
                DoubterAssessmentModel.qbank_id == qbank_id,
                DoubterAssessmentModel.round_id == round_id,
            )
        )
        return await self._assessment_from_model(stored) if stored is not None else None

    async def list_assessments(self, run_id: str) -> list[DoubterAssessment]:
        rows = (
            (
                await self.session.execute(
                    select(DoubterAssessmentModel)
                    .where(DoubterAssessmentModel.run_id == run_id)
                    .order_by(
                        DoubterAssessmentModel.round_id,
                        DoubterAssessmentModel.qbank_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [await self._assessment_from_model(item) for item in rows]

    async def add_revision(self, value: PlannerRevisionLineage) -> PlannerRevisionLineage:
        stored = await _insert_idempotent(
            self.session,
            PlannerRevisionLineageModel,
            revision_to_row(value),
            value.revision_lineage_id,
        )
        return revision_from_row(_model_mapping(stored))

    async def add_revision_patch(
        self, value: PlannerRevisionPatch
    ) -> PlannerRevisionPatch:
        row = value.model_dump(mode="python")
        row["capability_added"] = [item.value for item in value.capability_added]
        row["capability_removed"] = [item.value for item in value.capability_removed]
        row["metadata"] = row.pop("metadata")
        stored = await _insert_idempotent(
            self.session,
            PlannerRevisionPatchModel,
            row,
            value.patch_id,
        )
        return PlannerRevisionPatch.model_validate(_model_mapping(stored))

    async def list_revisions(self, run_id: str) -> list[PlannerRevisionLineage]:
        rows = (
            (
                await self.session.execute(
                    select(PlannerRevisionLineageModel)
                    .where(PlannerRevisionLineageModel.run_id == run_id)
                    .order_by(
                        PlannerRevisionLineageModel.target_round_id,
                        PlannerRevisionLineageModel.qbank_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [revision_from_row(_model_mapping(item)) for item in rows]

    async def list_revision_patches(self, run_id: str) -> list[PlannerRevisionPatch]:
        rows = (
            (
                await self.session.execute(
                    select(PlannerRevisionPatchModel)
                    .where(PlannerRevisionPatchModel.run_id == run_id)
                    .order_by(
                        PlannerRevisionPatchModel.target_round_id,
                        PlannerRevisionPatchModel.qbank_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [PlannerRevisionPatch.model_validate(_model_mapping(item)) for item in rows]

    async def add_round_projection(
        self,
        manifest: QBankRoundProjectionManifest,
        members: list[QBankRoundProjectionMember],
    ) -> QBankRoundProjectionManifest:
        if manifest.member_count != len(members) or any(
            item.projection_id != manifest.projection_id or item.run_id != manifest.run_id
            for item in members
        ):
            raise ValueError("round projection members do not match their manifest")
        stored = await _insert_idempotent(
            self.session,
            QBankRoundProjectionManifestModel,
            round_projection_to_row(manifest),
            manifest.projection_id,
        )
        for member in members:
            await _insert_idempotent(
                self.session,
                QBankRoundProjectionMemberModel,
                round_member_to_row(member),
                member.projection_member_id,
            )
        return round_projection_from_row(_model_mapping(stored))

    async def get_round_projection(
        self, run_id: str, round_id: int
    ) -> tuple[QBankRoundProjectionManifest, list[QBankRoundProjectionMember]] | None:
        stored = await self.session.scalar(
            select(QBankRoundProjectionManifestModel).where(
                QBankRoundProjectionManifestModel.run_id == run_id,
                QBankRoundProjectionManifestModel.round_id == round_id,
            )
        )
        if stored is None:
            return None
        members = (
            (
                await self.session.execute(
                    select(QBankRoundProjectionMemberModel)
                    .where(QBankRoundProjectionMemberModel.projection_id == stored.projection_id)
                    .order_by(QBankRoundProjectionMemberModel.qbank_id)
                )
            )
            .scalars()
            .all()
        )
        return (
            round_projection_from_row(_model_mapping(stored)),
            [round_member_from_row(_model_mapping(item)) for item in members],
        )

    async def add_routing_projection(
        self,
        manifest: RoutingProjectionManifest,
        members: list[RoutingProjectionMember],
    ) -> RoutingProjectionManifest:
        if manifest.member_count != len(members) or any(
            item.routing_projection_id != manifest.routing_projection_id
            or item.run_id != manifest.run_id
            for item in members
        ):
            raise ValueError("routing members do not match their manifest")
        stored = await _insert_idempotent(
            self.session,
            RoutingProjectionManifestModel,
            routing_projection_to_row(manifest),
            manifest.routing_projection_id,
        )
        for member in members:
            await _insert_idempotent(
                self.session,
                RoutingProjectionMemberModel,
                routing_member_to_row(member),
                member.routing_member_id,
            )
        return routing_projection_from_row(_model_mapping(stored))

    async def get_routing_projection(
        self, run_id: str
    ) -> tuple[RoutingProjectionManifest, list[RoutingProjectionMember]] | None:
        stored = await self.session.scalar(
            select(RoutingProjectionManifestModel).where(
                RoutingProjectionManifestModel.run_id == run_id
            )
        )
        if stored is None:
            return None
        members = (
            (
                await self.session.execute(
                    select(RoutingProjectionMemberModel)
                    .where(
                        RoutingProjectionMemberModel.routing_projection_id
                        == stored.routing_projection_id
                    )
                    .order_by(RoutingProjectionMemberModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return (
            routing_projection_from_row(_model_mapping(stored)),
            [routing_member_from_row(_model_mapping(item)) for item in members],
        )

    async def add_stage_component(self, value: StageComponentManifest) -> StageComponentManifest:
        stored = await _insert_idempotent(
            self.session,
            StageComponentManifestModel,
            stage_component_to_row(value),
            value.component_id,
        )
        return stage_component_from_row(_model_mapping(stored))

    async def list_stage_components(
        self, launch_id: str, *, ready_only: bool = False
    ) -> list[StageComponentManifest]:
        statement = select(StageComponentManifestModel).where(
            StageComponentManifestModel.launch_id == launch_id
        )
        if ready_only:
            statement = statement.where(
                StageComponentManifestModel.status == StageComponentStatus.READY.value
            )
        rows = (
            (
                await self.session.execute(
                    statement.order_by(
                        StageComponentManifestModel.created_at,
                        StageComponentManifestModel.component_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [stage_component_from_row(_model_mapping(item)) for item in rows]
