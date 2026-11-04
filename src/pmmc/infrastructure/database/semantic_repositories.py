"""Async Postgres repositories for QBank semantic matching."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pmmc.infrastructure.database.compiler_repositories import (
    _insert_idempotent,
    _model_mapping,
)
from pmmc.infrastructure.database.questioner_models import (
    EvaluationStructuredOutputItemDiagnosticModel,
    QBankSemanticMatchAssessmentModel,
    QBankSemanticMatchJobModel,
    QBankSemanticMatchRunModel,
    QBankSemanticMetricModel,
    RuntimeStructuredOutputItemDiagnosticModel,
)
from pmmc.infrastructure.database.semantic_mappers import (
    semantic_assessment_from_row,
    semantic_assessment_to_row,
    semantic_job_from_row,
    semantic_job_to_row,
    semantic_metric_from_row,
    semantic_metric_to_row,
    semantic_run_from_row,
    semantic_run_to_row,
    structured_diagnostic_from_row,
    structured_diagnostic_to_row,
)
from pmmc.schemas import (
    JobStatus,
    QBankSemanticMatchAssessment,
    QBankSemanticMatchJob,
    QBankSemanticMatchRun,
    QBankSemanticMatchStatus,
    QBankSemanticMetric,
    StructuredOutputItemDiagnostic,
)
from pmmc.schemas.base import utc_now


class PostgresStructuredOutputDiagnosticRepository:
    def __init__(self, session: AsyncSession, *, evaluation_scope: bool) -> None:
        self.session = session
        self.model = (
            EvaluationStructuredOutputItemDiagnosticModel
            if evaluation_scope
            else RuntimeStructuredOutputItemDiagnosticModel
        )

    async def add(
        self, value: StructuredOutputItemDiagnostic
    ) -> StructuredOutputItemDiagnostic:
        stored = await _insert_idempotent(
            self.session,
            self.model,
            structured_diagnostic_to_row(value),
            value.diagnostic_id,
        )
        return structured_diagnostic_from_row(_model_mapping(stored))

    async def list_for_call(
        self, model_call_id: str
    ) -> list[StructuredOutputItemDiagnostic]:
        rows = (
            (
                await self.session.execute(
                    select(self.model)
                    .where(self.model.model_call_id == model_call_id)
                    .order_by(self.model.item_ordinal)
                )
            )
            .scalars()
            .all()
        )
        return [
            structured_diagnostic_from_row(_model_mapping(item)) for item in rows
        ]


class PostgresQBankSemanticMatchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_run(
        self, value: QBankSemanticMatchRun
    ) -> QBankSemanticMatchRun:
        stored = await _insert_idempotent(
            self.session,
            QBankSemanticMatchRunModel,
            semantic_run_to_row(value),
            value.semantic_run_id,
        )
        return semantic_run_from_row(_model_mapping(stored))

    async def get_run(self, semantic_run_id: str) -> QBankSemanticMatchRun | None:
        stored = await self.session.get(QBankSemanticMatchRunModel, semantic_run_id)
        return semantic_run_from_row(_model_mapping(stored)) if stored else None

    async def add_job(
        self, value: QBankSemanticMatchJob
    ) -> QBankSemanticMatchJob:
        stored = await _insert_idempotent(
            self.session,
            QBankSemanticMatchJobModel,
            semantic_job_to_row(value),
            value.semantic_job_id,
        )
        return semantic_job_from_row(_model_mapping(stored))

    async def get_job(
        self, semantic_run_id: str, query_id: str, qbank_id: str
    ) -> QBankSemanticMatchJob | None:
        stored = await self.session.scalar(
            select(QBankSemanticMatchJobModel).where(
                QBankSemanticMatchJobModel.semantic_run_id == semantic_run_id,
                QBankSemanticMatchJobModel.query_id == query_id,
                QBankSemanticMatchJobModel.qbank_id == qbank_id,
            )
        )
        return semantic_job_from_row(_model_mapping(stored)) if stored else None

    async def get_job_by_id(
        self, semantic_job_id: str
    ) -> QBankSemanticMatchJob | None:
        stored = await self.session.get(QBankSemanticMatchJobModel, semantic_job_id)
        return semantic_job_from_row(_model_mapping(stored)) if stored else None

    async def list_jobs(
        self, semantic_run_id: str
    ) -> list[QBankSemanticMatchJob]:
        rows = list(
            await self.session.scalars(
                select(QBankSemanticMatchJobModel)
                .where(
                    QBankSemanticMatchJobModel.semantic_run_id == semantic_run_id
                )
                .order_by(
                    QBankSemanticMatchJobModel.query_id,
                    QBankSemanticMatchJobModel.qbank_id,
                )
            )
        )
        return [semantic_job_from_row(_model_mapping(item)) for item in rows]

    async def transition_run(
        self,
        semantic_run_id: str,
        status: QBankSemanticMatchStatus,
    ) -> QBankSemanticMatchRun:
        current = await self.get_run(semantic_run_id)
        if current is None:
            raise KeyError(semantic_run_id)
        allowed = {
            QBankSemanticMatchStatus.CREATED: {
                QBankSemanticMatchStatus.RUNNING,
                QBankSemanticMatchStatus.FAILED,
                QBankSemanticMatchStatus.CANCELLED,
            },
            QBankSemanticMatchStatus.RUNNING: {
                QBankSemanticMatchStatus.COMPLETED,
                QBankSemanticMatchStatus.FAILED,
                QBankSemanticMatchStatus.CANCELLED,
            },
            QBankSemanticMatchStatus.COMPLETED: set(),
            QBankSemanticMatchStatus.FAILED: set(),
            QBankSemanticMatchStatus.CANCELLED: set(),
        }
        if status is current.status:
            return current
        if status not in allowed[current.status]:
            raise ValueError(
                "invalid QBank semantic run transition: "
                f"{current.status.value} -> {status.value}"
            )
        await self.session.execute(
            update(QBankSemanticMatchRunModel)
            .where(
                QBankSemanticMatchRunModel.semantic_run_id == semantic_run_id
            )
            .values(
                status=status.value,
                completed_at=(
                    utc_now()
                    if status
                    in {
                        QBankSemanticMatchStatus.COMPLETED,
                        QBankSemanticMatchStatus.FAILED,
                        QBankSemanticMatchStatus.CANCELLED,
                    }
                    else None
                ),
            )
        )
        stored = await self.get_run(semantic_run_id)
        if stored is None:
            raise RuntimeError("semantic run disappeared after transition")
        return stored

    async def transition_job(
        self,
        semantic_job_id: str,
        status: JobStatus,
        *,
        model_call_id: str | None = None,
    ) -> QBankSemanticMatchJob:
        current = await self.get_job_by_id(semantic_job_id)
        if current is None:
            raise KeyError(semantic_job_id)
        allowed = {
            JobStatus.PENDING: {
                JobStatus.RUNNING,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            JobStatus.RUNNING: {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            JobStatus.SUCCEEDED: set(),
            JobStatus.FAILED: set(),
            JobStatus.CANCELLED: set(),
        }
        if status is current.status:
            if model_call_id is not None and current.model_call_id != model_call_id:
                raise ValueError("semantic job model-call identity conflicts")
            return current
        if status not in allowed[current.status]:
            raise ValueError(
                "invalid QBank semantic job transition: "
                f"{current.status.value} -> {status.value}"
            )
        attempt_count = current.attempt_count + (
            1 if status is JobStatus.RUNNING else 0
        )
        if attempt_count > current.max_attempts:
            raise ValueError("semantic job exhausted its attempts")
        values: dict[str, object] = {
            "status": status.value,
            "attempt_count": attempt_count,
            "finished_at": (
                utc_now()
                if status
                in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }
                else None
            ),
        }
        if model_call_id is not None:
            values["model_call_id"] = model_call_id
        await self.session.execute(
            update(QBankSemanticMatchJobModel)
            .where(
                QBankSemanticMatchJobModel.semantic_job_id == semantic_job_id
            )
            .values(**values)
        )
        stored = await self.get_job_by_id(semantic_job_id)
        if stored is None:
            raise RuntimeError("semantic job disappeared after transition")
        return stored

    async def add_assessment(
        self, value: QBankSemanticMatchAssessment
    ) -> QBankSemanticMatchAssessment:
        stored = await _insert_idempotent(
            self.session,
            QBankSemanticMatchAssessmentModel,
            semantic_assessment_to_row(value),
            value.semantic_assessment_id,
        )
        return semantic_assessment_from_row(_model_mapping(stored))

    async def get_assessment(
        self, semantic_job_id: str
    ) -> QBankSemanticMatchAssessment | None:
        stored = await self.session.scalar(
            select(QBankSemanticMatchAssessmentModel).where(
                QBankSemanticMatchAssessmentModel.semantic_job_id
                == semantic_job_id
            )
        )
        return (
            semantic_assessment_from_row(_model_mapping(stored))
            if stored is not None
            else None
        )

    async def add_metrics(self, values: list[QBankSemanticMetric]) -> None:
        for value in values:
            await _insert_idempotent(
                self.session,
                QBankSemanticMetricModel,
                semantic_metric_to_row(value),
                value.semantic_metric_id,
            )

    async def list_assessments(
        self, semantic_run_id: str
    ) -> list[QBankSemanticMatchAssessment]:
        rows = (
            (
                await self.session.execute(
                    select(QBankSemanticMatchAssessmentModel)
                    .where(
                        QBankSemanticMatchAssessmentModel.semantic_run_id
                        == semantic_run_id
                    )
                    .order_by(
                        QBankSemanticMatchAssessmentModel.query_id,
                        QBankSemanticMatchAssessmentModel.qbank_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [semantic_assessment_from_row(_model_mapping(item)) for item in rows]

    async def list_metrics(
        self, semantic_run_id: str
    ) -> list[QBankSemanticMetric]:
        rows = (
            (
                await self.session.execute(
                    select(QBankSemanticMetricModel)
                    .where(
                        QBankSemanticMetricModel.semantic_run_id == semantic_run_id
                    )
                    .order_by(
                        QBankSemanticMetricModel.query_id,
                        QBankSemanticMetricModel.metric_name,
                        QBankSemanticMetricModel.question_budget,
                        QBankSemanticMetricModel.k,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [semantic_metric_from_row(_model_mapping(item)) for item in rows]
