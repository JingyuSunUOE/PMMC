"""Transactional persistence for paired baseline suites and independent evaluation runs."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.infrastructure.database.errors import (
    InvalidStateTransitionError,
    PersistenceConflictError,
)
from pmmc.infrastructure.database.evaluation_mappers import (
    answer_normalization_to_row,
    context_evaluation_to_rows,
    evaluation_attempt_from_row,
    evaluation_attempt_to_row,
    evaluation_job_from_row,
    evaluation_job_to_row,
    evaluation_manifest_from_row,
    evaluation_manifest_to_row,
    evaluation_state_from_row,
    evaluation_state_to_row,
    evaluation_target_from_row,
    evaluation_target_to_row,
    gold_group_to_rows,
    suite_manifest_from_row,
    suite_manifest_to_row,
    suite_query_from_row,
    suite_query_to_row,
    suite_run_from_row,
    suite_run_to_row,
    suite_state_from_row,
    suite_state_to_row,
)
from pmmc.infrastructure.database.evaluation_models import (
    AnswerNormalizationModel,
    BaselineSuiteModel,
    BaselineSuiteQueryModel,
    BaselineSuiteRunModel,
    BaselineSuiteStateModel,
    ContextEvaluationMatchModel,
    ContextEvaluationModel,
    EvaluationJobAttemptModel,
    EvaluationJobModel,
    EvaluationRunModel,
    EvaluationRunStateModel,
    EvaluationTargetModel,
    GoldEvidenceGroupMemberModel,
    GoldEvidenceGroupModel,
)
from pmmc.infrastructure.database.experiment_mappers import (
    bad_case_label_to_row,
    judge_score_to_row,
    metric_score_to_row,
    retrieval_evaluation_to_rows,
)
from pmmc.infrastructure.database.experiment_models import (
    BadCaseLabelModel,
    JudgeScoreModel,
    MetricScoreModel,
    RetrievalEvaluationMatchModel,
    RetrievalEvaluationModel,
    RunQueryModel,
    RunStateModel,
)
from pmmc.infrastructure.database.experiment_repositories import (
    _insert_idempotent,
    _model_mapping,
)
from pmmc.infrastructure.database.models import RunManifestModel
from pmmc.infrastructure.database.query_executor_models import QueryExecutorRunModel
from pmmc.schemas import (
    AnswerNormalization,
    BadCaseLabel,
    BaselineMethod,
    BaselineSuiteManifest,
    BaselineSuiteQuery,
    BaselineSuiteRunLink,
    BaselineSuiteState,
    BaselineSuiteStatus,
    ContextEvidenceEvaluation,
    EvaluationJob,
    EvaluationJobAttempt,
    EvaluationRunManifest,
    EvaluationRunState,
    EvaluationRunStatus,
    EvaluationSourceKind,
    EvaluationStage,
    EvaluationTarget,
    JobStatus,
    JudgeScore,
    MetricScore,
    ResolvedGoldEvidenceGroup,
    RetrievalEvaluation,
    RunManifest,
    RunStatus,
)
from pmmc.schemas.base import utc_now

_RAG_METHODS = {
    BaselineMethod.CAPTION_TEXT_RAG.value,
    BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG.value,
    BaselineMethod.NATIVE_MULTIMODAL_RAG.value,
}
_ANSWER_METRICS = {
    "exact_match_trimmed",
    "normalized_exact_match",
    "token_f1",
    "bleu1",
}


class PostgresBaselineSuiteRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(
        self,
        manifest: BaselineSuiteManifest,
        queries: list[BaselineSuiteQuery],
    ) -> BaselineSuiteState:
        if [item.ordinal for item in queries] != list(range(len(queries))):
            raise ValueError("suite query ordinals must be contiguous and ordered from zero")
        if not queries:
            raise ValueError("baseline suite denominator cannot be empty")
        if any(
            item.suite_id != manifest.suite_id or item.snapshot_id != manifest.snapshot_id
            for item in queries
        ):
            raise ValueError("suite query does not belong to manifest")
        await _insert_idempotent(
            self.session,
            BaselineSuiteModel,
            suite_manifest_to_row(manifest),
            manifest.suite_id,
        )
        state = BaselineSuiteState(
            suite_id=manifest.suite_id,
            selected_count=len(queries),
        )
        stored = await self.session.get(BaselineSuiteStateModel, manifest.suite_id)
        if stored is None:
            await _insert_idempotent(
                self.session,
                BaselineSuiteStateModel,
                suite_state_to_row(state),
                manifest.suite_id,
            )
        else:
            state = suite_state_from_row(_model_mapping(stored))
            if state.selected_count != len(queries):
                raise PersistenceConflictError("suite denominator has a different size")
        for query in queries:
            await _insert_idempotent(
                self.session,
                BaselineSuiteQueryModel,
                suite_query_to_row(query),
                query.suite_id,
                query.snapshot_id,
                query.query_id,
            )
        return state

    async def get_manifest(self, suite_id: str) -> BaselineSuiteManifest:
        value = await self.session.get(BaselineSuiteModel, suite_id)
        if value is None:
            raise KeyError(f"baseline suite does not exist: {suite_id}")
        return suite_manifest_from_row(_model_mapping(value))

    async def get_state(self, suite_id: str) -> BaselineSuiteState:
        value = await self.session.get(BaselineSuiteStateModel, suite_id)
        if value is None:
            raise KeyError(f"baseline suite does not exist: {suite_id}")
        return suite_state_from_row(_model_mapping(value))

    async def get_queries(self, suite_id: str) -> list[BaselineSuiteQuery]:
        values = (
            (
                await self.session.execute(
                    select(BaselineSuiteQueryModel)
                    .where(BaselineSuiteQueryModel.suite_id == suite_id)
                    .order_by(BaselineSuiteQueryModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return [suite_query_from_row(_model_mapping(value)) for value in values]

    async def get_run_links(self, suite_id: str) -> list[BaselineSuiteRunLink]:
        values = (
            (
                await self.session.execute(
                    select(BaselineSuiteRunModel)
                    .where(BaselineSuiteRunModel.suite_id == suite_id)
                    .order_by(BaselineSuiteRunModel.method)
                )
            )
            .scalars()
            .all()
        )
        return [suite_run_from_row(_model_mapping(value)) for value in values]

    async def link_run(self, link: BaselineSuiteRunLink) -> BaselineSuiteRunLink:
        from pmmc.baselines.suite_contract import validate_suite_run_answer_contract

        manifest = await self.get_manifest(link.suite_id)
        if link.method not in manifest.methods:
            raise ValueError("linked method is not part of the suite")
        stored = await self.session.get(RunManifestModel, link.run_id)
        state = await self.session.get(RunStateModel, link.run_id)
        if stored is None or state is None:
            raise KeyError(f"baseline run does not exist: {link.run_id}")
        run = RunManifest.model_validate(stored.manifest_json)
        if state.status != RunStatus.COMPLETED.value:
            raise InvalidStateTransitionError("suite links require a completed baseline run")
        if BaselineMethod(run.method) is not link.method:
            raise ValueError("suite link method differs from the run manifest")
        if run.dataset.value != manifest.dataset.value:
            raise ValueError("suite and run datasets differ")
        if run.dataset_snapshot_ids != [manifest.snapshot_id]:
            raise ValueError("suite and run snapshots differ")
        if run.selection_hash != manifest.selection_hash:
            raise ValueError("suite and run query denominators differ")
        if run.answer_model_alias != manifest.answer_model_alias:
            raise ValueError("suite and run answer model aliases differ")
        answer_binding = validate_suite_run_answer_contract(manifest, run)
        if answer_binding["mode"] != "manifest":
            if link.metadata.get("answer_contract_binding", answer_binding) != answer_binding:
                raise PersistenceConflictError("suite link answer binding differs")
            link = link.model_copy(update={"metadata": {
                **link.metadata, "answer_contract_binding": answer_binding,
            }})
        expected = await self.get_queries(link.suite_id)
        actual = (
            await self.session.execute(
                select(RunQueryModel.query_id, RunQueryModel.ordinal)
                .where(RunQueryModel.run_id == link.run_id)
                .order_by(RunQueryModel.ordinal)
            )
        ).all()
        if [(item.query_id, item.ordinal) for item in expected] != list(actual):
            raise ValueError("suite and run query ordinals differ")
        await _insert_idempotent(
            self.session,
            BaselineSuiteRunModel,
            suite_run_to_row(link),
            link.suite_id,
            link.method.value,
        )
        return link

    async def transition(
        self,
        suite_id: str,
        status: BaselineSuiteStatus,
        *,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> BaselineSuiteState:
        row = await self.session.scalar(
            select(BaselineSuiteStateModel)
            .where(BaselineSuiteStateModel.suite_id == suite_id)
            .with_for_update()
        )
        if row is None:
            raise KeyError(f"baseline suite does not exist: {suite_id}")
        current = suite_state_from_row(_model_mapping(row))
        allowed = {
            BaselineSuiteStatus.CREATED: {
                BaselineSuiteStatus.RUNNING,
                BaselineSuiteStatus.FAILED,
                BaselineSuiteStatus.CANCELLED,
            },
            BaselineSuiteStatus.RUNNING: {
                BaselineSuiteStatus.READY,
                BaselineSuiteStatus.FAILED,
                BaselineSuiteStatus.CANCELLED,
            },
        }
        if status not in allowed.get(current.status, set()):
            raise InvalidStateTransitionError(
                f"cannot transition suite from {current.status.value} to {status.value}"
            )
        if status is BaselineSuiteStatus.FAILED and error_type is None:
            raise ValueError("failed baseline suites require error_type")
        now = utc_now()
        links = await self.get_run_links(suite_id)
        values: dict[str, Any] = {
            "status": status.value,
            "heartbeat_at": now,
            "completed_run_count": len(links),
            "error_type": error_type,
            "error_summary": error_summary,
        }
        if status is BaselineSuiteStatus.RUNNING:
            values["started_at"] = current.started_at or now
        if status in {
            BaselineSuiteStatus.READY,
            BaselineSuiteStatus.FAILED,
            BaselineSuiteStatus.CANCELLED,
        }:
            values["completed_at"] = now
        await self.session.execute(
            update(BaselineSuiteStateModel)
            .where(BaselineSuiteStateModel.suite_id == suite_id)
            .values(**values)
        )
        await self.session.flush()
        return await self.get_state(suite_id)


class PostgresEvaluationRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(
        self,
        manifest: EvaluationRunManifest,
        targets: list[EvaluationTarget],
        jobs: list[EvaluationJob],
    ) -> EvaluationRunState:
        if manifest.source_kind is EvaluationSourceKind.BASELINE_SUITE:
            suite = await self.session.get(BaselineSuiteStateModel, manifest.suite_id)
            if suite is None or suite.status != BaselineSuiteStatus.READY.value:
                raise InvalidStateTransitionError("evaluation requires a ready baseline suite")
        else:
            source = await self.session.get(QueryExecutorRunModel, manifest.source_run_id)
            if source is None or source.status != "completed":
                raise InvalidStateTransitionError(
                    "PMMC evaluation requires a completed query executor run"
                )
        if not targets:
            raise ValueError("evaluation requires at least one target")
        if any(
            item.evaluation_run_id != manifest.evaluation_run_id
            or item.source_kind is not manifest.source_kind
            or item.suite_id != manifest.suite_id
            for item in targets
        ):
            raise ValueError("evaluation target does not match manifest")
        if manifest.source_kind is EvaluationSourceKind.PMMC_QUERY_RUN and (
            len(targets) != 1
            or targets[0].run_id != manifest.source_run_id
            or targets[0].method != "pmmc"
        ):
            raise ValueError("PMMC evaluation requires one source-run target")
        selected_count = targets[0].selected_count
        if any(item.selected_count != selected_count for item in targets):
            raise ValueError("evaluation targets must share one denominator")
        expected_jobs = selected_count * len(targets) * len(EvaluationStage)
        if len(jobs) != expected_jobs:
            raise ValueError("evaluation requires deterministic, judge, and analyze jobs")
        await _insert_idempotent(
            self.session,
            EvaluationRunModel,
            evaluation_manifest_to_row(manifest),
            manifest.evaluation_run_id,
        )
        state = EvaluationRunState(
            evaluation_run_id=manifest.evaluation_run_id,
            selected_count=selected_count,
            target_count=len(targets),
        )
        stored = await self.session.get(EvaluationRunStateModel, manifest.evaluation_run_id)
        if stored is None:
            await _insert_idempotent(
                self.session,
                EvaluationRunStateModel,
                evaluation_state_to_row(state),
                manifest.evaluation_run_id,
            )
        else:
            state = evaluation_state_from_row(_model_mapping(stored))
            if (state.selected_count, state.target_count) != (
                selected_count,
                len(targets),
            ):
                raise PersistenceConflictError("evaluation denominator differs")
        for target in targets:
            await _insert_idempotent(
                self.session,
                EvaluationTargetModel,
                evaluation_target_to_row(target),
                target.evaluation_run_id,
                target.run_id,
            )
        for job in jobs:
            await _insert_idempotent(
                self.session,
                EvaluationJobModel,
                evaluation_job_to_row(job),
                job.evaluation_job_id,
            )
        return state

    async def get_manifest(self, evaluation_run_id: str) -> EvaluationRunManifest:
        value = await self.session.get(EvaluationRunModel, evaluation_run_id)
        if value is None:
            raise KeyError(f"evaluation run does not exist: {evaluation_run_id}")
        return evaluation_manifest_from_row(_model_mapping(value))

    async def get_state(self, evaluation_run_id: str) -> EvaluationRunState:
        value = await self.session.get(EvaluationRunStateModel, evaluation_run_id)
        if value is None:
            raise KeyError(f"evaluation run does not exist: {evaluation_run_id}")
        return evaluation_state_from_row(_model_mapping(value))

    async def get_targets(self, evaluation_run_id: str) -> list[EvaluationTarget]:
        values = (
            (
                await self.session.execute(
                    select(EvaluationTargetModel)
                    .where(EvaluationTargetModel.evaluation_run_id == evaluation_run_id)
                    .order_by(EvaluationTargetModel.method)
                )
            )
            .scalars()
            .all()
        )
        return [evaluation_target_from_row(_model_mapping(value)) for value in values]

    async def get_jobs(
        self,
        evaluation_run_id: str,
        *,
        stage: EvaluationStage | None = None,
    ) -> list[EvaluationJob]:
        statement = select(EvaluationJobModel).where(
            EvaluationJobModel.evaluation_run_id == evaluation_run_id
        )
        if stage is not None:
            statement = statement.where(EvaluationJobModel.stage == stage.value)
        values = (
            (
                await self.session.execute(
                    statement.order_by(
                        EvaluationJobModel.stage,
                        EvaluationJobModel.run_id,
                        EvaluationJobModel.query_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [evaluation_job_from_row(_model_mapping(value)) for value in values]

    async def stage_status_counts(
        self,
        evaluation_run_id: str,
        stage: EvaluationStage,
    ) -> dict[JobStatus, int]:
        rows = await self.session.execute(
            select(EvaluationJobModel.status, func.count())
            .where(
                EvaluationJobModel.evaluation_run_id == evaluation_run_id,
                EvaluationJobModel.stage == stage.value,
            )
            .group_by(EvaluationJobModel.status)
        )
        counts = {status: 0 for status in JobStatus}
        for status, count in rows:
            counts[JobStatus(status)] = int(count)
        return counts

    async def all_jobs_succeeded(self, evaluation_run_id: str) -> bool:
        total = int(
            await self.session.scalar(
                select(func.count())
                .select_from(EvaluationJobModel)
                .where(EvaluationJobModel.evaluation_run_id == evaluation_run_id)
            )
            or 0
        )
        succeeded = int(
            await self.session.scalar(
                select(func.count())
                .select_from(EvaluationJobModel)
                .where(
                    EvaluationJobModel.evaluation_run_id == evaluation_run_id,
                    EvaluationJobModel.status == JobStatus.SUCCEEDED.value,
                )
            )
            or 0
        )
        if total == 0:
            manifest = await self.get_manifest(evaluation_run_id)
            state = await self.get_state(evaluation_run_id)
            return (
                manifest.source_kind.value == "pmmc_query_run"
                and manifest.metadata.get("no_answer_policy") == "independent-answer-abstention-v1"
                and state.selected_count == 0
            )
        return total == succeeded

    async def transition(
        self,
        evaluation_run_id: str,
        status: EvaluationRunStatus,
        *,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> EvaluationRunState:
        row = await self.session.scalar(
            select(EvaluationRunStateModel)
            .where(EvaluationRunStateModel.evaluation_run_id == evaluation_run_id)
            .with_for_update()
        )
        if row is None:
            raise KeyError(f"evaluation run does not exist: {evaluation_run_id}")
        current = evaluation_state_from_row(_model_mapping(row))
        allowed = {
            EvaluationRunStatus.CREATED: {
                EvaluationRunStatus.RUNNING,
                EvaluationRunStatus.FAILED,
                EvaluationRunStatus.CANCELLED,
            },
            EvaluationRunStatus.RUNNING: {
                EvaluationRunStatus.COMPLETED,
                EvaluationRunStatus.FAILED,
                EvaluationRunStatus.CANCELLED,
            },
        }
        if status not in allowed.get(current.status, set()):
            raise InvalidStateTransitionError(
                f"cannot transition evaluation from {current.status.value} to {status.value}"
            )
        if status is EvaluationRunStatus.FAILED and error_type is None:
            raise ValueError("failed evaluation runs require error_type")
        if status is EvaluationRunStatus.COMPLETED:
            await self._verify_completion(current)
        now = utc_now()
        values: dict[str, Any] = {
            "status": status.value,
            "heartbeat_at": now,
            "error_type": error_type,
            "error_summary": error_summary,
        }
        if status is EvaluationRunStatus.RUNNING:
            values["started_at"] = current.started_at or now
        if status is EvaluationRunStatus.COMPLETED:
            values["succeeded_count"] = current.selected_count * current.target_count
            values["failed_count"] = 0
        if status in {
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.FAILED,
            EvaluationRunStatus.CANCELLED,
        }:
            values["completed_at"] = now
        await self.session.execute(
            update(EvaluationRunStateModel)
            .where(EvaluationRunStateModel.evaluation_run_id == evaluation_run_id)
            .values(**values)
        )
        await self.session.flush()
        return await self.get_state(evaluation_run_id)

    async def _verify_completion(self, state: EvaluationRunState) -> None:
        evaluation_run_id = state.evaluation_run_id
        unfinished = await self.session.scalar(
            select(EvaluationJobModel.evaluation_job_id)
            .where(
                EvaluationJobModel.evaluation_run_id == evaluation_run_id,
                EvaluationJobModel.status != JobStatus.SUCCEEDED.value,
            )
            .limit(1)
        )
        if unfinished is not None:
            raise InvalidStateTransitionError("evaluation jobs are incomplete")
        expected = state.selected_count * state.target_count
        normalizations = int(
            await self.session.scalar(
                select(func.count())
                .select_from(AnswerNormalizationModel)
                .where(AnswerNormalizationModel.evaluation_run_id == evaluation_run_id)
            )
            or 0
        )
        contexts = int(
            await self.session.scalar(
                select(func.count())
                .select_from(ContextEvaluationModel)
                .where(ContextEvaluationModel.evaluation_run_id == evaluation_run_id)
            )
            or 0
        )
        judges = int(
            await self.session.scalar(
                select(func.count())
                .select_from(JudgeScoreModel)
                .where(JudgeScoreModel.evaluation_run_id == evaluation_run_id)
            )
            or 0
        )
        labeled = int(
            await self.session.scalar(
                select(func.count(func.distinct(BadCaseLabelModel.answer_id))).where(
                    BadCaseLabelModel.evaluation_run_id == evaluation_run_id
                )
            )
            or 0
        )
        if (normalizations, contexts, judges, labeled) != (expected,) * 4:
            raise InvalidStateTransitionError("evaluation facts do not cover the denominator")
        answer_metric_rows = await self.session.execute(
            select(MetricScoreModel.answer_id, MetricScoreModel.metric_name).where(
                MetricScoreModel.evaluation_run_id == evaluation_run_id,
                MetricScoreModel.metric_family == "answer",
            )
        )
        by_answer: dict[str, set[str]] = {}
        for answer_id, name in answer_metric_rows:
            by_answer.setdefault(answer_id, set()).add(name)
        if len(by_answer) != expected or any(
            names != _ANSWER_METRICS for names in by_answer.values()
        ):
            raise InvalidStateTransitionError("answer metric coverage is incomplete")


class PostgresEvaluationJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, evaluation_job_id: str) -> EvaluationJob | None:
        value = await self.session.get(EvaluationJobModel, evaluation_job_id)
        return evaluation_job_from_row(_model_mapping(value)) if value is not None else None

    async def claim(
        self,
        worker_id: str,
        lease_seconds: int,
        *,
        evaluation_run_id: str,
        stage: EvaluationStage | None = None,
    ) -> EvaluationJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = utc_now()
        conditions = [
            EvaluationJobModel.evaluation_run_id == evaluation_run_id,
            EvaluationJobModel.available_at <= now,
            EvaluationJobModel.attempt_count < EvaluationJobModel.max_attempts,
            or_(
                EvaluationJobModel.status == JobStatus.PENDING.value,
                and_(
                    EvaluationJobModel.status == JobStatus.RUNNING.value,
                    EvaluationJobModel.lease_expires_at < now,
                ),
            ),
        ]
        if stage is not None:
            conditions.append(EvaluationJobModel.stage == stage.value)
        candidate = await self.session.scalar(
            select(EvaluationJobModel)
            .where(*conditions)
            .order_by(
                EvaluationJobModel.priority.desc(),
                EvaluationJobModel.available_at,
                EvaluationJobModel.evaluation_job_id,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if candidate is None:
            return None
        if candidate.status == JobStatus.RUNNING.value:
            await self._finish_open_attempt(candidate, now, "LeaseExpired")
        candidate.attempt_count += 1
        candidate.status = JobStatus.RUNNING.value
        candidate.claimed_by = worker_id
        candidate.lease_expires_at = now + timedelta(seconds=lease_seconds)
        candidate.started_at = candidate.started_at or now
        candidate.finished_at = None
        candidate.error_type = None
        candidate.error_summary = None
        attempt = EvaluationJobAttempt(
            attempt_id=f"{candidate.evaluation_job_id}:attempt:{candidate.attempt_count}",
            evaluation_job_id=candidate.evaluation_job_id,
            attempt_number=candidate.attempt_count,
            worker_id=worker_id,
            status=JobStatus.RUNNING,
            started_at=now,
        )
        await _insert_idempotent(
            self.session,
            EvaluationJobAttemptModel,
            evaluation_attempt_to_row(attempt),
            attempt.attempt_id,
        )
        await self.session.flush()
        return evaluation_job_from_row(_model_mapping(candidate))

    async def heartbeat(
        self,
        evaluation_job_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> EvaluationJob:
        now = utc_now()
        value = await self.session.scalar(
            select(EvaluationJobModel)
            .where(EvaluationJobModel.evaluation_job_id == evaluation_job_id)
            .with_for_update()
        )
        if value is None:
            raise KeyError(f"evaluation job does not exist: {evaluation_job_id}")
        if value.status != JobStatus.RUNNING.value or value.claimed_by != worker_id:
            raise InvalidStateTransitionError("only the active worker can heartbeat a job")
        value.lease_expires_at = now + timedelta(seconds=lease_seconds)
        await self.session.flush()
        return evaluation_job_from_row(_model_mapping(value))

    async def finish(
        self,
        evaluation_job_id: str,
        worker_id: str,
        *,
        success: bool,
        error_type: str | None = None,
        error_summary: str | None = None,
        retry_delay_seconds: float = 0,
        retryable: bool = True,
    ) -> EvaluationJob:
        value = await self.session.scalar(
            select(EvaluationJobModel)
            .where(EvaluationJobModel.evaluation_job_id == evaluation_job_id)
            .with_for_update()
        )
        if value is None:
            raise KeyError(f"evaluation job does not exist: {evaluation_job_id}")
        if value.status != JobStatus.RUNNING.value or value.claimed_by != worker_id:
            raise InvalidStateTransitionError("only the active worker can finish a job")
        now = utc_now()
        if success:
            value.status = JobStatus.SUCCEEDED.value
            value.finished_at = now
            attempt_status = JobStatus.SUCCEEDED
        else:
            if error_type is None:
                raise ValueError("failed evaluation jobs require error_type")
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
        value.claimed_by = None
        value.lease_expires_at = None
        await self.session.execute(
            update(EvaluationJobAttemptModel)
            .where(
                EvaluationJobAttemptModel.attempt_id
                == f"{evaluation_job_id}:attempt:{value.attempt_count}"
            )
            .values(
                status=attempt_status.value,
                finished_at=now,
                error_type=error_type,
                error_summary=error_summary,
            )
        )
        await self.session.flush()
        return evaluation_job_from_row(_model_mapping(value))

    async def _finish_open_attempt(
        self, value: EvaluationJobModel, finished_at: Any, error_type: str
    ) -> None:
        await self.session.execute(
            update(EvaluationJobAttemptModel)
            .where(
                EvaluationJobAttemptModel.evaluation_job_id == value.evaluation_job_id,
                EvaluationJobAttemptModel.attempt_number == value.attempt_count,
                EvaluationJobAttemptModel.finished_at.is_(None),
            )
            .values(
                status=JobStatus.FAILED.value,
                finished_at=finished_at,
                error_type=error_type,
                error_summary="evaluation job lease expired",
            )
        )

    async def attempts(self, evaluation_job_id: str) -> list[EvaluationJobAttempt]:
        values = (
            (
                await self.session.execute(
                    select(EvaluationJobAttemptModel)
                    .where(
                        EvaluationJobAttemptModel.evaluation_job_id == evaluation_job_id
                    )
                    .order_by(EvaluationJobAttemptModel.attempt_number)
                )
            )
            .scalars()
            .all()
        )
        return [evaluation_attempt_from_row(_model_mapping(value)) for value in values]


class PostgresEvaluationFactRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_normalization(self, value: AnswerNormalization) -> AnswerNormalization:
        await _insert_idempotent(
            self.session,
            AnswerNormalizationModel,
            answer_normalization_to_row(value),
            value.normalization_id,
        )
        return value

    async def add_gold_group(
        self, value: ResolvedGoldEvidenceGroup
    ) -> ResolvedGoldEvidenceGroup:
        rows = gold_group_to_rows(value)
        await _insert_idempotent(
            self.session,
            GoldEvidenceGroupModel,
            rows["group"][0],
            value.gold_group_id,
        )
        for row in rows["members"]:
            await _insert_idempotent(
                self.session,
                GoldEvidenceGroupMemberModel,
                row,
                row["gold_group_id"],
                row["member_type"],
                row["member_id"],
            )
        return value

    async def add_context(
        self, value: ContextEvidenceEvaluation
    ) -> ContextEvidenceEvaluation:
        rows = context_evaluation_to_rows(value)
        await _insert_idempotent(
            self.session,
            ContextEvaluationModel,
            rows["evaluation"][0],
            value.context_evaluation_id,
        )
        for row in rows["matches"]:
            await _insert_idempotent(
                self.session,
                ContextEvaluationMatchModel,
                row,
                row["context_evaluation_id"],
                row["gold_group_id"],
            )
        return value

    async def add_retrieval(self, value: RetrievalEvaluation) -> RetrievalEvaluation:
        if value.evaluation_run_id is None:
            raise ValueError("Step 5D retrieval evaluations require evaluation_run_id")
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
        if value.evaluation_run_id is None:
            raise ValueError("Step 5D metrics require evaluation_run_id")
        await _insert_idempotent(
            self.session,
            MetricScoreModel,
            metric_score_to_row(value),
            value.metric_score_id,
        )
        return value

    async def add_judge(self, value: JudgeScore) -> JudgeScore:
        if value.evaluation_run_id is None:
            raise ValueError("Step 5D judge scores require evaluation_run_id")
        await _insert_idempotent(
            self.session,
            JudgeScoreModel,
            judge_score_to_row(value),
            value.judge_score_id,
        )
        return value

    async def add_bad_case(self, value: BadCaseLabel) -> BadCaseLabel:
        if value.evaluation_run_id is None:
            raise ValueError("Step 5D bad-case labels require evaluation_run_id")
        await _insert_idempotent(
            self.session,
            BadCaseLabelModel,
            bad_case_label_to_row(value),
            value.bad_case_label_id,
        )
        return value

    async def stage_complete(
        self,
        evaluation_run_id: str,
        answer_id: str,
        stage: EvaluationStage,
    ) -> bool:
        if stage is EvaluationStage.DETERMINISTIC:
            normalization = await self.session.scalar(
                select(AnswerNormalizationModel.normalization_id).where(
                    AnswerNormalizationModel.evaluation_run_id == evaluation_run_id,
                    AnswerNormalizationModel.answer_id == answer_id,
                )
            )
            metric_count = int(
                await self.session.scalar(
                    select(func.count())
                    .select_from(MetricScoreModel)
                    .where(
                        MetricScoreModel.evaluation_run_id == evaluation_run_id,
                        MetricScoreModel.answer_id == answer_id,
                        MetricScoreModel.metric_family.in_(["answer", "retrieval", "context"]),
                    )
                )
                or 0
            )
            return normalization is not None and metric_count >= len(_ANSWER_METRICS)
        if stage is EvaluationStage.JUDGE:
            return (
                await self.session.scalar(
                    select(JudgeScoreModel.judge_score_id).where(
                        JudgeScoreModel.evaluation_run_id == evaluation_run_id,
                        JudgeScoreModel.answer_id == answer_id,
                    )
                )
                is not None
            )
        return (
            await self.session.scalar(
                select(BadCaseLabelModel.bad_case_label_id).where(
                    BadCaseLabelModel.evaluation_run_id == evaluation_run_id,
                    BadCaseLabelModel.answer_id == answer_id,
                    BadCaseLabelModel.is_primary.is_(True),
                )
            )
            is not None
        )


class EvaluationUnitOfWork:
    """One evaluator-role transaction with no runtime mutation repositories."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions
        self.session: AsyncSession | None = None
        self.runs: PostgresEvaluationRunRepository
        self.jobs: PostgresEvaluationJobRepository
        self.facts: PostgresEvaluationFactRepository

    async def __aenter__(self) -> EvaluationUnitOfWork:
        self.session = self.sessions()
        await self.session.begin()
        self.runs = PostgresEvaluationRunRepository(self.session)
        self.jobs = PostgresEvaluationJobRepository(self.session)
        self.facts = PostgresEvaluationFactRepository(self.session)
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
