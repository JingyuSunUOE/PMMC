"""Recoverable evaluator-only workflow for QBank semantic matching."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.benchmarks.ids import stable_hash
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.evaluation.qbank_protocols import (
    QBANK_EVIDENCE_RESOLUTION_V2,
    QBANK_SEMANTIC_COVERAGE_PROTOCOLS,
)
from pmmc.evaluation.qbank_semantic import (
    QBankSemanticJudgeService,
    QBankSemanticMatchConfig,
    derive_qbank_semantic_metrics,
    qbank_semantic_prompt_hash,
    qbank_semantic_response_schema_hash,
    select_semantic_candidate_union,
    semantic_assessment_from_trace,
    semantic_pair_hash,
)
from pmmc.evaluation.repositories import PostgresGoldRepository
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics
from pmmc.infrastructure.database.experiment_models import EvaluationModelCallModel
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.questioner_models import (
    QBankCoverageCandidateModel,
    QBankCoverageQueryModel,
)
from pmmc.infrastructure.database.questioner_repositories import (
    PostgresQuestionerRepository,
)
from pmmc.infrastructure.database.repositories import (
    PostgresImageRepository,
    PostgresRuntimeQueryRepository,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_evaluator_object_stores
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import (
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
)
from pmmc.models.protocols import ModelCallWriter
from pmmc.models.registry import ModelCapability, ModelRegistry
from pmmc.schemas import (
    DataScope,
    DatasetName,
    JobStatus,
    ModelCallTrace,
    QBankCoverageCandidate,
    QBankCoverageRunManifest,
    QBankCoverageStatus,
    QBankSemanticJudgeDimensions,
    QBankSemanticMatchAssessment,
    QBankSemanticMatchJob,
    QBankSemanticMatchRun,
    QBankSemanticMatchStatus,
    QBankSemanticReason,
    QuestionCandidate,
    StructuredOutputItemDiagnostic,
    StructuredOutputItemStatus,
    StructuredOutputStage,
    StructuredOutputValidationError,
    TraceStatus,
)


@dataclass(frozen=True)
class QBankSemanticWorkflowResult:
    semantic_run_id: str
    coverage_run_id: str
    pair_count: int
    semantic_match_count: int
    conservative_fallback_count: int
    reused: bool


SemanticEventValue = str | int | float | bool
SemanticEventHandler = Callable[[str, dict[str, SemanticEventValue]], None]

def expected_coverage_protocol_for_semantic(semantic_protocol_version: str) -> str:
    """Return the frozen evidence-proxy parent for one semantic protocol."""

    try:
        return QBANK_SEMANTIC_COVERAGE_PROTOCOLS[semantic_protocol_version]
    except KeyError as exc:
        raise ValueError(
            "unsupported QBank semantic protocol for coverage binding"
        ) from exc


def qbank_semantic_parent_preflight_report(
    *,
    coverage: QBankCoverageRunManifest,
    config: QBankSemanticMatchConfig,
    query_count: int,
    applicable_query_count: int,
) -> dict[str, object]:
    """Validate the evaluator-only parent without exposing benchmark content."""

    expected_protocol = expected_coverage_protocol_for_semantic(
        config.protocol_version
    )
    evidence_policy = coverage.metadata.get(
        "evidence_resolution_policy_version"
    )
    issues: list[str] = []
    if coverage.status is not QBankCoverageStatus.COMPLETED:
        issues.append("coverage_not_completed")
    if coverage.protocol_version != expected_protocol:
        issues.append("coverage_protocol_mismatch")
    if evidence_policy != QBANK_EVIDENCE_RESOLUTION_V2:
        issues.append("evidence_resolution_policy_mismatch")
    if coverage.question_budgets != config.question_budgets:
        issues.append("question_budget_grid_mismatch")
    if coverage.match_ks != config.match_ks:
        issues.append("match_k_grid_mismatch")
    metadata_query_ids = coverage.metadata.get("query_ids")
    if not isinstance(metadata_query_ids, list) or len(metadata_query_ids) != query_count:
        issues.append("coverage_denominator_mismatch")
    if applicable_query_count > query_count:
        issues.append("applicable_denominator_invalid")
    return {
        "applicable_query_count": applicable_query_count,
        "coverage_protocol": coverage.protocol_version,
        "dataset": coverage.dataset,
        "evidence_resolution_policy": evidence_policy,
        "expected_coverage_protocol": expected_protocol,
        "expected_evidence_resolution_policy": (
            QBANK_EVIDENCE_RESOLUTION_V2
        ),
        "issue_types": sorted(issues),
        "query_count": query_count,
        "semantic_protocol": config.protocol_version,
        "state": "ready" if not issues else "failed",
    }


async def inspect_qbank_semantic_parent(
    *,
    settings: Settings,
    coverage_run_id: str,
    config: QBankSemanticMatchConfig,
) -> dict[str, object]:
    """Inspect one semantic parent before allocating the Judge model pool."""

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with ExperimentUnitOfWork(
            sessions, DataScope.EVALUATION
        ) as uow:
            assert uow.qbank_coverage is not None
            coverage = await uow.qbank_coverage.get_run(coverage_run_id)
        if coverage is None:
            return {
                "applicable_query_count": 0,
                "coverage_protocol": None,
                "dataset": "unknown",
                "evidence_resolution_policy": None,
                "expected_coverage_protocol": (
                    expected_coverage_protocol_for_semantic(
                        config.protocol_version
                    )
                ),
                "expected_evidence_resolution_policy": (
                    QBANK_EVIDENCE_RESOLUTION_V2
                ),
                "issue_types": ["coverage_run_missing"],
                "query_count": 0,
                "semantic_protocol": config.protocol_version,
                "state": "failed",
            }
        async with sessions() as session:
            query_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(QBankCoverageQueryModel)
                    .where(
                        QBankCoverageQueryModel.coverage_run_id
                        == coverage_run_id
                    )
                )
                or 0
            )
            applicable_query_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(QBankCoverageQueryModel)
                    .where(
                        QBankCoverageQueryModel.coverage_run_id
                        == coverage_run_id,
                        QBankCoverageQueryModel.applicable,
                    )
                )
                or 0
            )
        return qbank_semantic_parent_preflight_report(
            coverage=coverage,
            config=config,
            query_count=query_count,
            applicable_query_count=applicable_query_count,
        )
    finally:
        await engine.dispose()


async def _await_workers(tasks: Sequence[asyncio.Task[None]]) -> None:
    """Cancel peer Judge tasks before evaluator resources are disposed."""

    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class QBankSemanticWorkflowError(RuntimeError):
    """Stage-scoped semantic workflow failure without benchmark content."""

    def __init__(
        self,
        *,
        phase: str,
        pair_ordinal: int | None = None,
    ) -> None:
        self.failure_stage = f"qbank_semantic_{phase}"
        self.error_code = "qbank_semantic_workflow_failed"
        self.pair_ordinal = pair_ordinal
        super().__init__("QBank semantic workflow failed")


def qbank_semantic_failure_diagnostics(
    error: BaseException,
) -> dict[str, str]:
    """Return allow-listed persistence details and an optional pair ordinal."""

    diagnostics = safe_database_error_diagnostics(error)
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        pair_ordinal = getattr(current, "pair_ordinal", None)
        if isinstance(pair_ordinal, int) and pair_ordinal >= 0:
            diagnostics["pair_ordinal"] = str(pair_ordinal)
            break
        current = current.__cause__ or current.__context__
    return diagnostics


def _emit_semantic_event(
    handler: SemanticEventHandler | None,
    phase: str,
    **details: SemanticEventValue,
) -> None:
    if handler is not None:
        handler(phase, details)


class _EvaluationModelCallWriter(ModelCallWriter):
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def add(self, trace: ModelCallTrace) -> ModelCallTrace:
        async with ExperimentUnitOfWork(
            self.sessions, DataScope.EVALUATION
        ) as uow:
            return await uow.model_calls.add(trace)


def _semantic_job_id(
    semantic_run_id: str,
    query_id: str,
    qbank_id: str,
) -> str:
    return stable_compiler_id(
        "qbank-semantic-job", [semantic_run_id, query_id, qbank_id]
    )


def _semantic_call_id(
    semantic_run_id: str,
    query_id: str,
    qbank_id: str,
    attempt_number: int,
) -> str:
    return "qbank-semantic-call-" + stable_hash(
        [semantic_run_id, query_id, qbank_id, attempt_number]
    )[:24]


def _conservative_assessment(
    *,
    trace: ModelCallTrace,
    semantic_run_id: str,
    semantic_job_id: str,
    coverage_run_id: str,
    query_id: str,
    qbank_id: str,
    evidence_proxy_complete: bool,
) -> QBankSemanticMatchAssessment:
    dimensions = QBankSemanticJudgeDimensions(
        same_information_need=False,
        candidate_answer_sufficient=False,
        contradiction_free=False,
        query_image_binding_compatible=False,
    )
    return QBankSemanticMatchAssessment(
        semantic_assessment_id=stable_compiler_id(
            "qbank-semantic-assessment",
            [semantic_run_id, query_id, qbank_id],
        ),
        semantic_run_id=semantic_run_id,
        semantic_job_id=semantic_job_id,
        coverage_run_id=coverage_run_id,
        query_id=query_id,
        qbank_id=qbank_id,
        model_call_id=trace.model_call_id,
        dimensions=dimensions,
        reason_codes=[QBankSemanticReason.PROVIDER_OUTPUT_INVALID],
        semantic_dimensions_passed=False,
        evidence_proxy_complete=evidence_proxy_complete,
        semantic_match=False,
        elapsed_ms=trace.latency_ms or 0.0,
        content_hash=canonical_content_hash(
            {
                "semantic_run_id": semantic_run_id,
                "query_id": query_id,
                "qbank_id": qbank_id,
                "provider_output_invalid": True,
                "evidence_proxy_complete": evidence_proxy_complete,
            }
        ),
        metadata={"conservative_fallback": True},
    )


def _semantic_diagnostic(
    *,
    trace: ModelCallTrace,
    semantic_run_id: str,
    pair_hash: str,
    parser_version: str,
) -> StructuredOutputItemDiagnostic:
    return StructuredOutputItemDiagnostic(
        diagnostic_id=stable_compiler_id(
            "structured-output-diagnostic",
            [trace.model_call_id, StructuredOutputStage.QBANK_SEMANTIC_JUDGE.value, 0],
        ),
        run_id=trace.run_id,
        model_call_id=trace.model_call_id,
        semantic_run_id=semantic_run_id,
        stage=StructuredOutputStage.QBANK_SEMANTIC_JUDGE,
        item_ordinal=0,
        item_hash=pair_hash,
        status=StructuredOutputItemStatus.CONSERVATIVE_REJECT,
        parser_version=parser_version,
        repair_attempt_count=trace.repair_attempt_count,
        errors=[
            StructuredOutputValidationError(
                location="result",
                error_type=trace.error_type or "provider_output_invalid",
            )
        ],
        metadata={"safe": True},
    )


async def _load_coverage_facts(
    sessions: async_sessionmaker[AsyncSession],
    coverage_run_id: str,
) -> tuple[
    QBankCoverageRunManifest,
    list[str],
    list[QBankCoverageCandidate],
]:
    async with ExperimentUnitOfWork(
        sessions, DataScope.EVALUATION
    ) as uow:
        assert uow.qbank_coverage is not None
        coverage = await uow.qbank_coverage.get_run(coverage_run_id)
    if coverage is None:
        raise KeyError(f"QBank coverage run does not exist: {coverage_run_id}")
    if coverage.status is not QBankCoverageStatus.COMPLETED:
        raise ValueError("QBank semantic matching requires completed coverage")
    async with sessions() as session:
        query_ids = list(
            await session.scalars(
                select(QBankCoverageQueryModel.query_id)
                .where(
                    QBankCoverageQueryModel.coverage_run_id == coverage_run_id,
                    QBankCoverageQueryModel.applicable,
                )
                .order_by(QBankCoverageQueryModel.ordinal)
            )
        )
        candidate_models = list(
            await session.scalars(
                select(QBankCoverageCandidateModel)
                .where(
                    QBankCoverageCandidateModel.coverage_run_id
                    == coverage_run_id,
                    QBankCoverageCandidateModel.rank <= 50,
                )
                .order_by(
                    QBankCoverageCandidateModel.query_id,
                    QBankCoverageCandidateModel.question_budget,
                    QBankCoverageCandidateModel.rank,
                )
            )
        )
    from pmmc.infrastructure.database.compiler_repositories import _model_mapping
    from pmmc.infrastructure.database.questioner_mappers import (
        coverage_candidate_from_row,
    )

    return (
        coverage,
        query_ids,
        [
            coverage_candidate_from_row(_model_mapping(item))
            for item in candidate_models
        ],
    )


async def run_qbank_semantic_match(
    *,
    settings: Settings,
    registry: ModelRegistry,
    coverage_run_id: str,
    config: QBankSemanticMatchConfig,
    semantic_run_id: str | None = None,
    on_pair: Callable[[str, str, int, int], None] | None = None,
    on_event: SemanticEventHandler | None = None,
) -> QBankSemanticWorkflowResult:
    """Judge the deduplicated union of all per-budget top-50 candidate pairs."""

    entry = registry.get(config.judge_model_alias)
    entry.require(ModelCapability.CHAT_MULTIMODAL)
    _emit_semantic_event(
        on_event, "schema_preflight", protocol_version=config.protocol_version,
        run_schema_version=config.run_schema_version, provider_schema_version=config.schema_version,
    )
    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    workflow_phase = "load_inputs"
    try:
        coverage, query_ids, ranked_candidates = await _load_coverage_facts(
            sessions, coverage_run_id
        )
        metadata_query_ids = coverage.metadata.get("query_ids")
        parent_report = qbank_semantic_parent_preflight_report(
            coverage=coverage,
            config=config,
            query_count=(
                len(metadata_query_ids)
                if isinstance(metadata_query_ids, list)
                else len(query_ids)
            ),
            applicable_query_count=len(query_ids),
        )
        if parent_report["state"] != "ready":
            raise ValueError(
                "semantic Judge parent preflight failed: "
                + ",".join(parent_report["issue_types"])  # type: ignore[arg-type]
            )

        runtime_queries = {
            item.query.query_id: item.query
            for item in await PostgresRuntimeQueryRepository(
                sessions
            ).list_snapshot(coverage.snapshot_id)
        }
        async with sessions() as session:
            questioner_repository = PostgresQuestionerRepository(session)
            questions = await questioner_repository.list_eligible_questions(
                coverage.compiler_run_id
            )
            expected_evidence = await questioner_repository.list_expected_evidence(
                coverage.compiler_run_id
            )
        question_by_id: dict[str, QuestionCandidate] = {
            item.qbank_id: item for item in questions
        }
        evidence_image_ids = sorted(
            {
                item.image_id
                for values in expected_evidence.values()
                for item in values
                if item.image_id is not None
            }
        )
        evidence_images = {
            item.image_id: item
            for item in await PostgresImageRepository(sessions).get_many(
                coverage.snapshot_id,
                evidence_image_ids,
            )
        }
        candidate_evidence_images = {
            qbank_id: tuple(
                evidence_images[image_id]
                for image_id in dict.fromkeys(
                    item.image_id
                    for item in values
                    if item.image_id is not None and item.image_id in evidence_images
                )
            )
            for qbank_id, values in expected_evidence.items()
        }
        gold_repository = PostgresGoldRepository(sessions)
        gold_answers: dict[str, str] = {}
        for query_id in query_ids:
            gold = await gold_repository.get(coverage.snapshot_id, query_id)
            if gold.answer is not None:
                gold_answers[query_id] = gold.answer

        union = select_semantic_candidate_union(
            ranked_candidates, candidate_pool_k=config.candidate_pool_k
        )
        proxy_by_pair: dict[tuple[str, str], bool] = {}
        for item in ranked_candidates:
            key = (item.query_id, item.qbank_id)
            proxy_by_pair[key] = (
                proxy_by_pair.get(key, False) or item.individually_sufficient
            )
        pairs = [
            item
            for item in union
            if item.query_id in gold_answers
            and item.query_id in runtime_queries
            and item.qbank_id in question_by_id
            and question_by_id[item.qbank_id].expected_answer is not None
        ]
        _emit_semantic_event(
            on_event,
            "prepared",
            query_count=len(query_ids),
            ranked_candidate_count=len(ranked_candidates),
            pair_count=len(pairs),
        )
        config_identity = canonical_content_hash(
            {
                "coverage_run_id": coverage_run_id,
                "config_hash": config.config_hash,
                "judge_model_hash": entry.resolved_hash,
                "prompt_hash": qbank_semantic_prompt_hash(config),
                "pair_keys": [
                    [item.query_id, item.qbank_id]
                    for item in sorted(
                        pairs, key=lambda value: (value.query_id, value.qbank_id)
                    )
                ],
            }
        )
        resolved_run_id = semantic_run_id or stable_compiler_id(
            "qbank-semantic-run", [coverage_run_id, config_identity]
        )
        workflow_phase = "validate_run_manifest"
        manifest = QBankSemanticMatchRun(
            semantic_run_id=resolved_run_id,
            schema_version=config.run_schema_version,
            protocol_version=config.protocol_version,
            coverage_run_id=coverage_run_id,
            compiler_run_id=coverage.compiler_run_id,
            snapshot_id=coverage.snapshot_id,
            dataset=DatasetName(coverage.dataset),
            judge_model_alias=config.judge_model_alias,
            judge_model_revision=entry.model_revision,
            prompt_version=config.prompt_version,
            prompt_hash=qbank_semantic_prompt_hash(config),
            renderer_version=config.renderer_version,
            parser_version=config.parser_version,
            config_hash=config_identity,
            pair_count=len(pairs),
            metadata={
                "query_count": len(query_ids),
                "gold_answer_query_count": len(gold_answers),
                "candidate_pool_k": config.candidate_pool_k,
                "semantic_config_hash": config.config_hash,
                "semantic_response_schema_hash": qbank_semantic_response_schema_hash(
                    config
                ),
                "candidate_evidence_image_count": len(evidence_image_ids),
            },
        )
        workflow_phase = "register_run_jobs"
        async with ExperimentUnitOfWork(
            sessions, DataScope.EVALUATION
        ) as uow:
            assert uow.qbank_semantic is not None
            existing = await uow.qbank_semantic.get_run(resolved_run_id)
            if existing is None:
                await uow.qbank_semantic.add_run(manifest)
                await uow.qbank_semantic.transition_run(
                    resolved_run_id, QBankSemanticMatchStatus.RUNNING
                )
            else:
                identity = manifest.model_copy(
                    update={
                        "created_at": existing.created_at,
                        "status": existing.status,
                        "completed_at": existing.completed_at,
                    }
                )
                if existing != identity:
                    raise ValueError(
                        "semantic resume conflicts with its frozen identity"
                    )
                if existing.status is QBankSemanticMatchStatus.COMPLETED:
                    assessments = (
                        await uow.qbank_semantic.list_assessments(resolved_run_id)
                    )
                    return QBankSemanticWorkflowResult(
                        semantic_run_id=resolved_run_id,
                        coverage_run_id=coverage_run_id,
                        pair_count=len(assessments),
                        semantic_match_count=sum(
                            item.semantic_match for item in assessments
                        ),
                        conservative_fallback_count=sum(
                            item.metadata.get("conservative_fallback") is True
                            for item in assessments
                        ),
                        reused=True,
                    )
                if existing.status is QBankSemanticMatchStatus.CREATED:
                    await uow.qbank_semantic.transition_run(
                        resolved_run_id, QBankSemanticMatchStatus.RUNNING
                    )
                elif existing.status is not QBankSemanticMatchStatus.RUNNING:
                    raise ValueError(
                        "semantic run cannot resume from a terminal failure"
                    )
            for pair in pairs:
                job = QBankSemanticMatchJob(
                    semantic_job_id=_semantic_job_id(
                        resolved_run_id, pair.query_id, pair.qbank_id
                    ),
                    semantic_run_id=resolved_run_id,
                    coverage_run_id=coverage_run_id,
                    query_id=pair.query_id,
                    qbank_id=pair.qbank_id,
                    pair_hash=semantic_pair_hash(
                        semantic_run_id=resolved_run_id,
                        query_id=pair.query_id,
                        qbank_id=pair.qbank_id,
                    ),
                    max_attempts=2,
                )
                stored = await uow.qbank_semantic.get_job(
                    resolved_run_id, pair.query_id, pair.qbank_id
                )
                if stored is None:
                    await uow.qbank_semantic.add_job(job)
                elif stored.pair_hash != job.pair_hash:
                    raise ValueError("semantic job pair identity conflicts")
        _emit_semantic_event(
            on_event,
            "jobs_registered",
            pair_count=len(pairs),
        )

        workflow_phase = "initialize_judge"
        runtime_store, evaluation_store = create_evaluator_object_stores(settings)
        artifacts = ExperimentArtifactService(
            sessions,
            evaluation_store,
            DataScope.EVALUATION,
            artifact_prefix=settings.experiment_artifact_prefix,
        )
        service = QBankSemanticJudgeService(
            execution=create_model_execution_service(
                settings,
                registry,
                artifact_service=artifacts,
                model_calls=_EvaluationModelCallWriter(sessions),
                data_scope=DataScope.EVALUATION,
            ),
            gateway=create_generation_gateway(
                settings,
                registry,
                config.judge_model_alias,
                image_url_resolver=create_image_url_resolver(
                    settings, runtime_store
                ),
            ),
            config=config,
            timeout_seconds=settings.model_request_timeout_seconds,
        )
        semaphore = asyncio.Semaphore(config.concurrency)
        completed = 0
        completed_lock = asyncio.Lock()

        async def judge_pair(
            pair: QBankCoverageCandidate,
            pair_ordinal: int,
        ) -> None:
            job_id = _semantic_job_id(
                resolved_run_id, pair.query_id, pair.qbank_id
            )
            async with semaphore:
                async with ExperimentUnitOfWork(
                    sessions, DataScope.EVALUATION
                ) as uow:
                    assert uow.qbank_semantic is not None
                    job = await uow.qbank_semantic.get_job_by_id(job_id)
                    assessment = await uow.qbank_semantic.get_assessment(job_id)
                    if job is None:
                        raise RuntimeError("semantic job disappeared")
                    if assessment is not None:
                        if job.status is JobStatus.RUNNING:
                            await uow.qbank_semantic.transition_job(
                                job_id,
                                JobStatus.SUCCEEDED,
                                model_call_id=assessment.model_call_id,
                            )
                        elif job.status is not JobStatus.SUCCEEDED:
                            raise RuntimeError(
                                "semantic assessment has a non-successful job"
                            )
                        return
                    if job.status is JobStatus.PENDING:
                        job = await uow.qbank_semantic.transition_job(
                            job_id, JobStatus.RUNNING
                        )
                    elif job.status is not JobStatus.RUNNING:
                        raise RuntimeError(
                            "semantic job cannot resume from terminal failure"
                        )
                    attempt_number = max(job.attempt_count, 1)
                    model_call_id = _semantic_call_id(
                        resolved_run_id,
                        pair.query_id,
                        pair.qbank_id,
                        attempt_number,
                    )
                    trace = await uow.model_calls.get(model_call_id)

                if trace is None:
                    try:
                        assessment = await service.assess(
                            semantic_run_id=resolved_run_id,
                            semantic_job_id=job_id,
                            coverage_run_id=coverage_run_id,
                            compiler_run_id=coverage.compiler_run_id,
                            snapshot_id=coverage.snapshot_id,
                            query=runtime_queries[pair.query_id],
                            candidate=question_by_id[pair.qbank_id],
                            gold_answer=gold_answers[pair.query_id],
                            evidence_images=candidate_evidence_images.get(
                                pair.qbank_id, ()
                            ),
                            evidence_proxy_complete=proxy_by_pair[
                                (pair.query_id, pair.qbank_id)
                            ],
                            attempt_number=attempt_number,
                        )
                    except Exception as exc:
                        async with ExperimentUnitOfWork(
                            sessions, DataScope.EVALUATION
                        ) as uow:
                            trace = await uow.model_calls.get(model_call_id)
                        if trace is None:
                            raise QBankSemanticWorkflowError(
                                phase="judge_model_call",
                                pair_ordinal=pair_ordinal,
                            ) from exc
                if trace is not None:
                    if (
                        trace.status is TraceStatus.SUCCESS
                        and trace.parsed_output is not None
                    ):
                        assessment = semantic_assessment_from_trace(
                            trace=trace,
                            semantic_run_id=resolved_run_id,
                            semantic_job_id=job_id,
                            coverage_run_id=coverage_run_id,
                            query_id=pair.query_id,
                            qbank_id=pair.qbank_id,
                            evidence_proxy_complete=proxy_by_pair[
                                (pair.query_id, pair.qbank_id)
                            ],
                            config=config,
                        )
                    else:
                        assessment = _conservative_assessment(
                            trace=trace,
                            semantic_run_id=resolved_run_id,
                            semantic_job_id=job_id,
                            coverage_run_id=coverage_run_id,
                            query_id=pair.query_id,
                            qbank_id=pair.qbank_id,
                            evidence_proxy_complete=proxy_by_pair[
                                (pair.query_id, pair.qbank_id)
                            ],
                        )
                if assessment is None:
                    raise RuntimeError("semantic assessment was not materialized")
                try:
                    async with ExperimentUnitOfWork(
                        sessions, DataScope.EVALUATION
                    ) as uow:
                        assert uow.qbank_semantic is not None
                        await uow.qbank_semantic.add_assessment(assessment)
                        if (
                            assessment.metadata.get("conservative_fallback")
                            is True
                        ):
                            assert trace is not None
                            await uow.structured_output_diagnostics.add(
                                _semantic_diagnostic(
                                    trace=trace,
                                    semantic_run_id=resolved_run_id,
                                    pair_hash=semantic_pair_hash(
                                        semantic_run_id=resolved_run_id,
                                        query_id=pair.query_id,
                                        qbank_id=pair.qbank_id,
                                    ),
                                    parser_version=config.parser_version,
                                )
                            )
                        await uow.qbank_semantic.transition_job(
                            job_id,
                            JobStatus.SUCCEEDED,
                            model_call_id=assessment.model_call_id,
                        )
                except Exception as exc:
                    raise QBankSemanticWorkflowError(
                        phase="persist_assessment",
                        pair_ordinal=pair_ordinal,
                    ) from exc

        async def process_pair(pair: QBankCoverageCandidate, pair_ordinal: int) -> None:
            nonlocal completed
            await judge_pair(pair, pair_ordinal)
            # Count verified reused assessments as well as newly persisted pairs.
            async with completed_lock:
                completed += 1
                if on_pair is not None:
                    on_pair(pair.query_id, pair.qbank_id, completed, len(pairs))

        workflow_phase = "judge_pairs"
        _emit_semantic_event(
            on_event,
            "judge_started",
            pair_count=len(pairs),
            concurrency=config.concurrency,
        )
        await _await_workers(
            [
                asyncio.create_task(process_pair(pair, pair_ordinal))
                for pair_ordinal, pair in enumerate(pairs)
            ]
        )
        _emit_semantic_event(
            on_event,
            "judge_completed",
            pair_count=len(pairs),
        )
        workflow_phase = "finalize"
        async with ExperimentUnitOfWork(
            sessions, DataScope.EVALUATION
        ) as uow:
            assert uow.qbank_semantic is not None
            jobs = await uow.qbank_semantic.list_jobs(resolved_run_id)
            assessments = await uow.qbank_semantic.list_assessments(
                resolved_run_id
            )
            if len(jobs) != len(pairs) or any(
                item.status is not JobStatus.SUCCEEDED for item in jobs
            ):
                raise RuntimeError("semantic job denominator is incomplete")
            if len(assessments) != len(pairs):
                raise RuntimeError("semantic assessment denominator is incomplete")
            await uow.qbank_semantic.add_metrics(
                list(
                    derive_qbank_semantic_metrics(
                        semantic_run_id=resolved_run_id,
                        candidates=ranked_candidates,
                        assessments=assessments,
                        config=config,
                        query_ids=query_ids,
                    )
                )
            )
            await uow.qbank_semantic.transition_run(
                resolved_run_id, QBankSemanticMatchStatus.COMPLETED
            )
        _emit_semantic_event(
            on_event,
            "completed",
            pair_count=len(assessments),
            semantic_match_count=sum(
                item.semantic_match for item in assessments
            ),
        )
        return QBankSemanticWorkflowResult(
            semantic_run_id=resolved_run_id,
            coverage_run_id=coverage_run_id,
            pair_count=len(assessments),
            semantic_match_count=sum(item.semantic_match for item in assessments),
            conservative_fallback_count=sum(
                item.metadata.get("conservative_fallback") is True
                for item in assessments
            ),
            reused=False,
        )
    except ValidationError as exc:
        # Never render Pydantic input dictionaries in the public CLI error.
        raise QBankSemanticWorkflowError(phase=workflow_phase) from exc
    except (QBankSemanticWorkflowError, ValueError):
        raise
    except Exception as exc:
        raise QBankSemanticWorkflowError(phase=workflow_phase) from exc
    finally:
        await engine.dispose()


async def verify_qbank_semantic_match_run(
    settings: Settings,
    semantic_run_id: str,
    config: QBankSemanticMatchConfig,
) -> dict[str, object]:
    """Verify denominator, job/call lineage, metric grid, and safe fallbacks."""

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    issues: list[str] = []
    try:
        async with ExperimentUnitOfWork(
            sessions, DataScope.EVALUATION
        ) as uow:
            assert uow.qbank_semantic is not None
            run = await uow.qbank_semantic.get_run(semantic_run_id)
            if run is None:
                raise KeyError(
                    f"QBank semantic run does not exist: {semantic_run_id}"
                )
            jobs = await uow.qbank_semantic.list_jobs(semantic_run_id)
            assessments = await uow.qbank_semantic.list_assessments(
                semantic_run_id
            )
            metrics = await uow.qbank_semantic.list_metrics(semantic_run_id)
        if run.status is not QBankSemanticMatchStatus.COMPLETED:
            issues.append("semantic run is not completed")
        if (run.protocol_version != config.protocol_version
                or run.schema_version != config.run_schema_version):
            issues.append("semantic run protocol/schema differs from verification config")
        if len(jobs) != run.pair_count:
            issues.append("semantic job denominator differs from the manifest")
        if len(assessments) != run.pair_count:
            issues.append(
                "semantic assessment denominator differs from the manifest"
            )
        if any(item.status is not JobStatus.SUCCEEDED for item in jobs):
            issues.append("semantic run contains a non-successful pair job")
        if len({(item.query_id, item.qbank_id) for item in jobs}) != len(jobs):
            issues.append("semantic pair jobs are duplicated")
        call_ids = {item.model_call_id for item in assessments}
        if call_ids != {
            item.model_call_id for item in jobs if item.model_call_id is not None
        }:
            issues.append("semantic job/model-call/assessment lineage differs")
        fallback_count = sum(
            item.metadata.get("conservative_fallback") is True
            for item in assessments
        )
        async with sessions() as session:
            query_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(QBankCoverageQueryModel)
                    .where(
                        QBankCoverageQueryModel.coverage_run_id
                        == run.coverage_run_id,
                        QBankCoverageQueryModel.applicable,
                    )
                )
                or 0
            )
        expected_metric_count = (
            query_count
            * len(config.question_budgets)
            * len(config.match_ks)
            * 2
        )
        if len(metrics) != expected_metric_count:
            issues.append("semantic metric grid is incomplete")
        if any(item.metric_version != config.protocol_version for item in metrics):
            issues.append("semantic metric protocol differs from verification config")
        async with sessions() as session:
            traces = list(
                await session.scalars(
                    select(EvaluationModelCallModel).where(
                        EvaluationModelCallModel.semantic_run_id == semantic_run_id
                    )
                )
            )
        if len(traces) != run.pair_count:
            issues.append("semantic model-call denominator is incomplete")
        if any(
            item.semantic_job_id is None
            or item.evaluation_run_id is not None
            or item.model_alias != run.judge_model_alias
            for item in traces
        ):
            issues.append("semantic model-call parent or model identity is invalid")
        return {
            "semantic_run_id": semantic_run_id,
            "coverage_run_id": run.coverage_run_id,
            "pair_count": run.pair_count,
            "semantic_match_count": sum(
                item.semantic_match for item in assessments
            ),
            "conservative_fallback_count": fallback_count,
            "structured_provider_fallback_zero": fallback_count == 0,
            "valid": not issues,
            "issues": issues,
        }
    finally:
        await engine.dispose()


__all__ = [
    "QBankSemanticWorkflowError",
    "QBankSemanticWorkflowResult",
    "qbank_semantic_failure_diagnostics",
    "run_qbank_semantic_match",
    "verify_qbank_semantic_match_run",
]
