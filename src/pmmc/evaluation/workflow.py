"""Recoverable batch orchestration for independent baseline evaluation runs."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.benchmarks.ids import stable_hash
from pmmc.evaluation.config import (
    BaselineEvaluationConfig,
    evaluation_prompt_sha256,
    load_evaluation_config,
    load_judge_prompt,
)
from pmmc.evaluation.errors import safe_error_diagnostics, safe_error_summary
from pmmc.evaluation.facts import PostgresEvaluationCaseLoader
from pmmc.evaluation.graph import BaselineEvaluationGraph
from pmmc.evaluation.judge import BaselineJudgeService
from pmmc.evaluation.metrics import ANSWER_METRIC_VERSIONS
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.evaluation_models import (
    AnswerNormalizationModel,
    ContextEvaluationModel,
    EvaluationJobModel,
)
from pmmc.infrastructure.database.evaluation_repositories import (
    EvaluationUnitOfWork,
    PostgresBaselineSuiteRepository,
)
from pmmc.infrastructure.database.experiment_models import (
    BadCaseLabelModel,
    EvaluationArtifactModel,
    EvaluationModelCallModel,
    FinalAnswerModel,
    JudgeScoreModel,
    MetricScoreModel,
    RunQueryModel,
    RuntimeArtifactModel,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.query_executor_models import (
    PmmcFallbackQueryLinkModel,
    QueryAnswerSelectionModel,
    QueryEvidenceSufficiencyAssessmentModel,
    QueryExecutorRunModel,
    QueryProgramBindingModel,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_evaluator_object_stores
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import (
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
)
from pmmc.models.registry import load_model_registry
from pmmc.schemas import (
    AppStage,
    AssetType,
    BaselineMethod,
    BaselineSuiteStatus,
    DataScope,
    EvaluationJob,
    EvaluationRunManifest,
    EvaluationRunStatus,
    EvaluationSourceKind,
    EvaluationStage,
    EvaluationTarget,
    JobStatus,
    ModelCallTrace,
    ModelCapability,
)
from pmmc.schemas.base import utc_now

_STEP7B_EVALUATION_VARIANTS = (
    "pmmc_r0",
    "pmmc_r1",
    "pmmc_r2",
    "pmmc_r3",
    "native_mm_counterfactual",
    "active_final_provisional",
    "no_fallback_counterfactual",
)
_STEP7B_REQUIRED_EVALUATION_VARIANTS = frozenset(
    {"native_mm_counterfactual", "active_final_provisional"}
)


def _evaluation_failure_is_retryable(
    error: Exception,
    diagnostics: dict[str, str],
) -> bool:
    """Retry only failures that can plausibly change without a code or data repair."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        retryable = getattr(current, "retryable", None)
        if isinstance(retryable, bool):
            return retryable
        current = current.__cause__ or current.__context__
    sqlstate = diagnostics.get("sqlstate", "")
    if sqlstate.startswith(("22", "23", "42")):
        return False
    if isinstance(error, (AssertionError, KeyError, TypeError, ValueError)):
        return False
    return True


async def _preflight_pmmc_evaluation_context_lineage(
    sessions: async_sessionmaker[AsyncSession],
    *,
    evaluation_run_id: str,
    query_executor_run_id: str,
) -> None:
    """Fail once before fan-out when active answers lack runtime context lineage."""

    async with sessions() as session:
        variant_rows = (
            await session.execute(
                select(EvaluationJobModel.answer_variant, func.count())
                .where(
                    EvaluationJobModel.evaluation_run_id == evaluation_run_id,
                    EvaluationJobModel.stage == EvaluationStage.DETERMINISTIC.value,
                )
                .group_by(EvaluationJobModel.answer_variant)
            )
        ).all()
        fallback_link_count = int(
            await session.scalar(
                select(func.count())
                .select_from(PmmcFallbackQueryLinkModel)
                .where(
                    PmmcFallbackQueryLinkModel.query_executor_run_id
                    == query_executor_run_id
                )
            )
            or 0
        )
        program_query_count = int(
            await session.scalar(
                select(func.count(func.distinct(QueryEvidenceSufficiencyAssessmentModel.query_id)))
                .join(
                    QueryProgramBindingModel,
                    QueryProgramBindingModel.query_program_binding_id
                    == QueryEvidenceSufficiencyAssessmentModel.query_program_binding_id,
                )
                .where(
                    QueryEvidenceSufficiencyAssessmentModel.query_executor_run_id
                    == query_executor_run_id,
                    QueryProgramBindingModel.round_id.in_((0, 1, 2, 3)),
                )
            )
            or 0
        )
        generated_variant_rows = (
            await session.execute(
                select(FinalAnswerModel.answer_variant, func.count())
                .where(FinalAnswerModel.run_id == query_executor_run_id)
                .group_by(FinalAnswerModel.answer_variant)
            )
        ).all()
        program_binding_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QueryProgramBindingModel)
                .where(
                    QueryProgramBindingModel.query_executor_run_id
                    == query_executor_run_id,
                    QueryProgramBindingModel.round_id.in_((0, 1, 2, 3)),
                )
            )
            or 0
        )
        sufficient_program_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QueryEvidenceSufficiencyAssessmentModel)
                .where(
                    QueryEvidenceSufficiencyAssessmentModel.query_executor_run_id
                    == query_executor_run_id,
                    QueryEvidenceSufficiencyAssessmentModel.sufficient.is_(True),
                )
            )
            or 0
        )
    variant_counts = {str(variant): int(count) for variant, count in variant_rows}
    generated_variant_counts = {
        str(variant): int(count) for variant, count in generated_variant_rows
    }
    supported = {*_STEP7B_EVALUATION_VARIANTS, "active_final"}
    unsupported_count = sum(
        count for variant, count in variant_counts.items() if variant not in supported
    )
    fallback_expected = variant_counts.get("native_mm_counterfactual", 0)
    program_expected = sum(
        count for variant, count in variant_counts.items() if variant.startswith("pmmc_r")
    )
    LOGGER.info(
        "evaluation_context_lineage_preflight",
        evaluation_run_id=evaluation_run_id,
        contract_version="typed-evaluation-context-v1",
        evaluation_scope="variant_aware",
        variant_counts=dict(sorted(variant_counts.items())),
        generated_answer_variant_counts=dict(sorted(generated_variant_counts.items())),
        fallback_expected=fallback_expected,
        fallback_link_count=fallback_link_count,
        program_expected=program_expected,
        program_binding_count=program_binding_count,
        program_execution_query_count=program_query_count,
        sufficient_program_count=sufficient_program_count,
        unsupported_count=unsupported_count,
    )
    if unsupported_count:
        raise ValueError("PMMC evaluation contains unsupported active answer variants")
    if fallback_link_count < fallback_expected:
        raise ValueError("PMMC evaluation fallback context lineage is incomplete")
    if program_expected and (program_query_count == 0 or program_binding_count == 0):
        raise ValueError("PMMC evaluation program execution lineage is incomplete")

LOGGER = structlog.get_logger(__name__)


class EvaluationRunPhase(StrEnum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"
    ANALYZE = "analyze"
    ALL = "all"

    @property
    def stages(self) -> tuple[EvaluationStage, ...]:
        if self is EvaluationRunPhase.ALL:
            return tuple(EvaluationStage)
        return (EvaluationStage(self.value),)


class EvaluationModelCallWriter:
    """Commit judge model calls through the evaluator-only persistence boundary."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def add(self, trace: ModelCallTrace) -> ModelCallTrace:
        async with ExperimentUnitOfWork(self.sessions, DataScope.EVALUATION) as uow:
            return await uow.model_calls.add(trace)


def evaluation_job_id(
    evaluation_run_id: str,
    run_id: str,
    query_id: str,
    stage: EvaluationStage,
    *,
    answer_variant: str = "primary",
) -> str:
    return "evaluation-job-" + stable_hash(
        [evaluation_run_id, run_id, query_id, answer_variant, stage.value]
    )[:24]


def _metric_versions() -> dict[str, str]:
    return {
        **ANSWER_METRIC_VERSIONS,
        "evidence_recall_at_k": "evidence-recall-v1",
        "evidence_hit_at_k": "evidence-hit-v1",
        "multi_hop_complete_at_k": "multi-hop-complete-v1",
        "image_hit_at_k": "image-hit-v1",
        "text_hit_at_k": "text-hit-v1",
        "mrr": "evidence-mrr-v1",
        "context_evidence_recall": "context-evidence-recall-v1",
        "context_multi_hop_complete": "context-multi-hop-complete-v1",
        "context_image_hit": "context-image-hit-v1",
        "context_text_hit": "context-text-hit-v1",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _evaluation_directory(settings: Settings, evaluation_run_id: str) -> Path:
    return settings.runs_path / "evaluations" / evaluation_run_id


async def create_evaluation_run(
    settings: Settings,
    suite_id: str,
) -> tuple[EvaluationRunManifest, Path]:
    """Register one immutable evaluation denominator without reading gold payloads."""

    config = load_evaluation_config(settings.resolved_evaluation_config_path)
    prompt = load_judge_prompt(config, settings.project_root)
    prompt_hash = evaluation_prompt_sha256(prompt)
    registry = load_model_registry(settings.resolved_model_registry_path)
    judge_model = registry.get(config.judge_model_alias)
    judge_model.require(ModelCapability.CHAT_TEXT)

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with sessions() as session:
            suites = PostgresBaselineSuiteRepository(session)
            suite = await suites.get_manifest(suite_id)
            suite_state = await suites.get_state(suite_id)
            suite_queries = await suites.get_queries(suite_id)
            links = await suites.get_run_links(suite_id)
            if suite_state.status is not BaselineSuiteStatus.READY:
                raise ValueError("evaluation requires a ready baseline suite")
            if {item.method for item in links} != set(suite.methods):
                raise ValueError("baseline suite does not link every configured method")

        suite_result_metadata: dict[str, object] = {
            key: suite.metadata[key]
            for key in ("smoke_fixture", "paper_result")
            if key in suite.metadata
        }
        payload: dict[str, object] = {
            "suite_id": suite_id,
            "suite_config_hash": suite.config_hash,
            "evaluation_config": config.model_dump(mode="json"),
            "judge_prompt_hash": prompt_hash,
            "judge_model": judge_model.model_dump(mode="json"),
            "metric_versions": _metric_versions(),
            **suite_result_metadata,
        }
        config_hash = stable_hash(payload)
        evaluation_run_id = "baseline-evaluation-" + stable_hash([suite_id, config_hash])[:24]

        async with EvaluationUnitOfWork(sessions) as uow:
            try:
                existing = await uow.runs.get_manifest(evaluation_run_id)
            except KeyError:
                existing = None
            if existing is not None:
                if existing.config_hash != config_hash or existing.suite_id != suite_id:
                    raise ValueError("evaluation run ID already belongs to another configuration")
                return existing, _evaluation_directory(settings, evaluation_run_id)

        created_at = utc_now()
        manifest = EvaluationRunManifest(
            evaluation_run_id=evaluation_run_id,
            suite_id=suite_id,
            protocol_version=config.protocol_version,
            normalizer_version=config.normalizer_version,
            evidence_policy_version=config.evidence_policy_version,
            bad_case_rule_version=config.bad_case_rule_version,
            judge_model_alias=config.judge_model_alias,
            judge_prompt_hash=prompt_hash,
            judge_rubric_version=config.judge_rubric_version,
            judge_repeat_count=config.judge_repeat_count,
            judge_score_values=config.judge_score_values,
            secondary_judge_enabled=config.secondary_judge_enabled,
            metric_versions=_metric_versions(),
            config_hash=config_hash,
            created_at=created_at,
            metadata={
                "evaluation_config": config.model_dump(mode="json"),
                "judge_model_config": judge_model.model_dump(mode="json"),
                "registry_subset_hash": registry.resolved_hash_for_aliases(
                    [config.judge_model_alias]
                ),
                **suite_result_metadata,
            },
        )
        targets = [
            EvaluationTarget(
                evaluation_run_id=evaluation_run_id,
                suite_id=suite_id,
                run_id=link.run_id,
                method=link.method,
                snapshot_id=suite.snapshot_id,
                selected_count=len(suite_queries),
                created_at=created_at,
            )
            for link in sorted(links, key=lambda item: item.method.value)
        ]

        run_ids = [item.run_id for item in targets]
        async with sessions() as session:
            answer_rows = (
                await session.execute(
                    select(
                        FinalAnswerModel.run_id,
                        FinalAnswerModel.query_id,
                        FinalAnswerModel.answer_id,
                    ).where(
                        FinalAnswerModel.run_id.in_(run_ids),
                        FinalAnswerModel.answer_variant == "primary",
                    )
                )
            ).all()
        answer_by_key = {
            (run_id, query_id): answer_id for run_id, query_id, answer_id in answer_rows
        }
        expected_keys = {
            (target.run_id, query.query_id) for target in targets for query in suite_queries
        }
        if set(answer_by_key) != expected_keys:
            missing = len(expected_keys - set(answer_by_key))
            extra = len(set(answer_by_key) - expected_keys)
            raise ValueError(
                "suite answer denominator differs from runtime facts "
                f"(missing={missing}, extra={extra})"
            )

        jobs = [
            EvaluationJob(
                evaluation_job_id=evaluation_job_id(
                    evaluation_run_id, target.run_id, query.query_id, stage
                ),
                evaluation_run_id=evaluation_run_id,
                run_id=target.run_id,
                snapshot_id=suite.snapshot_id,
                query_id=query.query_id,
                answer_id=answer_by_key[(target.run_id, query.query_id)],
                stage=stage,
                max_attempts=config.job_max_attempts,
                available_at=created_at,
                created_at=created_at,
                metadata={
                    "method": (
                        target.method.value
                        if isinstance(target.method, BaselineMethod)
                        else target.method
                    ),
                    "suite_query_ordinal": query.ordinal,
                },
            )
            for stage in EvaluationStage
            for target in targets
            for query in suite_queries
        ]
        async with EvaluationUnitOfWork(sessions) as uow:
            await uow.runs.register(manifest, targets, jobs)

        directory = _evaluation_directory(settings, evaluation_run_id)
        _write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        _write_jsonl(
            directory / "targets.jsonl",
            [item.model_dump(mode="json") for item in targets],
        )
        _write_jsonl(
            directory / "jobs.jsonl",
            [
                {
                    "evaluation_job_id": item.evaluation_job_id,
                    "run_id": item.run_id,
                    "query_id": item.query_id,
                    "stage": item.stage.value,
                }
                for item in jobs
            ],
        )
        return manifest, directory
    finally:
        await engine.dispose()


async def create_pmmc_evaluation_run(
    settings: Settings,
    query_executor_run_id: str,
) -> tuple[EvaluationRunManifest, Path]:
    """Register a method-blind evaluation over one completed PMMC query run."""

    config = load_evaluation_config(settings.resolved_evaluation_config_path)
    prompt = load_judge_prompt(config, settings.project_root)
    prompt_hash = evaluation_prompt_sha256(prompt)
    registry = load_model_registry(settings.resolved_model_registry_path)
    judge_model = registry.get(config.judge_model_alias)
    judge_model.require(ModelCapability.CHAT_TEXT)
    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with sessions() as session:
            source = await session.get(QueryExecutorRunModel, query_executor_run_id)
            if source is None or source.status != "completed":
                raise ValueError("PMMC evaluation requires a completed query executor run")
            query_rows = (
                (
                    await session.execute(
                        select(RunQueryModel)
                        .where(RunQueryModel.run_id == query_executor_run_id)
                        .order_by(RunQueryModel.ordinal)
                    )
                )
                .scalars()
                .all()
            )
            selections = (
                (
                    await session.execute(
                        select(QueryAnswerSelectionModel)
                        .where(
                            QueryAnswerSelectionModel.query_executor_run_id
                            == query_executor_run_id
                        )
                        .order_by(QueryAnswerSelectionModel.query_id)
                    )
                )
                .scalars()
                .all()
            )
            run_answers = (
                (
                    await session.execute(
                        select(FinalAnswerModel)
                        .where(FinalAnswerModel.run_id == query_executor_run_id)
                        .order_by(
                            FinalAnswerModel.query_id,
                            FinalAnswerModel.answer_variant,
                        )
                    )
                )
                .scalars()
                .all()
            )
            abstention_query_ids: set[str] = set()
            if source.schema_version == "0.2.0":
                from pmmc.infrastructure.database.query_executor_repositories import (
                    PostgresQueryExecutorRepository,
                )

                outcomes = PostgresQueryExecutorRepository(session)
                for query_row in query_rows:
                    if await outcomes.get_abstention(
                        query_executor_run_id, query_row.query_id, "native_mm_counterfactual",
                    ) is not None:
                        abstention_query_ids.add(query_row.query_id)
        if len(query_rows) != source.selected_query_count:
            raise ValueError("PMMC run query denominator differs from its manifest")
        query_ids = {item.query_id for item in query_rows}
        query_ordinal = {item.query_id: item.ordinal for item in query_rows}
        step7b = source.schema_version == "0.2.0"
        variant_counts: dict[str, int] = {}
        variant_missing_counts: dict[str, int] = {}
        if step7b:
            unsupported = sorted(
                {
                    item.answer_variant
                    for item in run_answers
                    if item.answer_variant not in _STEP7B_EVALUATION_VARIANTS
                }
            )
            if unsupported:
                raise ValueError(
                    "Step 7B PMMC evaluation contains unsupported answer variants: "
                    + ",".join(unsupported)
                )
            answer_keys = [
                (item.query_id, item.answer_variant) for item in run_answers
            ]
            if len(set(answer_keys)) != len(answer_keys):
                raise ValueError("Step 7B PMMC evaluation answer variants are not unique")
            if any(item.query_id not in query_ids for item in run_answers):
                raise ValueError("Step 7B PMMC answer escapes the query denominator")
            evaluation_answers = list(run_answers)
            for variant in _STEP7B_EVALUATION_VARIANTS:
                count = sum(item.answer_variant == variant for item in evaluation_answers)
                variant_counts[variant] = count
                variant_missing_counts[variant] = len(query_rows) - count
            terminal_by_variant = {
                "native_mm_counterfactual": abstention_query_ids,
                "active_final_provisional": {
                    item.query_id for item in selections
                    if item.terminal_failure and item.selected_answer_id is None
                    and item.failure_type
                },
            }
            incomplete_required = []
            for variant in _STEP7B_REQUIRED_EVALUATION_VARIANTS:
                answered = {
                    item.query_id for item in evaluation_answers if item.answer_variant == variant
                }
                no_answer = terminal_by_variant[variant]
                if answered & no_answer or answered | no_answer != query_ids:
                    incomplete_required.append(variant)
            if incomplete_required:
                raise ValueError(
                    "Step 7B PMMC evaluation required variants are incomplete: "
                    + ",".join(incomplete_required)
                )
        else:
            selected_answer_by_query = {
                item.query_id: item.selected_answer_id
                for item in selections
                if not item.terminal_failure and item.selected_answer_id is not None
            }
            if set(selected_answer_by_query) != query_ids:
                raise ValueError(
                    "unified PMMC evaluation requires one active_final answer per query"
                )
            selected_ids = set(selected_answer_by_query.values())
            evaluation_answers = [
                item for item in run_answers if item.answer_id in selected_ids
            ]
            if len(evaluation_answers) != len(query_rows):
                raise ValueError("PMMC active answer inventory is incomplete")
            for item in evaluation_answers:
                variant_counts[item.answer_variant] = (
                    variant_counts.get(item.answer_variant, 0) + 1
                )
        variant_order = {
            variant: ordinal
            for ordinal, variant in enumerate(_STEP7B_EVALUATION_VARIANTS)
        }
        evaluation_answers.sort(
            key=lambda item: (
                query_ordinal[item.query_id],
                variant_order.get(item.answer_variant, len(variant_order)),
                item.answer_id,
            )
        )
        answer_inventory_hash = stable_hash(
            [
                [item.query_id, item.answer_variant, item.answer_id]
                for item in evaluation_answers
            ]
        )

        payload: dict[str, object] = {
            "source_kind": EvaluationSourceKind.PMMC_QUERY_RUN.value,
            "source_run_id": query_executor_run_id,
            "source_content_hash": source.content_hash,
            "evaluation_config": config.model_dump(mode="json"),
            "judge_prompt_hash": prompt_hash,
            "judge_model": judge_model.model_dump(mode="json"),
            "metric_versions": _metric_versions(),
            "paper_result": source.paper_result,
            "evaluation_scope": "variant_aware" if step7b else "active_final_only",
            "answer_inventory_hash": answer_inventory_hash,
            "no_answer_policy": "independent-answer-abstention-v1" if step7b else None,
            "terminal_inventory_hash": stable_hash({
                "query_ids": sorted(query_ids),
                "fallback_abstentions": sorted(abstention_query_ids),
                "selections": sorted(
                    (item.query_id, item.selected_answer_id,
                     item.terminal_failure, item.failure_type)
                    for item in selections
                ),
            }) if step7b else None,
        }
        config_hash = stable_hash(payload)
        evaluation_run_id = (
            "pmmc-evaluation-" + stable_hash([query_executor_run_id, config_hash])[:24]
        )
        async with EvaluationUnitOfWork(sessions) as uow:
            try:
                existing = await uow.runs.get_manifest(evaluation_run_id)
            except KeyError:
                existing = None
            if existing is not None:
                if (
                    existing.source_run_id != query_executor_run_id
                    or existing.config_hash != config_hash
                ):
                    raise ValueError("PMMC evaluation identity conflict")
                return existing, _evaluation_directory(settings, evaluation_run_id)

        created_at = utc_now()
        manifest = EvaluationRunManifest(
            evaluation_run_id=evaluation_run_id,
            source_kind=EvaluationSourceKind.PMMC_QUERY_RUN,
            source_run_id=query_executor_run_id,
            protocol_version=config.protocol_version,
            normalizer_version=config.normalizer_version,
            evidence_policy_version=config.evidence_policy_version,
            bad_case_rule_version=config.bad_case_rule_version,
            judge_model_alias=config.judge_model_alias,
            judge_prompt_hash=prompt_hash,
            judge_rubric_version=config.judge_rubric_version,
            judge_repeat_count=config.judge_repeat_count,
            judge_score_values=config.judge_score_values,
            secondary_judge_enabled=config.secondary_judge_enabled,
            metric_versions=_metric_versions(),
            config_hash=config_hash,
            created_at=created_at,
            metadata={
                "evaluation_config": config.model_dump(mode="json"),
                "judge_model_config": judge_model.model_dump(mode="json"),
                "registry_subset_hash": registry.resolved_hash_for_aliases(
                    [config.judge_model_alias]
                ),
                "paper_result": source.paper_result,
                "query_count": len(query_rows),
                "evaluation_scope": (
                    "variant_aware" if step7b else "active_final_only"
                ),
                "answer_inventory_hash": answer_inventory_hash,
                "answer_variant_counts": dict(sorted(variant_counts.items())),
                "answer_variant_missing_counts": dict(
                    sorted(variant_missing_counts.items())
                ),
                "native_mm_abstention_count": len(abstention_query_ids),
                "no_answer_policy": "independent-answer-abstention-v1",
                "judge_denominator": "generated_answers_only",
                "scientific_denominator": len(query_rows),
            },
        )
        target = EvaluationTarget(
            evaluation_run_id=evaluation_run_id,
            source_kind=EvaluationSourceKind.PMMC_QUERY_RUN,
            run_id=query_executor_run_id,
            method="pmmc",
            snapshot_id=source.snapshot_id,
            selected_count=len(evaluation_answers),
            created_at=created_at,
            metadata={
                "query_count": len(query_rows),
                "evaluation_scope": (
                    "variant_aware" if step7b else "active_final_only"
                ),
                "answer_variant_counts": dict(sorted(variant_counts.items())),
            },
        )
        jobs = [
            EvaluationJob(
                evaluation_job_id=evaluation_job_id(
                    evaluation_run_id,
                    query_executor_run_id,
                    answer.query_id,
                    stage,
                    answer_variant=answer.answer_variant,
                ),
                evaluation_run_id=evaluation_run_id,
                run_id=query_executor_run_id,
                snapshot_id=source.snapshot_id,
                query_id=answer.query_id,
                answer_id=answer.answer_id,
                answer_variant=answer.answer_variant,
                stage=stage,
                max_attempts=config.job_max_attempts,
                available_at=created_at,
                created_at=created_at,
                metadata={
                    "method": "pmmc",
                    "answer_variant": answer.answer_variant,
                    "query_ordinal": query_ordinal[answer.query_id],
                },
            )
            for stage in EvaluationStage
            for answer in evaluation_answers
        ]
        async with EvaluationUnitOfWork(sessions) as uow:
            await uow.runs.register(manifest, [target], jobs)
        directory = _evaluation_directory(settings, evaluation_run_id)
        _write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        _write_jsonl(directory / "targets.jsonl", [target.model_dump(mode="json")])
        _write_jsonl(
            directory / "jobs.jsonl",
            [
                {
                    "evaluation_job_id": item.evaluation_job_id,
                    "run_id": item.run_id,
                    "query_id": item.query_id,
                    "answer_variant": item.answer_variant,
                    "stage": item.stage.value,
                }
                for item in jobs
            ],
        )
        return manifest, directory
    finally:
        await engine.dispose()


async def _heartbeat_job(
    sessions: async_sessionmaker[AsyncSession],
    job: EvaluationJob,
    worker_id: str,
    lease_seconds: int,
    stopped: asyncio.Event,
) -> None:
    interval = max(1.0, lease_seconds / 3)
    while True:
        try:
            await asyncio.wait_for(stopped.wait(), timeout=interval)
            return
        except TimeoutError:
            pass
        async with EvaluationUnitOfWork(sessions) as uow:
            await uow.jobs.heartbeat(job.evaluation_job_id, worker_id, lease_seconds)


async def _record_safe_job_error(
    artifacts: ExperimentArtifactService,
    job: EvaluationJob,
    diagnostics: dict[str, str],
) -> None:
    payload = json.dumps(
        {
            **diagnostics,
            "evaluation_job_id": job.evaluation_job_id,
            "evaluation_run_id": job.evaluation_run_id,
            "stage": job.stage.value,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    await artifacts.put_bytes(
        artifact_id=f"{job.evaluation_job_id}:error:{job.attempt_count}",
        run_id=job.run_id,
        evaluation_run_id=job.evaluation_run_id,
        snapshot_id=job.snapshot_id,
        query_id=job.query_id,
        artifact_type=AssetType.OTHER,
        data=payload,
        mime_type="application/json",
        extension="json",
    )


async def _run_evaluation_stage(
    *,
    sessions: async_sessionmaker[AsyncSession],
    graph: BaselineEvaluationGraph,
    artifacts: ExperimentArtifactService,
    evaluation_run_id: str,
    stage: EvaluationStage,
    config: BaselineEvaluationConfig,
    lease_seconds: int,
) -> dict[str, int]:
    worker_count = max(1, config.concurrency)
    count_lock = asyncio.Lock()
    async with EvaluationUnitOfWork(sessions) as uow:
        initial = await uow.runs.stage_status_counts(evaluation_run_id, stage)
    counts = {
        "succeeded": initial[JobStatus.SUCCEEDED],
        "failed": initial[JobStatus.FAILED] + initial[JobStatus.CANCELLED],
    }
    failure_reason_counts: dict[str, int] = {}
    total = sum(initial.values())
    processed = sum(
        initial[status] for status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)
    )
    initial_processed = processed
    last_reported_at = time.monotonic()
    LOGGER.info(
        "evaluation_stage_started",
        evaluation_run_id=evaluation_run_id,
        stage=stage.value,
        processed=processed,
        total=total,
        workers=worker_count,
        work_unit="query_variant_jobs",
        progress_interval_seconds=30,
    )

    async def record_terminal(status: str) -> None:
        nonlocal processed, last_reported_at
        async with count_lock:
            counts[status] += 1
            processed += 1
            current_processed = processed
            now = time.monotonic()
            should_report = (
                current_processed == total
                or current_processed == initial_processed + 1
                or status == "failed"
                or now - last_reported_at >= 30
            )
            if should_report:
                last_reported_at = now
            snapshot = dict(counts)
        if should_report:
            LOGGER.info(
                "evaluation_stage_progress",
                evaluation_run_id=evaluation_run_id,
                stage=stage.value,
                processed=current_processed,
                total=total,
                succeeded=snapshot["succeeded"],
                failed=snapshot["failed"],
                work_unit="query_variant_jobs",
            )

    async def worker(ordinal: int) -> None:
        worker_id = f"evaluation-{stage.value}-{os.getpid()}-{ordinal}"
        while True:
            async with EvaluationUnitOfWork(sessions) as uow:
                job = await uow.jobs.claim(
                    worker_id,
                    lease_seconds,
                    evaluation_run_id=evaluation_run_id,
                    stage=stage,
                )
            if job is None:
                return
            stopped = asyncio.Event()
            heartbeat = asyncio.create_task(
                _heartbeat_job(sessions, job, worker_id, lease_seconds, stopped)
            )
            try:
                async with EvaluationUnitOfWork(sessions) as uow:
                    complete = await uow.facts.stage_complete(
                        evaluation_run_id, job.answer_id, stage
                    )
                if not complete:
                    await graph.evaluate(job, stage)
                async with EvaluationUnitOfWork(sessions) as uow:
                    await uow.jobs.finish(job.evaluation_job_id, worker_id, success=True)
                await record_terminal("succeeded")
            except Exception as exc:
                diagnostics = safe_error_diagnostics(exc)
                retryable = _evaluation_failure_is_retryable(exc, diagnostics)
                LOGGER.error(
                    "evaluation_job_failed",
                    evaluation_run_id=job.evaluation_run_id,
                    evaluation_job_id=job.evaluation_job_id,
                    stage=job.stage.value,
                    attempt_count=job.attempt_count,
                    retryable=retryable,
                    **diagnostics,
                )
                with suppress(Exception):
                    await _record_safe_job_error(artifacts, job, diagnostics)
                async with EvaluationUnitOfWork(sessions) as uow:
                    result = await uow.jobs.finish(
                        job.evaluation_job_id,
                        worker_id,
                        success=False,
                        error_type=diagnostics["error_type"],
                        error_summary=safe_error_summary(diagnostics),
                        retry_delay_seconds=config.job_retry_delay_seconds,
                        retryable=retryable,
                    )
                if result.status is JobStatus.PENDING:
                    await asyncio.sleep(config.job_retry_delay_seconds)
                else:
                    reason = (
                        diagnostics.get("error_code")
                        or diagnostics.get("database_guard_code")
                        or diagnostics.get("constraint_name")
                        or diagnostics["error_type"]
                    )
                    async with count_lock:
                        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
                    await record_terminal("failed")
            finally:
                stopped.set()
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat

    await asyncio.gather(*(worker(index) for index in range(worker_count)))
    LOGGER.info(
        "evaluation_stage_done",
        evaluation_run_id=evaluation_run_id,
        stage=stage.value,
        processed=processed,
        total=total,
        succeeded=counts["succeeded"],
        failed=counts["failed"],
        failure_reason_counts=dict(sorted(failure_reason_counts.items())),
    )
    return counts


async def run_evaluation_workflow(
    *,
    settings: Settings,
    suite_id: str | None = None,
    query_executor_run_id: str | None = None,
    resume_evaluation_run_id: str | None = None,
    phase: EvaluationRunPhase = EvaluationRunPhase.ALL,
) -> tuple[EvaluationRunManifest, Path]:
    """Create or resume a gold-isolated evaluation run in ordered stages."""

    supplied = sum(
        value is not None for value in (suite_id, query_executor_run_id, resume_evaluation_run_id)
    )
    if supplied != 1:
        raise ValueError(
            "provide exactly one of suite_id, query_executor_run_id, or resume_evaluation_run_id"
        )
    if suite_id is not None:
        manifest, directory = await create_evaluation_run(settings, suite_id)
    elif query_executor_run_id is not None:
        manifest, directory = await create_pmmc_evaluation_run(
            settings,
            query_executor_run_id,
        )
    else:
        engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
        try:
            async with EvaluationUnitOfWork(sessions) as uow:
                manifest = await uow.runs.get_manifest(str(resume_evaluation_run_id))
            directory = _evaluation_directory(settings, manifest.evaluation_run_id)
        finally:
            await engine.dispose()

    config = load_evaluation_config(settings.resolved_evaluation_config_path)
    if manifest.metadata.get("evaluation_config") != config.model_dump(mode="json"):
        raise ValueError("current evaluation config differs from the immutable manifest")

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with EvaluationUnitOfWork(sessions) as uow:
            state = await uow.runs.get_state(manifest.evaluation_run_id)
            if state.status is EvaluationRunStatus.COMPLETED:
                return manifest, directory
            if state.status is EvaluationRunStatus.CREATED:
                await uow.runs.transition(manifest.evaluation_run_id, EvaluationRunStatus.RUNNING)
            elif state.status is not EvaluationRunStatus.RUNNING:
                raise ValueError(
                    f"evaluation cannot resume from terminal state {state.status.value}"
                )

        runtime_store, evaluation_store = create_evaluator_object_stores(settings)
        artifacts = ExperimentArtifactService(
            sessions,
            evaluation_store,
            DataScope.EVALUATION,
            artifact_prefix=settings.experiment_artifact_prefix,
        )
        registry = load_model_registry(settings.resolved_model_registry_path)
        judge_service: BaselineJudgeService | None = None
        if EvaluationStage.JUDGE in phase.stages:
            prompt = load_judge_prompt(config, settings.project_root)
            execution = create_model_execution_service(
                settings,
                registry,
                artifact_service=artifacts,
                model_calls=EvaluationModelCallWriter(sessions),
                data_scope=DataScope.EVALUATION,
            )
            judge_service = BaselineJudgeService(
                execution,
                create_generation_gateway(
                    settings,
                    registry,
                    config.judge_model_alias,
                    image_url_resolver=create_image_url_resolver(settings, runtime_store),
                ),
                config,
                prompt,
                settings.model_request_timeout_seconds,
            )
        graph = BaselineEvaluationGraph(
            loader=PostgresEvaluationCaseLoader(sessions),
            sessions=sessions,
            config=config,
            judge_service=judge_service,
        )

        for stage in phase.stages:
            if (
                stage is EvaluationStage.DETERMINISTIC
                and manifest.source_kind is EvaluationSourceKind.PMMC_QUERY_RUN
            ):
                assert manifest.source_run_id is not None
                await _preflight_pmmc_evaluation_context_lineage(
                    sessions,
                    evaluation_run_id=manifest.evaluation_run_id,
                    query_executor_run_id=manifest.source_run_id,
                )
            if stage is EvaluationStage.ANALYZE:
                async with EvaluationUnitOfWork(sessions) as uow:
                    for required in (
                        EvaluationStage.DETERMINISTIC,
                        EvaluationStage.JUDGE,
                    ):
                        status_counts = await uow.runs.stage_status_counts(
                            manifest.evaluation_run_id, required
                        )
                        if any(
                            status_counts[status]
                            for status in JobStatus
                            if status is not JobStatus.SUCCEEDED
                        ):
                            raise ValueError(
                                "analyze requires completed deterministic and judge stages"
                            )
            await _run_evaluation_stage(
                sessions=sessions,
                graph=graph,
                artifacts=artifacts,
                evaluation_run_id=manifest.evaluation_run_id,
                stage=stage,
                config=config,
                lease_seconds=settings.experiment_job_lease_seconds,
            )
            async with EvaluationUnitOfWork(sessions) as uow:
                status_counts = await uow.runs.stage_status_counts(
                    manifest.evaluation_run_id, stage
                )
                terminal_failures = (
                    status_counts[JobStatus.FAILED] + status_counts[JobStatus.CANCELLED]
                )
                if terminal_failures:
                    await uow.runs.transition(
                        manifest.evaluation_run_id,
                        EvaluationRunStatus.FAILED,
                        error_type="EvaluationJobsFailed",
                        error_summary=(f"{terminal_failures} {stage.value} jobs failed"),
                    )
                    raise RuntimeError(f"evaluation {stage.value} stage has failed jobs")
                incomplete = sum(
                    status_counts[status] for status in (JobStatus.PENDING, JobStatus.RUNNING)
                )
                if incomplete:
                    raise RuntimeError(
                        f"evaluation {stage.value} stage remains incomplete; "
                        "resume after active leases expire"
                    )

        async with EvaluationUnitOfWork(sessions) as uow:
            if await uow.runs.all_jobs_succeeded(manifest.evaluation_run_id):
                await uow.runs.transition(manifest.evaluation_run_id, EvaluationRunStatus.COMPLETED)
            state = await uow.runs.get_state(manifest.evaluation_run_id)
        _write_json(directory / "state.json", state.model_dump(mode="json"))
        return manifest, directory
    finally:
        await engine.dispose()


async def verify_evaluation_run(
    settings: Settings,
    evaluation_run_id: str,
    *,
    require_managed_runtime: bool = False,
) -> dict[str, Any]:
    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with EvaluationUnitOfWork(sessions) as uow:
            manifest = await uow.runs.get_manifest(evaluation_run_id)
            state = await uow.runs.get_state(evaluation_run_id)
            targets = await uow.runs.get_targets(evaluation_run_id)
            jobs = await uow.runs.get_jobs(evaluation_run_id)
        expected_jobs = state.selected_count * state.target_count * len(EvaluationStage)
        expected_answers = state.selected_count * state.target_count
        issues: list[str] = []
        if len(targets) != state.target_count:
            issues.append("evaluation target count differs from state")
        if len(jobs) != expected_jobs:
            issues.append("evaluation job count differs from paired denominator")
        if state.status is EvaluationRunStatus.COMPLETED and any(
            item.status is not JobStatus.SUCCEEDED for item in jobs
        ):
            issues.append("completed evaluation contains non-succeeded jobs")
        async with sessions() as session:
            normalization_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(AnswerNormalizationModel)
                    .where(AnswerNormalizationModel.evaluation_run_id == evaluation_run_id)
                )
                or 0
            )
            context_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ContextEvaluationModel)
                    .where(ContextEvaluationModel.evaluation_run_id == evaluation_run_id)
                )
                or 0
            )
            answer_metric_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(MetricScoreModel)
                    .where(
                        MetricScoreModel.evaluation_run_id == evaluation_run_id,
                        MetricScoreModel.metric_family == "answer",
                    )
                )
                or 0
            )
            judge_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(JudgeScoreModel)
                    .where(JudgeScoreModel.evaluation_run_id == evaluation_run_id)
                )
                or 0
            )
            primary_label_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(BadCaseLabelModel)
                    .where(
                        BadCaseLabelModel.evaluation_run_id == evaluation_run_id,
                        BadCaseLabelModel.is_primary.is_(True),
                    )
                )
                or 0
            )
            successful_calls = list(
                (
                    await session.scalars(
                        select(EvaluationModelCallModel).where(
                            EvaluationModelCallModel.evaluation_run_id == evaluation_run_id,
                            EvaluationModelCallModel.status == "success",
                        )
                    )
                ).all()
            )
            evaluation_artifacts = list(
                (
                    await session.scalars(
                        select(EvaluationArtifactModel).where(
                            EvaluationArtifactModel.evaluation_run_id == evaluation_run_id
                        )
                    )
                ).all()
            )
            runtime_namespace_leaks = int(
                await session.scalar(
                    select(func.count())
                    .select_from(RuntimeArtifactModel)
                    .where(
                        RuntimeArtifactModel.object_key.like(
                            f"{settings.experiment_artifact_prefix.strip('/')}/"
                            f"{evaluation_run_id}/%"
                        )
                    )
                )
                or 0
            )
        nonready_artifacts = sum(item.state != "ready" for item in evaluation_artifacts)
        expected_key_prefix = (
            f"{settings.experiment_artifact_prefix.strip('/')}/{evaluation_run_id}/"
        )
        expected_uri_prefix = f"s3://{settings.object_store_evaluation_bucket}/"
        misplaced_artifacts = sum(
            not item.object_key.startswith(expected_key_prefix)
            or item.storage_uri is None
            or not item.storage_uri.startswith(expected_uri_prefix)
            for item in evaluation_artifacts
        )
        coverage = {
            "normalizations": normalization_count,
            "context_evaluations": context_count,
            "answer_metrics": answer_metric_count,
            "judge_scores": judge_count,
            "primary_bad_case_labels": primary_label_count,
            "successful_judge_calls": len(successful_calls),
        }
        expected_coverage = {
            "normalizations": expected_answers,
            "context_evaluations": expected_answers,
            "answer_metrics": expected_answers * len(ANSWER_METRIC_VERSIONS),
            "judge_scores": expected_answers,
            "primary_bad_case_labels": expected_answers,
            "successful_judge_calls": expected_answers,
        }
        if state.status is EvaluationRunStatus.COMPLETED and coverage != expected_coverage:
            issues.append("completed evaluation fact coverage differs from denominator")
        if any(item.stage != AppStage.JUDGE.value for item in successful_calls):
            issues.append("evaluation contains a successful non-judge model call")
        incomplete_runtime_lineage = sum(
            item.runtime_id is None or item.runtime_service_id is None for item in successful_calls
        )
        if require_managed_runtime and incomplete_runtime_lineage:
            issues.append("judge model calls are missing managed-runtime lineage")
        if nonready_artifacts:
            issues.append("evaluation contains non-ready artifacts")
        if misplaced_artifacts:
            issues.append("evaluation artifacts are outside the evaluation bucket namespace")
        if runtime_namespace_leaks:
            issues.append("evaluation artifact namespace appears in runtime persistence")
        return {
            "evaluation_run_id": evaluation_run_id,
            "suite_id": manifest.suite_id,
            "status": state.status.value,
            "selected_count": state.selected_count,
            "target_count": state.target_count,
            "job_count": len(jobs),
            "job_status_counts": {
                status.value: sum(item.status is status for item in jobs) for status in JobStatus
            },
            "fact_coverage": coverage,
            "managed_judge_call_count": len(successful_calls) - incomplete_runtime_lineage,
            "nonready_artifact_count": nonready_artifacts,
            "misplaced_evaluation_artifact_count": misplaced_artifacts,
            "runtime_artifact_namespace_leak_count": runtime_namespace_leaks,
            "valid": not issues,
            "issues": issues,
        }
    finally:
        await engine.dispose()


async def inspect_evaluation_query(
    settings: Settings,
    evaluation_run_id: str,
    query_id: str,
) -> dict[str, Any]:
    """Return evaluator-only query detail for debugging and bad-case analysis."""

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        loader = PostgresEvaluationCaseLoader(sessions)
        async with EvaluationUnitOfWork(sessions) as uow:
            jobs = [
                item
                for item in await uow.runs.get_jobs(evaluation_run_id)
                if item.query_id == query_id and item.stage is EvaluationStage.DETERMINISTIC
            ]
        if not jobs:
            raise KeyError(f"query is not part of evaluation run: {query_id}")
        results: list[dict[str, Any]] = []
        for job in jobs:
            runtime = await loader.load_runtime(job)
            gold = await loader.load_gold(job)
            existing = await loader.load_existing(evaluation_run_id, job.answer_id)
            results.append(
                {
                    "run_id": job.run_id,
                    "method": runtime.answer.method,
                    "answer_id": job.answer_id,
                    "candidate_answer": runtime.answer.final_answer,
                    "reference_answer": gold.answer,
                    "metrics": [item.model_dump(mode="json") for item in existing.metrics],
                    "judge": (
                        existing.judge.model_dump(mode="json")
                        if existing.judge is not None
                        else None
                    ),
                    "bad_cases": [item.model_dump(mode="json") for item in existing.bad_cases],
                }
            )
        return {
            "evaluation_run_id": evaluation_run_id,
            "query_id": query_id,
            "results": sorted(results, key=lambda item: str(item["method"])),
        }
    finally:
        await engine.dispose()
