"""One-command, resumable orchestration for PMMC research experiments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from pmmc.benchmarks import create_adapter
from pmmc.benchmarks.snapshot import build_snapshot
from pmmc.experiments.execution import (
    CheckpointPolicy,
    ExecutionSchedule,
    ExperimentExecutionConfig,
    StagePerformanceRecord,
    load_execution_config,
)
from pmmc.experiments.progress import HierarchicalProgressReporter
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.locking import (
    assert_paper_lock,
    build_model_lock,
    load_model_lock,
    load_verified_model_cache_manifest,
    prepare_model_cache_audited,
    resolve_registry_from_lock,
    verify_model_cache,
    write_model_lock,
    write_resolved_registry,
    write_verified_model_cache_manifest,
)
from pmmc.models.registry import (
    ModelRegistry,
    load_model_registry_with_overlays,
)
from pmmc.schemas import (
    FOUNDATIONAL_BASELINE_METHODS,
    BaselineMethod,
    BenchmarkSelection,
    DatasetName,
    ModelCapability,
    ModelEndpointKind,
    ModelRole,
    ModelVerificationPolicy,
    PartitionRole,
    QuestionerPilotSample,
    QuestionType,
    StageComponentManifest,
    StageComponentStatus,
    ValidationLevel,
)
from pmmc.schemas.base import (
    ANSWER_OUTPUT_SCHEMA_VERSION,
    LAUNCHER_SCHEMA_VERSION,
    SUPPORTED_LAUNCHER_SCHEMA_VERSIONS,
    BaseSchema,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)


class DownloadPolicy(StrEnum):
    AUTO = "auto"
    NEVER = "never"
    ALWAYS_REVERIFY = "always-reverify"


class CalibrationPolicy(StrEnum):
    AUTO = "auto"
    REQUIRE = "require"
    SKIP = "skip"


_STAGE_MAJOR_LAUNCHER_VERSIONS = frozenset(
    {
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
        LAUNCHER_SCHEMA_VERSION,
    }
)

_PERIODIC_CHECKPOINT_DRAIN_TIMEOUT_SECONDS = 900.0
_PERIODIC_CHECKPOINT_DRAIN_REPORT_SECONDS = 30.0
_STEP7B_ALEMBIC_REVISION = "20260911_0029"
_STEP6D_RESUME_VERIFICATION_ENV = "PMMC_STEP6D_RESUME_VERIFICATION_ONLY"
_STEP6D_RESUME_VERIFICATION_MARKER_ENV = "PMMC_STEP6D_RESUME_VERIFICATION_MARKER"
_STEP7_RESUME_VERIFICATION_ENV = "PMMC_STEP7_RESUME_VERIFICATION_ONLY"
_STEP7_RESUME_VERIFICATION_MARKER_ENV = "PMMC_STEP7_RESUME_VERIFICATION_MARKER"
_STEP7_CONTROLLED_INTERRUPT_STAGE_ENV = "PMMC_STEP7_CONTROLLED_INTERRUPT_AFTER_STAGE"
_STEP7_CONTROLLED_INTERRUPT_MARKER_ENV = "PMMC_STEP7_CONTROLLED_INTERRUPT_MARKER"


class Step7ControlledInterruption(RuntimeError):
    """Harness-only interruption raised after a durable Step 7 milestone."""


class Step7PilotAcceptanceError(RuntimeError):
    """A verified diagnostic bundle is not necessarily an acceptable pilot."""

    def __init__(self, issues: Sequence[str]) -> None:
        super().__init__("Step 7 pilot acceptance failed; verified diagnostic bundle preserved")
        self.error_code = "step7_pilot_acceptance_failed"
        self.failure_stage = "export_verify"
        self.issue_count = len(issues)

    def safe_diagnostics(self) -> dict[str, object]:
        return {
            "error_code": self.error_code,
            "integrity_issue_count": self.issue_count,
            "transfer_verified": True,
            "next_action": "review_diagnostic_bundle_before_model_retry",
        }


class QuestionerPersistencePreflightError(RuntimeError):
    """A schema-contract failure, not a candidate scientific rejection."""

    def __init__(self, failed_checks: list[str]) -> None:
        super().__init__("Questioner persistence preflight failed: " + ",".join(failed_checks))
        self.error_code = (
            "questioner_persistence_revision_mismatch"
            if "alembic_revision" in failed_checks
            else "questioner_persistence_contract_mismatch"
        )
        self.database_diagnostics = {"error_code": self.error_code}


class Step7PrefixRecordError(RuntimeError):
    """Safe prefix-record failure with a stable diagnostic code."""

    def __init__(self, error_code: str) -> None:
        super().__init__("Step 7 prefix record verification failed")
        self.error_code = error_code


class Step7PrefixChildError(RuntimeError):
    """Carry bounded CPU-child database diagnostics into the failure capsule."""

    def __init__(self, *, phase: str, diagnostics: object) -> None:
        from pmmc.infrastructure.database.errors import safe_forwarded_database_diagnostics

        super().__init__("Step 7 prefix CPU command failed")
        self.failure_stage = "step7_prefix"
        self.persistence_operation = phase
        self.error_code = "step7_prefix_cpu_command_failed"
        self.database_diagnostics = safe_forwarded_database_diagnostics(diagnostics)


class CorpusPreparationError(RuntimeError):
    """Safe corpus-stage wrapper that preserves the original exception chain."""

    def __init__(self, *, stage: str, operation: str) -> None:
        super().__init__("corpus preparation failed")
        self.failure_stage = stage
        self.persistence_operation = operation
        self.error_code = f"corpus_{operation}_failed"


class PeriodicCheckpointShutdownError(RuntimeError):
    """Raised when an in-flight checkpoint cannot drain during stage shutdown."""


def _uses_stage_major_launcher(version: str) -> bool:
    return version in _STAGE_MAJOR_LAUNCHER_VERSIONS


class InfrastructureMode(StrEnum):
    SELF_CONTAINED = "self_contained"
    EXTERNAL = "external"


class ExperimentTarget(StrEnum):
    FOUNDATIONAL_BASELINES = "foundational_baselines"
    PMMC_QUESTIONER = "pmmc_questioner"
    PMMC_QUESTIONER_PILOT = "pmmc_questioner_pilot"
    PMMC_COMPILER_PROGRAMS = "pmmc_compiler_programs"
    PMMC_COMPILER_PROGRAM_PILOT = "pmmc_compiler_program_pilot"
    PMMC_COMPILER_R0_HARDENING_SMOKE = "pmmc_compiler_r0_hardening_smoke"
    PMMC_COMPILER_R0_QUALITY_PILOT = "pmmc_compiler_r0_quality_pilot"
    PMMC_COMPILER_VERIFIED_QBANK_PILOT = "pmmc_compiler_verified_qbank_pilot"
    PMMC_COMPILE_ONLY = "pmmc_compile_only"
    PMMC_STEP7_EXECUTOR_PILOT = "pmmc_step7_executor_pilot"
    PMMC_STEP7B_EXECUTOR_PILOT = "pmmc_step7b_executor_pilot"
    PMMC_ONLINE = "pmmc_online"
    PMMC = "pmmc"


def _is_questioner_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_QUESTIONER,
        ExperimentTarget.PMMC_QUESTIONER_PILOT,
        ExperimentTarget.PMMC_COMPILER_PROGRAMS,
        ExperimentTarget.PMMC_COMPILER_PROGRAM_PILOT,
        ExperimentTarget.PMMC_COMPILER_R0_HARDENING_SMOKE,
        ExperimentTarget.PMMC_COMPILER_R0_QUALITY_PILOT,
        ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT,
        ExperimentTarget.PMMC_COMPILE_ONLY,
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
    }


def _is_program_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_COMPILER_PROGRAMS,
        ExperimentTarget.PMMC_COMPILER_PROGRAM_PILOT,
        ExperimentTarget.PMMC_COMPILER_R0_HARDENING_SMOKE,
        ExperimentTarget.PMMC_COMPILER_R0_QUALITY_PILOT,
        ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT,
        ExperimentTarget.PMMC_COMPILE_ONLY,
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
    }


def _is_program_pilot_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_COMPILER_PROGRAM_PILOT,
        ExperimentTarget.PMMC_COMPILER_R0_HARDENING_SMOKE,
        ExperimentTarget.PMMC_COMPILER_R0_QUALITY_PILOT,
        ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT,
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
    }


def _is_semantic_program_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_COMPILER_R0_HARDENING_SMOKE,
        ExperimentTarget.PMMC_COMPILER_R0_QUALITY_PILOT,
        ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT,
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
    }


def _is_pilot_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_QUESTIONER_PILOT,
        ExperimentTarget.PMMC_COMPILER_PROGRAM_PILOT,
        ExperimentTarget.PMMC_COMPILER_R0_HARDENING_SMOKE,
        ExperimentTarget.PMMC_COMPILER_R0_QUALITY_PILOT,
        ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT,
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
    }


def _is_verified_qbank_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT,
        ExperimentTarget.PMMC_COMPILE_ONLY,
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_ONLINE,
    }


def _is_step7_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_ONLINE,
    }


def _is_step7_pilot_target(target: ExperimentTarget) -> bool:
    return target in {
        ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
    }


def _step7_coalesced_checkpoint_target(target: ExperimentTarget, stage_id: str) -> str | None:
    """Coalesce adjacent Step 7 facts at the next durable recovery boundary."""

    if not _is_step7_target(target):
        return None
    return {
        "snapshot_import:memlens": "snapshot_import:memgallery",
        "unit_prepare:memlens": "unit_prepare:memgallery",
        "dry_run:memlens": "dry_run:memgallery",
        "verified_qbank:memlens": "verified_qbank:memgallery",
        "qbank_coverage:memlens": "export:memgallery",
        "qbank_coverage:memgallery": "export:memgallery",
        "export:memlens": "export:memgallery",
        "evaluation_deterministic:model-stage": "evaluation_judge:model-stage",
        "qbank_semantic:model-stage": "evaluation_judge:model-stage",
        "evaluation_analyze:model-stage": "automatic_quality",
    }.get(stage_id)


_STEP7_PREFIX_QP_SOURCE_FINGERPRINT_KEY = "step7_prefix_qp_source_fingerprint"
_STEP7_PREFIX_QP_PARTIAL_MILESTONES = frozenset(
    {
        "step7_prefix:planner_failure",
        "step7_prefix:questioner_planner",
    }
)


def _step7_doubter_facts_reusable(reports: Sequence[Mapping[str, object]]) -> bool:
    """A restored prefix can skip inference only after every candidate is terminal."""

    return bool(reports) and all(
        row.get("valid") is True
        and row.get("stage_accounting_complete") is True
        and isinstance(candidate_count := row.get("candidate_count"), int)
        and candidate_count > 0
        and row.get("terminal_candidate_count") == row["candidate_count"]
        and not row.get("issues")
        for row in reports
    )


def _step7_prefix_partial_checkpoint_source_compatible(
    milestone: str,
    metadata: Mapping[str, object],
    current_compiler_fingerprint: str,
) -> bool:
    """Require an exact compiler identity for partial Q/P checkpoints."""

    if milestone not in _STEP7_PREFIX_QP_PARTIAL_MILESTONES:
        return True
    recorded = metadata.get(_STEP7_PREFIX_QP_SOURCE_FINGERPRINT_KEY)
    return isinstance(recorded, str) and recorded == current_compiler_fingerprint


_PRE_CHECKPOINT_REPLAYABLE_STAGE_IDS = frozenset(
    {
        "preflight",
        "model_prepare",
        "infrastructure",
        "model_calibration",
    }
)


def _is_pre_checkpoint_replayable_stage(stage_id: str) -> bool:
    """Return whether a stage can be cold-replayed before the first checkpoint."""

    return (
        stage_id in _PRE_CHECKPOINT_REPLAYABLE_STAGE_IDS
        or stage_id.startswith("benchmark:")
    )


class ExperimentLaunchStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentStageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentModelSelection(BaseSchema):
    answer_model: NonEmptyStr = "answer-qwen3.5-9b"
    judge_model: NonEmptyStr = "judge-qwen3.5-9b"
    text_embedding_model: NonEmptyStr = "qwen3-embedding-8b"
    contrastive_embedding_model: NonEmptyStr = "siglip2-so400m-patch14-384"
    multimodal_embedding_model: NonEmptyStr = "qwen3-vl-embedding-8b"
    compiler_model: NonEmptyStr | None = None
    qbank_embedding_model: NonEmptyStr | None = None
    reranker_model: NonEmptyStr | None = None

    def required_aliases(
        self,
        methods: list[BaselineMethod],
        *,
        include_judge: bool,
        target: ExperimentTarget = ExperimentTarget.FOUNDATIONAL_BASELINES,
    ) -> list[str]:
        if _is_questioner_target(target):
            if self.compiler_model is None:
                raise ValueError("Questioner launches require compiler_model")
            aliases = {
                self.compiler_model,
                self.text_embedding_model,
                self.multimodal_embedding_model,
            }
            if _is_semantic_program_target(target):
                aliases.add(self.judge_model)
            if _is_step7_target(target):
                aliases.add(self.answer_model)
            if target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
                if self.reranker_model is None:
                    raise ValueError("Step 7B launches require reranker_model")
                aliases.add(self.reranker_model)
            return sorted(aliases)
        if target is ExperimentTarget.PMMC_ONLINE:
            if self.reranker_model is None:
                raise ValueError("Step 7B online launches require reranker_model")
            return sorted(
                {
                    self.answer_model,
                    self.judge_model,
                    self.text_embedding_model,
                    self.multimodal_embedding_model,
                    self.reranker_model,
                }
            )
        # The shared corpus boundary is frozen with the text-embedding model's
        # tokenizer even when a development run selects only full-context methods.
        aliases = {self.answer_model, self.text_embedding_model}
        if include_judge:
            aliases.add(self.judge_model)
        if BaselineMethod.CAPTION_TEXT_RAG in methods:
            aliases.add(self.text_embedding_model)
        if BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG in methods:
            aliases.add(self.contrastive_embedding_model)
        if BaselineMethod.NATIVE_MULTIMODAL_RAG in methods:
            aliases.add(self.multimodal_embedding_model)
        return sorted(aliases)

    def runtime_roles(self) -> dict[ModelRole, str]:
        values = {
            ModelRole.ANSWER: self.answer_model,
            ModelRole.JUDGE: self.judge_model,
            ModelRole.TEXT_EMBEDDING: self.text_embedding_model,
            ModelRole.CONTRASTIVE_EMBEDDING: self.contrastive_embedding_model,
            ModelRole.MULTIMODAL_EMBEDDING: self.multimodal_embedding_model,
        }
        if self.compiler_model is not None:
            values[ModelRole.COMPILER] = self.compiler_model
        if self.reranker_model is not None:
            values[ModelRole.RERANKER] = self.reranker_model
        return values


class ExperimentDatasetSelection(BaseSchema):
    datasets: list[DatasetName] = Field(
        default_factory=lambda: [DatasetName.MEMLENS, DatasetName.MEMGALLERY]
    )
    question_types: list[QuestionType] = Field(default_factory=list)
    source_categories: list[NonEmptyStr] = Field(default_factory=list)
    scenario_ids: list[NonEmptyStr] = Field(default_factory=list)
    query_ids: list[NonEmptyStr] = Field(default_factory=list)
    limit: PositiveInt | None = None
    sample_size: PositiveInt | None = None
    seed: NonNegativeInt = 0
    shard_count: PositiveInt = 1
    shard_index: NonNegativeInt = 0
    include_excluded: bool = False
    partition_manifest_path: Path | None = None
    partition_role: PartitionRole | None = None

    @model_validator(mode="after")
    def _validate_selection(self) -> ExperimentDatasetSelection:
        if not self.datasets:
            raise ValueError("experiment requires at least one dataset")
        if DatasetName.UNKNOWN in self.datasets:
            raise ValueError("unknown is not a runnable benchmark")
        if len(self.datasets) != len(set(self.datasets)):
            raise ValueError("datasets cannot contain duplicates")
        if self.limit is not None and self.sample_size is not None:
            raise ValueError("limit and sample_size are mutually exclusive")
        if self.shard_index >= self.shard_count:
            raise ValueError("shard_index must be less than shard_count")
        if (self.partition_manifest_path is None) != (self.partition_role is None):
            raise ValueError(
                "partition_manifest_path and partition_role must be configured together"
            )
        ensure_unique_strings(self.source_categories, "source_categories")
        ensure_unique_strings(self.scenario_ids, "scenario_ids")
        ensure_unique_strings(self.query_ids, "query_ids")
        return self

    def for_dataset(self, dataset: DatasetName) -> BenchmarkSelection:
        if dataset not in self.datasets:
            raise ValueError(f"dataset is not part of this launch: {dataset.value}")
        return BenchmarkSelection(
            datasets=[dataset],
            question_types=self.question_types,
            source_categories=self.source_categories,
            scenario_ids=self.scenario_ids,
            query_ids=self.query_ids,
            limit=self.limit,
            sample_size=self.sample_size,
            seed=self.seed,
            shard_count=self.shard_count,
            shard_index=self.shard_index,
            include_excluded=self.include_excluded,
            metadata=(
                {
                    "partition_role": self.partition_role.value,
                }
                if self.partition_manifest_path is not None and self.partition_role is not None
                else {}
            ),
        )


class ExperimentInfrastructureConfig(BaseSchema):
    mode: InfrastructureMode = InfrastructureMode.SELF_CONTAINED
    wait_timeout_seconds: PositiveInt = 300
    migrate_database: bool = True
    sync_assets: bool = True
    archive_snapshot: bool = False
    archive_referenced_model_artifacts: bool = False


class ExperimentProgressConfig(BaseSchema):
    report_interval_seconds: float = Field(default=30.0, gt=0)
    non_tty: bool | None = None


class ExperimentLauncherConfig(BaseSchema):
    launcher_schema_version: NonEmptyStr = LAUNCHER_SCHEMA_VERSION
    experiment_name: NonEmptyStr = "five-foundational-baselines"
    target: ExperimentTarget = ExperimentTarget.FOUNDATIONAL_BASELINES
    mode: AppMode = AppMode.PAPER
    methods: list[BaselineMethod] = Field(
        default_factory=lambda: list(FOUNDATIONAL_BASELINE_METHODS)
    )
    models: ExperimentModelSelection = Field(default_factory=ExperimentModelSelection)
    selection: ExperimentDatasetSelection = Field(default_factory=ExperimentDatasetSelection)
    infrastructure: ExperimentInfrastructureConfig = Field(
        default_factory=ExperimentInfrastructureConfig
    )
    progress: ExperimentProgressConfig = Field(default_factory=ExperimentProgressConfig)
    runtime_profile_path: Path = Path("configs/runtime/h100-paper.yaml")
    execution_profile_path: Path | None = None
    questioner_config_path: Path | None = None
    questioner_pilot_config_path: Path | None = None
    planner_config_path: Path | None = None
    doubter_config_path: Path | None = None
    compiler_foundation_config_path: Path | None = None
    qbank_semantic_config_path: Path | None = None
    query_executor_config_path: Path | None = None
    questioner_freeze_manifest_path: Path | None = None
    compiler_program_freeze_manifest_path: Path | None = None
    verified_qbank_freeze_manifest_path: Path | None = None
    query_executor_freeze_manifest_path: Path | None = None
    step7b_freeze_manifest_path: Path | None = None
    model_registry_path: Path | None = None
    model_registry_overlays: list[Path] = Field(default_factory=list)
    download_policy: DownloadPolicy = DownloadPolicy.AUTO
    model_verification_policy: ModelVerificationPolicy | None = None
    calibration_policy: CalibrationPolicy = CalibrationPolicy.AUTO
    evaluate: bool = True
    retrieval_top_k: PositiveInt = 50
    answer_top_k: PositiveInt = 5
    total_context_tokens: PositiveInt = 32768
    reserved_output_tokens: PositiveInt = 1024
    answer_batch_size: PositiveInt = 8
    embedding_batch_size: PositiveInt = 32
    multimodal_embedding_batch_size: PositiveInt = 8
    model_concurrency: PositiveInt = 8
    tensor_parallel_size: PositiveInt | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_compile_only_alias(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        metadata = payload.get("metadata")
        if (
            payload.get("target") == ExperimentTarget.PMMC.value
            and isinstance(metadata, Mapping)
            and metadata.get("execution_mode") == "compile_only"
        ):
            payload["target"] = ExperimentTarget.PMMC_COMPILE_ONLY.value
        elif (
            payload.get("target") == ExperimentTarget.PMMC.value
            and isinstance(metadata, Mapping)
            and metadata.get("execution_mode") == "online"
        ):
            payload["target"] = ExperimentTarget.PMMC_ONLINE.value
        return payload

    @model_validator(mode="after")
    def _validate_launcher(self) -> ExperimentLauncherConfig:
        if self.launcher_schema_version not in SUPPORTED_LAUNCHER_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported launcher schema version: {self.launcher_schema_version}")
        if (
            _uses_stage_major_launcher(self.launcher_schema_version)
            and self.execution_profile_path is None
        ):
            raise ValueError("current launcher schema requires execution_profile_path")
        if (
            self.mode is AppMode.PAPER
            and bool(self.metadata.get("paper_result"))
            and self.model_verification_policy is ModelVerificationPolicy.IDENTITY_ONLY
        ):
            raise ValueError("paper results cannot use identity-only model verification")
        if self.target is ExperimentTarget.PMMC:
            raise ValueError(
                "use target pmmc_online (or metadata.execution_mode=online) for "
                "the Step 7 online executor"
            )
        if self.target is ExperimentTarget.FOUNDATIONAL_BASELINES:
            if not self.methods:
                raise ValueError("baseline experiments require at least one method")
            if len(self.methods) != len(set(self.methods)):
                raise ValueError("baseline methods cannot contain duplicates")
            if self.mode is AppMode.PAPER and set(self.methods) != set(
                FOUNDATIONAL_BASELINE_METHODS
            ):
                raise ValueError("paper baseline launches require all five foundational methods")
            if any(
                item is not None
                for item in (
                    self.models.compiler_model,
                    self.models.qbank_embedding_model,
                    self.models.reranker_model,
                )
            ):
                raise ValueError("compiler, QBank, and reranker models are not baseline roles")
            if bool(self.metadata.get("component_matrix")):
                if self.launcher_schema_version not in {
                    "0.4.0",
                    "0.5.0",
                    LAUNCHER_SCHEMA_VERSION,
                }:
                    raise ValueError(
                        "baseline component matrix requires launcher schema 0.4.0 or newer"
                    )
                if self.mode is not AppMode.PAPER:
                    raise ValueError("baseline component matrix requires paper mode")
                if set(self.selection.datasets) != {
                    DatasetName.MEMLENS,
                    DatasetName.MEMGALLERY,
                }:
                    raise ValueError("baseline component matrix requires both benchmarks")
                if set(self.methods) != set(FOUNDATIONAL_BASELINE_METHODS):
                    raise ValueError("baseline component matrix requires all five methods")
                if not self.evaluate or self.metadata.get("paper_result") is not True:
                    raise ValueError("baseline component matrix requires unified paper evaluation")
        elif _is_questioner_target(self.target):
            if self.methods:
                raise ValueError("Questioner launches do not execute baseline methods")
            if self.models.compiler_model is None:
                raise ValueError("Questioner launches require compiler_model")
            if self.models.qbank_embedding_model is not None:
                raise ValueError("Questioner uses the text and multimodal embedding roles directly")
            if self.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
                if self.models.reranker_model is None:
                    raise ValueError("Step 7B requires the bounded reranker role")
            elif self.models.reranker_model is not None:
                raise ValueError("only Step 7B supports a compiler-side reranker")
            if not self.infrastructure.archive_referenced_model_artifacts:
                raise ValueError("Questioner launches must archive referenced model artifacts")
            if _is_program_target(self.target):
                if _is_verified_qbank_target(self.target):
                    if self.target is ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT:
                        expected_launcher_versions = {"0.9.0"}
                    elif self.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
                        expected_launcher_versions = {LAUNCHER_SCHEMA_VERSION}
                    else:
                        expected_launcher_versions = {"0.8.0"}
                    if self.launcher_schema_version not in expected_launcher_versions:
                        raise ValueError(
                            "verified-QBank launch schema does not match its target version"
                        )
                    if self.doubter_config_path is None:
                        raise ValueError("verified-QBank launches require doubter_config_path")
                if _is_semantic_program_target(self.target):
                    if self.launcher_schema_version not in {
                        "0.7.0",
                        "0.8.0",
                        "0.9.0",
                        LAUNCHER_SCHEMA_VERSION,
                    }:
                        raise ValueError(
                            "semantic Planner launches require launcher schema 0.7.0 "
                            "or the current launcher schema"
                        )
                    if self.qbank_semantic_config_path is None:
                        raise ValueError(
                            "semantic r=0 Planner launches require qbank_semantic_config_path"
                        )
                elif self.launcher_schema_version not in {
                    "0.6.0",
                    "0.7.0",
                    "0.8.0",
                    "0.9.0",
                    LAUNCHER_SCHEMA_VERSION,
                }:
                    raise ValueError("Planner launches require launcher schema 0.6.0 or newer")
                if self.planner_config_path is None:
                    raise ValueError("Planner launches require planner_config_path")
                if self.compiler_foundation_config_path is None:
                    raise ValueError("Planner launches require compiler_foundation_config_path")
                if self.target in {
                    ExperimentTarget.PMMC_COMPILER_PROGRAMS,
                    ExperimentTarget.PMMC_COMPILE_ONLY,
                }:
                    if self.questioner_freeze_manifest_path is None:
                        raise ValueError(
                            "formal Planner launches require a Questioner Freeze Manifest"
                        )
                elif self.questioner_freeze_manifest_path is not None:
                    raise ValueError(
                        "program quality pilots use a development audit identity, not an "
                        "approved Questioner Freeze Manifest"
                    )
                if _is_step7_target(self.target):
                    if self.query_executor_config_path is None:
                        raise ValueError("Step 7 launches require query_executor_config_path")
            if _is_pilot_target(self.target):
                if self.questioner_pilot_config_path is None:
                    raise ValueError("compiler pilot launches require a pilot config")
                if set(self.selection.datasets) != {
                    DatasetName.MEMLENS,
                    DatasetName.MEMGALLERY,
                }:
                    raise ValueError("formal compiler pilot launches require both benchmarks")
                if not self.evaluate:
                    raise ValueError("compiler pilot launches require coverage evaluation")
                if any(
                    (
                        self.selection.question_types,
                        self.selection.source_categories,
                        self.selection.scenario_ids,
                        self.selection.query_ids,
                    )
                ):
                    raise ValueError(
                        "formal compiler pilots derive their denominator from the "
                        "frozen development partition"
                    )
                if (
                    self.selection.limit is not None
                    or self.selection.sample_size is not None
                    or self.selection.shard_count != 1
                    or self.selection.shard_index != 0
                ):
                    raise ValueError(
                        "formal compiler pilots do not allow limit, sample, or shard overrides"
                    )
                if (
                    self.selection.partition_role is not None
                    and self.selection.partition_role is not PartitionRole.DEVELOPMENT
                ):
                    raise ValueError("compiler quality pilots must use the development partition")
        elif self.target is ExperimentTarget.PMMC_ONLINE:
            if self.launcher_schema_version != LAUNCHER_SCHEMA_VERSION:
                raise ValueError("pmmc_online requires the current launcher schema")
            if self.methods:
                raise ValueError("pmmc_online does not execute baseline methods")
            if self.query_executor_config_path is None:
                raise ValueError("pmmc_online requires query_executor_config_path")
            required_freezes = (
                self.questioner_freeze_manifest_path,
                self.compiler_program_freeze_manifest_path,
                self.verified_qbank_freeze_manifest_path,
                self.query_executor_freeze_manifest_path,
                self.step7b_freeze_manifest_path,
            )
            if any(value is None for value in required_freezes):
                raise ValueError(
                    "pmmc_online requires all four approved child Freeze Manifests "
                    "and the aggregate Step 7B Freeze Manifest"
                )
            if self.models.reranker_model is None:
                raise ValueError("pmmc_online requires the frozen Step 7B reranker model")
            if not self.evaluate:
                raise ValueError("pmmc_online requires unified evaluation")
            if self.mode is AppMode.PAPER and self.metadata.get("paper_result") is True:
                if self.selection.partition_role is not PartitionRole.PAPER_TEST:
                    raise ValueError("paper pmmc_online runs require the paper_test partition")
                if self.selection.partition_manifest_path is None:
                    raise ValueError("paper pmmc_online runs require a frozen partition manifest")
            for key in ("compiler_run_ids", "coverage_run_ids", "fallback_run_ids"):
                values = self.metadata.get(key)
                if not isinstance(values, Mapping) or set(values) != {
                    dataset.value for dataset in self.selection.datasets
                }:
                    raise ValueError(f"pmmc_online metadata.{key} must map every selected dataset")
        if self.mode is AppMode.PAPER and self.selection.include_excluded:
            raise ValueError("paper launches cannot include excluded benchmark queries")
        if self.mode is AppMode.PAPER and self.calibration_policy is CalibrationPolicy.SKIP:
            raise ValueError("paper launches cannot skip managed-runtime calibration")
        if self.answer_top_k > self.retrieval_top_k:
            raise ValueError("answer_top_k cannot exceed retrieval_top_k")
        if self.reserved_output_tokens >= self.total_context_tokens:
            raise ValueError("reserved output tokens must be smaller than context tokens")
        return self

    @property
    def config_hash(self) -> str:
        return _stable_hash(self.model_dump(mode="json"))

    @property
    def launch_id(self) -> str:
        return "launch-" + self.config_hash[:24]


class ExperimentPlanItem(BaseSchema):
    stage_id: NonEmptyStr
    work_units: PositiveInt
    stage_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    actual_work_units: PositiveInt | None = None
    dataset: DatasetName | None = None
    method: BaselineMethod | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ExperimentExecutionPlan(BaseSchema):
    launch_id: NonEmptyStr
    config_hash: NonEmptyStr
    items: list[ExperimentPlanItem]

    @model_validator(mode="after")
    def _validate_items(self) -> ExperimentExecutionPlan:
        if not self.items:
            raise ValueError("execution plan cannot be empty")
        ensure_unique_strings([item.stage_id for item in self.items], "items.stage_id")
        return self

    @property
    def total_work_units(self) -> int:
        return sum(item.work_units for item in self.items)

    @property
    def total_stage_weight(self) -> float:
        return sum(item.stage_weight for item in self.items)

    def get(self, stage_id: str) -> ExperimentPlanItem:
        for item in self.items:
            if item.stage_id == stage_id:
                return item
        raise KeyError(f"execution plan has no stage: {stage_id}")


class ExperimentLaunchManifest(BaseSchema):
    launch_id: NonEmptyStr
    launcher_schema_version: NonEmptyStr = LAUNCHER_SCHEMA_VERSION
    config_hash: NonEmptyStr
    experiment_name: NonEmptyStr
    mode: AppMode
    methods: list[BaselineMethod]
    datasets: list[DatasetName]
    required_model_aliases: list[NonEmptyStr]
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class ExperimentStageResult(BaseSchema):
    stage_id: NonEmptyStr
    status: ExperimentStageStatus
    work_units: PositiveInt
    reused: bool = False
    started_at: UtcDateTime
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    artifact_ids: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)


class ExperimentLaunchState(BaseSchema):
    launch_id: NonEmptyStr
    status: ExperimentLaunchStatus = ExperimentLaunchStatus.PLANNED
    completed_stage_ids: list[NonEmptyStr] = Field(default_factory=list)
    stage_results: list[ExperimentStageResult] = Field(default_factory=list)
    dataset_artifacts: dict[str, JsonDict] = Field(default_factory=dict)
    started_at: UtcDateTime | None = None
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ModelPreparationResult(BaseSchema):
    lock_path: Path
    merged_registry_path: Path
    resolved_registry_path: Path
    required_aliases: list[NonEmptyStr]
    cache_states: dict[str, NonEmptyStr]
    model_size_bytes: dict[str, NonNegativeInt] = Field(default_factory=dict)
    downloaded_bytes: NonNegativeInt = 0
    cached_bytes: NonNegativeInt = 0
    registry_hash: NonEmptyStr
    registry_subset_hash: NonEmptyStr
    verification_policy: ModelVerificationPolicy = ModelVerificationPolicy.FULL
    verification_receipts: dict[str, NonEmptyStr] = Field(default_factory=dict)
    verification_cache_hits: list[NonEmptyStr] = Field(default_factory=list)
    verification_bytes_hashed: NonNegativeInt = 0
    verified_cache_manifest_path: Path | None = None
    verified_cache_manifest_hash: NonEmptyStr | None = None


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_STEP7_PREFIX_REPORT_IDENTITY_KEYS = (
    "dataset",
    "query_count",
    "doubter_feedback_count",
    "query_ids_hash",
    "compiler_run_id",
    "compiler_run_hash",
)


def _step7_prefix_exact_round_report_upgrade(
    stored_reports: object,
    current_reports: Sequence[Mapping[str, object]],
) -> dict[str, int] | None:
    """Validate the one-time v1 report upgrade after exact-round routing DDL."""

    if not isinstance(stored_reports, list) or len(stored_reports) != len(current_reports):
        return None
    stored_by_dataset: dict[str, Mapping[str, object]] = {}
    for value in stored_reports:
        if not isinstance(value, Mapping):
            return None
        dataset = value.get("dataset")
        if not isinstance(dataset, str) or dataset in stored_by_dataset:
            return None
        stored_by_dataset[dataset] = value
    current_by_dataset: dict[str, Mapping[str, object]] = {}
    for value in current_reports:
        dataset = value.get("dataset")
        if not isinstance(dataset, str) or dataset in current_by_dataset:
            return None
        current_by_dataset[dataset] = value
    if set(stored_by_dataset) != set(current_by_dataset):
        return None

    removed_routing_members = 0
    for dataset, current in current_by_dataset.items():
        stored = stored_by_dataset[dataset]
        if "active_routing_member_count" in stored:
            return None
        if any(stored.get(key) != current.get(key) for key in _STEP7_PREFIX_REPORT_IDENTITY_KEYS):
            return None
        stored_routing = stored.get("routing_member_count")
        current_routing = current.get("routing_member_count")
        current_active = current.get("active_routing_member_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (stored_routing, current_routing, current_active)
        ):
            return None
        assert isinstance(stored_routing, int)
        assert isinstance(current_routing, int)
        assert isinstance(current_active, int)
        if current_routing <= 0 or current_routing != current_active:
            return None
        if stored_routing < current_routing:
            return None
        removed_routing_members += stored_routing - current_routing
    return {
        "dataset_count": len(current_by_dataset),
        "removed_routing_member_count": removed_routing_members,
    }


def _work_int(item: dict[str, object], key: str) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"launcher work item {key!r} must be an integer")
    return value


def _work_strings(item: dict[str, object], key: str) -> list[str]:
    value = item.get(key)
    if not isinstance(value, list) or any(not isinstance(part, str) for part in value):
        raise TypeError(f"launcher work item {key!r} must be a string list")
    return value


def _work_dataset(item: dict[str, object]) -> DatasetName:
    value = item.get("dataset")
    if isinstance(value, DatasetName):
        return value
    if isinstance(value, str):
        return DatasetName(value)
    raise TypeError("launcher work item 'dataset' must be a dataset name")


def _work_mapping(item: dict[str, object], key: str) -> dict[str, object]:
    value = item.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"launcher work item {key!r} must be a mapping")
    return {str(part_key): part_value for part_key, part_value in value.items()}


def _mapping_int(values: Mapping[str, object], key: str, *, default: int = 0) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"launcher mapping {key!r} must be an integer")
    return value


def _step7_planner_preflight_issue_codes(
    values: Mapping[str, object],
) -> list[str]:
    """Return aggregate-only Planner accounting failures for the Step 7 prefix."""

    issues: list[str] = []
    eligible = _mapping_int(values, "eligible_candidate_count")
    if eligible == 0:
        issues.append("eligible_candidate_empty")
    if _mapping_int(values, "accounted_candidate_count") != eligible:
        issues.append("candidate_accounting_mismatch")
    for key, code in (
        ("duplicate_r0_program_count", "duplicate_r0_program"),
        ("duplicate_terminal_outcome_count", "duplicate_terminal_outcome"),
        ("unaccounted_candidate_count", "candidate_unaccounted"),
        (
            "terminal_outcome_outside_candidate_count",
            "terminal_outcome_outside_candidate",
        ),
        ("program_outside_candidate_count", "program_outside_candidate"),
        ("program_outcome_plan_mismatch_count", "program_outcome_plan_mismatch"),
        ("rejection_outcome_with_program_count", "rejection_outcome_with_program"),
        (
            "terminal_program_outcome_without_program_count",
            "terminal_program_without_program",
        ),
    ):
        if _mapping_int(values, key) != 0:
            issues.append(code)
    if _mapping_int(values, "answerable_program_count") == 0:
        issues.append("answerable_program_empty")
    return issues


def _step7_r0_preflight_record(
    *,
    dataset: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    return {
        "answerable_candidate_count": _mapping_int(manifest, "answerable_candidate_count"),
        "dataset": dataset,
        "downstream_viability": bool(manifest.get("downstream_viability")),
        "execution_count": _mapping_int(manifest, "execution_count"),
        "hard_gate_passed": bool(manifest.get("hard_gate_passed")),
        "program_count": _mapping_int(manifest, "program_count"),
        "protocol_issue_count": len(
            _object_list(
                manifest.get("protocol_integrity_issues", []),
                locator="R0 protocol_integrity_issues",
            )
        ),
    }


def _step7_r0_preflight_issue_codes(values: Mapping[str, object]) -> list[str]:
    issues: list[str] = []
    if not bool(values.get("hard_gate_passed")):
        issues.append("protocol_hard_gate_failed")
    if _mapping_int(values, "protocol_issue_count") != 0:
        issues.append("protocol_integrity_issue_present")
    if not bool(values.get("downstream_viability")):
        issues.append("downstream_viability_empty")
    if _mapping_int(values, "answerable_candidate_count") == 0:
        issues.append("answerable_candidate_empty")
    if _mapping_int(values, "program_count") == 0:
        issues.append("program_empty")
    if _mapping_int(values, "execution_count") == 0:
        issues.append("execution_empty")
    return issues


def _object_list(value: object, *, locator: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{locator} must be a list")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _performance_snapshot() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        import psutil
    except ImportError:
        read_bytes, write_bytes = 0, 0
    else:
        try:
            io = psutil.Process().io_counters()
            read_bytes, write_bytes = int(io.read_bytes), int(io.write_bytes)
        except (AttributeError, OSError, psutil.Error):
            read_bytes, write_bytes = 0, 0
    rss_scale = 1 if sys.platform == "darwin" else 1024
    return {
        "cpu_user_seconds": float(usage.ru_utime),
        "cpu_system_seconds": float(usage.ru_stime),
        "peak_rss_bytes": int(usage.ru_maxrss) * rss_scale,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
    }


def _stage_starts_managed_model(stage_id: str) -> bool:
    return stage_id in {
        "questioner:model-stage",
        "qbank_text:model-stage",
        "qbank_multimodal:model-stage",
        "compiler_program:model-stage",
        "doubter:model-stage",
        "program_text:model-stage",
        "program_multimodal:model-stage",
        "qbank_semantic:model-stage",
        "answer:model-stage",
        "evaluation_judge:model-stage",
    } or (stage_id.startswith("retrieval:") and stage_id.endswith(":model-stage"))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            payload = value.to_dict() if hasattr(value, "to_dict") else value
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


def _write_yaml(path: Path, value: object) -> None:
    import yaml  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        value.model_dump(mode="json", exclude_none=True) if hasattr(value, "model_dump") else value
    )
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_launcher_config(path: Path) -> ExperimentLauncherConfig:
    import yaml  # type: ignore[import-untyped]

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment launcher config root must be a mapping")
    return ExperimentLauncherConfig.model_validate(raw)


def _resolve(settings: Settings, path: Path) -> Path:
    return path if path.is_absolute() else settings.project_root / path


def _program_pilot_benchmark_root(settings: Settings) -> Path:
    """Return the common dataset parent expected by portable review image refs."""

    dataset_paths = {
        DatasetName.MEMLENS: settings.memlens_path.resolve(),
        DatasetName.MEMGALLERY: settings.memgallery_path.resolve(),
    }
    parents = {path.parent for path in dataset_paths.values()}
    if len(parents) != 1 or any(
        path.name != dataset.value for dataset, path in dataset_paths.items()
    ):
        raise ValueError(
            "program pilot benchmark roots must be sibling directories named "
            "'memlens' and 'memgallery'"
        )
    return next(iter(parents))


def _questioner_config_path(
    settings: Settings,
    config: ExperimentLauncherConfig,
) -> Path:
    return _resolve(
        settings,
        config.questioner_config_path or settings.questioner_config_path,
    )


def _launcher_source_inputs(
    settings: Settings, config: ExperimentLauncherConfig
) -> dict[str, JsonDict]:
    registry_path = _resolve(
        settings,
        config.model_registry_path or settings.resolved_model_registry_path,
    )
    overlay_paths = [_resolve(settings, path) for path in config.model_registry_overlays]
    registry = load_model_registry_with_overlays(registry_path, overlay_paths)
    required_aliases = config.models.required_aliases(
        config.methods,
        include_judge=config.evaluate,
        target=config.target,
    )
    paths: list[tuple[str, Path]] = [
        ("runtime_profile", _resolve(settings, config.runtime_profile_path)),
    ]
    if config.execution_profile_path is not None:
        paths.append(("execution_profile", _resolve(settings, config.execution_profile_path)))
    direct_inputs: dict[str, JsonDict] = {}
    if _is_questioner_target(config.target):
        from pmmc.compiler.questioner_config import load_questioner_config
        from pmmc.compiler.questioner_prompt import (
            questioner_prompt_sha256,
            questioner_response_schema_sha256,
        )

        questioner_config_path = _questioner_config_path(settings, config)
        questioner = load_questioner_config(questioner_config_path)
        paths.append(("questioner_config", questioner_config_path))
        if config.target is ExperimentTarget.PMMC_QUESTIONER_PILOT:
            assert config.questioner_pilot_config_path is not None
            paths.append(
                (
                    "questioner_pilot_config",
                    _resolve(settings, config.questioner_pilot_config_path),
                )
            )
        elif _is_program_pilot_target(config.target):
            assert config.questioner_pilot_config_path is not None
            paths.append(
                (
                    "questioner_pilot_config",
                    _resolve(settings, config.questioner_pilot_config_path),
                )
            )
        if _is_program_target(config.target):
            from pmmc.compiler.planner_config import load_planner_config
            from pmmc.compiler.planner_prompt import (
                planner_prompt_sha256,
                planner_response_schema_sha256,
            )

            assert config.planner_config_path is not None
            assert config.compiler_foundation_config_path is not None
            planner_path = _resolve(settings, config.planner_config_path)
            foundation_path = _resolve(settings, config.compiler_foundation_config_path)
            planner = load_planner_config(planner_path)
            paths.extend(
                [
                    ("planner_config", planner_path),
                    ("compiler_config", foundation_path),
                ]
            )
            if _is_verified_qbank_target(config.target):
                assert config.doubter_config_path is not None
                doubter_path = _resolve(settings, config.doubter_config_path)
                paths.append(("doubter_config", doubter_path))
                from pmmc.compiler.doubter_config import load_doubter_config
                from pmmc.compiler.doubter_prompt import (
                    doubter_prompt_sha256,
                    doubter_response_schema_sha256,
                )

                doubter = load_doubter_config(doubter_path)
                direct_inputs["doubter_prompt"] = {
                    "sha256": doubter_prompt_sha256(doubter),
                    "version": doubter.prompt_version,
                    "identity": True,
                }
                direct_inputs["doubter_response_schema"] = {
                    "sha256": doubter_response_schema_sha256(doubter),
                    "version": doubter.schema_version,
                    "identity": True,
                }
            if _is_semantic_program_target(config.target):
                assert config.qbank_semantic_config_path is not None
                semantic_path = _resolve(
                    settings,
                    config.qbank_semantic_config_path,
                )
                paths.append(("qbank_semantic_config", semantic_path))
                from pmmc.evaluation.qbank_semantic import (
                    load_qbank_semantic_config,
                    qbank_semantic_prompt_hash,
                    qbank_semantic_response_schema_hash,
                )

                semantic_config = load_qbank_semantic_config(semantic_path)
                direct_inputs["qbank_semantic_prompt"] = {
                    "sha256": qbank_semantic_prompt_hash(semantic_config),
                    "version": semantic_config.prompt_version,
                    "identity": True,
                }
                direct_inputs["qbank_semantic_response_schema"] = {
                    "sha256": qbank_semantic_response_schema_hash(semantic_config),
                    "version": semantic_config.schema_version,
                    "identity": True,
                }
            if config.target is ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT:
                from pmmc.compiler.config import load_compiler_config

                foundation = load_compiler_config(foundation_path)
                if not (
                    questioner.is_v34
                    and planner.is_v13
                    and doubter.is_v11
                    and semantic_config.is_v3
                    and foundation.operator_set_version == "pmmc-operators-v5"
                    and foundation.dry_run_policy == "evidence_driven_v5"
                    and foundation.r_max == 3
                ):
                    raise ValueError(
                        "Step 7 executor pilot requires Questioner v3.4, "
                        "Planner v1.3, Doubter v1.1, Semantic Judge v3, "
                        "operators v5, evidence-driven dry-run v5, and r_max=3"
                    )
            elif config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
                from pmmc.compiler.config import load_compiler_config

                foundation = load_compiler_config(foundation_path)
                if not (
                    questioner.is_v35
                    and planner.is_v14
                    and doubter.is_v12
                    and semantic_config.is_v4
                    and foundation.operator_set_version == "pmmc-operators-v6"
                    and foundation.dry_run_policy == "evidence_driven_v6"
                    and foundation.r_max == 3
                ):
                    raise ValueError(
                        "Step 7B executor pilot requires Questioner v3.5, "
                        "Planner v1.4, Doubter v1.2, Semantic Judge v4, "
                        "operators v6, evidence-driven dry-run v6, and r_max=3"
                    )
                from pmmc.runtime.qbank_reranker import (
                    RERANKER_PROMPT,
                    RERANKER_PROMPT_VERSION,
                )
                from pmmc.schemas import QBankRerankerOutput

                direct_inputs["qbank_reranker_prompt"] = {
                    "sha256": hashlib.sha256(RERANKER_PROMPT.encode("utf-8")).hexdigest(),
                    "version": RERANKER_PROMPT_VERSION,
                    "identity": True,
                }
                direct_inputs["qbank_reranker_response_schema"] = {
                    "sha256": _stable_hash(QBankRerankerOutput.model_json_schema()),
                    "version": "qbank-reranker-schema-v1",
                    "identity": True,
                }
            direct_inputs["planner_prompt"] = {
                "sha256": planner_prompt_sha256(planner),
                "version": planner.prompt_version,
                "identity": True,
            }
            direct_inputs["planner_response_schema"] = {
                "sha256": planner_response_schema_sha256(planner),
                "version": planner.schema_version,
                "identity": True,
            }
            if config.questioner_freeze_manifest_path is not None:
                paths.append(
                    (
                        "questioner_freeze_manifest",
                        _resolve(settings, config.questioner_freeze_manifest_path),
                    )
                )
        else:
            paths.append(("compiler_config", settings.resolved_compiler_config_path))
        direct_inputs["questioner_prompt"] = {
            "sha256": questioner_prompt_sha256(questioner.prompt_version),
            "identity": True,
        }
        if questioner.is_v2 or questioner.is_v3:
            direct_inputs["questioner_response_schema"] = {
                "sha256": questioner_response_schema_sha256(questioner.prompt_version),
                "identity": True,
            }
        if questioner.is_v3:
            direct_inputs["questioner_verifier_prompt"] = {
                "sha256": questioner_prompt_sha256(questioner.verifier_prompt_version),
                "version": questioner.verifier_prompt_version,
                "identity": True,
            }
            direct_inputs["questioner_verifier_response_schema"] = {
                "sha256": questioner_response_schema_sha256(questioner.verifier_prompt_version),
                "version": questioner.verifier_schema_version,
                "identity": True,
            }
    else:
        from pmmc.baselines.config import load_full_context_config
        from pmmc.baselines.rag_config import load_rag_config
        from pmmc.evaluation.config import load_evaluation_config

        paths.extend(
            [
                ("full_context_config", settings.resolved_full_context_config_path),
                ("rag_config", settings.resolved_rag_config_path),
                ("evaluation_config", settings.resolved_evaluation_config_path),
            ]
        )
        full = load_full_context_config(settings.resolved_full_context_config_path)
        rag = load_rag_config(settings.resolved_rag_config_path)
        evaluation = load_evaluation_config(settings.resolved_evaluation_config_path)
        paths.extend(
            [
                ("answer_prompt", full.resolved_prompt_path(settings.project_root)),
                ("rag_answer_prompt", rag.resolved_prompt_path(settings.project_root)),
                (
                    "judge_prompt",
                    evaluation.resolved_judge_prompt_path(settings.project_root),
                ),
            ]
        )
    if _is_step7_target(config.target):
        assert config.query_executor_config_path is not None
        from pmmc.runtime.query_executor_config import load_query_executor_config

        query_executor_path = _resolve(settings, config.query_executor_config_path)
        query_executor = load_query_executor_config(query_executor_path)
        if config.target is ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT:
            if query_executor.is_v2:
                raise ValueError("historical Step 7 pilot requires query executor v1")
        elif config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
            if not query_executor.is_v2:
                raise ValueError("Step 7B pilot requires query executor v2")
            if query_executor.reranker_model_alias != config.models.reranker_model:
                raise ValueError("Step 7B reranker aliases do not match")
        paths.append(
            (
                "query_executor_config",
                query_executor_path,
            )
        )
        paths.append(
            (
                "query_executor_answer_prompt",
                query_executor.resolved_answer_prompt_path(settings.project_root),
            )
        )
        existing_labels = {label for label, _path in paths}
        for label, value in (
            ("questioner_freeze_manifest", config.questioner_freeze_manifest_path),
            (
                "compiler_program_freeze_manifest",
                config.compiler_program_freeze_manifest_path,
            ),
            (
                "verified_qbank_freeze_manifest",
                config.verified_qbank_freeze_manifest_path,
            ),
            (
                "query_executor_freeze_manifest",
                config.query_executor_freeze_manifest_path,
            ),
            ("step7b_freeze_manifest", config.step7b_freeze_manifest_path),
        ):
            if value is not None and label not in existing_labels:
                paths.append((label, _resolve(settings, value)))
                existing_labels.add(label)
    inputs: dict[str, JsonDict] = {
        "model_registry_subset": {
            "sha256": registry.resolved_hash_for_aliases(required_aliases),
            "aliases": required_aliases,
            "identity": True,
        },
        "model_registry_full": {
            "sha256": registry.resolved_hash,
            "identity": False,
        },
        **direct_inputs,
    }
    container_image = os.environ.get("PMMC_CONTAINER_IMAGE", "").strip()
    if container_image:
        inputs["container_image"] = {
            "reference": container_image,
            "sha256": hashlib.sha256(container_image.encode("utf-8")).hexdigest(),
            "identity": not _is_verified_qbank_target(config.target),
        }
    audit_paths = [
        ("model_registry_source", registry_path),
        *(
            (f"model_registry_overlay_source:{index}", path)
            for index, path in enumerate(overlay_paths)
        ),
    ]
    for label, path in audit_paths:
        content = path.read_bytes()
        inputs[label] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "identity": False,
        }
    for label, path in paths:
        content = path.read_bytes()
        inputs[label] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "identity": True,
        }
    return inputs


def _source_identity(inputs: dict[str, JsonDict]) -> dict[str, JsonDict]:
    return {key: value for key, value in inputs.items() if value.get("identity", True) is not False}


def resolve_launcher_identity(
    *,
    settings: Settings,
    config: ExperimentLauncherConfig,
) -> tuple[str, str, dict[str, JsonDict]]:
    """Resolve the launch ID and its audited inputs without starting a run."""

    resolved_settings = settings.model_copy(update={"mode": config.mode})
    source_inputs = _launcher_source_inputs(resolved_settings, config)
    effective_config_hash = _stable_hash(
        {
            "launcher_config_hash": config.config_hash,
            "source_inputs": _source_identity(source_inputs),
        }
    )
    return (
        "launch-" + effective_config_hash[:24],
        effective_config_hash,
        source_inputs,
    )


def _assert_model_contracts(
    registry: ModelRegistry,
    config: ExperimentLauncherConfig,
) -> None:
    if _is_questioner_target(config.target):
        if config.models.compiler_model is None:
            raise ValueError("Questioner launches require compiler_model")
        compiler = registry.get(config.models.compiler_model)
        compiler.require(ModelCapability.CHAT_MULTIMODAL)
        if compiler.context_window is None or compiler.context_window < config.total_context_tokens:
            raise ValueError("compiler model context window is smaller than the Questioner budget")
        text_entry = registry.get(config.models.text_embedding_model)
        text_entry.require(ModelCapability.EMBED_TEXT)
        if text_entry.embedding_dimension != 4096:
            raise ValueError("Questioner text QBank embeddings must be 4096-dimensional")
        multimodal_entry = registry.get(config.models.multimodal_embedding_model)
        multimodal_entry.require(ModelCapability.EMBED_MULTIMODAL)
        if multimodal_entry.embedding_dimension != 4096:
            raise ValueError("Questioner multimodal QBank embeddings must be 4096-dimensional")
        if _is_semantic_program_target(config.target):
            judge = registry.get(config.models.judge_model)
            judge.require(ModelCapability.CHAT_TEXT)
            judge.require(ModelCapability.CHAT_MULTIMODAL)
        if _is_step7_target(config.target):
            answer = registry.get(config.models.answer_model)
            answer.require(ModelCapability.CHAT_TEXT)
            answer.require(ModelCapability.CHAT_MULTIMODAL)
            if answer.context_window is None or answer.context_window < config.total_context_tokens:
                raise ValueError("answer model context window is smaller than the Step 7 budget")
        if config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
            if config.models.reranker_model is None:
                raise ValueError("Step 7B requires reranker_model")
            reranker = registry.get(config.models.reranker_model)
            reranker.require(ModelCapability.CHAT_TEXT)
            reranker.require(ModelCapability.CHAT_MULTIMODAL)
            if (
                reranker.context_window is None
                or reranker.context_window < config.total_context_tokens
            ):
                raise ValueError("reranker model context window is smaller than the Step 7B budget")
            shared_entries = (compiler, answer, judge, reranker)
            physical_identities = {
                (
                    entry.source_repo_id,
                    entry.model_revision,
                    entry.expected_actual_model_id,
                )
                for entry in shared_entries
            }
            if len(physical_identities) != 1:
                raise ValueError(
                    "Step 7B shared chat roles must pin one physical repo/revision/model"
                )
        return
    answer = registry.get(config.models.answer_model)
    answer.require(ModelCapability.CHAT_TEXT)
    if answer.context_window is None or answer.context_window < config.total_context_tokens:
        raise ValueError("answer model context window is smaller than the experiment budget")
    if any(
        method
        in {
            BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
            BaselineMethod.NATIVE_MULTIMODAL_RAG,
        }
        for method in config.methods
    ):
        answer.require(ModelCapability.CHAT_MULTIMODAL)
    if config.evaluate:
        judge = registry.get(config.models.judge_model)
        judge.require(ModelCapability.CHAT_TEXT)
        judge.require(ModelCapability.CHAT_MULTIMODAL)
    registry.get(config.models.text_embedding_model).require(ModelCapability.EMBED_TEXT)
    contracts = (
        (
            BaselineMethod.CAPTION_TEXT_RAG,
            config.models.text_embedding_model,
            ModelCapability.EMBED_TEXT,
            ModelEndpointKind.VLLM_OPENAI,
        ),
        (
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
            config.models.contrastive_embedding_model,
            ModelCapability.EMBED_IMAGE,
            ModelEndpointKind.PMMC_CONTRASTIVE,
        ),
        (
            BaselineMethod.NATIVE_MULTIMODAL_RAG,
            config.models.multimodal_embedding_model,
            ModelCapability.EMBED_MULTIMODAL,
            ModelEndpointKind.VLLM_V2_EMBED,
        ),
    )
    for method, alias, capability, endpoint in contracts:
        if method not in config.methods:
            continue
        entry = registry.get(alias)
        entry.require(capability)
        if entry.endpoint_kind is not endpoint:
            raise ValueError(
                f"model alias {alias!r} uses {entry.endpoint_kind.value}, expected {endpoint.value}"
            )


class ModelPreparationService:
    """Lock, download, verify, and freeze only the aliases used by one launch."""

    def __init__(
        self,
        settings: Settings,
        config: ExperimentLauncherConfig,
        launch_dir: Path,
        reporter: HierarchicalProgressReporter | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.launch_dir = launch_dir
        self.reporter = reporter

    def prepare(self) -> tuple[ModelPreparationResult, ModelRegistry]:
        base_path = _resolve(
            self.settings,
            self.config.model_registry_path or self.settings.resolved_model_registry_path,
        )
        overlays = [_resolve(self.settings, path) for path in self.config.model_registry_overlays]
        registry = load_model_registry_with_overlays(base_path, overlays)
        _assert_model_contracts(registry, self.config)
        required = self.config.models.required_aliases(
            self.config.methods,
            include_judge=self.config.evaluate,
            target=self.config.target,
        )
        merged_path = self.launch_dir / "resolved" / "source-model-registry.yaml"
        write_resolved_registry(registry, merged_path)
        lock_path = self.launch_dir / "model_lock.json"
        lock = None
        candidate_paths = [lock_path]
        if self.settings.resolved_runtime_model_lock_path is not None:
            candidate_paths.append(self.settings.resolved_runtime_model_lock_path)
        candidate_paths.append(self.settings.resolved_model_lock_path)
        if self.settings.model_locks_path.is_dir():
            candidate_paths.extend(sorted(self.settings.model_locks_path.glob("*.json")))
        seen_paths: set[Path] = set()
        rejected_locks: list[str] = []
        for candidate_path in candidate_paths:
            candidate_path = candidate_path.resolve()
            if candidate_path in seen_paths or not candidate_path.is_file():
                continue
            seen_paths.add(candidate_path)
            try:
                candidate = load_model_lock(candidate_path)
                assert_paper_lock(candidate, registry, aliases=required)
                selected_entries = [
                    entry.model_copy(
                        update={"registry_entry_hash": registry.get(entry.alias).resolved_hash}
                    )
                    for entry in candidate.entries
                    if entry.alias in required
                ]
                lock = type(candidate)(
                    registry_hash=registry.resolved_hash,
                    registry_subset_hash=registry.resolved_hash_for_aliases(required),
                    selected_aliases=required,
                    entries=selected_entries,
                    created_at=candidate.created_at,
                    metadata={
                        **candidate.metadata,
                        **(
                            {"derived_from_lock": candidate_path.name}
                            if candidate_path != lock_path.resolve()
                            else {}
                        ),
                    },
                )
                # A self-resume must not change the checkpoint's lock bytes merely
                # by rewriting derivation metadata or JSON formatting.
                if candidate_path != lock_path.resolve() or lock != candidate:
                    write_model_lock(lock, lock_path)
                break
            except (KeyError, ValueError) as exc:
                if self.config.download_policy is DownloadPolicy.NEVER:
                    rejected_locks.append(f"{candidate_path.name}: {exc}")
                    continue
        if lock is None:
            if self.config.download_policy is DownloadPolicy.NEVER:
                detail = (
                    "; rejected lock candidates: " + "; ".join(rejected_locks)
                    if rejected_locks
                    else ""
                )
                raise FileNotFoundError(
                    "offline model preparation requires an existing valid lock" + detail
                )
            token = self.settings.hf_token.get_secret_value() if self.settings.hf_token else None
            _set_hf_online()
            lock = build_model_lock(registry, aliases=required, token=token)
            write_model_lock(lock, lock_path)

        cache_states: dict[str, str] = {}

        def progress(alias: str, state: str, current: int, total: int) -> None:
            previous = cache_states.get(alias)
            if state == "verified":
                cache_states[alias] = "downloaded" if previous == "downloading" else "cached"
            else:
                cache_states[alias] = state
            if self.reporter is not None:
                self.reporter.model_progress(alias, state, current, total)

        token = self.settings.hf_token.get_secret_value() if self.settings.hf_token else None
        offline = self.config.download_policy is DownloadPolicy.NEVER
        if not offline:
            _set_hf_online()
        verification_policy = self.config.model_verification_policy
        if self.config.download_policy is DownloadPolicy.ALWAYS_REVERIFY:
            verification_policy = ModelVerificationPolicy.FULL
        elif verification_policy is None:
            verification_policy = (
                ModelVerificationPolicy.CACHED_RECEIPT
                if self.config.mode is AppMode.PAPER
                else ModelVerificationPolicy.IDENTITY_ONLY
            )
        preparation = prepare_model_cache_audited(
            lock,
            self.settings.model_cache_path,
            token=token,
            aliases=required,
            local_files_only=offline,
            progress=progress,
            verification_policy=verification_policy,
            allow_full_fallback=not offline,
        )
        verified_cache_manifest_path = self.launch_dir / "resolved" / "model-cache-preparation.json"
        verified_cache_manifest = write_verified_model_cache_manifest(
            destination=verified_cache_manifest_path,
            lock_path=lock_path,
            lock=lock,
            preparation=preparation,
            verification_policy=verification_policy,
        )
        resolved_source = resolve_registry_from_lock(registry, lock)
        # Child commands must see only the aliases covered by this launch's
        # subset lock. Keeping unrelated source-registry entries here would
        # make paper-mode validation reject their intentionally floating
        # revisions even though those models are never loaded by the launch.
        resolved = resolved_source.model_copy(
            update={
                "models": [resolved_source.get(alias) for alias in required],
            }
        )
        resolved_path = self.launch_dir / "resolved" / "model-registry.yaml"
        write_resolved_registry(resolved, resolved_path)
        _set_hf_offline()
        if lock.registry_subset_hash is None:
            subset_hash = registry.resolved_hash_for_aliases(required)
        else:
            subset_hash = lock.registry_subset_hash
        model_sizes = {
            entry.alias: sum(item.size_bytes for item in entry.files)
            for entry in lock.entries
            if entry.alias in required
        }
        result = ModelPreparationResult(
            lock_path=lock_path,
            merged_registry_path=merged_path,
            resolved_registry_path=resolved_path,
            required_aliases=required,
            cache_states={alias: cache_states.get(alias, "verified") for alias in required},
            model_size_bytes=model_sizes,
            downloaded_bytes=sum(
                model_sizes[alias] for alias, state in cache_states.items() if state == "downloaded"
            ),
            cached_bytes=sum(
                model_sizes[alias] for alias, state in cache_states.items() if state == "cached"
            ),
            registry_hash=registry.resolved_hash,
            registry_subset_hash=subset_hash,
            verification_policy=verification_policy,
            verification_receipts={
                alias: verification.receipt_id
                for alias, verification in preparation.verifications.items()
                if verification.receipt_id is not None
            },
            verification_cache_hits=sorted(
                alias
                for alias, verification in preparation.verifications.items()
                if verification.receipt_cache_hit
            ),
            verification_bytes_hashed=sum(
                verification.bytes_hashed for verification in preparation.verifications.values()
            ),
            verified_cache_manifest_path=verified_cache_manifest_path,
            verified_cache_manifest_hash=verified_cache_manifest.content_hash,
        )
        _atomic_json(self.launch_dir / "model_selection.json", result)
        return result, resolved


def _set_hf_online() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(name, None)


def _set_hf_offline() -> None:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )


def build_execution_plan(
    settings: Settings,
    config: ExperimentLauncherConfig,
) -> ExperimentExecutionPlan:
    def finish_plan(values: list[ExperimentPlanItem]) -> ExperimentExecutionPlan:
        if _uses_stage_major_launcher(config.launcher_schema_version):
            weights = {
                "preflight": 0.5,
                "model_prepare": 2.0,
                "infrastructure": 0.5,
                "model_calibration": 1.0,
                "partition": 0.5,
                "pilot_sample": 0.5,
                "automatic_quality": 1.0,
                "review_bundle": 1.0,
                "export_verify": 0.5,
                "final_report": 0.5,
                "cleanup": 0.5,
            }

            def weight(item: ExperimentPlanItem) -> float:
                if item.stage_id in weights:
                    return weights[item.stage_id]
                model_stage_weights = {
                    "qbank_text:model-stage": 12.0,
                    "qbank_multimodal:model-stage": 16.0,
                    "compiler_program:model-stage": 38.0,
                    "doubter:model-stage": 28.0,
                    "program_text:model-stage": 18.0,
                    "program_multimodal:model-stage": 22.0,
                    "qbank_semantic:model-stage": 14.0,
                    "evaluation_deterministic:model-stage": 4.0,
                    "answer:model-stage": 24.0,
                    "evaluation_judge:model-stage": 10.0,
                    "evaluation_analyze:model-stage": 1.0,
                }
                if item.stage_id in model_stage_weights:
                    return model_stage_weights[item.stage_id]
                prefix = item.stage_id.split(":", 1)[0]
                return {
                    "benchmark": 1.0,
                    "snapshot_import": 2.0,
                    "corpus": 3.0,
                    "unit_prepare": 8.0,
                    "questioner": 30.0,
                    "qbank_index_text": 12.0,
                    "qbank_index_multimodal": 16.0,
                    "qbank_embed_text": 4.0,
                    "qbank_embed_multimodal": 5.0,
                    "qbank_coverage": 6.0,
                    "retrieval": 25.0,
                    "answer": 25.0,
                    "evaluation": 15.0,
                    "export": 1.0,
                    "questioner_verify": 0.5,
                    "planner_verify": 0.5,
                    "dry_run": 10.0,
                    "verified_qbank": 2.0,
                    "step7_prefix": 8.0,
                    "query_match": 8.0,
                    "query_execute": 14.0,
                    "query_verify": 1.0,
                    "step7_export": 2.0,
                }.get(prefix, 1.0)

            values = [item.model_copy(update={"stage_weight": weight(item)}) for item in values]
        return ExperimentExecutionPlan(
            launch_id=config.launch_id,
            config_hash=config.config_hash,
            items=values,
        )

    required_aliases = config.models.required_aliases(
        config.methods,
        include_judge=config.evaluate,
        target=config.target,
    )
    items = [
        ExperimentPlanItem(stage_id="preflight", work_units=1),
        ExperimentPlanItem(stage_id="model_prepare", work_units=max(1, len(required_aliases))),
        ExperimentPlanItem(stage_id="infrastructure", work_units=1),
        ExperimentPlanItem(stage_id="model_calibration", work_units=max(1, len(required_aliases))),
    ]
    if _is_pilot_target(config.target):
        from pmmc.pilots.config import load_questioner_pilot_config

        assert config.questioner_pilot_config_path is not None
        pilot_config = load_questioner_pilot_config(
            _resolve(settings, config.questioner_pilot_config_path)
        )
        dataset_counts: dict[DatasetName, dict[str, int]] = {}
        for dataset in config.selection.datasets:
            root = (
                settings.memlens_path
                if dataset is DatasetName.MEMLENS
                else settings.memgallery_path
            )
            adapter = create_adapter(
                dataset,
                root=root,
                project_root=settings.benchmark_asset_path,
            )
            counts = adapter.expected_counts()
            dataset_counts[dataset] = counts
            items.extend(
                [
                    ExperimentPlanItem(
                        stage_id=f"benchmark:{dataset.value}",
                        work_units=max(1, counts["source_records"] + counts["assets"]),
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"snapshot_import:{dataset.value}",
                        work_units=max(1, counts["memory_units"] + counts["queries"]),
                        dataset=dataset,
                    ),
                ]
            )
            if _is_step7_pilot_target(config.target):
                items.append(
                    ExperimentPlanItem(
                        stage_id=f"corpus:{dataset.value}",
                        work_units=max(1, counts["sessions"]),
                        dataset=dataset,
                    )
                )
        items.extend(
            [
                ExperimentPlanItem(stage_id="partition", work_units=1),
                ExperimentPlanItem(
                    stage_id="pilot_sample",
                    work_units=max(1, pilot_config.queries_per_dataset * len(dataset_counts)),
                ),
            ]
        )
        if _is_step7_pilot_target(config.target):
            items.append(
                ExperimentPlanItem(
                    stage_id="step7_prefix",
                    work_units=max(1, 2 * len(dataset_counts)),
                )
            )
        if _uses_stage_major_launcher(config.launcher_schema_version):
            for dataset in dataset_counts:
                items.append(
                    ExperimentPlanItem(
                        stage_id=f"unit_prepare:{dataset.value}",
                        work_units=pilot_config.queries_per_dataset,
                        dataset=dataset,
                    )
                )
            program_pilot = _is_program_pilot_target(config.target)
            verified_qbank = _is_verified_qbank_target(config.target)
            items.extend(
                [
                    ExperimentPlanItem(
                        stage_id=(
                            "compiler_program:model-stage"
                            if program_pilot
                            else "questioner:model-stage"
                        ),
                        work_units=max(1, pilot_config.queries_per_dataset * len(dataset_counts)),
                    ),
                    ExperimentPlanItem(
                        stage_id=(
                            "program_text:model-stage"
                            if program_pilot
                            else "qbank_text:model-stage"
                        ),
                        work_units=max(
                            1,
                            pilot_config.queries_per_dataset
                            * (4 if program_pilot else 2)
                            * len(dataset_counts),
                        ),
                    ),
                    ExperimentPlanItem(
                        stage_id=(
                            "program_multimodal:model-stage"
                            if program_pilot
                            else "qbank_multimodal:model-stage"
                        ),
                        work_units=max(
                            1,
                            pilot_config.queries_per_dataset
                            * (4 if program_pilot else 2)
                            * len(dataset_counts),
                        ),
                    ),
                ]
            )
            for dataset in dataset_counts:
                if program_pilot:
                    items.append(
                        ExperimentPlanItem(
                            stage_id=f"dry_run:{dataset.value}",
                            work_units=pilot_config.queries_per_dataset,
                            dataset=dataset,
                        )
                    )
            if verified_qbank:
                items.append(
                    ExperimentPlanItem(
                        stage_id="doubter:model-stage",
                        work_units=max(
                            1,
                            pilot_config.queries_per_dataset * len(dataset_counts) * 4,
                        ),
                    )
                )
                items.extend(
                    ExperimentPlanItem(
                        stage_id=f"verified_qbank:{dataset.value}",
                        work_units=max(1, pilot_config.queries_per_dataset),
                        dataset=dataset,
                    )
                    for dataset in dataset_counts
                )
            if _is_step7_pilot_target(config.target):
                for dataset in dataset_counts:
                    items.extend(
                        [
                            ExperimentPlanItem(
                                stage_id=f"query_match:{dataset.value}",
                                work_units=pilot_config.queries_per_dataset,
                                dataset=dataset,
                            ),
                            ExperimentPlanItem(
                                stage_id=f"query_execute:{dataset.value}",
                                work_units=pilot_config.queries_per_dataset * 4,
                                dataset=dataset,
                            ),
                        ]
                    )
                query_total = pilot_config.queries_per_dataset * len(dataset_counts)
                items.extend(
                    [
                        ExperimentPlanItem(
                            stage_id="answer:model-stage",
                            work_units=query_total * 5,
                        ),
                        ExperimentPlanItem(
                            stage_id="evaluation_deterministic:model-stage",
                            work_units=query_total,
                        ),
                        ExperimentPlanItem(
                            stage_id="evaluation_judge:model-stage",
                            work_units=query_total * 6,
                        ),
                        ExperimentPlanItem(
                            stage_id="evaluation_analyze:model-stage",
                            work_units=query_total,
                        ),
                    ]
                )
                items.extend(
                    ExperimentPlanItem(
                        stage_id=f"query_verify:{dataset.value}",
                        work_units=1,
                        dataset=dataset,
                    )
                    for dataset in dataset_counts
                )
            for dataset in dataset_counts:
                items.extend(
                    [
                        ExperimentPlanItem(
                            stage_id=(
                                f"planner_verify:{dataset.value}"
                                if program_pilot
                                else f"questioner_verify:{dataset.value}"
                            ),
                            work_units=1,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"qbank_coverage:{dataset.value}",
                            work_units=pilot_config.queries_per_dataset,
                            dataset=dataset,
                        ),
                    ]
                )
            if _is_semantic_program_target(config.target):
                items.append(
                    ExperimentPlanItem(
                        stage_id="qbank_semantic:model-stage",
                        work_units=max(
                            1,
                            pilot_config.queries_per_dataset * len(dataset_counts),
                        ),
                    )
                )
            for dataset in dataset_counts:
                items.append(
                    ExperimentPlanItem(
                        stage_id=f"export:{dataset.value}",
                        work_units=1,
                        dataset=dataset,
                    )
                )
        else:
            for dataset, counts in dataset_counts.items():
                unit_count = max(1, counts["sessions"])
                items.extend(
                    [
                        ExperimentPlanItem(
                            stage_id=f"questioner:{dataset.value}",
                            work_units=unit_count,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"qbank_index_text:{dataset.value}",
                            work_units=unit_count,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"qbank_index_multimodal:{dataset.value}",
                            work_units=unit_count,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"questioner_verify:{dataset.value}",
                            work_units=1,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"qbank_coverage:{dataset.value}",
                            work_units=pilot_config.queries_per_dataset,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"export:{dataset.value}", work_units=1, dataset=dataset
                        ),
                    ]
                )
        items.extend(
            [
                ExperimentPlanItem(stage_id="automatic_quality", work_units=1),
                ExperimentPlanItem(
                    stage_id="review_bundle",
                    work_units=(
                        (
                            152
                            if config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
                            else 144
                        )
                        if _is_step7_pilot_target(config.target)
                        else 200
                        if _is_program_pilot_target(config.target)
                        else pilot_config.candidate_review_count
                        + pilot_config.coverage_review_count
                    ),
                ),
                ExperimentPlanItem(stage_id="export_verify", work_units=1),
                ExperimentPlanItem(stage_id="final_report", work_units=1),
                ExperimentPlanItem(stage_id="cleanup", work_units=1),
            ]
        )
        return finish_plan(items)
    if config.target in {
        ExperimentTarget.PMMC_QUESTIONER,
        ExperimentTarget.PMMC_COMPILER_PROGRAMS,
        ExperimentTarget.PMMC_COMPILE_ONLY,
    }:
        program_target = _is_program_target(config.target)
        questioner_datasets: list[tuple[DatasetName, int, int]] = []
        for dataset in config.selection.datasets:
            root = (
                settings.memlens_path
                if dataset is DatasetName.MEMLENS
                else settings.memgallery_path
            )
            adapter = create_adapter(
                dataset,
                root=root,
                project_root=settings.benchmark_asset_path,
            )
            counts = adapter.expected_counts()
            selected_queries = (
                config.selection.limit or config.selection.sample_size or counts["eligible_queries"]
            )
            selected_queries = max(
                1,
                (selected_queries + config.selection.shard_count - 1)
                // config.selection.shard_count,
            )
            unit_count = max(1, counts["sessions"])
            questioner_datasets.append((dataset, unit_count, selected_queries))
            items.extend(
                [
                    ExperimentPlanItem(
                        stage_id=f"benchmark:{dataset.value}",
                        work_units=max(1, counts["source_records"] + counts["assets"]),
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"snapshot_import:{dataset.value}",
                        work_units=max(1, counts["memory_units"] + counts["queries"]),
                        dataset=dataset,
                    ),
                ]
            )
            if not _uses_stage_major_launcher(config.launcher_schema_version):
                items.extend(
                    [
                        ExperimentPlanItem(
                            stage_id=f"questioner:{dataset.value}",
                            work_units=unit_count,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"qbank_index_text:{dataset.value}",
                            work_units=unit_count,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"qbank_index_multimodal:{dataset.value}",
                            work_units=unit_count,
                            dataset=dataset,
                        ),
                        ExperimentPlanItem(
                            stage_id=f"questioner_verify:{dataset.value}",
                            work_units=1,
                            dataset=dataset,
                        ),
                    ]
                )
                if config.evaluate:
                    items.append(
                        ExperimentPlanItem(
                            stage_id=f"qbank_coverage:{dataset.value}",
                            work_units=selected_queries,
                            dataset=dataset,
                        )
                    )
                items.append(
                    ExperimentPlanItem(
                        stage_id=f"export:{dataset.value}",
                        work_units=1,
                        dataset=dataset,
                    )
                )
        if _uses_stage_major_launcher(config.launcher_schema_version):
            unit_total = sum(value[1] for value in questioner_datasets)
            query_total = sum(value[2] for value in questioner_datasets)
            items.extend(
                ExperimentPlanItem(
                    stage_id=f"unit_prepare:{dataset.value}",
                    work_units=unit_count,
                    dataset=dataset,
                )
                for dataset, unit_count, _selected_queries in questioner_datasets
            )
            items.extend(
                [
                    ExperimentPlanItem(
                        stage_id=(
                            "compiler_program:model-stage"
                            if program_target
                            else "questioner:model-stage"
                        ),
                        work_units=max(1, unit_total),
                    ),
                    ExperimentPlanItem(
                        stage_id=(
                            "program_text:model-stage"
                            if program_target
                            else "qbank_text:model-stage"
                        ),
                        work_units=max(
                            1,
                            (unit_total * (3 if program_target else 1))
                            + (query_total if config.evaluate else 0),
                        ),
                    ),
                    ExperimentPlanItem(
                        stage_id=(
                            "program_multimodal:model-stage"
                            if program_target
                            else "qbank_multimodal:model-stage"
                        ),
                        work_units=max(
                            1,
                            (unit_total * (3 if program_target else 1))
                            + (query_total if config.evaluate else 0),
                        ),
                    ),
                ]
            )
            if program_target:
                items.extend(
                    ExperimentPlanItem(
                        stage_id=f"dry_run:{dataset.value}",
                        work_units=max(1, _selected_queries),
                        dataset=dataset,
                    )
                    for dataset, _unit_count, _selected_queries in questioner_datasets
                )
                if _is_verified_qbank_target(config.target):
                    items.append(
                        ExperimentPlanItem(
                            stage_id="doubter:model-stage",
                            work_units=max(1, unit_total * 4),
                        )
                    )
                    items.extend(
                        ExperimentPlanItem(
                            stage_id=f"verified_qbank:{dataset.value}",
                            work_units=max(1, _selected_queries),
                            dataset=dataset,
                        )
                        for dataset, _unit_count, _selected_queries in questioner_datasets
                    )
            for dataset, _unit_count, _selected_queries in questioner_datasets:
                items.append(
                    ExperimentPlanItem(
                        stage_id=(
                            f"planner_verify:{dataset.value}"
                            if program_target
                            else f"questioner_verify:{dataset.value}"
                        ),
                        work_units=1,
                        dataset=dataset,
                    )
                )
                if config.evaluate:
                    continue
                items.append(
                    ExperimentPlanItem(
                        stage_id=f"export:{dataset.value}",
                        work_units=1,
                        dataset=dataset,
                    )
                )
            if config.evaluate:
                items.extend(
                    ExperimentPlanItem(
                        stage_id=f"qbank_coverage:{dataset.value}",
                        work_units=selected_queries,
                        dataset=dataset,
                    )
                    for dataset, _unit_count, selected_queries in questioner_datasets
                )
                items.extend(
                    ExperimentPlanItem(
                        stage_id=f"export:{dataset.value}",
                        work_units=1,
                        dataset=dataset,
                    )
                    for dataset, _unit_count, _selected_queries in questioner_datasets
                )
        items.extend(
            [
                ExperimentPlanItem(stage_id="final_report", work_units=1),
                ExperimentPlanItem(stage_id="cleanup", work_units=1),
            ]
        )
        return finish_plan(items)
    if config.target is ExperimentTarget.PMMC_ONLINE:
        selected_counts: dict[DatasetName, int] = {}
        for dataset in config.selection.datasets:
            root = (
                settings.memlens_path
                if dataset is DatasetName.MEMLENS
                else settings.memgallery_path
            )
            counts = create_adapter(
                dataset,
                root=root,
                project_root=settings.benchmark_asset_path,
            ).expected_counts()
            explicit = sum(
                query_id.startswith(f"{dataset.value}:query:")
                for query_id in config.selection.query_ids
            )
            selected = explicit or config.selection.limit or config.selection.sample_size
            selected_counts[dataset] = max(1, selected or counts["eligible_queries"])
            items.extend(
                [
                    ExperimentPlanItem(
                        stage_id=f"benchmark:{dataset.value}",
                        work_units=max(1, counts["source_records"] + counts["assets"]),
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"snapshot_import:{dataset.value}",
                        work_units=max(1, counts["memory_units"] + counts["queries"]),
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"query_match:{dataset.value}",
                        work_units=selected_counts[dataset],
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"query_execute:{dataset.value}",
                        work_units=selected_counts[dataset] * 4,
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"query_verify:{dataset.value}",
                        work_units=1,
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"export:{dataset.value}",
                        work_units=1,
                        dataset=dataset,
                    ),
                ]
            )
        selected_total = sum(selected_counts.values())
        items.extend(
            [
                ExperimentPlanItem(stage_id="answer:model-stage", work_units=selected_total * 5),
                ExperimentPlanItem(
                    stage_id="evaluation_deterministic:model-stage",
                    work_units=selected_total,
                ),
                ExperimentPlanItem(
                    stage_id="qbank_semantic:model-stage",
                    work_units=selected_total,
                ),
                ExperimentPlanItem(
                    stage_id="evaluation_judge:model-stage",
                    work_units=selected_total * 2,
                ),
                ExperimentPlanItem(
                    stage_id="evaluation_analyze:model-stage",
                    work_units=selected_total,
                ),
                ExperimentPlanItem(stage_id="final_report", work_units=1),
                ExperimentPlanItem(stage_id="cleanup", work_units=1),
            ]
        )
        return finish_plan(items)
    rag_methods = [
        method
        for method in config.methods
        if method
        in {
            BaselineMethod.CAPTION_TEXT_RAG,
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
            BaselineMethod.NATIVE_MULTIMODAL_RAG,
        }
    ]
    dataset_plan: list[tuple[DatasetName, dict[str, int], int]] = []
    for dataset in config.selection.datasets:
        root = settings.memlens_path if dataset is DatasetName.MEMLENS else settings.memgallery_path
        adapter = create_adapter(
            dataset,
            root=root,
            project_root=settings.benchmark_asset_path,
        )
        counts = adapter.expected_counts()
        selected = (
            config.selection.limit or config.selection.sample_size or counts["eligible_queries"]
        )
        selected = max(
            1,
            (selected + config.selection.shard_count - 1) // config.selection.shard_count,
        )
        dataset_plan.append((dataset, counts, selected))
        items.extend(
            [
                ExperimentPlanItem(
                    stage_id=f"benchmark:{dataset.value}",
                    work_units=max(1, counts["source_records"] + counts["assets"]),
                    dataset=dataset,
                ),
                ExperimentPlanItem(
                    stage_id=f"snapshot_import:{dataset.value}",
                    work_units=max(1, counts["memory_units"] + counts["queries"]),
                    dataset=dataset,
                ),
                ExperimentPlanItem(
                    stage_id=f"corpus:{dataset.value}",
                    work_units=max(1, counts["sessions"]),
                    dataset=dataset,
                ),
            ]
        )
        if not _uses_stage_major_launcher(config.launcher_schema_version):
            items.extend(
                [
                    ExperimentPlanItem(
                        stage_id=f"retrieval:{dataset.value}",
                        work_units=max(1, selected * max(1, len(rag_methods))),
                        dataset=dataset,
                    ),
                    ExperimentPlanItem(
                        stage_id=f"answer:{dataset.value}",
                        work_units=max(1, selected * len(config.methods)),
                        dataset=dataset,
                    ),
                ]
            )
            if config.evaluate:
                items.append(
                    ExperimentPlanItem(
                        stage_id=f"evaluation:{dataset.value}",
                        work_units=max(1, selected * len(config.methods) * 3),
                        dataset=dataset,
                    )
                )
            items.append(
                ExperimentPlanItem(
                    stage_id=f"export:{dataset.value}", work_units=1, dataset=dataset
                )
            )
    if _uses_stage_major_launcher(config.launcher_schema_version):
        selected_total = sum(value[2] for value in dataset_plan)
        for method in rag_methods:
            items.append(
                ExperimentPlanItem(
                    stage_id=f"retrieval:{method.value}:model-stage",
                    work_units=max(1, selected_total),
                )
            )
        items.append(
            ExperimentPlanItem(
                stage_id="answer:model-stage",
                work_units=max(1, selected_total * len(config.methods)),
            )
        )
        if config.evaluate:
            evaluation_units = max(1, selected_total * len(config.methods))
            items.extend(
                [
                    ExperimentPlanItem(
                        stage_id="evaluation_deterministic:model-stage",
                        work_units=evaluation_units,
                    ),
                    ExperimentPlanItem(
                        stage_id="evaluation_judge:model-stage",
                        work_units=evaluation_units,
                    ),
                    ExperimentPlanItem(
                        stage_id="evaluation_analyze:model-stage",
                        work_units=evaluation_units,
                    ),
                ]
            )
        items.extend(
            ExperimentPlanItem(stage_id=f"export:{dataset.value}", work_units=1, dataset=dataset)
            for dataset, _counts, _selected in dataset_plan
        )
    items.extend(
        [
            ExperimentPlanItem(stage_id="final_report", work_units=1),
            ExperimentPlanItem(stage_id="cleanup", work_units=1),
        ]
    )
    return finish_plan(items)


class ResearchExperimentLauncher:
    """Compose existing PMMC producers into one resumable research command."""

    def __init__(
        self,
        *,
        settings: Settings,
        config: ExperimentLauncherConfig,
        resume_launch_id: str | None = None,
        import_upstream_from_launch: str | None = None,
        dry_run: bool = False,
        prepare_only: bool = False,
        evaluate_only: bool = False,
    ) -> None:
        if prepare_only and evaluate_only:
            raise ValueError("prepare_only and evaluate_only are mutually exclusive")
        self.settings = settings.model_copy(update={"mode": config.mode})
        self.config = config
        if _uses_stage_major_launcher(config.launcher_schema_version):
            assert config.execution_profile_path is not None
            self.execution_config = load_execution_config(
                _resolve(self.settings, config.execution_profile_path)
            )
        else:
            self.execution_config = ExperimentExecutionConfig().model_copy(
                update={
                    "schedule": ExecutionSchedule.DATASET_MAJOR_V1,
                    "checkpoint": ExperimentExecutionConfig().checkpoint.model_copy(
                        update={"policy": CheckpointPolicy.DISABLED}
                    ),
                }
            )
        (
            self.launch_id,
            self.effective_config_hash,
            self.source_inputs,
        ) = resolve_launcher_identity(
            settings=self.settings,
            config=config,
        )
        self.resume = resume_launch_id is not None
        if import_upstream_from_launch is not None:
            from pmmc.experiments.upstream_import import validate_import_id

            validate_import_id(import_upstream_from_launch)
            if resume_launch_id is not None or dry_run or prepare_only or evaluate_only:
                raise ValueError("upstream_import_cannot_combine_with_resume_or_partial_run")
            if config.target is not ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
                raise ValueError("upstream_import_requires_step7b_pilot")
        self.import_upstream_from_launch = import_upstream_from_launch
        self.dry_run = dry_run
        self.prepare_only = prepare_only
        self.evaluate_only = evaluate_only
        if resume_launch_id is not None and resume_launch_id != self.launch_id:
            raise ValueError("resume launch ID differs from the resolved configuration")
        self.launch_dir = self.settings.runs_path / "launches" / self.launch_id
        self.plan = build_execution_plan(self.settings, config).model_copy(
            update={
                "launch_id": self.launch_id,
                "config_hash": self.effective_config_hash,
            }
        )
        self.state_path = self.launch_dir / "launch_state.json"
        self.state = ExperimentLaunchState(launch_id=self.launch_id)
        self.reporter: HierarchicalProgressReporter | None = None
        self.launch_settings = self.settings
        self.preparation: ModelPreparationResult | None = None
        self.resolved_registry: ModelRegistry | None = None
        self._checkpoint_restored = False
        self._checkpoint_ready = False
        self._checkpoint_lock = threading.Lock()
        self._stage_performance_metadata: dict[str, JsonDict] = {}
        self._execution_attempt_id: str | None = None

    def _replica_count(self, logical_stage: str) -> int:
        return self.execution_config.replicas.for_logical_stage(logical_stage)

    def run(self) -> dict[str, object]:
        self._initialize()
        self.state = self.state.model_copy(
            update={
                "status": ExperimentLaunchStatus.RUNNING,
                "started_at": self.state.started_at or utc_now(),
                "completed_at": None,
                "error_type": None,
            }
        )
        self._save_state()
        self.reporter = HierarchicalProgressReporter(
            launch_id=self.launch_id,
            events_path=self.launch_dir / "events.jsonl",
            overall_total=self.plan.total_work_units,
            stage_weights={item.stage_id: item.stage_weight for item in self.plan.items}
            if _uses_stage_major_launcher(self.config.launcher_schema_version)
            else None,
            is_tty=(
                None if self.config.progress.non_tty is None else not self.config.progress.non_tty
            ),
            report_interval_seconds=self.config.progress.report_interval_seconds,
        )
        try:
            with self.reporter:
                self._run_preflight()
                if self.dry_run:
                    summary = self._write_final_summary(dry_run=True)
                    self._complete_state()
                    return summary
                self._run_model_prepare()
                self._resolve_runtime_and_baseline_configs()
                if self.prepare_only:
                    summary = self._write_final_summary(prepare_only=True)
                    self._complete_state()
                    return summary
                self._run_infrastructure()
                self._run_calibration()
                if self.evaluate_only and _is_pilot_target(self.config.target):
                    raise ValueError(
                        "compiler pilot evaluate-only is unsupported; resume the "
                        "content-addressed pilot launch instead"
                    )
                if self.evaluate_only and _is_program_target(self.config.target):
                    raise ValueError(
                        "Planner evaluate-only is unsupported; resume the compiler launch"
                    )
                if self.evaluate_only and self.config.target is ExperimentTarget.PMMC_QUESTIONER:
                    self._run_existing_questioner_coverage()
                elif self.evaluate_only:
                    self._run_existing_evaluations()
                elif _is_step7_pilot_target(self.config.target):
                    self._run_step7_pilot()
                elif self.config.target is ExperimentTarget.PMMC_ONLINE:
                    self._run_step7_online()
                elif _is_program_pilot_target(self.config.target):
                    self._run_program_pilot()
                elif self.config.target in {
                    ExperimentTarget.PMMC_COMPILER_PROGRAMS,
                    ExperimentTarget.PMMC_COMPILE_ONLY,
                }:
                    self._run_program_datasets()
                elif self.config.target is ExperimentTarget.PMMC_QUESTIONER_PILOT:
                    self._run_questioner_pilot()
                elif self.config.target is ExperimentTarget.PMMC_QUESTIONER:
                    self._run_questioner_datasets()
                else:
                    self._run_datasets()
                summary = self._write_final_summary()
                self._finish_stage("final_report")
                self._run_cleanup()
                summary = self._write_final_summary()
                self._complete_state()
                return summary
        except KeyboardInterrupt:
            self.state = self.state.model_copy(
                update={
                    "status": ExperimentLaunchStatus.CANCELLED,
                    "completed_at": utc_now(),
                    "error_type": None,
                }
            )
            self._save_state()
            self._signal_sidecars()
            raise
        except Exception as exc:
            if self.reporter is not None:
                self.reporter.fail_stage(exc)
            errors = _read_jsonl(self.launch_dir / "errors.jsonl")
            errors.append(
                {
                    "created_at": utc_now().isoformat(),
                    "execution_attempt_id": self._execution_attempt_id,
                    "launch_id": self.launch_id,
                    "stage": self.reporter.stage_name if self.reporter else None,
                    "error_type": type(exc).__name__,
                }
            )
            _atomic_jsonl(self.launch_dir / "errors.jsonl", errors)
            self.state = self.state.model_copy(
                update={
                    "status": ExperimentLaunchStatus.FAILED,
                    "completed_at": utc_now(),
                    "error_type": type(exc).__name__,
                }
            )
            self._save_state()
            if _is_step7_target(self.config.target):
                try:
                    self._record_step7_failure(exc)
                except Exception as capsule_exc:
                    exc.add_note(
                        "Step 7 failure-capsule generation also failed: "
                        f"{type(capsule_exc).__name__}"
                    )
            elif _is_verified_qbank_target(self.config.target):
                try:
                    self._record_step6d_failure(exc)
                except Exception as capsule_exc:
                    exc.add_note(
                        "Step 6D failure-capsule generation also failed: "
                        f"{type(capsule_exc).__name__}"
                    )
            self._signal_sidecars()
            raise

    def _record_step7_failure(self, exc: Exception) -> None:
        """Persist a small, content-addressed, non-sensitive failure capsule."""

        from pmmc.baselines.answer_upgrade import AnswerUpgradeError
        from pmmc.experiments.upstream_import import ImportedProgramAuditError
        from pmmc.infrastructure.database.errors import (
            safe_database_error_diagnostics,
        )
        from pmmc.runtime.query_workflow import QueryExecutorParentContractError

        safe_error_diagnostics: dict[str, object] = dict(safe_database_error_diagnostics(exc))
        if isinstance(exc, QueryExecutorParentContractError):
            safe_error_diagnostics.update(exc.safe_diagnostics())
        if isinstance(exc, (
            ImportedProgramAuditError, AnswerUpgradeError, Step7PilotAcceptanceError,
        )):
            safe_error_diagnostics.update(exc.safe_diagnostics())
        stage = (
            self.reporter.stage_name
            if self.reporter is not None and self.reporter.stage_name
            else "launcher"
        )
        if safe_error_diagnostics.get("preflight_phase") == "before_post_index_chat":
            stage = "query_vector_preflight"
        if isinstance(exc, ImportedProgramAuditError):
            stage = f"imported_program_audit:{exc.dataset}"
        if isinstance(exc, AnswerUpgradeError):
            stage = f"fallback_answer_upgrade:{exc.dataset}"
        if isinstance(exc, Step7PilotAcceptanceError):
            stage = exc.failure_stage
        failed_operation = isinstance(exc, (ImportedProgramAuditError, AnswerUpgradeError))
        error_code = (
            exc.error_code
            if isinstance(exc, (
                ImportedProgramAuditError, AnswerUpgradeError, Step7PilotAcceptanceError,
            ))
            else type(exc).__name__
        )
        completed = list(self.state.completed_stage_ids)
        pending = [item.stage_id for item in self.plan.items if item.stage_id not in completed]
        attempt = self._execution_attempt_id or "unrecorded-attempt"

        def attempt_metadata(item: dict[str, object]) -> dict[str, object]:
            value = item.get("metadata")
            return value if isinstance(value, dict) else {}

        attempt_performance = [
            item
            for item in _read_jsonl(self.launch_dir / "stage_performance.jsonl")
            if attempt_metadata(item).get("execution_attempt_id") == attempt
        ]
        attempt_completed = sum(
            attempt_metadata(item).get("status") == "completed" for item in attempt_performance
        )
        attempt_failed = sum(
            attempt_metadata(item).get("status") == "failed" for item in attempt_performance
        )
        attempt_reused = sum(bool(item.get("reused")) for item in attempt_performance)
        latest_checkpoint = self.state.metadata.get("latest_checkpoint")
        restore_attempt = self.state.metadata.get("checkpoint_restore_attempt")
        restore_sequence = (
            restore_attempt.get("sequence") if isinstance(restore_attempt, dict) else None
        )
        latest_milestone = (
            str(restore_sequence)
            if isinstance(restore_sequence, str) and restore_sequence
            else Path(str(latest_checkpoint)).name
            if isinstance(latest_checkpoint, str) and latest_checkpoint
            else None
        )
        config_path = os.environ.get(
            "PMMC_EXPERIMENT_CONFIG_PATH",
            "configs/experiments/pmmc-step7-gpu-pilot.yaml",
        )
        for prefix in ("/workspace/", f"{self.settings.project_root.resolve()}/"):
            if config_path.startswith(prefix):
                config_path = config_path.removeprefix(prefix)
                break
        resume_command = (
            "python3 -m pmmc experiment run "
            f"--config {config_path} --resume-launch-id {self.launch_id}"
        )
        capsule_id = (
            "step7-failure-"
            + _stable_hash([self.launch_id, attempt, stage, type(exc).__name__, completed])[:24]
        )
        capsule_dir = (
            self.settings.outputs_path / "step7-pilots" / "failures" / self.launch_id / capsule_id
        )
        capsule_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        safe_log = capsule_dir / "safe_execution.log"
        safe_lines: list[str] = []
        for name in (
            "events.jsonl",
            "stage_results.jsonl",
            "stage_components.jsonl",
            "stage_component_errors.jsonl",
            "checkpoint_events.jsonl",
            "errors.jsonl",
        ):
            source = self.launch_dir / name
            if not source.is_file():
                continue
            safe_lines.append(json.dumps({"source": name}, sort_keys=True, separators=(",", ":")))
            safe_lines.extend(source.read_text(encoding="utf-8").splitlines())
        from pmmc.compiler.program_pilot import _redact_nonportable_string

        safe_text, redaction_count = _redact_nonportable_string("\n".join(safe_lines) + "\n")
        safe_log.write_text(safe_text, encoding="utf-8")
        safe_log.chmod(0o600)
        payload: dict[str, object] = {
            "capsule_id": capsule_id,
            "launch_id": self.launch_id,
            "execution_attempt_id": attempt,
            "stage": stage,
            "substage": (
                exc.phase if isinstance(exc, AnswerUpgradeError) else stage.split(":", 1)[0]
            ),
            "dataset": (
                stage.split(":", 1)[1]
                if ":" in stage
                and stage.split(":", 1)[1]
                in {item.value for item in self.config.selection.datasets}
                else None
            ),
            "error_code": error_code,
            "completed_count": len(completed),
            "pending_count": len(pending),
            "attempt_completed_count": attempt_completed,
            "attempt_failed_count": attempt_failed,
            "attempt_operation_failed_count": int(failed_operation),
            "attempt_reused_count": attempt_reused,
            "completed_stage_ids": completed,
            "pending_stage_ids": pending,
            "safe_error_diagnostics": safe_error_diagnostics,
            "latest_compatible_milestone": latest_milestone,
            "resume_command": resume_command,
            "safe_log_redaction_count": redaction_count,
            "safe_log_sha256": hashlib.sha256(safe_log.read_bytes()).hexdigest(),
        }
        payload["content_hash"] = _stable_hash(payload)
        _atomic_json(capsule_dir / "manifest.json", payload)
        (capsule_dir / "manifest.json").chmod(0o600)
        metadata = dict(self.state.metadata)
        failure_metadata = {
            **payload,
            "capsule_dir": str(capsule_dir),
        }
        metadata["step7_failure"] = failure_metadata
        if self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
            metadata["step7b_failure"] = failure_metadata
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()
        print(
            "STEP7_FAILURE "
            + json.dumps(
                {
                    "capsule": str(capsule_dir),
                    "attempt_completed": attempt_completed,
                    "attempt_failed": attempt_failed,
                    "attempt_reused": attempt_reused,
                    "completed": len(completed),
                    "error_code": error_code,
                    "attempt_operation_failed": int(failed_operation),
                    "safe_error_diagnostics": safe_error_diagnostics,
                    "latest_compatible_milestone": latest_milestone,
                    "pending": len(pending),
                    "resume_command": resume_command,
                    "stage": stage,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    def _initialize(self) -> None:
        if self.import_upstream_from_launch is not None:
            from pmmc.experiments.upstream_import import load_source

            load_source(self, self.import_upstream_from_launch)
        if self.launch_dir.exists() and not self.resume:
            raise FileExistsError(
                f"launch already exists; resume with --resume-launch-id {self.launch_id}"
            )
        self.launch_dir.mkdir(parents=True, exist_ok=self.resume)
        if self.resume:
            if not self.state_path.is_file():
                raise FileNotFoundError("resumed launch is missing launch_state.json")
            self.state = ExperimentLaunchState.from_json(
                self.state_path.read_text(encoding="utf-8")
            )
            existing = ExperimentLaunchManifest.from_json(
                (self.launch_dir / "launch_manifest.json").read_text(encoding="utf-8")
            )
            if existing.config_hash != self.effective_config_hash:
                raise ValueError("resumed launch configuration differs from its manifest")
            plan_path = self.launch_dir / "execution_plan.json"
            if not plan_path.is_file():
                raise FileNotFoundError("resumed launch is missing execution_plan.json")
            persisted_plan = ExperimentExecutionPlan.from_json(
                plan_path.read_text(encoding="utf-8")
            )
            if (
                persisted_plan.launch_id != self.plan.launch_id
                or persisted_plan.config_hash != self.plan.config_hash
                or len(persisted_plan.items) != len(self.plan.items)
            ):
                raise ValueError("resumed execution plan identity differs")
            for expected, persisted in zip(self.plan.items, persisted_plan.items, strict=True):
                if persisted.model_dump(exclude={"actual_work_units"}) != (
                    expected.model_dump(exclude={"actual_work_units"})
                ):
                    raise ValueError("resumed execution plan structure differs")
            self.plan = persisted_plan
            self._record_execution_attempt()
            return
        required = self.config.models.required_aliases(
            self.config.methods,
            include_judge=self.config.evaluate,
            target=self.config.target,
        )
        manifest = ExperimentLaunchManifest(
            launch_id=self.launch_id,
            config_hash=self.effective_config_hash,
            experiment_name=self.config.experiment_name,
            mode=self.config.mode,
            methods=self.config.methods,
            datasets=self.config.selection.datasets,
            required_model_aliases=required,
            metadata={
                "paper_result": (
                    self.config.mode is AppMode.PAPER
                    and not _is_pilot_target(self.config.target)
                    and bool(self.config.metadata.get("paper_result", True))
                ),
                "launcher_config_hash": self.config.config_hash,
                "source_inputs_hash": _stable_hash(_source_identity(self.source_inputs)),
            },
        )
        _atomic_json(self.launch_dir / "launch_manifest.json", manifest)
        _atomic_json(self.launch_dir / "execution_plan.json", self.plan)
        _atomic_json(self.launch_dir / "source_inputs.json", self.source_inputs)
        _write_yaml(self.launch_dir / "resolved_config.yaml", self.config)
        if self.import_upstream_from_launch is not None:
            self.state.metadata["upstream_import_source"] = self.import_upstream_from_launch
        self._save_state()
        self._record_execution_attempt()

    def _record_execution_attempt(self) -> None:
        if not _is_verified_qbank_target(self.config.target):
            return
        path = self.launch_dir / "execution_attempts.jsonl"
        values = _read_jsonl(path)
        created_at = utc_now()
        image = os.environ.get("PMMC_CONTAINER_IMAGE", "").strip() or ("unrecorded-execution-image")
        payload = {
            "attempt_ordinal": len(values),
            "created_at": created_at.isoformat(),
            "image_digest": image,
            "launch_id": self.launch_id,
            "resume": self.resume,
            "import_upstream_from_launch": self.import_upstream_from_launch,
            "scientific_config_hash": self.effective_config_hash,
        }
        payload["attempt_id"] = "execution-attempt-" + _stable_hash(payload)[:24]
        self._execution_attempt_id = str(payload["attempt_id"])
        values.append(payload)
        _atomic_jsonl(path, values)

    def _record_step6d_failure(self, exc: Exception) -> None:
        from pmmc.compiler.step6d_failure import (
            build_step6d_runtime_failure_capsule,
        )

        stage = (
            self.reporter.stage_name
            if self.reporter is not None and self.reporter.stage_name
            else "launcher"
        )
        completed = list(self.state.completed_stage_ids)
        pending = [item.stage_id for item in self.plan.items if item.stage_id not in completed]
        latest_checkpoint = self.state.metadata.get("latest_checkpoint")
        latest_milestone = (
            Path(latest_checkpoint).name
            if isinstance(latest_checkpoint, str) and latest_checkpoint
            else None
        )
        config_path = os.environ.get(
            "PMMC_EXPERIMENT_CONFIG_PATH",
            "/workspace/configs/experiments/pmmc-compiler-verified-qbank-pilot.yaml",
        )
        workspace_prefix = "/workspace/"
        project_prefix = f"{self.settings.project_root.resolve()}/"
        if config_path.startswith(workspace_prefix):
            config_path = config_path.removeprefix(workspace_prefix)
        elif config_path.startswith(project_prefix):
            config_path = config_path.removeprefix(project_prefix)
        resume_command = (
            "python3 -m pmmc experiment run "
            f"--config {config_path} --resume-launch-id {self.launch_id}"
        )
        dataset = None
        suffix = stage.split(":", 1)[1] if ":" in stage else None
        if suffix in {item.value for item in self.config.selection.datasets}:
            dataset = suffix
        round_id = None
        if stage == "doubter:model-stage":
            round_id = max(
                (
                    round_value
                    for value in _read_jsonl(self.launch_dir / "stage_components.jsonl")
                    for round_value in [value.get("round_id")]
                    if value.get("stage") == stage and isinstance(round_value, int)
                ),
                default=None,
            )
        capsule_dir, manifest = build_step6d_runtime_failure_capsule(
            launch_id=self.launch_id,
            stage=stage,
            error_type=type(exc).__name__,
            completed_stage_ids=completed,
            pending_stage_ids=pending,
            latest_milestone=latest_milestone,
            resume_command=resume_command,
            launch_dir=self.launch_dir,
            output_root=self.settings.outputs_path / "compiler-program-pilots",
        )
        payload = {
            "capsule_id": manifest["capsule_id"],
            "capsule_dir": str(capsule_dir),
            "completed_stage_count": len(completed),
            "dataset": dataset,
            "error_code": type(exc).__name__,
            "latest_compatible_milestone": latest_milestone,
            "pending_stage_count": len(pending),
            "resume_command": resume_command,
            "round": round_id,
            "stage": stage,
            "substage": stage.split(":", 1)[0],
        }
        print(
            "STEP6D_FAILURE " + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        metadata = dict(self.state.metadata)
        metadata["step6d_failure"] = payload
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()

    def _stage_component_scopes(self, stage_id: str) -> list[tuple[str | None, str | None]]:
        suffix = stage_id.split(":", 1)[1] if ":" in stage_id else None
        datasets = {item.value for item in self.config.selection.datasets}
        if suffix in datasets:
            values = self.state.dataset_artifacts.get(str(suffix), {})
            run_id = values.get("compiler_run_id")
            return [(str(suffix), str(run_id) if isinstance(run_id, str) else None)]
        scoped: list[tuple[str | None, str | None]] = [
            (dataset, str(values["compiler_run_id"]))
            for dataset, values in sorted(self.state.dataset_artifacts.items())
            if isinstance(values.get("compiler_run_id"), str)
        ]
        return scoped or [(None, None)]

    def _stage_component_artifact_hashes(self) -> list[str]:
        values: list[str] = []
        checkpoint = self.state.metadata.get("latest_checkpoint")
        if isinstance(checkpoint, str):
            manifest = Path(checkpoint) / "manifest.json"
            if manifest.is_file():
                values.append(hashlib.sha256(manifest.read_bytes()).hexdigest())
        return values

    def _record_stage_component(
        self,
        *,
        stage_id: str,
        result: ExperimentStageResult,
        status: StageComponentStatus,
    ) -> None:
        if not _is_verified_qbank_target(self.config.target):
            return
        from pmmc.experiments.stage_components import build_stage_component

        path = self.launch_dir / "stage_components.jsonl"
        prior_values = [StageComponentManifest.model_validate(item) for item in _read_jsonl(path)]
        image = os.environ.get("PMMC_CONTAINER_IMAGE", "").strip() or ("unrecorded-execution-image")
        attempt_id = self._execution_attempt_id or "execution-attempt-unrecorded"
        rounds: tuple[int | None, ...] = (
            (0, 1, 2, 3) if stage_id == "doubter:model-stage" else (None,)
        )
        new_values: list[StageComponentManifest] = []
        for dataset, run_id in self._stage_component_scopes(stage_id):
            scoped_prior = [
                item
                for item in prior_values
                if item.run_id == run_id
                and item.dataset == dataset
                and item.stage != stage_id
                and item.stage in self.state.completed_stage_ids
                and item.status is StageComponentStatus.READY
            ]
            upstream_hash = (
                scoped_prior[-1].output_hash
                if scoped_prior
                else _stable_hash(["stage-component-root", run_id, dataset])
            )
            selection = self.state.dataset_artifacts.get(dataset or "", {})
            for round_id in rounds:
                row_counts = {
                    "work_units": int(result.work_units),
                    "completed": int(status is StageComponentStatus.READY),
                    "failed": int(status is StageComponentStatus.FAILED),
                }
                component = build_stage_component(
                    project_root=self.settings.project_root,
                    launch_id=self.launch_id,
                    run_id=run_id,
                    stage=stage_id,
                    dataset=dataset,
                    round_id=round_id,
                    scientific_input_hash=_stable_hash(
                        {
                            "effective_config_hash": self.effective_config_hash,
                            "run_id": run_id,
                            "selection_hash": selection.get("selection_hash"),
                            "stage": stage_id,
                            "round_id": round_id,
                        }
                    ),
                    upstream_output_hash=upstream_hash,
                    row_counts=row_counts,
                    artifact_hashes=self._stage_component_artifact_hashes(),
                    image_digest=image,
                    status=status,
                    output_facts={
                        "run_id": run_id,
                        "stage": stage_id,
                        "round_id": round_id,
                        "status": status.value,
                        "row_counts": row_counts,
                        "dataset_state": {
                            key: selection.get(key)
                            for key in (
                                "snapshot_id",
                                "compiler_run_id",
                                "coverage_run_id",
                                "semantic_run_id",
                            )
                            if selection.get(key) is not None
                        },
                    },
                    metadata={
                        "error_type": result.error_type,
                        "execution_attempt_id": attempt_id,
                        "reused": result.reused,
                    },
                    alembic_revision=(
                        _STEP7B_ALEMBIC_REVISION
                        if self.config.target in {
                            ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
                            ExperimentTarget.PMMC_ONLINE,
                        }
                        else "20260730_0023"
                        if _is_step7_target(self.config.target)
                        else "20260726_0019"
                    ),
                )
                new_values.append(component)
                if status is StageComponentStatus.READY:
                    upstream_hash = component.output_hash
        all_values = [*prior_values, *new_values]
        _atomic_jsonl(path, [item.model_dump(mode="json") for item in all_values])

        if not self._checkpoint_ready:
            return

        async def persist() -> None:
            from pmmc.infrastructure.database.constants import DatabaseRole
            from pmmc.infrastructure.database.engine import create_session_factory
            from pmmc.infrastructure.database.experiment_repositories import (
                ExperimentUnitOfWork,
            )
            from pmmc.schemas import DataScope

            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                    assert uow.doubter is not None
                    for component in new_values:
                        if component.run_id is not None:
                            await uow.doubter.add_stage_component(component)
            finally:
                await engine.dispose()

        asyncio.run(persist())

    @contextmanager
    def _stage(self, stage_id: str) -> Iterator[ExperimentPlanItem]:
        item = self.plan.get(stage_id)
        if self.reporter is None:
            raise RuntimeError("launcher reporter has not started")
        self.reporter.start_stage(
            stage_id,
            total=item.actual_work_units or item.work_units,
            stage_weight=item.stage_weight,
        )
        started = utc_now()
        performance_started = _performance_snapshot()
        periodic_checkpoint = self._start_periodic_checkpoint(stage_id)
        try:
            yield item
            controller = periodic_checkpoint
            periodic_checkpoint = None
            self._stop_periodic_checkpoint(controller, stage_id=stage_id)
        except Exception as exc:
            controller = periodic_checkpoint
            periodic_checkpoint = None
            if controller is not None:
                try:
                    self._stop_periodic_checkpoint(controller, stage_id=stage_id)
                except Exception as checkpoint_exc:
                    exc.add_note(
                        f"periodic checkpoint shutdown also failed: {type(checkpoint_exc).__name__}"
                    )
            stage_metadata = self._stage_performance_metadata.pop(stage_id, {})
            completed_at = utc_now()
            result = ExperimentStageResult(
                stage_id=stage_id,
                status=ExperimentStageStatus.FAILED,
                work_units=item.actual_work_units or item.work_units,
                started_at=started,
                completed_at=completed_at,
                error_type=type(exc).__name__,
            )
            self._replace_stage_result(result)
            self._record_stage_performance(
                stage_id,
                started,
                completed_at,
                performance_started,
                metadata={
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    **stage_metadata,
                },
            )
            try:
                self._record_stage_component(
                    stage_id=stage_id,
                    result=result,
                    status=StageComponentStatus.FAILED,
                )
            except Exception as component_exc:
                exc.add_note(
                    f"failed stage-component recording also failed: {type(component_exc).__name__}"
                )
                component_errors = _read_jsonl(self.launch_dir / "stage_component_errors.jsonl")
                component_errors.append(
                    {
                        "created_at": utc_now().isoformat(),
                        "error_type": type(component_exc).__name__,
                        "execution_attempt_id": self._execution_attempt_id,
                        "launch_id": self.launch_id,
                        "stage": stage_id,
                    }
                )
                _atomic_jsonl(
                    self.launch_dir / "stage_component_errors.jsonl",
                    component_errors,
                )
            self._save_state()
            if (
                self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
                and stage_id == "doubter:model-stage"
            ):
                # The failed child has exited; preserve committed per-candidate facts
                # before managed services and the ephemeral object store are closed.
                try:
                    self._maybe_checkpoint("doubter:model-stage:failure")
                except Exception as checkpoint_exc:
                    exc.add_note(f"failure checkpoint also failed: {type(checkpoint_exc).__name__}")
            raise
        else:
            stage_metadata = self._stage_performance_metadata.pop(stage_id, {})
            hard_gate_noop = stage_metadata.get("state") == "not_started_by_hard_gate"
            if hard_gate_noop:
                stage_metadata = {
                    **stage_metadata,
                    "milestone_checkpoint": "deferred_no_new_facts",
                }
            self.reporter.finish_stage()
            completed_at = utc_now()
            result = ExperimentStageResult(
                stage_id=stage_id,
                status=ExperimentStageStatus.COMPLETED,
                work_units=item.actual_work_units or item.work_units,
                started_at=started,
                completed_at=completed_at,
            )
            self._replace_stage_result(result)
            self._record_stage_performance(
                stage_id,
                started,
                completed_at,
                performance_started,
                metadata={"status": "completed", **stage_metadata},
            )
            completed = list(dict.fromkeys([*self.state.completed_stage_ids, stage_id]))
            self.state = self.state.model_copy(update={"completed_stage_ids": completed})
            self._save_state()
            self._record_stage_component(
                stage_id=stage_id,
                result=result,
                status=StageComponentStatus.READY,
            )
            if hard_gate_noop:
                latest_checkpoint = self.state.metadata.get("latest_checkpoint")
                latest_sequence = (
                    Path(latest_checkpoint).name
                    if isinstance(latest_checkpoint, str) and latest_checkpoint
                    else "none"
                )
                print(
                    "CHECKPOINT "
                    f"stage={stage_id} state=deferred_no_new_facts "
                    "reason=not_started_by_hard_gate "
                    f"latest_ready_sequence={latest_sequence}",
                    flush=True,
                )
            else:
                self._maybe_checkpoint(stage_id)
                self._maybe_controlled_step7_interrupt(stage_id)
        finally:
            controller = periodic_checkpoint
            periodic_checkpoint = None
            if controller is not None:
                self._stop_periodic_checkpoint(controller, stage_id=stage_id)

    def _replace_stage_result(self, result: ExperimentStageResult) -> None:
        values = [item for item in self.state.stage_results if item.stage_id != result.stage_id]
        values.append(result)
        order = {item.stage_id: index for index, item in enumerate(self.plan.items)}
        values.sort(key=lambda item: order[item.stage_id])
        self.state = self.state.model_copy(update={"stage_results": values})
        _atomic_jsonl(self.launch_dir / "stage_results.jsonl", values)

    def _record_stage_performance(
        self,
        stage_id: str,
        started_at,
        completed_at,
        initial: dict[str, float | int],
        *,
        reused: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        final = _performance_snapshot()
        configured_startup_count = self._managed_model_startup_count(stage_id)
        performance_metadata = dict(metadata or {})
        if self._execution_attempt_id is not None:
            performance_metadata["execution_attempt_id"] = self._execution_attempt_id
        execution_state = "reused" if reused else str(performance_metadata.get("state", "executed"))
        performance_metadata["state"] = execution_state
        model_stage_not_started = execution_state in {
            "not_started_by_hard_gate",
            "deferred_to_shared_model_stage",
            "reused_active_model_service",
            "blocked_by_semantic_parent",
        }
        recorded_startup_count = (
            0 if reused or model_stage_not_started else configured_startup_count
        )
        record = StagePerformanceRecord(
            stage_id=stage_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=max(0.0, (completed_at - started_at).total_seconds()),
            reused=reused,
            cpu_user_seconds=max(
                0.0,
                float(final["cpu_user_seconds"]) - float(initial["cpu_user_seconds"]),
            ),
            cpu_system_seconds=max(
                0.0,
                float(final["cpu_system_seconds"]) - float(initial["cpu_system_seconds"]),
            ),
            peak_rss_bytes=int(final["peak_rss_bytes"]),
            read_bytes=max(0, int(final["read_bytes"]) - int(initial["read_bytes"])),
            write_bytes=max(0, int(final["write_bytes"]) - int(initial["write_bytes"])),
            model_startup_count=recorded_startup_count,
            metadata=performance_metadata,
        )
        path = self.launch_dir / "stage_performance.jsonl"
        values = _read_jsonl(path)
        values.append(record.model_dump(mode="json"))
        _atomic_jsonl(path, values)
        if _stage_starts_managed_model(stage_id):
            print(
                "MODEL_STAGE_ACCOUNTING "
                + json.dumps(
                    {
                        "configured_replica_count": configured_startup_count,
                        "execution_attempt_id": self._execution_attempt_id,
                        "execution_state": execution_state,
                        "recorded_startup_count": record.model_startup_count,
                        "reused": reused,
                        "skip_reason": performance_metadata.get("reason_code"),
                        "stage": stage_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    def _managed_model_startup_count(self, stage_id: str) -> int:
        if not _stage_starts_managed_model(stage_id):
            return 0
        role = {
            "questioner:model-stage": "compiler",
            "qbank_text:model-stage": "embedding_text",
            "qbank_multimodal:model-stage": "embedding_multimodal",
            "compiler_program:model-stage": "compiler",
            "doubter:model-stage": "compiler",
            "program_text:model-stage": "embedding_text",
            "program_multimodal:model-stage": "embedding_multimodal",
            "qbank_semantic:model-stage": "judge",
            "answer:model-stage": "answer",
            "evaluation_judge:model-stage": "judge",
        }.get(stage_id)
        if role is None and stage_id.startswith("retrieval:"):
            method = stage_id.removeprefix("retrieval:").removesuffix(":model-stage")
            role = {
                BaselineMethod.CAPTION_TEXT_RAG.value: "embedding_text",
                BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG.value: ("embedding_contrastive"),
                BaselineMethod.NATIVE_MULTIMODAL_RAG.value: ("embedding_multimodal"),
            }.get(method)
        return self._replica_count(role) if role is not None else 1

    def _set_stage_performance_metadata(self, stage_id: str, metadata: JsonDict) -> None:
        self._stage_performance_metadata[stage_id] = {
            **self._stage_performance_metadata.get(stage_id, {}),
            **metadata,
        }

    def _set_actual_work_units(self, stage_id: str, total: int) -> None:
        """Persist a verified denominator without changing stage weight or identity."""

        if total <= 0:
            raise ValueError("actual stage work units must be positive")
        item = self.plan.get(stage_id)
        item.actual_work_units = total
        _atomic_json(self.launch_dir / "execution_plan.json", self.plan)
        if self.reporter is not None and getattr(self.reporter, "stage_name", stage_id) == stage_id:
            self.reporter.set_stage_total(total)

    def _reuse_stage(self, stage_id: str) -> None:
        item = self.plan.get(stage_id)
        if self.reporter is None:
            raise RuntimeError("launcher reporter has not started")
        started = utc_now()
        performance_started = _performance_snapshot()
        self.reporter.start_stage(
            stage_id,
            total=item.actual_work_units or item.work_units,
            stage_weight=item.stage_weight,
        )
        self.reporter.finish_stage(reused=True)
        self._replace_stage_result(
            ExperimentStageResult(
                stage_id=stage_id,
                status=ExperimentStageStatus.COMPLETED,
                work_units=item.actual_work_units or item.work_units,
                reused=True,
                started_at=started,
                completed_at=utc_now(),
            )
        )
        self._record_stage_performance(
            stage_id,
            started,
            utc_now(),
            performance_started,
            reused=True,
            metadata={"status": "reused"},
        )
        completed = list(dict.fromkeys([*self.state.completed_stage_ids, stage_id]))
        self.state = self.state.model_copy(update={"completed_stage_ids": completed})
        self._save_state()

    def _reuse_restored_stage(self, stage_id: str) -> bool:
        """Reuse a source-compatible stage restored from a self-contained checkpoint."""

        if not (
            self.resume and self._checkpoint_restored and stage_id in self.state.completed_stage_ids
        ):
            return False
        self._reuse_stage(stage_id)
        print(
            "STEP6D_STAGE_REUSE "
            + json.dumps(
                {
                    "source": "compatible_checkpoint",
                    "stage": stage_id,
                    "state": "reused",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return True

    def _finish_stage(self, stage_id: str) -> None:
        with self._stage(stage_id) as item:
            assert self.reporter is not None
            self.reporter.advance(item.actual_work_units or item.work_units)

    def _run_preflight(self) -> None:
        with self._stage("preflight") as item:
            base = _resolve(
                self.settings,
                self.config.model_registry_path or self.settings.resolved_model_registry_path,
            )
            overlays = [
                _resolve(self.settings, path) for path in self.config.model_registry_overlays
            ]
            registry = load_model_registry_with_overlays(base, overlays)
            _assert_model_contracts(registry, self.config)
            if self.execution_config.replicas.maximum > 1 and not self.dry_run:
                from pmmc.runtime.hardware import NvmlHardwareProvider

                visible = len(NvmlHardwareProvider().inventory().devices)
                expected = int(
                    os.environ.get(
                        "PMMC_EXPECTED_GPU_COUNT",
                        str(self.execution_config.replicas.maximum),
                    )
                )
                if visible != expected or visible < self.execution_config.replicas.maximum:
                    raise RuntimeError(
                        "visible GPU count does not satisfy the replica execution profile: "
                        f"visible={visible}, expected={expected}, "
                        f"replicas={self.execution_config.replicas.maximum}"
                    )
            if not (self.dry_run or self.prepare_only or self.evaluate_only):
                for dataset in self.config.selection.datasets:
                    root = (
                        self.settings.memlens_path
                        if dataset is DatasetName.MEMLENS
                        else self.settings.memgallery_path
                    )
                    if not root.is_dir():
                        raise FileNotFoundError(f"benchmark root does not exist: {root}")
            assert self.reporter is not None
            self.reporter.advance(item.work_units)

    def _run_model_prepare(self) -> None:
        if self.resume and "model_prepare" in self.state.completed_stage_ids:
            try:
                result = ModelPreparationResult.from_json(
                    (self.launch_dir / "model_selection.json").read_text(encoding="utf-8")
                )
                registry = load_model_registry_with_overlays(result.merged_registry_path, [])
                lock = load_model_lock(result.lock_path)
                assert_paper_lock(lock, registry, aliases=result.required_aliases)
                if result.verified_cache_manifest_path is not None:
                    prepared = load_verified_model_cache_manifest(
                        result.verified_cache_manifest_path,
                        lock_path=result.lock_path,
                        lock=lock,
                        aliases=result.required_aliases,
                        verification_policy=result.verification_policy,
                    )
                    if set(prepared.paths) != set(result.required_aliases):
                        raise ValueError("resumed model preparation aliases differ")
                else:
                    verify_model_cache(
                        lock,
                        self.settings.model_cache_path,
                        aliases=result.required_aliases,
                        verification_policy=result.verification_policy,
                        allow_full_fallback=False,
                    )
                self.preparation = result
                self.resolved_registry = load_model_registry_with_overlays(
                    result.resolved_registry_path, []
                )
                self._reuse_stage("model_prepare")
                return
            except (FileNotFoundError, KeyError, ValueError):
                pass
        with self._stage("model_prepare"):
            service = ModelPreparationService(
                self.settings,
                self.config,
                self.launch_dir,
                self.reporter,
            )
            self.preparation, self.resolved_registry = service.prepare()

    def _resolve_runtime_and_baseline_configs(self) -> None:
        if _is_questioner_target(self.config.target):
            self._resolve_questioner_runtime_config()
            return
        from pmmc.baselines.config import load_full_context_config
        from pmmc.baselines.rag_config import load_rag_config
        from pmmc.evaluation.config import load_evaluation_config
        from pmmc.runtime.config import (
            load_runtime_profile,
            project_runtime_profile_roles,
            runtime_profile_safe_summary,
            write_runtime_profile,
        )

        if self.preparation is None:
            raise RuntimeError("model preparation has not completed")
        resolved_dir = self.launch_dir / "resolved"
        max_input = self.config.total_context_tokens - self.config.reserved_output_tokens
        full = load_full_context_config(self.settings.resolved_full_context_config_path)
        full = full.model_copy(
            update={
                "answer_model_alias": self.config.models.answer_model,
                "total_context_tokens": self.config.total_context_tokens,
                "reserved_output_tokens": self.config.reserved_output_tokens,
                "max_input_tokens": max_input,
                "max_output_tokens": self.config.reserved_output_tokens,
                "concurrency": self.config.answer_batch_size,
            }
        )
        rag = load_rag_config(self.settings.resolved_rag_config_path)
        rag = rag.model_copy(
            update={
                "answer_model_alias": self.config.models.answer_model,
                "caption_text_embedding_alias": self.config.models.text_embedding_model,
                "contrastive_embedding_alias": self.config.models.contrastive_embedding_model,
                "native_multimodal_embedding_alias": (
                    self.config.models.multimodal_embedding_model
                ),
                "retrieval_top_k": self.config.retrieval_top_k,
                "answer_top_k": self.config.answer_top_k,
                "total_context_tokens": self.config.total_context_tokens,
                "reserved_output_tokens": self.config.reserved_output_tokens,
                "max_input_tokens": max_input,
                "max_output_tokens": self.config.reserved_output_tokens,
                "caption_text_batch_size": self.config.embedding_batch_size,
                "contrastive_batch_size": self.config.embedding_batch_size,
                "native_multimodal_batch_size": (self.config.multimodal_embedding_batch_size),
                "embedding_concurrency": self.config.model_concurrency,
                "answer_concurrency": self.config.answer_batch_size,
            }
        )
        evaluation = load_evaluation_config(self.settings.resolved_evaluation_config_path)
        evaluation = evaluation.model_copy(
            update={
                "judge_model_alias": self.config.models.judge_model,
                "concurrency": self.config.model_concurrency,
            }
        )
        full_path = resolved_dir / "full-context.yaml"
        rag_path = resolved_dir / "rag.yaml"
        evaluation_path = resolved_dir / "evaluation.yaml"
        _write_yaml(full_path, full)
        _write_yaml(rag_path, rag)
        _write_yaml(evaluation_path, evaluation)

        profile = load_runtime_profile(_resolve(self.settings, self.config.runtime_profile_path))
        aliases_by_role = {ModelRole.ANSWER: self.config.models.answer_model}
        stages_by_role = {ModelRole.ANSWER: "answer"}
        if self.config.evaluate:
            aliases_by_role[ModelRole.JUDGE] = self.config.models.judge_model
            stages_by_role[ModelRole.JUDGE] = "judge"
        if BaselineMethod.CAPTION_TEXT_RAG in self.config.methods:
            aliases_by_role[ModelRole.TEXT_EMBEDDING] = self.config.models.text_embedding_model
            stages_by_role[ModelRole.TEXT_EMBEDDING] = "embedding_text"
        if BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG in self.config.methods:
            aliases_by_role[ModelRole.CONTRASTIVE_EMBEDDING] = (
                self.config.models.contrastive_embedding_model
            )
            stages_by_role[ModelRole.CONTRASTIVE_EMBEDDING] = "embedding_contrastive"
        if BaselineMethod.NATIVE_MULTIMODAL_RAG in self.config.methods:
            aliases_by_role[ModelRole.MULTIMODAL_EMBEDDING] = (
                self.config.models.multimodal_embedding_model
            )
            stages_by_role[ModelRole.MULTIMODAL_EMBEDDING] = "embedding_multimodal"
        profile = project_runtime_profile_roles(
            profile,
            aliases_by_role=aliases_by_role,
            stages_by_role=stages_by_role,
            model_lock_path=self.preparation.lock_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        print(
            "RUNTIME_PROFILE_PROJECTION "
            + json.dumps(
                {
                    **runtime_profile_safe_summary(profile),
                    "selected_roles": sorted(role.value for role in aliases_by_role),
                    "target": self.config.target.value,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        profile_path = resolved_dir / "runtime-profile.yaml"
        write_runtime_profile(profile, profile_path)
        self.launch_settings = self.settings.model_copy(
            update={
                # The runtime planner validates the lock against the source registry.
                # Managed child commands receive the commit-resolved registry below.
                "model_registry_path": self.preparation.merged_registry_path,
                "runtime_model_lock_path": self.preparation.lock_path,
                "runtime_profile_path": profile_path,
                "full_context_config_path": full_path,
                "rag_config_path": rag_path,
                "evaluation_config_path": evaluation_path,
                "runtime_require_calibration_paper": (
                    self.config.calibration_policy is not CalibrationPolicy.SKIP
                ),
            }
        )

    def _resolve_questioner_runtime_config(self) -> None:
        from pmmc.compiler.questioner_config import load_questioner_config
        from pmmc.runtime.config import (
            load_runtime_profile,
            project_runtime_profile_roles,
            runtime_profile_safe_summary,
            write_runtime_profile,
        )

        if self.preparation is None or self.config.models.compiler_model is None:
            raise RuntimeError("Questioner model preparation has not completed")
        resolved_dir = self.launch_dir / "resolved"
        questioner = load_questioner_config(
            _questioner_config_path(self.settings, self.config)
        ).model_copy(
            update={
                "compiler_model_alias": self.config.models.compiler_model,
                "text_embedding_model_alias": self.config.models.text_embedding_model,
                "multimodal_embedding_model_alias": (self.config.models.multimodal_embedding_model),
                "total_context_tokens": self.config.total_context_tokens,
                "reserved_output_tokens": self.config.reserved_output_tokens,
                "text_embedding_batch_size": self.config.embedding_batch_size,
                "multimodal_embedding_batch_size": (self.config.multimodal_embedding_batch_size),
                "embedding_concurrency": self.config.model_concurrency,
            }
        )
        questioner_path = resolved_dir / "questioner.yaml"
        _write_yaml(questioner_path, questioner)

        profile = load_runtime_profile(_resolve(self.settings, self.config.runtime_profile_path))
        aliases_by_role = {
            ModelRole.COMPILER: self.config.models.compiler_model,
            ModelRole.TEXT_EMBEDDING: self.config.models.text_embedding_model,
            ModelRole.MULTIMODAL_EMBEDDING: (self.config.models.multimodal_embedding_model),
        }
        stages_by_role = {
            ModelRole.COMPILER: "compiler",
            ModelRole.TEXT_EMBEDDING: "embedding_text",
            ModelRole.MULTIMODAL_EMBEDDING: "embedding_multimodal",
        }
        if _is_semantic_program_target(self.config.target):
            aliases_by_role[ModelRole.JUDGE] = self.config.models.judge_model
            stages_by_role[ModelRole.JUDGE] = "judge"
        if _is_step7_target(self.config.target):
            aliases_by_role[ModelRole.ANSWER] = self.config.models.answer_model
            stages_by_role[ModelRole.ANSWER] = "answer"
        if self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
            if self.config.models.reranker_model is None:
                raise RuntimeError("Step 7B runtime resolution requires reranker_model")
            aliases_by_role[ModelRole.RERANKER] = self.config.models.reranker_model
            # Reranking runs inside the post-index shared compiler session.
            stages_by_role[ModelRole.RERANKER] = "compiler"
        profile = project_runtime_profile_roles(
            profile,
            aliases_by_role=aliases_by_role,
            stages_by_role=stages_by_role,
            model_lock_path=self.preparation.lock_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        print(
            "RUNTIME_PROFILE_PROJECTION "
            + json.dumps(
                {
                    **runtime_profile_safe_summary(profile),
                    "selected_roles": sorted(role.value for role in aliases_by_role),
                    "target": self.config.target.value,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        profile_path = resolved_dir / "runtime-profile.yaml"
        write_runtime_profile(profile, profile_path)
        self.launch_settings = self.settings.model_copy(
            update={
                "model_registry_path": self.preparation.merged_registry_path,
                "runtime_model_lock_path": self.preparation.lock_path,
                "runtime_profile_path": profile_path,
                "questioner_config_path": questioner_path,
                "compiler_config_path": (
                    _resolve(self.settings, self.config.compiler_foundation_config_path)
                    if self.config.compiler_foundation_config_path is not None
                    else self.settings.resolved_compiler_config_path
                ),
                "token_counter_backend": (self.execution_config.unit_preparation.counter_backend),
                "local_asset_cache_enabled": (
                    self.execution_config.unit_preparation.local_asset_cache
                ),
                # Coverage scoring runs in the launcher process after managed
                # embedding services have stopped. Keep its execution settings
                # identical to those injected into managed child commands.
                "qbank_text_query_batch_size": (self.execution_config.qbank.text_query_batch_size),
                "qbank_multimodal_query_batch_size": (
                    self.execution_config.qbank.multimodal_query_batch_size
                ),
                "qbank_embedding_concurrency": (self.execution_config.qbank.embedding_concurrency),
                "qbank_score_concurrency": (self.execution_config.qbank.score_concurrency),
                "runtime_require_calibration_paper": (
                    self.config.calibration_policy is not CalibrationPolicy.SKIP
                ),
            }
        )

    def _child_environment(self) -> dict[str, str]:
        if self.preparation is None:
            raise RuntimeError("model preparation has not completed")
        values = {
            "PMMC_MODE": self.config.mode.value,
            "PMMC_MODEL_REGISTRY_PATH": str(self.preparation.resolved_registry_path),
            "PMMC_RUNTIME_MODEL_LOCK_PATH": str(self.preparation.lock_path),
            "PMMC_RUNTIME_PROFILE_PATH": str(self.launch_settings.resolved_runtime_profile_path),
            "PMMC_FULL_CONTEXT_CONFIG_PATH": str(
                self.launch_settings.resolved_full_context_config_path
            ),
            "PMMC_RAG_CONFIG_PATH": str(self.launch_settings.resolved_rag_config_path),
            "PMMC_EVALUATION_CONFIG_PATH": str(
                self.launch_settings.resolved_evaluation_config_path
            ),
            "HF_HOME": str(self.launch_settings.model_cache_path),
            "PMMC_MODEL_CACHE_DIR": str(self.launch_settings.model_cache_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PMMC_LAUNCH_ID": self.launch_id,
            "PMMC_MODEL_MAX_CONCURRENCY": str(self.config.model_concurrency),
            "PMMC_MODEL_VERIFICATION_POLICY": (self.preparation.verification_policy.value),
            "PMMC_PAPER_RESULT": str(bool(self.config.metadata.get("paper_result"))).lower(),
            "PMMC_TOKEN_COUNTER_BACKEND": (self.execution_config.unit_preparation.counter_backend),
            "PMMC_TOKEN_COUNTER_CACHE_DIR": str(self.launch_settings.token_counter_cache_path),
            "PMMC_LOCAL_ASSET_CACHE_ENABLED": str(
                self.execution_config.unit_preparation.local_asset_cache
            ).lower(),
            "PMMC_QBANK_TEXT_QUERY_BATCH_SIZE": str(
                self.execution_config.qbank.text_query_batch_size
            ),
            "PMMC_QBANK_MULTIMODAL_QUERY_BATCH_SIZE": str(
                self.execution_config.qbank.multimodal_query_batch_size
            ),
            "PMMC_QBANK_EMBEDDING_CONCURRENCY": str(
                self.execution_config.qbank.embedding_concurrency
            ),
            "PMMC_QBANK_SCORE_CONCURRENCY": str(self.execution_config.qbank.score_concurrency),
            "PMMC_QUESTIONER_WORKER_LIMIT": str(
                min(
                    self.execution_config.unit_preparation.workers,
                    self.config.model_concurrency,
                )
            ),
            "PMMC_COMPACT_CHILD_SUMMARY": "1",
        }
        if self.preparation.verified_cache_manifest_path is not None:
            values["PMMC_VERIFIED_MODEL_CACHE_MANIFEST"] = str(
                self.preparation.verified_cache_manifest_path
            )
        if _is_questioner_target(self.config.target):
            values["PMMC_QUESTIONER_CONFIG_PATH"] = str(
                self.launch_settings.resolved_questioner_config_path
            )
            values["PMMC_COMPILER_CONFIG_PATH"] = str(
                self.launch_settings.resolved_compiler_config_path
            )
        return values

    def _run_infrastructure(self) -> None:
        with self._stage("infrastructure") as item:
            self._wait_for_infrastructure()
            if self.config.infrastructure.migrate_database:
                self._upgrade_database()
            if (
                self.resume or self.import_upstream_from_launch
            ) and _is_verified_qbank_target(self.config.target):
                # Catalog-only checks must fail before restoring large private bundles.
                self._verify_infrastructure(phase="before_restore")
            self._restore_checkpoint_if_needed()
            self._verify_infrastructure()
            self._checkpoint_ready = True
            assert self.reporter is not None
            self.reporter.advance(item.work_units)

    def _wait_for_infrastructure(self) -> None:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.health import check_database_connectivity
        from pmmc.infrastructure.object_store import create_admin_object_stores

        if self.config.infrastructure.mode is InfrastructureMode.SELF_CONTAINED:
            self._assert_loopback_infrastructure()
        started = time.monotonic()
        deadline = started + self.config.infrastructure.wait_timeout_seconds
        last_error: Exception | None = None
        last_check = "not-started"
        last_reported_at = started - 30.0
        attempt_count = 0
        print(
            "INFRASTRUCTURE_WAIT "
            f"state=start mode={self.config.infrastructure.mode.value} "
            f"timeout_seconds={self.config.infrastructure.wait_timeout_seconds}",
            flush=True,
        )
        while time.monotonic() < deadline:
            attempt_count += 1
            try:
                missing_markers = self._missing_infrastructure_markers()
                if missing_markers:
                    last_check = "sidecar-markers:" + ",".join(missing_markers)
                    now = time.monotonic()
                    if attempt_count == 1 or now - last_reported_at >= 30.0:
                        print(
                            "INFRASTRUCTURE_WAIT "
                            f"state=pending attempts={attempt_count} "
                            f"elapsed_seconds={now - started:.1f} "
                            f"last_check={last_check}",
                            flush=True,
                        )
                        last_reported_at = now
                    time.sleep(2)
                    continue

                async def check() -> None:
                    engine = create_engine(self.launch_settings, DatabaseRole.ADMIN)
                    try:
                        await check_database_connectivity(engine, DatabaseRole.ADMIN)
                    finally:
                        await engine.dispose()
                    runtime_store, evaluation_store = create_admin_object_stores(
                        self.launch_settings
                    )
                    await runtime_store.check()
                    await evaluation_store.check()

                asyncio.run(check())
                print(
                    "INFRASTRUCTURE_WAIT "
                    f"state=ready attempts={attempt_count} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
                return
            except Exception as exc:
                last_error = exc
                last_check = type(exc).__name__
                now = time.monotonic()
                if attempt_count == 1 or now - last_reported_at >= 30.0:
                    print(
                        "INFRASTRUCTURE_WAIT "
                        f"state=pending attempts={attempt_count} "
                        f"elapsed_seconds={now - started:.1f} "
                        f"last_check={last_check}",
                        flush=True,
                    )
                    last_reported_at = now
                time.sleep(2)
        print(
            "INFRASTRUCTURE_WAIT "
            f"state=timeout attempts={attempt_count} "
            f"elapsed_seconds={time.monotonic() - started:.1f} "
            f"last_check={last_check}",
            flush=True,
        )
        raise TimeoutError(
            f"Postgres/MinIO did not become ready; last_check={last_check}"
        ) from last_error

    def _missing_infrastructure_markers(self) -> list[str]:
        if self.config.infrastructure.mode is not InfrastructureMode.SELF_CONTAINED:
            return []
        stop_file = os.environ.get("PMMC_LAUNCH_STOP_FILE")
        if not stop_file:
            return []
        control_dir = Path(stop_file).parent
        return [
            name for name in ("postgres-ready", "minio-ready") if not (control_dir / name).is_file()
        ]

    def _verify_infrastructure(self, *, phase: str = "after_restore") -> None:
        from sqlalchemy import text

        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.health import check_database
        from pmmc.infrastructure.object_store import create_admin_object_stores

        async def check() -> None:
            engine = create_engine(self.launch_settings, DatabaseRole.ADMIN)
            try:
                await check_database(engine, DatabaseRole.ADMIN)
                if _is_verified_qbank_target(self.config.target):
                    from pmmc.compiler.questioner_config import load_questioner_config
                    from pmmc.infrastructure.database.constants import (
                        CANDIDATE_REJECTION_CODE_VALUES,
                    )

                    questioner = load_questioner_config(
                        self.launch_settings.resolved_questioner_config_path
                    )
                    async with engine.connect() as connection:
                        alembic_revision = await connection.scalar(
                            text("SELECT version_num FROM alembic_version")
                        )
                        function_definition = await connection.scalar(
                            text(
                                "SELECT pg_get_functiondef(to_regprocedure("
                                "'pmmc_runtime.enforce_questioner_verifier_scope_v3()'"
                                "))"
                            )
                        )
                        generation_function_definition = await connection.scalar(
                            text(
                                "SELECT pg_get_functiondef(to_regprocedure("
                                "'pmmc_runtime.validate_questioner_fact_scope()'"
                                "))"
                            )
                        )
                        embedding_function_definition = await connection.scalar(
                            text(
                                "SELECT pg_get_functiondef(to_regprocedure("
                                "'pmmc_runtime.validate_compiler_embedding_lineage_v4()'"
                                "))"
                            )
                        )
                        completion_function_definition = await connection.scalar(
                            text(
                                "SELECT pg_get_functiondef(to_regprocedure("
                                "'pmmc_runtime.enforce_compiler_completion_v3()'"
                                "))"
                            )
                        )
                        gate_constraint_rows = await connection.execute(
                            text(
                                "SELECT pg_get_constraintdef(constraint_row.oid, true) "
                                "FROM pg_constraint constraint_row "
                                "JOIN pg_class relation_row "
                                "ON relation_row.oid = constraint_row.conrelid "
                                "JOIN pg_namespace namespace_row "
                                "ON namespace_row.oid = relation_row.relnamespace "
                                "WHERE namespace_row.nspname = 'pmmc_runtime' "
                                "AND relation_row.relname = "
                                "'questioner_candidate_gate_results' "
                                "AND constraint_row.contype = 'c' "
                                "ORDER BY constraint_row.conname"
                            )
                        )
                        gate_constraint_definitions = [
                            str(row[0] or "")
                            for row in gate_constraint_rows
                            if "rejection_codes" in str(row[0] or "")
                            and "<@" in str(row[0] or "")
                        ]
                    definition = function_definition if isinstance(function_definition, str) else ""
                    generation_definition = (
                        generation_function_definition
                        if isinstance(generation_function_definition, str)
                        else ""
                    )
                    embedding_definition = (
                        embedding_function_definition
                        if isinstance(embedding_function_definition, str)
                        else ""
                    )
                    completion_definition = (
                        completion_function_definition
                        if isinstance(completion_function_definition, str)
                        else ""
                    )
                    step7b = self.config.target in {
                        ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
                        ExperimentTarget.PMMC_ONLINE,
                    }
                    expected_revision = _STEP7B_ALEMBIC_REVISION if step7b else "20260730_0023"
                    if step7b:
                        from pmmc.infrastructure.database.completion_preflight import (
                            verify_step7b_completion_sql,
                        )

                        async with engine.connect() as connection:
                            await verify_step7b_completion_sql(connection)
                    expected_embedding_schema_clause = "compiler.compiler_schema_version IN " + (
                        "('0.6.0', '0.7.0', '0.8.0')" if step7b else "('0.6.0', '0.7.0')"
                    )
                    expected_completion_clause = "compiler.compiler_schema_version NOT IN " + (
                        "('0.4.0', '0.5.0', '0.6.0', '0.7.0', '0.8.0')"
                        if step7b
                        else "('0.4.0', '0.5.0', '0.6.0', '0.7.0')"
                    )
                    compiler_schema_version = questioner.compiler_schema_version
                    gate_definition = (
                        gate_constraint_definitions[0]
                        if len(gate_constraint_definitions) == 1
                        else "\n".join(gate_constraint_definitions)
                    )
                    missing_gate_codes = sorted(
                        code
                        for code in CANDIDATE_REJECTION_CODE_VALUES
                        if code not in gate_definition
                    )
                    checks = {
                        "alembic_revision": (str(alembic_revision or "") == expected_revision),
                        "compiler_schema": (f"WHEN '{compiler_schema_version}'" in definition),
                        "prompt_version": questioner.verifier_prompt_version in definition,
                        "single_candidate_input": "NEW.input_count" in definition,
                        "single_candidate_scope": "NEW.scope_qbank_id" in definition,
                        "failed_call_terminal_rejection": (
                            "NEW.reject_count = 1" in definition
                            and "NEW.pass_count = 0" in definition
                            and "NEW.revise_count = 0" in definition
                            and "provider_output_fallback" in definition
                        ),
                        "safe_constraint_diagnostics": (
                            "ck_qv_batches_failed_call_terminal_rejection" in definition
                        ),
                        "generation_terminal_rejection": (
                            compiler_schema_version in generation_definition
                            and "NEW.parse_status = 'failed'" in generation_definition
                            and "NEW.raw_candidate_count = 0" in generation_definition
                            and "generation_structured_output_invalid" in generation_definition
                        ),
                        "generation_terminal_artifact_ready": (
                            "artifact_state IS DISTINCT FROM 'ready'" in generation_definition
                            and "ck_questioner_generations_terminal_artifact_ready"
                            in generation_definition
                        ),
                        "generation_safe_constraint_diagnostics": (
                            "ck_questioner_generations_failed_call_terminal_rejection"
                            in generation_definition
                        ),
                        "program_query_embedding_lineage": (
                            expected_embedding_schema_clause in embedding_definition
                            and "compiler.target_stage = 'verified_qbank'" in embedding_definition
                            and "question_text_query" in embedding_definition
                            and "question_joint_query" in embedding_definition
                            and "ck_embeddings_compiler_source_lineage" in embedding_definition
                        ),
                        "verified_qbank_stage_accounting_delegated": (
                            expected_completion_clause in completion_definition
                        ),
                        "candidate_gate_rejection_constraint_unique": (
                            not step7b or len(gate_constraint_definitions) == 1
                        ),
                        "candidate_gate_rejection_codes": (
                            not step7b or not missing_gate_codes
                        ),
                    }
                    failed_checks = sorted(name for name, passed in checks.items() if not passed)
                    event = {
                        "alembic_revision": str(alembic_revision or "missing"),
                        "expected_alembic_revision": expected_revision,
                        "phase": phase,
                        "checks": checks,
                        "compiler_schema_version": compiler_schema_version,
                        "definition_sha256": {
                            "embedding_lineage": hashlib.sha256(
                                embedding_definition.encode("utf-8")
                            ).hexdigest(),
                            "compiler_completion": hashlib.sha256(
                                completion_definition.encode("utf-8")
                            ).hexdigest(),
                            "generation_scope": hashlib.sha256(
                                generation_definition.encode("utf-8")
                            ).hexdigest(),
                            "verifier_scope": hashlib.sha256(
                                definition.encode("utf-8")
                            ).hexdigest(),
                            "candidate_gate_rejection_codes": hashlib.sha256(
                                gate_definition.encode("utf-8")
                            ).hexdigest(),
                        },
                        "failed_checks": failed_checks,
                        "candidate_gate_rejection_code_count": len(
                            CANDIDATE_REJECTION_CODE_VALUES
                        ),
                        "candidate_gate_rejection_constraint_count": len(
                            gate_constraint_definitions
                        ),
                        "candidate_gate_missing_code_count": len(missing_gate_codes),
                        "prompt_version": questioner.verifier_prompt_version,
                        "state": "ready" if all(checks.values()) else "failed",
                    }
                    print(
                        "QUESTIONER_PERSISTENCE_PREFLIGHT "
                        + json.dumps(event, sort_keys=True, separators=(",", ":")),
                        flush=True,
                    )
                    if not all(checks.values()):
                        raise QuestionerPersistencePreflightError(failed_checks)
            finally:
                await engine.dispose()
            runtime_store, evaluation_store = create_admin_object_stores(self.launch_settings)
            await runtime_store.check()
            await evaluation_store.check()

        asyncio.run(check())

    def _assert_loopback_infrastructure(self) -> None:
        values = {
            "database_admin_url": (
                self.launch_settings.database_admin_url.get_secret_value()
                if self.launch_settings.database_admin_url
                else ""
            ),
            "database_url": (
                self.launch_settings.database_url.get_secret_value()
                if self.launch_settings.database_url
                else ""
            ),
            "database_evaluator_url": (
                self.launch_settings.database_evaluator_url.get_secret_value()
                if self.launch_settings.database_evaluator_url
                else ""
            ),
            "object_store_endpoint": self.launch_settings.object_store_endpoint or "",
        }
        for name, value in values.items():
            host = urlsplit(value).hostname
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError(f"self-contained infrastructure requires a loopback {name}")

    def _upgrade_database(self) -> None:
        import alembic.command as command
        from alembic.config import Config

        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import database_url_for_role

        configuration = Config(str(self.settings.project_root / "alembic.ini"))
        configuration.attributes["database_url"] = database_url_for_role(
            self.launch_settings, DatabaseRole.ADMIN
        )
        command.upgrade(configuration, "head")

    def _checkpoint_identity(self) -> dict[str, object]:
        model_lock_sha256 = None
        if self.preparation is not None and self.preparation.lock_path.is_file():
            model_lock_sha256 = _file_sha256(self.preparation.lock_path)
        snapshots = {
            dataset: value.get("snapshot_id")
            for dataset, value in sorted(self.state.dataset_artifacts.items())
            if value.get("snapshot_id") is not None
        }
        return {
            "launcher_schema_version": self.config.launcher_schema_version,
            "launch_config_hash": self.effective_config_hash,
            "execution_profile_hash": self.execution_config.config_hash,
            "source_inputs_hash": _stable_hash(_source_identity(self.source_inputs)),
            "model_lock_sha256": model_lock_sha256,
            "snapshots": snapshots,
        }

    def _checkpoint_service(self):  # type: ignore[no-untyped-def]
        from pmmc.experiments.checkpoint import ExperimentCheckpointService

        return ExperimentCheckpointService(
            self.launch_settings,
            launch_id=self.launch_id,
            root=self.settings.runs_path / "private" / "checkpoints",
            identity=self._checkpoint_identity(),
            retain_count=self.execution_config.checkpoint.retain_count,
            retain_milestones=_is_verified_qbank_target(self.config.target),
        )

    def _pre_checkpoint_resume_is_safe(self) -> bool:
        """Allow cold replay only before any durable scientific boundary."""

        return bool(self._pre_checkpoint_resume_diagnostics()["safe"])

    def _pre_checkpoint_resume_diagnostics(self) -> dict[str, object]:
        completed = set(self.state.completed_stage_ids)
        replayable = {
            stage_id
            for stage_id in completed
            if _is_pre_checkpoint_replayable_stage(stage_id)
        }
        non_replayable = completed - replayable
        has_latest_checkpoint = bool(self.state.metadata.get("latest_checkpoint"))
        has_restored_checkpoint = bool(self.state.metadata.get("restored_checkpoint"))
        dataset_artifact_count = len(self.state.dataset_artifacts)
        safe = (
            self.resume
            and not non_replayable
            and dataset_artifact_count == 0
            and not has_latest_checkpoint
            and not has_restored_checkpoint
        )
        return {
            "safe": safe,
            "resume_requested": self.resume,
            "completed_stage_count": len(completed),
            "replayable_stage_count": len(replayable),
            "non_replayable_stage_count": len(non_replayable),
            "dataset_artifact_count": dataset_artifact_count,
            "has_latest_checkpoint": has_latest_checkpoint,
            "has_restored_checkpoint": has_restored_checkpoint,
        }

    def _restore_checkpoint_if_needed(self) -> None:
        source_id = self.import_upstream_from_launch or self.state.metadata.get(
            "upstream_import_source"
        )
        if isinstance(source_id, str) and not self._checkpoint_service().directories():
            from pmmc.experiments.upstream_import import restore_upstream

            restore_upstream(self, source_id)
            return
        if (
            not self.resume
            or self.config.infrastructure.mode is not InfrastructureMode.SELF_CONTAINED
            or self.execution_config.checkpoint.policy is CheckpointPolicy.DISABLED
        ):
            return
        service = self._checkpoint_service()
        selected: Path | None = None
        selected_manifest = None
        selected_projection_rebuilds: set[str] = set()
        checkpoint_candidates = service.directories()
        scan_started = time.monotonic()
        scan_counts = {
            "checked": 0,
            "metadata_invalid": 0,
            "schema_incompatible": 0,
            "source_incompatible": 0,
        }
        print(
            "CHECKPOINT_RESTORE "
            f"launch_id={self.launch_id} state=scan "
            f"candidate_count={len(checkpoint_candidates)}",
            flush=True,
        )
        if _is_verified_qbank_target(self.config.target):
            from pmmc.experiments.checkpoint import (
                checkpoint_revision_is_restore_compatible,
            )
            from pmmc.experiments.stage_components import (
                checkpoint_projection_rebuild_allowed,
                source_fingerprint,
                source_fingerprints_compatible,
                tracks_scientific_compatibility,
            )

            database_revision = asyncio.run(service.current_database_revision())
            current_fingerprints: dict[str, str] = {}
            reported_compatible_stages: set[str] = set()

            def current_fingerprint(stage_id: str) -> str:
                value = current_fingerprints.get(stage_id)
                if value is None:
                    value = source_fingerprint(self.settings.project_root, stage_id)
                    current_fingerprints[stage_id] = value
                return value

            def report_scan(state: str) -> None:
                print(
                    "CHECKPOINT_RESTORE_SCAN "
                    f"state={state} "
                    f"checked={scan_counts['checked']} "
                    f"remaining={max(0, len(checkpoint_candidates) - scan_counts['checked'])} "
                    f"metadata_invalid={scan_counts['metadata_invalid']} "
                    f"schema_incompatible={scan_counts['schema_incompatible']} "
                    f"source_incompatible={scan_counts['source_incompatible']} "
                    f"elapsed_seconds={time.monotonic() - scan_started:.1f}",
                    flush=True,
                )

            component_path = self.launch_dir / "stage_components.jsonl"
            components = [
                StageComponentManifest.model_validate(item) for item in _read_jsonl(component_path)
            ]
            components_by_stage: dict[str, list[StageComponentManifest]] = {}
            latest_attempt_by_stage: dict[str, StageComponentManifest] = {}
            for component in components:
                previous_attempt = latest_attempt_by_stage.get(component.stage)
                if previous_attempt is None or previous_attempt.created_at < component.created_at:
                    latest_attempt_by_stage[component.stage] = component
                if component.status is StageComponentStatus.READY:
                    components_by_stage.setdefault(component.stage, []).append(component)
            for candidate in reversed(checkpoint_candidates):
                scan_counts["checked"] += 1
                try:
                    checkpoint_manifest = service.inspect_candidate(
                        candidate,
                        require_identity=False,
                    )
                except (FileNotFoundError, ValueError):
                    scan_counts["metadata_invalid"] += 1
                    if scan_counts["checked"] % 10 == 0:
                        report_scan("progress")
                    continue
                if not checkpoint_revision_is_restore_compatible(
                    checkpoint_manifest.alembic_revision,
                    database_revision,
                ):
                    scan_counts["schema_incompatible"] += 1
                    if scan_counts["checked"] % 10 == 0:
                        report_scan("progress")
                    continue
                if checkpoint_manifest.milestone.startswith("periodic:"):
                    periodic_stage = checkpoint_manifest.milestone.removeprefix("periodic:")
                    attempted = latest_attempt_by_stage.get(periodic_stage)
                    if attempted is not None and not source_fingerprints_compatible(
                        periodic_stage,
                        attempted.source_fingerprint,
                        current_fingerprint(periodic_stage),
                    ):
                        scan_counts["source_incompatible"] += 1
                        print(
                            "CHECKPOINT_RESTORE "
                            f"launch_id={self.launch_id} state=skipped "
                            f"sequence={candidate.name} "
                            "reason=periodic_stage_source_changed "
                            f"stage={periodic_stage}",
                            flush=True,
                        )
                        if scan_counts["checked"] % 10 == 0:
                            report_scan("progress")
                        continue
                state_path = candidate / "launch_state.json"
                if not state_path.is_file():
                    scan_counts["metadata_invalid"] += 1
                    continue
                try:
                    checkpoint_state = ExperimentLaunchState.from_json(
                        state_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    scan_counts["metadata_invalid"] += 1
                    continue
                if not _step7_prefix_partial_checkpoint_source_compatible(
                    checkpoint_manifest.milestone,
                    checkpoint_state.metadata,
                    current_fingerprint("compiler_program"),
                ):
                    scan_counts["source_incompatible"] += 1
                    print(
                        "CHECKPOINT_RESTORE "
                        f"launch_id={self.launch_id} state=skipped "
                        f"sequence={candidate.name} "
                        "reason=partial_qp_source_changed stage=compiler_program",
                        flush=True,
                    )
                    if scan_counts["checked"] % 10 == 0:
                        report_scan("progress")
                    continue
                compatible = True
                projection_rebuilds: set[str] = set()
                for stage_id in checkpoint_state.completed_stage_ids:
                    if not tracks_scientific_compatibility(stage_id):
                        continue
                    eligible_values = [
                        item
                        for item in components_by_stage.get(stage_id, [])
                        if item.created_at.isoformat() <= checkpoint_manifest.created_at
                    ]
                    latest_by_scope: dict[
                        tuple[str | None, int | None], StageComponentManifest
                    ] = {}
                    for item in eligible_values:
                        key = (item.dataset, item.round_id)
                        previous = latest_by_scope.get(key)
                        if previous is None or previous.created_at < item.created_at:
                            latest_by_scope[key] = item
                    values = list(latest_by_scope.values())
                    if not values:
                        compatible = False
                        break
                    current = current_fingerprint(stage_id)
                    if any(
                        not source_fingerprints_compatible(
                            stage_id,
                            item.source_fingerprint,
                            current,
                        )
                        for item in values
                    ):
                        if checkpoint_projection_rebuild_allowed(stage_id):
                            projection_rebuilds.add(stage_id)
                            continue
                        compatible = False
                        break
                    if stage_id not in reported_compatible_stages and any(
                        item.source_fingerprint != current for item in values
                    ):
                        print(
                            "CHECKPOINT_SOURCE_COMPATIBILITY "
                            f"stage={stage_id} mode=audited_non_scientific_transition",
                            flush=True,
                        )
                        reported_compatible_stages.add(stage_id)
                if compatible:
                    verification_started = time.monotonic()
                    print(
                        "CHECKPOINT_RESTORE_VERIFY "
                        f"state=start sequence={candidate.name} "
                        f"dump_size_bytes={checkpoint_manifest.postgres_dump_size_bytes}",
                        flush=True,
                    )
                    try:
                        selected_manifest = service.verify(
                            candidate,
                            require_identity=False,
                            verify_artifact_content=False,
                        )
                    except (FileNotFoundError, ValueError):
                        scan_counts["metadata_invalid"] += 1
                        print(
                            "CHECKPOINT_RESTORE_VERIFY "
                            f"state=failed sequence={candidate.name} "
                            f"elapsed_seconds={time.monotonic() - verification_started:.1f}",
                            flush=True,
                        )
                    else:
                        selected = candidate
                        selected_projection_rebuilds = projection_rebuilds
                        print(
                            "CHECKPOINT_RESTORE_VERIFY "
                            f"state=ready sequence={candidate.name} "
                            f"elapsed_seconds={time.monotonic() - verification_started:.1f}",
                            flush=True,
                        )
                        report_scan("selected")
                        break
                else:
                    scan_counts["source_incompatible"] += 1
                if scan_counts["checked"] % 10 == 0:
                    report_scan("progress")
        else:
            selected = service.latest()
        if selected is None:
            resume_diagnostics = self._pre_checkpoint_resume_diagnostics()
            print(
                "PRE_CHECKPOINT_RESUME_GATE "
                + json.dumps(
                    {
                        **resume_diagnostics,
                        "decision": (
                            "cold_replay" if resume_diagnostics["safe"] else "checkpoint_required"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if resume_diagnostics["safe"]:
                completed = sorted(self.state.completed_stage_ids)
                metadata = dict(self.state.metadata)
                metadata["pre_checkpoint_resume"] = {
                    "completed_stage_ids": completed,
                    "database_reinitialized": True,
                    "execution_attempt_id": self._execution_attempt_id,
                    "reason": "no_durable_scientific_facts_created",
                    "replayable_stage_count": resume_diagnostics[
                        "replayable_stage_count"
                    ],
                }
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()
                print(
                    "CHECKPOINT_RESTORE "
                    f"launch_id={self.launch_id} state=not_required "
                    "reason=pre_checkpoint_failure_without_runtime_facts "
                    "mode=cold_replay "
                    f"completed_stage_count={len(completed)}",
                    flush=True,
                )
                return
            print(
                f"CHECKPOINT_RESTORE launch_id={self.launch_id} state=not_found",
                flush=True,
            )
            raise FileNotFoundError(
                "self-contained resume requires a compatible verified experiment checkpoint"
            )
        if selected_manifest is None:
            verification_started = time.monotonic()
            print(
                f"CHECKPOINT_RESTORE_VERIFY state=start sequence={selected.name} source=latest",
                flush=True,
            )
            selected_manifest = service.verify(
                selected,
                require_identity=False,
                verify_artifact_content=False,
            )
            print(
                "CHECKPOINT_RESTORE_VERIFY "
                f"state=ready sequence={selected.name} "
                f"elapsed_seconds={time.monotonic() - verification_started:.1f}",
                flush=True,
            )
        print(
            "CHECKPOINT_RESTORE "
            f"launch_id={self.launch_id} state=selected "
            f"sequence={selected.name} milestone={selected_manifest.milestone}",
            flush=True,
        )
        print(
            "CHECKPOINT_RESTORE "
            f"launch_id={self.launch_id} state=restoring "
            f"sequence={selected.name} "
            f"checkpoint_revision={selected_manifest.alembic_revision} "
            "verification=complete",
            flush=True,
        )
        metadata = dict(self.state.metadata)
        metadata["checkpoint_restore_attempt"] = {
            "checkpoint_revision": selected_manifest.alembic_revision,
            "execution_attempt_id": self._execution_attempt_id,
            "milestone": selected_manifest.milestone,
            "sequence": selected.name,
        }
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()
        if source_id is not None:
            from pmmc.experiments.upstream_import import verify_receipt

            verify_receipt(self, ExperimentLaunchState.from_json(
                (selected / "launch_state.json").read_text(encoding="utf-8")
            ))
        restored = asyncio.run(service.restore(selected, verified_manifest=selected_manifest))
        self._checkpoint_restored = True
        checkpoint_state = restored / "launch_state.json"
        if checkpoint_state.is_file():
            restored_state = ExperimentLaunchState.from_json(
                checkpoint_state.read_text(encoding="utf-8")
            )
            if restored_state.launch_id != self.launch_id:
                raise ValueError("checkpoint launcher state has the wrong launch ID")
            self.state = restored_state
        if selected_projection_rebuilds:
            # Only completion flags are invalidated. Immutable source/checkpoint
            # files stay untouched and all scientific stages keep their gates.
            self.state = self.state.model_copy(update={
                "completed_stage_ids": [
                    stage for stage in self.state.completed_stage_ids
                    if stage not in selected_projection_rebuilds
                ],
                "stage_results": [
                    result for result in self.state.stage_results
                    if result.stage_id not in selected_projection_rebuilds
                ],
            })
            print("CHECKPOINT_PROJECTION_REBUILD " + json.dumps({
                "state": "required", "sequence": selected.name,
                "stages": sorted(selected_projection_rebuilds),
                "scientific_facts_retained": True, "old_exports_reusable": False,
            }, sort_keys=True, separators=(",", ":")), flush=True)
        metadata = dict(self.state.metadata)
        metadata["checkpoint_projection_rebuilds"] = sorted(selected_projection_rebuilds)
        metadata["restored_checkpoint"] = str(restored)
        # The checkpoint state is captured before its own directory can be
        # recorded, so make the verified restore the authoritative milestone.
        metadata["latest_checkpoint"] = str(restored)
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()
        print(
            "CHECKPOINT_RESTORE "
            f"launch_id={self.launch_id} state=ready "
            f"sequence={selected.name} "
            f"completed_stage_count={len(self.state.completed_stage_ids)}",
            flush=True,
        )

    def _maybe_checkpoint(self, stage_id: str) -> None:
        checkpoint = self.execution_config.checkpoint
        if (
            checkpoint.policy is CheckpointPolicy.DISABLED
            or self.config.infrastructure.mode is not InfrastructureMode.SELF_CONTAINED
            or stage_id
            in {
                "preflight",
                "model_prepare",
                "infrastructure",
                "model_calibration",
                "cleanup",
            }
        ):
            return
        coalesced_target = _step7_coalesced_checkpoint_target(
            self.config.target,
            stage_id,
        )
        if coalesced_target is not None:
            print(
                "CHECKPOINT "
                f"stage={stage_id} state=deferred "
                "reason=step7_coalesced_durable_boundary "
                f"next_durable_milestone={coalesced_target}",
                flush=True,
            )
            return
        if self._step6d_resume_verification_only():
            print(
                "CHECKPOINT "
                f"stage={stage_id} state=deferred "
                "reason=same_pod_resume_verification_read_only",
                flush=True,
            )
            return
        if (
            stage_id == "final_report"
            and self._step6d_restored_attempt_has_no_new_scientific_facts()
        ):
            print(
                "CHECKPOINT "
                f"stage={stage_id} state=deferred "
                "reason=restored_attempt_no_new_scientific_facts",
                flush=True,
            )
            return
        milestone_prefixes = (
            "snapshot_import:",
            "corpus:",
            "unit_prepare:",
            "questioner:",
            "questioner:model-stage",
            "compiler_program:model-stage",
            "doubter:model-stage",
            "step7_prefix",
            "program_text:model-stage",
            "program_multimodal:model-stage",
            "dry_run:",
            "verified_qbank:",
            "qbank_semantic:model-stage",
            "qbank_index_",
            "qbank_embed_",
            "qbank_coverage:",
            "retrieval:",
            "answer:",
            "evaluation:",
            "evaluation_",
            "export:",
            "automatic_quality",
            "review_bundle",
            "final_report",
        )
        if not stage_id.startswith(milestone_prefixes):
            return
        checkpoint_started = utc_now()
        performance_started = _performance_snapshot()
        if self.reporter is not None:
            self.reporter.report_operation(f"checkpoint:{stage_id}")
        try:
            with self._checkpoint_lock:
                service = self._checkpoint_service()
                directory = asyncio.run(
                    service.create(
                        stage_id,
                        stage_results_path=self.launch_dir / "stage_results.jsonl",
                        launch_state_path=self.state_path,
                    )
                )
        except Exception as exc:
            errors = _read_jsonl(self.launch_dir / "checkpoint_errors.jsonl")
            errors.append(
                {
                    "created_at": utc_now().isoformat(),
                    "execution_attempt_id": self._execution_attempt_id,
                    "stage_id": stage_id,
                    "error_type": type(exc).__name__,
                }
            )
            _atomic_jsonl(self.launch_dir / "checkpoint_errors.jsonl", errors)
            if self.config.mode is AppMode.PAPER:
                raise
            return
        finally:
            if self.reporter is not None:
                self.reporter.report_operation("launcher_idle")
        metadata = dict(self.state.metadata)
        metadata["latest_checkpoint"] = str(directory)
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()
        checkpoint_bytes = service.physical_size_bytes(directory)
        print(
            "CHECKPOINT_ACCOUNTING "
            + json.dumps(
                {
                    "physical_bytes": checkpoint_bytes,
                    "sequence": directory.name,
                    "stage": stage_id,
                    "state": "recorded",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        self._record_stage_performance(
            f"checkpoint:{stage_id}",
            checkpoint_started,
            utc_now(),
            performance_started,
            metadata={
                "status": "completed",
                "checkpoint_bytes": checkpoint_bytes,
            },
        )

    def _restored_checkpoint_matches(self, milestone: str) -> bool:
        """Return true when the restored checkpoint already seals this milestone."""

        restored = self.state.metadata.get("restored_checkpoint")
        if not isinstance(restored, str) or not restored:
            return False
        manifest_path = Path(restored) / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("launch_id") == self.launch_id and payload.get("milestone") == milestone

    def _step6d_resume_verification_only(self) -> bool:
        """Recognize a harness-only, same-Pod read-only resume pass."""

        step6d = os.environ.get(_STEP6D_RESUME_VERIFICATION_ENV) == "1"
        step7 = os.environ.get(_STEP7_RESUME_VERIFICATION_ENV) == "1"
        if not step6d and not step7:
            return False
        marker_value = os.environ.get(
            (
                _STEP7_RESUME_VERIFICATION_MARKER_ENV
                if step7
                else _STEP6D_RESUME_VERIFICATION_MARKER_ENV
            ),
            "",
        ).strip()
        marker = Path(marker_value) if marker_value else None
        expected_targets = (
            {
                ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
                ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
            }
            if step7
            else {ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT}
        )
        if (
            not self.resume
            or not self._checkpoint_restored
            or self.config.target not in expected_targets
            or marker is None
            or not marker.is_absolute()
            or not marker.is_file()
        ):
            raise RuntimeError(
                "read-only resume verification requires a restored matching "
                "pilot target and a Pod-local readiness marker"
            )
        return True

    def _maybe_controlled_step7_interrupt(self, stage_id: str) -> None:
        """Interrupt an armed Step 7 attempt only after its checkpoint is durable."""

        requested_stage = os.environ.get(_STEP7_CONTROLLED_INTERRUPT_STAGE_ENV, "").strip()
        if not requested_stage or requested_stage != stage_id:
            return
        if not _is_step7_pilot_target(self.config.target):
            raise ValueError("Step 7 controlled interruption used with another target")
        marker_value = os.environ.get(_STEP7_CONTROLLED_INTERRUPT_MARKER_ENV, "").strip()
        marker = Path(marker_value) if marker_value else None
        if marker is None or not marker.is_absolute():
            raise ValueError("Step 7 controlled interruption requires an absolute marker")
        latest = self.state.metadata.get("latest_checkpoint")
        if not isinstance(latest, str) or not (Path(latest) / "manifest.json").is_file():
            raise RuntimeError("Step 7 controlled interruption lacks a ready checkpoint")
        marker.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            marker,
            {
                "launch_id": self.launch_id,
                "stage_id": stage_id,
                "checkpoint": latest,
                "execution_attempt_id": self._execution_attempt_id,
            },
        )
        print(
            "STEP7_CONTROLLED_INTERRUPT "
            + json.dumps(
                {
                    "checkpoint_ready": True,
                    "launch_id": self.launch_id,
                    "stage": stage_id,
                    "state": "requested_after_milestone",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        raise Step7ControlledInterruption(f"controlled Step 7 interruption after {stage_id}")

    def _step6d_restored_attempt_has_no_new_scientific_facts(self) -> bool:
        """Detect a fully reused restore before writing an equivalent checkpoint."""

        if (
            not self.resume
            or not self._checkpoint_restored
            or self.config.target
            not in {
                ExperimentTarget.PMMC_COMPILER_VERIFIED_QBANK_PILOT,
                ExperimentTarget.PMMC_STEP7_EXECUTOR_PILOT,
                ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT,
            }
            or self._execution_attempt_id is None
        ):
            return False
        records = _read_jsonl(self.launch_dir / "stage_performance.jsonl")

        def record_metadata(item: dict[str, object]) -> dict[str, object]:
            value = item.get("metadata")
            return value if isinstance(value, dict) else {}

        scoped = [
            item
            for item in records
            if record_metadata(item).get("execution_attempt_id") == self._execution_attempt_id
        ]
        core_stages = {
            "compiler_program:model-stage",
            "program_text:model-stage",
            "program_multimodal:model-stage",
        }
        if _is_step7_pilot_target(self.config.target):
            core_stages.update(
                {
                    "step7_prefix",
                    "doubter:model-stage",
                    "answer:model-stage",
                    "evaluation_judge:model-stage",
                }
            )
        if not core_stages <= {str(item.get("stage_id")) for item in scoped}:
            return False
        non_scientific_stages = {
            "preflight",
            "model_prepare",
            "infrastructure",
            "model_calibration",
            "final_report",
            "cleanup",
        }
        scientific = [
            item
            for item in scoped
            if str(item.get("stage_id")) not in non_scientific_stages
            and not str(item.get("stage_id", "")).startswith("checkpoint:")
        ]
        return bool(scientific) and all(
            item.get("reused") is True
            or (record_metadata(item).get("state") == "not_started_by_hard_gate")
            for item in scientific
        )

    def _periodic_checkpoint_stage(self, stage_id: str) -> bool:
        return self._checkpoint_ready and stage_id.startswith(
            (
                "questioner:",
                "questioner:model-stage",
                "compiler_program:model-stage",
                "doubter:model-stage",
                "program_text:model-stage",
                "program_multimodal:model-stage",
                "dry_run:",
                "qbank_semantic:model-stage",
                "qbank_index_",
                "qbank_embed_",
                "query_match:",
                "query_execute:",
                "retrieval:",
                "answer:",
                "answer:model-stage",
                "evaluation_judge",
            )
        )

    def _start_periodic_checkpoint(
        self, stage_id: str
    ) -> tuple[threading.Event, threading.Thread] | None:
        from pmmc.experiments.checkpoint import CheckpointCancelled

        checkpoint = self.execution_config.checkpoint
        if (
            checkpoint.policy is CheckpointPolicy.DISABLED
            or self.config.infrastructure.mode is not InfrastructureMode.SELF_CONTAINED
            or not self._periodic_checkpoint_stage(stage_id)
        ):
            return None
        stop = threading.Event()

        def worker() -> None:
            service = self._checkpoint_service()
            try:
                last_jobs = asyncio.run(service.completed_job_count())
            except Exception:
                last_jobs = 0
            last_checkpoint = time.monotonic()
            poll_seconds = min(
                30.0,
                max(1.0, float(checkpoint.interval_seconds) / 10.0),
            )
            while not stop.wait(poll_seconds):
                try:
                    jobs = asyncio.run(service.completed_job_count())
                    due = (
                        time.monotonic() - last_checkpoint >= checkpoint.interval_seconds
                        or jobs - last_jobs >= checkpoint.completed_jobs_interval
                    )
                    if not due:
                        continue
                    with self._checkpoint_lock:
                        directory = asyncio.run(
                            service.create(
                                f"periodic:{stage_id}",
                                stage_results_path=(self.launch_dir / "stage_results.jsonl"),
                                launch_state_path=self.state_path,
                                cancel_event=stop,
                            )
                        )
                    last_jobs = jobs
                    last_checkpoint = time.monotonic()
                    self._append_checkpoint_event(
                        {
                            "created_at": utc_now().isoformat(),
                            "stage_id": stage_id,
                            "kind": "periodic",
                            "directory": str(directory),
                            "completed_jobs": jobs,
                            "status": "ready",
                        }
                    )
                except CheckpointCancelled:
                    self._append_checkpoint_event(
                        {
                            "created_at": utc_now().isoformat(),
                            "stage_id": stage_id,
                            "kind": "periodic",
                            "status": "cancelled",
                        }
                    )
                    return
                except Exception as exc:
                    self._append_checkpoint_event(
                        {
                            "created_at": utc_now().isoformat(),
                            "stage_id": stage_id,
                            "kind": "periodic",
                            "error_type": type(exc).__name__,
                            "status": "failed",
                        }
                    )

        thread = threading.Thread(
            target=worker,
            name=f"pmmc-checkpoint-{stage_id[:32]}",
            daemon=True,
        )
        thread.start()
        return stop, thread

    def _stop_periodic_checkpoint(
        self,
        controller: tuple[threading.Event, threading.Thread] | None,
        *,
        stage_id: str,
    ) -> None:
        if controller is None:
            return
        stop, thread = controller
        stop.set()
        if thread.is_alive():
            print(
                f"CHECKPOINT stage={stage_id} state=cancel_requested",
                flush=True,
            )
        started = time.monotonic()
        next_report = _PERIODIC_CHECKPOINT_DRAIN_REPORT_SECONDS
        while thread.is_alive():
            elapsed = time.monotonic() - started
            remaining = _PERIODIC_CHECKPOINT_DRAIN_TIMEOUT_SECONDS - elapsed
            if remaining <= 0:
                print(
                    f"CHECKPOINT stage={stage_id} state=drain_timeout elapsed={int(elapsed)}s",
                    flush=True,
                )
                raise PeriodicCheckpointShutdownError(
                    "periodic checkpoint worker did not stop within "
                    f"{int(_PERIODIC_CHECKPOINT_DRAIN_TIMEOUT_SECONDS)} seconds "
                    f"for stage {stage_id}"
                )
            thread.join(
                timeout=min(
                    _PERIODIC_CHECKPOINT_DRAIN_REPORT_SECONDS,
                    remaining,
                )
            )
            elapsed = time.monotonic() - started
            if thread.is_alive() and elapsed >= next_report:
                print(
                    f"CHECKPOINT stage={stage_id} state=draining elapsed={int(elapsed)}s",
                    flush=True,
                )
                next_report += _PERIODIC_CHECKPOINT_DRAIN_REPORT_SECONDS

    def _append_checkpoint_event(self, value: dict[str, object]) -> None:
        with self._checkpoint_lock:
            path = self.launch_dir / "checkpoint_events.jsonl"
            events = _read_jsonl(path)
            events.append(value)
            _atomic_jsonl(path, events)

    def _run_calibration(self) -> None:
        from pmmc.experiments.upstream_import import preflight_imported_program_audits

        preflight_imported_program_audits(self)
        with self._stage("model_calibration") as item:
            if self.config.calibration_policy is CalibrationPolicy.AUTO:
                from pmmc.runtime.config import (
                    load_runtime_profile,
                    runtime_profile_safe_summary,
                )
                from pmmc.runtime.manager import calibrate_runtime

                if self.preparation is None:
                    raise RuntimeError("model calibration requires model preparation")
                profile = load_runtime_profile(self.launch_settings.resolved_runtime_profile_path)
                print(
                    "MODEL_CALIBRATION_PROFILE "
                    + json.dumps(
                        {
                            **runtime_profile_safe_summary(profile),
                            "paper_result": bool(self.config.metadata.get("paper_result")),
                            "state": "validated",
                            "verification_policy": self.preparation.verification_policy.value,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                calibrate_runtime(
                    self.launch_settings,
                    profile_path=self.launch_settings.resolved_runtime_profile_path,
                    verification_policy=self.preparation.verification_policy,
                    verified_cache_manifest_path=(self.preparation.verified_cache_manifest_path),
                    paper_result=bool(self.config.metadata.get("paper_result")),
                )
            assert self.reporter is not None
            self.reporter.advance(item.work_units)

    def _run_questioner_datasets(self) -> None:
        if self.execution_config.schedule is ExecutionSchedule.MODEL_STAGE_MAJOR_V1:
            self._run_questioner_datasets_stage_major()
            return
        for dataset in self.config.selection.datasets:
            if self._reuse_completed_questioner_dataset(dataset):
                continue
            snapshot_path, snapshot_id = self._prepare_snapshot(dataset)
            selection = self._selection_for_dataset(dataset, snapshot_id=snapshot_id)
            conversations, selected_count = self._questioner_denominator(
                dataset,
                snapshot_id,
                selection=selection,
            )
            compiler_run_id = (
                "pmmc-questioner-"
                + _stable_hash([self.launch_id, dataset.value, snapshot_id, conversations])[:24]
            )
            self._run_questioner_stage(
                dataset,
                snapshot_id,
                compiler_run_id,
                conversations,
                split=str(selection.metadata.get("partition_role") or "") or None,
            )
            self._run_qbank_index_stage(
                dataset,
                compiler_run_id,
                phase="index_text",
                runtime_stage="embedding_text",
                plan_stage=f"qbank_index_text:{dataset.value}",
            )
            self._run_qbank_index_stage(
                dataset,
                compiler_run_id,
                phase="index_multimodal",
                runtime_stage="embedding_multimodal",
                plan_stage=f"qbank_index_multimodal:{dataset.value}",
            )
            questioner_verification = self._verify_questioner_dataset(
                dataset,
                compiler_run_id,
            )
            if self.config.evaluate:
                coverage_run_id = self._run_qbank_coverage_stage(
                    dataset,
                    compiler_run_id,
                    selected_count,
                    selection=selection,
                )
            else:
                coverage_run_id = None
            runtime_export, coverage_export, export_verification = self._export_questioner_dataset(
                dataset,
                compiler_run_id,
                coverage_run_id,
            )
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                "snapshot_id": snapshot_id,
                "snapshot_path": str(snapshot_path),
                "compiler_run_id": compiler_run_id,
                "conversation_count": len(conversations),
                "selected_count": selected_count,
                "coverage_run_id": coverage_run_id,
                "runtime_export_dir": str(runtime_export),
                "coverage_export_dir": (
                    str(coverage_export) if coverage_export is not None else None
                ),
                "questioner_verification": questioner_verification,
                "questioner_export_verification": export_verification,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

    def _run_questioner_datasets_stage_major(self) -> None:
        """Run ordinary Questioner experiments with one service start per model role."""

        from pmmc.evaluation.qbank_workflow import QBankCoveragePhase

        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        if self._reuse_completed_questioner_stage_major():
            return
        work: list[dict[str, object]] = []
        for dataset in self.config.selection.datasets:
            snapshot_path, snapshot_id = self._prepare_snapshot(dataset)
            selection = self._selection_for_dataset(dataset, snapshot_id=snapshot_id)
            conversations, selected_count = self._questioner_denominator(
                dataset,
                snapshot_id,
                selection=selection,
            )
            compiler_run_id = (
                "pmmc-questioner-"
                + _stable_hash([self.launch_id, dataset.value, snapshot_id, conversations])[:24]
            )
            preparation, unit_count, compiler_run_id = self._prepare_questioner_units_stage(
                dataset=dataset,
                snapshot_id=snapshot_id,
                compiler_run_id=compiler_run_id,
                conversations=conversations,
            )
            split_value = selection.metadata.get("partition_role")
            item: dict[str, object] = {
                "dataset": dataset.value,
                "snapshot_id": snapshot_id,
                "snapshot_path": snapshot_path,
                "compiler_run_id": compiler_run_id,
                "conversations": conversations,
                "selected_count": selected_count,
                "selection": selection,
                "preparation": preparation,
                "unit_count": unit_count,
                "split": str(split_value) if split_value is not None else None,
            }
            work.append(item)
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                "snapshot_id": snapshot_id,
                "snapshot_path": str(snapshot_path),
                "compiler_run_id": compiler_run_id,
                "conversation_count": len(conversations),
                "selected_count": selected_count,
                "unit_preparation_dir": str(preparation),
                "unit_count": unit_count,
                "execution_schedule": self.execution_config.schedule.value,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

        self._run_questioner_model_stage(work)
        self._run_qbank_model_stage(
            work,
            index_phase="index_text",
            embedding_phase=(QBankCoveragePhase.EMBED_TEXT.value if self.config.evaluate else None),
            runtime_stage="embedding_text",
            plan_stage="qbank_text:model-stage",
        )
        self._run_qbank_model_stage(
            work,
            index_phase="index_multimodal",
            embedding_phase=(
                QBankCoveragePhase.EMBED_MULTIMODAL.value if self.config.evaluate else None
            ),
            runtime_stage="embedding_multimodal",
            plan_stage="qbank_multimodal:model-stage",
        )
        for item in work:
            dataset = DatasetName(str(item["dataset"]))
            item["questioner_verification"] = self._verify_questioner_dataset(
                dataset, str(item["compiler_run_id"])
            )

        for item in work:
            dataset = DatasetName(str(item["dataset"]))
            coverage_run_id = None
            if self.config.evaluate:
                coverage_run_id = self._score_qbank_coverage_stage_major(item)
                if coverage_run_id != item["coverage_run_id"]:
                    raise ValueError("coverage run identity changed between phases")
            runtime_export, coverage_export, export_verification = self._export_questioner_dataset(
                dataset,
                str(item["compiler_run_id"]),
                coverage_run_id,
            )
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                **dict(artifacts[dataset.value]),
                "coverage_run_id": coverage_run_id,
                "runtime_export_dir": str(runtime_export),
                "coverage_export_dir": (
                    str(coverage_export) if coverage_export is not None else None
                ),
                "questioner_verification": item["questioner_verification"],
                "questioner_export_verification": export_verification,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

    def _reuse_completed_questioner_stage_major(
        self,
        *,
        already_executed: frozenset[str] = frozenset(),
        expected_snapshot_ids: dict[str, str] | None = None,
    ) -> bool:
        if not self.resume:
            return False
        expected_datasets = {item.value for item in self.config.selection.datasets}
        if set(self.state.dataset_artifacts) != expected_datasets:
            return False
        stage_ids = [
            item.stage_id
            for item in self.plan.items
            if item.stage_id
            not in {
                "preflight",
                "model_prepare",
                "infrastructure",
                "model_calibration",
                "final_report",
                "cleanup",
            }
        ]
        if not set(stage_ids) <= set(self.state.completed_stage_ids):
            return False

        from pmmc.compiler.preparation import CompilationUnitPreparationService
        from pmmc.compiler.questioner_export import verify_questioner_export
        from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
        from pmmc.evaluation.qbank_workflow import verify_qbank_coverage_run
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.questioner_repositories import (
            verify_questioner_run,
        )

        async def verify_compiler(run_id: str) -> dict[str, object]:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with sessions() as session:
                    return await verify_questioner_run(
                        session,
                        run_id,
                        require_managed_runtime=True,
                    )
            finally:
                await engine.dispose()

        for dataset in sorted(expected_datasets):
            values = self.state.dataset_artifacts[dataset]
            if expected_snapshot_ids is not None and str(
                values.get("snapshot_id")
            ) != expected_snapshot_ids.get(dataset):
                raise ValueError("resumed Questioner snapshot differs from the current benchmark")
            preparation = Path(str(values["unit_preparation_dir"]))
            CompilationUnitPreparationService.verify(preparation)
            if not asyncio.run(verify_compiler(str(values["compiler_run_id"])))["valid"]:
                return False
            verify_questioner_export(Path(str(values["runtime_export_dir"])))
            if self.config.evaluate:
                coverage = asyncio.run(
                    verify_qbank_coverage_run(self.launch_settings, str(values["coverage_run_id"]))
                )
                if not coverage["valid"]:
                    return False
                verify_qbank_coverage_export(Path(str(values["coverage_export_dir"])))
        if self.config.target is ExperimentTarget.PMMC_QUESTIONER_PILOT:
            from pmmc.pilots.partition import verify_partition

            partition_dir_value = self.state.metadata.get("partition_dir")
            pilot_dir_value = self.state.metadata.get("pilot_dir")
            if not isinstance(partition_dir_value, str) or not isinstance(pilot_dir_value, str):
                return False
            partition = verify_partition(Path(partition_dir_value))
            expected_snapshots = {
                dataset: str(self.state.dataset_artifacts[dataset]["snapshot_id"])
                for dataset in sorted(expected_datasets)
            }
            actual_snapshots = {item.dataset.value: item.snapshot_id for item in partition.datasets}
            if actual_snapshots != expected_snapshots:
                return False
            if not self._verify_resumed_questioner_pilot_bundle(
                pilot_dir=Path(pilot_dir_value),
                partition_id=partition.partition_id,
                expected_datasets=expected_datasets,
            ):
                return False
        for stage_id in stage_ids:
            if stage_id not in already_executed:
                self._reuse_stage(stage_id)
        return True

    def _verify_resumed_questioner_pilot_bundle(
        self,
        *,
        pilot_dir: Path,
        partition_id: str,
        expected_datasets: set[str],
    ) -> bool:
        """Verify the mode-specific offline pilot artifact restored by a checkpoint."""

        from pmmc.pilots.bundle import (
            verify_pilot_bundle,
            verify_questioner_smoke_bundle,
        )

        pilot_mode = self.state.metadata.get("pilot_mode")
        if pilot_mode == "smoke":
            smoke = verify_questioner_smoke_bundle(pilot_dir)
            expected_runtime_exports = {
                dataset: str(self.state.dataset_artifacts[dataset]["runtime_export_dir"])
                for dataset in sorted(expected_datasets)
            }
            expected_coverage_exports = {
                dataset: str(self.state.dataset_artifacts[dataset]["coverage_export_dir"])
                for dataset in sorted(expected_datasets)
            }
            return (
                smoke.get("launch_id") == self.launch_id
                and smoke.get("partition_id") == partition_id
                and self.state.metadata.get("pilot_id") == f"smoke-{self.launch_id}"
                and smoke.get("runtime_exports") == expected_runtime_exports
                and smoke.get("coverage_exports") == expected_coverage_exports
                and smoke == self.state.metadata.get("pilot_manifest")
            )
        if pilot_mode == "quality_pilot":
            pilot = verify_pilot_bundle(pilot_dir)
            return (
                pilot.pilot_id == self.state.metadata.get("pilot_id")
                and pilot.partition_id == partition_id
            )
        return False

    def _run_questioner_pilot(self) -> None:
        if self.execution_config.schedule is ExecutionSchedule.MODEL_STAGE_MAJOR_V1:
            self._run_questioner_pilot_stage_major()
            return
        from pmmc.pilots.analysis import analyze_questioner_quality
        from pmmc.pilots.bundle import build_pilot_bundle, verify_pilot_bundle
        from pmmc.pilots.config import load_questioner_pilot_config
        from pmmc.pilots.io import read_json
        from pmmc.pilots.partition import (
            build_partition,
            load_partition,
            sample_development_queries,
            verify_partition,
        )
        from pmmc.schemas import BenchmarkPartitionSpec

        if self.config.questioner_pilot_config_path is None:
            raise RuntimeError("Questioner pilot config is unavailable")
        pilot_config = load_questioner_pilot_config(
            _resolve(self.settings, self.config.questioner_pilot_config_path)
        )
        snapshot_dirs: dict[DatasetName, Path] = {}
        snapshot_ids: dict[DatasetName, str] = {}
        for dataset in self.config.selection.datasets:
            snapshot_dir, snapshot_id = self._prepare_snapshot(dataset)
            snapshot_dirs[dataset] = snapshot_dir
            snapshot_ids[dataset] = snapshot_id

        with self._stage("partition") as item:
            configured_partition = self.config.selection.partition_manifest_path
            if configured_partition is not None:
                partition_dir = _resolve(self.settings, configured_partition)
                partition_manifest = verify_partition(partition_dir)
            else:
                partition_dir, partition_manifest = build_partition(
                    snapshot_dirs=[
                        snapshot_dirs[dataset]
                        for dataset in sorted(snapshot_dirs, key=lambda value: value.value)
                    ],
                    output_root=self.settings.artifacts_path / "pilots" / "partitions",
                    spec=BenchmarkPartitionSpec(
                        protocol_version=pilot_config.partition_protocol,
                        fold_count=pilot_config.fold_count,
                        development_fold=pilot_config.development_fold,
                        seed=pilot_config.partition_seed,
                    ),
                )
            partition_snapshots = {
                value.dataset: value.snapshot_id for value in partition_manifest.datasets
            }
            if partition_snapshots != snapshot_ids:
                raise ValueError("Questioner pilot partition does not match the current snapshots")
            assert self.reporter is not None
            self.reporter.advance(item.work_units)

        with self._stage("pilot_sample") as item:
            _manifest, assignments = load_partition(partition_dir)
            samples = sample_development_queries(
                manifest=partition_manifest,
                assignments=assignments,
                config=pilot_config,
            )
            _atomic_json(
                self.launch_dir / "pilot_samples.json",
                [sample.model_dump(mode="json") for sample in samples],
            )
            assert self.reporter is not None
            self.reporter.advance(item.work_units)

        samples_by_dataset = {sample.dataset: sample for sample in samples}
        for dataset in self.config.selection.datasets:
            sample = samples_by_dataset[dataset]
            selection = self._pilot_selection(sample)
            conversations, selected_count = self._questioner_denominator(
                dataset,
                snapshot_ids[dataset],
                selection=selection,
            )
            if selected_count != len(sample.query_ids) or conversations != sorted(
                sample.conversation_ids
            ):
                raise ValueError(
                    "Questioner pilot sample differs from the imported query denominator"
                )
            compiler_run_id = (
                "pmmc-questioner-"
                + _stable_hash(
                    [
                        self.launch_id,
                        partition_manifest.partition_id,
                        sample.sample_id,
                        dataset.value,
                        snapshot_ids[dataset],
                        conversations,
                    ]
                )[:24]
            )
            self._run_questioner_stage(
                dataset,
                snapshot_ids[dataset],
                compiler_run_id,
                conversations,
                split=PartitionRole.DEVELOPMENT.value,
            )
            self._run_qbank_index_stage(
                dataset,
                compiler_run_id,
                phase="index_text",
                runtime_stage="embedding_text",
                plan_stage=f"qbank_index_text:{dataset.value}",
            )
            self._run_qbank_index_stage(
                dataset,
                compiler_run_id,
                phase="index_multimodal",
                runtime_stage="embedding_multimodal",
                plan_stage=f"qbank_index_multimodal:{dataset.value}",
            )
            questioner_verification = self._verify_questioner_dataset(
                dataset,
                compiler_run_id,
            )
            coverage_run_id = self._run_qbank_coverage_stage(
                dataset,
                compiler_run_id,
                selected_count,
                selection=selection,
            )
            runtime_export, coverage_export, export_verification = self._export_questioner_dataset(
                dataset,
                compiler_run_id,
                coverage_run_id,
            )
            if coverage_export is None:
                raise RuntimeError("Questioner pilot requires a coverage export")
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                "snapshot_id": snapshot_ids[dataset],
                "snapshot_path": str(snapshot_dirs[dataset]),
                "compiler_run_id": compiler_run_id,
                "conversation_count": len(conversations),
                "selected_count": selected_count,
                "selection_hash": sample.selection_hash,
                "pilot_sample_id": sample.sample_id,
                "partition_id": partition_manifest.partition_id,
                "coverage_run_id": coverage_run_id,
                "runtime_export_dir": str(runtime_export),
                "coverage_export_dir": str(coverage_export),
                "questioner_verification": questioner_verification,
                "questioner_export_verification": export_verification,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

        runtime_exports = {
            dataset: Path(str(self.state.dataset_artifacts[dataset.value]["runtime_export_dir"]))
            for dataset in self.config.selection.datasets
        }
        coverage_exports = {
            dataset: Path(str(self.state.dataset_artifacts[dataset.value]["coverage_export_dir"]))
            for dataset in self.config.selection.datasets
        }
        sample_query_ids = {query_id for sample in samples for query_id in sample.query_ids}
        selected_assignments = [
            assignment for assignment in assignments if assignment.query_id in sample_query_ids
        ]
        with self._stage("automatic_quality") as item:
            automatic = analyze_questioner_quality(
                config=pilot_config,
                runtime_exports=runtime_exports,
                coverage_exports=coverage_exports,
                assignments=selected_assignments,
            )
            _atomic_json(self.launch_dir / "automatic_quality.json", automatic)
            assert self.reporter is not None
            self.reporter.advance(item.work_units)

        with self._stage("review_bundle") as item:
            pilot_dir, pilot_manifest = build_pilot_bundle(
                config=pilot_config,
                partition_dir=partition_dir,
                samples=samples,
                runtime_exports=runtime_exports,
                coverage_exports=coverage_exports,
                snapshot_dirs=snapshot_dirs,
                output_root=(self.settings.outputs_path / "questioner-quality-pilots"),
            )
            assert self.reporter is not None
            self.reporter.advance(item.work_units)

        with self._stage("export_verify") as item:
            verified = verify_pilot_bundle(pilot_dir)
            if verified.content_hash != pilot_manifest.content_hash:
                raise ValueError("Questioner pilot bundle identity changed after export")
            assert self.reporter is not None
            self.reporter.advance(item.work_units)
        metadata = dict(self.state.metadata)
        metadata.update(
            {
                "pilot_dir": str(pilot_dir),
                "pilot_id": pilot_manifest.pilot_id,
                "pilot_status": pilot_manifest.status.value,
                "partition_dir": str(partition_dir),
                "partition_id": partition_manifest.partition_id,
                "pilot_manifest": read_json(pilot_dir / "manifest.json"),
                "paper_result": False,
            }
        )
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()

    def _prepare_questioner_units_stage(
        self,
        *,
        dataset: DatasetName,
        snapshot_id: str,
        compiler_run_id: str,
        conversations: list[str],
    ) -> tuple[Path, int, str]:
        from pmmc.compiler.preparation import (
            CompilationUnitPreparationService,
            slice_unit_preparation,
        )
        from pmmc.compiler.questioner_config import load_questioner_config
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.repositories import (
            PostgresMemoryRepository,
            PostgresSessionRepository,
        )

        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        stage_id = f"unit_prepare:{dataset.value}"
        with self._stage(stage_id):
            assert self.reporter is not None
            reporter = self.reporter
            self._set_actual_work_units(stage_id, len(conversations))
            questioner = load_questioner_config(
                self.launch_settings.resolved_questioner_config_path
            )

            async def load_inputs() -> tuple[list, list]:  # type: ignore[type-arg]
                engine, sessions = create_session_factory(
                    self.launch_settings, DatabaseRole.RUNTIME
                )
                try:
                    selected = set(conversations)
                    return await asyncio.gather(
                        PostgresSessionRepository(sessions).list_snapshot(
                            snapshot_id, conversation_ids=selected
                        ),
                        PostgresMemoryRepository(sessions).list_snapshot(
                            snapshot_id, conversation_ids=selected
                        ),
                    )
                finally:
                    await engine.dispose()

            sessions_for_run, memories_for_run = asyncio.run(load_inputs())
            last_complete = 0
            last_logged_complete = 0
            last_logged_at = 0.0

            def progress(
                complete: int,
                total: int,
                session_count: int,
                unit_count: int,
                metadata: JsonDict,
            ) -> None:
                nonlocal last_complete, last_logged_at, last_logged_complete
                if complete > last_complete:
                    reporter.advance(
                        complete - last_complete,
                        dataset=dataset.value,
                        phase="prepare_units",
                    )
                    last_complete = complete
                reporter.update_subtask(f"unit-prep:{dataset.value}", current=complete, total=total)
                now = time.monotonic()
                log_stride = max(1, (total + 9) // 10)
                should_log = (
                    complete == total
                    or last_logged_complete == 0
                    or complete - last_logged_complete >= log_stride
                    or now - last_logged_at >= 30.0
                )
                if not reporter.is_tty and should_log:
                    print(
                        "UNIT_PREP_PROGRESS "
                        + json.dumps(
                            {
                                "cached_counter_calls": metadata.get("cached_processor_calls", 0),
                                "completed_conversations": complete,
                                "counter_calls": metadata.get("processor_calls", 0),
                                "dataset": dataset.value,
                                "image_cache_hits": metadata.get("image_cache_hits", 0),
                                "request_cache_hits": metadata.get("request_cache_hits", 0),
                                "reused_conversations": metadata.get("reused_conversations", 0),
                                "session_count": session_count,
                                "total_conversations": total,
                                "unit_count": unit_count,
                            },
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    last_logged_at = now
                    last_logged_complete = complete

            manifest, _descriptors, root = CompilationUnitPreparationService(
                settings=self.launch_settings,
                registry=self.resolved_registry,
                config=questioner,
                workers=self.execution_config.unit_preparation.workers,
                max_tasks_per_child=(self.execution_config.unit_preparation.max_tasks_per_child),
                cache_policy=self.execution_config.unit_preparation.cache_policy,
            ).prepare(
                run_id=compiler_run_id,
                snapshot_id=snapshot_id,
                sessions=sessions_for_run,
                memories=memories_for_run,
                on_progress=progress,
            )
            max_units_raw = self.config.metadata.get("max_units_per_dataset")
            if max_units_raw is not None:
                manifest, root = slice_unit_preparation(
                    root,
                    max_units=int(max_units_raw),
                    replica_count=self._replica_count("compiler"),
                )
                reporter.console.print(
                    "UNIT_SLICE "
                    f"dataset={dataset.value} units={manifest.unit_count} "
                    f"replicas={manifest.metadata.get('replica_count', 1)} "
                    f"routing_salt={manifest.metadata.get('replica_routing_salt', 0)} "
                    "covered_replicas="
                    f"{manifest.metadata.get('covered_replica_indices', [0])}"
                )
            if last_complete < len(conversations):
                reporter.advance(len(conversations) - last_complete)
            self._set_stage_performance_metadata(
                stage_id,
                {
                    "conversation_count": manifest.conversation_count,
                    "session_count": len(sessions_for_run),
                    "memory_count": len(memories_for_run),
                    "unit_count": manifest.unit_count,
                    **manifest.metadata,
                },
            )
            return root, manifest.unit_count, manifest.run_id

    def _questioner_child_command(
        self,
        *,
        snapshot_id: str,
        compiler_run_id: str,
        conversations: list[str],
        preparation: Path,
        split: str | None,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "pmmc",
            "compile-memory",
            "--target-stage",
            "questions",
            "--phase",
            "questioner",
            "--snapshot",
            snapshot_id,
            "--run-id",
            compiler_run_id,
            "--unit-preparation",
            str(preparation),
            "--config",
            str(self.launch_settings.resolved_questioner_config_path),
            "--compiler-model",
            str(self.config.models.compiler_model),
            "--seed",
            str(self.config.selection.seed),
        ]
        if split is not None:
            command.extend(["--split", split])
        for conversation_id in conversations:
            command.extend(["--conversation-id", conversation_id])
        return command

    def _index_child_command(self, compiler_run_id: str, phase: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pmmc",
            "compile-memory",
            "--resume-run-id",
            compiler_run_id,
            "--phase",
            phase,
            "--config",
            str(self.launch_settings.resolved_questioner_config_path),
            "--compiler-model",
            str(self.config.models.compiler_model),
        ]

    def _run_questioner_model_stage(
        self,
        work: list[dict[str, object]],
    ) -> None:
        from pmmc.runtime.manager import run_managed_commands

        stage_id = "questioner:model-stage"
        total = sum(_work_int(item, "unit_count") for item in work)
        with self._stage(stage_id):
            assert self.reporter is not None
            reporter = self.reporter
            self._set_actual_work_units(stage_id, max(1, total))

            def completed(index: int, _command: list[str]) -> None:
                item = work[index - 1]
                reporter.advance(
                    _work_int(item, "unit_count"),
                    dataset=str(item["dataset"]),
                    phase="questioner",
                )

            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="compiler",
                commands=[
                    self._questioner_child_command(
                        snapshot_id=str(item["snapshot_id"]),
                        compiler_run_id=str(item["compiler_run_id"]),
                        conversations=_work_strings(item, "conversations"),
                        preparation=Path(str(item["preparation"])),
                        split=(str(item["split"]) if item.get("split") is not None else None),
                    )
                    for item in work
                ],
                environment_overrides=self._child_environment(),
                command_completed=completed,
                replica_count=self._replica_count("compiler"),
            )
            self._set_stage_performance_metadata(
                stage_id,
                {
                    "dataset_count": len(work),
                    "command_count": len(work),
                    "job_count": total,
                    "model_concurrency": self.config.model_concurrency,
                    "model_alias": str(self.config.models.compiler_model),
                },
            )

    def _run_qbank_model_stage(
        self,
        work: list[dict[str, object]],
        *,
        index_phase: str,
        embedding_phase: str | None,
        runtime_stage: str,
        plan_stage: str,
    ) -> None:
        from pmmc.evaluation.qbank_workflow import QBankCoveragePhase
        from pmmc.runtime.manager import run_managed_commands

        total = sum(
            _work_int(item, "unit_count")
            + (_work_int(item, "selected_count") if embedding_phase is not None else 0)
            for item in work
        )
        with self._stage(plan_stage):
            assert self.reporter is not None
            reporter = self.reporter
            self._set_actual_work_units(plan_stage, max(1, total))
            commands: list[list[str]] = []
            command_work: list[tuple[dict[str, object], str]] = []
            pending_embed_commands: dict[int, list[str]] = {}
            for item in work:
                commands.append(
                    self._index_child_command(str(item["compiler_run_id"]), index_phase)
                )
                command_work.append((item, "index"))
                if embedding_phase is None:
                    continue
                coverage_run_id: str | None = None
                if embedding_phase == QBankCoveragePhase.EMBED_MULTIMODAL.value:
                    frozen_id = item.get("coverage_run_id")
                    if not isinstance(frozen_id, str) or not frozen_id:
                        raise RuntimeError(
                            "multimodal QBank embedding requires a frozen coverage run ID"
                        )
                    coverage_run_id = frozen_id
                embed_command = self._coverage_command(
                    DatasetName(str(item["dataset"])),
                    str(item["compiler_run_id"]),
                    embedding_phase,
                    selection=item["selection"],  # type: ignore[arg-type]
                    coverage_run_id=coverage_run_id,
                )
                commands.append(embed_command)
                command_work.append((item, "embed"))
                if embedding_phase == QBankCoveragePhase.EMBED_TEXT.value:
                    pending_embed_commands[id(item)] = embed_command

            def completed(index: int, _command: list[str]) -> None:
                item, kind = command_work[index - 1]
                if kind == "index" and embedding_phase == QBankCoveragePhase.EMBED_TEXT.value:
                    coverage_run_id = self._register_qbank_coverage_identity(item)
                    embed_command = pending_embed_commands[id(item)]
                    embed_command.extend(["--resume-coverage-run-id", coverage_run_id])
                amount = _work_int(
                    item,
                    "unit_count" if kind == "index" else "selected_count",
                )
                reporter.advance(
                    amount,
                    dataset=str(item["dataset"]),
                    phase=index_phase if kind == "index" else str(embedding_phase),
                )

            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage=runtime_stage,
                commands=commands,
                environment_overrides=self._child_environment(),
                command_completed=completed,
                replica_count=self._replica_count(runtime_stage),
            )
            self._set_stage_performance_metadata(
                plan_stage,
                {
                    "dataset_count": len(work),
                    "command_count": len(commands),
                    "index_unit_count": sum(_work_int(item, "unit_count") for item in work),
                    "query_count": sum(_work_int(item, "selected_count") for item in work)
                    if embedding_phase is not None
                    else 0,
                    "embedding_phase": embedding_phase,
                    "embedding_concurrency": (self.execution_config.qbank.embedding_concurrency),
                },
            )

    def _register_qbank_coverage_identity(self, item: dict[str, object]) -> str:
        """Freeze one coverage identity before any query embedding is written."""

        from pmmc.compiler.questioner_config import load_questioner_config
        from pmmc.evaluation.qbank_workflow import QBankCoveragePhase, run_qbank_coverage

        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        selection = item.get("selection")
        if not isinstance(selection, BenchmarkSelection):
            raise TypeError("QBank work item has no benchmark selection")
        frozen_id = item.get("coverage_run_id")
        if frozen_id is not None and (not isinstance(frozen_id, str) or not frozen_id):
            raise TypeError("QBank work item has an invalid coverage run ID")
        registered = asyncio.run(
            run_qbank_coverage(
                settings=self.launch_settings,
                registry=self.resolved_registry,
                config=load_questioner_config(self.launch_settings.resolved_questioner_config_path),
                compiler_run_id=str(item["compiler_run_id"]),
                selection=selection,
                coverage_run_id=frozen_id,
                phase=QBankCoveragePhase.REGISTER,
            )
        )
        item["coverage_run_id"] = registered.coverage_run_id
        dataset = str(item["dataset"])
        artifacts = dict(self.state.dataset_artifacts)
        artifacts[dataset] = {
            **dict(artifacts.get(dataset, {})),
            "coverage_run_id": registered.coverage_run_id,
        }
        self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
        self._save_state()
        print(
            "QBANK_COVERAGE_IDENTITY "
            + json.dumps(
                {
                    "dataset": dataset,
                    "state": "persisted",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return registered.coverage_run_id

    def _score_qbank_coverage_stage_major(self, item: dict[str, object]) -> str:
        from pmmc.compiler.questioner_config import load_questioner_config
        from pmmc.evaluation.qbank_workflow import (
            QBankCoveragePhase,
            QBankCoverageScoringError,
            run_qbank_coverage,
            verify_qbank_coverage_run,
        )

        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        dataset = DatasetName(str(item["dataset"]))
        stage_id = f"qbank_coverage:{dataset.value}"
        with self._stage(stage_id) as plan_item:
            assert self.reporter is not None
            self.reporter.set_stage_total(plan_item.work_units + 1)
            frozen_coverage_run_id = item.get("coverage_run_id")
            if not isinstance(frozen_coverage_run_id, str) or not frozen_coverage_run_id:
                raise RuntimeError("frozen QBank coverage run identity is unavailable")
            completed = 0

            def query_completed(_query_id: str, current: int, total: int) -> None:
                nonlocal completed
                completed += 1
                assert self.reporter is not None
                self.reporter.advance(
                    1,
                    dataset=dataset.value,
                    queries=f"{current}/{total}",
                    phase="query_scoring",
                )

            try:
                result = asyncio.run(
                    run_qbank_coverage(
                        settings=self.launch_settings,
                        registry=self.resolved_registry,
                        config=load_questioner_config(
                            self.launch_settings.resolved_questioner_config_path
                        ),
                        compiler_run_id=str(item["compiler_run_id"]),
                        coverage_run_id=frozen_coverage_run_id,
                        selection=item["selection"],  # type: ignore[arg-type]
                        phase=QBankCoveragePhase.SCORE,
                        on_query=query_completed,
                    )
                )
            except QBankCoverageScoringError as exc:
                safe_locator = _stable_hash([dataset.value, exc.query_id])[:12]
                self._stage_performance_metadata.setdefault(stage_id, {})["qbank_score_error"] = {
                    "dataset": dataset.value,
                    "safe_locator": safe_locator,
                    "operation": exc.operation,
                    "cause_type": exc.cause_type,
                    "cause_code": exc.cause_code,
                }
                print(
                    "QBANK_SCORE_ERROR "
                    f"dataset={dataset.value} safe_locator={safe_locator} "
                    f"operation={exc.operation} cause_type={exc.cause_type} "
                    f"cause_code={exc.cause_code}",
                    flush=True,
                )
                raise
            report = asyncio.run(
                verify_qbank_coverage_run(self.launch_settings, result.coverage_run_id)
            )
            if not report["valid"]:
                raise ValueError("QBank coverage verification failed")
            assert self.reporter is not None
            remaining = max(0, plan_item.work_units - completed)
            if remaining:
                self.reporter.advance(
                    remaining,
                    dataset=dataset.value,
                    state="verified_reuse",
                )
            self.reporter.advance(1, dataset=dataset.value, phase="completion_verified")
            return result.coverage_run_id

    def _program_identity_options(self, item: dict[str, object]) -> list[str]:
        if _is_program_pilot_target(self.config.target):
            audit_hash = item.get("program_pilot_audit_hash")
            if not isinstance(audit_hash, str) or len(audit_hash) != 64:
                raise ValueError("program pilot work item has no stable audit identity")
            return ["--program-pilot-audit-hash", audit_hash]
        freeze_path = self.config.questioner_freeze_manifest_path
        if freeze_path is None:
            raise ValueError("formal Planner execution has no Questioner Freeze Manifest")
        return [
            "--questioner-freeze-manifest",
            str(_resolve(self.settings, freeze_path)),
        ]

    def _program_common_options(self, item: dict[str, object]) -> list[str]:
        if self.config.planner_config_path is None:
            raise ValueError("Planner config path is unavailable")
        if self.config.compiler_foundation_config_path is None:
            raise ValueError("compiler foundation config path is unavailable")
        options = [
            "--target-stage",
            ("verified_qbank" if _is_verified_qbank_target(self.config.target) else "programs"),
            "--config",
            str(self.launch_settings.resolved_questioner_config_path),
            "--planner-config",
            str(_resolve(self.settings, self.config.planner_config_path)),
            "--foundation-config",
            str(_resolve(self.settings, self.config.compiler_foundation_config_path)),
            *self._program_identity_options(item),
        ]
        if _is_verified_qbank_target(self.config.target):
            if self.config.doubter_config_path is None:
                raise ValueError("verified-QBank execution has no Doubter config")
            options.extend(
                [
                    "--doubter-config",
                    str(_resolve(self.settings, self.config.doubter_config_path)),
                ]
            )
        if item.get("diagnostic_force_revision") is True:
            options.append("--diagnostic-force-revision")
        return options

    def _program_questioner_command(self, item: dict[str, object]) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "pmmc",
            "compile-memory",
            *self._program_common_options(item),
            "--phase",
            "questioner",
            "--snapshot",
            str(item["snapshot_id"]),
            "--run-id",
            str(item["compiler_run_id"]),
            "--unit-preparation",
            str(item["preparation"]),
            "--seed",
            str(self.config.selection.seed),
        ]
        split = item.get("split")
        if split is not None:
            command.extend(["--split", str(split)])
        for conversation_id in _work_strings(item, "conversations"):
            command.extend(["--conversation-id", conversation_id])
        return command

    def _program_resume_command(
        self,
        item: dict[str, object],
        phase: str,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pmmc",
            "compile-memory",
            *self._program_common_options(item),
            "--resume-run-id",
            str(item["compiler_run_id"]),
            "--phase",
            phase,
        ]

    def _run_program_compiler_model_stage(
        self,
        work: list[dict[str, object]],
    ) -> None:
        from pmmc.runtime.manager import run_managed_commands
        from pmmc.smoke import (
            verify_step6c_planner_conformance,
            verify_step6d_compiler_conformance,
        )

        stage_id = "compiler_program:model-stage"
        legacy_conformance = (
            _is_program_pilot_target(self.config.target)
            and str(self.config.metadata.get("pilot_mode")) == "smoke"
        )
        prefix_conformance_ready = _is_step7_pilot_target(self.config.target) and isinstance(
            self.state.metadata.get(
                "step7b_compiler_conformance"
                if self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
                else "step6d_compiler_conformance"
            ),
            dict,
        )
        step6d_conformance = (
            _is_verified_qbank_target(self.config.target)
            and bool(self.config.metadata.get("run_conformance"))
            and not prefix_conformance_ready
        )
        command_work: list[tuple[dict[str, object] | None, str]] = []
        if step6d_conformance:
            command_work.append((None, "step6d_compiler_conformance"))
        command_work.extend(
            [
                *((item, "questioner") for item in work),
                *((item, "planner") for item in work),
            ]
        )
        if legacy_conformance:
            command_work.append((None, "planner_conformance"))
        legacy_conformance_path = self.launch_dir / "step6c_planner_conformance.json"
        step6d_conformance_path = self.launch_dir / "step6d_compiler_conformance.json"
        unit_total = sum(
            _work_int(item, "unit_count") for item, _ in command_work if item is not None
        )
        conformance_work_units = int(legacy_conformance) + int(step6d_conformance)
        total = unit_total + conformance_work_units
        with self._stage(stage_id):
            self._set_actual_work_units(stage_id, max(1, total))
            assert self.reporter is not None
            reporter = self.reporter
            completed_questioner = 0

            def completed(index: int, _command: list[str]) -> None:
                nonlocal completed_questioner
                item, phase = command_work[index - 1]
                if item is None:
                    reporter.advance(1, phase=phase)
                    print(
                        f"STEP6D_CONFORMANCE stage={phase} state=verified",
                        flush=True,
                    )
                    return
                reporter.advance(
                    _work_int(item, "unit_count"),
                    dataset=str(item["dataset"]),
                    phase=phase,
                )
                if phase == "questioner":
                    completed_questioner += 1
                    if completed_questioner == len(work):
                        self._maybe_checkpoint("questioner:model-stage")

            commands: list[list[str]] = []
            for item, phase in command_work:
                if item is None:
                    if self.config.planner_config_path is None:
                        raise ValueError("Planner conformance has no Planner config")
                    if self.config.compiler_foundation_config_path is None:
                        raise ValueError("Planner conformance has no foundation config")
                    if phase == "step6d_compiler_conformance":
                        if self.config.doubter_config_path is None:
                            raise ValueError("Step 6D conformance has no Doubter config")
                        commands.append(
                            [
                                sys.executable,
                                "-m",
                                "pmmc",
                                "smoke",
                                "verify-step6d-compiler-conformance",
                                "--questioner-config",
                                str(self.launch_settings.resolved_questioner_config_path),
                                "--planner-config",
                                str(
                                    _resolve(
                                        self.settings,
                                        self.config.planner_config_path,
                                    )
                                ),
                                "--doubter-config",
                                str(
                                    _resolve(
                                        self.settings,
                                        self.config.doubter_config_path,
                                    )
                                ),
                                "--foundation-config",
                                str(
                                    _resolve(
                                        self.settings,
                                        self.config.compiler_foundation_config_path,
                                    )
                                ),
                                "--expected-replica-count",
                                str(self._replica_count("compiler")),
                                "--result-json",
                                str(step6d_conformance_path),
                                "--compact-output",
                            ]
                        )
                    else:
                        commands.append(
                            [
                                sys.executable,
                                "-m",
                                "pmmc",
                                "smoke",
                                "verify-step6c-planner-conformance",
                                "--planner-config",
                                str(
                                    _resolve(
                                        self.settings,
                                        self.config.planner_config_path,
                                    )
                                ),
                                "--foundation-config",
                                str(
                                    _resolve(
                                        self.settings,
                                        self.config.compiler_foundation_config_path,
                                    )
                                ),
                                "--expected-replica-count",
                                str(self._replica_count("compiler")),
                                "--result-json",
                                str(legacy_conformance_path),
                            ]
                        )
                elif phase == "questioner":
                    commands.append(self._program_questioner_command(item))
                else:
                    commands.append(self._program_resume_command(item, phase))
            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="compiler",
                commands=commands,
                environment_overrides=self._child_environment(),
                command_completed=completed,
                replica_count=self._replica_count("compiler"),
            )
            if legacy_conformance:
                verified = verify_step6c_planner_conformance(
                    legacy_conformance_path,
                    expected_replica_count=self._replica_count("compiler"),
                )
                metadata = dict(self.state.metadata)
                metadata.update(
                    {
                        "step6c_planner_conformance_path": str(legacy_conformance_path),
                        "step6c_planner_conformance": verified.model_dump(mode="json"),
                    }
                )
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()
            if step6d_conformance:
                verified_step6d = verify_step6d_compiler_conformance(
                    step6d_conformance_path,
                    expected_replica_count=self._replica_count("compiler"),
                )
                metadata = dict(self.state.metadata)
                metadata.update(
                    {
                        "step6d_compiler_conformance_path": str(step6d_conformance_path),
                        "step6d_compiler_conformance": (verified_step6d.model_dump(mode="json")),
                    }
                )
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()
            self._set_stage_performance_metadata(
                stage_id,
                {
                    "dataset_count": len(work),
                    "command_count": len(commands),
                    "questioner_and_planner_unit_count": unit_total,
                    "compiler_conformance_work_units": conformance_work_units,
                    "model_alias": str(self.config.models.compiler_model),
                    "replica_count": self._replica_count("compiler"),
                    "conformance": bool(conformance_work_units),
                },
            )

    def _run_program_embedding_model_stage(
        self,
        work: list[dict[str, object]],
        *,
        channel: str,
    ) -> None:
        from pmmc.evaluation.qbank_workflow import QBankCoveragePhase
        from pmmc.runtime.manager import run_managed_commands

        if channel == "text":
            runtime_stage = "embedding_text"
            stage_id = "program_text:model-stage"
            phases = ("qbank_index_text", "memory_index_text", "program_query_text")
            coverage_phase = QBankCoveragePhase.EMBED_TEXT.value
        elif channel == "multimodal":
            runtime_stage = "embedding_multimodal"
            stage_id = "program_multimodal:model-stage"
            phases = (
                "qbank_index_multimodal",
                "memory_index_multimodal",
                "program_query_multimodal",
            )
            coverage_phase = QBankCoveragePhase.EMBED_MULTIMODAL.value
        else:
            raise ValueError("program embedding channel must be text or multimodal")

        commands: list[list[str]] = []
        command_work: list[tuple[dict[str, object], str]] = []
        pending_coverage: dict[int, list[str]] = {}
        for item in work:
            for phase in phases:
                commands.append(self._program_resume_command(item, phase))
                command_work.append((item, phase))
            if not self.config.evaluate:
                continue
            frozen_coverage_run_id = item.get("coverage_run_id")
            coverage_command = self._coverage_command(
                DatasetName(str(item["dataset"])),
                str(item["compiler_run_id"]),
                coverage_phase,
                selection=item["selection"],  # type: ignore[arg-type]
                coverage_run_id=(
                    str(frozen_coverage_run_id) if isinstance(frozen_coverage_run_id, str) else None
                ),
            )
            commands.append(coverage_command)
            command_work.append((item, "coverage_embedding"))
            if channel == "text":
                pending_coverage[id(item)] = coverage_command
            elif not isinstance(frozen_coverage_run_id, str):
                raise RuntimeError("multimodal program stage requires a frozen coverage run ID")
            if channel == "multimodal" and isinstance(item.get("fallback_suite_id"), str):
                corpus_id = str(item["fallback_corpus_id"])
                suite_id = str(item["fallback_suite_id"])
                commands.extend(
                    [
                        [
                            sys.executable,
                            "-m",
                            "pmmc",
                            "baseline",
                            "build-index",
                            "--method",
                            BaselineMethod.NATIVE_MULTIMODAL_RAG.value,
                            "--corpus-id",
                            corpus_id,
                        ],
                        [
                            sys.executable,
                            "-m",
                            "pmmc",
                            "baseline",
                            "verify-index",
                            "--method",
                            BaselineMethod.NATIVE_MULTIMODAL_RAG.value,
                            "--corpus-id",
                            corpus_id,
                        ],
                        [
                            sys.executable,
                            "-m",
                            "pmmc",
                            "baseline",
                            "run-suite",
                            "--suite-id",
                            suite_id,
                            "--phase",
                            "retrieve",
                            "--method",
                            BaselineMethod.NATIVE_MULTIMODAL_RAG.value,
                        ],
                    ]
                )
                command_work.extend(
                    [
                        (item, "fallback_index"),
                        (item, "fallback_index_verify"),
                        (item, "fallback_retrieve"),
                    ]
                )

        def phase_work_units(item: dict[str, object], phase: str) -> int:
            if phase in {"coverage_embedding", "fallback_retrieve"}:
                return _work_int(item, "selected_count")
            if phase == "fallback_index_verify":
                return 1
            return _work_int(item, "unit_count")

        total = sum(phase_work_units(item, phase) for item, phase in command_work)
        print(
            "EMBEDDING_STAGE_CONFIGURATION "
            + json.dumps(
                {
                    "channel": channel,
                    "command_count": len(commands),
                    "dataset_count": len(work),
                    "embedding_concurrency": (self.execution_config.qbank.embedding_concurrency),
                    "replica_count": self._replica_count(runtime_stage),
                    "stage": stage_id,
                    "work_unit_denominator": total,
                    "work_unit_protocol": "phase_weighted_units_v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        with self._stage(stage_id):
            self._set_actual_work_units(stage_id, max(1, total))
            assert self.reporter is not None
            reporter = self.reporter

            def completed(index: int, _command: list[str]) -> None:
                item, phase = command_work[index - 1]
                if self.config.evaluate and channel == "text" and phase == "qbank_index_text":
                    coverage_run_id = self._register_qbank_coverage_identity(item)
                    pending_coverage[id(item)].extend(["--resume-coverage-run-id", coverage_run_id])
                if channel == "multimodal" and phase == "fallback_index_verify":
                    dataset = str(item["dataset"])
                    checkpoint_stage = f"{stage_id}:fallback_index_ready:{dataset}"
                    checkpoint_reused = self._restored_checkpoint_matches(checkpoint_stage)
                    print(
                        "STEP7_FALLBACK_INDEX_MILESTONE "
                        + json.dumps(
                            {
                                "dataset": dataset,
                                "phase": phase,
                                "stage": checkpoint_stage,
                                "state": (
                                    "checkpoint_reused"
                                    if checkpoint_reused
                                    else "checkpoint_requested"
                                ),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    if not checkpoint_reused:
                        self._maybe_checkpoint(checkpoint_stage)
                amount = phase_work_units(item, phase)
                reporter.advance(
                    amount,
                    dataset=str(item["dataset"]),
                    phase=phase,
                )

            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage=runtime_stage,
                commands=commands,
                environment_overrides=self._child_environment(),
                command_completed=completed,
                replica_count=self._replica_count(runtime_stage),
            )
            self._set_stage_performance_metadata(
                stage_id,
                {
                    "dataset_count": len(work),
                    "command_count": len(commands),
                    "channel": channel,
                    "embedding_concurrency": (self.execution_config.qbank.embedding_concurrency),
                    "replica_count": self._replica_count(runtime_stage),
                },
            )

    def _run_program_dry_run_stage(self, item: dict[str, object]) -> None:
        dataset = DatasetName(str(item["dataset"]))
        stage_id = f"dry_run:{dataset.value}"
        facts = self._load_step7_planner_preflight(str(item["compiler_run_id"]))
        self._set_actual_work_units(
            stage_id, max(1, _mapping_int(facts, "program_count")),
        )
        with self._stage(stage_id) as plan_item:
            phases = (
                ("dry_run", "r0_input")
                if _is_verified_qbank_target(self.config.target)
                else ("dry_run", "finalize")
            )
            for phase in phases:
                completed = subprocess.run(
                    self._program_resume_command(item, phase),
                    check=False,
                    env={**os.environ, **self._child_environment()},
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"program {phase} failed for {dataset.value} with "
                        f"exit code {completed.returncode}"
                    )
            if _is_verified_qbank_target(self.config.target):
                item["r0_manifest"] = self._load_r0_manifest(str(item["compiler_run_id"]))
                item["r0_gate_passed"] = bool(
                    _work_mapping(item, "r0_manifest").get("hard_gate_passed")
                )
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset.value] = {
                    **dict(artifacts.get(dataset.value, {})),
                    "r0_gate_passed": item["r0_gate_passed"],
                    "r0_manifest": item["r0_manifest"],
                }
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                self._save_state()
            assert self.reporter is not None
            self.reporter.advance(
                plan_item.actual_work_units or plan_item.work_units,
                dataset=dataset.value,
            )

    def _find_r0_manifest(self, run_id: str) -> dict[str, object] | None:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.experiment_repositories import (
            ExperimentUnitOfWork,
        )
        from pmmc.schemas import DataScope

        async def load() -> dict[str, object] | None:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                    assert uow.doubter is not None
                    manifest = await uow.doubter.get_r0_manifest(run_id)
                return None if manifest is None else manifest.model_dump(mode="json")
            finally:
                await engine.dispose()

        return asyncio.run(load())

    def _load_r0_manifest(self, run_id: str) -> dict[str, object]:
        manifest = self._find_r0_manifest(run_id)
        if manifest is None:
            raise RuntimeError("R0 input phase did not persist a manifest")
        return manifest

    def _load_step7_planner_preflight(self, run_id: str) -> dict[str, object]:
        """Read aggregate-only Planner facts before starting embedding services."""

        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.experiment_repositories import (
            ExperimentUnitOfWork,
        )
        from pmmc.schemas import DataScope, QBankRouteClass

        async def load() -> dict[str, object]:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                    assert uow.questioner is not None
                    assert uow.memory_programs is not None
                    assert uow.planner is not None
                    candidates = await uow.questioner.list_eligible_questions(run_id)
                    programs = await uow.memory_programs.list_for_run(run_id)
                    outcomes = await uow.planner.list_outcomes(run_id)
                candidate_by_id = {item.qbank_id: item for item in candidates}
                candidate_ids = set(candidate_by_id)
                r0_programs = [item for item in programs if item.round_id == 0]
                program_by_candidate = {item.qbank_id: item for item in r0_programs}
                program_candidate_ids = {item.qbank_id for item in r0_programs}
                outcome_candidate_ids = {item.qbank_id for item in outcomes}
                terminal_program_outcome_ids = {
                    item.qbank_id for item in outcomes if item.plan_id is not None
                }
                rejection_outcome_ids = {item.qbank_id for item in outcomes if item.plan_id is None}
                program_outcome_plan_mismatches = {
                    item.qbank_id
                    for item in outcomes
                    if item.plan_id is not None
                    and item.qbank_id in program_by_candidate
                    and item.plan_id != program_by_candidate[item.qbank_id].plan_id
                }
                program_pending_outcomes = program_candidate_ids - outcome_candidate_ids
                if not program_candidate_ids & outcome_candidate_ids:
                    planner_fact_lifecycle = "pre_dry_run"
                elif not program_pending_outcomes:
                    planner_fact_lifecycle = "post_dry_run"
                else:
                    planner_fact_lifecycle = "partial_dry_run"
                outcome_counts: dict[str, int] = {}
                for outcome in outcomes:
                    key = outcome.status.value
                    outcome_counts[key] = outcome_counts.get(key, 0) + 1
                return {
                    "answerable_program_count": sum(
                        candidate_by_id.get(item.qbank_id) is not None
                        and candidate_by_id[item.qbank_id].route_class is QBankRouteClass.ANSWERABLE
                        for item in r0_programs
                    ),
                    "accounted_candidate_count": len(program_candidate_ids | outcome_candidate_ids),
                    "duplicate_r0_program_count": len(r0_programs) - len(program_candidate_ids),
                    "duplicate_terminal_outcome_count": len(outcomes) - len(outcome_candidate_ids),
                    "eligible_candidate_count": len(candidates),
                    "outcome_counts": dict(sorted(outcome_counts.items())),
                    "planner_fact_lifecycle": planner_fact_lifecycle,
                    "program_count": len(r0_programs),
                    "program_outside_candidate_count": len(program_candidate_ids - candidate_ids),
                    "program_outcome_overlap_count": len(
                        program_candidate_ids & outcome_candidate_ids
                    ),
                    "program_outcome_plan_mismatch_count": len(program_outcome_plan_mismatches),
                    "program_pending_terminal_outcome_count": len(program_pending_outcomes),
                    "rejection_outcome_count": len(rejection_outcome_ids),
                    "rejection_outcome_with_program_count": len(
                        rejection_outcome_ids & program_candidate_ids
                    ),
                    "terminal_outcome_count": len(outcomes),
                    "terminal_outcome_outside_candidate_count": len(
                        outcome_candidate_ids - candidate_ids
                    ),
                    "terminal_program_outcome_count": len(terminal_program_outcome_ids),
                    "terminal_program_outcome_without_program_count": len(
                        terminal_program_outcome_ids - program_candidate_ids
                    ),
                    "unaccounted_candidate_count": len(
                        candidate_ids - (program_candidate_ids | outcome_candidate_ids)
                    ),
                }
            finally:
                await engine.dispose()

        return asyncio.run(load())

    def _load_step7_doubter_preflight(self, run_id: str) -> dict[str, object]:
        from sqlalchemy import func, select

        from pmmc.compiler.doubter_runner import verify_doubter_run
        from pmmc.infrastructure.database.compiler_models import (
            CompilerUnitModel,
            CompilerUnitStageResultModel,
        )
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.doubter_models import CandidateStageJobModel
        from pmmc.infrastructure.database.engine import create_session_factory

        async def load() -> dict[str, object]:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                report = await verify_doubter_run(sessions, run_id=run_id)
                async with sessions() as session:
                    unit_ids = set(await session.scalars(
                        select(CompilerUnitModel.compiler_unit_id).where(
                            CompilerUnitModel.run_id == run_id
                        )
                    ))
                    stages = list(await session.scalars(
                        select(CompilerUnitStageResultModel).where(
                            CompilerUnitStageResultModel.run_id == run_id,
                            CompilerUnitStageResultModel.stage == "doubter",
                            CompilerUnitStageResultModel.status == "success",
                        )
                    ))
                    unfinished_jobs = int(await session.scalar(
                        select(func.count()).select_from(CandidateStageJobModel).where(
                            CandidateStageJobModel.run_id == run_id,
                            CandidateStageJobModel.stage != "decision",
                            CandidateStageJobModel.status.in_(["pending", "running", "failed"]),
                        )
                    ) or 0)
                safe_report = {key: report[key] for key in (
                    "valid", "issues", "candidate_count", "terminal_candidate_count",
                    "assessment_count", "revision_count", "round_counts",
                    "provider_terminal_rejection_count", "provider_integrity_violation_count",
                )}
                safe_report["stage_accounting_complete"] = (
                    bool(unit_ids)
                    and {item.compiler_unit_id for item in stages} == unit_ids
                    and len(stages) == len(unit_ids)
                    and sum(item.produced_count for item in stages) == report["assessment_count"]
                    and sum(item.accepted_count + item.rejected_count for item in stages)
                    == report["candidate_count"]
                    and unfinished_jobs == 0
                )
                return safe_report
            finally:
                await engine.dispose()

        return asyncio.run(load())

    def _complete_remediation_stage(
        self,
        stage_id: str,
        *,
        reason_code: str,
        datasets: list[str],
    ) -> None:
        with self._stage(stage_id) as plan_item:
            self._set_stage_performance_metadata(
                stage_id,
                {
                    "datasets": datasets,
                    "reason_code": reason_code,
                    "state": "not_started_by_hard_gate",
                },
            )
            assert self.reporter is not None
            self.reporter.advance(
                plan_item.actual_work_units or plan_item.work_units,
                state="not_started_by_hard_gate",
            )

    def _run_program_doubter_model_stage(
        self,
        work: list[dict[str, object]],
    ) -> None:
        """Run feedback/revision rounds in one second compiler service session."""

        from pmmc.runtime.manager import run_managed_commands

        stage_id = "doubter:model-stage"
        commands = [self._program_resume_command(item, "doubter") for item in work]
        counts = [
            _mapping_int(_work_mapping(item, "r0_manifest"), "program_count") for item in work
        ]
        total = max(1, sum(counts))
        self._set_actual_work_units(stage_id, total)
        self._print_next_stage_preflight(
            stage_id,
            following_stage="verified_qbank",
            dataset_count=len(work),
            denominator=total,
            required_facts=("r0_input_manifest", "program_dry_run_assessments"),
        )
        with self._stage(stage_id):
            self._set_actual_work_units(stage_id, total)
            assert self.reporter is not None
            reporter = self.reporter

            def completed(index: int, _command: list[str]) -> None:
                item = work[index - 1]
                reporter.advance(
                    counts[index - 1],
                    dataset=str(item["dataset"]),
                    phase="doubter_r0_r3",
                    denominator_kind="r0_programs_terminal_through_r3",
                )

            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="compiler",
                commands=commands,
                environment_overrides=self._child_environment(),
                command_completed=completed,
                replica_count=self._replica_count("compiler"),
            )
            self._set_stage_performance_metadata(
                stage_id,
                {
                    "dataset_count": len(work),
                    "command_count": len(commands),
                    "maximum_round": 3,
                    "active_round": 2,
                    "model_alias": str(self.config.models.compiler_model),
                    "replica_count": self._replica_count("compiler"),
                    **(
                        {
                            "physical_model_startup_owner": True,
                            "shared_logical_roles": [
                                "doubter",
                                "planner_revision",
                                "reranker",
                                "answer",
                                "semantic_judge",
                                "answer_judge",
                            ],
                        }
                        if self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
                        else {}
                    ),
                },
            )

    def _run_verified_qbank_stage(self, item: dict[str, object]) -> None:
        dataset = _work_dataset(item)
        stage_id = f"verified_qbank:{dataset.value}"
        self._print_next_stage_preflight(
            stage_id,
            following_stage=f"query_match:{dataset.value}",
            dataset_count=1,
            denominator=_work_int(item, "unit_count"),
            required_facts=("doubter_assessments", "round_0_programs"),
        )
        with self._stage(stage_id) as plan_item:
            for phase in ("decision", "finalize"):
                completed = subprocess.run(
                    self._program_resume_command(item, phase),
                    check=False,
                    env={**os.environ, **self._child_environment()},
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"verified QBank {phase} failed for {dataset.value} with "
                        f"exit code {completed.returncode}"
                    )
            assert self.reporter is not None
            self.reporter.advance(
                plan_item.actual_work_units or plan_item.work_units,
                dataset=dataset.value,
                active_round=2,
                maximum_round=3,
            )

    def _print_next_stage_preflight(
        self,
        stage: str,
        *,
        following_stage: str,
        dataset_count: int,
        denominator: int,
        required_facts: tuple[str, ...],
    ) -> None:
        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
        print(
            ("NEXT_TWO_STAGE_PREFLIGHT " if step7b else "NEXT_STAGE_PREFLIGHT ")
            + json.dumps(
                {
                    "dataset_count": dataset_count,
                    "denominator": denominator,
                    **({"next_stages": [stage, following_stage]} if step7b else {}),
                    "required_facts": list(required_facts),
                    "stage": stage,
                    "state": "ready",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    def _run_qbank_semantic_model_stage(
        self,
        work: list[dict[str, object]],
    ) -> None:
        from pmmc.evaluation.qbank_semantic import load_qbank_semantic_config
        from pmmc.evaluation.qbank_semantic_workflow import (
            inspect_qbank_semantic_parent,
            verify_qbank_semantic_match_run,
        )
        from pmmc.runtime.manager import run_managed_commands
        from pmmc.smoke import verify_step6c2_semantic_conformance

        if self.config.qbank_semantic_config_path is None:
            raise ValueError("semantic r=0 execution has no semantic Judge config")
        semantic_config_path = _resolve(
            self.settings,
            self.config.qbank_semantic_config_path,
        )
        semantic_config = load_qbank_semantic_config(semantic_config_path)
        if semantic_config.judge_model_alias != self.config.models.judge_model:
            raise ValueError(
                "semantic Judge config alias differs from the launcher model selection"
            )

        stage_id = "qbank_semantic:model-stage"
        legacy_conformance = (
            _is_program_pilot_target(self.config.target)
            and str(self.config.metadata.get("pilot_mode")) == "smoke"
        )
        step6d_conformance = _is_verified_qbank_target(self.config.target) and bool(
            self.config.metadata.get("run_conformance")
        )
        run_conformance = legacy_conformance or step6d_conformance
        conformance_path = (
            self.launch_dir / "step6d_semantic_conformance.json"
            if step6d_conformance
            else self.launch_dir / "step6c2_semantic_conformance.json"
        )
        commands: list[list[str]] = []
        command_work: list[tuple[dict[str, object] | None, Path]] = []
        if run_conformance:
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "pmmc",
                    "smoke",
                    "verify-step6c2-semantic-conformance",
                    "--semantic-config",
                    str(semantic_config_path),
                    "--expected-replica-count",
                    str(self._replica_count("judge")),
                    "--result-json",
                    str(conformance_path),
                    "--summary-only",
                ]
            )
            command_work.append((None, conformance_path))
        for item in work:
            dataset = _work_dataset(item)
            coverage_run_id = item.get("coverage_run_id")
            if not isinstance(coverage_run_id, str) or not coverage_run_id:
                raise ValueError(
                    f"semantic matching has no frozen coverage run for {dataset.value}"
                )
            result_path = self.launch_dir / "semantic" / f"{dataset.value}.result.json"
            command = [
                sys.executable,
                "-m",
                "pmmc",
                "evaluate",
                "qbank-semantic-match",
                "--coverage-run-id",
                coverage_run_id,
                "--config",
                str(semantic_config_path),
                "--result-json",
                str(result_path),
            ]
            existing_run_id = item.get("semantic_run_id")
            if isinstance(existing_run_id, str) and existing_run_id:
                command.extend(["--resume-semantic-run-id", existing_run_id])
            commands.append(command)
            command_work.append((item, result_path))

        total_queries = sum(_work_int(item, "selected_count") for item in work)
        with self._stage(stage_id):
            self._set_actual_work_units(
                stage_id,
                max(1, total_queries + int(run_conformance)),
            )
            assert self.reporter is not None
            reporter = self.reporter
            for item in work:
                dataset = _work_dataset(item)
                parent_report = asyncio.run(
                    inspect_qbank_semantic_parent(
                        settings=self.launch_settings,
                        coverage_run_id=str(item["coverage_run_id"]),
                        config=semantic_config,
                    )
                )
                print(
                    "QBANK_SEMANTIC_PARENT_PREFLIGHT "
                    + json.dumps(
                        parent_report,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                if parent_report.get("state") != "ready":
                    self._set_stage_performance_metadata(
                        stage_id,
                        {
                            "dataset": dataset.value,
                            "model_startup_count": 0,
                            "preflight": parent_report,
                            "state": "blocked_by_semantic_parent",
                        },
                    )
                    raise ValueError("semantic parent preflight failed before Judge startup")

            def completed(index: int, _command: list[str]) -> None:
                item, result_path = command_work[index - 1]
                if item is None:
                    reporter.advance(1, phase="semantic_conformance")
                    print(
                        "STEP6D_CONFORMANCE stage=semantic_judge state=verified",
                        flush=True,
                    )
                    return
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                semantic_run_id = payload.get("semantic_run_id")
                if not isinstance(semantic_run_id, str) or not semantic_run_id:
                    raise ValueError("semantic Judge result has no semantic_run_id")
                if payload.get("coverage_run_id") != item.get("coverage_run_id"):
                    raise ValueError("semantic Judge changed the coverage parent")
                item["semantic_run_id"] = semantic_run_id
                item["semantic_pair_count"] = int(payload.get("pair_count", 0))
                dataset_key = str(item["dataset"])
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset_key] = {
                    **dict(artifacts.get(dataset_key, {})),
                    "semantic_pair_count": item["semantic_pair_count"],
                    "semantic_run_id": semantic_run_id,
                }
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                self._save_state()
                reporter.advance(
                    _work_int(item, "selected_count"),
                    dataset=str(item["dataset"]),
                    pairs=int(payload.get("pair_count", 0)),
                )

            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="judge",
                commands=commands,
                environment_overrides=self._child_environment(),
                command_completed=completed,
                replica_count=self._replica_count("judge"),
            )

            if run_conformance:
                verified_conformance = verify_step6c2_semantic_conformance(
                    conformance_path,
                    expected_replica_count=self._replica_count("judge"),
                )
                metadata = dict(self.state.metadata)
                prefix = (
                    "step6d_semantic_conformance"
                    if step6d_conformance
                    else "step6c2_semantic_conformance"
                )
                metadata[f"{prefix}_path"] = str(conformance_path)
                metadata[prefix] = verified_conformance.model_dump(mode="json")
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()

            pair_count = 0
            for item in work:
                semantic_run_id = item.get("semantic_run_id")
                if not isinstance(semantic_run_id, str) or not semantic_run_id:
                    raise ValueError("semantic Judge command did not publish its run identity")
                report = asyncio.run(
                    verify_qbank_semantic_match_run(
                        self.launch_settings,
                        semantic_run_id,
                        semantic_config,
                    )
                )
                if not bool(report.get("valid")):
                    raise ValueError("semantic Judge run verification failed")
                raw_pair_count = report.get("pair_count", 0)
                if not isinstance(raw_pair_count, int):
                    raise ValueError("semantic Judge pair count is invalid")
                pair_count += raw_pair_count
                item["semantic_verification"] = report
            self._set_stage_performance_metadata(
                stage_id,
                {
                    "dataset_count": len(work),
                    "pair_count": pair_count,
                    "model_alias": self.config.models.judge_model,
                    "replica_count": self._replica_count("judge"),
                    "semantic_conformance": run_conformance,
                },
            )

    def _verify_program_dataset(
        self,
        dataset: DatasetName,
        run_id: str,
    ) -> dict[str, object]:
        from pmmc.infrastructure.database.compiler_repositories import verify_compiler_run
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory

        stage_id = f"planner_verify:{dataset.value}"
        with self._stage(stage_id) as plan_item:

            async def verify() -> dict[str, object]:
                engine, sessions = create_session_factory(
                    self.launch_settings, DatabaseRole.RUNTIME
                )
                try:
                    async with sessions() as session:
                        return await verify_compiler_run(session, run_id)
                finally:
                    await engine.dispose()

            report = asyncio.run(verify())
            if not bool(report.get("valid")):
                raise ValueError("Planner run verification failed")
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units, dataset=dataset.value)
            return report

    def _export_program_dataset(
        self,
        dataset: DatasetName,
        run_id: str,
        coverage_run_id: str,
    ) -> tuple[Path, Path, dict[str, object]]:
        from pmmc.compiler.program_export import verify_program_export
        from pmmc.evaluation.qbank_export import (
            QBankCoverageExporter,
            verify_qbank_coverage_export,
        )
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_export import ExperimentExporter
        from pmmc.infrastructure.object_store import create_runtime_object_store

        stage_id = f"export:{dataset.value}"
        with self._stage(stage_id) as plan_item:

            async def export_runtime() -> Path:
                engine, sessions = create_session_factory(
                    self.launch_settings, DatabaseRole.RUNTIME
                )
                try:
                    result = await ExperimentExporter(
                        sessions,
                        runtime_store=create_runtime_object_store(self.launch_settings),
                        batch_size=self.launch_settings.experiment_export_batch_size,
                    ).export(
                        run_id,
                        self.launch_settings.experiment_outputs_path,
                        include_referenced_model_artifacts=True,
                    )
                    return result.export_dir
                finally:
                    await engine.dispose()

            runtime_export = asyncio.run(export_runtime())
            runtime_verification = verify_program_export(runtime_export)
            coverage_result = asyncio.run(
                QBankCoverageExporter(self.launch_settings).export(coverage_run_id)
            )
            verify_qbank_coverage_export(coverage_result.export_dir)
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units, dataset=dataset.value)
            return runtime_export, coverage_result.export_dir, runtime_verification

    def _export_program_diagnostic_dataset(
        self,
        dataset: DatasetName,
        run_id: str,
        r0_manifest: dict[str, object],
    ) -> tuple[Path, dict[str, object]]:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_export import ExperimentExporter
        from pmmc.infrastructure.export_verification import verify_experiment_export
        from pmmc.infrastructure.object_store import create_runtime_object_store

        stage_id = f"export:{dataset.value}"
        with self._stage(stage_id) as plan_item:

            async def export_runtime() -> Path:
                engine, sessions = create_session_factory(
                    self.launch_settings, DatabaseRole.RUNTIME
                )
                try:
                    result = await ExperimentExporter(
                        sessions,
                        runtime_store=create_runtime_object_store(self.launch_settings),
                        batch_size=self.launch_settings.experiment_export_batch_size,
                    ).export(
                        run_id,
                        self.launch_settings.experiment_outputs_path,
                        require_completed=False,
                        include_referenced_model_artifacts=True,
                    )
                    return result.export_dir
                finally:
                    await engine.dispose()

            runtime_export = asyncio.run(export_runtime())
            manifest = verify_experiment_export(runtime_export)
            verification = {
                "valid": False,
                "diagnostic_export": True,
                "export_id": manifest["export_id"],
                "r0_hard_gate_passed": False,
                "r0_issue_codes": _object_list(
                    r0_manifest.get("issue_codes"),
                    locator="R0 issue_codes",
                ),
            }
            assert self.reporter is not None
            self.reporter.advance(
                plan_item.work_units,
                dataset=dataset.value,
                state="r0_needs_remediation",
            )
            return runtime_export, verification

    def _program_work_item(
        self,
        *,
        dataset: DatasetName,
        snapshot_id: str,
        snapshot_path: Path,
        selection: BenchmarkSelection,
        conversations: list[str],
        selected_count: int,
        compiler_run_id: str,
        split: str | None,
        sample: QuestionerPilotSample | None = None,
        partition_id: str | None = None,
    ) -> dict[str, object]:
        stage_id = f"unit_prepare:{dataset.value}"
        restored = self.state.dataset_artifacts.get(dataset.value, {})
        if (
            self.resume
            and self._checkpoint_restored
            and stage_id in self.state.completed_stage_ids
            and restored.get("unit_preparation_dir") is not None
        ):
            from pmmc.compiler.preparation import (
                CompilationUnitPreparationService,
            )

            preparation = Path(str(restored["unit_preparation_dir"]))
            manifest, _descriptors = CompilationUnitPreparationService.verify(preparation)
            if (
                manifest.snapshot_id != snapshot_id
                or manifest.conversation_ids != conversations
                or int(restored.get("selected_count", -1)) != selected_count
                or str(restored.get("compiler_run_id", "")) != manifest.run_id
            ):
                raise ValueError(
                    "restored unit preparation differs from the current pilot denominator"
                )
            unit_count = manifest.unit_count
            resolved_run_id = manifest.run_id
            self._set_actual_work_units(stage_id, max(1, unit_count))
            self._reuse_restored_stage(stage_id)
        else:
            preparation, unit_count, resolved_run_id = self._prepare_questioner_units_stage(
                dataset=dataset,
                snapshot_id=snapshot_id,
                compiler_run_id=compiler_run_id,
                conversations=conversations,
            )
        item: dict[str, object] = {
            "dataset": dataset.value,
            "snapshot_id": snapshot_id,
            "snapshot_path": snapshot_path,
            "compiler_run_id": resolved_run_id,
            "conversations": conversations,
            "selected_count": selected_count,
            "selection": selection,
            "preparation": preparation,
            "unit_count": unit_count,
            "split": split,
        }
        if sample is not None:
            item["sample"] = sample
        if partition_id is not None:
            origin = self.state.metadata.get("upstream_import")
            origin = origin if isinstance(origin, dict) else {}
            audit_inputs = {
                "protocol": (
                    "compiler-verified-qbank-development-audit-v1"
                    if _is_verified_qbank_target(self.config.target)
                    else (
                        "compiler-program-r0-development-audit-v2"
                        if _is_semantic_program_target(self.config.target)
                        else "compiler-program-development-audit-v1"
                    )
                ),
                "launch_id": self.launch_id,
                "partition_id": partition_id,
                "sample_id": sample.sample_id if sample is not None else None,
                "dataset": dataset.value,
                "snapshot_id": snapshot_id,
                "selection_hash": (
                    sample.selection_hash
                    if sample is not None
                    else _stable_hash(selection.model_dump(mode="json"))
                ),
                "conversation_ids": conversations,
                "unit_preparation_id": preparation.name,
                "launcher_config_hash": self.effective_config_hash,
            }
            if origin:
                from pmmc.experiments.upstream_import import imported_program_audit_hash

                item["program_pilot_audit_hash"] = imported_program_audit_hash(
                    self, dataset=dataset.value, inputs=audit_inputs,
                )
            else:
                item["program_pilot_audit_hash"] = _stable_hash(audit_inputs)
        for key in (
            "coverage_run_id",
            "r0_gate_passed",
            "r0_manifest",
            "semantic_run_id",
            "evaluation_run_id",
        ):
            if key in restored:
                item[key] = restored[key]
        artifacts = dict(self.state.dataset_artifacts)
        artifacts[dataset.value] = {
            **dict(restored),
            "snapshot_id": snapshot_id,
            "snapshot_path": str(snapshot_path),
            "compiler_run_id": resolved_run_id,
            "conversation_count": len(conversations),
            "selected_count": selected_count,
            "unit_count": unit_count,
            "unit_preparation_dir": str(preparation),
            "split": split,
            "execution_schedule": self.execution_config.schedule.value,
            **(
                {
                    "partition_id": partition_id,
                    "pilot_sample_id": sample.sample_id,
                    "selection_hash": sample.selection_hash,
                    "program_pilot_audit_hash": item["program_pilot_audit_hash"],
                }
                if sample is not None and partition_id is not None
                else {}
            ),
        }
        self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
        self._save_state()
        return item

    def _execute_program_work(
        self,
        work: list[dict[str, object]],
        *,
        run_semantic_judge: bool = True,
        upstream_only: bool = False,
        post_r0_only: bool = False,
    ) -> None:
        if upstream_only and post_r0_only:
            raise ValueError("program work cannot be both upstream-only and post-R0-only")
        if not post_r0_only:
            if not self._reuse_restored_stage("compiler_program:model-stage"):
                self._run_program_compiler_model_stage(work)
            can_reuse_text = not self.config.evaluate or all(
                isinstance(item.get("coverage_run_id"), str) for item in work
            )
            if not (can_reuse_text and self._reuse_restored_stage("program_text:model-stage")):
                self._run_program_embedding_model_stage(work, channel="text")
            if not self._reuse_restored_stage("program_multimodal:model-stage"):
                self._run_program_embedding_model_stage(work, channel="multimodal")
            for item in work:
                dry_run_stage = f"dry_run:{item['dataset']}"
                if self._reuse_restored_stage(dry_run_stage):
                    if _is_verified_qbank_target(self.config.target):
                        item["r0_manifest"] = self._load_r0_manifest(str(item["compiler_run_id"]))
                        item["r0_gate_passed"] = bool(
                            _work_mapping(item, "r0_manifest").get("hard_gate_passed")
                        )
                else:
                    self._run_program_dry_run_stage(item)
        if upstream_only:
            return

        if _is_verified_qbank_target(self.config.target):
            passing = [item for item in work if item.get("r0_gate_passed") is True]
            blocked = [item for item in work if item.get("r0_gate_passed") is not True]
            if passing:
                if not self._reuse_restored_stage("doubter:model-stage"):
                    self._run_program_doubter_model_stage(passing)
            else:
                self._complete_remediation_stage(
                    "doubter:model-stage",
                    reason_code="r0_input_gate_failed",
                    datasets=[str(item["dataset"]) for item in blocked],
                )
            for item in passing:
                if not self._reuse_restored_stage(f"verified_qbank:{item['dataset']}"):
                    self._run_verified_qbank_stage(item)
            for item in blocked:
                self._complete_remediation_stage(
                    f"verified_qbank:{item['dataset']}",
                    reason_code="r0_input_gate_failed",
                    datasets=[str(item["dataset"])],
                )

        for item in work:
            if item.get("r0_gate_passed") is False:
                r0_manifest = _work_mapping(item, "r0_manifest")
                item["program_verification"] = {
                    "valid": False,
                    "r0_hard_gate_passed": False,
                    "r0_issue_codes": _object_list(
                        r0_manifest.get("issue_codes"),
                        locator="R0 issue_codes",
                    ),
                    "state": "needs_remediation",
                }
                self._complete_remediation_stage(
                    f"planner_verify:{item['dataset']}",
                    reason_code="r0_input_gate_failed",
                    datasets=[str(item["dataset"])],
                )
            else:
                item["program_verification"] = self._verify_program_dataset(
                    _work_dataset(item), str(item["compiler_run_id"])
                )

        for item in work:
            if item.get("r0_gate_passed") is False:
                self._complete_remediation_stage(
                    f"qbank_coverage:{item['dataset']}",
                    reason_code="r0_input_gate_failed",
                    datasets=[str(item["dataset"])],
                )
                continue
            if self.config.target is ExperimentTarget.PMMC_COMPILE_ONLY:
                continue
            coverage_run_id: str | None = None
            if self.config.evaluate:
                coverage_run_id = self._score_qbank_coverage_stage_major(item)
                if coverage_run_id != item.get("coverage_run_id"):
                    raise ValueError("coverage run identity changed between program phases")
            if coverage_run_id is None:
                raise ValueError("Step 6C program export currently requires coverage evaluation")
            item["coverage_run_id"] = coverage_run_id

        if _is_semantic_program_target(self.config.target) and run_semantic_judge:
            semantic_work = [item for item in work if item.get("r0_gate_passed") is not False]
            if semantic_work:
                self._run_qbank_semantic_model_stage(semantic_work)
            else:
                self._complete_remediation_stage(
                    "qbank_semantic:model-stage",
                    reason_code="r0_input_gate_failed",
                    datasets=[str(item["dataset"]) for item in work],
                )

        for item in work:
            if item.get("r0_gate_passed") is False:
                runtime_export, export_verification = self._export_program_diagnostic_dataset(
                    _work_dataset(item),
                    str(item["compiler_run_id"]),
                    _work_mapping(item, "r0_manifest"),
                )
                dataset = _work_dataset(item)
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset.value] = {
                    **dict(artifacts[dataset.value]),
                    "runtime_export_dir": str(runtime_export),
                    "program_verification": item["program_verification"],
                    "program_export_verification": export_verification,
                    "r0_manifest": item["r0_manifest"],
                    "r0_gate_passed": False,
                }
                item["runtime_export_dir"] = str(runtime_export)
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                self._save_state()
                continue
            if self.config.target is ExperimentTarget.PMMC_COMPILE_ONLY:
                runtime_export, export_verification = self._export_verified_qbank_runtime_dataset(
                    _work_dataset(item),
                    str(item["compiler_run_id"]),
                )
                item["runtime_export_dir"] = str(runtime_export)
                dataset = _work_dataset(item)
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset.value] = {
                    **dict(artifacts[dataset.value]),
                    "runtime_export_dir": str(runtime_export),
                    "program_verification": item["program_verification"],
                    "program_export_verification": export_verification,
                    "execution_mode": "compile_only",
                }
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                self._save_state()
                continue
            coverage_run_id = str(item["coverage_run_id"])
            runtime_export, coverage_export, export_verification = self._export_program_dataset(
                _work_dataset(item),
                str(item["compiler_run_id"]),
                coverage_run_id,
            )
            item["runtime_export_dir"] = str(runtime_export)
            item["coverage_export_dir"] = str(coverage_export)
            dataset = _work_dataset(item)
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                **dict(artifacts[dataset.value]),
                "coverage_run_id": coverage_run_id,
                "runtime_export_dir": str(runtime_export),
                "coverage_export_dir": str(coverage_export),
                **(
                    {"semantic_run_id": str(item["semantic_run_id"])}
                    if item.get("semantic_run_id") is not None
                    else {}
                ),
                "program_verification": item["program_verification"],
                "program_export_verification": export_verification,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

    def _export_verified_qbank_runtime_dataset(
        self,
        dataset: DatasetName,
        run_id: str,
    ) -> tuple[Path, dict[str, object]]:
        from pmmc.compiler.program_export import verify_program_export
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_export import ExperimentExporter
        from pmmc.infrastructure.object_store import create_runtime_object_store

        stage_id = f"export:{dataset.value}"
        with self._stage(stage_id) as plan_item:

            async def export_runtime() -> Path:
                engine, sessions = create_session_factory(
                    self.launch_settings, DatabaseRole.RUNTIME
                )
                try:
                    result = await ExperimentExporter(
                        sessions,
                        runtime_store=create_runtime_object_store(self.launch_settings),
                        batch_size=self.launch_settings.experiment_export_batch_size,
                    ).export(
                        run_id,
                        self.launch_settings.experiment_outputs_path,
                        include_referenced_model_artifacts=True,
                    )
                    return result.export_dir
                finally:
                    await engine.dispose()

            runtime_export = asyncio.run(export_runtime())
            verification = verify_program_export(runtime_export)
            if not bool(verification.get("valid")):
                raise ValueError("compile-only verified-QBank export is invalid")
            assert self.reporter is not None
            self.reporter.advance(
                plan_item.work_units,
                dataset=dataset.value,
                state="verified_qbank_compiled",
            )
            return runtime_export, verification

    def _reuse_completed_program_stage_major(
        self,
        *,
        already_executed: frozenset[str] = frozenset(),
        expected_snapshot_ids: dict[str, str] | None = None,
    ) -> bool:
        if not self.resume:
            return False
        expected_datasets = {item.value for item in self.config.selection.datasets}
        if set(self.state.dataset_artifacts) != expected_datasets:
            return False
        stage_ids = [
            item.stage_id
            for item in self.plan.items
            if item.stage_id
            not in {
                "preflight",
                "model_prepare",
                "infrastructure",
                "model_calibration",
                "final_report",
                "cleanup",
            }
        ]
        if not set(stage_ids) <= set(self.state.completed_stage_ids):
            return False

        from pmmc.compiler.preparation import CompilationUnitPreparationService
        from pmmc.compiler.program_export import verify_program_export
        from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
        from pmmc.evaluation.qbank_workflow import verify_qbank_coverage_run
        from pmmc.infrastructure.database.compiler_repositories import verify_compiler_run
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.export_verification import verify_experiment_export

        async def verify_run(run_id: str) -> dict[str, object]:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with sessions() as session:
                    return await verify_compiler_run(session, run_id)
            finally:
                await engine.dispose()

        for dataset in sorted(expected_datasets):
            values = self.state.dataset_artifacts[dataset]
            if expected_snapshot_ids is not None and str(
                values.get("snapshot_id")
            ) != expected_snapshot_ids.get(dataset):
                raise ValueError("resumed Planner snapshot differs from the current benchmark")
            CompilationUnitPreparationService.verify(Path(str(values["unit_preparation_dir"])))
            r0_blocked = values.get("r0_gate_passed") is False
            if r0_blocked:
                diagnostic = verify_experiment_export(Path(str(values["runtime_export_dir"])))
                expected = values.get("program_export_verification")
                if (
                    not isinstance(expected, dict)
                    or expected.get("diagnostic_export") is not True
                    or expected.get("r0_hard_gate_passed") is not False
                    or diagnostic.get("export_id") != expected.get("export_id")
                ):
                    return False
            else:
                report = asyncio.run(verify_run(str(values["compiler_run_id"])))
                if not bool(report.get("valid")):
                    return False
                verify_program_export(Path(str(values["runtime_export_dir"])))
            if self.config.evaluate:
                if r0_blocked:
                    if values.get("coverage_export_dir") is not None:
                        return False
                    continue
                coverage = asyncio.run(
                    verify_qbank_coverage_run(self.launch_settings, str(values["coverage_run_id"]))
                )
                if not bool(coverage.get("valid")):
                    return False
                coverage_export = verify_qbank_coverage_export(
                    Path(str(values["coverage_export_dir"]))
                )
                if _is_semantic_program_target(self.config.target):
                    if not bool(coverage_export.get("semantic_match_present")):
                        return False
                    if coverage_export.get("semantic_run_id") != values.get("semantic_run_id"):
                        return False

        if _is_program_pilot_target(self.config.target):
            pilot_dir = self.state.metadata.get("pilot_dir")
            if not isinstance(pilot_dir, str):
                return False
            mode = str(self.state.metadata.get("pilot_mode", "quality_pilot"))
            if mode in {"quality_pilot", "gpu_pilot"}:
                manifest_value = self.state.metadata.get("pilot_manifest")
                if (
                    isinstance(manifest_value, dict)
                    and manifest_value.get("protocol_version") == "step6d-failure-capsule-v1"
                ):
                    from pmmc.compiler.step6d_failure import (
                        verify_step6d_failure_capsule,
                    )

                    capsule_manifest = verify_step6d_failure_capsule(Path(pilot_dir))
                    if capsule_manifest.get("capsule_id") != self.state.metadata.get("pilot_id"):
                        return False
                else:
                    from pmmc.compiler.program_pilot import (
                        verify_program_pilot_bundle,
                    )

                    checked_pilot = verify_program_pilot_bundle(Path(pilot_dir))
                    if checked_pilot.pilot_id != self.state.metadata.get("pilot_id"):
                        return False
                if mode == "gpu_pilot":
                    from pmmc.smoke import (
                        verify_step6c2_semantic_conformance,
                        verify_step6d_compiler_conformance,
                    )

                    r0_blocked_run = all(
                        self.state.dataset_artifacts[dataset].get("r0_gate_passed") is False
                        for dataset in expected_datasets
                    )
                    compiler = verify_step6d_compiler_conformance(
                        Path(
                            str(
                                self.state.metadata.get(
                                    "step6d_compiler_conformance_path",
                                    "",
                                )
                            )
                        ),
                        expected_replica_count=self._replica_count("compiler"),
                    )
                    if compiler.model_dump(mode="json") != self.state.metadata.get(
                        "step6d_compiler_conformance"
                    ):
                        raise ValueError("resumed Step 6D conformance identity changed")
                    if not r0_blocked_run:
                        semantic = verify_step6c2_semantic_conformance(
                            Path(
                                str(
                                    self.state.metadata.get(
                                        "step6d_semantic_conformance_path",
                                        "",
                                    )
                                )
                            ),
                            expected_replica_count=self._replica_count("judge"),
                        )
                        if semantic.model_dump(mode="json") != self.state.metadata.get(
                            "step6d_semantic_conformance"
                        ):
                            raise ValueError(
                                "resumed Step 6D semantic conformance identity changed"
                            )
                    prefix_path = Path(
                        str(
                            self.state.metadata.get(
                                "step6d_prefix_manifest_path",
                                "",
                            )
                        )
                    )
                    if not prefix_path.is_file():
                        raise ValueError("resumed Step 6D pilot is missing its prefix manifest")
                    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
                    expected_hash = prefix.pop("content_hash", None)
                    if expected_hash != _stable_hash(prefix) or {
                        **prefix,
                        "content_hash": expected_hash,
                    } != self.state.metadata.get("step6d_prefix_manifest"):
                        raise ValueError("resumed Step 6D prefix identity changed")
            elif mode == "smoke":
                from pmmc.smoke import (
                    verify_step6c2_semantic_conformance,
                    verify_step6c2_semantic_conformance_payload,
                    verify_step6c_planner_conformance,
                    verify_step6c_planner_conformance_payload,
                )

                conformance_value = self.state.metadata.get("step6c_planner_conformance")
                conformance = verify_step6c_planner_conformance_payload(
                    conformance_value,
                    expected_replica_count=self._replica_count("compiler"),
                )
                conformance_path = Path(
                    str(self.state.metadata.get("step6c_planner_conformance_path", ""))
                )
                stored_conformance = verify_step6c_planner_conformance(
                    conformance_path,
                    expected_replica_count=self._replica_count("compiler"),
                )
                if stored_conformance.content_hash != conformance.content_hash:
                    raise ValueError("resumed Planner conformance identity changed")
                smoke_manifest = Path(pilot_dir) / "manifest.json"
                if not smoke_manifest.is_file():
                    return False
                smoke = json.loads(smoke_manifest.read_text(encoding="utf-8"))
                if smoke.get("launch_id") != self.launch_id:
                    return False
                manifest_conformance = verify_step6c_planner_conformance_payload(
                    smoke.get("planner_conformance"),
                    expected_replica_count=self._replica_count("compiler"),
                )
                if manifest_conformance.content_hash != conformance.content_hash:
                    raise ValueError("smoke manifest Planner conformance identity changed")
                if _is_semantic_program_target(self.config.target):
                    semantic_conformance = verify_step6c2_semantic_conformance_payload(
                        self.state.metadata.get("step6c2_semantic_conformance"),
                        expected_replica_count=self._replica_count("judge"),
                    )
                    stored_semantic = verify_step6c2_semantic_conformance(
                        Path(
                            str(
                                self.state.metadata.get(
                                    "step6c2_semantic_conformance_path",
                                    "",
                                )
                            )
                        ),
                        expected_replica_count=self._replica_count("judge"),
                    )
                    if stored_semantic.content_hash != semantic_conformance.content_hash:
                        raise ValueError("resumed semantic conformance identity changed")
                    manifest_semantic = verify_step6c2_semantic_conformance_payload(
                        smoke.get("semantic_conformance"),
                        expected_replica_count=self._replica_count("judge"),
                    )
                    if manifest_semantic.content_hash != semantic_conformance.content_hash:
                        raise ValueError("smoke manifest semantic conformance identity changed")
            else:
                return False
        for stage_id in stage_ids:
            if stage_id not in already_executed:
                self._reuse_stage(stage_id)
        return True

    def _prepare_step7_fallback(
        self,
        *,
        item: dict[str, object],
        snapshot_path: Path,
        selection: BenchmarkSelection,
    ) -> None:
        """Register one Native-MM child with the exact PMMC query denominator."""

        from pmmc.schemas.baseline_suite import is_internal_pmmc_fallback_suite

        dataset = _work_dataset(item)
        fallback_metadata: dict[str, object] = {
            "internal_pmmc_fallback": True,
            "pilot_projection": True,
            "source_compiler_run_id": str(item["compiler_run_id"]),
        }
        if self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
            from pmmc.runtime.query_executor_config import load_query_executor_config

            assert self.config.query_executor_config_path is not None
            answer_config = load_query_executor_config(
                _resolve(self.settings, self.config.query_executor_config_path)
            )
            fallback_metadata.update(
                {
                    "answer_contract_version": "strict-answer-contract-v2",
                    "answer_prompt_version": answer_config.answer_prompt_version,
                    "answer_output_schema_version": ANSWER_OUTPUT_SCHEMA_VERSION,
                }
            )
        contract_metadata = {
            **fallback_metadata,
            "paper_result": False,
            "selection": selection.model_dump(mode="json"),
        }
        if not is_internal_pmmc_fallback_suite(
            [BaselineMethod.NATIVE_MULTIMODAL_RAG],
            contract_metadata,
        ):
            raise ValueError("Step 7 Native-MM fallback suite contract is invalid")
        print(
            "STEP7_FALLBACK_SUITE_PREFLIGHT "
            + json.dumps(
                {
                    "dataset": dataset.value,
                    "method_count": 1,
                    "paper_result": False,
                    "partition_role": selection.metadata.get("partition_role"),
                    "selected_query_count": len(selection.query_ids or []),
                    "state": "ready",
                    "suite_mode": "internal_native_mm_component",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        restored = self.state.dataset_artifacts.get(dataset.value, {})
        corpus_path_value = restored.get("fallback_corpus_path")
        corpus_id_value = restored.get("fallback_corpus_id")
        if (
            self.resume
            and self._checkpoint_restored
            and f"corpus:{dataset.value}" in self.state.completed_stage_ids
            and isinstance(corpus_path_value, str)
            and isinstance(corpus_id_value, str)
        ):
            from pmmc.baselines import validate_corpus

            corpus_path = Path(corpus_path_value)
            if self.reporter is not None:
                self.reporter.report_operation(f"fallback_corpus_verify:{dataset.value}")
            corpus = validate_corpus(corpus_path)
            if corpus.corpus_id != corpus_id_value or corpus.snapshot_id != item["snapshot_id"]:
                raise ValueError("restored Step 7 fallback corpus identity differs")
            self._reuse_restored_stage(f"corpus:{dataset.value}")
            corpus_id = corpus.corpus_id
        else:
            corpus_path, corpus_id = self._prepare_corpus(dataset, snapshot_path)

        imported = self.state.metadata.get("upstream_import")
        imported_fallback = isinstance(imported, dict)
        answer_upgrade = (
            isinstance(imported, dict)
            and "query_executor_answer_prompt" in imported.get("changed_inputs", [])
        )
        if imported_fallback:
            assert isinstance(imported, dict)
            from pmmc.experiments.upstream_import import verify_receipt

            verify_receipt(self, self.state)
            # Execution attempts change launch; the indexed child must not change identity.
            if not answer_upgrade:
                fallback_metadata["launcher_id"] = imported["source_launch_id"]
        suite_id, selected_count = self._create_suite(
            dataset,
            str(item["snapshot_id"]),
            corpus_id,
            methods=[BaselineMethod.NATIVE_MULTIMODAL_RAG],
            selection=selection,
            paper_result=False,
            extra_metadata=fallback_metadata,
            require_existing=imported_fallback and not answer_upgrade,
        )
        if selected_count != _work_int(item, "selected_count"):
            raise ValueError("Native-MM child denominator differs from the PMMC pilot")
        from pmmc.baselines.suite import baseline_suite_child_run_id

        fallback_run_id = baseline_suite_child_run_id(
            suite_id, BaselineMethod.NATIVE_MULTIMODAL_RAG
        )
        if answer_upgrade:
            from pmmc.baselines.answer_upgrade import (
                AnswerUpgradeError,
                rebind_native_fallback,
                resolved_answer_upgrade_settings,
            )

            if self.reporter is not None:
                self.reporter.report_operation(f"fallback_answer_upgrade:{dataset.value}")
            source_suite = restored.get("fallback_retrieval_source_suite_id") or restored.get(
                "fallback_suite_id"
            )
            if not isinstance(source_suite, str) or source_suite == suite_id:
                raise AnswerUpgradeError(
                    "answer_upgrade_source_suite_not_independent",
                    phase="source_suite", dataset=dataset.value,
                )
            if self.preparation is None:
                raise AnswerUpgradeError(
                    "answer_upgrade_model_preparation_missing",
                    phase="model_registry_load", dataset=dataset.value,
                )
            # CPU migration must use the same commit-resolved registry as GPU children.
            upgrade_settings = resolved_answer_upgrade_settings(
                self.launch_settings, self.preparation.resolved_registry_path
            )
            asyncio.run(rebind_native_fallback(
                settings=upgrade_settings,
                source_suite_id=source_suite, target_suite_id=suite_id,
            ))
            restored = {**restored, "fallback_retrieval_source_suite_id": source_suite}
        elif imported_fallback:
            print("STEP7B_FALLBACK_IDENTITY_RECOVERY " + json.dumps({
                "dataset": dataset.value, "state": "verified_existing",
                "identity_scope": "source_launch", "new_suite_created": False,
                "work_item_rebound": restored.get("fallback_run_id") != fallback_run_id,
                "model_calls": 0,
            }, sort_keys=True), flush=True)
        item.update(
            {
                "fallback_corpus_id": corpus_id,
                "fallback_corpus_path": str(corpus_path),
                "fallback_suite_id": suite_id,
                "fallback_run_id": fallback_run_id,
            }
        )
        artifacts = dict(self.state.dataset_artifacts)
        artifacts[dataset.value] = {
            **dict(artifacts.get(dataset.value, {})),
            **({"fallback_retrieval_source_suite_id":
                restored["fallback_retrieval_source_suite_id"]} if answer_upgrade else {}),
            "fallback_corpus_id": corpus_id,
            "fallback_corpus_path": str(corpus_path),
            "fallback_suite_id": suite_id,
            "fallback_run_id": fallback_run_id,
        }
        self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
        self._save_state()

    def _step7_prefix_samples(
        self,
        samples: list[QuestionerPilotSample],
        snapshot_ids: Mapping[DatasetName, str],
    ) -> list[QuestionerPilotSample]:
        """Derive a stable non-scientific 2-query sample for each dataset."""

        prefix_samples: list[QuestionerPilotSample] = []
        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
        for sample in samples:
            query_ids = list(sample.query_ids[:2])
            selection = BenchmarkSelection(
                datasets=[sample.dataset],
                query_ids=query_ids,
                seed=self.config.selection.seed,
                metadata={
                    "partition_id": sample.partition_id,
                    "partition_role": sample.role.value,
                    "step7_prefix": True,
                    "step7b_diagnostic_prefix": step7b,
                },
            )
            conversations, selected_count = self._questioner_denominator(
                sample.dataset,
                snapshot_ids[sample.dataset],
                selection=selection,
            )
            if selected_count != 2:
                raise ValueError("Step 7 prefix denominator differs from two queries")
            identity = {
                "protocol": (
                    "step7b-independent-diagnostic-prefix-v2"
                    if step7b
                    else "step7-independent-real-prefix-v1"
                ),
                "partition_id": sample.partition_id,
                "dataset": sample.dataset.value,
                "query_ids": query_ids,
                "conversation_ids": conversations,
            }
            prefix_samples.append(
                QuestionerPilotSample(
                    sample_id=("step7b-prefix-" if step7b else "step7-prefix-")
                    + _stable_hash(identity)[:24],
                    partition_id=sample.partition_id,
                    dataset=sample.dataset,
                    role=sample.role,
                    query_ids=query_ids,
                    conversation_ids=conversations,
                    selection_hash=_stable_hash(identity),
                )
            )
        return prefix_samples

    def _prepare_step7_prefix_work(
        self,
        *,
        samples: list[QuestionerPilotSample],
        snapshot_dirs: Mapping[DatasetName, Path],
        snapshot_ids: Mapping[DatasetName, str],
        partition_id: str,
    ) -> list[dict[str, object]]:
        """Prepare prefix units without touching main-run dataset artifacts."""

        from pmmc.compiler.preparation import CompilationUnitPreparationService
        from pmmc.compiler.questioner_config import load_questioner_config
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.repositories import (
            PostgresMemoryRepository,
            PostgresSessionRepository,
        )

        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        questioner = load_questioner_config(self.launch_settings.resolved_questioner_config_path)
        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT

        async def load_inputs(
            snapshot_id: str,
            conversations: Sequence[str],
        ) -> tuple[list, list]:  # type: ignore[type-arg]
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                selected = set(conversations)
                return await asyncio.gather(
                    PostgresSessionRepository(sessions).list_snapshot(
                        snapshot_id, conversation_ids=selected
                    ),
                    PostgresMemoryRepository(sessions).list_snapshot(
                        snapshot_id, conversation_ids=selected
                    ),
                )
            finally:
                await engine.dispose()

        work: list[dict[str, object]] = []
        for sample in samples:
            dataset = sample.dataset
            snapshot_id = snapshot_ids[dataset]
            selection = self._pilot_selection(sample)
            conversations, selected_count = self._questioner_denominator(
                dataset,
                snapshot_id,
                selection=selection,
            )
            if selected_count != 2 or conversations != sorted(sample.conversation_ids):
                raise ValueError("Step 7 prefix selection changed during preparation")
            compiler_run_id = (
                "pmmc-step7b-prefix-" if step7b else "pmmc-step7-prefix-"
            ) + _stable_hash(
                [
                    self.launch_id,
                    partition_id,
                    sample.sample_id,
                    dataset.value,
                    snapshot_id,
                    conversations,
                ]
            )[:24]

            sessions_for_run, memories_for_run = asyncio.run(
                load_inputs(snapshot_id, conversations)
            )
            manifest, _descriptors, preparation = CompilationUnitPreparationService(
                settings=self.launch_settings,
                registry=self.resolved_registry,
                config=questioner,
                workers=self.execution_config.unit_preparation.workers,
                max_tasks_per_child=(self.execution_config.unit_preparation.max_tasks_per_child),
                cache_policy=self.execution_config.unit_preparation.cache_policy,
            ).prepare(
                run_id=compiler_run_id,
                snapshot_id=snapshot_id,
                sessions=sessions_for_run,
                memories=memories_for_run,
            )
            audit_hash = _stable_hash(
                {
                    "protocol": (
                        "step7b-independent-diagnostic-prefix-audit-v2"
                        if step7b
                        else "step7-independent-real-prefix-audit-v1"
                    ),
                    "launch_id": self.launch_id,
                    "partition_id": partition_id,
                    "sample_id": sample.sample_id,
                    "dataset": dataset.value,
                    "snapshot_id": snapshot_id,
                    "selection_hash": sample.selection_hash,
                    "conversation_ids": conversations,
                    "unit_preparation_id": preparation.name,
                    "launcher_config_hash": self.effective_config_hash,
                    "scientific_sample": False,
                }
            )
            work.append(
                {
                    "dataset": dataset.value,
                    "snapshot_id": snapshot_id,
                    "snapshot_path": snapshot_dirs[dataset],
                    "compiler_run_id": manifest.run_id,
                    "conversations": conversations,
                    "selected_count": selected_count,
                    "selection": selection,
                    "preparation": preparation,
                    "unit_count": manifest.unit_count,
                    "split": PartitionRole.DEVELOPMENT.value,
                    "sample": sample,
                    "program_pilot_audit_hash": audit_hash,
                    "diagnostic_force_revision": step7b,
                }
            )
            print(
                "STEP7_PREFIX_PREP "
                + json.dumps(
                    {
                        "conversation_count": len(conversations),
                        "dataset": dataset.value,
                        "query_count": selected_count,
                        "unit_count": manifest.unit_count,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return work

    def _verify_step7_prefix_records(
        self,
        work: list[dict[str, object]],
        samples: list[QuestionerPilotSample],
    ) -> list[dict[str, object]]:
        """Verify that each independent prefix produced real feedback and routing."""

        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.errors import (
            safe_database_error_diagnostics,
        )
        from pmmc.infrastructure.database.experiment_repositories import (
            ExperimentUnitOfWork,
        )
        from pmmc.infrastructure.database.repositories import (
            PostgresRuntimeQueryRepository,
        )
        from pmmc.schemas import DataScope

        samples_by_dataset = {sample.dataset: sample for sample in samples}
        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
        event_prefix = "STEP7B" if step7b else "STEP7"

        async def verify_item(item: dict[str, object]) -> dict[str, object]:
            dataset = _work_dataset(item)
            sample = samples_by_dataset[dataset]
            prefix_query_ids = list(sample.query_ids)
            operation = "load_runtime_queries"
            query_ordinal: int | None = None
            loaded_query_count = 0
            routed_ids: set[str] = set()
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            print(
                f"{event_prefix}_PREFIX_RECORD_VERIFY "
                + json.dumps(
                    {
                        "compiler_run_hash": _stable_hash(str(item["compiler_run_id"])),
                        "dataset": dataset.value,
                        "expected_query_count": len(prefix_query_ids),
                        "operation": operation,
                        "state": "starting",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            try:
                queries = await PostgresRuntimeQueryRepository(sessions).list_ids(
                    str(item["snapshot_id"]), prefix_query_ids
                )
                loaded_query_count = len(queries)
                async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                    assert uow.doubter is not None
                    assert uow.qbank_routing is not None
                    assert uow.memory_programs is not None
                    assert uow.program_dry_runs is not None
                    operation = "load_routing_projection"
                    projection_value = await uow.doubter.get_routing_projection(
                        str(item["compiler_run_id"])
                    )
                    operation = "load_doubter_assessments"
                    assessments = await uow.doubter.list_assessments(str(item["compiler_run_id"]))
                    revisions = await uow.doubter.list_revisions(str(item["compiler_run_id"]))
                    patches = await uow.doubter.list_revision_patches(str(item["compiler_run_id"]))
                    programs = await uow.memory_programs.list_for_run(str(item["compiler_run_id"]))
                    dry_runs = await uow.program_dry_runs.list_for_run(str(item["compiler_run_id"]))
                    if projection_value is None:
                        raise ValueError("Step 7 prefix has no verified routing projection")
                    projection, active_members = projection_value
                    active_member_ids = {value.qbank_id for value in active_members}
                    assessment_ids = {value.qbank_id for value in assessments}
                    operation = "list_verified_routing_v3" if step7b else "list_verified_routing_v2"
                    for current_query_ordinal, query in enumerate(queries):
                        query_ordinal = current_query_ordinal
                        if query.available_after_session is None:
                            raise ValueError("Step 7 prefix query has no prospective cutoff")
                        list_verified = (
                            uow.qbank_routing.list_verified_v3
                            if step7b
                            else uow.qbank_routing.list_verified_v2
                        )
                        rows = await list_verified(
                            routing_projection_id=projection.routing_projection_id,
                            compiler_run_id=str(item["compiler_run_id"]),
                            snapshot_id=str(item["snapshot_id"]),
                            dataset=dataset.value,
                            conversation_id=query.conversation_id,
                            cutoff_session_id=query.available_after_session,
                            question_budget=8,
                            round_id=2,
                        )
                        routed_ids.update(value.qbank_id for value in rows)
                if len(queries) != 2:
                    raise ValueError("Step 7 prefix query import is incomplete")
                if not assessment_ids:
                    raise ValueError("Step 7 prefix has no real Doubter feedback")
                diagnostic_revision_count = 0
                diagnostic_r1_dry_run_count = 0
                diagnostic_r1_feedback_count = 0
                if step7b:
                    forced_r0 = {
                        value.qbank_id: value
                        for value in assessments
                        if value.round_id == 0
                        and value.action.value == "revise_program"
                        and value.metadata.get("diagnostic_forced_revision") is True
                    }
                    applied_patches = {
                        value.qbank_id: value
                        for value in patches
                        if value.target_round_id == 1
                        and value.applied
                        and value.after_program_hash is not None
                        and value.after_program_hash != value.before_program_hash
                    }
                    revisions_by_qbank = {
                        value.qbank_id: value for value in revisions if value.target_round_id == 1
                    }
                    r1_programs = {
                        value.qbank_id: value for value in programs if value.round_id == 1
                    }
                    dry_plan_ids = {value.plan_id for value in dry_runs}
                    r1_feedback_ids = {
                        value.qbank_id for value in assessments if value.round_id == 1
                    }
                    diagnostic_ids = (
                        set(forced_r0)
                        & set(applied_patches)
                        & set(revisions_by_qbank)
                        & set(r1_programs)
                    )
                    diagnostic_revision_count = len(diagnostic_ids)
                    diagnostic_r1_dry_run_count = sum(
                        r1_programs[qbank_id].plan_id in dry_plan_ids for qbank_id in diagnostic_ids
                    )
                    diagnostic_r1_feedback_count = len(diagnostic_ids & r1_feedback_ids)
                    if diagnostic_revision_count == 0:
                        raise ValueError("Step 7B prefix has no applied typed r0-to-r1 revision")
                    if diagnostic_r1_dry_run_count != diagnostic_revision_count:
                        raise ValueError("Step 7B prefix revision has no complete r1 real dry-run")
                    if diagnostic_r1_feedback_count != diagnostic_revision_count:
                        raise ValueError(
                            "Step 7B prefix r1 dry-run has no terminal Doubter feedback"
                        )
                if not routed_ids or not (routed_ids & assessment_ids):
                    raise ValueError(
                        "Step 7 prefix has no active-r2 member backed by real Doubter feedback"
                    )
                leaked_routing_ids = routed_ids - active_member_ids
                if leaked_routing_ids:
                    raise Step7PrefixRecordError("step7_prefix_routing_round_leakage")
                print(
                    f"{event_prefix}_PREFIX_RECORD_VERIFY "
                    + json.dumps(
                        {
                            "dataset": dataset.value,
                            "doubter_feedback_count": len(assessment_ids),
                            "diagnostic_revision_count": diagnostic_revision_count,
                            "diagnostic_r1_dry_run_count": diagnostic_r1_dry_run_count,
                            "diagnostic_r1_feedback_count": diagnostic_r1_feedback_count,
                            "active_routing_member_count": len(active_member_ids),
                            "operation": "prefix_fact_closure",
                            "query_count": loaded_query_count,
                            "routing_member_count": len(routed_ids),
                            "state": "ready",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return {
                    "dataset": dataset.value,
                    "query_count": len(prefix_query_ids),
                    "doubter_feedback_count": len(assessment_ids),
                    "diagnostic_revision_count": diagnostic_revision_count,
                    "diagnostic_r1_dry_run_count": diagnostic_r1_dry_run_count,
                    "diagnostic_r1_feedback_count": diagnostic_r1_feedback_count,
                    "active_routing_member_count": len(active_member_ids),
                    "routing_member_count": len(routed_ids),
                    "query_ids_hash": _stable_hash(prefix_query_ids),
                    "compiler_run_id": str(item["compiler_run_id"]),
                    "compiler_run_hash": _stable_hash(str(item["compiler_run_id"])),
                }
            except Exception as exc:
                print(
                    f"{event_prefix}_PREFIX_RECORD_VERIFY_FAILURE "
                    + json.dumps(
                        {
                            "dataset": dataset.value,
                            "diagnostics": safe_database_error_diagnostics(exc),
                            "loaded_query_count": loaded_query_count,
                            "operation": operation,
                            "query_ordinal": query_ordinal,
                            "routed_member_count": len(routed_ids),
                            "state": "failed",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                raise
            finally:
                await engine.dispose()

        reports = [asyncio.run(verify_item(item)) for item in work]
        if {str(item["dataset"]) for item in reports} != {
            dataset.value for dataset in self.config.selection.datasets
        }:
            raise ValueError("Step 7 prefix did not cover both datasets")
        return reports

    def _run_step7_prefix_cpu_phase(self, item: dict[str, object], phase: str) -> None:
        diagnostic = self.launch_dir / f"prefix-cpu-diagnostic-{time.time_ns()}.jsonl"
        completed = subprocess.run(
            self._program_resume_command(item, phase),
            check=False,
            env={
                **os.environ, **self._child_environment(),
                "PMMC_MANAGED_CHILD_DIAGNOSTIC_PATH": str(diagnostic),
                "PMMC_MANAGED_CHILD_COMMAND_COUNT": "1",
                "PMMC_MANAGED_CHILD_COMMAND_ORDINAL": "1",
            },
        )
        if completed.returncode == 0:
            return
        diagnostics: object = {}
        if diagnostic.is_file() and diagnostic.stat().st_size <= 65536:
            for line in diagnostic.read_text(encoding="utf-8").splitlines()[-8:]:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("diagnostics"), dict):
                    diagnostics = record["diagnostics"]
        error = Step7PrefixChildError(phase=phase, diagnostics=diagnostics)
        print("STEP7_PREFIX_CPU_FAILURE " + json.dumps({
            "phase": phase, "dataset": item["dataset"],
            "returncode": completed.returncode,
            **error.database_diagnostics,
        }, sort_keys=True, separators=(",", ":")), flush=True)
        raise error

    def _run_step7_independent_prefix(
        self,
        *,
        samples: list[QuestionerPilotSample],
        snapshot_dirs: Mapping[DatasetName, Path],
        snapshot_ids: Mapping[DatasetName, str],
        partition_id: str,
    ) -> None:
        """Run an independent 2+2 Q/P/R0/D prefix before the scientific pilot."""

        from pmmc.compiler.preparation import CompilationUnitPreparationService
        from pmmc.runtime.manager import run_managed_commands
        from pmmc.smoke import (
            Step6DCompilerConformanceResult,
            verify_step6d_compiler_conformance,
        )

        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
        event_prefix = "STEP7B" if step7b else "STEP7"
        metadata_key = "step7b_prefix" if step7b else "step7_prefix"
        metadata_path_key = "step7b_prefix_path" if step7b else "step7_prefix_path"
        protocol_version = (
            "step7b-independent-diagnostic-prefix-v2"
            if step7b
            else "step7-independent-real-prefix-v1"
        )
        prefix_samples = self._step7_prefix_samples(samples, snapshot_ids)
        stored = self.state.metadata.get(metadata_key)
        if (
            self.resume
            and self._checkpoint_restored
            and "step7_prefix" in self.state.completed_stage_ids
        ):
            if not isinstance(stored, dict):
                raise ValueError("restored Step 7 prefix metadata is missing")
            rows = stored.get("datasets")
            if not isinstance(rows, list) or len(rows) != len(prefix_samples):
                raise ValueError("restored Step 7 prefix dataset accounting differs")
            work: list[dict[str, object]] = []
            rows_by_dataset = {
                str(row.get("dataset")): row for row in rows if isinstance(row, dict)
            }
            for sample in prefix_samples:
                row = rows_by_dataset.get(sample.dataset.value)
                if row is None:
                    raise ValueError("restored Step 7 prefix is missing a dataset")
                if row.get("query_ids_hash") != _stable_hash(sample.query_ids):
                    raise ValueError("restored Step 7 prefix query identity changed")
                preparation = Path(str(row.get("unit_preparation_dir", "")))
                manifest, _descriptors = CompilationUnitPreparationService.verify(preparation)
                if manifest.snapshot_id != snapshot_ids[sample.dataset]:
                    raise ValueError("restored Step 7 prefix snapshot changed")
                work.append(
                    {
                        "dataset": sample.dataset.value,
                        "snapshot_id": manifest.snapshot_id,
                        "compiler_run_id": str(row["compiler_run_id"]),
                        "sample": sample,
                    }
                )
            reports = self._verify_step7_prefix_records(work, prefix_samples)
            reports_hash = _stable_hash(reports)
            if reports_hash != stored.get("reports_hash"):
                compatibility = (
                    _step7_prefix_exact_round_report_upgrade(rows, reports)
                    if stored.get("protocol_version") == "step7-independent-real-prefix-v1"
                    else None
                )
                if compatibility is None:
                    print(
                        "STEP7_PREFIX_RESTORE_COMPATIBILITY "
                        + json.dumps(
                            {
                                "error_code": "step7_prefix_restored_fact_mismatch",
                                "mode": "exact_round_derived_report_upgrade",
                                "state": "rejected",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    raise Step7PrefixRecordError("step7_prefix_restored_fact_mismatch")
                upgraded_rows: list[dict[str, object]] = []
                for report in reports:
                    upgraded = dict(report)
                    previous = rows_by_dataset[str(report["dataset"])]
                    upgraded["unit_preparation_dir"] = str(previous["unit_preparation_dir"])
                    upgraded_rows.append(upgraded)
                upgraded_payload = dict(stored)
                upgraded_payload.update(
                    {
                        "datasets": upgraded_rows,
                        "reports_hash": reports_hash,
                        "routing_verification_revision": "20260727_0021",
                    }
                )
                upgraded_payload.pop("content_hash", None)
                upgraded_payload["content_hash"] = _stable_hash(upgraded_payload)
                path = self.launch_dir / (
                    "step7b_prefix_verification.json"
                    if step7b
                    else "step7_prefix_verification.json"
                )
                _atomic_json(path, upgraded_payload)
                metadata = dict(self.state.metadata)
                metadata.update(
                    {
                        metadata_path_key: str(path),
                        metadata_key: upgraded_payload,
                    }
                )
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()
                print(
                    f"{event_prefix}_PREFIX_RESTORE_COMPATIBILITY "
                    + json.dumps(
                        {
                            **compatibility,
                            "mode": "exact_round_derived_report_upgrade",
                            "routing_verification_revision": "20260727_0021",
                            "state": "updated",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            if not self._reuse_restored_stage("step7_prefix"):
                raise RuntimeError("verified Step 7 prefix was not reusable")
            return

        with self._stage("step7_prefix") as plan_item:
            if plan_item.work_units != 4:
                raise ValueError("Step 7 prefix progress requires four fixed milestones")
            prefix_completed = 0

            def advance_prefix(milestone: str, *, next_stage: str) -> None:
                nonlocal prefix_completed
                prefix_completed += 1
                assert self.reporter is not None
                self.reporter.advance(
                    1, phase=milestone, state="ready", progress_unit="milestone"
                )
                print(
                    f"{event_prefix}_PREFIX_PROGRESS "
                    + json.dumps(
                        {
                            "completed": prefix_completed,
                            "milestone": milestone,
                            "next_stage": next_stage,
                            "state": "ready",
                            "total": plan_item.work_units,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

            work = self._prepare_step7_prefix_work(
                samples=prefix_samples,
                snapshot_dirs=snapshot_dirs,
                snapshot_ids=snapshot_ids,
                partition_id=partition_id,
            )
            if self.config.planner_config_path is None:
                raise ValueError("Step 7 prefix has no Planner config")
            if self.config.doubter_config_path is None:
                raise ValueError("Step 7 prefix has no Doubter config")
            if self.config.compiler_foundation_config_path is None:
                raise ValueError("Step 7 prefix has no compiler foundation config")

            conformance_path = self.launch_dir / (
                "step7b_compiler_conformance.json" if step7b else "step7_compiler_conformance.json"
            )
            compiler_conformance_command = [
                sys.executable,
                "-m",
                "pmmc",
                "smoke",
                "verify-step6d-compiler-conformance",
                "--questioner-config",
                str(self.launch_settings.resolved_questioner_config_path),
                "--planner-config",
                str(_resolve(self.settings, self.config.planner_config_path)),
                "--doubter-config",
                str(_resolve(self.settings, self.config.doubter_config_path)),
                "--foundation-config",
                str(
                    _resolve(
                        self.settings,
                        self.config.compiler_foundation_config_path,
                    )
                ),
                "--expected-replica-count",
                str(self._replica_count("compiler")),
                "--result-json",
                str(conformance_path),
                "--compact-output",
            ]

            def persist_compiler_conformance() -> Step6DCompilerConformanceResult:
                verified = verify_step6d_compiler_conformance(
                    conformance_path,
                    expected_replica_count=self._replica_count("compiler"),
                )
                conformance_metadata = dict(self.state.metadata)
                conformance_metadata.update(
                    {
                        (
                            "step7b_compiler_conformance_path"
                            if step7b
                            else "step6d_compiler_conformance_path"
                        ): str(conformance_path),
                        (
                            "step7b_compiler_conformance"
                            if step7b
                            else "step6d_compiler_conformance"
                        ): verified.model_dump(mode="json"),
                    }
                )
                self.state = self.state.model_copy(update={"metadata": conformance_metadata})
                self._save_state()
                return verified

            def seal_qp_checkpoint_source_identity() -> None:
                from pmmc.experiments.stage_components import source_fingerprint

                metadata = dict(self.state.metadata)
                metadata[_STEP7_PREFIX_QP_SOURCE_FINGERPRINT_KEY] = source_fingerprint(
                    self.settings.project_root,
                    "compiler_program",
                )
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()

            conformance: Step6DCompilerConformanceResult | None = None
            planner_preflight = [
                {
                    "dataset": str(item["dataset"]),
                    **self._load_step7_planner_preflight(str(item["compiler_run_id"])),
                }
                for item in work
            ]
            planner_issue_codes = {
                str(row["dataset"]): _step7_planner_preflight_issue_codes(row)
                for row in planner_preflight
            }
            compiler_qp_reused = not any(planner_issue_codes.values())
            print(
                f"{event_prefix}_PREFIX_COMPILER_PREFLIGHT "
                + json.dumps(
                    {
                        "dataset_count": len(planner_preflight),
                        "deferred_conformance_to": ("doubter" if compiler_qp_reused else None),
                        "reason_codes": {
                            dataset: issues
                            for dataset, issues in planner_issue_codes.items()
                            if issues
                        },
                        "state": (
                            "reused_qp_facts" if compiler_qp_reused else "model_stage_required"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not compiler_qp_reused:
                compiler_commands = [
                    compiler_conformance_command,
                    *(self._program_questioner_command(item) for item in work),
                    *(self._program_resume_command(item, "planner") for item in work),
                ]
                run_managed_commands(
                    self.launch_settings,
                    profile_path=self.launch_settings.resolved_runtime_profile_path,
                    logical_stage="compiler",
                    commands=compiler_commands,
                    environment_overrides=self._child_environment(),
                    replica_count=self._replica_count("compiler"),
                )
                conformance = persist_compiler_conformance()
                planner_preflight = [
                    {
                        "dataset": str(item["dataset"]),
                        **self._load_step7_planner_preflight(str(item["compiler_run_id"])),
                    }
                    for item in work
                ]
                planner_issue_codes = {
                    str(row["dataset"]): _step7_planner_preflight_issue_codes(row)
                    for row in planner_preflight
                }
            planner_blocked = {
                dataset: issues for dataset, issues in planner_issue_codes.items() if issues
            }
            print(
                f"{event_prefix}_PREFIX_PLANNER_PREFLIGHT "
                + json.dumps(
                    {
                        "blocking_reason_codes": planner_blocked,
                        "datasets": planner_preflight,
                        "next_stages": [
                            "embedding_text",
                            "embedding_multimodal",
                            "dry_run_r0",
                            "doubter",
                        ],
                        "state": "blocked" if planner_blocked else "ready",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if planner_blocked:
                seal_qp_checkpoint_source_identity()
                checkpoint_before = self.state.metadata.get("latest_checkpoint")
                self._maybe_checkpoint("step7_prefix:planner_failure")
                checkpoint_after = self.state.metadata.get("latest_checkpoint")
                print(
                    f"{event_prefix}_PREFIX_PLANNER_FAILURE "
                    + json.dumps(
                        {
                            "blocking_reason_codes": planner_blocked,
                            "checkpoint_created": bool(
                                isinstance(checkpoint_after, str)
                                and checkpoint_after
                                and checkpoint_after != checkpoint_before
                            ),
                            "checkpoint_sequence": (
                                Path(checkpoint_after).name
                                if isinstance(checkpoint_after, str) and checkpoint_after
                                else None
                            ),
                            "completed_dataset_count": len(planner_preflight)
                            - len(planner_blocked),
                            "next_action": "fix_planner_contract_then_resume",
                            "pending_stage_count": 4,
                            "source_identity_recorded": True,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                raise RuntimeError("Step 7 prefix Planner aggregate accounting preflight failed")
            seal_qp_checkpoint_source_identity()
            checkpoint_before = self.state.metadata.get("latest_checkpoint")
            qp_checkpoint_reused = bool(
                compiler_qp_reused
                and self._checkpoint_restored
                and isinstance(checkpoint_before, str)
                and (Path(checkpoint_before) / "manifest.json").is_file()
            )
            if not qp_checkpoint_reused:
                self._maybe_checkpoint("step7_prefix:questioner_planner")
            checkpoint_after = self.state.metadata.get("latest_checkpoint")
            qp_checkpoint_created = bool(
                isinstance(checkpoint_after, str)
                and checkpoint_after
                and checkpoint_after != checkpoint_before
            )
            print(
                f"{event_prefix}_PREFIX_QP_MILESTONE "
                + json.dumps(
                    {
                        "checkpoint_created": qp_checkpoint_created,
                        "checkpoint_reused": qp_checkpoint_reused,
                        "checkpoint_sequence": (
                            Path(checkpoint_after).name
                            if isinstance(checkpoint_after, str) and checkpoint_after
                            else None
                        ),
                        "dataset_count": len(planner_preflight),
                        "next_stage": "embedding_text",
                        "source_identity_recorded": True,
                        "state": (
                            "ready"
                            if qp_checkpoint_created or qp_checkpoint_reused
                            else "blocked"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not qp_checkpoint_created and not qp_checkpoint_reused:
                raise RuntimeError(
                    "Step 7 prefix Questioner/Planner facts were not captured by a durable "
                    "checkpoint"
                )
            advance_prefix("questioner_planner", next_stage="r0")

            existing_r0_manifests = [
                self._find_r0_manifest(str(item["compiler_run_id"])) for item in work
            ]
            reuse_r0 = all(manifest is not None for manifest in existing_r0_manifests)
            if reuse_r0:
                for item, r0_manifest_value in zip(work, existing_r0_manifests, strict=True):
                    assert r0_manifest_value is not None
                    item["r0_manifest"] = r0_manifest_value
                print(
                    f"{event_prefix}_PREFIX_R0_REUSE "
                    + json.dumps(
                        {
                            "dataset_count": len(work),
                            "reason": "verified_manifest_present",
                            "source_validation": "r0_input_recompute",
                            "skipped_model_stages": [
                                "embedding_text",
                                "embedding_multimodal",
                            ],
                            "state": "ready",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

            for channel, runtime_stage, channel_phases in (
                (
                    "text",
                    "embedding_text",
                    ("qbank_index_text", "memory_index_text", "program_query_text"),
                ),
                (
                    "multimodal",
                    "embedding_multimodal",
                    (
                        "qbank_index_multimodal",
                        "memory_index_multimodal",
                        "program_query_multimodal",
                    ),
                ),
            ):
                if reuse_r0:
                    print(
                        f"{event_prefix}_PREFIX_MODEL_STAGE "
                        + json.dumps(
                            {
                                "channel": channel,
                                "reason": "verified_r0_manifest_present",
                                "replica_count": self._replica_count(runtime_stage),
                                "startup_count": 0,
                                "state": "reused",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    continue
                commands = [
                    self._program_resume_command(item, phase)
                    for item in work
                    for phase in channel_phases
                ]
                print(
                    f"{event_prefix}_PREFIX_MODEL_STAGE "
                    + json.dumps(
                        {
                            "channel": channel,
                            "command_count": len(commands),
                            "replica_count": self._replica_count(runtime_stage),
                            "state": "starting",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                run_managed_commands(
                    self.launch_settings,
                    profile_path=self.launch_settings.resolved_runtime_profile_path,
                    logical_stage=runtime_stage,
                    commands=commands,
                    environment_overrides=self._child_environment(),
                    replica_count=self._replica_count(runtime_stage),
                )

            r0_preflight: list[dict[str, object]] = []
            for item in work:
                r0_phases: tuple[str, ...] = ("r0_input",) if reuse_r0 else ("dry_run", "r0_input")
                for phase in r0_phases:
                    self._run_step7_prefix_cpu_phase(item, phase)
                # Reload after r0_input compares the immutable manifest with its facts.
                item["r0_manifest"] = self._load_r0_manifest(str(item["compiler_run_id"]))
                r0_manifest = _work_mapping(item, "r0_manifest")
                r0_preflight.append(
                    _step7_r0_preflight_record(
                        dataset=str(item["dataset"]),
                        manifest=r0_manifest,
                    )
                )

            r0_issue_codes = {
                str(row["dataset"]): _step7_r0_preflight_issue_codes(row) for row in r0_preflight
            }
            r0_blocked = {dataset: issues for dataset, issues in r0_issue_codes.items() if issues}
            print(
                f"{event_prefix}_PREFIX_R0_PREFLIGHT "
                + json.dumps(
                    {
                        "blocking_reason_codes": r0_blocked,
                        "datasets": r0_preflight,
                        "next_stage": "doubter",
                        "state": "blocked" if r0_blocked else "ready",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if r0_blocked:
                print(
                    f"{event_prefix}_PREFIX_R0_FAILURE "
                    + json.dumps(
                        {
                            "blocking_reason_codes": r0_blocked,
                            "reused_manifest_count": (len(r0_preflight) if reuse_r0 else 0),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                raise RuntimeError(
                    "Step 7 prefix R0 facts are not viable for real Doubter execution"
                )

            checkpoint_before = self.state.metadata.get("latest_checkpoint")
            checkpoint_reused = bool(
                reuse_r0
                and self._checkpoint_restored
                and isinstance(checkpoint_before, str)
                and (Path(checkpoint_before) / "manifest.json").is_file()
            )
            if not checkpoint_reused:
                self._maybe_checkpoint("step7_prefix:r0")
            checkpoint_after = self.state.metadata.get("latest_checkpoint")
            checkpoint_created = bool(
                isinstance(checkpoint_after, str)
                and checkpoint_after
                and checkpoint_after != checkpoint_before
            )
            checkpoint_ready = checkpoint_created or checkpoint_reused
            print(
                f"{event_prefix}_PREFIX_R0_MILESTONE "
                + json.dumps(
                    {
                        "checkpoint_created": checkpoint_created,
                        "checkpoint_reused": checkpoint_reused,
                        "checkpoint_sequence": (
                            Path(checkpoint_after).name
                            if isinstance(checkpoint_after, str) and checkpoint_after
                            else None
                        ),
                        "dataset_count": len(r0_preflight),
                        "next_stage": "doubter",
                        "state": "ready" if checkpoint_ready else "blocked",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not checkpoint_ready:
                raise RuntimeError(
                    "Step 7 prefix R0 facts were not captured by a durable checkpoint"
                )
            advance_prefix("r0", next_stage="doubter")

            doubter_preflight = [
                {"dataset": str(item["dataset"]),
                 **self._load_step7_doubter_preflight(str(item["compiler_run_id"]))}
                for item in work
            ] if self._checkpoint_restored and reuse_r0 else []
            doubter_reused = _step7_doubter_facts_reusable(doubter_preflight)
            if doubter_reused and compiler_qp_reused:
                # Reuse only the hash-checked conformance saved in this restored launch.
                recorded_conformance = self.state.metadata.get(
                    "step7b_compiler_conformance" if step7b else "step6d_compiler_conformance"
                )
                if isinstance(recorded_conformance, dict) and conformance_path.is_file():
                    conformance = verify_step6d_compiler_conformance(
                        conformance_path,
                        expected_replica_count=self._replica_count("compiler"),
                    )
                    doubter_reused = (
                        conformance.content_hash == recorded_conformance.get("content_hash")
                    )
                else:
                    doubter_reused = False
            print(
                f"{event_prefix}_PREFIX_DOUBTER_REUSE_PREFLIGHT "
                + json.dumps({
                    "datasets": doubter_preflight,
                    "state": "reused" if doubter_reused else "model_stage_required",
                    "startup_count": 0 if doubter_reused else self._replica_count("compiler"),
                    "startup_count_kind": "planned",
                    "next_stages": ["decision", "finalize", "record_verification"],
                }, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            doubter_commands: list[list[str]] = []
            if compiler_qp_reused:
                doubter_commands.append(compiler_conformance_command)
            doubter_commands.extend(self._program_resume_command(item, "doubter") for item in work)
            try:
                if not doubter_reused:
                    run_managed_commands(
                        self.launch_settings,
                        profile_path=self.launch_settings.resolved_runtime_profile_path,
                        logical_stage="compiler",
                        commands=doubter_commands,
                        environment_overrides=self._child_environment(),
                        replica_count=self._replica_count("compiler"),
                    )
            except Exception:
                checkpoint_before = self.state.metadata.get("latest_checkpoint")
                self._maybe_checkpoint("step7_prefix:doubter_failure")
                checkpoint_after = self.state.metadata.get("latest_checkpoint")
                print(
                    f"{event_prefix}_PREFIX_DOUBTER_FAILURE_MILESTONE "
                    + json.dumps(
                        {
                            "checkpoint_created": bool(
                                isinstance(checkpoint_after, str)
                                and checkpoint_after
                                and checkpoint_after != checkpoint_before
                            ),
                            "checkpoint_sequence": (
                                Path(checkpoint_after).name
                                if isinstance(checkpoint_after, str) and checkpoint_after
                                else None
                            ),
                            "next_action": "inspect_safe_diagnostics_or_update_doubter_contract",
                            "state": "blocked",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                raise
            if compiler_qp_reused and not doubter_reused:
                conformance = persist_compiler_conformance()

            checkpoint_before = self.state.metadata.get("latest_checkpoint")
            if not doubter_reused:
                self._maybe_checkpoint("step7_prefix:doubter")
            checkpoint_after = self.state.metadata.get("latest_checkpoint")
            checkpoint_created = bool(
                isinstance(checkpoint_after, str)
                and checkpoint_after
                and checkpoint_after != checkpoint_before
            )
            print(
                f"{event_prefix}_PREFIX_DOUBTER_MILESTONE "
                + json.dumps(
                    {
                        "checkpoint_created": checkpoint_created,
                        "checkpoint_reused": doubter_reused,
                        "checkpoint_sequence": (
                            Path(checkpoint_after).name
                            if isinstance(checkpoint_after, str) and checkpoint_after
                            else None
                        ),
                        "dataset_count": len(work),
                        "next_stage": "decision",
                        "state": "ready" if checkpoint_created or doubter_reused else "blocked",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if not checkpoint_created and not doubter_reused:
                raise RuntimeError(
                    "Step 7 prefix Doubter facts were not captured by a durable checkpoint"
                )
            advance_prefix("doubter", next_stage="decision")

            for phase in ("decision", "finalize"):
                for item in work:
                    self._run_step7_prefix_cpu_phase(item, phase)
            reports = self._verify_step7_prefix_records(work, prefix_samples)
            if conformance is None:
                raise RuntimeError("Step 7 prefix compiler conformance is unavailable")
            payload: dict[str, object] = {
                "protocol_version": protocol_version,
                "launch_id": self.launch_id,
                "scientific_sample": False,
                "query_count_per_dataset": 2,
                "datasets": reports,
                "reports_hash": _stable_hash(reports),
                "compiler_conformance_hash": conformance.content_hash,
            }
            payload_datasets = payload["datasets"]
            assert isinstance(payload_datasets, list)
            for row, item in zip(payload_datasets, work, strict=True):
                assert isinstance(row, dict)
                row["unit_preparation_dir"] = str(item["preparation"])
            payload["content_hash"] = _stable_hash(payload)
            path = self.launch_dir / (
                "step7b_prefix_verification.json" if step7b else "step7_prefix_verification.json"
            )
            _atomic_json(path, payload)
            metadata = dict(self.state.metadata)
            metadata.update(
                {
                    metadata_path_key: str(path),
                    metadata_key: payload,
                }
            )
            self.state = self.state.model_copy(update={"metadata": metadata})
            self._save_state()
            advance_prefix("record_verification", next_stage="main_pilot")
            print(
                f"{event_prefix}_PREFIX "
                + json.dumps(
                    {
                        "dataset_count": len(reports),
                        "query_count_per_dataset": 2,
                        "scientific_sample": False,
                        "state": "independent_real_facts_verified",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    def _run_step7_pilot(self) -> None:
        """Run the frozen development compiler and online executor pilot."""

        if self.execution_config.schedule is not ExecutionSchedule.MODEL_STAGE_MAJOR_V1:
            raise ValueError("Step 7 requires model-stage-major execution")
        if self.config.questioner_pilot_config_path is None:
            raise RuntimeError("Step 7 pilot sampling config is unavailable")

        from pmmc.pilots.config import load_questioner_pilot_config
        from pmmc.pilots.partition import (
            build_partition,
            load_partition,
            sample_development_queries,
            verify_partition,
        )
        from pmmc.schemas import BenchmarkPartitionSpec

        pilot_config = load_questioner_pilot_config(
            _resolve(self.settings, self.config.questioner_pilot_config_path)
        )
        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
        expected_queries = 16 if step7b else 12
        expected_scenario_quota = 4 if step7b else 3
        expected_pilot_protocol = (
            "questioner-quality-pilot-v4" if step7b else "questioner-quality-pilot-v3"
        )
        if (
            pilot_config.protocol_version != expected_pilot_protocol
            or pilot_config.queries_per_dataset != expected_queries
        ):
            raise ValueError(
                f"{self.config.target.value} freezes {expected_queries} queries per dataset"
            )

        snapshot_dirs: dict[DatasetName, Path] = {}
        snapshot_ids: dict[DatasetName, str] = {}
        for dataset in self.config.selection.datasets:
            snapshot_dir, snapshot_id = self._prepare_snapshot(dataset)
            snapshot_dirs[dataset] = snapshot_dir
            snapshot_ids[dataset] = snapshot_id

        with self._stage("partition") as plan_item:
            configured = self.config.selection.partition_manifest_path
            if configured is None:
                partition_dir, partition_manifest = build_partition(
                    snapshot_dirs=[
                        snapshot_dirs[value]
                        for value in sorted(snapshot_dirs, key=lambda entry: entry.value)
                    ],
                    output_root=self.settings.artifacts_path / "pilots" / "partitions",
                    spec=BenchmarkPartitionSpec(
                        protocol_version=pilot_config.partition_protocol,
                        fold_count=pilot_config.fold_count,
                        development_fold=pilot_config.development_fold,
                        seed=pilot_config.partition_seed,
                    ),
                )
            else:
                partition_dir = _resolve(self.settings, configured)
                partition_manifest = verify_partition(partition_dir)
            if {
                item.dataset: item.snapshot_id for item in partition_manifest.datasets
            } != snapshot_ids:
                raise ValueError("Step 7 partition does not match current snapshots")
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)

        with self._stage("pilot_sample") as plan_item:
            _partition, assignments = load_partition(partition_dir)
            samples = sample_development_queries(
                manifest=partition_manifest,
                assignments=assignments,
                config=pilot_config,
            )
            if {sample.dataset for sample in samples} != set(self.config.selection.datasets):
                raise ValueError("Step 7 sample does not contain both datasets")
            for sample in samples:
                if len(sample.query_ids) != expected_queries:
                    raise ValueError("Step 7 sample denominator differs from its protocol")
                if sample.dataset is DatasetName.MEMGALLERY and (
                    len(sample.scenario_counts) != 4
                    or set(sample.scenario_counts.values()) != {expected_scenario_quota}
                ):
                    raise ValueError(
                        "Step 7 MemGallery sample does not satisfy the frozen scenario quota"
                    )
                if step7b and sample.dataset is DatasetName.MEMLENS:
                    category_counts: dict[str, int] = {}
                    for stratum, count in sample.stratum_counts.items():
                        category = stratum.split("|", 1)[0]
                        category_counts[category] = category_counts.get(category, 0) + count
                    if len(category_counts) != 4 or set(category_counts.values()) != {4}:
                        raise ValueError(
                            "Step 7B MEMLENS sample must contain four 4-query categories"
                        )
            _atomic_json(
                self.launch_dir / "pilot_samples.json",
                [item.model_dump(mode="json") for item in samples],
            )
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)

        self._run_step7_independent_prefix(
            samples=samples,
            snapshot_dirs=snapshot_dirs,
            snapshot_ids=snapshot_ids,
            partition_id=partition_manifest.partition_id,
        )

        samples_by_dataset = {item.dataset: item for item in samples}
        work: list[dict[str, object]] = []
        for dataset in self.config.selection.datasets:
            sample = samples_by_dataset[dataset]
            selection = self._pilot_selection(sample)
            conversations, selected_count = self._questioner_denominator(
                dataset, snapshot_ids[dataset], selection=selection
            )
            if selected_count != expected_queries or conversations != sorted(
                sample.conversation_ids
            ):
                raise ValueError("Step 7 sample differs from the imported denominator")
            compiler_run_id = (
                "pmmc-step7b-compiler-" if step7b else "pmmc-step7-compiler-"
            ) + _stable_hash(
                [
                    self.launch_id,
                    partition_manifest.partition_id,
                    sample.sample_id,
                    dataset.value,
                    snapshot_ids[dataset],
                    conversations,
                ]
            )[:24]
            item = self._program_work_item(
                dataset=dataset,
                snapshot_id=snapshot_ids[dataset],
                snapshot_path=snapshot_dirs[dataset],
                selection=selection,
                conversations=conversations,
                selected_count=selected_count,
                compiler_run_id=compiler_run_id,
                split=PartitionRole.DEVELOPMENT.value,
                sample=sample,
                partition_id=partition_manifest.partition_id,
            )
            self._prepare_step7_fallback(
                item=item,
                snapshot_path=snapshot_dirs[dataset],
                selection=selection,
            )
            work.append(item)

        if step7b:
            self._execute_program_work(
                work,
                run_semantic_judge=False,
                upstream_only=True,
            )
        else:
            self._execute_program_work(work, run_semantic_judge=False)
        blocked = [item for item in work if item.get("r0_gate_passed") is not True]
        if blocked:
            raise RuntimeError(
                "Step 7 real R0 gate failed for: "
                + ",".join(sorted(str(item["dataset"]) for item in blocked))
            )
        if step7b:
            self._run_step7b_post_index(work, samples)
        else:
            self._run_step7_answer_stage(work, samples)
            self._run_step7_evaluation_stages(work)
        self._finalize_step7_pilot(
            work=work,
            samples=samples,
            partition_id=partition_manifest.partition_id,
        )

    def _run_step7b_post_index(
        self, work: list[dict[str, object]], samples: list[QuestionerPilotSample],
    ) -> None:
        from pmmc.runtime.manager import managed_runtime_command_session
        from pmmc.runtime.query_executor_config import load_query_executor_config
        from pmmc.runtime.query_workflow import (
            QueryExecutorParentContractError,
            preflight_query_coverage_vectors,
        )

        if self.config.query_executor_config_path is None:
            raise RuntimeError("Step 7B query executor config is unavailable")
        query_config = load_query_executor_config(
            _resolve(self.settings, self.config.query_executor_config_path),
        )
        samples_by_dataset = {sample.dataset: sample for sample in samples}
        if self.reporter is not None:
            self.reporter.report_operation("query_vector_preflight")
        for item in work:
            dataset = _work_dataset(item)
            try:
                asyncio.run(preflight_query_coverage_vectors(
                    settings=self.launch_settings, config=query_config,
                    coverage_run_id=str(item["coverage_run_id"]),
                    source_compiler_run_id=str(item["compiler_run_id"]),
                    snapshot_id=str(item["snapshot_id"]), dataset=dataset,
                    query_ids=samples_by_dataset[dataset].query_ids,
                ))
            except QueryExecutorParentContractError as exc:
                exc.diagnostics.update({
                    "preflight_phase": "before_post_index_chat", "dataset": dataset.value,
                })
                raise

        from pmmc.baselines.suite_recovery import preflight_native_fallback_retrieval

        if self.reporter is not None:
            self.reporter.report_operation("fallback_retrieval_preflight")
        for item in work:
            asyncio.run(preflight_native_fallback_retrieval(
                settings=self.launch_settings,
                suite_id=str(item["fallback_suite_id"]),
                expected_run_id=str(item["fallback_run_id"]),
            ))

        cpu_recovery = (
            self.resume and self._checkpoint_restored
            and "doubter:model-stage" in self.state.completed_stage_ids
        )
        if cpu_recovery:
            print(
                "STEP7B_POST_INDEX_CPU_RECOVERY state=starting "
                "doubter=compatible_checkpoint gpu_startup=deferred "
                "next=program_verification_coverage_export",
                flush=True,
            )
            self._execute_program_work(work, run_semantic_judge=False, post_r0_only=True)
            print("STEP7B_POST_INDEX_CPU_RECOVERY state=ready", flush=True)

        terminal_stages = {
            "answer:model-stage", "evaluation_deterministic:model-stage",
            "qbank_semantic:model-stage", "evaluation_judge:model-stage",
        }
        terminal_recovery = cpu_recovery and terminal_stages <= set(
            self.state.completed_stage_ids
        )
        if terminal_recovery:
            self._run_step7_answer_stage(work, samples, require_terminal_reuse=True)
            self._run_step7_evaluation_stages(work, require_terminal_reuse=True)
            print("STEP7B_POST_INDEX_SHARED_STAGE " + json.dumps({
                "state": "terminal_facts_reused", "physical_startup_count": 0,
                "pending_model_commands": 0, "facts_verified": True,
            }, sort_keys=True, separators=(",", ":")), flush=True)
            return

        if self.reporter is not None:
            self.reporter.report_operation("model_startup:post_index_chat")
        print(
            "STEP7B_POST_INDEX_SHARED_STAGE "
            + json.dumps(
                {
                    "logical_roles": [
                        "doubter", "planner_revision", "reranker", "answer",
                        "semantic_judge", "answer_judge",
                    ],
                    "physical_model_alias": self.config.models.compiler_model,
                    "replica_count": self._replica_count("compiler"),
                    "state": "starting_once",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        with managed_runtime_command_session(
            self.launch_settings,
            profile_path=self.launch_settings.resolved_runtime_profile_path,
            logical_stage="compiler",
            environment_overrides=self._child_environment(),
            replica_count=self._replica_count("compiler"),
        ) as shared_session:
            if not cpu_recovery:
                self._execute_program_work(
                    work, run_semantic_judge=False, post_r0_only=True,
                )
            self._run_step7_answer_stage(work, samples)
            self._run_step7_evaluation_stages(work)
            print(
                "STEP7B_POST_INDEX_SHARED_STAGE "
                + json.dumps(
                    {
                        "command_batch_count": shared_session.batch_count,
                        "runtime_directory_id": shared_session.run_dir.name,
                        "state": "completed",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    def _run_step7_online(self) -> None:
        """Execute an approved Verified QBank against a frozen query denominator."""

        from pmmc.benchmarks.selection import select_queries, selection_hash
        from pmmc.compiler.program_pilot import (
            verify_program_freeze,
            verify_verified_qbank_freeze,
        )
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.repositories import (
            PostgresRuntimeQueryRepository,
        )
        from pmmc.pilots.partition import verify_partition
        from pmmc.runtime.query_executor_config import (
            apply_approved_routing_thresholds,
            load_query_executor_config,
            query_executor_protocol_hash,
            query_executor_routing_policy_hash,
        )
        from pmmc.runtime.query_workflow import create_query_executor_manifest
        from pmmc.runtime.step7_pilot import (
            _verify_questioner_freeze,
            verify_query_executor_freeze,
            verify_step7b_freeze,
        )

        if self.execution_config.schedule is not ExecutionSchedule.MODEL_STAGE_MAJOR_V1:
            raise ValueError("Step 7 online execution requires model-stage-major scheduling")
        freeze_paths = {
            "questioner": self.config.questioner_freeze_manifest_path,
            "program": self.config.compiler_program_freeze_manifest_path,
            "qbank": self.config.verified_qbank_freeze_manifest_path,
            "executor": self.config.query_executor_freeze_manifest_path,
            "step7b": self.config.step7b_freeze_manifest_path,
        }
        if any(path is None for path in freeze_paths.values()):
            raise RuntimeError("Step 7B online execution is missing an approved freeze")
        questioner_path = freeze_paths["questioner"]
        program_path = freeze_paths["program"]
        qbank_path = freeze_paths["qbank"]
        executor_path = freeze_paths["executor"]
        step7b_path = freeze_paths["step7b"]
        assert questioner_path is not None
        assert program_path is not None
        assert qbank_path is not None
        assert executor_path is not None
        assert step7b_path is not None
        questioner_freeze = _verify_questioner_freeze(_resolve(self.settings, questioner_path))
        program_freeze = verify_program_freeze(_resolve(self.settings, program_path))
        qbank_freeze = verify_verified_qbank_freeze(_resolve(self.settings, qbank_path))
        executor_freeze = verify_query_executor_freeze(_resolve(self.settings, executor_path))
        step7b_freeze = verify_step7b_freeze(_resolve(self.settings, step7b_path))
        if (
            questioner_freeze.questioner_protocol_version != "pmmc-questioner-v3.5"
            or program_freeze.operator_set_version != "pmmc-operators-v6"
            or program_freeze.program_policy_version != "planner-program-policy-v5"
            or program_freeze.dry_run_policy_version != "evidence_driven_v6"
            or program_freeze.r_max != 2
            or qbank_freeze.schema_version != "0.3.0"
            or executor_freeze.schema_version != "0.2.0"
        ):
            raise ValueError("Step 7B online execution requires the approved v3.5/v6 protocol")
        if (
            program_freeze.questioner_freeze_id != questioner_freeze.freeze_id
            or program_freeze.questioner_freeze_hash != questioner_freeze.content_hash
            or program_freeze.partition_id != questioner_freeze.partition_id
            or executor_freeze.verified_qbank_freeze_manifest_id != qbank_freeze.freeze_manifest_id
            or qbank_freeze.pilot_content_hash != executor_freeze.pilot_content_hash
            or len(
                {
                    questioner_freeze.decision_hash,
                    program_freeze.decision_hash,
                    qbank_freeze.decision_content_hash,
                    executor_freeze.decision_content_hash,
                }
            )
            != 1
        ):
            raise ValueError("Step 7B child Freeze Manifest lineage is inconsistent")
        aggregate_lineage = {
            "questioner": (
                step7b_freeze.questioner_freeze_hash,
                questioner_freeze.content_hash,
            ),
            "program": (
                step7b_freeze.compiler_program_freeze_hash,
                program_freeze.content_hash,
            ),
            "qbank": (
                step7b_freeze.verified_qbank_freeze_hash,
                qbank_freeze.content_hash,
            ),
            "executor": (
                step7b_freeze.query_executor_freeze_hash,
                executor_freeze.content_hash,
            ),
            "answer_contract": (
                step7b_freeze.answer_contract_hash,
                executor_freeze.answer_contract_hash,
            ),
            "decision": (
                step7b_freeze.decision_content_hash,
                executor_freeze.decision_content_hash,
            ),
        }
        if any(expected != actual for expected, actual in aggregate_lineage.values()):
            mismatches = sorted(
                key for key, (expected, actual) in aggregate_lineage.items() if expected != actual
            )
            raise ValueError(
                "Step 7B aggregate Freeze Manifest lineage is inconsistent: " + ",".join(mismatches)
            )
        if (
            step7b_freeze.active_round != executor_freeze.active_round
            or step7b_freeze.question_budget != executor_freeze.question_budget
            or step7b_freeze.match_threshold != executor_freeze.match_threshold
            or step7b_freeze.gap_threshold != executor_freeze.gap_threshold
        ):
            raise ValueError("Step 7B aggregate routing policy differs from its child freeze")
        if self.config.query_executor_config_path is None:
            raise RuntimeError("Step 7 query executor config is unavailable")
        query_config_path = _resolve(self.settings, self.config.query_executor_config_path)
        query_config = load_query_executor_config(query_config_path)
        if not query_config.is_v2:
            raise ValueError("Step 7B online execution requires query executor v2")
        answer_prompt_path = query_config.resolved_answer_prompt_path(self.settings.project_root)
        if (
            query_executor_protocol_hash(query_config)
            != executor_freeze.query_executor_protocol_hash
            or query_executor_routing_policy_hash(
                query_config,
                match_threshold=executor_freeze.match_threshold,
                gap_threshold=executor_freeze.gap_threshold,
            )
            != executor_freeze.approved_routing_policy_hash
            or hashlib.sha256(answer_prompt_path.read_bytes()).hexdigest()
            != executor_freeze.answer_prompt_hash
            or query_config.fallback_policy != executor_freeze.fallback_policy_version
            or query_config.reranker_model_alias != self.config.models.reranker_model
            or query_config.answer_contract_version != executor_freeze.answer_contract_version
            or query_config.answer_output_schema_version
            != executor_freeze.answer_output_schema_version
        ):
            raise ValueError("Step 7B runtime config differs from the approved executor freeze")
        query_config = apply_approved_routing_thresholds(
            query_config,
            match_threshold=executor_freeze.match_threshold,
            gap_threshold=executor_freeze.gap_threshold,
        )
        print(
            "STEP7B_APPROVED_ROUTING_POLICY "
            + json.dumps(
                {
                    "active_gap_threshold": query_config.active_gap_threshold,
                    "active_match_threshold": query_config.active_match_threshold,
                    "approved_routing_policy_hash": (executor_freeze.approved_routing_policy_hash),
                    "query_executor_protocol_hash": (executor_freeze.query_executor_protocol_hash),
                    "state": "materialized",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

        partition_path = self.config.selection.partition_manifest_path
        partition = None
        if partition_path is not None:
            partition_dir = _resolve(self.settings, partition_path)
            if partition_dir.name == "manifest.json":
                partition_dir = partition_dir.parent
            partition = verify_partition(partition_dir)
            if partition.partition_id != questioner_freeze.partition_id:
                raise ValueError("Step 7 partition differs from the approved compiler freeze")

        compiler_ids = {
            str(key): str(value)
            for key, value in dict(self.config.metadata["compiler_run_ids"]).items()
        }
        coverage_ids = {
            str(key): str(value)
            for key, value in dict(self.config.metadata["coverage_run_ids"]).items()
        }
        fallback_ids = {
            str(key): str(value)
            for key, value in dict(self.config.metadata["fallback_run_ids"]).items()
        }
        frozen_run_ids = {
            str(key): str(value)
            for key, value in dict(qbank_freeze.metadata.get("run_ids", {})).items()
        }
        if compiler_ids != frozen_run_ids:
            raise ValueError("configured compiler runs differ from the Verified QBank freeze")

        snapshot_dirs: dict[DatasetName, Path] = {}
        snapshot_ids: dict[DatasetName, str] = {}
        for dataset in self.config.selection.datasets:
            snapshot_dir, snapshot_id = self._prepare_snapshot(dataset)
            snapshot_dirs[dataset] = snapshot_dir
            snapshot_ids[dataset] = snapshot_id
        if set(snapshot_ids.values()) != set(qbank_freeze.snapshot_ids):
            raise ValueError("Step 7 snapshots differ from the Verified QBank freeze")
        if (
            partition is not None
            and {item.dataset: item.snapshot_id for item in partition.datasets} != snapshot_ids
        ):
            raise ValueError("Step 7 partition snapshots differ from imported snapshots")

        async def load_selected(
            dataset: DatasetName,
            snapshot_id: str,
            selection: BenchmarkSelection,
        ) -> tuple[list[str], list[str], dict[str, int], str]:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                records = await PostgresRuntimeQueryRepository(sessions).list_snapshot(snapshot_id)
                selected = select_queries(records, selection)
                if not selected:
                    raise ValueError("Step 7 online query denominator is empty")
                query_ids = [item.query.query_id for item in selected]
                conversations = sorted({item.query.conversation_id for item in selected})
                scenarios: dict[str, int] = {}
                for item in selected:
                    if item.source.scenario_id is not None:
                        scenario = item.source.scenario_id
                        scenarios[scenario] = scenarios.get(scenario, 0) + 1
                return (
                    query_ids,
                    conversations,
                    scenarios,
                    selection_hash(selection, selected),
                )
            finally:
                await engine.dispose()

        samples: list[QuestionerPilotSample] = []
        work: list[dict[str, object]] = []
        for dataset in self.config.selection.datasets:
            selection = self._selection_for_dataset(dataset, snapshot_id=snapshot_ids[dataset])
            query_ids, conversations, scenarios, resolved_selection_hash = asyncio.run(
                load_selected(dataset, snapshot_ids[dataset], selection)
            )
            role = self.config.selection.partition_role or PartitionRole.PAPER_TEST
            sample = QuestionerPilotSample(
                sample_id="step7-online-selection-"
                + _stable_hash(
                    [
                        self.launch_id,
                        dataset.value,
                        snapshot_ids[dataset],
                        resolved_selection_hash,
                    ]
                )[:24],
                partition_id=questioner_freeze.partition_id,
                dataset=dataset,
                role=role,
                query_ids=query_ids,
                conversation_ids=conversations,
                scenario_counts=scenarios,
                selection_hash=resolved_selection_hash,
            )
            item: dict[str, object] = {
                "dataset": dataset.value,
                "snapshot_id": snapshot_ids[dataset],
                "snapshot_path": snapshot_dirs[dataset],
                "compiler_run_id": compiler_ids[dataset.value],
                "coverage_run_id": coverage_ids[dataset.value],
                "fallback_run_id": fallback_ids[dataset.value],
                "selected_count": len(query_ids),
                "selection": selection,
                "split": role.value,
                "sample": sample,
            }
            parent_manifest = asyncio.run(
                create_query_executor_manifest(
                    settings=self.launch_settings,
                    config=query_config,
                    source_compiler_run_id=compiler_ids[dataset.value],
                    coverage_run_id=coverage_ids[dataset.value],
                    fallback_run_id=fallback_ids[dataset.value],
                    query_ids=query_ids,
                    pilot_projection=False,
                    answer_model_alias=self.config.models.answer_model,
                )
            )
            item["query_executor_run_id"] = parent_manifest.query_executor_run_id
            samples.append(sample)
            work.append(item)
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                **dict(artifacts.get(dataset.value, {})),
                "snapshot_id": snapshot_ids[dataset],
                "snapshot_path": str(snapshot_dirs[dataset]),
                "compiler_run_id": compiler_ids[dataset.value],
                "coverage_run_id": coverage_ids[dataset.value],
                "fallback_run_id": fallback_ids[dataset.value],
                "query_executor_run_id": parent_manifest.query_executor_run_id,
                "selected_count": len(query_ids),
                "selection_hash": resolved_selection_hash,
                "split": role.value,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

        print(
            "NEXT_STAGE_PREFLIGHT "
            + json.dumps(
                {
                    "answer_model": self.config.models.answer_model,
                    "dataset_count": len(work),
                    "freeze_chain_verified": True,
                    "query_count": sum(_work_int(item, "selected_count") for item in work),
                    "stage": "answer:model-stage",
                    "source_runs_verified": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        self._run_step7_answer_stage(
            work,
            samples,
            pilot_projection=False,
            run_fallback_answer_stage=False,
        )
        self._run_step7_evaluation_stages(work)
        self._finalize_step7_online(work)

    def _finalize_step7_online(self, work: list[dict[str, object]]) -> None:
        """Export and verify the runtime and evaluator facts for formal PMMC runs."""

        from pmmc.evaluation.export import EvaluationExporter, verify_evaluation_export
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_export import ExperimentExporter
        from pmmc.infrastructure.export_verification import verify_experiment_export
        from pmmc.infrastructure.object_store import create_runtime_object_store

        async def export_query_run(run_id: str) -> Path:
            runtime_engine, runtime_sessions = create_session_factory(
                self.launch_settings, DatabaseRole.RUNTIME
            )
            try:
                result = await ExperimentExporter(
                    runtime_sessions,
                    runtime_store=create_runtime_object_store(self.launch_settings),
                ).export(
                    run_id,
                    self.settings.outputs_path / "step7-query-runs",
                    include_referenced_model_artifacts=(
                        self.config.infrastructure.archive_referenced_model_artifacts
                    ),
                )
                return result.export_dir
            finally:
                await runtime_engine.dispose()

        for item in work:
            dataset = _work_dataset(item)
            with self._stage(f"export:{dataset.value}") as plan_item:
                query_export = asyncio.run(export_query_run(str(item["query_executor_run_id"])))
                query_manifest = verify_experiment_export(query_export)
                evaluation_result = asyncio.run(
                    EvaluationExporter(self.launch_settings).export(str(item["evaluation_run_id"]))
                )
                evaluation_manifest = verify_evaluation_export(evaluation_result.export_dir)
                item["query_runtime_export_dir"] = str(query_export)
                item["evaluation_export_dir"] = str(evaluation_result.export_dir)
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset.value] = {
                    **dict(artifacts.get(dataset.value, {})),
                    "query_runtime_export_dir": str(query_export),
                    "query_export_id": query_manifest["export_id"],
                    "evaluation_export_dir": str(evaluation_result.export_dir),
                    "evaluation_export_id": evaluation_manifest["export_id"],
                }
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                self._save_state()
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units, dataset=dataset.value)

    def _run_step7_answer_stage(
        self,
        work: list[dict[str, object]],
        samples: list[QuestionerPilotSample],
        *,
        pilot_projection: bool = True,
        run_fallback_answer_stage: bool = True,
        require_terminal_reuse: bool = False,
    ) -> None:
        """Generate Native-MM counterfactuals and PMMC answers in one model session."""

        from pmmc.baselines import verify_baseline_suite
        from pmmc.runtime.manager import run_managed_commands
        from pmmc.runtime.query_executor_config import load_query_executor_config
        from pmmc.runtime.query_workflow import (
            create_query_executor_manifest,
            verify_query_executor_run,
        )

        if self.config.query_executor_config_path is None:
            raise RuntimeError("Step 7 query executor config is unavailable")
        if self.config.compiler_foundation_config_path is None:
            raise RuntimeError("Step 7 foundation config is unavailable")
        query_config = _resolve(self.settings, self.config.query_executor_config_path)
        foundation_config = _resolve(self.settings, self.config.compiler_foundation_config_path)
        query_config_model = load_query_executor_config(query_config)
        samples_by_dataset = {item.dataset: item for item in samples}
        commands: list[list[str]] = []
        command_work: list[tuple[dict[str, object], str, Path | None]] = []
        reused_fallback_query_count = 0
        reused_query_variant_count = 0
        query_fact_reuse: set[DatasetName] = set()

        def record_query_result(
            item: dict[str, object],
            *,
            query_run_id: str,
            verification: dict[str, object],
            result_path: Path | None,
        ) -> None:
            dataset = _work_dataset(item)
            item["query_executor_run_id"] = query_run_id
            item["query_verification"] = verification
            artifacts = dict(self.state.dataset_artifacts)
            dataset_artifacts = {
                **dict(artifacts.get(dataset.value, {})),
                "query_executor_run_id": query_run_id,
                "query_verification": verification,
            }
            if result_path is not None:
                dataset_artifacts["query_result_path"] = str(result_path)
            artifacts[dataset.value] = dataset_artifacts
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

        for item in work:
            dataset = _work_dataset(item)
            sample = samples_by_dataset[dataset]
            count = _work_int(item, "selected_count")
            fallback_ready = not run_fallback_answer_stage
            if run_fallback_answer_stage:
                fallback_report = asyncio.run(
                    verify_baseline_suite(
                        self.launch_settings,
                        str(item["fallback_suite_id"]),
                    )
                )
                fallback_ready = (
                    fallback_report.get("valid") is True
                    and fallback_report.get("status") == "ready"
                    and _mapping_int(fallback_report, "selected_count") == count
                )
                if fallback_ready:
                    reused_fallback_query_count += count
                else:
                    commands.append(
                        [
                            sys.executable,
                            "-m",
                            "pmmc",
                            "baseline",
                            "run-suite",
                            "--suite-id",
                            str(item["fallback_suite_id"]),
                            "--phase",
                            "answer",
                            "--method",
                            BaselineMethod.NATIVE_MULTIMODAL_RAG.value,
                        ]
                    )
                    command_work.append((item, "fallback_answer", None))

            query_reused = False
            if fallback_ready:
                expected_manifest = asyncio.run(
                    create_query_executor_manifest(
                        settings=self.launch_settings,
                        config=query_config_model,
                        source_compiler_run_id=str(item["compiler_run_id"]),
                        coverage_run_id=str(item["coverage_run_id"]),
                        fallback_run_id=str(item["fallback_run_id"]),
                        query_ids=sample.query_ids,
                        pilot_projection=pilot_projection,
                        run_id=(
                            str(item["query_executor_run_id"])
                            if isinstance(item.get("query_executor_run_id"), str)
                            else None
                        ),
                        answer_model_alias=self.config.models.answer_model,
                    )
                )
                item["query_executor_run_id"] = expected_manifest.query_executor_run_id
                try:
                    verification = asyncio.run(
                        verify_query_executor_run(
                            self.launch_settings,
                            expected_manifest.query_executor_run_id,
                        )
                    )
                except KeyError:
                    verification = None
                if verification is not None:
                    if (
                        verification.get("manifest_hash") != expected_manifest.content_hash
                        or _mapping_int(verification, "selected_query_count") != count
                    ):
                        raise ValueError("Step 7 query executor resume manifest identity mismatch")
                    if verification.get("valid") is True:
                        record_query_result(
                            item,
                            query_run_id=expected_manifest.query_executor_run_id,
                            verification=verification,
                            result_path=None,
                        )
                        query_reused = True
                        query_fact_reuse.add(dataset)
                        reused_query_variant_count += count * 4
                    elif verification.get("status") not in {"created", "running"}:
                        raise ValueError("Step 7 query executor terminal facts are incomplete")

            if query_reused:
                continue
            result_path = self.launch_dir / "step7" / f"{dataset.value}.query-run.json"
            command = [
                sys.executable,
                "-m",
                "pmmc",
                "runtime",
                "execute-query-run",
                "--source-compiler-run-id",
                str(item["compiler_run_id"]),
                "--coverage-run-id",
                str(item["coverage_run_id"]),
                "--fallback-run-id",
                str(item["fallback_run_id"]),
                "--config",
                str(query_config),
                "--foundation-config",
                str(foundation_config),
                "--answer-model",
                self.config.models.answer_model,
                "--result-json",
                str(result_path),
            ]
            if pilot_projection:
                command.append("--pilot-projection")
            for query_id in sample.query_ids:
                command.extend(["--query-id", query_id])
            existing_run_id = item.get("query_executor_run_id")
            if isinstance(existing_run_id, str) and existing_run_id:
                command.extend(["--run-id", existing_run_id])
            commands.append(command)
            command_work.append((item, "query_executor", result_path))

        variant_count = 5 if run_fallback_answer_stage else 4
        answer_total = sum(_work_int(item, "selected_count") * variant_count for item in work)
        self._set_actual_work_units("answer:model-stage", max(1, answer_total))
        print(
            "STEP7_ANSWER_RESUME_PREFLIGHT "
            + json.dumps(
                {
                    "dataset_count": len(work),
                    "fallback_query_reused_count": reused_fallback_query_count,
                    "model_start_required": bool(commands),
                    "pending_command_count": len(commands),
                    "pending_fallback_command_count": sum(
                        phase == "fallback_answer" for _item, phase, _path in command_work
                    ),
                    "pending_query_command_count": sum(
                        phase == "query_executor" for _item, phase, _path in command_work
                    ),
                    "query_run_reused_count": len(query_fact_reuse),
                    "query_variant_reused_count": reused_query_variant_count,
                    "selected_query_count": sum(_work_int(item, "selected_count") for item in work),
                    "state": "ready",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        if require_terminal_reuse and commands:
            raise RuntimeError("step7b_terminal_answer_reuse_incomplete")
        if not commands:
            self._reuse_stage("answer:model-stage")
        else:
            with self._stage("answer:model-stage"):
                # Record ownership before the child runs: failures also reuse
                # the live Step 7B service and must not count eight new startups.
                self._set_stage_performance_metadata(
                    "answer:model-stage",
                    {
                        "answer_model_alias": self.config.models.answer_model,
                        "command_count": len(commands),
                        "dataset_count": len(work),
                        "fallback_query_count": sum(
                            _work_int(item, "selected_count") for item in work
                        ),
                        "fallback_query_reused_count": reused_fallback_query_count,
                        "model_start_required": True,
                        "pmmc_round_count": 4,
                        "query_variant_reused_count": reused_query_variant_count,
                        "replica_count": self._replica_count("answer"),
                        **(
                            {
                                "physical_model_owner": "doubter:model-stage",
                                "state": "reused_active_model_service",
                            }
                            if self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
                            else {}
                        ),
                    },
                )
                print("STEP7_ANSWER_COMMAND_PLAN " + json.dumps(
                    {
                        "progress_unit": "query_answer_variant",
                        "total": answer_total,
                        "commands": [
                            {"ordinal": index, "dataset": _work_dataset(item).value,
                             "phase": phase, "query_count": _work_int(item, "selected_count")}
                            for index, (item, phase, _path) in enumerate(command_work, 1)
                        ],
                    }, sort_keys=True, separators=(",", ":"),
                ), flush=True)
                assert self.reporter is not None
                reporter = self.reporter
                if reused_fallback_query_count:
                    reporter.advance(
                        reused_fallback_query_count,
                        phase="fallback_answer",
                        state="terminal_fact_reuse",
                    )
                if reused_query_variant_count:
                    reporter.advance(
                        reused_query_variant_count,
                        phase="query_executor",
                        state="terminal_fact_reuse",
                    )

                def completed(index: int, _command: list[str]) -> None:
                    item, phase, result_path = command_work[index - 1]
                    dataset = _work_dataset(item)
                    count = _work_int(item, "selected_count")
                    if phase == "fallback_answer":
                        report = asyncio.run(
                            verify_baseline_suite(
                                self.launch_settings,
                                str(item["fallback_suite_id"]),
                            )
                        )
                        if (
                            report.get("valid") is not True
                            or report.get("status") != "ready"
                            or _mapping_int(report, "selected_count") != count
                        ):
                            raise ValueError("Step 7 Native-MM fallback suite is incomplete")
                        reporter.advance(count, dataset=dataset.value, phase=phase)
                        return
                    assert result_path is not None
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    query_run_id = payload.get("query_executor_run_id")
                    verification = payload.get("verification")
                    if (
                        not isinstance(query_run_id, str)
                        or not query_run_id
                        or not isinstance(verification, dict)
                        or verification.get("valid") is not True
                        or int(payload.get("selected_count", -1)) != count
                    ):
                        raise ValueError("Step 7 query executor result is incomplete")
                    record_query_result(
                        item,
                        query_run_id=query_run_id,
                        verification=verification,
                        result_path=result_path,
                    )
                    reporter.advance(count * 4, dataset=dataset.value, phase=phase)

                try:
                    run_managed_commands(
                        self.launch_settings,
                        profile_path=self.launch_settings.resolved_runtime_profile_path,
                        logical_stage="answer",
                        commands=commands,
                        environment_overrides=self._child_environment(),
                        command_completed=completed,
                        replica_count=self._replica_count("answer"),
                    )
                except Exception as exc:
                    if self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT:
                        # Children have stopped; retain committed item facts before
                        # sidecars stop, without marking the failed stage complete.
                        print("STEP7B_ANSWER_FAILURE_CHECKPOINT state=starting "
                              "stage_completed=false", flush=True)
                        before = self.state.metadata.get("latest_checkpoint")
                        try:
                            self._maybe_checkpoint("answer:model-stage:failure")
                            saved = before != self.state.metadata.get("latest_checkpoint")
                            print("STEP7B_ANSWER_FAILURE_CHECKPOINT "
                                  f"state={'saved' if saved else 'not_saved'} "
                                  "stage_completed=false", flush=True)
                        except Exception as checkpoint_exc:
                            print("STEP7B_ANSWER_FAILURE_CHECKPOINT state=failed "
                                  f"error_type={type(checkpoint_exc).__name__}", flush=True)
                            exc.add_note("answer failure checkpoint could not be saved")
                    raise

        for item in work:
            dataset = _work_dataset(item)
            verification = _work_mapping(item, "query_verification")
            fact_counts = _work_mapping(verification, "fact_counts")
            print(
                "STEP7_QUERY_TERMINAL_FACTS "
                + json.dumps(
                    {
                        "dataset": dataset.value,
                        "failed_count": _mapping_int(fact_counts, "failed_count"),
                        "matched_count": _mapping_int(fact_counts, "matched_count"),
                        "selected_answer_count": _mapping_int(fact_counts, "selected_answer_count"),
                        "source": (
                            "pre_model_terminal_reuse"
                            if dataset in query_fact_reuse
                            else "managed_answer_stage"
                        ),
                        "state": "verified",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            with self._stage(f"query_match:{dataset.value}") as plan_item:
                assert self.reporter is not None
                self.reporter.advance(
                    plan_item.work_units,
                    dataset=dataset.value,
                    matched=_mapping_int(fact_counts, "matched_count"),
                    shared_stage="answer:model-stage",
                    state="terminal_fact_reuse" if dataset in query_fact_reuse else "verified",
                    progress_unit="queries",
                )
            with self._stage(f"query_execute:{dataset.value}") as plan_item:
                assert self.reporter is not None
                self.reporter.advance(
                    plan_item.work_units,
                    dataset=dataset.value,
                    selected=_mapping_int(fact_counts, "selected_answer_count"),
                    failed=_mapping_int(fact_counts, "failed_count"),
                    shared_stage="answer:model-stage",
                    state="terminal_fact_reuse" if dataset in query_fact_reuse else "verified",
                    progress_unit="query_rounds",
                )
            with self._stage(f"query_verify:{dataset.value}") as plan_item:
                if verification.get("valid") is not True:
                    raise ValueError("Step 7 query executor verification failed")
                assert self.reporter is not None
                self.reporter.advance(
                    plan_item.work_units,
                    dataset=dataset.value,
                    shared_stage="answer:model-stage",
                )

    def _run_step7_evaluation_stages(
        self,
        work: list[dict[str, object]],
        *, require_terminal_reuse: bool = False,
    ) -> None:
        """Evaluate PMMC and QBank semantics with one shared Judge service start."""

        from pmmc.evaluation.qbank_semantic import load_qbank_semantic_config
        from pmmc.evaluation.qbank_semantic_workflow import (
            inspect_qbank_semantic_parent,
            verify_qbank_semantic_match_run,
        )
        from pmmc.evaluation.workflow import (
            EvaluationRunPhase,
            run_evaluation_workflow,
            verify_evaluation_run,
        )
        from pmmc.runtime.manager import run_managed_commands
        from pmmc.smoke import verify_step6c2_semantic_conformance

        if self.config.qbank_semantic_config_path is None:
            raise RuntimeError("Step 7 semantic Judge config is unavailable")
        semantic_path = _resolve(self.settings, self.config.qbank_semantic_config_path)
        semantic_config = load_qbank_semantic_config(semantic_path)
        if semantic_config.judge_model_alias != self.config.models.judge_model:
            raise ValueError("Step 7 semantic and answer Judge aliases differ")

        restored_judge_stages = {
            "evaluation_deterministic:model-stage",
            "qbank_semantic:model-stage",
            "evaluation_judge:model-stage",
        }
        restored_judge_candidate = (
            self.resume
            and self._checkpoint_restored
            and restored_judge_stages <= set(self.state.completed_stage_ids)
            and all(
                isinstance(item.get("evaluation_run_id"), str)
                and isinstance(item.get("semantic_run_id"), str)
                for item in work
            )
        )
        if restored_judge_candidate:
            reuse_rows: list[dict[str, object]] = []
            for item in work:
                dataset = _work_dataset(item)
                evaluation_report = asyncio.run(
                    verify_evaluation_run(
                        self.launch_settings,
                        str(item["evaluation_run_id"]),
                        require_managed_runtime=True,
                    )
                )
                semantic_report = asyncio.run(
                    verify_qbank_semantic_match_run(
                        self.launch_settings,
                        str(item["semantic_run_id"]),
                        semantic_config,
                    )
                )
                expected_judge_facts = _mapping_int(
                    evaluation_report, "selected_count"
                ) * _mapping_int(evaluation_report, "target_count")
                fact_coverage = _work_mapping(evaluation_report, "fact_coverage")
                judge_facts_ready = (
                    evaluation_report.get("valid") is True
                    and _mapping_int(fact_coverage, "judge_scores") == expected_judge_facts
                    and _mapping_int(fact_coverage, "successful_judge_calls")
                    == expected_judge_facts
                    and _mapping_int(evaluation_report, "managed_judge_call_count")
                    == expected_judge_facts
                )
                semantic_facts_ready = semantic_report.get("valid") is True
                reuse_rows.append(
                    {
                        "dataset": dataset.value,
                        "evaluation_judge_fact_count": _mapping_int(fact_coverage, "judge_scores"),
                        "evaluation_judge_facts_ready": judge_facts_ready,
                        "semantic_facts_ready": semantic_facts_ready,
                        "semantic_pair_count": _mapping_int(semantic_report, "pair_count"),
                    }
                )
            judge_facts_reusable = all(
                row["evaluation_judge_facts_ready"] is True and row["semantic_facts_ready"] is True
                for row in reuse_rows
            )
            print(
                "STEP7_EVALUATION_RESUME_PREFLIGHT "
                + json.dumps(
                    {
                        "datasets": reuse_rows,
                        "model_start_required": not judge_facts_reusable,
                        "restored_stage_count": len(restored_judge_stages),
                        "state": (
                            "terminal_judge_fact_reuse_ready"
                            if judge_facts_reusable
                            else "terminal_judge_fact_reuse_rejected"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if judge_facts_reusable:
                for stage_id, count_key in (
                    ("qbank_semantic:model-stage", "semantic_pair_count"),
                    ("evaluation_judge:model-stage", "evaluation_judge_fact_count"),
                ):
                    verified_count = sum(_mapping_int(row, count_key) for row in reuse_rows)
                    self._set_actual_work_units(stage_id, max(1, verified_count))
                    self._set_stage_performance_metadata(stage_id, {
                        "denominator_kind": count_key,
                        "verified_fact_count": verified_count,
                        "model_startup_count": 0,
                    })
                for stage_id in (
                    "evaluation_deterministic:model-stage",
                    "qbank_semantic:model-stage",
                    "evaluation_judge:model-stage",
                ):
                    if not self._reuse_restored_stage(stage_id):
                        raise RuntimeError("Step 7 terminal Judge facts lost checkpoint ownership")
                with self._stage("evaluation_analyze:model-stage"):
                    total = sum(_work_int(item, "selected_count") for item in work)
                    self._set_actual_work_units("evaluation_analyze:model-stage", max(1, total))
                    for item in work:
                        asyncio.run(
                            run_evaluation_workflow(
                                settings=self.launch_settings,
                                resume_evaluation_run_id=str(item["evaluation_run_id"]),
                                phase=EvaluationRunPhase.ANALYZE,
                            )
                        )
                        report = asyncio.run(
                            verify_evaluation_run(
                                self.launch_settings,
                                str(item["evaluation_run_id"]),
                                require_managed_runtime=True,
                            )
                        )
                        if not bool(report.get("valid")):
                            raise ValueError("Step 7 restored evaluation analysis is incomplete")
                        item["evaluation_verification"] = report
                        assert self.reporter is not None
                        self.reporter.advance(
                            _work_int(item, "selected_count"),
                            dataset=str(item["dataset"]),
                            source="restored_judge_facts",
                        )
                return

        if require_terminal_reuse:
            raise RuntimeError("step7b_terminal_judge_reuse_incomplete")
        with self._stage("evaluation_deterministic:model-stage"):
            total = sum(_work_int(item, "selected_count") for item in work)
            self._set_actual_work_units("evaluation_deterministic:model-stage", max(1, total))
            for item in work:
                manifest, _directory = asyncio.run(
                    run_evaluation_workflow(
                        settings=self.launch_settings,
                        query_executor_run_id=str(item["query_executor_run_id"]),
                        phase=EvaluationRunPhase.DETERMINISTIC,
                    )
                )
                item["evaluation_run_id"] = manifest.evaluation_run_id
                dataset = _work_dataset(item)
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset.value] = {
                    **dict(artifacts.get(dataset.value, {})),
                    "evaluation_run_id": manifest.evaluation_run_id,
                }
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                self._save_state()
                assert self.reporter is not None
                self.reporter.advance(_work_int(item, "selected_count"), dataset=dataset.value)

        with self._stage("qbank_semantic:model-stage") as plan_item:
            self._set_stage_performance_metadata(
                "qbank_semantic:model-stage",
                {
                    "state": "deferred_to_shared_model_stage",
                    "shared_owner": "evaluation_judge:model-stage",
                    "model_startup_count": 0,
                    "completion_scope": "dispatch_only",
                },
            )
            assert self.reporter is not None
            self.reporter.advance(
                plan_item.work_units,
                state="deferred_to_shared_model_stage",
                completion_scope="dispatch_only",
            )

        commands: list[list[str]] = []
        command_work: list[tuple[str, dict[str, object] | None, Path | None]] = []
        run_conformance = bool(self.config.metadata.get("run_conformance"))
        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
        conformance_key = "step7b_semantic_conformance" if step7b else "step7_semantic_conformance"
        conformance_path_key = conformance_key + "_path"
        conformance_path = self.launch_dir / (conformance_key + ".json")
        if run_conformance:
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "pmmc",
                    "smoke",
                    "verify-step6c2-semantic-conformance",
                    "--semantic-config",
                    str(semantic_path),
                    "--expected-replica-count",
                    str(self._replica_count("judge")),
                    "--result-json",
                    str(conformance_path),
                    "--summary-only",
                ]
            )
            command_work.append(("conformance", None, conformance_path))
        for item in work:
            dataset = _work_dataset(item)
            result_path = self.launch_dir / "semantic" / f"{dataset.value}.result.json"
            semantic_command = [
                sys.executable,
                "-m",
                "pmmc",
                "evaluate",
                "qbank-semantic-match",
                "--coverage-run-id",
                str(item["coverage_run_id"]),
                "--config",
                str(semantic_path),
                "--result-json",
                str(result_path),
            ]
            if isinstance(item.get("semantic_run_id"), str):
                semantic_command.extend(["--resume-semantic-run-id", str(item["semantic_run_id"])])
            commands.append(semantic_command)
            command_work.append(("semantic", item, result_path))
        for item in work:
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "pmmc",
                    "evaluate",
                    "run",
                    "--resume-evaluation-run-id",
                    str(item["evaluation_run_id"]),
                    "--phase",
                    "judge",
                ]
            )
            command_work.append(("answer_judge", item, None))

        with self._stage("evaluation_judge:model-stage"):
            total = sum(_work_int(item, "selected_count") * 2 for item in work)
            self._set_actual_work_units(
                "evaluation_judge:model-stage",
                max(1, total + int(run_conformance)),
            )
            assert self.reporter is not None
            reporter = self.reporter
            for item in work:
                dataset = _work_dataset(item)
                parent_report = asyncio.run(
                    inspect_qbank_semantic_parent(
                        settings=self.launch_settings,
                        coverage_run_id=str(item["coverage_run_id"]),
                        config=semantic_config,
                    )
                )
                print(
                    "STEP7_SEMANTIC_PARENT_PREFLIGHT "
                    + json.dumps(
                        parent_report,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                if parent_report.get("state") != "ready":
                    self._set_stage_performance_metadata(
                        "evaluation_judge:model-stage",
                        {
                            "dataset": dataset.value,
                            "model_startup_count": 0,
                            "preflight": parent_report,
                            "state": "blocked_by_semantic_parent",
                        },
                    )
                    raise ValueError("Step 7 semantic parent preflight failed before Judge startup")

            def completed(index: int, _command: list[str]) -> None:
                phase, item, result_path = command_work[index - 1]
                if phase == "conformance":
                    reporter.advance(1, phase=phase)
                    return
                assert item is not None
                dataset = _work_dataset(item)
                if phase == "semantic":
                    assert result_path is not None
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    semantic_run_id = payload.get("semantic_run_id")
                    if not isinstance(semantic_run_id, str) or payload.get(
                        "coverage_run_id"
                    ) != item.get("coverage_run_id"):
                        raise ValueError("Step 7 semantic result identity differs")
                    item["semantic_run_id"] = semantic_run_id
                    item["semantic_pair_count"] = int(payload.get("pair_count", 0))
                    artifacts = dict(self.state.dataset_artifacts)
                    artifacts[dataset.value] = {
                        **dict(artifacts.get(dataset.value, {})),
                        "semantic_run_id": semantic_run_id,
                        "semantic_pair_count": item["semantic_pair_count"],
                    }
                    self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                    self._save_state()
                reporter.advance(
                    _work_int(item, "selected_count"),
                    dataset=dataset.value,
                    phase=phase,
                )

            # Set physical-service ownership before dispatch so failures retain it.
            self._set_stage_performance_metadata(
                "evaluation_judge:model-stage",
                {
                    "command_count": len(commands),
                    "dataset_count": len(work),
                    "model_alias": self.config.models.judge_model,
                    "replica_count": self._replica_count("judge"),
                    "shared_qbank_and_answer_judge": True,
                    "progress_unit": "query_phase_credits",
                    **(
                        {
                            "physical_model_owner": "doubter:model-stage",
                            "state": "reused_active_model_service",
                        }
                        if step7b
                        else {}
                    ),
                },
            )
            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="judge",
                commands=commands,
                environment_overrides=self._child_environment(),
                command_completed=completed,
                replica_count=self._replica_count("judge"),
            )
            if run_conformance:
                verified = verify_step6c2_semantic_conformance(
                    conformance_path,
                    expected_replica_count=self._replica_count("judge"),
                )
                metadata = dict(self.state.metadata)
                metadata.update(
                    {
                        conformance_key: verified.model_dump(mode="json"),
                        conformance_path_key: str(conformance_path),
                    }
                )
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()
            for item in work:
                semantic_report = asyncio.run(
                    verify_qbank_semantic_match_run(
                        self.launch_settings,
                        str(item["semantic_run_id"]),
                        semantic_config,
                    )
                )
                if not bool(semantic_report.get("valid")):
                    raise ValueError("Step 7 semantic Judge run verification failed")

        with self._stage("evaluation_analyze:model-stage"):
            total = sum(_work_int(item, "selected_count") for item in work)
            self._set_actual_work_units("evaluation_analyze:model-stage", max(1, total))
            for item in work:
                asyncio.run(
                    run_evaluation_workflow(
                        settings=self.launch_settings,
                        resume_evaluation_run_id=str(item["evaluation_run_id"]),
                        phase=EvaluationRunPhase.ANALYZE,
                    )
                )
                report = asyncio.run(
                    verify_evaluation_run(self.launch_settings, str(item["evaluation_run_id"]))
                )
                if not bool(report.get("valid")):
                    raise ValueError("Step 7 analyzed evaluation is incomplete")
                item["evaluation_verification"] = report
                assert self.reporter is not None
                self.reporter.advance(
                    _work_int(item, "selected_count"),
                    dataset=str(item["dataset"]),
                )

    def _finalize_step7_pilot(
        self,
        *,
        work: list[dict[str, object]],
        samples: list[QuestionerPilotSample],
        partition_id: str,
    ) -> None:
        """Freeze all Step 7 sources, build the transfer bundle, and verify it."""

        step7b = self.config.target is ExperimentTarget.PMMC_STEP7B_EXECUTOR_PILOT
        event_prefix = "STEP7B" if step7b else "STEP7"
        output_root = self.settings.outputs_path / ("step7b-pilots" if step7b else "step7-pilots")
        protocol_version = (
            "pmmc-step7b-executor-pilot-v1" if step7b else "pmmc-step7-executor-pilot-v1"
        )

        from pmmc.compiler.program_export import verify_program_export
        from pmmc.evaluation.export import EvaluationExporter, verify_evaluation_export
        from pmmc.evaluation.qbank_export import (
            QBankCoverageExporter,
            verify_qbank_coverage_export,
        )
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.experiment_repositories import (
            ExperimentUnitOfWork,
        )
        from pmmc.infrastructure.experiment_export import ExperimentExporter
        from pmmc.infrastructure.export_verification import verify_experiment_export
        from pmmc.infrastructure.object_store import create_runtime_object_store
        from pmmc.runtime.query_workflow import verify_query_executor_run
        from pmmc.runtime.step7_pilot import (
            build_step7_pilot_bundle,
        )
        from pmmc.schemas import (
            DataScope,
            Step7PilotDatasetSource,
            Step7PilotSourceManifest,
        )

        async def export_runtime(
            run_id: str,
            *,
            dataset: DatasetName,
            include_model_artifacts: bool,
            completion_authority: str = "generic_run_state",
        ) -> Path:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
                    assert uow.runs is not None
                    generic_state = await uow.runs.get_state(run_id)
                    specialized_status: str | None = None
                    if completion_authority == "query_executor_run":
                        assert uow.query_executor is not None
                        specialized_state = await uow.query_executor.get_state(run_id)
                        specialized_status = specialized_state.status.value
                        if specialized_status != "completed":
                            raise ValueError(
                                "query runtime export requires a completed query executor run"
                            )
                    elif completion_authority != "generic_run_state":
                        raise ValueError("unsupported runtime export completion authority")
                require_generic_completion = completion_authority == "generic_run_state"
                print(
                    event_prefix
                    + "_RUNTIME_EXPORT_PREFLIGHT "
                    + json.dumps(
                        {
                            "completion_authority": completion_authority,
                            "dataset": dataset.value,
                            "generic_run_status": generic_state.status.value,
                            "include_model_artifacts": include_model_artifacts,
                            "require_generic_completion": require_generic_completion,
                            "specialized_run_status": specialized_status,
                            "state": "ready",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                result = await ExperimentExporter(
                    sessions,
                    runtime_store=create_runtime_object_store(self.launch_settings),
                    batch_size=self.launch_settings.experiment_export_batch_size,
                ).export(
                    run_id,
                    self.launch_settings.experiment_outputs_path,
                    require_completed=require_generic_completion,
                    include_referenced_model_artifacts=include_model_artifacts,
                )
                return result.export_dir
            finally:
                await engine.dispose()

        source_rows: list[Step7PilotDatasetSource] = []
        safe_log = self.launch_dir / (
            "step7b_safe_execution.log" if step7b else "step7_safe_execution.log"
        )
        source_path = self.launch_dir / (
            "step7b_pilot_source.json" if step7b else "step7_pilot_source.json"
        )
        source_metadata_key = "step7b_pilot_source_path" if step7b else "step7_pilot_source_path"
        restored_source_path = self.state.metadata.get(source_metadata_key)
        if (
            self.resume
            and self._checkpoint_restored
            and "automatic_quality" in self.state.completed_stage_ids
            and isinstance(restored_source_path, str)
            and Path(restored_source_path).is_file()
        ):
            restored_source = Step7PilotSourceManifest.model_validate(
                json.loads(Path(restored_source_path).read_text(encoding="utf-8"))
            )
            if (
                restored_source.launch_id != self.launch_id
                or restored_source.partition_id != partition_id
                or restored_source.samples != samples
                or restored_source.benchmark_root != _program_pilot_benchmark_root(self.settings)
                or restored_source.protocol_version != protocol_version
                or restored_source.output_root != output_root
            ):
                raise ValueError("restored Step 7 pilot source identity differs")
            restored_paths = [
                path
                for row in restored_source.datasets
                for path in (
                    row.compiler_runtime_export_dir,
                    row.coverage_export_dir,
                    row.query_runtime_export_dir,
                    row.evaluation_export_dir,
                    row.snapshot_dir,
                )
            ]
            restored_safe_log = restored_source.safe_execution_log
            if all(path.is_dir() for path in restored_paths) and (
                restored_safe_log is None or restored_safe_log.is_file()
            ):
                if self._reuse_restored_stage("automatic_quality"):
                    source_rows = list(restored_source.datasets)
                    safe_log = restored_safe_log or safe_log
                    print(
                        event_prefix
                        + "_SOURCE_EXPORT_REUSE "
                        + json.dumps(
                            {
                                "dataset_count": len(source_rows),
                                "source_manifest_verified": True,
                                "state": "reused",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
            else:
                print(
                    event_prefix
                    + "_SOURCE_EXPORT_REUSE "
                    + json.dumps(
                        {
                            "reason": "source_artifact_missing",
                            "state": "fallback_to_export",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

        if not source_rows:
            with self._stage("automatic_quality") as plan_item:
                for item in work:
                    dataset = _work_dataset(item)
                    query_run_id = str(item["query_executor_run_id"])
                    query_verification = asyncio.run(
                        verify_query_executor_run(self.launch_settings, query_run_id)
                    )
                    if query_verification.get("valid") is not True:
                        raise ValueError(
                            f"Step 7 query run verification failed for {dataset.value}"
                        )
                    compiler_export = asyncio.run(
                        export_runtime(
                            str(item["compiler_run_id"]),
                            dataset=dataset,
                            include_model_artifacts=True,
                        )
                    )
                    verify_program_export(compiler_export)
                    coverage_result = asyncio.run(
                        QBankCoverageExporter(self.launch_settings).export(
                            str(item["coverage_run_id"])
                        )
                    )
                    coverage_manifest = verify_qbank_coverage_export(coverage_result.export_dir)
                    if coverage_manifest.get("semantic_match_present") is not True:
                        raise ValueError(
                            "Step 7 final coverage export is missing semantic Judge facts"
                        )
                    query_export = asyncio.run(
                        export_runtime(
                            query_run_id,
                            dataset=dataset,
                            include_model_artifacts=True,
                            completion_authority="query_executor_run",
                        )
                    )
                    verify_experiment_export(query_export)
                    evaluation_result = asyncio.run(
                        EvaluationExporter(self.launch_settings).export(
                            str(item["evaluation_run_id"])
                        )
                    )
                    verify_evaluation_export(evaluation_result.export_dir)
                    item.update(
                        {
                            "runtime_export_dir": str(compiler_export),
                            "coverage_export_dir": str(coverage_result.export_dir),
                            "query_runtime_export_dir": str(query_export),
                            "evaluation_export_dir": str(evaluation_result.export_dir),
                        }
                    )
                    source_rows.append(
                        Step7PilotDatasetSource(
                            dataset=dataset,
                            compiler_runtime_export_dir=compiler_export,
                            coverage_export_dir=coverage_result.export_dir,
                            query_runtime_export_dir=query_export,
                            evaluation_export_dir=evaluation_result.export_dir,
                            snapshot_dir=Path(str(item["snapshot_path"])),
                        )
                    )
                    artifacts = dict(self.state.dataset_artifacts)
                    artifacts[dataset.value] = {
                        **dict(artifacts.get(dataset.value, {})),
                        "runtime_export_dir": str(compiler_export),
                        "coverage_export_dir": str(coverage_result.export_dir),
                        "query_runtime_export_dir": str(query_export),
                        "evaluation_export_dir": str(evaluation_result.export_dir),
                        "query_verification": query_verification,
                    }
                    self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                    self._save_state()
                safe_lines: list[str] = []
                for name in (
                    "events.jsonl",
                    "stage_results.jsonl",
                    "stage_components.jsonl",
                    "stage_performance.jsonl",
                    "checkpoint_events.jsonl",
                    "errors.jsonl",
                ):
                    path = self.launch_dir / name
                    if path.is_file():
                        safe_lines.append(
                            json.dumps(
                                {"source": name},
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                        safe_lines.extend(path.read_text(encoding="utf-8").splitlines())
                from pmmc.compiler.program_pilot import _redact_nonportable_string

                safe_text, safe_redaction_count = _redact_nonportable_string(
                    "\n".join(safe_lines) + "\n"
                )
                safe_log.write_text(safe_text, encoding="utf-8")
                print(
                    event_prefix
                    + "_SAFE_LOG "
                    + json.dumps(
                        {
                            "line_count": len(safe_lines),
                            "redaction_count": safe_redaction_count,
                            "state": "ready",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                source = Step7PilotSourceManifest(
                    protocol_version=protocol_version,
                    launch_id=self.launch_id,
                    partition_id=partition_id,
                    samples=samples,
                    datasets=source_rows,
                    benchmark_root=_program_pilot_benchmark_root(self.settings),
                    output_root=output_root,
                    safe_execution_log=safe_log,
                )
                _atomic_json(source_path, source.model_dump(mode="json"))
                metadata = dict(self.state.metadata)
                metadata[source_metadata_key] = str(source_path)
                self.state = self.state.model_copy(update={"metadata": metadata})
                self._save_state()
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units, state="sources_verified")

        with self._stage("review_bundle") as plan_item:
            source_by_dataset = {item.dataset: item for item in source_rows}
            bundle_reported_work_units = 0

            def report_bundle_progress(amount: int, dataset: DatasetName) -> None:
                nonlocal bundle_reported_work_units
                remaining = plan_item.work_units - bundle_reported_work_units
                applied = min(amount, remaining)
                if applied <= 0:
                    return
                assert self.reporter is not None
                self.reporter.advance(
                    applied,
                    dataset=dataset.value,
                    state="review_cards_ready",
                    progress_unit="review_cards",
                    bundle_completed=False,
                )
                bundle_reported_work_units += applied

            pilot_dir, manifest = build_step7_pilot_bundle(
                launch_id=self.launch_id,
                partition_id=partition_id,
                samples=samples,
                compiler_runtime_exports={
                    dataset: value.compiler_runtime_export_dir
                    for dataset, value in source_by_dataset.items()
                },
                coverage_exports={
                    dataset: value.coverage_export_dir
                    for dataset, value in source_by_dataset.items()
                },
                query_runtime_exports={
                    dataset: value.query_runtime_export_dir
                    for dataset, value in source_by_dataset.items()
                },
                evaluation_exports={
                    dataset: value.evaluation_export_dir
                    for dataset, value in source_by_dataset.items()
                },
                snapshot_dirs={
                    dataset: value.snapshot_dir for dataset, value in source_by_dataset.items()
                },
                benchmark_root=_program_pilot_benchmark_root(self.settings),
                output_root=output_root,
                safe_execution_log=safe_log,
                progress_callback=report_bundle_progress,
                step7b=step7b,
                upstream_import_receipt=self.state.metadata.get("upstream_import"),
            )
            metadata = dict(self.state.metadata)
            metadata.update(
                {
                    "pilot_id": manifest.pilot_id,
                    "pilot_dir": str(pilot_dir),
                    "pilot_manifest": manifest.model_dump(mode="json"),
                    "pilot_mode": "step7b_gpu_pilot" if step7b else "step7_gpu_pilot",
                    "pilot_status": manifest.status.value,
                    "partition_id": partition_id,
                }
            )
            self.state = self.state.model_copy(update={"metadata": metadata})
            self._save_state()
            assert self.reporter is not None
            remaining = plan_item.work_units - bundle_reported_work_units
            if remaining > 0:
                self.reporter.advance(
                    remaining,
                    state=manifest.status.value,
                    query_cards=manifest.query_count,
                )

        self._verify_step7_pilot_export(
            pilot_dir=pilot_dir,
            expected_content_hash=manifest.content_hash,
            query_runtime_exports={
                dataset: value.query_runtime_export_dir
                for dataset, value in source_by_dataset.items()
            } if step7b else {},
        )

    def _verify_step7_pilot_export(
        self,
        *,
        pilot_dir: Path,
        expected_content_hash: str,
        query_runtime_exports: Mapping[DatasetName, Path],
    ) -> None:
        from pmmc.runtime.step7_pilot import verify_step7_pilot_bundle

        with self._stage("export_verify") as plan_item:
            checked = verify_step7_pilot_bundle(pilot_dir)
            if checked.content_hash != expected_content_hash:
                raise ValueError("Step 7 transfer verification changed pilot identity")
            for dataset, query_export in sorted(query_runtime_exports.items()):
                try:
                    diagnostics = _step7b_match_diagnostics(query_export)
                except (OSError, ValueError, TypeError, KeyError):
                    # Optional telemetry must not mask the verified acceptance result.
                    diagnostics = {"state": "unavailable", "reason": "diagnostic_source_invalid"}
                print(
                    "STEP7B_MATCH_DIAGNOSTICS " + json.dumps(
                        {"dataset": dataset.value, **diagnostics}, sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            print(
                "STEP7_PILOT_ACCEPTANCE " + json.dumps({
                    "state": checked.status.value,
                    "transfer_verified": True,
                    "integrity_passed": checked.automatic_integrity_passed,
                    "integrity_issue_count": len(checked.integrity_issue_types),
                }, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            if not checked.automatic_integrity_passed:
                # Fail inside the stage so progress and current-attempt accounting
                # cannot report export_verify completed / launcher failed.
                self.state = self.state.model_copy(update={"completed_stage_ids": [
                    stage for stage in self.state.completed_stage_ids if stage != "export_verify"
                ]})
                raise Step7PilotAcceptanceError(checked.integrity_issue_types)
            assert self.reporter is not None
            self.reporter.advance(
                plan_item.work_units,
                state=checked.status.value,
                integrity_passed=checked.automatic_integrity_passed,
            )

    def _run_program_pilot(self) -> None:
        if self.execution_config.schedule is not ExecutionSchedule.MODEL_STAGE_MAJOR_V1:
            raise ValueError("Step 6C program pilots require model-stage-major execution")
        benchmark_root = _program_pilot_benchmark_root(self.settings)
        from pmmc.compiler.program_pilot import (
            build_program_pilot_bundle,
            verify_program_pilot_bundle,
        )
        from pmmc.pilots.config import load_questioner_pilot_config
        from pmmc.pilots.partition import (
            build_partition,
            load_partition,
            sample_development_queries,
            verify_partition,
        )
        from pmmc.schemas import (
            BenchmarkPartitionSpec,
            CompilerProgramPilotSourceManifest,
            ProgramPilotDatasetSource,
        )

        if self.config.questioner_pilot_config_path is None:
            raise RuntimeError("program pilot config is unavailable")
        pilot_config = load_questioner_pilot_config(
            _resolve(self.settings, self.config.questioner_pilot_config_path)
        )
        snapshot_dirs: dict[DatasetName, Path] = {}
        snapshot_ids: dict[DatasetName, str] = {}
        for dataset in self.config.selection.datasets:
            snapshot_dir, snapshot_id = self._prepare_snapshot(dataset)
            snapshot_dirs[dataset] = snapshot_dir
            snapshot_ids[dataset] = snapshot_id
        snapshot_stages = frozenset(
            stage_id
            for dataset in self.config.selection.datasets
            for stage_id in (
                f"benchmark:{dataset.value}",
                f"snapshot_import:{dataset.value}",
            )
        )
        if self._reuse_completed_program_stage_major(
            already_executed=snapshot_stages,
            expected_snapshot_ids={
                dataset.value: snapshot_id for dataset, snapshot_id in snapshot_ids.items()
            },
        ):
            return

        with self._stage("partition") as plan_item:
            configured = self.config.selection.partition_manifest_path
            if configured is None:
                partition_dir, partition_manifest = build_partition(
                    snapshot_dirs=[
                        snapshot_dirs[value]
                        for value in sorted(snapshot_dirs, key=lambda entry: entry.value)
                    ],
                    output_root=self.settings.artifacts_path / "pilots" / "partitions",
                    spec=BenchmarkPartitionSpec(
                        protocol_version=pilot_config.partition_protocol,
                        fold_count=pilot_config.fold_count,
                        development_fold=pilot_config.development_fold,
                        seed=pilot_config.partition_seed,
                    ),
                )
            else:
                partition_dir = _resolve(self.settings, configured)
                partition_manifest = verify_partition(partition_dir)
            if {
                item.dataset: item.snapshot_id for item in partition_manifest.datasets
            } != snapshot_ids:
                raise ValueError("program pilot partition does not match current snapshots")
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)

        with self._stage("pilot_sample") as plan_item:
            _partition, assignments = load_partition(partition_dir)
            samples = sample_development_queries(
                manifest=partition_manifest,
                assignments=assignments,
                config=pilot_config,
            )
            _atomic_json(
                self.launch_dir / "pilot_samples.json",
                [item.model_dump(mode="json") for item in samples],
            )
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)

        configured_pilot_mode = str(self.config.metadata.get("pilot_mode", "quality_pilot"))
        if configured_pilot_mode == "gpu_pilot":
            prefix_count = int(self.config.metadata.get("prefix_query_count_per_dataset", 2))
            if prefix_count != 2:
                raise ValueError("Step 6D GPU pilot freezes a two-query prefix denominator")
            prefix_datasets: dict[str, dict[str, object]] = {
                item.dataset.value: {
                    "query_count": len(item.query_ids[:prefix_count]),
                    "query_id_hashes": [
                        hashlib.sha256(value.encode("utf-8")).hexdigest()
                        for value in item.query_ids[:prefix_count]
                    ],
                }
                for item in samples
            }
            prefix_payload: dict[str, object] = {
                "protocol_version": "step6d-two-query-prefix-v1",
                "launch_id": self.launch_id,
                "paper_result": False,
                "scientific_sample": False,
                "partition_id": partition_manifest.partition_id,
                "datasets": prefix_datasets,
            }
            if any(value["query_count"] != prefix_count for value in prefix_datasets.values()):
                raise ValueError("Step 6D GPU pilot prefix is smaller than two queries")
            prefix_payload["content_hash"] = _stable_hash(prefix_payload)
            prefix_path = self.launch_dir / "step6d_prefix_manifest.json"
            _atomic_json(prefix_path, prefix_payload)
            print(
                "STEP6D_PREFIX_CHECK "
                + json.dumps(
                    {
                        "dataset_count": len(samples),
                        "prefix_query_count_per_dataset": prefix_count,
                        "state": "ready",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            metadata = dict(self.state.metadata)
            metadata.update(
                {
                    "step6d_prefix_manifest": prefix_payload,
                    "step6d_prefix_manifest_path": str(prefix_path),
                }
            )
            self.state = self.state.model_copy(update={"metadata": metadata})
            self._save_state()

        samples_by_dataset = {item.dataset: item for item in samples}
        work: list[dict[str, object]] = []
        for dataset in self.config.selection.datasets:
            sample = samples_by_dataset[dataset]
            selection = self._pilot_selection(sample)
            conversations, selected_count = self._questioner_denominator(
                dataset, snapshot_ids[dataset], selection=selection
            )
            if selected_count != len(sample.query_ids) or conversations != sorted(
                sample.conversation_ids
            ):
                raise ValueError("program pilot sample differs from imported denominator")
            run_id = (
                "pmmc-programs-"
                + _stable_hash(
                    [
                        self.launch_id,
                        partition_manifest.partition_id,
                        sample.sample_id,
                        dataset.value,
                        snapshot_ids[dataset],
                        conversations,
                    ]
                )[:24]
            )
            work.append(
                self._program_work_item(
                    dataset=dataset,
                    snapshot_id=snapshot_ids[dataset],
                    snapshot_path=snapshot_dirs[dataset],
                    selection=selection,
                    conversations=conversations,
                    selected_count=selected_count,
                    compiler_run_id=run_id,
                    split=PartitionRole.DEVELOPMENT.value,
                    sample=sample,
                    partition_id=partition_manifest.partition_id,
                )
            )

        self._execute_program_work(work)
        blocked = [item for item in work if item.get("r0_gate_passed") is False]
        if blocked and _is_verified_qbank_target(self.config.target):
            from pmmc.compiler.step6d_failure import (
                build_step6d_failure_capsule,
                verify_step6d_failure_capsule,
            )

            with self._stage("automatic_quality") as plan_item:
                automatic = {
                    "protocol": "step6d-r0-remediation-v1",
                    "launch_id": self.launch_id,
                    "partition_id": partition_manifest.partition_id,
                    "paper_result": False,
                    "status": "needs_remediation",
                    "valid": False,
                    "datasets": {
                        str(item["dataset"]): {
                            "r0_manifest": item.get("r0_manifest"),
                            "program_verification": item.get("program_verification"),
                        }
                        for item in work
                    },
                }
                _atomic_json(
                    self.launch_dir / "program_automatic_quality.json",
                    automatic,
                )
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            with self._stage("review_bundle") as plan_item:
                pilot_dir, manifest_value = build_step6d_failure_capsule(
                    launch_id=self.launch_id,
                    partition_id=partition_manifest.partition_id,
                    samples=samples,
                    work=work,
                    launch_dir=self.launch_dir,
                    output_root=(self.settings.outputs_path / "compiler-program-pilots"),
                )
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            with self._stage("export_verify") as plan_item:
                verified = verify_step6d_failure_capsule(pilot_dir)
                if verified["content_hash"] != manifest_value["content_hash"]:
                    raise ValueError("Step 6D remediation capsule identity changed")
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            metadata = dict(self.state.metadata)
            metadata.update(
                {
                    "pilot_dir": str(pilot_dir),
                    "pilot_id": str(manifest_value["capsule_id"]),
                    "pilot_status": "needs_remediation",
                    "pilot_mode": str(
                        self.config.metadata.get(
                            "pilot_mode",
                            "quality_pilot",
                        )
                    ),
                    "pilot_manifest": manifest_value,
                    "partition_dir": str(partition_dir),
                    "partition_id": partition_manifest.partition_id,
                    "paper_result": False,
                    "execution_schedule": self.execution_config.schedule.value,
                }
            )
            self.state = self.state.model_copy(update={"metadata": metadata})
            self._save_state()
            return
        with self._stage("automatic_quality") as plan_item:

            def verification_passed(item: dict[str, object]) -> bool:
                value = item.get("program_verification")
                return isinstance(value, dict) and bool(value.get("valid"))

            def semantic_verification_passed(item: dict[str, object]) -> bool:
                value = item.get("semantic_verification")
                return isinstance(value, dict) and bool(value.get("valid"))

            automatic = {
                "protocol": (
                    "compiler-program-r0-automatic-integrity-v2"
                    if _is_semantic_program_target(self.config.target)
                    else "compiler-program-automatic-integrity-v1"
                ),
                "launch_id": self.launch_id,
                "partition_id": partition_manifest.partition_id,
                "paper_result": False,
                "datasets": {str(item["dataset"]): item["program_verification"] for item in work},
                "semantic_runs": {
                    str(item["dataset"]): item.get("semantic_verification")
                    for item in work
                    if item.get("semantic_verification") is not None
                },
                "valid": all(verification_passed(item) for item in work)
                and (
                    all(semantic_verification_passed(item) for item in work)
                    if _is_semantic_program_target(self.config.target)
                    else True
                ),
            }
            _atomic_json(self.launch_dir / "program_automatic_quality.json", automatic)
            if not automatic["valid"] and not _is_verified_qbank_target(self.config.target):
                raise ValueError("program pilot automatic integrity gate failed")
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)

        mode = str(self.config.metadata.get("pilot_mode", "quality_pilot"))
        if mode in {"quality_pilot", "gpu_pilot"}:
            pilot_source = CompilerProgramPilotSourceManifest(
                partition_id=partition_manifest.partition_id,
                samples=samples,
                datasets=[
                    ProgramPilotDatasetSource(
                        dataset=dataset,
                        runtime_export_dir=Path(
                            str(self.state.dataset_artifacts[dataset.value]["runtime_export_dir"])
                        ),
                        coverage_export_dir=Path(
                            str(self.state.dataset_artifacts[dataset.value]["coverage_export_dir"])
                        ),
                        snapshot_dir=snapshot_dirs[dataset],
                    )
                    for dataset in sorted(
                        self.config.selection.datasets,
                        key=lambda value: value.value,
                    )
                ],
                benchmark_root=benchmark_root,
                output_root=self.settings.outputs_path / "compiler-program-pilots",
            )
            _atomic_json(
                self.launch_dir / "program_pilot_source_manifest.json",
                pilot_source.model_dump(mode="json"),
            )
            print(
                "PROGRAM_PILOT_SOURCE "
                + json.dumps(
                    {
                        "dataset_count": len(pilot_source.datasets),
                        "phase": "ready",
                        "sample_count": len(pilot_source.samples),
                        "stage": "review_bundle",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            with self._stage("review_bundle") as plan_item:
                pilot_dir, pilot_manifest = build_program_pilot_bundle(
                    partition_id=partition_manifest.partition_id,
                    samples=samples,
                    runtime_exports={
                        dataset: Path(
                            str(self.state.dataset_artifacts[dataset.value]["runtime_export_dir"])
                        )
                        for dataset in self.config.selection.datasets
                    },
                    coverage_exports={
                        dataset: Path(
                            str(self.state.dataset_artifacts[dataset.value]["coverage_export_dir"])
                        )
                        for dataset in self.config.selection.datasets
                    },
                    snapshot_dirs=snapshot_dirs,
                    benchmark_root=benchmark_root,
                    output_root=self.settings.outputs_path / "compiler-program-pilots",
                )
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            with self._stage("export_verify") as plan_item:
                verified_pilot = verify_program_pilot_bundle(pilot_dir)
                if verified_pilot.content_hash != pilot_manifest.content_hash:
                    raise ValueError("program pilot identity changed after export")
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            pilot_id = pilot_manifest.pilot_id
            pilot_status = pilot_manifest.status.value
            manifest_value = pilot_manifest.model_dump(mode="json")
        elif mode == "smoke":
            pilot_id = f"smoke-{self.launch_id}"
            pilot_dir = self.settings.outputs_path / "compiler-program-pilots" / pilot_id
            with self._stage("review_bundle") as plan_item:
                pilot_dir.mkdir(parents=True, exist_ok=True)
                manifest_value = {
                    "mode": "smoke",
                    "launch_id": self.launch_id,
                    "pilot_id": pilot_id,
                    "partition_id": partition_manifest.partition_id,
                    "sample_ids": [item.sample_id for item in samples],
                    "runtime_exports": {
                        dataset.value: self.state.dataset_artifacts[dataset.value][
                            "runtime_export_dir"
                        ]
                        for dataset in self.config.selection.datasets
                    },
                    "coverage_exports": {
                        dataset.value: self.state.dataset_artifacts[dataset.value][
                            "coverage_export_dir"
                        ]
                        for dataset in self.config.selection.datasets
                    },
                    "semantic_runs": {
                        dataset.value: self.state.dataset_artifacts[dataset.value].get(
                            "semantic_run_id"
                        )
                        for dataset in self.config.selection.datasets
                    },
                    "program_verifications": {
                        str(item["dataset"]): item["program_verification"] for item in work
                    },
                    "planner_conformance": self.state.metadata.get("step6c_planner_conformance"),
                    "semantic_conformance": self.state.metadata.get("step6c2_semantic_conformance"),
                    "paper_result": False,
                }
                _atomic_json(pilot_dir / "manifest.json", manifest_value)
                _atomic_json(pilot_dir / "automatic_quality.json", automatic)
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            with self._stage("export_verify") as plan_item:
                stored = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
                if stored != manifest_value:
                    raise ValueError("program smoke manifest changed after export")
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            pilot_status = "completed_smoke"
        else:
            raise ValueError("unsupported program pilot mode")

        metadata = dict(self.state.metadata)
        metadata.update(
            {
                "pilot_dir": str(pilot_dir),
                "pilot_id": pilot_id,
                "pilot_status": pilot_status,
                "pilot_mode": mode,
                "pilot_manifest": manifest_value,
                "partition_dir": str(partition_dir),
                "partition_id": partition_manifest.partition_id,
                "paper_result": False,
                "execution_schedule": self.execution_config.schedule.value,
            }
        )
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()

    def _run_program_datasets(self) -> None:
        if self.execution_config.schedule is not ExecutionSchedule.MODEL_STAGE_MAJOR_V1:
            raise ValueError("Step 6C Planner requires model-stage-major execution")
        if (
            not self.config.evaluate
            and self.config.target is not ExperimentTarget.PMMC_COMPILE_ONLY
        ):
            raise ValueError("Step 6C Planner launcher requires QBank coverage evaluation")
        snapshot_dirs: dict[DatasetName, Path] = {}
        snapshot_ids: dict[DatasetName, str] = {}
        for dataset in self.config.selection.datasets:
            snapshot_dir, snapshot_id = self._prepare_snapshot(dataset)
            snapshot_dirs[dataset] = snapshot_dir
            snapshot_ids[dataset] = snapshot_id
        snapshot_stages = frozenset(
            stage_id
            for dataset in self.config.selection.datasets
            for stage_id in (
                f"benchmark:{dataset.value}",
                f"snapshot_import:{dataset.value}",
            )
        )
        if self._reuse_completed_program_stage_major(
            already_executed=snapshot_stages,
            expected_snapshot_ids={
                dataset.value: snapshot_id for dataset, snapshot_id in snapshot_ids.items()
            },
        ):
            return
        work: list[dict[str, object]] = []
        for dataset in self.config.selection.datasets:
            selection = self._selection_for_dataset(dataset, snapshot_id=snapshot_ids[dataset])
            conversations, selected_count = self._questioner_denominator(
                dataset, snapshot_ids[dataset], selection=selection
            )
            run_id = (
                "pmmc-programs-"
                + _stable_hash(
                    [
                        self.launch_id,
                        dataset.value,
                        snapshot_ids[dataset],
                        conversations,
                        _stable_hash(selection.model_dump(mode="json")),
                    ]
                )[:24]
            )
            split = selection.metadata.get("partition_role")
            work.append(
                self._program_work_item(
                    dataset=dataset,
                    snapshot_id=snapshot_ids[dataset],
                    snapshot_path=snapshot_dirs[dataset],
                    selection=selection,
                    conversations=conversations,
                    selected_count=selected_count,
                    compiler_run_id=run_id,
                    split=str(split) if split is not None else None,
                )
            )
        self._execute_program_work(work)

    def _run_questioner_pilot_stage_major(self) -> None:
        from pmmc.evaluation.qbank_workflow import QBankCoveragePhase
        from pmmc.pilots.analysis import analyze_questioner_quality
        from pmmc.pilots.bundle import build_pilot_bundle, verify_pilot_bundle
        from pmmc.pilots.config import load_questioner_pilot_config
        from pmmc.pilots.io import read_json
        from pmmc.pilots.partition import (
            build_partition,
            load_partition,
            sample_development_queries,
            verify_partition,
        )
        from pmmc.schemas import BenchmarkPartitionSpec

        if self.config.questioner_pilot_config_path is None:
            raise RuntimeError("Questioner pilot config is unavailable")
        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        pilot_config = load_questioner_pilot_config(
            _resolve(self.settings, self.config.questioner_pilot_config_path)
        )
        snapshot_dirs: dict[DatasetName, Path] = {}
        snapshot_ids: dict[DatasetName, str] = {}
        for dataset in self.config.selection.datasets:
            snapshot_dir, snapshot_id = self._prepare_snapshot(dataset)
            snapshot_dirs[dataset] = snapshot_dir
            snapshot_ids[dataset] = snapshot_id
        snapshot_stages = frozenset(
            stage_id
            for dataset in self.config.selection.datasets
            for stage_id in (
                f"benchmark:{dataset.value}",
                f"snapshot_import:{dataset.value}",
            )
        )
        if self._reuse_completed_questioner_stage_major(
            already_executed=snapshot_stages,
            expected_snapshot_ids={
                dataset.value: snapshot_id for dataset, snapshot_id in snapshot_ids.items()
            },
        ):
            return

        with self._stage("partition") as plan_item:
            configured = self.config.selection.partition_manifest_path
            if configured is not None:
                partition_dir = _resolve(self.settings, configured)
                partition_manifest = verify_partition(partition_dir)
            else:
                partition_dir, partition_manifest = build_partition(
                    snapshot_dirs=[
                        snapshot_dirs[value]
                        for value in sorted(snapshot_dirs, key=lambda entry: entry.value)
                    ],
                    output_root=self.settings.artifacts_path / "pilots" / "partitions",
                    spec=BenchmarkPartitionSpec(
                        protocol_version=pilot_config.partition_protocol,
                        fold_count=pilot_config.fold_count,
                        development_fold=pilot_config.development_fold,
                        seed=pilot_config.partition_seed,
                    ),
                )
            if {
                value.dataset: value.snapshot_id for value in partition_manifest.datasets
            } != snapshot_ids:
                raise ValueError("Questioner pilot partition does not match snapshots")
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)

        with self._stage("pilot_sample") as plan_item:
            _manifest, assignments = load_partition(partition_dir)
            samples = sample_development_queries(
                manifest=partition_manifest,
                assignments=assignments,
                config=pilot_config,
            )
            _atomic_json(
                self.launch_dir / "pilot_samples.json",
                [sample.model_dump(mode="json") for sample in samples],
            )
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)

        samples_by_dataset = {sample.dataset: sample for sample in samples}
        work: list[dict[str, object]] = []
        for dataset in self.config.selection.datasets:
            sample = samples_by_dataset[dataset]
            selection = self._pilot_selection(sample)
            conversations, selected_count = self._questioner_denominator(
                dataset, snapshot_ids[dataset], selection=selection
            )
            if selected_count != len(sample.query_ids) or conversations != sorted(
                sample.conversation_ids
            ):
                raise ValueError("pilot sample differs from imported query denominator")
            compiler_run_id = (
                "pmmc-questioner-"
                + _stable_hash(
                    [
                        self.launch_id,
                        partition_manifest.partition_id,
                        sample.sample_id,
                        dataset.value,
                        snapshot_ids[dataset],
                        conversations,
                    ]
                )[:24]
            )
            preparation, unit_count, compiler_run_id = self._prepare_questioner_units_stage(
                dataset=dataset,
                snapshot_id=snapshot_ids[dataset],
                compiler_run_id=compiler_run_id,
                conversations=conversations,
            )
            work.append(
                {
                    "dataset": dataset.value,
                    "snapshot_id": snapshot_ids[dataset],
                    "snapshot_path": snapshot_dirs[dataset],
                    "compiler_run_id": compiler_run_id,
                    "conversations": conversations,
                    "selected_count": selected_count,
                    "selection": selection,
                    "sample": sample,
                    "preparation": preparation,
                    "unit_count": unit_count,
                    "split": PartitionRole.DEVELOPMENT.value,
                }
            )

        self._run_questioner_model_stage(work)
        self._run_qbank_model_stage(
            work,
            index_phase="index_text",
            embedding_phase=QBankCoveragePhase.EMBED_TEXT.value,
            runtime_stage="embedding_text",
            plan_stage="qbank_text:model-stage",
        )
        self._run_qbank_model_stage(
            work,
            index_phase="index_multimodal",
            embedding_phase=QBankCoveragePhase.EMBED_MULTIMODAL.value,
            runtime_stage="embedding_multimodal",
            plan_stage="qbank_multimodal:model-stage",
        )
        for item in work:
            item["questioner_verification"] = self._verify_questioner_dataset(
                DatasetName(str(item["dataset"])), str(item["compiler_run_id"])
            )

        for item in work:
            dataset = DatasetName(str(item["dataset"]))
            coverage_run_id = self._score_qbank_coverage_stage_major(item)
            if coverage_run_id != item["coverage_run_id"]:
                raise ValueError("coverage run identity changed between phases")
            runtime_export, coverage_export, export_verification = self._export_questioner_dataset(
                dataset, str(item["compiler_run_id"]), coverage_run_id
            )
            if coverage_export is None:
                raise RuntimeError("Questioner pilot requires a coverage export")
            sample = QuestionerPilotSample.model_validate(item["sample"])
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                "snapshot_id": item["snapshot_id"],
                "snapshot_path": str(item["snapshot_path"]),
                "compiler_run_id": item["compiler_run_id"],
                "conversation_count": len(_work_strings(item, "conversations")),
                "selected_count": item["selected_count"],
                "selection_hash": sample.selection_hash,
                "pilot_sample_id": sample.sample_id,
                "partition_id": partition_manifest.partition_id,
                "coverage_run_id": coverage_run_id,
                "runtime_export_dir": str(runtime_export),
                "coverage_export_dir": str(coverage_export),
                "unit_preparation_dir": str(item["preparation"]),
                "questioner_verification": item["questioner_verification"],
                "questioner_export_verification": export_verification,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

        runtime_exports = {
            dataset: Path(str(self.state.dataset_artifacts[dataset.value]["runtime_export_dir"]))
            for dataset in self.config.selection.datasets
        }
        coverage_exports = {
            dataset: Path(str(self.state.dataset_artifacts[dataset.value]["coverage_export_dir"]))
            for dataset in self.config.selection.datasets
        }
        sample_query_ids = {query_id for sample in samples for query_id in sample.query_ids}
        selected_assignments = [
            assignment for assignment in assignments if assignment.query_id in sample_query_ids
        ]
        with self._stage("automatic_quality") as plan_item:
            automatic = analyze_questioner_quality(
                config=pilot_config,
                runtime_exports=runtime_exports,
                coverage_exports=coverage_exports,
                assignments=selected_assignments,
            )
            _atomic_json(self.launch_dir / "automatic_quality.json", automatic)
            assert self.reporter is not None
            self.reporter.advance(plan_item.work_units)
        pilot_mode = str(self.config.metadata.get("pilot_mode", "quality_pilot"))
        if pilot_mode == "smoke":
            pilot_dir = (
                self.settings.outputs_path / "questioner-quality-pilots" / f"smoke-{self.launch_id}"
            )
            with self._stage("review_bundle") as plan_item:
                pilot_dir.mkdir(parents=True, exist_ok=True)
                _atomic_json(
                    pilot_dir / "manifest.json",
                    {
                        "mode": "smoke",
                        "launch_id": self.launch_id,
                        "partition_id": partition_manifest.partition_id,
                        "sample_ids": [item.sample_id for item in samples],
                        "runtime_exports": {
                            key.value: str(value)
                            for key, value in sorted(
                                runtime_exports.items(), key=lambda item: item[0].value
                            )
                        },
                        "coverage_exports": {
                            key.value: str(value)
                            for key, value in sorted(
                                coverage_exports.items(), key=lambda item: item[0].value
                            )
                        },
                        "paper_result": False,
                    },
                )
                _atomic_json(pilot_dir / "automatic_quality.json", automatic)
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            with self._stage("export_verify") as plan_item:
                if not (pilot_dir / "manifest.json").is_file():
                    raise ValueError("Questioner smoke manifest is missing")
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            pilot_id = f"smoke-{self.launch_id}"
            pilot_status = "completed_smoke"
            pilot_manifest_value = read_json(pilot_dir / "manifest.json")
        elif pilot_mode == "quality_pilot":
            with self._stage("review_bundle") as plan_item:
                pilot_dir, pilot_manifest = build_pilot_bundle(
                    config=pilot_config,
                    partition_dir=partition_dir,
                    samples=samples,
                    runtime_exports=runtime_exports,
                    coverage_exports=coverage_exports,
                    snapshot_dirs=snapshot_dirs,
                    output_root=self.settings.outputs_path / "questioner-quality-pilots",
                )
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            with self._stage("export_verify") as plan_item:
                verified = verify_pilot_bundle(pilot_dir)
                if verified.content_hash != pilot_manifest.content_hash:
                    raise ValueError("Questioner pilot bundle identity changed after export")
                assert self.reporter is not None
                self.reporter.advance(plan_item.work_units)
            pilot_id = pilot_manifest.pilot_id
            pilot_status = pilot_manifest.status.value
            pilot_manifest_value = read_json(pilot_dir / "manifest.json")
        else:
            raise ValueError("unsupported Questioner pilot mode")
        metadata = dict(self.state.metadata)
        metadata.update(
            {
                "pilot_dir": str(pilot_dir),
                "pilot_id": pilot_id,
                "pilot_status": pilot_status,
                "partition_dir": str(partition_dir),
                "partition_id": partition_manifest.partition_id,
                "pilot_manifest": pilot_manifest_value,
                "pilot_mode": pilot_mode,
                "execution_schedule": self.execution_config.schedule.value,
                "paper_result": False,
            }
        )
        self.state = self.state.model_copy(update={"metadata": metadata})
        self._save_state()

    def _pilot_selection(self, sample: QuestionerPilotSample) -> BenchmarkSelection:
        return BenchmarkSelection(
            datasets=[sample.dataset],
            query_ids=sample.query_ids,
            seed=self.config.selection.seed,
            metadata={
                "partition_id": sample.partition_id,
                "partition_role": sample.role.value,
                "pilot_sample_id": sample.sample_id,
                "pilot_selection_hash": sample.selection_hash,
            },
        )

    def _selection_for_dataset(
        self,
        dataset: DatasetName,
        *,
        snapshot_id: str | None = None,
    ) -> BenchmarkSelection:
        selection = self.config.selection.for_dataset(dataset)
        partition_path = self.config.selection.partition_manifest_path
        partition_role = self.config.selection.partition_role
        if partition_path is None or partition_role is None:
            return selection
        from pmmc.pilots.partition import resolve_partition_selection

        partition_dir = _resolve(self.settings, partition_path)
        if partition_dir.name == "manifest.json":
            partition_dir = partition_dir.parent
        return resolve_partition_selection(
            partition_dir=partition_dir,
            role=partition_role,
            dataset=dataset,
            selection=selection,
            snapshot_id=snapshot_id,
        )

    def _questioner_denominator(
        self,
        dataset: DatasetName,
        snapshot_id: str,
        *,
        selection: BenchmarkSelection | None = None,
    ) -> tuple[list[str], int]:
        from pmmc.benchmarks.selection import select_queries
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.repositories import (
            PostgresRuntimeQueryRepository,
        )

        async def load() -> tuple[list[str], int]:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                records = await PostgresRuntimeQueryRepository(sessions).list_snapshot(snapshot_id)
                selected = select_queries(
                    records,
                    selection or self._selection_for_dataset(dataset, snapshot_id=snapshot_id),
                )
                if not selected:
                    raise ValueError("Questioner launcher query denominator is empty")
                conversations = sorted({item.query.conversation_id for item in selected})
                return conversations, len(selected)
            finally:
                await engine.dispose()

        return asyncio.run(load())

    def _run_questioner_stage(
        self,
        dataset: DatasetName,
        snapshot_id: str,
        compiler_run_id: str,
        conversations: list[str],
        split: str | None = None,
    ) -> None:
        from pmmc.runtime.manager import run_managed_command

        stage_id = f"questioner:{dataset.value}"
        with self._stage(stage_id) as item:
            command = [
                sys.executable,
                "-m",
                "pmmc",
                "compile-memory",
                "--target-stage",
                "questions",
                "--phase",
                "questioner",
                "--snapshot",
                snapshot_id,
                "--run-id",
                compiler_run_id,
                "--config",
                str(self.launch_settings.resolved_questioner_config_path),
                "--compiler-model",
                str(self.config.models.compiler_model),
                "--seed",
                str(self.config.selection.seed),
            ]
            for conversation_id in conversations:
                command.extend(["--conversation-id", conversation_id])
            if split is not None:
                command.extend(["--split", split])
            run_managed_command(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="compiler",
                command=command,
                environment_overrides=self._child_environment(),
            )
            assert self.reporter is not None
            self.reporter.advance(item.work_units, dataset=dataset.value)

    def _run_qbank_index_stage(
        self,
        dataset: DatasetName,
        compiler_run_id: str,
        *,
        phase: str,
        runtime_stage: str,
        plan_stage: str,
    ) -> None:
        from pmmc.runtime.manager import run_managed_command

        with self._stage(plan_stage) as item:
            run_managed_command(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage=runtime_stage,
                command=[
                    sys.executable,
                    "-m",
                    "pmmc",
                    "compile-memory",
                    "--resume-run-id",
                    compiler_run_id,
                    "--phase",
                    phase,
                    "--config",
                    str(self.launch_settings.resolved_questioner_config_path),
                    "--compiler-model",
                    str(self.config.models.compiler_model),
                ],
                environment_overrides=self._child_environment(),
            )
            assert self.reporter is not None
            self.reporter.advance(item.work_units, dataset=dataset.value)

    def _coverage_command(
        self,
        dataset: DatasetName,
        compiler_run_id: str,
        phase: str,
        *,
        selection: BenchmarkSelection | None = None,
        coverage_run_id: str | None = None,
    ) -> list[str]:
        resolved_selection = selection or self._selection_for_dataset(dataset)
        command = [
            sys.executable,
            "-m",
            "pmmc",
            "evaluate",
            "qbank-coverage",
            "--compiler-run-id",
            compiler_run_id,
            "--dataset",
            dataset.value,
            "--config",
            str(self.launch_settings.resolved_questioner_config_path),
            "--phase",
            phase,
            "--seed",
            str(resolved_selection.seed),
            "--shard-count",
            str(resolved_selection.shard_count),
            "--shard-index",
            str(resolved_selection.shard_index),
        ]
        if resolved_selection.limit is not None:
            command.extend(["--limit", str(resolved_selection.limit)])
        if resolved_selection.sample_size is not None:
            command.extend(["--sample-size", str(resolved_selection.sample_size)])
        if resolved_selection.include_excluded:
            command.append("--include-excluded")
        for question_type in resolved_selection.question_types:
            command.extend(["--question-type", question_type.value])
        for source_category in resolved_selection.source_categories:
            command.extend(["--source-category", source_category])
        for scenario_id in resolved_selection.scenario_ids:
            command.extend(["--scenario", scenario_id])
        for query_id in resolved_selection.query_ids:
            command.extend(["--query-id", query_id])
        if resolved_selection.metadata:
            command.extend(
                [
                    "--selection-metadata-json",
                    json.dumps(
                        resolved_selection.metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        if coverage_run_id is not None:
            command.extend(["--resume-coverage-run-id", coverage_run_id])
        return command

    def _run_qbank_coverage_stage(
        self,
        dataset: DatasetName,
        compiler_run_id: str,
        selected_count: int,
        *,
        selection: BenchmarkSelection | None = None,
    ) -> str:
        from pmmc.compiler.questioner_config import load_questioner_config
        from pmmc.evaluation.qbank_workflow import (
            QBankCoveragePhase,
            run_qbank_coverage,
            verify_qbank_coverage_run,
        )
        from pmmc.runtime.manager import run_managed_command

        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        stage_id = f"qbank_coverage:{dataset.value}"
        with self._stage(stage_id) as item:
            for runtime_stage, phase in (
                ("embedding_text", QBankCoveragePhase.EMBED_TEXT),
                ("embedding_multimodal", QBankCoveragePhase.EMBED_MULTIMODAL),
            ):
                run_managed_command(
                    self.launch_settings,
                    profile_path=self.launch_settings.resolved_runtime_profile_path,
                    logical_stage=runtime_stage,
                    command=self._coverage_command(
                        dataset,
                        compiler_run_id,
                        phase.value,
                        selection=selection,
                    ),
                    environment_overrides=self._child_environment(),
                )
            result = asyncio.run(
                run_qbank_coverage(
                    settings=self.launch_settings,
                    registry=self.resolved_registry,
                    config=load_questioner_config(
                        self.launch_settings.resolved_questioner_config_path
                    ),
                    compiler_run_id=compiler_run_id,
                    selection=(selection or self._selection_for_dataset(dataset)),
                    phase=QBankCoveragePhase.SCORE,
                )
            )
            report = asyncio.run(
                verify_qbank_coverage_run(self.launch_settings, result.coverage_run_id)
            )
            if not report["valid"]:
                raise ValueError("QBank coverage verification failed")
            assert self.reporter is not None
            self.reporter.advance(
                min(item.work_units, max(1, selected_count)), dataset=dataset.value
            )
            if self.reporter.stage_completed < item.work_units:
                self.reporter.advance(item.work_units - self.reporter.stage_completed)
            return result.coverage_run_id

    def _verify_questioner_dataset(
        self,
        dataset: DatasetName,
        compiler_run_id: str,
    ) -> dict[str, object]:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.questioner_repositories import (
            verify_questioner_run,
        )

        stage_id = f"questioner_verify:{dataset.value}"
        with self._stage(stage_id) as item:

            async def verify() -> dict[str, object]:
                engine, sessions = create_session_factory(
                    self.launch_settings,
                    DatabaseRole.RUNTIME,
                )
                try:
                    async with sessions() as session:
                        return await verify_questioner_run(
                            session,
                            compiler_run_id,
                            require_managed_runtime=True,
                        )
                finally:
                    await engine.dispose()

            report = asyncio.run(verify())
            if not report["valid"]:
                raise ValueError("Questioner run verification failed")
            assert self.reporter is not None
            self.reporter.advance(item.work_units, dataset=dataset.value)
            return report

    def _export_questioner_dataset(
        self,
        dataset: DatasetName,
        compiler_run_id: str,
        coverage_run_id: str | None,
    ) -> tuple[Path, Path | None, dict[str, object]]:
        from pmmc.compiler.questioner_export import verify_questioner_export
        from pmmc.evaluation.qbank_export import (
            QBankCoverageExporter,
            verify_qbank_coverage_export,
        )
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.experiment_export import ExperimentExporter
        from pmmc.infrastructure.object_store import create_runtime_object_store

        stage_id = f"export:{dataset.value}"
        with self._stage(stage_id) as item:

            async def export_runtime() -> Path:
                engine, sessions = create_session_factory(
                    self.launch_settings, DatabaseRole.RUNTIME
                )
                try:
                    result = await ExperimentExporter(
                        sessions,
                        runtime_store=create_runtime_object_store(self.launch_settings),
                        batch_size=self.launch_settings.experiment_export_batch_size,
                    ).export(
                        compiler_run_id,
                        self.launch_settings.experiment_outputs_path,
                        include_referenced_model_artifacts=True,
                    )
                    return result.export_dir
                finally:
                    await engine.dispose()

            runtime_export = asyncio.run(export_runtime())
            export_verification = verify_questioner_export(runtime_export)
            coverage_export = None
            if coverage_run_id is not None:
                result = asyncio.run(
                    QBankCoverageExporter(self.launch_settings).export(coverage_run_id)
                )
                verify_qbank_coverage_export(result.export_dir)
                coverage_export = result.export_dir
            assert self.reporter is not None
            self.reporter.advance(item.work_units, dataset=dataset.value)
            return runtime_export, coverage_export, export_verification

    def _reuse_completed_questioner_dataset(self, dataset: DatasetName) -> bool:
        if not self.resume:
            return False
        values = self.state.dataset_artifacts.get(dataset.value)
        if values is None:
            return False
        stages = [
            f"benchmark:{dataset.value}",
            f"snapshot_import:{dataset.value}",
            f"questioner:{dataset.value}",
            f"qbank_index_text:{dataset.value}",
            f"qbank_index_multimodal:{dataset.value}",
            f"questioner_verify:{dataset.value}",
        ]
        if self.config.evaluate:
            stages.append(f"qbank_coverage:{dataset.value}")
        stages.append(f"export:{dataset.value}")
        if not set(stages) <= set(self.state.completed_stage_ids):
            return False

        from pmmc.compiler.questioner_export import verify_questioner_export
        from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
        from pmmc.evaluation.qbank_workflow import verify_qbank_coverage_run
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.questioner_repositories import (
            verify_questioner_run,
        )

        async def verify_compiler() -> dict[str, object]:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with sessions() as session:
                    return await verify_questioner_run(
                        session,
                        str(values["compiler_run_id"]),
                        require_managed_runtime=True,
                    )
            finally:
                await engine.dispose()

        if not asyncio.run(verify_compiler())["valid"]:
            return False
        verify_questioner_export(Path(str(values["runtime_export_dir"])))
        if self.config.evaluate:
            coverage = asyncio.run(
                verify_qbank_coverage_run(self.launch_settings, str(values["coverage_run_id"]))
            )
            if not coverage["valid"]:
                return False
            verify_qbank_coverage_export(Path(str(values["coverage_export_dir"])))
        for stage in stages:
            self._reuse_stage(stage)
        return True

    def _run_existing_questioner_coverage(self) -> None:
        if not self.state.dataset_artifacts:
            raise ValueError("evaluate-only requires resumed Questioner artifacts")
        for dataset_name, values in sorted(self.state.dataset_artifacts.items()):
            dataset = DatasetName(dataset_name)
            compiler_run_id = str(values["compiler_run_id"])
            coverage_run_id = self._run_qbank_coverage_stage(
                dataset,
                compiler_run_id,
                int(values["selected_count"]),
            )
            _runtime, coverage_export, export_verification = self._export_questioner_dataset(
                dataset,
                compiler_run_id,
                coverage_run_id,
            )
            values["coverage_run_id"] = coverage_run_id
            values["coverage_export_dir"] = str(coverage_export)
            values["questioner_export_verification"] = export_verification
        self._save_state()

    def _run_datasets(self) -> None:
        if self.execution_config.schedule is ExecutionSchedule.MODEL_STAGE_MAJOR_V1:
            self._run_datasets_stage_major()
            return
        for dataset in self.config.selection.datasets:
            if self._reuse_completed_dataset(dataset):
                continue
            snapshot_path, snapshot_id = self._prepare_snapshot(dataset)
            corpus_path, corpus_id = self._prepare_corpus(dataset, snapshot_path)
            suite_id, selected_count = self._create_suite(dataset, snapshot_id, corpus_id)
            self._run_retrieval(dataset, suite_id, corpus_id, selected_count)
            self._run_answer(dataset, suite_id, selected_count)
            if self.config.evaluate:
                evaluation_run_id = self._run_evaluation(dataset, suite_id, selected_count)
                export_dir = self._export_evaluation(dataset, evaluation_run_id)
            else:
                evaluation_run_id = None
                export_dir = None
                self._finish_stage(f"export:{dataset.value}")
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                "snapshot_id": snapshot_id,
                "snapshot_path": str(snapshot_path),
                "corpus_id": corpus_id,
                "corpus_path": str(corpus_path),
                "suite_id": suite_id,
                "selected_count": selected_count,
                "evaluation_run_id": evaluation_run_id,
                "evaluation_export_dir": str(export_dir) if export_dir else None,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

    def _run_datasets_stage_major(self) -> None:
        from pmmc.baselines import (
            aggregate_baseline_family,
            export_baseline_component,
            import_baseline_component,
            register_baseline_family_from_suites,
            verify_baseline_component,
            verify_baseline_suite,
        )
        from pmmc.evaluation.workflow import (
            EvaluationRunPhase,
            run_evaluation_workflow,
            verify_evaluation_run,
        )
        from pmmc.runtime.manager import run_managed_commands

        component_matrix = bool(self.config.metadata.get("component_matrix"))
        if component_matrix:
            if self.config.mode is not AppMode.PAPER:
                raise ValueError("baseline component matrix requires paper mode")
            if set(self.config.selection.datasets) != {
                DatasetName.MEMLENS,
                DatasetName.MEMGALLERY,
            }:
                raise ValueError("baseline component matrix requires both benchmarks")
            if self.config.selection.partition_role is not PartitionRole.PAPER_TEST:
                raise ValueError("baseline component matrix requires --partition-role paper_test")
            if self.config.selection.partition_manifest_path is None:
                raise ValueError("baseline component matrix requires --partition-manifest")

        work: list[dict[str, object]] = []
        for dataset in self.config.selection.datasets:
            snapshot_path, snapshot_id = self._prepare_snapshot(dataset)
            corpus_path, corpus_id = self._prepare_corpus(dataset, snapshot_path)
            suite_id, selected_count = self._create_suite(
                dataset,
                snapshot_id,
                corpus_id,
                paper_result=False if component_matrix else None,
                extra_metadata={"baseline_family_source": True} if component_matrix else None,
            )
            item: dict[str, object] = {
                "dataset": dataset,
                "snapshot_id": snapshot_id,
                "snapshot_path": snapshot_path,
                "corpus_id": corpus_id,
                "corpus_path": corpus_path,
                "suite_id": suite_id,
                "selected_count": selected_count,
            }
            work.append(item)
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = {
                "snapshot_id": snapshot_id,
                "snapshot_path": str(snapshot_path),
                "corpus_id": corpus_id,
                "corpus_path": str(corpus_path),
                "suite_id": suite_id,
                "selected_count": selected_count,
                "execution_schedule": self.execution_config.schedule.value,
            }
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

        family_id: str | None = None
        if component_matrix:
            if self.preparation is None:
                raise RuntimeError("baseline family requires a resolved model lock")
            family, components = asyncio.run(
                register_baseline_family_from_suites(
                    self.launch_settings,
                    suite_ids={_work_dataset(item): str(item["suite_id"]) for item in work},
                    model_lock_path=self.preparation.lock_path,
                    judge_model_alias=self.config.models.judge_model,
                    metadata={
                        "launcher_id": self.launch_id,
                        "execution_schedule": self.execution_config.schedule.value,
                    },
                )
            )
            family_id = family.family_id
            by_dataset: dict[DatasetName, dict[BaselineMethod, str]] = {}
            for component in components:
                by_dataset.setdefault(component.dataset, {})[component.method] = (
                    component.component_id
                )
            for item in work:
                dataset = _work_dataset(item)
                item["components"] = by_dataset[dataset]
                values = dict(self.state.dataset_artifacts[dataset.value])
                values["baseline_family_id"] = family_id
                values["source_suite_id"] = values.pop("suite_id")
                values["component_ids"] = {
                    method.value: component_id
                    for method, component_id in sorted(
                        by_dataset[dataset].items(), key=lambda value: value[0].value
                    )
                }
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset.value] = values
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            metadata = dict(self.state.metadata)
            metadata["baseline_family_id"] = family_id
            self.state = self.state.model_copy(update={"metadata": metadata})
            self._save_state()

        rag_stages = {
            BaselineMethod.CAPTION_TEXT_RAG: "embedding_text",
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG: "embedding_contrastive",
            BaselineMethod.NATIVE_MULTIMODAL_RAG: "embedding_multimodal",
        }
        for method, runtime_stage in rag_stages.items():
            if method not in self.config.methods:
                continue
            stage_id = f"retrieval:{method.value}:model-stage"
            total = sum(_work_int(item, "selected_count") for item in work)
            with self._stage(stage_id):
                commands: list[list[str]] = []
                command_work: list[tuple[dict[str, object], str]] = []
                for item in work:
                    retrieve_command = [
                        sys.executable,
                        "-m",
                        "pmmc",
                        "baseline",
                        "run-suite",
                        "--suite-id",
                        str(item["suite_id"]),
                        "--phase",
                        "retrieve",
                        "--method",
                        method.value,
                    ]
                    if component_matrix:
                        component_ids = item.get("components")
                        if not isinstance(component_ids, dict):
                            raise RuntimeError("baseline component matrix is incomplete")
                        retrieve_command = [
                            sys.executable,
                            "-m",
                            "pmmc",
                            "baseline",
                            "run-component",
                            "--component-id",
                            str(component_ids[method]),
                            "--phase",
                            "retrieve",
                            "--managed-child",
                        ]
                    commands.extend(
                        [
                            [
                                sys.executable,
                                "-m",
                                "pmmc",
                                "baseline",
                                "build-index",
                                "--method",
                                method.value,
                                "--corpus-id",
                                str(item["corpus_id"]),
                            ],
                            [
                                sys.executable,
                                "-m",
                                "pmmc",
                                "baseline",
                                "verify-index",
                                "--method",
                                method.value,
                                "--corpus-id",
                                str(item["corpus_id"]),
                            ],
                            retrieve_command,
                        ]
                    )
                    command_work.extend([(item, "index"), (item, "verify"), (item, "retrieve")])

                def retrieval_completed(
                    index: int,
                    _command: list[str],
                    *,
                    _command_work: list[tuple[dict[str, object], str]] = command_work,
                    _method: BaselineMethod = method,
                ) -> None:
                    item, command_phase = _command_work[index - 1]
                    if command_phase != "retrieve":
                        return
                    if component_matrix:
                        component_ids = item.get("components")
                        if not isinstance(component_ids, dict):
                            raise RuntimeError("baseline component matrix is incomplete")
                        component_id = str(component_ids[_method])
                        report = asyncio.run(
                            verify_baseline_component(
                                self.launch_settings,
                                component_id,
                            )
                        )
                        if not report["valid"] or report["status"] != "retrieval_ready":
                            raise ValueError("baseline retrieval component verification failed")
                        dataset = _work_dataset(item)
                        values = dict(self.state.dataset_artifacts[dataset.value])
                        statuses = dict(values.get("component_states", {}))
                        statuses[_method.value] = "retrieval_ready"
                        values["component_states"] = statuses
                        artifacts = dict(self.state.dataset_artifacts)
                        artifacts[dataset.value] = values
                        self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                        self._save_state()
                        self._maybe_checkpoint(f"retrieval:component:{component_id}")
                    if self.reporter is not None:
                        self.reporter.advance(
                            _work_int(item, "selected_count"),
                            dataset=str(item["dataset"]),
                            method=_method.value,
                        )

                run_managed_commands(
                    self.launch_settings,
                    profile_path=self.launch_settings.resolved_runtime_profile_path,
                    logical_stage=runtime_stage,
                    commands=commands,
                    environment_overrides=self._child_environment(),
                    command_completed=retrieval_completed,
                    replica_count=self._replica_count(runtime_stage),
                )
                self._set_stage_performance_metadata(
                    stage_id,
                    {
                        "dataset_count": len(work),
                        "command_count": len(commands),
                        "query_count": total,
                        "method": method.value,
                    },
                )
                if self.reporter is not None and self.reporter.stage_completed < total:
                    self.reporter.advance(total - self.reporter.stage_completed)

        answer_total = sum(
            _work_int(item, "selected_count") * len(self.config.methods) for item in work
        )
        with self._stage("answer:model-stage"):
            answer_work: list[tuple[dict[str, object], BaselineMethod | None, str | None]] = []
            answer_commands: list[list[str]] = []
            if component_matrix:
                for item in work:
                    component_ids = item.get("components")
                    if not isinstance(component_ids, dict):
                        raise RuntimeError("baseline component matrix is incomplete")
                    for method in self.config.methods:
                        component_id = str(component_ids[method])
                        answer_commands.append(
                            [
                                sys.executable,
                                "-m",
                                "pmmc",
                                "baseline",
                                "run-component",
                                "--component-id",
                                component_id,
                                "--phase",
                                "answer",
                                "--managed-child",
                            ]
                        )
                        answer_work.append((item, method, component_id))
            else:
                for item in work:
                    answer_commands.append(
                        [
                            sys.executable,
                            "-m",
                            "pmmc",
                            "baseline",
                            "run-suite",
                            "--suite-id",
                            str(item["suite_id"]),
                            "--phase",
                            "answer",
                        ]
                    )
                    answer_work.append((item, None, None))

            def answer_completed(index: int, _command: list[str]) -> None:
                item, method, component_id = answer_work[index - 1]
                if component_id is not None:
                    bundle_dir, _bundle = asyncio.run(
                        export_baseline_component(
                            self.launch_settings,
                            component_id=component_id,
                            output_root=self.settings.runs_path / "baseline-components",
                        )
                    )
                    asyncio.run(
                        import_baseline_component(
                            self.launch_settings,
                            bundle_dir=bundle_dir,
                            imported_by=f"launcher:{self.launch_id}",
                        )
                    )
                    report = asyncio.run(
                        verify_baseline_component(
                            self.launch_settings,
                            component_id,
                            bundle_dir=bundle_dir,
                        )
                    )
                    if not report["valid"] or report["status"] != "verified":
                        raise ValueError("baseline component verification failed")
                    dataset = _work_dataset(item)
                    values = dict(self.state.dataset_artifacts[dataset.value])
                    bundles = dict(values.get("component_bundles", {}))
                    assert method is not None
                    bundles[method.value] = str(bundle_dir)
                    values["component_bundles"] = bundles
                    artifacts = dict(self.state.dataset_artifacts)
                    artifacts[dataset.value] = values
                    self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                    self._save_state()
                    self._maybe_checkpoint(f"answer:component:{component_id}")
                else:
                    report = asyncio.run(
                        verify_baseline_suite(self.launch_settings, str(item["suite_id"]))
                    )
                    if not report["valid"]:
                        raise ValueError("baseline suite verification failed")
                if self.reporter is not None:
                    self.reporter.advance(
                        _work_int(item, "selected_count")
                        * (1 if component_id is not None else len(self.config.methods)),
                        dataset=str(item["dataset"]),
                        method=method.value if method is not None else None,
                    )

            run_managed_commands(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="answer",
                commands=answer_commands,
                environment_overrides=self._child_environment(),
                command_completed=answer_completed,
                replica_count=self._replica_count("answer"),
            )
            self._set_stage_performance_metadata(
                "answer:model-stage",
                {
                    "dataset_count": len(work),
                    "command_count": len(answer_commands),
                    "job_count": answer_total,
                    "model_alias": self.config.models.answer_model,
                },
            )
            if self.reporter is not None and self.reporter.stage_completed < answer_total:
                self.reporter.advance(answer_total - self.reporter.stage_completed)

        if component_matrix:
            if family_id is None:
                raise RuntimeError("baseline component matrix has no family identity")
            aggregate = asyncio.run(aggregate_baseline_family(self.launch_settings, family_id))
            aggregate_suites = dict(aggregate["suite_ids"])
            for item in work:
                dataset = _work_dataset(item)
                item["suite_id"] = str(aggregate_suites[dataset.value])
                values = dict(self.state.dataset_artifacts[dataset.value])
                values["suite_id"] = item["suite_id"]
                values["paper_result"] = True
                artifacts = dict(self.state.dataset_artifacts)
                artifacts[dataset.value] = values
                self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

        if self.config.evaluate:
            with self._stage("evaluation_deterministic:model-stage"):
                for item in work:
                    manifest, _directory = asyncio.run(
                        run_evaluation_workflow(
                            settings=self.launch_settings,
                            suite_id=str(item["suite_id"]),
                            phase=EvaluationRunPhase.DETERMINISTIC,
                        )
                    )
                    item["evaluation_run_id"] = manifest.evaluation_run_id
                    values = dict(self.state.dataset_artifacts[_work_dataset(item).value])
                    values["evaluation_run_id"] = manifest.evaluation_run_id
                    artifacts = dict(self.state.dataset_artifacts)
                    artifacts[_work_dataset(item).value] = values
                    self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                    self._save_state()
                    if self.reporter is not None:
                        self.reporter.advance(
                            _work_int(item, "selected_count") * len(self.config.methods),
                            dataset=str(item["dataset"]),
                        )

            with self._stage("evaluation_judge:model-stage"):
                judge_commands = [
                    [
                        sys.executable,
                        "-m",
                        "pmmc",
                        "evaluate",
                        "run",
                        "--resume-evaluation-run-id",
                        str(item["evaluation_run_id"]),
                        "--phase",
                        "judge",
                    ]
                    for item in work
                ]

                def judge_completed(index: int, _command: list[str]) -> None:
                    item = work[index - 1]
                    if self.reporter is not None:
                        self.reporter.advance(
                            _work_int(item, "selected_count") * len(self.config.methods),
                            dataset=str(item["dataset"]),
                        )

                run_managed_commands(
                    self.launch_settings,
                    profile_path=self.launch_settings.resolved_runtime_profile_path,
                    logical_stage="judge",
                    commands=judge_commands,
                    environment_overrides=self._child_environment(),
                    command_completed=judge_completed,
                    replica_count=self._replica_count("judge"),
                )
                self._set_stage_performance_metadata(
                    "evaluation_judge:model-stage",
                    {
                        "dataset_count": len(work),
                        "command_count": len(judge_commands),
                        "job_count": sum(
                            _work_int(item, "selected_count") * len(self.config.methods)
                            for item in work
                        ),
                        "model_alias": self.config.models.judge_model,
                    },
                )

            with self._stage("evaluation_analyze:model-stage"):
                for item in work:
                    asyncio.run(
                        run_evaluation_workflow(
                            settings=self.launch_settings,
                            resume_evaluation_run_id=str(item["evaluation_run_id"]),
                            phase=EvaluationRunPhase.ANALYZE,
                        )
                    )
                    report = asyncio.run(
                        verify_evaluation_run(self.launch_settings, str(item["evaluation_run_id"]))
                    )
                    if not report["valid"]:
                        raise ValueError("evaluation run verification failed")
                    if self.reporter is not None:
                        self.reporter.advance(
                            _work_int(item, "selected_count") * len(self.config.methods),
                            dataset=str(item["dataset"]),
                        )

        for item in work:
            dataset = _work_dataset(item)
            evaluation_run_id = item.get("evaluation_run_id")
            export_dir = (
                self._export_evaluation(dataset, str(evaluation_run_id))
                if evaluation_run_id is not None
                else None
            )
            if export_dir is None:
                self._finish_stage(f"export:{dataset.value}")
            values = dict(self.state.dataset_artifacts[dataset.value])
            values["evaluation_export_dir"] = str(export_dir) if export_dir else None
            artifacts = dict(self.state.dataset_artifacts)
            artifacts[dataset.value] = values
            self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
            self._save_state()

    def _reuse_completed_dataset(self, dataset: DatasetName) -> bool:
        if not self.resume:
            return False
        values = self.state.dataset_artifacts.get(dataset.value)
        if values is None:
            return False
        stage_ids = [
            f"benchmark:{dataset.value}",
            f"snapshot_import:{dataset.value}",
            f"corpus:{dataset.value}",
            f"retrieval:{dataset.value}",
            f"answer:{dataset.value}",
        ]
        if self.config.evaluate:
            stage_ids.append(f"evaluation:{dataset.value}")
        stage_ids.append(f"export:{dataset.value}")
        if not set(stage_ids) <= set(self.state.completed_stage_ids):
            return False

        from pmmc.baselines import verify_baseline_suite

        suite_id = str(values["suite_id"])
        suite_report = asyncio.run(verify_baseline_suite(self.launch_settings, suite_id))
        if not suite_report["valid"]:
            return False
        if self.config.evaluate:
            from pmmc.evaluation.export import verify_evaluation_export
            from pmmc.evaluation.workflow import verify_evaluation_run

            evaluation_run_id = str(values["evaluation_run_id"])
            evaluation_report = asyncio.run(
                verify_evaluation_run(self.launch_settings, evaluation_run_id)
            )
            if not evaluation_report["valid"]:
                return False
            verify_evaluation_export(Path(str(values["evaluation_export_dir"])))
        for stage_id in stage_ids:
            self._reuse_stage(stage_id)
        return True

    def _rehydrate_restored_snapshot_storage(
        self,
        *,
        dataset: DatasetName,
        snapshot_path: Path,
        expected_asset_count: int,
    ) -> None:
        """Rebuild ephemeral object-store state omitted from a checkpoint."""

        archive_enabled = self.config.infrastructure.archive_snapshot
        assets_enabled = self.config.infrastructure.sync_assets
        if not archive_enabled and not assets_enabled:
            return
        if self._step6d_resume_verification_only():
            print(
                "CHECKPOINT_SNAPSHOT_STORAGE_REHYDRATION "
                + json.dumps(
                    {
                        "dataset": dataset.value,
                        "expected_asset_count": expected_asset_count,
                        "reason": "same_pod_assets_verified_by_first_pass",
                        "state": "reused_in_place",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            return

        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.storage_updates import (
            persist_asset_sync,
            persist_snapshot_archive,
        )
        from pmmc.infrastructure.object_store import create_admin_object_stores
        from pmmc.infrastructure.storage_sync import archive_snapshot, sync_assets

        if self.reporter is not None:
            self.reporter.report_operation(f"snapshot_storage_restore:{dataset.value}")
        started_at = time.monotonic()
        phase_started_at: dict[str, float] = {}
        phase_last_reported_at: dict[str, float] = {}
        print(
            "CHECKPOINT_SNAPSHOT_STORAGE_REHYDRATION "
            + json.dumps(
                {
                    "archive_snapshot": archive_enabled,
                    "dataset": dataset.value,
                    "expected_asset_count": expected_asset_count,
                    "state": "starting",
                    "sync_assets": assets_enabled,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

        def progress(phase: str, current: int, total: int) -> None:
            now = time.monotonic()
            phase_start = phase_started_at.setdefault(phase, now)
            last_reported = phase_last_reported_at.get(phase, phase_start)
            if current < total and now - last_reported < 30.0:
                return
            phase_last_reported_at[phase] = now
            elapsed = max(now - phase_start, 0.0)
            print(
                "CHECKPOINT_SNAPSHOT_STORAGE_REHYDRATION "
                + json.dumps(
                    {
                        "completed": current,
                        "dataset": dataset.value,
                        "elapsed_seconds": round(elapsed, 3),
                        "items_per_second": (
                            round(current / elapsed, 3) if elapsed > 0.0 else None
                        ),
                        "phase": phase,
                        "state": "progress",
                        "total": total,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

        async def rehydrate() -> dict[str, int]:
            engine = create_engine(self.launch_settings, DatabaseRole.ADMIN)
            runtime_store, evaluation_store = create_admin_object_stores(self.launch_settings)
            runtime_object_count = 0
            evaluation_object_count = 0
            synced_asset_count = 0
            synced_asset_bytes = 0
            skipped_remote_only = 0
            try:
                if archive_enabled:
                    archived = await archive_snapshot(
                        snapshot_path,
                        runtime_store=runtime_store,
                        evaluation_store=evaluation_store,
                        progress=progress,
                    )
                    await persist_snapshot_archive(engine, archived)
                    runtime_object_count = len(archived.runtime_objects)
                    evaluation_object_count = len(archived.evaluation_objects)
                if assets_enabled:
                    synchronized = await sync_assets(
                        snapshot_path,
                        project_root=self.settings.benchmark_asset_path,
                        runtime_store=runtime_store,
                        progress=progress,
                    )
                    synchronized_total = len(synchronized.synced) + synchronized.skipped_remote_only
                    if synchronized_total != expected_asset_count:
                        raise RuntimeError(
                            "restored snapshot asset synchronization count is incomplete"
                        )
                    await persist_asset_sync(engine, synchronized)
                    synced_asset_count = len(synchronized.synced)
                    synced_asset_bytes = sum(item.size_bytes for item in synchronized.synced)
                    skipped_remote_only = synchronized.skipped_remote_only
            finally:
                await engine.dispose()
            return {
                "evaluation_object_count": evaluation_object_count,
                "runtime_object_count": runtime_object_count,
                "skipped_remote_only": skipped_remote_only,
                "synced_asset_bytes": synced_asset_bytes,
                "synced_asset_count": synced_asset_count,
            }

        summary = asyncio.run(rehydrate())
        elapsed = max(time.monotonic() - started_at, 0.0)
        print(
            "CHECKPOINT_SNAPSHOT_STORAGE_REHYDRATION "
            + json.dumps(
                {
                    "dataset": dataset.value,
                    "elapsed_seconds": round(elapsed, 3),
                    "state": "ready",
                    **summary,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    def _prepare_snapshot(self, dataset: DatasetName) -> tuple[Path, str]:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.infrastructure.database.importer import CanonicalSnapshotImporter
        from pmmc.infrastructure.database.storage_updates import (
            persist_asset_sync,
            persist_snapshot_archive,
        )
        from pmmc.infrastructure.object_store import create_admin_object_stores
        from pmmc.infrastructure.storage_sync import archive_snapshot, sync_assets

        stage_id = f"benchmark:{dataset.value}"
        import_stage = f"snapshot_import:{dataset.value}"
        restored = self.state.dataset_artifacts.get(dataset.value, {})
        if (
            self.resume
            and self._checkpoint_restored
            and stage_id in self.state.completed_stage_ids
            and import_stage in self.state.completed_stage_ids
            and restored.get("snapshot_path") is not None
            and restored.get("snapshot_id") is not None
        ):
            from pmmc.benchmarks.snapshot import load_manifest

            snapshot_path = Path(str(restored["snapshot_path"]))
            if not snapshot_path.is_dir():
                raise FileNotFoundError(
                    "restored checkpoint benchmark snapshot directory is missing"
                )
            manifest = load_manifest(snapshot_path)
            if manifest.snapshot_id != str(restored["snapshot_id"]):
                raise ValueError("restored checkpoint benchmark snapshot identity changed")
            # Checkpoints intentionally omit canonical benchmark objects because
            # they are reproducible from the mounted benchmark. A new
            # self-contained Pod therefore has an empty MinIO even though the
            # database still contains the restored storage lineage.
            self._rehydrate_restored_snapshot_storage(
                dataset=dataset,
                snapshot_path=snapshot_path,
                expected_asset_count=int(manifest.counts.get("assets", 0)),
            )
            self._reuse_restored_stage(stage_id)
            self._reuse_restored_stage(import_stage)
            return snapshot_path, manifest.snapshot_id

        item = self.plan.get(stage_id)
        with self._stage(stage_id):
            root = (
                self.settings.memlens_path
                if dataset is DatasetName.MEMLENS
                else self.settings.memgallery_path
            )
            adapter = create_adapter(
                dataset,
                root=root,
                project_root=self.settings.benchmark_asset_path,
            )
            phase_counts: dict[str, int] = {}

            def progress(phase: str, current: int, _total: int | None) -> None:
                previous = phase_counts.get(phase, 0)
                delta = max(0, current - previous)
                phase_counts[phase] = current
                if delta and self.reporter is not None:
                    self.reporter.update_subtask(
                        phase,
                        current=current,
                        total=max(1, _total or item.work_units),
                    )
                    self.reporter.advance(delta, dataset=dataset.value, phase=phase)

            level = (
                ValidationLevel.FULL if self.config.mode is AppMode.PAPER else ValidationLevel.FAST
            )
            snapshot_path, manifest = build_snapshot(
                adapter,
                output_root=self.launch_settings.benchmark_artifacts_path,
                level=level,
                progress=progress,
            )
            restored = self.state.dataset_artifacts.get(dataset.value, {})
            restored_snapshot_id = restored.get("snapshot_id")
            if (
                self.resume
                and self._checkpoint_restored
                and restored_snapshot_id is not None
                and str(restored_snapshot_id) != manifest.snapshot_id
            ):
                raise ValueError("resumed checkpoint snapshot differs from the current benchmark")
            if self.reporter is not None and self.reporter.stage_completed < item.work_units:
                self.reporter.advance(item.work_units - self.reporter.stage_completed)

        with self._stage(import_stage):

            async def persist() -> None:
                engine = create_engine(self.launch_settings, DatabaseRole.ADMIN)
                try:
                    importer = CanonicalSnapshotImporter(
                        engine,
                        batch_size=self.launch_settings.database_import_batch_size,
                    )
                    await importer.import_snapshot(snapshot_path)
                    if (
                        self.config.infrastructure.sync_assets
                        or self.config.infrastructure.archive_snapshot
                    ):
                        runtime_store, evaluation_store = create_admin_object_stores(
                            self.launch_settings
                        )
                    if self.config.infrastructure.archive_snapshot:
                        archived = await archive_snapshot(
                            snapshot_path,
                            runtime_store=runtime_store,
                            evaluation_store=evaluation_store,
                        )
                        await persist_snapshot_archive(engine, archived)
                    if self.config.infrastructure.sync_assets:
                        result = await sync_assets(
                            snapshot_path,
                            project_root=self.settings.benchmark_asset_path,
                            runtime_store=runtime_store,
                        )
                        await persist_asset_sync(engine, result)
                finally:
                    await engine.dispose()

            asyncio.run(persist())
        return snapshot_path, manifest.snapshot_id

    def _prepare_corpus(self, dataset: DatasetName, snapshot_path: Path) -> tuple[Path, str]:
        from pmmc.baselines import build_corpus
        from pmmc.infrastructure.database.baseline_repositories import BaselineCorpusImporter
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_engine
        from pmmc.models import (
            HeuristicTokenCounter,
            TokenCounter,
            TransformersTokenCounter,
        )

        if self.resolved_registry is None:
            raise RuntimeError("resolved model registry is unavailable")
        if self.preparation is None:
            raise RuntimeError("model preparation is unavailable")
        stage_id = f"corpus:{dataset.value}"
        item = self.plan.get(stage_id)
        with self._stage(stage_id):
            entry = self.resolved_registry.get(self.config.models.text_embedding_model)
            counter: TokenCounter
            operation = "tokenizer_load"
            started_at = time.monotonic()
            try:
                if self.config.mode is AppMode.PAPER:
                    manifest_path = self.preparation.verified_cache_manifest_path
                    if manifest_path is None or not manifest_path.is_file():
                        raise FileNotFoundError(
                            "verified model-cache manifest is unavailable for corpus tokenization"
                        )
                    from pmmc.schemas import VerifiedModelCacheManifest

                    cache_manifest = VerifiedModelCacheManifest.from_json(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    alias = self.config.models.text_embedding_model
                    verification = cache_manifest.verifications.get(alias)
                    snapshot_value = cache_manifest.snapshot_paths.get(alias)
                    if verification is None or snapshot_value is None:
                        raise KeyError(
                            "verified model-cache manifest lacks the corpus tokenizer alias"
                        )
                    if (
                        verification.repo_id != entry.resolved_source_repo_id
                        or verification.commit_sha != entry.model_revision
                    ):
                        raise ValueError(
                            "corpus tokenizer model identity differs from verification"
                        )
                    tokenizer_snapshot = Path(snapshot_value).resolve(strict=True)
                    counter = TransformersTokenCounter(
                        entry.tokenizer_name,
                        entry.tokenizer_revision,
                        local_files_only=True,
                        cache_dir=self.launch_settings.model_cache_path,
                        snapshot_path=tokenizer_snapshot,
                    )
                else:
                    counter = HeuristicTokenCounter(entry.tokenizer_name, entry.tokenizer_revision)
                print(
                    "CORPUS_PREP "
                    + json.dumps(
                        {
                            "dataset": dataset.value,
                            "elapsed_seconds": round(time.monotonic() - started_at, 3),
                            "operation": operation,
                            "paper_mode": self.config.mode is AppMode.PAPER,
                            "state": "ready",
                            "tokenizer_source": (
                                "verified_snapshot"
                                if self.config.mode is AppMode.PAPER
                                else "heuristic"
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                phase_counts: dict[str, int] = {}

                def progress(phase: str, current: int, _total: int | None) -> None:
                    previous = phase_counts.get(phase, 0)
                    phase_counts[phase] = current
                    delta = max(0, current - previous)
                    if delta and self.reporter is not None:
                        self.reporter.update_subtask(
                            phase,
                            current=current,
                            total=max(1, _total or item.work_units),
                        )
                        self.reporter.advance(delta, dataset=dataset.value, phase=phase)

                operation = "build_or_reuse"
                corpus_path, manifest = build_corpus(
                    snapshot_path,
                    output_root=self.launch_settings.baseline_corpora_path,
                    counter=counter,
                    progress=progress,
                )
                print(
                    "CORPUS_PREP "
                    + json.dumps(
                        {
                            "dataset": dataset.value,
                            "document_count": manifest.document_count,
                            "elapsed_seconds": round(time.monotonic() - started_at, 3),
                            "operation": operation,
                            "state": "ready",
                            "window_count": manifest.window_count,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

                async def persist() -> None:
                    engine = create_engine(self.launch_settings, DatabaseRole.ADMIN)
                    try:
                        await BaselineCorpusImporter(
                            engine,
                            batch_size=self.launch_settings.database_import_batch_size,
                        ).import_corpus(corpus_path)
                    finally:
                        await engine.dispose()

                operation = "database_import"
                asyncio.run(persist())
                print(
                    "CORPUS_PREP "
                    + json.dumps(
                        {
                            "dataset": dataset.value,
                            "elapsed_seconds": round(time.monotonic() - started_at, 3),
                            "operation": operation,
                            "state": "ready",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                if _is_step7_target(self.config.target):
                    artifacts = dict(self.state.dataset_artifacts)
                    artifacts[dataset.value] = {
                        **dict(artifacts.get(dataset.value, {})),
                        "fallback_corpus_id": manifest.corpus_id,
                        "fallback_corpus_path": str(corpus_path),
                    }
                    self.state = self.state.model_copy(update={"dataset_artifacts": artifacts})
                    self._save_state()
                    print(
                        "CORPUS_PREP "
                        + json.dumps(
                            {
                                "dataset": dataset.value,
                                "operation": "checkpoint_state",
                                "state": "ready",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                if self.reporter is not None and self.reporter.stage_completed < item.work_units:
                    self.reporter.advance(item.work_units - self.reporter.stage_completed)
            except Exception as exc:
                from pmmc.infrastructure.database.errors import (
                    safe_database_error_diagnostics,
                )

                wrapped = CorpusPreparationError(stage=stage_id, operation=operation)
                wrapped.__cause__ = exc
                print(
                    "CORPUS_PREP_FAILURE "
                    + json.dumps(
                        {
                            "dataset": dataset.value,
                            "diagnostics": safe_database_error_diagnostics(wrapped),
                            "elapsed_seconds": round(time.monotonic() - started_at, 3),
                            "operation": operation,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                raise wrapped from exc
        return corpus_path, manifest.corpus_id

    def _create_suite(
        self,
        dataset: DatasetName,
        snapshot_id: str,
        corpus_id: str,
        *,
        methods: list[BaselineMethod] | None = None,
        selection: BenchmarkSelection | None = None,
        paper_result: bool | None = None,
        extra_metadata: dict[str, object] | None = None,
        require_existing: bool = False,
    ) -> tuple[str, int]:
        from pmmc.baselines import create_baseline_suite
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.evaluation_repositories import (
            PostgresBaselineSuiteRepository,
        )

        resolved_selection = selection or self._selection_for_dataset(
            dataset,
            snapshot_id=snapshot_id,
        )
        if resolved_selection.datasets != [dataset]:
            raise ValueError("baseline suite selection differs from its dataset")
        partition_metadata: dict[str, object] = {
            key: resolved_selection.metadata[key]
            for key in (
                "partition_id",
                "partition_role",
                "partition_dataset_assignment_hash",
            )
            if key in resolved_selection.metadata
        }
        manifest, _directory = asyncio.run(
            create_baseline_suite(
                settings=self.launch_settings,
                snapshot_id=snapshot_id,
                selection=resolved_selection,
                corpus_id=corpus_id,
                methods=methods or self.config.methods,
                answer_model_alias=self.config.models.answer_model,
                require_existing=require_existing,
                metadata={
                    "launcher_id": self.launch_id,
                    "paper_result": (
                        self.config.mode is AppMode.PAPER if paper_result is None else paper_result
                    ),
                    "model_registry_subset_hash": (
                        self.preparation.registry_subset_hash if self.preparation else None
                    ),
                    **partition_metadata,
                    **dict(extra_metadata or {}),
                },
            )
        )

        async def count() -> int:
            engine, sessions = create_session_factory(self.launch_settings, DatabaseRole.RUNTIME)
            try:
                async with sessions() as session:
                    rows = await PostgresBaselineSuiteRepository(session).get_queries(
                        manifest.suite_id
                    )
                    return len(rows)
            finally:
                await engine.dispose()

        return manifest.suite_id, asyncio.run(count())

    def _run_retrieval(
        self,
        dataset: DatasetName,
        suite_id: str,
        corpus_id: str,
        selected_count: int,
    ) -> None:
        from pmmc.runtime.manager import run_managed_commands

        stage_id = f"retrieval:{dataset.value}"
        rag_stages = {
            BaselineMethod.CAPTION_TEXT_RAG: "embedding_text",
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG: "embedding_contrastive",
            BaselineMethod.NATIVE_MULTIMODAL_RAG: "embedding_multimodal",
        }
        with self._stage(stage_id):
            for method, runtime_stage in rag_stages.items():
                if method not in self.config.methods:
                    continue
                if self.reporter is not None:
                    self.reporter.update_subtask(
                        method.value, current=0, total=max(1, selected_count)
                    )
                commands = [
                    [
                        sys.executable,
                        "-m",
                        "pmmc",
                        "baseline",
                        "build-index",
                        "--method",
                        method.value,
                        "--corpus-id",
                        corpus_id,
                    ],
                    [
                        sys.executable,
                        "-m",
                        "pmmc",
                        "baseline",
                        "run-suite",
                        "--suite-id",
                        suite_id,
                        "--phase",
                        "retrieve",
                        "--method",
                        method.value,
                    ],
                ]
                run_managed_commands(
                    self.launch_settings,
                    profile_path=self.launch_settings.resolved_runtime_profile_path,
                    logical_stage=runtime_stage,
                    commands=commands,
                    environment_overrides=self._child_environment(),
                    replica_count=self._replica_count(runtime_stage),
                )
                if self.reporter is not None:
                    self.reporter.update_subtask(
                        method.value,
                        current=selected_count,
                        total=max(1, selected_count),
                    )
                    self.reporter.advance(
                        selected_count,
                        dataset=dataset.value,
                        method=method.value,
                    )

    def _run_answer(self, dataset: DatasetName, suite_id: str, selected_count: int) -> None:
        from pmmc.baselines import verify_baseline_suite
        from pmmc.runtime.manager import run_managed_command

        stage_id = f"answer:{dataset.value}"
        with self._stage(stage_id):
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "answer_jobs",
                    current=0,
                    total=max(1, selected_count * len(self.config.methods)),
                )
            command = [
                sys.executable,
                "-m",
                "pmmc",
                "baseline",
                "run-suite",
                "--suite-id",
                suite_id,
                "--phase",
                "answer",
            ]
            run_managed_command(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="answer",
                command=command,
                environment_overrides=self._child_environment(),
            )
            report = asyncio.run(verify_baseline_suite(self.launch_settings, suite_id))
            if not report["valid"]:
                raise ValueError("baseline suite verification failed")
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "answer_jobs",
                    current=selected_count * len(self.config.methods),
                    total=max(1, selected_count * len(self.config.methods)),
                )
                self.reporter.advance(
                    selected_count * len(self.config.methods), dataset=dataset.value
                )

    def _run_evaluation(self, dataset: DatasetName, suite_id: str, selected_count: int) -> str:
        from pmmc.evaluation.workflow import (
            EvaluationRunPhase,
            run_evaluation_workflow,
            verify_evaluation_run,
        )
        from pmmc.runtime.manager import run_managed_command

        stage_id = f"evaluation:{dataset.value}"
        with self._stage(stage_id):
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "deterministic",
                    current=0,
                    total=max(1, selected_count * len(self.config.methods)),
                )
            manifest, _directory = asyncio.run(
                run_evaluation_workflow(
                    settings=self.launch_settings,
                    suite_id=suite_id,
                    phase=EvaluationRunPhase.DETERMINISTIC,
                )
            )
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "deterministic",
                    current=selected_count * len(self.config.methods),
                    total=max(1, selected_count * len(self.config.methods)),
                )
                self.reporter.advance(
                    selected_count * len(self.config.methods),
                    dataset=dataset.value,
                    phase="deterministic",
                )
            command = [
                sys.executable,
                "-m",
                "pmmc",
                "evaluate",
                "run",
                "--resume-evaluation-run-id",
                manifest.evaluation_run_id,
                "--phase",
                "judge",
            ]
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "judge",
                    current=0,
                    total=max(1, selected_count * len(self.config.methods)),
                )
            run_managed_command(
                self.launch_settings,
                profile_path=self.launch_settings.resolved_runtime_profile_path,
                logical_stage="judge",
                command=command,
                environment_overrides=self._child_environment(),
            )
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "judge",
                    current=selected_count * len(self.config.methods),
                    total=max(1, selected_count * len(self.config.methods)),
                )
                self.reporter.advance(
                    selected_count * len(self.config.methods),
                    dataset=dataset.value,
                    phase="judge",
                )
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "analyze",
                    current=0,
                    total=max(1, selected_count * len(self.config.methods)),
                )
            asyncio.run(
                run_evaluation_workflow(
                    settings=self.launch_settings,
                    resume_evaluation_run_id=manifest.evaluation_run_id,
                    phase=EvaluationRunPhase.ANALYZE,
                )
            )
            if self.reporter is not None:
                self.reporter.update_subtask(
                    "analyze",
                    current=selected_count * len(self.config.methods),
                    total=max(1, selected_count * len(self.config.methods)),
                )
                self.reporter.advance(
                    selected_count * len(self.config.methods),
                    dataset=dataset.value,
                    phase="analyze",
                )
            report = asyncio.run(
                verify_evaluation_run(self.launch_settings, manifest.evaluation_run_id)
            )
            if not report["valid"]:
                raise ValueError("evaluation run verification failed")
            return manifest.evaluation_run_id

    def _export_evaluation(self, dataset: DatasetName, evaluation_run_id: str) -> Path:
        from pmmc.evaluation.export import EvaluationExporter, verify_evaluation_export

        stage_id = f"export:{dataset.value}"
        with self._stage(stage_id) as item:
            result = asyncio.run(EvaluationExporter(self.launch_settings).export(evaluation_run_id))
            verify_evaluation_export(result.export_dir)
            assert self.reporter is not None
            self.reporter.advance(item.work_units, dataset=dataset.value)
            return result.export_dir

    def _run_existing_evaluations(self) -> None:
        if not self.state.dataset_artifacts:
            raise ValueError("evaluate-only requires resumed dataset suite artifacts")
        for dataset_name, values in sorted(self.state.dataset_artifacts.items()):
            dataset = DatasetName(dataset_name)
            suite_id = str(values["suite_id"])
            selected_count = int(values["selected_count"])
            evaluation_id = self._run_evaluation(dataset, suite_id, selected_count)
            export_dir = self._export_evaluation(dataset, evaluation_id)
            values["evaluation_run_id"] = evaluation_id
            values["evaluation_export_dir"] = str(export_dir)
        self._save_state()

    def _write_final_summary(
        self,
        *,
        dry_run: bool = False,
        prepare_only: bool = False,
    ) -> dict[str, object]:
        datasets: dict[str, object] = {}
        for dataset, artifacts in sorted(self.state.dataset_artifacts.items()):
            summary: dict[str, object] = dict(artifacts)
            coverage_export_value = artifacts.get("coverage_export_dir")
            if coverage_export_value:
                coverage_export = Path(str(coverage_export_value))
                summary["coverage_metrics"] = _read_jsonl(
                    coverage_export / "public" / "metric_summary.jsonl"
                )
                summary["questioner_summary"] = json.loads(
                    (coverage_export / "public" / "questioner_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                summary["coverage_summary"] = json.loads(
                    (coverage_export / "public" / "coverage_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
            export_dir_value = artifacts.get("evaluation_export_dir")
            if export_dir_value:
                export_dir = Path(str(export_dir_value))
                summary["method_metrics"] = _read_jsonl(
                    export_dir / "public" / "method_summary.jsonl"
                )
                summary["judge_metrics"] = _read_jsonl(
                    export_dir / "public" / "judge_method_summary.jsonl"
                )
                summary["bad_cases"] = _read_jsonl(
                    export_dir / "public" / "primary_bad_case_method_summary.jsonl"
                )
                summary["production_cost"] = _read_jsonl(
                    export_dir / "public" / "production_cost_summary.jsonl"
                )
                summary["evaluation_cost"] = _read_jsonl(
                    export_dir / "public" / "evaluation_cost_summary.jsonl"
                )
                summary["combined_cost"] = _read_jsonl(
                    export_dir / "public" / "combined_research_cost_summary.jsonl"
                )
            datasets[dataset] = summary
        model_identities = self._model_identities()
        stage_timings = [
            {
                "stage_id": item.stage_id,
                "seconds": (
                    (item.completed_at - item.started_at).total_seconds()
                    if item.completed_at is not None
                    else None
                ),
                "reused": item.reused,
                "status": item.status.value,
            }
            for item in self.state.stage_results
        ]
        pilot_summary: dict[str, object] | None = None
        pilot_dir_value = self.state.metadata.get("pilot_dir")
        if pilot_dir_value:
            pilot_dir = Path(str(pilot_dir_value))
            automatic_summary_path = pilot_dir / "public" / "automatic_summary.json"
            if self.state.metadata.get("pilot_mode") == "smoke":
                automatic_summary_path = pilot_dir / "automatic_quality.json"
            pilot_summary = {
                "pilot_id": self.state.metadata.get("pilot_id"),
                "status": self.state.metadata.get("pilot_status"),
                "partition_id": self.state.metadata.get("partition_id"),
                "pilot_dir": str(pilot_dir),
                "automatic_summary": json.loads(automatic_summary_path.read_text(encoding="utf-8")),
            }
        pilot_status = self.state.metadata.get("pilot_status")
        result: dict[str, object] = {
            "launch_id": self.launch_id,
            "status": (
                "dry_run"
                if dry_run
                else "prepared"
                if prepare_only
                else str(pilot_status)
                if _is_pilot_target(self.config.target)
                and pilot_status
                in {
                    "awaiting_review",
                    "awaiting_approval",
                    "awaiting_threshold_review",
                    "needs_remediation",
                }
                else "completed"
            ),
            "config_hash": self.effective_config_hash,
            "launcher_config_hash": self.config.config_hash,
            "methods": [item.value for item in self.config.methods],
            "target": self.config.target.value,
            "models": self.config.models.model_dump(mode="json"),
            "model_preparation": (
                self.preparation.model_dump(mode="json") if self.preparation else None
            ),
            "model_identities": model_identities,
            "runtime_calibrations": self._calibration_summaries(model_identities),
            "managed_runtime_cleanup_verified": (
                _is_questioner_target(self.config.target) and not dry_run and not prepare_only
            ),
            "execution": {
                "schedule": self.execution_config.schedule.value,
                "execution_profile_hash": self.execution_config.config_hash,
                "execution_attempt_id": self._execution_attempt_id,
                "checkpoint_policy": self.execution_config.checkpoint.policy.value,
                "upstream_import": self.state.metadata.get("upstream_import"),
                "restored_checkpoint": self.state.metadata.get("restored_checkpoint"),
                "latest_checkpoint": self.state.metadata.get("latest_checkpoint"),
                "stage_performance_path": str(self.launch_dir / "stage_performance.jsonl"),
                "stage_performance_record_count": len(
                    _read_jsonl(self.launch_dir / "stage_performance.jsonl")
                ),
                "checkpoint_event_count": len(
                    _read_jsonl(self.launch_dir / "checkpoint_events.jsonl")
                ),
                "planner_conformance": self.state.metadata.get("step6c_planner_conformance"),
                "semantic_conformance": self.state.metadata.get("step6c2_semantic_conformance"),
                "step6d_compiler_conformance": self.state.metadata.get(
                    "step6d_compiler_conformance"
                ),
                "step6d_semantic_conformance": self.state.metadata.get(
                    "step6d_semantic_conformance"
                ),
                "step6d_prefix_manifest": self.state.metadata.get("step6d_prefix_manifest"),
                "step7_prefix": self.state.metadata.get("step7_prefix"),
                "step7_prefix_path": self.state.metadata.get("step7_prefix_path"),
                "step7b_prefix": self.state.metadata.get("step7b_prefix"),
                "step7b_prefix_path": self.state.metadata.get("step7b_prefix_path"),
                "step7_semantic_conformance": self.state.metadata.get("step7_semantic_conformance"),
                "step7_semantic_conformance_path": self.state.metadata.get(
                    "step7_semantic_conformance_path"
                ),
                "step7b_semantic_conformance": self.state.metadata.get(
                    "step7b_semantic_conformance"
                ),
                "step7b_semantic_conformance_path": self.state.metadata.get(
                    "step7b_semantic_conformance_path"
                ),
            },
            "stage_timings": stage_timings,
            "datasets": datasets,
            "questioner_pilot": (
                pilot_summary
                if self.config.target is ExperimentTarget.PMMC_QUESTIONER_PILOT
                else None
            ),
            "compiler_program_pilot": (
                pilot_summary if _is_program_pilot_target(self.config.target) else None
            ),
            "launch_directory": str(self.launch_dir),
        }
        _atomic_json(self.launch_dir / "final_summary.json", result)
        markdown = [
            f"# PMMC experiment {self.launch_id}",
            "",
            f"- Status: `{result['status']}`",
            f"- Target: `{self.config.target.value}`",
            f"- Methods: {', '.join(item.value for item in self.config.methods) or 'n/a'}",
            f"- Datasets: {', '.join(sorted(datasets)) or 'not executed'}",
            f"- Config hash: `{self.effective_config_hash}`",
            f"- Execution schedule: `{self.execution_config.schedule.value}`",
            f"- Checkpoint policy: `{self.execution_config.checkpoint.policy.value}`",
            "",
        ]
        if pilot_summary is not None:
            markdown.extend(
                [
                    "## Quality pilot",
                    "",
                    f"- Pilot: `{pilot_summary.get('pilot_id')}`",
                    f"- Partition: `{pilot_summary.get('partition_id')}`",
                    (
                        "- Scientific gate: signed audit plus explicit research-lead approval"
                        if pilot_summary.get("status")
                        in {"awaiting_approval", "awaiting_threshold_review"}
                        else "- Scientific gate: explicit human review and approval"
                    ),
                    f"- Private audit/review bundle: `{pilot_summary.get('pilot_dir')}`",
                    "",
                ]
            )
        for dataset, value in sorted(datasets.items()):
            assert isinstance(value, dict)
            if _is_questioner_target(self.config.target):
                markdown.extend(
                    [
                        f"## {dataset}",
                        "",
                        f"- Compiler run: `{value.get('compiler_run_id')}`",
                        f"- Compilation conversations: {value.get('conversation_count')}",
                        f"- Coverage queries: {value.get('selected_count')}",
                        f"- Coverage run: `{value.get('coverage_run_id')}`",
                        "",
                    ]
                )
                questioner_summary = value.get("questioner_summary")
                if isinstance(questioner_summary, dict):
                    markdown.extend(
                        [
                            "### Questioner conformance",
                            "",
                            f"- Schema valid rate: {questioner_summary.get('schema_valid_rate')}",
                            "- Gate pass rate among schema-valid: "
                            f"{questioner_summary.get('gate_pass_rate_among_schema_valid')}",
                            "- Overall eligible rate: "
                            f"{questioner_summary.get('overall_eligible_rate')}",
                            "- Zero-valid unit rate: "
                            f"{questioner_summary.get('zero_valid_candidate_unit_rate')}",
                            "",
                        ]
                    )
                verification = value.get("questioner_verification")
                export_verification = value.get("questioner_export_verification")
                if isinstance(verification, dict) and isinstance(export_verification, dict):
                    markdown.extend(
                        [
                            f"- QBank collections: {verification.get('qbank_collection_count')}",
                            "- Managed model calls: "
                            f"{verification.get('managed_model_call_count')}",
                            "- Archived model artifact references: "
                            f"{export_verification.get('model_artifact_reference_count')}",
                            "",
                        ]
                    )
                coverage_metrics = value.get("coverage_metrics")
                if isinstance(coverage_metrics, list) and coverage_metrics:
                    markdown.extend(
                        [
                            "### QBank coverage",
                            "",
                            "| Metric | Budget | K | Mean | Applicable | N/A |",
                            "|---|---:|---:|---:|---:|---:|",
                        ]
                    )
                    for row in coverage_metrics:
                        if not isinstance(row, dict):
                            continue
                        mean = row.get("mean")
                        mean_text = "n/a" if mean is None else f"{float(mean):.4f}"
                        markdown.append(
                            f"| {row.get('metric_name')} | {row.get('question_budget')} | "
                            f"{row.get('k')} | {mean_text} | "
                            f"{row.get('applicable_count')} | "
                            f"{row.get('not_applicable_count')} |"
                        )
                    markdown.append("")
            else:
                markdown.extend(
                    [
                        f"## {dataset}",
                        "",
                        f"- Suite: `{value.get('suite_id')}`",
                        f"- Selected queries: {value.get('selected_count')}",
                        f"- Evaluation: `{value.get('evaluation_run_id')}`",
                        "",
                    ]
                )
            metrics = value.get("method_metrics")
            if isinstance(metrics, list) and metrics:
                markdown.extend(
                    [
                        "### Deterministic and evidence metrics",
                        "",
                        "| Method | Metric | Mean | Applicable | Unavailable |",
                        "|---|---|---:|---:|---:|",
                    ]
                )
                for row in metrics:
                    if not isinstance(row, dict):
                        continue
                    mean = row.get("mean")
                    mean_text = "n/a" if mean is None else f"{float(mean):.4f}"
                    markdown.append(
                        f"| {row.get('method')} | {row.get('metric_name')} | "
                        f"{mean_text} | {row.get('applicable_count')} | "
                        f"{row.get('unavailable_count')} |"
                    )
                markdown.append("")
            judges = value.get("judge_metrics")
            if isinstance(judges, list) and judges:
                markdown.extend(
                    [
                        "### Judge",
                        "",
                        "| Method | Mean | Correct | Partial | Incorrect |",
                        "|---|---:|---:|---:|---:|",
                    ]
                )
                for row in judges:
                    if not isinstance(row, dict):
                        continue
                    markdown.append(
                        f"| {row.get('method')} | {float(row.get('mean_score', 0)):.4f} | "
                        f"{row.get('correct_count')} | {row.get('partially_correct_count')} | "
                        f"{row.get('incorrect_count')} |"
                    )
                markdown.append("")
        (self.launch_dir / "final_summary.md").write_text("\n".join(markdown), encoding="utf-8")
        return result

    def _model_identities(self) -> list[dict[str, object]]:
        if self.preparation is None or self.resolved_registry is None:
            return []
        lock = load_model_lock(self.preparation.lock_path)
        roles_by_alias: dict[str, list[str]] = {}
        for role, alias in self.config.models.runtime_roles().items():
            if alias in self.preparation.required_aliases:
                roles_by_alias.setdefault(alias, []).append(role.value)
        return [
            {
                "alias": entry.alias,
                "roles": sorted(roles_by_alias.get(entry.alias, [])),
                "repo_id": entry.repo_id,
                "commit_sha": entry.commit_sha,
                "endpoint_kind": self.resolved_registry.get(entry.alias).endpoint_kind.value,
                "size_bytes": self.preparation.model_size_bytes.get(entry.alias, 0),
            }
            for entry in sorted(lock.entries, key=lambda item: item.alias)
            if entry.alias in self.preparation.required_aliases
        ]

    def _calibration_summaries(
        self, model_identities: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        expected = {(str(item["alias"]), str(item["commit_sha"])) for item in model_identities}
        summaries: list[dict[str, object]] = []
        for path in sorted(self.launch_settings.runtime_calibrations_path.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            identity = (str(payload.get("model_alias")), str(payload.get("model_commit")))
            if identity not in expected:
                continue
            summaries.append(
                {
                    "calibration_id": payload.get("calibration_id"),
                    "service_id": payload.get("service_id"),
                    "model_alias": payload.get("model_alias"),
                    "model_commit": payload.get("model_commit"),
                    "hardware_fingerprint": payload.get("hardware_fingerprint"),
                    "vllm_version": payload.get("vllm_version"),
                    "profile_hash": payload.get("profile_hash"),
                }
            )
        return summaries

    def _run_cleanup(self) -> None:
        self._finish_stage("cleanup")
        self._signal_sidecars()

    def _signal_sidecars(self) -> None:
        if (
            self.config.mode is AppMode.DEV
            and os.environ.get("PMMC_LAUNCH_SUPPRESS_SIDECAR_STOP") == "1"
        ):
            return
        value = os.environ.get("PMMC_LAUNCH_STOP_FILE")
        if value:
            path = Path(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def _complete_state(self) -> None:
        self.state = self.state.model_copy(
            update={
                "status": ExperimentLaunchStatus.COMPLETED,
                "completed_at": utc_now(),
                "error_type": None,
            }
        )
        self._save_state()

    def _save_state(self) -> None:
        _atomic_json(self.state_path, self.state)
        _atomic_json(
            self.launch_dir / "child_runs.json",
            {
                "launch_id": self.launch_id,
                "datasets": self.state.dataset_artifacts,
            },
        )


def _step7b_match_diagnostics(query_export: Path) -> dict[str, object]:
    """Content-free query-level diagnostics from already verified source exports.

    Fused candidates cover all saved grids, not just the active route. These
    counts distinguish absence from missing work, but never change acceptance.
    """

    def rows(name: str) -> list[dict[str, object]]:
        path = query_export / "runtime" / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError("matching diagnostic source is absent")
        return _read_jsonl(path)

    def query_set(items: Sequence[dict[str, object]]) -> set[str]:
        return {value for row in items if isinstance(value := row.get("query_id"), str)}

    queries = query_set(rows("run_queries"))
    b8_batches = {
        row.get("match_batch_id"): row.get("query_id")
        for row in rows("qbank_match_batches") if row.get("question_budget") == 8
    }
    fused = rows("qbank_match_fused_candidates")
    reranker = rows("qbank_reranker_assessments")
    bindings = rows("query_program_bindings")
    candidate_queries = queries & query_set(fused)
    b8_candidate_queries = queries & query_set([
        row for row in fused
        if row.get("match_batch_id") in b8_batches
        and b8_batches[row.get("match_batch_id")] == row.get("query_id")
    ])
    reranked_queries = queries & query_set(reranker)
    bound_queries = queries & query_set(bindings)
    missing_rerank = queries - reranked_queries
    missing_binding = queries - bound_queries
    levels = ("exact", "strong", "weak", "none", "unavailable")
    provider_counts = dict.fromkeys(levels, 0)
    effective_counts = dict.fromkeys(levels, 0)
    fail_counts = dict.fromkeys((
        "same_information_need", "candidate_can_answer_query",
        "answer_type_compatible", "modality_compatible",
    ), 0)
    downgraded = 0
    for row in reranker:
        if row.get("query_id") not in queries:
            continue
        metadata = row.get("metadata")
        provider = metadata.get("provider_match_level") if isinstance(metadata, dict) else None
        effective = row.get("match_level")
        provider_counts[provider if isinstance(provider, str) and provider in levels
                        else "unavailable"] += 1
        effective_counts[effective if isinstance(effective, str) and effective in levels
                         else "unavailable"] += 1
        downgraded += provider in ("exact", "strong") and effective == "none"
        for dimension in fail_counts:
            fail_counts[dimension] += row.get(dimension) == "fail"

    return {
        "state": "ready",
        "scope": "query_denominators_all_saved_grids_not_pair_completeness",
        "query_count": len(queries),
        "queries_with_fused_candidates": len(candidate_queries),
        "queries_without_fused_candidates": len(queries - candidate_queries),
        "reranked_query_count": len(reranked_queries),
        "unreranked_queries_with_candidates": len(missing_rerank & candidate_queries),
        "unreranked_queries_without_candidates": len(missing_rerank - candidate_queries),
        "program_bound_query_count": len(bound_queries),
        "unbound_queries_with_b8_candidates": len(missing_binding & b8_candidate_queries),
        "unbound_queries_without_b8_candidates": len(missing_binding - b8_candidate_queries),
        "out_of_scope_query_counts": {
            name: len(query_set(items) - queries)
            for name, items in (("fused", fused), ("reranker", reranker), ("bindings", bindings))
        },
        "reranker_provider_match_levels": provider_counts,
        "reranker_effective_match_levels": effective_counts,
        "reranker_required_dimension_failures": fail_counts,
        "reranker_exact_or_strong_downgraded_to_none": downgraded,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_launcher_command(
    *,
    settings: Settings,
    config: ExperimentLauncherConfig,
    resume_launch_id: str | None = None,
    import_upstream_from_launch: str | None = None,
    dry_run: bool = False,
    prepare_only: bool = False,
    evaluate_only: bool = False,
) -> dict[str, object]:
    """Public synchronous entry point used by Typer and generated Jobs."""

    return ResearchExperimentLauncher(
        settings=settings,
        config=config,
        resume_launch_id=resume_launch_id,
        import_upstream_from_launch=import_upstream_from_launch,
        dry_run=dry_run,
        prepare_only=prepare_only,
        evaluate_only=evaluate_only,
    ).run()
