"""Experiment manifest schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    EXPERIMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_COMPILER_SCHEMA_VERSIONS,
    SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS,
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.benchmark import SourceLocator


class PipelineEventType(StrEnum):
    """Structured local-pipeline event categories."""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    STAGE_STARTED = "stage_started"
    STAGE_PROGRESS = "stage_progress"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    ITEM_FAILED = "item_failed"


class PipelineEvent(BaseSchema):
    """Non-sensitive event emitted by the local benchmark pipeline."""

    run_id: NonEmptyStr
    event_type: PipelineEventType
    stage: NonEmptyStr
    timestamp: UtcDateTime = Field(default_factory=utc_now)
    dataset: DatasetName | None = None
    current: NonNegativeInt | None = None
    total: NonNegativeInt | None = None
    elapsed_ms: NonNegativeFloat | None = None
    query_id: NonEmptyStr | None = None
    memory_id: NonEmptyStr | None = None
    source: SourceLocator | None = None
    message: NonEmptyStr | None = None
    error_type: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_progress(self) -> PipelineEvent:
        if self.current is not None and self.total is not None and self.current > self.total:
            raise ValueError("current cannot exceed total")
        if self.event_type in {
            PipelineEventType.RUN_FAILED,
            PipelineEventType.STAGE_FAILED,
            PipelineEventType.ITEM_FAILED,
        }:
            if self.error_type is None:
                raise ValueError("failed pipeline events require error_type")
        return self


class PipelineStageSummary(BaseSchema):
    """Timing and item counts for one completed local-pipeline stage."""

    stage: NonEmptyStr
    success: bool
    started_at: UtcDateTime
    completed_at: UtcDateTime
    elapsed_ms: NonNegativeFloat
    processed: NonNegativeInt = 0
    total: NonNegativeInt | None = None
    succeeded: NonNegativeInt = 0
    failed: NonNegativeInt = 0
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_stage(self) -> PipelineStageSummary:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        if self.succeeded + self.failed > self.processed:
            raise ValueError("succeeded + failed cannot exceed processed")
        if self.success and self.failed:
            raise ValueError("a successful stage cannot contain failed items")
        if self.total is not None and self.processed > self.total:
            raise ValueError("processed cannot exceed total")
        return self


class PipelineRunSummary(BaseSchema):
    """Aggregate outcome for one local benchmark pipeline run."""

    run_id: NonEmptyStr
    datasets: list[DatasetName]
    success: bool
    started_at: UtcDateTime
    completed_at: UtcDateTime
    elapsed_ms: NonNegativeFloat
    stages: list[PipelineStageSummary] = Field(default_factory=list)
    selected_count: NonNegativeInt = 0
    succeeded_count: NonNegativeInt = 0
    failed_count: NonNegativeInt = 0
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_run(self) -> PipelineRunSummary:
        if not self.datasets:
            raise ValueError("datasets must contain at least one dataset")
        if len(self.datasets) != len(set(self.datasets)):
            raise ValueError("datasets must not contain duplicate values")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        if self.succeeded_count + self.failed_count > self.selected_count:
            raise ValueError("run result counts cannot exceed selected_count")
        if self.success and any(not stage.success for stage in self.stages):
            raise ValueError("a successful run cannot contain failed stages")
        return self


class RunManifest(BaseSchema):
    """Global configuration snapshot for one experiment run."""

    run_id: NonEmptyStr
    experiment_name: NonEmptyStr
    dataset: DatasetName
    dataset_version: NonEmptyStr | None = None
    split: NonEmptyStr | None = None
    method: NonEmptyStr
    paper_mode: bool = False
    answer_model_alias: NonEmptyStr | None = None
    answer_actual_model_id: NonEmptyStr | None = None
    compiler_model_alias: NonEmptyStr | None = None
    judge_model_alias: NonEmptyStr | None = None
    embedding_model_aliases: list[NonEmptyStr] = Field(default_factory=list)
    r_max: NonNegativeInt | None = None
    question_budget_max: PositiveInt | None = None
    match_topk: PositiveInt | None = None
    match_threshold: Probability | None = None
    fallback_policy: NonEmptyStr | None = None
    selection_hash: NonEmptyStr | None = None
    seed: NonNegativeInt | None = None
    retrieval_topk_max: PositiveInt | None = None
    prompt_hashes: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    model_configs: dict[NonEmptyStr, JsonDict] = Field(default_factory=dict)
    decoding_configs: dict[NonEmptyStr, JsonDict] = Field(default_factory=dict)
    retrieval_config: JsonDict = Field(default_factory=dict)
    environment_fingerprint: JsonDict = Field(default_factory=dict)
    config_hash: NonEmptyStr | None = None
    evaluation_policy_version: NonEmptyStr | None = None
    evaluation_exclusion_rules: dict[NonEmptyStr, list[NonEmptyStr]] = Field(default_factory=dict)
    dataset_snapshot_ids: list[NonEmptyStr] = Field(default_factory=list)
    prompt_registry_version: NonEmptyStr | None = None
    schema_version: NonEmptyStr = SCHEMA_VERSION
    experiment_schema_version: NonEmptyStr = EXPERIMENT_SCHEMA_VERSION
    compiler_schema_version: NonEmptyStr | None = None
    operator_set_version: NonEmptyStr | None = None
    operator_registry_hash: NonEmptyStr | None = None
    compiler_config_hash: NonEmptyStr | None = None
    git_commit: NonEmptyStr | None = None
    docker_image: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_aliases(self) -> RunManifest:
        ensure_unique_strings(self.embedding_model_aliases, "embedding_model_aliases")
        ensure_unique_strings(self.dataset_snapshot_ids, "dataset_snapshot_ids")
        for dataset, categories in self.evaluation_exclusion_rules.items():
            ensure_unique_strings(categories, f"evaluation_exclusion_rules[{dataset}]")
        if self.experiment_schema_version not in SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS))
            raise ValueError(f"unsupported experiment schema version; expected one of: {supported}")
        compiler_fields = (
            self.compiler_schema_version,
            self.operator_set_version,
            self.operator_registry_hash,
            self.compiler_config_hash,
        )
        if self.method == "pmmc_compiler" or any(value is not None for value in compiler_fields):
            if any(value is None for value in compiler_fields):
                raise ValueError(
                    "compiler runs require compiler schema, operator set, registry, "
                    "and config hashes"
                )
            if self.compiler_schema_version not in SUPPORTED_COMPILER_SCHEMA_VERSIONS:
                raise ValueError(
                    f"unsupported compiler schema version: {self.compiler_schema_version}"
                )
        return self
