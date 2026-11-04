"""Benchmark ingestion, selection, validation, and snapshot contracts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    SCHEMA_VERSION,
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    QuestionType,
    UtcDateTime,
    utc_now,
)
from pmmc.schemas.query import QueryGold, QueryInput

DEFAULT_EVALUATION_POLICY_VERSION = "pmmc-aaai27-v1"
_GOLD_METADATA_KEY_FRAGMENTS = ("answer", "clue", "gold")


def _reject_gold_metadata(value: JsonDict, field_name: str) -> None:
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                normalized_key = str(key).lower()
                if any(fragment in normalized_key for fragment in _GOLD_METADATA_KEY_FRAGMENTS):
                    raise ValueError(f"{field_name} cannot contain evaluation-only key: {key}")
                pending.append(child)
        elif isinstance(item, list):
            pending.extend(item)


class ValidationLevel(StrEnum):
    """Depth of local benchmark asset validation."""

    FAST = "fast"
    FULL = "full"


class ValidationSeverity(StrEnum):
    """Severity assigned to one validation issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SessionRecord(BaseSchema):
    """Canonical session boundary and dataset-logical time metadata."""

    session_id: NonEmptyStr
    dataset: DatasetName
    conversation_id: NonEmptyStr
    session_order: NonNegativeInt
    source_session_id: NonEmptyStr
    source_date_raw: NonEmptyStr | None = None
    source_date_iso: NonEmptyStr | None = None
    is_synthetic: bool = False
    metadata: JsonDict = Field(default_factory=dict)


class SourceLocator(BaseSchema):
    """Runtime-safe pointer back to a raw benchmark record."""

    dataset: DatasetName
    source_file: NonEmptyStr
    source_record_index: NonNegativeInt | None = None
    source_query_id: NonEmptyStr | None = None
    scenario_id: NonEmptyStr | None = None
    qa_index: NonNegativeInt | None = None
    source_session_id: NonEmptyStr | None = None
    source_round_id: NonEmptyStr | None = None
    source_turn_index: NonNegativeInt | None = None
    source_category: NonEmptyStr | None = None
    source_subtype: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source_file(self) -> SourceLocator:
        source_path = PurePosixPath(self.source_file)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("source_file must be a safe path relative to the benchmark root")
        _reject_gold_metadata(self.metadata, "SourceLocator.metadata")
        return self


class EvaluationEligibility(BaseSchema):
    """Whether a source query belongs to the default PMMC evaluation policy."""

    eligible: bool
    policy_version: NonEmptyStr = DEFAULT_EVALUATION_POLICY_VERSION
    exclusion_reason: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_exclusion(self) -> EvaluationEligibility:
        if not self.eligible and self.exclusion_reason is None:
            raise ValueError("ineligible benchmark queries require exclusion_reason")
        if self.eligible and self.exclusion_reason is not None:
            raise ValueError("eligible benchmark queries cannot include exclusion_reason")
        return self


class BenchmarkQuery(BaseSchema):
    """Runtime-safe canonical query plus source and eligibility metadata."""

    query: QueryInput
    source: SourceLocator
    eligibility: EvaluationEligibility
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_dataset(self) -> BenchmarkQuery:
        if self.query.dataset is not self.source.dataset:
            raise ValueError("query and source locator datasets must match")
        _reject_gold_metadata(self.metadata, "BenchmarkQuery.metadata")
        return self


class QueryGoldRecord(BaseSchema):
    """Evaluation-only gold record keyed by canonical query ID."""

    query_id: NonEmptyStr
    gold: QueryGold
    metadata: JsonDict = Field(default_factory=dict)


class BenchmarkSelection(BaseSchema):
    """Deterministic query-selection specification shared by all methods."""

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
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_selection(self) -> BenchmarkSelection:
        if not self.datasets:
            raise ValueError("datasets must contain at least one dataset")
        if DatasetName.UNKNOWN in self.datasets:
            raise ValueError("unknown is not a selectable benchmark dataset")
        for field_name in (
            "datasets",
            "question_types",
            "source_categories",
            "scenario_ids",
            "query_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicate values")
        if self.limit is not None and self.sample_size is not None:
            raise ValueError("limit and sample_size are mutually exclusive")
        if self.shard_index >= self.shard_count:
            raise ValueError("shard_index must be less than shard_count")
        if self.scenario_ids and DatasetName.MEMGALLERY not in self.datasets:
            raise ValueError("scenario_ids can only be used with the memgallery dataset")
        return self


class DatasetSnapshotManifest(BaseSchema):
    """Reproducibility manifest for one canonical benchmark snapshot."""

    snapshot_id: NonEmptyStr
    dataset: DatasetName
    dataset_version: NonEmptyStr
    adapter_name: NonEmptyStr
    adapter_version: NonEmptyStr
    schema_version: NonEmptyStr = SCHEMA_VERSION
    evaluation_policy_version: NonEmptyStr = DEFAULT_EVALUATION_POLICY_VERSION
    excluded_source_categories: list[NonEmptyStr] = Field(default_factory=list)
    source_content_hash: NonEmptyStr
    canonical_content_hash: NonEmptyStr
    counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    category_counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    eligible_category_counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    excluded_category_counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_excluded_categories(self) -> DatasetSnapshotManifest:
        if len(self.excluded_source_categories) != len(set(self.excluded_source_categories)):
            raise ValueError("excluded_source_categories must not contain duplicate values")
        return self


class ValidationIssue(BaseSchema):
    """One structured benchmark validation finding."""

    severity: ValidationSeverity
    code: NonEmptyStr
    message: NonEmptyStr
    source: SourceLocator | None = None
    object_id: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ValidationReport(BaseSchema):
    """Structured result of validating one raw or canonical benchmark."""

    dataset: DatasetName
    level: ValidationLevel
    success: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    started_at: UtcDateTime = Field(default_factory=utc_now)
    completed_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_report(self) -> ValidationReport:
        has_errors = any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)
        if self.success == has_errors:
            raise ValueError("success must be false exactly when validation errors exist")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self
