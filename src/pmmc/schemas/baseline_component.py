"""Portable, independently recoverable baseline component contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BASELINE_COMPONENT_SCHEMA_VERSION,
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.baseline import BaselineMethod
from pmmc.schemas.baseline_suite import FOUNDATIONAL_BASELINE_METHODS


class BaselineFamilyStatus(StrEnum):
    REGISTERED = "registered"
    ASSEMBLING = "assembling"
    READY = "ready"
    FAILED = "failed"


class BaselineComponentStatus(StrEnum):
    REGISTERED = "registered"
    RETRIEVAL_READY = "retrieval_ready"
    ANSWER_READY = "answer_ready"
    EXPORTED = "exported"
    IMPORTED = "imported"
    VERIFIED = "verified"
    FAILED = "failed"


_COMPONENT_TRANSITIONS: dict[BaselineComponentStatus, set[BaselineComponentStatus]] = {
    BaselineComponentStatus.REGISTERED: {
        BaselineComponentStatus.RETRIEVAL_READY,
        BaselineComponentStatus.ANSWER_READY,
        BaselineComponentStatus.FAILED,
    },
    BaselineComponentStatus.RETRIEVAL_READY: {
        BaselineComponentStatus.ANSWER_READY,
        BaselineComponentStatus.FAILED,
    },
    BaselineComponentStatus.ANSWER_READY: {
        BaselineComponentStatus.EXPORTED,
        BaselineComponentStatus.FAILED,
    },
    BaselineComponentStatus.EXPORTED: {
        BaselineComponentStatus.IMPORTED,
        BaselineComponentStatus.FAILED,
    },
    BaselineComponentStatus.IMPORTED: {
        BaselineComponentStatus.VERIFIED,
        BaselineComponentStatus.FAILED,
    },
}


def baseline_component_transition_allowed(
    current: BaselineComponentStatus,
    target: BaselineComponentStatus,
) -> bool:
    """Return whether one persisted component transition is legal."""

    return target in _COMPONENT_TRANSITIONS.get(current, set())


class BaselineSuiteFamilyManifest(BaseSchema):
    """Immutable scientific identity shared by all ten paper components."""

    schema_version: NonEmptyStr = BASELINE_COMPONENT_SCHEMA_VERSION
    family_id: NonEmptyStr
    partition_id: NonEmptyStr
    partition_assignment_hash: NonEmptyStr
    datasets: list[DatasetName] = Field(
        default_factory=lambda: [DatasetName.MEMLENS, DatasetName.MEMGALLERY]
    )
    methods: list[BaselineMethod] = Field(
        default_factory=lambda: list(FOUNDATIONAL_BASELINE_METHODS)
    )
    snapshot_ids: dict[DatasetName, NonEmptyStr]
    selection_hashes: dict[DatasetName, NonEmptyStr]
    selected_counts: dict[DatasetName, PositiveInt]
    answer_model_alias: NonEmptyStr
    answer_model_revision: NonEmptyStr
    judge_model_alias: NonEmptyStr
    judge_model_revision: NonEmptyStr
    answer_prompt_hash: NonEmptyStr
    judge_prompt_hash: NonEmptyStr
    decoding_config: JsonDict
    total_context_tokens: PositiveInt = 32768
    reserved_output_tokens: PositiveInt = 1024
    retrieval_top_k: PositiveInt = 50
    answer_top_k: PositiveInt = 5
    protocol_versions: dict[NonEmptyStr, NonEmptyStr]
    config_hash: NonEmptyStr
    paper_mode: bool = True
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_family(self) -> BaselineSuiteFamilyManifest:
        if self.schema_version != BASELINE_COMPONENT_SCHEMA_VERSION:
            raise ValueError("unsupported baseline component schema version")
        ensure_unique_strings([item.value for item in self.datasets], "datasets")
        ensure_unique_strings([item.value for item in self.methods], "methods")
        expected_datasets = {DatasetName.MEMLENS, DatasetName.MEMGALLERY}
        if self.paper_mode and set(self.datasets) != expected_datasets:
            raise ValueError("paper baseline families require both benchmarks")
        if self.paper_mode and set(self.methods) != set(FOUNDATIONAL_BASELINE_METHODS):
            raise ValueError("paper baseline families require all five methods")
        for mapping_name, mapping in (
            ("snapshot_ids", self.snapshot_ids),
            ("selection_hashes", self.selection_hashes),
            ("selected_counts", self.selected_counts),
        ):
            if set(mapping) != set(self.datasets):
                raise ValueError(f"{mapping_name} must cover every family dataset")
        if self.reserved_output_tokens >= self.total_context_tokens:
            raise ValueError("reserved output must be smaller than total context")
        if self.answer_top_k > self.retrieval_top_k:
            raise ValueError("answer_top_k cannot exceed retrieval_top_k")
        if self.paper_mode and self.metadata.get("paper_result") is True:
            raise ValueError("an unaggregated family cannot be a paper result")
        return self

    @property
    def expected_component_count(self) -> int:
        return len(self.datasets) * len(self.methods)


class BaselineComponentManifest(BaseSchema):
    """One dataset-method execution unit within a baseline family."""

    schema_version: NonEmptyStr = BASELINE_COMPONENT_SCHEMA_VERSION
    component_id: NonEmptyStr
    family_id: NonEmptyStr
    dataset: DatasetName
    method: BaselineMethod
    snapshot_id: NonEmptyStr
    selection_hash: NonEmptyStr
    selected_count: PositiveInt
    suite_id: NonEmptyStr | None = None
    run_id: NonEmptyStr | None = None
    corpus_id: NonEmptyStr | None = None
    scientific_config_hash: NonEmptyStr
    component_config_hash: NonEmptyStr
    paper_component: bool = True
    paper_result: bool = False
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_component(self) -> BaselineComponentManifest:
        if self.schema_version != BASELINE_COMPONENT_SCHEMA_VERSION:
            raise ValueError("unsupported baseline component schema version")
        if self.dataset is DatasetName.UNKNOWN:
            raise ValueError("baseline components require a concrete dataset")
        if self.paper_result:
            raise ValueError("individual baseline components are never paper results")
        return self


class BaselineSuiteFamilyState(BaseSchema):
    family_id: NonEmptyStr
    status: BaselineFamilyStatus = BaselineFamilyStatus.REGISTERED
    expected_component_count: PositiveInt
    verified_component_count: NonNegativeInt = 0
    updated_at: UtcDateTime = Field(default_factory=utc_now)
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_family_state(self) -> BaselineSuiteFamilyState:
        if self.verified_component_count > self.expected_component_count:
            raise ValueError("verified component count exceeds expected count")
        if self.status is BaselineFamilyStatus.READY:
            if self.verified_component_count != self.expected_component_count:
                raise ValueError("ready family requires every component")
            if self.completed_at is None:
                raise ValueError("ready family requires completed_at")
        if self.status is BaselineFamilyStatus.FAILED and self.error_type is None:
            raise ValueError("failed family requires error_type")
        return self


class BaselineComponentState(BaseSchema):
    component_id: NonEmptyStr
    status: BaselineComponentStatus = BaselineComponentStatus.REGISTERED
    transition_ordinal: NonNegativeInt = 0
    bundle_export_id: NonEmptyStr | None = None
    bundle_sha256: NonEmptyStr | None = None
    registered_at: UtcDateTime = Field(default_factory=utc_now)
    updated_at: UtcDateTime = Field(default_factory=utc_now)
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> BaselineComponentState:
        if self.status in {
            BaselineComponentStatus.EXPORTED,
            BaselineComponentStatus.IMPORTED,
            BaselineComponentStatus.VERIFIED,
        } and (self.bundle_export_id is None or self.bundle_sha256 is None):
            raise ValueError("exported component states require bundle identity")
        if self.status is BaselineComponentStatus.VERIFIED and self.completed_at is None:
            raise ValueError("verified component states require completed_at")
        if self.status is BaselineComponentStatus.FAILED and self.error_type is None:
            raise ValueError("failed component states require error_type")
        return self


class BaselineComponentImportRecord(BaseSchema):
    import_id: NonEmptyStr
    family_id: NonEmptyStr
    component_id: NonEmptyStr
    bundle_export_id: NonEmptyStr
    bundle_sha256: NonEmptyStr
    source_alembic_revision: NonEmptyStr
    importer_version: NonEmptyStr = "baseline-component-import-v1"
    imported_at: UtcDateTime = Field(default_factory=utc_now)
    imported_by: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)
