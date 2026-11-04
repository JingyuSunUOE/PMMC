"""Contracts for paired execution of the five foundational baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
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

FOUNDATIONAL_BASELINE_METHODS = tuple(BaselineMethod)


def is_release_baseline_suite(methods: Sequence[BaselineMethod], metadata: Mapping[str, object]) -> bool:
    digest = metadata.get("release_plan_hash")
    return (len(methods) == 1 and methods[0].value == metadata.get("standalone_method")
            and metadata.get("partition_role") == "development" and metadata.get("paper_result") is False
            and metadata.get("answer_contract_version") == "typed-answer-contract-v3"
            and isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest))


def is_step8_caption_fallback_suite(
    methods: Sequence[BaselineMethod], metadata: Mapping[str, object],
) -> bool:
    """Structural binding only; campaign admission verifies the approved protocol."""
    protocol_hash = metadata.get("method_protocol_hash")
    return (
        list(methods) == [BaselineMethod.NATIVE_MULTIMODAL_RAG]
        and metadata.get("internal_pmmc_fallback") is True
        and metadata.get("answer_contract_version") == "typed-answer-contract-v3"
        and metadata.get("allow_memory_raw_images") is False
        and isinstance(protocol_hash, str)
        and len(protocol_hash) == 64
        and all(char in "0123456789abcdef" for char in protocol_hash)
    )


def is_internal_pmmc_fallback_suite(
    methods: Sequence[BaselineMethod],
    metadata: Mapping[str, object],
) -> bool:
    """Recognize the development-only Native-MM child used by PMMC."""

    partition_roles: list[object] = []
    if "partition_role" in metadata:
        partition_roles.append(metadata["partition_role"])
    selection = metadata.get("selection")
    if isinstance(selection, Mapping):
        selection_metadata = selection.get("metadata")
        if isinstance(selection_metadata, Mapping) and "partition_role" in selection_metadata:
            partition_roles.append(selection_metadata["partition_role"])
    return (
        metadata.get("internal_pmmc_fallback") is True
        and metadata.get("paper_result") is False
        and list(methods) == [BaselineMethod.NATIVE_MULTIMODAL_RAG]
        and bool(partition_roles)
        and all(role == "development" for role in partition_roles)
    )


class BaselineSuiteStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BaselineSuiteManifest(BaseSchema):
    """Immutable paired-denominator configuration for a baseline suite."""

    suite_id: NonEmptyStr
    dataset: DatasetName
    snapshot_id: NonEmptyStr
    selection_hash: NonEmptyStr
    methods: list[BaselineMethod] = Field(
        default_factory=lambda: list(FOUNDATIONAL_BASELINE_METHODS)
    )
    answer_model_alias: NonEmptyStr
    answer_prompt_hash: NonEmptyStr
    decoding_config: JsonDict
    total_context_tokens: PositiveInt = 32768
    reserved_output_tokens: PositiveInt = 1024
    query_image_policy: NonEmptyStr = "preserve_query_images_v1"
    evaluation_policy_version: NonEmptyStr
    seed: NonNegativeInt = 0
    paper_mode: bool = False
    config_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_suite(self) -> BaselineSuiteManifest:
        if not self.methods:
            raise ValueError("baseline suites require at least one method")
        ensure_unique_strings([item.value for item in self.methods], "methods")
        component_mode = (
            self.metadata.get("paper_component") is True
            and self.metadata.get("paper_result") is False
            and len(self.methods) == 1
        )
        internal_fallback_mode = is_internal_pmmc_fallback_suite(
            self.methods,
            self.metadata,
        )
        if (
            self.paper_mode
            and set(self.methods) != set(FOUNDATIONAL_BASELINE_METHODS)
            and not component_mode
            and not internal_fallback_mode
            and not is_step8_caption_fallback_suite(self.methods, self.metadata)
            and not is_release_baseline_suite(self.methods, self.metadata)
        ):
            raise ValueError(
                "paper baseline suites require exactly five methods or one isolated component"
            )
        if self.reserved_output_tokens >= self.total_context_tokens:
            raise ValueError("reserved_output_tokens must be smaller than total_context_tokens")
        return self


class BaselineSuiteState(BaseSchema):
    suite_id: NonEmptyStr
    status: BaselineSuiteStatus = BaselineSuiteStatus.CREATED
    selected_count: PositiveInt
    completed_run_count: NonNegativeInt = 0
    started_at: UtcDateTime | None = None
    heartbeat_at: UtcDateTime | None = None
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> BaselineSuiteState:
        if self.completed_run_count > len(FOUNDATIONAL_BASELINE_METHODS):
            raise ValueError("completed_run_count exceeds foundational method count")
        if self.status is BaselineSuiteStatus.READY:
            if self.completed_run_count <= 0 or self.completed_at is None:
                raise ValueError("ready suites require completed runs and completed_at")
        if self.status is BaselineSuiteStatus.FAILED and self.error_type is None:
            raise ValueError("failed suites require error_type")
        return self


class BaselineSuiteRunLink(BaseSchema):
    suite_id: NonEmptyStr
    method: BaselineMethod
    run_id: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class BaselineSuiteQuery(BaseSchema):
    suite_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    ordinal: NonNegativeInt
    metadata: JsonDict = Field(default_factory=dict)
