"""Managed local model-runtime contracts.

These schemas intentionally contain no credentials, complete GPU UUIDs, prompt
content, or provider URLs. They are safe to persist as experiment provenance.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    MODEL_VERIFICATION_SCHEMA_VERSION,
    RUNTIME_REPLICA_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    BaseSchema,
    FiniteFloat,
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


class ModelVerificationPolicy(StrEnum):
    FULL = "full"
    CACHED_RECEIPT = "cached_receipt"
    IDENTITY_ONLY = "identity_only"


class ModelVerificationFileState(BaseSchema):
    path: NonEmptyStr
    size_bytes: NonNegativeInt
    mtime_ns: NonNegativeInt
    symlink_target: str | None = None
    blob_size_bytes: NonNegativeInt
    blob_mtime_ns: NonNegativeInt
    sha256: NonEmptyStr | None = None


class ModelVerificationReceipt(BaseSchema):
    schema_version: NonEmptyStr = MODEL_VERIFICATION_SCHEMA_VERSION
    verification_id: NonEmptyStr
    repo_id: NonEmptyStr
    commit_sha: NonEmptyStr
    physical_lock_hash: NonEmptyStr
    verifier_version: NonEmptyStr
    files: list[ModelVerificationFileState]
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_receipt(self) -> ModelVerificationReceipt:
        if self.schema_version != MODEL_VERIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported model verification receipt version")
        if not self.files:
            raise ValueError("model verification receipt requires files")
        ensure_unique_strings([item.path for item in self.files], "files.path")
        return self


class ModelVerificationResult(BaseSchema):
    alias: NonEmptyStr
    repo_id: NonEmptyStr
    commit_sha: NonEmptyStr
    snapshot_path: NonEmptyStr
    policy: ModelVerificationPolicy
    receipt_id: NonEmptyStr | None = None
    receipt_cache_hit: bool = False
    verified_files: NonNegativeInt
    bytes_hashed: NonNegativeInt = 0
    metadata: JsonDict = Field(default_factory=dict)


class VerifiedModelCacheManifest(BaseSchema):
    """One launch's immutable, secret-free model-cache preparation result."""

    schema_version: NonEmptyStr = MODEL_VERIFICATION_SCHEMA_VERSION
    lock_path: NonEmptyStr
    lock_sha256: NonEmptyStr
    registry_identity_hash: NonEmptyStr
    selected_aliases: list[NonEmptyStr]
    snapshot_paths: dict[NonEmptyStr, NonEmptyStr]
    verifications: dict[NonEmptyStr, ModelVerificationResult]
    verification_policy: ModelVerificationPolicy
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_manifest(self) -> VerifiedModelCacheManifest:
        if self.schema_version != MODEL_VERIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported verified model-cache manifest version")
        ensure_unique_strings(self.selected_aliases, "selected_aliases")
        aliases = set(self.selected_aliases)
        if aliases != set(self.snapshot_paths) or aliases != set(self.verifications):
            raise ValueError("verified model-cache mappings must cover selected aliases")
        if any(
            alias != verification.alias
            for alias, verification in self.verifications.items()
        ):
            raise ValueError("verified model-cache result has the wrong alias")
        return self


class RuntimeMode(StrEnum):
    """How model endpoints are supplied to PMMC."""

    EXTERNAL = "external"
    MANAGED_LOCAL = "managed_local"


class ModelRole(StrEnum):
    """Stable model-service roles used across baseline and PMMC workflows."""

    ANSWER = "answer"
    COMPILER = "compiler"
    JUDGE = "judge"
    TEXT_EMBEDDING = "text_embedding"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"
    CONTRASTIVE_EMBEDDING = "contrastive_embedding"
    RERANKER = "reranker"


class ManagedBackendKind(StrEnum):
    """Process backends currently managed by the local harness."""

    VLLM_CHAT = "vllm_chat"
    VLLM_EMBEDDING = "vllm_embedding"
    VLLM_MULTIMODAL_EMBEDDING = "vllm_multimodal_embedding"
    TRANSFORMERS_CONTRASTIVE = "transformers_contrastive"
    LITELLM_PROXY = "litellm_proxy"


class RuntimeStatus(StrEnum):
    PLANNED = "planned"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ManagedServiceStatus(StrEnum):
    PLANNED = "planned"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class CalibrationStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


class ModelFileRecord(BaseSchema):
    """One immutable file in a locked Hugging Face model snapshot."""

    path: NonEmptyStr
    size_bytes: NonNegativeInt
    sha256: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ModelLockEntry(BaseSchema):
    """Resolved immutable Hub identity for one model alias."""

    alias: NonEmptyStr
    repo_id: NonEmptyStr
    requested_revision: NonEmptyStr
    commit_sha: NonEmptyStr
    files: list[ModelFileRecord]
    snapshot_content_hash: NonEmptyStr
    registry_entry_hash: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_files(self) -> ModelLockEntry:
        if not self.files:
            raise ValueError("model lock entries require at least one file")
        ensure_unique_strings([item.path for item in self.files], "files.path")
        return self


class ModelLockManifest(BaseSchema):
    """Secret-free model lock generated before managed execution."""

    lock_version: NonEmptyStr = RUNTIME_SCHEMA_VERSION
    registry_hash: NonEmptyStr
    registry_subset_hash: NonEmptyStr | None = None
    selected_aliases: list[NonEmptyStr] = Field(default_factory=list)
    entries: list[ModelLockEntry]
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_entries(self) -> ModelLockManifest:
        if not self.entries:
            raise ValueError("model lock cannot be empty")
        ensure_unique_strings([item.alias for item in self.entries], "entries.alias")
        ensure_unique_strings(self.selected_aliases, "selected_aliases")
        if (self.registry_subset_hash is None) != (not self.selected_aliases):
            raise ValueError("registry_subset_hash and selected_aliases must be provided together")
        if self.selected_aliases and self.selected_aliases != sorted(
            item.alias for item in self.entries
        ):
            raise ValueError("selected_aliases must exactly match sorted lock entries")
        return self

    def get(self, alias: str) -> ModelLockEntry:
        for entry in self.entries:
            if entry.alias == alias:
                return entry
        raise KeyError(f"model alias is not locked: {alias}")


class GpuDeviceInfo(BaseSchema):
    """Non-sensitive view of one GPU exposed inside the current container."""

    device_id: NonEmptyStr
    visible_index: NonNegativeInt
    name: NonEmptyStr
    total_memory_bytes: PositiveInt
    free_memory_bytes: NonNegativeInt
    used_memory_bytes: NonNegativeInt
    compute_capability: NonEmptyStr | None = None
    pci_bus_id: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_memory(self) -> GpuDeviceInfo:
        if self.free_memory_bytes > self.total_memory_bytes:
            raise ValueError("GPU free memory cannot exceed total memory")
        if self.used_memory_bytes > self.total_memory_bytes:
            raise ValueError("GPU used memory cannot exceed total memory")
        return self


class GpuTopologyLink(BaseSchema):
    """Pairwise topology affinity between anonymized GPU identities."""

    source_device_id: NonEmptyStr
    target_device_id: NonEmptyStr
    link_type: NonEmptyStr
    affinity_score: NonNegativeInt = Field(le=100)

    @model_validator(mode="after")
    def _validate_link(self) -> GpuTopologyLink:
        if self.source_device_id == self.target_device_id:
            raise ValueError("GPU topology links must connect distinct devices")
        return self


class ModelMemoryEstimate(BaseSchema):
    """Conservative static lower bound used before authoritative calibration."""

    model_alias: NonEmptyStr
    model_commit: NonEmptyStr
    tensor_parallel_size: PositiveInt
    weight_bytes: PositiveInt
    sharded_weight_bytes_per_gpu: PositiveInt
    weight_safety_factor: FiniteFloat = Field(ge=1.0)
    service_overhead_bytes: NonNegativeInt
    system_reserve_bytes: NonNegativeInt
    required_bytes_per_gpu: PositiveInt
    gpu_memory_utilization: Probability
    max_model_len: PositiveInt | None = None
    max_num_seqs: PositiveInt | None = None
    exact_weight_measurement: bool = True
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_required_memory(self) -> ModelMemoryEstimate:
        minimum = int(self.sharded_weight_bytes_per_gpu * self.weight_safety_factor)
        minimum += self.service_overhead_bytes
        if self.required_bytes_per_gpu < minimum:
            raise ValueError("required_bytes_per_gpu is below the declared static lower bound")
        return self


class ManagedServiceSpec(BaseSchema):
    """Strict, secret-free specification for one managed child service."""

    service_id: NonEmptyStr
    role: ModelRole
    backend: ManagedBackendKind
    model_alias: NonEmptyStr
    stages: list[NonEmptyStr]
    dtype: NonEmptyStr = "bfloat16"
    quantization: NonEmptyStr | None = None
    max_model_len: PositiveInt | None = None
    max_num_seqs: PositiveInt | None = None
    max_num_batched_tokens: PositiveInt | None = None
    tensor_parallel_candidates: list[PositiveInt] = Field(default_factory=lambda: [1])
    gpu_memory_utilization: Probability = 0.9
    startup_timeout_seconds: PositiveInt = 900
    port: PositiveInt | None = Field(default=None, le=65535)
    extra_args: list[NonEmptyStr] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_service(self) -> ManagedServiceSpec:
        if not self.stages:
            raise ValueError("managed services require at least one runtime stage")
        ensure_unique_strings(self.stages, "stages")
        if self.tensor_parallel_candidates != sorted(set(self.tensor_parallel_candidates)):
            raise ValueError("tensor_parallel_candidates must be unique and ascending")
        if self.backend is ManagedBackendKind.LITELLM_PROXY:
            raise ValueError("LiteLLM proxy is generated by the harness, not a GPU service")
        served_roles = self.metadata.get("served_roles", [])
        served_aliases = self.metadata.get("served_aliases", [])
        if not isinstance(served_roles, list) or any(
            not isinstance(value, str) or value not in {role.value for role in ModelRole}
            for value in served_roles
        ):
            raise ValueError("served_roles must contain known model-role names")
        if len(served_roles) != len(set(served_roles)):
            raise ValueError("served_roles must be unique")
        if not isinstance(served_aliases, list) or any(
            not isinstance(value, str) or not value.strip() for value in served_aliases
        ):
            raise ValueError("served_aliases must contain non-empty aliases")
        if len(served_aliases) != len(set(served_aliases)):
            raise ValueError("served_aliases must be unique")
        thinking_control = self.metadata.get("supports_thinking_control", False)
        if not isinstance(thinking_control, bool):
            raise ValueError("supports_thinking_control must be boolean")
        if thinking_control and self.backend is not ManagedBackendKind.VLLM_CHAT:
            raise ValueError("thinking control is only supported by managed chat services")
        return self

    @property
    def all_served_aliases(self) -> list[str]:
        aliases = [self.model_alias, *self.metadata.get("served_aliases", [])]
        return list(dict.fromkeys(str(value) for value in aliases))


class RuntimeEndpointBinding(BaseSchema):
    runtime_service_id: NonEmptyStr
    service_id: NonEmptyStr
    endpoint: NonEmptyStr
    replica_index: NonNegativeInt


class RuntimeEndpointPool(BaseSchema):
    schema_version: NonEmptyStr = RUNTIME_REPLICA_SCHEMA_VERSION
    runtime_id: NonEmptyStr
    stage_id: NonEmptyStr
    model_alias: NonEmptyStr
    bindings: list[RuntimeEndpointBinding]

    @model_validator(mode="after")
    def _validate_pool(self) -> RuntimeEndpointPool:
        if self.schema_version != RUNTIME_REPLICA_SCHEMA_VERSION:
            raise ValueError("unsupported runtime endpoint-pool schema")
        if not self.bindings:
            raise ValueError("runtime endpoint pools require at least one binding")
        ensure_unique_strings(
            [item.runtime_service_id for item in self.bindings],
            "bindings.runtime_service_id",
        )
        if [item.replica_index for item in self.bindings] != list(range(len(self.bindings))):
            raise ValueError("runtime endpoint replica indexes must be contiguous")
        if any(not item.endpoint.startswith("http://127.0.0.1:") for item in self.bindings):
            raise ValueError("managed endpoint pools must remain loopback-only")
        return self


class GpuAllocation(BaseSchema):
    """One deterministic service-to-GPU allocation."""

    service_id: NonEmptyStr
    device_ids: list[NonEmptyStr]
    visible_indices: list[NonNegativeInt]
    tensor_parallel_size: PositiveInt
    memory_budget_bytes_per_gpu: PositiveInt
    estimate: ModelMemoryEstimate
    topology_score: NonNegativeFloat | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_devices(self) -> GpuAllocation:
        ensure_unique_strings(self.device_ids, "device_ids")
        if len(self.visible_indices) != len(set(self.visible_indices)):
            raise ValueError("visible_indices cannot contain duplicates")
        if len(self.device_ids) != self.tensor_parallel_size:
            raise ValueError("device count must equal tensor_parallel_size")
        if len(self.visible_indices) != self.tensor_parallel_size:
            raise ValueError("visible index count must equal tensor_parallel_size")
        if self.estimate.tensor_parallel_size != self.tensor_parallel_size:
            raise ValueError("allocation and estimate tensor parallel sizes must match")
        return self


class RuntimeStage(BaseSchema):
    """A set of services that can run concurrently on disjoint GPUs."""

    stage_id: NonEmptyStr
    ordinal: NonNegativeInt
    service_ids: list[NonEmptyStr]
    allocations: list[GpuAllocation]
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_stage(self) -> RuntimeStage:
        if not self.service_ids:
            raise ValueError("runtime stages require at least one service")
        ensure_unique_strings(self.service_ids, "service_ids")
        allocation_ids = [item.service_id for item in self.allocations]
        if allocation_ids != self.service_ids:
            raise ValueError("stage allocations must follow and cover service_ids")
        allocated_devices = [device for item in self.allocations for device in item.device_ids]
        ensure_unique_strings(allocated_devices, "allocations.device_ids")
        return self


class ManagedRuntimePlan(BaseSchema):
    """Immutable placement and staging plan for one managed runtime."""

    runtime_id: NonEmptyStr
    profile_name: NonEmptyStr
    profile_hash: NonEmptyStr
    hardware_fingerprint: NonEmptyStr
    runtime_schema_version: NonEmptyStr = RUNTIME_SCHEMA_VERSION
    mode: RuntimeMode = RuntimeMode.MANAGED_LOCAL
    services: list[ManagedServiceSpec]
    devices: list[GpuDeviceInfo]
    topology: list[GpuTopologyLink] = Field(default_factory=list)
    stages: list[RuntimeStage]
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_plan(self) -> ManagedRuntimePlan:
        ensure_unique_strings([item.service_id for item in self.services], "services.service_id")
        ensure_unique_strings([item.device_id for item in self.devices], "devices.device_id")
        ensure_unique_strings([item.stage_id for item in self.stages], "stages.stage_id")
        if [item.ordinal for item in self.stages] != list(range(len(self.stages))):
            raise ValueError("runtime stage ordinals must be contiguous from zero")
        known_services = {item.service_id for item in self.services}
        planned_services = {service for stage in self.stages for service in stage.service_ids}
        if not planned_services <= known_services:
            raise ValueError("runtime plan references an unknown service")
        return self


class RuntimeCalibration(BaseSchema):
    """Observed startup result tied to exact software, model, and hardware."""

    calibration_id: NonEmptyStr
    service_id: NonEmptyStr
    model_alias: NonEmptyStr
    model_commit: NonEmptyStr
    profile_hash: NonEmptyStr
    hardware_fingerprint: NonEmptyStr
    vllm_version: NonEmptyStr
    status: CalibrationStatus
    allocation: GpuAllocation
    observed_peak_memory_bytes: NonNegativeInt
    observed_idle_memory_bytes: NonNegativeInt
    server_config: JsonDict = Field(default_factory=dict)
    error_type: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_calibration(self) -> RuntimeCalibration:
        if self.status is CalibrationStatus.FAILED and self.error_type is None:
            raise ValueError("failed calibrations require error_type")
        if self.status is CalibrationStatus.READY and self.error_type is not None:
            raise ValueError("ready calibrations cannot contain error_type")
        return self


class ManagedRuntimeSession(BaseSchema):
    """Queryable lifecycle summary for a managed runtime invocation."""

    runtime_id: NonEmptyStr
    profile_name: NonEmptyStr
    profile_hash: NonEmptyStr
    hardware_fingerprint: NonEmptyStr
    status: RuntimeStatus
    vllm_version: NonEmptyStr | None = None
    litellm_version: NonEmptyStr | None = None
    plan_artifact_id: NonEmptyStr | None = None
    summary_artifact_id: NonEmptyStr | None = None
    started_at: UtcDateTime = Field(default_factory=utc_now)
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_session(self) -> ManagedRuntimeSession:
        terminal = {RuntimeStatus.COMPLETED, RuntimeStatus.FAILED, RuntimeStatus.CANCELLED}
        if self.status in terminal and self.completed_at is None:
            raise ValueError("terminal runtime sessions require completed_at")
        if self.status is RuntimeStatus.FAILED and self.error_type is None:
            raise ValueError("failed runtime sessions require error_type")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self


class ManagedServiceState(BaseSchema):
    """Queryable lifecycle and resource summary for one child service."""

    runtime_service_id: NonEmptyStr
    runtime_id: NonEmptyStr
    service_id: NonEmptyStr
    stage_id: NonEmptyStr
    role: ModelRole
    backend: ManagedBackendKind
    model_alias: NonEmptyStr
    status: ManagedServiceStatus
    allocation: GpuAllocation
    process_id: NonNegativeInt | None = None
    observed_peak_memory_bytes: NonNegativeInt | None = None
    started_at: UtcDateTime | None = None
    ready_at: UtcDateTime | None = None
    stopped_at: UtcDateTime | None = None
    exit_code: int | None = None
    error_type: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> ManagedServiceState:
        if self.status is ManagedServiceStatus.FAILED and self.error_type is None:
            raise ValueError("failed managed services require error_type")
        if self.ready_at is not None and self.started_at is None:
            raise ValueError("ready_at requires started_at")
        if self.stopped_at is not None and self.started_at is None:
            raise ValueError("stopped_at requires started_at")
        return self
