"""Fail-closed, measured capacity evidence for explicitly frozen Step 8 pools.

The isolated writer lives in step8_capacity_calibration. Normal campaign startup
only reads evidence; legacy profiles without requires_capacity_receipt are unaffected.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from pmmc.runtime.device_accounting import DeviceLifecycleEvidence, validate_device_lifecycle
from pmmc.runtime.memory_budget import VllmMemoryBudget, allocation_vllm_memory_budget
from pmmc.runtime.step8_capacity_native import NativeCapacityImageWorkload, verify_native_processor
from pmmc.schemas import (
    BaseSchema,
    GpuAllocation,
    ManagedServiceSpec,
    ModelCapability,
    RuntimeStage,
)

if TYPE_CHECKING:
    from pmmc.runtime.manager import PreparedRuntime


class CapacityAllocation(BaseSchema):
    service_id: str
    device_ids: list[str]
    visible_indices: list[Annotated[int, Field(strict=True, ge=0)]]


PositiveCount = Annotated[int, Field(strict=True, gt=0)]


class CapacityImageWorkload(BaseSchema):
    images_per_sequence: Annotated[int, Field(strict=True, ge=1, le=64)]
    image_width: Annotated[int, Field(strict=True, ge=32, le=4096)]
    image_height: Annotated[int, Field(strict=True, ge=32, le=4096)]


ImageWorkload = CapacityImageWorkload | NativeCapacityImageWorkload


def parse_image_workload(value: object) -> ImageWorkload:
    if isinstance(value, dict) and value.get("protocol") == "native-image-layouts-v1":
        return NativeCapacityImageWorkload.model_validate(value)
    return CapacityImageWorkload.model_validate(value)


class CapacityReplicaWorkload(BaseSchema):
    service_id: str
    completed_sequences: PositiveCount
    prompt_tokens_per_sequence: PositiveCount
    completion_tokens_per_sequence: PositiveCount
    observed_running_sequences: PositiveCount
    completed_image_sequences: Annotated[int, Field(strict=True, ge=0)] = 0
    image_request_hashes: list[str] = Field(default_factory=list)


class CapacityScenarioMeasurement(BaseSchema):
    layout_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    peak_memory_bytes_by_device: dict[str, PositiveCount]
    full_pool_concurrent_samples: PositiveCount
    replica_workloads: list[CapacityReplicaWorkload]


class Step8PoolCapacityReceipt(BaseSchema):
    protocol: Literal["step8-pool-capacity-v1", "step8-pool-capacity-v2", "step8-pool-capacity-v3",
                      "step8-pool-capacity-v4"]
    status: Literal["passed"]
    measured_at: datetime
    service_id: str
    model_alias: str
    model_repo_id: str
    model_commit: str
    model_snapshot_content_hash: str
    model_registry_entry_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    profile_hash: str
    hardware_fingerprint: str
    backend_version: str
    backend: str
    tensor_parallel_size: PositiveCount
    replica_count: PositiveCount
    allocations: list[CapacityAllocation]
    memory_budget_contracts: list[VllmMemoryBudget] = Field(default_factory=list)
    tested_concurrent_sequences_per_replica: PositiveCount
    tested_total_tokens_per_sequence: PositiveCount
    peak_memory_bytes_by_device: dict[str, PositiveCount]
    measurement_source: Literal["nvml_compute_process_memory", "nvml_allocated_device_total"]
    device_lifecycle: DeviceLifecycleEvidence | None = None
    full_pool_concurrent_samples: PositiveCount
    replica_workloads: list[CapacityReplicaWorkload]
    workload_kind: Literal["text", "multimodal"] = "text"
    image_workload: ImageWorkload | None = None
    scenario_measurements: list[CapacityScenarioMeasurement] = Field(default_factory=list)


def _capacity_json_option(source: ManagedServiceSpec, name: str) -> object:
    values = []
    for index, item in enumerate(source.extra_args):
        if item == name:
            if index + 1 >= len(source.extra_args):
                raise ValueError("Step 8 capacity image server option has no value")
            values.append(source.extra_args[index + 1])
        elif item.startswith(name + "="):
            values.append(item.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError("Step 8 capacity requires one explicit image server limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Step 8 capacity server option has duplicate JSON keys")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Step 8 capacity server option cannot contain nonfinite JSON values")

    try:
        return json.loads(values[0], object_pairs_hook=unique, parse_constant=nonfinite)
    except json.JSONDecodeError:
        raise ValueError("Step 8 capacity server option must be valid JSON") from None


def capacity_image_workload(
    source: ManagedServiceSpec, *, multimodal: bool
) -> ImageWorkload | None:
    """Bind measurement bounds to explicitly frozen server processor settings."""
    raw = source.metadata.get("capacity_image_workload")
    if raw is None:
        if multimodal:
            raise ValueError("Step 8 multimodal capacity requires explicit image workload bounds")
        return None
    bounds = parse_image_workload(raw)
    limits = _capacity_json_option(source, "--limit-mm-per-prompt")
    processor = _capacity_json_option(source, "--mm-processor-kwargs")
    if isinstance(bounds, NativeCapacityImageWorkload):
        prefix_flags = [arg for arg in source.extra_args if arg.split("=", 1)[0] in {
            "--enable-prefix-caching", "--no-enable-prefix-caching",
        }]
        if prefix_flags != ["--no-enable-prefix-caching"]:
            raise ValueError("native capacity requires explicit disabled prefix caching")
        if (limits != {"image": bounds.images_per_sequence} or processor != {}
                or type(limits.get("image")) is not int):
            raise ValueError("native capacity must preserve locked processor defaults")
        if not source.max_model_len or any(
            sum(image.image_tokens for image in layout.images) >= source.max_model_len
            for layout in bounds.layouts
        ):
            raise ValueError("capacity native layout exceeds context before text")
        return bounds
    if (
        not isinstance(limits, dict)
        or type(limits.get("image")) is not int
        or limits["image"] != bounds.images_per_sequence
        or not isinstance(processor, dict)
        or type(processor.get("max_pixels")) is not int
        or processor["max_pixels"] != bounds.image_width * bounds.image_height
    ):
        raise ValueError("Step 8 capacity image workload differs from enforced server limits")
    return bounds


def capacity_processor_kwargs(source: ManagedServiceSpec) -> dict[str, object]:
    """No inferred defaults: reuse the same reviewed kwargs at count and dispatch."""
    capacity_image_workload(source, multimodal=True)
    value = _capacity_json_option(source, "--mm-processor-kwargs")
    assert isinstance(value, dict)
    return value


@dataclass(frozen=True)
class CapacityPool:
    services: list[ManagedServiceSpec]
    allocations: list[GpuAllocation]
    filename: str
    identity: dict[str, object]


def capacity_pools(
    prepared: PreparedRuntime,
    stage: RuntimeStage,
    *,
    backend_version: Callable[[ManagedServiceSpec], str],
) -> list[CapacityPool]:
    """Resolve gated full pools from the source profile, not just mutable plan flags."""
    groups: dict[str, list[ManagedServiceSpec]] = {}
    sources = {item.service_id: item for item in prepared.profile.services}
    source_by_replica_id = {}
    for gated_source in sources.values():
        if gated_source.metadata.get("requires_capacity_receipt") is not True:
            continue
        source_by_replica_id[gated_source.service_id] = gated_source
        frozen = gated_source.metadata.get("frozen_replica_topology")
        count = frozen.get("replica_count") if isinstance(frozen, dict) else None
        if type(count) is int and count > 0:
            source_by_replica_id.update(
                {f"{gated_source.service_id}-r{i:02d}": gated_source for i in range(count)}
            )
    if set(stage.service_ids) & source_by_replica_id.keys():
        planned_ids = [spec.service_id for spec in prepared.plan.services]
        if any(planned_ids.count(service_id) != 1 for service_id in stage.service_ids):
            raise ValueError("Step 8 capacity stage has missing or duplicated planned services")
    for spec in prepared.plan.services:
        if spec.service_id not in stage.service_ids:
            continue
        template = str(spec.metadata.get("template_service_id", spec.service_id))
        source = sources.get(template)
        required_source = source_by_replica_id.get(spec.service_id)
        if required_source is not None and source is not required_source:
            raise ValueError("Step 8 capacity replica lost its source profile identity")
        if spec.metadata.get("requires_capacity_receipt") is True or (
            source is not None and source.metadata.get("requires_capacity_receipt") is True
        ):
            groups.setdefault(template, []).append(spec)
    pools = []
    for template, services in groups.items():
        source = sources.get(template)
        if source is None or source.metadata.get("requires_capacity_receipt") is not True:
            raise ValueError("Step 8 capacity pool has no matching gated source profile")
        if (
            prepared.plan.profile_hash != prepared.profile.resolved_hash
            or prepared.plan.hardware_fingerprint != prepared.inventory.fingerprint
        ):
            raise ValueError("Step 8 capacity plan identity differs from source")
        frozen = source.metadata.get("frozen_replica_topology")
        if not isinstance(frozen, dict):
            raise ValueError("Step 8 capacity gate requires frozen_replica_topology")
        tp, count = frozen.get("tensor_parallel_size"), frozen.get("replica_count")
        if (
            type(tp) is not int
            or type(count) is not int
            or tp <= 0
            or count <= 0
            or len(services) != count
            or source.tensor_parallel_candidates != [tp]
        ):
            raise ValueError("Step 8 runtime pool differs from frozen topology")
        replica_fields = {
            "replica_schema_version",
            "template_service_id",
            "replica_index",
            "replica_count",
        }
        source_value = source.model_dump(exclude={"service_id", "port"})
        indexes = []
        for spec in services:
            value = spec.model_dump(exclude={"service_id", "port"})
            value["metadata"] = {
                k: v for k, v in value["metadata"].items() if k not in replica_fields
            }
            if value != source_value:
                raise ValueError("Step 8 capacity replica differs from source profile")
            index = spec.metadata.get("replica_index", 0)
            if type(index) is not int or spec.metadata.get("replica_count", 1) != count:
                raise ValueError("Step 8 capacity replica identity is invalid")
            indexes.append(index)
        if sorted(indexes) != list(range(count)):
            raise ValueError("Step 8 capacity replica coverage is incomplete")
        if not source.max_num_seqs or not source.max_model_len:
            raise ValueError("Step 8 capacity requires explicit workload dimensions")
        filename = source.metadata.get("capacity_receipt_filename")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".json")
            or Path(filename).name != filename
            or filename.startswith(".")
        ):
            raise ValueError("Step 8 capacity receipt requires a plain JSON filename")
        ids = {item.service_id for item in services}
        allocations = sorted(
            (item for item in stage.allocations if item.service_id in ids),
            key=lambda item: item.service_id,
        )
        devices = [device for allocation in allocations for device in allocation.device_ids]
        indices = [index for allocation in allocations for index in allocation.visible_indices]
        inventory = {item.device_id: item.visible_index for item in prepared.inventory.devices}
        if (
            len(allocations) != count
            or {a.service_id for a in allocations} != ids
            or len(devices) != count * tp
            or len(set(devices)) != len(devices)
            or len(indices) != count * tp
            or len(set(indices)) != len(indices)
            or any(
                a.tensor_parallel_size != tp
                or len(a.device_ids) != tp
                or len(a.visible_indices) != tp
                for a in allocations
            )
            or any(
                inventory.get(device) != index
                for device, index in zip(devices, indices, strict=True)
            )
        ):
            raise ValueError("Step 8 capacity allocations must be concurrent and disjoint")
        version = backend_version(source)
        if (
            not version
            or version == "not-installed"
            or any(backend_version(spec) != version for spec in services)
        ):
            raise ValueError("Step 8 capacity backend identity is unavailable or inconsistent")
        entry = prepared.lock.get(source.model_alias)
        model = prepared.resolved_registry.get(source.model_alias)
        images = capacity_image_workload(
            source, multimodal=ModelCapability.CHAT_MULTIMODAL in model.capabilities,
        )
        if isinstance(images, NativeCapacityImageWorkload):
            snapshot = prepared.snapshot_paths.get(source.model_alias) or (
                prepared.settings.model_cache_path / ("models--" + entry.repo_id.replace("/", "--"))
                / "snapshots" / entry.commit_sha
            )
            verify_native_processor(snapshot, images)
        identity: dict[str, object] = {
            "service_id": template,
            "model_alias": source.model_alias,
            "model_repo_id": entry.repo_id,
            "model_commit": entry.commit_sha,
            "model_snapshot_content_hash": entry.snapshot_content_hash,
            "model_registry_entry_hash": model.resolved_hash,
            "profile_hash": prepared.profile.resolved_hash,
            "hardware_fingerprint": prepared.inventory.fingerprint,
            "backend": source.backend.value,
            "backend_version": version,
            "tensor_parallel_size": tp,
            "replica_count": count,
            "workload_kind": "multimodal" if images is not None else "text",
            "image_workload": images.model_dump() if images is not None else None,
            "allocations": [
                {
                    "service_id": a.service_id,
                    "device_ids": a.device_ids,
                    "visible_indices": a.visible_indices,
                }
                for a in allocations
            ],
        }
        if prepared.settings.gpu_memory_accounting == "allocated_device":
            identity["measurement_source"] = "nvml_allocated_device_total"
        if any("vllm_memory_budget" in a.metadata for a in allocations):
            identity["memory_budget_contracts"] = [
                allocation_vllm_memory_budget(a).model_dump() for a in allocations
            ]
        pools.append(CapacityPool(services, allocations, filename, identity))
    return pools


def validate_capacity_receipt(pool: CapacityPool, receipt: Step8PoolCapacityReceipt) -> None:
    """Shared read/write validation; a writer cannot authorize weaker evidence."""
    if receipt.model_dump(mode="python", include=set(pool.identity)) != pool.identity:
        raise ValueError("Step 8 measured capacity receipt identity does not match")
    source = pool.identity.get("measurement_source", "nvml_compute_process_memory")
    if receipt.measurement_source != source:
        raise ValueError("Step 8 capacity memory accounting policy differs")
    device_mode = source == "nvml_allocated_device_total"
    if device_mode:
        if receipt.protocol != "step8-pool-capacity-v4" or receipt.device_lifecycle is None:
            raise ValueError("Step 8 device capacity requires v4 lifecycle evidence")
        validate_device_lifecycle(receipt.device_lifecycle, pool.allocations,
                                  receipt.peak_memory_bytes_by_device)
        if any(v.sampled_at > receipt.measured_at for v in receipt.device_lifecycle.after):
            raise ValueError("Step 8 capacity was published before device release")
    elif receipt.protocol == "step8-pool-capacity-v4" or receipt.device_lifecycle is not None:
        raise ValueError("Step 8 process capacity cannot use device evidence")
    native = isinstance(receipt.image_workload, NativeCapacityImageWorkload)
    if receipt.workload_kind == "multimodal" and (
        receipt.protocol != ("step8-pool-capacity-v4" if device_mode else
                             "step8-pool-capacity-v3" if native else "step8-pool-capacity-v2")
        or receipt.image_workload is None
    ):
        raise ValueError("Step 8 vision capacity cannot use a text-only receipt")
    if native:
        assert isinstance(receipt.image_workload, NativeCapacityImageWorkload)
        expected = [layout.identity for layout in receipt.image_workload.layouts]
        if [item.layout_hash for item in receipt.scenario_measurements] != expected:
            raise ValueError("capacity native scenario coverage is incomplete")
        if receipt.replica_workloads != receipt.scenario_measurements[0].replica_workloads:
            raise ValueError("capacity native primary accounting differs")
        if receipt.full_pool_concurrent_samples != sum(
            item.full_pool_concurrent_samples for item in receipt.scenario_measurements
        ):
            raise ValueError("capacity native concurrency accounting differs")
        devices = receipt.peak_memory_bytes_by_device
        if any(set(item.peak_memory_bytes_by_device) != set(devices)
               for item in receipt.scenario_measurements) or devices != {
            device: max(item.peak_memory_bytes_by_device[device]
                        for item in receipt.scenario_measurements) for device in devices
        }:
            raise ValueError("capacity native peak accounting differs")
        all_hashes = [h for item in receipt.scenario_measurements
                      for work in item.replica_workloads for h in work.image_request_hashes]
        if len(all_hashes) != len(set(all_hashes)):
            raise ValueError("capacity native scenarios reused image requests")
        for item in receipt.scenario_measurements:
            _validate_measurement(pool, receipt.model_copy(update={
                "replica_workloads": item.replica_workloads,
                "peak_memory_bytes_by_device": item.peak_memory_bytes_by_device,
                "full_pool_concurrent_samples": item.full_pool_concurrent_samples,
            }))
    elif receipt.scenario_measurements:
        raise ValueError("legacy capacity cannot contain native scenarios")
    _validate_measurement(pool, receipt)


def _validate_measurement(pool: CapacityPool, receipt: Step8PoolCapacityReceipt) -> None:
    spec = pool.services[0]
    if (
        receipt.measured_at.tzinfo is None
        or receipt.measured_at > datetime.now(UTC)
        or receipt.tested_concurrent_sequences_per_replica != spec.max_num_seqs
        or receipt.tested_total_tokens_per_sequence != spec.max_model_len
    ):
        raise ValueError("Step 8 capacity receipt does not cover configured workload")
    ids = {item.service_id for item in pool.services}
    if (
        len(receipt.replica_workloads) != len(ids)
        or {item.service_id for item in receipt.replica_workloads} != ids
    ):
        raise ValueError("Step 8 capacity workload must cover every replica exactly once")
    for workload in receipt.replica_workloads:
        if (
            workload.completed_sequences != spec.max_num_seqs
            or workload.observed_running_sequences != spec.max_num_seqs
            or workload.prompt_tokens_per_sequence + workload.completion_tokens_per_sequence
            != spec.max_model_len
        ):
            raise ValueError("Step 8 capacity receipt does not cover configured workload")
        if receipt.workload_kind == "multimodal":
            if (
                workload.completed_image_sequences != spec.max_num_seqs
                or len(workload.image_request_hashes) != spec.max_num_seqs
                or any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                       for value in workload.image_request_hashes)
            ):
                raise ValueError("Step 8 capacity receipt lacks completed image requests")
        elif workload.completed_image_sequences or workload.image_request_hashes:
            raise ValueError("Step 8 text capacity contains inconsistent image evidence")
    hashes = [value for item in receipt.replica_workloads for value in item.image_request_hashes]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Step 8 capacity image requests must be distinct across the full pool")
    peaks = receipt.peak_memory_bytes_by_device
    devices = {device for a in pool.allocations for device in a.device_ids}
    if set(peaks) != devices or any(
        not 0 < peaks[device] <= allocation.memory_budget_bytes_per_gpu
        for allocation in pool.allocations
        for device in allocation.device_ids
    ):
        raise ValueError("Step 8 measured capacity exceeds the planned GPU memory budget")


def require_step8_capacity_receipts(
    prepared: PreparedRuntime,
    stage: RuntimeStage,
    *,
    backend_version: Callable[[ManagedServiceSpec], str],
) -> set[str]:
    """Return gated service IDs; their full-pool receipt supersedes idle calibration.

    Each filename is relative to Settings.runtime_calibrations_path. Unlike the
    legacy paper-only idle check, this opt-in capacity gate cannot be disabled by
    selecting dev mode or runtime_require_calibration_paper=False.
    """
    gated: set[str] = set()
    for pool in capacity_pools(prepared, stage, backend_version=backend_version):
        path = prepared.settings.runtime_calibrations_path / pool.filename
        if not path.is_file():
            raise ValueError(f"Step 8 measured capacity receipt is missing: {pool.filename}")
        receipt = Step8PoolCapacityReceipt.from_json(path.read_text(encoding="utf-8"))
        validate_capacity_receipt(pool, receipt)
        gated.update(item.service_id for item in pool.services)
    return gated
