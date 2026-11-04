"""Deterministic, topology-aware exclusive GPU placement."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pmmc.models.registry import ModelRegistry
from pmmc.runtime.config import RuntimeProfile
from pmmc.runtime.hardware import HardwareInventory
from pmmc.runtime.memory import estimate_service_memory
from pmmc.runtime.memory_budget import derive_vllm_memory_budget
from pmmc.schemas import (
    RUNTIME_REPLICA_SCHEMA_VERSION,
    GpuAllocation,
    GpuDeviceInfo,
    ManagedBackendKind,
    ManagedRuntimePlan,
    ManagedServiceSpec,
    ModelLockManifest,
    ModelRole,
    RuntimeStage,
)


class RuntimePlanningError(RuntimeError):
    """Raised when no safe placement exists on the GPUs visible to the Pod."""


def _replica_tensor_parallel_size(service: ManagedServiceSpec, count: int) -> int:
    frozen = service.metadata.get("frozen_replica_topology")
    if frozen is None:
        if 1 not in service.tensor_parallel_candidates:
            raise RuntimePlanningError("replicated services require TP=1 support")
        return 1
    if not isinstance(frozen, dict) or set(frozen) != {"tensor_parallel_size", "replica_count"}:
        raise RuntimePlanningError("invalid frozen_replica_topology capability")
    tp = frozen["tensor_parallel_size"]
    replicas = frozen["replica_count"]
    if any(type(value) is not int or value <= 0 for value in (tp, replicas)):
        raise RuntimePlanningError("frozen replica topology requires positive integers")
    if count != replicas or service.tensor_parallel_candidates != [tp]:
        raise RuntimePlanningError(
            "requested replicas/TP differ from frozen_replica_topology; "
            "explicit profile change and new capacity receipt required"
        )
    return tp


def _expand_replica_services(
    services: list[ManagedServiceSpec],
    replica_counts: Mapping[ModelRole, int],
) -> list[ManagedServiceSpec]:
    expanded: list[ManagedServiceSpec] = []
    ports: set[int] = set()
    for service in services:
        count = int(replica_counts.get(service.role, 1))
        if count <= 0:
            raise RuntimePlanningError("runtime replica counts must be positive")
        if count == 1:
            expanded.append(service)
            if service.port is not None:
                ports.add(service.port)
            continue
        if service.port is None:
            raise RuntimePlanningError("replicated services require an explicit base port")
        tp_size = _replica_tensor_parallel_size(service, count)
        next_port = service.port
        for replica_index in range(count):
            while next_port in ports:
                next_port += 1
            port = next_port
            if port > 65535:
                raise RuntimePlanningError("replica service ports exhausted")
            ports.add(port)
            next_port += 1
            expanded.append(
                service.model_copy(
                    update={
                        "service_id": f"{service.service_id}-r{replica_index:02d}",
                        "port": port,
                        "tensor_parallel_candidates": [tp_size],
                        "metadata": {
                            **service.metadata,
                            "replica_schema_version": RUNTIME_REPLICA_SCHEMA_VERSION,
                            "template_service_id": service.service_id,
                            "replica_index": replica_index,
                            "replica_count": count,
                        },
                    }
                )
            )
    return expanded


def _deduplicate_selected_services(
    services: list[ManagedServiceSpec],
) -> list[ManagedServiceSpec]:
    """Build one authoritative service catalog across logical stages."""

    unique: list[ManagedServiceSpec] = []
    by_id: dict[str, ManagedServiceSpec] = {}
    for service in services:
        existing = by_id.get(service.service_id)
        if existing is None:
            by_id[service.service_id] = service
            unique.append(service)
            continue
        if existing != service:
            raise RuntimePlanningError(
                "shared runtime service specification changed across logical stages: "
                f"{service.service_id}"
            )
    return unique


def make_runtime_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"runtime-{timestamp}-{uuid4().hex[:10]}"


def _topology_score(
    devices: tuple[GpuDeviceInfo, ...],
    inventory: HardwareInventory,
) -> float:
    if len(devices) <= 1:
        return 100.0
    lookup = {
        (link.source_device_id, link.target_device_id): link.affinity_score
        for link in inventory.topology
    }
    scores = [
        lookup.get((left.device_id, right.device_id), 0)
        for left, right in itertools.combinations(devices, 2)
    ]
    return sum(scores) / len(scores)


def _candidate_allocations(
    service: ManagedServiceSpec,
    profile: RuntimeProfile,
    lock: ModelLockManifest,
    snapshot_paths: dict[str, Path],
    inventory: HardwareInventory,
    available: list[GpuDeviceInfo],
) -> list[GpuAllocation]:
    entry = lock.get(service.model_alias)
    snapshot = snapshot_paths.get(service.model_alias)
    if snapshot is None:
        raise RuntimePlanningError(f"model cache path is missing: {service.model_alias}")
    candidates: list[GpuAllocation] = []
    for tp_size in service.tensor_parallel_candidates:
        if tp_size > len(available):
            continue
        for devices in itertools.combinations(available, tp_size):
            minimum_total = min(item.total_memory_bytes for item in devices)
            estimate = estimate_service_memory(
                service,
                entry,
                snapshot,
                profile,
                tensor_parallel_size=tp_size,
                gpu_total_memory_bytes=minimum_total,
            )
            budgets = [
                min(
                    math.floor(item.total_memory_bytes * estimate.gpu_memory_utilization),
                    item.free_memory_bytes - estimate.system_reserve_bytes,
                )
                for item in devices
            ]
            if not budgets or min(budgets) < estimate.required_bytes_per_gpu:
                continue
            metadata = {}
            if service.backend in (
                ManagedBackendKind.VLLM_CHAT, ManagedBackendKind.VLLM_EMBEDDING,
                ManagedBackendKind.VLLM_MULTIMODAL_EMBEDDING,
            ):
                try:
                    contract = derive_vllm_memory_budget(
                        devices, min(budgets), estimate.gpu_memory_utilization,
                    )
                except ValueError:
                    continue
                if any(int(t * contract.effective_utilization) < estimate.required_bytes_per_gpu
                       for t in contract.total_bytes):
                    continue
                metadata["vllm_memory_budget"] = contract.to_dict()
            candidates.append(
                GpuAllocation(
                    service_id=service.service_id,
                    device_ids=[item.device_id for item in devices],
                    visible_indices=[item.visible_index for item in devices],
                    tensor_parallel_size=tp_size,
                    memory_budget_bytes_per_gpu=min(budgets),
                    estimate=estimate,
                    topology_score=_topology_score(devices, inventory),
                    metadata=metadata,
                )
            )
        if candidates:
            break
    return sorted(
        candidates,
        key=lambda item: (
            item.tensor_parallel_size,
            -(item.topology_score or 0),
            item.visible_indices,
        ),
    )


def _estimated_weight(
    service: ManagedServiceSpec,
    profile: RuntimeProfile,
    lock: ModelLockManifest,
    snapshot_paths: dict[str, Path],
    inventory: HardwareInventory,
) -> int:
    smallest_tp = service.tensor_parallel_candidates[0]
    estimate = estimate_service_memory(
        service,
        lock.get(service.model_alias),
        snapshot_paths[service.model_alias],
        profile,
        tensor_parallel_size=smallest_tp,
        gpu_total_memory_bytes=max(item.total_memory_bytes for item in inventory.devices),
    )
    return estimate.weight_bytes


def _allocation_failure_detail(
    service: ManagedServiceSpec,
    profile: RuntimeProfile,
    lock: ModelLockManifest,
    snapshot_paths: dict[str, Path],
    inventory: HardwareInventory,
) -> str:
    """Return a deterministic preflight diagnostic without loading CUDA."""

    visible = len(inventory.devices)
    largest_memory = max(item.total_memory_bytes for item in inventory.devices)
    feasible_by_capacity: list[int] = []
    for tp_size in service.tensor_parallel_candidates:
        estimate = estimate_service_memory(
            service,
            lock.get(service.model_alias),
            snapshot_paths[service.model_alias],
            profile,
            tensor_parallel_size=tp_size,
            gpu_total_memory_bytes=largest_memory,
        )
        static_budget = min(
            math.floor(largest_memory * estimate.gpu_memory_utilization),
            largest_memory - estimate.system_reserve_bytes,
        )
        if static_budget >= estimate.required_bytes_per_gpu:
            feasible_by_capacity.append(tp_size)
    minimum = min(feasible_by_capacity) if feasible_by_capacity else None
    minimum_text = str(minimum) if minimum is not None else "more than profile TP candidates"
    return (
        f"visible_gpus={visible}, minimum_estimated_gpus={minimum_text}, "
        f"allowed_tp={service.tensor_parallel_candidates}"
    )


def _plan_logical_stage(
    logical_stage: str,
    services: list[ManagedServiceSpec],
    profile: RuntimeProfile,
    lock: ModelLockManifest,
    snapshot_paths: dict[str, Path],
    inventory: HardwareInventory,
) -> list[tuple[list[ManagedServiceSpec], list[GpuAllocation]]]:
    ordered = sorted(
        services,
        key=lambda item: (
            -_estimated_weight(item, profile, lock, snapshot_paths, inventory),
            item.service_id,
        ),
    )
    parts: list[tuple[list[ManagedServiceSpec], list[GpuAllocation]]] = []
    current_services: list[ManagedServiceSpec] = []
    current_allocations: list[GpuAllocation] = []
    available = list(inventory.devices)
    for service in ordered:
        candidates = _candidate_allocations(
            service,
            profile,
            lock,
            snapshot_paths,
            inventory,
            available,
        )
        if not candidates and current_services:
            parts.append((current_services, current_allocations))
            current_services = []
            current_allocations = []
            available = list(inventory.devices)
            candidates = _candidate_allocations(
                service,
                profile,
                lock,
                snapshot_paths,
                inventory,
                available,
            )
        if not candidates:
            raise RuntimePlanningError(
                f"no safe GPU allocation for service {service.service_id!r} "
                f"in logical stage {logical_stage!r}; "
                + _allocation_failure_detail(
                    service,
                    profile,
                    lock,
                    snapshot_paths,
                    inventory,
                )
            )
        allocation = candidates[0]
        used = set(allocation.device_ids)
        available = [item for item in available if item.device_id not in used]
        current_services.append(service)
        current_allocations.append(allocation)
    if current_services:
        parts.append((current_services, current_allocations))
    return parts


def build_runtime_plan(
    profile: RuntimeProfile,
    registry: ModelRegistry,
    lock: ModelLockManifest,
    snapshot_paths: dict[str, Path],
    inventory: HardwareInventory,
    *,
    runtime_id: str | None = None,
    logical_stage: str | None = None,
    replica_counts: Mapping[ModelRole, int] | None = None,
) -> ManagedRuntimePlan:
    if not inventory.devices:
        raise RuntimePlanningError("the Pod exposes no GPUs")
    if "h100" in profile.hardware_class.lower() and any(
        "h100" not in device.name.lower() for device in inventory.devices
    ):
        raise RuntimePlanningError(
            f"runtime profile {profile.hardware_class!r} does not match visible GPU models"
        )
    if lock.registry_hash != registry.resolved_hash:
        raise RuntimePlanningError("model lock and registry hashes differ")
    stage_names = [logical_stage] if logical_stage else profile.stage_order
    stages: list[RuntimeStage] = []
    selected_services: list[ManagedServiceSpec] = []
    resolved_replica_counts = dict(replica_counts or {})
    for stage_name in stage_names:
        services = _expand_replica_services(
            profile.services_for_stage(stage_name),
            resolved_replica_counts,
        )
        selected_services.extend(services)
        parts = _plan_logical_stage(
            stage_name,
            services,
            profile,
            lock,
            snapshot_paths,
            inventory,
        )
        if (
            any(int(resolved_replica_counts.get(item.role, 1)) > 1 for item in services)
            and len(parts) != 1
        ):
            requested = max(int(resolved_replica_counts.get(item.role, 1)) for item in services)
            raise RuntimePlanningError(
                "replica pools must run concurrently; "
                f"requested_replicas={requested}, visible_gpus={len(inventory.devices)}"
            )
        for part_index, (part_services, allocations) in enumerate(parts):
            stage_id = stage_name if len(parts) == 1 else f"{stage_name}:part:{part_index}"
            stages.append(
                RuntimeStage(
                    stage_id=stage_id,
                    ordinal=len(stages),
                    service_ids=[item.service_id for item in part_services],
                    allocations=allocations,
                    metadata={"logical_stage": stage_name},
                )
            )
    selected_ids = {service_id for stage in stages for service_id in stage.service_ids}
    selected_services = [item for item in selected_services if item.service_id in selected_ids]
    selected_services = _deduplicate_selected_services(selected_services)
    return ManagedRuntimePlan(
        runtime_id=runtime_id or make_runtime_id(),
        profile_name=profile.profile_name,
        profile_hash=profile.resolved_hash,
        hardware_fingerprint=inventory.fingerprint,
        services=selected_services,
        devices=inventory.devices,
        topology=inventory.topology,
        stages=stages,
        metadata={
            "hardware_class": profile.hardware_class,
            "driver_version": inventory.driver_version,
            "cuda_driver_version": inventory.cuda_driver_version,
            "placement_policy": "exclusive_then_sequential",
            "replica_schema_version": RUNTIME_REPLICA_SCHEMA_VERSION,
            "replica_counts": {
                role.value: count
                for role, count in sorted(
                    resolved_replica_counts.items(), key=lambda item: item[0].value
                )
            },
        },
    )
