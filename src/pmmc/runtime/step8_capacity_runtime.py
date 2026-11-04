"""Read-only binding of a worker's capacity guard to its active managed pool."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from pmmc.infrastructure.settings import Settings
from pmmc.models.locking import load_model_lock
from pmmc.models.registry import ModelRegistry, load_model_registry
from pmmc.runtime.config import load_runtime_profile
from pmmc.runtime.hardware import HardwareInventory
from pmmc.runtime.manager import ManagedRuntimeError, PreparedRuntime
from pmmc.runtime.step8_capacity import Step8PoolCapacityReceipt, capacity_pools
from pmmc.runtime.step8_capacity_requests import (
    CapacityRequestGuard,
    preflight_capacity_request_contract,
)
from pmmc.schemas import ManagedRuntimePlan, RuntimeMode


def capacity_endpoint(value: str) -> str:
    """Canonical service root, with no external host/path/credential escape."""
    try:
        url = urlsplit(value)
        if (
            url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
            or url.path not in {"", "/", "/v1", "/v1/"}
            or url.username or url.password or url.query or url.fragment
        ):
            raise ValueError
        return f"http://127.0.0.1:{url.port}"
    except ValueError:
        raise ManagedRuntimeError("capacity_managed_endpoint_required") from None


@dataclass(frozen=True)
class BoundCapacityRuntime:
    guard: CapacityRequestGuard
    contract: dict
    backend_version: str
    bindings: list[tuple[int, str, str]]


def _capacity_configuration(settings: Settings, registry: ModelRegistry, alias: str):
    """Resolve the frozen request contract without inspecting a running service."""
    entry = registry.get(alias)
    if entry.provider_config is None or entry.provider_config.kind != "qwen_vllm":
        return None
    required_27b = (
        settings.runtime_mode is RuntimeMode.MANAGED_LOCAL
        and entry.provider_model == "Qwen/Qwen3.5-27B"
    )
    profile_path = settings.resolved_runtime_profile_path
    if not profile_path.is_file():
        if required_27b:
            raise ManagedRuntimeError("capacity_source_profile_required")
        return None
    profile = load_runtime_profile(profile_path)
    sources = [spec for spec in profile.services if alias in spec.all_served_aliases]
    gated = [spec for spec in sources if spec.metadata.get("requires_capacity_receipt") is True]
    if not gated and not required_27b:
        return None
    if len(sources) != 1 or len(gated) != 1:
        raise ManagedRuntimeError("capacity_gated_source_required")
    source = gated[0]
    # Missing scientific configuration is the first actionable error, before I/O to models.
    contract = preflight_capacity_request_contract(source, entry)
    return entry, profile, contract


def load_capacity_tokenization_contract(
    settings: Settings, registry: ModelRegistry, alias: str,
) -> dict | None:
    """Exact counting needs locked local assets, not a resident inference pool.

    Matching and grounding can count 27B inputs while a 9B service is resident.
    This contract does not authorize generation or substitute for a capacity receipt.
    """
    configuration = _capacity_configuration(settings, registry, alias)
    if configuration is None:
        return None
    entry, profile, contract = configuration
    try:
        lock = load_model_lock(
            settings.resolved_runtime_model_lock_path
            or settings.resolve_path(profile.model_lock_path)
        )
        locked = lock.get(alias)
        if (
            entry.model_revision != locked.commit_sha
            or entry.tokenizer_revision != locked.commit_sha
            or entry.processor_revision != locked.commit_sha
            or entry.resolved_source_repo_id != locked.repo_id
            or entry.tokenizer_name != locked.repo_id or entry.processor_name != locked.repo_id
        ):
            raise ValueError("tokenization identity mismatch")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ManagedRuntimeError("capacity_tokenization_identity_invalid") from None
    return contract


def load_capacity_runtime(
    settings: Settings, registry: ModelRegistry, alias: str,
) -> BoundCapacityRuntime | None:
    """Bind generation to the active measured pool, without loading models or calling transport."""
    configuration = _capacity_configuration(settings, registry, alias)
    if configuration is None:
        return None
    entry, profile, contract = configuration
    runtime_id = settings.active_runtime_id
    if (
        settings.runtime_mode is not RuntimeMode.MANAGED_LOCAL
        or not runtime_id or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", runtime_id) is None
        or runtime_id in {".", ".."}
    ):
        raise ManagedRuntimeError("capacity_active_runtime_required")
    run_dir = settings.runs_path / "managed-runtime" / runtime_id
    try:
        plan = ManagedRuntimePlan.from_json((run_dir / "runtime-plan.json").read_text())
        resolved = load_model_registry(run_dir / "resolved-model-registry.yaml")
        versions = json.loads((run_dir / "software-versions.json").read_text())
        lock = load_model_lock(
            settings.resolved_runtime_model_lock_path
            or settings.resolve_path(profile.model_lock_path)
        )
        locked = lock.get(alias)
        if (
            plan.runtime_id != runtime_id or plan.profile_hash != profile.resolved_hash
            or resolved.get(alias).resolved_hash != entry.resolved_hash
            or entry.model_revision != locked.commit_sha
            or entry.tokenizer_revision != locked.commit_sha
            or entry.processor_revision != locked.commit_sha
            or entry.resolved_source_repo_id != locked.repo_id
            or entry.tokenizer_name != locked.repo_id or entry.processor_name != locked.repo_id
        ):
            raise ValueError("identity mismatch")
        version = versions.get("vllm")
        if not isinstance(version, str) or not version or version == "not-installed":
            raise ValueError("backend unavailable")
        inventory = HardwareInventory(
            devices=plan.devices, topology=plan.topology,
            driver_version=str(plan.metadata["driver_version"]),
            cuda_driver_version=plan.metadata.get("cuda_driver_version"),
        )
        prepared = PreparedRuntime(
            settings=settings, profile=profile, source_registry=resolved,
            resolved_registry=resolved, lock=lock, snapshot_paths={}, inventory=inventory,
            plan=plan, run_dir=run_dir,
            resolved_registry_path=run_dir / "resolved-model-registry.yaml",
        )
        matches = [
            (stage, pool) for stage in plan.stages
            for pool in capacity_pools(prepared, stage, backend_version=lambda _: version)
            if pool.identity["model_alias"] == alias
        ]
        if len(matches) != 1:
            raise ValueError("one active capacity pool required")
        stage, pool = matches[0]
        receipt = Step8PoolCapacityReceipt.from_json(
            (settings.runtime_calibrations_path / pool.filename).read_text(),
        )
        guard = CapacityRequestGuard(pool, receipt, entry)
        expected = sorted([
            (int(spec.metadata.get("replica_index", 0)),
             f"{runtime_id}:{stage.stage_id}:{spec.service_id}", f"http://127.0.0.1:{spec.port}")
            for spec in pool.services
        ])
        rows = settings.active_runtime_endpoint_pools.get(alias, [])
        if any(type(row.get("replica_index")) is not int for row in rows):
            raise ValueError("invalid replica ordinal")
        actual = sorted([
            (row["replica_index"], row["runtime_service_id"], capacity_endpoint(row["endpoint"]))
            for row in rows
        ])
        if actual != expected or settings.active_runtime_service_map.get(alias) != expected[0][1]:
            raise ValueError("active replica bindings mismatch")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ManagedRuntimeError("capacity_active_runtime_evidence_invalid") from None
    return BoundCapacityRuntime(guard, contract, version, expected)
