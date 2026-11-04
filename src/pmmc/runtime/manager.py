"""Managed local runtime orchestration for one Kubernetes Pod."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from pydantic import SecretStr

from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.locking import (
    assert_paper_lock,
    load_model_lock,
    load_verified_model_cache_manifest,
    prepare_model_cache_audited,
    resolve_registry_from_lock,
    write_resolved_registry,
)
from pmmc.models.registry import ModelRegistry, load_model_registry
from pmmc.runtime.config import RuntimeProfile, load_runtime_profile
from pmmc.runtime.device_accounting import (
    AllocatedDeviceAccounting,
    DeviceLifecycleEvidence,
    DeviceMemoryBudgetExceeded,
    validate_device_lifecycle,
)
from pmmc.runtime.errors import ManagedRuntimeError as ManagedRuntimeError
from pmmc.runtime.hardware import HardwareInventory, NvmlHardwareProvider
from pmmc.runtime.locking import RuntimeProcessLock
from pmmc.runtime.memory_budget import allocation_vllm_memory_budget
from pmmc.runtime.planner import build_runtime_plan
from pmmc.runtime.processes import (
    ContrastiveServiceRunner,
    LiteLLMServiceRunner,
    ManagedProcessError,
    ManagedProcessHandle,
    ManagedServiceRunner,
    VllmServiceRunner,
    describe_process,
    endpoint_overlay,
    fetch_service_info,
    random_runtime_key,
    redact_service_log,
    service_log_diagnostic_codes,
)
from pmmc.runtime.signals import ManagedSignalGuard
from pmmc.schemas import (
    CalibrationStatus,
    GpuAllocation,
    ManagedBackendKind,
    ManagedRuntimePlan,
    ManagedRuntimeSession,
    ManagedServiceSpec,
    ManagedServiceState,
    ManagedServiceStatus,
    ModelLockEntry,
    ModelLockManifest,
    ModelVerificationPolicy,
    ModelVerificationResult,
    RuntimeCalibration,
    RuntimeStage,
    RuntimeStatus,
)
from pmmc.schemas.base import utc_now


def _retain_runtime_failure(
    primary: BaseException | None,
    secondary: BaseException,
    phase: str,
) -> BaseException:
    if primary is None:
        return secondary
    if secondary is not primary:
        primary.add_note(f"{phase} failed: {type(secondary).__name__}")
    return primary


class ManagedChildCommandError(ManagedRuntimeError):
    """A child command failed after the managed services became ready."""

    def __init__(
        self,
        *,
        returncode: int,
        command_ordinal: int,
        command_count: int,
        locator: dict[str, object],
    ) -> None:
        self.returncode = returncode
        self.command_ordinal = command_ordinal
        self.command_count = command_count
        self.locator = dict(locator)
        action = str(locator.get("action", "unknown"))
        phase = str(locator.get("phase", "unknown"))
        super().__init__(
            "managed child command exited with code "
            f"{returncode} (ordinal={command_ordinal}/{command_count} "
            f"action={action} phase={phase})"
        )


def _managed_command_locator(command: list[str]) -> dict[str, object]:
    """Return a stable, argument-safe child command locator."""

    action = Path(command[0]).name if command else "unknown"
    try:
        module_index = command.index("-m")
    except ValueError:
        module_index = -1
    if (
        module_index >= 0
        and len(command) > module_index + 2
        and command[module_index + 1] == "pmmc"
    ):
        action = command[module_index + 2]

    locator: dict[str, object] = {"action": action}
    if module_index >= 0 and command[module_index + 1 : module_index + 4] == [
        "pmmc",
        "runtime",
        "execute-query-run",
    ]:
        locator["phase"] = "query_executor"
    for option, key in (("--phase", "phase"), ("--channel", "channel")):
        try:
            option_index = command.index(option)
        except ValueError:
            continue
        if option_index + 1 < len(command):
            locator[key] = command[option_index + 1]
    return locator


@dataclass(frozen=True)
class PreparedRuntime:
    settings: Settings
    profile: RuntimeProfile
    source_registry: ModelRegistry
    resolved_registry: ModelRegistry
    lock: ModelLockManifest
    snapshot_paths: dict[str, Path]
    inventory: HardwareInventory
    plan: ManagedRuntimePlan
    run_dir: Path
    resolved_registry_path: Path
    model_verifications: dict[str, ModelVerificationResult] | None = None
    local_staging: dict[str, object] | None = None


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_LOCAL_MODEL_STAGE_PROTOCOL = "verified-model-snapshot-local-stage-v1"
_COPY_BUFFER_BYTES = 8 * 1024 * 1024


def _enabled_environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _snapshot_stage_identity(entry: ModelLockEntry) -> str:
    return hashlib.sha256(
        _stable_json(
            {
                "commit_sha": entry.commit_sha,
                "repo_id": entry.repo_id,
                "snapshot_content_hash": entry.snapshot_content_hash,
            }
        ).encode("utf-8")
    ).hexdigest()


def _valid_local_snapshot(
    *,
    target: Path,
    entry: ModelLockEntry,
    identity: str,
) -> bool:
    manifest_path = target / "pmmc-local-snapshot.json"
    if not manifest_path.is_file():
        return False
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(value, dict):
        return False
    if (
        value.get("protocol") != _LOCAL_MODEL_STAGE_PROTOCOL
        or value.get("identity") != identity
        or value.get("commit_sha") != entry.commit_sha
        or value.get("snapshot_content_hash") != entry.snapshot_content_hash
        or value.get("file_count") != len(entry.files)
        or not isinstance(value.get("total_bytes"), int)
        or int(value["total_bytes"]) < 0
    ):
        return False
    for record in entry.files:
        candidate = target / record.path
        try:
            if not candidate.is_file() or (
                record.size_bytes and candidate.stat().st_size != record.size_bytes
            ):
                return False
        except OSError:
            return False
    return True


def _remove_local_snapshot(path: Path) -> None:
    if not path.exists():
        return
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir()),
        reverse=True,
    ):
        try:
            directory.chmod(0o700)
        except OSError:
            pass
    try:
        path.chmod(0o700)
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=False)


def _copy_locked_snapshot(
    *,
    source: Path,
    target: Path,
    entry: ModelLockEntry,
    identity: str,
) -> dict[str, object]:
    total_bytes = sum((source / item.path).stat().st_size for item in entry.files)
    target.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target.parent)
    reserve_bytes = 10 * 1024**3
    if usage.free < total_bytes + reserve_bytes:
        raise ManagedRuntimeError(
            "local scratch cannot hold the verified model snapshot and reserve"
        )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, mode=0o700)
    copied_bytes = 0
    started = time.monotonic()
    next_progress = started + 30.0
    file_states: list[dict[str, object]] = []
    try:
        for record in entry.files:
            relative = Path(record.path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ManagedRuntimeError("locked model file escapes local staging")
            source_file = source / relative
            destination = temporary / relative
            if not source_file.is_file():
                raise ManagedRuntimeError("verified source snapshot became incomplete")
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            written = 0
            with (
                source_file.open("rb") as source_stream,
                destination.open("wb") as destination_stream,
            ):
                while True:
                    chunk = source_stream.read(_COPY_BUFFER_BYTES)
                    if not chunk:
                        break
                    destination_stream.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    copied_bytes += len(chunk)
                    now = time.monotonic()
                    if now >= next_progress:
                        print(
                            "MODEL_SNAPSHOT_STAGE "
                            f"state=copying bytes={copied_bytes}/{total_bytes} "
                            f"elapsed={now - started:.1f}s",
                            flush=True,
                        )
                        next_progress = now + 30.0
            if record.size_bytes and written != record.size_bytes:
                raise ManagedRuntimeError("locally staged model file size differs")
            observed_sha256 = digest.hexdigest()
            if record.sha256 is not None and observed_sha256 != record.sha256:
                raise ManagedRuntimeError("locally staged model file hash differs")
            destination.chmod(0o444)
            file_states.append(
                {
                    "path": record.path,
                    "sha256": observed_sha256,
                    "size_bytes": written,
                }
            )
        manifest = {
            "commit_sha": entry.commit_sha,
            "file_count": len(entry.files),
            "files": file_states,
            "identity": identity,
            "protocol": _LOCAL_MODEL_STAGE_PROTOCOL,
            "snapshot_content_hash": entry.snapshot_content_hash,
            "total_bytes": total_bytes,
        }
        _write_json(temporary / "pmmc-local-snapshot.json", manifest)
        (temporary / "pmmc-local-snapshot.json").chmod(0o444)
        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        if target.exists():
            _remove_local_snapshot(target)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "cache_hit": False,
        "copied_bytes": copied_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "file_count": len(entry.files),
        "identity": identity,
        "total_bytes": total_bytes,
    }


def _stage_verified_snapshots_to_local(
    *,
    settings: Settings,
    lock: ModelLockManifest,
    aliases: list[str],
    snapshot_paths: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, object] | None]:
    if not _enabled_environment_flag("PMMC_STAGE_MODEL_SNAPSHOTS_TO_LOCAL"):
        return snapshot_paths, None
    if settings.local_scratch_dir is None:
        raise ManagedRuntimeError("local model staging requires PMMC_LOCAL_SCRATCH_DIR")
    scratch = settings.resolve_path(settings.local_scratch_dir) / "model-snapshots"
    scratch.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, tuple[ModelLockEntry, Path, set[str]]] = {}
    for alias in sorted(aliases):
        entry = lock.get(alias)
        identity = _snapshot_stage_identity(entry)
        source = snapshot_paths[alias].resolve(strict=True)
        existing = grouped.get(identity)
        if existing is None:
            grouped[identity] = (entry, source, {alias})
        else:
            existing[2].add(alias)

    staged = dict(snapshot_paths)
    records: list[dict[str, object]] = []
    print(
        "MODEL_SNAPSHOT_STAGE "
        f"state=starting unique_snapshots={len(grouped)} aliases={len(aliases)}",
        flush=True,
    )
    for identity, (entry, source, grouped_aliases) in sorted(grouped.items()):
        target = scratch / identity / "snapshot"
        if _valid_local_snapshot(target=target, entry=entry, identity=identity):
            record: dict[str, object] = {
                "cache_hit": True,
                "copied_bytes": 0,
                "elapsed_seconds": 0.0,
                "file_count": len(entry.files),
                "identity": identity,
                "total_bytes": sum((source / item.path).stat().st_size for item in entry.files),
            }
        else:
            if target.exists():
                _remove_local_snapshot(target)
            try:
                record = _copy_locked_snapshot(
                    source=source,
                    target=target,
                    entry=entry,
                    identity=identity,
                )
            except Exception as exc:
                print(
                    "MODEL_SNAPSHOT_STAGE "
                    f"state=failed alias_count={len(grouped_aliases)} "
                    f"file_count={len(entry.files)} error_type={type(exc).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
        for alias in grouped_aliases:
            staged[alias] = target
        records.append({**record, "alias_count": len(grouped_aliases)})
        print(
            "MODEL_SNAPSHOT_STAGE "
            f"state=ready cache_hit={str(bool(record['cache_hit'])).lower()} "
            f"alias_count={len(grouped_aliases)} file_count={record['file_count']} "
            f"copied_bytes={record['copied_bytes']} "
            f"elapsed={record['elapsed_seconds']}s",
            flush=True,
        )
    summary: dict[str, object] = {
        "alias_count": len(aliases),
        "protocol": _LOCAL_MODEL_STAGE_PROTOCOL,
        "records": records,
        "unique_snapshot_count": len(grouped),
    }
    return staged, summary


def _defer_verified_snapshots_to_logical_stage(
    *,
    lock: ModelLockManifest,
    aliases: list[str],
) -> dict[str, object] | None:
    """Record lazy local staging for identity-only calibration preparation."""

    if not _enabled_environment_flag("PMMC_STAGE_MODEL_SNAPSHOTS_TO_LOCAL"):
        return None
    unique_snapshot_count = len({_snapshot_stage_identity(lock.get(alias)) for alias in aliases})
    summary: dict[str, object] = {
        "alias_count": len(aliases),
        "protocol": _LOCAL_MODEL_STAGE_PROTOCOL,
        "records": [],
        "state": "deferred",
        "unique_snapshot_count": unique_snapshot_count,
    }
    print(
        "MODEL_SNAPSHOT_STAGE "
        "state=deferred reason=calibration_identity_check "
        f"unique_snapshots={unique_snapshot_count} aliases={len(aliases)}",
        flush=True,
    )
    return summary


def _database_enabled(settings: Settings) -> bool:
    return bool(settings.database_url and settings.database_url.get_secret_value().strip())


def _persist_session(settings: Settings, value: ManagedRuntimeSession) -> None:
    if not _database_enabled(settings):
        return

    async def write() -> None:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.runtime_repositories import (
            ManagedRuntimeRepository,
        )

        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session, session.begin():
                await ManagedRuntimeRepository(session).put_session(value)
        finally:
            await engine.dispose()

    asyncio.run(write())


def _persist_service(
    prepared: PreparedRuntime,
    value: ManagedServiceState,
) -> None:
    if not _database_enabled(prepared.settings):
        return

    async def write() -> None:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.runtime_repositories import (
            ManagedRuntimeRepository,
        )

        engine, sessions = create_session_factory(prepared.settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session, session.begin():
                repository = ManagedRuntimeRepository(session)
                await repository.put_service(value)
                await repository.put_allocations(
                    runtime_id=value.runtime_id,
                    runtime_service_id=value.runtime_service_id,
                    stage_id=value.stage_id,
                    allocation=value.allocation,
                    devices=prepared.plan.devices,
                )
        finally:
            await engine.dispose()

    asyncio.run(write())


def _persist_calibration(settings: Settings, value: RuntimeCalibration) -> None:
    if not _database_enabled(settings):
        return

    async def write() -> None:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.runtime_repositories import (
            ManagedRuntimeRepository,
        )

        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session, session.begin():
                await ManagedRuntimeRepository(session).put_calibration(value)
        finally:
            await engine.dispose()

    asyncio.run(write())


def _database_calibration(
    settings: Settings,
    calibration_identifier: str,
) -> RuntimeCalibration | None:
    if not _database_enabled(settings):
        return None

    async def read() -> RuntimeCalibration | None:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.runtime_repositories import (
            ManagedRuntimeRepository,
        )

        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                return await ManagedRuntimeRepository(session).get_calibration(
                    calibration_identifier
                )
        finally:
            await engine.dispose()

    return asyncio.run(read())


def _link_runtime_runs(settings: Settings, runtime_id: str) -> list[str]:
    if not _database_enabled(settings):
        return []

    async def write() -> list[str]:
        from pmmc.infrastructure.database.constants import DatabaseRole
        from pmmc.infrastructure.database.engine import create_session_factory
        from pmmc.infrastructure.database.runtime_repositories import (
            ManagedRuntimeRepository,
        )

        engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session, session.begin():
                return await ManagedRuntimeRepository(session).link_runs_from_model_calls(
                    runtime_id
                )
        finally:
            await engine.dispose()

    return asyncio.run(write())


def _resolve_profile_lock(settings: Settings, profile: RuntimeProfile) -> Path:
    return settings.resolved_runtime_model_lock_path or settings.resolve_path(
        profile.model_lock_path
    )


def _profile_aliases(profile: RuntimeProfile, logical_stage: str | None) -> list[str]:
    services = (
        profile.services if logical_stage is None else profile.services_for_stage(logical_stage)
    )
    return sorted({service.model_alias for service in services})


def _validate_runtime_ports(settings: Settings, profile: RuntimeProfile) -> None:
    """Keep managed GPU services inside the operator-approved loopback range."""

    if settings.runtime_port_start > settings.runtime_port_end:
        raise ManagedRuntimeError("runtime port range start cannot exceed its end")
    invalid = sorted(
        {
            service.port
            for service in profile.services
            if service.port is not None
            and not settings.runtime_port_start <= service.port <= settings.runtime_port_end
        }
    )
    if invalid:
        raise ManagedRuntimeError(
            "managed service ports fall outside PMMC_RUNTIME_PORT_START/END: "
            + ", ".join(str(port) for port in invalid)
        )


def _runtime_plan_safe_summary(
    plan: ManagedRuntimePlan,
    *,
    logical_stage: str | None,
    replica_count: int,
) -> dict[str, object]:
    stage_service_reference_count = sum(len(stage.service_ids) for stage in plan.stages)
    return {
        "logical_stage": logical_stage or "all",
        "replica_count": replica_count,
        "service_count": len(plan.services),
        "shared_stage_reference_count": max(
            0,
            stage_service_reference_count - len(plan.services),
        ),
        "stage_count": len(plan.stages),
        "stage_service_reference_count": stage_service_reference_count,
        "state": "ready",
    }


def prepare_runtime(
    settings: Settings,
    *,
    profile_path: Path | None = None,
    logical_stage: str | None = None,
    hardware_provider: NvmlHardwareProvider | None = None,
    replica_count: int = 1,
    verification_policy: ModelVerificationPolicy | None = None,
    paper_result: bool = False,
    verified_cache_manifest_path: Path | None = None,
    stage_model_snapshots_to_local: bool = True,
) -> PreparedRuntime:
    """Resolve immutable models, validate cache, inventory GPUs, and create a plan."""

    resolved_profile_path = (
        settings.resolve_path(profile_path)
        if profile_path is not None
        else settings.resolved_runtime_profile_path
    )
    profile = load_runtime_profile(resolved_profile_path)
    if profile.metadata.get("release_visible_gpus") == 0:
        raise ManagedRuntimeError("unmeasured CPU planning profile cannot start a GPU runtime")
    if replica_count <= 0:
        raise ManagedRuntimeError("replica_count must be positive")
    _validate_runtime_ports(settings, profile)
    registry = load_model_registry(settings.resolved_model_registry_path)
    lock = load_model_lock(_resolve_profile_lock(settings, profile))
    aliases = _profile_aliases(profile, logical_stage)
    try:
        assert_paper_lock(lock, registry, aliases=aliases)
    except (KeyError, ValueError) as exc:
        raise ManagedRuntimeError(
            "runtime profile lock does not match selected models; ensure "
            "PMMC_MODEL_REGISTRY_PATH and PMMC_RUNTIME_MODEL_LOCK_PATH were generated "
            "from the same registry"
        ) from exc
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    if verification_policy is None:
        policy_raw = os.environ.get(
            "PMMC_MODEL_VERIFICATION_POLICY",
            ModelVerificationPolicy.CACHED_RECEIPT.value,
        )
        try:
            verification_policy = ModelVerificationPolicy(policy_raw)
        except ValueError as exc:
            raise ManagedRuntimeError("unsupported model verification policy") from exc
    if verification_policy is ModelVerificationPolicy.IDENTITY_ONLY and (
        paper_result
        or os.environ.get("PMMC_PAPER_RESULT", "").strip().lower() in {"1", "true", "yes"}
    ):
        raise ManagedRuntimeError("paper results cannot use identity-only verification")
    if verified_cache_manifest_path is not None:
        cache_preparation = load_verified_model_cache_manifest(
            verified_cache_manifest_path,
            lock_path=_resolve_profile_lock(settings, profile),
            lock=lock,
            aliases=aliases,
            verification_policy=verification_policy,
        )
    else:
        cache_preparation = prepare_model_cache_audited(
            lock,
            settings.model_cache_path,
            token=token,
            aliases=aliases,
            local_files_only=settings.mode is AppMode.PAPER,
            verification_policy=verification_policy,
            allow_full_fallback=verification_policy is ModelVerificationPolicy.FULL,
        )
    if stage_model_snapshots_to_local:
        snapshot_paths, local_staging = _stage_verified_snapshots_to_local(
            settings=settings,
            lock=lock,
            aliases=aliases,
            snapshot_paths=cache_preparation.paths,
        )
    else:
        snapshot_paths = dict(cache_preparation.paths)
        local_staging = _defer_verified_snapshots_to_logical_stage(
            lock=lock,
            aliases=aliases,
        )
    resolved_registry = resolve_registry_from_lock(registry, lock)
    provider = hardware_provider or NvmlHardwareProvider()
    inventory = provider.inventory()
    selected_services = (
        profile.services if logical_stage is None else profile.services_for_stage(logical_stage)
    )
    replica_counts = (
        {service.role: replica_count for service in selected_services}
        if replica_count > 1
        else None
    )
    plan = build_runtime_plan(
        profile,
        registry,
        lock,
        snapshot_paths,
        inventory,
        logical_stage=logical_stage,
        replica_counts=replica_counts,
    )
    print(
        "RUNTIME_PLAN_CATALOG "
        + json.dumps(
            _runtime_plan_safe_summary(
                plan,
                logical_stage=logical_stage,
                replica_count=replica_count,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    run_dir = settings.runs_path / "managed-runtime" / plan.runtime_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_registry_path = run_dir / "resolved-model-registry.yaml"
    write_resolved_registry(resolved_registry, resolved_registry_path)
    _write_json(run_dir / "runtime-plan.json", plan)
    _write_json(
        run_dir / "model-verification.json",
        {
            "policy": verification_policy.value,
            "results": {
                alias: result.to_dict()
                for alias, result in sorted(cache_preparation.verifications.items())
            },
        },
    )
    if local_staging is not None:
        _write_json(run_dir / "model-local-staging.json", local_staging)
    _write_json(
        run_dir / "software-versions.json",
        {
            "vllm": _version("vllm"),
            "litellm": _version("litellm"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "python": sys.version.split()[0],
        },
    )
    return PreparedRuntime(
        settings=settings,
        profile=profile,
        source_registry=registry,
        resolved_registry=resolved_registry,
        lock=lock,
        snapshot_paths=snapshot_paths,
        inventory=inventory,
        plan=plan,
        run_dir=run_dir,
        resolved_registry_path=resolved_registry_path,
        model_verifications=cache_preparation.verifications,
        local_staging=local_staging,
    )


def plan_runtime(
    settings: Settings,
    *,
    profile_path: Path | None = None,
    logical_stage: str | None = None,
    replica_count: int = 1,
) -> PreparedRuntime:
    """Create a plan under the same Pod lock used by managed execution."""

    runtime_lock = RuntimeProcessLock(
        settings.runs_path / "managed-runtime" / ".runtime.lock",
        f"planning-{os.getpid()}",
    )
    with runtime_lock:
        prepared = prepare_runtime(
            settings,
            profile_path=profile_path,
            logical_stage=logical_stage,
            replica_count=replica_count,
        )
        runtime_lock.set_runtime_id(prepared.plan.runtime_id)
        return prepared


def calibration_id(
    prepared: PreparedRuntime,
    spec: ManagedServiceSpec,
    stage: RuntimeStage,
) -> str:
    allocation = next(item for item in stage.allocations if item.service_id == spec.service_id)
    template_service_id = str(spec.metadata.get("template_service_id", spec.service_id))
    if template_service_id != spec.service_id:
        reference_device = prepared.inventory.devices[0]
        device_ids = [reference_device.device_id]
        visible_indices = [reference_device.visible_index]
    else:
        device_ids = allocation.device_ids
        visible_indices = allocation.visible_indices
    payload = {
        "service_id": template_service_id,
        "model_commit": prepared.lock.get(spec.model_alias).commit_sha,
        "profile_hash": prepared.profile.resolved_hash,
        "hardware_fingerprint": prepared.inventory.fingerprint,
        "vllm_version": _service_backend_version(spec),
        "allocation": {
            "tensor_parallel_size": allocation.tensor_parallel_size,
            "device_ids": device_ids,
            "visible_indices": visible_indices,
        },
    }
    if prepared.settings.gpu_memory_accounting == "allocated_device":
        payload["gpu_memory_accounting"] = "allocated_device"
    if "vllm_memory_budget" in allocation.metadata:
        payload["vllm_memory_budget"] = allocation_vllm_memory_budget(allocation).model_dump(
            exclude={"visible_indices"},
        )
    return "cal-" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:32]


def calibration_path(settings: Settings, calibration_identifier: str) -> Path:
    return settings.runtime_calibrations_path / f"{calibration_identifier}.json"


def _service_backend_version(spec: ManagedServiceSpec) -> str:
    if spec.backend is ManagedBackendKind.TRANSFORMERS_CONTRASTIVE:
        return _version("transformers")
    return _version("vllm")


def _assert_calibration_identity(
    prepared: PreparedRuntime,
    spec: ManagedServiceSpec,
    stage: RuntimeStage,
    calibration: RuntimeCalibration,
) -> None:
    identifier = calibration_id(prepared, spec, stage)
    allocation = next(item for item in stage.allocations if item.service_id == spec.service_id)
    template_service_id = str(spec.metadata.get("template_service_id", spec.service_id))
    if template_service_id != spec.service_id:
        reference_device = prepared.inventory.devices[0]
        device_ids = [reference_device.device_id]
        visible_indices = [reference_device.visible_index]
    else:
        device_ids = allocation.device_ids
        visible_indices = allocation.visible_indices
    expected = {
        "calibration_id": identifier,
        "service_id": template_service_id,
        "model_alias": spec.model_alias,
        "model_commit": prepared.lock.get(spec.model_alias).commit_sha,
        "profile_hash": prepared.profile.resolved_hash,
        "hardware_fingerprint": prepared.inventory.fingerprint,
        "vllm_version": _service_backend_version(spec),
        "allocation": {
            "tensor_parallel_size": allocation.tensor_parallel_size,
            "device_ids": device_ids,
            "visible_indices": visible_indices,
        },
    }
    actual = {
        "calibration_id": calibration.calibration_id,
        "service_id": calibration.service_id,
        "model_alias": calibration.model_alias,
        "model_commit": calibration.model_commit,
        "profile_hash": calibration.profile_hash,
        "hardware_fingerprint": calibration.hardware_fingerprint,
        "vllm_version": calibration.vllm_version,
        "allocation": {
            "tensor_parallel_size": calibration.allocation.tensor_parallel_size,
            "device_ids": calibration.allocation.device_ids,
            "visible_indices": calibration.allocation.visible_indices,
        },
    }
    if "vllm_memory_budget" in allocation.metadata:
        expected["vllm_memory_budget"] = allocation_vllm_memory_budget(allocation).model_dump(
            exclude={"visible_indices"},
        )
        try:
            actual["vllm_memory_budget"] = allocation_vllm_memory_budget(
                calibration.allocation,
            ).model_dump(exclude={"visible_indices"})
        except ValueError:
            raise ManagedRuntimeError("runtime calibration memory policy does not match") from None
    if (
        calibration.status is not CalibrationStatus.READY
        or actual != expected
        or calibration.observed_idle_memory_bytes <= 0
        or calibration.observed_peak_memory_bytes <= 0
    ):
        raise ManagedRuntimeError(f"runtime calibration identity does not match: {identifier}")
    if prepared.settings.gpu_memory_accounting == "allocated_device":
        if calibration.metadata.get("gpu_memory_accounting") != "allocated_device":
            raise ManagedRuntimeError("runtime calibration memory scope does not match")
        evidence = DeviceLifecycleEvidence.model_validate(
            calibration.metadata.get("device_lifecycle")
        )
        validate_device_lifecycle(
            evidence, [calibration.allocation], evidence.peak_bytes_by_device
        )
        if calibration.observed_peak_memory_bytes > sum(evidence.peak_bytes_by_device.values()):
            raise ManagedRuntimeError("runtime calibration peak exceeds device evidence")


def require_matching_calibrations(prepared: PreparedRuntime, stage: RuntimeStage) -> None:
    from pmmc.runtime.step8_capacity import require_step8_capacity_receipts

    try:
        capacity_gated = require_step8_capacity_receipts(
            prepared,
            stage,
            backend_version=_service_backend_version,
        )
    except (ValueError, OSError) as exc:
        raise ManagedRuntimeError(str(exc)) from exc
    if prepared.settings.mode is not AppMode.PAPER:
        return
    if not (
        prepared.settings.runtime_require_calibration_paper
        and prepared.profile.require_calibration_in_paper
    ):
        return
    for service_id in stage.service_ids:
        if service_id in capacity_gated:
            continue
        spec = next(item for item in prepared.plan.services if item.service_id == service_id)
        identifier = calibration_id(prepared, spec, stage)
        path = calibration_path(prepared.settings, identifier)
        if not path.is_file():
            raise ManagedRuntimeError(
                f"paper runtime calibration is missing for service {service_id!r}: {identifier}"
            )
        calibration = RuntimeCalibration.from_json(path.read_text(encoding="utf-8"))
        _assert_calibration_identity(prepared, spec, stage, calibration)
        database_calibration = _database_calibration(prepared.settings, identifier)
        if _database_enabled(prepared.settings):
            if database_calibration is None:
                raise ManagedRuntimeError(
                    f"runtime calibration is missing from Postgres: {identifier}"
                )
            if database_calibration.to_dict() != calibration.to_dict():
                raise ManagedRuntimeError(
                    f"runtime calibration file and Postgres row differ: {identifier}"
                )


def _child_environment(prepared: PreparedRuntime) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HOME": str(prepared.settings.model_cache_path),
            "PMMC_MODEL_CACHE_DIR": str(prepared.settings.model_cache_path),
            "PMMC_GPU_MEMORY_ACCOUNTING": prepared.settings.gpu_memory_accounting,
            "PMMC_MODEL_REGISTRY_PATH": str(prepared.resolved_registry_path),
        }
    )
    if prepared.settings.mode is AppMode.PAPER:
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
    return environment


_SERVICE_SECRET_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LITELLM_API_KEY",
    "OPENAI_API_KEY",
    "PMMC_DATABASE_ADMIN_URL",
    "PMMC_DATABASE_EVALUATOR_URL",
    "PMMC_DATABASE_URL",
    "PMMC_MINIO_ROOT_PASSWORD",
    "PMMC_MINIO_ROOT_USER",
    "PMMC_OBJECT_STORE_ACCESS_KEY",
    "PMMC_OBJECT_STORE_ADMIN_ACCESS_KEY",
    "PMMC_OBJECT_STORE_ADMIN_SECRET_KEY",
    "PMMC_OBJECT_STORE_EVALUATOR_ACCESS_KEY",
    "PMMC_OBJECT_STORE_EVALUATOR_SECRET_KEY",
    "PMMC_OBJECT_STORE_SECRET_KEY",
    "VLLM_API_KEY",
}


def _service_environment(prepared: PreparedRuntime) -> dict[str, str]:
    """Give model servers CUDA/cache settings without unrelated application secrets."""

    environment = _child_environment(prepared)
    service_timeout = max(service.startup_timeout_seconds for service in prepared.plan.services)
    # Keep vLLM's own engine/RPC watchdogs aligned with PMMC's audited service
    # readiness deadline so an inner watchdog cannot end a valid cold start.
    environment.update(
        {
            "VLLM_ENGINE_READY_TIMEOUT_S": str(service_timeout),
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": str(service_timeout),
        }
    )
    thread_budget_raw = environment.get("PMMC_MODEL_CPU_THREADS_PER_REPLICA")
    if thread_budget_raw:
        try:
            thread_budget = int(thread_budget_raw)
        except ValueError as exc:
            raise ManagedRuntimeError(
                "PMMC_MODEL_CPU_THREADS_PER_REPLICA must be an integer"
            ) from exc
        if thread_budget <= 0:
            raise ManagedRuntimeError("PMMC_MODEL_CPU_THREADS_PER_REPLICA must be positive")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = str(thread_budget)
        environment["TOKENIZERS_PARALLELISM"] = "false"
    if prepared.settings.local_scratch_dir is not None:
        cache_root = (
            prepared.settings.resolve_path(prepared.settings.local_scratch_dir) / "vllm-cache"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        environment["VLLM_CACHE_ROOT"] = str(cache_root)
    for name in _SERVICE_SECRET_ENV_NAMES:
        environment.pop(name, None)
    secret_suffixes = (
        "_ACCESS_KEY",
        "_API_KEY",
        "_DATABASE_URL",
        "_PASSWORD",
        "_SECRET",
        "_SECRET_KEY",
        "_TOKEN",
    )
    for name in list(environment):
        if name.upper().endswith(secret_suffixes):
            environment.pop(name, None)
    return environment


def _settings_secret_values(settings: Settings) -> list[str]:
    values: set[str] = set()
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        if isinstance(value, SecretStr) and value.get_secret_value():
            values.add(value.get_secret_value())
    return sorted(values)


def _service_process_ids(handle: ManagedProcessHandle) -> set[int]:
    import psutil

    try:
        root = psutil.Process(handle.process.pid)
        return {root.pid, *(child.pid for child in root.children(recursive=True))}
    except (psutil.Error, ProcessLookupError):
        return {handle.process.pid}


def _wait_for_gpu_release(
    provider: NvmlHardwareProvider,
    process_ids: set[int],
    timeout_seconds: int,
) -> None:
    """Require every managed vLLM PID to disappear from NVML accounting."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        usage = provider.process_memory_bytes(process_ids)
        if not any(usage.values()):
            return
        time.sleep(0.25)
    raise ManagedProcessError("managed vLLM processes retained GPU memory after shutdown")


class _TelemetrySampler:
    def __init__(
        self,
        handles: list[ManagedProcessHandle],
        provider: NvmlHardwareProvider,
        output_path: Path,
        interval_seconds: float,
        device_accounting: AllocatedDeviceAccounting | None = None,
    ) -> None:
        self._handles_lock = threading.Lock()
        self._handles = {handle.service_id: handle for handle in handles}
        self.provider = provider
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.device_accounting = device_accounting
        self.stop_event = threading.Event()
        self.peak_by_service: dict[str, int | None] = {
            handle.service_id: None for handle in handles
        }
        self.failure_type: str | None = None
        self.failure_code: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def register_handle(self, handle: ManagedProcessHandle) -> None:
        """Add or replace a dynamically started service without racing the sampler."""

        with self._handles_lock:
            self._handles[handle.service_id] = handle
            self.peak_by_service.setdefault(handle.service_id, None)

    def _handle_snapshot(self) -> list[ManagedProcessHandle]:
        with self._handles_lock:
            return list(self._handles.values())

    def start(self) -> None:
        self.thread.start()

    def check_health(self) -> None:
        if self.failure_type is not None:
            raise ManagedRuntimeError(self.failure_code or "gpu_device_telemetry_failed")

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds * 2))
        if self.thread.is_alive() and self.failure_type is None:
            self.failure_type = "TelemetryThreadTimeout"

    def _run(self) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as stream:
                while not self.stop_event.is_set():
                    handles = self._handle_snapshot()
                    process_to_service: dict[int, str] = {}
                    if self.device_accounting is None:
                        for handle in handles:
                            for process_id in _service_process_ids(handle):
                                process_to_service[process_id] = handle.service_id
                    sample_error: str | None = None
                    device_samples = []
                    try:
                        if self.device_accounting is not None:
                            device_samples = self.device_accounting.sample()
                            usage = {}
                        else:
                            usage = self.provider.process_memory_bytes(set(process_to_service))
                    except BaseException as exc:
                        usage = {}
                        sample_error = type(exc).__name__
                        if isinstance(exc, DeviceMemoryBudgetExceeded):
                            device_samples = exc.samples
                        code = str(exc)
                        if code.startswith("gpu_device_") and code.replace("_", "").isalnum():
                            self.failure_code = code
                        self.failure_type = self.failure_type or sample_error
                        self.stop_event.set()
                    totals: dict[str, int | None] = {
                        handle.service_id: None if sample_error else 0 for handle in handles
                    }
                    for process_id, memory_bytes in usage.items():
                        service_id = process_to_service.get(process_id)
                        if service_id is not None:
                            totals[service_id] = (totals[service_id] or 0) + memory_bytes
                    if self.device_accounting is not None and device_samples:
                        by_device = {v.device_id: v.used_bytes for v in device_samples}
                        for allocation in self.device_accounting.allocations:
                            if allocation.service_id in totals:
                                totals[allocation.service_id] = sum(by_device[d]
                                                                   for d in allocation.device_ids)
                    with self._handles_lock:
                        for service_id, total in totals.items():
                            if total is not None:
                                self.peak_by_service[service_id] = max(
                                    self.peak_by_service.get(service_id) or 0, total
                                )
                    stream.write(
                        _stable_json(
                            {
                                "created_at": datetime.now(UTC).isoformat(),
                                "service_memory_bytes": totals,
                                "memory_scope": ("allocated_device_total_baseline_included"
                                                 if self.device_accounting else "process"),
                                **({"device_samples": [v.to_dict() for v in device_samples]}
                                   if self.device_accounting else {}),
                                **(
                                    {"measurement_error_type": sample_error} if sample_error else {}
                                ),
                                **({"measurement_error_code": self.failure_code}
                                   if self.failure_code else {}),
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                    self.stop_event.wait(self.interval_seconds)
        except BaseException as exc:
            self.failure_type = self.failure_type or type(exc).__name__
            self.stop_event.set()


class ActiveRuntimeStage:
    """Context manager for one concurrent, disjoint-GPU runtime stage."""

    def __init__(
        self,
        prepared: PreparedRuntime,
        stage: RuntimeStage,
        *,
        vllm_runner: VllmServiceRunner | None = None,
        contrastive_runner: ContrastiveServiceRunner | None = None,
        litellm_runner: LiteLLMServiceRunner | None = None,
        hardware_provider: NvmlHardwareProvider | None = None,
        require_calibration: bool = True,
    ) -> None:
        self.prepared = prepared
        self.stage = stage
        self.vllm_runner = vllm_runner or VllmServiceRunner()
        self.contrastive_runner = contrastive_runner or ContrastiveServiceRunner()
        self.litellm_runner = litellm_runner or LiteLLMServiceRunner()
        self.hardware_provider = hardware_provider or NvmlHardwareProvider()
        self.require_calibration = require_calibration
        self.api_key = random_runtime_key()
        self.services: list[tuple[ManagedServiceSpec, ManagedProcessHandle]] = []
        self.litellm: ManagedProcessHandle | None = None
        self.telemetry: _TelemetrySampler | None = None
        self.states: list[ManagedServiceState] = []
        self.device_accounting: AllocatedDeviceAccounting | None = None

    def _runner_for(self, spec: ManagedServiceSpec) -> ManagedServiceRunner:
        if spec.backend is ManagedBackendKind.TRANSFORMERS_CONTRASTIVE:
            return self.contrastive_runner
        return self.vllm_runner

    def _check_startup_health(self) -> None:
        if self.telemetry is not None:
            self.telemetry.check_health()
        if self.device_accounting is not None:
            self.device_accounting.sample()

    @property
    def environment(self) -> dict[str, str]:
        overlay = endpoint_overlay(
            self.services,
            litellm=self.litellm,
            api_key=self.api_key,
            runtime_id=self.prepared.plan.runtime_id,
            stage_id=self.stage.stage_id,
        )
        overlay["PMMC_MODEL_REGISTRY_PATH"] = str(self.prepared.resolved_registry_path)
        return {**_child_environment(self.prepared), **overlay}

    def __enter__(self) -> ActiveRuntimeStage:
        if self.require_calibration:
            require_matching_calibrations(self.prepared, self.stage)
        log_dir = self.prepared.run_dir / "services" / self.stage.stage_id.replace(":", "_")
        environment = _service_environment(self.prepared)
        started: list[tuple[ManagedServiceSpec, ManagedProcessHandle]] = []
        telemetry_handles: list[ManagedProcessHandle] = []
        service_started_at: dict[int, float] = {}
        try:
            requests: list[tuple[ManagedServiceSpec, GpuAllocation]] = []
            for service_id, allocation in zip(
                self.stage.service_ids, self.stage.allocations, strict=True
            ):
                spec = next(
                    item for item in self.prepared.plan.services if item.service_id == service_id
                )
                requests.append((spec, allocation))

            if self.prepared.settings.gpu_memory_accounting == "allocated_device":
                self.device_accounting = AllocatedDeviceAccounting(
                    self.hardware_provider, self.stage.allocations, self.prepared.inventory.devices,
                    on_budget_failure=lambda value: _write_json(
                        self.prepared.run_dir / f"device-budget-failure-{self.stage.ordinal}.json",
                        value,
                    ),
                )
                baseline = self.device_accounting.begin()
                _write_json(self.prepared.run_dir / f"device-baseline-{self.stage.ordinal}.json",
                            [v.to_dict() for v in baseline])
                print("RUNTIME_DEVICE_ACCOUNTING state=baseline_verified "
                      f"devices={len(baseline)} baseline_subtracted=false", flush=True)

            replicated_pool = len(requests) > 1 and all(
                int(spec.metadata.get("replica_count", 1)) > 1 for spec, _allocation in requests
            )
            lock_by_alias = {entry.alias: entry for entry in self.prepared.lock.entries}
            unique_model_entries = {
                (
                    lock_by_alias[spec.model_alias].repo_id,
                    lock_by_alias[spec.model_alias].commit_sha,
                ): lock_by_alias[spec.model_alias]
                for spec, _allocation in requests
            }
            local_scratch = self.prepared.settings.local_scratch_dir
            local_scratch_root = (
                self.prepared.settings.resolve_path(local_scratch).resolve()
                if local_scratch is not None
                else None
            )
            local_snapshot_count = sum(
                local_scratch_root is not None
                and self.prepared.snapshot_paths[spec.model_alias]
                .resolve()
                .is_relative_to(local_scratch_root)
                for spec, _allocation in requests
            )
            locked_model_bytes = sum(
                record.size_bytes
                for entry in unique_model_entries.values()
                for record in entry.files
            )
            follower_count = max(0, len(requests) - 1)
            follower_batch_raw = environment.get(
                "PMMC_MODEL_FOLLOWER_START_BATCH_SIZE",
                str(max(1, follower_count)),
            )
            try:
                follower_batch_size = int(follower_batch_raw)
            except ValueError as exc:
                raise ManagedRuntimeError(
                    "PMMC_MODEL_FOLLOWER_START_BATCH_SIZE must be an integer"
                ) from exc
            if follower_batch_size <= 0:
                raise ManagedRuntimeError("PMMC_MODEL_FOLLOWER_START_BATCH_SIZE must be positive")
            print(
                "RUNTIME_POOL_PREFLIGHT "
                f"stage={self.stage.stage_id} replicas={len(requests)} "
                f"unique_model_snapshots={len(unique_model_entries)} "
                f"locked_model_bytes={locked_model_bytes} "
                f"local_snapshot_count={local_snapshot_count} "
                f"cpu_threads_per_replica={environment.get('OMP_NUM_THREADS', 'runtime_default')} "
                f"follower_start_batch_size={min(follower_batch_size, max(1, follower_count))}",
                flush=True,
            )

            def start_service(index: int, *, attempt: int) -> None:
                self._check_startup_health()
                spec, allocation_value = requests[index]
                allocation = allocation_value
                if "vllm_memory_budget" in allocation.metadata:
                    budget = allocation_vllm_memory_budget(allocation)
                    print("RUNTIME_MEMORY_BUDGET " + _stable_json({
                        "service": spec.service_id, **budget.to_dict(),
                    }), flush=True)
                entry = self.prepared.resolved_registry.get(spec.model_alias)
                handle = self._runner_for(spec).start(
                    spec=spec,
                    allocation=allocation,
                    entry=entry,
                    snapshot_path=self.prepared.snapshot_paths[spec.model_alias],
                    api_key=self.api_key,
                    log_dir=log_dir,
                    environment=environment,
                )
                service_started_at[index] = time.monotonic()
                if index < len(started):
                    started[index] = (spec, handle)
                    telemetry_handles[index] = handle
                else:
                    started.append((spec, handle))
                    telemetry_handles.append(handle)
                if self.telemetry is not None:
                    self.telemetry.register_handle(handle)
                self.services = started
                replica_index = int(spec.metadata.get("replica_index", 0))
                state = ManagedServiceState(
                    runtime_service_id=(
                        f"{self.prepared.plan.runtime_id}:{self.stage.stage_id}:{spec.service_id}"
                    ),
                    runtime_id=self.prepared.plan.runtime_id,
                    service_id=spec.service_id,
                    stage_id=self.stage.stage_id,
                    role=spec.role,
                    backend=spec.backend,
                    model_alias=spec.model_alias,
                    status=ManagedServiceStatus.STARTING,
                    allocation=allocation,
                    process_id=handle.process.pid,
                    started_at=utc_now(),
                    metadata={
                        "replica_index": replica_index,
                        "replica_count": int(spec.metadata.get("replica_count", 1)),
                        "startup_attempts": attempt,
                        "supports_thinking_control": bool(
                            spec.metadata.get("supports_thinking_control", False)
                        ),
                        "gpu_memory_accounting": self.prepared.settings.gpu_memory_accounting,
                    },
                )
                if index < len(self.states):
                    self.states[index] = state
                else:
                    self.states.append(state)
                _persist_service(self.prepared, state)
                print(
                    "RUNTIME_SERVICE "
                    f"stage={self.stage.stage_id} service={spec.service_id} "
                    f"replica={replica_index} state=starting attempt={attempt}",
                    flush=True,
                )

            def wait_service(index: int) -> None:
                spec, handle = started[index]
                attempts = 2 if replicated_pool else 1
                attempt = 1
                started_waiting = service_started_at[index]

                def report_failure(error: BaseException, state: str, ordinal: int) -> None:
                    try:
                        diagnostic_codes = service_log_diagnostic_codes(handle.log_path)
                        print(
                            "RUNTIME_SERVICE "
                            f"stage={self.stage.stage_id} service={spec.service_id} "
                            f"replica={int(spec.metadata.get('replica_index', 0))} "
                            f"state={state} attempt={ordinal} "
                            f"error_type={type(error).__name__} "
                            f"diagnostic_codes={diagnostic_codes}",
                            flush=True,
                        )
                    except BaseException as secondary:
                        _retain_runtime_failure(error, secondary, "service startup diagnostic")

                while True:
                    try:
                        self._check_startup_health()
                        self._runner_for(spec).wait_ready(
                            handle,
                            expected_model_ids=[
                                self.prepared.resolved_registry.get(
                                    spec.model_alias
                                ).expected_actual_model_id
                            ],
                            timeout_seconds=spec.startup_timeout_seconds,
                            check_health=self._check_startup_health,
                        )
                        self._check_startup_health()
                        print(
                            "RUNTIME_SERVICE "
                            f"stage={self.stage.stage_id} service={spec.service_id} "
                            f"replica={int(spec.metadata.get('replica_index', 0))} "
                            f"state=ready attempt={attempt} "
                            f"elapsed={time.monotonic() - started_waiting:.1f}s",
                            flush=True,
                        )
                        return
                    except ManagedProcessError as exc:
                        self._check_startup_health()
                        if attempt >= attempts:
                            report_failure(exc, "failed", attempt)
                            raise
                        report_failure(exc, "retrying", attempt + 1)
                        try:
                            self._runner_for(spec).stop(
                                handle,
                                timeout_seconds=self.prepared.profile.shutdown_timeout_seconds,
                            )
                        except BaseException as secondary:
                            _retain_runtime_failure(exc, secondary, "startup retry stop")
                            raise exc from None
                        attempt += 1
                        start_service(index, attempt=attempt)
                        spec, handle = started[index]
                    except Exception as exc:
                        report_failure(exc, "failed", attempt)
                        raise

            # Eight cold replicas compiling into one cache can race on first
            # boot. Warm one leader, then fan out followers against the same
            # verified local cache. This changes startup scheduling only; all
            # service/model identities and request routing remain frozen.
            if replicated_pool:
                pool_started_at = time.monotonic()
                print(
                    "RUNTIME_POOL_STARTUP "
                    f"stage={self.stage.stage_id} state=leader_starting "
                    f"replicas={len(requests)} "
                    "strategy=leader_then_bounded_follower_waves",
                    flush=True,
                )
                start_service(0, attempt=1)
                self.telemetry = _TelemetrySampler(
                    telemetry_handles,
                    self.hardware_provider,
                    self.prepared.run_dir / "gpu-telemetry.jsonl",
                    self.prepared.profile.telemetry_interval_seconds,
                    device_accounting=self.device_accounting,
                )
                self.telemetry.start()
                wait_service(0)
                leader_ready_at = time.monotonic()
                # Bound each follower wave so local CPU/cache initialization
                # cannot turn a seven-process cold start into an I/O storm.
                wave_number = 0
                ready_followers = 0
                for wave_start in range(1, len(requests), follower_batch_size):
                    wave_number += 1
                    wave_indices = list(
                        range(
                            wave_start,
                            min(len(requests), wave_start + follower_batch_size),
                        )
                    )
                    wave_started_at = time.monotonic()
                    for index in wave_indices:
                        start_service(index, attempt=1)
                    print(
                        "RUNTIME_POOL_STARTUP "
                        f"stage={self.stage.stage_id} state=followers_started "
                        f"wave={wave_number} wave_size={len(wave_indices)} "
                        f"ready_followers={ready_followers}/{follower_count} "
                        f"leader_elapsed={leader_ready_at - pool_started_at:.1f}s",
                        flush=True,
                    )
                    for index in wave_indices:
                        wait_service(index)
                    ready_followers += len(wave_indices)
                    print(
                        "RUNTIME_POOL_STARTUP "
                        f"stage={self.stage.stage_id} state=follower_wave_ready "
                        f"wave={wave_number} wave_size={len(wave_indices)} "
                        f"ready_followers={ready_followers}/{follower_count} "
                        f"wave_elapsed={time.monotonic() - wave_started_at:.1f}s",
                        flush=True,
                    )
                print(
                    "RUNTIME_POOL_STARTUP "
                    f"stage={self.stage.stage_id} state=ready "
                    f"replicas={len(requests)} "
                    f"leader_elapsed={leader_ready_at - pool_started_at:.1f}s "
                    f"follower_wave_elapsed={time.monotonic() - leader_ready_at:.1f}s "
                    f"total_elapsed={time.monotonic() - pool_started_at:.1f}s",
                    flush=True,
                )
            else:
                for index in range(len(requests)):
                    start_service(index, attempt=1)
                self.telemetry = _TelemetrySampler(
                    telemetry_handles,
                    self.hardware_provider,
                    self.prepared.run_dir / "gpu-telemetry.jsonl",
                    self.prepared.profile.telemetry_interval_seconds,
                    device_accounting=self.device_accounting,
                )
                self.telemetry.start()
                for index in range(len(requests)):
                    wait_service(index)
            chat_services = [
                item for item in self.services if item[0].backend is ManagedBackendKind.VLLM_CHAT
            ]
            replicated_chat = any(
                int(spec.metadata.get("replica_count", 1)) > 1 for spec, _handle in chat_services
            )
            if chat_services and self.prepared.profile.enable_litellm and not replicated_chat:
                self.litellm = self.litellm_runner.start(
                    registry=self.prepared.resolved_registry,
                    chat_services=chat_services,
                    port=self.prepared.profile.litellm_port,
                    api_key=self.api_key,
                    log_dir=log_dir,
                    environment=environment,
                )
                self.litellm_runner.wait_ready(
                    self.litellm,
                    expected_model_ids=[
                        self.prepared.resolved_registry.get(spec.model_alias).provider_model
                        for spec, _handle in chat_services
                    ],
                    timeout_seconds=max(spec.startup_timeout_seconds for spec, _ in chat_services),
                    check_health=self._check_startup_health,
                )
            self._check_startup_health()
            ready_at = utc_now()
            self.states = [
                state.model_copy(
                    update={"status": ManagedServiceStatus.READY, "ready_at": ready_at}
                )
                for state in self.states
            ]
            for state in self.states:
                _persist_service(self.prepared, state)
            _write_json(
                self.prepared.run_dir / f"service-states-{self.stage.ordinal}.json",
                [state.to_dict() for state in self.states],
            )
            return self
        except BaseException as exc:
            try:
                service_states = [
                    {
                        "diagnostic_codes": service_log_diagnostic_codes(handle.log_path),
                        "exit_code": handle.process.poll(),
                        "replica": int(spec.metadata.get("replica_index", 0)),
                        "service": spec.service_id,
                    }
                    for spec, handle in started
                ]
                print(
                    "RUNTIME_POOL_FAILURE "
                    + _stable_json(
                        {
                            "error_type": type(exc).__name__,
                            "code": (str(exc) if str(exc).startswith("gpu_device_")
                                     and str(exc).replace("_", "").isalnum()
                                     else type(exc).__name__),
                            "services": service_states,
                            "stage": self.stage.stage_id,
                            "started_service_count": len(started),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            except BaseException as secondary:
                _retain_runtime_failure(exc, secondary, "startup diagnostic")
            self.services = started
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        primary = exc

        def failed(error: BaseException, phase: str) -> None:
            nonlocal primary
            primary = _retain_runtime_failure(primary, error, phase)

        def persist_state(value: ManagedServiceState) -> None:
            try:
                _persist_service(self.prepared, value)
            except BaseException as caught:
                failed(caught, "service persistence")

        process_ids: set[int] = set()
        for _spec, handle in self.services:
            process_ids.add(handle.process.pid)
            try:
                process_ids.update(_service_process_ids(handle))
            except BaseException as caught:
                failed(caught, "service process discovery")
        if exc_type is None and self.states:
            self.states = [
                state.model_copy(update={"status": ManagedServiceStatus.STOPPING})
                for state in self.states
            ]
            for state in self.states:
                persist_state(state)
        if self.litellm is not None:
            try:
                self.litellm_runner.stop(
                    self.litellm,
                    timeout_seconds=self.prepared.profile.shutdown_timeout_seconds,
                )
            except BaseException as caught:
                failed(caught, "proxy stop")
        exit_codes: dict[str, int] = {}
        for spec, handle in reversed(self.services):
            try:
                exit_codes[spec.service_id] = self._runner_for(spec).stop(
                    handle,
                    timeout_seconds=self.prepared.profile.shutdown_timeout_seconds,
                )
                if handle.process.poll() is None:
                    raise ManagedProcessError("managed service remains alive after stop")
            except BaseException as caught:
                failed(caught, "replica stop")
        if self.telemetry is not None:
            try:
                self.telemetry.stop()
            except BaseException as caught:
                failed(caught, "telemetry stop")
            if self.telemetry.failure_type is not None:
                failed(
                    ManagedProcessError(
                        self.telemetry.failure_code or
                        f"runtime telemetry sampler failed ({self.telemetry.failure_type})"
                    ),
                    "telemetry measurement",
                )
        try:
            if self.device_accounting is not None:
                if self.device_accounting.before:
                    evidence = self.device_accounting.finish(
                        self.prepared.profile.shutdown_timeout_seconds,
                    )
                    _write_json(self.prepared.run_dir / f"device-release-{self.stage.ordinal}.json",
                                evidence)
                    print("RUNTIME_DEVICE_ACCOUNTING state=release_verified "
                          f"devices={len(evidence.after)} consecutive_samples=2", flush=True)
            else:
                _wait_for_gpu_release(
                    self.hardware_provider,
                    process_ids,
                    self.prepared.profile.shutdown_timeout_seconds,
                )
        except BaseException as caught:
            failed(caught, "GPU release probe")
        stopped_at = utc_now()
        peaks = self.telemetry.peak_by_service if self.telemetry else {}
        self.states = [
            state.model_copy(
                update={
                    "status": (
                        ManagedServiceStatus.FAILED
                        if primary is not None
                        else ManagedServiceStatus.STOPPED
                    ),
                    "stopped_at": stopped_at,
                    "exit_code": exit_codes.get(state.service_id),
                    "error_type": (type(primary).__name__ if primary is not None else None),
                    "observed_peak_memory_bytes": peaks.get(state.service_id),
                }
            )
            for state in self.states
        ]
        for state in self.states:
            persist_state(state)
        try:
            _write_json(
                self.prepared.run_dir / f"service-states-{self.stage.ordinal}.json",
                [state.to_dict() for state in self.states],
            )
        except BaseException as caught:
            failed(caught, "service receipt")
        log_handles = [handle for _spec, handle in self.services]
        if self.litellm is not None:
            log_handles.append(self.litellm)
        for handle in log_handles:
            try:
                redact_service_log(
                    handle.log_path,
                    secrets_to_remove=[
                        self.api_key,
                        *_settings_secret_values(self.prepared.settings),
                    ],
                    roots=[
                        self.prepared.settings.model_cache_path,
                        self.prepared.settings.project_root,
                        self.prepared.run_dir.parent,
                    ],
                )
            except BaseException as caught:
                failed(caught, "service log redaction")
        if exc is None and primary is not None:
            raise primary


def _runtime_session(
    prepared: PreparedRuntime,
    status: RuntimeStatus,
    *,
    started_at,
    error_type: str | None = None,
) -> ManagedRuntimeSession:
    terminal = status in {
        RuntimeStatus.COMPLETED,
        RuntimeStatus.FAILED,
        RuntimeStatus.CANCELLED,
    }
    return ManagedRuntimeSession(
        runtime_id=prepared.plan.runtime_id,
        profile_name=prepared.profile.profile_name,
        profile_hash=prepared.profile.resolved_hash,
        hardware_fingerprint=prepared.inventory.fingerprint,
        status=status,
        vllm_version=_version("vllm"),
        litellm_version=_version("litellm"),
        started_at=started_at,
        completed_at=utc_now() if terminal else None,
        error_type=error_type,
        metadata={"runtime_directory_id": prepared.run_dir.name,
                  "gpu_memory_accounting": prepared.settings.gpu_memory_accounting},
    )


def _write_managed_failure_diagnostic(
    prepared: PreparedRuntime,
    exc: BaseException,
) -> dict[str, object]:
    """Persist and return a secret-free managed-runtime failure summary."""

    if isinstance(exc, (ManagedProcessError, ManagedRuntimeError)):
        error_summary = str(exc)
    else:
        error_summary = (
            "opaque-"
            + hashlib.sha256(
                f"{type(exc).__module__}.{type(exc).__qualname__}".encode()
            ).hexdigest()[:16]
        )
    service_logs = []
    unclassified_service_log_count = 0
    services_root = prepared.run_dir / "services"
    for path in sorted(services_root.rglob("*.log")) if services_root.is_dir() else []:
        codes = service_log_diagnostic_codes(path)
        if codes == ["unclassified"]:
            unclassified_service_log_count += 1
            continue
        service_logs.append(
            {
                "log": path.relative_to(prepared.run_dir).as_posix(),
                "diagnostic_codes": codes,
            }
        )
    payload: dict[str, object] = {
        "runtime_directory_id": prepared.run_dir.name,
        "runtime_id": prepared.plan.runtime_id,
        "stage_ids": [stage.stage_id for stage in prepared.plan.stages],
        "error_type": type(exc).__name__,
        "error_summary": error_summary,
        "service_logs": service_logs,
        "unclassified_service_log_count": unclassified_service_log_count,
    }
    if isinstance(exc, ManagedChildCommandError):
        payload["child_command"] = {
            "command_count": exc.command_count,
            "command_ordinal": exc.command_ordinal,
            "locator": exc.locator,
            "returncode": exc.returncode,
        }
        child_diagnostics_path = prepared.run_dir / "child-command-diagnostics.jsonl"
        child_diagnostics: list[dict[str, object]] = []
        if child_diagnostics_path.is_file():
            for line in child_diagnostics_path.read_text(encoding="utf-8").splitlines()[-20:]:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    child_diagnostics.append(value)
        payload["child_diagnostics"] = child_diagnostics
    _write_json(prepared.run_dir / "failure-diagnostic.json", payload)
    return payload


def _record_cancelled_session(settings: Settings, prepared: PreparedRuntime) -> None:
    path = prepared.run_dir / "runtime-session.json"
    started_at = utc_now()
    if path.is_file():
        try:
            started_at = ManagedRuntimeSession.from_json(
                path.read_text(encoding="utf-8")
            ).started_at
        except ValueError:
            pass
    cancelled = _runtime_session(
        prepared,
        RuntimeStatus.CANCELLED,
        started_at=started_at,
    )
    _write_json(path, cancelled)
    _persist_session(settings, cancelled)


def _record_failed_session(
    settings: Settings,
    prepared: PreparedRuntime,
    error_type: str,
) -> None:
    path = prepared.run_dir / "runtime-session.json"
    started_at = utc_now()
    if path.is_file():
        try:
            existing = ManagedRuntimeSession.from_json(path.read_text(encoding="utf-8"))
            if existing.status is RuntimeStatus.FAILED:
                return
            started_at = existing.started_at
        except ValueError:
            pass
    failed = _runtime_session(
        prepared,
        RuntimeStatus.FAILED,
        started_at=started_at,
        error_type=error_type,
    )
    _write_json(path, failed)
    _persist_session(settings, failed)


def run_managed_command(
    settings: Settings,
    *,
    profile_path: Path | None,
    logical_stage: str,
    command: list[str],
    environment_overrides: dict[str, str] | None = None,
    replica_count: int = 1,
) -> tuple[int, Path]:
    """Run one command against a concurrently active logical stage."""

    return run_managed_commands(
        settings,
        profile_path=profile_path,
        logical_stage=logical_stage,
        commands=[command],
        environment_overrides=environment_overrides,
        replica_count=replica_count,
    )


@dataclass
class ManagedRuntimeCommandSession:
    """One physical model service shared by several ordered command batches."""

    settings: Settings
    profile_path: Path | None
    logical_stage: str
    replica_count: int
    prepared: PreparedRuntime
    active: ActiveRuntimeStage
    environment_overrides: dict[str, str]
    batch_count: int = 0

    @property
    def run_dir(self) -> Path:
        return self.prepared.run_dir

    @property
    def supported_logical_stages(self) -> frozenset[str]:
        service_ids = set(self.prepared.plan.stages[0].service_ids)
        return frozenset(
            stage
            for service in self.prepared.plan.services
            if service.service_id in service_ids
            for stage in service.stages
        )

    def run_commands(
        self,
        *,
        logical_stage: str,
        commands: list[list[str]],
        environment_overrides: dict[str, str] | None = None,
        command_completed: Callable[[int, list[str]], None] | None = None,
        replica_count: int,
    ) -> tuple[int, Path]:
        if not commands or any(not command for command in commands):
            raise ValueError("managed runtime commands cannot be empty")
        if logical_stage not in self.supported_logical_stages:
            raise ManagedRuntimeError(
                "active managed runtime cannot serve requested logical stage "
                f"{logical_stage!r}; supported={sorted(self.supported_logical_stages)!r}"
            )
        if replica_count != self.replica_count:
            raise ManagedRuntimeError(
                "active managed runtime replica count differs from nested command batch"
            )

        self.batch_count += 1
        child_environment = {
            **self.active.environment,
            **self.environment_overrides,
            **(environment_overrides or {}),
        }
        child_diagnostic_path = self.run_dir / "child-command-diagnostics.jsonl"
        print(
            "MANAGED_SHARED_SESSION "
            + _stable_json(
                {
                    "batch_ordinal": self.batch_count,
                    "command_count": len(commands),
                    "logical_stage": logical_stage,
                    "physical_stage": self.logical_stage,
                    "runtime_directory_id": self.run_dir.name,
                    "service_startup_count": 0,
                    "state": "reusing_active_service",
                }
            ),
            flush=True,
        )
        for index, command in enumerate(commands, start=1):
            locator = _managed_command_locator(command)
            command_summary = {
                "batch_ordinal": self.batch_count,
                "command_count": len(commands),
                "command_ordinal": index,
                "locator": locator,
                "logical_stage": logical_stage,
            }
            print(
                "MANAGED_COMMAND " + _stable_json({**command_summary, "state": "starting"}),
                flush=True,
            )
            command_environment = {
                **child_environment,
                "PMMC_MANAGED_CHILD_COMMAND_COUNT": str(len(commands)),
                "PMMC_MANAGED_CHILD_COMMAND_ORDINAL": str(index),
                "PMMC_MANAGED_CHILD_DIAGNOSTIC_PATH": str(child_diagnostic_path),
                "PMMC_MANAGED_SHARED_BATCH_ORDINAL": str(self.batch_count),
                "PMMC_MANAGED_SHARED_LOGICAL_STAGE": logical_stage,
            }
            result = subprocess.run(command, env=command_environment, check=False)
            if result.returncode != 0:
                print(
                    "MANAGED_COMMAND_FAILURE "
                    + _stable_json({**command_summary, "returncode": result.returncode}),
                    file=sys.stderr,
                    flush=True,
                )
                raise ManagedChildCommandError(
                    returncode=result.returncode,
                    command_ordinal=index,
                    command_count=len(commands),
                    locator=locator,
                )
            print(
                "MANAGED_COMMAND " + _stable_json({**command_summary, "state": "completed"}),
                flush=True,
            )
            if command_completed is not None:
                command_completed(index, command)
        return 0, self.run_dir


_ACTIVE_MANAGED_COMMAND_SESSION: ContextVar[ManagedRuntimeCommandSession | None] = ContextVar(
    "pmmc_active_managed_command_session", default=None
)


@contextmanager
def managed_runtime_command_session(
    settings: Settings,
    *,
    profile_path: Path | None,
    logical_stage: str,
    environment_overrides: dict[str, str] | None = None,
    replica_count: int = 1,
) -> Iterator[ManagedRuntimeCommandSession]:
    """Keep one managed model service alive across dependent command batches."""

    if _ACTIVE_MANAGED_COMMAND_SESSION.get() is not None:
        raise ManagedRuntimeError("managed runtime command sessions cannot be nested")
    runtime_lock = RuntimeProcessLock(
        Path(
            os.environ.get(
                "PMMC_RUNTIME_PROCESS_LOCK_PATH",
                str(settings.runs_path / "managed-runtime" / ".runtime.lock"),
            )
        ),
        f"preparing-{os.getpid()}",
    )
    runtime_lock.acquire()
    signal_guard = ManagedSignalGuard()
    prepared: PreparedRuntime | None = None
    started_at = None
    primary: BaseException | None = None
    try:
        signal_guard.install()
        prepared = prepare_runtime(
            settings,
            profile_path=profile_path,
            logical_stage=logical_stage,
            replica_count=replica_count,
            verification_policy=ModelVerificationPolicy(
                (environment_overrides or {}).get(
                    "PMMC_MODEL_VERIFICATION_POLICY",
                    ModelVerificationPolicy.CACHED_RECEIPT.value,
                )
            ),
            paper_result=(environment_overrides or {}).get("PMMC_PAPER_RESULT", "false").lower()
            in {"1", "true", "yes"},
            verified_cache_manifest_path=(
                Path((environment_overrides or {})["PMMC_VERIFIED_MODEL_CACHE_MANIFEST"])
                if (environment_overrides or {}).get("PMMC_VERIFIED_MODEL_CACHE_MANIFEST")
                else None
            ),
        )
        runtime_lock.set_runtime_id(prepared.plan.runtime_id)
        if len(prepared.plan.stages) != 1:
            raise ManagedRuntimeError(
                "the requested command needs all stage services concurrently, but the GPU "
                "plan split them into multiple parts; run a narrower model stage or request "
                "more GPUs"
            )
        started_at = utc_now()
        starting_session = _runtime_session(
            prepared,
            RuntimeStatus.STARTING,
            started_at=started_at,
        )
        _write_json(prepared.run_dir / "runtime-session.json", starting_session)
        _persist_session(settings, starting_session)
        with ActiveRuntimeStage(prepared, prepared.plan.stages[0]) as active:
            running_session = _runtime_session(
                prepared,
                RuntimeStatus.RUNNING,
                started_at=started_at,
            )
            _write_json(prepared.run_dir / "runtime-session.json", running_session)
            _persist_session(settings, running_session)
            session = ManagedRuntimeCommandSession(
                settings=settings,
                profile_path=profile_path,
                logical_stage=logical_stage,
                replica_count=replica_count,
                prepared=prepared,
                active=active,
                environment_overrides=dict(environment_overrides or {}),
            )
            token = _ACTIVE_MANAGED_COMMAND_SESSION.set(session)
            try:
                print(
                    "MANAGED_SHARED_SESSION "
                    + _stable_json(
                        {
                            "logical_stage": logical_stage,
                            "replica_count": replica_count,
                            "runtime_directory_id": prepared.run_dir.name,
                            "service_startup_count": len(active.services),
                            "state": "started",
                            "supported_logical_stages": sorted(session.supported_logical_stages),
                        }
                    ),
                    flush=True,
                )
                yield session
            finally:
                _ACTIVE_MANAGED_COMMAND_SESSION.reset(token)
        completed_session = _runtime_session(
            prepared,
            RuntimeStatus.COMPLETED,
            started_at=started_at,
        )
        _write_json(prepared.run_dir / "runtime-session.json", completed_session)
        _persist_session(settings, completed_session)
        _link_runtime_runs(settings, prepared.plan.runtime_id)
    except BaseException as exc:
        primary = exc
        if prepared is not None and started_at is not None:
            cancelled = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            try:
                failed_session = _runtime_session(
                    prepared,
                    RuntimeStatus.CANCELLED if cancelled else RuntimeStatus.FAILED,
                    started_at=started_at,
                    error_type=None if cancelled else type(exc).__name__,
                )
            except BaseException as secondary:
                _retain_runtime_failure(exc, secondary, "runtime failure state")
            else:
                try:
                    _write_json(prepared.run_dir / "runtime-session.json", failed_session)
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "runtime failure receipt")
                try:
                    _persist_session(settings, failed_session)
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "runtime failure persistence")
            if not cancelled:
                try:
                    diagnostic = _write_managed_failure_diagnostic(prepared, exc)
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "runtime failure diagnostic")
                    diagnostic = {
                        "runtime_directory_id": prepared.run_dir.name,
                        "error_type": type(exc).__name__,
                        "diagnostic_error_type": type(secondary).__name__,
                    }
                try:
                    print(
                        "MANAGED_RUNTIME_FAILURE " + _stable_json(diagnostic),
                        file=sys.stderr,
                        flush=True,
                    )
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "runtime failure output")
        raise
    finally:
        cleanup_error = primary
        try:
            signal_guard.restore()
        except BaseException as secondary:
            cleanup_error = _retain_runtime_failure(cleanup_error, secondary, "signal restore")
        try:
            runtime_lock.release()
        except BaseException as secondary:
            cleanup_error = _retain_runtime_failure(
                cleanup_error, secondary, "runtime lock release"
            )
        if primary is None and cleanup_error is not None:
            raise cleanup_error


def run_managed_commands(
    settings: Settings,
    *,
    profile_path: Path | None,
    logical_stage: str,
    commands: list[list[str]],
    environment_overrides: dict[str, str] | None = None,
    command_completed: Callable[[int, list[str]], None] | None = None,
    replica_count: int = 1,
) -> tuple[int, Path]:
    """Run ordered commands while one logical model stage remains active."""

    if not commands or any(not command for command in commands):
        raise ValueError("managed runtime commands cannot be empty")
    active_session = _ACTIVE_MANAGED_COMMAND_SESSION.get()
    if active_session is not None:
        return active_session.run_commands(
            logical_stage=logical_stage,
            commands=commands,
            environment_overrides=environment_overrides,
            command_completed=command_completed,
            replica_count=replica_count,
        )
    with managed_runtime_command_session(
        settings,
        profile_path=profile_path,
        logical_stage=logical_stage,
        environment_overrides=environment_overrides,
        replica_count=replica_count,
    ) as session:
        return session.run_commands(
            logical_stage=logical_stage,
            commands=commands,
            environment_overrides=environment_overrides,
            command_completed=command_completed,
            replica_count=replica_count,
        )


def calibrate_runtime(
    settings: Settings,
    *,
    profile_path: Path | None = None,
    verification_policy: ModelVerificationPolicy | None = None,
    verified_cache_manifest_path: Path | None = None,
    paper_result: bool = False,
    logical_stage: str | None = None,
    verify_only: bool = False,
) -> tuple[list[RuntimeCalibration], Path]:
    """Start each distinct service and persist hardware-bound observed memory."""

    runtime_lock = RuntimeProcessLock(
        settings.runs_path / "managed-runtime" / ".runtime.lock",
        f"preparing-{os.getpid()}",
    )
    with runtime_lock, ManagedSignalGuard():
        prepared = prepare_runtime(
            settings,
            profile_path=profile_path,
            logical_stage=logical_stage,
            verification_policy=verification_policy,
            verified_cache_manifest_path=verified_cache_manifest_path,
            paper_result=paper_result,
            stage_model_snapshots_to_local=False,
        )
        runtime_lock.set_runtime_id(prepared.plan.runtime_id)
        try:
            return _calibrate_prepared_runtime(settings, prepared, verify_only=verify_only)
        except BaseException as exc:
            try:
                if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                    _record_cancelled_session(settings, prepared)
                else:
                    _record_failed_session(settings, prepared, type(exc).__name__)
            except BaseException as secondary:
                _retain_runtime_failure(exc, secondary, "calibration failure persistence")
            raise


def _calibrate_prepared_runtime(
    settings: Settings,
    prepared: PreparedRuntime,
    *,
    verify_only: bool = False,
) -> tuple[list[RuntimeCalibration], Path]:
    started_at = utc_now()
    starting_session = _runtime_session(
        prepared,
        RuntimeStatus.STARTING,
        started_at=started_at,
    )
    _write_json(prepared.run_dir / "runtime-session.json", starting_session)
    _persist_session(settings, starting_session)
    running_session = _runtime_session(
        prepared,
        RuntimeStatus.RUNNING,
        started_at=started_at,
    )
    _write_json(prepared.run_dir / "runtime-session.json", running_session)
    _persist_session(settings, running_session)
    seen: set[str] = set()
    calibrations: list[RuntimeCalibration] = []
    for stage in prepared.plan.stages:
        for service_id in stage.service_ids:
            spec = next(item for item in prepared.plan.services if item.service_id == service_id)
            allocation = next(item for item in stage.allocations if item.service_id == service_id)
            identity_stage = stage.model_copy(
                update={
                    "stage_id": "calibration-identity",
                    "ordinal": 0,
                    "service_ids": [service_id],
                    "allocations": [allocation],
                }
            )
            identifier = calibration_id(prepared, spec, identity_stage)
            if identifier in seen:
                continue
            seen.add(identifier)
            calibration_started = time.monotonic()
            ready_path = calibration_path(settings, identifier)
            if ready_path.is_file():
                calibration = RuntimeCalibration.from_json(ready_path.read_text(encoding="utf-8"))
                _assert_calibration_identity(prepared, spec, identity_stage, calibration)
                _persist_calibration(settings, calibration)
                calibrations.append(calibration)
                _calibration_cache_log(prepared, spec, identifier, "file_hit", calibration_started)
                continue
            stored = _database_calibration(settings, identifier)
            if stored is not None:
                _assert_calibration_identity(prepared, spec, identity_stage, stored)
                _write_json(ready_path, stored)
                calibrations.append(stored)
                _calibration_cache_log(
                    prepared, spec, identifier, "database_hit", calibration_started
                )
                continue
            _calibration_cache_log(prepared, spec, identifier, "identity_miss", calibration_started)
            if verify_only:
                raise ManagedRuntimeError("release runtime calibration missing for physical allocation")
            single_stage = identity_stage.model_copy(update={"stage_id": f"calibrate:{identifier}"})
            try:
                with ActiveRuntimeStage(
                    prepared,
                    single_stage,
                    require_calibration=False,
                ) as active:
                    assert active.telemetry is not None
                    handle = next(
                        handle
                        for active_spec, handle in active.services
                        if active_spec.service_id == service_id
                    )
                    server_info = fetch_service_info(handle)
                    time.sleep(max(1.0, prepared.profile.telemetry_interval_seconds))
                    idle = active.telemetry.peak_by_service.get(service_id)
                peak = active.telemetry.peak_by_service.get(service_id, idle)
                if idle is None or peak is None or idle <= 0 or peak <= 0:
                    raise ManagedRuntimeError("runtime calibration has no positive GPU measurement")
                calibration = RuntimeCalibration(
                    calibration_id=identifier,
                    service_id=service_id,
                    model_alias=spec.model_alias,
                    model_commit=prepared.lock.get(spec.model_alias).commit_sha,
                    profile_hash=prepared.profile.resolved_hash,
                    hardware_fingerprint=prepared.inventory.fingerprint,
                    vllm_version=_service_backend_version(spec),
                    status=CalibrationStatus.READY,
                    allocation=allocation,
                    observed_peak_memory_bytes=peak,
                    observed_idle_memory_bytes=idle,
                    server_config={
                        "profile": spec.model_dump(mode="json"),
                        "observed_server": server_info,
                        "driver_version": prepared.inventory.driver_version,
                        "cuda_driver_version": prepared.inventory.cuda_driver_version,
                    },
                    metadata={
                        "gpu_memory_accounting": settings.gpu_memory_accounting,
                        **({"device_lifecycle": active.device_accounting.release_evidence.to_dict()}
                           if active.device_accounting is not None
                           and active.device_accounting.release_evidence is not None else {}),
                    },
                )
            except BaseException as exc:
                failed_identifier = (
                    f"{identifier}:failed:{prepared.plan.runtime_id.rsplit('-', 1)[-1]}"
                )
                calibration = RuntimeCalibration(
                    calibration_id=failed_identifier,
                    service_id=service_id,
                    model_alias=spec.model_alias,
                    model_commit=prepared.lock.get(spec.model_alias).commit_sha,
                    profile_hash=prepared.profile.resolved_hash,
                    hardware_fingerprint=prepared.inventory.fingerprint,
                    vllm_version=_service_backend_version(spec),
                    status=CalibrationStatus.FAILED,
                    allocation=allocation,
                    observed_peak_memory_bytes=0,
                    observed_idle_memory_bytes=0,
                    server_config={"profile": spec.model_dump(mode="json")},
                    error_type=type(exc).__name__,
                )
                try:
                    _write_json(calibration_path(settings, failed_identifier), calibration)
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "failed calibration receipt")
                try:
                    _persist_calibration(settings, calibration)
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "failed calibration persistence")
                failed_session = _runtime_session(
                    prepared,
                    RuntimeStatus.FAILED,
                    started_at=started_at,
                    error_type=type(exc).__name__,
                )
                try:
                    _write_json(prepared.run_dir / "runtime-session.json", failed_session)
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "calibration session receipt")
                try:
                    _persist_session(settings, failed_session)
                except BaseException as secondary:
                    _retain_runtime_failure(exc, secondary, "calibration session persistence")
                raise
            _write_json(calibration_path(settings, identifier), calibration)
            _persist_calibration(settings, calibration)
            calibrations.append(calibration)
            _calibration_cache_log(prepared, spec, identifier, "measured", calibration_started)
    completed_session = _runtime_session(
        prepared,
        RuntimeStatus.COMPLETED,
        started_at=started_at,
    )
    _write_json(prepared.run_dir / "runtime-session.json", completed_session)
    _persist_session(settings, completed_session)
    return calibrations, prepared.run_dir


def _calibration_cache_log(
    prepared: PreparedRuntime,
    spec: ManagedServiceSpec,
    identifier: str,
    state: str,
    started: float,
) -> None:
    print(
        "MODEL_CALIBRATION_CACHE "
        + json.dumps(
            {
                "state": state,
                "model_alias": spec.model_alias,
                "identity_hash": hashlib.sha256(identifier.encode()).hexdigest()[:16],
                "hardware_hash": hashlib.sha256(
                    prepared.inventory.fingerprint.encode()
                ).hexdigest()[:16],
                "profile_hash": prepared.profile.resolved_hash[:16],
                "startup_required": state in {"identity_miss", "measured"},
                "startup_performed": state == "measured",
                "scope": "hardware_bound_calibration_not_scientific_work",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def test_managed_gateway(
    settings: Settings,
    *,
    profile_path: Path | None = None,
) -> tuple[int, Path]:
    """Load and issue a real probe to every service, sequentially when necessary."""

    runtime_lock = RuntimeProcessLock(
        settings.runs_path / "managed-runtime" / ".runtime.lock",
        f"preparing-{os.getpid()}",
    )
    with runtime_lock, ManagedSignalGuard():
        prepared = prepare_runtime(
            settings,
            profile_path=profile_path,
            logical_stage="gateway_smoke",
        )
        runtime_lock.set_runtime_id(prepared.plan.runtime_id)
        try:
            return _test_prepared_gateway(settings, prepared)
        except KeyboardInterrupt:
            _record_cancelled_session(settings, prepared)
            raise


def _test_prepared_gateway(
    settings: Settings,
    prepared: PreparedRuntime,
) -> tuple[int, Path]:
    started_at = utc_now()
    try:
        starting_session = _runtime_session(
            prepared,
            RuntimeStatus.STARTING,
            started_at=started_at,
        )
        _write_json(prepared.run_dir / "runtime-session.json", starting_session)
        _persist_session(settings, starting_session)
        running_session = _runtime_session(
            prepared,
            RuntimeStatus.RUNNING,
            started_at=started_at,
        )
        _write_json(prepared.run_dir / "runtime-session.json", running_session)
        _persist_session(settings, running_session)
        for stage in prepared.plan.stages:
            with ActiveRuntimeStage(prepared, stage) as active:
                for spec, handle in active.services:
                    environment = active.environment
                    command = [
                        sys.executable,
                        "-m",
                        "pmmc.runtime.probe",
                        "--alias",
                        spec.model_alias,
                    ]
                    result = subprocess.run(command, env=environment, check=False)
                    if result.returncode != 0:
                        raise ManagedProcessError(
                            f"managed endpoint check failed: {describe_process(handle)}"
                        )
        completed_session = _runtime_session(
            prepared,
            RuntimeStatus.COMPLETED,
            started_at=started_at,
        )
        _write_json(prepared.run_dir / "runtime-session.json", completed_session)
        _persist_session(settings, completed_session)
        return 0, prepared.run_dir
    except Exception as exc:
        failed_session = _runtime_session(
            prepared,
            RuntimeStatus.FAILED,
            started_at=started_at,
            error_type=type(exc).__name__,
        )
        _write_json(prepared.run_dir / "runtime-session.json", failed_session)
        _persist_session(settings, failed_session)
        raise
