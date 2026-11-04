"""Explicit Step8 cold model preparation, without GPU or provider inference calls.

Only config.model_registry/config.model_lock choose scientific identity. Runtime
defaults or a handler's resolved registry must not select or rewrite that identity.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pmmc.experiments.release_models import selected_release_model_aliases
from pmmc.experiments.step8_artifacts import durable_write, json_bytes, sha256_file
from pmmc.infrastructure.settings import Settings
from pmmc.models.locking import (
    ModelCachePreparation,
    assert_paper_lock,
    build_model_lock,
    prepare_model_cache_audited,
    write_verified_model_cache_manifest,
)
from pmmc.models.registry import ModelRegistry, load_model_registry
from pmmc.runtime.locking import RuntimeLockError, RuntimeProcessLock
from pmmc.schemas import ModelEndpointKind, ModelLockManifest, ModelVerificationPolicy
from pmmc.schemas.step8 import Step8BenchmarkConfig


class Step8ModelPreparationError(RuntimeError):
    """Print only code/report, never the underlying Hub exception or traceback locals."""

    def __init__(self, code: str, report: dict[str, Any]):
        super().__init__(code)
        self.code = code
        self.report = report


class _PreparationFailure(RuntimeError):
    pass


def _validate_lock(lock: ModelLockManifest, registry: ModelRegistry, aliases: list[str]) -> None:
    # The managed planner consumes the complete Step8 registry, not a legacy
    # registry subset, and checks this identity before placing services.
    if lock.registry_hash != registry.resolved_hash:
        raise _PreparationFailure("model_lock_registry_identity_mismatch")
    assert_paper_lock(lock, registry, aliases=aliases)
    # Multiple Step8 roles serving the same requested physical model must not
    # accidentally resolve different Hub commits while a floating ref changes.
    physical: dict[tuple[str, str], tuple[str, str]] = {}
    for entry in lock.entries:
        key = (entry.repo_id, entry.requested_revision)
        identity = (entry.commit_sha, entry.snapshot_content_hash)
        if physical.setdefault(key, identity) != identity:
            raise _PreparationFailure("physical_model_alias_identity_mismatch")


def _read_lock(path: Path) -> tuple[ModelLockManifest, str]:
    if path.is_symlink() or not path.is_file():
        raise _PreparationFailure("model_lock_must_be_regular_file")
    payload = path.read_bytes()
    return ModelLockManifest.model_validate_json(payload), hashlib.sha256(payload).hexdigest()


def _check_verification(
    preparation: ModelCachePreparation,
    lock: ModelLockManifest,
    *,
    cached: bool,
    previous: ModelCachePreparation | None = None,
    aliases: list[str] | None = None,
) -> None:
    entries = {
        entry.alias: entry for entry in lock.entries if aliases is None or entry.alias in aliases
    }
    if (
        (not entries if aliases is None else set(aliases) != set(entries))
        or set(preparation.paths) != set(entries)
        or set(preparation.verifications) != set(entries)
    ):
        raise _PreparationFailure("model_verification_alias_coverage_mismatch")
    policy = ModelVerificationPolicy.CACHED_RECEIPT if cached else ModelVerificationPolicy.FULL
    for alias, entry in entries.items():
        result = preparation.verifications[alias]
        if (
            result.alias != alias
            or result.repo_id != entry.repo_id
            or result.commit_sha != entry.commit_sha
            or result.policy is not policy
            or not result.receipt_id
            or result.receipt_cache_hit is not cached
            or result.verified_files != len(entry.files)
            or Path(result.snapshot_path) != preparation.paths[alias]
            or (cached and result.bytes_hashed != 0)
        ):
            raise _PreparationFailure("model_verification_receipt_contract_failed")
        if previous is not None and (
            result.receipt_id != previous.verifications[alias].receipt_id
            or preparation.paths[alias] != previous.paths[alias]
        ):
            raise _PreparationFailure("model_verification_changed_between_passes")


def prepare_step8_models(
    settings: Settings,
    config: Step8BenchmarkConfig,
    *,
    allow_download: bool = False,
    create_lock: bool = False,
) -> dict[str, Any]:
    """Validate/create once, FULL-verify weights, then require offline cached receipts.

    ``create_lock`` permits creation only when absent; it never refreshes or
    overwrites an existing lock. Creation needs ``allow_download`` for Hub
    metadata. That explicit flag also controls weight downloads, independently
    of Settings.mode; False never calls Hub authentication or online APIs.

    Re-running always performs FULL verification before checking the resulting
    receipts, even when a historical receipt exists. Closed-provider aliases are
    identity-checked by the existing locker and never downloaded or invoked.
    A failed download retains a newly published lock for an identical retry.

    Success persists a safe report and a cached-receipt preparation manifest
    below config.output_root/model-preparation. It does not approve a campaign,
    calibrate a service, rewrite either registry, or start any GPU process.
    The CLI must catch Step8ModelPreparationError, print only its report, and
    exit nonzero without traceback locals or the original Hub exception.
    """
    report: dict[str, Any] = {
        "protocol": "step8-model-preparation-v1",
        "scope": "model_artifacts_only",
        "ready": False,
        "allow_download": allow_download is True,
        "create_lock": create_lock is True,
        "lock_state": "unchecked",
        "provider_calls": 0,
        "gpu_operations": 0,
        "calibration": "not_checked_or_modified",
    }
    phase = "configuration"
    try:
        if type(allow_download) is not bool or type(create_lock) is not bool:
            raise _PreparationFailure("model_preparation_flags_must_be_boolean")
        registry_path = settings.resolve_path(config.model_registry).absolute()
        lock_path = settings.resolve_path(config.model_lock).absolute()
        cache_dir = settings.model_cache_path.absolute()
        registry_hash = sha256_file(registry_path)
        registry = load_model_registry(registry_path)
        if any(model.endpoint_kind is ModelEndpointKind.FAKE for model in registry.models):
            raise _PreparationFailure("step8_model_registry_cannot_contain_fake_models")
        aliases = sorted(model.alias for model in registry.models)
        report.update(
            managed_registry_path=str(registry_path),
            registry_file_sha256=registry_hash,
            registry_hash=registry.resolved_hash,
            lock_path=str(lock_path),
            cache_root=str(cache_dir),
        )
        phase = "lock"
        if not os.path.lexists(lock_path):
            if "PMMC_RELEASE_REQUEST" in os.environ:
                raise _PreparationFailure("release_requires_shipped_model_lock")
            if not create_lock:
                raise _PreparationFailure("model_lock_missing_create_not_authorized")
            if not allow_download:
                raise _PreparationFailure("model_lock_creation_requires_download_permission")
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex
        # Reuse the existing advisory-lock helper, but with a shared cache path,
        # not the GPU lifecycle path. Concurrent preparations fail fast.
        with RuntimeProcessLock(cache_dir / ".pmmc/step8-model-preparation.lock", run_id):
            if os.path.lexists(lock_path):
                report["lock_state"] = "reused"
            else:
                if "PMMC_RELEASE_REQUEST" in os.environ or not create_lock or not allow_download:
                    raise _PreparationFailure("model_lock_disappeared_during_preparation")
                token = settings.hf_token.get_secret_value() if settings.hf_token else None
                candidate = build_model_lock(registry, aliases=aliases, token=token)
                _validate_lock(candidate, registry, aliases)
                # immutable=True publishes by no-clobber link, not rename. Even
                # a competing creator with a different cache cannot replace it.
                durable_write(lock_path, json_bytes(candidate.to_dict()), immutable=True)
                report["lock_state"] = "created"
            lock, lock_hash = _read_lock(lock_path)
            _validate_lock(lock, registry, aliases)
            open_aliases = (
                selected_release_model_aliases(settings, config)
                if "PMMC_RELEASE_REQUEST" in os.environ
                else sorted(e.alias for e in lock.entries)
            )
            report.update(
                lock_sha256=lock_hash,
                local_model_aliases=open_aliases,
                native_provider_aliases=sorted(set(aliases) - {e.alias for e in lock.entries}),
                physical_snapshot_count=len(
                    {(e.repo_id, e.commit_sha) for e in lock.entries if e.alias in open_aliases}
                ),
                campaign_inputs={
                    "validation": "not_performed",
                    "created_or_modified": False,
                    "requirement": "preserve_real_partition_and_exposure_provenance",
                    "required_paths": {
                        name: str(settings.resolve_path(getattr(config, name)).absolute())
                        for name in ("partition_dir", "exposure_manifest", "regression_manifest")
                    },
                },
            )
            phase = "full_verification"
            token = (
                settings.hf_token.get_secret_value()
                if open_aliases and allow_download and settings.hf_token
                else None
            )
            full = prepare_model_cache_audited(
                lock,
                cache_dir,
                token=token,
                aliases=open_aliases,
                local_files_only=not allow_download,
                verification_policy=ModelVerificationPolicy.FULL,
                allow_full_fallback=True,
            )
            _check_verification(full, lock, cached=False, aliases=open_aliases)
            report["full_verification"] = {
                "policy": "full",
                "bytes_hashed": sum(result.bytes_hashed for result in full.verifications.values()),
                "verified_aliases": open_aliases,
            }
            phase = "cached_verification"
            cached = prepare_model_cache_audited(
                lock,
                cache_dir,
                token=None,
                aliases=open_aliases,
                local_files_only=True,
                verification_policy=ModelVerificationPolicy.CACHED_RECEIPT,
                allow_full_fallback=False,
            )
            _check_verification(cached, lock, cached=True, previous=full, aliases=open_aliases)
            if lock_path.is_symlink() or sha256_file(lock_path) != lock_hash:
                raise _PreparationFailure("model_lock_changed_during_preparation")
            if sha256_file(registry_path) != registry_hash:
                raise _PreparationFailure("model_registry_changed_during_preparation")
            report["cached_verification"] = {
                "policy": "cached_receipt",
                "offline": True,
                "bytes_hashed": 0,
                "verification_receipts": {
                    alias: result.receipt_id
                    for alias, result in sorted(cached.verifications.items())
                },
            }
            phase = "receipt"
            output = settings.resolve_path(config.output_root) / "model-preparation" / run_id
            manifest_path = output / "verified-cache.json"
            manifest = write_verified_model_cache_manifest(
                destination=manifest_path,
                lock_path=lock_path,
                lock=lock,
                preparation=cached,
                verification_policy=ModelVerificationPolicy.CACHED_RECEIPT,
            )
            if manifest.lock_sha256 != lock_hash:
                raise _PreparationFailure("model_lock_changed_during_preparation")
            completed = {
                **report,
                "ready": True,
                "verified_cache_manifest": str(manifest_path),
                "verified_cache_manifest_hash": manifest.content_hash,
                "report_path": str(output / "report.json"),
            }
            durable_write(output / "report.json", json_bytes(completed), immutable=True)
            return completed
    except Exception as exc:
        if isinstance(exc, _PreparationFailure):
            code = str(exc)
        elif isinstance(exc, RuntimeLockError):
            code = "model_preparation_already_running"
        else:
            code = f"model_preparation_{phase}_failed"
        report.update(failed_phase=phase, error_code=code)
        raise Step8ModelPreparationError(code, report) from None
