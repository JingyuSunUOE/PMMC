"""Executed inside the digest-pinned application image, not imported by the host."""

import json
import os
import sys


def models(argv):
    from huggingface_hub.errors import LocalEntryNotFoundError

    from pmmc.experiments.step8_campaign import load_benchmark_config
    from pmmc.experiments.step8_model_preparation import (
        _check_verification,
        _read_lock,
        _validate_lock,
    )
    from pmmc.infrastructure.settings import get_settings
    from pmmc.models.locking import ModelVerificationPolicy, prepare_model_cache_audited
    from pmmc.models.registry import load_model_registry

    settings = get_settings()
    config = load_benchmark_config(settings.resolve_path(argv[argv.index("--config") + 1]))
    lock_path = settings.resolve_path(config.model_lock)
    if lock_path.exists():
        lock, lock_hash = _read_lock(lock_path)
        registry = load_model_registry(settings.resolve_path(config.model_registry))
        aliases = sorted(model.alias for model in registry.models)
        _validate_lock(lock, registry, aliases)
        try:
            cached = prepare_model_cache_audited(
                lock,
                settings.model_cache_path,
                aliases=aliases,
                local_files_only=True,
                verification_policy=ModelVerificationPolicy.CACHED_RECEIPT,
                allow_full_fallback=False,
            )
        except (ValueError, OSError, LocalEntryNotFoundError):
            pass  # Native FULL verification/locked fetch must now succeed.
        else:
            _check_verification(cached, lock, cached=True)
            if _read_lock(lock_path)[1] != lock_hash:
                raise ValueError("release_lock_changed")
            print(
                json.dumps(
                    {
                        "protocol": "release-locked-cache-hit-v1",
                        "ready": True,
                        "policy": "cached_receipt",
                        "lock_sha256": lock_hash,
                    }
                )
            )
            return
    os.execvp(argv[0], argv)


def gpu(argv):
    from pmmc.infrastructure.settings import get_settings
    from pmmc.runtime.config import load_runtime_profile
    from pmmc.runtime.manager import _service_backend_version, calibrate_runtime, prepare_runtime
    from pmmc.runtime.step8_capacity import require_step8_capacity_receipts
    from pmmc.runtime.step8_capacity_calibration import calibrate_step8_capacity

    settings = get_settings()
    profile = load_runtime_profile(settings.resolved_runtime_profile_path)
    # Native calibration validates hardware/backend/model/profile identity and reuses
    # matching durable file or database records before starting any model service.
    calibrate_runtime(settings, profile_path=settings.resolved_runtime_profile_path)
    (service,) = profile.services_for_stage("answer_27b")
    count = service.metadata["frozen_replica_topology"]["replica_count"]
    prepared = prepare_runtime(
        settings,
        logical_stage="answer_27b",
        replica_count=count,
        stage_model_snapshots_to_local=False,
        paper_result=True,
    )
    (stage,) = prepared.plan.stages
    try:
        gated = require_step8_capacity_receipts(
            prepared, stage, backend_version=_service_backend_version
        )
    except (ValueError, FileNotFoundError):
        # This validator binds the full pool/layout to actual GPU IDs and backend.
        # Fresh native measurement publishes only after all probes and shutdown.
        calibrate_step8_capacity(settings, profile_path=settings.resolved_runtime_profile_path)
    else:
        if not gated or set(stage.service_ids) != gated:
            raise ValueError("release_capacity_gate_missing")
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    {"models": models, "gpu": gpu}[sys.argv[1]](sys.argv[2:])
