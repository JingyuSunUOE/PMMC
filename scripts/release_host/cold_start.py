"""Executed inside the digest-pinned application image, not imported by the host."""

import json
import os
import sys
from pathlib import Path


def models(argv):
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
    config = load_benchmark_config(settings.resolve_path(Path(argv[argv.index("--config") + 1])))
    lock_path = settings.resolve_path(config.model_lock)
    if lock_path.exists():
        lock, lock_hash = _read_lock(lock_path)
        registry = load_model_registry(settings.resolve_path(config.model_registry))
        aliases = sorted(model.alias for model in registry.models)
        _validate_lock(lock, registry, aliases)
        local = None
        if "PMMC_RELEASE_REQUEST" in os.environ:
            from pmmc.experiments.step8_model_preparation import selected_release_model_aliases

            local = selected_release_model_aliases(settings, config)
        try:
            cached = prepare_model_cache_audited(
                lock,
                settings.model_cache_path,
                aliases=local,
                local_files_only=True,
                verification_policy=ModelVerificationPolicy.CACHED_RECEIPT,
                allow_full_fallback=False,
            )
        except (ValueError, OSError):
            pass  # Native FULL verification/locked fetch must now succeed.
        else:
            _check_verification(cached, lock, cached=True, aliases=local)
            if _read_lock(lock_path)[1] != lock_hash:
                raise ValueError("release_lock_changed")
            print(
                json.dumps(
                    {
                        "protocol": "release-locked-cache-hit-v1",
                        "ready": True,
                        "policy": "cached_receipt",
                        "lock_sha256": lock_hash,
                        "local_model_aliases": local,
                    }
                )
            )
            return
    os.execvp(argv[0], argv)


def gpu(argv):
    from pmmc.infrastructure.settings import get_settings

    settings = get_settings()
    config = None
    if "PMMC_RELEASE_REQUEST" in os.environ:
        from pmmc.experiments.release_models import selected_release_model_aliases
        from pmmc.experiments.step8_campaign import load_benchmark_config

        config = load_benchmark_config(settings.resolve_path(Path(os.environ["PMMC_STEP8_CONFIG"])))
        topo = json.loads(os.environ.get("PMMC_RELEASE_TOPOLOGY", "null"))
        if topo is None or topo["gpus"] == 0:
            if selected_release_model_aliases(settings, config, calibration=True):
                raise ValueError("release_worker_positive_gpu_topology_required")
            os.execvp(argv[0], argv)
            return

    from pmmc.runtime.config import load_runtime_profile
    from pmmc.runtime.manager import _service_backend_version, calibrate_runtime, prepare_runtime
    from pmmc.runtime.step8_capacity import require_step8_capacity_receipts
    from pmmc.runtime.step8_capacity_calibration import calibrate_step8_capacity

    profiles = None
    if "PMMC_RELEASE_TOPOLOGY" in os.environ:
        from pmmc.runtime.release_topology import materialize

        profiles = materialize(settings)
        path = settings.resolve_path(profiles["runtime_profile"])
        settings = settings.model_copy(update={"runtime_profile_path": path})
        os.environ["PMMC_RUNTIME_PROFILE_PATH"] = str(path)
    profile = load_runtime_profile(settings.resolved_runtime_profile_path)
    if profiles is not None and profile.metadata.get("release_visible_gpus") == 0:
        raise ValueError("CPU-only planning profile cannot certify GPU capacity")
    # Native calibration validates hardware/backend/model/profile identity and reuses
    # matching durable file or database records before starting any model service.
    measure = os.environ.get("PMMC_RELEASE_MEASURE", "1") == "1"
    if config is not None:
        from pmmc.runtime.release_topology import calibration_stages

        if profiles is None:
            raise ValueError("release_worker_topology_missing")
        aliases = set(
            selected_release_model_aliases(
                settings, config, expected_profiles=profiles, calibration=True
            )
        )
        selected = calibration_stages(profile, aliases)
        for stage, _ in selected:
            calibrate_runtime(
                settings,
                profile_path=settings.resolved_runtime_profile_path,
                logical_stage=stage,
                verify_only=not measure,
            )
        if not any(s.metadata.get("requires_capacity_receipt") for _, s in selected):
            os.execvp(argv[0], argv)
            return
    else:
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
        if not measure:
            raise ValueError("release_capacity_measurement_required") from None
        calibrate_step8_capacity(settings, profile_path=settings.resolved_runtime_profile_path)
    else:
        if not gated or set(stage.service_ids) != gated:
            raise ValueError("release_capacity_gate_missing")
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    {"models": models, "gpu": gpu}[sys.argv[1]](sys.argv[2:])
