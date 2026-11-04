"""Local model artifacts required by a selected, dependency-closed native plan."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.experiments.step8_model_identity import (
    COMPILER_ALIAS,
    CONTRASTIVE_ALIAS,
    GROUNDING_ALIAS,
    JUDGE_ALIAS,
    MULTIMODAL_ALIAS,
    RERANKER_ALIAS,
    TEXT_ALIAS,
)
from pmmc.release_plan import plan_from_environment, plan_hash
from pmmc.schemas.step8 import ANSWER_BACKBONES, BenchmarkCampaignManifest, ResourceClass

_SERVICE_ALIASES = {
    ResourceClass.GPU_9B: {
        COMPILER_ALIAS,
        RERANKER_ALIAS,
        GROUNDING_ALIAS,
        JUDGE_ALIAS,
        ANSWER_BACKBONES[0],
    },
    ResourceClass.GPU_27B: {ANSWER_BACKBONES[1]},
    ResourceClass.GPU_EMBEDDING: {TEXT_ALIAS, MULTIMODAL_ALIAS, CONTRASTIVE_ALIAS},
}


def stage_requirements(
    directory: Path, campaign: BenchmarkCampaignManifest, selection_hash: str
) -> dict:
    ids = json.loads((directory / "manifests" / f"jobs-{selection_hash}.json").read_bytes())
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("release_model_job_inventory_invalid")
    by_stage: dict[str, set[str]] = {}
    calibration: dict[str, set[str]] = {}
    needs_gpu: dict[str, bool] = {}
    for key in ids:
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("release_model_job_identity_invalid")
        path = directory / "private/jobs" / f"{key}.json"
        if path.resolve() != path:
            raise ValueError("release_model_job_path_invalid")
        job = BenchmarkStageJob.model_validate_json(path.read_bytes())
        aliases = job.scientific_inputs["model_entry_hashes"]
        if (
            job.job_id != key
            or job.campaign_id != campaign.campaign_id
            or not job.release_stages
            or not isinstance(aliases, dict)
            or any(campaign.protocol.model_entry_hashes.get(a) != h for a, h in aliases.items())
        ):
            raise ValueError("release_model_job_identity_invalid")
        for stage in job.release_stages:
            by_stage.setdefault(stage, set()).update(aliases)
            # Some aliases are used for token budgets or lineage outside their
            # serving pool. Cache their files, but do not start those services.
            calibration.setdefault(stage, set()).update(
                _SERVICE_ALIASES.get(job.resource_class, set()).intersection(aliases)
            )
            needs_gpu[stage] = needs_gpu.get(stage, False) or job.resource_class in _SERVICE_ALIASES
    return {
        "model_aliases_by_stage": {k: sorted(v) for k, v in sorted(by_stage.items())},
        "calibration_aliases_by_stage": {k: sorted(v) for k, v in sorted(calibration.items())},
        "local_gpu_required_by_stage": dict(sorted(needs_gpu.items())),
    }


def selected_release_model_aliases(
    settings, config, *, expected_profiles=None, calibration=False
) -> list[str]:
    """Verify the owned pointer, then project the *full lock* to required HF artifacts."""
    from pmmc.models.locking import load_model_lock

    locked = load_model_lock(settings.resolve_path(config.model_lock))
    local = {entry.alias for entry in locked.entries}
    request = plan_from_environment(os.environ)
    if request is None:
        return sorted(local)  # compatibility with non-release native campaigns
    variant = os.environ.get("PMMC_RELEASE_VARIANT", request["variants"][0])
    if variant not in request["variants"]:
        raise ValueError("release_model_variant_outside_request")
    stage = os.environ.get("PMMC_RELEASE_STAGE")
    variants = request["variants"] if stage == "prepare-all" else [variant]
    names = set()
    for arm in variants:
        pointer = _bound_pointer(settings, config, request, arm, expected_profiles)
        field = "calibration_aliases_by_stage" if calibration else "model_aliases_by_stage"
        by_stage = pointer[field]
        if stage in (None, "all", "prepare-all"):
            names.update(alias for aliases in by_stage.values() for alias in aliases)
        elif stage in by_stage:
            names.update(by_stage[stage])
        else:
            raise ValueError("release_model_stage_outside_request")
    return sorted(names & local)


def _bound_pointer(settings, config, request, variant, expected_profiles):
    from pmmc.experiments.release_inputs import exploration_campaign_directory
    from pmmc.experiments.step8_campaign import load_benchmark_config

    run_id = request["run_id"]
    directory = exploration_campaign_directory(settings, run_id, variant=variant)
    frozen = load_benchmark_config(directory / "manifests/config.json")
    if expected_profiles is not None and expected_profiles != {
        "runtime_profile": frozen.runtime_profile,
        "execution_profile": frozen.execution_profile,
    }:
        raise ValueError("release_worker_topology_differs_from_frozen_plan")
    if (
        frozen.release_request != request
        or frozen.model_lock != config.model_lock
        or frozen.model_registry != config.model_registry
    ):
        raise ValueError("release_model_request_or_lock_mismatch")
    pointer = json.loads(
        (
            settings.outputs_path / "step8/exploration" / run_id / variant / "pointer.json"
        ).read_bytes()
    )
    if pointer["release_plan_hash"] != plan_hash(request):
        raise ValueError("release_model_pointer_identity_mismatch")
    return pointer
