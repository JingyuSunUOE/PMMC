"""Local model artifacts required by a selected, dependency-closed native plan."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.release_plan import plan_from_environment, plan_hash
from pmmc.schemas.step8 import BenchmarkCampaignManifest


def stage_aliases(
    directory: Path, campaign: BenchmarkCampaignManifest, selection_hash: str
) -> dict:
    ids = json.loads((directory / "manifests" / f"jobs-{selection_hash}.json").read_bytes())
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("release_model_job_inventory_invalid")
    by_stage: dict[str, set[str]] = {}
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
    return {k: sorted(v) for k, v in sorted(by_stage.items())}


def selected_release_model_aliases(settings, config, *, expected_profiles=None) -> list[str]:
    """Verify the owned pointer, then project the *full lock* to required HF artifacts."""
    from pmmc.experiments.release_inputs import exploration_campaign_directory
    from pmmc.experiments.step8_campaign import load_benchmark_config
    from pmmc.models.locking import load_model_lock

    locked = load_model_lock(settings.resolve_path(config.model_lock))
    local = {entry.alias for entry in locked.entries}
    request = plan_from_environment(os.environ)
    if request is None:
        return sorted(local)  # compatibility with non-release native campaigns
    variant = os.environ.get("PMMC_RELEASE_VARIANT", request["variants"][0])
    if variant not in request["variants"]:
        raise ValueError("release_model_variant_outside_request")
    run_id = request["run_id"]
    directory = exploration_campaign_directory(settings, run_id, variant=variant)
    frozen = load_benchmark_config(directory / "manifests/config.json")
    if expected_profiles is not None and expected_profiles != {
        "runtime_profile": frozen.runtime_profile, "execution_profile": frozen.execution_profile,
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
    by_stage = pointer["model_aliases_by_stage"]
    stage = os.environ.get("PMMC_RELEASE_STAGE")
    if stage and stage != "all":
        if stage not in by_stage:
            raise ValueError("release_model_stage_outside_request")
        return sorted(set(by_stage[stage]) & local)
    return sorted(set().union(*by_stage.values()) & local)
