"""Select dependency-closed slices of the original, immutable exploration DAG."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import write_json
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.schemas.step8 import (
    ANSWER_BACKBONES,
    BenchmarkCampaignManifest,
    ResourceClass,
    Step8BenchmarkConfig,
)

ReleaseStage = Literal["main", "analysis", "baselines"]
TARGET_OPERATIONS = {
    "main": "primary_evaluate",
    "analysis": "analysis_evaluate",
    "baselines": "evaluate",
}


def _closure(jobs, roots):
    chosen = set()
    frontier = list(roots)
    while frontier:
        key = frontier.pop()
        if key not in chosen:
            chosen.add(key)
            frontier.extend(jobs[key].dependency_ids)
    return chosen


def selected_campaign_jobs(jobs, request, variant):
    """Publish only selected production operations and their actual prerequisites."""
    from collections import defaultdict

    if request["stage"] == "analysis" and not request["with_deps"]:
        raise ValueError("source_analysis_cannot_plan_native_work_without_deps")
    stages = defaultdict(list)
    components = {
        "compile": {"compiler", "adopt_questioner"},
        "index": {"baseline_index", "suite", "baseline_retrieve", "query_embedding"},
        "match": {"query_match"}, "execute": {"query_execute"},
        "answer": {"baseline_answer", "query_answer", "native_fallback_generate"},
        # The native PMMC select phase grounds its own answer and persists the
        # terminal selection together. Neither frontier may skip that grounding.
        "ground": {"baseline_grounding", "native_fallback_ground", "query_grounding"},
        "select": {"query_grounding"},
        "score": {"primary_evaluate", "analysis_evaluate", "baseline_evaluate"},
        "seal": {"primary_evaluate", "analysis_evaluate", "baseline_evaluate"},
    }
    requested_ops = {op for c in request["components"] for op in components[c]}
    for stage, operation in {**TARGET_OPERATIONS, "baselines": "baseline_evaluate"}.items():
        if stage == "main":
            if request["stage"] == "baselines":
                continue
        elif stage == "analysis":
            if (variant != "wr" or request["stage"] not in {"all", "analysis"}
                    or "ablation" not in request["analyses"]):
                continue
        elif variant != "wr" or request["stage"] not in {"all", "baselines"}:
            continue
        chosen = _closure(jobs, [key for key, job in jobs.items() if job.operation == operation])
        if requested_ops:
            chosen = _closure(jobs, [k for k in chosen if jobs[k].operation in requested_ops])
        for key in chosen:
            stages[key].append(stage)
    if not stages:
        raise ValueError("empty_selected_campaign_dag")
    return [job.model_copy(update={"release_stages": tuple(stages[key])})
            for key, job in jobs.items() if key in stages]


def stage_job_ids(jobs: Iterable[BenchmarkStageJob], stage: ReleaseStage, *,
                  paid: bool) -> frozenset[str]:
    by_id = {job.job_id: job for job in jobs}
    if any(job.release_stages for job in by_id.values()):
        if not all(job.release_stages for job in by_id.values()):
            raise ValueError("mixed_release_job_inventories")
        selected = {k for k, j in by_id.items() if stage in j.release_stages}
        if not selected or _closure(by_id, selected) != selected:
            raise ValueError("empty_or_open_release_stage")
        if not paid:
            from functools import cache

            @cache
            def requires_api(k):
                job = by_id[k]
                return job.resource_class == ResourceClass.API or any(requires_api(p) for p in job.dependency_ids)

            selected = {k for k in selected if not requires_api(k)}
        return frozenset(selected)
    frontier = [
        key for key, job in by_id.items()
        if job.operation == TARGET_OPERATIONS[stage]
        and (stage != "baselines" or job.parameters.get("treatment") == "main")
        and (paid or job.parameters.get("answer_model") in ANSWER_BACKBONES[:2])
    ]
    if not frontier:
        raise ValueError("empty_exploration_stage")
    selected: set[str] = set()
    while frontier:
        key = frontier.pop()
        if key in selected:
            continue
        job = by_id[key]
        if not paid and job.resource_class is ResourceClass.API:
            raise ValueError("paid_dependency_in_open_exploration_scope")
        selected.add(key)
        frontier.extend(job.dependency_ids)
    return frozenset(selected)


def require_preceding_stage(
    jobs: list[BenchmarkStageJob], statuses: Mapping[str, str], stage: ReleaseStage,
) -> None:
    predecessors: dict[ReleaseStage, ReleaseStage] = {"analysis": "main", "baselines": "analysis"}
    predecessor = predecessors.get(stage)
    if predecessor is not None:
        required = stage_job_ids(jobs, predecessor, paid=True)
        if any(statuses.get(key) != "completed" for key in required):
            raise ValueError(f"exploration_{predecessor}_is_incomplete")


def _scope_payload(campaign, selection, stage, paid, selected, plan_hash):
    payload = dict(campaign=campaign, selection=selection, stage=stage, paid=paid, job_ids=sorted(selected))
    if plan_hash is not None:
        payload["release_plan_hash"] = plan_hash
    return payload


def freeze_selected_scopes(directory, selection, jobs):
    """Materialize small worker admission proofs alongside the immutable plan."""
    if not selection.release_plan_hash or not all(job.release_stages for job in jobs):
        raise ValueError("release_scope_inventory_missing_binding")
    bindings = {}
    for stage in sorted({s for job in jobs for s in job.release_stages}):
        bindings[stage] = {}
        for paid, key in ((False, "open"), (True, "paid")):
            selected = stage_job_ids(jobs, stage, paid=paid)
            payload = _scope_payload(selection.campaign_id, selection.content_hash, stage, paid,
                                     selected, selection.release_plan_hash)
            bindings[stage][key] = canonical_content_hash(payload)
    write_json(directory / "manifests" / f"release-scopes-{selection.content_hash}.json", {
        "campaign": selection.campaign_id, "selection": selection.content_hash,
        "release_plan_hash": selection.release_plan_hash, "bindings": bindings,
    }, immutable=True)


def _check_frozen_scope(directory, payload, digest):
    import json

    frozen = json.loads((directory / "manifests" / f"release-scopes-{payload['selection']}.json").read_bytes())
    if (any(frozen[key] != payload[key] for key in ("campaign", "selection", "release_plan_hash"))
            or type(payload["paid"]) is not bool
            or digest != frozen["bindings"].get(payload["stage"], {}).get("paid" if payload["paid"] else "open")):
        raise ValueError("release_worker_scope_outside_request")


def publish_scope(directory: Path, campaign_id: str, selection_hash: str, stage: ReleaseStage,
                  jobs: list[BenchmarkStageJob], paid: bool) -> tuple[str, frozenset[str]]:
    selected = stage_job_ids(jobs, stage, paid=paid)
    digest_plan = _plan_identity(directory, campaign_id, selection_hash) if any(j.release_stages for j in jobs) else None
    payload = _scope_payload(campaign_id, selection_hash, stage, paid, selected, digest_plan)
    digest = canonical_content_hash(payload)
    if digest_plan is not None:
        if not (directory / "manifests" / f"release-scopes-{selection_hash}.json").exists():
            # Upgrade an already published inventory only after checking every
            # immutable job file; a worker cannot synthesize its own allowlist.
            from pmmc.experiments.step8_campaign import persist_campaign_job_plan
            from pmmc.schemas.step8 import SelectionExpansionManifest

            if not (directory / "manifests" / f"jobs-{selection_hash}.json").is_file():
                raise ValueError("release_scope_missing_native_inventory")
            selection = SelectionExpansionManifest.model_validate_json(
                (directory / "manifests/selections" / f"{selection_hash}.json").read_bytes())
            persist_campaign_job_plan(directory=directory, selection=selection, jobs=jobs)
        _check_frozen_scope(directory, payload, digest)
    write_json(directory / "private/scopes" / f"{digest}.json", payload, immutable=True)
    return digest, selected


def read_scope(directory: Path, digest: str, campaign_id: str, selection_hash: str):
    import json
    import os
    import re

    from pmmc.release_plan import plan_from_environment, plan_hash as hash_plan

    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid_exploration_scope_hash")
    payload = json.loads((directory / "private/scopes" / f"{digest}.json").read_bytes())
    if (canonical_content_hash(payload) != digest or payload["campaign"] != campaign_id
            or payload["selection"] != selection_hash or not payload["job_ids"]):
        raise ValueError("exploration_scope_identity_changed")
    plan_hash = _plan_identity(directory, campaign_id, selection_hash)
    if plan_hash != payload.get("release_plan_hash"):
        raise ValueError("release_worker_request_identity_mismatch")
    # Compare host intent before a paid worker exports its native recipe to descendants.
    if payload["paid"]:
        host = plan_from_environment(os.environ)
        if host is not None and (hash_plan(host) != plan_hash
                                 or os.environ.get("PMMC_RELEASE_STAGE", payload["stage"]) != payload["stage"]):
            raise ValueError("release_worker_paid_host_intent_mismatch")
        arm = os.environ.get("PMMC_RELEASE_VARIANT")
        if host is not None and arm is not None:
            config = Step8BenchmarkConfig.model_validate_json((directory / "manifests/config.json").read_bytes())
            if arm != config.metadata.get("exploration_variant"):
                raise ValueError("release_worker_paid_host_variant_mismatch")
    if plan_hash is not None:
        _check_frozen_scope(directory, payload, digest)
    return tuple(payload["job_ids"])


def _plan_identity(directory: Path, campaign_id: str, selection_hash: str):
    from pmmc.release_plan import plan_hash
    from pmmc.schemas.step8 import SelectionExpansionManifest

    manifest = BenchmarkCampaignManifest.model_validate_json((directory / "manifests/campaign.json").read_bytes())
    config = Step8BenchmarkConfig.model_validate_json((directory / "manifests/config.json").read_bytes())
    selection = SelectionExpansionManifest.model_validate_json(
        (directory / "manifests/selections" / f"{selection_hash}.json").read_bytes())
    request = config.release_request
    digest = plan_hash(request) if request is not None else None
    if (manifest.campaign_id != campaign_id or selection.campaign_id != campaign_id
            or selection.content_hash != selection_hash or manifest.release_plan_hash != digest
            or selection.release_plan_hash != digest):
        raise ValueError("release_worker_request_identity_mismatch")
    return digest
