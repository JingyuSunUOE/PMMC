"""Run open-model work first, with paid work limited to the same selection's approval."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from pmmc.experiments.release_scope import (
    ReleaseStage,
    publish_scope,
    require_preceding_stage,
    stage_job_ids,
)
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_backend import ProductionComponentBackend
from pmmc.experiments.step8_cli import _latest_selection, _load, _register
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.experiments.step8_supervisor import _snapshot, supervise_campaign
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.step8 import CampaignPhase, ResourceClass


async def selection_spend(settings, approval_id, campaign_id, selection_hash, *,
                          plan_hash=None, scope_hash=None):
    from pmmc.experiments.step8_conformance_replacement import read_ledger
    from pmmc.release_plan import plan_from_environment
    from pmmc.release_plan import plan_hash as hash_plan

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        approval, rows = await read_ledger(sessions, approval_id)
    finally:
        await engine.dispose()
    plan = plan_from_environment(os.environ)
    if plan is not None:
        if plan_hash is not None and plan_hash != hash_plan(plan):
            raise ValueError("exploration_ledger_identity_mismatch")
        plan_hash = hash_plan(plan)
    if approval.get("plan_hash") != plan_hash:
        raise ValueError("exploration_ledger_identity_mismatch")
    if plan_hash is None:
        if scope_hash is not None or (approval["campaign_id"], approval["selection_hash"]) != (
                campaign_id, selection_hash):
            raise ValueError("exploration_ledger_identity_mismatch")
    elif not any((member["campaign_id"], member["selection_hash"]) == (campaign_id, selection_hash)
                 and (scope_hash is None or scope_hash in member["scopes"].values())
                 for member in approval["plan_bindings"].values()):
        raise ValueError("exploration_ledger_identity_mismatch")
    limit = approval["limit_usd"]
    used = approval["settled_usd"] + approval["reserved_usd"]
    return {
        "scope": ("plan_cumulative_not_additive_across_stages_or_variants" if plan_hash
                  else "selection_cumulative_not_additive_across_stages"),
        "approval_id": approval_id,
        "plan_hash": plan_hash,
        "budget_mode": "uncapped" if limit is None else "finite",
        "limit_usd": str(limit) if limit is not None else None,
        "remaining_usd": str(limit - used) if limit is not None else None,
        "new_calls_stopped": approval.get("stopped_at") is not None,
        "stopped_at": approval["stopped_at"].isoformat() if approval.get("stopped_at") else None,
        "stopped_by": approval.get("stopped_by"),
        "stop_reason": approval.get("stop_reason"),
        "settled_usd": str(approval["settled_usd"]),
        "reserved_usd": str(approval["reserved_usd"]),
        "request_states": dict(Counter(r["state"] for r in rows)),
        "unknown_reserved_usd": str(sum(
            r["reserved_usd"] for r in rows if r["state"] == "unknown"
        )),
    }


def require_comparison_order(settings: Settings, run_id: str, variant: str, stage: ReleaseStage):
    from pmmc.experiments.release_inputs import exploration_campaign_directory

    if variant != "wr" and stage != "main":
        raise ValueError("comparison_variants_only_run_the_primary_matrix")
    parents = (["wr"] if variant != "wr" else
               ["w", "r", "legacy"] if stage == "baselines" else [])
    for producer in parents:
        directory = exploration_campaign_directory(settings, run_id, variant=producer)
        manifest, _ = _load(directory)
        rows = asyncio.run(_snapshot(settings, manifest.campaign_id))
        required = stage_job_ids(
            (BenchmarkStageJob.model_validate(row["job"]) for row in rows), "main", paid=True,
        )
        if not required <= {r["job_id"] for r in rows if r["status"] == "completed"}:
            raise ValueError(f"exploration_{producer}_main_is_incomplete")


def execute_stage(settings: Settings, directory: Path, stage: ReleaseStage,
                  pool: Literal["all", "control", "gpu"]) -> dict:
    started = time.monotonic()
    manifest, config = _load(directory)
    selection = _latest_selection(directory)
    if manifest.phase is not CampaignPhase.EXPLORATION or selection.selection != "dev-all":
        raise ValueError("release_stage_requires_dev_all_not_paper")
    approval, _ = asyncio.run(_register(directory, budget=None, rate_card=None,
                                        approver=None, retry=False))
    requested = os.getenv("PMMC_STEP8_BUDGET_APPROVAL_ID")
    if requested and approval != requested:
        raise ValueError("explicit_approval_does_not_match_exploration_selection")
    approval = requested
    rows = asyncio.run(_snapshot(settings, manifest.campaign_id))
    jobs = [BenchmarkStageJob.model_validate(r["job"]) for r in rows]
    statuses = {row["job_id"]: row["status"] for row in rows}
    request = config.release_request
    if request is None:
        require_preceding_stage(jobs, statuses, stage)
    else:
        from pmmc.release_plan import plan_hash

        if manifest.release_plan_hash != plan_hash(request) or selection.release_plan_hash != manifest.release_plan_hash:
            raise ValueError("release_stage_request_identity_mismatch")
        if not all(job.release_stages for job in jobs):
            raise ValueError("release_stage_requires_selected_native_inventory")
    full_scope = stage_job_ids(jobs, stage, paid=True)
    pending_paid = approval is None and (request is None or any(
        job.job_id in full_scope and job.resource_class == ResourceClass.API
        and statuses[job.job_id] != "completed" for job in jobs))
    scope = publish_scope(directory, manifest.campaign_id, selection.content_hash, stage,
                          jobs, paid=not pending_paid)
    card = (directory / "manifests" / f"rate-card-{selection.content_hash}.json"
            if approval else None)
    pools = {"control": {ResourceClass.CPU, ResourceClass.API},
             "gpu": {ResourceClass.GPU_9B, ResourceClass.GPU_27B, ResourceClass.GPU_EMBEDDING},
             "all": set(ResourceClass)}
    backend = ProductionComponentBackend(
        settings=settings, config=config, campaign=manifest, campaign_dir=directory,
        store=CampaignArtifactStore(settings.resolve_path(config.output_root)),
    )
    print("PMMC_RELEASE " + json.dumps(dict(stage=stage, worker_pool=pool,
          scoped_jobs=len(scope[1]), closed_api_authorized=bool(approval))), flush=True)
    result = supervise_campaign(
        settings=backend.runtime_settings, manifest=manifest, config=config, directory=directory,
        rate_card=card, has_api_budget=bool(approval), resources=frozenset(pools[pool]),
        job_scope=scope,
    )
    observed = asyncio.run(_snapshot(settings, manifest.campaign_id))
    complete = {row["job_id"] for row in observed if row["status"] == "completed"}
    done = len(complete.intersection(scope[1]))
    state = ("waiting_for_workers" if done != len(scope[1]) else
             "awaiting_budget" if pending_paid else "stage_complete")
    report = dict(protocol="pmmc-release-stage-v1", state=state, campaign_id=manifest.campaign_id,
                  selection_hash=selection.content_hash, experiment_kind="exploratory",
                  paper_result=False, stage=stage, worker_pool=pool, scope_hash=scope[0],
                  scoped_jobs=len(scope[1]), completed=done, paid_authorized=bool(approval),
                  query_denominators={d.value: len(qs) for d, qs in selection.query_ids.items()},
                  invocation_wall_seconds=round(time.monotonic() - started, 3),
                  physical_startups=result["physical_startups"])
    write_json(directory / "reports" / f"release-{stage}-{pool}.json", report)
    if pool != "gpu":
        from pmmc.experiments.release_reporting import summarize_release_campaign

        metrics = summarize_release_campaign(directory, jobs=observed, scope=stage)
        if approval:
            metrics["selection_ledger"] = asyncio.run(selection_spend(
                settings, approval, manifest.campaign_id, selection.content_hash,
                plan_hash=manifest.release_plan_hash,
                scope_hash=scope[0] if request is not None else None))
        write_json(directory / "reports" / f"exploration-{stage}.json", metrics)
        if state == "stage_complete":
            from pmmc.experiments.release_readouts import seal_report

            seal_report(directory, stage, metrics, report)
    return report
