"""Stage-major physical-service scheduling over the persistent campaign DAG."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.experiments.step8_runner import record_startup_failure
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.step8_repositories import PostgresBenchmarkRepository
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.step8 import BenchmarkCampaignManifest, ResourceClass, Step8BenchmarkConfig


def ready_jobs(
    rows: Sequence[dict[str, Any]], *, now: datetime | None = None
) -> list[BenchmarkStageJob]:
    now = now or datetime.now(UTC)
    jobs = [(row, BenchmarkStageJob.model_validate(row["job"])) for row in rows]
    completed = {job.job_id for row, job in jobs if row["status"] == "completed"}

    def claimable(row: dict[str, Any]) -> bool:
        if row["status"] == "pending":
            return True
        expiry = row.get("lease_expires_at")
        if row["status"] != "running" or expiry is None:
            return False
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)
        return expiry < now

    return [
        job for row, job in jobs if claimable(row) and set(job.dependency_ids).issubset(completed)
    ]


def physical_stage(job: BenchmarkStageJob) -> str | None:
    if job.resource_class in {ResourceClass.CPU, ResourceClass.API}:
        return None
    if job.resource_class is ResourceClass.GPU_27B:
        return "answer_27b"
    if job.resource_class is ResourceClass.GPU_9B:
        return "compiler"
    return {
        "text_index": "embedding_text",
        "contrastive_index": "embedding_contrastive",
        "multimodal_index": "embedding_multimodal",
    }[job.stage]


def next_batch(jobs: Sequence[BenchmarkStageJob]) -> tuple[ResourceClass, str | None, list[str]]:
    if not jobs:
        raise ValueError("no_ready_campaign_jobs")
    priority = {
        "questioner": 10,
        "planner0": 11,
        "text_index": 20,
        "contrastive_index": 25,
        "multimodal_index": 30,
        "doubter": 50,
        "matching": 60,
        "answers": 70,
        "grounding": 90,
        "evaluation": 100,
    }
    first = min(
        jobs,
        key=lambda item: (
            0 if item.resource_class is ResourceClass.CPU else priority.get(item.stage, 110),
            item.resource_class.value,
            item.job_id,
        ),
    )
    physical = physical_stage(first)
    stages = sorted(
        {
            job.stage
            for job in jobs
            if job.resource_class == first.resource_class and physical_stage(job) == physical
        }
    )
    return first.resource_class, physical, stages


def physical_replica_count(
    settings: Settings,
    config: Step8BenchmarkConfig,
    physical: str,
) -> int:
    from pmmc.experiments.execution import load_execution_config
    from pmmc.runtime.config import load_runtime_profile

    profile = load_runtime_profile(settings.resolve_path(config.runtime_profile))
    counts: set[int] = set()
    for service in profile.services_for_stage(physical):
        frozen = service.metadata.get("frozen_replica_topology")
        if frozen is None:
            if physical == "answer_27b" or service.metadata.get("requires_capacity_receipt"):
                raise ValueError("step8_capacity_pool_requires_frozen_topology")
            execution = load_execution_config(settings.resolve_path(config.execution_profile))
            counts.add(execution.replicas.for_logical_stage(physical))
            continue
        if not isinstance(frozen, dict):
            raise ValueError("step8_invalid_frozen_replica_topology")
        tp, count = frozen.get("tensor_parallel_size"), frozen.get("replica_count")
        if (
            type(tp) is not int
            or type(count) is not int
            or tp <= 0
            or count <= 0
            or service.tensor_parallel_candidates != [tp]
        ):
            raise ValueError("step8_invalid_frozen_replica_topology")
        counts.add(count)
    if len(counts) != 1:
        raise ValueError("step8_inconsistent_physical_replica_counts")
    return counts.pop()


async def _snapshot(settings: Settings, campaign_id: str) -> list[dict[str, Any]]:
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        return await PostgresBenchmarkRepository(sessions).list_jobs(campaign_id)
    finally:
        await engine.dispose()


def supervise_campaign(
    *,
    settings: Settings,
    config: Step8BenchmarkConfig,
    manifest: BenchmarkCampaignManifest,
    directory: Path,
    has_api_budget: bool,
    rate_card: Path | None,
    resources: frozenset[ResourceClass] | None = None,
    job_scope: tuple[str, frozenset[str]] | None = None,
    event: Callable[[str], None] = print,
) -> dict[str, object]:
    import uuid

    from pmmc.experiments.step8_scratch import attempt_scratch

    identity = "attempt-" + uuid.uuid4().hex
    with attempt_scratch(settings, attempt_id=identity, campaign_dir=directory) as owned:
        return _supervise_campaign(
            settings=owned,
            config=config,
            manifest=manifest,
            directory=directory,
            has_api_budget=has_api_budget,
            rate_card=rate_card,
            resources=resources,
            job_scope=job_scope,
            event=event,
        )


def _supervise_campaign(
    *,
    settings: Settings,
    config: Step8BenchmarkConfig,
    manifest: BenchmarkCampaignManifest,
    directory: Path,
    has_api_budget: bool,
    rate_card: Path | None,
    resources: frozenset[ResourceClass] | None = None,
    job_scope: tuple[str, frozenset[str]] | None = None,
    event: Callable[[str], None] = print,
) -> dict[str, object]:
    """Keep compatible 9B services warm; CPU/API workers never own GPU leases.

    Model setup stays on the main thread for the managed signal/process guard.
    Database transactions are closed before starting or waiting on any child.
    """
    from pmmc.experiments.step8_artifacts import CampaignArtifactStore
    from pmmc.experiments.step8_checkpoint import CampaignCheckpointCoordinator
    from pmmc.experiments.step8_import_scheduler import drain_ready_imports
    from pmmc.experiments.step8_preflight import preflight_stage_window
    from pmmc.runtime.manager import managed_runtime_command_session

    allowed = resources or frozenset(ResourceClass)
    store = CampaignArtifactStore(settings.resolve_path(config.output_root))
    controller = ResourceClass.CPU in allowed
    coordinator = (
        CampaignCheckpointCoordinator(
            settings,
            manifest,
            directory,
            CampaignArtifactStore(settings.resolve_path(config.output_root)),
        )
        if controller
        else None
    )

    base_command = [
        sys.executable,
        "-m",
        "pmmc",
        "benchmark",
        "run",
        "--campaign-dir",
        str(directory),
    ]
    if rate_card is not None:
        base_command += ["--rate-card", str(rate_card)]
    if job_scope is not None:
        base_command += ["--exploration-scope", job_scope[0]]
    stage_startups: dict[str, int] = {}
    active_physical = None
    session = None
    idle_since: float | None = None
    api_process: subprocess.Popen | None = None
    stack = contextlib.ExitStack()
    primary: BaseException | None = None

    def close_runtime() -> None:
        from pmmc.experiments.step8_scratch import defer_scratch_cleanup

        try:
            stack.close()
        except BaseException:
            defer_scratch_cleanup("runtime_process_shutdown_unverified")
            raise

    try:
        while True:
            if api_process is not None and api_process.poll() is not None:
                if api_process.returncode:
                    raise RuntimeError("api_worker_failed_see_attempt_capsule")
                api_process = None
            rows = asyncio.run(_snapshot(settings, manifest.campaign_id))
            campaign_complete = bool(rows) and all(r["status"] == "completed" for r in rows)
            if coordinator is not None:
                asyncio.run(
                    coordinator.observe(
                        rows,
                        interval_seconds=config.execution.checkpoint_seconds,
                        job_interval=config.execution.checkpoint_jobs,
                    )
                )
            if job_scope is not None:
                rows = [r for r in rows if r["job_id"] in job_scope[1]]
                if len(rows) != len(job_scope[1]):
                    raise ValueError("registered_exploration_scope_incomplete")
            statuses = [row["status"] for row in rows]
            owned = [
                row
                for row in rows
                if BenchmarkStageJob.model_validate(row["job"]).resource_class in allowed
            ]
            if not rows:
                raise ValueError("campaign_has_no_registered_jobs")
            if any(value in {"failed", "unknown"} for value in statuses):
                raise RuntimeError("campaign_contains_failed_or_unknown_jobs")
            if all(row["status"] == "completed" for row in owned) and (
                not controller or all(value == "completed" for value in statuses)
            ):
                return {
                    "state": "worker_pool_completed",
                    "campaign_complete": campaign_complete,
                    "jobs": len(owned),
                    "physical_startups": stage_startups,
                }
            all_ready = ready_jobs(rows)
            ready = [job for job in all_ready if job.resource_class in allowed]
            if asyncio.run(drain_ready_imports(
                settings=settings, config=config, manifest=manifest, directory=directory,
                # A CPU controller can own verified reuse without allocating a
                # GPU worker. Native dispatch below still uses only allowed jobs.
                ready=all_ready if controller else ready, rows=rows, event=event,
            )):
                continue
            api_ready = [job for job in ready if job.resource_class is ResourceClass.API]
            if api_ready and api_process is None:
                if not has_api_budget:
                    raise ValueError("api_stage_requires_explicit_selection_budget")
                event(
                    "NEXT_TWO_STAGE_PREFLIGHT "
                    + json.dumps(
                        preflight_stage_window(
                            settings=settings,
                            config=config,
                            protocol=manifest.protocol,
                            directory=directory,
                            store=store,
                            rows=rows,
                            selected=api_ready,
                        ),
                        sort_keys=True,
                    )
                )
                api_process = subprocess.Popen(
                    base_command + ["--resource", ResourceClass.API.value],
                    stdin=subprocess.DEVNULL,
                    env=dict(os.environ),
                )
            ready = [job for job in ready if job.resource_class is not ResourceClass.API]
            if not ready:
                # A short CPU projection boundary must not reload all eight weights.
                # Bound this grace period so API waits cannot retain idle GPUs indefinitely.
                idle_since = idle_since if idle_since is not None else time.monotonic()
                if time.monotonic() - idle_since >= config.execution.gpu_idle_grace_seconds:
                    close_runtime()
                    session, active_physical = None, None
                if (
                    "running" not in statuses
                    and api_process is None
                    and not any(job.resource_class not in allowed for job in ready_jobs(rows))
                ):
                    raise RuntimeError("campaign_dependency_deadlock")
                event(
                    "STEP8_SCHEDULER "
                    + json.dumps(
                        {
                            "state": "waiting_on_workers",
                            "completed": statuses.count("completed"),
                            "pending": statuses.count("pending"),
                            "inflight": statuses.count("running"),
                            "gpu_service_active": session is not None,
                        },
                        sort_keys=True,
                    )
                )
                time.sleep(config.execution.heartbeat_seconds)
                continue
            idle_since = None
            resource, physical, stages = next_batch(ready)
            event(
                "NEXT_TWO_STAGE_PREFLIGHT "
                + json.dumps(
                    preflight_stage_window(
                        settings=settings,
                        config=config,
                        protocol=manifest.protocol,
                        directory=directory,
                        store=store,
                        rows=rows,
                        selected=[
                            job
                            for job in ready
                            if job.resource_class is resource and job.stage in stages
                        ],
                    ),
                    sort_keys=True,
                )
            )
            command = base_command + ["--resource", resource.value]
            for stage in stages:
                command.extend(["--stage", stage])
            if physical is None:
                subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
                continue
            if active_physical != physical:
                replicas = physical_replica_count(settings, config, physical)
                close_runtime()
                session = stack.enter_context(
                    managed_runtime_command_session(
                        settings,
                        profile_path=settings.resolve_path(config.runtime_profile),
                        logical_stage=physical,
                        replica_count=replicas,
                        environment_overrides={
                            "PMMC_PAPER_RESULT": str(manifest.paper_result).lower(),
                            "PMMC_MODEL_VERIFICATION_POLICY": "cached_receipt",
                        },
                    )
                )
                active_physical = physical
                stage_startups[physical] = stage_startups.get(physical, 0) + 1
            assert session is not None
            session.run_commands(
                logical_stage=physical,
                commands=[command],
                replica_count=session.replica_count,
            )
    except BaseException as error:
        primary = error
        if coordinator is not None:
            try:
                asyncio.run(coordinator("failure", {}))
            except BaseException as checkpoint_error:
                primary.add_note("supervisor checkpoint failed: " + type(checkpoint_error).__name__)
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            if primary is None:
                close_runtime()
            else:
                stack.__exit__(type(primary), primary, primary.__traceback__)
        except BaseException as secondary:
            from pmmc.experiments.step8_scratch import defer_scratch_cleanup

            defer_scratch_cleanup("runtime_process_shutdown_unverified")
            if primary is not None:
                primary.add_note("runtime cleanup failed: " + type(secondary).__name__)
            else:
                cleanup_error = secondary
        try:
            if api_process is not None and api_process.poll() is None:
                # Termination cannot resolve an in-flight provider spend reservation.
                api_process.terminate()
                try:
                    api_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    api_process.kill()
                    api_process.wait()
        except BaseException as secondary:
            from pmmc.experiments.step8_scratch import defer_scratch_cleanup

            defer_scratch_cleanup("api_process_shutdown_unverified")
            first = primary if primary is not None else cleanup_error
            if first is not None:
                first.add_note("API worker cleanup failed: " + type(secondary).__name__)
            else:
                cleanup_error = secondary
        first = primary if primary is not None else cleanup_error
        if first is not None:
            record_startup_failure(first, directory=directory, stage="supervisor")
        if primary is None and cleanup_error is not None:
            raise cleanup_error
