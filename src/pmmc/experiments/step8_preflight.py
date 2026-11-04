"""Check current and upcoming frozen dependencies before loading model weights."""

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.experiments.step8_config_contract import verify_frozen_execution_inputs
from pmmc.experiments.step8_jobs import (
    BenchmarkStageJob,
    ComponentReceipt,
    scientific_source_fingerprint,
)
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, ScientificProtocol, Step8BenchmarkConfig


def upcoming_stage_jobs(
    rows: Sequence[Mapping[str, Any]], selected: Sequence[BenchmarkStageJob],
) -> list[BenchmarkStageJob]:
    """Include two downstream stage names, without asserting future outputs exist."""
    current_stages = {job.stage for job in selected}
    jobs = {job.job_id: job for row in rows if row["status"] != "completed"
            for job in [BenchmarkStageJob.model_validate(row["job"]) ]}
    frontier = {job.job_id for job in selected}
    reached = set(frontier)
    following: list[str] = []
    while frontier and len(following) < 2:
        frontier = {key for key, job in jobs.items() if key not in reached
                    and set(job.dependency_ids) & frontier}
        reached.update(frontier)
        for stage in sorted({jobs[key].stage for key in frontier} - current_stages):
            if stage not in following:
                following.append(stage)
    included = current_stages | set(following[:2])
    return [jobs[key] for key in sorted(reached) if key in jobs and jobs[key].stage in included]


def preflight_stage_window(
    *, settings: Settings, config: Step8BenchmarkConfig, protocol: ScientificProtocol,
    directory: Path, store: CampaignArtifactStore,
    rows: Sequence[Mapping[str, Any]], selected: Sequence[BenchmarkStageJob],
) -> dict[str, Any]:
    verify_frozen_execution_inputs(settings, config, protocol)
    targets = upcoming_stage_jobs(rows, selected)
    fingerprints = {
        stage: scientific_source_fingerprint(settings.project_root, stage)
        for stage in {job.stage for job in targets}
    }
    from pmmc.experiments.step8_maintenance import execution_maintenance

    for job in targets:
        try:
            execution_maintenance(
                job=job, protocol=protocol, directory=directory,
                project_root=settings.project_root,
                actual_source_fingerprint=fingerprints[job.stage],
            )
        except ValueError as error:
            raise ValueError("step8_upcoming_stage_source_changed") from error
    by_id = {job.job_id: (row, job) for row in rows
             for job in [BenchmarkStageJob.model_validate(row["job"]) ]}
    dependencies = {key for job in targets for key in job.dependency_ids}
    artifacts: set[str] = set()
    verified = deferred = 0
    for key in sorted(dependencies):
        if key not in by_id:
            raise ValueError("step8_upcoming_dependency_missing")
        row, job = by_id[key]
        if row["status"] != "completed":
            deferred += 1
            continue
        receipt = ComponentReceipt.model_validate_json(
            (directory / "checkpoints/components" / f"{key}.json").read_bytes()
        )
        if (
            receipt.job_id != key or receipt.content_hash != row["receipt_hash"]
            or receipt.scientific_input_hash != job.scientific_input_hash
            or receipt.source_fingerprint != job.source_fingerprint
            or receipt.schema_revision != STEP8_ALEMBIC_REVISION
        ):
            raise ValueError("step8_upcoming_dependency_receipt_changed")
        artifacts.update(receipt.artifact_hashes)
        verified += 1
    for digest in sorted(artifacts):
        store.verify(digest)
    return {
        "state": "ready", "scope": "config_source_and_available_dependencies",
        "stages": dict(sorted(Counter(job.stage for job in targets).items())),
        "work_unit": "component_jobs", "verified_dependencies": verified,
        "deferred_dependencies": deferred, "verified_artifacts": len(artifacts),
        "collection_readiness": "checked_by_component_before_use",
    }
