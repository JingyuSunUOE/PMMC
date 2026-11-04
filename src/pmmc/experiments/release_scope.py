"""Select dependency-closed slices of the original, immutable exploration DAG."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import write_json
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.schemas.step8 import ANSWER_BACKBONES, ResourceClass

ReleaseStage = Literal["main", "analysis", "baselines"]
TARGET_OPERATIONS = {
    "main": "primary_evaluate",
    "analysis": "analysis_evaluate",
    "baselines": "evaluate",
}


def stage_job_ids(jobs: Iterable[BenchmarkStageJob], stage: ReleaseStage, *,
                  paid: bool) -> frozenset[str]:
    by_id = {job.job_id: job for job in jobs}
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


def publish_scope(directory: Path, campaign_id: str, selection_hash: str, stage: ReleaseStage,
                  jobs: list[BenchmarkStageJob], paid: bool) -> tuple[str, frozenset[str]]:
    selected = stage_job_ids(jobs, stage, paid=paid)
    payload = dict(campaign=campaign_id, selection=selection_hash, stage=stage, paid=paid,
                   job_ids=sorted(selected))
    digest = canonical_content_hash(payload)
    write_json(directory / "private/scopes" / f"{digest}.json", payload, immutable=True)
    return digest, selected


def read_scope(directory: Path, digest: str, campaign_id: str, selection_hash: str):
    import json
    import re

    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid_exploration_scope_hash")
    payload = json.loads((directory / "private/scopes" / f"{digest}.json").read_bytes())
    if (canonical_content_hash(payload) != digest or payload["campaign"] != campaign_id
            or payload["selection"] != selection_hash or not payload["job_ids"]):
        raise ValueError("exploration_scope_identity_changed")
    return tuple(payload["job_ids"])
