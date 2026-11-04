"""Read-only verification of the exact campaign DAG and durable component chain."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pmmc.experiments.step8_artifacts import CampaignArtifactStore, json_bytes
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt, validate_job_dag
from pmmc.schemas.step8 import STEP8_READABLE_REVISIONS, BenchmarkCampaignManifest


def _owned_file(directory: Path, path: Path) -> bool:
    root = directory.absolute()
    if not path.absolute().is_relative_to(root) or root.is_symlink():
        return False
    return (
        path.is_file()
        and path.resolve().is_relative_to(root.resolve())
        and not any(
            item.is_symlink()
            for item in (path, *path.parents)
            if item.absolute().is_relative_to(root)
        )
    )


def verify_components(
    *,
    directory: Path,
    campaign: BenchmarkCampaignManifest,
    rows: Sequence[dict[str, Any]],
    store: CampaignArtifactStore,
) -> dict[str, Any]:
    issues: set[str] = set()
    expected: set[str] = set()
    for path in sorted((directory / "manifests").glob("jobs-*.json")):
        if not _owned_file(directory, path):
            raise ValueError("unsafe_campaign_job_inventory")
        value = json.loads(path.read_text())
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))
        ):
            raise ValueError("invalid_campaign_job_inventory")
        expected.update(value)
    jobs: dict[str, BenchmarkStageJob] = {}
    receipts: dict[str, ComponentReceipt] = {}
    statuses: Counter[str] = Counter()
    for row in rows:
        job = BenchmarkStageJob.model_validate(row["job"])
        if job.job_id in jobs:
            issues.add("duplicate_database_stage_job")
        jobs[job.job_id] = job
        statuses[row["status"]] += 1
        if job.campaign_id != campaign.campaign_id:
            issues.add("foreign_campaign_stage_job")
        if job.source_fingerprint != campaign.protocol.source_fingerprints.get(job.stage):
            issues.add("stage_frozen_source_mismatch:" + job.stage)
        if row["status"] != "completed":
            issues.add("stage_not_completed:" + job.stage)
            continue
        path = directory / "checkpoints/components" / f"{job.job_id}.json"
        if not path.is_file():
            issues.add("component_receipt_missing:" + job.stage)
            continue
        if not _owned_file(directory, path):
            issues.add("component_receipt_unsafe:" + job.stage)
            continue
        try:
            receipt = ComponentReceipt.model_validate_json(path.read_bytes())
            if (
                receipt.job_id != job.job_id
                or receipt.scientific_input_hash != job.scientific_input_hash
                or receipt.source_fingerprint != job.source_fingerprint
                or receipt.schema_revision not in STEP8_READABLE_REVISIONS
                or receipt.content_hash != row["receipt_hash"]
                or set(receipt.upstream_receipt_hashes) != set(job.dependency_ids)
            ):
                issues.add("component_receipt_identity_mismatch:" + job.stage)
                continue
            output_hash = hashlib.sha256(json_bytes(receipt.output)).hexdigest()
            if output_hash not in receipt.artifact_hashes:
                issues.add("component_output_artifact_missing:" + job.stage)
            for digest in receipt.artifact_hashes:
                store.verify(digest)
            receipts[job.job_id] = receipt
        except (ValueError, OSError):
            issues.add("component_artifact_verification_failed:" + job.stage)
    if not expected or expected != set(jobs):
        issues.add("campaign_job_denominator_mismatch")
    try:
        validate_job_dag(list(jobs.values()))
    except ValueError:
        issues.add("campaign_dependency_graph_invalid")
    for receipt in receipts.values():
        if any(
            parent not in receipts or receipts[parent].content_hash != digest
            for parent, digest in receipt.upstream_receipt_hashes.items()
        ):
            issues.add("upstream_component_hash_chain_incomplete")
    return {
        "pipeline_complete": bool(expected) and not issues,
        "stage_count": len(jobs),
        "expected_stage_count": len(expected),
        "status_counts": dict(sorted(statuses.items())),
        "component_chain_hash": hashlib.sha256(
            json_bytes({key: item.content_hash for key, item in sorted(receipts.items())})
        ).hexdigest(),
        "issues": sorted(issues),
    }
