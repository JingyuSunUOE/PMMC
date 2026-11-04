"""Development-only source references; never relabel old results as new execution."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select, text

from pmmc.experiments.step8_artifacts import json_bytes, sha256_file
from pmmc.experiments.step8_backend import ProductionComponentBackend
from pmmc.experiments.step8_development_admission import (
    bound_identity,
    require_engineering_admission,
)
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.infrastructure.database.step8_models import (
    BenchmarkComponentReceiptModel,
    BenchmarkStageJobModel,
)
from pmmc.infrastructure.database.step8_repositories import (
    PostgresBenchmarkRepository,
    _insert_exact,
)

COMPILER_STAGES = (
    "unit_prepare",
    "questioner",
    "planner0",
    "text_index",
    "multimodal_index",
    "dry_run0",
    "r0_gate",
    "doubter",
    "projection",
)


class SourceReuseCatalog:
    def __init__(self, *, manifest, directory: Path, root: Path):
        self.directory = directory
        self.binding = bound_identity(manifest, directory)
        self.entries = {e["target_job"]: e for e in self.binding["entries"]}
        if len(self.entries) != len(self.binding["entries"]):
            raise ValueError("duplicate_source_reference")
        self.bound_ids = frozenset(self.entries)
        admission = require_engineering_admission(
            manifest=manifest, directory=directory, source_bound_job_ids=self.bound_ids
        )
        if admission is None:
            raise ValueError("source_reuse_engineering_admission_required")
        self.source = root / self.binding["source_directory"]
        self.jobs = {
            j.job_id: j
            for path in sorted((directory / "private/jobs").glob("*.json"))
            for j in [BenchmarkStageJob.model_validate_json(path.read_bytes())]
        }
        for target_id, entry in self.entries.items():
            job = self.jobs[target_id]
            if (
                job.campaign_id != self.binding["target_campaign_id"]
                or job.scientific_input_hash != entry["target_scientific_input_hash"]
                or job.source_fingerprint != entry["target_source_fingerprint"]
                or job.stage != entry["stage"]
                or job.operation != entry["operation"]
                or set(job.dependency_ids) != set(entry["upstream_bindings"])
            ):
                raise ValueError("source_reference_target_changed")
            receipt = self.receipt(target_id)
            parents = {
                self.entries[key]["source_job"]: self.entries[key]["source_receipt_hash"]
                for key in job.dependency_ids
            }
            if parents != receipt.upstream_receipt_hashes or entry["upstream_bindings"] != {
                key: {
                    "source_job": self.entries[key]["source_job"],
                    "source_receipt_hash": self.entries[key]["source_receipt_hash"],
                }
                for key in job.dependency_ids
            }:
                raise ValueError("source_reference_dependency_changed")

    def receipt(self, target_id: str) -> ComponentReceipt:
        entry = self.entries.get(target_id)
        path = (self.source if entry else self.directory) / "checkpoints/components"
        identity = entry["source_job"] if entry else target_id
        receipt = ComponentReceipt.model_validate_json((path / f"{identity}.json").read_bytes())
        if receipt.job_id != identity:
            raise ValueError("source_reference_receipt_job_changed")
        if entry and (
            receipt.content_hash != entry["source_receipt_hash"]
            or receipt.scientific_input_hash != entry["source_scientific_input_hash"]
            or receipt.source_fingerprint != entry["source_fingerprint"]
            or receipt.artifact_hashes != entry["source_artifact_hashes"]
            or hashlib.sha256(json_bytes(receipt.output)).hexdigest()
            != entry["source_output_sha256"]
        ):
            raise ValueError("source_reference_receipt_changed")
        return receipt

    def parent_value(self, job, key):
        seen, queue = set(), list(job.dependency_ids)
        while queue:
            identity = queue.pop(0)
            if identity in seen:
                continue
            seen.add(identity)
            receipt = self.receipt(identity)
            if key in receipt.output:
                return receipt.output[key]
            # Traverse the target graph, not the donor receipt's producer IDs.
            queue.extend(self.jobs[identity].dependency_ids)
        raise ValueError("required_component_parent_output_missing_" + key)

    def dependency(self, target_id):
        entry = self.entries.get(target_id)
        return (
            (entry["source_job"], self.binding["source_campaign_id"], entry["source_receipt_hash"])
            if entry
            else (target_id, self.binding["target_campaign_id"], None)
        )

    def provenance(self, job):
        return {
            "protocol": "step8-source-preserving-execution-v1",
            "binding_hash": self.binding["content_hash"],
            "executor_sha256": sha256_file(Path(__file__)),
            "upstream": {
                key: {
                    "producer_job_id": self.dependency(key)[0],
                    "producer_campaign_id": self.dependency(key)[1],
                    "receipt_hash": self.receipt(key).content_hash,
                }
                for key in job.dependency_ids
            },
            "scientific_approval": False,
            "paper_test_allowed": False,
        }


class SourcePreservingRepository(PostgresBenchmarkRepository):
    def __init__(self, sessions, *, catalog, job_ids):
        super().__init__(sessions, job_ids=tuple(job_ids))
        self.catalog = catalog
        if set(job_ids) & catalog.bound_ids or not set(job_ids) <= catalog.jobs.keys():
            raise ValueError("source_bound_jobs_must_not_be_executed")

    async def claim(self, campaign_id, resource, owner, lease_seconds):
        if (
            campaign_id != self.catalog.binding["target_campaign_id"]
            or not owner
            or lease_seconds < 30
        ):
            raise ValueError("invalid_source_preserving_claim")
        aliases = {
            key: {"job": e["source_job"], "hash": e["source_receipt_hash"]}
            for key, e in self.catalog.entries.items()
        }
        async with self.sessions.begin() as session:
            identity = (
                await session.execute(
                    text("""
                SELECT job_id FROM pmmc_runtime.benchmark_stage_jobs AS job
                WHERE campaign_id=:campaign AND resource_class=:resource
                  AND job_id = ANY(:jobs)
                  AND (status='pending' OR
                       (status='running' AND lease_expires_at < clock_timestamp()))
                  AND NOT EXISTS (
                    SELECT 1 FROM unnest(job.dependency_ids) AS dep(id)
                    LEFT JOIN pmmc_runtime.benchmark_stage_jobs AS upstream
                      ON upstream.job_id=COALESCE(CAST(:aliases AS jsonb)->dep.id->>'job', dep.id)
                     AND upstream.campaign_id=CASE WHEN CAST(:aliases AS jsonb) ? dep.id
                         THEN :source ELSE :campaign END
                    WHERE upstream.status IS DISTINCT FROM 'completed'
                       OR upstream.receipt_hash IS NULL
                       OR (CAST(:aliases AS jsonb) ? dep.id AND upstream.receipt_hash
                           IS DISTINCT FROM (CAST(:aliases AS jsonb)->dep.id->>'hash')))
                ORDER BY created_at, job_id FOR UPDATE OF job SKIP LOCKED LIMIT 1
            """),
                    {
                        "campaign": campaign_id,
                        "resource": resource.value,
                        "jobs": list(self.job_ids),
                        "aliases": json.dumps(aliases),
                        "source": self.catalog.binding["source_campaign_id"],
                    },
                )
            ).scalar_one_or_none()
            if identity is None:
                return None
            row = await session.get(BenchmarkStageJobModel, identity)
            row.status, row.lease_owner = "running", owner
            row.attempt_count += 1
            row.lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
            return BenchmarkStageJob.model_validate(row.payload), row.attempt_count

    async def _parents(self, session, job):
        identities = {key: self.catalog.dependency(key) for key in job.dependency_ids}
        rows = (
            await session.scalars(
                select(BenchmarkStageJobModel).where(
                    BenchmarkStageJobModel.job_id.in_([value[0] for value in identities.values()])
                )
            )
        ).all()
        by_id = {row.job_id: row for row in rows}
        result = {}
        for key, (identity, campaign, expected) in identities.items():
            row = by_id.get(identity)
            if (
                row is None
                or row.campaign_id != campaign
                or row.status != "completed"
                or not row.receipt_hash
                or (expected and row.receipt_hash != expected)
            ):
                raise ValueError("source_reference_database_fact_changed")
            if self.catalog.receipt(key).content_hash != row.receipt_hash:
                raise ValueError("source_reference_file_database_mismatch")
            result[key] = row.receipt_hash
        return result

    async def upstream_receipts(self, job):
        async with self.sessions() as session:
            return await self._parents(session, job)

    async def complete(self, receipt, owner, fence):
        if receipt.job_id not in self.job_ids:
            raise ValueError("completion_outside_source_preserving_scope")
        async with self.sessions.begin() as session:
            row = await self._owned(session, receipt.job_id, owner, fence)
            job = BenchmarkStageJob.model_validate(row.payload)
            if (
                row.campaign_id != self.catalog.binding["target_campaign_id"]
                or row.scientific_input_hash != receipt.scientific_input_hash
                or row.source_fingerprint != receipt.source_fingerprint
                or await self._parents(session, job) != receipt.upstream_receipt_hashes
                or receipt.output.get("source_reuse") != self.catalog.provenance(job)
            ):
                raise ValueError("source_preserving_completion_identity_changed")
            await _insert_exact(
                session,
                BenchmarkComponentReceiptModel,
                "receipt_hash",
                {
                    "receipt_hash": receipt.content_hash,
                    "component_identity": receipt.component_identity,
                    "job_id": receipt.job_id,
                    "payload": receipt.model_dump(mode="json"),
                    "artifact_hashes": receipt.artifact_hashes,
                    "created_at": datetime.now(UTC),
                },
            )
            row.status, row.receipt_hash = "completed", receipt.content_hash
            row.completed_at = datetime.now(UTC)
            row.lease_owner = row.lease_expires_at = None


class SourcePreservingBackend(ProductionComponentBackend):
    def __init__(self, *, catalog, **kwargs):
        super().__init__(**kwargs)
        self.catalog = catalog

    def execution_maintenance(self, job):
        from pmmc.experiments import step8_doubter_identity_revision as identity
        from pmmc.experiments import step8_doubter_revision as doubter
        from pmmc.experiments import step8_unit_budget_revision as revision

        if ((self.campaign_dir / identity.APPROVAL).is_file()
            and job.job_id in identity.policy()["remaining_job_ids"]):
            return identity.execution_revision(self, job)
        if ((self.campaign_dir / doubter.APPROVAL).is_file()
            and job.job_id in doubter.policy()["remaining_job_ids"]):
            return doubter.execution_revision(self, job)
        if (self.campaign_dir / revision.APPROVAL).is_file():
            return revision.execution_revision(self, job)
        return super().execution_maintenance(job)

    def _upstream(self, job):
        return [self.catalog.receipt(key) for key in job.dependency_ids]

    def _parent_value(self, job, key):
        return self.catalog.parent_value(job, key)

    async def execute(self, job, upstream):
        if (
            job.job_id in self.catalog.bound_ids
            or job.stage not in COMPILER_STAGES
            or job.operation not in {"units", "compiler"}
            or job.parameters.get("treatment", "main") != "main"
        ):
            raise ValueError("outside_new_main_compiler_execution_scope")
        payload, artifacts, counts = await super().execute(job, upstream)
        payload["source_reuse"] = self.catalog.provenance(job)
        return payload, list(dict.fromkeys([*artifacts, self.store.put_json(payload)])), counts
