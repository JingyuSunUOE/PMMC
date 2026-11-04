"""CPU-only adoption of exact ready jobs, using the ordinary fenced worker.

This is execution policy, not a new scientific job/resource identity. A negative
probe leaves native scheduling alone; an invalid proof is never a negative probe.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pmmc.experiments.step8_artifacts import CampaignArtifactStore, json_bytes
from pmmc.experiments.step8_backend import ProductionComponentBackend, compiler_instance_id
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_runner import CampaignWorker
from pmmc.experiments.step8_upstream_import import (
    R0_PREFIX_STAGES,
    R0PrefixImportProof,
    compatible_r0_prefix_proofs,
    r0_prefix_compatibility,
    verified_r0_prefix_stage,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.step8_repositories import PostgresBenchmarkRepository
from pmmc.infrastructure.database.step8_upstream_import import PostgresR0PrefixImportRepository
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.base import JsonDict
from pmmc.schemas.step8 import BenchmarkCampaignManifest, ResourceClass, Step8BenchmarkConfig


def importable_job(job: BenchmarkStageJob) -> bool:
    return (
        job.operation == "compiler"
        and job.stage in R0_PREFIX_STAGES
        and job.parameters.get("treatment") != "fixed_planner"
    )


async def probe_import(
    backend: ProductionComponentBackend,
    job: BenchmarkStageJob,
    *,
    pinned: R0PrefixImportProof | None = None,
) -> R0PrefixImportProof | None:
    """Read and verify producer/consumer facts without adopting or starting models."""
    if not importable_job(job):
        return None
    scope = dict(job.parameters)
    target = compiler_instance_id(backend.campaign, scope)
    compatibility = r0_prefix_compatibility(backend.campaign, scope)
    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:
        repository = PostgresR0PrefixImportRepository(
            sessions, create_runtime_object_store(backend.settings)
        )
        if job.stage != "questioner":
            lineage = await repository.verify_target(target, compatibility)
            if lineage is None:
                return None
            proof = R0PrefixImportProof.model_validate(lineage["proof_json"])
        else:
            registered, qlineage = await repository.target_registration(target)
            lineage = await repository.prefix_registration(target)
            if registered and qlineage is None:
                if lineage is not None:
                    raise ValueError("import_only_prefix_without_questioner_lineage")
                return None
            if lineage is not None:
                proof = R0PrefixImportProof.model_validate(lineage["proof_json"])
            elif pinned is not None:
                proof = pinned
            else:
                proofs = compatible_r0_prefix_proofs(
                    backend.store, compatibility, exclude_run_id=target
                )
                if qlineage is not None:
                    proofs = [
                        proof
                        for proof in proofs
                        if proof.questioner_proof.content_hash == qlineage["proof_hash"]
                    ]
                if not proofs:
                    return None
                proof = proofs[0]
            if (
                qlineage is not None
                and qlineage["proof_hash"] != proof.questioner_proof.content_hash
            ):
                raise ValueError("import_only_questioner_producer_changed")
            await repository.verify_source(proof, compatibility)
        if pinned is not None and proof.content_hash != pinned.content_hash:
            raise ValueError("import_only_producer_changed_after_probe")
        return proof
    finally:
        await engine.dispose()


class ImportOnlyComponentBackend(ProductionComponentBackend):
    """Production receipts, but with no route to native compiler/model execution."""

    def pin_imports(self, candidates: dict[str, R0PrefixImportProof]) -> None:
        self.import_candidates = dict(candidates)

    def _pinned(self, job: BenchmarkStageJob) -> R0PrefixImportProof:
        if not importable_job(job) or job.job_id not in self.import_candidates:
            raise ValueError("import_only_job_not_selected")
        return self.import_candidates[job.job_id]

    async def preflight(self, job: BenchmarkStageJob) -> None:
        proof = self._pinned(job)
        await super().preflight(job)
        if await probe_import(self, job, pinned=proof) is None:
            raise ValueError("import_only_facts_no_longer_available")

    async def execute(
        self, job: BenchmarkStageJob, upstream: dict[str, str]
    ) -> tuple[JsonDict, list[str], dict[str, int]]:
        self._pinned(job)
        return await super().execute(job, upstream)

    async def _compiler(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        proof = self._pinned(job)
        # Revalidate after claiming and again on disk-receipt recovery. A miss or
        # race here is an error, not permission to dispatch native generation.
        if await probe_import(self, job, pinned=proof) is None:
            raise ValueError("import_only_facts_no_longer_available")
        if job.stage == "questioner":
            q_digest = self.store.put_json(proof.questioner_proof.model_dump(mode="json"))
            result = await self._try_questioner_import(
                scope, required_proof_hash=q_digest, prefix_proof=proof
            )
        else:
            result = await verified_r0_prefix_stage(self, scope, job.stage)
        if result is None:
            raise ValueError("import_only_facts_no_longer_available")
        if result.get("r0_prefix_adoption", {}).get("proof_hash") != proof.content_hash:
            raise ValueError("import_only_result_producer_mismatch")
        return result

    async def verify_recovery(
        self, job: BenchmarkStageJob, upstream: dict[str, str], receipt: ComponentReceipt
    ) -> None:
        output, hashes, counts = await self.execute(job, upstream)

        def facts(value: JsonDict) -> JsonDict:
            result = copy.deepcopy(value)
            for key in ("questioner_adoption", "r0_prefix_adoption"):
                if isinstance(result.get(key), dict):
                    result[key].pop("reused", None)
            return result

        old_output = hashlib.sha256(json_bytes(receipt.output)).hexdigest()
        new_output = hashlib.sha256(json_bytes(output)).hexdigest()
        if (
            facts(receipt.output) != facts(output)
            or receipt.row_counts != counts
            or old_output not in receipt.artifact_hashes
            or set(receipt.artifact_hashes) - {old_output} != set(hashes) - {new_output}
        ):
            raise ValueError("import_only_recovery_fact_conflict")


async def drain_ready_imports(
    *,
    settings: Settings,
    config: Step8BenchmarkConfig,
    manifest: BenchmarkCampaignManifest,
    directory: Path,
    ready: Sequence[BenchmarkStageJob],
    rows: Sequence[dict[str, Any]],
    event: Callable[[str], None] = print,
) -> bool:
    """Drain only pinned ready IDs; True asks the supervisor for a fresh DAG view."""
    eligible = [job for job in ready if importable_job(job)]
    if not eligible:
        return False
    store = CampaignArtifactStore(settings.resolve_path(config.output_root))
    backend = ImportOnlyComponentBackend(
        settings=settings, config=config, campaign=manifest, campaign_dir=directory, store=store
    )
    candidates = {}
    for job in eligible:
        # Keep ordinary source/config preflight even for an entirely reused batch.
        await ProductionComponentBackend.preflight(backend, job)
        proof = await probe_import(backend, job)
        if proof is not None:
            candidates[job.job_id] = proof
    if not candidates:
        return False
    from pmmc.experiments.step8_preflight import preflight_stage_window

    event(
        "NEXT_TWO_STAGE_PREFLIGHT "
        + json.dumps(
            preflight_stage_window(
                settings=backend.runtime_settings,
                config=config,
                protocol=manifest.protocol,
                directory=directory,
                store=store,
                rows=rows,
                selected=[job for job in eligible if job.job_id in candidates],
            ),
            sort_keys=True,
        )
    )
    backend.pin_imports(candidates)
    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:
        for resource in sorted({job.resource_class for job in eligible}, key=lambda r: r.value):
            selected = [
                job
                for job in eligible
                if job.resource_class is resource and job.job_id in candidates
            ]
            if not selected:
                continue
            stages = tuple(sorted({job.stage for job in selected}))
            worker = CampaignWorker(
                repository=PostgresBenchmarkRepository(
                    sessions, stages, job_ids=tuple(job.job_id for job in selected)
                ),
                backend=backend,
                campaign_id=manifest.campaign_id,
                campaign_dir=directory,
                resource=resource,
                execution=config.execution,
                execution_image=os.environ.get("PMMC_CONTAINER_IMAGE", "local-source"),
                artifact_store=store,
                event=event,
                stages=stages,
                execution_resource=ResourceClass.CPU,
            )
            await worker.run()
    finally:
        await engine.dispose()
    # Another worker can win a claim. Refresh even when our exact-ID worker got
    # no jobs, rather than using a stale ready list to start an unnecessary GPU.
    return True
