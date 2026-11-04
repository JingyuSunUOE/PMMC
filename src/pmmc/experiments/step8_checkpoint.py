"""Campaign checkpoint coordination over the audited database/object-store backup."""

from __future__ import annotations

import asyncio
import fcntl
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.checkpoint import ExperimentCheckpointService
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, sha256_file, write_json
from pmmc.experiments.step8_backend import compiler_instance_id
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.step8_repositories import PostgresBenchmarkRepository
from pmmc.infrastructure.object_store import create_evaluator_object_stores
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, BenchmarkCampaignManifest

# Only identifiers and terminal scientific state enter these digests, never leases or payloads.
_RUNTIME_PROGRESS_SQL = text("""
WITH owned_calls AS (
    SELECT model_call_id, run_id, status FROM pmmc_runtime.model_calls
    WHERE run_id = ANY(CAST(:compiler_runs AS text[]))
       OR (metadata->>'step8_campaign_id' = :campaign
           AND metadata->>'step8_component_job_id' = ANY(CAST(:job_ids AS text[])))
), owned_runs AS (
    SELECT unnest(CAST(:compiler_runs AS text[])) AS run_id
    UNION SELECT run_id FROM owned_calls
), facts AS (
    SELECT 'runtime_model_calls' AS family, model_call_id AS fact_id,
           jsonb_build_array(model_call_id, status)::text AS fact
    FROM owned_calls
    UNION ALL
    SELECT 'compiler_jobs', stage_job_id,
           jsonb_build_array(stage_job_id, status, content_hash, finished_at)::text
    FROM pmmc_runtime.compiler_stage_jobs
    WHERE run_id = ANY(CAST(:compiler_runs AS text[]))
      AND status IN ('succeeded','failed','cancelled')
    UNION ALL
    SELECT 'candidate_jobs', candidate_stage_job_id,
           jsonb_build_array(candidate_stage_job_id, status, content_hash, finished_at)::text
    FROM pmmc_runtime.candidate_stage_jobs
    WHERE run_id = ANY(CAST(:compiler_runs AS text[]))
      AND status IN ('succeeded','failed','cancelled')
    UNION ALL
    SELECT 'query_jobs', query_stage_job_id,
           jsonb_build_array(query_stage_job_id, status, content_hash, finished_at)::text
    FROM pmmc_runtime.query_executor_stage_jobs
    WHERE query_executor_run_id IN (SELECT run_id FROM owned_runs)
      AND status IN ('succeeded','failed','cancelled')
    UNION ALL
    SELECT 'experiment_jobs', job_id,
           jsonb_build_array(job_id, status, finished_at)::text
    FROM pmmc_runtime.experiment_jobs
    WHERE run_id IN (SELECT run_id FROM owned_runs)
      AND status IN ('succeeded','failed','cancelled')
)
SELECT family, count(*) AS fact_count,
       md5(string_agg(fact, '' ORDER BY fact_id)) AS digest
FROM facts GROUP BY family ORDER BY family
""")

_EVALUATOR_PROGRESS_SQL = text("""
WITH owned_calls AS (
    SELECT model_call_id, evaluation_run_id, status FROM pmmc_evaluation.model_calls
    WHERE metadata->>'step8_campaign_id' = :campaign
      AND metadata->>'step8_component_job_id' = ANY(CAST(:job_ids AS text[]))
), owned_runs AS (
    SELECT DISTINCT evaluation_run_id FROM owned_calls WHERE evaluation_run_id IS NOT NULL
    UNION
    SELECT evaluation_run_id FROM pmmc_evaluation.evaluation_runs
    WHERE metadata->>'campaign_id' = :campaign
), facts AS (
    SELECT 'evaluation_model_calls' AS family, model_call_id AS fact_id,
           jsonb_build_array(model_call_id, status)::text AS fact
    FROM owned_calls
    UNION ALL
    SELECT 'evaluation_run_states', evaluation_run_id,
           jsonb_build_array(evaluation_run_id, status, succeeded_count, failed_count)::text
    FROM pmmc_evaluation.evaluation_run_states
    WHERE evaluation_run_id IN (SELECT evaluation_run_id FROM owned_runs)
    UNION ALL
    SELECT 'evaluation_jobs', evaluation_job_id,
           jsonb_build_array(evaluation_job_id, status, finished_at)::text
    FROM pmmc_evaluation.evaluation_jobs
    WHERE evaluation_run_id IN (SELECT evaluation_run_id FROM owned_runs)
      AND status IN ('succeeded','failed','cancelled')
    UNION ALL
    SELECT 'evaluation_artifacts', artifact_id,
           jsonb_build_array(artifact_id, sha256, state)::text
    FROM pmmc_evaluation.run_artifacts
    WHERE evaluation_run_id IN (SELECT evaluation_run_id FROM owned_runs) AND state='ready'
)
SELECT family, count(*) AS fact_count,
       md5(string_agg(fact, '' ORDER BY fact_id)) AS digest
FROM facts GROUP BY family ORDER BY family
""")


@dataclass(frozen=True)
class _ScientificProgress:
    receipts: dict[str, str]
    facts: dict[str, dict[str, int | str]]

    @property
    def fact_hash(self) -> str:
        return canonical_content_hash({"receipts": self.receipts, "scientific_facts": self.facts})

    @property
    def terminal_jobs(self) -> int:
        return len(self.receipts) + sum(
            int(value["count"]) for name, value in self.facts.items() if name.endswith("_jobs")
        )

    @property
    def has_facts(self) -> bool:
        return bool(self.receipts) or any(int(value["count"]) for value in self.facts.values())


class CampaignCheckpointCoordinator:
    def __init__(
        self,
        settings: Settings,
        manifest: BenchmarkCampaignManifest,
        directory: Path,
        store: CampaignArtifactStore,
    ) -> None:
        self.settings, self.manifest, self.directory, self.store = (
            settings,
            manifest,
            directory,
            store,
        )
        self.service = ExperimentCheckpointService(
            settings,
            launch_id=manifest.campaign_id,
            root=directory / "checkpoints/database",
            identity={
                "launcher_schema_version": "0.11.0",
                "launch_config_hash": manifest.protocol.content_hash,
                "source_inputs_hash": manifest.partition_id,
                "model_lock_sha256": manifest.protocol.model_lock_hash,
                "execution_profile_hash": None,
                "alembic_revision": STEP8_ALEMBIC_REVISION,
            },
            retain_count=2,
            retain_milestones=True,
            backup_stores_factory=lambda: create_evaluator_object_stores(settings),
        )
        self._observed_stages: set[str] = set()
        self._completed_at_checkpoint = 0
        self._fact_hash_at_checkpoint: str | None = None
        self._checkpoint_clock = time.monotonic()
        self._barrier_diagnostic: tuple[int, int] | None = None

    def _round_barriers(self, rows: list[dict[str, Any]]) -> dict[str, str]:
        jobs = {BenchmarkStageJob.model_validate(row["job"]).job_id: row["job"] for row in rows}
        result = {}
        for path in sorted((self.store.root / "_store/round-barriers").glob("*.json")):
            if path.is_symlink():
                raise ValueError("doubter_round_barrier_identity_mismatch")
            marker = json.loads(path.read_bytes())
            if marker.get("campaign_id") != self.manifest.campaign_id:
                continue
            value = self.store.load_json(marker["barrier_sha256"])
            job = jobs.get(value["job_id"])
            if (
                job is None
                or job["stage"] != "doubter"
                or value["source_fingerprint"] != job["source_fingerprint"]
                or value["campaign_id"] != self.manifest.campaign_id
                or value["run_id"] != compiler_instance_id(self.manifest, job["parameters"])
                or value["round_id"] not in range(4)
                or value.get("protocol") != "step8-doubter-round-barrier-v1"
                or marker.get("job_id") != value["job_id"]
                or marker.get("round_id") != value["round_id"]
                or path.stem
                != canonical_content_hash(
                    [value["campaign_id"], value["job_id"], value["run_id"], value["round_id"]]
                )
                or not isinstance(value.get("assessment_hashes"), dict)
            ):
                raise ValueError("doubter_round_barrier_identity_mismatch")
            result[path.stem] = marker["barrier_sha256"]
        return result

    async def _validated_round_barriers(self, rows: list[dict[str, Any]]) -> dict[str, str]:
        markers = self._round_barriers(rows)
        if not markers:
            return {}
        values = {key: self.store.load_json(digest) for key, digest in markers.items()}
        runs = sorted({value["run_id"] for value in values.values()})
        engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
        try:
            async with sessions() as session:
                await session.execute(text("SET TRANSACTION READ ONLY"))
                await session.execute(text("SET LOCAL statement_timeout = '15s'"))
                result = await session.execute(
                    text("""
                    SELECT run_id, round_id, assessment_id, content_hash
                    FROM pmmc_runtime.doubter_assessments
                    WHERE run_id = ANY(CAST(:runs AS text[]))
                """),
                    {"runs": runs},
                )
                actual: dict[tuple[str, int], dict[str, str]] = {}
                for row in result.mappings():
                    actual.setdefault((row["run_id"], row["round_id"]), {})[
                        row["assessment_id"]
                    ] = row["content_hash"]
        finally:
            await engine.dispose()
        valid = {}
        for key, value in values.items():
            expected = value["assessment_hashes"]
            found = actual.get((value["run_id"], value["round_id"]), {})
            if any(
                identifier in found and found[identifier] != digest
                for identifier, digest in expected.items()
            ):
                raise ValueError("doubter_round_barrier_assessment_changed")
            # File journals may be newer than a restored DB. They are not proof of
            # completion until the exact authoritative assessments exist again.
            if found == expected and any(run == value["run_id"] for run, _ in actual):
                valid[key] = markers[key]
        diagnostic = len(valid), len(markers) - len(valid)
        if diagnostic != self._barrier_diagnostic:
            print(
                "STEP8_ROUND_MILESTONES "
                + json.dumps(
                    {
                        "validated_barriers": diagnostic[0],
                        "database_unconfirmed_barriers": diagnostic[1],
                        "database_snapshot_policy": "coalesced_across_compiler_jobs",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            self._barrier_diagnostic = diagnostic
        return valid

    def _complete_round_milestones(
        self, rows: list[dict[str, Any]], barriers: dict[str, str]
    ) -> list[int]:
        expected = {
            BenchmarkStageJob.model_validate(row["job"]).job_id
            for row in rows
            if row["job"]["stage"] == "doubter"
        }
        by_round: dict[int, set[str]] = {}
        for digest in barriers.values():
            value = self.store.load_json(digest)
            by_round.setdefault(value["round_id"], set()).add(value["job_id"])
        return [
            round_id for round_id, jobs in sorted(by_round.items()) if expected and jobs == expected
        ]

    async def _progress(self, rows: list[dict[str, Any]]) -> _ScientificProgress:
        jobs = [BenchmarkStageJob.model_validate(row["job"]) for row in rows]
        if any(job.campaign_id != self.manifest.campaign_id for job in jobs):
            raise ValueError("checkpoint_foreign_campaign_job")
        params = {
            "campaign": self.manifest.campaign_id,
            "job_ids": sorted(job.job_id for job in jobs),
            "compiler_runs": sorted(
                {
                    compiler_instance_id(self.manifest, job.parameters)
                    for job in jobs
                    if job.operation == "compiler"
                }
            ),
        }
        facts: dict[str, dict[str, int | str]] = {}
        queries = [(DatabaseRole.RUNTIME, _RUNTIME_PROGRESS_SQL)]
        if any(job.operation == "evaluate" for job in jobs):
            queries.append((DatabaseRole.EVALUATOR, _EVALUATOR_PROGRESS_SQL))
        for role, statement in queries:
            engine, sessions = create_session_factory(self.settings, role)
            try:
                async with sessions() as session:
                    await session.connection(
                        execution_options={"isolation_level": "REPEATABLE READ"}
                    )
                    await session.execute(text("SET TRANSACTION READ ONLY"))
                    await session.execute(text("SET LOCAL statement_timeout = '15s'"))
                    result = await session.execute(statement, params)
                    for row in result.mappings():
                        facts[str(row["family"])] = {
                            "count": int(row["fact_count"]),
                            "digest": str(row["digest"]),
                        }
            finally:
                await engine.dispose()
        return _ScientificProgress(
            receipts={
                job.job_id: row["receipt_hash"]
                for job, row in zip(jobs, rows, strict=True)
                if row["status"] == "completed"
            },
            facts=facts,
        )

    async def observe(
        self, rows: list[dict[str, Any]], *, interval_seconds: int, job_interval: int
    ) -> None:
        """Observe durable scientific progress without acquiring GPU resources."""
        counts: dict[str, int] = {}
        stages: dict[str, list[str]] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            stages.setdefault(row["job"]["stage"], []).append(row["status"])
        finished = {
            stage for stage, values in stages.items() if all(v == "completed" for v in values)
        }
        newly_finished = sorted(finished - self._observed_stages)
        progress = await self._progress(rows)
        barriers = await self._validated_round_barriers(rows)
        milestones = self._complete_round_milestones(rows, barriers)
        marker = self.directory / "checkpoints/last-database-checkpoint.json"
        saved = (
            json.loads(marker.read_bytes()).get("round_milestones", []) if marker.is_file() else []
        )
        fact_hash = (
            canonical_content_hash([progress.fact_hash, barriers])
            if barriers
            else progress.fact_hash
        )
        if set(milestones) - set(saved):
            await self._checkpoint("doubter_round", counts)
        elif newly_finished:
            # One snapshot includes all newly completed stages; do not dump once per stage.
            await self._checkpoint(newly_finished[-1], counts)
        elif (
            progress.has_facts
            and fact_hash != self._fact_hash_at_checkpoint
            and (
                progress.terminal_jobs - self._completed_at_checkpoint >= job_interval
                or time.monotonic() - self._checkpoint_clock >= interval_seconds
            )
        ):
            await self._checkpoint("periodic", counts)

    async def __call__(self, stage: str, counts: dict[str, int]) -> None:
        await self._checkpoint(stage, counts)

    async def _checkpoint(self, stage: str, counts: dict[str, int]) -> None:
        lock_path = self.directory / "checkpoints/.coordinator.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
            try:
                jobs = await PostgresBenchmarkRepository(sessions).list_jobs(
                    self.manifest.campaign_id
                )
            finally:
                await engine.dispose()
            if stage not in {"periodic", "failure", "completed"} and any(
                row["job"]["stage"] == stage and row["status"] != "completed" for row in jobs
            ):
                return
            progress = await self._progress(jobs)
            barriers = await self._validated_round_barriers(jobs)
            milestones = self._complete_round_milestones(jobs, barriers)
            fact_hash = (
                canonical_content_hash([progress.fact_hash, barriers])
                if barriers
                else progress.fact_hash
            )
            marker = self.directory / "checkpoints/last-database-checkpoint.json"
            if not progress.has_facts or (
                marker.is_file() and json.loads(marker.read_text())["fact_hash"] == fact_hash
            ):
                return
            index = self.directory / "checkpoints/database-component-index.json"
            write_json(
                index,
                {
                    "component_receipts": progress.receipts,
                    "counts": counts,
                    "scientific_facts": progress.facts,
                    "fact_hash": fact_hash,
                    "round_barriers": barriers,
                    "round_milestones": milestones,
                },
            )
            checkpoint = await self.service.create(
                "periodic" if stage == "periodic" else "step8:" + stage,
                stage_results_path=index,
                launch_state_path=self.directory / "manifests/campaign.json",
            )
            write_json(
                checkpoint / "step8-component-hashes.json",
                {
                    "protocol_hash": self.manifest.protocol.content_hash,
                    "components": {
                        path.name: sha256_file(path)
                        for path in sorted(
                            (self.directory / "checkpoints/components").glob("*.json")
                        )
                    },
                    "fact_hash": fact_hash,
                },
                immutable=True,
            )
            write_json(
                marker,
                {
                    "fact_hash": fact_hash,
                    "sequence": checkpoint.name,
                    "terminal_jobs": progress.terminal_jobs,
                    "round_barriers": barriers,
                    "round_milestones": milestones,
                },
            )
            self._completed_at_checkpoint = progress.terminal_jobs
            self._fact_hash_at_checkpoint = fact_hash
            self._checkpoint_clock = time.monotonic()
            stages = {row["job"]["stage"] for row in jobs}
            self._observed_stages.update(
                name
                for name in stages
                if all(row["status"] == "completed" for row in jobs if row["job"]["stage"] == name)
            )

    async def restore_empty(self, checkpoint: Path) -> None:
        verified = await asyncio.to_thread(self.service.verify, checkpoint)
        if verified.alembic_revision != STEP8_ALEMBIC_REVISION:
            raise ValueError("step8_checkpoint_schema_incompatible")
        index = json.loads((checkpoint / "step8-component-hashes.json").read_text())
        if index["protocol_hash"] != self.manifest.protocol.content_hash:
            raise ValueError("step8_checkpoint_protocol_mismatch")
        for name, digest in index["components"].items():
            if (
                Path(name).name != name
                or sha256_file(self.directory / "checkpoints/components" / name) != digest
            ):
                raise ValueError("step8_checkpoint_component_changed")
        await self.service.restore(checkpoint, verified_manifest=verified, require_empty=True)
        engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
        try:
            async with sessions.begin() as session:
                # Restore occurs only into an empty service; old workers cannot own this DB.
                await session.execute(
                    text("""UPDATE pmmc_runtime.benchmark_stage_jobs
                    SET status='pending', lease_owner=NULL, lease_expires_at=NULL,
                        attempt_count=attempt_count+1
                    WHERE campaign_id=:campaign AND status='running'"""),
                    {"campaign": self.manifest.campaign_id},
                )
        finally:
            await engine.dispose()
