"""Bounded campaign workers with fenced completion and durable recovery receipts."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pmmc.experiments.step8_artifacts import (
    CampaignArtifactStore,
    json_bytes,
    sha256_file,
    write_json,
)
from pmmc.experiments.step8_failure import (
    failure_diagnostics,
    failure_frames,
    safe_failure_code,
    scientific_failure_state,
    write_failure_capsule,
)
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_telemetry import ResourceSampler
from pmmc.infrastructure.database.work_limit import DatabaseWorkLimit, database_work_scope
from pmmc.models.campaign_audit import CampaignModelAudit, campaign_model_scope
from pmmc.schemas.base import JsonDict
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, ResourceClass, Step8ExecutionConfig


def safe_failure_record(error: BaseException, *, stage: str, locator: str) -> dict[str, object]:
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    failure = {
        "stage": stage,
        "locator": locator,
        "error_code": safe_failure_code(error),
        "unknown_provider_outcome": getattr(error, "outcome", None) == "unknown",
        "affected_stages": getattr(error, "affected_stages", None),
        "diagnostics": failure_diagnostics(error),
        "frames": failure_frames(error),
    }
    return {**failure, **scientific_failure_state(failure)}


def retain_failure_capsule(
    error: BaseException,
    *,
    directory: Path,
    attempt: Path,
    failure: Mapping[str, object],
) -> None:
    """Retain safe recovery facts without replacing the exception being handled."""
    if getattr(error, "_step8_capsule_recorded", False):
        return
    try:
        attempt.mkdir(parents=True, exist_ok=True)
        write_failure_capsule(directory, attempt, failure)
    except Exception as secondary:
        error.add_note("failure capsule persistence failed: " + type(secondary).__name__)
    else:
        error.__dict__["_step8_capsule_recorded"] = True


@contextlib.contextmanager
def campaign_failure_boundary(directory: Path, *, stage: str) -> Iterator[None]:
    """Cover failures before a worker exists, including configuration and startup."""
    try:
        yield
    except BaseException as error:
        record_startup_failure(error, directory=directory, stage=stage)
        raise


def record_startup_failure(error: BaseException, *, directory: Path, stage: str) -> None:
    attempt_id = "startup-" + uuid.uuid4().hex
    retain_failure_capsule(
        error,
        directory=directory,
        attempt=directory / "attempts" / attempt_id,
        failure=safe_failure_record(error, stage=stage, locator=attempt_id[:12]),
    )


class CampaignRepository(Protocol):
    async def start_attempt(
        self,
        *,
        attempt_id: str,
        campaign_id: str,
        execution_image: str,
        resource_class: str,
        started_at: datetime,
    ) -> None: ...
    async def finish_attempt(self, attempt_id: str, state: str, receipt_hash: str) -> None: ...
    async def claim(
        self, campaign_id: str, resource: ResourceClass, owner: str, lease_seconds: int
    ) -> tuple[BenchmarkStageJob, int] | None: ...
    async def heartbeat(self, job_id: str, owner: str, fence: int, lease_seconds: int) -> None: ...
    async def upstream_receipts(self, job: BenchmarkStageJob) -> dict[str, str]: ...
    async def complete(self, receipt: ComponentReceipt, owner: str, fence: int) -> None: ...
    async def fail(
        self, job_id: str, owner: str, fence: int, code: str, *, unknown: bool
    ) -> None: ...
    async def counts(self, campaign_id: str) -> dict[str, int]: ...


class ComponentBackend(Protocol):
    async def preflight(self, job: BenchmarkStageJob) -> None: ...
    async def execute(
        self, job: BenchmarkStageJob, upstream: dict[str, str]
    ) -> tuple[JsonDict, list[str], dict[str, int]]: ...


class CampaignWorker:
    def __init__(
        self,
        *,
        repository: CampaignRepository,
        backend: ComponentBackend,
        campaign_id: str,
        campaign_dir: Path,
        resource: ResourceClass,
        execution: Step8ExecutionConfig,
        execution_image: str,
        artifact_store: CampaignArtifactStore,
        event: Callable[[str], None] = print,
        checkpoint: Callable[[str, dict[str, int]], Awaitable[None]] | None = None,
        attempt_id: str | None = None,
        stages: tuple[str, ...] = (),
        execution_resource: ResourceClass | None = None,
    ) -> None:
        self.repository, self.backend = repository, backend
        self.campaign_id, self.campaign_dir = campaign_id, campaign_dir
        self.resource, self.execution = resource, execution
        # Claims keep the immutable logical job resource. Import-only workers
        # execute on CPU and must not report a physical GPU reservation.
        self.execution_resource = execution_resource or resource
        self.execution_image, self.store = execution_image, artifact_store
        self.event, self.checkpoint = event, checkpoint
        self.attempt_id = attempt_id or "attempt-" + uuid.uuid4().hex
        self.owner_prefix = f"{socket.gethostname()[:32]}:{os.getpid()}:{self.attempt_id}"
        self.attempt_dir = campaign_dir / "attempts" / self.attempt_id
        self.completed, self.reused = 0, 0
        self.last_checkpoint_count = 0
        self.last_checkpoint_at = time.monotonic()
        self._checkpoint_lock = asyncio.Lock()
        self.first_failure: dict[str, object] | None = None
        self.sampler = ResourceSampler()
        self.stages = stages
        self.database_limit = DatabaseWorkLimit(
            execution.database_concurrency,
            shared_root=self.store.root / "_store/coordination/database",
        )
        self.request_semaphore = asyncio.Semaphore(
            execution.api_concurrency
            if resource is ResourceClass.API
            else execution.model_concurrency
        )

    def _emit(self, name: str, **fields: object) -> None:
        # Values here are counts, protocol identifiers and enums, never provider payloads.
        line = name + " " + json.dumps(fields, sort_keys=True, separators=(",", ":"))
        self.event(line)
        with (self.attempt_dir / "safe-execution.log").open("a", encoding="utf-8") as stream:
            stream.write(datetime.now(UTC).isoformat() + " " + line + "\n")
            stream.flush()

    def _record_failure(self, error: BaseException, *, stage: str, locator: str) -> None:
        if self.first_failure is not None:
            return
        self.first_failure = {
            **safe_failure_record(error, stage=stage, locator=locator),
            "completed": self.completed,
            "resumed": self.reused,
        }
        try:
            self._emit("STEP8_FAILURE", **self.first_failure)
        except Exception as secondary:
            error.add_note("failure logging failed: " + type(secondary).__name__)

    async def _lease(self, job: BenchmarkStageJob, owner: str, fence: int) -> None:
        while True:
            await asyncio.sleep(
                min(self.execution.heartbeat_seconds, self.execution.lease_seconds / 3)
            )
            await self.repository.heartbeat(job.job_id, owner, fence, self.execution.lease_seconds)

    async def _observe(self) -> None:
        while True:
            await asyncio.sleep(self.execution.heartbeat_seconds)
            started = time.monotonic()
            counts = await self.repository.counts(self.campaign_id)
            database_observation_seconds = time.monotonic() - started
            resources = await asyncio.to_thread(
                self.sampler.sample,
                gpu=self.execution_resource
                in {
                    ResourceClass.GPU_9B,
                    ResourceClass.GPU_27B,
                    ResourceClass.GPU_EMBEDDING,
                },
            )
            self._emit(
                "STEP8_PROGRESS",
                resource=self.execution_resource.value,
                logical_resource=self.resource.value,
                completed=counts.get("completed", 0),
                pending=counts.get("pending", 0),
                inflight=counts.get("running", 0),
                failed=counts.get("failed", 0),
                unknown=counts.get("unknown", 0),
                reused=self.reused,
                denominator=sum(counts.values()),
                eta_seconds=None,
                database_observation_seconds=database_observation_seconds,
                resources=resources,
                db_inflight=self.database_limit.inflight,
                db_peak=self.database_limit.peak,
                db_wait_seconds=self.database_limit.wait_seconds,
                db_limit_scope="shared_output_root_workers",
            )

    async def _maybe_checkpoint(self, job: BenchmarkStageJob) -> None:
        if self.checkpoint is None:
            return
        async with self._checkpoint_lock:
            changed = self.completed > self.last_checkpoint_count
            due = (
                self.completed - self.last_checkpoint_count >= self.execution.checkpoint_jobs
                or time.monotonic() - self.last_checkpoint_at >= self.execution.checkpoint_seconds
            )
            major = job.stage in {"r0_gate", "projection", "grounding", "evaluation"}
            if not changed or not (due or major):
                return
            counts = await self.repository.counts(self.campaign_id)
            await self.checkpoint(job.stage if major else "periodic", counts)
            self.last_checkpoint_count = self.completed
            self.last_checkpoint_at = time.monotonic()

    async def _one(self, ordinal: int) -> None:
        owner = f"{self.owner_prefix}:{ordinal}"
        while True:
            claimed = await self.repository.claim(
                self.campaign_id,
                self.resource,
                owner,
                self.execution.lease_seconds,
            )
            if claimed is None:
                return
            job, fence = claimed
            lease = asyncio.create_task(self._lease(job, owner, fence))
            work: asyncio.Task[tuple[JsonDict, list[str], dict[str, int]]] | None = None
            primary: BaseException | None = None
            started = time.monotonic()
            try:
                upstream = await self.repository.upstream_receipts(job)
                await self.backend.preflight(job)
                recovery_path = self.campaign_dir / "checkpoints/components" / f"{job.job_id}.json"
                if recovery_path.is_file():
                    receipt = ComponentReceipt.model_validate_json(recovery_path.read_text())
                    if (
                        receipt.job_id != job.job_id
                        or receipt.scientific_input_hash != job.scientific_input_hash
                        or receipt.source_fingerprint != job.source_fingerprint
                        or receipt.upstream_receipt_hashes != upstream
                        or receipt.schema_revision != STEP8_ALEMBIC_REVISION
                    ):
                        raise ValueError("incompatible_component_recovery_receipt")
                    for digest in receipt.artifact_hashes:
                        self.store.verify(digest)
                    if "maintenance_execution" in receipt.output:
                        maintenance_hook = getattr(self.backend, "execution_maintenance", None)
                        maintenance = maintenance_hook(job) if maintenance_hook else None
                        bound = receipt.output["maintenance_execution"]
                        if (
                            maintenance is None or bound != maintenance
                            or maintenance["proof"]["execution_image"] != self.execution_image
                            or maintenance["sha256"] not in receipt.artifact_hashes
                            or self.store.load_json(maintenance["sha256"]) != maintenance["proof"]
                        ):
                            raise ValueError("incompatible_maintenance_recovery_receipt")
                    verify_recovery = getattr(self.backend, "verify_recovery", None)
                    if verify_recovery is not None:
                        backend_receipt = receipt
                        if "maintenance_execution" in receipt.output:
                            # The backend verifies its own output, not the worker's envelope.
                            output_digest = hashlib.sha256(json_bytes(receipt.output)).hexdigest()
                            backend_receipt = receipt.model_copy(update={
                                "output": {k: v for k, v in receipt.output.items()
                                           if k != "maintenance_execution"},
                                "artifact_hashes": [d for d in receipt.artifact_hashes
                                                    if d not in {bound["sha256"], output_digest}],
                            })
                        await verify_recovery(job, upstream, backend_receipt)
                    self.reused += 1
                else:
                    maintenance_hook = getattr(self.backend, "execution_maintenance", None)
                    maintenance = maintenance_hook(job) if maintenance_hook else None
                    if (
                        maintenance is not None
                        and maintenance["proof"]["execution_image"] != self.execution_image
                    ):
                        raise ValueError("maintenance_worker_image_mismatch")
                    with campaign_model_scope(
                        CampaignModelAudit(
                            self.store,
                            self.campaign_id,
                            self.attempt_id,
                            job.job_id,
                            job.source_fingerprint,
                            self.request_semaphore,
                        )
                    ):
                        work = asyncio.create_task(self.backend.execute(job, upstream))
                    done, _ = await asyncio.wait({work, lease}, return_when=asyncio.FIRST_COMPLETED)
                    if lease in done:
                        lease.result()  # Propagate lease loss before an unfenced commit.
                        raise RuntimeError("lease_heartbeat_stopped")
                    output, hashes, counts = work.result()
                    if maintenance_hook and maintenance_hook(job) != maintenance:
                        raise ValueError("maintenance_execution_changed_during_work")
                    if "maintenance_execution" in output:
                        raise ValueError("maintenance_execution_output_already_owned")
                    if maintenance is not None:
                        digest = self.store.put_json(maintenance["proof"])
                        if digest != maintenance["sha256"]:
                            raise ValueError("maintenance_execution_proof_hash_mismatch")
                        output = {**output, "maintenance_execution": maintenance}
                        output_digest = self.store.put_json(output)
                        hashes = list(dict.fromkeys([*hashes, digest, output_digest]))
                    for digest in hashes:
                        self.store.verify(digest)
                    receipt = ComponentReceipt(
                        job_id=job.job_id,
                        scientific_input_hash=job.scientific_input_hash,
                        source_fingerprint=job.source_fingerprint,
                        schema_revision=STEP8_ALEMBIC_REVISION,
                        upstream_receipt_hashes=upstream,
                        output=output,
                        artifact_hashes=hashes,
                        row_counts=counts,
                    )
                    # Persist the successful component before the authoritative DB commit.
                    write_json(recovery_path, receipt.model_dump(mode="json"), immutable=True)
                await self.repository.complete(receipt, owner, fence)
                self.completed += 1
                self._emit(
                    "STEP8_COMPONENT",
                    stage=job.stage,
                    state="completed",
                    locator=job.job_id[:12],
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    reused=recovery_path.is_file() and work is None,
                    artifact_count=len(receipt.artifact_hashes),
                    row_counts=receipt.row_counts,
                )
                await self._maybe_checkpoint(job)
            except BaseException as exc:
                primary = exc
                if not isinstance(exc, Exception):
                    raise
                code = safe_failure_code(exc)
                unknown = getattr(exc, "outcome", None) == "unknown"
                with contextlib.suppress(Exception):
                    await self.repository.fail(job.job_id, owner, fence, code, unknown=unknown)
                self._record_failure(exc, stage=job.stage, locator=job.job_id[:12])
                raise
            finally:
                lease.cancel()
                if work is not None and not work.done():
                    work.cancel()
                    try:
                        await work
                    except asyncio.CancelledError:
                        pass
                    except Exception as secondary:
                        if primary is None:
                            raise
                        primary.add_note(
                            "component cancellation failed: " + type(secondary).__name__
                        )
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await lease

    async def run(self) -> dict[str, object]:
        with contextlib.ExitStack() as stack:
            from pmmc.experiments.step8_scratch import attempt_scratch
            from pmmc.infrastructure.settings import Settings

            settings = getattr(self.backend, "settings", None)
            if isinstance(settings, Settings):
                owned = stack.enter_context(attempt_scratch(
                    settings, attempt_id=self.attempt_id, campaign_dir=self.campaign_dir
                ))
                self.backend.settings = owned  # type: ignore[attr-defined]
            stack.enter_context(database_work_scope(self.database_limit))
            return await self._run()

    async def _run(self) -> dict[str, object]:
        observer: asyncio.Task[None] | None = None
        primary: BaseException | None = None
        result: dict[str, object] | None = None
        state = "failed"
        registered = False

        def fail(error: BaseException, stage: str) -> None:
            nonlocal primary, state
            if primary is None:
                primary = error
                state = "interrupted" if isinstance(error, asyncio.CancelledError) else "failed"
                self._record_failure(error, stage=stage, locator=self.attempt_id[:12])
            else:
                primary.add_note(stage + " failed: " + type(error).__name__)

        def capsule() -> None:
            if primary is not None and self.first_failure is not None:
                retain_failure_capsule(
                    primary,
                    directory=self.campaign_dir,
                    attempt=self.attempt_dir,
                    failure=self.first_failure,
                )

        try:
            self.attempt_dir.mkdir(parents=True, exist_ok=False)
            self.store.attempt_cache(self.attempt_id)
            started_at = datetime.now(UTC)
            write_json(
                self.attempt_dir / "attempt.json",
                {
                    "attempt_id": self.attempt_id,
                    "campaign_id": self.campaign_id,
                    "execution_image": self.execution_image,
                    "resource_class": self.execution_resource.value,
                    "logical_resource_class": self.resource.value,
                    "execution_profile_hash": getattr(self.backend, "execution_profile_hash", None),
                    "started_at": started_at.isoformat(),
                },
                immutable=True,
            )
            observer = asyncio.create_task(self._observe())
            await self.repository.start_attempt(
                attempt_id=self.attempt_id,
                campaign_id=self.campaign_id,
                execution_image=self.execution_image,
                resource_class=self.execution_resource.value,
                started_at=started_at,
            )
            registered = True
            self._emit(
                "STEP8_ATTEMPT", state="started", resource=self.execution_resource.value,
                logical_resource=self.resource.value,
            )
            # Model pools batch their own requests; one stage owner avoids duplicate startups.
            # CPU components already own bounded 32-worker pools internally. Running
            # 32 such components would multiply processes and DB pools to 1024.
            # Per-query answer/grounding/evaluation stages do not start nested CPU
            # pools. Fill the existing model pool across queries without multiplying
            # Questioner/dry-run worker pools. One process-wide limiter bounds calls.
            query_only = self.stages and set(self.stages) <= {"answers", "grounding", "evaluation"}
            workers = (
                min(self.execution.api_concurrency, 8)
                if (query_only or self.resource is ResourceClass.API)
                else 1
            )
            self._emit(
                "STEP8_WORKERS",
                component_workers=workers,
                database_work_limit=self.database_limit.maximum,
                model_request_limit=(
                    self.execution.api_concurrency
                    if self.resource is ResourceClass.API
                    else self.execution.model_concurrency
                ),
            )
            async with asyncio.TaskGroup() as group:
                for index in range(workers):
                    group.create_task(self._one(index))
            counts = await self.repository.counts(self.campaign_id)
            state = "completed"
            result = {
                "attempt_id": self.attempt_id,
                "worker_complete": True,
                "campaign_complete": bool(counts)
                and counts.get("completed", 0) == sum(counts.values()),
                "completed_this_attempt": self.completed,
                "reused": self.reused,
                "counts": counts,
            }
        except BaseException as exc:
            fail(exc, "worker")
        finally:
            if observer is not None:
                observer.cancel()
                try:
                    await observer
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    fail(exc, "observer")
            checkpoint_error = None
            if self.checkpoint is not None:
                try:
                    counts = await self.repository.counts(self.campaign_id)
                    campaign_complete = bool(counts) and counts.get("completed", 0) == sum(
                        counts.values()
                    )
                    if campaign_complete or state != "completed":
                        await self.checkpoint(
                            "completed" if campaign_complete else "failure", counts
                        )
                except BaseException as exc:
                    checkpoint_error = type(exc).__name__
                    fail(exc, "checkpoint")
            receipt_path = self.attempt_dir / "terminal.json"
            capsule()
            receipt_written = False
            try:
                write_json(
                    receipt_path,
                    {
                        "attempt_id": self.attempt_id,
                        "state": state,
                        "completed": self.completed,
                        "reused": self.reused,
                        "first_failure": self.first_failure,
                        "checkpoint_error": checkpoint_error,
                    },
                    immutable=True,
                )
                receipt_written = True
            except BaseException as exc:
                fail(exc, "terminal_receipt")
                capsule()
            if registered and receipt_written:
                try:
                    await self.repository.finish_attempt(
                        self.attempt_id, state, sha256_file(receipt_path)
                    )
                except BaseException as exc:
                    fail(exc, "attempt_commit")
                    capsule()
            if receipt_written and (
                primary is None or getattr(primary, "_step8_capsule_recorded", False)
            ):
                try:
                    self.store.clean_attempt(self.attempt_id, durable_receipt=receipt_path)
                except BaseException as exc:
                    fail(exc, "scratch_cleanup")
                    capsule()
        if primary is not None:
            raise primary
        assert result is not None
        return result
