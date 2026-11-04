"""Recoverable execution loop for the two full-context baselines."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.answering import (
    BaselineAnswerService,
    FullContextAnswerInputProvider,
    baseline_job_id,
    recover_answer_from_trace,
)
from pmmc.baselines.config import FullContextBaselineConfig
from pmmc.baselines.leases import baseline_worker_id
from pmmc.baselines.typed_answer import (
    process_typed_baseline_answer,
    typed_baseline_inventory,
    typed_context_job,
    typed_generation_job_id,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.schemas import (
    AppStage,
    AssetType,
    BaselineMethod,
    BenchmarkQuery,
    DataScope,
    ExperimentJob,
    FinalAnswerRecord,
    JobStatus,
    ModelCallTrace,
    RunStatus,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION

ItemCallback = Callable[[str, bool, str | None], None]


class TransactionalModelCallWriter:
    """Commit each provider trace independently from the long-running worker."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def add(self, trace: ModelCallTrace) -> ModelCallTrace:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            return await uow.model_calls.add(trace)


@dataclass(frozen=True)
class BaselineRunResult:
    run_id: str
    selected: int
    succeeded: int
    failed: int
    already_completed: bool = False
    answered: int | None = None
    abstained: int | None = None


class FullContextBaselineRunner:
    """Claim answer jobs, recover persisted facts, and commit one answer per query."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        input_provider: FullContextAnswerInputProvider,
        answer_service: BaselineAnswerService,
        error_artifacts: ExperimentArtifactService,
        method: BaselineMethod,
        config: FullContextBaselineConfig,
        lease_seconds: int,
        worker_prefix: str = "full-context",
        phase: str = "answer",
    ) -> None:
        if method not in {
            BaselineMethod.FULL_CONTEXT_CAPTIONS,
            BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
        }:
            raise ValueError("full-context runner requires a full-context method")
        if lease_seconds < 3:
            raise ValueError("baseline job lease must be at least three seconds")
        self.sessions = sessions
        self.input_provider = input_provider
        self.answer_service = answer_service
        self.error_artifacts = error_artifacts
        self.method = method
        self.config = config
        self.lease_seconds = lease_seconds
        self.worker_prefix = worker_prefix
        self.phase = phase
        self.job_stage = AppStage.BASELINE if phase == "generate" else AppStage.ANSWER

    @property
    def typed(self) -> bool:
        return (
            getattr(self.input_provider, "answer_contract_version", None)
            or self.config.answer_contract_version
        ) == TYPED_ANSWER_CONTRACT_VERSION

    async def execute(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        selected: list[BenchmarkQuery],
        on_item: ItemCallback | None = None,
    ) -> BaselineRunResult:
        if not selected:
            raise ValueError("baseline run requires at least one selected query")
        queries = {record.query.query_id: record for record in selected}
        if len(queries) != len(selected):
            raise ValueError("selected baseline queries contain duplicate IDs")

        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.runs is not None
            for record in selected:
                job_id = (typed_generation_job_id if self.phase == "generate" else baseline_job_id)(
                    run_id, record.query.query_id, self.method,
                )
                existing = await uow.jobs.get(job_id)
                if existing is None:
                    await uow.jobs.add(
                        ExperimentJob(
                            job_id=job_id,
                            run_id=run_id,
                            snapshot_id=snapshot_id,
                            query_id=record.query.query_id,
                            stage=self.job_stage,
                            variant="primary",
                            max_attempts=self.config.job_max_attempts,
                            metadata={"baseline_method": self.method.value},
                        )
                    )
                elif (
                    existing.run_id != run_id
                    or existing.snapshot_id != snapshot_id
                    or existing.query_id != record.query.query_id
                    or existing.stage is not self.job_stage
                    or existing.variant != "primary"
                    or existing.max_attempts != self.config.job_max_attempts
                    or existing.metadata.get("baseline_method") != self.method.value
                ):
                    raise ValueError(f"persisted baseline job conflicts with run config: {job_id}")
            state = await uow.runs.get_state(run_id)
            if state.status is RunStatus.CREATED:
                await uow.runs.transition(run_id, RunStatus.RUNNING)
            elif state.status is RunStatus.COMPLETED:
                if self.typed:
                    inventory = await typed_baseline_inventory(
                        uow.query_executor, run_id=run_id, selected=selected, method=self.method,
                    )
                    if inventory["missing"]:
                        raise ValueError("completed typed baseline is missing scientific terminals")
                return BaselineRunResult(
                    run_id,
                    len(selected),
                    len(selected),
                    0,
                    already_completed=True,
                    answered=inventory["answered"] if self.typed else None,
                    abstained=inventory["abstained"] if self.typed else None,
                )
            elif state.status is not RunStatus.RUNNING:
                raise RuntimeError(f"run cannot be resumed from state: {state.status.value}")

        async def worker(index: int) -> None:
            worker_id = baseline_worker_id(run_id, self.worker_prefix, index)
            while True:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    job = await uow.jobs.claim(
                        worker_id,
                        self.lease_seconds,
                        run_id=run_id,
                        stage=self.job_stage,
                    )
                    counts = (await uow.jobs.status_counts(run_id, stage=self.job_stage)
                              if job is None else None)
                if job is None:
                    assert counts is not None
                    unfinished = counts.get(JobStatus.PENDING, 0) + counts.get(
                        JobStatus.RUNNING, 0
                    )
                    if unfinished:
                        await asyncio.sleep(0.1)
                        continue
                    return
                record = queries.get(job.query_id or "")
                if record is None:
                    failed_job = await self._finish_failed(
                        job, worker_id, KeyError("query assignment missing")
                    )
                    if on_item and failed_job.status is JobStatus.FAILED:
                        on_item(job.query_id or "unknown", False, "KeyError")
                    continue
                try:
                    await self._process(job, worker_id, record)
                except Exception as exc:
                    failed_job = await self._finish_failed(job, worker_id, exc)
                    if on_item and failed_job.status is JobStatus.FAILED:
                        on_item(record.query.query_id, False, type(exc).__name__)
                else:
                    if on_item:
                        on_item(record.query.query_id, True, None)

        await asyncio.gather(
            *(worker(index) for index in range(self.config.concurrency))
        )
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.runs is not None
            counts = await uow.jobs.status_counts(run_id, stage=self.job_stage)
            succeeded = counts.get(JobStatus.SUCCEEDED, 0)
            failed = counts.get(JobStatus.FAILED, 0)
            terminal_inventory = None
            if self.typed and self.phase != "generate":
                terminal_inventory = await typed_baseline_inventory(
                    uow.query_executor, run_id=run_id, selected=selected, method=self.method,
                )
                if terminal_inventory["answered"] + terminal_inventory["abstained"] != succeeded:
                    raise ValueError("typed terminal and succeeded-job counts disagree")
            if failed or succeeded != len(selected):
                await uow.runs.transition(
                    run_id,
                    RunStatus.FAILED,
                    error_type="BaselineJobFailure",
                    error_summary="one or more full-context jobs exhausted their attempts",
                )
        result = BaselineRunResult(
            run_id, len(selected), succeeded, failed,
            answered=terminal_inventory["answered"] if terminal_inventory else None,
            abstained=terminal_inventory["abstained"] if terminal_inventory else None,
        )
        if failed:
            raise RuntimeError(f"{failed} full-context baseline job(s) failed")
        return result

    async def _process(
        self,
        job: ExperimentJob,
        worker_id: str,
        record: BenchmarkQuery,
    ) -> None:
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job, worker_id, stop))
        try:
            context_job = job
            if self.typed and self.phase != "generate":
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    context_job = await typed_context_job(uow, job, self.method)
            prepared = await self.input_provider.prepare(
                run_id=job.run_id,
                snapshot_id=job.snapshot_id,
                query=record.query,
                method=self.method,
                job_id=context_job.job_id,
                created_at=context_job.created_at,
            )
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.contexts is not None and uow.jobs is not None
                await uow.jobs.heartbeat(
                    job.job_id, worker_id, self.lease_seconds,
                    expected_attempt_count=job.attempt_count,
                )
                await uow.contexts.add(prepared.context_assembly)

            if prepared.metadata.get("answer_contract_version") == TYPED_ANSWER_CONTRACT_VERSION:
                await process_typed_baseline_answer(
                    prepared=prepared, job=job, worker_id=worker_id, method=self.method,
                    answer_service=self.answer_service,
                    uow_factory=lambda: ExperimentUnitOfWork(self.sessions),
                    phase=self.phase,
                )
                return

            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.answers is not None
                existing = await uow.answers.get_for_query(
                    job.run_id,
                    record.query.query_id,
                )
            if existing is None:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    successful_call = await uow.model_calls.get_successful_for_job(job.job_id)
                if successful_call is not None:
                    answer = recover_answer_from_trace(
                        prepared,
                        job,
                        successful_call.parsed_output,
                        actual_model_id=successful_call.actual_model_id,
                        model_call_id=successful_call.model_call_id,
                        input_tokens=successful_call.input_tokens,
                        output_tokens=successful_call.output_tokens,
                        latency_ms=successful_call.latency_ms,
                        langfuse_trace_id=successful_call.langfuse_trace_id,
                    )
                else:
                    generated = await self.answer_service.answer(prepared, job)
                    if not isinstance(generated, FinalAnswerRecord):
                        raise ValueError("legacy baseline job returned a typed answer")
                    answer = generated
            else:
                answer = existing

            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.answers is not None and uow.jobs is not None
                await uow.jobs.finish(
                    job.job_id, worker_id, success=True,
                    expected_attempt_count=job.attempt_count,
                )
                await uow.answers.add(answer)
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, job: ExperimentJob, worker_id: str, stop: asyncio.Event) -> None:
        interval = self.lease_seconds / 3
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    await uow.jobs.heartbeat(
                        job.job_id, worker_id, self.lease_seconds,
                        expected_attempt_count=job.attempt_count,
                    )

    async def _finish_failed(
        self,
        job: ExperimentJob,
        worker_id: str,
        error: BaseException,
    ) -> ExperimentJob:
        payload = json.dumps(
            {"error_type": type(error).__name__, "message": "baseline job failed"},
            sort_keys=True,
        ).encode("utf-8")
        artifact_id: str | None = None
        try:
            artifact = await self.error_artifacts.put_bytes(
                artifact_id=f"{job.job_id}:attempt:{job.attempt_count}:error",
                run_id=job.run_id,
                snapshot_id=job.snapshot_id,
                query_id=job.query_id,
                artifact_type=AssetType.OTHER,
                data=payload,
                mime_type="application/json",
                extension="json",
            )
            artifact_id = artifact.artifact_id
        except Exception:
            artifact_id = None
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None
            return await uow.jobs.finish(
                job.job_id,
                worker_id,
                success=False,
                error_type=type(error).__name__,
                error_summary="baseline job failed; inspect the runtime error artifact",
                error_artifact_id=artifact_id,
                retry_delay_seconds=self.config.job_retry_delay_seconds,
                expected_attempt_count=job.attempt_count,
            )
