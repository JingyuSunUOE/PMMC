"""Recoverable retrieval and answer workers for the three RAG baselines."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import suppress

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.answering import BaselineAnswerService, recover_answer_from_trace
from pmmc.baselines.leases import baseline_worker_id
from pmmc.baselines.rag_config import RagBaselineConfig
from pmmc.baselines.rag_context import RagAnswerInputProvider
from pmmc.baselines.rag_embedding import (
    RAG_METHODS,
    produce_query_embeddings,
    restore_query_embedding_bundle,
)
from pmmc.baselines.rag_retrieval import retrieval_batch_id, search_query_windows
from pmmc.baselines.typed_answer import (
    process_typed_baseline_answer,
    typed_baseline_inventory,
    typed_context_job,
    typed_generation_job_id,
)
from pmmc.benchmarks.ids import stable_hash
from pmmc.infrastructure.database.errors import safe_database_error_diagnostics
from pmmc.infrastructure.database.experiment_models import ExperimentJobModel
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.models.execution import ModelExecutionService
from pmmc.models.protocols import EmbeddingGateway
from pmmc.runtime.answering import InvalidRuntimeAnswer, UnsupportedRuntimeAnswer
from pmmc.schemas import (
    AppStage,
    AssetType,
    BaselineMethod,
    BenchmarkQuery,
    EmbeddingCollection,
    EmbeddingCollectionStatus,
    ExperimentJob,
    FinalAnswerRecord,
    JobStatus,
    RunStatus,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION, TypedAnswerResult

ItemCallback = Callable[[str, bool, str | None], None]


class RagAnswerOperationError(RuntimeError):
    """Keep the failing operation without publishing the underlying message."""

    def __init__(self, operation: str, error: Exception) -> None:
        self.operation = operation
        self.error = error
        super().__init__("RAG answer operation failed")


def _rag_failure_diagnostics(error: BaseException) -> dict[str, object]:
    operation = "query_job"
    if isinstance(error, RagAnswerOperationError):
        operation, error = error.operation, error.error
    payload: dict[str, object] = {
        "error_type": type(error).__name__,
        "operation": operation,
        "failure_kind": "execution_error",
    }
    traceback = error.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("pmmc."):
            payload["failure_origin"] = {
                "module": module,
                "function": traceback.tb_frame.f_code.co_name,
                "line": traceback.tb_lineno,
            }
        traceback = traceback.tb_next
    if isinstance(error, InvalidRuntimeAnswer):
        payload.update(
            failure_kind="answer_contract_failure",
            contract_failure_codes=[item.value for item in error.failures],
            format_attempt_failure_codes=[
                [item.value for item in values] for values in error.attempt_failures
            ],
            format_attempt_schema_diagnostics=list(error.attempt_schema_diagnostics),
        )
    elif isinstance(error, UnsupportedRuntimeAnswer):
        payload["failure_kind"] = "evidence_abstention"
        if error.output is not None and error.output.abstain_reason is not None:
            payload["abstain_reason"] = error.output.abstain_reason.value
    elif isinstance(error, ValidationError):
        # Locations may contain provider-controlled dictionary keys. Keep only
        # Pydantic error types/counts; never input values, context or messages.
        payload["validation_error_types"] = dict(Counter(
            item["type"] for item in error.errors(include_input=False, include_context=False)
        ))
    database = safe_database_error_diagnostics(error)
    for key in (
        "sqlstate", "database_error_type", "constraint_name", "table_name",
        "os_error_name", "provider_status_code", "provider_error_code", "error_code",
    ):
        value = database.get(key)
        if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_]{1,128}", value):
            payload[key] = value
    return payload


class RagAnswerBatchFailure(RuntimeError):
    """Carry aggregate safe job diagnostics across the managed child boundary."""

    def __init__(self, diagnostics: dict[str, object]) -> None:
        self.diagnostics = diagnostics
        super().__init__("RAG answer jobs failed; inspect RAG_JOB_FAILURE diagnostics")


class RagAnswerCompletionFailure(RagAnswerBatchFailure):
    """A run-level failure must not be reported as failed answer items."""

    def __init__(self, diagnostics: dict[str, object]) -> None:
        self.diagnostics = diagnostics
        RuntimeError.__init__(self, "RAG run completion failed; answer item counts are unchanged")


def rag_job_id(
    run_id: str, query_id: str, method: BaselineMethod, stage: AppStage
) -> str:
    if stage is AppStage.BASELINE:
        return typed_generation_job_id(run_id, query_id, method)
    if stage not in {AppStage.RETRIEVE, AppStage.ANSWER}:
        raise ValueError("RAG query jobs must be retrieve or answer jobs")
    return "job-rag-" + stable_hash([run_id, query_id, method.value, stage.value])[:24]


def _immutable_job(value: ExperimentJob) -> dict[str, object]:
    return value.model_dump(
        mode="json",
        exclude={
            "status",
            "attempt_count",
            "claimed_by",
            "lease_expires_at",
            "available_at",
            "created_at",
            "started_at",
            "finished_at",
            "error_type",
            "error_summary",
            "error_artifact_id",
        },
    )


class _RagJobRunner:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        error_artifacts: ExperimentArtifactService,
        method: BaselineMethod,
        config: RagBaselineConfig,
        lease_seconds: int,
        stage: AppStage,
    ) -> None:
        if method not in RAG_METHODS:
            raise ValueError("RAG runner requires one of the three RAG methods")
        if lease_seconds < 3:
            raise ValueError("RAG job lease must be at least three seconds")
        self.sessions = sessions
        self.error_artifacts = error_artifacts
        self.method = method
        self.config = config
        self.lease_seconds = lease_seconds
        self.stage = stage
        self._terminal_failures: dict[str, dict[str, object]] = {}

    async def _ensure_jobs(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        selected: list[BenchmarkQuery],
    ) -> None:
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None and uow.runs is not None
            for record in selected:
                expected = ExperimentJob(
                    job_id=rag_job_id(
                        run_id, record.query.query_id, self.method, self.stage
                    ),
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    query_id=record.query.query_id,
                    stage=self.stage,
                    variant=("retrieval" if self.stage is AppStage.RETRIEVE else "primary"),
                    max_attempts=self.config.job_max_attempts,
                    metadata={"baseline_method": self.method.value},
                )
                stored = await uow.jobs.get(expected.job_id)
                if stored is None:
                    await uow.jobs.add(expected)
                else:
                    if stored.metadata.get("retrieval_reuse") is not None:
                        from pmmc.baselines.answer_upgrade import verify_rebound_retrieval_job

                        assert uow.session is not None
                        await verify_rebound_retrieval_job(uow.session, stored)
                        expected = expected.model_copy(update={"metadata": stored.metadata})
                    if _immutable_job(stored) != _immutable_job(expected):
                        raise ValueError(f"persisted RAG job conflicts with run: {expected.job_id}")
            state = await uow.runs.get_state(run_id)
            if state.status is RunStatus.CREATED:
                await uow.runs.transition(run_id, RunStatus.RUNNING)
            elif state.status is not RunStatus.RUNNING:
                raise RuntimeError(f"RAG run cannot execute from state: {state.status.value}")
            assert uow.session is not None
            persisted_query_ids = list(
                (
                    await uow.session.scalars(
                        select(ExperimentJobModel.query_id).where(
                            ExperimentJobModel.run_id == run_id,
                            ExperimentJobModel.stage == self.stage.value,
                        )
                    )
                ).all()
            )
            expected_query_ids = {record.query.query_id for record in selected}
            actual_query_ids = {
                value for value in persisted_query_ids if isinstance(value, str) and value
            }
            missing_count = len(expected_query_ids - actual_query_ids)
            unexpected_count = len(actual_query_ids - expected_query_ids) + sum(
                1 for value in persisted_query_ids if not isinstance(value, str) or not value
            )
            if missing_count or unexpected_count:
                print(
                    "RAG_JOB_SCOPE_PREFLIGHT "
                    + json.dumps(
                        {
                            "expected_count": len(expected_query_ids),
                            "missing_count": missing_count,
                            "persisted_count": len(persisted_query_ids),
                            "stage": self.stage.value,
                            "state": "rejected",
                            "unexpected_count": unexpected_count,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                raise RuntimeError("persisted RAG jobs differ from the frozen query denominator")
            print(
                "RAG_JOB_SCOPE_PREFLIGHT "
                + json.dumps(
                    {
                        "expected_count": len(expected_query_ids),
                        "persisted_count": len(persisted_query_ids),
                        "stage": self.stage.value,
                        "state": "ready",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    async def _claim_loop(
        self,
        *,
        run_id: str,
        records: dict[str, BenchmarkQuery],
        concurrency: int,
        process: Callable[[ExperimentJob, str, BenchmarkQuery], Awaitable[None]],
        on_item: ItemCallback | None,
    ) -> tuple[int, int]:
        self._terminal_failures.clear()

        async def worker(index: int) -> None:
            worker_id = baseline_worker_id(run_id, f"rag-{self.stage.value}", index)
            while True:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    job = await uow.jobs.claim(
                        worker_id,
                        self.lease_seconds,
                        run_id=run_id,
                        stage=self.stage,
                    )
                    counts = (
                        await uow.jobs.status_counts(run_id, stage=self.stage)
                        if job is None
                        else None
                    )
                if job is None:
                    assert counts is not None
                    if counts.get(JobStatus.PENDING, 0) + counts.get(JobStatus.RUNNING, 0):
                        await asyncio.sleep(0.1)
                        continue
                    return
                record = records.get(job.query_id or "")
                if record is None:
                    terminal = await self._finish_failed(
                        job, worker_id, KeyError("query assignment missing")
                    )
                    if terminal.status is JobStatus.FAILED:
                        print(
                            "RAG_JOB_SCOPE_MISMATCH "
                            + json.dumps(
                                {
                                    "stage": self.stage.value,
                                    "state": "failed",
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                    continue
                try:
                    await process(job, worker_id, record)
                except Exception as exc:
                    terminal = await self._finish_failed(job, worker_id, exc)
                    if on_item and terminal.status is JobStatus.FAILED:
                        on_item(record.query.query_id, False,
                                str(_rag_failure_diagnostics(exc)["error_type"]))
                else:
                    if on_item:
                        on_item(record.query.query_id, True, None)

        await asyncio.gather(*(worker(index) for index in range(concurrency)))
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.jobs is not None
            counts = await uow.jobs.status_counts(run_id, stage=self.stage)
        return counts.get(JobStatus.SUCCEEDED, 0), counts.get(JobStatus.FAILED, 0)

    async def _heartbeat(self, job: ExperimentJob, worker_id: str, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.lease_seconds / 3)
            except TimeoutError:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    await uow.jobs.heartbeat(
                        job.job_id, worker_id, self.lease_seconds,
                        expected_attempt_count=job.attempt_count,
                    )

    async def _finish_failed(
        self, job: ExperimentJob, worker_id: str, error: BaseException
    ) -> ExperimentJob:
        diagnostics = _rag_failure_diagnostics(error)
        # The answer service already enforces its one format retry. Requeuing a
        # validated abstention or exhausted format failure would bypass that limit.
        retryable = diagnostics["failure_kind"] not in {
            "evidence_abstention", "answer_contract_failure",
        }
        diagnostics["job_retryable"] = retryable
        payload = json.dumps(
            diagnostics,
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
            terminal = await uow.jobs.finish(
                job.job_id,
                worker_id,
                success=False,
                error_type=str(diagnostics["error_type"]),
                error_summary="RAG query job failed; inspect runtime artifact",
                error_artifact_id=artifact_id,
                retry_delay_seconds=self.config.job_retry_delay_seconds,
                retryable=retryable,
                expected_attempt_count=job.attempt_count,
            )
        if terminal.status is JobStatus.FAILED:
            self._terminal_failures[job.job_id] = diagnostics
        print(
            "RAG_JOB_FAILURE " + json.dumps(
                {
                    **diagnostics,
                    "stage": self.stage.value,
                    "method": self.method.value,
                    "locator_hash": stable_hash(job.job_id)[:16],
                    "attempt": job.attempt_count,
                    "max_attempts": job.max_attempts,
                    "state": terminal.status.value,
                    "error_artifact_saved": artifact_id is not None,
                }, sort_keys=True, separators=(",", ":"),
            ), flush=True,
        )
        return terminal


class RagRetrievalRunner(_RagJobRunner):
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        gateway: EmbeddingGateway,
        error_artifacts: ExperimentArtifactService,
        method: BaselineMethod,
        config: RagBaselineConfig,
        corpus_id: str,
        document_collection: EmbeddingCollection,
        query_collection: EmbeddingCollection,
        lease_seconds: int,
    ) -> None:
        super().__init__(
            sessions=sessions,
            error_artifacts=error_artifacts,
            method=method,
            config=config,
            lease_seconds=lease_seconds,
            stage=AppStage.RETRIEVE,
        )
        if document_collection.status is not EmbeddingCollectionStatus.READY:
            raise ValueError("RAG retrieval requires a ready document collection")
        if query_collection.status not in {
            EmbeddingCollectionStatus.BUILDING,
            EmbeddingCollectionStatus.READY,
        }:
            raise ValueError("RAG retrieval requires a writable or ready query collection")
        self.execution = execution
        self.gateway = gateway
        self.corpus_id = corpus_id
        self.document_collection = document_collection
        self.query_collection = query_collection

    async def execute(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        selected: list[BenchmarkQuery],
        on_item: ItemCallback | None = None,
    ) -> tuple[int, int]:
        if not selected:
            raise ValueError("RAG retrieval requires at least one query")
        records = {record.query.query_id: record for record in selected}
        if len(records) != len(selected):
            raise ValueError("selected RAG queries contain duplicate IDs")
        await self._ensure_jobs(
            run_id=run_id, snapshot_id=snapshot_id, selected=selected
        )
        succeeded, failed = await self._claim_loop(
            run_id=run_id,
            records=records,
            concurrency=self.config.embedding_concurrency,
            process=self._process,
            on_item=on_item,
        )
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.embeddings is not None and uow.runs is not None
            if failed:
                await uow.runs.transition(
                    run_id,
                    RunStatus.FAILED,
                    error_type="RagRetrievalFailure",
                    error_summary="one or more retrieval jobs exhausted their attempts",
                )
            else:
                expected = sum(
                    (
                        2 + len(record.query.query_image_refs)
                        if self.method is BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG
                        else 1
                    )
                    for record in selected
                )
                actual = await uow.embeddings.count_records(
                    self.query_collection.collection_id
                )
                if actual != expected:
                    raise RuntimeError("query embedding collection is incomplete")
                await uow.embeddings.mark_collection_ready(
                    self.query_collection.collection_id,
                    index_type="none",
                    index_params={"ann": False, "role": "query"},
                )
        if failed:
            raise RuntimeError(f"{failed} RAG retrieval job(s) failed")
        return succeeded, failed

    async def _process(
        self, job: ExperimentJob, worker_id: str, record: BenchmarkQuery
    ) -> None:
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job, worker_id, stop))
        try:
            batch_id = retrieval_batch_id(job.run_id, record.query.query_id, self.method)
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.retrieval is not None and uow.embeddings is not None
                existing_batch = await uow.retrieval.get_batch(batch_id)
                persisted = await uow.embeddings.get_query_records(
                    self.query_collection.collection_id, record.query.query_id
                )
            if existing_batch is not None:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    assert uow.jobs is not None
                    await uow.jobs.finish(
                        job.job_id, worker_id, success=True,
                        expected_attempt_count=job.attempt_count,
                    )
                return

            if persisted:
                bundle = restore_query_embedding_bundle(
                    records=persisted,
                    retrieval_batch_id=batch_id,
                    method=self.method,
                    expected_image_count=len(record.query.query_image_refs),
                )
                embed_latency_ms = 0.0
            else:
                started = time.perf_counter()
                bundle = await produce_query_embeddings(
                    query=record,
                    method=self.method,
                    collection=self.query_collection,
                    config=self.config,
                    gateway=self.gateway,
                    execution=self.execution,
                    retrieval_batch_id=batch_id,
                    job_id=job.job_id,
                    attempt_number=job.attempt_count,
                )
                if (
                    self.query_collection.actual_model_id is not None
                    and bundle.actual_model_id != self.query_collection.actual_model_id
                ):
                    raise ValueError(
                        "query embedding provider identity differs from collection identity"
                    )
                embed_latency_ms = (time.perf_counter() - started) * 1000
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.embeddings is not None
                assert uow.retrieval is not None
                assert uow.jobs is not None
                await uow.jobs.finish(
                    job.job_id, worker_id, success=True,
                    expected_attempt_count=job.attempt_count,
                )
                await uow.embeddings.add_records(bundle.records)
                facts = await search_query_windows(
                    search=uow.embeddings,
                    run_id=job.run_id,
                    snapshot_id=job.snapshot_id,
                    corpus_id=self.corpus_id,
                    query=record,
                    method=self.method,
                    document_collection_id=self.document_collection.collection_id,
                    query_collection_id=self.query_collection.collection_id,
                    query_vector=bundle.effective_vector,
                    query_embeddings=bundle.references,
                    requested_top_k=self.config.retrieval_top_k,
                    embed_latency_ms=embed_latency_ms,
                )
                await uow.retrieval.add_batch(
                    facts.batch,
                    facts.traces,
                    components=facts.components,
                    query_embeddings=facts.query_embeddings,
                )
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat


class RagAnswerRunner(_RagJobRunner):
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        input_provider: RagAnswerInputProvider,
        answer_service: BaselineAnswerService,
        error_artifacts: ExperimentArtifactService,
        method: BaselineMethod,
        config: RagBaselineConfig,
        lease_seconds: int,
        phase: str = "answer",
    ) -> None:
        super().__init__(
            sessions=sessions,
            error_artifacts=error_artifacts,
            method=method,
            config=config,
            lease_seconds=lease_seconds,
            stage=AppStage.BASELINE if phase == "generate" else AppStage.ANSWER,
        )
        self.input_provider = input_provider
        self.answer_service = answer_service
        self._abstained_queries: set[str] = set()
        self.phase = phase

    async def execute(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        selected: list[BenchmarkQuery],
        on_item: ItemCallback | None = None,
    ) -> tuple[int, int]:
        records = {record.query.query_id: record for record in selected}
        if not records or len(records) != len(selected):
            raise ValueError("RAG answer phase requires unique selected queries")
        await self._ensure_jobs(
            run_id=run_id, snapshot_id=snapshot_id, selected=selected
        )
        succeeded, failed = await self._claim_loop(
            run_id=run_id,
            records=records,
            concurrency=self.config.answer_concurrency,
            process=self._process,
            on_item=on_item,
        )
        if self.phase == "generate":
            if failed or succeeded != len(selected):
                raise RuntimeError("typed baseline generation did not finish every candidate")
            return succeeded, failed
        async with ExperimentUnitOfWork(self.sessions) as uow:
            assert uow.query_executor is not None
            typed = (
                getattr(self.input_provider, "answer_contract_version", None)
                or self.config.answer_contract_version
            ) == TYPED_ANSWER_CONTRACT_VERSION
            if typed:
                inventory = await typed_baseline_inventory(
                    uow.query_executor, run_id=run_id, selected=selected, method=self.method,
                )
                if inventory["answered"] + inventory["abstained"] != succeeded:
                    raise ValueError("typed terminal and succeeded-job counts disagree")
                abstained_count = inventory["abstained"]
            else:
                abstained_count = await uow.query_executor.abstention_count(run_id)
        diagnostics: dict[str, object] = {
            "expected_count": len(selected),
            "succeeded_count": succeeded,
            "succeeded_count_semantics": "processed_jobs_not_answer_success",
            "answered_count": succeeded - abstained_count,
            "abstained_count": abstained_count,
            "failed_count": failed,
            "terminal_count": succeeded + failed,
            "failure_kind_counts": dict(Counter(
                str(item["failure_kind"]) for item in self._terminal_failures.values()
            )),
            "error_type_counts": dict(Counter(
                str(item["error_type"]) for item in self._terminal_failures.values()
            )),
            "operation_counts": dict(Counter(
                str(item["operation"]) for item in self._terminal_failures.values()
            )),
            "contract_failure_counts": dict(Counter(
                str(code)
                for item in self._terminal_failures.values()
                if isinstance(codes := item.get("contract_failure_codes"), list)
                for code in codes
            )),
            "unobserved_failed_count": max(0, failed - len(self._terminal_failures)),
        }
        print("RAG_ANSWER_ACCOUNTING " + json.dumps(
            diagnostics, sort_keys=True, separators=(",", ":"),
        ), flush=True)
        try:
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.runs is not None
                if failed:
                    await uow.runs.transition(
                        run_id,
                        RunStatus.FAILED,
                        error_type="RagAnswerFailure",
                        error_summary="one or more answer jobs exhausted their attempts",
                    )
                else:
                    assert uow.jobs is not None
                    retrieval_status = await uow.jobs.status_counts(run_id, stage=AppStage.RETRIEVE)
                    answer_status = await uow.jobs.status_counts(run_id, stage=AppStage.ANSWER)
                    retrieval_jobs = retrieval_status.get(JobStatus.SUCCEEDED, 0)
                    answer_jobs = answer_status.get(JobStatus.SUCCEEDED, 0)
                    diagnostics["completion_job_inventory"] = {
                        "expected_query_count": len(selected),
                        "retrieval_succeeded_count": retrieval_jobs,
                        "answer_succeeded_count": answer_jobs,
                        "missing_retrieval_job_count": max(0, len(selected) - retrieval_jobs),
                        "missing_answer_job_count": max(0, len(selected) - answer_jobs),
                        "state": ("ready" if retrieval_jobs == answer_jobs == len(selected)
                                  else "job_denominator_mismatch"),
                    }
                    print("RAG_RUN_COMPLETION_PREFLIGHT " + json.dumps(
                        diagnostics["completion_job_inventory"], sort_keys=True,
                    ), flush=True)
                    await uow.runs.transition(run_id, RunStatus.COMPLETED)
        except Exception as exc:
            completion = {
                **diagnostics, "failure_scope": "run_completion",
                "completion_error_type": type(exc).__name__,
            }
            print("RAG_RUN_COMPLETION_FAILURE " + json.dumps(
                completion, sort_keys=True, separators=(",", ":"),
            ), flush=True)
            raise RagAnswerCompletionFailure(completion) from exc
        if failed:
            raise RagAnswerBatchFailure(diagnostics)
        return succeeded, failed

    async def _process(
        self, job: ExperimentJob, worker_id: str, record: BenchmarkQuery
    ) -> None:
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job, worker_id, stop))
        operation = "prepare_context"
        try:
            context_job = job
            if self.phase != "generate" and (
                getattr(self.input_provider, "answer_contract_version", None)
                or self.config.answer_contract_version
            ) == TYPED_ANSWER_CONTRACT_VERSION:
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
            operation = "persist_context"
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.contexts is not None and uow.jobs is not None
                await uow.jobs.heartbeat(
                    job.job_id, worker_id, self.lease_seconds,
                    expected_attempt_count=job.attempt_count,
                )
                await uow.contexts.add(prepared.context_assembly)
            operation = "load_answer"
            if prepared.metadata.get("answer_contract_version") == TYPED_ANSWER_CONTRACT_VERSION:
                result = await process_typed_baseline_answer(
                    prepared=prepared, job=job, worker_id=worker_id, method=self.method,
                    answer_service=self.answer_service,
                    uow_factory=lambda: ExperimentUnitOfWork(self.sessions),
                    phase=self.phase,
                )
                if isinstance(result, TypedAnswerResult) and result.terminal_state == "abstained":
                    self._abstained_queries.add(record.query.query_id)
                return
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.answers is not None
                answer = await uow.answers.get_for_query(
                    job.run_id, record.query.query_id
                )
                abstention = None
                if prepared.metadata.get("answer_contract_version") == "strict-answer-contract-v2":
                    assert uow.query_executor is not None
                    abstention = await uow.query_executor.get_abstention(
                        job.run_id, record.query.query_id,
                    )
                if abstention is not None:
                    if answer is not None:
                        raise ValueError("answer and abstention terminals overlap")
                    assert uow.jobs is not None
                    await uow.jobs.finish(
                        job.job_id, worker_id, success=True,
                        expected_attempt_count=job.attempt_count,
                    )
                    self._abstained_queries.add(record.query.query_id)
                    return
            if answer is None:
                async with ExperimentUnitOfWork(self.sessions) as uow:
                    successful = await uow.model_calls.get_successful_for_job(job.job_id)
                if (
                    successful is not None
                    and prepared.metadata.get("answer_contract_version")
                    != "strict-answer-contract-v2"
                ):
                    answer = recover_answer_from_trace(
                        prepared,
                        job,
                        successful.parsed_output,
                        actual_model_id=successful.actual_model_id,
                        model_call_id=successful.model_call_id,
                        input_tokens=successful.input_tokens,
                        output_tokens=successful.output_tokens,
                        latency_ms=successful.latency_ms,
                        langfuse_trace_id=successful.langfuse_trace_id,
                    )
                else:
                    operation = "generate_answer"
                    if successful is None:
                        generated = await self.answer_service.answer(prepared, job)
                    else:
                        generated = await self.answer_service.answer(
                            prepared, job, recovered_trace=successful,
                        )
                    if not isinstance(generated, FinalAnswerRecord):
                        raise ValueError("legacy RAG job returned a typed answer")
                    answer = generated
            if answer is None:
                raise ValueError("legacy RAG job has no final answer")
            operation = "persist_answer"
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.answers is not None and uow.jobs is not None
                await uow.jobs.finish(
                    job.job_id, worker_id, success=True,
                    expected_attempt_count=job.attempt_count,
                )
                await uow.answers.add(answer)
        except UnsupportedRuntimeAnswer as exc:
            if exc.fact is None:
                raise RagAnswerOperationError(operation, exc) from exc
            operation = "persist_abstention"
            async with ExperimentUnitOfWork(self.sessions) as uow:
                assert uow.query_executor is not None and uow.jobs is not None
                await uow.jobs.finish(
                    job.job_id, worker_id, success=True,
                    expected_attempt_count=job.attempt_count,
                )
                await uow.query_executor.add_structured_answer(exc.fact)
            self._abstained_queries.add(record.query.query_id)
        except Exception as exc:
            raise RagAnswerOperationError(operation, exc) from exc
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
