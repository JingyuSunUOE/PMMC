"""Recoverable, bounded-concurrency runner for the Online Query Executor."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.experiments.manifest import run_config_hash
from pmmc.infrastructure.database.errors import (
    PersistenceConflictError,
    safe_database_error_diagnostics,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.query_executor_repositories import (
    QueryExecutorPersistenceConflictError,
)
from pmmc.runtime.qbank_reranker import RerankerProviderContractError
from pmmc.runtime.query_executor import (
    OnlineQueryExecutor,
    QueryAttemptLease,
    QueryExecutorGraphState,
)
from pmmc.schemas import (
    DataScope,
    JobStatus,
    QueryExecutorRunManifest,
    QueryExecutorRunState,
    QueryExecutorStage,
    QueryExecutorStageJob,
    RunManifest,
    RunQueryAssignment,
)
from pmmc.schemas.base import utc_now

ProgressCallback = Callable[[dict[str, int]], None]

_RUNTIME_RUN_AUDIT_FIELDS = frozenset({"config_hash", "created_at"})


def _query_runner_event(state: str, **values: object) -> None:
    print(
        "STEP7_QUERY_RUNNER "
        + json.dumps(
            {"state": state, **values},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _active_query_worker_count(configured: int, selected: int) -> int:
    if configured <= 0 or selected <= 0:
        raise ValueError("query worker denominator must be positive")
    return min(configured, selected)


def _runtime_run_registration_conflicts(
    actual: RunManifest,
    expected: RunManifest,
) -> tuple[str, ...]:
    """Compare scientific identity while preserving first-insert audit fields."""

    if actual.config_hash != run_config_hash(actual) or expected.config_hash != run_config_hash(
        expected
    ):
        return ("config_hash",)
    actual_payload = actual.model_dump(mode="json")
    expected_payload = expected.model_dump(mode="json")
    return tuple(
        name
        for name in sorted(expected_payload)
        if name not in _RUNTIME_RUN_AUDIT_FIELDS
        and actual_payload.get(name) != expected_payload.get(name)
    )


def _safe_query_job_error(exc: BaseException) -> dict[str, object]:
    return {
        key: value
        for key, value in safe_database_error_diagnostics(exc).items()
        if re.fullmatch(r"[A-Za-z0-9_]{1,128}", value)
    }


def _query_job_retryable(exc: BaseException) -> bool:
    # Retrying an immutable/constraint violation cannot repair its input and can
    # replace the useful first error with a model-call idempotency conflict.
    return not isinstance(
        exc,
        (IntegrityError, PersistenceConflictError, RerankerProviderContractError),
    )


@dataclass(frozen=True)
class QueryExecutorBatchResult:
    query_executor_run_id: str
    selected_count: int
    succeeded_count: int
    failed_count: int
    resumed_count: int
    technical_failed_count: int = 0
    failure_diagnostics: tuple[dict[str, object], ...] = ()


class QueryExecutorBatchRunner:
    """Claim one whole-query job at a time; model/database work stays outside leases."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        executor: OnlineQueryExecutor,
        concurrency: int = 32,
        lease_seconds: int = 900,
        phase: QueryExecutorStage = QueryExecutorStage.SELECT,
    ) -> None:
        if concurrency <= 0 or lease_seconds <= 0:
            raise ValueError("query runner concurrency and lease must be positive")
        self.sessions = sessions
        self.executor = executor
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.phase = phase

    async def run(
        self,
        manifest: QueryExecutorRunManifest,
        query_ids: Sequence[str],
        *,
        on_progress: ProgressCallback | None = None,
        prepared_state: QueryExecutorRunState | None = None,
    ) -> QueryExecutorBatchResult:
        if len(query_ids) != manifest.selected_query_count or len(set(query_ids)) != len(query_ids):
            raise ValueError("query executor denominator must be unique and complete")
        reused_prepared_state = prepared_state is not None
        state = prepared_state or await self.prepare(manifest, query_ids)
        if (
            state.query_executor_run_id != manifest.query_executor_run_id
            or state.selected_count != len(query_ids)
        ):
            raise ValueError("prepared query executor state does not match the denominator")
        if state.status.value == "completed":
            _query_runner_event(
                "terminal_reuse",
                selected_count=state.selected_count,
                selected_answer_count=state.selected_answer_count,
                failed_count=state.failed_count,
                no_answer_count=state.failed_count,
                technical_failed_count=0,
            )
            return QueryExecutorBatchResult(
                query_executor_run_id=manifest.query_executor_run_id,
                selected_count=state.selected_count,
                succeeded_count=state.selected_answer_count,
                failed_count=state.failed_count,
                resumed_count=state.selected_count,
            )

        totals: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        route_confidences: list[float] = []
        route_gaps: list[float] = []
        totals_lock = asyncio.Lock()
        active_worker_count = _active_query_worker_count(
            self.concurrency,
            len(query_ids),
        )
        _query_runner_event(
            "starting",
            active_worker_count=active_worker_count,
            configured_concurrency=self.concurrency,
            prepared_state_reused=reused_prepared_state,
            selected_count=len(query_ids),
        )

        async def worker(index: int) -> None:
            worker_id = f"query-executor-{index:02d}"
            while True:
                if self.executor.config.is_v3:
                    worker_id = f"query-executor-{index:02d}-{uuid4().hex}"
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.query_executor is not None
                    job = await uow.query_executor.claim_stage_job(
                        run_id=manifest.query_executor_run_id,
                        stage=self.phase.value,
                        worker_id=worker_id,
                        now=utc_now(),
                        lease_seconds=self.lease_seconds,
                    )
                if job is None:
                    return
                resumed = job.attempt_count > 1
                stop_heartbeat = asyncio.Event()
                heartbeat = asyncio.create_task(
                    self._heartbeat(
                        job.query_stage_job_id,
                        worker_id,
                        stop_heartbeat,
                        attempt_count=job.attempt_count if self.executor.config.is_v3 else None,
                    )
                )
                phase_options: dict[str, Any] = (
                    {
                        "phase": self.phase.value,
                        "lease": QueryAttemptLease(
                            job.query_stage_job_id, worker_id, job.attempt_count
                        ),
                    }
                    if self.executor.config.is_v3
                    else {}
                )
                operation = asyncio.create_task(
                    self.executor.execute(
                        run_id=manifest.query_executor_run_id,
                        query_id=job.query_id,
                        **phase_options,
                    )
                )
                operation_error: Exception | None = None
                operation_result: QueryExecutorGraphState | None = None
                try:
                    done, _pending = await asyncio.wait(
                        {operation, heartbeat},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if operation in done:
                        operation_result = await operation
                        if heartbeat.done():
                            heartbeat_error = heartbeat.exception()
                            if heartbeat_error is not None:
                                raise heartbeat_error
                    else:
                        heartbeat_error = heartbeat.exception()
                        if heartbeat_error is None:
                            raise RuntimeError(
                                "query stage heartbeat stopped before operation completed"
                            )
                        raise heartbeat_error
                except Exception as exc:
                    operation_error = exc
                finally:
                    stop_heartbeat.set()
                    if not operation.done():
                        operation.cancel()
                    heartbeat.cancel()
                    results = await asyncio.gather(
                        operation,
                        heartbeat,
                        return_exceptions=True,
                    )
                    for result in results:
                        if isinstance(result, Exception):
                            operation_error = operation_error or result
                if operation_error is not None:
                    safe_error = _safe_query_job_error(operation_error)
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.query_executor is not None
                        finished = await uow.query_executor.finish_stage_job(
                            job_id=job.query_stage_job_id,
                            worker_id=worker_id,
                            success=False,
                            now=utc_now(),
                            error_type=type(operation_error).__name__,
                            error_summary=json.dumps(safe_error, sort_keys=True),
                            retryable=_query_job_retryable(operation_error),
                            **(
                                {"attempt_count": job.attempt_count}
                                if self.executor.config.is_v3
                                else {}
                            ),
                        )
                    async with totals_lock:
                        totals["resumed"] += int(resumed)
                        totals["retrying"] += int(finished.status is JobStatus.PENDING)
                        if finished.status is JobStatus.FAILED:
                            totals["failed"] += 1
                        signature = json.dumps(safe_error, sort_keys=True)
                        failures[signature] += 1
                        if failures[signature] == 1:
                            _query_runner_event(
                                (
                                    "job_retrying"
                                    if finished.status is JobStatus.PENDING
                                    else "job_failed"
                                ),
                                attempt_count=finished.attempt_count,
                                max_attempts=finished.max_attempts,
                                retryable=_query_job_retryable(operation_error),
                                item_hash=canonical_content_hash(
                                    [manifest.query_executor_run_id, job.query_id]
                                )[:16],
                                **safe_error,
                            )
                        if on_progress is not None:
                            on_progress(dict(totals))
                    if finished.status is JobStatus.FAILED:
                        continue
                else:
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.query_executor is not None
                        await uow.query_executor.finish_stage_job(
                            job_id=job.query_stage_job_id,
                            worker_id=worker_id,
                            success=True,
                            now=utc_now(),
                            **(
                                {"attempt_count": job.attempt_count}
                                if self.executor.config.is_v3
                                else {}
                            ),
                        )
                    async with totals_lock:
                        assert operation_result is not None
                        totals["succeeded"] += 1
                        totals["resumed"] += int(resumed)
                        totals["routes"] += int("route_decision" in operation_result)
                        route_decision = operation_result.get("route_decision")
                        if route_decision is not None:
                            totals[f"route_outcome:{route_decision.outcome.value}"] += 1
                            if route_decision.fallback_reason is not None:
                                totals[
                                    f"route_fallback_reason:{route_decision.fallback_reason.value}"
                                ] += 1
                            if route_decision.confidence is not None:
                                route_confidences.append(float(route_decision.confidence))
                            if route_decision.top1_top2_gap is not None:
                                route_gaps.append(float(route_decision.top1_top2_gap))
                        totals["bound_programs"] += len(operation_result.get("frozen_by_round", {}))
                        totals["executions"] += len(operation_result.get("execution_by_round", {}))
                        totals["sufficient"] += sum(
                            int(value.sufficient)
                            for value in operation_result.get("sufficiency_by_round", {}).values()
                        )
                        totals["pmmc_answers"] += len(operation_result.get("pmmc_answers", {}))
                        fallback_decision = operation_result.get("fallback_decision")
                        totals["active_fallback"] += int(
                            bool(
                                fallback_decision is not None and fallback_decision.active_fallback
                            )
                        )
                        if (
                            fallback_decision is not None
                            and fallback_decision.active_fallback
                            and fallback_decision.reason is not None
                        ):
                            totals[f"fallback_reason:{fallback_decision.reason.value}"] += 1
                        answer_selection = operation_result.get("answer_selection")
                        totals["terminal_failures"] += int(
                            bool(answer_selection is not None and answer_selection.terminal_failure)
                        )
                        if on_progress is not None:
                            on_progress(dict(totals))

        tasks = [asyncio.create_task(worker(index)) for index in range(active_worker_count)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        operation = "stage_counts"
        fact_counts: dict[str, int] = {}
        if self.phase is not QueryExecutorStage.SELECT:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.query_executor is not None
                counts = await uow.query_executor.stage_status_counts(
                    manifest.query_executor_run_id,
                    self.phase.value,
                )
            if counts.get("pending", 0) or counts.get("running", 0):
                raise RuntimeError("runtime phase stopped with unfinished jobs")
            return QueryExecutorBatchResult(
                query_executor_run_id=manifest.query_executor_run_id,
                selected_count=len(query_ids),
                succeeded_count=counts.get("succeeded", 0),
                failed_count=counts.get("failed", 0),
                resumed_count=totals["resumed"],
                technical_failed_count=counts.get("failed", 0),
            )
        _query_runner_event("finalization_started", query_jobs_finished=True, run_completed=False)
        try:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.query_executor is not None
                counts = await uow.query_executor.stage_status_counts(
                    manifest.query_executor_run_id, QueryExecutorStage.SELECT.value
                )
                failed = counts.get(JobStatus.FAILED.value, 0)
                pending = counts.get(JobStatus.PENDING.value, 0) + counts.get(
                    JobStatus.RUNNING.value, 0
                )
                if pending:
                    raise RuntimeError("query executor workers stopped with unfinished jobs")
                operation = "terminal_fact_counts"
                fact_counts = await uow.query_executor.terminal_fact_counts(
                    manifest.query_executor_run_id
                )
                operation = "transition_run"
                await uow.query_executor.transition_run(
                    manifest.query_executor_run_id,
                    status="completed" if failed == 0 else "failed",
                    counts={
                        **{
                            key: fact_counts[key]
                            for key in (
                                "matched_count",
                                "executed_count",
                                "pmmc_answer_count",
                                "fallback_answer_count",
                                "selected_answer_count",
                            )
                        },
                        "failed_count": fact_counts["failed_count"] + failed,
                    },
                    now=utc_now(),
                    error_type="QueryItemFailures" if failed else None,
                    error_summary=(
                        "one or more query items have terminal technical failures"
                        if failed
                        else None
                    ),
                )
                operation = "read_final_state"
                final_state = await uow.query_executor.get_state(manifest.query_executor_run_id)
                operation = "commit"
        except Exception as exc:
            _query_runner_event(
                "finalization_failed",
                operation=operation,
                run_completed=False,
                fact_counts=fact_counts,
                **_safe_query_job_error(exc),
            )
            raise
        _query_runner_event(
            "failed"
            if failed
            else "completed_with_no_answers"
            if final_state.failed_count
            else "completed",
            active_fallback_count=totals["active_fallback"],
            executed_program_count=totals["executions"],
            execution_counter_scope="durably_completed_queries_only",
            failed_count=final_state.failed_count,
            failed_count_semantics="technical_failures_plus_no_answer",
            technical_failed_count=failed,
            failure_diagnostics=[
                {**json.loads(key), "count": value} for key, value in sorted(failures.items())
            ],
            no_answer_count=fact_counts["failed_count"],
            fallback_reason_counts={
                key.removeprefix("fallback_reason:"): value
                for key, value in sorted(totals.items())
                if key.startswith("fallback_reason:")
            },
            pmmc_answer_count=totals["pmmc_answers"],
            route_confidence_summary={
                "count": len(route_confidences),
                "max": round(max(route_confidences), 6) if route_confidences else None,
                "mean": (
                    round(sum(route_confidences) / len(route_confidences), 6)
                    if route_confidences
                    else None
                ),
                "min": round(min(route_confidences), 6) if route_confidences else None,
            },
            route_fallback_reason_counts={
                key.removeprefix("route_fallback_reason:"): value
                for key, value in sorted(totals.items())
                if key.startswith("route_fallback_reason:")
            },
            route_gap_summary={
                "count": len(route_gaps),
                "max": round(max(route_gaps), 6) if route_gaps else None,
                "mean": (round(sum(route_gaps) / len(route_gaps), 6) if route_gaps else None),
                "min": round(min(route_gaps), 6) if route_gaps else None,
            },
            route_outcome_counts={
                key.removeprefix("route_outcome:"): value
                for key, value in sorted(totals.items())
                if key.startswith("route_outcome:")
            },
            selected_answer_count=final_state.selected_answer_count,
            sufficient_round_count=totals["sufficient"],
        )
        return QueryExecutorBatchResult(
            query_executor_run_id=manifest.query_executor_run_id,
            selected_count=len(query_ids),
            succeeded_count=final_state.selected_answer_count,
            failed_count=final_state.failed_count,
            resumed_count=totals["resumed"],
            technical_failed_count=failed,
            failure_diagnostics=tuple(
                {**json.loads(key), "count": value} for key, value in sorted(failures.items())
            ),
        )

    async def _heartbeat(
        self,
        job_id: str,
        worker_id: str,
        stopped: asyncio.Event,
        attempt_count: int | None = None,
    ) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while True:
            try:
                await asyncio.wait_for(stopped.wait(), timeout=interval)
                return
            except TimeoutError:
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.query_executor is not None
                    await uow.query_executor.heartbeat_stage_job(
                        job_id=job_id,
                        worker_id=worker_id,
                        now=utc_now(),
                        lease_seconds=self.lease_seconds,
                        **({"attempt_count": attempt_count} if attempt_count is not None else {}),
                    )

    async def prepare(
        self,
        manifest: QueryExecutorRunManifest,
        query_ids: Sequence[str],
    ) -> QueryExecutorRunState:
        """Register the parent run before embedding/model stages begin."""

        if len(query_ids) != manifest.selected_query_count or len(set(query_ids)) != len(query_ids):
            raise ValueError("query executor denominator must be unique and complete")
        jobs = [self._job(manifest, query_id, self.phase) for query_id in query_ids]
        run_manifest, assignments = self._runtime_run_registration(
            manifest,
            query_ids,
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.runs is not None
            assert uow.query_executor is not None
            existing_manifest: RunManifest | None = None
            try:
                existing_manifest = await uow.runs.get_manifest(run_manifest.run_id)
            except KeyError:
                pass
            if existing_manifest is None:
                try:
                    await uow.runs.register(run_manifest, assignments)
                except PersistenceConflictError as exc:
                    raise QueryExecutorPersistenceConflictError(
                        "query executor generic run registration conflicted",
                        operation="register_query_executor_runtime_run",
                        error_code="runtime_run_registration_conflict",
                    ) from exc
                _query_runner_event(
                    "runtime_registration_preflight",
                    audit_fields_reused=0,
                    existing_run=False,
                    scientific_identity_match=True,
                    selected_count=len(query_ids),
                )
            else:
                conflicts = _runtime_run_registration_conflicts(
                    existing_manifest,
                    run_manifest,
                )
                if conflicts:
                    raise QueryExecutorPersistenceConflictError(
                        "query executor generic run scientific identity differs",
                        operation="register_query_executor_runtime_run",
                        error_code="runtime_run_scientific_identity_conflict",
                    )
                existing_state = await uow.runs.get_state(run_manifest.run_id)
                if existing_state.selected_count != len(assignments):
                    raise QueryExecutorPersistenceConflictError(
                        "query executor generic run denominator differs",
                        operation="register_query_executor_runtime_run",
                        error_code="runtime_run_query_denominator_conflict",
                    )
                _query_runner_event(
                    "runtime_registration_preflight",
                    audit_fields_reused=len(_RUNTIME_RUN_AUDIT_FIELDS),
                    existing_run=True,
                    scientific_identity_match=True,
                    selected_count=len(query_ids),
                )
            await uow.query_executor.add_run(manifest)
            await uow.query_executor.add_stage_jobs(jobs)
            state = await uow.query_executor.get_state(manifest.query_executor_run_id)
            if state.status.value == "created":
                await uow.query_executor.transition_run(
                    manifest.query_executor_run_id,
                    status="running",
                    now=utc_now(),
                )
                state = await uow.query_executor.get_state(manifest.query_executor_run_id)
        return state

    @staticmethod
    def _job(
        manifest: QueryExecutorRunManifest,
        query_id: str,
        phase: QueryExecutorStage = QueryExecutorStage.SELECT,
    ) -> QueryExecutorStageJob:
        payload = {
            "run_id": manifest.query_executor_run_id,
            "query_id": query_id,
            "stage": phase.value,
            "config_hash": manifest.config_hash,
        }
        return QueryExecutorStageJob(
            query_stage_job_id=stable_compiler_id("query-stage-job", payload),
            query_executor_run_id=manifest.query_executor_run_id,
            snapshot_id=manifest.snapshot_id,
            query_id=query_id,
            stage=phase,
            max_attempts=2,
            content_hash=canonical_content_hash(payload),
        )

    @staticmethod
    def _runtime_run_registration(
        manifest: QueryExecutorRunManifest,
        query_ids: Sequence[str],
    ) -> tuple[RunManifest, list[RunQueryAssignment]]:
        selection_hash = canonical_content_hash(
            {
                "snapshot_id": manifest.snapshot_id,
                "dataset": manifest.dataset.value,
                "query_ids": list(query_ids),
            }
        )
        generic = RunManifest(
            run_id=manifest.query_executor_run_id,
            experiment_name="pmmc-online-query-executor",
            dataset=manifest.dataset,
            split="development" if manifest.pilot_projection else "paper_test",
            method="pmmc",
            paper_mode=manifest.paper_result,
            answer_model_alias=str(
                manifest.metadata.get("answer_model_alias", "answer-qwen3.5-9b")
            ),
            embedding_model_aliases=[
                "qwen3-embedding-8b",
                "qwen3-vl-embedding-8b",
            ],
            r_max=manifest.max_round,
            question_budget_max=manifest.question_budget,
            match_topk=50,
            match_threshold=float(manifest.metadata.get("match_threshold", 0.60)),
            fallback_policy="native_mm_rag_v1",
            prompt_hashes={
                str(manifest.metadata["answer_prompt_version"]): str(
                    manifest.metadata["answer_prompt_hash"]
                )
            },
            selection_hash=selection_hash,
            dataset_snapshot_ids=[manifest.snapshot_id],
            metadata={
                "source_kind": "pmmc_query_run",
                "source_compiler_run_id": manifest.source_compiler_run_id,
                "routing_projection_id": manifest.routing_projection_id,
                "query_executor_schema_version": manifest.schema_version,
                "pilot_projection": manifest.pilot_projection,
            },
        )
        generic = generic.model_copy(update={"config_hash": run_config_hash(generic)})
        assignments = [
            RunQueryAssignment(
                run_id=manifest.query_executor_run_id,
                snapshot_id=manifest.snapshot_id,
                query_id=query_id,
                ordinal=ordinal,
                selection_hash=selection_hash,
            )
            for ordinal, query_id in enumerate(query_ids)
        ]
        return generic, assignments


__all__ = ["QueryExecutorBatchResult", "QueryExecutorBatchRunner"]
