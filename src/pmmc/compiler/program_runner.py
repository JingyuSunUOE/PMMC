"""Real operator-kernel dry-run stage for Planner v1 programs."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.config import CompilerFoundationConfig
from pmmc.compiler.dry_run import assess_program_dry_run, dry_run_outcome_status
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.memory_index_runner import ProgramQueryVectorSet
from pmmc.compiler.planner import planner_terminal_outcome
from pmmc.compiler.planner_runner import _validation_fact
from pmmc.evidence_lineage import EvidenceLineageResolver
from pmmc.infrastructure.database.compiler_repositories import (
    SessionScopedOperatorDataRepository,
)
from pmmc.infrastructure.database.errors import (
    safe_database_error_diagnostics,
    safe_database_error_summary,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import PostgresMemoryRepository
from pmmc.memory.operators import LinearProgramExecutor, ProgramLineageValidator, ProgramValidator
from pmmc.retrieval.operators import build_operator_registry
from pmmc.schemas import (
    CompilationStage,
    CompilationStageResult,
    CompilationUnit,
    CompilerStageJob,
    CompilerStageJobStatus,
    CompilerStageKind,
    DataScope,
    MemoryOperatorIndexManifest,
    MemoryProgram,
    OperatorEvidence,
    OperatorExecutionContext,
    OperatorExecutionPurpose,
    PlannerOutcomeStatus,
    ProgramDryRunAssessment,
    ProgramPolicyIssue,
    ProgramValidationKind,
    QuestionCandidate,
    TraceStatus,
)
from pmmc.schemas.base import utc_now

DryRunItemCallback = Callable[[str, int, int], None]

_DRY_RUN_COUNT_KEYS = (
    "executions",
    "ready",
    "insufficient",
    "fallback",
    "abstain",
    "failed",
    "lineage_rejected",
    "query_vector_unavailable",
)


class ProgramDryRunPersistenceError(RuntimeError):
    """A persistence failure rendered without SQL text or bound values."""

    def __init__(self, stage: str, error: BaseException) -> None:
        self.failure_stage = f"dry_run_{stage}"
        self.error_code = "dry_run_persistence_failed"
        diagnostics = safe_database_error_diagnostics(error)
        diagnostics["failure_stage"] = self.failure_stage
        diagnostics["error_code"] = self.error_code
        self.safe_summary = safe_database_error_summary(diagnostics)
        super().__init__(self.safe_summary)


def _normalized_dry_run_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """Return the complete progress/accounting contract, including zero values."""

    return {key: int(counts.get(key, 0)) for key in _DRY_RUN_COUNT_KEYS}


async def _await_workers(tasks: Sequence[asyncio.Task[None]]) -> None:
    """Cancel and await peers before an engine can be disposed after a failure."""

    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass(frozen=True)
class ProgramDryRunResult:
    run_id: str
    unit_count: int
    execution_count: int
    program_ready_count: int
    insufficient_count: int
    fallback_count: int
    abstain_count: int
    execution_failed_count: int
    lineage_rejected_count: int
    query_vector_unavailable_count: int


class ProgramDryRunRunner:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        config: CompilerFoundationConfig,
        lease_seconds: int,
        concurrency: int = 32,
    ) -> None:
        if config.operator_set_version not in {
            "pmmc-operators-v2",
            "pmmc-operators-v3",
            "pmmc-operators-v4",
            "pmmc-operators-v5",
            "pmmc-operators-v6",
            "pmmc-operators-v7",
        }:
            raise ValueError(
                "Planner dry-run requires pmmc-operators-v2 through v7"
            )
        if lease_seconds <= 0 or concurrency <= 0:
            raise ValueError("dry-run lease and concurrency must be positive")
        self.sessions = sessions
        self.config = config
        self.lease_seconds = lease_seconds
        self.concurrency = concurrency

    async def execute(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        units: Sequence[CompilationUnit],
        memory_index: MemoryOperatorIndexManifest,
        query_vectors: Mapping[str, ProgramQueryVectorSet],
        on_item: DryRunItemCallback | None = None,
    ) -> ProgramDryRunResult:
        if memory_index.status != "ready" or memory_index.run_id != run_id:
            raise ValueError("dry-run requires the run's ready memory operator index")
        unit_by_id = {item.compiler_unit_id: item for item in units}
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.memory_programs is not None
            assert uow.questioner is not None
            assert uow.compiler_stage_jobs is not None
            programs = await uow.memory_programs.list_for_run(run_id)
            candidates = await uow.questioner.list_eligible_questions(run_id)
            expected = await uow.questioner.list_expected_evidence(run_id)
            candidate_by_id = {item.qbank_id: item for item in candidates}
            for unit in units:
                job = self._stage_job(run_id, unit.compiler_unit_id)
                existing = await uow.compiler_stage_jobs.get(job.stage_job_id)
                if existing is None:
                    await uow.compiler_stage_jobs.add(job)
                elif existing.content_hash != job.content_hash:
                    raise ValueError("persisted dry-run stage job identity conflicts")
        programs_by_unit: dict[str, list[MemoryProgram]] = defaultdict(list)
        for program in programs:
            if program.compiler_unit_id not in unit_by_id:
                raise ValueError("program is outside the dry-run unit denominator")
            programs_by_unit[program.compiler_unit_id].append(program)

        totals: defaultdict[str, int] = defaultdict(int)
        totals_lock = asyncio.Lock()

        async def worker(index: int) -> None:
            worker_id = f"dry-run-{index:02d}"
            while True:
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.compiler_stage_jobs is not None
                    job = await uow.compiler_stage_jobs.claim(
                        run_id=run_id,
                        stage=CompilerStageKind.DRY_RUN,
                        worker_id=worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                if job is None:
                    return
                unit = unit_by_id[job.compiler_unit_id]
                try:
                    counts = _normalized_dry_run_counts(
                        await self._process_unit(
                            job=job,
                            unit=unit,
                            programs=programs_by_unit.get(unit.compiler_unit_id, []),
                            candidates=candidate_by_id,
                            expected=expected,
                            memory_index=memory_index,
                            query_vectors=query_vectors,
                        )
                    )
                except Exception as exc:
                    diagnostics = safe_database_error_diagnostics(exc)
                    try:
                        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                            assert uow.compiler_stage_jobs is not None
                            await uow.compiler_stage_jobs.finish(
                                job.stage_job_id,
                                worker_id=worker_id,
                                succeeded=False,
                                error_type=type(exc).__name__,
                                error_summary=safe_database_error_summary(diagnostics),
                            )
                    except Exception as finish_error:
                        exc.add_note(
                            f"failed to persist dry-run job failure: {type(finish_error).__name__}"
                        )
                    raise
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.compiler_stage_jobs is not None
                    await uow.compiler_stage_jobs.finish(
                        job.stage_job_id, worker_id=worker_id, succeeded=True
                    )
                async with totals_lock:
                    for key, value in counts.items():
                        totals[key] += value
                if on_item is not None:
                    on_item(
                        unit.compiler_unit_id,
                        counts["executions"],
                        counts["insufficient"],
                    )

        tasks = [asyncio.create_task(worker(index)) for index in range(self.concurrency)]
        await _await_workers(tasks)
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_stage_jobs is not None
            jobs = await uow.compiler_stage_jobs.list_for_run(run_id, CompilerStageKind.DRY_RUN)
        if sum(item.status is CompilerStageJobStatus.SUCCEEDED for item in jobs) != len(units):
            raise RuntimeError("dry-run did not complete every compiler unit")
        return ProgramDryRunResult(
            run_id=run_id,
            unit_count=len(units),
            execution_count=totals["executions"],
            program_ready_count=totals["ready"],
            insufficient_count=totals["insufficient"],
            fallback_count=totals["fallback"],
            abstain_count=totals["abstain"],
            execution_failed_count=totals["failed"],
            lineage_rejected_count=totals["lineage_rejected"],
            query_vector_unavailable_count=totals["query_vector_unavailable"],
        )

    def _stage_job(self, run_id: str, compiler_unit_id: str) -> CompilerStageJob:
        return CompilerStageJob(
            stage_job_id=stable_compiler_id(
                "compiler-stage-job", [run_id, compiler_unit_id, "dry_run", 0]
            ),
            run_id=run_id,
            compiler_unit_id=compiler_unit_id,
            stage=CompilerStageKind.DRY_RUN,
            content_hash=canonical_content_hash(
                [run_id, compiler_unit_id, "dry_run", 0, self.config.dry_run_policy]
            ),
            max_attempts=2,
        )

    async def _process_unit(
        self,
        *,
        job: CompilerStageJob,
        unit: CompilationUnit,
        programs: Sequence[MemoryProgram],
        candidates: Mapping[str, QuestionCandidate],
        expected: Mapping[str, Sequence[OperatorEvidence]],
        memory_index: MemoryOperatorIndexManifest,
        query_vectors: Mapping[str, ProgramQueryVectorSet],
    ) -> dict[str, int]:
        started_at = utc_now()
        counts: defaultdict[str, int] = defaultdict(int)
        visible_memories = [
            item
            async for item in PostgresMemoryRepository(self.sessions).iter_visible(
                unit.snapshot_id,
                unit.conversation_id,
                unit.available_after_session,
            )
        ]
        lineage = EvidenceLineageResolver(visible_memories)
        for program in sorted(programs, key=lambda item: item.qbank_id):
            candidate = candidates.get(program.qbank_id)
            vectors = query_vectors.get(program.qbank_id)
            if candidate is None:
                raise RuntimeError("persisted program is missing its Questioner candidate")
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.planner is not None
                assert uow.program_dry_runs is not None
                persisted = await uow.program_dry_runs.get(program.plan_id)
            if persisted is not None:
                status = dry_run_outcome_status(persisted[0])
                await self._ensure_outcome(
                    run_id=job.run_id,
                    unit_id=unit.compiler_unit_id,
                    candidate=candidate,
                    program=program,
                    status=status,
                    reason_code=persisted[0].failure_code or status.value,
                )
                self._count_status(counts, status, executed=True)
                continue
            if vectors is None:
                await self._ensure_outcome(
                    run_id=job.run_id,
                    unit_id=unit.compiler_unit_id,
                    candidate=candidate,
                    program=program,
                    status=PlannerOutcomeStatus.QUERY_VECTOR_UNAVAILABLE,
                    reason_code="program_query_vector_unavailable",
                )
                self._count_status(
                    counts,
                    PlannerOutcomeStatus.QUERY_VECTOR_UNAVAILABLE,
                    executed=False,
                )
                continue
            execution_id = stable_compiler_id("program-execution", [job.run_id, program.plan_id, 0])
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.program_executions is not None
                execution = await uow.program_executions.get(execution_id)
            data = SessionScopedOperatorDataRepository(self.sessions)
            registry = build_operator_registry(data, self.config)
            validator = ProgramValidator(registry, self.config)
            lineage_validator = ProgramLineageValidator(registry, data)
            context = OperatorExecutionContext(
                run_id=job.run_id,
                source_compiler_run_id=job.run_id,
                snapshot_id=unit.snapshot_id,
                conversation_id=unit.conversation_id,
                cutoff_session_id=unit.available_after_session,
                compiler_unit_id=unit.compiler_unit_id,
                question=candidate.canonical_question,
                required_modalities=candidate.required_modalities,
                collection_bindings={
                    "text_memory": memory_index.text_collection_id,
                    "visual_memory": memory_index.visual_collection_id,
                },
                query_vectors={
                    "question_text_query": vectors.text_vector,
                    "question_joint_query": vectors.joint_vector,
                },
                metadata={"round_id": 0},
            )
            try:
                await lineage_validator.validate(program, context)
            except Exception as exc:
                issue = ProgramPolicyIssue(
                    code="program_lineage_rejected",
                    message="program failed database lineage validation",
                    metadata={"error_type": type(exc).__name__},
                )
                lineage_fact = _validation_fact(
                    run_id=job.run_id,
                    plan_id=program.plan_id,
                    kind=ProgramValidationKind.LINEAGE,
                    validator_version="program-lineage-v2",
                    valid=False,
                    issues=[issue],
                )
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.planner is not None
                    existing = await uow.planner.get_validation(
                        program.plan_id, ProgramValidationKind.LINEAGE
                    )
                    if existing is None:
                        await uow.planner.add_validation(lineage_fact)
                await self._ensure_outcome(
                    run_id=job.run_id,
                    unit_id=unit.compiler_unit_id,
                    candidate=candidate,
                    program=program,
                    status=PlannerOutcomeStatus.LINEAGE_REJECTED,
                    reason_code="program_lineage_rejected",
                )
                self._count_status(
                    counts,
                    PlannerOutcomeStatus.LINEAGE_REJECTED,
                    executed=False,
                )
                continue
            if execution is None:
                executor = LinearProgramExecutor(registry, validator, data, self.config)
                execution = await executor.execute(
                    execution_id=execution_id,
                    program=program,
                    context=context,
                    purpose=OperatorExecutionPurpose.COMPILER_DRY_RUN,
                )
            lineage_fact = _validation_fact(
                run_id=job.run_id,
                plan_id=program.plan_id,
                kind=ProgramValidationKind.LINEAGE,
                validator_version="program-lineage-v2",
                valid=True,
                issues=[],
            )
            assessment, matches = assess_program_dry_run(
                candidate=candidate,
                expected_evidence=expected.get(candidate.qbank_id, ()),
                result=execution,
                lineage=lineage,
                require_expected_complete=(
                    self.config.operator_set_version in {"pmmc-operators-v6", "pmmc-operators-v7"}
                ),
            )
            status = dry_run_outcome_status(assessment)
            outcome = planner_terminal_outcome(
                run_id=job.run_id,
                compiler_unit_id=unit.compiler_unit_id,
                qbank_id=candidate.qbank_id,
                plan_id=program.plan_id,
                status=status,
                reason_code=assessment.failure_code or status.value,
            )
            try:
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.planner is not None
                    assert uow.program_executions is not None
                    assert uow.program_dry_runs is not None
                    existing_lineage = await uow.planner.get_validation(
                        program.plan_id, ProgramValidationKind.LINEAGE
                    )
                    if existing_lineage is None:
                        await uow.planner.add_validation(lineage_fact)
                    elif not existing_lineage.valid:
                        raise RuntimeError("persisted program lineage validation conflicts")
                    await uow.program_executions.add(execution)
                    await uow.program_dry_runs.add(assessment, list(matches))
                    existing_outcome = await uow.planner.get_outcome(candidate.qbank_id)
                    if existing_outcome is None:
                        await uow.planner.add_outcome(outcome)
                    elif (
                        existing_outcome.status is not status
                        or existing_outcome.plan_id != program.plan_id
                    ):
                        raise RuntimeError("persisted Planner terminal outcome conflicts")
            except ProgramDryRunPersistenceError:
                raise
            except Exception as exc:
                raise ProgramDryRunPersistenceError("execution_bundle", exc) from exc
            self._count_status(counts, status, executed=True)

        normalized_counts = _normalized_dry_run_counts(counts)
        completed_at = utc_now()
        stage = CompilationStageResult(
            stage_result_id=stable_compiler_id(
                "compiler-stage", [job.run_id, unit.compiler_unit_id, "dry_run", 0]
            ),
            run_id=job.run_id,
            compiler_unit_id=unit.compiler_unit_id,
            stage=CompilationStage.DRY_RUN,
            attempt_number=max(1, job.attempt_count),
            status=TraceStatus.SUCCESS,
            produced_count=len(programs),
            accepted_count=(
                normalized_counts["ready"]
                + normalized_counts["fallback"]
                + normalized_counts["abstain"]
            ),
            rejected_count=(
                normalized_counts["insufficient"]
                + normalized_counts["failed"]
                + normalized_counts["lineage_rejected"]
                + normalized_counts["query_vector_unavailable"]
            ),
            result_hash=canonical_content_hash(
                [job.run_id, unit.compiler_unit_id, normalized_counts]
            ),
            started_at=started_at,
            completed_at=completed_at,
            elapsed_ms=max(0.0, (completed_at - started_at).total_seconds() * 1000),
            metadata={"round_id": 0},
        )
        try:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.compiler_units is not None
                await uow.compiler_units.add_stage_result(stage)
        except ProgramDryRunPersistenceError:
            raise
        except Exception as exc:
            raise ProgramDryRunPersistenceError("stage_result", exc) from exc
        return normalized_counts

    async def _ensure_outcome(
        self,
        *,
        run_id: str,
        unit_id: str,
        candidate: QuestionCandidate,
        program: MemoryProgram,
        status: PlannerOutcomeStatus,
        reason_code: str,
    ) -> None:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.planner is not None
            existing = await uow.planner.get_outcome(candidate.qbank_id)
            if existing is not None:
                if existing.status is not status or existing.plan_id != program.plan_id:
                    raise RuntimeError("persisted Planner terminal outcome conflicts")
                return
            await uow.planner.add_outcome(
                planner_terminal_outcome(
                    run_id=run_id,
                    compiler_unit_id=unit_id,
                    qbank_id=candidate.qbank_id,
                    plan_id=program.plan_id,
                    status=status,
                    reason_code=reason_code,
                )
            )

    @staticmethod
    def _count_status(
        counts: dict[str, int],
        status: PlannerOutcomeStatus,
        *,
        executed: bool,
    ) -> None:
        if executed:
            counts["executions"] += 1
        if status is PlannerOutcomeStatus.PROGRAM_READY:
            counts["ready"] += 1
        elif status is PlannerOutcomeStatus.DRY_RUN_INSUFFICIENT:
            counts["insufficient"] += 1
        elif status is PlannerOutcomeStatus.PROGRAM_READY_WITH_FALLBACK:
            counts["fallback"] += 1
        elif status is PlannerOutcomeStatus.PROGRAM_READY_WITH_ABSTAIN:
            counts["abstain"] += 1
        elif status is PlannerOutcomeStatus.EXECUTION_FAILED:
            counts["failed"] += 1
        elif status is PlannerOutcomeStatus.LINEAGE_REJECTED:
            counts["lineage_rejected"] += 1
        elif status is PlannerOutcomeStatus.QUERY_VECTOR_UNAVAILABLE:
            counts["query_vector_unavailable"] += 1


async def execute_program_round(
    *,
    sessions: async_sessionmaker[AsyncSession],
    config: CompilerFoundationConfig,
    run_id: str,
    unit: CompilationUnit,
    candidate: QuestionCandidate,
    program: MemoryProgram,
    expected_evidence: Sequence[OperatorEvidence],
    memory_index: MemoryOperatorIndexManifest,
    query_vectors: ProgramQueryVectorSet,
) -> ProgramDryRunAssessment:
    """Execute and persist one revision-round program without mutating r0 outcomes."""

    if program.run_id != run_id or program.compiler_unit_id != unit.compiler_unit_id:
        raise ValueError("revision program is outside the frozen candidate scope")
    if program.qbank_id != candidate.qbank_id or query_vectors.qbank_id != candidate.qbank_id:
        raise ValueError("revision dry-run candidate scope is inconsistent")
    if memory_index.run_id != run_id or memory_index.status != "ready":
        raise ValueError("revision dry-run requires the run's ready memory index")
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.program_dry_runs is not None
        assert uow.program_executions is not None
        persisted = await uow.program_dry_runs.get(program.plan_id)
        if persisted is not None:
            return persisted[0]
        execution_id = stable_compiler_id(
            "program-execution", [run_id, program.plan_id, program.round_id]
        )
        execution = await uow.program_executions.get(execution_id)

    visible_memories = [
        item
        async for item in PostgresMemoryRepository(sessions).iter_visible(
            unit.snapshot_id,
            unit.conversation_id,
            unit.available_after_session,
        )
    ]
    lineage = EvidenceLineageResolver(visible_memories)
    data = SessionScopedOperatorDataRepository(sessions)
    registry = build_operator_registry(data, config)
    validator = ProgramValidator(registry, config)
    lineage_validator = ProgramLineageValidator(registry, data)
    context = OperatorExecutionContext(
        run_id=run_id,
        source_compiler_run_id=run_id,
        snapshot_id=unit.snapshot_id,
        conversation_id=unit.conversation_id,
        cutoff_session_id=unit.available_after_session,
        compiler_unit_id=unit.compiler_unit_id,
        question=candidate.canonical_question,
        required_modalities=candidate.required_modalities,
        collection_bindings={
            "text_memory": memory_index.text_collection_id,
            "visual_memory": memory_index.visual_collection_id,
        },
        query_vectors={
            "question_text_query": query_vectors.text_vector,
            "question_joint_query": query_vectors.joint_vector,
        },
        metadata={"round_id": program.round_id},
    )
    await lineage_validator.validate(program, context)
    if execution is None:
        execution = await LinearProgramExecutor(
            registry, validator, data, config
        ).execute(
            execution_id=execution_id,
            program=program,
            context=context,
            purpose=OperatorExecutionPurpose.COMPILER_DRY_RUN,
        )
    assessment, matches = assess_program_dry_run(
        candidate=candidate,
        expected_evidence=expected_evidence,
        result=execution,
        lineage=lineage,
        require_expected_complete=(
            config.operator_set_version in {"pmmc-operators-v6", "pmmc-operators-v7"}
        ),
    )
    lineage_fact = _validation_fact(
        run_id=run_id,
        plan_id=program.plan_id,
        kind=ProgramValidationKind.LINEAGE,
        validator_version="program-lineage-v3",
        valid=True,
        issues=[],
    )
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.planner is not None
        assert uow.program_executions is not None
        assert uow.program_dry_runs is not None
        prior = await uow.planner.get_validation(
            program.plan_id, ProgramValidationKind.LINEAGE
        )
        if prior is None:
            await uow.planner.add_validation(lineage_fact)
        elif not prior.valid:
            raise RuntimeError("revision program has conflicting lineage validation")
        await uow.program_executions.add(execution)
        await uow.program_dry_runs.add(assessment, list(matches))
    return assessment
