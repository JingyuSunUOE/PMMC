"""Recoverable Planner v1 producer for Questioner-eligible candidates."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.planner import (
    PlannerProgramPolicy,
    PlannerStrategyPolicyError,
    build_planner_requirement_contract,
    materialize_planner_program,
    materialize_planner_strategy_program,
    planner_terminal_outcome,
)
from pmmc.compiler.planner_config import PlannerConfig
from pmmc.compiler.planner_prompt import (
    PlannerEvidenceView,
    planner_output_parser,
    planner_prompt_sha256,
    planner_response_schema_sha256,
    render_planner_request,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import (
    PostgresImageRepository,
    PostgresMemoryRepository,
)
from pmmc.memory.operators import ProgramValidator
from pmmc.models import GenerationGateway, ModelExecutionService
from pmmc.models.execution import ModelCallContext
from pmmc.schemas import (
    AppStage,
    CompilationStage,
    CompilationStageResult,
    CompilationUnit,
    CompilerStageJob,
    CompilerStageJobStatus,
    CompilerStageKind,
    DataScope,
    OperatorEvidence,
    PlannerBatchProgram,
    PlannerCandidateDraftFact,
    PlannerCandidateOutcome,
    PlannerGenerationBatch,
    PlannerGenerationEnvelope,
    PlannerOutcomeStatus,
    PlannerStrategyDraftV12,
    PlannerTopKTier,
    ProgramPolicyIssue,
    ProgramPolicyResult,
    ProgramPolicySeverity,
    ProgramValidationIssueFact,
    ProgramValidationKind,
    ProgramValidationResultFact,
    QBankRouteClass,
    QuestionCandidate,
    StructuredOutputItemDiagnostic,
    StructuredOutputItemStatus,
    StructuredOutputStage,
    StructuredOutputValidationError,
    TraceStatus,
)
from pmmc.schemas.base import utc_now

PlannerItemCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PlannerRunResult:
    run_id: str
    unit_count: int
    candidate_count: int
    program_count: int
    rejected_count: int
    reused_unit_count: int
    outcome_counts: dict[str, int]


def _safe_batch_diagnostics(
    payload: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    values = payload.get("invalid_diagnostics", [])
    if not isinstance(values, list):
        return [], {}
    diagnostics: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        ordinal = value.get("candidate_ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            continue
        raw_errors = value.get("errors", [])
        if not isinstance(raw_errors, list):
            raw_errors = []
        errors: list[dict[str, str]] = []
        for raw_error in raw_errors:
            if not isinstance(raw_error, Mapping):
                continue
            location = raw_error.get("location")
            error_type = raw_error.get("type")
            if not isinstance(location, str) or not isinstance(error_type, str):
                continue
            if not location or not error_type or len(location) > 256 or len(error_type) > 128:
                continue
            if not all(char.isalnum() or char in "._" for char in location):
                continue
            if not all(char.isalnum() or char in "._-" for char in error_type):
                continue
            errors.append({"location": location, "type": error_type})
            counts[f"{location}|{error_type}"] += 1
        diagnostics.append(
            {
                "candidate_ordinal": ordinal,
                "errors": sorted(errors, key=lambda item: (item["location"], item["type"])),
            }
        )
    diagnostics.sort(key=lambda item: cast(int, item["candidate_ordinal"]))
    return diagnostics, dict(sorted(counts.items()))


def _persistable_validation_errors(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    fallback_type: str,
) -> list[StructuredOutputValidationError]:
    """Preserve field-level safe diagnostics without provider values."""

    errors: list[StructuredOutputValidationError] = []
    seen: set[tuple[str, str]] = set()
    for diagnostic in diagnostics:
        values = diagnostic.get("errors", [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            location = value.get("location")
            error_type = value.get("type")
            if not isinstance(location, str) or not isinstance(error_type, str):
                continue
            key = (location, error_type)
            if key in seen:
                continue
            seen.add(key)
            errors.append(
                StructuredOutputValidationError(
                    location=location,
                    error_type=error_type,
                )
            )
            if len(errors) == 16:
                return errors
    return errors or [StructuredOutputValidationError(location="$", error_type=fallback_type)]


def _allocate_diagnostic_ordinal(
    used: set[int],
    *,
    preferred: int,
    overflow: int,
) -> tuple[int, int]:
    if preferred not in used:
        used.add(preferred)
        return preferred, overflow
    while overflow in used:
        overflow += 1
    used.add(overflow)
    return overflow, overflow + 1


class PostgresPlannerInputProvider:
    """Build evidence-scoped Planner requests from canonical repositories."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions
        self.memories = PostgresMemoryRepository(sessions)
        self.images = PostgresImageRepository(sessions)

    async def evidence_views(
        self,
        *,
        unit: CompilationUnit,
        candidates: Sequence[tuple[str, QuestionCandidate]],
        expected: Mapping[str, Sequence[OperatorEvidence]],
    ) -> dict[str, list[PlannerEvidenceView]]:
        canonical = await self.memories.list_snapshot(
            unit.snapshot_id, conversation_ids={unit.conversation_id}
        )
        memory_by_id = {item.memory_id: item for item in canonical}
        result: dict[str, list[PlannerEvidenceView]] = {}
        for candidate_alias, candidate in candidates:
            evidence = list(expected.get(candidate.qbank_id, ()))
            image_ids = list(
                dict.fromkeys(item.image_id for item in evidence if item.image_id is not None)
            )
            image_by_id = {
                item.image_id: item
                for item in await self.images.get_many(unit.snapshot_id, image_ids)
            }
            views: list[PlannerEvidenceView] = []
            for ordinal, item in enumerate(evidence, start=1):
                memory = memory_by_id.get(item.memory_id or "")
                text = None
                if memory is not None:
                    text = memory.text or memory.caption
                views.append(
                    PlannerEvidenceView(
                        alias=f"E{ordinal:02d}",
                        evidence=item,
                        text=text,
                        image=image_by_id.get(item.image_id or ""),
                    )
                )
            result[candidate_alias] = views
        return result


def _validation_fact(
    *,
    run_id: str,
    plan_id: str,
    kind: ProgramValidationKind,
    validator_version: str,
    valid: bool,
    issues: Sequence[ProgramPolicyIssue],
) -> ProgramValidationResultFact:
    result_id = stable_compiler_id(
        "program-validation", [run_id, plan_id, kind.value, validator_version]
    )
    facts = [
        ProgramValidationIssueFact(
            validation_issue_id=stable_compiler_id(
                "program-validation-issue", [result_id, ordinal, issue.code]
            ),
            validation_result_id=result_id,
            run_id=run_id,
            ordinal=ordinal,
            code=issue.code,
            severity=ProgramPolicySeverity(issue.severity.value),
            step_id=issue.step_id,
            message=issue.message,
            metadata=dict(issue.metadata),
        )
        for ordinal, issue in enumerate(issues)
    ]
    payload = {
        "run_id": run_id,
        "plan_id": plan_id,
        "kind": kind.value,
        "validator": validator_version,
        "valid": valid,
        "issues": [item.model_dump(mode="json") for item in facts],
    }
    return ProgramValidationResultFact(
        validation_result_id=result_id,
        run_id=run_id,
        plan_id=plan_id,
        validation_kind=kind,
        validator_version=validator_version,
        valid=valid,
        issues=facts,
        content_hash=canonical_content_hash(payload),
    )


class PlannerRunner:
    """Produce round-0 programs without holding transactions during model calls."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        gateway: GenerationGateway,
        config: PlannerConfig,
        static_validator: ProgramValidator,
        lease_seconds: int,
        concurrency: int,
        input_provider: PostgresPlannerInputProvider | None = None,
        diagnostic_force_weak_r0: bool = False,
    ) -> None:
        if lease_seconds <= 0 or concurrency <= 0:
            raise ValueError("Planner lease and concurrency must be positive")
        self.sessions = sessions
        self.execution = execution
        self.gateway = gateway
        self.config = config
        self.static_validator = static_validator
        self.policy = PlannerProgramPolicy(
            config,
            registry=static_validator.registry,
        )
        self.lease_seconds = lease_seconds
        self.concurrency = concurrency
        self.input_provider = input_provider or PostgresPlannerInputProvider(sessions)
        if diagnostic_force_weak_r0 and not config.is_v14:
            raise ValueError("diagnostic weak-r0 mode requires Planner v1.4")
        self.diagnostic_force_weak_r0 = diagnostic_force_weak_r0
        self._strategy_policy_rejections: Counter[str] = Counter()
        self._native_output_rejections: Counter[str] = Counter()
        self._native_output_rejected_candidates = 0

    def _record_strategy_policy_rejection(
        self,
        error: PlannerStrategyPolicyError,
        *,
        unit_ordinal: int,
        candidate_ordinal: int,
        round_id: int,
    ) -> None:
        self._strategy_policy_rejections[error.code] += 1
        count = self._strategy_policy_rejections[error.code]
        if count != 1 and count % 50 != 0:
            return
        event = {
            **error.safe_diagnostics(),
            "candidate_ordinal": candidate_ordinal,
            "occurrence_count": count,
            "round_id": round_id,
            "unit_ordinal": unit_ordinal,
        }
        print(
            "PLANNER_POLICY_REJECTION " + json.dumps(event, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def _record_native_output_rejection(
        self,
        reasons: Mapping[str, int],
        *,
        unit_ordinal: int,
        candidate_ordinal: int,
    ) -> None:
        safe_reasons = dict(reasons) or {"planner_strategy_invalid": 1}
        self._native_output_rejected_candidates += 1
        for code, occurrences in safe_reasons.items():
            self._native_output_rejections[code] += max(1, int(occurrences))
        rejected_candidates = self._native_output_rejected_candidates
        if rejected_candidates != 1 and rejected_candidates % 25 != 0:
            return
        print(
            "PLANNER_NATIVE_OUTPUT_REJECTION "
            + json.dumps(
                {
                    "candidate_ordinal": candidate_ordinal,
                    "diagnostic_counts": dict(sorted(safe_reasons.items())),
                    "rejected_candidate_count": rejected_candidates,
                    "unit_ordinal": unit_ordinal,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    async def execute(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        units: Sequence[CompilationUnit],
        on_item: PlannerItemCallback | None = None,
    ) -> PlannerRunResult:
        if not units:
            raise ValueError("Planner run requires compilation units")
        self._strategy_policy_rejections.clear()
        self._native_output_rejections.clear()
        self._native_output_rejected_candidates = 0
        unit_by_id = {item.compiler_unit_id: item for item in units}
        if len(unit_by_id) != len(units):
            raise ValueError("Planner units contain duplicate IDs")
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.questioner is not None
            assert uow.compiler_units is not None
            assert uow.compiler_stage_jobs is not None
            candidates = await uow.questioner.list_eligible_questions(run_id)
            expected = await uow.questioner.list_expected_evidence(run_id)
            by_unit: dict[str, list[QuestionCandidate]] = defaultdict(list)
            for candidate in candidates:
                if candidate.compiler_unit_id not in unit_by_id:
                    raise ValueError("eligible candidate is outside the Planner denominator")
                by_unit[candidate.compiler_unit_id].append(candidate)
            for unit in units:
                questioner_stage = await uow.compiler_units.get_stage_result(
                    unit.compiler_unit_id, CompilationStage.QUESTIONER
                )
                if questioner_stage is None or questioner_stage.status is not TraceStatus.SUCCESS:
                    raise ValueError("Planner requires a successful Questioner stage per unit")
                stage_job = self._stage_job(run_id, unit.compiler_unit_id)
                existing = await uow.compiler_stage_jobs.get(stage_job.stage_job_id)
                if existing is None:
                    await uow.compiler_stage_jobs.add(stage_job)
                elif existing.content_hash != stage_job.content_hash:
                    raise ValueError("persisted Planner stage job identity conflicts")
            initial_jobs = await uow.compiler_stage_jobs.list_for_run(
                run_id, CompilerStageKind.PLANNER
            )
            reused_unit_count = sum(
                item.status is CompilerStageJobStatus.SUCCEEDED for item in initial_jobs
            )

        counters = {"programs": 0, "rejected": 0, "reused": 0}
        counter_lock = asyncio.Lock()

        async def worker(index: int) -> None:
            worker_id = f"planner-{index:02d}"
            while True:
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.compiler_stage_jobs is not None
                    claimed = await uow.compiler_stage_jobs.claim(
                        run_id=run_id,
                        stage=CompilerStageKind.PLANNER,
                        worker_id=worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                if claimed is None:
                    return
                unit = unit_by_id[claimed.compiler_unit_id]
                values = sorted(
                    by_unit.get(unit.compiler_unit_id, []),
                    key=lambda item: (item.generation_rank, item.qbank_id),
                )
                try:
                    programs, rejected = await self._process_unit(
                        stage_job=claimed,
                        unit=unit,
                        candidates=values,
                        expected=expected,
                    )
                except Exception as exc:
                    try:
                        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                            assert uow.compiler_stage_jobs is not None
                            await uow.compiler_stage_jobs.finish(
                                claimed.stage_job_id,
                                worker_id=worker_id,
                                succeeded=False,
                                error_type=type(exc).__name__,
                                error_summary="Planner unit processing failed",
                            )
                    except Exception as finish_error:
                        exc.add_note(
                            f"failed to persist Planner job failure: {type(finish_error).__name__}"
                        )
                    raise
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.compiler_stage_jobs is not None
                    await uow.compiler_stage_jobs.finish(
                        claimed.stage_job_id,
                        worker_id=worker_id,
                        succeeded=True,
                    )
                async with counter_lock:
                    counters["programs"] += programs
                    counters["rejected"] += rejected
                if on_item is not None:
                    on_item(unit.compiler_unit_id, programs, rejected)

        tasks = [asyncio.create_task(worker(index)) for index in range(self.concurrency)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_stage_jobs is not None
            jobs = await uow.compiler_stage_jobs.list_for_run(run_id, CompilerStageKind.PLANNER)
        succeeded = sum(item.status is CompilerStageJobStatus.SUCCEEDED for item in jobs)
        if succeeded != len(units):
            raise RuntimeError("Planner did not complete every compiler unit")
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.memory_programs is not None and uow.planner is not None
            persisted_programs = await uow.memory_programs.list_for_run(run_id)
            persisted_outcomes = await uow.planner.list_outcomes(run_id)
        if self._strategy_policy_rejections:
            print(
                "PLANNER_POLICY_SUMMARY "
                + json.dumps(
                    {
                        "counts": dict(sorted(self._strategy_policy_rejections.items())),
                        "scope": "execution_attempt_new",
                        "total": sum(self._strategy_policy_rejections.values()),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        if self.config.is_v12 or self.config.is_v13:
            outcome_counts = Counter(item.status.value for item in persisted_outcomes)
            print(
                "PLANNER_NATIVE_OUTPUT_SUMMARY "
                + json.dumps(
                    {
                        "candidate_count": len(candidates),
                        "diagnostic_counts": dict(sorted(self._native_output_rejections.items())),
                        "parse_failed_count": outcome_counts.get(
                            PlannerOutcomeStatus.PARSE_FAILED.value,
                            0,
                        ),
                        "rejected_candidate_count": self._native_output_rejected_candidates,
                        "persisted_program_count": len(persisted_programs),
                        "reused_unit_count": reused_unit_count,
                        "scope": "execution_attempt_new_diagnostics_with_persisted_totals",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        planner_rejections = {
            PlannerOutcomeStatus.PARSE_FAILED,
            PlannerOutcomeStatus.STATIC_REJECTED,
            PlannerOutcomeStatus.POLICY_REJECTED,
        }
        return PlannerRunResult(
            run_id=run_id,
            unit_count=len(units),
            candidate_count=len(candidates),
            program_count=len(persisted_programs),
            rejected_count=sum(item.status in planner_rejections for item in persisted_outcomes),
            reused_unit_count=reused_unit_count,
            outcome_counts=dict(
                sorted(Counter(item.status.value for item in persisted_outcomes).items())
            ),
        )

    def _stage_job(self, run_id: str, compiler_unit_id: str) -> CompilerStageJob:
        stage_job_id = stable_compiler_id(
            "compiler-stage-job", [run_id, compiler_unit_id, "planner", 0]
        )
        return CompilerStageJob(
            stage_job_id=stage_job_id,
            run_id=run_id,
            compiler_unit_id=compiler_unit_id,
            stage=CompilerStageKind.PLANNER,
            content_hash=canonical_content_hash(
                [run_id, compiler_unit_id, "planner", 0, self.config.protocol_version]
            ),
            max_attempts=2,
        )

    async def _load_trace(self, model_call_id: str):  # type: ignore[no-untyped-def]
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            return await uow.model_calls.get(model_call_id)

    async def _process_unit(
        self,
        *,
        stage_job: CompilerStageJob,
        unit: CompilationUnit,
        candidates: Sequence[QuestionCandidate],
        expected: Mapping[str, Sequence[OperatorEvidence]],
    ) -> tuple[int, int]:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_units is not None
            completed = await uow.compiler_units.get_stage_result(
                unit.compiler_unit_id, CompilationStage.PLANNER
            )
        if completed is not None:
            if completed.status is not TraceStatus.SUCCESS:
                raise RuntimeError("persisted Planner stage is not successful")
            return completed.accepted_count, completed.rejected_count
        if self.config.is_v12 or self.config.is_v13:
            return await self._process_unit_v12(
                stage_job=stage_job,
                unit=unit,
                candidates=candidates,
                expected=expected,
            )

        started_at = utc_now()
        program_count = 0
        rejected_count = 0
        for batch_ordinal, offset in enumerate(
            range(0, len(candidates), self.config.candidates_per_batch)
        ):
            batch_candidates = list(candidates[offset : offset + self.config.candidates_per_batch])
            aliases = [f"Q{offset + index + 1:02d}" for index in range(len(batch_candidates))]
            paired = list(zip(aliases, batch_candidates, strict=True))
            contracts = {
                alias: build_planner_requirement_contract(
                    run_id=stage_job.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    candidate=candidate,
                )
                for alias, candidate in paired
            }
            if self.config.is_v11:
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.planner is not None
                    for contract in contracts.values():
                        await uow.planner.add_requirement_contract(contract)
            model_call_id = stable_compiler_id(
                "planner-model-call",
                [stage_job.run_id, unit.compiler_unit_id, batch_ordinal, aliases],
            )
            trace = await self._load_trace(model_call_id)
            if trace is None:
                views = (
                    {}
                    if self.config.is_v11
                    else await self.input_provider.evidence_views(
                        unit=unit,
                        candidates=[(alias, candidate) for alias, candidate in paired],
                        expected=expected,
                    )
                )
                request = render_planner_request(
                    request_id=model_call_id,
                    config=self.config,
                    candidates=[(alias, candidate) for alias, candidate in paired],
                    evidence_views=views,
                    requirement_contracts=contracts,
                )
                audited = await self.execution.generate_audited(
                    self.gateway,
                    request,
                    ModelCallContext(
                        model_call_id=model_call_id,
                        run_id=stage_job.run_id,
                        snapshot_id=unit.snapshot_id,
                        conversation_id=unit.conversation_id,
                        job_id=stable_compiler_id(
                            "compile-job",
                            [stage_job.run_id, unit.compiler_unit_id, "questions"],
                        ),
                        compiler_unit_id=unit.compiler_unit_id,
                        stage=AppStage.PLANNER,
                    ),
                    output_parser=planner_output_parser(
                        expected_aliases=aliases,
                        config=self.config,
                    ),
                )
                trace = audited.trace
            if trace.status is not TraceStatus.SUCCESS or trace.parsed_output is None:
                raise RuntimeError("Planner model call did not produce recoverable output")
            payload = dict(trace.parsed_output)
            envelope = PlannerGenerationEnvelope.model_validate(
                {"programs": payload.get("programs", [])}
            )
            draft_by_alias = {item.candidate_alias: item for item in envelope.programs}
            invalid = {
                item
                for item in payload.get("invalid_candidate_aliases", [])
                if isinstance(item, str)
            }
            diagnostics, diagnostic_counts = _safe_batch_diagnostics(payload)
            unexpected_alias_count = payload.get("unexpected_alias_count", 0)
            if (
                not isinstance(unexpected_alias_count, int)
                or isinstance(unexpected_alias_count, bool)
                or unexpected_alias_count < 0
            ):
                unexpected_alias_count = 0
            batch = PlannerGenerationBatch(
                planner_batch_id=stable_compiler_id(
                    "planner-generation-batch",
                    [stage_job.run_id, unit.compiler_unit_id, batch_ordinal],
                ),
                run_id=stage_job.run_id,
                compiler_unit_id=unit.compiler_unit_id,
                stage_job_id=stage_job.stage_job_id,
                model_call_id=trace.model_call_id,
                batch_ordinal=batch_ordinal,
                candidate_qbank_ids=[item.qbank_id for item in batch_candidates],
                parsed_count=len(draft_by_alias),
                invalid_count=len(batch_candidates) - len(draft_by_alias),
                prompt_hash=planner_prompt_sha256(self.config),
                schema_hash=planner_response_schema_sha256(self.config),
                content_hash=canonical_content_hash(payload),
                created_at=trace.created_at,
                metadata={
                    "repair_attempt_count": trace.repair_attempt_count,
                    "parse_error": (
                        trace.parse_error
                        if trace.parse_error
                        in {
                            "planner_json_repaired",
                            "planner_json_invalid",
                            "planner_envelope_invalid",
                        }
                        else ("parser_error_present" if trace.parse_error else None)
                    ),
                    "invalid_diagnostic_count": len(diagnostics),
                    "invalid_diagnostics": diagnostics,
                    "invalid_diagnostic_counts": diagnostic_counts,
                    "unexpected_alias_count": unexpected_alias_count,
                },
            )
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.planner is not None
                await uow.planner.add_batch(batch)
                used_diagnostic_ordinals: set[int] = set()
                overflow_diagnostic_ordinal = len(batch_candidates) + 1

                for diagnostic in diagnostics:
                    source_ordinal = cast(int, diagnostic["candidate_ordinal"])
                    ordinal, overflow_diagnostic_ordinal = _allocate_diagnostic_ordinal(
                        used_diagnostic_ordinals,
                        preferred=source_ordinal,
                        overflow=overflow_diagnostic_ordinal,
                    )
                    raw_errors = cast(list[dict[str, str]], diagnostic.get("errors", []))
                    errors = [
                        StructuredOutputValidationError(
                            location=item["location"],
                            error_type=item["type"],
                        )
                        for item in raw_errors
                    ] or [
                        StructuredOutputValidationError(
                            location="$", error_type="planner_item_invalid"
                        )
                    ]
                    await uow.structured_output_diagnostics.add(
                        StructuredOutputItemDiagnostic(
                            diagnostic_id=stable_compiler_id(
                                "structured-output-diagnostic",
                                [trace.model_call_id, "planner", ordinal],
                            ),
                            run_id=stage_job.run_id,
                            model_call_id=trace.model_call_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            stage=StructuredOutputStage.PLANNER,
                            item_ordinal=ordinal,
                            item_hash=canonical_content_hash(
                                [
                                    trace.model_call_id,
                                    ordinal,
                                    "planner_invalid",
                                ]
                            ),
                            status=StructuredOutputItemStatus.INVALID,
                            parser_version=self.config.parser_version,
                            errors=errors,
                            created_at=trace.created_at,
                            metadata={"source_item_ordinal": source_ordinal},
                        )
                    )
                if trace.repair_attempt_count:
                    source_ordinal = len(batch_candidates) + 1
                    ordinal, overflow_diagnostic_ordinal = _allocate_diagnostic_ordinal(
                        used_diagnostic_ordinals,
                        preferred=source_ordinal,
                        overflow=overflow_diagnostic_ordinal,
                    )
                    await uow.structured_output_diagnostics.add(
                        StructuredOutputItemDiagnostic(
                            diagnostic_id=stable_compiler_id(
                                "structured-output-diagnostic",
                                [trace.model_call_id, "planner", ordinal],
                            ),
                            run_id=stage_job.run_id,
                            model_call_id=trace.model_call_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            stage=StructuredOutputStage.PLANNER,
                            item_ordinal=ordinal,
                            item_hash=batch.content_hash,
                            status=StructuredOutputItemStatus.REPAIRED,
                            parser_version=self.config.parser_version,
                            repair_attempt_count=trace.repair_attempt_count,
                            errors=[
                                StructuredOutputValidationError(
                                    location="$", error_type="json_repaired"
                                )
                            ],
                            created_at=trace.created_at,
                            metadata={"source_item_ordinal": source_ordinal},
                        )
                    )
            for ordinal, (alias, candidate) in enumerate(paired):
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.planner is not None
                    persisted_outcome = await uow.planner.get_outcome(candidate.qbank_id)
                if persisted_outcome is not None:
                    if persisted_outcome.run_id != stage_job.run_id:
                        raise RuntimeError("persisted Planner outcome belongs to another run")
                    if persisted_outcome.plan_id is None:
                        rejected_count += 1
                    else:
                        program_count += 1
                    continue

                draft = draft_by_alias.get(alias)
                draft_fact = PlannerCandidateDraftFact(
                    planner_candidate_draft_id=stable_compiler_id(
                        "planner-candidate-draft",
                        [batch.planner_batch_id, candidate.qbank_id],
                    ),
                    run_id=stage_job.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    planner_batch_id=batch.planner_batch_id,
                    qbank_id=candidate.qbank_id,
                    requirement_contract_id=contracts[alias].requirement_contract_id,
                    item_ordinal=ordinal,
                    item_hash=canonical_content_hash(
                        draft.model_dump(mode="json") if draft is not None else alias
                    ),
                    parsed=draft is not None and alias not in invalid,
                    program_hash=(
                        canonical_content_hash(draft.model_dump(mode="json"))
                        if draft is not None and alias not in invalid
                        else None
                    ),
                    content_hash=canonical_content_hash(
                        [
                            batch.planner_batch_id,
                            candidate.qbank_id,
                            draft.model_dump(mode="json") if draft is not None else None,
                            alias in invalid,
                        ]
                    ),
                    created_at=trace.created_at,
                    metadata={
                        "provider_schema_conformant": draft is not None and alias not in invalid
                    },
                )
                if self.config.is_v11:
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.planner is not None
                        await uow.planner.add_candidate_draft(draft_fact)
                if draft is None or alias in invalid:
                    await self._persist_outcome(
                        planner_terminal_outcome(
                            run_id=stage_job.run_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            qbank_id=candidate.qbank_id,
                            planner_batch_id=batch.planner_batch_id,
                            status=PlannerOutcomeStatus.PARSE_FAILED,
                            reason_code="planner_output_invalid",
                        )
                    )
                    rejected_count += 1
                    continue
                program = materialize_planner_program(
                    run_id=stage_job.run_id,
                    snapshot_id=unit.snapshot_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    producer_model_call_id=trace.model_call_id,
                    candidate=candidate,
                    draft=draft,
                    config=self.config,
                )
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.memory_programs is not None
                    persisted_program = await uow.memory_programs.get(program.plan_id)
                if persisted_program is not None:
                    if persisted_program.program_hash != program.program_hash:
                        raise RuntimeError("persisted Planner program content conflicts")
                    program = persisted_program
                static = self.static_validator.validate(program)
                if not static.valid:
                    await self._persist_outcome(
                        planner_terminal_outcome(
                            run_id=stage_job.run_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            qbank_id=candidate.qbank_id,
                            planner_batch_id=batch.planner_batch_id,
                            status=PlannerOutcomeStatus.STATIC_REJECTED,
                            reason_code="static_validation_failed",
                            metadata={"issue_codes": [item.code for item in static.issues]},
                        )
                    )
                    rejected_count += 1
                    continue
                policy: ProgramPolicyResult = self.policy.validate(
                    program,
                    candidate,
                    contracts[alias] if self.config.is_v11 else None,
                )
                if not policy.valid:
                    await self._persist_outcome(
                        planner_terminal_outcome(
                            run_id=stage_job.run_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            qbank_id=candidate.qbank_id,
                            planner_batch_id=batch.planner_batch_id,
                            status=PlannerOutcomeStatus.POLICY_REJECTED,
                            reason_code="planner_policy_failed",
                            metadata={"issue_codes": [item.code for item in policy.issues]},
                        )
                    )
                    rejected_count += 1
                    continue
                static_issues = [
                    ProgramPolicyIssue(
                        code=item.code,
                        message=item.message,
                        severity=ProgramPolicySeverity(item.severity.value),
                        step_id=item.step_id,
                        metadata=dict(item.metadata),
                    )
                    for item in static.issues
                ]
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.memory_programs is not None and uow.planner is not None
                    await uow.memory_programs.add(
                        program,
                        expected_evidence=list(expected.get(candidate.qbank_id, ())),
                    )
                    await uow.planner.add_program_link(
                        PlannerBatchProgram(
                            planner_batch_id=batch.planner_batch_id,
                            run_id=stage_job.run_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            qbank_id=candidate.qbank_id,
                            plan_id=program.plan_id,
                            ordinal=ordinal,
                            content_hash=canonical_content_hash(
                                [batch.planner_batch_id, candidate.qbank_id, program.plan_id]
                            ),
                        )
                    )
                    static_fact = _validation_fact(
                        run_id=stage_job.run_id,
                        plan_id=program.plan_id,
                        kind=ProgramValidationKind.STATIC,
                        validator_version="memory-program-static-v2",
                        valid=True,
                        issues=static_issues,
                    )
                    policy_fact = _validation_fact(
                        run_id=stage_job.run_id,
                        plan_id=program.plan_id,
                        kind=ProgramValidationKind.POLICY,
                        validator_version=self.config.policy_version,
                        valid=True,
                        issues=policy.issues,
                    )
                    for fact in (static_fact, policy_fact):
                        existing = await uow.planner.get_validation(
                            program.plan_id, fact.validation_kind
                        )
                        if existing is None:
                            await uow.planner.add_validation(fact)
                        elif existing.content_hash != fact.content_hash or not existing.valid:
                            raise RuntimeError("persisted Planner validation content conflicts")
                program_count += 1

        completed_at = utc_now()
        stage = CompilationStageResult(
            stage_result_id=stable_compiler_id(
                "compiler-stage", [stage_job.run_id, unit.compiler_unit_id, "planner", 0]
            ),
            run_id=stage_job.run_id,
            compiler_unit_id=unit.compiler_unit_id,
            stage=CompilationStage.PLANNER,
            attempt_number=max(1, stage_job.attempt_count),
            status=TraceStatus.SUCCESS,
            produced_count=len(candidates),
            accepted_count=program_count,
            rejected_count=rejected_count,
            result_hash=canonical_content_hash(
                [stage_job.run_id, unit.compiler_unit_id, program_count, rejected_count]
            ),
            started_at=started_at,
            completed_at=completed_at,
            elapsed_ms=max(0.0, (completed_at - started_at).total_seconds() * 1000),
            metadata={"round_id": 0},
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_units is not None
            await uow.compiler_units.add_stage_result(stage)
        return program_count, rejected_count

    async def _process_unit_v12(
        self,
        *,
        stage_job: CompilerStageJob,
        unit: CompilationUnit,
        candidates: Sequence[QuestionCandidate],
        expected: Mapping[str, Sequence[OperatorEvidence]],
    ) -> tuple[int, int]:
        """Persist native single-candidate strategies and server-built v4 programs."""

        started_at = utc_now()
        program_count = 0
        rejected_count = 0
        control_count = 0
        for ordinal, candidate in enumerate(candidates):
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.planner is not None
                persisted_outcome = await uow.planner.get_outcome(candidate.qbank_id)
            if persisted_outcome is not None:
                if persisted_outcome.run_id != stage_job.run_id:
                    raise RuntimeError("persisted Planner outcome belongs to another run")
                if persisted_outcome.plan_id is None:
                    rejected_count += 1
                else:
                    program_count += 1
                continue

            try:
                contract = build_planner_requirement_contract(
                    run_id=stage_job.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    candidate=candidate,
                    config=self.config,
                )
            except PlannerStrategyPolicyError as exc:
                await self._persist_outcome(
                    planner_terminal_outcome(
                        run_id=stage_job.run_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        qbank_id=candidate.qbank_id,
                        status=PlannerOutcomeStatus.POLICY_REJECTED,
                        reason_code=exc.code,
                        metadata={
                            "issue_codes": [exc.code],
                            "strategy_policy_diagnostics": exc.safe_diagnostics(),
                            "provider_call_required": False,
                        },
                    )
                )
                self._record_strategy_policy_rejection(
                    exc, unit_ordinal=unit.ordinal, candidate_ordinal=ordinal, round_id=0,
                )
                rejected_count += 1
                continue
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.planner is not None
                await uow.planner.add_requirement_contract(contract)

            batch: PlannerGenerationBatch | None = None
            strategy: PlannerStrategyDraftV12 | None = None
            producer_model_call_id: str | None = None
            native_output = contract.refusal_only
            if contract.refusal_only:
                control_count += 1
            elif self.config.is_v15 and self.config.planning_mode == "fixed":
                strategy = self.config.fixed_strategy
                native_output = True
            else:
                alias = "Q01"
                model_call_id = stable_compiler_id(
                    "planner-model-call",
                    [
                        stage_job.run_id,
                        unit.compiler_unit_id,
                        candidate.qbank_id,
                        0,
                        self.config.protocol_version,
                    ],
                )
                trace = await self._load_trace(model_call_id)
                if trace is None:
                    request = render_planner_request(
                        request_id=model_call_id,
                        config=self.config,
                        candidates=[(alias, candidate)],
                        evidence_views={},
                        requirement_contracts={alias: contract},
                    )
                    audited = await self.execution.generate_audited(
                        self.gateway,
                        request,
                        ModelCallContext(
                            model_call_id=model_call_id,
                            run_id=stage_job.run_id,
                            snapshot_id=unit.snapshot_id,
                            conversation_id=unit.conversation_id,
                            job_id=stable_compiler_id(
                                "compile-job",
                                [stage_job.run_id, unit.compiler_unit_id, "questions"],
                            ),
                            compiler_unit_id=unit.compiler_unit_id,
                            stage=AppStage.PLANNER,
                        ),
                        output_parser=planner_output_parser(
                            expected_aliases=[alias],
                            config=self.config,
                        ),
                    )
                    trace = audited.trace

                payload = (
                    dict(trace.parsed_output)
                    if trace.status is TraceStatus.SUCCESS and trace.parsed_output is not None
                    else {}
                )
                raw_strategies = payload.get("strategies", [])
                if isinstance(raw_strategies, list) and len(raw_strategies) == 1:
                    try:
                        strategy = PlannerStrategyDraftV12.model_validate(raw_strategies[0])
                    except (TypeError, ValueError):
                        strategy = None
                native_output = (
                    strategy is not None
                    and trace.repair_attempt_count == 0
                    and not trace.parse_error
                )
                producer_model_call_id = trace.model_call_id
                diagnostics, diagnostic_counts = _safe_batch_diagnostics(payload)
                batch = PlannerGenerationBatch(
                    planner_batch_id=stable_compiler_id(
                        "planner-generation-batch",
                        [
                            stage_job.run_id,
                            unit.compiler_unit_id,
                            candidate.qbank_id,
                            self.config.protocol_version,
                        ],
                    ),
                    run_id=stage_job.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    stage_job_id=stage_job.stage_job_id,
                    model_call_id=trace.model_call_id,
                    batch_ordinal=ordinal,
                    candidate_qbank_ids=[candidate.qbank_id],
                    parsed_count=int(native_output),
                    invalid_count=int(not native_output),
                    prompt_hash=planner_prompt_sha256(self.config),
                    schema_hash=planner_response_schema_sha256(self.config),
                    content_hash=canonical_content_hash(payload),
                    created_at=trace.created_at,
                    metadata={
                        "single_candidate_scope": True,
                        "provider_schema_conformant": native_output,
                        "repair_attempt_count": trace.repair_attempt_count,
                        "parse_error": ("parser_error_present" if trace.parse_error else None),
                        "invalid_diagnostic_count": len(diagnostics),
                        "invalid_diagnostic_counts": diagnostic_counts,
                    },
                )
                draft_payload = strategy.model_dump(mode="json") if strategy is not None else None
                draft_fact = PlannerCandidateDraftFact(
                    planner_candidate_draft_id=stable_compiler_id(
                        "planner-candidate-draft",
                        [batch.planner_batch_id, candidate.qbank_id],
                    ),
                    run_id=stage_job.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    planner_batch_id=batch.planner_batch_id,
                    qbank_id=candidate.qbank_id,
                    requirement_contract_id=contract.requirement_contract_id,
                    item_ordinal=0,
                    item_hash=canonical_content_hash(
                        draft_payload if draft_payload is not None else alias
                    ),
                    parsed=native_output,
                    program_hash=(canonical_content_hash(draft_payload) if native_output else None),
                    content_hash=canonical_content_hash(
                        [
                            batch.planner_batch_id,
                            candidate.qbank_id,
                            draft_payload,
                            native_output,
                        ]
                    ),
                    created_at=trace.created_at,
                    metadata={
                        "provider_schema_conformant": native_output,
                        "single_candidate_scope": True,
                    },
                )
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.planner is not None
                    await uow.planner.add_batch(batch)
                    await uow.planner.add_candidate_draft(draft_fact)
                    if not native_output:
                        error_type = (
                            "json_repaired"
                            if trace.repair_attempt_count
                            else "planner_strategy_invalid"
                        )
                        await uow.structured_output_diagnostics.add(
                            StructuredOutputItemDiagnostic(
                                diagnostic_id=stable_compiler_id(
                                    "structured-output-diagnostic",
                                    [trace.model_call_id, "planner", 0],
                                ),
                                run_id=stage_job.run_id,
                                model_call_id=trace.model_call_id,
                                compiler_unit_id=unit.compiler_unit_id,
                                stage=StructuredOutputStage.PLANNER,
                                item_ordinal=0,
                                item_hash=canonical_content_hash(
                                    [trace.model_call_id, candidate.qbank_id]
                                ),
                                status=(
                                    StructuredOutputItemStatus.REPAIRED
                                    if trace.repair_attempt_count
                                    else StructuredOutputItemStatus.INVALID
                                ),
                                parser_version=self.config.parser_version,
                                repair_attempt_count=trace.repair_attempt_count,
                                errors=_persistable_validation_errors(
                                    diagnostics,
                                    fallback_type=error_type,
                                ),
                                created_at=trace.created_at,
                                metadata={
                                    "diagnostic_counts": diagnostic_counts,
                                    "single_candidate_scope": True,
                                },
                            )
                        )
                if not native_output:
                    rejection_reasons = dict(diagnostic_counts)
                    if trace.repair_attempt_count:
                        rejection_reasons["json_repaired"] = trace.repair_attempt_count
                    elif trace.parse_error:
                        rejection_reasons["parser_error_present"] = 1
                    elif not raw_strategies and not diagnostic_counts:
                        rejection_reasons["strategy_missing"] = 1
                    self._record_native_output_rejection(
                        rejection_reasons,
                        unit_ordinal=unit.ordinal,
                        candidate_ordinal=ordinal,
                    )
                    await self._persist_outcome(
                        planner_terminal_outcome(
                            run_id=stage_job.run_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            qbank_id=candidate.qbank_id,
                            planner_batch_id=batch.planner_batch_id,
                            status=PlannerOutcomeStatus.PARSE_FAILED,
                            reason_code="planner_strategy_not_native",
                        )
                    )
                    rejected_count += 1
                    continue

                if (
                    self.diagnostic_force_weak_r0
                    and candidate.route_class is QBankRouteClass.ANSWERABLE
                ):
                    assert strategy is not None
                    original_tier = strategy.top_k_tier
                    strategy = strategy.model_copy(update={"top_k_tier": PlannerTopKTier.NARROW})
                    print(
                        "STEP7B_DIAGNOSTIC_WEAK_R0 "
                        + json.dumps(
                            {
                                "candidate_ordinal": ordinal,
                                "original_top_k": original_tier.top_k,
                                "controlled_top_k": strategy.top_k_tier.top_k,
                                "state": "applied",
                                "unit_ordinal": unit.ordinal,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

            try:
                program = materialize_planner_strategy_program(
                    run_id=stage_job.run_id,
                    snapshot_id=unit.snapshot_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    producer_model_call_id=producer_model_call_id,
                    candidate=candidate,
                    requirement=contract,
                    strategy=strategy,
                    config=self.config,
                    round_id=0,
                )
            except PlannerStrategyPolicyError as exc:
                policy_diagnostics = exc.safe_diagnostics()
                await self._persist_outcome(
                    planner_terminal_outcome(
                        run_id=stage_job.run_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        qbank_id=candidate.qbank_id,
                        planner_batch_id=(batch.planner_batch_id if batch is not None else None),
                        status=PlannerOutcomeStatus.POLICY_REJECTED,
                        reason_code="planner_strategy_policy_failed",
                        metadata={
                            "issue_codes": [exc.code],
                            "strategy_policy_diagnostics": policy_diagnostics,
                        },
                    )
                )
                self._record_strategy_policy_rejection(
                    exc,
                    unit_ordinal=unit.ordinal,
                    candidate_ordinal=ordinal,
                    round_id=0,
                )
                rejected_count += 1
                continue
            static = self.static_validator.validate(program)
            if not static.valid:
                await self._persist_outcome(
                    planner_terminal_outcome(
                        run_id=stage_job.run_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        qbank_id=candidate.qbank_id,
                        planner_batch_id=(batch.planner_batch_id if batch is not None else None),
                        status=PlannerOutcomeStatus.STATIC_REJECTED,
                        reason_code="static_validation_failed",
                        metadata={"issue_codes": [item.code for item in static.issues]},
                    )
                )
                rejected_count += 1
                continue
            policy = self.policy.validate(program, candidate, contract)
            if not policy.valid:
                await self._persist_outcome(
                    planner_terminal_outcome(
                        run_id=stage_job.run_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        qbank_id=candidate.qbank_id,
                        planner_batch_id=(batch.planner_batch_id if batch is not None else None),
                        status=PlannerOutcomeStatus.POLICY_REJECTED,
                        reason_code="planner_policy_failed",
                        metadata={"issue_codes": [item.code for item in policy.issues]},
                    )
                )
                rejected_count += 1
                continue

            static_issues = [
                ProgramPolicyIssue(
                    code=item.code,
                    message=item.message,
                    severity=ProgramPolicySeverity(item.severity.value),
                    step_id=item.step_id,
                    metadata=dict(item.metadata),
                )
                for item in static.issues
            ]
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.memory_programs is not None and uow.planner is not None
                await uow.memory_programs.add(
                    program,
                    expected_evidence=list(expected.get(candidate.qbank_id, ())),
                )
                if batch is not None:
                    await uow.planner.add_program_link(
                        PlannerBatchProgram(
                            planner_batch_id=batch.planner_batch_id,
                            run_id=stage_job.run_id,
                            compiler_unit_id=unit.compiler_unit_id,
                            qbank_id=candidate.qbank_id,
                            plan_id=program.plan_id,
                            ordinal=0,
                            content_hash=canonical_content_hash(
                                [batch.planner_batch_id, candidate.qbank_id, program.plan_id]
                            ),
                        )
                    )
                for fact in (
                    _validation_fact(
                        run_id=stage_job.run_id,
                        plan_id=program.plan_id,
                        kind=ProgramValidationKind.STATIC,
                        validator_version=(
                            "memory-program-static-v7"
                            if self.config.is_v15
                            else "memory-program-static-v3"
                        ),
                        valid=True,
                        issues=static_issues,
                    ),
                    _validation_fact(
                        run_id=stage_job.run_id,
                        plan_id=program.plan_id,
                        kind=ProgramValidationKind.POLICY,
                        validator_version=self.config.policy_version,
                        valid=True,
                        issues=policy.issues,
                    ),
                ):
                    existing = await uow.planner.get_validation(
                        program.plan_id, fact.validation_kind
                    )
                    if existing is None:
                        await uow.planner.add_validation(fact)
                    elif existing.content_hash != fact.content_hash or not existing.valid:
                        raise RuntimeError("persisted Planner validation content conflicts")
            program_count += 1

        completed_at = utc_now()
        stage = CompilationStageResult(
            stage_result_id=stable_compiler_id(
                "compiler-stage", [stage_job.run_id, unit.compiler_unit_id, "planner", 0]
            ),
            run_id=stage_job.run_id,
            compiler_unit_id=unit.compiler_unit_id,
            stage=CompilationStage.PLANNER,
            attempt_number=max(1, stage_job.attempt_count),
            status=TraceStatus.SUCCESS,
            produced_count=len(candidates),
            accepted_count=program_count,
            rejected_count=rejected_count,
            result_hash=canonical_content_hash(
                [
                    stage_job.run_id,
                    unit.compiler_unit_id,
                    "planner-v1.2",
                    program_count,
                    rejected_count,
                ]
            ),
            started_at=started_at,
            completed_at=completed_at,
            elapsed_ms=max(0.0, (completed_at - started_at).total_seconds() * 1000),
            metadata={
                "round_id": 0,
                "single_candidate_scope": True,
                "server_control_programs": control_count,
            },
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_units is not None
            await uow.compiler_units.add_stage_result(stage)
        return program_count, rejected_count

    async def _persist_outcome(self, value: PlannerCandidateOutcome) -> None:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.planner is not None
            await uow.planner.add_outcome(value)
