"""Recoverable evidence-grounded Doubter and Planner revision loop."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.config import CompilerFoundationConfig
from pmmc.compiler.doubter import (
    derive_doubter_action,
    derive_doubter_directives,
    derive_doubter_issues,
    derive_dry_run_issues,
    enforce_v13_dimensions,
    normalize_v11_dimensions,
    validate_provider_issue_consistency,
)
from pmmc.compiler.doubter_config import DoubterConfig
from pmmc.compiler.doubter_graph import (
    DoubterControlTrace,
    run_doubter_control_graph,
)
from pmmc.compiler.doubter_prompt import (
    DoubterEvidenceView,
    doubter_output_parser,
    doubter_prompt_sha256,
    render_doubter_request,
)
from pmmc.compiler.ids import (
    canonical_content_hash,
    feedback_content_hash,
    stable_compiler_id,
)
from pmmc.compiler.memory_index_runner import ProgramQueryVectorSet
from pmmc.compiler.planner import (
    PlannerProgramPolicy,
    PlannerStrategyPolicyError,
    materialize_planner_strategy_program,
)
from pmmc.compiler.planner_config import PlannerConfig
from pmmc.compiler.planner_prompt import (
    planner_output_parser,
    planner_response_schema_sha256,
    planner_revision_prompt_sha256,
    render_planner_request,
)
from pmmc.compiler.planner_revision import apply_planner_directives
from pmmc.compiler.planner_runner import _validation_fact
from pmmc.compiler.program_runner import execute_program_round
from pmmc.infrastructure.database.errors import (
    safe_database_error_diagnostics,
    safe_database_error_summary,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.repositories import (
    PostgresImageRepository,
    PostgresMemoryRepository,
)
from pmmc.memory.operators import ProgramExecutionResult, ProgramValidator
from pmmc.models import GenerationGateway, ModelExecutionService, TokenCounter
from pmmc.models.execution import ModelCallContext
from pmmc.models.gateways import ModelGatewayError
from pmmc.schemas import (
    AppStage,
    CandidateStageJob,
    CandidateStageJobStatus,
    CandidateStageKind,
    CompilationStage,
    CompilationStageResult,
    CompilationUnit,
    DataScope,
    DoubterAction,
    DoubterAssessment,
    DoubterDimensions,
    DoubterFeedback,
    DoubterIssueCode,
    DoubterProviderResult,
    DoubterProviderResultV11,
    EvidenceSufficiency,
    GenerationRequest,
    MemoryOperatorIndexManifest,
    MemoryProgram,
    MemoryUnit,
    OperatorEvidence,
    PlannerCapability,
    PlannerRequirementContract,
    PlannerRevisionLineage,
    PlannerRevisionPatch,
    PlannerStrategyDraftV12,
    ProgramDryRunAssessment,
    ProgramPolicyIssue,
    ProgramPolicySeverity,
    ProgramValidationKind,
    QBankRouteClass,
    QuestionCandidate,
    QuestionerAnswerClaim,
    StructuredOutputItemDiagnostic,
    StructuredOutputItemStatus,
    StructuredOutputStage,
    StructuredOutputValidationError,
    TraceStatus,
)
from pmmc.schemas.doubter import PlannerDirectiveEffect

DoubterProgressCallback = Callable[[int, int, int, Mapping[str, int]], None]


@asynccontextmanager
async def _doubter_persistence(
    operation: str, *, round_id: int, unit_ordinal: int | None
) -> AsyncIterator[None]:
    """Preserve safe substage diagnostics across transaction rollback."""
    try:
        yield
    except Exception as exc:
        if getattr(exc, "persistence_operation", None):
            raise
        exc.failure_stage = "doubter_persistence"  # type: ignore[attr-defined]
        exc.persistence_operation = operation  # type: ignore[attr-defined]
        print(
            "DOUBTER_PERSISTENCE_FAILURE "
            + json.dumps(
                {
                    **safe_database_error_diagnostics(exc),
                    "round_id": round_id,
                    **({"unit_ordinal": unit_ordinal} if unit_ordinal is not None else {}),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        raise


def _safe_metadata_count(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _compiler_job_id(run_id: str, compiler_unit_id: str) -> str:
    """Return the Questioner compile job that owns all unit-scoped Q/P/D calls."""

    return stable_compiler_id(
        "compile-job",
        [run_id, compiler_unit_id, "questions"],
    )


def _control_trace_for_candidate(
    *,
    qbank_id: str,
    assessments: Mapping[tuple[str, int], DoubterAssessment],
    max_round: int,
) -> DoubterControlTrace:
    values = [
        assessments[(qbank_id, round_id)]
        for round_id in range(max_round + 1)
        if (qbank_id, round_id) in assessments
    ]
    final = values[-1] if values else None
    return run_doubter_control_graph(
        [item.action for item in values],
        max_round=max_round,
        max_rounds_exhausted=bool(
            final is not None and DoubterIssueCode.MAX_ROUNDS_EXHAUSTED in final.issue_codes
        ),
    )


async def _await_workers(tasks: Sequence[asyncio.Task[None]]) -> None:
    """Cancel peer workers before database resources can be disposed."""

    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass(frozen=True)
class DoubterRunResult:
    run_id: str
    candidate_count: int
    assessment_count: int
    revision_count: int
    passed_count: int
    rejected_count: int
    revision_unavailable_count: int
    exhausted_count: int
    round_counts: dict[int, dict[str, int]]


class DoubterInputGateError(RuntimeError):
    """Safe, aggregate-only failure raised before any Doubter model request."""

    diagnostic_event = "DOUBTER_INPUT_PREFLIGHT_FAILURE"

    def __init__(self, code: str, diagnostics: Mapping[str, object]) -> None:
        self.code = code
        self._diagnostics = {"error_code": code, **dict(diagnostics)}
        self.safe_summary = f"Doubter input preflight failed ({code})"
        super().__init__(self.safe_summary)

    def safe_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)


class DoubterModelRequestError(RuntimeError):
    """Safe request-shape diagnostics for one failed Doubter provider call."""

    diagnostic_event = "DOUBTER_MODEL_REQUEST_FAILURE"

    def __init__(
        self,
        *,
        round_id: int,
        unit_ordinal: int,
        request_ordinal: int,
        request_metadata: Mapping[str, object],
        cause: ModelGatewayError,
    ) -> None:
        self.error_code = "doubter_model_request_failed"
        self.provider_status_code = cause.provider_status_code
        self.retryable = cause.retryable
        allowlisted = {
            key: request_metadata[key]
            for key in (
                "evidence_projection_version",
                "evidence_count",
                "input_image_count",
                "input_unique_image_count",
                "duplicate_image_reference_count",
                "attached_image_count",
                "omitted_image_count",
                "input_text_chars",
                "rendered_text_chars",
                "truncated_text_item_count",
                "estimated_input_tokens",
                "input_tokens",
                "max_input_tokens",
                "token_count_estimated",
                "token_budget_image_reduction_count",
            )
            if key in request_metadata
        }
        self._diagnostics = {
            "error_code": self.error_code,
            "provider_error_code": cause.error_code,
            "provider_status_code": cause.provider_status_code,
            "retryable": cause.retryable,
            "round_id": round_id,
            "unit_ordinal": unit_ordinal,
            "request_ordinal": request_ordinal,
            **allowlisted,
        }
        self.safe_summary = (
            "Doubter model request failed "
            f"(round={round_id}, unit_ordinal={unit_ordinal}, "
            f"provider={cause.error_code})"
        )
        super().__init__(self.safe_summary)

    def safe_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)


class DoubterProtocolGateError(RuntimeError):
    """Preserve aggregate protocol failures through managed child diagnostics."""

    diagnostic_event = "DOUBTER_PROTOCOL_FAILURE"
    error_code = "doubter_protocol_integrity_failed"
    safe_summary = "Doubter protocol integrity failed; inspect aggregate issue codes"

    def __init__(self, verification: Mapping[str, object], *, phase: str) -> None:
        self._diagnostics = {
            "error_code": self.error_code,
            "phase": phase,
            **{
                key: verification[key]
                for key in (
                    "issues",
                    "assessment_count",
                    "provider_nonconformant_count",
                    "provider_terminal_rejection_count",
                    "provider_integrity_violation_count",
                )
                if key in verification
            },
        }
        super().__init__(self.safe_summary)

    def safe_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)


async def verify_doubter_run(
    sessions: async_sessionmaker[AsyncSession],
    *,
    run_id: str,
) -> dict[str, object]:
    """Verify continuous candidate-level feedback and revision lineage."""

    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.compiler_runs is not None
        assert uow.questioner is not None
        assert uow.doubter is not None
        compiler = await uow.compiler_runs.get(run_id)
        if compiler is None:
            raise KeyError(f"compiler run does not exist: {run_id}")
        eligible_candidates = await uow.questioner.list_eligible_questions(run_id)
        r0 = await uow.doubter.get_r0_manifest(run_id)
        assessments = await uow.doubter.list_assessments(run_id)
        revisions = await uow.doubter.list_revisions(run_id)
        assert uow.memory_programs is not None
        programs = await uow.memory_programs.list_for_run(run_id)
        revision_jobs = await uow.doubter.list_candidate_jobs(
            run_id,
            stage=CandidateStageKind.PLANNER_REVISION,
        )
        rejection_diagnostics: dict[str, list[StructuredOutputItemDiagnostic]] = {}
        if compiler.compiler_schema_version in {"0.8.0", "0.9.0"}:
            for item in assessments:
                if not item.provider_schema_conformant:
                    rejection_diagnostics[
                        item.assessment_id
                    ] = await uow.structured_output_diagnostics.list_for_call(
                        item.producer_model_call_id
                    )
    r0_candidate_ids = {item.qbank_id for item in programs if item.round_id == 0}
    candidates = [item for item in eligible_candidates if item.qbank_id in r0_candidate_ids]
    by_candidate: dict[str, list[DoubterAssessment]] = defaultdict(list)
    for item in assessments:
        by_candidate[item.qbank_id].append(item)
    revision_targets = {(item.qbank_id, item.target_round_id) for item in revisions}
    unavailable_revision_targets = {
        (item.qbank_id, item.round_id)
        for item in revision_jobs
        if item.status is CandidateStageJobStatus.CANCELLED
        and item.metadata.get("terminal_reason") == "planner_revision_unavailable"
    }
    issues: list[str] = []
    if any(item.status is CandidateStageJobStatus.FAILED for item in revision_jobs):
        issues.append("planner_revision_job_failed")
    if any(
        item.status is CandidateStageJobStatus.CANCELLED
        and item.metadata.get("terminal_reason") != "planner_revision_unavailable"
        for item in revision_jobs
    ):
        issues.append("planner_revision_cancellation_invalid")
    if any(
        item.status
        in {
            CandidateStageJobStatus.PENDING,
            CandidateStageJobStatus.RUNNING,
        }
        for item in revision_jobs
    ):
        issues.append("planner_revision_job_unfinished")
    terminal_count = 0
    consumed_unavailable_targets: set[tuple[str, int]] = set()
    for candidate in candidates:
        values = sorted(
            by_candidate.get(candidate.qbank_id, []),
            key=lambda item: item.round_id,
        )
        rounds = [item.round_id for item in values]
        if not rounds or rounds != list(range(len(rounds))):
            issues.append("candidate_round_gap")
            continue
        terminal = False
        for ordinal, assessment in enumerate(values):
            if terminal:
                issues.append("assessment_after_terminal")
                break
            if assessment.action in {
                DoubterAction.PASS,
                DoubterAction.REJECT_CANDIDATE,
            }:
                terminal = True
                continue
            target = assessment.round_id + 1
            if assessment.round_id >= 3:
                issues.append("revision_after_max_round")
                terminal = True
            elif (candidate.qbank_id, target) in unavailable_revision_targets:
                terminal = True
                consumed_unavailable_targets.add((candidate.qbank_id, target))
            elif (
                (candidate.qbank_id, target) not in revision_targets
                or ordinal + 1 >= len(values)
                or values[ordinal + 1].round_id != target
            ):
                issues.append("revision_lineage_incomplete")
                terminal = True
        if terminal:
            terminal_count += 1
        else:
            issues.append("candidate_terminal_missing")
    if consumed_unavailable_targets != unavailable_revision_targets:
        issues.append("planner_revision_cancellation_orphaned")
    candidate_ids = {item.qbank_id for item in candidates}
    if any(item.qbank_id not in candidate_ids for item in assessments):
        issues.append("assessment_candidate_scope_mismatch")
    provider_nonconformant_count = sum(not item.provider_schema_conformant for item in assessments)
    provider_terminal_rejection_count = 0
    provider_integrity_violation_count = 0
    if compiler.compiler_schema_version == "0.7.0" and provider_nonconformant_count:
        issues.append("doubter_provider_schema_fallback_present")
    if compiler.compiler_schema_version in {"0.8.0", "0.9.0"}:
        for item in assessments:
            if item.provider_schema_conformant:
                if item.metadata.get("provider_parser_diagnostic_codes"):
                    issues.append("doubter_native_output_has_parser_diagnostics")
                    provider_integrity_violation_count += 1
                continue
            diagnostics = rejection_diagnostics[item.assessment_id]
            diagnostic_valid = len(diagnostics) == 1 and all(
                diagnostic.run_id == run_id
                and diagnostic.model_call_id == item.producer_model_call_id
                and diagnostic.compiler_unit_id == item.compiler_unit_id
                and diagnostic.stage is StructuredOutputStage.DOUBTER
                and diagnostic.status is StructuredOutputItemStatus.CONSERVATIVE_REJECT
                and diagnostic.repair_attempt_count == 0
                and bool(diagnostic.errors)
                for diagnostic in diagnostics
            )
            # Invalid output may terminate one candidate, never create a pass or a revision.
            safely_rejected = (
                item.action is DoubterAction.REJECT_CANDIDATE
                and item.issue_codes == [DoubterIssueCode.STRUCTURED_OUTPUT_INVALID]
                and not item.directives
                and not any(item.dimensions.model_dump().values())
                and not item.metadata.get("diagnostic_forced_revision")
                and diagnostic_valid
                and not any(
                    program.qbank_id == item.qbank_id and program.round_id > item.round_id
                    for program in programs
                )
            )
            if safely_rejected:
                provider_terminal_rejection_count += 1
            else:
                issues.append("doubter_nonconformant_output_not_safely_rejected")
                provider_integrity_violation_count += 1
    if r0 is None or not r0.hard_gate_passed:
        issues.append("r0_input_gate_missing")
    return {
        "run_id": run_id,
        "valid": not issues,
        "issues": sorted(set(issues)),
        "eligible_candidate_count": len(eligible_candidates),
        "candidate_count": len(candidates),
        "assessment_count": len(assessments),
        "provider_nonconformant_count": provider_nonconformant_count,
        "provider_terminal_rejection_count": provider_terminal_rejection_count,
        "provider_integrity_violation_count": provider_integrity_violation_count,
        "provider_schema_conformant_count": (len(assessments) - provider_nonconformant_count),
        "revision_count": len(revisions),
        "unavailable_revision_job_count": len(unavailable_revision_targets),
        "terminal_candidate_count": terminal_count,
        "round_counts": {
            str(round_id): dict(
                Counter(item.action.value for item in assessments if item.round_id == round_id)
            )
            for round_id in range(4)
        },
    }


class PostgresDoubterInputProvider:
    """Materialize only source and actually retrieved evidence for one candidate."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions
        self.memories = PostgresMemoryRepository(sessions)
        self.images = PostgresImageRepository(sessions)

    async def evidence_views(
        self,
        *,
        unit: CompilationUnit,
        source: Sequence[OperatorEvidence],
        execution: ProgramExecutionResult,
        limit: int,
        strict_roles: bool = False,
        claims: Sequence[QuestionerAnswerClaim] = (),
        refusal_only: bool = False,
        raw_images_allowed: bool = True,
    ) -> list[DoubterEvidenceView]:
        memories = await self.memories.list_snapshot(
            unit.snapshot_id,
            conversation_ids={unit.conversation_id},
        )
        memory_by_id = {item.memory_id: item for item in memories}
        if strict_roles:
            return await self._strict_evidence_views(
                unit=unit,
                source=source,
                execution=execution,
                limit=limit,
                claims=claims,
                memory_by_id=memory_by_id,
                refusal_only=refusal_only,
                raw_images_allowed=raw_images_allowed,
            )
        typed: dict[
            tuple[str, str],
            tuple[str | None, str | None, bool],
        ] = {}

        def add_evidence(item: OperatorEvidence, *, source_support: bool) -> None:
            direct_id = item.memory_id or item.image_id or item.session_id or item.caption_id
            if direct_id is None:
                raise ValueError("Doubter evidence lacks a typed canonical reference")
            owner = item.memory_id
            metadata_owner = item.metadata.get("owning_memory_id")
            if owner is None and isinstance(metadata_owner, str) and metadata_owner:
                owner = metadata_owner
            key = (item.evidence_type.value, direct_id)
            prior = typed.get(key)
            if prior is None:
                typed[key] = (owner, item.image_id, source_support)
                return
            prior_owner, prior_image_id, prior_source_support = prior
            if prior_owner is not None and owner is not None and prior_owner != owner:
                raise ValueError("Doubter evidence has conflicting owning memories")
            typed[key] = (
                prior_owner or owner,
                prior_image_id or item.image_id,
                prior_source_support or source_support,
            )

        for source_item in source:
            add_evidence(source_item, source_support=True)
        for execution_item in execution.evidence:
            add_evidence(
                OperatorEvidence(
                    evidence_id=execution_item.evidence_id,
                    evidence_type=execution_item.evidence_type,
                    memory_id=execution_item.memory_id,
                    image_id=execution_item.image_id,
                    session_id=execution_item.session_id,
                    caption_id=execution_item.caption_id,
                    score=execution_item.score,
                    metadata=execution_item.metadata,
                ),
                source_support=False,
            )
        unique = list(typed.items())[:limit]
        image_ids = list(
            dict.fromkeys(
                image_id
                for _key, (_owner, image_id, _source_support) in unique
                if image_id is not None
            )
        )
        image_by_id = {
            item.image_id: item for item in await self.images.get_many(unit.snapshot_id, image_ids)
        }
        values: list[DoubterEvidenceView] = []
        for ordinal, (
            (evidence_type, _direct_id),
            (owner_memory_id, image_id, source_support),
        ) in enumerate(unique, start=1):
            memory = memory_by_id.get(owner_memory_id or "")
            values.append(
                DoubterEvidenceView(
                    alias=f"E{ordinal:02d}",
                    evidence_type=evidence_type,
                    text=(memory.text or memory.caption) if memory is not None else None,
                    image=image_by_id.get(image_id or ""),
                    source_support=source_support,
                )
            )
        return values

    async def _strict_evidence_views(
        self,
        *,
        unit: CompilationUnit,
        source: Sequence[OperatorEvidence],
        execution: ProgramExecutionResult,
        limit: int,
        claims: Sequence[QuestionerAnswerClaim],
        memory_by_id: Mapping[str, MemoryUnit],
        refusal_only: bool = False,
        raw_images_allowed: bool = True,
    ) -> list[DoubterEvidenceView]:
        inspected = {
            image_id
            for step in execution.steps
            if step.op_name == "inspect_raw_image" and step.success
            for image_id in step.retrieved_image_ids
        }
        image_ids = list(
            dict.fromkeys(
                [
                    *(s.image_id for s in source if s.image_id),
                    *sorted(inspected),
                ]
            )
        )
        images = (
            {i.image_id: i for i in await self.images.get_many(unit.snapshot_id, image_ids)}
            if raw_images_allowed
            else {}
        )
        support: list[DoubterEvidenceView] = []
        for item in source:
            owner = item.memory_id or item.metadata.get("owning_memory_id")
            memory = memory_by_id.get(owner or "")
            speaker = memory.speaker.value if memory is not None else None
            matching = [
                c
                for c in claims
                if c.evidence_id
                in {
                    item.evidence_id,
                    item.memory_id,
                    item.image_id,
                }
            ]
            if item.image_id:
                native_supports = [
                    getattr(c, "metadata", {}).get("native_support", {}) for c in matching
                ]
                if owner is None:
                    owner = next(
                        (
                            native.get("owning_memory_id")
                            for native in native_supports
                            if native.get("owning_memory_id") in memory_by_id
                        ),
                        None,
                    )
                    memory = memory_by_id.get(owner or "")
                    speaker = memory.speaker.value if memory is not None else None
                caption = (memory.caption or memory.text) if memory is not None else None
                image_valid = (
                    bool(images.get(item.image_id)) if raw_images_allowed else bool(caption)
                ) and any(
                    native.get("source_field") == "image"
                    and native.get("image_id") == item.image_id
                    and native.get("owning_memory_id") == owner
                    and native.get("speaker") == speaker
                    and item.image_id in getattr(memory, "image_ids", [])
                    for native in native_supports
                )
                if refusal_only:
                    image_valid = (
                        bool(images.get(item.image_id)) if raw_images_allowed else bool(caption)
                    )
                support.append(
                    DoubterEvidenceView(
                        alias=f"S{len(support) + 1:02d}",
                        evidence_type=item.evidence_type.value if raw_images_allowed else "caption",
                        text=caption if not raw_images_allowed else None,
                        image=images.get(item.image_id),
                        source_support=True,
                        speaker=speaker,
                        span_exists=image_valid,
                    )
                )
            else:
                matching_or_missing: Sequence[QuestionerAnswerClaim | None] = matching or [None]
                for claim in matching_or_missing:
                    quote = claim.support_quote if claim is not None else None
                    if refusal_only and memory is not None:
                        support.append(
                            DoubterEvidenceView(
                                alias=f"S{len(support) + 1:02d}",
                                evidence_type=item.evidence_type.value,
                                text=memory.text or memory.caption,
                                source_support=True,
                                speaker=speaker,
                            )
                        )
                        continue
                    claim_metadata = getattr(claim, "metadata", {})
                    native = claim_metadata.get("native_support", {})
                    field = native.get("source_field")
                    text = getattr(memory, field, None) if field in {"text", "caption"} else None
                    start, end = native.get("start_char"), native.get("end_char")
                    exists = bool(
                        quote
                        and isinstance(text, str)
                        and isinstance(start, int)
                        and isinstance(end, int)
                        and 0 <= start < end <= len(text)
                        and text[start:end] == quote == native.get("exact_text")
                        and native.get("owning_memory_id") == owner
                        and native.get("speaker") == speaker
                    )
                    support.append(
                        DoubterEvidenceView(
                            alias=f"S{len(support) + 1:02d}",
                            evidence_type=item.evidence_type.value,
                            text=quote,
                            source_support=True,
                            speaker=speaker,
                            claim_text=claim.claim_text if claim is not None else None,
                            span_exists=exists,
                            claim_subject=claim_metadata.get("subject"),
                            claim_assertion=claim_metadata.get("assertion"),
                        )
                    )
        path: list[DoubterEvidenceView] = []
        seen: set[tuple[object, ...]] = set()
        for path_item in execution.evidence:
            key = (
                path_item.evidence_type,
                path_item.memory_id,
                path_item.image_id,
                path_item.caption_id,
                path_item.session_id,
            )
            if key in seen:
                continue
            seen.add(key)
            memory = memory_by_id.get(path_item.memory_id or "")
            raw = path_item.image_id in inspected
            path.append(
                DoubterEvidenceView(
                    alias=f"R{len(path) + 1:02d}",
                    evidence_type=path_item.evidence_type.value,
                    text=(memory.text or memory.caption) if memory is not None else None,
                    image=images.get(path_item.image_id or "") if raw else None,
                    speaker=memory.speaker.value if memory is not None else None,
                    raw_image_inspected=raw,
                )
            )
        # Reserve room for both roles; source duplication must not erase a real path.
        source_limit = min(len(support), max(1, limit // 2)) if path else limit
        projected_support = [
            replace(item, support_complete=len(support) <= source_limit)
            for item in support[:source_limit]
        ]
        return [*projected_support, *path[: limit - source_limit]]


def _job(
    *,
    run_id: str,
    unit_id: str,
    qbank_id: str,
    stage: CandidateStageKind,
    round_id: int,
) -> CandidateStageJob:
    payload = [run_id, unit_id, qbank_id, stage.value, round_id]
    return CandidateStageJob(
        candidate_stage_job_id=stable_compiler_id("candidate-stage-job", payload),
        run_id=run_id,
        compiler_unit_id=unit_id,
        qbank_id=qbank_id,
        stage=stage,
        round_id=round_id,
        content_hash=canonical_content_hash(payload),
    )


def _top_k(program: MemoryProgram) -> int:
    values = [
        value
        for step in program.operator_sequence
        for value in [step.args.get("top_k")]
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    return max(values, default=5)


def _safe_capability_trace(program: MemoryProgram) -> dict[str, object]:
    return {
        "round_id": program.round_id,
        "operators": [item.op_name for item in program.operator_sequence],
        "output_types": [item.expected_output_type for item in program.operator_sequence],
        "top_k": _top_k(program),
    }


async def _select_doubter_generation(
    *,
    base_call_id: str,
    candidate_attempt: int,
    get_trace: Callable[[str], Awaitable[Any]],
    has_saved: Callable[[GenerationRequest, str], Awaitable[bool]] | None,
    legacy_request: Callable[[str], GenerationRequest],
    exact_request: Callable[[], Awaitable[GenerationRequest]],
) -> tuple[str, GenerationRequest | None, Any]:
    # Old attempts may have a durable response but no committed trace/assessment.
    old_ids = [base_call_id] + [
        stable_compiler_id("doubter-model-call-recovery", [base_call_id, attempt])
        for attempt in range(1, candidate_attempt + 1)
    ]
    untraced = []
    for call_id in old_ids:
        trace = await get_trace(call_id)
        if trace is not None:
            if trace.status is TraceStatus.SUCCESS:
                return call_id, None, trace
            continue
        untraced.append(call_id)
        if has_saved:
            request = legacy_request(call_id)
            if await has_saved(request, call_id):
                return call_id, request, None

    request = await exact_request()
    # The first exact-count release could journal under an old ID. Recover it too.
    if has_saved:
        for call_id in untraced:
            old_request = request.model_copy(update={"request_id": call_id})
            if await has_saved(old_request, call_id):
                return call_id, old_request, None
    call_id = stable_compiler_id("doubter-model-call-exact", [
        base_call_id, "native-request-v1",
        canonical_content_hash(request.model_dump(mode="json", exclude={"request_id"})),
    ])
    trace = await get_trace(call_id)
    return call_id, request.model_copy(update={"request_id": call_id}), trace


class DoubterRunner:
    """Execute feedback_r -> optional plan_{r+1} -> real dry-run_{r+1}."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        execution: ModelExecutionService,
        doubter_gateway: GenerationGateway,
        planner_gateway: GenerationGateway,
        doubter_config: DoubterConfig,
        planner_config: PlannerConfig,
        foundation: CompilerFoundationConfig,
        static_validator: ProgramValidator,
        lease_seconds: int,
        concurrency: int = 64,
        dry_run_concurrency: int = 32,
        database_work_concurrency: int = 24,
        input_provider: PostgresDoubterInputProvider | None = None,
        token_counter: TokenCounter | None = None,
        recovery_token_counter: TokenCounter | None = None,
        diagnostic_force_revision: bool = False,
    ) -> None:
        compatible = (
            (
                planner_config.is_v12
                and foundation.operator_set_version == "pmmc-operators-v4"
                and not doubter_config.is_v11
            )
            or (
                planner_config.is_v13
                and foundation.operator_set_version == "pmmc-operators-v5"
                and doubter_config.is_v11
            )
            or (
                planner_config.is_v14
                and foundation.operator_set_version == "pmmc-operators-v6"
                and doubter_config.is_v12
            )
            or (
                planner_config.is_v15
                and foundation.operator_set_version == "pmmc-operators-v7"
                and doubter_config.is_v13
            )
        )
        if not compatible:
            raise ValueError("Doubter protocol must match Planner and operator-set versions")
        if min(lease_seconds, concurrency, dry_run_concurrency, database_work_concurrency) <= 0:
            raise ValueError("Doubter concurrency and lease values must be positive")
        self.sessions = sessions
        self.execution = execution
        self.doubter_gateway = doubter_gateway
        self.planner_gateway = planner_gateway
        self.config = doubter_config
        self.planner_config = planner_config
        self.foundation = foundation
        self.static_validator = static_validator
        self.policy = PlannerProgramPolicy(planner_config, registry=static_validator.registry)
        self.lease_seconds = lease_seconds
        self.concurrency = concurrency
        self.dry_run_concurrency = dry_run_concurrency
        self.database_work_concurrency = database_work_concurrency
        self.input_provider = input_provider or PostgresDoubterInputProvider(sessions)
        self.token_counter = token_counter
        self.recovery_token_counter = recovery_token_counter
        if diagnostic_force_revision and not doubter_config.is_v12:
            raise ValueError("diagnostic forced revision requires Doubter v1.2")
        self.diagnostic_force_revision = diagnostic_force_revision
        self._diagnostic_target_qbank_id: str | None = None
        self._revision_policy_rejections: Counter[str] = Counter()
        self._revision_unavailable_reasons: Counter[str] = Counter()
        self._request_projection_count = 0
        self._bounded_request_projection_count = 0
        self._request_projection_totals: Counter[str] = Counter()
        self._typed_patch_counts: Counter[tuple[int, int, int]] = Counter()

    async def execute(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        units: Sequence[CompilationUnit],
        memory_index: MemoryOperatorIndexManifest,
        query_vectors: Mapping[str, ProgramQueryVectorSet],
        on_progress: DoubterProgressCallback | None = None,
    ) -> DoubterRunResult:
        self._revision_policy_rejections.clear()
        self._revision_unavailable_reasons.clear()
        self._request_projection_count = 0
        self._bounded_request_projection_count = 0
        self._request_projection_totals.clear()
        self._typed_patch_counts.clear()
        unit_by_id = {item.compiler_unit_id: item for item in units}
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.questioner is not None
            assert uow.memory_programs is not None
            assert uow.planner is not None
            assert uow.program_dry_runs is not None
            assert uow.doubter is not None
            r0 = await uow.doubter.get_r0_manifest(run_id)
            if r0 is None or not r0.hard_gate_passed:
                raise RuntimeError("Doubter cannot start before a passing R0 input gate")
            eligible_candidates = await uow.questioner.list_eligible_questions(run_id)
            expected = await uow.questioner.list_expected_evidence(run_id)
            programs = await uow.memory_programs.list_for_run(run_id)
            dry_runs = await uow.program_dry_runs.list_for_run(run_id)
            contracts = {
                item.qbank_id: item for item in await uow.planner.list_requirement_contracts(run_id)
            }
        program_by_key = {(item.qbank_id, item.round_id): item for item in programs}
        dry_by_plan = {item.plan_id: item for item in dry_runs}
        r0_programs = {
            qbank_id: program
            for (qbank_id, round_id), program in program_by_key.items()
            if round_id == 0
        }
        candidates = [
            candidate
            for candidate in eligible_candidates
            if candidate.qbank_id in r0_programs
            and r0_programs[candidate.qbank_id].plan_id in dry_by_plan
        ]
        self._diagnostic_target_qbank_id = (
            min(
                candidate.qbank_id
                for candidate in candidates
                if candidate.route_class is QBankRouteClass.ANSWERABLE
            )
            if self.diagnostic_force_revision
            and any(candidate.route_class is QBankRouteClass.ANSWERABLE for candidate in candidates)
            else None
        )
        candidate_by_id = {item.qbank_id: item for item in candidates}
        missing_dry_run_count = sum(
            program.plan_id not in dry_by_plan for program in r0_programs.values()
        )
        missing_contract_count = sum(
            candidate.qbank_id not in contracts for candidate in candidates
        )
        missing_query_vector_count = sum(
            candidate.qbank_id not in query_vectors for candidate in candidates
        )
        answerable_r0_count = sum(
            candidate.route_class is QBankRouteClass.ANSWERABLE for candidate in candidates
        )
        preflight = {
            "answerable_r0_candidate_count": answerable_r0_count,
            "eligible_candidate_count": len(eligible_candidates),
            "missing_contract_count": missing_contract_count,
            "missing_dry_run_count": missing_dry_run_count,
            "missing_query_vector_count": missing_query_vector_count,
            "r0_candidate_count": len(candidates),
            "scientific_rejection_count": len(eligible_candidates) - len(candidates),
        }
        if missing_dry_run_count or missing_contract_count or missing_query_vector_count:
            raise DoubterInputGateError("r0_candidate_facts_incomplete", preflight)
        if not self.config.is_v13 and (not r0.downstream_viability or answerable_r0_count == 0):
            raise DoubterInputGateError("no_answerable_r0_candidate", preflight)
        print(
            "DOUBTER_INPUT_PREFLIGHT "
            + json.dumps(
                {**preflight, "state": "ready"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        for candidate in candidates:
            unit_id = candidate.compiler_unit_id
            if unit_id not in unit_by_id:
                raise DoubterInputGateError(
                    "candidate_outside_unit_denominator",
                    preflight,
                )

        for round_id in range(self.config.max_round + 1):
            assessments = await self._assessment_map(run_id)
            round_inputs = [
                candidate
                for candidate in candidates
                if (candidate.qbank_id, round_id) in program_by_key
                and program_by_key[(candidate.qbank_id, round_id)].plan_id in dry_by_plan
            ]
            current = self._round_job_candidates(round_inputs, assessments, round_id)
            await self._run_doubter_jobs(
                run_id=run_id,
                round_id=round_id,
                candidates=current,
                unit_by_id=unit_by_id,
                program_by_key=program_by_key,
                dry_by_plan=dry_by_plan,
                contracts=contracts,
                expected=expected,
                on_progress=on_progress,
            )
            assessments = await self._assessment_map(run_id)
            round_assessments = [
                value
                for (qbank_id, value_round), value in assessments.items()
                if value_round == round_id and qbank_id in candidate_by_id
            ]
            print(
                "DOUBTER_ROUND_ASSESSMENT_SUMMARY "
                + json.dumps(
                    {
                        "round_id": round_id,
                        "assessed": len(round_assessments),
                        "round_input": len(round_inputs),
                        "scheduled_candidates": len(current),
                        "revision_candidate_count": sum(
                            item.action is DoubterAction.REVISE_PROGRAM
                            for item in round_assessments
                        ),
                        "actions": dict(Counter(item.action.value for item in round_assessments)),
                        "next_stage": (
                            "planner_revision" if round_id < self.config.max_round else "decision"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if round_id < self.config.max_round:
                revisable = [
                    candidate
                    for candidate in current
                    if assessments[(candidate.qbank_id, round_id)].action
                    is DoubterAction.REVISE_PROGRAM
                ]
                await self._run_revision_jobs(
                    run_id=run_id,
                    round_id=round_id,
                    candidates=revisable,
                    unit_by_id=unit_by_id,
                    program_by_key=program_by_key,
                    contracts=contracts,
                    assessments=assessments,
                    expected=expected,
                    memory_index=memory_index,
                    query_vectors=query_vectors,
                )
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.memory_programs is not None
                    programs = await uow.memory_programs.list_for_run(run_id)
                program_by_key = {(item.qbank_id, item.round_id): item for item in programs}
                await self._run_revision_dry_jobs(
                    run_id=run_id,
                    target_round=round_id + 1,
                    candidates=revisable,
                    unit_by_id=unit_by_id,
                    program_by_key=program_by_key,
                    expected=expected,
                    memory_index=memory_index,
                    query_vectors=query_vectors,
                )
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.program_dry_runs is not None
                    dry_runs = await uow.program_dry_runs.list_for_run(run_id)
                dry_by_plan = {item.plan_id: item for item in dry_runs}
            from pmmc.models.campaign_milestones import record_doubter_round

            record_doubter_round(
                run_id=run_id,
                round_id=round_id,
                assessment_hashes={
                    item.assessment_id: item.content_hash for item in round_assessments
                },
            )
            if on_progress is not None:
                round_assessments = [
                    value
                    for (qbank_id, value_round), value in assessments.items()
                    if value_round == round_id and qbank_id in candidate_by_id
                ]
                on_progress(
                    round_id,
                    len(round_assessments),
                    len(round_inputs),
                    dict(Counter(item.action.value for item in round_assessments)),
                )
        assessments = await self._assessment_map(run_id)
        revisions = await self._revision_count(run_id)
        control_traces = {
            candidate.qbank_id: _control_trace_for_candidate(
                qbank_id=candidate.qbank_id,
                assessments=assessments,
                max_round=self.config.max_round,
            )
            for candidate in candidates
        }
        unavailable_revision_targets = await self._unavailable_revision_targets(run_id)
        terminal = self._terminal_candidates(
            assessments,
            unavailable_revision_targets=unavailable_revision_targets,
        )
        passed = sum(value[0] == "pass" for value in terminal.values())
        exhausted = sum(value[0] == "max_rounds_exhausted" for value in terminal.values())
        unavailable = sum(value[0] == "planner_revision_unavailable" for value in terminal.values())
        round_counts: dict[int, dict[str, int]] = {}
        for round_id in range(self.config.max_round + 1):
            round_counts[round_id] = dict(
                Counter(
                    value.action.value
                    for (qbank_id, item_round), value in assessments.items()
                    if item_round == round_id and qbank_id in candidate_by_id
                )
            )
        await self._persist_stage_results(
            run_id=run_id,
            units=units,
            candidates=candidates,
            assessments=assessments,
            control_traces=control_traces,
            unavailable_revision_targets=unavailable_revision_targets,
            created_at=r0.created_at,
        )
        issue_counts = Counter(
            issue.value for assessment in assessments.values() for issue in assessment.issue_codes
        )
        dimension_failure_counts = Counter(
            name
            for assessment in assessments.values()
            if assessment.provider_schema_conformant
            for name, value in assessment.dimensions.model_dump(mode="python").items()
            if value is False
        )
        not_applicable_counts: Counter[str] = Counter()
        for assessment in assessments.values():
            verdicts = assessment.metadata.get("provider_dimension_verdicts")
            if not isinstance(verdicts, Mapping):
                continue
            not_applicable_counts.update(
                str(name) for name, value in verdicts.items() if value == "not_applicable"
            )
        terminal_counts = Counter(value[0] for value in terminal.values())
        provider_error_counts = Counter(
            str(assessment.metadata.get("provider_error_code", "legacy_unclassified"))
            for assessment in assessments.values()
            if not assessment.provider_schema_conformant
        )
        parser_diagnostic_counts = Counter(
            str(code)
            for assessment in assessments.values()
            for code in (
                assessment.metadata.get("provider_parser_diagnostic_codes", [])
                if isinstance(assessment.metadata.get("provider_parser_diagnostic_codes"), list)
                else []
            )
        )
        provider_native_count = sum(
            assessment.provider_schema_conformant for assessment in assessments.values()
        )
        print(
            "DOUBTER_OUTCOME_SUMMARY "
            + json.dumps(
                {
                    "assessment_count": len(assessments),
                    "candidate_count": len(candidates),
                    "dimension_failure_counts": dict(sorted(dimension_failure_counts.items())),
                    "dimension_failure_denominator": provider_native_count,
                    "dimension_failure_scope": "native_assessments_only",
                    "issue_counts": dict(sorted(issue_counts.items())),
                    "provider_fallback_count": (len(assessments) - provider_native_count),
                    "provider_native_count": provider_native_count,
                    "provider_error_counts": dict(sorted(provider_error_counts.items())),
                    "provider_parser_diagnostic_counts": dict(
                        sorted(parser_diagnostic_counts.items())
                    ),
                    "provider_not_applicable_counts": dict(sorted(not_applicable_counts.items())),
                    "request_projection": {
                        "scope": "execution_attempt_new_requests",
                        "bounded_request_count": self._bounded_request_projection_count,
                        "duplicate_image_reference_count": self._request_projection_totals[
                            "duplicate_image_reference_count"
                        ],
                        "exact_token_counted_request_count": self._request_projection_totals[
                            "exact_token_counted_request_count"
                        ],
                        "exact_input_token_sum": self._request_projection_totals[
                            "exact_input_token_sum"
                        ],
                        "omitted_image_count": self._request_projection_totals[
                            "omitted_image_count"
                        ],
                        "request_count": self._request_projection_count,
                        "token_budget_image_reduction_count": self._request_projection_totals[
                            "token_budget_image_reduction_count"
                        ],
                        "truncated_text_item_count": self._request_projection_totals[
                            "truncated_text_item_count"
                        ],
                    },
                    "revision_count": revisions,
                    "new_typed_patch_count": self._typed_patch_counts.total(),
                    "typed_patch_top_k_transitions": [
                        {
                            "round_id": r,
                            "before_top_k": before,
                            "after_top_k": after,
                            "count": count,
                        }
                        for (r, before, after), count in sorted(self._typed_patch_counts.items())
                    ],
                    "revision_unavailable_reason_counts": dict(
                        sorted(self._revision_unavailable_reasons.items())
                    ),
                    "round_counts": round_counts,
                    "terminal_counts": dict(sorted(terminal_counts.items())),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        if self._revision_policy_rejections:
            print(
                "PLANNER_REVISION_POLICY_SUMMARY "
                + json.dumps(
                    {
                        "counts": dict(sorted(self._revision_policy_rejections.items())),
                        "scope": "execution_attempt_new",
                        "total": sum(self._revision_policy_rejections.values()),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return DoubterRunResult(
            run_id=run_id,
            candidate_count=len(candidates),
            assessment_count=len(assessments),
            revision_count=revisions,
            passed_count=passed,
            rejected_count=len(terminal) - passed,
            revision_unavailable_count=unavailable,
            exhausted_count=exhausted,
            round_counts=round_counts,
        )

    def _record_revision_unavailable(
        self,
        reason: str,
        *,
        target_round: int,
        unit_ordinal: int,
        diagnostics: Mapping[str, object] | None = None,
        emit: bool = True,
    ) -> None:
        self._revision_unavailable_reasons[reason] += 1
        count = self._revision_unavailable_reasons[reason]
        if not emit or (count != 1 and count % 50 != 0):
            return
        print(
            "PLANNER_REVISION_UNAVAILABLE "
            + json.dumps(
                {
                    "diagnostics": dict(diagnostics or {}),
                    "occurrence_count": count,
                    "reason": reason,
                    "round_id": target_round,
                    "unit_ordinal": unit_ordinal,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    async def _persist_stage_results(
        self,
        *,
        run_id: str,
        units: Sequence[CompilationUnit],
        candidates: Sequence[QuestionCandidate],
        assessments: Mapping[tuple[str, int], DoubterAssessment],
        control_traces: Mapping[str, DoubterControlTrace],
        unavailable_revision_targets: set[tuple[str, int]],
        created_at: datetime,
    ) -> None:
        candidate_unit = {item.qbank_id: item.compiler_unit_id for item in candidates}
        by_unit: dict[str, list[DoubterAssessment]] = {item.compiler_unit_id: [] for item in units}
        for (qbank_id, _), assessment in assessments.items():
            unit_id = candidate_unit.get(qbank_id)
            if unit_id is not None:
                by_unit[unit_id].append(assessment)
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.compiler_units is not None
            for unit in units:
                values = by_unit[unit.compiler_unit_id]
                terminal = self._terminal_candidates(
                    {(item.qbank_id, item.round_id): item for item in values},
                    unavailable_revision_targets=unavailable_revision_targets,
                )
                accepted = sum(item[0] == "pass" for item in terminal.values())
                rejected = len(terminal) - accepted
                counts = {
                    "produced": len(values),
                    "accepted": accepted,
                    "rejected": rejected,
                }
                stage = CompilationStageResult(
                    stage_result_id=stable_compiler_id(
                        "compiler-stage",
                        [run_id, unit.compiler_unit_id, "doubter", 0],
                    ),
                    run_id=run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    stage=CompilationStage.DOUBTER,
                    status=TraceStatus.SUCCESS,
                    produced_count=counts["produced"],
                    accepted_count=counts["accepted"],
                    rejected_count=counts["rejected"],
                    result_hash=canonical_content_hash([run_id, unit.compiler_unit_id, counts]),
                    started_at=created_at,
                    completed_at=created_at,
                    elapsed_ms=0.0,
                    metadata={
                        "rounds": [0, 1, 2, 3],
                        "control_graph": "pmmc-doubter-control-graph-v1",
                        "control_terminal_counts": dict(
                            Counter(
                                control_traces[item.qbank_id].terminal
                                for item in candidates
                                if item.compiler_unit_id == unit.compiler_unit_id
                            )
                        ),
                        "control_trace_hash": canonical_content_hash(
                            sorted(
                                control_traces[item.qbank_id].trace_hash
                                for item in candidates
                                if item.compiler_unit_id == unit.compiler_unit_id
                            )
                        ),
                    },
                )
                await uow.compiler_units.add_stage_result(stage)

    async def _assessment_map(self, run_id: str) -> dict[tuple[str, int], DoubterAssessment]:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            values = await uow.doubter.list_assessments(run_id)
        return {(item.qbank_id, item.round_id): item for item in values}

    async def _revision_count(self, run_id: str) -> int:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            return len(await uow.doubter.list_revisions(run_id))

    async def _unavailable_revision_targets(self, run_id: str) -> set[tuple[str, int]]:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            jobs = await uow.doubter.list_candidate_jobs(
                run_id,
                stage=CandidateStageKind.PLANNER_REVISION,
            )
        return {
            (item.qbank_id, item.round_id)
            for item in jobs
            if item.status is CandidateStageJobStatus.CANCELLED
            and item.metadata.get("terminal_reason") == "planner_revision_unavailable"
        }

    def _terminal_candidates(
        self,
        assessments: Mapping[tuple[str, int], DoubterAssessment],
        *,
        unavailable_revision_targets: set[tuple[str, int]] | None = None,
    ) -> dict[str, tuple[str, int, DoubterAssessment]]:
        result: dict[str, tuple[str, int, DoubterAssessment]] = {}
        unavailable = unavailable_revision_targets or set()
        for (_, round_id), item in sorted(assessments.items(), key=lambda value: value[0][1]):
            if item.qbank_id in result:
                continue
            if item.action is DoubterAction.PASS:
                result[item.qbank_id] = ("pass", round_id, item)
            elif item.action is DoubterAction.REJECT_CANDIDATE:
                outcome = (
                    "max_rounds_exhausted"
                    if DoubterIssueCode.MAX_ROUNDS_EXHAUSTED in item.issue_codes
                    else "reject"
                )
                result[item.qbank_id] = (outcome, round_id, item)
            elif (
                item.action is DoubterAction.REVISE_PROGRAM
                and (item.qbank_id, round_id + 1) in unavailable
            ):
                result[item.qbank_id] = (
                    "planner_revision_unavailable",
                    round_id,
                    item,
                )
        return result

    def _round_job_candidates(
        self,
        candidates: Sequence[QuestionCandidate],
        assessments: Mapping[tuple[str, int], DoubterAssessment],
        round_id: int,
    ) -> list[QuestionCandidate]:
        terminal = self._terminal_candidates(assessments)
        # An assessment may commit before its leased job finishes. Reclaim that
        # job through the normal fence; _assess_one reuses the saved assessment.
        return [
            item for item in candidates
            if item.qbank_id not in terminal or (item.qbank_id, round_id) in assessments
        ]

    async def _ensure_jobs(
        self,
        *,
        run_id: str,
        stage: CandidateStageKind,
        round_id: int,
        candidates: Sequence[QuestionCandidate],
    ) -> None:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            for candidate in candidates:
                assert candidate.compiler_unit_id is not None
                value = _job(
                    run_id=run_id,
                    unit_id=candidate.compiler_unit_id,
                    qbank_id=candidate.qbank_id,
                    stage=stage,
                    round_id=round_id,
                )
                existing = await uow.doubter.get_candidate_job(value.candidate_stage_job_id)
                if existing is None:
                    await uow.doubter.add_candidate_job(value)
                elif existing.content_hash != value.content_hash:
                    raise ValueError("candidate stage job identity conflict")

    async def _run_doubter_jobs(
        self,
        *,
        run_id: str,
        round_id: int,
        candidates: Sequence[QuestionCandidate],
        unit_by_id: Mapping[str, CompilationUnit],
        program_by_key: Mapping[tuple[str, int], MemoryProgram],
        dry_by_plan: Mapping[str, ProgramDryRunAssessment],
        contracts: Mapping[str, PlannerRequirementContract | None],
        expected: Mapping[str, Sequence[OperatorEvidence]],
        on_progress: DoubterProgressCallback | None,
    ) -> None:
        if not candidates:
            return
        await self._ensure_jobs(
            run_id=run_id,
            stage=CandidateStageKind.DOUBTER,
            round_id=round_id,
            candidates=candidates,
        )
        candidate_by_id = {item.qbank_id: item for item in candidates}
        existing = await self._assessment_map(run_id)
        action_counts = Counter(
            item.action.value
            for (qbank_id, item_round), item in existing.items()
            if item_round == round_id and qbank_id in candidate_by_id
        )
        completed = sum(action_counts.values())
        reported_ids = {
            qbank_id
            for qbank_id, item_round in existing
            if item_round == round_id and qbank_id in candidate_by_id
        }
        completed_lock = asyncio.Lock()

        async def worker(index: int) -> None:
            nonlocal completed
            worker_id = f"doubter-r{round_id}-{index:02d}"
            while True:
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.doubter is not None
                    job = await uow.doubter.claim_candidate_job(
                        run_id=run_id,
                        stage=CandidateStageKind.DOUBTER,
                        round_id=round_id,
                        worker_id=worker_id,
                        lease_seconds=self.lease_seconds,
                        qbank_ids=set(candidate_by_id),
                    )
                if job is None:
                    return
                candidate = candidate_by_id.get(job.qbank_id)
                try:
                    if candidate is not None:
                        contract = contracts[candidate.qbank_id]
                        if contract is None:
                            raise RuntimeError("Doubter requirement contract is missing")
                        program = program_by_key[(candidate.qbank_id, round_id)]
                        assessment = dry_by_plan[program.plan_id]
                        unit = unit_by_id[job.compiler_unit_id]
                        result = await self._assess_one(
                            run_id=run_id,
                            unit=unit,
                            candidate=candidate,
                            contract=contract,
                            program=program,
                            dry_run=assessment,
                            source_evidence=expected.get(candidate.qbank_id, ()),
                            candidate_stage_attempt=job.attempt_count,
                        )
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.doubter is not None
                        await uow.doubter.finish_candidate_job(
                            job.candidate_stage_job_id,
                            worker_id=worker_id,
                            succeeded=True,
                        )
                except Exception as exc:
                    diagnostics = safe_database_error_diagnostics(exc)
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.doubter is not None
                        await uow.doubter.finish_candidate_job(
                            job.candidate_stage_job_id,
                            worker_id=worker_id,
                            succeeded=False,
                            error_type=type(exc).__name__,
                            error_summary=safe_database_error_summary(diagnostics),
                        )
                    raise
                async with completed_lock:
                    if candidate is not None and candidate.qbank_id not in reported_ids:
                        reported_ids.add(candidate.qbank_id)
                        completed += 1
                        action_counts[result.action.value] += 1
                    if on_progress is not None and completed % 10 == 0:
                        on_progress(round_id, completed, len(candidates), dict(action_counts))

        await _await_workers(
            [
                asyncio.create_task(worker(index))
                for index in range(min(self.concurrency, max(1, len(candidates))))
            ]
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            jobs = await uow.doubter.list_candidate_jobs(
                run_id, stage=CandidateStageKind.DOUBTER, round_id=round_id
            )
        relevant = [item for item in jobs if item.qbank_id in candidate_by_id]
        if any(item.status is not CandidateStageJobStatus.SUCCEEDED for item in relevant):
            raise RuntimeError(f"Doubter round {round_id} has unfinished candidate jobs")

    def _record_request_preflight(
        self,
        *,
        round_id: int,
        unit_ordinal: int,
        request_metadata: Mapping[str, object],
    ) -> int:
        self._request_projection_count += 1
        request_ordinal = self._request_projection_count
        omitted_images = _safe_metadata_count(request_metadata, "omitted_image_count")
        duplicate_images = _safe_metadata_count(request_metadata, "duplicate_image_reference_count")
        token_budget_reductions = _safe_metadata_count(
            request_metadata, "token_budget_image_reduction_count"
        )
        truncated_texts = _safe_metadata_count(request_metadata, "truncated_text_item_count")
        bounded = any(
            value > 0
            for value in (
                omitted_images,
                duplicate_images,
                token_budget_reductions,
                truncated_texts,
            )
        )
        if bounded:
            self._bounded_request_projection_count += 1
        self._request_projection_totals["omitted_image_count"] += omitted_images
        self._request_projection_totals["duplicate_image_reference_count"] += duplicate_images
        self._request_projection_totals["token_budget_image_reduction_count"] += (
            token_budget_reductions
        )
        if not bool(request_metadata.get("token_count_estimated", True)):
            self._request_projection_totals["exact_token_counted_request_count"] += 1
            self._request_projection_totals["exact_input_token_sum"] += _safe_metadata_count(
                request_metadata, "input_tokens"
            )
        self._request_projection_totals["truncated_text_item_count"] += truncated_texts
        bounded_ordinal = self._bounded_request_projection_count
        emit = (
            request_ordinal == 1
            or request_ordinal % 50 == 0
            or (bounded and (bounded_ordinal <= 3 or bounded_ordinal % 50 == 0))
        )
        if emit:
            print(
                "DOUBTER_REQUEST_PREFLIGHT "
                + json.dumps(
                    {
                        "attached_images": _safe_metadata_count(
                            request_metadata, "attached_image_count"
                        ),
                        "duplicate_image_references": duplicate_images,
                        "bounded_request_count": bounded_ordinal,
                        "estimated_input_tokens": _safe_metadata_count(
                            request_metadata, "estimated_input_tokens"
                        ),
                        "exact_input_tokens": _safe_metadata_count(
                            request_metadata, "input_tokens"
                        ),
                        "evidence_count": _safe_metadata_count(request_metadata, "evidence_count"),
                        "max_input_tokens": _safe_metadata_count(
                            request_metadata, "max_input_tokens"
                        ),
                        "omitted_images": omitted_images,
                        "token_budget_image_reductions": token_budget_reductions,
                        "token_count_estimated": bool(
                            request_metadata.get("token_count_estimated", True)
                        ),
                        "unique_input_images": _safe_metadata_count(
                            request_metadata, "input_unique_image_count"
                        ),
                        "rendered_text_chars": _safe_metadata_count(
                            request_metadata, "rendered_text_chars"
                        ),
                        "request_ordinal": request_ordinal,
                        "round": round_id,
                        "state": "bounded" if bounded else "ready",
                        "truncated_text_items": truncated_texts,
                        "unit_ordinal": unit_ordinal,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return request_ordinal

    async def _assess_one(
        self,
        *,
        run_id: str,
        unit: CompilationUnit,
        candidate: QuestionCandidate,
        contract: PlannerRequirementContract,
        program: MemoryProgram,
        dry_run: ProgramDryRunAssessment,
        source_evidence: Sequence[OperatorEvidence],
        candidate_stage_attempt: int,
    ) -> DoubterAssessment:
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            existing = await uow.doubter.get_assessment(
                run_id, candidate.qbank_id, program.round_id
            )
            if existing is not None:
                return existing
            assert uow.program_executions is not None
            execution = await uow.program_executions.get(dry_run.execution_id)
            claims = []
            if self.config.is_v13:
                assert uow.questioner is not None
                claims = await uow.questioner.list_answer_claims(candidate.qbank_id)
        if execution is None:
            raise RuntimeError("Doubter dry-run execution is missing")
        role_options: dict[str, Any] = (
            {
                "strict_roles": True,
                "claims": claims,
                "refusal_only": candidate.route_class is QBankRouteClass.CONTROL,
                "raw_images_allowed": self.planner_config.memory_raw_image_policy != "forbid",
            }
            if self.config.is_v13
            else {}
        )
        views = await self.input_provider.evidence_views(
            unit=unit,
            source=source_evidence,
            execution=execution,
            limit=self.config.max_evidence_items,
            **role_options,
        )
        base_model_call_id = stable_compiler_id(
            "doubter-model-call",
            [run_id, candidate.qbank_id, program.plan_id, program.round_id],
        )
        async def get_trace(call_id):
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                return await uow.model_calls.get(call_id)

        render_inputs = dict(
            config=self.config, candidate=candidate, requirement=contract,
            program=program, assessment=dry_run, evidence_views=views,
        )

        async def exact_request():
            prepare_images = getattr(self.token_counter, "prepare_images", None)
            if prepare_images is not None:
                await prepare_images([v.image for v in views if v.image is not None])
            return render_doubter_request(
                request_id=base_model_call_id, **render_inputs, token_counter=self.token_counter,
            )

        model_call_id, request, trace = await _select_doubter_generation(
            base_call_id=base_model_call_id, candidate_attempt=candidate_stage_attempt,
            get_trace=get_trace,
            has_saved=getattr(self.execution, "has_saved_generation", None),
            legacy_request=lambda call_id: render_doubter_request(
                request_id=call_id, **render_inputs, token_counter=self.recovery_token_counter,
            ),
            exact_request=exact_request,
        )
        request_metadata: dict[str, object] = {}
        if trace is None:
            assert request is not None
            request_metadata = dict(request.metadata)
            request_ordinal = self._record_request_preflight(
                round_id=program.round_id,
                unit_ordinal=unit.ordinal,
                request_metadata=request_metadata,
            )
            try:
                audited = await self.execution.generate_audited(
                    self.doubter_gateway,
                    request,
                    ModelCallContext(
                        model_call_id=model_call_id,
                        run_id=run_id,
                        snapshot_id=unit.snapshot_id,
                        conversation_id=unit.conversation_id,
                        job_id=_compiler_job_id(run_id, unit.compiler_unit_id),
                        compiler_unit_id=unit.compiler_unit_id,
                        qbank_id=candidate.qbank_id,
                        plan_id=program.plan_id,
                        stage=AppStage.DOUBTER,
                    ),
                    output_parser=doubter_output_parser(config=self.config),
                )
            except ModelGatewayError as exc:
                raise DoubterModelRequestError(
                    round_id=program.round_id,
                    unit_ordinal=unit.ordinal,
                    request_ordinal=request_ordinal,
                    request_metadata=request_metadata,
                    cause=exc,
                ) from exc
            trace = audited.trace
        else:
            stored_request_metadata = trace.request_config.get("request_metadata")
            if isinstance(stored_request_metadata, Mapping):
                request_metadata = dict(stored_request_metadata)
            if trace.status is TraceStatus.FAILED:
                raise DoubterModelRequestError(
                    round_id=program.round_id,
                    unit_ordinal=unit.ordinal,
                    request_ordinal=self._request_projection_count + 1,
                    request_metadata=request_metadata,
                    cause=ModelGatewayError(
                        trace.error_type or "StoredModelCallFailure",
                        retryable=False,
                    ),
                )
        raw = dict(trace.parsed_output or {})
        parser_diagnostic_codes = (
            [str(value) for value in raw.get("diagnostic_codes", []) if isinstance(value, str)]
            if isinstance(raw.get("diagnostic_codes"), list)
            else []
        )
        provider_conformant = bool(
            trace.status is TraceStatus.SUCCESS
            and trace.repair_attempt_count == 0
            and not trace.parse_error
            and raw.get("provider_schema_conformant", False)
        )
        provider: DoubterProviderResult | DoubterProviderResultV11 | None = None
        provider_error_code: str | None = None
        provider_error_dimension: str | None = None
        if provider_conformant and isinstance(raw.get("result"), Mapping):
            try:
                if self.config.is_v11:
                    provider = DoubterProviderResultV11.model_validate(raw["result"])
                    dimensions = normalize_v11_dimensions(
                        result=provider,
                        route_class=candidate.route_class,
                        requirement=contract,
                    )
                    issues = derive_doubter_issues(
                        dimensions=dimensions,
                        route_class=candidate.route_class,
                    )
                else:
                    provider = DoubterProviderResult.model_validate(raw["result"])
                    dimensions = provider.dimensions
                    issues = validate_provider_issue_consistency(
                        result=provider,
                        route_class=candidate.route_class,
                    )
            except (TypeError, ValueError) as exc:
                message = str(exc)
                provider_error_code = (
                    "applicability_contract_invalid"
                    if "Doubter dimension" in message
                    else "post_parse_validation_invalid"
                )
                provider_error_dimension = next(
                    (
                        name
                        for name in (
                            "candidate_supported",
                            "answer_entailed",
                            "path_sufficient",
                            "modality_sufficient",
                            "raw_image_sufficient",
                            "context_sufficient",
                            "multi_hop_sufficient",
                            "temporal_consistent",
                            "retrieval_nonredundant",
                            "refusal_appropriate",
                        )
                        if name in message
                    ),
                    None,
                )
                provider = None
        if provider is None:
            if provider_error_code is None:
                provider_error_code = "structured_output_invalid"
            dimensions = DoubterDimensions(
                candidate_supported=False,
                answer_entailed=False,
                path_sufficient=False,
                modality_sufficient=False,
                raw_image_sufficient=False,
                context_sufficient=False,
                multi_hop_sufficient=False,
                temporal_consistent=False,
                non_redundant=False,
                refusal_appropriate=False,
            )
            issues = [DoubterIssueCode.STRUCTURED_OUTPUT_INVALID]
            provider_conformant = False
        else:
            if self.config.is_v13:
                source_views = [v for v in views if v.source_support]
                path_views = [v for v in views if not v.source_support]
                dimensions = enforce_v13_dimensions(
                    dimensions=dimensions,
                    candidate=candidate,
                    requirement=contract,
                    assessment=dry_run,
                    source_present=bool(source_views),
                    source_spans_valid=(
                        all(v.span_exists is not False and v.support_complete for v in source_views)
                        and not any(
                            v.claim_subject == "user"
                            and v.speaker == "assistant"
                            and v.claim_assertion in {"fact", "plan"}
                            for v in source_views
                        )
                        and request_metadata.get("source_support_complete", False) is True
                    ),
                    path_present=bool(path_views),
                    inspected_path_image_present=any(
                        v.raw_image_inspected and v.image for v in path_views
                    )
                    and _safe_metadata_count(request_metadata, "rendered_raw_path_image_count") > 0,
                )
                issues = derive_doubter_issues(
                    dimensions=dimensions, route_class=candidate.route_class
                )
            if self.config.is_v12 and candidate.route_class is QBankRouteClass.ANSWERABLE:
                overrides: dict[str, bool] = {}
                if (
                    not dry_run.nonempty_evidence
                    or not dry_run.expected_evidence_complete
                    or not dry_run.pre_control_sufficient
                ):
                    overrides["path_sufficient"] = False
                if not dry_run.required_modality_sufficient:
                    overrides["modality_sufficient"] = False
                if any(item.value == "raw_image" for item in contract.required_modalities):
                    overrides["raw_image_sufficient"] = dry_run.raw_image_reached
                if PlannerCapability.CONTEXT in contract.required_capabilities:
                    overrides["context_sufficient"] = dry_run.context_evidence_contributed
                if PlannerCapability.TEMPORAL in contract.required_capabilities:
                    overrides["temporal_consistent"] = dry_run.temporal_evidence_contributed
                if contract.minimum_evidence_groups > 1:
                    overrides["multi_hop_sufficient"] = (
                        dry_run.matched_discriminative_group_count
                        >= contract.minimum_evidence_groups
                    )
                if overrides:
                    dimensions = dimensions.model_copy(update=overrides)
                    issues = derive_doubter_issues(
                        dimensions=dimensions,
                        route_class=candidate.route_class,
                    )
            issues = list(
                dict.fromkeys(
                    [
                        *issues,
                        *derive_dry_run_issues(
                            candidate=candidate,
                            requirement=contract,
                            assessment=dry_run,
                        ),
                    ]
                )
            )
        diagnostic_forced_revision = bool(
            self.diagnostic_force_revision
            and provider_conformant
            and program.round_id == 0
            and candidate.qbank_id == self._diagnostic_target_qbank_id
        )
        if diagnostic_forced_revision:
            dimensions = dimensions.model_copy(
                update={
                    "candidate_supported": True,
                    "answer_entailed": True,
                    "path_sufficient": False,
                    "modality_sufficient": True,
                    "raw_image_sufficient": True,
                    "context_sufficient": True,
                    "multi_hop_sufficient": True,
                    "temporal_consistent": True,
                    "non_redundant": True,
                    "refusal_appropriate": True,
                }
            )
            issues = [DoubterIssueCode.PATH_INSUFFICIENT]
            print(
                "STEP7B_DIAGNOSTIC_FORCED_REVISION "
                + json.dumps(
                    {
                        "round_id": program.round_id,
                        "state": "revision_required",
                        "unit_ordinal": unit.ordinal,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        action = derive_doubter_action(
            issues=issues,
            round_id=program.round_id,
            max_round=self.config.max_round,
        )
        fixed_verification_only = (
            self.config.is_v13
            and self.planner_config.is_v15
            and self.planner_config.planning_mode == "fixed"
        )
        if fixed_verification_only and action is DoubterAction.REVISE_PROGRAM:
            action = DoubterAction.REJECT_CANDIDATE
        if (
            action is DoubterAction.REJECT_CANDIDATE
            and program.round_id >= self.config.max_round
            and not any(item in _DIRECT_REJECT for item in issues)
        ):
            issues = [*issues, DoubterIssueCode.MAX_ROUNDS_EXHAUSTED]
        feedback_id = stable_compiler_id(
            "doubter-feedback",
            [run_id, candidate.qbank_id, program.plan_id, program.round_id],
        )
        directives = (
            derive_doubter_directives(
                feedback_id=feedback_id,
                run_id=run_id,
                candidate=candidate,
                requirement=contract,
                source_round_id=program.round_id,
                issues=issues,
                current_top_k=_top_k(program),
                strict_capabilities=self.config.is_v13,
            )
            if action is DoubterAction.REVISE_PROGRAM
            else []
        )
        payload = {
            "feedback_id": feedback_id,
            "run_id": run_id,
            "qbank_id": candidate.qbank_id,
            "plan_id": program.plan_id,
            "round_id": program.round_id,
            "dimensions": dimensions.model_dump(mode="json"),
            "issues": [item.value for item in issues],
            "action": action.value,
            "directives": [item.model_dump(mode="json") for item in directives],
            "provider_schema_conformant": provider_conformant,
        }
        assessment = DoubterAssessment(
            assessment_id=stable_compiler_id("doubter-assessment", payload),
            feedback_id=feedback_id,
            run_id=run_id,
            snapshot_id=unit.snapshot_id,
            compiler_unit_id=unit.compiler_unit_id,
            qbank_id=candidate.qbank_id,
            plan_id=program.plan_id,
            execution_id=dry_run.execution_id,
            producer_model_call_id=trace.model_call_id,
            round_id=program.round_id,
            prompt_version=self.config.prompt_version,
            **({"schema_version": "0.4.0"} if self.config.is_v13 else {}),
            dimensions=dimensions,
            issue_codes=issues,
            action=action,
            directives=directives,
            provider_schema_conformant=provider_conformant,
            content_hash=canonical_content_hash(payload),
            created_at=trace.created_at,
            metadata={
                "single_candidate_scope": True,
                "prompt_hash": doubter_prompt_sha256(self.config),
                "source_evidence_count": sum(item.source_support for item in views),
                "retrieved_evidence_count": sum(not item.source_support for item in views),
                "provider_dimension_verdicts": (
                    provider.dimensions.model_dump(mode="json")
                    if self.config.is_v11 and provider is not None
                    else {
                        name: "pass" if passed else "fail"
                        for name, passed in dimensions.model_dump(mode="python").items()
                    }
                ),
                "provider_error_code": provider_error_code,
                "provider_error_dimension": provider_error_dimension,
                "provider_parser_diagnostic_codes": parser_diagnostic_codes,
                "request_projection": request_metadata,
                "diagnostic_forced_revision": diagnostic_forced_revision,
                **(
                    {
                        "fixed_planner_verification_only": fixed_verification_only,
                        "scientific_terminal_reason": (
                            "fixed_planner_verification_failed"
                            if fixed_verification_only and issues
                            else None
                        ),
                        "path_facts": {
                            "technical": dry_run.execution_success,
                            "nonempty": dry_run.nonempty_evidence,
                            "expected_group_complete": dry_run.expected_evidence_complete,
                            "expected_group_recall": dry_run.expected_evidence_recall,
                            "matched_group_count": dry_run.matched_discriminative_group_count,
                            "modalities": dry_run.required_modality_sufficient,
                            "raw_image": dry_run.raw_image_reached,
                            "semantic_path_assessment": (
                                "model_assessed_pass"
                                if dimensions.path_sufficient
                                else "model_assessed_fail"
                            )
                            if provider_conformant
                            else "not_evaluated",
                            "semantic_answer_support": dry_run.semantic_answer_support.value,
                        },
                    }
                    if self.config.is_v13
                    else {}
                ),
            },
        )
        feedback = DoubterFeedback(
            feedback_id=feedback_id,
            run_id=run_id,
            snapshot_id=unit.snapshot_id,
            compiler_unit_id=unit.compiler_unit_id,
            qbank_id=candidate.qbank_id,
            plan_id=program.plan_id,
            program_execution_id=dry_run.execution_id,
            producer_model_call_id=trace.model_call_id,
            feedback_hash="pending",
            round_id=program.round_id,
            verifier_pass=action is DoubterAction.PASS,
            verifier_score=(1.0 if action is DoubterAction.PASS else 0.0),
            evidence_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
                if action is DoubterAction.PASS
                else EvidenceSufficiency.INSUFFICIENT
            ),
            failure_reason=(issues[0].value if issues else None),
            suggested_revision=[item.directive.value for item in directives],
            missing_modalities=[
                item for item in candidate.required_modalities if not dimensions.modality_sufficient
            ],
            should_abstain=(
                action is DoubterAction.REJECT_CANDIDATE
                or candidate.route_class is QBankRouteClass.CONTROL
            ),
            raw_response_asset_id=trace.output_asset_id,
            created_at=trace.created_at,
            metadata={
                "typed_feedback": True,
                "provider_schema_conformant": provider_conformant,
            },
        )
        feedback = feedback.model_copy(update={"feedback_hash": feedback_content_hash(feedback)})
        async with (
            _doubter_persistence(
                "feedback_assessment_bundle", round_id=program.round_id, unit_ordinal=unit.ordinal
            ),
            ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow,
        ):
            assert uow.qbank_compiler is not None
            assert uow.doubter is not None
            await uow.qbank_compiler.add_feedback(feedback)
            await uow.doubter.add_assessment(assessment)
            if not provider_conformant:
                raw_validation_errors = raw.get("validation_errors")
                safe_validation_errors = (
                    [
                        StructuredOutputValidationError(
                            location=str(item.get("location", "$")),
                            error_type=str(item.get("error_type", "validation_error")),
                        )
                        for item in raw_validation_errors[:10]
                        if isinstance(item, Mapping)
                    ]
                    if isinstance(raw_validation_errors, list)
                    else []
                )
                await uow.structured_output_diagnostics.add(
                    StructuredOutputItemDiagnostic(
                        diagnostic_id=stable_compiler_id(
                            "structured-output-diagnostic",
                            [trace.model_call_id, "doubter", 0],
                        ),
                        run_id=run_id,
                        model_call_id=trace.model_call_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        stage=StructuredOutputStage.DOUBTER,
                        item_ordinal=0,
                        item_hash=canonical_content_hash([trace.model_call_id, candidate.qbank_id]),
                        status=StructuredOutputItemStatus.CONSERVATIVE_REJECT,
                        parser_version=self.config.parser_version,
                        errors=(
                            safe_validation_errors
                            or [
                                StructuredOutputValidationError(
                                    location=(provider_error_dimension or "$"),
                                    error_type=(
                                        provider_error_code or "doubter_structured_output_invalid"
                                    ),
                                )
                            ]
                        ),
                        created_at=trace.created_at,
                    )
                )
        if not provider_conformant:
            print(
                "DOUBTER_PROVIDER_REJECTION "
                + json.dumps(
                    {
                        "round_id": program.round_id,
                        "unit_ordinal": unit.ordinal,
                        "state": "terminal_rejected",
                        "error_code": provider_error_code,
                        "parser_error_codes": parser_diagnostic_codes,
                        "finish_reason": raw.get("provider_finish_reason", "unknown"),
                        "output_char_count": raw.get("provider_output_char_count"),
                        "diagnostic_persisted": True,
                        "repair_attempts": 0,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return assessment

    async def _run_revision_jobs(
        self,
        *,
        run_id: str,
        round_id: int,
        candidates: Sequence[QuestionCandidate],
        unit_by_id: Mapping[str, CompilationUnit],
        program_by_key: Mapping[tuple[str, int], MemoryProgram],
        contracts: Mapping[str, PlannerRequirementContract | None],
        assessments: Mapping[tuple[str, int], DoubterAssessment],
        expected: Mapping[str, Sequence[OperatorEvidence]],
        memory_index: MemoryOperatorIndexManifest,
        query_vectors: Mapping[str, ProgramQueryVectorSet],
    ) -> None:
        target_round = round_id + 1
        if not candidates:
            return
        await self._ensure_jobs(
            run_id=run_id,
            stage=CandidateStageKind.PLANNER_REVISION,
            round_id=target_round,
            candidates=candidates,
        )
        by_id = {item.qbank_id: item for item in candidates}
        started = time.monotonic()
        print(
            "DOUBTER_REVISION_PIPELINE "
            + json.dumps(
                {
                    "state": "starting",
                    "round_id": target_round,
                    "candidates": len(candidates),
                    "database_work_limit": self.database_work_concurrency,
                    "revision_workers": min(
                        8 if self.planner_config.is_v14 else self.concurrency, len(candidates)
                    ),
                    "dry_run_workers": self.dry_run_concurrency,
                    "model_concurrency": self.concurrency,
                    "recovery_lookup": "candidate_round",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        # Typed patches and dry-runs share a CPU/DB budget, separate from model concurrency.
        database_slots = asyncio.Semaphore(self.database_work_concurrency)
        dry_run_semaphore = asyncio.Semaphore(self.dry_run_concurrency)
        dry_run_tasks: list[asyncio.Task[None]] = []

        async def worker(index: int) -> None:
            worker_id = f"planner-revision-r{target_round}-{index:02d}"
            while True:
                async with (
                    database_slots,
                    _doubter_persistence(
                        "revision_worker", round_id=target_round, unit_ordinal=None
                    ),
                ):
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.doubter is not None
                        job = await uow.doubter.claim_candidate_job(
                            run_id=run_id,
                            stage=CandidateStageKind.PLANNER_REVISION,
                            round_id=target_round,
                            worker_id=worker_id,
                            lease_seconds=self.lease_seconds,
                            qbank_ids=set(by_id),
                        )
                    if job is None:
                        return
                    candidate = by_id.get(job.qbank_id)
                    try:
                        if candidate is not None:
                            contract = contracts[candidate.qbank_id]
                            if contract is None:
                                raise RuntimeError("revision requirement contract is missing")
                            revised = await self._revise_one(
                                run_id=run_id,
                                unit=unit_by_id[job.compiler_unit_id],
                                candidate=candidate,
                                contract=contract,
                                source_program=program_by_key[(candidate.qbank_id, round_id)],
                                assessment=assessments[(candidate.qbank_id, round_id)],
                                expected=expected.get(candidate.qbank_id, ()),
                            )
                            if revised is None:
                                async with ExperimentUnitOfWork(
                                    self.sessions, DataScope.RUNTIME
                                ) as uow:
                                    assert uow.doubter is not None
                                    await uow.doubter.cancel_candidate_job(
                                        job.candidate_stage_job_id,
                                        worker_id=worker_id,
                                        terminal_reason="planner_revision_unavailable",
                                    )
                                continue
                            await self._ensure_jobs(
                                run_id=run_id,
                                stage=CandidateStageKind.DRY_RUN,
                                round_id=target_round,
                                candidates=[candidate],
                            )
                            dry_run_tasks.append(
                                asyncio.create_task(
                                    self._run_pipelined_revision_dry_job(
                                        run_id=run_id,
                                        target_round=target_round,
                                        candidate=candidate,
                                        unit=unit_by_id[job.compiler_unit_id],
                                        program=revised,
                                        expected_evidence=expected.get(candidate.qbank_id, ()),
                                        memory_index=memory_index,
                                        query_vectors=query_vectors[candidate.qbank_id],
                                        semaphore=dry_run_semaphore,
                                        database_slots=database_slots,
                                    )
                                )
                            )
                        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                            assert uow.doubter is not None
                            await uow.doubter.finish_candidate_job(
                                job.candidate_stage_job_id,
                                worker_id=worker_id,
                                succeeded=True,
                            )
                    except Exception as exc:
                        diagnostics = safe_database_error_diagnostics(exc)
                        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                            assert uow.doubter is not None
                            await uow.doubter.finish_candidate_job(
                                job.candidate_stage_job_id,
                                worker_id=worker_id,
                                succeeded=False,
                                error_type=type(exc).__name__,
                                error_summary=safe_database_error_summary(diagnostics),
                            )
                        raise

        revision_tasks = [
            asyncio.create_task(worker(index))
            for index in range(
                min(
                    8 if self.planner_config.is_v14 else self.concurrency,
                    max(1, len(candidates)),
                )
            )
        ]
        try:
            await _await_workers(revision_tasks)
        except BaseException:
            for task in dry_run_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*dry_run_tasks, return_exceptions=True)
            raise
        if dry_run_tasks:
            await _await_workers(dry_run_tasks)
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            jobs = await uow.doubter.list_candidate_jobs(
                run_id,
                stage=CandidateStageKind.PLANNER_REVISION,
                round_id=target_round,
            )
        relevant = [item for item in jobs if item.qbank_id in by_id]
        if any(
            item.status
            not in {
                CandidateStageJobStatus.SUCCEEDED,
                CandidateStageJobStatus.CANCELLED,
            }
            for item in relevant
        ):
            raise RuntimeError(
                f"Planner revision round {target_round} has unfinished candidate jobs"
            )
        print(
            "DOUBTER_REVISION_PIPELINE "
            + json.dumps(
                {
                    "state": "completed",
                    "round_id": target_round,
                    "revision_jobs": len(relevant),
                    "dry_run_tasks": len(dry_run_tasks),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    async def _run_pipelined_revision_dry_job(
        self,
        *,
        run_id: str,
        target_round: int,
        candidate: QuestionCandidate,
        unit: CompilationUnit,
        program: MemoryProgram,
        expected_evidence: Sequence[OperatorEvidence],
        memory_index: MemoryOperatorIndexManifest,
        query_vectors: ProgramQueryVectorSet,
        semaphore: asyncio.Semaphore,
        database_slots: asyncio.Semaphore,
    ) -> None:
        """Execute one revised program while other GPU revisions continue."""

        worker_id = (
            f"pipeline-dry-r{target_round}-{canonical_content_hash(candidate.qbank_id)[:10]}"
        )
        async with (
            semaphore,
            database_slots,
            _doubter_persistence(
                "revision_dry_run", round_id=target_round, unit_ordinal=unit.ordinal
            ),
        ):
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                assert uow.doubter is not None
                job = await uow.doubter.claim_candidate_job(
                    run_id=run_id,
                    stage=CandidateStageKind.DRY_RUN,
                    round_id=target_round,
                    worker_id=worker_id,
                    lease_seconds=self.lease_seconds,
                    qbank_ids={candidate.qbank_id},
                )
            if job is None:
                return
            try:
                await execute_program_round(
                    sessions=self.sessions,
                    config=self.foundation,
                    run_id=run_id,
                    unit=unit,
                    candidate=candidate,
                    program=program,
                    expected_evidence=expected_evidence,
                    memory_index=memory_index,
                    query_vectors=query_vectors,
                )
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.doubter is not None
                    await uow.doubter.finish_candidate_job(
                        job.candidate_stage_job_id,
                        worker_id=worker_id,
                        succeeded=True,
                    )
            except Exception as exc:
                diagnostics = safe_database_error_diagnostics(exc)
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.doubter is not None
                    await uow.doubter.finish_candidate_job(
                        job.candidate_stage_job_id,
                        worker_id=worker_id,
                        succeeded=False,
                        error_type=type(exc).__name__,
                        error_summary=safe_database_error_summary(diagnostics),
                    )
                raise

    async def _revise_one(
        self,
        *,
        run_id: str,
        unit: CompilationUnit,
        candidate: QuestionCandidate,
        contract: PlannerRequirementContract,
        source_program: MemoryProgram,
        assessment: DoubterAssessment,
        expected: Sequence[OperatorEvidence],
    ) -> MemoryProgram | None:
        target_round = source_program.round_id + 1
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.memory_programs is not None
            existing = await uow.memory_programs.get_for_candidate_round(
                run_id, candidate.qbank_id, target_round
            )
        if existing is not None:
            return existing
        source_strategy = source_program.metadata.get("strategy")
        if not isinstance(source_strategy, Mapping):
            self._record_revision_unavailable(
                "source_strategy_missing",
                target_round=target_round,
                unit_ordinal=unit.ordinal,
            )
            return None
        if self.planner_config.is_v14 or self.planner_config.is_v15:
            return await self._apply_typed_revision_patch(
                run_id=run_id,
                unit=unit,
                candidate=candidate,
                contract=contract,
                source_program=source_program,
                source_strategy=source_strategy,
                assessment=assessment,
                expected=expected,
            )
        context = {
            "prior_strategy": dict(source_strategy),
            "capability_trace": _safe_capability_trace(source_program),
            "directives": [
                {
                    "directive": item.directive.value,
                    "top_k_tier": item.top_k_tier,
                }
                for item in assessment.directives
            ],
        }
        model_call_id = stable_compiler_id(
            "planner-revision-model-call",
            [run_id, candidate.qbank_id, source_program.plan_id, target_round],
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            trace = await uow.model_calls.get(model_call_id)
        if trace is None:
            request = render_planner_request(
                request_id=model_call_id,
                config=self.planner_config,
                candidates=[("Q01", candidate)],
                requirement_contracts={"Q01": contract},
                revision_context=context,
            )
            audited = await self.execution.generate_audited(
                self.planner_gateway,
                request,
                ModelCallContext(
                    model_call_id=model_call_id,
                    run_id=run_id,
                    snapshot_id=unit.snapshot_id,
                    conversation_id=unit.conversation_id,
                    job_id=_compiler_job_id(run_id, unit.compiler_unit_id),
                    compiler_unit_id=unit.compiler_unit_id,
                    qbank_id=candidate.qbank_id,
                    plan_id=source_program.plan_id,
                    parent_model_call_id=assessment.producer_model_call_id,
                    stage=AppStage.PLANNER,
                ),
                output_parser=planner_output_parser(
                    expected_aliases=["Q01"],
                    config=self.planner_config,
                ),
            )
            trace = audited.trace
        raw = dict(trace.parsed_output or {})
        strategies = raw.get("strategies", [])
        strategy = None
        if (
            trace.status is TraceStatus.SUCCESS
            and trace.repair_attempt_count == 0
            and not trace.parse_error
            and isinstance(strategies, list)
            and len(strategies) == 1
        ):
            try:
                strategy = PlannerStrategyDraftV12.model_validate(strategies[0])
            except (TypeError, ValueError):
                strategy = None
        if strategy is None:
            invalid_diagnostics = raw.get("invalid_diagnostics", [])
            safe_parser_errors: list[dict[str, object]] = []
            if isinstance(invalid_diagnostics, list):
                for item in invalid_diagnostics[:4]:
                    if not isinstance(item, Mapping):
                        continue
                    errors = item.get("errors", [])
                    safe_errors = []
                    if isinstance(errors, list):
                        for error in errors[:8]:
                            if isinstance(error, Mapping):
                                safe_errors.append(
                                    {
                                        "location": str(error.get("location", "unknown")),
                                        "type": str(error.get("type", "validation_error")),
                                    }
                                )
                    safe_parser_errors.append(
                        {
                            "candidate_ordinal": item.get("candidate_ordinal", 0),
                            "errors": safe_errors,
                        }
                    )
            self._record_revision_unavailable(
                "planner_revision_output_invalid",
                target_round=target_round,
                unit_ordinal=unit.ordinal,
                diagnostics={
                    "parse_error_present": bool(trace.parse_error),
                    "repair_attempt_count": trace.repair_attempt_count,
                    "safe_parser_diagnostics": safe_parser_errors,
                    "strategy_count": len(strategies) if isinstance(strategies, list) else 0,
                    "trace_status": trace.status.value,
                },
            )
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                await uow.structured_output_diagnostics.add(
                    StructuredOutputItemDiagnostic(
                        diagnostic_id=stable_compiler_id(
                            "structured-output-diagnostic",
                            [trace.model_call_id, "planner_revision", 0],
                        ),
                        run_id=run_id,
                        model_call_id=trace.model_call_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        stage=StructuredOutputStage.PLANNER_REVISION,
                        item_ordinal=0,
                        item_hash=canonical_content_hash(
                            [trace.model_call_id, candidate.qbank_id, target_round]
                        ),
                        status=StructuredOutputItemStatus.CONSERVATIVE_REJECT,
                        parser_version=self.planner_config.parser_version,
                        errors=[
                            StructuredOutputValidationError(
                                location="$",
                                error_type="planner_revision_output_invalid",
                            )
                        ],
                        created_at=trace.created_at,
                    )
                )
            return None
        try:
            program = materialize_planner_strategy_program(
                run_id=run_id,
                snapshot_id=unit.snapshot_id,
                compiler_unit_id=unit.compiler_unit_id,
                producer_model_call_id=trace.model_call_id,
                candidate=candidate,
                requirement=contract,
                strategy=strategy,
                config=self.planner_config,
                round_id=target_round,
                revision_directives=assessment.directives,
            )
        except PlannerStrategyPolicyError as exc:
            diagnostics = exc.safe_diagnostics()
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                await uow.structured_output_diagnostics.add(
                    StructuredOutputItemDiagnostic(
                        diagnostic_id=stable_compiler_id(
                            "structured-output-diagnostic",
                            [trace.model_call_id, "planner_revision_policy", 0],
                        ),
                        run_id=run_id,
                        model_call_id=trace.model_call_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        stage=StructuredOutputStage.PLANNER_REVISION,
                        item_ordinal=0,
                        item_hash=canonical_content_hash(
                            [trace.model_call_id, target_round, exc.code]
                        ),
                        status=StructuredOutputItemStatus.CONSERVATIVE_REJECT,
                        parser_version=self.planner_config.parser_version,
                        errors=[StructuredOutputValidationError(location="$", error_type=exc.code)],
                        created_at=trace.created_at,
                        metadata={"strategy_policy_diagnostics": diagnostics},
                    )
                )
            self._revision_policy_rejections[exc.code] += 1
            self._record_revision_unavailable(
                exc.code,
                target_round=target_round,
                unit_ordinal=unit.ordinal,
                emit=False,
            )
            count = self._revision_policy_rejections[exc.code]
            if count == 1 or count % 50 == 0:
                print(
                    "PLANNER_REVISION_POLICY_REJECTION "
                    + json.dumps(
                        {
                            **diagnostics,
                            "occurrence_count": count,
                            "round_id": target_round,
                            "unit_ordinal": unit.ordinal,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            return None
        static = self.static_validator.validate(program)
        policy = self.policy.validate(program, candidate, contract)
        if not static.valid or not policy.valid:
            self._record_revision_unavailable(
                "post_materialization_validation_invalid",
                target_round=target_round,
                unit_ordinal=unit.ordinal,
                diagnostics={
                    "policy_issue_codes": sorted({item.code for item in policy.issues}),
                    "static_issue_codes": sorted({item.code for item in static.issues}),
                },
            )
            return None
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
        directive_hash = canonical_content_hash(
            [item.model_dump(mode="json") for item in assessment.directives]
        )
        lineage_payload = {
            "run_id": run_id,
            "qbank_id": candidate.qbank_id,
            "source_plan_id": source_program.plan_id,
            "revised_plan_id": program.plan_id,
            "source_feedback_id": assessment.feedback_id,
            "target_round_id": target_round,
            "producer_model_call_id": trace.model_call_id,
            "directive_hash": directive_hash,
        }
        lineage = PlannerRevisionLineage(
            revision_lineage_id=stable_compiler_id("planner-revision-lineage", lineage_payload),
            **lineage_payload,
            content_hash=canonical_content_hash(lineage_payload),
            created_at=trace.created_at,
            metadata={
                "single_candidate_scope": True,
                "prompt_hash": planner_revision_prompt_sha256(),
                "schema_hash": planner_response_schema_sha256(self.planner_config),
            },
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.memory_programs is not None
            assert uow.planner is not None
            assert uow.doubter is not None
            await uow.memory_programs.add(program, expected_evidence=list(expected))
            for fact in (
                _validation_fact(
                    run_id=run_id,
                    plan_id=program.plan_id,
                    kind=ProgramValidationKind.STATIC,
                    validator_version="memory-program-static-v4",
                    valid=True,
                    issues=static_issues,
                ),
                _validation_fact(
                    run_id=run_id,
                    plan_id=program.plan_id,
                    kind=ProgramValidationKind.POLICY,
                    validator_version=self.planner_config.policy_version,
                    valid=True,
                    issues=policy.issues,
                ),
            ):
                await uow.planner.add_validation(fact)
            await uow.doubter.add_revision(lineage)
        return program

    async def _apply_typed_revision_patch(
        self,
        *,
        run_id: str,
        unit: CompilationUnit,
        candidate: QuestionCandidate,
        contract: PlannerRequirementContract,
        source_program: MemoryProgram,
        source_strategy: Mapping[str, object],
        assessment: DoubterAssessment,
        expected: Sequence[OperatorEvidence],
    ) -> MemoryProgram | None:
        """Apply v1.4 directives without asking Planner to regenerate a strategy."""

        target_round = source_program.round_id + 1
        directive_payload = [
            item.model_dump(mode="json")
            for item in sorted(assessment.directives, key=lambda item: item.ordinal)
        ]
        directive_hash = canonical_content_hash(directive_payload)
        before_strategy_hash = canonical_content_hash(source_strategy)
        directive_effects: list[PlannerDirectiveEffect] = []

        def build_patch(
            *,
            applied: bool,
            rejection_code: str | None,
            revised_program: MemoryProgram | None = None,
        ) -> PlannerRevisionPatch:
            before_capabilities = {
                str(value)
                for value in source_program.metadata.get("effective_capabilities", [])
                if str(value) in {item.value for item in PlannerCapability}
            }
            after_capabilities = (
                {
                    str(value)
                    for value in revised_program.metadata.get("effective_capabilities", [])
                    if str(value) in {item.value for item in PlannerCapability}
                }
                if revised_program is not None
                else before_capabilities
            )
            payload = {
                "run_id": run_id,
                "qbank_id": candidate.qbank_id,
                "source_plan_id": source_program.plan_id,
                "target_round_id": target_round,
                "directive_hash": directive_hash,
                "before_strategy_hash": before_strategy_hash,
                "after_strategy_hash": canonical_content_hash(
                    revised_program.metadata.get("strategy", {})
                    if revised_program is not None
                    else source_strategy
                ),
                "before_program_hash": source_program.program_hash,
                "after_program_hash": (
                    revised_program.program_hash if revised_program is not None else None
                ),
                "applied": applied,
                "rejection_code": rejection_code,
                "capability_added": sorted(after_capabilities - before_capabilities),
                "capability_removed": sorted(before_capabilities - after_capabilities),
            }
            effect_payload = [e.model_dump(mode="json") for e in directive_effects]
            identity = {
                **payload,
                **({"directive_effects": effect_payload} if self.planner_config.is_v15 else {}),
            }
            return PlannerRevisionPatch(
                patch_id=stable_compiler_id("planner-revision-patch", identity),
                **payload,
                content_hash=canonical_content_hash(identity),
                created_at=assessment.created_at,
                metadata={
                    "typed_patch": True,
                    "source_feedback_id": assessment.feedback_id,
                    **(
                        {
                            "directive_effects": [
                                e.model_dump(mode="json") for e in directive_effects
                            ],
                        }
                        if self.planner_config.is_v15
                        else {}
                    ),
                },
            )

        async def persist_patch(*, applied: bool, rejection_code: str | None) -> None:
            async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as patch_uow:
                assert patch_uow.doubter is not None
                await patch_uow.doubter.add_revision_patch(
                    build_patch(applied=applied, rejection_code=rejection_code)
                )

        try:
            if self.planner_config.is_v15:
                program, directive_effects = apply_planner_directives(
                    source_program=source_program,
                    requirement=contract,
                    directives=assessment.directives,
                    config=self.planner_config,
                )
                if program is None:
                    reason = (
                        "candidate_unrepairable"
                        if any(e.status.value == "inapplicable" for e in directive_effects)
                        else "revision_saturated"
                    )
                    self._record_revision_unavailable(
                        reason,
                        target_round=target_round,
                        unit_ordinal=unit.ordinal,
                        diagnostics={"typed_patch": True},
                    )
                    await persist_patch(applied=False, rejection_code=reason)
                    return None
            else:
                strategy = PlannerStrategyDraftV12.model_validate(source_strategy)
                program = materialize_planner_strategy_program(
                    run_id=run_id,
                    snapshot_id=unit.snapshot_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    producer_model_call_id=None,
                    candidate=candidate,
                    requirement=contract,
                    strategy=strategy,
                    config=self.planner_config,
                    round_id=target_round,
                    revision_directives=assessment.directives,
                )
        except (PlannerStrategyPolicyError, TypeError, ValueError) as exc:
            code = (
                exc.code
                if isinstance(exc, PlannerStrategyPolicyError)
                else "planner_typed_patch_invalid"
            )
            self._record_revision_unavailable(
                code,
                target_round=target_round,
                unit_ordinal=unit.ordinal,
                diagnostics={
                    "directive_count": len(assessment.directives),
                    "typed_patch": True,
                },
            )
            await persist_patch(applied=False, rejection_code=code)
            return None

        def semantic_steps(value: MemoryProgram) -> list[dict[str, object]]:
            return [
                {
                    "op_name": step.op_name,
                    "args": step.args,
                    "expected_output_type": step.expected_output_type,
                }
                for step in value.operator_sequence
            ]

        before_steps = semantic_steps(source_program)
        after_steps = semantic_steps(program)
        if before_steps == after_steps:
            self._record_revision_unavailable(
                "planner_revision_patch_noop",
                target_round=target_round,
                unit_ordinal=unit.ordinal,
                diagnostics={
                    "directive_count": len(assessment.directives),
                    "typed_patch": True,
                },
            )
            await persist_patch(applied=False, rejection_code="planner_revision_patch_noop")
            return None
        static = self.static_validator.validate(program)
        policy = self.policy.validate(program, candidate, contract)
        if not static.valid or not policy.valid:
            self._record_revision_unavailable(
                "planner_revision_patch_validation_failed",
                target_round=target_round,
                unit_ordinal=unit.ordinal,
                diagnostics={
                    "policy_issue_codes": sorted({item.code for item in policy.issues}),
                    "static_issue_codes": sorted({item.code for item in static.issues}),
                    "typed_patch": True,
                },
            )
            await persist_patch(
                applied=False,
                rejection_code="planner_revision_patch_validation_failed",
            )
            return None
        before_capabilities = set(source_program.metadata.get("effective_capabilities", []))
        after_capabilities = set(program.metadata.get("effective_capabilities", []))
        patch_metadata = {
            "typed_patch": True,
            "program_origin": "server_typed_patch",
            "directive_model_call_stage": "doubter",
            "before_strategy_hash": canonical_content_hash(source_strategy),
            "after_strategy_hash": canonical_content_hash(program.metadata.get("strategy", {})),
            "before_program_hash": source_program.program_hash,
            "after_program_hash": program.program_hash,
            "before_step_hash": canonical_content_hash(before_steps),
            "after_step_hash": canonical_content_hash(after_steps),
            "capability_added": sorted(after_capabilities - before_capabilities),
            "capability_removed": sorted(before_capabilities - after_capabilities),
            "directive_payload_hash": directive_hash,
            **(
                {
                    "directive_effects": [e.model_dump(mode="json") for e in directive_effects],
                }
                if self.planner_config.is_v15
                else {}
            ),
        }
        lineage_payload = {
            "run_id": run_id,
            "qbank_id": candidate.qbank_id,
            "source_plan_id": source_program.plan_id,
            "revised_plan_id": program.plan_id,
            "source_feedback_id": assessment.feedback_id,
            "target_round_id": target_round,
            "producer_model_call_id": assessment.producer_model_call_id,
            "directive_hash": directive_hash,
        }
        lineage = PlannerRevisionLineage(
            revision_lineage_id=stable_compiler_id("planner-revision-lineage", lineage_payload),
            **lineage_payload,
            content_hash=canonical_content_hash(lineage_payload),
            created_at=assessment.created_at,
            metadata=patch_metadata,
        )
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
        async with (
            _doubter_persistence(
                "typed_revision_bundle", round_id=target_round, unit_ordinal=unit.ordinal
            ),
            ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow,
        ):
            assert uow.memory_programs is not None
            assert uow.planner is not None
            assert uow.doubter is not None
            await uow.memory_programs.add(program, expected_evidence=list(expected))
            for fact in (
                _validation_fact(
                    run_id=run_id,
                    plan_id=program.plan_id,
                    kind=ProgramValidationKind.STATIC,
                    validator_version=(
                        "memory-program-static-v7"
                        if self.planner_config.is_v15
                        else "memory-program-static-v6"
                        if self.foundation.operator_set_version == "pmmc-operators-v6"
                        else "memory-program-static-v5"
                    ),
                    valid=True,
                    issues=static_issues,
                ),
                _validation_fact(
                    run_id=run_id,
                    plan_id=program.plan_id,
                    kind=ProgramValidationKind.POLICY,
                    validator_version=self.planner_config.policy_version,
                    valid=True,
                    issues=policy.issues,
                ),
            ):
                await uow.planner.add_validation(fact)
            await uow.doubter.add_revision(lineage)
            await uow.doubter.add_revision_patch(
                build_patch(applied=True, rejection_code=None, revised_program=program)
            )
        self._typed_patch_counts[(target_round, _top_k(source_program), _top_k(program))] += 1
        occurrence_count = self._typed_patch_counts.total()
        if occurrence_count == 1 or occurrence_count % 50 == 0:
            print(
                "PLANNER_TYPED_PATCH "
                + json.dumps(
                    {
                        "capability_added": patch_metadata["capability_added"],
                        "capability_removed": patch_metadata["capability_removed"],
                        "directive_count": len(assessment.directives),
                        "program_origin": "server_typed_patch",
                        "directive_model_call_stage": "doubter",
                        "atomic_persistence": True,
                        "program_changed": before_steps != after_steps,
                        "before_top_k": _top_k(source_program),
                        "after_top_k": _top_k(program),
                        "occurrence_count": occurrence_count,
                        "scope": "execution_attempt_new",
                        "round_id": target_round,
                        "state": "applied",
                        "unit_ordinal": unit.ordinal,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return program

    async def _run_revision_dry_jobs(
        self,
        *,
        run_id: str,
        target_round: int,
        candidates: Sequence[QuestionCandidate],
        unit_by_id: Mapping[str, CompilationUnit],
        program_by_key: Mapping[tuple[str, int], MemoryProgram],
        expected: Mapping[str, Sequence[OperatorEvidence]],
        memory_index: MemoryOperatorIndexManifest,
        query_vectors: Mapping[str, ProgramQueryVectorSet],
    ) -> None:
        runnable = [item for item in candidates if (item.qbank_id, target_round) in program_by_key]
        if not runnable:
            return
        await self._ensure_jobs(
            run_id=run_id,
            stage=CandidateStageKind.DRY_RUN,
            round_id=target_round,
            candidates=runnable,
        )
        by_id = {item.qbank_id: item for item in runnable}
        semaphore = asyncio.Semaphore(self.dry_run_concurrency)

        async def worker(index: int) -> None:
            worker_id = f"dry-run-r{target_round}-{index:02d}"
            while True:
                async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                    assert uow.doubter is not None
                    job = await uow.doubter.claim_candidate_job(
                        run_id=run_id,
                        stage=CandidateStageKind.DRY_RUN,
                        round_id=target_round,
                        worker_id=worker_id,
                        lease_seconds=self.lease_seconds,
                        qbank_ids=set(by_id),
                    )
                if job is None:
                    return
                candidate = by_id.get(job.qbank_id)
                try:
                    if candidate is not None:
                        async with semaphore:
                            await execute_program_round(
                                sessions=self.sessions,
                                config=self.foundation,
                                run_id=run_id,
                                unit=unit_by_id[job.compiler_unit_id],
                                candidate=candidate,
                                program=program_by_key[(candidate.qbank_id, target_round)],
                                expected_evidence=expected.get(candidate.qbank_id, ()),
                                memory_index=memory_index,
                                query_vectors=query_vectors[candidate.qbank_id],
                            )
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.doubter is not None
                        await uow.doubter.finish_candidate_job(
                            job.candidate_stage_job_id,
                            worker_id=worker_id,
                            succeeded=True,
                        )
                except Exception as exc:
                    diagnostics = safe_database_error_diagnostics(exc)
                    async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
                        assert uow.doubter is not None
                        await uow.doubter.finish_candidate_job(
                            job.candidate_stage_job_id,
                            worker_id=worker_id,
                            succeeded=False,
                            error_type=type(exc).__name__,
                            error_summary=safe_database_error_summary(diagnostics),
                        )
                    raise

        await _await_workers(
            [
                asyncio.create_task(worker(index))
                for index in range(min(self.dry_run_concurrency, max(1, len(runnable))))
            ]
        )
        async with ExperimentUnitOfWork(self.sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            jobs = await uow.doubter.list_candidate_jobs(
                run_id,
                stage=CandidateStageKind.DRY_RUN,
                round_id=target_round,
            )
        relevant = [item for item in jobs if item.qbank_id in by_id]
        if any(item.status is not CandidateStageJobStatus.SUCCEEDED for item in relevant):
            raise RuntimeError(f"Dry-run round {target_round} has unfinished candidate jobs")


_DIRECT_REJECT = {
    DoubterIssueCode.CANDIDATE_UNSUPPORTED,
    DoubterIssueCode.ANSWER_NOT_ENTAILED,
    DoubterIssueCode.REFUSAL_INAPPROPRIATE,
    DoubterIssueCode.STRUCTURED_OUTPUT_INVALID,
}


__all__ = [
    "DoubterInputGateError",
    "DoubterRunResult",
    "DoubterRunner",
    "PostgresDoubterInputProvider",
    "verify_doubter_run",
]
