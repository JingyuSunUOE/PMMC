"""Round projections, active-r2 decisions, and runtime-safe QBank freezing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import (
    canonical_content_hash,
    qbank_decision_content_hash,
    stable_compiler_id,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.schemas import (
    CandidateStageJob,
    CandidateStageJobStatus,
    CandidateStageKind,
    CompilationStage,
    CompilationStageResult,
    DataScope,
    DoubterAction,
    DoubterAssessment,
    DoubterIssueCode,
    MemoryProgram,
    PlannerRevisionPatch,
    ProgramDryRunAssessment,
    QBankDecision,
    QBankRoundProjectionManifest,
    QBankRoundProjectionMember,
    QBankRouteClass,
    QuestionCandidate,
    R0InputManifest,
    RoundProjectionOutcome,
    RoutingProjectionManifest,
    RoutingProjectionMember,
    TraceStatus,
)


@dataclass(frozen=True)
class VerifiedQBankResult:
    run_id: str
    projection_ids: tuple[str, ...]
    routing_projection_id: str
    candidate_count: int
    accepted_count: int
    answerable_count: int
    control_count: int
    active_round: int = 2
    max_round: int = 3


def _r0_ready(
    candidate: QuestionCandidate,
    assessment: ProgramDryRunAssessment,
) -> bool:
    if candidate.route_class is QBankRouteClass.CONTROL:
        return assessment.execution_success and assessment.abstain_triggered
    return (
        assessment.execution_success
        and assessment.nonempty_evidence
        and assessment.required_modality_sufficient
        and assessment.pre_control_sufficient
    )


def _terminal_assessment(
    assessments: Mapping[tuple[str, int], DoubterAssessment],
    *,
    qbank_id: str,
    projection_round: int,
) -> DoubterAssessment | None:
    for round_id in range(projection_round + 1):
        value = assessments.get((qbank_id, round_id))
        if value is not None and value.action in {
            DoubterAction.PASS,
            DoubterAction.REJECT_CANDIDATE,
        }:
            return value
    return None


def _last_assessment(
    assessments: Mapping[tuple[str, int], DoubterAssessment],
    *,
    qbank_id: str,
    projection_round: int,
) -> DoubterAssessment | None:
    return next(
        (
            assessments[(qbank_id, round_id)]
            for round_id in range(projection_round, -1, -1)
            if (qbank_id, round_id) in assessments
        ),
        None,
    )


def _project_member(
    *,
    run_id: str,
    projection_id: str,
    projection_round: int,
    candidate: QuestionCandidate,
    programs: Mapping[tuple[str, int], MemoryProgram],
    dry_runs: Mapping[str, ProgramDryRunAssessment],
    assessments: Mapping[tuple[str, int], DoubterAssessment],
    step8: bool = False,
    revision_patches: Sequence[PlannerRevisionPatch] = (),
) -> QBankRoundProjectionMember:
    plan: MemoryProgram | None = None
    feedback: DoubterAssessment | None = None
    passed_at: int | None = None
    reason = "pending"

    if projection_round == 0:
        plan = programs.get((candidate.qbank_id, 0))
        dry_run = dry_runs.get(plan.plan_id) if plan is not None else None
        if (
            step8
            and plan is not None
            and dry_run is not None
            and (
                plan.round_id != 0
                or plan.qbank_id != candidate.qbank_id
                or plan.run_id != run_id
                or dry_run.plan_id != plan.plan_id
                or dry_run.qbank_id != candidate.qbank_id
                or dry_run.run_id != run_id
            )
        ):
            raise RuntimeError("R0 native program/dry-run lineage mismatch")
        if plan is not None and dry_run is not None and _r0_ready(candidate, dry_run):
            outcome = RoundProjectionOutcome.ACCEPTED
            passed_at = 0
            reason = "r0_real_dry_run_ready"
        else:
            outcome = RoundProjectionOutcome.REJECTED
            plan = None
            reason = "r0_real_dry_run_not_ready"
    else:
        r0_plan = programs.get((candidate.qbank_id, 0))
        r0_dry_run = dry_runs.get(r0_plan.plan_id) if r0_plan is not None else None
        if r0_plan is None or r0_dry_run is None:
            outcome = RoundProjectionOutcome.REJECTED
            reason = "r0_candidate_not_admitted_to_doubter"
        else:
            feedback = _terminal_assessment(
                assessments,
                qbank_id=candidate.qbank_id,
                projection_round=projection_round,
            )
            if feedback is not None and feedback.action is DoubterAction.PASS:
                if not feedback.provider_schema_conformant:
                    raise RuntimeError("Doubter pass requires native provider output")
                plan = programs.get((candidate.qbank_id, feedback.round_id))
                if plan is None:
                    raise RuntimeError("Doubter pass references a missing program")
                dry_run = dry_runs.get(plan.plan_id)
                if step8 and (
                    feedback.plan_id != plan.plan_id
                    or dry_run is None
                    or feedback.execution_id != dry_run.execution_id
                    or not all(feedback.dimensions.model_dump().values())
                    or (
                        candidate.route_class is QBankRouteClass.ANSWERABLE
                        and not dry_run.expected_evidence_complete
                    )
                ):
                    raise RuntimeError(
                        "Step8 pass violates source/path or dry-run lineage contract"
                    )
                if dry_run is None or not _r0_ready(candidate, dry_run):
                    raise RuntimeError("Doubter pass cannot override an insufficient real dry-run")
                passed_at = feedback.round_id
                outcome = (
                    RoundProjectionOutcome.ACCEPTED
                    if passed_at == projection_round
                    else RoundProjectionOutcome.CARRIED_FORWARD
                )
                reason = "doubter_pass"
            elif feedback is not None:
                outcome = (
                    RoundProjectionOutcome.MAX_ROUNDS_EXHAUSTED
                    if DoubterIssueCode.MAX_ROUNDS_EXHAUSTED in feedback.issue_codes
                    else RoundProjectionOutcome.REJECTED
                )
                reason = (
                    "max_rounds_exhausted"
                    if outcome is RoundProjectionOutcome.MAX_ROUNDS_EXHAUSTED
                    else "doubter_reject"
                )
            else:
                last = _last_assessment(
                    assessments,
                    qbank_id=candidate.qbank_id,
                    projection_round=projection_round,
                )
                if (
                    last is not None
                    and last.action is DoubterAction.REVISE_PROGRAM
                    and (candidate.qbank_id, last.round_id + 1) not in programs
                    and projection_round > last.round_id
                ):
                    feedback = last
                    outcome = RoundProjectionOutcome.REJECTED
                    reason = "planner_revision_unavailable"
                else:
                    outcome = RoundProjectionOutcome.PENDING

    terminal_metadata: dict[str, object] = {}
    if step8:
        terminal_metadata["projection_terminal"] = True
        terminal_metadata["projection_round"] = projection_round
        if projection_round == 0:
            # Do not even read Doubter maps or patch facts for the no-Doubter arm.
            terminal_metadata["r0_uses_doubter_facts"] = False
        else:
            all_feedback = sorted(
                (a for (q, _r), a in assessments.items() if q == candidate.qbank_id),
                key=lambda a: a.round_id,
            )
            all_programs = [p for (q, _r), p in programs.items() if q == candidate.qbank_id]
            patches = [p for p in revision_patches if p.qbank_id == candidate.qbank_id]
            terminal = next(
                (
                    a
                    for a in all_feedback
                    if a.action
                    in {
                        DoubterAction.PASS,
                        DoubterAction.REJECT_CANDIDATE,
                    }
                ),
                None,
            )
            if terminal is not None:
                run_reason = (
                    "accepted"
                    if terminal.action is DoubterAction.PASS
                    else "max_revision_depth_exhausted"
                    if DoubterIssueCode.MAX_ROUNDS_EXHAUSTED in terminal.issue_codes
                    else str(
                        terminal.metadata.get("scientific_terminal_reason") or "candidate_rejected"
                    )
                )
            elif patches and not max(patches, key=lambda p: p.target_round_id).applied:
                run_reason = str(max(patches, key=lambda p: p.target_round_id).rejection_code)
            elif not all_programs:
                run_reason = "initial_program_unavailable"
            else:
                run_reason = "planner_revision_unavailable"
            terminal_metadata.update(
                {
                    "run_terminal_reason": run_reason,
                    "feedback_call_count": len({a.producer_model_call_id for a in all_feedback}),
                    "actual_max_revision_depth": max((p.round_id for p in all_programs), default=0),
                }
            )
            if outcome is RoundProjectionOutcome.PENDING:
                outcome = RoundProjectionOutcome.REJECTED
                reason = "not_accepted_within_revision_depth"
            elif reason == "planner_revision_unavailable":
                reason = run_reason
        terminal_metadata["projection_reason"] = reason

    member_payload = {
        "projection_id": projection_id,
        "run_id": run_id,
        "qbank_id": candidate.qbank_id,
        "route_class": candidate.route_class.value,
        "outcome": outcome.value,
        "effective_plan_id": plan.plan_id if plan is not None else None,
        "effective_feedback_id": feedback.feedback_id if feedback is not None else None,
        "passed_at_round": passed_at,
        "reason": reason,
        **({"terminal_state": terminal_metadata} if step8 else {}),
    }
    return QBankRoundProjectionMember(
        projection_member_id=stable_compiler_id("qbank-round-projection-member", member_payload),
        projection_id=projection_id,
        run_id=run_id,
        qbank_id=candidate.qbank_id,
        route_class=candidate.route_class,
        outcome=outcome,
        effective_plan_id=plan.plan_id if plan is not None else None,
        effective_feedback_id=(feedback.feedback_id if feedback is not None else None),
        passed_at_round=passed_at,
        content_hash=canonical_content_hash(member_payload),
        metadata={"reason_code": reason, **terminal_metadata},
    )


def build_round_projection(
    *,
    run_id: str,
    snapshot_id: str,
    r0_manifest: R0InputManifest,
    projection_round: int,
    candidates: Sequence[QuestionCandidate],
    programs: Mapping[tuple[str, int], MemoryProgram],
    dry_runs: Mapping[str, ProgramDryRunAssessment],
    assessments: Mapping[tuple[str, int], DoubterAssessment],
    compiler_schema_version: str | None = None,
    revision_patches: Sequence[PlannerRevisionPatch] = (),
) -> tuple[QBankRoundProjectionManifest, list[QBankRoundProjectionMember]]:
    projection_payload = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "round_id": projection_round,
        "source_r0_manifest_id": r0_manifest.r0_manifest_id,
        "r0_facts_hash": r0_manifest.facts_hash,
    }
    projection_id = stable_compiler_id("qbank-round-projection", projection_payload)
    members = [
        _project_member(
            run_id=run_id,
            projection_id=projection_id,
            projection_round=projection_round,
            candidate=candidate,
            programs=programs,
            dry_runs=dry_runs,
            assessments=assessments,
            step8=compiler_schema_version == "0.9.0",
            revision_patches=revision_patches,
        )
        for candidate in sorted(candidates, key=lambda item: item.qbank_id)
    ]
    accepted = sum(
        item.outcome in {RoundProjectionOutcome.ACCEPTED, RoundProjectionOutcome.CARRIED_FORWARD}
        for item in members
    )
    rejected = sum(
        item.outcome
        in {RoundProjectionOutcome.REJECTED, RoundProjectionOutcome.MAX_ROUNDS_EXHAUSTED}
        for item in members
    )
    members_hash = canonical_content_hash([item.model_dump(mode="json") for item in members])
    manifest_payload = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "round_id": projection_round,
        "source_r0_manifest_id": r0_manifest.r0_manifest_id,
        "projection_id": projection_id,
        "member_count": len(members),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "pending_count": len(members) - accepted - rejected,
        "members_hash": members_hash,
    }
    manifest = QBankRoundProjectionManifest(
        **manifest_payload,
        **({"schema_version": "0.4.0"} if compiler_schema_version == "0.9.0" else {}),
        content_hash=canonical_content_hash(manifest_payload),
        created_at=r0_manifest.created_at,
        metadata={
            "r0_uses_doubter_facts": False if projection_round == 0 else None,
            "active_projection": projection_round == 2,
            "r0_facts_hash": r0_manifest.facts_hash,
            **(
                {
                    "projection_policy_version": "verified-qbank-projection-v4",
                    "routing_schema_version": "verified-qbank-routing-v4",
                    "revision_depth_semantics": "maximum_plan_revision_depth",
                }
                if compiler_schema_version == "0.9.0"
                else {}
            ),
        },
    )
    return manifest, members


def _decision(
    *,
    run_id: str,
    snapshot_id: str,
    candidate: QuestionCandidate,
    member: QBankRoundProjectionMember,
    created_at: datetime,
) -> QBankDecision:
    accepted = member.outcome in {
        RoundProjectionOutcome.ACCEPTED,
        RoundProjectionOutcome.CARRIED_FORWARD,
    }
    reason = str(member.metadata.get("reason_code") or member.outcome.value)
    payload = {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "compiler_unit_id": candidate.compiler_unit_id,
        "qbank_id": candidate.qbank_id,
        "accepted": accepted,
        "final_plan_id": member.effective_plan_id if accepted else None,
        "final_feedback_id": member.effective_feedback_id,
        "reason_code": reason,
        "reason_summary": None,
        "verifier_pass": accepted,
        "verifier_score": 1.0 if accepted else 0.0,
    }
    if candidate.compiler_unit_id is None:
        raise ValueError("QBank decisions require compiler-unit lineage")
    value = QBankDecision(
        decision_id=stable_compiler_id("qbank-decision", [run_id, candidate.qbank_id, 2]),
        **payload,
        decision_hash="pending",
        created_at=created_at,
        metadata={
            "active_round": 2,
            "source_projection_member_id": member.projection_member_id,
            **(
                {
                    "run_terminal_reason": member.metadata["run_terminal_reason"],
                    "projection_reason": member.metadata["projection_reason"],
                    "feedback_call_count": member.metadata["feedback_call_count"],
                    "actual_max_revision_depth": member.metadata["actual_max_revision_depth"],
                }
                if "run_terminal_reason" in member.metadata
                else {}
            ),
        },
    )
    return value.model_copy(update={"decision_hash": qbank_decision_content_hash(value)})


async def _persist_decision_stage_results(
    sessions: async_sessionmaker[AsyncSession],
    *,
    run_id: str,
    decisions: Sequence[QBankDecision],
    created_at: datetime,
) -> None:
    decisions_by_unit: dict[str, list[QBankDecision]] = {}
    for decision in decisions:
        decisions_by_unit.setdefault(decision.compiler_unit_id, []).append(decision)
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.compiler_units is not None
        units = await uow.compiler_units.list_for_run(run_id)
        for unit in units:
            values = decisions_by_unit.get(unit.compiler_unit_id, [])
            accepted_count = sum(item.accepted for item in values)
            counts = {
                "produced": len(values),
                "accepted": accepted_count,
                "rejected": len(values) - accepted_count,
            }
            await uow.compiler_units.add_stage_result(
                CompilationStageResult(
                    stage_result_id=stable_compiler_id(
                        "compiler-stage",
                        [run_id, unit.compiler_unit_id, "decision", 0],
                    ),
                    run_id=run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    stage=CompilationStage.DECISION,
                    status=TraceStatus.SUCCESS,
                    produced_count=counts["produced"],
                    accepted_count=counts["accepted"],
                    rejected_count=counts["rejected"],
                    result_hash=canonical_content_hash([run_id, unit.compiler_unit_id, counts]),
                    started_at=created_at,
                    completed_at=created_at,
                    elapsed_ms=0.0,
                    metadata={"active_round": 2, "max_round": 3},
                )
            )


def _decision_job(
    *,
    run_id: str,
    candidate: QuestionCandidate,
    created_at: datetime,
) -> CandidateStageJob:
    if candidate.compiler_unit_id is None:
        raise ValueError("decision jobs require compiler-unit lineage")
    payload = [
        run_id,
        candidate.compiler_unit_id,
        candidate.qbank_id,
        CandidateStageKind.DECISION.value,
        2,
    ]
    return CandidateStageJob(
        candidate_stage_job_id=stable_compiler_id("candidate-stage-job", payload),
        run_id=run_id,
        compiler_unit_id=candidate.compiler_unit_id,
        qbank_id=candidate.qbank_id,
        stage=CandidateStageKind.DECISION,
        round_id=2,
        status=CandidateStageJobStatus.SUCCEEDED,
        attempt_count=1,
        started_at=created_at,
        finished_at=created_at,
        available_at=created_at,
        created_at=created_at,
        content_hash=canonical_content_hash(payload),
        metadata={"active_round": 2, "deterministic": True},
    )


async def freeze_verified_qbank(
    sessions: async_sessionmaker[AsyncSession],
    *,
    run_id: str,
) -> VerifiedQBankResult:
    """Persist r0-r3 projections and the runtime-safe active-r2 projection."""

    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.compiler_runs is not None
        assert uow.questioner is not None
        assert uow.memory_programs is not None
        assert uow.program_dry_runs is not None
        assert uow.qbank_indexes is not None
        assert uow.doubter is not None
        assert uow.qbank_compiler is not None
        compiler = await uow.compiler_runs.get(run_id)
        if compiler is None:
            raise KeyError(f"compiler run does not exist: {run_id}")
        r0_manifest = await uow.doubter.get_r0_manifest(run_id)
        if r0_manifest is None or not r0_manifest.hard_gate_passed:
            raise RuntimeError("verified QBank requires a passing immutable R0 input")
        candidates = await uow.questioner.list_eligible_questions(run_id)
        programs = await uow.memory_programs.list_for_run(run_id)
        dry_values = await uow.program_dry_runs.list_for_run(run_id)
        assessment_values = await uow.doubter.list_assessments(run_id)
        revision_patches = (
            await uow.doubter.list_revision_patches(run_id)
            if compiler.compiler_schema_version == "0.9.0"
            else []
        )
        qbank_index = await uow.qbank_indexes.get_for_run(run_id)
        existing_routing = await uow.doubter.get_routing_projection(run_id)
        existing_decisions = await uow.qbank_compiler.list_decisions(run_id)

    if qbank_index is None or qbank_index.status != "ready":
        raise RuntimeError("verified QBank requires a ready source QBank index")
    if existing_routing is not None:
        manifest, members = existing_routing
        if len(existing_decisions) != len(candidates):
            raise RuntimeError("persisted routing projection has incomplete decisions")
        async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
            assert uow.doubter is not None
            for candidate in candidates:
                await uow.doubter.add_candidate_job(
                    _decision_job(
                        run_id=run_id,
                        candidate=candidate,
                        created_at=r0_manifest.created_at,
                    )
                )
        await _persist_decision_stage_results(
            sessions,
            run_id=run_id,
            decisions=existing_decisions,
            created_at=r0_manifest.created_at,
        )
        return VerifiedQBankResult(
            run_id=run_id,
            projection_ids=tuple(
                stable_compiler_id(
                    "qbank-round-projection",
                    {
                        "run_id": run_id,
                        "snapshot_id": compiler.snapshot_id,
                        "round_id": round_id,
                        "source_r0_manifest_id": r0_manifest.r0_manifest_id,
                        "r0_facts_hash": r0_manifest.facts_hash,
                    },
                )
                for round_id in range(4)
            ),
            routing_projection_id=manifest.routing_projection_id,
            candidate_count=len(candidates),
            accepted_count=len(members),
            answerable_count=manifest.answerable_count,
            control_count=manifest.control_count,
        )

    program_by_key = {(item.qbank_id, item.round_id): item for item in programs}
    dry_by_plan = {item.plan_id: item for item in dry_values}
    assessment_by_key = {(item.qbank_id, item.round_id): item for item in assessment_values}
    projections = [
        build_round_projection(
            run_id=run_id,
            snapshot_id=compiler.snapshot_id,
            r0_manifest=r0_manifest,
            projection_round=round_id,
            candidates=candidates,
            programs=program_by_key,
            dry_runs=dry_by_plan,
            assessments=assessment_by_key,
            compiler_schema_version=compiler.compiler_schema_version,
            revision_patches=revision_patches,
        )
        for round_id in range(4)
    ]
    active_manifest, active_members = projections[2]
    candidate_by_id = {item.qbank_id: item for item in candidates}
    decisions = [
        _decision(
            run_id=run_id,
            snapshot_id=compiler.snapshot_id,
            candidate=candidate_by_id[item.qbank_id],
            member=item,
            created_at=r0_manifest.created_at,
        )
        for item in active_members
    ]
    accepted_members = [
        item
        for item in active_members
        if item.outcome in {RoundProjectionOutcome.ACCEPTED, RoundProjectionOutcome.CARRIED_FORWARD}
    ]
    decision_hash = canonical_content_hash([item.decision_hash for item in decisions])
    routing_id = stable_compiler_id(
        "routing-projection",
        [run_id, active_manifest.projection_id, decision_hash],
    )
    routing_members = [
        RoutingProjectionMember(
            routing_member_id=stable_compiler_id(
                "routing-projection-member",
                [routing_id, item.qbank_id, ordinal],
            ),
            routing_projection_id=routing_id,
            run_id=run_id,
            qbank_id=item.qbank_id,
            plan_id=str(item.effective_plan_id),
            route_class=item.route_class,
            source_projection_member_id=item.projection_member_id,
            ordinal=ordinal,
            content_hash=canonical_content_hash(
                [
                    routing_id,
                    item.qbank_id,
                    item.effective_plan_id,
                    item.route_class.value,
                    ordinal,
                ]
            ),
            metadata={"active_round": 2},
        )
        for ordinal, item in enumerate(accepted_members)
    ]
    members_hash = canonical_content_hash(
        [item.model_dump(mode="json") for item in routing_members]
    )
    lexical_collection_id = stable_compiler_id(
        "qbank-lexical-projection",
        [run_id, qbank_index.qbank_index_id, active_manifest.projection_id],
    )
    routing_payload = {
        "routing_projection_id": routing_id,
        "run_id": run_id,
        "snapshot_id": compiler.snapshot_id,
        "active_round": 2,
        "max_round": 3,
        "source_projection_id": active_manifest.projection_id,
        "decision_hash": decision_hash,
        "lexical_collection_id": lexical_collection_id,
        "text_collection_id": qbank_index.text_question_collection_id,
        "multimodal_collection_id": qbank_index.multimodal_question_collection_id,
        "member_count": len(routing_members),
        "answerable_count": sum(
            item.route_class is QBankRouteClass.ANSWERABLE for item in routing_members
        ),
        "control_count": sum(
            item.route_class is QBankRouteClass.CONTROL for item in routing_members
        ),
        "members_hash": members_hash,
    }
    routing_manifest = RoutingProjectionManifest(
        **routing_payload,
        **({"schema_version": "0.4.0"} if compiler.compiler_schema_version == "0.9.0" else {}),
        content_hash=canonical_content_hash(routing_payload),
        created_at=r0_manifest.created_at,
        metadata={
            "source_qbank_index_id": qbank_index.qbank_index_id,
            "embedding_recomputed": False,
            "answerable_scientific_budget_only": True,
            **(
                {
                    "projection_policy_version": "verified-qbank-projection-v4",
                    "routing_schema_version": "verified-qbank-routing-v4",
                    "exact_round_membership": True,
                }
                if compiler.compiler_schema_version == "0.9.0"
                else {}
            ),
        },
    )
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.doubter is not None
        assert uow.qbank_compiler is not None
        for projection_manifest, projection_members in projections:
            await uow.doubter.add_round_projection(
                projection_manifest,
                projection_members,
            )
        for decision in decisions:
            await uow.qbank_compiler.add_decision(decision)
        for candidate in candidates:
            await uow.doubter.add_candidate_job(
                _decision_job(
                    run_id=run_id,
                    candidate=candidate,
                    created_at=r0_manifest.created_at,
                )
            )
        await uow.doubter.add_routing_projection(
            routing_manifest,
            routing_members,
        )
    await _persist_decision_stage_results(
        sessions,
        run_id=run_id,
        decisions=decisions,
        created_at=r0_manifest.created_at,
    )
    return VerifiedQBankResult(
        run_id=run_id,
        projection_ids=tuple(item[0].projection_id for item in projections),
        routing_projection_id=routing_manifest.routing_projection_id,
        candidate_count=len(candidates),
        accepted_count=len(routing_members),
        answerable_count=routing_manifest.answerable_count,
        control_count=routing_manifest.control_count,
    )


async def verify_routing_projection(
    sessions: async_sessionmaker[AsyncSession],
    *,
    run_id: str,
) -> dict[str, object]:
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.questioner is not None
        assert uow.doubter is not None
        assert uow.qbank_compiler is not None
        assert uow.memory_programs is not None
        assert uow.program_dry_runs is not None
        assert uow.qbank_indexes is not None
        candidates = await uow.questioner.list_eligible_questions(run_id)
        r0_manifest = await uow.doubter.get_r0_manifest(run_id)
        projections = [
            await uow.doubter.get_round_projection(run_id, round_id) for round_id in range(4)
        ]
        routing = await uow.doubter.get_routing_projection(run_id)
        decisions = await uow.qbank_compiler.list_decisions(run_id)
        programs = await uow.memory_programs.list_for_run(run_id)
        dry_runs = await uow.program_dry_runs.list_for_run(run_id)
        qbank_index = await uow.qbank_indexes.get_for_run(run_id)
        decision_jobs = await uow.doubter.list_candidate_jobs(
            run_id,
            stage=CandidateStageKind.DECISION,
            round_id=2,
        )
    issues: list[str] = []
    candidate_by_id = {item.qbank_id: item for item in candidates}
    candidate_ids = set(candidate_by_id)
    if len(candidate_by_id) != len(candidates):
        issues.append("eligible_candidate_id_duplicate")
    program_by_id = {item.plan_id: item for item in programs}
    dry_run_by_plan = {item.plan_id: item for item in dry_runs}
    if len(program_by_id) != len(programs):
        issues.append("program_id_duplicate")
    if len(dry_run_by_plan) != len(dry_runs):
        issues.append("dry_run_plan_duplicate")
    if r0_manifest is None or not r0_manifest.hard_gate_passed:
        issues.append("r0_input_manifest_invalid")
    if any(item is None for item in projections):
        issues.append("round_projection_missing")
    projection_by_round: dict[
        int,
        tuple[QBankRoundProjectionManifest, dict[str, QBankRoundProjectionMember]],
    ] = {}
    if r0_manifest is not None:
        for round_id, projection in enumerate(projections):
            if projection is None:
                continue
            manifest, members = projection
            member_by_id = {item.qbank_id: item for item in members}
            projection_by_round[round_id] = (manifest, member_by_id)
            if (
                len(member_by_id) != len(members)
                or set(member_by_id) != candidate_ids
                or manifest.member_count != len(members)
            ):
                issues.append(f"round_projection_denominator_mismatch:r{round_id}")
            projection_identity = {
                "run_id": run_id,
                "snapshot_id": r0_manifest.snapshot_id,
                "round_id": round_id,
                "source_r0_manifest_id": r0_manifest.r0_manifest_id,
                "r0_facts_hash": r0_manifest.facts_hash,
            }
            expected_projection_id = stable_compiler_id(
                "qbank-round-projection",
                projection_identity,
            )
            if (
                manifest.projection_id != expected_projection_id
                or manifest.run_id != run_id
                or manifest.snapshot_id != r0_manifest.snapshot_id
                or manifest.round_id != round_id
                or manifest.source_r0_manifest_id != r0_manifest.r0_manifest_id
                or manifest.metadata.get("r0_facts_hash") != r0_manifest.facts_hash
            ):
                issues.append(f"round_projection_identity_mismatch:r{round_id}")
            accepted_count = 0
            rejected_count = 0
            for qbank_id, member in member_by_id.items():
                candidate = candidate_by_id.get(qbank_id)
                reason = str(member.metadata.get("reason_code") or "")
                member_payload = {
                    "projection_id": manifest.projection_id,
                    "run_id": run_id,
                    "qbank_id": qbank_id,
                    "route_class": member.route_class.value,
                    "outcome": member.outcome.value,
                    "effective_plan_id": member.effective_plan_id,
                    "effective_feedback_id": member.effective_feedback_id,
                    "passed_at_round": member.passed_at_round,
                    "reason": reason,
                    **(
                        {
                            "terminal_state": {
                                key: value
                                for key, value in member.metadata.items()
                                if key != "reason_code"
                            }
                        }
                        if manifest.schema_version == "0.4.0"
                        else {}
                    ),
                }
                if (
                    candidate is None
                    or member.projection_id != manifest.projection_id
                    or member.run_id != run_id
                    or member.route_class is not candidate.route_class
                    or member.projection_member_id
                    != stable_compiler_id(
                        "qbank-round-projection-member",
                        member_payload,
                    )
                    or member.content_hash != canonical_content_hash(member_payload)
                    or (member.passed_at_round is not None and member.passed_at_round > round_id)
                ):
                    issues.append(f"round_projection_member_identity_mismatch:r{round_id}")
                if member.effective_plan_id is not None:
                    program = program_by_id.get(member.effective_plan_id)
                    dry_run = dry_run_by_plan.get(member.effective_plan_id)
                    if (
                        program is None
                        or program.qbank_id != qbank_id
                        or program.round_id > round_id
                        or dry_run is None
                        or candidate is None
                        or not _r0_ready(candidate, dry_run)
                    ):
                        issues.append(f"round_projection_effective_plan_invalid:r{round_id}")
                if manifest.schema_version == "0.4.0":
                    if not member.metadata.get("projection_terminal") or (
                        member.metadata.get("projection_round") != round_id
                    ):
                        issues.append(f"round_projection_terminal_metadata_invalid:r{round_id}")
                    if round_id == 0 and (
                        member.effective_feedback_id is not None
                        or member.metadata.get("r0_uses_doubter_facts") is not False
                    ):
                        issues.append("round_projection_r0_doubter_contamination")
                if member.outcome in {
                    RoundProjectionOutcome.ACCEPTED,
                    RoundProjectionOutcome.CARRIED_FORWARD,
                }:
                    accepted_count += 1
                elif member.outcome in {
                    RoundProjectionOutcome.REJECTED,
                    RoundProjectionOutcome.MAX_ROUNDS_EXHAUSTED,
                }:
                    rejected_count += 1
            pending_count = len(members) - accepted_count - rejected_count
            members_hash = canonical_content_hash(
                [
                    item.model_dump(mode="json")
                    for item in sorted(members, key=lambda item: item.qbank_id)
                ]
            )
            manifest_payload = {
                "run_id": run_id,
                "snapshot_id": r0_manifest.snapshot_id,
                "round_id": round_id,
                "source_r0_manifest_id": r0_manifest.r0_manifest_id,
                "projection_id": manifest.projection_id,
                "member_count": len(members),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "pending_count": pending_count,
                "members_hash": members_hash,
            }
            if (
                manifest.accepted_count != accepted_count
                or manifest.rejected_count != rejected_count
                or manifest.pending_count != pending_count
                or manifest.members_hash != members_hash
                or manifest.content_hash != canonical_content_hash(manifest_payload)
            ):
                issues.append(f"round_projection_hash_or_count_mismatch:r{round_id}")
            if pending_count and (round_id == 3 or manifest.schema_version == "0.4.0"):
                issues.append(f"round_projection_r{round_id}_pending")
    if routing is None:
        issues.append("routing_projection_missing")
        routing_count = 0
    else:
        routing_manifest, routing_members = routing
        routing_count = len(routing_members)
        if (
            routing_manifest.member_count != routing_count
            or routing_manifest.active_round != 2
            or routing_manifest.max_round != 3
        ):
            issues.append("routing_member_count_or_round_mismatch")
    decision_by_id = {item.qbank_id: item for item in decisions}
    if len(decision_by_id) != len(decisions) or set(decision_by_id) != candidate_ids:
        issues.append("decision_denominator_mismatch")
    active_projection = projection_by_round.get(2)
    active_members = active_projection[1] if active_projection is not None else {}
    for qbank_id, decision in decision_by_id.items():
        active_member = active_members.get(qbank_id)
        accepted = active_member is not None and active_member.outcome in {
            RoundProjectionOutcome.ACCEPTED,
            RoundProjectionOutcome.CARRIED_FORWARD,
        }
        if (
            active_member is None
            or decision.run_id != run_id
            or decision.snapshot_id != (r0_manifest.snapshot_id if r0_manifest is not None else "")
            or decision.accepted != accepted
            or decision.final_plan_id != (active_member.effective_plan_id if accepted else None)
            or decision.final_feedback_id != active_member.effective_feedback_id
            or decision.metadata.get("source_projection_member_id")
            != active_member.projection_member_id
            or decision.decision_hash != qbank_decision_content_hash(decision)
        ):
            issues.append("active_r2_decision_lineage_mismatch")
    if (
        {item.qbank_id for item in decision_jobs} != candidate_ids
        or len(decision_jobs) != len(candidates)
        or any(item.status is not CandidateStageJobStatus.SUCCEEDED for item in decision_jobs)
    ):
        issues.append("decision_job_denominator_mismatch")
    accepted_ids = {item.qbank_id for item in decisions if item.accepted}
    routing_ids = {item.qbank_id for item in routing[1]} if routing is not None else set()
    if accepted_ids != routing_ids:
        issues.append("decision_routing_mismatch")
    if routing is not None and active_projection is not None:
        routing_manifest, routing_members = routing
        active_manifest, active_member_by_id = active_projection
        decision_hash = canonical_content_hash(
            [item.decision_hash for item in sorted(decisions, key=lambda item: item.qbank_id)]
        )
        expected_lexical_id = (
            stable_compiler_id(
                "qbank-lexical-projection",
                [
                    run_id,
                    qbank_index.qbank_index_id,
                    active_manifest.projection_id,
                ],
            )
            if qbank_index is not None
            else None
        )
        if (
            routing_manifest.source_projection_id != active_manifest.projection_id
            or routing_manifest.decision_hash != decision_hash
            or qbank_index is None
            or qbank_index.status != "ready"
            or routing_manifest.lexical_collection_id != expected_lexical_id
            or routing_manifest.text_collection_id != qbank_index.text_question_collection_id
            or routing_manifest.multimodal_collection_id
            != qbank_index.multimodal_question_collection_id
        ):
            issues.append("routing_projection_source_identity_mismatch")
        sorted_routing = sorted(routing_members, key=lambda item: item.ordinal)
        if [item.ordinal for item in sorted_routing] != list(range(len(sorted_routing))):
            issues.append("routing_member_ordinals_incomplete")
        answerable_count = 0
        control_count = 0
        for routing_member in sorted_routing:
            source_member = active_member_by_id.get(routing_member.qbank_id)
            routing_decision = decision_by_id.get(routing_member.qbank_id)
            routing_member_payload = [
                routing_manifest.routing_projection_id,
                routing_member.qbank_id,
                routing_member.plan_id,
                routing_member.route_class.value,
                routing_member.ordinal,
            ]
            if (
                source_member is None
                or routing_decision is None
                or not routing_decision.accepted
                or routing_member.routing_projection_id != routing_manifest.routing_projection_id
                or routing_member.run_id != run_id
                or routing_member.source_projection_member_id != source_member.projection_member_id
                or routing_member.plan_id != source_member.effective_plan_id
                or routing_member.route_class is not source_member.route_class
                or routing_member.content_hash != canonical_content_hash(routing_member_payload)
            ):
                issues.append("routing_member_lineage_mismatch")
            if routing_member.route_class is QBankRouteClass.ANSWERABLE:
                answerable_count += 1
            else:
                control_count += 1
        members_hash = canonical_content_hash(
            [item.model_dump(mode="json") for item in sorted_routing]
        )
        routing_payload = {
            "routing_projection_id": routing_manifest.routing_projection_id,
            "run_id": run_id,
            "snapshot_id": (r0_manifest.snapshot_id if r0_manifest is not None else ""),
            "active_round": 2,
            "max_round": 3,
            "source_projection_id": active_manifest.projection_id,
            "decision_hash": decision_hash,
            "lexical_collection_id": routing_manifest.lexical_collection_id,
            "text_collection_id": routing_manifest.text_collection_id,
            "multimodal_collection_id": routing_manifest.multimodal_collection_id,
            "member_count": len(sorted_routing),
            "answerable_count": answerable_count,
            "control_count": control_count,
            "members_hash": members_hash,
        }
        if (
            routing_manifest.answerable_count != answerable_count
            or routing_manifest.control_count != control_count
            or routing_manifest.members_hash != members_hash
            or routing_manifest.content_hash != canonical_content_hash(routing_payload)
        ):
            issues.append("routing_projection_hash_or_count_mismatch")
    return {
        "run_id": run_id,
        "valid": not issues,
        "issues": issues,
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "decision_job_count": len(decision_jobs),
        "routing_member_count": routing_count,
        "projection_count": sum(item is not None for item in projections),
        "active_round": 2,
        "max_round": 3,
    }


__all__ = [
    "VerifiedQBankResult",
    "build_round_projection",
    "freeze_verified_qbank",
    "verify_routing_projection",
]
