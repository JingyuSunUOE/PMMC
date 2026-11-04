"""Synthetic round-zero compiler facts shared by the Step 8 import tests."""

from datetime import UTC, datetime
from pathlib import Path

from pmmc.compiler.ids import canonical_content_hash
from pmmc.compiler.planner import (
    build_planner_requirement_contract,
    materialize_planner_strategy_program,
)
from pmmc.compiler.planner_config import load_planner_config
from pmmc.schemas.base import QuestionType
from pmmc.schemas.planner import (
    PlannerRequirementContract,
    PlannerRetrievalChannel,
    PlannerStrategyDraftV12,
    PlannerTopKTier,
    ProgramDryRunAssessment,
)
from pmmc.schemas.program import MemoryProgram
from pmmc.schemas.qbank import QuestionCandidate

_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def candidate() -> QuestionCandidate:
    return QuestionCandidate(
        qbank_id="q1",
        run_id="run",
        snapshot_id="snap",
        compiler_unit_id="unit",
        producer_model_call_id="q-call",
        question_hash="qh",
        conversation_id="conversation",
        created_after_session="s1",
        canonical_question="What color was the notebook?",
        expected_answer="Blue",
        question_type=QuestionType.TEXTUAL_FACT,
        source_memory_ids=["m1"],
        created_at=_TIME,
    )


def fixture() -> tuple[QuestionCandidate, PlannerRequirementContract, MemoryProgram]:
    q = candidate()
    config = load_planner_config(
        Path(__file__).resolve().parents[1] / "configs/compiler/planner-v1.5.yaml"
    )
    requirement = build_planner_requirement_contract(
        run_id="run", compiler_unit_id="unit", candidate=q, config=config
    )
    strategy = PlannerStrategyDraftV12(
        candidate_alias="Q1",
        retrieval_channels=[PlannerRetrievalChannel.TEXT_DENSE],
        top_k_tier=PlannerTopKTier.NARROW,
        strategy_code="text_fact",
        confidence=1.0,
    )
    program = materialize_planner_strategy_program(
        run_id="run",
        snapshot_id="snap",
        compiler_unit_id="unit",
        producer_model_call_id="p-call",
        candidate=q,
        requirement=requirement,
        strategy=strategy,
        config=config,
    )
    return (
        q,
        requirement.model_copy(update={"created_at": _TIME}),
        program.model_copy(update={"created_at": _TIME}),
    )


def dry_run(program: MemoryProgram) -> ProgramDryRunAssessment:
    facts = dict(
        run_id=program.run_id,
        compiler_unit_id=program.compiler_unit_id,
        qbank_id=program.qbank_id,
        plan_id=program.plan_id,
        execution_id="exec",
        status="success",
        control_action="continue",
        execution_success=True,
        nonempty_evidence=True,
        evidence_sufficient=True,
        expected_group_count=0,
        matched_group_count=0,
        expected_evidence_complete=True,
        modality_coverage={"text": True},
        required_modality_sufficient=True,
        pre_control_sufficient=True,
        raw_image_reached=False,
        context_operator_used=False,
        temporal_operator_used=False,
        successful_step_count=len(program.operator_sequence),
        elapsed_ms=1.0,
    )
    return ProgramDryRunAssessment(
        assessment_id="assessment",
        content_hash=canonical_content_hash(facts),
        created_at=_TIME,
        **facts,
    )
