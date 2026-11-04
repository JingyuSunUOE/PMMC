from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pmmc.runtime.qbank_matcher import (
    MatchGridKey,
    RuntimeMatchResult,
    RuntimeQBankChannelHit,
    apply_reranker_assessments,
    decide_runtime_route,
    rank_runtime_match_grid,
    reranker_pair_ids,
)
from pmmc.runtime.query_executor_config import load_query_executor_config
from pmmc.schemas import (
    Modality,
    QBankRerankerAssessment,
    QBankRerankerDimensions,
    QBankRouteClass,
    QBankRuntimeChannel,
    QueryEvidenceRelevanceAssessment,
    QueryEvidenceRelevanceItem,
    QueryEvidenceSufficiencyAssessment,
    QueryFallbackReason,
    QueryRouteOutcome,
    RerankerDimensionVerdict,
    RerankerMatchLevel,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_query_executor_config(ROOT / "configs/runtime/query-executor-v2.yaml")


def _hit(
    qbank_id: str,
    *,
    channel: QBankRuntimeChannel,
    score: float,
    generation_rank: int = 1,
    accepted_round: int = 0,
) -> RuntimeQBankChannelHit:
    return RuntimeQBankChannelHit(
        qbank_id=qbank_id,
        plan_id=f"plan-{qbank_id}",
        route_class=QBankRouteClass.ANSWERABLE,
        generation_rank=generation_rank,
        accepted_round=accepted_round,
        required_modalities=(Modality.TEXT,),
        channel=channel,
        raw_score=score,
        candidate_question=f"Question for {qbank_id}?",
        question_type="textual_fact",
    )


def _assessment(
    qbank_id: str,
    *,
    level: RerankerMatchLevel,
) -> QBankRerankerAssessment:
    verdict = (
        RerankerDimensionVerdict.FAIL
        if level is RerankerMatchLevel.NONE
        else RerankerDimensionVerdict.PASS
    )
    effective = RerankerMatchLevel.NONE if verdict is RerankerDimensionVerdict.FAIL else level
    return QBankRerankerAssessment(
        reranker_assessment_id=f"rerank-{qbank_id}",
        query_executor_run_id="query-run-1",
        query_id="query-1",
        qbank_id=qbank_id,
        model_call_id=f"model-call-{qbank_id}",
        dimensions=QBankRerankerDimensions(
            same_information_need=verdict,
            candidate_can_answer_query=RerankerDimensionVerdict.PASS,
            answer_type_compatible=RerankerDimensionVerdict.PASS,
            modality_compatible=RerankerDimensionVerdict.NOT_APPLICABLE,
        ),
        provider_match_level=level,
        effective_match_level=effective,
        reranker_score=effective.score,
        provider_schema_conformant=True,
        content_hash=f"hash-{qbank_id}",
    )


def _base_grid() -> dict[MatchGridKey, RuntimeMatchResult]:
    return rank_runtime_match_grid(
        query_executor_run_id="query-run-1",
        query_id="query-1",
        hits=[
            _hit("q1", channel=QBankRuntimeChannel.LEXICAL, score=9.0),
            _hit("q1", channel=QBankRuntimeChannel.TEXT, score=0.9),
            _hit(
                "q2",
                channel=QBankRuntimeChannel.LEXICAL,
                score=12.0,
                generation_rank=2,
                accepted_round=3,
            ),
            _hit(
                "q2",
                channel=QBankRuntimeChannel.TEXT,
                score=0.95,
                generation_rank=2,
                accepted_round=3,
            ),
        ],
        has_query_image=False,
        config=CONFIG,
    )


def test_matcher_recomputes_all_sixteen_budget_round_projections() -> None:
    grid = _base_grid()
    assert len(grid) == 16
    assert [item.qbank_id for item in grid[MatchGridKey(1, 0)].candidates] == ["q1"]
    assert [item.qbank_id for item in grid[MatchGridKey(8, 2)].candidates] == ["q1"]
    assert {item.qbank_id for item in grid[MatchGridKey(2, 3)].candidates} == {
        "q1",
        "q2",
    }

    q1 = grid[MatchGridKey(1, 0)].candidates[0]
    lexical = 9.0 / 10.0
    text = (0.9 + 1.0) / 2.0
    expected_base = 0.4 * lexical + 0.6 * text
    assert q1.base_score == pytest.approx(expected_base)
    assert q1.base_reliable_score == pytest.approx(expected_base)


def test_reranker_union_deduplicates_pairs_across_all_grid_cells() -> None:
    assert reranker_pair_ids(_base_grid(), top_k=20) == ("q1", "q2")


def test_reranker_score_controls_active_route_and_weak_none_fallback() -> None:
    grid = _base_grid()
    exact = apply_reranker_assessments(
        grid=grid,
        assessments={"q1": _assessment("q1", level=RerankerMatchLevel.EXACT)},
        config=CONFIG,
    )
    active = exact[MatchGridKey(8, 2)].candidates
    assert active[0].route_score == pytest.approx(
        0.4 * (active[0].base_reliable_score or 0.0) + 0.6
    )
    decision = decide_runtime_route(
        query_executor_run_id="query-run-1",
        query_id="query-1",
        match_batch_id="match-1",
        candidates=active,
        config=CONFIG,
    )
    assert decision.outcome is QueryRouteOutcome.PROGRAM
    assert decision.selected_qbank_id == "q1"

    rejected = apply_reranker_assessments(
        grid=grid,
        assessments={"q1": _assessment("q1", level=RerankerMatchLevel.NONE)},
        config=CONFIG,
    )
    fallback = decide_runtime_route(
        query_executor_run_id="query-run-1",
        query_id="query-1",
        match_batch_id="match-1",
        candidates=rejected[MatchGridKey(8, 2)].candidates,
        config=CONFIG,
    )
    assert fallback.outcome is QueryRouteOutcome.FALLBACK
    assert fallback.fallback_reason is QueryFallbackReason.RERANKER_REJECTED


def test_query_evidence_relevance_is_server_derived_per_group() -> None:
    item = QueryEvidenceRelevanceItem(
        relevance_item_id="item-1",
        relevance_assessment_id="assessment-1",
        query_executor_run_id="query-run-1",
        query_id="query-1",
        execution_id="execution-1",
        ordinal=0,
        evidence_id_hash="evidence-hash",
        evidence_group_hash="group-hash",
        modality=Modality.TEXT,
        cosine_similarity=0.2,
        normalized_score=0.6,
        relevant=True,
        threshold=0.55,
        content_hash="item-hash",
    )
    assert item.relevant
    assessment = QueryEvidenceRelevanceAssessment(
        relevance_assessment_id="assessment-1",
        query_executor_run_id="query-run-1",
        query_id="query-1",
        query_program_binding_id="binding-1",
        relevance_threshold=0.55,
        required_group_count=2,
        relevant_group_count=1,
        passed=False,
        content_hash="assessment-hash",
    )
    assert not assessment.passed

    with pytest.raises(ValidationError, match="server-derived"):
        QueryEvidenceRelevanceItem.model_validate(
            {**item.model_dump(mode="python"), "relevant": False}
        )


def test_runtime_evidence_v2_cannot_be_sufficient_when_query_relevance_fails() -> None:
    assessment = QueryEvidenceSufficiencyAssessment(
        sufficiency_assessment_id="sufficiency-1",
        query_executor_run_id="query-run-1",
        query_id="query-1",
        query_program_binding_id="binding-1",
        execution_id="execution-1",
        technical_success=True,
        nonempty_evidence=True,
        actual_capabilities=["text"],
        required_capabilities=["text"],
        actual_modalities=[Modality.TEXT],
        required_modalities=[Modality.TEXT],
        independent_group_count=1,
        minimum_evidence_groups=1,
        raw_image_reached=False,
        raw_image_required=False,
        structural_requirement_success=True,
        query_evidence_relevance=False,
        answer_support=False,
        relevant_evidence_group_count=0,
        relevance_threshold=0.55,
        sufficient=False,
        failure_reason=QueryFallbackReason.RUNTIME_RELEVANCE_INSUFFICIENT,
        content_hash="sufficiency-hash",
        metadata={"sufficiency_policy_version": "query-runtime-evidence-v2"},
    )
    assert assessment.structural_requirement_success
    assert not assessment.answer_support
    assert not assessment.sufficient

    with pytest.raises(ValidationError, match="server-derived"):
        QueryEvidenceSufficiencyAssessment.model_validate(
            {**assessment.model_dump(mode="python"), "sufficient": True, "failure_reason": None}
        )
