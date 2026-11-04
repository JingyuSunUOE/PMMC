"""Deterministic lexical/text/multimodal QBank rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from pmmc.compiler.ids import stable_compiler_id
from pmmc.schemas import QBankCoverageCandidate, RankingScore


@dataclass(frozen=True)
class QBankMatchInput:
    qbank_id: str
    generation_rank: int
    lexical_rank: int | None = None
    text_semantic_rank: int | None = None
    multimodal_semantic_rank: int | None = None
    lexical_raw_score: float | None = None
    text_semantic_raw_score: float | None = None
    multimodal_semantic_raw_score: float | None = None


def _rrf(rank: int | None, constant: int) -> float:
    return 0.0 if rank is None else 1.0 / (constant + rank)


def rank_qbank_candidates(
    *,
    coverage_run_id: str,
    query_id: str,
    candidates: list[QBankMatchInput],
    question_budget: int,
    has_query_image: bool,
    rrf_constant: int = 60,
    top_k: int = 50,
) -> list[QBankCoverageCandidate]:
    if question_budget < 1 or rrf_constant < 1 or top_k < 1:
        raise ValueError("QBank matcher limits must be positive")
    eligible = [item for item in candidates if item.generation_rank <= question_budget]
    weights = (
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        if has_query_image
        else (0.5, 0.5, 0.0)
    )
    scored = [
        (
            weights[0] * _rrf(item.lexical_rank, rrf_constant)
            + weights[1] * _rrf(item.text_semantic_rank, rrf_constant)
            + weights[2] * _rrf(item.multimodal_semantic_rank, rrf_constant),
            item,
        )
        for item in eligible
    ]
    scored.sort(key=lambda item: (-item[0], item[1].qbank_id))
    result: list[QBankCoverageCandidate] = []
    for rank, (fused, item) in enumerate(scored[:top_k], start=1):
        result.append(
            QBankCoverageCandidate(
                coverage_candidate_id=stable_compiler_id(
                    "qbank-coverage-candidate",
                    [coverage_run_id, query_id, question_budget, rank, item.qbank_id],
                ),
                coverage_run_id=coverage_run_id,
                query_id=query_id,
                qbank_id=item.qbank_id,
                question_budget=question_budget,
                rank=rank,
                lexical_rank=item.lexical_rank,
                text_semantic_rank=item.text_semantic_rank,
                multimodal_semantic_rank=item.multimodal_semantic_rank,
                lexical_score=(
                    RankingScore(score_type="postgres_fts", raw_value=item.lexical_raw_score)
                    if item.lexical_raw_score is not None
                    else None
                ),
                text_semantic_score=(
                    RankingScore(score_type="cosine", raw_value=item.text_semantic_raw_score)
                    if item.text_semantic_raw_score is not None
                    else None
                ),
                multimodal_semantic_score=(
                    RankingScore(score_type="cosine", raw_value=item.multimodal_semantic_raw_score)
                    if item.multimodal_semantic_raw_score is not None
                    else None
                ),
                final_score=RankingScore(
                    score_type="weighted_rrf",
                    raw_value=fused,
                    metadata={
                        "rrf_constant": rrf_constant,
                        "lexical_weight": weights[0],
                        "text_weight": weights[1],
                        "multimodal_weight": weights[2],
                    },
                ),
            )
        )
    return result
