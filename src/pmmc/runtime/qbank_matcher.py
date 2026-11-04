"""Auditable lexical/text/multimodal matching over a verified QBank projection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.runtime.query_executor_config import QueryExecutorConfig
from pmmc.schemas import (
    Modality,
    QBankMatchFusedCandidate,
    QBankRerankerAssessment,
    QBankRouteClass,
    QBankRuntimeChannel,
    QueryFallbackReason,
    QueryRouteDecision,
    QueryRouteOutcome,
    RerankerMatchLevel,
)


@dataclass(frozen=True)
class RuntimeQBankChannelHit:
    """One pre-ranking-safe hit returned after joining the routing projection."""

    qbank_id: str
    plan_id: str
    route_class: QBankRouteClass
    generation_rank: int
    accepted_round: int
    required_modalities: tuple[Modality, ...]
    channel: QBankRuntimeChannel
    raw_score: float
    query_image_required: bool = False
    candidate_question: str = ""
    question_type: str = "unknown"
    projection_round: int | None = None
    source_program_hash: str | None = None


@dataclass(frozen=True)
class MatchGridKey:
    question_budget: int
    round_id: int


@dataclass(frozen=True)
class RuntimeMatchResult:
    key: MatchGridKey
    candidates: tuple[QBankMatchFusedCandidate, ...]


def _weights(
    config: QueryExecutorConfig,
    *,
    has_query_image: bool,
) -> dict[QBankRuntimeChannel, float]:
    source = config.multimodal_query_weights if has_query_image else config.text_query_weights
    return {QBankRuntimeChannel(key): value for key, value in source.items()}


def _deduplicate_hits(
    hits: list[RuntimeQBankChannelHit],
) -> dict[QBankRuntimeChannel, dict[str, RuntimeQBankChannelHit]]:
    result: dict[QBankRuntimeChannel, dict[str, RuntimeQBankChannelHit]] = defaultdict(dict)
    for hit in hits:
        existing = result[hit.channel].get(hit.qbank_id)
        if existing is None or (hit.raw_score, hit.qbank_id) > (
            existing.raw_score,
            existing.qbank_id,
        ):
            result[hit.channel][hit.qbank_id] = hit
    return result


def rank_runtime_match_grid(
    *,
    query_executor_run_id: str,
    query_id: str,
    hits: list[RuntimeQBankChannelHit],
    has_query_image: bool,
    config: QueryExecutorConfig,
) -> dict[MatchGridKey, RuntimeMatchResult]:
    """Re-rank every budget/round from full channel scores, never from b8/r3 top-50."""

    if config.is_v3 and any(hit.projection_round not in config.rounds for hit in hits):
        raise ValueError("v3 matching requires exact round membership on every hit")
    deduplicated = _deduplicate_hits(hits)
    weights = _weights(config, has_query_image=has_query_image)
    max_rrf = sum(weights.values()) / (config.rrf_constant + 1)
    if max_rrf <= 0:
        raise ValueError("runtime matcher has no active channels")
    output: dict[MatchGridKey, RuntimeMatchResult] = {}
    for question_budget in config.question_budgets:
        for round_id in config.rounds:
            if config.is_v3:
                deduplicated = _deduplicate_hits(
                    [hit for hit in hits if hit.projection_round == round_id]
                )
            channel_ranks: dict[tuple[QBankRuntimeChannel, QBankRouteClass], dict[str, int]] = {}
            candidate_by_id: dict[str, RuntimeQBankChannelHit] = {}
            for channel in weights:
                for route_class in QBankRouteClass:
                    eligible = [
                        item
                        for item in deduplicated.get(channel, {}).values()
                        if item.route_class is route_class
                        and item.generation_rank <= question_budget
                        and (config.is_v3 or item.accepted_round <= round_id)
                    ]
                    eligible.sort(key=lambda item: (-item.raw_score, item.qbank_id))
                    channel_ranks[(channel, route_class)] = {
                        item.qbank_id: rank for rank, item in enumerate(eligible, start=1)
                    }
                    for item in eligible:
                        previous = candidate_by_id.get(item.qbank_id)
                        if previous is not None and (
                            previous.plan_id != item.plan_id
                            or previous.route_class is not item.route_class
                            or previous.generation_rank != item.generation_rank
                            or previous.accepted_round != item.accepted_round
                        ):
                            raise ValueError("channel hits disagree on verified routing identity")
                        candidate_by_id[item.qbank_id] = item

            scored: dict[
                QBankRouteClass,
                list[tuple[float, float, str, RuntimeQBankChannelHit, dict[str, float]]],
            ] = defaultdict(list)
            for qbank_id, item in candidate_by_id.items():
                raw_rrf = sum(
                    weight / (config.rrf_constant + rank)
                    for channel, weight in weights.items()
                    if (rank := channel_ranks[(channel, item.route_class)].get(qbank_id))
                    is not None
                )
                normalized: dict[str, float] = {}
                for channel in weights:
                    hit = deduplicated.get(channel, {}).get(qbank_id)
                    if hit is None:
                        continue
                    if channel is QBankRuntimeChannel.LEXICAL:
                        value = max(0.0, hit.raw_score)
                        normalized[channel.value] = value / (1.0 + value)
                    else:
                        normalized[channel.value] = min(1.0, max(0.0, (hit.raw_score + 1.0) / 2.0))
                available_weight = sum(
                    weight for channel, weight in weights.items() if channel.value in normalized
                )
                base_score = (
                    sum(
                        weights[QBankRuntimeChannel(channel)] * value
                        for channel, value in normalized.items()
                    )
                    / available_weight
                    if available_weight
                    else 0.0
                )
                base_reliable = base_score * (0.75 + 0.25 * available_weight)
                primary_score = base_reliable if config.is_v2 else raw_rrf
                scored[item.route_class].append(
                    (primary_score, raw_rrf, qbank_id, item, normalized)
                )
            for route_scores in scored.values():
                route_scores.sort(key=lambda item: (-item[0], -item[1], item[2]))

            key = MatchGridKey(question_budget=question_budget, round_id=round_id)
            candidates: list[QBankMatchFusedCandidate] = []
            for route_class in QBankRouteClass:
                for rank, (_primary, raw_rrf, qbank_id, item, normalized) in enumerate(
                    scored.get(route_class, [])[: config.match_top_k], start=1
                ):
                    available_weight = sum(
                        weight for channel, weight in weights.items() if channel.value in normalized
                    )
                    base_score = (
                        sum(
                            weights[QBankRuntimeChannel(channel)] * value
                            for channel, value in normalized.items()
                        )
                        / available_weight
                        if available_weight
                        else 0.0
                    )
                    base_reliable_score = base_score * (0.75 + 0.25 * available_weight)
                    if config.is_v3:
                        # Missing channels contribute zero; declared weights never change.
                        base_score = min(1.0, max(0.0, 61.0 * raw_rrf))
                        base_reliable_score = base_score
                    modality_compatible = not item.query_image_required or has_query_image
                    payload = {
                        "query_executor_run_id": query_executor_run_id,
                        "query_id": query_id,
                        "question_budget": question_budget,
                        "round_id": round_id,
                        "route_class": route_class.value,
                        "rank": rank,
                        "qbank_id": qbank_id,
                        "raw_rrf": raw_rrf,
                    }
                    candidates.append(
                        QBankMatchFusedCandidate(
                            fused_candidate_id=stable_compiler_id(
                                "runtime-qbank-fused-candidate", payload
                            ),
                            match_batch_id=stable_compiler_id(
                                "runtime-qbank-match-batch",
                                [query_executor_run_id, query_id, question_budget, round_id],
                            ),
                            query_id=query_id,
                            qbank_id=qbank_id,
                            plan_id=item.plan_id,
                            route_class=item.route_class,
                            rank=rank,
                            lexical_rank=channel_ranks.get(
                                (QBankRuntimeChannel.LEXICAL, item.route_class), {}
                            ).get(qbank_id),
                            text_rank=channel_ranks.get(
                                (QBankRuntimeChannel.TEXT, item.route_class), {}
                            ).get(qbank_id),
                            multimodal_rank=channel_ranks.get(
                                (QBankRuntimeChannel.MULTIMODAL, item.route_class), {}
                            ).get(qbank_id),
                            raw_rrf=raw_rrf,
                            confidence=(
                                base_reliable_score
                                if config.uses_reranker
                                else min(1.0, max(0.0, raw_rrf / max_rrf))
                            ),
                            modality_compatible=modality_compatible,
                            lexical_normalized_score=normalized.get("lexical"),
                            text_normalized_score=normalized.get("text"),
                            multimodal_normalized_score=normalized.get("multimodal"),
                            available_weight_sum=available_weight,
                            base_score=base_score,
                            base_reliable_score=base_reliable_score,
                            content_hash=canonical_content_hash(payload),
                            metadata={
                                **(
                                    {
                                        "projection_round": round_id,
                                        "question_budget": question_budget,
                                        "source_program_hash": item.source_program_hash,
                                        "base_score_transform": "normalized-weighted-rrf-v1",
                                    }
                                    if config.is_v3
                                    else {}
                                ),
                                "generation_rank": item.generation_rank,
                                "accepted_round": item.accepted_round,
                                "route_class_ranked_separately": True,
                                "rrf_constant": config.rrf_constant,
                                "active_weights": {
                                    channel.value: weight for channel, weight in weights.items()
                                },
                                "candidate_question": item.candidate_question,
                                "question_type": item.question_type,
                                "required_modalities": [
                                    value.value for value in item.required_modalities
                                ],
                            },
                        )
                    )
            output[key] = RuntimeMatchResult(key=key, candidates=tuple(candidates))
    return output


def reranker_pair_ids(
    grid: dict[MatchGridKey, RuntimeMatchResult],
    *,
    top_k: int = 20,
    separate_route_classes: bool = False,
) -> tuple[str, ...]:
    """Return the deterministic union of each independently ranked base top-k."""

    values: set[str] = set()
    for result in grid.values():
        if separate_route_classes:
            for route_class in QBankRouteClass:
                values.update(
                    item.qbank_id
                    for item in result.candidates
                    if item.route_class is route_class and item.rank <= top_k
                )
        else:
            values.update(item.qbank_id for item in result.candidates[:top_k])
    return tuple(sorted(values))


def apply_reranker_assessments(
    *,
    grid: dict[MatchGridKey, RuntimeMatchResult],
    assessments: dict[str, QBankRerankerAssessment],
    config: QueryExecutorConfig,
) -> dict[MatchGridKey, RuntimeMatchResult]:
    """Apply one deduplicated pair assessment to every budget/round projection."""

    if not config.reranker_enabled:
        return grid

    def updated_candidate(
        candidate: QBankMatchFusedCandidate, **updates: object
    ) -> QBankMatchFusedCandidate:
        data = candidate.model_dump(mode="python")
        data.update(updates)
        data["content_hash"] = canonical_content_hash(
            {key: value for key, value in data.items() if key not in {"content_hash", "metadata"}}
        )
        return QBankMatchFusedCandidate.model_validate(data)

    output: dict[MatchGridKey, RuntimeMatchResult] = {}
    for key, result in grid.items():
        by_class: dict[QBankRouteClass, list[QBankMatchFusedCandidate]] = defaultdict(list)
        for candidate in result.candidates:
            assessment = assessments.get(candidate.qbank_id)
            if config.is_v3 and candidate.rank > config.reranker_top_k:
                assessment = None
            reranker_score = assessment.reranker_score if assessment is not None else 0.0
            match_level = (
                assessment.effective_match_level
                if assessment is not None
                else RerankerMatchLevel.NONE
            )
            base_score = candidate.base_reliable_score or 0.0
            route_score = (
                config.base_score_weight * base_score
                + config.reranker_score_weight * reranker_score
            )
            by_class[candidate.route_class].append(
                updated_candidate(
                    candidate,
                    confidence=route_score,
                    reranker_score=reranker_score,
                    reranker_match_level=match_level,
                    route_score=route_score,
                    metadata={
                        **candidate.metadata,
                        "reranker_assessment_id": (
                            assessment.reranker_assessment_id if assessment is not None else None
                        ),
                    },
                )
            )
        ranked: list[QBankMatchFusedCandidate] = []
        for route_class in QBankRouteClass:
            values = by_class.get(route_class, [])
            values.sort(
                key=lambda item: (
                    -(item.route_score or 0.0),
                    -(item.reranker_score or 0.0),
                    -item.raw_rrf,
                    item.qbank_id,
                )
            )
            for rank, item in enumerate(values[: config.match_top_k], start=1):
                ranked.append(updated_candidate(item, rank=rank))
        output[key] = RuntimeMatchResult(key=key, candidates=tuple(ranked))
    return output


def decide_runtime_route(
    *,
    query_executor_run_id: str,
    query_id: str,
    match_batch_id: str,
    candidates: tuple[QBankMatchFusedCandidate, ...],
    config: QueryExecutorConfig,
) -> QueryRouteDecision:
    """Apply the frozen confidence, ambiguity, modality, and control gates."""

    answerable = tuple(
        item for item in candidates if item.route_class == QBankRouteClass.ANSWERABLE.value
    )
    controls = tuple(
        item for item in candidates if item.route_class == QBankRouteClass.CONTROL.value
    )
    top = answerable[0] if answerable else None
    gap = _candidate_gap(answerable)
    control_top = controls[0] if controls else None
    control_gap = _candidate_gap(controls)
    outcome = QueryRouteOutcome.PROGRAM
    reason: QueryFallbackReason | None = None
    selected_qbank_id: str | None = top.qbank_id if top is not None else None
    selected_plan_id: str | None = top.plan_id if top is not None else None
    if top is None and _passes_route_gate(control_top, control_gap, config):
        top = control_top
        gap = control_gap
        outcome = QueryRouteOutcome.CONTROL
        reason = QueryFallbackReason.CONTROL_ROUTE
    elif top is None:
        top = control_top
        gap = control_gap
        outcome = QueryRouteOutcome.FALLBACK
        reason = QueryFallbackReason.NO_QBANK_CANDIDATE
    elif not top.modality_compatible:
        outcome = QueryRouteOutcome.FALLBACK
        reason = QueryFallbackReason.REQUIREMENT_INSUFFICIENT
    elif config.reranker_enabled and top.reranker_match_level not in {
        RerankerMatchLevel.EXACT,
        RerankerMatchLevel.STRONG,
    }:
        outcome = QueryRouteOutcome.FALLBACK
        reason = QueryFallbackReason.RERANKER_REJECTED
    elif top.confidence < config.route_match_threshold:
        if _passes_route_gate(control_top, control_gap, config):
            top = control_top
            gap = control_gap
            outcome = QueryRouteOutcome.CONTROL
            reason = QueryFallbackReason.CONTROL_ROUTE
        else:
            outcome = QueryRouteOutcome.FALLBACK
            reason = QueryFallbackReason.LOW_MATCH_CONFIDENCE
    elif gap is not None and gap < config.route_gap_threshold:
        if _passes_route_gate(control_top, control_gap, config):
            top = control_top
            gap = control_gap
            outcome = QueryRouteOutcome.CONTROL
            reason = QueryFallbackReason.CONTROL_ROUTE
        else:
            outcome = QueryRouteOutcome.FALLBACK
            reason = QueryFallbackReason.AMBIGUOUS_MATCH
    elif top.route_class == QBankRouteClass.CONTROL.value:
        # Defensive only: ``top`` is selected from the answerable partition above.
        outcome = QueryRouteOutcome.CONTROL
        reason = QueryFallbackReason.CONTROL_ROUTE
    if outcome is not QueryRouteOutcome.PROGRAM:
        selected_qbank_id = None
        selected_plan_id = None

    counterfactual = answerable[0] if answerable else control_top

    payload = {
        "query_executor_run_id": query_executor_run_id,
        "query_id": query_id,
        "match_batch_id": match_batch_id,
        "outcome": outcome.value,
        "selected_qbank_id": selected_qbank_id,
        "selected_plan_id": selected_plan_id,
        "confidence": top.confidence if top is not None else None,
        "gap": gap,
        "reason": reason.value if reason is not None else None,
    }
    return QueryRouteDecision(
        route_decision_id=stable_compiler_id("runtime-query-route", payload),
        query_executor_run_id=query_executor_run_id,
        match_batch_id=match_batch_id,
        query_id=query_id,
        outcome=outcome,
        selected_qbank_id=selected_qbank_id,
        selected_plan_id=selected_plan_id,
        confidence=top.confidence if top is not None else None,
        top1_top2_gap=gap,
        match_threshold=config.route_match_threshold,
        gap_threshold=config.route_gap_threshold,
        fallback_reason=reason,
        counterfactual_qbank_id=(counterfactual.qbank_id if counterfactual is not None else None),
        content_hash=canonical_content_hash(payload),
        metadata={
            "answerable_candidate_count": len(answerable),
            "control_candidate_count": len(controls),
            "control_confidence": (control_top.confidence if control_top is not None else None),
            "control_gap": control_gap,
        },
    )


def _candidate_gap(
    candidates: tuple[QBankMatchFusedCandidate, ...],
) -> float | None:
    if len(candidates) < 2:
        return None
    return max(0.0, min(1.0, candidates[0].confidence - candidates[1].confidence))


def _passes_route_gate(
    candidate: QBankMatchFusedCandidate | None,
    gap: float | None,
    config: QueryExecutorConfig,
) -> bool:
    return bool(
        candidate is not None
        and candidate.modality_compatible
        and (
            not config.reranker_enabled
            or candidate.reranker_match_level
            in {RerankerMatchLevel.EXACT, RerankerMatchLevel.STRONG}
        )
        and candidate.confidence >= config.route_match_threshold
        and (gap is None or gap >= config.route_gap_threshold)
    )


def counterfactual_program_candidate(
    candidates: tuple[QBankMatchFusedCandidate, ...],
) -> QBankMatchFusedCandidate | None:
    """Prefer an answerable program for analysis; control is only a last resort."""

    return next(
        (item for item in candidates if item.route_class == QBankRouteClass.ANSWERABLE.value),
        candidates[0] if candidates else None,
    )


__all__ = [
    "MatchGridKey",
    "RuntimeMatchResult",
    "RuntimeQBankChannelHit",
    "counterfactual_program_candidate",
    "apply_reranker_assessments",
    "decide_runtime_route",
    "rank_runtime_match_grid",
    "reranker_pair_ids",
]
