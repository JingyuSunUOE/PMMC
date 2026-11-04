"""Versioned deterministic failure attribution for foundational baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pmmc.benchmarks.ids import stable_hash
from pmmc.schemas import (
    BadCaseCategory,
    BadCaseLabel,
    JudgeScore,
    JudgeVerdict,
    MetricScore,
)

BAD_CASE_PRIORITY = (
    BadCaseCategory.MODEL_CALL_FAILURE,
    BadCaseCategory.GOLD_UNRESOLVABLE,
    BadCaseCategory.RETRIEVAL_TOP50_MISS,
    BadCaseCategory.RETRIEVAL_TOP5_MISS,
    BadCaseCategory.CONTEXT_BUDGET_EXCLUSION,
    BadCaseCategory.VISUAL_EVIDENCE_MISS,
    BadCaseCategory.TEXT_EVIDENCE_MISS,
    BadCaseCategory.EVIDENCE_FOUND_ANSWER_WRONG,
    BadCaseCategory.ANSWER_FORMAT_MISMATCH,
    BadCaseCategory.JUDGE_METRIC_DISAGREEMENT,
    BadCaseCategory.NONE,
)


@dataclass(frozen=True)
class BadCaseInputs:
    model_failed: bool = False
    gold_resolvable: bool = True
    context_budget_excluded: bool = False


def _value(metrics: list[MetricScore], name: str) -> float | None:
    match = next((item for item in metrics if item.metric_name == name), None)
    return match.value if match is not None and match.applicable else None


def classify_bad_cases(
    *,
    evaluation_run_id: str,
    answer_id: str,
    query_id: str,
    run_id: str,
    metrics: list[MetricScore],
    judge: JudgeScore | None,
    inputs: BadCaseInputs,
    rule_version: str = "bad-case-rules-v1",
    created_at: datetime | None = None,
) -> list[BadCaseLabel]:
    categories: set[BadCaseCategory] = set()
    if inputs.model_failed:
        categories.add(BadCaseCategory.MODEL_CALL_FAILURE)
    if not inputs.gold_resolvable:
        categories.add(BadCaseCategory.GOLD_UNRESOLVABLE)

    recall50 = _value(metrics, "evidence_recall_at_50")
    recall5 = _value(metrics, "evidence_recall_at_5")
    context_recall = _value(metrics, "context_evidence_recall")
    image_hit = _value(metrics, "context_image_hit")
    text_hit = _value(metrics, "context_text_hit")
    raw_em = _value(metrics, "exact_match_trimmed")
    normalized_em = _value(metrics, "normalized_exact_match")
    token_f1 = _value(metrics, "token_f1")

    if recall50 == 0:
        categories.add(BadCaseCategory.RETRIEVAL_TOP50_MISS)
    elif recall5 == 0:
        categories.add(BadCaseCategory.RETRIEVAL_TOP5_MISS)
    if (
        inputs.context_budget_excluded
        and recall5 is not None
        and context_recall is not None
        and recall5 > context_recall
    ):
        categories.add(BadCaseCategory.CONTEXT_BUDGET_EXCLUSION)
    if image_hit == 0:
        categories.add(BadCaseCategory.VISUAL_EVIDENCE_MISS)
    if text_hit == 0:
        categories.add(BadCaseCategory.TEXT_EVIDENCE_MISS)
    if raw_em == 0 and normalized_em == 1:
        categories.add(BadCaseCategory.ANSWER_FORMAT_MISMATCH)
    if context_recall is not None and context_recall > 0 and (
        (judge is not None and judge.verdict is JudgeVerdict.INCORRECT)
        or (token_f1 is not None and token_f1 < 0.5)
    ):
        categories.add(BadCaseCategory.EVIDENCE_FOUND_ANSWER_WRONG)
    if judge is not None and (
        (judge.verdict is JudgeVerdict.CORRECT and token_f1 is not None and token_f1 < 0.5)
        or (judge.verdict is JudgeVerdict.INCORRECT and normalized_em == 1)
    ):
        categories.add(BadCaseCategory.JUDGE_METRIC_DISAGREEMENT)
    if not categories:
        categories.add(BadCaseCategory.NONE)

    ordered = [item for item in BAD_CASE_PRIORITY if item in categories]
    primary = ordered[0]
    return [
        BadCaseLabel(
            bad_case_label_id="bad-case-"
            + stable_hash(
                [evaluation_run_id, answer_id, category.value, rule_version]
            )[:24],
            evaluation_run_id=evaluation_run_id,
            answer_id=answer_id,
            query_id=query_id,
            run_id=run_id,
            category=category,
            label_source="automatic",
            rule_version=rule_version,
            is_primary=category is primary,
            confidence=1.0,
            **({"created_at": created_at} if created_at is not None else {}),
        )
        for category in ordered
    ]
