"""Deterministic answer metrics used by every baseline method."""

from __future__ import annotations

import hashlib
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from pmmc.benchmarks.ids import stable_hash
from pmmc.schemas import AnswerNormalization, MetricFamily, MetricScore

ANSWER_METRIC_VERSIONS = {
    "exact_match_trimmed": "exact_match_trimmed_v1",
    "normalized_exact_match": "normalized_exact_match_squad_v1",
    "token_f1": "token_f1_squad_v1",
    "bleu1": "bleu1_sacrebleu_v1",
}

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
def trim_answer(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).strip())


def normalize_answer(value: str) -> str:
    normalized = trim_answer(value).casefold()
    normalized = normalized.translate(_PUNCT_TRANSLATION)
    normalized = _ARTICLES.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def _token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()
    if not candidate_tokens or not reference_tokens:
        return float(candidate_tokens == reference_tokens)
    overlap = Counter(candidate_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _bleu1(candidate: str, reference: str) -> float:
    if not candidate and not reference:
        return 1.0
    if not candidate or not reference:
        return 0.0
    try:
        from sacrebleu.metrics import BLEU
    except ModuleNotFoundError:
        # BLEU-1 is clipped unigram precision multiplied by the brevity penalty.
        # The full server install uses SacreBLEU; this exact one-gram fallback keeps
        # schema-only development environments importable.
        import math

        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        overlap = Counter(candidate_tokens) & Counter(reference_tokens)
        precision = sum(overlap.values()) / len(candidate_tokens)
        brevity_penalty = (
            1.0
            if len(candidate_tokens) >= len(reference_tokens)
            else math.exp(1 - len(reference_tokens) / len(candidate_tokens))
        )
        return brevity_penalty * precision
    bleu = BLEU(
        lowercase=False,
        tokenize="none",
        smooth_method="exp",
        effective_order=True,
        max_ngram_order=1,
    )
    return max(0.0, min(1.0, bleu.sentence_score(candidate, [reference]).score / 100.0))


@dataclass(frozen=True)
class AnswerMetricBundle:
    normalization: AnswerNormalization
    metrics: tuple[MetricScore, ...]


def score_answer(
    *,
    evaluation_run_id: str,
    answer_id: str,
    query_id: str,
    run_id: str,
    candidate: str,
    reference: str | None,
    normalizer_version: str = "answer-normalizer-v1",
    created_at: datetime | None = None,
) -> AnswerMetricBundle:
    """Return the frozen four metrics without mutating the runtime answer."""

    trimmed_candidate = trim_answer(candidate)
    normalized_candidate = normalize_answer(candidate)
    reference_value = reference or ""
    trimmed_reference = trim_answer(reference_value)
    normalized_reference = normalize_answer(reference_value)
    normalization = AnswerNormalization(
        normalization_id="norm-"
        + stable_hash([evaluation_run_id, answer_id, normalizer_version])[:24],
        evaluation_run_id=evaluation_run_id,
        answer_id=answer_id,
        query_id=query_id,
        run_id=run_id,
        normalizer_version=normalizer_version,
        candidate_hash=hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        reference_hash=hashlib.sha256(reference_value.encode("utf-8")).hexdigest(),
        normalized_candidate=normalized_candidate,
        normalized_reference=normalized_reference,
        **({"created_at": created_at} if created_at is not None else {}),
    )
    values = {
        "exact_match_trimmed": float(trimmed_candidate == trimmed_reference),
        "normalized_exact_match": float(normalized_candidate == normalized_reference),
        "token_f1": _token_f1(normalized_candidate, normalized_reference),
        "bleu1": _bleu1(normalized_candidate, normalized_reference),
    }
    metrics: list[MetricScore] = []
    for name, version in ANSWER_METRIC_VERSIONS.items():
        applicable = reference is not None
        metrics.append(
            MetricScore(
                metric_score_id="metric-"
                + stable_hash([evaluation_run_id, answer_id, name, version])[:24],
                evaluation_run_id=evaluation_run_id,
                answer_id=answer_id,
                query_id=query_id,
                run_id=run_id,
                metric_family=MetricFamily.ANSWER,
                metric_name=name,
                metric_version=version,
                scorer="pmmc.evaluation.metrics",
                applicable=applicable,
                value=values[name] if applicable else None,
                unavailable_reason=None if applicable else "gold_answer_missing",
                **({"created_at": created_at} if created_at is not None else {}),
            )
        )
    return AnswerMetricBundle(normalization=normalization, metrics=tuple(metrics))
