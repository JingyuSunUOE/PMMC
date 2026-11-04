"""Pure, paired Step 8 analysis. No runtime, database, or approval side effects.

Adapters must supply the *entire* frozen selection and distinguish legal abstention
from engineering failure. Cluster IDs are conversation IDs for MEMLENS and scenario
IDs for MemGallery. Resampling uncertainty is not model-seed uncertainty.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal


@dataclass(frozen=True, order=True)
class MethodVariant:
    method: str
    variant: str

    def __post_init__(self) -> None:
        if not self.method.strip() or not self.variant.strip():
            raise ValueError("method and source variant are required")


AnswerKind = Literal[
    "text", "number", "boolean", "date", "image", "image_set", "ordered_image_list"
]


@dataclass(frozen=True)
class TypedAnswer:
    """Image IDs are evaluator-resolved canonical IDs, never evidence citations."""

    kind: AnswerKind
    text: str | None = None
    number: Decimal | None = None
    boolean: bool | None = None
    date_value: date | None = None
    image_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fields = {
            "text": self.text,
            "number": self.number,
            "boolean": self.boolean,
            "date": self.date_value,
        }
        if self.kind in fields:
            if fields[self.kind] is None or self.image_ids:
                raise ValueError("typed answer payload does not match kind")
            if any(value is not None for kind, value in fields.items() if kind != self.kind):
                raise ValueError("mixed typed answer payloads are forbidden")
        elif self.kind in {"image", "image_set", "ordered_image_list"}:
            if any(value is not None for value in fields.values()):
                raise ValueError("image answers cannot contain text/number payloads")
            if not isinstance(self.image_ids, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in self.image_ids
            ):
                raise ValueError("image IDs must be a tuple of nonempty canonical IDs")
            if len(set(self.image_ids)) != len(self.image_ids):
                raise ValueError("duplicate image IDs are invalid")
            if self.kind == "image" and len(self.image_ids) != 1:
                raise ValueError("single image answer requires exactly one image ID")
        else:
            raise ValueError("unsupported answer kind")
        if self.kind == "text" and not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.kind == "number" and (
            not isinstance(self.number, Decimal) or not self.number.is_finite()
        ):
            raise ValueError("number must be a finite Decimal")
        if self.kind == "boolean" and type(self.boolean) is not bool:
            raise ValueError("boolean must be a bool")
        if self.kind == "date" and type(self.date_value) is not date:
            raise ValueError("date_value must be a date")


@dataclass(frozen=True)
class TypedScore:
    value: float | None
    applicable: bool = True
    metric: str = "typed_exact_match"
    supplementary: tuple[tuple[str, float], ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.metric or self.applicable != (self.value is not None):
            raise ValueError("applicable scores require a value; non-applicable scores forbid it")
        for value in (self.value, *(value for _, value in self.supplementary)):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("scores must be finite and in [0, 1]")
        if not self.applicable and not self.unavailable_reason:
            raise ValueError("non-applicable scores require an explicit reason")


def _text(value: str) -> str:
    # Do not strip punctuation/articles: that corrupts signs, quantities and dates.
    return re.sub(r"\s+", " ", value.strip()).casefold()


def score_typed_answer(
    candidate: TypedAnswer,
    reference: TypedAnswer | None,
    *,
    numeric_absolute_tolerance: Decimal = Decimal(0),
) -> TypedScore:
    """Deterministic typed EM with text/set F1 diagnostics; no semantic Judge calls."""
    if (
        not isinstance(numeric_absolute_tolerance, Decimal)
        or not numeric_absolute_tolerance.is_finite()
        or numeric_absolute_tolerance < 0
    ):
        raise ValueError("numeric tolerance must be a finite nonnegative Decimal")
    if reference is None or (reference.kind == "text" and not _text(reference.text or "")):
        return TypedScore(None, False, unavailable_reason="gold_missing")
    if candidate.kind != reference.kind:
        return TypedScore(0.0)
    if reference.kind == "text":
        left, right = _text(candidate.text or ""), _text(reference.text or "")
        a, b = left.split(), right.split()
        overlap = sum((Counter(a) & Counter(b)).values())
        f1 = 2 * overlap / (len(a) + len(b)) if a or b else 0.0
        return TypedScore(float(left == right), supplementary=(("text_token_f1", f1),))
    if reference.kind == "number":
        assert candidate.number is not None and reference.number is not None
        return TypedScore(
            float(abs(candidate.number - reference.number) <= numeric_absolute_tolerance)
        )
    if reference.kind == "date":
        return TypedScore(float(candidate.date_value == reference.date_value))
    if reference.kind == "boolean":
        return TypedScore(float(candidate.boolean == reference.boolean))
    if reference.kind == "image_set":
        a_set, b_set = set(candidate.image_ids), set(reference.image_ids)
        f1 = 2 * len(a_set & b_set) / (len(a_set) + len(b_set)) if a_set or b_set else 1.0
        return TypedScore(float(a_set == b_set), supplementary=(("image_set_f1", f1),))
    return TypedScore(float(candidate.image_ids == reference.image_ids))


class TerminalStatus(StrEnum):
    ANSWERED = "answered"
    ABSTENTION = "abstention"
    ENGINEERING_MISSING = "engineering_missing"


@dataclass(frozen=True)
class SelectedQuery:
    query_id: str
    dataset: str
    cluster_id: str
    cluster_kind: Literal["conversation", "scenario"]
    applicable: bool = True
    strata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.query_id or not self.dataset or not self.cluster_id:
            raise ValueError("query, dataset and cluster identity are required")
        expected = {"memlens": "conversation", "memgallery": "scenario"}.get(
            self.dataset.casefold()
        )
        if self.cluster_kind not in {"conversation", "scenario"} or (
            expected is not None and self.cluster_kind != expected
        ):
            raise ValueError("cluster kind must match the dataset's dependence structure")
        if len(dict(self.strata)) != len(self.strata) or any(
            not k or not v for k, v in self.strata
        ):
            raise ValueError("strata must have unique, nonempty names and values")

    @property
    def cluster(self) -> tuple[str, str]:
        return self.dataset, self.cluster_id


@dataclass(frozen=True)
class ScoreObservation:
    query_id: str
    method: MethodVariant
    status: TerminalStatus
    score: TypedScore | None = None
    judge_value: float | None = None
    judge_method_blind: bool | None = None

    def __post_init__(self) -> None:
        if self.status not in set(TerminalStatus):
            raise ValueError("unknown terminal status")
        if self.status == TerminalStatus.ANSWERED and self.score is None:
            raise ValueError("answered observations require a typed score")
        if self.status != TerminalStatus.ANSWERED and self.score is not None:
            raise ValueError("abstention/missing observations cannot carry answer scores")
        if self.judge_value is not None:
            if not math.isfinite(self.judge_value) or not 0 <= self.judge_value <= 1:
                raise ValueError("Judge scores must be finite and in [0, 1]")
            if self.judge_method_blind is not True or self.status != TerminalStatus.ANSWERED:
                raise ValueError("Judge observations must be method-blind answered facts")


@dataclass(frozen=True)
class StatisticsConfig:
    seed: int = 0
    bootstrap_resamples: int = 10_000
    permutation_resamples: int = 10_000
    confidence_level: float = 0.95
    metric: str = "typed_exact_match"

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("a declared integer seed is required")
        if any(
            type(n) is not int or n < 1
            for n in (self.bootstrap_resamples, self.permutation_resamples)
        ):
            raise ValueError("resample counts must be positive integers")
        if not 0 < self.confidence_level < 1 or not self.metric:
            raise ValueError("invalid confidence level or metric")


@dataclass(frozen=True)
class MethodSummary:
    method: MethodVariant
    status: str
    selected_count: int
    applicable_count: int
    answered_count: int
    abstention_count: int
    engineering_missing_count: int
    missing_query_ids: tuple[str, ...]
    cluster_count: int
    micro: float | None
    cluster_macro: float | None
    conditional_answered_micro: float | None


def _index(
    selected: tuple[SelectedQuery, ...],
    observations: tuple[ScoreObservation, ...],
    config: StatisticsConfig,
) -> dict[tuple[str, MethodVariant], ScoreObservation]:
    selection = {q.query_id: q for q in selected}
    if not selection or len(selection) != len(selected):
        raise ValueError("selection must be nonempty with unique query IDs")
    index = {}
    for row in observations:
        if row.query_id not in selection:
            raise ValueError("observation outside the frozen selection")
        key = row.query_id, row.method
        if key in index:
            raise ValueError("duplicate query/method/variant observation")
        if row.score is not None and (
            row.score.metric != config.metric
            or row.score.applicable != selection[row.query_id].applicable
        ):
            raise ValueError("metric/applicability differs from the frozen selection")
        index[key] = row
    return index


def _value(query: SelectedQuery, row: ScoreObservation) -> float | None:
    if not query.applicable or row.status == TerminalStatus.ENGINEERING_MISSING:
        return None
    if row.status == TerminalStatus.ABSTENTION:
        return 0.0
    assert row.score is not None
    return row.score.value


def summarize_method(
    selected: tuple[SelectedQuery, ...],
    observations: tuple[ScoreObservation, ...],
    method: MethodVariant,
    config: StatisticsConfig = StatisticsConfig(),
) -> MethodSummary:
    index = _index(selected, observations, config)
    missing, values, answered_values = [], [], []
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    answered = abstained = 0
    for query in sorted(selected, key=lambda q: q.query_id):
        row = index.get((query.query_id, method))
        if row is None or row.status == TerminalStatus.ENGINEERING_MISSING:
            missing.append(query.query_id)
            continue
        answered += row.status == TerminalStatus.ANSWERED
        abstained += row.status == TerminalStatus.ABSTENTION
        value = _value(query, row)
        if value is not None:
            values.append(value)
            groups[query.cluster].append(value)
            if row.status == TerminalStatus.ANSWERED:
                answered_values.append(value)
    complete = not missing
    return MethodSummary(
        method,
        "incomplete" if missing else "complete" if values else "not_applicable",
        len(selected),
        sum(q.applicable for q in selected),
        answered,
        abstained,
        len(missing),
        tuple(missing),
        len({q.cluster for q in selected if q.applicable}),
        math.fsum(values) / len(values) if complete and values else None,
        math.fsum(math.fsum(v) / len(v) for v in groups.values()) / len(groups)
        if complete and groups
        else None,
        math.fsum(answered_values) / len(answered_values) if complete and answered_values else None,
    )


Estimand = Literal["micro", "cluster_macro"]


@dataclass(frozen=True)
class PlannedComparison:
    comparison_id: str
    left: MethodVariant
    right: MethodVariant
    estimand: Estimand = "micro"
    exploratory: bool = False

    def __post_init__(self) -> None:
        if not self.comparison_id or self.left == self.right:
            raise ValueError("comparison requires an ID and two distinct method/variants")
        if self.estimand not in {"micro", "cluster_macro"}:
            raise ValueError("unknown estimand")


@dataclass(frozen=True)
class EffectEstimate:
    estimand: Estimand
    difference: float
    confidence_interval: tuple[float, float] | None
    leave_one_cluster_out: tuple[tuple[tuple[str, str], float], ...]


@dataclass(frozen=True)
class ComparisonResult:
    comparison: PlannedComparison
    status: str
    selected_count: int
    applicable_count: int
    cluster_count: int
    micro: EffectEstimate | None
    cluster_macro: EffectEstimate | None
    p_value: float | None
    holm_p_value: float | None
    permutation_mode: str | None
    permutation_draws: int
    warnings: tuple[str, ...]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired_cluster_comparison(
    selected: tuple[SelectedQuery, ...],
    observations: tuple[ScoreObservation, ...],
    comparison: PlannedComparison,
    config: StatisticsConfig = StatisticsConfig(),
) -> ComparisonResult:
    """Left minus right; bootstrap whole paired clusters, permute cluster signs.

    The two-sided permutation null is cluster-level exchangeability of the two
    methods, not independence of queries. One inferential estimand is predeclared
    per comparison; both estimands still receive descriptive bootstrap intervals.
    """
    index = _index(selected, observations, config)
    summaries = [
        summarize_method(selected, observations, method, config)
        for method in (comparison.left, comparison.right)
    ]
    n = sum(q.applicable for q in selected)
    clusters = len({q.cluster for q in selected if q.applicable})
    base = ComparisonResult(
        comparison=comparison,
        status="incomplete",
        selected_count=len(selected),
        applicable_count=n,
        cluster_count=clusters,
        micro=None,
        cluster_macro=None,
        p_value=None,
        holm_p_value=None,
        permutation_mode=None,
        permutation_draws=0,
        warnings=(),
    )
    incomplete = any(summary.status == "incomplete" for summary in summaries)
    if incomplete or not n:
        return replace(
            base,
            status="incomplete" if incomplete else "not_applicable",
            micro=None,
            cluster_macro=None,
            p_value=None,
            permutation_mode=None,
            permutation_draws=0,
            warnings=(),
        )
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for query in sorted(selected, key=lambda q: q.query_id):
        if query.applicable:
            a = _value(query, index[query.query_id, comparison.left])
            b = _value(query, index[query.query_id, comparison.right])
            assert a is not None and b is not None
            groups[query.cluster].append(a - b)
    keys = sorted(groups)
    totals = [math.fsum(groups[k]) for k in keys]
    sizes = [len(groups[k]) for k in keys]
    means = [total / size for total, size in zip(totals, sizes, strict=True)]
    micro, macro = math.fsum(totals) / n, math.fsum(means) / clusters
    if clusters < 2:
        return replace(
            base,
            status="insufficient_clusters",
            micro=EffectEstimate("micro", micro, None, ()),
            cluster_macro=EffectEstimate("cluster_macro", macro, None, ()),
            p_value=None,
            permutation_mode=None,
            permutation_draws=0,
            warnings=("fewer_than_two_clusters",),
        )
    rng = random.Random(config.seed)
    micro_samples, macro_samples = [], []
    for _ in range(config.bootstrap_resamples):
        draw = [rng.randrange(clusters) for _ in keys]
        micro_samples.append(math.fsum(totals[i] for i in draw) / sum(sizes[i] for i in draw))
        macro_samples.append(math.fsum(means[i] for i in draw) / clusters)
    alpha = (1 - config.confidence_level) / 2
    micro_ci = (_quantile(micro_samples, alpha), _quantile(micro_samples, 1 - alpha))
    macro_ci = (_quantile(macro_samples, alpha), _quantile(macro_samples, 1 - alpha))
    micro_loo = tuple(
        (key, math.fsum(totals[:i] + totals[i + 1 :]) / (n - sizes[i]))
        for i, key in enumerate(keys)
    )
    macro_loo = tuple(
        (key, math.fsum(means[:i] + means[i + 1 :]) / (clusters - 1)) for i, key in enumerate(keys)
    )
    weights = totals if comparison.estimand == "micro" else means
    observed = abs(math.fsum(weights))
    exact = 2**clusters <= config.permutation_resamples
    rng = random.Random(config.seed)
    signs = (
        itertools.product((-1, 1), repeat=clusters)
        if exact
        else (tuple(rng.choice((-1, 1)) for _ in keys) for _ in range(config.permutation_resamples))
    )
    extreme = draws = 0
    for sign in signs:
        statistic = abs(math.fsum(s * w for s, w in zip(sign, weights, strict=True)))
        extreme += statistic >= observed - 1e-12
        draws += 1
    p_value = extreme / draws if exact else (extreme + 1) / (draws + 1)
    return replace(
        base,
        status="complete",
        micro=EffectEstimate("micro", micro, micro_ci, micro_loo),
        cluster_macro=EffectEstimate("cluster_macro", macro, macro_ci, macro_loo),
        p_value=p_value,
        permutation_mode="exact" if exact else "monte_carlo_plus_one",
        permutation_draws=draws,
        warnings=(
            "cluster_resampling_is_not_model_seed_replication",
            "limited_clusters_require_cautious_interpretation",
        ),
    )


def holm_adjust(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Holm step-down adjusted p-values in input order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("invalid p-value")
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return tuple(adjusted)


def compare_family(
    selected: tuple[SelectedQuery, ...],
    observations: tuple[ScoreObservation, ...],
    comparisons: tuple[PlannedComparison, ...],
    config: StatisticsConfig = StatisticsConfig(),
) -> tuple[ComparisonResult, ...]:
    """A missing primary test blocks Holm for the whole predeclared family."""
    if len({c.comparison_id for c in comparisons}) != len(comparisons):
        raise ValueError("duplicate comparison ID")
    results = [paired_cluster_comparison(selected, observations, c, config) for c in comparisons]
    primary = [i for i, result in enumerate(results) if not result.comparison.exploratory]
    if all(results[i].p_value is not None for i in primary):
        values = tuple(value for i in primary if (value := results[i].p_value) is not None)
        for index, adjusted in zip(primary, holm_adjust(values), strict=True):
            results[index] = replace(results[index], holm_p_value=adjusted)
    else:
        for index in primary:
            results[index] = replace(
                results[index], warnings=(*results[index].warnings, "primary_family_incomplete")
            )
    return tuple(results)
