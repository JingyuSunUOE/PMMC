"""Pure Step 8 report assembly and null-preserving physical/logical accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pmmc.evaluation.paper_statistics import (
    ComparisonResult,
    MethodSummary,
    MethodVariant,
    PlannedComparison,
    ScoreObservation,
    SelectedQuery,
    StatisticsConfig,
    compare_family,
    summarize_method,
)


@dataclass(frozen=True)
class PairingContract:
    snapshot_id: str
    partition_id: str
    cutoff: str
    selected_query_ids: tuple[str, ...]
    answer_model_alias: str
    answer_model_commit: str
    answer_prompt_hash: str
    answer_contract: str
    judge_version: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.snapshot_id,
                self.partition_id,
                self.cutoff,
                self.answer_model_alias,
                self.answer_model_commit,
                self.answer_prompt_hash,
                self.answer_contract,
                self.judge_version,
            )
        ):
            raise ValueError("all paired evaluation identities must be explicit")
        if not self.selected_query_ids or len(set(self.selected_query_ids)) != len(
            self.selected_query_ids
        ):
            raise ValueError("paired contract needs unique selected query IDs")
        object.__setattr__(self, "selected_query_ids", tuple(sorted(self.selected_query_ids)))


@dataclass(frozen=True)
class MethodContract:
    method: MethodVariant
    pairing: PairingContract
    context_budget_id: str
    truncated_query_count: int
    context_image_count: int

    def __post_init__(self) -> None:
        if (
            not self.context_budget_id
            or min(self.truncated_query_count, self.context_image_count) < 0
        ):
            raise ValueError("frozen context budget and nonnegative context counters are required")


@dataclass(frozen=True)
class StratumSpec:
    name: str
    dataset: str


@dataclass(frozen=True)
class StratumSummary:
    name: str
    value: str
    dataset: str
    methods: tuple[MethodSummary, ...]


@dataclass(frozen=True)
class JudgeDisagreement:
    query_id: str
    method: MethodVariant
    typed_value: float
    judge_value: float


@dataclass(frozen=True)
class CostFact:
    """One physical call, including failed/retried and research-only calls."""

    physical_call_id: str
    cost_usd: Decimal | None = None
    rate_card_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    gpu_active_seconds: float | None = None
    gpu_reserved_seconds: float | None = None
    cpu_seconds: float | None = None
    storage_byte_seconds: float | None = None
    network_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.physical_call_id:
            raise ValueError("physical call identity is required")
        if self.cost_usd is not None and (
            not isinstance(self.cost_usd, Decimal)
            or not self.cost_usd.is_finite()
            or self.cost_usd < 0
            or not self.rate_card_id
        ):
            raise ValueError("known USD costs require a finite nonnegative Decimal and rate card")
        for name in _RESOURCE_FIELDS:
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("resource measurements must be finite and nonnegative")


_RESOURCE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "gpu_active_seconds",
    "gpu_reserved_seconds",
    "cpu_seconds",
    "storage_byte_seconds",
    "network_bytes",
)


@dataclass(frozen=True)
class CostUse:
    """Logical role use of a physical fact; Native-MM baseline/fallback stay distinct."""

    use_id: str
    physical_call_id: str
    method: MethodVariant
    role: str
    scope: Literal["active_path", "logical_ablation"]

    def __post_init__(self) -> None:
        if not self.use_id or not self.physical_call_id or not self.role:
            raise ValueError("cost use, physical fact and role identities are required")
        if self.scope not in {"active_path", "logical_ablation"}:
            raise ValueError("unknown cost scope")


@dataclass(frozen=True)
class QuantityTotal:
    name: str
    total: int | float | None
    known_subtotal: int | float
    unknown_count: int


@dataclass(frozen=True)
class CostTotals:
    count: int
    cost_usd: Decimal | None
    known_usd_subtotal: Decimal
    unknown_usd_count: int
    rate_cards: tuple[str, ...]
    resources: tuple[QuantityTotal, ...]


@dataclass(frozen=True)
class MethodCost:
    method: MethodVariant
    scope: str
    totals: CostTotals


@dataclass(frozen=True)
class CostReport:
    actual_research: CostTotals
    method_costs: tuple[MethodCost, ...]
    uses: tuple[CostUse, ...]


def _cost_total(facts: tuple[CostFact, ...]) -> CostTotals:
    unknown = sum(f.cost_usd is None for f in facts)
    subtotal = sum((f.cost_usd for f in facts if f.cost_usd is not None), Decimal(0))
    resources = []
    for name in _RESOURCE_FIELDS:
        values = [getattr(f, name) for f in facts]
        missing = sum(value is None for value in values)
        known = sum(value for value in values if value is not None)
        resources.append(QuantityTotal(name, None if missing else known, known, missing))
    return CostTotals(
        len(facts),
        None if unknown else subtotal,
        subtotal,
        unknown,
        tuple(sorted({f.rate_card_id for f in facts if f.rate_card_id})),
        tuple(resources),
    )


def summarize_costs(facts: tuple[CostFact, ...], uses: tuple[CostUse, ...]) -> CostReport:
    """Actual calls deduplicate by identity; each declared role use pays logical cost.

    Caller supplies measured facts, not budget-proportional token allocations.
    Provider time is not interpreted as end-to-end online latency.
    """
    index: dict[str, CostFact] = {}
    for fact in facts:
        if fact.physical_call_id in index and index[fact.physical_call_id] != fact:
            raise ValueError("conflicting measurements for one physical call")
        index[fact.physical_call_id] = fact
    if len({u.use_id for u in uses}) != len(uses):
        raise ValueError("duplicate cost use identity")
    if any(use.physical_call_id not in index for use in uses):
        raise ValueError("cost use references a missing physical fact")
    summaries = []
    for method, scope in sorted({(use.method, use.scope) for use in uses}):
        matched = tuple(
            index[u.physical_call_id]
            for u in sorted(uses, key=lambda u: u.use_id)
            if u.method == method and u.scope == scope
        )
        if scope == "active_path":
            matched = tuple({f.physical_call_id: f for f in matched}.values())
        summaries.append(MethodCost(method, scope, _cost_total(matched)))
    return CostReport(
        _cost_total(tuple(index[k] for k in sorted(index))),
        tuple(summaries),
        tuple(sorted(uses, key=lambda u: u.use_id)),
    )


@dataclass(frozen=True)
class BreakEven:
    queries: Decimal | None
    reason: str | None


def break_even_queries(
    write_cost: Decimal | None,
    ours_query_cost: Decimal | None,
    baseline_query_cost: Decimal | None,
) -> BreakEven:
    """C_write/n + C_query under one common memory lifecycle and cost unit."""
    costs = (write_cost, ours_query_cost, baseline_query_cost)
    if any(value is None for value in costs):
        return BreakEven(None, "unknown_cost")
    if any(not isinstance(v, Decimal) or not v.is_finite() or v < 0 for v in costs):
        raise ValueError("costs must be finite nonnegative Decimals in the same unit")
    assert (
        write_cost is not None and ours_query_cost is not None and baseline_query_cost is not None
    )
    if baseline_query_cost <= ours_query_cost:
        return BreakEven(None, "no_positive_query_saving")
    return BreakEven(write_cost / (baseline_query_cost - ours_query_cost), None)


@dataclass(frozen=True)
class PaperReport:
    status: str
    config: StatisticsConfig
    contracts: tuple[MethodContract, ...]
    methods: tuple[MethodSummary, ...]
    strata: tuple[StratumSummary, ...]
    comparisons: tuple[ComparisonResult, ...]
    judge_disagreements: tuple[JudgeDisagreement, ...]
    costs: CostReport | None
    warnings: tuple[str, ...]


def build_paper_report(
    *,
    selected: tuple[SelectedQuery, ...],
    observations: tuple[ScoreObservation, ...],
    contracts: tuple[MethodContract, ...],
    comparisons: tuple[PlannedComparison, ...] = (),
    strata: tuple[StratumSpec, ...] = (),
    config: StatisticsConfig = StatisticsConfig(),
    costs: CostReport | None = None,
) -> PaperReport:
    """Assemble descriptive/inferential results; never approve scientific quality."""
    methods = tuple(sorted(c.method for c in contracts))
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("one frozen contract per method/variant is required")
    if any(c.pairing != contracts[0].pairing for c in contracts):
        raise ValueError("methods have incompatible paired evaluation contracts")
    if contracts[0].pairing.selected_query_ids != tuple(sorted(q.query_id for q in selected)):
        raise ValueError("selected denominator differs from frozen paired contract")
    if any(row.method not in methods for row in observations) or any(
        c.left not in methods or c.right not in methods for c in comparisons
    ):
        raise ValueError("unregistered method/variant in analysis inputs")
    summaries = tuple(
        summarize_method(selected, observations, method, config) for method in methods
    )
    subsets: list[tuple[str, str, str, tuple[SelectedQuery, ...]]] = []
    for dataset in sorted({q.dataset for q in selected}):
        subsets.append(
            ("dataset", dataset, dataset, tuple(q for q in selected if q.dataset == dataset))
        )
    if len(set(strata)) != len(strata):
        raise ValueError("duplicate predeclared stratum")
    for spec in strata:
        members = tuple(q for q in selected if q.dataset == spec.dataset)
        if not members or any(spec.name not in dict(q.strata) for q in members):
            raise ValueError("predeclared stratum is missing for selected queries")
        for value in sorted({dict(q.strata)[spec.name] for q in members}):
            subset = tuple(q for q in members if dict(q.strata)[spec.name] == value)
            subsets.append((spec.name, value, spec.dataset, subset))
    stratum_summaries = []
    for name, value, dataset, subset in subsets:
        ids = {q.query_id for q in subset}
        rows = tuple(row for row in observations if row.query_id in ids)
        stratum_summaries.append(
            StratumSummary(
                name,
                value,
                dataset,
                tuple(summarize_method(subset, rows, m, config) for m in methods),
            )
        )
    disagreements = tuple(
        JudgeDisagreement(row.query_id, row.method, row.score.value, row.judge_value)
        for row in sorted(observations, key=lambda r: (r.query_id, r.method))
        if row.score is not None
        and row.score.value is not None
        and row.judge_value is not None
        and row.score.value != row.judge_value
    )
    return PaperReport(
        "incomplete" if any(s.status == "incomplete" for s in summaries) else "complete",
        config,
        tuple(sorted(contracts, key=lambda c: c.method)),
        summaries,
        tuple(stratum_summaries),
        compare_family(selected, observations, comparisons, config),
        disagreements,
        costs,
        (
            "not_scientific_threshold_approval",
            "semantic_judge_requires_fixed_human_validation",
            "pilot_results_do_not_support_strong_statistical_claims",
        ),
    )
