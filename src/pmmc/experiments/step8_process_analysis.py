"""Paired descriptive process reports; never call a model or shrink a denominator."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from pmmc.schemas.step8 import ExperimentCell


def summarize_process_metrics(facts) -> dict[str, Any]:
    """Consume only verified evaluator receipts and emit aggregate, identifier-free metrics."""
    from pmmc.schemas.step8_process import PROCESS_METRIC_NAMES, ProcessMetricsReport

    matrix = {cell.cell_id: cell for cell in facts.cells}
    records = {}
    unavailable = {"semantic_cover", "semantic_match", "oracle_answer_accuracy"}
    for job_id in sorted(facts.verified_jobs):
        if facts.jobs[job_id].stage != "evaluation":
            continue
        report = facts.receipts[job_id].output
        if report.get("process_metrics") is None:
            continue
        observations = ProcessMetricsReport.model_validate(report["process_metrics"])
        scores = {row["cell_id"]: row for row in report["scores"]}
        if len(observations.cells) != len(scores) or {c.cell_id for c in observations.cells} != set(
            scores
        ):
            raise ValueError("process metric cell denominator differs from evaluation")
        unavailable.update(observations.unavailable_metrics)
        for row in observations.cells:
            cell = matrix.get(row.cell_id)
            score = scores[row.cell_id]
            if (
                cell is None
                or row.query_id != report["query_id"]
                or row.method != cell.method
                or row.source_fact_id != score.get("source_fact_id")
                or row.source_fact_hash != score.get("source_fact_hash")
                or row.question_budget != (cell.question_budget if cell.method == "pmmc" else None)
                or row.round_id != (cell.round_id if cell.method == "pmmc" else None)
                or set(row.metrics) != set(PROCESS_METRIC_NAMES)
            ):
                raise ValueError("process metric evaluation binding mismatch")
            key = row.cell_id, row.query_id
            if key in records:
                raise ValueError("duplicate process query/cell fact")
            records[key] = row
    summaries = []
    for cell in facts.cells:
        selected = facts.selection.query_ids[cell.dataset]
        metrics = {}
        for name in PROCESS_METRIC_NAMES:
            evaluated = []
            reasons: Counter[str] = Counter()
            counts: Counter[str] = Counter()
            for query in selected:
                observation = records.get((cell.cell_id, query))
                if observation is None:
                    counts["incomplete"] += 1
                    reasons["process_report_missing"] += 1
                    continue
                value = observation.metrics[name]
                counts[value.status] += 1
                if value.status == "evaluated":
                    evaluated.append(value)
                elif value.reason:
                    reasons[value.reason] += 1
            complete = not counts["incomplete"] and not counts["not_evaluated"]
            numerator = sum(v.numerator or 0 for v in evaluated)
            denominator = sum(v.denominator or 0 for v in evaluated)
            kinds = {v.denominator_kind for v in evaluated}
            if len(kinds) > 1:
                raise ValueError("process metric denominator kinds differ across queries")
            metrics[name] = {
                "status": (
                    "incomplete"
                    if counts["incomplete"]
                    else "not_evaluated"
                    if counts["not_evaluated"]
                    else "complete"
                    if evaluated
                    else "not_applicable"
                ),
                "selected_count": len(selected),
                "evaluated_count": len(evaluated),
                "not_applicable_count": counts["not_applicable"],
                "not_evaluated_count": counts["not_evaluated"],
                "engineering_missing_count": counts["incomplete"],
                "not_executed_count": reasons["program_not_executed"],
                "observed_numerator": numerator,
                "observed_denominator": denominator,
                "denominator_kind": next(iter(kinds)) if kinds else None,
                "query_mean": math.fsum(v.value or 0 for v in evaluated) / len(evaluated)
                if complete and evaluated
                else None,
                "pooled_ratio": numerator / denominator if complete and denominator else None,
                "reason_counts": dict(sorted(reasons.items())),
            }
        summaries.append(
            {"cell_id": cell.cell_id, **cell.model_dump(mode="json"), "metrics": metrics}
        )
    blocked = [
        {"cell_id": cell["cell_id"], "metric": name, "status": value["status"]}
        for cell in summaries
        for name, value in cell["metrics"].items()
        if value["status"] in {"incomplete", "not_evaluated"}
    ]
    return {
        "schema_version": "step8-process-analysis-v1",
        "status": "incomplete" if not summaries or blocked else "complete_against_declared_metrics",
        "completion_policy": "all_declared_applicable_metrics_evaluated",
        "required_metrics": list(PROCESS_METRIC_NAMES),
        "blocking_metrics": blocked,
        "cells": summaries,
        "unavailable_metrics": sorted(unavailable),
        "scientific_approval": False,
        "warnings": [
            "candidate_support_proxy_is_not_runtime_evidence_hit",
            "program_rates_condition_on_execution_with_selected_denominator_retained",
            "not_evaluated_metrics_are_not_zero",
            "oracle_evidence_proxy_is_not_oracle_answer_accuracy",
        ],
    }


def require_complete_process_metrics(report: dict[str, Any]) -> None:
    """Approval rechecks the denominator, not just a producer's top-level status.

    Metrics outside the declared process contract (for example Oracle answer
    accuracy) are separately reported as unavailable. An applicable member of
    this contract cannot use that escape hatch; legal non-applicability can.
    """
    from pmmc.schemas.step8_process import PROCESS_METRIC_NAMES

    if report.get("status") != "complete_against_declared_metrics" or not report.get("cells"):
        raise ValueError("approval_process_metric_denominator_incomplete")
    for cell in report["cells"]:
        metrics = cell.get("metrics", {})
        if set(metrics) != set(PROCESS_METRIC_NAMES):
            raise ValueError("approval_process_metric_denominator_incomplete")
        for value in metrics.values():
            if (
                value.get("status") not in {"complete", "not_applicable"}
                or value.get("engineering_missing_count") != 0
                or value.get("not_evaluated_count") != 0
                or value.get("selected_count")
                != value.get("evaluated_count", 0) + value.get("not_applicable_count", 0)
            ):
                raise ValueError("approval_process_metric_denominator_incomplete")


def _value(row: dict | None, metric: str) -> float | None:
    if row is None or row.get("engineering_error"):
        return None
    applicable = row.get("applicable" if metric == "typed_exact_match" else "judge_applicable")
    if applicable is not True:
        return None
    if row.get("status") == "abstention":
        return 0.0
    if row.get("status") != "answered" or not row.get("typed_answer_id"):
        return None
    if metric == "method_blind_judge" and (
        row.get("judge_method_blind") is not True or not row.get("judge_model_call_id")
    ):
        return None
    value = row.get("value" if metric == "typed_exact_match" else "judge_value")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return float(value)


def paired_transitions(
    cells: tuple[ExperimentCell, ...],
    selected: dict[str, tuple[str, ...]],
    rows: dict[tuple[str, str], dict],
) -> list[dict[str, Any]]:
    """Compare saved variants only; forced-top1 is never a no-fallback substitute.

    Round changes are descriptive paired associations, not operator causal effects.
    Missing or non-applicable observations remain explicit, never scored as zero.
    """
    index: dict[tuple[str, str, str, int, int, str], ExperimentCell] = {
        (
            cell.dataset.value,
            cell.answer_model,
            cell.treatment.value,
            cell.question_budget,
            cell.round_id,
            cell.source_variant,
        ): cell
        for cell in cells
        if cell.method == "pmmc"
    }
    reports = []
    for key, after in sorted(index.items()):
        dataset, model, treatment, budget, round_id, variant = key
        if variant != "active":
            continue
        comparisons = [("fallback_on_off", (*key[:-1], "no_fallback"))]
        if round_id > 0:
            comparisons.append(
                ("round_revision", (dataset, model, treatment, budget, round_id - 1, variant))
            )
        for kind, before_key in comparisons:
            before = index.get(before_key)
            if before is None:
                continue
            for metric in ("typed_exact_match", "method_blind_judge"):
                pairs, missing, inapplicable = [], 0, 0
                for query in selected[dataset]:
                    a = rows.get((before.cell_id, query))
                    b = rows.get((after.cell_id, query))
                    name = "applicable" if metric == "typed_exact_match" else "judge_applicable"
                    if any(
                        value is None
                        or value.get("engineering_error")
                        or value.get("status") not in {"answered", "abstention"}
                        or (value.get("status") == "answered" and not value.get("typed_answer_id"))
                        for value in (a, b)
                    ):
                        missing += 1
                        continue
                    if (
                        a is not None
                        and b is not None
                        and (a.get(name) is False and b.get(name) is False)
                    ):
                        inapplicable += 1
                        continue
                    left, right = _value(a, metric), _value(b, metric)
                    if left is None or right is None:
                        missing += 1
                    else:
                        pairs.append((left, right))
                complete = missing == 0
                reports.append(
                    {
                        "kind": kind,
                        "metric": metric,
                        "dataset": dataset,
                        "answer_model": model,
                        "treatment": treatment,
                        "question_budget": budget,
                        "before_round": before.round_id,
                        "after_round": round_id,
                        "before_cell_id": before.cell_id,
                        "after_cell_id": after.cell_id,
                        "selected_count": len(selected[dataset]),
                        "paired_count": len(pairs),
                        "not_applicable_count": inapplicable,
                        "missing_pair_count": missing,
                        "status": (
                            "incomplete" if missing else "complete" if pairs else "not_applicable"
                        ),
                        "improved_count": sum(b > a for a, b in pairs),
                        "degraded_count": sum(b < a for a, b in pairs),
                        "unchanged_count": sum(b == a for a, b in pairs),
                        "mean_delta": math.fsum(b - a for a, b in pairs) / len(pairs)
                        if complete and pairs
                        else None,
                        "causal_operator_effect": False,
                    }
                )
    return reports
