"""Offline Judge summaries with explicit applicability and complete paired denominators."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from pmmc.evaluation.paper_statistics import (
    ComparisonResult,
    MethodVariant,
    PlannedComparison,
    ScoreObservation,
    SelectedQuery,
    StatisticsConfig,
    TerminalStatus,
    TypedScore,
    compare_family,
    holm_adjust,
    summarize_method,
)
from pmmc.schemas.step8 import BASELINE_METHODS, ExperimentCell

METRIC = "method_blind_judge"


def judge_observation(query_id: str, cell: ExperimentCell, row: dict) -> ScoreObservation:
    method = MethodVariant(cell.method, cell.cell_id)
    missing = ScoreObservation(query_id, method, TerminalStatus.ENGINEERING_MISSING)
    applicable = row.get("judge_applicable")
    if type(applicable) is not bool or row.get("engineering_error"):
        return missing
    status = TerminalStatus(row["status"])
    if status is TerminalStatus.ENGINEERING_MISSING:
        return missing
    if status is TerminalStatus.ABSTENTION:
        if row.get("judge_value") is not None:
            raise ValueError("abstention cannot carry a Judge score")
        return ScoreObservation(query_id, method, status)
    if not row.get("typed_answer_id"):
        return missing
    if not applicable:
        if row.get("judge_value") is not None:
            raise ValueError("non-applicable query cannot carry a Judge score")
        return ScoreObservation(
            query_id,
            method,
            status,
            TypedScore(None, False, METRIC, unavailable_reason="judge_not_applicable"),
        )
    if (
        row.get("judge_value") is None
        or row.get("judge_method_blind") is not True
        or not row.get("judge_model_call_id")
    ):
        return missing
    return ScoreObservation(query_id, method, status, TypedScore(row["judge_value"], metric=METRIC))


def summarize_judge(
    *,
    cells: tuple[ExperimentCell, ...],
    selections: dict[str, tuple[SelectedQuery, ...]],
    rows: dict[tuple[str, str], dict],
    answer_models: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    """Do not infer applicability from whether the model happened to answer.

    These are a separately corrected, explicitly secondary comparison family.
    They never replace typed scores or confer scientific approval.
    """
    config = StatisticsConfig(seed=seed, metric=METRIC)
    applicability: dict[tuple[str, str], bool] = {}
    for cell in cells:
        for query in selections[cell.dataset.value]:
            row = rows.get((cell.cell_id, query.query_id), {})
            value = row.get("judge_applicable")
            if type(value) is not bool:
                continue
            key = query.dataset, query.query_id
            if key in applicability and applicability[key] != value:
                raise ValueError("Judge applicability differs across methods or variants")
            applicability[key] = value
    selected = {
        dataset: tuple(
            replace(q, applicable=applicability.get((dataset, q.query_id), False)) for q in queries
        )
        for dataset, queries in selections.items()
    }
    observations = {
        cell.cell_id: tuple(
            judge_observation(query.query_id, cell, rows[(cell.cell_id, query.query_id)])
            for query in selected[cell.dataset.value]
            if (cell.cell_id, query.query_id) in rows
        )
        for cell in cells
    }
    summaries = []
    for cell in cells:
        queries = selected[cell.dataset.value]
        summary = asdict(
            summarize_method(
                queries,
                observations[cell.cell_id],
                MethodVariant(cell.method, cell.cell_id),
                config,
            )
        )
        summary.pop("missing_query_ids")
        summary.pop("method")
        pairs = [
            rows[(cell.cell_id, query.query_id)]
            for query in queries
            if (cell.cell_id, query.query_id) in rows
            and rows[(cell.cell_id, query.query_id)].get("judge_applicable") is True
            and not rows[(cell.cell_id, query.query_id)].get("engineering_error")
            and rows[(cell.cell_id, query.query_id)].get("judge_model_call_id")
            and rows[(cell.cell_id, query.query_id)].get("judge_method_blind") is True
            and rows[(cell.cell_id, query.query_id)].get("judge_value") is not None
            and rows[(cell.cell_id, query.query_id)].get("value") is not None
        ]
        summaries.append(
            {
                **cell.model_dump(mode="json"),
                **summary,
                "unknown_applicability_count": sum(
                    (q.dataset, q.query_id) not in applicability for q in queries
                ),
                "typed_judge_paired_count": len(pairs),
                "typed_judge_disagreement_count": sum(
                    row["value"] != row["judge_value"] for row in pairs
                ),
            }
        )
    comparisons: list[ComparisonResult] = []
    contexts: list[tuple[str, str]] = []
    for dataset, queries in sorted(selected.items()):
        for model in answer_models:
            main = {
                c.method: c
                for c in cells
                if c.dataset.value == dataset
                and c.answer_model == model
                and c.treatment.value == "main"
                and c.question_budget == 8
                and c.round_id == 2
                and c.source_variant == "active"
            }
            family = tuple(
                PlannedComparison(
                    f"judge:{dataset}:{model}:pmmc-vs-{baseline}",
                    MethodVariant("pmmc", main["pmmc"].cell_id if "pmmc" in main else "unassigned"),
                    MethodVariant(
                        baseline, main[baseline].cell_id if baseline in main else "unassigned"
                    ),
                    exploratory=True,
                )
                for baseline in BASELINE_METHODS
            )
            comparisons.extend(
                compare_family(
                    queries,
                    tuple(o for c in main.values() for o in observations[c.cell_id]),
                    family,
                    config,
                )
            )
            contexts.extend((dataset, model) for _ in family)
    if comparisons and all(c.p_value is not None for c in comparisons):
        adjusted = holm_adjust(tuple(c.p_value for c in comparisons if c.p_value is not None))
        comparisons = [
            replace(c, holm_p_value=p) for c, p in zip(comparisons, adjusted, strict=True)
        ]
    else:
        comparisons = [
            replace(
                c,
                holm_p_value=None,
                warnings=tuple(sorted(set((*c.warnings, "judge_family_incomplete")))),
            )
            for c in comparisons
        ]
    public = []
    for result, (dataset, model) in zip(comparisons, contexts, strict=True):
        value = asdict(result)
        for name in ("micro", "cluster_macro"):
            if value[name] is not None:
                value[name]["leave_one_cluster_out"] = [
                    v for _, v in value[name]["leave_one_cluster_out"]
                ]
        public.append({"dataset": dataset, "answer_model": model, **value})
    return {
        "metric": METRIC,
        "status": "complete"
        if summaries and all(s["status"] in {"complete", "not_applicable"} for s in summaries)
        else "incomplete",
        "comparison_family": "secondary_judge_all_declared_main_dataset_backbone_baselines",
        "scientific_approval": False,
        "cells": summaries,
        "comparisons": public,
        "warnings": ["missing_judge_applicability_is_not_zero", "typed_images_use_typed_scores"],
    }
