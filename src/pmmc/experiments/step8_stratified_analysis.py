"""Descriptive category/scenario estimates over the same frozen query denominator."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

from pmmc.evaluation.paper_statistics import (
    MethodVariant,
    SelectedQuery,
    StatisticsConfig,
    summarize_method,
)
from pmmc.experiments.step8_judge_analysis import METRIC, judge_observation

DECLARED_STRATA = {"memlens": "source_category", "memgallery": "scenario_id"}


def summarize_strata(*, cells, selections, observations, rows, seed: int) -> dict:
    """Do not mistake conversation-cluster macro for category/scenario macro.

    Group labels come only from evaluator-selected query metadata. Missing
    labels or scores block the macro; they never remove queries from selection.
    These descriptive summaries do not add post-hoc hypothesis tests.
    """
    summaries = []
    for cell in cells:
        selected: tuple[SelectedQuery, ...] = selections[cell.dataset.value]
        field = DECLARED_STRATA[cell.dataset.value]
        grouped: dict[str, list[SelectedQuery]] = {}
        for query in selected:
            label = dict(query.strata).get(field)
            if label:
                grouped.setdefault(label, []).append(query)
        unknown = len(selected) - sum(len(group) for group in grouped.values())
        for metric in ("typed_exact_match", METRIC):
            config = StatisticsConfig(seed=seed, metric=metric)
            groups = []
            for label, members in sorted(grouped.items()):
                ids = {query.query_id for query in members}
                if metric == METRIC:
                    group = tuple(
                        replace(
                            q,
                            applicable=rows.get((cell.cell_id, q.query_id), {}).get(
                                "judge_applicable"
                            )
                            is True,
                        )
                        for q in members
                    )
                    values = tuple(
                        judge_observation(q.query_id, cell, rows[cell.cell_id, q.query_id])
                        for q in group
                        if (cell.cell_id, q.query_id) in rows
                    )
                else:
                    group = tuple(members)
                    values = tuple(o for o in observations[cell.cell_id] if o.query_id in ids)
                value = asdict(
                    summarize_method(
                        group, values, MethodVariant(cell.method, cell.cell_id), config
                    )
                )
                value.pop("missing_query_ids")
                value.pop("method")
                groups.append({"stratum": label, **value})
            complete = (
                bool(selected)
                and unknown == 0
                and all(group["status"] in {"complete", "not_applicable"} for group in groups)
            )
            applicable = [group for group in groups if group["status"] == "complete"]
            summaries.append(
                {
                    "cell_id": cell.cell_id,
                    **cell.model_dump(mode="json"),
                    "metric": metric,
                    "stratum_field": field,
                    "status": "incomplete"
                    if not complete
                    else "complete"
                    if applicable
                    else "not_applicable",
                    "selected_count": len(selected),
                    "stratum_count": len(groups),
                    "applicable_stratum_count": len(applicable),
                    "unknown_stratum_count": unknown,
                    "stratum_macro": math.fsum(g["micro"] for g in applicable) / len(applicable)
                    if complete and applicable
                    else None,
                    "stratum_macro_basis": "equal_weight_mean_of_applicable_stratum_query_means",
                    "groups": groups,
                }
            )
    return {
        "schema_version": "step8-stratified-analysis-v1",
        "status": "complete"
        if summaries and all(s["status"] in {"complete", "not_applicable"} for s in summaries)
        else "incomplete",
        "declared_strata": DECLARED_STRATA.copy(),
        "cells": summaries,
        "scientific_approval": False,
        "warnings": [
            "descriptive_strata_do_not_add_primary_comparisons",
            "missing_stratum_metadata_does_not_shrink_selection",
            "category_or_scenario_macro_is_not_conversation_cluster_macro",
        ],
    }
