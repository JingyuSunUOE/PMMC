"""Offline automatic analysis over verified Questioner and coverage exports."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pmmc.compiler.questioner_export import verify_questioner_export
from pmmc.evaluation.qbank_export import verify_qbank_coverage_export
from pmmc.pilots.config import QuestionerPilotConfig
from pmmc.pilots.io import read_jsonl
from pmmc.schemas import BenchmarkPartitionAssignment, DatasetName


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(repetitions)
    )
    lower = samples[max(0, math.floor(0.025 * repetitions) - 1)]
    upper = samples[min(repetitions - 1, math.ceil(0.975 * repetitions) - 1)]
    return lower, upper


def _distribution(values: list[str]) -> list[dict[str, Any]]:
    return [{"value": key, "count": count} for key, count in sorted(Counter(values).items())]


def _metric_is_binary(name: str) -> bool:
    return (
        any(marker in name for marker in ("cover", "match", "fallback", "hit", "complete"))
        and "evidence_recall" not in name
    )


def _coverage_curves(
    *,
    dataset: DatasetName,
    metrics: list[dict[str, Any]],
    config: QuestionerPilotConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        grouped[
            (
                str(metric["metric_name"]),
                metric.get("question_budget"),
                metric.get("k"),
            )
        ].append(metric)
    curves: list[dict[str, Any]] = []
    for (name, budget, k), values in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] or 0, item[0][2] or 0),
    ):
        applicable = [float(item["value"]) for item in values if item.get("applicable")]
        interval: tuple[float, float] | None
        interval_method: str
        if _metric_is_binary(name):
            interval = wilson_interval(sum(value >= 0.5 for value in applicable), len(applicable))
            interval_method = "wilson_95"
        else:
            interval = bootstrap_mean_interval(
                applicable,
                seed=config.bootstrap_seed,
                repetitions=config.bootstrap_repetitions,
            )
            interval_method = "percentile_bootstrap_95"
        curves.append(
            {
                "dataset": dataset.value,
                "metric_name": name,
                "question_budget": budget,
                "k": k,
                "query_count": len(values),
                "applicable_count": len(applicable),
                "not_applicable_count": len(values) - len(applicable),
                "mean": sum(applicable) / len(applicable) if applicable else None,
                "interval_method": interval_method,
                "ci95_low": interval[0] if interval else None,
                "ci95_high": interval[1] if interval else None,
                "underpowered": len(applicable) < config.underpowered_stratum_size,
            }
        )
    return curves


def _marginal_rows(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            str(item["dataset"]),
            str(item["metric_name"]),
            item.get("question_budget"),
            item.get("k"),
        ): item
        for item in curves
    }
    result: list[dict[str, Any]] = []
    for item in curves:
        mean = item.get("mean")
        if mean is None:
            continue
        budget = item.get("question_budget")
        k = item.get("k")
        if budget in {2, 4, 8}:
            prior_budget = {2: 1, 4: 2, 8: 4}[int(budget)]
            prior = indexed.get((str(item["dataset"]), str(item["metric_name"]), prior_budget, k))
            if prior is not None and prior.get("mean") is not None:
                result.append(
                    {
                        "dataset": item["dataset"],
                        "metric_name": item["metric_name"],
                        "axis": "question_budget",
                        "from": prior_budget,
                        "to": budget,
                        "k": k,
                        "gain": float(mean) - float(prior["mean"]),
                    }
                )
    by_budget: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for item in curves:
        if item.get("k") is not None and item.get("mean") is not None:
            by_budget[
                (
                    str(item["dataset"]),
                    str(item["metric_name"]),
                    item.get("question_budget"),
                )
            ].append(item)
    for (dataset, metric_name, budget), values in sorted(by_budget.items()):
        ordered = sorted(values, key=lambda item: int(item["k"]))
        for previous_curve, current_curve in zip(ordered, ordered[1:], strict=False):
            result.append(
                {
                    "dataset": dataset,
                    "metric_name": metric_name,
                    "axis": "retrieval_k",
                    "from": int(previous_curve["k"]),
                    "to": int(current_curve["k"]),
                    "question_budget": budget,
                    "gain": float(current_curve["mean"]) - float(previous_curve["mean"]),
                }
            )
    return sorted(
        result,
        key=lambda item: (
            str(item["dataset"]),
            str(item["metric_name"]),
            str(item["axis"]),
            int(item["from"]),
            int(item.get("k") or item.get("question_budget") or 0),
        ),
    )


def _interval_summary(
    values: list[float],
    *,
    metric_name: str,
    config: QuestionerPilotConfig,
) -> dict[str, Any]:
    if _metric_is_binary(metric_name):
        interval = wilson_interval(sum(value >= 0.5 for value in values), len(values))
        method = "wilson_95"
    else:
        interval = bootstrap_mean_interval(
            values,
            seed=config.bootstrap_seed,
            repetitions=config.bootstrap_repetitions,
        )
        method = "percentile_bootstrap_95"
    return {
        "applicable_count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "interval_method": method,
        "ci95_low": interval[0] if interval else None,
        "ci95_high": interval[1] if interval else None,
        "underpowered": len(values) < config.underpowered_stratum_size,
    }


def _stratified_coverage(
    *,
    dataset: DatasetName,
    metrics: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    assignment_by_query: dict[str, BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
) -> list[dict[str, Any]]:
    query_by_id = {str(item["query_id"]): item for item in queries}
    grouped: dict[tuple[str, str, str, int | None, int | None], list[dict[str, Any]]] = defaultdict(
        list
    )
    for metric in metrics:
        query_id = str(metric.get("query_id") or "")
        assignment = assignment_by_query.get(query_id)
        query = query_by_id.get(query_id)
        if assignment is None or query is None:
            continue
        dimensions = {
            "source_category": assignment.source_category,
            "has_query_image": str(bool(query.get("has_query_image"))).lower(),
            "hop_scope": (
                "multi_hop" if int(query.get("gold_group_count", 0)) > 1 else "single_hop"
            ),
        }
        for dimension, value in dimensions.items():
            grouped[
                (
                    dimension,
                    value,
                    str(metric["metric_name"]),
                    metric.get("question_budget"),
                    metric.get("k"),
                )
            ].append(metric)
    result: list[dict[str, Any]] = []
    for (dimension, value, metric_name, budget, k), rows in sorted(
        grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        applicable = [
            float(row["value"])
            for row in rows
            if row.get("applicable") and row.get("value") is not None
        ]
        result.append(
            {
                "dataset": dataset.value,
                "dimension": dimension,
                "stratum": value,
                "metric_name": metric_name,
                "question_budget": budget,
                "k": k,
                "query_count": len(rows),
                "not_applicable_count": len(rows) - len(applicable),
                **_interval_summary(
                    applicable,
                    metric_name=metric_name,
                    config=config,
                ),
            }
        )
    return result


def _scenario_macro_coverage(
    *,
    dataset: DatasetName,
    metrics: list[dict[str, Any]],
    assignment_by_query: dict[str, BenchmarkPartitionAssignment],
    config: QuestionerPilotConfig,
) -> list[dict[str, Any]]:
    if dataset is not DatasetName.MEMGALLERY:
        return []
    grouped: dict[tuple[str, int | None, int | None], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for metric in metrics:
        if not metric.get("applicable") or metric.get("value") is None:
            continue
        assignment = assignment_by_query.get(str(metric.get("query_id") or ""))
        if assignment is None or assignment.scenario_id is None:
            continue
        grouped[
            (
                str(metric["metric_name"]),
                metric.get("question_budget"),
                metric.get("k"),
            )
        ][assignment.scenario_id].append(float(metric["value"]))
    result: list[dict[str, Any]] = []
    for (metric_name, budget, k), scenario_values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            int(item[0][1] or 0),
            int(item[0][2] or 0),
        ),
    ):
        scenario_means = [
            sum(values) / len(values) for _scenario, values in sorted(scenario_values.items())
        ]
        interval = bootstrap_mean_interval(
            scenario_means,
            seed=config.bootstrap_seed,
            repetitions=config.bootstrap_repetitions,
        )
        result.append(
            {
                "dataset": dataset.value,
                "metric_name": metric_name,
                "question_budget": budget,
                "k": k,
                "scenario_count": len(scenario_means),
                "scenario_macro_mean": (
                    sum(scenario_means) / len(scenario_means) if scenario_means else None
                ),
                "interval_method": "scenario_cluster_bootstrap_95",
                "ci95_low": interval[0] if interval else None,
                "ci95_high": interval[1] if interval else None,
                "underpowered": len(scenario_means) < 4,
            }
        )
    return result


def _round_block_summary(
    *,
    dataset: DatasetName,
    units: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    assignments: list[BenchmarkPartitionAssignment],
) -> dict[str, Any] | None:
    if dataset is not DatasetName.MEMGALLERY:
        return None
    scenario_by_conversation = {
        item.conversation_id: item.scenario_id
        for item in assignments
        if item.dataset is dataset and item.scenario_id is not None
    }
    eligible_by_unit = Counter(
        str(item["compiler_unit_id"])
        for item in gates
        if item.get("status") == "eligible_for_planning"
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        scenario = scenario_by_conversation.get(str(unit["conversation_id"]))
        if scenario is not None:
            grouped[scenario].append(unit)
    scenario_rows: list[dict[str, Any]] = []
    for scenario, values in sorted(grouped.items()):
        source_rounds = {
            str(round_id)
            for unit in values
            for round_id in dict(unit.get("metadata", {})).get("anchor_source_round_ids", [])
        }
        eligible = sum(eligible_by_unit[str(unit["compiler_unit_id"])] for unit in values)
        token_counts = [
            int(dict(unit.get("metadata", {})).get("prepared_input_tokens", 0)) for unit in values
        ]
        image_counts = [
            int(dict(unit.get("metadata", {})).get("input_image_count", 0)) for unit in values
        ]
        scenario_rows.append(
            {
                "scenario_id": scenario,
                "unit_count": len(values),
                "source_round_count": len(source_rounds),
                "eligible_candidate_count": eligible,
                "eligible_candidates_per_unit": eligible / len(values),
                "candidates_per_source_round": (
                    eligible / len(source_rounds) if source_rounds else None
                ),
                "mean_input_tokens": sum(token_counts) / len(token_counts),
                "mean_input_images": sum(image_counts) / len(image_counts),
            }
        )
    return {
        "scenario_count": len(scenario_rows),
        "unit_count": sum(int(item["unit_count"]) for item in scenario_rows),
        "source_round_count": sum(int(item["source_round_count"]) for item in scenario_rows),
        "scenario_rows": scenario_rows,
    }


def _failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        dimensions = {
            "source_category": str(row["source_category"]),
            "has_query_image": str(bool(row["has_query_image"])).lower(),
            "hop_scope": ("multi_hop" if int(row["gold_group_count"]) > 1 else "single_hop"),
        }
        for label in row["labels"]:
            for dimension, stratum in dimensions.items():
                counts[(str(row["dataset"]), dimension, stratum, str(label))] += 1
    return [
        {
            "dataset": dataset,
            "dimension": dimension,
            "stratum": stratum,
            "failure_category": category,
            "count": count,
        }
        for (dataset, dimension, stratum, category), count in sorted(counts.items())
    ]


def analyze_questioner_quality(
    *,
    config: QuestionerPilotConfig,
    runtime_exports: dict[DatasetName, Path],
    coverage_exports: dict[DatasetName, Path],
    assignments: list[BenchmarkPartitionAssignment],
) -> dict[str, Any]:
    if set(runtime_exports) != set(coverage_exports):
        raise ValueError("automatic quality analysis requires paired runtime/coverage exports")
    assignment_by_query = {item.query_id: item for item in assignments}
    curves: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    scenario_macro_rows: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    failure_rows: list[dict[str, Any]] = []
    source_export_ids: dict[str, dict[str, str]] = {}
    round_block_summaries: dict[str, Any] = {}
    for dataset in sorted(runtime_exports, key=lambda item: item.value):
        runtime_export = runtime_exports[dataset]
        coverage_export = coverage_exports[dataset]
        runtime_manifest = verify_questioner_export(runtime_export)
        coverage_manifest = verify_qbank_coverage_export(coverage_export)
        source_export_ids[dataset.value] = {
            "runtime_export_id": str(runtime_manifest["export_id"]),
            "coverage_export_id": str(coverage_manifest["export_id"]),
        }
        runtime = runtime_export / "runtime"
        private = coverage_export / "private"
        generations = read_jsonl(runtime / "questioner_generations.jsonl")
        units = read_jsonl(runtime / "compiler_units.jsonl")
        gates = read_jsonl(runtime / "questioner_candidate_gate_results.jsonl")
        questions = read_jsonl(runtime / "qbank_questions.jsonl")
        questions_by_id = {str(item["qbank_id"]): item for item in questions}
        paraphrases = read_jsonl(runtime / "qbank_paraphrases.jsonl")
        duplicates = read_jsonl(runtime / "qbank_duplicate_links.jsonl")
        links = read_jsonl(runtime / "compiler_unit_links.jsonl")
        evidence_refs = read_jsonl(runtime / "qbank_evidence_refs.jsonl")
        opportunity_path = runtime / "questioner_opportunities.jsonl"
        opportunities = read_jsonl(opportunity_path) if opportunity_path.exists() else []
        verification_path = runtime / "questioner_candidate_verifications.jsonl"
        verifications = read_jsonl(verification_path) if verification_path.exists() else []
        supersession_path = runtime / "questioner_candidate_supersessions.jsonl"
        supersessions = read_jsonl(supersession_path) if supersession_path.exists() else []
        metrics = read_jsonl(private / "coverage_metrics.jsonl")
        labels = read_jsonl(private / "coverage_labels.jsonl")
        queries = read_jsonl(private / "coverage_queries.jsonl")

        raw = sum(int(item.get("raw_candidate_count", 0)) for item in generations)
        valid = sum(int(item.get("valid_candidate_count", 0)) for item in generations)
        invalid = sum(int(item.get("invalid_candidate_count", 0)) for item in generations)
        eligible = sum(item.get("status") == "eligible_for_planning" for item in gates)
        rejected = len(gates) - eligible
        zero_valid = sum(int(item.get("valid_candidate_count", 0)) == 0 for item in generations)
        hard_duplicates = sum(bool(item.get("hard_duplicate")) for item in duplicates)
        semantic_neighbors = len(duplicates) - hard_duplicates
        failure_counts = Counter(str(item.get("category", "unknown")) for item in labels)
        for query in queries:
            query_id = str(query["query_id"])
            assignment = assignment_by_query.get(query_id)
            query_labels = sorted(
                {str(item["category"]) for item in labels if item.get("query_id") == query_id}
            )
            failure_rows.append(
                {
                    "dataset": dataset.value,
                    "query_id": query_id,
                    "source_category": (
                        assignment.source_category if assignment is not None else "unknown"
                    ),
                    "has_query_image": bool(query.get("has_query_image")),
                    "gold_group_count": int(query.get("gold_group_count", 0)),
                    "applicable": bool(query.get("applicable")),
                    "labels": query_labels or ["success"],
                }
            )

        curves.extend(_coverage_curves(dataset=dataset, metrics=metrics, config=config))
        stratified_rows.extend(
            _stratified_coverage(
                dataset=dataset,
                metrics=metrics,
                queries=queries,
                assignment_by_query=assignment_by_query,
                config=config,
            )
        )
        scenario_macro_rows.extend(
            _scenario_macro_coverage(
                dataset=dataset,
                metrics=metrics,
                assignment_by_query=assignment_by_query,
                config=config,
            )
        )
        round_summary = _round_block_summary(
            dataset=dataset,
            units=units,
            gates=gates,
            assignments=assignments,
        )
        if round_summary is not None:
            round_block_summaries[dataset.value] = round_summary
        for name, values in (
            ("question_type", [str(item.get("question_type", "unknown")) for item in questions]),
            ("reasoning_type", [str(item.get("reasoning_type", "unknown")) for item in questions]),
            (
                "required_modality",
                [
                    str(modality)
                    for item in questions
                    for modality in item.get("required_modalities", [])
                ],
            ),
            ("hop_count", [str(item.get("hop_count", "unknown")) for item in questions]),
            (
                "opportunity_type",
                [str(item.get("opportunity_type", "unknown")) for item in opportunities],
            ),
            (
                "verifier_verdict",
                [str(item.get("verdict", "unknown")) for item in verifications],
            ),
            ("unit_relation", [str(item.get("relation_type", "unknown")) for item in links]),
            (
                "evidence_type",
                [str(item.get("evidence_type", "unknown")) for item in evidence_refs],
            ),
        ):
            for row in _distribution(values):
                distributions.append({"dataset": dataset.value, "dimension": name, **row})
        rejection_codes = [str(code) for item in gates for code in item.get("rejection_codes", [])]
        for row in _distribution(rejection_codes):
            distributions.append(
                {"dataset": dataset.value, "dimension": "gate_rejection_code", **row}
            )
        diagnostic_types: list[str] = []
        diagnostic_locations: list[str] = []
        for generation in generations:
            metadata = dict(generation.get("metadata", {}))
            for diagnostic in metadata.get("invalid_diagnostics", []):
                for error in diagnostic.get("errors", []):
                    diagnostic_types.append(str(error.get("type", "unknown")))
                    diagnostic_locations.append(str(error.get("location", "unknown")))
        for dimension, values in (
            ("schema_error_type", diagnostic_types),
            ("schema_error_location", diagnostic_locations),
        ):
            for row in _distribution(values):
                distributions.append({"dataset": dataset.value, "dimension": dimension, **row})
        dataset_summaries[dataset.value] = {
            "unit_count": len(generations),
            "raw_candidate_count": raw,
            "schema_valid_candidate_count": valid,
            "schema_invalid_candidate_count": invalid,
            "eligible_candidate_count": eligible,
            "gate_rejected_candidate_count": rejected,
            "paraphrase_count": len(paraphrases),
            "hard_duplicate_count": hard_duplicates,
            "semantic_neighbor_count": semantic_neighbors,
            "opportunity_count": len(opportunities),
            "verification_count": len(verifications),
            "verifier_pass_count": sum(
                item.get("verdict") == "pass" for item in verifications
            ),
            "verifier_revise_count": sum(
                item.get("verdict") == "revise" for item in verifications
            ),
            "verifier_reject_count": sum(
                item.get("verdict") == "reject" for item in verifications
            ),
            "revision_count": len(supersessions),
            "revision_rescue_count": sum(
                item.get("status") == "eligible_for_planning"
                and questions_by_id.get(str(item.get("qbank_id")), {}).get(
                    "candidate_version"
                )
                == 1
                for item in gates
            ),
            "zero_valid_unit_count": zero_valid,
            "schema_valid_rate": valid / raw if raw else None,
            "gate_pass_rate_among_schema_valid": eligible / valid if valid else None,
            "overall_eligible_rate": eligible / raw if raw else None,
            "zero_valid_unit_rate": zero_valid / len(generations) if generations else None,
            "failure_counts": dict(sorted(failure_counts.items())),
        }
    return {
        "protocol_version": config.protocol_version,
        "source_exports": source_export_ids,
        "datasets": dataset_summaries,
        "coverage_curves": curves,
        "stratified_coverage": stratified_rows,
        "scenario_macro_coverage": scenario_macro_rows,
        "round_block_summaries": round_block_summaries,
        "marginal_analysis": _marginal_rows(curves),
        "distributions": sorted(
            distributions,
            key=lambda item: (str(item["dataset"]), str(item["dimension"]), str(item["value"])),
        ),
        "failures": sorted(
            failure_rows,
            key=lambda item: (str(item["dataset"]), str(item["query_id"])),
        ),
        "failure_summary": _failure_summary(failure_rows),
    }
