"""Offline semantic Judge summaries over verified selected query/cell receipts.

Full-bank Cover and saved-top-50 Match@K are query-level binary observations,
not candidate-pair ratios, evidence proxies, or answer accuracy measurements.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pmmc.evaluation.qbank_semantic import semantic_reason_codes
from pmmc.evaluation.step8_qbank_semantic import (
    JUDGE_ALIAS,
    VERSION,
    SemanticCellResult,
    SemanticObservation,
    SemanticPairResult,
    Step8SemanticReport,
)
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8_process import MATCH_KS

if TYPE_CHECKING:
    from pmmc.experiments.step8_analysis import CampaignFacts
    from pmmc.experiments.step8_jobs import BenchmarkStageJob
    from pmmc.schemas.step8 import ExperimentCell

_METRICS = ("semantic_cover", *(f"semantic_match_at_{k}" for k in MATCH_KS))


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("step8_semantic_analysis_" + code)


def _hash(value: str | None) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _observations(cell: SemanticCellResult) -> dict[str, SemanticObservation]:
    return {
        "semantic_cover": cell.semantic_cover,
        **{f"semantic_match_at_{k}": cell.semantic_match_at_k[k] for k in MATCH_KS},
    }


def _validate_pairs(report: Step8SemanticReport) -> None:
    pairs = {pair.qbank_id: pair for pair in report.pairs}
    _require(
        len(pairs) == len(report.pairs)
        and len({pair.pair_hash for pair in report.pairs}) == len(report.pairs),
        "duplicate_pair",
    )
    for pair in report.pairs:
        _require(
            bool(pair.qbank_id)
            and all(
                _hash(value) for value in (pair.pair_hash, pair.question_hash, pair.support_hash)
            )
            and pair.model_call_id == "step8-semantic-" + pair.pair_hash,
            "pair_identity_invalid",
        )
        if pair.status == "evaluated":
            _require(pair.dimensions is not None, "pair_dimensions_missing")
            assert pair.dimensions is not None
            reasons = tuple(reason.value for reason in semantic_reason_codes(pair.dimensions))
            _require(
                pair.semantic_match is (not reasons)
                and pair.reason_codes == reasons
                and pair.error_code is None
                and _hash(pair.trace_hash),
                "pair_verdict_invalid",
            )
        else:
            _require(
                pair.semantic_match is None
                and pair.dimensions is None
                and not pair.reason_codes
                and bool(pair.error_code)
                and (pair.trace_hash is None or _hash(pair.trace_hash)),
                "incomplete_pair_has_score",
            )
    _require(
        report.status
        == ("evaluated" if all(p.status == "evaluated" for p in report.pairs) else "incomplete"),
        "report_status_mismatch",
    )
    union: set[str] = set()
    for cell in report.cells:
        pool = cell.pool
        cover, match = pool.cover_qbank_ids, pool.match_qbank_ids
        _require(
            len(set(cover)) == len(cover)
            and len(set(match)) == len(match)
            and all(cover)
            and set(match) <= set(cover)
            and len(match) <= 50
            and set(cover) <= pairs.keys()
            and set(cell.semantic_match_at_k) == set(MATCH_KS),
            "pool_denominator_invalid",
        )
        union.update(cover)
        for observation, ids in (
            (cell.semantic_cover, cover),
            *((cell.semantic_match_at_k[k], match[:k]) for k in MATCH_KS),
        ):
            evaluated = [pairs[key] for key in ids if pairs[key].status == "evaluated"]
            complete = len(evaluated) == len(ids)
            _require(
                observation.pair_count == len(ids)
                and observation.evaluated_pair_count == len(evaluated)
                and observation.status == ("evaluated" if complete else "incomplete")
                and observation.value
                == (int(any(p.semantic_match for p in evaluated)) if complete else None),
                "observation_denominator_or_value_invalid",
            )
    _require(union == pairs.keys(), "pair_pool_coverage_mismatch")


def _bound_report(
    facts: CampaignFacts,
    job: BenchmarkStageJob,
    output: dict[str, Any],
    matrix: dict[str, ExperimentCell],
) -> Step8SemanticReport:
    try:
        report = Step8SemanticReport.model_validate_json(
            json.dumps(output["qbank_semantic"], allow_nan=False),
            strict=True,
        )
    except (ValueError, TypeError):
        raise ValueError("step8_semantic_analysis_report_schema_invalid") from None
    selected = output.get("selected_query")
    _require(isinstance(selected, dict), "selected_query_missing")
    assert isinstance(selected, dict)
    query, dataset = selected.get("query_id"), selected.get("dataset")
    scope = job.scientific_inputs
    _require(
        output.get("schema_version") == "step8-query-evaluation-v1"
        and job.campaign_id == output.get("campaign_id") == facts.manifest.campaign_id
        and isinstance(query, str)
        and isinstance(dataset, str)
        and query == output.get("query_id") == scope.get("query_id") == report.query_id
        and dataset == scope.get("dataset")
        and dataset in facts.selection.query_ids
        and query in facts.selection.query_ids[DatasetName(dataset)]
        and report.snapshot_id
        == scope.get("snapshot_id")
        == facts.manifest.snapshot_ids[DatasetName(dataset)]
        and isinstance(output.get("query_executor_run_id"), str)
        and bool(output["query_executor_run_id"]),
        "evaluation_scope_mismatch",
    )
    _require(
        output["qbank_semantic"].get("protocol_version") == report.protocol_version == VERSION
        and report.judge_model_alias == JUDGE_ALIAS == facts.manifest.protocol.judge_alias
        and all(_hash(v) for v in (report.protocol_hash, report.plan_hash, report.judge_model_hash))
        and all(
            (
                report.evaluation_run_id,
                report.owner_query_run_id,
                report.canonical_query_run_id,
                report.compiler_run_id,
            )
        ),
        "semantic_identity_invalid",
    )
    scores = output.get("scores")
    _require(
        isinstance(scores, list) and all(isinstance(s, dict) for s in scores), "scores_invalid"
    )
    assert isinstance(scores, list)
    by_cell = {}
    for score in scores:
        identity = score.get("cell_id")
        _require(isinstance(identity, str) and identity in matrix, "cell_outside_matrix")
        cell = matrix[identity]
        _require(
            cell.dataset.value == dataset
            and cell.answer_model == scope.get("answer_model")
            and cell.treatment.value == scope.get("treatment", "main")
            and score.get("method") == cell.method
            and cell.protocol_hash == facts.manifest.protocol.content_hash,
            "cell_scope_mismatch",
        )
        _require(identity not in by_cell, "duplicate_score_cell")
        by_cell[identity] = score
    expected = {key for key in by_cell if matrix[key].method == "pmmc"}
    _require(
        bool(expected)
        and len(report.cells) == len(expected)
        and {row.pool.cell_id for row in report.cells} == expected,
        "semantic_cell_denominator_mismatch",
    )
    for row in report.cells:
        pool = row.pool
        cell, score = matrix[pool.cell_id], by_cell[pool.cell_id]
        _require(
            pool.question_budget == cell.question_budget
            and pool.round_id == cell.round_id
            and pool.source_fact_id == score.get("source_fact_id")
            and pool.source_fact_hash == score.get("source_fact_hash")
            and all((pool.source_fact_id, pool.match_batch_id, pool.round_projection_id))
            and all(
                _hash(v)
                for v in (
                    pool.source_fact_hash,
                    pool.match_batch_hash,
                    pool.round_projection_hash,
                    pool.round_members_hash,
                )
            ),
            "cell_lineage_mismatch",
        )
    _validate_pairs(report)
    return report


def summarize_campaign_semantics(facts: CampaignFacts) -> dict[str, Any]:
    """Missing evidence is incomplete; malformed or conflicting evidence is rejected.

    CampaignFacts supplies the CAS-verified receipt/dependency boundary. Shared pair
    owners may differ from a consuming backbone's query run; observations are never
    copied into an absent query/cell receipt merely because another backbone has one.
    """
    matrix = {cell.cell_id: cell for cell in facts.cells}
    _require(len(matrix) == len(facts.cells), "duplicate_matrix_cell")
    records: dict[tuple[str, str], SemanticCellResult] = {}
    judge_identities: set[tuple[str, str]] = set()
    owners: dict[tuple[str, str, str], tuple[str, str]] = {}
    shared: dict[tuple[str, str, str, str], SemanticPairResult] = {}
    for job_id in sorted(facts.verified_jobs):
        job = facts.jobs[job_id]
        if job.stage != "evaluation":
            continue
        output = facts.receipts[job_id].output
        payload = output.get("qbank_semantic")
        if payload is None:
            continue
        # These are the producer's explicit no-report envelopes, not binary scores.
        if (
            isinstance(payload, dict)
            and payload.get("status") in {"incomplete", "not_applicable"}
            and "cells" not in payload
        ):
            _require(
                set(payload) <= {"status", "reason", "error_code"}, "missing_report_schema_invalid"
            )
            continue
        report = _bound_report(facts, job, output, matrix)
        judge_identities.add((report.protocol_hash, report.judge_model_hash))
        _require(len(judge_identities) == 1, "mixed_judge_protocol")
        scope = report.snapshot_id, report.query_id, report.compiler_run_id
        owner = report.evaluation_run_id, report.owner_query_run_id
        _require(scope not in owners or owners[scope] == owner, "shared_owner_conflict")
        owners[scope] = owner
        for pair in report.pairs:
            key = (*scope, pair.qbank_id)
            previous = shared.get(key)
            if previous is not None:
                _require(
                    previous.pair_hash == pair.pair_hash
                    and (
                        previous.status != "evaluated"
                        or pair.status != "evaluated"
                        or previous == pair
                    ),
                    "shared_pair_conflict",
                )
            if previous is None or pair.status == "evaluated":
                shared[key] = pair
        for row in report.cells:
            record_key = row.pool.cell_id, report.query_id
            _require(record_key not in records, "duplicate_query_cell")
            records[record_key] = row
    summaries = []
    for cell in facts.cells:
        selected = facts.selection.query_ids[cell.dataset]
        metrics = {}
        for name in _METRICS:
            applicable = cell.method == "pmmc"
            observations = (
                [
                    _observations(records[(cell.cell_id, query)])[name]
                    for query in selected
                    if (cell.cell_id, query) in records
                ]
                if applicable
                else []
            )
            evaluated = [v for v in observations if v.status == "evaluated"]
            missing = len(selected) - len(evaluated) if applicable else 0
            numerator = sum(v.value for v in evaluated if v.value is not None)
            metrics[name] = {
                "status": "not_applicable"
                if not applicable
                else "incomplete"
                if missing
                else "complete",
                "selected_count": len(selected),
                "evaluated_count": len(evaluated),
                "engineering_missing_count": missing,
                "missing_report_count": len(selected) - len(observations) if applicable else 0,
                "incomplete_observation_count": len(observations) - len(evaluated),
                "not_applicable_count": 0 if applicable else len(selected),
                "empty_pool_count": sum(v.pair_count == 0 for v in evaluated),
                "observed_numerator": numerator,
                "observed_denominator": len(evaluated),
                "denominator_kind": "selected_queries_per_cell",
                "query_mean": numerator / len(selected)
                if applicable and selected and not missing
                else None,
                "reported_pair_count": sum(v.pair_count for v in observations),
                "evaluated_pair_count": sum(v.evaluated_pair_count for v in observations),
            }
        summaries.append({**cell.model_dump(mode="json"), "metrics": metrics})
    return {
        "schema_version": "step8-semantic-analysis-v1",
        "status": "incomplete"
        if any(m["engineering_missing_count"] for c in summaries for m in c["metrics"].values())
        else "complete"
        if any(c.method == "pmmc" for c in facts.cells)
        else "not_applicable",
        "scope": "selected_query_cell_receipts",
        "cover_pool": "full_exact_round_eligible_answerable_bank",
        "match_pool": "saved_runtime_answerable_top_50_prefixes",
        "match_ks": list(MATCH_KS),
        "cells": summaries,
        "unavailable_metrics": ["oracle_answer_accuracy"],
        "scientific_approval": False,
        "warnings": [
            "semantic_judge_scores_are_not_evidence_proxies_or_answer_accuracy",
            "pair_counts_are_diagnostics_not_query_denominators",
            "missing_semantic_facts_are_not_zero",
            "human_semantic_validation_required",
        ],
    }


summarise_campaign_semantics = summarize_campaign_semantics
