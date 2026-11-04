"""Offline actual-Oracle summaries and evaluator inventory materialization hooks."""

from __future__ import annotations

import hashlib
from typing import Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.step8_oracle_hook import oracle_not_evaluated, summarize_oracle_reports
from pmmc.experiments.step8_artifacts import json_bytes
from pmmc.experiments.step8_audit import CampaignAuditInventory
from pmmc.runtime.answering import validate_typed_answer_input
from pmmc.schemas import AnswerGenerationInput, ProgramExecutionTrace
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.query_executor import runtime_fact_payload


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("oracle_analysis_" + reason)


def _validate_bundle_envelope(report: dict[str, Any], bundle: dict[str, Any]) -> None:
    digest = hashlib.sha256(json_bytes(bundle)).hexdigest()
    _require(digest == report.get("private_bundle_sha256"), "bundle_checksum_mismatch")
    private = report.get("private_artifact")
    _require(
        isinstance(private, dict) and private.get("sha256") == digest,
        "private_evaluator_artifact_missing",
    )
    _require(
        canonical_content_hash(bundle.get("report"))
        == canonical_content_hash(
            {
                key: value
                for key, value in report.items()
                if key not in {"private_bundle_sha256", "private_artifact"}
            }
        ),
        "bundle_report_mismatch",
    )


def validate_oracle_bundle(report: dict[str, Any], bundle: dict[str, Any]) -> set[str]:
    """Reject proxy reports and altered/missing execution, context or answer facts."""
    _require(
        summarize_oracle_reports([report], selected_query_ids=[report["query_id"]])["status"]
        == "completed",
        "actual_denominator_incomplete",
    )
    _validate_bundle_envelope(report, bundle)
    rows = report["candidates"]
    _require(
        set(bundle["private_facts"]) == {row["fact_sha256"] for row in rows},
        "private_fact_denominator_mismatch",
    )
    _require(
        set(bundle["program_inputs"]) == {row["qbank_id"] for row in rows},
        "program_denominator_mismatch",
    )
    calls: set[str] = set()
    for row in rows:
        source = bundle["program_inputs"][row["qbank_id"]]
        fact = bundle["private_facts"][row["fact_sha256"]]
        _require(
            hashlib.sha256(json_bytes(fact)).hexdigest() == row["fact_sha256"]
            and canonical_content_hash(source) == row["program_input_hash"],
            "candidate_fact_checksum_mismatch",
        )
        answer = TypedAnswerResult.model_validate(fact["answer"])
        execution = ProgramExecutionTrace.model_validate(fact["execution"]["trace"])
        prepared = AnswerGenerationInput.model_validate(fact["prepared"])
        validate_typed_answer_input(prepared)
        _require(
            fact["status"] == "completed"
            and execution.status == "success"
            and execution.plan_id == source["program"]["plan_id"] == row["plan_id"]
            and answer.run_id == execution.run_id == prepared.run_id == report["query_run_id"]
            and answer.query_id
            == execution.query_id
            == prepared.query.query_id
            == report["query_id"]
            and answer.snapshot_id
            == execution.snapshot_id
            == prepared.snapshot_id
            == report["snapshot_id"]
            and answer.answer_variant == prepared.answer_variant
            and answer.answer_variant.startswith("oracle-at-50-")
            and answer.typed_answer_id == row["typed_answer_id"]
            and canonical_content_hash(runtime_fact_payload(answer)) == row["answer_hash"]
            and answer.terminal_state == row["terminal_state"]
            and bool(answer.calls),
            "actual_execution_answer_lineage_mismatch",
        )
        calls.update(call.model_call_id for call in answer.calls)
        if row["judge"] is not None:
            judge = row["judge"]
            _require(
                judge["evaluation_run_id"] == report["evaluation_run_id"]
                and judge["answer_id"] == answer.typed_answer_id
                and judge["query_id"] == answer.query_id
                and judge["llm_judge_score"] == row["score"],
                "judge_lineage_mismatch",
            )
            calls.add(judge["model_call_id"])
    return calls


def _inventory_artifacts(facts):
    from pmmc.experiments.step8_analysis import read_campaign_file

    name = "private/audit-inventory.json"
    raw = facts.files.get(name)
    if raw is None:
        raw = read_campaign_file(facts.directory, name)
    inventory = CampaignAuditInventory.model_validate_json(raw)
    _require(
        (inventory.campaign_id, inventory.selection_hash)
        == (
            facts.manifest.campaign_id,
            facts.selection.content_hash,
        ),
        "inventory_selection_mismatch",
    )
    refs = {ref.artifact_id: ref for ref in inventory.artifacts}
    source_ref = refs["inventory-provenance"]
    _require(
        source_ref.sha256 == inventory.source_snapshot_sha256 and source_ref.scope == "evaluation",
        "inventory_provenance_unbound",
    )
    source = facts.store.load_json(source_ref.sha256)
    _require(
        source.get("schema_version") == "step8-database-audit-snapshot-v1"
        and source.get("campaign_id") == inventory.campaign_id
        and source.get("selection_hash") == inventory.selection_hash
        and source.get("source_role") == "evaluator"
        and source.get("isolation") == "repeatable_read_read_only"
        and source.get("database_snapshot", {}).get("database_role") == "pmmc_evaluator"
        and bool(source.get("database_snapshot", {}).get("snapshot")),
        "inventory_evaluator_snapshot_missing",
    )
    facts.files[name] = raw
    facts.input_hashes[name] = hashlib.sha256(raw).hexdigest()
    return refs


def summarize_campaign_oracles(facts) -> dict[str, Any]:
    """One declared cell per dataset, all selected queries, checksum-verified private facts."""
    declaration = facts.manifest.protocol.oracle_at_50
    if declaration is None:
        return oracle_not_evaluated()
    try:
        inventory_refs = _inventory_artifacts(facts)
    except (KeyError, OSError, ValueError):
        inventory_refs = {}
    selected_cells = [
        cell
        for cell in facts.cells
        if (
            cell.method,
            cell.answer_model,
            cell.treatment,
            cell.source_variant,
            cell.question_budget,
            cell.round_id,
        )
        == (
            "pmmc",
            declaration.answer_model,
            declaration.treatment,
            declaration.variant,
            declaration.question_budget,
            declaration.round_id,
        )
    ]
    _require(
        len(selected_cells) == len(facts.selection.query_ids)
        and {cell.dataset for cell in selected_cells} == set(facts.selection.query_ids),
        "declared_cell_matrix_incomplete",
    )
    records: dict[str, list[dict[str, Any]]] = {cell.cell_id: [] for cell in selected_cells}
    matrix = {cell.cell_id: cell for cell in selected_cells}
    for job_id in sorted(facts.verified_jobs):
        job = facts.jobs[job_id]
        if job.stage != "evaluation":
            continue
        output = facts.receipts[job_id].output
        report = output.get("oracle_at_50", {})
        if report.get("status") != "completed":
            _require(report.get("oracle_accuracy") is None, "incomplete_report_has_accuracy")
            continue
        cell_id = report["source_identity"]["cell_id"]
        _require(cell_id in matrix, "undeclared_cell")
        cell = matrix[cell_id]
        _require(
            output["campaign_id"] == facts.manifest.campaign_id == job.campaign_id
            and output["query_id"] == report["query_id"] == job.scientific_inputs["query_id"]
            and output["query_executor_run_id"] == report["query_run_id"]
            and output["evaluation_run_id"] == report["evaluation_run_id"]
            and job.scientific_inputs["dataset"] == cell.dataset
            and report["snapshot_id"] == facts.manifest.snapshot_ids[cell.dataset]
            and report["query_id"] in facts.selection.query_ids[cell.dataset]
            and report["runner_identity"]["protocol"]
            == {
                "campaign_id": facts.manifest.campaign_id,
                "protocol_hash": facts.manifest.protocol.content_hash,
                "declaration": declaration.model_dump(mode="json"),
                "cell_id": cell_id,
                "cost_scope": "research_only_evaluator_oracle",
            },
            "campaign_scope_mismatch",
        )
        proof_ref = inventory_refs.get("oracle-proof:" + report["content_hash"])
        if proof_ref is None or proof_ref.scope != "evaluation":
            continue
        try:
            bundle = facts.store.load_json(report["private_bundle_sha256"])
            proof = facts.store.load_json(proof_ref.sha256)
        except (OSError, ValueError):
            # No private execution facts in the verified inventory means no accuracy.
            continue
        call_ids = validate_oracle_bundle(report, bundle)
        _require(
            proof
            == {
                "report_hash": report["content_hash"],
                "bundle_sha256": report["private_bundle_sha256"],
                "model_call_ids": sorted(call_ids),
                "evaluation_run_id": report["evaluation_run_id"],
                "cost_scope": "research_only_evaluator_oracle",
            },
            "inventory_proof_mismatch",
        )
        records[cell_id].append(report)
    cells = []
    for cell in selected_cells:
        summary = summarize_oracle_reports(
            records[cell.cell_id], selected_query_ids=facts.selection.query_ids[cell.dataset]
        )
        missing = summary.pop("missing_query_ids")
        cells.append(
            {
                "cell_id": cell.cell_id,
                "dataset": cell.dataset.value,
                **summary,
                "missing_query_count": len(missing),
                "observed_query_count": summary["denominator"] - len(missing),
            }
        )
    return {
        "schema_version": "step8-oracle-analysis-v1",
        "metric": "Oracle@50",
        "status": "completed" if all(c["status"] == "completed" for c in cells) else "incomplete",
        "declaration": declaration.model_dump(mode="json"),
        "cells": cells,
        "cost_scope": "research_only_evaluator_oracle",
        "scientific_approval": False,
    }


async def materialize_oracle_inventory(builder, manifest, payload) -> None:
    """Call from the parent's private-payload collector with its existing fetch closure."""
    if manifest.protocol.oracle_at_50 is None:
        return
    from pmmc.infrastructure.database.experiment_models import (
        EvaluationArtifactModel,
        EvaluationModelCallModel,
    )

    snapshot = builder.snapshot
    calls = {row["model_call_id"]: row for row in snapshot.rows(EvaluationModelCallModel)}
    for artifact in snapshot.rows(EvaluationArtifactModel):
        if not artifact["artifact_id"].startswith(("step8-scores-", "step8-incomplete-")):
            continue
        try:
            ref = await payload("evaluation", artifact["artifact_id"], {"process_facts"})
            if ref is None:
                continue
            output = builder.store.load_json(builder.artifacts[ref].sha256)
            report = output.get("oracle_at_50", {})
            if report.get("status") not in {"completed", "incomplete", "failed"}:
                continue
            _require(output["campaign_id"] == manifest.campaign_id, "inventory_campaign_mismatch")
            owner = output["evaluation_run_id"], output["query_executor_run_id"], output["query_id"]
            for name, expected in zip(
                ("evaluation_run_id", "query_run_id", "query_id"), owner, strict=True
            ):
                _require(report.get(name, expected) == expected, "inventory_report_owner_mismatch")
            bundle = None
            private = report.get("private_artifact")
            if private is not None:
                ref = await payload(
                    "evaluation", private["artifact_id"], {"process_facts", "review_samples"}
                )
                if ref is not None:
                    bundle = builder.store.load_json(builder.artifacts[ref].sha256)
            # Failed generation/grounding may leave no answer.calls in the bundle.
            # Export the owner's full attempt ledger, not just successful answer refs.
            owned_calls = {
                identity: call
                for identity, call in calls.items()
                if call.get("evaluation_run_id") == owner[0]
            }
            complete_payloads = private is None or bundle is not None
            for call in owned_calls.values():
                _require(
                    (call["evaluation_run_id"], call["run_id"], call["query_id"]) == owner
                    and (
                        report.get("snapshot_id") is None
                        or call["snapshot_id"] == report["snapshot_id"]
                    ),
                    "inventory_call_owner_mismatch",
                )
                error = (call.get("metadata") or {}).get("error_artifact_id")
                raw_ids = [call.get("input_asset_id"), call.get("output_asset_id"), error]
                complete_payloads &= bool(raw_ids[0] and (raw_ids[1] or raw_ids[2]))
                for identity in dict.fromkeys(identity for identity in raw_ids if identity):
                    complete_payloads &= bool(
                        await payload("evaluation", identity, {"review_samples", "failures"})
                    )
            if bundle is not None:
                if not isinstance(bundle, dict):
                    raise ValueError("oracle_analysis_inventory_private_bundle_invalid")
                _validate_bundle_envelope(report, bundle)
            if report["status"] != "completed":
                _require(
                    report.get("oracle_accuracy") is None and report.get("oracle_score") is None,
                    "incomplete_report_has_accuracy",
                )
                if bundle is not None:
                    referenced_calls = {
                        call["model_call_id"]
                        for fact in bundle["private_facts"].values()
                        for call in (fact.get("answer") or {}).get("calls", [])
                    }
                    referenced_calls.update(
                        row["judge"]["model_call_id"]
                        for row in report.get("candidates", [])
                        if row.get("judge") is not None
                    )
                    complete_payloads &= referenced_calls <= owned_calls.keys()
                    for digest, fact in bundle["private_facts"].items():
                        _require(
                            hashlib.sha256(json_bytes(fact)).hexdigest() == digest,
                            "candidate_fact_checksum_mismatch",
                        )
                builder.fact(
                    "oracle-diagnostic:" + canonical_content_hash([owner, report]),
                    {
                        "status": report["status"],
                        "oracle_accuracy": None,
                        "bundle_sha256": report.get("private_bundle_sha256"),
                        "model_call_ids": sorted(owned_calls),
                        "evaluation_run_id": owner[0],
                        "raw_attempts_complete": complete_payloads,
                        "cost_scope": "research_only_evaluator_oracle",
                    },
                    "model_payload",
                    "evaluation",
                )
                if not complete_payloads:
                    snapshot.missing(
                        "oracle_attempt_payloads_incomplete", {"review_samples", "failures"}
                    )
                continue
            if not isinstance(bundle, dict) or not complete_payloads:
                raise ValueError("oracle_analysis_inventory_private_bundle_or_calls_missing")
            call_ids = validate_oracle_bundle(report, bundle)
            _require(
                call_ids <= owned_calls.keys()
                and all(owned_calls[identity]["status"] == "success" for identity in call_ids),
                "inventory_referenced_calls_incomplete",
            )
            builder.fact(
                "oracle-proof:" + report["content_hash"],
                {
                    "report_hash": report["content_hash"],
                    "bundle_sha256": report["private_bundle_sha256"],
                    "model_call_ids": sorted(call_ids),
                    "evaluation_run_id": report["evaluation_run_id"],
                    "cost_scope": "research_only_evaluator_oracle",
                },
                "model_payload",
                "evaluation",
            )
        except (KeyError, TypeError, ValueError, OSError):
            snapshot.missing("oracle_private_proof_incomplete", {"process_facts", "review_samples"})
