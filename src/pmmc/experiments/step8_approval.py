"""Derive development approval evidence from checked cohorts, never user booleans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_analysis import (
    _collect_scores,
    _observation,
    _verify_threshold_v2,
    load_campaign_facts,
    read_campaign_file,
    threshold_proof_hashes,
)
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, SnapshotArtifactReader
from pmmc.experiments.step8_audit import CampaignAuditInventory, load_process_facts
from pmmc.experiments.step8_process_analysis import (
    require_complete_process_metrics,
    summarize_process_metrics,
)
from pmmc.experiments.step8_selection import ExposureManifest
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.step8 import (
    CampaignPhase,
    ScientificProtocol,
    SelectionExpansionManifest,
    ThresholdApprovalEvidence,
    build_experiment_matrix,
)

_POLICY_FIELDS = (
    "match_threshold",
    "gap_threshold",
    "no_reranker_match_threshold",
    "no_reranker_gap_threshold",
    "text_relevance_threshold",
    "multimodal_relevance_threshold",
)


def _complete_score(score) -> bool:
    return (
        isinstance(score, dict)
        and score.get("status") in {"answered", "abstention"}
        and not score.get("engineering_error")
        and not score.get("unavailable_reason")
    )


def _selected_scores(facts, protocol, inventory, snapshot, reader: SnapshotArtifactReader):
    """Require the selected grid point, not completion of unrelated exploratory points."""
    payload = read_campaign_file(
        facts.directory, f"private/threshold-facts/{facts.selection.content_hash}.json"
    )
    digest = hashlib.sha256(payload).hexdigest()
    inventoried = {a.sha256 for a in inventory.artifacts}
    if (
        digest
        not in {
            a.sha256
            for a in inventory.artifacts
            if a.scope == "evaluation" and a.kind == "variant_cell_reuse"
        }
        or reader.read_bytes(digest) != payload
    ):
        raise ValueError("approval_threshold_proof_not_in_fresh_inventory")
    proof = json.loads(payload)
    if (
        proof.get("schema_version") != "step8-threshold-facts-v2"
        or proof.get("campaign_id") != facts.manifest.campaign_id
        or proof.get("selection_hash") != facts.selection.content_hash
        or proof.get("phase") != "development"
        or proof.get("source_role") != "evaluator"
        or proof.get("database_snapshot") != snapshot["database_snapshot"]
        or proof.get("scientific_approval") is not False
    ):
        raise ValueError("approval_threshold_proof_identity_mismatch")

    def keyed(rows):
        result = {}
        for row in rows:
            key = row["cell_id"], row["query_id"], row["axis"], tuple(row["thresholds"])
            if key in result:
                raise ValueError("approval_duplicate_threshold_proof")
            result[key] = row
        return result

    rows, coverage = keyed(proof["rows"]), keyed(proof["coverage"])
    selected = {}
    used = []
    changed_relevance = any(
        getattr(protocol, name) != getattr(facts.manifest.protocol, name)
        for name in ("text_relevance_threshold", "multimodal_relevance_threshold")
    )
    for cell in facts.cells:
        if cell.method != "pmmc" or cell.source_variant == "forced_top1":
            continue
        thresholds: tuple[float, ...] = (
            (protocol.no_reranker_match_threshold, protocol.no_reranker_gap_threshold)
            if cell.treatment.value == "no_reranker"
            else (protocol.match_threshold, protocol.gap_threshold)
        )
        axis = "approved_policy" if changed_relevance else "match_gap"
        if changed_relevance:
            thresholds = (
                *thresholds,
                protocol.text_relevance_threshold,
                protocol.multimodal_relevance_threshold,
            )
        for query_id in facts.selection.query_ids[cell.dataset]:
            key = cell.cell_id, query_id, axis, thresholds
            if key not in rows or coverage.get(key, {}).get("status") != "complete":
                raise ValueError("approval_selected_threshold_incomplete")
            saved = rows[key]
            context = reader.load_json(saved["context_sha256"])
            if (
                not set(threshold_proof_hashes(saved)) <= inventoried
                or not set(threshold_proof_hashes(context)) <= inventoried
            ):
                raise ValueError("approval_threshold_dependencies_not_in_fresh_inventory")
            score = saved["score"]
            expected = {
                "schema_version": (
                    "step8-threshold-context-v3"
                    if changed_relevance
                    else "step8-threshold-context-v2"
                ),
                "campaign_id": facts.manifest.campaign_id,
                "cell_id": cell.cell_id,
                "query_id": query_id,
                "axis": axis,
                "thresholds": list(thresholds),
                "typed_answer_id": score.get("typed_answer_id"),
                "question_budget": cell.question_budget,
                "round_id": cell.round_id,
                "answer_model": cell.answer_model,
                "treatment": cell.treatment.value,
                "fallback_enabled": cell.source_variant != "no_fallback",
                "selected_answer_method": score.get("answer_source_method"),
                "relevance_policy": (
                    "generated_evidence_subset"
                    if changed_relevance
                    else "unchanged_saved_evidence_only"
                ),
                "scientific_approval": False,
            }
            if any(context.get(k) != v for k, v in expected.items()):
                raise ValueError("approval_selected_threshold_context_mismatch")
            route = context.get("selected_route", {})
            if (route.get("match_threshold"), route.get("gap_threshold")) != thresholds[:2]:
                raise ValueError("approval_selected_threshold_route_mismatch")
            if changed_relevance and context.get("relevance_thresholds") != list(thresholds[2:]):
                raise ValueError("approval_selected_relevance_context_mismatch")
            _verify_threshold_v2(reader, saved, context)
            if not _complete_score(score):
                raise ValueError("approval_selected_threshold_evaluation_incomplete")
            _observation(query_id, cell, score)
            if saved.get("answer_sha256") is not None:
                answer = TypedAnswerResult.model_validate(reader.load_json(saved["answer_sha256"]))
                answer_proof = context.get("answer_proof") or {}
                trace_ref = answer_proof.get("calls", {}).get(answer.answer_model_call_id, {})
                if not trace_ref.get("trace_sha256"):
                    raise ValueError("approval_selected_threshold_answer_call_missing")
                trace = reader.load_json(trace_ref["trace_sha256"])
                if (
                    answer.typed_answer_id != score.get("typed_answer_id")
                    or answer.query_id != query_id
                    or answer.snapshot_id != facts.manifest.snapshot_ids[cell.dataset]
                    or (
                        trace.get("model_alias"),
                        trace.get("run_id"),
                        trace.get("query_id"),
                        trace.get("model_call_id"),
                        trace.get("status"),
                        trace.get("actual_model_id"),
                    )
                    != (
                        cell.answer_model,
                        answer.run_id,
                        query_id,
                        answer.answer_model_call_id,
                        "success",
                        answer.actual_model_id,
                    )
                    or answer.method != score.get("answer_source_method")
                    or answer.answer_variant != score.get("answer_source_variant")
                    or answer.terminal_state != score.get("terminal_state")
                    or answer_proof.get("typed_answer_sha256") != saved["answer_sha256"]
                ):
                    raise ValueError("approval_selected_threshold_answer_mismatch")
            selected[cell.cell_id, query_id] = score
            used.append(saved["context_sha256"])
    return selected, digest, sorted(used)


def approval_evidence(
    directory: Path,
    store: CampaignArtifactStore,
    *,
    fresh_inventory: CampaignAuditInventory | None = None,
    match_threshold: float | None = None,
    gap_threshold: float | None = None,
    no_reranker_match_threshold: float | None = None,
    no_reranker_gap_threshold: float | None = None,
    text_relevance_threshold: float | None = None,
    multimodal_relevance_threshold: float | None = None,
) -> dict:
    reader = SnapshotArtifactReader(store)
    cohorts = {}
    for path in sorted((directory / "manifests/selections").glob("*.json")):
        selected = SelectionExpansionManifest.model_validate_json(
            read_campaign_file(directory, str(path.relative_to(directory)))
        )
        if selected.phase is not CampaignPhase.DEVELOPMENT or selected.selection in cohorts:
            raise ValueError("approval_requires_unambiguous_development_cohorts")
        cohorts[selected.selection] = selected
    if set(cohorts) != {"regression", "confirmation"}:
        raise ValueError("approval_requires_regression_and_confirmation")
    regression, confirmation = cohorts["regression"], cohorts["confirmation"]
    if confirmation.parent_selection_hash != regression.content_hash or any(
        set(ids) & set(confirmation.query_ids[dataset])
        for dataset, ids in regression.query_ids.items()
    ):
        raise ValueError("confirmation_must_be_disjoint_and_follow_regression")
    facts_by_cohort = {}
    original_scores = {}
    provisional = None
    for name, selection in cohorts.items():
        facts = load_campaign_facts(
            directory, selection_hash=selection.content_hash, artifact_store=store
        )
        if len(facts.verified_jobs) != len(facts.jobs) or facts.missing_artifact_count:
            raise ValueError("approval_component_or_artifact_missing")
        if provisional is not None and provisional != facts.manifest.protocol:
            raise ValueError("approval_protocol_changed_between_cohorts")
        provisional = facts.manifest.protocol
        if {c.cell_id for c in facts.cells} != {
            c.cell_id for c in build_experiment_matrix(provisional)
        }:
            raise ValueError("approval_matrix_incomplete")
        _, scores = _collect_scores(facts)
        process_report = summarize_process_metrics(facts)
        require_complete_process_metrics(process_report)
        from pmmc.experiments.step8_semantic_analysis import summarize_campaign_semantics

        if summarize_campaign_semantics(facts)["status"] != "complete":
            raise ValueError("approval_semantic_denominator_incomplete")
        for cell in facts.cells:
            for query_id in selection.query_ids[cell.dataset]:
                score = scores.get((cell.cell_id, query_id))
                if not _complete_score(score):
                    raise ValueError("approval_evaluation_denominator_incomplete")
        facts_by_cohort[name] = facts
        original_scores[name] = scores
    assert provisional is not None
    updates = {
        "match_threshold": match_threshold,
        "gap_threshold": gap_threshold,
        "no_reranker_match_threshold": no_reranker_match_threshold,
        "no_reranker_gap_threshold": no_reranker_gap_threshold,
        "text_relevance_threshold": text_relevance_threshold,
        "multimodal_relevance_threshold": multimodal_relevance_threshold,
    }
    protocol = ScientificProtocol.model_validate(
        {
            **provisional.model_dump(),
            **{k: v for k, v in updates.items() if v is not None},
        }
    )
    selected_policy = {k: getattr(protocol, k) for k in _POLICY_FIELDS}
    inventory = CampaignAuditInventory.model_validate_json(
        read_campaign_file(directory, "private/audit-inventory.json")
    )
    if fresh_inventory is None or inventory != fresh_inventory:
        raise ValueError("approval_requires_fresh_authoritative_inventory")
    if (
        not inventory.complete
        or inventory.selection_hash != confirmation.content_hash
        or inventory.campaign_id != confirmation.campaign_id
    ):
        raise ValueError("approval_audit_inventory_incomplete")
    required = {
        "reranker_calls",
        "failures",
        "review_samples",
        "attempts",
        "runtime_answers",
        "compiler_patches",
        "variant_cell_reuse",
        "cost_facts",
        "performance",
        "gold",
        "process_facts",
    }
    if set(inventory.coverage) != required:
        raise ValueError("approval_audit_coverage_incomplete")
    actual_records = {(record.kind, record.record_id) for record in inventory.records}
    expected_records = {
        (kind, identity) for kind, identity in inventory.expected_records if kind != "attempt"
    }
    if actual_records != expected_records:
        raise ValueError("approval_audit_records_incomplete")
    local_attempts = {path.name for path in (directory / "attempts").iterdir() if path.is_dir()}
    if (
        local_attempts
        != {identity for kind, identity in inventory.expected_records if kind == "attempt"}
        or not local_attempts
    ):
        raise ValueError("approval_attempt_inventory_incomplete")
    for artifact in inventory.artifacts:
        if reader.verified_size(artifact.sha256) != artifact.size_bytes:
            raise ValueError("approval_audit_artifact_size_mismatch")
    snapshot = reader.load_json(inventory.source_snapshot_sha256)
    if (
        snapshot.get("campaign_id") != confirmation.campaign_id
        or snapshot.get("selection_hash") != confirmation.content_hash
        or snapshot.get("source_role") != "evaluator"
        or snapshot.get("isolation") != "repeatable_read_read_only"
        or not snapshot.get("database_snapshot")
        or snapshot.get("missing")
    ):
        raise ValueError("approval_audit_snapshot_identity_mismatch")
    load_process_facts(inventory, store, require_complete=True)
    exposure = ExposureManifest.model_validate_json(
        read_campaign_file(directory, "manifests/exposure.json")
    )
    if (
        not exposure.complete
        or exposure.content_hash != provisional.exposure_manifest_hash
        or exposure.partition_hash != provisional.partition_hash
    ):
        raise ValueError("approval_exposure_history_incomplete")
    receipts = {}
    receipt_payloads = {}
    proof_hashes = {}
    for name, facts in facts_by_cohort.items():
        selected_scores, proof_hash, contexts = _selected_scores(
            facts, protocol, inventory, snapshot, reader
        )
        viable = {dataset: 0 for dataset in facts.selection.query_ids}
        for cell in facts.cells:
            for query_id in facts.selection.query_ids[cell.dataset]:
                score = selected_scores.get(
                    (cell.cell_id, query_id), original_scores[name][cell.cell_id, query_id]
                )
                if (
                    cell.method == "pmmc"
                    and cell.answer_model == "answer-qwen3.5-9b"
                    and cell.treatment.value == "main"
                    and cell.question_budget == 8
                    and cell.round_id == 2
                    and cell.source_variant == "active"
                    and score.get("terminal_state") == "answered"
                    and score.get("answer_source_method") == "pmmc"
                ):
                    viable[cell.dataset] += 1
        if not all(viable.values()):
            raise ValueError("approval_real_pmmc_path_missing_at_selected_threshold")
        proof_hashes[name] = proof_hash
        receipt_payloads[name] = {
            "campaign_id": facts.manifest.campaign_id,
            "provisional_protocol_hash": provisional.content_hash,
            "selected_protocol_hash": protocol.content_hash,
            "selected_policy": selected_policy,
            "selection": facts.selection.content_hash,
            "threshold_proof_sha256": proof_hash,
            "selected_contexts": contexts,
            "components": {
                key: facts.receipts[key].content_hash for key in sorted(facts.verified_jobs)
            },
            "viable": {key.value: value for key, value in viable.items()},
        }
        receipts[name] = canonical_content_hash(receipt_payloads[name])
    inventory_sha = store.put_json(inventory.model_dump(mode="json"))
    threshold_approval = ThresholdApprovalEvidence(
        schema_version=(
            "step8-threshold-approval-v2"
            if any(
                getattr(protocol, k) != getattr(provisional, k)
                for k in ("text_relevance_threshold", "multimodal_relevance_threshold")
            )
            else "step8-threshold-approval-v1"
        ),
        source_campaign_id=confirmation.campaign_id,
        source_campaign_manifest_hash=canonical_content_hash(
            facts.manifest.model_dump(mode="json")
        ),
        provisional_protocol=provisional,
        selected_policy=selected_policy,
        cohort_selection_hashes={name: s.content_hash for name, s in cohorts.items()},
        cohort_proof_sha256=proof_hashes,
        cohort_receipt_hashes=receipts,
        audit_inventory_sha256=inventory_sha,
    )
    return {
        "protocol_hash": provisional.content_hash,
        "selected_protocol": protocol,
        "threshold_approval": threshold_approval,
        "cohort_receipts": receipt_payloads,
        "regression_receipt_hash": receipts["regression"],
        "confirmation_receipt_hash": receipts["confirmation"],
        "protocol_integrity_passed": True,
        "scientific_viability_passed": True,
        "exposure_complete": True,
    }
