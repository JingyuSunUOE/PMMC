"""Evaluator-snapshot evidence for physical calls adopted through an R0 prefix."""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import json_bytes
from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_questioner_cost_dependencies import (
    _call_hash,
    _NoSourceIO,
    _sealed_json,
)
from pmmc.experiments.step8_upstream_import import (
    R0_ADOPTION_VERSION,
    R0_PREFIX_STAGES,
    R0PrefixImportProof,
    r0_prefix_compatibility,
)
from pmmc.schemas.step8 import BenchmarkCampaignManifest

R0_COST_DEPENDENCIES_VERSION = "step8-r0-prefix-cost-dependencies-v1"
R0_COST_DEPENDENCIES_ARTIFACT = "cost-dependencies:r0-prefix-imports"


def _proof_sha(proof):
    return hashlib.sha256(json_bytes(proof.model_dump(mode="json"))).hexdigest()


def validate_r0_source_receipts(facts, proof: R0PrefixImportProof) -> None:
    """Check committed DB identities, the recursive DAG, and exact proof bytes."""
    owner = facts["campaign"]
    campaign = BenchmarkCampaignManifest.model_validate(owner["payload"])
    scope = {
        k: proof.compatibility["questioner"][k]
        for k in ("snapshot_id", "dataset", "conversation_id")
    }
    scope["treatment"] = proof.compatibility["treatment"]
    if (
        owner["campaign_id"] != proof.source_campaign_id
        or campaign.campaign_id != proof.source_campaign_id
        or owner["protocol_hash"] != proof.source_protocol_hash
        or campaign.protocol.content_hash != proof.source_protocol_hash
        or r0_prefix_compatibility(campaign, scope) != proof.compatibility
    ):
        raise DependencyUnproven("r0_cost_source_campaign_mismatch")
    jobs, receipts = facts["jobs"], facts["receipts"]
    roots = facts["stage_job_ids"]
    if len(roots) != len(set(roots)):
        raise DependencyUnproven("r0_cost_source_stage_duplicate")
    checked, active, committed = set(), set(), {}

    def visit(identity):
        if identity in active:
            raise DependencyUnproven("r0_cost_receipt_cycle")
        if identity in checked:
            return
        active.add(identity)
        row = jobs[identity]
        definition = BenchmarkStageJob.model_validate(row["payload"])
        stored = receipts[row["receipt_hash"]]
        receipt = ComponentReceipt.model_validate(stored["payload"])
        if (
            row["job_id"] != identity
            or definition.job_id != identity
            or row["campaign_id"] != campaign.campaign_id
            or definition.campaign_id != campaign.campaign_id
            or definition.stage != row["stage"]
            or definition.source_fingerprint != row["source_fingerprint"]
            or row["source_fingerprint"] != campaign.protocol.source_fingerprints[row["stage"]]
            or row["status"] != "completed"
            or definition.scientific_input_hash != row["scientific_input_hash"]
            or definition.dependency_ids != row["dependency_ids"]
            or receipt.content_hash != row["receipt_hash"]
            or stored["receipt_hash"] != row["receipt_hash"]
            or stored["job_id"] != identity
            or receipt.job_id != identity
            or receipt.scientific_input_hash != row["scientific_input_hash"]
            or receipt.source_fingerprint != row["source_fingerprint"]
            or receipt.schema_revision != proof.schema_revision
            or set(receipt.upstream_receipt_hashes) != set(row["dependency_ids"])
            or stored["component_identity"] != receipt.component_identity
            or stored["artifact_hashes"] != receipt.artifact_hashes
        ):
            raise DependencyUnproven("r0_cost_source_receipt_identity_mismatch")
        if identity in roots:
            if (
                row["stage"] in committed
                or receipt.output.get("compiler_run_id") != proof.source_run_id
            ):
                raise DependencyUnproven("r0_cost_source_stage_owner_mismatch")
            committed[row["stage"]] = receipt
        for dependency, digest in receipt.upstream_receipt_hashes.items():
            if jobs[dependency]["receipt_hash"] != digest:
                raise DependencyUnproven("r0_cost_upstream_receipt_mismatch")
            visit(dependency)
        active.remove(identity)
        checked.add(identity)

    for identity in roots:
        visit(identity)
    if (
        set(committed) != set(R0_PREFIX_STAGES)
        or checked != set(jobs)
        or {r["receipt_hash"] for r in jobs.values()} != set(receipts)
    ):
        raise DependencyUnproven("r0_cost_source_receipt_denominator_mismatch")
    for stage, key, value in (
        ("r0_gate", "r0_prefix_import_proof_sha256", proof),
        ("questioner", "questioner_import_proof_sha256", proof.questioner_proof),
    ):
        digest = _proof_sha(value)
        if (
            committed[stage].output.get(key) != digest
            or digest not in committed[stage].artifact_hashes
        ):
            raise DependencyUnproven("r0_cost_source_proof_not_committed")


def validate_r0_cost_dependencies(facts: dict[str, Any]) -> dict[str, str]:
    from pmmc.infrastructure.database.step8_upstream_import import (
        project_r0_prefix_rows,
        r0_prefix_rows_hash,
        r0_prefix_table_contract_hash,
    )

    source, target = facts["source"], facts["target"]
    sh, th = source["headers"], target["headers"]
    sr, tr = sh["compiler"]["run_id"], th["compiler"]["run_id"]
    lineage = th["compiler"]["metadata"]["r0_prefix_upstream_import"]
    proof = R0PrefixImportProof.model_validate(lineage["proof_json"])
    if any(
        len(source["rows"][name]) != 1
        for name in (
            "r0_input_manifests",
            "qbank_index_manifests",
            "memory_operator_index_manifests",
        )
    ):
        raise DependencyUnproven("r0_cost_manifest_denominator_mismatch")
    if (
        sr == tr
        or proof.source_run_id != sr
        or lineage["version"] != R0_ADOPTION_VERSION
        or lineage["source_run_id"] != sr
        or lineage["source_campaign_id"] != proof.source_campaign_id
        or lineage["source_protocol_hash"] != proof.source_protocol_hash
        or lineage["proof_hash"] != proof.content_hash
        or lineage["source_proof_sha256"] != _proof_sha(proof)
        or lineage["source_facts_hash"] != proof.facts_hash
        or lineage["compatibility_hash"] != proof.compatibility_hash
        or lineage["imported_stages"] != list(R0_PREFIX_STAGES[1:])
        or type(lineage["provider_calls"]) is not int
        or lineage["provider_calls"] != 0
        or th["manifest"]["metadata"].get("r0_prefix_upstream_import") != lineage
        or sh["manifest"]["run_id"] != sr
        or th["manifest"]["run_id"] != tr
        or sh["compiler"]["snapshot_id"] != th["compiler"]["snapshot_id"]
        or sh["compiler"]["snapshot_id"] != proof.compatibility["questioner"]["snapshot_id"]
        or sh["compiler"].get("compiler_schema_version") != "0.9.0"
        or th["compiler"].get("compiler_schema_version") != "0.9.0"
        or lineage["target_selection_hash"] != th["manifest"]["selection_hash"]
        or proof.table_contract_hash != r0_prefix_table_contract_hash()
        or canonical_content_hash(source) != proof.facts_hash
        or canonical_content_hash(facts["source_questioner"]) != proof.questioner_proof.facts_hash
        or {k: len(v) for k, v in source["rows"].items()} != proof.row_counts
        or {r["artifact_id"]: r["sha256"] for r in source["rows"]["run_artifacts"]}
        != proof.artifact_hashes
        or source["rows"]["r0_input_manifests"][0]["content_hash"] != proof.r0_content_hash
    ):
        raise DependencyUnproven("r0_cost_graph_identity_mismatch")
    validate_r0_source_receipts(facts["source_receipts"], proof)
    projected, identities = project_r0_prefix_rows(
        source["rows"],
        facts["source_questioner"]["rows"],
        proof=proof,
        target_run_id=tr,
        target_selection_hash=lineage["target_selection_hash"],
    )
    if (
        r0_prefix_rows_hash(target["rows"]) != r0_prefix_rows_hash(projected)
        or r0_prefix_rows_hash(projected) != lineage["target_rows_hash"]
        or canonical_content_hash(identities) != lineage["identity_map_hash"]
    ):
        raise DependencyUnproven("r0_cost_import_projection_mismatch")
    calls = source["rows"]["model_calls"]
    by_id = {c["model_call_id"]: c for c in calls}
    if len(by_id) != len(calls) or any(
        c["run_id"] != sr
        or c.get("query_id") is not None
        or c.get("snapshot_id") != sh["compiler"]["snapshot_id"]
        or c["stage"] not in {"questioner", "planner", "embed"}
        or (c.get("parent_model_call_id") and c["parent_model_call_id"] not in by_id)
        for c in calls
    ):
        raise DependencyUnproven("r0_cost_producer_scope_mismatch")
    return {identities[c["model_call_id"]]: c["model_call_id"] for c in calls}


async def _collect_source_receipts(session, proof):
    from pmmc.infrastructure.database.step8_models import (
        BenchmarkCampaignModel,
        BenchmarkComponentReceiptModel,
        BenchmarkStageJobModel,
    )

    def record(row, fields):
        return {key: getattr(row, key) for key in fields}

    owner = await session.get(BenchmarkCampaignModel, proof.source_campaign_id)
    pairs = (
        await session.execute(
            select(BenchmarkStageJobModel, BenchmarkComponentReceiptModel)
            .join(
                BenchmarkComponentReceiptModel,
                BenchmarkStageJobModel.receipt_hash == BenchmarkComponentReceiptModel.receipt_hash,
            )
            .where(
                BenchmarkStageJobModel.campaign_id == proof.source_campaign_id,
                BenchmarkStageJobModel.stage.in_(R0_PREFIX_STAGES),
                BenchmarkStageJobModel.status == "completed",
                BenchmarkComponentReceiptModel.payload["output"]["compiler_run_id"].astext
                == proof.source_run_id,
            )
        )
    ).all()
    pending = {job.job_id: (job, receipt) for job, receipt in pairs}
    roots = [job.job_id for job, _ in pairs]
    jobs, receipts = {}, {}
    while pending:
        identity = min(pending)
        job, receipt = pending.pop(identity)
        if identity in jobs:
            continue
        jobs[identity] = record(
            job,
            (
                "job_id",
                "campaign_id",
                "stage",
                "source_fingerprint",
                "scientific_input_hash",
                "status",
                "dependency_ids",
                "receipt_hash",
                "payload",
            ),
        )
        receipts[receipt.receipt_hash] = record(
            receipt,
            (
                "receipt_hash",
                "job_id",
                "component_identity",
                "artifact_hashes",
                "payload",
            ),
        )
        for dependency in job.dependency_ids:
            if dependency not in jobs:
                parent_job = await session.get(BenchmarkStageJobModel, dependency)
                parent_receipt = await session.get(
                    BenchmarkComponentReceiptModel,
                    parent_job.receipt_hash,
                )
                pending[dependency] = parent_job, parent_receipt
    return {
        "campaign": record(owner, ("campaign_id", "protocol_hash", "payload")),
        "stage_job_ids": sorted(roots),
        "jobs": jobs,
        "receipts": receipts,
    }


async def collect_r0_prefix_cost_dependencies(
    session,
    *,
    compiler_run_ids,
    campaign_id: str,
    selection_hash: str,
    database_snapshot: dict,
) -> dict[str, Any]:
    """Borrow only the inventory's evaluator session; never call adoption or sealing."""
    from pmmc.infrastructure.database.compiler_models import CompilerRunModel
    from pmmc.infrastructure.database.step8_upstream_import import PostgresR0PrefixProofRepository

    forbidden: Any = _NoSourceIO()
    reader = PostgresR0PrefixProofRepository(forbidden, forbidden)
    pending, seen, runs = set(compiler_run_ids), set(), set()
    calls: set[str] = set()
    adopted: set[str] = set()
    entries = []
    while pending:
        target_id = min(pending)
        pending.remove(target_id)
        if target_id in seen:
            continue
        seen.add(target_id)
        facts, reason, source_id = {}, None, None
        try:
            async with session.begin_nested():
                compiler = await session.get(CompilerRunModel, target_id)
                lineage = compiler.metadata_json.get("r0_prefix_upstream_import")
                if lineage is None:
                    continue
                proof = R0PrefixImportProof.model_validate(lineage["proof_json"])
                source_id = proof.source_run_id
                sh, sr = await reader._read_prefix(session, source_id)
                qh, qr = await reader._read(session, source_id)
                th, tr = await reader._read_prefix(session, target_id, require_r0=False)
                facts = _sealed_json(
                    {
                        "source": {"headers": sh, "rows": sr},
                        "source_questioner": {"headers": qh, "rows": qr},
                        "target": {"headers": th, "rows": tr},
                        "source_receipts": await _collect_source_receipts(session, proof),
                    }
                )
                adopted.update(validate_r0_cost_dependencies(facts))
                calls.update(c["model_call_id"] for c in sr["model_calls"])
                runs.add(source_id)
                pending.add(source_id)
        except DependencyUnproven as exc:
            reason = str(exc)
        except (
            SQLAlchemyError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            RecursionError,
        ):
            reason = "r0_cost_source_unavailable_or_invalid"
        entries.append(
            {
                "target_run_id": target_id,
                "source_run_id": source_id,
                "status": "incomplete" if reason else "complete",
                "reasons": [reason] if reason else [],
                "facts": facts,
                "facts_hash": canonical_content_hash(facts),
            }
        )
    return {
        "schema_version": R0_COST_DEPENDENCIES_VERSION,
        "campaign_id": campaign_id,
        "selection_hash": selection_hash,
        "source_role": "evaluator",
        "database_snapshot": database_snapshot,
        "imports": entries,
        "dependency_run_ids": sorted(runs),
        "dependency_model_call_ids": sorted(calls),
        "adopted_model_call_ids": sorted(adopted),
    }


def r0_cost_alias_edges(snapshot, calls: dict[str, dict]) -> dict[str, str]:
    imported = {k for k, c in calls.items() if "r0_prefix_import" in (c.get("metadata") or {})}
    if not imported and R0_COST_DEPENDENCIES_ARTIFACT not in snapshot.artifacts:
        return {}
    manifest = snapshot.dependency_artifact(
        R0_COST_DEPENDENCIES_ARTIFACT, R0_COST_DEPENDENCIES_VERSION
    )
    aliases: dict[str, str] = {}
    for entry in manifest["imports"]:
        facts = entry["facts"]
        if (
            entry["status"] != "complete"
            or entry["reasons"]
            or canonical_content_hash(facts) != entry["facts_hash"]
            or facts["source"]["headers"]["compiler"]["run_id"] != entry["source_run_id"]
            or facts["target"]["headers"]["compiler"]["run_id"] != entry["target_run_id"]
        ):
            raise DependencyUnproven("r0_cost_graph_unproven")
        try:
            mapped = validate_r0_cost_dependencies(facts)
        except (IndexError, RecursionError):
            raise DependencyUnproven("r0_cost_graph_structure_invalid") from None
        for graph in (facts["source"], facts["target"]):
            for row in graph["rows"]["model_calls"]:
                if _call_hash(calls.get(row["model_call_id"])) != _call_hash(row):
                    raise DependencyUnproven("r0_cost_exported_call_mismatch")
        # Q retains its own proof domain and is verified by the Q collector.
        mapped = {k: v for k, v in mapped.items() if k in imported}
        if aliases.keys() & mapped.keys():
            raise DependencyUnproven("r0_cost_graph_duplicated")
        aliases.update(mapped)
    if aliases.keys() != imported:
        raise DependencyUnproven("r0_cost_alias_denominator_mismatch")
    return aliases
