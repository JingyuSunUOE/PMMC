"""Evaluator-snapshot cost lineage for combined Q/Planner0/index adoption.

Adoption is a namespace projection, not a provider dispatch. Validate the whole
projection before aliasing any call, including failed attempts and shared-index
producers. Historical Q-only adoptions retain their separate proof domain.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_model_identity import compiler_model_entry_hashes
from pmmc.experiments.step8_questioner_cost_dependencies import (
    _call_hash,
    _NoSourceIO,
    _sealed_json,
)
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, BenchmarkCampaignManifest

PARTIAL_COST_DEPENDENCIES_VERSION = "step8-partial-prefix-cost-dependencies-v1"
PARTIAL_COST_DEPENDENCIES_ARTIFACT = "cost-dependencies:partial-prefix-imports"
PARTIAL_PREFIX_STAGES = ("questioner", "planner0", "text_index", "multimodal_index")


def _unique_calls(rows):
    calls = {row["model_call_id"]: row for row in rows}
    if len(calls) != len(rows) or any(not isinstance(key, str) or not key for key in calls):
        raise DependencyUnproven("partial_cost_call_identity_invalid")
    return calls


def _validate_source_receipts(facts, proof):
    evidence = facts["source_receipts"]
    owner = evidence["campaign"]
    campaign = BenchmarkCampaignManifest.model_validate(owner["payload"])
    target = facts["target_campaign"]
    target_campaign = BenchmarkCampaignManifest.model_validate(target["payload"])
    if (
        owner["campaign_id"] != campaign.campaign_id
        or campaign.campaign_id != proof["source_campaign_id"]
        or owner["protocol_hash"] != campaign.protocol.content_hash
        or campaign.protocol.content_hash != proof["source_protocol_hash"]
        or target["campaign_id"] != target_campaign.campaign_id
        or target_campaign.campaign_id != proof["target_campaign_id"]
        or target["protocol_hash"] != target_campaign.protocol.content_hash
        or target_campaign.protocol.content_hash != proof["target_protocol_hash"]
        or campaign.campaign_id == target_campaign.campaign_id
        or not campaign.diagnostic_only
        or not target_campaign.diagnostic_only
        or campaign.paper_result
        or target_campaign.paper_result
        or proof["compatibility_hash"]
        != target_campaign.protocol.config_hashes.get("partial_import_policy")
    ):
        raise DependencyUnproven("partial_cost_campaign_identity_mismatch")
    before, after = campaign.protocol, target_campaign.protocol
    if (
        any(
            getattr(before, name) != getattr(after, name)
            for name in (
                "seed",
                "partition_hash",
                "exposure_manifest_hash",
                "questioner",
                "verifier",
                "planner",
                "compiler_alias",
            )
        )
        or any(
            before.config_hashes[name] != after.config_hashes[name]
            for name in ("questioner_config", "planner_config", "foundation_config")
        )
        or compiler_model_entry_hashes(before) != compiler_model_entry_hashes(after)
    ):
        raise DependencyUnproven("partial_cost_scientific_identity_mismatch")
    jobs, receipts, roots = evidence["jobs"], evidence["receipts"], evidence["stage_job_ids"]
    if len(roots) != len(set(roots)):
        raise DependencyUnproven("partial_cost_source_stage_duplicate")
    scope = proof["source_scope"]
    if set(scope) != {"dataset", "snapshot_id", "conversation_id", "treatment"}:
        raise DependencyUnproven("partial_cost_source_scope_invalid")
    checked, active, stages = set(), set(), {}

    def visit(identity):
        if identity in active:
            raise DependencyUnproven("partial_cost_receipt_cycle")
        if identity in checked:
            return
        active.add(identity)
        row = jobs[identity]
        job = BenchmarkStageJob.model_validate(row["payload"])
        stored = receipts[row["receipt_hash"]]
        receipt = ComponentReceipt.model_validate(stored["payload"])
        if (
            row["job_id"] != identity
            or job.job_id != identity
            or row["campaign_id"] != campaign.campaign_id
            or job.campaign_id != campaign.campaign_id
            or row["stage"] != job.stage
            or row["status"] != "completed"
            or row["source_fingerprint"] != job.source_fingerprint
            or job.source_fingerprint != campaign.protocol.source_fingerprints[job.stage]
            or row["scientific_input_hash"] != job.scientific_input_hash
            or row["dependency_ids"] != job.dependency_ids
            or receipt.job_id != identity
            or stored["job_id"] != identity
            or receipt.source_fingerprint != job.source_fingerprint
            or receipt.scientific_input_hash != job.scientific_input_hash
            or receipt.schema_revision != proof["schema_revision"]
            or receipt.content_hash != row["receipt_hash"]
            or stored["receipt_hash"] != receipt.content_hash
            or stored["component_identity"] != receipt.component_identity
            or stored["artifact_hashes"] != receipt.artifact_hashes
            or set(receipt.upstream_receipt_hashes) != set(job.dependency_ids)
        ):
            raise DependencyUnproven("partial_cost_source_receipt_identity_mismatch")
        if identity in roots:
            if (
                job.stage in stages
                or receipt.output.get("compiler_run_id") != proof["source_run_id"]
                or any(job.parameters.get(key) != value for key, value in scope.items())
            ):
                raise DependencyUnproven("partial_cost_source_stage_owner_mismatch")
            stages[job.stage] = receipt.content_hash
        for dependency, digest in receipt.upstream_receipt_hashes.items():
            if jobs[dependency]["receipt_hash"] != digest:
                raise DependencyUnproven("partial_cost_upstream_receipt_mismatch")
            visit(dependency)
        active.remove(identity)
        checked.add(identity)

    for identity in roots:
        visit(identity)
    if (
        set(stages) != set(PARTIAL_PREFIX_STAGES)
        or stages != proof["source_receipts"]
        or checked != set(jobs)
        or {row["receipt_hash"] for row in jobs.values()} != set(receipts)
    ):
        raise DependencyUnproven("partial_cost_source_receipt_denominator_mismatch")


def validate_partial_cost_dependencies(facts: dict[str, Any]) -> dict[str, str]:
    """Reproduce every imported FK projection before returning alias edges."""
    from pmmc.infrastructure.database.step8_partial_import import (
        _stable_headers,
        partial_prefix_facts_hash,
        partial_prefix_rows_hash,
        partial_prefix_table_contract_hash,
        project_partial_prefix_rows,
    )

    source, target = facts["source"], facts["target"]
    sh, th = source["headers"], target["headers"]
    sr, tr = sh["compiler"]["run_id"], th["compiler"]["run_id"]
    lineage = th["compiler"]["metadata"]["partial_prefix_upstream_import"]
    proof = lineage["proof_json"]
    if (
        sr == tr
        or proof["version"] != "step8-partial-prefix-proof-v1"
        or lineage["version"] != "step8-partial-prefix-adoption-v1"
        or proof["source_run_id"] != sr
        or lineage["source_run_id"] != sr
        or lineage["source_campaign_id"] != proof["source_campaign_id"]
        or lineage["source_protocol_hash"] != proof["source_protocol_hash"]
        or lineage["target_campaign_id"] != proof["target_campaign_id"]
        or lineage["target_protocol_hash"] != proof["target_protocol_hash"]
        or lineage["compatibility_hash"] != proof["compatibility_hash"]
        or lineage["proof_hash"] != canonical_content_hash(proof)
        or proof["schema_revision"] != STEP8_ALEMBIC_REVISION
        or proof["table_contract_hash"] != partial_prefix_table_contract_hash()
        or proof["covered_stages"] != list(PARTIAL_PREFIX_STAGES)
        or lineage["imported_stages"] != list(PARTIAL_PREFIX_STAGES)
        or type(lineage["provider_calls"]) is not int
        or lineage["provider_calls"] != 0
        or th["manifest"]["metadata"].get("partial_prefix_upstream_import") != lineage
        or sh["manifest"]["run_id"] != sr
        or th["manifest"]["run_id"] != tr
        or sh["compiler"]["snapshot_id"] != th["compiler"]["snapshot_id"]
        or sh["compiler"]["snapshot_id"] != proof["source_scope"]["snapshot_id"]
        or sh["compiler"]["compiler_schema_version"] != "0.9.0"
        or th["compiler"]["compiler_schema_version"] != "0.9.0"
        or partial_prefix_facts_hash(sh, source["rows"]) != lineage["source_facts_hash"]
        or lineage["source_facts_hash"] != proof["facts_hash"]
        or lineage["target_selection_hash"] != th["manifest"]["selection_hash"]
        or canonical_content_hash(
            _stable_headers({key: th[key] for key in ("compiler", "manifest")})
        )
        != lineage["target_registration_hash"]
        or sh["compiler"]["metadata"].get("questioner_upstream_import")
        != proof["source_questioner_upstream_import"]
        or {name: len(rows) for name, rows in source["rows"].items()} != proof["row_counts"]
        or {row["artifact_id"]: row["sha256"] for row in source["rows"]["run_artifacts"]}
        != proof["artifact_hashes"]
    ):
        raise DependencyUnproven("partial_cost_graph_identity_mismatch")
    _validate_source_receipts(facts, proof)
    projected, identities = project_partial_prefix_rows(
        source["rows"],
        proof=proof,
        target_run_id=tr,
        target_selection_hash=lineage["target_selection_hash"],
        memory_raw_image_policy=sh["manifest"]["metadata"]["questioner_config"].get(
            "memory_raw_image_policy", "allow"
        ),
    )
    if (
        partial_prefix_rows_hash(projected) != lineage["target_rows_hash"]
        or partial_prefix_rows_hash(target["rows"]) != lineage["target_rows_hash"]
        or canonical_content_hash(identities) != lineage["identity_map_hash"]
    ):
        raise DependencyUnproven("partial_cost_import_projection_mismatch")
    calls = _unique_calls(source["rows"]["model_calls"])
    target_calls = _unique_calls(target["rows"]["model_calls"])
    if not calls:
        raise DependencyUnproven("partial_cost_call_denominator_empty")
    aliases, measured = {}, {}
    for identity, call in calls.items():
        if (
            call["snapshot_id"] != sh["compiler"]["snapshot_id"]
            or call.get("query_id") is not None
            or call.get("plan_id") is not None
            or call["stage"] not in {"questioner", "planner", "embed"}
            or (call["run_id"] != sr and call["stage"] != "embed")
            or type(call.get("cache_hit")) is not bool
            or call["status"] not in {"success", "failed"}
            or type(call.get("retry_count")) is not int
            or call["retry_count"] < 0
        ):
            raise DependencyUnproven("partial_cost_producer_scope_mismatch")
        current, seen = identity, set()
        while current:
            if current in seen or current not in calls:
                raise DependencyUnproven("partial_cost_parent_call_unproven")
            seen.add(current)
            current = calls[current].get("parent_model_call_id")
        adopted_id = identities[identity]
        adopted = target_calls[adopted_id]
        if (
            adopted["run_id"] != tr
            or adopted.get("cache_hit") is not True
            or "partial_prefix_import" not in (adopted.get("metadata") or {})
            or {"upstream_import", "r0_prefix_import"} & (adopted.get("metadata") or {}).keys()
        ):
            raise DependencyUnproven("partial_cost_adopted_call_domain_mismatch")
        aliases[adopted_id] = identity
        measured[adopted_id] = {
            "source_run_id": call["run_id"],
            "source_model_call_id": identity,
            "source_row_hash": canonical_content_hash(call),
            "source_cache_hit": call["cache_hit"],
        }
    if set(aliases) != set(target_calls) or lineage["model_call_aliases"] != measured:
        raise DependencyUnproven("partial_cost_alias_denominator_mismatch")
    return aliases


async def _collect_source_receipts(session, proof):
    from pmmc.infrastructure.database.step8_models import (
        BenchmarkCampaignModel,
        BenchmarkComponentReceiptModel,
        BenchmarkStageJobModel,
    )

    def record(row, fields):
        return {key: getattr(row, key) for key in fields}

    owner = await session.get(BenchmarkCampaignModel, proof["source_campaign_id"])
    target = await session.get(BenchmarkCampaignModel, proof["target_campaign_id"])
    pairs = (
        await session.execute(
            select(BenchmarkStageJobModel, BenchmarkComponentReceiptModel)
            .join(
                BenchmarkComponentReceiptModel,
                BenchmarkStageJobModel.receipt_hash == BenchmarkComponentReceiptModel.receipt_hash,
            )
            .where(
                BenchmarkStageJobModel.campaign_id == proof["source_campaign_id"],
                BenchmarkStageJobModel.stage.in_(PARTIAL_PREFIX_STAGES),
                BenchmarkStageJobModel.status == "completed",
                BenchmarkComponentReceiptModel.payload["output"]["compiler_run_id"].astext
                == proof["source_run_id"],
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
            receipt, ("receipt_hash", "job_id", "component_identity", "artifact_hashes", "payload")
        )
        for dependency in job.dependency_ids:
            if dependency not in jobs:
                parent = await session.get(BenchmarkStageJobModel, dependency)
                pending[dependency] = (
                    parent,
                    await session.get(BenchmarkComponentReceiptModel, parent.receipt_hash),
                )
    return {
        "source_receipts": {
            "campaign": record(owner, ("campaign_id", "protocol_hash", "payload")),
            "stage_job_ids": sorted(roots),
            "jobs": jobs,
            "receipts": receipts,
        },
        "target_campaign": record(target, ("campaign_id", "protocol_hash", "payload")),
    }


async def verify_partial_source_receipts(session, proof: dict[str, Any]) -> None:
    """Verify frozen donor receipts using the caller's read-only session only."""
    evidence = _sealed_json(await _collect_source_receipts(session, proof))
    _validate_source_receipts(evidence, proof)


async def collect_partial_prefix_cost_dependencies(
    session, *, compiler_run_ids, campaign_id: str, selection_hash: str, database_snapshot: dict
) -> dict[str, Any]:
    """Read only through the supplied evaluator transaction; never seal or adopt."""
    from pmmc.infrastructure.database.compiler_models import CompilerRunModel
    from pmmc.infrastructure.database.step8_partial_import import (
        PostgresPartialPrefixImportRepository,
    )

    forbidden: Any = _NoSourceIO()
    reader = PostgresPartialPrefixImportRepository(forbidden, forbidden, validate_proof=forbidden)
    pending, seen = set(compiler_run_ids), set()
    required = set(pending)
    entries: list[dict[str, Any]] = []
    calls: set[str] = set()
    runs: set[str] = set()
    adopted: set[str] = set()
    compiler_ids: set[str] = set()
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
                if compiler is None:
                    if target_id in required:
                        raise DependencyUnproven("partial_cost_compiler_missing")
                    continue
                compiler_ids.add(target_id)
                lineage = compiler.metadata_json.get("partial_prefix_upstream_import")
                if lineage is None:
                    historical = compiler.metadata_json.get("questioner_upstream_import")
                    if historical is not None:
                        pending.add(historical["source_run_id"])
                    continue
                source_id = lineage["source_run_id"]
                if not isinstance(source_id, str) or not source_id or source_id == target_id:
                    raise DependencyUnproven("partial_cost_source_identity_invalid")
                sh, sr = await reader._read_prefix(session, source_id)
                th, tr = await reader._read_prefix(session, target_id)
                facts = _sealed_json(
                    {
                        "source": {"headers": sh, "rows": sr},
                        "target": {"headers": th, "rows": tr},
                        **await _collect_source_receipts(session, lineage["proof_json"]),
                    }
                )
                mapped = validate_partial_cost_dependencies(facts)
                adopted.update(mapped)
                calls.update(row["model_call_id"] for row in sr["model_calls"])
                owners = {row["run_id"] for row in sr["model_calls"]} | {source_id}
                runs.update(owners)
                pending.update(owners)
                required.add(source_id)
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
            ImportError,
        ):
            reason = "partial_cost_source_unavailable_or_invalid"
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
        "schema_version": PARTIAL_COST_DEPENDENCIES_VERSION,
        "campaign_id": campaign_id,
        "selection_hash": selection_hash,
        "source_role": "evaluator",
        "database_snapshot": database_snapshot,
        "imports": entries,
        "dependency_run_ids": sorted(runs),
        "dependency_model_call_ids": sorted(calls),
        "dependency_compiler_run_ids": sorted(compiler_ids),
        "adopted_model_call_ids": sorted(adopted),
    }


def partial_cost_alias_edges(snapshot, calls: dict[str, dict]) -> dict[str, str]:
    imported = {
        key for key, row in calls.items() if "partial_prefix_import" in (row.get("metadata") or {})
    }
    if not imported and PARTIAL_COST_DEPENDENCIES_ARTIFACT not in snapshot.artifacts:
        return {}
    manifest = snapshot.dependency_artifact(
        PARTIAL_COST_DEPENDENCIES_ARTIFACT, PARTIAL_COST_DEPENDENCIES_VERSION
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
            raise DependencyUnproven("partial_cost_graph_unproven")
        try:
            mapped = validate_partial_cost_dependencies(facts)
        except (IndexError, RecursionError, ImportError):
            raise DependencyUnproven("partial_cost_graph_structure_invalid") from None
        for graph in (facts["source"], facts["target"]):
            for row in graph["rows"]["model_calls"]:
                if _call_hash(calls.get(row["model_call_id"])) != _call_hash(row):
                    raise DependencyUnproven("partial_cost_exported_call_mismatch")
        if aliases.keys() & mapped.keys():
            raise DependencyUnproven("partial_cost_graph_duplicated")
        aliases.update(mapped)
    if aliases.keys() != imported:
        raise DependencyUnproven("partial_cost_alias_denominator_mismatch")
    return aliases
