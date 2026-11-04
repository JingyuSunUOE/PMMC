"""Complete PMMC bank and explicit CPU-execution cost dependency evidence.

All reads borrow the evaluator transaction. A completed compiler run supplies
the whole call denominator, not just producers of accepted programs. Execution
proofs use the concrete CPU-only operator contract and stored vector bindings;
plan IDs are never used to guess which model calls an execution consumed.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Table, or_, select
from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.config import OPERATOR_SET_V7
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import json_bytes, sha256_file
from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
from pmmc.experiments.step8_questioner_cost_dependencies import _call_hash, _sealed_json

PMMC_COST_DEPENDENCIES_VERSION = "step8-pmmc-cost-dependencies-v1"
PMMC_COST_DEPENDENCIES_ARTIFACT = "cost-dependencies:pmmc"


def kernel_contract():
    from pmmc.compiler.config import compiler_config_hash, load_compiler_config
    from pmmc.infrastructure.settings import get_settings
    from pmmc.retrieval.operators import build_operator_registry

    root = Path(__file__).resolve().parents[1]
    config = load_compiler_config(get_settings().project_root / "configs/compiler/pmmc-v7.yaml")
    paths = (
        "memory/operators.py",
        "retrieval/operators.py",
        "runtime/query_backend.py",
        "compiler/program_runner.py",
        "infrastructure/database/compiler_repositories.py",
    )
    return {
        "version": "pmmc-persisted-vector-cpu-kernel-v1",
        "operators": list(OPERATOR_SET_V7),
        "provider_calls": 0,
        "operator_registry_hash": build_operator_registry(cast(Any, None), config).resolved_hash,
        "compiler_config_hash": compiler_config_hash(config),
        "source_hashes": {p: sha256_file(root / p) for p in paths},
    }


def _kernel_source(stage):
    from pmmc.experiments.step8_jobs import scientific_source_fingerprint
    from pmmc.infrastructure.settings import get_settings

    return scientific_source_fingerprint(get_settings().project_root, stage)


def _verify_compiler_references(lineage):
    from pmmc.experiments.release_compiler_reuse import verify_reference_lineage

    try:
        verify_reference_lineage(lineage)
    except (KeyError, TypeError, ValueError) as exc:
        raise DependencyUnproven("pmmc_compiler_reference_unproven") from exc


async def _kernel_receipt_lineage(read, compiler_run, query_run=None):
    from pmmc.infrastructure.database import step8_models as sm

    output = sm.BenchmarkComponentReceiptModel.payload["output"]
    predicates = [output["compiler_run_id"].as_string() == compiler_run]
    if query_run is not None:
        predicates.append(output["query_executor_run_id"].as_string() == query_run)
    receipts = await read(sm.BenchmarkComponentReceiptModel, or_(*predicates))
    by_job = _unique(receipts, "job_id")
    jobs = await read(
        sm.BenchmarkStageJobModel, sm.BenchmarkStageJobModel.job_id.in_(tuple(by_job))
    )
    if {j["job_id"] for j in jobs} != set(by_job):
        raise DependencyUnproven("pmmc_kernel_receipt_owner_missing")
    campaigns = _unique(
        await read(
            sm.BenchmarkCampaignModel,
            sm.BenchmarkCampaignModel.campaign_id.in_({j["campaign_id"] for j in jobs}),
        ),
        "campaign_id",
    )
    return [
        {"campaign": campaigns[j["campaign_id"]], "job": j, "receipt": by_job[j["job_id"]]}
        for j in jobs
    ]


def _validate_kernel_lineage(lineage, trace, program):
    from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
    from pmmc.schemas.step8 import BenchmarkCampaignManifest

    stage = (
        "execution"
        if trace["purpose"] == "query_time"
        else "dry_run0"
        if program["round_id"] == 0
        else "doubter"
    )
    owner_key = "query_executor_run_id" if stage == "execution" else "compiler_run_id"
    owner_id = trace["run_id"] if stage == "execution" else program["run_id"]
    roots = []
    _verify_compiler_references(lineage)
    for entry in lineage:
        owner, row, stored = entry["campaign"], entry["job"], entry["receipt"]
        campaign = BenchmarkCampaignManifest.model_validate(owner["payload"])
        job = BenchmarkStageJob.model_validate(row["payload"])
        receipt = ComponentReceipt.model_validate(stored["payload"])
        if job.stage != stage or receipt.output.get(owner_key) != owner_id:
            continue
        if (
            owner["campaign_id"] != campaign.campaign_id
            or owner["protocol_hash"] != campaign.protocol.content_hash
            or row["campaign_id"] != campaign.campaign_id
            or job.campaign_id != campaign.campaign_id
            or row["status"] != "completed"
            or row["job_id"] != job.job_id
            or receipt.job_id != job.job_id
            or stored["job_id"] != job.job_id
            or row["stage"] != stage
            or row["source_fingerprint"] != job.source_fingerprint
            or receipt.source_fingerprint != job.source_fingerprint
            or job.source_fingerprint != campaign.protocol.source_fingerprints[stage]
            or row["scientific_input_hash"] != job.scientific_input_hash
            or receipt.scientific_input_hash != job.scientific_input_hash
            or row["dependency_ids"] != job.dependency_ids
            or set(receipt.upstream_receipt_hashes) != set(job.dependency_ids)
            or row["receipt_hash"] != receipt.content_hash
            or stored["receipt_hash"] != receipt.content_hash
            or stored["component_identity"] != receipt.component_identity
            or stored["artifact_hashes"] != receipt.artifact_hashes
        ):
            raise DependencyUnproven("pmmc_execution_producer_kernel_unproven")
        if "compiler_producer_reference" in receipt.output:
            # Already checked against the unique committed physical chain above.
            # A consumption receipt contributes no second production or model call.
            continue
        actual_source = _kernel_source(stage)
        if job.source_fingerprint != actual_source:
            # A later approved patch retains exact receipts, including their older proofs.
            from pmmc.experiments.step8_compile_job_maintenance import retained_producer

            if retained_producer(job, receipt, campaign, actual_source):
                roots.append(entry)
                continue
        if job.source_fingerprint != actual_source or "maintenance_execution" in receipt.output:
            # Maintenance authorizes execution source only, never rewritten producer facts.
            try:
                maintenance = receipt.output.get("maintenance_execution")
                if (
                    not isinstance(maintenance, dict)
                    or set(maintenance) != {"proof", "sha256"}
                    or not isinstance(maintenance["proof"], dict)
                ):
                    raise ValueError("maintenance_execution_proof_not_bound")
                proof = maintenance["proof"]
                digest = hashlib.sha256(json_bytes(proof)).hexdigest()
                if maintenance["sha256"] != digest or digest not in receipt.artifact_hashes:
                    raise ValueError("maintenance_execution_proof_not_bound")
                from pmmc.experiments.step8_maintenance import validate_maintenance_execution

                validate_maintenance_execution(
                    proof,
                    campaign_id=campaign.campaign_id,
                    protocol_hash=campaign.protocol.content_hash,
                    stage=stage,
                    frozen_source_fingerprint=job.source_fingerprint,
                    actual_source_fingerprint=actual_source,
                )
            except (ImportError, KeyError, TypeError, ValueError):
                raise DependencyUnproven("pmmc_execution_producer_kernel_unproven") from None
        roots.append(entry)
    if len(roots) != 1:
        raise DependencyUnproven("pmmc_execution_producer_receipt_missing")


def _unique(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise DependencyUnproven("pmmc_cost_duplicate_identity")
    return result


def _one(rows):
    if len(rows) != 1:
        raise DependencyUnproven("pmmc_cost_singleton_denominator_invalid")
    return rows[0]


def _call_graph(calls, snapshot):
    by_id = _unique(calls, "model_call_id")
    for c in calls:
        if (
            c["snapshot_id"] != snapshot
            or c.get("query_id") is not None
            or c["stage"] not in {"questioner", "planner", "doubter", "embed"}
            or c["status"] not in {"success", "failed"}
            or type(c.get("retry_count")) is not int
            or c["retry_count"] != 0
        ):
            raise DependencyUnproven("pmmc_write_call_or_retry_unproven")
        current, seen = c, set()
        while current.get("parent_model_call_id"):
            key = current["parent_model_call_id"]
            if key in seen or key not in by_id:
                raise DependencyUnproven("pmmc_write_parent_call_unproven")
            seen.add(key)
            current = by_id[key]
    return by_id


def _indexes(facts, calls):
    from pmmc.infrastructure.database.step8_upstream_import import validate_r0_memory_vectors
    from pmmc.schemas import MemoryUnit

    rows, compiler = facts["rows"], facts["compiler"]
    snapshot, run = compiler["snapshot_id"], compiler["run_id"]
    qindex = _one(rows["qbank_index_manifests"])
    memory = _one(rows["memory_operator_index_manifests"])
    collections = _unique(facts["collections"], "collection_id")
    vectors = _unique(facts["vectors"], "embedding_id")
    eligible = {
        r["qbank_id"]
        for r in rows["questioner_candidate_gate_results"]
        if r["status"] == "eligible_for_planning"
    }
    paraphrases = {
        r["paraphrase_id"] for r in rows["qbank_paraphrases"] if r["qbank_id"] in eligible
    }
    if (
        qindex["status"] != "ready"
        or memory["status"] != "ready"
        or qindex["compiler_run_id"] != run
        or memory["run_id"] != run
        or qindex["eligible_question_count"] != len(eligible)
        or qindex["paraphrase_count"] != len(paraphrases)
    ):
        raise DependencyUnproven("pmmc_index_manifest_not_closed")
    required = {v for m in (qindex, memory) for k, v in m.items() if k.endswith("collection_id")}
    refs = rows["program_query_embeddings"]
    if len(refs) != 2 * len(eligible) or {(r["qbank_id"], r["channel"]) for r in refs} != {
        (qid, channel)
        for qid in eligible
        for channel in ("question_text_query", "question_joint_query")
    }:
        raise DependencyUnproven("pmmc_program_query_vector_denominator_mismatch")
    batches = _unique(rows["program_query_embedding_batches"], "query_embedding_batch_id")
    for ref in refs:
        vector = vectors[ref["embedding_id"]]
        batch = batches[ref["query_embedding_batch_id"]]
        if (
            vector["input_hash"] != ref["input_hash"]
            or vector["qbank_id"] != ref["qbank_id"]
            or vector["model_call_id"] != batch["model_call_id"]
            or ref["qbank_id"] not in batch["qbank_ids"]
            or batch["channel"] != ref["channel"]
        ):
            raise DependencyUnproven("pmmc_program_query_vector_lineage_mismatch")
        required.add(vector["collection_id"])
    if set(collections) != required:
        raise DependencyUnproven("pmmc_index_collection_denominator_mismatch")
    for collection in collections.values():
        if (
            collection["status"] != "ready"
            or collection["snapshot_id"] != snapshot
            or collection["dimension"] != 4096
            or not collection["model_revision"]
        ):
            raise DependencyUnproven("pmmc_index_collection_identity_mismatch")
    for vector in vectors.values():
        collection, call = collections[vector["collection_id"]], calls[vector["model_call_id"]]
        if (
            call["run_id"] != collection["run_id"]
            or call["stage"] != "embed"
            or call["status"] != "success"
            or call["model_alias"] != collection["model_alias"]
            or call["actual_model_id"]
            != (
                collection.get("actual_model_id")
                or {
                    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
                    "qwen3-vl-embedding-8b": "Qwen/Qwen3-VL-Embedding-8B",
                }.get(collection["model_alias"])
            )
            or vector["dimension"] != 4096
            or len(vector["embedding"]) != 4096
            or not all(math.isfinite(v) for v in vector["embedding"])
        ):
            raise DependencyUnproven("pmmc_index_vector_producer_mismatch")
        family = (
            "text"
            if collection["model_alias"] == compiler["memory_text_model_alias"]
            else "multimodal"
        )
        if (
            collection["model_alias"] != compiler["memory_" + family + "_model_alias"]
            or collection["model_revision"] != compiler["memory_" + family + "_model_revision"]
        ):
            raise DependencyUnproven("pmmc_index_model_revision_mismatch")
    for field in (
        "text_question_collection_id",
        "multimodal_question_collection_id",
        "text_paraphrase_collection_id",
        "multimodal_paraphrase_collection_id",
    ):
        selected = [v for v in vectors.values() if v["collection_id"] == qindex[field]]
        key, expected = (
            ("paraphrase_id", paraphrases) if "paraphrase" in field else ("qbank_id", eligible)
        )
        if len(selected) != len(expected) or {v[key] for v in selected} != expected:
            raise DependencyUnproven("pmmc_qbank_index_membership_mismatch")
    validate_r0_memory_vectors(
        {"memory_operator_index_manifests": [memory], "embeddings": list(vectors.values())},
        [MemoryUnit.model_validate(m) for m in facts["canonical_memories"]],
        memory_raw_image_policy=facts["memory_raw_image_policy"],
    )
    return eligible


def _execution_facts(facts, execution):
    rows = facts["rows"]
    program = _one([p for p in rows["memory_programs"] if p["plan_id"] == execution["plan_id"]])
    return {
        "execution": execution,
        "program": program,
        "compiler": facts["compiler"],
        "program_steps": [s for s in rows["program_steps"] if s["plan_id"] == program["plan_id"]],
        "steps": [
            s
            for s in rows["program_execution_steps"]
            if s["execution_id"] == execution["execution_id"]
        ],
        "memory_index": _one(rows["memory_operator_index_manifests"]),
        "query_vectors": [
            r for r in rows["program_query_embeddings"] if r["qbank_id"] == program["qbank_id"]
        ],
        "vectors": facts["vectors"],
        "model_calls": [
            c
            for c in facts["calls"]
            if (c.get("metadata") or {}).get("execution_id") == execution["execution_id"]
        ],
        "kernel_contract": facts["kernel_contract"],
        "kernel_lineage": facts["kernel_lineage"],
    }


def _validate_graph_execution(facts, execution, seen=frozenset()):
    """Imported traces retain measured source bindings; validate the sealed source."""
    from pmmc.experiments.step8_r0_cost_dependencies import validate_r0_cost_dependencies

    imported = (execution.get("metadata") or {}).get("r0_prefix_import")
    if imported is None:
        return validate_execution_dependencies(_execution_facts(facts, execution))
    key = (facts["compiler"]["run_id"], execution["execution_id"])
    if key in seen:
        raise DependencyUnproven("pmmc_imported_execution_cycle")
    entry = _one([e for e in facts["r0_imports"] if e["target_run_id"] == key[0]])
    if (
        entry["status"] != "complete"
        or canonical_content_hash(entry["facts"]) != entry["facts_hash"]
    ):
        raise DependencyUnproven("pmmc_imported_execution_proof_incomplete")
    proof = entry["facts"]
    aliases = validate_r0_cost_dependencies(proof)
    target_rows = proof["target"]["rows"]
    # Compare the consumed rows, not just their asserted import metadata.
    for name, field, value in (
        ("program_executions", "execution_id", execution["execution_id"]),
        ("program_execution_steps", "execution_id", execution["execution_id"]),
        ("program_steps", "plan_id", execution["plan_id"]),
    ):
        current = [r for r in facts["rows"][name] if r[field] == value]
        sealed = [r for r in target_rows[name] if r[field] == value]
        if canonical_content_hash(current) != canonical_content_hash(sealed):
            raise DependencyUnproven("pmmc_imported_execution_consumed_rows_mismatch")
    source = proof["source"]
    receipts = proof["source_receipts"]
    source_facts = {
        "compiler": source["headers"]["compiler"],
        "rows": source["rows"],
        "vectors": source["rows"]["embeddings"],
        "calls": source["rows"]["model_calls"],
        "kernel_contract": facts["kernel_contract"],
        "r0_imports": facts["r0_imports"],
        "kernel_lineage": [
            {
                "campaign": receipts["campaign"],
                "job": job,
                "receipt": receipts["receipts"][job["receipt_hash"]],
            }
            for job in receipts["jobs"].values()
        ],
    }
    source_execution = _one(
        [
            r
            for r in source["rows"]["program_executions"]
            if r["execution_id"] == imported["source_primary_key"]["execution_id"]
        ]
    )
    dependencies = _validate_graph_execution(source_facts, source_execution, seen | {key})
    reverse = {source: target for target, source in aliases.items()}
    return {role: {reverse[i] for i in ids} for role, ids in dependencies.items()}


def validate_write_graph(facts) -> dict[int, set[str]]:
    """Return whole-bank physical dependencies per deployed revision depth."""
    from pmmc.schemas.doubter import QBankRoundProjectionMember

    compiler, rows = facts["compiler"], facts["rows"]
    run, snapshot = compiler["run_id"], compiler["snapshot_id"]
    _verify_compiler_references(facts["kernel_lineage"])
    if (
        facts["state"]["run_id"] != run
        or facts["state"]["status"] != "completed"
        or not facts["state"]["completed_at"]
        or compiler["compiler_schema_version"] != "0.9.0"
        or facts["manifest"]["run_id"] != run
        or facts["manifest"]["manifest_json"]["dataset_snapshot_ids"] != [snapshot]
        or facts["kernel_contract"] != kernel_contract()
    ):
        raise DependencyUnproven("pmmc_write_run_not_terminal")
    units = _unique(rows["compiler_units"], "compiler_unit_id")
    if (
        not units
        or len(units) != compiler["unit_count"]
        or set(units)
        != {
            r["compiler_unit_id"]
            for r in rows["compiler_unit_stage_results"]
            if r["stage"] == "questioner" and r["status"] == "success"
        }
    ):
        raise DependencyUnproven("pmmc_write_unit_denominator_mismatch")
    for table in ("experiment_jobs", "compiler_stage_jobs", "candidate_stage_jobs"):
        if any(
            r["run_id"] != run or r["status"] not in {"succeeded", "cancelled"} for r in rows[table]
        ):
            raise DependencyUnproven("pmmc_write_jobs_not_terminal")
    for stage in ("planner", "dry_run"):
        initial = [
            r for r in rows["compiler_stage_jobs"] if r["stage"] == stage and r["round_id"] == 0
        ]
        if len(initial) != len(units) or {r["compiler_unit_id"] for r in initial} != set(units):
            raise DependencyUnproven("pmmc_write_stage_denominator_mismatch")
    r0 = _one(rows["r0_input_manifests"])
    if r0["run_id"] != run or not r0["hard_gate_passed"]:
        raise DependencyUnproven("pmmc_write_r0_not_closed")
    calls = _call_graph(facts["calls"], snapshot)
    if {c["model_call_id"] for c in facts["calls"] if c["run_id"] == run} != {
        c["model_call_id"] for c in rows["model_calls"]
    }:
        raise DependencyUnproven("pmmc_write_whole_run_call_denominator_mismatch")
    eligible = _indexes(facts, calls)
    if r0["candidate_count"] != len(eligible):
        raise DependencyUnproven("pmmc_write_candidate_denominator_mismatch")
    programs = _unique(rows["memory_programs"], "plan_id")
    feedback = _unique(rows["doubter_feedback"], "feedback_id")
    lineage = _unique(rows["planner_revision_lineage"], "revised_plan_id")
    for program in programs.values():
        if program["run_id"] != run or program["snapshot_id"] != snapshot:
            raise DependencyUnproven("pmmc_write_program_scope_mismatch")
        if program.get("producer_model_call_id"):
            if program["producer_model_call_id"] not in calls:
                raise DependencyUnproven("pmmc_write_program_producer_missing")
        else:
            link = lineage[program["plan_id"]]
            source, d = programs[link["source_plan_id"]], feedback[link["source_feedback_id"]]
            if (
                link["target_round_id"] != program["round_id"]
                or source["round_id"] + 1 != program["round_id"]
                or source["qbank_id"] != program["qbank_id"]
                or d["plan_id"] != source["plan_id"]
                or d["producer_model_call_id"] != link["producer_model_call_id"]
                or link["producer_model_call_id"] not in calls
            ):
                raise DependencyUnproven("pmmc_write_typed_patch_lineage_invalid")
    # Every known producer relationship must be in the complete call denominator.
    for values in rows.values():
        for row in values:
            for field in ("model_call_id", "producer_model_call_id"):
                if row.get(field) and row[field] not in calls:
                    raise DependencyUnproven("pmmc_write_referenced_call_missing")
    for call in facts["calls"]:
        if call["run_id"] != run:
            collection = (
                (call.get("request_config") or {}).get("request_metadata", {}).get("collection_id")
            )
            if call["stage"] != "embed" or collection not in {
                c["collection_id"] for c in facts["collections"]
            }:
                raise DependencyUnproven("pmmc_shared_index_attempt_scope_unproven")
    for execution in rows["program_executions"]:
        _validate_graph_execution(facts, execution)
    result = {}
    projections = _unique(rows["qbank_round_projection_manifests"], "round_id")
    if set(projections) != set(range(4)):
        raise DependencyUnproven("pmmc_write_round_projection_missing")
    for round_id, projection in projections.items():
        members = [
            r
            for r in rows["qbank_round_projection_members"]
            if r["projection_id"] == projection["projection_id"]
        ]
        native = [QBankRoundProjectionMember.model_validate(m) for m in members]
        digest = canonical_content_hash(
            [m.model_dump(mode="json") for m in sorted(native, key=lambda m: m.qbank_id)]
        )
        if (
            len(members) != len(eligible)
            or {m["qbank_id"] for m in members} != eligible
            or projection["member_count"] != len(members)
            or projection["pending_count"] != 0
            or projection["members_hash"] != digest
            or projection["source_r0_manifest_id"] != r0["r0_manifest_id"]
        ):
            raise DependencyUnproven("pmmc_write_round_denominator_unproven")
        selected = set()
        for identity, call in calls.items():
            stage = call["stage"]
            if stage in {"questioner", "embed"}:
                selected.add(identity)
            elif stage == "planner" and call.get("plan_id") is None:
                selected.add(identity)
            else:
                source_round = programs[call["plan_id"]]["round_id"]
                depth = source_round + (stage == "planner")
                if depth <= round_id and (stage != "doubter" or round_id > 0):
                    selected.add(identity)
        # Parentage is an explicit producer edge, not a second role allocation.
        pending = list(selected)
        while pending:
            parent = calls[pending.pop()].get("parent_model_call_id")
            if parent and parent not in selected:
                selected.add(parent)
                pending.append(parent)
        result[round_id] = selected
    return result


def validate_execution_dependencies(facts) -> dict[str, set[str]]:
    if "imported_execution_graph" in facts:
        return _validate_graph_execution(facts["imported_execution_graph"], facts["execution"])
    trace, program, compiler = facts["execution"], facts["program"], facts["compiler"]
    if (
        facts["kernel_contract"] != kernel_contract()
        or compiler["operator_set_version"] != "pmmc-operators-v7"
        or compiler["operator_registry_hash"] != facts["kernel_contract"]["operator_registry_hash"]
        or compiler["compiler_config_hash"] != facts["kernel_contract"]["compiler_config_hash"]
        or program.get("operator_registry_hash", compiler["operator_registry_hash"])
        != compiler["operator_registry_hash"]
        or program["operator_set_version"] != compiler["operator_set_version"]
        or trace["source_compiler_run_id"] != compiler["run_id"]
        or program["run_id"] != compiler["run_id"]
        or trace["plan_id"] != program["plan_id"]
        or trace["snapshot_id"] != compiler["snapshot_id"]
        or trace["status"] not in {"success", "failed"}
        or not trace["completed_at"]
    ):
        raise DependencyUnproven("pmmc_execution_contract_unproven")
    _validate_kernel_lineage(facts["kernel_lineage"], trace, program)
    planned = sorted(facts["program_steps"], key=lambda s: s["step_id"])
    actual = sorted(facts["steps"], key=lambda s: s["step_id"])
    if not planned or not actual or len(actual) > len(planned):
        raise DependencyUnproven("pmmc_execution_step_denominator_unproven")
    for expected, step in zip(planned, actual, strict=False):
        output = step["output"]
        if (
            expected["op_name"] not in OPERATOR_SET_V7
            or (step["step_id"], step["op_name"]) != (expected["step_id"], expected["op_name"])
            or any(
                step[k] != trace[k]
                for k in ("execution_id", "run_id", "snapshot_id", "plan_id", "query_id")
            )
            or step["output_hash"] != canonical_content_hash(output)
            or step["success"] is not output["success"]
        ):
            raise DependencyUnproven("pmmc_execution_step_binding_mismatch")
    last = actual[-1]["output"]
    if len(actual) < len(planned) and last["success"] and last["control_action"] == "continue":
        raise DependencyUnproven("pmmc_execution_truncated_without_stop")
    if any(
        not s["output"]["success"] or s["output"]["control_action"] != "continue"
        for s in actual[:-1]
    ):
        raise DependencyUnproven("pmmc_execution_continued_after_stop")
    if (trace["status"] == "success") != last["success"]:
        raise DependencyUnproven("pmmc_execution_terminal_status_mismatch")
    memory = facts["memory_index"]
    bindings = {
        "text_memory": memory["text_collection_id"],
        "visual_memory": memory["visual_collection_id"],
    }
    if trace["bindings_hash"] != canonical_content_hash(bindings) or memory["status"] != "ready":
        raise DependencyUnproven("pmmc_execution_collection_binding_mismatch")
    refs = facts["query_vectors"]
    if (
        len(refs) != 2
        or {r["channel"] for r in refs} != {"question_text_query", "question_joint_query"}
        or any(
            r["qbank_id"] != program["qbank_id"] or r["run_id"] != compiler["run_id"] for r in refs
        )
    ):
        raise DependencyUnproven("pmmc_execution_cached_vector_denominator_mismatch")
    vectors = _unique(facts["vectors"], "embedding_id")
    write = set()
    for ref in refs:
        vector = vectors[ref["embedding_id"]]
        if vector["input_hash"] != ref["input_hash"] or vector["qbank_id"] != program["qbank_id"]:
            raise DependencyUnproven("pmmc_execution_cached_vector_identity_mismatch")
        write.add(vector["model_call_id"])
    if facts["model_calls"]:
        # This registry only performs deterministic repository operations. An
        # extension that invokes providers needs its own explicit contract.
        raise DependencyUnproven("pmmc_cpu_execution_has_unexpected_provider_call")
    return {"query": set(), "write": write}


async def collect_pmmc_cost_dependencies(
    session,
    *,
    compiler_run_ids,
    execution_ids,
    campaign_id: str,
    selection_hash: str,
    database_snapshot: dict,
    r0_dependencies: dict | None = None,
) -> dict[str, Any]:
    from pmmc.infrastructure.database import compiler_models as cm
    from pmmc.infrastructure.database import doubter_models as dm
    from pmmc.infrastructure.database import experiment_models as xm
    from pmmc.infrastructure.database import models as bm
    from pmmc.infrastructure.database.repositories import PostgresMemoryRepository
    from pmmc.infrastructure.database.step8_upstream_import import (
        QUESTIONER_TABLES,
        R0_PREFIX_TABLES,
    )

    async def read(table, predicate):
        table = getattr(table, "__table__", table)
        statement = (
            select(table)
            .where(predicate)
            .execution_options(
                step8_inventory_table=table.fullname, step8_cost_dependency_table=table.fullname
            )
        )
        values = [dict(r) for r in (await session.execute(statement)).mappings()]
        if table.name == "embeddings":
            for row in values:
                row["embedding"] = [float(v) for v in row["embedding"]]
        return values

    @asynccontextmanager
    async def borrowed():
        yield session

    async def imports(run, compiler):
        nonlocal r0_dependencies
        if not compiler.get("metadata", {}).get("r0_prefix_upstream_import"):
            return []
        if r0_dependencies is None:
            from pmmc.experiments.step8_r0_cost_dependencies import (
                collect_r0_prefix_cost_dependencies,
            )

            r0_dependencies = await collect_r0_prefix_cost_dependencies(
                session,
                compiler_run_ids=compiler_run_ids,
                campaign_id=campaign_id,
                selection_hash=selection_hash,
                database_snapshot=database_snapshot,
            )
        return r0_dependencies["imports"]

    tables = {t.name: t for t in (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)}
    for model in (
        dm.CandidateStageJobModel,
        dm.PlannerRevisionLineageModel,
        dm.QBankRoundProjectionManifestModel,
        dm.QBankRoundProjectionMemberModel,
        cm.DoubterFeedbackModel,
    ):
        tables[model.__tablename__] = cast(Table, model.__table__)
    tables.pop("embedding_collections")
    tables.pop("embeddings")
    graphs, executions = [], []
    dependency_calls: set[str] = set()
    dependency_runs: set[str] = set()
    for run in sorted(set(compiler_run_ids)):
        facts, reason = {}, None
        try:
            async with session.begin_nested():
                compiler = _one(await read(cm.CompilerRunModel, cm.CompilerRunModel.run_id == run))
                manifest = _one(await read(bm.RunManifestModel, bm.RunManifestModel.run_id == run))
                rows = {}
                for name, table in tables.items():
                    if name == "job_attempts":
                        jobs = tables["experiment_jobs"]
                        rows[name] = await read(
                            table,
                            table.c.job_id.in_(select(jobs.c.job_id).where(jobs.c.run_id == run)),
                        )
                        continue
                    column = (
                        table.c.compiler_run_id
                        if name == "qbank_index_manifests"
                        else table.c.run_id
                    )
                    rows[name] = await read(table, column == run)
                cids = {
                    v
                    for name in ("qbank_index_manifests", "memory_operator_index_manifests")
                    for row in rows[name]
                    for k, v in row.items()
                    if k.endswith("collection_id")
                }
                query_ids = {r["embedding_id"] for r in rows["program_query_embeddings"]}
                qvectors = await read(
                    xm.EmbeddingModel, xm.EmbeddingModel.embedding_id.in_(query_ids)
                )
                cids.update(v["collection_id"] for v in qvectors)
                collections = await read(
                    xm.EmbeddingCollectionModel, xm.EmbeddingCollectionModel.collection_id.in_(cids)
                )
                vectors = await read(xm.EmbeddingModel, xm.EmbeddingModel.collection_id.in_(cids))
                calls = await read(
                    xm.RuntimeModelCallModel,
                    or_(
                        xm.RuntimeModelCallModel.run_id == run,
                        xm.RuntimeModelCallModel.model_call_id.in_(
                            {v["model_call_id"] for v in vectors}
                        ),
                        xm.RuntimeModelCallModel.request_config["request_metadata"][
                            "collection_id"
                        ].astext.in_(cids),
                    ),
                )
                memories = await PostgresMemoryRepository(cast(Any, borrowed)).list_snapshot(
                    compiler["snapshot_id"],
                    conversation_ids=set(compiler["metadata"]["conversation_ids"]),
                )
                facts = _sealed_json(
                    {
                        "compiler": compiler,
                        "manifest": manifest,
                        "state": _one(await read(xm.RunStateModel, xm.RunStateModel.run_id == run)),
                        "rows": rows,
                        "calls": calls,
                        "collections": collections,
                        "vectors": vectors,
                        "canonical_memories": [m.model_dump(mode="json") for m in memories],
                        "memory_raw_image_policy": manifest["manifest_json"]["metadata"][
                            "questioner_config"
                        ].get("memory_raw_image_policy", "allow"),
                        "kernel_contract": kernel_contract(),
                        "kernel_lineage": await _kernel_receipt_lineage(read, run),
                        "r0_imports": await imports(run, compiler),
                    }
                )
                validate_write_graph(facts)
                dependency_calls.update(c["model_call_id"] for c in calls)
                dependency_runs.update(c["run_id"] for c in calls)
        except (SQLAlchemyError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            reason = "pmmc_write_graph_unavailable_or_incomplete"
        graphs.append(
            {
                "run_id": run,
                "status": "incomplete" if reason else "complete",
                "reasons": [reason] if reason else [],
                "facts": facts,
                "facts_hash": canonical_content_hash(facts),
            }
        )
    for identity in sorted(set(execution_ids)):
        facts, reason = {}, None
        try:
            async with session.begin_nested():
                execution = _one(
                    await read(
                        cm.ProgramExecutionModel, cm.ProgramExecutionModel.execution_id == identity
                    )
                )
                compiler_id = execution["source_compiler_run_id"]
                compiler = _one(
                    await read(cm.CompilerRunModel, cm.CompilerRunModel.run_id == compiler_id)
                )
                program = _one(
                    await read(
                        cm.MemoryProgramModel, cm.MemoryProgramModel.plan_id == execution["plan_id"]
                    )
                )
                refs = await read(
                    cm.ProgramQueryEmbeddingModel,
                    (cm.ProgramQueryEmbeddingModel.run_id == compiler_id)
                    & (cm.ProgramQueryEmbeddingModel.qbank_id == program["qbank_id"]),
                )
                facts = _sealed_json(
                    {
                        "execution": execution,
                        "program": program,
                        "compiler": compiler,
                        "program_steps": await read(
                            cm.ProgramStepModel, cm.ProgramStepModel.plan_id == program["plan_id"]
                        ),
                        "steps": await read(
                            cm.ProgramExecutionStepModel,
                            cm.ProgramExecutionStepModel.execution_id == identity,
                        ),
                        "memory_index": _one(
                            await read(
                                cm.MemoryOperatorIndexManifestModel,
                                cm.MemoryOperatorIndexManifestModel.run_id == compiler_id,
                            )
                        ),
                        "query_vectors": refs,
                        "vectors": await read(
                            xm.EmbeddingModel,
                            xm.EmbeddingModel.embedding_id.in_({r["embedding_id"] for r in refs}),
                        ),
                        "model_calls": await read(
                            xm.RuntimeModelCallModel,
                            xm.RuntimeModelCallModel.metadata_json["execution_id"].astext
                            == identity,
                        ),
                        "kernel_contract": kernel_contract(),
                        "kernel_lineage": await _kernel_receipt_lineage(
                            read, compiler_id, execution["run_id"]
                        ),
                    }
                )
                if (execution.get("metadata") or {}).get("r0_prefix_import"):
                    entries = await imports(compiler_id, compiler)
                    entry = _one([e for e in entries if e["target_run_id"] == compiler_id])
                    target = entry["facts"]["target"]
                    facts = _sealed_json(
                        {
                            "execution": execution,
                            "imported_execution_graph": {
                                "compiler": compiler,
                                "rows": target["rows"],
                                "calls": target["rows"]["model_calls"],
                                "vectors": target["rows"]["embeddings"],
                                "kernel_contract": kernel_contract(),
                                "kernel_lineage": [],
                                "r0_imports": entries,
                            },
                        }
                    )
                validate_execution_dependencies(facts)
        except (SQLAlchemyError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            reason = "pmmc_execution_dependencies_unavailable_or_incomplete"
        executions.append(
            {
                "execution_id": identity,
                "status": "incomplete" if reason else "complete",
                "reasons": [reason] if reason else [],
                "facts": facts,
                "facts_hash": canonical_content_hash(facts),
            }
        )
    return {
        "schema_version": PMMC_COST_DEPENDENCIES_VERSION,
        "campaign_id": campaign_id,
        "selection_hash": selection_hash,
        "source_role": "evaluator",
        "database_snapshot": database_snapshot,
        "write_graphs": graphs,
        "executions": executions,
        "dependency_model_call_ids": sorted(dependency_calls),
        "dependency_run_ids": sorted(dependency_runs),
        "adopted_model_call_ids": [],
    }


def pmmc_entry(snapshot, section, key, identity):
    manifest = snapshot.dependency_artifact(
        PMMC_COST_DEPENDENCIES_ARTIFACT, PMMC_COST_DEPENDENCIES_VERSION
    )
    entry = _one([r for r in manifest[section] if r[key] == identity])
    if (
        entry["status"] != "complete"
        or entry["reasons"]
        or canonical_content_hash(entry["facts"]) != entry["facts_hash"]
    ):
        raise DependencyUnproven("pmmc_cost_dependency_proof_incomplete")
    return entry["facts"]


def bound_write_dependencies(snapshot, calls, run_id, round_id):
    facts = pmmc_entry(snapshot, "write_graphs", "run_id", run_id)
    if facts["compiler"]["run_id"] != run_id:
        raise DependencyUnproven("pmmc_write_owner_mismatch")
    selected = validate_write_graph(facts)[round_id]
    for call in facts["calls"]:
        if _call_hash(calls.get(call["model_call_id"])) != _call_hash(call):
            raise DependencyUnproven("pmmc_write_exported_call_mismatch")
        if call.get("cache_hit") is True and call["model_call_id"] not in snapshot.call_aliases:
            raise DependencyUnproven("pmmc_write_cache_alias_unproven")
    return selected
