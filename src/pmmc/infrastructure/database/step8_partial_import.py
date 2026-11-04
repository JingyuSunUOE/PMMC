"""Bounded Q/P0/index adoption. This module never reads or constructs an R0 proof.

Campaign policy and receipt authorization belong to the caller. Database facts,
object bytes, ownership, and recovery aliases are independently checked here.
All public proof values are dictionaries; no full-prefix proof subtype is used.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, NoReturn, cast

from sqlalchemy import Table, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import (
    canonical_content_hash,
    memory_program_content_hash,
    stable_compiler_id,
)
from pmmc.compiler.memory_index import make_memory_operator_index
from pmmc.experiments.manifest import run_config_hash
from pmmc.experiments.step8_upstream_import import remap_questioner_rows
from pmmc.infrastructure.database import compiler_models as c
from pmmc.infrastructure.database import questioner_models as q
from pmmc.infrastructure.database.compiler_mappers import memory_program_from_rows
from pmmc.infrastructure.database.experiment_models import EmbeddingCollectionModel, EmbeddingModel
from pmmc.infrastructure.database.experiment_repositories import PostgresRunRepository
from pmmc.infrastructure.database.models import RunManifestModel
from pmmc.infrastructure.database.step8_upstream_import import (
    QUESTIONER_TABLES,
    PostgresQuestionerImportRepository,
    _rehash_questions,
    _sort_rows,
    validate_r0_memory_vectors,
)
from pmmc.infrastructure.object_store import ObjectStore
from pmmc.schemas import RunManifest, RunStatus
from pmmc.schemas.base import utc_now
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION

PROOF_VERSION = "step8-partial-prefix-proof-v1"
ADOPTION_VERSION = "step8-partial-prefix-adoption-v1"
LINEAGE_KEY = "partial_prefix_upstream_import"
ROW_LINEAGE_KEY = "partial_prefix_import"
PARTIAL_STAGES = ("questioner", "planner0", "text_index", "multimodal_index")
_P0_REJECTIONS = ("parse_failed", "static_rejected", "policy_rejected")
_P0_VALIDATIONS = ("static", "policy")
Rows = dict[str, list[dict[str, Any]]]
ProofValidator = Callable[[dict[str, Any]], Awaitable[None]]

# Explicitly excludes execution, dry-run, R0, revisions, Doubter and evaluation.
# The order is also the insertion order, with parents before their children.
PARTIAL_PREFIX_TABLES: tuple[Table, ...] = (
    *QUESTIONER_TABLES,
    *(
        cast(Table, model.__table__)
        for model in (
            c.CompilerStageJobModel,
            c.PlannerGenerationBatchModel,
            c.PlannerRequirementContractModel,
            c.PlannerCandidateDraftModel,
            c.MemoryProgramModel,
            c.ProgramStepModel,
            c.ProgramExpectedEvidenceRefModel,
            c.PlannerBatchProgramModel,
            c.PlannerCandidateOutcomeModel,
            c.ProgramValidationResultModel,
            c.ProgramValidationIssueModel,
            EmbeddingCollectionModel,
            EmbeddingModel,
            q.QBankIndexManifestModel,
            c.MemoryOperatorIndexManifestModel,
            c.ProgramQueryEmbeddingBatchModel,
            c.ProgramQueryEmbeddingModel,
        )
    ),
)
_TABLES = {t.name: t for t in PARTIAL_PREFIX_TABLES}
_KEYS = {t.name: [col.name for col in t.primary_key] for t in PARTIAL_PREFIX_TABLES}
_FOREIGN_TABLES = {"embedding_collections", "model_calls", "experiment_jobs", "run_artifacts"}


def _fail(code: str) -> NoReturn:
    raise ValueError("partial_prefix_import_" + code)


def partial_prefix_table_contract_hash() -> str:
    return canonical_content_hash(
        {
            t.name: {
                "columns": [
                    (col.name, str(col.type), col.nullable, col.computed is not None)
                    for col in t.columns
                ],
                "primary_key": _KEYS[t.name],
                "foreign_keys": sorted(
                    (fk.parent.name, fk.target_fullname) for fk in t.foreign_keys
                ),
            }
            for t in PARTIAL_PREFIX_TABLES
        }
    )


def normalize_partial_prefix_rows(rows: Rows) -> Rows:
    if set(rows) - set(_TABLES):
        _fail("table_allowlist_mismatch")
    result = copy.deepcopy(rows)
    # This legacy projection is changed by D, but is not part of program identity.
    for row in result.get("memory_programs", []):
        row["accepted"] = True
    return {name: _sort_rows(_TABLES[name], values) for name, values in result.items()}


def partial_prefix_rows_hash(rows: Rows) -> str:
    return canonical_content_hash(normalize_partial_prefix_rows(rows))


def _stable_headers(headers: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(headers)
    value["compiler"].pop("active_round", None)
    for owner in (value["compiler"], value["manifest"]):
        owner.get("metadata", {}).pop(LINEAGE_KEY, None)
        owner.get("metadata", {}).pop("active_round", None)
    value["manifest"].pop("config_hash", None)
    return value


def partial_prefix_facts_hash(headers: dict[str, Any], rows: Rows) -> str:
    return canonical_content_hash(
        {"headers": _stable_headers(headers), "rows": normalize_partial_prefix_rows(rows)}
    )


def _check_foreign_keys(rows: Rows) -> None:
    """Check imported FK closure, including composite run/unit ownership."""
    for table in PARTIAL_PREFIX_TABLES:
        for constraint in table.foreign_key_constraints:
            elements = list(constraint.elements)
            remote = elements[0].target_fullname.rsplit(".", 2)[-2]
            if remote not in rows:
                continue  # Canonical snapshots/memories and runtime sessions stay in place.
            parent_keys = [e.target_fullname.rsplit(".", 1)[-1] for e in elements]
            parents = {tuple(r.get(k) for k in parent_keys) for r in rows[remote]}
            for row in rows[table.name]:
                key = tuple(row.get(e.parent.name) for e in elements)
                if all(v is not None for v in key) and key not in parents:
                    _fail("foreign_key_not_closed_" + table.name)


def validate_partial_prefix_rows(rows: Rows, run_id: str) -> None:
    """Validate pre-execution denominators without weakening any old validator."""
    if set(rows) != set(_TABLES):
        _fail("table_allowlist_mismatch")
    for table in PARTIAL_PREFIX_TABLES:
        values = rows[table.name]
        if len({tuple(r[k] for k in _KEYS[table.name]) for r in values}) != len(values):
            _fail("duplicate_primary_key")
        for row in values:
            if set(row) - set(table.c.keys()):
                _fail("unknown_column")
            if "run_id" in table.c and table.name not in _FOREIGN_TABLES:
                if row.get("run_id") != run_id:
                    _fail("foreign_fact_owner")
            if "round_id" in table.c and row.get("round_id") != 0:
                _fail("revision_forbidden")
    _check_foreign_keys(rows)
    units = {r["compiler_unit_id"] for r in rows["compiler_units"]}
    complete = [
        r
        for r in rows["compiler_unit_stage_results"]
        if r["stage"] == "questioner" and r["status"] == "success"
    ]
    if any(
        r["stage"] not in {"questioner", "planner"} for r in rows["compiler_unit_stage_results"]
    ):
        _fail("non_prefix_unit_stage")
    if (
        not units
        or len(complete) != len(units)
        or {
            r["compiler_unit_id"]
            for r in complete
            if r["stage"] == "questioner" and r["status"] == "success"
        }
        != units
    ):
        _fail("questioner_not_complete")
    jobs = rows["compiler_stage_jobs"]
    if (
        len(jobs) != len(units)
        or {r["compiler_unit_id"] for r in jobs} != units
        or any(r["stage"] != "planner" or r["status"] != "succeeded" for r in jobs)
    ):
        _fail("planner_not_complete")
    if any(
        r.get("accepted") is not None or r.get("rejection_reason") is not None
        for r in rows["qbank_questions"]
    ):
        _fail("legacy_question_decision_forbidden")
    eligible = {
        r["qbank_id"]
        for r in rows["questioner_candidate_gate_results"]
        if r["status"] == "eligible_for_planning"
    }
    outcomes = rows["planner_candidate_outcomes"]
    rejected = {r["qbank_id"] for r in outcomes}
    planned = {r["qbank_id"] for r in rows["memory_programs"]}
    if (
        len(rejected) != len(outcomes)
        or len(planned) != len(rows["memory_programs"])
        or rejected & planned
        or rejected | planned != eligible
        or any(r["status"] not in _P0_REJECTIONS or r.get("plan_id") is not None for r in outcomes)
    ):
        _fail("planner_denominator_mismatch")
    if any(r["validation_kind"] not in _P0_VALIDATIONS for r in rows["program_validation_results"]):
        _fail("non_prefix_validation")
    for row in rows["memory_programs"]:
        if row["qbank_id"] not in eligible:
            _fail("ineligible_program")
        program = memory_program_from_rows(
            row,
            [s for s in rows["program_steps"] if s["plan_id"] == row["plan_id"]],
            [e for e in rows["program_expected_evidence_refs"] if e["plan_id"] == row["plan_id"]],
        )
        if program.program_hash != memory_program_content_hash(program):
            _fail("program_hash_mismatch")
    manifests = rows["qbank_index_manifests"] + rows["memory_operator_index_manifests"]
    if (
        len(rows["qbank_index_manifests"]) != 1
        or len(rows["memory_operator_index_manifests"]) != 1
        or any(r["status"] != "ready" for r in manifests)
        or rows["qbank_index_manifests"][0]["compiler_run_id"] != run_id
    ):
        _fail("index_not_ready")
    collections = {r["collection_id"]: r for r in rows["embedding_collections"]}
    vectors = {r["embedding_id"]: r for r in rows["embeddings"]}
    referenced = {v for r in manifests for k, v in r.items() if k.endswith("collection_id")}
    refs = rows["program_query_embeddings"]
    if len(refs) != 2 * len(eligible) or {(r["qbank_id"], r["channel"]) for r in refs} != {
        (qid, channel)
        for qid in eligible
        for channel in ("question_text_query", "question_joint_query")
    }:
        _fail("query_vector_denominator_mismatch")
    for ref in refs:
        vector = vectors.get(ref["embedding_id"])
        if (
            vector is None
            or vector["qbank_id"] != ref["qbank_id"]
            or vector["input_hash"] != ref["input_hash"]
        ):
            _fail("query_vector_lineage_mismatch")
        referenced.add(vector["collection_id"])
    if set(collections) != referenced:
        _fail("collection_denominator_mismatch")
    memory_index = rows["memory_operator_index_manifests"][0]
    shared = {memory_index["text_collection_id"], memory_index["visual_collection_id"]}
    for collection in collections.values():
        if (
            collection["status"] != "ready"
            or collection["dimension"] != 4096
            or not collection["model_revision"]
            or collection.get("corpus_id") is not None
            or collection["snapshot_id"] != memory_index["snapshot_id"]
        ):
            _fail("collection_identity_invalid")
        if collection.get("run_id") != run_id and collection["collection_id"] not in shared:
            _fail("foreign_non_memory_collection_unsupported")
    for vector in vectors.values():
        values = vector["embedding"]
        if (
            vector["collection_id"] not in collections
            or vector["dimension"] != 4096
            or len(values) != 4096
            or not all(math.isfinite(v) for v in values)
            or vector.get("retrieval_document_id") is not None
        ):
            _fail("vector_bytes_or_scope_invalid")
    qi = rows["qbank_index_manifests"][0]
    paraphrases = {
        r["paraphrase_id"] for r in rows["qbank_paraphrases"] if r["qbank_id"] in eligible
    }
    if qi["eligible_question_count"] != len(eligible) or qi["paraphrase_count"] != len(paraphrases):
        _fail("qbank_index_count_mismatch")
    for field in (
        "text_question_collection_id",
        "multimodal_question_collection_id",
        "text_paraphrase_collection_id",
        "multimodal_paraphrase_collection_id",
    ):
        records = [v for v in vectors.values() if v["collection_id"] == qi[field]]
        key, expected = (
            ("paraphrase_id", paraphrases) if "paraphrase" in field else ("qbank_id", eligible)
        )
        if len(records) != len(expected) or {r[key] for r in records} != expected:
            _fail("qbank_vector_denominator_mismatch")
    calls = {r["model_call_id"]: r for r in rows["model_calls"]}
    for vector in vectors.values():
        producer = calls.get(vector.get("model_call_id"))
        if producer is None or producer["stage"] != "embed" or producer["status"] != "success":
            _fail("vector_producer_unproven")
    foreign_calls = set()
    for call in calls.values():
        if (
            call["stage"] not in {"questioner", "planner", "embed"}
            or any(
                call.get(k) is not None
                for k in ("query_id", "plan_id", "step_id", "context_assembly_id")
            )
            or call["status"] not in {"success", "failed"}
        ):
            _fail("non_prefix_call")
        if call.get("run_id") != run_id:
            foreign_calls.add(call["model_call_id"])
            if call["stage"] != "embed" or any(
                call.get(k) is not None for k in ("compiler_unit_id", "qbank_id")
            ):
                _fail("foreign_non_embedding_call_unsupported")
        if call.get("job_id") is None and call["stage"] != "embed":
            _fail("unbound_model_call")
        if call.get("snapshot_id") not in {None, memory_index["snapshot_id"]}:
            _fail("call_snapshot_mismatch")
    # A foreign graph must be reached from a shared memory vector, not merely
    # from a matching run ID. Parent requests and failed recovery attempts count.
    allowed_foreign = {
        v["model_call_id"]
        for v in vectors.values()
        if v["collection_id"] in shared and v.get("model_call_id")
    }
    while True:
        expanded = allowed_foreign | {
            calls[cid]["parent_model_call_id"]
            for cid in allowed_foreign
            if cid in calls and calls[cid].get("parent_model_call_id")
        }
        expanded |= {
            cid
            for cid, call in calls.items()
            if call.get("metadata", {}).get(ROW_LINEAGE_KEY, {}).get("recovery_for_model_call_id")
            in allowed_foreign
        }
        if expanded == allowed_foreign:
            break
        allowed_foreign = expanded
    # Recovery siblings are also recognized by immutable producer request metadata.
    families = {_recovery_family(calls[cid]) for cid in allowed_foreign if cid in calls}
    allowed_foreign |= {
        cid
        for cid, call in calls.items()
        if _recovery_family(call) is not None and _recovery_family(call) in families
    }
    if not foreign_calls <= allowed_foreign:
        _fail("foreign_call_not_reachable")
    _sort_rows(_TABLES["model_calls"], list(calls.values()))
    for row in rows["experiment_jobs"]:
        if (
            row["stage"] not in {"compile", "embed"}
            or row["status"] != "succeeded"
            or row.get("query_id") is not None
            or (row["run_id"] != run_id and row["stage"] != "embed")
        ):
            _fail("model_job_not_terminal")
    if any(r["status"] not in {"succeeded", "failed", "cancelled"} for r in rows["job_attempts"]):
        _fail("attempt_not_terminal")
    if any(r.get("query_id") is not None for r in rows["run_artifacts"]):
        _fail("non_prefix_artifact")


def _recovery_family(call: Mapping[str, Any]) -> tuple[Any, ...] | None:
    meta = call.get("request_config", {}).get("request_metadata", {})
    if meta.get("task") == "memory_operator_index":
        return (call["run_id"], meta.get("collection_id"), meta.get("batch_ordinal"))
    return None


async def _read_table(session: AsyncSession, name: str, predicate: Any) -> list[dict[str, Any]]:
    table = _TABLES[name]
    columns = [col for col in table.columns if col.computed is None]
    result = [
        dict(r) for r in (await session.execute(select(*columns).where(predicate))).mappings()
    ]
    if name == "embeddings":
        for row in result:
            row["embedding"] = [float(v) for v in row["embedding"]]
    return result


async def read_partial_prefix_rows(
    session: AsyncSession,
    run_id: str,
    *,
    questioner_repository: PostgresQuestionerImportRepository,
) -> tuple[dict[str, Any], Rows]:
    """Read a closed prefix from the caller's snapshot; never inspect execution tables."""
    headers, rows = await questioner_repository._read(
        session, run_id, include_index_diagnostics=True
    )
    plans = _TABLES["memory_programs"]
    plan_ids = select(plans.c.plan_id).where(plans.c.run_id == run_id, plans.c.round_id == 0)
    validation = _TABLES["program_validation_results"]
    validation_ids = select(validation.c.validation_result_id).where(
        validation.c.run_id == run_id,
        validation.c.plan_id.in_(plan_ids),
        validation.c.validation_kind.in_(_P0_VALIDATIONS),
    )
    for table in PARTIAL_PREFIX_TABLES[len(QUESTIONER_TABLES) :]:
        name = table.name
        if name in {"embeddings", "embedding_collections"}:
            continue
        owner = table.c.compiler_run_id if name == "qbank_index_manifests" else table.c.run_id
        predicate = owner == run_id
        if "round_id" in table.c:
            predicate &= table.c.round_id == 0
        if "plan_id" in table.c and name != "planner_candidate_outcomes":
            predicate &= table.c.plan_id.in_(plan_ids)
        if name == "compiler_stage_jobs":
            predicate &= table.c.stage == "planner"
        if name == "planner_candidate_outcomes":
            predicate &= table.c.status.in_(_P0_REJECTIONS)
            predicate &= table.c.plan_id.is_(None)
        if name == "program_validation_results":
            predicate &= table.c.validation_kind.in_(_P0_VALIDATIONS)
        if name == "program_validation_issues":
            predicate &= table.c.validation_result_id.in_(validation_ids)
        rows[name] = await _read_table(session, name, predicate)
    stages = _TABLES["compiler_unit_stage_results"]
    rows[stages.name] = await _read_table(
        session,
        stages.name,
        (stages.c.run_id == run_id) & stages.c.stage.in_(["questioner", "planner"]),
    )
    collection_ids = {
        v
        for name in ("qbank_index_manifests", "memory_operator_index_manifests")
        for r in rows[name]
        for k, v in r.items()
        if k.endswith("collection_id")
    }
    vectors = _TABLES["embeddings"]
    query_ids = {r["embedding_id"] for r in rows["program_query_embeddings"]}
    query_vectors = await _read_table(session, "embeddings", vectors.c.embedding_id.in_(query_ids))
    collection_ids.update(r["collection_id"] for r in query_vectors)
    rows["embeddings"] = await _read_table(
        session, "embeddings", vectors.c.collection_id.in_(collection_ids)
    )
    collections = _TABLES["embedding_collections"]
    rows["embedding_collections"] = await _read_table(
        session, "embedding_collections", collections.c.collection_id.in_(collection_ids)
    )
    call_ids = {
        r["model_call_id"]
        for name in (
            "model_calls",
            "planner_generation_batches",
            "program_query_embedding_batches",
            "embeddings",
        )
        for r in rows[name]
        if r.get("model_call_id")
    }
    # Direct failed embedding attempts have no FK from a successful vector. A
    # persisted projection explicitly binds those extra calls in its alias map.
    call_ids.update(
        headers["compiler"]["metadata"].get(LINEAGE_KEY, {}).get("model_call_aliases", {})
    )
    calls = _TABLES["model_calls"]
    # Include all Q/P0 attempts in their original queue jobs, but not later D calls.
    jobs = _TABLES["experiment_jobs"]
    qjob_ids = {r["job_id"] for r in rows["experiment_jobs"]}
    seen: set[str] | None = None
    while call_ids != seen:
        seen = set(call_ids)
        call_rows = await _read_table(session, "model_calls", calls.c.model_call_id.in_(call_ids))
        if {r["model_call_id"] for r in call_rows} != call_ids:
            _fail("call_ancestry_missing")
        call_ids.update(r["parent_model_call_id"] for r in call_rows if r["parent_model_call_id"])
        job_ids = qjob_ids | {r["job_id"] for r in call_rows if r["job_id"]}
        siblings = await _read_table(
            session,
            "model_calls",
            calls.c.job_id.in_(job_ids)
            & calls.c.stage.in_(["questioner", "planner", "embed"])
            & calls.c.plan_id.is_(None)
            & calls.c.query_id.is_(None),
        )
        call_ids.update(r["model_call_id"] for r in siblings)
        for call in call_rows:
            meta = call.get("request_config", {}).get("request_metadata", {})
            count = meta.get("recovery_attempt", 0)
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 32:
                _fail("invalid_recovery_attempt")
            if not count:
                continue
            if meta.get("task") == "memory_operator_index":
                logical = stable_compiler_id(
                    "memory-index-model-call", [meta["collection_id"], meta["batch_ordinal"]]
                )
            elif meta.get("task") == "program_query_embedding":
                batch = next(
                    (
                        r
                        for r in rows["program_query_embedding_batches"]
                        if r["model_call_id"] == call["model_call_id"]
                    ),
                    None,
                )
                if batch is None:
                    continue  # A failed predecessor was already reached from the successful batch.
                revisions = {
                    collection["model_revision"]
                    for vector in rows["embeddings"]
                    if vector.get("model_call_id") == call["model_call_id"]
                    for collection in rows["embedding_collections"]
                    if collection["collection_id"] == vector["collection_id"]
                }
                if len(revisions) != 1:
                    _fail("query_recovery_revision_missing")
                logical = stable_compiler_id(
                    "program-query-model-call",
                    [batch["query_embedding_batch_id"], next(iter(revisions))],
                )
            else:
                _fail("unsupported_embedding_recovery")
            expected = stable_compiler_id(
                "embedding-model-call-attempt", [call["run_id"], logical, count]
            )
            if call["model_call_id"] != expected:
                # Projected calls keep producer request IDs in immutable request_config.
                origin = call.get("metadata", {}).get(ROW_LINEAGE_KEY, {})
                if origin:
                    continue
                _fail("recovery_identity_mismatch")
            call_ids.add(logical)
            call_ids.update(
                stable_compiler_id("embedding-model-call-attempt", [call["run_id"], logical, n])
                for n in range(1, count)
            )
    rows["model_calls"] = call_rows
    job_ids = qjob_ids | {r["job_id"] for r in call_rows if r["job_id"]}
    rows["experiment_jobs"] = await _read_table(
        session, "experiment_jobs", jobs.c.job_id.in_(job_ids)
    )
    if {r["job_id"] for r in rows["experiment_jobs"]} != job_ids:
        _fail("job_ancestry_missing")
    attempts = _TABLES["job_attempts"]
    rows["job_attempts"] = await _read_table(
        session, "job_attempts", attempts.c.job_id.in_(job_ids)
    )
    artifact_ids = {r[k] for r in call_rows for k in ("input_asset_id", "output_asset_id") if r[k]}
    artifact_ids.update(
        r["metadata"]["error_artifact_id"]
        for r in call_rows
        if r.get("metadata", {}).get("error_artifact_id")
    )
    artifact_ids.update(
        r["error_artifact_id"]
        for name in ("experiment_jobs", "job_attempts")
        for r in rows[name]
        if r["error_artifact_id"]
    )
    artifacts = _TABLES["run_artifacts"]
    rows["run_artifacts"] = await _read_table(
        session, "run_artifacts", artifacts.c.artifact_id.in_(artifact_ids)
    )
    if {r["artifact_id"] for r in rows["run_artifacts"]} != artifact_ids:
        _fail("artifact_ancestry_missing")
    diagnostics = _TABLES["structured_output_item_diagnostics"]
    rows[diagnostics.name] = await _read_table(
        session, diagnostics.name, diagnostics.c.model_call_id.in_(call_ids)
    )
    rows = normalize_partial_prefix_rows(rows)
    validate_partial_prefix_rows(rows, run_id)
    from pmmc.infrastructure.database.repositories import PostgresMemoryRepository

    @asynccontextmanager
    async def same_snapshot():
        yield session

    memories = await PostgresMemoryRepository(cast(Any, same_snapshot)).list_snapshot(
        headers["compiler"]["snapshot_id"],
        conversation_ids=set(headers["compiler"]["metadata"]["conversation_ids"]),
    )
    policy = headers["manifest"]["metadata"]["questioner_config"].get(
        "memory_raw_image_policy", "allow"
    )
    # This helper checks memory inputs/bytes only; it does not require dry-run or R0.
    validate_r0_memory_vectors(rows, memories, memory_raw_image_policy=policy)
    headers["canonical_memories"] = [m.model_dump(mode="json") for m in memories]
    return headers, rows


def project_partial_prefix_rows(
    rows: Rows,
    *,
    proof: Mapping[str, Any],
    target_run_id: str,
    target_selection_hash: str,
    memory_raw_image_policy: str = "allow",
) -> tuple[Rows, dict[str, str]]:
    source_run_id = proof["source_run_id"]
    if source_run_id == target_run_id:
        _fail("same_owner")
    rows = normalize_partial_prefix_rows(rows)
    validate_partial_prefix_rows(rows, source_run_id)
    source_index = rows["memory_operator_index_manifests"][0]
    index, text_collection, visual_collection = make_memory_operator_index(
        **{
            k: source_index[k]
            for k in (
                "snapshot_id",
                "dataset",
                "text_model_alias",
                "text_model_revision",
                "multimodal_model_alias",
                "multimodal_model_revision",
            )
        },
        run_id=target_run_id,
        selection_hash=target_selection_hash,
        memory_raw_image_policy=memory_raw_image_policy,
    )
    projected, identities = remap_questioner_rows(
        rows,
        primary_keys=_KEYS,
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        proof_hash=canonical_content_hash(proof),
        identity_overrides={
            source_index["memory_index_id"]: index.memory_index_id,
            source_index["text_collection_id"]: text_collection.collection_id,
            source_index["visual_collection_id"]: visual_collection.collection_id,
        },
    )
    _rehash_questions(projected)
    for name, values in projected.items():
        for source, target in zip(rows[name], values, strict=True):
            if "run_id" in target:
                target["run_id"] = target_run_id
            for field in (
                "args",
                "output",
                "message",
                "error",
                "embedding",
                "provider_request_id",
                "litellm_call_id",
                "vllm_request_id",
                "langfuse_trace_id",
            ):
                if field in source:
                    target[field] = copy.deepcopy(source[field])
            if "metadata" in target:
                old = copy.deepcopy(source.get("metadata", {}))
                for key in ("upstream_import", "r0_prefix_import", ROW_LINEAGE_KEY):
                    target["metadata"].pop(key, None)
                target["metadata"][ROW_LINEAGE_KEY] = {
                    "version": ADOPTION_VERSION,
                    "source_run_id": source.get("run_id", source_run_id),
                    "source_primary_key": {k: source[k] for k in _KEYS[name]},
                    "source_row_hash": canonical_content_hash(source),
                    "source_metadata": old,
                    "proof_hash": canonical_content_hash(proof),
                }
            if name == "compiler_stage_jobs":
                target["claimed_by"] = target["lease_expires_at"] = None
            if "content_hash" in target and name != "questioner_candidate_gate_results":
                target["content_hash"] = canonical_content_hash(
                    {
                        "version": ADOPTION_VERSION,
                        "table": name,
                        "source_content_hash": source["content_hash"],
                        "target": {k: v for k, v in target.items() if k != "content_hash"},
                    }
                )
    projected["memory_operator_index_manifests"][0].update(
        memory_index_id=index.memory_index_id,
        selection_hash=target_selection_hash,
        content_hash=index.content_hash,
    )
    for collection in (text_collection, visual_collection):
        next(
            r
            for r in projected["embedding_collections"]
            if r["collection_id"] == collection.collection_id
        )["metadata"].update(collection.metadata)
    contracts = {r["qbank_id"]: r for r in projected["planner_requirement_contracts"]}
    for row in projected["memory_programs"]:
        if "requirement_contract_hash" in row["metadata"] and row["qbank_id"] in contracts:
            row["metadata"]["requirement_contract_hash"] = contracts[row["qbank_id"]][
                "content_hash"
            ]
        program = memory_program_from_rows(
            row,
            [r for r in projected["program_steps"] if r["plan_id"] == row["plan_id"]],
            [
                r
                for r in projected["program_expected_evidence_refs"]
                if r["plan_id"] == row["plan_id"]
            ],
        )
        row["program_hash"] = memory_program_content_hash(program)
    projected = normalize_partial_prefix_rows(projected)
    validate_partial_prefix_rows(projected, target_run_id)
    return projected, identities


def model_call_aliases(source: Rows, identities: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    return {
        identities[r["model_call_id"]]: {
            "source_run_id": r["run_id"],
            "source_model_call_id": r["model_call_id"],
            "source_row_hash": canonical_content_hash(r),
            "source_cache_hit": r["cache_hit"],
        }
        for r in source["model_calls"]
    }


class PostgresPartialPrefixImportRepository:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        store: ObjectStore,
        *,
        validate_proof: ProofValidator | None = None,
    ):
        self.sessions, self.store, self.validate_proof = sessions, store, validate_proof
        self.questioner = PostgresQuestionerImportRepository(sessions, store)

    async def _read_prefix(self, session: AsyncSession, run_id: str) -> tuple[dict[str, Any], Rows]:
        return await read_partial_prefix_rows(
            session, run_id, questioner_repository=self.questioner
        )

    async def read_source(self, run_id: str) -> tuple[dict[str, Any], Rows]:
        async with self.sessions() as session:
            await session.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            return await self._read_prefix(session, run_id)

    async def seal_prefix(self, *, run_id: str) -> dict[str, Any]:
        headers, rows = await self.read_source(run_id)
        return {
            "source_run_id": run_id,
            "table_contract_hash": partial_prefix_table_contract_hash(),
            "facts_hash": partial_prefix_facts_hash(headers, rows),
            "row_counts": {name: len(values) for name, values in rows.items()},
            "artifact_hashes": await self.questioner._artifacts(rows["run_artifacts"]),
        }

    async def seal(
        self,
        *,
        run_id: str,
        campaign_id: str,
        protocol_hash: str,
        compatibility_hash: str,
        target_campaign_id: str,
        target_protocol_hash: str,
        source_receipts: Mapping[str, str],
        source_scope: Mapping[str, Any],
    ) -> dict[str, Any]:
        headers, rows = await self.read_source(run_id)
        scope = dict(source_scope)
        if (
            headers["compiler"]["snapshot_id"] != scope.get("snapshot_id")
            or headers["manifest"]["dataset"] != scope.get("dataset")
            or headers["compiler"]["metadata"].get("conversation_ids")
            != [scope.get("conversation_id")]
        ):
            _fail("source_scope_mismatch")
        historical_q = headers["compiler"]["metadata"].get("questioner_upstream_import")
        if scope.get("treatment") == "fixed_planner" and not historical_q:
            _fail("fixed_planner_questioner_lineage_missing")
        proof = {
            "version": PROOF_VERSION,
            "source_campaign_id": campaign_id,
            "source_protocol_hash": protocol_hash,
            "source_run_id": run_id,
            "target_campaign_id": target_campaign_id,
            "target_protocol_hash": target_protocol_hash,
            "compatibility_hash": compatibility_hash,
            "schema_revision": STEP8_ALEMBIC_REVISION,
            "table_contract_hash": partial_prefix_table_contract_hash(),
            "facts_hash": partial_prefix_facts_hash(headers, rows),
            "row_counts": {name: len(values) for name, values in rows.items()},
            "artifact_hashes": await self.questioner._artifacts(rows["run_artifacts"]),
            "covered_stages": list(PARTIAL_STAGES),
            "source_receipts": dict(source_receipts),
            "source_scope": scope,
            "source_questioner_upstream_import": copy.deepcopy(historical_q),
        }
        await self._validate_proof(proof, compatibility_hash)
        return proof

    async def _validate_proof(self, proof: Mapping[str, Any], compatibility_hash: str) -> None:
        if (
            proof.get("version") != PROOF_VERSION
            or proof.get("compatibility_hash") != compatibility_hash
            or not compatibility_hash
            or proof.get("covered_stages") != list(PARTIAL_STAGES)
            or proof.get("table_contract_hash") != partial_prefix_table_contract_hash()
            or proof.get("schema_revision", STEP8_ALEMBIC_REVISION) != STEP8_ALEMBIC_REVISION
            or not set(PARTIAL_STAGES) <= set(proof.get("source_receipts", {}))
        ):
            _fail("proof_contract_mismatch")
        if self.validate_proof is not None:
            await self.validate_proof(copy.deepcopy(dict(proof)))

    def _match_source(self, proof: Mapping[str, Any], headers: dict[str, Any], rows: Rows) -> None:
        if (
            partial_prefix_facts_hash(headers, rows) != proof["facts_hash"]
            or {name: len(values) for name, values in rows.items()} != proof["row_counts"]
        ):
            _fail("source_facts_changed")

    async def prefix_registration(self, run_id: str) -> dict[str, Any] | None:
        async with self.sessions() as session:
            row = await session.get(c.CompilerRunModel, run_id)
            return copy.deepcopy(row.metadata_json.get(LINEAGE_KEY)) if row else None

    async def adopt(
        self,
        *,
        proof: Mapping[str, Any],
        compatibility_hash: str,
        target_run_id: str,
        compiler_updates: Mapping[str, Any],
        manifest_metadata: Mapping[str, Any],
        prompt_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        """Atomically register and append the complete prefix, or verify exact recovery."""
        proof = copy.deepcopy(dict(proof))
        await self._validate_proof(proof, compatibility_hash)
        if proof["source_run_id"] == target_run_id:
            _fail("same_owner")
        existing = await self.prefix_registration(target_run_id)
        if existing is not None:
            if existing["proof_hash"] != canonical_content_hash(proof):
                _fail("target_lineage_conflict")
            verified = await self.verify_adopted(target_run_id, compatibility_hash)
            assert verified is not None
            return {**verified, "reused": True}
        headers, rows = await self.read_source(proof["source_run_id"])
        self._match_source(proof, headers, rows)
        if await self.questioner._artifacts(rows["run_artifacts"]) != proof["artifact_hashes"]:
            _fail("artifact_proof_mismatch")
        # Unit identities are the native target-run identities, not copied source IDs.
        from pmmc.compiler.ids import compilation_unit_instance_id

        unit_ids = [
            compilation_unit_instance_id(run_id=target_run_id, input_hash=r["input_hash"])
            for r in rows["compiler_units"]
        ]
        selection_hash = canonical_content_hash(
            {
                "snapshot_id": headers["compiler"]["snapshot_id"],
                "conversation_ids": headers["compiler"]["metadata"]["conversation_ids"],
                "unit_ids": unit_ids,
                "seed": headers["manifest"]["seed"],
                "shard_count": 1,
                "shard_index": 0,
                "split": headers["manifest"]["split"],
                "diagnostic_force_revision": False,
            }
        )
        policy = headers["manifest"]["metadata"]["questioner_config"].get(
            "memory_raw_image_policy", "allow"
        )
        projected, identities = project_partial_prefix_rows(
            rows,
            proof=proof,
            target_run_id=target_run_id,
            target_selection_hash=selection_hash,
            memory_raw_image_policy=policy,
        )
        forbidden = {"run_id", "snapshot_id", "metadata", "unit_count"}
        if forbidden & compiler_updates.keys() or LINEAGE_KEY in manifest_metadata:
            _fail("reserved_registration_override")

        def clean_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
            return {
                k: copy.deepcopy(v)
                for k, v in value.items()
                if k not in {"questioner_upstream_import", "r0_prefix_upstream_import", LINEAGE_KEY}
            }

        manifest = RunManifest.model_validate(
            {
                **headers["manifest"],
                "run_id": target_run_id,
                "metadata": {
                    **clean_metadata(headers["manifest"]["metadata"]),
                    **manifest_metadata,
                },
                "prompt_hashes": dict(prompt_hashes),
                "selection_hash": selection_hash,
                "created_at": utc_now(),
                "config_hash": None,
            }
        )
        compiler = {
            **headers["compiler"],
            **compiler_updates,
            "run_id": target_run_id,
            "created_at": manifest.created_at,
            "metadata": {
                **clean_metadata(headers["compiler"]["metadata"]),
                **manifest_metadata,
                "selection_hash": selection_hash,
            },
        }
        lineage = {
            "version": ADOPTION_VERSION,
            "proof_json": proof,
            "proof_hash": canonical_content_hash(proof),
            "source_run_id": proof["source_run_id"],
            "source_campaign_id": proof["source_campaign_id"],
            "source_protocol_hash": proof["source_protocol_hash"],
            "target_campaign_id": proof.get("target_campaign_id"),
            "target_protocol_hash": proof.get("target_protocol_hash"),
            "source_facts_hash": proof["facts_hash"],
            "target_rows_hash": partial_prefix_rows_hash(projected),
            "identity_map_hash": canonical_content_hash(identities),
            "target_selection_hash": selection_hash,
            "compatibility_hash": compatibility_hash,
            "imported_stages": list(PARTIAL_STAGES),
            "model_call_aliases": model_call_aliases(rows, identities),
            "provider_calls": 0,
            "unit_count": len(projected["compiler_units"]),
            "eligible_count": sum(
                r["status"] == "eligible_for_planning"
                for r in projected["questioner_candidate_gate_results"]
            ),
            "program_count": len(projected["memory_programs"]),
        }
        lineage["target_registration_hash"] = canonical_content_hash(
            _stable_headers({"compiler": compiler, "manifest": manifest.model_dump(mode="json")})
        )
        compiler["metadata"][LINEAGE_KEY] = lineage
        manifest = manifest.model_copy(
            update={"metadata": {**manifest.metadata, LINEAGE_KEY: lineage}}
        )
        manifest = manifest.model_copy(update={"config_hash": run_config_hash(manifest)})
        async with self.sessions() as session, session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:identity, 0))"),
                {"identity": target_run_id},
            )
            current_headers, current_rows = await self._read_prefix(session, proof["source_run_id"])
            self._match_source(proof, current_headers, current_rows)
            if (
                await session.get(c.CompilerRunModel, target_run_id) is not None
                or await session.get(RunManifestModel, target_run_id) is not None
            ):
                _fail("target_already_registered")
            await PostgresRunRepository(session).register(manifest, [])
            await session.execute(
                insert(cast(Table, c.CompilerRunModel.__table__)).values(compiler)
            )
            for table in PARTIAL_PREFIX_TABLES:
                for row in projected[table.name]:
                    values = (
                        {**row, "status": "building"}
                        if table.name == "embedding_collections"
                        else row
                    )
                    await session.execute(insert(table).values(values))
                if table.name == "embeddings":
                    # Publish even empty collections only after all vector writes.
                    # The enclosing transaction also covers dependent index manifests.
                    collections = _TABLES["embedding_collections"]
                    for row in projected["embedding_collections"]:
                        await session.execute(
                            update(collections)
                            .where(collections.c.collection_id == row["collection_id"])
                            .values(status="ready")
                        )
            await PostgresRunRepository(session).transition(target_run_id, RunStatus.RUNNING)
            await session.flush()
            _, actual = await self._read_prefix(session, target_run_id)
            if partial_prefix_rows_hash(actual) != lineage["target_rows_hash"]:
                _fail("post_insert_fact_conflict")
        return {**lineage, "reused": False}

    async def verify_adopted(self, run_id: str, compatibility_hash: str) -> dict[str, Any] | None:
        """Read-only recovery verification; later execution/D tables are not consulted."""
        lineage = await self.prefix_registration(run_id)
        if lineage is None:
            return None
        proof = lineage["proof_json"]
        await self._validate_proof(proof, compatibility_hash)
        if (
            lineage.get("version") != ADOPTION_VERSION
            or lineage.get("proof_hash") != canonical_content_hash(proof)
            or lineage.get("compatibility_hash") != compatibility_hash
            or lineage.get("provider_calls") != 0
            or lineage.get("imported_stages") != list(PARTIAL_STAGES)
        ):
            _fail("target_lineage_conflict")
        async with self.sessions() as session:
            await session.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            source_headers, source_rows = await self._read_prefix(session, proof["source_run_id"])
            self._match_source(proof, source_headers, source_rows)
            headers, actual = await self._read_prefix(session, run_id)
        if headers["manifest"]["metadata"].get(LINEAGE_KEY) != lineage:
            _fail("manifest_lineage_conflict")
        registration = {k: headers[k] for k in ("compiler", "manifest")}
        if (
            canonical_content_hash(_stable_headers(registration))
            != lineage["target_registration_hash"]
        ):
            _fail("target_registration_changed")
        policy = source_headers["manifest"]["metadata"]["questioner_config"].get(
            "memory_raw_image_policy", "allow"
        )
        expected, identities = project_partial_prefix_rows(
            source_rows,
            proof=proof,
            target_run_id=run_id,
            target_selection_hash=lineage["target_selection_hash"],
            memory_raw_image_policy=policy,
        )
        if (
            partial_prefix_rows_hash(expected) != lineage["target_rows_hash"]
            or partial_prefix_rows_hash(actual) != lineage["target_rows_hash"]
            or canonical_content_hash(identities) != lineage["identity_map_hash"]
            or model_call_aliases(source_rows, identities) != lineage["model_call_aliases"]
        ):
            _fail("target_fact_or_alias_conflict")
        if (
            await self.questioner._artifacts(source_rows["run_artifacts"])
            != proof["artifact_hashes"]
        ):
            _fail("artifact_proof_mismatch")
        await self.questioner._artifacts(actual["run_artifacts"])
        return lineage
