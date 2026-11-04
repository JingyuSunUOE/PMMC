"""Read-only proof of RAG index model-call dependencies, never price estimates.

The inventory calls the collector inside its evaluator repeatable-read snapshot.
Whole collection membership is read independently of retrieval hits. A completed
run alone is not proof: batches, document hashes, vectors and calls must close.
Retries remain unproven until attempt-level dispatch journals are available.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from pmmc.baselines.rag_config import RagBaselineConfig
from pmmc.compiler.ids import canonical_content_hash
from pmmc.infrastructure.database import baseline_models as blm
from pmmc.infrastructure.database import experiment_models as xm
from pmmc.infrastructure.database import models as bm
from pmmc.schemas import BaselineMethod

RAG_COST_DEPENDENCIES_VERSION = "step8-rag-index-cost-dependencies-v1"
RAG_COST_DEPENDENCIES_ARTIFACT = "cost-dependencies:rag-indexes"
CALL_FIELDS = (
    "model_call_id",
    "run_id",
    "snapshot_id",
    "query_id",
    "job_id",
    "stage",
    "status",
    "model_alias",
    "actual_model_id",
    "parent_model_call_id",
    "retry_count",
    "cache_hit",
)


class DependencyUnproven(ValueError):
    """A safe reason code, without SQL, document text or exception parameters."""


def load_dependency_artifact(snapshot, identity: str, version: str) -> dict:
    """Bind dependency facts to provenance, not just a self-described snapshot ID."""
    ref = snapshot.artifacts.get(identity)
    bindings = snapshot.source.get("dependency_artifact_hashes", {})
    if (
        ref is None
        or ref.scope != "evaluation"
        or ref.kind != "performance"
        or not isinstance(bindings, dict)
        or bindings.get(identity) != ref.sha256
    ):
        raise DependencyUnproven("dependency_artifact_not_bound_to_source_snapshot")
    value = snapshot.artifact(identity)
    if (
        value.get("schema_version") != version
        or value.get("campaign_id") != snapshot.inventory.campaign_id
        or value.get("selection_hash") != snapshot.inventory.selection_hash
        or value.get("source_role") != "evaluator"
        or value.get("database_snapshot") != snapshot.source["database_snapshot"]
    ):
        raise DependencyUnproven("dependency_artifact_snapshot_mismatch")
    return value


def _unique(rows, key):
    result = {}
    for row in rows:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise DependencyUnproven("index_membership_identity_invalid")
        result[identity] = row
    return result


def validate_rag_index_dependencies(facts: dict[str, Any]) -> tuple[str, ...]:
    """Return all producer call IDs only after complete, first-attempt closure."""
    collection, run, state, corpus = (facts[k] for k in ("collection", "run", "state", "corpus"))
    cid, rid, snapshot = (collection[k] for k in ("collection_id", "run_id", "snapshot_id"))
    manifest = run["manifest_json"]
    metadata = manifest.get("metadata", {})
    if (
        not rid
        or run["run_id"] != rid
        or state["run_id"] != rid
        or state["status"] != "completed"
        or collection["status"] != "ready"
        or corpus["status"] != "ready"
        or corpus["corpus_id"] != collection["corpus_id"]
        or corpus["snapshot_id"] != snapshot
        or manifest.get("dataset_snapshot_ids") != [snapshot]
        or run["experiment_name"] != "baseline_index_build"
        or metadata.get("index_build") is not True
        or metadata.get("corpus_content_hash") != corpus["content_hash"]
        or manifest.get("retrieval_config", {}).get("corpus_id") != corpus["corpus_id"]
    ):
        raise DependencyUnproven("index_run_or_collection_not_closed")
    method = BaselineMethod(run["method"].removesuffix("_index"))
    if run["method"] != method.value + "_index":
        raise DependencyUnproven("index_method_unproven")
    config = RagBaselineConfig.model_validate(metadata["rag_config"])
    representations = {v.value for v in config.representations_for(method)}
    if (
        set(collection["metadata"].get("representations", [])) != representations
        or collection["model_alias"] != config.embedding_alias_for(method)
        or not isinstance(collection["actual_model_id"], str)
        or not collection["actual_model_id"]
        or type(collection["dimension"]) is not int
        or collection["dimension"] <= 0
    ):
        raise DependencyUnproven("index_representation_mismatch")
    documents = _unique(facts["documents"], "document_id")
    vectors = _unique(facts["vectors"], "source_id")
    jobs = _unique(facts["jobs"], "job_id")
    calls = _unique(facts["calls"], "model_call_id")
    if not documents or set(vectors) != set(documents) or not jobs:
        raise DependencyUnproven("index_document_vector_membership_incomplete")
    if len({v["embedding_id"] for v in vectors.values()}) != len(vectors):
        raise DependencyUnproven("index_embedding_identity_duplicate")
    seen: set[str] = set()
    ordinals: set[int] = set()
    producer_ids: set[str] = set()
    for job in jobs.values():
        jm = job["metadata"]
        ordinal = jm.get("batch_ordinal")
        members = jm.get("document_ids", [])
        if (
            job["run_id"] != rid
            or job["snapshot_id"] != snapshot
            or job.get("query_id") is not None
            or job["stage"] != "embed"
            or job["status"] != "succeeded"
            or jm.get("collection_id") != cid
            or type(ordinal) is not int
            or ordinal < 0
            or ordinal in ordinals
            or not isinstance(members, list)
            or not members
            or any(not isinstance(v, str) for v in members)
            or len(set(members)) != len(members)
            or seen.intersection(members)
            or not set(members) <= documents.keys()
        ):
            raise DependencyUnproven("index_batch_membership_unproven")
        if type(job["attempt_count"]) is not int or job["attempt_count"] != 1:
            raise DependencyUnproven("index_retry_attempt_dependencies_unproven")
        ordinals.add(ordinal)
        seen.update(members)
        owned = [c for c in calls.values() if c.get("job_id") == job["job_id"]]
        if len(owned) != 1:
            raise DependencyUnproven("index_batch_call_denominator_unproven")
        call = owned[0]
        if (
            call["run_id"] != rid
            or call["snapshot_id"] != snapshot
            or call.get("query_id") is not None
            or call["stage"] != "embed"
            or call["status"] != "success"
            or call.get("parent_model_call_id")
            or type(call.get("retry_count")) is not int
            or call["retry_count"] != 0
            or call.get("cache_hit") is not False
            or call["model_alias"] != collection["model_alias"]
            or call["actual_model_id"] != collection["actual_model_id"]
        ):
            raise DependencyUnproven("index_call_scope_or_attempt_unproven")
        producer_ids.add(call["model_call_id"])
        for identity in members:
            doc, vector = documents[identity], vectors[identity]
            if (
                doc["snapshot_id"] != snapshot
                or doc["corpus_id"] != corpus["corpus_id"]
                or doc["representation"] not in representations
                or vector["collection_id"] != cid
                or vector["retrieval_document_id"] != identity
                or vector["input_hash"] != doc["content_hash"]
                or vector["model_call_id"] != call["model_call_id"]
                or vector["dimension"] != collection["dimension"]
            ):
                raise DependencyUnproven("index_vector_producer_or_content_mismatch")
    if (
        seen != documents.keys()
        or ordinals != set(range(len(jobs)))
        or producer_ids != calls.keys()
    ):
        raise DependencyUnproven("index_complete_producer_membership_unproven")
    return tuple(sorted(producer_ids))


async def collect_rag_index_cost_dependencies(
    session, *, collection_ids, campaign_id: str, selection_hash: str, database_snapshot: dict
) -> dict[str, Any]:
    """Collect in caller's read-only transaction; no new connection or writes.

    The returned dependency IDs must additionally be exported through the normal
    inventory model-call/artifact path. Private membership contains IDs/hashes,
    but no document text, image locations, vector values or connection data.
    """

    async def read(model, names, condition):
        table = model.__table__
        statement = select(*(table.c[name] for name in names)).where(condition)
        statement = statement.order_by(*table.primary_key.columns).execution_options(
            step8_cost_dependency_table=table.fullname, step8_inventory_table=table.fullname
        )
        result = await session.execute(statement)
        return [dict(row) for row in result.mappings().all()]

    entries: list[dict[str, Any]] = []
    dependency_calls: set[str] = set()
    dependency_runs: set[str] = set()
    for cid in sorted(set(collection_ids)):
        facts: dict[str, Any] = {}
        reason = None
        try:
            async with session.begin_nested():
                collections = await read(
                    xm.EmbeddingCollectionModel,
                    (
                        "collection_id",
                        "run_id",
                        "snapshot_id",
                        "corpus_id",
                        "model_alias",
                        "actual_model_id",
                        "dimension",
                        "status",
                        "metadata",
                    ),
                    xm.EmbeddingCollectionModel.collection_id == cid,
                )
                if len(collections) != 1:
                    raise DependencyUnproven("index_collection_missing")
                collection = facts["collection"] = collections[0]
                rid, corpus_id = collection["run_id"], collection["corpus_id"]
                if not rid or not corpus_id:
                    raise DependencyUnproven("index_collection_owner_missing")
                for key, model, fields, condition in (
                    (
                        "run",
                        bm.RunManifestModel,
                        ("run_id", "experiment_name", "method", "manifest_json"),
                        bm.RunManifestModel.run_id == rid,
                    ),
                    (
                        "state",
                        xm.RunStateModel,
                        ("run_id", "status"),
                        xm.RunStateModel.run_id == rid,
                    ),
                    (
                        "corpus",
                        blm.RetrievalCorpusModel,
                        ("corpus_id", "snapshot_id", "content_hash", "status"),
                        blm.RetrievalCorpusModel.corpus_id == corpus_id,
                    ),
                ):
                    rows = await read(model, fields, condition)
                    if len(rows) != 1:
                        raise DependencyUnproven("index_producer_state_missing")
                    facts[key] = rows[0]
                facts["documents"] = await read(
                    blm.RetrievalDocumentModel,
                    ("document_id", "corpus_id", "snapshot_id", "representation", "content_hash"),
                    (blm.RetrievalDocumentModel.corpus_id == corpus_id)
                    & blm.RetrievalDocumentModel.representation.in_(
                        collection["metadata"].get("representations", [])
                    ),
                )
                facts["vectors"] = await read(
                    xm.EmbeddingModel,
                    (
                        "embedding_id",
                        "collection_id",
                        "source_id",
                        "retrieval_document_id",
                        "input_hash",
                        "model_call_id",
                        "dimension",
                    ),
                    xm.EmbeddingModel.collection_id == cid,
                )
                facts["jobs"] = await read(
                    xm.ExperimentJobModel,
                    (
                        "job_id",
                        "run_id",
                        "snapshot_id",
                        "query_id",
                        "stage",
                        "status",
                        "attempt_count",
                        "metadata",
                    ),
                    xm.ExperimentJobModel.run_id == rid,
                )
                facts["calls"] = await read(
                    xm.RuntimeModelCallModel, CALL_FIELDS, xm.RuntimeModelCallModel.run_id == rid
                )
                # Include known costs even for an unclosed index; closure stays null.
                dependency_runs.add(rid)
                dependency_calls.update(c["model_call_id"] for c in facts["calls"])
                validate_rag_index_dependencies(facts)
        except DependencyUnproven as exc:
            reason = str(exc)
        except (SQLAlchemyError, ValueError, KeyError, TypeError, AttributeError):
            reason = "index_dependency_source_unavailable_or_invalid"
        entries.append(
            {
                "collection_id": cid,
                "status": "incomplete" if reason else "complete",
                "reasons": [reason] if reason else [],
                "facts": facts,
                "facts_hash": canonical_content_hash(facts),
            }
        )
    return {
        "schema_version": RAG_COST_DEPENDENCIES_VERSION,
        "campaign_id": campaign_id,
        "selection_hash": selection_hash,
        "source_role": "evaluator",
        "database_snapshot": database_snapshot,
        "collections": entries,
        "dependency_model_call_ids": sorted(dependency_calls),
        "dependency_run_ids": sorted(dependency_runs),
    }
