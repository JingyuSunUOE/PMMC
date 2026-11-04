from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from pmmc.baselines.rag_config import RagBaselineConfig
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_cost_analysis import RUNTIME, summarize_campaign_costs
from pmmc.experiments.step8_cost_dependencies import (
    RAG_COST_DEPENDENCIES_ARTIFACT,
    RAG_COST_DEPENDENCIES_VERSION,
    DependencyUnproven,
    collect_rag_index_cost_dependencies,
    validate_rag_index_dependencies,
)
from pmmc.experiments.step8_questioner_cost_dependencies import (
    QUESTIONER_COST_DEPENDENCIES_ARTIFACT,
    QUESTIONER_COST_DEPENDENCIES_VERSION,
    _sealed_json,
    collect_questioner_cost_dependencies,
    validate_questioner_cost_dependencies,
)
from pmmc.experiments.step8_upstream_import import IMPORT_VERSION, remap_questioner_rows
from pmmc.infrastructure.database.step8_upstream_import import (
    QUESTIONER_TABLES,
    PostgresQuestionerImportRepository,
    _sort_rows,
)
from pmmc.schemas import BaselineMethod
from test_step8_cost_analysis import Fixture


def adoption_graph(source_calls, *, target="adopted", source_graph=None):
    source_id = source_calls[0]["run_id"]
    source = source_graph or {
        "headers": {
            "compiler": {
                "run_id": source_id,
                "snapshot_id": "snapshot",
                "metadata": {},
                "compiler_schema_version": "0.9.0",
                "unit_count": 1,
            },
            "manifest": {"run_id": source_id, "metadata": {}},
        },
        "rows": {t.name: [] for t in QUESTIONER_TABLES},
    }
    source = deepcopy(source)
    if source_graph is None:
        source["rows"]["compiler_units"] = [
            {
                "compiler_unit_id": "unit",
                "run_id": source_id,
                "input_hash": "a" * 64,
                "metadata": {},
            }
        ]
        source["rows"]["compiler_unit_stage_results"] = [
            {
                "stage_result_id": "stage",
                "compiler_unit_id": "unit",
                "run_id": source_id,
                "stage": "questioner",
                "status": "success",
                "metadata": {},
            }
        ]
    source["rows"]["model_calls"] = deepcopy(source_calls)
    rebound, identities = remap_questioner_rows(
        source["rows"],
        primary_keys={t.name: [k.name for k in t.primary_key] for t in QUESTIONER_TABLES},
        source_run_id=source_id,
        target_run_id=target,
        proof_hash="f" * 64,
    )
    rebound = {t.name: _sort_rows(t, rebound[t.name]) for t in QUESTIONER_TABLES}
    lineage = {
        "version": IMPORT_VERSION,
        "source_run_id": source_id,
        "proof_hash": "f" * 64,
        "provider_calls": 0,
        "imported_stages": ["questioner"],
        "source_facts_hash": canonical_content_hash(source),
        "target_rows_hash": canonical_content_hash(rebound),
        "identity_map_hash": canonical_content_hash(identities),
    }
    target_graph = {
        "headers": {
            "compiler": {
                "run_id": target,
                "snapshot_id": "snapshot",
                "compiler_schema_version": "0.9.0",
                "unit_count": 1,
                "metadata": {"questioner_upstream_import": lineage},
            },
            "manifest": {"run_id": target, "metadata": {"questioner_upstream_import": lineage}},
        },
        "rows": rebound,
    }
    return {"source": source, "target": target_graph}


def bind_imports(case, graphs):
    calls = case.tables[RUNTIME + "model_calls"]
    by_id = {c["model_call_id"]: c for c in calls}
    external, runs = set(), set()
    for graph in graphs:
        for side in ("source", "target"):
            by_id.update((c["model_call_id"], c) for c in graph[side]["rows"]["model_calls"])
        external.update(c["model_call_id"] for c in graph["source"]["rows"]["model_calls"])
        runs.add(graph["source"]["headers"]["compiler"]["run_id"])
    case.tables[RUNTIME + "model_calls"] = list(by_id.values())
    proof = {
        "schema_version": QUESTIONER_COST_DEPENDENCIES_VERSION,
        "campaign_id": case.facts.manifest.campaign_id,
        "selection_hash": case.facts.selection.content_hash,
        "source_role": "evaluator",
        "database_snapshot": {"database_role": "pmmc_evaluator", "snapshot": "1:2:"},
        "imports": [
            {
                "target_run_id": g["target"]["headers"]["compiler"]["run_id"],
                "source_run_id": g["source"]["headers"]["compiler"]["run_id"],
                "status": "complete",
                "reasons": [],
                "facts": g,
                "facts_hash": canonical_content_hash(g),
            }
            for g in graphs
        ],
        "dependency_model_call_ids": sorted(external),
        "dependency_run_ids": sorted(runs),
    }
    ref = case.artifact(
        QUESTIONER_COST_DEPENDENCIES_ARTIFACT, proof, kind="performance", scope="evaluation"
    )
    updates = {
        "dependency_model_call_ids": sorted(external),
        "dependency_run_ids": sorted(runs),
        "dependency_artifact_hashes": {ref.artifact_id: ref.sha256},
    }
    case.seal(source_updates=updates)
    return updates


def source_call(case, identity="q-call", **kwargs):
    return case.call(
        identity,
        run_id="q-source",
        query_id=None,
        snapshot_id="snapshot",
        stage="questioner",
        cache_hit=False,
        metadata={},
        **kwargs,
    )


def test_import_aliases_are_external_once_even_without_paid_settlement(tmp_path):
    case = Fixture(tmp_path)
    original = source_call(case)
    graph = adoption_graph([original])
    graph2 = adoption_graph(
        graph["target"]["rows"]["model_calls"], target="second", source_graph=graph["target"]
    )
    bind_imports(case, [graph, graph2])
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["count"] == 0
    reused = result["reused_dependencies"]["totals"]
    assert reused["count"] == 1 and reused["unknown_usd_count"] == 1
    assert reused["cost_usd"] is None
    assert reused["resources"][0]["total"] == 10


def test_import_of_current_paid_source_does_not_hide_or_duplicate_spend(tmp_path):
    case = Fixture(tmp_path)
    original = source_call(case, request="source-request")
    case.settlement("source-request", amount="3.50")
    bind_imports(case, [adoption_graph([original])])
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["count"] == 1
    assert result["actual_research"]["totals"]["cost_usd"] == "3.50"
    assert result["reused_dependencies"]["totals"]["count"] == 0


def test_import_seal_datetime_encoding_matches_inventory_export(tmp_path):
    case = Fixture(tmp_path)
    original = source_call(case, created_at=datetime(2026, 9, 25, tzinfo=UTC))
    graph = _sealed_json(adoption_graph([original]))
    updates = bind_imports(case, [graph])
    for row in case.tables[RUNTIME + "model_calls"]:
        row["created_at"] = "2026-09-25T00:00:00Z"
    case.seal(source_updates=updates)
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["count"] == 0
    assert result["reused_dependencies"]["totals"]["count"] == 1


@pytest.mark.parametrize("tamper", ["unbound", "tokens", "scope", "hash", "missing", "false_cache"])
def test_import_proof_fails_closed_without_exposing_payloads(tmp_path, tamper):
    case = Fixture(tmp_path)
    original = source_call(case)
    graph = adoption_graph([original])
    updates = bind_imports(case, [graph])
    if tamper == "unbound":
        updates["dependency_artifact_hashes"] = {}
    elif tamper == "tokens":
        case.tables[RUNTIME + "model_calls"][-1]["input_tokens"] = 99
    elif tamper == "scope":
        graph["source"]["rows"]["model_calls"][0]["query_id"] = "PRIVATE-query"
        bind_imports(case, [graph])
    elif tamper == "hash":
        graph["target"]["headers"]["compiler"]["metadata"]["questioner_upstream_import"][
            "source_facts_hash"
        ] = "b" * 64
        bind_imports(case, [graph])
    elif tamper == "missing":
        del case.artifacts[QUESTIONER_COST_DEPENDENCIES_ARTIFACT]
    elif tamper == "false_cache":
        case.tables[RUNTIME + "model_calls"][-1]["cache_hit"] = False
    if tamper not in {"scope", "hash"}:
        case.seal(source_updates=updates)
    result = summarize_campaign_costs(case.facts)
    assert result["status"] == "incomplete"
    assert result["actual_research"]["totals"] is None
    assert "PRIVATE-query" not in str(result)


def rag_graph(*, collection="index", snapshot="snapshot", method="caption_text_rag"):
    config = RagBaselineConfig()
    reps = [v.value for v in config.representations_for(BaselineMethod(method))]
    alias = config.embedding_alias_for(BaselineMethod(method))
    run = "build-" + collection
    return {
        "collection": {
            "collection_id": collection,
            "snapshot_id": snapshot,
            "run_id": run,
            "corpus_id": "corpus",
            "status": "ready",
            "dimension": 2,
            "model_alias": alias,
            "actual_model_id": "commit",
            "metadata": {"representations": reps},
        },
        "run": {
            "run_id": run,
            "experiment_name": "baseline_index_build",
            "method": method + "_index",
            "manifest_json": {
                "dataset_snapshot_ids": [snapshot],
                "retrieval_config": {"corpus_id": "corpus"},
                "metadata": {
                    "index_build": True,
                    "corpus_content_hash": "corpus-hash",
                    "rag_config": config.model_dump(mode="json"),
                },
            },
        },
        "state": {"run_id": run, "status": "completed"},
        "corpus": {
            "corpus_id": "corpus",
            "snapshot_id": snapshot,
            "status": "ready",
            "content_hash": "corpus-hash",
        },
        "documents": [
            {
                "document_id": "doc",
                "snapshot_id": snapshot,
                "corpus_id": "corpus",
                "representation": reps[0],
                "content_hash": "doc-hash",
            }
        ],
        "vectors": [
            {
                "embedding_id": "vec",
                "source_id": "doc",
                "collection_id": collection,
                "retrieval_document_id": "doc",
                "input_hash": "doc-hash",
                "model_call_id": "call-" + run,
                "dimension": 2,
            }
        ],
        "jobs": [
            {
                "job_id": "job",
                "run_id": run,
                "snapshot_id": snapshot,
                "query_id": None,
                "stage": "embed",
                "status": "succeeded",
                "attempt_count": 1,
                "metadata": {
                    "collection_id": collection,
                    "batch_ordinal": 0,
                    "document_ids": ["doc"],
                },
            }
        ],
        "calls": [
            {
                "model_call_id": "call-" + run,
                "run_id": run,
                "snapshot_id": snapshot,
                "query_id": None,
                "job_id": "job",
                "stage": "embed",
                "status": "success",
                "model_alias": alias,
                "actual_model_id": "commit",
                "parent_model_call_id": None,
                "retry_count": 0,
                "cache_hit": False,
            }
        ],
    }


@pytest.mark.parametrize("tamper", [None, "retry", "document", "call", "hash", "ordinal"])
def test_rag_closure_requires_whole_membership_not_hits_or_completed_flag(tamper):
    facts = rag_graph()
    if tamper == "retry":
        facts["jobs"][0]["attempt_count"] = 2
    elif tamper == "document":
        facts["documents"].append({**facts["documents"][0], "document_id": "unretrieved"})
    elif tamper == "call":
        facts["calls"].append({**facts["calls"][0], "model_call_id": "hidden-failed-attempt"})
    elif tamper == "hash":
        facts["vectors"][0]["input_hash"] = "stale"
    elif tamper == "ordinal":
        facts["jobs"][0]["metadata"]["batch_ordinal"] = True
    if tamper is None:
        assert validate_rag_index_dependencies(facts) == ("call-build-index",)
    else:
        with pytest.raises(DependencyUnproven):
            validate_rag_index_dependencies(facts)


@pytest.mark.parametrize("priced", [True, False])
def test_shared_rag_index_enables_real_amortization_without_multiplying_write_cost(
    tmp_path, priced
):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    case.restrict_methods({"caption_text_rag"})
    entries, calls = [], []
    for dataset, snapshot in case.facts.manifest.snapshot_ids.items():
        graph = rag_graph(collection="shared-" + dataset, snapshot=snapshot)
        trace = case.call(
            **{
                "identity": graph["calls"][0]["model_call_id"],
                **{k: v for k, v in graph["calls"][0].items() if k != "model_call_id"},
            }
        )
        if priced:
            case.price(trace, amount="6.00")
        calls.append(trace["model_call_id"])
        for batch in case.tables[RUNTIME + "retrieval_batches"]:
            if batch["snapshot_id"] == snapshot:
                batch["document_collection_id"] = graph["collection"]["collection_id"]
        entries.append(
            {
                "collection_id": graph["collection"]["collection_id"],
                "status": "complete",
                "reasons": [],
                "facts": graph,
                "facts_hash": canonical_content_hash(graph),
            }
        )
    for row in case.tables[RUNTIME + "model_calls"]:
        if row["model_call_id"] not in calls:
            case.price(row)
    payload = {
        "schema_version": RAG_COST_DEPENDENCIES_VERSION,
        "campaign_id": case.facts.manifest.campaign_id,
        "selection_hash": case.facts.selection.content_hash,
        "source_role": "evaluator",
        "database_snapshot": {"database_role": "pmmc_evaluator", "snapshot": "1:2:"},
        "collections": entries,
    }
    ref = case.artifact(
        RAG_COST_DEPENDENCIES_ARTIFACT, payload, kind="performance", scope="evaluation"
    )
    case.seal(source_updates={"dependency_artifact_hashes": {ref.artifact_id: ref.sha256}})
    result = summarize_campaign_costs(case.facts)
    for cell in result["active_path"]["cells"]:
        assert cell["write"]["status"] == "complete", cell["write"]
        assert cell["write"]["totals"]["count"] == 1
        assert cell["write"]["totals"]["cost_usd"] == ("6.00" if priced else None)
    assert all((v["cost_usd"] is not None) == priced for v in result["amortized"])
    for cost in result["active_path"]["logical_ablation_costs"]:
        # Nine query-role uses plus one lifecycle write, not three writes.
        assert cost["totals"]["count"] == 10
        assert cost["totals"]["cost_usd"] == ("17.25" if priced else None)


def add_performance(case, updates):
    for scope in ("runtime", "evaluation"):
        case.artifact(
            "model-call-performance:" + scope,
            {
                "calls": [
                    {
                        "model_call_id": row["model_call_id"],
                        "run_id": row["run_id"],
                        "latency_ms": row.get("latency_ms"),
                        "latency_ms_semantics": "excludes_queue_and_artifact_persistence",
                        **{
                            k: row["request_config"].get(k)
                            for k in ("queue_wait_ms", "artifact_persistence_ms", "end_to_end_ms")
                        },
                    }
                    for row in case.tables["pmmc_" + scope + ".model_calls"]
                ]
            },
            kind="performance",
            scope=scope,
        )
    case.seal(source_updates=updates)


def timed():
    return {
        "latency_ms": 100,
        "total_tokens": 12,
        "request_config": {"queue_wait_ms": 2, "artifact_persistence_ms": 3, "end_to_end_ms": 105},
    }


def test_performance_excludes_import_aliases_and_partitions_historical_producers(tmp_path):
    from pmmc.experiments.step8_performance_analysis import summarize_campaign_performance

    case = Fixture(tmp_path)
    original = source_call(case, **timed())
    updates = bind_imports(case, [adoption_graph([original])])
    add_performance(case, updates)
    groups = summarize_campaign_performance(case.facts)["model_calls"]["runtime"]
    assert groups["campaign"]["call_count"] == 0
    assert groups["reused_dependencies"]["call_count"] == 1
    assert groups["reused_dependencies"]["tokens"]["input_tokens"]["sum"] == 10


def test_performance_counts_verified_dispatch_replay_once(tmp_path):
    from pmmc.experiments.step8_performance_analysis import summarize_campaign_performance

    case = Fixture(tmp_path)
    case.call("original", request="request", cache_hit=False, **timed())
    case.call("replayed", request="request", cache_hit=False, **timed())
    case.settlement("request")
    add_performance(case, {})
    groups = summarize_campaign_performance(case.facts)["model_calls"]["runtime"]
    assert groups["campaign"]["call_count"] == 1
    assert groups["campaign"]["tokens"]["input_tokens"]["sum"] == 10


def test_deeper_prefix_imports_are_unknown_not_fresh_or_free_calls(tmp_path):
    from pmmc.experiments.step8_performance_analysis import summarize_campaign_performance

    case = Fixture(tmp_path)
    case.call(
        "r0-alias",
        cache_hit=True,
        stage="planner",
        **timed(),
        metadata={"r0_prefix_import": {"source_run_id": "PRIVATE-source"}},
    )
    add_performance(case, {})
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"] is None
    assert result["actual_research"]["reasons"] == [
        "dependency_artifact_not_bound_to_source_snapshot"
    ]
    assert "PRIVATE-source" not in str(result)
    performance = summarize_campaign_performance(case.facts)
    assert performance["model_calls"]["runtime"]["campaign"]["call_count"] is None


def test_rag_collector_is_read_only_and_reports_missing_privilege():
    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class Session:
        def begin_nested(self):
            return Transaction()

        async def execute(self, statement):
            assert statement.is_select
            assert statement.get_execution_options()["step8_cost_dependency_table"].startswith(
                "pmmc_runtime."
            )
            raise SQLAlchemyError("PRIVATE-dsn-credential")

    result = asyncio.run(
        collect_rag_index_cost_dependencies(
            Session(),
            collection_ids=["index-a", "index-b"],
            campaign_id="campaign",
            selection_hash="selection",
            database_snapshot={"snapshot": "same"},
        )
    )
    assert [e["status"] for e in result["collections"]] == ["incomplete", "incomplete"]
    assert result["dependency_model_call_ids"] == []
    assert "PRIVATE" not in str(result)


def test_questioner_collector_uses_callers_transaction_and_sanitizes_failures(
    tmp_path, monkeypatch
):
    case = Fixture(tmp_path)
    graph = adoption_graph([source_call(case, created_at=datetime(2026, 9, 25, tzinfo=UTC))])
    graphs = {"q-source": graph["source"], "adopted": graph["target"]}
    seen = []

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class Session:
        def begin_nested(self):
            return Transaction()

        async def get(self, model, identity):
            return SimpleNamespace(
                metadata_json=graphs[identity]["headers"]["compiler"]["metadata"]
            )

    async def read(self, session, run_id):
        seen.append(run_id)
        return graphs[run_id]["headers"], graphs[run_id]["rows"]

    monkeypatch.setattr(PostgresQuestionerImportRepository, "_read", read)
    session = Session()
    args = dict(
        compiler_run_ids=["adopted"],
        campaign_id="campaign",
        selection_hash="selection",
        database_snapshot={"snapshot": "same"},
    )
    result = asyncio.run(collect_questioner_cost_dependencies(session, **args))
    assert result["dependency_model_call_ids"] == ["q-call"]
    assert result["dependency_run_ids"] == ["q-source"]
    assert result["adopted_model_call_ids"] == [
        graph["target"]["rows"]["model_calls"][0]["model_call_id"]
    ]
    assert seen == ["q-source", "adopted"]
    assert validate_questioner_cost_dependencies(result["imports"][0]["facts"])
    json.dumps(result, allow_nan=False)

    async def unavailable(*args, **kwargs):
        raise ValueError("PRIVATE-dsn-token")

    monkeypatch.setattr(PostgresQuestionerImportRepository, "_read", unavailable)
    result = asyncio.run(collect_questioner_cost_dependencies(session, **args))
    assert result["imports"][0]["status"] == "incomplete"
    assert "PRIVATE" not in str(result)
