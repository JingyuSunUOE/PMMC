from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.ids import canonical_content_hash
from pmmc.compiler.memory_index import make_memory_operator_index
from pmmc.experiments.step8_cost_analysis import (
    EVALUATION,
    RUNTIME,
    _Deployment,
    _normalize,
    _Snapshot,
    summarize_campaign_costs,
)
from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_r0_cost_dependencies import (
    R0_COST_DEPENDENCIES_ARTIFACT,
    R0_COST_DEPENDENCIES_VERSION,
    _proof_sha,
    collect_r0_prefix_cost_dependencies,
    validate_r0_cost_dependencies,
)
from pmmc.experiments.step8_upstream_import import (
    IMPORT_VERSION,
    R0_ADOPTION_VERSION,
    R0_PREFIX_STAGES,
    R0PrefixImportProof,
    remap_questioner_rows,
)
from pmmc.infrastructure.database.step8_upstream_import import (
    QUESTIONER_TABLES,
    PostgresR0PrefixProofRepository,
    _sort_rows,
    project_r0_prefix_rows,
    r0_prefix_rows_hash,
)
from test_step8_cost_analysis import Fixture
from test_step8_cost_dependencies import adoption_graph, bind_imports, timed
from test_step8_r0_upstream_import import native_empty_prefix, prefix_proof
from test_step8_upstream_import import campaign


def source_receipts(proof, owner=None):
    owner = owner or campaign()
    jobs, receipts, upstream = {}, {}, {}
    for stage in R0_PREFIX_STAGES:
        job = BenchmarkStageJob(
            campaign_id=owner.campaign_id,
            stage=stage,
            component_key=stage,
            resource_class="cpu",
            source_fingerprint=owner.protocol.source_fingerprints[stage],
            scientific_inputs={"stage": stage},
            operation="compiler",
            dependency_ids=list(upstream),
        )
        output = {"compiler_run_id": proof.source_run_id}
        artifacts = []
        if stage in {"questioner", "r0_gate"}:
            key, value = (
                ("questioner_import_proof_sha256", proof.questioner_proof)
                if stage == "questioner"
                else ("r0_prefix_import_proof_sha256", proof)
            )
            output[key] = _proof_sha(value)
            artifacts.append(output[key])
        receipt = ComponentReceipt(
            job_id=job.job_id,
            scientific_input_hash=job.scientific_input_hash,
            source_fingerprint=job.source_fingerprint,
            schema_revision=proof.schema_revision,
            upstream_receipt_hashes=upstream,
            output=output,
            artifact_hashes=artifacts,
            row_counts={"outputs": 1},
        )
        jobs[job.job_id] = {
            **job.model_dump(mode="json"),
            "job_id": job.job_id,
            "scientific_input_hash": job.scientific_input_hash,
            "status": "completed",
            "receipt_hash": receipt.content_hash,
            "payload": job.model_dump(mode="json"),
        }
        receipts[receipt.content_hash] = {
            "receipt_hash": receipt.content_hash,
            "job_id": job.job_id,
            "component_identity": receipt.component_identity,
            "artifact_hashes": artifacts,
            "payload": receipt.model_dump(mode="json"),
        }
        upstream = {job.job_id: receipt.content_hash}
    return {
        "campaign": {
            "campaign_id": owner.campaign_id,
            "protocol_hash": owner.protocol.content_hash,
            "payload": owner.model_dump(mode="json"),
        },
        "jobs": jobs,
        "receipts": receipts,
        "stage_job_ids": list(jobs),
    }


def r0_graph(case, *, priced=False):
    qcall = case.call(
        "q",
        run_id="source-run",
        snapshot_id="snapshot-memlens",
        query_id=None,
        stage="questioner",
        cache_hit=False,
        metadata={},
        **timed(),
    )
    qsource = adoption_graph([qcall])["source"]
    qsource["headers"]["compiler"]["snapshot_id"] = "snapshot-memlens"
    value = prefix_proof()
    qproof = value.questioner_proof.model_copy(
        update={
            "facts_hash": canonical_content_hash(qsource),
            "row_counts": {k: len(v) for k, v in qsource["rows"].items()},
        }
    )
    qrows, qidentities = remap_questioner_rows(
        qsource["rows"],
        primary_keys={t.name: [k.name for k in t.primary_key] for t in QUESTIONER_TABLES},
        source_run_id="source-run",
        target_run_id="adopted",
        proof_hash=qproof.content_hash,
    )
    qrows = {t.name: _sort_rows(t, qrows[t.name]) for t in QUESTIONER_TABLES}
    qlineage = {
        "version": IMPORT_VERSION,
        "source_run_id": "source-run",
        "proof_hash": qproof.content_hash,
        "provider_calls": 0,
        "imported_stages": ["questioner"],
        "source_facts_hash": canonical_content_hash(qsource),
        "target_rows_hash": canonical_content_hash(qrows),
        "identity_map_hash": canonical_content_hash(qidentities),
    }
    qheaders = deepcopy(qsource["headers"])
    for owner in qheaders.values():
        owner["run_id"] = "adopted"
        owner["metadata"]["questioner_upstream_import"] = qlineage
    qgraph = {"source": qsource, "target": {"headers": qheaders, "rows": qrows}}
    rows = native_empty_prefix()
    rows.update(deepcopy(qsource["rows"]))
    for row in rows["compiler_stage_jobs"]:
        row["stage_job_id"] = "job:" + row["stage_job_id"]
    for values in rows.values():
        for row in values:
            if row.get("run_id") == "source":
                row["run_id"] = "source-run"
            row.setdefault("metadata", {})
    index, _, _ = make_memory_operator_index(
        run_id="source-run",
        snapshot_id="snapshot-memlens",
        dataset="memlens",
        selection_hash="selection",
        text_model_alias="text",
        text_model_revision="t-commit",
        multimodal_model_alias="joint",
        multimodal_model_revision="m-commit",
    )
    rows["memory_operator_index_manifests"] = [index.model_dump(mode="json") | {"status": "ready"}]
    for row in rows["embedding_collections"]:
        row["collection_id"] = {
            "tm": index.text_collection_id,
            "vm": index.visual_collection_id,
        }.get(row["collection_id"], row["collection_id"])
    for identity, stage, status in (
        ("p", "planner", "success"),
        ("p-failed", "planner", "failed"),
        ("embedding-call", "embed", "success"),
    ):
        row = case.call(
            identity,
            request="p-dispatch" if priced and identity == "p" else None,
            run_id="source-run",
            snapshot_id="snapshot-memlens",
            query_id=None,
            stage=stage,
            cache_hit=False,
            status=status,
            metadata={},
            **timed(),
        )
        rows["model_calls"].append(deepcopy(row))
    rows["r0_input_manifests"][0]["content_hash"] = value.r0_content_hash
    source = {"headers": {**deepcopy(qsource["headers"]), "canonical_memories": []}, "rows": rows}
    value = value.model_copy(
        update={
            "questioner_proof": qproof,
            "facts_hash": canonical_content_hash(source),
            "row_counts": {k: len(v) for k, v in rows.items()},
        }
    )
    projected, identities = project_r0_prefix_rows(
        rows,
        qsource["rows"],
        proof=value,
        target_run_id="adopted",
        target_selection_hash="target",
    )
    lineage = {
        "version": R0_ADOPTION_VERSION,
        "source_run_id": value.source_run_id,
        "source_campaign_id": value.source_campaign_id,
        "source_protocol_hash": value.source_protocol_hash,
        "proof_json": value.model_dump(mode="json"),
        "proof_hash": value.content_hash,
        "source_proof_sha256": _proof_sha(value),
        "source_facts_hash": value.facts_hash,
        "target_rows_hash": r0_prefix_rows_hash(projected),
        "identity_map_hash": canonical_content_hash(identities),
        "target_selection_hash": "target",
        "compatibility_hash": value.compatibility_hash,
        "imported_stages": list(R0_PREFIX_STAGES[1:]),
        "provider_calls": 0,
    }
    th = deepcopy(qheaders)
    th["manifest"]["selection_hash"] = "target"
    for owner in th.values():
        owner["metadata"]["r0_prefix_upstream_import"] = lineage
    return {
        "source": source,
        "target": {"headers": th, "rows": projected},
        "source_questioner": qsource,
        "source_receipts": source_receipts(value),
    }, qgraph


def bind_r0(case, graph, qgraph, *, additional=()):
    graphs = [(graph, qgraph), *additional]
    updates = bind_imports(case, [q for _, q in graphs])
    calls = {c["model_call_id"]: c for c in case.tables[RUNTIME + "model_calls"]}
    for graph, _ in graphs:
        for side in ("source", "target"):
            calls.update((c["model_call_id"], c) for c in graph[side]["rows"]["model_calls"])
    case.tables[RUNTIME + "model_calls"] = list(calls.values())
    updates["dependency_model_call_ids"] = sorted(
        {c["model_call_id"] for graph, _ in graphs for c in graph["source"]["rows"]["model_calls"]}
    )
    manifest = {
        "schema_version": R0_COST_DEPENDENCIES_VERSION,
        "campaign_id": case.facts.manifest.campaign_id,
        "selection_hash": case.facts.selection.content_hash,
        "source_role": "evaluator",
        "database_snapshot": {"database_role": "pmmc_evaluator", "snapshot": "1:2:"},
        "imports": [
            {
                "source_run_id": graph["source"]["headers"]["compiler"]["run_id"],
                "target_run_id": graph["target"]["headers"]["compiler"]["run_id"],
                "status": "complete",
                "reasons": [],
                "facts": graph,
                "facts_hash": canonical_content_hash(graph),
            }
            for graph, _ in graphs
        ],
    }
    ref = case.artifact(
        R0_COST_DEPENDENCIES_ARTIFACT, manifest, kind="performance", scope="evaluation"
    )
    updates["dependency_artifact_hashes"][ref.artifact_id] = ref.sha256
    case.seal(source_updates=updates)
    return updates


def test_full_r0_projection_aliases_success_failure_and_embedding(tmp_path):
    case = Fixture(tmp_path)
    graph, qgraph = r0_graph(case)
    assert set(validate_r0_cost_dependencies(graph).values()) == {
        "q",
        "p",
        "p-failed",
        "embedding-call",
    }
    bind_r0(case, graph, qgraph)
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["count"] == 0
    assert result["actual_research"]["totals"]["cost_usd"] == "0"
    assert result["reused_dependencies"]["totals"]["count"] == 4
    assert result["reused_dependencies"]["totals"]["cost_usd"] is None


def test_r0_current_settlement_is_counted_once_with_aliases(tmp_path):
    case = Fixture(tmp_path)
    graph, qgraph = r0_graph(case, priced=True)
    case.settlement("p-dispatch", amount="7.25")
    bind_r0(case, graph, qgraph)
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["count"] == 1
    assert result["actual_research"]["totals"]["cost_usd"] == "7.25"
    assert result["reused_dependencies"]["totals"]["count"] == 3


def test_r0_performance_retains_original_failed_attempt_once(tmp_path):
    from pmmc.experiments.step8_performance_analysis import summarize_campaign_performance
    from test_step8_cost_dependencies import add_performance

    case = Fixture(tmp_path)
    graph, qgraph = r0_graph(case)
    updates = bind_r0(case, graph, qgraph)
    add_performance(case, updates)
    runtime = summarize_campaign_performance(case.facts)["model_calls"]["runtime"]
    assert runtime["campaign"]["call_count"] == 0
    assert runtime["reused_dependencies"]["call_count"] == 4
    assert runtime["reused_dependencies"]["tokens"]["input_tokens"]["sum"] == 40


def readopt(graph):
    source = deepcopy(graph["target"])
    source["headers"]["canonical_memories"] = []
    source["rows"]["r0_input_manifests"] = deepcopy(graph["source"]["rows"]["r0_input_manifests"])
    old = R0PrefixImportProof.model_validate(
        source["headers"]["compiler"]["metadata"]["r0_prefix_upstream_import"]["proof_json"]
    )
    qsource = {
        "headers": {
            k: deepcopy(v) for k, v in source["headers"].items() if k != "canonical_memories"
        },
        "rows": {t.name: deepcopy(source["rows"][t.name]) for t in QUESTIONER_TABLES},
    }
    qsource["rows"]["model_calls"] = [
        c for c in qsource["rows"]["model_calls"] if c["stage"] == "questioner"
    ]
    qproof = old.questioner_proof.model_copy(
        update={
            "source_run_id": "adopted",
            "facts_hash": canonical_content_hash(qsource),
            "row_counts": {k: len(v) for k, v in qsource["rows"].items()},
        }
    )
    proof = old.model_copy(
        update={
            "source_run_id": "adopted",
            "questioner_proof": qproof,
            "facts_hash": canonical_content_hash(source),
            "row_counts": {k: len(v) for k, v in source["rows"].items()},
        }
    )
    projected, identities = project_r0_prefix_rows(
        source["rows"],
        qsource["rows"],
        proof=proof,
        target_run_id="readopted",
        target_selection_hash="next",
    )
    qrows = {t.name: deepcopy(projected[t.name]) for t in QUESTIONER_TABLES}
    qrows["model_calls"] = [c for c in qrows["model_calls"] if c["stage"] == "questioner"]
    _, qidentities = remap_questioner_rows(
        qsource["rows"],
        primary_keys={t.name: [k.name for k in t.primary_key] for t in QUESTIONER_TABLES},
        source_run_id="adopted",
        target_run_id="readopted",
        proof_hash=qproof.content_hash,
    )
    headers = deepcopy(qsource["headers"])
    qlineage = {
        "version": IMPORT_VERSION,
        "source_run_id": "adopted",
        "proof_hash": qproof.content_hash,
        "provider_calls": 0,
        "imported_stages": ["questioner"],
        "source_facts_hash": canonical_content_hash(qsource),
        "target_rows_hash": canonical_content_hash(qrows),
        "identity_map_hash": canonical_content_hash(qidentities),
    }
    lineage = {
        **headers["compiler"]["metadata"]["r0_prefix_upstream_import"],
        "source_run_id": "adopted",
        "proof_json": proof.model_dump(mode="json"),
        "proof_hash": proof.content_hash,
        "source_proof_sha256": _proof_sha(proof),
        "source_facts_hash": proof.facts_hash,
        "target_rows_hash": r0_prefix_rows_hash(projected),
        "identity_map_hash": canonical_content_hash(identities),
        "target_selection_hash": "next",
    }
    for owner in headers.values():
        owner["run_id"] = "readopted"
        owner["metadata"].update(
            questioner_upstream_import=qlineage, r0_prefix_upstream_import=lineage
        )
    headers["manifest"]["selection_hash"] = "next"
    return {
        "source": source,
        "target": {"headers": headers, "rows": projected},
        "source_questioner": qsource,
        "source_receipts": source_receipts(proof),
    }, {"source": qsource, "target": {"headers": headers, "rows": qrows}}


def test_transitive_r0_and_q_aliases_resolve_to_four_original_events(tmp_path):
    case = Fixture(tmp_path)
    graph, qgraph = r0_graph(case)
    next_graph, next_q = readopt(graph)
    assert len(validate_r0_cost_dependencies(next_graph)) == 4
    bind_r0(case, graph, qgraph, additional=[(next_graph, next_q)])
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["count"] == 0
    assert result["reused_dependencies"]["totals"]["count"] == 4


def test_deployment_write_includes_all_imported_calls_not_just_selected_program(tmp_path):
    case = Fixture(tmp_path)
    graph, qgraph = r0_graph(case)
    bind_r0(case, graph, qgraph)
    snapshot = _Snapshot(case.facts)
    _, owners, calls, _, _ = _normalize(snapshot, case.tables)
    process = object.__new__(_Deployment)
    process.snapshot, process.owners, process.calls = snapshot, owners, calls
    process.one = lambda *args, **kwargs: graph["target"]["headers"]["compiler"]
    write = {}
    process.compiler(
        SimpleNamespace(binding=None),
        None,
        {"source_compiler_run_id": "adopted", "snapshot_id": "snapshot-memlens"},
        {},
        write,
    )
    assert len(set(write.values())) == 4
    assert {role for _, role in write} == {"questioner_imported_graph", "r0_prefix_imported_graph"}


@pytest.mark.parametrize("change", ["unbound", "missing_call", "changed_tokens"])
def test_r0_exported_evidence_must_match_bound_proof(tmp_path, change):
    case = Fixture(tmp_path)
    graph, qgraph = r0_graph(case)
    updates = bind_r0(case, graph, qgraph)
    if change == "unbound":
        updates["dependency_artifact_hashes"].pop(R0_COST_DEPENDENCIES_ARTIFACT)
    else:
        calls = case.tables[RUNTIME + "model_calls"]
        row = next(c for c in calls if c["model_call_id"] == "p")
        if change == "missing_call":
            calls.remove(row)
        else:
            row["input_tokens"] = 999
    case.seal(source_updates=updates)
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"] is None


def test_valid_receipt_hash_cannot_substitute_different_proof(tmp_path):
    case = Fixture(tmp_path)
    graph, _ = r0_graph(case)
    records = graph["source_receipts"]
    job = next(j for j in records["jobs"].values() if j["stage"] == "r0_gate")
    stored = records["receipts"].pop(job["receipt_hash"])
    stored["payload"]["output"]["r0_prefix_import_proof_sha256"] = "f" * 64
    receipt = ComponentReceipt.model_validate(stored["payload"])
    job["receipt_hash"] = receipt.content_hash
    stored.update(receipt_hash=receipt.content_hash, component_identity=receipt.component_identity)
    records["receipts"][receipt.content_hash] = stored
    with pytest.raises(DependencyUnproven, match="r0_cost_source_proof_not_committed"):
        validate_r0_cost_dependencies(graph)


@pytest.mark.parametrize(
    "mutation", ["proof_sha", "call", "receipt", "stage", "owner", "dependency"]
)
def test_r0_cost_proofs_fail_closed(tmp_path, mutation):
    case = Fixture(tmp_path)
    graph, _ = r0_graph(case)
    records = graph["source_receipts"]
    if mutation == "proof_sha":
        graph["target"]["headers"]["compiler"]["metadata"]["r0_prefix_upstream_import"][
            "source_proof_sha256"
        ] = "f" * 64
    elif mutation == "call":
        graph["target"]["rows"]["model_calls"].pop()
    elif mutation == "receipt":
        next(iter(records["receipts"].values()))["payload"]["output"]["compiler_run_id"] = "other"
    elif mutation == "stage":
        next(iter(records["jobs"].values()))["stage"] = "planner0"
    elif mutation == "owner":
        records["campaign"]["campaign_id"] = "other"
    else:
        next(iter(records["jobs"].values()))["status"] = "failed"
    with pytest.raises(DependencyUnproven):
        validate_r0_cost_dependencies(graph)


def test_r0_collector_borrows_only_supplied_session(monkeypatch, tmp_path):
    case = Fixture(tmp_path)
    graph, _ = r0_graph(case)
    lineage = graph["target"]["headers"]["compiler"]["metadata"]["r0_prefix_upstream_import"]
    receipts = graph["source_receipts"]

    class Session:
        def begin_nested(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, identity):
            if model.__tablename__ == "compiler_runs":
                return SimpleNamespace(
                    metadata_json={"r0_prefix_upstream_import": lineage}
                    if identity == "adopted"
                    else {}
                )
            if model.__tablename__ == "benchmark_campaigns":
                return SimpleNamespace(**receipts["campaign"])
            rows = (
                receipts["jobs"]
                if model.__tablename__ == "benchmark_stage_jobs"
                else receipts["receipts"]
            )
            return SimpleNamespace(**rows[identity])

        async def execute(self, statement):
            assert statement.is_select
            return SimpleNamespace(
                all=lambda: [
                    (
                        SimpleNamespace(**job),
                        SimpleNamespace(**receipts["receipts"][job["receipt_hash"]]),
                    )
                    for job in receipts["jobs"].values()
                ]
            )

    session = Session()

    async def read_prefix(self, supplied, identity, **kwargs):
        assert supplied is session
        side = graph["target" if identity == "adopted" else "source"]
        return side["headers"], side["rows"]

    async def read_q(self, supplied, identity):
        assert supplied is session and identity == "source-run"
        return graph["source_questioner"]["headers"], graph["source_questioner"]["rows"]

    monkeypatch.setattr(PostgresR0PrefixProofRepository, "_read_prefix", read_prefix)
    monkeypatch.setattr(PostgresR0PrefixProofRepository, "_read", read_q)
    result = asyncio.run(
        collect_r0_prefix_cost_dependencies(
            session,
            compiler_run_ids=["adopted"],
            campaign_id="campaign",
            selection_hash="selection",
            database_snapshot={"snapshot": "one"},
        )
    )
    assert result["imports"][0]["status"] == "complete"
    assert result["dependency_model_call_ids"] == ["embedding-call", "p", "p-failed", "q"]
    assert len(result["adopted_model_call_ids"]) == 4

    async def denied(*args, **kwargs):
        raise SQLAlchemyError("PRIVATE PASSWORD QUERY")

    monkeypatch.setattr(PostgresR0PrefixProofRepository, "_read_prefix", denied)
    result = asyncio.run(
        collect_r0_prefix_cost_dependencies(
            session,
            compiler_run_ids=["adopted"],
            campaign_id="campaign",
            selection_hash="selection",
            database_snapshot={"snapshot": "one"},
        )
    )
    assert result["imports"][0]["status"] == "incomplete"
    assert "PRIVATE" not in str(result)
    assert not result["adopted_model_call_ids"]


@pytest.mark.parametrize("stage", ["oracle", "future_evaluator_stage"])
def test_evaluator_only_calls_never_leak_into_deployment(tmp_path, stage):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.seal()
    before = summarize_campaign_costs(case.facts)
    case.call("evaluator-extra", scope=EVALUATION, stage=stage, model_alias="compiler-qwen3.5-9b")
    case.seal()
    after = summarize_campaign_costs(case.facts)
    assert (
        after["actual_research"]["totals"]["count"]
        == before["actual_research"]["totals"]["count"] + 1
    )
    assert after["actual_research"]["totals"]["cost_usd"] is None
    assert after["active_path"] == before["active_path"]
    assert after["amortized"] == before["amortized"]
