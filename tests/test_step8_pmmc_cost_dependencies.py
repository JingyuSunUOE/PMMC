from __future__ import annotations

import asyncio
import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import json_bytes
from pmmc.experiments.step8_cost_analysis import (
    RUNTIME,
    _Deployment,
    _normalize,
    _Snapshot,
    summarize_campaign_costs,
)
from pmmc.experiments.step8_cost_dependencies import DependencyUnproven
from pmmc.experiments.step8_pmmc_cost_dependencies import (
    PMMC_COST_DEPENDENCIES_ARTIFACT,
    PMMC_COST_DEPENDENCIES_VERSION,
    _kernel_source,
    _validate_kernel_lineage,
    collect_pmmc_cost_dependencies,
    kernel_contract,
    validate_execution_dependencies,
    validate_write_graph,
)
from pmmc.experiments.step8_questioner_cost_dependencies import _sealed_json
from test_step8_cost_analysis import Fixture
from test_step8_r0_cost_dependencies import r0_graph


def kernel_lineage(run, stage="dry_run0", *, source_fingerprint=None, owner=None):
    from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
    from test_step8_upstream_import import campaign

    source_fingerprint = source_fingerprint or _kernel_source(stage)
    owner = owner or campaign()
    owner = owner.model_copy(
        update={
            "protocol": owner.protocol.model_copy(
                update={
                    "source_fingerprints": owner.protocol.source_fingerprints
                    | {stage: source_fingerprint},
                }
            )
        }
    )
    job = BenchmarkStageJob(
        campaign_id=owner.campaign_id,
        stage=stage,
        component_key=run,
        source_fingerprint=source_fingerprint,
        scientific_inputs={"owner": run},
        resource_class="cpu",
        operation="compiler",
    )
    receipt = ComponentReceipt(
        job_id=job.job_id,
        scientific_input_hash=job.scientific_input_hash,
        source_fingerprint=job.source_fingerprint,
        schema_revision="0.9.0",
        upstream_receipt_hashes={},
        output={"query_executor_run_id" if stage == "execution" else "compiler_run_id": run},
        artifact_hashes=[],
        row_counts={"outputs": 1},
    )
    return [
        {
            "campaign": {
                "campaign_id": owner.campaign_id,
                "protocol_hash": owner.protocol.content_hash,
                "payload": owner.model_dump(mode="json"),
            },
            "job": job.model_dump(mode="json")
            | {
                "job_id": job.job_id,
                "scientific_input_hash": job.scientific_input_hash,
                "receipt_hash": receipt.content_hash,
                "status": "completed",
                "payload": job.model_dump(mode="json"),
            },
            "receipt": {
                "receipt_hash": receipt.content_hash,
                "job_id": job.job_id,
                "component_identity": receipt.component_identity,
                "artifact_hashes": [],
                "payload": receipt.model_dump(mode="json"),
            },
        }
    ]


def write_graph(case):
    source = r0_graph(case)[0]["source"]
    rows = source["rows"]
    for collection in rows["embedding_collections"]:
        collection["snapshot_id"] = "snapshot-memlens"
    rows["qbank_index_manifests"][0]["compiler_run_id"] = "source-run"
    rows["r0_input_manifests"][0]["run_id"] = "source-run"
    for row in rows["compiler_stage_jobs"]:
        row["run_id"] = "source-run"
    for name in (
        "doubter_feedback",
        "candidate_stage_jobs",
        "planner_revision_lineage",
        "qbank_round_projection_members",
    ):
        rows[name] = []
    rows["qbank_round_projection_manifests"] = [
        {
            "projection_id": "projection-" + str(i),
            "round_id": i,
            "member_count": 0,
            "pending_count": 0,
            "members_hash": canonical_content_hash([]),
            "source_r0_manifest_id": rows["r0_input_manifests"][0]["r0_manifest_id"],
        }
        for i in range(4)
    ]
    return {
        "prefix_headers": source["headers"],
        "compiler": source["headers"]["compiler"] | {"operator_set_version": "pmmc-operators-v7"},
        "manifest": {
            "run_id": "source-run",
            "manifest_json": {
                "dataset_snapshot_ids": ["snapshot-memlens"],
                "metadata": {"questioner_config": {}},
            },
        },
        "state": {"run_id": "source-run", "status": "completed", "completed_at": "2026-09-25"},
        "rows": rows,
        "calls": deepcopy(rows["model_calls"]),
        "collections": rows["embedding_collections"],
        "vectors": rows["embeddings"],
        "canonical_memories": [],
        "memory_raw_image_policy": "allow",
        "kernel_contract": kernel_contract(),
        "kernel_lineage": [],
        "r0_imports": [],
    }


def nonempty_write_graph(case):
    from pmmc.compiler.ids import memory_program_content_hash
    from pmmc.compiler.memory_index import make_memory_operator_index
    from pmmc.infrastructure.database.compiler_mappers import memory_program_from_rows
    from pmmc.schemas.doubter import QBankRoundProjectionMember
    from test_step8_r0_upstream_import import native_program_prefix

    graph = write_graph(case)
    native, _, _ = native_program_prefix()
    rows, compiler = graph["rows"], graph["compiler"]
    for name in (
        "qbank_questions",
        "qbank_evidence_refs",
        "questioner_candidate_gate_results",
        "planner_requirement_contracts",
        "planner_candidate_outcomes",
        "memory_programs",
        "program_steps",
        "program_dry_run_assessments",
    ):
        rows[name] = _sealed_json(native[name])
        for row in rows[name]:
            for key in ("run_id", "source_compiler_run_id"):
                if key in row:
                    row[key] = "source-run"
            if "snapshot_id" in row:
                row["snapshot_id"] = "snapshot-memlens"
            if row.get("producer_model_call_id"):
                row["producer_model_call_id"] = "q" if name == "qbank_questions" else "p"
    compiler.update(
        operator_registry_hash=kernel_contract()["operator_registry_hash"],
        compiler_config_hash=kernel_contract()["compiler_config_hash"],
        memory_text_model_alias="qwen3-embedding-8b",
        memory_text_model_revision="a" * 40,
        memory_multimodal_model_alias="qwen3-vl-embedding-8b",
        memory_multimodal_model_revision="b" * 40,
    )
    graph["prefix_headers"]["compiler"] = deepcopy(compiler)
    index, text, visual = make_memory_operator_index(
        run_id="source-run",
        snapshot_id="snapshot-memlens",
        dataset="memlens",
        selection_hash="selection",
        text_model_alias=compiler["memory_text_model_alias"],
        text_model_revision=compiler["memory_text_model_revision"],
        multimodal_model_alias=compiler["memory_multimodal_model_alias"],
        multimodal_model_revision=compiler["memory_multimodal_model_revision"],
    )
    rows["memory_operator_index_manifests"] = [index.model_dump(mode="json") | {"status": "ready"}]
    collections = [
        text.model_dump(mode="json") | {"status": "ready"},
        visual.model_dump(mode="json") | {"status": "ready"},
    ]
    for cid in ("tq", "mq", "tp", "mp", "qt", "qm"):
        family = "multimodal" if cid in {"mq", "mp", "qm"} else "text"
        collections.append(
            {
                "collection_id": cid,
                "run_id": "source-run",
                "snapshot_id": "snapshot-memlens",
                "model_alias": compiler["memory_" + family + "_model_alias"],
                "actual_model_id": "Qwen/Qwen3-VL-Embedding-8B"
                if family == "multimodal"
                else "Qwen/Qwen3-Embedding-8B",
                "model_revision": compiler["memory_" + family + "_model_revision"],
                "dimension": 4096,
                "status": "ready",
                "metadata": {},
            }
        )
    graph["collections"] = rows["embedding_collections"] = collections
    embed = next(
        c for c in case.tables[RUNTIME + "model_calls"] if c["model_call_id"] == "embedding-call"
    )
    embed.update(model_alias="qwen3-embedding-8b", actual_model_id="Qwen/Qwen3-Embedding-8B")
    case.call(
        "joint-embedding-call",
        run_id="source-run",
        snapshot_id="snapshot-memlens",
        query_id=None,
        stage="embed",
        cache_hit=False,
        model_alias="qwen3-vl-embedding-8b",
        actual_model_id="Qwen/Qwen3-VL-Embedding-8B",
        metadata={},
    )
    graph["vectors"] = rows["embeddings"] = [
        {
            "embedding_id": "vector-" + cid,
            "collection_id": cid,
            "qbank_id": "q1",
            "dimension": 4096,
            "embedding": [1.0] + [0.0] * 4095,
            "input_hash": "input-" + cid,
            "metadata": {},
            "model_call_id": "joint-embedding-call" if cid in {"mq", "qm"} else "embedding-call",
        }
        for cid in ("tq", "mq", "qt", "qm")
    ]
    rows["program_query_embeddings"], rows["program_query_embedding_batches"] = [], []
    for cid, channel in (("qt", "question_text_query"), ("qm", "question_joint_query")):
        rows["program_query_embeddings"].append(
            {
                "program_query_embedding_id": "ref-" + cid,
                "query_embedding_batch_id": "batch-" + cid,
                "run_id": "source-run",
                "qbank_id": "q1",
                "channel": channel,
                "embedding_id": "vector-" + cid,
                "input_hash": "input-" + cid,
                "metadata": {},
            }
        )
        rows["program_query_embedding_batches"].append(
            {
                "query_embedding_batch_id": "batch-" + cid,
                "run_id": "source-run",
                "channel": channel,
                "qbank_ids": ["q1"],
                "metadata": {},
                "model_call_id": "joint-embedding-call" if cid == "qm" else "embedding-call",
            }
        )
    rows["qbank_index_manifests"][0]["eligible_question_count"] = 1
    rows["r0_input_manifests"][0].update(
        candidate_count=1, program_count=1, execution_count=1, terminal_outcome_count=1
    )
    program = rows["memory_programs"][0]
    program["program_hash"] = memory_program_content_hash(
        memory_program_from_rows(program, rows["program_steps"], [])
    )
    trace = {
        "execution_id": "exec",
        "run_id": "source-run",
        "source_compiler_run_id": "source-run",
        "snapshot_id": "snapshot-memlens",
        "plan_id": program["plan_id"],
        "query_id": None,
        "purpose": "compiler_dry_run",
        "status": "success",
        "completed_at": "2026-09-25",
        "metadata": {},
        "bindings_hash": canonical_content_hash(
            {"text_memory": index.text_collection_id, "visual_memory": index.visual_collection_id}
        ),
    }
    rows["program_executions"] = [trace]
    output = {"success": True, "control_action": "continue"}
    rows["program_execution_steps"] = [
        {
            **trace,
            "step_id": s["step_id"],
            "op_name": s["op_name"],
            "output": output,
            "output_hash": canonical_content_hash(output),
            "success": True,
        }
        for s in rows["program_steps"]
    ]
    for assessment in rows["program_dry_run_assessments"]:
        assessment["execution_id"] = "exec"
    graph["kernel_lineage"] = kernel_lineage("source-run")
    for identity, status in (("d", "success"), ("d-failed", "failed")):
        case.call(
            identity,
            run_id="source-run",
            snapshot_id="snapshot-memlens",
            query_id=None,
            stage="doubter",
            plan_id=program["plan_id"],
            status=status,
            cache_hit=False,
        )
    rows["doubter_feedback"] = [
        {
            "feedback_id": "feedback",
            "run_id": "source-run",
            "qbank_id": "q1",
            "plan_id": program["plan_id"],
            "round_id": 0,
            "producer_model_call_id": "d",
        }
    ]
    for projection in rows["qbank_round_projection_manifests"]:
        member = QBankRoundProjectionMember(
            projection_member_id="member-" + str(projection["round_id"]),
            projection_id=projection["projection_id"],
            run_id="source-run",
            qbank_id="q1",
            route_class="answerable",
            outcome="accepted",
            effective_plan_id=program["plan_id"],
            effective_feedback_id="feedback" if projection["round_id"] else None,
            content_hash="member-hash",
        )
        rows["qbank_round_projection_members"].append(member.model_dump(mode="json"))
        projection.update(
            member_count=1, members_hash=canonical_content_hash([member.model_dump(mode="json")])
        )
    graph["calls"] = deepcopy(
        [c for c in case.tables[RUNTIME + "model_calls"] if c["run_id"] == "source-run"]
    )
    rows["model_calls"] = deepcopy(graph["calls"])
    return graph


def execution_graph():
    output = {"success": True, "control_action": "continue"}
    bindings = {"text_memory": "text", "visual_memory": "visual"}
    trace = {
        "execution_id": "execution",
        "run_id": "runtime",
        "source_compiler_run_id": "compiler",
        "snapshot_id": "snapshot",
        "plan_id": "program",
        "query_id": "query",
        "purpose": "query_time",
        "status": "success",
        "completed_at": "2026-09-25",
        "bindings_hash": canonical_content_hash(bindings),
    }
    return {
        "execution": trace,
        "program": {
            "plan_id": "program",
            "run_id": "compiler",
            "qbank_id": "candidate",
            "operator_set_version": "pmmc-operators-v7",
            "round_id": 0,
        },
        "compiler": {
            "run_id": "compiler",
            "snapshot_id": "snapshot",
            "operator_set_version": "pmmc-operators-v7",
            "operator_registry_hash": kernel_contract()["operator_registry_hash"],
            "compiler_config_hash": kernel_contract()["compiler_config_hash"],
        },
        "program_steps": [{"step_id": 0, "op_name": "retrieve_text_by_keyword"}],
        "steps": [
            {
                **trace,
                "step_id": 0,
                "op_name": "retrieve_text_by_keyword",
                "output": output,
                "output_hash": canonical_content_hash(output),
                "success": True,
            }
        ],
        "memory_index": {
            "status": "ready",
            "text_collection_id": "text",
            "visual_collection_id": "visual",
        },
        "query_vectors": [
            {
                "qbank_id": "candidate",
                "run_id": "compiler",
                "channel": c,
                "embedding_id": c,
                "input_hash": "input",
            }
            for c in ("question_text_query", "question_joint_query")
        ],
        "vectors": [
            {
                "embedding_id": c,
                "qbank_id": "candidate",
                "input_hash": "input",
                "model_call_id": "shared-producer",
            }
            for c in ("question_text_query", "question_joint_query")
        ],
        "model_calls": [],
        "kernel_contract": kernel_contract(),
        "kernel_lineage": kernel_lineage("runtime", "execution"),
    }


def bind_pmmc(case, graphs=(), executions=()):
    manifest = {
        "schema_version": PMMC_COST_DEPENDENCIES_VERSION,
        "campaign_id": case.facts.manifest.campaign_id,
        "selection_hash": case.facts.selection.content_hash,
        "source_role": "evaluator",
        "database_snapshot": {"database_role": "pmmc_evaluator", "snapshot": "1:2:"},
        "write_graphs": [
            {
                "run_id": g["compiler"]["run_id"],
                "status": "complete",
                "reasons": [],
                "facts": g,
                "facts_hash": canonical_content_hash(g),
            }
            for g in graphs
        ],
        "executions": [
            {
                "execution_id": e["execution"]["execution_id"],
                "status": "complete",
                "reasons": [],
                "facts": e,
                "facts_hash": canonical_content_hash(e),
            }
            for e in executions
        ],
    }
    ref = case.artifact(
        PMMC_COST_DEPENDENCIES_ARTIFACT, manifest, kind="performance", scope="evaluation"
    )
    updates = {"dependency_artifact_hashes": {ref.artifact_id: ref.sha256}}
    case.seal(source_updates=updates)
    return updates


def test_whole_terminal_bank_closes_native_write_and_preserves_failed_cost(tmp_path):
    case = Fixture(tmp_path)
    graph = write_graph(case)
    expected = {"q", "p", "p-failed", "embedding-call"}
    assert validate_write_graph(graph) == {r: expected for r in range(4)}
    for row in case.tables[RUNTIME + "model_calls"]:
        case.price(row, "2.00")
    graph["calls"] = deepcopy(case.tables[RUNTIME + "model_calls"])
    graph["rows"]["model_calls"] = deepcopy(graph["calls"])
    bind_pmmc(case, [graph])
    snapshot = _Snapshot(case.facts)
    measured, owners, calls, _, _ = _normalize(snapshot, case.tables)
    process = object.__new__(_Deployment)
    process.snapshot, process.owners, process.calls = snapshot, owners, calls
    process.one = lambda *args, **kwargs: graph["compiler"]
    write = {}
    assert (
        process.compiler(
            SimpleNamespace(binding=None),
            SimpleNamespace(round_id=2),
            {"source_compiler_run_id": "source-run", "snapshot_id": "snapshot-memlens"},
            {},
            write,
        )
        is True
    )
    assert len(set(write.values())) == 4
    assert sum(measured[i].cost_usd for i in set(write.values())) == 8


def test_nonempty_bank_repo_identity_revision_and_round_costs(tmp_path):
    graph = nonempty_write_graph(Fixture(tmp_path))
    rounds = validate_write_graph(graph)
    assert rounds[0] == {"q", "p", "p-failed", "embedding-call", "joint-embedding-call"}
    assert rounds[1] == rounds[0] | {"d", "d-failed"}
    assert rounds[3] == rounds[1]
    assert graph["calls"][3]["actual_model_id"] == "Qwen/Qwen3-Embedding-8B"
    changed = deepcopy(graph)
    changed["calls"][3]["actual_model_id"] = "a" * 40
    with pytest.raises(DependencyUnproven, match="producer_mismatch"):
        validate_write_graph(changed)


def prepare_active_case(case, graph):
    from pmmc.experiments.step8_cost_dependencies import (
        RAG_COST_DEPENDENCIES_ARTIFACT,
        RAG_COST_DEPENDENCIES_VERSION,
    )
    from test_step8_cost_dependencies import rag_graph

    case.facts.manifest = case.facts.manifest.model_copy(
        update={
            "snapshot_ids": {
                k: "snapshot-memlens" if k.value == "memlens" else v
                for k, v in case.facts.manifest.snapshot_ids.items()
            },
        }
    )
    case.bind_answers()
    case.bind_process()
    case.bind_match_embeddings()
    case.restrict_methods({"pmmc"})
    case.facts.cells = tuple(c for c in case.facts.cells if c.dataset.value == "memlens")
    selected = {c.cell_id for c in case.facts.cells}
    for job_id, receipt in list(case.facts.receipts.items()):
        output = deepcopy(receipt.output)
        output["scores"] = [s for s in output["scores"] if s["cell_id"] in selected]
        case.facts.receipts[job_id] = receipt.model_copy(
            update={"output": output, "artifact_hashes": [case.facts.store.put_json(output)]}
        )
        for state in case.tables["pmmc_evaluation.evaluation_run_states"]:
            if state["evaluation_run_id"] == output["evaluation_run_id"]:
                state["metadata"]["completion_report"] = output
    owner = graph["compiler"]["run_id"]
    for name in ("query_executor_runs", "qbank_match_batches"):
        for row in case.tables[RUNTIME + name]:
            if row["snapshot_id"] == "snapshot-memlens":
                row["source_compiler_run_id"] = owner
    case.tables[RUNTIME + "compiler_runs"].append(graph["compiler"])
    index = rag_graph(
        collection="fallback-index", snapshot="snapshot-memlens", method="native_multimodal_rag"
    )
    trace = index["calls"][0]
    case.call(trace["model_call_id"], **{k: v for k, v in trace.items() if k != "model_call_id"})
    for batch in case.tables[RUNTIME + "retrieval_batches"]:
        if batch["snapshot_id"] == "snapshot-memlens":
            batch["document_collection_id"] = "fallback-index"
    ref = case.artifact(
        RAG_COST_DEPENDENCIES_ARTIFACT,
        {
            "schema_version": RAG_COST_DEPENDENCIES_VERSION,
            "campaign_id": case.facts.manifest.campaign_id,
            "selection_hash": case.facts.selection.content_hash,
            "source_role": "evaluator",
            "database_snapshot": {"database_role": "pmmc_evaluator", "snapshot": "1:2:"},
            "collections": [
                {
                    "collection_id": "fallback-index",
                    "status": "complete",
                    "reasons": [],
                    "facts": index,
                    "facts_hash": canonical_content_hash(index),
                }
            ],
        },
        kind="performance",
        scope="evaluation",
    )
    return {ref.artifact_id: ref.sha256}


@pytest.mark.parametrize("maintenance", [False, True])
def test_nonempty_native_complete_active_write_and_amortization(tmp_path, monkeypatch, maintenance):
    case = Fixture(tmp_path)
    graph = nonempty_write_graph(case)
    if maintenance:
        graph["kernel_lineage"], _, _, _, validator = maintenance_kernel_lineage(
            monkeypatch, run="source-run"
        )
        retained = retained_compiler_lineage(graph["kernel_lineage"])
        retained_bytes = json_bytes(retained)
        graph["kernel_lineage"] = retained + graph["kernel_lineage"]
    hashes = prepare_active_case(case, graph)
    for row in case.tables[RUNTIME + "model_calls"]:
        case.price(row, "2.00")
    graph["calls"] = deepcopy(
        [c for c in case.tables[RUNTIME + "model_calls"] if c["run_id"] == "source-run"]
    )
    graph["rows"]["model_calls"] = deepcopy(graph["calls"])
    updates = bind_pmmc(case, [graph])
    updates["dependency_artifact_hashes"].update(hashes)
    case.seal(source_updates=updates)
    result = summarize_campaign_costs(case.facts)
    assert result["active_path"]["cells"]
    for cell in result["active_path"]["cells"]:
        assert cell["write"]["status"] == "complete", cell["write"]
        assert cell["query"]["status"] == "complete", cell["query"]
        assert cell["write"]["totals"]["cost_usd"] == (
            "12.00" if cell["round_id"] == 0 else "16.00"
        )
    assert all(r["cost_usd"] is not None and r["reason"] is None for r in result["amortized"])
    if maintenance:
        validator.assert_called()
        assert json_bytes(graph["kernel_lineage"][: len(retained)]) == retained_bytes
        assert all(
            "maintenance_execution" not in e["receipt"]["payload"]["output"] for e in retained
        )


def imported_write_graph(case, graph):
    from pmmc.experiments.step8_r0_cost_dependencies import (
        _proof_sha,
        validate_r0_cost_dependencies,
    )
    from pmmc.experiments.step8_upstream_import import (
        IMPORT_VERSION,
        R0_ADOPTION_VERSION,
        R0_PREFIX_STAGES,
        r0_prefix_compatibility,
        remap_questioner_rows,
    )
    from pmmc.infrastructure.database.step8_upstream_import import (
        QUESTIONER_TABLES,
        R0_PREFIX_TABLES,
        _rehash_questions,
        _sort_rows,
        project_r0_prefix_rows,
        r0_prefix_rows_hash,
    )
    from pmmc.schemas.step8 import BenchmarkCampaignManifest
    from test_step8_r0_cost_dependencies import bind_r0, source_receipts
    from test_step8_r0_upstream_import import prefix_proof

    source_rows = {
        t.name: deepcopy(graph["rows"][t.name]) for t in (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)
    }
    source_rows["model_calls"] = [c for c in source_rows["model_calls"] if c["stage"] != "doubter"]
    source = {"headers": deepcopy(graph["prefix_headers"]), "rows": source_rows}
    qsource = {
        "headers": deepcopy(graph["prefix_headers"]),
        "rows": {t.name: deepcopy(source_rows[t.name]) for t in QUESTIONER_TABLES},
    }
    qsource["rows"]["model_calls"] = [
        c for c in qsource["rows"]["model_calls"] if c["stage"] == "questioner"
    ]
    owner = BenchmarkCampaignManifest.model_validate(
        graph["kernel_lineage"][0]["campaign"]["payload"]
    )
    base = prefix_proof()
    scope = {
        k: base.compatibility["questioner"][k]
        for k in ("snapshot_id", "dataset", "conversation_id")
    }
    scope["treatment"] = base.compatibility["treatment"]
    compat = r0_prefix_compatibility(owner, scope)
    qproof = base.questioner_proof.model_copy(
        update={
            "source_protocol_hash": owner.protocol.content_hash,
            "compatibility": compat["questioner"],
            "compatibility_hash": canonical_content_hash(compat["questioner"]),
            "facts_hash": canonical_content_hash(qsource),
            "row_counts": {k: len(v) for k, v in qsource["rows"].items()},
            "eligible_count": 1,
        }
    )
    proof = base.model_copy(
        update={
            "source_protocol_hash": owner.protocol.content_hash,
            "compatibility": compat,
            "compatibility_hash": canonical_content_hash(compat),
            "questioner_proof": qproof,
            "facts_hash": canonical_content_hash(source),
            "row_counts": {k: len(v) for k, v in source_rows.items()},
        }
    )
    qrows, qids = remap_questioner_rows(
        qsource["rows"],
        primary_keys={t.name: [k.name for k in t.primary_key] for t in QUESTIONER_TABLES},
        source_run_id="source-run",
        target_run_id="adopted",
        proof_hash=qproof.content_hash,
    )
    _rehash_questions(qrows)
    qrows = {t.name: _sort_rows(t, qrows[t.name]) for t in QUESTIONER_TABLES}
    qlineage = {
        "version": IMPORT_VERSION,
        "source_run_id": "source-run",
        "proof_hash": qproof.content_hash,
        "provider_calls": 0,
        "imported_stages": ["questioner"],
        "source_facts_hash": canonical_content_hash(qsource),
        "target_rows_hash": canonical_content_hash(qrows),
        "identity_map_hash": canonical_content_hash(qids),
    }
    qheaders = deepcopy(graph["prefix_headers"])
    for key in ("compiler", "manifest"):
        qheaders[key]["run_id"] = "adopted"
        qheaders[key]["metadata"]["questioner_upstream_import"] = qlineage
    qgraph = {"source": qsource, "target": {"headers": qheaders, "rows": qrows}}
    projected, ids = project_r0_prefix_rows(
        source_rows,
        qsource["rows"],
        proof=proof,
        target_run_id="adopted",
        target_selection_hash="target",
    )
    lineage = {
        "version": R0_ADOPTION_VERSION,
        "source_run_id": "source-run",
        "source_campaign_id": proof.source_campaign_id,
        "source_protocol_hash": proof.source_protocol_hash,
        "proof_json": proof.model_dump(mode="json"),
        "proof_hash": proof.content_hash,
        "source_proof_sha256": _proof_sha(proof),
        "source_facts_hash": proof.facts_hash,
        "target_rows_hash": r0_prefix_rows_hash(projected),
        "identity_map_hash": canonical_content_hash(ids),
        "target_selection_hash": "target",
        "compatibility_hash": proof.compatibility_hash,
        "imported_stages": list(R0_PREFIX_STAGES[1:]),
        "provider_calls": 0,
    }
    headers = deepcopy(qheaders)
    headers["manifest"]["selection_hash"] = "target"
    for key in ("compiler", "manifest"):
        headers[key]["metadata"]["r0_prefix_upstream_import"] = lineage
    r0 = {
        "source": source,
        "target": {"headers": headers, "rows": projected},
        "source_questioner": qsource,
        "source_receipts": source_receipts(proof, owner),
    }
    validate_r0_cost_dependencies(r0)
    updates = bind_r0(case, r0, qgraph)
    target = deepcopy(graph)
    target["compiler"] = headers["compiler"]
    target["manifest"]["run_id"] = target["state"]["run_id"] = "adopted"
    target["rows"].update(deepcopy(projected))
    target["rows"]["r0_input_manifests"] = [
        graph["rows"]["r0_input_manifests"][0] | {"run_id": "adopted"}
    ]
    for name in ("doubter_feedback", "qbank_round_projection_members"):
        for row in target["rows"][name]:
            row["run_id"] = "adopted"
            for field in ("qbank_id", "plan_id", "effective_plan_id"):
                if row.get(field):
                    row[field] = ids[row[field]]
    for projection in target["rows"]["qbank_round_projection_manifests"]:
        members = [
            m
            for m in target["rows"]["qbank_round_projection_members"]
            if m["projection_id"] == projection["projection_id"]
        ]
        projection["members_hash"] = canonical_content_hash(members)
    for call in case.tables[RUNTIME + "model_calls"]:
        if call["model_call_id"] in {"d", "d-failed"}:
            call["run_id"], call["plan_id"] = "adopted", ids[call["plan_id"]]
    target["calls"] = deepcopy(
        [c for c in case.tables[RUNTIME + "model_calls"] if c["run_id"] == "adopted"]
    )
    target["rows"]["model_calls"] = deepcopy(target["calls"])
    target["collections"], target["vectors"] = (
        target["rows"]["embedding_collections"],
        target["rows"]["embeddings"],
    )
    target["r0_imports"] = [
        {
            "target_run_id": "adopted",
            "source_run_id": "source-run",
            "status": "complete",
            "facts": r0,
            "facts_hash": canonical_content_hash(r0),
        }
    ]
    return target, updates


def test_imported_source_binding_has_complete_active_cost_without_duplicate_calls(tmp_path):
    case = Fixture(tmp_path)
    graph = nonempty_write_graph(case)
    hashes = prepare_active_case(case, graph)
    for row in case.tables[RUNTIME + "model_calls"]:
        case.price(row, "2.00")
    graph["calls"] = deepcopy(
        [c for c in case.tables[RUNTIME + "model_calls"] if c["run_id"] == "source-run"]
    )
    graph["rows"]["model_calls"] = deepcopy(graph["calls"])
    target, updates = imported_write_graph(case, graph)
    assert len(validate_write_graph(target)[2]) == 7
    source_hash = graph["rows"]["program_executions"][0]["bindings_hash"]
    assert target["rows"]["program_executions"][0]["bindings_hash"] == source_hash
    index = target["rows"]["memory_operator_index_manifests"][0]
    assert (
        canonical_content_hash(
            {
                "text_memory": index["text_collection_id"],
                "visual_memory": index["visual_collection_id"],
            }
        )
        != source_hash
    )
    for name in ("query_executor_runs", "qbank_match_batches"):
        for row in case.tables[RUNTIME + name]:
            if row["snapshot_id"] == "snapshot-memlens":
                row["source_compiler_run_id"] = "adopted"
    case.tables[RUNTIME + "compiler_runs"].append(target["compiler"])
    pmmc = bind_pmmc(case, [target])
    updates["dependency_artifact_hashes"].update(hashes | pmmc["dependency_artifact_hashes"])
    case.seal(source_updates=updates)
    result = summarize_campaign_costs(case.facts)
    cell = result["active_path"]["cells"][0]
    assert cell["write"]["status"] == "complete", cell["write"]
    assert (
        cell["write"]["totals"]["count"] == 8
    )  # Seven physical bank calls plus shared fallback index.
    assert cell["write"]["totals"]["cost_usd"] == "16.00"
    assert all(r["cost_usd"] is not None and r["reason"] is None for r in result["amortized"])


def test_kernel_source_is_not_cached_across_source_changes(monkeypatch):
    from pmmc.experiments import step8_jobs

    values = iter(("before", "after"))
    monkeypatch.setattr(step8_jobs, "scientific_source_fingerprint", lambda *args: next(values))
    assert _kernel_source("execution") == "before"
    assert _kernel_source("execution") == "after"


@pytest.mark.parametrize("stage", ["dry_run0", "doubter", "execution"])
def test_kernel_source_uses_configured_root_from_installed_layout(monkeypatch, tmp_path, stage):
    from pmmc.experiments import step8_pmmc_cost_dependencies
    from pmmc.infrastructure import settings

    root = Path(step8_pmmc_cost_dependencies.__file__).resolve().parents[3]
    expected = _kernel_source(stage)
    installed = tmp_path / "opt/venv/lib/python3.12/site-packages/pmmc/experiments"
    monkeypatch.setattr(
        step8_pmmc_cost_dependencies,
        "__file__",
        str(installed / "step8_pmmc_cost_dependencies.py"),
    )
    monkeypatch.setenv("PMMC_PROJECT_ROOT", str(root))
    monkeypatch.setattr(settings, "get_settings", lambda: settings.Settings(_env_file=None))
    monkeypatch.chdir(tmp_path)

    assert _kernel_source(stage) == expected


def test_kernel_contract_uses_configured_root_and_installed_sources(monkeypatch, tmp_path):
    from pmmc.experiments import step8_pmmc_cost_dependencies
    from pmmc.infrastructure import settings

    root = Path(step8_pmmc_cost_dependencies.__file__).resolve().parents[3]
    expected = kernel_contract()
    installed = tmp_path / "opt/venv/lib/python3.12/site-packages/pmmc"
    for relative in expected["source_hashes"]:
        source = installed / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        data = f"# installed fixture: {relative}\n".encode()
        source.write_bytes(data)
        expected["source_hashes"][relative] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        step8_pmmc_cost_dependencies,
        "__file__",
        str(installed / "experiments/step8_pmmc_cost_dependencies.py"),
    )
    monkeypatch.setenv("PMMC_PROJECT_ROOT", str(root))
    monkeypatch.setattr(settings, "get_settings", lambda: settings.Settings(_env_file=None))
    monkeypatch.chdir(tmp_path)

    assert kernel_contract() == expected


def test_kernel_roots_do_not_fall_back_when_configured_workspace_is_missing(monkeypatch, tmp_path):
    from pmmc.infrastructure import settings

    monkeypatch.setenv("PMMC_PROJECT_ROOT", str(tmp_path / "missing-workspace"))
    monkeypatch.setattr(settings, "get_settings", lambda: settings.Settings(_env_file=None))

    with pytest.raises(ValueError, match="stage source contract is missing"):
        _kernel_source("execution")
    with pytest.raises(FileNotFoundError):
        kernel_contract()


def maintenance_kernel_lineage(monkeypatch, stage="dry_run0", *, run="runtime"):
    from pmmc.experiments import step8_pmmc_cost_dependencies
    from pmmc.experiments.step8_jobs import ComponentReceipt

    frozen, actual = "a" * 64, "b" * 64
    lineage = kernel_lineage(run, stage, source_fingerprint=frozen)
    entry = lineage[0]
    proof = {"policy": "maintenance-fixture"}
    digest = hashlib.sha256(json_bytes(proof)).hexdigest()
    receipt = ComponentReceipt.model_validate(entry["receipt"]["payload"])
    receipt = receipt.model_copy(
        update={
            "output": receipt.output
            | {"maintenance_execution": {"proof": proof, "sha256": digest}},
            "artifact_hashes": [digest],
        }
    )
    entry["receipt"].update(
        payload=receipt.model_dump(mode="json"),
        receipt_hash=receipt.content_hash,
        artifact_hashes=receipt.artifact_hashes,
    )
    entry["job"]["receipt_hash"] = receipt.content_hash
    monkeypatch.setattr(step8_pmmc_cost_dependencies, "_kernel_source", lambda _: actual)
    module = ModuleType("pmmc.experiments.step8_maintenance")
    validator = Mock()
    module.validate_maintenance_execution = validator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    trace = {
        "purpose": "query_time" if stage == "execution" else "compiler_dry_run",
        "run_id": run,
    }
    program = {"run_id": run, "round_id": int(stage == "doubter")}
    return lineage, trace, program, proof, validator


def retained_compiler_lineage(lineage):
    from pmmc.schemas.step8 import BenchmarkCampaignManifest

    owner = BenchmarkCampaignManifest.model_validate(lineage[0]["campaign"]["payload"])
    output = lineage[0]["receipt"]["payload"]["output"]
    run = output.get("compiler_run_id", output.get("query_executor_run_id"))
    return [
        kernel_lineage(
            run, stage, owner=owner, source_fingerprint=owner.protocol.source_fingerprints[stage]
        )[0]
        for stage in ("questioner", "planner0", "text_index", "multimodal_index")
    ]


@pytest.mark.parametrize("stage", ["dry_run0", "doubter", "execution"])
def test_kernel_maintenance_proof_binds_only_execution_source(monkeypatch, stage):
    from pmmc.experiments import step8_pmmc_cost_dependencies

    lineage, trace, program, proof, validator = maintenance_kernel_lineage(monkeypatch, stage)
    current_source = Mock(return_value="b" * 64)
    monkeypatch.setattr(step8_pmmc_cost_dependencies, "_kernel_source", current_source)
    retained = retained_compiler_lineage(lineage)
    assert all(e["job"]["source_fingerprint"] != "b" * 64 for e in retained)
    assert all("maintenance_execution" not in e["receipt"]["payload"]["output"] for e in retained)
    lineage = retained + lineage
    before = deepcopy(lineage)
    _validate_kernel_lineage(lineage, trace, program)
    current_source.assert_called_once_with(stage)
    validator.assert_called_once_with(
        proof,
        campaign_id=lineage[0]["campaign"]["campaign_id"],
        protocol_hash=lineage[0]["campaign"]["protocol_hash"],
        stage=stage,
        frozen_source_fingerprint="a" * 64,
        actual_source_fingerprint="b" * 64,
    )
    assert lineage == before


@pytest.mark.parametrize("stage", ["dry_run0", "doubter", "execution"])
def test_retained_producer_receipts_cannot_replace_kernel_receipt(monkeypatch, stage):
    lineage, trace, program, _, validator = maintenance_kernel_lineage(monkeypatch, stage)
    retained = retained_compiler_lineage(lineage)
    before = json_bytes(retained)
    with pytest.raises(DependencyUnproven, match="pmmc_execution_producer_receipt_missing"):
        _validate_kernel_lineage(retained, trace, program)
    validator.assert_not_called()
    assert json_bytes(retained) == before


@pytest.mark.parametrize("stage", ["dry_run0", "doubter", "execution"])
def test_retained_old_execution_receipt_is_not_generically_grandfathered(monkeypatch, stage):
    from pmmc.experiments import step8_pmmc_cost_dependencies

    lineage = kernel_lineage("runtime", stage, source_fingerprint="a" * 64)
    monkeypatch.setattr(step8_pmmc_cost_dependencies, "_kernel_source", lambda _: "b" * 64)
    before = json_bytes(lineage)
    with pytest.raises(DependencyUnproven, match="pmmc_execution_producer_kernel_unproven"):
        _validate_kernel_lineage(
            lineage,
            {
                "purpose": "query_time" if stage == "execution" else "compiler_dry_run",
                "run_id": "runtime",
            },
            {"run_id": "runtime", "round_id": int(stage == "doubter")},
        )
    assert json_bytes(lineage) == before


def test_later_patch_retains_exact_doubter_receipt_with_older_maintenance_proof(monkeypatch):
    from pmmc.experiments import step8_compile_job_maintenance as binding

    lineage, trace, program, _, validator = maintenance_kernel_lineage(monkeypatch, "doubter")
    expected = lineage[0]["receipt"]["receipt_hash"]
    monkeypatch.setattr(
        binding, "retained_producer",
        lambda job, receipt, campaign, actual: receipt.content_hash == expected,
    )
    before = json_bytes(lineage)
    _validate_kernel_lineage(lineage, trace, program)
    assert json_bytes(lineage) == before
    validator.assert_not_called()


@pytest.mark.parametrize(
    "change",
    ["missing", "invalid", "invalid_proof", "extra_field", "unbound", "tampered", "wrong_digest"],
)
def test_kernel_source_mismatch_requires_bound_maintenance_proof(monkeypatch, change):
    from pmmc.experiments.step8_jobs import ComponentReceipt

    lineage, trace, program, _, validator = maintenance_kernel_lineage(monkeypatch)
    entry = lineage[0]
    payload = entry["receipt"]["payload"]
    if change == "missing":
        payload["output"].pop("maintenance_execution")
    elif change == "invalid":
        payload["output"]["maintenance_execution"] = "not-a-proof"
    elif change == "invalid_proof":
        payload["output"]["maintenance_execution"]["proof"] = "not-a-proof"
    elif change == "extra_field":
        payload["output"]["maintenance_execution"]["bypass"] = True
    elif change == "unbound":
        payload["artifact_hashes"] = []
    elif change == "tampered":
        payload["output"]["maintenance_execution"]["proof"]["policy"] = "other"
    else:
        digest = canonical_content_hash(payload["output"]["maintenance_execution"]["proof"])
        payload["output"]["maintenance_execution"]["sha256"] = digest
        payload["artifact_hashes"] = [digest]
    receipt = ComponentReceipt.model_validate(payload)
    entry["receipt"].update(
        receipt_hash=receipt.content_hash, artifact_hashes=receipt.artifact_hashes
    )
    entry["job"]["receipt_hash"] = receipt.content_hash
    with pytest.raises(DependencyUnproven, match="pmmc_execution_producer_kernel_unproven"):
        _validate_kernel_lineage(lineage, trace, program)
    validator.assert_not_called()


def test_kernel_maintenance_validator_rejection_fails_closed(monkeypatch):
    lineage, trace, program, _, validator = maintenance_kernel_lineage(monkeypatch)
    validator.side_effect = ValueError("maintenance_identity_or_source_transition_invalid")
    with pytest.raises(DependencyUnproven, match="pmmc_execution_producer_kernel_unproven"):
        _validate_kernel_lineage(lineage, trace, program)
    validator.assert_called_once()


def test_kernel_maintenance_validator_unavailable_fails_closed(monkeypatch):
    lineage, trace, program, _, validator = maintenance_kernel_lineage(monkeypatch)
    monkeypatch.setitem(sys.modules, "pmmc.experiments.step8_maintenance", None)
    with pytest.raises(DependencyUnproven, match="pmmc_execution_producer_kernel_unproven"):
        _validate_kernel_lineage(lineage, trace, program)
    validator.assert_not_called()


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("campaign", "campaign_id", "foreign"),
        ("campaign", "protocol_hash", "foreign"),
        ("job", "campaign_id", "foreign"),
        ("job", "job_id", "foreign"),
        ("job", "stage", "foreign"),
        ("job", "status", "pending"),
        ("job", "source_fingerprint", "foreign"),
        ("job", "scientific_input_hash", "foreign"),
        ("job", "dependency_ids", ["foreign"]),
        ("job", "receipt_hash", "foreign"),
        ("receipt", "job_id", "foreign"),
        ("receipt", "receipt_hash", "foreign"),
        ("receipt", "component_identity", "foreign"),
        ("receipt", "artifact_hashes", []),
    ],
)
def test_maintenance_cannot_relax_frozen_kernel_lineage(monkeypatch, section, field, value):
    lineage, trace, program, _, validator = maintenance_kernel_lineage(monkeypatch)
    lineage[0][section][field] = value
    with pytest.raises(DependencyUnproven, match="pmmc_execution_producer_kernel_unproven"):
        _validate_kernel_lineage(lineage, trace, program)
    validator.assert_not_called()


def test_native_kernel_lineage_needs_no_maintenance_module(monkeypatch):
    lineage = kernel_lineage("runtime")
    monkeypatch.setitem(sys.modules, "pmmc.experiments.step8_maintenance", None)
    before = deepcopy(lineage)
    _validate_kernel_lineage(
        lineage,
        {"purpose": "compiler_dry_run", "run_id": "runtime"},
        {"run_id": "runtime", "round_id": 0},
    )
    assert lineage == before


def test_maintenance_preserves_execution_cost_dependencies(monkeypatch):
    facts = execution_graph()
    expected = validate_execution_dependencies(facts)
    lineage, _, _, _, validator = maintenance_kernel_lineage(monkeypatch, "execution")
    facts["kernel_lineage"] = lineage
    before = deepcopy(facts)
    assert (
        validate_execution_dependencies(facts)
        == expected
        == {
            "query": set(),
            "write": {"shared-producer"},
        }
    )
    assert facts == before
    validator.assert_called_once()


@pytest.mark.parametrize(
    "change",
    ["running", "missing_call", "pending_job", "missing_index", "hidden_retry", "projection"],
)
def test_write_graph_missing_terminal_denominator_stays_unknown(tmp_path, change):
    graph = write_graph(Fixture(tmp_path))
    if change == "running":
        graph["state"]["status"] = "running"
    elif change == "missing_call":
        graph["calls"].pop()
    elif change == "pending_job":
        graph["rows"]["compiler_stage_jobs"][0]["status"] = "running"
    elif change == "missing_index":
        graph["collections"].pop()
    elif change == "hidden_retry":
        graph["calls"][0]["retry_count"] = 1
    else:
        graph["rows"]["qbank_round_projection_manifests"].pop()
    with pytest.raises(DependencyUnproven):
        validate_write_graph(graph)


def test_explicit_cpu_execution_has_zero_calls_and_deduplicated_cached_producer():
    assert validate_execution_dependencies(execution_graph()) == {
        "query": set(),
        "write": {"shared-producer"},
    }


@pytest.mark.parametrize(
    "change",
    [
        "missing_step",
        "truncated",
        "foreign_execution",
        "opcode",
        "output_hash",
        "bindings",
        "cache",
        "provider",
        "kernel",
        "registry",
        "producer_source",
    ],
)
def test_execution_denominator_and_cache_proof_fail_closed(change):
    facts = execution_graph()
    if change == "missing_step":
        facts["steps"] = []
    elif change == "truncated":
        facts["program_steps"].append({"step_id": 1, "op_name": "inspect_raw_image"})
    elif change == "foreign_execution":
        facts["steps"][0]["execution_id"] = "other"
    elif change == "opcode":
        facts["program_steps"][0]["op_name"] = "unknown_provider_operator"
    elif change == "output_hash":
        facts["steps"][0]["output"]["new_field"] = "tampered"
    elif change == "bindings":
        facts["execution"]["bindings_hash"] = "wrong"
    elif change == "cache":
        facts["query_vectors"].pop()
    elif change == "provider":
        facts["model_calls"] = [{"model_call_id": "unexplained"}]
    elif change == "kernel":
        facts["kernel_contract"]["provider_calls"] = 1
    elif change == "registry":
        facts["compiler"]["operator_registry_hash"] = "unrecognized-provider-enabled-registry"
    else:
        facts["kernel_lineage"][0]["job"]["source_fingerprint"] = "unrecognized-kernel"
    with pytest.raises(DependencyUnproven):
        validate_execution_dependencies(facts)


def test_write_proof_gap_does_not_erase_verified_research(tmp_path):
    case = Fixture(tmp_path)
    row = case.call("research")
    case.price(row, "4.00")
    case.seal(
        complete=False,
        source_updates={"missing": {"cost_facts": ["cost_dependency_proof_incomplete"]}},
    )
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["cost_usd"] == "4.00"
    assert result["active_path"]["status"] == "incomplete"


@pytest.mark.parametrize("nonempty", [False, True])
def test_collector_uses_borrowed_snapshot_and_real_table_queries(monkeypatch, tmp_path, nonempty):
    from pmmc.infrastructure.database.repositories import PostgresMemoryRepository

    graph = (nonempty_write_graph if nonempty else write_graph)(Fixture(tmp_path))
    graph["compiler"]["metadata"]["conversation_ids"] = []
    tables = {
        **graph["rows"],
        "compiler_runs": [graph["compiler"]],
        "run_manifests": [graph["manifest"]],
        "run_states": [graph["state"]],
        "embedding_collections": graph["collections"],
        "embeddings": graph["vectors"],
        "model_calls": graph["calls"],
        "benchmark_component_receipts": [e["receipt"] for e in graph["kernel_lineage"]],
        "benchmark_stage_jobs": [e["job"] for e in graph["kernel_lineage"]],
        "benchmark_campaigns": [e["campaign"] for e in graph["kernel_lineage"]],
    }

    class Session:
        def begin_nested(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement):
            assert statement.is_select
            name = statement.get_execution_options()["step8_inventory_table"].split(".")[-1]
            return SimpleNamespace(mappings=lambda: tables[name])

    async def memories(self, *args, **kwargs):
        return []

    monkeypatch.setattr(PostgresMemoryRepository, "list_snapshot", memories)
    result = asyncio.run(
        collect_pmmc_cost_dependencies(
            Session(),
            compiler_run_ids=["source-run"],
            execution_ids=[],
            campaign_id="campaign",
            selection_hash="selection",
            database_snapshot={"snapshot": "one"},
        )
    )
    assert result["write_graphs"][0]["status"] == "complete"
    assert result["dependency_model_call_ids"] == sorted(c["model_call_id"] for c in graph["calls"])


def test_execution_consumer_closes_cpu_query_without_recharging_cached_vectors(tmp_path):
    from pmmc.experiments.step8_pmmc_cost_dependencies import _execution_facts

    case = Fixture(tmp_path)
    graph = nonempty_write_graph(case)
    trace = deepcopy(graph["rows"]["program_executions"][0])
    trace.update(
        execution_id="query-exec", run_id="runtime", query_id="query", purpose="query_time"
    )
    execution = _execution_facts(graph, graph["rows"]["program_executions"][0])
    execution["execution"] = trace
    execution["steps"] = [
        s | {k: trace[k] for k in ("execution_id", "run_id", "query_id", "purpose")}
        for s in execution["steps"]
    ]
    execution["kernel_lineage"] = kernel_lineage("runtime", "execution")
    bind_pmmc(case, [graph], [execution])
    snapshot = _Snapshot(case.facts)
    _, owners, calls, _, _ = _normalize(snapshot, case.tables)
    process = object.__new__(_Deployment)
    process.snapshot, process.owners, process.calls = snapshot, owners, calls
    process.one = lambda *args, **kwargs: trace
    process.match_owner = lambda runtime: {
        "query_executor_run_id": "runtime",
        "source_compiler_run_id": "source-run",
    }
    query = {}
    process.execution(
        SimpleNamespace(
            execution_id="query-exec",
            query_id="query",
            binding=SimpleNamespace(plan_id=trace["plan_id"], round_id=2),
        ),
        query,
    )
    assert query == {}
