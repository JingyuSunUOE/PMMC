from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_analysis import load_campaign_facts
from pmmc.experiments.step8_artifacts import json_bytes, write_json
from pmmc.experiments.step8_audit import (
    PROCESS_REQUIRED_TABLES,
    CampaignAuditInventory,
    PrivateAuditArtifact,
)
from pmmc.experiments.step8_cost_analysis import (
    AMORTIZATION_N,
    EVALUATION,
    RUNTIME,
    _Deployment,
    _Snapshot,
    _Unavailable,
    summarize_campaign_costs,
)
from pmmc.models.execution import _generation_manifest
from pmmc.schemas.model import GenerationRequest
from pmmc.schemas.query_executor import (
    ExperimentCell as RuntimeCell,
)
from pmmc.schemas.query_executor import (
    QueryExperimentCellResult,
    QueryRouteDecision,
    runtime_fact_payload,
)
from test_step8_analysis import campaign_fixture
from test_step8_workflow import answer


class Fixture:
    def __init__(self, tmp_path):
        directory, store, *_ = campaign_fixture(tmp_path)
        self.facts = load_campaign_facts(directory, artifact_store=store)
        self.tables: dict[str, list[dict]] = {
            RUNTIME + "model_calls": [],
            EVALUATION + "model_calls": [],
            RUNTIME + "benchmark_budget_approvals": [],
            RUNTIME + "benchmark_spend_reservations": [],
            RUNTIME + "query_runtime_typed_answers": [],
            RUNTIME + "query_experiment_cell_results": [],
            EVALUATION + "evaluation_run_states": [],
        }
        self.artifacts = {}
        self.conformance = []
        self.extra_runs = set()
        self.process = False

    def artifact(self, identity, value, *, kind="cost_fact", scope="runtime"):
        raw = json_bytes(value)
        ref = PrivateAuditArtifact(
            artifact_id=identity,
            kind=kind,
            scope=scope,
            sha256=self.facts.store.put(raw),
            size_bytes=len(raw),
        )
        self.artifacts[identity] = ref
        return ref

    @staticmethod
    def request(identity, model_alias="fixture-model"):
        return GenerationRequest(
            request_id=identity,
            model_alias=model_alias,
            max_output_tokens=512,
            messages=[{"role": "user", "parts": [{"kind": "text", "text": "private"}]}],
        )

    def call(self, identity, *, scope=RUNTIME, request=None, role="answer", **changes):
        row = {
            "model_call_id": identity,
            "run_id": "run",
            "query_id": "private-query",
            "status": "success",
            "model_alias": "fixture-model",
            "actual_model_id": "commit",
            "input_tokens": 10,
            "output_tokens": 2,
            "cost_usd": None,
            "retry_count": 0,
            "parent_model_call_id": None,
            "request_config": {"request_metadata": {"model_role": role}},
            **changes,
        }
        if request is not None:
            row["input_asset_id"] = identity + ":input"
            prefix = "runtime:" if scope == RUNTIME else "evaluation:"
            ref = self.artifact(
                prefix + row["input_asset_id"],
                _generation_manifest(self.request(request, row["model_alias"])),
                kind="model_payload",
                scope="runtime" if scope == RUNTIME else "evaluation",
            )
            row["prompt_hash"] = ref.sha256
        self.tables[scope + "model_calls"].append(row)
        return row

    def settlement(
        self,
        request,
        *,
        amount="1.25",
        state="settled",
        error=None,
        attempt="attempt",
        model_alias="fixture-model",
    ):
        approval = {
            "approval_id": "approval",
            "campaign_id": self.facts.manifest.campaign_id,
            "selection_hash": self.facts.selection.content_hash,
            "rate_card_hash": "a" * 64,
        }
        self.tables[RUNTIME + "benchmark_budget_approvals"] = [approval]
        intent = {
            "approval": approval,
            "request": self.request(request, model_alias).model_dump(mode="json"),
        }
        digest = canonical_content_hash(intent)
        self.artifact("provider-dispatch:" + digest, {**intent, "attempt_id": attempt})
        envelope = {
            "request_id": request,
            "request_hash": digest,
            "approval": approval,
            "result": None,
            "error": error,
        }
        response = self.artifact("provider-envelope:" + digest, envelope, kind="model_payload")
        self.tables[RUNTIME + "benchmark_spend_reservations"].append(
            {
                "request_id": request,
                "approval_id": "approval",
                "request_hash": digest,
                "response_hash": response.sha256,
                "state": state,
                "actual_usd": amount,
                "reserved_usd": "999.00",
                "provider": "fixture",
            }
        )

    def seal(self, *, complete=True, source_updates=None):
        for name, rows in self.tables.items():
            self.artifact(
                "table:" + name,
                {"table": name, "rows": rows},
                scope="evaluation" if name.startswith(EVALUATION) else "runtime",
            )
        calls = self.tables[RUNTIME + "model_calls"] + self.tables[EVALUATION + "model_calls"]
        source = {
            "schema_version": "step8-database-audit-snapshot-v1",
            "campaign_id": self.facts.manifest.campaign_id,
            "selection_hash": self.facts.selection.content_hash,
            "source_role": "evaluator",
            "isolation": "repeatable_read_read_only",
            "database_snapshot": {"database_role": "pmmc_evaluator", "snapshot": "1:2:"},
            "run_ids": sorted({r["run_id"] for r in calls} | self.extra_runs),
            "evaluation_run_ids": [],
            "conformance_run_ids": self.conformance,
            "table_hashes": {
                name: canonical_content_hash(rows) for name, rows in self.tables.items()
            },
            "table_counts": {name: len(rows) for name, rows in self.tables.items()},
            "missing": {},
            **(source_updates or {}),
        }
        ref = self.artifact("inventory-provenance", source, kind="performance", scope="evaluation")
        if self.process:
            self.artifact(
                "process-facts:index",
                {
                    "schema_version": "step8-process-facts-v1",
                    "campaign_id": self.facts.manifest.campaign_id,
                    "source_role": "evaluator",
                    "database_snapshot": source["database_snapshot"],
                    "status": "complete",
                    "missing": [],
                    "scope": {"corpus_scope": "referenced_documents_and_windows_not_entire_corpus"},
                    "tables": {
                        name: {
                            "artifact_id": "table:" + name,
                            "sha256": self.artifacts["table:" + name].sha256,
                            "row_count": len(rows),
                            "status": "available",
                        }
                        for name, rows in self.tables.items()
                        if name.startswith(RUNTIME)
                    },
                },
                kind="performance",
            )
        inventory = CampaignAuditInventory(
            campaign_id=self.facts.manifest.campaign_id,
            selection_hash=self.facts.selection.content_hash,
            source_role="evaluator",
            source_snapshot_sha256=ref.sha256,
            scope="all_campaign_attempts",
            complete=complete,
            coverage=("cost_facts", "runtime_answers", "variant_cell_reuse")
            + (("process_facts",) if self.process else ()),
            conformance_run_ids=tuple(self.conformance),
            expected_records=(("attempt", "attempt"),),
            artifacts=tuple(self.artifacts.values()),
        )
        write_json(
            self.facts.directory / "private/audit-inventory.json", inventory.model_dump(mode="json")
        )
        self.facts.files.pop("private/audit-inventory.json", None)
        return inventory

    def bind_answers(self, *, abstain=False):
        definitions = {c.cell_id: c for c in self.facts.cells}
        for job_id, receipt in list(self.facts.receipts.items()):
            report = deepcopy(receipt.output)
            query = report["query_id"]
            dataset = self.facts.jobs[job_id].parameters["dataset"]
            snapshot = self.facts.manifest.snapshot_ids[dataset]
            native = None
            for score in sorted(report["scores"], key=lambda s: s["method"] == "pmmc"):
                definition = definitions[score["cell_id"]]
                if definition.method != "pmmc":
                    identifier = canonical_content_hash([dataset, query, definition.method])
                    value = answer(
                        "abstention" if abstain else "text",
                        text="PRIVATE_ANSWER",
                        method=definition.method,
                        run_id=dataset + ":" + definition.method,
                        variant="primary",
                        identifier=identifier,
                    )
                    value = value.model_copy(
                        update={
                            "query_id": query,
                            "snapshot_id": snapshot,
                            "grounding": value.grounding.model_copy(update={"query_id": query})
                            if value.grounding
                            else None,
                        }
                    )
                    raw = value.model_dump(mode="json")
                    row = {
                        "typed_answer_id": value.typed_answer_id,
                        "query_executor_run_id": value.run_id,
                        "query_id": query,
                        "answer_model_call_id": value.answer_model_call_id,
                        "terminal_state": value.terminal_state,
                        "payload": {"metadata": {"typed_result": raw}},
                        "content_hash": canonical_content_hash(raw),
                    }
                    self.tables[RUNTIME + "query_runtime_typed_answers"].append(row)
                    for call in value.calls:
                        actual = value.actual_model_id
                        if call.role != "answer":
                            assert value.grounding is not None
                            actual = value.grounding.actual_model_id
                        self.call(
                            call.model_call_id,
                            role=call.role,
                            run_id=value.run_id,
                            query_id=query,
                            actual_model_id=actual,
                        )
                    if definition.method == "native_multimodal_rag":
                        native = value
                    score.update(
                        source_fact_id=value.typed_answer_id,
                        source_fact_hash=row["content_hash"],
                        typed_answer_id=value.typed_answer_id,
                        terminal_state=value.terminal_state,
                    )
                else:
                    assert native is not None
                    value = native
                    run = dataset + ":pmmc"
                    runtime_cell = RuntimeCell(
                        question_budget=definition.question_budget,
                        revision_depth=definition.round_id,
                        routing_policy_hash="policy",
                        seed=42,
                        source_projection_id="projection",
                        source_projection_hash="projection-hash",
                    )
                    route = QueryRouteDecision(
                        route_decision_id="route",
                        query_executor_run_id=run,
                        match_batch_id="match-" + value.typed_answer_id,
                        query_id=query,
                        outcome="fallback",
                        match_threshold=0.7,
                        gap_threshold=0.1,
                        fallback_reason="no_qbank_candidate",
                        content_hash="route-hash",
                    )
                    result = QueryExperimentCellResult(
                        cell_result_id="cell-" + value.typed_answer_id,
                        query_executor_run_id=run,
                        query_id=query,
                        cell_id=canonical_content_hash(runtime_cell.model_dump(mode="json")),
                        cell=runtime_cell,
                        route=route,
                        fallback_answer_id=value.typed_answer_id,
                        selected_answer_id=value.typed_answer_id,
                        terminal_state=value.terminal_state,
                        active_fallback=True,
                        content_hash="pending",
                    )
                    hashed = result.model_dump(
                        mode="json", exclude={"created_at", "cell_result_id", "content_hash"}
                    )
                    result = result.model_copy(
                        update={
                            "content_hash": canonical_content_hash(runtime_fact_payload(hashed))
                        }
                    )
                    self.tables[RUNTIME + "query_experiment_cell_results"].append(
                        {
                            **{
                                name: getattr(result, name)
                                for name in (
                                    "cell_result_id",
                                    "query_executor_run_id",
                                    "query_id",
                                    "cell_id",
                                    "selected_answer_id",
                                    "terminal_state",
                                    "content_hash",
                                )
                            },
                            "question_budget": definition.question_budget,
                            "revision_depth": definition.round_id,
                            "treatment": "pmmc",
                            "payload": result.model_dump(mode="json"),
                        }
                    )
                    score.update(
                        source_fact_id=result.cell_result_id,
                        source_fact_hash=result.content_hash,
                        typed_answer_id=value.typed_answer_id,
                        terminal_state=value.terminal_state,
                    )
            report["evaluation_run_id"] = "eval-" + job_id
            self.tables[EVALUATION + "evaluation_run_states"].append(
                {
                    "evaluation_run_id": report["evaluation_run_id"],
                    "status": "completed",
                    "metadata": {"completion_report": report},
                }
            )
            self.facts.receipts[job_id] = receipt.model_copy(
                update={
                    "output": report,
                    "artifact_hashes": [self.facts.store.put_json(report)],
                }
            )

    def price(self, row, amount="1.25"):
        request = "request-" + row["model_call_id"]
        row["input_asset_id"] = "input-" + row["model_call_id"]
        row["prompt_hash"] = self.artifact(
            "runtime:" + row["input_asset_id"],
            _generation_manifest(self.request(request, row["model_alias"])),
            kind="model_payload",
        ).sha256
        self.settlement(request, amount=amount, model_alias=row["model_alias"])

    def bind_process(self):
        self.process = True
        for name in PROCESS_REQUIRED_TABLES:
            self.tables.setdefault(name, [])
        calls = {r["model_call_id"]: r for r in self.tables[RUNTIME + "model_calls"]}
        run_ids, shared = set(), {}
        for row in self.tables[RUNTIME + "query_runtime_typed_answers"]:
            value = row["payload"]["metadata"]["typed_result"]
            run, query, snapshot = value["run_id"], value["query_id"], value["snapshot_id"]
            if run not in run_ids:
                self.tables[RUNTIME + "run_manifests"].append(
                    {"run_id": run, "method": value["method"]}
                )
                run_ids.add(run)
            self.tables[RUNTIME + "run_queries"].append(
                {"run_id": run, "query_id": query, "snapshot_id": snapshot}
            )
            context_id = "context-" + row["typed_answer_id"]
            calls[row["answer_model_call_id"]]["context_assembly_id"] = context_id
            batch_id = None
            if value["method"].endswith("_rag"):
                batch_id = "batch-" + row["typed_answer_id"]
                if run not in shared:
                    shared[run] = self.call(
                        "embed-" + run, run_id=run, query_id=None, role="query_embedding"
                    )
                vector_id, collection = "vector-" + row["typed_answer_id"], "collection-" + run
                self.tables[RUNTIME + "retrieval_batches"].append(
                    {
                        "retrieval_batch_id": batch_id,
                        "run_id": run,
                        "query_id": query,
                        "snapshot_id": snapshot,
                        "status": "success",
                        "reranker_model_call_id": None,
                    }
                )
                self.tables[RUNTIME + "retrieval_batch_query_embeddings"].append(
                    {
                        "retrieval_batch_id": batch_id,
                        "run_id": run,
                        "query_id": query,
                        "snapshot_id": snapshot,
                        "embedding_id": vector_id,
                        "query_embedding_collection_id": collection,
                        "used_for_search": True,
                    }
                )
                self.tables[RUNTIME + "embeddings"].append(
                    {
                        "embedding_id": vector_id,
                        "collection_id": collection,
                        "model_call_id": shared[run]["model_call_id"],
                    }
                )
                if not any(
                    c["collection_id"] == collection
                    for c in self.tables[RUNTIME + "embedding_collections"]
                ):
                    self.tables[RUNTIME + "embedding_collections"].append(
                        {
                            "collection_id": collection,
                            "snapshot_id": snapshot,
                        }
                    )
            self.tables[RUNTIME + "context_assemblies"].append(
                {
                    "context_assembly_id": context_id,
                    "run_id": run,
                    "query_id": query,
                    "snapshot_id": snapshot,
                    "method": value["method"],
                    "retrieval_batch_id": batch_id,
                }
            )
        for cell in self.tables[RUNTIME + "query_experiment_cell_results"]:
            run, query = cell["query_executor_run_id"], cell["query_id"]
            snapshot = next(
                value
                for dataset, value in self.facts.manifest.snapshot_ids.items()
                if dataset.value == run.split(":")[0]
            )
            compiler = "compiler-" + run
            if not any(
                r["query_executor_run_id"] == run
                for r in self.tables[RUNTIME + "query_executor_runs"]
            ):
                self.tables[RUNTIME + "query_executor_runs"].append(
                    {
                        "query_executor_run_id": run,
                        "source_compiler_run_id": compiler,
                        "routing_projection_id": "routing-" + run,
                        "snapshot_id": snapshot,
                    }
                )
                self.tables[RUNTIME + "compiler_runs"].append(
                    {"run_id": compiler, "snapshot_id": snapshot}
                )
            batch_id = cell["payload"]["route"]["match_batch_id"]
            qbank = "candidate-" + batch_id
            assessment = "assess-" + batch_id
            self.tables[RUNTIME + "qbank_match_batches"].append(
                {
                    "match_batch_id": batch_id,
                    "query_executor_run_id": run,
                    "query_id": query,
                    "active_round": cell["revision_depth"],
                    "question_budget": cell["question_budget"],
                    "source_compiler_run_id": compiler,
                    "routing_projection_id": "routing-" + run,
                    "snapshot_id": snapshot,
                    "fused_count": 1,
                    "conversation_id": "conversation",
                }
            )
            self.tables[RUNTIME + "qbank_match_fused_candidates"].append(
                {
                    "match_batch_id": batch_id,
                    "qbank_id": qbank,
                    "rank": 1,
                    "route_class": "answerable",
                    "raw_rrf": 0.01,
                    "metadata": {"reranker_assessment_id": assessment},
                }
            )
            call = self.call("rerank-" + batch_id, run_id=run, query_id=query, role="reranker")
            self.tables[RUNTIME + "qbank_reranker_assessments"].append(
                {
                    "reranker_assessment_id": assessment,
                    "query_executor_run_id": run,
                    "query_id": query,
                    "qbank_id": qbank,
                    "model_call_id": call["model_call_id"],
                }
            )

    def restrict_methods(self, methods):
        self.facts.cells = tuple(c for c in self.facts.cells if c.method in methods)
        for job_id, receipt in list(self.facts.receipts.items()):
            output = deepcopy(receipt.output)
            output["scores"] = [s for s in output["scores"] if s["method"] in methods]
            self.facts.receipts[job_id] = receipt.model_copy(
                update={
                    "output": output,
                    "artifact_hashes": [self.facts.store.put_json(output)],
                }
            )
            state = next(
                r
                for r in self.tables[EVALUATION + "evaluation_run_states"]
                if r["evaluation_run_id"] == output["evaluation_run_id"]
            )
            state["metadata"]["completion_report"] = output
        self.facts.manifest = self.facts.manifest.model_copy(
            update={
                "matrix_hash": canonical_content_hash(
                    [c.model_dump(mode="json") for c in self.facts.cells]
                ),
            }
        )

    def bind_match_embeddings(self):
        self.tables[RUNTIME + "query_embeddings"] = []
        self.tables[RUNTIME + "query_embedding_batches"] = []
        for run in self.tables[RUNTIME + "query_executor_runs"]:
            run.update(schema_version="0.3.0", config_hash="frozen", metadata={})
        for match in self.tables[RUNTIME + "qbank_match_batches"]:
            for channel in ("text", "multimodal"):
                identity = match["match_batch_id"] + ":" + channel
                call = self.call(
                    "cache-call-" + identity,
                    run_id="cache-owner-" + identity,
                    query_id=match["query_id"],
                    role="query_embedding",
                )
                self.tables[RUNTIME + "query_embedding_batches"].append(
                    {
                        "query_embedding_batch_id": identity,
                        "query_executor_run_id": match["query_executor_run_id"],
                        "snapshot_id": match["snapshot_id"],
                        "channel": channel,
                        "query_ids": [match["query_id"]],
                        "expected_count": 1,
                        "committed_count": 1,
                        "model_call_id": call["model_call_id"],
                        "metadata": {"runtime_owned_reuse": True},
                    }
                )
                self.tables[RUNTIME + "query_embeddings"].append(
                    {
                        "query_embedding_batch_id": identity,
                        "query_executor_run_id": match["query_executor_run_id"],
                        "snapshot_id": match["snapshot_id"],
                        "channel": channel,
                        "query_id": match["query_id"],
                        "embedding_id": identity,
                        "collection_id": identity,
                    }
                )
                self.tables[RUNTIME + "embeddings"].append(
                    {
                        "embedding_id": identity,
                        "collection_id": identity,
                        "model_call_id": call["model_call_id"],
                    }
                )
                self.tables[RUNTIME + "embedding_collections"].append(
                    {
                        "collection_id": identity,
                        "snapshot_id": match["snapshot_id"],
                    }
                )


def test_absent_inventory_is_safe_incomplete(tmp_path):
    case = Fixture(tmp_path)
    result = summarize_campaign_costs(case.facts)
    assert result["status"] == "incomplete"
    assert result["actual_research"]["totals"] is None
    assert result["actual_research"]["reasons"] == ["inventory_or_cas_unavailable"]


def test_physical_duplicates_unknown_self_host_and_all_research_categories(tmp_path):
    case = Fixture(tmp_path)
    row = case.call("paid", request="request")
    case.tables[RUNTIME + "model_calls"].append(deepcopy(row))
    case.settlement("request")
    case.call("local", cost_usd=999.0, model_alias="gpt-known-name-must-not-price")
    case.call("failed", status="failed", input_tokens=None, output_tokens=None)
    case.call("retry", role="answer_format_retry", parent_model_call_id="failed")
    case.call("judge", scope=EVALUATION)
    case.call("probe", run_id="conformance")
    case.conformance = ["conformance"]
    case.seal()
    result = summarize_campaign_costs(case.facts)
    actual = result["actual_research"]
    assert actual["status"] == "complete" and actual["pricing_status"] == "incomplete"
    assert actual["totals"]["count"] == 6
    assert actual["totals"]["cost_usd"] is None
    assert actual["totals"]["known_usd_subtotal"] == "1.25"
    assert actual["totals"]["unknown_usd_count"] == 5
    for category in ("failed_or_unknown", "explicit_retry", "evaluator", "conformance"):
        assert result["research_breakdown"][category]["count"] == 1
    assert all(r["total"] is None for r in actual["totals"]["resources"] if "gpu" in r["name"])
    public = json.dumps(result)
    assert "private-query" not in public and "gpt-known-name" not in public
    assert "PRIVATE_ANSWER" not in public


def test_verified_dispatch_replay_deduplicates_distinct_traces(tmp_path):
    case = Fixture(tmp_path)
    case.call("one", request="same")
    case.call("two", request="same")
    case.settlement("same", amount="0.50")
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["count"] == 1
    assert result["actual_research"]["totals"]["cost_usd"] == "0.50"


@pytest.mark.parametrize("state", ["unknown", "reserved", "cancelled"])
def test_reserved_budget_is_never_actual_spend(tmp_path, state):
    case = Fixture(tmp_path)
    case.call("call", request="request")
    case.settlement("request", amount="200", state=state)
    case.seal()
    totals = summarize_campaign_costs(case.facts)["actual_research"]["totals"]
    assert totals["cost_usd"] is None and totals["known_usd_subtotal"] == "0"


def test_orphan_dispatch_counts_once_without_a_model_trace(tmp_path):
    case = Fixture(tmp_path)
    case.conformance = ["probe"]
    case.settlement("orphan", amount="0.20", error={"error_code": "failure"}, attempt="probe")
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["cost_usd"] == "0.20"
    assert result["research_breakdown"]["unpersisted_dispatch"]["count"] == 1
    assert result["research_breakdown"]["conformance"]["count"] == 1


def test_unlinked_dispatch_is_not_double_counted_as_a_known_total(tmp_path):
    case = Fixture(tmp_path)
    case.call("unbound")
    case.settlement("maybe-same")
    case.seal()
    actual = summarize_campaign_costs(case.facts)["actual_research"]
    assert actual["status"] == "incomplete" and actual["totals"]["cost_usd"] is None
    assert actual["totals"]["known_usd_subtotal"] == "0"
    assert "unlinked_dispatch_may_duplicate_trace" in actual["reasons"]


@pytest.mark.parametrize("abstain", [False, True])
def test_proven_native_baseline_fallback_reuse_is_partial_not_full_active_cost(tmp_path, abstain):
    case = Fixture(tmp_path)
    case.bind_answers(abstain=abstain)
    case.seal()
    result = summarize_campaign_costs(case.facts)
    projection = result["answer_role_projection"]
    assert projection["status"] == "complete", projection
    assert projection["expected_query_cell_count"] == projection["proven_query_cell_count"] == 36
    assert projection["shared_selected_answer_count"] == 6
    assert projection["reused_answer_cell_count"] == 6
    assert projection["role_use_counts"]["native_mm_baseline:answer"] == 12
    assert projection["role_use_counts"]["native_mm_fallback:answer"] == 12
    assert result["actual_research"]["totals"]["count"] == (30 if abstain else 60)
    assert (
        projection["active_use_count"] == projection["logical_use_count"] == (36 if abstain else 72)
    )
    assert result["active_path"]["status"] == "incomplete"
    assert result["amortization_n"] == list(AMORTIZATION_N)
    assert all(r["cost_usd"] is None and r["reason"] for r in result["amortized"])
    assert result["latency"]["value"] is None
    assert not any("PRIVATE_" in str(value) for value in result.values())


@pytest.mark.parametrize("missing", ["answer", "call", "completion", "source_cell"])
def test_missing_use_proof_preserves_full_denominator_and_research(tmp_path, missing):
    case = Fixture(tmp_path)
    case.bind_answers()
    name = {
        "answer": "query_runtime_typed_answers",
        "call": "model_calls",
        "completion": "evaluation_run_states",
        "source_cell": "query_experiment_cell_results",
    }[missing]
    scope = EVALUATION if missing == "completion" else RUNTIME
    case.tables[scope + name].pop()
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["status"] == "complete"
    projected = result["answer_role_projection"]
    assert projected["status"] == "incomplete"
    assert projected["expected_query_cell_count"] == 36
    assert projected["missing_query_cell_count"] > 0


@pytest.mark.parametrize("corruption", ["source_hash", "table_count", "selection", "role", "cas"])
def test_inventory_and_source_snapshot_proof_fail_closed(tmp_path, corruption):
    case = Fixture(tmp_path)
    case.call("call")
    overrides: dict[str, Any] = {}
    if corruption == "table_count":
        overrides["table_counts"] = {}
    if corruption == "selection":
        overrides["selection_hash"] = "f" * 64
    if corruption == "role":
        overrides["database_snapshot"] = {"database_role": "pmmc_runtime", "snapshot": "1:2:"}
    inventory = case.seal(source_updates=overrides)
    if corruption == "source_hash":
        inventory = inventory.model_copy(update={"source_snapshot_sha256": "f" * 64})
        write_json(
            case.facts.directory / "private/audit-inventory.json", inventory.model_dump(mode="json")
        )
    if corruption == "cas":
        case.facts.store.verify(case.artifacts["table:" + RUNTIME + "model_calls"].sha256).unlink()
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["status"] == "incomplete"
    assert result["actual_research"]["totals"] is None


@pytest.mark.parametrize("mutation", ["conflict", "negative", "retry"])
def test_conflicting_measurements_or_hidden_retries_cannot_claim_complete(tmp_path, mutation):
    case = Fixture(tmp_path)
    row = case.call("call")
    if mutation == "conflict":
        case.tables[RUNTIME + "model_calls"].append({**row, "input_tokens": 999})
    elif mutation == "negative":
        row["input_tokens"] = -1
    else:
        row["retry_count"] = 2
    case.seal()
    actual = summarize_campaign_costs(case.facts)["actual_research"]
    assert actual["status"] == "incomplete"
    assert actual["totals"] is None or actual["totals"]["cost_usd"] is None


def test_missing_rate_card_metadata_does_not_use_model_name_or_trace_price(tmp_path):
    case = Fixture(tmp_path)
    case.call("call", request="request", cost_usd=42)
    case.settlement("request")
    case.tables[RUNTIME + "benchmark_budget_approvals"][0]["rate_card_hash"] = None
    case.seal()
    assert summarize_campaign_costs(case.facts)["actual_research"]["totals"] is None


def test_incomplete_empty_inventory_does_not_report_zero_complete_cost(tmp_path):
    case = Fixture(tmp_path)
    case.seal(complete=False)
    actual = summarize_campaign_costs(case.facts)["actual_research"]
    assert actual["status"] == "incomplete" and actual["totals"]["cost_usd"] is None


def test_offline_summary_does_not_write_or_use_network_database_or_gold(tmp_path, monkeypatch):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    inventory = case.seal()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    import socket

    from sqlalchemy.ext.asyncio import AsyncSession

    def forbidden(*args, **kwargs):
        raise AssertionError("external I/O forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(AsyncSession, "execute", forbidden)
    original = case.facts.store.verify

    def verify(digest):
        assert digest in {a.sha256 for a in inventory.artifacts} | {
            h for receipt in case.facts.receipts.values() for h in receipt.artifact_hashes
        }
        return original(digest)

    monkeypatch.setattr(case.facts.store, "verify", verify)
    first = summarize_campaign_costs(case.facts)
    assert first == summarize_campaign_costs(case.facts)
    assert first["answer_role_projection"]["status"] == "complete"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_partial_answer_proof_does_not_import_workflow_or_need_a_live_facade(tmp_path):
    case = Fixture(tmp_path)
    case.seal()
    facts = replace(case.facts, verified_jobs=set())
    report = summarize_campaign_costs(facts)
    assert report["answer_role_projection"]["missing_query_cell_count"] == 36


@pytest.mark.parametrize("tamper", ["typed_hash", "runtime_hash", "model", "status", "role"])
def test_answer_use_rejects_tampered_lineage_without_losing_denominator(tmp_path, tamper):
    case = Fixture(tmp_path)
    case.bind_answers()
    if tamper == "typed_hash":
        case.tables[RUNTIME + "query_runtime_typed_answers"][0]["content_hash"] = "bad"
    elif tamper == "runtime_hash":
        case.tables[RUNTIME + "query_experiment_cell_results"][0]["payload"]["active_fallback"] = (
            False
        )
    else:
        row = case.tables[RUNTIME + "model_calls"][0]
        if tamper == "model":
            row["actual_model_id"] = "different-model"
        elif tamper == "status":
            row["status"] = "failed"
        else:
            row["request_config"]["request_metadata"]["model_role"] = "unrelated"
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["status"] == "complete"
    assert result["answer_role_projection"]["status"] == "incomplete"
    assert result["answer_role_projection"]["expected_query_cell_count"] == 36
    assert result["answer_role_projection"]["missing_query_cell_count"] > 0


def test_shared_paid_native_answer_is_research_once_but_two_logical_roles(tmp_path):
    case = Fixture(tmp_path)
    case.bind_answers()
    row = next(
        r
        for r in case.tables[RUNTIME + "model_calls"]
        if (
            r["run_id"].endswith("native_multimodal_rag")
            and r["request_config"]["request_metadata"]["model_role"] == "answer"
        )
    )
    row["input_asset_id"] = "paid-input"
    row["prompt_hash"] = case.artifact(
        "runtime:paid-input",
        _generation_manifest(case.request("paid")),
        kind="model_payload",
    ).sha256
    case.settlement("paid", amount="3.50")
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["known_usd_subtotal"] == "3.50"
    priced_uses = [
        r
        for r in result["answer_role_projection"]["method_costs"]
        if r["totals"]["known_usd_subtotal"] == "3.50"
    ]
    assert {(r["method"]["method"], r["scope"]) for r in priced_uses} == {
        (method, scope)
        for method in ("pmmc", "native_multimodal_rag")
        for scope in ("active_path", "logical_ablation")
    }
    assert all(r["totals"]["cost_usd"] is None for r in priced_uses)


def test_parent_call_lineage_alone_does_not_misclassify_grounding_as_retry(tmp_path):
    case = Fixture(tmp_path)
    case.call("answer")
    case.call("ground", role="answer_grounding", parent_model_call_id="answer")
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["research_breakdown"]["explicit_retry"]["count"] == 0


FULL_CONTEXT = {"full_context_captions", "full_context_raw_images"}


def test_break_even_uses_incremental_write_and_same_backbone_scope():
    from pmmc.experiments.step8_cost_analysis import _break_even_comparisons

    def cell(method, write, query, **updates):
        return {
            "cell_id": method,
            "method": method,
            "dataset": "memlens",
            "answer_model": "9b",
            "treatment": "main",
            "question_budget": 8,
            "round_id": 2,
            "source_variant": "active",
            "selected_query_count": 10,
            "write": {"status": "complete", "totals": {"cost_usd": write}},
            "query": {"status": "complete", "totals": {"cost_usd": query}},
            **updates,
        }

    ours = cell("pmmc", "20", "10")
    baseline = cell("native_multimodal_rag", "10", "30")
    report = _break_even_comparisons([ours, baseline, cell("other", "0", "80", answer_model="27b")])
    assert len(report) == 1
    assert report[0]["incremental_write_usd"] == "10"
    assert report[0]["per_query_saving_usd"] == "2"
    assert report[0]["break_even_queries"] == "5"
    assert report[0]["minimum_queries_for_nonhigher_cost"] == 5
    baseline["write"]["totals"]["cost_usd"] = None
    report = _break_even_comparisons([ours, baseline])[0]
    assert report["status"] == "incomplete"
    assert report["break_even_queries"] is None
    assert report["reason"] == "unknown_model_call_price"
    baseline["write"]["status"] = "incomplete"
    assert (
        _break_even_comparisons([ours, baseline])[0]["reason"]
        == "write_or_query_dependency_proof_incomplete"
    )


def test_break_even_no_query_saving_is_not_unknown_or_zero_cost():
    from pmmc.experiments.step8_cost_analysis import _break_even_comparisons

    common = {
        "dataset": "memlens",
        "answer_model": "9b",
        "treatment": "main",
        "question_budget": 8,
        "round_id": 2,
        "source_variant": "active",
        "selected_query_count": 1,
        "write": {"status": "complete", "totals": {"cost_usd": "0"}},
        "query": {"status": "complete", "totals": {"cost_usd": "2"}},
    }
    ours = {**common, "cell_id": "pmmc", "method": "pmmc"}
    baseline = {**common, "cell_id": "baseline", "method": "full_context_captions"}
    report = _break_even_comparisons([ours, baseline])[0]
    assert report["status"] == "complete" and report["reason"] == "no_positive_query_saving"
    assert report["break_even_queries"] is None


@pytest.mark.parametrize("abstain", [False, True])
def test_full_context_proves_model_call_cost_without_spreading_research(tmp_path, abstain):
    case = Fixture(tmp_path)
    case.bind_answers(abstain=abstain)
    case.bind_process()
    for row in case.tables[RUNTIME + "model_calls"]:
        case.price(row)
    case.price(case.call("unrelated-research", run_id="other-run"), amount="1000")
    case.restrict_methods(FULL_CONTEXT)
    case.seal()
    result = summarize_campaign_costs(case.facts)
    active = result["active_path"]
    assert active["status"] == "complete", active
    assert active["scope"] == "audited_model_calls_only_not_infrastructure_tco"
    assert active["totals"] is None
    assert len(active["cells"]) == 4
    for cell in active["cells"]:
        assert cell["selected_query_count"] == 3
        assert cell["write"]["totals"]["count"] == 0
        assert cell["write"]["totals"]["cost_usd"] == "0"
        assert cell["query"]["totals"]["count"] == (3 if abstain else 6)
        assert cell["query"]["totals"]["cost_usd"] == ("3.75" if abstain else "7.50")
    assert len(result["amortized"]) == 4 * len(AMORTIZATION_N)
    for value in result["amortized"]:
        assert value["n"] in AMORTIZATION_N
        assert value["c_write_usd"] == "0"
        assert value["c_query_usd"] == ("1.25" if abstain else "2.50")
        assert value["cost_usd"] == value["c_query_usd"]
        assert value["reason"] is None
    assert Decimal(result["actual_research"]["totals"]["cost_usd"]) > 1000
    assert result["latency"]["value"] is None


def test_rag_query_batch_is_physical_once_and_each_logical_use_pays(tmp_path):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    for row in case.tables[RUNTIME + "model_calls"]:
        case.price(row)
    case.seal()
    result = summarize_campaign_costs(case.facts)
    active = result["active_path"]
    rag = [c for c in active["cells"] if c["method"].endswith("_rag")]
    assert len(rag) == 6
    for cell in rag:
        assert cell["query"]["status"] == "complete", cell
        assert cell["query"]["totals"]["count"] == 7
        assert cell["query"]["totals"]["cost_usd"] == "8.75"
        assert cell["write"]["status"] == "incomplete"
        assert cell["write"]["totals"]["cost_usd"] is None
        assert (
            "missing_source:complete_document_index_membership_and_producer_calls"
            in cell["write"]["reasons"]
        )
    for logical in active["logical_ablation_costs"]:
        if logical["method"]["method"].endswith("_rag"):
            assert logical["totals"]["count"] == 9
            assert logical["totals"]["known_usd_subtotal"] == "11.25"
            assert logical["totals"]["cost_usd"] is None
            assert logical["status"] == "incomplete"
    for value in result["amortized"]:
        if value["method"].endswith("_rag"):
            assert value["cost_usd"] is None and value["c_write_usd"] is None
            assert Decimal(value["c_query_usd"]) == Decimal("8.75") / 3


def test_pmmc_consumed_reranker_and_native_fallback_exclude_other_research_calls(tmp_path):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    for row in case.tables[RUNTIME + "model_calls"]:
        case.price(row)
    source = case.tables[RUNTIME + "qbank_reranker_assessments"][0]
    case.price(
        case.call(
            "unused-reranker",
            run_id=source["query_executor_run_id"],
            query_id=source["query_id"],
            role="reranker",
        ),
        amount="200",
    )
    case.seal()
    result = summarize_campaign_costs(case.facts)
    pmmc = [c for c in result["active_path"]["cells"] if c["method"] == "pmmc"]
    assert len(pmmc) == 2
    comparisons = result["active_path"]["break_even"]
    assert len(comparisons) == 10
    assert all(c["status"] == "incomplete" and c["break_even_queries"] is None for c in comparisons)
    for cell in pmmc:
        assert cell["query"]["status"] == cell["write"]["status"] == "incomplete"
        # Six Native-MM answer/grounding calls, one shared embedding batch,
        # three consumed rerank calls. The unused reranker remains research only.
        assert cell["query"]["totals"]["count"] == 10
        assert cell["query"]["totals"]["known_usd_subtotal"] == "12.50"
        assert cell["query"]["totals"]["cost_usd"] is None
        assert set(cell["query"]["reasons"]) & {
            "missing_source:pmmc_runtime.query_embeddings",
            "match_embedding_channel_coverage_incomplete",
        }
        assert any("complete_compiler" in reason for reason in cell["write"]["reasons"])
    assert Decimal(result["actual_research"]["totals"]["cost_usd"]) > 200


@pytest.mark.parametrize("missing", ["context", "run", "query", "snapshot", "unassigned_call"])
def test_missing_full_context_proof_never_manufactures_zero_write(tmp_path, missing):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    case.restrict_methods(FULL_CONTEXT)
    context = next(
        r for r in case.tables[RUNTIME + "context_assemblies"] if r["method"] in FULL_CONTEXT
    )
    if missing == "context":
        case.tables[RUNTIME + "context_assemblies"].remove(context)
    elif missing in {"run", "query"}:
        name = "run_manifests" if missing == "run" else "run_queries"
        case.tables[RUNTIME + name] = [
            r for r in case.tables[RUNTIME + name] if r["run_id"] != context["run_id"]
        ]
    elif missing == "snapshot":
        context["snapshot_id"] = "other-snapshot"
    else:
        case.call("unassigned", run_id=context["run_id"], query_id=None)
    case.seal()
    result = summarize_campaign_costs(case.facts)
    broken = [c for c in result["active_path"]["cells"] if c["write"]["status"] == "incomplete"]
    assert broken
    assert all(c["write"]["totals"]["cost_usd"] is None for c in broken)
    ids = {c["cell_id"] for c in broken}
    assert all(
        v["cost_usd"] is None and v["reason"] for v in result["amortized"] if v["cell_id"] in ids
    )
    assert sum(c["selected_query_count"] for c in result["active_path"]["cells"]) == 12


def test_proved_path_with_unknown_self_host_price_still_has_null_amortization(tmp_path):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    case.restrict_methods(FULL_CONTEXT)
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["active_path"]["status"] == "complete"
    assert all(c["query"]["totals"]["cost_usd"] is None for c in result["active_path"]["cells"])
    assert all(
        v["cost_usd"] is None and v["reason"] == "unknown_model_call_price"
        for v in result["amortized"]
    )


@pytest.mark.parametrize("tamper", ["snapshot", "hash", "count"])
def test_process_index_binding_fails_closed_without_losing_research(tmp_path, tamper):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    case.restrict_methods(FULL_CONTEXT)
    inventory = case.seal()
    index = case.facts.store.load_json(case.artifacts["process-facts:index"].sha256)
    if tamper == "snapshot":
        index["database_snapshot"] = {"snapshot": "foreign"}
    else:
        entry = index["tables"][RUNTIME + "context_assemblies"]
        entry["sha256" if tamper == "hash" else "row_count"] = "f" * 64 if tamper == "hash" else 0
    case.artifact("process-facts:index", index, kind="performance")
    write_json(
        case.facts.directory / "private/audit-inventory.json",
        inventory.model_copy(
            update={"artifacts": tuple(case.artifacts.values())},
        ).model_dump(mode="json"),
    )
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["status"] == "complete"
    assert result["active_path"]["status"] == "incomplete"
    assert all(v["cost_usd"] is None for v in result["amortized"])


@pytest.mark.parametrize("tamper", [None, "channel", "snapshot", "coverage", "producer"])
def test_embedding_relations_enable_fallback_query_cost_but_not_invented_write(tmp_path, tamper):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    case.bind_match_embeddings()
    for row in case.tables[RUNTIME + "model_calls"]:
        case.price(row)
    batches = case.tables[RUNTIME + "query_embedding_batches"]
    vectors = case.tables[RUNTIME + "query_embeddings"]
    if tamper == "channel":
        vectors.pop()
    elif tamper == "snapshot":
        batches[-1]["snapshot_id"] = "foreign"
    elif tamper == "coverage":
        batches[-1]["metadata"]["coverage_projection"] = True
    elif tamper == "producer":
        vectors[-1]["embedding_id"] = vectors[0]["embedding_id"]
    case.seal()
    result = summarize_campaign_costs(case.facts)
    cells = [c for c in result["active_path"]["cells"] if c["method"] == "pmmc"]
    if tamper is None:
        for cell in cells:
            assert cell["query"]["status"] == "complete", cell
            assert cell["query"]["totals"]["count"] == 16
            assert cell["query"]["totals"]["cost_usd"] == "20.00"
            assert cell["write"]["status"] == "incomplete"
        for value in result["amortized"]:
            if value["method"] == "pmmc":
                assert value["c_write_usd"] is None and value["cost_usd"] is None
                assert Decimal(value["c_query_usd"]) == Decimal("20.00") / 3
    else:
        assert any(c["query"]["status"] == "incomplete" for c in cells)
        assert all(
            c["query"]["totals"]["cost_usd"] is None
            for c in cells
            if c["query"]["status"] == "incomplete"
        )


@pytest.mark.parametrize("incompatible", [False, True])
def test_source_match_reuse_checks_frozen_owner_and_uses_original_physical_calls(
    tmp_path, incompatible
):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    case.bind_match_embeddings()
    source = case.tables[RUNTIME + "query_executor_runs"][0]
    source_id = source["query_executor_run_id"]
    target = {
        **source,
        "query_executor_run_id": "target-answer-backbone",
        "metadata": {"source_match_run_id": source_id},
    }
    if incompatible:
        target["config_hash"] = "different"
    case.tables[RUNTIME + "query_executor_runs"].append(target)
    cell = next(
        c
        for c in case.tables[RUNTIME + "query_experiment_cell_results"]
        if c["query_executor_run_id"] == source_id
    )
    runtime = QueryExperimentCellResult.model_validate(cell["payload"]).model_copy(
        update={
            "query_executor_run_id": target["query_executor_run_id"],
            "metadata": {"source_match_run_id": source_id},
        }
    )
    case.seal()
    calls = {(RUNTIME, r["model_call_id"]): r for r in case.tables[RUNTIME + "model_calls"]}
    owners = {key: key[1] for key in calls}
    process = _Deployment(_Snapshot(case.facts), owners, calls)
    uses: dict[tuple[str, str], str] = {}
    if incompatible:
        with pytest.raises(_Unavailable, match="source_match_cost_identity_mismatch"):
            process.match_embeddings(runtime, uses)
        assert not uses
    else:
        process.match_embeddings(runtime, uses)
        assert len(uses) == 2
        assert all(
            calls[RUNTIME, identity]["run_id"].startswith("cache-owner-") for identity, _ in uses
        )
        definition = next(
            c
            for c in case.facts.cells
            if c.method == "pmmc"
            and case.facts.manifest.snapshot_ids[c.dataset] == source["snapshot_id"]
        )
        process.match(runtime, definition, uses)
        assert len(uses) == 3


@pytest.mark.parametrize("tamper", [None, "missing", "foreign_run", "feedback_plan", "cycle"])
def test_typed_revision_inherits_original_planner_and_source_doubter_calls(tmp_path, tamper):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    run = case.tables[RUNTIME + "query_executor_runs"][0]
    batch = next(
        b
        for b in case.tables[RUNTIME + "qbank_match_batches"]
        if b["query_executor_run_id"] == run["query_executor_run_id"]
    )
    compiler = run["source_compiler_run_id"]
    for identity in ("questioner", "planner", "d0", "d1", "d2"):
        case.call(identity, run_id=compiler, query_id=None, role=identity)
    candidate = {
        "qbank_id": "qbank",
        "run_id": compiler,
        "generation_rank": 1,
        "conversation_id": batch["conversation_id"],
        "producer_model_call_id": "questioner",
    }
    case.tables[RUNTIME + "qbank_questions"].append(candidate)
    for round_id in range(3):
        case.tables[RUNTIME + "memory_programs"].append(
            {
                "plan_id": f"p{round_id}",
                "round_id": round_id,
                "run_id": compiler,
                "qbank_id": "qbank",
                "snapshot_id": run["snapshot_id"],
                "producer_model_call_id": "planner" if round_id == 0 else None,
                "metadata": {"program_origin": "model" if round_id == 0 else "server_typed_patch"},
            }
        )
        case.tables[RUNTIME + "doubter_feedback"].append(
            {
                "feedback_id": f"feedback-{round_id}",
                "round_id": round_id,
                "plan_id": f"p{round_id}",
                "run_id": compiler,
                "qbank_id": "qbank",
                "producer_model_call_id": f"d{round_id}",
            }
        )
        if round_id:
            case.tables[RUNTIME + "planner_revision_lineage"].append(
                {
                    "revised_plan_id": f"p{round_id}",
                    "source_plan_id": f"p{round_id - 1}",
                    "run_id": compiler,
                    "qbank_id": "qbank",
                    "target_round_id": round_id,
                    "source_feedback_id": f"feedback-{round_id - 1}",
                    "producer_model_call_id": f"d{round_id - 1}",
                }
            )
    binding = {
        "query_program_binding_id": "binding",
        "query_executor_run_id": run["query_executor_run_id"],
        "query_id": batch["query_id"],
        "plan_id": "p2",
        "qbank_id": "qbank",
    }
    case.tables[RUNTIME + "query_program_bindings"].append(binding)
    case.tables[RUNTIME + "qbank_round_projection_manifests"].append(
        {
            "run_id": compiler,
            "round_id": 2,
            "projection_id": "projection-2",
        }
    )
    case.tables[RUNTIME + "qbank_round_projection_members"].append(
        {
            "projection_id": "projection-2",
            "qbank_id": "qbank",
            "effective_plan_id": "p2",
            "effective_feedback_id": "feedback-2",
        }
    )
    lineage = case.tables[RUNTIME + "planner_revision_lineage"]
    if tamper == "missing":
        lineage.pop()
    elif tamper == "foreign_run":
        lineage[-1]["run_id"] = "foreign"
    elif tamper == "feedback_plan":
        case.tables[RUNTIME + "doubter_feedback"][1]["plan_id"] = "wrong"
    elif tamper == "cycle":
        lineage[-1]["source_plan_id"] = "p2"
    case.seal()
    calls = {(RUNTIME, r["model_call_id"]): r for r in case.tables[RUNTIME + "model_calls"]}
    process = _Deployment(_Snapshot(case.facts), {key: key[1] for key in calls}, calls)
    runtime = SimpleNamespace(
        query_id=batch["query_id"],
        binding=SimpleNamespace(**binding),
    )
    definition = SimpleNamespace(round_id=2, question_budget=8)
    uses: dict[tuple[str, str], str] = {}
    if tamper:
        with pytest.raises(_Unavailable):
            process.compiler(runtime, definition, run, batch, uses)
    else:
        process.compiler(runtime, definition, run, batch, uses)
        assert uses == {
            (identity, role): identity
            for identity, role in (
                ("questioner", "questioner_producer"),
                ("planner", "planner_producer"),
                ("d0", "doubter_producer"),
                ("d1", "doubter_producer"),
                ("d2", "doubter_producer"),
            )
        }
        assert sum(role == "planner_producer" for _, role in uses) == 1


def test_r0_planner_cost_never_reads_revision_or_doubter_tables(tmp_path, monkeypatch):
    case = Fixture(tmp_path)
    case.bind_process()
    call = case.call("planner", run_id="compiler", query_id=None)
    case.seal()
    process = _Deployment(
        _Snapshot(case.facts), {(RUNTIME, "planner"): "planner"}, {(RUNTIME, "planner"): call}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("r0 must not read Doubter or revision lineage")

    monkeypatch.setattr(process, "rows", forbidden)
    uses: dict[tuple[str, str], str] = {}
    process.planner_dependencies(
        {"plan_id": "p0", "round_id": 0, "run_id": "compiler", "producer_model_call_id": "planner"},
        uses,
    )
    assert uses == {("planner", "planner_producer"): "planner"}
    with pytest.raises(_Unavailable, match="planner_root_producer_missing"):
        process.planner_dependencies(
            {"plan_id": "bad-p0", "round_id": 0, "producer_model_call_id": None}, {}
        )


def test_external_cache_dependencies_are_active_inputs_not_current_research_spend(tmp_path):
    case = Fixture(tmp_path)
    case.bind_answers()
    case.bind_process()
    case.bind_match_embeddings()
    dependencies = []
    for row in case.tables[RUNTIME + "model_calls"]:
        if row["run_id"].startswith("cache-owner-"):
            dependencies.append(row)
        else:
            case.price(row)
    case.seal(
        source_updates={
            "dependency_model_call_ids": [r["model_call_id"] for r in dependencies],
            "dependency_run_ids": [r["run_id"] for r in dependencies],
        }
    )
    result = summarize_campaign_costs(case.facts)
    research = result["actual_research"]
    assert research["status"] == research["pricing_status"] == "complete"
    assert research["totals"]["count"] == 72
    assert research["totals"]["cost_usd"] == "90.00"
    assert result["research_breakdown"]["runtime"]["count"] == 72
    assert result["reused_dependencies"]["totals"]["count"] == 12
    assert result["reused_dependencies"]["totals"]["cost_usd"] is None
    for cell in result["active_path"]["cells"]:
        if cell["method"] == "pmmc":
            assert cell["query"]["status"] == "complete"
            assert cell["query"]["totals"]["count"] == 16
            assert cell["query"]["totals"]["cost_usd"] is None
            assert cell["query"]["totals"]["unknown_usd_count"] == 6


def test_dependency_labels_cannot_hide_a_current_campaign_paid_dispatch(tmp_path):
    case = Fixture(tmp_path)
    case.call("paid-dependency", run_id="cache-owner", request="paid")
    case.settlement("paid", amount="2.00")
    case.seal(
        source_updates={
            "dependency_model_call_ids": ["paid-dependency"],
            "dependency_run_ids": ["cache-owner"],
        }
    )
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"]["cost_usd"] == "2.00"
    assert result["reused_dependencies"]["totals"]["count"] == 0


def test_request_id_and_alias_do_not_prove_billed_prompt_identity(tmp_path):
    case = Fixture(tmp_path)
    row = case.call("trace", request="same-id")
    case.settlement("same-id")
    payload = _generation_manifest(case.request("same-id"))
    payload["messages"][0]["parts"][0]["text"] = "PRIVATE-substituted-prompt"
    row["prompt_hash"] = case.artifact(
        "runtime:" + row["input_asset_id"], payload, kind="model_payload"
    ).sha256
    case.seal()
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"] is None
    assert result["actual_research"]["reasons"] == ["settlement_trace_request_content_mismatch"]
    assert "PRIVATE" not in str(result)


@pytest.mark.parametrize("field", ["database_snapshot", "missing", "table_counts", "table_hashes"])
def test_malformed_source_proof_is_incomplete_not_an_exception(tmp_path, field):
    case = Fixture(tmp_path)
    case.seal(source_updates={field: None})
    result = summarize_campaign_costs(case.facts)
    assert result["actual_research"]["totals"] is None
