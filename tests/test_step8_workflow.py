from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation import step8_workflow as workflow
from pmmc.evaluation.config import BaselineEvaluationConfig
from pmmc.evaluation.paper_statistics import TerminalStatus, TypedAnswer
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.schemas import DataScope, GoldEvidence, ImageRef, MemoryUnit, QueryGold, QueryInput
from pmmc.schemas.answer_output import (
    AnswerContractV3,
    AnswerRoleCall,
    GroundingAssessment,
    GroundingFact,
    TypedAnswerImageItem,
    TypedAnswerResult,
)
from pmmc.schemas.query_executor import runtime_fact_payload
from pmmc.schemas.step8 import BASELINE_METHODS

ROOT = Path(__file__).resolve().parents[1]


def test_pmmc_scope_keeps_full_assignments_and_requires_native_fallback():
    cells = [SimpleNamespace(
        cell_id=str(i), method=m, payload={"method": m, "treatment": "main"},
    ) for i, m in enumerate(["pmmc", "pmmc", *BASELINE_METHODS])]
    whole = workflow.evaluation_definitions(cells, "main", "all")
    scoped = workflow.evaluation_definitions(cells, "main", "pmmc")
    assert scoped == tuple(c for c in whole if c["method"] == "pmmc")
    assert len(scoped) == 2
    with pytest.raises(ValueError, match="assignments_missing"):
        workflow.evaluation_definitions(cells[:-1], "main", "pmmc")
    workflow.validate_baseline_links({"native_multimodal_rag": "fallback"}, "main", "pmmc")
    with pytest.raises(ValueError, match="baseline_link"):
        workflow.validate_baseline_links({}, "main", "pmmc")
    with pytest.raises(ValueError, match="baseline_link"):
        workflow.validate_baseline_links({"native_multimodal_rag": "fallback"}, "main", "all")


def test_priority_native_phases_do_not_complete_deferred_dag_jobs(monkeypatch):
    from pmmc.experiments import step8_priority as priority

    backend = SimpleNamespace(
        preflight=AsyncMock(),
        _parent_value=lambda job, key: "run" if key == "query_executor_run_id" else "suite",
        _answer_settings=lambda *args: "settings", _query_phase=AsyncMock(return_value={
            "query_executor_run_id": "run", "suite_id": "suite", "phase": "select",
        }), settings="settings", campaign=SimpleNamespace(campaign_id="campaign"),
    )
    job = SimpleNamespace(operation="query_answer", parameters={
        "treatment": "main", "answer_model": "answer", "query_id": "query",
    })
    suite = AsyncMock()
    evaluate = AsyncMock(return_value={"status": "complete", "method_scope": "pmmc"})
    monkeypatch.setattr(priority, "run_baseline_suite", suite)
    monkeypatch.setattr(priority, "evaluate_step8_query", evaluate)
    selected = asyncio.run(priority.complete_primary_query(backend, job))
    assert [m.value for m in suite.call_args.kwargs["methods"]] == ["native_multimodal_rag"]
    assert suite.call_args.kwargs["phase"].value == "answer"
    backend._query_phase.assert_awaited_once_with(job, job.parameters, "select")
    asyncio.run(priority.evaluate_primary_query(backend, job, selected))
    assert evaluate.call_args.kwargs == {"method_scope": "pmmc"}
    assert not hasattr(backend, "execute")


def test_priority_scoring_scope_has_distinct_receipt_but_shared_judge_identity(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer())], QueryGold(answer="May"))
    pmmc = fake.run(method_scope="pmmc")
    identity = fake.judge.call_args.kwargs["judge_identity"]
    fake.state["cached"] = None
    all_methods = fake.run()
    assert pmmc["method_scope"] == "pmmc" and not pmmc["all_methods_evaluated"]
    assert pmmc["scores"] == all_methods["scores"]
    assert pmmc["identity_hash"] != all_methods["identity_hash"]
    assert identity == fake.judge.call_args.kwargs["judge_identity"]


def test_judge_trace_reuse_keeps_original_evaluation_provenance(monkeypatch):
    trace = SimpleNamespace(status=SimpleNamespace(value="success"),
                            evaluation_run_id="original-evaluation")
    calls = SimpleNamespace(get=AsyncMock(return_value=trace))

    class Unit:
        async def __aenter__(self):
            return SimpleNamespace(model_calls=calls)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(workflow, "ExperimentUnitOfWork", lambda *args: Unit())
    scored = Mock(return_value="score")
    monkeypatch.setattr(workflow, "judge_score_from_trace", scored)
    execution = SimpleNamespace(generate_audited=AsyncMock())
    for evaluation in ("pmmc-evaluation", "all-method-evaluation"):
        assert asyncio.run(workflow._judge_one(
            sessions=object(), execution=execution, gateway=object(),
            config=SimpleNamespace(job_max_attempts=1), prompt="fixed", settings=object(),
            evaluation_run_id=evaluation,
            inputs=SimpleNamespace(query=SimpleNamespace(query_id="query")), cell=cell(answer()),
            reference=SimpleNamespace(text="May"), judge_identity="same-judge-contract",
        )) == "score"
        assert scored.call_args.kwargs["trace"] is trace
    assert calls.get.call_args_list[0] == calls.get.call_args_list[1]
    assert trace.evaluation_run_id == "original-evaluation"
    execution.generate_audited.assert_not_awaited()


def answer(
    kind="text",
    *,
    text="May",
    images=(),
    method="pmmc",
    run_id="run",
    variant="pmmc_b8_r2",
    identifier="typed",
):
    supported = kind != "abstention"
    items = [
        TypedAnswerImageItem(
            ordinal=i,
            image_alias=f"I{i + 1:02}",
            image_id=image,
            public_media_ref="media:" + str(i) * 64,
        )
        for i, image in enumerate(images)
    ]
    output = AnswerContractV3(
        answer_kind=kind,
        answer_text=text if kind in {"text", "number", "boolean"} else None,
        selected_image_aliases=[item.image_alias for item in items],
        supported=supported,
        used_evidence_aliases=["E01"] if supported else [],
        abstain_reason=None if supported else "insufficient_evidence",
    )
    calls = [AnswerRoleCall(role="answer", model_call_id="call-" + identifier)]
    grounding = None
    if supported:
        calls.append(AnswerRoleCall(role="answer_grounding", model_call_id="ground-" + identifier))
        grounding = GroundingFact(
            grounding_id="grounding",
            run_id=run_id,
            query_id="query",
            answer_model_call_id=calls[0].model_call_id,
            model_alias="grounding",
            actual_model_id="grounding-commit",
            assessment=GroundingAssessment(
                supported=True,
                query_closure="pass",
                source_support="pass",
                entity_consistency="pass",
                time_consistency="not_applicable",
                set_consistency="not_applicable",
            ),
            used_evidence_aliases=["E01"],
            evidence_content_hash="hash",
            ungated_output_hash="hash",
            calls=calls[1:],
        )
    return TypedAnswerResult(
        typed_answer_id=identifier,
        run_id=run_id,
        snapshot_id="snapshot",
        query_id="query",
        method=method,
        answer_variant=variant,
        output=output,
        ungated_output=output,
        image_items=items,
        ungated_image_items=items,
        answer_model_call_id=calls[0].model_call_id,
        actual_model_id="answer-commit",
        terminal_state="answered" if supported else "abstained",
        grounding=grounding,
        calls=calls,
    )


def cell(value=None, *, terminal=None, identifier="cell", method="pmmc"):
    return workflow.EvaluationCellInput(
        identifier,
        method,
        identifier,
        "active",
        terminal or value.terminal_state,
        "run",
        "fact-" + identifier,
        "hash-" + identifier,
        value,
    )


@pytest.mark.parametrize(
    "kind,images,expected",
    [
        ("image", ("canonical-a",), 1),
        ("image_set", ("canonical-b", "canonical-a"), 1),
        ("ordered_image_list", ("canonical-b", "canonical-a"), 0),
    ],
)
def test_native_image_scoring_uses_selected_ids_not_aliases_or_placeholders(kind, images, expected):
    value = answer(kind, images=images)
    ref = TypedAnswer(kind, image_ids=tuple(sorted(images)))
    observation = workflow.score_cell("query", cell(value), ref)
    assert observation.score.value == expected
    assert workflow.typed_candidate(value).text is None


def test_gold_image_evidence_is_not_an_image_answer():
    gold = QueryGold(evidence=GoldEvidence(image_ids=["not-an-answer"]))
    assert workflow.typed_reference(gold) is None
    explicit = gold.model_copy(
        update={
            "metadata": {
                "typed_answer": {
                    "kind": "image",
                    "image_ids": ["actual-answer"],
                }
            }
        }
    )
    assert workflow.typed_reference(explicit) == TypedAnswer("image", image_ids=("actual-answer",))
    text = QueryGold(answer="[image]")
    result = workflow.score_cell(
        "query", cell(answer("image", images=("actual-answer",))), workflow.typed_reference(text)
    )
    assert result.score.value == 0


@pytest.mark.parametrize(
    "kind,text,reference", [("number", "-1.5", "-1.5"), ("boolean", "true", "True")]
)
def test_legacy_scalar_gold_uses_lossless_typed_parsing(kind, text, reference):
    result = workflow.score_cell(
        "query", cell(answer(kind, text=text)), TypedAnswer("text", text=reference)
    )
    assert result.score.value == 1
    wrong = workflow.score_cell(
        "query",
        cell(answer(kind, text=text)),
        TypedAnswer("text", text="1.5" if kind == "number" else "false"),
    )
    assert wrong.score.value == 0


def test_reference_numeric_and_date_annotations():
    numeric = QueryGold(answer="2", metadata={"typed_answer": {"kind": "number", "number": "2.0"}})
    assert workflow.typed_reference(numeric).number == Decimal("2.0")
    dated = QueryGold(
        answer="2026-09-20",
        metadata={
            "typed_answer": {
                "kind": "date",
                "date_value": "2026-09-20",
            }
        },
    )
    assert workflow.typed_reference(dated).date_value.isoformat() == "2026-09-20"


def definitions():
    return tuple(
        {
            "cell_id": method,
            "method": method,
            "source_variant": "active",
            "question_budget": 8,
            "round_id": 2,
            "treatment": "main",
        }
        for method in ("pmmc", *BASELINE_METHODS)
    )


def bind(*, results=(), answers=(), defs=None):
    return workflow.bind_cell_inputs(
        definitions=definitions() if defs is None else defs,
        result_rows=results,
        answers={a.typed_answer_id: a for a in answers},
        answer_hashes={a.typed_answer_id: "hash" for a in answers},
        query_run_id="run",
        baseline_runs={m: m + "-run" for m in BASELINE_METHODS},
        query_id="query",
    )


def runtime_result(**changes):
    return SimpleNamespace(
        **{
            "treatment": "pmmc",
            "question_budget": 8,
            "revision_depth": 2,
            "query_executor_run_id": "run",
            "query_id": "query",
            "cell_result_id": "runtime-cell",
            "content_hash": "hash",
            "selected_answer_id": None,
            "terminal_state": "no_answer",
            **changes,
        }
    )


def test_binding_preserves_all_selected_cells_and_does_not_drop_missing_baselines():
    bound = bind(results=(runtime_result(),))
    assert len(bound) == 6
    for entry in bound:
        result = workflow.score_cell("query", entry, TypedAnswer("text", text="May"))
        assert result.status == (
            TerminalStatus.ABSTENTION
            if entry.method == "pmmc"
            else TerminalStatus.ENGINEERING_MISSING
        )


@pytest.mark.parametrize("treatment", ["main", "fixed_planner", "no_raw_image", "no_reranker"])
def test_treatment_binding_is_not_assumed_to_be_main(treatment):
    defs = ({**definitions()[0], "treatment": treatment},)
    results = (runtime_result(treatment="pmmc" if treatment == "main" else treatment),)
    assert bind(results=results, defs=defs)[0].terminal_state == "no_answer"


def test_runtime_selected_typed_answer_missing_is_engineering_failure():
    bound = bind(results=(runtime_result(terminal_state="answered", selected_answer_id="lost"),))
    selected = next(c for c in bound if c.method == "pmmc")
    assert selected.terminal_state == "engineering_missing"
    assert selected.error_code == "selected_typed_answer_missing"


def test_baseline_only_primary_typed_variant_can_satisfy_cell():
    method = BASELINE_METHODS[0]
    wrong = answer(method=method, run_id=method + "-run", variant="counterfactual")
    assert next(c for c in bind(answers=(wrong,)) if c.method == method).answer is None
    primary = answer(method=method, run_id=method + "-run", variant="primary")
    assert next(c for c in bind(answers=(primary,)) if c.method == method).answer == primary


def test_native_fallback_reuse_is_explicit_and_no_fallback_cannot_use_it():
    native = answer(
        method="native_multimodal_rag", run_id="native_multimodal_rag-run", variant="primary"
    )
    result = runtime_result(terminal_state="answered", selected_answer_id=native.typed_answer_id)
    assert (
        next(c for c in bind(results=(result,), answers=(native,)) if c.method == "pmmc").answer
        == native
    )
    defs = ({**definitions()[0], "source_variant": "no_fallback"},)
    with pytest.raises(ValueError, match="selected_answer_scope_mismatch"):
        bind(
            results=(
                runtime_result(
                    treatment="no_fallback",
                    terminal_state="answered",
                    selected_answer_id=native.typed_answer_id,
                ),
            ),
            answers=(native,),
            defs=defs,
        )


def test_duplicate_and_foreign_runtime_cells_fail_closed():
    with pytest.raises(ValueError, match="duplicate_runtime_cell"):
        bind(results=(runtime_result(), runtime_result()))
    with pytest.raises(ValueError, match="cell_scope_mismatch"):
        bind(results=(runtime_result(query_id="other-query"),))
    with pytest.raises(ValueError, match="nonanswer_has_selected_answer"):
        bind(results=(runtime_result(selected_answer_id="typed"),))


def test_typed_answer_payload_hash_and_lineage_are_verified():
    value = answer()
    payload = runtime_fact_payload(value)
    row = SimpleNamespace(
        payload={"metadata": {"typed_result": payload}},
        content_hash=canonical_content_hash(payload),
        typed_answer_id=value.typed_answer_id,
        query_executor_run_id=value.run_id,
        query_id=value.query_id,
        terminal_state=value.terminal_state,
        answer_model_call_id=value.answer_model_call_id,
    )
    assert workflow._decode_answer(row).typed_answer_id == value.typed_answer_id
    row.query_id = "other"
    with pytest.raises(ValueError, match="lineage_mismatch"):
        workflow._decode_answer(row)
    row.content_hash = "wrong"
    with pytest.raises(ValueError, match="hash_mismatch"):
        workflow._decode_answer(row)


def source_memory():
    return MemoryUnit(
        memory_id="memory",
        dataset="memgallery",
        conversation_id="conversation",
        session_id="cutoff",
        turn_id="turn",
        timestamp_order=1,
        available_after_session="cutoff",
        text="Synthetic canonical source only",
        image_refs=[
            ImageRef(image_id=canonical, metadata={"source_image_id": source})
            for canonical, source in (("canonical-a", "D1_IMG_001"), ("canonical-b", "D1_IMG_002"))
        ],
    )


def facade_fixture(monkeypatch, cells, gold=QueryGold(answer="May"), *, memories=()):
    from pmmc.schemas.step8_process import (
        PROCESS_METRIC_NAMES,
        ProcessCellMetrics,
        ProcessMetric,
        ProcessMetricsReport,
    )

    query = QueryInput(
        query_id="query",
        dataset="memgallery",
        conversation_id="conversation",
        available_after_session="cutoff",
        raw_query="Which month?",
        metadata={"scenario_id": "scenario", "source_category": "synthetic-category"},
    )
    config = BaselineEvaluationConfig()
    inputs = workflow.Step8EvaluationInputs(
        query, "snapshot", "protocol", config.judge_model_alias, tuple(cells)
    )
    # This facade fixture isolates answer evaluation; the process reader/scorer has
    # its own authoritative-fact tests. Do not teach production code about mocks.
    process_report = ProcessMetricsReport(
        status="complete",
        cells=[
            ProcessCellMetrics(
                **{
                    name: getattr(c, name)
                    for name in (
                        "cell_id",
                        "method",
                        "source_run_id",
                        "source_fact_id",
                        "source_fact_hash",
                        "question_budget",
                        "round_id",
                    )
                },
                query_id=query.query_id,
                metrics={
                    name: ProcessMetric(
                        status="not_evaluated",
                        reason="isolated_answer_facade_fixture",
                        denominator=1,
                    )
                    for name in PROCESS_METRIC_NAMES
                },
            )
            for c in cells
        ],
    )
    monkeypatch.setattr(
        workflow.step8_process_metrics,
        "evaluate_process_metrics",
        AsyncMock(return_value=process_report),
    )
    semantic_prepare = AsyncMock(return_value=SimpleNamespace(identity={"plan_hash": "semantic"}))
    semantic_evaluate = AsyncMock(
        return_value=SimpleNamespace(
            model_dump=lambda **kwargs: {"status": "evaluated", "pairs": [], "cells": []},
        )
    )
    monkeypatch.setattr(
        workflow.step8_semantic_workflow,
        "prepare_query_semantics",
        semantic_prepare,
    )
    monkeypatch.setattr(
        workflow.step8_semantic_workflow,
        "evaluate_query_semantics",
        semantic_evaluate,
    )
    state = {"cached": None, "objects": {}, "artifacts": []}
    session = SimpleNamespace(scalar=AsyncMock(side_effect=lambda *args: state["cached"]))

    class SessionContext:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    engine = SimpleNamespace(dispose=AsyncMock())
    factory = Mock(return_value=(engine, SessionContext))
    monkeypatch.setattr(workflow, "create_session_factory", factory)
    monkeypatch.setattr(workflow, "_load_inputs", AsyncMock(return_value=inputs))
    gold_repo = SimpleNamespace(get=AsyncMock(return_value=gold))
    monkeypatch.setattr(workflow, "PostgresGoldRepository", lambda sessions: gold_repo)

    async def iter_visible(*args):
        for memory in memories:
            yield memory

    memory_repo = SimpleNamespace(iter_visible=Mock(side_effect=iter_visible))
    memory_factory = Mock(return_value=memory_repo)
    monkeypatch.setattr(workflow, "PostgresMemoryRepository", memory_factory)
    monkeypatch.setattr(workflow, "load_evaluation_config", lambda path: config)
    monkeypatch.setattr(workflow, "load_judge_prompt", lambda *args: "fixed prompt")
    monkeypatch.setattr(
        workflow,
        "load_model_registry",
        lambda path: SimpleNamespace(resolved_hash_for_aliases=lambda aliases: "model-lock"),
    )
    register = AsyncMock()
    finish = AsyncMock()
    monkeypatch.setattr(workflow, "_register", register)
    monkeypatch.setattr(workflow, "_finish", finish)
    store = SimpleNamespace(get_bytes=AsyncMock(side_effect=lambda key: state["objects"][key]))
    monkeypatch.setattr(
        workflow, "create_evaluator_object_stores", lambda settings: (object(), store)
    )

    async def put_bytes(**kwargs):
        payload = kwargs["data"]
        key = kwargs["artifact_id"]
        ref = SimpleNamespace(
            artifact_id=key,
            object_key=key,
            storage_uri="private:" + key,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        state["objects"][key] = payload
        state["artifacts"].append(ref)
        if key.startswith("step8-scores-"):
            state["cached"] = ref
        return ref

    artifact_service = SimpleNamespace(put_bytes=AsyncMock(side_effect=put_bytes))
    artifact_factory = Mock(return_value=artifact_service)
    monkeypatch.setattr(workflow, "ExperimentArtifactService", artifact_factory)
    execution_factory = Mock(return_value=object())
    monkeypatch.setattr(workflow, "create_model_execution_service", execution_factory)
    monkeypatch.setattr(workflow, "create_generation_gateway", Mock(return_value=object()))
    monkeypatch.setattr(workflow, "create_image_url_resolver", Mock(return_value=object()))
    judge = AsyncMock(return_value=SimpleNamespace(llm_judge_score=1, model_call_id="judge-call"))
    monkeypatch.setattr(workflow, "_judge_one", judge)
    settings = SimpleNamespace(
        resolved_evaluation_config_path="eval",
        project_root=ROOT,
        resolved_model_registry_path="models",
        experiment_artifact_prefix="eval",
    )
    settings.model_copy = lambda **kwargs: settings

    def run(**kwargs):
        return asyncio.run(
            workflow.evaluate_step8_query(settings, "campaign", "run", "suite", "query", **kwargs)
        )

    return SimpleNamespace(
        run=run,
        engine=engine,
        factory=factory,
        gold=gold_repo,
        register=register,
        finish=finish,
        judge=judge,
        state=state,
        artifacts=artifact_service,
        artifact_factory=artifact_factory,
        execution_factory=execution_factory,
        inputs=inputs,
        memories=memory_repo,
        memory_factory=memory_factory,
        semantic_prepare=semantic_prepare,
        semantic_evaluate=semantic_evaluate,
    )


def test_facade_noanswer_zero_and_image_score_without_text_judge(monkeypatch):
    image = answer("image", images=("canonical-a",))
    gold = QueryGold(
        answer="[image]",
        metadata={
            "typed_answer": {
                "kind": "image",
                "image_ids": ["canonical-a"],
            }
        },
    )
    fake = facade_fixture(
        monkeypatch,
        [cell(image), cell(terminal="no_answer", identifier="empty")],
        gold,
        memories=(source_memory(),),
    )
    result = fake.run()
    assert [r["value"] for r in result["scores"]] == [1, 0]
    assert result["selected_cell_count"] == 2
    assert result["status"] == "complete" and result["scientific_approval"] is False
    assert fake.factory.call_args.args[1] == DatabaseRole.EVALUATOR
    assert fake.artifact_factory.call_args.args[2] == DataScope.EVALUATION
    fake.judge.assert_not_awaited()
    fake.semantic_evaluate.assert_awaited_once()
    fake.finish.assert_awaited_once()
    fake.engine.dispose.assert_awaited_once()
    assert "canonical-a" not in json.dumps(result)
    assert "[image]" not in json.dumps(result)


def test_facade_missing_cell_persists_incomplete_and_never_marks_complete(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(terminal="engineering_missing")])
    with pytest.raises(workflow.Step8EvaluationIncomplete) as caught:
        fake.run()
    result = caught.value.result
    assert result["selected_cell_count"] == 1 and result["scores"][0]["value"] is None
    assert result["artifact"]["artifact_id"].startswith("step8-incomplete-")
    assert result["status"] == "incomplete"
    fake.finish.assert_not_awaited()
    fake.engine.dispose.assert_awaited_once()


def test_facade_evidence_only_gold_stays_na_not_false_zero(monkeypatch):
    fake = facade_fixture(
        monkeypatch, [cell(answer())], QueryGold(evidence=GoldEvidence(image_ids=["i"]))
    )
    with pytest.raises(workflow.Step8EvaluationIncomplete) as caught:
        fake.run()
    result = caught.value.result
    assert result["status"] == "incomplete"
    assert result["selected_cell_count"] == 1 and result["applicable_cell_count"] == 0
    assert result["scores"][0]["value"] is None
    assert result["scores"][0]["unavailable_reason"] == "typed_reference_unresolved"
    assert result["scores"][0]["engineering_error"] is None
    fake.finish.assert_not_awaited()
    fake.judge.assert_not_awaited()


def test_facade_shared_fallback_judge_call_is_reused_and_complete_artifact_resumes(monkeypatch):
    native = answer(method="native_multimodal_rag", variant="primary")
    fake = facade_fixture(
        monkeypatch, [cell(native), cell(native, identifier="native", method=native.method)]
    )
    result = fake.run()
    assert result["selected_cell_count"] == 2 and fake.judge.await_count == 1
    assert {r["judge_model_call_id"] for r in result["scores"]} == {"judge-call"}
    assert fake.run() == result
    assert fake.judge.await_count == 1 and fake.artifacts.put_bytes.await_count == 1
    assert fake.engine.dispose.await_count == 2


def test_facade_judge_failure_diagnostic_does_not_poison_success_cache(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer())])
    fake.judge.side_effect = TimeoutError("private gold must not leak")
    with pytest.raises(workflow.Step8EvaluationIncomplete) as caught:
        fake.run()
    assert "private gold" not in json.dumps(caught.value.result)
    assert caught.value.result["scores"][0]["engineering_error"] == "judge_engineering_missing"
    fake.judge.side_effect = None
    result = fake.run()
    assert result["status"] == "complete" and fake.judge.await_count == 2
    assert [a.artifact_id.split("-")[1] for a in fake.state["artifacts"]] == [
        "incomplete",
        "scores",
    ]
    fake.finish.assert_awaited_once()


@pytest.mark.parametrize("phase", ["prepare", "evaluate"])
def test_semantic_engineering_failure_blocks_completion_without_leaking_payload(monkeypatch, phase):
    fake = facade_fixture(monkeypatch, [cell(answer())])
    operation = getattr(fake, "semantic_" + phase)
    operation.side_effect = ValueError("private query and gold must not leak")
    with pytest.raises(workflow.Step8EvaluationIncomplete) as caught:
        fake.run()
    result = caught.value.result
    assert result["qbank_semantic"]["status"] == "incomplete"
    assert "private query" not in json.dumps(result)
    fake.finish.assert_not_awaited()
    operation.side_effect = None
    assert fake.run()["status"] == "complete"
    fake.finish.assert_awaited_once()


def test_semantic_plan_identity_and_completed_report_resume(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer())])
    first = fake.run()
    assert fake.register.call_args.kwargs["metadata"]["qbank_semantic_identity"] == {
        "plan_hash": "semantic",
    }
    assert fake.run() == first
    assert fake.semantic_prepare.await_count == 2
    fake.semantic_evaluate.assert_awaited_once()
    fake.execution_factory.assert_called_once()


def test_baseline_only_evaluation_does_not_invent_qbank_semantic_denominator(monkeypatch):
    native = answer(method="native_multimodal_rag", variant="primary")
    fake = facade_fixture(monkeypatch, [cell(native, method=native.method)])
    result = fake.run()
    assert result["qbank_semantic"] == {
        "status": "not_applicable",
        "reason": "no_pmmc_cells",
    }
    fake.semantic_prepare.assert_not_awaited()
    fake.semantic_evaluate.assert_not_awaited()


def test_facade_checks_cached_object_checksum_and_identity(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(terminal="no_answer")])
    fake.run()
    ref = fake.state["cached"]
    fake.state["objects"][ref.object_key] = b"{}"
    with pytest.raises(ValueError, match="checksum_mismatch"):
        fake.run()
    ref.sha256 = hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(ValueError, match="identity_mismatch"):
        fake.run()


def test_facade_rejects_unfrozen_judge_alias_before_registering(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(terminal="no_answer")])
    monkeypatch.setattr(
        workflow, "_load_inputs", AsyncMock(return_value=replace(fake.inputs, judge_alias="other"))
    )
    with pytest.raises(ValueError, match="frozen_judge_alias_mismatch"):
        fake.run()
    fake.register.assert_not_awaited()
    fake.engine.dispose.assert_awaited_once()


@pytest.mark.parametrize(
    "raw,kind,images,text,expected",
    [
        ("D1_IMG_001", "image", ("canonical-a",), None, 1),
        ("D1_IMG_001", "image", ("canonical-b",), None, 0),
        ("Image set: D1_IMG_001, D1_IMG_002", "image_set", ("canonical-b", "canonical-a"), None, 1),
        (
            "Order: D1_IMG_001, D1_IMG_002",
            "ordered_image_list",
            ("canonical-a", "canonical-b"),
            None,
            1,
        ),
        (
            "Order: D1_IMG_001, D1_IMG_002",
            "ordered_image_list",
            ("canonical-b", "canonical-a"),
            None,
            0,
        ),
        ("3 images", "number", (), "3", 1),
        ("3 images", "number", (), "2", 0),
        ("false", "boolean", (), "false", 1),
    ],
)
def test_production_resolver_lossless_source_answer(monkeypatch, raw, kind, images, text, expected):
    gold = QueryGold(answer=raw, evidence=GoldEvidence(image_ids=["unrelated-gold-evidence"]))
    fake = facade_fixture(
        monkeypatch,
        [cell(answer(kind, images=images, text=text))],
        gold,
        memories=(source_memory(),),
    )
    monkeypatch.setattr(workflow, "typed_reference", Mock(side_effect=AssertionError("legacy")))
    result = fake.run()
    row = result["scores"][0]
    assert row["value"] == expected and row["applicable"]
    assert row["unavailable_reason"] is None
    assert result["reference_annotation"]["kind"] == kind
    assert result["reference_annotation"]["status"] == "resolved"
    fake.memories.iter_visible.assert_called_once_with("snapshot", "conversation", "cutoff")
    assert fake.memory_factory.call_args.args[0] is fake.factory.return_value[1]
    assert fake.factory.call_args.args[1] == DatabaseRole.EVALUATOR
    assert result["selected_query"]["strata"] == [
        ["source_category", "synthetic-category"],
        ["scenario_id", "scenario"],
    ]
    fake.judge.assert_not_awaited()
    metadata = fake.register.call_args.kwargs["metadata"]
    annotation = result["reference_annotation"]
    preimage = metadata["reference_annotation_payload"]
    assert json.loads(preimage) == {k: v for k, v in annotation.items() if k != "hash"}
    assert hashlib.sha256(preimage.encode()).hexdigest() == annotation["hash"]
    assert annotation == metadata["reference_annotation"]
    assert annotation == fake.finish.call_args.args[3]["reference_annotation"]
    assert annotation["query_hash"] == metadata["query_hash"]
    assert annotation["gold_hash"] == metadata["gold_hash"]
    assert "unrelated-gold-evidence" not in json.dumps(result)
    assert gold.metadata == {}  # Evaluator attestation never mutates canonical gold.


def test_production_mixed_text_and_unavailable_image_cells_keep_text_judge(monkeypatch):
    fake = facade_fixture(
        monkeypatch,
        [
            cell(answer()),
            cell(
                answer("image", images=("canonical-a",), identifier="image"),
                identifier="image-cell",
            ),
        ],
        memories=(source_memory(),),
    )
    with pytest.raises(workflow.Step8EvaluationIncomplete) as caught:
        fake.run()
    result = caught.value.result
    text, image = result["scores"]
    assert (text["value"], text["judge_value"], text["applicable"]) == (1, 1, True)
    assert image["value"] is None and image["applicable"] is False
    assert image["unavailable_reason"] == "typed_reference_unresolved"
    assert image["status"] == "answered" and image["engineering_error"] is None
    assert image["judge_value"] is None and image["typed_judge_disagreement"] is None
    assert result["selected_cell_count"] == 2 and result["applicable_cell_count"] == 1
    fake.judge.assert_awaited_once()
    fake.finish.assert_not_awaited()


@pytest.mark.parametrize("raw", ["D1_IMG_001, D1_IMG_002", "Image: unknown"])
def test_unresolved_gold_does_not_claim_judge_applicability(monkeypatch, raw):
    fake = facade_fixture(
        monkeypatch,
        [cell(answer(text=raw)), cell(terminal="no_answer", identifier="empty")],
        QueryGold(answer=raw),
        memories=(source_memory(),),
    )
    with pytest.raises(workflow.Step8EvaluationIncomplete) as caught:
        fake.run()
    result = caught.value.result
    assert all(row["value"] is None and not row["applicable"] for row in result["scores"])
    assert all(
        row["unavailable_reason"] == "typed_reference_unresolved" for row in result["scores"]
    )
    assert all(row["judge_applicable"] is False for row in result["scores"])
    assert result["scores"][0]["judge_value"] is None
    assert result["scores"][0]["typed_judge_disagreement"] is None
    fake.judge.assert_not_awaited()
    fake.finish.assert_not_awaited()


@pytest.mark.parametrize("raw", ["D1_IMG_001", "3 images", "true"])
def test_derived_reference_suppresses_only_image_gold_text_judge(monkeypatch, raw):
    fake = facade_fixture(
        monkeypatch, [cell(answer(text=raw))], QueryGold(answer=raw), memories=(source_memory(),)
    )
    result = fake.run()
    assert result["scores"][0]["value"] == 0  # Known typed mismatch, not unavailable.
    assert result["scores"][0]["judge_applicable"] is (raw != "D1_IMG_001")
    assert result["scores"][0]["judge_value"] == (None if raw == "D1_IMG_001" else 1)
    if raw == "D1_IMG_001":
        fake.judge.assert_not_awaited()
    else:
        assert fake.judge.call_args.kwargs["reference"].text == raw


@pytest.mark.parametrize("explicit", [False, True])
def test_image_answer_and_gold_evidence_cannot_supply_missing_source_refs(monkeypatch, explicit):
    gold = QueryGold(
        answer="D1_IMG_001",
        evidence=GoldEvidence(image_ids=["canonical-a"]),
        metadata={"typed_answer": {"kind": "image", "image_ids": ["canonical-a"]}}
        if explicit
        else {},
    )
    fake = facade_fixture(monkeypatch, [cell(answer("image", images=("canonical-a",)))], gold)
    with pytest.raises(workflow.Step8EvaluationIncomplete) as caught:
        fake.run()
    assert caught.value.result["scores"][0]["value"] is None
    assert caught.value.result["scores"][0]["unavailable_reason"] == "typed_reference_unresolved"
    fake.finish.assert_not_awaited()


def test_missing_gold_is_explicitly_not_applicable_not_zero(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer()), cell(terminal="no_answer", identifier="n")])
    fake.gold.get.side_effect = KeyError("missing")
    result = fake.run()
    assert result["reference_annotation"]["status"] == "missing"
    assert result["applicable_cell_count"] == 0
    assert all(
        row["value"] is None and row["unavailable_reason"] == "gold_missing"
        for row in result["scores"]
    )
    fake.judge.assert_not_awaited()


def test_source_loading_failure_is_fail_closed_before_registration(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer())])
    fake.memories.iter_visible.side_effect = KeyError("cutoff missing")
    with pytest.raises(KeyError, match="cutoff missing"):
        fake.run()
    fake.register.assert_not_awaited()
    fake.finish.assert_not_awaited()
    fake.engine.dispose.assert_awaited_once()


def test_missing_source_cutoff_is_not_replaced_by_latest_or_evidence(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer())])
    inputs = replace(
        fake.inputs, query=fake.inputs.query.model_copy(update={"available_after_session": None})
    )
    monkeypatch.setattr(workflow, "_load_inputs", AsyncMock(return_value=inputs))
    with pytest.raises(ValueError, match="source_cutoff_missing"):
        fake.run()
    fake.memories.iter_visible.assert_not_called()
    fake.register.assert_not_awaited()
    fake.finish.assert_not_awaited()


def test_source_inventory_and_resolver_implementation_are_evaluation_identity(monkeypatch):
    memories = [source_memory()]
    fake = facade_fixture(monkeypatch, [cell(answer())], memories=memories)
    first = fake.run()
    memories.clear()
    fake.state["cached"] = None
    second = fake.run()
    assert first["evaluation_run_id"] != second["evaluation_run_id"]
    assert (
        first["reference_annotation"]["source_refs_hash"]
        != second["reference_annotation"]["source_refs_hash"]
    )
    assert (
        second["reference_annotation"]["resolver_source_hash"]
        == hashlib.sha256(Path(workflow.step8_typed_gold.__file__).read_bytes()).hexdigest()
    )


def test_cached_report_requires_exact_reference_annotation(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer())])
    report = fake.run()
    report.pop("artifact")
    report["reference_annotation"]["kind"] = "image"
    payload = json.dumps(report).encode()
    cached = fake.state["cached"]
    fake.state["objects"][cached.object_key] = payload
    cached.sha256 = hashlib.sha256(payload).hexdigest()
    fake.finish.reset_mock()
    with pytest.raises(ValueError, match="identity_mismatch"):
        fake.run()
    fake.finish.assert_not_awaited()


def test_gold_free_runtime_imports_do_not_load_evaluator_modules():
    script = """
import sys
sys.path.insert(0, 'src')
import pmmc.runtime.query_workflow
import pmmc.baselines.rag_workflow
import pmmc.experiments.step8_backend
assert 'pmmc.evaluation.step8_typed_gold' not in sys.modules
assert 'pmmc.evaluation.step8_workflow' not in sys.modules
assert 'pmmc.evaluation.repositories' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_finish_writes_exact_completion_annotation_and_preserves_state_metadata():
    state = SimpleNamespace(status="running", metadata_json={"campaign_job_owned": True})
    session = SimpleNamespace(get=AsyncMock(return_value=state), execute=AsyncMock())

    class Transaction:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    sessions = SimpleNamespace(begin=Transaction)
    report = {"reference_annotation": {"hash": "a" * 64}, "scores": []}
    artifact = {"artifact_id": "scores", "sha256": "b" * 64, "object_key": "not-in-state"}
    asyncio.run(workflow._finish(sessions, "evaluation", 2, report, artifact))
    params = session.execute.call_args.args[0].compile().params
    assert params["metadata"] == {
        "campaign_job_owned": True,
        "completion_report": report,
        "completion_artifact": {"artifact_id": "scores", "sha256": "b" * 64},
    }
    assert params["status"] == "completed" and params["succeeded_count"] == 2
    session.get.assert_awaited_once_with(
        workflow.EvaluationRunStateModel, "evaluation", with_for_update=True
    )
