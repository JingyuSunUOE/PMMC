"""Offline release-frontier checks, including Step 8's three-row terminal contract."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pmmc.evaluation import step8_workflow as workflow
from pmmc.evaluation.step8_relevance_grid import evaluate_relevance_grid_answers
from pmmc.evaluation.step8_typed_gold import resolve_typed_gold
from pmmc.experiments import step8_campaign
from pmmc.experiments.release_scope import TARGET_OPERATIONS, stage_job_ids
from pmmc.runtime import query_workflow
from pmmc.runtime.query_executor import OnlineQueryExecutor
from pmmc.schemas import DataScope, QueryGold
from pmmc.schemas.query_executor import RerankerMatchLevel
from pmmc.schemas.step8 import BASELINE_METHODS
from test_release_reporting import ROOT, Campaign
from test_step8_campaign import protocol
from test_step8_relevance_grid import GridBackend
from test_step8_runtime import CONFIG
from test_step8_workflow import answer, cell, facade_fixture


def test_real_9b_manifest_can_prepare_before_native_answer_and_matches_0032_grid(monkeypatch):
    model = "answer-qwen3.5-9b"
    dataset = query_workflow.DatasetName.MEMLENS
    compiler = SimpleNamespace(
        compiler_schema_version="0.9.0",
        operator_set_version="pmmc-operators-v7",
        target_stage=query_workflow.CompilationTargetStage.VERIFIED_QBANK,
        active_round=2,
        max_round=3,
        snapshot_id="s",
    )
    projection = SimpleNamespace(
        run_id="compiler",
        snapshot_id="s",
        schema_version="0.4.0",
        active_round=2,
        max_round=3,
        member_count=0,
        answerable_count=0,
        routing_projection_id="projection",
        content_hash="a" * 64,
    )
    native = SimpleNamespace(
        method="native_multimodal_rag",
        dataset=dataset,
        dataset_snapshot_ids=["s"],
        answer_model_alias=model,
        metadata={"answer_contract_version": CONFIG.answer_contract_version},
    )
    parents = SimpleNamespace(
        compiler_runs=SimpleNamespace(get=AsyncMock(return_value=compiler)),
        doubter=SimpleNamespace(get_routing_projection=AsyncMock(return_value=(projection, []))),
        runs=SimpleNamespace(
            get_manifest=AsyncMock(return_value=native),
            get_state=AsyncMock(
                side_effect=lambda key: SimpleNamespace(
                    status=query_workflow.RunStatus.COMPLETED
                    if key == "compiler"
                    else query_workflow.RunStatus.RUNNING
                )
            ),
        ),
        answers=object(),
        query_executor=SimpleNamespace(get_typed_answer_for_variant=AsyncMock(return_value=None)),
    )

    class Unit:
        async def __aenter__(self):
            return parents

        async def __aexit__(self, *_):
            pass

    monkeypatch.setattr(query_workflow, "ExperimentUnitOfWork", lambda *_: Unit())
    monkeypatch.setattr(
        query_workflow,
        "create_session_factory",
        lambda *_: (SimpleNamespace(dispose=AsyncMock()), object()),
    )
    monkeypatch.setattr(
        query_workflow,
        "PostgresRuntimeQueryRepository",
        lambda *_: SimpleNamespace(
            list_ids=AsyncMock(
                return_value=[SimpleNamespace(query_id="q", dataset=dataset, query_image_refs=[])]
            )
        ),
    )
    args = dict(
        settings=SimpleNamespace(project_root=ROOT),
        config=CONFIG,
        source_compiler_run_id="compiler",
        coverage_run_id=None,
        fallback_run_id="native",
        query_ids=["q"],
        pilot_projection=True,
    )
    primary, analysis = (
        asyncio.run(
            query_workflow.create_query_executor_manifest(
                **args, primary_only=flag, allow_pending_fallback=True
            )
        )
        for flag in (True, False)
    )
    assert primary.query_executor_run_id != analysis.query_executor_run_id
    assert primary.metadata["readout"] == "exploration-primary-b8-r2"
    assert primary.metadata["execute_full_grid"] is False and primary.metadata[
        "declared_cells"
    ] == [[8, 2]]
    assert (
        primary.metadata["expected_cell_count"] == 3
        and analysis.metadata["expected_cell_count"] == 48
    )
    with pytest.raises(ValueError, match="fallback_parent"):
        asyncio.run(query_workflow.create_query_executor_manifest(**args, primary_only=True))


@pytest.mark.parametrize("route_accepted", [False, True])
def test_primary_three_terminal_facts_but_only_one_routed_answer_context(route_accepted):
    async def scenario():
        backend = GridBackend()
        backend.primary_readout = True
        backend.execute_full_grid = False
        backend.level = RerankerMatchLevel.EXACT if route_accepted else RerankerMatchLevel.NONE
        executor = OnlineQueryExecutor(backend=backend, config=CONFIG)
        run = {"run_id": "query-run-1", "query_id": "query-1"}
        matched = await executor.execute(**run, phase="match")
        assert not backend.events
        assert backend.prepared_subsets == ([("high",)] if route_accepted else [])
        await executor.execute(**run, phase="answer")
        assert len(backend.events) == int(route_accepted)
        result = await executor.execute(**run, phase="select")
        assert len(backend.events) == 2 * int(route_accepted)
        facts = result["cell_results"]
        assert {(f.cell.question_budget, f.cell.revision_depth) for f in facts} == {(8, 2)}
        assert {f.cell.treatment for f in facts} == {
            "pmmc",
            "no_fallback",
            "forced_top1_counterfactual",
        }
        (active,) = [f for f in facts if f.cell.treatment == "pmmc"]
        assert active.active_fallback is not route_accepted
        assert active.terminal_state in {"answered", "abstained"}
        if not route_accepted:
            assert all(
                f.selected_answer_id is None and f.terminal_state == "no_answer"
                for f in facts
                if f is not active
            )
        query = matched["query"]
        judge = AsyncMock(side_effect=AssertionError("primary grid must not dispatch a Judge"))
        evidence = await evaluate_relevance_grid_answers(
            **{
                "scope": DataScope.EVALUATION,
                "query": query,
                "snapshot_id": "snapshot-1",
                "query_run_id": run["run_id"],
                "cells": (active,),
                "answers": result["typed_answers"],
                "resolution": resolve_typed_gold(query, QueryGold(answer="Blue")),
                "thresholds": CONFIG.relevance_cosine_grid,
                "judge": judge,
                "primary_thresholds": (0.3, 0.3),
            }
        )
        judge.assert_not_awaited()
        assert evidence["status"] == "completed" and evidence["point_count"] == 1
        assert evidence["answer_count"] == int(route_accepted) and not evidence["answers"]
        assert evidence["points"][0]["status"] == (
            "completed" if route_accepted else "primary_not_executed"
        )

    asyncio.run(scenario())


def test_primary_does_not_dispatch_semantic_or_oracle_work(monkeypatch):
    fake = facade_fixture(monkeypatch, [cell(answer())], QueryGold(answer="May"))
    semantic = AsyncMock(side_effect=AssertionError("deferred semantic work"))
    oracle = AsyncMock(side_effect=AssertionError("deferred oracle work"))
    monkeypatch.setattr(workflow.step8_semantic_workflow, "prepare_query_semantics", semantic)
    monkeypatch.setattr(workflow, "prepare_configured_oracle", oracle)
    report = fake.run(method_scope="pmmc_main")
    assert report["selected_cell_count"] == 1 and report["status"] == "complete"
    assert report["qbank_semantic"] == {"status": "not_evaluated", "reason": "analysis_scope_only"}
    assert report["all_methods_evaluated"] is False
    semantic.assert_not_awaited()
    oracle.assert_not_awaited()


def test_resumed_primary_checkpoint_cannot_dispatch_extra_prepared_keys():
    async def scenario():
        backend = GridBackend()
        backend.primary_readout = True
        backend.execute_full_grid = False
        executor = OnlineQueryExecutor(backend=backend, config=CONFIG)
        run = {"run_id": "query-run-1", "query_id": "query-1"}
        await executor.execute(**run, phase="match")
        prepared = backend.phases["match"]["prepared"]
        prepared["off-grid"] = next(iter(prepared.values()))
        with pytest.raises(ValueError, match="unauthorized"):
            await executor.execute(**run, phase="answer")
        assert backend.events == []

    asyncio.run(scenario())


def test_paired_scope_does_not_mix_analysis_cells_or_accept_partial_baselines():
    rows = [
        SimpleNamespace(
            cell_id=f"{method}-{budget}-{variant}",
            method=method,
            payload={
                "method": method,
                "source_variant": variant,
                "treatment": "main",
                "round_id": 2,
                "question_budget": budget,
            },
        )
        for method, budget, variant in (
            *[(m, 8, "active") for m in ("pmmc", *BASELINE_METHODS)],
            ("pmmc", 1, "active"),
            ("pmmc", 8, "forced_top1"),
            ("pmmc", 8, "no_fallback"),
        )
    ]
    paired = workflow.evaluation_definitions(rows, "main", "paired_main")
    analysis = workflow.evaluation_definitions(rows, "main", "pmmc")
    assert len(paired) == 6 and {row["method"] for row in paired} == {"pmmc", *BASELINE_METHODS}
    assert len(analysis) == 4 and {row["method"] for row in analysis} == {"pmmc"}
    with pytest.raises(ValueError, match="baseline_link"):
        workflow.validate_baseline_links({"native_multimodal_rag": "run"}, "main", "paired_main")


def test_real_planner_frontiers_for_open_primary_analysis_and_baselines(tmp_path, monkeypatch):
    campaign = Campaign(tmp_path)
    monkeypatch.setattr(step8_campaign, "scientific_source_fingerprint", lambda *_: "d" * 64)
    jobs = step8_campaign.build_campaign_jobs(
        manifest=campaign.manifest.model_copy(update={"protocol": protocol()}),
        selection=campaign.selection,
        project_root=ROOT,
        conversations={
            dataset: {"conversation": ids} for dataset, ids in campaign.selection.query_ids.items()
        },
    )
    for stage in ("main", "analysis", "baselines"):
        chosen = stage_job_ids(jobs, stage, paid=False)
        scoped = [job for job in jobs if job.job_id in chosen]
        ops = {job.operation for job in scoped}
        assert ops.intersection(TARGET_OPERATIONS.values()) == {TARGET_OPERATIONS[stage]}
        assert all(
            job.resource_class != "api" and set(job.dependency_ids) <= chosen for job in scoped
        )
        if stage == "baselines":
            assert "baseline_answer" in ops and all(
                j.parameters["treatment"] == "main" for j in scoped if j.operation == "evaluate"
            )
        else:
            assert not ops.intersection({"baseline_answer", "baseline_grounding"})
            assert "native_fallback_ground" in ops
