import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.release_host.runner import launch

from pmmc.experiments import release_readouts as analysis
from pmmc.release_plan import normalize_plan, plan_hash
from test_release_host import Simulated, config


def test_source_only_analysis_does_not_start_models_or_services(tmp_path):
    cfg, sim = config(tmp_path), Simulated()
    plan = normalize_plan(
        dict(
            run_id="cost",
            stage="analysis",
            analyses=["cost"],
            from_run="parent",
            models=["answer-gpt-6-luna"],
        )
    )

    def outcome(job, role):
        assert role == "analysis-report"
        return dict(
            state="missing_dependencies", plan_hash=plan_hash(plan), missing_units=["wr-main"]
        )

    sim.outcome = outcome
    sim.reconcile = lambda: pytest.fail("reader must not reconcile services")
    assert (
        launch(tmp_path / "state", cfg, "cost", plan, factory=sim.factory, sleep=lambda _: None)
        == 4
    )
    assert [x[0] for x in sim.started] == ["analysis-report"]
    assert (tmp_path / "state/runs/cost/diagnostics.tar.gz").is_file()


def test_standalone_baseline_does_not_wait_for_main(tmp_path):
    cfg, sim = config(tmp_path), Simulated()
    original = sim.outcome

    def outcome(job, role):
        if role in ("control", "gpu"):
            return dict(
                stage="baselines",
                scope_hash="1" * 64,
                state="stage_complete",
                paid_authorized=False,
                scoped_jobs=8,
                completed=8,
            )
        return original(job, role)

    sim.outcome = outcome
    plan = normalize_plan(
        dict(
            run_id="rag",
            stage="baselines",
            benchmarks=["memlens"],
            models=["answer-qwen3.5-9b"],
            methods=["caption_text_rag"],
        )
    )
    assert (
        launch(tmp_path / "state", cfg, "rag", plan, factory=sim.factory, sleep=lambda _: None) == 0
    )
    assert {x[1] for x in sim.started} == {"baselines"}
    before = len(sim.started)
    assert launch(tmp_path / "state", cfg, "rag", plan, factory=sim.factory, resume=True) == 0
    assert before == len(sim.started)


def test_intent_hash_is_canonical_and_strict():
    base = dict(
        run_id="my-run", stage="main", models=["answer-qwen3.5-9b", "answer-claude-haiku-5.5"]
    )
    assert plan_hash(base) == plan_hash({**base, "models": base["models"][::-1]})
    assert plan_hash(base) != plan_hash({**base, "models": base["models"][:1]})
    for update in (
        {"benchmarks": []},
        {"stage": "paper"},
        {"methods": ["caption_text_rag"]},
        {"run_id": "../other"},
        {"models": ["other"]},
        {"gpu": 8},
    ):
        with pytest.raises(ValueError):
            normalize_plan({**base, **update})


def test_budget_fence_uses_live_worker_without_host_run_lock(tmp_path, monkeypatch):
    from scripts.release_host import budget_ops as ops
    from scripts.release_host.runner import atomic

    cfg = config(tmp_path)
    root = tmp_path / "host"
    plan = normalize_plan({"run_id": "demo"})
    folder = root / "runs/demo"
    private = root / "instances" / cfg["instance"]
    folder.mkdir(parents=True)
    private.mkdir(parents=True)
    atomic(private / "identity.json", dict(deployment="instance", secrets={}))
    atomic(
        folder / "progress.json",
        dict(status="running", batch={"control": {"state": "running", "job": "job-abc"}}),
    )

    class Backend:
        kube = ["kubectl"]

        def __init__(self, *args):
            pass

        def state(self, job):
            return "running"

    monkeypatch.setattr(ops, "Kubernetes", Backend)

    def execute(argv):
        assert argv[1] == "exec" and "stop-exploration-budget" in argv
        return json.dumps(
            dict(
                plan_hash=plan_hash(plan),
                approval_id="a" * 64,
                new_calls_stopped=True,
                settled_usd="2",
                reserved_usd="0.1",
            )
        )

    monkeypatch.setattr(ops, "invoke", execute)
    assert (
        ops.audit(
            root,
            cfg,
            plan,
            dict(
                kind="stop-budget",
                variant="wr",
                approval_id="a" * 64,
                selection_hash="b" * 64,
                operator_id="test",
                reason_code="operator_stop",
            ),
        )
        == 0
    )
    assert json.loads((folder / "progress.json").read_text())["status"] == "running"


def test_seal_is_stable_when_only_cumulative_ledger_changes(tmp_path):
    receipt = dict(
        campaign_id="c", selection_hash="s", scope_hash="h", stage="main", state="stage_complete"
    )
    data = dict(
        campaign_id="c",
        selection_hash="s",
        scope="main",
        paper_result=False,
        groups=[],
        selection_ledger={"settled_usd": "1.00"},
    )
    analysis.seal_report(tmp_path, "main", data, receipt)
    seal = (tmp_path / "reports/seal-main.json").read_bytes()
    analysis.seal_report(
        tmp_path, "main", {**data, "selection_ledger": {"settled_usd": "2.00"}}, receipt
    )
    assert (tmp_path / "reports/seal-main.json").read_bytes() == seal
    with pytest.raises(ValueError, match="immutable"):
        analysis.seal_report(tmp_path, "main", {**data, "groups": [{}]}, receipt)


def test_cost_readout_preserves_missing_costs_and_source_scope(tmp_path, monkeypatch):
    data = {
        "groups": [
            {
                "dataset": "memlens",
                "answer_model": "answer-qwen3.5-9b",
                "selected_queries": 1,
                "cells": [{"evaluation_job_states": {"completed": 1}}],
            }
        ]
    }
    monkeypatch.setattr(
        analysis,
        "_read",
        lambda settings, run, arm, stage: (
            (data, {"run_id": run}, None) if run == "parent" else None
        ),
    )
    req = normalize_plan(
        dict(
            run_id="child",
            stage="analysis",
            from_run="parent",
            analyses=["cost"],
            benchmarks=["memlens"],
            models=["answer-qwen3.5-9b"],
        )
    )
    summary = analysis.run_analysis(SimpleNamespace(outputs_path=tmp_path), req)
    path = (
        Path(tmp_path) / "step8/exploration/child/reports/analyses" / (summary["sha256"] + ".json")
    )
    assert json.loads(path.read_text())["sections"]["cost"][0]["selection_ledger"] is None
    with pytest.raises(ValueError, match="does not cover"):
        analysis.run_analysis(
            SimpleNamespace(outputs_path=tmp_path), {**req, "benchmarks": ["memgallery"]}
        )
