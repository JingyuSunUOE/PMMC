"""Exercise host-generated commands and native admission without Kubernetes or provider clients."""

import asyncio
import json
import os
import sys
from contextvars import copy_context
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from scripts.release_host.backends import receipt, release_env
from scripts.release_host.runner import units
from scripts.release_host.workloads import command, secret_refs
from typer.testing import CliRunner

from pmmc.experiments import release_execution, release_prepare_variants, step8_cli
from pmmc.experiments.release_reporting import _plan_paid_stages, verify_release_budget_pointer
from pmmc.experiments.release_scope import read_scope
from pmmc.experiments.step8_artifacts import write_json
from pmmc.infrastructure.database.step8_models import (
    BenchmarkBudgetApprovalModel,
    BenchmarkSelectionModel,
)
from pmmc.infrastructure.database.step8_repositories import (
    BudgetExceededError,
    PostgresBenchmarkRepository,
)
from pmmc.models.campaign_audit import CampaignModelAudit, campaign_model_scope
from pmmc.models.step8_accounting import SelectionApproval
from pmmc.release_plan import normalize_plan, plan_hash
from pmmc.schemas.step8 import ResourceClass
from test_release_plan_budget import MemoryLedger, budget_children
from test_release_reporting import GPT, ROOT
from test_step8_accounting import request as paid_request
from test_step8_accounting import setup as fake_accounting


def test_ablation_dependency_scopes_match_readouts():
    plan = normalize_plan(
        dict(run_id="trial", stage="analysis", analyses=["ablation"], with_deps=True)
    )
    assert _plan_paid_stages(plan, "wr") == {"main", "analysis"}
    assert all(_plan_paid_stages(plan, arm) == {"main"} for arm in ("w", "r", "legacy"))
    assert not _plan_paid_stages({**plan, "with_deps": False}, "wr")


@pytest.mark.parametrize("limit", [Decimal("2"), None])
def test_host_quote_approve_dispatch_and_stop(tmp_path, monkeypatch, limit):
    plan = normalize_plan(dict(run_id="trial", stage="all", models=[GPT]))
    dirs, rows, card = budget_children(tmp_path, plan)
    settings = SimpleNamespace(
        project_root=ROOT,
        outputs_path=tmp_path / "out",
        resolve_path=lambda p: ROOT / p if str(p).startswith("configs/") else tmp_path / "store",
    )
    monkeypatch.setattr(step8_cli, "get_settings", lambda: settings)
    monkeypatch.setattr("pmmc.infrastructure.settings.get_settings", lambda: settings)
    monkeypatch.setattr(release_prepare_variants, "get_settings", lambda: settings)
    cfg = dict(
        images={"app": "test@sha256:" + "d" * 64},
        native=dict(benchmark_config="x", model_registry="x", model_lock="x", runtime_profile="x"),
        topology={"gpus": 1},
        resources={"gpu": {"cpu": 3}, "control": {"cpu": 2}},
        models=dict(calibrate=False, allow_download=False, use_hf_token=False),
    )
    prepared = []

    def prepare(settings, run_id, variant):
        selected = step8_cli._latest_selection(dirs[variant]).content_hash
        info = dict(
            run_id=run_id,
            variant=variant,
            selected={},
            release_plan_hash=plan_hash(plan),
            selection_hash=selected,
            state="planned",
            phase="exploration",
            paper_result=False,
            selection="dev-all",
        )
        write_json(
            settings.outputs_path / "step8/exploration" / run_id / variant / "pointer.json", info
        )
        prepared.append(variant)
        return info

    def directory(settings, run_id, variant):
        assert variant in prepared and run_id == plan["run_id"]
        return dirs[variant]

    monkeypatch.setattr(release_prepare_variants, "prepare_exploration", prepare)
    monkeypatch.setattr("pmmc.experiments.release_inputs.exploration_campaign_directory", directory)
    req = dict(kind="quote", plan=plan, variant="wr", stage="main", rate_card=str(card))
    assert command("prepare", cfg, "trial", "main", req)[2:4] == [
        "pmmc.experiments.release_prepare_variants",
        "trial",
    ]
    monkeypatch.setattr(sys, "argv", ["prepare", "trial", "wr"])
    monkeypatch.setenv(
        "PMMC_RELEASE_REQUEST",
        release_env("quote", cfg, "ci", None, req, "main")["PMMC_RELEASE_REQUEST"],
    )
    release_prepare_variants.main()
    assert prepared == plan["variants"]

    db = MemoryLedger(limit)
    db.row = None
    original_get = db.get

    async def get(model, key, **kwargs):
        if model is BenchmarkSelectionModel:
            arm = next(v for v in dirs if step8_cli._latest_selection(dirs[v]).content_hash == key)
            return SimpleNamespace(campaign_id=arm, payload={"release_plan_hash": plan_hash(plan)})
        return await original_get(model, key, **kwargs)

    async def insert(session, model, key, values, compare):
        assert model == BenchmarkBudgetApprovalModel
        if not db.row:
            db.row = model(**values)
        assert db.row.approval_id == values["approval_id"]
        return db.row

    monkeypatch.setattr(db, "get", get)
    monkeypatch.setattr("pmmc.infrastructure.database.step8_repositories._insert_exact", insert)
    monkeypatch.setattr(
        step8_cli, "create_session_factory", lambda *_: (SimpleNamespace(dispose=AsyncMock()), db)
    )
    monkeypatch.setattr(
        "pmmc.infrastructure.database.engine.create_session_factory",
        step8_cli.create_session_factory,
    )
    for method in ("register", "add_jobs", "counts"):
        monkeypatch.setattr(PostgresBenchmarkRepository, method, AsyncMock(return_value={}))
    snapshot = AsyncMock(side_effect=lambda settings, cid: rows[cid])
    monkeypatch.setattr("pmmc.experiments.step8_supervisor._snapshot", snapshot)
    monkeypatch.setattr(release_execution, "_snapshot", snapshot)

    def invoke(role, arg, stage, approval=None):
        env = release_env(role, cfg, "ci", approval, arg, stage)
        argv = command(role, cfg, "trial", stage, arg)
        result = CliRunner().invoke(step8_cli.app, argv[4:], env=env)
        assert result.exit_code == 0, (result.output, result.exception)
        return receipt(role, result.output.splitlines()[-1])

    quote = invoke("quote", req, "main")
    req.update(
        kind="approve",
        selection_hash=quote["selection_hash"],
        approver_id="human",
        uncapped=limit is None,
        limit_usd=str(limit) if limit else None,
    )
    approved = invoke("approve", req, "main")
    aid = approved["approval_id"]
    assert approved["plan_hash"] == plan_hash(plan) and db.row.limit_usd == limit
    assert set(db.row.plan_bindings) == set(plan["variants"])
    assert "OPENAI_API_KEY" not in secret_refs("quote", cfg, None, plan)

    # Exercise the native worker's exact scope/approval/card checks. Replace only
    # the post-admission IO/scientific runtime with a fake accounted transport.
    def fake_backend(**kwargs):
        return SimpleNamespace(runtime_settings=settings)

    monkeypatch.setattr(release_execution, "ProductionComponentBackend", fake_backend)
    monkeypatch.setattr(
        "pmmc.experiments.release_reporting.summarize_release_campaign", lambda *a, **k: {}
    )
    monkeypatch.setattr("pmmc.experiments.release_readouts.seal_report", lambda *a: None)
    monkeypatch.setattr(release_execution, "selection_spend", AsyncMock(return_value={}))
    calls = []

    class WorkerDone(Exception):
        pass

    def supervise(*, directory, job_scope, rate_card, has_api_budget, manifest, **kwargs):
        arm = manifest.campaign_id
        assert has_api_budget and os.environ["PMMC_STEP8_BUDGET_APPROVAL_ID"] == aid
        job = rows[arm][0]
        if job["status"] != "completed":

            def dispatch(_):
                wrapped, fake, gateway, store = fake_accounting(tmp_path / "transport")
                fake.approval = db.row
                native = PostgresBenchmarkRepository(db)
                fake.reserve, fake.settle = native.reserve, native.settle
                wrapped.approval = SelectionApproval(
                    arm, db.row.plan_bindings[arm]["selection_hash"], aid, db.row.rate_card_hash
                )
                request_id = "paid-" + arm
                gateway.result = gateway.result.model_copy(update={"request_id": request_id})
                audit = CampaignModelAudit(store, arm, "test", job["job_id"], "d" * 64)
                with campaign_model_scope(audit):
                    asyncio.run(
                        wrapped.generate(
                            paid_request().model_copy(update={"request_id": request_id})
                        )
                    )
                calls.append(arm)
                raise WorkerDone

            monkeypatch.setattr(step8_cli, "_preflight_response_journal", dispatch)
            with pytest.raises(WorkerDone):
                copy_context().run(
                    step8_cli._run_campaign,
                    campaign_dir=directory,
                    resource=ResourceClass.API,
                    worker_pool="all",
                    stage=None,
                    budget_usd=None,
                    rate_card=rate_card,
                    approver_id=None,
                    retry_failed=False,
                    export_on_complete=False,
                    exploration_scope=job_scope[0],
                )
            job["status"] = "completed"
        return {"physical_startups": {}}

    monkeypatch.setattr(release_execution, "supervise_campaign", supervise)
    for unit in units(plan):
        if unit == "source-analysis":
            continue
        arm, stage = unit.split("-")
        run = dict(kind="run", plan=plan, variant=arm)
        env = release_env("pointer", cfg, "ci", aid, run, stage)
        assert env["PMMC_STEP8_BUDGET_APPROVAL_ID"] == aid
        with monkeypatch.context() as isolated:
            for key, val in env.items():
                isolated.setenv(key, val)
            ptr = command("pointer", cfg, "trial", stage, run)
            isolated.setattr(sys, "argv", ["-c", *ptr[3:]])
            exec(ptr[2], {})
            verify_release_budget_pointer(
                settings, run_id="trial", variant=arm, stage=stage, approval_id=aid
            )
        result = invoke("control", run, stage, aid)
        assert (
            result["paid_authorized"]
            and result["scope_hash"] == db.row.plan_bindings[arm]["scopes"][stage]
        )
    assert sorted(calls) == sorted(plan["variants"]) and db.row.settled_usd == Decimal("0.4")

    arm = "w"
    selected = db.row.plan_bindings[arm]["selection_hash"]
    stage_scope = db.row.plan_bindings[arm]["scopes"]["main"]
    for key, bad in [
        ("PMMC_RELEASE_REQUEST", json.dumps({**plan, "run_id": "other"})),
        ("PMMC_RELEASE_STAGE", "analysis"),
        ("PMMC_RELEASE_VARIANT", "wr"),
    ]:
        with monkeypatch.context() as isolated:
            isolated.setenv(key, bad)
            with pytest.raises(ValueError, match="paid_host"):
                read_scope(dirs[arm], stage_scope, arm, selected)
    path = dirs[arm] / f"manifests/budget-{selected}.json"
    saved = json.loads(path.read_bytes())
    write_json(path, {**saved, "limit_usd": "999"})
    with pytest.raises(ValueError):
        verify_release_budget_pointer(
            settings, run_id="trial", variant=arm, stage="main", approval_id=aid
        )
    write_json(path, saved)
    stop = CliRunner().invoke(
        step8_cli.app,
        [
            "stop-exploration-budget",
            "--run-id",
            "trial",
            "--variant",
            arm,
            "--plan-hash",
            plan_hash(plan),
            "--approval-id",
            aid,
            "--selection-hash",
            selected,
            "--operator-id",
            "test",
            "--reason-code",
            "operator_stop",
        ],
    )
    assert stop.exit_code == 0 and json.loads(stop.output)["new_calls_stopped"]
    native = PostgresBenchmarkRepository(db)
    repeated = asyncio.run(
        native.stop_plan_budget(
            approval_id=aid,
            campaign_id="wr",
            selection_hash=db.row.plan_bindings["wr"]["selection_hash"],
            plan_hash=plan_hash(plan),
            operator_id="another",
            reason_code="different_reason",
        )
    )
    assert repeated["stopped_by"] == "test" and repeated["stop_reason"] == "operator_stop"
    assert repeated["stopped_at"] == json.loads(stop.output)["stopped_at"]
    with pytest.raises(BudgetExceededError, match="permanently_stopped"):
        asyncio.run(
            native.reserve(
                aid, "late", "b" * 64, "test", Decimal("1"), binding=db.bind("wr", "main")
            )
        )
    reapprove = CliRunner().invoke(step8_cli.app, command("approve", cfg, "trial", "main", req)[4:])
    assert reapprove.exit_code != 0 and "permanently_stopped" in str(reapprove.exception)
