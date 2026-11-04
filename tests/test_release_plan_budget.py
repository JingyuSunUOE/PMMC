from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from typer.testing import CliRunner

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.release_execution import selection_spend
from pmmc.experiments.release_reporting import approve_release_budget, quote_release_plan
from pmmc.experiments.step8_artifacts import write_json
from pmmc.experiments.step8_cli import app
from pmmc.experiments.step8_jobs import BenchmarkStageJob
from pmmc.infrastructure.database.step8_models import (
    BenchmarkBudgetApprovalModel,
    BenchmarkSelectionModel,
    BenchmarkSpendReservationModel,
)
from pmmc.infrastructure.database.step8_repositories import (
    BudgetExceededError,
    PostgresBenchmarkRepository,
    validate_budget_policy,
)
from pmmc.models.campaign_audit import CampaignModelAudit, campaign_model_scope
from pmmc.models.providers import ProviderCallError
from pmmc.models.step8_accounting import SelectionApproval, bind_release_worker
from pmmc.release_plan import normalize_plan, plan_hash
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, build_experiment_matrix
from test_release_reporting import GPT, ROOT, Campaign
from test_step8_accounting import request as paid_request
from test_step8_accounting import setup as fake_accounting


class MemoryLedger:
    """Exercise the production repo with short locked transactions, no external services."""

    def __init__(self, limit):
        self.lock = asyncio.Lock()
        self.row = BenchmarkBudgetApprovalModel(
            approval_id="a" * 64,
            campaign_id="wr",
            selection_hash="1" * 64,
            plan_hash="b" * 64,
            rate_card_hash="c" * 64,
            approver_id="reviewer",
            plan_bindings={
                "wr": dict(
                    campaign_id="wr",
                    selection_hash="1" * 64,
                    scopes={"main": "d" * 64, "baselines": "e" * 64},
                ),
                "w": dict(campaign_id="w", selection_hash="2" * 64, scopes={"main": "f" * 64}),
            },
            settled_usd=Decimal(0),
            reserved_usd=Decimal(0),
            limit_usd=limit,
        )
        self.calls = {}

    @asynccontextmanager
    async def begin(self):
        async with self.lock:
            yield self

    async def get(self, model, key, **kwargs):
        if model is BenchmarkBudgetApprovalModel:
            assert kwargs["with_for_update"] and key == self.row.approval_id
            return self.row
        assert model is BenchmarkSpendReservationModel
        return self.calls.get(key)

    async def refresh(self, row, with_for_update):
        assert with_for_update

    def add(self, row):
        assert row.request_id not in self.calls
        self.calls[row.request_id] = row

    def bind(self, variant, stage):
        member = self.row.plan_bindings[variant]
        return dict(
            plan_hash=self.row.plan_hash,
            scope_hash=member["scopes"][stage],
            campaign_id=member["campaign_id"],
            selection_hash=member["selection_hash"],
        )


@pytest.mark.parametrize("limit", [None, Decimal("5.00")])
def test_shared_reservations_unknowns_and_dispatch_stop(limit):
    db = MemoryLedger(limit)
    repo = PostgresBenchmarkRepository(db)
    wr, w = db.bind("wr", "main"), db.bind("w", "main")

    async def reserve(key, binding, amount):
        return await repo.reserve(
            db.row.approval_id, key, "a" * 64, "openai", Decimal(amount), binding=binding
        )

    async def run():
        for tamper in (
            {"campaign_id": "unapproved"},
            {"selection_hash": wr["selection_hash"]},
            {"scope_hash": wr["scope_hash"]},
            {"plan_hash": "0" * 64},
        ):
            with pytest.raises(ValueError, match="binding"):
                await reserve("bad", {**w, **tamper}, "1")
        with pytest.raises(ValueError, match="binding"):
            await reserve("bad", None, "1")
        assert not db.calls
        assert await reserve("unknown", wr, "2") == "reserved"
        await repo.settle("unknown", None, None)
        assert await reserve("unknown", wr, "2") == "unknown" and db.row.reserved_usd == 2
        results = await asyncio.gather(
            reserve("w-main", w, "3"),
            reserve("wr-baselines", db.bind("wr", "baselines"), "3"),
            return_exceptions=True,
        )
        if limit is None:
            assert results == ["reserved"] * 2 and db.row.reserved_usd == 8
            assert db.row.stopped_at is None
        else:
            assert results.count("reserved") == 1
            assert sum(isinstance(v, BudgetExceededError) for v in results) == 1
            assert db.row.reserved_usd == 5 and db.row.stopped_at is not None
        await repo.settle("w-main", Decimal(1), "d" * 64)
        if limit is not None:
            with pytest.raises(BudgetExceededError):
                await reserve("after-stop", w, "1")
            assert "after-stop" not in db.calls and db.row.reserved_usd == 2
        await repo.settle("unknown", Decimal("1"), "c" * 64)
        assert db.row.settled_usd == 2

    asyncio.run(run())


def test_plan_approval_identity_and_legacy_replay(monkeypatch):
    db = MemoryLedger(None)
    repo = PostgresBenchmarkRepository(db)
    saved = AsyncMock(return_value=SimpleNamespace(stopped_at=None))
    monkeypatch.setattr("pmmc.infrastructure.database.step8_repositories._insert_exact", saved)

    async def registered(model, key):
        assert model is BenchmarkSelectionModel
        member = next(m for m in db.row.plan_bindings.values() if m["selection_hash"] == key)
        return SimpleNamespace(
            campaign_id=member["campaign_id"], payload={"release_plan_hash": db.row.plan_hash}
        )

    monkeypatch.setattr(db, "get", registered)
    base = dict(
        campaign_id="wr", selection_hash="1" * 64, approver_id="user", rate_card_hash="c" * 64
    )
    plan = dict(plan_hash=db.row.plan_hash, plan_bindings=db.row.plan_bindings)

    async def check():
        old = await repo.approve_budget(**base, limit_usd=Decimal("1.00"))
        assert old == canonical_content_hash({**base, "limit_usd": "1.00"})
        finite = await repo.approve_budget(**base, **plan, limit_usd=Decimal("1.00"))
        assert finite != old
        assert finite == await repo.approve_budget(**base, **plan, limit_usd=Decimal("1"))
        uncapped = await repo.approve_budget(**base, **plan, limit_usd=None, uncapped=True)
        assert uncapped != finite
        assert saved.await_args.args[3]["limit_usd"] is None
        with pytest.raises(ValueError, match="not_registered"):
            await repo.approve_budget(
                **base, **{**plan, "plan_hash": "0" * 64}, limit_usd=Decimal("1")
            )

    asyncio.run(check())


def test_disputed_usage_preserves_hold_and_stops_all_new_plan_calls():
    db = MemoryLedger(None)
    repo = PostgresBenchmarkRepository(db)

    async def check():
        await repo.reserve(
            db.row.approval_id,
            "disputed",
            "d" * 64,
            "openai",
            Decimal(1),
            binding=db.bind("w", "main"),
        )
        with pytest.raises(BudgetExceededError, match="worst_case"):
            await repo.settle("disputed", Decimal(2), "c" * 64)
        assert db.calls["disputed"].state == "unknown"
        assert db.row.reserved_usd == 1 and db.row.settled_usd == 0
        with pytest.raises(BudgetExceededError):
            await repo.reserve(
                db.row.approval_id,
                "new",
                "e" * 64,
                "openai",
                Decimal(1),
                binding=db.bind("wr", "baselines"),
            )
        assert "new" not in db.calls

    asyncio.run(check())


@pytest.mark.parametrize(
    "amount,uncapped",
    [
        (None, False),
        (Decimal(1), True),
        (Decimal("NaN"), False),
        (Decimal(0), False),
        (Decimal("Infinity"), False),
        (Decimal("0.000000001"), False),
    ],
)
def test_invalid_policies(amount, uncapped):
    with pytest.raises(ValueError):
        validate_budget_policy(amount, uncapped)


def test_paid_dispatch_requires_verified_child_job_scope(tmp_path, monkeypatch):
    plan = normalize_plan(dict(run_id="trial", stage="main", models=[GPT], variants=["wr", "w"]))
    monkeypatch.setenv("PMMC_RELEASE_REQUEST", json.dumps(plan))
    wrapped, fake, gateway, store = fake_accounting(tmp_path)
    db = MemoryLedger(None)
    repo = PostgresBenchmarkRepository(db)
    fake.reserve, fake.settle = repo.reserve, repo.settle
    db.row.plan_hash = plan_hash(plan)
    fake.approval = db.row
    member = db.bind("w", "main")
    wrapped.approval = SelectionApproval(
        "w", member["selection_hash"], db.row.approval_id, db.row.rate_card_hash
    )

    async def run():
        with pytest.raises(ProviderCallError):
            await wrapped.generate(paid_request())
        with campaign_model_scope(CampaignModelAudit(store, "w", "attempt", "job1", "f" * 64)):
            bind_release_worker(plan, "w", member["selection_hash"], member["scope_hash"], ["job2"])
            with pytest.raises(ProviderCallError):
                await wrapped.generate(paid_request())
            bind_release_worker(plan, "w", member["selection_hash"], "0" * 64, ["job1"])
            with pytest.raises(ProviderCallError):
                await wrapped.generate(paid_request())
            assert gateway.calls == 0 and not db.calls
            bind_release_worker(plan, "w", member["selection_hash"], member["scope_hash"], ["job1"])
            result = await wrapped.generate(paid_request())
            assert await wrapped.generate(paid_request()) == result
        assert gateway.calls == 1 and db.row.settled_usd == Decimal("0.1")

    asyncio.run(run())


def test_native_policy_flags_are_exclusive(tmp_path):
    card = tmp_path / "rates"
    card.touch()
    args = [
        "approve-exploration",
        "--run-id",
        "test",
        "--selection-hash",
        "1" * 64,
        "--approver-id",
        "test",
        "--rate-card",
        str(card),
    ]
    for extra in ([], ["--uncapped", "--limit-usd", "1"]):
        result = CliRunner().invoke(app, [*args, *extra])
        assert result.exit_code == 2 and "choose exactly one" in result.output


def test_successor_accepts_legacy_and_nullable_plan_limits():
    graph = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    assert graph.get_heads() == [STEP8_ALEMBIC_REVISION]
    assert graph.get_revision(STEP8_ALEMBIC_REVISION).down_revision == "20261104_0037"
    assert all(
        BenchmarkBudgetApprovalModel.__table__.c[name].nullable
        for name in ("limit_usd", "plan_hash", "plan_bindings")
    )


def budget_children(tmp_path, plan):
    digest = plan_hash(plan)
    dirs, rows = {}, {}
    for arm in plan["variants"]:
        fixture = Campaign(tmp_path / arm)
        folder = fixture.dir
        dirs[arm] = folder
        config_file = folder / "manifests/config.json"
        config = json.loads(config_file.read_bytes())
        config["metadata"].update(
            release_request=plan, exploration_variant=arm, exploration_run_id="trial"
        )
        write_json(config_file, config)
        cells = build_experiment_matrix(fixture.protocol, plan, variant=arm)
        manifest = fixture.manifest.model_copy(
            update={
                "release_plan_hash": digest,
                "campaign_id": arm,
                "matrix_hash": canonical_content_hash([c.model_dump(mode="json") for c in cells]),
            }
        )
        write_json(folder / "manifests/campaign.json", manifest.model_dump(mode="json"))
        old = folder / f"manifests/selections/{fixture.selection.content_hash}.json"
        selection = fixture.selection.model_copy(
            update={"release_plan_hash": digest, "campaign_id": arm}
        )
        old.rename(old.with_name(selection.content_hash + ".json"))
        write_json(
            old.with_name(selection.content_hash + ".json"), selection.model_dump(mode="json")
        )
        job = BenchmarkStageJob(
            campaign_id=arm,
            stage="answers",
            component_key="answer",
            operation="query_answer",
            resource_class="api",
            source_fingerprint="d" * 64,
            release_stages=(
                ("main", "analysis", "baselines")
                if plan["stage"] == "all" and arm == "wr"
                else ("main",)
            ),
            scientific_inputs={
                "query": "q0",
                "model_entry_hashes": {GPT: fixture.protocol.model_entry_hashes[GPT]},
            },
            parameters={"dataset": "memlens", "answer_model": GPT, "readout": "primary"},
        )
        payload = job.model_dump(mode="json")
        write_json(folder / f"private/jobs/{job.job_id}.json", payload)
        write_json(folder / f"manifests/jobs-{selection.content_hash}.json", [job.job_id])
        rows[arm] = [dict(status="pending", job_id=job.job_id, job=payload)]
    return dirs, rows, fixture.card


def test_plan_quote_approval_and_report_bind_both_children(tmp_path, monkeypatch):
    plan = normalize_plan(dict(run_id="trial", stage="main", models=[GPT], variants=["wr", "w"]))
    digest = plan_hash(plan)
    monkeypatch.setenv("PMMC_RELEASE_REQUEST", json.dumps(plan))
    dirs, rows, card = budget_children(tmp_path, plan)
    settings = SimpleNamespace(project_root=ROOT)
    monkeypatch.setattr("pmmc.infrastructure.settings.get_settings", lambda: settings)
    monkeypatch.setattr(
        "pmmc.experiments.release_inputs.exploration_campaign_directory",
        lambda settings, run_id, variant: dirs[variant],
    )
    monkeypatch.setattr("pmmc.experiments.step8_cli._register", AsyncMock())
    monkeypatch.setattr(
        "pmmc.experiments.step8_supervisor._snapshot",
        AsyncMock(side_effect=lambda settings, cid: rows[cid]),
    )
    anchor = dirs["w"]
    quote = asyncio.run(quote_release_plan(settings, anchor, scope="main", rate_card=card))
    assert quote["plan_hash"] == digest and set(quote["plan_bindings"]) == {"wr", "w"}
    assert len({q["scope_hash"] for q in quote["stages"]}) == 2
    path = anchor / "reports/budget-quote-main.json"
    args = dict(
        uncapped=True,
        selection_hash=quote["selection_hash"],
        rate_card=card,
        scope="main",
        approver_id="human",
        project_root=ROOT,
    )
    register = AsyncMock(return_value="a" * 64)
    monkeypatch.setattr(PostgresBenchmarkRepository, "approve_budget", register)
    monkeypatch.setattr(
        "pmmc.experiments.step8_cli.create_session_factory",
        lambda *_: (SimpleNamespace(dispose=AsyncMock()), object()),
    )
    write_json(path, {**quote, "plan_bindings": {"w": quote["plan_bindings"]["w"]}})
    with pytest.raises(ValueError, match="saved_quote"):
        asyncio.run(approve_release_budget(anchor, **args))
    register.assert_not_called()
    write_json(path, quote)
    result = asyncio.run(approve_release_budget(anchor, **args))
    assert result["limit_usd"] is None and result["approval_id"] == "a" * 64
    assert (
        register.await_args.kwargs["campaign_id"] == "wr"
    )  # Same canonical anchor from either child.
    for arm, member in quote["plan_bindings"].items():
        saved = json.loads(
            (dirs[arm] / f"manifests/budget-{member['selection_hash']}.json").read_bytes()
        )
        assert (
            saved["approval_id"] == result["approval_id"]
            and saved["scope_hashes"] == member["scopes"]
        )
    with pytest.raises(ValueError, match="conflicting"):
        asyncio.run(
            approve_release_budget(anchor, **{**args, "uncapped": False, "limit_usd": Decimal(1)})
        )
    ledger = dict(
        plan_hash=digest,
        plan_bindings=quote["plan_bindings"],
        limit_usd=None,
        reserved_usd=Decimal("4.20"),
        settled_usd=Decimal(1),
    )
    monkeypatch.setattr(
        "pmmc.experiments.release_execution.create_session_factory",
        lambda *_: (SimpleNamespace(dispose=AsyncMock()), object()),
    )
    monkeypatch.setattr(
        "pmmc.experiments.step8_conformance_replacement.read_ledger",
        AsyncMock(return_value=(ledger, [dict(state="unknown", reserved_usd=Decimal("4.20"))])),
    )
    spend = asyncio.run(
        selection_spend(settings, result["approval_id"], "w", quote["selection_hash"])
    )
    assert spend["remaining_usd"] is None and spend["unknown_reserved_usd"] == "4.20"
    monkeypatch.setattr("pmmc.experiments.step8_cli.get_settings", lambda: settings)
    command = CliRunner().invoke(app, ["spend-exploration", "--run-id", "trial", "--variant", "w"])
    assert command.exit_code == 0 and json.loads(command.output)["ledger"] == spend
    with pytest.raises(ValueError, match="identity_mismatch"):
        asyncio.run(selection_spend(settings, result["approval_id"], "wr", quote["selection_hash"]))
