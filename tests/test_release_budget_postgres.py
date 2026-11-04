"""Opt-in real ledger tests; never accept a live database URL or existing credentials."""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import subprocess
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pytest

MODE = os.environ.get("PMMC_TEST_DISPOSABLE_LEDGER")
pytestmark = pytest.mark.skipif(
    MODE not in {"local", "kind"}, reason="opt-in disposable PostgreSQL"
)


@contextmanager
def _local_pg(tmp):
    from sqlalchemy.engine import URL

    bindir = (
        os.environ.get("PMMC_TEST_PG_BINDIR")
        or subprocess.check_output(["pg_config", "--bindir"], text=True).strip()
    )
    initdb, ctl = (Path(bindir) / name for name in ("initdb", "pg_ctl"))
    assert initdb.is_file() and ctl.is_file(), "install PostgreSQL server binaries for local mode"
    data = tmp / "data"
    subprocess.run(
        [initdb, "-D", data, "-U", "ci_owner", "-A", "trust", "--no-locale", "-E", "UTF8"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # Use an abstractly short private socket path even on macOS's long pytest tmp paths.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pmmc-ledger-", dir="/tmp") as socketdir:
        subprocess.run(
            [
                ctl,
                "-D",
                data,
                "-l",
                tmp / "server.log",
                "-w",
                "-t",
                "20",
                "-o",
                f"-h '' -k {socketdir}",
                "start",
            ],
            check=True,
        )
        try:
            yield URL.create(
                "postgresql+psycopg",
                username="ci_owner",
                database="postgres",
                query={"host": socketdir},
            )
        finally:
            subprocess.run([ctl, "-D", data, "-m", "immediate", "-w", "stop"], check=True)


@contextmanager
def _kind_pg():
    import selectors

    from scripts.kind_persistence import CONTEXT, NAME, NS, sql
    from sqlalchemy.engine import URL

    if not (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("KUBECONFIG")
        and CONTEXT == "kind-pmmc-release-ci"
        and NS == "pmmc-storage-test"
        and sql("SELECT count(*) FROM persistence_marker") == "1"
    ):
        raise RuntimeError("requires disposable release Kind persistence fixture")
    user = "ledger_" + uuid.uuid4().hex
    password = secrets.token_hex(24)
    # Keep the fresh test login out of command arguments and failure messages.
    kube = ["kubectl", "--context", CONTEXT, "-n", NS]
    subprocess.run(
        [
            *kube,
            "exec",
            "-i",
            f"{NAME}-postgres-0",
            "--",
            "psql",
            "-XAtq",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "pmmc_owner",
            "-d",
            "pmmc",
        ],
        input=f"CREATE ROLE {user} LOGIN PASSWORD '{password}';",
        text=True,
        check=True,
    )
    try:
        sql(f"CREATE DATABASE {user} OWNER {user}")
        runtime_exists = sql("SELECT count(*) FROM pg_roles WHERE rolname='pmmc_runtime'") == "1"
        if not runtime_exists:
            sql("CREATE ROLE pmmc_runtime NOLOGIN")
        try:
            sql(f"GRANT pmmc_runtime TO {user}")
            with subprocess.Popen(
                [
                    *kube,
                    "port-forward",
                    "--address",
                    "127.0.0.1",
                    f"pod/{NAME}-postgres-0",
                    ":5432",
                ],
                stdout=subprocess.PIPE,
                text=True,
            ) as forward:
                try:
                    with selectors.DefaultSelector() as selector:
                        selector.register(forward.stdout, selectors.EVENT_READ)
                        assert selector.select(timeout=20), "port forward did not become ready"
                        line = forward.stdout.readline()
                    port = re.fullmatch(r"Forwarding from 127\.0\.0\.1:(\d+) -> 5432\s*", line)
                    assert port, "unexpected loopback forwarding endpoint"
                    yield URL.create(
                        "postgresql+psycopg",
                        username=user,
                        password=password,
                        host="127.0.0.1",
                        port=int(port[1]),
                        database=user,
                    )
                finally:
                    forward.terminate()
        finally:
            sql(f"DROP DATABASE {user} WITH (FORCE)")
            if not runtime_exists:
                sql("DROP ROLE pmmc_runtime")
    finally:
        sql(f"DROP ROLE {user}")


def test_release_budget_migration_and_concurrent_workers(tmp_path):
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from pmmc.experiments.step8_conformance_replacement import read_ledger
    from pmmc.infrastructure.database.step8_repositories import (
        BudgetExceededError,
        PostgresBenchmarkRepository,
    )

    root = Path(__file__).resolve().parents[1]
    graph = ScriptDirectory.from_config(Config(str(root / "alembic.ini")))
    ddl = graph.get_revision("20260920_0030").module._TABLE_DDL
    predecessor = graph.get_revision("20261104_0037").module
    successor = graph.get_revision("20261104_0038").module
    members = {
        "wr": dict(campaign_id="wr", selection_hash="1" * 64, scopes={"main": "3" * 64}),
        "w": dict(campaign_id="w", selection_hash="2" * 64, scopes={"main": "4" * 64}),
    }
    with _local_pg(tmp_path) if MODE == "local" else _kind_pg() as url:
        owner = create_engine(url)
        with owner.begin() as conn:
            if MODE == "local":
                conn.exec_driver_sql("CREATE ROLE pmmc_runtime NOLOGIN")
            conn.exec_driver_sql("CREATE SCHEMA pmmc_runtime")
            conn.exec_driver_sql("GRANT USAGE ON SCHEMA pmmc_runtime TO pmmc_runtime")
            tables = (
                "method_protocol_freezes",
                "benchmark_campaigns",
                "benchmark_selection_expansions",
                "benchmark_budget_approvals",
                "benchmark_spend_reservations",
            )
            for table in tables:
                conn.exec_driver_sql(f"CREATE TABLE pmmc_runtime.{table} ({ddl[table]})")
            conn.exec_driver_sql(
                "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA pmmc_runtime TO pmmc_runtime"
            )
            conn.exec_driver_sql(
                "GRANT UPDATE (settled_usd, reserved_usd) ON "
                "pmmc_runtime.benchmark_budget_approvals TO pmmc_runtime"
            )
            conn.exec_driver_sql(
                "GRANT UPDATE (actual_usd, response_hash, state) ON "
                "pmmc_runtime.benchmark_spend_reservations TO pmmc_runtime"
            )
            with Operations.context(MigrationContext.configure(conn)):
                predecessor.upgrade()
            for name, sel in [
                ("old", "0" * 64),
                *((v, m["selection_hash"]) for v, m in members.items()),
            ]:
                conn.execute(
                    text(
                        "INSERT INTO pmmc_runtime.benchmark_campaigns VALUES "
                        "(:name, :h, NULL, 'exploration', 'planned', :h, '{}', now())"
                    ),
                    dict(name=name, h="c" * 64),
                )
                conn.execute(
                    text(
                        "INSERT INTO pmmc_runtime.benchmark_selection_expansions VALUES "
                        "(:sel, :name, 'dev-all', NULL, "
                        "jsonb_build_object('release_plan_hash', CAST(:h AS TEXT)), now())"
                    ),
                    dict(name=name, sel=sel, h="b" * 64),
                )
            conn.execute(
                text(
                    "INSERT INTO pmmc_runtime.benchmark_budget_approvals VALUES "
                    "(:a, 'old', :s, 'legacy', :r, 10, 2, 1, now())"
                ),
                dict(a="9" * 64, s="0" * 64, r="8" * 64),
            )
            with Operations.context(MigrationContext.configure(conn)):
                successor.upgrade()
                successor.downgrade()  # Legacy approvals survive both directions.
                successor.upgrade()
            assert conn.exec_driver_sql(
                "SELECT limit_usd, settled_usd, reserved_usd, plan_hash "
                "FROM pmmc_runtime.benchmark_budget_approvals"
            ).one() == (10, 2, 1, None)

        async def exercise(limit):
            engine = create_async_engine(url, connect_args={"options": "-c role=pmmc_runtime"})
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            repo = PostgresBenchmarkRepository(sessions)
            args = dict(
                campaign_id="wr",
                selection_hash=members["wr"]["selection_hash"],
                approver_id="ci",
                rate_card_hash="e" * 64,
                limit_usd=limit,
                uncapped=limit is None,
                plan_hash="b" * 64,
                plan_bindings=members,
            )
            try:
                approval = await repo.approve_budget(**args)
                assert approval == await repo.approve_budget(**args)

                async def reserve(i):
                    arm = ("wr", "w")[i % 2]
                    return await repo.reserve(
                        approval,
                        f"call{i}",
                        "a" * 64,
                        "test",
                        Decimal("2"),
                        binding=dict(
                            plan_hash="b" * 64,
                            campaign_id=arm,
                            selection_hash=members[arm]["selection_hash"],
                            scope_hash=members[arm]["scopes"]["main"],
                        ),
                    )

                results = await asyncio.gather(
                    *(reserve(i) for i in range(8)), return_exceptions=True
                )
                successes = [i for i, result in enumerate(results) if result == "reserved"]
                assert len(successes) == (8 if limit is None else 2)
                assert all(
                    result == "reserved" or isinstance(result, BudgetExceededError)
                    for result in results
                )
                await repo.settle(f"call{successes[0]}", None, None)
                stopped = await repo.stop_plan_budget(
                    approval_id=approval,
                    plan_hash="b" * 64,
                    campaign_id="w",
                    selection_hash=members["w"]["selection_hash"],
                    operator_id="ci",
                    reason_code="operator_stop",
                )
                assert stopped["stop_reason"] == (
                    "operator_stop" if limit is None else "approved_api_budget_exhausted"
                )
                with pytest.raises(BudgetExceededError):
                    await reserve(99)
                with pytest.raises(ValueError, match="permanently_stopped"):
                    await repo.approve_budget(**args)
                await repo.settle(f"call{successes[1]}", Decimal("1"), "c" * 64)
                snapshot, calls = await read_ledger(sessions, approval)
                assert len(calls) == len(successes) and snapshot["limit_usd"] == limit
                assert (
                    snapshot["reserved_usd"] == 2 * (len(successes) - 1)
                    and snapshot["settled_usd"] == 1
                )
                assert any(r["state"] == "unknown" for r in calls)
            finally:
                await engine.dispose()

        for limit in (Decimal("4"), None):
            asyncio.run(exercise(limit))
            with pytest.raises(DBAPIError), owner.begin() as conn:
                conn.exec_driver_sql(
                    "UPDATE pmmc_runtime.benchmark_budget_approvals SET stopped_at=NULL, "
                    "stopped_by=NULL, stop_reason=NULL WHERE plan_hash IS NOT NULL"
                )
            with pytest.raises(DBAPIError), owner.begin() as conn:
                with Operations.context(MigrationContext.configure(conn)):
                    successor.downgrade()
            # Isolate the next scenario inside this disposable database only.
            with owner.begin() as conn:
                conn.exec_driver_sql("DELETE FROM pmmc_runtime.benchmark_spend_reservations")
                conn.exec_driver_sql(
                    "DELETE FROM pmmc_runtime.benchmark_budget_approvals "
                    "WHERE plan_hash IS NOT NULL"
                )
        owner.dispose()
