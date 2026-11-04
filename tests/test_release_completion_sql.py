"""Exercise the installed PL/pgSQL trigger with native reports, without a server.

Uses a locally cached PGlite package (or PMMC_TEST_PGLITE pointing to its
distribution directory). Never downloads packages or opens a network socket.
"""

import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.dialects.postgresql import dialect

from pmmc.evaluation import step8_workflow
from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.step8_evaluation_scope import install_evaluation_scope
from test_release_reporting import ROOT
from test_step8_workflow import answer, cell, facade_fixture

_BRIDGE = """
const { PGlite } = require(process.argv[1]);
(async () => {
  const db = new PGlite();
  const lines = require('readline').createInterface({input: process.stdin});
  for await (const line of lines) {
    const [sql, params] = JSON.parse(line);
    try { console.log(JSON.stringify({rows:(await db.query(sql,params)).rows})); }
    catch(e) { console.log(JSON.stringify({error:e.message, code:e.code})); }
  }
  await db.close();
})().catch(e => {console.error(e); process.exit(1)});
"""


@pytest.fixture
def embedded_pg(tmp_path):
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node:
        pytest.skip("embedded SQL audit requires Node")
    module = os.getenv("PMMC_TEST_PGLITE")
    if not module:
        if not npm:
            pytest.skip("no local PGlite package")
        # npm pack writes to its global cache even in offline mode. Read only
        # this known public tarball using npm's own integrity-checking reader.
        cache = Path(
            subprocess.check_output(
                [
                    npm,
                    "config",
                    "get",
                    "cache",
                    "--offline",
                    "--no-update-notifier",
                    "--logs-max=0",
                ],
                text=True,
            ).strip()
        )
        fetch = """const load = require('module').createRequire(process.argv[1]);
          load('cacache').get(process.argv[2], 'make-fetch-happen:request-cache:' +
            'https://registry.npmjs.org/@electric-sql/pglite/-/pglite-0.2.17.tgz')
            .then(x=>process.stdout.write(x.data)).catch(()=>process.exit(1))"""
        pack = subprocess.run(
            [node, "-e", fetch, str(Path(npm).resolve()), str(cache / "_cacache")],
            capture_output=True,
            timeout=30,
        )
        if pack.returncode:
            pytest.skip("PGlite 0.2.17 not cached; set PMMC_TEST_PGLITE to run offline SQL audit")
        with tarfile.open(fileobj=io.BytesIO(pack.stdout)) as bundle:
            bundle.extractall(tmp_path, filter="data")
        module = str(tmp_path / "package/dist/index.cjs")
    proc = subprocess.Popen(
        [node, "-e", _BRIDGE, module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def execute(statement, params=()):
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps([statement.replace("%s", "$1"), params]) + "\n")
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(f"{reply['code']}: {reply['error']}")
        return SimpleNamespace(fetchone=lambda: next(iter(reply["rows"]), None))

    try:
        yield SimpleNamespace(execute=execute)
    finally:
        proc.stdin.close()
        assert proc.wait(timeout=20) == 0


@pytest.fixture
def installed_pg(embedded_pg, monkeypatch):
    pg = embedded_pg
    revision = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_revision(
        "20260920_0032"
    )
    migration_sql = []
    monkeypatch.setattr(revision.module, "op", SimpleNamespace(execute=migration_sql.append))
    revision.module._create_evaluator_completion()
    base, trigger = migration_sql[:2]

    # Mirror the native catalog names and types. Other stages' constraints are
    # not under test: only their persisted source facts are supplied here.
    for schema in ("pmmc_runtime", "pmmc_evaluation"):
        pg.execute(f"CREATE SCHEMA {schema}")
    tables = set(re.findall(r"pmmc_(?:runtime|evaluation)\.\w+", base)) | {
        "pmmc_runtime.baseline_suites",
        "pmmc_runtime.baseline_suite_queries",
        "pmmc_evaluation.evaluation_run_states",
    }
    tables.discard("pmmc_evaluation.enforce_step8_evaluation_completion")
    for name in sorted(tables):
        columns = Base.metadata.tables[name].columns
        pg.execute(
            f"CREATE TABLE {name} ("
            + ",".join(f'"{c.name}" {c.type.compile(dialect=dialect())}' for c in columns)
            + ")"
        )

    pg.execute(base)
    pg.execute(trigger)
    first = install_evaluation_scope(pg)
    assert first["changed"] and not install_evaluation_scope(pg)["changed"]
    assert (
        pg.execute(
            "SELECT tgfoid::regprocedure::text AS fn FROM pg_trigger "
            "WHERE tgname='step8_evaluation_completion'"
        ).fetchone()["fn"]
        == first["function"]
    )

    def insert(name, **row):
        params = [json.dumps(v) if isinstance(v, dict) else v for v in row.values()]
        pg.execute(
            f"INSERT INTO {name} ({','.join(row)}) VALUES ("
            + ",".join(f"${i}" for i in range(1, len(params) + 1))
            + ")",
            params,
        )

    pg.insert = insert
    return pg


def test_standalone_completion_trigger_and_recovery(installed_pg, monkeypatch):
    pg, insert = installed_pg, installed_pg.insert

    method = "full_context_captions"
    typed = answer("abstention", method=method, variant="primary")
    binding = cell(typed, identifier="baseline", method=method)
    binding = replace(binding, source_fact_id=typed.typed_answer_id, source_variant="primary")
    fixture = facade_fixture(monkeypatch, [binding])
    result = fixture.run(method_scope="baseline")
    meta = fixture.register.await_args.kwargs["metadata"]
    artifact = result.pop("artifact")
    run = result["evaluation_run_id"]
    ds, model, snapshot = (
        "memgallery",
        "answer-qwen3.5-9b",
        result["reference_annotation"]["snapshot_id"],
    )
    plan_hash = "b" * 64
    insert(
        "pmmc_runtime.benchmark_campaigns",
        campaign_id="campaign",
        protocol_hash=meta["protocol_hash"],
        phase="exploration",
        payload={"release_plan_hash": plan_hash, "snapshot_ids": {ds: snapshot}},
    )
    insert(
        "pmmc_runtime.baseline_suites",
        suite_id="suite",
        dataset=ds,
        snapshot_id=snapshot,
        answer_model_alias=model,
        methods=[method],
        metadata={
            "release_plan_hash": plan_hash,
            "release_campaign_id": "campaign",
            "method_protocol_hash": meta["protocol_hash"],
            "partition_role": "development",
            "paper_result": False,
            "answer_contract_version": "typed-answer-contract-v3",
            "standalone_method": method,
        },
    )
    insert(
        "pmmc_runtime.baseline_suite_queries",
        suite_id="suite",
        query_id="query",
        snapshot_id=snapshot,
    )
    insert("pmmc_runtime.baseline_suite_runs", suite_id="suite", method=method, run_id="run")
    insert("pmmc_runtime.run_states", run_id="run", status="completed")
    insert("pmmc_runtime.run_queries", run_id="run", query_id="query")
    insert(
        "pmmc_runtime.benchmark_cell_assignments",
        campaign_id="campaign",
        query_id="query",
        cell_id="baseline",
    )
    insert(
        "pmmc_runtime.benchmark_experiment_cells",
        cell_id="baseline",
        protocol_hash=meta["protocol_hash"],
        dataset=ds,
        answer_model=model,
        method=method,
        payload={
            "treatment": "main",
            "source_variant": "active",
            "question_budget": 8,
            "round_id": 2,
        },
    )
    insert(
        "pmmc_runtime.query_runtime_typed_answers",
        query_executor_run_id="run",
        query_id="query",
        typed_answer_id=typed.typed_answer_id,
        content_hash=binding.source_fact_hash,
        terminal_state=typed.terminal_state,
        payload={"metadata": {"typed_result": typed.model_dump(mode="json")}},
    )
    insert(
        "pmmc_evaluation.query_gold",
        query_id="query",
        snapshot_id=snapshot,
        answer="May",
        gold_metadata={},
    )
    insert(
        "pmmc_evaluation.evaluation_runs",
        evaluation_run_id=run,
        protocol_version=result["schema_version"],
        config_hash=result["identity_hash"],
        source_kind="baseline_suite",
        legacy_source_run_id="run",
        suite_id="suite",
        metadata=meta,
    )
    insert(
        "pmmc_evaluation.run_artifacts",
        artifact_id=artifact["artifact_id"],
        sha256=artifact["sha256"],
        evaluation_run_id=run,
        run_id="run",
        query_id="query",
        snapshot_id=snapshot,
        artifact_type="trace_export",
        mime_type="application/json",
        state="ready",
        ready_at="2026-01-01",
        size_bytes=10,
        storage_uri=artifact["storage_uri"],
        object_key=artifact["object_key"],
    )
    insert(
        "pmmc_evaluation.evaluation_run_states",
        evaluation_run_id=run,
        status="running",
        selected_count=1,
        target_count=1,
        succeeded_count=0,
        failed_count=0,
    )

    def finish(report):
        return pg.execute(
            "UPDATE pmmc_evaluation.evaluation_run_states SET status='completed', "
            "succeeded_count=1, completed_at=now(), metadata=$1",
            [
                json.dumps(
                    {
                        "completion_report": report,
                        "completion_artifact": {k: artifact[k] for k in ("artifact_id", "sha256")},
                    }
                )
            ],
        )

    with pytest.raises(RuntimeError, match="55000: step8_evaluation_method_scope_invalid"):
        finish({**result, "method_scope": "paired_main"})
    with pytest.raises(RuntimeError, match="55000: step8_evaluation_artifact_or_identity_invalid"):
        finish({**result, "query_executor_run_id": "not-a-pmmc-run"})
    pg.execute("UPDATE pmmc_runtime.run_states SET status='running'")
    with pytest.raises(RuntimeError, match="55000: step8_evaluation_artifact_or_identity_invalid"):
        finish(result)
    pg.execute("UPDATE pmmc_runtime.run_states SET status='completed'")
    assert (
        pg.execute("SELECT status FROM pmmc_evaluation.evaluation_run_states").fetchone()["status"]
        == "running"
    )

    class Session:
        @staticmethod
        async def get(model, key):
            table = model.__table__
            (pk,) = table.primary_key.columns
            r = pg.execute(f"SELECT * FROM {table.fullname} WHERE {pk.name}=$1", [key]).fetchone()
            if r and "metadata" in r:
                r["metadata_json"] = r.pop("metadata")
            return SimpleNamespace(**r) if r else None

    @asynccontextmanager
    async def sessions():
        yield Session()

    def recover():
        asyncio.run(
            step8_workflow.verify_selected_score_recovery(
                None,
                {**result, "artifact": artifact},
                campaign_id="campaign",
                suite_id="suite",
                source_run_id="run",
                query_id="query",
                snapshot_id=snapshot,
                method_scope="baseline",
            )
        )

    monkeypatch.setattr(
        step8_workflow,
        "create_session_factory",
        lambda *_: (SimpleNamespace(dispose=AsyncMock()), sessions),
    )
    with pytest.raises(ValueError, match="requires_committed_evaluation"):
        recover()
    finish(result)
    recover()
    key = artifact["object_key"]
    original = fixture.state["objects"][key]
    fixture.state["objects"][key] = b"damaged"
    with pytest.raises(ValueError, match="artifact_changed"):
        recover()
    fixture.state["objects"][key] = original
    pg.execute("UPDATE pmmc_evaluation.run_artifacts SET state='pending'")
    with pytest.raises(ValueError, match="requires_committed_evaluation"):
        recover()
    assert (
        pg.execute("SELECT status FROM pmmc_evaluation.evaluation_run_states").fetchone()["status"]
        == "completed"
    )


@pytest.mark.parametrize("scope", ["pmmc_main", "pmmc"])
def test_installed_trigger_uses_the_runtime_readout_not_just_a_scope_label(
    installed_pg, monkeypatch, scope
):
    pg, insert = installed_pg, installed_pg.insert
    primary = scope == "pmmc_main"
    # Both assignments exist in the campaign. The primary evaluator must score
    # exactly the active cell; the analysis evaluator must score both.
    definitions = [("active", "pmmc"), ("no_fallback", "no_fallback")]
    inputs = [
        replace(
            cell(terminal="no_answer", identifier=v),
            source_variant=v,
            question_budget=8,
            round_id=2,
        )
        for v, _ in definitions
    ]
    fixture = facade_fixture(monkeypatch, inputs[:1] if primary else inputs)
    report = fixture.run(method_scope=scope)
    meta = fixture.register.await_args.kwargs["metadata"]
    artifact = report.pop("artifact")
    run = report["evaluation_run_id"]
    expected = len(report["scores"])
    assert expected == (1 if primary else 2)
    plan = "a" * 64
    insert(
        "pmmc_runtime.benchmark_campaigns",
        campaign_id="campaign",
        phase="exploration",
        protocol_hash=meta["protocol_hash"],
        payload={"release_plan_hash": plan},
    )
    insert(
        "pmmc_runtime.baseline_suites",
        suite_id="suite",
        methods=["native_multimodal_rag"],
        metadata={
            "release_plan_hash": plan,
            "release_campaign_id": "campaign",
            "method_protocol_hash": meta["protocol_hash"],
            "internal_pmmc_fallback": True,
        },
    )
    insert(
        "pmmc_runtime.baseline_suite_runs",
        suite_id="suite",
        method="native_multimodal_rag",
        run_id="fallback",
    )
    insert(
        "pmmc_runtime.query_executor_runs",
        query_executor_run_id="run",
        schema_version="0.3.0",
        snapshot_id="snapshot",
        dataset="memgallery",
        status="completed",
        metadata={
            "treatment": "main",
            "answer_model_alias": "answer-qwen3.5-9b",
            "fallback_run_id": "fallback",
            **(
                {
                    "readout": "exploration-primary-b8-r2",
                    "execute_full_grid": False,
                    "declared_cells": [[8, 2]],
                    "expected_cell_count": 3,
                }
                if primary
                else {}
            ),
        },
    )
    insert("pmmc_runtime.run_queries", run_id="run", query_id="query")
    for (variant, treatment), binding in zip(definitions, inputs, strict=True):
        insert(
            "pmmc_runtime.benchmark_cell_assignments",
            campaign_id="campaign",
            query_id="query",
            cell_id=variant,
        )
        insert(
            "pmmc_runtime.benchmark_experiment_cells",
            cell_id=variant,
            method="pmmc",
            dataset="memgallery",
            protocol_hash=meta["protocol_hash"],
            answer_model="answer-qwen3.5-9b",
            payload=dict(source_variant=variant, treatment="main", question_budget=8, round_id=2),
        )
        insert(
            "pmmc_runtime.query_experiment_cell_results",
            cell_result_id=binding.source_fact_id,
            query_executor_run_id="run",
            query_id="query",
            content_hash=binding.source_fact_hash,
            treatment=treatment,
            question_budget=8,
            revision_depth=2,
            terminal_state="no_answer",
        )
    insert(
        "pmmc_evaluation.query_gold",
        query_id="query",
        snapshot_id="snapshot",
        answer="May",
        gold_metadata={},
    )
    insert(
        "pmmc_evaluation.evaluation_runs",
        evaluation_run_id=run,
        config_hash=report["identity_hash"],
        protocol_version=report["schema_version"],
        source_kind="pmmc_query_run",
        legacy_source_run_id="run",
        metadata=meta,
    )
    insert(
        "pmmc_evaluation.run_artifacts",
        evaluation_run_id=run,
        run_id="run",
        query_id="query",
        snapshot_id="snapshot",
        artifact_id=artifact["artifact_id"],
        sha256=artifact["sha256"],
        storage_uri=artifact["storage_uri"],
        state="ready",
        ready_at="2026-01-01",
        size_bytes=10,
        artifact_type="trace_export",
        mime_type="application/json",
    )
    insert(
        "pmmc_evaluation.evaluation_run_states",
        evaluation_run_id=run,
        status="running",
        selected_count=1,
        target_count=expected,
        failed_count=0,
        succeeded_count=0,
    )

    def finish(method):
        pg.execute(
            "UPDATE pmmc_evaluation.evaluation_runs SET metadata=$1",
            [json.dumps({**meta, "method_scope": method})],
        )
        pg.execute(
            "UPDATE pmmc_evaluation.evaluation_run_states SET status='completed', completed_at=now(),"
            " succeeded_count=target_count,metadata=$1",
            [
                json.dumps(
                    {
                        "completion_report": {**report, "method_scope": method},
                        "completion_artifact": artifact,
                    }
                )
            ],
        )

    with pytest.raises(RuntimeError, match="55000: step8_evaluation_primary_scope_invalid"):
        finish("pmmc" if primary else "pmmc_main")
    with pytest.raises(RuntimeError, match="55000: step8_selected_evaluation_scope_invalid"):
        finish("paired_main")
    finish(scope)
