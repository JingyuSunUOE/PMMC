"""Rendered K8s intent -> native artifacts -> CPU reader. No service is contacted."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from scripts.release_host.backends import Kubernetes
from scripts.release_host.backends import receipt as host_receipt
from scripts.release_host.config import ROOT as ROOT  # isolated image-lock fixture
from scripts.release_host.runner import launch

from pmmc import release_asset_receipt
from pmmc.baselines.suite import baseline_suite_child_run_id
from pmmc.benchmarks import release_data
from pmmc.benchmarks.canonical import iter_jsonl
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments import release_execution
from pmmc.experiments.release_inputs import prepare_exploration
from pmmc.experiments.release_models import selected_release_model_aliases
from pmmc.experiments.release_reporting import summarize_release_campaign
from pmmc.experiments.release_scope import publish_scope, read_scope
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_backend import ProductionComponentBackend
from pmmc.experiments.step8_campaign import load_benchmark_config
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_supervisor import ready_jobs
from pmmc.release_plan import normalize_plan, plan_hash
from pmmc.schemas import BaselineMethod, BenchmarkQuery
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, BenchmarkCampaignManifest, ExperimentCell
from test_release_host import Simulated
from test_release_inputs import project as project  # pytest discovers this shared fixture
from test_release_kubernetes import config
from test_step8_workflow import answer, cell, facade_fixture


def container(cfg, request, role):
    intent = dict(
        kind="analysis-report" if role == "analysis-report" else "run", variant="wr", plan=request
    )
    host = Kubernetes(cfg, cfg["instance"], None, {}, None, intent)
    job = host.manifest("fixture", role, request["run_id"], request["stage"])
    return job["spec"]["template"]["spec"]["containers"][0]


def public_env(c):
    return {row["name"]: row["value"] for row in c["env"] if "value" in row}


def verified_fixture_assets(project, request, tmp_path, monkeypatch):
    # /pmmc-stage and /pmmc-inputs are the writable and read-only mounts of the
    # same data. Verify the fixture's already installed tree without fetching it.
    rows = [
        (p.relative_to(project.memlens_path).as_posix(), p.stat().st_size,
         hashlib.sha256(p.read_bytes()).hexdigest())
        for p in project.memlens_path.rglob("*") if p.is_file()
    ]
    lock = tmp_path / "fixture-benchmarks.lock.json"
    write_json(lock, {"protocol": "pmmc-public-data-lock-v1", "datasets": {"memlens": {
        "file_count": len(rows), "materialized_sha256": release_data._tree_digest(rows),
        "files": [{"target": name, "size": size, "sha256": sha,
                   "url": f"https://huggingface.co/datasets/fixture/resolve/test/{name}"}
                  for name, size, sha in rows],
    }}})
    monkeypatch.setattr(release_asset_receipt, "SOURCE_LOCK", lock)
    monkeypatch.setattr(release_data, "urlopen", lambda *a, **kw: pytest.fail("unexpected download"))
    sources = release_data.prepare_data(project.benchmark_asset_path, request["benchmarks"], lock_path=lock)
    assert sources["benchmarks"]["memlens"]["state"] == "verified"
    raw = {"ready": True, "protocol": "release-assets-v2", "sources": sources,
           "plan_hash": plan_hash(request), "file_counts": {"memlens": len(rows)}}
    release_asset_receipt.validate_asset_receipt(raw, request)
    return host_receipt("assets", json.dumps(raw))


def assert_tampered_seals_rejected(directory, run_reader, readout_root, open_scope_hash):
    seal_path = directory / "reports/seal-main.json"
    seal = json.loads(seal_path.read_bytes())
    scopes = directory / "private/scopes"
    native = json.loads((scopes / f"{seal['scope_hash']}.json").read_bytes())
    assert len(native["job_ids"]) > 1
    # This open-model campaign has the same jobs in both legitimate scopes.
    # Only the full scope can attest completion, even when membership matches.
    assert native["paid"] is True and open_scope_hash != seal["scope_hash"]
    assert json.loads((scopes / f"{open_scope_hash}.json").read_bytes()) == {**native, "paid": False}

    def scope_file(payload, digest=None):
        key = digest or canonical_content_hash(payload)
        write_json(scopes / f"{key}.json", payload, immutable=True)
        return {"scope_hash": key}

    cases = [
        ({"protocol": "unknown"}, "invalid source seal scope"),
        ({"stage": "baselines"}, "invalid source seal scope"),
        ({"scope_hash": "0" * 64}, "source seal does not identify the requested stage"),
        ({"scope_hash": open_scope_hash}, "source seal does not identify the requested stage"),
        (scope_file({**native, "stage": "baselines"}), "source seal does not identify the requested stage"),
        (scope_file(native, canonical_content_hash("not this scope")), "exploration_scope_identity_changed"),
        (scope_file({**native, "selection": "0" * 64}), "exploration_scope_identity_changed"),
        (scope_file({**native, "job_ids": native["job_ids"][1:]}), "release_worker_scope_outside_request"),
    ]
    # The seal itself must authenticate the scope even if timing evidence was not retained.
    observations = directory / "reports/observations/main"
    retained = directory / "private/retained-main-observations"
    observations.rename(retained)
    try:
        for mutation, reason in cases:
            write_json(seal_path, {**seal, **mutation})
            rejected = run_reader()
            assert rejected.returncode != 0, mutation
            assert f"ValueError: {reason}" in rejected.stderr, (mutation, rejected.stderr)
            assert not (readout_root / "reports").exists(), mutation
    finally:
        write_json(seal_path, seal)
        retained.rename(observations)


def score(project, backend, job, parents, monkeypatch):
    directory, manifest = backend.campaign_dir, backend.campaign
    p = job.parameters
    query = next(
        row.query
        for row in iter_jsonl(
            project.benchmark_artifacts_path / p["dataset"] / p["snapshot_id"] / "queries.jsonl",
            BenchmarkQuery,
        )
        if row.query.query_id == p["query_id"]
    )
    (definition,) = [
        ExperimentCell.model_validate(row)
        for row in json.loads((directory / "manifests/matrix.json").read_bytes())
    ]
    standalone = job.operation == "baseline_evaluate"
    run_id = (
        baseline_suite_child_run_id("suite", BaselineMethod(definition.method))
        if standalone
        else "fixture-query-run"
    )
    typed = answer(
        "abstention", method=definition.method, variant="primary", run_id=run_id
    ).model_copy(update={"query_id": query.query_id, "snapshot_id": p["snapshot_id"]})
    binding = replace(
        cell(typed, identifier=definition.cell_id, method=definition.method),
        source_fact_id=typed.typed_answer_id if standalone else "synthetic-cell-result",
        source_run_id=run_id,
        source_variant="primary" if standalone else "active",
        question_budget=8,
        round_id=2,
    )
    fixture = facade_fixture(
        monkeypatch,
        [binding],
        query=query,
        snapshot_id=p["snapshot_id"],
        protocol_hash=manifest.protocol.content_hash,
    )
    parent_path = directory/"checkpoints/components"/f"{job.dependency_ids[0]}.json"
    saved = json.loads(parent_path.read_bytes())
    write_json(parent_path, {**saved,"output":{**saved["output"],"suite_id":"substituted"}})
    with pytest.raises(ValueError,match="selected_component_parent_receipt_changed"):
        asyncio.run(backend.execute(job,parents))
    fixture.register.assert_not_awaited()
    write_json(parent_path,saved)
    result, hashes, counts = asyncio.run(backend.execute(job, parents))
    assert counts == {"component_outputs": 1} and hashes == [backend.store.put_json(result)]
    assert result["baseline_run_id" if standalone else "query_executor_run_id"] == run_id
    before = len(fixture.state["artifacts"])
    assert asyncio.run(backend.execute(job, parents)) == (result, hashes, counts)
    assert len(fixture.state["artifacts"]) == before
    return result


@pytest.mark.usefixtures("installed_host_lock")
@pytest.mark.parametrize("stage", ["baselines", "main"])
def test_k8s_native_handoff_and_source_reader(tmp_path, project, monkeypatch, stage):
    cfg = config(tmp_path, gpus=1)
    request = normalize_plan(
        dict(
            run_id="science",
            stage=stage,
            benchmarks=["memlens"],
            models=["answer-qwen3.5-9b"],
            **({"methods": ["caption_text_rag"]} if stage == "baselines" else {}),
        )
    )
    # Prepare and GPU/control see the same canonical intent, not an implicit main stage.
    for role in ("assets", "prepare", "models", "control", "gpu"):
        c = container(cfg, request, role)
        env = public_env(c)
        assert json.loads(env["PMMC_RELEASE_REQUEST"]) == request
        assert env["PMMC_RELEASE_STAGE"] == stage
        assert not any(r["name"] in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"} for r in c["env"])
        mounts = {m["mountPath"]: m for m in c["volumeMounts"]}
        if role == "assets":
            assert env["PMMC_BENCHMARK_STAGING_ROOT"] == "/pmmc-stage"
            assert set(mounts) == {"/pmmc-stage"} and not mounts["/pmmc-stage"]["readOnly"]
            input_subpath = mounts["/pmmc-stage"]["subPath"]
        elif role == "prepare":
            assert mounts["/pmmc-inputs"]["readOnly"] and mounts["/pmmc-inputs"]["subPath"] == input_subpath
    monkeypatch.setenv("PMMC_RELEASE_REQUEST", env["PMMC_RELEASE_REQUEST"])
    monkeypatch.setenv("PMMC_RELEASE_STAGE", stage)
    project.memgallery_path.rename(project.memgallery_path.with_name("not-mounted"))
    assets = verified_fixture_assets(project, request, tmp_path, monkeypatch)
    ptr = prepare_exploration(project, "science")
    directory = project.project_root / ptr["campaign_dir"]
    config_file = load_benchmark_config(directory / "manifests/config.json")
    assert config_file.release_request == request and ptr["selected"] == {"memlens": 1}
    assert ptr["release_plan_hash"] == plan_hash(request)
    assert ptr["gpu_stages"] == [stage]
    aliases = selected_release_model_aliases(project, config_file)
    if stage == "baselines":
        assert set(aliases) == {
            "answer-qwen3.5-9b",
            "grounding-qwen3.5-9b",
            "judge-qwen3.5-9b",
            "qwen3-embedding-8b",
        }
    manifest = BenchmarkCampaignManifest.model_validate_json(
        (directory / "manifests/campaign.json").read_bytes()
    )
    ids = json.loads((directory / "manifests" / f"jobs-{ptr['selection_hash']}.json").read_bytes())
    jobs = [
        BenchmarkStageJob.model_validate_json(
            (directory / "private/jobs" / f"{key}.json").read_bytes()
        )
        for key in ids
    ]
    digest, scoped = publish_scope(
        directory, manifest.campaign_id, ptr["selection_hash"], stage, jobs, False
    )
    assert scoped == set(ids)
    # Interrupted publication of a run pointer must recover the same native inventory.
    (project.project_root / ptr["pointer_path"]).rename(
        directory / "private/interrupted-pointer.json"
    )
    assert prepare_exploration(project, "science") == ptr
    assert set(read_scope(directory, digest, manifest.campaign_id, ptr["selection_hash"])) == scoped

    rows = {
        j.job_id: dict(
            job_id=j.job_id, job=j.model_dump(mode="json"), status="pending", receipt_hash=None
        )
        for j in jobs
    }
    store = CampaignArtifactStore(directory.parent)
    backend = ProductionComponentBackend(
        settings=project, config=config_file, campaign=manifest, campaign_dir=directory, store=store
    )
    while frontier := ready_jobs(list(rows.values())):
        for job in frontier:
            parents = {p: rows[p]["receipt_hash"] for p in job.dependency_ids}
            if job.stage == "evaluation":
                output = score(project, backend, job, parents, monkeypatch)
            elif job.operation == "baseline_grounding":
                output = {"suite_id": "suite", "answer_phase": "ground"}
            elif job.operation == "query_grounding":
                output = {
                    "suite_id": "suite",
                    "query_executor_run_id": "fixture-query-run",
                    "phase": "select",
                }
            else:
                output = {"fixture_operation": job.operation}
            receipt = ComponentReceipt(
                job_id=job.job_id,
                scientific_input_hash=job.scientific_input_hash,
                source_fingerprint=job.source_fingerprint,
                schema_revision=STEP8_ALEMBIC_REVISION,
                upstream_receipt_hashes=parents,
                output=output,
                artifact_hashes=[store.put_json(output)],
                row_counts={"component_outputs": 1},
            )
            write_json(
                directory / "checkpoints/components" / f"{job.job_id}.json",
                receipt.model_dump(mode="json"),
                immutable=True,
            )
            rows[job.job_id].update(status="completed", receipt_hash=receipt.content_hash)
    metrics = summarize_release_campaign(directory, jobs=list(rows.values()), scope=stage)
    (group,) = metrics["groups"]
    (observation,) = group["cells"]
    assert observation["method"] == ("caption_text_rag" if stage == "baselines" else "pmmc")
    assert observation["evaluation_job_states"] == {"completed": 1}
    monkeypatch.delenv("PMMC_STEP8_BUDGET_APPROVAL_ID", raising=False)
    monkeypatch.setattr(release_execution, "_register", AsyncMock(return_value=(None, {})))
    monkeypatch.setattr(release_execution, "_snapshot", AsyncMock(return_value=list(rows.values())))
    monkeypatch.setattr(
        release_execution, "supervise_campaign", lambda **_: {"physical_startups": {}}
    )
    sim = Simulated()

    def outcome(job, role):
        if role == "assets":
            return assets
        if role == "prepare":
            return host_receipt("prepare", json.dumps(ptr))
        if role in {"control", "gpu"}:
            return release_execution.execute_stage(project, directory, stage, role)
        return Simulated.outcome(sim, job, role)

    sim.outcome = outcome
    host_state = tmp_path / "host-state"
    assert (
        launch(host_state, cfg, "science", request, factory=sim.factory, sleep=lambda _: None) == 0
    )
    assert [r[0] for r in sim.started] == [
        "iam",
        "bootstrap",
        "assets",
        "prepare",
        "models",
        "pointer",
        "control",
        "gpu",
    ]
    before = list(sim.started)
    assert (
        launch(
            host_state,
            cfg,
            "science",
            request,
            factory=sim.factory,
            resume=True,
            sleep=lambda _: None,
        )
        == 0
    )
    assert sim.started == before
    if stage == "main":
        reader_plan = normalize_plan(
            dict(
                run_id="readout",
                stage="analysis",
                from_run="science",
                benchmarks=["memlens"],
                models=request["models"],
                analyses=["coverage", "cost", "failures"],
            )
        )
        reader = container(cfg, reader_plan, "analysis-report")
        assert not any("valueFrom" in row for row in reader["env"])
        assert "nvidia.com/gpu" not in reader["resources"]["limits"]
        assert all(
            m["readOnly"] for m in reader["volumeMounts"] if m["mountPath"] == "/workspace/outputs"
        )
        argv = reader["command"]
        assert argv[:2] == ["python", "-m"]
        child_env = {
            "PATH": os.defpath,
            **public_env(reader),
            "PMMC_PROJECT_ROOT": str(project.project_root),
            "PMMC_BENCHMARK_ASSET_ROOT": "/not-mounted",
            "PMMC_MODEL_CACHE_DIR": "/not-mounted",
            "PYTHONPATH": "src",
        }
        def run_reader():
            return subprocess.run(
                [sys.executable, *argv[1:]],
                cwd=project.project_root,
                env=child_env,
                text=True,
                capture_output=True,
                timeout=45,
            )

        readout_root = project.outputs_path / "step8/exploration/readout"
        assert_tampered_seals_rejected(directory, run_reader, readout_root, open_scope_hash=digest)
        proc = run_reader()
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["state"] == "complete" and len(result["sources"]) == 1
        assert result["sources"][0]["run_id"] == "science"
        assert result["sources"][0]["selection_hash"] == ptr["selection_hash"]
        out = (
            project.outputs_path
            / "step8/exploration/readout/reports/analyses"
            / f"{result['sha256']}.json"
        )
        assert json.loads(out.read_bytes())["sections"]["coverage"][0]["selected"] == 1
        assert not (project.outputs_path / "step8/exploration/readout/wr").exists()
        reader_sim = Simulated()
        reader_sim.reconcile = lambda: pytest.fail("CPU reader reconciled worker services")
        reader_sim.outcome = lambda job, role: (
            result if role == "analysis-report" else pytest.fail(role)
        )
        assert (
            launch(
                host_state,
                cfg,
                "readout",
                reader_plan,
                factory=reader_sim.factory,
                sleep=lambda _: None,
            )
            == 0
        )
        assert [r[0] for r in reader_sim.started] == ["analysis-report"]
