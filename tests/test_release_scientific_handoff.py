"""Rendered K8s intent -> native artifacts -> CPU reader. No service is contacted."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from pmmc.benchmarks.canonical import iter_jsonl
from pmmc.experiments import release_execution
from pmmc.experiments.release_inputs import prepare_exploration
from pmmc.experiments.release_models import selected_release_model_aliases
from pmmc.experiments.release_reporting import summarize_release_campaign
from pmmc.experiments.release_scope import publish_scope, read_scope
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_campaign import load_benchmark_config
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.experiments.step8_supervisor import ready_jobs
from pmmc.release_plan import normalize_plan, plan_hash
from pmmc.schemas import BenchmarkQuery
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, BenchmarkCampaignManifest, ExperimentCell
from scripts.release_host.backends import Kubernetes
from scripts.release_host.config import ROOT as ROOT  # isolated image-lock fixture
from scripts.release_host.runner import launch
from test_release_host import Simulated
from test_release_inputs import project  # noqa: F401 - fixture
from test_release_kubernetes import config
from test_step8_workflow import answer, cell, facade_fixture


def container(cfg, request, role):
    intent = dict(kind="analysis-report" if role == "analysis-report" else "run",
                  variant="wr", plan=request)
    host = Kubernetes(cfg, cfg["instance"], None, {}, None, intent)
    job = host.manifest("fixture", role, request["run_id"], request["stage"])
    return job["spec"]["template"]["spec"]["containers"][0]


def public_env(c):
    return {row["name"]: row["value"] for row in c["env"] if "value" in row}


def score(project, directory, manifest, job, monkeypatch):
    p = job.parameters
    query = next(row.query for row in iter_jsonl(project.benchmark_artifacts_path /
                 p["dataset"] / p["snapshot_id"] / "queries.jsonl", BenchmarkQuery)
                 if row.query.query_id == p["query_id"])
    (definition,) = [ExperimentCell.model_validate(row) for row in json.loads(
        (directory / "manifests/matrix.json").read_bytes())]
    typed = answer("abstention", method=definition.method, variant="primary").model_copy(
        update={"query_id":query.query_id, "snapshot_id":p["snapshot_id"]})
    standalone = job.operation == "baseline_evaluate"
    binding = replace(cell(typed, identifier=definition.cell_id, method=definition.method),
                      source_fact_id=typed.typed_answer_id if standalone else "synthetic-cell-result",
                      source_variant="primary" if standalone else "active", question_budget=8, round_id=2)
    fixture = facade_fixture(monkeypatch, [binding], query=query, snapshot_id=p["snapshot_id"],
                             protocol_hash=manifest.protocol.content_hash)
    scope = "baseline" if standalone else "pmmc_main"
    result = fixture.run(campaign_id=manifest.campaign_id, method_scope=scope)
    before = len(fixture.state["artifacts"])
    assert fixture.run(campaign_id=manifest.campaign_id, method_scope=scope) == result
    assert len(fixture.state["artifacts"]) == before
    return result


@pytest.mark.usefixtures("installed_host_lock")
@pytest.mark.parametrize("stage", ["baselines", "main"])
def test_k8s_native_handoff_and_source_reader(tmp_path, project, monkeypatch, stage):
    cfg = config(tmp_path, gpus=1)
    request = normalize_plan(dict(run_id="science", stage=stage, benchmarks=["memlens"],
                             models=["answer-qwen3.5-9b"], **(
                             {"methods":["caption_text_rag"]} if stage == "baselines" else {})))
    # Prepare and GPU/control see the same canonical intent, not an implicit main stage.
    for role in ("prepare", "models", "control", "gpu"):
        c = container(cfg, request, role)
        env = public_env(c)
        assert json.loads(env["PMMC_RELEASE_REQUEST"]) == request
        assert env["PMMC_RELEASE_STAGE"] == stage
        assert not any(r["name"] in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"} for r in c["env"])
    monkeypatch.setenv("PMMC_RELEASE_REQUEST", env["PMMC_RELEASE_REQUEST"])
    monkeypatch.setenv("PMMC_RELEASE_STAGE", stage)
    project.memgallery_path.rename(project.memgallery_path.with_name("not-mounted"))
    ptr = prepare_exploration(project, "science")
    directory = project.project_root / ptr["campaign_dir"]
    config_file = load_benchmark_config(directory / "manifests/config.json")
    assert config_file.release_request == request and ptr["selected"] == {"memlens":1}
    assert ptr["release_plan_hash"] == plan_hash(request)
    aliases = selected_release_model_aliases(project, config_file)
    if stage == "baselines":
        assert set(aliases) == {"answer-qwen3.5-9b", "grounding-qwen3.5-9b", "judge-qwen3.5-9b",
                                "qwen3-embedding-8b"}
    manifest = BenchmarkCampaignManifest.model_validate_json((directory/"manifests/campaign.json").read_bytes())
    ids = json.loads((directory/"manifests"/f"jobs-{ptr['selection_hash']}.json").read_bytes())
    jobs = [BenchmarkStageJob.model_validate_json((directory/"private/jobs"/f"{key}.json").read_bytes()) for key in ids]
    digest, scoped = publish_scope(directory, manifest.campaign_id, ptr["selection_hash"], stage, jobs, False)
    assert scoped == set(ids)
    # Interrupted publication of a run pointer must recover the same native inventory.
    (project.project_root / ptr["pointer_path"]).rename(directory/"private/interrupted-pointer.json")
    assert prepare_exploration(project, "science") == ptr
    assert set(read_scope(directory, digest, manifest.campaign_id, ptr["selection_hash"])) == scoped

    rows = {j.job_id:dict(job_id=j.job_id,job=j.model_dump(mode="json"),status="pending",receipt_hash=None)
            for j in jobs}
    store = CampaignArtifactStore(directory.parent)
    while frontier := ready_jobs(list(rows.values())):
        for job in frontier:
            output = (score(project,directory,manifest,job,monkeypatch) if job.stage == "evaluation"
                      else {"fixture_operation":job.operation})
            receipt = ComponentReceipt(job_id=job.job_id, scientific_input_hash=job.scientific_input_hash,
                source_fingerprint=job.source_fingerprint, schema_revision=STEP8_ALEMBIC_REVISION,
                upstream_receipt_hashes={p:rows[p]["receipt_hash"] for p in job.dependency_ids},
                output=output, artifact_hashes=[store.put_json(output)], row_counts={"component_outputs":1})
            write_json(directory/"checkpoints/components"/f"{job.job_id}.json",receipt.model_dump(mode="json"),
                       immutable=True)
            rows[job.job_id].update(status="completed", receipt_hash=receipt.content_hash)
    metrics = summarize_release_campaign(directory,jobs=list(rows.values()),scope=stage)
    group, = metrics["groups"]
    observation, = group["cells"]
    assert observation["method"] == ("caption_text_rag" if stage=="baselines" else "pmmc")
    assert observation["evaluation_job_states"] == {"completed":1}
    monkeypatch.delenv("PMMC_STEP8_BUDGET_APPROVAL_ID", raising=False)
    monkeypatch.setattr(release_execution, "_register", AsyncMock(return_value=(None, {})))
    monkeypatch.setattr(release_execution, "_snapshot", AsyncMock(return_value=list(rows.values())))
    monkeypatch.setattr(release_execution, "supervise_campaign", lambda **_: {"physical_startups":{}})
    sim = Simulated()
    def outcome(job, role):
        if role == "prepare":
            return ptr
        if role in {"control","gpu"}:
            return release_execution.execute_stage(project, directory, stage, role)
        return "complete"
    sim.outcome = outcome
    host_state = tmp_path/"host-state"
    assert launch(host_state,cfg,"science",request,factory=sim.factory,sleep=lambda _:None)==0
    assert [r[0] for r in sim.started] == [
        "iam","bootstrap","assets","prepare","models","pointer","control","gpu"]
    before = list(sim.started)
    assert launch(host_state,cfg,"science",request,factory=sim.factory,resume=True,sleep=lambda _:None)==0
    assert sim.started == before
    if stage=="main":
        reader_plan=normalize_plan(dict(run_id="readout",stage="analysis",from_run="science",
            benchmarks=["memlens"],models=request["models"],analyses=["coverage","cost","failures"]))
        reader=container(cfg,reader_plan,"analysis-report")
        assert not any("valueFrom" in row for row in reader["env"])
        assert "nvidia.com/gpu" not in reader["resources"]["limits"]
        assert all(m["readOnly"] for m in reader["volumeMounts"] if m["mountPath"]=="/workspace/outputs")
        argv=reader["command"]
        assert argv[:2]==["python","-m"]
        child_env={"PATH":os.defpath, **public_env(reader),
                   "PMMC_PROJECT_ROOT":str(project.project_root), "PMMC_BENCHMARK_ASSET_ROOT":"/not-mounted",
                   "PMMC_MODEL_CACHE_DIR":"/not-mounted", "PYTHONPATH":"src"}
        proc=subprocess.run([sys.executable,*argv[1:]],cwd=project.project_root,env=child_env,
                            text=True,capture_output=True,timeout=45,check=True)
        result=json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["state"]=="complete" and len(result["sources"])==1
        out=project.outputs_path/"step8/exploration/readout/reports/analyses"/f"{result['sha256']}.json"
        assert json.loads(out.read_bytes())["sections"]["coverage"][0]["selected"]==1
        assert not (project.outputs_path/"step8/exploration/readout/wr").exists()
        reader_sim=Simulated()
        reader_sim.reconcile=lambda: pytest.fail("CPU reader reconciled worker services")
        reader_sim.outcome=lambda job,role: result if role=="analysis-report" else pytest.fail(role)
        assert launch(host_state,cfg,"readout",reader_plan,factory=reader_sim.factory,sleep=lambda _:None)==0
        assert [r[0] for r in reader_sim.started] == ["analysis-report"]
