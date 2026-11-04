"""Selected release plans exercise the native DAG and evaluator without a service."""

import ast
import asyncio
import json
import shutil
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation import step8_workflow
from pmmc.experiments import release_execution
from pmmc.experiments.release_reporting import summarize_release_campaign
from pmmc.experiments.release_scope import publish_scope, read_scope, stage_job_ids
from pmmc.experiments.step8_artifacts import write_json
from pmmc.experiments.step8_campaign import (
    build_campaign_jobs,
    load_benchmark_config,
    persist_campaign_job_plan,
)
from pmmc.infrastructure.database.step8_evaluation_scope import (
    _v2_definition,
    scoped_completion_function,
)
from pmmc.infrastructure.settings import Settings
from pmmc.release_plan import normalize_plan, plan_hash
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import (
    ANSWER_BACKBONES,
    BenchmarkCampaignManifest,
    SelectionExpansionManifest,
    build_experiment_matrix,
)
from test_release_reporting import ROOT
from test_step8_campaign import protocol
from test_step8_workflow import answer, cell, facade_fixture


def selected(tmp_path, *, stage="main", variant="wr", **options):
    request = normalize_plan(
        {
            "run_id": "study",
            "stage": stage,
            "benchmarks": ["memlens"],
            "models": [ANSWER_BACKBONES[1]],
            **options,
        }
    )
    digest = plan_hash(request)
    proto = protocol()
    matrix = build_experiment_matrix(proto, request, variant=variant)
    data = DatasetName.MEMLENS
    manifest = BenchmarkCampaignManifest(
        campaign_id="campaign",
        phase="exploration",
        protocol=proto,
        partition_id="split",
        snapshot_ids={data: "snapshot"},
        created_at=datetime.now(UTC),
        matrix_hash=canonical_content_hash([c.model_dump(mode="json") for c in matrix]),
        release_plan_hash=digest,
    )
    selection = SelectionExpansionManifest(
        campaign_id="campaign",
        phase="exploration",
        selection="dev-all",
        partition_hash=proto.partition_hash,
        query_ids={data: ["query"]},
        added_query_ids={data: ["query"]},
        stratum_counts={data: {"s": 1}},
        eligible_counts={data: 1},
        release_plan_hash=digest,
    )
    jobs = build_campaign_jobs(
        manifest=manifest,
        selection=selection,
        conversations={data: {"c": ["query"]}},
        project_root=ROOT,
        request=request,
        variant=variant,
    )
    config = load_benchmark_config(ROOT / "configs/experiments/step8-benchmark.yaml").model_copy(
        update={"metadata": {"release_request": request, "exploration_variant": variant}}
    )
    for name, obj in [
        ("config", config),
        ("campaign", manifest),
        (f"selections/{selection.content_hash}", selection),
    ]:
        write_json(tmp_path / "manifests" / f"{name}.json", obj.model_dump(mode="json"))
    persist_campaign_job_plan(directory=tmp_path, selection=selection, jobs=jobs)
    return request, selection, jobs


@pytest.mark.parametrize(
    "method", ["full_context_captions", "caption_text_rag", "native_multimodal_rag"]
)
def test_baseline_only_is_one_real_arm(tmp_path, method):
    _, selection, jobs = selected(tmp_path, stage="baselines", methods=[method])
    expected = {"snapshot", "suite", "baseline_answer", "baseline_grounding", "baseline_evaluate"}
    if method.endswith("rag"):
        expected.update({"corpus", "baseline_index", "baseline_retrieve"})
    assert {j.operation for j in jobs} == expected
    assert {j.parameters["method"] for j in jobs if "method" in j.parameters} == {method}
    assert {j.parameters.get("answer_model") for j in jobs} == {None, ANSWER_BACKBONES[1]}
    assert stage_job_ids(jobs, "baselines", paid=False) == {j.job_id for j in jobs}
    assert all(
        j.release_stages == ("baselines",) and j.parameters["dataset"] == "memlens" for j in jobs
    )
    report = summarize_release_campaign(tmp_path, jobs=[], scope="baselines")
    (group,) = report["groups"]
    assert group["answer_model"] == ANSWER_BACKBONES[1] and len(group["cells"]) == 1
    assert (
        group["cells"][0]["method"] == method and report["selection_hash"] == selection.content_hash
    )


def test_main_selected_non9b_uses_only_native_fallback(tmp_path):
    _, _, jobs = selected(tmp_path)
    ops = {j.operation for j in jobs}
    assert {
        "query_answer",
        "native_fallback_generate",
        "native_fallback_ground",
        "primary_evaluate",
    } <= ops
    assert not ops & {"baseline_answer", "baseline_grounding", "evaluate", "analysis_evaluate"}
    assert all(j.parameters.get("answer_model") in (None, ANSWER_BACKBONES[1]) for j in jobs)
    assert all(
        j.parameters.get("method", "native_multimodal_rag") == "native_multimodal_rag" for j in jobs
    )
    assert all(j.parameters.get("fallback_only") for j in jobs if j.operation == "suite")


def test_components_and_paid_boundary(tmp_path):
    _, _, jobs = selected(
        tmp_path,
        stage="baselines",
        methods=["caption_text_rag"],
        models=[ANSWER_BACKBONES[2]],
        components=["answer"],
    )
    ops = {j.operation for j in jobs}
    assert ops == {
        "snapshot",
        "suite",
        "corpus",
        "baseline_index",
        "baseline_retrieve",
        "baseline_answer",
    }
    unpaid = stage_job_ids(jobs, "baselines", paid=False)
    assert unpaid == {j.job_id for j in jobs if j.operation != "baseline_answer"}
    reports = summarize_release_campaign(tmp_path, jobs=[], scope="baselines")
    assert reports["groups"][0]["cells"] == []
    assert reports["groups"][0]["query_evaluation_states"] == {"not_requested": 1}


def test_worker_rechecks_immutable_request_and_exact_scope(tmp_path, monkeypatch):
    _, selection, jobs = selected(tmp_path, components=["compile"])
    assert "query_answer" not in {j.operation for j in jobs}
    digest, allowed = publish_scope(
        tmp_path, "campaign", selection.content_hash, "main", jobs, paid=False
    )
    monkeypatch.setenv("PMMC_RELEASE_REQUEST", "not trusted by this worker")
    assert set(read_scope(tmp_path, digest, "campaign", selection.content_hash)) == allowed
    payload = json.loads((tmp_path / "private/scopes" / f"{digest}.json").read_bytes())
    payload["job_ids"] = payload["job_ids"][:-1]
    forged = canonical_content_hash(payload)
    write_json(tmp_path / "private/scopes" / f"{forged}.json", payload)
    with pytest.raises(ValueError, match="outside_request"):
        read_scope(tmp_path, forged, "campaign", selection.content_hash)
    path = tmp_path / "manifests/config.json"
    config = json.loads(path.read_bytes())
    config["metadata"]["release_request"]["components"] = ["answer"]
    write_json(path, config)
    with pytest.raises(ValueError, match="request_identity_mismatch"):
        read_scope(tmp_path, digest, "campaign", selection.content_hash)


def test_completed_open_baseline_seals_without_legacy_stage_or_budget(tmp_path, monkeypatch):
    request, selection, jobs = selected(tmp_path, stage="baselines", methods=["caption_text_rag"])
    settings = Settings(_env_file=None, project_root=tmp_path)
    config = load_benchmark_config(tmp_path / "manifests/config.json")
    profile = settings.resolve_path(config.runtime_profile)
    profile.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / config.runtime_profile, profile)
    rows = [dict(job=j.model_dump(mode="json"), job_id=j.job_id, status="completed") for j in jobs]
    monkeypatch.delenv("PMMC_STEP8_BUDGET_APPROVAL_ID", raising=False)
    monkeypatch.setattr(release_execution, "_register", AsyncMock(return_value=(None, {})))
    monkeypatch.setattr(release_execution, "_snapshot", AsyncMock(return_value=rows))
    monkeypatch.setattr(
        release_execution,
        "ProductionComponentBackend",
        lambda **_: SimpleNamespace(runtime_settings=object()),
    )
    monkeypatch.setattr(
        release_execution, "supervise_campaign", lambda **_: {"physical_startups": {}}
    )
    monkeypatch.setattr(
        "pmmc.experiments.release_reporting.summarize_release_campaign",
        lambda *_, **kw: dict(
            campaign_id=selection.campaign_id,
            selection_hash=selection.content_hash,
            scope=kw["scope"],
            paper_result=False,
            plan_hash=plan_hash(request),
        ),
    )
    report = release_execution.execute_stage(settings, tmp_path, "baselines", "control")
    seal = json.loads((tmp_path / "reports/seal-baselines.json").read_bytes())
    assert report["state"] == "stage_complete" and not report["paid_authorized"]
    assert report["scoped_jobs"] == len(jobs) == report["completed"]
    assert (
        seal["scope_hash"] == report["scope_hash"]
        and seal["selection_hash"] == selection.content_hash
    )


def test_no_analysis_grid_without_ablation_or_source_permission(tmp_path):
    _, _, jobs = selected(
        tmp_path, stage="all", analyses=["coverage"], methods=["caption_text_rag"]
    )
    assert {s for j in jobs for s in j.release_stages} == {"main", "baselines"}
    with pytest.raises(ValueError, match="without_deps"):
        selected(tmp_path / "source", stage="analysis", analyses=["coverage"], from_run="previous")


def test_baseline_receipt_uses_real_child_not_pmmc(monkeypatch):
    method = "full_context_captions"
    value = answer().model_copy(update={"method": method, "answer_variant": "primary"})
    fixture = facade_fixture(monkeypatch, [cell(value, method=method)])
    result = fixture.run(method_scope="baseline")
    assert result["query_executor_run_id"] is None and result["baseline_run_id"] == "run"
    assert result["selected_cell_count"] == 1 and not result["all_methods_evaluated"]
    assert (
        result["scores"][0]["method"] == method
        and result["qbank_semantic"]["status"] == "not_evaluated"
    )


def test_standalone_loader_checks_child_and_never_queries_pmmc(monkeypatch):
    method = "full_context_captions"
    suite = SimpleNamespace(
        suite_id="suite",
        methods=[method],
        dataset="memlens",
        snapshot_id="snapshot",
        answer_model_alias=ANSWER_BACKBONES[1],
        metadata_json=dict(
            standalone_method=method,
            partition_role="development",
            paper_result=False,
            answer_contract_version="typed-answer-contract-v3",
            method_protocol_hash="protocol",
            release_plan_hash="a" * 64,
            release_campaign_id="campaign",
        ),
    )
    campaign = SimpleNamespace(
        campaign_id="campaign",
        phase="exploration",
        protocol_hash="protocol",
        payload=dict(
            release_plan_hash="a" * 64,
            snapshot_ids={"memlens": "snapshot"},
            protocol={"judge_alias": "judge"},
        ),
    )
    child = SimpleNamespace(run_id="run")
    state = SimpleNamespace(status="completed")
    actual = answer(method=method, variant="primary")
    payload = actual.model_dump(mode="json")
    row = SimpleNamespace(
        typed_answer_id=actual.typed_answer_id,
        content_hash=canonical_content_hash(payload),
        payload={"metadata": {"typed_result": payload}},
        query_executor_run_id="run",
        query_id="query",
        terminal_state=actual.terminal_state,
        answer_model_call_id=actual.answer_model_call_id,
    )
    definition = SimpleNamespace(
        cell_id="cell",
        method=method,
        payload=dict(
            method=method, treatment="main", question_budget=8, round_id=2, source_variant="active"
        ),
    )
    statements = []
    results = iter([["query"], ["query"], [definition], [row]])

    async def scalars(stmt):
        statements.append(str(stmt))
        return SimpleNamespace(all=lambda: next(results))

    session = SimpleNamespace(
        scalar=AsyncMock(return_value=child), get=AsyncMock(return_value=state), scalars=scalars
    )
    monkeypatch.setattr(
        step8_workflow, "PostgresRuntimeQueryRepository", lambda _: SimpleNamespace(get=AsyncMock())
    )
    loaded = asyncio.run(
        step8_workflow._load_standalone_inputs(None, session, campaign, suite, "run", "query")
    )
    assert loaded.cells[0].answer == actual and loaded.cells[0].source_run_id == "run"
    assert all(
        "query_executor_runs" not in stmt and "query_experiment_cell_results" not in stmt
        for stmt in statements
    )
    child.run_id = "wrong"
    with pytest.raises(ValueError, match="child_incomplete"):
        asyncio.run(
            step8_workflow._load_standalone_inputs(None, session, campaign, suite, "run", "query")
        )


def test_sql_upgrade_from_base_and_v2_is_idempotent_and_fail_closed():
    source = next((ROOT / "alembic/versions").glob("*0032*")).read_text()
    base = next(
        n.value
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.startswith(
            "CREATE FUNCTION pmmc_evaluation.enforce_step8_evaluation_completion"
        )
    )
    sql = scoped_completion_function(base)
    assert (
        sql == scoped_completion_function(_v2_definition(base)) == scoped_completion_function(sql)
    )
    assert "b.method=evaluation.metadata->>'standalone_method'" in sql
    assert (
        "step8_evaluation_judge_lineage_invalid" in sql
        and "step8_evaluation_source_lineage_invalid" in sql
    )
    with pytest.raises(ValueError, match="drift"):
        scoped_completion_function(sql.replace("u.methods[1]=b.method", "true", 1))


def test_recovered_worker_rejects_output_not_bound_to_original_blob(tmp_path):
    from pmmc.experiments.step8_artifacts import CampaignArtifactStore
    from pmmc.experiments.step8_runner import CampaignWorker
    from pmmc.schemas.step8 import Step8ExecutionConfig
    from test_step8_campaign import FakeRepository

    _, _, jobs = selected(tmp_path, stage="baselines", methods=["full_context_captions"])
    (snapshot,) = (j for j in jobs if j.operation == "snapshot")
    db = FakeRepository(snapshot)
    store = CampaignArtifactStore(tmp_path)
    original = {"snapshot_id": "snapshot"}
    backend = SimpleNamespace(
        preflight=AsyncMock(),
        execute=AsyncMock(
            return_value=(original, [store.put_json(original)], {"component_outputs": 1})
        ),
    )

    def run():
        return asyncio.run(
            CampaignWorker(
                repository=db,
                backend=backend,
                campaign_id=snapshot.campaign_id,
                campaign_dir=tmp_path,
                resource=snapshot.resource_class,
                execution=Step8ExecutionConfig(cpu_workers=1),
                execution_image="fixture",
                artifact_store=store,
                event=lambda _: None,
            ).run()
        )

    assert run()["completed_this_attempt"] == 1
    db.state = "pending"
    assert run()["reused"] == 1
    path = tmp_path / "checkpoints/components" / f"{snapshot.job_id}.json"
    saved = json.loads(path.read_bytes())
    write_json(path, {**saved, "output": {"snapshot_id": "different"}})
    db.state = "pending"
    with pytest.raises(ExceptionGroup) as errors:
        run()
    assert any(
        isinstance(exc, ValueError) and str(exc) == "incompatible_component_recovery_receipt"
        for exc in errors.value.exceptions
    )
    assert db.state == "failed"
    backend.execute.assert_awaited_once()
