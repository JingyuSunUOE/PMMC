from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_analysis import analyze_campaign, load_campaign_facts
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import (
    ANSWER_BACKBONES,
    STEP8_ALEMBIC_REVISION,
    BenchmarkCampaignManifest,
    CampaignPhase,
    ResourceClass,
    ScientificProtocol,
    SelectionExpansionManifest,
    build_experiment_matrix,
)
from step8_model_fixtures import model_entry_hashes

DATASETS = (DatasetName.MEMLENS, DatasetName.MEMGALLERY)


def test_analysis_identity_binds_postprocessing_without_mutating_stage_identity():
    from pmmc.experiments.step8_analysis import analysis_source_hashes

    before = analysis_source_hashes()
    assert before == analysis_source_hashes()
    assert "evaluation/paper_statistics.py" in before
    assert "experiments/step8_judge_analysis.py" in before
    assert "experiments/step8_artifacts.py" in before
    assert "experiments/step8_r0_cost_dependencies.py" in before
    assert all(len(value) == 64 for value in before.values())
    assert all(not Path(name).is_absolute() for name in before)


def test_relevance_grid_is_executor_treatment_not_native_baseline(tmp_path):
    from pmmc.experiments.step8_analysis import _threshold_axes

    directory, store, *_ = campaign_fixture(tmp_path)
    facts = load_campaign_facts(directory, artifact_store=store)
    for cell in facts.cells:
        axes = {axis: grid for axis, grid, _ in _threshold_axes(cell, facts.manifest.protocol)}
        if cell.method == "pmmc":
            assert len(axes["relevance_grid"]) == 25
            assert (0.1, 0.5) in axes["relevance_grid"]
        else:
            assert axes == {}


def campaign_fixture(tmp_path: Path, *, full=False, fallback=False):
    directory = tmp_path / "campaign"
    store = CampaignArtifactStore(tmp_path)
    protocol = ScientificProtocol(
        model_lock_hash="model-lock",
        model_entry_hashes=model_entry_hashes(),
        partition_hash="partition",
        exposure_manifest_hash="exposure",
        source_fingerprints={"evaluation": "evaluation-source"},
        config_hashes={"config": "hash"},
    )
    cells = build_experiment_matrix(protocol)
    if not full:
        cells = [
            c
            for c in cells
            if c.answer_model == ANSWER_BACKBONES[0]
            and c.treatment.value == "main"
            and c.question_budget == 8
            and c.round_id == 2
            and c.source_variant == "active"
        ]
    manifest = BenchmarkCampaignManifest(
        campaign_id="test-campaign",
        phase=CampaignPhase.DEVELOPMENT,
        protocol=protocol,
        partition_id="partition-id",
        snapshot_ids={d: d.value + "-snapshot" for d in DATASETS},
        matrix_hash=canonical_content_hash([c.model_dump(mode="json") for c in cells]),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    query_ids = {d: ["PRIVATE_QUERY_1", "PRIVATE_QUERY_2", "PRIVATE_QUERY_3"] for d in DATASETS}
    selection = SelectionExpansionManifest(
        campaign_id=manifest.campaign_id,
        phase=manifest.phase,
        selection="regression",
        partition_hash=protocol.partition_hash,
        query_ids=query_ids,
        added_query_ids=query_ids,
        stratum_counts={d: {"all": 3} for d in DATASETS},
        eligible_counts={d: 3 for d in DATASETS},
    )
    write_json(directory / "manifests/campaign.json", manifest.model_dump(mode="json"))
    write_json(directory / "manifests/matrix.json", [c.model_dump(mode="json") for c in cells])
    write_json(
        directory / f"manifests/selections/{selection.content_hash}.json",
        selection.model_dump(mode="json"),
    )
    jobs = []
    for dataset in DATASETS:
        for qindex, qid in enumerate(query_ids[dataset]):
            for model in protocol.answer_backbones:
                treatments = sorted(
                    {
                        c.treatment.value
                        for c in cells
                        if c.dataset == dataset and c.answer_model == model
                    }
                )
                for treatment in treatments:
                    scope = {
                        "dataset": dataset.value,
                        "snapshot_id": manifest.snapshot_ids[dataset],
                        "query_id": qid,
                        "answer_model": model,
                        "treatment": treatment,
                    }
                    job = BenchmarkStageJob(
                        campaign_id=manifest.campaign_id,
                        stage="evaluation",
                        component_key="evaluation-" + canonical_content_hash(scope),
                        resource_class=ResourceClass.GPU_9B,
                        source_fingerprint="evaluation-source",
                        scientific_inputs=scope,
                        operation="evaluate",
                        parameters=scope,
                    )
                    scores = []
                    for cell in cells:
                        if (cell.dataset, cell.answer_model, cell.treatment.value) != (
                            dataset,
                            model,
                            treatment,
                        ):
                            continue
                        answer_id = "answer-" + canonical_content_hash([qid, cell.cell_id])
                        scores.append(
                            {
                                "cell_id": cell.cell_id,
                                "method": cell.method,
                                "variant": (
                                    f"{cell.method}_b{cell.question_budget}_r{cell.round_id}"
                                ),
                                "source_variant": cell.source_variant
                                if cell.method == "pmmc"
                                else "primary",
                                "answer_source_method": "native_multimodal_rag"
                                if fallback and cell.method == "pmmc"
                                else cell.method,
                                "answer_source_variant": "primary"
                                if fallback or cell.method != "pmmc"
                                else "pmmc_b8_r2",
                                "source_fact_id": "fact-" + answer_id,
                                "source_fact_hash": canonical_content_hash(answer_id),
                                "typed_answer_id": answer_id,
                                "terminal_state": "answered",
                                "status": "answered",
                                "applicable": True,
                                "metric": "typed_exact_match",
                                "value": float(cell.method == "pmmc" and qindex < 2),
                                "judge_value": 1.0,
                                "judge_method_blind": True,
                                "engineering_error": None,
                            }
                        )
                    output = {
                        "campaign_id": manifest.campaign_id,
                        "query_id": qid,
                        "status": "complete",
                        "selected_query": {
                            "query_id": qid,
                            "dataset": dataset.value,
                            "cluster_id": "PRIVATE_CLUSTER_" + str(qindex // 2),
                            "cluster_kind": "scenario"
                            if dataset == DatasetName.MEMGALLERY
                            else "conversation",
                            "applicable": True,
                            "strata": [],
                        },
                        "scores": scores,
                    }
                    receipt = ComponentReceipt(
                        job_id=job.job_id,
                        scientific_input_hash=job.scientific_input_hash,
                        source_fingerprint=job.source_fingerprint,
                        schema_revision=STEP8_ALEMBIC_REVISION,
                        upstream_receipt_hashes={},
                        output=output,
                        artifact_hashes=[store.put_json(output)],
                        row_counts={"component_outputs": 1},
                    )
                    write_json(
                        directory / f"private/jobs/{job.job_id}.json", job.model_dump(mode="json")
                    )
                    write_json(
                        directory / f"checkpoints/components/{job.job_id}.json",
                        receipt.model_dump(mode="json"),
                    )
                    jobs.append(job)
    write_json(
        directory / f"manifests/jobs-{selection.content_hash}.json", [j.job_id for j in jobs]
    )
    return directory, store, manifest, selection, cells, jobs


@pytest.mark.parametrize(
    "phase,labels,queries,error",
    [
        ("development", ["regression", "confirmation"], [["r1", "r2"], ["c1"]], None),
        ("development", ["regression", "confirmation"], [["r1"], ["r1", "c1"]], "disjoint"),
        ("development", ["confirmation", "regression"], [["c1"], ["r1"]], "advance"),
        ("development", ["regression", "regression"], [["r1"], ["r2"]], "advance"),
        ("paper", ["5", "20", "100"], [["q1"], ["q1", "q2"], ["q1", "q2", "q3"]], None),
        ("paper", ["5", "20"], [["q1", "q2"], ["q1"]], "removed"),
        ("paper", ["20", "5"], [["q1"], ["q1", "q2"]], "backwards"),
    ],
)
def test_loader_and_inventory_obey_cohort_chain(tmp_path, phase, labels, queries, error):
    from pmmc.experiments.step8_inventory import _selection

    directory, store, manifest, original, _, jobs = campaign_fixture(tmp_path)
    manifest = BenchmarkCampaignManifest.model_validate(
        {
            **manifest.model_dump(mode="json"),
            "phase": phase,
            "paper_result": phase == "paper",
            "protocol_freeze_hash": "approved" if phase == "paper" else None,
        }
    )
    write_json(directory / "manifests/campaign.json", manifest.model_dump(mode="json"))
    (directory / f"manifests/selections/{original.content_hash}.json").unlink()
    selections = []
    for label, ids in zip(labels, queries, strict=True):
        selection = SelectionExpansionManifest.model_validate(
            {
                **original.model_dump(mode="json"),
                "phase": phase,
                "selection": label,
                "parent_selection_hash": selections[-1].content_hash if selections else None,
                "query_ids": {d: ids for d in DATASETS},
                "added_query_ids": {d: ids for d in DATASETS},
            }
        )
        selections.append(selection)
        write_json(
            directory / f"manifests/selections/{selection.content_hash}.json",
            selection.model_dump(mode="json"),
        )
        write_json(
            directory / f"manifests/jobs-{selection.content_hash}.json", [j.job_id for j in jobs]
        )
    if error:
        with pytest.raises(ValueError, match=error):
            load_campaign_facts(directory, artifact_store=store)
        with pytest.raises(ValueError, match=error):
            _selection(directory, manifest)
    else:
        assert _selection(directory, manifest) == selections[-1]
        for selection in selections:
            facts = load_campaign_facts(
                directory, selection_hash=selection.content_hash, artifact_store=store
            )
            assert facts.selection == selection


def replace_output(directory, store, job, mutate):
    path = directory / f"checkpoints/components/{job.job_id}.json"
    receipt = json.loads(path.read_bytes())
    mutate(receipt["output"])
    receipt["artifact_hashes"] = [store.put_json(receipt["output"])]
    write_json(path, receipt)


def test_completed_campaign_full_matrix_bootstrap_holm_macro_and_public_privacy(tmp_path):
    directory, _, _, _, cells, _ = campaign_fixture(tmp_path, full=True)
    report = analyze_campaign(directory)
    assert report["status"] == "complete"
    assert report["analysis"]["bootstrap_replicates"] == 10_000
    assert report["expected_query_cell_count"] == 3 * len(cells)
    assert report["scientific_approval"] is False
    assert report["real_pmmc_viability_supported"] is True
    pmmc = next(s for s in report["cells"] if s["method"] == "pmmc")
    assert pmmc["micro"] == pytest.approx(2 / 3)
    assert pmmc["cluster_macro"] == 0.5
    assert len(report["comparisons"]) == 40
    assert all(
        c["status"] == "complete" and c["holm_p_value"] is not None for c in report["comparisons"]
    )
    assert all(m["status"] == "complete" for m in report["macro"])
    payload = json.dumps(report)
    assert "PRIVATE_QUERY" not in payload and "PRIVATE_CLUSTER" not in payload
    assert analyze_campaign(directory) == report
    assert len(list((directory / "reports").glob("analysis-*.json"))) == 1


def test_missing_query_preserves_denominator_and_blocks_family(tmp_path):
    directory, _, _, _, _, jobs = campaign_fixture(tmp_path)
    missing = jobs[0]
    (directory / f"checkpoints/components/{missing.job_id}.json").unlink()
    report = analyze_campaign(directory)
    assert report["status"] == "incomplete"
    assert report["selected_counts"] == {"memlens": 3, "memgallery": 3}
    summary = next(s for s in report["cells"] if s["dataset"] == missing.parameters["dataset"])
    assert summary["selected_count"] == 3 and summary["engineering_missing_count"] == 1
    assert summary["micro"] is None and summary["cluster_count"] is None
    assert all(c["holm_p_value"] is None for c in report["comparisons"])
    assert report["real_pmmc_viability_supported"] is False


def test_fallback_active_success_is_not_real_pmmc_viability(tmp_path):
    directory, *_ = campaign_fixture(tmp_path, fallback=True)
    report = analyze_campaign(directory, write=False)
    assert report["status"] == "complete"
    assert report["real_pmmc_viability_supported"] is False
    for row in report["cells"]:
        if row["method"] == "pmmc":
            assert row["fallback_answer_count"] == 3
            assert row["true_pmmc_answer_count"] == 0


def test_production_report_includes_frozen_category_and_scenario_macro(tmp_path):
    directory, store, _, _, _, jobs = campaign_fixture(tmp_path)

    def stratify(output):
        selected = output["selected_query"]
        label = "first" if selected["query_id"].endswith(("1", "2")) else "second"
        field = "source_category" if selected["dataset"] == "memlens" else "scenario_id"
        selected["strata"] = [[field, label]]
        for score in output["scores"]:
            score.update(judge_applicable=True, judge_model_call_id="judge-call")

    for job in jobs:
        replace_output(directory, store, job, stratify)
    report = analyze_campaign(directory, write=False)
    strata = report["stratified_quality"]
    assert strata["status"] == "complete"
    assert (
        "experiments/step8_stratified_analysis.py" in report["analysis"]["analysis_source_hashes"]
    )
    pmmc = [
        s for s in strata["cells"] if s["method"] == "pmmc" and s["metric"] == "typed_exact_match"
    ]
    assert {s["stratum_field"] for s in pmmc} == {"source_category", "scenario_id"}
    assert all(s["stratum_macro"] == 0.5 and s["selected_count"] == 3 for s in pmmc)
    assert "PRIVATE" not in json.dumps(strata)


@pytest.mark.parametrize("mutation", ["duplicate", "cluster", "cell"])
def test_conflicting_or_wrong_evaluation_facts_rejected(tmp_path, mutation):
    directory, store, _, _, _, jobs = campaign_fixture(tmp_path, full=mutation == "cluster")

    def mutate(output):
        if mutation == "duplicate":
            output["scores"].append(output["scores"][0])
        elif mutation == "cell":
            output["scores"][0].update(cell_id="wrong")
        else:
            output["selected_query"].update(cluster_id="different")

    replace_output(directory, store, jobs[0], mutate)
    with pytest.raises(ValueError, match="duplicate|conflicting|outside"):
        analyze_campaign(directory, write=False)


@pytest.mark.parametrize("mutation", ["source", "matrix", "cas", "schema"])
def test_frozen_identities_and_actual_cas_are_checked(tmp_path, mutation):
    directory, store, _, _, _, jobs = campaign_fixture(tmp_path)
    receipt_path = directory / f"checkpoints/components/{jobs[0].job_id}.json"
    receipt = json.loads(receipt_path.read_bytes())
    if mutation == "cas":
        store.object_path(receipt["artifact_hashes"][0]).write_bytes(b"corrupt")
        report = analyze_campaign(directory, write=False)
        assert report["status"] == "incomplete"
        assert report["missing_cas_artifact_count"] == 1
    else:
        if mutation in {"source", "schema"}:
            receipt["source_fingerprint" if mutation == "source" else "schema_revision"] = "changed"
            write_json(receipt_path, receipt)
        else:
            write_json(directory / "manifests/matrix.json", [])
        with pytest.raises(ValueError, match="mismatch"):
            analyze_campaign(directory, write=False)


@pytest.mark.parametrize("incomplete_coverage", [False, True])
def test_threshold_grid_only_consumes_saved_context_specific_answers(tmp_path, incomplete_coverage):
    from test_step8_workflow import answer

    directory, store, manifest, selection, cells, jobs = campaign_fixture(tmp_path)
    cell = next(c for c in cells if c.method == "pmmc" and c.dataset == DatasetName.MEMLENS)
    rows = []
    for job in jobs:
        if job.parameters["dataset"] != "memlens":
            continue
        output = json.loads((directory / f"checkpoints/components/{job.job_id}.json").read_bytes())[
            "output"
        ]
        score = next(s for s in output["scores"] if s["cell_id"] == cell.cell_id)
        saved_answer = answer(identifier=score["typed_answer_id"]).model_dump(mode="json")
        saved_answer["query_id"] = output["query_id"]
        saved_answer["grounding"]["query_id"] = output["query_id"]
        rows.append(
            {
                "cell_id": cell.cell_id,
                "query_id": output["query_id"],
                "axis": "match_gap",
                "thresholds": [0.5, 0.0],
                "context_sha256": store.put_json(
                    {
                        "query_id": output["query_id"],
                        "cell_id": cell.cell_id,
                        "axis": "match_gap",
                        "thresholds": [0.5, 0.0],
                        "typed_answer_id": score["typed_answer_id"],
                        "context": "saved-facts",
                    }
                ),
                "answer_sha256": store.put_json(saved_answer),
                "score": score,
            }
        )
    write_json(
        directory / "private/threshold-facts.json",
        {
            "schema_version": "step8-threshold-facts-v1",
            "campaign_id": manifest.campaign_id,
            "selection_hash": selection.content_hash,
            "rows": rows,
            "coverage": [
                {
                    **{key: rows[0][key] for key in ("cell_id", "query_id", "axis", "thresholds")},
                    "status": "incomplete",
                }
            ]
            if incomplete_coverage
            else [],
        },
    )
    report = analyze_campaign(directory)
    scan = next(
        s
        for s in report["threshold_scans"]
        if s["cell_id"] == cell.cell_id
        and s["axis"] == "match_gap"
        and s["thresholds"] == [0.5, 0.0]
    )
    if incomplete_coverage:
        assert scan["status"] == "incomplete"
        assert scan["selected_count"] == 3 and scan["engineering_missing_count"] == 1
        assert scan["micro"] is None
        return
    assert scan["status"] == "complete"
    assert all(
        s["status"] == "incomplete"
        for s in report["threshold_scans"]
        if not s["frozen_point"] and s != scan
    )
    store.object_path(rows[0]["answer_sha256"]).unlink()
    changed = analyze_campaign(directory)
    missing = next(
        s
        for s in changed["threshold_scans"]
        if s["cell_id"] == cell.cell_id
        and s["axis"] == "match_gap"
        and s["thresholds"] == [0.5, 0.0]
    )
    assert missing["status"] == "incomplete" and missing["micro"] is None
    assert missing["selected_count"] == 3 and missing["engineering_missing_count"] == 1
    assert len(list((directory / "reports").glob("analysis-*.json"))) == 2


def test_gold_unavailable_remains_not_applicable_and_abstention_is_not_failure(tmp_path):
    directory, store, _, _, _, jobs = campaign_fixture(tmp_path)
    for job in jobs:

        def mutate(output):
            output["selected_query"]["applicable"] = False
            for row in output["scores"]:
                row.update(applicable=False, value=None, unavailable_reason="gold_missing")

        replace_output(directory, store, job, mutate)
    report = analyze_campaign(directory, write=False)
    assert report["status"] == "complete"
    assert all(s["status"] == "not_applicable" for s in report["cells"])
    assert all(c["p_value"] is None for c in report["comparisons"])


def test_symlinked_campaign_receipt_is_rejected(tmp_path):
    directory, _, _, _, _, jobs = campaign_fixture(tmp_path)
    path = directory / f"checkpoints/components/{jobs[0].job_id}.json"
    target = tmp_path / "elsewhere.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_campaign_facts(directory)


def test_legal_abstention_scores_zero_without_removing_selected_queries(tmp_path):
    directory, store, _, _, _, jobs = campaign_fixture(tmp_path)

    def abstain(output):
        for row in output["scores"]:
            row.update(status="abstention", terminal_state="abstained", value=None)

    for job in jobs:
        replace_output(directory, store, job, abstain)
    report = analyze_campaign(directory, write=False)
    assert report["status"] == "complete"
    assert report["real_pmmc_viability_supported"] is False
    for cell in report["cells"]:
        assert cell["micro"] == 0.0
        assert cell["abstention_count"] == cell["selected_count"] == 3
        assert cell["engineering_missing_count"] == 0


def test_absent_typed_answer_is_engineering_missing_even_with_a_score(tmp_path):
    directory, store, _, _, _, jobs = campaign_fixture(tmp_path)
    replace_output(
        directory, store, jobs[0], lambda output: output["scores"][0].update(typed_answer_id=None)
    )
    report = analyze_campaign(directory, write=False)
    assert report["status"] == "incomplete"
    assert any(c["engineering_missing_count"] == 1 for c in report["cells"])


def test_produced_thresholds_analyze_refusal_missing_cas_and_private_ids(tmp_path, monkeypatch):
    from pmmc.evaluation.paper_statistics import SelectedQuery, StatisticsConfig
    from pmmc.experiments.step8_analysis import _threshold_scans
    from step8_threshold_fixtures import produce_thresholds, threshold_fixture

    case = threshold_fixture(tmp_path, monkeypatch)
    produced = produce_thresholds(case)
    facts = SimpleNamespace(
        directory=case.directory,
        store=case.store,
        selection=case.selection,
        manifest=case.manifest,
        cells=case.cells,
        files={},
        input_hashes={},
    )
    metadata = {
        ("memlens", "query-1"): SelectedQuery(
            "query-1", "memlens", "private-cluster", "conversation"
        )
    }
    scans = _threshold_scans(facts, metadata, {}, StatisticsConfig())
    assert "query-1" not in json.dumps(scans)
    assert "private-cluster" not in json.dumps(scans)
    assert all(
        r["status"] == "complete" and r["selected_count"] == 1
        for r in scans
        if r["axis"] in {"match_gap", "fallback_on_off"}
    )
    assert all(
        r["status"] == "incomplete"
        for r in scans
        if r["axis"].endswith("relevance") and r["thresholds"] != [0.3]
    )
    row = next(r for r in produced["rows"] if r["score"]["answer_source_method"] == "pmmc")
    assert row["score"]["status"] == "abstention" and row["score"]["value"] == 0
    context = case.store.load_json(row["context_sha256"])
    proof = next(iter(context["answer_proof"]["calls"].values()))
    case.store.verify(proof["output_sha256"]).write_bytes(b"corrupt CAS payload")
    scans = _threshold_scans(facts, metadata, {}, StatisticsConfig())
    target = next(
        r
        for r in scans
        if (r["cell_id"], r["axis"], r["thresholds"])
        == (row["cell_id"], row["axis"], row["thresholds"])
    )
    assert target["status"] == "incomplete" and target["engineering_missing_count"] == 1


@pytest.mark.parametrize("coverage_status", [None, "complete", "incomplete"])
@pytest.mark.parametrize("changed_policy", [False, True])
def test_joint_policy_never_reuses_original_scores_without_proof(
    tmp_path, coverage_status, changed_policy
):
    from pmmc.experiments.step8_analysis import selected_policy_axis
    from pmmc.experiments.step8_approval import _POLICY_FIELDS

    directory, _, manifest, selection, cells, _ = campaign_fixture(tmp_path)
    policy = manifest.protocol.model_copy(
        update={"text_relevance_threshold": 0.1} if changed_policy else {}
    )
    coverage = [
        {
            "cell_id": cell.cell_id,
            "query_id": query,
            "axis": "approved_policy",
            "thresholds": list(selected_policy_axis(cell, policy)[2]),
            "status": coverage_status,
        }
        for cell in cells
        if cell.method == "pmmc"
        for query in selection.query_ids[cell.dataset]
        if coverage_status is not None
    ]
    write_json(
        directory / f"private/threshold-facts/{selection.content_hash}.json",
        {
            "schema_version": "step8-threshold-facts-v2",
            "campaign_id": manifest.campaign_id,
            "selection_hash": selection.content_hash,
            "selected_policy": {key: getattr(policy, key) for key in _POLICY_FIELDS},
            "rows": [],
            "coverage": coverage,
        },
    )
    report = analyze_campaign(directory, write=False)
    joint = [row for row in report["threshold_scans"] if row["axis"] == "approved_policy"]
    assert joint and all(
        row["status"] == "incomplete"
        and row["selected_count"] == row["engineering_missing_count"] == 3
        and row["micro"] is None
        for row in joint
    )
    assert all(
        row["status"] == "complete"
        for row in report["threshold_scans"]
        if row["axis"] == "match_gap" and row["frozen_point"]
    )


@pytest.mark.parametrize(
    "mutation", [None, "row", "coverage", "incomplete", "legacy", "context", "answer"]
)
def test_joint_policy_requires_verified_v3_and_complete_coverage(tmp_path, monkeypatch, mutation):
    import asyncio

    from pmmc.evaluation.paper_statistics import SelectedQuery, StatisticsConfig
    from pmmc.experiments import step8_inventory as inv
    from pmmc.experiments.step8_analysis import _threshold_scans
    from step8_threshold_fixtures import relevance_fixture

    case = relevance_fixture(tmp_path, monkeypatch)
    policy = case.manifest.protocol.model_copy(update={"text_relevance_threshold": 0.1})
    asyncio.run(
        inv._populate(
            inv._Builder(case.snapshot, case.store),
            case.directory,
            case.manifest,
            case.settings,
            selected_policy=policy,
        )
    )
    path = case.directory / f"private/threshold-facts/{case.selection.content_hash}.json"
    proof = json.loads(path.read_bytes())
    row = next(
        row
        for row in proof["rows"]
        if row["axis"] == "approved_policy" and row["score"]["source_variant"] == "active"
    )

    def matches(item):
        return all(item[key] == row[key] for key in ("cell_id", "query_id", "axis", "thresholds"))

    if mutation == "row":
        proof["rows"].remove(row)
    elif mutation == "coverage":
        proof["coverage"] = [item for item in proof["coverage"] if not matches(item)]
    elif mutation == "incomplete":
        next(item for item in proof["coverage"] if matches(item))["status"] = "incomplete"
    elif mutation == "legacy":
        context = case.store.load_json(row["context_sha256"])
        context.pop("schema_version")
        row["context_sha256"] = case.store.put_json(context)
    elif mutation in {"context", "answer"}:
        case.store.object_path(row[mutation + "_sha256"]).write_bytes(b"corrupt")
    write_json(path, proof)
    facts = SimpleNamespace(
        directory=case.directory,
        store=case.store,
        selection=case.selection,
        manifest=case.manifest,
        cells=case.cells,
        files={},
        input_hashes={},
    )
    metadata = {
        ("memlens", "query-1"): SelectedQuery("query-1", "memlens", "cluster", "conversation")
    }
    originals = {(score["cell_id"], "query-1"): score for score in case.report["scores"]}
    assert originals[row["cell_id"], "query-1"]["value"] == 1
    scan = next(
        item
        for item in _threshold_scans(facts, metadata, originals, StatisticsConfig())
        if item["cell_id"] == row["cell_id"] and item["axis"] == "approved_policy"
    )
    assert scan["selected_count"] == 1
    assert scan["status"] == ("complete" if mutation is None else "incomplete")
    assert scan["engineering_missing_count"] == (0 if mutation is None else 1)
    assert scan["micro"] == (0 if mutation is None else None)


def test_threshold_proof_reader_is_scoped_and_deduplicates_reads(tmp_path, monkeypatch):
    from collections import Counter

    from pmmc.evaluation.paper_statistics import SelectedQuery, StatisticsConfig
    from pmmc.experiments import step8_analysis as analysis
    from pmmc.experiments.step8_artifacts import SnapshotArtifactReader
    from step8_threshold_fixtures import produce_thresholds, threshold_fixture

    case = threshold_fixture(tmp_path, monkeypatch)
    produce_thresholds(case)
    readers = []

    class CountingReader(SnapshotArtifactReader):
        def __init__(self, store):
            super().__init__(store)
            self.reads = Counter()
            readers.append(self)

        def _read_object(self, path, digest, signature, **kwargs):
            self.reads[digest] += 1
            return super()._read_object(path, digest, signature, **kwargs)

    monkeypatch.setattr(analysis, "SnapshotArtifactReader", CountingReader)
    facts = SimpleNamespace(
        directory=case.directory,
        store=case.store,
        selection=case.selection,
        manifest=case.manifest,
        cells=case.cells,
        files={},
        input_hashes={},
    )
    metadata = {
        ("memlens", "query-1"): SelectedQuery("query-1", "memlens", "cluster", "conversation")
    }
    first = analysis._threshold_scans(facts, metadata, {}, StatisticsConfig())
    second = analysis._threshold_scans(facts, metadata, {}, StatisticsConfig())
    assert first == second
    assert len(readers) == 2
    assert readers[0].reads == readers[1].reads
    assert len(readers[0].reads) > 10
    assert set(readers[0].reads.values()) == {1}
