from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_campaign import build_campaign_jobs
from pmmc.experiments.step8_jobs import (
    SOURCE_OWNERS,
    BenchmarkStageJob,
    invalidated_downstream,
    validate_job_dag,
)
from pmmc.experiments.step8_runner import CampaignWorker
from pmmc.experiments.step8_selection import ExposureManifest, select_benchmark_queries
from pmmc.schemas.base import DatasetName
from pmmc.schemas.pilot import BenchmarkPartitionAssignment, PartitionRole
from pmmc.schemas.step8 import (
    ANSWER_BACKBONES,
    BenchmarkCampaignManifest,
    CampaignPhase,
    ExperimentCell,
    MethodProtocolFreeze,
    ResourceClass,
    ScientificProtocol,
    Step8ExecutionConfig,
    Treatment,
    build_experiment_matrix,
)
from step8_model_fixtures import model_entry_hashes, model_lock

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (DatasetName.MEMLENS, DatasetName.MEMGALLERY)


def protocol() -> ScientificProtocol:
    return ScientificProtocol(
        model_lock_hash="a" * 64,
        model_entry_hashes=model_entry_hashes(),
        partition_hash="b" * 64,
        exposure_manifest_hash="c" * 64,
        source_fingerprints={key: "d" * 64 for key in SOURCE_OWNERS},
        config_hashes={
            key: "e" * 64
            for key in (
                "questioner_config",
                "planner_config",
                "foundation_config",
                "doubter_config",
                "query_executor_config",
                "baseline_full_context_config",
                "baseline_rag_config",
                "evaluation_config",
            )
        },
    )


def population() -> list[BenchmarkPartitionAssignment]:
    return [
        BenchmarkPartitionAssignment(
            partition_id="partition",
            dataset=dataset,
            snapshot_id="snapshot-" + dataset.value,
            query_id=f"{dataset.value}-{role.value}-{category}-{index}",
            conversation_id=f"conversation-{role.value}-{category}",
            scenario_id=f"scenario-{category}" if dataset is DatasetName.MEMGALLERY else None,
            source_category=f"category-{category}",
            has_query_image=False,
            fold=0 if role is PartitionRole.DEVELOPMENT else 1,
            role=role,
            ordinal=index,
        )
        for dataset in DATASETS
        for role in PartitionRole
        for category in range(4)
        for index in range(12)
    ]


def exposure(complete=True) -> ExposureManifest:
    return ExposureManifest(
        partition_hash="b" * 64,
        complete=complete,
        query_ids={key: [] for key in DATASETS},
        provenance_hashes=["a" * 64],
    )


def regression() -> dict:
    return {
        dataset: [
            row.query_id
            for row in population()
            if row.dataset is dataset and row.role is PartitionRole.DEVELOPMENT and row.ordinal < 4
        ]
        for dataset in DATASETS
    }


def choose(phase, selection, previous=None, history=None):
    return select_benchmark_queries(
        campaign_id="campaign",
        phase=phase,
        selection=selection,
        partition_hash="b" * 64,
        assignments=population(),
        exposure=history or exposure(),
        regression_ids=regression(),
        previous=previous,
    )


def test_paper_selection_nested_nonempty_no_repeated_assignments():
    small = choose(CampaignPhase.PAPER, "5")
    medium = choose(CampaignPhase.PAPER, "20", small)
    full = choose(CampaignPhase.PAPER, "100", medium)
    for dataset in DATASETS:
        assert (
            set(small.query_ids[dataset])
            < set(medium.query_ids[dataset])
            < set(full.query_ids[dataset])
        )
        assert set(medium.added_query_ids[dataset]).isdisjoint(small.query_ids[dataset])
        assert len(full.query_ids[dataset]) == 48
    assert medium.parent_selection_hash == small.content_hash


def test_exploration_selects_all_development_even_with_unknown_history():
    selection = choose(CampaignPhase.EXPLORATION, "dev-all", history=exposure(complete=False))
    for dataset in DATASETS:
        expected = {r.query_id for r in population() if
                    r.dataset == dataset and r.role is PartitionRole.DEVELOPMENT}
        assert set(selection.query_ids[dataset]) == expected
    for phase, name in [(CampaignPhase.PAPER, "dev-all"), (CampaignPhase.DEVELOPMENT, "dev-all"),
                        (CampaignPhase.EXPLORATION, "confirmation")]:
        with pytest.raises(ValueError):
            choose(phase, name)


def test_exploration_scopes_keep_paid_calls_and_baselines_out_of_open_main():
    from pmmc.experiments.release_scope import require_preceding_stage, stage_job_ids

    selected = choose(CampaignPhase.EXPLORATION, "dev-all")
    manifest = BenchmarkCampaignManifest(
        campaign_id="campaign", phase=CampaignPhase.EXPLORATION, protocol=protocol(),
        partition_id="partition", snapshot_ids={k: k.value for k in DATASETS},
        matrix_hash="matrix", created_at=datetime.now(UTC),
    )
    # Use one conversation per dataset, without changing the synthetic denominator.
    jobs = build_campaign_jobs(manifest=manifest, selection=selected, project_root=ROOT,
                               conversations={k: {"c": selected.query_ids[k]} for k in DATASETS})
    scope = stage_job_ids(jobs, "main", paid=False)
    selected_jobs = [j for j in jobs if j.job_id in scope]
    assert not any(j.resource_class is ResourceClass.API for j in selected_jobs)
    assert not any(j.operation in {"baseline_answer", "baseline_grounding", "evaluate",
                                   "analysis_evaluate"} for j in selected_jobs)
    assert all(j.parameters.get("method") in {None, "native_multimodal_rag"} for j in selected_jobs)
    assert sum(j.operation == "primary_evaluate" for j in selected_jobs) == 2 * sum(
        len(qs) for qs in selected.query_ids.values())
    with pytest.raises(ValueError, match="main_is_incomplete"):
        require_preceding_stage(jobs, dict.fromkeys(scope, "completed"), "analysis")


@pytest.mark.parametrize(
    "identities",
    [
        {},
        {"": "a" * 64},
        {" ": "a" * 64},
        {"model": "A" * 64},
        {"model": "a" * 63},
        {"model": "g" * 64},
    ],
)
def test_protocol_model_entry_identity_rejects_empty_keys_and_non_sha256(identities):
    with pytest.raises(ValueError, match="model entry identities"):
        ScientificProtocol.model_validate(
            {**protocol().model_dump(), "model_entry_hashes": identities}
        )


def test_protocol_model_entry_identity_is_required_and_keeps_whole_lock():
    value = protocol().model_dump()
    assert value["model_lock_hash"] == "a" * 64
    assert value["model_entry_hashes"] == model_entry_hashes()
    del value["model_entry_hashes"]
    with pytest.raises(ValueError, match="model_entry_hashes"):
        ScientificProtocol.model_validate(value)


def test_development_confirmation_disjoint_and_stratified():
    original = choose(CampaignPhase.DEVELOPMENT, "regression")
    confirmed = choose(CampaignPhase.DEVELOPMENT, "confirmation", original)
    for dataset in DATASETS:
        assert len(confirmed.query_ids[dataset]) == 16
        assert not set(original.query_ids[dataset]) & set(confirmed.query_ids[dataset])
        assert set(confirmed.stratum_counts[dataset].values()) == {4}


@pytest.mark.parametrize(
    "history",
    [
        exposure(False),
        exposure().model_copy(
            update={
                "query_ids": {
                    DatasetName.MEMLENS: ["memlens-paper_test-0-0"],
                    DatasetName.MEMGALLERY: [],
                }
            }
        ),
    ],
)
def test_paper_rejects_unknown_or_exposed_denominator(history):
    with pytest.raises(ValueError, match="exposure"):
        choose(CampaignPhase.PAPER, "5", history=history)


def test_matrix_four_backbones_full_grid_only_9b_no_duplicate_no_doubter():
    cells = build_experiment_matrix(protocol())
    assert len(cells) == len({item.cell_id for item in cells})
    for dataset in DATASETS:
        for model in ANSWER_BACKBONES:
            main = [
                item
                for item in cells
                if item.dataset is dataset
                and item.answer_model == model
                and item.treatment is Treatment.MAIN
                and item.question_budget == 8
                and item.round_id == 2
                and item.source_variant == "active"
            ]
            assert len(main) == 6
    assert not any("no_doubter" in item.readable_name for item in cells)
    assert (
        len(
            [
                item
                for item in cells
                if item.method == "pmmc"
                and item.treatment is Treatment.MAIN
                and item.source_variant == "active"
                and item.answer_model == ANSWER_BACKBONES[0]
            ]
        )
        == 32
    )


def test_unapproved_freeze_and_non9b_ablations_rejected():
    with pytest.raises(ValueError):
        MethodProtocolFreeze(
            protocol=protocol(),
            decision="approved",
            reviewer_id="reviewer",
            rationale_sha256="r",
            development_receipt_hash="d",
            confirmation_receipt_hash="c",
            approved_at=datetime.now(UTC),
            protocol_integrity_passed=True,
            scientific_viability_passed=False,
            exposure_complete=True,
        )
    with pytest.raises(ValueError):
        ExperimentCell(
            protocol_hash="p",
            dataset=DatasetName.MEMLENS,
            answer_model=ANSWER_BACKBONES[2],
            method="pmmc",
            round_id=1,
        )


def test_paper_plan_requires_config_to_match_selected_freeze_thresholds(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pmmc.experiments import step8_campaign as campaign
    from pmmc.schemas.step8 import Step8BenchmarkConfig
    from step8_fixture_approval import saved_protocol_freeze

    config = Step8BenchmarkConfig(
        partition_dir=tmp_path / "partition",
        exposure_manifest=tmp_path / "exposure.json",
        regression_manifest=tmp_path / "regression.json",
        model_lock=tmp_path / "synthetic-lock.json",
        output_root=tmp_path / "paper",
        match_threshold=0.7,
        gap_threshold=0.1,
    )
    write_json(config.exposure_manifest, exposure().model_dump(mode="json"))
    write_json(config.regression_manifest, {"query_ids": regression()})
    write_json(config.model_lock, model_lock().model_dump(mode="json"))
    partition = SimpleNamespace(
        content_hash=exposure().partition_hash,
        partition_id="partition",
        datasets=[SimpleNamespace(dataset=d, snapshot_id=d.value + "-snapshot") for d in DATASETS],
    )
    monkeypatch.setattr(campaign, "verify_partition", lambda _: partition)
    monkeypatch.setattr(campaign, "load_partition", lambda _: (partition, population()))
    monkeypatch.setattr(campaign, "sha256_file", lambda _: "e" * 64)
    monkeypatch.setattr(campaign, "scientific_source_fingerprint", lambda root, stage: "d" * 64)
    selected = campaign.make_protocol(config, ROOT)
    freeze = saved_protocol_freeze(selected, tmp_path / "synthetic-approval")
    freeze_path = tmp_path / "freeze.json"
    write_json(freeze_path, freeze.model_dump(mode="json"))
    for field, wrong in (
        ("match_threshold", 0.6),
        ("gap_threshold", 0.05),
        ("no_reranker_match_threshold", 0.8),
        ("no_reranker_gap_threshold", 0.0),
    ):
        with pytest.raises(ValueError, match="differs from approval"):
            campaign.plan_campaign(
                config=config.model_copy(update={field: wrong}),
                project_root=ROOT,
                phase=CampaignPhase.PAPER,
                selection="5",
                freeze_path=freeze_path,
            )
    _, planned, _ = campaign.plan_campaign(
        config=config,
        project_root=ROOT,
        phase=CampaignPhase.PAPER,
        selection="5",
        freeze_path=freeze_path,
    )
    assert planned.protocol == selected
    assert planned.protocol_freeze_hash == freeze.content_hash
    assert planned.protocol.match_threshold == 0.7 and planned.protocol.gap_threshold == 0.1


def test_dag_compiler_scope_never_contains_query_or_gold():
    selected = choose(CampaignPhase.DEVELOPMENT, "regression")
    campaign = BenchmarkCampaignManifest(
        campaign_id="campaign",
        phase=CampaignPhase.DEVELOPMENT,
        protocol=protocol(),
        partition_id="partition",
        snapshot_ids={key: "s-" + key.value for key in DATASETS},
        matrix_hash="m",
        created_at=datetime.now(UTC),
    )
    conversations = {key: {"one-conversation": selected.query_ids[key]} for key in DATASETS}
    jobs = build_campaign_jobs(
        manifest=campaign, selection=selected, conversations=conversations, project_root=ROOT
    )
    validate_job_dag(jobs)
    q = [item for item in jobs if item.stage == "questioner"]
    assert len(q) == 6  # 2 conversations x three independent compiler treatments.
    assert all("query_id" not in item.parameters and "gold" not in item.parameters for item in q)
    changed = dict(protocol().source_fingerprints)
    actual = {item.stage: item.source_fingerprint for item in jobs}
    changed.update(actual)
    changed["answers"] = "new-answer-source"
    invalid = invalidated_downstream(jobs, changed)
    assert not any(key.startswith("questioner-") for key in invalid)
    assert any(key.startswith("grounding-") for key in invalid)


def test_response_journal_and_safe_gc_retain_successful_payload(tmp_path):
    store = CampaignArtifactStore(tmp_path)
    digest = store.journal_response(
        request_id="request",
        request_hash="a" * 64,
        response={"answer": "private"},
        attempt_id="attempt-a",
    )
    assert store.restored_response(request_id="request", request_hash="a" * 64) == {
        "answer": "private"
    }
    store.attempt_cache("attempt-a")
    terminal = tmp_path / "terminal.json"
    write_json(terminal, {"attempt_id": "attempt-a", "state": "failed"})
    assert store.clean_attempt("attempt-a", durable_receipt=terminal)["removed"]
    trace_hash = store.put_json({"provider_latency_ms": 23})
    write_json(
        tmp_path / "_store/model-traces" / ("a" * 64 + ".json"),
        {
            "request_hash": "a" * 64,
            "trace_sha256": trace_hash,
        },
    )
    store.gc(referenced_hashes=[], reference_scan_complete=True, apply=True, now=10**12)
    assert store.verify(digest).is_file()
    assert store.verify(trace_hash).is_file()
    with pytest.raises(ValueError):
        store.gc(referenced_hashes=[], reference_scan_complete=False)


class FakeRepository:
    def __init__(self, job):
        self.job, self.state, self.fence = job, "pending", 0

    async def start_attempt(self, **kwargs):
        self.attempt = kwargs

    async def finish_attempt(self, attempt_id, state, receipt_hash):
        assert attempt_id == self.attempt["attempt_id"]
        self.attempt_terminal = state, receipt_hash

    async def claim(self, campaign, resource, owner, seconds):
        if self.state != "pending":
            return None
        self.state, self.owner = "running", owner
        self.fence += 1
        return self.job, self.fence

    async def heartbeat(self, *args):
        return None

    async def upstream_receipts(self, job):
        return {}

    async def complete(self, receipt, owner, fence):
        assert owner == self.owner and fence == self.fence
        self.state = "completed"

    async def fail(self, *args, **kwargs):
        self.state = "failed"

    async def counts(self, campaign):
        return {self.state: 1}


def test_worker_replays_completed_component_without_backend_call(tmp_path):
    job = BenchmarkStageJob(
        campaign_id="campaign",
        stage="snapshot",
        component_key="snapshot",
        resource_class=ResourceClass.CPU,
        source_fingerprint="f",
        scientific_inputs={"snapshot": "s"},
        operation="snapshot",
    )
    repository = FakeRepository(job)

    class Backend:
        calls = 0

        async def preflight(self, job):
            return None

        async def execute(self, job, upstream):
            self.calls += 1
            return {"snapshot_id": "s"}, [], {"queries": 1}

    backend = Backend()

    async def invoke():
        return await CampaignWorker(
            repository=repository,
            backend=backend,
            campaign_id="campaign",
            campaign_dir=tmp_path / "campaign",
            resource=ResourceClass.CPU,
            execution=Step8ExecutionConfig(cpu_workers=2),
            execution_image="image",
            artifact_store=CampaignArtifactStore(tmp_path),
            event=lambda _: None,
        ).run()

    first = asyncio.run(invoke())
    repository.state = "pending"
    second = asyncio.run(invoke())
    assert first["campaign_complete"] and second["reused"] == 1 and backend.calls == 1
    assert not list((tmp_path / "_cache").glob("attempt-*"))
    receipts = list((tmp_path / "campaign/attempts").glob("*/terminal.json"))
    assert len(receipts) == 2 and all(
        json.loads(path.read_text())["state"] == "completed" for path in receipts
    )


def test_worker_setup_failure_preserves_capsule_before_scratch_cleanup(tmp_path):
    job = BenchmarkStageJob(
        campaign_id="campaign",
        stage="snapshot",
        component_key="snapshot",
        resource_class=ResourceClass.CPU,
        source_fingerprint="f",
        scientific_inputs={"snapshot": "s"},
        operation="snapshot",
    )

    class Repository(FakeRepository):
        async def start_attempt(self, **kwargs):
            raise TimeoutError("private database URL")

    class Backend:
        async def preflight(self, job):
            raise AssertionError("must not reach component")

        async def execute(self, job, upstream):
            raise AssertionError("must not reach component")

    worker = CampaignWorker(
        repository=Repository(job),
        backend=Backend(),
        campaign_id="campaign",
        campaign_dir=tmp_path / "campaign",
        resource=ResourceClass.CPU,
        execution=Step8ExecutionConfig(),
        execution_image="image",
        artifact_store=CampaignArtifactStore(tmp_path),
        event=lambda _: None,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(worker.run())
    capsule = json.loads((worker.attempt_dir / "failure-capsule.json").read_text())
    assert capsule["failure"]["stage"] == "worker"
    assert capsule["failure"]["error_code"] == "TimeoutError"
    assert "private database" not in (worker.attempt_dir / "safe-execution.log").read_text()
    assert not list((tmp_path / "_cache").glob("attempt-*"))
