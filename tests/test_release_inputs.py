"""Offline contracts for the release CPU image, planner and deployment pointer."""

import json
import os
import shutil
from pathlib import Path

import pytest

from pmmc.experiments import release_inputs as release
from pmmc.experiments.step8_model_identity import compute_model_entry_hashes
from pmmc.experiments.step8_selection import ExposureManifest
from pmmc.infrastructure.settings import Settings
from pmmc.models.locking import load_model_lock
from pmmc.models.registry import load_model_registry
from pmmc.pilots.partition import load_partition
from pmmc.schemas import DatasetName

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.benchmark_assets
def test_full_release_matches_private_original_partition(tmp_path):
    """Opt-in direct assignment comparison; never plan jobs or print private IDs."""
    reference = os.environ.get("PMMC_EXPLORATION_REFERENCE_PARTITION")
    if not reference:
        pytest.skip("no private reference requested")
    historical = release.verify_partition(Path(reference))
    _, old = load_partition(Path(reference))
    settings = Settings(
        _env_file=None,
        project_root=ROOT,
        artifacts_dir=tmp_path / "artifacts",
        outputs_dir=tmp_path / "outputs",
    )
    paths = []
    for dataset, raw in (
        (DatasetName.MEMLENS, settings.memlens_path),
        (DatasetName.MEMGALLERY, settings.memgallery_path),
    ):
        adapter = release.create_adapter(
            dataset, root=raw, project_root=settings.benchmark_asset_path
        )
        path, _ = release.build_snapshot(
            adapter,
            level=release.ValidationLevel.FULL,
            output_root=settings.benchmark_artifacts_path,
        )
        assert (
            release.preflight_snapshot(path).validation_report.level == release.ValidationLevel.FULL
        )
        paths.append(path)
    partition = release._build_original_partition(paths, tmp_path / "partitions")
    release.verify_partition(partition)
    membership = release._original_membership(ROOT, partition)
    assert membership["reference_partition_hash"] == historical.content_hash
    _, new = load_partition(partition)
    for dataset, dev_count in ((DatasetName.MEMLENS, 140), (DatasetName.MEMGALLERY, 287)):
        before = {r.query_id: (r.fold, r.role) for r in old if r.dataset == dataset}
        after = {r.query_id: (r.fold, r.role) for r in new if r.dataset == dataset}
        if before != after:
            pytest.fail(
                f"{dataset.value}: FULL assignments differ from the named private reference"
            )
        assert sum(fold == 0 for fold, _ in after.values()) == dev_count
    assert not settings.outputs_path.exists()


def test_portable_model_and_capacity_inputs():
    config = release.load_benchmark_config(ROOT / release._TEMPLATE)
    registry = load_model_registry(ROOT / config.model_registry)
    lock = load_model_lock(ROOT / config.model_lock)
    assert set(compute_model_entry_hashes(registry, lock)) == {m.alias for m in registry.models}
    assert len({e.commit_sha for e in lock.entries}) == 5
    for entry in lock.entries:
        model = registry.get(entry.alias)
        assert entry.requested_revision == model.model_revision == entry.commit_sha
        assert model.tokenizer_revision == entry.commit_sha
        if model.processor_name:
            assert model.processor_revision == entry.commit_sha
    runtime = release.load_runtime_profile(ROOT / config.runtime_profile)
    assert runtime.model_lock_path == config.model_lock
    (gated,) = [s for s in runtime.services if s.metadata.get("requires_capacity_receipt")]
    assert gated.metadata["frozen_replica_topology"] == {
        "tensor_parallel_size": 2,
        "replica_count": 4,
    }
    assert release.capacity_image_workload(gated, multimodal=True).layouts


@pytest.fixture
def project(tmp_path, memgallery_fixture_root, memlens_fixture_root, monkeypatch):
    # Production uses the built-in full package counts. Small fixtures keep
    # real canonicalization, checksums, exact folds, and Typer plan affordable.
    root = tmp_path.resolve()
    for model in ("src", "configs", "prompts"):
        shutil.copytree(ROOT / model, root / model, ignore=shutil.ignore_patterns("__pycache__"))
    path = memlens_fixture_root / "data/dataset_32k.json"
    (row,) = json.loads(path.read_bytes())
    path.write_text(
        json.dumps(
            [
                {
                    **row,
                    "question_type": "information_extraction",
                    "question_subtype": "information_extraction",
                    "question_id": f"q{i}",
                }
                for i in range(5)
            ]
        )
    )
    path = memgallery_fixture_root / "data/dialogs/scenario_one.json"
    template = path.read_text()
    for i in range(1, 20):
        scenario = f"sc-{i:02}"
        shutil.copytree(
            memgallery_fixture_root / "images/scenario_one",
            memgallery_fixture_root / "images" / scenario,
        )
        (path.parent / f"{scenario}.json").write_text(template.replace("scenario_one", scenario))
    build = release.build_snapshot

    def snapshot(adapter, **kw):
        assert kw["level"] == release.ValidationLevel.FULL
        return build(adapter, **kw, enforce_expected_counts=False)

    monkeypatch.setattr(release, "build_snapshot", snapshot)
    scan = release.scan_adapter
    monkeypatch.setattr(
        release,
        "scan_adapter",
        lambda adapter, level: scan(adapter, level, enforce_expected_counts=False),
    )
    settings = Settings(_env_file=None, project_root=root)
    # Independent reference for these synthetic records only. Never change the
    # shipped original recipe or make the release gate trust a candidate split.
    reference = json.loads((root / release._SPLIT).read_bytes())
    fixture_hashes = {
        "memlens": (
            "fade4836600fa9eddbf1ec9db2603df940d83ffc1a4a08deca91e69e03e2ca04",
            "ff754fe3eee192ae63854dc24d6731e9bc2817d0a14f0d3d7d0405d22513f332",
            "76126337c6c02e3135a000db0a6240b838823e3ad2139246c1890d8caddefdd1",
        ),
        "memgallery": (
            "7f8b3ea2817d8b3fa64a24d4086f23c8c9d0f41f18f414eab211294d3a955b0a",
            "396cc4c78b92a0cb737e9c020de79f02cf7c40ab55fe6db7900a0e31bd674c5d",
            "e20a51e7261dd38678354e9c9723091f8bd8af750fe6848a47bcae4ef554a58b",
        ),
    }
    for dataset, source in (
        (DatasetName.MEMLENS, memlens_fixture_root),
        (DatasetName.MEMGALLERY, memgallery_fixture_root),
    ):
        adapter = release.create_adapter(dataset, root=source, project_root=root)
        _, full = snapshot(
            adapter,
            output_root=settings.benchmark_artifacts_path,
            level=release.ValidationLevel.FULL,
        )
        folds, dev, paper = fixture_hashes[dataset.value]
        reference["datasets"][dataset.value] = dict(
            full_snapshot_id=full.snapshot_id,
            canonical_sha256=full.canonical_content_hash,
            assignment_sha256=folds,
            development_sha256=dev,
            paper_sha256=paper,
        )
    release.write_json(root / release._SPLIT, reference)
    return settings


def test_actual_cli_plan_and_pointer_repair(project, monkeypatch):
    result = release.prepare_exploration(project, "test-native")
    root = project.project_root
    assert result["selected"] == {"memlens": 1, "memgallery": 4}
    assert result["variant"] == "wr" and not result["gpu_benchmark_ready"]
    assert (
        release.exploration_campaign_directory(project, "test-native")
        == root / result["campaign_dir"]
    )
    directory = root / result["campaign_dir"]
    assert list((directory / "private/jobs").glob("*.json"))
    assert not list((directory / "manifests").glob("budget-*.json"))
    assert "GOLD_ANSWER_SENTINEL" not in json.dumps(result)
    assert result["split_membership"]["dataset_split_version"] == release._SPLIT_VERSION
    pointer = root / result["pointer_path"]
    assert json.loads(pointer.read_bytes()) == result
    config = release.load_benchmark_config(root / result["config_path"])
    # The actual scientific planner is an independent oracle for ownership.
    frozen = release.make_protocol(config, root)
    for name in ("step8_cli", "release_reporting", "step8_export"):
        with (root / f"src/pmmc/experiments/{name}.py").open("a") as stream:
            stream.write("\nTEST_OPERATIONAL_REVISION = 2\n")
    monkeypatch.setenv("PMMC_CONTAINER_IMAGE", "different-operational-revision")
    assert release.make_protocol(config, root) == frozen
    assert release.exploration_campaign_directory(project, "test-native") == directory
    pointer.unlink()  # Simulate death after the native inventory but before publication.
    assert release.prepare_exploration(project, "test-native") == result
    # A self-consistent dev-all manifest with the right count is not enough.
    path = next((directory / "manifests/selections").glob("*.json"))
    saved_bytes = path.read_bytes()
    tampered = json.loads(saved_bytes)
    paper_q = next(
        a.query_id
        for a in load_partition(root / config.partition_dir)[1]
        if a.dataset == DatasetName.MEMLENS and a.fold != 0
    )
    for field in ("query_ids", "added_query_ids"):
        tampered[field]["memlens"] = [paper_q]
    wrong = release.SelectionExpansionManifest.model_validate(tampered)
    path.unlink()
    alternate = path.with_name(wrong.content_hash + ".json")
    release.write_json(alternate, wrong.model_dump(mode="json"))
    with pytest.raises(ValueError, match="selection_identity_mismatch"):
        release.exploration_campaign_directory(project, "test-native")
    alternate.unlink()
    path.write_bytes(saved_bytes)
    owned = root / "src/pmmc/compiler/opportunities.py"
    original_science = owned.read_bytes()
    owned.write_bytes(original_science + b"\nSCIENTIFIC_REVISION = 2\n")
    assert (
        release.make_protocol(config, root).source_fingerprints["unit_prepare"]
        != frozen.source_fingerprints["unit_prepare"]
    )
    with pytest.raises(ValueError, match="identity_mismatch"):
        release.exploration_campaign_directory(project, "test-native")
    owned.write_bytes(original_science)
    source = root / "prompts/runtime/pmmc_answer_contract_v3.txt"
    original = source.read_bytes()
    source.write_bytes(original + b"\nChanged scientific instruction.\n")
    assert release.make_protocol(config, root).source_fingerprints != frozen.source_fingerprints
    with pytest.raises(ValueError, match="identity_mismatch"):
        release.exploration_campaign_directory(project, "test-native")
    with pytest.raises(ValueError, match="source_changed"):
        release.prepare_exploration(project, "test-native")
    source.write_bytes(original)
    raw = project.memlens_path / "data/dataset_32k.json"
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="raw_inputs_changed"):
        release.prepare_exploration(project, "test-native")
    assert json.loads(pointer.read_bytes()) == result
    pointer.write_text(json.dumps({**result, "variant": "legacy"}))
    with pytest.raises(ValueError, match="identity_mismatch"):
        release.exploration_campaign_directory(project, "test-native")


def test_variants_share_frozen_history_and_not_campaign_identity(project, monkeypatch):
    # Exercise the native campaign planner; the CLI/DAG publication path is
    # covered above. Only the heavy job-file inventory is stubbed in this test.
    from pmmc.experiments.step8_campaign import plan_campaign

    def plan(settings, config_path, existing, *, lock_fd):
        directory, _, chosen = plan_campaign(
            config=release.load_benchmark_config(config_path),
            project_root=settings.project_root,
            phase=release.CampaignPhase.EXPLORATION,
            selection="dev-all",
            campaign_dir=existing,
        )
        release.write_json(
            directory / f"manifests/jobs-{chosen.content_hash}.json", ["fixture"], immutable=True
        )
        return directory

    monkeypatch.setattr(release, "_plan", plan)
    main = release.prepare_exploration(project, "comparison")
    config = release.load_benchmark_config(project.project_root / main["config_path"])
    manifest, rows = load_partition(project.project_root / config.partition_dir)
    known = {d: [next(r for r in rows if r.dataset is d).query_id] for d in release._DATASETS}
    # New evidence is visible to the next run, never changes this run's cohort.
    history = project.benchmark_asset_path / "known-exposure.json"
    record = ExposureManifest(complete=False, partition_hash=manifest.content_hash, query_ids=known)
    release.write_json(history, record.model_dump(mode="json"))
    protocols = set()
    for variant in ("w", "r", "wr", "legacy"):
        pointer = release.prepare_exploration(project, "comparison", variant=variant)
        assert pointer["partition_hash"] == main["partition_hash"]
        assert pointer["exposure_hash"] == main["exposure_hash"]
        assert pointer["selected"] == main["selected"]
        target = release.load_benchmark_config(project.project_root / pointer["config_path"])
        assert (target.questioner_config == release._WRITER) == ("w" in variant)
        assert (target.query_executor_config == release._READER) == ("r" in variant)
        protocols.add(pointer["protocol_hash"])
    assert len(protocols) == 4
    second = release.prepare_exploration(project, "second")
    assert second["exposure_hash"] != main["exposure_hash"]
    config = release.load_benchmark_config(project.project_root / second["config_path"])
    saved = ExposureManifest.model_validate_json(
        (project.project_root / config.exposure_manifest).read_bytes()
    )
    assert not saved.complete and saved.query_ids == known
    history.unlink()  # The previous run's frozen private cohort still records these IDs.
    third = release.prepare_exploration(project, "third")
    assert second["protocol_hash"] == third["protocol_hash"]
    assert second["campaign_id"] != third["campaign_id"]
    # All variants bind both W/R configs even when the selected variant is legacy.
    legacy = release.exploration_campaign_directory(project, "comparison", variant="legacy")
    legacy_config = release.load_benchmark_config(legacy / "manifests/config.json")
    native = release.make_protocol(legacy_config, project.project_root)
    writer = project.project_root / release._WRITER
    writer.write_bytes(
        writer.read_bytes().replace(b"fts_relation_top_k: 8", b"fts_relation_top_k: 7")
    )
    assert release.make_protocol(legacy_config, project.project_root) == native
    with pytest.raises(ValueError, match="identity_mismatch"):
        release.exploration_campaign_directory(project, "comparison", variant="legacy")


def test_recognize_private_receipt_without_shared_ledger(project):
    run = release._run_root(project, "private", "wr")
    run.mkdir(parents=True)
    data = release._bootstrap_inputs(project, run)
    path = run / "inputs" / data["partition_id"]
    _, assignments = load_partition(path)
    ids = {d: [next(a for a in assignments if a.dataset == d).query_id] for d in release._DATASETS}
    event = dict(
        protocol="step8-exposure-event-v1",
        purpose="production_dispatch",
        campaign_id="archived",
        partition_hash="earlier",
        query_ids=ids,
    )
    digest = release.canonical_content_hash(event)
    receipt = project.outputs_path / f"step8/archive/private/exposure/{digest}.json"
    release.write_json(receipt, event)
    known = release._known_exposure(project, path)
    assert not known.complete and known.query_ids == ids and digest in known.provenance_hashes
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="private_exposure_event_invalid"):
        release._known_exposure(project, path)


def test_refuse_unknown_history_or_ambiguous_recovery(project):
    run = release._run_root(project, "conflict", "wr")
    run.mkdir(parents=True)
    record = ExposureManifest(
        partition_hash="historical",
        complete=False,
        query_ids={DatasetName.MEMLENS: ["outside"], DatasetName.MEMGALLERY: []},
    )
    path = project.artifacts_path / "step8-inputs/exposure.json"
    release.write_json(path, record.model_dump(mode="json"))
    with pytest.raises(ValueError, match="outside the frozen partition"):
        release.prepare_exploration(project, "conflict")
    assert not (run / "inputs").exists()
    assert not (run / "wr/pointer.json").exists()
    path.unlink()
    data = release._bootstrap_inputs(project, run)
    config = release._config(project, run, "wr", data)
    for name in ("2026_exploration_1", "2027_exploration_2"):
        path = project.outputs_path / "step8" / name / "manifests/config.json"
        release.write_json(path, config.model_dump(mode="json"))
    with pytest.raises(ValueError, match="ambiguous"):
        release.prepare_exploration(project, "conflict")
    assert not (run / "wr/pointer.json").exists()


@pytest.mark.parametrize("cause", ["same_counts_swapped_members", "v3", "new_data"])
def test_refuse_unsafe_split_before_planning(project, monkeypatch, cause):
    native = release.native_partition
    if cause == "same_counts_swapped_members":
        original = native._assign_folds

        def permute(records, spec):
            assignments = original(records, spec)
            if records[0].query.dataset == DatasetName.MEMLENS:
                a, b = (
                    next(key for key, value in assignments.items() if value == fold)
                    for fold in (0, 1)
                )
                assignments[a], assignments[b] = assignments[b], assignments[a]
            return assignments

        monkeypatch.setattr(native, "_assign_folds", permute)
    if cause == "new_data":
        raw = project.memlens_path / "data/dataset_32k.json"
        raw.write_bytes(raw.read_bytes() + b"\n")
    else:
        spec = (
            release._V2
            if cause != "v3"
            else release._V2.model_copy(
                update={"protocol_version": "scenario-balanced-five-fold-v3"}
            )
        )
        monkeypatch.setattr(
            release,
            "_build_original_partition",
            lambda paths, staging: native.build_partition(
                snapshot_dirs=paths, output_root=staging, spec=spec
            )[0],
        )
    monkeypatch.setattr(release, "_plan", lambda *a, **k: pytest.fail("unsafe plan attempted"))
    code = {
        "same_counts_swapped_members": "membership_differs",
        "v3": "original_split_required",
        "new_data": "requires_versioned_dataset_and_split",
    }[cause]
    with pytest.raises(ValueError, match=code):
        release.prepare_exploration(project, "invalid")
    assert not (release._run_root(project, "invalid", "wr") / "inputs").exists()


@pytest.mark.parametrize("name,variant", [("../outside", "wr"), ("good", "../bad"), ("", "w")])
def test_invalid_owner_has_no_side_effect(tmp_path, name, variant):
    with pytest.raises(ValueError, match="invalid_exploration"):
        release.prepare_exploration(
            Settings(_env_file=None, project_root=tmp_path), name, variant=variant
        )
    assert not list(tmp_path.iterdir())
