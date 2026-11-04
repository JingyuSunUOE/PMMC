"""Offline integrity and publication checks; no connections to real upstreams."""

import hashlib
import io
import json
import tarfile

import pytest

from pmmc.benchmarks import release_data as data


def _fixture(tmp_path):
    raw = b'{"fixture":true}\n'
    digest = hashlib.sha256(raw).hexdigest()
    row = dict(
        target="data/dataset_32k.json",
        url="https://huggingface.co/datasets/fixture",
        size=len(raw),
        sha256=digest,
    )
    spec = dict(
        files=[row],
        file_count=1,
        materialized_sha256=data._tree_digest([(row["target"], len(raw), digest)]),
    )
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(dict(protocol="pmmc-public-data-lock-v1", datasets={"memlens": spec}))
    )
    return raw, lock


def test_download_verify_offline_repair_and_atomic_failure(tmp_path, monkeypatch):
    raw, lock = _fixture(tmp_path)
    root = tmp_path / "volume"
    monkeypatch.setattr(data, "urlopen", lambda *a, **kw: io.BytesIO(raw))

    def run():
        return data.prepare_data(root, ["memlens"], lock_path=lock)["benchmarks"]["memlens"][
            "state"
        ]

    assert run() == "installed"
    monkeypatch.setattr(data, "urlopen", lambda *a, **kw: pytest.fail("unexpected network access"))
    assert run() == "verified"
    assert not (root / "benchmarks/memgallery").exists()
    published = root / "benchmarks/memlens/data/dataset_32k.json"
    published.write_bytes(b"partial")
    assert run() == "repaired"
    assert published.read_bytes() == raw
    assert run() == "verified"
    stage = root / ".release-data-stage/memlens"
    assert [f.read_bytes() for f in stage.glob("candidate-*/data/dataset_32k.json")] == [b"partial"]
    published.write_bytes(b"partial")
    next((stage / "blobs").iterdir()).write_bytes(b"damaged cache")
    monkeypatch.setattr(data, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong"))
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="failed verification"):
        run()
    assert published.read_bytes() == b"partial"


def test_untrusted_archive_cannot_escape_candidate(tmp_path):
    tar = tmp_path / "locked.tar.gz"
    with tarfile.open(tar, "w:gz") as archive:
        info = tarfile.TarInfo("release_images/../../escape.png")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        data._unpack(tar, tmp_path / "candidate")
    assert not (tmp_path / "escape.png").exists()


def test_gallery_inventory_checked_before_payload_download():
    spec = json.loads(data.LOCK.read_bytes())["datasets"]["memgallery"]
    with pytest.raises(ValueError, match="revision mismatch"):
        data._gallery_files(spec, {"sha": "main"})
    with pytest.raises(ValueError, match="inventory changed"):
        data._gallery_files(spec, {"sha": spec["hf_revision"], "siblings": []})


def test_read_only_analysis_rejects_preparation_before_data_access(tmp_path, monkeypatch):
    from pmmc.experiments import release_assets, release_inputs
    from pmmc.infrastructure.settings import Settings
    from pmmc.release_plan import normalize_plan

    plan = normalize_plan(dict(run_id="read-only", stage="analysis", analyses=["coverage"]))
    monkeypatch.setenv("PMMC_RELEASE_REQUEST", json.dumps(plan))
    monkeypatch.setenv("PMMC_BENCHMARK_STAGING_ROOT", str(tmp_path))
    monkeypatch.setattr(
        release_assets, "prepare_data", lambda *a, **kw: pytest.fail("unexpected data job")
    )
    with pytest.raises(SystemExit):
        release_assets.main()
    with pytest.raises(ValueError, match="without_deps"):
        release_inputs.prepare_exploration(
            Settings(_env_file=None, project_root=tmp_path), "read-only"
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dataset", ["memgallery", "memlens"])
def test_assets_workload_from_empty_volume_without_services(dataset, tmp_path, monkeypatch, capsys):
    """Exercise the actual host command, mount, request and receipt with synthetic Hub bytes."""
    from importlib import import_module

    from scripts.release_host.backends import mount_options, receipt, release_env, role_refs
    from scripts.release_host.workloads import command

    from pmmc import release_asset_receipt
    from pmmc.release_plan import normalize_plan, plan_hash

    raw, lock = _fixture(tmp_path)
    routes = {"https://huggingface.co/datasets/fixture": raw}
    if dataset == "memgallery":
        rev = "a" * 40
        repo = "fixture/gallery"
        source = "data/dialog/fixture.json"
        digest = hashlib.sha256(raw).hexdigest()
        spec = dict(
            hf_dataset=repo,
            hf_revision=rev,
            regular_blob_sha256={source: digest},
            files=[],
            file_count=1,
            materialized_sha256=data._tree_digest(
                [("data/dialogs/fixture.json", len(raw), digest)]
            ),
        )
        lock.write_text(
            json.dumps(dict(protocol="pmmc-public-data-lock-v1", datasets={dataset: spec}))
        )
        routes = {
            f"https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true": json.dumps(
                dict(sha=rev, siblings=[dict(rfilename=source, size=len(raw))])
            ).encode(),
            f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{source}": raw,
        }
    root = tmp_path / "volume"
    root.mkdir()
    cfg = dict(
        images={"app": "fixture@sha256:" + "a" * 64},
        native=dict(
            benchmark_config="config",
            model_registry="registry",
            model_lock="lock",
            runtime_profile="runtime",
        ),
        models=dict(allow_download=False, use_hf_token=True),
        mounts={"inputs": str(root)},
    )
    plan = normalize_plan(
        dict(
            run_id="fixture",
            stage="main",
            benchmarks=[dataset],
            variants=["w"],
            models=["answer-gpt-6-luna"],
            components=["compile"],
        )
    )
    req = dict(plan=plan, variant="w", kind="prepare")
    argv = command("assets", cfg, "fixture", "main", req)
    assert argv[:2] == ["python", "-m"]
    entry = import_module(argv[2])
    mounts = list(mount_options(cfg, "assets", "fixture", approval=None))
    assert mounts == [(str(root), "/pmmc-stage", False)]
    assert role_refs("assets", cfg, None, req) == {}
    env = release_env("assets", cfg, "test", None, req, "main")
    assert env["PMMC_BENCHMARK_STAGING_ROOT"] == "/pmmc-stage"
    assert env["PMMC_BENCHMARK_ASSET_ROOT"] == "/pmmc-inputs"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PMMC_BENCHMARK_STAGING_ROOT", str(root))
    monkeypatch.setattr(entry, "SOURCE_LOCK", lock)
    monkeypatch.setattr(release_asset_receipt, "SOURCE_LOCK", lock)
    calls = []

    def fetch(url, *, timeout):
        calls.append(url)
        return io.BytesIO(routes[url])

    monkeypatch.setattr(data, "urlopen", fetch)
    for state in ("installed", "verified"):
        entry.main()
        final = capsys.readouterr().out.strip().splitlines()[-1]
        assert receipt("assets", final)["benchmarks"] == [dataset]
        report = json.loads(final)
        assert report["plan_hash"] == plan_hash(plan)
        assert release_asset_receipt.validate_asset_receipt(report, plan)["file_counts"] == {
            dataset: 1
        }
        with pytest.raises(ValueError, match="request_mismatch"):
            release_asset_receipt.validate_asset_receipt(report, {**plan, "run_id": "other"})
        with pytest.raises(ValueError, match="locked_tree_mismatch"):
            release_asset_receipt.validate_asset_receipt(
                {**report, "file_counts": {dataset: 2}}, plan
            )
        other = "memlens" if dataset == "memgallery" else "memgallery"
        forged_plan = {**plan, "benchmarks": [other]}
        with pytest.raises(ValueError, match="request_mismatch"):
            release_asset_receipt.validate_asset_receipt(
                {**report, "plan_hash": plan_hash(forged_plan)}, forged_plan
            )
        selected = report["sources"]["benchmarks"]
        assert set(selected) == {dataset} and selected[dataset]["state"] == state
        forged = json.loads(final)
        forged["sources"]["benchmarks"][dataset]["files"] = forged["file_counts"][dataset] = 9
        with pytest.raises(ValueError, match="locked_tree_mismatch"):
            release_asset_receipt.validate_asset_receipt(forged, plan)
    assert sorted(calls) == sorted(routes)


@pytest.mark.parametrize("name", ["memlens", "memgallery"])
@pytest.mark.benchmark_assets
def test_explicit_cold_upstream_membership(name, tmp_path):
    """Opt-in: fetch only this benchmark into an empty volume, then reproduce its full split."""
    import os
    from pathlib import Path

    from pmmc.experiments import release_inputs as release
    from pmmc.schemas import DatasetName

    if os.environ.get("PMMC_TEST_PUBLIC_COLD_DOWNLOAD") != "1":
        pytest.skip("explicit public cold-download opt-in required")
    volume = tmp_path / "volume"
    assert data.prepare_data(volume, [name])["benchmarks"][name]["state"] == "installed"
    assert data.prepare_data(volume, [name])["benchmarks"][name]["state"] == "verified"
    dataset = DatasetName(name)
    adapter = release.create_adapter(
        dataset, root=volume / "benchmarks" / name, project_root=volume
    )
    path, _ = release.build_snapshot(
        adapter, output_root=tmp_path / "canonical", level=release.ValidationLevel.FULL
    )
    release.preflight_snapshot(path)
    part = release._build_original_partition([path], tmp_path / "partitions")
    manifest = release.verify_partition(part)
    assert manifest.datasets[0].development_query_count == {"memlens": 140, "memgallery": 287}[name]
    proof = release._original_membership(Path.cwd(), part, (dataset,))
    assert set(proof["datasets"]) == {name}
