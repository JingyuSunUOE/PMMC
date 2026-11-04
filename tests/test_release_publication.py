"""Offline fixtures for committed source, candidate lock, IAM contexts and Kind gates."""

import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from scripts import kind_persistence as kind
from scripts import release_images as images
from scripts import release_source as source

ROOT = Path(__file__).resolve().parents[1]
APP = "ghcr.io/example/pmmc@sha256:" + "a" * 64
IAM = "ghcr.io/example/pmmc-iam@sha256:" + "b" * 64


def test_archive_reproducible_and_verifiable(tmp_path):
    if not (ROOT / ".git").exists():
        pytest.skip("immutable Git object database is absent in extracted SOURCE")
    files = source.source()  # deliberately ignores changed, untracked and ignored working files
    first, second = tmp_path / "source.tar.gz", tmp_path / "repeat.tar.gz"
    source.archive(files, first)
    source.archive(dict(reversed(list(files.items()))), second)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        assert all(m.isfile() and m.mtime == m.uid == m.gid == 0 for m in archive)
        archive.extractall(tmp_path / "copy", filter="data")
    report = subprocess.check_output(
        [sys.executable, "-B", ROOT / "docker/verify_public_source.py", tmp_path / "copy"]
    )
    assert json.loads(report)["files_verified"] == len(files) - 1
    files["Dockerfile"] = (0o644, b"tampered\n")
    with pytest.raises(ValueError, match="SOURCE.sha256"):
        source.archive(files, tmp_path / "bad.tar.gz")
    assert not (tmp_path / "bad.tar.gz").exists()


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        "configs/.env.private",
        "deploy/release/images.lock.json",
        "deploy/release/Images.lock.json",
        "src/../outside",
        "/etc/passwd",
        "benchmarks/memlens/data/snapshot.json",
        "src/private/log.py",
        "scripts/host.key",
        "docs/photograph.jpg",
        "configs/local-credentials.json",
        "configs/*",
        ".git/config",
    ],
)
def test_forbidden_source_paths(name):
    with pytest.raises(ValueError):
        source.inventory([name])


def test_reject_links_before_reading_blobs(monkeypatch):
    monkeypatch.setattr(
        source,
        "git",
        lambda *args, **kw: (
            str(Path.cwd()).encode()
            if args[0] == "rev-parse"
            else b"120000 blob abc\tscripts/not-source.py\0"
        ),
    )
    with pytest.raises(ValueError, match="links"):
        source.entries(False)


def test_source_membership_must_be_explicit(monkeypatch):
    # Include inventory itself but not the new path in the committed inventory contents.
    monkeypatch.setattr(
        source,
        "entries",
        lambda index: {
            ".dockerignore": (0o644, "a"),
            "SOURCE.sha256": (0o644, "b"),
            "scripts/unreviewed.py": (0o644, "c"),
        },
    )
    monkeypatch.setattr(
        source, "git", lambda *args: source.inventory({".dockerignore", "SOURCE.sha256"})
    )
    with pytest.raises(ValueError, match="inventory"):
        source.source()


@pytest.fixture
def lock(tmp_path, monkeypatch):
    p = tmp_path / images.LOCK
    p.parent.mkdir(parents=True)
    (tmp_path / images.UPSTREAM).write_bytes((ROOT / images.UPSTREAM).read_bytes())
    (tmp_path / "SOURCE.sha256").write_bytes(b"fixture-manifest\n")
    monkeypatch.setattr(images, "commit", lambda root: "c" * 40)
    monkeypatch.setattr(
        images, "committed_blob", lambda root, rev, path: (root / path).read_bytes()
    )
    raw = images.produce(tmp_path, APP, IAM)
    p.write_bytes(raw)
    monkeypatch.setenv("PMMC_IMAGE_LOCK_SHA256", images.sha(raw))
    return json.loads(raw)


def test_lock_mirrors_are_identity_preserving(tmp_path, lock):
    mirror = "registry.internal:5000/mirror/app@sha256:" + "a" * 64
    assert images.load_images(tmp_path, {"app": mirror}) == {**lock["images"], "app": mirror}
    with pytest.raises(ValueError, match="changes reviewed"):
        images.load_images(tmp_path, {"app": mirror[:-1] + "d"})
    with pytest.raises(ValueError, match="invalid image mirrors"):
        images.load_images(tmp_path, {"extra": mirror})


def test_runtime_cannot_bypass_lock_with_explicit_toml_digests(tmp_path, monkeypatch, lock):
    from scripts.release_host import config

    file = tmp_path / "operator.toml"
    file.write_text("[images]\n" + "\n".join(f'{k} = "{v}"' for k, v in lock["images"].items()))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / images.LOCK).unlink()
    with pytest.raises(config.Error, match="images.lock.json"):
        config.load_config(file, "kubernetes")


def test_cli_loads_verified_asset_without_registry_or_cluster(tmp_path, host_release_source):
    root, checksum = host_release_source
    template = (root / "deploy/release/kubernetes.toml").read_text()
    data = re.sub(r"REPLACE_WITH_[A-Z0-9_]+", "fixture", template)
    data = data.replace('discovery = "live"', 'discovery = "declared"')
    cfg = tmp_path / "deploy.toml"
    cfg.write_text(data + "\n[capacity]\ngpus=1\ncpus=16\nmemory_gib=96\n")
    args = [
        sys.executable,
        "-B",
        root / "scripts/pmmc",
        "plan",
        "--config",
        cfg,
        "--model",
        "answer-qwen3.5-9b",
        "--run-id",
        "test-image-lock",
    ]
    env = {
        "HOME": str(tmp_path),
        "PATH": "",
        "PYTHONPATH": "src",
        "PMMC_DEPLOY_STATE": str(tmp_path / "state"),
        "PMMC_IMAGE_LOCK_SHA256": checksum,
    }
    output = subprocess.check_output(args, env=env, cwd=root, timeout=20)
    assert json.loads(output)["selection"]["models"] == ["answer-qwen3.5-9b"]
    env["PMMC_IMAGE_LOCK_SHA256"] = "0" * 64
    result = subprocess.run(args, env=env, cwd=root, capture_output=True, timeout=20)
    assert result.returncode and b"images.lock.json" in result.stderr


def test_external_checksum_and_source_identity(tmp_path, monkeypatch, lock):
    monkeypatch.delenv("PMMC_IMAGE_LOCK_SHA256")
    with pytest.raises(ValueError, match="reviewed release asset"):
        images.load_images(tmp_path)
    p = tmp_path / images.LOCK
    monkeypatch.setenv("PMMC_IMAGE_LOCK_SHA256", images.sha(p.read_bytes()))
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="reviewed release asset"):
        images.load_images(tmp_path)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(images, "commit", lambda root: "d" * 40)
    with pytest.raises(ValueError, match="source identity"):
        images.validate(tmp_path, lock)
    monkeypatch.setattr(images, "commit", lambda root: "c" * 40)
    (tmp_path / "SOURCE.sha256").write_bytes(b"other source")
    with pytest.raises(ValueError, match="source identity"):
        images.validate(tmp_path, lock)


def test_saved_deployment_must_match_current_release(tmp_path, monkeypatch, lock):
    from scripts.release_host import config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    config.verify_frozen_images({"images": lock["images"]})
    for stale in ({}, {**lock["images"], "app": APP[:-1] + "c"}):
        with pytest.raises(config.Error):
            config.verify_frozen_images({"images": stale})


def test_lock_rejects_confusion(tmp_path, lock):
    for role, ref in [
        ("app", APP.replace("ghcr.io", "example.org")),
        ("iam", APP),
        ("iam", IAM.replace("@", ":latest@")),
        ("app", APP.split("@")[0]),
        ("postgres", "docker.io/pgvector/pgvector@sha256:" + "c" * 64),
    ]:
        with pytest.raises(ValueError):
            images.validate(tmp_path, {**lock, "images": {**lock["images"], role: ref}})
    with pytest.raises(ValueError, match="duplicate"):
        images.strict_json(b'{"a":1,"a":2}')
    p = tmp_path / images.LOCK
    p.rename(p.with_name("real"))
    p.symlink_to("real")
    with pytest.raises(ValueError, match="linked"):
        images.load_images(tmp_path)


def test_real_renderer_claims_and_pins():
    upstream = json.loads((ROOT / images.UPSTREAM).read_bytes())
    backend = kind.service_backend(upstream)
    for role in kind.ROLES:
        pvc, service, sts = backend.service_objects(role)
        pod = sts["spec"]["template"]["spec"]
        assert pvc["kind"] == "PersistentVolumeClaim" and service["kind"] == "Service"
        assert pvc["metadata"]["name"] == f"{kind.NAME}-{role}"
        assert pod["volumes"] == [
            {
                "name": "data",
                "persistentVolumeClaim": {"claimName": pvc["metadata"]["name"]},
            }
        ]
        assert pod["containers"][0]["image"] == upstream[role]
        assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]


@pytest.mark.parametrize("fault", [None, "old-pod", "new-pv", "lost-db-row", "lost-object"])
def test_persistence_gate_observes_both_replacements(monkeypatch, fault):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("KUBECONFIG", "/unused-fixture")
    monkeypatch.setattr(kind.secrets, "token_hex", lambda n: "a" * (n * 2))
    generations = dict.fromkeys([f"{kind.NAME}-{role}" for role in kind.ROLES], 0)
    markers = {}
    removed = set()

    def fixture_kube(*args, obj=None):
        if args[0] == "scale":
            n = args[1].split("/")[1]
            if args[2] == "--replicas=0":
                removed.add(n)
            else:
                assert n in removed and markers.keys() == {"postgres", "minio"}
                generations[n] += 1
        elif args[0] == "wait":
            assert args[1] == "--for=delete" and args[2][4:-2] in removed
        elif args[0] == "get":
            kind_name, name = args[1:3]
            if kind_name == "pod":
                count = generations[name[:-2]] if fault != "old-pod" else 0
                return json.dumps({"metadata": {"uid": f"{name}-{count}"}})
            volume = name + ("-new" if fault == "new-pv" and generations[name] else "")
            return json.dumps(
                {
                    "metadata": {"uid": name},
                    "status": {"phase": "Bound"},
                    "spec": {"volumeName": volume, "accessModes": ["ReadWriteOnce"]},
                }
            )
        elif args[0] == "exec":
            cmd = args[-1]
            if "pg_isready" in cmd:
                return ""
            if "current_setting" in cmd:
                return "16"
            if "INSERT INTO" in cmd or "mc pipe" in cmd:
                assert "a" * 48 in cmd
                markers["postgres" if "INSERT" in cmd else "minio"] = "a" * 48
            elif cmd.startswith("SELECT value") or "mc cat" in cmd:
                role = "postgres" if "SELECT" in cmd else "minio"
                assert all(generations.values())
                return (
                    ""
                    if fault == {"postgres": "lost-db-row", "minio": "lost-object"}[role]
                    else markers[role]
                )
            else:
                pytest.fail("unexpected probe")
        return ""

    monkeypatch.setattr(kind, "kube", fixture_kube)
    if fault is None:
        kind.check()
        assert len(removed) == 2 and all(generations.values())
    else:
        with pytest.raises(ValueError, match="replacement pods"):
            kind.check()


def test_cold_iam_context_matches_copy_and_verified_mc():
    upstream = json.loads((ROOT / images.UPSTREAM).read_bytes())
    context = ROOT / "docker/minio"
    iam = (context / "step8.Dockerfile").read_text()
    ignores = (context / ".dockerignore").read_text().splitlines()
    root_inventory = (ROOT / ".dockerignore").read_text().splitlines()
    for path in ("step8.Dockerfile", "step8_iam.py", ".dockerignore"):
        assert f"!docker/minio/{path}" in root_inventory
    assert ignores[0] == "**" and "!step8_iam.py" in ignores
    assert "COPY --chmod=0555 step8_iam.py " in iam and (context / "step8_iam.py").is_file()
    assert (
        upstream["minio"] in iam
        and upstream["postgres"] in (ROOT / "docker/postgres/Dockerfile").read_text()
    )
    assert "sha256sum -c -" in iam and "FROM ${PMMC_IMAGE}" in iam


def test_publish_depends_on_storage_and_never_creates_git_release():
    path = ROOT / ".github/workflows/release-check.yml"
    jobs = yaml.safe_load(path.read_bytes())["jobs"]
    job = jobs["linux-image"]
    assert "kind-storage" in job["needs"] and "clean-source" in job["needs"]
    assert job["permissions"]["contents"] == "read"
    assert "workflow_dispatch" in job["if"] and "publish_images" in job["if"]
    attest = [s for s in job["steps"] if "attest-build-provenance@" in s.get("uses", "")]
    assert attest and attest[0]["if"] == "inputs.publish_images"
