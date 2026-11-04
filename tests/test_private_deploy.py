"""Private deploy contract: synthetic inputs, no network or cluster access."""

import argparse
import base64
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest
from scripts.release_host import backends, private_io, private_secrets
from scripts.release_host import private_deploy as deploy
from scripts.release_host import private_policy as pricing
from scripts.release_host.config import Error
from scripts.release_host.private_io import json_bytes, sha256
from scripts.release_host.private_source import IMAGE_LOCK, source_files
from scripts.release_host.secrets import import_secrets

from pmmc.release_plan import normalize_plan


def write(path, data, mode=0o600):
    path.write_bytes(data)
    path.chmod(mode)
    return path


def public_fixture():
    refs = {
        k: f"{'ghcr.io' if i < 2 else 'registry.example'}/fixture/{k}@sha256:{str(i + 1) * 64}"
        for i, k in enumerate(("app", "iam", "postgres", "minio"))
    }
    revisions = {
        "github_revision": "9" * 40,
        "hf_revision": "a" * 40,
        "materialized_sha256": "d" * 64,
    }
    files = {
        "scripts/pmmc": b"#!/usr/bin/env python3\n",
        "configs/release/prepare-models.yaml": b"model_registry: configs/release/models.yaml\n",
        "configs/release/models.yaml": (
            b"models:\n- provider_model: fixture-model\n  provider_config: {kind: openai}\n"
        ),
        "deploy/release/upstream.json": json_bytes({k: refs[k] for k in ("postgres", "minio")}),
        "configs/release/models.lock.json": json_bytes(
            {"entries": [{"commit_sha": "1" * 40, "snapshot_content_hash": "8" * 64}]}
        ),
        deploy.BENCH_LOCK: json_bytes(
            {
                "protocol": "pmmc-public-data-lock-v1",
                "datasets": dict.fromkeys(("memlens", "memgallery"), revisions),
            }
        ),
    }
    files["SOURCE.sha256"] = "".join(
        f"{sha256(v)}  {k}\n" for k, v in sorted(files.items())
    ).encode()
    images = json_bytes(
        {
            "schema_version": 1,
            "source_commit": "e" * 40,
            "source_manifest_sha256": sha256(files["SOURCE.sha256"]),
            "images": refs,
        }
    )
    config = b"""profile = "kubernetes"
instance = "trial"
[cluster]
context = "private-fixture-context"
namespace = "trial-ns"
shared_pvc = "fixture-pvc"
storage_class = "fixture-sc"
[mounts]
inputs = "site/inputs"
outputs = "site/outputs"
cache = "site/cache"
artifacts = "site/artifacts"
[models]
use_hf_token = true
"""
    return files, images, config


def tar_bytes(files, *, extras=(), prefix="", compress=True):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz" if compress else "w") as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(prefix + name)
            member.mode = 0o755 if data.startswith(b"#!") else 0o644
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        for member in extras:
            tar.addfile(member)
    return buf.getvalue()


@pytest.fixture
def packet(tmp_path, monkeypatch):
    files, images, config = public_fixture()
    archive = tar_bytes(files, prefix="fixture/")
    captured = {}

    def mock_age(argv, data, **kwargs):
        assert argv[0] == "age"
        if "--encrypt" in argv:
            captured["plain"] = data
            return deploy.HEADER + sha256(data).encode()
        assert "--decrypt" in argv and kwargs["pass_fds"]
        assert argv[-1] == f"/dev/fd/{kwargs['pass_fds'][0]}"
        return captured["plain"]

    monkeypatch.setattr(deploy, "invoke", mock_age)
    args = argparse.Namespace(
        recipient="age1q8w4ur23clxd5mzfsh79vn6pg0kaytjeq8w4ur23clxd5mzfsh7sresg5r",
        source_sha256=sha256(archive),
        images_sha256=sha256(images),
        source_archive=write(tmp_path / "public.tar.gz", archive),
        images_lock=write(tmp_path / "images.lock.json", images),
        config=write(tmp_path / "operator.toml", config),
        dotenv=write(
            tmp_path / "fixture-input", b"HF_TOKEN=fixture-hub\nUNRELATED_KEY=discard-fixture\n"
        ),
        role=["hub"],
        output=tmp_path / "packet.age",
    )
    receipt = deploy.pack(args)
    install = argparse.Namespace(
        bundle=args.output,
        bundle_sha256=receipt["bundle_sha256"],
        source_sha256=args.source_sha256,
        images_sha256=args.images_sha256,
        identity=write(tmp_path / "external-identity", b"stub, never supplied to real age"),
        destination=tmp_path / "new",
        role=None,
        rotate=False,
    )
    return args, install, captured, files


def test_roundtrip_no_private_plaintext(packet, monkeypatch):
    args, install, captured, files = packet
    monkeypatch.setattr(
        deploy, "import_for_config", lambda *a, **kw: pytest.fail("unexpected cluster call")
    )
    assert json.loads(captured["plain"])["credentials"] == {"HF_TOKEN": "fixture-hub"}
    assert b"discard-fixture" not in captured["plain"]
    result = deploy.install(install)
    assert result["installed"]
    assert stat.S_IMODE(args.output.stat().st_mode) == 0o600
    dest = install.destination
    assert stat.S_IMODE(dest.stat().st_mode) == 0o700
    assert (dest / "deployment.age").read_bytes() == args.output.read_bytes()
    assert (dest / "SOURCE.tar.gz").read_bytes() == args.source_archive.read_bytes()
    for name, data in files.items():
        assert (dest / "source" / name).read_bytes() == data
    assert os.access(dest / "source/scripts/pmmc", os.X_OK)
    assert (dest / "source" / IMAGE_LOCK).read_bytes() == args.images_lock.read_bytes()
    locks = json.loads((dest / "assets.lock.json").read_bytes())
    assert locks["benchmark_lock_sha256"] == sha256(files[deploy.BENCH_LOCK])
    for f in dest.rglob("*"):
        if f.is_file():
            assert all(
                s not in f.read_bytes()
                for s in (b"fixture-hub", b"private-fixture-context", b"discard-fixture")
            )
    with pytest.raises(FileExistsError):
        deploy.install(install)
    with pytest.raises(FileExistsError):
        deploy.pack(args)


def reviewed_fixture(files):
    model = "fixture-model"
    card = {
        "estimated": False,
        "models": {model: dict.fromkeys(pricing.PRICE_KEYS, "1.25")},
        "rate_limits": {
            model: {
                "provider": "openai",
                "account_scope": "synthetic-scope",
                "model_scope": model,
                "requests_per_minute": 1,
                "tokens_per_minute": 32768,
            }
        },
    }
    import yaml

    rates = yaml.safe_dump(card).encode()
    review = {
        "protocol": "pmmc-release-rate-review-v1",
        "currency": "USD",
        "pricing_basis": "standard_global_first_party_synchronous_before_tax",
        "reviewed_by": "fixture-reviewer",
        "reviewed_date": "2020-01-01",
        "rate_card_file": pricing.CARD,
        "rate_card_sha256": sha256(rates),
        "model_registry_sha256": sha256(files["configs/release/models.yaml"]),
        "pricing_sources": {model: "https://pricing.example.invalid/fixture"},
    }
    return {pricing.CARD: rates, pricing.REVIEW: json_bytes(review)}


def test_ciphertext_policy_install_and_immutable_import(packet, tmp_path, monkeypatch):
    args, install, captured, files = packet
    pair = reviewed_fixture(files)
    args.output = tmp_path / "pricing.age"
    args.rate_card = write(tmp_path / "card-input", pair[pricing.CARD])
    args.provider_policy = write(tmp_path / "review-input", pair[pricing.REVIEW])
    receipt = deploy.pack(args)
    install.bundle, install.bundle_sha256 = args.output, receipt["bundle_sha256"]
    opened = deploy.decrypt(install)
    assert opened.policy == pair and opened.cfg["billing"]["policy_secret"] is True
    assert opened.cfg["native"]["rate_card"] == "/pmmc-inputs/rate-card.yaml"
    assert opened.cfg["billing"]["preferred_mode"] == "finite"
    assert json.loads(captured["plain"])["format"] == deploy.POLICY_FORMAT
    stored = {}
    calls = []

    def kube(argv, data=None, *, kube):
        assert kube and argv[:5] == [
            "kubectl",
            "--context",
            "private-fixture-context",
            "--namespace",
            "trial-ns",
        ]
        calls.append(argv)
        assert not any("synthetic-scope" in word for word in argv)
        if "get" in argv:
            assert argv[argv.index("secret") + 1] == "trial-rates"
            return json_bytes(stored) if stored else b""
        assert argv[-3:] == ["create", "-f", "-"] and not stored
        stored.update(json.loads(data))
        assert stored["immutable"] is True and stored["metadata"] == {
            "name": "trial-rates",
            "namespace": "trial-ns",
        }
        assert {k: base64.b64decode(v, validate=True) for k, v in stored["data"].items()} == pair
        return b""

    monkeypatch.setattr(pricing, "invoke", kube)
    assert deploy.install(install)["pricing_policy"].startswith("bundled") and not calls
    for f in install.destination.rglob("*"):
        if f.is_file():
            assert all(s not in f.read_bytes() for s in (b"synthetic-scope", b"fixture-reviewer"))
    install.private_action = "import-policy"
    assert deploy.run_private_command(install) == {"pricing_policy": "created"}
    assert deploy.run_private_command(install) == {"pricing_policy": "unchanged"}
    for change in (
        {"immutable": False},
        {"data": {**stored["data"], "HF_TOKEN": "eA=="}},
        {"data": {**stored["data"], pricing.CARD: "eA=="}},
    ):
        old = dict(stored)
        stored.update(change)
        with pytest.raises(Error, match="refusing replacement"):
            deploy.run_private_command(install)
        stored.update(old)
    assert [argv for argv in calls if "create" in argv] == [calls[1]]
    envelope = json.loads(captured["plain"])
    envelope["pricing_policy"][pricing.CARD] = deploy.blob(pair[pricing.CARD] + b"\n")
    captured["plain"] = json_bytes(envelope)
    count = len(calls)
    with pytest.raises(Error, match="checksum mismatch"):
        deploy.run_private_command(install)
    assert len(calls) == count


def test_policy_optional_pair_never_implicitly_imported(packet, tmp_path, monkeypatch):
    args, install, _, _ = packet
    args.rate_card = tmp_path / "never-read"
    with pytest.raises(Error, match="together"):
        deploy.pack(args)
    monkeypatch.setattr(pricing, "invoke", lambda *a, **kw: pytest.fail("unexpected cluster call"))
    install.private_action = "import-policy"
    with pytest.raises(Error, match="independently supplied policy"):
        deploy.run_private_command(install)
    assert deploy.install(install)["pricing_policy"] == pricing.ABSENT


@pytest.mark.parametrize(
    "bad",
    [
        "card-hash",
        "registry-hash",
        "card-name",
        "date",
        "budget",
        "url-auth",
        "price-sources",
        "estimated",
        "rate-limit",
        "duplicate",
        "credential",
        "oversized",
        "alias",
    ],
)
def test_invalid_pricing_fails_closed(packet, tmp_path, bad):
    args, _, _, files = packet
    pair = reviewed_fixture(files)
    rates = pair[pricing.CARD]
    review = json.loads(pair[pricing.REVIEW])
    if bad == "card-hash":
        rates += b"\n"
    elif bad == "registry-hash":
        review["model_registry_sha256"] = "0" * 64
    elif bad == "card-name":
        review["rate_card_file"] = "other.yaml"
    elif bad == "date":
        review["reviewed_date"] = "2020-02-31"
    elif bad == "budget":
        review["approval_id"] = "0" * 64
    elif bad == "url-auth":
        review["pricing_sources"]["fixture-model"] = "https://fixture-secret@site.invalid/rates"
    elif bad == "price-sources":
        review["pricing_sources"] = {}
    else:
        rates = {
            "estimated": rates.replace(b"estimated: false", b"estimated: true"),
            "rate-limit": rates.replace(b"32768", b"1"),
            "duplicate": rates + b"estimated: false\n",
            "credential": rates + b"OPENAI_API_KEY: fixture-secret\n",
            "oversized": b"x" * (pricing.POLICY_LIMIT + 1),
            "alias": rates + b"hidden: &hidden fixture-secret\ncopy: *hidden\n",
        }[bad]
        review["rate_card_sha256"] = sha256(rates)
    args.rate_card = write(tmp_path / "rates-test", rates)
    args.provider_policy = write(tmp_path / "review-test", json_bytes(review))
    args.output = tmp_path / "never-created.age"
    with pytest.raises(Error) as err:
        deploy.pack(args)
    assert not args.output.exists() and "fixture-secret" not in str(err.value)


def test_real_age_roundtrip(tmp_path):
    """No user identity, credentials, cluster, or mocked crypto. The keys die with the fixture."""
    if not all(shutil.which(tool) for tool in ("age", "age-keygen")):
        if os.environ.get("PMMC_REQUIRE_REAL_AGE") == "1":
            pytest.fail("CPU release CI requires both age and age-keygen")
        pytest.skip("age CLI not installed")
    with tempfile.TemporaryDirectory(dir=tmp_path) as tmp:
        directory = Path(tmp).resolve()
        identity, wrong = directory / "identity", directory / "wrong"
        for file in (identity, wrong):
            subprocess.run(
                ["age-keygen", "-o", str(file)],
                check=True,
                capture_output=True,
                env={"PATH": os.environ["PATH"], "HOME": str(directory)},
                timeout=15,
            )
        recipient = subprocess.check_output(
            ["age-keygen", "-y", str(identity)],
            text=True,
            timeout=15,
            env={"PATH": os.environ["PATH"], "HOME": str(directory)},
        ).strip()
        files, images, config = public_fixture()
        archive = tar_bytes(files)
        args = argparse.Namespace(
            recipient=recipient,
            source_sha256=sha256(archive),
            images_sha256=sha256(images),
            source_archive=write(directory / "source.tar.gz", archive),
            images_lock=write(directory / "images.lock.json", images),
            config=write(directory / "site.toml", config),
            role=["providers"],
            dotenv=write(directory / "fixture-keys", b"ANTHROPIC_API_KEY=fixture-provider-only\n"),
            output=directory / "packet.age",
        )
        receipt = deploy.pack(args)
        install = argparse.Namespace(
            bundle=args.output,
            bundle_sha256=receipt["bundle_sha256"],
            source_sha256=args.source_sha256,
            images_sha256=args.images_sha256,
            identity=wrong,
            destination=directory / "installed",
            role=None,
            rotate=False,
        )
        cipher = args.output.read_bytes()
        assert cipher.startswith(deploy.HEADER)
        forbidden = (b"private-fixture-context", b"fixture-provider-only")
        assert not any(value in cipher for value in forbidden)
        with pytest.raises(Error, match="age operation failed"):
            deploy.install(install)
        assert not install.destination.exists()
        install.identity = identity
        # Match the outer digest of the corrupted packet: age itself must refuse it.
        corrupt = write(directory / "tampered.age", cipher[:-1] + bytes([cipher[-1] ^ 1]))
        install.bundle, install.bundle_sha256 = corrupt, sha256(corrupt.read_bytes())
        with pytest.raises(Error, match="age operation failed"):
            deploy.install(install)
        assert not install.destination.exists()
        ok = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "scripts.release_host.private_deploy",
                "private",
                "install",
                "--bundle",
                str(args.output),
                "--bundle-sha256",
                receipt["bundle_sha256"],
                "--source-sha256",
                args.source_sha256,
                "--images-sha256",
                args.images_sha256,
                "--identity",
                str(identity),
                "--destination",
                str(install.destination),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            env={"PATH": os.environ["PATH"], "PYTHONPATH": "src", "HOME": str(directory)},
        )
        assert ok.returncode == 0, ok.stderr.decode()
        assert json.loads(ok.stdout)["installed"]
        assert not any(value in ok.stdout + ok.stderr for value in forbidden)
        assert (install.destination / "source" / IMAGE_LOCK).read_bytes() == images
        assert (install.destination / "SOURCE.tar.gz").read_bytes() == archive
        for output in install.destination.rglob("*"):
            if output.is_file():
                assert not any(value in output.read_bytes() for value in forbidden)


def test_pipe_handoff_never_exports_credentials_or_writes_a_file(packet, tmp_path):
    _, install, _, _ = packet
    install.private_action = "config"
    with (tmp_path / "not-a-pipe").open("wb") as regular:
        install.to_fd = regular.fileno()
        with pytest.raises(Error, match="explicit pipe"):
            deploy.run_private_command(install)
    assert (tmp_path / "not-a-pipe").read_bytes() == b""
    read_end, write_end = os.pipe()
    with os.fdopen(read_end, "rb") as reader, os.fdopen(write_end, "wb") as writer:
        install.to_fd = writer.fileno()
        assert deploy.run_private_command(install) == {"config_sent_to_pipe": True}
        writer.close()
        cfg = reader.read()
    assert deploy.parse_config(cfg)["cluster"]["context"] == "private-fixture-context"
    assert b"fixture-hub" not in cfg and b"HF_TOKEN" not in cfg
    assert not install.destination.exists()


def test_import_ciphertext_without_installation(packet, monkeypatch):
    _, install, _, _ = packet
    monkeypatch.setattr(
        deploy, "import_for_config", lambda cfg, values, roles, rotate: {"hub": "unchanged"}
    )
    install.private_action = "bundle-import"
    install.role = ["hub"]
    assert deploy.run_private_command(install) == {"hub": "unchanged"}
    assert not install.destination.exists()


@pytest.mark.parametrize("bad", ["manifest", "mirror", "model", "benchmark", "private-setting"])
def test_lock_mismatch_or_unpinned_asset_refused(bad):
    files, lock, config = public_fixture()
    if bad == "manifest":
        obj = json.loads(lock)
        obj["source_manifest_sha256"] = "a" * 64
        lock = json_bytes(obj)
    elif bad == "mirror":
        config += b'[images]\napp="mirror.example/a@sha256:' + b"4" * 64 + b'"\n'
    elif bad == "private-setting":
        config += b'[operator]\nsecret="fixture-secret"\n'
    else:
        if bad == "model":
            files["configs/release/models.lock.json"] = json_bytes(
                {"entries": [{"commit_sha": "main"}]}
            )
        else:
            bench = json.loads(files[deploy.BENCH_LOCK])
            bench["datasets"]["memlens"]["hf_revision"] = "main"
            files[deploy.BENCH_LOCK] = json_bytes(bench)
        del files["SOURCE.sha256"]
        files["SOURCE.sha256"] = "".join(
            f"{sha256(v)}  {k}\n" for k, v in sorted(files.items())
        ).encode()
        obj = json.loads(lock)
        obj["source_manifest_sha256"] = sha256(files["SOURCE.sha256"])
        lock = json_bytes(obj)
    archive = tar_bytes(files)
    with pytest.raises(Error) as error:
        deploy.release_inputs(archive, sha256(archive), lock, sha256(lock), config)
    assert "fixture-secret" not in str(error.value)


@pytest.mark.parametrize("recipient", [None, "", "-p", "age-secret-key-fixture", "age1" + "q" * 58])
def test_no_public_recipient_no_input_or_archive(tmp_path, recipient):
    with pytest.raises(Error, match="recipient"):
        deploy.pack(argparse.Namespace(recipient=recipient))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["bundle_sha256", "source_sha256", "images_sha256"])
def test_independent_pins_required_before_side_effects(packet, field, monkeypatch):
    _, install, _, _ = packet
    setattr(install, field, "9" * 64)
    monkeypatch.setattr(
        deploy, "import_for_config", lambda *a, **kw: pytest.fail("unexpected cluster call")
    )
    install.role = ["hub"]
    with pytest.raises(Error, match="checksum"):
        deploy.install(install)
    assert not install.destination.exists()


@pytest.mark.parametrize(
    "mutation", ["extra", "unknown_secret", "changed_config", "changed_assets", "duplicate_json"]
)
def test_inner_integrity_and_schema_fail_closed(packet, mutation):
    _, install, captured, _ = packet
    obj = json.loads(captured["plain"])
    if mutation == "extra":
        obj["identity"] = "unexpected"
    elif mutation == "unknown_secret":
        obj["credentials"]["DATABASE_URL"] = "fixture-private"
    elif mutation == "changed_config":
        obj["private_config"]["sha256"] = "bad"
    elif mutation == "changed_assets":
        obj["assets_lock"] = deploy.blob(b"{}")
    captured["plain"] = (
        json_bytes(obj) if mutation != "duplicate_json" else b'{"format":0,"format":0}'
    )
    with pytest.raises(Error):
        deploy.install(install)
    assert not install.destination.exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside", tarfile.REGTYPE),
        ("/outside", tarfile.REGTYPE),
        ("scripts/../outside", tarfile.REGTYPE),
        ("scripts\\outside", tarfile.REGTYPE),
        ("scripts/link", tarfile.SYMTYPE),
        ("scripts/hard", tarfile.LNKTYPE),
        ("scripts/pipe", tarfile.FIFOTYPE),
        ("scripts/pmmc", tarfile.REGTYPE),
        ("SCRIPTS/collision", tarfile.REGTYPE),
        ("scripts/pax", tarfile.XHDTYPE),
        ("scripts/sparse", tarfile.GNUTYPE_SPARSE),
        ("scripts/pmmc/sub", tarfile.REGTYPE),
        ("scripts/empty", tarfile.DIRTYPE),
        (".env", tarfile.REGTYPE),
        ("scripts/secret.key", tarfile.REGTYPE),
    ],
)
def test_archive_attack_paths_and_types(name, kind):
    files, _, _ = public_fixture()
    member = tarfile.TarInfo(name)
    member.type = kind
    if kind in (tarfile.LNKTYPE, tarfile.SYMTYPE):
        member.linkname = "../outside"
    archive = tar_bytes(files, extras=[member])
    with pytest.raises(Error):
        source_files(archive, sha256(archive))


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE]
)
def test_reader_never_extracts_special_member_without_header_hook(monkeypatch, kind):
    monkeypatch.setattr("scripts.release_host.private_source.check_headers", lambda raw: None)
    monkeypatch.setattr(
        tarfile.TarFile, "extractfile", lambda *a: pytest.fail("attempted special file read")
    )
    member = tarfile.TarInfo("scripts/bad")
    member.type = kind
    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        member.linkname = "scripts/target"
    archive = tar_bytes({}, extras=[member])
    with pytest.raises(Error, match="unsupported public archive record"):
        source_files(archive, sha256(archive))


@pytest.mark.parametrize(
    "failure", [None, KeyError("private-fixture"), TypeError("private-fixture")]
)
def test_reader_normalizes_invalid_regular_stream(monkeypatch, failure):
    def broken(*args):
        if failure is not None:
            raise failure
        return None

    monkeypatch.setattr(tarfile.TarFile, "extractfile", broken)
    archive = tar_bytes({"scripts/test": b"fixture"})
    with pytest.raises(Error) as error:
        source_files(archive, sha256(archive))
    assert "private-fixture" not in str(error.value)


def test_source_integrity_size_and_trailer(monkeypatch):
    files, _, _ = public_fixture()
    # Gzip header followed by an invalid deflate block type.
    malformed = bytes.fromhex("1f8b08000000000000ff07")
    with pytest.raises(Error, match="invalid public source archive"):
        source_files(malformed, sha256(malformed))
    archive = tar_bytes(files, compress=False) + b"hidden trailer"
    with pytest.raises(Error, match="trailer"):
        source_files(archive, sha256(archive))
    files["scripts/pmmc"] += b"changed"
    archive = tar_bytes(files)
    with pytest.raises(Error, match="SOURCE.sha256"):
        source_files(archive, sha256(archive))
    monkeypatch.setattr("scripts.release_host.private_source.EXPANDED_LIMIT", 1000)
    with pytest.raises(Error, match="size limit"):
        source_files(archive, sha256(archive))


def test_no_follow_private_files_and_destinations(tmp_path, packet):
    args, install, _, _ = packet
    bad = tmp_path / "bad"
    bad.symlink_to(args.config)
    with pytest.raises(OSError):
        private_io.read_file(bad, 65536, private=True)
    d = tmp_path / "dir"
    d.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        private_io.read_file(d / "operator.toml", 65536, private=True)
    install.destination = d / "escaped"
    with pytest.raises(OSError):
        deploy.install(install)
    assert not (tmp_path / "escaped").exists()
    os.link(args.config, tmp_path / "hard")
    with pytest.raises(Error, match="one link"):
        private_io.read_file(args.config, 65536, private=True)
    install.identity.chmod(0o644)
    with pytest.raises(Error, match="owner-only"):
        deploy.install(install)


def test_install_scopes_one_requested_role(packet, monkeypatch):
    _, install, _, _ = packet
    seen = []
    monkeypatch.setattr(
        deploy, "import_for_config", lambda cfg, vals, roles, **kw: seen.append((vals, roles))
    )
    install.role = ["providers"]
    with pytest.raises(Error, match="unavailable"):
        deploy.install(install)
    assert not install.destination.exists() and not seen
    install.role = ["hub"]
    deploy.install(install)
    assert seen == [({"HF_TOKEN": "fixture-hub"}, ["hub"])]


def cluster_stub(monkeypatch):
    store, calls = {}, []

    def kube(argv, data=None, *, kube):
        assert kube and argv[:5] == ["kubectl", "--context", "test", "--namespace", "trial-ns"]
        assert not any("fixture-value" in word for word in argv)
        calls.append((argv, data))
        if "get" in argv:
            return json_bytes(store[argv[8]]) if argv[8] in store else b""
        body = json.loads(data)
        if "create" in argv:
            name = body["metadata"]["name"]
            assert name not in store and "annotations" not in body["metadata"]
            body["metadata"]["resourceVersion"] = "7"
            store[name] = body
        else:
            assert argv[6:8] == ["patch", "secret"] and argv[-1] == "--patch-file=/dev/stdin"
            obj = store[argv[8]]
            assert body[0] == {"op": "test", "path": "/metadata/resourceVersion", "value": "7"}
            assert body[1]["path"] == "/data"
            obj["data"] = body[1]["value"]
        return b"redacted output not forwarded"

    monkeypatch.setattr(private_secrets, "invoke", kube)
    return store, calls


def test_cli_adapter_scope_repeat_and_rotation(tmp_path, monkeypatch):
    store, calls = cluster_stub(monkeypatch)
    cfg = {
        "profile": "kubernetes",
        "models": {"use_hf_token": True},
        "instance": "trial",
        "cluster": {"context": "test", "namespace": "trial-ns"},
    }
    env = write(
        tmp_path / "synthetic",
        b"export OPENAI_API_KEY='fixture-value-a'\n"
        b"ANTHROPIC_API_KEY=fixture-value-b # comment\nHF_TOKEN=fixture-value-c\n"
        b"PMMC_DATABASE_ADMIN_URL=$(not-executed)\nOTHER=discard-fixture\n",
    )
    assert import_secrets(cfg, env) == {"providers": "created", "hub": "created"}
    assert set(store) == {"trial-providers", "trial-hub"}
    assert set(store["trial-providers"]["data"]) == {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    assert set(store["trial-hub"]["data"]) == {"HF_TOKEN"}
    n = len(calls)
    assert import_secrets(cfg, env) == {"providers": "unchanged", "hub": "unchanged"}
    assert len(calls) == n + 2 and all("get" in argv for argv, _ in calls[n:])
    store["trial-hub"]["data"]["HF_TOKEN"] = base64.b64encode(b"prior").decode()
    with pytest.raises(Error, match="rotation"):
        import_secrets(cfg, env)
    assert not any("patch" in argv for argv, _ in calls)
    assert import_secrets(cfg, env, roles=["hub"], rotate=True) == {"hub": "rotated"}
    cfg["models"]["use_hf_token"] = False
    with pytest.raises(Error, match="disabled"):
        import_secrets(cfg, env, roles=["hub"])


@pytest.mark.parametrize(
    "model,key",
    [
        ("answer-gpt-6-luna", "OPENAI_API_KEY"),
        ("answer-claude-haiku-5.5", "ANTHROPIC_API_KEY"),
    ],
)
def test_provider_only_role_needs_no_unselected_provider_or_hub(tmp_path, monkeypatch, model, key):
    store, _ = cluster_stub(monkeypatch)
    files, images, config = public_fixture()
    archive = tar_bytes(files)
    _, cfg, _ = deploy.release_inputs(archive, sha256(archive), images, sha256(images), config)
    cfg["models"]["use_hf_token"] = False
    cfg["cluster"]["context"] = "test"
    env = write(tmp_path / "selected", f"{key}=fixture-value-one\n".encode())
    assert import_secrets(cfg, env) == {"providers": "created"}
    assert import_secrets(cfg, env) == {"providers": "unchanged"}
    assert set(store) == {"trial-providers"} and set(store["trial-providers"]["data"]) == {key}

    created = []
    looked_up = []

    def kube(argv, obj=None):
        if "get" in argv:
            name = argv[argv.index("secret") + 1]
            looked_up.append(name)
            return " ".join(store[name]["data"])
        assert "create" in argv and obj["kind"] == "Job"
        created.append(obj)

    monkeypatch.setattr(backends, "invoke", kube)
    plan = normalize_plan({"run_id": "demo", "models": [model]})
    host = backends.Kubernetes(
        cfg, "trial", tmp_path, {}, "a" * 64, {"plan": plan, "kind": "run", "variant": "wr"}
    )
    monkeypatch.setattr(host, "check_quota", lambda **kw: None)
    host.start("fixture-job", "control", "demo", "main")
    assert looked_up == ["trial-providers"]
    pod_env = created[0]["spec"]["template"]["spec"]["containers"][0]["env"]
    assert [
        e["valueFrom"]["secretKeyRef"] for e in pod_env if e["name"] in private_secrets.PROVIDER
    ] == [{"name": "trial-providers", "key": key}]
    plan["models"] = ["answer-qwen3.5-9b"]
    host.start("open-job", "control", "demo", "main")
    assert looked_up == ["trial-providers"]
    assert not any(
        e["name"] in private_secrets.PROVIDER
        for e in created[-1]["spec"]["template"]["spec"]["containers"][0]["env"]
    )
    plan["models"] = [
        "answer-gpt-6-luna" if key == "ANTHROPIC_API_KEY" else "answer-claude-haiku-5.5"
    ]
    with pytest.raises(Error, match="missing required keys"):
        host.start("missing-provider", "control", "demo", "main")
    assert looked_up == ["trial-providers", "trial-providers"]
    host.request = {}
    with pytest.raises(Error, match="explicit model selection"):
        host.start("missing-plan", "control", "demo", "main")
    assert looked_up == ["trial-providers", "trial-providers"] and len(created) == 2


@pytest.mark.parametrize("hazard", ["unmanaged", "immutable", "extra-key"])
def test_rotation_rejects_other_owners_and_extra_keys(tmp_path, hazard, monkeypatch):
    store, calls = cluster_stub(monkeypatch)
    values = {"HF_TOKEN": "fixture-value-new"}
    target = dict(context="test", namespace="trial-ns", instance="trial")
    private_secrets.import_values(values, ["hub"], **target)
    existing = store["trial-hub"]
    existing["data"]["HF_TOKEN"] = "YmFk"
    if hazard == "unmanaged":
        existing["metadata"]["labels"] = {}
    elif hazard == "immutable":
        existing["immutable"] = True
    else:
        existing["data"]["ADMIN"] = "YmFk"
    with pytest.raises(Error):
        private_secrets.import_values(values, ["hub"], **target, rotate=True)
    assert not any("patch" in argv for argv, _ in calls)


@pytest.mark.parametrize(
    "raw",
    [
        b"OPENAI_API_KEY=",
        b"HF_TOKEN=$(no-shell)",
        b"HF_TOKEN=fixture\nHF_TOKEN=again",
        b"OTHER='multi\nHF_TOKEN=fixture\nline'",
        b"HF_TOKEN=\xff",
        b"DATABASE_URL=discard",
    ],
)
def test_parse_errors_redact(raw):
    with pytest.raises(Error) as err:
        private_secrets.parse_dotenv(raw)
    assert not any(x in str(err.value) for x in ("fixture", "no-shell", "discard", "again"))


def test_subprocess_stdin_only_and_redaction(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-fixture")
    monkeypatch.setenv("AGE_PASSPHRASE", "ambient-fixture")

    def fail(argv, **kw):
        assert kw["input"] == b"fixture-value"
        assert kw["stderr"] == private_io.subprocess.DEVNULL
        assert all(v != "ambient-fixture" for v in kw["env"].values())
        return argparse.Namespace(returncode=1, stdout=b"fixture-value diagnostic")

    monkeypatch.setattr(private_io.subprocess, "run", fail)
    with pytest.raises(Error, match="kubectl operation failed") as err:
        private_io.invoke(["kubectl", "create", "-f", "-"], b"fixture-value", kube=True)
    assert "fixture" not in str(err.value)
    assert capsys.readouterr() == ("", "")
