"""Offline release controller tests: all container/cluster boundaries are simulated."""

import io
import json
import subprocess
import sys
import tarfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.release_host import backends, cold_start
from scripts.release_host.backends import Compose, Kueue, ReceiptError, receipt
from scripts.release_host.config import ROOT, TARGETS, Error, load_config
from scripts.release_host.evidence import collect, safe_log
from scripts.release_host.quota import headroom, peak, pending_reason
from scripts.release_host.runner import launch, read
from scripts.release_host.workloads import ADMIN, IAM_KEYS, PROVIDER, VERIFY_POINTER, credentials

HASH = "a" * 64


def config(tmp, profile="local-nvidia", **replace):
    raw = (ROOT / "deploy/release" / f"{profile}.toml").read_text()
    for placeholder in ("APP", "STEP8_IAM", "PG16_PGVECTOR", "MINIO"):
        raw = raw.replace(
            f"REPLACE_WITH_PINNED_{placeholder}_IMAGE", f"example.com/pmmc/test@sha256:{HASH}"
        )
    for placeholder in (
        "KUBECTL_CONTEXT",
        "NAMESPACE",
        "LOCALQUEUE",
        "BOUND_RWX_CLAIM",
        "APPROVED_RWO_CLASS",
    ):
        raw = raw.replace(f"REPLACE_WITH_{placeholder}", "fixture")
    raw = raw.replace("/srv/pmmc-release", str(tmp / "mounts"))
    for key, value in replace.items():
        raw = raw.replace(key, value)
    file = tmp / "deploy.toml"
    file.write_text(raw)
    return load_config(file, profile)


def pool(role, stage, paid, *, waiting=False):
    return json.dumps(
        dict(
            protocol="pmmc-release-stage-v1",
            state="waiting_for_workers"
            if waiting
            else "stage_complete"
            if paid
            else "awaiting_budget",
            stage=stage,
            worker_pool=role,
            scope_hash=HASH,
            paper_result=False,
            paid_authorized=paid,
            scoped_jobs=6,
            completed=4 if waiting else 6,
        )
    )


def test_public_cli_offline_plan(tmp_path):
    config(tmp_path)
    result = subprocess.run(
        [
            ROOT / "scripts/pmmc",
            "plan",
            "--config",
            tmp_path / "deploy.toml",
            "--profile",
            "local-nvidia",
            "--stage",
            "main",
        ],
        capture_output=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert [r["role"] for r in plan["units"]["wr-main"][-1]] == ["control", "gpu"]
    assert not (tmp_path / "mounts").exists()
    result = subprocess.run(
        [ROOT / "scripts/pmmc", "plan", "--profile", "eidf"], capture_output=True
    )
    assert result.returncode and b"digest" in result.stderr


@pytest.mark.parametrize(
    "old,new",
    [
        ("example.com/pmmc/test@sha256:" + HASH, "app:latest"),
        ("calibrate = true", "unexpected = true"),
        ('instance = "pmmc-release"', 'instance = "../outside"'),
        ('inputs = "release/inputs"', 'inputs = "release/../inputs"'),
        ('artifacts = "release/artifacts"', 'artifacts = "release/outputs/inside"'),
    ],
)
def test_invalid_config(tmp_path, old, new):
    with pytest.raises(Error):
        config(tmp_path, "eidf", **{old: new})


@pytest.mark.parametrize("profile", ["eidf", "local-nvidia"])
def test_render_roles(tmp_path, profile, monkeypatch):
    cfg = config(tmp_path, profile)
    name = "instance-test"
    for key in PROVIDER:
        monkeypatch.setenv(key, "sensitive-never-log")
    values = credentials(name)
    adapter = (Kueue if profile == "eidf" else Compose)(
        cfg, name, tmp_path, values, HASH, {"variant": "r"}
    )

    def role(r):
        if profile == "eidf":
            manifest = adapter.manifest("test", r, "demo", "main")
            pod = manifest["spec"]["template"]["spec"]
            assert not pod["automountServiceAccountToken"] and manifest["spec"]["suspend"]
            container = pod["containers"][0]
            return container["command"], {e["name"] for e in container["env"]}, container
        container = adapter.manifest(r, "demo", "main")["services"][r]
        return container["entrypoint"], set(container["environment"]), container

    for key in ("control", "gpu"):
        cmd, env, c = role(key)
        assert cmd[-11:] == [
            "pmmc",
            "benchmark",
            "run-exploration",
            "--run-id",
            "demo",
            "--variant",
            "r",
            "--stage",
            "main",
            "--worker-pool",
            key,
        ]
        assert "PMMC_OBJECT_STORE_ADMIN_SECRET_KEY" not in env
        assert (ADMIN in env) == (key == "control")
        assert (set(PROVIDER) <= env) == (key == "control")
        assert "HF_TOKEN" not in env and "PMMC_DATABASE_ONLINE_URL" in env
        if profile == "eidf":
            assert ("nvidia.com/gpu" in c["resources"]["limits"]) == (key == "gpu")
            assert sorted(
                m["mountPath"] for m in c["volumeMounts"] if m["name"] == "shared"
            ) == sorted(TARGETS.values())
        else:
            assert ("deploy" in c) == (key == "gpu")
    assert "--allow-download" in role("models")[0]
    assert "--create-lock" not in role("models")[0]
    assert not set(PROVIDER).intersection(role("prepare")[1])
    assert not set(IAM_KEYS).intersection(role("iam")[1])
    if profile == "eidf":
        for resource in adapter.service_objects("postgres"):
            assert resource["metadata"]["namespace"] == "fixture"
        sts = adapter.service_objects("minio")[2]
        assert (
            sts["metadata"]["labels"].items()
            <= sts["spec"]["template"]["metadata"]["labels"].items()
        )
        assert "resources" in sts["spec"]["template"]["spec"]["containers"][0]
    adapter.approval = None
    assert not set(PROVIDER).intersection(role("control")[1])


class Simulated:
    def __init__(self):
        self.jobs, self.started, self.cancelled = {}, [], []
        self.crash, self.fail, self.mismatch = False, None, False

    def factory(self, cfg, name, private, creds, approval, request):
        self.paid, self.req, self.cfg = bool(approval), request, cfg
        return self

    def reconcile(self):
        pass

    def start(self, job, role, run, stage):
        self.current_run = run
        assert job not in self.jobs, "duplicate remote workload"
        self.jobs[job] = role, stage
        self.started.append((role, stage, self.req["variant"], self.paid))
        if self.crash and role == "gpu":
            self.crash = False
            raise Error("client disconnected after successful submit")

    def state(self, job):
        if job not in self.jobs:
            return "missing"
        if self.jobs[job][0] == self.fail:
            return "failed"
        if self.fail and self.jobs[job][0] == "control":
            return "running"
        return "complete"

    def outcome(self, job, role):
        if role in ("gpu", "control"):
            d = receipt(role, pool(role, self.jobs[job][1], self.paid, waiting=role == "gpu"))
            if self.mismatch and role == "control":
                d["scope_hash"] = "b" * 64
            return d
        if role == "approve":
            return {"approval_id": HASH, "selection_hash": HASH, "limit_usd": "2"}
        if role == "prepare":
            return {
                "run_id": self.current_run,
                "variant": self.req["variant"],
                "selection_hash": HASH,
            }
        return "complete"

    def stop(self, job):
        self.cancelled.append(job)

    def tail(self, job):
        return [{"error_code": "NativeFixtureError"}]

    def health(self):
        return []

    def evidence(self, run, variant):
        return {"schema": "pmmc-host-evidence-v1", "files": {}}


def test_resume_scope_budget_and_all_analysis_variants(tmp_path):
    cfg, sim = config(tmp_path), Simulated()
    root = tmp_path / "state"
    progress = root / "runs/demo/progress.json"

    def go(stage=None, **kw):
        return launch(root, cfg, "demo", stage, factory=sim.factory, sleep=lambda _: None, **kw)

    sim.crash = True
    with pytest.raises(Error, match="disconnected"):
        go("main")
    assert go(resume=True) == 3 and read(progress)["units"]["wr-main"] == "awaiting_budget"
    assert sum(r == "gpu" for r, *_ in sim.started) == 1
    n = len(sim.started)
    assert go(resume=True) == 3 and len(sim.started) == n
    with pytest.raises(Error, match="complete"):
        go("analysis")

    def approve(v):
        req = dict(
            kind="approve",
            variant=v,
            limit_usd="2",
            selection_hash=HASH,
            approver_id="operator",
            rate_card="/pmmc-inputs/card.json",
        )
        assert go("main", variant=v, action=req) == 0

    approve("wr")
    assert go(resume=True) == 0
    assert go("analysis") == 3
    units = read(progress)["units"]
    assert units["wr-analysis"] == "completed" and units["legacy-main"] == "awaiting_budget"
    assert ("control", "main", "w", False) in sim.started
    with pytest.raises(Error, match="analysis"):
        go("baselines")
    for variant in ("w", "r", "legacy"):
        approve(variant)
        assert go("analysis", resume=True, variant=variant) == 0
    assert go("baselines") == 0 and read(progress)["completed_stages"] == [
        "main",
        "analysis",
        "baselines",
    ]
    n = len(sim.started)
    assert go("baselines", resume=True) == 0 and n == len(sim.started)


def test_failure_cancels_peer_and_preserves_diagnostics(tmp_path):
    cfg, sim, root = config(tmp_path), Simulated(), tmp_path / "state"
    sim.fail = "gpu"
    stream = io.StringIO()
    with redirect_stdout(stream):
        assert launch(root, cfg, "demo", "main", factory=sim.factory, sleep=lambda _: None) == 1
    assert sim.cancelled and "SYNC_REVIEW_FILE=" in stream.getvalue()
    secret = read(root / "instances/pmmc-release/identity.json")["secrets"]["PMMC_DATABASE_URL"]
    with tarfile.open(root / "runs/demo/diagnostics.tar.gz") as arc:
        data = b"".join(arc.extractfile(n).read() for n in arc.getnames())
    assert b"NativeFixtureError" in data and secret.encode() not in data
    sim.fail, sim.mismatch = None, True
    assert (
        launch(root, cfg, "demo", None, resume=True, factory=sim.factory, sleep=lambda _: None) == 1
    )


@pytest.mark.parametrize(
    "role,body",
    [
        ("iam", '{"ready":false}'),
        ("models", "{}"),
        ("gpu", "ok"),
        ("control", '{"state":"completed"}'),
        ("control", pool("gpu", "main", False)),
        ("control", pool("control", "main", False, waiting=True)),
    ],
)
def test_incomplete_receipts_fail_closed(role, body):
    with pytest.raises(ReceiptError):
        receipt(role, body)


def test_projected_diagnostics(tmp_path):
    root = tmp_path / "step8"
    campaign = root / "123_exploration_abc"

    def put(p, doc):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc))

    put(
        root / "exploration/demo/wr/pointer.json",
        dict(
            run_id="demo",
            variant="wr",
            campaign_dir=f"outputs/step8/{campaign.name}",
            selection_hash=HASH,
        ),
    )
    put(
        campaign / "manifests/config.json",
        {"metadata": {"exploration_run_id": "demo", "exploration_variant": "wr"}},
    )
    raw = "sensitive-provider-body"
    path = campaign / "attempts" / ("startup-" + "a" * 32)
    put(
        path / "failure-capsule.json",
        {
            "failure": {
                "error_code": "ColdFailure",
                "message": raw,
                "frames": [
                    {
                        "module": "pmmc.runtime.manager",
                        "function": "prepare_runtime",
                        "line": 89,
                        "locals": raw,
                    }
                ],
            }
        },
    )
    (path / "safe-execution.log").write_text('time STEP8 {"completed":4,"query":"' + raw + '"}\n')
    put(
        campaign / "reports/release-main-metrics.json", {"metrics": [{"accuracy": 0.5, "raw": raw}]}
    )
    (campaign / "reports/release-main-gpu.json").symlink_to(path / "failure-capsule.json")
    data = json.dumps(collect(tmp_path, "demo", "wr"))
    assert "ColdFailure" in data and "prepare_runtime" in data and "accuracy" in data
    assert raw not in data and "release-main-gpu.json" not in data
    assert (
        safe_log('File "/workspace/src/pmmc/test.py", line 15, in safe_fn')[0]["frames"][0]["line"]
        == 15
    )


def test_cluster_reader_requires_admission_and_ro_output_only(tmp_path, monkeypatch):
    cfg = config(tmp_path, "eidf")
    kube = Kueue(cfg, "instance", tmp_path, credentials("instance"), None)
    objects = []

    def fake(argv, doc=None, **kw):
        if doc is not None:
            objects.append(doc)
        if "logs" in argv:
            return '{"schema":"pmmc-host-evidence-v1","files":{}}'
        if "get" in argv:
            return "instance"
        return ""

    monkeypatch.setattr(backends, "invoke", fake)
    kube.evidence("demo", "wr")
    (obj,) = objects
    pod = obj["spec"]["template"]["spec"]
    assert obj["spec"]["suspend"] and not pod["automountServiceAccountToken"]
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    c = pod["containers"][0]
    assert c["volumeMounts"] == [
        {
            "name": "shared",
            "subPath": "release/outputs",
            "mountPath": "/workspace/outputs",
            "readOnly": True,
        }
    ]
    assert c["env"] == [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}]


@pytest.mark.parametrize("profile", ["local-nvidia", "eidf"])
def test_backend_command_paths_with_simulated_cli(tmp_path, monkeypatch, profile):
    cfg = config(tmp_path, profile)
    for path in cfg["mounts"].values():
        if profile == "local-nvidia":
            Path(path).mkdir(parents=True)
    jobs, objects, calls = {}, {}, []
    owner = []

    def fake(argv, doc=None, **kw):
        calls.append(argv)
        if profile == "local-nvidia":
            assert argv[:3] == ["docker", "--context", "default"]
            if "compose" in argv:
                assert "--env-file" in argv and argv[argv.index("--env-file") + 1] == "/dev/null"
                owner[:] = [argv[argv.index("--project-name") + 1]]
                if "run" in argv:
                    jobs[argv[argv.index("--name") + 1]] = argv[-1]
                    assert "--no-deps" in argv and argv[-1] in doc["services"]
                else:
                    assert argv[-2:] == ["postgres", "minio"] and "--wait" in argv
            elif "ps" in argv and "name=^/" in str(argv):
                name = next(a[7:-1] for a in argv if a.startswith("name=^/"))
                return name if name in jobs else ""
            elif "inspect" in argv:
                return f"{owner[0]} exited 0"
        else:
            assert argv[:5] == ["kubectl", "--context", "fixture", "-n", "fixture"]
            if "create" in argv or "apply" in argv:
                assert doc["metadata"]["namespace"] == "fixture"
                name = doc["metadata"]["name"]
                objects[name, doc["kind"].lower()] = doc
                if doc["kind"] == "Job":
                    jobs[name] = doc["spec"]["template"]["spec"]["containers"][0]["name"]
                    assert doc["spec"]["suspend"] and doc["spec"]["backoffLimit"] == 0
                return ""
            if "get" in argv:
                kind, name = argv[6:8]
                if kind == "pvc" and name == "fixture":
                    return '{"spec":{"accessModes":["ReadWriteMany"]},"status":{"phase":"Bound"}}'
                if kind in ("pods", "resourcequota"):
                    return '{"items":[]}'
                if kind == "localqueues.kueue.x-k8s.io":
                    return "{}"
                obj = objects.get((name, kind.lower()))
                if "--ignore-not-found" in argv:
                    return obj["metadata"]["labels"][backends.OWNER] if obj else ""
                if kind == "job":
                    return json.dumps(
                        {**obj, "status": {"conditions": [{"type": "Complete", "status": "True"}]}}
                    )
                if kind == "statefulset":
                    return json.dumps(
                        {
                            "metadata": {"generation": 1},
                            "status": {
                                "readyReplicas": 1,
                                "updatedReplicas": 1,
                                "observedGeneration": 1,
                                "updateRevision": "abc",
                                "currentRevision": "abc",
                            },
                        }
                    )
                if kind == "secret":
                    return json.dumps(obj)
            if "wait" in argv:
                return ""
        if "logs" in argv:
            job = (
                argv[-1].removeprefix("job/")
                if profile == "local-nvidia"
                else next(a[4:] for a in argv if a.startswith("job/"))
            )
            if "--limit-bytes=524288" in argv:
                return '{"schema":"pmmc-host-evidence-v1","files":{}}'
            if jobs.get(job) in ("control", "gpu"):
                return pool(jobs[job], "main", False, waiting=jobs[job] == "gpu")
            if jobs.get(job) == "prepare":
                return json.dumps(
                    {
                        "state": "planned",
                        "phase": "exploration",
                        "paper_result": False,
                        "selection": "dev-all",
                        "selection_hash": HASH,
                        "run_id": "contract",
                        "variant": "wr",
                    }
                )
            protocols = {
                "iam": "step8-minio-iam-v1",
                "bootstrap": "step8-bootstrap-v1",
                "models": "step8-model-preparation-v1",
            }
            return json.dumps({"ready": True, "protocol": protocols.get(jobs.get(job))})
        return ""

    monkeypatch.setattr(backends, "invoke", fake)
    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    assert launch(tmp_path / "state", cfg, "contract", "main", sleep=lambda _: None) == 3
    assert {"iam", "bootstrap", "models", "prepare", "pointer", "control", "gpu"} <= set(
        jobs.values()
    )
    assert not any("--force-conflicts" in argv for argv in calls)


@pytest.mark.parametrize("warm", [False, True])
def test_lock_first_model_cache_probe(tmp_path, monkeypatch, warm):
    file = tmp_path / "lock"
    file.touch()
    registry = SimpleNamespace(models=[SimpleNamespace(alias="alias")])
    verified = []

    def probe(*args, **kw):
        assert kw["local_files_only"] and not kw["allow_full_fallback"]
        if not warm:
            raise FileNotFoundError
        return "cache"

    modules = {
        "huggingface_hub.errors": dict(
            LocalEntryNotFoundError=type("LocalEntryNotFoundError", (Exception,), {})
        ),
        "pmmc.experiments.step8_campaign": dict(
            load_benchmark_config=lambda _: SimpleNamespace(model_lock=file, model_registry=file)
        ),
        "pmmc.experiments.step8_model_preparation": dict(
            _check_verification=lambda *a, **kw: verified.append("verified"),
            _read_lock=lambda _: ("lock", HASH),
            _validate_lock=lambda *a: verified.append("pinned"),
        ),
        "pmmc.infrastructure.settings": dict(
            get_settings=lambda: SimpleNamespace(resolve_path=Path, model_cache_path=tmp_path)
        ),
        "pmmc.models.locking": dict(
            prepare_model_cache_audited=probe,
            ModelVerificationPolicy=SimpleNamespace(CACHED_RECEIPT="cached_receipt"),
        ),
        "pmmc.models.registry": dict(load_model_registry=lambda _: registry),
    }
    for name, vals in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**vals))
    monkeypatch.setattr(cold_start.os, "execvp", lambda prog, args: verified.append("native_full"))
    cold_start.models(
        [
            "python",
            "-m",
            "pmmc",
            "benchmark",
            "prepare-models",
            "--config",
            "config",
            "--allow-download",
        ]
    )
    assert verified == ["pinned", "verified" if warm else "native_full"]


@pytest.mark.parametrize("warm", [False, True])
def test_calibrates_only_unverified_physical_pool(monkeypatch, warm):
    calls = []
    settings = SimpleNamespace(resolved_runtime_profile_path="runtime")
    profile = SimpleNamespace(
        services_for_stage=lambda _: [
            SimpleNamespace(metadata={"frozen_replica_topology": {"replica_count": 4}})
        ]
    )

    def validate(*args, **kw):
        if not warm:
            raise ValueError("identity mismatch")
        return {"service"}

    modules = {
        "pmmc.infrastructure.settings": dict(get_settings=lambda: settings),
        "pmmc.runtime.config": dict(load_runtime_profile=lambda _: profile),
        "pmmc.runtime.manager": dict(
            _service_backend_version=None,
            calibrate_runtime=lambda *a, **k: calls.append("reconcile"),
            prepare_runtime=lambda *a, **k: SimpleNamespace(
                plan=SimpleNamespace(stages=[SimpleNamespace(service_ids=["service"])])
            ),
        ),
        "pmmc.runtime.step8_capacity": dict(require_step8_capacity_receipts=validate),
        "pmmc.runtime.step8_capacity_calibration": dict(
            calibrate_step8_capacity=lambda *a, **k: calls.append("measure")
        ),
    }
    for name, vals in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**vals))
    monkeypatch.setattr(cold_start.os, "execvp", lambda *a: calls.append("native_worker"))
    cold_start.gpu(["python", "-m", "pmmc"])
    assert calls == (
        ["reconcile", "native_worker"] if warm else ["reconcile", "measure", "native_worker"]
    )


def test_quota_projection_includes_other_pods_and_cpu_override(tmp_path):
    cfg = config(tmp_path, "eidf")
    assert peak(cfg) == {"cpu": 92, "memory": 768 * 2**30, "nvidia.com/gpu": 8}
    quota = {
        "spec": {"hard": {"requests.cpu": "100"}},
        "status": {"used": {"requests.cpu": "12500m"}},
    }
    (report,) = headroom(cfg, [quota], [], workers=True)
    assert not report["fits"] and report["projected"] == "104.500"
    pod = {"spec": {"containers": [{"name": "postgres", "resources": {"requests": {"cpu": "6"}}}]}}
    (report,) = headroom(cfg, [quota], [pod], workers=True)
    assert report["fits"]
    file = tmp_path / "deploy.toml"
    file.write_text(file.read_text() + "\n[resources.control]\ncpu = 40\n")
    with pytest.raises(Error, match="CPU budget"):
        load_config(file, "eidf")
    assert (
        pending_reason([{"status": "False", "message": "Insufficient quota to admit job"}])
        == "insufficient_quota_or_resources"
    )


def test_pending_worker_deadline_cancels_both_pools(tmp_path, monkeypatch):
    cfg = config(tmp_path, "eidf")
    cfg["cluster"]["admission_timeout_seconds"] = 1
    sim = Simulated()
    original = sim.state

    def state(job):
        if job in sim.jobs:
            if sim.jobs[job][0] == "gpu":
                return "queued:workload-abc:insufficient_quota_or_resources"
            if sim.jobs[job][0] == "control":
                return "running"
        return original(job)

    monkeypatch.setattr(sim, "state", state)
    tick = iter(range(10, 100))
    monkeypatch.setattr("scripts.release_host.runner.time.time", lambda: next(tick))
    assert (
        launch(tmp_path / "state", cfg, "queued", "main", factory=sim.factory, sleep=lambda _: None)
        == 1
    )
    data = read(tmp_path / "state/runs/queued/progress.json")
    assert (
        data["batch"]["gpu"]["reason"]
        == "admission_timeout:queued:workload-abc:insufficient_quota_or_resources"
    )
    assert len(sim.cancelled) == 2


def test_native_quote_is_projected_not_promoted_to_approval():
    quote = {
        "selection_hash": HASH,
        "rate_card_sha256": HASH,
        "approval_created": False,
        "scope": "analysis",
        "provider_response": "never disclose",
    }
    amounts = {"standard_rate_scenario_usd": "1.23", "one_pass_reservation_bound_usd": "10.5"}
    quote.update(pending_scope=amounts, pending_additional_after_preceding_scopes=amounts)
    result = receipt("quote", json.dumps(quote))
    assert result["pending_scope"] == amounts and not result["approval_created"]
    assert "provider_response" not in result
    assert safe_log('{"error_code":"SECRETTOKEN"}') == []


def test_readonly_preflight_rejects_other_variant_budget(tmp_path, monkeypatch):
    import hashlib

    releases = tmp_path / "step8/exploration/trial"
    for v in ("wr", "w"):
        pointer = releases / v / "pointer.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text(json.dumps({"selection_hash": HASH}))
        manifests = tmp_path / v / "manifests"
        manifests.mkdir(parents=True)
        rates = b"reviewed rates"
        (manifests / f"rate-card-{HASH}.json").write_bytes(rates)
        (manifests / f"budget-{HASH}.json").write_text(
            json.dumps(
                {
                    "approval_id": ("a" if v == "wr" else "b") * 64,
                    "selection_hash": HASH,
                    "rate_card_hash": hashlib.sha256(rates).hexdigest(),
                }
            )
        )
    monkeypatch.setitem(
        sys.modules,
        "pmmc.infrastructure.settings",
        SimpleNamespace(get_settings=lambda: SimpleNamespace(outputs_path=tmp_path)),
    )
    monkeypatch.setitem(
        sys.modules,
        "pmmc.experiments.release_inputs",
        SimpleNamespace(
            exploration_campaign_directory=lambda settings, run, variant: tmp_path / variant
        ),
    )
    monkeypatch.delenv("PMMC_STEP8_RATE_CARD_PATH", raising=False)
    monkeypatch.setenv("PMMC_STEP8_BUDGET_APPROVAL_ID", HASH)
    monkeypatch.setattr(sys, "argv", ["python", "trial", "wr"])
    exec(VERIFY_POINTER, {})
    monkeypatch.setattr(sys, "argv", ["python", "trial", "w"])
    with pytest.raises(SystemExit, match="mismatch"):
        exec(VERIFY_POINTER, {})


def test_existing_managed_secret_never_rotated(tmp_path, monkeypatch):
    cfg = config(tmp_path, "eidf")
    backend = Kueue(cfg, "pmmc-owner", tmp_path, {}, None)
    calls = []

    def fake(argv, *a):
        calls.append(argv)
        return (
            "pmmc-owner"
            if "--ignore-not-found" in argv
            else '{"immutable":true,"data":{"k":"different"}}'
        )

    monkeypatch.setattr(backends, "invoke", fake)
    secret = backend.obj("Secret", "pmmc-owner-credentials")
    secret.update(immutable=True, data={"k": "expected"})
    with pytest.raises(Error, match="refusing rotation"):
        backend.ensure(secret)
    assert not any("apply" in cmd or "create" in cmd for cmd in calls)
