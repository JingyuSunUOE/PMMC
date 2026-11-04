"""Offline release controller tests: all container/cluster boundaries are simulated."""

import io
import json
import sys
import tarfile
import tomllib
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.release_host import backends, cold_start
from scripts.release_host.backends import Kueue, ReceiptError, receipt
from scripts.release_host.config import ROOT, Error, load_config, validate_config
from scripts.release_host.evidence import collect, safe_log
from scripts.release_host.quota import headroom, peak, pending_reason
from scripts.release_host.runner import launch, read
from scripts.release_host.workloads import VERIFY_POINTER, credentials

from pmmc.release_plan import normalize_plan, plan_hash

HASH = "a" * 64
pytestmark = pytest.mark.usefixtures("installed_host_lock")


def config(tmp, profile="kubernetes", **replace):
    raw = (ROOT / "deploy/release/kubernetes.toml").read_text()
    raw = raw.replace(
        "[images]",
        "[images]\n"
        + "\n".join(
            f'{k}="example.com/pmmc/{k}@sha256:{HASH}"' for k in ("app", "iam", "postgres", "minio")
        ),
    )
    raw = raw.replace('discovery = "live"', 'discovery = "declared"')
    raw += "\n[capacity]\ngpus=1\ncpus=16\nmemory_gib=96\n"
    for p in ("APP", "STEP8_IAM", "PG16_PGVECTOR", "MINIO"):
        raw = raw.replace(
            f"REPLACE_WITH_PINNED_{p}_IMAGE", f"example.com/pmmc/{p.lower()}@sha256:{HASH}"
        )
    for p in (
        "KUBECTL_CONTEXT",
        "NAMESPACE",
        "LOCALQUEUE",
        "BOUND_RWX_CLAIM",
        "APPROVED_RWO_CLASS",
    ):
        raw = raw.replace(f"REPLACE_WITH_{p}", "fixture")
    for key, value in replace.items():
        raw = raw.replace(key, value)
    path = tmp / "deploy.toml"
    path.write_text(raw)
    return validate_config(tomllib.loads(raw), profile)


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


def test_public_cli_offline_plan(tmp_path, monkeypatch, capsys):
    from scripts.release_host import cli

    cfg = config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *a, **kw: cfg)
    assert (
        cli.main(
            [
                "plan",
                "--config",
                str(tmp_path / "deploy.toml"),
                "--stage",
                "baselines",
                "--benchmark",
                "memgallery",
                "--method",
                "caption_text_rag",
                "--model",
                "answer-qwen3.5-9b",
                "--run-id",
                "demo",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["stages"] == ["wr-baselines"] and plan["selection"]["benchmarks"] == ["memgallery"]
    assert (
        plan["selection"]["methods"] == ["caption_text_rag"] and not plan["selection"]["analyses"]
    )
    assert not plan["models_with_possible_api_cost"] and len(plan["plan_hash"]) == 64
    assert not (tmp_path / "mounts").exists()


@pytest.mark.parametrize(
    "old,new",
    [
        ("example.com/pmmc/app@sha256:" + HASH, "app:latest"),
        ("calibrate = true", "unexpected = true"),
        ('instance = "pmmc-release"', 'instance = "../outside"'),
        ('inputs = "release/inputs"', 'inputs = "release/../inputs"'),
        ('artifacts = "release/artifacts"', 'artifacts = "release/outputs/inside"'),
    ],
)
def test_invalid_config(tmp_path, old, new):
    with pytest.raises(Error):
        config(tmp_path, **{old: new})


def test_render_roles(tmp_path):
    cfg = config(tmp_path)
    plan = normalize_plan({"run_id": "demo", "stage": "main", "models": ["answer-gpt-6-luna"]})
    adapter = Kueue(
        cfg,
        "instance-test",
        tmp_path,
        credentials("instance-test"),
        HASH,
        {"variant": "wr", "kind": "run", "plan": plan},
    )
    for role in ("control", "gpu", "assets", "prepare", "models", "analysis-report"):
        pod = adapter.manifest("example", role, "demo", "main")["spec"]["template"]["spec"]
        assert not pod["automountServiceAccountToken"]
        c = pod["containers"][0]
        env = {x["name"] for x in c["env"]}
        assert "PMMC_OBJECT_STORE_ADMIN_SECRET_KEY" not in env
        assert ("OPENAI_API_KEY" in env) == (role == "control")
        assert "ANTHROPIC_API_KEY" not in env
        assert ("nvidia.com/gpu" in c["resources"]["limits"]) == (role == "gpu")
        if role == "assets":
            assert "HF_TOKEN" not in env
        if role in ("gpu", "control"):
            assert "--worker-pool" in c["command"]
    sts = [adapter.service_objects(role)[2] for role in ("postgres", "minio")]
    assert (
        len(
            {
                s["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"]
                for s in sts
            }
        )
        == 2
    )


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
            return {
                "approval_id": HASH,
                "selection_hash": HASH,
                "limit_usd": None,
                "plan_hash": plan_hash(self.req["plan"]),
            }
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


def test_resume_same_plan_shared_approval_and_no_resubmit(tmp_path):
    cfg, sim = config(tmp_path), Simulated()
    root = tmp_path / "state"
    plan = normalize_plan({"run_id": "demo", "stage": "main"})

    def go(**kw):
        return launch(root, cfg, "demo", plan, factory=sim.factory, sleep=lambda _: None, **kw)

    sim.crash = True
    with pytest.raises(Error, match="disconnected"):
        go()
    assert go(resume=True) == 3 and sum(r == "gpu" for r, *_ in sim.started) == 1
    n = len(sim.started)
    assert go(resume=True) == 3 and len(sim.started) == n
    assert (
        go(
            action=dict(
                kind="approve",
                stage="main",
                variant="wr",
                selection_hash=HASH,
                approver_id="test",
                limit_usd=None,
                uncapped=True,
            )
        )
        == 0
    )
    assert go(resume=True) == 0
    n = len(sim.started)
    assert go(resume=True) == 0 and len(sim.started) == n
    with pytest.raises(Error, match="frozen"):
        launch(
            root,
            cfg,
            "demo",
            normalize_plan({"run_id": "demo", "stage": "baselines"}),
            factory=sim.factory,
        )


def test_failure_cancels_peer_and_preserves_diagnostics(tmp_path):
    cfg, sim, root = config(tmp_path), Simulated(), tmp_path / "state"
    sim.fail = "gpu"
    stream = io.StringIO()
    with redirect_stdout(stream):
        assert (
            launch(
                root,
                cfg,
                "demo",
                normalize_plan({"run_id": "demo"}),
                factory=sim.factory,
                sleep=lambda _: None,
            )
            == 1
        )
    assert sim.cancelled and "SYNC_REVIEW_FILE=" in stream.getvalue()
    secret = read(root / "instances/pmmc-release/identity.json")["secrets"]["PMMC_DATABASE_URL"]
    with tarfile.open(root / "runs/demo/diagnostics.tar.gz") as arc:
        data = b"".join(arc.extractfile(n).read() for n in arc.getnames())
    assert b"NativeFixtureError" in data and secret.encode() not in data
    sim.fail, sim.mismatch = None, True
    assert (
        launch(
            root,
            cfg,
            "demo",
            normalize_plan({"run_id": "demo"}),
            resume=True,
            factory=sim.factory,
            sleep=lambda _: None,
        )
        == 1
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
    cfg = config(tmp_path, **{"# queue =": "queue ="})
    kube = Kueue(cfg, "instance", tmp_path, credentials("instance"), None)
    objects = []

    def fake(argv, doc=None, **kw):
        if doc is not None:
            objects.append(doc)
        if "logs" in argv:
            return '{"schema":"pmmc-host-evidence-v1","files":{}}'
        if "get" in argv:
            return json.dumps({"metadata": {"labels": {backends.OWNER: "instance"}}})
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


# The real K8s reconcile/readback contract lives in the kind acceptance test.


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
    cfg = config(tmp_path)
    used = peak(cfg)
    assert used["cpu"] <= 16 and used["memory"] <= 96 * 2**30 and used["nvidia.com/gpu"] == 1
    quota = {
        "spec": {"hard": {"requests.cpu": str(used["cpu"])}},
        "status": {"used": {"requests.cpu": "500m"}},
    }
    (report,) = headroom(cfg, [quota], [], workers=True)
    assert not report["fits"]
    pod = {"spec": {"containers": [{"name": "postgres", "resources": {"requests": {"cpu": "6"}}}]}}
    (report,) = headroom(cfg, [quota], [pod], workers=True)
    assert report["fits"]
    file = tmp_path / "deploy.toml"
    file.write_text(file.read_text() + "\n[resources.control]\ncpu = 40\n")
    with pytest.raises(Error):
        load_config(file, "kubernetes")
    assert (
        pending_reason([{"status": "False", "message": "Insufficient quota to admit job"}])
        == "insufficient_quota_or_resources"
    )


def test_pending_worker_deadline_cancels_both_pools(tmp_path, monkeypatch):
    cfg = config(tmp_path)
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
        launch(
            tmp_path / "state",
            cfg,
            "queued",
            normalize_plan({"run_id": "queued"}),
            factory=sim.factory,
            sleep=lambda _: None,
        )
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
    cfg = config(tmp_path)
    backend = Kueue(cfg, "pmmc-owner", tmp_path, {}, None)
    calls = []

    def fake(argv, *a):
        calls.append(argv)
        return json.dumps(
            dict(
                metadata={"labels": {backends.OWNER: "pmmc-owner"}},
                immutable=True,
                data={"k": "different"},
            )
        )

    monkeypatch.setattr(backends, "invoke", fake)
    secret = backend.obj("Secret", "pmmc-owner-credentials")
    secret.update(immutable=True, data={"k": "expected"})
    with pytest.raises(Error, match="refusing rotation"):
        backend.ensure(secret)
    assert not any("apply" in cmd or "create" in cmd for cmd in calls)
