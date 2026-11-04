"""Offline generic Kubernetes rendering, admission and restart contracts."""

import copy
import json
import os
import re

import pytest
from scripts.release_host import backends
from scripts.release_host.backends import CREDENTIAL_UID, OWNER, Kubernetes
from scripts.release_host.config import ROOT, Error, load_config
from scripts.release_host.discovery import from_snapshot, pod_projection, projected_pods, requests
from scripts.release_host.quota import peak
from scripts.release_host.volume_layout import directory, main as initialize_layout

pytestmark = pytest.mark.usefixtures("installed_host_lock")


def config(tmp, *, overrides=None, extra="", gpus=4):
    data = (ROOT / "deploy/release/kubernetes.toml").read_text()
    data = re.sub(r"REPLACE_WITH_[A-Z0-9_]+", "fixture", data)
    data += f"\n[capacity]\ngpus = {gpus}\ncpus = 16\nmemory_gib = 96\n"
    data = data.replace("[cluster]", "[cluster]\n" + extra)
    file = tmp / "deploy.toml"
    file.write_text(data)
    return load_config(file, "kubernetes", resource_overrides=overrides)


def adapter(cfg):
    plan = {"variants": ["wr"], "models": ["answer-qwen3.5-9b"], "run_id": "trial"}
    return Kubernetes(
        cfg,
        cfg["instance"],
        None,
        {"test": "dummy"},
        None,
        {"kind": "run", "variant": "wr", "plan": plan},
    )


@pytest.mark.parametrize("gpus", [1, 2])
@pytest.mark.parametrize("queued", [False, True])
def test_generic_render_and_privileges(tmp_path, gpus, queued):
    cfg = config(
        tmp_path,
        overrides={"gpus": str(gpus), "cpus": "12", "memory": "64Gi"},
        extra='queue = "optional"' if queued else "",
    )
    backend = adapter(cfg)
    for role in (
        "assets",
        "models",
        "prepare",
        "prepare-capacity",
        "control",
        "gpu",
        "analysis-report",
    ):
        job = backend.manifest("job", role, "trial", "main")
        pod = job["spec"]["template"]["spec"]
        c = pod["containers"][0]
        env = {item["name"]: item.get("value") for item in c["env"]}
        assert json.loads(env["PMMC_RELEASE_REQUEST"]) == backend.request["plan"]
        assert env["PMMC_RELEASE_VARIANT"] == "wr" and env["PMMC_RELEASE_STAGE"] == "main"
        assert pod["automountServiceAccountToken"] is False
        assert job["spec"]["suspend"] is queued
        assert ("kueue.x-k8s.io/queue-name" in job["metadata"]["labels"]) is queued
        if role in ("gpu", "prepare-capacity"):
            assert (
                c["resources"]["requests"]["nvidia.com/gpu"]
                == env["PMMC_EXPECTED_GPU_COUNT"]
                == str(gpus)
            )
            assert json.loads(env["PMMC_RELEASE_TOPOLOGY"])["gpus"] == gpus
            assert pod["nodeSelector"] == {"kubernetes.io/os": "linux"}
        else:
            assert "nvidia.com/gpu" not in c["resources"]["limits"]
            assert env["NVIDIA_VISIBLE_DEVICES"] == "void"
        mounts = {m["mountPath"]: m for m in c["volumeMounts"]}
        if role == "analysis-report":
            assert all("valueFrom" not in item for item in c["env"])
            assert mounts["/workspace/outputs"]["readOnly"]
            assert not mounts["/workspace/outputs/step8/exploration/trial/reports"]["readOnly"]
            assert "/pmmc-inputs" not in mounts and "/pmmc-cache" not in mounts
            setup = pod["initContainers"][0]
            assert setup["name"] == "report-directory"
            assert setup["env"] == [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}]
        elif role == "assets":
            assert set(mounts) == {"/pmmc-stage"} and not mounts["/pmmc-stage"]["readOnly"]
            assert env["PMMC_BENCHMARK_STAGING_ROOT"] == "/pmmc-stage"
        else:
            assert mounts["/pmmc-inputs"]["readOnly"]
    assert peak(cfg)["cpu"] <= 12 and peak(cfg)["memory"] <= 64 * 2**30
    assert peak(cfg, roles=["bootstrap"])["nvidia.com/gpu"] == 0


def test_rwo_pins_every_consumer(tmp_path):
    cfg = config(tmp_path, extra='node_hostname = "gpu-01"')
    cfg["cluster"]["shared_access_mode"] = "ReadWriteOnce"
    b = adapter(cfg)
    for obj in (
        b.service_objects("postgres")[-1],
        b.manifest("report", "analysis-report", "trial", "main"),
    ):
        assert obj["spec"]["template"]["spec"]["nodeSelector"] == {
            "kubernetes.io/os": "linux",
            "kubernetes.io/hostname": "gpu-01",
        }
    file = tmp_path / "deploy.toml"
    file.write_text(
        file.read_text()
        .replace('node_hostname = "gpu-01"', "")
        .replace('shared_access_mode = "ReadWriteMany"', 'shared_access_mode = "ReadWriteOnce"')
    )
    with pytest.raises(Error, match="node_hostname"):
        load_config(file, "kubernetes")


class Cluster:
    """API simulation: persist creations with apiserver identities; no real clients."""

    def __init__(self):
        self.objects = {}
        self.writes = []

    def __call__(self, argv, document=None, **kwargs):
        if document is not None:
            key = document["kind"], document["metadata"]["name"]
            self.writes.append(key)
            obj = copy.deepcopy(document)
            obj["metadata"].update(uid="uid-" + key[1], resourceVersion="1", generation=1)
            if key[0] == "PersistentVolumeClaim":
                obj["status"] = {"phase": "Pending"}  # WFFC: no consumer yet.
            self.objects[key] = obj
            return ""
        if "wait" in argv:
            return ""
        assert "get" in argv and not any("kueue.x-k8s.io" in x for x in argv)
        kind, name = argv[argv.index("get") + 1 :][:2]
        if kind == "pvc" and name == "fixture":
            return json.dumps(
                {"spec": {"accessModes": ["ReadWriteMany"]}, "status": {"phase": "Pending"}}
            )
        if kind in ("pods", "resourcequota"):
            return '{"items":[]}'
        value = self.objects.get((kind, name))
        return json.dumps(value) if value else ""


def test_restart_and_wffc_claims(tmp_path, monkeypatch):
    b, api = adapter(config(tmp_path)), Cluster()
    monkeypatch.setattr(backends, "invoke", api)
    monkeypatch.setattr(b, "wait_services", lambda: None)
    b.reconcile()
    uid = api.objects["Secret", b.name + "-credentials"]["metadata"]["uid"]
    pvcs = [obj for (kind, _), obj in api.objects.items() if kind == "PersistentVolumeClaim"]
    assert len(pvcs) == 2
    assert {p["metadata"]["name"] for p in pvcs} == {b.name + "-postgres", b.name + "-minio"}
    assert all(p["metadata"]["annotations"][CREDENTIAL_UID] == uid for p in pvcs)
    assert all(p["spec"]["accessModes"] == ["ReadWriteOnce"] for p in pvcs)
    api.writes.clear()
    b.reconcile()
    assert {k for k, _ in api.writes} == {"Service", "StatefulSet"}
    assert len([row for row in b.health() if "pvc" in row]) == 2


@pytest.mark.parametrize(
    "damage, message",
    [
        ("unlabelled", "foreign"),
        ("wrong-uid", "credential identity"),
        ("missing-secret", "without its original"),
        ("changed-secret", "rotation"),
        ("storage", "storage contract"),
        ("missing-pvc", "missing its data PVC"),
    ],
)
def test_restart_rejects_foreign_state_before_writes(tmp_path, monkeypatch, damage, message):
    b, api = adapter(config(tmp_path)), Cluster()
    monkeypatch.setattr(backends, "invoke", api)
    monkeypatch.setattr(b, "wait_services", lambda: None)
    b.reconcile()
    pvc = api.objects["PersistentVolumeClaim", b.name + "-minio"]
    secret = ("Secret", b.name + "-credentials")
    if damage == "unlabelled":
        pvc["metadata"]["labels"].pop(OWNER)
    elif damage == "wrong-uid":
        pvc["metadata"]["annotations"][CREDENTIAL_UID] = "other"
    elif damage == "missing-secret":
        del api.objects[secret]
    elif damage == "changed-secret":
        api.objects[secret]["data"] = {}
    elif damage == "storage":
        pvc["spec"]["resources"]["requests"]["storage"] = "100Gi"
    else:
        del api.objects["PersistentVolumeClaim", b.name + "-minio"]
    api.writes.clear()
    with pytest.raises(Error, match=message):
        b.reconcile()
    assert not api.writes


def node(name, gpu):
    return {
        "metadata": {
            "name": name,
            "labels": {"kubernetes.io/hostname": name, "kubernetes.io/os": "linux"},
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "allocatable": {"cpu": "20", "memory": "100Gi", "nvidia.com/gpu": str(gpu)},
        },
    }


def test_live_bounds_one_node_no_aggregation(tmp_path):
    cfg = config(tmp_path)
    pods = [
        {
            "metadata": {"namespace": "other"},
            "spec": {
                "nodeName": "node-b",
                "containers": [
                    {
                        "resources": {
                            "requests": {"cpu": "10", "memory": "52Gi", "nvidia.com/gpu": "2"}
                        }
                    }
                ],
            },
        }
    ]
    quota = {"requests.nvidia.com/gpu": "3", "requests.cpu": "15"}
    quotas = [
        {
            "spec": {"hard": quota},
            "status": {
                "hard": quota,
                "used": {"requests.nvidia.com/gpu": "1", "requests.cpu": "4500m"},
            },
        }
    ]
    resolved = from_snapshot(cfg, [node("node-a", 1), node("node-b", 4)], pods, quotas, None)
    assert resolved["topology"] == {"gpus": 2, "cpus": 10, "memory_gib": 48, "answer_tp": 2}
    assert resolved["resource_policy"]["node_hostname"] == "node-b"
    pod = adapter(resolved).manifest("job", "gpu", "trial", "main")["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"kubernetes.io/os": "linux", "kubernetes.io/hostname": "node-b"}
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert cfg["topology"]["gpus"] == 4
    manual = config(tmp_path, overrides={"gpus": "4"})
    with pytest.raises(Error, match="single node"):
        from_snapshot(manual, [node("node-a", 2), node("node-b", 2)], [], [], None)


def test_discovery_fail_closed_and_read_only_phase(tmp_path, monkeypatch):
    cfg = config(tmp_path)

    def denied(*args):
        raise Error("permission_denied")

    monkeypatch.setattr(backends, "invoke", denied)
    with pytest.raises(Error, match="grant read-only discovery"):
        backends.resolve_kubernetes_resources(cfg)
    tiny = node("cpu", 0)
    tiny["status"]["allocatable"] = {"cpu": "1", "memory": "3Gi"}
    cfg = from_snapshot(cfg, [tiny], [], [], {"analysis-report"})
    assert cfg["topology"]["gpus"] == 0 and cfg["resources"]["analysis-report"]["cpu"] == 1
    api = Cluster()
    monkeypatch.setattr(backends, "invoke", api)
    adapter(cfg).reconcile_reader()
    assert not api.writes


def test_minimal_manual_reports_do_not_reserve_inactive_services(tmp_path, monkeypatch):
    config(tmp_path, overrides={"cpus": "1", "memory": "3Gi"})
    path = tmp_path / "deploy.toml"
    path.write_text(path.read_text().replace('discovery = "live"', 'discovery = "declared"'))
    cfg = load_config(path, "kubernetes", resource_overrides={"cpus": "1", "memory": "3Gi"})
    monkeypatch.setattr(backends, "invoke", lambda *_: pytest.fail("unexpected discovery"))
    resolved = backends.resolve_kubernetes_resources(cfg, roles={"analysis-report"})
    assert resolved["topology"] == {"gpus": 0, "answer_tp": 1, "cpus": 1, "memory_gib": 3}
    with pytest.raises(Error, match="declared capacity"):
        backends.resolve_kubernetes_resources(cfg, roles={"gpu", "control"})


def test_selectors_and_tolerations_match_rendering(tmp_path):
    cfg = config(
        tmp_path,
        extra='node_selector = {"example.org/pool" = "research"}\n'
        'tolerations = [{key="nvidia.com/gpu", operator="Exists", effect="NoSchedule"}]',
    )
    gpu = node("eligible", 2)
    gpu["spec"] = {"taints": [{"key": "nvidia.com/gpu", "value": "yes", "effect": "NoSchedule"}]}
    gpu["metadata"]["labels"]["example.org/pool"] = "research"
    rejected = node("other-pool", 4)
    resolved = from_snapshot(cfg, [rejected, gpu], [], [], None)
    b = adapter(resolved)
    for obj in (b.service_objects("postgres")[-1], b.manifest("job", "gpu", "trial", "main")):
        pod = obj["spec"]["template"]["spec"]
        assert pod["nodeSelector"]["example.org/pool"] == "research"
        assert pod["tolerations"] == cfg["cluster"]["tolerations"]
    gpu["spec"]["taints"][0]["effect"] = "NoExecute"
    with pytest.raises(Error, match="no eligible single node"):
        from_snapshot(cfg, [rejected, gpu], [], [], None)
    cfg["cluster"]["tolerations"] = [
        {"key": "nvidia.com/gpu", "operator": "Equal", "value": "wrong"}
    ]
    with pytest.raises(Error, match="no eligible single node"):
        from_snapshot(cfg, [gpu], [], [], None)


def test_managed_shared_claim_and_layout_once(tmp_path, monkeypatch):
    cfg = config(
        tmp_path,
        extra='create_shared_pvc = true\nshared_storage_class = "rwx"\n'
        'shared_storage_gib = 100\nimage_pull_secrets = ["mirror-pull"]',
    )
    b, api = adapter(cfg), Cluster()
    monkeypatch.setattr(backends, "invoke", api)
    monkeypatch.setattr(b, "wait_services", lambda: None)
    b.reconcile()
    pvc = api.objects["PersistentVolumeClaim", "fixture"]
    assert pvc["spec"] == {
        "accessModes": ["ReadWriteMany"],
        "storageClassName": "rwx",
        "resources": {"requests": {"storage": "100Gi"}},
    }
    assert pvc["metadata"]["annotations"][CREDENTIAL_UID] == "uid-" + b.name + "-credentials"
    (layout,) = [obj for (k, _), obj in api.objects.items() if k == "Job"]
    pod = layout["spec"]["template"]["spec"]
    assert not pod["automountServiceAccountToken"]
    assert "initContainers" not in pod
    for c in pod["containers"]:
        assert c["env"] == [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}]
    assert pod["containers"][0]["securityContext"]["runAsUser"] == cfg["identity"]["uid"]
    for obj in (
        layout,
        b.service_objects("minio")[-1],
        b.service_objects("postgres")[-1],
        b.manifest("trial", "assets", "trial", "main"),
        b.manifest("trial", "analysis-report", "trial", "main"),
    ):
        pod = obj["spec"]["template"]["spec"]
        assert pod["imagePullSecrets"] == [{"name": "mirror-pull"}]
        assert pod["securityContext"] == {
            "runAsUser": cfg["identity"]["uid"],
            "runAsGroup": cfg["identity"]["gid"],
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        }
        for c in (*pod.get("initContainers", []), *pod["containers"]):
            sc = c["securityContext"]
            assert sc.get("runAsUser", cfg["identity"]["uid"]) == cfg["identity"]["uid"]
            assert not sc["allowPrivilegeEscalation"] and sc["capabilities"] == {"drop": ["ALL"]}
    api.writes.clear()
    b.reconcile()
    assert not any(k in ("Job", "Secret", "PersistentVolumeClaim") for k, _ in api.writes)
    del pvc["metadata"]["labels"][OWNER]
    api.writes.clear()
    with pytest.raises(Error, match="foreign"):
        b.reconcile()
    assert not api.writes


def test_layout_no_follow_no_recursive_repair(tmp_path, monkeypatch):
    relative = "release/outputs"
    directory(tmp_path, relative)
    leaf = tmp_path / relative
    marker = leaf / "keep"
    marker.write_bytes(b"unchanged")
    marker.chmod(0o400)
    directory(tmp_path, relative)
    assert marker.read_bytes() == b"unchanged" and marker.stat().st_mode & 0o777 == 0o400
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    for mode in ("create", "report"):
        with pytest.raises(PermissionError, match="non-root"):
            initialize_layout([mode, "10001", "10001", "unused"])
    (leaf / "symlink").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        directory(tmp_path, relative + "/symlink/new")
    assert not (tmp_path / "new").exists()


def test_verified_release_images_required(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    assert cfg["images"]["app"].startswith("ghcr.io/fixture/app@sha256:")
    file = tmp_path / "deploy.toml"
    bad = file.read_text().replace("[images]", '[images]\napp = "repo/app@sha256:' + "0" * 64 + '"')
    file.write_text(bad)
    with pytest.raises(Error, match="IMAGE_LOCK_SHA256"):
        load_config(file, "kubernetes")
    monkeypatch.delenv("PMMC_IMAGE_LOCK_SHA256")
    with pytest.raises(Error, match="IMAGE_LOCK_SHA256"):
        config(tmp_path)


def test_read_only_inventory_projection_accounts_for_in_place_resize():
    assert not any(path in pod_projection() for path in ("env", "command", "volume", "annotations"))
    row = (
        "\t".join(
            [
                "other",
                "",
                "",
                "Running",
                "gpu",
                '{"cpu":"100m"}',
                "",
                'c={"requests":{"cpu":"1"}}|',
                "",
                'c={"cpu":"2"}={"cpu":"3"}|',
                "",
            ]
        )
        + "#"
    )
    (pod,) = projected_pods(row)
    assert str(requests(pod["spec"], "cpu")) == "3.100"
    with pytest.raises(ValueError, match="incomplete"):
        projected_pods(row[:-1])


def test_planning_fallback_only_for_gpu_absence(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    roles = {"assets", "prepare", "models", "control", "gpu"}
    nodes = [node("cpu", 0)]
    calls = []

    def read(cmd):
        calls.append(cmd)
        if "nodes" in cmd:
            return json.dumps({"items": nodes})
        if "resourcequota" in cmd:
            return '{"items":[]}'
        assert "pods" in cmd
        return ""

    monkeypatch.setattr(backends, "invoke", read)
    resolve = backends.resolve_planning_resources
    with pytest.raises(Error, match="no available single-node GPU"):
        resolve(cfg, roles=roles)
    calls.clear()
    cpu = resolve(cfg, roles=roles, allow_cpu_fallback=True)
    assert len(calls) == 3  # Retry uses the same observation, never another live inventory.
    assert cpu["resource_policy"]["cpu_planning_only"] is True and cpu["topology"]["gpus"] == 0
    b = adapter(cpu)
    for role in ("gpu", "prepare-capacity"):
        with pytest.raises(Error, match="GPU"):
            b.manifest("job", role, "trial", "main")
    assert (
        "nvidia.com/gpu"
        not in b.manifest("job", "control", "trial", "main")["spec"]["template"]["spec"][
            "containers"
        ][0]["resources"]["requests"]
    )
    nodes[:] = [node("worker", 1)]
    gpu = resolve(cfg, roles=roles, allow_cpu_fallback=True)
    assert gpu["topology"]["gpus"] == 1 and not gpu["resource_policy"].get("cpu_planning_only")
    with pytest.raises(Error, match="exceeds declared single-node capacity"):
        resolve(config(tmp_path, overrides={"gpus": 2}), roles=roles, allow_cpu_fallback=True)
    nodes.clear()
    with pytest.raises(Error, match="no ready schedulable Linux node"):
        resolve(cfg, roles=roles, allow_cpu_fallback=True)
    monkeypatch.setattr(backends, "invoke", lambda _: (_ for _ in ()).throw(Error("denied")))
    with pytest.raises(Error, match="grant read-only discovery"):
        resolve(cfg, roles=roles, allow_cpu_fallback=True)


def test_declared_zero_gpu_plan(tmp_path):
    config(tmp_path, gpus=0)
    path = tmp_path / "deploy.toml"
    path.write_text(path.read_text().replace('discovery = "live"', 'discovery = "declared"'))
    cfg = load_config(path, "kubernetes")
    with pytest.raises(Error, match="does not permit a GPU"):
        backends.resolve_planning_resources(cfg)
    assert (
        backends.resolve_planning_resources(cfg, allow_cpu_fallback=True)["topology"]["gpus"] == 0
    )


def test_control_admission_does_not_require_a_gpu_peer(tmp_path, monkeypatch):
    b = adapter(config(tmp_path))
    hard = {"requests.nvidia.com/gpu":"0"}
    def read(cmd):
        quotas = [{"spec":{"hard":hard}, "status":{"hard":hard,"used":hard}}]
        return json.dumps({"items":quotas if "resourcequota" in cmd else []})
    monkeypatch.setattr(backends, "invoke", read)
    b.check_quota(roles=("control",))
    assert b.quota["peak"]["nvidia.com/gpu"] == 0
    with pytest.raises(Error, match="insufficient namespace quota"):
        b.check_quota(roles=("gpu",))


def test_private_rate_files_are_role_scoped_without_authorizing(tmp_path):
    config(tmp_path)
    file = tmp_path / "deploy.toml"
    enabled = (
        file.read_text()
        .replace("policy_secret = false", "policy_secret = true")
        .replace('preferred_mode = "finite"', 'preferred_mode = "uncapped"')
    )
    file.write_text(enabled)
    cfg = load_config(file, "kubernetes")
    b = adapter(cfg)
    b.request.update(
        rate_card=cfg["native"]["rate_card"],
        selection_hash="a" * 64,
        approver_id="operator",
        limit_usd="1",
    )
    for role in (
        "iam",
        "assets",
        "models",
        "prepare",
        "prepare-capacity",
        "quote",
        "approve",
        "pointer",
        "control",
        "gpu",
        "analysis-report",
        "bootstrap",
    ):
        job = b.manifest("trial", role, "trial", "main")
        pod = job["spec"]["template"]["spec"]
        c = pod["containers"][0]
        mounts = {m["mountPath"]: m for m in c["volumeMounts"] if m["name"] == "rates"}
        rate_users = role in ("quote", "approve")
        assert bool(mounts) == rate_users
        if rate_users:
            (vol,) = [v for v in pod["volumes"] if v["name"] == "rates"]
            assert vol["secret"]["secretName"] == b.name + "-rates"
            assert {item["key"] for item in vol["secret"]["items"]} == {
                "rate-card.yaml",
                "provider-policy.json",
            }
            assert set(mounts) == {
                "/pmmc-inputs/rate-card.yaml",
                "/pmmc-inputs/provider-policy.json",
            }
            assert all(m["readOnly"] for m in mounts.values())
        assert "--uncapped" not in c["command"]
        assert all(
            e["name"]
            not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PMMC_STEP8_BUDGET_APPROVAL_ID")
            for e in c["env"]
        )
    b.approval = "a" * 64
    paid = b.manifest("trial", "control", "trial", "main")["spec"]["template"]["spec"]
    assert {v["name"] for v in paid["volumes"]} >= {"rates"}
    for role in ("pointer", "gpu"):
        c = b.manifest("trial", role, "trial", "main")["spec"]["template"]["spec"]["containers"][0]
        assert "PMMC_STEP8_RATE_CARD_PATH" not in {e["name"] for e in c["env"]}
    b.request["rate_card"] = "/pmmc-inputs/different"
    with pytest.raises(Error, match="conflicts"):
        b.manifest("trial", "approve", "trial", "main")
    file.write_text(enabled.replace("[native]", '[native]\nrate_card = "/pmmc-inputs/different"'))
    with pytest.raises(Error, match="native.rate_card"):
        load_config(file, "kubernetes")


def test_open_control_submits_without_private_billing_or_provider_secrets(tmp_path, monkeypatch):
    b = adapter(config(tmp_path))
    b.cfg["billing"]["policy_secret"] = True
    b.creds = {}
    submitted = []
    def invoke(argv, document=None, **kwargs):
        if "get" in argv:
            assert "secret" not in argv
            return '{"items":[]}'
        assert "create" in argv
        submitted.append(document)
    monkeypatch.setattr(backends, "invoke", invoke)
    b.start("trial", "control", "trial", "main")
    (job,) = submitted
    pod = job["spec"]["template"]["spec"]
    assert "rates" not in {v["name"] for v in pod["volumes"]}
    assert not {"OPENAI_API_KEY","ANTHROPIC_API_KEY","PMMC_STEP8_RATE_CARD_PATH",
                "PMMC_STEP8_BUDGET_APPROVAL_ID"}.intersection(e["name"] for e in pod["containers"][0]["env"])
