"""Kubernetes adapter. Never forward CLI output into host diagnostics."""

import base64
import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

from .config import RATE_FILES, SECRET_RATE_PATH, STAGES, TARGETS, Error, check, identifier
from .evidence import safe_log
from .quota import headroom, peak, pending_reason, quantity
from .workloads import (
    GPU_ROLES,
    IAM_KEYS,
    PROVIDER,
    RESOURCES,
    command,
    environment,
    secret_refs,
)

OWNER = "pmmc-release-instance"
IAM_DIR = "/var/run/secrets/pmmc-step8-minio"
CREDENTIAL_UID = "release.pmmc.io/credential-uid"
RATE_ROLES = ("quote", "approve", "control")


def policy_mount(cfg, role):
    return cfg.get("billing", {}).get("policy_secret") and role in RATE_ROLES


def resolve_kubernetes_resources(cfg, *, roles=None):
    from .discovery import resolve

    return resolve(cfg, invoke, roles)


def release_env(role, cfg, name, approval, request, stage):
    env = environment(role, cfg, name, approval)
    if request and "plan" in request:
        env["PMMC_RELEASE_REQUEST"] = json.dumps(request["plan"], separators=(",", ":"))
        env["PMMC_RELEASE_STAGE"] = stage
        env["PMMC_RELEASE_VARIANT"] = request.get("variant", request["plan"]["variants"][0])
    if role in (*GPU_ROLES, "prepare"):
        env["PMMC_RELEASE_TOPOLOGY"] = json.dumps(native_topology(cfg), sort_keys=True)
    if role in GPU_ROLES:
        env["PMMC_RELEASE_MEASURE"] = "1" if cfg["models"]["calibrate"] else "0"
    else:
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
    if role == "assets":
        env["PMMC_BENCHMARK_STAGING_ROOT"] = "/pmmc-stage"
    if cfg.get("billing", {}).get("policy_secret"):
        if role in RATE_ROLES:
            env["PMMC_STEP8_RATE_CARD_PATH"] = SECRET_RATE_PATH
        else:
            # Native pointer/GPU workers verify the archived card, not this Secret.
            env.pop("PMMC_STEP8_RATE_CARD_PATH", None)
    return env


def native_topology(cfg):
    return {**cfg["topology"], "workers": cfg["resources"]["control"]["cpu"],
            "source_profile": cfg["native"]["runtime_profile"]}


def role_refs(role, cfg, approval, request):
    if role == "analysis-report":
        return {}
    return secret_refs(role, cfg, approval, (request or {}).get("plan"))


def report_subdir(run):
    return "step8/exploration/" + identifier(run) + "/reports"


def mount_options(cfg, role, run):
    if role == "assets":
        yield cfg["mounts"]["inputs"], "/pmmc-stage", False
        return
    for key, dest in TARGETS.items():
        if role == "analysis-report" and key != "outputs":
            continue
        if key == "inputs" and policy_mount(cfg, role):
            # Sibling mounts, not nested mounts inside a read-only PVC. Secret
            # files need no placeholder or mutation in operator-owned inputs.
            subtrees = {"benchmarks", *(p.removeprefix(dest + "/").split("/", 1)[0]
                        for p in cfg["native"].values() if p.startswith(dest + "/"))} - set(RATE_FILES)
            for subdir in sorted(subtrees):
                yield cfg["mounts"][key] + "/" + subdir, dest + "/" + subdir, True
            continue
        yield cfg["mounts"][key], dest, (
            role in ("analysis-report", "pointer") or key == "inputs" and role != "assets"
        )
    if role == "analysis-report":
        subdir = "/" + report_subdir(run)
        yield cfg["mounts"]["outputs"] + subdir, TARGETS["outputs"] + subdir, False


def invoke(argv, document=None, *, timeout=120, stderr=False):
    try:
        result = subprocess.run(
            argv,
            input="" if document is None else json.dumps(document),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise Error(f"{argv[0]} unavailable or timed out; inspect the backend") from exc
    if result.returncode:
        message = result.stderr.lower()
        cause = next(
            (
                code
                for needles, code in (
                    (("exceeded quota", "insufficient quota"), "insufficient_quota"),
                    (("forbidden", "permission denied"), "permission_denied"),
                    (("unauthorized", "authentication required"), "authentication_required"),
                    (("timed out", "deadline exceeded"), "timeout"),
                    (("not found", "notfound"), "not_found"),
                    (("conflict", "already exists"), "resource_conflict"),
                    (("unhealthy", "not healthy"), "service_not_healthy"),
                )
                if any(term in message for term in needles)
            ),
            "inspect_private_backend",
        )
        raise Error(f"{argv[0]} operation failed (exit {result.returncode}; {cause})")
    return (result.stdout + (result.stderr if stderr else "")).strip()


class ReceiptError(Error):
    pass


def receipt(role, line):
    if role == "pointer":
        return "complete"
    try:
        data = json.loads(line)
    except ValueError as exc:
        raise ReceiptError(f"missing final JSON receipt from {role}") from exc
    if not isinstance(data, dict):
        raise ReceiptError(f"invalid {role} receipt")
    if role == "analysis-report":
        if (data.get("protocol") != "pmmc-analysis-readout-v2"
                or data.get("paper_result") is not False
                or data.get("state") not in ("complete","missing_dependencies")
                or not re.fullmatch(r"[a-f0-9]{64}",str(data.get("plan_hash","")))
                or not isinstance(data.get("missing_units"),list)
                or any(not re.fullmatch(r"(wr|w|r|legacy)-main|wr-analysis",u) for u in data["missing_units"])):
            raise ReceiptError("invalid analysis source receipt")
        return {k:data[k] for k in ("plan_hash","state","missing_units")}
    if role in {"iam", "bootstrap", "models","assets","prepare-capacity"}:
        protocols = {
            "iam": ("step8-minio-iam-v1",),
            "bootstrap": ("step8-bootstrap-v1",),
            "models": ("step8-model-preparation-v1", "release-locked-cache-hit-v1"),
            "assets": ("release-assets-v2",),
            "prepare-capacity": ("release-capacity-v2",),
        }
        if data.get("ready") is not True or data.get("protocol") not in protocols[role]:
            raise ReceiptError(f"{role} did not report ready")
        return "complete"
    if role == "prepare":
        if (
            data.get("state") != "planned"
            or data.get("paper_result") is not False
            or data.get("phase") != "exploration"
            or data.get("selection") != "dev-all"
            or not isinstance(data.get("selection_hash"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", data["selection_hash"])
        ):
            raise ReceiptError("invalid prepare receipt")
        return {k: data.get(k) for k in ("run_id", "variant", "selection_hash", "release_plan_hash","selected")}
    if role in {"quote", "approve"}:
        fields = ["selection_hash", "rate_card_sha256"]
        if role == "approve":
            fields.append("approval_id")
        if not all(
            isinstance(data.get(k), str) and re.fullmatch(r"[a-f0-9]{64}", data[k]) for k in fields
        ):
            raise ReceiptError(f"invalid {role} identity")
        summary = {k: data[k] for k in fields}
        amount = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
        if role == "approve":
            if not (data.get("budget_mode") == "uncapped" and data.get("limit_usd") is None
                    or isinstance(data.get("limit_usd"),str) and re.fullmatch(amount,data["limit_usd"])):
                raise ReceiptError("invalid approval amount")
            summary["limit_usd"] = data["limit_usd"]
        if "plan_hash" in data:
            if not re.fullmatch(r"[a-f0-9]{64}",str(data["plan_hash"])):
                raise ReceiptError("invalid plan approval")
            summary["plan_hash"] = data["plan_hash"]
        if role == "quote":
            if data.get("approval_created") is not False or data.get("scope") not in STAGES:
                raise ReceiptError("invalid quote receipt")
            summary.update(scope=data["scope"], approval_created=False)
            for section in ("pending_scope", "pending_additional_after_preceding_scopes"):
                estimates = data.get(section)
                if section == "pending_additional_after_preceding_scopes" and data.get("schema")=="release-plan-api-estimate-v2":
                    continue
                summary[section] = {}
                for key in ("standard_rate_scenario_usd", "one_pass_reservation_bound_usd"):
                    if not isinstance(estimates, dict) or key not in estimates:
                        raise ReceiptError("missing quote estimates")
                    v = estimates[key]
                    if v is not None and not (isinstance(v, str) and re.fullmatch(amount, v)):
                        raise ReceiptError("invalid quote amount")
                    summary[section][key] = v
        return summary
    state = data.get("state")
    n, done = data.get("scoped_jobs"), data.get("completed")
    if not (
        data.get("protocol") == "pmmc-release-stage-v1"
        and data.get("worker_pool") == role
        and data.get("paper_result") is False
        and data.get("stage") in STAGES
        and type(n) is int
        and type(done) is int
        and 0 <= done <= n
        and n > 0
        and type(data.get("paid_authorized")) is bool
        and isinstance(data.get("scope_hash"), str)
        and re.fullmatch(r"[0-9a-f]{64}", data["scope_hash"])
        and (
            (state == "awaiting_budget" and done == n and not data["paid_authorized"])
            or (state == "stage_complete" and done == n)
            or (state == "waiting_for_workers" and role == "gpu" and done < n)
        )
    ):
        raise ReceiptError(f"invalid {role} stage receipt")
    return {
        k: data[k]
        for k in ("state", "stage", "scope_hash", "scoped_jobs", "completed", "paid_authorized")
    }




class Kubernetes:
    def __init__(self, cfg, name, private, creds, approval, request=None):
        self.cfg, self.name, self.creds, self.approval = cfg, name, creds, approval
        self.request = request
        self.cluster = cfg["cluster"]
        self.kube = [
            "kubectl",
            "--context",
            self.cluster["context"],
            "-n",
            self.cluster["namespace"],
        ]

    def obj(self, kind, name, spec=None, *, queued=False):
        version = {"Job": "batch/v1", "StatefulSet": "apps/v1"}.get(kind, "v1")
        labels = {OWNER: self.name}
        if queued and self.cluster.get("queue"):
            labels["kueue.x-k8s.io/queue-name"] = self.cluster["queue"]
        obj = {
            "apiVersion": version,
            "kind": kind,
            "metadata": {"name": name, "namespace": self.cluster["namespace"], "labels": labels},
        }
        if spec is not None:
            obj["spec"] = spec
        return obj

    def existing(self, kind, name):
        raw = invoke([*self.kube, "get", kind, name, "--ignore-not-found", "-o", "json"])
        if not raw:
            return None
        obj = json.loads(raw)
        meta = obj.get("metadata", {})
        check(
            meta.get("labels", {}).get(OWNER) == self.name and not meta.get("deletionTimestamp"),
            "refusing to adopt a foreign or terminating cluster resource",
        )
        return obj

    def owned(self, kind, name):
        return self.existing(kind, name) is not None

    def validate_existing(self, obj, stored):
        kind = obj["kind"]
        if kind == "Secret":
            check(
                stored.get("immutable") is True and stored.get("data") == obj["data"],
                "existing managed Secret differs from host identity; refusing rotation",
            )
        if kind == "PersistentVolumeClaim":
            existing, expected = stored["spec"], obj["spec"]
            uid = obj["metadata"].get("annotations", {}).get(CREDENTIAL_UID)
            check(
                uid and stored["metadata"].get("annotations", {}).get(CREDENTIAL_UID) == uid,
                "existing data PVC belongs to a different credential identity",
            )
            check(
                existing.get("accessModes") == expected["accessModes"]
                and existing.get("volumeMode", "Filesystem") == "Filesystem"
                and existing.get("storageClassName") == expected["storageClassName"]
                and quantity(existing["resources"]["requests"]["storage"])
                == quantity(expected["resources"]["requests"]["storage"])
                and stored.get("status", {}).get("phase") in ("Pending", "Bound"),
                "existing PVC differs from the requested storage contract",
            )

    def ensure(self, obj):
        kind, name = obj["kind"], obj["metadata"]["name"]
        stored = self.existing(kind, name)
        if stored is None:
            invoke([*self.kube, "create", "-f", "-"], obj)
        else:
            self.validate_existing(obj, stored)
            if kind not in {"Secret", "PersistentVolumeClaim"}:
                obj["metadata"]["resourceVersion"] = stored["metadata"]["resourceVersion"]
                invoke(
                    [
                        *self.kube,
                        "apply",
                        "--server-side",
                        "--field-manager=pmmc-release",
                        "-f",
                        "-",
                    ],
                    obj,
                )

    def secret(self, key):
        if key in PROVIDER:
            return self.cfg["instance"] + "-providers"
        if key == "HF_TOKEN":
            return self.cfg["instance"] + "-hub"
        return self.name + "-credentials"

    def refs(self, role):
        return [
            {"name": k, "valueFrom": {"secretKeyRef": {"name": self.secret(src), "key": src}}}
            for k, src in role_refs(role, self.cfg, self.approval, self.request).items()
        ]

    def resources(self, role):
        r = self.cfg["resources"][role]
        spec = {"cpu": str(r["cpu"]), "memory": f"{r['memory_gib']}Gi"}
        if role in GPU_ROLES:
            check(self.cfg["topology"]["gpus"] > 0, "GPU workload requires a nonzero GPU total")
            spec["nvidia.com/gpu"] = str(self.cfg["topology"]["gpus"])
        return {"requests": spec, "limits": spec}

    def service_objects(self, role):
        name = f"{self.name}-{role}"
        pg = role == "postgres"
        selector = {"app": name}
        port = 5432 if pg else 9000
        pvc = self.obj(
            "PersistentVolumeClaim",
            name,
            {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.cluster["storage_class"],
                "resources": {"requests": {"storage": f"{self.cluster[role + '_storage_gib']}Gi"}},
            },
        )
        svc = self.obj(
            "Service",
            name,
            {"clusterIP": "None", "selector": selector, "ports": [{"name": role, "port": port}]},
        )
        pod = {
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": name}}],
        }
        self.place(pod)
        container = {
            "name": role,
            "image": self.cfg["images"][role],
            "env": [*self.refs(role), {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}],
            "ports": [{"containerPort": port}],
            "resources": self.resources(role),
            "volumeMounts": [
                {"name": "data", "mountPath": "/var/lib/postgresql/data" if pg else "/data"}
            ],
        }
        if pg:
            container["env"].extend(
                {"name": k, "value": v}
                for k, v in {
                    "POSTGRES_USER": "pmmc_owner",
                    "POSTGRES_DB": "pmmc",
                    "PGDATA": "/var/lib/postgresql/data/pgdata",
                    "PMMC_RELEASE_BOOTSTRAP_ROLES": "1",
                }.items()
            )
            container["readinessProbe"] = {
                "exec": {"command": ["pg_isready", "-U", "pmmc_owner", "-d", "pmmc"]}
            }
        else:
            container.update(
                args=["server", "/data"],
                readinessProbe={"httpGet": {"path": "/minio/health/ready", "port": port}},
            )
        pod["containers"] = [container]
        sts = self.obj(
            "StatefulSet",
            name,
            {
                "serviceName": name,
                "replicas": 1,
                "selector": {"matchLabels": selector},
                "template": {"metadata": {}, "spec": pod},
            },
            queued=True,
        )
        sts["spec"]["template"]["metadata"]["labels"] = {**sts["metadata"]["labels"], **selector}
        return pvc, svc, sts

    def place(self, pod, *, workload=False):
        if pulls := self.cluster.get("image_pull_secrets"):
            pod["imagePullSecrets"] = [{"name": secret} for secret in pulls]
        selected = self.cfg["resource_policy"].get("node_hostname") if workload else None
        pod["nodeSelector"] = {"kubernetes.io/os": "linux", **self.cluster.get("node_selector", {})}
        if host := selected or self.cluster.get("node_hostname"):
            pod["nodeSelector"]["kubernetes.io/hostname"] = host
        if tolerations := self.cluster.get("tolerations"):
            pod["tolerations"] = [dict(t) for t in tolerations]

    def reconcile_reader(self):
        claim = json.loads(
            invoke(
                [
                    *self.kube,
                    "get",
                    "pvc",
                    self.cluster["shared_pvc"],
                    "-o",
                    "json",
                ]
            )
        )
        check(
            self.cluster["shared_access_mode"] in claim["spec"]["accessModes"]
            and claim["spec"].get("volumeMode", "Filesystem") == "Filesystem"
            and not claim.get("metadata", {}).get("deletionTimestamp")
            and claim.get("status", {}).get("phase") in ("Pending", "Bound"),
            "shared PVC does not satisfy the configured access mode, or is unavailable",
        )
        meta = claim.get("metadata", {})
        check(
            meta.get("labels", {}).get(OWNER) in (None, self.name),
            "shared PVC is owned by a different instance",
        )
        if self.cluster.get("create_shared_pvc"):
            check(
                meta.get("labels", {}).get(OWNER) == self.name
                and meta.get("annotations", {}).get(CREDENTIAL_UID),
                "foreign managed shared PVC",
            )
            expected = self.shared_claim()
            expected["metadata"]["annotations"] = {
                CREDENTIAL_UID: meta["annotations"][CREDENTIAL_UID]
            }
            self.validate_existing(expected, claim)

    def shared_claim(self):
        return self.obj(
            "PersistentVolumeClaim",
            self.cluster["shared_pvc"],
            {
                "accessModes": [self.cluster["shared_access_mode"]],
                "storageClassName": self.cluster["shared_storage_class"],
                "resources": {"requests": {"storage": f"{self.cluster['shared_storage_gib']}Gi"}},
            },
        )

    def ensure_layout(self, claim):
        uid = claim["metadata"]["uid"]
        identity = self.cfg["identity"]
        paths = sorted(self.cfg["mounts"].values())
        args = [str(identity["uid"]), str(identity["gid"]), json.dumps(paths)]
        script = Path(__file__).with_name("volume_layout.py").read_text()
        image = self.cfg["images"]["app"]
        digest = hashlib.sha256(json.dumps([uid, args, image, script]).encode()).hexdigest()
        name = self.name + "-layout-" + digest[:12]
        container = {
            "name": "verify-layout",
            "image": image,
            "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}],
            "command": ["python", "-B", "-c", script, "verify", *args],
            "resources": self.resources("bootstrap"),
            "volumeMounts": [{"name": "shared", "mountPath": "/layout"}],
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": identity["uid"],
                "runAsGroup": identity["gid"],
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            },
        }
        init = {
            **container,
            "name": "create-layout",
            "command": ["python", "-B", "-c", script, "create", *args],
            "securityContext": {
                **container["securityContext"],
                "runAsUser": 0,
                "runAsGroup": 0,
                "runAsNonRoot": False,
                "capabilities": {"drop": ["ALL"], "add": ["CHOWN"]},
            },
        }
        pod = {
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "restartPolicy": "Never",
            "volumes": [
                {
                    "name": "shared",
                    "persistentVolumeClaim": {"claimName": self.cluster["shared_pvc"]},
                }
            ],
            "initContainers": [init],
            "containers": [container],
        }
        self.place(pod, workload=True)
        job = self.obj(
            "Job",
            name,
            {
                "backoffLimit": 0,
                "suspend": bool(self.cluster.get("queue")),
                "template": {"metadata": {"labels": {OWNER: self.name}}, "spec": pod},
            },
            queued=True,
        )
        job["metadata"]["annotations"] = {"release.pmmc.io/layout": digest}
        current = self.existing("Job", name)
        if current is None:
            invoke([*self.kube, "create", "-f", "-"], job)
        else:
            check(
                current["metadata"].get("annotations", {}).get("release.pmmc.io/layout") == digest,
                "managed storage layout Job identity mismatch",
            )
            check(
                not current.get("status", {}).get("failed"),
                "managed storage layout previously failed",
            )
        seconds = self.cluster["admission_timeout_seconds"]
        invoke(
            [
                *self.kube,
                "wait",
                "--for=condition=complete",
                f"--timeout={seconds}s",
                "job/" + name,
            ],
            timeout=seconds + 15,
        )

    def reconcile(self):
        managed = self.cluster.get("create_shared_pvc")
        if not managed:
            self.reconcile_reader()
        self.check_quota(roles=("bootstrap",) if managed else ())
        secret = self.obj("Secret", self.secret(IAM_KEYS[0]))
        secret.update(
            immutable=True,
            type="Opaque",
            data={k: base64.b64encode(v.encode()).decode() for k, v in self.creds.items()},
        )
        stored = self.existing("Secret", secret["metadata"]["name"])
        services = [obj for role in ("postgres", "minio") for obj in self.service_objects(role)]
        if managed:
            services.insert(0, self.shared_claim())
        if stored is not None:
            self.validate_existing(secret, stored)
        # Validate all pre-existing objects before any writes. Never create a
        # fresh credential identity over old data, even if the owner label matches.
        found = set()
        for obj in services:
            existing = self.existing(obj["kind"], obj["metadata"]["name"])
            if existing is not None:
                found.add((obj["kind"], obj["metadata"]["name"]))
                check(stored is not None, "service exists without its original credential Secret")
                if obj["kind"] == "PersistentVolumeClaim":
                    obj["metadata"]["annotations"] = {CREDENTIAL_UID: stored["metadata"]["uid"]}
                self.validate_existing(obj, existing)
        for kind, name in found:
            if kind == "StatefulSet":
                check(
                    ("PersistentVolumeClaim", name) in found,
                    "existing service StatefulSet is missing its data PVC",
                )
        self.ensure(secret)
        stored = self.existing("Secret", secret["metadata"]["name"])
        check(stored is not None and stored["metadata"].get("uid"), "credential Secret UID missing")
        self.validate_existing(secret, stored)
        for obj in services:
            if obj["kind"] == "PersistentVolumeClaim":
                obj["metadata"]["annotations"] = {CREDENTIAL_UID: stored["metadata"]["uid"]}
            self.ensure(obj)
            if (
                managed
                and obj["kind"] == "PersistentVolumeClaim"
                and obj["metadata"]["name"] == self.cluster["shared_pvc"]
            ):
                claim = self.existing("PersistentVolumeClaim", self.cluster["shared_pvc"])
                check(claim is not None, "managed shared PVC disappeared")
                self.validate_existing(obj, claim)
                self.ensure_layout(claim)
        self.wait_services()

    def check_queue(self):
        if not self.cluster.get("queue"):
            return
        queue = json.loads(
            invoke(
                [
                    *self.kube,
                    "get",
                    "localqueues.kueue.x-k8s.io",
                    self.cluster["queue"],
                    "-o",
                    "json",
                ]
            )
        )
        check(
            not any(
                c["type"] == "Active" and c["status"] == "False"
                for c in queue.get("status", {}).get("conditions", [])
            ),
            "Kueue LocalQueue inactive",
        )

    def check_quota(self, *, roles=None, services=True):
        from .discovery import held_service

        self.check_queue()
        paired = (
            roles
            and set(roles).intersection(("gpu", "control"))
            and (self.request or {}).get("kind") == "run"
        )
        sizing_roles = ("control", "gpu") if paired else roles
        wanted = peak(self.cfg, roles=sizing_roles, services=services)
        bounds = self.cfg["topology"]
        check(
            wanted["cpu"] <= bounds["cpus"] and wanted["memory"] <= bounds["memory_gib"] * 2**30,
            "phase exceeds frozen CPU or memory ceiling",
        )
        if wanted["nvidia.com/gpu"]:
            check(
                all(
                    self.cfg["resources"][role]["cpu"] >= bounds["gpus"]
                    for role in set(sizing_roles or GPU_ROLES).intersection(GPU_ROLES)
                ),
                "insufficient frozen CPU allocation for requested GPU replicas",
            )
        if paired and roles == ("control",):
            roles = sizing_roles
        quotas = json.loads(invoke([*self.kube, "get", "resourcequota", "-o", "json"]))["items"]
        pods = json.loads(
            invoke([*self.kube, "get", "pods", "-l", f"{OWNER}={self.name}", "-o", "json"])
        )["items"]
        try:
            limits = headroom(self.cfg, quotas, [p for p in pods if held_service(self.cfg, p)],
                              roles=roles, services=services)
        except ValueError as exc:
            raise Error("namespace ResourceQuota not comparable; inspect before launch") from exc
        self.quota = {
            "queue": self.cluster.get("queue"),
            "peak": peak(self.cfg, roles=roles, services=services),
            "node_hostname": self.cfg["resource_policy"].get(
                "node_hostname", self.cluster.get("node_hostname")
            ),
            "resource_source": self.cfg["resource_policy"]["source"],
            "limits": limits,
        }
        print("PMMC_ADMISSION " + json.dumps(self.quota), flush=True)
        for limit in limits:
            check(
                limit["fits"],
                f"insufficient namespace quota for {limit['resource']}: "
                f"projected {limit['projected']}, hard {limit['hard']}",
            )

    def wait_services(self):
        until = time.monotonic() + self.cluster["admission_timeout_seconds"]
        while True:
            pending = []
            for role in ("postgres", "minio"):
                name = f"{self.name}-{role}"
                sts = json.loads(invoke([*self.kube, "get", "statefulset", name, "-o", "json"]))
                status = sts.get("status", {})
                if not (
                    status.get("readyReplicas") == status.get("updatedReplicas") == 1
                    and status.get("observedGeneration") == sts["metadata"]["generation"]
                    and status.get("updateRevision")
                    and status["updateRevision"] == status.get("currentRevision")
                ):
                    pending.append(f"{role}:{self.workload_reason(name)}")
            if not pending:
                return
            print("PMMC_SERVICE_WAIT " + ", ".join(pending), flush=True)
            left = until - time.monotonic()
            check(left > 0, "service admission/readiness timeout: " + ", ".join(pending))
            time.sleep(min(15, left))

    def workload_reason(self, name):
        if not self.cluster.get("queue"):
            return "awaiting_pod_or_storage"
        workloads = json.loads(
            invoke([*self.kube, "get", "workloads.kueue.x-k8s.io", "-o", "json"])
        )
        for workload in workloads["items"]:
            owners = workload["metadata"].get("ownerReferences", [])
            if any(o.get("name") == name for o in owners):
                ident = workload["metadata"].get("name", "")
                if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,252}", ident):
                    ident = "unknown"
                return (
                    ident + ":" + pending_reason(workload.get("status", {}).get("conditions", []))
                )
        return "awaiting_workload_or_pod"

    def manifest(self, job, role, run, stage):
        if policy_mount(self.cfg, role) and role in ("quote", "approve"):
            check((self.request or {}).get("rate_card") == SECRET_RATE_PATH,
                  "rate-card override conflicts with the configured billing Secret")
        volumes, mounts = [], []
        pod = {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "volumes": volumes,
        }
        self.place(pod, workload=True)
        env = release_env(role, self.cfg, self.name, self.approval, self.request, stage)
        container = {
            "name": role,
            "image": self.cfg["images"]["iam" if role == "iam" else "app"],
            "command": command(role, self.cfg, run, stage, self.request),
            "workingDir": "/workspace",
            "env": [*({"name": k, "value": v} for k, v in env.items()), *self.refs(role)],
            "volumeMounts": mounts,
            "resources": self.resources(role),
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "capabilities": {"drop": ["ALL"]},
            },
        }
        pod["containers"] = [container]
        ident = self.cfg["identity"]
        pod["securityContext"] = {
            "runAsUser": ident["uid"],
            "runAsGroup": ident["gid"],
            "runAsNonRoot": True,
        }
        if role == "iam":
            pod["securityContext"]["fsGroup"] = ident["gid"]
            container["securityContext"]["readOnlyRootFilesystem"] = True
            volumes.extend(
                [
                    {
                        "name": "iam",
                        "secret": {
                            "secretName": self.secret(IAM_KEYS[0]),
                            "defaultMode": 0o440,
                            "items": [{"key": k, "path": k} for k in IAM_KEYS],
                        },
                    },
                    {"name": "scratch", "emptyDir": {"medium": "Memory", "sizeLimit": "64Mi"}},
                ]
            )
            mounts.extend(
                [
                    {"name": "iam", "mountPath": IAM_DIR, "readOnly": True},
                    {"name": "scratch", "mountPath": "/tmp"},
                ]
            )
        else:
            volumes.append(
                {
                    "name": "shared",
                    "persistentVolumeClaim": {"claimName": self.cluster["shared_pvc"]},
                }
            )
            mounts.extend(
                {
                    "name": "shared",
                    "subPath": source,
                    "mountPath": dest,
                    "readOnly": readonly,
                }
                for source, dest, readonly in mount_options(self.cfg, role, run)
            )
            if role == "analysis-report":
                pod["initContainers"] = [
                    {
                        "name": "report-directory",
                        "image": container["image"],
                        "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}],
                        "command": [
                            "python",
                            "-B",
                            "-c",
                            Path(__file__).with_name("volume_layout.py").read_text(),
                            "report",
                            report_subdir(run),
                        ],
                        "resources": container["resources"],
                        "securityContext": {
                            **container["securityContext"],
                            "readOnlyRootFilesystem": True,
                        },
                        "volumeMounts": [
                            {
                                "name": "shared",
                                "mountPath": "/layout",
                                "subPath": self.cfg["mounts"]["outputs"],
                            }
                        ],
                    }
                ]
                container["securityContext"]["readOnlyRootFilesystem"] = True
                volumes.append({"name": "tmp", "emptyDir": {}})
                mounts.append({"name": "tmp", "mountPath": "/tmp"})
        if policy_mount(self.cfg, role):
            volumes.append({"name": "rates", "secret": {
                "secretName": self.cfg["instance"] + "-rates", "defaultMode": 0o444,
                "items": [{"key": k, "path": k} for k in RATE_FILES],
            }})
            mounts.extend({"name": "rates", "subPath": k, "mountPath": TARGETS["inputs"] + "/" + k,
                           "readOnly": True} for k in RATE_FILES)
        if role in GPU_ROLES:
            for name, path, spec in [
                ("scratch", "/pmmc-scratch", {}),
                (
                    "shm",
                    "/dev/shm",
                    {
                        "medium": "Memory",
                        "sizeLimit": f"{max(1, min(64, self.cfg['resources'][role]['memory_gib'] // 4))}Gi",
                    },
                ),
            ]:
                volumes.append({"name": name, "emptyDir": spec})
                mounts.append({"name": name, "mountPath": path})
        obj = self.obj(
            "Job",
            job,
            {
                "suspend": bool(self.cluster.get("queue")),
                "backoffLimit": 0,
                "activeDeadlineSeconds": 604800,
                "template": {"metadata": {}, "spec": pod},
            },
            queued=True,
        )
        obj["spec"]["template"]["metadata"]["labels"] = obj["metadata"]["labels"]
        return obj

    def start(self, job, role, run, stage):
        # Count active peers through ResourceQuota.used. Never charge CPU-only
        # stages for a future hypothetical GPU worker.
        self.check_quota(roles=(role,), services=role != "analysis-report")
        external_keys = {}
        for key in role_refs(role, self.cfg, self.approval, self.request).values():
            if key in (*PROVIDER, "HF_TOKEN"):
                external_keys.setdefault(self.secret(key), set()).add(key)
        if policy_mount(self.cfg, role):
            external_keys[self.cfg["instance"] + "-rates"] = set(RATE_FILES)
        for external, required in external_keys.items():
            keys = invoke(
                [
                    *self.kube,
                    "get",
                    "secret",
                    external,
                    "-o",
                    "go-template={{range $key,$val := .data}}{{$key}} {{end}}",
                ]
            ).split()
            check(
                required <= set(keys),
                "external workload Secret is missing required keys",
            )
        invoke([*self.kube, "create", "-f", "-"], self.manifest(job, role, run, stage))

    def state(self, job):
        if not self.owned("job", job):
            return "missing"
        data = json.loads(invoke([*self.kube, "get", "job", job, "-o", "json"]))
        conditions = {
            v["type"] for v in data.get("status", {}).get("conditions", []) if v["status"] == "True"
        }
        if "Failed" in conditions:
            return "failed"
        if "Complete" in conditions:
            return "complete"
        if data["spec"].get("suspend"):
            return "queued:" + self.workload_reason(job)
        pods = json.loads(
            invoke([*self.kube, "get", "pods", "-l", f"job-name={job}", "-o", "json"])
        )
        for p in pods["items"]:
            status = p.get("status", {})
            if status.get("phase") in ("Running", "Succeeded", "Failed"):
                return "running"
            for c in status.get("containerStatuses", []):
                reason = c.get("state", {}).get("waiting", {}).get("reason")
                if reason in (
                    "CreateContainerConfigError",
                    "ImagePullBackOff",
                    "ErrImagePull",
                    "ContainerCreating",
                ):
                    return "queued:" + reason
            return "queued:" + pending_reason(status.get("conditions", []))
        return "queued:" + self.workload_reason(job)

    def outcome(self, job, role):
        if role == "pointer":
            return "complete"
        return receipt(role, invoke([*self.kube, "logs", f"job/{job}", "--tail=1"]))

    def stop(self, job):
        check(self.owned("job", job), "job missing before cancellation")
        invoke(
            [
                *self.kube,
                "delete",
                "job",
                job,
                "--cascade=foreground",
                "--wait=true",
                "--timeout=180s",
            ],
            timeout=200,
        )

    def tail(self, job):
        return safe_log(
            invoke([*self.kube, "logs", "job/" + job, "--limit-bytes=65536", "--tail=64"])
        )

    def health(self):
        pods = json.loads(
            invoke([*self.kube, "get", "pods", "-l", f"{OWNER}={self.name}", "-o", "json"])
        )
        rows = []
        for role in ("postgres", "minio"):
            name = f"{self.name}-{role}"
            sts = self.existing("StatefulSet", name)
            pvc = self.existing("PersistentVolumeClaim", name)
            status = (sts or {}).get("status", {})
            phase = (pvc or {}).get("status", {}).get("phase")
            rows.append(
                {
                    "service": role,
                    "state": "missing"
                    if sts is None
                    else "ready"
                    if status.get("readyReplicas") == 1
                    else "pending",
                    "pvc": phase if phase in ("Bound", "Pending", "Lost") else "missing",
                }
            )
        reasons = {
            "Completed",
            "Error",
            "OOMKilled",
            "Evicted",
            "ImagePullBackOff",
            "ErrImagePull",
            "CreateContainerConfigError",
            "ContainerCreating",
            "FailedMount",
            "Unschedulable",
            "PodScheduled",
            "InvalidImageName",
            "CrashLoopBackOff",
        }
        for p in pods["items"][-64:]:
            status = p.get("status", {})
            for c in status.get("containerStatuses", []):
                detail = c.get("state", {})
                value = detail.get("terminated", detail.get("waiting", {}))
                if c.get("name") in RESOURCES:
                    rows.append(
                        {
                            "role": c["name"],
                            "ready": c.get("ready") is True,
                            "reason": value.get("reason")
                            if value.get("reason") in reasons
                            else None,
                            "exit": value.get("exitCode")
                            if type(value.get("exitCode")) is int
                            else None,
                        }
                    )
        return rows

    def evidence(self, run, variant):
        # The host need not have filesystem access to the shared claim.
        job = self.name + "-review-" + uuid.uuid4().hex[:10]
        obj = self.manifest(job, "pointer", run, "main")
        spec = obj["spec"]["template"]["spec"]
        container = spec["containers"][0]
        container["env"] = [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}]
        container["volumeMounts"] = [
            m for m in container["volumeMounts"] if m["mountPath"] == TARGETS["outputs"]
        ]
        container["command"] = [
            "python",
            "-c",
            Path(__file__).with_name("evidence.py").read_text(),
            TARGETS["outputs"],
            run,
            variant,
        ]
        spec["volumes"][0]["persistentVolumeClaim"]["readOnly"] = True
        obj["spec"].update(activeDeadlineSeconds=120, ttlSecondsAfterFinished=600)
        print(
            f"PMMC_REVIEW_WAIT job={job} queue={self.cluster.get('queue')} timeout=120s", flush=True
        )
        invoke([*self.kube, "create", "-f", "-"], obj)
        try:
            invoke(
                [*self.kube, "wait", "--for=condition=complete", "--timeout=120s", "job/" + job],
                timeout=130,
            )
            data = json.loads(invoke([*self.kube, "logs", "job/" + job, "--limit-bytes=524288"]))
            check(
                isinstance(data, dict) and data.get("schema") == "pmmc-host-evidence-v1",
                "invalid shared PVC collection",
            )
            return data
        finally:
            # Also remove a reader that was never admitted. Do not touch workload Jobs or PVCs.
            self.stop(job)


Kueue = Kubernetes
