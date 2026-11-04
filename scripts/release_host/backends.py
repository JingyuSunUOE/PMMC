"""Docker and Kubernetes adapters. Never forward CLI output into host diagnostics."""

import base64
import json
import os
import platform
import re
import subprocess
import time
import uuid
from pathlib import Path

from .config import STAGES, TARGETS, Error, check
from .evidence import collect, safe_log
from .quota import headroom, peak, pending_reason
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
    if role in {"iam", "bootstrap", "models"}:
        protocols = {
            "iam": ("step8-minio-iam-v1",),
            "bootstrap": ("step8-bootstrap-v1",),
            "models": ("step8-model-preparation-v1", "release-locked-cache-hit-v1"),
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
        return {k: data.get(k) for k in ("run_id", "variant", "selection_hash")}
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
            if not isinstance(data.get("limit_usd"), str) or not re.fullmatch(
                amount, data["limit_usd"]
            ):
                raise ReceiptError("invalid approval amount")
            summary["limit_usd"] = data["limit_usd"]
        else:
            if data.get("approval_created") is not False or data.get("scope") not in STAGES:
                raise ReceiptError("invalid quote receipt")
            summary.update(scope=data["scope"], approval_created=False)
            for section in ("pending_scope", "pending_additional_after_preceding_scopes"):
                estimates = data.get(section)
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
            or (state == "stage_complete" and done == n and data["paid_authorized"])
            or (state == "waiting_for_workers" and role == "gpu" and done < n)
        )
    ):
        raise ReceiptError(f"invalid {role} stage receipt")
    return {
        k: data[k]
        for k in ("state", "stage", "scope_hash", "scoped_jobs", "completed", "paid_authorized")
    }


class Compose:
    def __init__(self, cfg, name, private, creds, approval, request=None):
        self.cfg, self.name, self.private = cfg, name, private
        self.creds, self.approval = creds, approval
        self.request = request
        self.docker = ["docker", "--context", "default"]

    def manifest(self, role=None, run=None, stage=None):
        cfg = self.cfg
        services = {}
        for service, volume, dest in [
            ("postgres", "pgdata", "/var/lib/postgresql/data"),
            ("minio", "s3data", "/data"),
        ]:
            services[service] = {
                "image": cfg["images"][service],
                "labels": {OWNER: self.name},
                "restart": "unless-stopped",
                "networks": {"default": {"aliases": [f"{self.name}-{service}"]}},
                "volumes": [f"{volume}:{dest}"],
                "environment": {
                    k: self.creds[v] for k, v in secret_refs(service, cfg, None).items()
                },
            }
        services["postgres"]["environment"].update(
            POSTGRES_USER="pmmc_owner", POSTGRES_DB="pmmc", PGDATA="/var/lib/postgresql/data/pgdata"
        )
        services["minio"]["command"] = ["server", "/data"]
        for service, test in [
            ("postgres", ["pg_isready", "-U", "pmmc_owner", "-d", "pmmc"]),
            ("minio", ["curl", "-fsS", "http://localhost:9000/minio/health/ready"]),
        ]:
            services[service]["healthcheck"] = {
                "test": ["CMD", *test],
                "interval": "5s",
                "timeout": "5s",
                "retries": 100,
            }
        if role:
            env = environment(role, cfg, self.name, self.approval)
            for key, source in secret_refs(role, cfg, self.approval).items():
                value = self.creds.get(source) or os.environ.get(source)
                check(value, f"missing credential {source}")
                env[key] = value.replace("$", "$$")
            job = services[role] = {
                "image": cfg["images"]["iam" if role == "iam" else "app"],
                "labels": {OWNER: self.name},
                "working_dir": "/workspace",
                "environment": env,
                "entrypoint": command(role, cfg, run, stage, self.request),
                "command": [],
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
            }
            if role == "iam":
                job.update(
                    user=f"{os.getuid()}:{os.getgid()}",
                    read_only=True,
                    tmpfs=["/tmp"],
                    volumes=[f"{self.private / 'iam'}:{IAM_DIR}:ro"],
                )
            else:
                job["user"] = f"{cfg['identity']['uid']}:{cfg['identity']['gid']}"
                job["volumes"] = [
                    {
                        "type": "bind",
                        "source": cfg["mounts"][k],
                        "target": dest,
                        "read_only": k == "inputs" or role == "pointer",
                        "bind": {"create_host_path": False},
                    }
                    for k, dest in TARGETS.items()
                ]
            if role in GPU_ROLES:
                env["PMMC_LOCAL_SCRATCH_DIR"] = "/tmp/pmmc-scratch"
                env["PMMC_TOKEN_COUNTER_CACHE_DIR"] = "/tmp/pmmc-scratch/token-counter"
                job.update(
                    shm_size="64gb",
                    deploy={
                        "resources": {
                            "reservations": {
                                "devices": [
                                    {"driver": "nvidia", "count": 8, "capabilities": ["gpu"]}
                                ]
                            }
                        }
                    },
                )
        for role, spec in services.items():
            r = cfg["resources"][role]
            spec.update(cpus=r["cpu"], mem_limit=f"{r['memory_gib']}g")
        return {"services": services, "volumes": {"pgdata": {}, "s3data": {}}}

    def compose(self, manifest, *args):
        return invoke(
            [
                *self.docker,
                "compose",
                "--project-name",
                self.name,
                "--env-file",
                "/dev/null",
                "-f",
                "-",
                *args,
            ],
            manifest,
            timeout=900,
        )

    def reconcile(self):
        check(platform.system() == "Linux", "local-nvidia requires a Linux host")
        check(
            all(Path(p).is_dir() for p in self.cfg["mounts"].values()),
            "pre-create the four dedicated bind mount directories",
        )
        folder = self.private / "iam"
        folder.mkdir(exist_ok=True, mode=0o700)
        for k in IAM_KEYS:
            path = folder / k
            if not path.exists():
                path.write_text(self.creds[k])
            check(path.read_text() == self.creds[k], "local IAM file mismatch")
        self.compose(
            self.manifest(), "up", "-d", "--wait", "--wait-timeout", "600", "postgres", "minio"
        )

    def start(self, job, role, run, stage):
        self.compose(
            self.manifest(role, run, stage),
            "run",
            "--no-deps",
            "--detach",
            "-T",
            "--name",
            job,
            role,
        )

    def state(self, job):
        if not invoke(
            [*self.docker, "ps", "-a", "--filter", f"name=^/{job}$", "--format", "{{.Names}}"]
        ):
            return "missing"
        info = invoke(
            [
                *self.docker,
                "inspect",
                "--type",
                "container",
                "--format",
                '{{ index .Config.Labels "pmmc-release-instance" }} '
                "{{.State.Status}} {{.State.ExitCode}}",
                job,
            ]
        ).split()
        check(len(info) == 3 and info[0] == self.name, "foreign or malformed Compose job")
        if info[1] == "exited":
            return "complete" if info[2] == "0" else "failed"
        return "failed" if info[1] in ("dead", "removing") else "running"

    def outcome(self, job, role):
        if role == "pointer":
            return "complete"
        # Native final JSON is stdout. Docker sends stderr to a separate stream.
        return receipt(role, invoke([*self.docker, "logs", "--tail", "1", job]))

    def stop(self, job):
        invoke([*self.docker, "stop", "--time", "30", job])

    def evidence(self, run, variant):
        return collect(Path(self.cfg["mounts"]["outputs"]), run, variant)

    def tail(self, job):
        return safe_log(invoke([*self.docker, "logs", "--tail", "64", job], stderr=True))

    def health(self):
        rows = []
        for service in ("postgres", "minio"):
            ids = invoke(
                [
                    *self.docker,
                    "ps",
                    "-aq",
                    "--filter",
                    f"label=com.docker.compose.project={self.name}",
                    "--filter",
                    f"label=com.docker.compose.service={service}",
                ]
            ).split()[:2]
            for cid in ids:
                health = invoke(
                    [
                        *self.docker,
                        "inspect",
                        "--format",
                        "{{.State.Status}} {{.State.ExitCode}} {{.State.Health.Status}}",
                        cid,
                    ]
                )
                if re.fullmatch(
                    r"(running|exited|created|dead|restarting) \d+ (healthy|unhealthy|starting)",
                    health,
                ):
                    rows.append({"service": service, "state": health})
        return rows


class Kueue:
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
        if queued:
            labels["kueue.x-k8s.io/queue-name"] = self.cluster["queue"]
        obj = {
            "apiVersion": version,
            "kind": kind,
            "metadata": {"name": name, "namespace": self.cluster["namespace"], "labels": labels},
        }
        if spec is not None:
            obj["spec"] = spec
        return obj

    def owned(self, kind, name):
        label = invoke(
            [
                *self.kube,
                "get",
                kind,
                name,
                "--ignore-not-found",
                "-o",
                'go-template={{index .metadata.labels "pmmc-release-instance"}}',
            ]
        )
        check(label in ("", self.name), "refusing to adopt a foreign cluster resource")
        return bool(label)

    def ensure(self, obj):
        kind, name = obj["kind"], obj["metadata"]["name"]
        if not self.owned(kind, name):
            invoke([*self.kube, "create", "-f", "-"], obj)
        elif kind == "Secret":
            stored = json.loads(invoke([*self.kube, "get", "secret", name, "-o", "json"]))
            check(
                stored.get("immutable") is True and stored.get("data") == obj["data"],
                "existing managed Secret differs from host identity; refusing rotation",
            )
        else:
            invoke(
                [*self.kube, "apply", "--server-side", "--field-manager=pmmc-release", "-f", "-"],
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
            for k, src in secret_refs(role, self.cfg, self.approval).items()
        ]

    def resources(self, role):
        r = self.cfg["resources"][role]
        spec = {"cpu": str(r["cpu"]), "memory": f"{r['memory_gib']}Gi"}
        if role in GPU_ROLES:
            spec["nvidia.com/gpu"] = "8"
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
                "resources": {"requests": {"storage": "300Gi" if pg else "120Gi"}},
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
        container = {
            "name": role,
            "image": self.cfg["images"][role],
            "env": self.refs(role),
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

    def reconcile(self):
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
            "ReadWriteMany" in claim["spec"]["accessModes"] and claim["status"]["phase"] == "Bound",
            "shared PVC must be Bound with ReadWriteMany access",
        )
        self.check_quota()
        secret = self.obj("Secret", self.secret(IAM_KEYS[0]))
        secret.update(
            immutable=True,
            type="Opaque",
            data={k: base64.b64encode(v.encode()).decode() for k, v in self.creds.items()},
        )
        self.ensure(secret)
        for role in ("postgres", "minio"):
            for obj in self.service_objects(role):
                self.ensure(obj)
        self.wait_services()

    def check_quota(self):
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
        quotas = json.loads(invoke([*self.kube, "get", "resourcequota", "-o", "json"]))["items"]
        pods = json.loads(
            invoke([*self.kube, "get", "pods", "-l", f"{OWNER}={self.name}", "-o", "json"])
        )["items"]
        workers = not self.request or self.request["kind"] == "run"
        try:
            limits = headroom(self.cfg, quotas, pods, workers=workers)
        except ValueError as exc:
            raise Error("namespace ResourceQuota not comparable; inspect before launch") from exc
        self.quota = {
            "queue": self.cluster["queue"],
            "peak": peak(self.cfg, workers),
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
        volumes, mounts = [], []
        pod = {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "volumes": volumes,
        }
        env = environment(role, self.cfg, self.name, self.approval)
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
                    "subPath": self.cfg["mounts"][k],
                    "mountPath": dest,
                    "readOnly": k == "inputs" or role == "pointer",
                }
                for k, dest in TARGETS.items()
            )
        if role in GPU_ROLES:
            pod["nodeSelector"] = {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"}
            for name, path, spec in [
                ("scratch", "/pmmc-scratch", {"sizeLimit": "300Gi"}),
                ("shm", "/dev/shm", {"medium": "Memory", "sizeLimit": "64Gi"}),
            ]:
                volumes.append({"name": name, "emptyDir": spec})
                mounts.append({"name": name, "mountPath": path})
        obj = self.obj(
            "Job",
            job,
            {
                "suspend": True,
                "backoffLimit": 0,
                "activeDeadlineSeconds": 604800,
                "template": {"metadata": {}, "spec": pod},
            },
            queued=True,
        )
        obj["spec"]["template"]["metadata"]["labels"] = obj["metadata"]["labels"]
        return obj

    def start(self, job, role, run, stage):
        if role == "control":
            self.check_quota()
        for external in {
            self.secret(k)
            for k in secret_refs(role, self.cfg, self.approval).values()
            if k in (*PROVIDER, "HF_TOKEN")
        }:
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
            required = [
                k
                for k in secret_refs(role, self.cfg, self.approval).values()
                if self.secret(k) == external
            ]
            check(
                set(required) <= set(keys),
                "external provider/model Secret is missing required keys",
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
        # The host need not have filesystem access to the EIDF shared claim.
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
        print(f"PMMC_REVIEW_WAIT job={job} queue={self.cluster['queue']} timeout=120s", flush=True)
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
