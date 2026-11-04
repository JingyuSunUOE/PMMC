"""Portable Kubernetes Job rendering for one-command PMMC experiments."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from pmmc.experiments.launcher import ExperimentLauncherConfig, InfrastructureMode
from pmmc.schemas.base import BaseSchema, JsonDict, NonEmptyStr, PositiveInt

_CONTAINER_PROJECT_ROOT = Path("/workspace")
_MAX_JOB_COMPONENT_LENGTH = 57


class _QuotedKubernetesLabel(str):
    """A label value that must remain a string across YAML 1.1 parsers."""


def _quote_kubernetes_label_values(value: object) -> object:
    if isinstance(value, list):
        return [_quote_kubernetes_label_values(item) for item in value]
    if not isinstance(value, dict):
        return value

    result: dict[object, object] = {}
    for key, item in value.items():
        if key == "labels" and isinstance(item, dict):
            labels: dict[str, _QuotedKubernetesLabel] = {}
            for label_key, label_value in item.items():
                if not isinstance(label_key, str) or not isinstance(label_value, str):
                    raise ValueError(
                        "Kubernetes metadata.labels keys and values must be strings"
                    )
                labels[label_key] = _QuotedKubernetesLabel(label_value)
            result[key] = labels
        else:
            result[key] = _quote_kubernetes_label_values(item)
    return result


class KubernetesNfsStorage(BaseSchema):
    server: NonEmptyStr
    path: NonEmptyStr = "/"
    mount_path: Path = Path("/data")
    read_only: bool = False

    @model_validator(mode="after")
    def _validate_mount(self) -> KubernetesNfsStorage:
        if not self.mount_path.is_absolute():
            raise ValueError("Kubernetes NFS mount_path must be absolute")
        export_path = PurePosixPath(self.path)
        if not export_path.is_absolute() or ".." in export_path.parts:
            raise ValueError("Kubernetes NFS path must be absolute and traversal-safe")
        return self


class KubernetesDeploymentConfig(BaseSchema):
    deployment_version: Literal["pmmc-kubernetes-experiment-v1"] = (
        "pmmc-kubernetes-experiment-v1"
    )
    image: NonEmptyStr
    image_pull_policy: NonEmptyStr = "Always"
    namespace: NonEmptyStr = "default"
    secret_name: NonEmptyStr = "pmmc-runtime-secrets"
    queue_name: NonEmptyStr | None = None
    job_name_prefix: NonEmptyStr | None = None
    service_account_name: NonEmptyStr | None = None
    cpu: NonEmptyStr = "24"
    memory: NonEmptyStr = "128Gi"
    gpus: PositiveInt = 1
    shared_memory: NonEmptyStr = "64Gi"
    active_deadline_seconds: PositiveInt = 604800
    termination_grace_period_seconds: PositiveInt = 180
    node_selector: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=lambda: {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"}
    )
    storage: KubernetesNfsStorage
    project_data_root: Path
    hf_cache_path: Path
    model_locks_path: Path
    runs_path: Path
    artifacts_path: Path
    outputs_path: Path
    entrypoint_script: Path | None = None
    postgres_cpu: NonEmptyStr = "2"
    postgres_memory: NonEmptyStr = "8Gi"
    minio_cpu: NonEmptyStr = "1"
    minio_memory: NonEmptyStr = "4Gi"
    scratch_size: NonEmptyStr = "200Gi"
    postgres_empty_dir_size: NonEmptyStr = "200Gi"
    minio_empty_dir_size: NonEmptyStr = "100Gi"
    env: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_paths(self) -> KubernetesDeploymentConfig:
        if self.job_name_prefix is not None and (
            len(self.job_name_prefix) > 48
            or re.fullmatch(
                r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
                self.job_name_prefix,
            )
            is None
        ):
            raise ValueError(
                "deployment job_name_prefix must be a lowercase Kubernetes name "
                "with at most 48 characters"
            )
        for name in (
            "project_data_root",
            "hf_cache_path",
            "model_locks_path",
            "runs_path",
            "artifacts_path",
            "outputs_path",
        ):
            if not getattr(self, name).is_absolute():
                raise ValueError(f"deployment {name} must be absolute")
        if ".." in self.project_data_root.parts:
            raise ValueError("deployment project_data_root cannot contain '..'")
        if not self.project_data_root.is_relative_to(self.storage.mount_path):
            raise ValueError(
                "deployment project_data_root must be inside the NFS mount_path"
            )
        if self.entrypoint_script is not None and (
            self.entrypoint_script.is_absolute()
            or ".." in self.entrypoint_script.parts
            or self.entrypoint_script.suffix != ".sh"
        ):
            raise ValueError(
                "deployment entrypoint_script must be a relative shell script"
            )
        sensitive = [
            name
            for name in self.env
            if any(
                marker in name.upper()
                for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
            )
        ]
        if sensitive:
            raise ValueError(
                "deployment env cannot contain credentials; use Kubernetes Secret: "
                + ", ".join(sorted(sensitive))
            )
        return self


def load_deployment_config(path: Path) -> KubernetesDeploymentConfig:
    import yaml  # type: ignore[import-untyped]

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Kubernetes deployment config root must be a mapping")
    return KubernetesDeploymentConfig.model_validate(raw)


def _config_map(name: str, files: dict[str, str]) -> dict[str, object]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name},
        "data": files,
    }


def _secret_env(name: str, secret_name: str, *, optional: bool = False) -> dict[str, object]:
    reference: dict[str, object] = {"name": secret_name, "key": name}
    if optional:
        reference["optional"] = True
    return {"name": name, "valueFrom": {"secretKeyRef": reference}}


def _job_component_name(
    *,
    experiment_name: str,
    configured_prefix: str | None,
    nonce: str,
) -> str:
    if configured_prefix is None:
        experiment_slug = re.sub(r"[^a-z0-9]+", "-", experiment_name.casefold()).strip(
            "-"
        )
        if experiment_slug.startswith("pmmc-"):
            experiment_slug = experiment_slug.removeprefix("pmmc-")
        prefix = f"aaai-pmmc-{experiment_slug or 'experiment'}"
    else:
        prefix = configured_prefix
    prefix_limit = _MAX_JOB_COMPONENT_LENGTH - len(nonce) - 1
    prefix = prefix[:prefix_limit].rstrip("-")
    if not prefix:
        raise ValueError("Kubernetes Job name prefix is empty after normalization")
    return f"{prefix}-{nonce}"


_DATABASE_SECRET_NAMES = (
    "PMMC_DATABASE_ADMIN_URL",
    "PMMC_DATABASE_URL",
    "PMMC_DATABASE_EVALUATOR_URL",
)
_OBJECT_STORE_SECRET_NAMES = (
    "PMMC_OBJECT_STORE_REGION",
    "PMMC_OBJECT_STORE_SECURE",
    "PMMC_OBJECT_STORE_PATH_STYLE",
    "PMMC_OBJECT_STORE_BUCKET",
    "PMMC_OBJECT_STORE_EVALUATION_BUCKET",
    "PMMC_OBJECT_STORE_ACCESS_KEY",
    "PMMC_OBJECT_STORE_SECRET_KEY",
    "PMMC_OBJECT_STORE_EVALUATOR_ACCESS_KEY",
    "PMMC_OBJECT_STORE_EVALUATOR_SECRET_KEY",
    "PMMC_OBJECT_STORE_ADMIN_ACCESS_KEY",
    "PMMC_OBJECT_STORE_ADMIN_SECRET_KEY",
)


def render_experiment_job(
    *,
    project_root: Path,
    experiment_config_path: Path,
    experiment: ExperimentLauncherConfig,
    deployment: KubernetesDeploymentConfig,
    gpus: int | None = None,
    cpu: str | None = None,
    memory: str | None = None,
    shared_memory: str | None = None,
    postgres_cpu: str | None = None,
    postgres_memory: str | None = None,
    minio_cpu: str | None = None,
    minio_memory: str | None = None,
    scratch_size: str | None = None,
    job_name_prefix: str | None = None,
    cli_overrides: list[str] | None = None,
    render_nonce: str | None = None,
) -> list[dict[str, object]]:
    """Return ConfigMaps and one generated-name Job without embedding credentials."""

    resource_overrides = {
        key: value
        for key, value in {
            "cpu": cpu,
            "memory": memory,
            "shared_memory": shared_memory,
            "postgres_cpu": postgres_cpu,
            "postgres_memory": postgres_memory,
            "minio_cpu": minio_cpu,
            "minio_memory": minio_memory,
            "scratch_size": scratch_size,
            "job_name_prefix": job_name_prefix,
        }.items()
        if value is not None
    }
    if resource_overrides:
        deployment = KubernetesDeploymentConfig.model_validate(
            {**deployment.model_dump(mode="python"), **resource_overrides}
        )
    resolved_config = experiment_config_path.resolve()
    try:
        config_in_image = Path("/workspace") / resolved_config.relative_to(
            project_root.resolve()
        )
    except ValueError as exc:
        raise ValueError("experiment config must be inside the project image context") from exc
    gpu_count = gpus or deployment.gpus
    if gpu_count <= 0:
        raise ValueError("Kubernetes experiment requires at least one GPU")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "experiment": experiment.config_hash,
                "deployment": deployment.model_dump(mode="json"),
                "gpus": gpu_count,
                "overrides": cli_overrides or [],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    nonce = render_nonce or uuid4().hex[:8]
    if not re.fullmatch(r"[a-z0-9]{1,16}", nonce):
        raise ValueError("render nonce must contain 1-16 lowercase letters or digits")
    component = _job_component_name(
        experiment_name=experiment.experiment_name,
        configured_prefix=deployment.job_name_prefix,
        nonce=nonce,
    )
    script_name = f"{component}-sidecars"
    postgres_name = f"{component}-postgres-init"

    command_args = [
        "--config",
        str(config_in_image),
        *(cli_overrides or []),
    ]
    main_script = (
        "set -eu\n"
        "heartbeat() { while :; do touch /pmmc-control/main-heartbeat; sleep 30; done; }\n"
        "heartbeat & heartbeat_pid=$!\n"
        "cleanup() { kill \"$heartbeat_pid\" 2>/dev/null || true; "
        "touch /pmmc-control/stop; }\n"
        "trap cleanup EXIT INT TERM\n"
        "if [ -n \"${PMMC_EXPERIMENT_ENTRYPOINT:-}\" ]; then\n"
        "  bash \"/workspace/$PMMC_EXPERIMENT_ENTRYPOINT\" \"$@\"\n"
        "else\n"
        "  python3 -m pmmc experiment run \"$@\"\n"
        "fi\n"
    )
    labels = {
        "app.kubernetes.io/name": "pmmc",
        "app.kubernetes.io/component": "research-experiment",
        "pmmc-launch-config": experiment.config_hash[:16],
        "pmmc-render-fingerprint": fingerprint,
    }
    if deployment.queue_name:
        labels["kueue.x-k8s.io/queue-name"] = deployment.queue_name

    common_env: list[dict[str, object]] = [
        {"name": "PMMC_MODE", "value": experiment.mode.value},
        {"name": "PMMC_RUNTIME_MODE", "value": "managed_local"},
        {"name": "PMMC_CONTAINER_IMAGE", "value": deployment.image},
        {"name": "PMMC_EXPERIMENT_CONFIG_PATH", "value": str(config_in_image)},
        {"name": "PMMC_PROJECT_ROOT", "value": str(_CONTAINER_PROJECT_ROOT)},
        {
            "name": "PMMC_BENCHMARK_ASSET_ROOT",
            "value": str(deployment.project_data_root),
        },
        {
            "name": "PMMC_MEMLENS_ROOT",
            "value": str(deployment.project_data_root / "benchmarks/memlens"),
        },
        {
            "name": "PMMC_MEMGALLERY_ROOT",
            "value": str(deployment.project_data_root / "benchmarks/memgallery"),
        },
        {"name": "PMMC_ARTIFACTS_DIR", "value": str(deployment.artifacts_path)},
        {"name": "PMMC_OUTPUTS_DIR", "value": str(deployment.outputs_path)},
        {"name": "PMMC_RUNS_DIR", "value": str(deployment.runs_path)},
        {
            "name": "PMMC_BASELINE_CORPORA_DIR",
            "value": str(deployment.artifacts_path / "baselines/corpora"),
        },
        {
            "name": "PMMC_MODEL_LOCK_DIR",
            "value": str(deployment.model_locks_path),
        },
        {
            "name": "PMMC_MODEL_LOCK_PATH",
            "value": str(deployment.model_locks_path / "paper.lock.json"),
        },
        {
            "name": "PMMC_RUNTIME_CALIBRATION_DIR",
            "value": str(deployment.artifacts_path / "runtime/calibrations"),
        },
        {"name": "HF_HOME", "value": str(deployment.hf_cache_path)},
        {"name": "PMMC_MODEL_CACHE_DIR", "value": str(deployment.hf_cache_path)},
        {"name": "PMMC_LAUNCH_STOP_FILE", "value": "/pmmc-control/stop"},
        {"name": "PMMC_RUNTIME_PROCESS_LOCK_PATH", "value": "/pmmc-control/runtime.lock"},
        {"name": "PMMC_EXPECTED_GPU_COUNT", "value": str(gpu_count)},
        {"name": "PMMC_LOCAL_SCRATCH_DIR", "value": "/pmmc-scratch"},
        {
            "name": "PMMC_TOKEN_COUNTER_CACHE_DIR",
            "value": "/pmmc-scratch/token-counter",
        },
        _secret_env("HF_TOKEN", deployment.secret_name, optional=True),
    ]
    if deployment.entrypoint_script is not None:
        common_env.append(
            {
                "name": "PMMC_EXPERIMENT_ENTRYPOINT",
                "value": deployment.entrypoint_script.as_posix(),
            }
        )
    secret_names = (
        *_DATABASE_SECRET_NAMES,
        *_OBJECT_STORE_SECRET_NAMES,
        "PMMC_OBJECT_STORE_ENDPOINT",
    )
    common_env.extend(
        _secret_env(name, deployment.secret_name)
        for name in secret_names
        if name not in deployment.env
    )
    if "PMMC_OBJECT_STORE_MODEL_ENDPOINT" not in deployment.env:
        if experiment.infrastructure.mode is InfrastructureMode.SELF_CONTAINED:
            common_env.append(
                {
                    "name": "PMMC_OBJECT_STORE_MODEL_ENDPOINT",
                    "value": "http://127.0.0.1:9000",
                }
            )
        else:
            common_env.append(
                _secret_env(
                    "PMMC_OBJECT_STORE_MODEL_ENDPOINT",
                    deployment.secret_name,
                    optional=True,
                )
            )
    common_env.extend(
        {"name": name, "value": value} for name, value in sorted(deployment.env.items())
    )

    volumes: list[dict[str, object]] = [
        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": deployment.shared_memory}},
        {"name": "control", "emptyDir": {"sizeLimit": "64Mi"}},
        {"name": "scratch", "emptyDir": {"sizeLimit": deployment.scratch_size}},
        {
            "name": "shared-storage",
            "nfs": {
                "server": deployment.storage.server,
                "path": deployment.storage.path,
                "readOnly": deployment.storage.read_only,
            },
        },
    ]
    containers: list[dict[str, object]] = [
        {
            "name": "pmmc",
            "image": deployment.image,
            "imagePullPolicy": deployment.image_pull_policy,
            "command": ["/usr/bin/tini", "--", "/bin/sh", "-ec", main_script, "pmmc"],
            "args": command_args,
            "securityContext": {
                "runAsUser": 10001,
                "runAsGroup": 10001,
                "runAsNonRoot": True,
                "allowPrivilegeEscalation": False,
            },
            "env": common_env,
            "resources": {
                "requests": {
                    "cpu": deployment.cpu,
                    "memory": deployment.memory,
                    "nvidia.com/gpu": str(gpu_count),
                },
                "limits": {
                    "cpu": deployment.cpu,
                    "memory": deployment.memory,
                    "nvidia.com/gpu": str(gpu_count),
                },
            },
            "volumeMounts": [
                {"name": "dshm", "mountPath": "/dev/shm"},
                {"name": "control", "mountPath": "/pmmc-control"},
                {"name": "scratch", "mountPath": "/pmmc-scratch"},
                {
                    "name": "shared-storage",
                    "mountPath": str(deployment.storage.mount_path),
                    "readOnly": deployment.storage.read_only,
                },
            ],
        }
    ]
    resources: list[dict[str, object]] = []
    if experiment.infrastructure.mode is InfrastructureMode.SELF_CONTAINED:
        script_root = project_root / "deploy/k8s/step5d-gpu-smoke"
        script_files = {
            name: (script_root / name).read_text(encoding="utf-8")
            for name in (
                "postgres-sidecar.sh",
                "minio-sidecar.sh",
                "minio-initialize.sh",
                "minio-init.sh",
                "watchdog.sh",
            )
        }
        resources.extend(
            [
                _config_map(script_name, script_files),
                _config_map(
                    postgres_name,
                    {
                        "init-roles.sh": (
                            script_root / "postgres-init-roles.sh"
                        ).read_text(encoding="utf-8")
                    },
                ),
            ]
        )
        volumes.extend(
            [
                {
                    "name": "postgres-data",
                    "emptyDir": {"sizeLimit": deployment.postgres_empty_dir_size},
                },
                {
                    "name": "minio-data",
                    "emptyDir": {"sizeLimit": deployment.minio_empty_dir_size},
                },
                {
                    "name": "sidecar-scripts",
                    "configMap": {"name": script_name, "defaultMode": 365},
                },
                {
                    "name": "postgres-init",
                    "configMap": {"name": postgres_name, "defaultMode": 365},
                },
            ]
        )
        containers.extend(_self_contained_containers(deployment, script_name, postgres_name))

    pod_spec: dict[str, object] = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": deployment.termination_grace_period_seconds,
        "initContainers": [
            _ephemeral_volume_initializer(
                self_contained=(
                    experiment.infrastructure.mode is InfrastructureMode.SELF_CONTAINED
                )
            )
        ],
        "containers": containers,
        "volumes": volumes,
        "nodeSelector": deployment.node_selector,
    }
    if deployment.service_account_name:
        pod_spec["serviceAccountName"] = deployment.service_account_name
    job: dict[str, object] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": f"{component}-",
            "namespace": deployment.namespace,
            "labels": labels,
        },
        "spec": {
            "parallelism": 1,
            "completions": 1,
            "backoffLimit": 0,
            "activeDeadlineSeconds": deployment.active_deadline_seconds,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }
    resources.append(job)
    for resource in resources:
        metadata = resource.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("namespace", deployment.namespace)
    return resources


def _ephemeral_volume_initializer(*, self_contained: bool) -> dict[str, object]:
    paths = ["/pmmc-control", "/pmmc-scratch"]
    mounts = [
        {"name": "control", "mountPath": "/pmmc-control"},
        {"name": "scratch", "mountPath": "/pmmc-scratch"},
    ]
    if self_contained:
        paths.extend(["/postgres-data", "/minio-data"])
        mounts.extend(
            [
                {"name": "postgres-data", "mountPath": "/postgres-data"},
                {"name": "minio-data", "mountPath": "/minio-data"},
            ]
        )
    return {
        "name": "prepare-ephemeral-volumes",
        "image": "busybox:1.36.1",
        "command": ["/bin/sh", "-ec"],
        "args": ["chmod 0777 " + " ".join(paths)],
        "securityContext": {"runAsUser": 0, "runAsGroup": 0},
        "resources": {
            "requests": {"cpu": "50m", "memory": "32Mi"},
            "limits": {"cpu": "100m", "memory": "64Mi"},
        },
        "volumeMounts": mounts,
    }


def _self_contained_containers(
    deployment: KubernetesDeploymentConfig,
    script_name: str,
    postgres_name: str,
) -> list[dict[str, object]]:
    del script_name, postgres_name
    secret = deployment.secret_name
    stop_env = [
        {"name": "PMMC_STEP5D_GPU_SMOKE_CONTROL_DIR", "value": "/pmmc-control"},
        {"name": "PMMC_STEP5D_GPU_SMOKE_STOP_FILE", "value": "/pmmc-control/stop"},
    ]
    mounts = [
        {"name": "control", "mountPath": "/pmmc-control"},
        {"name": "sidecar-scripts", "mountPath": "/config", "readOnly": True},
    ]
    return [
        {
            "name": "postgres",
            "image": "pgvector/pgvector:pg16",
            "command": ["/bin/sh", "/config/postgres-sidecar.sh"],
            "env": [
                {
                    "name": "POSTGRES_DB",
                    "valueFrom": {
                        "secretKeyRef": {"name": secret, "key": "PMMC_POSTGRES_DB"}
                    },
                },
                {"name": "POSTGRES_USER", "value": "pmmc_owner"},
                {
                    "name": "POSTGRES_PASSWORD",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": secret,
                            "key": "PMMC_POSTGRES_ADMIN_PASSWORD",
                        }
                    },
                },
                _secret_env("PMMC_POSTGRES_RUNTIME_PASSWORD", secret),
                _secret_env("PMMC_POSTGRES_EVALUATOR_PASSWORD", secret),
                *stop_env,
            ],
            "resources": {
                "requests": {
                    "cpu": deployment.postgres_cpu,
                    "memory": deployment.postgres_memory,
                },
                "limits": {
                    "cpu": deployment.postgres_cpu,
                    "memory": deployment.postgres_memory,
                },
            },
            "volumeMounts": [
                *mounts,
                {"name": "postgres-data", "mountPath": "/var/lib/postgresql/data"},
                {
                    "name": "postgres-init",
                    "mountPath": "/docker-entrypoint-initdb.d/10-pmmc-roles.sh",
                    "subPath": "init-roles.sh",
                    "readOnly": True,
                },
            ],
        },
        {
            "name": "minio",
            "image": (
                "quay.io/minio/minio:RELEASE.2025-04-22T22-12-26Z"
                "@sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e"
            ),
            "command": ["/bin/sh", "/config/minio-sidecar.sh"],
            "env": [
                {
                    "name": "MINIO_ROOT_USER",
                    "valueFrom": {
                        "secretKeyRef": {"name": secret, "key": "PMMC_MINIO_ROOT_USER"}
                    },
                },
                {
                    "name": "MINIO_ROOT_PASSWORD",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": secret,
                            "key": "PMMC_MINIO_ROOT_PASSWORD",
                        }
                    },
                },
                *stop_env,
            ],
            "resources": {
                "requests": {
                    "cpu": deployment.minio_cpu,
                    "memory": deployment.minio_memory,
                },
                "limits": {
                    "cpu": deployment.minio_cpu,
                    "memory": deployment.minio_memory,
                },
            },
            "volumeMounts": [
                *mounts,
                {"name": "minio-data", "mountPath": "/data"},
            ],
        },
        {
            "name": "minio-init",
            "image": (
                "quay.io/minio/mc:RELEASE.2025-04-16T18-13-26Z"
                "@sha256:aead63c77f9db9107f1696fb08ecb0faeda23729cde94b0f663edf4fe09728e3"
            ),
            "command": ["/bin/sh", "/config/minio-initialize.sh"],
            "env": [
                {"name": "PMMC_MINIO_ENDPOINT", "value": "http://127.0.0.1:9000"},
                _secret_env("PMMC_MINIO_ROOT_USER", secret),
                _secret_env("PMMC_MINIO_ROOT_PASSWORD", secret),
                *(
                    _secret_env(name, secret)
                    for name in (
                        "PMMC_OBJECT_STORE_BUCKET",
                        "PMMC_OBJECT_STORE_EVALUATION_BUCKET",
                        "PMMC_OBJECT_STORE_ACCESS_KEY",
                        "PMMC_OBJECT_STORE_SECRET_KEY",
                        "PMMC_OBJECT_STORE_EVALUATOR_ACCESS_KEY",
                        "PMMC_OBJECT_STORE_EVALUATOR_SECRET_KEY",
                    )
                ),
                *stop_env,
            ],
            "resources": {
                "requests": {"cpu": "200m", "memory": "256Mi"},
                "limits": {"cpu": "500m", "memory": "512Mi"},
            },
            "volumeMounts": mounts,
        },
        {
            "name": "watchdog",
            "image": "busybox:1.36.1",
            "command": ["/bin/sh", "/config/watchdog.sh"],
            "env": [
                *stop_env,
                {
                    "name": "PMMC_STEP5D_GPU_SMOKE_HEARTBEAT_TIMEOUT_SECONDS",
                    "value": "300",
                },
            ],
            "resources": {
                "requests": {"cpu": "50m", "memory": "32Mi"},
                "limits": {"cpu": "100m", "memory": "64Mi"},
            },
            "volumeMounts": mounts,
        },
    ]


def write_job_yaml(resources: list[dict[str, object]], output: Path) -> None:
    import yaml  # type: ignore[import-untyped]

    class _KubernetesSafeDumper(yaml.SafeDumper):
        pass

    _KubernetesSafeDumper.add_representer(
        _QuotedKubernetesLabel,
        lambda dumper, value: dumper.represent_scalar(
            "tag:yaml.org,2002:str",
            str(value),
            style='"',
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "---\n".join(
        yaml.dump(
            _quote_kubernetes_label_values(item),
            Dumper=_KubernetesSafeDumper,
            allow_unicode=False,
            sort_keys=False,
        )
        for item in resources
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)


def render_command(
    resources: list[dict[str, object]],
    output: Path = Path("job.generated.yaml"),
) -> str:
    """Return commands that submit all resources and capture only the Job name."""

    job = next(item for item in resources if item.get("kind") == "Job")
    metadata = job.get("metadata")
    namespace = metadata.get("namespace") if isinstance(metadata, dict) else None
    suffix = f" -n {shlex.quote(str(namespace))}" if namespace else ""
    source = shlex.quote(str(output))
    return "\n".join(
        (
            'JOB="$(kubectl create '
            f"-f {source}{suffix} -o name "
            "| awk -F/ '$1 == \"job.batch\" || $1 == \"job\" {print $2}')\"",
            'test -n "$JOB"',
            'echo "JOB=$JOB"',
        )
    )
