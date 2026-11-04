"""Normalize deployment inputs before any backend or credential operations."""

import re
import tomllib
from pathlib import Path, PurePosixPath

from .quota import DEFAULTS, peak

PROFILES = ("eidf", "local-nvidia")
STAGES = ("main", "analysis", "baselines")
VARIANTS = ("wr", "w", "r", "legacy")
SUITE = "exploration-dev-all"
TARGETS = {
    "inputs": "/pmmc-inputs",
    "outputs": "/workspace/outputs",
    "artifacts": "/workspace/artifacts",
    "cache": "/pmmc-cache",
}
DNS = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
IMAGE = r"[a-z0-9][a-z0-9.:/_-]*@sha256:[0-9a-f]{64}"
ROOT = Path(__file__).resolve().parents[2]


class Error(Exception):
    """Messages are printable; do not include subprocess output or credentials."""


def check(ok, message):
    if not ok:
        raise Error(message)


def identifier(value):
    check(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value),
        "invalid run ID",
    )
    return value


def approval_id(value):
    check(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        "expected a native SHA-256 budget approval ID",
    )
    return value


def table(value, keys, required=()):
    check(
        isinstance(value, dict)
        and not (value.keys() - set(keys))
        and set(required) <= value.keys(),
        "unknown or missing deployment setting",
    )
    return value


def clean_path(value, *, absolute):
    check(
        isinstance(value, str)
        and value
        and re.fullmatch(r"[/a-zA-Z0-9._-]+", value)
        and PurePosixPath(value).is_absolute() == absolute
        and not {"", ".", ".."}.intersection(value.lstrip("/").split("/"))
        and str(PurePosixPath(value)) == value,
        "invalid mount or configuration path",
    )
    return value


def native_path(value):
    clean_path(value, absolute=True)
    check(
        value.startswith(("/workspace/configs/", "/pmmc-inputs/", "/workspace/artifacts/")),
        "native configuration must reside under configs, inputs or artifacts",
    )
    return value


def distinct_paths(values):
    paths = [PurePosixPath(v) for v in values]
    check(
        all(
            a != b and a not in b.parents and b not in a.parents
            for i, a in enumerate(paths)
            for b in paths[i + 1 :]
        ),
        "mounts must be disjoint",
    )


def load_config(file, profile):
    try:
        with Path(file).open("rb") as stream:
            cfg = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise Error("cannot read deployment TOML") from exc
    keys = (
        "profile",
        "instance",
        "images",
        "mounts",
        "native",
        "models",
        "cluster",
        "identity",
        "resources",
    )
    table(cfg, keys, keys[:4])
    check(cfg["profile"] == profile and profile in PROFILES, "configuration/profile mismatch")
    check(
        isinstance(cfg["instance"], str)
        and re.fullmatch(DNS, cfg["instance"])
        and len(cfg["instance"]) <= 20,
        "invalid instance name (max 20 characters)",
    )
    names = {"app", "iam", "postgres", "minio"}
    images = table(cfg["images"], names, names)
    for ref in images.values():
        check(
            isinstance(ref, str) and re.fullmatch(IMAGE, ref) and ref[-64:] != "0" * 64,
            "each image needs an operator-verified repository@sha256 digest",
        )
    mounts = table(cfg["mounts"], TARGETS, TARGETS)
    for key, v in mounts.items():
        clean_path(v, absolute=profile == "local-nvidia")
        if profile == "local-nvidia":
            mounts[key] = str(Path(v).resolve())
            check(
                not Path(mounts[key]).is_relative_to(ROOT) and not ROOT.is_relative_to(mounts[key]),
                "use a dedicated external mount",
            )
    distinct_paths(mounts.values())
    native = cfg.setdefault("native", {})
    table(
        native,
        (
            "benchmark_config",
            "release_profile",
            "model_registry",
            "model_lock",
            "runtime_profile",
            "rate_card",
        ),
    )
    for key, default in {
        "benchmark_config": "/workspace/configs/release/prepare-models.yaml",
        "model_registry": "/workspace/configs/release/models.yaml",
        "model_lock": "/workspace/configs/release/models.lock.json",
        "runtime_profile": "/workspace/configs/release/runtime.yaml",
    }.items():
        native.setdefault(key, default)
    for path in native.values():
        native_path(path)
    models = cfg.setdefault("models", {})
    table(models, ("allow_download", "create_lock", "use_hf_token", "calibrate"))
    for key in ("allow_download", "create_lock", "use_hf_token", "calibrate"):
        check(
            type(models.setdefault(key, key in ("allow_download", "calibrate"))) is bool,
            "models options must be boolean",
        )
    check(not models["use_hf_token"] or models["allow_download"], "HF token requires download")
    check(
        not models["create_lock"],
        "public release requires an existing pinned lock; creation is disabled",
    )
    identity = cfg.setdefault("identity", {})
    table(identity, ("uid", "gid"))
    for k in ("uid", "gid"):
        check(
            type(identity.setdefault(k, 10001)) is int and 0 < identity[k] < 2**31,
            "app UID/GID must be positive integers",
        )
    resource = table(cfg.setdefault("resources", {}), DEFAULTS)
    for role, (cpu, gib) in DEFAULTS.items():
        spec = table(resource.setdefault(role, {}), ("cpu", "memory_gib"))
        for key, default in (("cpu", cpu), ("memory_gib", gib)):
            check(
                type(spec.setdefault(key, default)) is int and 0 < spec[key] <= 4096,
                "resource settings must be positive bounded integers",
            )
    check(resource["gpu"]["cpu"] >= 8, "eight GPU replicas require at least eight CPUs")
    if profile == "eidf":
        keys = ("context", "namespace", "queue", "shared_pvc", "storage_class")
        cluster = table(
            cfg.get("cluster"), (*keys, "cpu_budget", "admission_timeout_seconds"), keys
        )
        for k in keys:
            v = cluster[k]
            pattern = r"[A-Za-z0-9][A-Za-z0-9_.:@/-]*" if k == "context" else DNS
            check(
                isinstance(v, str) and re.fullmatch(pattern, v) and len(v) <= 253,
                "invalid cluster identifier",
            )
        for k, default, high in (
            ("cpu_budget", 100, 4096),
            ("admission_timeout_seconds", 600, 86400),
        ):
            check(
                type(cluster.setdefault(k, default)) is int and 1 <= cluster[k] <= high,
                "invalid cluster CPU budget or admission timeout",
            )
        check(
            peak(cfg)["cpu"] <= cluster["cpu_budget"],
            "insufficient configured CPU budget for peak requests",
        )
    else:
        check("cluster" not in cfg, "cluster settings are only valid on eidf")
    return cfg
