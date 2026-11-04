"""Normalize deployment inputs before any backend or credential operations."""

import re
import tomllib
from pathlib import Path, PurePosixPath

PROFILES = ("kubernetes",)
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
RATE_FILES = ("rate-card.yaml", "provider-policy.json")
SECRET_RATE_PATH = TARGETS["inputs"] + "/" + RATE_FILES[0]


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


def label_part(value):
    return (
        isinstance(value, str)
        and len(value) <= 63
        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?", value)
    )


def label_key(value):
    if not isinstance(value, str):
        return False
    prefix, sep, name = value.rpartition("/")
    return label_part(name) and (
        not sep
        or (
            len(prefix) <= 253
            and all(len(p) <= 63 and re.fullmatch(DNS, p) for p in prefix.split("."))
        )
    )


def placement(cluster):
    selector = cluster.setdefault("node_selector", {})
    check(
        isinstance(selector, dict)
        and all(label_key(k) and (v == "" or label_part(v)) for k, v in selector.items()),
        "invalid Kubernetes node selector",
    )
    check(
        selector.get("kubernetes.io/os", "linux") == "linux",
        "release workloads require Linux nodes",
    )
    if host := selector.get("kubernetes.io/hostname"):
        check(
            host == cluster.setdefault("node_hostname", host), "conflicting node hostname selectors"
        )
    tolerations = cluster.setdefault("tolerations", [])
    check(isinstance(tolerations, list), "tolerations must be a list")
    for tol in tolerations:
        table(tol, ("key", "operator", "value", "effect", "tolerationSeconds"), ("key",))
        op, value = tol.setdefault("operator", "Equal"), tol.get("value", "")
        check(
            label_key(tol["key"])
            and op in ("Exists", "Equal")
            and (value == "" or label_part(value))
            and (op != "Exists" or value == "")
            and tol.get("effect", "") in ("", "NoExecute", "NoSchedule", "PreferNoSchedule"),
            "invalid Kubernetes toleration",
        )
        if "tolerationSeconds" in tol:
            check(
                tol.get("effect") == "NoExecute"
                and type(tol["tolerationSeconds"]) is int
                and 0 < tol["tolerationSeconds"] <= 2**63 - 1,
                "invalid NoExecute duration",
            )


def load_config(file, profile, *, resource_overrides=None):
    try:
        with Path(file).open("rb") as stream:
            cfg = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise Error("cannot read deployment TOML") from exc
    cfg["images"] = verified_images(cfg.get("images"))
    return validate_config(cfg, profile, resource_overrides=resource_overrides)


def verified_images(overrides):
    if __package__ == "release_host":  # ./scripts/pmmc
        from release_images import load_images
    else:
        from ..release_images import load_images
    try:
        return load_images(ROOT, overrides)
    except (ValueError, RecursionError) as exc:
        raise Error(
            "verify deploy/release/images.lock.json, PMMC_IMAGE_LOCK_SHA256 and image mirrors"
        ) from exc


def verify_frozen_images(cfg):
    check(
        verified_images(cfg.get("images")) == cfg.get("images"),
        "frozen deployment does not match this release's images",
    )


def validate_config(cfg, profile, *, resource_overrides=None):
    """Validate a parsed deployment, including one decrypted in memory."""
    from .topology import normalize

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
        "capacity",
        "billing",
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
    for v in mounts.values():
        clean_path(v, absolute=False)
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
    billing = table(cfg.setdefault("billing", {}), ("policy_secret", "preferred_mode"))
    check(type(billing.setdefault("policy_secret", False)) is bool, "policy_secret must be boolean")
    check(
        billing.setdefault("preferred_mode", "finite") in ("finite", "uncapped"),
        "preferred_mode must be finite or uncapped",
    )
    if billing["policy_secret"]:
        check(
            native.setdefault("rate_card", SECRET_RATE_PATH) == SECRET_RATE_PATH,
            "billing policy Secret requires native.rate_card = " + SECRET_RATE_PATH,
        )
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
    normalize(cfg, resource_overrides)
    if profile == "kubernetes":
        keys = ("context", "namespace", "shared_pvc", "storage_class")
        cluster = table(
            cfg.get("cluster"),
            (
                *keys,
                "queue",
                "shared_access_mode",
                "node_hostname",
                "node_selector",
                "tolerations",
                "discovery",
                "postgres_storage_gib",
                "minio_storage_gib",
                "image_pull_secrets",
                "create_shared_pvc",
                "shared_storage_class",
                "shared_storage_gib",
                "admission_timeout_seconds",
            ),
            keys,
        )
        for k in (
            *keys,
            *(k for k in ("queue", "node_hostname", "shared_storage_class") if k in cluster),
        ):
            v = cluster[k]
            pattern = (
                r"[A-Za-z0-9][A-Za-z0-9_.:@/-]*"
                if k == "context"
                else (DNS if k in ("namespace", "queue") else rf"{DNS}(?:\.{DNS})*")
            )
            check(
                isinstance(v, str) and re.fullmatch(pattern, v) and len(v) <= 253,
                "invalid cluster identifier",
            )
        mode = cluster.setdefault("shared_access_mode", "ReadWriteMany")
        placement(cluster)
        pulls = cluster.setdefault("image_pull_secrets", [])
        check(
            isinstance(pulls, list)
            and all(
                isinstance(v, str) and len(v) <= 253 and re.fullmatch(rf"{DNS}(?:\.{DNS})*", v)
                for v in pulls
            )
            and len(set(pulls)) == len(pulls),
            "invalid image_pull_secrets",
        )
        create = cluster.setdefault("create_shared_pvc", False)
        check(type(create) is bool, "create_shared_pvc must be boolean")
        if create:
            check(
                "shared_storage_class" in cluster
                and type(cluster.get("shared_storage_gib")) is int
                and 1 <= cluster["shared_storage_gib"] <= 65536,
                "managed shared PVC requires an explicit storage class and size in GiB",
            )
        else:
            check(
                not {"shared_storage_gib", "shared_storage_class"}.intersection(cluster),
                "shared storage provisioning settings require create_shared_pvc=true",
            )
        discovery = cluster.setdefault("discovery", "live")
        check(discovery in ("live", "declared"), "discovery must be live or declared")
        check(
            discovery != "declared" or cfg["resource_policy"]["explicit_envelope"],
            "declared discovery requires an explicit single-node capacity envelope",
        )
        check(mode in ("ReadWriteMany", "ReadWriteOnce"), "unknown shared PVC access mode")
        check(
            mode != "ReadWriteOnce" or "node_hostname" in cluster,
            "single-node RWO shared storage requires a node_hostname",
        )
        check(
            cluster["shared_pvc"] not in {cfg["instance"] + "-" + s for s in ("postgres", "minio")},
            "shared PVC must be independent of service PVCs",
        )
        for k, default, high in (
            ("postgres_storage_gib", 32, 65536),
            ("minio_storage_gib", 64, 65536),
            ("admission_timeout_seconds", 600, 86400),
        ):
            check(
                type(cluster.setdefault(k, default)) is int and 1 <= cluster[k] <= high,
                "invalid service PVC size or admission timeout",
            )
    else:
        check("cluster" not in cfg, "cluster settings require the kubernetes profile")
    return cfg
