"""Pure host-side sizing. The declared envelope is one node, never cluster totals."""

import re

from .config import check, table
from .quota import DEFAULTS, GPU_ROLES, peak

BOUNDS = {"gpus": (0, 64), "cpus": (1, 4096), "memory_gib": (1, 65536)}


def _bounded(key, value):
    lo, hi = BOUNDS[key]
    check(type(value) is int and lo <= value <= hi, f"invalid single-node {key}")
    return value


def normalize(cfg, overrides, *, roles=None):
    capacity = table(cfg.setdefault("capacity", {}), BOUNDS)
    explicit = set(capacity) == set(BOUNDS)
    declared_axes = sorted(capacity)
    # Conservative single-node envelope, not a promise of successful calibration.
    for key, default in (("gpus", 1), ("cpus", 16), ("memory_gib", 96)):
        _bounded(key, capacity.setdefault(key, default))
    r = table(cfg.setdefault("resources", {}), (*BOUNDS, "answer_tp", *DEFAULTS))
    overrides = dict(
        table({} if overrides is None else overrides, (*BOUNDS, "memory", "answer_tp"))
    )
    if "memory" in overrides:
        check("memory_gib" not in overrides, "duplicate memory override")
        value = overrides.pop("memory")
        if isinstance(value, str) and re.fullmatch(r"[1-9]\d*Gi", value):
            value = value[:-2]
        overrides["memory_gib"] = value
    for key, value in overrides.items():
        if isinstance(value, str) and re.fullmatch(r"[0-9]{1,6}", value):
            overrides[key] = int(value)
    cfg["resource_policy"] = {
        "explicit_envelope": explicit,
        "declared_axes": declared_axes,
        "requested": {k: overrides.get(k, r.get(k, "auto")) for k in (*BOUNDS, "answer_tp")},
        "per_role": {k: dict(v) for k, v in r.items() if k in DEFAULTS and isinstance(v, dict)},
        "source": "declared_unverified",
    }
    totals = {}
    for key in BOUNDS:
        value = overrides.get(key, r.get(key, "auto"))
        if value == "auto":
            value = capacity[key]
        totals[key] = _bounded(key, value)
        if key not in declared_axes:
            capacity[key] = max(value, capacity[key])
        check(value <= capacity[key], f"{key} exceeds declared single-node capacity")
    count = totals["gpus"]
    if roles is None and count == 0:
        roles = {"analysis-report"}
    tp = overrides.get("answer_tp", r.get("answer_tp", "auto"))
    if tp == "auto":
        tp = 2 if count >= 2 and count % 2 == 0 else 1
    check(
        type(tp) is int and tp in (1, 2, 4, 8) and (count == 0 or count % tp == 0),
        "answer_tp must divide the single-node GPU count",
    )
    cfg["topology"] = {**totals, "answer_tp": tp}
    r.update(cfg["topology"])
    # Respect explicit per-role requests. Only automatic values may shrink.
    for role, (cpu, mem) in DEFAULTS.items():
        spec = table(r.setdefault(role, {}), ("cpu", "memory_gib"))
        for key, total, default in (("cpu", "cpus", cpu), ("memory_gib", "memory_gib", mem)):
            reserved = []
            if role not in ("postgres", "minio") and roles != {"analysis-report"}:
                reserved = ["postgres", "minio"]
            if role == "gpu":
                reserved.append("control")
            free = totals[total] - sum(r[s][key] for s in reserved)
            auto = free if role in GPU_ROLES else min(default, free)
            active = roles is None or role in roles
            val = spec.setdefault(key, auto if active else max(1, auto))
            check(
                type(val) is int and val > 0 and (not active or val <= free),
                f"insufficient {key} budget for {role}",
            )
    if count and (roles is None or set(GPU_ROLES).intersection(roles)):
        check(
            all(r[s]["cpu"] >= count for s in GPU_ROLES if roles is None or s in roles),
            "insufficient GPU pod CPUs for the requested replica pool",
        )
    usage = peak(cfg, paid_workers=bool(count), roles=roles, services=roles != {"analysis-report"})
    check(
        usage["cpu"] <= totals["cpus"] and usage["memory"] <= totals["memory_gib"] * 2**30,
        "phase exceeds single-node resource budget",
    )
