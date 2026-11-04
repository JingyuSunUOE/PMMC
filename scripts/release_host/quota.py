"""Resource requests and conservative namespace headroom checks (no cluster clients)."""

import re
from decimal import Decimal

DEFAULTS = {
    "postgres": (1, 2),
    "minio": (1, 2),
    "control": (2, 8),
    "gpu": (4, 32),
    "prepare-capacity": (4, 32),
    "prepare": (2, 8),
    "assets": (2, 4),
    "analysis-report": (2, 8),
    "iam": (1, 1),
    "bootstrap": (1, 4),
    "models": (2, 8),
    "pointer": (1, 1),
    "quote": (4, 16),
    "approve": (4, 16),
}


GPU_ROLES = ("gpu", "prepare-capacity")


def peak(cfg, paid_workers=True, *, roles=None, services=True):
    """Peak of sequential phases, not the sum of mutually exclusive Jobs."""
    r = cfg["resources"]
    if roles is None:
        roles = set(DEFAULTS) - (set() if paid_workers else set(GPU_ROLES))
    roles = set(roles) - {"postgres", "minio"}
    phases = [[], *([s] for s in roles)]
    if {"control", "gpu"} <= roles:
        phases.append(["control", "gpu"])
    if services:
        phases = [["postgres", "minio", *phase] for phase in phases]
    return {
        "cpu": max(sum(r[s]["cpu"] for s in row) for row in phases),
        "memory": max(sum(r[s]["memory_gib"] for s in row) for row in phases) * 2**30,
        "nvidia.com/gpu": cfg["topology"]["gpus"] if roles.intersection(GPU_ROLES) else 0,
    }


def quantity(q):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z]*)", str(q))
    mult = {
        "": 1,
        "n": Decimal("1e-9"),
        "u": Decimal("1e-6"),
        "m": Decimal(".001"),
        **{k: 1000**i for i, k in enumerate(("k", "M", "G", "T", "P", "E"), 1)},
        **{k: 1024**i for i, k in enumerate(("Ki", "Mi", "Gi", "Ti", "Pi", "Ei"), 1)},
    }
    if not m or m[2] not in mult:
        raise ValueError("unsupported Kubernetes resource quantity")
    return Decimal(m[1]) * mult[m[2]]


def headroom(cfg, quotas, pods, *, workers=True, roles=None, services=True):
    report = []
    peak_use = peak(cfg, workers, roles=roles, services=services)
    for quota in quotas:
        status = quota.get("status", {})
        for key, limit in (status.get("hard") or quota["spec"]["hard"]).items():
            short = key.split(".", 1)[1] if key.startswith(("requests.", "limits.")) else key
            if short not in peak_use:
                continue
            if key not in status.get("used", {}):
                raise ValueError("namespace quota usage is unavailable")
            hard, used = quantity(limit), quantity(status["used"][key])
            held = Decimal(0)
            # With scoped quotas we cannot prove these Pods count towards usage.
            if services and not (quota["spec"].get("scopes") or quota["spec"].get("scopeSelector")):
                for pod in pods:
                    if pod.get("status", {}).get("phase") in ("Failed", "Succeeded"):
                        continue
                    for c in pod["spec"]["containers"]:
                        # Only persistent services are already included in every phase.
                        # Other active Jobs may belong to a different run.
                        if c["name"] not in ("postgres", "minio"):
                            continue
                        field = "limits" if key.startswith("limits.") else "requests"
                        held += quantity(c.get("resources", {}).get(field, {}).get(short, 0))
            projected = max(Decimal(0), used - held) + peak_use[short]
            report.append(
                {
                    "resource": key,
                    "hard": str(hard),
                    "used": str(used),
                    "projected": str(projected),
                    "fits": projected <= hard,
                }
            )
    return report


def pending_reason(conditions):
    for c in reversed(conditions):
        if c.get("status") == "True":
            continue
        message = str(c.get("message", "")).lower()
        if any(s in message for s in ("quota", "insufficient", "exceed")):
            return "insufficient_quota_or_resources"
        if c.get("reason") in (
            "Pending",
            "Inadmissible",
            "Unschedulable",
            "AdmissionCheck",
            "Inactive",
        ):
            return c["reason"]
    return "awaiting_admission"
