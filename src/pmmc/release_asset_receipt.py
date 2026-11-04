"""Dependency-free verification shared by the assets image and the release host."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pmmc.release_plan import BENCHMARKS, plan_hash

SOURCE_LOCK = Path(__file__).resolve().parents[2] / "configs/release/benchmarks.lock.json"


def validate_asset_receipt(payload, plan=None, *, lock_path=None) -> dict:
    """Certify exact locked counts/trees; with a plan, also verify assignment and intent."""
    raw = (SOURCE_LOCK if lock_path is None else lock_path).read_bytes()
    lock = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("invalid_release_assets_receipt")
    sources = payload.get("sources")
    if (
        payload.get("protocol") != "release-assets-v2"
        or payload.get("ready") is not True
        or not isinstance(payload.get("plan_hash"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", payload["plan_hash"])
        or not isinstance(sources, dict)
        or lock["protocol"] != "pmmc-public-data-lock-v1"
        or sources.get("protocol") != lock["protocol"]
        or sources.get("lock_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("invalid_release_assets_receipt")
    trees = sources.get("benchmarks")
    counts = payload.get("file_counts")
    if (
        not isinstance(trees, dict)
        or not trees
        or not set(trees) <= set(BENCHMARKS)
        or not isinstance(counts, dict)
        or set(counts) != set(trees)
    ):
        raise ValueError("invalid_release_assets_benchmarks")
    if plan is not None and (
        payload["plan_hash"] != plan_hash(plan) or set(trees) != set(plan["benchmarks"])
    ):
        raise ValueError("release_assets_request_mismatch")
    for name, tree in trees.items():
        spec = lock["datasets"][name]
        if (
            not isinstance(tree, dict)
            or tree.get("state") not in ("installed", "verified", "repaired")
            or type(tree.get("files")) is not int
            or tree["files"] != spec["file_count"]
            or type(counts[name]) is not int
            or counts[name] != tree["files"]
            or tree.get("materialized_sha256") != spec["materialized_sha256"]
        ):
            raise ValueError("release_assets_locked_tree_mismatch")
    return {"plan_hash": payload["plan_hash"], "benchmarks": sorted(trees), "file_counts": counts}
