"""Bounded projections only; can run on the host or standalone in a PVC reader Job."""

import hashlib
import heapq
import json
import math
import re
import sys
from pathlib import Path

LIMIT = 512 * 1024
HASHES = {
    "selection_hash",
    "protocol_hash",
    "scope_hash",
    "partition_hash",
    "campaign_sha256",
    "config_sha256",
    "exposure_hash",
    "job_inventory_sha256",
    "source_identity",
    "approval_id",
}
NUMBERS = {
    "completed",
    "scoped_jobs",
    "inflight",
    "pending",
    "resumed",
    "total",
    "correct",
    "accuracy",
    "mean",
    "mean_observed",
    "selected",
    "coverage",
    "denominator",
    "numerator",
    "count",
    "observed",
    "bytes",
    "bytes_received",
    "total_bytes",
    "bytes_downloaded",
    "cached_files",
    "files",
    "files_completed",
    "elapsed_seconds",
    "items_per_second",
    "cache_hits",
}
FLAGS = {
    "paid_authorized",
    "paper_result",
    "ready",
    "unknown_provider_outcome",
    "scientific_approval",
}
STATES = {
    "awaiting_budget",
    "stage_complete",
    "waiting_for_workers",
    "failed",
    "completed",
    "pending",
    "running",
    "planned",
    "control",
    "gpu",
    "main",
    "analysis",
    "baselines",
}


def project(data):
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if k in HASHES and isinstance(v, str) and re.fullmatch("[a-f0-9]{64}", v):
            out[k] = v
        elif k in NUMBERS and type(v) in (int, float) and abs(v) < 10**15 and math.isfinite(v):
            out[k] = v
        elif k in FLAGS and type(v) is bool:
            out[k] = v
        elif k in ("state", "stage", "worker_pool") and isinstance(v, str) and v in STATES:
            out[k] = v
        elif k == "dataset" and v in ("memlens", "memgallery"):
            out[k] = v
        elif k == "phase" and v in (
            "check",
            "download",
            "materialize",
            "repaired",
            "installed",
            "verified",
        ):
            out[k] = v
        elif (
            k in ("error_code", "code")
            and isinstance(v, str)
            and re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,95}", v)
        ):
            out[k] = v
        elif k == "frames" and isinstance(v, list):
            out[k] = [
                dict(module=f["module"], function=f["function"], line=f["line"])
                for f in v[-8:]
                if isinstance(f, dict)
                and re.fullmatch(r"pmmc(?:\.[A-Za-z_][A-Za-z0-9_]*)+", str(f.get("module", "")))
                and re.fullmatch("[A-Za-z_][A-Za-z0-9_]{0,99}", str(f.get("function", "")))
                and type(f.get("line")) is int
                and 0 < f["line"] < 10**7
            ]
    return out


def safe_log(text):
    events = []
    for line in text[-65536:].splitlines()[-128:]:
        start = line.find("{")
        if start >= 0:
            try:
                value = json.loads(line[start:])
                event = re.search(
                    r"(?:^|\s)(STEP8_[A-Z0-9_]{1,48}|PMMC_[A-Z0-9_]{1,48})\s*$", line[:start]
                )
                if not isinstance(value, dict):
                    continue
                if not event and value.get("protocol") not in (
                    "step8-bootstrap-v1",
                    "step8-model-preparation-v1",
                    "step8-minio-iam-v1",
                    "pmmc-release-stage-v1",
                    "release-assets-v2",
                ):
                    continue
                fields = project(value)
            except (ValueError, RecursionError):
                continue
            if fields:
                events.append(fields)
        else:
            frame = re.search(
                r'File "(?:[^"]*/)?pmmc/([a-zA-Z0-9_/]+)\.py", line (\d+), in (\w+)', line
            )
            if frame:
                events.append(
                    project(
                        {
                            "frames": [
                                dict(
                                    module="pmmc." + frame[1].replace("/", "."),
                                    line=int(frame[2]),
                                    function=frame[3],
                                )
                            ]
                        }
                    )
                )
            else:
                error = re.match(
                    r"^(ValueError|OSError|PermissionError|FileNotFoundError|"
                    r"RuntimeError|ModuleNotFoundError|ImportError|KeyError):",
                    line,
                )
                if error:
                    events.append({"exception_type": error[1]})
    return events[-24:]


def collect(outputs, run, variant):
    root = outputs / "step8"
    files = {}
    report = {"schema": "pmmc-host-evidence-v1", "files": files}

    def read(path, label):
        if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
            return {}
        size = path.stat().st_size
        row = files[label] = {"bytes": size}
        with path.open("rb") as stream:
            if path.suffix == ".log":
                stream.seek(max(0, size - 65536))
                row["events"] = safe_log(stream.read(65536).decode(errors="replace"))
                return {}
            limit = 4 * LIMIT if label.startswith("reports/") else LIMIT
            if size > limit:
                row["omitted"] = "size_limit"
                return {}
            payload = stream.read(limit + 1)
        if len(payload) != size:
            row["omitted"] = "changed_during_read"
            return {}
        row["sha256"] = hashlib.sha256(payload).hexdigest()
        try:
            value = json.loads(payload)
        except (ValueError, RecursionError):
            return {}
        if not isinstance(value, dict):
            return {}
        row["fields"] = project(value.get("failure", value))
        # Only numeric aggregate leaves, with fixed schema labels; never query IDs.
        if label.startswith("reports/"):
            metrics = []
            nodes = [value]
            for _ in range(1024):
                if not nodes or len(metrics) == 32:
                    break
                node = nodes.pop()
                if isinstance(node, dict):
                    nums = {k: v for k, v in project(node).items() if k in NUMBERS}
                    if nums:
                        metrics.append(nums)
                    nodes.extend(v for v in list(node.values())[:64] if isinstance(v, (dict, list)))
                elif isinstance(node, list):
                    nodes.extend(node[:32])
            if metrics:
                row["aggregate_samples"] = metrics
        return value

    pointer = read(root / "exploration" / run / variant / "pointer.json", "pointer.json")
    directory = pointer.get("campaign_dir")
    if not (
        pointer.get("run_id") == run
        and pointer.get("variant") == variant
        and isinstance(directory, str)
        and re.fullmatch(r"outputs/step8/[A-Za-z0-9._-]+_exploration_[a-f0-9]+", directory)
    ):
        report["pointer"] = "unavailable_or_invalid"
        return report
    campaign = root / Path(directory).name
    config = read(campaign / "manifests/config.json", "manifests/config.json")
    metadata = config.get("metadata") or {}
    if (
        not isinstance(metadata, dict)
        or metadata.get("exploration_run_id") != run
        or metadata.get("exploration_variant") != variant
    ):
        report["pointer"] = "campaign_identity_mismatch"
        return report
    read(campaign / "manifests/campaign.json", "manifests/campaign.json")
    selection = pointer.get("selection_hash")
    if isinstance(selection, str) and re.fullmatch("[a-f0-9]{64}", selection):
        read(campaign / f"manifests/selections/{selection}.json", "manifests/selection.json")
        read(campaign / f"manifests/budget-{selection}.json", "manifests/budget.json")
    for stage in ("main", "analysis", "baselines"):
        for suffix in ("control", "gpu", "metrics"):
            name = f"reports/release-{stage}-{suffix}.json"
            read(campaign / name, name)
        for name in (f"reports/exploration-{stage}.json", f"reports/budget-quote-{stage}.json"):
            read(campaign / name, name)
    attempts = campaign / "attempts"
    if attempts.resolve() == attempts and attempts.is_dir():
        latest = heapq.nlargest(
            8,
            (
                (p.stat().st_mtime_ns, p)
                for p in attempts.iterdir()
                if re.fullmatch(r"(startup|attempt)-[a-f0-9]{32}", p.name)
                and not p.is_symlink()
                and p.is_dir()
            ),
        )
        for _, path in latest:
            for name in ("failure-capsule.json", "terminal.json", "safe-execution.log"):
                read(path / name, f"attempts/{path.name}/{name}")
    return report


if __name__ == "__main__":
    print(json.dumps(collect(Path(sys.argv[1]), sys.argv[2], sys.argv[3]), separators=(",", ":")))
