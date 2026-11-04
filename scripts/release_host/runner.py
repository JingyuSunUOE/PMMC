"""Durable single-controller workflow. Backend Jobs/containers outlive the CLI."""

import fcntl
import hashlib
import io
import json
import os
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .backends import Compose, Kueue, ReceiptError
from .config import ROOT, SUITE, VARIANTS, Error, approval_id, check, distinct_paths, identifier
from .workloads import credentials


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Error(f"missing or invalid {path.name}") from exc


@contextmanager
def lock(path):
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Error("another host launcher owns this deployment") from exc
        yield


def state_root():
    path = (
        Path(os.getenv("PMMC_DEPLOY_STATE", "~/.local/state/pmmc-release")).expanduser().resolve()
    )
    check(not path.is_relative_to(ROOT), "host state must be outside the source checkout")
    return path


def bundle(folder):
    path = folder / "diagnostics.tar.gz"
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
        temp = Path(f.name)
    try:
        with tarfile.open(temp, mode="w:gz") as tar:
            for item in (
                "deployment.json",
                "progress.json",
                "events.jsonl",
                *(p.name for p in sorted(folder.glob("evidence-*.json"))[-32:]),
            ):
                if (folder / item).is_file():
                    if item == "events.jsonl":
                        with (folder / item).open("rb") as events:
                            size = os.fstat(events.fileno()).st_size
                            events.seek(max(0, size - 512 * 1024))
                            if events.tell():
                                events.readline(512 * 1024)
                            data = events.read(512 * 1024)
                        header = tarfile.TarInfo(item)
                        header.size = len(data)
                        tar.addfile(header, io.BytesIO(data))
                        continue
                    tar.add(folder / item, arcname=item, recursive=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def save(folder, state, event):
    state["updated"] = now()
    state["event"] = event
    atomic(folder / "progress.json", state)
    record = {k: state[k] for k in ("updated", "run", "task", "stage", "status", "event", "batch")}
    line = json.dumps(record, sort_keys=True)
    with (folder / "events.jsonl").open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def collect_review(folder, state, backend):
    record = {
        "task": state["task"],
        "attempt": state["attempt"],
        "status": state["status"],
        "workloads": {},
        "event": state["event"],
        "host_error": state.get("host_error"),
        "admission": getattr(backend, "quota", None),
    }
    try:
        record["services"] = backend.health()
    except (Error, OSError, ValueError):
        record["services"] = {"state": "unavailable"}
    for role, info in (state["batch"] or {}).items():
        try:
            record["workloads"][role] = backend.tail(info["job"])
        except (Error, OSError, ValueError):
            record["workloads"][role] = {"state": "unavailable"}
    try:
        record["native"] = backend.evidence(state["run"], state["request"]["variant"])
    except (Error, OSError, ValueError):
        record["native"] = {"state": "collection_unavailable"}
    atomic(folder / f"evidence-{state['attempt']:08d}.json", record)


def groups(kind):
    if kind in ("quote", "approve"):
        return [["iam"], ["bootstrap"], ["prepare"], [kind]]
    return [["iam"], ["bootstrap"], ["models"], ["prepare"], ["pointer"], ["control", "gpu"]]


def collect_batch(folder, state, backend, name, *, sleep=time.sleep):
    batch = state["batch"]
    if not batch:
        tag = digest([state["run"], state["task"], state["attempt"]])[:12]
        batch = state["batch"] = {
            role: {"job": f"{name}-{tag}-{role}", "state": "planned", "submitted": False}
            for role in groups(state["request"]["kind"])[state["index"]]
        }
        save(folder, state, "intent")
    since = time.monotonic()
    while True:
        for role, info in batch.items():
            if any(i["state"] == "failed" for i in batch.values()):
                break
            if info["state"] in (
                "complete",
                "failed",
                "awaiting_budget",
                "cancelled",
                "waiting_for_workers",
            ):
                continue
            job = info["job"]
            actual = backend.state(job)
            if actual == "missing":
                check(
                    not info["submitted"],
                    "previously submitted workload missing; manual review required",
                )
                backend.start(job, role, state["run"], state["stage"])
                actual = "submitted"
            info["submitted"] = True
            if actual.startswith("queued:"):
                queued_since = info.setdefault("queued_since", time.time())
                timeout = backend.cfg["cluster"]["admission_timeout_seconds"]
                if time.time() - queued_since >= timeout:
                    info.update(reason="admission_timeout:" + actual, stop_requested=True)
                    actual = "failed"
            else:
                info.pop("queued_since", None)
            if actual == "complete":
                try:
                    actual = backend.outcome(job, role)
                    if role == "prepare":
                        if (
                            actual["run_id"] != state["run"]
                            or actual["variant"] != state["request"]["variant"]
                        ):
                            raise ReceiptError("native pointer run or variant mismatch")
                        info["receipt"] = actual
                        actual = "complete"
                    elif role in ("approve", "quote"):
                        req = state["request"]
                        if role == "approve" and (
                            actual["selection_hash"] != req["selection_hash"]
                            or Decimal(actual["limit_usd"]) != Decimal(req["limit_usd"])
                        ):
                            raise ReceiptError("approval receipt mismatch")
                        if role == "quote" and actual["scope"] != state["stage"]:
                            raise ReceiptError("quote scope mismatch")
                        info["receipt"] = state["result"] = actual
                        actual = "complete"
                    elif isinstance(actual, dict):
                        if actual["stage"] != state["stage"] or actual["paid_authorized"] != bool(
                            state["approval"]
                        ):
                            raise ReceiptError("native stage or paid authorization mismatch")
                        info["receipt"] = actual
                        actual = (
                            "complete" if actual["state"] == "stage_complete" else actual["state"]
                        )
                except ReceiptError as exc:
                    info["reason"] = str(exc)
                    actual = "failed"
            if actual != info["state"]:
                info["state"] = actual
                save(folder, state, "workload")
            if actual == "failed":
                break
        if any(i["state"] == "failed" for i in batch.values()):
            for info in batch.values():
                if info.get("stop_requested") or info["state"] not in (
                    "complete",
                    "failed",
                    "awaiting_budget",
                    "cancelled",
                    "waiting_for_workers",
                ):
                    current = backend.state(info["job"])
                    if current not in ("missing", "failed", "complete"):
                        backend.stop(info["job"])
                    if not info.pop("stop_requested", False):
                        info["state"] = "cancelled"
                    save(folder, state, "peer_stopped")
            return "failed"
        if all(
            i["state"] in ("complete", "awaiting_budget", "waiting_for_workers")
            for i in batch.values()
        ):
            if "control" in batch:
                receipts = [batch[r].get("receipt", {}) for r in ("control", "gpu")]
                if not receipts[0] or receipts[0].get("scope_hash") != receipts[1].get(
                    "scope_hash"
                ):
                    save(folder, state, "native_scope_mismatch")
                    return "failed"
                return batch["control"]["state"]
            return "complete"
        if time.monotonic() - since >= 30:
            for info in batch.values():
                if info["state"] == "running":
                    try:
                        info["native_progress"] = backend.tail(info["job"])[-2:]
                    except (Error, OSError, ValueError):
                        info["native_progress"] = []
            save(folder, state, "heartbeat")
            since = time.monotonic()
        sleep(5)


def execute(folder, state, backend, *, sleep=time.sleep):
    while state["index"] < len(groups(state["request"]["kind"])):
        result = collect_batch(folder, state, backend, state["deployment"], sleep=sleep)
        if result != "complete":
            state["status"] = result
            break
        state["batch"] = None
        state["index"] += 1
        save(folder, state, "group_complete")
    else:
        state["status"] = "completed"
    if state["request"]["kind"] == "run":
        state["units"][state["task"]] = state["status"]
        if state["status"] == "completed" and state["approval"]:
            state["approvals"][state["request"]["variant"]] = state["approval"]
        state["completed_stages"] = [
            stage
            for stage, units in MILESTONES.items()
            if all(state["units"].get(unit) == "completed" for unit in units)
        ]
    elif state["status"] == "completed" and state["request"]["kind"] == "approve":
        state["approvals"][state["request"]["variant"]] = state["result"]["approval_id"]
    save(folder, state, "task_terminal")


MILESTONES = {
    "main": ["wr-main"],
    "analysis": ["wr-analysis", "w-main", "r-main", "legacy-main"],
    "baselines": ["wr-baselines"],
}


def launch(
    root,
    cfg,
    run,
    stage,
    approval=None,
    *,
    resume=False,
    variant=None,
    action=None,
    factory=None,
    sleep=time.sleep,
):
    identifier(run)
    if approval:
        approval_id(approval)
    if cfg["profile"] == "local-nvidia":
        distinct_paths([str(root), *cfg["mounts"].values()])
    private = root / "instances" / cfg["instance"]
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    check(private.stat().st_mode & 0o077 == 0, "instance directory must be private")
    folder = root / "runs" / run
    with lock(private / "lock"):
        active_file = private / "active.json"
        check(
            not active_file.exists() or read(active_file) in (None, run),
            "resume the deployment's active run first",
        )
        identity = private / "identity.json"
        if not identity.exists():
            check(
                not resume and not folder.exists(), "deployment identity lost; restore host backup"
            )
            name = cfg["instance"] + "-" + uuid.uuid4().hex[:8]
            atomic(
                identity, {"deployment": name, "config": digest(cfg), "secrets": credentials(name)}
            )
        owner = read(identity)
        check(
            owner["config"] == digest(cfg), "deployment configuration changed; use a new instance"
        )
        if not folder.exists():
            check(not resume, "unknown run ID")
            check(stage == "main", "start main before follow-up stages")
            folder.mkdir(parents=True, mode=0o700)
            atomic(folder / "deployment.json", cfg)
            state = {
                "run": run,
                "suite": SUITE,
                "stage": stage,
                "status": "new",
                "attempt": 0,
                "index": 0,
                "batch": None,
                "completed_stages": [],
                "approval": None,
                "task": None,
                "units": {},
                "approvals": {},
                "work_stage": stage,
                "work_variant": variant,
                "deployment": owner["deployment"],
                "native_output": f"/workspace/outputs/step8/exploration/{run}",
            }
            save(folder, state, "created")
        else:
            check(read(folder / "deployment.json") == cfg, "run configuration mismatch")
            state = read(folder / "progress.json")
        if resume and not (stage or variant or action):
            if state["status"] == "running" and state["request"]["kind"] != "run":
                action = state["request"]
                stage = state["stage"]
            else:
                stage, variant = state["work_stage"], state["work_variant"]
        stage = stage or state["work_stage"]
        check(variant is None or variant in VARIANTS, "unknown variant")
        if action:
            requests = [(stage, action)]
        else:
            if variant not in (None, "wr") and stage == "main":
                stage = "analysis"
            check(not variant or stage == "analysis" or variant == "wr", "invalid stage/variant")
            check(
                not approval or variant or stage != "analysis",
                "select one variant for external approval",
            )
            requests = [
                (s, {"kind": "run", "variant": v})
                for unit in MILESTONES[stage]
                for v, s in [unit.split("-")]
                if not variant or variant == v
            ]
            if stage != "main":
                predecessor = "main" if stage == "analysis" else "analysis"
                check(
                    predecessor in state["completed_stages"],
                    f"complete all {predecessor} units first",
                )

        def key(s, req):
            prefix = "" if req["kind"] == "run" else req["kind"] + "-"
            return f"{prefix}{req['variant']}-{s}"

        if state["status"] == "running":
            check(
                any(
                    key(s, req) == state["task"] and req == state["request"] for s, req in requests
                ),
                "resume the interrupted operation before switching",
            )
            requests.sort(key=lambda pair: key(*pair) != state["task"])
        else:
            atomic(active_file, None)
        waiting = False
        for native_stage, req in requests:
            task = key(native_stage, req)
            v = req["variant"]
            if v != "wr":
                check(
                    "main" in state["completed_stages"],
                    "complete WR main before alternate variants",
                )
            paid = (approval or state["approvals"].get(v)) if req["kind"] == "run" else None
            previous = state["units"].get(task)
            if state["status"] == "running" and task == state["task"]:
                paid = approval or state["approval"]
                previous = None
            if previous == "failed":
                check(resume, "use resume to retry the failed scientific unit")
            if previous == "completed":
                continue
            if previous == "awaiting_budget" and not paid:
                print(f"{task}: awaiting_budget (not completed)")
                waiting = True
                continue
            if state["status"] != "running":
                state.update(
                    stage=native_stage,
                    request=req,
                    task=task,
                    status="running",
                    attempt=state["attempt"] + 1,
                    index=0,
                    batch=None,
                    result=None,
                    approval=paid,
                    host_error=None,
                )
                if not action:
                    state["work_stage"] = stage
                    state["work_variant"] = variant
            else:
                check(
                    not approval or paid == state["approval"],
                    "cannot change approval for a live attempt",
                )
            atomic(active_file, run)
            save(folder, state, "resume" if resume else "launch")
            cls = factory or (Compose if cfg["profile"] == "local-nvidia" else Kueue)
            backend = cls(
                cfg, owner["deployment"], private, owner["secrets"], state["approval"], req
            )
            try:
                if not state["batch"]:
                    backend.reconcile()
                execute(folder, state, backend, sleep=sleep)
            except (Error, OSError, KeyboardInterrupt) as exc:
                state["host_error"] = str(exc) if isinstance(exc, Error) else type(exc).__name__
                save(folder, state, "interrupted_or_backend_error")
                raise
            finally:
                collect_review(folder, state, backend)
                print(f"SYNC_REVIEW_FILE={bundle(folder)}", flush=True)
            atomic(active_file, None)
            if state["status"] == "failed":
                return 1
            waiting |= state["status"] == "awaiting_budget"
        return 3 if waiting else 0
