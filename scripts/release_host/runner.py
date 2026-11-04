"""Durable single-controller workflow. Backend Jobs/containers outlive the CLI."""

import fcntl
import hashlib
import io
import json
import os
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from pmmc.release_plan import normalize_plan, plan_hash

from .backends import Kueue, ReceiptError, resolve_kubernetes_resources
from .config import ROOT, SUITE, Error, check, identifier
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
                    if item == "deployment.json":
                        cfg = read(folder / item)
                        data = json.dumps(
                            dict(
                                profile=cfg["profile"],
                                topology=cfg.get("topology"),
                                images={k: v.rsplit("@", 1)[-1] for k, v in cfg["images"].items()},
                            )
                        ).encode()
                        header = tarfile.TarInfo("deployment-summary.json")
                        header.size = len(data)
                        tar.addfile(header, io.BytesIO(data))
                        continue
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
    roles = ", ".join(f"{r}:{i['state']}" for r, i in (state["batch"] or {}).items())
    print(
        f"[{state['updated']}] {state['run']} {state['task'] or state['stage']} "
        f"{event} | {roles or state['status']}",
        flush=True,
    )
    if event == "heartbeat":
        for role, info in (state["batch"] or {}).items():
            for progress in info.get("native_progress", []):
                print(f"  {role}: {json.dumps(progress, sort_keys=True)}", flush=True)
    if state.get("result") and event == "task_terminal":
        print("PMMC_RECEIPT " + json.dumps(state["result"], sort_keys=True), flush=True)


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
    if kind == "services":
        return [["iam"], ["bootstrap"]]
    if kind == "analysis-report":
        return [["analysis-report"]]
    if kind in ("quote", "approve"):
        return [["iam"], ["bootstrap"], ["assets"], ["prepare"], [kind]]
    prep = [["iam"], ["bootstrap"], ["assets"], ["prepare"], ["models"]]
    return prep + (
        [["prepare-capacity"]] if kind == "prepare" else [["pointer"], ["control", "gpu"]]
    )


def planned_roles(plan, kind="run"):
    if kind == "run" and plan["stage"] == "analysis" and not plan["with_deps"]:
        kind = "analysis-report"
    roles = {role for group in groups(kind) for role in group}
    if kind == "run" and plan["stage"] in ("analysis", "all"):
        roles.add("analysis-report")
    return roles


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
                        state.setdefault("prepared", {})[actual["variant"]] = actual
                        actual = "complete"
                    elif role == "analysis-report":
                        if actual["plan_hash"] != state["plan_hash"]:
                            raise ReceiptError("analysis receipt belongs to another intent")
                        info["receipt"] = state["result"] = actual
                        actual = "complete"
                    elif role in ("approve", "quote"):
                        req = state["request"]
                        if role == "approve" and (
                            actual["selection_hash"] != req["selection_hash"]
                            or actual.get("plan_hash") != state["plan_hash"]
                            or (actual["limit_usd"] is None) != req["uncapped"]
                            or (
                                not req["uncapped"]
                                and Decimal(actual["limit_usd"]) != Decimal(req["limit_usd"])
                            )
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
                receipts = [batch[r].get("receipt", {}) for r in ("control", "gpu") if r in batch]
                if not all(receipts) or len({r.get("scope_hash") for r in receipts}) != 1:
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


def units(plan):
    stage, variants = plan["stage"], plan["variants"]
    if stage == "main":
        return [f"{v}-main" for v in variants]
    if stage == "baselines":
        return ["wr-baselines"]
    if stage == "analysis":
        return ["source-analysis"]
    result = ["wr-main"]
    if "ablation" in plan["analyses"]:
        result.append("wr-analysis")
    result.extend(f"{v}-main" for v in variants if v != "wr")
    return [*result, "source-analysis", "wr-baselines"]


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
        if state["request"]["kind"] == "analysis-report" and state["result"]["missing_units"]:
            state["status"] = "missing_dependencies"
    if state["request"]["kind"] == "run":
        state["units"][state["task"]] = state["status"]
    if state["request"]["kind"] == "approve" and state["status"] == "completed":
        state["authorized_plan"] = state["result"]["approval_id"]
    save(folder, state, "task_terminal")


def deployment_identity(cfg):
    # Worker budgets may differ across runs; data, credentials and service ownership may not.
    cluster = cfg["cluster"]
    return digest(
        {k: cfg[k] for k in ("instance", "identity", "mounts")}
        | {
            "context": cluster["context"],
            "namespace": cluster["namespace"],
            "shared_pvc": cluster["shared_pvc"],
            "storage_class": cluster["storage_class"],
            "postgres": cfg["images"]["postgres"],
            "minio": cfg["images"]["minio"],
        }
    )


def service_status(root, cfg):
    folder = root / "instances" / cfg["instance"]
    path = folder / "identity.json"
    if not path.is_file():
        return dict(state="uninitialized", instance=cfg["instance"])
    owner = read(path)
    check(owner["config"] == deployment_identity(cfg), "local service identity mismatch")
    backend = Kueue(cfg, owner["deployment"], folder, owner["secrets"], None, {"kind": "services"})
    return backend.health()


def launch(root, cfg, run, plan, *, action=None, resume=False, factory=None, sleep=time.sleep):
    identifier(run)
    plan = normalize_plan(plan)
    check(plan["run_id"] == run, "plan run ID mismatch")
    fingerprint = plan_hash(plan)
    private = root / "instances" / cfg["instance"]
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    check(private.stat().st_mode & 0o077 == 0, "instance directory must be private")
    folder = root / "runs" / run
    with lock(private / "lock"):
        requested_cfg = cfg
        if folder.exists():
            stored = read(folder / "deployment.json")
            check(
                cfg in (stored, read(folder / "requested-deployment.json")),
                "frozen deployment or resource limits changed",
            )
            cfg = stored
        elif factory is None:
            cfg = resolve_kubernetes_resources(
                cfg, roles=planned_roles(plan, (action or {}).get("kind", "run"))
            )
        active = private / "active.json"
        check(
            not active.exists() or read(active) in (None, run),
            "resume active deployment operation first",
        )
        identity = private / "identity.json"
        if not identity.exists():
            check(
                not resume and not folder.exists(),
                "lost deployment identity; do not reset existing services",
            )
            name = cfg["instance"]
            atomic(
                identity,
                dict(deployment=name, config=deployment_identity(cfg), secrets=credentials(name)),
            )
        owner = read(identity)
        check(owner["config"] == deployment_identity(cfg), "service/storage identity changed")
        if not folder.exists():
            check(not resume, "unknown run ID")
            folder.mkdir(parents=True, mode=0o700)
            atomic(folder / "deployment.json", cfg)
            atomic(folder / "requested-deployment.json", requested_cfg)
            atomic(folder / "plan.json", plan)
            state = dict(
                run=run,
                plan_hash=fingerprint,
                suite=SUITE,
                stage=plan["stage"],
                status="new",
                attempt=0,
                index=0,
                batch=None,
                approval=None,
                authorized_plan=None,
                task=None,
                units={},
                deployment=owner["deployment"],
                result=None,
            )
            save(folder, state, "created")
        else:
            check(
                read(folder / "deployment.json") == cfg and read(folder / "plan.json") == plan,
                "frozen plan or resource limits changed; use a new run ID",
            )
            state = read(folder / "progress.json")
        if resume and state["status"] == "running" and not action:
            if state["request"]["kind"] not in ("analysis-report", "run"):
                action = {k: v for k, v in state["request"].items() if k != "plan"}

        def invoke(stage, req, task):
            req = {**req, "plan": plan}
            paid = state.get("authorized_plan") if req["kind"] == "run" else None
            if state["status"] == "running":
                check(
                    task == state["task"] and req == state["request"],
                    "resume interrupted operation before switching",
                )
                paid = state["approval"]
            else:
                state.update(
                    stage=stage,
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
            atomic(active, run)
            save(folder, state, "resume" if resume else "launch")
            backend = (factory or Kueue)(
                cfg, owner["deployment"], private, owner["secrets"], paid, req
            )
            try:
                if not state["batch"]:
                    if req["kind"] == "analysis-report":
                        if hasattr(backend, "reconcile_reader"):
                            backend.reconcile_reader()
                    else:
                        backend.reconcile()
                execute(folder, state, backend, sleep=sleep)
            except (Error, OSError, KeyboardInterrupt) as exc:
                state["host_error"] = str(exc) if isinstance(exc, Error) else type(exc).__name__
                save(folder, state, "interrupted_or_backend_error")
                raise
            finally:
                collect_review(folder, state, backend)
                print(f"SYNC_REVIEW_FILE={bundle(folder)}", flush=True)
            atomic(active, None)
            return {"completed": 0, "awaiting_budget": 3, "missing_dependencies": 4}.get(
                state["status"], 1
            )

        if action:
            return invoke(action["stage"], action, action["kind"] + "-" + action["variant"])

        def scientific(unit):
            if state["units"].get(unit) == "completed":
                print(f"PMMC_CACHE unit={unit} state=complete", flush=True)
                return 0
            if state["units"].get(unit) == "awaiting_budget" and not state.get("authorized_plan"):
                print(f"PMMC_PAUSED unit={unit} reason=awaiting_budget", flush=True)
                return 3
            check(
                state["units"].get(unit) != "failed" or resume,
                "resume required after failed workload",
            )
            v, s = unit.split("-")
            return invoke(s, dict(kind="run", variant=v), unit)

        if state["status"] == "running":
            req = state["request"]
            rc = invoke(state["stage"], req, state["task"])
            if rc not in (0, 4):
                return rc
        for unit in units(plan):
            if unit != "source-analysis":
                rc = scientific(unit)
            else:
                req = dict(kind="analysis-report", variant=plan["variants"][0])
                rc = invoke("analysis", req, unit)
                if rc == 4 and plan["with_deps"]:
                    for missing in state["result"]["missing_units"]:
                        rc = scientific(missing)
                        if rc:
                            return rc
                    rc = invoke("analysis", req, unit)
            if rc:
                return rc
        return 0
