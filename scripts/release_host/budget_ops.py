"""Audit or fence a shared budget without waiting for the controller's run lock."""

import json
import tempfile
import time
import uuid
from pathlib import Path

from pmmc.release_plan import plan_hash

from .backends import Kubernetes, invoke, receipt
from .config import Error, check
from .runner import atomic, bundle, read
from .workloads import command


def audit(root, cfg, plan, request):
    run = plan["run_id"]
    folder = root / "runs" / run
    state = read(folder / "progress.json")
    private = root / "instances" / cfg["instance"]
    owner = read(private / "identity.json")
    request = {**request, "plan": plan, "plan_hash": plan_hash(plan)}
    role = request["kind"]
    backend = Kubernetes(cfg, owner["deployment"], private, owner["secrets"], None, request)
    out = folder / "budget-audits"
    out.mkdir(exist_ok=True, mode=0o700)
    log = Path(tempfile.mkdtemp(dir=out))
    try:
        live = (state.get("batch") or {}).get("control")
        if (
            state["status"] == "running"
            and live
            and live["state"] == "running"
            and backend.state(live["job"]) == "running"
        ):
            raw = invoke(
                [
                    *backend.kube,
                    "exec",
                    "job/" + live["job"],
                    "-c",
                    "control",
                    "--",
                    *command(role, cfg, run, "main", request),
                ]
            )
            result = receipt(role, raw.splitlines()[-1])
        else:
            job = cfg["instance"] + "-budget-" + uuid.uuid4().hex[:12]
            backend.start(job, role, run, "main")
            until = time.monotonic() + cfg["cluster"]["admission_timeout_seconds"]
            last = None
            while (status := backend.state(job)) != "complete":
                if status != last:
                    print(f"PMMC_BUDGET job={job} state={status}", flush=True)
                    last = status
                check(status not in ("missing", "failed"), "budget operation failed; verify ledger")
                check(time.monotonic() < until, "budget operation not confirmed; inspect its Job")
                time.sleep(3)
            result = backend.outcome(job, role)
        check(result["plan_hash"] == request["plan_hash"], "wrong plan in budget receipt")
        if role == "stop-budget":
            check(result.get("approval_id") == request["approval_id"], "wrong stopped approval")
        atomic(log / "progress.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, Error, KeyboardInterrupt, ValueError):
        atomic(
            log / "progress.json",
            dict(status="unconfirmed", operation=role, plan_hash=request["plan_hash"]),
        )
        raise
    finally:
        print(f"SYNC_REVIEW_FILE={bundle(log)}", flush=True)
