"""Public, dependency-light Kubernetes controller."""

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pmmc.release_plan import (
    ANALYSES,
    BENCHMARKS,
    COMPONENTS,
    METHODS,
    MODELS,
    STAGES,
    VARIANTS,
    normalize_plan,
    plan_hash,
)

from .backends import Kueue, resolve_kubernetes_resources
from .config import (
    ROOT,
    SUITE,
    Error,
    approval_id,
    check,
    identifier,
    load_config,
    native_path,
    verify_frozen_images,
)
from .runner import (
    atomic,
    bundle,
    collect_review,
    launch,
    planned_roles,
    read,
    service_status,
    state_root,
)


def parser():
    cli = argparse.ArgumentParser(
        prog="./scripts/pmmc", description="PMMC Kubernetes exploration controller"
    )
    sub = cli.add_subparsers(dest="verb", required=True)
    from .private_deploy import add_private_commands

    add_private_commands(sub)
    p = sub.add_parser("services")
    p.add_argument("operation", choices=("up", "status"))
    p.add_argument("--config")
    p = sub.add_parser("secrets")
    p.add_argument("operation", choices=("import",))
    p.add_argument("--env-file", required=True)
    p.add_argument("--role", choices=("providers", "hub"), action="append")
    p.add_argument("--rotate", action="store_true")
    p.add_argument("--config")
    for verb in ("plan", "prepare", "run", "resume", "quote", "approve", "status", "diagnostics"):
        p = sub.add_parser(verb)
        p.add_argument(
            "--run", "--run-id", dest="run", required=verb not in ("plan", "run", "prepare")
        )
        if verb in ("status", "diagnostics"):
            if verb == "diagnostics":
                p.add_argument("--refresh", action="store_true")
            continue
        p.add_argument("--config", help="deployment TOML, defaults to PMMC_DEPLOY_CONFIG")
        if verb in ("quote", "approve"):
            p.add_argument("--variant", choices=VARIANTS, default="wr")
            p.add_argument("--stage", choices=("main", "analysis", "baselines"))
            p.add_argument("--rate-card")
            if verb == "approve":
                p.add_argument("--selection-hash", required=True)
                p.add_argument("--plan-hash", required=True, help="exact hash from plan or quote")
                p.add_argument("--approver-id", required=True)
                mode = p.add_mutually_exclusive_group(required=True)
                mode.add_argument("--limit-usd")
                mode.add_argument("--uncapped", action="store_true")
            continue
        if verb == "resume":
            continue
        p.add_argument("--profile", choices=("kubernetes",), default="kubernetes")
        p.add_argument("--suite", choices=(SUITE,), default=SUITE)
        p.add_argument("--stage", choices=STAGES, default="main")
        for flag, choices in (
            ("benchmark", BENCHMARKS),
            ("model", MODELS),
            ("method", METHODS),
            ("variant", VARIANTS),
            ("analysis", ANALYSES),
            ("component", COMPONENTS),
        ):
            p.add_argument("--" + flag, action="append", choices=choices)
        p.add_argument("--from-run")
        p.add_argument("--with-deps", action="store_true")
        p.add_argument("--gpus", help="GPU ceiling or auto")
        p.add_argument("--cpus", help="total CPU ceiling or auto")
        p.add_argument("--memory", help="total memory ceiling (e.g. 160Gi) or auto")
    return cli


def public_plan(args, run):
    return normalize_plan(
        dict(
            run_id=run,
            stage=args.stage,
            benchmarks=args.benchmark,
            models=args.model,
            methods=args.method,
            variants=args.variant,
            analyses=args.analysis,
            components=args.component,
            from_run=args.from_run,
            with_deps=args.with_deps,
        )
    )


def describe(cfg, plan):
    from .runner import units

    paid = [x for x in plan["models"] if x in MODELS[2:]]
    only_read = plan["stage"] == "analysis" and not plan["with_deps"]
    return dict(
        run_id=plan["run_id"],
        plan_hash=plan_hash(plan),
        selection=plan,
        stages=units(plan),
        resources=cfg["resources"],
        paper_result=False,
        gpu_required=not only_read,
        models_with_possible_api_cost=[] if only_read else paid,
        paid_state="quote_and_explicit_approval_required"
        if paid and not only_read
        else "no_paid_targets",
        reuse="verified producer receipts only; query-level inventory available after prepare",
        denominator="computed from the selected snapshots; not a count of API calls",
    )


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    started = time.time_ns()
    folder = None
    try:
        if args.verb == "private":
            from .private_deploy import run_private_command

            print(json.dumps(run_private_command(args), sort_keys=True))
            return 0
        root = state_root()
        run = identifier(getattr(args, "run", None) or "exp-" + uuid.uuid4().hex[:16])
        folder = root / "runs" / run
        if args.verb in ("status", "diagnostics"):
            state = read(folder / "progress.json")
            if args.verb == "status":
                print(json.dumps(state, indent=2))
            else:
                if args.refresh:
                    cfg = read(folder / "deployment.json")
                    verify_frozen_images(cfg)
                    private = root / "instances" / cfg["instance"]
                    owner = read(private / "identity.json")
                    backend = Kueue(
                        cfg, owner["deployment"], private, owner["secrets"], None, state["request"]
                    )
                    collect_review(folder, state, backend)
                print(f"SYNC_REVIEW_FILE={bundle(folder)}")
            return 0
        file = args.config or os.getenv("PMMC_DEPLOY_CONFIG")
        if args.verb in ("resume", "quote", "approve"):
            cfg = read(folder / "deployment.json")
            verify_frozen_images(cfg)
            plan = normalize_plan(read(folder / "plan.json"))
            if args.config:
                check(
                    load_config(file, "kubernetes") == read(folder / "requested-deployment.json"),
                    "resume uses a frozen deployment",
                )
        else:
            overrides = {
                k: getattr(args, k)
                for k in ("gpus", "cpus", "memory")
                if getattr(args, k, None) is not None
            }
            cfg = load_config(
                file or ROOT / "deploy/release/kubernetes.toml",
                "kubernetes",
                resource_overrides=overrides,
            )
            if args.verb == "secrets":
                from .secrets import import_secrets

                print(
                    json.dumps(
                        import_secrets(cfg, args.env_file, roles=args.role, rotate=args.rotate),
                        sort_keys=True,
                    )
                )
                return 0
            if args.verb == "services" and args.operation == "status":
                print(json.dumps(service_status(root, cfg), sort_keys=True))
                return 0
            plan = (
                normalize_plan({"run_id": run})
                if args.verb == "services"
                else public_plan(args, run)
            )
        if args.verb == "plan":
            details = describe(resolve_kubernetes_resources(cfg, roles=planned_roles(plan)), plan)
            if (folder / "plan.json").exists() and read(folder / "plan.json") == plan:
                details["prepared_selections"] = read(folder / "progress.json").get("prepared", {})
            print(json.dumps(details, sort_keys=True, indent=2))
            return 0
        action = None
        if args.verb in ("quote", "approve"):
            card = args.rate_card or cfg["native"].get("rate_card")
            check(card, "supply a reviewed private --rate-card")
            stage = args.stage or ("main" if plan["stage"] == "all" else plan["stage"])
            check(args.variant in plan["variants"], "variant not declared by frozen plan")
            action = dict(
                kind=args.verb, variant=args.variant, stage=stage, rate_card=native_path(card)
            )
            if args.verb == "approve":
                check(
                    args.plan_hash == plan_hash(plan),
                    "approval must match the frozen experiment plan",
                )
                amount = None
                if not args.uncapped:
                    try:
                        amount = Decimal(args.limit_usd)
                    except InvalidOperation as exc:
                        raise Error("invalid budget") from exc
                    check(amount.is_finite() and amount > 0, "require positive finite USD limit")
                action.update(
                    selection_hash=approval_id(args.selection_hash),
                    approver_id=identifier(args.approver_id),
                    uncapped=args.uncapped,
                    limit_usd=str(amount) if amount is not None else None,
                )
        elif args.verb in ("services", "prepare"):
            kind = (
                "analysis-report"
                if args.verb == "prepare" and plan["stage"] == "analysis" and not plan["with_deps"]
                else args.verb
            )
            action = dict(
                kind=kind,
                variant=plan["variants"][0],
                stage="main" if plan["stage"] == "all" else plan["stage"],
            )
        code = launch(root, cfg, run, plan, action=action, resume=args.verb == "resume")
        if (
            code == 0
            and (folder / "progress.json").is_file()
            and (
                not (folder / "diagnostics.tar.gz").is_file()
                or (folder / "diagnostics.tar.gz").stat().st_mtime_ns < started
            )
        ):
            print(f"SYNC_REVIEW_FILE={bundle(folder)}", flush=True)
        return code
    except (Error, OSError, ValueError) as exc:
        # Do not print subprocess stderr, environment values, URLs or secret-file contents.
        message = (
            str(exc) if isinstance(exc, Error) or type(exc) is ValueError else type(exc).__name__
        )
        print(f"pmmc: {message}", file=sys.stderr)
        _error_bundle(folder, started, args.verb, exc)
        return 1
    except KeyboardInterrupt:
        print("pmmc: interrupted; workload may be detached. Resume the same run.", file=sys.stderr)
        _error_bundle(folder, started, args.verb, KeyboardInterrupt())
        return 130


def _error_bundle(folder, started, verb, exc):
    if verb not in {"services", "prepare", "run", "resume", "approve", "quote"}:
        return
    try:
        if (
            folder
            and (folder / "diagnostics.tar.gz").is_file()
            and (folder / "diagnostics.tar.gz").stat().st_mtime_ns >= started
        ):
            return  # The workload already printed its richer evidence package.
        base = state_root() / "host-errors"
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(dir=base, prefix="launch-"))
        atomic(
            out / "progress.json",
            dict(
                operation=verb, status="not_started_or_interrupted", error_type=type(exc).__name__
            ),
        )
        print(f"SYNC_REVIEW_FILE={bundle(out)}", flush=True)
    except (Error, OSError):
        print("pmmc: diagnostic storage is not writable", file=sys.stderr)
