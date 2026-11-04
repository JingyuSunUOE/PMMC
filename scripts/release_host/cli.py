import argparse
import json
import os
import sys
import uuid
from decimal import Decimal, InvalidOperation

from .backends import Compose, Kueue
from .config import (
    PROFILES,
    ROOT,
    STAGES,
    SUITE,
    VARIANTS,
    Error,
    approval_id,
    check,
    identifier,
    load_config,
    native_path,
)
from .runner import MILESTONES, bundle, collect_review, groups, launch, read, state_root
from .workloads import command, secret_refs


def parser():
    cli = argparse.ArgumentParser(prog="./scripts/pmmc", description="PMMC release host controller")
    sub = cli.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "run", "resume", "quote", "approve", "status", "diagnostics"):
        p = sub.add_parser(verb)
        p.add_argument("--run", "--run-id", dest="run", required=verb not in ("plan", "run"))
        if verb in ("status", "diagnostics"):
            if verb == "diagnostics":
                p.add_argument(
                    "--refresh", action="store_true", help="collect current backend/PVC evidence"
                )
            continue
        p.add_argument("--config", help="deployment TOML (or PMMC_DEPLOY_CONFIG)")
        p.add_argument("--stage", choices=STAGES, default=None if verb == "resume" else "main")
        p.add_argument("--variant", choices=VARIANTS, required=verb in ("quote", "approve"))
        if verb in ("quote", "approve"):
            p.add_argument("--rate-card", help="container path to reviewed rate card")
            if verb == "approve":
                p.add_argument("--selection-hash", required=True)
                p.add_argument("--approver-id", required=True)
                p.add_argument("--limit-usd", required=True)
            continue
        p.add_argument("--budget-approval", metavar="ID", help="existing native budget approval ID")
        if verb != "resume":
            p.add_argument("--profile", required=True, choices=PROFILES)
            p.add_argument("--suite", choices=[SUITE], default=SUITE)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    try:
        root = state_root()
        run = identifier(args.run) if args.run else "exp-" + uuid.uuid4().hex[:16]
        folder = root / "runs" / run
        if args.verb == "status":
            print(json.dumps(read(folder / "progress.json"), indent=2))
            return 0
        if args.verb == "diagnostics":
            state = read(folder / "progress.json")
            if args.refresh:
                cfg = read(folder / "deployment.json")
                cls = Compose if cfg["profile"] == "local-nvidia" else Kueue
                private = root / "instances" / cfg["instance"]
                owner = read(private / "identity.json")
                backend = cls(
                    cfg, state["deployment"], private, owner["secrets"], None, state["request"]
                )
                collect_review(folder, state, backend)
            print(f"SYNC_REVIEW_FILE={bundle(folder)}")
            return 0
        file = args.config or os.getenv("PMMC_DEPLOY_CONFIG")
        if args.verb in ("resume", "quote", "approve"):
            saved = read(folder / "deployment.json")
            cfg = load_config(file, saved["profile"]) if file else saved
            check(cfg == saved, "resume requires the frozen deployment configuration")
        else:
            cfg = load_config(
                file or ROOT / "deploy/release" / f"{args.profile}.toml", args.profile
            )
        if args.verb == "plan":
            if args.budget_approval:
                approval_id(args.budget_approval)
            stage = (
                "analysis"
                if args.variant not in (None, "wr") and args.stage == "main"
                else args.stage
            )
            units = {}
            for unit in MILESTONES[stage]:
                v, s = unit.split("-")
                if args.variant and args.variant != v:
                    continue
                rendered = []
                for group in groups("run"):
                    jobs = []
                    for role in group:
                        argv = command(role, cfg, run, s, {"variant": v})
                        if argv[:2] == ["python", "-c"]:
                            argv[2] = "<embedded host helper>"
                        jobs.append(
                            {
                                "role": role,
                                "argv": argv,
                                "env_secret_keys": sorted(
                                    secret_refs(role, cfg, args.budget_approval)
                                ),
                            }
                        )
                    rendered.append(jobs)
                units[unit] = rendered
            check(units, "invalid stage/variant")
            check(
                not args.budget_approval or len(units) == 1,
                "select one variant for external approval",
            )
            print(json.dumps({"configuration": cfg, "run": run, "units": units}, indent=2))
            return 0
        action = None
        if args.verb in ("quote", "approve"):
            card = args.rate_card or cfg["native"].get("rate_card")
            check(card, "provide --rate-card or native.rate_card")
            action = {"kind": args.verb, "variant": args.variant, "rate_card": native_path(card)}
            if args.verb == "approve":
                try:
                    amount = Decimal(args.limit_usd)
                except InvalidOperation as exc:
                    raise Error("invalid USD limit") from exc
                check(amount.is_finite() and amount > 0, "expected a positive finite USD limit")
                action.update(
                    selection_hash=approval_id(args.selection_hash),
                    approver_id=identifier(args.approver_id),
                    limit_usd=str(amount),
                )
        return launch(
            root,
            cfg,
            run,
            args.stage,
            getattr(args, "budget_approval", None),
            resume=args.verb == "resume",
            variant=args.variant,
            action=action,
        )
    except (Error, OSError) as exc:
        # OSError may contain private paths. External command output is never echoed.
        print(
            f"pmmc: {exc if isinstance(exc, Error) else 'host filesystem error'}", file=sys.stderr
        )
        return 1
    except KeyboardInterrupt:
        print("pmmc: interrupted; detached work may still be running. Use resume.", file=sys.stderr)
        return 130
