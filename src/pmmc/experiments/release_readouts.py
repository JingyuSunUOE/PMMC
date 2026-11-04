"""Content-addressed exploratory readouts; the analysis worker has no provider credentials."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pmmc.release_plan import normalize_plan, plan_hash

from .release_comparison import ARMS, compare_primary
from .step8_artifacts import json_bytes, write_json


def seal_report(directory, stage, metrics, report):
    if report["state"] != "stage_complete" or metrics["paper_result"] is not False:
        raise ValueError("only completed exploratory results may be sealed")
    for key in ("campaign_id", "selection_hash", "stage"):
        if report[key] != metrics["scope" if key == "stage" else key]:
            raise ValueError("exploratory result identity mismatch")
    root = Path(directory) / "reports"
    dst = root / f"seal-{stage}.json"
    evidence = {k: v for k, v in metrics.items() if k != "selection_ledger"}
    digest = hashlib.sha256(json_bytes(evidence)).hexdigest()
    write_json(root / "sealed" / f"{digest}.json", evidence, immutable=True)
    seal = dict(
        protocol="pmmc-scoped-readout-v2",
        sha256=digest,
        campaign_id=report["campaign_id"],
        selection_hash=report["selection_hash"],
        scope_hash=report["scope_hash"],
        stage=stage,
    )
    write_json(dst, seal, immutable=True)
    observation = {
        "scope_hash": seal["scope_hash"],
        "worker_pool": report.get("worker_pool"),
        "invocation_wall_seconds": report.get("invocation_wall_seconds"),
        "physical_startups": report.get("physical_startups"),
        "scope_completed": report.get("completed"),
        "scope_total": report.get("scoped_jobs"),
    }
    obs_sha = hashlib.sha256(json_bytes(observation)).hexdigest()
    write_json(root / "observations" / stage / (obs_sha + ".json"), observation, immutable=True)
    # A selection ledger can change after a later stage; never sum these cumulative observations.
    if "selection_ledger" in metrics:
        write_json(
            root / f"ledger-{stage}.json",
            dict(
                campaign_id=seal["campaign_id"],
                selection_hash=seal["selection_hash"],
                observation=metrics["selection_ledger"],
            ),
        )


def _read(settings, run, arm, stage):
    from pmmc.compiler.ids import canonical_content_hash
    from pmmc.schemas.step8 import BenchmarkCampaignManifest, SelectionExpansionManifest

    top = (settings.outputs_path / "step8").resolve()
    pointer = top / "exploration" / run / arm / "pointer.json"
    if not pointer.is_file():
        return None

    def read(path):
        if path.is_symlink() or not path.resolve().is_relative_to(top):
            raise ValueError("readout path escapes outputs")
        return path.read_bytes()

    p = json.loads(read(pointer))
    if (
        p["run_id"] != run
        or p["variant"] != arm
        or p["phase"] != "exploration"
        or p["paper_result"] is not False
        or p["state"] != "planned"
    ):
        raise ValueError("invalid source pointer")
    directory = (settings.project_root / p["campaign_dir"]).resolve()
    if directory.parent != top:
        raise ValueError("invalid campaign directory")
    manifests = directory / "manifests"
    raw_campaign = read(manifests / "campaign.json")
    if hashlib.sha256(raw_campaign).hexdigest() != p["campaign_sha256"]:
        raise ValueError("source campaign digest changed")
    campaign = BenchmarkCampaignManifest.model_validate_json(raw_campaign)
    h = p["selection_hash"]
    if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("invalid source selection")
    sel = SelectionExpansionManifest.model_validate_json(
        read(manifests / "selections" / (h + ".json"))
    )
    if (
        sel.content_hash != h
        or sel.campaign_id != campaign.campaign_id
        or sel.selection != "dev-all"
        or sel.phase.value != "exploration"
        or campaign.phase.value != "exploration"
        or campaign.paper_result
        or campaign.protocol.content_hash != p["protocol_hash"]
    ):
        raise ValueError("source selection or protocol mismatch")
    root = directory / "reports"
    seal_path = root / f"seal-{stage}.json"
    if not seal_path.exists():
        return None
    seal = json.loads(read(seal_path))
    from .release_scope import verified_scope_payload

    scope_hash = seal.get("scope_hash")
    if (
        seal.get("protocol") != "pmmc-scoped-readout-v2"
        or seal.get("stage") != stage
        or not isinstance(scope_hash, str)
        or len(scope_hash) != 64
        or any(c not in "0123456789abcdef" for c in scope_hash)
    ):
        raise ValueError("invalid source seal scope")
    scope_file = directory / "private/scopes" / f"{scope_hash}.json"
    if not scope_file.is_file() or json.loads(read(scope_file))["stage"] != stage:
        raise ValueError("source seal does not identify the requested stage")
    scope = verified_scope_payload(directory, scope_hash, campaign.campaign_id, h)
    if scope["stage"] != stage or scope["paid"] is not True:
        raise ValueError("source seal does not identify the requested stage")
    digest = seal["sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid readout digest")
    raw = read(root / "sealed" / f"{digest}.json")
    data = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != digest
        or data["exploration_run_id"] != run
        or data["exploration_variant"] != arm
        or data["scope"] != stage
        or data["paper_result"]
        or data["campaign_id"] != seal["campaign_id"]
        or data["campaign_id"] != campaign.campaign_id
        or data["selection_hash"] != seal["selection_hash"]
        or data["selection_hash"] != h
        or data["protocol_hash"] != campaign.protocol.content_hash
    ):
        raise ValueError("source readout identity changed")
    ledger_file = root / f"ledger-{stage}.json"
    ledger = json.loads(read(ledger_file)) if ledger_file.exists() else None
    if ledger and any(ledger[k] != seal[k] for k in ("campaign_id", "selection_hash")):
        raise ValueError("ledger observation belongs to another selection")
    signature = campaign.protocol.model_dump(
        mode="json",
        exclude={
            "answer_backbones",
            "partition_hash",
            "exposure_manifest_hash",
            "config_hashes",
            "questioner",
            "executor",
        },
    )
    signature["config_hashes"] = {
        k: v
        for k, v in campaign.protocol.config_hashes.items()
        if k not in ("questioner_config", "query_executor_config")
    }
    identity = dict(
        run_id=run,
        variant=arm,
        stage=stage,
        report_sha256=digest,
        protocol_hash=data["protocol_hash"],
        selection_hash=h,
        peer_hash=canonical_content_hash(signature),
        cohorts={
            ds.value: canonical_content_hash([campaign.snapshot_ids[ds], sorted(ids)])
            for ds, ids in sel.query_ids.items()
        },
    )
    observations = []
    for path in sorted((root / "observations" / stage).glob("*.json")):
        raw_obs = read(path)
        obs = json.loads(raw_obs)
        if (
            hashlib.sha256(raw_obs).hexdigest() != path.stem
            or obs["scope_hash"] != seal["scope_hash"]
        ):
            raise ValueError("timing observation does not identify the sealed scope")
        observations.append(obs)
    return {**data, "invocations": observations}, identity, ledger


def run_analysis(settings, raw_plan):
    plan = normalize_plan(raw_plan)
    run, source = plan["run_id"], plan["from_run"]
    required = [(v, "main") for v in plan["variants"]]
    if "ablation" in plan["analyses"] and "wr" in plan["variants"]:
        required.append(("wr", "analysis"))
    found, missing = {}, []
    for arm, stage in required:
        for owner in dict.fromkeys([run, source or run]):
            item = _read(settings, owner, arm, stage)
            if item is not None:
                break
        else:
            missing.append(f"{arm}-{stage}")
            continue
        observed, identity, ledger = item
        selected = {(dataset, model) for dataset in plan["benchmarks"] for model in plan["models"]}
        groups = [g for g in observed["groups"] if (g["dataset"], g["answer_model"]) in selected]
        if {(g["dataset"], g["answer_model"]) for g in groups} != selected:
            raise ValueError("readout does not cover selected benchmark/model pairs")
        if any(
            not g["cells"]
            or not g["selected_queries"]
            or any(
                c["evaluation_job_states"] != {"completed": g["selected_queries"]}
                for c in g["cells"]
            )
            for g in groups
        ):
            missing.append(f"{arm}-{stage}")
            continue
        found[arm if stage == "main" else "wr-analysis"] = (
            {**observed, "groups": groups},
            identity,
            ledger,
        )
    summary = dict(
        protocol="pmmc-analysis-readout-v2",
        plan_hash=plan_hash(plan),
        state="missing_dependencies" if missing else "complete",
        missing_units=missing,
        paper_result=False,
        sources=[x[1] for x in found.values()],
    )
    if missing:
        return summary
    if len(found) > 1:
        identities = {
            (identity["peer_hash"], tuple(identity["cohorts"][d] for d in plan["benchmarks"]))
            for _, identity, _ in found.values()
        }
        if len(identities) != 1:
            raise ValueError("cannot pair different scientific inputs or selected cohorts")
    sections = {}
    for analysis in plan["analyses"]:
        if analysis == "ablation":
            sections[analysis] = (
                compare_primary({a: found[a][0] for a in ARMS})
                if set(ARMS) <= set(found)
                else {"state": "not_paired", "observed_arms": sorted(set(found) & set(ARMS))}
            )
            if "wr-analysis" in found:
                sections["treatment_grid"] = found["wr-analysis"][0]["groups"]
        elif analysis == "cost":
            sections[analysis] = [
                dict(
                    source=x[1],
                    selection_ledger=x[2],
                    wall_seconds=x[0].get("timing_seconds"),
                    measured_invocations=x[0].get("invocations", []),
                    note="Concurrent invocation times and cumulative ledgers are not additive. Null is unmeasured.",
                )
                for x in found.values()
            ]
        else:
            fields = (
                ("observed_outcomes", "evaluation_job_states", "process_unavailable_reasons")
                if analysis == "failures"
                else ("process", "typed_exact_match", "judge")
            )
            sections[analysis] = [
                dict(
                    source=x[1],
                    dataset=g["dataset"],
                    model=g["answer_model"],
                    method=c["method"],
                    selected=g["selected_queries"],
                    **{key: c.get(key) for key in fields},
                )
                for x in found.values()
                for g in x[0]["groups"]
                for c in g["cells"]
            ]
    root = settings.outputs_path / "step8/exploration" / run / "reports"
    payload = {**summary, "sections": sections}
    digest = hashlib.sha256(json_bytes(payload)).hexdigest()
    write_json(root / "analyses" / f"{digest}.json", payload, immutable=True)
    write_json(root / "analysis.json", dict(sha256=digest, plan_hash=summary["plan_hash"]))
    return {**summary, "sha256": digest}


def main():
    from pmmc.infrastructure.settings import get_settings

    plan = json.loads(os.environ["PMMC_RELEASE_REQUEST"])
    print(json.dumps(run_analysis(get_settings(), plan), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
