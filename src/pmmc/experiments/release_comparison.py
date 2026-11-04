"""Paired descriptive readout of the four separately executed exploratory variants."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import write_json

ARMS = ("wr", "w", "r", "legacy")


def compare_primary(reports: dict[str, dict]) -> dict:
    if set(reports) != set(ARMS):
        raise ValueError("all four primary arms are required")
    rows = {}
    for arm, report in reports.items():
        if (report["exploration_variant"] != arm or report["scope"] != "main"
                or report["phase"] != "exploration" or report["paper_result"]):
            raise ValueError("invalid exploration arm")
        by_key = {}
        for group in report["groups"]:
            primary, = group["cells"]
            evidence = primary["evidence"] or []
            if (len(evidence) != group["selected_queries"]
                    or primary["evaluation_job_states"] != {"completed": len(evidence)}):
                raise ValueError("paired primary cohort not complete")
            for row in evidence:
                key = (group["dataset"], group["answer_model"], row["query_id"])
                if key in by_key:
                    raise ValueError("duplicate paired observation")
                by_key[key] = row
        rows[arm] = by_key
    if not rows["wr"] or any(rows[a].keys() != rows["wr"].keys() for a in ARMS):
        raise ValueError("exploration arm denominator changed")
    groups = []
    for dataset, model in sorted({k[:2] for k in rows["wr"]}):
        keys = [k for k in rows["wr"] if k[:2] == (dataset, model)]
        effects = {}
        for a, b in (("w", "legacy"), ("r", "legacy"), ("wr", "w"), ("wr", "r")):
            values: dict[str, object] = {}
            for metric in ("typed_score", "judge_score"):
                diff = [rows[a][k][metric] - rows[b][k][metric] for k in keys
                        if rows[a][k][metric] is not None and rows[b][k][metric] is not None]
                values[metric] = dict(observed_pairs=len(diff), selected=len(keys),
                                      observed_mean_difference=fmean(diff) if diff else None)
            values["supported_pmmc_rate_difference"] = sum(
                int(rows[a][k]["outcome"] == "supported_pmmc")
                - int(rows[b][k]["outcome"] == "supported_pmmc") for k in keys
            ) / len(keys)
            effects[f"{a}_minus_{b}"] = values
        groups.append(dict(dataset=dataset, answer_model=model,
                           selected=len(keys), effects=effects))
    return dict(schema="pmmc-exploration-factorial-readout-v1", paper_result=False,
                estimand="paired_descriptive_not_confirmatory", groups=groups,
                run_id=reports["wr"]["exploration_run_id"],
                arm_evidence_hashes={a: canonical_content_hash({
                    "campaign_id": reports[a]["campaign_id"],
                    "protocol_hash": reports[a]["protocol_hash"],
                    "selection_hash": reports[a]["selection_hash"],
                    "evidence": [(k, rows[a][k]) for k in sorted(rows[a])],
                }) for a in ARMS})


def record_comparison(settings, run_id: str, *, require: bool = False) -> Path | None:
    from pmmc.experiments.release_inputs import exploration_campaign_directory

    root = settings.outputs_path / "step8/exploration" / run_id
    reports = {}
    for arm in ARMS:
        if not (root / arm / "pointer.json").is_file():
            if require:
                raise ValueError("comparison arm missing")
            return None
        directory = exploration_campaign_directory(settings, run_id, variant=arm)
        metrics = directory / "reports/exploration-main.json"
        receipt = directory / "reports/release-main-control.json"
        if not receipt.exists():
            receipt = directory / "reports/release-main-all.json"
        if not (receipt.exists() and metrics.exists()
                and json.loads(receipt.read_bytes())["state"] == "stage_complete"):
            if require:
                raise ValueError("comparison arm not complete")
            return None
        reports[arm] = json.loads(metrics.read_bytes())
        if reports[arm]["exploration_run_id"] != run_id:
            raise ValueError("comparison run changed")
    output = root / "reports/paired-primary.json"
    write_json(output, compare_primary(reports), immutable=True)
    return output
