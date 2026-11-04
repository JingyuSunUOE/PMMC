from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.release_reporting import (
    approve_release_budget,
    estimate_release_budget,
    summarize_release_campaign,
)
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, sha256_file, write_json
from pmmc.experiments.step8_campaign import load_benchmark_config
from pmmc.experiments.step8_jobs import SOURCE_OWNERS, BenchmarkStageJob, ComponentReceipt
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import (
    ANSWER_BACKBONES,
    BASELINE_METHODS,
    STEP8_ALEMBIC_REVISION,
    BenchmarkCampaignManifest,
    ResourceClass,
    ScientificProtocol,
    SelectionExpansionManifest,
    build_experiment_matrix,
)
from step8_model_fixtures import model_entry_hashes

ROOT = Path(__file__).resolve().parents[1]
GPT, CLAUDE = ANSWER_BACKBONES[-2:]


class Campaign:
    def __init__(self, root):
        self.dir = root / "exploration"
        self.store = CampaignArtifactStore(root)
        config = load_benchmark_config(
            ROOT / "configs/experiments/step8-benchmark.yaml"
        ).model_copy(update={"model_registry": Path("configs/release/models.yaml")})
        self.protocol = ScientificProtocol(
            model_lock_hash="a" * 64,
            model_entry_hashes=model_entry_hashes(),
            partition_hash="b" * 64,
            exposure_manifest_hash="c" * 64,
            source_fingerprints=dict.fromkeys(SOURCE_OWNERS, "d" * 64),
            config_hashes={"model_registry": sha256_file(ROOT / config.model_registry)},
        )
        template = ROOT / "configs/models/step8-provider-rate-card.template.yaml"
        card = yaml.safe_load(template.read_bytes())
        card["estimated"] = False
        for limits in card["rate_limits"].values():
            limits.update(
                account_scope="offline-test", tokens_per_minute=32768, requests_per_minute=1
            )
        self.card = root / "user-provided-rates" / "test-card.yaml"
        self.review = self.card.with_name("provider-policy.json")
        write_json(self.card, card)
        write_json(
            self.review,
            {
                "protocol": "pmmc-release-rate-review-v1",
                "reviewed_date": "2026-10-01",
                "reviewed_by": "test reviewer",
                "currency": "USD",
                "pricing_basis": "standard_global_first_party_synchronous_before_tax",
                "rate_card_file": self.card.name,
                "rate_card_sha256": sha256_file(self.card),
                "model_registry_sha256": self.protocol.config_hashes["model_registry"],
                "pricing_sources": {k: f"https://example.invalid/{k}" for k in card["models"]},
            },
        )
        self.cells = build_experiment_matrix(self.protocol)
        datasets = (DatasetName.MEMLENS, DatasetName.MEMGALLERY)
        ids = {k: [f"q{i}" for i in range(6)] for k in datasets}
        self.manifest = BenchmarkCampaignManifest(
            campaign_id="fresh",
            phase="exploration",
            protocol=self.protocol,
            partition_id="split",
            snapshot_ids={k: k.value for k in datasets},
            created_at=datetime.now(UTC),
            matrix_hash=canonical_content_hash([c.model_dump(mode="json") for c in self.cells]),
        )
        self.selection = SelectionExpansionManifest(
            campaign_id="fresh",
            phase="exploration",
            selection="dev-all",
            query_ids=ids,
            added_query_ids=ids,
            eligible_counts={k: len(v) for k, v in ids.items()},
            stratum_counts={k: {"all": len(v)} for k, v in ids.items()},
            partition_hash=self.protocol.partition_hash,
        )
        for name, value in (
            ("campaign", self.manifest),
            ("config", config),
            (f"selections/{self.selection.content_hash}", self.selection),
        ):
            write_json(self.dir / f"manifests/{name}.json", value.model_dump(mode="json"))
        self.rows = {}

    def add(
        self,
        op,
        stage="evaluation",
        *,
        q="q0",
        model=GPT,
        state="pending",
        parents=(),
        primary=False,
    ):
        params = {"dataset": "memlens", "query_id": q, "treatment": "main", "answer_model": model}
        if primary:
            params["readout"] = "primary"
        job = BenchmarkStageJob(
            campaign_id="fresh",
            stage=stage,
            component_key=f"c{len(self.rows)}",
            resource_class=ResourceClass.API if stage == "answers" else ResourceClass.GPU_9B,
            source_fingerprint=self.protocol.source_fingerprints[stage],
            scientific_inputs=params,
            operation=op,
            parameters=params,
            dependency_ids=[p.job_id for p in parents],
        )
        key = job.job_id
        payload = job.model_dump(mode="json")
        self.rows[key] = {
            "job": payload,
            "job_id": key,
            "status": state,
            "receipt_hash": None,
            "attempt_count": int(state == "failed"),
            "error_code": None,
        }
        write_json(self.dir / f"private/jobs/{key}.json", payload)
        write_json(self.dir / f"manifests/jobs-{self.selection.content_hash}.json", list(self.rows))
        return job

    def complete(self, job, output):
        receipt = ComponentReceipt(
            job_id=job.job_id,
            scientific_input_hash=job.scientific_input_hash,
            source_fingerprint=job.source_fingerprint,
            schema_revision=STEP8_ALEMBIC_REVISION,
            upstream_receipt_hashes={k: self.rows[k]["receipt_hash"] for k in job.dependency_ids},
            output=output,
            artifact_hashes=[self.store.put_json(output)],
            row_counts={"component_outputs": 1},
        )
        write_json(
            self.dir / f"checkpoints/components/{job.job_id}.json", receipt.model_dump(mode="json")
        )
        self.rows[job.job_id].update(status="completed", receipt_hash=receipt.content_hash)
        return receipt

    def score(self, job, source="pmmc", terminal="answered"):
        p = job.parameters
        main = [
            c
            for c in self.cells
            if c.dataset == p["dataset"]
            and c.answer_model == p["answer_model"]
            and c.treatment == "main"
        ]
        scope = {
            "primary_evaluate": "pmmc_main",
            "evaluate": "paired_main",
            "analysis_evaluate": "pmmc",
        }[job.operation]
        cells = [
            c
            for c in main
            if (c.method == "pmmc" or scope == "paired_main")
            and (
                scope == "pmmc"
                or (c.question_budget, c.round_id, c.source_variant) == (8, 2, "active")
            )
        ]
        scores = [
            {
                "cell_id": c.cell_id,
                "method": c.method,
                "source_variant": c.source_variant,
                "answer_source_method": source if c.method == "pmmc" else c.method,
                "source_fact_hash": f"fact-{p['query_id']}",
                "answer_source_variant": "primary",
                "terminal_state": terminal,
                "typed_answer_id": "answer" if terminal != "no_answer" else None,
                "value": 0,
                "judge_value": None,
            }
            for c in cells
        ]
        return {
            "campaign_id": "fresh",
            "query_id": p["query_id"],
            "query_executor_run_id": "run",
            "status": "complete",
            "method_scope": scope,
            "selected_query": {"dataset": p["dataset"], "query_id": p["query_id"]},
            "selected_cell_count": len(scores),
            "scores": scores,
        }

    def summary(self, scope):
        return summarize_release_campaign(self.dir, jobs=list(self.rows.values()), scope=scope)

    def estimate(self, scope):
        return estimate_release_budget(
            self.dir,
            jobs=list(self.rows.values()),
            scope=scope,
            rate_card=self.card,
            project_root=ROOT,
        )


@pytest.fixture
def campaign(tmp_path):
    return Campaign(tmp_path)


def _group(report, model=GPT):
    return next(
        g for g in report["groups"] if (g["dataset"], g["answer_model"]) == ("memlens", model)
    )


def test_partial_main_keeps_all_selected_and_distinguishes_answer_sources(campaign):
    c = campaign
    embedding = c.add("query_embedding", "multimodal_index")
    c.complete(
        embedding,
        {"joint_pair_count": 1, "joint_model_call_count": 0, "joint_reused_batch_count": 1},
    )
    for i, (source, terminal) in enumerate(
        [
            ("pmmc", "answered"),
            ("native_multimodal_rag", "answered"),
            ("pmmc", "abstained"),
            (None, "no_answer"),
        ]
    ):
        job = c.add("primary_evaluate", primary=True, q=f"q{i}", parents=(embedding,))
        output = c.score(job, source, terminal)
        if i == 0:
            score = output["scores"][0]
            output["process_metrics"] = {
                "cells": [
                    {
                        "cell_id": score["cell_id"],
                        "method": "pmmc",
                        "source_fact_hash": score["source_fact_hash"],
                        "query_id": "q0",
                        "metrics": {
                            "fallback_triggered": {
                                "status": "evaluated",
                                "value": 0,
                                "numerator": 0,
                                "denominator": 1,
                            }
                        },
                        "provenance": {"round_members_hash": "members", "eligible_bank_count": 0},
                    }
                ]
            }
            output["relevance_grid"] = {
                "status": "completed",
                "point_count": 1,
                "points": [
                    {
                        "status": "completed",
                        "joint_policy": "observed",
                        "typed_answer_id": "answer",
                        "arrived_joint_image_hashes": ["arrived"],
                    }
                ],
            }
        c.complete(job, output)
    c.add(
        "primary_evaluate",
        primary=True,
        q="q4",
        parents=(c.add("query_answer", "answers", q="q4", state="unknown"),),
    )
    c.add("primary_evaluate", primary=True, q="q5", state="failed")
    c.add("analysis_evaluate")  # Other scopes are not substitutes for a missing main receipt.
    write_json(c.dir / "reports/legacy.json", {"pmmc_answer_count": 1000, "cost_usd": 0})
    result = c.summary("main")
    group = _group(result)
    assert result["paper_result"] is False and result["actual_cost_usd"] is None
    assert result["timing_seconds"] is None
    assert group["selected_queries"] == 6
    assert group["query_evaluation_states"] == {"completed": 4, "pending": 1, "failed": 1}
    assert group["queries_with_pending_evaluation_blocked_by_failure_or_unknown"] == 1
    (cell,) = group["cells"]
    assert cell["observed_outcomes"] == {
        "supported_pmmc": 1,
        "fallback_answer": 1,
        "refusal": 1,
        "no_answer": 1,
    }
    assert cell["typed_exact_match"] == {"mean_observed": 0, "observed": 4, "selected": 6}
    assert cell["judge"]["mean_observed"] is None
    assert cell["process"]["fallback_triggered"]["mean_observed"] == 0
    assert cell["process"]["qbank_full_bank_proxy_cover"]["mean_observed"] is None
    assert cell["evidence"][0]["eligible_bank_count"] == 0
    assert result["joint_embedding_evidence"][0]["joint_model_call_count"] == 0
    assert any(
        e["joint_points"] and e["joint_points"][0]["arrived_joint_image_hashes"] == ["arrived"]
        for e in result["r_evaluation_evidence"]
    )
    assert _group(result, CLAUDE)["cells"][0]["observed_outcomes"] is None


def test_analysis_and_baselines_only_count_their_own_receipts(campaign):
    c = campaign
    for op in ("analysis_evaluate", "evaluate"):
        job = c.add(op)
        c.complete(job, c.score(job))
    analysis = _group(c.summary("analysis"))["cells"]
    assert {x["method"] for x in analysis} == {"pmmc"}
    assert {x["variant"] for x in analysis} == {"active", "no_fallback", "forced_top1"}
    assert {tuple(x["observed_outcomes"]) for x in analysis} == {
        ("supported_pmmc",),
        ("counterfactual_pmmc",),
    }
    paired = _group(c.summary("baselines"))["cells"]
    assert {x["method"] for x in paired} == {"pmmc", *BASELINE_METHODS}
    assert all(
        x["selected"] == 6 and x["evaluation_job_states"] == {"completed": 1, "not_planned": 5}
        for x in paired
    )


@pytest.mark.parametrize("damage", ["db_hash", "cas", "job_id", "scope", "historical", "upstream"])
def test_unverified_or_historical_evidence_fails_closed(campaign, damage):
    c = campaign
    parent = c.add("query_answer", "answers")
    c.complete(parent, {"phase": "answer"})
    job = c.add("primary_evaluate", primary=True, parents=(parent,))
    output = c.score(job)
    if damage == "scope":
        output["query_id"] = "elsewhere"
    receipt = c.complete(job, output)
    if damage == "db_hash":
        c.rows[job.job_id]["receipt_hash"] = "a" * 64
    if damage == "cas":
        c.store.object_path(receipt.artifact_hashes[0]).write_bytes(b"corrupt")
    if damage == "job_id":
        c.rows[job.job_id]["job_id"] = parent.job_id
    if damage == "historical":
        write_json(
            c.dir / "manifests/campaign.json",
            {**c.manifest.model_dump(mode="json"), "phase": "development"},
        )
    if damage == "upstream":
        c.rows[parent.job_id]["status"] = "failed"
    with pytest.raises(ValueError):
        c.summary("main")


def test_pending_estimates_are_disjoint_and_not_an_approval(campaign):
    c = campaign
    main = c.add("query_answer", "answers", primary=True, model=CLAUDE)
    fallback = c.add(
        "native_fallback_generate", "answers", primary=True, parents=(main,), model=CLAUDE
    )
    c.add("primary_evaluate", primary=True, parents=(fallback,), model=CLAUDE)
    analysis = c.add("query_answer", "answers")
    reused = c.add("native_fallback_generate", "answers", parents=(analysis,), state="completed")
    c.add("analysis_evaluate", parents=(reused,))
    baseline = c.add("baseline_answer", "answers")
    unknown = c.add("baseline_answer", "answers", q="q1", state="unknown")
    c.add("evaluate", parents=(reused, baseline, unknown))
    estimates = {scope: c.estimate(scope) for scope in ("main", "analysis", "baselines")}
    main_rows = estimates["main"]["pending_api_groups"]
    assert {r["operation"]: r["generation_calls_at_most"] for r in main_rows} == {
        "query_answer": 2,
        "native_fallback_generate": 2,
    }
    assert {r["provider_output_token_cap"] for r in main_rows} == {512}
    assert all(
        r["provider_output_token_cap"] == 4096 for r in estimates["analysis"]["pending_api_groups"]
    )
    paired = estimates["baselines"]
    assert paired["api_job_states"] == {"pending": 2, "completed": 1, "unknown": 1}
    (incremental,) = [r for r in paired["pending_api_groups"] if not r["in_preceding_scope"]]
    assert (
        incremental["operation"] == "baseline_answer"
        and incremental["generation_calls_at_most"] == 10
    )
    assert (
        incremental["one_pass_reservation_bound_usd"]
        == paired["pending_additional_after_preceding_scopes"]["one_pass_reservation_bound_usd"]
    )
    assert Decimal(incremental["standard_rate_scenario_usd"]) < Decimal(
        incremental["one_pass_reservation_bound_usd"]
    )
    assert paired["approval_created"] is False and paired["legacy_budget_allocated"] is False
    assert not list((c.dir / "manifests").glob("budget-*.json"))
    del c.rows[baseline.job_id]
    missing = c.estimate("baselines")
    assert missing["api_job_states"]["unregistered"] == 1
    assert missing["pending_scope"]["one_pass_reservation_bound_usd"] is None


def test_guard_requires_explicit_matching_selection_and_review_before_register(
    campaign, tmp_path, monkeypatch
):
    c = campaign
    register = AsyncMock(return_value=("approved-id", {}))
    monkeypatch.setattr("pmmc.experiments.step8_cli._register", register)
    kwargs = {
        "selection_hash": c.selection.content_hash,
        "rate_card": c.card,
        "project_root": ROOT,
        "approver_id": "reviewer",
        "limit_usd": Decimal("3.00"),
    }
    tampered = c.card.with_name("renamed-card.yaml")
    tampered.write_bytes(c.card.read_bytes() + b"\n")
    for updates in (
        {"selection_hash": "old"},
        {"approver_id": " "},
        {"limit_usd": Decimal("NaN")},
        {"rate_card": tampered},
    ):
        with pytest.raises(ValueError):
            asyncio.run(approve_release_budget(c.dir, **{**kwargs, **updates}))
    register.assert_not_awaited()
    with pytest.raises(ValueError, match="saved quote"):
        asyncio.run(approve_release_budget(c.dir, **kwargs))
    c.add("primary_evaluate", parents=(c.add("query_answer", "answers"),))
    quote_path = c.dir / "reports/budget-quote-main.json"
    quote = c.estimate("main")
    for stale in (
        {"selection_hash": "other"},
        {"rate_card_sha256": "different"},
        {"review_sha256": "outdated"},
        {"scope": "analysis"},
    ):
        write_json(quote_path, {**quote, **stale})
        with pytest.raises(ValueError, match="saved quote"):
            asyncio.run(approve_release_budget(c.dir, **kwargs))
    write_json(quote_path, quote)
    result = asyncio.run(approve_release_budget(c.dir, **kwargs))
    register.assert_awaited_once_with(
        c.dir, budget=Decimal("3.00"), rate_card=c.card, approver="reviewer", retry=False
    )
    assert (
        result["approval_id"] == "approved-id"
        and result["selection_hash"] == c.selection.content_hash
    )
    write_json(
        c.dir / f"manifests/budget-{c.selection.content_hash}.json",
        {
            "selection_hash": c.selection.content_hash,
            "rate_card_hash": sha256_file(c.card),
            "approver_id": "reviewer",
            "limit_usd": "2.00",
        },
    )
    with pytest.raises(ValueError, match="conflicting"):
        asyncio.run(approve_release_budget(c.dir, **kwargs))
    assert register.await_count == 1


@pytest.mark.parametrize(
    "change",
    [
        {"budget": {"limit_usd": "20"}},
        {"model_registry_sha256": "f" * 64},
        {"pricing_sources": {"gpt-6-luna": "https://example.invalid"}},
        {"reviewed_by": " "},
        {"rate_card_sha256": "e" * 64},
        {"protocol": "step8-provider-policy-review-v2"},
    ],
)
def test_public_pricing_review_fails_closed(campaign, change):
    c = campaign
    c.add("primary_evaluate", parents=(c.add("query_answer", "answers"),))
    write_json(c.review, {**json.loads(c.review.read_bytes()), **change})
    with pytest.raises(ValueError):
        c.estimate("main")
