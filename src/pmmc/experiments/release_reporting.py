"""Read-only, selection-scoped v0.1 exploration readouts and API estimates.

Pass one authoritative repository.list_jobs(campaign_id) snapshot to either
report. Neither function registers jobs, retries failures, or dispatches calls.

Review a copy of configs/models/step8-provider-rate-card.template.yaml and
place provider-policy.json beside that card. Example (placeholders intentionally
cannot authorize a call; supply the actual card filename, hashes, reviewer,
review date and sources for every configured paid model):

    {
      "protocol": "pmmc-release-rate-review-v1",
      "currency": "USD",
      "pricing_basis": "standard_global_first_party_synchronous_before_tax",
      "reviewed_by": "<reviewer>",
      "reviewed_date": "YYYY-MM-DD",
      "rate_card_file": "rates.yaml",
      "rate_card_sha256": "<sha256 of the exact rates.yaml bytes>",
      "model_registry_sha256": "<sha256 of the campaign model registry file>",
      "pricing_sources": {
        "gpt-6-luna": "https://example.invalid/reviewed-openai-price",
        "claude-haiku-5-5": "https://example.invalid/reviewed-anthropic-price"
      }
    }

Compute file hashes with shasum -a 256. The review identifies rates, not
available funds. Budget, balance, credential and approval fields are rejected.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.release_scope import TARGET_OPERATIONS, ReleaseStage, stage_job_ids
from pmmc.experiments.step8_artifacts import (
    CampaignArtifactStore,
    SnapshotArtifactReader,
    json_bytes,
    sha256_file,
)
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt, validate_job_dag
from pmmc.experiments.step8_verification import _owned_file
from pmmc.schemas.step8 import (
    BASELINE_METHODS,
    RELEVANCE_THRESHOLDS,
    STEP8_ALEMBIC_REVISION,
    BenchmarkCampaignManifest,
    CampaignPhase,
    ExperimentCell,
    ResourceClass,
    SelectionExpansionManifest,
    Step8BenchmarkConfig,
    build_experiment_matrix,
)
from pmmc.schemas.step8_process import PROCESS_METRIC_NAMES, ProcessMetric

Row = dict[str, Any]
_STATES = {"pending", "running", "failed", "unknown", "completed"}
_METHOD_SCOPE = {"main": "pmmc_main", "analysis": "pmmc", "baselines": "paired_main"}
_STAGE_ORDER: tuple[ReleaseStage, ...] = ("main", "analysis", "baselines")
_REVIEW_FILE = "provider-policy.json"
_USD_QUANTUM = Decimal("0.00000001")
_Digest = Annotated[str, Field(pattern="^[a-f0-9]{64}$")]


class _PricingReview(BaseModel):
    """Only price provenance belongs here, never an account or a spend limit."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    protocol: Literal["pmmc-release-rate-review-v1"]
    currency: Literal["USD"]
    pricing_basis: Literal["standard_global_first_party_synchronous_before_tax"]
    reviewed_by: Annotated[str, Field(min_length=1)]
    reviewed_date: date
    rate_card_file: str
    rate_card_sha256: _Digest
    model_registry_sha256: _Digest
    pricing_sources: dict[str, HttpUrl]


def _read(directory: Path, relative: str | Path) -> bytes:
    path = directory / relative
    if not _owned_file(directory, path):
        raise ValueError(f"missing or unsafe campaign file: {relative}")
    return path.read_bytes()


def _identity(directory: Path) -> tuple[BenchmarkCampaignManifest, SelectionExpansionManifest]:
    manifest = BenchmarkCampaignManifest.model_validate_json(
        _read(directory, "manifests/campaign.json")
    )
    if manifest.phase is not CampaignPhase.EXPLORATION:
        raise ValueError("v0.1 reporting requires an exploration campaign")
    paths = list((directory / "manifests/selections").glob("*.json"))
    if len(paths) != 1:
        raise ValueError("exploration requires a single dev-all selection")
    selection = SelectionExpansionManifest.model_validate_json(
        _read(directory, paths[0].relative_to(directory))
    )
    if (
        selection.phase != manifest.phase
        or selection.campaign_id != manifest.campaign_id
        or selection.partition_hash != manifest.protocol.partition_hash
        or paths[0].stem != selection.content_hash
    ):
        raise ValueError("exploration selection identity mismatch")
    return manifest, selection


@dataclass
class _Snapshot:
    manifest: BenchmarkCampaignManifest
    selection: SelectionExpansionManifest
    jobs: dict[str, BenchmarkStageJob]
    rows: dict[str, Row]
    scoped: frozenset[str]

    def status(self, key: str) -> str:
        return self.rows[key]["status"] if key in self.rows else "unregistered"

    def header(self, scope: ReleaseStage) -> Row:
        return {
            "release": "v0.1",
            "phase": "exploration",
            "scope": scope,
            "paper_result": False,
            "scientific_approval": False,
            "campaign_id": self.manifest.campaign_id,
            "selection": self.selection.selection,
            "selection_hash": self.selection.content_hash,
            "protocol_hash": self.manifest.protocol.content_hash,
            "stage_job_counts": dict(sorted(Counter(self.status(k) for k in self.scoped).items())),
        }


def _snapshot(directory: Path, rows: Sequence[Row], scope: ReleaseStage) -> _Snapshot:
    if scope not in TARGET_OPERATIONS:
        raise ValueError("unknown exploration scope")
    manifest, selection = _identity(directory)
    inventory = json.loads(_read(directory, f"manifests/jobs-{selection.content_hash}.json"))
    if (
        not isinstance(inventory, list)
        or not inventory
        or any(
            not isinstance(k, str) or len(k) != 64 or not all(c in "0123456789abcdef" for c in k)
            for k in inventory
        )
        or len(set(inventory)) != len(inventory)
    ):
        raise ValueError("invalid exploration job inventory")
    by_id = {}
    for row in rows:
        job = BenchmarkStageJob.model_validate(row["job"])
        if (
            job.job_id in by_id
            or row["status"] not in _STATES
            or row.get("job_id", job.job_id) != job.job_id
        ):
            raise ValueError("duplicate job, mismatched job_id or invalid repository state")
        by_id[job.job_id] = row
    if by_id.keys() - set(inventory):
        raise ValueError("repository job outside exploration inventory")
    jobs = {}
    for key in inventory:
        # The job hash alone does not cover operation, dependencies or resource class.
        job = BenchmarkStageJob.model_validate_json(_read(directory, f"private/jobs/{key}.json"))
        if (
            job.job_id != key
            or job.campaign_id != manifest.campaign_id
            or manifest.protocol.source_fingerprints.get(job.stage) != job.source_fingerprint
            or (key in by_id and job.model_dump(mode="json") != by_id[key]["job"])
        ):
            raise ValueError("exploration job payload or frozen identity mismatch")
        jobs[key] = job
    validate_job_dag(list(jobs.values()))
    return _Snapshot(
        manifest, selection, jobs, by_id, stage_job_ids(jobs.values(), scope, paid=True)
    )


def _receipts(directory: Path, snapshot: _Snapshot) -> dict[str, ComponentReceipt]:
    """Trust only completed DB rows with a matching receipt, output CAS and parent chain."""
    reader = SnapshotArtifactReader(CampaignArtifactStore(directory.parent))
    receipts = {}
    for key in sorted(snapshot.scoped):
        if snapshot.status(key) != "completed":
            continue
        job = snapshot.jobs[key]
        receipt = ComponentReceipt.model_validate_json(
            _read(directory, f"checkpoints/components/{key}.json")
        )
        if (
            receipt.job_id != key
            or receipt.content_hash != snapshot.rows[key]["receipt_hash"]
            or receipt.scientific_input_hash != job.scientific_input_hash
            or receipt.source_fingerprint != job.source_fingerprint
            or receipt.schema_revision != STEP8_ALEMBIC_REVISION
            or set(receipt.upstream_receipt_hashes) != set(job.dependency_ids)
            or any(
                snapshot.status(parent) != "completed"
                or snapshot.rows[parent]["receipt_hash"] != digest
                for parent, digest in receipt.upstream_receipt_hashes.items()
            )
        ):
            raise ValueError("exploration component receipt identity mismatch")
        output_hash = hashlib.sha256(json_bytes(receipt.output)).hexdigest()
        if output_hash not in receipt.artifact_hashes:
            raise ValueError("exploration component has no bound output artifact")
        for digest in receipt.artifact_hashes:
            reader.verify(digest)
        receipts[key] = receipt
    return receipts


def _reported_cell(cell: ExperimentCell, scope: ReleaseStage) -> bool:
    primary = (cell.source_variant, cell.question_budget, cell.round_id, cell.treatment.value) == (
        "active",
        8,
        2,
        "main",
    )
    if scope == "analysis":
        return cell.method == "pmmc"
    return primary and (scope == "baselines" or cell.method == "pmmc")


def _outcome(cell: ExperimentCell, score: Row) -> str:
    terminal, source = score["terminal_state"], score.get("answer_source_method")
    if score.get("engineering_error"):
        return "engineering_missing"
    if terminal == "no_answer":
        return "no_answer"
    if terminal == "abstained":
        return "refusal"
    if terminal != "answered":
        return "engineering_missing"
    if not all(
        score.get(k) for k in ("typed_answer_id", "answer_source_variant", "source_fact_hash")
    ):
        return "source_unavailable"
    if cell.method != "pmmc":
        return "baseline_answer" if source == cell.method else "source_unavailable"
    if source == "native_multimodal_rag":
        return "fallback_answer"
    if source == "pmmc":
        return "supported_pmmc" if cell.source_variant == "active" else "counterfactual_pmmc"
    return "source_unavailable"


def _scores(
    receipt: ComponentReceipt, job: BenchmarkStageJob, scope: ReleaseStage
) -> dict[str, Row]:
    report = receipt.output
    p = job.parameters
    if (
        report.get("campaign_id") != job.campaign_id
        or report.get("query_id") != p["query_id"]
        or report.get("method_scope") != _METHOD_SCOPE[scope]
        or report.get("status") != "complete"
        or report.get("selected_query", {}).get("dataset") != p["dataset"]
        or report.get("selected_query", {}).get("query_id") != p["query_id"]
    ):
        raise ValueError("exploration evaluator scope mismatch")
    rows = report["scores"]
    scores = {s["cell_id"]: s for s in rows}
    if len(rows) != len(scores) or len(rows) != report["selected_cell_count"]:
        raise ValueError("exploration evaluator cell denominator mismatch")
    return scores


def _metric(values: list[float], selected: int) -> Row:
    return {
        "mean_observed": sum(values) / len(values) if values else None,
        "observed": len(values),
        "selected": selected,
    }


def _evidence(
    receipt: ComponentReceipt, receipt_hash: str, score: Row, cell: ExperimentCell
) -> Row:
    report = receipt.output
    found = [
        c
        for c in report.get("process_metrics", {}).get("cells", [])
        if c["cell_id"] == cell.cell_id
    ]
    if len(found) > 1:
        raise ValueError("duplicate process evidence")
    process = found[0] if found else {}
    if process and (
        process.get("query_id") != report["query_id"]
        or process.get("source_fact_hash") != score.get("source_fact_hash")
        or process.get("method") != cell.method
    ):
        raise ValueError("process evidence not bound to score")
    provenance = process.get("provenance", {})
    semantic = report.get("qbank_semantic", {})
    pools = [c for c in semantic.get("cells", []) if c["pool"]["cell_id"] == cell.cell_id]
    if len(pools) > 1 or (
        pools and pools[0]["pool"]["source_fact_hash"] != score.get("source_fact_hash")
    ):
        raise ValueError("semantic QBank evidence not bound to score")
    return {
        "query_id": report["query_id"],
        "receipt_hash": receipt_hash,
        "outcome": _outcome(cell, score),
        "typed_score": score.get("value"),
        "judge_score": score.get("judge_value"),
        "query_run_id": report.get("query_executor_run_id"),
        "source_fact_hash": score.get("source_fact_hash"),
        "typed_answer_id": score.get("typed_answer_id"),
        "answer_source_method": score.get("answer_source_method"),
        "round_members_hash": provenance.get("round_members_hash"),
        "match_batch_hash": provenance.get("match_batch_hash"),
        "eligible_bank_count": provenance.get("eligible_bank_count"),
        "ranked_answerable_count": provenance.get("ranked_answerable_count"),
        "semantic_cover": pools[0]["semantic_cover"] if pools else None,
        "semantic_match_at_k": pools[0]["semantic_match_at_k"] if pools else None,
        "relevance_grid_hash": report.get("relevance_grid", {}).get("content_hash"),
        "process_metrics": {
            name: ProcessMetric.model_validate(metric).model_dump()
            for name, metric in process.get("metrics", {}).items()
            if name in PROCESS_METRIC_NAMES
        },
    }


def _group_status(states: set[str]) -> str:
    return next(
        (s for s in ("unknown", "failed", "unregistered", "running", "pending") if s in states),
        "completed",
    )


def _relevance_provenance(k: str, receipt_hash: str, report: Row) -> Row:
    grid = report.get("relevance_grid", {})
    points = grid.get("points")
    joint_fields = (
        "grid_point_id",
        "cell_id",
        "joint_policy",
        "joint_threshold",
        "typed_answer_id",
        "joint_image_hashes",
        "arrived_joint_image_hashes",
        "answer_support",
        "status",
    )
    return {
        "job_id": k,
        "receipt_hash": receipt_hash,
        "query_id": report["query_id"],
        "query_run_id": report.get("query_executor_run_id"),
        "grid_status": grid.get("status"),
        "grid_hash": grid.get("content_hash"),
        "points": grid.get("point_count"),
        "distinct_generated_answers": grid.get("answer_count"),
        "point_states": dict(Counter(p["status"] for p in points)) if points is not None else None,
        # Older evaluator serializers omit joint point fields even when the
        # underlying runtime validated them. Never impute a pass or zero arrivals.
        "joint_points": [
            {field: point.get(field) for field in joint_fields}
            for point in (points or [])
            if point.get("joint_policy") is not None
        ]
        or None,
    }


def summarize_release_campaign(
    campaign_dir: Path,
    *,
    jobs: Sequence[Row],
    scope: ReleaseStage,
) -> Row:
    """Summarize one scope, all selected queries/backbones, without reading historical reports."""
    snapshot = _snapshot(campaign_dir, jobs, scope)
    receipts = _receipts(campaign_dir, snapshot)
    manifest, selection = snapshot.manifest, snapshot.selection
    config = Step8BenchmarkConfig.model_validate_json(_read(campaign_dir, "manifests/config.json"))
    matrix = build_experiment_matrix(manifest.protocol)
    if canonical_content_hash([c.model_dump(mode="json") for c in matrix]) != manifest.matrix_hash:
        raise ValueError("exploration matrix identity mismatch")
    targets = {}
    scores = {}
    for k in sorted(snapshot.scoped):
        job = snapshot.jobs[k]
        if job.operation != TARGET_OPERATIONS[scope]:
            continue
        p = job.parameters
        key = (p["dataset"], p["answer_model"], p["query_id"], p["treatment"])
        if (
            key in targets
            or p["query_id"] not in selection.query_ids.get(p["dataset"], [])
            or p["answer_model"] not in manifest.protocol.answer_backbones
        ):
            raise ValueError("duplicate or out-of-selection evaluator job")
        targets[key] = k
        if k in receipts:
            scores[k] = _scores(receipts[k], job, scope)

    @cache
    def failed_or_unknown_ancestor(key: str) -> bool:
        return any(
            snapshot.status(dep) in {"failed", "unknown"} or failed_or_unknown_ancestor(dep)
            for dep in snapshot.jobs[key].dependency_ids
        )

    groups = []
    for dataset, ids in sorted(selection.query_ids.items()):
        for model in manifest.protocol.answer_backbones:
            cells = [
                c
                for c in matrix
                if c.dataset == dataset and c.answer_model == model and _reported_cell(c, scope)
            ]
            observations = []
            by_query: dict[str, set[str]] = defaultdict(set)
            blocked = set()
            for cell in cells:
                counts: Counter[str] = Counter()
                outcomes: Counter[str] = Counter()
                typed, judged, evidence = [], [], []
                process_values: dict[str, list[float]] = defaultdict(list)
                reasons: Counter[str] = Counter()
                for q in ids:
                    target = targets.get((dataset.value, model, q, cell.treatment.value))
                    state = snapshot.status(target) if target is not None else "not_planned"
                    counts[state] += 1
                    by_query[q].add(state)
                    if (
                        state == "pending"
                        and target is not None
                        and failed_or_unknown_ancestor(target)
                    ):
                        blocked.add(q)
                    if target is None or target not in scores:
                        continue
                    s = scores[target].get(cell.cell_id)
                    if (
                        s is None
                        or s["method"] != cell.method
                        or (cell.method == "pmmc" and s["source_variant"] != cell.source_variant)
                    ):
                        raise ValueError(
                            "completed evaluator omitted or mismatched a selected cell"
                        )
                    outcomes[_outcome(cell, s)] += 1
                    if s.get("value") is not None:
                        typed.append(s["value"])
                    if s.get("judge_value") is not None:
                        judged.append(s["judge_value"])
                    fact = _evidence(
                        receipts[target], snapshot.rows[target]["receipt_hash"], s, cell
                    )
                    for name, metric in fact.pop("process_metrics").items():
                        if metric["status"] == "evaluated":
                            process_values[name].append(metric["value"])
                        else:
                            reasons[metric["reason"]] += 1
                    evidence.append(fact)
                observations.append(
                    {
                        "cell_id": cell.cell_id,
                        "method": cell.method,
                        "variant": cell.source_variant,
                        "treatment": cell.treatment.value,
                        "question_budget": cell.question_budget,
                        "round": cell.round_id,
                        "selected": len(ids),
                        "evaluation_job_states": dict(counts),
                        "observed_outcomes": dict(outcomes) if evidence else None,
                        "typed_exact_match": _metric(typed, len(ids)),
                        "judge": _metric(judged, len(ids)),
                        "process": {
                            name: _metric(process_values[name], len(ids))
                            for name in PROCESS_METRIC_NAMES
                        },
                        "process_unavailable_reasons": dict(reasons),
                        "evidence": evidence,
                    }
                )
            states = Counter(
                "not_planned" if "not_planned" in by_query[q] else _group_status(by_query[q])
                for q in ids
            )
            job_states = Counter(
                snapshot.status(k)
                for k in snapshot.scoped
                if snapshot.jobs[k].parameters.get("dataset") == dataset.value
                and snapshot.jobs[k].parameters.get("answer_model") == model
            )
            groups.append(
                {
                    "dataset": dataset.value,
                    "answer_model": model,
                    "selected_queries": len(ids),
                    "query_evaluation_states": dict(states),
                    "queries_with_pending_evaluation_blocked_by_failure_or_unknown": len(blocked),
                    "model_job_states": dict(job_states),
                    "cells": observations,
                }
            )
    protocol = manifest.protocol
    return {
        **snapshot.header(scope),
        "schema": "v0.1-exploration-readout-v1",
        "groups": groups,
        "exploration_variant": config.metadata.get("exploration_variant"),
        "exploration_run_id": config.metadata.get("exploration_run_id"),
        "configured_main_thresholds": {
            "match": protocol.match_threshold,
            "gap": protocol.gap_threshold,
            "text_relevance": protocol.text_relevance_threshold,
            "multimodal_relevance": protocol.multimodal_relevance_threshold,
        },
        "joint_embedding_evidence": [
            {
                "job_id": k,
                "receipt_hash": snapshot.rows[k]["receipt_hash"],
                **{
                    name: snapshot.jobs[k].parameters.get(name)
                    for name in ("dataset", "answer_model", "query_id", "treatment")
                },
                **{
                    name: receipt.output.get(name)
                    for name in (
                        "joint_pair_count",
                        "joint_model_call_count",
                        "joint_reused_batch_count",
                    )
                },
            }
            for k, receipt in receipts.items()
            if snapshot.jobs[k].operation == "query_embedding"
            and "joint_pair_count" in receipt.output
        ]
        or None,
        "r_evaluation_evidence": [
            _relevance_provenance(k, snapshot.rows[k]["receipt_hash"], receipts[k].output)
            for k in sorted(scores)
        ]
        or None,
        "compiler_evidence": [
            {
                "job_id": k,
                "receipt_hash": snapshot.rows[k]["receipt_hash"],
                "stage": snapshot.jobs[k].stage,
                "dataset": snapshot.jobs[k].parameters["dataset"],
                "conversation_id": snapshot.jobs[k].parameters.get("conversation_id"),
                "treatment": snapshot.jobs[k].parameters.get("treatment"),
                **{
                    name: receipt.output.get(name)
                    for name in (
                        "compiler_run_id",
                        "program_count",
                        "eligible_candidate_count",
                        "compiled_instance_hash",
                    )
                },
            }
            for k, receipt in receipts.items()
            if snapshot.jobs[k].operation in {"compiler", "adopt_questioner"}
        ]
        or None,
        "timing_seconds": None,
        "actual_cost_usd": None,
        "limitations": [
            "Observed means have explicit coverage; they are not full-denominator accuracy.",
            "Configured thresholds are not observed counterfactual policy scores.",
            "Prepared joint embeddings do not prove an image reached a selected answer.",
            "These receipts do not establish wall time, API settlements or GPU prices.",
        ],
    }


def _reviewed_rates(
    directory: Path,
    manifest: BenchmarkCampaignManifest,
    rate_card: Path,
    project_root: Path,
):
    from pmmc.models.registry import load_model_registry
    from pmmc.models.step8_accounting import load_rate_card, preflight_provider_rate_card

    review_bytes = _read(rate_card.parent, _REVIEW_FILE)
    review = _PricingReview.model_validate_json(review_bytes)
    config = Step8BenchmarkConfig.model_validate_json(_read(directory, "manifests/config.json"))
    registry_path = project_root / config.model_registry
    registry_hash = sha256_file(registry_path)
    if (
        registry_hash != manifest.protocol.config_hashes["model_registry"]
        or registry_hash != review.model_registry_sha256
        or review.rate_card_file != rate_card.name
    ):
        raise ValueError("pricing review must identify this card and frozen registry")
    registry = load_model_registry(registry_path)
    rates = load_rate_card(rate_card)
    preflight = preflight_provider_rate_card(
        rate_card, registry, manifest.protocol.answer_backbones
    )
    if (
        rates.sha256 != review.rate_card_sha256
        or preflight["rate_card_hash"] != rates.sha256
        or rates.card.estimated
    ):
        raise ValueError("a reviewed, hash-matching provider rate card is required")
    models = {m["provider_model"] for m in preflight["models"]}
    if not models or any(
        not any(rates.card.models[model].model_dump().values()) for model in models
    ):
        raise ValueError("release API work needs positive reviewed rates")
    if set(review.pricing_sources) != models or any(
        url.scheme != "https" or url.username or url.password
        for url in review.pricing_sources.values()
    ):
        raise ValueError("pricing review needs an HTTPS price source per configured paid model")
    return rates, registry, hashlib.sha256(review_bytes).hexdigest()


def _contexts(job: BenchmarkStageJob) -> int:
    # Each paid PMMC backbone consumes one b8/r2 bank, never the 9B b/R grid.
    # Match/gap decisions and no_fallback/forced_top1 select existing answers.
    if job.operation == "query_answer":
        return 1 if job.parameters.get("readout") == "primary" else len(RELEVANCE_THRESHOLDS) ** 2
    if job.operation == "native_fallback_generate":
        return 1
    if job.operation == "baseline_answer":
        return 1 if job.parameters.get("treatment") == "no_raw_image" else len(BASELINE_METHODS)
    raise ValueError(f"no reviewed paid-call bound for operation {job.operation}")


def _unit_prices(rates, entry) -> Row:
    from pmmc.models.provider_config import TYPED_ANSWER_OUTPUT_TOKENS
    from pmmc.models.provider_token_budget import SHARED_CONTEXT_TOKENS
    from pmmc.schemas.model import GenerationRequest

    config = entry.provider_config
    if config is None or config.kind not in {"openai", "anthropic"} or not entry.context_window:
        raise ValueError("API job requires a reviewed paid model with a context limit")
    output = config.typed_answer_output_tokens
    inputs = SHARED_CONTEXT_TOKENS - output
    request = GenerationRequest.model_validate(
        {
            "request_id": "offline-budget-estimate",
            "model_alias": entry.alias,
            "messages": [{"role": "user", "parts": [{"kind": "text", "text": "estimate"}]}],
            "max_output_tokens": TYPED_ANSWER_OUTPUT_TOKENS,
            "provider_max_output_tokens": output,
        }
    )
    prices = rates.card.models[entry.provider_model]
    input_rate = max(
        prices.input_usd_per_million,
        prices.cache_creation_input_usd_per_million,
        prices.cached_input_usd_per_million,
    )
    # The ledger reserves at the entire model context + its conservative tier
    # multiplier. This differs from an assembly-budget planning scenario.
    reservation = rates.worst_case(request, entry, output).quantize(
        _USD_QUANTUM, rounding=ROUND_CEILING
    )
    per_call = (inputs * input_rate + output * prices.output_usd_per_million) / Decimal(1_000_000)
    return {
        "provider_model": entry.provider_model,
        "input_token_allowance": inputs,
        "provider_output_token_cap": output,
        "public_output_token_cap": TYPED_ANSWER_OUTPUT_TOKENS,
        "standard_rate_scenario_per_call_usd": str(per_call),
        "ledger_reservation_per_call_usd": str(reservation),
    }


def estimate_release_budget(
    campaign_dir: Path,
    *,
    jobs: Sequence[Row],
    scope: ReleaseStage,
    rate_card: Path,
    project_root: Path,
) -> Row:
    """Price only registered pending API jobs in the dependency-closed requested stage.

    Report both the stand-alone scope and the increment over earlier scopes,
    so overlapping scientific jobs are not accidentally budgeted twice.
    """
    snapshot = _snapshot(campaign_dir, jobs, scope)
    rates, registry, review_hash = _reviewed_rates(
        campaign_dir, snapshot.manifest, rate_card, project_root
    )
    preceding: set[str] = set()
    for stage in _STAGE_ORDER:
        if stage == scope:
            break
        preceding.update(stage_job_ids(snapshot.jobs.values(), stage, paid=True))
    counts: Counter[tuple[str, str, str, int, bool]] = Counter()
    api_states: Counter[str] = Counter()
    for k in sorted(snapshot.scoped):
        job = snapshot.jobs[k]
        if job.resource_class != ResourceClass.API:
            continue
        state = snapshot.status(k)
        api_states[state] += 1
        if state == "pending":
            counts[
                (
                    job.parameters["dataset"],
                    job.parameters["answer_model"],
                    job.operation,
                    _contexts(job),
                    k in preceding,
                )
            ] += 1
    totals = [Decimal(0), Decimal(0)]
    incremental = [Decimal(0), Decimal(0)]
    groups = []
    for (dataset, model, operation, contexts, shared), count in sorted(counts.items()):
        prices = _unit_prices(rates, registry.get(model))
        calls = count * contexts * 2  # Initial generation plus at most one typed format repair.
        scenario = calls * Decimal(prices["standard_rate_scenario_per_call_usd"])
        reserve = calls * Decimal(prices["ledger_reservation_per_call_usd"])
        totals[0] += scenario
        totals[1] += reserve
        if not shared:
            incremental[0] += scenario
            incremental[1] += reserve
        groups.append(
            {
                "dataset": dataset,
                "answer_model": model,
                "operation": operation,
                "in_preceding_scope": shared,
                "pending_jobs": count,
                "contexts_per_job_at_most": contexts,
                "generation_calls_at_most": calls,
                **prices,
                "standard_rate_scenario_usd": str(scenario),
                "one_pass_reservation_bound_usd": str(reserve),
            }
        )
    registered = not api_states["unregistered"]

    def bounds(pair: list[Decimal]) -> Row:
        return {
            "standard_rate_scenario_usd": str(pair[0]) if registered else None,
            "one_pass_reservation_bound_usd": str(pair[1]) if registered else None,
        }

    return {
        **snapshot.header(scope),
        "schema": "v0.1-pending-api-estimate-v1",
        "rate_card_sha256": rates.sha256,
        "review": _REVIEW_FILE,
        "review_sha256": review_hash,
        "api_job_states": dict(sorted(api_states.items())),
        "pending_api_groups": groups,
        "pending_scope": bounds(totals),
        "pending_additional_after_preceding_scopes": bounds(incremental),
        "complete_api_inventory_registered": registered,
        "approval_created": False,
        "requires_explicit_selection_approval": True,
        "legacy_budget_allocated": False,
        "limitations": [
            "This is an offline estimate, not an invoice or spending authority.",
            "Assembly estimates include images. Actual usage and cache shares may differ.",
            "Bounds include one format repair per context, no job retries or transport resends.",
            "Distinct sufficient contexts may be fewer; baseline requests may be shared. "
            "No unverified reuse discount is applied.",
            "Only pending jobs are priced. Reconcile existing holds before approving spend.",
            "Scopes overlap. Sum only incremental estimates from the same snapshot.",
            "Historical balances, GPU cost, taxes and external clients are excluded.",
        ],
    }


async def approve_release_budget(
    campaign_dir: Path,
    *,
    selection_hash: str,
    rate_card: Path,
    project_root: Path,
    approver_id: str,
    limit_usd: Decimal,
    scope: ReleaseStage | None = None,
) -> Row:
    """Explicit opt-in only. Register this selection's limit; never start workers."""
    from pmmc.experiments.step8_cli import _register

    manifest, selection = _identity(campaign_dir)
    if selection_hash != selection.content_hash or not approver_id.strip():
        raise ValueError("explicit approver and exact exploration selection hash required")
    if (
        not limit_usd.is_finite()
        or not 0 < limit_usd < 10**12
        or limit_usd.quantize(_USD_QUANTUM) != limit_usd
    ):
        raise ValueError("explicit positive finite NUMERIC(20,8) limit required")
    rates, _, review_hash = _reviewed_rates(campaign_dir, manifest, rate_card, project_root)
    stages = [scope] if scope is not None else list(TARGET_OPERATIONS)
    if scope is not None and scope not in TARGET_OPERATIONS:
        raise ValueError("unknown exploration approval scope")
    for stage in stages:
        relative = f"reports/budget-quote-{stage}.json"
        if not (campaign_dir / relative).exists():
            continue
        quote = json.loads(_read(campaign_dir, relative))
        if (
            quote.get("schema") == "v0.1-pending-api-estimate-v1"
            and quote.get("scope") == stage
            and quote.get("campaign_id") == manifest.campaign_id
            and quote.get("selection_hash") == selection_hash
            and quote.get("rate_card_sha256") == rates.sha256
            and quote.get("review_sha256") == review_hash
            and quote.get("complete_api_inventory_registered") is True
        ):
            break
    else:
        raise ValueError("approval requires a saved quote for this selection, review and rate card")
    saved = campaign_dir / "manifests" / f"budget-{selection_hash}.json"
    if saved.exists():
        approval = json.loads(_read(campaign_dir, saved.relative_to(campaign_dir)))
        if (
            approval["selection_hash"] != selection_hash
            or approval["rate_card_hash"] != rates.sha256
            or Decimal(approval["limit_usd"]) != limit_usd
            or approval["approver_id"] != approver_id
        ):
            raise ValueError("conflicting immutable exploration approval")
        # Preserve the canonical amount spelling for the existing immutable file.
        limit_usd = Decimal(approval["limit_usd"])
    approval_id, _ = await _register(
        campaign_dir, budget=limit_usd, rate_card=rate_card, approver=approver_id, retry=False
    )
    if not approval_id:
        raise ValueError("register did not return a selection-specific approval")
    return {
        "campaign_id": manifest.campaign_id,
        "selection_hash": selection_hash,
        "approval_id": approval_id,
        "rate_card_sha256": rates.sha256,
        "limit_usd": str(limit_usd),
    }
