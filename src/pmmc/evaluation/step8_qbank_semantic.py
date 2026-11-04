"""Evaluator-only Step8 semantic pairs, independent of answer backbone and b/r.

The caller owns one canonical evaluation parent on the source-match query run,
loads ProcessFacts/gold/images using evaluator permissions, and persists the final
report before completing that parent. No legacy semantic/coverage rows are made.
Successful calls use the existing audited execution and campaign response journal.
An unresolved dispatch is never retried automatically or scored as a negative.
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from pmmc.compiler.ids import canonical_content_hash, question_content_hash
from pmmc.evaluation import qbank_semantic, step8_process_metrics
from pmmc.evaluation.qbank_semantic import (
    QBankSemanticMatchConfig,
    build_qbank_semantic_request,
    parse_qbank_semantic_output,
    qbank_semantic_prompt_hash,
    qbank_semantic_response_schema_hash,
    semantic_reason_codes,
)
from pmmc.evaluation.step8_process_metrics import ProcessFacts, _bound_fact
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.models import ModelCallContext, ModelExecutionService, ParsedModelOutput
from pmmc.models.campaign_audit import current_campaign_audit
from pmmc.models.execution import _generation_manifest
from pmmc.models.gateways import RoutedGenerationGateway
from pmmc.models.protocols import GenerationGateway
from pmmc.models.routing import stable_replica_index
from pmmc.schemas import (
    AppStage,
    DataScope,
    GenerationRequest,
    GenerationResult,
    ImageRef,
    ModelCallTrace,
    QueryInput,
    QuestionCandidate,
    TraceStatus,
)
from pmmc.schemas.semantic import QBankSemanticJudgeDimensions, QBankSemanticJudgeEnvelopeV2
from pmmc.schemas.step8_process import MATCH_KS, ProcessCellBinding

VERSION: Final = "step8-qbank-semantic-v2"
JUDGE_ALIAS: Final = "judge-qwen3.5-9b"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Step8SemanticError(ValueError):
    """Code-only error: never expose private query, gold, images or credentials."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Step8SemanticError(code)


class Step8SemanticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["step8-qbank-semantic-v2"] = VERSION
    judge_model_alias: Literal["judge-qwen3.5-9b"] = JUDGE_ALIAS
    max_output_tokens: int = Field(default=512, ge=1, le=1024)
    timeout_seconds: float = Field(default=120.0, gt=0)
    concurrency: int = Field(default=64, ge=1, le=64)

    @property
    def renderer(self) -> QBankSemanticMatchConfig:
        # V3 is dimensions-only and does not call the pair a Step7B candidate.
        return QBankSemanticMatchConfig(max_output_tokens=self.max_output_tokens)

    @property
    def protocol_hash(self) -> str:
        return canonical_content_hash(
            {
                "config": self.model_dump(mode="json"),
                "renderer_config": self.renderer.model_dump(mode="json"),
                "prompt_hash": qbank_semantic_prompt_hash(self.renderer),
                "response_schema_hash": qbank_semantic_response_schema_hash(self.renderer),
                "response_schema_policy": "boolean_iff_both_have_images_else_not_applicable",
                "source_hashes": {
                    name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for name, path in (
                        ("adapter", __file__),
                        ("dimensions_renderer_parser", qbank_semantic.__file__),
                        ("runtime_lineage_binding", step8_process_metrics.__file__),
                    )
                },
                "cover_pool": "all_exact_round_eligible_answerable_questions",
                "match_pool": "saved_runtime_answerable_rank_lte_50",
                "match_ks": list(MATCH_KS),
                "verdict": "all_semantic_dimensions_without_evidence_proxy_gate",
                "dispatch": "one_attempt_fail_closed_on_unknown_dispatch",
            }
        )


class SemanticCellPool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str
    question_budget: int
    round_id: int
    source_fact_id: str
    source_fact_hash: str
    match_batch_id: str
    match_batch_hash: str
    round_projection_id: str
    round_projection_hash: str
    round_members_hash: str
    cover_qbank_ids: tuple[str, ...]
    match_qbank_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticPair:
    qbank_id: str
    pair_hash: str
    question_hash: str
    support_hash: str
    request: GenerationRequest

    @property
    def call_id(self) -> str:
        return self.request.request_id


@dataclass(frozen=True)
class Step8SemanticPlan:
    query: QueryInput
    snapshot_id: str
    compiler_run_id: str
    canonical_query_run_id: str
    protocol_hash: str
    judge_model_hash: str
    owner_scope_hash: str
    cells: tuple[SemanticCellPool, ...]
    pairs: tuple[SemanticPair, ...]
    content_hash: str


class SemanticEvaluationOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_run_id: str
    query_run_id: str
    scope_hash: str


def build_step8_semantic_owner(
    plan: Step8SemanticPlan, *, canonical_query_run_id: str
) -> SemanticEvaluationOwner:
    """Choose one pre-registered query run, shared by main/no-reranker/backbones.

    The caller registers this evaluation parent with legacy_source_run_id set to
    query_run_id. That run must contain the snapshot/query in runtime.run_queries;
    a compiler run is NOT a valid substitute. No new runtime rows are created here.
    """
    _require(bool(canonical_query_run_id), "semantic_owner_query_run_missing")
    return SemanticEvaluationOwner(
        evaluation_run_id="step8-semantic-eval-" + plan.owner_scope_hash[:32],
        query_run_id=canonical_query_run_id,
        scope_hash=plan.owner_scope_hash,
    )


def _cell_pool(binding, query, snapshot_id, query_run_id, facts) -> SemanticCellPool:
    fact = _bound_fact(binding, query, snapshot_id, query_run_id, facts)
    match = facts.matches.get(fact.route.match_batch_id)
    _require(match is not None, "semantic_match_batch_missing")
    assert match is not None and facts.run is not None
    batch = match.batch
    budget, round_id = binding.question_budget, binding.round_id
    _require(
        (
            batch.query_executor_run_id,
            batch.query_id,
            batch.snapshot_id,
            batch.conversation_id,
            batch.cutoff_session_id,
            batch.question_budget,
            batch.active_round,
            batch.source_compiler_run_id,
            batch.routing_projection_id,
        )
        == (
            fact.route.query_executor_run_id,
            query.query_id,
            snapshot_id,
            query.conversation_id,
            query.available_after_session,
            budget,
            round_id,
            facts.run["source_compiler_run_id"],
            facts.run["routing_projection_id"],
        ),
        "semantic_match_scope_mismatch",
    )
    classes = {"answerable": 0, "control": 1}
    ranked = sorted(match.candidates, key=lambda c: (classes[c.route_class.value], c.rank))
    channels = sorted(
        match.channels, key=lambda c: (c["channel"], classes[c["route_class"]], c["rank"])
    )
    _require(
        batch.fused_count == len(ranked)
        and batch.content_hash
        == canonical_content_hash(
            {
                "run_id": batch.query_executor_run_id,
                "query_id": query.query_id,
                "projection_id": batch.routing_projection_id,
                "question_budget": budget,
                "round_id": round_id,
                "channel_hashes": [c["content_hash"] for c in channels],
                "fused_hashes": [c.content_hash for c in ranked],
            }
        ),
        "semantic_match_hash_mismatch",
    )
    projection = facts.rounds.get(round_id)
    _require(projection is not None, "semantic_exact_round_missing")
    assert projection is not None
    manifest, members = projection
    _require(
        manifest.schema_version == "0.4.0"
        and manifest.run_id == batch.source_compiler_run_id
        and manifest.snapshot_id == snapshot_id
        and manifest.round_id == round_id
        and manifest.pending_count == 0
        and manifest.member_count == len(members)
        and manifest.accepted_count
        == sum(m.outcome in {"accepted", "carried_forward"} for m in members)
        and manifest.rejected_count
        == sum(m.outcome in {"rejected", "max_rounds_exhausted"} for m in members)
        and len({m.qbank_id for m in members}) == len(members)
        and {m.qbank_id for m in members} == set(facts.questions)
        and manifest.members_hash
        == canonical_content_hash(
            [m.model_dump(mode="json") for m in sorted(members, key=lambda m: m.qbank_id)]
        )
        and manifest.content_hash
        == canonical_content_hash(
            manifest.model_dump(
                mode="json", exclude={"schema_version", "content_hash", "created_at", "metadata"}
            )
        ),
        "semantic_round_denominator_or_hash_mismatch",
    )
    eligible = {}
    for member in members:
        question = QuestionCandidate.model_validate(facts.questions[member.qbank_id])
        _require(
            member.run_id == manifest.run_id
            and member.projection_id == manifest.projection_id
            and question.run_id == manifest.run_id
            and question.snapshot_id == snapshot_id
            and question_content_hash(question)
            == question.question_hash
            == question.candidate_content_hash,
            "semantic_question_or_member_lineage_mismatch",
        )
        if (
            member.outcome in {"accepted", "carried_forward"}
            and member.route_class == "answerable"
            and question.conversation_id == query.conversation_id
            and question.created_after_session in facts.visible_sessions
            and question.generation_rank <= budget
        ):
            eligible[member.qbank_id] = member
    answerable = []
    for candidate in ranked:
        legacy_payload = {
            "query_executor_run_id": batch.query_executor_run_id,
            "query_id": query.query_id,
            "question_budget": budget,
            "round_id": round_id,
            "route_class": candidate.route_class.value,
            "rank": candidate.rank,
            "qbank_id": candidate.qbank_id,
            "raw_rrf": candidate.raw_rrf,
        }
        _require(
            candidate.match_batch_id == batch.match_batch_id
            and candidate.query_id == query.query_id
            and candidate.content_hash
            in {
                canonical_content_hash(legacy_payload),
                canonical_content_hash(
                    candidate.model_dump(mode="json", exclude={"content_hash", "metadata"})
                ),
            },
            "semantic_ranked_candidate_hash_mismatch",
        )
        if candidate.route_class == "answerable":
            _require(
                candidate.qbank_id in eligible
                and candidate.plan_id == eligible[candidate.qbank_id].effective_plan_id,
                "semantic_ranked_candidate_outside_exact_bank",
            )
            answerable.append(candidate)
    _require(
        [c.rank for c in answerable] == list(range(1, len(answerable) + 1))
        and len({c.qbank_id for c in answerable}) == len(answerable),
        "semantic_rank_denominator_invalid",
    )
    return SemanticCellPool(
        cell_id=binding.cell_id,
        question_budget=budget,
        round_id=round_id,
        source_fact_id=binding.source_fact_id,
        source_fact_hash=binding.source_fact_hash,
        match_batch_id=batch.match_batch_id,
        match_batch_hash=batch.content_hash,
        round_projection_id=manifest.projection_id,
        round_projection_hash=manifest.content_hash,
        round_members_hash=manifest.members_hash,
        cover_qbank_ids=tuple(sorted(eligible)),
        match_qbank_ids=tuple(c.qbank_id for c in answerable if c.rank <= 50),
    )


def build_step8_semantic_plan(
    *,
    query: QueryInput,
    snapshot_id: str,
    query_run_id: str,
    compiler_schema_version: str,
    facts: ProcessFacts,
    bindings: Sequence[ProcessCellBinding],
    gold_answer: str,
    evidence_images: Mapping[str, Sequence[ImageRef]],
    judge_model_hash: str,
    config: Step8SemanticConfig | None = None,
) -> Step8SemanticPlan:
    """Validate an exact grid and union its FULL banks, not only retrieved pairs.

    ``evidence_images`` must come from the expected-support image IDs through an
    evaluator-owned loader. The adapter neither opens DB connections nor obtains
    any object-store/runtime credentials. Gold is never a retrieval input.
    """
    config = config or Step8SemanticConfig()
    _require(compiler_schema_version == "0.9.0", "semantic_requires_compiler_0_9")
    _require(bool(gold_answer.strip()), "semantic_gold_answer_missing")
    _require(bool(_SHA256.fullmatch(judge_model_hash)), "semantic_model_hash_invalid")
    _require(bool(bindings), "semantic_grid_empty")
    _require(all(b.method == "pmmc" for b in bindings), "semantic_non_pmmc_binding")
    _require(
        len({b.cell_id for b in bindings}) == len(bindings),
        "semantic_duplicate_cell",
    )
    _require(
        all(b.question_budget in {1, 2, 4, 8} and b.round_id in {0, 1, 2, 3} for b in bindings),
        "semantic_cell_outside_grid",
    )
    cells = tuple(
        _cell_pool(b, query, snapshot_id, query_run_id, facts)
        for b in sorted(bindings, key=lambda b: (b.question_budget, b.round_id, b.cell_id))
    )
    assert facts.run is not None
    compiler_run_id = str(facts.run["source_compiler_run_id"])
    canonical_run_id = str(facts.run.get("metadata", {}).get("source_match_run_id") or query_run_id)
    query_hash = canonical_content_hash(query.model_dump(mode="json"))
    owner_scope_hash = canonical_content_hash(
        {
            "protocol_hash": config.protocol_hash,
            "judge_model_hash": judge_model_hash,
            "compiler_run_id": compiler_run_id,
            "snapshot_id": snapshot_id,
            "query_hash": query_hash,
            "gold_answer_hash": canonical_content_hash(gold_answer),
        }
    )
    pairs = []
    for qbank_id in sorted({q for cell in cells for q in cell.cover_qbank_ids}):
        question = QuestionCandidate.model_validate(facts.questions[qbank_id])
        refs = sorted(facts.support.get(qbank_id, []), key=lambda r: r["ordinal"])
        _require(
            bool(refs)
            and [r["evidence_id"] for r in refs] == question.evidence_ids
            and len({r["ordinal"] for r in refs}) == len(refs)
            and all(
                r["qbank_id"] == qbank_id
                and r["run_id"] == compiler_run_id
                and r["snapshot_id"] == snapshot_id
                and r["role"] == "expected"
                for r in refs
            ),
            "semantic_expected_support_missing_or_mismatched",
        )
        images = sorted(evidence_images.get(qbank_id, ()), key=lambda image: image.image_id)
        _require(
            {image.image_id for image in images}
            == {r["image_id"] for r in refs if r.get("image_id")}
            and len({i.image_id for i in images}) == len(images),
            "semantic_evidence_images_missing_or_foreign",
        )
        _require(
            all(
                i.sha256 and _SHA256.fullmatch(i.sha256) for i in (*query.query_image_refs, *images)
            ),
            "semantic_image_content_hash_missing",
        )
        _require(bool(question.expected_answer), "semantic_candidate_answer_missing")
        request = build_qbank_semantic_request(
            semantic_run_id="step8",
            semantic_job_id="pair",
            pair_alias="P01",
            query=query,
            candidate=question,
            gold_answer=gold_answer,
            evidence_images=images,
            config=config.renderer,
            timeout_seconds=config.timeout_seconds,
        )
        response_format = copy.deepcopy(request.response_format)
        schema = response_format["json_schema"]["schema"]
        # Applicability is known before dispatch, not a model decision. Keep all
        # images and the other semantic dimensions unchanged.
        binding_required = bool(query.query_image_refs and images)
        schema["$defs"]["QBankSemanticJudgeDimensions"]["properties"][
            "query_image_binding_compatible"
        ] = {"type": "boolean"} if binding_required else {
            "type": "string", "const": "not_applicable"
        }
        request = request.model_copy(update={"response_format": response_format})
        support_hash = canonical_content_hash(refs)
        question_hash = question_content_hash(question)
        pair_hash = canonical_content_hash(
            {
                "protocol_hash": config.protocol_hash,
                "judge_model_hash": judge_model_hash,
                "snapshot_id": snapshot_id,
                "compiler_run_id": compiler_run_id,
                "query_hash": query_hash,
                "qbank_id": qbank_id,
                "question_hash": question_hash,
                "support_hash": support_hash,
                "messages": [m.model_dump(mode="json") for m in request.messages],
            }
        )
        request = request.model_copy(
            update={
                "request_id": "step8-semantic-" + pair_hash,
                "metadata": {
                    "task": "step8_qbank_semantic",
                    "protocol_version": VERSION,
                    "protocol_hash": config.protocol_hash,
                    "semantic_pair_hash": pair_hash,
                },
            }
        )
        pairs.append(SemanticPair(qbank_id, pair_hash, question_hash, support_hash, request))
    plan_hash = canonical_content_hash(
        {
            "protocol_hash": config.protocol_hash,
            "compiler_run_id": compiler_run_id,
            "query_hash": query_hash,
            "snapshot_id": snapshot_id,
            "canonical_query_run_id": canonical_run_id,
            "owner_scope_hash": owner_scope_hash,
            "cells": [c.model_dump(mode="json") for c in cells],
            "pair_hashes": [p.pair_hash for p in pairs],
        }
    )
    return Step8SemanticPlan(
        query,
        snapshot_id,
        compiler_run_id,
        canonical_run_id,
        config.protocol_hash,
        judge_model_hash,
        owner_scope_hash,
        cells,
        tuple(pairs),
        plan_hash,
    )


class SemanticPairResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    qbank_id: str
    pair_hash: str
    model_call_id: str
    question_hash: str
    support_hash: str
    status: Literal["evaluated", "incomplete"]
    semantic_match: bool | None = None
    dimensions: QBankSemanticJudgeDimensions | None = None
    reason_codes: tuple[str, ...] = ()
    error_code: str | None = None
    trace_hash: str | None = None
    response_transition: dict[str, str] | None = None


class SemanticObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["evaluated", "incomplete"]
    value: int | None
    pair_count: int
    evaluated_pair_count: int


class SemanticCellResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: SemanticCellPool
    semantic_cover: SemanticObservation
    semantic_match_at_k: dict[int, SemanticObservation]


class Step8SemanticReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = VERSION
    protocol_hash: str
    plan_hash: str
    evaluation_run_id: str
    owner_query_run_id: str
    canonical_query_run_id: str
    compiler_run_id: str
    snapshot_id: str
    query_id: str
    judge_model_alias: str = JUDGE_ALIAS
    judge_model_hash: str
    status: Literal["evaluated", "incomplete"]
    cells: tuple[SemanticCellResult, ...]
    pairs: tuple[SemanticPairResult, ...]
    oracle_answer_accuracy: Literal["not_evaluated"] = "not_evaluated"

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


_RESPONSE_TRANSITION: ContextVar[tuple[Step8SemanticReport, str] | None] = ContextVar(
    "step8_semantic_response_transition", default=None
)

_EXPLICIT_REPLACEMENT: ContextVar[tuple[dict, str] | None] = ContextVar(
    "step8_semantic_explicit_replacement", default=None
)


@contextmanager
def semantic_explicit_replacement(approval: dict, *, approval_sha256: str):
    """One reviewed unknown call may have one separately identified successor."""
    _require(canonical_content_hash(approval) == approval_sha256,
             "semantic_replacement_approval_changed")
    _require(approval.get("max_replacements") == 1
             and approval.get("previous_outcome") == "unknown"
             and approval.get("previous_outcome_preserved") is True,
             "semantic_replacement_scope_invalid")
    token = _EXPLICIT_REPLACEMENT.set((copy.deepcopy(approval), approval_sha256))
    try:
        yield
    finally:
        _EXPLICIT_REPLACEMENT.reset(token)


async def _explicit_replacement(pair, plan, owner, load_trace, audit):
    approved = _EXPLICIT_REPLACEMENT.get()
    if approved is None or pair.call_id != approved[0]["previous_model_call_id"]:
        return pair, None
    record, approval_hash = approved
    _require(audit.campaign_id == record["campaign_id"]
             and owner.evaluation_run_id == record["evaluation_run_id"]
             and pair.pair_hash == record["pair_hash"]
             and canonical_content_hash(_generation_manifest(pair.request))
             == record["request_content_hash"], "semantic_replacement_request_changed")
    trace = await load_trace(pair.call_id)
    _require(trace is not None, "semantic_replacement_original_trace_missing")
    _validate_trace(trace, pair, plan, owner)
    _require(canonical_content_hash(trace.model_dump(mode="json")) == record["trace_hash"]
             and trace.status is TraceStatus.FAILED and not trace.retry_count
             and trace.error_type == "ProviderCallError" and trace.output_asset_id is None,
             "semantic_replacement_original_trace_changed")
    call_id = "step8-semantic-replacement-" + canonical_content_hash({
        "previous_model_call_id": pair.call_id, "approval_sha256": approval_hash,
    })
    provenance = {
        "action": "explicit_unknown_replacement",
        "approval_sha256": approval_hash,
        "previous_model_call_id": pair.call_id,
        "previous_trace_hash": record["trace_hash"],
        "previous_outcome": "unknown",
    }
    successor = replace(pair, request=pair.request.model_copy(update={"request_id": call_id}))
    return successor, provenance


@contextmanager
def semantic_response_transition(report: Step8SemanticReport, *, approval_sha256: str):
    """Explicit evaluator-only carry-forward; never rewrite an old call or verdict."""
    _require(report.protocol_version == "step8-qbank-semantic-v1",
             "semantic_transition_source_version_invalid")
    _require(bool(_SHA256.fullmatch(approval_sha256)), "semantic_transition_approval_missing")
    token = _RESPONSE_TRANSITION.set((report, approval_sha256))
    try:
        yield
    finally:
        _RESPONSE_TRANSITION.reset(token)


async def _previous_response(pair, plan, owner, load_trace, store, model, campaign_id):
    """Reuse a conforming response, or link an explicit replacement to its old trace."""
    path = store.root / "_store/semantic-pairs" / (pair.pair_hash + ".transition.json")
    prior = _RESPONSE_TRANSITION.get()
    saved = json.loads(path.read_bytes()) if path.exists() else None
    if saved is None:
        if prior is None or prior[0].query_id != plan.query.query_id:
            return None, None
        report, approval_hash = prior
        _require(
            (report.snapshot_id, report.compiler_run_id, report.canonical_query_run_id,
             report.judge_model_hash) ==
            (plan.snapshot_id, plan.compiler_run_id, plan.canonical_query_run_id,
             plan.judge_model_hash), "semantic_transition_scope_changed")
        matches = [p for p in report.pairs if p.qbank_id == pair.qbank_id]
        _require(len(matches) == 1, "semantic_transition_pair_missing")
        previous = matches[0]
        protocol_hash = report.protocol_hash
        evaluation_run_id = report.evaluation_run_id
    else:
        _require(saved["campaign_id"] == campaign_id and saved["pair_hash"] == pair.pair_hash,
                 "semantic_transition_owner_changed")
        previous = SemanticPairResult.model_validate(saved["previous"])
        protocol_hash = saved["protocol_hash"]
        evaluation_run_id = saved["evaluation_run_id"]
        approval_hash = saved["approval_sha256"]
        _require(bool(_SHA256.fullmatch(approval_hash)), "semantic_transition_approval_missing")
    old_hash = canonical_content_hash({
        "protocol_hash": protocol_hash, "judge_model_hash": plan.judge_model_hash,
        "snapshot_id": plan.snapshot_id, "compiler_run_id": plan.compiler_run_id,
        "query_hash": canonical_content_hash(plan.query.model_dump(mode="json")),
        "qbank_id": pair.qbank_id, "question_hash": pair.question_hash,
        "support_hash": pair.support_hash,
        "messages": [m.model_dump(mode="json") for m in pair.request.messages],
    })
    _require(old_hash == previous.pair_hash and pair.question_hash == previous.question_hash
             and pair.support_hash == previous.support_hash,
             "semantic_transition_request_changed")
    old_format = copy.deepcopy(pair.request.response_format)
    old_format["json_schema"]["schema"] = QBankSemanticJudgeEnvelopeV2.model_json_schema()
    old_pair = replace(pair, pair_hash=old_hash, request=pair.request.model_copy(update={
        "request_id": "step8-semantic-" + old_hash,
        "response_format": old_format,
        "metadata": {"task": "step8_qbank_semantic", "protocol_version": "step8-qbank-semantic-v1",
                     "protocol_hash": protocol_hash, "semantic_pair_hash": old_hash},
    }))
    audit = current_campaign_audit()
    key = audit.identity(old_pair.call_id, _generation_manifest(old_pair.request),
                         plan.judge_model_hash)
    _require(audit.restore_result(old_pair.call_id, key) is not None,
             "semantic_transition_original_request_missing")
    trace = await load_trace(old_pair.call_id)
    if trace is None and saved is not None:
        trace = ModelCallTrace.model_validate(store.load_json(saved["trace_sha256"]))
    _require(trace is not None, "semantic_transition_original_trace_missing")
    _validate_trace(trace, old_pair, plan, owner.model_copy(update={
        "evaluation_run_id": evaluation_run_id}))
    _require(canonical_content_hash(trace.model_dump(mode="json")) == previous.trace_hash
             and trace.actual_model_id == model.expected_actual_model_id
             and not trace.retry_count
             and not trace.request_config.get("runtime_replica_failover_count"),
             "semantic_transition_trace_changed")
    result = _pair_result(old_pair, trace)
    _require(result == previous, "semantic_transition_result_changed")
    _require(result.status == "evaluated" or result.error_code == "semantic_dimensions_invalid",
             "semantic_transition_outcome_not_replaceable")
    if saved is None:
        write_json(path, {
            "campaign_id": campaign_id, "pair_hash": pair.pair_hash,
            "approval_sha256": approval_hash, "protocol_hash": protocol_hash,
            "evaluation_run_id": evaluation_run_id, "previous": previous.model_dump(mode="json"),
            "trace_sha256": store.put_json(trace.model_dump(mode="json")),
        }, immutable=True)
    provenance = {"approval_sha256": approval_hash, "previous_protocol_hash": protocol_hash,
                  "previous_pair_hash": old_hash, "previous_model_call_id": old_pair.call_id,
                  "previous_trace_hash": previous.trace_hash,
                  "action": "reuse_conforming" if result.status == "evaluated" else
                  "replace_dimensions_invalid"}
    if result.status == "evaluated":
        return result.model_copy(update={"pair_hash": pair.pair_hash,
                                         "response_transition": provenance}), provenance
    return None, provenance


@asynccontextmanager
async def _pair_lock(store: CampaignArtifactStore, pair_hash: str) -> AsyncIterator[None]:
    # A shared campaign PVC, not process-local locks, serializes backbone consumers.
    directory = store.root / "_store/semantic-pairs"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / (pair_hash + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class _SingleDispatchGateway:
    """Select a managed replica once without the runtime router's failover loop."""

    def __init__(self, gateway: GenerationGateway, before_dispatch: Callable[[], None]):
        self.gateway = gateway
        self.before_dispatch = before_dispatch

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        gateway = self.gateway
        if not isinstance(gateway, RoutedGenerationGateway):
            self.before_dispatch()
            return await gateway.generate(request)
        index = stable_replica_index(request.request_id, len(gateway.gateways))
        replica, service_id, selected = gateway.gateways[index]
        self.before_dispatch()
        result = await selected.generate(request)
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "runtime_service_id": service_id,
                    "replica_index": replica,
                    "replica_count": len(gateway.gateways),
                    "replica_failover_count": 0,
                }
            }
        )


def _validate_trace(trace, pair, plan, owner) -> None:
    _require(
        trace.model_call_id == pair.call_id
        and trace.data_scope is DataScope.EVALUATION
        and trace.stage is AppStage.JUDGE
        and trace.model_alias == JUDGE_ALIAS
        and trace.evaluation_run_id == owner.evaluation_run_id
        and trace.semantic_run_id is None
        and trace.run_id == owner.query_run_id
        and trace.snapshot_id == plan.snapshot_id
        and trace.query_id == plan.query.query_id
        and trace.qbank_id == pair.qbank_id
        and trace.request_config.get("request_metadata") == pair.request.metadata,
        "semantic_cached_trace_lineage_mismatch",
    )


def _pair_result(
    pair, trace: ModelCallTrace | None, error: str | None = None
) -> SemanticPairResult:
    base: dict[str, Any] = {
        "qbank_id": pair.qbank_id,
        "pair_hash": pair.pair_hash,
        "model_call_id": pair.call_id,
        "question_hash": pair.question_hash,
        "support_hash": pair.support_hash,
        "trace_hash": canonical_content_hash(trace.model_dump(mode="json")) if trace else None,
    }
    if error or trace is None or trace.status is not TraceStatus.SUCCESS:
        return SemanticPairResult(
            **base, status="incomplete", error_code=error or "semantic_call_failed"
        )
    parsed_output = dict(trace.parsed_output or {})
    contract = parsed_output.pop("step8_output_contract", {})
    error_code = (
        "semantic_output_repaired"
        if trace.repair_attempt_count
        else "semantic_output_truncated"
        if contract.get("finish_reason") == "length"
        else "semantic_finish_reason_invalid"
        if contract.get("finish_reason") != "stop"
        else "semantic_output_not_native"
        if contract.get("native_structured_output") is not True
        else "semantic_output_invalid"
        if trace.parse_error
        else None
    )
    if error_code:
        return SemanticPairResult(**base, status="incomplete", error_code=error_code)
    try:
        result = QBankSemanticJudgeEnvelopeV2.model_validate(parsed_output).result
        if result.pair_alias != "P01":
            raise ValueError("pair_alias")
        payload = json.loads(pair.request.messages[1].parts[0].text or "{}")
        binding_required = (
            payload["query_has_image"] and payload["candidate_evidence_image_count"] > 0
        )
        binding = result.dimensions.query_image_binding_compatible
        if (binding_required and binding == "not_applicable") or (
            not binding_required and binding != "not_applicable"
        ):
            raise ValueError("binding_dimension")
    except ValueError:
        return SemanticPairResult(
            **base, status="incomplete", error_code="semantic_dimensions_invalid"
        )
    reasons = tuple(reason.value for reason in semantic_reason_codes(result.dimensions))
    return SemanticPairResult(
        **base,
        status="evaluated",
        semantic_match=not reasons,
        dimensions=result.dimensions,
        reason_codes=reasons,
    )


def _parse_step8_output(result: GenerationResult, config: Step8SemanticConfig) -> ParsedModelOutput:
    """Persist diagnostics and cost even when a provider's JSON is inadmissible."""
    contract = {"finish_reason": result.finish_reason, "native_structured_output": False}
    try:
        QBankSemanticJudgeEnvelopeV2.model_validate_json(result.text, strict=True)
    except ValueError:
        native_schema_valid = False
    else:
        native_schema_valid = True
    try:
        parsed = parse_qbank_semantic_output(result.text, config=config.renderer)
    except Exception:
        return ParsedModelOutput(
            value={"step8_output_contract": contract}, initial_error="semantic_output_invalid"
        )
    contract["native_structured_output"] = (
        native_schema_valid
        and parsed.repair_attempt_count == 0
        and parsed.initial_error is None
        and result.metadata.get("native_structured_output") is not False
    )
    return ParsedModelOutput(
        value={**parsed.value, "step8_output_contract": contract},
        repair_attempt_count=parsed.repair_attempt_count,
        initial_error="semantic_output_repaired" if parsed.initial_error else None,
    )


async def evaluate_step8_semantic_plan(
    plan: Step8SemanticPlan,
    *,
    owner: SemanticEvaluationOwner,
    execution: ModelExecutionService,
    gateway: GenerationGateway,
    load_trace: Callable[[str], Awaitable[ModelCallTrace | None]],
    config: Step8SemanticConfig | None = None,
) -> Step8SemanticReport:
    """Run/recover the union once under an evaluator-owned canonical parent.

    ``load_trace`` must use the evaluator model-call repository. All consumers
    must share this parent, campaign store, and source fingerprint. A persisted
    dispatch intent without a trace/response is incomplete, not permission to
    repeat a possibly completed provider call. The workflow owns reconciliation.
    """
    config = config or Step8SemanticConfig()
    _require(execution.data_scope is DataScope.EVALUATION, "semantic_requires_evaluator_execution")
    _require(execution.max_attempts == 1, "semantic_requires_single_execution_attempt")
    _require(config.protocol_hash == plan.protocol_hash, "semantic_protocol_hash_mismatch")
    _require(
        owner == build_step8_semantic_owner(plan, canonical_query_run_id=owner.query_run_id),
        "semantic_owner_scope_mismatch",
    )
    model = execution.registry.get(JUDGE_ALIAS)
    model_hash = execution.registry.resolved_hash_for_aliases([JUDGE_ALIAS])
    _require(model_hash == plan.judge_model_hash, "semantic_judge_model_hash_mismatch")
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", model.model_revision)), "semantic_judge_unpinned")
    audit = current_campaign_audit()
    _require(audit is not None, "semantic_requires_campaign_journal")
    assert audit is not None
    store = audit.store

    async def judge_pair(pair: SemanticPair) -> SemanticPairResult:
        async with _pair_lock(store, pair.pair_hash):
            reused, transition = await _previous_response(
                pair, plan, owner, load_trace, store, model, audit.campaign_id)
            if reused is not None:
                return reused
            pair, replacement = await _explicit_replacement(pair, plan, owner, load_trace, audit)
            if replacement is not None:
                _require(transition is None, "semantic_replacement_transition_conflict")
                transition = replacement
            root = store.root / "_store/semantic-pairs"
            storage_key = pair.pair_hash if replacement is None else pair.call_id
            intent = root / (storage_key + ".dispatch.json")
            receipt = root / (storage_key + ".success.json")
            identity = {
                "pair_hash": pair.pair_hash,
                "evaluation_run_id": owner.evaluation_run_id,
                "canonical_query_run_id": owner.query_run_id,
                "protocol_hash": plan.protocol_hash,
                "source_fingerprint": audit.source_fingerprint,
            }
            if replacement is not None:
                identity["explicit_replacement"] = replacement
            if intent.exists():
                _require(
                    json.loads(intent.read_bytes()) == identity, "semantic_pair_owner_conflict"
                )
            trace = await load_trace(pair.call_id)
            if trace is None and receipt.exists():
                saved = json.loads(receipt.read_bytes())
                _require(saved["identity"] == identity, "semantic_receipt_owner_conflict")
                trace = ModelCallTrace.model_validate(store.load_json(saved["trace_sha256"]))
                _validate_trace(trace, pair, plan, owner)
                _require(trace.status is TraceStatus.SUCCESS, "semantic_receipt_not_successful")
                await execution.model_calls.add(trace)
            if trace is None:
                key = audit.identity(pair.call_id, _generation_manifest(pair.request), model_hash)
                journal = audit.restore_result(pair.call_id, key)
                if intent.exists() and journal is None:
                    return _pair_result(pair, None, "semantic_dispatch_outcome_unknown")
                try:
                    audited = await execution.generate_audited(
                        _SingleDispatchGateway(
                            gateway, lambda: write_json(intent, identity, immutable=True)
                        ),
                        pair.request,
                        ModelCallContext(
                            model_call_id=pair.call_id,
                            run_id=owner.query_run_id,
                            evaluation_run_id=owner.evaluation_run_id,
                            snapshot_id=plan.snapshot_id,
                            query_id=plan.query.query_id,
                            conversation_id=plan.query.conversation_id,
                            qbank_id=pair.qbank_id,
                            stage=AppStage.JUDGE,
                            parent_model_call_id=(replacement or {}).get("previous_model_call_id"),
                        ),
                        output_parser=lambda result: _parse_step8_output(result, config),
                    )
                    trace = audited.trace
                except Exception:
                    # Keep the durable intent: a failed/unknown call is not a semantic negative.
                    trace = await load_trace(pair.call_id)
                    if trace is None:
                        return _pair_result(pair, None, "semantic_audited_call_incomplete")
            _validate_trace(trace, pair, plan, owner)
            contract_error = (
                "semantic_actual_model_mismatch"
                if trace.status is TraceStatus.SUCCESS
                and trace.actual_model_id != model.expected_actual_model_id
                else "semantic_multiple_dispatches_recorded"
                if trace.retry_count or trace.request_config.get("runtime_replica_failover_count")
                else None
            )
            result = _pair_result(pair, trace, contract_error)
            if transition is not None:
                result = result.model_copy(update={"response_transition": transition})
            if result.status == "evaluated":
                write_json(
                    receipt,
                    {
                        "identity": identity,
                        "trace_sha256": store.put_json(trace.model_dump(mode="json")),
                    },
                    immutable=True,
                )
            return result

    # Only a bounded number of tasks exist; ModelExecutionService also acquires
    # the campaign-wide request semaphore before each actual provider dispatch.
    pending = iter(enumerate(plan.pairs))
    slots: list[SemanticPairResult | None] = [None] * len(plan.pairs)
    completed = 0

    async def worker() -> None:
        nonlocal completed
        for index, pair in pending:
            try:
                slots[index] = await judge_pair(pair)
            except Step8SemanticError as exc:
                print("STEP8_SEMANTIC " + json.dumps({
                    "state": "failed", "error_code": str(exc), "pair": pair.pair_hash,
                    "completed": completed, "total": len(plan.pairs),
                }), flush=True)
                raise
            completed += 1
            result = slots[index]
            if completed == 1 or completed % 25 == 0 or completed == len(plan.pairs) or (
                result.status == "incomplete"
            ):
                print("STEP8_SEMANTIC " + json.dumps({
                    "state": result.status, "completed": completed, "total": len(plan.pairs),
                    "error_code": result.error_code,
                    "response_action": (result.response_transition or {}).get("action", "current"),
                }), flush=True)

    tasks = [asyncio.create_task(worker()) for _ in range(min(config.concurrency, len(plan.pairs)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    results = [result for result in slots if result is not None]
    _require(len(results) == len(plan.pairs), "semantic_pair_denominator_incomplete")
    by_id = {r.qbank_id: r for r in results}

    def observation(ids: Sequence[str]) -> SemanticObservation:
        evaluated = [by_id[q] for q in ids if by_id[q].status == "evaluated"]
        complete = len(evaluated) == len(ids)
        return SemanticObservation(
            status="evaluated" if complete else "incomplete",
            value=int(any(r.semantic_match for r in evaluated)) if complete else None,
            pair_count=len(ids),
            evaluated_pair_count=len(evaluated),
        )

    return Step8SemanticReport(
        protocol_hash=plan.protocol_hash,
        plan_hash=plan.content_hash,
        evaluation_run_id=owner.evaluation_run_id,
        owner_query_run_id=owner.query_run_id,
        canonical_query_run_id=plan.canonical_query_run_id,
        compiler_run_id=plan.compiler_run_id,
        snapshot_id=plan.snapshot_id,
        query_id=plan.query.query_id,
        judge_model_hash=plan.judge_model_hash,
        status="evaluated" if all(r.status == "evaluated" for r in results) else "incomplete",
        pairs=tuple(results),
        cells=tuple(
            SemanticCellResult(
                pool=cell,
                semantic_cover=observation(cell.cover_qbank_ids),
                semantic_match_at_k={k: observation(cell.match_qbank_ids[:k]) for k in MATCH_KS},
            )
            for cell in plan.cells
        ),
    )
