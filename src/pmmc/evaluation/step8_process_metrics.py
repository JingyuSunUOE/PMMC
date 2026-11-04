"""Evaluator-only, model-free Step8 process observations.

Query denominators never depend on successful retrieval or answer generation.
Candidate-support proxies are explicitly separate from actual runtime evidence.
No gold text, evidence identifiers, or model payloads leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text, tuple_
from sqlalchemy.exc import SQLAlchemyError

from pmmc.compiler.ids import canonical_content_hash, question_content_hash
from pmmc.evaluation.evidence import (
    RankedEvidenceCandidate,
    matched_group_ids,
    matched_image_group_ids,
    resolve_gold_evidence,
)
from pmmc.evaluation.qbank_coverage import QBankCandidateEvidence, score_qbank_coverage
from pmmc.evidence_lineage import EvidenceLineageResolver
from pmmc.infrastructure.database import compiler_models as cm
from pmmc.infrastructure.database import query_executor_models as qm
from pmmc.infrastructure.database import questioner_models as qquestioner
from pmmc.infrastructure.database.compiler_mappers import question_from_rows
from pmmc.infrastructure.database.compiler_repositories import (
    PostgresProgramExecutionRepository,
    _model_mapping,
)
from pmmc.infrastructure.database.doubter_repositories import PostgresDoubterRepository
from pmmc.infrastructure.database.models import BenchmarkSessionModel
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.schemas import (
    EvidenceMatchRole,
    EvidenceMemberType,
    MemoryUnit,
    QBankCoverageCandidate,
    QueryGold,
    QueryInput,
    QuestionCandidate,
    RankingScore,
    ResolvedGoldEvidenceGroup,
)
from pmmc.schemas.doubter import QBankRoundProjectionManifest, QBankRoundProjectionMember
from pmmc.schemas.query_executor import (
    QBankMatchBatch,
    QBankMatchFusedCandidate,
    QueryExperimentCellResult,
    runtime_fact_payload,
)
from pmmc.schemas.step8_process import (
    MATCH_KS,
    QBANK,
    RETRIEVED,
    TECHNICAL,
)
from pmmc.schemas.step8_process import (
    PROCESS_METRIC_NAMES as PROCESS_METRIC_NAMES,
)
from pmmc.schemas.step8_process import (
    ProcessCellBinding as ProcessCellBinding,
)
from pmmc.schemas.step8_process import (
    ProcessCellMetrics as ProcessCellMetrics,
)
from pmmc.schemas.step8_process import (
    ProcessMetric as ProcessMetric,
)
from pmmc.schemas.step8_process import (
    ProcessMetricsReport as ProcessMetricsReport,
)

VERSION = "step8-process-metrics-v1"
# The deterministic scorer's last registered hardened semantics. Step8 supplies
# its own exact runtime ranking and round bank, not the legacy coverage workflow.
PROXY_SCORER_VERSION = "qbank-coverage-v6"


@dataclass(frozen=True)
class MatchFacts:
    batch: QBankMatchBatch
    candidates: tuple[QBankMatchFusedCandidate, ...]
    channels: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProcessFacts:
    load_error: str | None = None
    run: dict[str, Any] | None = None
    source_run: dict[str, Any] | None = None
    source_match_cells: tuple[dict[str, Any], ...] = ()
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    executions: dict[str, ProgramExecutionResult] = field(default_factory=dict)
    program_step_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)
    matches: dict[str, MatchFacts] = field(default_factory=dict)
    rounds: dict[int, tuple[QBankRoundProjectionManifest, list[QBankRoundProjectionMember]]] = (
        field(default_factory=dict)
    )
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)
    support: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visible_sessions: frozenset[str] = frozenset()


class _Unavailable(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _Unavailable(reason)


def _missing(reason: str, status="incomplete", *, denominator=1, kind="query") -> ProcessMetric:
    return ProcessMetric(
        status=status, reason=reason, denominator=denominator, denominator_kind=kind
    )


def _ratio(numerator: int, denominator: int = 1, kind="query") -> ProcessMetric:
    return ProcessMetric(
        status="evaluated",
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        denominator_kind=kind,
    )


def _bound_fact(binding, query, snapshot_id, query_run_id, facts) -> QueryExperimentCellResult:
    _require(facts.load_error is None, facts.load_error or "process_facts_unavailable")
    run = facts.run
    _require(run is not None, "query_run_missing")
    _require(
        run["query_executor_run_id"] == query_run_id
        and run["snapshot_id"] == snapshot_id
        and run["dataset"] == query.dataset.value
        and run["schema_version"] == "0.3.0"
        and binding.source_run_id == query_run_id,
        "query_run_scope_mismatch",
    )
    row = facts.cells.get(binding.source_fact_id)
    _require(row is not None, "cell_fact_missing")
    fact = QueryExperimentCellResult.model_validate(row["payload"])
    digest = canonical_content_hash(
        runtime_fact_payload(
            fact.model_dump(mode="json", exclude={"cell_result_id", "content_hash", "created_at"})
        )
    )
    _require(
        digest == fact.content_hash == row["content_hash"] == binding.source_fact_hash
        and fact.cell_result_id == row["cell_result_id"] == binding.source_fact_id
        and fact.query_executor_run_id == row["query_executor_run_id"] == query_run_id
        and fact.query_id == row["query_id"] == query.query_id
        and fact.cell_id == row["cell_id"]
        and fact.cell.question_budget == binding.question_budget
        and fact.cell.revision_depth == binding.round_id,
        "cell_fact_identity_mismatch",
    )
    _require(
        query.available_after_session in facts.visible_sessions,
        "query_cutoff_lineage_missing",
    )
    process_run_id = query_run_id
    source_id = run.get("metadata", {}).get("source_match_run_id")
    if source_id is not None:
        source_run = facts.source_run
        _require(source_id != query_run_id and source_run is not None, "source_match_run_missing")
        _require(
            source_run["query_executor_run_id"] == source_id
            and source_run["schema_version"] == "0.3.0"
            and all(
                source_run[key] == run[key]
                for key in (
                    "snapshot_id",
                    "dataset",
                    "source_compiler_run_id",
                    "routing_projection_id",
                    "config_hash",
                )
            )
            and not source_run.get("metadata", {}).get("source_match_run_id")
            and fact.metadata.get("source_match_run_id") == source_id,
            "source_match_run_scope_mismatch",
        )
        source_cells = [
            raw for raw in facts.source_match_cells if raw.get("cell_id") == fact.cell_id
        ]
        _require(len(source_cells) == 1, "source_match_cell_missing_or_duplicate")
        source = QueryExperimentCellResult.model_validate(source_cells[0])
        _require(
            source.query_executor_run_id == source_id
            and source.query_id == query.query_id
            and source.content_hash
            == canonical_content_hash(
                runtime_fact_payload(
                    source.model_dump(
                        mode="json", exclude={"cell_result_id", "content_hash", "created_at"}
                    )
                )
            )
            and all(
                runtime_fact_payload(getattr(source, key))
                == runtime_fact_payload(getattr(fact, key))
                for key in (
                    "cell",
                    "route",
                    "binding",
                    "execution_id",
                    "sufficiency",
                    "relevance",
                    "relevance_items",
                )
            ),
            "source_match_cell_identity_mismatch",
        )
        process_run_id = source_id
    else:
        _require(not fact.metadata.get("source_match_run_id"), "unbound_source_match_run")
    _require(
        fact.route.query_executor_run_id == process_run_id
        and fact.route.query_id == query.query_id
        and (
            fact.binding is not None
            or (
                fact.execution_id is None
                and fact.sufficiency is None
                and fact.error_type is None
                and fact.route.outcome.value != "program"
            )
        )
        and (
            fact.binding is None
            or (
                fact.binding.query_executor_run_id == process_run_id
                and fact.binding.query_id == query.query_id
                and fact.binding.round_id == binding.round_id
                and fact.binding.routing_projection_id == run["routing_projection_id"]
            )
        ),
        "cell_process_scope_mismatch",
    )
    return fact


def _candidate(ref: dict, resolver, memories, visible_sessions, rank=1):
    typed = {key: ref.get(key) for key in ("memory_id", "image_id", "caption_id", "session_id")}
    _require(sum(value is not None for value in typed.values()) == 1, "evidence_reference_invalid")
    _require(
        ref["evidence_id"] in set(typed.values()) - {None}, "evidence_reference_identity_mismatch"
    )
    memory_ids = {m.memory_id for m in memories}
    image_ids = {i for m in memories for i in (*m.image_ids, *(r.image_id for r in m.image_refs))}
    captions = {m.metadata.get("caption_id") for m in memories} - {None}
    for key, allowed in (
        ("memory_id", memory_ids),
        ("image_id", image_ids),
        ("caption_id", captions),
        ("session_id", visible_sessions),
    ):
        _require(typed[key] is None or typed[key] in allowed, "evidence_outside_visible_lineage")
    closure = resolver.expand(
        memory_ids=[typed["memory_id"]] if typed["memory_id"] else (),
        image_ids=[typed["image_id"]] if typed["image_id"] else (),
        caption_ids=[typed["caption_id"]] if typed["caption_id"] else (),
        session_ids=[typed["session_id"]] if typed["session_id"] else (),
    )
    # Never expand a memory/session downward to images or sibling memories.
    return RankedEvidenceCandidate(
        candidate_id=str(ref["evidence_id"]),
        rank=rank,
        memory_ids=closure.memory_ids,
        session_ids=closure.session_ids,
        image_ids=closure.image_ids,
        evidence_ids=closure.evidence_group_ids,
    )


def _runtime_execution(fact, facts, query, snapshot_id):
    result = facts.executions.get(fact.execution_id)
    _require(result is not None and fact.binding is not None, "runtime_execution_missing")
    trace = result.trace
    _require(
        trace.execution_id == fact.execution_id
        and trace.run_id == fact.route.query_executor_run_id
        and trace.query_id == query.query_id
        and trace.snapshot_id == snapshot_id
        and trace.purpose.value == "query_time"
        and trace.source_compiler_run_id == facts.run["source_compiler_run_id"]
        and trace.plan_id == fact.binding.plan_id,
        "runtime_execution_scope_mismatch",
    )
    _require(bool(result.steps), "runtime_execution_steps_missing")
    expected_steps = facts.program_step_ids.get(trace.plan_id)
    observed_steps = tuple(step.step_id for step in result.steps)
    _require(
        bool(expected_steps)
        and observed_steps == expected_steps[: len(observed_steps)]
        and (
            trace.status.value != "success"
            or trace.control_action.value != "continue"
            or observed_steps == expected_steps
        ),
        "runtime_execution_steps_incomplete",
    )
    _require(
        (trace.status.value == "success") == all(step.success for step in result.steps),
        "runtime_execution_status_mismatch",
    )
    expected = set()
    for step in result.steps:
        _require(
            step.execution_id == trace.execution_id
            and step.run_id == trace.run_id
            and step.snapshot_id == snapshot_id
            and step.query_id == query.query_id
            and step.plan_id == trace.plan_id
            and canonical_content_hash(step.output) == step.output_hash,
            "execution_step_hash_or_scope_mismatch",
        )
        if step.output.get("error_type") != "EvidenceLimitExceeded":
            for key in ("memory", "image", "caption", "session"):
                expected.update(
                    (key, value) for value in step.output.get(f"retrieved_{key}_ids", [])
                )
    actual = {
        (key, getattr(ref, key + "_id"))
        for ref in result.evidence
        for key in ("memory", "image", "caption", "session")
        if getattr(ref, key + "_id") is not None
    }
    _require(expected == actual, "runtime_evidence_refs_incomplete")
    _require(
        sorted(ref.rank for ref in result.evidence) == list(range(1, len(result.evidence) + 1))
        and all(
            ref.execution_id == trace.execution_id and ref.snapshot_id == snapshot_id
            for ref in result.evidence
        ),
        "runtime_evidence_rank_or_scope_mismatch",
    )
    return result


def _retrieved(fact, facts, query, snapshot_id, groups, resolver, memories):
    result = _runtime_execution(fact, facts, query, snapshot_id)
    candidates = [
        _candidate(
            ref.model_dump(mode="json"), resolver, memories, facts.visible_sessions, ref.rank
        )
        for ref in result.evidence
    ]
    matched = set().union(*(matched_group_ids(c, groups, resolver) for c in candidates))
    image_groups = {
        g.gold_group_id
        for g in groups
        for m in g.members
        if m.member_type is EvidenceMemberType.IMAGE
        and m.match_role is EvidenceMatchRole.DISCRIMINATIVE
    }
    image_matches = set().union(*(matched_image_group_ids(c, groups) for c in candidates))
    return {
        "retrieved_evidence_recall": _ratio(len(matched), len(groups), "gold_groups"),
        "retrieved_group_complete": _ratio(int(len(matched) == len(groups))),
        "retrieved_image_hit": _ratio(int(bool(image_matches)))
        if image_groups
        else _missing("gold_image_evidence_unavailable", "not_applicable", denominator=0),
    }


def _sufficiency_execution(fact, execution):
    """V3 sufficiency describes the saved relevance subset, not raw retrieval."""
    from pmmc.runtime.query_executor_config import JOINT_POLICY
    from pmmc.runtime.relevance_grid import (
        RELEVANCE_GRID_PROTOCOL,
        select_execution_evidence,
        validate_joint_outcome,
    )

    assessment = fact.sufficiency
    points = fact.metadata.get("relevance_grid", [])
    point_id = assessment.metadata.get("relevance_grid_point_id") if assessment else None
    if not point_id and not points:
        return execution, {"sufficiency_evidence_basis": "bound_query_time_execution"}
    _require(
        assessment is not None
        and assessment.metadata.get("sufficiency_policy_version") == "query-runtime-evidence-v3",
        "sufficiency_projection_policy_mismatch",
    )
    _require(
        isinstance(points, list) and all(isinstance(p, dict) for p in points) and bool(point_id),
        "sufficiency_projection_missing",
    )
    selected = [
        p for p in points if p.get("source_grid_point_id", p.get("grid_point_id")) == point_id
    ]
    _require(len(selected) == 1, "sufficiency_projection_missing")
    point = selected[0]
    thresholds = assessment.metadata.get("channel_thresholds", {})
    hashes = point.get("selected_reference_hashes")
    _require(
        point.get("protocol") == RELEVANCE_GRID_PROTOCOL
        and point.get("query_executor_run_id") == fact.query_executor_run_id
        and point.get("source_query_executor_run_id", fact.query_executor_run_id)
        == execution.trace.run_id
        and point.get("query_id") == fact.query_id
        and point.get("execution_id") == execution.trace.execution_id
        and point.get("binding_id") == fact.binding.query_program_binding_id
        and thresholds.get("text") is not None
        and thresholds.get("multimodal") is not None
        and point.get("text_threshold") == thresholds["text"]
        and point.get("multimodal_threshold") == thresholds["multimodal"]
        and isinstance(hashes, list)
        and all(isinstance(h, str) for h in hashes)
        and point.get("evidence_subset_hash") == canonical_content_hash(hashes)
        and point.get("score_manifest_hash")
        == canonical_content_hash(sorted(item.content_hash for item in fact.relevance_items)),
        "sufficiency_projection_identity_mismatch",
    )
    projected = select_execution_evidence(execution, hashes)
    joint_details = {}
    if point.get("joint_policy") == JOINT_POLICY:
        validate_joint_outcome(point, fact.metadata.get("joint_contexts", {}))
        joint_details = {
            "joint_policy": JOINT_POLICY,
            "arrived_joint_image_hashes": point["arrived_joint_image_hashes"],
            "selected_raw_image_reached": assessment.raw_image_reached,
        }
    return projected, {
        "sufficiency_evidence_basis": "saved_runtime_relevance_projection",
        "sufficiency_evidence_subset_hash": point["evidence_subset_hash"],
        **joint_details,
    }


def _qbank(fact, facts, query, snapshot_id, groups, resolver, memories):
    match = facts.matches.get(fact.route.match_batch_id)
    _require(match is not None, "runtime_match_batch_missing")
    batch = match.batch
    budget, round_id = fact.cell.question_budget, fact.cell.revision_depth
    _require(
        (
            batch.query_executor_run_id,
            batch.query_id,
            batch.snapshot_id,
            batch.conversation_id,
            batch.cutoff_session_id,
            batch.question_budget,
            batch.active_round,
        )
        == (
            fact.route.query_executor_run_id,
            query.query_id,
            snapshot_id,
            query.conversation_id,
            query.available_after_session,
            budget,
            round_id,
        )
        and batch.source_compiler_run_id == facts.run["source_compiler_run_id"]
        and batch.routing_projection_id == facts.run["routing_projection_id"],
        "runtime_match_scope_mismatch",
    )
    classes = {"answerable": 0, "control": 1}
    candidates = sorted(match.candidates, key=lambda c: (classes[c.route_class.value], c.rank))
    channels = sorted(
        match.channels, key=lambda c: (c["channel"], classes[c["route_class"]], c["rank"])
    )
    _require(
        len(candidates) == batch.fused_count
        and canonical_content_hash(
            {
                "run_id": batch.query_executor_run_id,
                "query_id": batch.query_id,
                "projection_id": batch.routing_projection_id,
                "question_budget": budget,
                "round_id": round_id,
                "channel_hashes": [c["content_hash"] for c in channels],
                "fused_hashes": [c.content_hash for c in candidates],
            }
        )
        == batch.content_hash,
        "runtime_match_denominator_or_hash_mismatch",
    )
    projection = facts.rounds.get(round_id)
    _require(projection is not None, "full_bank_round_missing")
    manifest, members = projection
    _require(
        manifest.run_id == batch.source_compiler_run_id
        and manifest.snapshot_id == snapshot_id
        and manifest.round_id == round_id
        and manifest.schema_version == "0.4.0"
        and manifest.pending_count == 0
        and manifest.member_count == len(members)
        and manifest.accepted_count
        == sum(m.outcome in {"accepted", "carried_forward"} for m in members)
        and manifest.rejected_count
        == sum(m.outcome in {"rejected", "max_rounds_exhausted"} for m in members)
        and len({m.qbank_id for m in members}) == len(members)
        and manifest.members_hash
        == canonical_content_hash(
            [m.model_dump(mode="json") for m in sorted(members, key=lambda m: m.qbank_id)]
        )
        and manifest.content_hash
        == canonical_content_hash(
            manifest.model_dump(
                mode="json", exclude={"schema_version", "content_hash", "created_at", "metadata"}
            )
        )
        and {m.qbank_id for m in members} == set(facts.questions),
        "full_bank_round_identity_or_denominator_mismatch",
    )
    eligible = {}
    for member in members:
        _require(
            member.run_id == manifest.run_id and member.projection_id == manifest.projection_id,
            "full_bank_member_scope_mismatch",
        )
        question = facts.questions[member.qbank_id]
        candidate_question = QuestionCandidate.model_validate(question)
        _require(
            question["run_id"] == manifest.run_id and question["snapshot_id"] == snapshot_id,
            "full_bank_question_scope_mismatch",
        )
        _require(
            question_content_hash(candidate_question)
            == candidate_question.candidate_content_hash
            == candidate_question.question_hash,
            "full_bank_question_content_mismatch",
        )
        if (
            member.outcome in {"accepted", "carried_forward"}
            and member.route_class == "answerable"
            and question["conversation_id"] == query.conversation_id
            and question["created_after_session"] in facts.visible_sessions
            and question["generation_rank"] <= budget
        ):
            eligible[member.qbank_id] = member
    evidence = {}
    for qbank_id in eligible:
        refs = facts.support.get(qbank_id)
        _require(bool(refs), "full_bank_candidate_support_missing")
        _require(
            [ref["evidence_id"] for ref in sorted(refs, key=lambda r: r["ordinal"])]
            == facts.questions[qbank_id]["evidence_ids"],
            "full_bank_candidate_support_incomplete",
        )
        resolved = []
        for ref in refs:
            _require(
                ref["qbank_id"] == qbank_id
                and ref["run_id"] == manifest.run_id
                and ref["snapshot_id"] == snapshot_id
                and ref["role"] == "expected",
                "candidate_support_scope_mismatch",
            )
            resolved.append(_candidate(ref, resolver, memories, facts.visible_sessions))
        evidence[qbank_id] = QBankCandidateEvidence(
            qbank_id=qbank_id,
            **{
                key: frozenset().union(*(getattr(c, key) for c in resolved))
                for key in ("memory_ids", "session_ids", "image_ids", "evidence_ids")
            },
        )
    ranked = []
    for candidate in candidates:
        _require(
            candidate.match_batch_id == batch.match_batch_id
            and candidate.query_id == query.query_id,
            "runtime_candidate_scope_mismatch",
        )
        original_payload = {
            "query_executor_run_id": batch.query_executor_run_id,
            "query_id": batch.query_id,
            "question_budget": budget,
            "round_id": round_id,
            "route_class": candidate.route_class.value,
            "rank": candidate.rank,
            "qbank_id": candidate.qbank_id,
            "raw_rrf": candidate.raw_rrf,
        }
        _require(
            candidate.content_hash
            in {
                canonical_content_hash(original_payload),
                canonical_content_hash(
                    candidate.model_dump(mode="json", exclude={"content_hash", "metadata"})
                ),
            },
            "runtime_candidate_hash_mismatch",
        )
        if candidate.route_class != "answerable":
            continue
        _require(
            candidate.qbank_id in eligible
            and candidate.plan_id == eligible[candidate.qbank_id].effective_plan_id,
            "runtime_candidate_outside_exact_round_bank",
        )
        ranked.append(
            QBankCoverageCandidate(
                coverage_candidate_id=candidate.fused_candidate_id,
                coverage_run_id=batch.match_batch_id,
                query_id=query.query_id,
                qbank_id=candidate.qbank_id,
                question_budget=budget,
                rank=candidate.rank,
                final_score=RankingScore(
                    score_type="saved_runtime_confidence", raw_value=candidate.confidence
                ),
            )
        )
    _require(
        [c.rank for c in ranked] == list(range(1, len(ranked) + 1))
        and len({c.qbank_id for c in ranked}) == len(ranked),
        "runtime_candidate_ranks_invalid",
    )
    bundle = score_qbank_coverage(
        coverage_run_id=batch.match_batch_id,
        query_id=query.query_id,
        ranked_by_budget={budget: ranked},
        evidence_by_qbank=evidence,
        oracle_qbank_ids_by_budget={budget: sorted(eligible)},
        gold_groups=groups,
        match_ks=list(MATCH_KS),
        lineage_resolver=resolver,
        protocol_version=PROXY_SCORER_VERSION,
    )
    values = {(m.metric_name, m.k): m.value for m in bundle.metrics}
    metrics = {
        f"qbank_proxy_{kind}_at_{k}": _ratio(int(values[f"qbank_evidence_proxy_{kind}_at_k", k]))
        for k in MATCH_KS
        for kind in ("cover", "match")
    }
    metrics.update(
        {
            f"qbank_full_bank_proxy_{kind}": _ratio(
                int(values[f"oracle_qbank_evidence_proxy_{kind}", None])
            )
            for kind in ("cover", "match")
        }
    )
    return metrics, {
        "match_batch_hash": batch.content_hash,
        "round_members_hash": manifest.members_hash,
        "eligible_bank_count": len(eligible),
        "ranked_answerable_count": len(ranked),
        "candidate_support_hash": canonical_content_hash(
            {
                key: sorted(facts.support[key], key=lambda r: r["ordinal"])
                for key in sorted(eligible)
            }
        ),
        "qbank_evidence_basis": "frozen_candidate_expected_support_not_runtime_retrieval",
        "proxy_scorer_version": PROXY_SCORER_VERSION,
    }


def compute_process_metrics(
    *,
    query: QueryInput,
    snapshot_id: str,
    query_run_id: str,
    cells: tuple[ProcessCellBinding, ...],
    facts: ProcessFacts,
    gold: QueryGold | None,
    memories: list[MemoryUnit],
) -> ProcessMetricsReport:
    """Pure scoring boundary. Absent facts never become negative observations."""
    if not cells or len({c.cell_id for c in cells}) != len(cells):
        raise ValueError("process_selected_cell_denominator_invalid")
    resolver = EvidenceLineageResolver(memories)
    groups: tuple[ResolvedGoldEvidenceGroup, ...] = ()
    if gold is not None:
        groups = resolve_gold_evidence(
            evaluation_run_id=query_run_id,
            snapshot_id=snapshot_id,
            query_id=query.query_id,
            dataset=query.dataset,
            gold=gold,
            memories=memories,
            policy_version="evidence-resolution-v2",
        )
    result = []
    for binding in cells:
        metrics = {name: _missing("cell_fact_missing") for name in (*TECHNICAL, *RETRIEVED, *QBANK)}
        provenance: dict[str, Any] = {"retrieved_evidence_basis": "bound_query_time_execution"}
        provenance["gold_lineage_hash"] = canonical_content_hash(
            {
                "groups": [g.model_dump(mode="json", exclude={"created_at"}) for g in groups],
                "memories": [
                    m.model_dump(mode="json") for m in sorted(memories, key=lambda m: m.memory_id)
                ],
            }
        )
        if binding.method != "pmmc":
            provenance["retrieved_evidence_basis"] = (
                "bound_answer_request_included_evidence"
                if binding.method in {"full_context_captions", "full_context_raw_images"}
                else "bound_retrieval_batch_before_answer_budget"
            )
            metrics = {
                name: _missing("pmmc_process_only", "not_applicable", denominator=0)
                for name in (*TECHNICAL, *QBANK)
            }
            metrics.update(
                {
                    name: _missing("baseline_retrieval_adapter_not_evaluated", "not_evaluated")
                    for name in RETRIEVED
                }
            )
        else:
            try:
                fact = _bound_fact(binding, query, snapshot_id, query_run_id, facts)
                _require(
                    all(
                        m.dataset == query.dataset
                        and m.conversation_id == query.conversation_id
                        and m.session_id in facts.visible_sessions
                        for m in memories
                    ),
                    "gold_lineage_scope_mismatch",
                )
            except (ValueError, KeyError, TypeError) as exc:
                reason = str(exc) if isinstance(exc, _Unavailable) else "cell_fact_invalid"
                metrics = {name: _missing(reason) for name in metrics}
            else:
                metrics["fallback_triggered"] = _ratio(int(fact.active_fallback))
                assessment = fact.sufficiency
                if fact.binding is None:
                    for name in TECHNICAL[:-1]:
                        metrics[name] = _missing(
                            "program_not_executed", "not_applicable", denominator=0
                        )
                elif facts.executions.get(fact.execution_id or "") is None:
                    for name in TECHNICAL[:-1]:
                        metrics[name] = _missing("runtime_execution_missing")
                elif assessment is not None:
                    valid = (
                        assessment.query_executor_run_id == fact.route.query_executor_run_id
                        and assessment.query_id == query.query_id
                        and assessment.execution_id == fact.execution_id
                        and fact.binding is not None
                        and assessment.query_program_binding_id
                        == fact.binding.query_program_binding_id
                    )
                    for name, value in zip(
                        TECHNICAL[:-1],
                        (
                            assessment.technical_success,
                            assessment.nonempty_evidence,
                            assessment.structural_requirement_success,
                            assessment.query_evidence_relevance,
                            assessment.raw_image_reached,
                        ),
                        strict=True,
                    ):
                        metrics[name] = (
                            _ratio(int(value))
                            if valid and value is not None
                            else _missing("sufficiency_fact_incomplete")
                        )
                else:
                    for name in TECHNICAL[:-1]:
                        metrics[name] = _missing("sufficiency_missing")
                if fact.binding is not None:
                    try:
                        execution = _runtime_execution(fact, facts, query, snapshot_id)
                        projected, details = _sufficiency_execution(fact, execution)
                        _require(
                            assessment is not None
                            and assessment.technical_success
                            == (projected.trace.status.value == "success")
                            and assessment.nonempty_evidence == bool(projected.evidence),
                            "sufficiency_execution_mismatch",
                        )
                        provenance.update(details)
                        if "arrived_joint_image_hashes" in details:
                            metrics["raw_image_reached"] = _ratio(
                                int(bool(details["arrived_joint_image_hashes"]))
                            )
                        provenance["execution_facts_hash"] = canonical_content_hash(
                            runtime_fact_payload(
                                {
                                    "trace": execution.trace.model_dump(mode="json"),
                                    "steps": [s.model_dump(mode="json") for s in execution.steps],
                                    "evidence": [
                                        e.model_dump(mode="json") for e in execution.evidence
                                    ],
                                }
                            )
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        reason = (
                            str(exc) if isinstance(exc, _Unavailable) else "execution_fact_invalid"
                        )
                        metrics.update({name: _missing(reason) for name in TECHNICAL[:-1]})
                if not groups:
                    for name in (*RETRIEVED, *QBANK):
                        metrics[name] = _missing(
                            "gold_evidence_unresolvable", "not_evaluated", denominator=None
                        )
                else:
                    for names, scorer in ((RETRIEVED, _retrieved), (QBANK, _qbank)):
                        if names == RETRIEVED and fact.binding is None:
                            metrics.update(
                                {
                                    name: _missing(
                                        "program_not_executed", "not_applicable", denominator=0
                                    )
                                    for name in names
                                }
                            )
                            continue
                        try:
                            scored = scorer(
                                fact, facts, query, snapshot_id, groups, resolver, memories
                            )
                            if isinstance(scored, tuple):
                                scored, details = scored
                                provenance.update(details)
                            metrics.update(scored)
                        except (ValueError, KeyError, TypeError) as exc:
                            reason = (
                                str(exc)
                                if isinstance(exc, _Unavailable)
                                else "process_fact_invalid"
                            )
                            metrics.update({name: _missing(reason) for name in names})
        recall = metrics["retrieved_evidence_recall"]
        if recall.status != "evaluated":
            metrics["retrieved_evidence_recall"] = recall.model_copy(
                update={
                    "denominator_kind": "gold_groups",
                    "denominator": 0 if recall.status == "not_applicable" else len(groups) or None,
                }
            )
        result.append(
            ProcessCellMetrics(
                **binding.model_dump(),
                query_id=query.query_id,
                metrics=metrics,
                provenance=provenance,
            )
        )
    return ProcessMetricsReport(
        status="incomplete"
        if any(
            m.status in {"incomplete", "not_evaluated"} for c in result for m in c.metrics.values()
        )
        else "complete",
        cells=result,
    )


def _eligible_question_statement(compiler_run_id: str, snapshot_id: str):
    """Independent compiler denominator, never inferred from projection membership."""
    question = cm.QBankQuestionModel
    unit = cm.CompilerUnitModel
    gate = qquestioner.QuestionerCandidateGateResultModel
    verification = qquestioner.QuestionerCandidateVerificationModel
    return select(question).where(
        question.run_id == compiler_run_id,
        question.snapshot_id == snapshot_id,
        select(unit.compiler_unit_id)
        .where(
            unit.compiler_unit_id == question.compiler_unit_id,
            unit.run_id == question.run_id,
            unit.snapshot_id == question.snapshot_id,
            unit.conversation_id == question.conversation_id,
        )
        .exists(),
        select(gate.gate_result_id)
        .where(
            gate.qbank_id == question.qbank_id,
            gate.run_id == question.run_id,
            gate.compiler_unit_id == question.compiler_unit_id,
            gate.status == "eligible_for_planning",
        )
        .exists(),
        select(verification.candidate_verification_id)
        .where(
            verification.qbank_id == question.qbank_id,
            verification.run_id == question.run_id,
            verification.compiler_unit_id == question.compiler_unit_id,
            verification.verification_round == question.candidate_version,
            verification.verdict == "pass",
        )
        .exists(),
    )


async def load_process_facts(sessions, *, query: QueryInput, query_run_id: str) -> ProcessFacts:
    """Read authoritative facts in a separate read-only evaluator transaction."""
    async with sessions() as session, session.begin():
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        run = await session.get(qm.QueryExecutorRunModel, query_run_id)
        if run is None:
            return ProcessFacts()
        source_run = None
        source_match_cells = ()
        match_run_id = query_run_id
        source_id = run.metadata_json.get("source_match_run_id")
        if source_id is not None:
            source_run = await session.get(qm.QueryExecutorRunModel, source_id)
            _require(
                source_run is not None
                and source_id != query_run_id
                and source_run.schema_version == "0.3.0"
                and all(
                    getattr(source_run, key) == getattr(run, key)
                    for key in (
                        "snapshot_id",
                        "dataset",
                        "source_compiler_run_id",
                        "routing_projection_id",
                        "config_hash",
                    )
                )
                and not source_run.metadata_json.get("source_match_run_id"),
                "source_match_run_scope_mismatch",
            )
            checkpoint = await session.scalar(
                select(qm.QueryRuntimePhaseStateModel).where(
                    qm.QueryRuntimePhaseStateModel.query_executor_run_id == source_id,
                    qm.QueryRuntimePhaseStateModel.query_id == query.query_id,
                    qm.QueryRuntimePhaseStateModel.phase == "match",
                )
            )
            _require(
                checkpoint is not None
                and canonical_content_hash(checkpoint.payload) == checkpoint.content_hash,
                "source_match_checkpoint_missing_or_invalid",
            )
            source_match_cells = tuple(checkpoint.payload["cells"])
            match_run_id = source_id

        async def rows(model, *where):
            return list((await session.scalars(select(model).where(*where))).all())

        cells = await rows(
            qm.QueryExperimentCellResultModel,
            qm.QueryExperimentCellResultModel.query_executor_run_id == query_run_id,
            qm.QueryExperimentCellResultModel.query_id == query.query_id,
        )
        batches = await rows(
            qm.QBankMatchBatchModel,
            qm.QBankMatchBatchModel.query_executor_run_id == match_run_id,
            qm.QBankMatchBatchModel.query_id == query.query_id,
        )
        matches = {}
        for batch in batches:
            candidates = await rows(
                qm.QBankMatchFusedCandidateModel,
                qm.QBankMatchFusedCandidateModel.match_batch_id == batch.match_batch_id,
            )
            channels = await rows(
                qm.QBankMatchChannelCandidateModel,
                qm.QBankMatchChannelCandidateModel.match_batch_id == batch.match_batch_id,
            )
            matches[batch.match_batch_id] = MatchFacts(
                QBankMatchBatch.model_validate(_model_mapping(batch)),
                tuple(
                    QBankMatchFusedCandidate.model_validate(_model_mapping(c)) for c in candidates
                ),
                tuple(_model_mapping(c) for c in channels),
            )
        repo = PostgresProgramExecutionRepository(session)
        executions = {}
        for execution_id in {row.payload.get("execution_id") for row in cells} - {None}:
            execution = await repo.get(execution_id)
            if execution is not None:
                executions[execution_id] = execution
        plan_steps = await rows(
            cm.ProgramStepModel,
            cm.ProgramStepModel.plan_id.in_({e.trace.plan_id for e in executions.values()}),
        )
        program_step_ids = {
            plan: tuple(sorted(s.step_id for s in plan_steps if s.plan_id == plan))
            for plan in {e.trace.plan_id for e in executions.values()}
        }
        rounds = {}
        doubter = PostgresDoubterRepository(session)
        for round_id in {row.revision_depth for row in cells}:
            projection = await doubter.get_round_projection(run.source_compiler_run_id, round_id)
            if projection is not None:
                rounds[round_id] = projection
        questions = list(
            (
                await session.scalars(
                    _eligible_question_statement(run.source_compiler_run_id, run.snapshot_id)
                )
            ).all()
        )
        eligible_keys = [(q.qbank_id, q.compiler_unit_id) for q in questions]
        refs = await rows(
            cm.QBankEvidenceRefModel,
            cm.QBankEvidenceRefModel.run_id == run.source_compiler_run_id,
            cm.QBankEvidenceRefModel.snapshot_id == run.snapshot_id,
            tuple_(
                cm.QBankEvidenceRefModel.qbank_id, cm.QBankEvidenceRefModel.compiler_unit_id
            ).in_(eligible_keys),
        )
        paraphrases = await rows(
            cm.QBankParaphraseModel,
            cm.QBankParaphraseModel.run_id == run.source_compiler_run_id,
            cm.QBankParaphraseModel.snapshot_id == run.snapshot_id,
            tuple_(cm.QBankParaphraseModel.qbank_id, cm.QBankParaphraseModel.compiler_unit_id).in_(
                eligible_keys
            ),
        )
        support: dict[str, list[dict[str, Any]]] = {}
        for ref in refs:
            if ref.role == "expected":
                support.setdefault(ref.qbank_id, []).append(_model_mapping(ref))
        cutoff = await session.get(
            BenchmarkSessionModel, (run.snapshot_id, query.available_after_session)
        )
        visible = (
            []
            if cutoff is None or cutoff.conversation_id != query.conversation_id
            else await rows(
                BenchmarkSessionModel,
                BenchmarkSessionModel.snapshot_id == run.snapshot_id,
                BenchmarkSessionModel.conversation_id == query.conversation_id,
                BenchmarkSessionModel.session_order <= cutoff.session_order,
            )
        )
        return ProcessFacts(
            run=_model_mapping(run),
            source_run=_model_mapping(source_run) if source_run is not None else None,
            source_match_cells=source_match_cells,
            cells={r.cell_result_id: _model_mapping(r) for r in cells},
            executions=executions,
            program_step_ids=program_step_ids,
            matches=matches,
            rounds=rounds,
            questions={
                r.qbank_id: question_from_rows(
                    _model_mapping(r),
                    [_model_mapping(p) for p in paraphrases if p.qbank_id == r.qbank_id],
                    [_model_mapping(ref) for ref in refs if ref.qbank_id == r.qbank_id],
                ).model_dump(mode="json")
                for r in questions
            },
            support=support,
            visible_sessions=frozenset(r.session_id for r in visible),
        )


async def evaluate_process_metrics(sessions, **kwargs) -> ProcessMetricsReport:
    try:
        facts = (await load_process_facts(
            sessions, query=kwargs["query"], query_run_id=kwargs["query_run_id"]
        ) if any(c.method == "pmmc" for c in kwargs["cells"]) else ProcessFacts())
    except SQLAlchemyError:
        facts = ProcessFacts(load_error="process_database_unavailable")
    except (ValueError, KeyError, TypeError):
        facts = ProcessFacts(load_error="process_database_facts_invalid")
    report = compute_process_metrics(facts=facts, **kwargs)
    cells = []
    for binding, cell in zip(kwargs["cells"], report.cells, strict=True):
        if binding.method != "pmmc":
            from pmmc.evaluation.step8_baseline_process import evaluate_baseline_retrieval_metrics

            metrics = await evaluate_baseline_retrieval_metrics(
                sessions,
                query=kwargs["query"],
                snapshot_id=kwargs["snapshot_id"],
                binding=binding,
                gold=kwargs["gold"],
                memories=kwargs["memories"],
            )
            if set(metrics) != set(RETRIEVED):
                raise ValueError("baseline_process_metric_contract_mismatch")
            cell = cell.model_copy(update={"metrics": {**cell.metrics, **metrics}})
        cells.append(cell)
    return ProcessMetricsReport(
        cells=cells,
        status="incomplete"
        if any(
            m.status in {"incomplete", "not_evaluated"} for c in cells for m in c.metrics.values()
        )
        else "complete",
    )
