"""Validate generated relevance contexts before projecting an offline policy."""

from __future__ import annotations

from math import isfinite
from typing import Any

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.models.execution import _generation_manifest
from pmmc.runtime.answering import map_typed_image_items
from pmmc.runtime.grounding import (
    GROUNDING_MODEL_ALIAS,
    grounding_evidence_hash,
    source_content_parts,
    strict_json_object,
)
from pmmc.runtime.qbank_matcher import decide_runtime_route
from pmmc.runtime.query_executor_config import JOINT_POLICY, QueryExecutorConfig
from pmmc.runtime.relevance_grid import validate_joint_outcome, validate_relevance_grid_points
from pmmc.schemas import GenerationRequest, ModelMessage, QBankRouteClass
from pmmc.schemas.answer_output import (
    AnswerContractV3,
    GroundingAssessment,
    TypedAnswerContext,
    TypedAnswerResult,
)
from pmmc.schemas.query_executor import (
    QBankMatchBatch,
    QBankMatchFusedCandidate,
    QueryExperimentCellResult,
    runtime_fact_payload,
)
from pmmc.schemas.step8 import GAP_THRESHOLDS, MATCH_THRESHOLDS, RELEVANCE_THRESHOLDS


def relevance_point(
    fact: QueryExperimentCellResult, report: dict[str, Any], text: float, multimodal: float
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if text not in RELEVANCE_THRESHOLDS or multimodal not in RELEVANCE_THRESHOLDS:
        raise ValueError("relevance_policy_outside_frozen_grid")
    grid = report.get("relevance_grid", {})
    if (
        grid.get("protocol") != "step8-relevance-answer-evaluation-v1"
        or grid.get("content_hash")
        != canonical_content_hash({k: v for k, v in grid.items() if k != "content_hash"})
        or grid.get("query_id") != fact.query_id
        or grid.get("query_run_id") != fact.query_executor_run_id
        or fact.content_hash not in grid.get("source_cell_hashes", [])
    ):
        raise ValueError("relevance_evaluation_source_mismatch")
    points = fact.metadata.get("relevance_grid")
    if (
        points is None
        and fact.binding is None
        and fact.execution_id is None
        and not fact.error_type
    ):
        if [link for link in grid.get("points", []) if link.get("cell_id") == fact.cell_id] != [
            {"cell_id": fact.cell_id, "status": "no_candidate"}
        ]:
            raise ValueError("relevance_no_candidate_evaluation_missing")
        return None, None
    if not isinstance(points, list):
        raise ValueError("relevance_contexts_not_generated")
    validate_relevance_grid_points(
        points,
        run_id=fact.query_executor_run_id,
        query_id=fact.query_id,
        execution_id=fact.execution_id or "",
        thresholds=list(RELEVANCE_THRESHOLDS),
    )
    matches = [
        p for p in points if (p["text_threshold"], p["multimodal_threshold"]) == (text, multimodal)
    ]
    if len(matches) != 1:
        raise ValueError("relevance_context_denominator_invalid")
    point = matches[0]
    _verify_point_id(point)
    validate_joint_outcome(point, fact.metadata.get("joint_contexts", {}))
    if (
        fact.binding is None
        or point.get("binding_id") != fact.binding.query_program_binding_id
        or point.get("source_cell_id") != fact.cell_id
        or point.get("score_manifest_hash")
        != canonical_content_hash(sorted(item.content_hash for item in fact.relevance_items))
        or point.get("score_coverage_complete") is not True
        or point.get("unscored_reference_hashes") != []
        or point.get("error_type") is not None
    ):
        raise ValueError("relevance_context_source_mismatch")
    links = [
        p
        for p in grid.get("points", [])
        if p.get("cell_id") == fact.cell_id and p.get("grid_point_id") == point["grid_point_id"]
    ]
    if len(links) != 1 or any(
        links[0].get(k) != point.get(k)
        for k in (
            "text_threshold",
            "multimodal_threshold",
            "evidence_subset_hash",
            "status",
        )
    ):
        raise ValueError("relevance_context_evaluation_link_missing")
    if point.get("joint_policy") == JOINT_POLICY and any(
        links[0].get(key) != point.get(key) for key in (
            "joint_policy", "joint_threshold", "joint_image_hashes", "arrived_joint_image_hashes",
            "score_manifest_hash", "answer_support",
        )
    ):
        raise ValueError("joint evidence differs from the evaluation link")
    if point["status"] == "context_rejected":
        return point, None
    if point["status"] == "relevance_rejected" and point.get("sufficient") is False:
        if (
            any(
                point.get(key) is not None
                for key in (
                    "typed_answer_id",
                    "answer_model_call_id",
                    "answer_identity_hash",
                    "prepared_key",
                    "terminal_state",
                )
            )
            or point.get("grounding_call_ids") != []
        ):
            raise ValueError("relevance_rejected_context_has_answer")
        return point, None
    if (
        point["status"] != "completed"
        or point.get("sufficient") is not True
        or point.get("structural_requirement_success") is not True
        or point.get("required_roles_satisfied") is not True
        or not point.get("prepared_key")
        or not point.get("answer_identity_hash")
        or links[0].get("typed_answer_id") != point.get("typed_answer_id")
    ):
        raise ValueError("relevance_context_answer_incomplete")
    scores = [
        s
        for s in grid.get("answers", [])
        if s.get("typed_answer_id") == point.get("typed_answer_id")
    ]
    if len(scores) != 1 or scores[0].get("status") != "completed":
        raise ValueError("relevance_context_score_missing")
    scored = scores[0]
    if scored.get("terminal_state") != point.get("terminal_state"):
        raise ValueError("relevance_context_score_terminal_mismatch")
    _validate_score(scored)
    return point, scored


def _verify_point_id(point: dict[str, Any]) -> None:
    # The producer names the pre-generation point. Restore only the fields that
    # answering resolves; the selection, raw score manifest and gates stay bound.
    original = {
        key: value
        for key, value in point.items()
        if key
        not in {
            "grid_point_id",
            "source_grid_point_id",
            "source_query_executor_run_id",
        }
    }
    original.update(
        status="pending_answer" if point.get("sufficient") else "relevance_rejected",
        answer_support=None,
        prepared_key=None,
        answer_identity_hash=None,
        typed_answer_id=None,
        answer_model_call_id=None,
        grounding_call_ids=[],
        terminal_state=None,
        error_type=None,
        semantic_relation_status="not_applicable"
        if point.get("semantic_relation_status") == "not_applicable"
        else "pending_grounding",
    )
    if point.get("joint_policy") == JOINT_POLICY:
        original["arrived_joint_image_hashes"] = []
    if "source_grid_point_id" in point:
        original["query_executor_run_id"] = point["source_query_executor_run_id"]
    source_id = stable_compiler_id("query-relevance-grid-point", original)
    expected = source_id
    if "source_grid_point_id" in point:
        if source_id != point["source_grid_point_id"]:
            raise ValueError("relevance_context_point_identity_mismatch")
        expected = stable_compiler_id(
            "query-relevance-grid-consumer", [point["query_executor_run_id"], source_id]
        )
    if point.get("grid_point_id") != expected:
        raise ValueError("relevance_context_point_identity_mismatch")


def _validate_score(scored: dict[str, Any]) -> None:
    value, judge = scored.get("typed_score"), scored.get("judge_score")
    if (
        not isinstance(value, (float, int))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0 <= value <= 1
        or (judge is not None and (type(judge) not in {float, int} or judge not in {0.0, 0.5, 1.0}))
        or (judge is not None and not scored.get("judge_call_id"))
        or (
            judge is None
            and (
                scored.get("judge_call_id") is not None
                or not scored.get("judge_unavailable_reason")
            )
        )
        or (scored.get("terminal_state") != "answered" and (value != 0 or judge is not None))
    ):
        raise ValueError("relevance_context_score_invalid")
    assessment = scored.get("typed_assessment")
    if assessment is not None and (
        assessment.get("value") != value
        or assessment.get("applicable") is not True
        or assessment.get("unavailable_reason") is not None
    ):
        raise ValueError("relevance_context_score_assessment_mismatch")


def project_relevance_score(
    base: dict[str, Any],
    point: dict[str, Any],
    scored: dict[str, Any],
    answer: TypedAnswerResult,
    fact_hash: str,
) -> dict[str, Any]:
    _validate_score(scored)
    if (
        scored.get("answer_content_hash") != canonical_content_hash(runtime_fact_payload(answer))
        or point.get("typed_answer_id") != answer.typed_answer_id
        or point.get("answer_model_call_id") != answer.answer_model_call_id
        or scored.get("typed_answer_id") != answer.typed_answer_id
        or point.get("terminal_state") != answer.terminal_state
        or point.get("query_id") != answer.query_id
        or point.get("query_executor_run_id") != answer.run_id
        or sorted(point.get("grounding_call_ids", []))
        != sorted(
            call.model_call_id
            for call in answer.calls
            if call.role in {"answer_grounding", "grounding_format_retry"}
        )
        or point.get("answer_support")
        != (answer.grounding.assessment.supported if answer.grounding else False)
        or fact_hash != canonical_content_hash(runtime_fact_payload(answer))
        or answer.method != "pmmc"
        or scored.get("status") != "completed"
        or scored.get("terminal_state") != answer.terminal_state
        or answer.terminal_state not in {"answered", "abstained"}
    ):
        raise ValueError("relevance_answer_score_identity_mismatch")
    judge = scored.get("judge_score")
    return {
        **base,
        "typed_answer_id": answer.typed_answer_id,
        "source_fact_id": answer.typed_answer_id,
        "source_fact_hash": fact_hash,
        "terminal_state": answer.terminal_state,
        "status": "answered" if answer.terminal_state == "answered" else "abstention",
        "value": scored["typed_score"],
        "judge_value": judge,
        "answer_source_method": answer.method,
        "answer_source_variant": answer.answer_variant,
        "judge_model_call_id": scored.get("judge_call_id"),
        "judge_applicable": judge is not None,
        "judge_method_blind": True if judge is not None else None,
        "judge_unavailable_reason": scored.get("judge_unavailable_reason"),
        "typed_judge_disagreement": scored["typed_score"] != judge if judge is not None else None,
        "unavailable_reason": None,
        "engineering_error": None,
    }


def no_answer_score(base: dict[str, Any], fact: QueryExperimentCellResult) -> dict[str, Any]:
    return {
        **base,
        "typed_answer_id": None,
        "source_fact_id": fact.cell_result_id,
        "source_fact_hash": fact.content_hash,
        "terminal_state": "no_answer",
        "status": "abstention",
        "value": 0.0,
        "judge_value": None,
        "judge_model_call_id": None,
        "judge_applicable": False,
        "judge_method_blind": None,
        "judge_unavailable_reason": "answer_abstained",
        "typed_judge_disagreement": None,
        "answer_source_method": None,
        "answer_source_variant": None,
        "unavailable_reason": None,
        "engineering_error": None,
    }


def verify_relevance_projection(
    store: CampaignArtifactStore, saved: dict[str, Any], context: dict[str, Any]
) -> None:
    """Verify a projection against inventoried CAS roots; never call a model or scorer.

    CAS proves byte identity, not authorship. The caller must also bind these roots
    to its authoritative campaign inventory (as approval does).
    """
    try:
        _verify_projection(store, saved, context)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("relevance_projection_malformed_proof") from exc


def _verify_projection(
    store: CampaignArtifactStore, saved: dict[str, Any], context: dict[str, Any]
) -> None:
    from pmmc.experiments.step8_analysis import threshold_proof_hashes

    for digest in threshold_proof_hashes([context, saved]):
        store.verify(digest)
    if (
        context.get("schema_version") != "step8-threshold-context-v3"
        or context.get("relevance_policy") != "generated_evidence_subset"
        or store.load_json(saved["context_sha256"]) != context
        or saved["score_report_sha256"] != context["score_report_sha256"]
        or saved["terminal_fact_sha256"] != context["source_cell_sha256"]
        or any(saved[key] != context[key] for key in ("cell_id", "query_id", "axis", "thresholds"))
        or saved["score"]["cell_id"] != context["cell_id"]
        or saved["score"]["variant"] != context["cell_id"]
        or saved["score"]["method"] != "pmmc"
        or saved["score"]["source_variant"] not in {"active", "no_fallback"}
        or type(context["fallback_enabled"]) is not bool
        or context["fallback_enabled"] != (saved["score"]["source_variant"] == "active")
    ):
        raise ValueError("relevance_projection_envelope_mismatch")
    fact = QueryExperimentCellResult.model_validate(store.load_json(context["source_cell_sha256"]))
    report = store.load_json(context["score_report_sha256"])
    identity = store.load_json(context["evaluation_identity_sha256"])
    if (
        fact.content_hash
        != canonical_content_hash(
            runtime_fact_payload(
                fact.model_dump(exclude={"cell_result_id", "content_hash", "created_at"})
            )
        )
        or report.get("identity_hash") != canonical_content_hash(identity)
        or report.get("campaign_id") != context["campaign_id"]
        or report.get("query_id") != context["query_id"]
        or report.get("query_executor_run_id") != fact.query_executor_run_id
        or identity.get("campaign_id") != context["campaign_id"]
        or identity.get("query_id") != fact.query_id
        or report.get("relevance_grid", {}).get("snapshot_id") != identity.get("snapshot_id")
        or context["query_id"] != fact.query_id
        or identity.get("source_run_id") != fact.query_executor_run_id
        or context["question_budget"] != fact.cell.question_budget
        or context["round_id"] != fact.cell.revision_depth
        or fact.cell_result_id
        != stable_compiler_id(
            "query-cell-result", [fact.query_executor_run_id, fact.query_id, fact.cell_id]
        )
    ):
        raise ValueError("relevance_projection_source_identity_mismatch")
    point, scored = relevance_point(fact, report, *context["relevance_thresholds"])
    if point != context["selected_grid_point"]:
        raise ValueError("relevance_projection_point_changed")
    matching = store.load_json(context["matching_sha256"])
    config = QueryExecutorConfig.model_validate(context["route_config"])
    _verify_policy(context, fact, config)
    candidates = _verify_matching(store, matching, fact, config, identity["snapshot_id"])
    route = decide_runtime_route(
        query_executor_run_id=fact.query_executor_run_id,
        query_id=fact.query_id,
        match_batch_id=fact.route.match_batch_id,
        candidates=candidates,
        config=config,
    )
    if runtime_fact_payload(route) != context["selected_route"]:
        raise ValueError("relevance_projection_route_changed")
    primary = [s for s in report["scores"] if s["source_fact_id"] == fact.cell_result_id]
    if len(primary) != 1 or (
        primary[0].get("source_fact_hash") != fact.content_hash
        or primary[0].get("typed_answer_id") != fact.selected_answer_id
        or primary[0].get("terminal_state") != fact.terminal_state
        or primary[0].get("method") != "pmmc"
        or primary[0].get("source_variant") != "active"
    ):
        raise ValueError("relevance_projection_primary_score_missing")
    source = {
        "cell_id": primary[0]["cell_id"],
        "source_id": fact.cell_result_id,
        "source_hash": fact.content_hash,
        "terminal_state": fact.terminal_state,
    }
    if identity.get("sources", []).count(source) != 1:
        raise ValueError("relevance_projection_evaluation_source_mismatch")
    accepted = bool(
        route.outcome.value == "program"
        and point
        and point.get("sufficient")
        and point.get("terminal_state") in {"answered", "abstained"}
        and fact.binding is not None
        and fact.binding.plan_id == route.selected_plan_id
        and fact.binding.qbank_id == route.selected_qbank_id
    )
    expected_id = (
        point["typed_answer_id"]
        if accepted and point
        else (fact.fallback_answer_id if context["fallback_enabled"] else None)
    )
    if (
        context["typed_answer_id"] != expected_id
        or saved["score"]["typed_answer_id"] != expected_id
    ):
        raise ValueError("relevance_projection_answer_selection_changed")
    if context["policy_outcome"] != (
        "pmmc" if accepted else "fallback" if context["fallback_enabled"] else "no_answer"
    ) or (expected_id is None and context["fallback_enabled"]):
        raise ValueError("relevance_projection_outcome_mismatch")
    if expected_id is not None:
        answer, row = _verify_answer(store, saved, context, identity, fact, accepted, point)
        if accepted:
            if point is None or scored is None:
                raise ValueError("relevance_projection_score_missing")
            expected = project_relevance_score(
                primary[0], point, scored, answer, row["content_hash"]
            )
        else:
            matches = [
                s
                for s in report["scores"]
                if s.get("typed_answer_id") == expected_id
                and s.get("method") == "native_multimodal_rag"
                and s.get("source_variant") == "primary"
            ]
            if len(matches) != 1 or (
                matches[0]["source_fact_hash"] != row["content_hash"]
                or matches[0]["source_fact_id"] != expected_id
                or matches[0]["terminal_state"] != answer.terminal_state
                or {
                    "cell_id": matches[0]["cell_id"],
                    "source_id": expected_id,
                    "source_hash": row["content_hash"],
                    "terminal_state": answer.terminal_state,
                }
                not in identity.get("sources", [])
            ):
                raise ValueError("relevance_projection_fallback_score_missing")
            expected = matches[0]
    else:
        if saved.get("answer_sha256") is not None or context["answer_proof"] is not None:
            raise ValueError("relevance_projection_no_answer_has_proof")
        expected = no_answer_score(primary[0], fact)
    projected = {
        **expected,
        **{k: saved["score"][k] for k in ("cell_id", "variant", "method", "source_variant")},
    }
    if (
        saved["score"] != projected
        or context["source_score"] != expected
        or context["selected_answer_method"] != expected["answer_source_method"]
    ):
        raise ValueError("relevance_projection_score_changed")


def _verify_policy(
    context: dict[str, Any], fact: QueryExperimentCellResult, config: QueryExecutorConfig
) -> None:
    text, image = context["relevance_thresholds"]
    thresholds = context["thresholds"]
    expected = {
        "text_relevance": [text],
        "multimodal_relevance": [image],
        "relevance_grid": [text, image],
        "approved_policy": [config.route_match_threshold, config.route_gap_threshold, text, image],
    }
    if (
        context["axis"] not in expected
        or thresholds != expected[context["axis"]]
        or config.route_match_threshold not in MATCH_THRESHOLDS
        or config.route_gap_threshold not in GAP_THRESHOLDS
        or (
            context["axis"] == "text_relevance"
            and image != config.multimodal_evidence_relevance_threshold
        )
        or (
            context["axis"] == "multimodal_relevance"
            and text != config.query_evidence_relevance_threshold
        )
        or not config.is_v3
        or config.treatment != context["treatment"]
        or fact.cell.treatment
        != ("pmmc" if context["treatment"] == "main" else context["treatment"])
    ):
        raise ValueError("relevance_projection_policy_mismatch")
    if context["axis"] != "approved_policy" and (
        config.route_match_threshold != fact.route.match_threshold
        or config.route_gap_threshold != fact.route.gap_threshold
    ):
        raise ValueError("relevance_projection_unrequested_route_change")


def _verify_matching(
    store: CampaignArtifactStore,
    matching: dict[str, Any],
    fact: QueryExperimentCellResult,
    config: QueryExecutorConfig,
    snapshot_id: str,
) -> tuple[QBankMatchFusedCandidate, ...]:
    from pmmc.experiments.step8_analysis import threshold_proof_hashes

    for digest in threshold_proof_hashes(matching):
        store.verify(digest)
    batch = QBankMatchBatch.model_validate(matching["batch"])
    candidates = tuple(
        QBankMatchFusedCandidate.model_validate(row) for row in matching["candidates"]
    )
    channels = matching["channels"]
    classes = {value: index for index, value in enumerate(QBankRouteClass)}
    payload = {
        "run_id": batch.query_executor_run_id,
        "query_id": batch.query_id,
        "projection_id": batch.routing_projection_id,
        "question_budget": batch.question_budget,
        "round_id": batch.active_round,
        "channel_hashes": [row["content_hash"] for row in channels],
        "fused_hashes": [candidate.content_hash for candidate in candidates],
    }
    if (
        (
            batch.query_executor_run_id,
            batch.query_id,
            batch.match_batch_id,
            batch.snapshot_id,
            batch.question_budget,
            batch.active_round,
        )
        != (
            fact.query_executor_run_id,
            fact.query_id,
            fact.route.match_batch_id,
            snapshot_id,
            fact.cell.question_budget,
            fact.cell.revision_depth,
        )
        or batch.content_hash != canonical_content_hash(payload)
        or len(candidates) != batch.fused_count
        or len({candidate.fused_candidate_id for candidate in candidates}) != len(candidates)
        or list(candidates) != sorted(candidates, key=lambda c: (classes[c.route_class], c.rank))
        or any(
            (row["match_batch_id"], row["query_id"]) != (batch.match_batch_id, fact.query_id)
            for row in channels
        )
    ):
        raise ValueError("relevance_projection_matching_identity_mismatch")
    for candidate in candidates:
        if (
            candidate.match_batch_id != batch.match_batch_id
            or candidate.query_id != fact.query_id
            or (
                config.reranker_enabled
                and candidate.content_hash
                != canonical_content_hash(
                    candidate.model_dump(mode="json", exclude={"content_hash", "metadata"})
                )
            )
            or (
                not config.reranker_enabled
                and candidate.confidence != min(1.0, max(0.0, 61 * candidate.raw_rrf))
            )
        ):
            raise ValueError("relevance_projection_candidate_mismatch")
    for route_class in QBankRouteClass:
        ranks = [c.rank for c in candidates if c.route_class == route_class]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("relevance_projection_candidate_ranks_invalid")
    frozen_config = config.model_copy(
        update={
            "active_match_threshold": fact.route.match_threshold,
            "active_gap_threshold": fact.route.gap_threshold,
            "no_reranker_match_threshold": fact.route.match_threshold,
            "no_reranker_gap_threshold": fact.route.gap_threshold,
        }
    )
    frozen = decide_runtime_route(
        query_executor_run_id=fact.query_executor_run_id,
        query_id=fact.query_id,
        match_batch_id=batch.match_batch_id,
        candidates=candidates,
        config=frozen_config,
    )
    if runtime_fact_payload(frozen) != runtime_fact_payload(fact.route):
        raise ValueError("relevance_projection_frozen_route_mismatch")
    return candidates


def _verify_call(
    store: CampaignArtifactStore, proof: dict[str, Any], call_id: str, run_id: str, query_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = store.load_json(proof["trace_sha256"])
    request = store.load_json(proof["input_sha256"])
    if (
        trace.get("model_call_id") != call_id
        or trace.get("run_id") != run_id
        or trace.get("query_id") != query_id
        or trace.get("status") != "success"
        or trace.get("prompt_hash") != proof["input_sha256"]
        or not proof.get("output_sha256")
        or proof.get("failure_sha256") is not None
        or trace.get("model_alias") != request.get("model_alias")
        or trace.get("request_config", {}).get("request_metadata") != request.get("metadata")
    ):
        raise ValueError("relevance_projection_call_identity_mismatch")
    store.verify(proof["output_sha256"])
    return trace, request


def _source_parts(context: TypedAnswerContext, aliases: list[str] | None = None) -> list[Any]:
    request = GenerationRequest(
        request_id="proof",
        model_alias="proof",
        max_output_tokens=1,
        messages=[ModelMessage(role="user", parts=source_content_parts(context, aliases))],
    )
    manifest: Any = _generation_manifest(request)
    return manifest["messages"][0]["parts"]


def _verify_answer(
    store: CampaignArtifactStore,
    saved: dict[str, Any],
    context: dict[str, Any],
    identity: dict[str, Any],
    fact: QueryExperimentCellResult,
    accepted: bool,
    point: dict[str, Any] | None,
) -> tuple[TypedAnswerResult, dict[str, Any]]:
    proof = context["answer_proof"]
    row = store.load_json(proof["typed_fact_sha256"])
    raw = store.load_json(saved["answer_sha256"])
    answer = TypedAnswerResult.model_validate(raw)
    method = "pmmc" if accepted else "native_multimodal_rag"
    if (
        proof["typed_answer_sha256"] != saved["answer_sha256"]
        or row["payload"]["metadata"]["typed_result"] != raw
        or canonical_content_hash(raw) != row["content_hash"]
        or answer.typed_answer_id != context["typed_answer_id"]
        or answer.query_id != fact.query_id
        or answer.snapshot_id != identity["snapshot_id"]
        or answer.method != method
        or (not accepted and answer.answer_variant != "primary")
        or (accepted and answer.run_id != fact.query_executor_run_id)
        or answer.terminal_state not in {"answered", "abstained"}
        or any(
            row.get(key) != value
            for key, value in {
                "typed_answer_id": answer.typed_answer_id,
                "query_executor_run_id": answer.run_id,
                "query_id": answer.query_id,
                "answer_model_call_id": answer.answer_model_call_id,
                "terminal_state": answer.terminal_state,
                "answer_identity_hash": proof["answer_identity_hash"],
            }.items()
        )
        or (
            accepted
            and (point is None or point["answer_identity_hash"] != proof["answer_identity_hash"])
        )
        or set(proof["calls"]) != {call.model_call_id for call in answer.calls}
        or proof["grounding"] != runtime_fact_payload(answer.grounding)
        or any(
            row["payload"].get(key) != row.get(key)
            for key in (
                "typed_answer_id",
                "query_executor_run_id",
                "query_id",
                "answer_model_call_id",
                "terminal_state",
                "answer_identity_hash",
                "content_hash",
            )
        )
        or row["payload"].get("output") != answer.output.model_dump(mode="json")
        or row["payload"].get("ungated_output") != answer.ungated_output.model_dump(mode="json")
        or row["payload"].get("grounding") != runtime_fact_payload(answer.grounding)
    ):
        raise ValueError("relevance_projection_answer_identity_mismatch")
    typed = TypedAnswerContext.model_validate(store.load_json(proof["typed_context_sha256"]))
    if accepted and point is not None and point.get("joint_policy") == JOINT_POLICY:
        receipt = fact.metadata["joint_contexts"][point["prepared_key"]]
        native = {canonical_content_hash(image.image_id) for source in typed.sources
                  for image in source.images if image.image is not None}
        if receipt["context_hash"] != canonical_content_hash(typed.model_dump(mode="json")) or set(
            point["arrived_joint_image_hashes"]
        ) != native.intersection(point["joint_image_hashes"]):
            raise ValueError("joint arrival differs from the generated source evidence")
    requests = {}
    queries = set()
    for call in answer.calls:
        trace, request = _verify_call(
            store,
            proof["calls"][call.model_call_id],
            call.model_call_id,
            answer.run_id,
            answer.query_id,
        )
        grounding = call.role in {"answer_grounding", "grounding_format_retry"}
        alias = (
            answer.grounding.model_alias
            if grounding and answer.grounding
            else context["answer_model"]
        )
        model = (
            answer.grounding.actual_model_id
            if grounding and answer.grounding
            else answer.actual_model_id
        )
        if (
            trace.get("model_alias") != alias
            or trace.get("actual_model_id") != model
            or request.get("metadata", {}).get("model_role") != call.role
        ):
            raise ValueError("relevance_projection_answer_model_mismatch")
        requests[call.model_call_id] = request
        if call.model_call_id == answer.answer_model_call_id:
            parsed = trace.get("parsed_output") or {}
            if (
                parsed.get("finish_reason") not in {"stop", "eos", "end_turn"}
                or AnswerContractV3.model_validate(strict_json_object(parsed["text"]))
                != answer.ungated_output
            ):
                raise ValueError("relevance_projection_answer_output_mismatch")
        if grounding and answer.grounding and call == answer.grounding.calls[-1]:
            parsed = trace.get("parsed_output") or {}
            if (
                parsed.get("finish_reason") not in {"stop", "eos", "end_turn"}
                or GroundingAssessment.model_validate(strict_json_object(parsed["text"]))
                != answer.grounding.assessment
            ):
                raise ValueError("relevance_projection_grounding_output_mismatch")
        if not grounding:
            parts = request["messages"][1]["parts"]
            expected = _source_parts(typed)
            if (
                request["metadata"].get("typed_answer_context") != typed.model_dump(mode="json")
                or parts[0] != {"kind": "text", "text": "[SOURCE EVIDENCE]"}
                or parts[1 : 1 + len(expected)] != expected
            ):
                raise ValueError("relevance_projection_actual_context_mismatch")
            query = strict_json_object(parts[1 + len(expected)]["text"])
            if (
                set(query) != {"query"}
                or not isinstance(query["query"], str)
                or any(part["kind"] != "image" for part in parts[2 + len(expected) :])
            ):
                raise ValueError("relevance_projection_actual_context_mismatch")
            queries.add(query["query"])
    if not set(answer.ungated_output.used_evidence_aliases) <= {
        source.evidence_alias for source in typed.sources
    }:
        raise ValueError("relevance_projection_unknown_evidence_alias")
    if (
        len(queries) != 1
        or map_typed_image_items(answer.ungated_output, typed) != answer.ungated_image_items
        or map_typed_image_items(answer.output, typed) != answer.image_items
    ):
        raise ValueError("relevance_projection_answer_context_mismatch")
    if accepted and point is not None and point.get("semantic_relation_status") == "not_applicable":
        first = next(call for call in answer.calls if call.role == "answer")
        request_model = GenerationRequest.model_validate(requests[first.model_call_id])
        config = QueryExecutorConfig.model_validate(context["route_config"])
        calculated = canonical_content_hash(
            {
                "run_id": answer.run_id,
                "query_id": answer.query_id,
                "request": request_model.model_dump(
                    mode="json", exclude={"request_id", "metadata"}
                ),
                "typed_context": typed.model_dump(mode="json"),
                "runtime_relation_contract": None,
                "grounding_failure_policy": config.grounding_failure_policy,
                "contract": config.answer_contract_version,
            }
        )
        if calculated != proof["answer_identity_hash"]:
            raise ValueError("relevance_projection_request_identity_mismatch")
    if answer.grounding is not None:
        _verify_grounding(answer, typed, requests, point if accepted else None, queries.pop())
    return answer, row


def _verify_grounding(
    answer: TypedAnswerResult,
    typed: TypedAnswerContext,
    requests: dict[str, Any],
    point: dict[str, Any] | None,
    query: str,
) -> None:
    grounding = answer.grounding
    assert grounding is not None
    aliases = list(answer.ungated_output.used_evidence_aliases)
    relation_required = (
        point is not None and point.get("semantic_relation_status") != "not_applicable"
    )
    for call in grounding.calls:
        request = requests[call.model_call_id]
        parts = request["messages"][1]["parts"]
        proposal = strict_json_object(parts[0]["text"])
        relation = proposal.get("required_evidence_relation")
        if relation is not None:
            aliases = [source.evidence_alias for source in typed.sources]
            supplied = relation["supplied_sources"]
            groups = {group for source in supplied for group in source["group_aliases"]}
            contract_hash = canonical_content_hash(
                {
                    "protocol": "runtime-independent-relation-grounding-v1",
                    "required_semantic_roles": ["independent_group_relation"],
                    "minimum_evidence_groups": relation["minimum_evidence_groups"],
                    "semantic_relation_status": "not_evaluated",
                    "sources": supplied,
                }
            )
            if (
                [source["evidence_alias"] for source in supplied] != aliases
                or relation.get("requirement") != "independent_group_relation"
                or relation.get("contract_hash") != contract_hash
                or request["metadata"].get("runtime_relation_contract_hash") != contract_hash
                or relation["claimed_evidence_aliases"]
                != answer.ungated_output.used_evidence_aliases
                or relation["minimum_evidence_groups"] < 2
                or relation["supplied_independent_group_count"] != len(groups)
                or (
                    grounding.assessment.supported
                    and len(groups) < relation["minimum_evidence_groups"]
                )
            ):
                raise ValueError("relevance_projection_relation_scope_mismatch")
        elif relation_required:
            raise ValueError("relevance_projection_relation_grounding_missing")
        expected = _source_parts(typed, aliases)
        if (
            proposal.get("query") != query
            or proposal.get("proposed_answer") != answer.ungated_output.model_dump(mode="json")
            or (expected and parts[-len(expected) :] != expected)
            or any(part["kind"] != "image" for part in parts[1 : len(parts) - len(expected)])
        ):
            raise ValueError("relevance_projection_grounding_request_mismatch")
    if (
        grounding.model_alias != GROUNDING_MODEL_ALIAS
        or grounding.ungated_output_hash
        != canonical_content_hash(answer.ungated_output.model_dump(mode="json"))
        or grounding.used_evidence_aliases != answer.ungated_output.used_evidence_aliases
        or grounding.evidence_content_hash != grounding_evidence_hash(typed, aliases)
    ):
        raise ValueError("relevance_projection_grounding_context_mismatch")
