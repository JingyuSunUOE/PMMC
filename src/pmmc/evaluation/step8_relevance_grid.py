"""Evaluator-only scoring of *generated* runtime relevance-grid answers.

The campaign evaluator supplies authoritative typed-answer rows and its existing
method-blind Judge callback. This adapter performs no runtime writes or retrieval
and never infers an answer for an ungenerated context.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from pmmc.benchmarks.ids import stable_hash
from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.step8_typed_gold import TypedGoldResolution, score_resolved_typed_answer
from pmmc.runtime.query_executor_config import JOINT_POLICY
from pmmc.runtime.relevance_grid import validate_joint_outcome, validate_relevance_grid_points
from pmmc.schemas import DataScope, QueryInput
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.query_executor import QueryExperimentCellResult, runtime_fact_payload

RelevanceJudge = Callable[[TypedAnswerResult], Awaitable[Mapping[str, Any]]]


async def evaluate_relevance_grid_answers(
    *,
    scope: DataScope,
    query: QueryInput,
    snapshot_id: str,
    query_run_id: str,
    cells: Sequence[QueryExperimentCellResult],
    answers: Mapping[str, TypedAnswerResult],
    resolution: TypedGoldResolution,
    thresholds: list[float],
    judge: RelevanceJudge | None,
    primary_thresholds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Call once per query/backbone, sharing each answer's score across all grid cells.

The caller must use evaluator credentials, bind this report to its registered
evaluation and persist it before completion. Required Judge failures remain
incomplete, and no value is derived from an evidence-only oracle proxy.
"""
    if scope is not DataScope.EVALUATION:
        raise ValueError("relevance-grid scoring requires the evaluator boundary")
    if (resolution.query_id != query.query_id
            or resolution.query_hash != stable_hash(query.model_dump(mode="json"))):
        raise ValueError("relevance-grid gold resolution has mismatched query lineage")
    # Keep the established typed conversion/scorer without importing the production
    # workflow at module load time (the workflow can consume this adapter).
    from pmmc.evaluation.step8_workflow import typed_candidate

    pending: dict[str, TypedAnswerResult] = {}
    point_links: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    source_hashes: list[str] = []
    if primary_thresholds is not None and (
        len(cells) != 1 or cells[0].cell.treatment != "pmmc"
        or (cells[0].cell.question_budget, cells[0].cell.revision_depth) != (8, 2)
        or not all(t in thresholds for t in primary_thresholds)
    ):
        raise ValueError("primary relevance readout requires exactly its active b8/r2 cell")
    for cell in cells:
        if cell.cell.treatment in {"no_fallback", "forced_top1_counterfactual"}:
            continue
        if (cell.query_executor_run_id != query_run_id or cell.query_id != query.query_id
                or cell.content_hash != canonical_content_hash(runtime_fact_payload(
                    cell.model_dump(exclude={"cell_result_id", "content_hash", "created_at"})
                ))):
            raise ValueError("relevance-grid cell scope or content hash mismatch")
        source_hashes.append(cell.content_hash)
        points = cell.metadata.get("relevance_grid")
        if points is None:
            if cell.binding is None and cell.execution_id is None and not cell.error_type:
                point_links.append({"cell_id": cell.cell_id, "status": (
                    "primary_not_executed" if primary_thresholds is not None else "no_candidate"
                )})
                continue
            missing.append({"cell_id": cell.cell_id, "error_code": "relevance_grid_missing"})
            continue
        validate_relevance_grid_points(
            points, run_id=query_run_id, query_id=query.query_id,
            execution_id=cell.execution_id or "", thresholds=thresholds,
        )
        for point in points:
            if primary_thresholds is not None and (
                point["text_threshold"], point["multimodal_threshold"]
            ) != primary_thresholds:
                if point["status"] != "deferred_analysis" or any(
                    point.get(name) for name in (
                        "prepared_key", "typed_answer_id", "answer_model_call_id",
                        "grounding_call_ids", "arrived_joint_image_hashes",
                    )
                ):
                    raise ValueError("primary readout contains an off-grid request or answer")
                continue
            link = {name: point[name] for name in (
                "grid_point_id", "text_threshold", "multimodal_threshold",
                "evidence_subset_hash", "status",
            )}
            link["cell_id"] = cell.cell_id
            point_links.append(link)
            validate_joint_outcome(point, cell.metadata.get("joint_contexts", {}))
            if point.get("joint_policy") == JOINT_POLICY:
                link.update({key: point[key] for key in (
                    "joint_policy", "joint_threshold", "joint_image_hashes",
                    "arrived_joint_image_hashes", "score_manifest_hash", "answer_support",
                )})
            if point["status"] == "context_rejected":
                continue
            if point["status"] == "relevance_rejected" and point.get("sufficient") is False:
                continue
            answer_id = point.get("typed_answer_id")
            answer = answers.get(answer_id) if isinstance(answer_id, str) else None
            if point["status"] != "completed" or answer is None:
                missing.append({"grid_point_id": point["grid_point_id"],
                                "error_code": "relevance_grid_answer_missing"})
                continue
            if (answer.run_id != query_run_id or answer.query_id != query.query_id
                    or answer.snapshot_id != snapshot_id
                    or answer.answer_model_call_id != point.get("answer_model_call_id")
                    or answer.terminal_state != point.get("terminal_state")
                    or sorted(call.model_call_id for call in answer.calls
                              if call.role in {"answer_grounding", "grounding_format_retry"})
                    != sorted(point.get("grounding_call_ids", []))):
                raise ValueError("relevance-grid answer lineage mismatch")
            pending[answer.typed_answer_id] = answer
            link["typed_answer_id"] = answer.typed_answer_id

    scored: list[dict[str, Any]] = []
    # Primary's sole selected answer is scored once by the owning evaluation;
    # this artifact attests the active R point, not a second score or Judge call.
    for answer_id, answer in (() if primary_thresholds is not None else sorted(pending.items())):
        row: dict[str, Any] = {
            "typed_answer_id": answer_id,
            "answer_content_hash": canonical_content_hash(runtime_fact_payload(answer)),
            "terminal_state": answer.terminal_state,
            "status": "completed", "typed_score": None, "judge_score": None,
            "judge_call_id": None,
        }
        if resolution.reference is None:
            row.update(status="incomplete", error_code=resolution.unavailable_reason)
        elif answer.terminal_state != "answered":
            row.update(typed_score=0.0, judge_unavailable_reason="answer_abstained")
        else:
            candidate = typed_candidate(answer)
            if candidate is None:
                raise ValueError("answered relevance context has no typed candidate")
            score = score_resolved_typed_answer(candidate, resolution)
            row["typed_assessment"] = asdict(score)
            row["typed_score"] = score.value
            if not score.applicable:
                row.update(status="incomplete", error_code=score.unavailable_reason)
            needs_judge = candidate.kind == "text" and resolution.text_reference is not None
            if needs_judge:
                try:
                    if judge is None:
                        raise ValueError("judge_unavailable")
                    result = await judge(answer)
                    value = result["value"]
                    call_id = result["model_call_id"]
                    if value not in {0.0, 0.5, 1.0} or isinstance(value, bool) or not call_id:
                        raise ValueError("invalid_judge_receipt")
                    row.update(judge_score=float(value), judge_call_id=call_id)
                except Exception:
                    row.update(status="incomplete", error_code="relevance_grid_judge_missing")
            else:
                row["judge_unavailable_reason"] = "typed_reference_not_text_judge"
        scored.append(row)
    payload = {
        "protocol": "step8-relevance-answer-evaluation-v1",
        "query_run_id": query_run_id, "query_id": query.query_id, "snapshot_id": snapshot_id,
        "source_cell_hashes": sorted(source_hashes), "gold_resolution": resolution.audit_metadata(),
        "point_count": len(point_links), "answer_count": len(pending),
        **({"readout": "primary_active_point", "thresholds": list(primary_thresholds)}
           if primary_thresholds is not None else {}),
        "status": "incomplete" if missing or any(row["status"] != "completed" for row in scored)
        else "completed",
        "points": point_links, "answers": scored, "missing": missing,
    }
    return {**payload, "content_hash": canonical_content_hash(payload)}
