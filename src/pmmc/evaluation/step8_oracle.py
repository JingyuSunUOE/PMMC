"""Actual evaluator-only Oracle@50 over the immutable saved answerable ranking.

Integration hooks: build_oracle_pool from load_process_facts and one assigned
ProcessCellBinding; load runtime-safe v3 programs/vectors for its candidates;
register an evaluator owner and construct OracleProgramRunner/OracleAnswerJudge;
call evaluate_oracle_at_50 and retain its private CAS artifact. Do not put this
report, selected programs, answers, or gold in runtime cells or routing state.

This bounded branch does not grant protocol approval or claim campaign completion.
It executes every candidate (at most 50), including after an early correct answer.
Accuracy is null unless the entire retrieved pool has actual evaluated outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from pmmc.benchmarks.ids import stable_hash
from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.evaluation.config import BaselineEvaluationConfig, evaluation_prompt_sha256
from pmmc.evaluation.judge import build_judge_request, judge_score_from_trace, parse_judge_output
from pmmc.evaluation.step8_oracle_execution import (
    OracleModelExecution,
    OracleProgramInput,
    OracleProgramRunner,
    require_evaluator,
)
from pmmc.evaluation.step8_process_metrics import ProcessFacts
from pmmc.evaluation.step8_qbank_semantic import _cell_pool
from pmmc.evaluation.step8_typed_gold import TypedGoldResolution, score_resolved_typed_answer
from pmmc.models.execution import ModelCallContext
from pmmc.schemas import AppStage, DataScope, QueryInput
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.query_executor import QBankMatchFusedCandidate, runtime_fact_payload
from pmmc.schemas.step8_process import ProcessCellBinding

VERSION = "step8-executed-oracle-at-50-v1"


@dataclass(frozen=True)
class OraclePool:
    query: QueryInput
    snapshot_id: str
    query_run_id: str
    compiler_run_id: str
    candidates: tuple[QBankMatchFusedCandidate, ...]
    predicted_questions: dict[str, str]
    source_identity: dict[str, Any]
    content_hash: str

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "protocol": VERSION,
            "query": self.query.model_dump(mode="json"),
            "snapshot_id": self.snapshot_id,
            "query_run_id": self.query_run_id,
            "compiler_run_id": self.compiler_run_id,
            "candidates": [c.model_dump(mode="json") for c in self.candidates],
            "predicted_questions": self.predicted_questions,
            "source": self.source_identity,
        }


def build_oracle_pool(
    *,
    scope: DataScope,
    query: QueryInput,
    snapshot_id: str,
    query_run_id: str,
    binding: ProcessCellBinding,
    facts: ProcessFacts,
) -> OraclePool:
    """Validate complete batch/round/budget lineage before taking saved ranks 1..50."""
    require_evaluator(scope)
    cell = _cell_pool(binding, query, snapshot_id, query_run_id, facts)
    assert facts.run is not None
    match = facts.matches[cell.match_batch_id]
    candidates = tuple(
        sorted(
            (c for c in match.candidates if c.route_class == "answerable" and c.rank <= 50),
            key=lambda c: c.rank,
        )
    )
    if tuple(c.qbank_id for c in candidates) != cell.match_qbank_ids:
        raise ValueError("oracle_saved_pool_mismatch")
    value = OraclePool(
        query=query,
        snapshot_id=snapshot_id,
        query_run_id=query_run_id,
        compiler_run_id=facts.run["source_compiler_run_id"],
        candidates=candidates,
        predicted_questions={
            c.qbank_id: facts.questions[c.qbank_id]["canonical_question"] for c in candidates
        },
        source_identity=cell.model_dump(mode="json"),
        content_hash="pending",
    )
    from dataclasses import replace

    return replace(value, content_hash=canonical_content_hash(value.payload))


class OracleAnswerJudge:
    """The existing method-blind Judge, dispatched only through evaluator audit."""

    def __init__(
        self,
        *,
        execution: OracleModelExecution,
        gateway,
        config: BaselineEvaluationConfig,
        prompt: str,
        query: QueryInput,
        resolution: TypedGoldResolution,
    ):
        if not isinstance(execution, OracleModelExecution):
            raise ValueError("oracle_judge_evaluator_required")
        self.execution = execution
        self.gateway = gateway
        self.config = config
        self.prompt = prompt
        self.query = query
        self.resolution = resolution
        self.identity = {
            "config": config.model_dump(mode="json"),
            "prompt_hash": evaluation_prompt_sha256(prompt),
            "model_hash": execution.execution.registry.resolved_hash_for_aliases(
                [config.judge_model_alias]
            ),
        }

    async def score(self, answer: TypedAnswerResult) -> dict[str, Any]:
        reference = self.resolution.text_reference
        if reference is None or reference.text is None or answer.output.answer_text is None:
            raise ValueError("oracle_judge_reference_missing")
        request = build_judge_request(
            evaluation_run_id=self.execution.evaluation_run_id,
            answer_id=answer.typed_answer_id,
            query=self.query,
            candidate_answer=answer.output.answer_text,
            reference_answer=reference.text,
            prompt=self.prompt,
            config=self.config,
            timeout_seconds=120.0,
        )
        call_id = stable_compiler_id(
            "oracle-judge-call",
            [
                self.execution.evaluation_run_id,
                answer.typed_answer_id,
                self.identity,
                self.resolution.gold_hash,
            ],
        )
        audited = await self.execution.generate_audited(
            self.gateway,
            request.model_copy(update={"request_id": call_id}),
            ModelCallContext(
                model_call_id=call_id,
                run_id=answer.run_id,
                snapshot_id=answer.snapshot_id,
                query_id=answer.query_id,
                conversation_id=self.query.conversation_id,
                evaluation_run_id=self.execution.evaluation_run_id,
                stage=AppStage.JUDGE,
            ),
            output_parser=lambda result: parse_judge_output(result.text),
        )
        score = judge_score_from_trace(
            trace=audited.trace,
            evaluation_run_id=self.execution.evaluation_run_id,
            answer_id=answer.typed_answer_id,
            query_id=answer.query_id,
            run_id=answer.run_id,
            config=self.config,
        )
        return score.model_dump(mode="json")


async def evaluate_oracle_at_50(
    *,
    scope: DataScope,
    pool: OraclePool,
    programs: Mapping[str, OracleProgramInput],
    runner: OracleProgramRunner,
    resolution: TypedGoldResolution,
    judge: OracleAnswerJudge | None = None,
) -> dict[str, Any]:
    """Return a private, persisted report; incomplete denominators have no accuracy.

    Text uses the established Judge grade (accuracy requires grade 1); non-text
    uses typed exact match. Ties choose lower saved rank. No proxy participates.
    An empty *verified* retrieval pool is a zero, not an engineering omission.
    """
    require_evaluator(scope)
    if (
        pool.content_hash != canonical_content_hash(pool.payload)
        or len(pool.candidates) > 50
        or [c.rank for c in pool.candidates] != list(range(1, len(pool.candidates) + 1))
        or resolution.query_id != pool.query.query_id
        or resolution.query_hash != stable_hash(pool.query.model_dump(mode="json"))
        or runner.source_run_id != pool.query_run_id
    ):
        raise ValueError("oracle_pool_or_reference_identity_mismatch")
    if judge is not None and (
        judge.resolution != resolution
        or judge.query != pool.query
        or judge.execution is not runner.answer_service.execution
    ):
        raise ValueError("oracle_judge_owner_or_reference_mismatch")
    if set(programs) - {c.qbank_id for c in pool.candidates}:
        raise ValueError("oracle_program_outside_top50")
    from pmmc.evaluation.step8_workflow import typed_candidate

    rows: list[dict[str, Any]] = []
    for candidate in pool.candidates:
        row: dict[str, Any] = {
            "qbank_id": candidate.qbank_id,
            "plan_id": candidate.plan_id,
            "rank": candidate.rank,
            "candidate_hash": candidate.content_hash,
            "status": "incomplete",
            "score": None,
            "typed_score": None,
            "judge": None,
        }
        rows.append(row)
        source = programs.get(candidate.qbank_id)
        if source is None:
            row["error_code"] = "oracle_program_missing"
            continue
        source.validate(
            compiler_run_id=pool.compiler_run_id,
            snapshot_id=pool.snapshot_id,
            qbank_id=candidate.qbank_id,
            plan_id=candidate.plan_id,
        )
        if source.predicted_question != pool.predicted_questions[
            candidate.qbank_id
        ] or source.source_program_hash != candidate.metadata.get(
            "source_program_hash", source.source_program_hash
        ):
            raise ValueError("oracle_frozen_program_identity_mismatch")
        row["program_input_hash"] = canonical_content_hash(source.payload)
        try:
            fact = await runner.run(
                query=pool.query,
                snapshot_id=pool.snapshot_id,
                source=source,
                candidate_identity={
                    "pool_hash": pool.content_hash,
                    "candidate_hash": candidate.content_hash,
                    "rank": candidate.rank,
                },
            )
            row["fact_sha256"] = runner.journal.store.put_json(fact)
            if fact["status"] != "completed":
                row["error_code"] = fact["error_code"]
                continue
            answer = TypedAnswerResult.model_validate(fact["answer"])
            if (
                answer.run_id != pool.query_run_id
                or answer.query_id != pool.query.query_id
                or answer.snapshot_id != pool.snapshot_id
                or not answer.answer_variant.startswith("oracle-at-50-")
                or not answer.calls
            ):
                raise ValueError("oracle_answer_lineage_mismatch")
            row.update(
                typed_answer_id=answer.typed_answer_id,
                terminal_state=answer.terminal_state,
                answer_hash=canonical_content_hash(runtime_fact_payload(answer)),
            )
            if resolution.reference is None:
                row["error_code"] = resolution.unavailable_reason
                continue
            if answer.terminal_state == "abstained":
                row.update(status="completed", score=0.0, score_kind="actual_abstention")
                continue
            if answer.terminal_state != "answered":
                row["error_code"] = "oracle_unexpected_fallback"
                continue
            typed = typed_candidate(answer)
            if typed is None:
                raise ValueError("oracle_answered_without_candidate")
            scored = score_resolved_typed_answer(typed, resolution)
            row["typed_score"] = asdict(scored)
            if not scored.applicable:
                row["error_code"] = scored.unavailable_reason
                continue
            value = scored.value
            row["score_kind"] = "typed_exact_match"
            if (
                typed.kind == "text"
                and resolution.reference.kind == "text"
                and resolution.text_reference is not None
            ):
                if judge is None:
                    row["error_code"] = "oracle_judge_missing"
                    continue
                judged = await judge.score(answer)
                value = judged["llm_judge_score"]
                if (
                    type(value) not in {int, float}
                    or value not in {0.0, 0.5, 1.0}
                    or not judged.get("model_call_id")
                ):
                    raise ValueError("oracle_judge_receipt_invalid")
                row.update(
                    judge={
                        key: judged[key]
                        for key in (
                            "model_call_id",
                            "evaluation_run_id",
                            "answer_id",
                            "query_id",
                            "judge_model_alias",
                            "judge_actual_model_id",
                            "llm_judge_score",
                        )
                    },
                    score_kind="method_blind_judge",
                )
            row.update(status="completed", score=value)
        except Exception as exc:
            row.update(error_code="oracle_candidate_incomplete", error_type=type(exc).__name__)
    completed = [row for row in rows if row["status"] == "completed"]
    complete = len(completed) == len(rows) and resolution.reference is not None
    best = max(completed, key=lambda row: (row["score"], -row["rank"])) if completed else None
    payload = {
        "protocol": VERSION,
        "metric": "Oracle@50",
        "scope": "saved_retrieved_top50_only",
        "evaluation_run_id": runner.evaluation_run_id,
        "pool_hash": pool.content_hash,
        "source_identity": pool.source_identity,
        "query_run_id": pool.query_run_id,
        "query_id": pool.query.query_id,
        "snapshot_id": pool.snapshot_id,
        "runner_identity": runner.identity,
        "judge_identity": judge.identity if judge else None,
        "reference_identity": {
            **resolution.audit_metadata(),
            "gold_hash": resolution.gold_hash,
            "query_hash": resolution.query_hash,
        },
        "selection_policy": "max_actual_answer_score_then_saved_rank",
        "cost_scope": "research_only_evaluator_oracle",
        "status": "completed" if complete else "incomplete",
        "query_denominator": 1,
        "candidate_denominator": len(rows),
        "completed_candidate_count": len(completed),
        "candidates": rows,
        "selected_qbank_id": best["qbank_id"] if complete and best else None,
        "selected_typed_answer_id": best["typed_answer_id"] if complete and best else None,
        "oracle_score": (best["score"] if best else 0.0) if complete else None,
        "oracle_accuracy": float(best is not None and best["score"] == 1.0) if complete else None,
        "unavailable_reason": None if complete else "actual_candidate_denominator_incomplete",
    }
    report = {**payload, "content_hash": canonical_content_hash(payload)}
    return {**report, "artifact_sha256": runner.journal.store.put_json(report)}
