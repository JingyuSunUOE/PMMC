"""Per-query/cell execution facade for the optional Step8 Oracle evaluator.

The production loader builds this only for the cell declared in the saved
protocol. It reuses the pipeline's registered evaluator and private model audit;
an undeclared campaign incurs no Oracle retrieval, execution or model cost.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.step8_oracle import (
    OracleAnswerJudge,
    OraclePool,
    evaluate_oracle_at_50,
)
from pmmc.evaluation.step8_oracle_execution import (
    OracleJournal,
    OracleModelExecution,
    OracleProgramInput,
    OracleProgramRunner,
)
from pmmc.experiments.step8_artifacts import json_bytes
from pmmc.memory.operators import LinearProgramExecutor
from pmmc.models.execution import ModelExecutionService
from pmmc.models.protocols import GenerationGateway, TokenCounter
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.answering import SharedAnswerGenerationService
from pmmc.runtime.grounding import AuditedGroundingService, load_step8_prompt
from pmmc.schemas import AssetType, DataScope


def oracle_not_evaluated(reason: str = "oracle_not_opted_in") -> dict[str, Any]:
    return {
        "status": "not_evaluated",
        "metric": "Oracle@50",
        "oracle_accuracy": None,
        "oracle_score": None,
        "unavailable_reason": reason,
    }


@dataclass(frozen=True)
class OracleEvaluationHook:
    pool: OraclePool
    programs: Mapping[str, OracleProgramInput]
    executor: LinearProgramExecutor
    answer_inputs: PmmcEvidenceAnswerInputProvider
    answer_gateway: GenerationGateway
    grounding_gateway: GenerationGateway
    grounding_counter: TokenCounter
    journal: OracleJournal
    protocol_identity: dict[str, Any]

    def selected(self, query_run_id: str, query_id: str) -> bool:
        return (self.pool.query_run_id, self.pool.query.query_id) == (query_run_id, query_id)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "pool_hash": self.pool.content_hash,
            "program_inputs": {
                key: canonical_content_hash(value.payload)
                for key, value in sorted(self.programs.items())
            },
            "protocol": self.protocol_identity,
            "answer_alias": self.answer_inputs.answer_model_alias,
            "answer_prompt_hash": self.answer_inputs.instruction_hash,
            "grounding_prompt_hash": canonical_content_hash(
                load_step8_prompt("answer_grounding_v1.txt")
            ),
            "context_tokens": self.answer_inputs.total_context_tokens,
            "raw_images": self.answer_inputs.allow_memory_raw_images,
            "foundation_hash": canonical_content_hash(self.executor.config.model_dump(mode="json")),
            "implementation_hashes": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "step8_oracle.py",
                    "step8_oracle_execution.py",
                    "step8_oracle_hook.py",
                    "step8_oracle_production.py",
                )
            },
        }

    async def run(
        self,
        *,
        evaluation_run_id,
        execution,
        judge_gateway,
        judge_config,
        judge_prompt,
        resolution,
        artifact_service=None,
    ):
        if artifact_service is not None and artifact_service.scope is not DataScope.EVALUATION:
            raise ValueError("oracle_private_artifacts_require_evaluator")
        wrapper = OracleModelExecution(
            execution,
            evaluation_run_id=evaluation_run_id,
            source_run_id=self.pool.query_run_id,
            journal=self.journal,
        )
        # The adapter implements generate_audited while enforcing evaluator ownership.
        evaluator = cast(ModelExecutionService, wrapper)
        grounding = AuditedGroundingService(
            execution=evaluator,
            gateway=self.grounding_gateway,
            counter=self.grounding_counter,
        )
        answers = SharedAnswerGenerationService(
            execution=evaluator,
            gateway=self.answer_gateway,
            grounding=grounding,
        )
        runner = OracleProgramRunner(
            scope=DataScope.EVALUATION,
            executor=self.executor,
            answer_inputs=self.answer_inputs,
            answer_service=answers,
            journal=self.journal,
            protocol_identity=self.protocol_identity,
        )
        judge = OracleAnswerJudge(
            execution=wrapper,
            gateway=judge_gateway,
            config=judge_config,
            prompt=judge_prompt,
            query=self.pool.query,
            resolution=resolution,
        )
        report = await evaluate_oracle_at_50(
            scope=DataScope.EVALUATION,
            pool=self.pool,
            programs=self.programs,
            runner=runner,
            resolution=resolution,
            judge=judge,
        )
        # Upload raw facts separately: campaign receipts must not carry raw answers.
        bundle = {
            "report": report,
            "private_facts": {
                row["fact_sha256"]: self.journal.store.load_json(row["fact_sha256"])
                for row in report["candidates"]
                if "fact_sha256" in row
            },
            "program_inputs": {key: value.payload for key, value in self.programs.items()},
        }
        digest = self.journal.store.put_json(bundle)
        reference = None
        if artifact_service is not None:
            artifact = await artifact_service.put_bytes(
                artifact_id="step8-oracle-facts-" + digest[:32],
                run_id=self.pool.query_run_id,
                evaluation_run_id=evaluation_run_id,
                artifact_type=AssetType.TRACE_EXPORT,
                data=json_bytes(bundle),
                mime_type="application/json",
                extension="json",
                snapshot_id=self.pool.snapshot_id,
                query_id=self.pool.query.query_id,
            )
            reference = {
                key: getattr(artifact, key)
                for key in (
                    "artifact_id",
                    "sha256",
                    "object_key",
                    "storage_uri",
                )
            }
        return {**report, "private_bundle_sha256": digest, "private_artifact": reference}


def summarize_oracle_reports(
    reports: Sequence[dict[str, Any]],
    *,
    selected_query_ids: Sequence[str],
) -> dict[str, Any]:
    """Strict selected-query denominator, never a successful-only or proxy mean."""
    if len(selected_query_ids) != len(set(selected_query_ids)):
        raise ValueError("oracle_duplicate_selected_query")
    by_query = {report["query_id"]: report for report in reports}
    if len(by_query) != len(reports) or set(by_query) - set(selected_query_ids):
        raise ValueError("oracle_report_denominator_mismatch")

    def complete_report(report):
        if not isinstance(report, dict):
            return False
        denominator = report.get("candidate_denominator")
        rows = report.get("candidates")
        if (
            report.get("status") != "completed"
            or report.get("protocol") != "step8-executed-oracle-at-50-v1"
            or type(denominator) is not int
            or not 0 <= denominator <= 50
            or not isinstance(rows, list)
            or len(rows) != denominator
            or report.get("completed_candidate_count") != denominator
            or report.get("query_denominator") != 1
            or report.get("content_hash")
            != canonical_content_hash(
                {
                    key: value
                    for key, value in report.items()
                    if key
                    not in {
                        "content_hash",
                        "artifact_sha256",
                        "private_bundle_sha256",
                        "private_artifact",
                    }
                }
            )
        ):
            return False
        if any(
            not isinstance(row, dict)
            or row.get("status") != "completed"
            or type(row.get("score")) not in {int, float}
            or row["score"] not in {0, 0.5, 1}
            or not row.get("fact_sha256")
            or not row.get("typed_answer_id")
            for row in rows
        ):
            return False
        best = max((row["score"] for row in rows), default=0.0)
        return (
            [row["rank"] for row in rows] == list(range(1, denominator + 1))
            and len({row["qbank_id"] for row in rows}) == denominator
            and type(report.get("oracle_accuracy")) in {int, float}
            and report["oracle_accuracy"] == float(best == 1)
            and report.get("oracle_score") == best
        )

    missing = [
        query_id for query_id in selected_query_ids if not complete_report(by_query.get(query_id))
    ]
    denominator = len(selected_query_ids)
    complete = denominator > 0 and not missing
    return {
        "metric": "Oracle@50",
        "denominator": denominator,
        "status": "completed" if complete else "incomplete" if denominator else "not_evaluated",
        "oracle_accuracy": sum(by_query[q]["oracle_accuracy"] for q in selected_query_ids)
        / denominator
        if complete
        else None,
        "missing_query_ids": missing,
        "unavailable_reason": None if complete else "actual_oracle_denominator_incomplete",
    }
