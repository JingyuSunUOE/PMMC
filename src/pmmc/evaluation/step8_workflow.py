"""Evaluator-only Step 8 adapter, independent of the legacy answer workflow.

Scores live in immutable evaluator artifacts because legacy metric/Judge tables
require final_answers foreign keys. No placeholder final-answer rows are created.
The existing method-blind Judge request/parser and audited execution service are
reused; typed images are never submitted to the text Judge as placeholder strings.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert

from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation import (
    paper_statistics,
    step8_process_metrics,
    step8_relevance_grid,
    step8_semantic_workflow,
    step8_typed_gold,
)
from pmmc.evaluation.config import (
    evaluation_prompt_sha256,
    load_evaluation_config,
    load_judge_prompt,
)
from pmmc.evaluation.judge import build_judge_request, judge_score_from_trace, parse_judge_output
from pmmc.evaluation.paper_statistics import (
    MethodVariant,
    ScoreObservation,
    SelectedQuery,
    TerminalStatus,
    TypedAnswer,
    score_typed_answer,
    summarize_method,
)
from pmmc.evaluation.repositories import PostgresGoldRepository
from pmmc.evaluation.step8_oracle_hook import OracleEvaluationHook, oracle_not_evaluated
from pmmc.evaluation.step8_oracle_production import prepare_configured_oracle, select_oracle_cell
from pmmc.evaluation.step8_typed_gold import (
    UNRESOLVED,
    TypedGoldResolution,
    resolve_typed_gold,
    score_resolved_typed_answer,
    text_judge_reference,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.evaluation_models import (
    BaselineSuiteModel,
    BaselineSuiteQueryModel,
    BaselineSuiteRunModel,
    EvaluationRunModel,
    EvaluationRunStateModel,
)
from pmmc.infrastructure.database.experiment_models import (
    EvaluationArtifactModel,
    RunQueryModel,
    RunStateModel,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.query_executor_models import (
    QueryExecutorRunModel,
    QueryExperimentCellResultModel,
    QueryRuntimeTypedAnswerModel,
)
from pmmc.infrastructure.database.repositories import (
    PostgresMemoryRepository,
    PostgresRuntimeQueryRepository,
)
from pmmc.infrastructure.database.step8_models import (
    BenchmarkCampaignModel,
    BenchmarkCellAssignmentModel,
    ExperimentCellModel,
)
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_evaluator_object_stores
from pmmc.infrastructure.settings import Settings
from pmmc.models import ModelCallContext
from pmmc.models.factory import (
    create_generation_gateway,
    create_image_url_resolver,
    create_model_execution_service,
)
from pmmc.models.registry import load_model_registry
from pmmc.schemas import AppStage, AssetType, DataScope, QueryGold, QueryInput
from pmmc.schemas.answer_output import TypedAnswerResult
from pmmc.schemas.base import utc_now
from pmmc.schemas.query_executor import QueryExperimentCellResult, runtime_fact_payload
from pmmc.schemas.step8 import BASELINE_METHODS, RELEVANCE_THRESHOLDS, OracleAt50Protocol

_VERSION = "step8-query-evaluation-v1"
_CONTRACT = "typed-answer-contract-v3"
MethodScope = Literal["all", "pmmc", "pmmc_main", "paired_main", "baseline"]
_SCOPES = {"all", "pmmc", "pmmc_main", "paired_main", "baseline"}
_VARIANTS = {
    "active": "pmmc",
    "no_fallback": "no_fallback",
    "forced_top1": "forced_top1_counterfactual",
}


@dataclass(frozen=True)
class EvaluationCellInput:
    cell_id: str
    method: str
    variant: str
    source_variant: str
    terminal_state: str
    source_run_id: str
    source_fact_id: str | None
    source_fact_hash: str | None
    answer: TypedAnswerResult | None
    error_code: str | None = None
    question_budget: int | None = None
    round_id: int | None = None


@dataclass(frozen=True)
class Step8EvaluationInputs:
    query: QueryInput
    snapshot_id: str
    protocol_hash: str
    judge_alias: str
    cells: tuple[EvaluationCellInput, ...]
    runtime_cells: tuple[QueryExperimentCellResult, ...] = ()
    runtime_answers: tuple[TypedAnswerResult, ...] = ()
    oracle_config: OracleAt50Protocol | None = None
    oracle_cell_id: str | None = None
    primary_thresholds: tuple[float, float] | None = None


def judge_query_applicability(resolution: TypedGoldResolution) -> tuple[bool, str | None]:
    """Reference-side applicability is independent of any candidate or terminal state."""
    image_kinds = {"image", "image_set", "ordered_image_list"}
    if resolution.expected_kind in image_kinds or (
        resolution.reference is not None and resolution.reference.kind in image_kinds
    ):
        return False, "image_typed_gold_not_text_judge"
    if resolution.status in {"missing", "unresolved"}:
        return False, resolution.unavailable_reason
    if resolution.text_reference is None:
        return False, "text_judge_reference_unavailable"
    return True, None


def judge_unavailable_reason(
    *,
    applicable: bool,
    gold_reason: str | None,
    terminal_state: str,
    candidate_kind: str | None,
    judge_present: bool,
) -> str | None:
    if not applicable:
        return gold_reason
    if terminal_state in {"abstained", "no_answer"}:
        return "answer_abstained"
    if terminal_state != "answered" or candidate_kind is None:
        return "answer_engineering_missing"
    if candidate_kind != "text":
        return "candidate_not_text"
    return None if judge_present else "judge_engineering_missing"


class Step8EvaluationIncomplete(RuntimeError):
    """The persisted report is inspectable, but must not complete a campaign job."""

    def __init__(self, result: dict[str, Any]):
        super().__init__("step8_evaluation_incomplete")
        self.result = result


def typed_candidate(answer: TypedAnswerResult) -> TypedAnswer | None:
    output = answer.output
    kind = output.answer_kind.value
    if kind == "abstention":
        return None
    if kind in {"image", "image_set", "ordered_image_list"}:
        # Only selected canonical image items count, never used evidence aliases.
        return TypedAnswer(
            cast(Literal["image", "image_set", "ordered_image_list"], kind),
            image_ids=tuple(item.image_id for item in answer.image_items),
        )
    if kind == "number":
        assert output.answer_text is not None
        return TypedAnswer("number", number=Decimal(output.answer_text))
    if kind == "boolean":
        return TypedAnswer("boolean", boolean=output.answer_text == "true")
    return TypedAnswer("text", text=output.answer_text)


def typed_reference(gold: QueryGold | None) -> TypedAnswer | None:
    """Read explicit typed gold, never mistake evidence images for answer images.

    Typed annotations use metadata.typed_answer with TypedAnswer's field names.
    Legacy scalar answer strings stay text; they do not establish image identity.
    """
    if gold is None:
        return None
    explicit = gold.metadata.get("typed_answer")
    if explicit is None:
        return TypedAnswer("text", text=gold.answer) if gold.answer else None
    if not isinstance(explicit, dict):
        raise ValueError("step8_typed_gold_invalid")
    payload = dict(explicit)
    if "image_ids" in payload:
        payload["image_ids"] = tuple(payload["image_ids"])
    if payload.get("number") is not None:
        payload["number"] = Decimal(str(payload["number"]))
    if payload.get("date_value") is not None:
        payload["date_value"] = date.fromisoformat(payload["date_value"])
    return TypedAnswer(**payload)


def _compatible_scalar(candidate: TypedAnswer, reference: TypedAnswer) -> TypedAnswer:
    # Legacy scalar references are strings, but numeric/boolean values have exact
    # lossless parsers. Image references require explicit evaluator annotations.
    if reference.kind != "text":
        return reference
    text = (reference.text or "").strip()
    if candidate.kind == "number" and re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", text
    ):
        return TypedAnswer("number", number=Decimal(text))
    if candidate.kind == "boolean" and text.casefold() in {"true", "false"}:
        return TypedAnswer("boolean", boolean=text.casefold() == "true")
    return reference


def score_cell(
    query_id: str,
    cell: EvaluationCellInput,
    reference: TypedAnswer | None,
) -> ScoreObservation:
    method = MethodVariant(cell.method, cell.variant)
    if cell.terminal_state in {"abstained", "no_answer"}:
        return ScoreObservation(query_id, method, TerminalStatus.ABSTENTION)
    if cell.terminal_state != "answered" or cell.answer is None:
        return ScoreObservation(query_id, method, TerminalStatus.ENGINEERING_MISSING)
    candidate = typed_candidate(cell.answer)
    if candidate is None:
        raise ValueError("step8_answer_terminal_mismatch")
    score = score_typed_answer(
        candidate, _compatible_scalar(candidate, reference) if reference else None
    )
    return ScoreObservation(query_id, method, TerminalStatus.ANSWERED, score)


def _score_resolved_cell(
    query_id: str, cell: EvaluationCellInput, resolution: TypedGoldResolution
) -> ScoreObservation:
    method = MethodVariant(cell.method, cell.variant)
    if cell.terminal_state in {"abstained", "no_answer"}:
        return ScoreObservation(query_id, method, TerminalStatus.ABSTENTION)
    if cell.terminal_state != "answered" or cell.answer is None:
        return ScoreObservation(query_id, method, TerminalStatus.ENGINEERING_MISSING)
    candidate = typed_candidate(cell.answer)
    if candidate is None:
        raise ValueError("step8_answer_terminal_mismatch")
    return ScoreObservation(
        query_id,
        method,
        TerminalStatus.ANSWERED,
        score_resolved_typed_answer(candidate, resolution),
    )


def _decode_answer(row: QueryRuntimeTypedAnswerModel) -> TypedAnswerResult:
    payload = row.payload.get("metadata", {}).get("typed_result")
    if not isinstance(payload, dict) or canonical_content_hash(payload) != row.content_hash:
        raise ValueError("step8_typed_answer_hash_mismatch")
    answer = TypedAnswerResult.model_validate(payload)
    if (
        answer.typed_answer_id != row.typed_answer_id
        or answer.run_id != row.query_executor_run_id
        or answer.query_id != row.query_id
        or answer.terminal_state != row.terminal_state
        or answer.answer_model_call_id != row.answer_model_call_id
    ):
        raise ValueError("step8_typed_answer_lineage_mismatch")
    return answer


def _validate_cell_row(row: QueryExperimentCellResultModel) -> None:
    fact = QueryExperimentCellResult.model_validate(row.payload)
    payload = fact.model_dump(mode="json", exclude={"cell_result_id", "content_hash", "created_at"})
    if canonical_content_hash(runtime_fact_payload(payload)) != row.content_hash:
        raise ValueError("step8_cell_result_hash_mismatch")
    if any(
        getattr(fact, key) != getattr(row, key)
        for key in (
            "cell_result_id",
            "query_executor_run_id",
            "query_id",
            "cell_id",
            "terminal_state",
            "selected_answer_id",
            "content_hash",
        )
    ) or any(
        getattr(fact.cell, key) != getattr(row, key)
        for key in (
            "treatment",
            "question_budget",
            "revision_depth",
        )
    ):
        raise ValueError("step8_cell_result_lineage_mismatch")


def bind_cell_inputs(
    *,
    definitions: tuple[dict[str, Any], ...],
    result_rows: tuple[Any, ...],
    answers: dict[str, TypedAnswerResult],
    answer_hashes: dict[str, str],
    query_run_id: str,
    baseline_runs: dict[str, str],
    query_id: str,
) -> tuple[EvaluationCellInput, ...]:
    """Bind the complete assignment universe, never an inner join on successes."""
    result_by_key = {}
    for row in result_rows:
        key = row.treatment, row.question_budget, row.revision_depth
        if key in result_by_key:
            raise ValueError("step8_duplicate_runtime_cell")
        if row.query_executor_run_id != query_run_id or row.query_id != query_id:
            raise ValueError("step8_cell_scope_mismatch")
        result_by_key[key] = row
    bound = []
    for definition in sorted(definitions, key=lambda d: d["cell_id"]):
        method, variant = definition["method"], definition["cell_id"]
        source_id = source_hash = error = None
        answer = None
        terminal = "engineering_missing"
        source_variant = definition["source_variant"]
        run_id = query_run_id if method == "pmmc" else baseline_runs[method]
        if method == "pmmc":
            treatment = _VARIANTS[source_variant]
            if source_variant == "active" and definition["treatment"] != "main":
                treatment = definition["treatment"]
            key = (treatment, definition["question_budget"], definition["round_id"])
            result = result_by_key.get(key)
            if result is not None:
                source_id, source_hash = result.cell_result_id, result.content_hash
                terminal = result.terminal_state
                answer = answers.get(result.selected_answer_id)
                if terminal in {"no_answer", "technical_failure"} and result.selected_answer_id:
                    raise ValueError("step8_nonanswer_has_selected_answer")
                if terminal in {"answered", "abstained"} and answer is None:
                    terminal, error = "engineering_missing", "selected_typed_answer_missing"
                if answer is not None and (
                    answer.run_id not in {query_run_id, baseline_runs["native_multimodal_rag"]}
                    or answer.query_id != query_id
                    or answer.terminal_state != terminal
                    or (answer.run_id == query_run_id and answer.method != "pmmc")
                    or (
                        answer.run_id != query_run_id
                        and (
                            answer.method != "native_multimodal_rag"
                            or answer.answer_variant != "primary"
                            or source_variant in {"no_fallback", "forced_top1"}
                        )
                    )
                ):
                    raise ValueError("step8_selected_answer_scope_mismatch")
        else:
            source_variant = "primary"
            candidates = [
                a
                for a in answers.values()
                if a.run_id == run_id
                and a.query_id == query_id
                and a.answer_variant == source_variant
            ]
            if len(candidates) > 1:
                raise ValueError("step8_duplicate_baseline_variant")
            if candidates:
                answer = candidates[0]
                if answer.method != method or answer.terminal_state not in {
                    "answered",
                    "abstained",
                }:
                    raise ValueError("step8_baseline_terminal_mismatch")
                terminal = answer.terminal_state
                source_id = answer.typed_answer_id
                source_hash = answer_hashes[source_id]
        bound.append(
            EvaluationCellInput(
                variant,
                method,
                variant,
                source_variant,
                terminal,
                run_id,
                source_id,
                source_hash,
                answer,
                error,
                definition["question_budget"],
                definition["round_id"],
            )
        )
    if not bound or len({c.cell_id for c in bound}) != len(bound):
        raise ValueError("step8_invalid_selected_cell_denominator")
    return tuple(bound)


def evaluation_definitions(cells, treatment, method_scope, *, selected=False, standalone_method=None):
    """Validate the frozen assignments before selecting an explicit execution scope."""
    if method_scope not in _SCOPES:
        raise ValueError("step8_evaluation_scope_invalid")
    if selected and method_scope not in {"pmmc_main", "pmmc", "baseline"}:
        raise ValueError("step8_selected_campaign_cannot_claim_paired_evaluation")
    definitions = tuple(
        {**row.payload, "cell_id": row.cell_id}
        for row in cells
        if (row.method != "pmmc" and treatment == "main")
        or (row.method == "pmmc" and row.payload["treatment"] == treatment)
    )
    if method_scope == "baseline":
        matches = tuple(d for d in definitions if d["method"] == standalone_method)
        if (not selected or standalone_method not in BASELINE_METHODS or len(matches) != 1
                or treatment != "main"):
            raise ValueError("step8_standalone_assignment_missing_or_ambiguous")
        return matches
    if not any(d["method"] == "pmmc" for d in definitions) or not selected and {
        d["method"] for d in definitions if d["method"] != "pmmc"
    } != (set(BASELINE_METHODS) if treatment == "main" else set()):
        raise ValueError("step8_paired_cell_assignments_missing")
    if method_scope in {"pmmc_main", "paired_main"}:
        primary = tuple(d for d in definitions if (method_scope == "paired_main" or d["method"] == "pmmc")
                        and d["source_variant"] == "active" and d["treatment"] == "main"
                        and (d["question_budget"], d["round_id"]) == (8, 2))
        methods = {"pmmc", *BASELINE_METHODS} if method_scope == "paired_main" else {"pmmc"}
        if len(primary) != len(methods) or {d["method"] for d in primary} != methods:
            raise ValueError("exploration_primary_assignment_missing_or_ambiguous")
        return primary
    return tuple(d for d in definitions if method_scope == "all" or d["method"] == "pmmc")


def validate_baseline_links(baseline_runs, treatment, method_scope):
    if method_scope not in _SCOPES:
        raise ValueError("step8_evaluation_scope_invalid")
    expected = {"native_multimodal_rag"} if treatment == "no_raw_image" else set(BASELINE_METHODS)
    actual = set(baseline_runs)
    if (
        not actual.issubset(expected)
        or "native_multimodal_rag" not in actual
        or (method_scope in {"all", "paired_main"} and actual != expected)
    ):
        raise ValueError("step8_baseline_link_contract_mismatch")


async def _load_inputs(
    sessions, campaign_id: str, run_id: str, suite_id: str, query_id: str,
    *, method_scope: MethodScope = "all",
) -> Step8EvaluationInputs:
    async with sessions() as session:
        campaign = await session.get(BenchmarkCampaignModel, campaign_id)
        suite = await session.get(BaselineSuiteModel, suite_id)
        if method_scope == "baseline":
            return await _load_standalone_inputs(sessions, session, campaign, suite, run_id, query_id)
        run = await session.get(QueryExecutorRunModel, run_id)
        if campaign is None or run is None or suite is None:
            raise ValueError("step8_evaluation_parent_missing")
        primary = method_scope == "pmmc_main"
        if primary != (run.metadata_json.get("readout") == "exploration-primary-b8-r2") or (
            primary and (campaign.phase != "exploration" or run.metadata_json.get("execute_full_grid")
                         is not False or run.metadata_json.get("declared_cells") != [[8, 2]])
        ):
            raise ValueError("exploration_primary_scope_requires_its_declared_runtime_readout")
        if method_scope == "paired_main" and (
            campaign.phase != "exploration" or run.metadata_json.get("treatment") != "main"
        ):
            raise ValueError("exploration_paired_scope_requires_main_treatment")
        if (
            run.schema_version != "0.3.0" or run.status != "completed"
            or run.snapshot_id != suite.snapshot_id
            or run.metadata_json.get("answer_contract_version") != _CONTRACT
            or suite.metadata_json.get("answer_contract_version") != _CONTRACT
            or run.metadata_json.get("answer_model_alias") != suite.answer_model_alias
            or run.metadata_json.get("answer_prompt_hash") != suite.answer_prompt_hash
            or suite.metadata_json.get("method_protocol_hash") != campaign.protocol_hash
        ):
            raise ValueError("step8_paired_contract_mismatch")
        runtime_ids = set(
            (
                await session.scalars(
                    select(RunQueryModel.query_id).where(RunQueryModel.run_id == run_id)
                )
            ).all()
        )
        suite_ids = set(
            (
                await session.scalars(
                    select(BaselineSuiteQueryModel.query_id).where(
                        BaselineSuiteQueryModel.suite_id == suite_id
                    )
                )
            ).all()
        )
        if query_id not in runtime_ids or runtime_ids != suite_ids:
            raise ValueError("step8_paired_query_denominator_mismatch")
        links = (
            await session.scalars(
                select(BaselineSuiteRunModel).where(BaselineSuiteRunModel.suite_id == suite_id)
            )
        ).all()
        baseline_runs = {row.method: row.run_id for row in links}
        treatment = run.metadata_json.get("treatment")
        validate_baseline_links(baseline_runs, treatment, method_scope)
        if campaign.payload.get("release_plan_hash") and (
            suite.metadata_json.get("release_plan_hash") != campaign.payload["release_plan_hash"]
            or suite.metadata_json.get("release_campaign_id") != campaign_id
            or suite.metadata_json.get("internal_pmmc_fallback") is not True
            or set(baseline_runs) != {"native_multimodal_rag"}
        ):
            raise ValueError("step8_selected_pmmc_fallback_suite_mismatch")
        if treatment == "no_raw_image" and (
            suite.metadata_json.get("allow_memory_raw_images") is not False
            or not suite.metadata_json.get("internal_pmmc_fallback")
        ):
            raise ValueError("step8_caption_only_fallback_contract_missing")
        if run.metadata_json.get("fallback_run_id") != baseline_runs["native_multimodal_rag"]:
            raise ValueError("step8_fallback_suite_mismatch")
        cells = (
            await session.scalars(
                select(ExperimentCellModel)
                .join(
                    BenchmarkCellAssignmentModel,
                    BenchmarkCellAssignmentModel.cell_id == ExperimentCellModel.cell_id,
                )
                .where(
                    BenchmarkCellAssignmentModel.campaign_id == campaign_id,
                    BenchmarkCellAssignmentModel.query_id == query_id,
                    ExperimentCellModel.dataset == run.dataset,
                    ExperimentCellModel.answer_model == suite.answer_model_alias,
                )
            )
        ).all()
        definitions = evaluation_definitions(cells, treatment, method_scope,
                                             selected=bool(campaign.payload.get("release_plan_hash")))
        if any(row.protocol_hash != campaign.protocol_hash for row in cells):
            raise ValueError("step8_cell_protocol_mismatch")
        result_query = select(QueryExperimentCellResultModel).where(
            QueryExperimentCellResultModel.query_executor_run_id == run_id,
            QueryExperimentCellResultModel.query_id == query_id,
        )
        narrow = method_scope in {"pmmc_main", "paired_main"}
        if narrow:
            result_query = result_query.where(
                QueryExperimentCellResultModel.treatment == "pmmc",
                QueryExperimentCellResultModel.question_budget == 8,
                QueryExperimentCellResultModel.revision_depth == 2,
            )
        results = tuple((await session.scalars(result_query)).all())
        for row in results:
            _validate_cell_row(row)
        answer_runs = (list(baseline_runs.values()) if method_scope in {"all", "paired_main"}
                       else [baseline_runs["native_multimodal_rag"]])
        answer_query = select(QueryRuntimeTypedAnswerModel).where(
            QueryRuntimeTypedAnswerModel.query_executor_run_id.in_([run_id, *answer_runs]),
            QueryRuntimeTypedAnswerModel.query_id == query_id,
        )
        if narrow:
            active_ids = {answer_id for row in results for answer_id in (
                row.payload.get("pmmc_answer_id"), row.selected_answer_id
            ) if answer_id}
            answer_query = answer_query.where(or_(
                QueryRuntimeTypedAnswerModel.query_executor_run_id != run_id,
                QueryRuntimeTypedAnswerModel.typed_answer_id.in_(active_ids),
            ))
        rows = (await session.scalars(answer_query)).all()
        answers = {row.typed_answer_id: _decode_answer(row) for row in rows}
        if any(a.snapshot_id != run.snapshot_id for a in answers.values()):
            raise ValueError("step8_answer_snapshot_mismatch")
        if len({a.actual_model_id for a in answers.values()}) > 1:
            raise ValueError("step8_answer_model_identity_mismatch")
        query = await PostgresRuntimeQueryRepository(sessions).get(run.snapshot_id, query_id)
        oracle_payload = campaign.payload["protocol"].get("oracle_at_50")
        oracle_config = (
            OracleAt50Protocol.model_validate(oracle_payload)
            if oracle_payload is not None and method_scope not in {"pmmc_main", "paired_main"}
            else None
        )
        return Step8EvaluationInputs(
            query,
            run.snapshot_id,
            campaign.protocol_hash,
            campaign.payload["protocol"]["judge_alias"],
            bind_cell_inputs(
                definitions=definitions,
                result_rows=results,
                answers=answers,
                answer_hashes={r.typed_answer_id: r.content_hash for r in rows},
                query_run_id=run_id,
                baseline_runs=baseline_runs,
                query_id=query_id,
            ),
            tuple(QueryExperimentCellResult.model_validate(row.payload) for row in results),
            tuple(answer for answer in answers.values() if answer.run_id == run_id),
            oracle_config,
            select_oracle_cell(
                oracle_config,
                definitions,
                answer_model=suite.answer_model_alias,
                treatment=treatment,
            ),
            (campaign.payload["protocol"]["text_relevance_threshold"],
             campaign.payload["protocol"]["multimodal_relevance_threshold"]) if primary else None,
        )


async def _load_standalone_inputs(sessions, session, campaign, suite, run_id, query_id):
    """Bind the actual completed suite child without manufacturing a PMMC run."""
    from pmmc.schemas.baseline import BaselineMethod
    from pmmc.schemas.baseline_suite import is_release_baseline_suite

    if (campaign is None or suite is None or campaign.phase != "exploration"
            or not is_release_baseline_suite([BaselineMethod(m) for m in suite.methods], suite.metadata_json)
            or suite.metadata_json["release_plan_hash"] != campaign.payload.get("release_plan_hash")
            or suite.metadata_json.get("release_campaign_id") != campaign.campaign_id
            or suite.metadata_json.get("method_protocol_hash") != campaign.protocol_hash
            or campaign.payload["snapshot_ids"].get(suite.dataset) != suite.snapshot_id):
        raise ValueError("step8_standalone_suite_contract_mismatch")
    method, = suite.methods
    link = await session.scalar(select(BaselineSuiteRunModel).where(
        BaselineSuiteRunModel.suite_id == suite.suite_id, BaselineSuiteRunModel.method == method))
    state = await session.get(RunStateModel, run_id)
    if link is None or link.run_id != run_id or state is None or state.status != "completed":
        raise ValueError("step8_standalone_child_incomplete")
    suite_queries = set((await session.scalars(select(BaselineSuiteQueryModel.query_id).where(
        BaselineSuiteQueryModel.suite_id == suite.suite_id))).all())
    child_queries = set((await session.scalars(select(RunQueryModel.query_id).where(
        RunQueryModel.run_id == run_id))).all())
    if suite_queries != {query_id} or child_queries != suite_queries:
        raise ValueError("step8_standalone_query_denominator_mismatch")
    cells = (await session.scalars(select(ExperimentCellModel).join(
        BenchmarkCellAssignmentModel, BenchmarkCellAssignmentModel.cell_id == ExperimentCellModel.cell_id
    ).where(BenchmarkCellAssignmentModel.campaign_id == campaign.campaign_id,
            BenchmarkCellAssignmentModel.query_id == query_id,
            ExperimentCellModel.protocol_hash == campaign.protocol_hash,
            ExperimentCellModel.dataset == suite.dataset,
            ExperimentCellModel.answer_model == suite.answer_model_alias,
            ExperimentCellModel.method == method))).all()
    definitions = evaluation_definitions(cells, "main", "baseline", selected=True, standalone_method=method)
    rows = (await session.scalars(select(QueryRuntimeTypedAnswerModel).where(
        QueryRuntimeTypedAnswerModel.query_executor_run_id == run_id,
        QueryRuntimeTypedAnswerModel.query_id == query_id))).all()
    answers = {row.typed_answer_id: _decode_answer(row) for row in rows}
    if any(a.snapshot_id != suite.snapshot_id or a.method != method for a in answers.values()):
        raise ValueError("step8_standalone_answer_mismatch")
    query = await PostgresRuntimeQueryRepository(sessions).get(suite.snapshot_id, query_id)
    return Step8EvaluationInputs(query, suite.snapshot_id, campaign.protocol_hash,
        campaign.payload["protocol"]["judge_alias"], bind_cell_inputs(
            definitions=definitions, result_rows=(), answers=answers,
            answer_hashes={r.typed_answer_id: r.content_hash for r in rows},
            query_run_id=run_id, baseline_runs={method: run_id}, query_id=query_id))


class _ModelCallWriter:
    def __init__(self, sessions):
        self.sessions = sessions

    async def add(self, trace):
        async with ExperimentUnitOfWork(self.sessions, DataScope.EVALUATION) as uow:
            return await uow.model_calls.add(trace)


async def _judge_one(
    *,
    sessions,
    execution,
    gateway,
    config,
    prompt,
    settings,
    evaluation_run_id,
    inputs,
    cell,
    reference,
    judge_identity,
):
    answer = cell.answer
    assert answer is not None
    for attempt in range(1, config.job_max_attempts + 1):
        call_id = (
            "step8-judge-"
            + canonical_content_hash([
                judge_identity, answer.typed_answer_id,
                canonical_content_hash(runtime_fact_payload(answer)), reference.text, attempt,
            ])[:32]
        )
        async with ExperimentUnitOfWork(sessions, DataScope.EVALUATION) as uow:
            trace = await uow.model_calls.get(call_id)
        if trace is None:
            request = build_judge_request(
                evaluation_run_id=evaluation_run_id,
                answer_id=answer.typed_answer_id,
                query=inputs.query,
                candidate_answer=answer.output.answer_text,
                reference_answer=reference.text,
                prompt=prompt,
                config=config,
                timeout_seconds=settings.model_request_timeout_seconds,
            ).model_copy(update={"request_id": call_id})
            audited = await execution.generate_audited(
                gateway,
                request,
                ModelCallContext(
                    model_call_id=call_id,
                    run_id=answer.run_id,
                    evaluation_run_id=evaluation_run_id,
                    snapshot_id=inputs.snapshot_id,
                    query_id=inputs.query.query_id,
                    conversation_id=inputs.query.conversation_id,
                    stage=AppStage.JUDGE,
                ),
                output_parser=lambda result: parse_judge_output(result.text),
            )
            trace = audited.trace
        if trace.status.value == "success":
            return judge_score_from_trace(
                trace=trace,
                evaluation_run_id=evaluation_run_id,
                answer_id=answer.typed_answer_id,
                query_id=inputs.query.query_id,
                run_id=answer.run_id,
                config=config,
            )
    raise ValueError("step8_judge_attempts_exhausted")


async def _register(
    sessions, *, evaluation_run_id, config_hash, run_id, inputs, config, prompt_hash, metadata
):
    now = utc_now()
    async with sessions.begin() as session:
        await session.execute(
            insert(EvaluationRunModel)
            .values(
                evaluation_run_id=evaluation_run_id,
                source_kind="baseline_suite" if metadata["method_scope"] == "baseline" else "pmmc_query_run",
                suite_id=metadata["suite_id"] if metadata["method_scope"] == "baseline" else None,
                legacy_source_run_id=run_id,
                protocol_version=_VERSION,
                normalizer_version="typed-exact-preserve-scalars-v1",
                evidence_policy_version="not_evaluated",
                bad_case_rule_version="not_evaluated",
                judge_model_alias=config.judge_model_alias,
                judge_prompt_hash=prompt_hash,
                judge_rubric_version=config.judge_rubric_version,
                judge_repeat_count=1,
                judge_score_values=config.judge_score_values,
                secondary_judge_enabled=False,
                metric_versions={"typed_exact_match": "typed-exact-preserve-scalars-v1"},
                config_hash=config_hash,
                created_at=now,
                metadata=metadata,
            )
            .on_conflict_do_nothing(index_elements=["evaluation_run_id"])
        )
        stored = await session.get(EvaluationRunModel, evaluation_run_id)
        if stored.config_hash != config_hash or stored.metadata_json != metadata:
            raise ValueError("step8_evaluation_identity_conflict")
        await session.execute(
            insert(EvaluationRunStateModel)
            .values(
                evaluation_run_id=evaluation_run_id,
                status="running",
                selected_count=1,
                target_count=len(inputs.cells),
                succeeded_count=0,
                failed_count=0,
                started_at=now,
                metadata={"adapter": _VERSION, "campaign_job_owned": True},
            )
            .on_conflict_do_nothing(index_elements=["evaluation_run_id"])
        )


async def evaluate_step8_query(
    settings: Settings,
    campaign_id: str,
    query_executor_run_id: str,
    suite_id: str,
    query_id: str,
    *,
    oracle_hook: OracleEvaluationHook | None = None,
    method_scope: MethodScope = "all",
) -> dict[str, Any]:
    """Persist every assigned cell in the explicit method scope, without resampling.

    Missing engineering facts or Judge failures raise Step8EvaluationIncomplete
    with ``.result`` after writing a private diagnostic artifact. Gold and raw answer
    text never appear in the returned receipt. The function never grants approval.
    """
    if method_scope not in _SCOPES:
        raise ValueError("step8_evaluation_scope_invalid")
    primary = method_scope == "pmmc_main"
    auxiliary = method_scope not in {"pmmc_main", "paired_main", "baseline"}
    if not auxiliary and oracle_hook is not None:
        raise ValueError("main and paired readouts cannot dispatch an oracle")
    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        inputs = await _load_inputs(
            sessions, campaign_id, query_executor_run_id, suite_id, query_id,
            method_scope=method_scope,
        )
        try:
            gold = await PostgresGoldRepository(sessions).get(inputs.snapshot_id, query_id)
        except KeyError:
            gold = None
        # The evaluator reads the complete canonical cutoff, never a retrieval or
        # gold-evidence subset. Candidate outputs cannot determine reference kinds.
        cutoff = inputs.query.available_after_session
        if cutoff is None:
            raise ValueError("step8_typed_gold_source_cutoff_missing")
        source_memories = [
            memory
            async for memory in PostgresMemoryRepository(sessions).iter_visible(
                inputs.snapshot_id,
                inputs.query.conversation_id,
                cutoff,
            )
        ]
        resolution = resolve_typed_gold(inputs.query, gold, source_memories=source_memories)
        process_bindings = tuple(
            step8_process_metrics.ProcessCellBinding(
                **{
                    name: getattr(cell, name)
                    for name in (
                        "cell_id",
                        "method",
                        "source_run_id",
                        "source_fact_id",
                        "source_fact_hash",
                        "question_budget",
                        "round_id",
                    )
                }
            )
            for cell in inputs.cells
        )
        process_metrics = (
            await step8_process_metrics.evaluate_process_metrics(
                sessions,
                query=inputs.query,
                snapshot_id=inputs.snapshot_id,
                query_run_id=query_executor_run_id,
                cells=process_bindings,
                gold=gold,
                memories=source_memories,
            )
        ).model_dump(mode="json")
        annotation = resolution.reference_annotation(
            inputs.snapshot_id,
            resolver_source_hash=hashlib.sha256(
                Path(step8_typed_gold.__file__).read_bytes()
            ).hexdigest(),
        )
        config = load_evaluation_config(settings.resolved_evaluation_config_path)
        if config.judge_model_alias != inputs.judge_alias:
            raise ValueError("step8_frozen_judge_alias_mismatch")
        prompt = load_judge_prompt(config, settings.project_root)
        registry = load_model_registry(settings.resolved_model_registry_path)
        oracle_report = oracle_not_evaluated(
            "analysis_scope_only" if not auxiliary else "oracle_cell_not_selected"
            if inputs.oracle_config is not None
            else "oracle_not_opted_in"
        )
        if inputs.oracle_config is not None and inputs.oracle_cell_id is not None:
            if oracle_hook is not None:
                raise ValueError("oracle_declared_production_cell_cannot_be_overridden")
            try:
                oracle_hook = await prepare_configured_oracle(
                    sessions=sessions,
                    settings=settings,
                    campaign_id=campaign_id,
                    query_run_id=query_executor_run_id,
                    query=inputs.query,
                    snapshot_id=inputs.snapshot_id,
                    binding=next(c for c in process_bindings if c.cell_id == inputs.oracle_cell_id),
                    declaration=inputs.oracle_config,
                    protocol_hash=inputs.protocol_hash,
                    registry=registry,
                )
            except Exception as exc:
                oracle_report = {
                    **oracle_not_evaluated("oracle_preparation_incomplete"),
                    "status": "incomplete",
                    "error_type": type(exc).__name__,
                    "cell_id": inputs.oracle_cell_id,
                    "declaration": inputs.oracle_config.model_dump(mode="json"),
                }
        semantic_bindings = tuple(c for c in process_bindings if c.method == "pmmc" and auxiliary)
        semantic_prepared = None
        semantic_report: dict[str, Any] = {
            "status": "not_evaluated" if not auxiliary else "not_applicable",
            "reason": "analysis_scope_only" if not auxiliary else "no_pmmc_cells",
        }
        semantic_identity: dict[str, Any] = dict(semantic_report)
        if semantic_bindings:
            try:
                semantic_prepared = await step8_semantic_workflow.prepare_query_semantics(
                    sessions=sessions,
                    query=inputs.query,
                    snapshot_id=inputs.snapshot_id,
                    query_run_id=query_executor_run_id,
                    bindings=semantic_bindings,
                    gold=gold,
                    registry=registry,
                )
                semantic_identity = semantic_prepared.identity
            except Exception:
                # A missing semantic denominator cannot be replaced by proxy hit
                # scores, and raw DB/model messages must not enter receipts.
                semantic_report = {
                    "status": "incomplete",
                    "error_code": "semantic_preparation_failed",
                }
                semantic_identity = dict(semantic_report)
        config_identity = {
            "adapter": _VERSION,
            "method_scope": method_scope,
            **({"standalone_method": inputs.cells[0].method} if method_scope == "baseline" else {}),
            "oracle_identity": (
                {
                    **oracle_hook.identity,
                    "model_hash": registry.resolved_hash_for_aliases(
                        [
                            oracle_hook.answer_inputs.answer_model_alias,
                            "grounding-qwen3.5-9b",
                            config.judge_model_alias,
                        ]
                    ),
                }
                if oracle_hook is not None and oracle_hook.selected(query_executor_run_id, query_id)
                else oracle_report
            ),
            "campaign_id": campaign_id,
            "suite_id": suite_id,
            "source_run_id": query_executor_run_id,
            "query_id": query_id,
            "snapshot_id": inputs.snapshot_id,
            "protocol_hash": inputs.protocol_hash,
            "query_hash": canonical_content_hash(inputs.query.model_dump(mode="json")),
            "gold_hash": canonical_content_hash(gold.model_dump(mode="json") if gold else None),
            "reference_annotation": annotation,
            # SQL verifies this exact hash preimage and its JSON equality. PostgreSQL
            # jsonb::text is not the canonical Python encoding (spaces/Unicode differ).
            "reference_annotation_payload": json.dumps(
                {key: value for key, value in annotation.items() if key != "hash"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "sources": [
                {
                    "cell_id": c.cell_id,
                    "source_id": c.source_fact_id,
                    "source_hash": c.source_fact_hash,
                    "terminal_state": c.terminal_state,
                }
                for c in inputs.cells
            ],
            "judge_config": config.model_dump(mode="json"),
            "judge_prompt_hash": evaluation_prompt_sha256(prompt),
            "judge_model_hash": registry.resolved_hash_for_aliases([config.judge_model_alias]),
            "adapter_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scorer_source_hash": hashlib.sha256(
                Path(paper_statistics.__file__).read_bytes()
            ).hexdigest(),
            "process_metrics_hash": canonical_content_hash(process_metrics),
            "qbank_semantic_identity": semantic_identity,
            "process_metrics_source_hash": hashlib.sha256(
                Path(step8_process_metrics.__file__).read_bytes()
            ).hexdigest(),
            "relevance_grid_identity": {
                "source_hash": hashlib.sha256(
                    Path(step8_relevance_grid.__file__).read_bytes()
                ).hexdigest(),
                "thresholds": list(RELEVANCE_THRESHOLDS),
                "cell_hashes": sorted(cell.content_hash for cell in inputs.runtime_cells),
                "answer_hashes": sorted(
                    canonical_content_hash(runtime_fact_payload(answer))
                    for answer in inputs.runtime_answers
                ),
            },
        }
        config_hash = canonical_content_hash(config_identity)
        # Scope expansion must reuse a real Judge trace, not relabel its original
        # evaluation_run_id or repeat an already successful model call.
        judge_identity = canonical_content_hash({
            key: config_identity[key] for key in (
                "campaign_id", "protocol_hash", "query_hash", "gold_hash", "snapshot_id", "source_run_id",
                "judge_config", "judge_prompt_hash", "judge_model_hash", "adapter_source_hash",
            )
        })
        evaluation_run_id = "step8-evaluation-" + config_hash[:32]
        await _register(
            sessions,
            evaluation_run_id=evaluation_run_id,
            config_hash=config_hash,
            run_id=query_executor_run_id,
            inputs=inputs,
            config=config,
            prompt_hash=config_identity["judge_prompt_hash"],
            metadata=config_identity,
        )
        runtime_store, evaluation_store = create_evaluator_object_stores(settings)
        artifacts = ExperimentArtifactService(
            sessions,
            evaluation_store,
            DataScope.EVALUATION,
            artifact_prefix=settings.experiment_artifact_prefix,
        )
        async with sessions() as session:
            cached = await session.scalar(
                select(EvaluationArtifactModel).where(
                    EvaluationArtifactModel.evaluation_run_id == evaluation_run_id,
                    EvaluationArtifactModel.artifact_id == "step8-scores-" + config_hash[:32],
                    EvaluationArtifactModel.state == "ready",
                )
            )
        if cached is not None:
            payload = await evaluation_store.get_bytes(cached.object_key)
            if hashlib.sha256(payload).hexdigest() != cached.sha256:
                raise ValueError("step8_score_artifact_checksum_mismatch")
            report = json.loads(payload)
            if (
                report.get("identity_hash") != config_hash
                or report.get("evaluation_run_id") != evaluation_run_id
                or report.get("reference_annotation") != annotation
            ):
                raise ValueError("step8_score_artifact_identity_mismatch")
            result = {
                **report,
                "artifact": {
                    "artifact_id": cached.artifact_id,
                    "sha256": cached.sha256,
                    "object_key": cached.object_key,
                    "storage_uri": cached.storage_uri,
                },
            }
            if report["status"] != "complete":
                raise Step8EvaluationIncomplete(result)
            await _finish(
                sessions, evaluation_run_id, len(inputs.cells), report, result["artifact"]
            )
            return result
        execution = gateway = None
        if semantic_prepared is not None:
            try:
                execution = create_model_execution_service(
                    settings.model_copy(
                        update={
                            "model_dev_max_attempts": 1,
                            "model_paper_max_attempts": 1,
                        }
                    ),
                    registry,
                    artifact_service=artifacts,
                    model_calls=_ModelCallWriter(sessions),
                    data_scope=DataScope.EVALUATION,
                )
                gateway = create_generation_gateway(
                    settings,
                    registry,
                    config.judge_model_alias,
                    image_url_resolver=create_image_url_resolver(settings, runtime_store),
                )
                semantic_report = (
                    await step8_semantic_workflow.evaluate_query_semantics(
                        semantic_prepared,
                        sessions=sessions,
                        execution=execution,
                        gateway=gateway,
                    )
                ).model_dump(mode="json")
            except Exception as exc:
                semantic_report = {
                    "status": "incomplete",
                    "error_code": "semantic_evaluation_failed",
                    "error_type": type(exc).__name__,
                }
        judged: dict[str, Any] = {}
        cluster_kind: Literal["conversation", "scenario"] = (
            "scenario" if inputs.query.dataset.value == "memgallery" else "conversation"
        )
        cluster_id = (
            inputs.query.metadata.get("scenario_id")
            if cluster_kind == "scenario"
            else inputs.query.conversation_id
        )
        if not cluster_id:
            raise ValueError("step8_cluster_identity_missing")
        selected = SelectedQuery(
            query_id,
            inputs.query.dataset.value,
            str(cluster_id),
            cluster_kind,
            applicable=resolution.reference is not None,
            strata=tuple(
                (key, str(inputs.query.metadata[key]))
                for key in ("source_category", "scenario_id")
                if inputs.query.metadata.get(key)
            ),
        )
        rows = []
        judge_applicable, judge_gold_reason = judge_query_applicability(resolution)

        async def judge_answer(answer: TypedAnswerResult):
            nonlocal execution, gateway
            key = answer.typed_answer_id
            if key not in judged:
                try:
                    candidate = typed_candidate(answer)
                    reference = text_judge_reference(candidate, resolution) if candidate else None
                    if reference is None:
                        raise ValueError("relevance_grid_judge_reference_missing")
                    if execution is None:
                        execution = create_model_execution_service(
                            settings,
                            registry,
                            artifact_service=artifacts,
                            model_calls=_ModelCallWriter(sessions),
                            data_scope=DataScope.EVALUATION,
                        )
                        gateway = create_generation_gateway(
                            settings,
                            registry,
                            config.judge_model_alias,
                            image_url_resolver=create_image_url_resolver(settings, runtime_store),
                        )
                    judged[key] = await _judge_one(
                        sessions=sessions,
                        execution=execution,
                        gateway=gateway,
                        config=config,
                        prompt=prompt,
                        settings=settings,
                        evaluation_run_id=evaluation_run_id,
                        inputs=inputs,
                        cell=EvaluationCellInput(
                            key,
                            answer.method,
                            answer.answer_variant,
                            answer.answer_variant,
                            answer.terminal_state,
                            answer.run_id,
                            key,
                            canonical_content_hash(runtime_fact_payload(answer)),
                            answer,
                        ),
                        reference=reference,
                        judge_identity=judge_identity,
                    )
                except Exception:
                    # Missing calls stay engineering failures, never synthetic scores.
                    judged[key] = None
            return judged[key]

        async def judge_grid_answer(answer: TypedAnswerResult) -> dict[str, Any]:
            result = await judge_answer(answer)
            if result is None:
                raise ValueError("relevance_grid_judge_missing")
            return {"value": result.llm_judge_score, "model_call_id": result.model_call_id}

        for cell in inputs.cells:
            observation = _score_resolved_cell(query_id, cell, resolution)
            judge = None
            error = cell.error_code
            candidate = typed_candidate(cell.answer) if cell.answer else None
            judge_reference = (
                text_judge_reference(candidate, resolution)
                if candidate and judge_applicable
                else None
            )
            applicable = observation.score.applicable if observation.score else selected.applicable
            unavailable = (
                observation.score.unavailable_reason
                if observation.score
                else resolution.unavailable_reason
            )
            if observation.status == TerminalStatus.ANSWERED and judge_reference is not None:
                assert cell.answer is not None
                judge = await judge_answer(cell.answer)
                if judge is None:
                    error = "judge_engineering_missing"
            # Mixed text/typed cells keep their real applicability. Any unresolved
            # cell makes the whole report incomplete, so no paired denominator shrinks.
            summary = summarize_method(
                (replace(selected, applicable=applicable),), (observation,), observation.method
            )
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "method": cell.method,
                    "variant": cell.variant,
                    "source_variant": cell.source_variant,
                    "answer_source_variant": cell.answer.answer_variant if cell.answer else None,
                    "answer_source_method": cell.answer.method if cell.answer else None,
                    "source_fact_id": cell.source_fact_id,
                    "source_fact_hash": cell.source_fact_hash,
                    "typed_answer_id": cell.answer.typed_answer_id if cell.answer else None,
                    "terminal_state": cell.terminal_state,
                    "status": observation.status.value,
                    "applicable": applicable,
                    "metric": "typed_exact_match",
                    "value": summary.micro,
                    "judge_value": judge.llm_judge_score if judge else None,
                    "judge_applicable": judge_applicable,
                    "judge_unavailable_reason": judge_unavailable_reason(
                        applicable=judge_applicable,
                        gold_reason=judge_gold_reason,
                        terminal_state=cell.terminal_state,
                        candidate_kind=candidate.kind if candidate else None,
                        judge_present=judge is not None,
                    ),
                    "judge_model_call_id": judge.model_call_id if judge else None,
                    "judge_method_blind": True if judge else None,
                    "typed_judge_disagreement": summary.micro != judge.llm_judge_score
                    if judge and summary.micro is not None
                    else None,
                    "unavailable_reason": unavailable or error,
                    "engineering_error": error,
                }
            )
        relevance_grid = (
            await step8_relevance_grid.evaluate_relevance_grid_answers(
                scope=DataScope.EVALUATION,
                query=inputs.query,
                snapshot_id=inputs.snapshot_id,
                query_run_id=query_executor_run_id,
                cells=inputs.runtime_cells,
                answers={answer.typed_answer_id: answer for answer in inputs.runtime_answers},
                resolution=resolution,
                thresholds=list(RELEVANCE_THRESHOLDS),
                judge=judge_grid_answer,
                **({"primary_thresholds": inputs.primary_thresholds} if primary else {}),
            )
            if inputs.runtime_cells and method_scope != "paired_main"
            else {
                "status": "not_evaluated",
                "reason": "analysis_scope_only" if method_scope == "paired_main" else "no_runtime_cell_facts",
            }
        )
        if oracle_hook is not None and oracle_hook.selected(query_executor_run_id, query_id):
            try:
                async with sessions() as session:
                    await step8_semantic_workflow._require_evaluator(session)
                if oracle_hook.pool.query != inputs.query:
                    raise ValueError("oracle_hook_query_scope_mismatch")
                oracle_execution = create_model_execution_service(
                    settings.model_copy(
                        update={
                            "model_dev_max_attempts": 1,
                            "model_paper_max_attempts": 1,
                        }
                    ),
                    registry,
                    artifact_service=artifacts,
                    model_calls=_ModelCallWriter(sessions),
                    data_scope=DataScope.EVALUATION,
                )
                oracle_gateway = create_generation_gateway(
                    settings,
                    registry,
                    config.judge_model_alias,
                    image_url_resolver=create_image_url_resolver(settings, runtime_store),
                )
                oracle_report = await oracle_hook.run(
                    evaluation_run_id=evaluation_run_id,
                    execution=oracle_execution,
                    judge_gateway=oracle_gateway,
                    judge_config=config,
                    judge_prompt=prompt,
                    resolution=resolution,
                    artifact_service=artifacts,
                )
            except Exception as exc:
                oracle_report = {
                    **oracle_not_evaluated("oracle_execution_incomplete"),
                    "status": "incomplete",
                    "error_type": type(exc).__name__,
                }
        incomplete = (
            process_metrics["status"] == "incomplete"
            or semantic_report["status"] == "incomplete"
            or relevance_grid["status"] == "incomplete"
            or oracle_report["status"] == "incomplete"
            or any(
                r["status"] == "engineering_missing"
                or r["engineering_error"]
                or r["unavailable_reason"] == UNRESOLVED
                for r in rows
            )
        )
        report = {
            "schema_version": _VERSION,
            "evaluation_run_id": evaluation_run_id,
            "campaign_id": campaign_id,
            "suite_id": suite_id,
            "query_executor_run_id": None if method_scope == "baseline" else query_executor_run_id,
            **({"baseline_run_id": query_executor_run_id} if method_scope == "baseline" else {}),
            "query_id": query_id,
            "status": "incomplete" if incomplete else "complete",
            "method_scope": method_scope,
            "all_methods_evaluated": not incomplete and method_scope == "all"
            and {"pmmc", *BASELINE_METHODS}.issubset({cell.method for cell in inputs.cells}),
            **({"all_primary_methods_evaluated": not incomplete and
                {cell.method for cell in inputs.cells} == {"pmmc", *BASELINE_METHODS}}
               if method_scope == "paired_main" else {}),
            "scientific_approval": False,
            "reference_annotation": annotation,
            "selected_query": asdict(selected),
            "selected_cell_count": len(rows),
            "applicable_cell_count": sum(bool(row["applicable"]) for row in rows),
            "scores": rows,
            "process_metrics": process_metrics,
            "qbank_semantic": semantic_report,
            "relevance_grid": relevance_grid,
            "oracle_at_50": oracle_report,
            "identity_hash": config_hash,
            "warnings": [
                "not_scientific_threshold_approval",
                "human_semantic_validation_required",
                "image_gold_requires_lossless_source_or_explicit_annotation",
            ],
        }
        payload = (
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        report = json.loads(payload)
        artifact = await artifacts.put_bytes(
            # Failed attempts are immutable diagnostics, not a cached success that
            # would prevent recovery when an audited Judge call can be retried.
            artifact_id=(
                "step8-incomplete-" + hashlib.sha256(payload).hexdigest()[:32]
                if incomplete
                else "step8-scores-" + config_hash[:32]
            ),
            run_id=query_executor_run_id,
            evaluation_run_id=evaluation_run_id,
            artifact_type=AssetType.TRACE_EXPORT,
            data=payload,
            mime_type="application/json",
            extension="json",
            snapshot_id=inputs.snapshot_id,
            query_id=query_id,
        )
        result = {
            **report,
            "artifact": {
                key: getattr(artifact, key)
                for key in ("artifact_id", "sha256", "object_key", "storage_uri")
            },
        }
        if incomplete:
            raise Step8EvaluationIncomplete(result)
        await _finish(sessions, evaluation_run_id, len(rows), report, result["artifact"])
        return result
    finally:
        await engine.dispose()


async def verify_selected_score_recovery(
    settings: Settings, output: dict, *, campaign_id: str, suite_id: str,
    source_run_id: str, query_id: str, snapshot_id: str, method_scope: MethodScope,
) -> None:
    """A saved worker receipt cannot replace the SQL-triggered completion attestation."""
    report = {k: v for k, v in output.items() if k != "artifact"}
    artifact = output.get("artifact") or {}
    digest = hashlib.sha256((json.dumps(report, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False) + "\n").encode()).hexdigest()
    expected = {"campaign_id": campaign_id, "suite_id": suite_id, "query_id": query_id,
                "method_scope": method_scope, "status": "complete", "schema_version": _VERSION,
                "scientific_approval": False,
                "query_executor_run_id": None if method_scope == "baseline" else source_run_id}
    if method_scope == "baseline":
        expected["baseline_run_id"] = source_run_id
    if any(report.get(k) != v for k, v in expected.items()) or digest != artifact.get("sha256"):
        raise ValueError("release_score_recovery_identity_changed")
    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with sessions() as session:
            run = report["evaluation_run_id"]
            state = await session.get(EvaluationRunStateModel, run)
            row = await session.get(EvaluationArtifactModel, artifact["artifact_id"])
            if (state is None or state.status != "completed"
                    or state.metadata_json.get("completion_report") != report
                    or state.metadata_json.get("completion_artifact") != {
                        "artifact_id": artifact["artifact_id"], "sha256": digest}
                    or row is None or row.state != "ready" or row.evaluation_run_id != run
                    or row.run_id != source_run_id or row.query_id != query_id or row.snapshot_id != snapshot_id
                    or any(getattr(row, k) != artifact.get(k) for k in ("sha256", "object_key", "storage_uri"))):
                raise ValueError("release_score_recovery_requires_committed_evaluation")
    finally:
        await engine.dispose()


async def _finish(
    sessions, evaluation_run_id: str, count: int, report: dict, artifact: dict
) -> None:
    async with sessions.begin() as session:
        state = await session.get(EvaluationRunStateModel, evaluation_run_id, with_for_update=True)
        if state is None:
            raise ValueError("step8_evaluation_state_missing")
        if state.status == "completed":
            return
        await session.execute(
            update(EvaluationRunStateModel)
            .where(
                EvaluationRunStateModel.evaluation_run_id == evaluation_run_id,
            )
            .values(
                status="completed",
                succeeded_count=count,
                failed_count=0,
                completed_at=utc_now(),
                metadata_json={
                    **state.metadata_json,
                    "completion_report": report,
                    "completion_artifact": {
                        key: artifact[key] for key in ("artifact_id", "sha256")
                    },
                },
            )
        )
