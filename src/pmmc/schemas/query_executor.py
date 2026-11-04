"""Runtime-safe contracts for the Step 7 online PMMC query executor."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    ANSWER_OUTPUT_SCHEMA_VERSION,
    QUERY_EXECUTOR_SCHEMA_VERSION,
    STEP7_PILOT_SCHEMA_VERSION,
    STEP7B_PILOT_SCHEMA_VERSION,
    BaseSchema,
    DatasetName,
    JsonDict,
    Modality,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.doubter import QBankRouteClass
from pmmc.schemas.model import GenerationRequest
from pmmc.schemas.persistence import JobStatus
from pmmc.schemas.pilot import QuestionerPilotSample
from pmmc.schemas.query import QueryInput


class QueryExecutorRunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueryExecutorStage(StrEnum):
    EMBED = "embed"
    MATCH = "match"
    EXECUTE = "execute"
    ANSWER = "answer"
    FALLBACK = "fallback"
    SELECT = "select"


class QBankRuntimeChannel(StrEnum):
    LEXICAL = "lexical"
    TEXT = "text"
    MULTIMODAL = "multimodal"


class QueryRouteOutcome(StrEnum):
    PROGRAM = "program"
    FALLBACK = "fallback"
    CONTROL = "control"
    TERMINAL_FAILURE = "terminal_failure"


class QueryFallbackReason(StrEnum):
    NO_QBANK_CANDIDATE = "no_qbank_candidate"
    LOW_MATCH_CONFIDENCE = "low_match_confidence"
    AMBIGUOUS_MATCH = "ambiguous_match"
    CONTROL_ROUTE = "control_route"
    PROGRAM_VALIDATION_FAILED = "program_validation_failed"
    PROGRAM_EXECUTION_FAILED = "program_execution_failed"
    EMPTY_EVIDENCE = "empty_evidence"
    REQUIREMENT_INSUFFICIENT = "requirement_insufficient"
    ANSWER_GENERATION_FAILED = "answer_generation_failed"
    RERANKER_REJECTED = "reranker_rejected"
    RUNTIME_RELEVANCE_INSUFFICIENT = "runtime_relevance_insufficient"
    ANSWER_CONTRACT_FAILED = "answer_contract_failed"


class RerankerDimensionVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class RerankerMatchLevel(StrEnum):
    EXACT = "exact"
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"

    @property
    def score(self) -> float:
        return {
            RerankerMatchLevel.EXACT: 1.0,
            RerankerMatchLevel.STRONG: 0.75,
            RerankerMatchLevel.WEAK: 0.35,
            RerankerMatchLevel.NONE: 0.0,
        }[self]


RequiredRerankerVerdict = Literal[RerankerDimensionVerdict.PASS, RerankerDimensionVerdict.FAIL]


class QBankRerankerDimensions(BaseSchema):
    same_information_need: RequiredRerankerVerdict
    candidate_can_answer_query: RequiredRerankerVerdict
    answer_type_compatible: RequiredRerankerVerdict
    modality_compatible: RerankerDimensionVerdict


class QBankRerankerOutput(BaseSchema):
    same_information_need: RequiredRerankerVerdict
    candidate_can_answer_query: RequiredRerankerVerdict
    answer_type_compatible: RequiredRerankerVerdict
    modality_compatible: RerankerDimensionVerdict
    match_level: RerankerMatchLevel


class QBankRerankerAssessment(BaseSchema):
    reranker_assessment_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    qbank_id: NonEmptyStr
    model_call_id: NonEmptyStr
    prompt_version: NonEmptyStr = "pmmc-qbank-reranker-v1"
    dimensions: QBankRerankerDimensions
    provider_match_level: RerankerMatchLevel
    effective_match_level: RerankerMatchLevel
    reranker_score: Probability
    provider_schema_conformant: bool
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_server_score(self) -> QBankRerankerAssessment:
        required = (
            self.dimensions.same_information_need,
            self.dimensions.candidate_can_answer_query,
            self.dimensions.answer_type_compatible,
        )
        modality_failed = self.dimensions.modality_compatible is RerankerDimensionVerdict.FAIL
        forced_none = (
            any(item is not RerankerDimensionVerdict.PASS for item in required) or modality_failed
        )
        expected_level = RerankerMatchLevel.NONE if forced_none else self.provider_match_level
        if self.effective_match_level is not expected_level:
            raise ValueError("reranker effective level must be server-derived")
        if self.reranker_score != expected_level.score:
            raise ValueError("reranker score must be server-derived")
        return self


class QueryAnswerVariant(StrEnum):
    PMMC_R0 = "pmmc_r0"
    PMMC_R1 = "pmmc_r1"
    PMMC_R2 = "pmmc_r2"
    PMMC_R3 = "pmmc_r3"
    NATIVE_MM_COUNTERFACTUAL = "native_mm_counterfactual"
    ACTIVE_FINAL = "active_final"
    ACTIVE_FINAL_PROVISIONAL = "active_final_provisional"
    NO_FALLBACK_COUNTERFACTUAL = "no_fallback_counterfactual"
    FORCED_TOP1_COUNTERFACTUAL = "forced_top1_counterfactual"
    PMMC_B1_R0 = "pmmc_b1_r0"
    PMMC_B1_R1 = "pmmc_b1_r1"
    PMMC_B1_R2 = "pmmc_b1_r2"
    PMMC_B1_R3 = "pmmc_b1_r3"
    PMMC_B2_R0 = "pmmc_b2_r0"
    PMMC_B2_R1 = "pmmc_b2_r1"
    PMMC_B2_R2 = "pmmc_b2_r2"
    PMMC_B2_R3 = "pmmc_b2_r3"
    PMMC_B4_R0 = "pmmc_b4_r0"
    PMMC_B4_R1 = "pmmc_b4_r1"
    PMMC_B4_R2 = "pmmc_b4_r2"
    PMMC_B4_R3 = "pmmc_b4_r3"
    PMMC_B8_R0 = "pmmc_b8_r0"
    PMMC_B8_R1 = "pmmc_b8_r1"
    PMMC_B8_R2 = "pmmc_b8_r2"
    PMMC_B8_R3 = "pmmc_b8_r3"


class QueryExecutorRunManifest(BaseSchema):
    query_executor_run_id: NonEmptyStr
    source_compiler_run_id: NonEmptyStr
    routing_projection_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: DatasetName
    schema_version: NonEmptyStr = QUERY_EXECUTOR_SCHEMA_VERSION
    config_hash: NonEmptyStr
    active_round: NonNegativeInt = 2
    max_round: NonNegativeInt = 3
    question_budget: PositiveInt = 8
    selected_query_count: PositiveInt
    pilot_projection: bool = False
    paper_result: bool = False
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_protocol(self) -> QueryExecutorRunManifest:
        if self.active_round != 2 or self.max_round != 3:
            raise ValueError("Step 7 freezes active r2 and maximum r3")
        if self.question_budget != 8:
            raise ValueError("Step 7 freezes the active question budget at eight")
        if self.paper_result and self.pilot_projection:
            raise ValueError("pilot projections cannot produce paper results")
        prompt_version = self.metadata.get("answer_prompt_version")
        prompt_hash = self.metadata.get("answer_prompt_hash")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("query executor run requires answer_prompt_version")
        if (
            not isinstance(prompt_hash, str)
            or len(prompt_hash) != 64
            or any(character not in "0123456789abcdef" for character in prompt_hash)
        ):
            raise ValueError("query executor run requires a lowercase SHA256 prompt hash")
        if self.schema_version == "0.2.0":
            expected_metadata = {
                "matcher_version": "pmmc-qbank-matcher-v2",
                "graph_version": "pmmc-online-query-graph-v2",
                "sufficiency_policy_version": "query-runtime-evidence-v2",
                "reranker_version": "pmmc-bounded-reranker-v1",
                "reranker_prompt_version": "pmmc-qbank-reranker-v1",
                "reranker_model_alias": "reranker-qwen3.5-9b",
                "answer_contract_version": "strict-answer-contract-v2",
                "answer_output_schema_version": ANSWER_OUTPUT_SCHEMA_VERSION,
            }
            for name, expected in expected_metadata.items():
                if self.metadata.get(name) != expected:
                    raise ValueError(f"Step 7B query run has invalid metadata identity: {name}")
            for name in ("reranker_schema_hash", "answer_output_schema_hash"):
                value = self.metadata.get(name)
                if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    raise ValueError(f"Step 7B query run has invalid {name}")
        if self.schema_version == "0.3.0":
            expected_metadata = {
                "matcher_version": "pmmc-qbank-matcher-v3",
                "graph_version": "pmmc-online-query-graph-v3",
                "sufficiency_policy_version": "query-runtime-evidence-v3",
                "answer_prompt_version": "pmmc-answer-contract-v3",
                "answer_contract_version": "typed-answer-contract-v3",
                "answer_output_schema_version": "0.2.0",
                "embedding_ownership": "runtime",
                "source_compiler_schema_version": "0.9.0",
                "source_operator_set_version": "pmmc-operators-v7",
                "routing_projection_schema_version": "0.4.0",
            }
            for name, expected in expected_metadata.items():
                if self.metadata.get(name) != expected:
                    raise ValueError(f"Step 8 query run has invalid metadata identity: {name}")
            full_grid = self.metadata.get("execute_full_grid") is True
            cells = (
                [[budget, round_id] for budget in [1, 2, 4, 8] for round_id in range(4)]
                if full_grid
                else [[8, 2]]
            )
            if (
                self.metadata.get("declared_cells") != cells
                or self.metadata.get("expected_cell_count") != len(cells) * 3
            ):
                raise ValueError("Step 8 requires three projections for every declared cell")
            if full_grid and (
                self.metadata.get("treatment") != "main"
                or self.metadata.get("answer_model_alias") != "answer-qwen3.5-9b"
            ):
                raise ValueError("Step 8 full grid is reserved for the main 9B method")
        return self


class QueryExecutorRunState(BaseSchema):
    query_executor_run_id: NonEmptyStr
    status: QueryExecutorRunStatus = QueryExecutorRunStatus.CREATED
    selected_count: PositiveInt
    matched_count: NonNegativeInt = 0
    executed_count: NonNegativeInt = 0
    pmmc_answer_count: NonNegativeInt = 0
    fallback_answer_count: NonNegativeInt = 0
    selected_answer_count: NonNegativeInt = 0
    failed_count: NonNegativeInt = 0
    started_at: UtcDateTime | None = None
    heartbeat_at: UtcDateTime | None = None
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> QueryExecutorRunState:
        for value in (
            self.matched_count,
            self.executed_count,
            self.pmmc_answer_count,
            self.fallback_answer_count,
            self.selected_answer_count,
            self.failed_count,
        ):
            if value > self.selected_count:
                raise ValueError("query executor counts cannot exceed selected_count")
        if self.status is QueryExecutorRunStatus.COMPLETED and (
            self.selected_answer_count + self.failed_count != self.selected_count
            or self.fallback_answer_count != self.selected_count
            or self.completed_at is None
        ):
            raise ValueError(
                "completed query runs require one fallback and one terminal outcome per query"
            )
        if self.status is QueryExecutorRunStatus.FAILED and self.error_type is None:
            raise ValueError("failed query runs require error_type")
        return self


class QueryExecutorStageJob(BaseSchema):
    query_stage_job_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    stage: QueryExecutorStage
    round_id: NonNegativeInt | None = None
    status: JobStatus = JobStatus.PENDING
    attempt_count: NonNegativeInt = 0
    max_attempts: PositiveInt = 2
    claimed_by: NonEmptyStr | None = None
    lease_expires_at: UtcDateTime | None = None
    available_at: UtcDateTime = Field(default_factory=utc_now)
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_job(self) -> QueryExecutorStageJob:
        running = self.status is JobStatus.RUNNING
        if running != (self.claimed_by is not None and self.lease_expires_at is not None):
            raise ValueError("only running query jobs may hold a complete lease")
        terminal = self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
        if terminal != (self.finished_at is not None):
            raise ValueError("terminal query jobs require finished_at")
        if self.status is JobStatus.FAILED and self.error_type is None:
            raise ValueError("failed query jobs require error_type")
        if self.round_id is not None and self.round_id > 3:
            raise ValueError("query execution round cannot exceed three")
        return self


class QueryEmbeddingBatch(BaseSchema):
    query_embedding_batch_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    model_call_id: NonEmptyStr | None = None
    channel: QBankRuntimeChannel
    batch_ordinal: NonNegativeInt
    query_ids: list[NonEmptyStr] = Field(min_length=1)
    expected_count: PositiveInt
    committed_count: NonNegativeInt
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_batch(self) -> QueryEmbeddingBatch:
        ensure_unique_strings(self.query_ids, "query_ids")
        if self.expected_count != len(self.query_ids):
            raise ValueError("embedding expected_count must equal query_ids")
        if self.committed_count not in {0, self.expected_count}:
            raise ValueError("query embedding batches commit atomically")
        if self.channel is QBankRuntimeChannel.LEXICAL:
            raise ValueError("lexical matching does not create query embeddings")
        projected = self.metadata.get("coverage_projection") is True
        source_call = self.metadata.get("source_evaluation_model_call_id")
        if projected:
            if (
                self.model_call_id is not None
                or not isinstance(source_call, str)
                or not source_call
            ):
                raise ValueError(
                    "coverage-projected embeddings require only evaluator call lineage"
                )
        elif self.model_call_id is None:
            raise ValueError("native runtime embedding batches require model_call_id")
        return self


class QueryEmbeddingFact(BaseSchema):
    query_embedding_id: NonEmptyStr
    query_embedding_batch_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    channel: QBankRuntimeChannel
    embedding_id: NonEmptyStr
    collection_id: NonEmptyStr
    vector_dimension: PositiveInt = 4096
    input_hash: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_embedding(self) -> QueryEmbeddingFact:
        if self.channel is QBankRuntimeChannel.LEXICAL:
            raise ValueError("lexical matching does not use an embedding fact")
        if self.vector_dimension != 4096:
            raise ValueError("Step 7 query embeddings are frozen at 4096 dimensions")
        return self


class QBankMatchBatch(BaseSchema):
    match_batch_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    source_compiler_run_id: NonEmptyStr
    routing_projection_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    conversation_id: NonEmptyStr
    cutoff_session_id: NonEmptyStr
    active_round: NonNegativeInt
    question_budget: PositiveInt
    requested_top_k: PositiveInt = 50
    channel_count: PositiveInt
    fused_count: NonNegativeInt
    config_hash: NonEmptyStr
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_match_batch(self) -> QBankMatchBatch:
        if self.active_round > 3 or self.question_budget not in {1, 2, 4, 8}:
            raise ValueError("match batch round or budget is outside the frozen grid")
        if self.requested_top_k != 50:
            raise ValueError("Step 7 matcher persists top-50")
        if self.channel_count not in {2, 3} or self.fused_count > 100:
            raise ValueError("match batch channel or route-class top-50 count is invalid")
        return self


class QBankMatchChannelCandidate(BaseSchema):
    channel_candidate_id: NonEmptyStr
    match_batch_id: NonEmptyStr
    query_id: NonEmptyStr
    qbank_id: NonEmptyStr
    route_class: QBankRouteClass
    channel: QBankRuntimeChannel
    rank: PositiveInt
    raw_score: float
    generation_rank: PositiveInt
    accepted_round: NonNegativeInt
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ranks(self) -> QBankMatchChannelCandidate:
        if self.rank > 50 or self.generation_rank > 8 or self.accepted_round > 3:
            raise ValueError("channel candidate rank exceeds the frozen Step 7 grid")
        return self


class QBankLexicalTermFact(BaseSchema):
    lexical_term_fact_id: NonEmptyStr
    match_batch_id: NonEmptyStr
    query_id: NonEmptyStr
    ordinal: NonNegativeInt
    term_hash: NonEmptyStr
    term_kind: NonEmptyStr
    exact_boost: bool
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_lexical_term(self) -> QBankLexicalTermFact:
        if self.ordinal > 11:
            raise ValueError("Step 7B lexical extraction is bounded to twelve terms")
        if self.term_kind not in {"lexeme", "entity", "number", "date"}:
            raise ValueError("unsupported lexical term kind")
        return self


class QBankMatchFusedCandidate(BaseSchema):
    fused_candidate_id: NonEmptyStr
    match_batch_id: NonEmptyStr
    query_id: NonEmptyStr
    qbank_id: NonEmptyStr
    plan_id: NonEmptyStr
    route_class: QBankRouteClass
    rank: PositiveInt
    lexical_rank: PositiveInt | None = None
    text_rank: PositiveInt | None = None
    multimodal_rank: PositiveInt | None = None
    raw_rrf: NonNegativeFloat
    confidence: Probability
    modality_compatible: bool
    lexical_normalized_score: Probability | None = None
    text_normalized_score: Probability | None = None
    multimodal_normalized_score: Probability | None = None
    available_weight_sum: Probability | None = None
    base_score: Probability | None = None
    base_reliable_score: Probability | None = None
    reranker_score: Probability | None = None
    reranker_match_level: RerankerMatchLevel | None = None
    route_score: Probability | None = None
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_rank(self) -> QBankMatchFusedCandidate:
        if self.rank > 50:
            raise ValueError("fused candidate rank exceeds the route-class top-50")
        return self


class QueryRouteDecision(BaseSchema):
    route_decision_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    match_batch_id: NonEmptyStr
    query_id: NonEmptyStr
    outcome: QueryRouteOutcome
    selected_qbank_id: NonEmptyStr | None = None
    selected_plan_id: NonEmptyStr | None = None
    confidence: Probability | None = None
    top1_top2_gap: Probability | None = None
    match_threshold: Probability
    gap_threshold: Probability
    fallback_reason: QueryFallbackReason | None = None
    counterfactual_qbank_id: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_decision(self) -> QueryRouteDecision:
        selected = self.outcome is QueryRouteOutcome.PROGRAM
        if selected != (self.selected_qbank_id is not None):
            raise ValueError("program routes require exactly one selected qbank item")
        if selected != (self.selected_plan_id is not None):
            raise ValueError("program routes require exactly one selected program")
        fallback = self.outcome in {QueryRouteOutcome.FALLBACK, QueryRouteOutcome.CONTROL}
        if fallback != (self.fallback_reason is not None):
            raise ValueError("fallback/control routes require exactly one fallback reason")
        if selected and self.confidence is None:
            raise ValueError("selected program routes require confidence")
        return self


class QueryProgramBinding(BaseSchema):
    query_program_binding_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    route_decision_id: NonEmptyStr
    routing_projection_id: NonEmptyStr
    qbank_id: NonEmptyStr
    plan_id: NonEmptyStr
    round_id: NonNegativeInt
    program_hash: NonEmptyStr
    operator_set_version: NonEmptyStr
    requirement_contract_hash: NonEmptyStr
    counterfactual: bool = False
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_round(self) -> QueryProgramBinding:
        if self.round_id > 3:
            raise ValueError("bound program round cannot exceed three")
        return self


class QueryEvidenceSufficiencyAssessment(BaseSchema):
    sufficiency_assessment_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    query_program_binding_id: NonEmptyStr
    execution_id: NonEmptyStr
    technical_success: bool
    nonempty_evidence: bool
    actual_capabilities: list[NonEmptyStr]
    required_capabilities: list[NonEmptyStr]
    actual_modalities: list[Modality]
    required_modalities: list[Modality]
    independent_group_count: NonNegativeInt
    minimum_evidence_groups: PositiveInt
    raw_image_reached: bool
    raw_image_required: bool
    structural_requirement_success: bool | None = None
    query_evidence_relevance: bool | None = None
    answer_support: bool | None = None
    relevant_evidence_group_count: NonNegativeInt | None = None
    relevance_threshold: Probability | None = None
    sufficient: bool
    failure_reason: QueryFallbackReason | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sufficiency(self) -> QueryEvidenceSufficiencyAssessment:
        ensure_unique_strings(self.actual_capabilities, "actual_capabilities")
        ensure_unique_strings(self.required_capabilities, "required_capabilities")
        if len(self.actual_modalities) != len(set(self.actual_modalities)):
            raise ValueError("actual_modalities cannot contain duplicates")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities cannot contain duplicates")
        structural = (
            self.technical_success
            and self.nonempty_evidence
            and set(self.required_capabilities) <= set(self.actual_capabilities)
            and set(self.required_modalities) <= set(self.actual_modalities)
            and self.independent_group_count >= self.minimum_evidence_groups
            and (not self.raw_image_required or self.raw_image_reached)
        )
        if (
            self.structural_requirement_success is not None
            and self.structural_requirement_success != structural
        ):
            raise ValueError("structural requirement success must be server-derived")
        v2 = self.metadata.get("sufficiency_policy_version") == "query-runtime-evidence-v2"
        expected = structural
        if v2:
            if (
                self.structural_requirement_success is None
                or self.query_evidence_relevance is None
                or self.answer_support is None
                or self.relevance_threshold is None
            ):
                raise ValueError("runtime evidence v2 requires relevance fields")
            expected = structural and self.query_evidence_relevance and self.answer_support
        if self.metadata.get("sufficiency_policy_version") == "query-runtime-evidence-v3":
            if self.answer_support is not None:
                raise ValueError("pre-answer sufficiency cannot assert answer support")
            if self.structural_requirement_success is None or self.query_evidence_relevance is None:
                raise ValueError("v3 requires separate structural and relevance decisions")
            expected = structural and self.query_evidence_relevance
        if self.sufficient != expected:
            raise ValueError("query evidence sufficiency must be server-derived")
        if self.sufficient == (self.failure_reason is not None):
            raise ValueError("only insufficient assessments require a failure reason")
        return self


class QueryEvidenceRelevanceAssessment(BaseSchema):
    relevance_assessment_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    query_program_binding_id: NonEmptyStr
    relevance_threshold: Probability
    required_group_count: PositiveInt
    relevant_group_count: NonNegativeInt
    passed: bool
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_group_decision(self) -> QueryEvidenceRelevanceAssessment:
        expected = self.relevant_group_count >= self.required_group_count
        if self.metadata.get("policy_version") == "query-runtime-evidence-v3":
            expected = expected and self.metadata.get("required_roles_satisfied") is True
        if self.passed != expected:
            raise ValueError("relevance group decision must be server-derived")
        return self


class QueryEvidenceRelevanceItem(BaseSchema):
    relevance_item_id: NonEmptyStr
    relevance_assessment_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    execution_id: NonEmptyStr
    ordinal: NonNegativeInt
    evidence_id_hash: NonEmptyStr
    evidence_group_hash: NonEmptyStr
    modality: Modality
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    normalized_score: Probability
    relevant: bool
    threshold: Probability
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_relevance(self) -> QueryEvidenceRelevanceItem:
        expected = min(1.0, max(0.0, (self.cosine_similarity + 1.0) / 2.0))
        if abs(self.normalized_score - expected) > 1e-9:
            raise ValueError("normalized relevance score must be server-derived")
        score = (
            self.cosine_similarity
            if self.metadata.get("score_transform") == "raw-cosine-v1"
            else self.normalized_score
        )
        if self.relevant != (score >= self.threshold):
            raise ValueError("relevance decision must be server-derived")
        return self


class QueryExecutionReuseLink(BaseSchema):
    execution_reuse_link_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    source_execution_id: NonEmptyStr
    target_round_id: NonNegativeInt
    execution_identity_hash: NonEmptyStr
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class QueryFallbackDecision(BaseSchema):
    fallback_decision_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    fallback_run_id: NonEmptyStr
    fallback_query_id: NonEmptyStr
    counterfactual_completed: bool
    active_fallback: bool
    reason: QueryFallbackReason | None = None
    fallback_answer_id: NonEmptyStr | None = None
    error_type: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fallback(self) -> QueryFallbackDecision:
        if self.counterfactual_completed != (self.fallback_answer_id is not None):
            raise ValueError("completed fallback counterfactual requires exactly one answer")
        if self.counterfactual_completed == (self.error_type is not None):
            raise ValueError("failed fallback counterfactual requires an error type")
        if self.active_fallback != (self.reason is not None):
            raise ValueError("active fallback requires exactly one reason")
        return self


class QueryAnswerSelection(BaseSchema):
    answer_selection_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    active_variant: QueryAnswerVariant
    selected_answer_id: NonEmptyStr | None = None
    pmmc_answer_id: NonEmptyStr | None = None
    fallback_answer_id: NonEmptyStr | None = None
    terminal_failure: bool = False
    failure_type: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_selection(self) -> QueryAnswerSelection:
        if self.terminal_failure:
            if self.selected_answer_id is not None or self.failure_type is None:
                raise ValueError("terminal answer failures require only a failure type")
        elif self.selected_answer_id is None or self.failure_type is not None:
            raise ValueError("successful answer selection requires selected_answer_id")
        if self.active_variant in {
            QueryAnswerVariant.ACTIVE_FINAL,
            QueryAnswerVariant.ACTIVE_FINAL_PROVISIONAL,
        }:
            raise ValueError("active final is a persisted alias, not a source variant")
        return self


class PmmcFallbackRunLink(BaseSchema):
    query_executor_run_id: NonEmptyStr
    fallback_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    method_id: NonEmptyStr = "native_multimodal_rag"
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class PmmcFallbackQueryLink(BaseSchema):
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    fallback_run_id: NonEmptyStr
    fallback_query_id: NonEmptyStr
    fallback_answer_id: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)


class AnswerGenerationInput(BaseSchema):
    """Method-neutral, gold-free input to the shared answer model."""

    method_id: NonEmptyStr
    answer_variant: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query: QueryInput
    request: GenerationRequest
    evidence_ids_in_prompt: list[NonEmptyStr] = Field(default_factory=list)
    image_ids_in_prompt: list[NonEmptyStr] = Field(default_factory=list)
    source_lineage: JsonDict
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_input(self) -> AnswerGenerationInput:
        ensure_unique_strings(self.evidence_ids_in_prompt, "evidence_ids_in_prompt")
        ensure_unique_strings(self.image_ids_in_prompt, "image_ids_in_prompt")
        request_images = [
            part.image.image_id
            for message in self.request.messages
            for part in message.parts
            if part.image is not None
        ]
        if request_images != self.image_ids_in_prompt:
            raise ValueError("answer input image lineage must preserve request order")
        forbidden = {"gold", "expected_answer", "expected_evidence", "semantic_judge"}
        if forbidden & set(self.source_lineage):
            raise ValueError("runtime answer lineage contains evaluation/compiler-private data")
        return self


class ExperimentCell(BaseSchema):
    """Complete experiment identity, independent of answer display aliases."""

    treatment: Literal[
        "pmmc",
        "fixed_planner",
        "no_raw_image",
        "no_reranker",
        "no_fallback",
        "forced_top1_counterfactual",
    ] = "pmmc"
    question_budget: Literal[1, 2, 4, 8]
    revision_depth: int = Field(ge=0, le=3)
    routing_policy_hash: NonEmptyStr
    answer_contract: Literal["typed-answer-contract-v3"] = "typed-answer-contract-v3"
    seed: int
    source_projection_id: NonEmptyStr
    source_projection_hash: NonEmptyStr

    @property
    def variant(self) -> str:
        return f"pmmc_b{self.question_budget}_r{self.revision_depth}"


def runtime_fact_payload(value: object) -> object:
    """Scientific identity excludes observation timestamps, including nested facts."""
    if isinstance(value, BaseSchema):
        return runtime_fact_payload(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            key: runtime_fact_payload(item) for key, item in value.items() if key != "created_at"
        }
    if isinstance(value, (list, tuple)):
        return [runtime_fact_payload(item) for item in value]
    return value


class QueryRuntimeTypedAnswerFact(BaseSchema):
    typed_answer_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    answer_identity_hash: NonEmptyStr
    answer_model_call_id: NonEmptyStr
    terminal_state: Literal["answered", "abstained", "fallback_required"]
    output: JsonDict
    ungated_output: JsonDict
    image_items: list[JsonDict] = Field(default_factory=list)
    grounding: JsonDict | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_grounding_boundary(self) -> QueryRuntimeTypedAnswerFact:
        if self.terminal_state == "answered" and self.grounding is None:
            raise ValueError("supported typed answers require an independent grounding fact")
        return self


class QueryExperimentCellResult(BaseSchema):
    cell_result_id: NonEmptyStr
    query_executor_run_id: NonEmptyStr
    query_id: NonEmptyStr
    cell_id: NonEmptyStr
    cell: ExperimentCell
    route: QueryRouteDecision
    binding: QueryProgramBinding | None = None
    execution_id: NonEmptyStr | None = None
    execution_source_cell_id: NonEmptyStr | None = None
    sufficiency: QueryEvidenceSufficiencyAssessment | None = None
    relevance: QueryEvidenceRelevanceAssessment | None = None
    relevance_items: list[QueryEvidenceRelevanceItem] = Field(default_factory=list)
    pmmc_answer_id: NonEmptyStr | None = None
    answer_source_cell_id: NonEmptyStr | None = None
    fallback_answer_id: NonEmptyStr | None = None
    selected_answer_id: NonEmptyStr | None = None
    terminal_state: Literal["answered", "abstained", "no_answer", "technical_failure"]
    error_type: NonEmptyStr | None = None
    active_fallback: bool = False
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_cell(self) -> QueryExperimentCellResult:
        if self.cell.treatment == "no_fallback":
            if self.active_fallback or (
                self.selected_answer_id is not None
                and self.route.outcome is not QueryRouteOutcome.PROGRAM
            ):
                raise ValueError("no-fallback must preserve the route gate")
        if self.terminal_state in {"answered", "abstained"} and self.selected_answer_id is None:
            raise ValueError("answer terminal requires a typed answer fact")
        return self


class QueryExecutorFreezeManifest(BaseSchema):
    freeze_manifest_id: NonEmptyStr
    schema_version: NonEmptyStr = QUERY_EXECUTOR_SCHEMA_VERSION
    verified_qbank_freeze_manifest_id: NonEmptyStr
    query_executor_config_hash: NonEmptyStr
    query_executor_protocol_hash: NonEmptyStr | None = None
    approved_routing_policy_hash: NonEmptyStr | None = None
    approved_selection_hash: NonEmptyStr | None = None
    answer_prompt_hash: NonEmptyStr
    reranker_identity_hash: NonEmptyStr | None = None
    runtime_evidence_policy_version: NonEmptyStr | None = None
    answer_contract_version: NonEmptyStr | None = None
    answer_output_schema_version: NonEmptyStr | None = None
    answer_contract_hash: NonEmptyStr | None = None
    fallback_policy_version: NonEmptyStr
    active_round: NonNegativeInt = 2
    question_budget: PositiveInt = 8
    match_threshold: Probability
    gap_threshold: Probability
    threshold_grid: list[Probability]
    gap_grid: list[Probability]
    pilot_content_hash: NonEmptyStr
    approved_by: NonEmptyStr
    approved_at: UtcDateTime
    decision_content_hash: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_freeze(self) -> QueryExecutorFreezeManifest:
        if self.active_round != 2 or self.question_budget != 8:
            raise ValueError("Step 7 freeze requires r2/b8")
        if self.match_threshold not in self.threshold_grid:
            raise ValueError("match threshold must be selected from the frozen grid")
        if self.gap_threshold not in self.gap_grid:
            raise ValueError("gap threshold must be selected from the frozen grid")
        if self.schema_version == "0.2.0":
            if self.runtime_evidence_policy_version != "query-runtime-evidence-v2":
                raise ValueError("Step 7B freeze requires runtime evidence policy v2")
            if self.answer_contract_version != "strict-answer-contract-v2":
                raise ValueError("Step 7B freeze requires strict Answer Contract v2")
            if self.answer_output_schema_version != ANSWER_OUTPUT_SCHEMA_VERSION:
                raise ValueError("Step 7B freeze has the wrong answer output schema")
            for field_name in (
                "query_executor_protocol_hash",
                "approved_routing_policy_hash",
                "approved_selection_hash",
                "reranker_identity_hash",
                "answer_contract_hash",
            ):
                value = getattr(self, field_name)
                if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    raise ValueError(f"Step 7B freeze has invalid {field_name}")
        return self


class Step7BFreezeManifest(BaseSchema):
    """Aggregate, Step 8-facing identity for an approved Step 7B pilot."""

    freeze_manifest_id: NonEmptyStr
    pilot_id: NonEmptyStr
    schema_version: NonEmptyStr = STEP7B_PILOT_SCHEMA_VERSION
    questioner_freeze_hash: NonEmptyStr
    compiler_program_freeze_hash: NonEmptyStr
    verified_qbank_freeze_hash: NonEmptyStr
    query_executor_freeze_hash: NonEmptyStr
    answer_contract_hash: NonEmptyStr
    active_round: NonNegativeInt = 2
    question_budget: PositiveInt = 8
    match_threshold: Probability
    gap_threshold: Probability
    approved_by: NonEmptyStr
    approved_at: UtcDateTime
    decision_content_hash: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_step7b_freeze(self) -> Step7BFreezeManifest:
        if self.schema_version != STEP7B_PILOT_SCHEMA_VERSION:
            raise ValueError("Step 7B freeze has an unsupported schema version")
        if self.active_round != 2 or self.question_budget != 8:
            raise ValueError("Step 7B freeze requires active r2/b8")
        for field_name in (
            "questioner_freeze_hash",
            "compiler_program_freeze_hash",
            "verified_qbank_freeze_hash",
            "query_executor_freeze_hash",
            "answer_contract_hash",
            "decision_content_hash",
            "content_hash",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, field_name)) is None:
                raise ValueError(f"Step 7B freeze has invalid SHA256: {field_name}")
        return self


class Step7PilotStatus(StrEnum):
    BUILDING = "building"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_THRESHOLD_REVIEW = "awaiting_threshold_review"
    NEEDS_REMEDIATION = "needs_remediation"
    FAILED = "failed"
    APPROVED = "approved"
    REJECTED = "rejected"


class Step7PilotDatasetSource(BaseSchema):
    dataset: DatasetName
    compiler_runtime_export_dir: Path
    coverage_export_dir: Path
    query_runtime_export_dir: Path
    evaluation_export_dir: Path
    snapshot_dir: Path

    @model_validator(mode="after")
    def _validate_dataset(self) -> Step7PilotDatasetSource:
        if self.dataset is DatasetName.UNKNOWN:
            raise ValueError("Step 7 pilot source requires a concrete dataset")
        return self


class Step7PilotSourceManifest(BaseSchema):
    protocol_version: NonEmptyStr = "pmmc-step7-executor-pilot-v1"
    launch_id: NonEmptyStr
    partition_id: NonEmptyStr
    samples: list[QuestionerPilotSample]
    datasets: list[Step7PilotDatasetSource]
    benchmark_root: Path
    output_root: Path
    safe_execution_log: Path | None = None

    @model_validator(mode="after")
    def _validate_sources(self) -> Step7PilotSourceManifest:
        sample_datasets = {item.dataset for item in self.samples}
        source_datasets = {item.dataset for item in self.datasets}
        if len(sample_datasets) != len(self.samples):
            raise ValueError("Step 7 pilot samples repeat a dataset")
        if len(source_datasets) != len(self.datasets):
            raise ValueError("Step 7 pilot sources repeat a dataset")
        if sample_datasets != source_datasets:
            raise ValueError("Step 7 pilot samples and export sources differ")
        if sample_datasets != {DatasetName.MEMLENS, DatasetName.MEMGALLERY}:
            raise ValueError("Step 7 pilot requires both benchmark datasets")
        if any(item.partition_id != self.partition_id for item in self.samples):
            raise ValueError("Step 7 pilot samples use another partition")
        expected = (
            16
            if self.protocol_version == "pmmc-step7b-executor-pilot-v1"
            else 12
            if self.protocol_version == "pmmc-step7-executor-pilot-v1"
            else None
        )
        if expected is None:
            raise ValueError("unsupported Step 7 pilot source protocol")
        if any(len(item.query_ids) != expected for item in self.samples):
            raise ValueError("Step 7 pilot query denominator differs from its protocol")
        return self


class Step7PilotArtifactRef(BaseSchema):
    role: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    size_bytes: NonNegativeInt
    private: bool = False

    @model_validator(mode="after")
    def _validate_artifact(self) -> Step7PilotArtifactRef:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Step 7 pilot artifact must stay below the bundle root")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("Step 7 pilot artifact requires a SHA256 digest")
        return self


class Step7PilotManifest(BaseSchema):
    pilot_id: NonEmptyStr
    schema_version: NonEmptyStr = STEP7_PILOT_SCHEMA_VERSION
    protocol_version: NonEmptyStr = "pmmc-step7-executor-pilot-v1"
    launch_id: NonEmptyStr
    partition_id: NonEmptyStr
    samples: list[QuestionerPilotSample]
    status: Step7PilotStatus
    query_executor_run_ids: list[NonEmptyStr]
    compiler_run_ids: list[NonEmptyStr]
    source_compiler_export_ids: dict[NonEmptyStr, NonEmptyStr]
    source_coverage_export_ids: dict[NonEmptyStr, NonEmptyStr]
    source_query_export_ids: dict[NonEmptyStr, NonEmptyStr]
    source_evaluation_export_ids: dict[NonEmptyStr, NonEmptyStr]
    query_count: PositiveInt
    prefix_query_count: PositiveInt
    automatic_integrity_passed: bool
    integrity_issue_types: list[NonEmptyStr] = Field(default_factory=list)
    artifacts: list[Step7PilotArtifactRef] = Field(default_factory=list)
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_pilot(self) -> Step7PilotManifest:
        protocol = (self.schema_version, self.protocol_version)
        if protocol not in {
            (STEP7_PILOT_SCHEMA_VERSION, "pmmc-step7-executor-pilot-v1"),
            (STEP7B_PILOT_SCHEMA_VERSION, "pmmc-step7b-executor-pilot-v1"),
        }:
            raise ValueError("unsupported Step 7 pilot schema/protocol pair")
        step7b = self.protocol_version == "pmmc-step7b-executor-pilot-v1"
        ensure_unique_strings(self.query_executor_run_ids, "query_executor_run_ids")
        ensure_unique_strings(self.compiler_run_ids, "compiler_run_ids")
        ensure_unique_strings(self.integrity_issue_types, "integrity_issue_types")
        ensure_unique_strings(
            [item.relative_path for item in self.artifacts], "Step 7 artifact paths"
        )
        if len(self.samples) != 2 or {item.dataset for item in self.samples} != {
            DatasetName.MEMLENS,
            DatasetName.MEMGALLERY,
        }:
            raise ValueError("Step 7 pilot requires one sample for each benchmark")
        if any(item.partition_id != self.partition_id for item in self.samples):
            raise ValueError("Step 7 pilot samples use another partition")
        datasets = {item.dataset.value for item in self.samples}
        for values in (
            self.source_compiler_export_ids,
            self.source_coverage_export_ids,
            self.source_query_export_ids,
            self.source_evaluation_export_ids,
        ):
            if set(values) != datasets:
                raise ValueError("Step 7 source export IDs do not cover both datasets")
        if len(self.query_executor_run_ids) != 2 or len(self.compiler_run_ids) != 2:
            raise ValueError("Step 7 pilot requires two compiler and query runs")
        if self.query_count != sum(len(item.query_ids) for item in self.samples):
            raise ValueError("Step 7 pilot query denominator differs from its samples")
        if self.query_count != (32 if step7b else 24) or self.prefix_query_count != 4:
            raise ValueError("Step 7 pilot query and diagnostic-prefix denominators are frozen")
        if self.automatic_integrity_passed == bool(self.integrity_issue_types):
            raise ValueError("Step 7 pilot integrity status and issues disagree")
        if self.status is Step7PilotStatus.AWAITING_APPROVAL and not (
            self.automatic_integrity_passed
        ):
            raise ValueError("only an integrity-passing pilot may await approval")
        if step7b:
            if self.status is Step7PilotStatus.AWAITING_APPROVAL:
                raise ValueError("Step 7B must await an explicit threshold review")
            if (
                self.status is Step7PilotStatus.AWAITING_THRESHOLD_REVIEW
                and not self.automatic_integrity_passed
            ):
                raise ValueError("only an integrity-passing Step 7B pilot may await review")
            if (
                self.status is Step7PilotStatus.NEEDS_REMEDIATION
                and self.automatic_integrity_passed
            ):
                raise ValueError("Step 7B remediation status requires integrity issues")
        return self


class Step7PilotDecision(BaseSchema):
    pilot_id: NonEmptyStr
    decision: NonEmptyStr
    reviewer_id: NonEmptyStr
    rationale: NonEmptyStr
    pilot_content_hash: NonEmptyStr
    match_threshold: Probability
    gap_threshold: Probability
    decided_at: UtcDateTime = Field(default_factory=utc_now)
    content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_decision(self) -> Step7PilotDecision:
        if self.decision not in {"approve", "reject"}:
            raise ValueError("Step 7 decision must be approve or reject")
        for field_name in ("pilot_content_hash", "content_hash"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, field_name)) is None:
                raise ValueError(f"Step 7 decision has invalid SHA256: {field_name}")
        return self


__all__ = [
    "AnswerGenerationInput",
    "PmmcFallbackQueryLink",
    "PmmcFallbackRunLink",
    "QBankMatchBatch",
    "QBankMatchChannelCandidate",
    "QBankMatchFusedCandidate",
    "QBankLexicalTermFact",
    "QBankRuntimeChannel",
    "QueryAnswerSelection",
    "QueryAnswerVariant",
    "QueryEmbeddingBatch",
    "QueryEmbeddingFact",
    "QueryEvidenceRelevanceAssessment",
    "QueryEvidenceRelevanceItem",
    "QueryEvidenceSufficiencyAssessment",
    "QueryExecutionReuseLink",
    "QueryExecutorFreezeManifest",
    "QueryExecutorRunManifest",
    "QueryExecutorRunState",
    "QueryExecutorRunStatus",
    "QueryExecutorStage",
    "QueryExecutorStageJob",
    "QueryFallbackDecision",
    "QueryFallbackReason",
    "QueryProgramBinding",
    "QueryRouteDecision",
    "QueryRouteOutcome",
    "QBankRerankerAssessment",
    "QBankRerankerDimensions",
    "QBankRerankerOutput",
    "RerankerDimensionVerdict",
    "RerankerMatchLevel",
    "Step7PilotArtifactRef",
    "Step7BFreezeManifest",
    "Step7PilotDatasetSource",
    "Step7PilotDecision",
    "Step7PilotManifest",
    "Step7PilotSourceManifest",
    "Step7PilotStatus",
]
