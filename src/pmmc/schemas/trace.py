"""Trace schemas for offline analysis and reproducibility."""

from __future__ import annotations

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    AppStage,
    BaseSchema,
    EvidenceType,
    JsonDict,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    TraceStatus,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.baseline import QueryEmbeddingRole, RepresentationKind
from pmmc.schemas.model import EmbeddingResult
from pmmc.schemas.persistence import DataScope
from pmmc.schemas.qbank import QBankMatchCandidate
from pmmc.schemas.score import RankingScore


class ModelCallTrace(BaseSchema):
    """Trace for one LLM, embedding, or reranker call."""

    model_call_id: NonEmptyStr
    run_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr | None = None
    semantic_run_id: NonEmptyStr | None = None
    data_scope: DataScope = DataScope.RUNTIME
    snapshot_id: NonEmptyStr | None = None
    query_id: NonEmptyStr | None = None
    conversation_id: NonEmptyStr | None = None
    job_id: NonEmptyStr | None = None
    compiler_unit_id: NonEmptyStr | None = None
    evaluation_job_id: NonEmptyStr | None = None
    semantic_job_id: NonEmptyStr | None = None
    qbank_id: NonEmptyStr | None = None
    plan_id: NonEmptyStr | None = None
    step_id: NonNegativeInt | None = None
    parent_model_call_id: NonEmptyStr | None = None
    context_assembly_id: NonEmptyStr | None = None
    runtime_id: NonEmptyStr | None = None
    runtime_service_id: NonEmptyStr | None = None
    stage: AppStage
    provider: NonEmptyStr | None = None
    model_alias: NonEmptyStr
    actual_model_id: NonEmptyStr | None = None
    prompt_hash: NonEmptyStr | None = None
    input_asset_id: NonEmptyStr | None = None
    output_asset_id: NonEmptyStr | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    total_tokens: NonNegativeInt | None = None
    cost_usd: NonNegativeFloat | None = None
    latency_ms: NonNegativeFloat | None = None
    temperature: NonNegativeFloat | None = None
    top_p: Probability | None = None
    seed: NonNegativeInt | None = None
    parsed_output: JsonDict | None = None
    parse_error: NonEmptyStr | None = None
    repair_attempt_count: NonNegativeInt = 0
    cache_hit: bool = False
    retry_count: NonNegativeInt = 0
    fallback_used: bool = False
    litellm_call_id: NonEmptyStr | None = None
    vllm_request_id: NonEmptyStr | None = None
    langfuse_trace_id: NonEmptyStr | None = None
    provider_request_id: NonEmptyStr | None = None
    request_config: JsonDict = Field(default_factory=dict)
    error_type: NonEmptyStr | None = None
    status: TraceStatus = TraceStatus.SUCCESS
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_call(self) -> ModelCallTrace:
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        if self.status is TraceStatus.FAILED and self.error_type is None:
            raise ValueError("failed model calls require error_type")
        if self.error_type is not None and self.status is TraceStatus.SUCCESS:
            raise ValueError("model calls with error_type cannot have success status")
        if self.stage is AppStage.JUDGE and self.data_scope is not DataScope.EVALUATION:
            raise ValueError("judge model calls must use evaluation data scope")
        if self.data_scope is DataScope.RUNTIME and any(
            value is not None
            for value in (
                self.evaluation_run_id,
                self.evaluation_job_id,
                self.semantic_run_id,
                self.semantic_job_id,
            )
        ):
            raise ValueError("runtime model calls cannot reference evaluation scopes")
        if self.data_scope is DataScope.EVALUATION and self.job_id is not None:
            raise ValueError("evaluation model calls cannot reference a runtime job")
        if self.evaluation_job_id is not None and self.evaluation_run_id is None:
            raise ValueError("evaluation_job_id requires evaluation_run_id")
        if self.semantic_job_id is not None and self.semantic_run_id is None:
            raise ValueError("semantic_job_id requires semantic_run_id")
        if self.data_scope is DataScope.EVALUATION and (
            (self.evaluation_run_id is None) == (self.semantic_run_id is None)
        ):
            raise ValueError(
                "evaluation model calls require exactly one evaluation parent"
            )
        if self.stage is AppStage.JUDGE and (
            self.evaluation_run_id is None and self.semantic_run_id is None
        ):
            raise ValueError("judge model calls require an evaluation parent")
        if self.query_id is not None and self.snapshot_id is None:
            raise ValueError("query-scoped model calls require snapshot_id")
        if self.compiler_unit_id is not None and self.snapshot_id is None:
            raise ValueError("compiler-unit model calls require snapshot_id")
        if self.compiler_unit_id is not None:
            if self.data_scope is not DataScope.RUNTIME or self.job_id is None:
                raise ValueError(
                    "compiler-unit model calls require runtime scope and a compile job"
                )
            if self.stage not in {
                AppStage.QUESTIONER,
                AppStage.PLANNER,
                AppStage.DOUBTER,
            }:
                raise ValueError(
                    "compiler-unit model calls must use questioner, planner, or doubter stage"
                )
        if self.parent_model_call_id == self.model_call_id:
            raise ValueError("model calls cannot be their own parent")
        if self.runtime_service_id is not None and self.runtime_id is None:
            raise ValueError("runtime_service_id requires runtime_id")
        return self


class AuditedEmbeddingResult(BaseSchema):
    """Embedding provider result paired with the persisted model-call trace."""

    result: EmbeddingResult
    trace: ModelCallTrace


class RetrievalBatchTrace(BaseSchema):
    """One retrieval invocation containing an ordered candidate list."""

    retrieval_batch_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    stage: AppStage
    method: NonEmptyStr
    retriever_name: NonEmptyStr
    index_name: NonEmptyStr | None = None
    chunking_strategy: NonEmptyStr | None = None
    requested_top_k: PositiveInt
    returned_count: NonNegativeInt
    embedding_collection_id: NonEmptyStr | None = None
    query_embedding_collection_id: NonEmptyStr | None = None
    corpus_id: NonEmptyStr | None = None
    reranker_model_call_id: NonEmptyStr | None = None
    plan_id: NonEmptyStr | None = None
    step_id: NonNegativeInt | None = None
    embed_latency_ms: NonNegativeFloat | None = None
    search_latency_ms: NonNegativeFloat | None = None
    rerank_latency_ms: NonNegativeFloat | None = None
    total_latency_ms: NonNegativeFloat | None = None
    status: TraceStatus = TraceStatus.SUCCESS
    error_type: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_batch(self) -> RetrievalBatchTrace:
        if self.returned_count > self.requested_top_k:
            raise ValueError("returned_count cannot exceed requested_top_k")
        if self.status is TraceStatus.FAILED and self.error_type is None:
            raise ValueError("failed retrieval batches require error_type")
        if self.error_type is not None and self.status is TraceStatus.SUCCESS:
            raise ValueError("retrieval batches with error_type cannot be successful")
        return self


class RetrievalTrace(BaseSchema):
    """One method-produced retrieval result without evaluation labels."""

    retrieval_trace_id: NonEmptyStr
    retrieval_batch_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    method: NonEmptyStr
    retriever_name: NonEmptyStr
    index_name: NonEmptyStr | None = None
    chunking_strategy: NonEmptyStr | None = None
    rank: PositiveInt
    retrieved_id: NonEmptyStr
    retrieval_document_id: NonEmptyStr | None = None
    retrieval_window_id: NonEmptyStr | None = None
    retrieved_type: EvidenceType = EvidenceType.UNKNOWN
    score: RankingScore | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_document_link(self) -> RetrievalTrace:
        if self.retrieval_window_id is not None:
            if self.retrieved_id != self.retrieval_window_id:
                raise ValueError("window-level retrieved_id must equal retrieval_window_id")
            if self.retrieval_document_id is None:
                raise ValueError("window-level retrieval requires a winning document")
        elif (
            self.retrieval_document_id is not None
            and self.retrieval_document_id != self.retrieved_id
        ):
            raise ValueError("document-level retrieval_document_id must equal retrieved_id")
        return self


class RetrievalTraceComponent(BaseSchema):
    """Persisted document score supporting one window-level retrieval trace."""

    retrieval_trace_id: NonEmptyStr
    retrieval_document_id: NonEmptyStr
    embedding_collection_id: NonEmptyStr
    representation: RepresentationKind
    ordinal: NonNegativeInt
    score: RankingScore
    is_winner: bool = False
    metadata: JsonDict = Field(default_factory=dict)


class RetrievalQueryEmbeddingRef(BaseSchema):
    """Link a retrieval batch to its component and effective query vectors."""

    retrieval_batch_id: NonEmptyStr
    embedding_id: NonEmptyStr
    role: QueryEmbeddingRole
    ordinal: NonNegativeInt
    used_for_search: bool = False
    metadata: JsonDict = Field(default_factory=dict)


class RetrievalEvaluation(BaseSchema):
    """Evaluation-only gold annotation for one retrieval trace."""

    retrieval_evaluation_id: NonEmptyStr
    retrieval_trace_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr | None = None
    is_gold_evidence: bool
    matched_gold_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ids(self) -> RetrievalEvaluation:
        ensure_unique_strings(self.matched_gold_evidence_ids, "matched_gold_evidence_ids")
        if self.is_gold_evidence != bool(self.matched_gold_evidence_ids):
            raise ValueError("is_gold_evidence must agree with matched_gold_evidence_ids")
        return self


class QueryMatchingTrace(BaseSchema):
    """Trace for matching a runtime-safe query to the question bank."""

    query_id: NonEmptyStr
    run_id: NonEmptyStr
    raw_query: NonEmptyStr | None = None
    query_image_ids: list[NonEmptyStr] = Field(default_factory=list)
    top_qbank_candidates: list[QBankMatchCandidate] = Field(default_factory=list)
    selected_qbank_id: NonEmptyStr | None = None
    top1_top2_normalized_gap: Probability | None = None
    match_confidence: Probability | None = None
    fallback_triggered: bool = False
    fallback_reason: NonEmptyStr | None = None
    status: TraceStatus = TraceStatus.SUCCESS
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_matching(self) -> QueryMatchingTrace:
        ensure_unique_strings(self.query_image_ids, "query_image_ids")
        qbank_ids = [candidate.qbank_id for candidate in self.top_qbank_candidates]
        ensure_unique_strings(qbank_ids, "top_qbank_candidates.qbank_id")

        ranks = [candidate.rank for candidate in self.top_qbank_candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("candidate ranks must be unique, contiguous, and ordered from 1")

        if self.selected_qbank_id is not None:
            selected = next(
                (
                    candidate
                    for candidate in self.top_qbank_candidates
                    if candidate.qbank_id == self.selected_qbank_id
                ),
                None,
            )
            if selected is None:
                raise ValueError("selected_qbank_id must exist in top_qbank_candidates")
            if selected.final_score is None or selected.final_score.normalized_value is None:
                raise ValueError("selected qbank candidate requires a normalized final score")

        if self.top1_top2_normalized_gap is not None:
            if len(self.top_qbank_candidates) < 2:
                raise ValueError("top1_top2_normalized_gap requires at least two candidates")
            top_two_scores = [candidate.final_score for candidate in self.top_qbank_candidates[:2]]
            if any(score is None or score.normalized_value is None for score in top_two_scores):
                raise ValueError(
                    "top1_top2_normalized_gap requires normalized final scores "
                    "for top two candidates"
                )
        if self.fallback_triggered and self.fallback_reason is None:
            raise ValueError("fallback_triggered requires fallback_reason")
        if not self.fallback_triggered and self.fallback_reason is not None:
            raise ValueError("fallback_reason is only valid when fallback is triggered")
        return self


class ProgramExecutionStepTrace(BaseSchema):
    """Trace for one query-time memory-program operator step."""

    execution_id: NonEmptyStr | None = None
    query_id: NonEmptyStr | None = None
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr | None = None
    compiler_unit_id: NonEmptyStr | None = None
    plan_id: NonEmptyStr
    step_id: NonNegativeInt
    op_name: NonEmptyStr
    retrieved_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_image_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_caption_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    scores: dict[NonEmptyStr, RankingScore] = Field(default_factory=dict)
    input_hash: NonEmptyStr | None = None
    output_hash: NonEmptyStr | None = None
    output: JsonDict = Field(default_factory=dict)
    success: bool = True
    error: NonEmptyStr | None = None
    latency_ms: NonNegativeFloat | None = None
    status: TraceStatus = TraceStatus.SUCCESS
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_execution(self) -> ProgramExecutionStepTrace:
        for field_name in (
            "retrieved_memory_ids",
            "retrieved_image_ids",
            "retrieved_caption_ids",
            "retrieved_session_ids",
        ):
            ensure_unique_strings(getattr(self, field_name), field_name)
        if self.success and self.error is not None:
            raise ValueError("successful execution step cannot include error")
        if not self.success and self.error is None:
            raise ValueError("failed execution step requires error")
        if self.status is TraceStatus.FAILED and self.success:
            raise ValueError("failed trace status requires success=False")
        if self.query_id is None and self.compiler_unit_id is None:
            raise ValueError("execution steps require either query_id or compiler_unit_id")
        if self.execution_id is not None and self.snapshot_id is None:
            raise ValueError("persisted execution steps require snapshot_id")
        return self
