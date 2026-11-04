"""Shared contracts for the five foundational PMMC baselines."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.model import GenerationRequest
from pmmc.schemas.persistence import DistanceMetric
from pmmc.schemas.query import QueryInput
from pmmc.schemas.score import RankingScore


class BaselineMethod(StrEnum):
    """Stable method IDs used by manifests, tables, and paper exports."""

    FULL_CONTEXT_CAPTIONS = "full_context_captions"
    FULL_CONTEXT_RAW_IMAGES = "full_context_raw_images"
    CAPTION_TEXT_RAG = "caption_text_rag"
    CONTRASTIVE_MULTIMODAL_RAG = "contrastive_multimodal_rag"
    NATIVE_MULTIMODAL_RAG = "native_multimodal_rag"


class RepresentationKind(StrEnum):
    TEXT_CAPTION = "text_caption"
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class CorpusStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class TruncationPolicy(StrEnum):
    KEEP_RECENT_COMPLETE_UNITS_V1 = "keep_recent_complete_units_v1"
    RETRIEVAL_RANK_COMPLETE_UNITS_V1 = "retrieval_rank_complete_units_v1"


class QueryEmbeddingRole(StrEnum):
    """Role of one persisted query-side embedding in a retrieval batch."""

    TEXT = "text"
    IMAGE = "image"
    JOINT = "joint"
    FUSED = "fused"


class ContextCandidateDisposition(StrEnum):
    """How one answer-time retrieval candidate affected the final prompt."""

    INCLUDED = "included"
    DEDUPLICATED = "deduplicated"
    BUDGET_EXCLUDED = "budget_excluded"


class RetrievalScoreComponent(BaseSchema):
    """One document-level score contributing to a window-level result."""

    retrieval_document_id: NonEmptyStr
    embedding_collection_id: NonEmptyStr
    representation: RepresentationKind
    score: RankingScore
    metadata: JsonDict = Field(default_factory=dict)


class WindowSearchHit(BaseSchema):
    """One exact-search result aggregated to a unique parent memory window."""

    window_id: NonEmptyStr
    rank: PositiveInt
    score: RankingScore
    winning_document_id: NonEmptyStr
    components: list[RetrievalScoreComponent]
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_components(self) -> WindowSearchHit:
        if not self.components:
            raise ValueError("window search hits require at least one score component")
        component_ids = [item.retrieval_document_id for item in self.components]
        ensure_unique_strings(component_ids, "components.retrieval_document_id")
        if self.winning_document_id not in component_ids:
            raise ValueError("winning_document_id must exist in components")
        winner = next(
            item
            for item in self.components
            if item.retrieval_document_id == self.winning_document_id
        )
        if winner.score != self.score:
            raise ValueError("window score must equal the winning document score")
        return self


class ContextRetrievalCandidate(BaseSchema):
    """Answer-time disposition and contribution of one ranked window."""

    retrieval_trace_id: NonEmptyStr
    window_id: NonEmptyStr
    rank: PositiveInt
    disposition: ContextCandidateDisposition
    contributed_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contribution(self) -> ContextRetrievalCandidate:
        ensure_unique_strings(self.contributed_memory_ids, "contributed_memory_ids")
        if (
            self.disposition is ContextCandidateDisposition.INCLUDED
            and not self.contributed_memory_ids
        ):
            raise ValueError("included retrieval candidates must contribute memory units")
        if (
            self.disposition is not ContextCandidateDisposition.INCLUDED
            and self.contributed_memory_ids
        ):
            raise ValueError("excluded or deduplicated candidates cannot contribute memory units")
        return self


class BaselineCorpusSpec(BaseSchema):
    """Complete, hashable recipe for one snapshot-derived retrieval corpus."""

    snapshot_id: NonEmptyStr
    dataset: DatasetName
    snapshot_content_hash: NonEmptyStr
    tokenizer_alias: NonEmptyStr
    tokenizer_revision: NonEmptyStr
    renderer_version: NonEmptyStr = "baseline-renderer-v1"
    window_strategy: NonEmptyStr = "session_token_window_v1"
    max_text_tokens: PositiveInt = 512
    max_images: PositiveInt = 4
    overlap_memory_units: NonNegativeInt = 1
    cross_session: bool = False
    representations: list[RepresentationKind] = Field(
        default_factory=lambda: list(RepresentationKind)
    )
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_spec(self) -> BaselineCorpusSpec:
        if self.cross_session:
            raise ValueError("session_token_window_v1 cannot cross session boundaries")
        if not self.representations:
            raise ValueError("representations cannot be empty")
        if len(self.representations) != len(set(self.representations)):
            raise ValueError("representations cannot contain duplicates")
        return self


class BaselineCorpusManifest(BaseSchema):
    """Immutable lineage and content hashes for one derived corpus."""

    corpus_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: DatasetName
    spec: BaselineCorpusSpec
    status: CorpusStatus = CorpusStatus.READY
    content_hash: NonEmptyStr
    windows_hash: NonEmptyStr
    documents_hash: NonEmptyStr
    window_count: NonNegativeInt
    document_count: NonNegativeInt
    image_document_count: NonNegativeInt
    unique_memory_count: NonNegativeInt
    unique_image_count: NonNegativeInt
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_lineage(self) -> BaselineCorpusManifest:
        if self.snapshot_id != self.spec.snapshot_id or self.dataset is not self.spec.dataset:
            raise ValueError("manifest and corpus spec snapshot lineage must match")
        if self.image_document_count > self.document_count:
            raise ValueError("image_document_count cannot exceed document_count")
        return self


class MemoryWindow(BaseSchema):
    """One deterministic, session-local group of whole MemoryUnit records."""

    window_id: NonEmptyStr
    corpus_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: DatasetName
    conversation_id: NonEmptyStr
    session_id: NonEmptyStr
    window_ordinal: NonNegativeInt
    memory_ids: list[NonEmptyStr]
    start_timestamp_order: NonNegativeInt
    end_timestamp_order: NonNegativeInt
    text_caption_token_count: NonNegativeInt
    image_ids: list[NonEmptyStr] = Field(default_factory=list)
    available_after_session: NonEmptyStr
    oversized_single_unit: bool = False
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_window(self) -> MemoryWindow:
        if not self.memory_ids:
            raise ValueError("memory windows require at least one memory ID")
        ensure_unique_strings(self.memory_ids, "memory_ids")
        ensure_unique_strings(self.image_ids, "image_ids")
        if self.end_timestamp_order < self.start_timestamp_order:
            raise ValueError("window end order cannot precede start order")
        if self.oversized_single_unit and len(self.memory_ids) != 1:
            raise ValueError("oversized_single_unit windows must contain exactly one unit")
        return self


class RetrievalDocument(BaseSchema):
    """One representation-specific retrieval document derived from a window."""

    document_id: NonEmptyStr
    corpus_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: DatasetName
    conversation_id: NonEmptyStr
    session_id: NonEmptyStr
    parent_window_id: NonEmptyStr
    representation: RepresentationKind
    memory_ids: list[NonEmptyStr]
    image_ids: list[NonEmptyStr] = Field(default_factory=list)
    rendered_text: str | None = None
    token_count: NonNegativeInt | None = None
    available_after_session: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_document(self) -> RetrievalDocument:
        if not self.memory_ids:
            raise ValueError("retrieval documents require source memory IDs")
        ensure_unique_strings(self.memory_ids, "memory_ids")
        ensure_unique_strings(self.image_ids, "image_ids")
        if self.representation in {RepresentationKind.TEXT, RepresentationKind.TEXT_CAPTION}:
            if not self.rendered_text:
                raise ValueError("text representations require rendered_text")
            if self.representation is RepresentationKind.TEXT and self.image_ids:
                raise ValueError("text representation cannot include images")
        elif self.representation is RepresentationKind.IMAGE:
            if len(self.image_ids) != 1 or self.rendered_text is not None:
                raise ValueError("image representation requires exactly one image and no text")
        elif not self.rendered_text and not self.image_ids:
            raise ValueError("multimodal representation requires text or images")
        return self


class MemoryTokenCount(BaseSchema):
    """Auditable exact or estimated count for one context candidate."""

    memory_id: NonEmptyStr
    text_tokens: NonNegativeInt
    image_tokens: NonNegativeInt = 0


class ContextAssemblyTrace(BaseSchema):
    """Deterministic audit of one full-context prompt budget decision."""

    context_assembly_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    conversation_id: NonEmptyStr
    job_id: NonEmptyStr | None = None
    retrieval_batch_id: NonEmptyStr | None = None
    method: BaselineMethod
    representation: RepresentationKind
    truncation_policy: TruncationPolicy = TruncationPolicy.KEEP_RECENT_COMPLETE_UNITS_V1
    tokenizer_alias: NonEmptyStr
    tokenizer_revision: NonEmptyStr
    total_context_tokens: PositiveInt = 32768
    reserved_output_tokens: PositiveInt = 1024
    max_input_tokens: PositiveInt = 31744
    instruction_tokens: NonNegativeInt
    query_tokens: NonNegativeInt
    query_image_tokens: NonNegativeInt = 0
    memory_text_tokens: NonNegativeInt = 0
    memory_image_tokens: NonNegativeInt = 0
    request_token_adjustment: int = 0
    input_tokens: NonNegativeInt
    estimated: bool
    memory_token_counts: list[MemoryTokenCount] = Field(default_factory=list)
    included_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    truncated_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    included_image_ids: list[NonEmptyStr] = Field(default_factory=list)
    truncated_image_ids: list[NonEmptyStr] = Field(default_factory=list)
    query_image_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieval_candidates: list[ContextRetrievalCandidate] = Field(default_factory=list)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_budget(self) -> ContextAssemblyTrace:
        if self.max_input_tokens != self.total_context_tokens - self.reserved_output_tokens:
            raise ValueError("max_input_tokens must equal total minus reserved output")
        expected = (
            self.instruction_tokens
            + self.query_tokens
            + self.query_image_tokens
            + self.memory_text_tokens
            + self.memory_image_tokens
            + self.request_token_adjustment
        )
        if self.input_tokens != expected:
            raise ValueError("input_tokens must equal the recorded budget components")
        if self.input_tokens > self.max_input_tokens:
            raise ValueError("assembled context exceeds max_input_tokens")
        for field_name in (
            "included_memory_ids",
            "truncated_memory_ids",
            "included_image_ids",
            "truncated_image_ids",
            "query_image_ids",
        ):
            ensure_unique_strings(getattr(self, field_name), field_name)
        if set(self.included_memory_ids) & set(self.truncated_memory_ids):
            raise ValueError("memory IDs cannot be both included and truncated")
        if set(self.included_image_ids) & set(self.truncated_image_ids):
            raise ValueError("image IDs cannot be both included and truncated")
        token_count_ids = [item.memory_id for item in self.memory_token_counts]
        ensure_unique_strings(token_count_ids, "memory_token_counts.memory_id")
        if set(token_count_ids) != set(
            [*self.included_memory_ids, *self.truncated_memory_ids]
        ):
            raise ValueError(
                "memory_token_counts must cover every included and truncated memory exactly once"
            )
        if self.method is BaselineMethod.FULL_CONTEXT_CAPTIONS:
            if self.representation is not RepresentationKind.TEXT_CAPTION:
                raise ValueError("caption full-context requires text_caption representation")
        if self.method is BaselineMethod.FULL_CONTEXT_RAW_IMAGES:
            if self.representation is not RepresentationKind.MULTIMODAL:
                raise ValueError("raw-image full-context requires multimodal representation")
        rag_methods = {
            BaselineMethod.CAPTION_TEXT_RAG,
            BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
            BaselineMethod.NATIVE_MULTIMODAL_RAG,
        }
        if self.method in rag_methods:
            if self.retrieval_batch_id is None:
                raise ValueError("RAG context assemblies require retrieval_batch_id")
            if self.truncation_policy is not TruncationPolicy.RETRIEVAL_RANK_COMPLETE_UNITS_V1:
                raise ValueError("RAG context assemblies require retrieval-rank truncation")
            ranks = [item.rank for item in self.retrieval_candidates]
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError("RAG context candidate ranks must be contiguous from one")
            ensure_unique_strings(
                [item.retrieval_trace_id for item in self.retrieval_candidates],
                "retrieval_candidates.retrieval_trace_id",
            )
            ensure_unique_strings(
                [item.window_id for item in self.retrieval_candidates],
                "retrieval_candidates.window_id",
            )
            contributed = [
                memory_id
                for item in self.retrieval_candidates
                for memory_id in item.contributed_memory_ids
            ]
            ensure_unique_strings(contributed, "retrieval_candidates.contributed_memory_ids")
            if contributed != self.included_memory_ids:
                raise ValueError(
                    "RAG candidate contributions must equal included_memory_ids in prompt order"
                )
        elif self.retrieval_batch_id is not None or self.retrieval_candidates:
            raise ValueError("full-context assemblies cannot include retrieval lineage")
        return self


class PreparedAnswerInput(BaseSchema):
    """Runtime-safe, fully budgeted answer input shared by baseline producers."""

    method: BaselineMethod
    query: QueryInput
    request: GenerationRequest
    context_assembly: ContextAssemblyTrace
    evidence_ids_in_prompt: list[NonEmptyStr] = Field(default_factory=list)
    image_ids_in_prompt: list[NonEmptyStr] = Field(default_factory=list)
    missing_caption_memory_count: NonNegativeInt = 0
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_lineage(self) -> PreparedAnswerInput:
        ensure_unique_strings(self.evidence_ids_in_prompt, "evidence_ids_in_prompt")
        ensure_unique_strings(self.image_ids_in_prompt, "image_ids_in_prompt")
        if self.query.query_id != self.context_assembly.query_id:
            raise ValueError("prepared query and context assembly must match")
        if self.method is not self.context_assembly.method:
            raise ValueError("prepared method and context assembly must match")
        if self.evidence_ids_in_prompt != self.context_assembly.included_memory_ids:
            raise ValueError("prepared evidence IDs must equal included context memories")
        request_images = [
            part.image.image_id
            for message in self.request.messages
            for part in message.parts
            if part.image is not None
        ]
        if request_images != self.image_ids_in_prompt:
            raise ValueError("prepared image IDs must preserve final request order")
        return self


class EmbeddingSearchHit(BaseSchema):
    """Runtime-safe exact-search hit returned by the embedding repository."""

    embedding_id: NonEmptyStr
    collection_id: NonEmptyStr
    retrieval_document_id: NonEmptyStr
    rank: PositiveInt
    score: RankingScore
    distance_metric: DistanceMetric
    metadata: JsonDict = Field(default_factory=dict)
