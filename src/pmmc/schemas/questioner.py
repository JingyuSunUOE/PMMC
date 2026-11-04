"""Questioner production, QBank indexing, and coverage-gate contracts."""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    FiniteFloat,
    JsonDict,
    Modality,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    QuestionType,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.compiler import OperatorEvidence
from pmmc.schemas.persistence import EmbeddingCollectionStatus
from pmmc.schemas.score import RankingScore

QUESTIONER_V1_TYPES = frozenset(
    {
        QuestionType.TEXTUAL_FACT,
        QuestionType.VISUAL_DETAIL,
        QuestionType.IMAGE_TEXT_BINDING,
        QuestionType.VISUAL_SEARCH,
        QuestionType.MULTI_HOP,
        QuestionType.TEMPORAL,
        QuestionType.KNOWLEDGE_UPDATE,
        QuestionType.CONFLICT_DETECTION,
        QuestionType.ANSWER_REFUSAL,
        QuestionType.PERSONAL_PREFERENCE,
    }
)


def _normalize_user_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


class QuestionReasoningType(StrEnum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    UPDATE = "update"
    CONFLICT = "conflict"
    REFUSAL = "refusal"


class QuestionerV2QuestionType(StrEnum):
    """Question types accepted by the provider-facing Questioner v2 schema."""

    TEXTUAL_FACT = "textual_fact"
    VISUAL_DETAIL = "visual_detail"
    IMAGE_TEXT_BINDING = "image_text_binding"
    VISUAL_SEARCH = "visual_search"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    KNOWLEDGE_UPDATE = "knowledge_update"
    CONFLICT_DETECTION = "conflict_detection"
    ANSWER_REFUSAL = "answer_refusal"
    PERSONAL_PREFERENCE = "personal_preference"


class QuestionerV2EvidenceType(StrEnum):
    MEMORY = "memory"
    IMAGE = "image"
    SESSION = "session"
    CAPTION = "caption"


class QuestionerV2Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class QuestionerCandidateKind(StrEnum):
    ANSWERABLE = "answerable"
    REFUSAL = "refusal"


class QuestionerCandidateOrigin(StrEnum):
    GENERATED = "generated"
    REVISED = "revised"


class QuestionerAnswerClaimKind(StrEnum):
    """Evidence-backed claim used to assemble a v3.3 expected answer."""

    TEXT = "text"
    VISUAL = "visual"


class QuestionerAnswerType(StrEnum):
    ENTITY = "entity"
    VALUE = "value"
    COUNT = "count"
    LIST = "list"
    BOOLEAN = "boolean"
    EVENT = "event"
    REFUSAL = "refusal"


class QuestionOpportunityType(StrEnum):
    LOCAL_FACT = "local_fact"
    VISUAL_DETAIL = "visual_detail"
    IMAGE_TEXT_BINDING = "image_text_binding"
    VISUAL_SEARCH = "visual_search"
    TEMPORAL_PAIR = "temporal_pair"
    UPDATE_PAIR = "update_pair"
    CONFLICT_PAIR = "conflict_pair"
    PREFERENCE = "preference"
    MULTI_EVIDENCE = "multi_evidence"
    REFUSAL_BOUNDARY = "refusal_boundary"


class QuestionerVerificationVerdict(StrEnum):
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


class QuestionerVerificationReason(StrEnum):
    NOT_USER_FACING = "not_user_facing"
    NOT_FUTURE_PLAUSIBLE = "not_future_plausible"
    NOT_ANSWERABLE = "not_answerable"
    ANSWER_UNSUPPORTED = "answer_unsupported"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    TAXONOMY_MISMATCH = "taxonomy_mismatch"
    MODALITY_MISMATCH = "modality_mismatch"
    VISUAL_NOT_NEEDED = "visual_not_needed"
    MULTI_HOP_NOT_GENUINE = "multi_hop_not_genuine"
    TOO_GENERIC = "too_generic"
    HISTORICAL_REPLAY = "historical_replay"
    INTERNAL_METADATA = "internal_metadata"
    UNSAFE_LEAKAGE = "unsafe_leakage"


class CompilationUnitRelationType(StrEnum):
    ADJACENT_SESSION = "adjacent_session"
    SHARED_ENTITY = "shared_entity"
    SHARED_IMAGE = "shared_image"
    PROFILE_PREFERENCE = "profile_preference"
    TEMPORAL_NEIGHBOR = "temporal_neighbor"


class QuestionerParseStatus(StrEnum):
    PARSED = "parsed"
    REPAIRED = "repaired"
    PARTIAL = "partial"
    FAILED = "failed"


class CandidateGateStatus(StrEnum):
    ELIGIBLE_FOR_PLANNING = "eligible_for_planning"
    REJECTED = "rejected"


class CandidateRejectionCode(StrEnum):
    INVALID_SCHEMA = "invalid_schema"
    UNSUPPORTED_QUESTION_TYPE = "unsupported_question_type"
    FUTURE_OR_FOREIGN_REFERENCE = "future_or_foreign_reference"
    MISSING_EXPECTED_ANSWER = "missing_expected_answer"
    MISSING_EXPECTED_EVIDENCE = "missing_expected_evidence"
    MODALITY_MISMATCH = "modality_mismatch"
    VISUAL_EVIDENCE_MISSING = "visual_evidence_missing"
    MULTI_HOP_EVIDENCE_MISSING = "multi_hop_evidence_missing"
    DUPLICATE = "duplicate"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERNAL_IDENTIFIER_LEAK = "internal_identifier_leak"
    SYSTEM_METADATA_REFERENCE = "system_metadata_reference"
    ASSISTANT_FOLLOW_UP = "assistant_follow_up"
    HISTORICAL_QUESTION_REPLAY = "historical_question_replay"
    GENERIC_EXTERNAL_QUESTION = "generic_external_question"
    PLACEHOLDER_EXPECTED_ANSWER = "placeholder_expected_answer"
    OPPORTUNITY_MISMATCH = "opportunity_mismatch"
    TAXONOMY_INVARIANT_FAILED = "taxonomy_invariant_failed"
    REASONING_INVARIANT_FAILED = "reasoning_invariant_failed"
    EVIDENCE_NOT_ENTAILED = "evidence_not_entailed"
    SEMANTIC_REVISION_REQUIRED = "semantic_revision_required"
    SEMANTIC_VERIFICATION_FAILED = "semantic_verification_failed"
    REVISION_EXHAUSTED = "revision_exhausted"
    QUESTION_ANSWER_CLOSURE_FAILED = "question_answer_closure_failed"
    ANSWER_TYPE_MISMATCH = "answer_type_mismatch"
    QUESTION_AS_ANSWER = "question_as_answer"
    AI_DISCLAIMER = "ai_disclaimer"
    INCOMPLETE_ANSWER = "incomplete_answer"
    SOURCE_UNNATURAL = "source_unnatural"


# Static gates run before semantic verification; completion/export must agree.
QUESTIONER_STATIC_REJECTION_CODES = frozenset(
    code.value
    for code in (
        CandidateRejectionCode.INVALID_SCHEMA,
        CandidateRejectionCode.UNSUPPORTED_QUESTION_TYPE,
        CandidateRejectionCode.FUTURE_OR_FOREIGN_REFERENCE,
        CandidateRejectionCode.MISSING_EXPECTED_ANSWER,
        CandidateRejectionCode.MISSING_EXPECTED_EVIDENCE,
        CandidateRejectionCode.MODALITY_MISMATCH,
        CandidateRejectionCode.VISUAL_EVIDENCE_MISSING,
        CandidateRejectionCode.MULTI_HOP_EVIDENCE_MISSING,
        CandidateRejectionCode.INTERNAL_IDENTIFIER_LEAK,
        CandidateRejectionCode.SYSTEM_METADATA_REFERENCE,
        CandidateRejectionCode.ASSISTANT_FOLLOW_UP,
        CandidateRejectionCode.HISTORICAL_QUESTION_REPLAY,
        CandidateRejectionCode.GENERIC_EXTERNAL_QUESTION,
        CandidateRejectionCode.PLACEHOLDER_EXPECTED_ANSWER,
        CandidateRejectionCode.OPPORTUNITY_MISMATCH,
        CandidateRejectionCode.TAXONOMY_INVARIANT_FAILED,
        CandidateRejectionCode.REASONING_INVARIANT_FAILED,
        CandidateRejectionCode.EVIDENCE_NOT_ENTAILED,
        CandidateRejectionCode.QUESTION_ANSWER_CLOSURE_FAILED,
        CandidateRejectionCode.ANSWER_TYPE_MISMATCH,
        CandidateRejectionCode.QUESTION_AS_ANSWER,
        CandidateRejectionCode.AI_DISCLAIMER,
        CandidateRejectionCode.INCOMPLETE_ANSWER,
        CandidateRejectionCode.SOURCE_UNNATURAL,
    )
)


class QBankMatchChannel(StrEnum):
    LEXICAL = "lexical"
    TEXT_SEMANTIC = "text_semantic"
    MULTIMODAL_SEMANTIC = "multimodal_semantic"
    HYBRID_RRF = "hybrid_rrf"


class QBankCoverageStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CoverageGateMode(StrEnum):
    REPORT_ONLY = "report_only"
    ENFORCE = "enforce"


class QBankCoverageLabelCategory(StrEnum):
    QUESTIONER_COVERAGE_FAILURE = "questioner_coverage_failure"
    MATCHING_FAILURE = "matching_failure"
    GOLD_UNRESOLVABLE = "gold_unresolvable"


class QuestionerCandidateDraft(BaseSchema):
    """Provider-facing candidate before stable experiment lineage is assigned."""

    generated_question: NonEmptyStr
    question_type: QuestionType
    reasoning_type: QuestionReasoningType
    expected_answer: NonEmptyStr | None = None
    source_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    expected_evidence: list[OperatorEvidence] = Field(default_factory=list)
    required_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    evidence_requirement: list[NonEmptyStr] = Field(default_factory=list)
    hop_count: PositiveInt = 1
    paraphrases: list[NonEmptyStr] = Field(default_factory=list)
    confidence: Probability = 0.0
    opportunity_id: NonEmptyStr | None = None
    candidate_kind: QuestionerCandidateKind = QuestionerCandidateKind.ANSWERABLE
    candidate_origin: QuestionerCandidateOrigin = QuestionerCandidateOrigin.GENERATED
    candidate_version: int = Field(default=0, ge=0, le=1)
    supersedes_qbank_id: NonEmptyStr | None = None
    answer_type: QuestionerAnswerType | None = None

    @model_validator(mode="after")
    def _validate_draft(self) -> QuestionerCandidateDraft:
        ensure_unique_strings(self.source_memory_ids, "source_memory_ids")
        ensure_unique_strings(self.paraphrases, "paraphrases")
        ensure_unique_strings(
            [item.evidence_id for item in self.expected_evidence],
            "expected_evidence.evidence_id",
        )
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicates")
        if not self.required_modalities:
            raise ValueError("required_modalities must not be empty")
        revised = self.candidate_origin is QuestionerCandidateOrigin.REVISED
        if revised != (self.candidate_version == 1):
            raise ValueError("revised draft requires candidate_version=1")
        if revised != (self.supersedes_qbank_id is not None):
            raise ValueError("revised draft requires supersedes_qbank_id")
        return self


class QuestionerBatch(BaseSchema):
    candidates: list[QuestionerCandidateDraft] = Field(default_factory=list, max_length=8)


class QuestionerEvidenceDraftV2(BaseSchema):
    """Simple typed reference emitted by a Questioner v2 provider."""

    evidence_id: NonEmptyStr
    evidence_type: QuestionerV2EvidenceType


class QuestionerCandidateDraftV2(BaseSchema):
    """Reduced provider contract mapped into the internal candidate contract."""

    generated_question: NonEmptyStr
    question_type: QuestionerV2QuestionType
    reasoning_type: QuestionReasoningType
    expected_answer: NonEmptyStr | None = None
    source_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    expected_evidence: list[QuestionerEvidenceDraftV2] = Field(default_factory=list)
    required_modalities: list[QuestionerV2Modality] = Field(
        default_factory=lambda: [QuestionerV2Modality.TEXT]
    )
    evidence_requirement: list[NonEmptyStr] = Field(default_factory=list)
    hop_count: PositiveInt = 1
    paraphrases: list[NonEmptyStr] = Field(default_factory=list, max_length=3)
    confidence: Probability = 0.0

    @model_validator(mode="after")
    def _validate_draft(self) -> QuestionerCandidateDraftV2:
        ensure_unique_strings(self.source_memory_ids, "source_memory_ids")
        ensure_unique_strings(self.paraphrases, "paraphrases")
        ensure_unique_strings(
            [item.evidence_id for item in self.expected_evidence],
            "expected_evidence.evidence_id",
        )
        if not self.required_modalities:
            raise ValueError("required_modalities must not be empty")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicates")
        return self


class QuestionerBatchV2(BaseSchema):
    candidates: list[QuestionerCandidateDraftV2] = Field(
        default_factory=list,
        max_length=8,
    )


class QuestionerCandidateDraftV3(BaseSchema):
    """Alias-only provider contract for Questioner v3."""

    candidate_kind: QuestionerCandidateKind
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    question_type: QuestionerV2QuestionType
    reasoning_type: QuestionReasoningType
    expected_answer: NonEmptyStr | None
    evidence_aliases: list[NonEmptyStr]
    required_modalities: list[QuestionerV2Modality]
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability

    @model_validator(mode="after")
    def _validate_v3_draft(self) -> QuestionerCandidateDraftV3:
        ensure_unique_strings(self.evidence_aliases, "evidence_aliases")
        ensure_unique_strings(self.paraphrases, "paraphrases")
        if not self.required_modalities:
            raise ValueError("required_modalities must not be empty")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicates")
        refusal = self.candidate_kind is QuestionerCandidateKind.REFUSAL
        if refusal:
            if self.question_type is not QuestionerV2QuestionType.ANSWER_REFUSAL:
                raise ValueError("refusal candidates require answer_refusal")
            if self.reasoning_type is not QuestionReasoningType.REFUSAL:
                raise ValueError("refusal candidates require refusal reasoning")
            if self.expected_answer is not None or self.evidence_aliases:
                raise ValueError("refusal candidates cannot contain answer evidence")
        elif self.expected_answer is None or not self.evidence_aliases:
            raise ValueError("answerable candidates require answer and evidence")
        return self


class QuestionerBatchV3(BaseSchema):
    candidates: list[QuestionerCandidateDraftV3] = Field(max_length=8)


class QuestionerCandidateDraftV32(BaseSchema):
    """Questioner v3.2 provider contract with server-derived evidence requirements."""

    candidate_kind: QuestionerCandidateKind
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    question_type: QuestionerV2QuestionType
    reasoning_type: QuestionReasoningType
    expected_answer: NonEmptyStr | None
    evidence_aliases: list[NonEmptyStr]
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability

    @model_validator(mode="after")
    def _validate_v32_draft(self) -> QuestionerCandidateDraftV32:
        ensure_unique_strings(self.evidence_aliases, "evidence_aliases")
        ensure_unique_strings(self.paraphrases, "paraphrases")
        refusal = self.candidate_kind is QuestionerCandidateKind.REFUSAL
        if refusal:
            if self.question_type is not QuestionerV2QuestionType.ANSWER_REFUSAL:
                raise ValueError("refusal candidates require answer_refusal")
            if self.reasoning_type is not QuestionReasoningType.REFUSAL:
                raise ValueError("refusal candidates require refusal reasoning")
            if self.expected_answer is not None or self.evidence_aliases:
                raise ValueError("refusal candidates cannot contain answer evidence")
        elif self.expected_answer is None or not self.evidence_aliases:
            raise ValueError("answerable candidates require answer and evidence")
        return self


class QuestionerAnswerableDraftV32(BaseSchema):
    """Provider-only branch whose JSON schema encodes answerable invariants."""

    candidate_kind: Literal[QuestionerCandidateKind.ANSWERABLE]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    question_type: Literal[
        QuestionerV2QuestionType.TEXTUAL_FACT,
        QuestionerV2QuestionType.VISUAL_DETAIL,
        QuestionerV2QuestionType.IMAGE_TEXT_BINDING,
        QuestionerV2QuestionType.VISUAL_SEARCH,
        QuestionerV2QuestionType.MULTI_HOP,
        QuestionerV2QuestionType.TEMPORAL,
        QuestionerV2QuestionType.KNOWLEDGE_UPDATE,
        QuestionerV2QuestionType.CONFLICT_DETECTION,
        QuestionerV2QuestionType.PERSONAL_PREFERENCE,
    ]
    reasoning_type: Literal[
        QuestionReasoningType.SINGLE_HOP,
        QuestionReasoningType.MULTI_HOP,
        QuestionReasoningType.TEMPORAL,
        QuestionReasoningType.UPDATE,
        QuestionReasoningType.CONFLICT,
    ]
    expected_answer: NonEmptyStr
    evidence_aliases: list[NonEmptyStr] = Field(min_length=1)
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability


class QuestionerRefusalDraftV32(BaseSchema):
    """Provider-only branch whose JSON schema encodes refusal invariants."""

    candidate_kind: Literal[QuestionerCandidateKind.REFUSAL]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    question_type: Literal[QuestionerV2QuestionType.ANSWER_REFUSAL]
    reasoning_type: Literal[QuestionReasoningType.REFUSAL]
    expected_answer: None
    evidence_aliases: list[NonEmptyStr] = Field(max_length=0)
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability


QuestionerProviderCandidateV32 = QuestionerAnswerableDraftV32 | QuestionerRefusalDraftV32


class QuestionerBatchV32(BaseSchema):
    candidates: list[QuestionerProviderCandidateV32] = Field(max_length=8)


class QuestionerAnswerClaimDraftV33(BaseSchema):
    """One provider claim grounded in a single local evidence alias."""

    claim_kind: QuestionerAnswerClaimKind
    evidence_alias: NonEmptyStr
    claim_text: NonEmptyStr
    support_quote: NonEmptyStr | None

    @model_validator(mode="after")
    def _validate_claim(self) -> QuestionerAnswerClaimDraftV33:
        if self.claim_kind is QuestionerAnswerClaimKind.TEXT:
            if self.support_quote is None:
                raise ValueError("text answer claims require a support quote")
            if len(self.support_quote.split()) > 32:
                raise ValueError("text answer-claim support quotes are limited to 32 tokens")
        elif self.support_quote is not None:
            raise ValueError("visual answer claims bind an image instead of a text quote")
        return self


class QuestionerTextAnswerClaimDraftV33(QuestionerAnswerClaimDraftV33):
    """Provider claim whose text quote is required by the JSON schema."""

    claim_kind: Literal[QuestionerAnswerClaimKind.TEXT]
    support_quote: NonEmptyStr

    @model_validator(mode="after")
    def _validate_claim(self) -> QuestionerTextAnswerClaimDraftV33:
        # Quote length is a semantic gate because JSON Schema cannot count tokens.
        return self


class QuestionerVisualAnswerClaimDraftV33(QuestionerAnswerClaimDraftV33):
    """Provider claim whose visual binding forbids a text quote in JSON."""

    claim_kind: Literal[QuestionerAnswerClaimKind.VISUAL]
    support_quote: None

    @model_validator(mode="after")
    def _validate_claim(self) -> QuestionerVisualAnswerClaimDraftV33:
        return self


QuestionerProviderAnswerClaimDraftV33 = Annotated[
    QuestionerTextAnswerClaimDraftV33 | QuestionerVisualAnswerClaimDraftV33,
    Field(discriminator="claim_kind"),
]


class QuestionerAnswerableDraftV33(BaseSchema):
    """Native v3.3 answerable candidate; answer/evidence are server-derived."""

    candidate_kind: Literal[QuestionerCandidateKind.ANSWERABLE]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    question_type: Literal[
        QuestionerV2QuestionType.TEXTUAL_FACT,
        QuestionerV2QuestionType.VISUAL_DETAIL,
        QuestionerV2QuestionType.IMAGE_TEXT_BINDING,
        QuestionerV2QuestionType.VISUAL_SEARCH,
        QuestionerV2QuestionType.MULTI_HOP,
        QuestionerV2QuestionType.TEMPORAL,
        QuestionerV2QuestionType.KNOWLEDGE_UPDATE,
        QuestionerV2QuestionType.CONFLICT_DETECTION,
        QuestionerV2QuestionType.PERSONAL_PREFERENCE,
    ]
    reasoning_type: Literal[
        QuestionReasoningType.SINGLE_HOP,
        QuestionReasoningType.MULTI_HOP,
        QuestionReasoningType.TEMPORAL,
        QuestionReasoningType.UPDATE,
        QuestionReasoningType.CONFLICT,
    ]
    answer_claims: list[QuestionerProviderAnswerClaimDraftV33] = Field(
        min_length=1,
        max_length=8,
    )
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability


class QuestionerControlDraftV33(BaseSchema):
    """One optional control/refusal candidate outside the answerable budget."""

    candidate_kind: Literal[QuestionerCandidateKind.REFUSAL]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    question_type: Literal[QuestionerV2QuestionType.ANSWER_REFUSAL]
    reasoning_type: Literal[QuestionReasoningType.REFUSAL]
    answer_claims: list[QuestionerAnswerClaimDraftV33] = Field(max_length=0)
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability


QuestionerProviderCandidateV33 = QuestionerAnswerableDraftV33 | QuestionerControlDraftV33


class QuestionerBatchV33(BaseSchema):
    """Separate scientific and control budgets prevent refusal domination."""

    candidates: list[QuestionerAnswerableDraftV33] = Field(max_length=8)
    control_candidates: list[QuestionerControlDraftV33] = Field(max_length=1)


class QuestionerTextAnswerClaimDraftV34(BaseSchema):
    """Text claim bound to one server-rendered sentence span alias."""

    claim_kind: Literal[QuestionerAnswerClaimKind.TEXT]
    support_alias: NonEmptyStr
    claim_text: NonEmptyStr


class QuestionerVisualAnswerClaimDraftV34(BaseSchema):
    """Visual claim bound to one real image alias."""

    claim_kind: Literal[QuestionerAnswerClaimKind.VISUAL]
    support_alias: NonEmptyStr
    claim_text: NonEmptyStr


QuestionerProviderAnswerClaimDraftV34 = Annotated[
    QuestionerTextAnswerClaimDraftV34 | QuestionerVisualAnswerClaimDraftV34,
    Field(discriminator="claim_kind"),
]


class QuestionerAnswerableDraftV34(BaseSchema):
    """Minimal native v3.4 candidate; taxonomy and evidence are server-derived."""

    candidate_kind: Literal[QuestionerCandidateKind.ANSWERABLE]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    answer_claims: list[QuestionerProviderAnswerClaimDraftV34] = Field(
        min_length=1,
        max_length=8,
    )
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability

    @model_validator(mode="after")
    def _validate_user_facing_text(self) -> QuestionerAnswerableDraftV34:
        ensure_unique_strings(self.paraphrases, "paraphrases")
        normalized = {_normalize_user_question(item) for item in self.paraphrases}
        if len(normalized) != len(self.paraphrases):
            raise ValueError("paraphrases must be unique after normalization")
        if _normalize_user_question(self.generated_question) in normalized:
            raise ValueError("canonical question cannot be repeated as a paraphrase")
        return self


class QuestionerControlDraftV34(BaseSchema):
    candidate_kind: Literal[QuestionerCandidateKind.REFUSAL]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    answer_claims: list[QuestionerProviderAnswerClaimDraftV34] = Field(max_length=0)
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability

    @model_validator(mode="after")
    def _validate_user_facing_text(self) -> QuestionerControlDraftV34:
        ensure_unique_strings(self.paraphrases, "paraphrases")
        normalized = {_normalize_user_question(item) for item in self.paraphrases}
        if len(normalized) != len(self.paraphrases):
            raise ValueError("paraphrases must be unique after normalization")
        if _normalize_user_question(self.generated_question) in normalized:
            raise ValueError("canonical question cannot be repeated as a paraphrase")
        return self


class QuestionerBatchV34(BaseSchema):
    candidates: list[QuestionerAnswerableDraftV34] = Field(max_length=8)
    control_candidates: list[QuestionerControlDraftV34] = Field(max_length=1)


class QuestionerTextAnswerClaimDraftV35(BaseSchema):
    claim_kind: Literal[QuestionerAnswerClaimKind.TEXT]
    answer_type: QuestionerAnswerType
    support_alias: NonEmptyStr
    claim_text: NonEmptyStr


class QuestionerVisualAnswerClaimDraftV35(BaseSchema):
    claim_kind: Literal[QuestionerAnswerClaimKind.VISUAL]
    answer_type: QuestionerAnswerType
    support_alias: NonEmptyStr
    claim_text: NonEmptyStr


QuestionerProviderAnswerClaimDraftV35 = Annotated[
    QuestionerTextAnswerClaimDraftV35 | QuestionerVisualAnswerClaimDraftV35,
    Field(discriminator="claim_kind"),
]


class QuestionerAnswerableDraftV35(BaseSchema):
    candidate_kind: Literal[QuestionerCandidateKind.ANSWERABLE]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    answer_claims: list[QuestionerProviderAnswerClaimDraftV35] = Field(
        min_length=1,
        max_length=8,
    )
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability

    @model_validator(mode="after")
    def _validate_user_facing_text(self) -> QuestionerAnswerableDraftV35:
        ensure_unique_strings(self.paraphrases, "paraphrases")
        normalized = {_normalize_user_question(item) for item in self.paraphrases}
        if len(normalized) != len(self.paraphrases):
            raise ValueError("paraphrases must be unique after normalization")
        if _normalize_user_question(self.generated_question) in normalized:
            raise ValueError("canonical question cannot be repeated as a paraphrase")
        if QuestionerAnswerType.REFUSAL in {item.answer_type for item in self.answer_claims}:
            raise ValueError("answerable candidates cannot emit refusal claims")
        return self


class QuestionerControlDraftV35(BaseSchema):
    candidate_kind: Literal[QuestionerCandidateKind.REFUSAL]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    answer_claims: list[QuestionerProviderAnswerClaimDraftV35] = Field(max_length=0)
    paraphrases: list[NonEmptyStr] = Field(max_length=3)
    confidence: Probability


class QuestionerBatchV35(BaseSchema):
    candidates: list[QuestionerAnswerableDraftV35] = Field(max_length=8)
    control_candidates: list[QuestionerControlDraftV35] = Field(max_length=1)


class QuestionerClaimDraftV36(BaseSchema):
    """Native support references, deliberately independent of the v3.3 quote DTO."""

    claim_kind: QuestionerAnswerClaimKind
    support_alias: NonEmptyStr
    claim_text: NonEmptyStr
    subject: Literal["user", "assistant", "other"]
    assertion: Literal["fact", "plan", "advice"]


class QuestionerAnswerableDraftV36(BaseSchema):
    candidate_kind: Literal[QuestionerCandidateKind.ANSWERABLE]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    answer_type: Literal["entity", "value", "count", "list", "boolean", "event"]
    answer_claims: list[QuestionerClaimDraftV36] = Field(min_length=1, max_length=8)
    confidence: Probability


class QuestionerControlDraftV36(BaseSchema):
    candidate_kind: Literal[QuestionerCandidateKind.REFUSAL]
    opportunity_id: NonEmptyStr
    generated_question: NonEmptyStr
    answer_claims: list[QuestionerClaimDraftV36] = Field(max_length=0)
    confidence: Probability


class QuestionerBatchV36(BaseSchema):
    candidates: list[QuestionerAnswerableDraftV36] = Field(max_length=8)
    control_candidates: list[QuestionerControlDraftV36] = Field(max_length=1)


class QuestionerNativeSupport(BaseSchema):
    support_id: NonEmptyStr
    support_alias: NonEmptyStr
    evidence_alias: NonEmptyStr
    owning_memory_id: NonEmptyStr
    speaker: NonEmptyStr
    source_field: Literal["text", "caption", "image"]
    start_char: NonNegativeInt | None = None
    end_char: NonNegativeInt | None = None
    exact_text: NonEmptyStr | None = None
    image_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_support(self) -> QuestionerNativeSupport:
        if self.source_field == "image":
            if self.image_id is None or any(
                value is not None for value in (self.start_char, self.end_char, self.exact_text)
            ):
                raise ValueError("native_image_support_invalid")
        elif (
            self.start_char is None
            or self.end_char is None
            or self.exact_text is None
            or self.end_char <= self.start_char
            or self.end_char - self.start_char != len(self.exact_text)
            or self.image_id is not None
        ):
            raise ValueError("native_span_range_invalid")
        return self


class QuestionerResolvedClaimV36(QuestionerClaimDraftV36):
    support: QuestionerNativeSupport

    @property
    def support_quote(self) -> str | None:
        return self.support.exact_text


class QuestionerVerifierCandidateV36(QuestionerCandidateDraftV3):
    answer_type: QuestionerAnswerType
    exact_claims: list[QuestionerResolvedClaimV36]
    image_set_scope: bool = False


class QuestionerNativeCandidateDraft(QuestionerCandidateDraft):
    exact_claims: list[QuestionerResolvedClaimV36]


class QuestionerAnswerClaim(BaseSchema):
    answer_claim_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    generation_id: NonEmptyStr
    qbank_id: NonEmptyStr
    ordinal: NonNegativeInt
    claim_kind: QuestionerAnswerClaimKind
    evidence_alias_hash: NonEmptyStr
    evidence_id: NonEmptyStr
    claim_text: NonEmptyStr
    support_quote: NonEmptyStr | None
    quote_verified: bool
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_persisted_claim(self) -> QuestionerAnswerClaim:
        if self.claim_kind is QuestionerAnswerClaimKind.TEXT and (
            self.support_quote is None or not self.quote_verified
        ):
            raise ValueError("persisted text claims require a verified quote")
        if self.claim_kind is QuestionerAnswerClaimKind.VISUAL and self.support_quote is not None:
            raise ValueError("persisted visual claims cannot contain a quote")
        return self


class QBankCandidateFamily(BaseSchema):
    family_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    primary_qbank_id: NonEmptyStr
    normalized_question_hash: NonEmptyStr
    answer_claim_hash: NonEmptyStr
    requirement_contract_hash: NonEmptyStr
    member_count: PositiveInt
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class QBankCandidateFamilyMember(BaseSchema):
    family_member_id: NonEmptyStr
    family_id: NonEmptyStr
    run_id: NonEmptyStr
    qbank_id: NonEmptyStr
    ordinal: NonNegativeInt
    is_primary: bool
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)


class QuestionOpportunityEvidenceRef(BaseSchema):
    opportunity_evidence_ref_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    opportunity_id: NonEmptyStr
    alias: NonEmptyStr
    ordinal: NonNegativeInt
    evidence: OperatorEvidence
    evidence_group_id: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)


class QuestionOpportunity(BaseSchema):
    opportunity_id: NonEmptyStr
    opportunity_set_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    ordinal: NonNegativeInt
    opportunity_type: QuestionOpportunityType
    allowed_question_types: list[QuestionerV2QuestionType]
    allowed_reasoning_types: list[QuestionReasoningType]
    required_modalities: list[QuestionerV2Modality]
    minimum_evidence_groups: PositiveInt = 1
    evidence_aliases: list[NonEmptyStr]
    relation_link_ids: list[NonEmptyStr] = Field(default_factory=list)
    speaker_scope: list[NonEmptyStr] = Field(default_factory=list)
    evidence_group_ids: list[NonEmptyStr] = Field(default_factory=list)
    relation_content_hash: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_opportunity(self) -> QuestionOpportunity:
        for name in (
            "allowed_question_types",
            "allowed_reasoning_types",
            "required_modalities",
            "evidence_aliases",
            "relation_link_ids",
            "speaker_scope",
            "evidence_group_ids",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if not self.allowed_question_types or not self.allowed_reasoning_types:
            raise ValueError("opportunity requires allowed taxonomy values")
        if not self.required_modalities:
            raise ValueError("opportunity requires modalities")
        if self.opportunity_type is not QuestionOpportunityType.REFUSAL_BOUNDARY and (
            not self.evidence_aliases
        ):
            raise ValueError("answerable opportunity requires evidence aliases")
        if self.evidence_group_ids and (
            len(self.evidence_group_ids) < self.minimum_evidence_groups
        ):
            raise ValueError("opportunity evidence groups cannot understate its minimum")
        return self


class QuestionOpportunitySet(BaseSchema):
    opportunity_set_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    builder_version: NonEmptyStr
    opportunity_count: int = Field(ge=1, le=24)
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)


class QuestionerVerificationItemV1(BaseSchema):
    candidate_alias: NonEmptyStr
    verdict: QuestionerVerificationVerdict
    reason_codes: list[QuestionerVerificationReason]
    replacement: QuestionerCandidateDraftV3 | None

    @model_validator(mode="after")
    def _validate_verdict(self) -> QuestionerVerificationItemV1:
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("verifier reason_codes must not contain duplicates")
        if self.verdict is QuestionerVerificationVerdict.PASS:
            if self.reason_codes or self.replacement is not None:
                raise ValueError("pass verdict cannot contain reasons or replacement")
        elif not self.reason_codes:
            raise ValueError("non-pass verdict requires reason codes")
        if (self.verdict is QuestionerVerificationVerdict.REVISE) != (self.replacement is not None):
            raise ValueError("revise verdict requires exactly one replacement")
        return self


class QuestionerVerificationBatchV1(BaseSchema):
    results: list[QuestionerVerificationItemV1] = Field(max_length=8)


class QuestionerVerificationDimensionsV2(BaseSchema):
    """Auditable verifier dimensions; the server derives the final verdict."""

    user_facing: bool
    future_plausible: bool
    not_historical_replay: bool
    answer_explicitly_entailed: bool
    evidence_sufficient: bool
    taxonomy_correct: bool
    modality_correct: bool
    visual_necessary: bool
    multi_hop_genuine: bool
    metadata_free: bool


class QuestionerVerificationItemV2(BaseSchema):
    candidate_alias: NonEmptyStr
    requested_action: QuestionerVerificationVerdict
    dimensions: QuestionerVerificationDimensionsV2
    reason_codes: list[QuestionerVerificationReason]
    replacement: QuestionerCandidateDraftV3 | None

    @model_validator(mode="after")
    def _validate_action(self) -> QuestionerVerificationItemV2:
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("verifier reason_codes must not contain duplicates")
        if (self.requested_action is QuestionerVerificationVerdict.REVISE) != (
            self.replacement is not None
        ):
            raise ValueError("revise action requires exactly one replacement")
        if self.requested_action is QuestionerVerificationVerdict.PASS and (
            self.reason_codes or self.replacement is not None
        ):
            raise ValueError("pass action cannot contain reasons or replacement")
        if self.requested_action is not QuestionerVerificationVerdict.PASS and (
            not self.reason_codes
        ):
            raise ValueError("non-pass action requires reason codes")
        return self


class QuestionerVerificationBatchV2(BaseSchema):
    results: list[QuestionerVerificationItemV2] = Field(max_length=8)


class QuestionerVerificationItemV3(BaseSchema):
    """Provider result for one evidence-scoped candidate."""

    candidate_alias: NonEmptyStr
    dimensions: QuestionerVerificationDimensionsV2
    reason_codes: list[QuestionerVerificationReason]

    @model_validator(mode="after")
    def _validate_dimensions(self) -> QuestionerVerificationItemV3:
        ensure_unique_strings([item.value for item in self.reason_codes], "reason_codes")
        return self


class QuestionerVerificationEnvelopeV3(BaseSchema):
    result: QuestionerVerificationItemV3


class QuestionerDimensionVerdict(StrEnum):
    TRUE = "true"
    FALSE = "false"
    NOT_APPLICABLE = "not_applicable"


class QuestionerVerificationDimensionsV5(BaseSchema):
    """Dimensions-only v5 response; applicability and action are server-owned."""

    user_facing: QuestionerDimensionVerdict
    future_plausible: QuestionerDimensionVerdict
    not_historical_replay: QuestionerDimensionVerdict
    answer_addresses_question: QuestionerDimensionVerdict
    answer_explicitly_entailed: QuestionerDimensionVerdict
    speaker_subject_aligned: QuestionerDimensionVerdict
    evidence_sufficient: QuestionerDimensionVerdict
    taxonomy_compatible: QuestionerDimensionVerdict
    visual_necessary: QuestionerDimensionVerdict
    multi_hop_genuine: QuestionerDimensionVerdict
    metadata_free: QuestionerDimensionVerdict


class QuestionerVerificationDimensionsV6(QuestionerVerificationDimensionsV5):
    question_answer_compatible: QuestionerDimensionVerdict
    answer_type_correct: QuestionerDimensionVerdict
    source_naturalness: QuestionerDimensionVerdict


class QuestionerVerificationItemV6(BaseSchema):
    candidate_alias: NonEmptyStr
    dimensions: QuestionerVerificationDimensionsV6


class QuestionerVerificationEnvelopeV6(BaseSchema):
    result: QuestionerVerificationItemV6


class QuestionerClaimVerificationV7(BaseSchema):
    claim_index: NonNegativeInt
    support_alias: NonEmptyStr
    span_exists: bool
    claim_entailed: bool
    speaker_subject_aligned: bool


class QuestionerVerificationItemV7(BaseSchema):
    candidate_alias: NonEmptyStr
    dimensions: QuestionerVerificationDimensionsV6
    answer_question_closed: bool
    same_entity_attribute: QuestionerDimensionVerdict
    explicit_count_set: QuestionerDimensionVerdict
    claims: list[QuestionerClaimVerificationV7] = Field(max_length=8)


class QuestionerVerificationEnvelopeV7(BaseSchema):
    result: QuestionerVerificationItemV7


class QuestionerVerificationItemV5(BaseSchema):
    candidate_alias: NonEmptyStr
    dimensions: QuestionerVerificationDimensionsV5


class QuestionerVerificationEnvelopeV5(BaseSchema):
    result: QuestionerVerificationItemV5


class QuestionerRevisionDraftV1(BaseSchema):
    candidate_alias: NonEmptyStr
    generated_question: NonEmptyStr
    question_type: QuestionerV2QuestionType
    reasoning_type: QuestionReasoningType
    paraphrases: list[NonEmptyStr] = Field(max_length=3)

    @model_validator(mode="after")
    def _validate_paraphrases(self) -> QuestionerRevisionDraftV1:
        ensure_unique_strings(self.paraphrases, "paraphrases")
        return self


class QuestionerRevisionEnvelopeV1(BaseSchema):
    replacement: QuestionerRevisionDraftV1


class QuestionerRevisionDraftV2(BaseSchema):
    """Wording-only revision for v3.4; semantic fields remain server-owned."""

    candidate_alias: NonEmptyStr
    generated_question: NonEmptyStr
    paraphrases: list[NonEmptyStr] = Field(max_length=3)

    @model_validator(mode="after")
    def _validate_paraphrases(self) -> QuestionerRevisionDraftV2:
        ensure_unique_strings(self.paraphrases, "paraphrases")
        normalized = {_normalize_user_question(item) for item in self.paraphrases}
        if len(normalized) != len(self.paraphrases):
            raise ValueError("paraphrases must be unique after normalization")
        if _normalize_user_question(self.generated_question) in normalized:
            raise ValueError("canonical question cannot be repeated as a paraphrase")
        return self


class QuestionerRevisionEnvelopeV2(BaseSchema):
    replacement: QuestionerRevisionDraftV2


class QuestionerVerificationBatch(BaseSchema):
    verification_batch_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    generation_id: NonEmptyStr
    model_call_id: NonEmptyStr
    scope_qbank_id: NonEmptyStr | None = None
    verification_round: NonNegativeInt
    prompt_version: NonEmptyStr
    schema_hash: NonEmptyStr
    input_count: NonNegativeInt
    pass_count: NonNegativeInt
    revise_count: NonNegativeInt
    reject_count: NonNegativeInt
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_verification_counts(self) -> QuestionerVerificationBatch:
        if self.verification_round not in {0, 1}:
            raise ValueError("verification_round must be 0 or 1")
        if self.pass_count + self.revise_count + self.reject_count != self.input_count:
            raise ValueError("verification batch counts must reconcile")
        return self


class QuestionerCandidateVerification(BaseSchema):
    candidate_verification_id: NonEmptyStr
    verification_batch_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    generation_id: NonEmptyStr
    qbank_id: NonEmptyStr
    candidate_alias: NonEmptyStr
    verification_round: NonNegativeInt
    verdict: QuestionerVerificationVerdict
    reason_codes: list[QuestionerVerificationReason]
    replacement_qbank_id: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_candidate_verification(self) -> QuestionerCandidateVerification:
        if self.verification_round not in {0, 1}:
            raise ValueError("verification_round must be 0 or 1")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("verification reason_codes must not contain duplicates")
        if self.verdict is QuestionerVerificationVerdict.PASS:
            if self.reason_codes or self.replacement_qbank_id is not None:
                raise ValueError("pass verification cannot contain rejection payload")
        elif not self.reason_codes:
            raise ValueError("non-pass verification requires reason codes")
        if (self.verdict is QuestionerVerificationVerdict.REVISE) != (
            self.replacement_qbank_id is not None
        ):
            raise ValueError("revise verification requires a replacement qbank ID")
        return self


class QuestionerCandidateSupersession(BaseSchema):
    supersession_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    generation_id: NonEmptyStr
    source_qbank_id: NonEmptyStr
    replacement_qbank_id: NonEmptyStr
    revision_round: PositiveInt = 1
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_supersession(self) -> QuestionerCandidateSupersession:
        if self.revision_round != 1:
            raise ValueError("Questioner v3 allows exactly one semantic revision")
        if self.source_qbank_id == self.replacement_qbank_id:
            raise ValueError("supersession source and replacement must differ")
        return self


class CompilationUnitLink(BaseSchema):
    link_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    anchor_memory_id: NonEmptyStr
    support_memory_id: NonEmptyStr
    relation_type: CompilationUnitRelationType
    ordinal: NonNegativeInt
    relation_key_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_link(self) -> CompilationUnitLink:
        if self.anchor_memory_id == self.support_memory_id:
            raise ValueError("compiler unit links require distinct anchor and support memories")
        return self


class QuestionerGeneration(BaseSchema):
    generation_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    job_id: NonEmptyStr
    model_call_id: NonEmptyStr
    attempt_number: PositiveInt
    prompt_version: NonEmptyStr
    renderer_version: NonEmptyStr
    parser_version: NonEmptyStr
    prompt_hash: NonEmptyStr
    input_hash: NonEmptyStr
    output_hash: NonEmptyStr | None = None
    parse_status: QuestionerParseStatus
    raw_candidate_count: NonNegativeInt = 0
    valid_candidate_count: NonNegativeInt = 0
    invalid_candidate_count: NonNegativeInt = 0
    repair_attempt_count: NonNegativeInt = 0
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts(self) -> QuestionerGeneration:
        if self.valid_candidate_count + self.invalid_candidate_count != self.raw_candidate_count:
            raise ValueError("questioner generation candidate counts must reconcile")
        if self.parse_status is QuestionerParseStatus.FAILED and self.valid_candidate_count:
            raise ValueError("failed generations cannot contain valid candidates")
        return self


class QuestionCandidateGateResult(BaseSchema):
    gate_result_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    generation_id: NonEmptyStr
    qbank_id: NonEmptyStr
    gate_version: NonEmptyStr
    status: CandidateGateStatus
    rejection_codes: list[CandidateRejectionCode] = Field(default_factory=list)
    duplicate_of_qbank_id: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_gate(self) -> QuestionCandidateGateResult:
        if len(self.rejection_codes) != len(set(self.rejection_codes)):
            raise ValueError("rejection_codes must not contain duplicates")
        rejected = self.status is CandidateGateStatus.REJECTED
        if rejected != bool(self.rejection_codes):
            raise ValueError("rejected gate results require rejection codes")
        if self.duplicate_of_qbank_id is not None and CandidateRejectionCode.DUPLICATE not in (
            self.rejection_codes
        ):
            raise ValueError("duplicate_of_qbank_id requires the duplicate rejection code")
        return self


class QBankDuplicateLink(BaseSchema):
    duplicate_link_id: NonEmptyStr
    run_id: NonEmptyStr
    duplicate_qbank_id: NonEmptyStr
    primary_qbank_id: NonEmptyStr
    method: NonEmptyStr
    score: RankingScore | None = None
    hard_duplicate: bool = False
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ids(self) -> QBankDuplicateLink:
        if self.duplicate_qbank_id == self.primary_qbank_id:
            raise ValueError("duplicate and primary QBank IDs must differ")
        return self


class QBankIndexManifest(BaseSchema):
    qbank_index_id: NonEmptyStr
    compiler_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    gate_version: NonEmptyStr
    matcher_version: NonEmptyStr
    text_question_collection_id: NonEmptyStr
    text_paraphrase_collection_id: NonEmptyStr
    multimodal_question_collection_id: NonEmptyStr
    multimodal_paraphrase_collection_id: NonEmptyStr
    eligible_question_count: NonNegativeInt
    paraphrase_count: NonNegativeInt
    rrf_constant: PositiveInt = 60
    per_channel_top_k: PositiveInt = 50
    status: EmbeddingCollectionStatus = EmbeddingCollectionStatus.BUILDING
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts(self) -> QBankIndexManifest:
        if self.paraphrase_count and self.eligible_question_count == 0:
            raise ValueError("QBank paraphrases require eligible questions")
        return self


class QBankCoverageRunManifest(BaseSchema):
    coverage_run_id: NonEmptyStr
    compiler_run_id: NonEmptyStr
    qbank_index_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: NonEmptyStr
    protocol_version: NonEmptyStr
    selection_hash: NonEmptyStr
    config_hash: NonEmptyStr
    question_budgets: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 4, 8])
    match_ks: list[PositiveInt] = Field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    gate_mode: CoverageGateMode = CoverageGateMode.REPORT_ONLY
    status: QBankCoverageStatus = QBankCoverageStatus.CREATED
    created_at: UtcDateTime = Field(default_factory=utc_now)
    completed_at: UtcDateTime | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_grid(self) -> QBankCoverageRunManifest:
        if self.question_budgets != sorted(set(self.question_budgets)):
            raise ValueError("question_budgets must be sorted and unique")
        if self.match_ks != sorted(set(self.match_ks)):
            raise ValueError("match_ks must be sorted and unique")
        if self.status is QBankCoverageStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed coverage runs require completed_at")
        if self.status is not QBankCoverageStatus.COMPLETED and self.completed_at is not None:
            raise ValueError("only completed coverage runs may set completed_at")
        return self


class QBankCoverageQuery(BaseSchema):
    coverage_run_id: NonEmptyStr
    query_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    cutoff_session_id: NonEmptyStr
    ordinal: NonNegativeInt
    has_query_image: bool = False
    applicable: bool = True
    gold_group_count: NonNegativeInt = 0
    unavailable_reason: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_applicability(self) -> QBankCoverageQuery:
        if self.applicable and (self.gold_group_count == 0 or self.unavailable_reason is not None):
            raise ValueError("applicable coverage queries require gold groups and no reason")
        if not self.applicable and self.unavailable_reason is None:
            raise ValueError("non-applicable coverage queries require a reason")
        return self


class QBankCoverageQueryEmbedding(BaseSchema):
    query_embedding_id: NonEmptyStr
    coverage_run_id: NonEmptyStr
    query_id: NonEmptyStr
    channel: QBankMatchChannel
    model_alias: NonEmptyStr
    model_revision: NonEmptyStr
    input_hash: NonEmptyStr
    dimension: PositiveInt
    vector: list[FiniteFloat]
    model_call_id: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_vector(self) -> QBankCoverageQueryEmbedding:
        if len(self.vector) != self.dimension:
            raise ValueError("query embedding dimension does not match vector length")
        if self.channel not in {
            QBankMatchChannel.TEXT_SEMANTIC,
            QBankMatchChannel.MULTIMODAL_SEMANTIC,
        }:
            raise ValueError("query embeddings require a semantic channel")
        return self


class QBankCoverageCandidate(BaseSchema):
    coverage_candidate_id: NonEmptyStr
    coverage_run_id: NonEmptyStr
    query_id: NonEmptyStr
    qbank_id: NonEmptyStr
    question_budget: PositiveInt
    rank: PositiveInt
    lexical_rank: PositiveInt | None = None
    text_semantic_rank: PositiveInt | None = None
    multimodal_semantic_rank: PositiveInt | None = None
    lexical_score: RankingScore | None = None
    text_semantic_score: RankingScore | None = None
    multimodal_semantic_score: RankingScore | None = None
    final_score: RankingScore
    matched_gold_group_ids: list[NonEmptyStr] = Field(default_factory=list)
    individually_sufficient: bool = False
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_matches(self) -> QBankCoverageCandidate:
        ensure_unique_strings(self.matched_gold_group_ids, "matched_gold_group_ids")
        return self


class QBankCoverageMetric(BaseSchema):
    metric_id: NonEmptyStr
    coverage_run_id: NonEmptyStr
    query_id: NonEmptyStr | None = None
    metric_name: NonEmptyStr
    metric_version: NonEmptyStr
    question_budget: PositiveInt | None = None
    k: PositiveInt | None = None
    applicable: bool = True
    value: Probability | None = None
    unavailable_reason: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_metric(self) -> QBankCoverageMetric:
        if self.applicable != (self.value is not None):
            raise ValueError("applicable coverage metrics require a value")
        if self.applicable and self.unavailable_reason is not None:
            raise ValueError("applicable coverage metrics cannot have a reason")
        if not self.applicable and self.unavailable_reason is None:
            raise ValueError("non-applicable coverage metrics require a reason")
        return self


class QBankCoverageLabel(BaseSchema):
    label_id: NonEmptyStr
    coverage_run_id: NonEmptyStr
    query_id: NonEmptyStr
    category: QBankCoverageLabelCategory
    question_budget: PositiveInt | None = None
    k: PositiveInt | None = None
    is_primary: bool = True
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class QBankCoverageGateReport(BaseSchema):
    gate_report_id: NonEmptyStr
    coverage_run_id: NonEmptyStr
    gate_version: NonEmptyStr
    mode: CoverageGateMode
    hard_integrity_passed: bool
    scientific_thresholds_passed: bool | None = None
    query_count: NonNegativeInt
    applicable_query_count: NonNegativeInt
    metric_summary: dict[NonEmptyStr, FiniteFloat] = Field(default_factory=dict)
    threshold_summary: dict[NonEmptyStr, FiniteFloat] = Field(default_factory=dict)
    failure_counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    elapsed_ms: NonNegativeFloat = 0.0
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_report(self) -> QBankCoverageGateReport:
        if self.applicable_query_count > self.query_count:
            raise ValueError("applicable query count cannot exceed query count")
        if self.mode is CoverageGateMode.REPORT_ONLY:
            if self.scientific_thresholds_passed is not None:
                raise ValueError("report-only gate cannot claim a scientific threshold result")
        elif self.scientific_thresholds_passed is None:
            raise ValueError("enforced gate requires a scientific threshold result")
        return self
