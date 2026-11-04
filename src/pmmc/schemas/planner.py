"""Strict contracts for Planner v1 and evidence-driven program dry-runs."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from pmmc.schemas.base import (
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
from pmmc.schemas.program import OperatorControlAction

_UNSAFE_PROVIDER_TEXT = re.compile(
    r"(?:```|https?://|file://|\.\./|(?:^|\s)/(?:etc|proc|sys|var|tmp)/|"
    r"\b(?:select\s+.+\s+from|insert\s+into|delete\s+from|drop\s+table|"
    r"update\s+.+\s+set|import\s+[A-Za-z_]|exec\s*\(|eval\s*\(|sudo\s+|"
    r"bash\s+-c|sh\s+-c)\b|(?:\$\(|&&|;\s*(?:rm|curl|wget)\b)|"
    r"\b(?:memlens|memgallery):(?:memory|session|query|image|caption):"
    r"[A-Za-z0-9._:-]+\b|\b(?:compiler-unit|qbank|collection|rag-index|"
    r"memory-index|snapshot|run|memory|session|image)-[0-9a-f]{8,}\b|"
    r"\b[MOISCE]\d{2}\b)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _contains_unsafe_provider_text(value: object) -> bool:
    if isinstance(value, str):
        return _UNSAFE_PROVIDER_TEXT.search(value) is not None
    if isinstance(value, list):
        return any(_contains_unsafe_provider_text(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_unsafe_provider_text(item) for item in value.values())
    return False


class PlannerOutcomeStatus(StrEnum):
    PROGRAM_READY = "program_ready"
    PROGRAM_READY_WITH_FALLBACK = "program_ready_with_fallback"
    PROGRAM_READY_WITH_ABSTAIN = "program_ready_with_abstain"
    PARSE_FAILED = "parse_failed"
    STATIC_REJECTED = "static_rejected"
    POLICY_REJECTED = "policy_rejected"
    LINEAGE_REJECTED = "lineage_rejected"
    QUERY_VECTOR_UNAVAILABLE = "query_vector_unavailable"
    EXECUTION_FAILED = "execution_failed"
    DRY_RUN_INSUFFICIENT = "dry_run_insufficient"


class ProgramPolicySeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class EvidenceMatchKind(StrEnum):
    DIRECT = "direct"
    DERIVED = "derived"
    GROUP = "group"
    CONTEXTUAL = "contextual"


class SufficiencyStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class FallbackCondition(StrEnum):
    ALWAYS = "always"
    IF_INSUFFICIENT = "if_insufficient"


class PlannerRequiredModality(StrEnum):
    TEXT = "text"
    CAPTION = "caption"
    IMAGE = "image"
    RAW_IMAGE = "raw_image"


class PlannerCapability(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    RAW_IMAGE = "raw_image"
    IMAGE_TEXT = "image_text"
    CONTEXT = "context"
    TEMPORAL = "temporal"
    EVIDENCE_GROUPS = "evidence_groups"
    VERIFICATION = "verification"
    CONTROL = "control"


class PlannerTermSource(StrEnum):
    CANONICAL_QUESTION = "canonical_question"


class PlannerRetrievalChannel(StrEnum):
    KEYWORD = "keyword"
    TEXT_DENSE = "text_dense"
    VISUAL_DENSE = "visual_dense"


class PlannerExpansionCapability(StrEnum):
    IMAGE_TEXT = "image_text"
    CONTEXT = "context"
    TEMPORAL = "temporal"
    RAW_IMAGE = "raw_image"


class PlannerTopKTier(StrEnum):
    NARROW = "narrow"
    STANDARD = "standard"
    BROAD = "broad"

    @property
    def top_k(self) -> int:
        return {
            PlannerTopKTier.NARROW: 5,
            PlannerTopKTier.STANDARD: 20,
            PlannerTopKTier.BROAD: 50,
        }[self]


class SemanticSupportStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class PlannerEvidenceBudget(BaseSchema):
    """Preflight upper bounds, counted as typed evidence references after deduplication."""

    policy_version: Literal["planner-evidence-budget-v1"] = "planner-evidence-budget-v1"
    total_evidence_max: PositiveInt
    total_upper_bound: NonNegativeInt
    step_quotas: dict[NonNegativeInt, NonNegativeInt]
    root_top_k: dict[NonNegativeInt, PositiveInt]
    expansion_fanout: dict[NonNegativeInt, PositiveInt]

    @model_validator(mode="after")
    def _bounded(self) -> PlannerEvidenceBudget:
        if self.total_upper_bound > self.total_evidence_max:
            raise ValueError("program exceeds its preflight evidence budget")
        if sum(self.step_quotas.values()) != self.total_upper_bound:
            raise ValueError("step evidence quotas must reconcile")
        return self


class CompilerStageKind(StrEnum):
    PLANNER = "planner"
    DRY_RUN = "dry_run"


class CompilerStageJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgramValidationKind(StrEnum):
    STATIC = "static"
    POLICY = "policy"
    LINEAGE = "lineage"


class CompilerStageJob(BaseSchema):
    stage_job_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    stage: CompilerStageKind
    round_id: NonNegativeInt = 0
    status: CompilerStageJobStatus = CompilerStageJobStatus.PENDING
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
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> CompilerStageJob:
        if self.status is CompilerStageJobStatus.RUNNING and (
            self.claimed_by is None or self.lease_expires_at is None
        ):
            raise ValueError("running compiler stage jobs require a lease owner and expiry")
        if self.status is not CompilerStageJobStatus.RUNNING and (
            self.claimed_by is not None or self.lease_expires_at is not None
        ):
            raise ValueError("only running compiler stage jobs may hold a lease")
        if (
            self.status
            in {
                CompilerStageJobStatus.SUCCEEDED,
                CompilerStageJobStatus.FAILED,
                CompilerStageJobStatus.CANCELLED,
            }
            and self.finished_at is None
        ):
            raise ValueError("terminal compiler stage jobs require finished_at")
        if self.status is CompilerStageJobStatus.FAILED and self.error_type is None:
            raise ValueError("failed compiler stage jobs require error_type")
        return self


class PlannerKeywordStepDraft(BaseSchema):
    op_name: Literal["retrieve_text_by_keyword"]
    top_k: PositiveInt
    terms: list[NonEmptyStr] | None = Field(default=None, min_length=1, max_length=8)
    term_source: PlannerTermSource | None = None
    term_budget: PositiveInt | None = Field(default=None, le=8)

    @model_validator(mode="after")
    def _validate_keyword_source(self) -> PlannerKeywordStepDraft:
        legacy = self.terms is not None
        server_derived = self.term_source is not None or self.term_budget is not None
        if legacy == server_derived:
            raise ValueError("keyword retrieval requires legacy terms or server-derived terms")
        if server_derived and (
            self.term_source is not PlannerTermSource.CANONICAL_QUESTION or self.term_budget is None
        ):
            raise ValueError("server-derived keywords require canonical_question and a budget")
        return self


class PlannerKeywordStepDraftV11(BaseSchema):
    op_name: Literal["retrieve_text_by_keyword"]
    top_k: PositiveInt
    term_source: Literal[PlannerTermSource.CANONICAL_QUESTION]
    term_budget: PositiveInt = Field(le=8)


class PlannerEmbeddingStepDraft(BaseSchema):
    op_name: Literal["retrieve_text_by_embedding", "retrieve_image_by_visual_embedding"]
    top_k: PositiveInt
    vector_role: Literal["question_text_query", "question_joint_query"]


class PlannerImageStepDraft(BaseSchema):
    op_name: Literal[
        "retrieve_image_text_pair", "retrieve_context_around_image", "inspect_raw_image"
    ]
    source_step_ids: list[NonNegativeInt]
    radius: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _validate_radius(self) -> PlannerImageStepDraft:
        if self.op_name == "retrieve_context_around_image":
            if self.radius is None or self.radius > 2:
                raise ValueError("context radius must be supplied and no greater than 2")
        elif self.radius is not None:
            raise ValueError("radius is only valid for retrieve_context_around_image")
        return self


class PlannerTemporalStepDraft(BaseSchema):
    op_name: Literal["expand_temporal_neighbors"]
    source_step_ids: list[NonNegativeInt]
    radius: PositiveInt = Field(le=2)


class PlannerVerifyStepDraft(BaseSchema):
    op_name: Literal["verify_evidence_sufficiency"]
    source_step_ids: list[NonNegativeInt]
    required_modalities: list[PlannerRequiredModality]
    minimum_evidence_groups: NonNegativeInt = 1


class PlannerFallbackStepDraft(BaseSchema):
    op_name: Literal["fallback_to_multimodal_rag"]
    source_step_ids: list[NonNegativeInt]
    condition: FallbackCondition
    reason_code: NonEmptyStr


class PlannerAbstainStepDraft(BaseSchema):
    op_name: Literal["abstain_if_no_evidence"]
    source_step_ids: list[NonNegativeInt] = Field(default_factory=list)
    minimum_evidence_groups: NonNegativeInt = 1
    required_modalities: list[PlannerRequiredModality] = Field(default_factory=list)
    reason_code: NonEmptyStr


PlannerStepDraft = Annotated[
    PlannerKeywordStepDraft
    | PlannerEmbeddingStepDraft
    | PlannerImageStepDraft
    | PlannerTemporalStepDraft
    | PlannerVerifyStepDraft
    | PlannerFallbackStepDraft
    | PlannerAbstainStepDraft,
    Field(discriminator="op_name"),
]

PlannerStepDraftV11 = Annotated[
    PlannerKeywordStepDraftV11
    | PlannerEmbeddingStepDraft
    | PlannerImageStepDraft
    | PlannerTemporalStepDraft
    | PlannerVerifyStepDraft
    | PlannerFallbackStepDraft
    | PlannerAbstainStepDraft,
    Field(discriminator="op_name"),
]


class PlannerProgramDraft(BaseSchema):
    candidate_alias: NonEmptyStr
    steps: list[PlannerStepDraft] = Field(min_length=1, max_length=8)
    strategy_code: NonEmptyStr
    rationale_summary: NonEmptyStr
    confidence: Probability

    @model_validator(mode="after")
    def _reject_unsafe_provider_literals(self) -> PlannerProgramDraft:
        if _contains_unsafe_provider_text(self.model_dump(mode="json")):
            raise ValueError("Planner output contains code, a path/URL, or a canonical identifier")
        return self


class PlannerGenerationEnvelope(BaseSchema):
    programs: list[PlannerProgramDraft] = Field(default_factory=list, max_length=4)


class PlannerProgramDraftV11(BaseSchema):
    candidate_alias: NonEmptyStr
    steps: list[PlannerStepDraftV11] = Field(min_length=1, max_length=8)
    strategy_code: NonEmptyStr
    rationale_summary: NonEmptyStr
    confidence: Probability

    @model_validator(mode="after")
    def _reject_unsafe_provider_literals(self) -> PlannerProgramDraftV11:
        if _contains_unsafe_provider_text(self.model_dump(mode="json")):
            raise ValueError("Planner output contains code, a path/URL, or a canonical identifier")
        return self


class PlannerGenerationEnvelopeV11(BaseSchema):
    programs: list[PlannerProgramDraftV11] = Field(default_factory=list, max_length=4)


class PlannerStrategyDraftV12(BaseSchema):
    """Non-oracle single-candidate strategy selected by Planner v1.2."""

    candidate_alias: NonEmptyStr
    retrieval_channels: list[PlannerRetrievalChannel] = Field(
        min_length=1,
        max_length=3,
        json_schema_extra={"uniqueItems": True},
    )
    expansion_capabilities: list[PlannerExpansionCapability] = Field(
        default_factory=list,
        max_length=4,
        json_schema_extra={"uniqueItems": True},
    )
    top_k_tier: PlannerTopKTier
    strategy_code: NonEmptyStr = Field(
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    confidence: Probability

    @field_validator("retrieval_channels")
    @classmethod
    def _validate_retrieval_channels(
        cls,
        values: list[PlannerRetrievalChannel],
    ) -> list[PlannerRetrievalChannel]:
        ensure_unique_strings(
            [item.value for item in values],
            "retrieval_channels",
        )
        return values

    @field_validator("expansion_capabilities")
    @classmethod
    def _validate_expansion_capabilities(
        cls,
        values: list[PlannerExpansionCapability],
    ) -> list[PlannerExpansionCapability]:
        ensure_unique_strings(
            [item.value for item in values],
            "expansion_capabilities",
        )
        return values

    @field_validator("candidate_alias", "strategy_code")
    @classmethod
    def _reject_unsafe_strategy_text(cls, value: str) -> str:
        if _contains_unsafe_provider_text(value):
            raise ValueError("Planner strategy contains an unsafe provider literal")
        return value


class PlannerStrategyEnvelopeV12(BaseSchema):
    strategy: PlannerStrategyDraftV12


class PlannerRequirementContract(BaseSchema):
    requirement_contract_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    qbank_id: NonEmptyStr
    required_modalities: list[PlannerRequiredModality]
    minimum_evidence_groups: PositiveInt
    required_capabilities: list[PlannerCapability]
    allowed_capabilities: list[PlannerCapability] = Field(default_factory=list)
    fallback_required: bool
    refusal_only: bool = False
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> PlannerRequirementContract:
        if not self.required_modalities:
            raise ValueError("Planner requirement contract needs modalities")
        ensure_unique_strings(
            [item.value for item in self.required_modalities], "required_modalities"
        )
        ensure_unique_strings(
            [item.value for item in self.required_capabilities], "required_capabilities"
        )
        ensure_unique_strings(
            [item.value for item in self.allowed_capabilities], "allowed_capabilities"
        )
        if self.allowed_capabilities and not (
            set(self.required_capabilities) <= set(self.allowed_capabilities)
        ):
            raise ValueError("required capabilities must be allowed")
        if self.refusal_only and self.fallback_required:
            raise ValueError("refusal contracts cannot require fallback")
        return self


class PlannerCandidateDraftFact(BaseSchema):
    planner_candidate_draft_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    planner_batch_id: NonEmptyStr
    qbank_id: NonEmptyStr
    requirement_contract_id: NonEmptyStr
    item_ordinal: NonNegativeInt
    item_hash: NonEmptyStr
    parsed: bool
    program_hash: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_parsed(self) -> PlannerCandidateDraftFact:
        if self.parsed != (self.program_hash is not None):
            raise ValueError("parsed Planner drafts require program_hash")
        return self


class PlannerGenerationBatch(BaseSchema):
    planner_batch_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    stage_job_id: NonEmptyStr
    model_call_id: NonEmptyStr
    batch_ordinal: NonNegativeInt
    candidate_qbank_ids: list[NonEmptyStr]
    parsed_count: NonNegativeInt
    invalid_count: NonNegativeInt
    prompt_hash: NonEmptyStr
    schema_hash: NonEmptyStr
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts(self) -> PlannerGenerationBatch:
        ensure_unique_strings(self.candidate_qbank_ids, "candidate_qbank_ids")
        if self.parsed_count + self.invalid_count != len(self.candidate_qbank_ids):
            raise ValueError("planner batch counts must close over candidate_qbank_ids")
        return self


class PlannerCandidateOutcome(BaseSchema):
    planner_outcome_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    qbank_id: NonEmptyStr
    planner_batch_id: NonEmptyStr | None = None
    plan_id: NonEmptyStr | None = None
    status: PlannerOutcomeStatus
    reason_code: NonEmptyStr
    reason_summary: NonEmptyStr | None = None
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_plan(self) -> PlannerCandidateOutcome:
        has_program = self.status in {
            PlannerOutcomeStatus.PROGRAM_READY,
            PlannerOutcomeStatus.PROGRAM_READY_WITH_FALLBACK,
            PlannerOutcomeStatus.PROGRAM_READY_WITH_ABSTAIN,
            PlannerOutcomeStatus.EXECUTION_FAILED,
            PlannerOutcomeStatus.DRY_RUN_INSUFFICIENT,
            PlannerOutcomeStatus.LINEAGE_REJECTED,
            PlannerOutcomeStatus.QUERY_VECTOR_UNAVAILABLE,
        }
        if has_program != (self.plan_id is not None):
            raise ValueError("planner outcome plan_id does not match its status")
        return self


class PlannerBatchProgram(BaseSchema):
    planner_batch_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    qbank_id: NonEmptyStr
    plan_id: NonEmptyStr
    ordinal: NonNegativeInt
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)


class ProgramPolicyIssue(BaseSchema):
    code: NonEmptyStr
    message: NonEmptyStr
    severity: ProgramPolicySeverity = ProgramPolicySeverity.ERROR
    step_id: NonNegativeInt | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ProgramPolicyResult(BaseSchema):
    policy_result_id: NonEmptyStr
    run_id: NonEmptyStr
    plan_id: NonEmptyStr
    policy_version: NonEmptyStr
    valid: bool
    issues: list[ProgramPolicyIssue] = Field(default_factory=list)
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate_result(self) -> ProgramPolicyResult:
        has_error = any(item.severity is ProgramPolicySeverity.ERROR for item in self.issues)
        if self.valid == has_error:
            raise ValueError("valid must be the inverse of policy errors")
        return self


class ProgramValidationIssueFact(BaseSchema):
    validation_issue_id: NonEmptyStr
    validation_result_id: NonEmptyStr
    run_id: NonEmptyStr
    ordinal: NonNegativeInt
    code: NonEmptyStr
    severity: ProgramPolicySeverity
    step_id: NonNegativeInt | None = None
    message: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)


class ProgramValidationResultFact(BaseSchema):
    validation_result_id: NonEmptyStr
    run_id: NonEmptyStr
    plan_id: NonEmptyStr
    validation_kind: ProgramValidationKind
    validator_version: NonEmptyStr
    valid: bool
    issues: list[ProgramValidationIssueFact] = Field(default_factory=list)
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_issues(self) -> ProgramValidationResultFact:
        if any(item.validation_result_id != self.validation_result_id for item in self.issues):
            raise ValueError("validation issues must belong to their result")
        if [item.ordinal for item in self.issues] != list(range(len(self.issues))):
            raise ValueError("validation issue ordinals must be contiguous")
        has_error = any(item.severity is ProgramPolicySeverity.ERROR for item in self.issues)
        if self.valid == has_error:
            raise ValueError("valid must be the inverse of validation errors")
        return self


class ProgramEvidenceMatch(BaseSchema):
    match_id: NonEmptyStr
    assessment_id: NonEmptyStr
    run_id: NonEmptyStr
    expected_evidence_id: NonEmptyStr
    actual_evidence_id: NonEmptyStr
    expected_evidence_type: EvidenceType
    actual_evidence_type: EvidenceType
    match_kind: EvidenceMatchKind
    evidence_group_id: NonEmptyStr | None = None
    direct_match: bool = False
    derived_match: bool = False
    contextual_only: bool = False
    discriminative_match: bool = False
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_match_flags(self) -> ProgramEvidenceMatch:
        if not any(
            (
                self.direct_match,
                self.derived_match,
                self.contextual_only,
                self.discriminative_match,
            )
        ):
            inferred = {
                EvidenceMatchKind.DIRECT: "direct_match",
                EvidenceMatchKind.DERIVED: "derived_match",
                EvidenceMatchKind.CONTEXTUAL: "contextual_only",
                EvidenceMatchKind.GROUP: "discriminative_match",
            }[self.match_kind]
            object.__setattr__(self, inferred, True)
        if self.direct_match and self.derived_match:
            raise ValueError("evidence match cannot be both direct and derived")
        if self.contextual_only and self.discriminative_match:
            raise ValueError("contextual evidence cannot be discriminative")
        if self.match_kind is EvidenceMatchKind.DIRECT and not self.direct_match:
            raise ValueError("direct match kind requires direct_match")
        if self.match_kind is EvidenceMatchKind.DERIVED and not self.derived_match:
            raise ValueError("derived match kind requires derived_match")
        if self.match_kind is EvidenceMatchKind.CONTEXTUAL and not self.contextual_only:
            raise ValueError("contextual match kind requires contextual_only")
        return self


class ProgramDryRunAssessment(BaseSchema):
    assessment_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    qbank_id: NonEmptyStr
    plan_id: NonEmptyStr
    execution_id: NonEmptyStr
    status: TraceStatus
    control_action: OperatorControlAction
    execution_success: bool
    nonempty_evidence: bool
    evidence_sufficient: bool
    expected_group_count: NonNegativeInt
    matched_group_count: NonNegativeInt
    expected_evidence_recall: Probability | None = None
    expected_evidence_complete: bool
    modality_coverage: dict[NonEmptyStr, bool] = Field(default_factory=dict)
    raw_image_reached: bool
    context_operator_used: bool
    temporal_operator_used: bool
    context_evidence_contributed: bool = False
    temporal_evidence_contributed: bool = False
    successful_step_count: NonNegativeInt = 0
    required_modality_sufficient: bool = False
    discriminative_group_count: NonNegativeInt = 0
    matched_discriminative_group_count: NonNegativeInt = 0
    contextual_match_count: NonNegativeInt = 0
    pre_control_sufficient: bool = False
    fallback_triggered: bool = False
    abstain_triggered: bool = False
    semantic_answer_support: SemanticSupportStatus = SemanticSupportStatus.NOT_EVALUATED
    missing_evidence_group_ids: list[NonEmptyStr] = Field(default_factory=list)
    failure_code: NonEmptyStr | None = None
    elapsed_ms: NonNegativeFloat
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts(self) -> ProgramDryRunAssessment:
        if self.matched_group_count > self.expected_group_count:
            raise ValueError("matched_group_count cannot exceed expected_group_count")
        if self.matched_discriminative_group_count > self.discriminative_group_count:
            raise ValueError("matched discriminative groups exceed their denominator")
        if self.fallback_triggered and self.abstain_triggered:
            raise ValueError("dry-run cannot trigger fallback and abstain together")
        if self.context_evidence_contributed and not self.context_operator_used:
            raise ValueError("context evidence contribution requires the context operator")
        if self.temporal_evidence_contributed and not self.temporal_operator_used:
            raise ValueError("temporal evidence contribution requires the temporal operator")
        if self.expected_group_count == 0 and self.expected_evidence_recall is not None:
            raise ValueError("zero expected groups require a not-applicable recall")
        if (
            self.expected_evidence_complete
            and self.matched_group_count != self.expected_group_count
        ):
            raise ValueError("complete evidence requires every expected group")
        if (
            self.metadata.get("assessment_policy") == "program-dry-run-assessment-v3"
            and self.expected_group_count > 0
            and (self.expected_evidence_recall or 0.0) == 0.0
            and self.evidence_sufficient
        ):
            raise ValueError(
                "Step 7B expected evidence recall is zero but path is marked sufficient"
            )
        ensure_unique_strings(self.missing_evidence_group_ids, "missing_evidence_group_ids")
        return self


class MemoryOperatorIndexManifest(BaseSchema):
    memory_index_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: NonEmptyStr
    selection_hash: NonEmptyStr
    representation_version: NonEmptyStr
    text_collection_id: NonEmptyStr
    visual_collection_id: NonEmptyStr
    text_model_alias: NonEmptyStr
    text_model_revision: NonEmptyStr
    multimodal_model_alias: NonEmptyStr
    multimodal_model_revision: NonEmptyStr
    dimension: PositiveInt = 4096
    distance_metric: Literal["cosine"] = "cosine"
    status: Literal["building", "ready", "failed"] = "building"
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_collections(self) -> MemoryOperatorIndexManifest:
        if self.text_collection_id == self.visual_collection_id:
            raise ValueError("text and visual memory collections must be distinct")
        return self


class ProgramQueryEmbeddingBatch(BaseSchema):
    query_embedding_batch_id: NonEmptyStr
    run_id: NonEmptyStr
    channel: Literal["question_text_query", "question_joint_query"]
    model_call_id: NonEmptyStr
    batch_ordinal: NonNegativeInt
    qbank_ids: list[NonEmptyStr] = Field(min_length=1)
    dimension: PositiveInt = 4096
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_qbank_ids(self) -> ProgramQueryEmbeddingBatch:
        ensure_unique_strings(self.qbank_ids, "qbank_ids")
        if self.dimension != 4096:
            raise ValueError("program query embeddings must be 4096-dimensional")
        return self


class ProgramQueryEmbedding(BaseSchema):
    program_query_embedding_id: NonEmptyStr
    query_embedding_batch_id: NonEmptyStr
    run_id: NonEmptyStr
    qbank_id: NonEmptyStr
    channel: Literal["question_text_query", "question_joint_query"]
    embedding_id: NonEmptyStr
    input_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)
