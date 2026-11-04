"""Typed contracts for Step 6D feedback, revision, and QBank freezing."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    DOUBTER_SCHEMA_VERSION,
    STAGE_COMPONENT_SCHEMA_VERSION,
    VERIFIED_QBANK_SCHEMA_VERSION,
    BaseSchema,
    JsonDict,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.planner import PlannerCapability


class QBankRouteClass(StrEnum):
    """Runtime routing class; scientific question budgets use answerable only."""

    ANSWERABLE = "answerable"
    CONTROL = "control"


class DoubterAction(StrEnum):
    PASS = "pass"
    REVISE_PROGRAM = "revise_program"
    REJECT_CANDIDATE = "reject_candidate"


class DoubterIssueCode(StrEnum):
    CANDIDATE_UNSUPPORTED = "candidate_unsupported"
    ANSWER_NOT_ENTAILED = "answer_not_entailed"
    PATH_INSUFFICIENT = "path_insufficient"
    MODALITY_MISSING = "modality_missing"
    RAW_IMAGE_MISSING = "raw_image_missing"
    CONTEXT_MISSING = "context_missing"
    MULTI_HOP_INCOMPLETE = "multi_hop_incomplete"
    TEMPORAL_INCONSISTENT = "temporal_inconsistent"
    REDUNDANT_RETRIEVAL = "redundant_retrieval"
    REFUSAL_INAPPROPRIATE = "refusal_inappropriate"
    STRUCTURED_OUTPUT_INVALID = "structured_output_invalid"
    MAX_ROUNDS_EXHAUSTED = "max_rounds_exhausted"


class DoubterDirectiveCode(StrEnum):
    ADD_TEXT_RETRIEVAL = "add_text_retrieval"
    ADD_VISUAL_RETRIEVAL = "add_visual_retrieval"
    ADD_IMAGE_TEXT_PAIR = "add_image_text_pair"
    ADD_CONTEXT = "add_context"
    ADD_TEMPORAL = "add_temporal"
    ADD_RAW_IMAGE_INSPECTION = "add_raw_image_inspection"
    INCREASE_TOP_K = "increase_top_k"
    REMOVE_REDUNDANT_RETRIEVAL = "remove_redundant_retrieval"


class PlannerDirectiveStatus(StrEnum):
    APPLIED = "applied"
    ALREADY_SATISFIED = "already_satisfied"
    INAPPLICABLE = "inapplicable"
    SATURATED = "saturated"
    NOOP = "noop"


class PlannerDirectiveEffect(BaseSchema):
    directive_id: NonEmptyStr
    directive: DoubterDirectiveCode
    status: PlannerDirectiveStatus
    reason_code: NonEmptyStr
    source_step_ids: list[NonNegativeInt] = Field(default_factory=list)
    before_capabilities: list[PlannerCapability]
    after_capabilities: list[PlannerCapability]
    before_top_k: dict[NonNegativeInt, PositiveInt]
    after_top_k: dict[NonNegativeInt, PositiveInt]
    before_program_hash: NonEmptyStr
    after_program_hash: NonEmptyStr

    @model_validator(mode="after")
    def _real_effect(self) -> PlannerDirectiveEffect:
        changed = self.before_program_hash != self.after_program_hash
        if changed != (self.status is PlannerDirectiveStatus.APPLIED):
            raise ValueError("only applied directives may change program dataflow")
        return self


class DoubterDimensionVerdict(StrEnum):
    """Provider verdict for a single applicable Doubter dimension."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class DoubterDimensions(BaseSchema):
    """Evidence-grounded dimensions returned by the provider."""

    candidate_supported: bool
    answer_entailed: bool
    path_sufficient: bool
    modality_sufficient: bool
    raw_image_sufficient: bool
    context_sufficient: bool
    multi_hop_sufficient: bool
    temporal_consistent: bool
    non_redundant: bool
    refusal_appropriate: bool


class DoubterProviderDimensionsV11(BaseSchema):
    """Dimensions-only v1.1 response; applicability is enforced by the service."""

    candidate_supported: DoubterDimensionVerdict
    answer_entailed: DoubterDimensionVerdict
    path_sufficient: DoubterDimensionVerdict
    modality_sufficient: DoubterDimensionVerdict
    raw_image_sufficient: DoubterDimensionVerdict
    context_sufficient: DoubterDimensionVerdict
    multi_hop_sufficient: DoubterDimensionVerdict
    temporal_consistent: DoubterDimensionVerdict
    retrieval_nonredundant: DoubterDimensionVerdict
    refusal_appropriate: DoubterDimensionVerdict


class DoubterProviderResultV11(BaseSchema):
    """Step 7 provider output without model-authored issues or actions."""

    candidate_alias: NonEmptyStr
    dimensions: DoubterProviderDimensionsV11


class DoubterProviderEnvelopeV11(BaseSchema):
    result: DoubterProviderResultV11


class DoubterProviderResult(BaseSchema):
    """Strict provider result. The service, not the model, derives the action."""

    candidate_alias: NonEmptyStr
    dimensions: DoubterDimensions
    issue_codes: list[DoubterIssueCode]

    @model_validator(mode="after")
    def _validate_issues(self) -> DoubterProviderResult:
        ensure_unique_strings([item.value for item in self.issue_codes], "issue_codes")
        return self


class DoubterProviderEnvelope(BaseSchema):
    result: DoubterProviderResult


class DoubterDirective(BaseSchema):
    directive_id: NonEmptyStr
    feedback_id: NonEmptyStr
    run_id: NonEmptyStr
    qbank_id: NonEmptyStr
    round_id: PositiveInt
    ordinal: NonNegativeInt
    directive: DoubterDirectiveCode
    top_k_tier: PositiveInt | None = Field(default=None, le=50)
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_top_k(self) -> DoubterDirective:
        if (self.directive is DoubterDirectiveCode.INCREASE_TOP_K) != (self.top_k_tier is not None):
            raise ValueError("increase_top_k requires exactly one top_k_tier")
        return self


class DoubterAssessment(BaseSchema):
    assessment_id: NonEmptyStr
    feedback_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    qbank_id: NonEmptyStr
    plan_id: NonEmptyStr
    execution_id: NonEmptyStr
    producer_model_call_id: NonEmptyStr
    round_id: NonNegativeInt
    schema_version: NonEmptyStr = DOUBTER_SCHEMA_VERSION
    prompt_version: NonEmptyStr
    dimensions: DoubterDimensions
    issue_codes: list[DoubterIssueCode]
    action: DoubterAction
    directives: list[DoubterDirective] = Field(default_factory=list)
    provider_schema_conformant: bool
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_action(self) -> DoubterAssessment:
        ensure_unique_strings([item.value for item in self.issue_codes], "issue_codes")
        if self.round_id > 3:
            raise ValueError("Step 6D supports feedback rounds 0..3")
        if self.action is DoubterAction.PASS and (self.issue_codes or self.directives):
            raise ValueError("pass assessments cannot contain issues or directives")
        if self.action is DoubterAction.REVISE_PROGRAM and not self.directives:
            raise ValueError("program revision requires typed directives")
        if self.action is DoubterAction.REJECT_CANDIDATE and self.directives:
            raise ValueError("rejected candidates cannot contain revision directives")
        if any(item.feedback_id != self.feedback_id for item in self.directives):
            raise ValueError("Doubter directives must belong to their assessment feedback")
        return self


class PlannerRevisionLineage(BaseSchema):
    revision_lineage_id: NonEmptyStr
    run_id: NonEmptyStr
    qbank_id: NonEmptyStr
    source_plan_id: NonEmptyStr
    revised_plan_id: NonEmptyStr
    source_feedback_id: NonEmptyStr
    target_round_id: PositiveInt
    producer_model_call_id: NonEmptyStr
    directive_hash: NonEmptyStr
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_round(self) -> PlannerRevisionLineage:
        if self.target_round_id > 3:
            raise ValueError("Step 6D revision round cannot exceed 3")
        if self.source_plan_id == self.revised_plan_id:
            raise ValueError("a Planner revision must create a new program")
        return self


class PlannerRevisionPatch(BaseSchema):
    patch_id: NonEmptyStr
    run_id: NonEmptyStr
    qbank_id: NonEmptyStr
    source_plan_id: NonEmptyStr
    target_round_id: PositiveInt
    directive_hash: NonEmptyStr
    before_strategy_hash: NonEmptyStr
    after_strategy_hash: NonEmptyStr
    before_program_hash: NonEmptyStr
    after_program_hash: NonEmptyStr | None = None
    applied: bool
    rejection_code: NonEmptyStr | None = None
    capability_added: list[PlannerCapability] = Field(default_factory=list)
    capability_removed: list[PlannerCapability] = Field(default_factory=list)
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @property
    def directive_effects(self) -> list[PlannerDirectiveEffect]:
        # Existing persistence round-trips metadata without a new SQL column.
        return [
            PlannerDirectiveEffect.model_validate(item)
            for item in self.metadata.get("directive_effects", [])
        ]

    @model_validator(mode="after")
    def _validate_patch(self) -> PlannerRevisionPatch:
        if self.target_round_id > 3:
            raise ValueError("Planner revision patch round cannot exceed 3")
        if self.applied != (self.after_program_hash is not None):
            raise ValueError("applied patches require an after program hash")
        if self.applied == (self.rejection_code is not None):
            raise ValueError("rejected patches require exactly one rejection code")
        if self.applied and self.before_program_hash == self.after_program_hash:
            raise ValueError("applied Planner revision patches cannot be no-ops")
        return self


class R0InputManifest(BaseSchema):
    r0_manifest_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    schema_version: NonEmptyStr = DOUBTER_SCHEMA_VERSION
    candidate_count: NonNegativeInt
    answerable_candidate_count: NonNegativeInt
    control_candidate_count: NonNegativeInt
    program_count: NonNegativeInt
    execution_count: NonNegativeInt
    terminal_outcome_count: NonNegativeInt
    questioner_identity_hash: NonEmptyStr
    planner_identity_hash: NonEmptyStr
    operator_registry_hash: NonEmptyStr
    memory_index_hash: NonEmptyStr
    query_vector_hash: NonEmptyStr
    facts_hash: NonEmptyStr
    artifact_hash: NonEmptyStr
    hard_gate_passed: bool
    issue_codes: list[NonEmptyStr] = Field(default_factory=list)
    protocol_integrity_issues: list[NonEmptyStr] = Field(default_factory=list)
    candidate_scientific_rejections: JsonDict = Field(default_factory=dict)
    coverage_zero_units: NonNegativeInt = 0
    downstream_viability: bool = False
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_accounting(self) -> R0InputManifest:
        if self.answerable_candidate_count + self.control_candidate_count != self.candidate_count:
            raise ValueError("R0 route-class candidate counts must reconcile")
        if self.hard_gate_passed == bool(self.issue_codes):
            raise ValueError("R0 gate status and issue codes are inconsistent")
        if self.protocol_integrity_issues and set(self.protocol_integrity_issues) != set(
            self.issue_codes
        ):
            raise ValueError("R0 protocol issues must match legacy issue codes")
        return self


class CandidateStageKind(StrEnum):
    DOUBTER = "doubter"
    PLANNER_REVISION = "planner_revision"
    DRY_RUN = "dry_run"
    DECISION = "decision"


class CandidateStageJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CandidateStageJob(BaseSchema):
    candidate_stage_job_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    qbank_id: NonEmptyStr
    stage: CandidateStageKind
    round_id: NonNegativeInt
    status: CandidateStageJobStatus = CandidateStageJobStatus.PENDING
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
    def _validate_job(self) -> CandidateStageJob:
        if self.round_id > 3:
            raise ValueError("Step 6D candidate job round cannot exceed 3")
        running = self.status is CandidateStageJobStatus.RUNNING
        if running != (self.claimed_by is not None):
            raise ValueError("only running jobs have a claimant")
        if running != (self.lease_expires_at is not None):
            raise ValueError("only running jobs have a lease")
        terminal = self.status in {
            CandidateStageJobStatus.SUCCEEDED,
            CandidateStageJobStatus.FAILED,
            CandidateStageJobStatus.CANCELLED,
        }
        if terminal != (self.finished_at is not None):
            raise ValueError("terminal candidate jobs require finished_at")
        if self.status is CandidateStageJobStatus.FAILED and self.error_type is None:
            raise ValueError("failed candidate jobs require error_type")
        if self.status is not CandidateStageJobStatus.FAILED and self.error_type is not None:
            raise ValueError("only failed candidate jobs may contain error_type")
        return self


class RoundProjectionOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"
    MAX_ROUNDS_EXHAUSTED = "max_rounds_exhausted"
    CARRIED_FORWARD = "carried_forward"


class QBankRoundProjectionMember(BaseSchema):
    projection_member_id: NonEmptyStr
    projection_id: NonEmptyStr
    run_id: NonEmptyStr
    qbank_id: NonEmptyStr
    route_class: QBankRouteClass
    outcome: RoundProjectionOutcome
    effective_plan_id: NonEmptyStr | None = None
    effective_feedback_id: NonEmptyStr | None = None
    passed_at_round: NonNegativeInt | None = None
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_outcome(self) -> QBankRoundProjectionMember:
        accepted = self.outcome in {
            RoundProjectionOutcome.ACCEPTED,
            RoundProjectionOutcome.CARRIED_FORWARD,
        }
        if accepted != (self.effective_plan_id is not None):
            raise ValueError("accepted projection members require an effective plan")
        if self.passed_at_round is not None and self.passed_at_round > 3:
            raise ValueError("passed_at_round cannot exceed 3")
        return self


class QBankRoundProjectionManifest(BaseSchema):
    projection_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    schema_version: NonEmptyStr = VERIFIED_QBANK_SCHEMA_VERSION
    round_id: NonNegativeInt
    source_r0_manifest_id: NonEmptyStr
    member_count: NonNegativeInt
    accepted_count: NonNegativeInt
    rejected_count: NonNegativeInt
    pending_count: NonNegativeInt
    members_hash: NonEmptyStr
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts(self) -> QBankRoundProjectionManifest:
        if self.round_id > 3:
            raise ValueError("projection round cannot exceed 3")
        if self.accepted_count + self.rejected_count + self.pending_count != self.member_count:
            raise ValueError("round projection counts must reconcile")
        return self


class RoutingProjectionMember(BaseSchema):
    routing_member_id: NonEmptyStr
    routing_projection_id: NonEmptyStr
    run_id: NonEmptyStr
    qbank_id: NonEmptyStr
    plan_id: NonEmptyStr
    route_class: QBankRouteClass
    source_projection_member_id: NonEmptyStr
    ordinal: NonNegativeInt
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)


class RoutingProjectionManifest(BaseSchema):
    routing_projection_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    schema_version: NonEmptyStr = VERIFIED_QBANK_SCHEMA_VERSION
    active_round: NonNegativeInt = 2
    max_round: NonNegativeInt = 3
    source_projection_id: NonEmptyStr
    decision_hash: NonEmptyStr
    lexical_collection_id: NonEmptyStr
    text_collection_id: NonEmptyStr
    multimodal_collection_id: NonEmptyStr
    member_count: NonNegativeInt
    answerable_count: NonNegativeInt
    control_count: NonNegativeInt
    members_hash: NonEmptyStr
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_projection(self) -> RoutingProjectionManifest:
        if self.active_round != 2 or self.max_round != 3:
            raise ValueError("Step 6D freezes active_round=2 and max_round=3")
        if self.answerable_count + self.control_count != self.member_count:
            raise ValueError("routing projection route-class counts must reconcile")
        if (
            len(
                {
                    self.lexical_collection_id,
                    self.text_collection_id,
                    self.multimodal_collection_id,
                }
            )
            != 3
        ):
            raise ValueError("routing projection collections must be distinct")
        return self


class StageComponentStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class StageComponentManifest(BaseSchema):
    component_id: NonEmptyStr
    launch_id: NonEmptyStr
    run_id: NonEmptyStr | None = None
    schema_version: NonEmptyStr = STAGE_COMPONENT_SCHEMA_VERSION
    stage: NonEmptyStr
    dataset: NonEmptyStr | None = None
    round_id: NonNegativeInt | None = None
    scientific_input_hash: NonEmptyStr
    source_fingerprint: NonEmptyStr
    upstream_output_hash: NonEmptyStr
    output_hash: NonEmptyStr
    alembic_revision: NonEmptyStr
    row_counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    artifact_hashes: list[NonEmptyStr] = Field(default_factory=list)
    status: StageComponentStatus
    image_digest: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    completed_at: UtcDateTime | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_status(self) -> StageComponentManifest:
        ensure_unique_strings(self.artifact_hashes, "artifact_hashes")
        terminal = self.status in {StageComponentStatus.READY, StageComponentStatus.FAILED}
        if terminal != (self.completed_at is not None):
            raise ValueError("terminal stage components require completed_at")
        if self.round_id is not None and self.round_id > 3:
            raise ValueError("stage component round cannot exceed 3")
        return self


class VerifiedQBankFreezeManifest(BaseSchema):
    freeze_manifest_id: NonEmptyStr
    schema_version: NonEmptyStr = VERIFIED_QBANK_SCHEMA_VERSION
    run_id: NonEmptyStr
    snapshot_ids: list[NonEmptyStr]
    active_round: NonNegativeInt = 2
    max_round: NonNegativeInt = 3
    questioner_identity_hash: NonEmptyStr
    planner_identity_hash: NonEmptyStr
    doubter_identity_hash: NonEmptyStr
    operator_registry_hash: NonEmptyStr
    routing_projection_hash: NonEmptyStr
    semantic_judge_identity_hash: NonEmptyStr
    pilot_content_hash: NonEmptyStr
    approved_by: NonEmptyStr
    approved_at: UtcDateTime
    decision_content_hash: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_freeze(self) -> VerifiedQBankFreezeManifest:
        ensure_unique_strings(self.snapshot_ids, "snapshot_ids")
        if not self.snapshot_ids:
            raise ValueError("Verified QBank freeze requires snapshots")
        if self.active_round != 2 or self.max_round != 3:
            raise ValueError("Verified QBank freeze requires active r2 and maximum r3")
        return self


class DoubterRoundMetrics(BaseSchema):
    round_id: NonNegativeInt
    candidate_count: NonNegativeInt
    passed_count: NonNegativeInt
    revised_count: NonNegativeInt
    rejected_count: NonNegativeInt
    exhausted_count: NonNegativeInt
    rescued_count: NonNegativeInt
    model_call_count: NonNegativeInt
    token_count: NonNegativeInt
    elapsed_ms: NonNegativeFloat
    estimated_cost: NonNegativeFloat = 0.0
    pass_rate: Probability | None = None

    @model_validator(mode="after")
    def _validate_metrics(self) -> DoubterRoundMetrics:
        if self.round_id > 3:
            raise ValueError("Doubter round metrics cannot exceed 3")
        if (
            self.passed_count + self.revised_count + self.rejected_count + self.exhausted_count
            != self.candidate_count
        ):
            raise ValueError("Doubter round outcome counts must reconcile")
        return self
