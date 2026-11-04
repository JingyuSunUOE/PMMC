"""Versioned contracts for PMMC write-time compilation and operator execution."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    COMPILER_SCHEMA_VERSION,
    SUPPORTED_COMPILER_SCHEMA_VERSIONS,
    BaseSchema,
    EvidenceType,
    FiniteFloat,
    JsonDict,
    Modality,
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
from pmmc.schemas.program import OperatorControlAction, OperatorExecutionPurpose
from pmmc.schemas.score import RankingScore


class CompilationTargetStage(StrEnum):
    """Completeness level requested from one compiler run."""

    QUESTIONS = "questions"
    PROGRAMS = "programs"
    VERIFIED_QBANK = "verified_qbank"


class CompilationStage(StrEnum):
    """Auditable stages inside one compilation-unit job."""

    QUESTIONER = "questioner"
    PLANNER = "planner"
    DRY_RUN = "dry_run"
    DOUBTER = "doubter"
    DECISION = "decision"


class CompilationUnitMemoryRole(StrEnum):
    """Role of a canonical memory in a deterministic compilation unit."""

    ANCHOR = "anchor"
    SUPPORT = "support"


class ProgramValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class QBankEvidenceRole(StrEnum):
    SOURCE = "source"
    EXPECTED = "expected"
    CANDIDATE_SOURCE_SUPPORT = "candidate_source_support"
    COMPILER_RETRIEVED_PATH = "compiler_retrieved_path"
    QUERY_RUNTIME_EVIDENCE = "query_runtime_evidence"
    EVALUATOR_GOLD = "evaluator_gold"


class CompilerRunSpec(BaseSchema):
    """Compiler-specific immutable denominator linked to a queryless experiment run."""

    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    compiler_schema_version: NonEmptyStr = COMPILER_SCHEMA_VERSION
    target_stage: CompilationTargetStage
    operator_set_version: NonEmptyStr
    operator_registry_hash: NonEmptyStr
    compiler_config_hash: NonEmptyStr
    questioner_config_hash: NonEmptyStr | None = None
    questioner_model_alias: NonEmptyStr | None = None
    questioner_model_revision: NonEmptyStr | None = None
    questioner_prompt_version: NonEmptyStr | None = None
    questioner_prompt_hash: NonEmptyStr | None = None
    questioner_renderer_version: NonEmptyStr | None = None
    questioner_parser_version: NonEmptyStr | None = None
    unit_builder_version: NonEmptyStr | None = None
    candidate_gate_version: NonEmptyStr | None = None
    dedupe_version: NonEmptyStr | None = None
    qbank_matcher_version: NonEmptyStr | None = None
    questioner_freeze_manifest_hash: NonEmptyStr | None = None
    planner_config_hash: NonEmptyStr | None = None
    planner_model_alias: NonEmptyStr | None = None
    planner_model_revision: NonEmptyStr | None = None
    planner_prompt_version: NonEmptyStr | None = None
    planner_prompt_hash: NonEmptyStr | None = None
    planner_renderer_version: NonEmptyStr | None = None
    planner_parser_version: NonEmptyStr | None = None
    planner_schema_version: NonEmptyStr | None = None
    program_policy_version: NonEmptyStr | None = None
    dry_run_policy_version: NonEmptyStr | None = None
    memory_index_representation_version: NonEmptyStr | None = None
    memory_text_model_alias: NonEmptyStr | None = None
    memory_text_model_revision: NonEmptyStr | None = None
    memory_multimodal_model_alias: NonEmptyStr | None = None
    memory_multimodal_model_revision: NonEmptyStr | None = None
    doubter_config_hash: NonEmptyStr | None = None
    doubter_model_alias: NonEmptyStr | None = None
    doubter_model_revision: NonEmptyStr | None = None
    doubter_prompt_version: NonEmptyStr | None = None
    doubter_prompt_hash: NonEmptyStr | None = None
    doubter_renderer_version: NonEmptyStr | None = None
    doubter_parser_version: NonEmptyStr | None = None
    doubter_schema_version: NonEmptyStr | None = None
    r0_input_policy_version: NonEmptyStr | None = None
    routing_projection_version: NonEmptyStr | None = None
    active_round: NonNegativeInt | None = None
    max_round: NonNegativeInt | None = None
    unit_count: NonNegativeInt
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_version(self) -> CompilerRunSpec:
        if self.compiler_schema_version not in SUPPORTED_COMPILER_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported compiler schema version: {self.compiler_schema_version}"
            )
        identity = (
            self.questioner_config_hash,
            self.questioner_model_alias,
            self.questioner_model_revision,
            self.questioner_prompt_version,
            self.questioner_prompt_hash,
            self.questioner_renderer_version,
            self.questioner_parser_version,
            self.unit_builder_version,
            self.candidate_gate_version,
            self.dedupe_version,
            self.qbank_matcher_version,
        )
        if self.compiler_schema_version in {
            "0.2.0",
            "0.3.0",
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
            "0.9.0",
        } and any(value is None for value in identity):
            raise ValueError(
                "Questioner compiler schemas require the complete producer identity"
            )
        if self.compiler_schema_version == "0.1.0" and any(
            value is not None for value in identity
        ):
            raise ValueError(
                "legacy compiler schema 0.1.0 cannot contain Questioner-v1 identity"
            )
        planner_identity = (
            self.questioner_freeze_manifest_hash,
            self.planner_config_hash,
            self.planner_model_alias,
            self.planner_model_revision,
            self.planner_prompt_version,
            self.planner_prompt_hash,
            self.planner_renderer_version,
            self.planner_parser_version,
            self.planner_schema_version,
            self.program_policy_version,
            self.dry_run_policy_version,
            self.memory_index_representation_version,
            self.memory_text_model_alias,
            self.memory_text_model_revision,
            self.memory_multimodal_model_alias,
            self.memory_multimodal_model_revision,
        )
        if self.compiler_schema_version in {
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
            "0.9.0",
        }:
            expected_target = (
                CompilationTargetStage.VERIFIED_QBANK
                if self.compiler_schema_version in {"0.6.0", "0.7.0", "0.8.0", "0.9.0"}
                else CompilationTargetStage.PROGRAMS
            )
            if self.target_stage is not expected_target:
                raise ValueError(
                    "Planner compiler target stage does not match its schema version"
                )
            if any(value is None for value in planner_identity):
                raise ValueError("Planner compiler schemas require Planner identity")
        elif any(value is not None for value in planner_identity):
            raise ValueError("legacy compiler schemas cannot contain Planner v1 identity")
        doubter_identity = (
            self.doubter_config_hash,
            self.doubter_model_alias,
            self.doubter_model_revision,
            self.doubter_prompt_version,
            self.doubter_prompt_hash,
            self.doubter_renderer_version,
            self.doubter_parser_version,
            self.doubter_schema_version,
            self.r0_input_policy_version,
            self.routing_projection_version,
            self.active_round,
            self.max_round,
        )
        if self.compiler_schema_version in {"0.6.0", "0.7.0", "0.8.0", "0.9.0"}:
            if any(value is None for value in doubter_identity):
                raise ValueError("verified QBank compiler runs require Doubter identity")
            if self.active_round != 2 or self.max_round != 3:
                raise ValueError(
                    "verified QBank compiler runs require active r2 and maximum r3"
                )
        elif any(value is not None for value in doubter_identity):
            raise ValueError("pre-Step-6D compiler schemas cannot contain Doubter identity")
        return self


class CompilationUnitMemory(BaseSchema):
    """One ordered canonical-memory member of a compilation unit."""

    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    memory_id: NonEmptyStr
    ordinal: NonNegativeInt
    role: CompilationUnitMemoryRole
    metadata: JsonDict = Field(default_factory=dict)


class CompilationUnit(BaseSchema):
    """A bounded, prospective-safe set of canonical memories compiled together."""

    compiler_unit_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    available_after_session: NonEmptyStr
    ordinal: NonNegativeInt
    builder_version: NonEmptyStr = Field(max_length=128)
    input_hash: NonEmptyStr
    memories: list[CompilationUnitMemory]
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_members(self) -> CompilationUnit:
        if not self.memories:
            raise ValueError("compilation unit requires at least one memory")
        if [item.ordinal for item in self.memories] != list(range(len(self.memories))):
            raise ValueError("compilation unit memory ordinals must be contiguous from 0")
        ensure_unique_strings([item.memory_id for item in self.memories], "memories.memory_id")
        if not any(item.role is CompilationUnitMemoryRole.ANCHOR for item in self.memories):
            raise ValueError("compilation unit requires at least one anchor memory")
        for item in self.memories:
            if item.snapshot_id != self.snapshot_id:
                raise ValueError("compilation unit memories must use the unit snapshot")
            if item.conversation_id != self.conversation_id:
                raise ValueError("compilation unit memories must use the unit conversation")
        return self


class CompilationStageResult(BaseSchema):
    """Immutable outcome for one internal stage of one compilation unit."""

    stage_result_id: NonEmptyStr
    run_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    stage: CompilationStage
    attempt_number: PositiveInt = 1
    status: TraceStatus
    produced_count: NonNegativeInt = 0
    accepted_count: NonNegativeInt = 0
    rejected_count: NonNegativeInt = 0
    result_hash: NonEmptyStr
    started_at: UtcDateTime
    completed_at: UtcDateTime
    elapsed_ms: NonNegativeFloat
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_result(self) -> CompilationStageResult:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        if self.accepted_count + self.rejected_count > self.produced_count:
            raise ValueError("accepted and rejected counts cannot exceed produced_count")
        if self.status is TraceStatus.FAILED and self.error_type is None:
            raise ValueError("failed compiler stage results require error_type")
        if self.status is not TraceStatus.FAILED and self.error_type is not None:
            raise ValueError("non-failed compiler stage results cannot include error_type")
        return self


class OperatorSpec(BaseSchema):
    """Frozen public identity and resource contract for one memory operator."""

    name: NonEmptyStr
    version: NonEmptyStr
    description: NonEmptyStr
    args_schema_hash: NonEmptyStr
    output_type: NonEmptyStr
    evidence_type: EvidenceType
    terminal: bool = False
    collection_role: NonEmptyStr | None = None
    max_top_k: PositiveInt | None = None
    metadata: JsonDict = Field(default_factory=dict)


class OperatorEvidence(BaseSchema):
    """Typed evidence reference returned by an operator data adapter."""

    evidence_id: NonEmptyStr
    evidence_type: EvidenceType
    memory_id: NonEmptyStr | None = None
    image_id: NonEmptyStr | None = None
    session_id: NonEmptyStr | None = None
    caption_id: NonEmptyStr | None = None
    score: RankingScore | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_typed_reference(self) -> OperatorEvidence:
        typed = (self.memory_id, self.image_id, self.session_id, self.caption_id)
        if sum(value is not None for value in typed) != 1:
            raise ValueError("operator evidence requires exactly one typed canonical reference")
        _validate_evidence_type(
            evidence_type=self.evidence_type,
            memory_id=self.memory_id,
            image_id=self.image_id,
            session_id=self.session_id,
            caption_id=self.caption_id,
        )
        return self


class OperatorExecutionContext(BaseSchema):
    """Immutable scope and precomputed vectors supplied to an operator program."""

    run_id: NonEmptyStr
    source_compiler_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    cutoff_session_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr | None = None
    query_id: NonEmptyStr | None = None
    question: NonEmptyStr
    query_image_ids: list[NonEmptyStr] = Field(default_factory=list)
    required_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    collection_bindings: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    query_vectors: dict[NonEmptyStr, list[FiniteFloat]] = Field(default_factory=dict)
    constraints: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_context(self) -> OperatorExecutionContext:
        ensure_unique_strings(self.query_image_ids, "query_image_ids")
        if not self.required_modalities:
            raise ValueError("required_modalities must contain at least one item")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicate values")
        if any(not vector for vector in self.query_vectors.values()):
            raise ValueError("query vectors must not be empty")
        return self


class ProgramValidationIssue(BaseSchema):
    code: NonEmptyStr
    message: NonEmptyStr
    severity: ProgramValidationSeverity = ProgramValidationSeverity.ERROR
    step_id: NonNegativeInt | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ProgramValidationResult(BaseSchema):
    valid: bool
    program_hash: NonEmptyStr
    operator_registry_hash: NonEmptyStr
    issues: list[ProgramValidationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_status(self) -> ProgramValidationResult:
        has_errors = any(
            issue.severity is ProgramValidationSeverity.ERROR for issue in self.issues
        )
        if self.valid == has_errors:
            raise ValueError("valid must be the inverse of whether error issues exist")
        return self


class ProgramExecutionEvidenceRef(BaseSchema):
    """One ranked, typed evidence item emitted during a program execution."""

    evidence_ref_id: NonEmptyStr
    execution_id: NonEmptyStr
    step_id: NonNegativeInt
    snapshot_id: NonEmptyStr
    evidence_id: NonEmptyStr
    evidence_type: EvidenceType
    memory_id: NonEmptyStr | None = None
    image_id: NonEmptyStr | None = None
    session_id: NonEmptyStr | None = None
    caption_id: NonEmptyStr | None = None
    rank: PositiveInt
    score: RankingScore | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reference(self) -> ProgramExecutionEvidenceRef:
        typed = (self.memory_id, self.image_id, self.session_id, self.caption_id)
        if sum(value is not None for value in typed) != 1:
            raise ValueError(
                "program execution evidence requires exactly one typed reference"
            )
        _validate_evidence_type(
            evidence_type=self.evidence_type,
            memory_id=self.memory_id,
            image_id=self.image_id,
            session_id=self.session_id,
            caption_id=self.caption_id,
        )
        return self


def _validate_evidence_type(
    *,
    evidence_type: EvidenceType,
    memory_id: str | None,
    image_id: str | None,
    session_id: str | None,
    caption_id: str | None,
) -> None:
    allowed = {
        "memory": {
            EvidenceType.MEMORY,
            EvidenceType.TEXT,
            EvidenceType.CAPTION,
            EvidenceType.IMAGE_TEXT_PAIR,
        },
        "image": {
            EvidenceType.IMAGE,
            EvidenceType.RAW_IMAGE,
            EvidenceType.IMAGE_TEXT_PAIR,
        },
        "session": {EvidenceType.SESSION},
        "caption": {EvidenceType.CAPTION},
    }
    kind = (
        "memory"
        if memory_id is not None
        else "image"
        if image_id is not None
        else "session"
        if session_id is not None
        else "caption"
        if caption_id is not None
        else "unknown"
    )
    if evidence_type not in allowed.get(kind, set()):
        raise ValueError(f"evidence_type is incompatible with the {kind} reference")


class ProgramExecutionTrace(BaseSchema):
    """Aggregate record for compiler dry-run or future query-time execution."""

    execution_id: NonEmptyStr
    run_id: NonEmptyStr
    source_compiler_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    plan_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr | None = None
    query_id: NonEmptyStr | None = None
    purpose: OperatorExecutionPurpose
    attempt_number: PositiveInt = 1
    status: TraceStatus
    bindings_hash: NonEmptyStr
    control_action: OperatorControlAction = OperatorControlAction.CONTINUE
    started_at: UtcDateTime
    completed_at: UtcDateTime
    elapsed_ms: NonNegativeFloat
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_execution(self) -> ProgramExecutionTrace:
        if (
            self.purpose is OperatorExecutionPurpose.COMPILER_DRY_RUN
            and self.compiler_unit_id is None
        ):
            raise ValueError("compiler dry-run execution requires compiler_unit_id")
        if (
            self.purpose is OperatorExecutionPurpose.COMPILER_DRY_RUN
            and self.query_id is not None
        ):
            raise ValueError("compiler dry-run execution cannot bind a benchmark query")
        if (
            self.purpose is OperatorExecutionPurpose.COMPILER_DRY_RUN
            and self.run_id != self.source_compiler_run_id
        ):
            raise ValueError("compiler dry-run must belong to its source compiler run")
        if self.purpose is OperatorExecutionPurpose.QUERY_TIME and self.query_id is None:
            raise ValueError("query-time execution requires query_id")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        if self.status is TraceStatus.FAILED and self.error_type is None:
            raise ValueError("failed program executions require error_type")
        if self.status is not TraceStatus.FAILED and self.error_type is not None:
            raise ValueError("non-failed program executions cannot include error_type")
        return self


class QBankDecision(BaseSchema):
    """Final append-only acceptance decision for one generated question."""

    decision_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr
    qbank_id: NonEmptyStr
    accepted: bool
    final_plan_id: NonEmptyStr | None = None
    final_feedback_id: NonEmptyStr | None = None
    reason_code: NonEmptyStr
    reason_summary: NonEmptyStr | None = None
    verifier_pass: bool = False
    verifier_score: Probability | None = None
    decision_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_decision(self) -> QBankDecision:
        if self.accepted and self.final_plan_id is None:
            raise ValueError("accepted qbank decisions require final_plan_id")
        if self.accepted and not self.verifier_pass:
            raise ValueError("accepted qbank decisions require verifier_pass")
        if not self.accepted and self.verifier_pass:
            raise ValueError("rejected qbank decisions cannot pass verification")
        return self
