"""Memory program and operator schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    EvidenceType,
    JsonDict,
    Modality,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    Probability,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.score import RankingScore


class OperatorControlAction(StrEnum):
    """Control signal emitted by terminal evidence-control operators."""

    CONTINUE = "continue"
    FALLBACK = "fallback"
    ABSTAIN = "abstain"


class OperatorExecutionPurpose(StrEnum):
    COMPILER_DRY_RUN = "compiler_dry_run"
    QUERY_TIME = "query_time"


class ProgramStep(BaseSchema):
    """One executable operator step in a memory program."""

    step_id: NonNegativeInt
    op_name: NonEmptyStr
    args: JsonDict = Field(default_factory=dict)
    expected_output_type: NonEmptyStr
    expected_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ids(self) -> ProgramStep:
        ensure_unique_strings(self.expected_evidence_ids, "expected_evidence_ids")
        return self


class MemoryProgram(BaseSchema):
    """Planner output: a typed, executable memory-access program."""

    plan_id: NonEmptyStr
    qbank_id: NonEmptyStr
    run_id: NonEmptyStr | None = None
    snapshot_id: NonEmptyStr | None = None
    compiler_unit_id: NonEmptyStr | None = None
    producer_model_call_id: NonEmptyStr | None = None
    operator_set_version: NonEmptyStr | None = None
    round_id: NonNegativeInt
    operator_sequence: list[ProgramStep]
    expected_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    fallback_operator: NonEmptyStr | None = None
    required_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    planner_rationale_short: NonEmptyStr
    planner_confidence: Probability
    program_hash: NonEmptyStr | None = None
    accepted: bool = True
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_program(self) -> MemoryProgram:
        if not self.operator_sequence:
            raise ValueError("operator_sequence must contain at least one item")
        step_ids = [step.step_id for step in self.operator_sequence]
        if step_ids != list(range(len(step_ids))):
            raise ValueError("operator_sequence step IDs must be contiguous and ordered from 0")
        ensure_unique_strings(self.expected_evidence_ids, "expected_evidence_ids")
        if not self.required_modalities:
            raise ValueError("required_modalities must contain at least one item")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicate values")
        if self.compiler_unit_id is not None:
            if self.run_id is None or self.snapshot_id is None:
                raise ValueError("compiler-scoped programs require run_id and snapshot_id")
            if self.operator_set_version is None or self.program_hash is None:
                raise ValueError(
                    "compiler-scoped programs require operator_set_version and program_hash"
                )
        return self


class OperatorInput(BaseSchema):
    """Normalized input passed to a PMMC operator."""

    run_id: NonEmptyStr | None = None
    snapshot_id: NonEmptyStr | None = None
    query_id: NonEmptyStr | None = None
    conversation_id: NonEmptyStr
    cutoff_session_id: NonEmptyStr | None = None
    compiler_unit_id: NonEmptyStr | None = None
    question: NonEmptyStr | None = None
    args: JsonDict = Field(default_factory=dict)
    constraints: JsonDict = Field(default_factory=dict)
    previous_step_outputs: list[JsonDict] = Field(default_factory=list)
    collection_bindings: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)


class OperatorOutput(BaseSchema):
    """Normalized output returned by a PMMC operator."""

    retrieved_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_image_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_caption_ids: list[NonEmptyStr] = Field(default_factory=list)
    scores: dict[NonEmptyStr, RankingScore] = Field(default_factory=dict)
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    control_action: OperatorControlAction = OperatorControlAction.CONTINUE
    success: bool = True
    latency_ms: NonNegativeFloat | None = None
    error: NonEmptyStr | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_output(self) -> OperatorOutput:
        ensure_unique_strings(self.retrieved_memory_ids, "retrieved_memory_ids")
        ensure_unique_strings(self.retrieved_image_ids, "retrieved_image_ids")
        ensure_unique_strings(self.retrieved_session_ids, "retrieved_session_ids")
        ensure_unique_strings(self.retrieved_caption_ids, "retrieved_caption_ids")
        if self.success and (self.error is not None or self.error_type is not None):
            raise ValueError("successful operator output cannot include error")
        if not self.success and self.error is None and self.error_type is None:
            raise ValueError("failed operator output requires error or error_type")
        if self.control_action is not OperatorControlAction.CONTINUE and not self.success:
            raise ValueError("control actions can only be emitted by successful operators")
        return self


class EvidenceBundle(BaseSchema):
    """Evidence passed from program execution into answer generation."""

    evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    text_ids: list[NonEmptyStr] = Field(default_factory=list)
    caption_ids: list[NonEmptyStr] = Field(default_factory=list)
    image_ids: list[NonEmptyStr] = Field(default_factory=list)
    session_ids: list[NonEmptyStr] = Field(default_factory=list)
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ids(self) -> EvidenceBundle:
        for field_name in (
            "evidence_ids",
            "memory_ids",
            "text_ids",
            "caption_ids",
            "image_ids",
            "session_ids",
        ):
            ensure_unique_strings(getattr(self, field_name), field_name)
        return self
