"""Doubter feedback schemas."""

from __future__ import annotations

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    EvidenceSufficiency,
    JsonDict,
    Modality,
    NonEmptyStr,
    NonNegativeInt,
    Probability,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)


class DoubterFeedback(BaseSchema):
    """Evidence-grounded feedback from the Doubter."""

    feedback_id: NonEmptyStr
    run_id: NonEmptyStr | None = None
    snapshot_id: NonEmptyStr | None = None
    compiler_unit_id: NonEmptyStr | None = None
    qbank_id: NonEmptyStr
    plan_id: NonEmptyStr
    program_execution_id: NonEmptyStr | None = None
    producer_model_call_id: NonEmptyStr | None = None
    feedback_hash: NonEmptyStr | None = None
    round_id: NonNegativeInt
    verifier_pass: bool
    verifier_score: Probability
    evidence_sufficiency: EvidenceSufficiency
    failure_reason: NonEmptyStr | None = None
    suggested_revision: list[NonEmptyStr] = Field(default_factory=list)
    missing_modalities: list[Modality] = Field(default_factory=list)
    should_abstain: bool = False
    raw_response_asset_id: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_feedback(self) -> DoubterFeedback:
        ensure_unique_strings(self.suggested_revision, "suggested_revision")
        if len(self.missing_modalities) != len(set(self.missing_modalities)):
            raise ValueError("missing_modalities must not contain duplicate values")
        if not self.verifier_pass and not (self.failure_reason or self.suggested_revision):
            raise ValueError("failed verification requires failure_reason or suggested_revision")
        if self.compiler_unit_id is not None and (
            self.run_id is None or self.snapshot_id is None or self.feedback_hash is None
        ):
            raise ValueError(
                "compiler-scoped feedback requires run_id, snapshot_id, and feedback_hash"
            )
        return self
