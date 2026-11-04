"""Contracts for structured-output diagnostics and evaluator-only QBank semantics."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    QBANK_SEMANTIC_SCHEMA_VERSION,
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.persistence import JobStatus

SEMANTIC_RUN_SCHEMA_VERSIONS = {
    "qbank-semantic-match-v1": "0.1.0",
    "qbank-semantic-match-v2": "0.2.0",
    "qbank-semantic-match-v3": "0.3.0",
    "qbank-semantic-match-v4": "0.4.0",
}


def semantic_run_schema_version(protocol_version: str) -> str:
    try:
        return SEMANTIC_RUN_SCHEMA_VERSIONS[protocol_version]
    except KeyError:
        raise ValueError("unsupported QBank semantic protocol") from None


class StructuredOutputStage(StrEnum):
    QUESTIONER = "questioner"
    QUESTIONER_VERIFIER = "questioner_verifier"
    QUESTIONER_REVISION = "questioner_revision"
    PLANNER = "planner"
    PLANNER_REVISION = "planner_revision"
    DOUBTER = "doubter"
    QBANK_SEMANTIC_JUDGE = "qbank_semantic_judge"


class StructuredOutputItemStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    REPAIRED = "repaired"
    CONSERVATIVE_REJECT = "conservative_reject"


class StructuredOutputValidationError(BaseSchema):
    location: NonEmptyStr
    error_type: NonEmptyStr


class StructuredOutputItemDiagnostic(BaseSchema):
    diagnostic_id: NonEmptyStr
    run_id: NonEmptyStr
    model_call_id: NonEmptyStr
    compiler_unit_id: NonEmptyStr | None = None
    semantic_run_id: NonEmptyStr | None = None
    stage: StructuredOutputStage
    item_ordinal: NonNegativeInt
    item_hash: NonEmptyStr
    status: StructuredOutputItemStatus
    parser_version: NonEmptyStr
    repair_attempt_count: NonNegativeInt = 0
    errors: list[StructuredOutputValidationError] = Field(default_factory=list)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_diagnostic(self) -> StructuredOutputItemDiagnostic:
        if self.status is StructuredOutputItemStatus.VALID and self.errors:
            raise ValueError("valid structured output items cannot contain errors")
        if self.status is not StructuredOutputItemStatus.VALID and not self.errors:
            raise ValueError("non-valid structured output items require safe diagnostics")
        if self.compiler_unit_id is None and self.semantic_run_id is None:
            raise ValueError("structured output diagnostics require compiler or semantic scope")
        if self.compiler_unit_id is not None and self.semantic_run_id is not None:
            raise ValueError("structured output diagnostics cannot cross data scopes")
        if re.fullmatch(r"[0-9a-f]{64}", self.item_hash) is None:
            raise ValueError("structured output item_hash must be SHA256")
        return self


class EvidenceMatchRole(StrEnum):
    DISCRIMINATIVE = "discriminative"
    CONTEXTUAL = "contextual"


class QBankSemanticMatchStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QBankSemanticReason(StrEnum):
    DIFFERENT_INFORMATION_NEED = "different_information_need"
    CANDIDATE_ANSWER_INSUFFICIENT = "candidate_answer_insufficient"
    CONTRADICTION = "contradiction"
    QUERY_IMAGE_BINDING_MISMATCH = "query_image_binding_mismatch"
    EVIDENCE_PROXY_INCOMPLETE = "evidence_proxy_incomplete"
    PROVIDER_OUTPUT_INVALID = "provider_output_invalid"


class QBankSemanticJudgeDimensions(BaseSchema):
    same_information_need: bool
    candidate_answer_sufficient: bool
    contradiction_free: bool
    query_image_binding_compatible: bool | Literal["not_applicable"]


class QBankSemanticJudgeResult(BaseSchema):
    pair_alias: NonEmptyStr
    dimensions: QBankSemanticJudgeDimensions
    reason_codes: list[QBankSemanticReason]

    @model_validator(mode="after")
    def _validate_reasons(self) -> QBankSemanticJudgeResult:
        ensure_unique_strings([item.value for item in self.reason_codes], "reason_codes")
        return self


class QBankSemanticJudgeEnvelope(BaseSchema):
    result: QBankSemanticJudgeResult


class QBankSemanticJudgeResultV2(BaseSchema):
    """Provider-facing v2 result; reason codes are derived by the service."""

    pair_alias: NonEmptyStr
    dimensions: QBankSemanticJudgeDimensions


class QBankSemanticJudgeEnvelopeV2(BaseSchema):
    result: QBankSemanticJudgeResultV2


class QBankSemanticMatchRun(BaseSchema):
    semantic_run_id: NonEmptyStr
    schema_version: NonEmptyStr = QBANK_SEMANTIC_SCHEMA_VERSION
    protocol_version: NonEmptyStr = "qbank-semantic-match-v4"
    coverage_run_id: NonEmptyStr
    compiler_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: DatasetName
    judge_model_alias: NonEmptyStr
    judge_model_revision: NonEmptyStr
    prompt_version: NonEmptyStr
    prompt_hash: NonEmptyStr
    renderer_version: NonEmptyStr
    parser_version: NonEmptyStr
    config_hash: NonEmptyStr
    pair_count: NonNegativeInt
    status: QBankSemanticMatchStatus = QBankSemanticMatchStatus.CREATED
    created_at: UtcDateTime = Field(default_factory=utc_now)
    completed_at: UtcDateTime | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_run(self) -> QBankSemanticMatchRun:
        if self.schema_version != semantic_run_schema_version(self.protocol_version):
            raise ValueError("unsupported QBank semantic schema")
        if re.fullmatch(r"[0-9a-f]{40}", self.judge_model_revision) is None:
            raise ValueError("QBank semantic judge requires a pinned model commit")
        terminal = self.status in {
            QBankSemanticMatchStatus.COMPLETED,
            QBankSemanticMatchStatus.FAILED,
            QBankSemanticMatchStatus.CANCELLED,
        }
        if terminal != (self.completed_at is not None):
            raise ValueError("terminal QBank semantic runs require completed_at")
        return self


class QBankSemanticMatchJob(BaseSchema):
    semantic_job_id: NonEmptyStr
    semantic_run_id: NonEmptyStr
    coverage_run_id: NonEmptyStr
    query_id: NonEmptyStr
    qbank_id: NonEmptyStr
    pair_hash: NonEmptyStr
    status: JobStatus = JobStatus.PENDING
    attempt_count: NonNegativeInt = 0
    max_attempts: PositiveInt = 2
    model_call_id: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    finished_at: UtcDateTime | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_job(self) -> QBankSemanticMatchJob:
        if re.fullmatch(r"[0-9a-f]{64}", self.pair_hash) is None:
            raise ValueError("semantic pair_hash must be SHA256")
        if self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            if self.finished_at is None:
                raise ValueError("terminal semantic jobs require finished_at")
        if self.status is JobStatus.SUCCEEDED and self.model_call_id is None:
            raise ValueError("successful semantic jobs require a model call")
        return self


class QBankSemanticMatchAssessment(BaseSchema):
    semantic_assessment_id: NonEmptyStr
    semantic_run_id: NonEmptyStr
    semantic_job_id: NonEmptyStr
    coverage_run_id: NonEmptyStr
    query_id: NonEmptyStr
    qbank_id: NonEmptyStr
    model_call_id: NonEmptyStr
    dimensions: QBankSemanticJudgeDimensions
    reason_codes: list[QBankSemanticReason] = Field(default_factory=list)
    semantic_dimensions_passed: bool
    evidence_proxy_complete: bool
    semantic_match: bool
    elapsed_ms: NonNegativeFloat
    content_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_assessment(self) -> QBankSemanticMatchAssessment:
        ensure_unique_strings([item.value for item in self.reason_codes], "reason_codes")
        dimensions_passed = (
            self.dimensions.same_information_need
            and self.dimensions.candidate_answer_sufficient
            and self.dimensions.contradiction_free
            and self.dimensions.query_image_binding_compatible is not False
        )
        if self.semantic_dimensions_passed != dimensions_passed:
            raise ValueError("semantic_dimensions_passed disagrees with dimensions")
        if self.semantic_match != (
            self.semantic_dimensions_passed and self.evidence_proxy_complete
        ):
            raise ValueError("semantic_match requires dimensions and evidence proxy")
        return self


class QBankSemanticMetric(BaseSchema):
    semantic_metric_id: NonEmptyStr
    semantic_run_id: NonEmptyStr
    query_id: NonEmptyStr | None = None
    metric_name: NonEmptyStr
    metric_version: NonEmptyStr = "qbank-semantic-match-v2"
    question_budget: PositiveInt
    k: PositiveInt
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)
