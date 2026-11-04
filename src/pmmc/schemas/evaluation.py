"""Answer, metric, judge, and evaluation-run contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from pmmc.schemas.base import (
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
from pmmc.schemas.baseline import BaselineMethod
from pmmc.schemas.persistence import JobStatus
from pmmc.schemas.semantic import EvidenceMatchRole


class MetricFamily(StrEnum):
    ANSWER = "answer"
    RETRIEVAL = "retrieval"
    CONTEXT = "context"
    EFFICIENCY = "efficiency"


class EvidenceGranularity(StrEnum):
    GENERIC = "generic"
    SESSION = "session"
    MEMORY = "memory"
    IMAGE = "image"
    TEXT = "text"


class EvidenceMemberType(StrEnum):
    EVIDENCE = "evidence"
    SESSION = "session"
    MEMORY = "memory"
    IMAGE = "image"


class JudgeVerdict(StrEnum):
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    CORRECT = "correct"


class EvaluationStage(StrEnum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"
    ANALYZE = "analyze"


class EvaluationRunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationSourceKind(StrEnum):
    """Authoritative runtime source consumed by the unified evaluator."""

    BASELINE_SUITE = "baseline_suite"
    PMMC_QUERY_RUN = "pmmc_query_run"


class EvaluationContextSourceKind(StrEnum):
    """Runtime fact that supplied the evidence evaluated for an answer."""

    BASELINE_CONTEXT = "baseline_context"
    PROGRAM_EXECUTION = "program_execution"


class BadCaseCategory(StrEnum):
    """Failure categories shared by baselines and future PMMC analysis."""

    RETRIEVAL_TOP50_MISS = "retrieval_top50_miss"
    RETRIEVAL_TOP5_MISS = "retrieval_top5_miss"
    CONTEXT_BUDGET_EXCLUSION = "context_budget_exclusion"
    VISUAL_EVIDENCE_MISS = "visual_evidence_miss"
    TEXT_EVIDENCE_MISS = "text_evidence_miss"
    EVIDENCE_FOUND_ANSWER_WRONG = "evidence_found_answer_wrong"
    ANSWER_FORMAT_MISMATCH = "answer_format_mismatch"
    JUDGE_METRIC_DISAGREEMENT = "judge_metric_disagreement"
    MODEL_CALL_FAILURE = "model_call_failure"
    GOLD_UNRESOLVABLE = "gold_unresolvable"
    QUESTIONER_COVERAGE_FAILURE = "questioner_coverage_failure"
    MATCHING_FAILURE = "matching_failure"
    PLANNER_FAILURE = "planner_failure"
    DOUBTER_FAILURE = "doubter_failure"
    EVIDENCE_INSUFFICIENCY = "evidence_insufficiency"
    ANSWER_GENERATION_FAILURE = "answer_generation_failure"
    UPDATE_CONFLICT_FAILURE = "update_conflict_failure"
    NONE = "none"
    UNKNOWN = "unknown"


class EvaluationRunManifest(BaseSchema):
    evaluation_run_id: NonEmptyStr
    source_kind: EvaluationSourceKind = EvaluationSourceKind.BASELINE_SUITE
    suite_id: NonEmptyStr | None = None
    source_run_id: NonEmptyStr | None = None
    protocol_version: NonEmptyStr
    normalizer_version: NonEmptyStr
    evidence_policy_version: NonEmptyStr
    bad_case_rule_version: NonEmptyStr
    judge_model_alias: NonEmptyStr
    judge_prompt_hash: NonEmptyStr
    judge_rubric_version: NonEmptyStr
    judge_repeat_count: PositiveInt = 1
    judge_score_values: list[Probability] = Field(default_factory=lambda: [0.0, 0.5, 1.0])
    secondary_judge_enabled: bool = False
    metric_versions: dict[NonEmptyStr, NonEmptyStr]
    config_hash: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_manifest(self) -> EvaluationRunManifest:
        if self.judge_score_values != [0.0, 0.5, 1.0]:
            raise ValueError("baseline-evaluation-v1 requires judge scores [0, 0.5, 1]")
        if self.secondary_judge_enabled:
            raise ValueError("secondary judge execution is not enabled in Step 5D")
        if self.source_kind is EvaluationSourceKind.BASELINE_SUITE:
            if self.suite_id is None or self.source_run_id is not None:
                raise ValueError("baseline evaluation requires only suite_id")
        elif self.suite_id is not None or self.source_run_id is None:
            raise ValueError("PMMC evaluation requires only source_run_id")
        return self


class EvaluationRunState(BaseSchema):
    evaluation_run_id: NonEmptyStr
    status: EvaluationRunStatus = EvaluationRunStatus.CREATED
    selected_count: NonNegativeInt
    target_count: PositiveInt
    succeeded_count: NonNegativeInt = 0
    failed_count: NonNegativeInt = 0
    started_at: UtcDateTime | None = None
    heartbeat_at: UtcDateTime | None = None
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> EvaluationRunState:
        expected = self.selected_count * self.target_count
        if self.succeeded_count + self.failed_count > expected:
            raise ValueError("evaluation result counts exceed the paired denominator")
        if self.status is EvaluationRunStatus.COMPLETED:
            if self.succeeded_count != expected or self.failed_count or self.completed_at is None:
                raise ValueError("completed evaluation runs require full successful coverage")
        if self.status is EvaluationRunStatus.FAILED and self.error_type is None:
            raise ValueError("failed evaluation runs require error_type")
        return self


class EvaluationTarget(BaseSchema):
    evaluation_run_id: NonEmptyStr
    source_kind: EvaluationSourceKind = EvaluationSourceKind.BASELINE_SUITE
    suite_id: NonEmptyStr | None = None
    run_id: NonEmptyStr
    method: BaselineMethod | Literal["pmmc"]
    snapshot_id: NonEmptyStr
    selected_count: NonNegativeInt
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> EvaluationTarget:
        if self.source_kind is EvaluationSourceKind.BASELINE_SUITE:
            if (
                self.suite_id is None or not isinstance(self.method, BaselineMethod)
                or not self.selected_count
            ):
                raise ValueError("baseline targets require suite_id and BaselineMethod")
        elif self.suite_id is not None or self.method != "pmmc":
            raise ValueError("PMMC targets require method='pmmc' and no suite_id")
        return self


class EvaluationJob(BaseSchema):
    evaluation_job_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    answer_id: NonEmptyStr
    answer_variant: NonEmptyStr = "primary"
    stage: EvaluationStage
    status: JobStatus = JobStatus.PENDING
    priority: NonNegativeInt = 0
    attempt_count: NonNegativeInt = 0
    max_attempts: PositiveInt = 2
    claimed_by: NonEmptyStr | None = None
    lease_expires_at: UtcDateTime | None = None
    available_at: UtcDateTime = Field(default_factory=utc_now)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_job(self) -> EvaluationJob:
        if self.status is JobStatus.RUNNING and (
            self.claimed_by is None or self.lease_expires_at is None
        ):
            raise ValueError("running evaluation jobs require a worker and lease")
        if self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            if self.finished_at is None:
                raise ValueError("terminal evaluation jobs require finished_at")
        if self.status is JobStatus.FAILED and self.error_type is None:
            raise ValueError("failed evaluation jobs require error_type")
        return self


class EvaluationJobAttempt(BaseSchema):
    attempt_id: NonEmptyStr
    evaluation_job_id: NonEmptyStr
    attempt_number: PositiveInt
    worker_id: NonEmptyStr
    status: JobStatus
    started_at: UtcDateTime
    finished_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)


class AnswerNormalization(BaseSchema):
    normalization_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr
    answer_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    normalizer_version: NonEmptyStr
    candidate_hash: NonEmptyStr
    reference_hash: NonEmptyStr
    normalized_candidate: str
    normalized_reference: str
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class ResolvedGoldEvidenceMember(BaseSchema):
    member_type: EvidenceMemberType
    member_id: NonEmptyStr
    derived: bool = False
    match_role: EvidenceMatchRole = EvidenceMatchRole.DISCRIMINATIVE
    metadata: JsonDict = Field(default_factory=dict)


class ResolvedGoldEvidenceGroup(BaseSchema):
    gold_group_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    granularity: EvidenceGranularity
    source_gold_id: NonEmptyStr
    members: list[ResolvedGoldEvidenceMember]
    resolution_policy_version: NonEmptyStr
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_members(self) -> ResolvedGoldEvidenceGroup:
        if not self.members:
            raise ValueError("resolved gold evidence groups require at least one member")
        keys = [f"{item.member_type.value}:{item.member_id}" for item in self.members]
        ensure_unique_strings(keys, "members")
        return self


class ContextEvidenceEvaluation(BaseSchema):
    context_evaluation_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr
    answer_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    context_source_kind: EvaluationContextSourceKind = (
        EvaluationContextSourceKind.BASELINE_CONTEXT
    )
    context_assembly_id: NonEmptyStr | None = None
    program_execution_id: NonEmptyStr | None = None
    granularity: EvidenceGranularity
    applicable: bool
    matched_gold_group_ids: list[NonEmptyStr] = Field(default_factory=list)
    gold_group_count: NonNegativeInt = 0
    policy_version: NonEmptyStr
    unavailable_reason: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_context(self) -> ContextEvidenceEvaluation:
        if self.context_source_kind is EvaluationContextSourceKind.BASELINE_CONTEXT:
            if self.context_assembly_id is None or self.program_execution_id is not None:
                raise ValueError(
                    "baseline context evaluation requires only context_assembly_id"
                )
            if self.context_assembly_id.startswith("pmmc-active-context:"):
                raise ValueError("synthetic PMMC context assembly IDs are forbidden")
        elif self.program_execution_id is None or self.context_assembly_id is not None:
            raise ValueError(
                "program context evaluation requires only program_execution_id"
            )
        ensure_unique_strings(self.matched_gold_group_ids, "matched_gold_group_ids")
        if len(self.matched_gold_group_ids) > self.gold_group_count:
            raise ValueError("matched gold groups cannot exceed gold_group_count")
        if self.applicable and self.gold_group_count == 0:
            raise ValueError("applicable context evaluation requires gold groups")
        if self.applicable and self.unavailable_reason is not None:
            raise ValueError("applicable context evaluation cannot have unavailable_reason")
        if not self.applicable and self.unavailable_reason is None:
            raise ValueError("non-applicable context evaluation requires unavailable_reason")
        return self


class MetricScore(BaseSchema):
    """One normalized metric value or an explicit not-applicable outcome."""

    metric_score_id: NonEmptyStr
    answer_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr | None = None
    metric_family: MetricFamily = MetricFamily.ANSWER
    metric_name: NonEmptyStr
    metric_version: NonEmptyStr
    k: PositiveInt | None = None
    scorer: NonEmptyStr | None = None
    applicable: bool = True
    value: Probability | None
    unavailable_reason: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_metric(self) -> MetricScore:
        if self.applicable and self.value is None:
            raise ValueError("applicable metrics require a value")
        if self.applicable and self.unavailable_reason is not None:
            raise ValueError("applicable metrics cannot have unavailable_reason")
        if not self.applicable and self.value is not None:
            raise ValueError("non-applicable metrics cannot have a value")
        if not self.applicable and self.unavailable_reason is None:
            raise ValueError("non-applicable metrics require unavailable_reason")
        return self


class FinalAnswerRecord(BaseSchema):
    """Method-produced final answer without evaluation gold data."""

    answer_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    answer_variant: NonEmptyStr = "primary"
    method: NonEmptyStr
    answer_model_alias: NonEmptyStr
    actual_model_id: NonEmptyStr | None = None
    evidence_ids_in_prompt: list[NonEmptyStr] = Field(default_factory=list)
    image_ids_in_prompt: list[NonEmptyStr] = Field(default_factory=list)
    final_answer: NonEmptyStr
    normalized_answer: NonEmptyStr | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    latency_ms: NonNegativeFloat | None = None
    model_call_id: NonEmptyStr | None = None
    langfuse_trace_id: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ids(self) -> FinalAnswerRecord:
        ensure_unique_strings(self.evidence_ids_in_prompt, "evidence_ids_in_prompt")
        ensure_unique_strings(self.image_ids_in_prompt, "image_ids_in_prompt")
        return self


class JudgeScore(BaseSchema):
    """Method-blind LLM judge result for one final answer."""

    judge_score_id: NonEmptyStr
    answer_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr | None = None
    judge_model_alias: NonEmptyStr
    judge_actual_model_id: NonEmptyStr | None = None
    judge_prompt_hash: NonEmptyStr | None = None
    model_call_id: NonEmptyStr | None = None
    rubric_version: NonEmptyStr
    repeat_index: NonNegativeInt = 0
    verdict: JudgeVerdict | None = None
    f1: Probability | None = None
    bleu1: Probability | None = None
    exact_match: bool | None = None
    llm_judge_score: Probability | None = None
    judge_reason: NonEmptyStr | None = None
    bad_case_category: BadCaseCategory = BadCaseCategory.UNKNOWN
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_judge(self) -> JudgeScore:
        if self.verdict is not None:
            expected = {
                JudgeVerdict.INCORRECT: 0.0,
                JudgeVerdict.PARTIALLY_CORRECT: 0.5,
                JudgeVerdict.CORRECT: 1.0,
            }[self.verdict]
            if self.llm_judge_score != expected:
                raise ValueError("judge verdict and score must use the fixed 0/0.5/1 rubric")
            if self.judge_reason is None:
                raise ValueError("structured judge outcomes require a reason")
        return self


class BadCaseLabel(BaseSchema):
    """Evaluation-only automatic or human failure annotation."""

    bad_case_label_id: NonEmptyStr
    answer_id: NonEmptyStr
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    evaluation_run_id: NonEmptyStr | None = None
    category: BadCaseCategory
    label_source: NonEmptyStr
    rule_version: NonEmptyStr | None = None
    is_primary: bool = False
    annotator_id: NonEmptyStr | None = None
    confidence: Probability | None = None
    rationale_asset_id: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)
