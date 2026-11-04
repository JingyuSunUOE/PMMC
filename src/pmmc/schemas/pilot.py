"""Portable contracts for Questioner quality pilots and human approval."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    QUESTIONER_PILOT_SCHEMA_VERSION,
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Probability,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)


class PartitionRole(StrEnum):
    DEVELOPMENT = "development"
    PAPER_TEST = "paper_test"


class QuestionerPilotStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewTaskKind(StrEnum):
    CANDIDATE_INTRINSIC = "candidate_intrinsic"
    QUERY_COVERAGE = "query_coverage"


class ReviewRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class ReviewRating(StrEnum):
    NO = "no"
    PARTIAL = "partial"
    YES = "yes"


class CoverageFailureRoot(StrEnum):
    QUESTIONER = "questioner"
    UNIT_BUILDER = "unit_builder"
    EVIDENCE_LINEAGE = "evidence_lineage"
    GOLD_RESOLUTION = "gold_resolution"
    MATCHER = "matcher"
    NO_ERROR = "no_error"
    UNRESOLVABLE = "unresolvable"


class PilotDecisionValue(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class BenchmarkPartitionSpec(BaseSchema):
    protocol_version: NonEmptyStr = "conversation-stratified-five-fold-v1"
    fold_count: PositiveInt = 5
    development_fold: NonNegativeInt = 0
    seed: NonNegativeInt = 0
    eligibility_policy_version: NonEmptyStr = "pmmc-aaai27-v1"

    @model_validator(mode="after")
    def _validate_fold(self) -> BenchmarkPartitionSpec:
        if self.protocol_version not in {
            "conversation-stratified-five-fold-v1",
            "scenario-balanced-five-fold-v2",
            "scenario-balanced-five-fold-v3",
        }:
            raise ValueError("unsupported benchmark partition protocol")
        if self.fold_count < 2:
            raise ValueError("partition requires at least two folds")
        if self.development_fold >= self.fold_count:
            raise ValueError("development_fold must be less than fold_count")
        return self


class BenchmarkPartitionAssignment(BaseSchema):
    partition_id: NonEmptyStr
    dataset: DatasetName
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    conversation_id: NonEmptyStr
    scenario_id: NonEmptyStr | None = None
    source_category: NonEmptyStr
    has_query_image: bool
    fold: NonNegativeInt
    role: PartitionRole
    ordinal: NonNegativeInt


class DatasetPartition(BaseSchema):
    dataset: DatasetName
    snapshot_id: NonEmptyStr
    snapshot_content_hash: NonEmptyStr
    query_count: PositiveInt
    conversation_count: PositiveInt
    development_query_count: NonNegativeInt
    paper_test_query_count: NonNegativeInt
    assignment_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_counts(self) -> DatasetPartition:
        if self.development_query_count + self.paper_test_query_count != self.query_count:
            raise ValueError("dataset partition query counts must reconcile")
        return self


class BenchmarkPartitionManifest(BaseSchema):
    partition_id: NonEmptyStr
    schema_version: NonEmptyStr = QUESTIONER_PILOT_SCHEMA_VERSION
    spec: BenchmarkPartitionSpec
    datasets: list[DatasetPartition]
    assignment_count: PositiveInt
    assignments_sha256: NonEmptyStr
    balance_report_sha256: NonEmptyStr
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_datasets(self) -> BenchmarkPartitionManifest:
        if not self.datasets:
            raise ValueError("partition manifest requires at least one dataset")
        if len(self.datasets) != len({item.dataset for item in self.datasets}):
            raise ValueError("partition manifest datasets cannot repeat")
        if sum(item.query_count for item in self.datasets) != self.assignment_count:
            raise ValueError("partition assignment count does not match dataset counts")
        return self


class QuestionerPilotSample(BaseSchema):
    sample_id: NonEmptyStr
    partition_id: NonEmptyStr
    dataset: DatasetName
    role: PartitionRole = PartitionRole.DEVELOPMENT
    query_ids: list[NonEmptyStr]
    conversation_ids: list[NonEmptyStr]
    stratum_counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    scenario_counts: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)
    selection_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_sample(self) -> QuestionerPilotSample:
        if not self.query_ids or not self.conversation_ids:
            raise ValueError("pilot sample requires queries and conversations")
        ensure_unique_strings(self.query_ids, "query_ids")
        ensure_unique_strings(self.conversation_ids, "conversation_ids")
        return self


class QuestionerPilotArtifactRef(BaseSchema):
    role: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    size_bytes: NonNegativeInt
    private: bool = False

    @model_validator(mode="after")
    def _validate_path(self) -> QuestionerPilotArtifactRef:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("pilot artifact path must stay below the bundle root")
        return self


class QuestionerPilotManifest(BaseSchema):
    pilot_id: NonEmptyStr
    schema_version: NonEmptyStr = QUESTIONER_PILOT_SCHEMA_VERSION
    protocol_version: NonEmptyStr
    config_hash: NonEmptyStr
    partition_id: NonEmptyStr
    samples: list[QuestionerPilotSample]
    status: QuestionerPilotStatus = QuestionerPilotStatus.PLANNED
    source_runtime_export_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    source_coverage_export_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    artifacts: list[QuestionerPilotArtifactRef] = Field(default_factory=list)
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_samples(self) -> QuestionerPilotManifest:
        if not self.samples:
            raise ValueError("pilot manifest requires at least one sample")
        if len(self.samples) != len({item.dataset for item in self.samples}):
            raise ValueError("pilot samples cannot repeat a dataset")
        if any(item.partition_id != self.partition_id for item in self.samples):
            raise ValueError("pilot samples must use the manifest partition")
        return self


class QuestionerPilotAuditManifest(BaseSchema):
    """Signed v3 audit denominator used for explicit research approval."""

    pilot_id: NonEmptyStr
    protocol_version: NonEmptyStr
    partition_id: NonEmptyStr
    candidate_audit_count: NonNegativeInt
    coverage_audit_count: NonNegativeInt
    candidate_audit_sha256: NonEmptyStr
    coverage_audit_sha256: NonEmptyStr
    automatic_summary_sha256: NonEmptyStr
    source_runtime_manifest_hashes: dict[NonEmptyStr, NonEmptyStr]
    source_coverage_manifest_hashes: dict[NonEmptyStr, NonEmptyStr]
    automatic_integrity_passed: bool
    integrity_issue_types: list[NonEmptyStr] = Field(default_factory=list)
    severe_terminal_error_count: NonNegativeInt = 0
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_v3_audit(self) -> QuestionerPilotAuditManifest:
        if self.protocol_version != "questioner-quality-pilot-v3":
            raise ValueError("Questioner audit manifests require pilot v3")
        if self.candidate_audit_count != 120 or self.coverage_audit_count != 40:
            raise ValueError("Questioner v3 audit requires 120 candidate and 40 coverage cards")
        if self.automatic_integrity_passed == bool(self.integrity_issue_types):
            raise ValueError("Questioner audit integrity status and issues disagree")
        if set(self.source_runtime_manifest_hashes) != set(
            self.source_coverage_manifest_hashes
        ):
            raise ValueError("Questioner audit source datasets do not match")
        return self


class QuestionerPilotState(BaseSchema):
    pilot_id: NonEmptyStr
    status: QuestionerPilotStatus
    updated_at: UtcDateTime = Field(default_factory=utc_now)
    review_id: NonEmptyStr | None = None
    decision_hash: NonEmptyStr | None = None
    freeze_id: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> QuestionerPilotState:
        if self.status in {
            QuestionerPilotStatus.APPROVED,
            QuestionerPilotStatus.REJECTED,
        } and self.decision_hash is None:
            raise ValueError("terminal pilot states require a decision hash")
        if self.status is QuestionerPilotStatus.APPROVED and self.freeze_id is None:
            raise ValueError("approved pilot states require a freeze ID")
        if self.status is not QuestionerPilotStatus.APPROVED and self.freeze_id is not None:
            raise ValueError("only an approved pilot state can reference a freeze")
        return self


class QuestionerReviewTask(BaseSchema):
    task_id: NonEmptyStr
    pilot_id: NonEmptyStr
    kind: ReviewTaskKind
    dataset: DatasetName
    bucket: NonEmptyStr
    source_id: NonEmptyStr
    qbank_id: NonEmptyStr | None = None
    query_id: NonEmptyStr | None = None
    task_hash: NonEmptyStr
    payload: JsonDict
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> QuestionerReviewTask:
        if self.kind is ReviewTaskKind.CANDIDATE_INTRINSIC:
            if self.qbank_id is None or self.query_id is not None:
                raise ValueError("candidate review tasks require only qbank_id")
        elif self.query_id is None or self.qbank_id is not None:
            raise ValueError("coverage review tasks require only query_id")
        return self


class ReviewAssignment(BaseSchema):
    assignment_id: NonEmptyStr
    pilot_id: NonEmptyStr
    role: ReviewRole
    reviewer_slot: NonEmptyStr
    task_ids: list[NonEmptyStr]
    assignment_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_tasks(self) -> ReviewAssignment:
        if not self.task_ids:
            raise ValueError("review assignment requires tasks")
        ensure_unique_strings(self.task_ids, "task_ids")
        return self


class CandidateQualityAnnotation(BaseSchema):
    annotation_id: NonEmptyStr
    task_id: NonEmptyStr
    task_hash: NonEmptyStr
    reviewer_id: NonEmptyStr
    role: ReviewRole
    answerability: ReviewRating
    expected_answer_support: ReviewRating
    evidence_sufficiency: ReviewRating
    future_usefulness: ReviewRating
    taxonomy_correctness: ReviewRating
    modality_correctness: ReviewRating
    visual_necessity: ReviewRating | None = None
    multi_hop_validity: ReviewRating | None = None
    template_like: bool = False
    duplicate: bool = False
    over_trivial: bool = False
    leakage: bool = False
    notes: str | None = None
    annotation_hash: NonEmptyStr


class CoverageQualityAnnotation(BaseSchema):
    annotation_id: NonEmptyStr
    task_id: NonEmptyStr
    task_hash: NonEmptyStr
    reviewer_id: NonEmptyStr
    role: ReviewRole
    semantic_coverage: ReviewRating
    automatic_classification_correct: bool | None = None
    failure_root: CoverageFailureRoot
    leakage: bool = False
    notes: str | None = None
    annotation_hash: NonEmptyStr


class ReviewAdjudication(BaseSchema):
    adjudication_id: NonEmptyStr
    task_id: NonEmptyStr
    primary_annotation_hash: NonEmptyStr
    secondary_annotation_hash: NonEmptyStr
    resolved_annotation_hash: NonEmptyStr
    resolved_annotation: JsonDict
    adjudicator_id: NonEmptyStr
    rationale: NonEmptyStr
    content_hash: NonEmptyStr


class ReviewAgreementReport(BaseSchema):
    pilot_id: NonEmptyStr
    overlap_count: NonNegativeInt
    raw_agreement: dict[NonEmptyStr, Probability | None] = Field(default_factory=dict)
    cohen_kappa: dict[NonEmptyStr, float | None] = Field(default_factory=dict)
    weighted_kappa: dict[NonEmptyStr, float | None] = Field(default_factory=dict)
    disagreement_task_ids: list[NonEmptyStr] = Field(default_factory=list)
    adjudicated_task_ids: list[NonEmptyStr] = Field(default_factory=list)
    content_hash: NonEmptyStr


class QuestionerPilotDecision(BaseSchema):
    pilot_id: NonEmptyStr
    decision: PilotDecisionValue
    reviewer_id: NonEmptyStr
    rationale: NonEmptyStr
    review_manifest_hash: NonEmptyStr | None = None
    audit_manifest_hash: NonEmptyStr | None = None
    decided_at: UtcDateTime = Field(default_factory=utc_now)
    content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_evidence(self) -> QuestionerPilotDecision:
        if (self.review_manifest_hash is None) == (self.audit_manifest_hash is None):
            raise ValueError("pilot decision requires exactly one approval evidence hash")
        return self


class QuestionerFreezeManifest(BaseSchema):
    freeze_id: NonEmptyStr
    pilot_id: NonEmptyStr
    decision_hash: NonEmptyStr
    partition_id: NonEmptyStr
    pilot_sample_ids: list[NonEmptyStr]
    questioner_config_hash: NonEmptyStr
    prompt_hash: NonEmptyStr
    response_schema_hash: NonEmptyStr
    renderer_version: NonEmptyStr
    parser_version: NonEmptyStr
    unit_builder_version: NonEmptyStr
    gate_version: NonEmptyStr
    dedupe_version: NonEmptyStr
    family_dedupe_version: NonEmptyStr | None = None
    matcher_version: NonEmptyStr
    model_commits: dict[NonEmptyStr, NonEmptyStr]
    runtime_export_ids: dict[NonEmptyStr, NonEmptyStr]
    coverage_export_ids: dict[NonEmptyStr, NonEmptyStr]
    questioner_protocol_version: NonEmptyStr | None = None
    prompt_version: NonEmptyStr | None = None
    review_hash: NonEmptyStr | None = None
    agreement_report_hash: NonEmptyStr | None = None
    audit_manifest_hash: NonEmptyStr | None = None
    opportunity_builder_version: NonEmptyStr | None = None
    relation_selector_version: NonEmptyStr | None = None
    opportunity_builder_hash: NonEmptyStr | None = None
    verifier_prompt_version: NonEmptyStr | None = None
    verifier_prompt_hash: NonEmptyStr | None = None
    verifier_schema_hash: NonEmptyStr | None = None
    verifier_protocol_hash: NonEmptyStr | None = None
    approved_by: NonEmptyStr
    approved_at: UtcDateTime
    content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_freeze(self) -> QuestionerFreezeManifest:
        ensure_unique_strings(self.pilot_sample_ids, "pilot_sample_ids")
        if not self.model_commits:
            raise ValueError("Questioner freeze requires pinned model commits")
        legacy_review = self.review_hash is not None or self.agreement_report_hash is not None
        if legacy_review and (
            self.review_hash is None or self.agreement_report_hash is None
        ):
            raise ValueError("legacy freeze requires review and agreement hashes")
        if legacy_review == (self.audit_manifest_hash is not None):
            raise ValueError("freeze requires legacy review or v3 audit evidence")
        v3_fields = (
            self.opportunity_builder_version,
            self.relation_selector_version,
            self.opportunity_builder_hash,
            self.verifier_prompt_version,
            self.verifier_prompt_hash,
            self.verifier_schema_hash,
            self.verifier_protocol_hash,
        )
        if self.audit_manifest_hash is not None and any(value is None for value in v3_fields):
            raise ValueError("v3 freeze requires complete opportunity/verifier identity")
        if self.audit_manifest_hash is not None:
            protocol_versions = {
                "pmmc-questioner-v3": {
                    "prompt_version": "pmmc-questioner-v3",
                    "renderer_version": "questioner-multimodal-renderer-v3",
                    "parser_version": "questioner-json-parser-v3",
                    "gate_version": "questioner-hard-gate-v3",
                    "opportunity_builder_version": "questioner-opportunity-builder-v1",
                    "verifier_prompt_version": "pmmc-questioner-verifier-v1",
                },
                "pmmc-questioner-v3.1": {
                    "prompt_version": "pmmc-questioner-v3.1",
                    "renderer_version": "questioner-safe-alias-renderer-v3.1",
                    "parser_version": "questioner-json-parser-v3.1",
                    "gate_version": "questioner-static-semantic-gate-v3.1",
                    "opportunity_builder_version": "questioner-opportunity-builder-v2",
                    "verifier_prompt_version": "pmmc-questioner-verifier-v2",
                },
                "pmmc-questioner-v3.2": {
                    "prompt_version": "pmmc-questioner-v3.2",
                    "renderer_version": "questioner-safe-alias-renderer-v3.2",
                    "parser_version": "questioner-json-parser-v3.2",
                    "gate_version": "questioner-static-semantic-gate-v3.2",
                    "opportunity_builder_version": "questioner-opportunity-builder-v3",
                    "verifier_prompt_version": "pmmc-questioner-verifier-v3",
                },
                "pmmc-questioner-v3.3": {
                    "prompt_version": "pmmc-questioner-v3.3",
                    "renderer_version": "questioner-safe-alias-renderer-v3.3",
                    "parser_version": "questioner-json-parser-v3.3",
                    "gate_version": "questioner-static-semantic-gate-v3.3",
                    "opportunity_builder_version": "questioner-opportunity-builder-v4",
                    "verifier_prompt_version": "pmmc-questioner-verifier-v4",
                },
                "pmmc-questioner-v3.4": {
                    "prompt_version": "pmmc-questioner-v3.4",
                    "renderer_version": "questioner-safe-alias-renderer-v3.4",
                    "parser_version": "questioner-json-parser-v3.4",
                    "gate_version": "questioner-static-semantic-gate-v3.4",
                    "opportunity_builder_version": "questioner-opportunity-builder-v5",
                    "verifier_prompt_version": "pmmc-questioner-verifier-v5",
                },
                "pmmc-questioner-v3.5": {
                    "prompt_version": "pmmc-questioner-v3.5",
                    "renderer_version": "questioner-safe-alias-renderer-v3.5",
                    "parser_version": "questioner-json-parser-v3.5",
                    "gate_version": "questioner-static-semantic-gate-v3.5",
                    "opportunity_builder_version": "questioner-opportunity-builder-v6",
                    "verifier_prompt_version": "pmmc-questioner-verifier-v6",
                },
            }
            protocol = self.questioner_protocol_version
            if protocol not in protocol_versions:
                raise ValueError("freeze has an unsupported Questioner v3 protocol")
            expected_versions = {
                "questioner_protocol_version": protocol,
                **protocol_versions[protocol],
                "dedupe_version": (
                    "questioner-exact-dedupe-v2"
                    if protocol
                    in {
                        "pmmc-questioner-v3.3",
                        "pmmc-questioner-v3.4",
                        "pmmc-questioner-v3.5",
                    }
                    else "questioner-exact-dedupe-v1"
                ),
                "matcher_version": "qbank-hybrid-rrf-v1",
                "unit_builder_version": (
                    "session-opportunity-candidate-set-v6"
                    if protocol == "pmmc-questioner-v3.5"
                    else (
                        "session-opportunity-candidate-set-v5"
                        if protocol == "pmmc-questioner-v3.4"
                        else (
                            "session-opportunity-candidate-set-v4"
                            if protocol == "pmmc-questioner-v3.3"
                            else (
                                "session-opportunity-candidate-set-v3"
                                if protocol == "pmmc-questioner-v3.2"
                                else "session-opportunity-candidate-set-v2"
                            )
                        )
                    )
                ),
                "relation_selector_version": (
                    "postgres-simple-fts-validated-v5"
                    if protocol == "pmmc-questioner-v3.5"
                    else (
                        "postgres-simple-fts-validated-v4"
                        if protocol == "pmmc-questioner-v3.4"
                        else (
                            "postgres-simple-fts-validated-v3"
                            if protocol == "pmmc-questioner-v3.3"
                            else (
                                "postgres-simple-fts-validated-v2"
                                if protocol == "pmmc-questioner-v3.2"
                                else "postgres-simple-fts-v1"
                            )
                        )
                    )
                ),
            }
            for field_name, expected in expected_versions.items():
                if getattr(self, field_name) != expected:
                    raise ValueError(
                        f"v3 freeze has invalid {field_name}: expected {expected}"
                    )
            if protocol in {
                "pmmc-questioner-v3.3",
                "pmmc-questioner-v3.4",
                "pmmc-questioner-v3.5",
            }:
                expected_family = (
                    "qbank-family-dedupe-v4"
                    if protocol == "pmmc-questioner-v3.5"
                    else (
                        "qbank-family-dedupe-v3"
                        if protocol == "pmmc-questioner-v3.4"
                        else "qbank-family-dedupe-v2"
                    )
                )
                if self.family_dedupe_version != expected_family:
                    raise ValueError(
                        f"{protocol} freeze has invalid family dedupe"
                    )
            elif self.family_dedupe_version is not None:
                raise ValueError("historical Questioner freeze cannot claim family dedupe")
            hash_fields = (
                "questioner_config_hash",
                "prompt_hash",
                "response_schema_hash",
                "audit_manifest_hash",
                "opportunity_builder_hash",
                "verifier_prompt_hash",
                "verifier_schema_hash",
                "verifier_protocol_hash",
            )
            for field_name in hash_fields:
                value = getattr(self, field_name)
                if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    raise ValueError(f"v3 freeze has invalid SHA256 field: {field_name}")
            if any(
                re.fullmatch(r"[0-9a-f]{40}", revision) is None
                for revision in self.model_commits.values()
            ):
                raise ValueError("v3 freeze requires 40-character model commits")
        return self
