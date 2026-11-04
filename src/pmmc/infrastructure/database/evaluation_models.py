"""SQLAlchemy models introduced by unified baseline evaluation migration 0006."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import (
    BASELINE_SUITE_STATUS_VALUES,
    DATASET_VALUES,
    EVALUATION_RUN_STATUS_VALUES,
    EVALUATION_SCHEMA,
    EVALUATION_STAGE_VALUES,
    EVIDENCE_GRANULARITY_VALUES,
    EVIDENCE_MEMBER_TYPE_VALUES,
    JOB_STATUS_VALUES,
    RUNTIME_SCHEMA,
    sql_text_values,
)

JsonValue = dict[str, Any]


class BaselineSuiteModel(Base):
    __tablename__ = "baseline_suites"
    __table_args__ = (
        CheckConstraint(f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"),
        CheckConstraint("total_context_tokens > 0", name="total_context_positive"),
        CheckConstraint(
            "reserved_output_tokens > 0 AND reserved_output_tokens < total_context_tokens",
            name="reserved_output_valid",
        ),
        CheckConstraint("seed >= 0", name="seed_nonnegative"),
        UniqueConstraint("suite_id", "snapshot_id"),
        {"schema": RUNTIME_SCHEMA},
    )

    suite_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    selection_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    methods: Mapped[list[str]] = mapped_column(ARRAY(String(128)), nullable=False)
    answer_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    answer_prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    decoding_config: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    total_context_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    query_image_policy: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluation_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    paper_mode: Mapped[bool] = mapped_column(Boolean, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class BaselineSuiteStateModel(Base):
    __tablename__ = "baseline_suite_states"
    __table_args__ = (
        CheckConstraint(
            f"status IN {sql_text_values(BASELINE_SUITE_STATUS_VALUES)}", name="status_values"
        ),
        CheckConstraint("selected_count > 0", name="selected_count_positive"),
        CheckConstraint("completed_run_count >= 0", name="completed_count_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    suite_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.baseline_suites.suite_id"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_run_count: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class BaselineSuiteRunModel(Base):
    __tablename__ = "baseline_suite_runs"
    __table_args__ = (
        PrimaryKeyConstraint("suite_id", "method"),
        UniqueConstraint("suite_id", "run_id"),
        UniqueConstraint("suite_id", "method", "run_id"),
        {"schema": RUNTIME_SCHEMA},
    )

    suite_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.baseline_suites.suite_id"), nullable=False
    )
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class BaselineSuiteQueryModel(Base):
    __tablename__ = "baseline_suite_queries"
    __table_args__ = (
        PrimaryKeyConstraint("suite_id", "snapshot_id", "query_id"),
        UniqueConstraint("suite_id", "ordinal"),
        ForeignKeyConstraint(
            ["suite_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.baseline_suites.suite_id",
                f"{RUNTIME_SCHEMA}.baseline_suites.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_queries.query_id",
            ],
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    suite_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        CheckConstraint(
            "source_kind IN ('baseline_suite','pmmc_query_run')",
            name="source_kind_values",
        ),
        CheckConstraint(
            "suite_id IS NOT NULL OR legacy_source_run_id IS NOT NULL",
            name="suite_or_legacy_source_present",
        ),
        UniqueConstraint("suite_id", "config_hash"),
        {"schema": EVALUATION_SCHEMA},
    )

    evaluation_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="baseline_suite"
    )
    suite_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.baseline_suites.suite_id")
    )
    legacy_source_run_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id")
    )
    protocol_version: Mapped[str] = mapped_column(String(128), nullable=False)
    normalizer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    bad_case_rule_version: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    judge_prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_rubric_version: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_repeat_count: Mapped[int] = mapped_column(Integer, nullable=False)
    judge_score_values: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    secondary_judge_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metric_versions: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class EvaluationRunStateModel(Base):
    __tablename__ = "evaluation_run_states"
    __table_args__ = (
        CheckConstraint(
            f"status IN {sql_text_values(EVALUATION_RUN_STATUS_VALUES)}", name="status_values"
        ),
        CheckConstraint("selected_count >= 0 AND target_count > 0", name="denominator_positive"),
        CheckConstraint(
            "succeeded_count >= 0 AND failed_count >= 0 AND "
            "succeeded_count + failed_count <= selected_count * target_count",
            name="counts_valid",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    target_count: Mapped[int] = mapped_column(Integer, nullable=False)
    succeeded_count: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class EvaluationTargetModel(Base):
    __tablename__ = "evaluation_targets"
    __table_args__ = (
        CheckConstraint(
            "source_kind IN ('baseline_suite','pmmc_query_run')",
            name="source_kind_values",
        ),
        PrimaryKeyConstraint("evaluation_run_id", "run_id"),
        UniqueConstraint("evaluation_run_id", "method"),
        UniqueConstraint("evaluation_run_id", "run_id", "snapshot_id"),
        ForeignKeyConstraint(
            ["suite_id", "method", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.baseline_suite_runs.suite_id",
                f"{RUNTIME_SCHEMA}.baseline_suite_runs.method",
                f"{RUNTIME_SCHEMA}.baseline_suite_runs.run_id",
            ],
        ),
        CheckConstraint("selected_count >= 0", name="selected_count_positive"),
        {"schema": EVALUATION_SCHEMA},
    )

    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    source_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="baseline_suite"
    )
    suite_id: Mapped[str | None] = mapped_column(String(128))
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class EvaluationJobModel(Base):
    __tablename__ = "evaluation_jobs"
    __table_args__ = (
        UniqueConstraint("evaluation_job_id", "evaluation_run_id"),
        UniqueConstraint(
            "evaluation_run_id",
            "run_id",
            "query_id",
            "answer_variant",
            "stage",
        ),
        ForeignKeyConstraint(
            ["evaluation_run_id", "run_id", "snapshot_id"],
            [
                f"{EVALUATION_SCHEMA}.evaluation_targets.evaluation_run_id",
                f"{EVALUATION_SCHEMA}.evaluation_targets.run_id",
                f"{EVALUATION_SCHEMA}.evaluation_targets.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["answer_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        CheckConstraint(
            f"stage IN {sql_text_values(EVALUATION_STAGE_VALUES)}",
            name="stage_values",
        ),
        CheckConstraint(f"status IN {sql_text_values(JOB_STATUS_VALUES)}", name="status_values"),
        CheckConstraint("priority >= 0", name="priority_nonnegative"),
        CheckConstraint("attempt_count >= 0 AND max_attempts > 0", name="attempts_valid"),
        CheckConstraint(
            "status != 'running' OR (claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="running_lease_present",
        ),
        CheckConstraint(
            "status NOT IN ('succeeded','failed','cancelled') OR finished_at IS NOT NULL",
            name="terminal_finished_present",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    evaluation_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    answer_variant: Mapped[str] = mapped_column(String(128), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    claimed_by: Mapped[str | None] = mapped_column(String(256))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class EvaluationJobAttemptModel(Base):
    __tablename__ = "evaluation_job_attempts"
    __table_args__ = (
        UniqueConstraint("evaluation_job_id", "attempt_number"),
        CheckConstraint(f"status IN {sql_text_values(JOB_STATUS_VALUES)}", name="status_values"),
        CheckConstraint("attempt_number > 0", name="attempt_positive"),
        {"schema": EVALUATION_SCHEMA},
    )

    attempt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_job_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_jobs.evaluation_job_id"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_id: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class AnswerNormalizationModel(Base):
    __tablename__ = "answer_normalizations"
    __table_args__ = (
        UniqueConstraint("evaluation_run_id", "answer_id", "normalizer_version"),
        ForeignKeyConstraint(
            ["answer_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    normalization_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    normalizer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized_candidate: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_reference: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class GoldEvidenceGroupModel(Base):
    __tablename__ = "gold_evidence_groups"
    __table_args__ = (
        UniqueConstraint("evaluation_run_id", "query_id", "source_gold_id"),
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.query_gold.snapshot_id",
                f"{EVALUATION_SCHEMA}.query_gold.query_id",
            ],
        ),
        CheckConstraint(
            f"granularity IN {sql_text_values(EVIDENCE_GRANULARITY_VALUES)}",
            name="granularity_values",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    gold_group_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    granularity: Mapped[str] = mapped_column(String(32), nullable=False)
    source_gold_id: Mapped[str] = mapped_column(String(256), nullable=False)
    resolution_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class GoldEvidenceGroupMemberModel(Base):
    __tablename__ = "gold_evidence_group_members"
    __table_args__ = (
        PrimaryKeyConstraint("gold_group_id", "member_type", "member_id"),
        CheckConstraint(
            f"member_type IN {sql_text_values(EVIDENCE_MEMBER_TYPE_VALUES)}",
            name="member_type_values",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    gold_group_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.gold_evidence_groups.gold_group_id"), nullable=False
    )
    member_type: Mapped[str] = mapped_column(String(32), nullable=False)
    member_id: Mapped[str] = mapped_column(String(256), nullable=False)
    derived: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class ContextEvaluationModel(Base):
    __tablename__ = "context_evaluations"
    __table_args__ = (
        UniqueConstraint("evaluation_run_id", "answer_id", "granularity"),
        CheckConstraint(
            f"granularity IN {sql_text_values(EVIDENCE_GRANULARITY_VALUES)}",
            name="granularity_values",
        ),
        CheckConstraint("gold_group_count >= 0", name="gold_count_nonnegative"),
        CheckConstraint(
            "(applicable AND gold_group_count > 0 AND unavailable_reason IS NULL) OR "
            "(NOT applicable AND unavailable_reason IS NOT NULL)",
            name="applicability_consistent",
        ),
        CheckConstraint(
            "(context_source_kind = 'baseline_context' "
            "AND context_assembly_id IS NOT NULL AND program_execution_id IS NULL) OR "
            "(context_source_kind = 'program_execution' "
            "AND context_assembly_id IS NULL AND program_execution_id IS NOT NULL)",
            name="context_source_exactly_one",
        ),
        ForeignKeyConstraint(
            ["answer_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    context_evaluation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    context_source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    context_assembly_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.context_assemblies.context_assembly_id")
    )
    program_execution_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.program_executions.execution_id")
    )
    granularity: Mapped[str] = mapped_column(String(32), nullable=False)
    applicable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gold_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    unavailable_reason: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class ContextEvaluationMatchModel(Base):
    __tablename__ = "context_evaluation_matches"
    __table_args__ = (
        PrimaryKeyConstraint("context_evaluation_id", "gold_group_id"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        UniqueConstraint("context_evaluation_id", "ordinal"),
        {"schema": EVALUATION_SCHEMA},
    )

    context_evaluation_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.context_evaluations.context_evaluation_id"),
        nullable=False,
    )
    gold_group_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.gold_evidence_groups.gold_group_id"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
