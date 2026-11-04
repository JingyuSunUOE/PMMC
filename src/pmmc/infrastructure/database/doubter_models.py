"""SQLAlchemy models for Step 6D Doubter and verified-QBank facts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import RUNTIME_SCHEMA

JsonValue = dict[str, Any]


class R0InputManifestModel(Base):
    __tablename__ = "r0_input_manifests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_runs.run_id",
                f"{RUNTIME_SCHEMA}.compiler_runs.snapshot_id",
            ],
        ),
        UniqueConstraint("run_id"),
        UniqueConstraint("r0_manifest_id", "run_id"),
        CheckConstraint(
            "candidate_count >= 0 AND answerable_candidate_count >= 0 "
            "AND control_candidate_count >= 0 AND program_count >= 0 "
            "AND execution_count >= 0 AND terminal_outcome_count >= 0",
            name="counts_nonnegative",
        ),
        CheckConstraint(
            "candidate_count = answerable_candidate_count + control_candidate_count",
            name="route_counts_reconcile",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    r0_manifest_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    answerable_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    control_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    program_count: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_count: Mapped[int] = mapped_column(Integer, nullable=False)
    terminal_outcome_count: Mapped[int] = mapped_column(Integer, nullable=False)
    questioner_identity_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    planner_identity_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    operator_registry_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    memory_index_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    query_vector_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    facts_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    hard_gate_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    issue_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    protocol_integrity_issues: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    candidate_scientific_rejections: Mapped[JsonValue] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    coverage_zero_units: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    downstream_viability: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class CandidateStageJobModel(Base):
    __tablename__ = "candidate_stage_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        UniqueConstraint("run_id", "qbank_id", "stage", "round_id"),
        CheckConstraint(
            "stage IN ('doubter','planner_revision','dry_run','decision')",
            name="stage_values",
        ),
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="status_values",
        ),
        CheckConstraint("round_id BETWEEN 0 AND 3", name="round_values"),
        CheckConstraint("attempt_count >= 0 AND max_attempts > 0", name="attempt_counts_valid"),
        {"schema": RUNTIME_SCHEMA},
    )

    candidate_stage_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    claimed_by: Mapped[str | None] = mapped_column(String(256))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class DoubterAssessmentModel(Base):
    __tablename__ = "doubter_assessments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["feedback_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.doubter_feedback.feedback_id",
                f"{RUNTIME_SCHEMA}.doubter_feedback.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["producer_model_call_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
            ],
        ),
        UniqueConstraint("run_id", "qbank_id", "round_id"),
        UniqueConstraint("assessment_id", "run_id"),
        CheckConstraint("round_id BETWEEN 0 AND 3", name="round_values"),
        CheckConstraint(
            "action IN ('pass','revise_program','reject_candidate')",
            name="action_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    assessment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    feedback_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    execution_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.program_executions.execution_id"), nullable=False
    )
    producer_model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    issue_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_schema_conformant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class DoubterAssessmentDimensionModel(Base):
    __tablename__ = "doubter_assessment_dimensions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["assessment_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.doubter_assessments.assessment_id",
                f"{RUNTIME_SCHEMA}.doubter_assessments.run_id",
            ],
        ),
        PrimaryKeyConstraint("assessment_id", "dimension"),
        CheckConstraint(
            "dimension IN ('candidate_supported','answer_entailed','path_sufficient',"
            "'modality_sufficient','raw_image_sufficient','context_sufficient',"
            "'multi_hop_sufficient','temporal_consistent','non_redundant',"
            "'refusal_appropriate')",
            name="dimension_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    assessment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[str] = mapped_column(String(64), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False, default="pass")


class DoubterDirectiveModel(Base):
    __tablename__ = "doubter_directives"
    __table_args__ = (
        ForeignKeyConstraint(
            ["feedback_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.doubter_feedback.feedback_id",
                f"{RUNTIME_SCHEMA}.doubter_feedback.run_id",
            ],
        ),
        UniqueConstraint("feedback_id", "ordinal"),
        CheckConstraint("round_id BETWEEN 1 AND 3", name="round_values"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            "directive IN ('add_text_retrieval','add_visual_retrieval',"
            "'add_image_text_pair','add_context','add_temporal',"
            "'add_raw_image_inspection','increase_top_k',"
            "'remove_redundant_retrieval')",
            name="directive_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    directive_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    feedback_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    directive: Mapped[str] = mapped_column(String(64), nullable=False)
    top_k_tier: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PlannerRevisionLineageModel(Base):
    __tablename__ = "planner_revision_lineage"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["revised_plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["source_feedback_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.doubter_feedback.feedback_id",
                f"{RUNTIME_SCHEMA}.doubter_feedback.run_id",
            ],
        ),
        UniqueConstraint("run_id", "qbank_id", "target_round_id"),
        CheckConstraint("target_round_id BETWEEN 1 AND 3", name="round_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    revision_lineage_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    revised_plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_feedback_id: Mapped[str] = mapped_column(String(128), nullable=False)
    target_round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    producer_model_call_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id"), nullable=False
    )
    directive_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PlannerRevisionPatchModel(Base):
    """Deterministic Step 7B patch application and no-op rejection fact."""

    __tablename__ = "planner_revision_patches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        UniqueConstraint("run_id", "qbank_id", "target_round_id"),
        CheckConstraint("target_round_id BETWEEN 1 AND 3", name="round_values"),
        CheckConstraint(
            "(applied AND after_program_hash IS NOT NULL AND rejection_code IS NULL "
            "AND before_program_hash <> after_program_hash) OR "
            "(NOT applied AND after_program_hash IS NULL AND rejection_code IS NOT NULL)",
            name="application_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    patch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    target_round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    directive_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    before_strategy_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    after_strategy_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    before_program_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    after_program_hash: Mapped[str | None] = mapped_column(String(128))
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejection_code: Mapped[str | None] = mapped_column(String(64))
    capability_added: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    capability_removed: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankRoundProjectionManifestModel(Base):
    __tablename__ = "qbank_round_projection_manifests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_r0_manifest_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.r0_input_manifests.r0_manifest_id",
                f"{RUNTIME_SCHEMA}.r0_input_manifests.run_id",
            ],
        ),
        UniqueConstraint("run_id", "round_id"),
        UniqueConstraint("projection_id", "run_id"),
        CheckConstraint("round_id BETWEEN 0 AND 3", name="round_values"),
        CheckConstraint(
            "member_count = accepted_count + rejected_count + pending_count",
            name="counts_reconcile",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    projection_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_r0_manifest_id: Mapped[str] = mapped_column(String(128), nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    accepted_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    pending_count: Mapped[int] = mapped_column(Integer, nullable=False)
    members_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankRoundProjectionMemberModel(Base):
    __tablename__ = "qbank_round_projection_members"
    __table_args__ = (
        ForeignKeyConstraint(
            ["projection_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_round_projection_manifests.projection_id",
                f"{RUNTIME_SCHEMA}.qbank_round_projection_manifests.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("projection_id", "qbank_id"),
        UniqueConstraint("projection_member_id", "run_id"),
        CheckConstraint("route_class IN ('answerable','control')", name="route_class_values"),
        CheckConstraint(
            "outcome IN ('accepted','rejected','pending','max_rounds_exhausted','carried_forward')",
            name="outcome_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    projection_member_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    projection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    route_class: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    effective_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.memory_programs.plan_id")
    )
    effective_feedback_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.doubter_feedback.feedback_id")
    )
    passed_at_round: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RoutingProjectionManifestModel(Base):
    __tablename__ = "routing_projection_manifests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_projection_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_round_projection_manifests.projection_id",
                f"{RUNTIME_SCHEMA}.qbank_round_projection_manifests.run_id",
            ],
        ),
        UniqueConstraint("run_id"),
        UniqueConstraint("routing_projection_id", "run_id"),
        CheckConstraint("active_round = 2 AND max_round = 3", name="rounds_frozen"),
        CheckConstraint(
            "member_count = answerable_count + control_count",
            name="route_counts_reconcile",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    routing_projection_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    active_round: Mapped[int] = mapped_column(Integer, nullable=False)
    max_round: Mapped[int] = mapped_column(Integer, nullable=False)
    source_projection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    decision_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    lexical_collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    text_collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    multimodal_collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    answerable_count: Mapped[int] = mapped_column(Integer, nullable=False)
    control_count: Mapped[int] = mapped_column(Integer, nullable=False)
    members_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RoutingProjectionMemberModel(Base):
    __tablename__ = "routing_projection_members"
    __table_args__ = (
        ForeignKeyConstraint(
            ["routing_projection_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.routing_projection_manifests.routing_projection_id",
                f"{RUNTIME_SCHEMA}.routing_projection_manifests.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["source_projection_member_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_round_projection_members.projection_member_id",
                f"{RUNTIME_SCHEMA}.qbank_round_projection_members.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("routing_projection_id", "ordinal"),
        UniqueConstraint("run_id", "qbank_id"),
        CheckConstraint("route_class IN ('answerable','control')", name="route_class_values"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    routing_member_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    routing_projection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.memory_programs.plan_id"), nullable=False
    )
    route_class: Mapped[str] = mapped_column(String(16), nullable=False)
    source_projection_member_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class StageComponentManifestModel(Base):
    __tablename__ = "stage_component_manifests"
    __table_args__ = (
        CheckConstraint("status IN ('building','ready','failed')", name="status_values"),
        CheckConstraint("round_id IS NULL OR round_id BETWEEN 0 AND 3", name="round_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    component_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    launch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(128))
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str | None] = mapped_column(String(32))
    round_id: Mapped[int | None] = mapped_column(Integer)
    scientific_input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    upstream_output_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    output_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    alembic_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    row_counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    artifact_hashes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    image_digest: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
