# ruff: noqa: E501
"""Add Step 6D Doubter rounds and verified-QBank projections.

Revision ID: 20260725_0018
Revises: 20260724_0017
"""

from collections.abc import Callable, Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260725_0018"
down_revision: str | None = "20260724_0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"

_NEW_TABLES = (
    "questioner_answer_claims",
    "qbank_candidate_families",
    "qbank_candidate_family_members",
    "r0_input_manifests",
    "candidate_stage_jobs",
    "doubter_assessments",
    "doubter_assessment_dimensions",
    "doubter_directives",
    "planner_revision_lineage",
    "qbank_round_projection_manifests",
    "qbank_round_projection_members",
    "routing_projection_manifests",
    "routing_projection_members",
    "stage_component_manifests",
)


def _migration_event(*, phase: str, state: str, detail: str | None = None) -> None:
    fields = [f"phase={phase}", f"state={state}"]
    if detail is not None:
        fields.append(f"detail={detail}")
    print("STEP6D_MIGRATION " + " ".join(fields), flush=True)


def _run_upgrade_phase(name: str, operation: Callable[[], None]) -> None:
    _migration_event(phase=name, state="start")
    try:
        operation()
    except Exception as exc:
        _migration_event(
            phase=name,
            state="failed",
            detail=type(exc).__name__,
        )
        raise
    _migration_event(phase=name, state="done")


def upgrade() -> None:
    _migration_event(phase="upgrade", state="start", detail="0017_to_0018")
    _run_upgrade_phase("compiler_identity", _extend_compiler_identity)
    _run_upgrade_phase("qbank_contract", _extend_qbank_contract)
    _run_upgrade_phase("questioner_v33_tables", _create_questioner_v33_tables)
    _run_upgrade_phase("doubter_tables", _create_doubter_tables)
    _run_upgrade_phase("structured_diagnostics", _extend_structured_diagnostics)
    _run_upgrade_phase(
        "questioner_generation_scope_guard",
        lambda: _replace_questioner_generation_scope_guard(enable_v06=True),
    )
    _run_upgrade_phase(
        "questioner_verifier_scope_guard",
        lambda: _replace_questioner_verifier_scope_guard(enable_v06=True),
    )
    _run_upgrade_phase(
        "questioner_revision_guard",
        lambda: _extend_questioner_revision_guard(enable_v06=True),
    )
    _run_upgrade_phase(
        "verified_qbank_completion_guard",
        _create_verified_qbank_completion_guard,
    )
    _run_upgrade_phase("runtime_routing_view", _create_runtime_routing_view)
    _run_upgrade_phase("permissions", _grant_permissions)
    _migration_event(phase="upgrade", state="done", detail="0017_to_0018")


def _extend_compiler_identity() -> None:
    columns = (
        ("doubter_config_hash", sa.String(128)),
        ("doubter_model_alias", sa.String(256)),
        ("doubter_model_revision", sa.String(256)),
        ("doubter_prompt_version", sa.String(128)),
        ("doubter_prompt_hash", sa.String(128)),
        ("doubter_renderer_version", sa.String(128)),
        ("doubter_parser_version", sa.String(128)),
        ("doubter_schema_version", sa.String(128)),
        ("r0_input_policy_version", sa.String(128)),
        ("routing_projection_version", sa.String(128)),
        ("active_round", sa.Integer()),
        ("max_round", sa.Integer()),
    )
    for name, column_type in columns:
        op.add_column(
            "compiler_runs",
            sa.Column(name, column_type, nullable=True),
            schema=RUNTIME,
        )
    op.create_check_constraint(
        "verified_qbank_rounds",
        "compiler_runs",
        "(compiler_schema_version != '0.6.0') OR "
        "(target_stage = 'verified_qbank' AND active_round = 2 AND max_round = 3 "
        "AND doubter_config_hash IS NOT NULL "
        "AND doubter_model_alias IS NOT NULL "
        "AND doubter_model_revision IS NOT NULL "
        "AND doubter_prompt_version IS NOT NULL "
        "AND doubter_prompt_hash IS NOT NULL "
        "AND doubter_renderer_version IS NOT NULL "
        "AND doubter_parser_version IS NOT NULL "
        "AND doubter_schema_version IS NOT NULL "
        "AND r0_input_policy_version IS NOT NULL "
        "AND routing_projection_version IS NOT NULL)",
        schema=RUNTIME,
    )


def _extend_qbank_contract() -> None:
    op.add_column(
        "qbank_questions",
        sa.Column(
            "route_class",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'answerable'"),
        ),
        schema=RUNTIME,
    )
    op.execute(
        f"""UPDATE {RUNTIME}.qbank_questions
        SET route_class = CASE
          WHEN question_type = 'answer_refusal' THEN 'control'
          ELSE 'answerable'
        END"""
    )
    op.alter_column(
        "qbank_questions",
        "route_class",
        server_default=None,
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "route_class_values",
        "qbank_questions",
        "route_class IN ('answerable','control')",
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "route_class_question_type",
        "qbank_questions",
        "(route_class = 'control' AND question_type = 'answer_refusal') OR "
        "(route_class = 'answerable' AND question_type != 'answer_refusal')",
        schema=RUNTIME,
    )
    op.create_index(
        "ix_qbank_questions_run_route_class",
        "qbank_questions",
        ["run_id", "route_class"],
        schema=RUNTIME,
    )
    op.create_unique_constraint(
        "uq_doubter_feedback_feedback_run",
        "doubter_feedback",
        ["feedback_id", "run_id"],
        schema=RUNTIME,
    )


def _create_questioner_v33_tables() -> None:
    op.create_table(
        "questioner_answer_claims",
        sa.Column("answer_claim_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("compiler_unit_id", sa.String(128), nullable=False),
        sa.Column("generation_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("claim_kind", sa.String(16), nullable=False),
        sa.Column("evidence_alias_hash", sa.String(128), nullable=False),
        sa.Column("evidence_id", sa.String(256), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("support_quote", sa.Text()),
        sa.Column("quote_verified", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["qbank_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME}.qbank_questions.qbank_id",
                f"{RUNTIME}.qbank_questions.run_id",
                f"{RUNTIME}.qbank_questions.compiler_unit_id",
            ],
        ),
        sa.ForeignKeyConstraint(
            ["generation_id", "run_id"],
            [
                f"{RUNTIME}.questioner_generations.generation_id",
                f"{RUNTIME}.questioner_generations.run_id",
            ],
        ),
        sa.UniqueConstraint("qbank_id", "ordinal"),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        sa.CheckConstraint(
            "claim_kind IN ('text','visual')", name="claim_kind_values"
        ),
        sa.CheckConstraint(
            "(claim_kind = 'text' AND support_quote IS NOT NULL AND quote_verified) "
            "OR (claim_kind = 'visual' AND support_quote IS NULL)",
            name="claim_grounding_values",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "qbank_candidate_families",
        sa.Column("family_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("conversation_id", sa.String(256), nullable=False),
        sa.Column("primary_qbank_id", sa.String(128), nullable=False),
        sa.Column("normalized_question_hash", sa.String(128), nullable=False),
        sa.Column("answer_claim_hash", sa.String(128), nullable=False),
        sa.Column("requirement_contract_hash", sa.String(128), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["primary_qbank_id", "run_id"],
            [f"{RUNTIME}.qbank_questions.qbank_id", f"{RUNTIME}.qbank_questions.run_id"],
        ),
        sa.UniqueConstraint("family_id", "run_id"),
        sa.UniqueConstraint(
            "run_id",
            "conversation_id",
            "normalized_question_hash",
            "answer_claim_hash",
            "requirement_contract_hash",
            name="qbank_candidate_family_identity",
        ),
        sa.CheckConstraint("member_count > 0", name="member_count_positive"),
        schema=RUNTIME,
    )
    op.create_table(
        "qbank_candidate_family_members",
        sa.Column("family_member_id", sa.String(128), primary_key=True),
        sa.Column("family_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["family_id", "run_id"],
            [
                f"{RUNTIME}.qbank_candidate_families.family_id",
                f"{RUNTIME}.qbank_candidate_families.run_id",
            ],
        ),
        sa.ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [f"{RUNTIME}.qbank_questions.qbank_id", f"{RUNTIME}.qbank_questions.run_id"],
        ),
        sa.UniqueConstraint("family_id", "ordinal"),
        sa.UniqueConstraint("run_id", "qbank_id"),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        schema=RUNTIME,
    )


def _create_doubter_tables() -> None:
    op.create_table(
        "r0_input_manifests",
        sa.Column("r0_manifest_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("answerable_candidate_count", sa.Integer(), nullable=False),
        sa.Column("control_candidate_count", sa.Integer(), nullable=False),
        sa.Column("program_count", sa.Integer(), nullable=False),
        sa.Column("execution_count", sa.Integer(), nullable=False),
        sa.Column("terminal_outcome_count", sa.Integer(), nullable=False),
        sa.Column("questioner_identity_hash", sa.String(128), nullable=False),
        sa.Column("planner_identity_hash", sa.String(128), nullable=False),
        sa.Column("operator_registry_hash", sa.String(128), nullable=False),
        sa.Column("memory_index_hash", sa.String(128), nullable=False),
        sa.Column("query_vector_hash", sa.String(128), nullable=False),
        sa.Column("facts_hash", sa.String(128), nullable=False),
        sa.Column("artifact_hash", sa.String(128), nullable=False),
        sa.Column("hard_gate_passed", sa.Boolean(), nullable=False),
        sa.Column("issue_codes", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [f"{RUNTIME}.compiler_runs.run_id", f"{RUNTIME}.compiler_runs.snapshot_id"],
        ),
        sa.UniqueConstraint("run_id"),
        sa.UniqueConstraint("r0_manifest_id", "run_id"),
        sa.CheckConstraint(
            "candidate_count >= 0 AND answerable_candidate_count >= 0 "
            "AND control_candidate_count >= 0 AND program_count >= 0 "
            "AND execution_count >= 0 AND terminal_outcome_count >= 0",
            name="counts_nonnegative",
        ),
        sa.CheckConstraint(
            "candidate_count = answerable_candidate_count + control_candidate_count",
            name="route_counts_reconcile",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "candidate_stage_jobs",
        sa.Column("candidate_stage_job_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("compiler_unit_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("claimed_by", sa.String(256)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_type", sa.String(256)),
        sa.Column("error_summary", sa.Text()),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [f"{RUNTIME}.qbank_questions.qbank_id", f"{RUNTIME}.qbank_questions.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [f"{RUNTIME}.compiler_units.compiler_unit_id", f"{RUNTIME}.compiler_units.run_id"],
        ),
        sa.UniqueConstraint("run_id", "qbank_id", "stage", "round_id"),
        sa.CheckConstraint(
            "stage IN ('doubter','planner_revision','dry_run','decision')",
            name="stage_values",
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="status_values",
        ),
        sa.CheckConstraint("round_id BETWEEN 0 AND 3", name="round_values"),
        sa.CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0", name="attempt_counts_valid"
        ),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_candidate_stage_jobs_claim",
        "candidate_stage_jobs",
        ["run_id", "stage", "round_id", "status", "available_at"],
        schema=RUNTIME,
    )
    op.create_table(
        "doubter_assessments",
        sa.Column("assessment_id", sa.String(128), primary_key=True),
        sa.Column("feedback_id", sa.String(128), nullable=False, unique=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("compiler_unit_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("plan_id", sa.String(128), nullable=False),
        sa.Column("execution_id", sa.String(128), nullable=False),
        sa.Column("producer_model_call_id", sa.String(128), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("prompt_version", sa.String(128), nullable=False),
        sa.Column("issue_codes", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("provider_schema_conformant", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feedback_id", "run_id"],
            [f"{RUNTIME}.doubter_feedback.feedback_id", f"{RUNTIME}.doubter_feedback.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["plan_id", "run_id"],
            [f"{RUNTIME}.memory_programs.plan_id", f"{RUNTIME}.memory_programs.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["producer_model_call_id", "run_id"],
            [f"{RUNTIME}.model_calls.model_call_id", f"{RUNTIME}.model_calls.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            [f"{RUNTIME}.program_executions.execution_id"],
        ),
        sa.UniqueConstraint("assessment_id", "run_id"),
        sa.UniqueConstraint("run_id", "qbank_id", "round_id"),
        sa.CheckConstraint("round_id BETWEEN 0 AND 3", name="round_values"),
        sa.CheckConstraint(
            "action IN ('pass','revise_program','reject_candidate')",
            name="action_values",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "doubter_assessment_dimensions",
        sa.Column("assessment_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("dimension", sa.String(64), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["assessment_id", "run_id"],
            [f"{RUNTIME}.doubter_assessments.assessment_id", f"{RUNTIME}.doubter_assessments.run_id"],
        ),
        sa.PrimaryKeyConstraint("assessment_id", "dimension"),
        sa.CheckConstraint(
            "dimension IN ('candidate_supported','answer_entailed','path_sufficient',"
            "'modality_sufficient','raw_image_sufficient','context_sufficient',"
            "'multi_hop_sufficient','temporal_consistent','non_redundant',"
            "'refusal_appropriate')",
            name="dimension_values",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "doubter_directives",
        sa.Column("directive_id", sa.String(128), primary_key=True),
        sa.Column("feedback_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("directive", sa.String(64), nullable=False),
        sa.Column("top_k_tier", sa.Integer()),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feedback_id", "run_id"],
            [f"{RUNTIME}.doubter_feedback.feedback_id", f"{RUNTIME}.doubter_feedback.run_id"],
        ),
        sa.UniqueConstraint("feedback_id", "ordinal"),
        sa.CheckConstraint("round_id BETWEEN 1 AND 3", name="round_values"),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        sa.CheckConstraint(
            "directive IN ('add_text_retrieval','add_visual_retrieval',"
            "'add_image_text_pair','add_context','add_temporal',"
            "'add_raw_image_inspection','increase_top_k',"
            "'remove_redundant_retrieval')",
            name="directive_values",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "planner_revision_lineage",
        sa.Column("revision_lineage_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("source_plan_id", sa.String(128), nullable=False),
        sa.Column("revised_plan_id", sa.String(128), nullable=False),
        sa.Column("source_feedback_id", sa.String(128), nullable=False),
        sa.Column("target_round_id", sa.Integer(), nullable=False),
        sa.Column("producer_model_call_id", sa.String(128), nullable=False),
        sa.Column("directive_hash", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_plan_id", "run_id"],
            [f"{RUNTIME}.memory_programs.plan_id", f"{RUNTIME}.memory_programs.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["revised_plan_id", "run_id"],
            [f"{RUNTIME}.memory_programs.plan_id", f"{RUNTIME}.memory_programs.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["source_feedback_id", "run_id"],
            [f"{RUNTIME}.doubter_feedback.feedback_id", f"{RUNTIME}.doubter_feedback.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["producer_model_call_id"],
            [f"{RUNTIME}.model_calls.model_call_id"],
        ),
        sa.UniqueConstraint("run_id", "qbank_id", "target_round_id"),
        sa.CheckConstraint("target_round_id BETWEEN 1 AND 3", name="round_values"),
        schema=RUNTIME,
    )
    _create_projection_tables()
    _create_stage_component_table()


def _create_projection_tables() -> None:
    op.create_table(
        "qbank_round_projection_manifests",
        sa.Column("projection_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("source_r0_manifest_id", sa.String(128), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("accepted_count", sa.Integer(), nullable=False),
        sa.Column("rejected_count", sa.Integer(), nullable=False),
        sa.Column("pending_count", sa.Integer(), nullable=False),
        sa.Column("members_hash", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_r0_manifest_id", "run_id"],
            [f"{RUNTIME}.r0_input_manifests.r0_manifest_id", f"{RUNTIME}.r0_input_manifests.run_id"],
        ),
        sa.UniqueConstraint("run_id", "round_id"),
        sa.UniqueConstraint("projection_id", "run_id"),
        sa.CheckConstraint("round_id BETWEEN 0 AND 3", name="round_values"),
        sa.CheckConstraint(
            "member_count = accepted_count + rejected_count + pending_count",
            name="counts_reconcile",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "qbank_round_projection_members",
        sa.Column("projection_member_id", sa.String(128), primary_key=True),
        sa.Column("projection_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("route_class", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("effective_plan_id", sa.String(128)),
        sa.Column("effective_feedback_id", sa.String(128)),
        sa.Column("passed_at_round", sa.Integer()),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["projection_id", "run_id"],
            [f"{RUNTIME}.qbank_round_projection_manifests.projection_id", f"{RUNTIME}.qbank_round_projection_manifests.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [f"{RUNTIME}.qbank_questions.qbank_id", f"{RUNTIME}.qbank_questions.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["effective_plan_id"],
            [f"{RUNTIME}.memory_programs.plan_id"],
        ),
        sa.ForeignKeyConstraint(
            ["effective_feedback_id"],
            [f"{RUNTIME}.doubter_feedback.feedback_id"],
        ),
        sa.UniqueConstraint("projection_id", "qbank_id"),
        sa.UniqueConstraint("projection_member_id", "run_id"),
        sa.CheckConstraint(
            "route_class IN ('answerable','control')", name="route_class_values"
        ),
        sa.CheckConstraint(
            "outcome IN ('accepted','rejected','pending','max_rounds_exhausted','carried_forward')",
            name="outcome_values",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "routing_projection_manifests",
        sa.Column("routing_projection_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("active_round", sa.Integer(), nullable=False),
        sa.Column("max_round", sa.Integer(), nullable=False),
        sa.Column("source_projection_id", sa.String(128), nullable=False),
        sa.Column("decision_hash", sa.String(128), nullable=False),
        sa.Column("lexical_collection_id", sa.String(128), nullable=False),
        sa.Column("text_collection_id", sa.String(128), nullable=False),
        sa.Column("multimodal_collection_id", sa.String(128), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("answerable_count", sa.Integer(), nullable=False),
        sa.Column("control_count", sa.Integer(), nullable=False),
        sa.Column("members_hash", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_projection_id", "run_id"],
            [f"{RUNTIME}.qbank_round_projection_manifests.projection_id", f"{RUNTIME}.qbank_round_projection_manifests.run_id"],
        ),
        sa.UniqueConstraint("run_id"),
        sa.UniqueConstraint("routing_projection_id", "run_id"),
        sa.CheckConstraint("active_round = 2 AND max_round = 3", name="rounds_frozen"),
        sa.CheckConstraint(
            "member_count = answerable_count + control_count",
            name="route_counts_reconcile",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "routing_projection_members",
        sa.Column("routing_member_id", sa.String(128), primary_key=True),
        sa.Column("routing_projection_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("plan_id", sa.String(128), nullable=False),
        sa.Column("route_class", sa.String(16), nullable=False),
        sa.Column("source_projection_member_id", sa.String(128), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["routing_projection_id", "run_id"],
            [f"{RUNTIME}.routing_projection_manifests.routing_projection_id", f"{RUNTIME}.routing_projection_manifests.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["source_projection_member_id", "run_id"],
            [f"{RUNTIME}.qbank_round_projection_members.projection_member_id", f"{RUNTIME}.qbank_round_projection_members.run_id"],
        ),
        sa.ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [f"{RUNTIME}.qbank_questions.qbank_id", f"{RUNTIME}.qbank_questions.run_id"],
        ),
        sa.ForeignKeyConstraint(["plan_id"], [f"{RUNTIME}.memory_programs.plan_id"]),
        sa.UniqueConstraint("routing_projection_id", "ordinal"),
        sa.UniqueConstraint("run_id", "qbank_id"),
        sa.CheckConstraint(
            "route_class IN ('answerable','control')", name="route_class_values"
        ),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        schema=RUNTIME,
    )


def _create_stage_component_table() -> None:
    op.create_table(
        "stage_component_manifests",
        sa.Column("component_id", sa.String(128), primary_key=True),
        sa.Column("launch_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128)),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(128), nullable=False),
        sa.Column("dataset", sa.String(32)),
        sa.Column("round_id", sa.Integer()),
        sa.Column("scientific_input_hash", sa.String(128), nullable=False),
        sa.Column("source_fingerprint", sa.String(128), nullable=False),
        sa.Column("upstream_output_hash", sa.String(128), nullable=False),
        sa.Column("output_hash", sa.String(128), nullable=False),
        sa.Column("alembic_revision", sa.String(64), nullable=False),
        sa.Column("row_counts", postgresql.JSONB(), nullable=False),
        sa.Column("artifact_hashes", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("image_digest", sa.String(256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint(
            "status IN ('building','ready','failed')", name="status_values"
        ),
        sa.CheckConstraint(
            "round_id IS NULL OR round_id BETWEEN 0 AND 3", name="round_values"
        ),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_stage_component_lookup",
        "stage_component_manifests",
        ["launch_id", "stage", "status"],
        schema=RUNTIME,
    )


def _replace_structured_diagnostics_stage_check(expression: str) -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """SELECT constraints.conname
            FROM pg_catalog.pg_constraint AS constraints
            JOIN pg_catalog.pg_class AS relations
              ON relations.oid = constraints.conrelid
            JOIN pg_catalog.pg_namespace AS namespaces
              ON namespaces.oid = relations.relnamespace
            WHERE namespaces.nspname = :schema_name
              AND relations.relname = :table_name
              AND constraints.contype = 'c'
              AND lower(pg_catalog.pg_get_expr(
                    constraints.conbin, constraints.conrelid
                  )) ~ '(^|[^a-z_])stage([^a-z_]|$)'
            ORDER BY constraints.conname"""
        ),
        {
            "schema_name": RUNTIME,
            "table_name": "structured_output_item_diagnostics",
        },
    ).all()
    constraint_names = [str(row[0]) for row in rows]
    if len(constraint_names) != 1:
        raise RuntimeError(
            "expected exactly one structured-output stage CHECK constraint; "
            f"found {len(constraint_names)}"
        )
    old_name = constraint_names[0]
    _migration_event(
        phase="structured_diagnostics_stage_check",
        state="replace",
        detail=old_name,
    )
    op.drop_constraint(
        op.f(old_name),
        "structured_output_item_diagnostics",
        schema=RUNTIME,
        type_="check",
    )
    _migration_event(
        phase="structured_diagnostics_stage_check",
        state="dropped",
        detail=old_name,
    )
    op.create_check_constraint(
        "stage_values",
        "structured_output_item_diagnostics",
        expression,
        schema=RUNTIME,
    )
    _migration_event(
        phase="structured_diagnostics_stage_check",
        state="ready",
        detail="ck_structured_output_item_diagnostics_stage_values",
    )


def _extend_structured_diagnostics() -> None:
    _replace_structured_diagnostics_stage_check(
        "stage IN ('questioner','questioner_verifier','questioner_revision',"
        "'planner','planner_revision','doubter')"
    )


def _extend_questioner_revision_guard(*, enable_v06: bool) -> None:
    """Keep the audited v3.2 revision chain valid for compiler schema 0.6."""

    old_condition = "compiler_version != '0.5.0'"
    new_condition = (
        "compiler_version NOT IN ('0.5.0', '0.6.0')"
        if enable_v06
        else old_condition
    )
    source_condition = (
        old_condition
        if enable_v06
        else "compiler_version NOT IN ('0.5.0', '0.6.0')"
    )
    source_literal = "'" + source_condition.replace("'", "''") + "'"
    target_literal = "'" + new_condition.replace("'", "''") + "'"
    op.execute(
        f"""DO $$
        DECLARE function_definition text;
        BEGIN
          SELECT pg_get_functiondef(
            '{RUNTIME}.enforce_questioner_candidate_scope_v5()'::regprocedure
          ) INTO function_definition;
          IF position({source_literal} in function_definition) = 0 THEN
            RAISE EXCEPTION 'unexpected Questioner candidate guard definition';
          END IF;
          function_definition := replace(
            function_definition,
            {source_literal},
            {target_literal}
          );
          EXECUTE function_definition;
        END $$"""
    )


def _replace_questioner_generation_scope_guard(*, enable_v06: bool) -> None:
    """Allow only an archived final v3.3 envelope failure to close as zero candidates."""

    v06_terminal_clause = (
        """
               OR (
                 compiler_version = '0.6.0'
                 AND call.status = 'failed'
                 AND call.output_asset_id IS NOT NULL
                 AND call.parse_error IS NOT NULL
                 AND COALESCE(
                   call.request_config->'request_metadata'->>'task', ''
                 ) = 'pmmc_questioner'
                 AND NEW.parse_status = 'failed'
                 AND NEW.raw_candidate_count = 0
                 AND NEW.valid_candidate_count = 0
                 AND NEW.invalid_candidate_count = 0
                 AND NEW.repair_attempt_count = 0
                 AND NEW.output_hash IS NOT NULL
                 AND COALESCE(
                   NEW.metadata->>'native_structured_output', 'true'
                 ) = 'false'
                 AND COALESCE(NEW.metadata->>'response_artifact_ready', 'false') = 'true'
                 AND COALESCE(NEW.metadata->>'terminal_reason', '')
                   = 'generation_structured_output_invalid'
               )
        """
        if enable_v06
        else ""
    )
    op.execute(
        f"""CREATE OR REPLACE FUNCTION {RUNTIME}.validate_questioner_fact_scope()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE anchor_role text; support_role text; generation record; question record;
                call record; compiler_version text; artifact_state text;
        BEGIN
            IF TG_TABLE_NAME = 'compiler_unit_links' THEN
                SELECT role INTO anchor_role FROM {RUNTIME}.compiler_unit_memories
                 WHERE compiler_unit_id = NEW.compiler_unit_id
                   AND memory_id = NEW.anchor_memory_id;
                SELECT role INTO support_role FROM {RUNTIME}.compiler_unit_memories
                 WHERE compiler_unit_id = NEW.compiler_unit_id
                   AND memory_id = NEW.support_memory_id;
                IF anchor_role != 'anchor' OR support_role != 'support' THEN
                    RAISE EXCEPTION 'compiler unit links must connect anchor to support'
                      USING ERRCODE = '23514',
                            SCHEMA = '{RUNTIME}',
                            TABLE = 'compiler_unit_links',
                            CONSTRAINT = 'ck_compiler_unit_links_anchor_support';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_generations' THEN
                SELECT calls.* INTO call FROM {RUNTIME}.model_calls calls
                 WHERE calls.model_call_id = NEW.model_call_id
                   AND calls.run_id = NEW.run_id;
                SELECT compiler_schema_version INTO compiler_version
                  FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
                IF call.model_call_id IS NULL
                   OR call.compiler_unit_id != NEW.compiler_unit_id
                   OR call.job_id != NEW.job_id
                   OR call.stage != 'questioner' THEN
                    RAISE EXCEPTION 'Questioner generation has inconsistent model-call lineage'
                      USING ERRCODE = '23514',
                            SCHEMA = '{RUNTIME}',
                            TABLE = 'questioner_generations',
                            CONSTRAINT = 'ck_questioner_generations_model_lineage';
                END IF;
                IF NOT (
                    call.status = 'success'
                    {v06_terminal_clause}
                ) THEN
                    RAISE EXCEPTION 'Questioner failed call is not a terminal generation rejection'
                      USING ERRCODE = '23514',
                            SCHEMA = '{RUNTIME}',
                            TABLE = 'questioner_generations',
                            CONSTRAINT = 'ck_questioner_generations_failed_call_terminal_rejection';
                END IF;
                IF call.status = 'failed' THEN
                    SELECT state INTO artifact_state FROM {RUNTIME}.run_artifacts
                     WHERE artifact_id = call.output_asset_id AND run_id = NEW.run_id;
                    IF artifact_state IS DISTINCT FROM 'ready' THEN
                        RAISE EXCEPTION 'Questioner terminal response artifact is not ready'
                          USING ERRCODE = '23514',
                                SCHEMA = '{RUNTIME}',
                                TABLE = 'questioner_generations',
                                CONSTRAINT = 'ck_questioner_generations_terminal_artifact_ready';
                    END IF;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.experiment_jobs jobs
                    WHERE jobs.job_id = NEW.job_id
                      AND jobs.run_id = NEW.run_id
                      AND jobs.compiler_unit_id = NEW.compiler_unit_id
                      AND jobs.stage = 'compile'
                ) THEN
                    RAISE EXCEPTION 'Questioner generation has no matching COMPILE job'
                      USING ERRCODE = '23514',
                            SCHEMA = '{RUNTIME}',
                            TABLE = 'questioner_generations',
                            CONSTRAINT = 'ck_questioner_generations_compile_job';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_candidate_gate_results' THEN
                SELECT * INTO generation FROM {RUNTIME}.questioner_generations
                 WHERE generation_id = NEW.generation_id AND run_id = NEW.run_id;
                SELECT * INTO question FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id AND run_id = NEW.run_id;
                IF generation.compiler_unit_id != NEW.compiler_unit_id
                   OR question.compiler_unit_id != NEW.compiler_unit_id
                   OR question.generation_id != NEW.generation_id
                   OR question.candidate_content_hash != NEW.content_hash THEN
                    RAISE EXCEPTION 'Questioner gate result has inconsistent candidate lineage'
                      USING ERRCODE = '23514',
                            SCHEMA = '{RUNTIME}',
                            TABLE = 'questioner_candidate_gate_results',
                            CONSTRAINT = 'ck_questioner_gate_candidate_lineage';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.validate_questioner_fact_scope() FROM PUBLIC"
    )


def _replace_questioner_verifier_scope_guard(*, enable_v06: bool) -> None:
    """Keep verifier model-call lineage strict for the active compiler schema."""

    v06_prompt_case = (
        "WHEN '0.6.0' THEN 'pmmc-questioner-verifier-v4'"
        if enable_v06
        else ""
    )
    single_candidate_versions = (
        "compiler_version IN ('0.5.0', '0.6.0')"
        if enable_v06
        else "compiler_version = '0.5.0'"
    )
    v06_failed_call_clause = (
        """
                    OR (
                      compiler_version = '0.6.0'
                      AND call_row.status = 'failed'
                      AND call_row.output_asset_id IS NOT NULL
                      AND call_row.parse_error IS NOT NULL
                      AND NEW.scope_qbank_id IS NOT NULL
                      AND NEW.input_count = 1
                      AND NEW.pass_count = 0
                      AND NEW.revise_count = 0
                      AND NEW.reject_count = 1
                      AND COALESCE(
                        (NEW.metadata->>'provider_output_fallback')::boolean,
                        false
                      )
                      AND COALESCE(
                        (NEW.metadata->>'single_candidate_scope')::boolean,
                        false
                      )
                    )
        """
        if enable_v06
        else ""
    )
    op.execute(
        f"""CREATE OR REPLACE FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v3()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE call_row record; generation_call record; generation_row record;
                compiler_version text; expected_prompt_version text;
        BEGIN
            SELECT * INTO call_row FROM {RUNTIME}.model_calls
             WHERE model_call_id = NEW.model_call_id AND run_id = NEW.run_id;
            SELECT * INTO generation_row FROM {RUNTIME}.questioner_generations
             WHERE generation_id = NEW.generation_id AND run_id = NEW.run_id;
            SELECT * INTO generation_call FROM {RUNTIME}.model_calls
             WHERE model_call_id = generation_row.model_call_id AND run_id = NEW.run_id;
            SELECT compiler_schema_version INTO compiler_version
              FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
            expected_prompt_version := CASE compiler_version
                WHEN '0.3.0' THEN 'pmmc-questioner-verifier-v1'
                WHEN '0.4.0' THEN 'pmmc-questioner-verifier-v2'
                WHEN '0.5.0' THEN 'pmmc-questioner-verifier-v3'
                {v06_prompt_case}
                ELSE NULL
            END;
            IF call_row.model_call_id IS NULL
               OR generation_row.generation_id IS NULL
               OR generation_call.model_call_id IS NULL THEN
                RAISE EXCEPTION 'Questioner verifier batch parent facts are incomplete'
                    USING ERRCODE = '23514',
                          SCHEMA = '{RUNTIME}',
                          TABLE = 'questioner_verification_batches',
                          CONSTRAINT = 'ck_qv_batches_parent_facts';
            END IF;
            IF expected_prompt_version IS NULL
               OR NEW.prompt_version != expected_prompt_version
               OR COALESCE(call_row.request_config->'request_metadata'->>'task', '')
                  != 'pmmc_questioner_verifier' THEN
                RAISE EXCEPTION 'Questioner verifier batch protocol is inconsistent'
                    USING ERRCODE = '23514',
                          SCHEMA = '{RUNTIME}',
                          TABLE = 'questioner_verification_batches',
                          CONSTRAINT = 'ck_qv_batches_protocol';
            END IF;
            IF call_row.compiler_unit_id != NEW.compiler_unit_id
               OR call_row.stage != 'questioner'
               OR call_row.parent_model_call_id != generation_row.model_call_id
               OR call_row.model_alias != generation_call.model_alias
               OR call_row.actual_model_id IS DISTINCT FROM generation_call.actual_model_id
               OR call_row.job_id IS DISTINCT FROM generation_call.job_id THEN
                RAISE EXCEPTION 'Questioner verifier batch model lineage is inconsistent'
                    USING ERRCODE = '23514',
                          SCHEMA = '{RUNTIME}',
                          TABLE = 'questioner_verification_batches',
                          CONSTRAINT = 'ck_qv_batches_model_lineage';
            END IF;
            IF ({single_candidate_versions} AND (
                 NEW.scope_qbank_id IS NULL OR NEW.input_count != 1)) THEN
                RAISE EXCEPTION 'Questioner verifier batch candidate scope is inconsistent'
                    USING ERRCODE = '23514',
                          SCHEMA = '{RUNTIME}',
                          TABLE = 'questioner_verification_batches',
                          CONSTRAINT = 'ck_qv_batches_single_candidate_scope';
            END IF;
            IF NOT (
                    call_row.status = 'success'
                    OR (
                      compiler_version = '0.5.0'
                      AND call_row.status = 'failed'
                      AND call_row.output_asset_id IS NOT NULL
                      AND call_row.parse_error IS NOT NULL
                      AND COALESCE(
                        (NEW.metadata->>'provider_output_fallback')::boolean,
                        false
                      )
                    )
                    {v06_failed_call_clause}
               ) THEN
                RAISE EXCEPTION 'Questioner verifier failed call is not a terminal rejection'
                    USING ERRCODE = '23514',
                          SCHEMA = '{RUNTIME}',
                          TABLE = 'questioner_verification_batches',
                          CONSTRAINT = 'ck_qv_batches_failed_call_terminal_rejection';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v3() "
        "FROM PUBLIC"
    )


def _create_verified_qbank_completion_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_verified_qbank_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler {RUNTIME}.compiler_runs%ROWTYPE; bad integer; candidates integer;
                manifest {RUNTIME}.r0_input_manifests%ROWTYPE;
                routing {RUNTIME}.routing_projection_manifests%ROWTYPE;
        BEGIN
          IF NEW.status != 'completed' THEN RETURN NEW; END IF;
          SELECT * INTO compiler FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
          IF NOT FOUND OR compiler.compiler_schema_version != '0.6.0' THEN RETURN NEW; END IF;
          IF compiler.target_stage != 'verified_qbank'
             OR compiler.active_round != 2 OR compiler.max_round != 3 THEN
            RAISE EXCEPTION 'verified-QBank run identity is incomplete'
              USING ERRCODE = '55000';
          END IF;
          SELECT * INTO manifest FROM {RUNTIME}.r0_input_manifests
           WHERE run_id = NEW.run_id;
          IF NOT FOUND OR NOT manifest.hard_gate_passed THEN
            RAISE EXCEPTION 'verified-QBank requires a passing R0 input manifest'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO candidates
            FROM {RUNTIME}.qbank_questions questions
           WHERE questions.run_id = NEW.run_id
             AND EXISTS (
               SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gates
                WHERE gates.run_id = questions.run_id
                  AND gates.qbank_id = questions.qbank_id
                  AND gates.status = 'eligible_for_planning'
             )
             AND EXISTS (
               SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verifies
                WHERE verifies.run_id = questions.run_id
                  AND verifies.qbank_id = questions.qbank_id
                  AND verifies.verification_round = questions.candidate_version
                  AND verifies.verdict = 'pass'
             );
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_manifests
           WHERE run_id = NEW.run_id AND round_id BETWEEN 0 AND 3;
          IF bad != 4 THEN
            RAISE EXCEPTION 'verified-QBank requires r0-r3 projections'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_manifests p
           WHERE p.run_id = NEW.run_id
             AND p.member_count != candidates;
          IF bad != 0 THEN
            RAISE EXCEPTION 'round projection denominator mismatch'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.qbank_round_projection_manifests projection
            CROSS JOIN {RUNTIME}.qbank_questions questions
            LEFT JOIN {RUNTIME}.qbank_round_projection_members members
              ON members.projection_id = projection.projection_id
             AND members.run_id = projection.run_id
             AND members.qbank_id = questions.qbank_id
           WHERE projection.run_id = NEW.run_id
             AND questions.run_id = NEW.run_id
             AND EXISTS (
               SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gates
                WHERE gates.run_id = questions.run_id
                  AND gates.qbank_id = questions.qbank_id
                  AND gates.status = 'eligible_for_planning'
             )
             AND EXISTS (
               SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verifies
                WHERE verifies.run_id = questions.run_id
                  AND verifies.qbank_id = questions.qbank_id
                  AND verifies.verification_round = questions.candidate_version
                  AND verifies.verdict = 'pass'
             )
             AND members.projection_member_id IS NULL;
          IF bad != 0 THEN
            RAISE EXCEPTION 'round projections omit eligible candidates'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.qbank_round_projection_members members
            JOIN {RUNTIME}.qbank_round_projection_manifests projection
              ON projection.projection_id = members.projection_id
             AND projection.run_id = members.run_id
           WHERE members.run_id = NEW.run_id
             AND NOT EXISTS (
               SELECT 1
                 FROM {RUNTIME}.questioner_candidate_gate_results gates
                 JOIN {RUNTIME}.qbank_questions questions
                   ON questions.run_id = gates.run_id
                  AND questions.qbank_id = gates.qbank_id
                 JOIN {RUNTIME}.questioner_candidate_verifications verifies
                   ON verifies.run_id = questions.run_id
                  AND verifies.qbank_id = questions.qbank_id
                  AND verifies.verification_round = questions.candidate_version
                WHERE gates.run_id = members.run_id
                  AND gates.qbank_id = members.qbank_id
                  AND gates.status = 'eligible_for_planning'
                  AND verifies.verdict = 'pass'
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'round projections contain ineligible candidates'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.qbank_questions questions
           WHERE questions.run_id = NEW.run_id
             AND EXISTS (
               SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gates
                WHERE gates.run_id = questions.run_id
                  AND gates.qbank_id = questions.qbank_id
                  AND gates.status = 'eligible_for_planning'
             )
             AND NOT EXISTS (
               SELECT 1 FROM {RUNTIME}.doubter_assessments assessments
                WHERE assessments.run_id = questions.run_id
                  AND assessments.qbank_id = questions.qbank_id
                  AND assessments.round_id = 0
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'Doubter round zero denominator is incomplete'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.doubter_assessments assessments
           WHERE assessments.run_id = NEW.run_id
             AND (
               (assessments.round_id > 0 AND NOT EXISTS (
                 SELECT 1 FROM {RUNTIME}.planner_revision_lineage revisions
                  WHERE revisions.run_id = assessments.run_id
                    AND revisions.qbank_id = assessments.qbank_id
                    AND revisions.target_round_id = assessments.round_id
                    AND revisions.revised_plan_id = assessments.plan_id
               ))
               OR EXISTS (
                 SELECT 1 FROM {RUNTIME}.doubter_assessments prior
                  WHERE prior.run_id = assessments.run_id
                    AND prior.qbank_id = assessments.qbank_id
                    AND prior.round_id < assessments.round_id
                    AND prior.action IN ('pass','reject_candidate')
               )
               OR (
                 assessments.round_id = 3
                 AND assessments.action = 'revise_program'
               )
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'Doubter assessment rounds or revision lineage are invalid'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.doubter_assessments assessments
           WHERE assessments.run_id = NEW.run_id
             AND assessments.action = 'revise_program'
             AND assessments.round_id < 3
             AND NOT EXISTS (
               SELECT 1 FROM {RUNTIME}.planner_revision_lineage revisions
                WHERE revisions.run_id = assessments.run_id
                  AND revisions.qbank_id = assessments.qbank_id
                  AND revisions.source_feedback_id = assessments.feedback_id
                  AND revisions.target_round_id = assessments.round_id + 1
             )
             AND NOT EXISTS (
               SELECT 1 FROM {RUNTIME}.candidate_stage_jobs jobs
                WHERE jobs.run_id = assessments.run_id
                  AND jobs.qbank_id = assessments.qbank_id
                  AND jobs.stage = 'planner_revision'
                  AND jobs.round_id = assessments.round_id + 1
                  AND jobs.status = 'cancelled'
                  AND jobs.metadata->>'terminal_reason' =
                      'planner_revision_unavailable'
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'Doubter revision chain is incomplete'
              USING ERRCODE = '55000';
          END IF;
          SELECT * INTO routing FROM {RUNTIME}.routing_projection_manifests
           WHERE run_id = NEW.run_id;
          IF NOT FOUND OR routing.active_round != 2 OR routing.max_round != 3
             OR NOT EXISTS (
               SELECT 1 FROM {RUNTIME}.qbank_round_projection_manifests projection
                WHERE projection.run_id = routing.run_id
                  AND projection.projection_id = routing.source_projection_id
                  AND projection.round_id = 2
             )
             OR routing.member_count != (
               SELECT count(*) FROM {RUNTIME}.routing_projection_members members
                WHERE members.routing_projection_id = routing.routing_projection_id
             ) THEN
            RAISE EXCEPTION 'active r2 routing projection is missing'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.routing_projection_members members
           JOIN {RUNTIME}.qbank_round_projection_members source
             ON source.projection_member_id = members.source_projection_member_id
            AND source.run_id = members.run_id
           JOIN {RUNTIME}.qbank_round_projection_manifests projection
             ON projection.projection_id = source.projection_id
            AND projection.run_id = source.run_id
           WHERE members.run_id = NEW.run_id
             AND (projection.round_id != 2
                  OR source.qbank_id != members.qbank_id
                  OR source.outcome NOT IN ('accepted','carried_forward')
                  OR source.effective_plan_id IS DISTINCT FROM members.plan_id);
          IF bad != 0 THEN
            RAISE EXCEPTION 'routing projection contains an invalid r2 member'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_decisions decisions
           WHERE decisions.run_id = NEW.run_id;
          IF bad != candidates THEN
            RAISE EXCEPTION 'active r2 decisions do not close the candidate denominator'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_decisions decisions
           LEFT JOIN {RUNTIME}.routing_projection_members members
             ON members.run_id = decisions.run_id
            AND members.qbank_id = decisions.qbank_id
           WHERE decisions.run_id = NEW.run_id
             AND ((decisions.accepted AND members.routing_member_id IS NULL)
                  OR (NOT decisions.accepted AND members.routing_member_id IS NOT NULL));
          IF bad != 0 THEN
            RAISE EXCEPTION 'decisions and routing projection disagree'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.candidate_stage_jobs jobs
           WHERE jobs.run_id = NEW.run_id
             AND jobs.status != 'succeeded'
             AND NOT (
               jobs.stage = 'planner_revision'
               AND jobs.status = 'cancelled'
               AND jobs.metadata->>'terminal_reason' =
                   'planner_revision_unavailable'
               AND EXISTS (
                 SELECT 1 FROM {RUNTIME}.doubter_assessments assessments
                  WHERE assessments.run_id = jobs.run_id
                    AND assessments.qbank_id = jobs.qbank_id
                    AND assessments.round_id + 1 = jobs.round_id
                    AND assessments.action = 'revise_program'
               )
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'verified-QBank contains unfinished candidate jobs'
              USING ERRCODE = '55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.qbank_questions questions
           WHERE questions.run_id = NEW.run_id
             AND EXISTS (
               SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gates
                WHERE gates.run_id = questions.run_id
                  AND gates.qbank_id = questions.qbank_id
                  AND gates.status = 'eligible_for_planning'
             )
             AND NOT (
               EXISTS (
                 SELECT 1 FROM {RUNTIME}.doubter_assessments assessments
                  WHERE assessments.run_id = questions.run_id
                    AND assessments.qbank_id = questions.qbank_id
                    AND assessments.action IN ('pass','reject_candidate')
               )
               OR EXISTS (
                 SELECT 1
                   FROM {RUNTIME}.doubter_assessments assessments
                   JOIN {RUNTIME}.candidate_stage_jobs jobs
                     ON jobs.run_id = assessments.run_id
                    AND jobs.qbank_id = assessments.qbank_id
                    AND jobs.stage = 'planner_revision'
                    AND jobs.round_id = assessments.round_id + 1
                  WHERE assessments.run_id = questions.run_id
                    AND assessments.qbank_id = questions.qbank_id
                    AND assessments.action = 'revise_program'
                    AND jobs.status = 'cancelled'
                    AND jobs.metadata->>'terminal_reason' =
                        'planner_revision_unavailable'
               )
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'verified-QBank candidate terminal outcomes are incomplete'
              USING ERRCODE = '55000';
          END IF;
          RETURN NEW;
        END $$"""
    )
    op.execute(
        f"""CREATE TRIGGER verified_qbank_completion_integrity
        BEFORE UPDATE ON {RUNTIME}.run_states
        FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_verified_qbank_completion()"""
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_verified_qbank_completion() FROM PUBLIC"
    )


def _create_runtime_routing_view() -> None:
    op.execute(
        f"""CREATE VIEW {RUNTIME}.verified_qbank_routing AS
        SELECT members.routing_projection_id,
               members.run_id,
               manifests.snapshot_id,
               questions.conversation_id,
               questions.created_after_session,
               questions.qbank_id,
               questions.canonical_question,
               questions.required_modalities,
               COALESCE(
                   (
                       SELECT array_agg(paraphrases.paraphrase ORDER BY paraphrases.ordinal)
                         FROM {RUNTIME}.qbank_paraphrases paraphrases
                        WHERE paraphrases.qbank_id = questions.qbank_id
                          AND paraphrases.run_id = questions.run_id
                   ),
                   ARRAY[]::text[]
               ) AS paraphrases,
               members.route_class,
               members.plan_id AS final_plan_id,
               members.ordinal
          FROM {RUNTIME}.routing_projection_members members
          JOIN {RUNTIME}.routing_projection_manifests manifests
            ON manifests.routing_projection_id = members.routing_projection_id
           AND manifests.run_id = members.run_id
          JOIN {RUNTIME}.qbank_questions questions
            ON questions.qbank_id = members.qbank_id
           AND questions.run_id = members.run_id"""
    )


def _grant_permissions() -> None:
    for table in _NEW_TABLES:
        op.execute(f"ALTER TABLE {RUNTIME}.{table} OWNER TO pmmc_owner")
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {RUNTIME}.{table} TO pmmc_runtime"
        )
        op.execute(f"GRANT SELECT ON {RUNTIME}.{table} TO pmmc_evaluator")
    op.execute(f"ALTER VIEW {RUNTIME}.verified_qbank_routing OWNER TO pmmc_owner")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing TO pmmc_runtime")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing TO pmmc_evaluator")


def downgrade() -> None:
    _migration_event(phase="downgrade", state="start", detail="0018_to_0017")
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing")
    op.execute(
        f"DROP TRIGGER IF EXISTS verified_qbank_completion_integrity "
        f"ON {RUNTIME}.run_states"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_verified_qbank_completion()"
    )
    _replace_questioner_generation_scope_guard(enable_v06=False)
    _replace_questioner_verifier_scope_guard(enable_v06=False)
    _extend_questioner_revision_guard(enable_v06=False)
    _replace_structured_diagnostics_stage_check(
        "stage IN ('questioner','questioner_verifier','questioner_revision','planner')",
    )
    for table in reversed(_NEW_TABLES):
        op.drop_table(table, schema=RUNTIME)
    op.drop_constraint(
        "uq_doubter_feedback_feedback_run",
        "doubter_feedback",
        schema=RUNTIME,
        type_="unique",
    )
    op.drop_index(
        "ix_qbank_questions_run_route_class",
        table_name="qbank_questions",
        schema=RUNTIME,
    )
    op.drop_constraint(
        op.f("ck_qbank_questions_route_class_question_type"),
        "qbank_questions",
        schema=RUNTIME,
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_qbank_questions_route_class_values"),
        "qbank_questions",
        schema=RUNTIME,
        type_="check",
    )
    op.drop_column("qbank_questions", "route_class", schema=RUNTIME)
    op.drop_constraint(
        op.f("ck_compiler_runs_verified_qbank_rounds"),
        "compiler_runs",
        schema=RUNTIME,
        type_="check",
    )
    for name in (
        "max_round",
        "active_round",
        "routing_projection_version",
        "r0_input_policy_version",
        "doubter_schema_version",
        "doubter_parser_version",
        "doubter_renderer_version",
        "doubter_prompt_hash",
        "doubter_prompt_version",
        "doubter_model_revision",
        "doubter_model_alias",
        "doubter_config_hash",
    ):
        op.drop_column("compiler_runs", name, schema=RUNTIME)
    _migration_event(phase="downgrade", state="done", detail="0018_to_0017")
