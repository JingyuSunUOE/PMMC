"""Internal SQLAlchemy models for PMMC write-time compiler persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import (
    COMPILATION_STAGE_VALUES,
    COMPILATION_TARGET_STAGE_VALUES,
    COMPILATION_UNIT_MEMORY_ROLE_VALUES,
    EVIDENCE_SUFFICIENCY_VALUES,
    EVIDENCE_TYPE_VALUES,
    MODALITY_VALUES,
    OPERATOR_CONTROL_ACTION_VALUES,
    OPERATOR_EXECUTION_PURPOSE_VALUES,
    QBANK_EVIDENCE_ROLE_VALUES,
    QUESTION_REASONING_TYPE_VALUES,
    QUESTION_TYPE_VALUES,
    RUNTIME_SCHEMA,
    TRACE_STATUS_VALUES,
    sql_text_array,
    sql_text_values,
)

JsonValue = dict[str, Any]


class CompilerRunModel(Base):
    __tablename__ = "compiler_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        UniqueConstraint("run_id", "snapshot_id"),
        CheckConstraint(
            f"target_stage IN {sql_text_values(COMPILATION_TARGET_STAGE_VALUES)}",
            name="target_stage_values",
        ),
        CheckConstraint("unit_count >= 0", name="unit_count_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), primary_key=True
    )
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    target_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    operator_set_version: Mapped[str] = mapped_column(String(128), nullable=False)
    operator_registry_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    questioner_config_hash: Mapped[str | None] = mapped_column(String(128))
    questioner_model_alias: Mapped[str | None] = mapped_column(String(256))
    questioner_model_revision: Mapped[str | None] = mapped_column(String(256))
    questioner_prompt_version: Mapped[str | None] = mapped_column(String(128))
    questioner_prompt_hash: Mapped[str | None] = mapped_column(String(128))
    questioner_renderer_version: Mapped[str | None] = mapped_column(String(128))
    questioner_parser_version: Mapped[str | None] = mapped_column(String(128))
    unit_builder_version: Mapped[str | None] = mapped_column(String(128))
    candidate_gate_version: Mapped[str | None] = mapped_column(String(128))
    dedupe_version: Mapped[str | None] = mapped_column(String(128))
    qbank_matcher_version: Mapped[str | None] = mapped_column(String(128))
    questioner_freeze_manifest_hash: Mapped[str | None] = mapped_column(String(128))
    planner_config_hash: Mapped[str | None] = mapped_column(String(128))
    planner_model_alias: Mapped[str | None] = mapped_column(String(256))
    planner_model_revision: Mapped[str | None] = mapped_column(String(256))
    planner_prompt_version: Mapped[str | None] = mapped_column(String(128))
    planner_prompt_hash: Mapped[str | None] = mapped_column(String(128))
    planner_renderer_version: Mapped[str | None] = mapped_column(String(128))
    planner_parser_version: Mapped[str | None] = mapped_column(String(128))
    planner_schema_version: Mapped[str | None] = mapped_column(String(128))
    program_policy_version: Mapped[str | None] = mapped_column(String(128))
    dry_run_policy_version: Mapped[str | None] = mapped_column(String(128))
    memory_index_representation_version: Mapped[str | None] = mapped_column(String(128))
    memory_text_model_alias: Mapped[str | None] = mapped_column(String(256))
    memory_text_model_revision: Mapped[str | None] = mapped_column(String(256))
    memory_multimodal_model_alias: Mapped[str | None] = mapped_column(String(256))
    memory_multimodal_model_revision: Mapped[str | None] = mapped_column(String(256))
    doubter_config_hash: Mapped[str | None] = mapped_column(String(128))
    doubter_model_alias: Mapped[str | None] = mapped_column(String(256))
    doubter_model_revision: Mapped[str | None] = mapped_column(String(256))
    doubter_prompt_version: Mapped[str | None] = mapped_column(String(128))
    doubter_prompt_hash: Mapped[str | None] = mapped_column(String(128))
    doubter_renderer_version: Mapped[str | None] = mapped_column(String(128))
    doubter_parser_version: Mapped[str | None] = mapped_column(String(128))
    doubter_schema_version: Mapped[str | None] = mapped_column(String(128))
    r0_input_policy_version: Mapped[str | None] = mapped_column(String(128))
    routing_projection_version: Mapped[str | None] = mapped_column(String(128))
    active_round: Mapped[int | None] = mapped_column(Integer)
    max_round: Mapped[int | None] = mapped_column(Integer)
    unit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class CompilerUnitModel(Base):
    __tablename__ = "compiler_units"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_runs.run_id",
                f"{RUNTIME_SCHEMA}.compiler_runs.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "available_after_session"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_sessions.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.conversation_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.session_id",
            ],
        ),
        UniqueConstraint("compiler_unit_id", "run_id"),
        UniqueConstraint("compiler_unit_id", "run_id", "snapshot_id"),
        UniqueConstraint("compiler_unit_id", "run_id", "snapshot_id", "conversation_id"),
        UniqueConstraint("run_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    compiler_unit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    available_after_session: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    builder_version: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class CompilerUnitMemoryModel(Base):
    __tablename__ = "compiler_unit_memories"
    __table_args__ = (
        PrimaryKeyConstraint("compiler_unit_id", "memory_id"),
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id", "snapshot_id", "conversation_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
                f"{RUNTIME_SCHEMA}.compiler_units.snapshot_id",
                f"{RUNTIME_SCHEMA}.compiler_units.conversation_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "memory_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_units.snapshot_id",
                f"{RUNTIME_SCHEMA}.memory_units.conversation_id",
                f"{RUNTIME_SCHEMA}.memory_units.memory_id",
            ],
        ),
        UniqueConstraint("compiler_unit_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            f"role IN {sql_text_values(COMPILATION_UNIT_MEMORY_ROLE_VALUES)}",
            name="role_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    memory_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class CompilerUnitStageResultModel(Base):
    __tablename__ = "compiler_unit_stage_results"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        UniqueConstraint("compiler_unit_id", "stage", "attempt_number"),
        Index(
            "uq_compiler_unit_stage_success",
            "compiler_unit_id",
            "stage",
            unique=True,
            postgresql_where=sql_text("status = 'success'"),
        ),
        CheckConstraint(
            f"stage IN {sql_text_values(COMPILATION_STAGE_VALUES)}",
            name="stage_values",
        ),
        CheckConstraint(f"status IN {sql_text_values(TRACE_STATUS_VALUES)}", name="status_values"),
        CheckConstraint("attempt_number > 0", name="attempt_number_positive"),
        CheckConstraint(
            "produced_count >= 0 AND accepted_count >= 0 AND rejected_count >= 0",
            name="counts_nonnegative",
        ),
        CheckConstraint(
            "accepted_count + rejected_count <= produced_count",
            name="counts_bounded",
        ),
        CheckConstraint("elapsed_ms >= 0", name="elapsed_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    stage_result_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    produced_count: Mapped[int] = mapped_column(Integer, nullable=False)
    accepted_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    result_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankQuestionModel(Base):
    __tablename__ = "qbank_questions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id", "snapshot_id", "conversation_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
                f"{RUNTIME_SCHEMA}.compiler_units.snapshot_id",
                f"{RUNTIME_SCHEMA}.compiler_units.conversation_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "created_after_session"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_sessions.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.conversation_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.session_id",
            ],
        ),
        ForeignKeyConstraint(
            ["opportunity_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.questioner_opportunities.opportunity_id",
                f"{RUNTIME_SCHEMA}.questioner_opportunities.run_id",
            ],
            use_alter=True,
            name="fk_qbank_questions_opportunity",
        ),
        ForeignKeyConstraint(
            ["supersedes_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
            use_alter=True,
            name="fk_qbank_questions_supersedes",
        ),
        UniqueConstraint("qbank_id", "run_id"),
        UniqueConstraint(
            "qbank_id",
            "run_id",
            "compiler_unit_id",
            name="uq_qbank_questions_run_unit",
        ),
        UniqueConstraint(
            "compiler_unit_id",
            "generation_rank",
            "candidate_version",
            name="unit_rank_version",
        ),
        CheckConstraint(
            f"question_type IN {sql_text_values(QUESTION_TYPE_VALUES)}",
            name="question_type_values",
        ),
        CheckConstraint(
            "reasoning_type IS NULL OR "
            f"reasoning_type IN {sql_text_values(QUESTION_REASONING_TYPE_VALUES)}",
            name="reasoning_type_values",
        ),
        CheckConstraint("hop_count > 0", name="hop_count_positive"),
        CheckConstraint("generation_rank > 0", name="generation_rank_positive"),
        CheckConstraint(
            "candidate_origin IN ('generated', 'revised')",
            name="candidate_origin_values",
        ),
        CheckConstraint(
            "candidate_version IN (0, 1) AND "
            "((candidate_origin = 'generated' AND candidate_version = 0 "
            "AND supersedes_qbank_id IS NULL) OR "
            "(candidate_origin = 'revised' AND candidate_version = 1 "
            "AND supersedes_qbank_id IS NOT NULL))",
            name="candidate_version_lineage",
        ),
        CheckConstraint(
            f"required_modalities <@ {sql_text_array(MODALITY_VALUES)}",
            name="required_modalities_values",
        ),
        CheckConstraint("cardinality(required_modalities) > 0", name="modalities_nonempty"),
        CheckConstraint(
            "answer_type IS NULL OR answer_type IN "
            "('entity','value','count','list','boolean','event','refusal')",
            name="answer_type_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    qbank_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            f"{RUNTIME_SCHEMA}.questioner_generations.generation_id",
            use_alter=True,
            name="fk_qbank_questions_generation",
        )
    )
    producer_model_call_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id")
    )
    question_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    created_after_session: Mapped[str] = mapped_column(String(256), nullable=False)
    canonical_question: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(64), nullable=False)
    answer_type: Mapped[str | None] = mapped_column(String(32))
    reasoning_type: Mapped[str | None] = mapped_column(String(32))
    candidate_content_hash: Mapped[str | None] = mapped_column(String(128))
    opportunity_id: Mapped[str | None] = mapped_column(String(128))
    candidate_origin: Mapped[str] = mapped_column(String(16), nullable=False)
    candidate_version: Mapped[int] = mapped_column(Integer, nullable=False)
    supersedes_qbank_id: Mapped[str | None] = mapped_column(String(128))
    route_class: Mapped[str] = mapped_column(
        String(16), nullable=False, default="answerable"
    )
    expected_answer: Mapped[str | None] = mapped_column(Text)
    required_modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    evidence_requirement: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    hop_count: Mapped[int] = mapped_column(Integer, nullable=False)
    generation_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    questioner_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    accepted_legacy: Mapped[bool | None] = mapped_column("accepted", Boolean)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    search_document: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(canonical_question, ''))", persisted=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankParaphraseModel(Base):
    __tablename__ = "qbank_paraphrases"
    __table_args__ = (
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("qbank_id", "ordinal"),
        UniqueConstraint("qbank_id", "paraphrase_hash"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    paraphrase_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    paraphrase: Mapped[str] = mapped_column(Text, nullable=False)
    paraphrase_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    search_document: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(paraphrase, ''))", persisted=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankEvidenceRefModel(Base):
    __tablename__ = "qbank_evidence_refs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        CheckConstraint(
            f"role IN {sql_text_values(QBANK_EVIDENCE_ROLE_VALUES)}", name="role_values"
        ),
        CheckConstraint(
            f"evidence_type IN {sql_text_values(EVIDENCE_TYPE_VALUES)}",
            name="evidence_type_values",
        ),
        CheckConstraint(
            "num_nonnulls(memory_id, image_id, session_id, caption_id) = 1",
            name="one_typed_reference",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    evidence_ref_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(256), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    memory_id: Mapped[str | None] = mapped_column(String(256))
    image_id: Mapped[str | None] = mapped_column(String(256))
    session_id: Mapped[str | None] = mapped_column(String(256))
    caption_id: Mapped[str | None] = mapped_column(String(256))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class MemoryProgramModel(Base):
    __tablename__ = "memory_programs"
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
        UniqueConstraint("plan_id", "run_id"),
        UniqueConstraint(
            "plan_id",
            "run_id",
            "compiler_unit_id",
            name="uq_memory_programs_run_unit",
        ),
        UniqueConstraint("qbank_id", "round_id"),
        CheckConstraint("round_id >= 0", name="round_id_nonnegative"),
        CheckConstraint(
            f"required_modalities <@ {sql_text_array(MODALITY_VALUES)}",
            name="required_modalities_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    plan_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    producer_model_call_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id")
    )
    operator_set_version: Mapped[str] = mapped_column(String(128), nullable=False)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    fallback_operator: Mapped[str | None] = mapped_column(String(128))
    required_modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    planner_rationale_short: Mapped[str] = mapped_column(Text, nullable=False)
    planner_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    program_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    accepted_legacy: Mapped[bool] = mapped_column("accepted", Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramStepModel(Base):
    __tablename__ = "program_steps"
    __table_args__ = (
        PrimaryKeyConstraint("plan_id", "step_id"),
        ForeignKeyConstraint(
            ["plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        CheckConstraint("step_id >= 0", name="step_id_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    op_name: Mapped[str] = mapped_column(String(128), nullable=False)
    args_json: Mapped[JsonValue] = mapped_column("args", JSONB, nullable=False)
    expected_output_type: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramExpectedEvidenceRefModel(Base):
    __tablename__ = "program_expected_evidence_refs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["plan_id", "step_id"],
            [
                f"{RUNTIME_SCHEMA}.program_steps.plan_id",
                f"{RUNTIME_SCHEMA}.program_steps.step_id",
            ],
        ),
        UniqueConstraint(
            "plan_id",
            "step_id",
            "ordinal",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "num_nonnulls(memory_id, image_id, session_id, caption_id) = 1",
            name="one_typed_reference",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    evidence_ref_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    step_id: Mapped[int | None] = mapped_column(Integer)
    evidence_id: Mapped[str] = mapped_column(String(256), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    memory_id: Mapped[str | None] = mapped_column(String(256))
    image_id: Mapped[str | None] = mapped_column(String(256))
    session_id: Mapped[str | None] = mapped_column(String(256))
    caption_id: Mapped[str | None] = mapped_column(String(256))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramExecutionModel(Base):
    __tablename__ = "program_executions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["plan_id", "source_compiler_run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "source_compiler_run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.run_queries.run_id",
                f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.run_queries.query_id",
            ],
        ),
        UniqueConstraint("execution_id", "run_id"),
        CheckConstraint(
            f"purpose IN {sql_text_values(OPERATOR_EXECUTION_PURPOSE_VALUES)}",
            name="purpose_values",
        ),
        CheckConstraint(f"status IN {sql_text_values(TRACE_STATUS_VALUES)}", name="status_values"),
        CheckConstraint(
            f"control_action IN {sql_text_values(OPERATOR_CONTROL_ACTION_VALUES)}",
            name="control_action_values",
        ),
        CheckConstraint("attempt_number > 0", name="attempt_number_positive"),
        CheckConstraint("elapsed_ms >= 0", name="elapsed_nonnegative"),
        CheckConstraint(
            "(purpose = 'compiler_dry_run' AND compiler_unit_id IS NOT NULL "
            "AND query_id IS NULL) OR "
            "(purpose = 'query_time' AND query_id IS NOT NULL)",
            name="purpose_scope_present",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    execution_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    source_compiler_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str | None] = mapped_column(String(128))
    query_id: Mapped[str | None] = mapped_column(String(256))
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    bindings_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    control_action: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramExecutionStepModel(Base):
    __tablename__ = "program_execution_steps"
    __table_args__ = (
        PrimaryKeyConstraint("execution_id", "step_id"),
        ForeignKeyConstraint(
            ["execution_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.program_executions.execution_id",
                f"{RUNTIME_SCHEMA}.program_executions.run_id",
            ],
        ),
        CheckConstraint("step_id >= 0", name="step_id_nonnegative"),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_nonnegative"),
        CheckConstraint(f"status IN {sql_text_values(TRACE_STATUS_VALUES)}", name="status_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    execution_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str | None] = mapped_column(String(128))
    query_id: Mapped[str | None] = mapped_column(String(256))
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    op_name: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    output_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    output_json: Mapped[JsonValue] = mapped_column("output", JSONB, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramExecutionEvidenceRefModel(Base):
    __tablename__ = "program_execution_evidence_refs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["execution_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.program_executions.execution_id",
                f"{RUNTIME_SCHEMA}.program_executions.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["execution_id", "step_id"],
            [
                f"{RUNTIME_SCHEMA}.program_execution_steps.execution_id",
                f"{RUNTIME_SCHEMA}.program_execution_steps.step_id",
            ],
        ),
        UniqueConstraint("execution_id", "rank"),
        CheckConstraint("rank > 0", name="rank_positive"),
        CheckConstraint(
            f"evidence_type IN {sql_text_values(EVIDENCE_TYPE_VALUES)}",
            name="evidence_type_values",
        ),
        CheckConstraint(
            "num_nonnulls(memory_id, image_id, session_id, caption_id) = 1",
            name="one_typed_reference",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    evidence_ref_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    execution_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(256), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    memory_id: Mapped[str | None] = mapped_column(String(256))
    image_id: Mapped[str | None] = mapped_column(String(256))
    session_id: Mapped[str | None] = mapped_column(String(256))
    caption_id: Mapped[str | None] = mapped_column(String(256))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score_json: Mapped[JsonValue | None] = mapped_column("score", JSONB)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class DoubterFeedbackModel(Base):
    __tablename__ = "doubter_feedback"
    __table_args__ = (
        ForeignKeyConstraint(
            ["plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        UniqueConstraint("qbank_id", "round_id"),
        UniqueConstraint("feedback_id", "run_id"),
        CheckConstraint("round_id >= 0", name="round_id_nonnegative"),
        CheckConstraint(
            f"evidence_sufficiency IN {sql_text_values(EVIDENCE_SUFFICIENCY_VALUES)}",
            name="evidence_sufficiency_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    feedback_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    program_execution_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.program_executions.execution_id")
    )
    producer_model_call_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id")
    )
    feedback_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    verifier_pass: Mapped[bool] = mapped_column(Boolean, nullable=False)
    verifier_score: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_sufficiency: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    suggested_revision: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    missing_modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    should_abstain: Mapped[bool] = mapped_column(Boolean, nullable=False)
    raw_response_asset_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankDecisionModel(Base):
    __tablename__ = "qbank_decisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("qbank_id"),
        CheckConstraint(
            "(accepted AND final_plan_id IS NOT NULL AND verifier_pass) OR "
            "(NOT accepted AND NOT verifier_pass)",
            name="decision_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    final_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.memory_programs.plan_id")
    )
    final_feedback_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.doubter_feedback.feedback_id")
    )
    reason_code: Mapped[str] = mapped_column(String(128), nullable=False)
    reason_summary: Mapped[str | None] = mapped_column(Text)
    verifier_pass: Mapped[bool] = mapped_column(Boolean, nullable=False)
    verifier_score: Mapped[float | None] = mapped_column(Float)
    decision_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class CompilerStageJobModel(Base):
    __tablename__ = "compiler_stage_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        UniqueConstraint("run_id", "compiler_unit_id", "stage", "round_id"),
        UniqueConstraint("stage_job_id", "run_id", "compiler_unit_id"),
        CheckConstraint("stage IN ('planner','dry_run')", name="stage_values"),
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="status_values",
        ),
        CheckConstraint("round_id >= 0", name="round_id_nonnegative"),
        CheckConstraint("attempt_count >= 0", name="attempt_count_nonnegative"),
        CheckConstraint("max_attempts > 0", name="max_attempts_positive"),
        {"schema": RUNTIME_SCHEMA},
    )

    stage_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
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


class PlannerGenerationBatchModel(Base):
    __tablename__ = "planner_generation_batches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["stage_job_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_stage_jobs.stage_job_id",
                f"{RUNTIME_SCHEMA}.compiler_stage_jobs.run_id",
                f"{RUNTIME_SCHEMA}.compiler_stage_jobs.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["model_call_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
            ],
        ),
        UniqueConstraint("run_id", "compiler_unit_id", "batch_ordinal"),
        UniqueConstraint("planner_batch_id", "run_id", "compiler_unit_id"),
        CheckConstraint("batch_ordinal >= 0", name="batch_ordinal_nonnegative"),
        CheckConstraint(
            "parsed_count >= 0 AND invalid_count >= 0",
            name="counts_nonnegative",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    planner_batch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stage_job_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    batch_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_qbank_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    parsed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    invalid_count: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PlannerRequirementContractModel(Base):
    __tablename__ = "planner_requirement_contracts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["qbank_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.compiler_unit_id",
            ],
        ),
        UniqueConstraint("qbank_id"),
        UniqueConstraint("requirement_contract_id", "run_id", "compiler_unit_id"),
        CheckConstraint("minimum_evidence_groups > 0", name="evidence_groups_positive"),
        CheckConstraint(
            "cardinality(required_modalities) > 0 AND required_modalities <@ "
            "ARRAY['text','caption','image','raw_image']::text[]",
            name="modalities_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    requirement_contract_id: Mapped[str] = mapped_column(
        String(128), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    required_modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    minimum_evidence_groups: Mapped[int] = mapped_column(Integer, nullable=False)
    required_capabilities: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False
    )
    allowed_capabilities: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    fallback_required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    refusal_only: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PlannerCandidateDraftModel(Base):
    __tablename__ = "planner_candidate_drafts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["planner_batch_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.planner_generation_batches.planner_batch_id",
                f"{RUNTIME_SCHEMA}.planner_generation_batches.run_id",
                f"{RUNTIME_SCHEMA}.planner_generation_batches.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["requirement_contract_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.planner_requirement_contracts.requirement_contract_id",
                f"{RUNTIME_SCHEMA}.planner_requirement_contracts.run_id",
                f"{RUNTIME_SCHEMA}.planner_requirement_contracts.compiler_unit_id",
            ],
        ),
        UniqueConstraint("planner_batch_id", "item_ordinal"),
        UniqueConstraint("qbank_id"),
        CheckConstraint("item_ordinal >= 0", name="item_ordinal_nonnegative"),
        CheckConstraint(
            "(parsed AND program_hash IS NOT NULL) OR "
            "(NOT parsed AND program_hash IS NULL)",
            name="parsed_program_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    planner_candidate_draft_id: Mapped[str] = mapped_column(
        String(128), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    planner_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    requirement_contract_id: Mapped[str] = mapped_column(String(128), nullable=False)
    item_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    item_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    parsed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    program_hash: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PlannerCandidateOutcomeModel(Base):
    __tablename__ = "planner_candidate_outcomes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["qbank_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["planner_batch_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.planner_generation_batches.planner_batch_id",
                f"{RUNTIME_SCHEMA}.planner_generation_batches.run_id",
                f"{RUNTIME_SCHEMA}.planner_generation_batches.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["plan_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
                f"{RUNTIME_SCHEMA}.memory_programs.compiler_unit_id",
            ],
        ),
        UniqueConstraint("qbank_id"),
        {"schema": RUNTIME_SCHEMA},
    )

    planner_outcome_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    planner_batch_id: Mapped[str | None] = mapped_column(String(128))
    plan_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(128), nullable=False)
    reason_summary: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PlannerBatchProgramModel(Base):
    __tablename__ = "planner_batch_programs"
    __table_args__ = (
        PrimaryKeyConstraint("planner_batch_id", "plan_id"),
        ForeignKeyConstraint(
            ["planner_batch_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.planner_generation_batches.planner_batch_id",
                f"{RUNTIME_SCHEMA}.planner_generation_batches.run_id",
                f"{RUNTIME_SCHEMA}.planner_generation_batches.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["plan_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
                f"{RUNTIME_SCHEMA}.memory_programs.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.compiler_unit_id",
            ],
        ),
        UniqueConstraint("qbank_id"),
        {"schema": RUNTIME_SCHEMA},
    )

    planner_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramValidationResultModel(Base):
    __tablename__ = "program_validation_results"
    __table_args__ = (
        ForeignKeyConstraint(
            ["plan_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
            ],
        ),
        UniqueConstraint("plan_id", "validation_kind"),
        UniqueConstraint("validation_result_id", "run_id"),
        {"schema": RUNTIME_SCHEMA},
    )

    validation_result_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    validation_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    validator_version: Mapped[str] = mapped_column(String(128), nullable=False)
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    issue_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramValidationIssueModel(Base):
    __tablename__ = "program_validation_issues"
    __table_args__ = (
        ForeignKeyConstraint(
            ["validation_result_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.program_validation_results.validation_result_id",
                f"{RUNTIME_SCHEMA}.program_validation_results.run_id",
            ],
        ),
        UniqueConstraint("validation_result_id", "ordinal"),
        {"schema": RUNTIME_SCHEMA},
    )

    validation_issue_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    validation_result_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    step_id: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class MemoryOperatorIndexManifestModel(Base):
    __tablename__ = "memory_operator_index_manifests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_runs.run_id",
                f"{RUNTIME_SCHEMA}.compiler_runs.snapshot_id",
            ],
        ),
        UniqueConstraint("run_id"),
        CheckConstraint("dimension = 4096", name="dimension_4096"),
        CheckConstraint("distance_metric = 'cosine'", name="distance_metric_cosine"),
        CheckConstraint("status IN ('building','ready','failed')", name="status_values"),
        CheckConstraint("text_collection_id != visual_collection_id", name="collections_distinct"),
        {"schema": RUNTIME_SCHEMA},
    )

    memory_index_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    selection_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    representation_version: Mapped[str] = mapped_column(String(128), nullable=False)
    text_collection_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embedding_collections.collection_id"), nullable=False
    )
    visual_collection_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embedding_collections.collection_id"), nullable=False
    )
    text_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    text_model_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    multimodal_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    multimodal_model_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramQueryEmbeddingBatchModel(Base):
    __tablename__ = "program_query_embedding_batches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_call_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
            ],
        ),
        UniqueConstraint("run_id", "channel", "batch_ordinal"),
        UniqueConstraint("query_embedding_batch_id", "run_id"),
        CheckConstraint("dimension = 4096", name="dimension_4096"),
        {"schema": RUNTIME_SCHEMA},
    )

    query_embedding_batch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.compiler_runs.run_id"), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    batch_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    qbank_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramQueryEmbeddingModel(Base):
    __tablename__ = "program_query_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_embedding_batch_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.program_query_embedding_batches.query_embedding_batch_id",
                f"{RUNTIME_SCHEMA}.program_query_embedding_batches.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("run_id", "qbank_id", "channel"),
        {"schema": RUNTIME_SCHEMA},
    )

    program_query_embedding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_embedding_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    embedding_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embeddings.embedding_id"), nullable=False
    )
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramDryRunAssessmentModel(Base):
    __tablename__ = "program_dry_run_assessments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["qbank_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["plan_id", "run_id", "compiler_unit_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_programs.plan_id",
                f"{RUNTIME_SCHEMA}.memory_programs.run_id",
                f"{RUNTIME_SCHEMA}.memory_programs.compiler_unit_id",
            ],
        ),
        ForeignKeyConstraint(
            ["execution_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.program_executions.execution_id",
                f"{RUNTIME_SCHEMA}.program_executions.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        UniqueConstraint("plan_id"),
        UniqueConstraint("assessment_id", "run_id"),
        CheckConstraint(
            "NOT ((metadata->>'assessment_policy') = "
            "'program-dry-run-assessment-v3' "
            "AND expected_group_count > 0 "
            "AND COALESCE(expected_evidence_recall, 0) = 0 "
            "AND evidence_sufficient)",
            name="expected_recall_sufficient",
        ),
        CheckConstraint(
            "NOT context_evidence_contributed OR context_operator_used",
            name="context_requires_operator",
        ),
        CheckConstraint(
            "NOT temporal_evidence_contributed OR temporal_operator_used",
            name="temporal_requires_operator",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    assessment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(128), nullable=False)
    execution_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    control_action: Mapped[str] = mapped_column(String(16), nullable=False)
    execution_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    nonempty_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False)
    evidence_sufficient: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    matched_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_evidence_recall: Mapped[float | None] = mapped_column(Float)
    expected_evidence_complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    modality_coverage: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    raw_image_reached: Mapped[bool] = mapped_column(Boolean, nullable=False)
    context_operator_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    temporal_operator_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    context_evidence_contributed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    temporal_evidence_contributed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    successful_step_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    required_modality_sufficient: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    discriminative_group_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    matched_discriminative_group_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    contextual_match_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    pre_control_sufficient: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    fallback_triggered: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    abstain_triggered: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    semantic_answer_support: Mapped[str] = mapped_column(
        String(32), nullable=False, default="not_evaluated"
    )
    missing_evidence_group_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(128))
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ProgramDryRunEvidenceMatchModel(Base):
    __tablename__ = "program_dry_run_evidence_matches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["assessment_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.program_dry_run_assessments.assessment_id",
                f"{RUNTIME_SCHEMA}.program_dry_run_assessments.run_id",
            ],
        ),
        UniqueConstraint("assessment_id", "expected_evidence_id", "actual_evidence_id"),
        {"schema": RUNTIME_SCHEMA},
    )

    match_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    assessment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    expected_evidence_id: Mapped[str] = mapped_column(String(256), nullable=False)
    actual_evidence_id: Mapped[str] = mapped_column(String(256), nullable=False)
    expected_evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actual_evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    match_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    evidence_group_id: Mapped[str | None] = mapped_column(String(256))
    direct_match: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    derived_match: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    contextual_only: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    discriminative_match: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


Index(
    "ix_qbank_questions_search_document",
    QBankQuestionModel.search_document,
    postgresql_using="gin",
)
Index(
    "ix_qbank_paraphrases_search_document",
    QBankParaphraseModel.search_document,
    postgresql_using="gin",
)
