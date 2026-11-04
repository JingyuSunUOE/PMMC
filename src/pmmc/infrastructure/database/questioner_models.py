"""SQLAlchemy models for Questioner production and QBank coverage evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import (
    CANDIDATE_GATE_STATUS_VALUES,
    CANDIDATE_REJECTION_CODE_VALUES,
    COMPILATION_UNIT_RELATION_TYPE_VALUES,
    COVERAGE_GATE_MODE_VALUES,
    DATASET_VALUES,
    EMBEDDING_COLLECTION_STATUS_VALUES,
    EVALUATION_SCHEMA,
    EVIDENCE_GRANULARITY_VALUES,
    EVIDENCE_MEMBER_TYPE_VALUES,
    QBANK_COVERAGE_LABEL_VALUES,
    QBANK_COVERAGE_STATUS_VALUES,
    QBANK_MATCH_CHANNEL_VALUES,
    QUESTION_OPPORTUNITY_TYPE_VALUES,
    QUESTIONER_PARSE_STATUS_VALUES,
    QUESTIONER_VERIFICATION_REASON_VALUES,
    QUESTIONER_VERIFICATION_VERDICT_VALUES,
    RUNTIME_SCHEMA,
    sql_text_array,
    sql_text_values,
)

JsonValue = dict[str, Any]


class QuestionOpportunitySetModel(Base):
    __tablename__ = "questioner_opportunity_sets"
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
        UniqueConstraint("compiler_unit_id"),
        UniqueConstraint("opportunity_set_id", "run_id"),
        CheckConstraint(
            "opportunity_count BETWEEN 1 AND 24", name="opportunity_count_values"
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    opportunity_set_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    builder_version: Mapped[str] = mapped_column(String(128), nullable=False)
    opportunity_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionOpportunityModel(Base):
    __tablename__ = "questioner_opportunities"
    __table_args__ = (
        ForeignKeyConstraint(
            ["opportunity_set_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.questioner_opportunity_sets.opportunity_set_id",
                f"{RUNTIME_SCHEMA}.questioner_opportunity_sets.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id", "snapshot_id", "conversation_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
                f"{RUNTIME_SCHEMA}.compiler_units.snapshot_id",
                f"{RUNTIME_SCHEMA}.compiler_units.conversation_id",
            ],
        ),
        UniqueConstraint("opportunity_set_id", "ordinal"),
        UniqueConstraint("opportunity_id", "run_id"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            f"opportunity_type IN {sql_text_values(QUESTION_OPPORTUNITY_TYPE_VALUES)}",
            name="opportunity_type_values",
        ),
        CheckConstraint(
            "cardinality(allowed_question_types) > 0 AND allowed_question_types <@ "
            "ARRAY['textual_fact','visual_detail','image_text_binding','visual_search',"
            "'multi_hop','temporal','knowledge_update','conflict_detection',"
            "'answer_refusal','personal_preference']::text[]",
            name="question_types_values",
        ),
        CheckConstraint(
            "cardinality(allowed_reasoning_types) > 0 AND allowed_reasoning_types <@ "
            "ARRAY['single_hop','multi_hop','temporal','update','conflict','refusal']::text[]",
            name="reasoning_types_values",
        ),
        CheckConstraint(
            "cardinality(required_modalities) > 0 AND required_modalities <@ "
            "ARRAY['text','image']::text[]",
            name="modalities_values",
        ),
        CheckConstraint("minimum_evidence_groups > 0", name="evidence_groups_positive"),
        {"schema": RUNTIME_SCHEMA},
    )

    opportunity_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    opportunity_set_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    opportunity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    allowed_question_types: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    allowed_reasoning_types: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    required_modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    minimum_evidence_groups: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    relation_link_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    speaker_scope: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    evidence_group_ids: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    relation_content_hash: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionOpportunityEvidenceRefModel(Base):
    __tablename__ = "questioner_opportunity_evidence_refs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["opportunity_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.questioner_opportunities.opportunity_id",
                f"{RUNTIME_SCHEMA}.questioner_opportunities.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        UniqueConstraint("opportunity_id", "ordinal"),
        UniqueConstraint("opportunity_id", "alias"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            "evidence_type IN ('memory','image','session','caption')",
            name="evidence_type_values",
        ),
        CheckConstraint(
            "(evidence_type = 'memory' AND memory_id IS NOT NULL AND image_id IS NULL "
            "AND session_id IS NULL AND caption_id IS NULL) OR "
            "(evidence_type = 'caption' AND memory_id IS NOT NULL AND image_id IS NULL "
            "AND session_id IS NULL) OR "
            "(evidence_type = 'image' AND image_id IS NOT NULL AND memory_id IS NULL "
            "AND session_id IS NULL AND caption_id IS NULL) OR "
            "(evidence_type = 'session' AND session_id IS NOT NULL AND memory_id IS NULL "
            "AND image_id IS NULL AND caption_id IS NULL)",
            name="evidence_lineage_values",
        ),
        CheckConstraint(
            "(evidence_type IN ('memory','caption') AND evidence_id = memory_id) OR "
            "(evidence_type = 'image' AND evidence_id = image_id) OR "
            "(evidence_type = 'session' AND evidence_id = session_id)",
            name="evidence_id_matches_lineage",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    opportunity_evidence_ref_id: Mapped[str] = mapped_column(
        String(128), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    opportunity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    alias: Mapped[str] = mapped_column(String(32), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(256), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    memory_id: Mapped[str | None] = mapped_column(String(256))
    image_id: Mapped[str | None] = mapped_column(String(256))
    session_id: Mapped[str | None] = mapped_column(String(256))
    caption_id: Mapped[str | None] = mapped_column(String(256))
    evidence_group_id: Mapped[str] = mapped_column(String(256), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class CompilerUnitLinkModel(Base):
    __tablename__ = "compiler_unit_links"
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
            ["compiler_unit_id", "anchor_memory_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_unit_memories.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_unit_memories.memory_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "support_memory_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_unit_memories.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_unit_memories.memory_id",
            ],
        ),
        UniqueConstraint("compiler_unit_id", "ordinal"),
        UniqueConstraint(
            "compiler_unit_id",
            "anchor_memory_id",
            "support_memory_id",
            "relation_type",
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            "anchor_memory_id != support_memory_id", name="memories_distinct"
        ),
        CheckConstraint(
            f"relation_type IN {sql_text_values(COMPILATION_UNIT_RELATION_TYPE_VALUES)}",
            name="relation_type_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    link_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    anchor_memory_id: Mapped[str] = mapped_column(String(256), nullable=False)
    support_memory_id: Mapped[str] = mapped_column(String(256), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    relation_key_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionerGenerationModel(Base):
    __tablename__ = "questioner_generations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["job_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.experiment_jobs.job_id",
                f"{RUNTIME_SCHEMA}.experiment_jobs.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["model_call_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
            ],
        ),
        UniqueConstraint("compiler_unit_id", "attempt_number"),
        UniqueConstraint("model_call_id"),
        UniqueConstraint("generation_id", "run_id"),
        CheckConstraint("attempt_number > 0", name="attempt_number_positive"),
        CheckConstraint(
            f"parse_status IN {sql_text_values(QUESTIONER_PARSE_STATUS_VALUES)}",
            name="parse_status_values",
        ),
        CheckConstraint(
            "raw_candidate_count >= 0 AND valid_candidate_count >= 0 "
            "AND invalid_candidate_count >= 0",
            name="candidate_counts_nonnegative",
        ),
        CheckConstraint(
            "valid_candidate_count + invalid_candidate_count = raw_candidate_count",
            name="candidate_counts_reconcile",
        ),
        CheckConstraint("repair_attempt_count >= 0", name="repair_count_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    generation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    job_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    renderer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    output_hash: Mapped[str | None] = mapped_column(String(128))
    parse_status: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    valid_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    invalid_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    repair_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionerAnswerClaimModel(Base):
    __tablename__ = "questioner_answer_claims"
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
            ["generation_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.questioner_generations.generation_id",
                f"{RUNTIME_SCHEMA}.questioner_generations.run_id",
            ],
        ),
        UniqueConstraint("qbank_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            "claim_kind IN ('text','visual')", name="claim_kind_values"
        ),
        CheckConstraint(
            "(claim_kind = 'text' AND support_quote IS NOT NULL "
            "AND quote_verified) OR "
            "(claim_kind = 'visual' AND support_quote IS NULL)",
            name="claim_grounding_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    answer_claim_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    evidence_alias_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(256), nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    support_quote: Mapped[str | None] = mapped_column(Text)
    quote_verified: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCandidateFamilyModel(Base):
    __tablename__ = "qbank_candidate_families"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "conversation_id",
            "normalized_question_hash",
            "answer_claim_hash",
            "requirement_contract_hash",
            name="qbank_candidate_family_identity",
        ),
        ForeignKeyConstraint(
            ["primary_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("family_id", "run_id"),
        CheckConstraint("member_count > 0", name="member_count_positive"),
        {"schema": RUNTIME_SCHEMA},
    )

    family_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    primary_qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    normalized_question_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    answer_claim_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    requirement_contract_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCandidateFamilyMemberModel(Base):
    __tablename__ = "qbank_candidate_family_members"
    __table_args__ = (
        ForeignKeyConstraint(
            ["family_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_candidate_families.family_id",
                f"{RUNTIME_SCHEMA}.qbank_candidate_families.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("family_id", "ordinal"),
        UniqueConstraint("run_id", "qbank_id"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    family_member_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RuntimeStructuredOutputItemDiagnosticModel(Base):
    __tablename__ = "structured_output_item_diagnostics"
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_call_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        UniqueConstraint("model_call_id", "stage", "item_ordinal"),
        CheckConstraint(
            "stage IN ('questioner','questioner_verifier','questioner_revision',"
            "'planner','planner_revision','doubter')",
            name="stage_values",
        ),
        CheckConstraint(
            "status IN ('valid','invalid','repaired','conservative_reject')",
            name="status_values",
        ),
        CheckConstraint("item_ordinal >= 0", name="item_ordinal_nonnegative"),
        CheckConstraint(
            "repair_attempt_count >= 0", name="repair_count_nonnegative"
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    diagnostic_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    semantic_run_id: Mapped[str | None] = mapped_column(String(128))
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    item_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    item_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    repair_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    errors: Mapped[list[JsonValue]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionerVerificationBatchModel(Base):
    __tablename__ = "questioner_verification_batches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["generation_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.questioner_generations.generation_id",
                f"{RUNTIME_SCHEMA}.questioner_generations.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["model_call_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
            ],
        ),
        UniqueConstraint("model_call_id"),
        UniqueConstraint(
            "compiler_unit_id",
            "generation_id",
            "verification_round",
            "scope_qbank_id",
        ),
        UniqueConstraint(
            "verification_batch_id",
            "run_id",
            "compiler_unit_id",
            "generation_id",
            "verification_round",
        ),
        CheckConstraint("verification_round IN (0, 1)", name="round_values"),
        CheckConstraint(
            "input_count >= 0 AND pass_count >= 0 AND revise_count >= 0 "
            "AND reject_count >= 0",
            name="counts_nonnegative",
        ),
        CheckConstraint(
            "pass_count + revise_count + reject_count = input_count",
            name="counts_reconcile",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    verification_batch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    scope_qbank_id: Mapped[str | None] = mapped_column(String(128))
    verification_round: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    input_count: Mapped[int] = mapped_column(Integer, nullable=False)
    pass_count: Mapped[int] = mapped_column(Integer, nullable=False)
    revise_count: Mapped[int] = mapped_column(Integer, nullable=False)
    reject_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionerCandidateVerificationModel(Base):
    __tablename__ = "questioner_candidate_verifications"
    __table_args__ = (
        ForeignKeyConstraint(
            [
                "verification_batch_id",
                "run_id",
                "compiler_unit_id",
                "generation_id",
                "verification_round",
            ],
            [
                f"{RUNTIME_SCHEMA}.questioner_verification_batches.verification_batch_id",
                f"{RUNTIME_SCHEMA}.questioner_verification_batches.run_id",
                f"{RUNTIME_SCHEMA}.questioner_verification_batches.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.questioner_verification_batches.generation_id",
                f"{RUNTIME_SCHEMA}.questioner_verification_batches.verification_round",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["replacement_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
            use_alter=True,
            name="fk_questioner_candidate_verifications_replacement",
        ),
        UniqueConstraint("qbank_id", "verification_round"),
        UniqueConstraint("verification_batch_id", "candidate_alias"),
        CheckConstraint("verification_round IN (0, 1)", name="round_values"),
        CheckConstraint(
            f"verdict IN {sql_text_values(QUESTIONER_VERIFICATION_VERDICT_VALUES)}",
            name="verdict_values",
        ),
        CheckConstraint(
            f"reason_codes <@ {sql_text_array(QUESTIONER_VERIFICATION_REASON_VALUES)}",
            name="reason_code_values",
        ),
        CheckConstraint(
            "(verdict = 'pass' AND cardinality(reason_codes) = 0 "
            "AND replacement_qbank_id IS NULL) OR "
            "(verdict = 'revise' AND cardinality(reason_codes) > 0 "
            "AND replacement_qbank_id IS NOT NULL) OR "
            "(verdict = 'reject' AND cardinality(reason_codes) > 0 "
            "AND replacement_qbank_id IS NULL)",
            name="verdict_payload_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    candidate_verification_id: Mapped[str] = mapped_column(
        String(128), primary_key=True
    )
    verification_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_alias: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_round: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    reason_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    replacement_qbank_id: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionerCandidateSupersessionModel(Base):
    __tablename__ = "questioner_candidate_supersessions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["replacement_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("source_qbank_id"),
        UniqueConstraint("replacement_qbank_id"),
        CheckConstraint("revision_round = 1", name="revision_round_one"),
        CheckConstraint(
            "source_qbank_id != replacement_qbank_id", name="questions_distinct"
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    supersession_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    replacement_qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    revision_round: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QuestionerCandidateGateResultModel(Base):
    __tablename__ = "questioner_candidate_gate_results"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["generation_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.questioner_generations.generation_id",
                f"{RUNTIME_SCHEMA}.questioner_generations.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["duplicate_of_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("qbank_id", "gate_version"),
        CheckConstraint(
            f"status IN {sql_text_values(CANDIDATE_GATE_STATUS_VALUES)}",
            name="status_values",
        ),
        CheckConstraint(
            f"rejection_codes <@ {sql_text_array(CANDIDATE_REJECTION_CODE_VALUES)}",
            name="rejection_code_values",
        ),
        CheckConstraint(
            "(status = 'eligible_for_planning' AND cardinality(rejection_codes) = 0) "
            "OR (status = 'rejected' AND cardinality(rejection_codes) > 0)",
            name="status_rejections_consistent",
        ),
        CheckConstraint(
            "duplicate_of_qbank_id IS NULL OR 'duplicate' = ANY(rejection_codes)",
            name="duplicate_reference_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    gate_result_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    gate_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    rejection_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    duplicate_of_qbank_id: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankDuplicateLinkModel(Base):
    __tablename__ = "qbank_duplicate_links"
    __table_args__ = (
        ForeignKeyConstraint(
            ["duplicate_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["primary_qbank_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id",
                f"{RUNTIME_SCHEMA}.qbank_questions.run_id",
            ],
        ),
        UniqueConstraint("duplicate_qbank_id", "method"),
        CheckConstraint(
            "duplicate_qbank_id != primary_qbank_id", name="questions_distinct"
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    duplicate_link_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    duplicate_qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    primary_qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[JsonValue | None] = mapped_column(JSONB)
    hard_duplicate: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankIndexManifestModel(Base):
    __tablename__ = "qbank_index_manifests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_runs.run_id",
                f"{RUNTIME_SCHEMA}.compiler_runs.snapshot_id",
            ],
        ),
        UniqueConstraint("compiler_run_id"),
        UniqueConstraint("qbank_index_id", "compiler_run_id"),
        CheckConstraint("eligible_question_count >= 0", name="question_count_nonnegative"),
        CheckConstraint("paraphrase_count >= 0", name="paraphrase_count_nonnegative"),
        CheckConstraint("rrf_constant > 0", name="rrf_constant_positive"),
        CheckConstraint("per_channel_top_k > 0", name="top_k_positive"),
        CheckConstraint(
            f"status IN {sql_text_values(EMBEDDING_COLLECTION_STATUS_VALUES)}",
            name="status_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    qbank_index_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    compiler_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    gate_version: Mapped[str] = mapped_column(String(128), nullable=False)
    matcher_version: Mapped[str] = mapped_column(String(128), nullable=False)
    text_question_collection_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embedding_collections.collection_id"), nullable=False
    )
    text_paraphrase_collection_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embedding_collections.collection_id"), nullable=False
    )
    multimodal_question_collection_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embedding_collections.collection_id"), nullable=False
    )
    multimodal_paraphrase_collection_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embedding_collections.collection_id"), nullable=False
    )
    eligible_question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    paraphrase_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rrf_constant: Mapped[int] = mapped_column(Integer, nullable=False)
    per_channel_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageRunModel(Base):
    __tablename__ = "qbank_coverage_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["compiler_run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_runs.run_id",
                f"{RUNTIME_SCHEMA}.compiler_runs.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_index_id", "compiler_run_id"],
            [
                f"{RUNTIME_SCHEMA}.qbank_index_manifests.qbank_index_id",
                f"{RUNTIME_SCHEMA}.qbank_index_manifests.compiler_run_id",
            ],
        ),
        UniqueConstraint("coverage_run_id", "compiler_run_id", "snapshot_id"),
        CheckConstraint(f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"),
        CheckConstraint(
            f"gate_mode IN {sql_text_values(COVERAGE_GATE_MODE_VALUES)}",
            name="gate_mode_values",
        ),
        CheckConstraint(
            f"status IN {sql_text_values(QBANK_COVERAGE_STATUS_VALUES)}",
            name="status_values",
        ),
        CheckConstraint("cardinality(question_budgets) > 0", name="budgets_nonempty"),
        CheckConstraint("cardinality(match_ks) > 0", name="ks_nonempty"),
        CheckConstraint(
            "(status = 'completed' AND completed_at IS NOT NULL) OR "
            "(status != 'completed' AND completed_at IS NULL)",
            name="completion_consistent",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    coverage_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    compiler_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_index_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    protocol_version: Mapped[str] = mapped_column(String(128), nullable=False)
    selection_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    question_budgets: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    match_ks: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    gate_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageQueryModel(Base):
    __tablename__ = "qbank_coverage_queries"
    __table_args__ = (
        PrimaryKeyConstraint("coverage_run_id", "query_id"),
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "cutoff_session_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_sessions.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.conversation_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.session_id",
            ],
        ),
        UniqueConstraint("coverage_run_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint("gold_group_count >= 0", name="gold_count_nonnegative"),
        CheckConstraint(
            "(applicable AND gold_group_count > 0 AND unavailable_reason IS NULL) OR "
            "(NOT applicable AND unavailable_reason IS NOT NULL)",
            name="applicability_consistent",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    coverage_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.qbank_coverage_runs.coverage_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    cutoff_session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    has_query_image: Mapped[bool] = mapped_column(Boolean, nullable=False)
    applicable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gold_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unavailable_reason: Mapped[str | None] = mapped_column(String(256))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageQueryEmbeddingModel(Base):
    __tablename__ = "qbank_coverage_query_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["coverage_run_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.query_id",
            ],
        ),
        UniqueConstraint("coverage_run_id", "query_id", "channel"),
        CheckConstraint(
            f"channel IN {sql_text_values(QBANK_MATCH_CHANNEL_VALUES)}",
            name="channel_values",
        ),
        CheckConstraint(
            "channel IN ('text_semantic','multimodal_semantic')",
            name="semantic_channel",
        ),
        CheckConstraint("dimension > 0", name="dimension_positive"),
        CheckConstraint("vector_dims(embedding) = dimension", name="dimension_matches"),
        {"schema": EVALUATION_SCHEMA},
    )

    query_embedding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    coverage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    model_call_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageCandidateModel(Base):
    __tablename__ = "qbank_coverage_candidates"
    __table_args__ = (
        ForeignKeyConstraint(
            ["coverage_run_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["qbank_id"], [f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id"]
        ),
        UniqueConstraint("coverage_run_id", "query_id", "question_budget", "rank"),
        UniqueConstraint(
            "coverage_run_id", "query_id", "question_budget", "qbank_id"
        ),
        CheckConstraint("question_budget > 0", name="budget_positive"),
        CheckConstraint("rank > 0", name="rank_positive"),
        CheckConstraint(
            "lexical_rank IS NULL OR lexical_rank > 0", name="lexical_rank_positive"
        ),
        CheckConstraint(
            "text_semantic_rank IS NULL OR text_semantic_rank > 0",
            name="text_rank_positive",
        ),
        CheckConstraint(
            "multimodal_semantic_rank IS NULL OR multimodal_semantic_rank > 0",
            name="multimodal_rank_positive",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    coverage_candidate_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    coverage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    question_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    lexical_rank: Mapped[int | None] = mapped_column(Integer)
    text_semantic_rank: Mapped[int | None] = mapped_column(Integer)
    multimodal_semantic_rank: Mapped[int | None] = mapped_column(Integer)
    lexical_score: Mapped[JsonValue | None] = mapped_column(JSONB)
    text_semantic_score: Mapped[JsonValue | None] = mapped_column(JSONB)
    multimodal_semantic_score: Mapped[JsonValue | None] = mapped_column(JSONB)
    final_score: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    matched_gold_group_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    individually_sufficient: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageGoldGroupModel(Base):
    __tablename__ = "qbank_coverage_gold_groups"
    __table_args__ = (
        ForeignKeyConstraint(
            ["coverage_run_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.query_gold.snapshot_id",
                f"{EVALUATION_SCHEMA}.query_gold.query_id",
            ],
        ),
        UniqueConstraint("coverage_run_id", "query_id", "source_gold_id"),
        CheckConstraint(
            f"granularity IN {sql_text_values(EVIDENCE_GRANULARITY_VALUES)}",
            name="granularity_values",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    gold_group_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    coverage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    granularity: Mapped[str] = mapped_column(String(32), nullable=False)
    source_gold_id: Mapped[str] = mapped_column(String(256), nullable=False)
    resolution_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageGoldGroupMemberModel(Base):
    __tablename__ = "qbank_coverage_gold_group_members"
    __table_args__ = (
        PrimaryKeyConstraint("gold_group_id", "member_type", "member_id"),
        CheckConstraint(
            f"member_type IN {sql_text_values(EVIDENCE_MEMBER_TYPE_VALUES)}",
            name="member_type_values",
        ),
        CheckConstraint(
            "match_role IN ('discriminative','contextual')",
            name="match_role_values",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    gold_group_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.qbank_coverage_gold_groups.gold_group_id"),
        nullable=False,
    )
    member_type: Mapped[str] = mapped_column(String(32), nullable=False)
    member_id: Mapped[str] = mapped_column(String(256), nullable=False)
    derived: Mapped[bool] = mapped_column(Boolean, nullable=False)
    match_role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="discriminative"
    )
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageMatchModel(Base):
    __tablename__ = "qbank_coverage_matches"
    __table_args__ = (
        PrimaryKeyConstraint("coverage_candidate_id", "gold_group_id"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        UniqueConstraint("coverage_candidate_id", "ordinal"),
        {"schema": EVALUATION_SCHEMA},
    )

    coverage_candidate_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.qbank_coverage_candidates.coverage_candidate_id"),
        nullable=False,
    )
    gold_group_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.qbank_coverage_gold_groups.gold_group_id"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    matched_member_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)


class QBankCoverageMetricModel(Base):
    __tablename__ = "qbank_coverage_metrics"
    __table_args__ = (
        ForeignKeyConstraint(
            ["coverage_run_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.query_id",
            ],
        ),
        CheckConstraint("question_budget IS NULL OR question_budget > 0", name="budget_positive"),
        CheckConstraint("k IS NULL OR k > 0", name="k_positive"),
        CheckConstraint(
            "(applicable AND value IS NOT NULL AND unavailable_reason IS NULL) OR "
            "(NOT applicable AND value IS NULL AND unavailable_reason IS NOT NULL)",
            name="applicability_consistent",
        ),
        CheckConstraint("value IS NULL OR (value >= 0 AND value <= 1)", name="value_probability"),
        {"schema": EVALUATION_SCHEMA},
    )

    metric_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    coverage_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.qbank_coverage_runs.coverage_run_id"),
        nullable=False,
    )
    query_id: Mapped[str | None] = mapped_column(String(256))
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_version: Mapped[str] = mapped_column(String(128), nullable=False)
    question_budget: Mapped[int | None] = mapped_column(Integer)
    k: Mapped[int | None] = mapped_column(Integer)
    applicable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    unavailable_reason: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageLabelModel(Base):
    __tablename__ = "qbank_coverage_labels"
    __table_args__ = (
        ForeignKeyConstraint(
            ["coverage_run_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.query_id",
            ],
        ),
        CheckConstraint(
            f"category IN {sql_text_values(QBANK_COVERAGE_LABEL_VALUES)}",
            name="category_values",
        ),
        CheckConstraint("question_budget IS NULL OR question_budget > 0", name="budget_positive"),
        CheckConstraint("k IS NULL OR k > 0", name="k_positive"),
        {"schema": EVALUATION_SCHEMA},
    )

    label_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    coverage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    question_budget: Mapped[int | None] = mapped_column(Integer)
    k: Mapped[int | None] = mapped_column(Integer)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankCoverageGateReportModel(Base):
    __tablename__ = "qbank_coverage_gate_reports"
    __table_args__ = (
        UniqueConstraint("coverage_run_id"),
        CheckConstraint(
            f"mode IN {sql_text_values(COVERAGE_GATE_MODE_VALUES)}", name="mode_values"
        ),
        CheckConstraint("query_count >= 0", name="query_count_nonnegative"),
        CheckConstraint("applicable_query_count >= 0", name="applicable_count_nonnegative"),
        CheckConstraint(
            "applicable_query_count <= query_count", name="applicable_count_bounded"
        ),
        CheckConstraint("elapsed_ms >= 0", name="elapsed_nonnegative"),
        CheckConstraint(
            "(mode = 'report_only' AND scientific_thresholds_passed IS NULL) OR "
            "(mode = 'enforce' AND scientific_thresholds_passed IS NOT NULL)",
            name="scientific_result_consistent",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    gate_report_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    coverage_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.qbank_coverage_runs.coverage_run_id"),
        nullable=False,
    )
    gate_version: Mapped[str] = mapped_column(String(128), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    hard_integrity_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    scientific_thresholds_passed: Mapped[bool | None] = mapped_column(Boolean)
    query_count: Mapped[int] = mapped_column(Integer, nullable=False)
    applicable_query_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_summary: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    threshold_summary: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    failure_counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankSemanticMatchRunModel(Base):
    __tablename__ = "qbank_semantic_match_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["coverage_run_id", "compiler_run_id", "snapshot_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_coverage_runs.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_runs.compiler_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_runs.snapshot_id",
            ],
        ),
        UniqueConstraint("coverage_run_id"),
        UniqueConstraint("semantic_run_id", "coverage_run_id"),
        CheckConstraint(
            "status IN ('created','running','completed','failed','cancelled')",
            name="status_values",
        ),
        CheckConstraint("pair_count >= 0", name="pair_count_nonnegative"),
        {"schema": EVALUATION_SCHEMA},
    )

    semantic_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    protocol_version: Mapped[str] = mapped_column(String(128), nullable=False)
    coverage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    judge_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    judge_model_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    renderer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    pair_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankSemanticMatchJobModel(Base):
    __tablename__ = "qbank_semantic_match_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["semantic_run_id", "coverage_run_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_semantic_match_runs.semantic_run_id",
                f"{EVALUATION_SCHEMA}.qbank_semantic_match_runs.coverage_run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["coverage_run_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_coverage_queries.query_id",
            ],
        ),
        UniqueConstraint("semantic_run_id", "query_id", "qbank_id"),
        UniqueConstraint("semantic_job_id", "semantic_run_id"),
        UniqueConstraint(
            "semantic_job_id",
            "semantic_run_id",
            "coverage_run_id",
            "query_id",
            "qbank_id",
        ),
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="status_values",
        ),
        CheckConstraint("attempt_count >= 0", name="attempt_count_nonnegative"),
        CheckConstraint("max_attempts > 0", name="max_attempts_positive"),
        {"schema": EVALUATION_SCHEMA},
    )

    semantic_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    semantic_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    coverage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    pair_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    model_call_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankSemanticMatchAssessmentModel(Base):
    __tablename__ = "qbank_semantic_match_assessments"
    __table_args__ = (
        ForeignKeyConstraint(
            [
                "semantic_job_id",
                "semantic_run_id",
                "coverage_run_id",
                "query_id",
                "qbank_id",
            ],
            [
                f"{EVALUATION_SCHEMA}.qbank_semantic_match_jobs.semantic_job_id",
                f"{EVALUATION_SCHEMA}.qbank_semantic_match_jobs.semantic_run_id",
                f"{EVALUATION_SCHEMA}.qbank_semantic_match_jobs.coverage_run_id",
                f"{EVALUATION_SCHEMA}.qbank_semantic_match_jobs.query_id",
                f"{EVALUATION_SCHEMA}.qbank_semantic_match_jobs.qbank_id",
            ],
        ),
        ForeignKeyConstraint(
            ["model_call_id", "semantic_run_id", "semantic_job_id"],
            [
                f"{EVALUATION_SCHEMA}.model_calls.model_call_id",
                f"{EVALUATION_SCHEMA}.model_calls.semantic_run_id",
                f"{EVALUATION_SCHEMA}.model_calls.semantic_job_id",
            ],
        ),
        UniqueConstraint("semantic_job_id"),
        CheckConstraint("elapsed_ms >= 0", name="elapsed_nonnegative"),
        CheckConstraint(
            "semantic_match = (semantic_dimensions_passed AND evidence_proxy_complete)",
            name="semantic_match_consistent",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    semantic_assessment_id: Mapped[str] = mapped_column(
        String(128), primary_key=True
    )
    semantic_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    semantic_job_id: Mapped[str] = mapped_column(String(128), nullable=False)
    coverage_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dimensions: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    reason_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    semantic_dimensions_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    evidence_proxy_complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    semantic_match: Mapped[bool] = mapped_column(Boolean, nullable=False)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankSemanticMetricModel(Base):
    __tablename__ = "qbank_semantic_match_metrics"
    __table_args__ = (
        ForeignKeyConstraint(
            ["semantic_run_id"],
            [f"{EVALUATION_SCHEMA}.qbank_semantic_match_runs.semantic_run_id"],
        ),
        UniqueConstraint(
            "semantic_run_id",
            "query_id",
            "metric_name",
            "question_budget",
            "k",
        ),
        CheckConstraint("question_budget > 0", name="budget_positive"),
        CheckConstraint("k > 0", name="k_positive"),
        CheckConstraint("value >= 0 AND value <= 1", name="value_probability"),
        {"schema": EVALUATION_SCHEMA},
    )

    semantic_metric_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    semantic_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str | None] = mapped_column(String(256))
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_version: Mapped[str] = mapped_column(String(128), nullable=False)
    question_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    k: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class EvaluationStructuredOutputItemDiagnosticModel(Base):
    __tablename__ = "structured_output_item_diagnostics"
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_call_id", "semantic_run_id"],
            [
                f"{EVALUATION_SCHEMA}.model_calls.model_call_id",
                f"{EVALUATION_SCHEMA}.model_calls.semantic_run_id",
            ],
        ),
        UniqueConstraint("model_call_id", "stage", "item_ordinal"),
        CheckConstraint("stage = 'qbank_semantic_judge'", name="stage_values"),
        CheckConstraint(
            "status IN ('valid','invalid','repaired','conservative_reject')",
            name="status_values",
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    diagnostic_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_unit_id: Mapped[str | None] = mapped_column(String(128))
    semantic_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    item_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    item_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    repair_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    errors: Mapped[list[JsonValue]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
