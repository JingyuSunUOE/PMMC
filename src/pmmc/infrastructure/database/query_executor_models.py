"""SQLAlchemy models for the Step 7 online query executor."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import RUNTIME_SCHEMA

JsonValue = dict[str, Any]


class QueryRuntimeEmbeddingCacheModel(Base):
    __tablename__ = "query_runtime_embedding_cache"
    __table_args__ = (
        CheckConstraint("channel IN ('text','multimodal')", name="channel_values"),
        {"schema": RUNTIME_SCHEMA},
    )
    cache_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    model_entry_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(f"{RUNTIME_SCHEMA}.embeddings.embedding_id"),
        nullable=False,
    )
    collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryRuntimePhaseStateModel(Base):
    __tablename__ = "query_runtime_phase_states"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id", "phase"),
        CheckConstraint("phase IN ('match','answer')", name="phase_values"),
        {"schema": RUNTIME_SCHEMA},
    )
    phase_state_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    payload: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryRuntimeTypedAnswerModel(Base):
    __tablename__ = "query_runtime_typed_answers"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id", "answer_identity_hash"),
        CheckConstraint(
            "terminal_state IN ('answered','abstained','fallback_required')",
            name="terminal_state_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    typed_answer_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    answer_identity_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    answer_model_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    terminal_state: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryExperimentCellResultModel(Base):
    __tablename__ = "query_experiment_cell_results"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id", "cell_id"),
        CheckConstraint(
            "question_budget IN (1,2,4,8) AND revision_depth BETWEEN 0 AND 3",
            name="cell_grid_values",
        ),
        CheckConstraint(
            "terminal_state IN ('answered','abstained','no_answer','technical_failure')",
            name="terminal_state_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    cell_result_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    cell_id: Mapped[str] = mapped_column(String(128), nullable=False)
    treatment: Mapped[str] = mapped_column(String(64), nullable=False)
    question_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    revision_depth: Mapped[int] = mapped_column(Integer, nullable=False)
    terminal_state: Mapped[str] = mapped_column(String(32), nullable=False)
    selected_answer_id: Mapped[str | None] = mapped_column(
        String(128),
        ForeignKey(f"{RUNTIME_SCHEMA}.query_runtime_typed_answers.typed_answer_id"),
    )
    payload: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryExecutorRunModel(Base):
    __tablename__ = "query_executor_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_executor_run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        UniqueConstraint("query_executor_run_id", "snapshot_id"),
        ForeignKeyConstraint(
            ["source_compiler_run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_runs.run_id",
                f"{RUNTIME_SCHEMA}.compiler_runs.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["routing_projection_id", "source_compiler_run_id"],
            [
                f"{RUNTIME_SCHEMA}.routing_projection_manifests.routing_projection_id",
                f"{RUNTIME_SCHEMA}.routing_projection_manifests.run_id",
            ],
        ),
        CheckConstraint(
            "status IN ('created','running','completed','failed','cancelled')",
            name="status_values",
        ),
        CheckConstraint("active_round = 2 AND max_round = 3", name="rounds_frozen"),
        CheckConstraint("question_budget = 8", name="question_budget_frozen"),
        CheckConstraint(
            "selected_query_count > 0 AND matched_count >= 0 AND executed_count >= 0 "
            "AND pmmc_answer_count >= 0 AND fallback_answer_count >= 0 "
            "AND selected_answer_count >= 0 AND failed_count >= 0",
            name="counts_valid",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    query_executor_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_compiler_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    routing_projection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    active_round: Mapped[int] = mapped_column(Integer, nullable=False)
    max_round: Mapped[int] = mapped_column(Integer, nullable=False)
    question_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_query_count: Mapped[int] = mapped_column(Integer, nullable=False)
    pilot_projection: Mapped[bool] = mapped_column(Boolean, nullable=False)
    paper_result: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="created")
    matched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    executed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pmmc_answer_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fallback_answer_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    selected_answer_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryExecutorStageJobModel(Base):
    __tablename__ = "query_executor_stage_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_executor_run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.run_queries.run_id",
                f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.run_queries.query_id",
            ],
        ),
        UniqueConstraint("query_executor_run_id", "query_id", "stage", "round_id"),
        CheckConstraint(
            "stage IN ('embed','match','execute','answer','fallback','select')",
            name="stage_values",
        ),
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="status_values",
        ),
        CheckConstraint("round_id IS NULL OR round_id BETWEEN 0 AND 3", name="round_values"),
        CheckConstraint("attempt_count >= 0 AND max_attempts > 0", name="attempts_valid"),
        {"schema": RUNTIME_SCHEMA},
    )

    query_stage_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    round_id: Mapped[int | None] = mapped_column(Integer)
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
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryEmbeddingBatchModel(Base):
    __tablename__ = "query_embedding_batches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_executor_run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id",
                f"{RUNTIME_SCHEMA}.query_executor_runs.snapshot_id",
            ],
        ),
        UniqueConstraint("query_embedding_batch_id", "query_executor_run_id"),
        UniqueConstraint("query_executor_run_id", "channel", "batch_ordinal"),
        CheckConstraint("channel IN ('text','multimodal')", name="channel_values"),
        CheckConstraint("batch_ordinal >= 0", name="batch_ordinal_nonnegative"),
        CheckConstraint(
            "expected_count > 0 AND committed_count IN (0, expected_count)",
            name="atomic_commit",
        ),
        CheckConstraint(
            "(model_call_id IS NOT NULL AND "
            "NOT COALESCE((metadata->>'coverage_projection')::boolean, false)) OR "
            "(model_call_id IS NULL AND "
            "COALESCE((metadata->>'coverage_projection')::boolean, false) AND "
            "COALESCE(metadata->>'source_evaluation_model_call_id', '') <> '')",
            name="runtime_or_coverage_lineage",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    query_embedding_batch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_call_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id"), nullable=True
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    batch_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    query_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    expected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    committed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryEmbeddingModel(Base):
    __tablename__ = "query_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_embedding_batch_id", "query_executor_run_id"],
            [
                f"{RUNTIME_SCHEMA}.query_embedding_batches.query_embedding_batch_id",
                f"{RUNTIME_SCHEMA}.query_embedding_batches.query_executor_run_id",
            ],
        ),
        UniqueConstraint("query_executor_run_id", "query_id", "channel"),
        CheckConstraint("channel IN ('text','multimodal')", name="channel_values"),
        CheckConstraint("vector_dimension = 4096", name="dimension_frozen"),
        {"schema": RUNTIME_SCHEMA},
    )

    query_embedding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_embedding_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_executor_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    embedding_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embeddings.embedding_id"), nullable=False
    )
    collection_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.embedding_collections.collection_id"), nullable=False
    )
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankMatchBatchModel(Base):
    __tablename__ = "qbank_match_batches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_executor_run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.run_queries.run_id",
                f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.run_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["routing_projection_id", "source_compiler_run_id"],
            [
                f"{RUNTIME_SCHEMA}.routing_projection_manifests.routing_projection_id",
                f"{RUNTIME_SCHEMA}.routing_projection_manifests.run_id",
            ],
        ),
        UniqueConstraint("query_executor_run_id", "query_id", "active_round", "question_budget"),
        CheckConstraint("active_round BETWEEN 0 AND 3", name="round_values"),
        CheckConstraint("question_budget IN (1,2,4,8)", name="budget_values"),
        CheckConstraint("requested_top_k = 50", name="top_k_frozen"),
        {"schema": RUNTIME_SCHEMA},
    )

    match_batch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_compiler_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    routing_projection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    cutoff_session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    active_round: Mapped[int] = mapped_column(Integer, nullable=False)
    question_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    requested_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    channel_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fused_count: Mapped[int] = mapped_column(Integer, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankMatchChannelCandidateModel(Base):
    __tablename__ = "qbank_match_channel_candidates"
    __table_args__ = (
        UniqueConstraint("match_batch_id", "route_class", "channel", "rank"),
        UniqueConstraint("match_batch_id", "channel", "qbank_id"),
        CheckConstraint("route_class IN ('answerable','control')", name="route_class_values"),
        CheckConstraint("channel IN ('lexical','text','multimodal')", name="channel_values"),
        CheckConstraint(
            "rank BETWEEN 1 AND 50 AND generation_rank BETWEEN 1 AND 8",
            name="ranks_in_grid",
        ),
        CheckConstraint("accepted_round BETWEEN 0 AND 3", name="round_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    channel_candidate_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    match_batch_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.qbank_match_batches.match_batch_id"), nullable=False
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    qbank_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id"), nullable=False
    )
    route_class: Mapped[str] = mapped_column(String(16), nullable=False)
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    generation_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    accepted_round: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankLexicalTermFactModel(Base):
    """Content-free lexical audit fact for deterministic Step 7B FTS."""

    __tablename__ = "qbank_lexical_term_facts"
    __table_args__ = (
        UniqueConstraint("match_batch_id", "ordinal"),
        CheckConstraint("ordinal BETWEEN 0 AND 11", name="ordinal_values"),
        CheckConstraint(
            "term_kind IN ('lexeme','entity','number','date')",
            name="term_kind_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    lexical_term_fact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    match_batch_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.qbank_match_batches.match_batch_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    term_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    term_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    exact_boost: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankMatchFusedCandidateModel(Base):
    __tablename__ = "qbank_match_fused_candidates"
    __table_args__ = (
        UniqueConstraint("match_batch_id", "route_class", "rank"),
        UniqueConstraint("match_batch_id", "qbank_id"),
        CheckConstraint(
            "rank BETWEEN 1 AND 50 AND confidence BETWEEN 0 AND 1",
            name="rank_confidence_valid",
        ),
        CheckConstraint("route_class IN ('answerable','control')", name="route_class_values"),
        CheckConstraint(
            "(base_score IS NULL OR base_score BETWEEN 0 AND 1) AND "
            "(base_reliable_score IS NULL OR base_reliable_score BETWEEN 0 AND 1) AND "
            "(reranker_score IS NULL OR reranker_score BETWEEN 0 AND 1) AND "
            "(route_score IS NULL OR route_score BETWEEN 0 AND 1) AND "
            "(reranker_match_level IS NULL OR reranker_match_level IN "
            "('exact','strong','weak','none'))",
            name="step7b_score_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    fused_candidate_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    match_batch_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.qbank_match_batches.match_batch_id"), nullable=False
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    qbank_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id"), nullable=False
    )
    plan_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.memory_programs.plan_id"), nullable=False
    )
    route_class: Mapped[str] = mapped_column(String(16), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    lexical_rank: Mapped[int | None] = mapped_column(Integer)
    text_rank: Mapped[int | None] = mapped_column(Integer)
    multimodal_rank: Mapped[int | None] = mapped_column(Integer)
    raw_rrf: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    modality_compatible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    lexical_normalized_score: Mapped[float | None] = mapped_column(Float)
    text_normalized_score: Mapped[float | None] = mapped_column(Float)
    multimodal_normalized_score: Mapped[float | None] = mapped_column(Float)
    available_weight_sum: Mapped[float | None] = mapped_column(Float)
    base_score: Mapped[float | None] = mapped_column(Float)
    base_reliable_score: Mapped[float | None] = mapped_column(Float)
    reranker_score: Mapped[float | None] = mapped_column(Float)
    reranker_match_level: Mapped[str | None] = mapped_column(String(16))
    route_score: Mapped[float | None] = mapped_column(Float)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QBankRerankerAssessmentModel(Base):
    __tablename__ = "qbank_reranker_assessments"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id", "qbank_id"),
        CheckConstraint(
            "same_information_need IN ('pass','fail') AND "
            "candidate_can_answer_query IN ('pass','fail') AND "
            "answer_type_compatible IN ('pass','fail') AND "
            "modality_compatible IN ('pass','fail','not_applicable')",
            name="dimension_values",
        ),
        CheckConstraint(
            "match_level IN ('exact','strong','weak','none')",
            name="match_level_values",
        ),
        CheckConstraint("reranker_score BETWEEN 0 AND 1", name="score_valid"),
        {"schema": RUNTIME_SCHEMA},
    )

    reranker_assessment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    qbank_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id"), nullable=False
    )
    model_call_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id"), nullable=False
    )
    same_information_need: Mapped[str] = mapped_column(String(16), nullable=False)
    candidate_can_answer_query: Mapped[str] = mapped_column(String(16), nullable=False)
    answer_type_compatible: Mapped[str] = mapped_column(String(16), nullable=False)
    modality_compatible: Mapped[str] = mapped_column(String(16), nullable=False)
    match_level: Mapped[str] = mapped_column(String(16), nullable=False)
    reranker_score: Mapped[float] = mapped_column(Float, nullable=False)
    provider_schema_conformant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryRouteDecisionModel(Base):
    __tablename__ = "query_route_decisions"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id"),
        CheckConstraint(
            "outcome IN ('program','fallback','control','terminal_failure')",
            name="outcome_values",
        ),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0 AND 1", name="confidence_valid"
        ),
        CheckConstraint("top1_top2_gap IS NULL OR top1_top2_gap BETWEEN 0 AND 1", name="gap_valid"),
        {"schema": RUNTIME_SCHEMA},
    )

    route_decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"), nullable=False
    )
    match_batch_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.qbank_match_batches.match_batch_id"), nullable=False
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    selected_qbank_id: Mapped[str | None] = mapped_column(String(128))
    selected_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.memory_programs.plan_id")
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    top1_top2_gap: Mapped[float | None] = mapped_column(Float)
    match_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    gap_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(String(64))
    counterfactual_qbank_id: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryProgramBindingModel(Base):
    __tablename__ = "query_program_bindings"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id", "round_id"),
        UniqueConstraint("query_program_binding_id", "query_executor_run_id", "query_id"),
        CheckConstraint("round_id BETWEEN 0 AND 3", name="round_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    query_program_binding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"), nullable=False
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    route_decision_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_route_decisions.route_decision_id"), nullable=False
    )
    routing_projection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    qbank_id: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.memory_programs.plan_id"), nullable=False
    )
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    program_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    operator_set_version: Mapped[str] = mapped_column(String(128), nullable=False)
    requirement_contract_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    counterfactual: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryEvidenceSufficiencyAssessmentModel(Base):
    __tablename__ = "query_evidence_sufficiency_assessments"
    __table_args__ = (
        UniqueConstraint("query_program_binding_id"),
        ForeignKeyConstraint(
            [
                "query_program_binding_id",
                "query_executor_run_id",
                "query_id",
            ],
            [
                f"{RUNTIME_SCHEMA}.query_program_bindings.query_program_binding_id",
                f"{RUNTIME_SCHEMA}.query_program_bindings.query_executor_run_id",
                f"{RUNTIME_SCHEMA}.query_program_bindings.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["execution_id", "query_executor_run_id"],
            [
                f"{RUNTIME_SCHEMA}.program_executions.execution_id",
                f"{RUNTIME_SCHEMA}.program_executions.run_id",
            ],
        ),
        CheckConstraint(
            "independent_group_count >= 0 AND minimum_evidence_groups > 0", name="groups_valid"
        ),
        CheckConstraint(
            "((metadata->>'sufficiency_policy_version' = 'query-runtime-evidence-v2' "
            "AND structural_requirement_success IS NOT NULL "
            "AND query_evidence_relevance IS NOT NULL AND answer_support IS NOT NULL "
            "AND relevance_threshold IS NOT NULL "
            "AND sufficient = (technical_success AND structural_requirement_success "
            "AND query_evidence_relevance AND answer_support)) OR "
            "(COALESCE(metadata->>'sufficiency_policy_version', '') <> "
            "'query-runtime-evidence-v2' AND sufficient = "
            "(technical_success AND nonempty_evidence "
            "AND required_capabilities <@ actual_capabilities "
            "AND required_modalities <@ actual_modalities "
            "AND independent_group_count >= minimum_evidence_groups "
            "AND (NOT raw_image_required OR raw_image_reached))))",
            name="server_derived_v2",
        ),
        CheckConstraint(
            "sufficient <> (failure_reason IS NOT NULL)",
            name="failure_reason_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    sufficiency_assessment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    query_program_binding_id: Mapped[str] = mapped_column(String(128), nullable=False)
    execution_id: Mapped[str] = mapped_column(String(128), nullable=False)
    technical_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    nonempty_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False)
    actual_capabilities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    required_capabilities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    actual_modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    required_modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    independent_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_evidence_groups: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_image_reached: Mapped[bool] = mapped_column(Boolean, nullable=False)
    raw_image_required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    structural_requirement_success: Mapped[bool | None] = mapped_column(Boolean)
    query_evidence_relevance: Mapped[bool | None] = mapped_column(Boolean)
    answer_support: Mapped[bool | None] = mapped_column(Boolean)
    relevant_evidence_group_count: Mapped[int | None] = mapped_column(Integer)
    relevance_threshold: Mapped[float | None] = mapped_column(Float)
    sufficient: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryEvidenceRelevanceAssessmentModel(Base):
    __tablename__ = "query_evidence_relevance_assessments"
    __table_args__ = (
        UniqueConstraint("query_program_binding_id"),
        CheckConstraint(
            "relevance_threshold BETWEEN 0 AND 1 AND required_group_count > 0 "
            "AND relevant_group_count >= 0",
            name="relevance_counts_valid",
        ),
        CheckConstraint(
            "passed = (relevant_group_count >= required_group_count)",
            name="relevance_server_derived",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    relevance_assessment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    query_program_binding_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_program_bindings.query_program_binding_id"),
        nullable=False,
    )
    relevance_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    required_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    relevant_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryEvidenceRelevanceItemModel(Base):
    __tablename__ = "query_evidence_relevance_items"
    __table_args__ = (
        UniqueConstraint("relevance_assessment_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            "cosine_similarity BETWEEN -1 AND 1 "
            "AND normalized_score BETWEEN 0 AND 1 "
            "AND threshold BETWEEN 0 AND 1",
            name="score_valid",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    relevance_item_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    relevance_assessment_id: Mapped[str] = mapped_column(
        ForeignKey(
            f"{RUNTIME_SCHEMA}.query_evidence_relevance_assessments.relevance_assessment_id"
        ),
        nullable=False,
    )
    query_executor_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    execution_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.program_executions.execution_id"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_id_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_group_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    cosine_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    normalized_score: Mapped[float] = mapped_column(Float, nullable=False)
    relevant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryFallbackDecisionModel(Base):
    __tablename__ = "query_fallback_decisions"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id"),
        ForeignKeyConstraint(
            ["fallback_answer_id", "fallback_run_id", "fallback_query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        CheckConstraint(
            "(counterfactual_completed AND fallback_answer_id IS NOT NULL AND error_type IS NULL) "
            "OR (NOT counterfactual_completed AND fallback_answer_id IS NULL "
            "AND error_type IS NOT NULL)",
            name="counterfactual_terminal",
        ),
        CheckConstraint(
            "active_fallback = (reason IS NOT NULL)",
            name="active_reason_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    fallback_decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    fallback_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    fallback_query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    counterfactual_completed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    active_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(64))
    fallback_answer_id: Mapped[str | None] = mapped_column(String(128))
    error_type: Mapped[str | None] = mapped_column(String(256))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryAnswerSelectionModel(Base):
    __tablename__ = "query_answer_selections"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id"),
        ForeignKeyConstraint(
            ["selected_answer_id", "query_executor_run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["pmmc_answer_id", "query_executor_run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        CheckConstraint(
            "active_variant IN ('pmmc_r0','pmmc_r1','pmmc_r2','pmmc_r3',"
            "'native_mm_counterfactual','no_fallback_counterfactual')",
            name="variant_values",
        ),
        CheckConstraint(
            "(terminal_failure AND selected_answer_id IS NULL AND failure_type IS NOT NULL) "
            "OR (NOT terminal_failure AND selected_answer_id IS NOT NULL "
            "AND failure_type IS NULL)",
            name="terminal_selection_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    answer_selection_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    active_variant: Mapped[str] = mapped_column(String(64), nullable=False)
    selected_answer_id: Mapped[str | None] = mapped_column(String(128))
    pmmc_answer_id: Mapped[str | None] = mapped_column(String(128))
    fallback_answer_id: Mapped[str | None] = mapped_column(String(128))
    terminal_failure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failure_type: Mapped[str | None] = mapped_column(String(256))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PmmcFallbackRunLinkModel(Base):
    __tablename__ = "pmmc_fallback_run_links"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "fallback_run_id"),
        ForeignKeyConstraint(
            ["fallback_run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        {"schema": RUNTIME_SCHEMA},
    )
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"), primary_key=True
    )
    fallback_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    method_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class PmmcFallbackQueryLinkModel(Base):
    __tablename__ = "pmmc_fallback_query_links"
    __table_args__ = (
        ForeignKeyConstraint(
            ["query_executor_run_id", "fallback_run_id"],
            [
                f"{RUNTIME_SCHEMA}.pmmc_fallback_run_links.query_executor_run_id",
                f"{RUNTIME_SCHEMA}.pmmc_fallback_run_links.fallback_run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["fallback_answer_id", "fallback_run_id", "fallback_query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        {"schema": RUNTIME_SCHEMA},
    )
    query_executor_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    fallback_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    fallback_query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    fallback_answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryExecutorFreezeManifestModel(Base):
    __tablename__ = "query_executor_freeze_manifests"
    __table_args__ = (
        UniqueConstraint("verified_qbank_freeze_manifest_id", "query_executor_config_hash"),
        CheckConstraint("active_round = 2 AND question_budget = 8", name="active_protocol_frozen"),
        CheckConstraint(
            "schema_version <> '0.2.0' OR "
            "(query_executor_protocol_hash ~ '^[0-9a-f]{64}$' AND "
            "approved_routing_policy_hash ~ '^[0-9a-f]{64}$' AND "
            "approved_selection_hash ~ '^[0-9a-f]{64}$')",
            name="approved_routing_identity",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    freeze_manifest_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    verified_qbank_freeze_manifest_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_executor_config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    query_executor_protocol_hash: Mapped[str | None] = mapped_column(String(128))
    approved_routing_policy_hash: Mapped[str | None] = mapped_column(String(128))
    approved_selection_hash: Mapped[str | None] = mapped_column(String(128))
    answer_prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    fallback_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    active_round: Mapped[int] = mapped_column(Integer, nullable=False)
    question_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    match_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    gap_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_grid: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    gap_grid: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    pilot_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    approved_by: Mapped[str] = mapped_column(String(256), nullable=False)
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class StructuredAnswerOutputModel(Base):
    __tablename__ = "structured_answer_outputs"
    __table_args__ = (
        UniqueConstraint("run_id", "query_id", "answer_variant", "attempt_ordinal"),
        CheckConstraint("attempt_ordinal IN (0, 1)", name="attempt_ordinal_values"),
        CheckConstraint("parse_status IN ('accepted','rejected')", name="parse_status_values"),
        CheckConstraint(
            "(supported AND answer IS NOT NULL AND abstain_reason IS NULL "
            "AND cardinality(used_evidence_aliases) > 0) OR "
            "(NOT supported AND answer IS NULL AND abstain_reason IS NOT NULL)",
            name="support_contract",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    structured_answer_output_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    answer_variant: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    model_call_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id"), nullable=False
    )
    answer_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.final_answers.answer_id")
    )
    answer: Mapped[str | None] = mapped_column(Text)
    supported: Mapped[bool] = mapped_column(Boolean, nullable=False)
    used_evidence_aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    abstain_reason: Mapped[str | None] = mapped_column(String(64))
    parse_status: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_schema_conformant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    finish_reason: Mapped[str | None] = mapped_column(String(32))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class AnswerFormatDiagnosticModel(Base):
    __tablename__ = "answer_format_diagnostics"
    __table_args__ = (
        UniqueConstraint("run_id", "query_id", "answer_variant", "attempt_ordinal"),
        CheckConstraint("attempt_ordinal IN (0, 1)", name="attempt_ordinal_values"),
        CheckConstraint(
            "accepted = (cardinality(failure_codes) = 0)",
            name="accepted_failure_consistency",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    diagnostic_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    answer_variant: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_call_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id")
    )
    failure_codes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    provider_schema_conformant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class AnswerReuseLinkModel(Base):
    __tablename__ = "answer_reuse_links"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id", "target_variant"),
        {"schema": RUNTIME_SCHEMA},
    )

    answer_reuse_link_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_answer_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.final_answers.answer_id"), nullable=False
    )
    target_variant: Mapped[str] = mapped_column(String(128), nullable=False)
    context_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryExecutionReuseLinkModel(Base):
    __tablename__ = "query_execution_reuse_links"
    __table_args__ = (
        UniqueConstraint("query_executor_run_id", "query_id", "target_round_id"),
        CheckConstraint("target_round_id BETWEEN 0 AND 3", name="round_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    execution_reuse_link_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_executor_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.query_executor_runs.query_executor_run_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_execution_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.program_executions.execution_id"), nullable=False
    )
    target_round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_identity_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class Step7BPilotModel(Base):
    __tablename__ = "step7b_pilots"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running','failed','needs_remediation',"
            "'awaiting_threshold_review','approved','rejected')",
            name="status_values",
        ),
        CheckConstraint("paper_result = false", name="development_only"),
        {"schema": RUNTIME_SCHEMA},
    )

    pilot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    compiler_run_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    query_executor_run_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    partition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    paper_result: Mapped[bool] = mapped_column(Boolean, nullable=False)
    protocol_integrity_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    viability_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class Step7BThresholdDecisionModel(Base):
    __tablename__ = "step7b_threshold_decisions"
    __table_args__ = (
        UniqueConstraint("pilot_id"),
        CheckConstraint("match_threshold IN (0.5,0.6,0.7,0.8)", name="match_grid"),
        CheckConstraint("gap_threshold IN (0.0,0.05,0.1)", name="gap_grid"),
        CheckConstraint("decision IN ('approve','reject')", name="decision_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    threshold_decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    pilot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.step7b_pilots.pilot_id"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(256), nullable=False)
    match_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    gap_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    rationale_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    decision_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class Step7BFreezeManifestModel(Base):
    __tablename__ = "step7b_freeze_manifests"
    __table_args__ = (
        UniqueConstraint("pilot_id"),
        CheckConstraint("active_round = 2 AND question_budget = 8", name="active_protocol"),
        {"schema": RUNTIME_SCHEMA},
    )

    freeze_manifest_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    pilot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.step7b_pilots.pilot_id"), nullable=False
    )
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    questioner_freeze_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    compiler_program_freeze_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    verified_qbank_freeze_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    query_executor_freeze_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    answer_contract_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    active_round: Mapped[int] = mapped_column(Integer, nullable=False)
    question_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    match_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    gap_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
