"""Internal SQLAlchemy models for baseline corpus and context foundation."""

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
from pmmc.infrastructure.database.constants import (
    BASELINE_REPRESENTATION_VALUES,
    CORPUS_STATUS_VALUES,
    DATASET_VALUES,
    RUNTIME_SCHEMA,
    sql_text_values,
)

JsonValue = dict[str, Any]


class RetrievalCorpusModel(Base):
    __tablename__ = "retrieval_corpora"
    __table_args__ = (
        UniqueConstraint("corpus_id", "snapshot_id"),
        CheckConstraint(
            f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"
        ),
        CheckConstraint(
            f"status IN {sql_text_values(CORPUS_STATUS_VALUES)}", name="status_values"
        ),
        CheckConstraint("max_text_tokens > 0", name="max_text_tokens_positive"),
        CheckConstraint("max_images > 0", name="max_images_positive"),
        CheckConstraint("overlap_memory_units >= 0", name="overlap_nonnegative"),
        CheckConstraint(
            "window_count >= 0 AND document_count >= 0 AND image_document_count >= 0",
            name="counts_nonnegative",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    corpus_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    windows_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    documents_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    tokenizer_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    tokenizer_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    renderer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    window_strategy: Mapped[str] = mapped_column(String(128), nullable=False)
    max_text_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_images: Mapped[int] = mapped_column(Integer, nullable=False)
    overlap_memory_units: Mapped[int] = mapped_column(Integer, nullable=False)
    cross_session: Mapped[bool] = mapped_column(Boolean, nullable=False)
    representations: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    window_count: Mapped[int] = mapped_column(Integer, nullable=False)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    image_document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unique_memory_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unique_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    spec_json: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    manifest_json: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class MemoryWindowModel(Base):
    __tablename__ = "memory_windows"
    __table_args__ = (
        ForeignKeyConstraint(
            ["corpus_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_corpora.corpus_id",
                f"{RUNTIME_SCHEMA}.retrieval_corpora.snapshot_id",
            ],
        ),
        UniqueConstraint("window_id", "corpus_id"),
        UniqueConstraint("window_id", "snapshot_id", "conversation_id"),
        UniqueConstraint("corpus_id", "session_id", "window_ordinal"),
        CheckConstraint("window_ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            "start_timestamp_order >= 0 AND end_timestamp_order >= start_timestamp_order",
            name="orders_valid",
        ),
        CheckConstraint("text_caption_token_count >= 0", name="tokens_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    window_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    window_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    start_timestamp_order: Mapped[int] = mapped_column(Integer, nullable=False)
    end_timestamp_order: Mapped[int] = mapped_column(Integer, nullable=False)
    text_caption_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    available_after_session: Mapped[str] = mapped_column(String(256), nullable=False)
    oversized_single_unit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class MemoryWindowUnitModel(Base):
    __tablename__ = "memory_window_units"
    __table_args__ = (
        PrimaryKeyConstraint("window_id", "memory_id"),
        UniqueConstraint("window_id", "ordinal"),
        ForeignKeyConstraint(
            ["window_id", "snapshot_id", "conversation_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_windows.window_id",
                f"{RUNTIME_SCHEMA}.memory_windows.snapshot_id",
                f"{RUNTIME_SCHEMA}.memory_windows.conversation_id",
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
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    window_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    memory_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class RetrievalDocumentModel(Base):
    __tablename__ = "retrieval_documents"
    __table_args__ = (
        ForeignKeyConstraint(
            ["corpus_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_corpora.corpus_id",
                f"{RUNTIME_SCHEMA}.retrieval_corpora.snapshot_id",
            ],
        ),
        UniqueConstraint("document_id", "snapshot_id"),
        ForeignKeyConstraint(
            ["parent_window_id", "corpus_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_windows.window_id",
                f"{RUNTIME_SCHEMA}.memory_windows.corpus_id",
            ],
        ),
        CheckConstraint(
            f"representation IN {sql_text_values(BASELINE_REPRESENTATION_VALUES)}",
            name="representation_values",
        ),
        CheckConstraint("token_count IS NULL OR token_count >= 0", name="tokens_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    parent_window_id: Mapped[str] = mapped_column(String(128), nullable=False)
    representation: Mapped[str] = mapped_column(String(32), nullable=False)
    memory_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    rendered_text: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    available_after_session: Mapped[str] = mapped_column(String(256), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class RetrievalDocumentImageModel(Base):
    __tablename__ = "retrieval_document_images"
    __table_args__ = (
        PrimaryKeyConstraint("document_id", "image_id"),
        UniqueConstraint("document_id", "ordinal"),
        ForeignKeyConstraint(
            ["document_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_documents.document_id",
                f"{RUNTIME_SCHEMA}.retrieval_documents.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME_SCHEMA}.images.snapshot_id", f"{RUNTIME_SCHEMA}.images.image_id"],
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    document_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    image_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class ContextAssemblyModel(Base):
    __tablename__ = "context_assemblies"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.run_queries.run_id",
                f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.run_queries.query_id",
            ],
        ),
        UniqueConstraint("context_assembly_id", "run_id"),
        UniqueConstraint("context_assembly_id", "snapshot_id"),
        UniqueConstraint("context_assembly_id", "snapshot_id", "conversation_id"),
        UniqueConstraint(
            "context_assembly_id",
            "run_id",
            "snapshot_id",
            "query_id",
            "conversation_id",
        ),
        ForeignKeyConstraint(
            ["job_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.experiment_jobs.job_id",
                f"{RUNTIME_SCHEMA}.experiment_jobs.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "query_id", "conversation_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_queries.query_id",
                f"{RUNTIME_SCHEMA}.benchmark_queries.conversation_id",
            ],
        ),
        ForeignKeyConstraint(
            ["retrieval_batch_id", "run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_batches.retrieval_batch_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.run_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.snapshot_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.query_id",
            ],
        ),
        CheckConstraint("total_context_tokens > 0", name="total_positive"),
        CheckConstraint("reserved_output_tokens > 0", name="reserved_positive"),
        CheckConstraint(
            "max_input_tokens = total_context_tokens - reserved_output_tokens",
            name="input_budget_consistent",
        ),
        CheckConstraint(
            "input_tokens = instruction_tokens + query_tokens + query_image_tokens + "
            "memory_text_tokens + memory_image_tokens + request_token_adjustment",
            name="input_components_consistent",
        ),
        CheckConstraint("input_tokens <= max_input_tokens", name="input_within_budget"),
        {"schema": RUNTIME_SCHEMA},
    )

    context_assembly_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(128))
    retrieval_batch_id: Mapped[str | None] = mapped_column(String(128))
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    representation: Mapped[str] = mapped_column(String(32), nullable=False)
    truncation_policy: Mapped[str] = mapped_column(String(128), nullable=False)
    tokenizer_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    tokenizer_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    total_context_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    instruction_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    query_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    query_image_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    memory_text_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    memory_image_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    request_token_adjustment: Mapped[int] = mapped_column(Integer, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class ContextAssemblyRetrievalCandidateModel(Base):
    __tablename__ = "context_assembly_retrieval_candidates"
    __table_args__ = (
        PrimaryKeyConstraint("context_assembly_id", "retrieval_trace_id"),
        UniqueConstraint("context_assembly_id", "rank"),
        ForeignKeyConstraint(
            [
                "context_assembly_id",
                "run_id",
                "snapshot_id",
                "query_id",
                "conversation_id",
            ],
            [
                f"{RUNTIME_SCHEMA}.context_assemblies.context_assembly_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.run_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.snapshot_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.query_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.conversation_id",
            ],
        ),
        ForeignKeyConstraint(
            ["retrieval_trace_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_traces.retrieval_trace_id",
                f"{RUNTIME_SCHEMA}.retrieval_traces.run_id",
                f"{RUNTIME_SCHEMA}.retrieval_traces.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["window_id", "snapshot_id", "conversation_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_windows.window_id",
                f"{RUNTIME_SCHEMA}.memory_windows.snapshot_id",
                f"{RUNTIME_SCHEMA}.memory_windows.conversation_id",
            ],
        ),
        CheckConstraint("rank > 0", name="rank_positive"),
        CheckConstraint(
            "disposition IN ('included','deduplicated','budget_excluded')",
            name="disposition_values",
        ),
        CheckConstraint(
            "(disposition = 'included' AND cardinality(contributed_memory_ids) > 0) OR "
            "(disposition != 'included' AND cardinality(contributed_memory_ids) = 0)",
            name="contribution_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    context_assembly_id: Mapped[str] = mapped_column(String(128), nullable=False)
    retrieval_trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    window_id: Mapped[str] = mapped_column(String(128), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    disposition: Mapped[str] = mapped_column(String(32), nullable=False)
    contributed_memory_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ContextAssemblyMemoryUnitModel(Base):
    __tablename__ = "context_assembly_memory_units"
    __table_args__ = (
        PrimaryKeyConstraint("context_assembly_id", "memory_id"),
        UniqueConstraint("context_assembly_id", "ordinal"),
        ForeignKeyConstraint(
            ["context_assembly_id", "snapshot_id", "conversation_id"],
            [
                f"{RUNTIME_SCHEMA}.context_assemblies.context_assembly_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.snapshot_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.conversation_id",
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
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint("text_tokens >= 0 AND image_tokens >= 0", name="tokens_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    context_assembly_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    memory_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    included: Mapped[bool] = mapped_column(Boolean, nullable=False)
    text_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    image_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ContextAssemblyImageModel(Base):
    __tablename__ = "context_assembly_images"
    __table_args__ = (
        PrimaryKeyConstraint("context_assembly_id", "image_id", "role"),
        UniqueConstraint("context_assembly_id", "role", "ordinal"),
        ForeignKeyConstraint(
            ["context_assembly_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.context_assemblies.context_assembly_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME_SCHEMA}.images.snapshot_id", f"{RUNTIME_SCHEMA}.images.image_id"],
        ),
        CheckConstraint("role IN ('included','truncated','query')", name="role_values"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    context_assembly_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    image_id: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
