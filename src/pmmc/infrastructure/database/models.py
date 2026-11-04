"""Internal SQLAlchemy models for canonical benchmark persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import (
    ASSET_TYPE_VALUES,
    DATASET_VALUES,
    EVALUATION_SCHEMA,
    MODALITY_VALUES,
    QUESTION_TYPE_VALUES,
    RUNTIME_SCHEMA,
    SPEAKER_VALUES,
    sql_text_array,
    sql_text_values,
)

JsonValue = dict[str, Any]


class DatasetSnapshotModel(Base):
    """Immutable canonical snapshot manifest and import state."""

    __tablename__ = "dataset_snapshots"
    __table_args__ = (
        CheckConstraint(f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"),
        CheckConstraint(
            "import_state IN ('importing', 'ready', 'failed')",
            name="import_state_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(128), nullable=False)
    adapter_name: Mapped[str] = mapped_column(String(128), nullable=False)
    adapter_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    evaluation_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    excluded_source_categories: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    source_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    canonical_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    category_counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    eligible_category_counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    excluded_category_counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    manifest_json: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    import_state: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    runtime_archive_uri: Mapped[str | None] = mapped_column(Text)
    evaluation_archive_uri: Mapped[str | None] = mapped_column(Text)
    validation_report_uri: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class SnapshotImportAttemptModel(Base):
    """Audit record for every import attempt, including failures."""

    __tablename__ = "snapshot_import_attempts"
    __table_args__ = (
        CheckConstraint("status IN ('importing', 'ready', 'failed')", name="status_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    attempt_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    imported_counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class BenchmarkSessionModel(Base):
    """Canonical benchmark session with dataset-logical ordering."""

    __tablename__ = "benchmark_sessions"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "session_id"),
        UniqueConstraint("snapshot_id", "conversation_id", "session_id"),
        UniqueConstraint("snapshot_id", "conversation_id", "session_order"),
        CheckConstraint(f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"),
        CheckConstraint("session_order >= 0", name="session_order_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    session_order: Mapped[int] = mapped_column(Integer, nullable=False)
    source_session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_date_raw: Mapped[str | None] = mapped_column(Text)
    source_date_iso: Mapped[str | None] = mapped_column(String(64))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class AssetModel(Base):
    """Metadata for large benchmark artifacts stored outside Postgres."""

    __tablename__ = "assets"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "asset_id"),
        CheckConstraint(
            f"asset_type IN {sql_text_values(ASSET_TYPE_VALUES)}",
            name="asset_type_values",
        ),
        CheckConstraint("size_bytes IS NULL OR size_bytes >= 0", name="size_nonnegative"),
        CheckConstraint("width IS NULL OR width >= 0", name="width_nonnegative"),
        CheckConstraint("height IS NULL OR height >= 0", name="height_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    asset_id: Mapped[str] = mapped_column(String(256), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    local_path: Mapped[str | None] = mapped_column(Text)
    remote_url: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(128))
    mime_type: Mapped[str | None] = mapped_column(String(256))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    source_id: Mapped[str | None] = mapped_column(String(256))
    run_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    storage_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ImageModel(Base):
    """Normalized image identity extracted from canonical ImageRef objects."""

    __tablename__ = "images"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "image_id"),
        ForeignKeyConstraint(
            ["snapshot_id", "asset_id"],
            [f"{RUNTIME_SCHEMA}.assets.snapshot_id", f"{RUNTIME_SCHEMA}.assets.asset_id"],
        ),
        CheckConstraint("width IS NULL OR width >= 0", name="width_nonnegative"),
        CheckConstraint("height IS NULL OR height >= 0", name="height_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    image_id: Mapped[str] = mapped_column(String(256), nullable=False)
    asset_id: Mapped[str | None] = mapped_column(String(256))
    local_path: Mapped[str | None] = mapped_column(Text)
    remote_url: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(128))
    mime_type: Mapped[str | None] = mapped_column(String(256))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class MemoryUnitModel(Base):
    """Runtime-safe long-term memory unit."""

    __tablename__ = "memory_units"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "memory_id"),
        ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "session_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_sessions.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.conversation_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.session_id",
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
        UniqueConstraint("snapshot_id", "conversation_id", "turn_id"),
        UniqueConstraint("snapshot_id", "conversation_id", "timestamp_order"),
        UniqueConstraint(
            "snapshot_id",
            "conversation_id",
            "memory_id",
            name="uq_memory_units_snapshot_conversation_memory",
        ),
        CheckConstraint(f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"),
        CheckConstraint(f"speaker IN {sql_text_values(SPEAKER_VALUES)}", name="speaker_values"),
        CheckConstraint("timestamp_order >= 0", name="timestamp_order_nonnegative"),
        CheckConstraint("token_count IS NULL OR token_count >= 0", name="token_nonnegative"),
        CheckConstraint("cardinality(modalities) > 0", name="modalities_nonempty"),
        CheckConstraint(
            f"modalities <@ {sql_text_array(MODALITY_VALUES)}",
            name="modalities_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    memory_id: Mapped[str] = mapped_column(String(256), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    turn_id: Mapped[str] = mapped_column(String(256), nullable=False)
    timestamp_order: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    search_document: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', coalesce(text, '') || ' ' || coalesce(caption, ''))",
            persisted=True,
        ),
        nullable=False,
    )
    entity_mentions: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    modalities: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    available_after_session: Mapped[str] = mapped_column(String(256), nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


Index(
    "ix_memory_units_search_document",
    MemoryUnitModel.search_document,
    postgresql_using="gin",
)
Index(
    "ix_memory_units_entity_mentions",
    MemoryUnitModel.entity_mentions,
    postgresql_using="gin",
)


class MemoryUnitImageModel(Base):
    """Ordered image binding for one memory unit."""

    __tablename__ = "memory_unit_images"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "memory_id", "image_id"),
        UniqueConstraint("snapshot_id", "memory_id", "ordinal"),
        ForeignKeyConstraint(
            ["snapshot_id", "memory_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_units.snapshot_id",
                f"{RUNTIME_SCHEMA}.memory_units.memory_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME_SCHEMA}.images.snapshot_id", f"{RUNTIME_SCHEMA}.images.image_id"],
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    memory_id: Mapped[str] = mapped_column(String(256), nullable=False)
    image_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class BenchmarkQueryModel(Base):
    """Runtime-safe benchmark query and source locator."""

    __tablename__ = "benchmark_queries"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "query_id"),
        UniqueConstraint(
            "snapshot_id",
            "query_id",
            "conversation_id",
            name="uq_benchmark_queries_snapshot_query_conversation",
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "available_after_session"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_sessions.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.conversation_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.session_id",
            ],
        ),
        CheckConstraint(f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"),
        CheckConstraint(
            f"question_type IN {sql_text_values(QUESTION_TYPE_VALUES)}",
            name="question_type_values",
        ),
        CheckConstraint(
            "(eligible AND exclusion_reason IS NULL) OR "
            "(NOT eligible AND exclusion_reason IS NOT NULL)",
            name="eligibility_reason_consistent",
        ),
        CheckConstraint(
            "source_record_index IS NULL OR source_record_index >= 0",
            name="source_record_index_nonnegative",
        ),
        CheckConstraint("qa_index IS NULL OR qa_index >= 0", name="qa_index_nonnegative"),
        CheckConstraint(
            "source_turn_index IS NULL OR source_turn_index >= 0",
            name="source_turn_index_nonnegative",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(64), nullable=False)
    query_subtype: Mapped[str | None] = mapped_column(String(128))
    available_after_session: Mapped[str | None] = mapped_column(String(256))
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    evaluation_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    exclusion_reason: Mapped[str | None] = mapped_column(Text)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    source_record_index: Mapped[int | None] = mapped_column(Integer)
    source_query_id: Mapped[str | None] = mapped_column(String(256))
    scenario_id: Mapped[str | None] = mapped_column(String(256))
    qa_index: Mapped[int | None] = mapped_column(Integer)
    source_session_id: Mapped[str | None] = mapped_column(String(256))
    source_round_id: Mapped[str | None] = mapped_column(String(256))
    source_turn_index: Mapped[int | None] = mapped_column(Integer)
    source_category: Mapped[str | None] = mapped_column(String(128))
    source_subtype: Mapped[str | None] = mapped_column(String(128))
    query_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    source_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    eligibility_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class QueryImageModel(Base):
    """Ordered runtime query image binding."""

    __tablename__ = "query_images"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "query_id", "image_id"),
        UniqueConstraint("snapshot_id", "query_id", "ordinal"),
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME_SCHEMA}.images.snapshot_id", f"{RUNTIME_SCHEMA}.images.image_id"],
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    image_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class RunManifestModel(Base):
    """Queryable experiment run manifest plus lossless JSON representation."""

    __tablename__ = "run_manifests"
    __table_args__ = (
        CheckConstraint(f"dataset IN {sql_text_values(DATASET_VALUES)}", name="dataset_values"),
        {"schema": RUNTIME_SCHEMA},
    )

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    experiment_name: Mapped[str] = mapped_column(String(256), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_version: Mapped[str | None] = mapped_column(String(128))
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    paper_mode: Mapped[bool] = mapped_column(Boolean, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    experiment_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    selection_hash: Mapped[str | None] = mapped_column(String(128))
    seed: Mapped[int | None] = mapped_column(Integer)
    retrieval_topk_max: Mapped[int | None] = mapped_column(Integer)
    config_hash: Mapped[str | None] = mapped_column(String(128))
    git_commit: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    manifest_json: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)


class RunSnapshotLinkModel(Base):
    """Many-to-many link between runs and immutable dataset snapshots."""

    __tablename__ = "run_snapshot_links"
    __table_args__ = (
        ForeignKeyConstraint(["run_id"], [f"{RUNTIME_SCHEMA}.run_manifests.run_id"]),
        ForeignKeyConstraint(["snapshot_id"], [f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"]),
        {"schema": RUNTIME_SCHEMA},
    )

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)


class QueryGoldModel(Base):
    """Evaluation-only answer and top-level evidence metadata."""

    __tablename__ = "query_gold"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_queries.query_id",
            ],
        ),
        {"schema": EVALUATION_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    answer: Mapped[str | None] = mapped_column(Text)
    evidence_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    gold_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    record_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)


class QueryGoldEvidenceIdModel(Base):
    __tablename__ = "query_gold_evidence_ids"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.query_gold.snapshot_id",
                f"{EVALUATION_SCHEMA}.query_gold.query_id",
            ],
        ),
        UniqueConstraint("snapshot_id", "query_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": EVALUATION_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    evidence_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class QueryGoldMemoryIdModel(Base):
    __tablename__ = "query_gold_memory_ids"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.query_gold.snapshot_id",
                f"{EVALUATION_SCHEMA}.query_gold.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "memory_id"],
            [
                f"{RUNTIME_SCHEMA}.memory_units.snapshot_id",
                f"{RUNTIME_SCHEMA}.memory_units.memory_id",
            ],
        ),
        UniqueConstraint("snapshot_id", "query_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": EVALUATION_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    memory_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class QueryGoldSessionIdModel(Base):
    __tablename__ = "query_gold_session_ids"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.query_gold.snapshot_id",
                f"{EVALUATION_SCHEMA}.query_gold.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "session_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_sessions.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_sessions.session_id",
            ],
        ),
        UniqueConstraint("snapshot_id", "query_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": EVALUATION_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class QueryGoldImageIdModel(Base):
    __tablename__ = "query_gold_image_ids"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.query_gold.snapshot_id",
                f"{EVALUATION_SCHEMA}.query_gold.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME_SCHEMA}.images.snapshot_id", f"{RUNTIME_SCHEMA}.images.image_id"],
        ),
        UniqueConstraint("snapshot_id", "query_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": EVALUATION_SCHEMA},
    )

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    image_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


Index(
    "ix_sessions_snapshot_conversation_order",
    BenchmarkSessionModel.snapshot_id,
    BenchmarkSessionModel.conversation_id,
    BenchmarkSessionModel.session_order,
)
Index(
    "ix_memories_snapshot_conversation_order",
    MemoryUnitModel.snapshot_id,
    MemoryUnitModel.conversation_id,
    MemoryUnitModel.timestamp_order,
)
Index(
    "ix_queries_snapshot_eligibility_type",
    BenchmarkQueryModel.snapshot_id,
    BenchmarkQueryModel.eligible,
    BenchmarkQueryModel.question_type,
)
Index(
    "ix_queries_snapshot_source_category_scenario",
    BenchmarkQueryModel.snapshot_id,
    BenchmarkQueryModel.source_category,
    BenchmarkQueryModel.scenario_id,
)
Index("ix_assets_sha256", AssetModel.sha256)
Index(
    "ix_gold_evidence_snapshot_evidence",
    QueryGoldEvidenceIdModel.snapshot_id,
    QueryGoldEvidenceIdModel.evidence_id,
)
Index(
    "ix_gold_memory_snapshot_memory",
    QueryGoldMemoryIdModel.snapshot_id,
    QueryGoldMemoryIdModel.memory_id,
)
Index(
    "ix_gold_session_snapshot_session",
    QueryGoldSessionIdModel.snapshot_id,
    QueryGoldSessionIdModel.session_id,
)
Index(
    "ix_gold_image_snapshot_image",
    QueryGoldImageIdModel.snapshot_id,
    QueryGoldImageIdModel.image_id,
)
