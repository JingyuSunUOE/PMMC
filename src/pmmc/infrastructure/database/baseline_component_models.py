"""SQLAlchemy models for independently recoverable baseline components."""

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
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import RUNTIME_SCHEMA, sql_text_values

JsonValue = dict[str, Any]

_FAMILY_STATUS_VALUES = ("registered", "assembling", "ready", "failed")
_COMPONENT_STATUS_VALUES = (
    "registered",
    "retrieval_ready",
    "answer_ready",
    "exported",
    "imported",
    "verified",
    "failed",
)


class BaselineSuiteFamilyModel(Base):
    __tablename__ = "baseline_suite_families"
    __table_args__ = (
        CheckConstraint(
            f"status IN {sql_text_values(_FAMILY_STATUS_VALUES)}",
            name="status_values",
        ),
        CheckConstraint("cardinality(datasets) > 0", name="datasets_nonempty"),
        CheckConstraint("cardinality(methods) > 0", name="methods_nonempty"),
        CheckConstraint("expected_component_count > 0", name="component_count_positive"),
        CheckConstraint(
            "expected_component_count = cardinality(datasets) * cardinality(methods)",
            name="component_matrix_count",
        ),
        CheckConstraint(
            "verified_component_count >= 0 AND "
            "verified_component_count <= expected_component_count",
            name="verified_count_valid",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    family_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    partition_id: Mapped[str] = mapped_column(String(128), nullable=False)
    partition_assignment_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    datasets: Mapped[list[str]] = mapped_column(ARRAY(String(32)), nullable=False)
    methods: Mapped[list[str]] = mapped_column(ARRAY(String(128)), nullable=False)
    snapshot_ids: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    selection_hashes: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    selected_counts: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    answer_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    answer_model_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    judge_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    judge_model_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    answer_prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_prompt_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    decoding_config: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    total_context_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    answer_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    protocol_versions: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    paper_mode: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    expected_component_count: Mapped[int] = mapped_column(Integer, nullable=False)
    verified_component_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class BaselineSuiteFamilyComponentModel(Base):
    __tablename__ = "baseline_suite_family_components"
    __table_args__ = (
        UniqueConstraint("family_id", "dataset", "method"),
        UniqueConstraint("family_id", "component_id"),
        CheckConstraint(
            f"status IN {sql_text_values(_COMPONENT_STATUS_VALUES)}",
            name="status_values",
        ),
        CheckConstraint("selected_count > 0", name="selected_count_positive"),
        CheckConstraint("transition_ordinal >= 0", name="transition_ordinal_nonnegative"),
        CheckConstraint("paper_result = false", name="not_paper_result"),
        CheckConstraint("paper_component = true", name="paper_component_required"),
        {"schema": RUNTIME_SCHEMA},
    )

    component_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    family_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.baseline_suite_families.family_id"), nullable=False
    )
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    selection_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    suite_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.baseline_suites.suite_id")
    )
    run_id: Mapped[str | None] = mapped_column(ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"))
    corpus_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.retrieval_corpora.corpus_id")
    )
    scientific_config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    component_config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    paper_component: Mapped[bool] = mapped_column(Boolean, nullable=False)
    paper_result: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    transition_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    bundle_export_id: Mapped[str | None] = mapped_column(String(128))
    bundle_sha256: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)


class BaselineComponentImportModel(Base):
    __tablename__ = "baseline_component_imports"
    __table_args__ = (
        ForeignKeyConstraint(
            ["family_id", "component_id"],
            [
                f"{RUNTIME_SCHEMA}.baseline_suite_family_components.family_id",
                f"{RUNTIME_SCHEMA}.baseline_suite_family_components.component_id",
            ],
        ),
        UniqueConstraint("component_id", "bundle_export_id"),
        {"schema": RUNTIME_SCHEMA},
    )

    import_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(128), nullable=False)
    component_id: Mapped[str] = mapped_column(String(128), nullable=False)
    bundle_export_id: Mapped[str] = mapped_column(String(128), nullable=False)
    bundle_sha256: Mapped[str] = mapped_column(String(128), nullable=False)
    source_alembic_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    importer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    imported_by: Mapped[str] = mapped_column(String(256), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column("metadata", JSONB, nullable=False)
