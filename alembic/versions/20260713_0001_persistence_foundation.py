"""Create canonical persistence schemas and access boundaries.

Revision ID: 20260713_0001
Revises: None
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260713_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"
DATASET_VALUES = ("memlens", "memgallery", "unknown")
ASSET_TYPE_VALUES = (
    "raw_image",
    "resized_image",
    "prompt",
    "model_response",
    "trace_export",
    "embedding_dump",
    "dataset_snapshot",
    "other",
)
SPEAKER_VALUES = ("user", "assistant", "system", "unknown")
MODALITY_VALUES = (
    "text",
    "caption",
    "image",
    "raw_image",
    "text_embedding",
    "image_embedding",
    "joint_mm_embedding",
    "embedding",
    "multimodal",
)
QUESTION_TYPE_VALUES = (
    "textual_fact",
    "visual_detail",
    "image_text_binding",
    "visual_search",
    "multi_hop",
    "temporal",
    "knowledge_update",
    "conflict_detection",
    "answer_refusal",
    "personal_preference",
    "test_time_learning",
    "visual_reasoning",
    "multi_entity_reasoning",
    "unknown",
)


def _sql_values(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"


def _sql_text_array(values: tuple[str, ...]) -> str:
    return "ARRAY[" + ", ".join(f"'{value}'" for value in values) + "]::text[]"


def _metadata() -> sa.Column:
    return sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {RUNTIME}")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {EVALUATION}")

    op.create_table(
        "dataset_snapshots",
        sa.Column("snapshot_id", sa.String(128), primary_key=True),
        sa.Column("dataset", sa.String(32), nullable=False),
        sa.Column("dataset_version", sa.String(128), nullable=False),
        sa.Column("adapter_name", sa.String(128), nullable=False),
        sa.Column("adapter_version", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("evaluation_policy_version", sa.String(64), nullable=False),
        sa.Column("excluded_source_categories", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("source_content_hash", sa.String(128), nullable=False),
        sa.Column("canonical_content_hash", sa.String(128), nullable=False),
        sa.Column("counts", postgresql.JSONB(), nullable=False),
        sa.Column("category_counts", postgresql.JSONB(), nullable=False),
        sa.Column("eligible_category_counts", postgresql.JSONB(), nullable=False),
        sa.Column("excluded_category_counts", postgresql.JSONB(), nullable=False),
        sa.Column("manifest_json", postgresql.JSONB(), nullable=False),
        sa.Column("import_state", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True)),
        sa.Column("runtime_archive_uri", sa.Text()),
        sa.Column("evaluation_archive_uri", sa.Text()),
        sa.Column("validation_report_uri", sa.Text()),
        _metadata(),
        sa.CheckConstraint(
            f"dataset IN {_sql_values(DATASET_VALUES)}",
            name="dataset_values",
        ),
        sa.CheckConstraint(
            "import_state IN ('importing', 'ready', 'failed')",
            name="import_state_values",
        ),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_dataset_snapshots_dataset",
        "dataset_snapshots",
        ["dataset"],
        schema=RUNTIME,
    )

    op.create_table(
        "snapshot_import_attempts",
        sa.Column("attempt_id", sa.String(64), primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("run_id", sa.String(128)),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("imported_counts", postgresql.JSONB(), nullable=False),
        sa.Column("error_type", sa.String(256)),
        sa.Column("error_message", sa.Text()),
        _metadata(),
        sa.CheckConstraint(
            "status IN ('importing', 'ready', 'failed')",
            name="status_values",
        ),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_snapshot_import_attempts_run_id",
        "snapshot_import_attempts",
        ["run_id"],
        schema=RUNTIME,
    )

    op.create_table(
        "benchmark_sessions",
        sa.Column(
            "snapshot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(256), nullable=False),
        sa.Column("dataset", sa.String(32), nullable=False),
        sa.Column("conversation_id", sa.String(256), nullable=False),
        sa.Column("session_order", sa.Integer(), nullable=False),
        sa.Column("source_session_id", sa.String(256), nullable=False),
        sa.Column("source_date_raw", sa.Text()),
        sa.Column("source_date_iso", sa.String(64)),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        _metadata(),
        sa.PrimaryKeyConstraint("snapshot_id", "session_id"),
        sa.UniqueConstraint("snapshot_id", "conversation_id", "session_id"),
        sa.UniqueConstraint("snapshot_id", "conversation_id", "session_order"),
        sa.CheckConstraint(
            f"dataset IN {_sql_values(DATASET_VALUES)}",
            name="dataset_values",
        ),
        sa.CheckConstraint("session_order >= 0", name="session_order_nonnegative"),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_sessions_snapshot_conversation_order",
        "benchmark_sessions",
        ["snapshot_id", "conversation_id", "session_order"],
        schema=RUNTIME,
    )

    op.create_table(
        "assets",
        sa.Column(
            "snapshot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("asset_id", sa.String(256), nullable=False),
        sa.Column("asset_type", sa.String(64), nullable=False),
        sa.Column("storage_uri", sa.Text()),
        sa.Column("local_path", sa.Text()),
        sa.Column("remote_url", sa.Text()),
        sa.Column("sha256", sa.String(128)),
        sa.Column("mime_type", sa.String(256)),
        sa.Column("size_bytes", sa.BigInteger()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("source_id", sa.String(256)),
        sa.Column("run_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("storage_synced_at", sa.DateTime(timezone=True)),
        _metadata(),
        sa.PrimaryKeyConstraint("snapshot_id", "asset_id"),
        sa.CheckConstraint(
            f"asset_type IN {_sql_values(ASSET_TYPE_VALUES)}",
            name="asset_type_values",
        ),
        sa.CheckConstraint("size_bytes IS NULL OR size_bytes >= 0", name="size_nonnegative"),
        sa.CheckConstraint("width IS NULL OR width >= 0", name="width_nonnegative"),
        sa.CheckConstraint("height IS NULL OR height >= 0", name="height_nonnegative"),
        schema=RUNTIME,
    )
    op.create_index("ix_assets_sha256", "assets", ["sha256"], schema=RUNTIME)

    op.create_table(
        "images",
        sa.Column(
            "snapshot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("image_id", sa.String(256), nullable=False),
        sa.Column("asset_id", sa.String(256)),
        sa.Column("local_path", sa.Text()),
        sa.Column("remote_url", sa.Text()),
        sa.Column("storage_uri", sa.Text()),
        sa.Column("caption", sa.Text()),
        sa.Column("sha256", sa.String(128)),
        sa.Column("mime_type", sa.String(256)),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        _metadata(),
        sa.PrimaryKeyConstraint("snapshot_id", "image_id"),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "asset_id"],
            [f"{RUNTIME}.assets.snapshot_id", f"{RUNTIME}.assets.asset_id"],
        ),
        sa.CheckConstraint("width IS NULL OR width >= 0", name="width_nonnegative"),
        sa.CheckConstraint("height IS NULL OR height >= 0", name="height_nonnegative"),
        schema=RUNTIME,
    )

    op.create_table(
        "memory_units",
        sa.Column(
            "snapshot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("memory_id", sa.String(256), nullable=False),
        sa.Column("dataset", sa.String(32), nullable=False),
        sa.Column("conversation_id", sa.String(256), nullable=False),
        sa.Column("session_id", sa.String(256), nullable=False),
        sa.Column("turn_id", sa.String(256), nullable=False),
        sa.Column("timestamp_order", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(32), nullable=False),
        sa.Column("text", sa.Text()),
        sa.Column("caption", sa.Text()),
        sa.Column("entity_mentions", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("modalities", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("available_after_session", sa.String(256), nullable=False),
        sa.Column("token_count", sa.Integer()),
        _metadata(),
        sa.PrimaryKeyConstraint("snapshot_id", "memory_id"),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "session_id"],
            [
                f"{RUNTIME}.benchmark_sessions.snapshot_id",
                f"{RUNTIME}.benchmark_sessions.conversation_id",
                f"{RUNTIME}.benchmark_sessions.session_id",
            ],
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "available_after_session"],
            [
                f"{RUNTIME}.benchmark_sessions.snapshot_id",
                f"{RUNTIME}.benchmark_sessions.conversation_id",
                f"{RUNTIME}.benchmark_sessions.session_id",
            ],
        ),
        sa.UniqueConstraint("snapshot_id", "conversation_id", "turn_id"),
        sa.UniqueConstraint("snapshot_id", "conversation_id", "timestamp_order"),
        sa.CheckConstraint(
            f"dataset IN {_sql_values(DATASET_VALUES)}",
            name="dataset_values",
        ),
        sa.CheckConstraint(
            f"speaker IN {_sql_values(SPEAKER_VALUES)}",
            name="speaker_values",
        ),
        sa.CheckConstraint("timestamp_order >= 0", name="timestamp_order_nonnegative"),
        sa.CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="token_nonnegative",
        ),
        sa.CheckConstraint("cardinality(modalities) > 0", name="modalities_nonempty"),
        sa.CheckConstraint(
            f"modalities <@ {_sql_text_array(MODALITY_VALUES)}",
            name="modalities_values",
        ),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_memories_snapshot_conversation_order",
        "memory_units",
        ["snapshot_id", "conversation_id", "timestamp_order"],
        schema=RUNTIME,
    )

    op.create_table(
        "memory_unit_images",
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("memory_id", sa.String(256), nullable=False),
        sa.Column("image_id", sa.String(256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "memory_id", "image_id"),
        sa.UniqueConstraint("snapshot_id", "memory_id", "ordinal"),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "memory_id"],
            [f"{RUNTIME}.memory_units.snapshot_id", f"{RUNTIME}.memory_units.memory_id"],
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME}.images.snapshot_id", f"{RUNTIME}.images.image_id"],
        ),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        schema=RUNTIME,
    )

    op.create_table(
        "benchmark_queries",
        sa.Column(
            "snapshot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.dataset_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column("dataset", sa.String(32), nullable=False),
        sa.Column("conversation_id", sa.String(256), nullable=False),
        sa.Column("raw_query", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(64), nullable=False),
        sa.Column("query_subtype", sa.String(128)),
        sa.Column("available_after_session", sa.String(256)),
        sa.Column("eligible", sa.Boolean(), nullable=False),
        sa.Column("evaluation_policy_version", sa.String(64), nullable=False),
        sa.Column("exclusion_reason", sa.Text()),
        sa.Column("source_file", sa.Text(), nullable=False),
        sa.Column("source_record_index", sa.Integer()),
        sa.Column("source_query_id", sa.String(256)),
        sa.Column("scenario_id", sa.String(256)),
        sa.Column("qa_index", sa.Integer()),
        sa.Column("source_session_id", sa.String(256)),
        sa.Column("source_round_id", sa.String(256)),
        sa.Column("source_turn_index", sa.Integer()),
        sa.Column("source_category", sa.String(128)),
        sa.Column("source_subtype", sa.String(128)),
        sa.Column("query_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("source_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("eligibility_metadata", postgresql.JSONB(), nullable=False),
        _metadata(),
        sa.PrimaryKeyConstraint("snapshot_id", "query_id"),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "conversation_id", "available_after_session"],
            [
                f"{RUNTIME}.benchmark_sessions.snapshot_id",
                f"{RUNTIME}.benchmark_sessions.conversation_id",
                f"{RUNTIME}.benchmark_sessions.session_id",
            ],
        ),
        sa.CheckConstraint(
            f"dataset IN {_sql_values(DATASET_VALUES)}",
            name="dataset_values",
        ),
        sa.CheckConstraint(
            f"question_type IN {_sql_values(QUESTION_TYPE_VALUES)}",
            name="question_type_values",
        ),
        sa.CheckConstraint(
            "(eligible AND exclusion_reason IS NULL) OR "
            "(NOT eligible AND exclusion_reason IS NOT NULL)",
            name="eligibility_reason_consistent",
        ),
        sa.CheckConstraint(
            "source_record_index IS NULL OR source_record_index >= 0",
            name="source_record_index_nonnegative",
        ),
        sa.CheckConstraint("qa_index IS NULL OR qa_index >= 0", name="qa_index_nonnegative"),
        sa.CheckConstraint(
            "source_turn_index IS NULL OR source_turn_index >= 0",
            name="source_turn_index_nonnegative",
        ),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_queries_snapshot_eligibility_type",
        "benchmark_queries",
        ["snapshot_id", "eligible", "question_type"],
        schema=RUNTIME,
    )
    op.create_index(
        "ix_queries_snapshot_source_category_scenario",
        "benchmark_queries",
        ["snapshot_id", "source_category", "scenario_id"],
        schema=RUNTIME,
    )

    op.create_table(
        "query_images",
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column("image_id", sa.String(256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "query_id", "image_id"),
        sa.UniqueConstraint("snapshot_id", "query_id", "ordinal"),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [f"{RUNTIME}.benchmark_queries.snapshot_id", f"{RUNTIME}.benchmark_queries.query_id"],
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME}.images.snapshot_id", f"{RUNTIME}.images.image_id"],
        ),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        schema=RUNTIME,
    )

    op.create_table(
        "run_manifests",
        sa.Column("run_id", sa.String(128), primary_key=True),
        sa.Column("experiment_name", sa.String(256), nullable=False),
        sa.Column("dataset", sa.String(32), nullable=False),
        sa.Column("dataset_version", sa.String(128)),
        sa.Column("method", sa.String(128), nullable=False),
        sa.Column("paper_mode", sa.Boolean(), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("git_commit", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("manifest_json", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint(
            f"dataset IN {_sql_values(DATASET_VALUES)}",
            name="dataset_values",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "run_snapshot_links",
        sa.Column(
            "run_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.run_manifests.run_id"),
            primary_key=True,
        ),
        sa.Column(
            "snapshot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.dataset_snapshots.snapshot_id"),
            primary_key=True,
        ),
        schema=RUNTIME,
    )

    op.create_table(
        "query_gold",
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column("answer", sa.Text()),
        sa.Column("evidence_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("gold_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("record_metadata", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "query_id"),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [f"{RUNTIME}.benchmark_queries.snapshot_id", f"{RUNTIME}.benchmark_queries.query_id"],
        ),
        schema=EVALUATION,
    )

    _create_gold_child("query_gold_evidence_ids", "evidence_id", None)
    _create_gold_child("query_gold_memory_ids", "memory_id", "memory_units")
    _create_gold_child("query_gold_session_ids", "session_id", "benchmark_sessions")
    _create_gold_child("query_gold_image_ids", "image_id", "images")
    op.create_index(
        "ix_gold_evidence_snapshot_evidence",
        "query_gold_evidence_ids",
        ["snapshot_id", "evidence_id"],
        schema=EVALUATION,
    )
    op.create_index(
        "ix_gold_memory_snapshot_memory",
        "query_gold_memory_ids",
        ["snapshot_id", "memory_id"],
        schema=EVALUATION,
    )
    op.create_index(
        "ix_gold_session_snapshot_session",
        "query_gold_session_ids",
        ["snapshot_id", "session_id"],
        schema=EVALUATION,
    )
    op.create_index(
        "ix_gold_image_snapshot_image",
        "query_gold_image_ids",
        ["snapshot_id", "image_id"],
        schema=EVALUATION,
    )

    op.execute(f"REVOKE ALL ON SCHEMA {EVALUATION} FROM PUBLIC")
    op.execute(f"GRANT USAGE ON SCHEMA {RUNTIME} TO pmmc_runtime, pmmc_evaluator")
    op.execute(f"GRANT USAGE ON SCHEMA {EVALUATION} TO pmmc_evaluator")
    op.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {RUNTIME} TO pmmc_runtime")
    op.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {RUNTIME} TO pmmc_evaluator")
    op.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {EVALUATION} TO pmmc_evaluator")
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {RUNTIME} "
        "GRANT SELECT ON TABLES TO pmmc_runtime, pmmc_evaluator"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {EVALUATION} GRANT SELECT ON TABLES TO pmmc_evaluator"
    )


def _create_gold_child(table: str, id_column: str, runtime_table: str | None) -> None:
    constraints: list[sa.SchemaItem] = [
        sa.PrimaryKeyConstraint("snapshot_id", "query_id", id_column),
        sa.UniqueConstraint("snapshot_id", "query_id", "ordinal"),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [f"{EVALUATION}.query_gold.snapshot_id", f"{EVALUATION}.query_gold.query_id"],
        ),
    ]
    if runtime_table is not None:
        constraints.append(
            sa.ForeignKeyConstraint(
                ["snapshot_id", id_column],
                [
                    f"{RUNTIME}.{runtime_table}.snapshot_id",
                    f"{RUNTIME}.{runtime_table}.{id_column}",
                ],
            )
        )
    op.create_table(
        table,
        sa.Column("snapshot_id", sa.String(128), nullable=False),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column(id_column, sa.String(256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        *constraints,
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        schema=EVALUATION,
    )


def downgrade() -> None:
    # Intended only for disposable development databases.
    op.execute(f"DROP SCHEMA IF EXISTS {EVALUATION} CASCADE")
    op.execute(f"DROP SCHEMA IF EXISTS {RUNTIME} CASCADE")
