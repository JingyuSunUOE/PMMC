"""Internal SQLAlchemy models for experiment facts and evaluation results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import (
    APP_STAGE_VALUES,
    ARTIFACT_STATE_VALUES,
    ASSET_TYPE_VALUES,
    BAD_CASE_CATEGORY_VALUES,
    DISTANCE_METRIC_VALUES,
    EMBEDDING_COLLECTION_STATUS_VALUES,
    EMBEDDING_SOURCE_TYPE_VALUES,
    EVALUATION_SCHEMA,
    EVIDENCE_TYPE_VALUES,
    JOB_STATUS_VALUES,
    JUDGE_VERDICT_VALUES,
    METRIC_FAMILY_VALUES,
    RUN_STATUS_VALUES,
    RUNTIME_SCHEMA,
    SCORE_DIRECTION_VALUES,
    TRACE_STATUS_VALUES,
    sql_text_values,
)

JsonValue = dict[str, Any]


class RunStateModel(Base):
    __tablename__ = "run_states"
    __table_args__ = (
        ForeignKeyConstraint(
            ["error_artifact_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.run_artifacts.artifact_id",
                f"{RUNTIME_SCHEMA}.run_artifacts.run_id",
            ],
            use_alter=True,
            name="fk_run_states_error_artifact",
        ),
        CheckConstraint(f"status IN {sql_text_values(RUN_STATUS_VALUES)}", name="status_values"),
        CheckConstraint("selected_count >= 0", name="selected_count_nonnegative"),
        CheckConstraint("succeeded_count >= 0", name="succeeded_count_nonnegative"),
        CheckConstraint("failed_count >= 0", name="failed_count_nonnegative"),
        CheckConstraint(
            "succeeded_count + failed_count <= selected_count",
            name="result_counts_bounded",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    error_artifact_id: Mapped[str | None] = mapped_column(String(128))
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    succeeded_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RunQueryModel(Base):
    __tablename__ = "run_queries"
    __table_args__ = (
        PrimaryKeyConstraint("run_id", "snapshot_id", "query_id"),
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.benchmark_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.benchmark_queries.query_id",
            ],
        ),
        UniqueConstraint("run_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint("shard_count > 0", name="shard_count_positive"),
        CheckConstraint("shard_index >= 0 AND shard_index < shard_count", name="shard_valid"),
        {"schema": RUNTIME_SCHEMA},
    )

    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    selection_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    shard_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    shard_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RunStageSummaryModel(Base):
    __tablename__ = "run_stage_summaries"
    __table_args__ = (
        PrimaryKeyConstraint("run_id", "stage"),
        CheckConstraint("elapsed_ms >= 0", name="elapsed_nonnegative"),
        CheckConstraint(
            "processed >= 0 AND succeeded >= 0 AND failed >= 0", name="counts_nonnegative"
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False)
    processed: Mapped[int] = mapped_column(Integer, nullable=False)
    total: Mapped[int | None] = mapped_column(Integer)
    succeeded: Mapped[int] = mapped_column(Integer, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ExperimentJobModel(Base):
    __tablename__ = "experiment_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["error_artifact_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.run_artifacts.artifact_id",
                f"{RUNTIME_SCHEMA}.run_artifacts.run_id",
            ],
            use_alter=True,
            name="fk_experiment_jobs_error_artifact",
        ),
        ForeignKeyConstraint(
            ["run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.run_queries.run_id",
                f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.run_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["compiler_unit_id", "run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
                f"{RUNTIME_SCHEMA}.compiler_units.snapshot_id",
            ],
            use_alter=True,
            name="fk_experiment_jobs_compiler_unit",
        ),
        UniqueConstraint("job_id", "run_id"),
        UniqueConstraint("run_id", "snapshot_id", "query_id", "stage", "variant"),
        UniqueConstraint("run_id", "compiler_unit_id", "stage", "variant"),
        CheckConstraint(f"status IN {sql_text_values(JOB_STATUS_VALUES)}", name="status_values"),
        CheckConstraint(f"stage IN {sql_text_values(APP_STAGE_VALUES)}", name="stage_values"),
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
        CheckConstraint(
            "status != 'failed' OR error_type IS NOT NULL", name="failed_error_present"
        ),
        CheckConstraint(
            "(stage = 'compile' AND compiler_unit_id IS NOT NULL AND query_id IS NULL) OR "
            "(stage != 'compile' AND compiler_unit_id IS NULL)",
            name="compiler_job_scope",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str | None] = mapped_column(String(256))
    compiler_unit_id: Mapped[str | None] = mapped_column(String(128))
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    variant: Mapped[str] = mapped_column(String(128), nullable=False)
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
    error_artifact_id: Mapped[str | None] = mapped_column(String(128))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class JobAttemptModel(Base):
    __tablename__ = "job_attempts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["error_artifact_id"],
            [f"{RUNTIME_SCHEMA}.run_artifacts.artifact_id"],
            use_alter=True,
            name="fk_job_attempts_error_artifact",
        ),
        UniqueConstraint("job_id", "attempt_number"),
        CheckConstraint(f"status IN {sql_text_values(JOB_STATUS_VALUES)}", name="status_values"),
        CheckConstraint("attempt_number > 0", name="attempt_number_positive"),
        CheckConstraint(
            "status NOT IN ('succeeded','failed','cancelled') OR finished_at IS NOT NULL",
            name="terminal_finished_present",
        ),
        CheckConstraint(
            "status != 'failed' OR error_type IS NOT NULL", name="failed_error_present"
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    attempt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.experiment_jobs.job_id"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_id: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    error_summary: Mapped[str | None] = mapped_column(Text)
    error_artifact_id: Mapped[str | None] = mapped_column(String(128))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


def _artifact_constraints(schema: str) -> tuple[Any, ...]:
    return (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        UniqueConstraint("artifact_id", "run_id"),
        *(
            (
                ForeignKeyConstraint(
                    ["run_id", "snapshot_id", "query_id"],
                    [
                        f"{RUNTIME_SCHEMA}.run_queries.run_id",
                        f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                        f"{RUNTIME_SCHEMA}.run_queries.query_id",
                    ],
                ),
            )
            if schema == RUNTIME_SCHEMA
            else ()
        ),
        CheckConstraint(
            f"artifact_type IN {sql_text_values(ASSET_TYPE_VALUES)}", name="type_values"
        ),
        CheckConstraint(f"state IN {sql_text_values(ARTIFACT_STATE_VALUES)}", name="state_values"),
        CheckConstraint("size_bytes IS NULL OR size_bytes >= 0", name="size_nonnegative"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_format"),
        CheckConstraint(
            "state != 'ready' OR "
            "(storage_uri IS NOT NULL AND size_bytes IS NOT NULL AND ready_at IS NOT NULL)",
            name="ready_fields_present",
        ),
        CheckConstraint(
            "(state = 'failed' AND error_type IS NOT NULL) OR "
            "(state != 'failed' AND error_type IS NULL)",
            name="error_state_consistent",
        ),
        CheckConstraint(
            "query_id IS NULL OR snapshot_id IS NOT NULL",
            name="query_snapshot_present",
        ),
        *(
            (
                CheckConstraint(
                    "((evaluation_run_id IS NOT NULL)::integer + "
                    "(semantic_run_id IS NOT NULL)::integer) = 1",
                    name="exactly_one_evaluation_parent",
                ),
            )
            if schema == EVALUATION_SCHEMA
            else ()
        ),
        {"schema": schema},
    )


class RuntimeArtifactModel(Base):
    __tablename__ = "run_artifacts"
    __table_args__ = _artifact_constraints(RUNTIME_SCHEMA)

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    snapshot_id: Mapped[str | None] = mapped_column(String(128))
    query_id: Mapped[str | None] = mapped_column(String(256))
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class EvaluationArtifactModel(Base):
    __tablename__ = "run_artifacts"
    __table_args__ = _artifact_constraints(EVALUATION_SCHEMA)

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id")
    )
    semantic_run_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.qbank_semantic_match_runs.semantic_run_id")
    )
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    snapshot_id: Mapped[str | None] = mapped_column(String(128))
    query_id: Mapped[str | None] = mapped_column(String(256))
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class EmbeddingCollectionModel(Base):
    __tablename__ = "embedding_collections"
    __table_args__ = (
        ForeignKeyConstraint(
            ["corpus_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_corpora.corpus_id",
                f"{RUNTIME_SCHEMA}.retrieval_corpora.snapshot_id",
            ],
        ),
        UniqueConstraint("collection_id", "dimension"),
        UniqueConstraint("collection_id", "snapshot_id"),
        CheckConstraint(
            f"source_type IN {sql_text_values(EMBEDDING_SOURCE_TYPE_VALUES)}", name="source_values"
        ),
        CheckConstraint(
            f"distance_metric IN {sql_text_values(DISTANCE_METRIC_VALUES)}", name="distance_values"
        ),
        CheckConstraint(
            f"status IN {sql_text_values(EMBEDDING_COLLECTION_STATUS_VALUES)}", name="status_values"
        ),
        CheckConstraint("dimension > 0", name="dimension_positive"),
        CheckConstraint(
            "(normalized AND normalization_method IS NOT NULL) OR "
            "(NOT normalized AND normalization_method IS NULL)",
            name="normalization_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    collection_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.dataset_snapshots.snapshot_id"), nullable=False
    )
    corpus_id: Mapped[str | None] = mapped_column(String(128))
    run_id: Mapped[str | None] = mapped_column(ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"))
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(128))
    model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    actual_model_id: Mapped[str | None] = mapped_column(String(512))
    model_revision: Mapped[str | None] = mapped_column(String(256))
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(32), nullable=False)
    normalized: Mapped[bool] = mapped_column(Boolean, nullable=False)
    normalization_method: Mapped[str | None] = mapped_column(String(128))
    preprocessing_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    index_type: Mapped[str | None] = mapped_column(String(64))
    index_params: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class EmbeddingModel(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["collection_id", "dimension"],
            [
                f"{RUNTIME_SCHEMA}.embedding_collections.collection_id",
                f"{RUNTIME_SCHEMA}.embedding_collections.dimension",
            ],
        ),
        ForeignKeyConstraint(
            ["retrieval_document_id"],
            [f"{RUNTIME_SCHEMA}.retrieval_documents.document_id"],
        ),
        ForeignKeyConstraint(
            ["qbank_id"],
            [f"{RUNTIME_SCHEMA}.qbank_questions.qbank_id"],
            use_alter=True,
            name="fk_embeddings_qbank_question",
        ),
        ForeignKeyConstraint(
            ["paraphrase_id"],
            [f"{RUNTIME_SCHEMA}.qbank_paraphrases.paraphrase_id"],
            use_alter=True,
            name="fk_embeddings_qbank_paraphrase",
        ),
        UniqueConstraint("collection_id", "source_id"),
        UniqueConstraint("embedding_id", "collection_id"),
        CheckConstraint("dimension > 0", name="dimension_positive"),
        CheckConstraint("vector_dims(embedding) = dimension", name="vector_dimension_matches"),
        CheckConstraint(
            "num_nonnulls(retrieval_document_id, qbank_id, paraphrase_id) <= 1",
            name="one_typed_compiler_source",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    embedding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_id: Mapped[str] = mapped_column(String(256), nullable=False)
    retrieval_document_id: Mapped[str | None] = mapped_column(String(128))
    qbank_id: Mapped[str | None] = mapped_column(String(128))
    paraphrase_id: Mapped[str | None] = mapped_column(String(128))
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    model_call_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.model_calls.model_call_id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


def _model_call_constraints(schema: str) -> tuple[Any, ...]:
    job_constraint = (
        ForeignKeyConstraint(
            ["job_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.experiment_jobs.job_id",
                f"{RUNTIME_SCHEMA}.experiment_jobs.run_id",
            ],
        )
        if schema == RUNTIME_SCHEMA
        else ForeignKeyConstraint(
            ["evaluation_job_id", "evaluation_run_id"],
            [
                f"{EVALUATION_SCHEMA}.evaluation_jobs.evaluation_job_id",
                f"{EVALUATION_SCHEMA}.evaluation_jobs.evaluation_run_id",
            ],
        )
    )
    compiler_unit_constraints = (
        (
            ForeignKeyConstraint(
            ["compiler_unit_id", "run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.compiler_units.compiler_unit_id",
                f"{RUNTIME_SCHEMA}.compiler_units.run_id",
                f"{RUNTIME_SCHEMA}.compiler_units.snapshot_id",
            ],
            use_alter=True,
            name="fk_runtime_model_calls_compiler_unit",
            ),
        )
        if schema == RUNTIME_SCHEMA
        else ()
    )
    semantic_constraints = (
        (
            ForeignKeyConstraint(
                ["semantic_job_id", "semantic_run_id"],
                [
                    f"{EVALUATION_SCHEMA}.qbank_semantic_match_jobs.semantic_job_id",
                    f"{EVALUATION_SCHEMA}.qbank_semantic_match_jobs.semantic_run_id",
                ],
                use_alter=True,
                name="fk_evaluation_model_calls_semantic_job",
            ),
            UniqueConstraint(
                "model_call_id",
                "semantic_run_id",
                name="uq_evaluation_model_calls_semantic_run",
            ),
            UniqueConstraint(
                "model_call_id",
                "semantic_run_id",
                "semantic_job_id",
                name="uq_evaluation_model_calls_semantic_scope",
            ),
            CheckConstraint(
                "(evaluation_run_id IS NOT NULL)::integer + "
                "(semantic_run_id IS NOT NULL)::integer = 1",
                name="exactly_one_evaluation_parent",
            ),
            CheckConstraint(
                "(semantic_run_id IS NULL AND semantic_job_id IS NULL) OR "
                "(semantic_run_id IS NOT NULL AND semantic_job_id IS NOT NULL)",
                name="semantic_scope_complete",
            ),
        )
        if schema == EVALUATION_SCHEMA
        else ()
    )
    return (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.run_snapshot_links.run_id",
                f"{RUNTIME_SCHEMA}.run_snapshot_links.snapshot_id",
            ],
        ),
        *(
            (
                ForeignKeyConstraint(
                    ["run_id", "snapshot_id", "query_id"],
                    [
                        f"{RUNTIME_SCHEMA}.run_queries.run_id",
                        f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                        f"{RUNTIME_SCHEMA}.run_queries.query_id",
                    ],
                ),
            )
            if schema == RUNTIME_SCHEMA
            else ()
        ),
        job_constraint,
        *compiler_unit_constraints,
        *semantic_constraints,
        ForeignKeyConstraint(
            ["parent_model_call_id", "run_id"],
            [
                f"{schema}.model_calls.model_call_id",
                f"{schema}.model_calls.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["input_asset_id", "run_id"],
            [
                f"{schema}.run_artifacts.artifact_id",
                f"{schema}.run_artifacts.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["output_asset_id", "run_id"],
            [
                f"{schema}.run_artifacts.artifact_id",
                f"{schema}.run_artifacts.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["context_assembly_id", "run_id"],
            [
                f"{RUNTIME_SCHEMA}.context_assemblies.context_assembly_id",
                f"{RUNTIME_SCHEMA}.context_assemblies.run_id",
            ],
        ),
        ForeignKeyConstraint(
            ["runtime_id"],
            [f"{RUNTIME_SCHEMA}.managed_runtime_sessions.runtime_id"],
        ),
        ForeignKeyConstraint(
            ["runtime_service_id", "runtime_id"],
            [
                f"{RUNTIME_SCHEMA}.managed_runtime_services.runtime_service_id",
                f"{RUNTIME_SCHEMA}.managed_runtime_services.runtime_id",
            ],
        ),
        UniqueConstraint("model_call_id", "run_id"),
        UniqueConstraint("model_call_id", "run_id", "query_id"),
        CheckConstraint(f"status IN {sql_text_values(TRACE_STATUS_VALUES)}", name="status_values"),
        CheckConstraint(f"stage IN {sql_text_values(APP_STAGE_VALUES)}", name="stage_values"),
        CheckConstraint("step_id IS NULL OR step_id >= 0", name="step_nonnegative"),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_nonnegative"
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0", name="output_tokens_nonnegative"
        ),
        CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0", name="total_tokens_nonnegative"
        ),
        CheckConstraint("cost_usd IS NULL OR cost_usd >= 0", name="cost_nonnegative"),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_nonnegative"),
        CheckConstraint("temperature IS NULL OR temperature >= 0", name="temperature_nonnegative"),
        CheckConstraint("top_p IS NULL OR (top_p >= 0 AND top_p <= 1)", name="top_p_probability"),
        CheckConstraint("seed IS NULL OR seed >= 0", name="seed_nonnegative"),
        CheckConstraint("repair_attempt_count >= 0", name="repair_count_nonnegative"),
        CheckConstraint("retry_count >= 0", name="retry_count_nonnegative"),
        CheckConstraint(
            "input_tokens IS NULL OR output_tokens IS NULL OR total_tokens IS NULL OR "
            "total_tokens = input_tokens + output_tokens",
            name="token_total_consistent",
        ),
        CheckConstraint(
            "status != 'failed' OR error_type IS NOT NULL",
            name="failed_error_present",
        ),
        CheckConstraint(
            "error_type IS NULL OR status != 'success'",
            name="error_not_success",
        ),
        CheckConstraint(
            "query_id IS NULL OR snapshot_id IS NOT NULL",
            name="query_snapshot_present",
        ),
        CheckConstraint(
            "parent_model_call_id IS NULL OR parent_model_call_id != model_call_id",
            name="parent_not_self",
        ),
        CheckConstraint(
            "runtime_service_id IS NULL OR runtime_id IS NOT NULL",
            name="runtime_service_runtime_present",
        ),
        *(
            (
                CheckConstraint(
                    "compiler_unit_id IS NULL OR "
                    "(job_id IS NOT NULL AND stage IN ('questioner','planner','doubter'))",
                    name="compiler_call_scope",
                ),
            )
            if schema == RUNTIME_SCHEMA
            else ()
        ),
        {"schema": schema},
    )


class RuntimeModelCallModel(Base):
    __tablename__ = "model_calls"
    __table_args__ = _model_call_constraints(RUNTIME_SCHEMA)

    model_call_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    snapshot_id: Mapped[str | None] = mapped_column(String(128))
    query_id: Mapped[str | None] = mapped_column(String(256))
    compiler_unit_id: Mapped[str | None] = mapped_column(String(128))
    conversation_id: Mapped[str | None] = mapped_column(String(256))
    job_id: Mapped[str | None] = mapped_column(String(128))
    qbank_id: Mapped[str | None] = mapped_column(String(256))
    plan_id: Mapped[str | None] = mapped_column(String(256))
    step_id: Mapped[int | None] = mapped_column(Integer)
    parent_model_call_id: Mapped[str | None] = mapped_column(String(128))
    context_assembly_id: Mapped[str | None] = mapped_column(String(128))
    runtime_id: Mapped[str | None] = mapped_column(String(128))
    runtime_service_id: Mapped[str | None] = mapped_column(String(256))
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(128))
    model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    actual_model_id: Mapped[str | None] = mapped_column(String(512))
    prompt_hash: Mapped[str | None] = mapped_column(String(128))
    input_asset_id: Mapped[str | None] = mapped_column(String(128))
    output_asset_id: Mapped[str | None] = mapped_column(String(128))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    temperature: Mapped[float | None] = mapped_column(Float)
    top_p: Mapped[float | None] = mapped_column(Float)
    seed: Mapped[int | None] = mapped_column(Integer)
    parsed_output: Mapped[JsonValue | None] = mapped_column(JSONB)
    parse_error: Mapped[str | None] = mapped_column(Text)
    repair_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    litellm_call_id: Mapped[str | None] = mapped_column(String(256))
    vllm_request_id: Mapped[str | None] = mapped_column(String(256))
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(256))
    provider_request_id: Mapped[str | None] = mapped_column(String(256))
    request_config: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    error_type: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class EvaluationModelCallModel(Base):
    __tablename__ = "model_calls"
    __table_args__ = _model_call_constraints(EVALUATION_SCHEMA)

    model_call_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id")
    )
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), nullable=False
    )
    snapshot_id: Mapped[str | None] = mapped_column(String(128))
    query_id: Mapped[str | None] = mapped_column(String(256))
    conversation_id: Mapped[str | None] = mapped_column(String(256))
    evaluation_job_id: Mapped[str | None] = mapped_column(String(128))
    semantic_run_id: Mapped[str | None] = mapped_column(String(128))
    semantic_job_id: Mapped[str | None] = mapped_column(String(128))
    qbank_id: Mapped[str | None] = mapped_column(String(256))
    plan_id: Mapped[str | None] = mapped_column(String(256))
    step_id: Mapped[int | None] = mapped_column(Integer)
    parent_model_call_id: Mapped[str | None] = mapped_column(String(128))
    context_assembly_id: Mapped[str | None] = mapped_column(String(128))
    runtime_id: Mapped[str | None] = mapped_column(String(128))
    runtime_service_id: Mapped[str | None] = mapped_column(String(256))
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(128))
    model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    actual_model_id: Mapped[str | None] = mapped_column(String(512))
    prompt_hash: Mapped[str | None] = mapped_column(String(128))
    input_asset_id: Mapped[str | None] = mapped_column(String(128))
    output_asset_id: Mapped[str | None] = mapped_column(String(128))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    temperature: Mapped[float | None] = mapped_column(Float)
    top_p: Mapped[float | None] = mapped_column(Float)
    seed: Mapped[int | None] = mapped_column(Integer)
    parsed_output: Mapped[JsonValue | None] = mapped_column(JSONB)
    parse_error: Mapped[str | None] = mapped_column(Text)
    repair_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    litellm_call_id: Mapped[str | None] = mapped_column(String(256))
    vllm_request_id: Mapped[str | None] = mapped_column(String(256))
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(256))
    provider_request_id: Mapped[str | None] = mapped_column(String(256))
    request_config: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    error_type: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RetrievalBatchModel(Base):
    __tablename__ = "retrieval_batches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.run_queries.run_id",
                f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.run_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["embedding_collection_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.embedding_collections.collection_id",
                f"{RUNTIME_SCHEMA}.embedding_collections.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["query_embedding_collection_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.embedding_collections.collection_id",
                f"{RUNTIME_SCHEMA}.embedding_collections.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["corpus_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_corpora.corpus_id",
                f"{RUNTIME_SCHEMA}.retrieval_corpora.snapshot_id",
            ],
        ),
        ForeignKeyConstraint(
            ["reranker_model_call_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
                f"{RUNTIME_SCHEMA}.model_calls.query_id",
            ],
        ),
        UniqueConstraint(
            "retrieval_batch_id",
            "run_id",
            "query_id",
            "method",
            "retriever_name",
        ),
        UniqueConstraint("retrieval_batch_id", "run_id", "snapshot_id", "query_id"),
        CheckConstraint("requested_top_k > 0", name="topk_positive"),
        CheckConstraint(
            "returned_count >= 0 AND returned_count <= requested_top_k", name="returned_bounded"
        ),
        CheckConstraint(f"status IN {sql_text_values(TRACE_STATUS_VALUES)}", name="status_values"),
        CheckConstraint(f"stage IN {sql_text_values(APP_STAGE_VALUES)}", name="stage_values"),
        CheckConstraint("step_id IS NULL OR step_id >= 0", name="step_nonnegative"),
        CheckConstraint(
            "embed_latency_ms IS NULL OR embed_latency_ms >= 0",
            name="embed_latency_nonnegative",
        ),
        CheckConstraint(
            "search_latency_ms IS NULL OR search_latency_ms >= 0",
            name="search_latency_nonnegative",
        ),
        CheckConstraint(
            "rerank_latency_ms IS NULL OR rerank_latency_ms >= 0",
            name="rerank_latency_nonnegative",
        ),
        CheckConstraint(
            "total_latency_ms IS NULL OR total_latency_ms >= 0",
            name="total_latency_nonnegative",
        ),
        CheckConstraint(
            "status != 'failed' OR error_type IS NOT NULL", name="failed_error_present"
        ),
        CheckConstraint(
            "error_type IS NULL OR status != 'success'",
            name="error_not_success",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    retrieval_batch_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    retriever_name: Mapped[str] = mapped_column(String(256), nullable=False)
    index_name: Mapped[str | None] = mapped_column(String(256))
    chunking_strategy: Mapped[str | None] = mapped_column(String(256))
    requested_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    returned_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_collection_id: Mapped[str | None] = mapped_column(String(128))
    query_embedding_collection_id: Mapped[str | None] = mapped_column(String(128))
    corpus_id: Mapped[str | None] = mapped_column(String(128))
    reranker_model_call_id: Mapped[str | None] = mapped_column(String(128))
    plan_id: Mapped[str | None] = mapped_column(String(256))
    step_id: Mapped[int | None] = mapped_column(Integer)
    embed_latency_ms: Mapped[float | None] = mapped_column(Float)
    search_latency_ms: Mapped[float | None] = mapped_column(Float)
    rerank_latency_ms: Mapped[float | None] = mapped_column(Float)
    total_latency_ms: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RetrievalTraceModel(Base):
    __tablename__ = "retrieval_traces"
    __table_args__ = (
        UniqueConstraint("retrieval_batch_id", "rank"),
        UniqueConstraint("retrieval_trace_id", "run_id", "query_id"),
        ForeignKeyConstraint(
            [
                "retrieval_batch_id",
                "run_id",
                "query_id",
                "method",
                "retriever_name",
            ],
            [
                f"{RUNTIME_SCHEMA}.retrieval_batches.retrieval_batch_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.run_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.query_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.method",
                f"{RUNTIME_SCHEMA}.retrieval_batches.retriever_name",
            ],
        ),
        ForeignKeyConstraint(
            ["retrieval_document_id"],
            [f"{RUNTIME_SCHEMA}.retrieval_documents.document_id"],
        ),
        ForeignKeyConstraint(
            ["retrieval_window_id"],
            [f"{RUNTIME_SCHEMA}.memory_windows.window_id"],
        ),
        CheckConstraint("rank > 0", name="rank_positive"),
        CheckConstraint(
            f"retrieved_type IN {sql_text_values(EVIDENCE_TYPE_VALUES)}",
            name="retrieved_type_values",
        ),
        CheckConstraint(
            "score_direction IS NULL OR "
            f"score_direction IN {sql_text_values(SCORE_DIRECTION_VALUES)}",
            name="score_direction_values",
        ),
        CheckConstraint(
            "(score_type IS NULL AND score_raw_value IS NULL AND "
            "score_normalized_value IS NULL AND score_direction IS NULL AND "
            "normalization_method IS NULL AND normalization_version IS NULL) OR "
            "(score_type IS NOT NULL AND score_raw_value IS NOT NULL AND "
            "score_direction IS NOT NULL)",
            name="score_fields_consistent",
        ),
        CheckConstraint(
            "(score_normalized_value IS NULL AND normalization_method IS NULL) OR "
            "(score_normalized_value IS NOT NULL AND normalization_method IS NOT NULL)",
            name="score_normalization_consistent",
        ),
        CheckConstraint(
            "normalization_version IS NULL OR normalization_method IS NOT NULL",
            name="score_normalization_version_consistent",
        ),
        CheckConstraint(
            "score_normalized_value IS NULL OR "
            "(score_normalized_value >= 0 AND score_normalized_value <= 1)",
            name="normalized_score_probability",
        ),
        CheckConstraint(
            "(retrieval_window_id IS NOT NULL AND retrieval_document_id IS NOT NULL "
            "AND retrieved_id = retrieval_window_id) OR "
            "(retrieval_window_id IS NULL AND "
            "(retrieval_document_id IS NULL OR retrieved_id = retrieval_document_id))",
            name="retrieval_target_consistent",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    retrieval_trace_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    retrieval_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    retriever_name: Mapped[str] = mapped_column(String(256), nullable=False)
    index_name: Mapped[str | None] = mapped_column(String(256))
    chunking_strategy: Mapped[str | None] = mapped_column(String(256))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieved_id: Mapped[str] = mapped_column(String(256), nullable=False)
    retrieval_document_id: Mapped[str | None] = mapped_column(String(128))
    retrieval_window_id: Mapped[str | None] = mapped_column(String(128))
    retrieved_type: Mapped[str] = mapped_column(String(64), nullable=False)
    score_type: Mapped[str | None] = mapped_column(String(128))
    score_raw_value: Mapped[float | None] = mapped_column(Float)
    score_normalized_value: Mapped[float | None] = mapped_column(Float)
    score_direction: Mapped[str | None] = mapped_column(String(32))
    normalization_method: Mapped[str | None] = mapped_column(String(128))
    normalization_version: Mapped[str | None] = mapped_column(String(128))
    score_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RetrievalBatchQueryEmbeddingModel(Base):
    __tablename__ = "retrieval_batch_query_embeddings"
    __table_args__ = (
        PrimaryKeyConstraint("retrieval_batch_id", "embedding_id"),
        UniqueConstraint("retrieval_batch_id", "role", "ordinal"),
        ForeignKeyConstraint(
            ["retrieval_batch_id", "run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_batches.retrieval_batch_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.run_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.snapshot_id",
                f"{RUNTIME_SCHEMA}.retrieval_batches.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["embedding_id", "query_embedding_collection_id"],
            [
                f"{RUNTIME_SCHEMA}.embeddings.embedding_id",
                f"{RUNTIME_SCHEMA}.embeddings.collection_id",
            ],
        ),
        CheckConstraint("role IN ('text','image','joint','fused')", name="role_values"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    retrieval_batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_embedding_collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    used_for_search: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RetrievalTraceComponentModel(Base):
    __tablename__ = "retrieval_trace_components"
    __table_args__ = (
        PrimaryKeyConstraint("retrieval_trace_id", "retrieval_document_id"),
        UniqueConstraint("retrieval_trace_id", "ordinal"),
        ForeignKeyConstraint(
            ["retrieval_trace_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_traces.retrieval_trace_id",
                f"{RUNTIME_SCHEMA}.retrieval_traces.run_id",
                f"{RUNTIME_SCHEMA}.retrieval_traces.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["retrieval_document_id"],
            [f"{RUNTIME_SCHEMA}.retrieval_documents.document_id"],
        ),
        ForeignKeyConstraint(
            ["embedding_collection_id", "snapshot_id"],
            [
                f"{RUNTIME_SCHEMA}.embedding_collections.collection_id",
                f"{RUNTIME_SCHEMA}.embedding_collections.snapshot_id",
            ],
        ),
        CheckConstraint(
            "representation IN ('text_caption','text','image','multimodal')",
            name="representation_values",
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        CheckConstraint(
            "score_direction IN ('higher_is_better','lower_is_better')",
            name="score_direction_values",
        ),
        CheckConstraint(
            "(score_normalized_value IS NULL AND normalization_method IS NULL) OR "
            "(score_normalized_value IS NOT NULL AND normalization_method IS NOT NULL)",
            name="score_normalization_consistent",
        ),
        CheckConstraint(
            "score_normalized_value IS NULL OR "
            "(score_normalized_value >= 0 AND score_normalized_value <= 1)",
            name="normalized_score_probability",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    retrieval_trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    retrieval_document_id: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    representation: Mapped[str] = mapped_column(String(32), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    score_type: Mapped[str] = mapped_column(String(128), nullable=False)
    score_raw_value: Mapped[float] = mapped_column(Float, nullable=False)
    score_normalized_value: Mapped[float | None] = mapped_column(Float)
    score_direction: Mapped[str] = mapped_column(String(32), nullable=False)
    normalization_method: Mapped[str | None] = mapped_column(String(128))
    normalization_version: Mapped[str | None] = mapped_column(String(128))
    score_metadata: Mapped[JsonValue] = mapped_column(JSONB, nullable=False, default=dict)
    is_winner: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class FinalAnswerModel(Base):
    __tablename__ = "final_answers"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "snapshot_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.run_queries.run_id",
                f"{RUNTIME_SCHEMA}.run_queries.snapshot_id",
                f"{RUNTIME_SCHEMA}.run_queries.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["model_call_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.model_calls.model_call_id",
                f"{RUNTIME_SCHEMA}.model_calls.run_id",
                f"{RUNTIME_SCHEMA}.model_calls.query_id",
            ],
        ),
        UniqueConstraint("run_id", "query_id", "answer_variant"),
        UniqueConstraint("answer_id", "run_id", "query_id"),
        CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="input_nonnegative"),
        CheckConstraint("output_tokens IS NULL OR output_tokens >= 0", name="output_nonnegative"),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )

    answer_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    answer_variant: Mapped[str] = mapped_column(String(128), nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    answer_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    actual_model_id: Mapped[str | None] = mapped_column(String(512))
    final_answer: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_answer: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    model_call_id: Mapped[str | None] = mapped_column(String(128))
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class FinalAnswerEvidenceIdModel(Base):
    __tablename__ = "final_answer_evidence_ids"
    __table_args__ = (
        UniqueConstraint("answer_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )
    answer_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.final_answers.answer_id"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class FinalAnswerImageIdModel(Base):
    __tablename__ = "final_answer_image_ids"
    __table_args__ = (
        UniqueConstraint("answer_id", "ordinal"),
        ForeignKeyConstraint(
            ["snapshot_id", "image_id"],
            [f"{RUNTIME_SCHEMA}.images.snapshot_id", f"{RUNTIME_SCHEMA}.images.image_id"],
        ),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": RUNTIME_SCHEMA},
    )
    answer_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.final_answers.answer_id"), primary_key=True
    )
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    image_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class RetrievalEvaluationModel(Base):
    __tablename__ = "retrieval_evaluations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["retrieval_trace_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.retrieval_traces.retrieval_trace_id",
                f"{RUNTIME_SCHEMA}.retrieval_traces.run_id",
                f"{RUNTIME_SCHEMA}.retrieval_traces.query_id",
            ],
        ),
        UniqueConstraint("evaluation_run_id", "retrieval_trace_id"),
        {"schema": EVALUATION_SCHEMA},
    )
    retrieval_evaluation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    retrieval_trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    is_gold_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RetrievalEvaluationMatchModel(Base):
    __tablename__ = "retrieval_evaluation_matches"
    __table_args__ = (
        UniqueConstraint("retrieval_evaluation_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        {"schema": EVALUATION_SCHEMA},
    )
    retrieval_evaluation_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.retrieval_evaluations.retrieval_evaluation_id"),
        primary_key=True,
    )
    gold_evidence_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class MetricScoreModel(Base):
    __tablename__ = "metric_scores"
    __table_args__ = (
        CheckConstraint("value IS NULL OR (value >= 0 AND value <= 1)", name="value_probability"),
        CheckConstraint(
            f"metric_family IN {sql_text_values(METRIC_FAMILY_VALUES)}",
            name="metric_family_values",
        ),
        CheckConstraint("k IS NULL OR k > 0", name="k_positive"),
        CheckConstraint(
            "(applicable AND value IS NOT NULL AND unavailable_reason IS NULL) OR "
            "(NOT applicable AND value IS NULL AND unavailable_reason IS NOT NULL)",
            name="applicability_consistent",
        ),
        UniqueConstraint(
            "evaluation_run_id", "answer_id", "metric_name", "metric_version"
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
    metric_score_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_family: Mapped[str] = mapped_column(String(32), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_version: Mapped[str] = mapped_column(String(128), nullable=False)
    k: Mapped[int | None] = mapped_column(Integer)
    scorer: Mapped[str | None] = mapped_column(String(256))
    applicable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    unavailable_reason: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class JudgeScoreModel(Base):
    __tablename__ = "judge_scores"
    __table_args__ = (
        UniqueConstraint(
            "evaluation_run_id",
            "answer_id",
            "judge_model_alias",
            "rubric_version",
            "repeat_index",
        ),
        CheckConstraint("repeat_index >= 0", name="repeat_nonnegative"),
        CheckConstraint("f1 IS NULL OR (f1 >= 0 AND f1 <= 1)", name="f1_probability"),
        CheckConstraint("bleu1 IS NULL OR (bleu1 >= 0 AND bleu1 <= 1)", name="bleu_probability"),
        CheckConstraint(
            "llm_judge_score IS NULL OR (llm_judge_score >= 0 AND llm_judge_score <= 1)",
            name="judge_probability",
        ),
        CheckConstraint(
            f"bad_case_category IN {sql_text_values(BAD_CASE_CATEGORY_VALUES)}",
            name="bad_case_category_values",
        ),
        CheckConstraint(
            f"verdict IS NULL OR verdict IN {sql_text_values(JUDGE_VERDICT_VALUES)}",
            name="verdict_values",
        ),
        ForeignKeyConstraint(
            ["answer_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        ForeignKeyConstraint(
            ["model_call_id", "run_id", "query_id"],
            [
                f"{EVALUATION_SCHEMA}.model_calls.model_call_id",
                f"{EVALUATION_SCHEMA}.model_calls.run_id",
                f"{EVALUATION_SCHEMA}.model_calls.query_id",
            ],
        ),
        {"schema": EVALUATION_SCHEMA},
    )
    judge_score_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    judge_actual_model_id: Mapped[str | None] = mapped_column(String(512))
    judge_prompt_hash: Mapped[str | None] = mapped_column(String(128))
    model_call_id: Mapped[str | None] = mapped_column(String(128))
    rubric_version: Mapped[str] = mapped_column(String(128), nullable=False)
    repeat_index: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict: Mapped[str | None] = mapped_column(String(32))
    f1: Mapped[float | None] = mapped_column(Float)
    bleu1: Mapped[float | None] = mapped_column(Float)
    exact_match: Mapped[bool | None] = mapped_column(Boolean)
    llm_judge_score: Mapped[float | None] = mapped_column(Float)
    judge_reason: Mapped[str | None] = mapped_column(Text)
    bad_case_category: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class BadCaseLabelModel(Base):
    __tablename__ = "bad_case_labels"
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_probability",
        ),
        CheckConstraint(
            f"category IN {sql_text_values(BAD_CASE_CATEGORY_VALUES)}",
            name="category_values",
        ),
        ForeignKeyConstraint(
            ["answer_id", "run_id", "query_id"],
            [
                f"{RUNTIME_SCHEMA}.final_answers.answer_id",
                f"{RUNTIME_SCHEMA}.final_answers.run_id",
                f"{RUNTIME_SCHEMA}.final_answers.query_id",
            ],
        ),
        UniqueConstraint(
            "evaluation_run_id", "answer_id", "category", "label_source", "rule_version"
        ),
        ForeignKeyConstraint(
            ["rationale_asset_id", "run_id"],
            [
                f"{EVALUATION_SCHEMA}.run_artifacts.artifact_id",
                f"{EVALUATION_SCHEMA}.run_artifacts.run_id",
            ],
        ),
        {"schema": EVALUATION_SCHEMA},
    )
    bad_case_label_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EVALUATION_SCHEMA}.evaluation_runs.evaluation_run_id"), nullable=False
    )
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    label_source: Mapped[str] = mapped_column(String(128), nullable=False)
    rule_version: Mapped[str | None] = mapped_column(String(128))
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False)
    annotator_id: Mapped[str | None] = mapped_column(String(256))
    confidence: Mapped[float | None] = mapped_column(Float)
    rationale_asset_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


Index(
    "ix_jobs_claim",
    ExperimentJobModel.status,
    ExperimentJobModel.available_at,
    ExperimentJobModel.priority,
)
Index("ix_model_calls_run_stage", RuntimeModelCallModel.run_id, RuntimeModelCallModel.stage)
Index("ix_retrieval_batches_run_query", RetrievalBatchModel.run_id, RetrievalBatchModel.query_id)
Index(
    "ix_retrieval_traces_batch_rank",
    RetrievalTraceModel.retrieval_batch_id,
    RetrievalTraceModel.rank,
)
Index("ix_final_answers_run_query", FinalAnswerModel.run_id, FinalAnswerModel.query_id)
