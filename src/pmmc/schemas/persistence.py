"""Experiment persistence contracts shared by runners and infrastructure."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import Field, field_validator, model_validator

from pmmc.schemas.base import (
    AppStage,
    AssetType,
    BaseSchema,
    FiniteFloat,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    UtcDateTime,
    utc_now,
)


class DataScope(StrEnum):
    """Database and object-storage visibility boundary."""

    RUNTIME = "runtime"
    EVALUATION = "evaluation"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactState(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    L2 = "l2"


class EmbeddingSourceType(StrEnum):
    MEMORY_TEXT = "memory_text"
    MEMORY_CAPTION = "memory_caption"
    MEMORY_IMAGE = "memory_image"
    MEMORY_IMAGE_TEXT_PAIR = "memory_image_text_pair"
    QBANK_QUESTION = "qbank_question"
    QBANK_PARAPHRASE = "qbank_paraphrase"
    QUERY = "query"
    RETRIEVAL_DOCUMENT_TEXT_CAPTION = "retrieval_document_text_caption"
    RETRIEVAL_DOCUMENT_TEXT = "retrieval_document_text"
    RETRIEVAL_DOCUMENT_IMAGE = "retrieval_document_image"
    RETRIEVAL_DOCUMENT_MULTIMODAL = "retrieval_document_multimodal"
    RETRIEVAL_DOCUMENT_CONTRASTIVE = "retrieval_document_contrastive"


class EmbeddingCollectionStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class RunState(BaseSchema):
    """Mutable lifecycle state kept separate from an immutable run manifest."""

    run_id: NonEmptyStr
    status: RunStatus = RunStatus.CREATED
    started_at: UtcDateTime | None = None
    heartbeat_at: UtcDateTime | None = None
    completed_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    error_artifact_id: NonEmptyStr | None = None
    selected_count: NonNegativeInt = 0
    succeeded_count: NonNegativeInt = 0
    failed_count: NonNegativeInt = 0
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> RunState:
        if self.succeeded_count + self.failed_count > self.selected_count:
            raise ValueError("run result counts cannot exceed selected_count")
        if self.status is RunStatus.RUNNING and self.started_at is None:
            raise ValueError("running runs require started_at")
        if self.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            if self.completed_at is None:
                raise ValueError("terminal runs require completed_at")
        if self.status is RunStatus.COMPLETED:
            if self.succeeded_count != self.selected_count or self.failed_count != 0:
                raise ValueError("completed runs require every selected query to succeed")
        if self.status is RunStatus.FAILED and self.error_type is None:
            raise ValueError("failed runs require error_type")
        if self.completed_at is not None and self.started_at is not None:
            if self.completed_at < self.started_at:
                raise ValueError("completed_at cannot precede started_at")
        return self


class RunQueryAssignment(BaseSchema):
    """One runtime-safe query selected into an experiment run."""

    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr
    ordinal: NonNegativeInt
    selection_hash: NonEmptyStr
    shard_count: PositiveInt = 1
    shard_index: NonNegativeInt = 0
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_shard(self) -> RunQueryAssignment:
        if self.shard_index >= self.shard_count:
            raise ValueError("shard_index must be less than shard_count")
        return self


class ExperimentJob(BaseSchema):
    """Persisted unit of benchmark work claimable by one worker."""

    job_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    query_id: NonEmptyStr | None = None
    compiler_unit_id: NonEmptyStr | None = None
    stage: AppStage
    variant: NonEmptyStr = "primary"
    status: JobStatus = JobStatus.PENDING
    priority: NonNegativeInt = 0
    attempt_count: NonNegativeInt = 0
    max_attempts: PositiveInt = 1
    claimed_by: NonEmptyStr | None = None
    lease_expires_at: UtcDateTime | None = None
    available_at: UtcDateTime = Field(default_factory=utc_now)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    error_artifact_id: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_job(self) -> ExperimentJob:
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count cannot exceed max_attempts")
        if self.status is JobStatus.RUNNING:
            if self.claimed_by is None or self.lease_expires_at is None:
                raise ValueError("running jobs require claimed_by and lease_expires_at")
        if self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            if self.finished_at is None:
                raise ValueError("terminal jobs require finished_at")
        if self.status is JobStatus.FAILED and self.error_type is None:
            raise ValueError("failed jobs require error_type")
        if self.stage is AppStage.COMPILE:
            if self.query_id is not None or self.compiler_unit_id is None:
                raise ValueError(
                    "compile jobs require compiler_unit_id and cannot reference a query"
                )
        elif self.compiler_unit_id is not None:
            raise ValueError("compiler_unit_id is only valid for compile jobs")
        return self


class JobAttempt(BaseSchema):
    """Immutable audit record for one job execution attempt."""

    attempt_id: NonEmptyStr
    job_id: NonEmptyStr
    attempt_number: PositiveInt
    worker_id: NonEmptyStr
    status: JobStatus = JobStatus.RUNNING
    started_at: UtcDateTime = Field(default_factory=utc_now)
    finished_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    error_summary: NonEmptyStr | None = None
    error_artifact_id: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_attempt(self) -> JobAttempt:
        if self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            if self.finished_at is None:
                raise ValueError("terminal attempts require finished_at")
        if self.status is JobStatus.FAILED and self.error_type is None:
            raise ValueError("failed attempts require error_type")
        return self


class ExperimentArtifact(BaseSchema):
    """Metadata for one immutable run artifact stored outside Postgres."""

    artifact_id: NonEmptyStr
    run_id: NonEmptyStr | None
    evaluation_run_id: NonEmptyStr | None = None
    semantic_run_id: NonEmptyStr | None = None
    snapshot_id: NonEmptyStr | None = None
    query_id: NonEmptyStr | None = None
    artifact_type: AssetType
    data_scope: DataScope
    state: ArtifactState = ArtifactState.PENDING
    object_key: NonEmptyStr
    storage_uri: NonEmptyStr | None = None
    sha256: NonEmptyStr
    size_bytes: NonNegativeInt | None = None
    mime_type: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    ready_at: UtcDateTime | None = None
    error_type: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @field_validator("object_key")
    @classmethod
    def _validate_object_key(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
            raise ValueError("object_key must be a safe relative POSIX path")
        return value

    @model_validator(mode="after")
    def _validate_artifact(self) -> ExperimentArtifact:
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
        if self.state is ArtifactState.READY:
            if self.storage_uri is None or self.size_bytes is None or self.ready_at is None:
                raise ValueError("ready artifacts require URI, size, and ready_at")
        if self.state is ArtifactState.FAILED and self.error_type is None:
            raise ValueError("failed artifacts require error_type")
        if self.state is not ArtifactState.FAILED and self.error_type is not None:
            raise ValueError("error_type is only valid for failed artifacts")
        if self.query_id is not None and self.snapshot_id is None:
            raise ValueError("query-scoped artifacts require snapshot_id")
        if self.data_scope is DataScope.RUNTIME:
            if (
                self.run_id is None
                or self.evaluation_run_id is not None
                or self.semantic_run_id is not None
            ):
                raise ValueError(
                    "runtime artifacts require run_id and no evaluation parent"
                )
        else:
            parent_count = sum(
                value is not None
                for value in (self.evaluation_run_id, self.semantic_run_id)
            )
            if parent_count != 1:
                raise ValueError(
                    "evaluation artifacts require exactly one evaluation parent"
                )
            if self.query_id is not None and self.run_id is None:
                raise ValueError(
                    "query-scoped evaluation artifacts require a source run_id"
                )
        return self


class EmbeddingCollection(BaseSchema):
    """Versioned embedding space and its reproducibility metadata."""

    collection_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    corpus_id: NonEmptyStr | None = None
    run_id: NonEmptyStr | None = None
    source_type: EmbeddingSourceType
    provider: NonEmptyStr | None = None
    model_alias: NonEmptyStr
    actual_model_id: NonEmptyStr | None = None
    model_revision: NonEmptyStr | None = None
    dimension: PositiveInt
    distance_metric: DistanceMetric
    normalized: bool = False
    normalization_method: NonEmptyStr | None = None
    preprocessing_hash: NonEmptyStr
    index_type: NonEmptyStr | None = None
    index_params: JsonDict = Field(default_factory=dict)
    status: EmbeddingCollectionStatus = EmbeddingCollectionStatus.BUILDING
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_collection(self) -> EmbeddingCollection:
        if self.normalized and self.normalization_method is None:
            raise ValueError("normalized collections require normalization_method")
        if not self.normalized and self.normalization_method is not None:
            raise ValueError("normalization_method requires normalized=True")
        document_sources = {
            EmbeddingSourceType.RETRIEVAL_DOCUMENT_TEXT_CAPTION,
            EmbeddingSourceType.RETRIEVAL_DOCUMENT_TEXT,
            EmbeddingSourceType.RETRIEVAL_DOCUMENT_IMAGE,
            EmbeddingSourceType.RETRIEVAL_DOCUMENT_MULTIMODAL,
            EmbeddingSourceType.RETRIEVAL_DOCUMENT_CONTRASTIVE,
        }
        if self.source_type in document_sources and self.corpus_id is None:
            raise ValueError("retrieval-document collections require corpus_id")
        return self


class EmbeddingRecord(BaseSchema):
    """One finite vector associated with a canonical source ID."""

    embedding_id: NonEmptyStr
    collection_id: NonEmptyStr
    source_id: NonEmptyStr
    retrieval_document_id: NonEmptyStr | None = None
    qbank_id: NonEmptyStr | None = None
    paraphrase_id: NonEmptyStr | None = None
    input_hash: NonEmptyStr
    dimension: PositiveInt
    vector: list[FiniteFloat]
    model_call_id: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_vector(self) -> EmbeddingRecord:
        if len(self.vector) != self.dimension:
            raise ValueError("vector length must equal dimension")
        if (
            self.retrieval_document_id is not None
            and self.retrieval_document_id != self.source_id
        ):
            raise ValueError("retrieval_document_id must equal source_id when provided")
        typed_sources = (
            self.retrieval_document_id,
            self.qbank_id,
            self.paraphrase_id,
        )
        if sum(value is not None for value in typed_sources) > 1:
            raise ValueError("embedding records can reference only one typed source")
        if self.qbank_id is not None and self.qbank_id != self.source_id:
            raise ValueError("qbank_id must equal source_id when provided")
        if self.paraphrase_id is not None and self.paraphrase_id != self.source_id:
            raise ValueError("paraphrase_id must equal source_id when provided")
        return self
