"""Execution-only configuration for reproducible research throughput."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    JsonDict,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    UtcDateTime,
)


class ExecutionSchedule(StrEnum):
    """Supported launcher schedules without changing scientific identity."""

    DATASET_MAJOR_V1 = "dataset-major-v1"
    MODEL_STAGE_MAJOR_V1 = "model-stage-major-v1"


class UnitPreparationCachePolicy(StrEnum):
    DISABLED = "disabled"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    REFRESH = "refresh"


class CheckpointPolicy(StrEnum):
    DISABLED = "disabled"
    MILESTONE = "milestone"


class UnitPreparationExecutionConfig(BaseSchema):
    workers: PositiveInt = 8
    max_tasks_per_child: PositiveInt = 16
    cache_policy: UnitPreparationCachePolicy = UnitPreparationCachePolicy.READ_WRITE
    counter_backend: NonEmptyStr = "qwen-vl-cached-exact-v1"
    local_asset_cache: bool = True


class QBankExecutionConfig(BaseSchema):
    text_query_batch_size: PositiveInt = 32
    multimodal_query_batch_size: PositiveInt = 8
    embedding_concurrency: PositiveInt = 4
    score_concurrency: PositiveInt = 8


class CheckpointExecutionConfig(BaseSchema):
    policy: CheckpointPolicy = CheckpointPolicy.MILESTONE
    interval_seconds: PositiveInt = 1800
    completed_jobs_interval: PositiveInt = 250
    retain_count: PositiveInt = 2


class ProgressExecutionConfig(BaseSchema):
    protocol: NonEmptyStr = "weighted-hierarchical-progress-v2"


class ReplicaPoolExecutionConfig(BaseSchema):
    compiler: PositiveInt = 1
    text_embedding: PositiveInt = 1
    contrastive_embedding: PositiveInt = 1
    multimodal_embedding: PositiveInt = 1
    answer: PositiveInt = 1
    judge: PositiveInt = 1

    def for_logical_stage(self, stage: str) -> int:
        return {
            "compiler": self.compiler,
            "embedding_text": self.text_embedding,
            "embedding_contrastive": self.contrastive_embedding,
            "embedding_multimodal": self.multimodal_embedding,
            "answer": self.answer,
            "judge": self.judge,
        }.get(stage, 1)

    @property
    def maximum(self) -> int:
        return max(
            self.compiler,
            self.text_embedding,
            self.contrastive_embedding,
            self.multimodal_embedding,
            self.answer,
            self.judge,
        )


class ExperimentExecutionConfig(BaseSchema):
    """Engineering controls excluded from method and Questioner identities."""

    execution_protocol: NonEmptyStr = "pmmc-throughput-v1"
    schedule: ExecutionSchedule = ExecutionSchedule.MODEL_STAGE_MAJOR_V1
    unit_preparation: UnitPreparationExecutionConfig = Field(
        default_factory=UnitPreparationExecutionConfig
    )
    qbank: QBankExecutionConfig = Field(default_factory=QBankExecutionConfig)
    checkpoint: CheckpointExecutionConfig = Field(
        default_factory=CheckpointExecutionConfig
    )
    progress: ProgressExecutionConfig = Field(default_factory=ProgressExecutionConfig)
    replicas: ReplicaPoolExecutionConfig = Field(
        default_factory=ReplicaPoolExecutionConfig
    )

    @model_validator(mode="after")
    def _validate_protocol(self) -> ExperimentExecutionConfig:
        if self.execution_protocol != "pmmc-throughput-v1":
            raise ValueError("unsupported execution protocol")
        if self.progress.protocol != "weighted-hierarchical-progress-v2":
            raise ValueError("unsupported progress protocol")
        return self

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class StagePerformanceRecord(BaseSchema):
    """Secret-free process and I/O facts for one launcher stage."""

    stage_id: NonEmptyStr
    started_at: UtcDateTime
    completed_at: UtcDateTime
    duration_seconds: NonNegativeFloat
    reused: bool = False
    cpu_user_seconds: NonNegativeFloat
    cpu_system_seconds: NonNegativeFloat
    peak_rss_bytes: NonNegativeInt
    read_bytes: NonNegativeInt
    write_bytes: NonNegativeInt
    model_startup_count: NonNegativeInt = 0
    metadata: JsonDict = Field(default_factory=dict)


def load_execution_config(path: Path) -> ExperimentExecutionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("execution config root must be a mapping")
    return ExperimentExecutionConfig.model_validate(raw)
