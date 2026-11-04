"""Internal SQLAlchemy models for managed local model runtimes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import (
    CALIBRATION_STATUS_VALUES,
    MANAGED_BACKEND_VALUES,
    MANAGED_RUNTIME_STATUS_VALUES,
    MANAGED_SERVICE_STATUS_VALUES,
    MODEL_ROLE_VALUES,
    RUNTIME_SCHEMA,
    sql_text_values,
)

JsonValue = dict[str, Any]


class ManagedRuntimeSessionModel(Base):
    __tablename__ = "managed_runtime_sessions"
    __table_args__ = (
        CheckConstraint(
            f"status IN {sql_text_values(MANAGED_RUNTIME_STATUS_VALUES)}",
            name="status_values",
        ),
        CheckConstraint(
            "status NOT IN ('completed','failed','cancelled') OR completed_at IS NOT NULL",
            name="terminal_completed_present",
        ),
        CheckConstraint(
            "status != 'failed' OR error_type IS NOT NULL",
            name="failed_error_present",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    runtime_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    profile_name: Mapped[str] = mapped_column(String(128), nullable=False)
    profile_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    hardware_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    vllm_version: Mapped[str | None] = mapped_column(String(64))
    litellm_version: Mapped[str | None] = mapped_column(String(64))
    plan_artifact_id: Mapped[str | None] = mapped_column(String(128))
    summary_artifact_id: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(256))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class ManagedRuntimeServiceModel(Base):
    __tablename__ = "managed_runtime_services"
    __table_args__ = (
        UniqueConstraint("runtime_service_id", "runtime_id"),
        UniqueConstraint("runtime_id", "service_id", "stage_id"),
        CheckConstraint(
            f"role IN {sql_text_values(MODEL_ROLE_VALUES)}",
            name="role_values",
        ),
        CheckConstraint(
            f"backend IN {sql_text_values(MANAGED_BACKEND_VALUES)}",
            name="backend_values",
        ),
        CheckConstraint(
            f"status IN {sql_text_values(MANAGED_SERVICE_STATUS_VALUES)}",
            name="status_values",
        ),
        CheckConstraint(
            "observed_peak_memory_bytes IS NULL OR observed_peak_memory_bytes >= 0",
            name="peak_memory_nonnegative",
        ),
        CheckConstraint(
            "status != 'failed' OR error_type IS NOT NULL",
            name="failed_error_present",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    runtime_service_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    runtime_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.managed_runtime_sessions.runtime_id"),
        nullable=False,
    )
    service_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stage_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    backend: Mapped[str] = mapped_column(String(64), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    allocation_json: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    process_id: Mapped[int | None] = mapped_column(Integer)
    observed_peak_memory_bytes: Mapped[int | None] = mapped_column(BigInteger)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_code: Mapped[int | None] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(String(256))
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RuntimeGpuAllocationModel(Base):
    __tablename__ = "runtime_gpu_allocations"
    __table_args__ = (
        PrimaryKeyConstraint("runtime_service_id", "device_ordinal"),
        ForeignKeyConstraint(
            ["runtime_service_id", "runtime_id"],
            [
                f"{RUNTIME_SCHEMA}.managed_runtime_services.runtime_service_id",
                f"{RUNTIME_SCHEMA}.managed_runtime_services.runtime_id",
            ],
        ),
        CheckConstraint("device_ordinal >= 0", name="device_ordinal_nonnegative"),
        CheckConstraint("visible_index >= 0", name="visible_index_nonnegative"),
        CheckConstraint("tensor_parallel_size > 0", name="tp_positive"),
        CheckConstraint("memory_budget_bytes > 0", name="memory_budget_positive"),
        {"schema": RUNTIME_SCHEMA},
    )

    runtime_service_id: Mapped[str] = mapped_column(String(256), nullable=False)
    runtime_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stage_id: Mapped[str] = mapped_column(String(128), nullable=False)
    device_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    anonymized_device_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visible_index: Mapped[int] = mapped_column(Integer, nullable=False)
    gpu_name: Mapped[str] = mapped_column(String(256), nullable=False)
    total_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tensor_parallel_size: Mapped[int] = mapped_column(Integer, nullable=False)
    memory_budget_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    topology_score: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RuntimeCalibrationModel(Base):
    __tablename__ = "runtime_calibrations"
    __table_args__ = (
        CheckConstraint(
            f"status IN {sql_text_values(CALIBRATION_STATUS_VALUES)}",
            name="status_values",
        ),
        CheckConstraint(
            "observed_peak_memory_bytes >= 0 AND observed_idle_memory_bytes >= 0",
            name="memory_nonnegative",
        ),
        CheckConstraint(
            "status != 'failed' OR error_type IS NOT NULL",
            name="failed_error_present",
        ),
        {"schema": RUNTIME_SCHEMA},
    )

    calibration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    service_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(256), nullable=False)
    model_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    hardware_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    vllm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    allocation_json: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    observed_peak_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    observed_idle_memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    server_config: Mapped[JsonValue] = mapped_column(JSONB, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )


class RunRuntimeLinkModel(Base):
    __tablename__ = "run_runtime_links"
    __table_args__ = ({"schema": RUNTIME_SCHEMA},)

    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.run_manifests.run_id"), primary_key=True
    )
    runtime_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.managed_runtime_sessions.runtime_id"),
        primary_key=True,
    )
    linked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[JsonValue] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
