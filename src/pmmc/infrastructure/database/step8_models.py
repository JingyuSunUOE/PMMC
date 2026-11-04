"""Campaign orchestration facts; existing scientific model-call tables stay authoritative."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import RUNTIME_SCHEMA


class MethodProtocolFreezeModel(Base):
    __tablename__ = "method_protocol_freezes"
    __table_args__ = (
        CheckConstraint("decision IN ('approved','rejected')", name="decision_values"),
        {"schema": RUNTIME_SCHEMA},
    )
    freeze_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    protocol_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(256), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BenchmarkCampaignModel(Base):
    __tablename__ = "benchmark_campaigns"
    __table_args__ = (
        CheckConstraint("phase IN ('development','exploration','paper')", name="phase_values"),
        CheckConstraint(
            "status IN ('planned','running','failed','awaiting_approval','completed')",
            name="status_values",
        ),
        CheckConstraint("phase <> 'paper' OR freeze_hash IS NOT NULL", name="paper_freeze"),
        {"schema": RUNTIME_SCHEMA},
    )
    campaign_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    freeze_hash: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.method_protocol_freezes.freeze_hash"),
    )
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BenchmarkSelectionModel(Base):
    __tablename__ = "benchmark_selection_expansions"
    __table_args__ = ({"schema": RUNTIME_SCHEMA},)
    selection_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_campaigns.campaign_id"),
        nullable=False,
        index=True,
    )
    selection: Mapped[str] = mapped_column(String(32), nullable=False)
    parent_selection_hash: Mapped[str | None] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_selection_expansions.selection_hash"),
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ExperimentCellModel(Base):
    __tablename__ = "benchmark_experiment_cells"
    __table_args__ = ({"schema": RUNTIME_SCHEMA},)
    cell_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    method: Mapped[str] = mapped_column(String(64), nullable=False)
    answer_model: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class BenchmarkCellAssignmentModel(Base):
    __tablename__ = "benchmark_cell_assignments"
    __table_args__ = (
        UniqueConstraint("campaign_id", "cell_id", "query_id"),
        {"schema": RUNTIME_SCHEMA},
    )
    assignment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_campaigns.campaign_id"),
        nullable=False,
    )
    cell_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_experiment_cells.cell_id"),
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(String(256), nullable=False)
    first_selection_hash: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_selection_expansions.selection_hash"),
        nullable=False,
    )


class BenchmarkExecutionAttemptModel(Base):
    __tablename__ = "benchmark_execution_attempts"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running','completed','failed','interrupted','budget_exhausted')",
            name="status_values",
        ),
        {"schema": RUNTIME_SCHEMA},
    )
    attempt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_campaigns.campaign_id"),
        nullable=False,
        index=True,
    )
    execution_image: Mapped[str] = mapped_column(Text, nullable=False)
    resource_class: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    receipt_hash: Mapped[str | None] = mapped_column(String(64))


class BenchmarkStageJobModel(Base):
    __tablename__ = "benchmark_stage_jobs"
    __table_args__ = (
        UniqueConstraint("campaign_id", "component_key"),
        CheckConstraint(
            "status IN ('pending','running','completed','failed','unknown')",
            name="status_values",
        ),
        CheckConstraint("attempt_count >= 0", name="attempt_count_nonnegative"),
        CheckConstraint(
            "status <> 'completed' OR receipt_hash IS NOT NULL",
            name="completed_receipt",
        ),
        {"schema": RUNTIME_SCHEMA},
    )
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_campaigns.campaign_id"),
        nullable=False,
        index=True,
    )
    component_key: Mapped[str] = mapped_column(String(128), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_class: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    scientific_input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    dependency_ids: Mapped[list[str]] = mapped_column(ARRAY(String(64)), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    receipt_hash: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BenchmarkComponentReceiptModel(Base):
    __tablename__ = "benchmark_component_receipts"
    __table_args__ = ({"schema": RUNTIME_SCHEMA},)
    receipt_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    component_identity: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_stage_jobs.job_id"),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    artifact_hashes: Mapped[list[str]] = mapped_column(ARRAY(String(64)), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BenchmarkBudgetApprovalModel(Base):
    __tablename__ = "benchmark_budget_approvals"
    __table_args__ = (
        UniqueConstraint("campaign_id", "selection_hash"),
        UniqueConstraint("plan_hash"),
        CheckConstraint("limit_usd > 0", name="positive_limit"),
        CheckConstraint(
            "(plan_hash IS NULL AND plan_bindings IS NULL AND limit_usd IS NOT NULL)"
            " OR (plan_hash IS NOT NULL AND plan_bindings IS NOT NULL)",
            name="budget_authority",
        ),
        CheckConstraint("settled_usd >= 0 AND reserved_usd >= 0", name="spend_nonnegative"),
        CheckConstraint(
            "(stopped_at IS NULL AND stopped_by IS NULL AND stop_reason IS NULL)"
            " OR (plan_hash IS NOT NULL AND stopped_at IS NOT NULL"
            " AND stopped_by IS NOT NULL AND stop_reason IS NOT NULL)", name="stop_audit"
        ),
        {"schema": RUNTIME_SCHEMA},
    )
    approval_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_campaigns.campaign_id"),
        nullable=False,
    )
    selection_hash: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_selection_expansions.selection_hash"),
        nullable=False,
    )
    approver_id: Mapped[str] = mapped_column(String(256), nullable=False)
    rate_card_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_hash: Mapped[str | None] = mapped_column(String(64))
    plan_bindings: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_by: Mapped[str | None] = mapped_column(String(256))
    stop_reason: Mapped[str | None] = mapped_column(String(128))
    limit_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    settled_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=Decimal(0))
    reserved_usd: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False, default=Decimal(0)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BenchmarkSpendReservationModel(Base):
    __tablename__ = "benchmark_spend_reservations"
    __table_args__ = (
        CheckConstraint("reserved_usd >= 0", name="reservation_nonnegative"),
        CheckConstraint(
            "state IN ('reserved','settled','unknown','cancelled')", name="state_values"
        ),
        {"schema": RUNTIME_SCHEMA},
    )
    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    approval_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_budget_approvals.approval_id"),
        nullable=False,
    )
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    reserved_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    actual_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    response_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
