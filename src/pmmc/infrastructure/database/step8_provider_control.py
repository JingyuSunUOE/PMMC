"""Cross-worker rate windows and append-only, evidence-verified spend decisions.

Tables are created by the Step 8 additive migration, never at provider call time.
All admissions share DB time and a scope lock, including across campaigns/Pods.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from pmmc.compiler.ids import canonical_content_hash
from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import RUNTIME_SCHEMA
from pmmc.infrastructure.database.step8_models import (
    BenchmarkBudgetApprovalModel,
    BenchmarkSpendReservationModel,
)
from pmmc.models.step8_rate_limits import ProviderRateLimit, RateWindowEntry, admission_delay


class ProviderRateScopeModel(Base):
    __tablename__ = "benchmark_provider_rate_scopes"
    __table_args__ = ({"schema": RUNTIME_SCHEMA},)
    scope_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    blocked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProviderRateAdmissionModel(Base):
    __tablename__ = "benchmark_provider_rate_admissions"
    __table_args__ = (
        CheckConstraint("token_units > 0", name="tokens_positive"),
        Index("ix_provider_rate_scope_time", "scope_hash", "admitted_at"),
        Index("ix_provider_rate_request_hash", "request_hash"),
        {"schema": RUNTIME_SCHEMA},
    )
    dispatch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_hash: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_provider_rate_scopes.scope_hash"), nullable=False
    )
    token_units: Mapped[int] = mapped_column(Integer, nullable=False)
    admitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProviderReconciliationModel(Base):
    __tablename__ = "benchmark_provider_reconciliations"
    __table_args__ = (
        CheckConstraint("disposition IN ('completed','not_executed')", name="disposition_values"),
        {"schema": RUNTIME_SCHEMA},
    )
    request_id: Mapped[str] = mapped_column(
        ForeignKey(f"{RUNTIME_SCHEMA}.benchmark_spend_reservations.request_id"), primary_key=True
    )
    reconciliation_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    disposition: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


PROVIDER_CONTROL_TABLES = (
    ProviderRateScopeModel.__table__,
    ProviderRateAdmissionModel.__table__,
    ProviderReconciliationModel.__table__,
)


class PostgresProviderControlRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def _scope(
        self, session: AsyncSession, limit: ProviderRateLimit
    ) -> ProviderRateScopeModel:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
            {"scope": "step8-provider-rate:" + limit.scope_hash},
        )
        row = await session.get(ProviderRateScopeModel, limit.scope_hash)
        if row is None:
            row = ProviderRateScopeModel(scope_hash=limit.scope_hash, policy=limit.model_dump())
            session.add(row)
            await session.flush()
        elif row.policy != limit.model_dump():
            raise ValueError(
                "provider rate scope policy conflict; explicit reconfiguration required"
            )
        return row

    async def try_admit(
        self,
        limit: ProviderRateLimit,
        *,
        dispatch_id: str,
        token_units: int,
        request_hash: str | None = None,
    ) -> float:
        request_hash = request_hash or dispatch_id
        async with self.sessions.begin() as session:
            scope = await self._scope(session, limit)
            now: datetime = (await session.execute(text("SELECT clock_timestamp()"))).scalar_one()
            old = await session.get(ProviderRateAdmissionModel, dispatch_id)
            if old is not None:
                if (
                    old.scope_hash != limit.scope_hash
                    or old.token_units != token_units
                    or old.request_hash != request_hash
                ):
                    raise ValueError("provider admission identity conflict")
                # A delayed/restarted dispatch needs a fresh rate slot; returning
                # success for an expired old slot could cause a burst after resume.
                if (now - old.admitted_at).total_seconds() < 60:
                    return (
                        max(0.0, (scope.blocked_until - now).total_seconds())
                        if scope.blocked_until
                        else 0.0
                    )
                raise ValueError(
                    "expired provider admission requires explicit dispatch reconciliation"
                )
            rows = (
                (
                    await session.execute(
                        select(ProviderRateAdmissionModel).where(
                            ProviderRateAdmissionModel.scope_hash == limit.scope_hash,
                            ProviderRateAdmissionModel.admitted_at > now - timedelta(seconds=60),
                        )
                    )
                )
                .scalars()
                .all()
            )
            delay = admission_delay(
                limit,
                token_units=token_units,
                now=now,
                entries=[RateWindowEntry(row.admitted_at, row.token_units) for row in rows],
                blocked_until=scope.blocked_until,
            )
            if delay == 0:
                session.add(
                    ProviderRateAdmissionModel(
                        dispatch_id=dispatch_id,
                        request_hash=request_hash,
                        scope_hash=limit.scope_hash,
                        token_units=token_units,
                        admitted_at=now,
                    )
                )
            return delay

    async def defer(self, limit: ProviderRateLimit, *, seconds: float) -> None:
        if not 0 <= seconds < 365 * 86400:
            raise ValueError("invalid provider Retry-After duration")
        async with self.sessions.begin() as session:
            scope = await self._scope(session, limit)
            now: datetime = (await session.execute(text("SELECT clock_timestamp()"))).scalar_one()
            deadline = now + timedelta(seconds=seconds)
            if scope.blocked_until is None or deadline > scope.blocked_until:
                scope.blocked_until = deadline

    async def reconciliation(self, request_id: str, request_hash: str) -> dict[str, Any] | None:
        async with self.sessions() as session:
            row = await session.get(ProviderReconciliationModel, request_id)
            if row is None:
                return None
            if row.request_hash != request_hash:
                raise ValueError("provider reconciliation identity conflict")
            return dict(row.payload)

    async def reconcile_verified(
        self,
        *,
        request_id: str,
        request_hash: str,
        approval_id: str,
        disposition: Literal["completed", "not_executed"],
        actual_usd: Decimal,
        response_hash: str | None,
        reconciliation_hash: str,
        payload: dict[str, Any],
    ) -> None:
        """Called only after the service verifies immutable external evidence.

        The decision and monetary settlement commit atomically. No job is reset,
        no dispatch marker is deleted, and the original journal is not rewritten.
        """
        if not actual_usd.is_finite() or actual_usd < 0:
            raise ValueError("invalid reconciled provider cost")
        if disposition == "not_executed" and (actual_usd != 0 or response_hash is not None):
            raise ValueError("nonexecution proof cannot include spend or a response")
        if disposition == "completed" and response_hash is None:
            raise ValueError("completed reconciliation requires a verified response")
        expected = {
            "request_id": request_id,
            "request_hash": request_hash,
            "approval_id": approval_id,
            "disposition": disposition,
            "response_sha256": response_hash,
            "reconciliation_hash": reconciliation_hash,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("provider reconciliation audit identity mismatch")
        if Decimal(str(payload.get("actual_usd"))) != actual_usd:
            raise ValueError("provider reconciliation audit cost mismatch")
        unsigned = {
            key: value
            for key, value in payload.items()
            if key not in {"reconciliation_hash", "receipt_sha256"}
        }
        if canonical_content_hash(unsigned) != reconciliation_hash:
            raise ValueError("provider reconciliation audit checksum mismatch")
        async with self.sessions.begin() as session:
            approval = await session.get(
                BenchmarkBudgetApprovalModel, approval_id, with_for_update=True
            )
            row = await session.get(
                BenchmarkSpendReservationModel, request_id, with_for_update=True
            )
            if (
                approval is None
                or row is None
                or row.approval_id != approval_id
                or row.request_hash != request_hash
            ):
                raise ValueError("provider reconciliation reservation identity mismatch")
            old = await session.get(ProviderReconciliationModel, request_id)
            if old is not None:
                if old.reconciliation_hash != reconciliation_hash or old.payload != payload:
                    raise ValueError("provider reconciliation is immutable")
                return
            if row.state not in {"reserved", "unknown"} or actual_usd > row.reserved_usd:
                raise ValueError(
                    "provider reconciliation cannot rewrite settlement or exceed reservation"
                )
            approval.reserved_usd -= row.reserved_usd
            approval.settled_usd += actual_usd
            row.actual_usd, row.response_hash = actual_usd, response_hash
            row.state = "settled" if disposition == "completed" else "cancelled"
            # The insert trigger validates the settled reservation in this transaction.
            await session.flush()
            session.add(
                ProviderReconciliationModel(
                    request_id=request_id,
                    request_hash=request_hash,
                    reconciliation_hash=reconciliation_hash,
                    disposition=disposition,
                    payload=payload,
                    created_at=(
                        await session.execute(text("SELECT clock_timestamp()"))
                    ).scalar_one(),
                )
            )
