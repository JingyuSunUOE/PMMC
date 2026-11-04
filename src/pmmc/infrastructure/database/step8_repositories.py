"""Short-transaction campaign jobs, fenced commits, and monetary reservations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from re import fullmatch
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt, validate_job_dag
from pmmc.infrastructure.database.step8_models import (
    BenchmarkBudgetApprovalModel,
    BenchmarkCampaignModel,
    BenchmarkCellAssignmentModel,
    BenchmarkComponentReceiptModel,
    BenchmarkExecutionAttemptModel,
    BenchmarkSelectionModel,
    BenchmarkSpendReservationModel,
    BenchmarkStageJobModel,
    ExperimentCellModel,
    MethodProtocolFreezeModel,
)
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    ExperimentCell,
    MethodProtocolFreeze,
    ResourceClass,
    SelectionExpansionManifest,
)


class LeaseLostError(RuntimeError):
    """The committing attempt no longer owns the authoritative job lease."""


class BudgetExceededError(RuntimeError):
    """Actual spend plus worst-case in-flight reservations exceeds approval."""


def _stop_once(approval: BenchmarkBudgetApprovalModel, actor: str, reason: str) -> None:
    if approval.stopped_at is None:
        approval.stopped_at = datetime.now(UTC)
        approval.stopped_by = actor
        approval.stop_reason = reason


def _digest(value: Any) -> bool:
    return isinstance(value, str) and fullmatch(r"[a-f0-9]{64}", value) is not None


def validate_budget_policy(limit: Decimal | None, uncapped: bool) -> None:
    if uncapped != (limit is None):
        raise ValueError("choose exactly one of --limit-usd or --uncapped")
    if limit is not None and (
        not limit.is_finite() or not 0 < limit < 10**12
        or limit != limit.quantize(Decimal("0.00000001"))
    ):
        raise ValueError("explicit positive finite NUMERIC(20,8) limit required")


def require_budget_binding(
    row: Any, *, campaign_id: str, selection_hash: str,
    plan_hash: str | None, scope_hash: str | None,
) -> None:
    """Verify an existing capability, never infer a child from its parent."""
    if plan_hash is None and scope_hash is None:
        if (getattr(row, "plan_hash", None) is None and row.limit_usd is not None
                and (row.campaign_id, row.selection_hash) == (campaign_id, selection_hash)):
            return
    elif (_digest(plan_hash) and _digest(scope_hash) and row.plan_hash == plan_hash
          and any((member["campaign_id"], member["selection_hash"]) == (campaign_id, selection_hash)
                  and scope_hash in member["scopes"].values()
                  for member in row.plan_bindings.values())):
        return
    raise ValueError("release_budget_plan_or_scope_binding_mismatch")


async def _insert_exact(
    session: AsyncSession,
    model: Any,
    key: str,
    values: dict[str, Any],
    compare: tuple[str, ...] = ("payload",),
) -> Any:
    await session.execute(insert(model).values(**values).on_conflict_do_nothing())
    row = await session.get(model, values[key])
    if row is None or any(getattr(row, field) != values[field] for field in compare):
        raise ValueError("immutable campaign fact conflicts with existing identity")
    return row


class PostgresBenchmarkRepository:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], stages: tuple[str, ...] = (),
        *, job_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.sessions = sessions
        self.stages = stages
        if job_ids is not None and (
            any(not isinstance(key, str) or not key for key in job_ids)
            or len(set(job_ids)) != len(job_ids)
        ):
            raise ValueError("invalid_exact_job_filter")
        self.job_ids = job_ids

    async def start_attempt(
        self,
        *,
        attempt_id: str,
        campaign_id: str,
        execution_image: str,
        resource_class: str,
        started_at: datetime,
    ) -> None:
        async with self.sessions.begin() as session:
            await _insert_exact(
                session,
                BenchmarkExecutionAttemptModel,
                "attempt_id",
                {
                    "attempt_id": attempt_id,
                    "campaign_id": campaign_id,
                    "execution_image": execution_image,
                    "resource_class": resource_class,
                    "status": "running",
                    "started_at": started_at,
                },
                compare=("campaign_id", "execution_image", "resource_class", "started_at"),
            )

    async def finish_attempt(self, attempt_id: str, state: str, receipt_hash: str) -> None:
        if state not in {"completed", "failed", "interrupted", "budget_exhausted"}:
            raise ValueError("invalid_campaign_attempt_terminal")
        async with self.sessions.begin() as session:
            row = await session.get(
                BenchmarkExecutionAttemptModel, attempt_id, with_for_update=True
            )
            if row is None:
                raise ValueError("campaign_attempt_missing")
            if row.status != "running":
                if row.status != state or row.receipt_hash != receipt_hash:
                    raise ValueError("campaign_attempt_terminal_conflict")
                return
            row.status, row.receipt_hash = state, receipt_hash
            row.completed_at = datetime.now(UTC)

    async def register(
        self,
        campaign: BenchmarkCampaignManifest,
        selection: SelectionExpansionManifest,
        cells: list[ExperimentCell],
        freeze: MethodProtocolFreeze | None = None,
    ) -> None:
        now = datetime.now(UTC)
        payload = campaign.model_dump(mode="json")
        selection_hash = selection.content_hash
        async with self.sessions.begin() as session:
            if freeze is not None:
                await _insert_exact(
                    session,
                    MethodProtocolFreezeModel,
                    "freeze_hash",
                    {
                        "freeze_hash": freeze.content_hash,
                        "protocol_hash": freeze.protocol.content_hash,
                        "decision": freeze.decision,
                        "reviewer_id": freeze.reviewer_id,
                        "payload": freeze.model_dump(mode="json"),
                        "created_at": freeze.approved_at,
                    },
                )
            await _insert_exact(
                session,
                BenchmarkCampaignModel,
                "campaign_id",
                {
                    "campaign_id": campaign.campaign_id,
                    "protocol_hash": campaign.protocol.content_hash,
                    "freeze_hash": campaign.protocol_freeze_hash,
                    "phase": campaign.phase.value,
                    "status": "planned",
                    "manifest_hash": canonical_content_hash(payload),
                    "payload": payload,
                    "created_at": campaign.created_at,
                },
            )
            if selection.campaign_id != campaign.campaign_id:
                raise ValueError("selection belongs to another campaign")
            await _insert_exact(
                session,
                BenchmarkSelectionModel,
                "selection_hash",
                {
                    "selection_hash": selection.content_hash,
                    "campaign_id": campaign.campaign_id,
                    "selection": selection.selection,
                    "parent_selection_hash": selection.parent_selection_hash,
                    "payload": selection.model_dump(mode="json"),
                    "created_at": now,
                },
            )
            for cell in cells:
                if cell.protocol_hash != campaign.protocol.content_hash:
                    raise ValueError("cell belongs to another scientific protocol")
                await _insert_exact(
                    session,
                    ExperimentCellModel,
                    "cell_id",
                    {
                        "cell_id": cell.cell_id,
                        "protocol_hash": cell.protocol_hash,
                        "dataset": cell.dataset.value,
                        "method": cell.method,
                        "answer_model": cell.answer_model,
                        "payload": cell.model_dump(mode="json"),
                    },
                )
                assignments = []
                for query_id in selection.query_ids[cell.dataset]:
                    identity = {
                        "campaign_id": campaign.campaign_id,
                        "cell_id": cell.cell_id,
                        "query_id": query_id,
                    }
                    assignments.append(dict(assignment_id=canonical_content_hash(identity),
                                            **identity, first_selection_hash=selection_hash))
                for start in range(0, len(assignments), 500):
                    await session.execute(insert(BenchmarkCellAssignmentModel)
                                          .values(assignments[start:start + 500])
                                          .on_conflict_do_nothing())

    async def add_jobs(self, jobs: list[BenchmarkStageJob]) -> None:
        async with self.sessions.begin() as session:
            existing = set((await session.scalars(select(BenchmarkStageJobModel.job_id).where(
                BenchmarkStageJobModel.campaign_id.in_({j.campaign_id for j in jobs})
            ))).all())
            validate_job_dag(jobs, existing_ids=existing)
            # Batched registration never updates existing status, lease, or attempt data.
            for start in range(0, len(jobs), 250):
                batch = [{
                        "job_id": job.job_id,
                        "campaign_id": job.campaign_id,
                        "component_key": job.component_key,
                        "stage": job.stage,
                        "resource_class": job.resource_class.value,
                        "scientific_input_hash": job.scientific_input_hash,
                        "source_fingerprint": job.source_fingerprint,
                        "dependency_ids": job.dependency_ids,
                        "payload": job.model_dump(mode="json"),
                        "status": "pending",
                        "attempt_count": 0,
                        "created_at": datetime.now(UTC),
                    } for job in jobs[start:start + 250]]
                await session.execute(insert(BenchmarkStageJobModel).values(batch)
                                      .on_conflict_do_nothing())
                expected = {v["job_id"]: v["payload"] for v in batch}
                stored = (await session.execute(select(BenchmarkStageJobModel.job_id,
                    BenchmarkStageJobModel.payload).where(
                        BenchmarkStageJobModel.job_id.in_(expected)
                    ))).all()
                saved = {key: value for key, value in stored}
                if expected != saved:
                    raise ValueError("immutable campaign fact conflicts with existing identity")

    async def claim(
        self,
        campaign_id: str,
        resource: ResourceClass,
        owner: str,
        lease_seconds: int,
    ) -> tuple[BenchmarkStageJob, int] | None:
        if not owner or lease_seconds < 30:
            raise ValueError("bounded lease and unique attempt worker ID required")
        # SKIP LOCKED fences ownership without keeping a transaction open during work.
        async with self.sessions.begin() as session:
            result = await session.execute(
                text("""
                SELECT job_id FROM pmmc_runtime.benchmark_stage_jobs AS job
                WHERE campaign_id=:campaign AND resource_class=:resource
                  AND (:all_stages OR stage = ANY(:stages))
                  AND (:all_jobs OR job_id = ANY(:job_ids))
                  AND (status='pending' OR
                       (status='running' AND lease_expires_at < clock_timestamp()))
                  AND NOT EXISTS (
                    SELECT 1 FROM unnest(job.dependency_ids) AS dep(id)
                    LEFT JOIN pmmc_runtime.benchmark_stage_jobs AS upstream
                      ON upstream.job_id=dep.id AND upstream.campaign_id=job.campaign_id
                    WHERE upstream.status IS DISTINCT FROM 'completed')
                ORDER BY created_at, job_id FOR UPDATE SKIP LOCKED LIMIT 1
            """),
                {
                    "campaign": campaign_id,
                    "resource": resource.value,
                    "all_stages": not self.stages,
                    "stages": list(self.stages),
                    "all_jobs": self.job_ids is None,
                    "job_ids": list(self.job_ids or ()),
                },
            )
            job_id = result.scalar_one_or_none()
            if job_id is None:
                return None
            row = await session.get(BenchmarkStageJobModel, job_id)
            assert row is not None
            row.status, row.lease_owner = "running", owner
            row.attempt_count += 1
            row.lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
            return BenchmarkStageJob.model_validate(row.payload), row.attempt_count

    async def _owned(
        self,
        session: AsyncSession,
        job_id: str,
        owner: str,
        fence: int,
    ) -> BenchmarkStageJobModel:
        row = await session.get(BenchmarkStageJobModel, job_id, with_for_update=True)
        if (
            row is None
            or row.status != "running"
            or row.lease_owner != owner
            or row.attempt_count != fence
            or row.lease_expires_at is None
            or row.lease_expires_at <= datetime.now(UTC)
        ):
            raise LeaseLostError("stage_job_lease_lost")
        return row

    async def heartbeat(self, job_id: str, owner: str, fence: int, lease_seconds: int) -> None:
        async with self.sessions.begin() as session:
            row = await self._owned(session, job_id, owner, fence)
            row.lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)

    async def upstream_receipts(self, job: BenchmarkStageJob) -> dict[str, str]:
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(BenchmarkStageJobModel).where(
                        BenchmarkStageJobModel.job_id.in_(job.dependency_ids),
                        BenchmarkStageJobModel.campaign_id == job.campaign_id,
                    )
                )
            ).all()
            if len(rows) != len(job.dependency_ids) or any(
                row.status != "completed" or row.receipt_hash is None for row in rows
            ):
                raise ValueError("upstream component receipt is incomplete")
            return {row.job_id: str(row.receipt_hash) for row in rows}

    async def complete(self, receipt: ComponentReceipt, owner: str, fence: int) -> None:
        async with self.sessions.begin() as session:
            row = await self._owned(session, receipt.job_id, owner, fence)
            if (
                row.scientific_input_hash != receipt.scientific_input_hash
                or row.source_fingerprint != receipt.source_fingerprint
            ):
                raise ValueError("component receipt identity mismatch")
            upstream = (
                await session.scalars(
                    select(BenchmarkStageJobModel).where(
                        BenchmarkStageJobModel.job_id.in_(row.dependency_ids),
                    )
                )
            ).all()
            if (
                len(upstream) != len(row.dependency_ids)
                or {item.job_id: item.receipt_hash for item in upstream}
                != receipt.upstream_receipt_hashes
            ):
                raise ValueError("component upstream receipt mismatch")
            await _insert_exact(
                session,
                BenchmarkComponentReceiptModel,
                "receipt_hash",
                {
                    "receipt_hash": receipt.content_hash,
                    "component_identity": receipt.component_identity,
                    "job_id": receipt.job_id,
                    "payload": receipt.model_dump(mode="json"),
                    "artifact_hashes": receipt.artifact_hashes,
                    "created_at": datetime.now(UTC),
                },
            )
            row.status, row.receipt_hash = "completed", receipt.content_hash
            row.completed_at = datetime.now(UTC)
            row.lease_owner = row.lease_expires_at = None

    async def fail(self, job_id: str, owner: str, fence: int, code: str, *, unknown: bool) -> None:
        if not code.isidentifier() or len(code) > 128:
            code = "stage_execution_failed"
        async with self.sessions.begin() as session:
            row = await self._owned(session, job_id, owner, fence)
            row.status, row.error_code = ("unknown" if unknown else "failed"), code
            row.lease_owner = row.lease_expires_at = None

    async def counts(self, campaign_id: str) -> dict[str, int]:
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(
                        BenchmarkStageJobModel.status,
                        func.count(),
                    )
                    .where(BenchmarkStageJobModel.campaign_id == campaign_id)
                    .group_by(
                        BenchmarkStageJobModel.status,
                    )
                )
            ).all()
            return {str(state): int(count) for state, count in rows}

    async def list_jobs(self, campaign_id: str) -> list[dict[str, Any]]:
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(BenchmarkStageJobModel)
                    .where(
                        BenchmarkStageJobModel.campaign_id == campaign_id,
                    )
                    .order_by(BenchmarkStageJobModel.created_at, BenchmarkStageJobModel.job_id)
                )
            ).all()
            return [
                {
                    "job_id": row.job_id,
                    "job": row.payload,
                    "status": row.status,
                    "receipt_hash": row.receipt_hash,
                    "error_code": row.error_code,
                    "attempt_count": row.attempt_count,
                    "lease_expires_at": (
                        row.lease_expires_at.isoformat() if row.lease_expires_at else None
                    ),
                }
                for row in rows
            ]

    async def retry_failed(self, campaign_id: str) -> int:
        async with self.sessions.begin() as session:
            rows = (
                await session.scalars(
                    select(BenchmarkStageJobModel)
                    .where(
                        BenchmarkStageJobModel.campaign_id == campaign_id,
                        BenchmarkStageJobModel.status == "failed",
                    )
                    .with_for_update()
                )
            ).all()
            for row in rows:
                row.status, row.error_code = "pending", None
            # Unknown provider outcomes require reconciliation, never automatic resend.
            return len(rows)

    async def approve_budget(
        self,
        *,
        campaign_id: str,
        selection_hash: str,
        approver_id: str,
        rate_card_hash: str,
        limit_usd: Decimal | None,
        uncapped: bool = False,
        plan_hash: str | None = None,
        plan_bindings: dict | None = None,
    ) -> str:
        validate_budget_policy(limit_usd, uncapped)
        if not approver_id.strip() or (uncapped and plan_hash is None):
            raise ValueError("explicit approver and plan-bound uncapped policy required")
        identity: dict[str, Any] = {
            "campaign_id": campaign_id,
            "selection_hash": selection_hash,
            "approver_id": approver_id,
            "rate_card_hash": rate_card_hash,
            "limit_usd": str(limit_usd) if limit_usd is not None else None,
        }
        if plan_hash is not None or plan_bindings is not None:
            if not _digest(plan_hash) or not plan_bindings:
                raise ValueError("immutable plan and native bindings required")
            members = set()
            for member in plan_bindings.values():
                if (not member["campaign_id"] or not _digest(member["selection_hash"])
                        or not member["scopes"] or not all(
                            _digest(digest) for digest in member["scopes"].values())):
                    raise ValueError("invalid native budget scope")
                members.add((member["campaign_id"], member["selection_hash"]))
            if (campaign_id, selection_hash) not in members or len(members) != len(plan_bindings):
                raise ValueError("ambiguous plan budget bindings")
            identity.update(plan_hash=plan_hash, plan_bindings=plan_bindings)
            if limit_usd is not None:
                identity["limit_usd"] = format(limit_usd.normalize(), "f")
        approval_id = canonical_content_hash(identity)
        async with self.sessions.begin() as session:
            if plan_bindings:
                for member in plan_bindings.values():
                    child = await session.get(BenchmarkSelectionModel, member["selection_hash"])
                    if (child is None or child.campaign_id != member["campaign_id"]
                            or child.payload.get("release_plan_hash") != plan_hash):
                        raise ValueError("release_budget_selection_not_registered")
            approval = await _insert_exact(
                session,
                BenchmarkBudgetApprovalModel,
                "approval_id",
                {
                    **identity,
                    "approval_id": approval_id,
                    "limit_usd": limit_usd,
                    "settled_usd": Decimal(0),
                    "reserved_usd": Decimal(0),
                    "created_at": datetime.now(UTC),
                },
                compare=(
                    "campaign_id",
                    "selection_hash",
                    "approver_id",
                    "rate_card_hash",
                    "limit_usd",
                ) + (("plan_hash", "plan_bindings") if plan_hash else ()),
            )
            if plan_hash and approval.stopped_at is not None:
                raise ValueError("release_plan_approval_permanently_stopped")
        return approval_id

    async def stop_plan_budget(
        self, *, approval_id: str, plan_hash: str, campaign_id: str, selection_hash: str,
        operator_id: str, reason_code: str,
    ) -> dict[str, Any]:
        if (not _digest(approval_id) or not _digest(plan_hash) or not _digest(selection_hash)
                or not fullmatch(r"[A-Za-z0-9][A-Za-z0-9.@_-]{0,255}", operator_id)
                or not fullmatch(r"[a-z][a-z0-9_]{0,127}", reason_code)):
            raise ValueError("release_budget_stop_requires_plan_selection_operator_and_reason_code")
        async with self.sessions.begin() as session:
            row = await session.get(BenchmarkBudgetApprovalModel, approval_id, with_for_update=True)
            if (row is None or row.plan_hash != plan_hash or not any(
                (m["campaign_id"], m["selection_hash"]) == (campaign_id, selection_hash)
                for m in row.plan_bindings.values()
            )):
                raise ValueError("release_budget_stop_plan_or_selection_mismatch")
            _stop_once(row, operator_id, reason_code)
            return {
                "approval_id": row.approval_id, "plan_hash": row.plan_hash,
                "selection_hash": selection_hash, "campaign_id": campaign_id,
                "new_calls_stopped": True, "stopped_at": row.stopped_at.isoformat(),
                "stopped_by": row.stopped_by, "stop_reason": row.stop_reason,
                "settled_usd": str(row.settled_usd), "reserved_usd": str(row.reserved_usd),
            }

    async def reserve(
        self,
        approval_id: str,
        request_id: str,
        request_hash: str,
        provider: str,
        worst_case_usd: Decimal,
        *,
        binding: dict[str, str] | None = None,
    ) -> str:
        if (not worst_case_usd.is_finite() or not 0 <= worst_case_usd < 10**12
                or worst_case_usd != worst_case_usd.quantize(Decimal("0.00000001"))):
            raise ValueError("finite nonnegative worst-case cost required")
        async with self.sessions.begin() as session:
            approval = await session.get(
                BenchmarkBudgetApprovalModel, approval_id, with_for_update=True
            )
            if approval is None:
                raise BudgetExceededError("explicit_selection_budget_approval_required")
            if approval.plan_hash is not None or binding is not None:
                if not binding:
                    raise ValueError("release_budget_binding_required")
                require_budget_binding(approval, **binding)
            old = await session.get(BenchmarkSpendReservationModel, request_id)
            if old is not None:
                if (
                    old.request_hash != request_hash
                    or old.approval_id != approval_id
                    or old.provider != provider
                    or old.reserved_usd != worst_case_usd
                ):
                    raise ValueError("request reservation identity conflict")
                return old.state
            if approval.stopped_at is not None:
                raise BudgetExceededError("release_plan_approval_permanently_stopped")
            if (approval.limit_usd is not None
                    and approval.settled_usd + approval.reserved_usd + worst_case_usd
                    > approval.limit_usd):
                if approval.plan_hash is None:
                    raise BudgetExceededError("approved_api_budget_exhausted")
                # Commit the stop *before* raising. Other stages may still be settling
                # their admitted calls, but they cannot launch new ones on this plan.
                _stop_once(approval, "system", "approved_api_budget_exhausted")
            else:
                approval.reserved_usd += worst_case_usd
                session.add(
                    BenchmarkSpendReservationModel(
                        request_id=request_id,
                        approval_id=approval_id,
                        request_hash=request_hash,
                        provider=provider,
                        reserved_usd=worst_case_usd,
                        state="reserved",
                        created_at=datetime.now(UTC),
                    )
                )
                return "reserved"
        raise BudgetExceededError("approved_api_budget_exhausted")

    async def settle(
        self, request_id: str, actual_usd: Decimal | None, response_hash: str | None
    ) -> None:
        if actual_usd is not None and (not actual_usd.is_finite() or actual_usd < 0):
            raise ValueError("invalid actual API cost")
        async with self.sessions.begin() as session:
            reservation = await session.get(BenchmarkSpendReservationModel, request_id)
            if reservation is None:
                raise ValueError("unknown API reservation")
            # Lock order matches reserve: approval first, request second.
            approval = await session.get(
                BenchmarkBudgetApprovalModel, reservation.approval_id, with_for_update=True
            )
            await session.refresh(reservation, with_for_update=True)
            assert approval is not None
            # Verified nonexecution has already released this reservation. A
            # delayed worker must not reopen it or release the same funds twice.
            if reservation.state == "cancelled":
                raise ValueError("API settlement cannot rewrite cancelled reservation")
            if reservation.state == "settled":
                if (
                    reservation.actual_usd != actual_usd
                    or reservation.response_hash != response_hash
                ):
                    raise ValueError("API settlement cannot be rewritten")
                return
            if actual_usd is None:
                reservation.state = "unknown"
                reservation.response_hash = response_hash
                return  # Unknown usage retains its worst-case reservation.
            if actual_usd > reservation.reserved_usd:
                if approval.plan_hash is None:
                    raise BudgetExceededError("provider_exceeded_frozen_worst_case")
                # The price bound itself failed. Preserve the disputed response and
                # hold, and fence every child until the operator reconciles it.
                _stop_once(approval, "system", "provider_exceeded_frozen_worst_case")
                reservation.state, reservation.response_hash = "unknown", response_hash
            else:
                approval.reserved_usd -= reservation.reserved_usd
                approval.settled_usd += actual_usd
                reservation.actual_usd, reservation.response_hash = actual_usd, response_hash
                reservation.state = "settled"
                return
        raise BudgetExceededError("provider_exceeded_frozen_worst_case")
