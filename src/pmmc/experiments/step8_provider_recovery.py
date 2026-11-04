"""Read-only ledger/journal diagnosis; never settle or redispatch unknown calls."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select, text

from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.experiments.step8_failure import failure_diagnostics
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.step8_models import (
    BenchmarkBudgetApprovalModel,
    BenchmarkSpendReservationModel,
)
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.step8 import BenchmarkCampaignManifest


def _safe_identifier(value: Any) -> str | None:
    return (
        value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
        else None
    )


def inspect_reservation_journal(
    store: CampaignArtifactStore, row: dict[str, Any],
) -> dict[str, Any]:
    """Report presence and verified metadata only, never a prompt/response body."""
    request_id, request_hash = row["request_id"], row["request_hash"]
    report = {
        key: row[key] for key in (
            "request_id", "request_hash", "approval_id", "provider", "state",
            "reserved_usd", "actual_usd", "response_hash",
        )
    }
    report["created_at"] = str(row["created_at"])
    report["reserved_usd"] = str(row["reserved_usd"])
    report["actual_usd"] = None if row["actual_usd"] is None else str(row["actual_usd"])
    if not re.fullmatch(r"[a-f0-9]{64}", request_hash):
        raise ValueError("invalid reservation request digest")
    dispatch = store.root / "_store/provider-dispatch" / request_hash
    report["dispatch_claim_present"] = dispatch.exists()
    report["dispatch_intent_present"] = (dispatch / "intent.json").is_file()
    report["journal_state"] = "missing"
    try:
        envelope = store.restored_response(request_id=request_id, request_hash=request_hash)
        if envelope is not None:
            if (
                envelope.get("request_id") != request_id
                or envelope.get("request_hash") != request_hash
                or not isinstance(envelope.get("approval"), dict)
                or envelope["approval"].get("approval_id") != row["approval_id"]
            ):
                raise ValueError("accounting journal identity mismatch")
            report["journal_state"] = "checksum_and_reservation_identity_verified"
            report["result_present"] = isinstance(envelope.get("result"), dict)
            error = envelope.get("error")
            if isinstance(error, dict):
                report["provider_error"] = {
                    key: _safe_identifier(error.get(key))
                    for key in ("error_code", "outcome", "provider_request_id")
                }
                status = error.get("provider_status_code")
                report["provider_error"]["provider_status_code"] = (
                    status if type(status) is int and 100 <= status <= 599 else None
                )
        if row["response_hash"] is not None:
            store.verify(row["response_hash"])
            report["ledger_response_checksum_verified"] = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["journal_state"] = "invalid_or_unavailable"
        report["diagnostics"] = failure_diagnostics(exc)
    report["requires_review"] = (
        row["state"] in {"reserved", "unknown"}
        or report["journal_state"] == "invalid_or_unavailable"
        or (row["state"] == "settled" and row["response_hash"] is None)
    )
    return report


async def inspect_provider_recovery(
    settings: Settings, campaign_dir: Path,
) -> dict[str, Any]:
    manifest = BenchmarkCampaignManifest.model_validate_json(
        (campaign_dir / "manifests/campaign.json").read_bytes()
    )
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session, session.begin():
            # Enforced by PostgreSQL as well as by the SELECT-only implementation.
            await session.execute(text(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            ))
            approvals = (await session.execute(select(
                BenchmarkBudgetApprovalModel.approval_id,
                BenchmarkBudgetApprovalModel.selection_hash,
                BenchmarkBudgetApprovalModel.rate_card_hash,
                BenchmarkBudgetApprovalModel.limit_usd,
                BenchmarkBudgetApprovalModel.settled_usd,
                BenchmarkBudgetApprovalModel.reserved_usd,
            ).where(
                BenchmarkBudgetApprovalModel.campaign_id == manifest.campaign_id
            ).order_by(BenchmarkBudgetApprovalModel.approval_id))).mappings().all()
            reservations = (await session.execute(select(
                BenchmarkSpendReservationModel.__table__
            ).join(BenchmarkBudgetApprovalModel, (
                BenchmarkSpendReservationModel.approval_id
                == BenchmarkBudgetApprovalModel.approval_id
            )).where(
                BenchmarkBudgetApprovalModel.campaign_id == manifest.campaign_id
            ).order_by(BenchmarkSpendReservationModel.created_at,
                       BenchmarkSpendReservationModel.request_id))).mappings().all()
    finally:
        await engine.dispose()
    store = CampaignArtifactStore(campaign_dir.parent)
    rows = [inspect_reservation_journal(store, dict(row)) for row in reservations]
    return {
        "protocol": "step8-provider-recovery-inspection-v1", "state": "inspected",
        "campaign_id": manifest.campaign_id, "database_read_only": True,
        "provider_calls": 0, "gpu_operations": 0, "budget_changed": False,
        "resume_authorized": False,
        "approvals": [{key: str(value) for key, value in row.items()} for row in approvals],
        "reservation_counts": dict(Counter(row["state"] for row in rows)),
        "reservations": rows,
        "requires_review": not approvals or any(row["requires_review"] for row in rows),
        "scope": "ledger_and_local_journals_not_provider_billing_confirmation",
    }
