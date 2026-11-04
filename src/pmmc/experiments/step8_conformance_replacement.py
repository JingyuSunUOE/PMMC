"""Explicit, single replacement of a lost synthetic response; not reconciliation.

The original reservation remains unknown and held. Authorization pins one new
native plan under the same approval. Neither consent nor retries enlarge budget.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, sha256_file, write_json
from pmmc.infrastructure.database.step8_models import (
    BenchmarkBudgetApprovalModel,
    BenchmarkSpendReservationModel,
)

if TYPE_CHECKING:
    from pmmc.experiments.step8_conformance import ConformancePlan
    from pmmc.models.registry import ModelRegistry

NATIVE_ALIASES = ("answer-gpt-6-luna", "answer-claude-haiku-5.5")
VERSION = "step8-explicit-conformance-replacement-v1"


class ReplacementError(ValueError):
    """Only constant, content-free codes may be supplied."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ReplacementError(code)


async def read_ledger(sessions: Any, approval_id: str) -> tuple[dict, list[dict]]:
    async with sessions() as session, session.begin():
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        approval = (
            (
                await session.execute(
                    select(BenchmarkBudgetApprovalModel.__table__).where(
                        BenchmarkBudgetApprovalModel.approval_id == approval_id
                    )
                )
            )
            .mappings()
            .one()
        )
        rows = (
            (
                await session.execute(
                    select(BenchmarkSpendReservationModel.__table__).where(
                        BenchmarkSpendReservationModel.approval_id == approval_id
                    )
                )
            )
            .mappings()
            .all()
        )
    return dict(approval), [dict(row) for row in rows]


def authorization_path(store: CampaignArtifactStore, approval_id: str) -> Path:
    _require(bool(re.fullmatch(r"[a-f0-9]{64}", approval_id)), "replacement_approval_id_invalid")
    return store.root / "_store/conformance-replacements" / (approval_id + ".json")


def _plan_identity(plan: ConformancePlan, approval: dict) -> dict:
    return {
        "campaign_id": plan.campaign_id,
        "selection_hash": plan.selection_hash,
        "approval_id": approval["approval_id"],
        "rate_card_hash": approval["rate_card_hash"],
        "limit_usd": format(Decimal(approval["limit_usd"]).normalize(), "f"),
        "plan_hash": plan.plan_hash,
        "run_id": plan.run_id,
        "requests": {p.request.model_alias: p.request.request_id for p in plan.probes},
        "guard_sha256": sha256_file(Path(__file__)),
    }


def _scope(plan: ConformancePlan, approval: dict) -> None:
    _require(
        approval["campaign_id"] == plan.campaign_id
        and approval["selection_hash"] == plan.selection_hash,
        "replacement_approval_scope_mismatch",
    )


def _request_body(request: dict) -> dict:
    # Only run/request identity may change, never prompt, schema, images or model.
    value = {**request, "metadata": dict(request.get("metadata", {}))}
    value.pop("request_id", None)
    value["metadata"].pop("conformance_run_id", None)
    return value


def _original(
    store: CampaignArtifactStore,
    plan: ConformancePlan,
    registry: ModelRegistry,
    approval: dict,
    rows: list[dict],
    request_id: str,
    request_hash: str,
    held_usd: Decimal,
) -> tuple[str, str]:
    _require(bool(re.fullmatch(r"[a-f0-9]{64}", request_hash)), "replacement_request_hash_invalid")
    matches = [row for row in rows if row["request_id"] == request_id]
    _require(len(matches) == 1, "replacement_original_reservation_missing")
    row = matches[0]
    _require(
        row["request_hash"] == request_hash
        and row["state"] == "unknown"
        and row["actual_usd"] is None
        and row["response_hash"] is None
        and Decimal(row["reserved_usd"]) == held_usd > 0
        and Decimal(approval["reserved_usd"]) >= held_usd,
        "replacement_original_must_remain_unknown_and_held",
    )
    _require(
        store.restored_response(request_id=request_id, request_hash=request_hash) is None,
        "replacement_original_response_now_available",
    )
    intent_path = store.root / "_store/provider-dispatch" / request_hash / "intent.json"
    intent = json.loads(intent_path.read_bytes())
    identity = {k: v for k, v in intent.items() if k != "attempt_id"}
    _require(canonical_content_hash(identity) == request_hash, "replacement_intent_corrupt")
    old = identity["request"]
    metadata = old["metadata"]
    alias = old["model_alias"]
    _require(alias in NATIVE_ALIASES, "replacement_native_probe_required")
    entry = registry.get(alias)
    probe = next(p for p in plan.probes if p.request.model_alias == alias)
    _require(
        identity.get("protocol") == "step8-provider-accounting-v1"
        and request_id.startswith("step8-conformance-request-")
        and old["request_id"] == request_id
        and request_id != probe.request.request_id
        and identity["approval"]
        == {
            "campaign_id": plan.campaign_id,
            "selection_hash": plan.selection_hash,
            "approval_id": approval["approval_id"],
            "rate_card_hash": approval["rate_card_hash"],
        }
        and identity["registry_entry_hash"] == entry.resolved_hash
        and entry.provider_config is not None
        and row["provider"] == entry.provider_config.kind
        and Decimal(identity["worst_case_usd"]) == held_usd
        and metadata.get("conformance_probe_scope") == "synthetic_transport"
        and metadata.get("scientific_assignments") == 0
        and metadata.get("paper_result") is False
        and metadata.get("conformance_run_id") == intent.get("attempt_id")
        and _request_body(old) == _request_body(probe.request.model_dump(mode="json")),
        "replacement_original_not_identical_synthetic_probe",
    )
    return alias, sha256_file(intent_path)


def validate_authorization(
    authorization: dict,
    *,
    store: CampaignArtifactStore,
    plan: ConformancePlan,
    registry: ModelRegistry,
    approval: dict,
    rows: list[dict],
) -> None:
    _scope(plan, approval)
    _require(
        authorization.get("protocol") == VERSION
        and authorization.get("max_replacements") == 1
        and authorization.get("scientific_assignments") == 0
        and authorization.get("budget_increased") is False
        and authorization.get("identity") == _plan_identity(plan, approval)
        and authorization.get("content_hash")
        == canonical_content_hash({k: v for k, v in authorization.items() if k != "content_hash"}),
        "replacement_authorization_stale_or_corrupt",
    )
    old = authorization["original"]
    alias, intent_hash = _original(
        store,
        plan,
        registry,
        approval,
        rows,
        old["request_id"],
        old["request_hash"],
        Decimal(old["reserved_usd"]),
    )
    _require(
        old["intent_sha256"] == intent_hash
        and authorization["replacement_request_id"]
        == _plan_identity(plan, approval)["requests"][alias],
        "replacement_link_invalid",
    )
    permitted = {old["request_id"], *(p.request.request_id for p in plan.probes)}
    _require(
        all(
            not row["request_id"].startswith("step8-conformance-request-")
            or row["request_id"] in permitted
            for row in rows
        ),
        "replacement_unexpected_paid_history",
    )


def authorize_replacement(
    *,
    store: CampaignArtifactStore,
    plan: ConformancePlan,
    registry: ModelRegistry,
    approval: dict,
    rows: list[dict],
    original_request_id: str,
    original_request_hash: str,
    original_reserved_usd: Decimal,
    budget_usd: Decimal,
    approver_id: str,
) -> dict:
    _scope(plan, approval)
    _require(
        plan.probe_aliases == NATIVE_ALIASES and plan.roles == ("answer",),
        "replacement_requires_exact_native_probe_pair",
    )
    _require(Decimal(approval["limit_usd"]) == budget_usd > 0, "replacement_budget_limit_mismatch")
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", approver_id)), "replacement_approver_invalid"
    )
    alias, intent_hash = _original(
        store,
        plan,
        registry,
        approval,
        rows,
        original_request_id,
        original_request_hash,
        original_reserved_usd,
    )
    identity = _plan_identity(plan, approval)
    record = {
        "protocol": VERSION,
        "identity": identity,
        "approver_id": approver_id,
        "original": {
            "request_id": original_request_id,
            "request_hash": original_request_hash,
            "reserved_usd": str(original_reserved_usd.normalize()),
            "actual_usd": None,
            "state": "unknown",
            "intent_sha256": intent_hash,
        },
        "replacement_request_id": identity["requests"][alias],
        "max_replacements": 1,
        "scientific_assignments": 0,
        "budget_increased": False,
    }
    record["content_hash"] = canonical_content_hash(record)
    validate_authorization(
        record, store=store, plan=plan, registry=registry, approval=approval, rows=rows
    )
    write_json(authorization_path(store, approval["approval_id"]), record, immutable=True)
    return record


async def guard_paid_plan(
    *,
    sessions: Any,
    store: CampaignArtifactStore,
    plan: ConformancePlan,
    registry: ModelRegistry,
    approval_id: str,
) -> dict | None:
    """No changing paid probe identity implicitly, including across image revisions."""
    approval, rows = await read_ledger(sessions, approval_id)
    _scope(plan, approval)
    path = authorization_path(store, approval_id)
    if path.exists():
        record = json.loads(path.read_bytes())
        validate_authorization(
            record, store=store, plan=plan, registry=registry, approval=approval, rows=rows
        )
        return record
    current = {p.request.request_id for p in plan.probes}
    _require(
        all(
            not row["request_id"].startswith("step8-conformance-request-")
            or row["request_id"] in current
            for row in rows
        ),
        "conformance_prior_plan_requires_explicit_replacement",
    )
    # Atomic immutable pin also prevents concurrent, initially empty-ledger plans.
    write_json(
        store.root / "_store/conformance-plans" / (approval_id + ".json"),
        _plan_identity(plan, approval),
        immutable=True,
    )
    return None
