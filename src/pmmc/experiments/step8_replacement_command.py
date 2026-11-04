"""CPU-only explicit replacement entry point; never creates a budget approval."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from pmmc.experiments.step8_artifacts import CampaignArtifactStore, sha256_file, write_json
from pmmc.experiments.step8_backend import ProductionComponentBackend
from pmmc.experiments.step8_config_contract import verify_frozen_execution_inputs
from pmmc.experiments.step8_conformance import build_step8_conformance_plan, run_step8_conformance
from pmmc.experiments.step8_conformance_replacement import (
    NATIVE_ALIASES,
    ReplacementError,
    authorize_replacement,
    read_ledger,
)
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.settings import Settings
from pmmc.models.registry import load_model_registry
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    SelectionExpansionManifest,
    Step8BenchmarkConfig,
)


@contextmanager
def replacement_heartbeat():
    started, stop = time.monotonic(), threading.Event()

    def emit(state):
        print(json.dumps({
            "event": "STEP8_EXPLICIT_REPLACEMENT", "state": state,
            "elapsed_seconds": int(time.monotonic() - started),
            "gpu_operations": 0, "scientific_assignments": 0,
        }, sort_keys=True), flush=True)

    def heartbeat():
        while not stop.wait(30):
            emit("running")

    emit("start")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


async def replace_probe(
    *,
    settings: Settings,
    campaign_dir: Path,
    manifest: BenchmarkCampaignManifest,
    config: Step8BenchmarkConfig,
    selection: SelectionExpansionManifest,
    rate_card: Path,
    approval_id: str,
    original_request_id: str,
    original_request_hash: str,
    original_reserved_usd: Decimal,
    budget_usd: Decimal,
    approver_id: str,
    execute: bool,
) -> dict:
    root = settings.resolve_path(config.output_root)
    if (
        root != campaign_dir.resolve().parent
        or manifest.paper_result
        or selection.campaign_id != manifest.campaign_id
    ):
        raise ReplacementError("replacement_development_campaign_required")
    store = CampaignArtifactStore(root)
    settings = settings.model_copy(
        update={
            "step8_budget_approval_id": approval_id,
            "step8_rate_card_path": rate_card.resolve(),
            "step8_output_root": root,
        }
    )
    backend = ProductionComponentBackend(
        settings=settings, config=config, campaign=manifest, campaign_dir=campaign_dir, store=store
    )
    verify_frozen_execution_inputs(backend.runtime_settings, config, manifest.protocol)
    registry = load_model_registry(backend.settings.resolved_model_registry_path)
    plan = build_step8_conformance_plan(
        settings=backend.settings,
        registry=registry,
        protocol_hash=manifest.protocol.content_hash,
        campaign_id=manifest.campaign_id,
        selection_hash=selection.content_hash,
        probe_aliases=NATIVE_ALIASES,
    )
    engine, sessions = create_session_factory(backend.settings, DatabaseRole.RUNTIME)
    try:
        approval, rows = await read_ledger(sessions, approval_id)
        if sha256_file(rate_card) != approval["rate_card_hash"]:
            raise ReplacementError("replacement_rate_card_mismatch")
        # Probe the actual journal path BEFORE consent/dispatch. No DB writes here.
        storage = store.preflight_response_journal()
        record = authorize_replacement(
            store=store,
            plan=plan,
            registry=registry,
            approval=approval,
            rows=rows,
            original_request_id=original_request_id,
            original_request_hash=original_request_hash,
            original_reserved_usd=original_reserved_usd,
            budget_usd=budget_usd,
            approver_id=approver_id,
        )
        write_json(
            campaign_dir / "manifests" / ("conformance-replacement-" + approval_id + ".json"),
            record,
            immutable=True,
        )
        print(
            json.dumps(
                {
                    "event": "STEP8_EXPLICIT_REPLACEMENT",
                    "state": "authorized",
                    "authorization_hash": record["content_hash"],
                    "execute": execute,
                    "original_reserved_usd": record["original"]["reserved_usd"],
                    "budget_usd": str(budget_usd),
                    "gpu_operations": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        report = {
            "protocol": record["protocol"],
            "state": "authorized_not_executed",
            "authorization_hash": record["content_hash"],
            "campaign_id": manifest.campaign_id,
            "original_request_id": original_request_id,
            "original_actual_usd": None,
            "original_reserved_usd": record["original"]["reserved_usd"],
            "budget_usd": str(budget_usd),
            "gpu_operations": 0,
            "scientific_assignments": 0,
            "billing_complete": False,
            "journal_storage": storage,
        }
        if execute:
            receipt = await run_step8_conformance(
                settings=backend.settings,
                registry=registry,
                protocol_hash=manifest.protocol.content_hash,
                campaign_id=manifest.campaign_id,
                selection_hash=selection.content_hash,
                output_root=root,
                probe_aliases=NATIVE_ALIASES,
            )
            write_json(
                campaign_dir / "private/conformance" / (receipt.run_id + ".json"),
                receipt.model_dump(mode="json"),
                immutable=True,
            )
            after, _ = await read_ledger(sessions, approval_id)
            report.update(
                {
                    "state": "replacement_conformance_passed",
                    "receipt_hash": receipt.content_hash,
                    "conformance_run_id": receipt.run_id,
                    "probes": len(receipt.probes),
                    "settled_usd": str(after["settled_usd"]),
                    "reserved_usd": str(after["reserved_usd"]),
                }
            )
        return report
    finally:
        await engine.dispose()
