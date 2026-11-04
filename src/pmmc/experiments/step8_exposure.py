"""Append-only exposure history shared by every campaign in one output root."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import write_json
from pmmc.schemas.base import DatasetName
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    CampaignPhase,
    SelectionExpansionManifest,
)


@contextmanager
def _ledger_lock(root: Path):
    directory = root / "_store/exposure"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("exposure_ledger_symlink_forbidden")
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield directory


def _check(
    directory: Path, manifest: BenchmarkCampaignManifest, selection: SelectionExpansionManifest
) -> None:
    if selection.campaign_id != manifest.campaign_id:
        raise ValueError("exposure_selection_campaign_mismatch")
    if selection.partition_hash != manifest.protocol.partition_hash:
        raise ValueError("exposure_selection_partition_mismatch")
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink():
            raise ValueError("exposure_event_symlink_forbidden")
        event = json.loads(path.read_bytes())
        if (
            event.get("protocol") != "step8-exposure-event-v1"
            or canonical_content_hash(event) != path.stem
        ):
            raise ValueError("exposure_event_integrity_failed")
        if manifest.phase is not CampaignPhase.PAPER:
            same_selection = (
                event["campaign_id"] == manifest.campaign_id
                and event["protocol_hash"] == manifest.protocol.content_hash
                and selection.content_hash in event["selection_hashes"]
            )
            if (
                selection.selection == "confirmation"
                and not same_selection
                and any(
                    set(ids) & set(event["query_ids"].get(dataset.value, []))
                    for dataset, ids in selection.query_ids.items()
                )
            ):
                raise ValueError("confirmation_queries_exposed_after_selection")
            continue
        # A prior paper run of the same frozen method is legitimate nested expansion.
        compatible = (
            event["phase"] == "paper"
            and event["protocol_hash"] == manifest.protocol.content_hash
            and event["partition_hash"] == selection.partition_hash
        )
        if compatible:
            continue
        if any(
            set(ids) & set(event["query_ids"].get(dataset.value, []))
            for dataset, ids in selection.query_ids.items()
        ):
            raise ValueError("paper_queries_exposed_under_another_protocol")


def assert_compatible_exposure_history(
    output_root: Path,
    *,
    manifest: BenchmarkCampaignManifest,
    selection: SelectionExpansionManifest,
) -> None:
    with _ledger_lock(output_root) as directory:
        _check(directory, manifest, selection)


def recorded_exposure_query_ids(output_root: Path) -> dict[DatasetName, list[str]]:
    """Exclude prior production exposure from a new confirmation selection."""
    datasets = (DatasetName.MEMLENS, DatasetName.MEMGALLERY)
    result: dict[DatasetName, set[str]] = {dataset: set() for dataset in datasets}
    with _ledger_lock(output_root) as directory:
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                raise ValueError("exposure_event_symlink_forbidden")
            event = json.loads(path.read_bytes())
            if (
                event.get("protocol") != "step8-exposure-event-v1"
                or canonical_content_hash(event) != path.stem
                or set(event.get("query_ids", {})) != {d.value for d in datasets}
            ):
                raise ValueError("exposure_event_integrity_failed")
            for dataset in datasets:
                ids = event["query_ids"][dataset.value]
                if not isinstance(ids, list) or any(not isinstance(q, str) or not q for q in ids):
                    raise ValueError("exposure_query_identity_invalid")
                result[dataset].update(ids)
    return {dataset: sorted(ids) for dataset, ids in result.items()}


def record_campaign_exposure(output_root: Path, campaign_dir: Path) -> str:
    """Conservatively mark all assigned queries before any production component runs."""
    manifest = BenchmarkCampaignManifest.model_validate_json(
        (campaign_dir / "manifests/campaign.json").read_bytes()
    )
    selections = [
        SelectionExpansionManifest.model_validate_json(path.read_bytes())
        for path in sorted((campaign_dir / "manifests/selections").glob("*.json"))
    ]
    if not selections:
        raise ValueError("exposure_requires_frozen_selection")
    with _ledger_lock(output_root) as directory:
        for selection in selections:
            _check(directory, manifest, selection)
        query_ids = {
            dataset.value: sorted({q for s in selections for q in s.query_ids.get(dataset, ())})
            for dataset in (DatasetName.MEMLENS, DatasetName.MEMGALLERY)
        }
        event = {
            "protocol": "step8-exposure-event-v1",
            "campaign_id": manifest.campaign_id,
            "protocol_hash": manifest.protocol.content_hash,
            "partition_hash": manifest.protocol.partition_hash,
            "phase": manifest.phase.value,
            "purpose": "production_dispatch",
            "selection_hashes": sorted(s.content_hash for s in selections),
            "query_ids": query_ids,
        }
        digest = canonical_content_hash(event)
        write_json(directory / f"{digest}.json", event, immutable=True)
        write_json(campaign_dir / f"private/exposure/{digest}.json", event, immutable=True)
    return digest
