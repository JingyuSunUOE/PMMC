"""Explicit canonical snapshot and benchmark asset synchronization."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pmmc.benchmarks.canonical import iter_jsonl, preflight_snapshot
from pmmc.infrastructure.object_store import (
    ObjectMetadata,
    ObjectStore,
    asset_object_key,
    resolve_local_asset,
    sha256_file,
    snapshot_object_key,
)
from pmmc.schemas import AssetRef

StorageProgress = Callable[[str, int, int], None]

RUNTIME_SNAPSHOT_FILES = (
    "manifest.json",
    "validation_report.json",
    "sessions.jsonl",
    "assets.jsonl",
    "memory_units.jsonl",
    "queries.jsonl",
)
EVALUATION_SNAPSHOT_FILES = ("query_gold.jsonl",)


@dataclass(frozen=True)
class SnapshotArchiveResult:
    snapshot_id: str
    runtime_objects: dict[str, ObjectMetadata]
    evaluation_objects: dict[str, ObjectMetadata]


@dataclass(frozen=True)
class AssetSyncRecord:
    asset_id: str
    uri: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AssetSyncResult:
    snapshot_id: str
    synced: list[AssetSyncRecord]
    skipped_remote_only: int


async def archive_snapshot(
    snapshot_path: Path,
    *,
    runtime_store: ObjectStore,
    evaluation_store: ObjectStore,
    progress: StorageProgress | None = None,
) -> SnapshotArchiveResult:
    """Archive runtime and gold files into separately authorized buckets."""

    report = await asyncio.to_thread(preflight_snapshot, snapshot_path)
    manifest = report.manifest
    runtime_objects: dict[str, ObjectMetadata] = {}
    evaluation_objects: dict[str, ObjectMetadata] = {}
    total = len(RUNTIME_SNAPSHOT_FILES) + len(EVALUATION_SNAPSHOT_FILES)
    current = 0
    for file_name in RUNTIME_SNAPSHOT_FILES:
        key = snapshot_object_key(manifest.dataset.value, manifest.snapshot_id, file_name)
        runtime_objects[file_name] = await runtime_store.put_file(snapshot_path / file_name, key)
        current += 1
        if progress is not None:
            progress("archive_snapshot", current, total)
    for file_name in EVALUATION_SNAPSHOT_FILES:
        key = snapshot_object_key(manifest.dataset.value, manifest.snapshot_id, file_name)
        evaluation_objects[file_name] = await evaluation_store.put_file(
            snapshot_path / file_name, key
        )
        current += 1
        if progress is not None:
            progress("archive_snapshot", current, total)
    return SnapshotArchiveResult(
        snapshot_id=manifest.snapshot_id,
        runtime_objects=runtime_objects,
        evaluation_objects=evaluation_objects,
    )


async def sync_assets(
    snapshot_path: Path,
    *,
    project_root: Path,
    runtime_store: ObjectStore,
    progress: StorageProgress | None = None,
    expected_asset_sha256: Mapping[str, str] | None = None,
) -> AssetSyncResult:
    """Upload local benchmark assets by content hash without downloading URLs."""

    report = await asyncio.to_thread(preflight_snapshot, snapshot_path)
    assets = list(iter_jsonl(snapshot_path / "assets.jsonl", AssetRef))
    if expected_asset_sha256 is not None and (
        set(expected_asset_sha256) != {asset.asset_id for asset in assets}
        or any(not re.fullmatch(r"[a-f0-9]{64}", value)
               for value in expected_asset_sha256.values())
    ):
        raise ValueError("snapshot_asset_checksum_coverage_mismatch")
    synced: list[AssetSyncRecord] = []
    skipped = 0
    for current, asset in enumerate(assets, start=1):
        if asset.asset_id is None:
            raise ValueError("canonical benchmark asset is missing asset_id")
        if asset.local_path is None:
            if expected_asset_sha256 is not None:
                raise ValueError("snapshot_validated_asset_not_local")
            skipped += 1
        else:
            source = resolve_local_asset(project_root, asset.local_path)
            expected = (expected_asset_sha256[asset.asset_id]
                        if expected_asset_sha256 is not None else asset.sha256)
            if asset.sha256 is not None and expected != asset.sha256:
                raise ValueError("snapshot_asset_checksum_identity_mismatch")
            # Canonical snapshots already bind local assets to a verified hash.
            # Let put_file perform the single content read and validate it.
            digest = (
                expected.lower()
                if expected is not None
                else await asyncio.to_thread(sha256_file, source)
            )
            key = asset_object_key(report.manifest.dataset.value, digest)
            uploaded = await runtime_store.put_file(
                source,
                key,
                expected_sha256=digest,
            )
            synced.append(
                AssetSyncRecord(
                    asset_id=asset.asset_id,
                    uri=uploaded.uri,
                    sha256=digest,
                    size_bytes=uploaded.size_bytes,
                )
            )
        if progress is not None:
            progress("sync_assets", current, len(assets))
    return AssetSyncResult(
        snapshot_id=report.manifest.snapshot_id,
        synced=synced,
        skipped_remote_only=skipped,
    )


async def verify_archive(
    snapshot_path: Path,
    *,
    runtime_store: ObjectStore,
    evaluation_store: ObjectStore,
) -> SnapshotArchiveResult:
    """Verify each archived object against its local canonical file checksum."""

    report = await asyncio.to_thread(preflight_snapshot, snapshot_path)
    runtime_objects: dict[str, ObjectMetadata] = {}
    evaluation_objects: dict[str, ObjectMetadata] = {}
    for file_name, store, target in (
        *((name, runtime_store, runtime_objects) for name in RUNTIME_SNAPSHOT_FILES),
        *((name, evaluation_store, evaluation_objects) for name in EVALUATION_SNAPSHOT_FILES),
    ):
        key = snapshot_object_key(
            report.manifest.dataset.value, report.manifest.snapshot_id, file_name
        )
        remote = await store.head(key)
        if remote is None:
            raise FileNotFoundError(f"archived object does not exist: {file_name}")
        local_hash = await asyncio.to_thread(sha256_file, snapshot_path / file_name)
        if remote.sha256 != local_hash:
            raise ValueError(f"archived object checksum mismatch: {file_name}")
        target[file_name] = remote
    return SnapshotArchiveResult(
        snapshot_id=report.manifest.snapshot_id,
        runtime_objects=runtime_objects,
        evaluation_objects=evaluation_objects,
    )
