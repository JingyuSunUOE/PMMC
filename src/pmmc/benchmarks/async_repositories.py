"""Async adapters over immutable local canonical JSONL snapshots."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from pmmc.benchmarks.repositories import MemoryCorpus, RuntimeQueryStore, SessionStore
from pmmc.benchmarks.snapshot import find_snapshot, load_manifest
from pmmc.schemas import (
    BenchmarkQuery,
    BenchmarkSelection,
    DatasetSnapshotManifest,
    MemoryUnit,
    SessionRecord,
)


class JsonlSnapshotRepository:
    def __init__(self, snapshots_root: Path) -> None:
        self.snapshots_root = snapshots_root

    async def get_manifest(self, snapshot_id: str) -> DatasetSnapshotManifest:
        path = find_snapshot(self.snapshots_root, snapshot_id)
        return await asyncio.to_thread(load_manifest, path)


class JsonlMemoryRepository:
    def __init__(self, snapshots_root: Path) -> None:
        self.snapshots_root = snapshots_root

    async def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[MemoryUnit]:
        path = find_snapshot(self.snapshots_root, snapshot_id)
        records = await asyncio.to_thread(
            lambda: list(MemoryCorpus(path).iter_visible(conversation_id, cutoff_session_id))
        )
        for record in records:
            yield record


class JsonlSessionRepository:
    def __init__(self, snapshots_root: Path) -> None:
        self.snapshots_root = snapshots_root

    async def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[SessionRecord]:
        path = find_snapshot(self.snapshots_root, snapshot_id)
        records = await asyncio.to_thread(
            lambda: list(
                SessionStore(path).iter_visible(conversation_id, cutoff_session_id)
            )
        )
        for record in records:
            yield record


class JsonlRuntimeQueryRepository:
    def __init__(self, snapshots_root: Path) -> None:
        self.snapshots_root = snapshots_root

    async def select(
        self,
        snapshot_id: str,
        selection: BenchmarkSelection,
    ) -> list[BenchmarkQuery]:
        path = find_snapshot(self.snapshots_root, snapshot_id)
        return await asyncio.to_thread(RuntimeQueryStore(path).select, selection)
