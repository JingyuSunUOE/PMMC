"""Narrow readers that preserve runtime/gold access boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pmmc.benchmarks.selection import select_queries
from pmmc.schemas import (
    BenchmarkQuery,
    BenchmarkSelection,
    MemoryUnit,
    QueryGold,
    QueryGoldRecord,
    QueryInput,
    SessionRecord,
)


def _iter_jsonl(path: Path, schema_type):  # type: ignore[no-untyped-def]
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield schema_type.from_json(line)


class MemoryCorpus:
    """Read canonical memories without exposing benchmark queries or gold."""

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path

    def iter_all(self) -> Iterator[MemoryUnit]:
        yield from _iter_jsonl(self.snapshot_path / "memory_units.jsonl", MemoryUnit)

    def iter_visible(
        self,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> Iterator[MemoryUnit]:
        sessions = {
            session.session_id: session
            for session in _iter_jsonl(self.snapshot_path / "sessions.jsonl", SessionRecord)
            if session.conversation_id == conversation_id
        }
        try:
            cutoff_order = sessions[cutoff_session_id].session_order
        except KeyError as exc:
            raise KeyError(
                f"cutoff session does not exist in conversation: {cutoff_session_id}"
            ) from exc
        for memory in self.iter_all():
            if memory.conversation_id != conversation_id:
                continue
            available_after = sessions.get(memory.available_after_session)
            if available_after is not None and available_after.session_order <= cutoff_order:
                yield memory


class SessionStore:
    """Read canonical session boundaries without consulting memory metadata."""

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path

    def iter_visible(
        self,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> Iterator[SessionRecord]:
        sessions = sorted(
            (
                session
                for session in _iter_jsonl(
                    self.snapshot_path / "sessions.jsonl", SessionRecord
                )
                if session.conversation_id == conversation_id
            ),
            key=lambda item: item.session_order,
        )
        by_id = {session.session_id: session for session in sessions}
        try:
            cutoff_order = by_id[cutoff_session_id].session_order
        except KeyError as exc:
            raise KeyError(
                f"cutoff session does not exist in conversation: {cutoff_session_id}"
            ) from exc
        yield from (session for session in sessions if session.session_order <= cutoff_order)


class RuntimeQueryStore:
    """Read and select runtime-safe benchmark queries."""

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path

    def iter_records(self) -> Iterator[BenchmarkQuery]:
        yield from _iter_jsonl(self.snapshot_path / "queries.jsonl", BenchmarkQuery)

    def select(self, selection: BenchmarkSelection) -> list[BenchmarkQuery]:
        return select_queries(self.iter_records(), selection)

    def iter_queries(self, selection: BenchmarkSelection) -> Iterator[QueryInput]:
        for record in self.select(selection):
            yield record.query


class GoldStore:
    """Evaluation-only gold reader keyed by canonical query ID."""

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path
        self._records: dict[str, QueryGold] | None = None

    def _load(self) -> dict[str, QueryGold]:
        if self._records is None:
            self._records = {
                record.query_id: record.gold
                for record in _iter_jsonl(
                    self.snapshot_path / "query_gold.jsonl",
                    QueryGoldRecord,
                )
            }
        return self._records

    def get(self, query_id: str) -> QueryGold:
        try:
            return self._load()[query_id]
        except KeyError as exc:
            raise KeyError(f"gold query record does not exist: {query_id}") from exc
