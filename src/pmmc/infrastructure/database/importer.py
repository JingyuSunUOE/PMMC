"""Atomic import of validated canonical snapshots into Postgres."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Table, delete, func, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from pmmc.benchmarks.canonical import iter_jsonl, preflight_snapshot
from pmmc.infrastructure.database.constants import SnapshotImportState
from pmmc.infrastructure.database.errors import SnapshotImportError
from pmmc.infrastructure.database.mappers import (
    asset_to_row,
    benchmark_query_to_row,
    image_to_row,
    manifest_to_row,
    memory_image_refs,
    memory_to_row,
    query_gold_to_rows,
    run_manifest_to_row,
    session_to_row,
)
from pmmc.infrastructure.database.models import (
    AssetModel,
    BenchmarkQueryModel,
    BenchmarkSessionModel,
    DatasetSnapshotModel,
    ImageModel,
    MemoryUnitImageModel,
    MemoryUnitModel,
    QueryGoldEvidenceIdModel,
    QueryGoldImageIdModel,
    QueryGoldMemoryIdModel,
    QueryGoldModel,
    QueryGoldSessionIdModel,
    QueryImageModel,
    RunManifestModel,
    RunSnapshotLinkModel,
    SnapshotImportAttemptModel,
)
from pmmc.schemas import (
    AssetRef,
    BenchmarkQuery,
    DatasetSnapshotManifest,
    MemoryUnit,
    QueryGoldRecord,
    RunManifest,
    SessionRecord,
)
from pmmc.schemas.base import utc_now

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class SnapshotImportResult:
    """Non-sensitive summary of one import attempt."""

    snapshot_id: str
    attempt_id: str
    state: SnapshotImportState
    counts: dict[str, int]
    idempotent: bool = False


def _chunks(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _table(model: Any) -> Table:
    return model.__table__


def _safe_error_message(error: BaseException) -> str:
    if isinstance(error, SnapshotImportError):
        return " ".join(str(error).split())[:500]
    return "database import failed; inspect the local run log for the error type"


class CanonicalSnapshotImporter:
    """Import one immutable canonical snapshot with transaction-level integrity."""

    def __init__(self, engine: AsyncEngine, *, batch_size: int = 2000) -> None:
        if not 1 <= batch_size <= 10_000:
            raise ValueError("batch_size must be between 1 and 10000")
        self.engine = engine
        self.batch_size = batch_size

    async def import_snapshot(
        self,
        snapshot_path: Path,
        *,
        run_manifest: RunManifest | None = None,
        progress: ProgressCallback | None = None,
    ) -> SnapshotImportResult:
        """Preflight and atomically import a snapshot using the owner connection."""

        report = await asyncio.to_thread(preflight_snapshot, snapshot_path)
        manifest = report.manifest
        attempt_id = uuid4().hex
        async with self.engine.connect() as connection:
            await connection.execute(
                text("SELECT pg_advisory_lock(hashtextextended(:snapshot_id, 0))"),
                {"snapshot_id": manifest.snapshot_id},
            )
            await connection.commit()
            try:
                existing = await self._get_snapshot(connection, manifest.snapshot_id)
                await connection.rollback()
                if existing is not None:
                    try:
                        self._assert_same_snapshot(existing, manifest.to_dict())
                    except SnapshotImportError as exc:
                        await self._record_existing_failed_attempt(
                            connection,
                            attempt_id=attempt_id,
                            snapshot_id=manifest.snapshot_id,
                            run_id=run_manifest.run_id if run_manifest else None,
                            counts=report.counts,
                            error=exc,
                            metadata={"existing_snapshot": True},
                        )
                        raise
                    if existing["import_state"] == SnapshotImportState.READY.value:
                        try:
                            await self.verify_snapshot(
                                connection, manifest.snapshot_id, report.counts
                            )
                            await connection.rollback()
                            await self._record_idempotent_attempt(
                                connection,
                                attempt_id=attempt_id,
                                snapshot_id=manifest.snapshot_id,
                                run_id=run_manifest.run_id if run_manifest else None,
                                counts=report.counts,
                                run_manifest=run_manifest,
                            )
                        except Exception as exc:
                            if connection.in_transaction():
                                await connection.rollback()
                            await self._record_existing_failed_attempt(
                                connection,
                                attempt_id=attempt_id,
                                snapshot_id=manifest.snapshot_id,
                                run_id=run_manifest.run_id if run_manifest else None,
                                counts=report.counts,
                                error=exc,
                                metadata={
                                    "existing_snapshot": True,
                                    "idempotent_verification": True,
                                },
                            )
                            if isinstance(exc, SnapshotImportError):
                                raise
                            raise SnapshotImportError(
                                f"snapshot verification failed: {type(exc).__name__}"
                            ) from exc
                        return SnapshotImportResult(
                            snapshot_id=manifest.snapshot_id,
                            attempt_id=attempt_id,
                            state=SnapshotImportState.READY,
                            counts=report.counts,
                            idempotent=True,
                        )

                await self._start_attempt(
                    connection,
                    attempt_id=attempt_id,
                    manifest=manifest,
                    run_id=run_manifest.run_id if run_manifest else None,
                    existing=existing is not None,
                )
                try:
                    async with connection.begin():
                        await self._clear_failed_snapshot_rows(connection, manifest.snapshot_id)
                        await self._insert_snapshot_data(
                            connection,
                            snapshot_path=snapshot_path,
                            snapshot_id=manifest.snapshot_id,
                            images=report.images,
                            run_manifest=run_manifest,
                            progress=progress,
                        )
                        await self.verify_snapshot(connection, manifest.snapshot_id, report.counts)
                        await connection.execute(
                            update(DatasetSnapshotModel)
                            .where(DatasetSnapshotModel.snapshot_id == manifest.snapshot_id)
                            .values(
                                import_state=SnapshotImportState.READY.value,
                                imported_at=utc_now(),
                            )
                        )
                        await connection.execute(
                            update(SnapshotImportAttemptModel)
                            .where(SnapshotImportAttemptModel.attempt_id == attempt_id)
                            .values(
                                status=SnapshotImportState.READY.value,
                                completed_at=utc_now(),
                                imported_counts=report.counts,
                            )
                        )
                except Exception as exc:
                    await connection.rollback()
                    await self._mark_failed(
                        connection,
                        snapshot_id=manifest.snapshot_id,
                        attempt_id=attempt_id,
                        error=exc,
                    )
                    if isinstance(exc, SnapshotImportError):
                        raise
                    raise SnapshotImportError(
                        f"snapshot import failed: {type(exc).__name__}"
                    ) from exc
            finally:
                if connection.in_transaction():
                    await connection.rollback()
                await connection.execute(
                    text("SELECT pg_advisory_unlock(hashtextextended(:snapshot_id, 0))"),
                    {"snapshot_id": manifest.snapshot_id},
                )
                await connection.commit()

        return SnapshotImportResult(
            snapshot_id=manifest.snapshot_id,
            attempt_id=attempt_id,
            state=SnapshotImportState.READY,
            counts=report.counts,
        )

    async def _get_snapshot(
        self,
        connection: AsyncConnection,
        snapshot_id: str,
    ) -> Mapping[str, Any] | None:
        result = await connection.execute(
            select(DatasetSnapshotModel.__table__).where(
                DatasetSnapshotModel.snapshot_id == snapshot_id
            )
        )
        row = result.mappings().first()
        return dict(row) if row is not None else None

    @staticmethod
    def _assert_same_snapshot(
        existing: Mapping[str, Any],
        incoming: dict[str, Any],
    ) -> None:
        immutable_fields = (
            "dataset",
            "dataset_version",
            "adapter_name",
            "adapter_version",
            "schema_version",
            "evaluation_policy_version",
            "excluded_source_categories",
            "source_content_hash",
            "canonical_content_hash",
            "counts",
            "category_counts",
            "eligible_category_counts",
            "excluded_category_counts",
        )
        if any(existing[field] != incoming[field] for field in immutable_fields):
            raise SnapshotImportError("snapshot ID already exists with different immutable content")

    async def _start_attempt(
        self,
        connection: AsyncConnection,
        *,
        attempt_id: str,
        manifest: DatasetSnapshotManifest,
        run_id: str | None,
        existing: bool,
    ) -> None:
        async with connection.begin():
            if existing:
                await connection.execute(
                    update(DatasetSnapshotModel)
                    .where(DatasetSnapshotModel.snapshot_id == manifest.snapshot_id)
                    .values(import_state=SnapshotImportState.IMPORTING.value)
                )
            else:
                await connection.execute(
                    insert(_table(DatasetSnapshotModel)),
                    [
                        manifest_to_row(
                            manifest,
                            import_state=SnapshotImportState.IMPORTING.value,
                        )
                    ],
                )
            await connection.execute(
                insert(_table(SnapshotImportAttemptModel)),
                [
                    {
                        "attempt_id": attempt_id,
                        "snapshot_id": manifest.snapshot_id,
                        "run_id": run_id,
                        "status": SnapshotImportState.IMPORTING.value,
                        "started_at": utc_now(),
                        "imported_counts": {},
                        "metadata": {},
                    }
                ],
            )

    async def _record_idempotent_attempt(
        self,
        connection: AsyncConnection,
        *,
        attempt_id: str,
        snapshot_id: str,
        run_id: str | None,
        counts: dict[str, int],
        run_manifest: RunManifest | None,
    ) -> None:
        now = utc_now()
        async with connection.begin():
            if run_manifest is not None:
                await self._insert_run_manifest(connection, run_manifest, snapshot_id)
            await connection.execute(
                insert(_table(SnapshotImportAttemptModel)),
                [
                    {
                        "attempt_id": attempt_id,
                        "snapshot_id": snapshot_id,
                        "run_id": run_id,
                        "status": SnapshotImportState.READY.value,
                        "started_at": now,
                        "completed_at": now,
                        "imported_counts": counts,
                        "metadata": {"idempotent": True},
                    }
                ],
            )

    async def _record_existing_failed_attempt(
        self,
        connection: AsyncConnection,
        *,
        attempt_id: str,
        snapshot_id: str,
        run_id: str | None,
        counts: dict[str, int],
        error: BaseException,
        metadata: dict[str, Any],
    ) -> None:
        """Audit a rejected ready-snapshot attempt without changing its state."""

        now = utc_now()
        async with connection.begin():
            await connection.execute(
                insert(_table(SnapshotImportAttemptModel)),
                [
                    {
                        "attempt_id": attempt_id,
                        "snapshot_id": snapshot_id,
                        "run_id": run_id,
                        "status": SnapshotImportState.FAILED.value,
                        "started_at": now,
                        "completed_at": now,
                        "imported_counts": counts,
                        "error_type": type(error).__name__,
                        "error_message": _safe_error_message(error),
                        "metadata": metadata,
                    }
                ],
            )

    async def _mark_failed(
        self,
        connection: AsyncConnection,
        *,
        snapshot_id: str,
        attempt_id: str,
        error: BaseException,
    ) -> None:
        async with connection.begin():
            await connection.execute(
                update(DatasetSnapshotModel)
                .where(DatasetSnapshotModel.snapshot_id == snapshot_id)
                .values(import_state=SnapshotImportState.FAILED.value)
            )
            await connection.execute(
                update(SnapshotImportAttemptModel)
                .where(SnapshotImportAttemptModel.attempt_id == attempt_id)
                .values(
                    status=SnapshotImportState.FAILED.value,
                    completed_at=utc_now(),
                    error_type=type(error).__name__,
                    error_message=_safe_error_message(error),
                )
            )

    async def _clear_failed_snapshot_rows(
        self,
        connection: AsyncConnection,
        snapshot_id: str,
    ) -> None:
        for model in (
            QueryGoldEvidenceIdModel,
            QueryGoldMemoryIdModel,
            QueryGoldSessionIdModel,
            QueryGoldImageIdModel,
            QueryGoldModel,
            QueryImageModel,
            BenchmarkQueryModel,
            MemoryUnitImageModel,
            MemoryUnitModel,
            ImageModel,
            AssetModel,
            BenchmarkSessionModel,
        ):
            await connection.execute(
                delete(_table(model)).where(_table(model).c.snapshot_id == snapshot_id)
            )

    async def _insert_snapshot_data(
        self,
        connection: AsyncConnection,
        *,
        snapshot_path: Path,
        snapshot_id: str,
        images: dict[str, Any],
        run_manifest: RunManifest | None,
        progress: ProgressCallback | None,
    ) -> None:
        if run_manifest is not None:
            await self._insert_run_manifest(connection, run_manifest, snapshot_id)

        await self._insert_stream(
            connection,
            BenchmarkSessionModel,
            (
                session_to_row(snapshot_id, value)
                for value in iter_jsonl(snapshot_path / "sessions.jsonl", SessionRecord)
            ),
            stage="sessions",
            total=self._line_count(snapshot_path / "sessions.jsonl"),
            progress=progress,
        )
        await self._insert_stream(
            connection,
            AssetModel,
            (
                asset_to_row(snapshot_id, value)
                for value in iter_jsonl(snapshot_path / "assets.jsonl", AssetRef)
            ),
            stage="assets",
            total=self._line_count(snapshot_path / "assets.jsonl"),
            progress=progress,
        )
        await self._insert_stream(
            connection,
            ImageModel,
            (image_to_row(snapshot_id, images[key]) for key in sorted(images)),
            stage="images",
            total=len(images),
            progress=progress,
        )

        memory_total = self._line_count(snapshot_path / "memory_units.jsonl")
        memory_links: list[dict[str, Any]] = []

        def memory_rows() -> Iterator[dict[str, Any]]:
            for memory in iter_jsonl(snapshot_path / "memory_units.jsonl", MemoryUnit):
                for ordinal, image in enumerate(memory_image_refs(memory)):
                    memory_links.append(
                        {
                            "snapshot_id": snapshot_id,
                            "memory_id": memory.memory_id,
                            "image_id": image.image_id,
                            "ordinal": ordinal,
                        }
                    )
                yield memory_to_row(snapshot_id, memory)

        await self._insert_stream(
            connection,
            MemoryUnitModel,
            memory_rows(),
            stage="memory_units",
            total=memory_total,
            progress=progress,
        )
        await self._insert_stream(
            connection,
            MemoryUnitImageModel,
            iter(memory_links),
            stage="memory_images",
            total=len(memory_links),
            progress=progress,
        )

        query_total = self._line_count(snapshot_path / "queries.jsonl")
        query_links: list[dict[str, Any]] = []

        def query_rows() -> Iterator[dict[str, Any]]:
            for record in iter_jsonl(snapshot_path / "queries.jsonl", BenchmarkQuery):
                for ordinal, image in enumerate(record.query.query_image_refs):
                    query_links.append(
                        {
                            "snapshot_id": snapshot_id,
                            "query_id": record.query.query_id,
                            "image_id": image.image_id,
                            "ordinal": ordinal,
                        }
                    )
                yield benchmark_query_to_row(snapshot_id, record)

        await self._insert_stream(
            connection,
            BenchmarkQueryModel,
            query_rows(),
            stage="queries",
            total=query_total,
            progress=progress,
        )
        await self._insert_stream(
            connection,
            QueryImageModel,
            iter(query_links),
            stage="query_images",
            total=len(query_links),
            progress=progress,
        )

        gold_rows: dict[str, list[dict[str, Any]]] = {
            "query_gold": [],
            "evidence_ids": [],
            "memory_ids": [],
            "session_ids": [],
            "image_ids": [],
        }
        for record in iter_jsonl(snapshot_path / "query_gold.jsonl", QueryGoldRecord):
            mapped = query_gold_to_rows(snapshot_id, record)
            for key, values in mapped.items():
                gold_rows[key].extend(values)
        for stage, model, key in (
            ("query_gold", QueryGoldModel, "query_gold"),
            ("gold_evidence_ids", QueryGoldEvidenceIdModel, "evidence_ids"),
            ("gold_memory_ids", QueryGoldMemoryIdModel, "memory_ids"),
            ("gold_session_ids", QueryGoldSessionIdModel, "session_ids"),
            ("gold_image_ids", QueryGoldImageIdModel, "image_ids"),
        ):
            await self._insert_stream(
                connection,
                model,
                iter(gold_rows[key]),
                stage=stage,
                total=len(gold_rows[key]),
                progress=progress,
            )

    async def _insert_run_manifest(
        self,
        connection: AsyncConnection,
        manifest: RunManifest,
        snapshot_id: str,
    ) -> None:
        result = await connection.execute(
            select(RunManifestModel.__table__).where(RunManifestModel.run_id == manifest.run_id)
        )
        existing = result.mappings().first()
        if existing is None:
            await connection.execute(
                insert(_table(RunManifestModel)), [run_manifest_to_row(manifest)]
            )
        elif existing["manifest_json"] != manifest.to_dict():
            raise SnapshotImportError("run ID already exists with a different manifest")
        existing_link = await connection.scalar(
            select(RunSnapshotLinkModel.run_id).where(
                RunSnapshotLinkModel.run_id == manifest.run_id,
                RunSnapshotLinkModel.snapshot_id == snapshot_id,
            )
        )
        if existing_link is None:
            await connection.execute(
                insert(_table(RunSnapshotLinkModel)),
                [{"run_id": manifest.run_id, "snapshot_id": snapshot_id}],
            )

    async def _insert_stream(
        self,
        connection: AsyncConnection,
        model: Any,
        rows: Iterable[dict[str, Any]],
        *,
        stage: str,
        total: int,
        progress: ProgressCallback | None,
    ) -> None:
        current = 0
        for batch in _chunks(rows, self.batch_size):
            await connection.execute(insert(_table(model)), batch)
            current += len(batch)
            if progress is not None:
                progress(stage, current, total)

    @staticmethod
    def _line_count(path: Path) -> int:
        with path.open(encoding="utf-8") as stream:
            return sum(bool(line.strip()) for line in stream)

    async def verify_snapshot(
        self,
        connection: AsyncConnection,
        snapshot_id: str,
        expected_counts: dict[str, int],
    ) -> dict[str, int]:
        """Verify persisted counts without reading evaluation values."""

        models = {
            "sessions": BenchmarkSessionModel,
            "assets": AssetModel,
            "images": ImageModel,
            "memory_units": MemoryUnitModel,
            "queries": BenchmarkQueryModel,
            "query_gold": QueryGoldModel,
        }
        actual: dict[str, int] = {}
        for key, model in models.items():
            table = _table(model)
            value = await connection.scalar(
                select(func.count()).select_from(table).where(table.c.snapshot_id == snapshot_id)
            )
            actual[key] = int(value or 0)
        for key in models:
            if key in expected_counts and actual[key] != expected_counts[key]:
                raise SnapshotImportError(
                    f"persisted count mismatch for {key}: "
                    f"expected {expected_counts[key]}, got {actual[key]}"
                )
        return actual
