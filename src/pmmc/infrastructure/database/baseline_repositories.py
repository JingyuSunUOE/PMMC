"""Postgres repositories and atomic importer for baseline foundation objects."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from sqlalchemy import Table, func, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from pmmc.infrastructure.database.baseline_mappers import (
    context_from_parts,
    context_to_rows,
    corpus_to_row,
    document_from_parts,
    document_to_rows,
    window_from_parts,
    window_to_rows,
)
from pmmc.infrastructure.database.baseline_models import (
    ContextAssemblyImageModel,
    ContextAssemblyMemoryUnitModel,
    ContextAssemblyModel,
    ContextAssemblyRetrievalCandidateModel,
    MemoryWindowModel,
    MemoryWindowUnitModel,
    RetrievalCorpusModel,
    RetrievalDocumentImageModel,
    RetrievalDocumentModel,
)
from pmmc.infrastructure.database.errors import PersistenceConflictError
from pmmc.infrastructure.database.experiment_repositories import _insert_idempotent
from pmmc.infrastructure.database.models import DatasetSnapshotModel, MemoryUnitImageModel
from pmmc.schemas import (
    BaselineCorpusManifest,
    ContextAssemblyTrace,
    MemoryWindow,
    RepresentationKind,
    RetrievalDocument,
)

ProgressCallback = Callable[[str, int, int], None]


def _table(model: Any) -> Table:
    return model.__table__


def _mapping(value: Any) -> dict[str, Any]:
    return {
        prop.columns[0].name: getattr(value, prop.key)
        for prop in value.__mapper__.column_attrs
    }


def _corpus_manifest_identity(value: BaselineCorpusManifest) -> dict[str, Any]:
    """Return immutable corpus content while excluding wall-clock provenance."""

    return value.model_dump(mode="json", exclude={"created_at"})


def _batched(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, size)):
        yield chunk


class BaselineCorpusImporter:
    """Import a validated corpus with advisory lock and one database transaction."""

    def __init__(self, engine: AsyncEngine, *, batch_size: int = 2000) -> None:
        if not 1 <= batch_size <= 10_000:
            raise ValueError("batch_size must be between 1 and 10000")
        self.engine = engine
        self.batch_size = batch_size

    async def import_corpus(
        self,
        corpus_path: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> tuple[BaselineCorpusManifest, bool]:
        # Import lazily so the infrastructure module can be loaded before the
        # baseline package's public API without creating a repository/index cycle.
        from pmmc.baselines.corpus import CorpusReader

        reader = await asyncio.to_thread(CorpusReader, corpus_path)
        manifest = reader.manifest
        async with self.engine.connect() as connection:
            async with connection.begin():
                await connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:corpus_id, 0))"),
                    {"corpus_id": manifest.corpus_id},
                )
                snapshot = (
                    await connection.execute(
                        select(_table(DatasetSnapshotModel)).where(
                            _table(DatasetSnapshotModel).c.snapshot_id == manifest.snapshot_id
                        )
                    )
                ).mappings().one_or_none()
                if snapshot is None or snapshot["import_state"] != "ready":
                    raise ValueError("corpus snapshot must be imported and ready")
                if (
                    snapshot["dataset"] != manifest.dataset.value
                    or snapshot["canonical_content_hash"]
                    != manifest.spec.snapshot_content_hash
                ):
                    raise PersistenceConflictError(
                        "corpus dataset/content hash does not match its imported snapshot"
                    )
                existing = await connection.execute(
                    select(_table(RetrievalCorpusModel)).where(
                        _table(RetrievalCorpusModel).c.corpus_id == manifest.corpus_id
                    )
                )
                stored = existing.mappings().one_or_none()
                if stored is not None:
                    stored_manifest = BaselineCorpusManifest.model_validate(
                        stored["manifest_json"]
                    )
                    if _corpus_manifest_identity(
                        stored_manifest
                    ) != _corpus_manifest_identity(manifest):
                        raise PersistenceConflictError(
                            "corpus ID already exists with different manifest content"
                        )
                    await self._verify_counts(connection, manifest)
                    if stored["status"] != "ready":
                        raise PersistenceConflictError("existing corpus is not ready")
                    return manifest, True

                await connection.execute(
                    insert(_table(RetrievalCorpusModel)).values(
                        corpus_to_row(manifest, building=True)
                    )
                )
                imported_windows = 0
                for window_chunk in _batched(reader.iter_windows(), self.batch_size):
                    window_mapped = [window_to_rows(window) for window in window_chunk]
                    await connection.execute(
                        insert(_table(MemoryWindowModel)),
                        [rows["window"][0] for rows in window_mapped],
                    )
                    unit_rows = [row for rows in window_mapped for row in rows["units"]]
                    if unit_rows:
                        await connection.execute(
                            insert(_table(MemoryWindowUnitModel)), unit_rows
                        )
                    imported_windows += len(window_chunk)
                    if progress is not None:
                        progress("windows", imported_windows, manifest.window_count)
                imported_documents = 0
                for document_chunk in _batched(
                    reader.iter_documents(), self.batch_size
                ):
                    document_mapped = [
                        document_to_rows(document) for document in document_chunk
                    ]
                    await connection.execute(
                        insert(_table(RetrievalDocumentModel)),
                        [rows["document"][0] for rows in document_mapped],
                    )
                    image_rows = [
                        row for rows in document_mapped for row in rows["images"]
                    ]
                    if image_rows:
                        await connection.execute(
                            insert(_table(RetrievalDocumentImageModel)), image_rows
                        )
                    imported_documents += len(document_chunk)
                    if progress is not None:
                        progress(
                            "documents",
                            imported_documents,
                            manifest.document_count,
                        )
                await self._verify_counts(connection, manifest)
                await connection.execute(
                    update(_table(RetrievalCorpusModel))
                    .where(
                        _table(RetrievalCorpusModel).c.corpus_id == manifest.corpus_id
                    )
                    .values(status="ready")
                )
        return manifest, False

    @staticmethod
    async def _verify_counts(connection: Any, manifest: BaselineCorpusManifest) -> None:
        window_count = await connection.scalar(
            select(func.count()).select_from(_table(MemoryWindowModel)).where(
                _table(MemoryWindowModel).c.corpus_id == manifest.corpus_id
            )
        )
        document_count = await connection.scalar(
            select(func.count()).select_from(_table(RetrievalDocumentModel)).where(
                _table(RetrievalDocumentModel).c.corpus_id == manifest.corpus_id
            )
        )
        if window_count != manifest.window_count or document_count != manifest.document_count:
            raise PersistenceConflictError("persisted corpus counts do not match manifest")


class PostgresCorpusRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_manifest(self, corpus_id: str) -> BaselineCorpusManifest:
        value = await self.session.get(RetrievalCorpusModel, corpus_id)
        if value is None or value.status != "ready":
            raise KeyError(f"ready corpus does not exist: {corpus_id}")
        return BaselineCorpusManifest.model_validate(value.manifest_json)

    async def iter_windows(self, corpus_id: str) -> AsyncIterator[MemoryWindow]:
        statement = (
            select(MemoryWindowModel)
            .where(MemoryWindowModel.corpus_id == corpus_id)
            .order_by(
                MemoryWindowModel.conversation_id,
                MemoryWindowModel.start_timestamp_order,
                MemoryWindowModel.window_id,
            )
        )
        for value in (await self.session.execute(statement)).scalars():
            units = list(
                (
                    await self.session.execute(
                        select(MemoryWindowUnitModel)
                        .where(MemoryWindowUnitModel.window_id == value.window_id)
                        .order_by(MemoryWindowUnitModel.ordinal)
                    )
                ).scalars()
            )
            image_ids: list[str] = []
            for unit in units:
                image_ids.extend(
                    (
                        await self.session.execute(
                            select(MemoryUnitImageModel.image_id)
                            .where(
                                MemoryUnitImageModel.snapshot_id == value.snapshot_id,
                                MemoryUnitImageModel.memory_id == unit.memory_id,
                            )
                            .order_by(MemoryUnitImageModel.ordinal)
                        )
                    )
                    .scalars()
                    .all()
                )
            yield window_from_parts(
                _mapping(value),
                [item.memory_id for item in units],
                list(dict.fromkeys(image_ids)),
            )

    async def iter_documents(self, corpus_id: str) -> AsyncIterator[RetrievalDocument]:
        values = (
            (
                await self.session.execute(
                    select(RetrievalDocumentModel)
                    .where(RetrievalDocumentModel.corpus_id == corpus_id)
                    .order_by(RetrievalDocumentModel.document_id)
                )
            )
            .scalars()
            .all()
        )
        for value in values:
            images = list(
                (
                    await self.session.execute(
                        select(RetrievalDocumentImageModel.image_id)
                        .where(RetrievalDocumentImageModel.document_id == value.document_id)
                        .order_by(RetrievalDocumentImageModel.ordinal)
                    )
                ).scalars()
            )
            yield document_from_parts(_mapping(value), images)

    async def get_window(self, corpus_id: str, window_id: str) -> MemoryWindow:
        value = await self.session.get(MemoryWindowModel, window_id)
        if value is None or value.corpus_id != corpus_id:
            raise KeyError(f"memory window does not exist in corpus: {window_id}")
        units = list(
            (
                await self.session.execute(
                    select(MemoryWindowUnitModel)
                    .where(MemoryWindowUnitModel.window_id == window_id)
                    .order_by(MemoryWindowUnitModel.ordinal)
                )
            ).scalars()
        )
        image_ids: list[str] = []
        for unit in units:
            image_ids.extend(
                (
                    await self.session.execute(
                        select(MemoryUnitImageModel.image_id)
                        .where(
                            MemoryUnitImageModel.snapshot_id == value.snapshot_id,
                            MemoryUnitImageModel.memory_id == unit.memory_id,
                        )
                        .order_by(MemoryUnitImageModel.ordinal)
                    )
                ).scalars()
            )
        return window_from_parts(
            _mapping(value),
            [item.memory_id for item in units],
            list(dict.fromkeys(image_ids)),
        )

    async def get_document(self, corpus_id: str, document_id: str) -> RetrievalDocument:
        value = await self.session.get(RetrievalDocumentModel, document_id)
        if value is None or value.corpus_id != corpus_id:
            raise KeyError(f"retrieval document does not exist in corpus: {document_id}")
        images = list(
            (
                await self.session.execute(
                    select(RetrievalDocumentImageModel.image_id)
                    .where(RetrievalDocumentImageModel.document_id == document_id)
                    .order_by(RetrievalDocumentImageModel.ordinal)
                )
            ).scalars()
        )
        return document_from_parts(_mapping(value), images)

    async def list_document_ids(
        self,
        corpus_id: str,
        representations: list[RepresentationKind],
    ) -> list[str]:
        if not representations:
            raise ValueError("at least one retrieval representation is required")
        return list(
            (
                await self.session.execute(
                    select(RetrievalDocumentModel.document_id)
                    .where(
                        RetrievalDocumentModel.corpus_id == corpus_id,
                        RetrievalDocumentModel.representation.in_(
                            [item.value for item in representations]
                        ),
                    )
                    .order_by(RetrievalDocumentModel.document_id)
                )
            ).scalars()
        )

    async def get_documents(
        self, corpus_id: str, document_ids: list[str]
    ) -> list[RetrievalDocument]:
        if not document_ids:
            return []
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document IDs cannot contain duplicates")
        values = list(
            (
                await self.session.execute(
                    select(RetrievalDocumentModel).where(
                        RetrievalDocumentModel.corpus_id == corpus_id,
                        RetrievalDocumentModel.document_id.in_(document_ids),
                    )
                )
            ).scalars()
        )
        by_id = {value.document_id: value for value in values}
        missing = set(document_ids) - set(by_id)
        if missing:
            raise KeyError(f"retrieval document does not exist: {sorted(missing)[0]}")
        image_rows = (
            await self.session.execute(
                select(RetrievalDocumentImageModel)
                .where(RetrievalDocumentImageModel.document_id.in_(document_ids))
                .order_by(
                    RetrievalDocumentImageModel.document_id,
                    RetrievalDocumentImageModel.ordinal,
                )
            )
        ).scalars()
        images: dict[str, list[str]] = {}
        for row in image_rows:
            images.setdefault(row.document_id, []).append(row.image_id)
        return [
            document_from_parts(_mapping(by_id[document_id]), images.get(document_id, []))
            for document_id in document_ids
        ]


class PostgresContextAssemblyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, value: ContextAssemblyTrace) -> ContextAssemblyTrace:
        rows = context_to_rows(value)
        await _insert_idempotent(
            self.session,
            ContextAssemblyModel,
            rows["context"][0],
            value.context_assembly_id,
        )
        for row in rows["units"]:
            await _insert_idempotent(
                self.session,
                ContextAssemblyMemoryUnitModel,
                row,
                (row["context_assembly_id"], row["memory_id"]),
            )
        for row in rows["images"]:
            await _insert_idempotent(
                self.session,
                ContextAssemblyImageModel,
                row,
                (row["context_assembly_id"], row["image_id"], row["role"]),
            )
        for row in rows["candidates"]:
            await _insert_idempotent(
                self.session,
                ContextAssemblyRetrievalCandidateModel,
                row,
                (row["context_assembly_id"], row["retrieval_trace_id"]),
            )
        return value

    async def get(self, context_assembly_id: str) -> ContextAssemblyTrace:
        parent = await self.session.get(ContextAssemblyModel, context_assembly_id)
        if parent is None:
            raise KeyError(f"context assembly does not exist: {context_assembly_id}")
        units = list(
            (
                await self.session.execute(
                    select(ContextAssemblyMemoryUnitModel)
                    .where(
                        ContextAssemblyMemoryUnitModel.context_assembly_id
                        == context_assembly_id
                    )
                    .order_by(ContextAssemblyMemoryUnitModel.ordinal)
                )
            ).scalars()
        )
        images = list(
            (
                await self.session.execute(
                    select(ContextAssemblyImageModel)
                    .where(ContextAssemblyImageModel.context_assembly_id == context_assembly_id)
                    .order_by(ContextAssemblyImageModel.role, ContextAssemblyImageModel.ordinal)
                )
            ).scalars()
        )
        candidates = list(
            (
                await self.session.execute(
                    select(ContextAssemblyRetrievalCandidateModel)
                    .where(
                        ContextAssemblyRetrievalCandidateModel.context_assembly_id
                        == context_assembly_id
                    )
                    .order_by(ContextAssemblyRetrievalCandidateModel.rank)
                )
            ).scalars()
        )
        return context_from_parts(
            _mapping(parent),
            [_mapping(item) for item in units],
            [_mapping(item) for item in images],
            [_mapping(item) for item in candidates],
        )


async def verify_persisted_corpus(
    engine: AsyncEngine,
    corpus_id: str,
) -> dict[str, object]:
    """Return non-sensitive corpus state/count verification."""

    async with engine.connect() as connection:
        row = (
            await connection.execute(
                select(_table(RetrievalCorpusModel)).where(
                    _table(RetrievalCorpusModel).c.corpus_id == corpus_id
                )
            )
        ).mappings().one_or_none()
        if row is None:
            raise KeyError(f"corpus does not exist: {corpus_id}")
        manifest = BaselineCorpusManifest.model_validate(row["manifest_json"])
        await BaselineCorpusImporter._verify_counts(connection, manifest)
        return {
            "corpus_id": corpus_id,
            "snapshot_id": row["snapshot_id"],
            "status": row["status"],
            "window_count": row["window_count"],
            "document_count": row["document_count"],
            "content_hash": row["content_hash"],
            "valid": row["status"] == "ready",
        }
