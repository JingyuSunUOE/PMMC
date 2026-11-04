"""Runtime-safe asynchronous Postgres repositories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Sequence

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from pmmc.benchmarks.selection import select_queries
from pmmc.infrastructure.database.image_asset_view import RuntimeImageAssetView
from pmmc.infrastructure.database.mappers import (
    benchmark_query_from_row,
    image_from_row,
    manifest_from_row,
    memory_from_row,
    session_from_row,
)
from pmmc.infrastructure.database.models import (
    AssetModel,
    BenchmarkQueryModel,
    BenchmarkSessionModel,
    DatasetSnapshotModel,
    ImageModel,
    MemoryUnitImageModel,
    MemoryUnitModel,
    QueryImageModel,
)
from pmmc.schemas import (
    BenchmarkQuery,
    BenchmarkSelection,
    DatasetSnapshotManifest,
    ImageRef,
    MemoryUnit,
    QueryInput,
    SessionRecord,
)


def _merged_image_field(
    image_id: str,
    field_name: str,
    image_value: object | None,
    asset_value: object | None,
) -> object | None:
    if image_value is not None and asset_value is not None and image_value != asset_value:
        raise ValueError(f"conflicting {field_name} for image_id={image_id}")
    return image_value if image_value is not None else asset_value


def _image_mapping(
    value: ImageModel,
    asset: AssetModel | RuntimeImageAssetView | None = None,
) -> dict[str, object]:
    return {
        "image_id": value.image_id,
        "asset_id": value.asset_id,
        "local_path": _merged_image_field(
            value.image_id, "local_path", value.local_path, asset.local_path if asset else None
        ),
        "remote_url": _merged_image_field(
            value.image_id, "remote_url", value.remote_url, asset.remote_url if asset else None
        ),
        "storage_uri": _merged_image_field(
            value.image_id,
            "storage_uri",
            value.storage_uri,
            asset.storage_uri if asset else None,
        ),
        "caption": value.caption,
        "sha256": _merged_image_field(
            value.image_id, "sha256", value.sha256, asset.sha256 if asset else None
        ),
        "mime_type": _merged_image_field(
            value.image_id, "mime_type", value.mime_type, asset.mime_type if asset else None
        ),
        "width": _merged_image_field(
            value.image_id, "width", value.width, asset.width if asset else None
        ),
        "height": _merged_image_field(
            value.image_id, "height", value.height, asset.height if asset else None
        ),
        "metadata": value.metadata_json,
    }


def _memory_mapping(value: MemoryUnitModel) -> dict[str, object]:
    return {
        "memory_id": value.memory_id,
        "dataset": value.dataset,
        "conversation_id": value.conversation_id,
        "session_id": value.session_id,
        "turn_id": value.turn_id,
        "timestamp_order": value.timestamp_order,
        "speaker": value.speaker,
        "text": value.text,
        "caption": value.caption,
        "entity_mentions": value.entity_mentions,
        "modalities": value.modalities,
        "available_after_session": value.available_after_session,
        "token_count": value.token_count,
        "metadata": value.metadata_json,
    }


def _query_mapping(value: BenchmarkQueryModel) -> dict[str, object]:
    return {
        "query_id": value.query_id,
        "dataset": value.dataset,
        "conversation_id": value.conversation_id,
        "raw_query": value.raw_query,
        "question_type": value.question_type,
        "query_subtype": value.query_subtype,
        "available_after_session": value.available_after_session,
        "eligible": value.eligible,
        "evaluation_policy_version": value.evaluation_policy_version,
        "exclusion_reason": value.exclusion_reason,
        "source_file": value.source_file,
        "source_record_index": value.source_record_index,
        "source_query_id": value.source_query_id,
        "scenario_id": value.scenario_id,
        "qa_index": value.qa_index,
        "source_session_id": value.source_session_id,
        "source_round_id": value.source_round_id,
        "source_turn_index": value.source_turn_index,
        "source_category": value.source_category,
        "source_subtype": value.source_subtype,
        "query_metadata": value.query_metadata,
        "source_metadata": value.source_metadata,
        "eligibility_metadata": value.eligibility_metadata,
        "metadata": value.metadata_json,
    }


class PostgresSnapshotRepository:
    """Read immutable snapshot manifests from the runtime schema."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def get_manifest(self, snapshot_id: str) -> DatasetSnapshotManifest:
        async with self.sessions() as session:
            value = await session.scalar(
                select(DatasetSnapshotModel).where(
                    DatasetSnapshotModel.snapshot_id == snapshot_id,
                    DatasetSnapshotModel.import_state == "ready",
                )
            )
            if value is None:
                raise KeyError(f"ready snapshot does not exist: {snapshot_id}")
            return manifest_from_row({"manifest_json": value.manifest_json})


class PostgresImageRepository:
    """Materialize canonical image metadata with object-store asset fields merged in."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def get_many(self, snapshot_id: str, image_ids: list[str]) -> list[ImageRef]:
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("image_ids cannot contain duplicates")
        if not image_ids:
            return []
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(ImageModel, RuntimeImageAssetView)
                    .outerjoin(
                        RuntimeImageAssetView,
                        and_(
                            RuntimeImageAssetView.snapshot_id == ImageModel.snapshot_id,
                            RuntimeImageAssetView.asset_id == ImageModel.asset_id,
                        ),
                    )
                    .where(
                        ImageModel.snapshot_id == snapshot_id,
                        ImageModel.image_id.in_(image_ids),
                    )
                )
            ).all()
        mapped = {
            image.image_id: image_from_row(_image_mapping(image, asset))
            for image, asset in rows
        }
        missing = [image_id for image_id in image_ids if image_id not in mapped]
        if missing:
            raise KeyError(f"canonical image does not exist: {missing[0]}")
        return [mapped[image_id] for image_id in image_ids]


class PostgresMemoryRepository:
    """Read memory using the prospective available-after-session boundary."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def list_snapshot(
        self,
        snapshot_id: str,
        *,
        conversation_ids: set[str] | None = None,
    ) -> list[MemoryUnit]:
        """Materialize canonical memories once for deterministic write-time compilation."""

        async with self.sessions() as session:
            available = aliased(BenchmarkSessionModel)
            statement = (
                select(MemoryUnitModel)
                .join(
                    available,
                    and_(
                        available.snapshot_id == MemoryUnitModel.snapshot_id,
                        available.conversation_id == MemoryUnitModel.conversation_id,
                        available.session_id == MemoryUnitModel.available_after_session,
                    ),
                )
                .where(MemoryUnitModel.snapshot_id == snapshot_id)
                .order_by(
                    MemoryUnitModel.conversation_id,
                    available.session_order,
                    MemoryUnitModel.timestamp_order,
                    MemoryUnitModel.memory_id,
                )
            )
            if conversation_ids is not None:
                if not conversation_ids:
                    return []
                statement = statement.where(
                    MemoryUnitModel.conversation_id.in_(conversation_ids)
                )
            memories = list((await session.scalars(statement)).all())
            image_map = await self._memory_images(
                session, snapshot_id, [item.memory_id for item in memories]
            )
        return [
            memory_from_row(_memory_mapping(memory), image_map.get(memory.memory_id, []))
            for memory in memories
        ]

    async def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[MemoryUnit]:
        async with self.sessions() as session:
            cutoff_order = await session.scalar(
                select(BenchmarkSessionModel.session_order).where(
                    BenchmarkSessionModel.snapshot_id == snapshot_id,
                    BenchmarkSessionModel.conversation_id == conversation_id,
                    BenchmarkSessionModel.session_id == cutoff_session_id,
                )
            )
            if cutoff_order is None:
                raise KeyError(
                    f"cutoff session does not exist in conversation: {cutoff_session_id}"
                )
            available = aliased(BenchmarkSessionModel)
            memories = list(
                (
                    await session.scalars(
                        select(MemoryUnitModel)
                        .join(
                            available,
                            and_(
                                available.snapshot_id == MemoryUnitModel.snapshot_id,
                                available.conversation_id == MemoryUnitModel.conversation_id,
                                available.session_id == MemoryUnitModel.available_after_session,
                            ),
                        )
                        .where(
                            MemoryUnitModel.snapshot_id == snapshot_id,
                            MemoryUnitModel.conversation_id == conversation_id,
                            available.session_order <= cutoff_order,
                        )
                        .order_by(MemoryUnitModel.timestamp_order)
                    )
                ).all()
            )
            image_map = await self._memory_images(
                session, snapshot_id, [item.memory_id for item in memories]
            )
        for memory in memories:
            yield memory_from_row(_memory_mapping(memory), image_map.get(memory.memory_id, []))

    @staticmethod
    async def _memory_images(
        session: AsyncSession,
        snapshot_id: str,
        memory_ids: list[str],
    ) -> dict[str, list[ImageRef]]:
        if not memory_ids:
            return {}
        result = await session.execute(
            select(MemoryUnitImageModel.memory_id, ImageModel, RuntimeImageAssetView)
            .join(
                ImageModel,
                and_(
                    ImageModel.snapshot_id == MemoryUnitImageModel.snapshot_id,
                    ImageModel.image_id == MemoryUnitImageModel.image_id,
                ),
            )
            .outerjoin(
                RuntimeImageAssetView,
                and_(
                    RuntimeImageAssetView.snapshot_id == ImageModel.snapshot_id,
                    RuntimeImageAssetView.asset_id == ImageModel.asset_id,
                ),
            )
            .where(
                MemoryUnitImageModel.snapshot_id == snapshot_id,
                MemoryUnitImageModel.memory_id.in_(memory_ids),
            )
            .order_by(MemoryUnitImageModel.memory_id, MemoryUnitImageModel.ordinal)
        )
        mapped: dict[str, list[ImageRef]] = defaultdict(list)
        for memory_id, image, asset in result.all():
            mapped[memory_id].append(image_from_row(_image_mapping(image, asset)))
        return dict(mapped)


class PostgresSessionRepository:
    """Read canonical sessions up to the prospective query cutoff."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def list_snapshot(
        self,
        snapshot_id: str,
        *,
        conversation_ids: set[str] | None = None,
    ) -> list[SessionRecord]:
        """Return the stable session denominator used by the Questioner unit builder."""

        async with self.sessions() as session:
            statement = (
                select(BenchmarkSessionModel)
                .where(BenchmarkSessionModel.snapshot_id == snapshot_id)
                .order_by(
                    BenchmarkSessionModel.conversation_id,
                    BenchmarkSessionModel.session_order,
                    BenchmarkSessionModel.session_id,
                )
            )
            if conversation_ids is not None:
                if not conversation_ids:
                    return []
                statement = statement.where(
                    BenchmarkSessionModel.conversation_id.in_(conversation_ids)
                )
            values = list((await session.scalars(statement)).all())
        return [
            session_from_row(
                {
                    "session_id": value.session_id,
                    "dataset": value.dataset,
                    "conversation_id": value.conversation_id,
                    "session_order": value.session_order,
                    "source_session_id": value.source_session_id,
                    "source_date_raw": value.source_date_raw,
                    "source_date_iso": value.source_date_iso,
                    "is_synthetic": value.is_synthetic,
                    "metadata": value.metadata_json,
                }
            )
            for value in values
        ]

    async def iter_visible(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> AsyncIterator[SessionRecord]:
        async with self.sessions() as session:
            cutoff_order = await session.scalar(
                select(BenchmarkSessionModel.session_order).where(
                    BenchmarkSessionModel.snapshot_id == snapshot_id,
                    BenchmarkSessionModel.conversation_id == conversation_id,
                    BenchmarkSessionModel.session_id == cutoff_session_id,
                )
            )
            if cutoff_order is None:
                raise KeyError(
                    f"cutoff session does not exist in conversation: {cutoff_session_id}"
                )
            values = list(
                (
                    await session.scalars(
                        select(BenchmarkSessionModel)
                        .where(
                            BenchmarkSessionModel.snapshot_id == snapshot_id,
                            BenchmarkSessionModel.conversation_id == conversation_id,
                            BenchmarkSessionModel.session_order <= cutoff_order,
                        )
                        .order_by(BenchmarkSessionModel.session_order)
                    )
                ).all()
            )
        for value in values:
            yield session_from_row(
                {
                    "session_id": value.session_id,
                    "dataset": value.dataset,
                    "conversation_id": value.conversation_id,
                    "session_order": value.session_order,
                    "source_session_id": value.source_session_id,
                    "source_date_raw": value.source_date_raw,
                    "source_date_iso": value.source_date_iso,
                    "is_synthetic": value.is_synthetic,
                    "metadata": value.metadata_json,
                }
            )


class PostgresRuntimeQueryRepository:
    """Return only runtime-safe query records and reuse deterministic selection."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def get(self, snapshot_id: str, query_id: str) -> QueryInput:
        """Load one runtime-safe query without materializing evaluation gold."""

        async with self.sessions() as session:
            query = await session.scalar(
                select(BenchmarkQueryModel).where(
                    BenchmarkQueryModel.snapshot_id == snapshot_id,
                    BenchmarkQueryModel.query_id == query_id,
                )
            )
            if query is None:
                raise KeyError(f"runtime query does not exist: {query_id}")
            image_map = await self._query_images(session, snapshot_id, query_ids=[query_id])
        return benchmark_query_from_row(
            _query_mapping(query), image_map.get(query.query_id, [])
        ).query

    async def list_ids(
        self,
        snapshot_id: str,
        query_ids: Sequence[str],
    ) -> list[QueryInput]:
        """Load a stable runtime-only denominator without reading evaluation gold."""

        ordered_ids = list(query_ids)
        if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("runtime query IDs must be non-empty and unique")
        async with self.sessions() as session:
            values = list(
                (
                    await session.scalars(
                        select(BenchmarkQueryModel).where(
                            BenchmarkQueryModel.snapshot_id == snapshot_id,
                            BenchmarkQueryModel.query_id.in_(ordered_ids),
                        )
                    )
                ).all()
            )
            by_id = {value.query_id: value for value in values}
            if set(by_id) != set(ordered_ids):
                missing = next(value for value in ordered_ids if value not in by_id)
                raise KeyError(f"runtime query does not exist: {missing}")
            image_map = await self._query_images(
                session,
                snapshot_id,
                query_ids=ordered_ids,
            )
        return [
            benchmark_query_from_row(
                _query_mapping(by_id[query_id]), image_map.get(query_id, [])
            ).query
            for query_id in ordered_ids
        ]

    async def list_snapshot(self, snapshot_id: str) -> list[BenchmarkQuery]:
        async with self.sessions() as session:
            queries = list(
                (
                    await session.scalars(
                        select(BenchmarkQueryModel)
                        .where(BenchmarkQueryModel.snapshot_id == snapshot_id)
                        .order_by(BenchmarkQueryModel.query_id)
                    )
                ).all()
            )
            image_map = await self._query_images(session, snapshot_id)
        return [
            benchmark_query_from_row(
                _query_mapping(query), image_map.get(query.query_id, [])
            )
            for query in queries
        ]

    async def select(
        self,
        snapshot_id: str,
        selection: BenchmarkSelection,
    ) -> list[BenchmarkQuery]:
        return select_queries(await self.list_snapshot(snapshot_id), selection)

    @staticmethod
    async def _query_images(
        session: AsyncSession,
        snapshot_id: str,
        *,
        query_ids: Sequence[str] | None = None,
    ) -> dict[str, list[ImageRef]]:
        statement = (
            select(QueryImageModel.query_id, ImageModel, RuntimeImageAssetView)
            .join(
                ImageModel,
                and_(
                    ImageModel.snapshot_id == QueryImageModel.snapshot_id,
                    ImageModel.image_id == QueryImageModel.image_id,
                ),
            )
            .outerjoin(
                RuntimeImageAssetView,
                and_(
                    RuntimeImageAssetView.snapshot_id == ImageModel.snapshot_id,
                    RuntimeImageAssetView.asset_id == ImageModel.asset_id,
                ),
            )
            .where(QueryImageModel.snapshot_id == snapshot_id)
            .order_by(QueryImageModel.query_id, QueryImageModel.ordinal)
        )
        if query_ids is not None:
            statement = statement.where(QueryImageModel.query_id.in_(query_ids))
        result = await session.execute(statement)
        mapped: dict[str, list[ImageRef]] = defaultdict(list)
        for query_id, image, asset in result.all():
            mapped[query_id].append(image_from_row(_image_mapping(image, asset)))
        return dict(mapped)
