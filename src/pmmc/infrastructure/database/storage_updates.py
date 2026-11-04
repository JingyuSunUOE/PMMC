"""Owner-only persistence of verified object storage locations."""

from __future__ import annotations

from sqlalchemy import bindparam, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from pmmc.infrastructure.database.errors import SnapshotImportError
from pmmc.infrastructure.database.models import AssetModel, DatasetSnapshotModel
from pmmc.infrastructure.storage_sync import AssetSyncResult, SnapshotArchiveResult
from pmmc.schemas.base import utc_now


async def persist_snapshot_archive(
    engine: AsyncEngine,
    result: SnapshotArchiveResult,
) -> None:
    runtime_manifest = result.runtime_objects["manifest.json"]
    evaluation_gold = result.evaluation_objects["query_gold.jsonl"]
    validation = result.runtime_objects["validation_report.json"]
    async with engine.begin() as connection:
        updated = await connection.execute(
            update(DatasetSnapshotModel)
            .where(
                DatasetSnapshotModel.snapshot_id == result.snapshot_id,
                DatasetSnapshotModel.import_state == "ready",
            )
            .values(
                runtime_archive_uri=runtime_manifest.uri.rsplit("/", 1)[0],
                evaluation_archive_uri=evaluation_gold.uri.rsplit("/", 1)[0],
                validation_report_uri=validation.uri,
            )
        )
        if updated.rowcount != 1:
            raise SnapshotImportError(
                "snapshot must be imported and ready before archive metadata is persisted"
            )


async def persist_asset_sync(engine: AsyncEngine, result: AssetSyncResult) -> None:
    expected_ids = {item.asset_id for item in result.synced}
    if len(expected_ids) != len(result.synced):
        raise SnapshotImportError("asset synchronization result contains duplicate IDs")
    statement = (
        update(AssetModel)
        .where(
            AssetModel.snapshot_id == bindparam("p_snapshot_id"),
            AssetModel.asset_id == bindparam("p_asset_id"),
        )
        .values(
            storage_uri=bindparam("p_storage_uri"),
            sha256=bindparam("p_sha256"),
            size_bytes=bindparam("p_size_bytes"),
            storage_synced_at=bindparam("p_synced_at"),
        )
    )
    rows = [
        {
            "p_snapshot_id": result.snapshot_id,
            "p_asset_id": item.asset_id,
            "p_storage_uri": item.uri,
            "p_sha256": item.sha256,
            "p_size_bytes": item.size_bytes,
            "p_synced_at": utc_now(),
        }
        for item in result.synced
    ]
    async with engine.begin() as connection:
        snapshot_state = await connection.scalar(
            select(DatasetSnapshotModel.import_state).where(
                DatasetSnapshotModel.snapshot_id == result.snapshot_id
            )
        )
        if snapshot_state != "ready":
            raise SnapshotImportError(
                "snapshot must be imported and ready before asset metadata is persisted"
            )
        if not rows:
            return
        persisted_ids = set(
            (
                await connection.scalars(
                    select(AssetModel.asset_id).where(
                        AssetModel.snapshot_id == result.snapshot_id,
                        AssetModel.asset_id.in_(expected_ids),
                    )
                )
            ).all()
        )
        if persisted_ids != expected_ids:
            raise SnapshotImportError("not all synchronized assets exist in the ready snapshot")
        updated = await connection.execute(statement, rows)
        if updated.rowcount not in {-1, len(rows)}:
            raise SnapshotImportError("not all synchronized assets exist in Postgres")
