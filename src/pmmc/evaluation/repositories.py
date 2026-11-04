"""Evaluation-only gold repositories.

This module must not be imported by runtime answer or retrieval packages.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.benchmarks.repositories import GoldStore
from pmmc.benchmarks.snapshot import find_snapshot
from pmmc.infrastructure.database.mappers import query_gold_from_parts
from pmmc.infrastructure.database.models import (
    QueryGoldEvidenceIdModel,
    QueryGoldImageIdModel,
    QueryGoldMemoryIdModel,
    QueryGoldModel,
    QueryGoldSessionIdModel,
)
from pmmc.schemas import QueryGold


class JsonlGoldRepository:
    def __init__(self, snapshots_root: Path) -> None:
        self.snapshots_root = snapshots_root

    async def get(self, snapshot_id: str, query_id: str) -> QueryGold:
        path = find_snapshot(self.snapshots_root, snapshot_id)
        return await asyncio.to_thread(GoldStore(path).get, query_id)


class PostgresGoldRepository:
    """Read gold data using an evaluator-scoped session factory only."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def get(self, snapshot_id: str, query_id: str) -> QueryGold:
        async with self.sessions() as session:
            parent = await session.scalar(
                select(QueryGoldModel).where(
                    QueryGoldModel.snapshot_id == snapshot_id,
                    QueryGoldModel.query_id == query_id,
                )
            )
            if parent is None:
                raise KeyError(f"gold query record does not exist: {query_id}")

            evidence_ids = list(
                (
                    await session.scalars(
                        select(QueryGoldEvidenceIdModel.evidence_id)
                        .where(
                            QueryGoldEvidenceIdModel.snapshot_id == snapshot_id,
                            QueryGoldEvidenceIdModel.query_id == query_id,
                        )
                        .order_by(QueryGoldEvidenceIdModel.ordinal)
                    )
                ).all()
            )
            memory_ids = list(
                (
                    await session.scalars(
                        select(QueryGoldMemoryIdModel.memory_id)
                        .where(
                            QueryGoldMemoryIdModel.snapshot_id == snapshot_id,
                            QueryGoldMemoryIdModel.query_id == query_id,
                        )
                        .order_by(QueryGoldMemoryIdModel.ordinal)
                    )
                ).all()
            )
            session_ids = list(
                (
                    await session.scalars(
                        select(QueryGoldSessionIdModel.session_id)
                        .where(
                            QueryGoldSessionIdModel.snapshot_id == snapshot_id,
                            QueryGoldSessionIdModel.query_id == query_id,
                        )
                        .order_by(QueryGoldSessionIdModel.ordinal)
                    )
                ).all()
            )
            image_ids = list(
                (
                    await session.scalars(
                        select(QueryGoldImageIdModel.image_id)
                        .where(
                            QueryGoldImageIdModel.snapshot_id == snapshot_id,
                            QueryGoldImageIdModel.query_id == query_id,
                        )
                        .order_by(QueryGoldImageIdModel.ordinal)
                    )
                ).all()
            )

            return query_gold_from_parts(
                {
                    "answer": parent.answer,
                    "evidence_metadata": parent.evidence_metadata,
                    "gold_metadata": parent.gold_metadata,
                },
                evidence_ids=evidence_ids,
                memory_ids=memory_ids,
                session_ids=session_ids,
                image_ids=image_ids,
            )
