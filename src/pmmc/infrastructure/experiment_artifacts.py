"""Scope-safe orchestration for immutable experiment artifact uploads."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.infrastructure.database.experiment_models import (
    EvaluationArtifactModel,
    RuntimeArtifactModel,
)
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.object_store import ObjectStore, run_artifact_object_key
from pmmc.schemas import (
    ArtifactState,
    AssetType,
    DataScope,
    ExperimentArtifact,
)


class ExperimentArtifactService:
    """Persist artifact metadata and bytes using a recoverable state machine."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        store: ObjectStore,
        scope: DataScope,
        artifact_prefix: str = "runs",
    ) -> None:
        self.sessions = sessions
        self.store = store
        self.scope = scope
        self.artifact_prefix = artifact_prefix

    async def put_bytes(
        self,
        *,
        artifact_id: str,
        run_id: str,
        evaluation_run_id: str | None = None,
        semantic_run_id: str | None = None,
        artifact_type: AssetType,
        data: bytes,
        mime_type: str,
        extension: str | None = None,
        snapshot_id: str | None = None,
        query_id: str | None = None,
    ) -> ExperimentArtifact:
        digest = hashlib.sha256(data).hexdigest()
        if self.scope is DataScope.EVALUATION and (
            (evaluation_run_id is None) == (semantic_run_id is None)
        ):
            raise ValueError(
                "evaluation artifacts require exactly one evaluation parent"
            )
        if self.scope is DataScope.RUNTIME and (
            evaluation_run_id is not None or semantic_run_id is not None
        ):
            raise ValueError("runtime artifacts cannot reference evaluation parents")
        object_namespace = evaluation_run_id or semantic_run_id or run_id
        key = run_artifact_object_key(
            object_namespace,
            artifact_type.value,
            digest,
            extension=extension,
            prefix=self.artifact_prefix,
        )
        pending = ExperimentArtifact(
            artifact_id=artifact_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            semantic_run_id=semantic_run_id,
            snapshot_id=snapshot_id,
            query_id=query_id,
            artifact_type=artifact_type,
            data_scope=self.scope,
            state=ArtifactState.PENDING,
            object_key=key,
            sha256=digest,
            mime_type=mime_type,
        )
        async with ExperimentUnitOfWork(self.sessions, self.scope) as uow:
            existing = await uow.artifacts.get(artifact_id)
            if existing is None:
                await uow.artifacts.add(pending)
            else:
                self._validate_identity(existing, pending)
                if existing.state is ArtifactState.READY:
                    remote = await self.store.head(existing.object_key)
                    if remote is None or remote.sha256 != digest:
                        raise RuntimeError("ready artifact is missing or has an invalid checksum")
                    return existing
                if existing.state is ArtifactState.FAILED:
                    await uow.artifacts.retry_failed(artifact_id)
        try:
            uploaded = await self.store.put_bytes(
                data,
                key,
                expected_sha256=digest,
                content_type=mime_type,
            )
        except Exception as exc:
            async with ExperimentUnitOfWork(self.sessions, self.scope) as uow:
                await uow.artifacts.mark_failed(artifact_id, type(exc).__name__)
            raise
        async with ExperimentUnitOfWork(self.sessions, self.scope) as uow:
            return await uow.artifacts.mark_ready(
                artifact_id,
                storage_uri=uploaded.uri,
                size_bytes=uploaded.size_bytes,
            )

    @staticmethod
    def _validate_identity(existing: ExperimentArtifact, incoming: ExperimentArtifact) -> None:
        for field_name in (
            "run_id",
            "evaluation_run_id",
            "semantic_run_id",
            "snapshot_id",
            "query_id",
            "artifact_type",
            "data_scope",
            "object_key",
            "sha256",
            "mime_type",
        ):
            if getattr(existing, field_name) != getattr(incoming, field_name):
                raise ValueError("artifact ID already belongs to different content")


async def verify_run_artifacts(
    session: AsyncSession,
    store: ObjectStore,
    run_id: str,
    scope: DataScope,
) -> dict[str, Any]:
    """Verify persisted ready artifacts against one scope-specific object store."""

    model: Any = RuntimeArtifactModel if scope is DataScope.RUNTIME else EvaluationArtifactModel
    rows: Sequence[RuntimeArtifactModel | EvaluationArtifactModel] = (
        (
            await session.execute(
                select(model).where(model.run_id == run_id).order_by(model.artifact_id)
            )
        )
        .scalars()
        .all()
    )
    failures: list[str] = []
    for row in rows:
        if row.state != ArtifactState.READY.value:
            failures.append(f"{row.artifact_id}:state={row.state}")
            continue
        remote = await store.head(row.object_key)
        if remote is None:
            failures.append(f"{row.artifact_id}:missing")
            continue
        if remote.sha256 != row.sha256 or remote.size_bytes != row.size_bytes:
            failures.append(f"{row.artifact_id}:checksum_or_size")
        if remote.uri != row.storage_uri:
            failures.append(f"{row.artifact_id}:uri")
    return {
        "run_id": run_id,
        "scope": scope.value,
        "artifact_count": len(rows),
        "valid": not failures,
        "failures": failures,
    }
