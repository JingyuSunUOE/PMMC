"""Consistent private checkpoints for self-contained research launches."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from pmmc.infrastructure.database.constants import (
    EVALUATION_SCHEMA,
    RUNTIME_SCHEMA,
    DatabaseRole,
)
from pmmc.infrastructure.database.engine import create_engine, database_url_for_role
from pmmc.infrastructure.database.experiment_models import (
    EvaluationArtifactModel,
    RuntimeArtifactModel,
)
from pmmc.infrastructure.object_store import ObjectStore, create_admin_object_stores
from pmmc.infrastructure.settings import Settings
from pmmc.schemas import EXECUTION_CACHE_SCHEMA_VERSION
from pmmc.schemas.base import BaseSchema, JsonDict, NonEmptyStr, NonNegativeInt, utc_now

_CHECKPOINT_PROGRESS_INTERVAL_SECONDS = 30.0
_CHECKPOINT_RESTORE_OBJECT_CONCURRENCY = 16
_FORWARD_COMPATIBLE_DATA_ONLY_REVISIONS = {
    ("20260725_0018", "20260726_0019"),
    ("20260727_0020", "20260727_0021"),
    ("20260727_0021", "20260729_0022"),
    ("20260729_0022", "20260730_0023"),
    ("20260802_0024", "20260803_0025"),
    ("20260803_0025", "20260907_0026"),
    ("20260907_0026", "20260907_0027"),
    ("20260907_0027", "20260907_0028"),
    ("20260907_0028", "20260911_0029"),
}


def checkpoint_revision_restore_path(
    checkpoint_revision: str,
    database_revision: str,
) -> tuple[str, ...] | None:
    """Return an audited forward-only data restore path, if one exists."""

    if checkpoint_revision == database_revision:
        return (checkpoint_revision,)
    frontier: list[tuple[str, tuple[str, ...]]] = [
        (checkpoint_revision, (checkpoint_revision,))
    ]
    visited = {checkpoint_revision}
    while frontier:
        current, path = frontier.pop(0)
        destinations = sorted(
            target
            for source, target in _FORWARD_COMPATIBLE_DATA_ONLY_REVISIONS
            if source == current
        )
        for target in destinations:
            if target in visited:
                continue
            candidate = (*path, target)
            if target == database_revision:
                return candidate
            visited.add(target)
            frontier.append((target, candidate))
    return None


def checkpoint_revision_is_restore_compatible(
    checkpoint_revision: str,
    database_revision: str,
) -> bool:
    """Allow only paths composed of audited DDL-only forward migrations."""

    return (
        checkpoint_revision_restore_path(checkpoint_revision, database_revision)
        is not None
    )


class CheckpointCancelled(RuntimeError):
    """Raised when an obsolete periodic checkpoint is cancelled safely."""


class CheckpointIdentityMismatch(ValueError):
    """Expose only fixed identity field names, never their private values."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.mismatch_keys = keys
        super().__init__("checkpoint identity differs from the current launch: " + ",".join(keys))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value.model_dump(mode="json") if isinstance(value, BaseSchema) else value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _checkpoint_log(milestone: str, state: str, **values: object) -> None:
    fields = [f"stage={milestone}", f"state={state}"]
    fields.extend(f"{key}={value}" for key, value in sorted(values.items()))
    print("CHECKPOINT " + " ".join(fields), flush=True)


def _checkpoint_restore_log(phase: str, state: str, **values: object) -> None:
    payload = {"phase": phase, "state": state, **values}
    print(
        "CHECKPOINT_RESTORE_PROGRESS "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


class CheckpointArtifactRef(BaseSchema):
    scope: NonEmptyStr
    artifact_id: NonEmptyStr
    object_key: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    size_bytes: NonNegativeInt
    # Historical manifests materialize every artifact below their own directory.
    # New milestone checkpoints may reference an earlier retained milestone in the
    # same launch, avoiding tens of thousands of redundant NFS hard-link writes.
    storage_sequence: NonNegativeInt | None = None


class ExperimentCheckpointManifest(BaseSchema):
    execution_cache_schema_version: NonEmptyStr = EXECUTION_CACHE_SCHEMA_VERSION
    launch_id: NonEmptyStr
    sequence: NonNegativeInt
    milestone: NonEmptyStr
    identity: JsonDict
    alembic_revision: NonEmptyStr
    postgres_dump_sha256: NonEmptyStr
    postgres_dump_size_bytes: NonNegativeInt
    stage_results_sha256: NonEmptyStr | None = None
    stage_results_size_bytes: NonNegativeInt | None = None
    launch_state_sha256: NonEmptyStr | None = None
    launch_state_size_bytes: NonNegativeInt | None = None
    artifacts: list[CheckpointArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
    content_hash: NonEmptyStr


class _CheckpointCandidateMetadata(BaseModel):
    """Small restore header that ignores the potentially large artifact list."""

    model_config = ConfigDict(extra="ignore")

    launch_id: NonEmptyStr
    sequence: NonNegativeInt
    milestone: NonEmptyStr
    identity: JsonDict
    alembic_revision: NonEmptyStr
    postgres_dump_size_bytes: NonNegativeInt
    launch_state_sha256: NonEmptyStr | None = None
    launch_state_size_bytes: NonNegativeInt | None = None
    created_at: str


@dataclass(frozen=True)
class _ArtifactRow:
    scope: str
    artifact_id: str
    object_key: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _ReusableArtifact:
    path: Path
    storage_sequence: int
    durable: bool = True


class ExperimentCheckpointService:
    """Create and restore database/object-store checkpoints without credentials."""

    def __init__(
        self,
        settings: Settings,
        *,
        launch_id: str,
        root: Path,
        identity: dict[str, object],
        retain_count: int = 2,
        retain_milestones: bool = False,
        backup_stores_factory: Callable[[], tuple[ObjectStore, ObjectStore]] | None = None,
    ) -> None:
        if retain_count <= 0:
            raise ValueError("checkpoint retain_count must be positive")
        self.settings = settings
        self.launch_id = launch_id
        self.root = root / launch_id
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        self.retain_count = retain_count
        self.retain_milestones = retain_milestones
        # Backup only reads objects. Restore keeps its separate admin requirement.
        self._backup_stores_factory = backup_stores_factory

    def latest(self) -> Path | None:
        candidates = self.directories()
        return candidates[-1] if candidates else None

    def directories(self) -> list[Path]:
        """Return verified-candidate checkpoint directories in sequence order."""

        if not self.root.is_dir():
            return []
        return sorted(
            [
            path.parent
            for path in self.root.glob("*/manifest.json")
            if path.parent.name.isdigit()
            ],
            key=lambda path: int(path.name),
        )

    def verify(
        self,
        directory: Path,
        *,
        require_identity: bool = True,
        verify_artifact_content: bool = True,
        verify_inherited_artifacts: bool = True,
    ) -> ExperimentCheckpointManifest:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("checkpoint manifest is missing")
        manifest = ExperimentCheckpointManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        )
        self._validate_candidate_identity(manifest, require_identity=require_identity)
        if not self._manifest_hash_is_valid(manifest):
            raise ValueError("checkpoint manifest content hash is invalid")
        self._verify_optional_payload(
            directory,
            "launch_state.json",
            manifest.launch_state_sha256,
            manifest.launch_state_size_bytes,
        )
        dump = directory / "postgres.dump"
        if (
            not dump.is_file()
            or dump.stat().st_size != manifest.postgres_dump_size_bytes
            or _sha256(dump) != manifest.postgres_dump_sha256
        ):
            raise ValueError("checkpoint Postgres dump failed verification")
        checked_artifacts: set[tuple[Path, int, str]] = set()
        for artifact in manifest.artifacts:
            storage_sequence = artifact.storage_sequence or manifest.sequence
            if storage_sequence <= 0 or storage_sequence > manifest.sequence:
                raise ValueError("checkpoint artifact storage sequence is invalid")
            if not verify_inherited_artifacts and storage_sequence != manifest.sequence:
                continue
            path = self._artifact_path(directory, artifact)
            check_key = (path, artifact.size_bytes, artifact.sha256)
            if check_key in checked_artifacts:
                continue
            if (
                not path.is_file()
                or path.stat().st_size != artifact.size_bytes
                or (
                    verify_artifact_content
                    and _sha256(path) != artifact.sha256
                )
            ):
                raise ValueError(
                    f"checkpoint artifact failed verification: {artifact.artifact_id}"
                )
            checked_artifacts.add(check_key)
        self._verify_optional_payload(
            directory,
            "stage_results.json",
            manifest.stage_results_sha256,
            manifest.stage_results_size_bytes,
        )
        return manifest

    def inspect_candidate(
        self,
        directory: Path,
        *,
        require_identity: bool = True,
    ) -> _CheckpointCandidateMetadata:
        """Inspect restore metadata without scanning the dump or artifact inventory.

        This is only a candidate-selection check. Call :meth:`verify` before
        restoring the selected checkpoint.
        """

        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("checkpoint manifest is missing")
        manifest = _CheckpointCandidateMetadata.model_validate_json(
            manifest_path.read_bytes()
        )
        self._validate_candidate_identity(manifest, require_identity=require_identity)
        self._verify_optional_payload(
            directory,
            "launch_state.json",
            manifest.launch_state_sha256,
            manifest.launch_state_size_bytes,
        )
        return manifest

    def _validate_candidate_identity(
        self,
        manifest: ExperimentCheckpointManifest | _CheckpointCandidateMetadata,
        *,
        require_identity: bool,
    ) -> None:
        if manifest.launch_id != self.launch_id:
            raise ValueError("checkpoint launch ID does not match")
        if require_identity and manifest.identity != self.identity:
            raise ValueError("checkpoint identity differs from the current launch")

    @staticmethod
    def _verify_optional_payload(
        directory: Path,
        name: str,
        expected_hash: str | None,
        expected_size: int | None,
    ) -> None:
        if (expected_hash is None) != (expected_size is None):
            raise ValueError(f"checkpoint {name} metadata is incomplete")
        if expected_hash is None:
            return
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_hash
        ):
            raise ValueError(f"checkpoint {name} failed verification")

    async def create(
        self,
        milestone: str,
        stage_results_path: Path | None = None,
        launch_state_path: Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        started = time.monotonic()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        latest = self.latest()
        sequence = int(latest.name) + 1 if latest is not None else 1
        temporary = self.root / f".{sequence:06d}.tmp-{uuid4().hex}"
        final = self.root / f"{sequence:06d}"
        temporary.mkdir(mode=0o700)
        dump_path = temporary / "postgres.dump"
        _checkpoint_log(
            milestone,
            "starting",
            previous_sequence=(latest.name if latest is not None else "none"),
            sequence=sequence,
        )
        try:
            self._raise_if_cancelled(cancel_event)
            _checkpoint_log(milestone, "database_dump_start", sequence=sequence)
            dump_task = asyncio.create_task(
                self._consistent_dump(
                    dump_path,
                    cancel_event=cancel_event,
                )
            )
            while not dump_task.done():
                done, _ = await asyncio.wait(
                    {dump_task},
                    timeout=_CHECKPOINT_PROGRESS_INTERVAL_SECONDS,
                )
                if done:
                    break
                _checkpoint_log(
                    milestone,
                    "database_dump_progress",
                    elapsed_seconds=round(time.monotonic() - started, 1),
                    sequence=sequence,
                )
            revision, rows = await dump_task
            self._raise_if_cancelled(cancel_event)
            _checkpoint_log(
                milestone,
                "database_dump_ready",
                artifact_count=len(rows),
                dump_bytes=dump_path.stat().st_size,
                elapsed_seconds=round(time.monotonic() - started, 1),
                sequence=sequence,
            )
            references: list[CheckpointArtifactRef] = []
            reusable = self._reusable_artifacts(latest)
            incremental_reference_mode = self.retain_milestones
            completed = 0
            downloaded = 0
            downloaded_bytes = 0
            reused = 0
            reused_bytes = 0
            inherited = 0
            inherited_bytes = 0
            materialized = 0
            next_report = time.monotonic() + _CHECKPOINT_PROGRESS_INTERVAL_SECONDS
            reusable_row_count = sum(
                int((row.scope, row.sha256, row.size_bytes) in reusable)
                for row in rows
            )
            _checkpoint_log(
                milestone,
                "artifact_archive_start",
                download_concurrency=8,
                archive_mode=(
                    "incremental_reference"
                    if incremental_reference_mode
                    else "self_contained_materialization"
                ),
                artifact_count=len(rows),
                reusable_content_count=len(reusable),
                reusable_row_count=reusable_row_count,
                download_required_count=len(rows) - reusable_row_count,
                materialization_required_count=sum(
                    not (
                        incremental_reference_mode
                        and (source := reusable.get((row.scope, row.sha256, row.size_bytes)))
                        and source.durable
                    )
                    for row in rows
                ),
                sequence=sequence,
            )
            if rows:
                stores: dict[str, ObjectStore] | None = None
                content_locks: dict[tuple[str, str, int], asyncio.Lock] = {}

                async def archive_one(row: _ArtifactRow) -> None:
                    nonlocal stores, completed, downloaded, downloaded_bytes
                    nonlocal reused, reused_bytes, inherited, inherited_bytes
                    nonlocal materialized, next_report
                    key = (row.scope, row.sha256, row.size_bytes)
                    async with content_locks.setdefault(key, asyncio.Lock()):
                        self._raise_if_cancelled(cancel_event)
                        relative = Path(
                            f"{row.scope}_objects/sha256/{row.sha256[:2]}/{row.sha256}"
                        )
                        target = temporary / relative
                        source = reusable.get((row.scope, row.sha256, row.size_bytes))
                        storage_sequence = sequence
                        if source is not None and source.durable and incremental_reference_mode:
                            storage_sequence = source.storage_sequence
                            inherited += 1
                            inherited_bytes += row.size_bytes
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.parent.chmod(0o700)
                        if storage_sequence == sequence and not target.exists():
                            if source is not None:
                                if not source.durable and (
                                    source.path.stat().st_size != row.size_bytes
                                    or _sha256(source.path) != row.sha256
                                ):
                                    raise ValueError("periodic checkpoint artifact changed")
                                try:
                                    os.link(source.path, target)
                                except OSError:
                                    shutil.copyfile(source.path, target)
                                reused += 1
                                reused_bytes += row.size_bytes
                            else:
                                if stores is None:
                                    runtime_store, evaluation_store = (
                                        self._backup_stores_factory()
                                        if self._backup_stores_factory is not None
                                        else create_admin_object_stores(self.settings)
                                    )
                                    stores = {
                                        "runtime": runtime_store,
                                        "evaluation": evaluation_store,
                                    }
                                    _checkpoint_log(
                                        milestone, "artifact_reader_ready", sequence=sequence,
                                        reader=("configured_backup_reader"
                                                if self._backup_stores_factory is not None
                                                else "admin"),
                                        operation="read_only",
                                    )
                                payload = await stores[row.scope].get_bytes(row.object_key)
                                self._raise_if_cancelled(cancel_event)
                                if (
                                    len(payload) != row.size_bytes
                                    or hashlib.sha256(payload).hexdigest() != row.sha256
                                ):
                                    raise ValueError(
                                        "ready artifact changed during checkpoint: "
                                        f"{row.artifact_id}"
                                    )
                                target.write_bytes(payload)
                                downloaded += 1
                                downloaded_bytes += row.size_bytes
                            target.chmod(0o600)
                            materialized += 1
                        references.append(
                            CheckpointArtifactRef(
                                scope=row.scope,
                                artifact_id=row.artifact_id,
                                object_key=row.object_key,
                                relative_path=str(relative),
                                sha256=row.sha256,
                                size_bytes=row.size_bytes,
                                storage_sequence=storage_sequence,
                            )
                        )
                        completed += 1
                        now = time.monotonic()
                        if now >= next_report:
                            _checkpoint_log(
                                milestone,
                                "artifact_progress",
                                artifacts=f"{completed}/{len(rows)}",
                                downloaded=downloaded,
                                downloaded_bytes=downloaded_bytes,
                                elapsed_seconds=round(now - started, 1),
                                inherited=inherited,
                                inherited_bytes=inherited_bytes,
                                materialized=materialized,
                                reused=reused,
                                reused_bytes=reused_bytes,
                                sequence=sequence,
                            )
                            next_report = now + _CHECKPOINT_PROGRESS_INTERVAL_SECONDS

                # Bound live payloads and drain cancelled readers before removing temp files.
                for offset in range(0, len(rows), 8):
                    tasks = [
                        asyncio.create_task(archive_one(row))
                        for row in rows[offset:offset + 8]
                    ]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
            self._raise_if_cancelled(cancel_event)
            if stage_results_path is not None and stage_results_path.is_file():
                shutil.copyfile(stage_results_path, temporary / "stage_results.json")
                (temporary / "stage_results.json").chmod(0o600)
            if launch_state_path is not None and launch_state_path.is_file():
                shutil.copyfile(launch_state_path, temporary / "launch_state.json")
                (temporary / "launch_state.json").chmod(0o600)
            dump_path.chmod(0o600)
            manifest = ExperimentCheckpointManifest(
                launch_id=self.launch_id,
                sequence=sequence,
                milestone=milestone,
                identity=self.identity,
                alembic_revision=revision,
                postgres_dump_sha256=_sha256(dump_path),
                postgres_dump_size_bytes=dump_path.stat().st_size,
                stage_results_sha256=(
                    _sha256(temporary / "stage_results.json")
                    if (temporary / "stage_results.json").is_file()
                    else None
                ),
                stage_results_size_bytes=(
                    (temporary / "stage_results.json").stat().st_size
                    if (temporary / "stage_results.json").is_file()
                    else None
                ),
                launch_state_sha256=(
                    _sha256(temporary / "launch_state.json")
                    if (temporary / "launch_state.json").is_file()
                    else None
                ),
                launch_state_size_bytes=(
                    (temporary / "launch_state.json").stat().st_size
                    if (temporary / "launch_state.json").is_file()
                    else None
                ),
                artifacts=sorted(
                    references, key=lambda item: (item.scope, item.artifact_id)
                ),
                content_hash="pending",
            )
            manifest = manifest.model_copy(
                update={"content_hash": self._manifest_hash(manifest)}
            )
            _write_json(temporary / "manifest.json", manifest)
            _write_json(
                temporary / "verification.json",
                {
                    "valid": True,
                    "artifact_count": len(references),
                    "artifact_verification": (
                        "content_hash_or_verified_predecessor"
                    ),
                    "downloaded_artifact_count": downloaded,
                    "inherited_artifact_count": inherited,
                    "materialized_artifact_count": materialized,
                    "reused_artifact_count": reused,
                    "postgres_dump_sha256": manifest.postgres_dump_sha256,
                },
            )
            self._raise_if_cancelled(cancel_event)
            temporary.replace(final)
            self.verify(
                final,
                verify_artifact_content=False,
                verify_inherited_artifacts=False,
            )
            self._prune()
            _checkpoint_log(
                milestone,
                "ready",
                artifact_count=len(references),
                downloaded=downloaded,
                downloaded_bytes=downloaded_bytes,
                elapsed_seconds=round(time.monotonic() - started, 1),
                inherited=inherited,
                inherited_bytes=inherited_bytes,
                materialized=materialized,
                reused=reused,
                reused_bytes=reused_bytes,
                sequence=sequence,
            )
            return final
        except CheckpointCancelled:
            _checkpoint_log(
                milestone,
                "cancelled",
                elapsed_seconds=round(time.monotonic() - started, 1),
                sequence=sequence,
            )
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except Exception as exc:
            _checkpoint_log(
                milestone,
                "failed",
                elapsed_seconds=round(time.monotonic() - started, 1),
                error_type=type(exc).__name__,
                sequence=sequence,
            )
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    async def restore_latest(self) -> Path | None:
        directory = self.latest()
        if directory is None:
            return None
        return await self.restore(directory)

    async def current_database_revision(self) -> str:
        """Read the already-migrated target revision once for restore selection."""

        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            async with engine.connect() as connection:
                revision = await connection.scalar(
                    text("SELECT version_num FROM alembic_version")
                )
            if revision is None:
                raise RuntimeError("database Alembic revision is unavailable")
            return str(revision)
        finally:
            await engine.dispose()

    async def restore(
        self,
        directory: Path,
        *,
        verified_manifest: ExperimentCheckpointManifest | None = None,
        require_empty: bool = False,
    ) -> Path:
        """Restore one explicitly selected, identity-compatible checkpoint."""

        manifest = verified_manifest or self.verify(directory, require_identity=False)
        if manifest.launch_id != self.launch_id:
            raise ValueError("verified checkpoint launch ID does not match")
        self._assert_restore_identity(manifest.identity)
        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            async with engine.connect() as connection:
                revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
                count = await connection.scalar(
                    text(f"SELECT count(*) FROM {RUNTIME_SCHEMA}.run_manifests")
                )
            database_revision = str(revision)
            revision_path = checkpoint_revision_restore_path(
                manifest.alembic_revision,
                database_revision,
            )
            if revision_path is None:
                print(
                    "CHECKPOINT_RESTORE_SCHEMA "
                    "state=incompatible "
                    f"checkpoint_revision={manifest.alembic_revision} "
                    f"database_revision={database_revision}",
                    flush=True,
                )
                raise ValueError(
                    "checkpoint Alembic revision has no audited data-only path "
                    "to the empty database"
                )
            if database_revision != manifest.alembic_revision:
                print(
                    "CHECKPOINT_RESTORE_SCHEMA "
                    "state=compatible "
                    f"checkpoint_revision={manifest.alembic_revision} "
                    f"database_revision={database_revision} "
                    "compatibility=data_only_forward_ddl "
                    f"hop_count={len(revision_path) - 1} "
                    f"revision_path={'->'.join(revision_path)}",
                    flush=True,
                )
            if int(count or 0) > 0:
                if require_empty:
                    raise ValueError("upstream_import_requires_empty_database")
                return directory
            if require_empty:
                async with engine.connect() as connection:
                    for table in ("dataset_snapshots", "compiler_runs", "query_executor_runs"):
                        statement = text(f"SELECT count(*) FROM {RUNTIME_SCHEMA}.{table}")
                        if await connection.scalar(statement):
                            raise ValueError("upstream_import_requires_empty_database")
        finally:
            await engine.dispose()
        await self._restore_database_with_progress(
            directory / "postgres.dump",
            sequence=manifest.sequence,
        )
        if manifest.artifacts:
            await self._restore_artifacts(
                directory,
                manifest,
            )
        await self._clear_expired_leases()
        await self._reconcile_restored_runtime_state()
        return directory

    def import_seed(
        self, source: ExperimentCheckpointService, directory: Path,
        manifest: ExperimentCheckpointManifest, state_path: Path, results_path: Path,
    ) -> Path:
        """Materialize a verified import under the destination, retaining source bytes."""
        if self.launch_id == source.launch_id or manifest.launch_id != source.launch_id:
            raise ValueError("upstream_seed_launch_invalid")
        if self.directories():
            raise ValueError("upstream_seed_destination_not_empty")
        if directory.resolve().parent != source.root.resolve():
            raise ValueError("upstream_seed_directory_invalid")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.root / f".import-{uuid4().hex}"
        temporary.mkdir(mode=0o700)
        final = self.root / "000001"
        started = time.monotonic()
        last_report = started
        try:
            paths = [(directory / "postgres.dump", temporary / "postgres.dump")]
            artifacts = []
            for artifact in manifest.artifacts:
                if artifact.scope not in {"runtime", "evaluation"} or not re.fullmatch(
                    r"[a-f0-9]{64}", artifact.sha256
                ):
                    raise ValueError("upstream_seed_artifact_identity_invalid")
                original = source._artifact_path(directory, artifact)
                if not original.resolve().is_relative_to(source.root.resolve()):
                    raise ValueError("upstream_seed_artifact_outside_source")
                relative = (
                    f"{artifact.scope}_objects/sha256/{artifact.sha256[:2]}/{artifact.sha256}"
                )
                paths.append((original, temporary / relative))
                artifacts.append(artifact.model_copy(update={
                    "relative_path": relative, "storage_sequence": 1,
                }))
            for index, (original, target) in enumerate(paths, 1):
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    # Do not chmod shared inodes or alter the source checkpoint.
                    if original.stat().st_mode & 0o777 == 0o600:
                        try:
                            os.link(original, target)
                        except OSError:
                            shutil.copyfile(original, target)
                            target.chmod(0o600)
                    else:
                        shutil.copyfile(original, target)
                        target.chmod(0o600)
                now = time.monotonic()
                if now - last_report >= _CHECKPOINT_PROGRESS_INTERVAL_SECONDS:
                    _checkpoint_log("upstream_import", "seed_materialization", completed=index,
                                    total=len(paths), elapsed_seconds=round(now - started, 1))
                    last_report = now
            for original, name in (
                (state_path, "launch_state.json"), (results_path, "stage_results.json")
            ):
                shutil.copyfile(original, temporary / name)
                (temporary / name).chmod(0o600)
            seed = manifest.model_copy(update={
                "launch_id": self.launch_id, "sequence": 1, "milestone": "upstream_import",
                "identity": self.identity, "artifacts": artifacts,
                "created_at": utc_now().isoformat(),
                "launch_state_sha256": _sha256(temporary / "launch_state.json"),
                "launch_state_size_bytes": (temporary / "launch_state.json").stat().st_size,
                "stage_results_sha256": _sha256(temporary / "stage_results.json"),
                "stage_results_size_bytes": (temporary / "stage_results.json").stat().st_size,
            })
            seed = seed.model_copy(update={"content_hash": self._manifest_hash(seed)})
            (temporary / "manifest.json").write_text(seed.model_dump_json(indent=2) + "\n")
            (temporary / "manifest.json").chmod(0o600)
            temporary.rename(final)
            _checkpoint_log("upstream_import", "seed_ready", sequence=1, artifacts=len(artifacts),
                            downloaded_bytes=0, database_dump_count=0)
            return final
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    async def _restore_database_with_progress(
        self,
        source: Path,
        *,
        sequence: int,
    ) -> None:
        started = time.monotonic()
        size_bytes = source.stat().st_size
        _checkpoint_restore_log(
            "database",
            "starting",
            dump_bytes=size_bytes,
            sequence=sequence,
        )
        task = asyncio.create_task(asyncio.to_thread(self._pg_restore, source))
        try:
            while not task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=_CHECKPOINT_PROGRESS_INTERVAL_SECONDS,
                    )
                except TimeoutError:
                    _checkpoint_restore_log(
                        "database",
                        "progress",
                        dump_bytes=size_bytes,
                        elapsed_seconds=round(time.monotonic() - started, 1),
                        sequence=sequence,
                    )
            await task
        except Exception as exc:
            _checkpoint_restore_log(
                "database",
                "failed",
                elapsed_seconds=round(time.monotonic() - started, 1),
                error_type=type(exc).__name__,
                sequence=sequence,
            )
            raise
        _checkpoint_restore_log(
            "database",
            "ready",
            dump_bytes=size_bytes,
            elapsed_seconds=round(time.monotonic() - started, 1),
            sequence=sequence,
        )

    async def _restore_artifacts(
        self,
        directory: Path,
        manifest: ExperimentCheckpointManifest,
    ) -> None:
        runtime_store, evaluation_store = create_admin_object_stores(self.settings)
        stores = {"runtime": runtime_store, "evaluation": evaluation_store}
        total = len(manifest.artifacts)
        total_bytes = sum(item.size_bytes for item in manifest.artifacts)
        completed = 0
        completed_bytes = 0
        started = time.monotonic()
        last_progress = started
        _checkpoint_restore_log(
            "object_storage",
            "starting",
            concurrency=_CHECKPOINT_RESTORE_OBJECT_CONCURRENCY,
            sequence=manifest.sequence,
            total=total,
            total_bytes=total_bytes,
        )

        async def restore_one(artifact: CheckpointArtifactRef) -> int:
            path = self._artifact_path(directory, artifact)
            await stores[artifact.scope].put_file(
                path,
                artifact.object_key,
                expected_sha256=artifact.sha256,
            )
            return artifact.size_bytes

        try:
            for offset in range(
                0, total, _CHECKPOINT_RESTORE_OBJECT_CONCURRENCY
            ):
                batch = manifest.artifacts[
                    offset : offset + _CHECKPOINT_RESTORE_OBJECT_CONCURRENCY
                ]
                restored_bytes = await asyncio.gather(
                    *(restore_one(artifact) for artifact in batch)
                )
                completed += len(batch)
                completed_bytes += sum(restored_bytes)
                now = time.monotonic()
                if (
                    completed == total
                    or now - last_progress >= _CHECKPOINT_PROGRESS_INTERVAL_SECONDS
                ):
                    elapsed = max(now - started, 1e-9)
                    _checkpoint_restore_log(
                        "object_storage",
                        "ready" if completed == total else "progress",
                        completed=completed,
                        completed_bytes=completed_bytes,
                        elapsed_seconds=round(elapsed, 1),
                        items_per_second=round(completed / elapsed, 3),
                        sequence=manifest.sequence,
                        total=total,
                        total_bytes=total_bytes,
                    )
                    last_progress = now
        except Exception as exc:
            _checkpoint_restore_log(
                "object_storage",
                "failed",
                completed=completed,
                completed_bytes=completed_bytes,
                elapsed_seconds=round(time.monotonic() - started, 1),
                error_type=type(exc).__name__,
                sequence=manifest.sequence,
                total=total,
            )
            raise

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise CheckpointCancelled("periodic checkpoint is obsolete")

    def _reusable_artifacts(
        self,
        directory: Path | None,
    ) -> dict[tuple[str, str, int], _ReusableArtifact]:
        if directory is None:
            return {}
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return {}
        try:
            manifest = ExperimentCheckpointManifest.from_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return {}
        if (
            manifest.launch_id != self.launch_id
            or manifest.identity != self.identity
            or not self._manifest_hash_is_valid(manifest)
        ):
            return {}
        reusable: dict[tuple[str, str, int], _ReusableArtifact] = {}
        durable_sequences: dict[int, bool] = {}
        for artifact in manifest.artifacts:
            durable: bool | None = True
            storage_sequence = artifact.storage_sequence or manifest.sequence
            if storage_sequence <= 0 or storage_sequence > manifest.sequence:
                return {}
            if self.retain_milestones:
                durable = durable_sequences.get(storage_sequence)
                if durable is None:
                    durable = self._storage_sequence_is_durable(storage_sequence)
                    durable_sequences[storage_sequence] = durable
                # Prunable content may be copied/linked, but never referenced.
            reusable[(artifact.scope, artifact.sha256, artifact.size_bytes)] = (
                _ReusableArtifact(
                    path=self._artifact_path(directory, artifact),
                    storage_sequence=storage_sequence,
                    durable=bool(durable),
                )
            )
        return reusable

    def _storage_sequence_is_durable(self, sequence: int) -> bool:
        manifest_path = self.root / f"{sequence:06d}" / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = ExperimentCheckpointManifest.from_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return False
        return (
            manifest.launch_id == self.launch_id
            and manifest.identity == self.identity
            and manifest.sequence == sequence
            and self._manifest_hash_is_valid(manifest)
            and not manifest.milestone.startswith("periodic:")
        )

    def physical_size_bytes(self, directory: Path) -> int:
        """Return physical checkpoint bytes without recursively statting all refs."""

        manifest = ExperimentCheckpointManifest.from_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        current_payloads = {
            (artifact.scope, artifact.sha256, artifact.size_bytes)
            for artifact in manifest.artifacts
            if (artifact.storage_sequence or manifest.sequence) == manifest.sequence
        }
        fixed = sum(
            path.stat().st_size
            for path in (
                directory / "postgres.dump",
                directory / "manifest.json",
                directory / "verification.json",
                directory / "stage_results.json",
                directory / "launch_state.json",
            )
            if path.is_file()
        )
        return fixed + sum(size for _scope, _sha256_value, size in current_payloads)

    def _artifact_path(
        self,
        directory: Path,
        artifact: CheckpointArtifactRef,
    ) -> Path:
        relative = Path(artifact.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint artifact path is not relative and bounded")
        storage_sequence = artifact.storage_sequence
        base = (
            directory
            if storage_sequence is None
            else self.root / f"{storage_sequence:06d}"
        )
        return base / relative

    async def completed_job_count(self) -> int:
        """Return persisted successes used by periodic checkpoint scheduling."""

        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            async with engine.connect() as connection:
                runtime = await connection.scalar(
                    text(
                        f"SELECT count(*) FROM {RUNTIME_SCHEMA}.experiment_jobs "
                        "WHERE status='succeeded'"
                    )
                )
                evaluation = await connection.scalar(
                    text(
                        f"SELECT count(*) FROM {EVALUATION_SCHEMA}.evaluation_jobs "
                        "WHERE status='succeeded'"
                    )
                )
            return int(runtime or 0) + int(evaluation or 0)
        finally:
            await engine.dispose()

    async def _consistent_dump(
        self,
        output: Path,
        *,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, list[_ArtifactRow]]:
        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            async with engine.connect() as connection:
                transaction = await connection.begin()
                try:
                    await connection.exec_driver_sql(
                        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                    )
                    snapshot = str(await connection.scalar(text("SELECT pg_export_snapshot()")))
                    revision = str(
                        await connection.scalar(text("SELECT version_num FROM alembic_version"))
                    )
                    runtime_rows = (
                        await connection.execute(
                            select(
                                RuntimeArtifactModel.artifact_id,
                                RuntimeArtifactModel.object_key,
                                RuntimeArtifactModel.sha256,
                                RuntimeArtifactModel.size_bytes,
                            ).where(RuntimeArtifactModel.state == "ready")
                        )
                    ).all()
                    evaluation_rows = (
                        await connection.execute(
                            select(
                                EvaluationArtifactModel.artifact_id,
                                EvaluationArtifactModel.object_key,
                                EvaluationArtifactModel.sha256,
                                EvaluationArtifactModel.size_bytes,
                            ).where(EvaluationArtifactModel.state == "ready")
                        )
                    ).all()
                    self._raise_if_cancelled(cancel_event)
                    await asyncio.to_thread(
                        self._pg_dump,
                        output,
                        snapshot,
                        cancel_event,
                    )
                finally:
                    await transaction.rollback()
        finally:
            await engine.dispose()
        rows = [
            _ArtifactRow(scope, str(row[0]), str(row[1]), str(row[2]), int(row[3]))
            for scope, values in (
                ("runtime", runtime_rows),
                ("evaluation", evaluation_rows),
            )
            for row in values
            if row[3] is not None
        ]
        rows.sort(key=lambda item: (item.scope, item.artifact_id))
        return revision, rows

    def _database_command(self, executable: str) -> tuple[list[str], dict[str, str]]:
        url = make_url(database_url_for_role(self.settings, DatabaseRole.ADMIN))
        command = [
            executable,
            "--host",
            str(url.host or "127.0.0.1"),
            "--port",
            str(url.port or 5432),
            "--username",
            str(url.username or "pmmc_owner"),
            "--dbname",
            str(url.database or "pmmc"),
        ]
        environment = dict(os.environ)
        environment["PGPASSWORD"] = str(url.password or "")
        return command, environment

    def _pg_dump(
        self,
        output: Path,
        snapshot: str,
        cancel_event: threading.Event | None = None,
    ) -> None:
        command, environment = self._database_command("pg_dump")
        command.extend(
            [
                "--format=custom",
                "--data-only",
                "--no-owner",
                "--no-privileges",
                f"--snapshot={snapshot}",
                f"--schema={RUNTIME_SCHEMA}",
                f"--schema={EVALUATION_SCHEMA}",
                f"--file={output}",
            ]
        )
        if cancel_event is None:
            subprocess.run(command, env=environment, check=True, capture_output=True)
            return
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while process.poll() is None:
            if cancel_event.wait(0.25):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise CheckpointCancelled("periodic checkpoint dump was cancelled")
        stdout, stderr = process.communicate()
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                command,
                output=stdout,
                stderr=stderr,
            )

    def _pg_restore(self, source: Path) -> None:
        command, environment = self._database_command("pg_restore")
        command.extend(
            [
                "--data-only",
                "--disable-triggers",
                "--exit-on-error",
                "--no-owner",
                "--no-privileges",
                str(source),
            ]
        )
        subprocess.run(command, env=environment, check=True, capture_output=True)

    async def _clear_expired_leases(self) -> None:
        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            async with engine.begin() as connection:
                for statement in _expired_lease_cleanup_sql():
                    await connection.execute(text(statement))
        finally:
            await engine.dispose()

    async def _reconcile_restored_runtime_state(self) -> None:
        """Close process state that cannot survive a checkpoint restore."""

        session_sql, service_sql = _restored_runtime_reconciliation_sql()
        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            async with engine.begin() as connection:
                session_count = int(
                    (await connection.scalar(text(session_sql))) or 0
                )
                service_count = int(
                    (await connection.scalar(text(service_sql))) or 0
                )
        finally:
            await engine.dispose()
        _checkpoint_restore_log(
            "runtime_state",
            "reconciled",
            cancelled_session_count=session_count,
            stopped_service_count=service_count,
        )

    def _prune(self) -> None:
        if not self.retain_milestones:
            directories = self.directories()
            for path in directories[: -self.retain_count]:
                shutil.rmtree(path)
            return
        periodic: list[Path] = []
        for path in self.directories():
            try:
                manifest = ExperimentCheckpointManifest.from_json(
                    (path / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if manifest.milestone.startswith("periodic:"):
                periodic.append(path)
        for path in periodic[: -self.retain_count]:
            shutil.rmtree(path)

    @staticmethod
    def _manifest_hash(manifest: ExperimentCheckpointManifest) -> str:
        payload = manifest.model_dump(mode="json", exclude={"content_hash"})
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _manifest_hash_is_valid(
        cls,
        manifest: ExperimentCheckpointManifest,
    ) -> bool:
        if manifest.content_hash == cls._manifest_hash(manifest):
            return True
        # Checkpoints written before incremental milestone storage do not have
        # ``storage_sequence`` in their artifact objects. Preserve their exact
        # historical hash representation while rejecting a new manifest that
        # explicitly sets the field to null.
        if any("storage_sequence" in item.model_fields_set for item in manifest.artifacts):
            return False
        payload = manifest.model_dump(mode="json", exclude={"content_hash"})
        for artifact in payload.get("artifacts", []):
            if isinstance(artifact, dict):
                artifact.pop("storage_sequence", None)
        legacy_hash = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return manifest.content_hash == legacy_hash

    def _assert_restore_identity(self, stored: JsonDict) -> None:
        """Allow only the dynamic snapshot map to roll back with launcher state."""

        immutable_keys = {
            "launcher_schema_version",
            "launch_config_hash",
            "execution_profile_hash",
            "source_inputs_hash",
            "model_lock_sha256",
        }
        mismatches = tuple(
            key for key in sorted(immutable_keys) if stored.get(key) != self.identity.get(key)
        )
        if mismatches:
            raise CheckpointIdentityMismatch(mismatches)


def verify_postgres_client_major(expected_major: int = 16) -> dict[str, object]:
    """Verify that dump and restore clients match the Pod-local Postgres major."""

    versions: dict[str, str] = {}
    for executable in ("pg_dump", "pg_restore"):
        result = subprocess.run(
            [executable, "--version"], check=True, capture_output=True, text=True
        )
        versions[executable] = result.stdout.strip()
        if f" {expected_major}." not in result.stdout:
            raise RuntimeError(
                f"{executable} must use PostgreSQL {expected_major}; "
                f"found {result.stdout.strip() or 'unknown version'}"
            )
    return {"expected_major": expected_major, "versions": versions}


def _expired_lease_cleanup_sql() -> tuple[str, ...]:
    """Return schema-correct cleanup statements for restored job leases."""

    statements: list[str] = []
    for schema, jobs, attempts, job_id, attempt_job_id in (
        (
            RUNTIME_SCHEMA,
            "experiment_jobs",
            "job_attempts",
            "job_id",
            "job_id",
        ),
        (
            EVALUATION_SCHEMA,
            "evaluation_jobs",
            "evaluation_job_attempts",
            "evaluation_job_id",
            "evaluation_job_id",
        ),
    ):
        statements.append(
            f"""UPDATE {schema}.{attempts} AS attempts
            SET status='failed', finished_at=now(), error_type='LeaseExpired',
                error_summary='checkpoint restore cleared an expired worker lease'
            FROM {schema}.{jobs} AS jobs
            WHERE attempts.{attempt_job_id}=jobs.{job_id}
              AND attempts.attempt_number=jobs.attempt_count
              AND attempts.status='running'
              AND jobs.status='running'
              AND jobs.lease_expires_at < now()"""
        )
        error_artifact = ", error_artifact_id=NULL" if schema == RUNTIME_SCHEMA else ""
        statements.append(
            f"""UPDATE {schema}.{jobs} AS jobs
            SET status=CASE
                    WHEN jobs.attempt_count < jobs.max_attempts THEN 'pending'
                    ELSE 'failed'
                END,
                claimed_by=NULL,
                lease_expires_at=NULL,
                available_at=CASE
                    WHEN jobs.attempt_count < jobs.max_attempts THEN now()
                    ELSE jobs.available_at
                END,
                finished_at=CASE
                    WHEN jobs.attempt_count < jobs.max_attempts THEN NULL
                    ELSE now()
                END,
                error_type='LeaseExpired',
                error_summary=CASE
                    WHEN jobs.attempt_count < jobs.max_attempts
                        THEN 'checkpoint restore cleared an expired worker lease'
                    ELSE 'job lease expired after the final attempt'
                END{error_artifact}
            WHERE jobs.status='running'
              AND jobs.lease_expires_at < now()"""
        )
    return tuple(statements)


def _restored_runtime_reconciliation_sql() -> tuple[str, str]:
    """Return deterministic cleanup for process state restored into a new pod."""

    sessions = f"""WITH reconciled AS (
        UPDATE {RUNTIME_SCHEMA}.managed_runtime_sessions
        SET status='cancelled', completed_at=COALESCE(completed_at, now()),
            metadata=COALESCE(metadata, '{{}}'::jsonb) || jsonb_build_object(
                'checkpoint_restore_reconciled', true,
                'checkpoint_restore_previous_status', status
            )
        WHERE status IN ('planned','starting','running')
        RETURNING 1
    ) SELECT count(*) FROM reconciled"""
    services = f"""WITH reconciled AS (
        UPDATE {RUNTIME_SCHEMA}.managed_runtime_services
        SET status='stopped',
            stopped_at=CASE
                WHEN started_at IS NULL THEN stopped_at
                ELSE COALESCE(stopped_at, now())
            END,
            metadata=COALESCE(metadata, '{{}}'::jsonb) || jsonb_build_object(
                'checkpoint_restore_reconciled', true,
                'checkpoint_restore_previous_status', status
            )
        WHERE status IN ('planned','starting','ready','stopping')
        RETURNING 1
    ) SELECT count(*) FROM reconciled"""
    return sessions, services
