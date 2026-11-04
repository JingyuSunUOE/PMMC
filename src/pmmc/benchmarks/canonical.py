"""Integrity checks for immutable canonical benchmark snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from pmmc.benchmarks.ids import canonical_snapshot_id, legacy_canonical_snapshot_id
from pmmc.infrastructure.database.errors import SnapshotIntegrityError
from pmmc.infrastructure.database.mappers import merge_image_refs
from pmmc.schemas import (
    SCHEMA_VERSION,
    AssetRef,
    BaseSchema,
    BenchmarkQuery,
    DatasetSnapshotManifest,
    ImageRef,
    MemoryUnit,
    QueryGoldRecord,
    SessionRecord,
    ValidationReport,
)

CANONICAL_FILES = (
    "sessions.jsonl",
    "assets.jsonl",
    "memory_units.jsonl",
    "queries.jsonl",
    "query_gold.jsonl",
)

SchemaValue = TypeVar("SchemaValue", bound=BaseSchema)


@dataclass(frozen=True)
class CanonicalPreflightReport:
    """Validated snapshot facts needed by the database importer."""

    manifest: DatasetSnapshotManifest
    validation_report: ValidationReport
    counts: dict[str, int]
    images: dict[str, ImageRef]


def iter_jsonl(path: Path, schema_type: type[SchemaValue]) -> Iterator[SchemaValue]:
    """Stream one canonical JSONL file through its public Pydantic contract."""

    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    yield schema_type.from_json(line)
                except ValidationError as exc:
                    raise SnapshotIntegrityError(
                        f"invalid {path.name} record at line {line_number}: "
                        f"{exc.error_count()} validation error(s)"
                    ) from exc
    except OSError as exc:
        raise SnapshotIntegrityError(f"cannot read canonical file: {path.name}") from exc


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _hashable(item)
            for key, item in value.items()
            if key not in {"created_at", "started_at", "completed_at"}
        }
    if isinstance(value, list):
        return [_hashable(item) for item in value]
    return value


def _canonical_bytes(value: BaseSchema) -> bytes:
    return (
        json.dumps(
            _hashable(value.to_dict()),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def recompute_canonical_content_hash(snapshot_path: Path) -> str:
    """Reproduce the Step 3 writer's stable, timestamp-independent digest."""

    digest = hashlib.sha256()
    for file_name in CANONICAL_FILES:
        digest.update(file_name.encode("utf-8"))

    for asset in iter_jsonl(snapshot_path / "assets.jsonl", AssetRef):
        digest.update(_canonical_bytes(asset))

    sessions_by_conversation: OrderedDict[str, list[SessionRecord]] = OrderedDict()
    for session in iter_jsonl(snapshot_path / "sessions.jsonl", SessionRecord):
        sessions_by_conversation.setdefault(session.conversation_id, []).append(session)

    memories_by_conversation: dict[str, list[MemoryUnit]] = {}
    for memory in iter_jsonl(snapshot_path / "memory_units.jsonl", MemoryUnit):
        memories_by_conversation.setdefault(memory.conversation_id, []).append(memory)

    queries_by_conversation: dict[str, list[BenchmarkQuery]] = {}
    for query in iter_jsonl(snapshot_path / "queries.jsonl", BenchmarkQuery):
        queries_by_conversation.setdefault(query.query.conversation_id, []).append(query)

    gold_by_query = {
        record.query_id: record
        for record in iter_jsonl(snapshot_path / "query_gold.jsonl", QueryGoldRecord)
    }

    for conversation_id, sessions in sessions_by_conversation.items():
        for session in sessions:
            digest.update(_canonical_bytes(session))
        for memory in memories_by_conversation.get(conversation_id, []):
            digest.update(_canonical_bytes(memory))
        for query in queries_by_conversation.get(conversation_id, []):
            digest.update(_canonical_bytes(query))
            gold = gold_by_query.get(query.query.query_id)
            if gold is None:
                raise SnapshotIntegrityError(f"query has no gold record: {query.query.query_id}")
            digest.update(_canonical_bytes(gold))

    return digest.hexdigest()


def _require_unique(identifier: str, seen: set[str], kind: str) -> None:
    if identifier in seen:
        raise SnapshotIntegrityError(f"duplicate {kind} ID: {identifier}")
    seen.add(identifier)


def load_snapshot_manifest(snapshot_path: Path) -> DatasetSnapshotManifest:
    """Load and validate the canonical snapshot manifest."""

    manifest_path = snapshot_path / "manifest.json"
    try:
        manifest = DatasetSnapshotManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise SnapshotIntegrityError("invalid or missing manifest.json") from exc
    if manifest.snapshot_id != snapshot_path.name:
        raise SnapshotIntegrityError("manifest snapshot_id does not match directory name")
    if manifest.schema_version != SCHEMA_VERSION:
        raise SnapshotIntegrityError(
            f"unsupported schema version: {manifest.schema_version}; expected {SCHEMA_VERSION}"
        )
    return manifest


def preflight_snapshot(snapshot_path: Path) -> CanonicalPreflightReport:
    """Validate references, counts, and digest before any database write."""

    if not snapshot_path.is_dir():
        raise SnapshotIntegrityError(f"snapshot directory does not exist: {snapshot_path.name}")
    for file_name in (*CANONICAL_FILES, "manifest.json", "validation_report.json"):
        path = snapshot_path / file_name
        if not path.is_file() or path.stat().st_size == 0:
            raise SnapshotIntegrityError(f"missing or empty canonical file: {file_name}")

    try:
        manifest_payload = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError("invalid or missing manifest.json") from exc
    manifest = load_snapshot_manifest(snapshot_path)
    try:
        validation_report = ValidationReport.from_json(
            (snapshot_path / "validation_report.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise SnapshotIntegrityError("invalid validation_report.json") from exc
    if validation_report.dataset is not manifest.dataset:
        raise SnapshotIntegrityError("validation report dataset does not match manifest")
    if not validation_report.success:
        raise SnapshotIntegrityError("snapshot validation report is not successful")
    if validation_report.counts != manifest.counts:
        raise SnapshotIntegrityError("validation report counts do not match manifest")

    computed_hash = recompute_canonical_content_hash(snapshot_path)
    if computed_hash != manifest.canonical_content_hash:
        raise SnapshotIntegrityError("canonical content hash does not match manifest")
    expected_snapshot_id = canonical_snapshot_id(
        manifest.dataset,
        source_content_hash=manifest.source_content_hash,
        canonical_content_hash=manifest.canonical_content_hash,
        schema_version=manifest.schema_version,
        adapter_name=manifest.adapter_name,
        adapter_version=manifest.adapter_version,
        dataset_version=manifest.dataset_version,
        evaluation_policy_version=manifest.evaluation_policy_version,
        excluded_source_categories=manifest.excluded_source_categories,
    )
    if manifest.snapshot_id != expected_snapshot_id:
        legacy_snapshot_id = legacy_canonical_snapshot_id(
            manifest.dataset,
            source_content_hash=manifest.source_content_hash,
            canonical_content_hash=manifest.canonical_content_hash,
        )
        is_legacy_manifest = "excluded_source_categories" not in manifest_payload
        if not is_legacy_manifest or manifest.snapshot_id != legacy_snapshot_id:
            raise SnapshotIntegrityError("snapshot ID does not match manifest fingerprint")

    asset_ids: set[str] = set()
    for asset in iter_jsonl(snapshot_path / "assets.jsonl", AssetRef):
        if asset.asset_id is None:
            raise SnapshotIntegrityError("canonical benchmark asset is missing asset_id")
        _require_unique(asset.asset_id, asset_ids, "asset")

    session_ids: set[str] = set()
    session_conversations: dict[str, str] = {}
    session_orders: dict[str, int] = {}
    for session in iter_jsonl(snapshot_path / "sessions.jsonl", SessionRecord):
        if session.dataset is not manifest.dataset:
            raise SnapshotIntegrityError("session dataset does not match manifest")
        _require_unique(session.session_id, session_ids, "session")
        session_conversations[session.session_id] = session.conversation_id
        session_orders[session.session_id] = session.session_order

    images: dict[str, ImageRef] = {}
    memory_ids: set[str] = set()
    memory_conversations: dict[str, str] = {}
    memory_available_orders: dict[str, int] = {}
    memory_count = 0
    for memory in iter_jsonl(snapshot_path / "memory_units.jsonl", MemoryUnit):
        memory_count += 1
        if memory.dataset is not manifest.dataset:
            raise SnapshotIntegrityError("memory dataset does not match manifest")
        _require_unique(memory.memory_id, memory_ids, "memory")
        for session_id in (memory.session_id, memory.available_after_session):
            if session_conversations.get(session_id) != memory.conversation_id:
                raise SnapshotIntegrityError(
                    f"memory references an invalid conversation session: {memory.memory_id}"
                )
        memory_conversations[memory.memory_id] = memory.conversation_id
        memory_available_orders[memory.memory_id] = session_orders[memory.available_after_session]
        refs = {item.image_id: item for item in memory.image_refs}
        for image_id in memory.image_ids:
            refs.setdefault(image_id, ImageRef(image_id=image_id))
        for image in refs.values():
            images[image.image_id] = merge_image_refs(images.get(image.image_id), image)

    query_ids: set[str] = set()
    eligible_count = 0
    excluded_count = 0
    query_count = 0
    conversation_ids = set(session_conversations.values())
    query_conversations: dict[str, str] = {}
    query_cutoff_orders: dict[str, int | None] = {}
    for record in iter_jsonl(snapshot_path / "queries.jsonl", BenchmarkQuery):
        query_count += 1
        query = record.query
        if query.dataset is not manifest.dataset:
            raise SnapshotIntegrityError("query dataset does not match manifest")
        _require_unique(query.query_id, query_ids, "query")
        if query.conversation_id not in conversation_ids:
            raise SnapshotIntegrityError(
                f"query references an unknown conversation: {query.query_id}"
            )
        query_conversations[query.query_id] = query.conversation_id
        if query.available_after_session is not None:
            if session_conversations.get(query.available_after_session) != query.conversation_id:
                raise SnapshotIntegrityError(
                    f"query references an invalid cutoff session: {query.query_id}"
                )
            query_cutoff_orders[query.query_id] = session_orders[query.available_after_session]
        else:
            query_cutoff_orders[query.query_id] = None
        eligible_count += int(record.eligibility.eligible)
        excluded_count += int(not record.eligibility.eligible)
        for image in query.query_image_refs:
            images[image.image_id] = merge_image_refs(images.get(image.image_id), image)

    for image in images.values():
        if image.asset_id is not None and image.asset_id not in asset_ids:
            raise SnapshotIntegrityError(f"image references an unknown asset: {image.image_id}")

    gold_query_ids: set[str] = set()
    gold_count = 0
    for gold_record in iter_jsonl(snapshot_path / "query_gold.jsonl", QueryGoldRecord):
        gold_count += 1
        _require_unique(gold_record.query_id, gold_query_ids, "gold query")
        if gold_record.query_id not in query_ids:
            raise SnapshotIntegrityError(
                f"gold record references an unknown query: {gold_record.query_id}"
            )
        evidence = gold_record.gold.evidence
        missing_memories = set(evidence.memory_ids) - memory_ids
        missing_sessions = set(evidence.session_ids) - session_ids
        missing_images = set(evidence.image_ids) - set(images)
        if missing_memories or missing_sessions or missing_images:
            raise SnapshotIntegrityError(
                f"gold record has unresolved typed evidence: {gold_record.query_id}"
            )
        query_conversation = query_conversations[gold_record.query_id]
        if any(
            memory_conversations[memory_id] != query_conversation
            for memory_id in evidence.memory_ids
        ) or any(
            session_conversations[session_id] != query_conversation
            for session_id in evidence.session_ids
        ):
            raise SnapshotIntegrityError(
                f"gold evidence crosses query conversation: {gold_record.query_id}"
            )
        cutoff_order = query_cutoff_orders[gold_record.query_id]
        if cutoff_order is not None and (
            any(
                memory_available_orders[memory_id] > cutoff_order
                for memory_id in evidence.memory_ids
            )
            or any(session_orders[session_id] > cutoff_order for session_id in evidence.session_ids)
        ):
            raise SnapshotIntegrityError(
                f"gold evidence is unavailable at query cutoff: {gold_record.query_id}"
            )
    if gold_query_ids != query_ids:
        raise SnapshotIntegrityError("runtime queries and gold records are not one-to-one")

    counts = {
        "sessions": len(session_ids),
        "assets": len(asset_ids),
        "memory_units": memory_count,
        "queries": query_count,
        "eligible_queries": eligible_count,
        "excluded_queries": excluded_count,
        "query_gold": gold_count,
        "images": len(images),
    }
    for key in (
        "sessions",
        "assets",
        "memory_units",
        "queries",
        "eligible_queries",
        "excluded_queries",
    ):
        expected = manifest.counts.get(key)
        if expected is not None and counts[key] != expected:
            raise SnapshotIntegrityError(
                f"manifest count mismatch for {key}: expected {expected}, got {counts[key]}"
            )
    expected_images = manifest.counts.get("referenced_unique_images")
    if expected_images is not None and counts["images"] != expected_images:
        raise SnapshotIntegrityError(
            "manifest count mismatch for referenced_unique_images: "
            f"expected {expected_images}, got {counts['images']}"
        )
    return CanonicalPreflightReport(
        manifest=manifest,
        validation_report=validation_report,
        counts=counts,
        images=images,
    )
