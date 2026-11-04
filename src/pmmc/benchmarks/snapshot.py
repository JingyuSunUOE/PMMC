"""Atomic canonical JSONL snapshot generation and discovery."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pmmc.benchmarks.base import BenchmarkAdapter, BenchmarkDataError
from pmmc.benchmarks.ids import canonical_snapshot_id
from pmmc.benchmarks.taxonomy import EXCLUDED_SOURCE_CATEGORIES
from pmmc.benchmarks.validation import ProgressCallback, scan_adapter
from pmmc.schemas import (
    DEFAULT_EVALUATION_POLICY_VERSION,
    SCHEMA_VERSION,
    BenchmarkQuery,
    DatasetSnapshotManifest,
    QueryGoldRecord,
    ValidationLevel,
)

CANONICAL_FILES = (
    "sessions.jsonl",
    "assets.jsonl",
    "memory_units.jsonl",
    "queries.jsonl",
    "query_gold.jsonl",
)


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


class _JsonlWriter:
    def __init__(self, path: Path, digest: Any) -> None:
        self.path = path
        self._stream = path.open("w", encoding="utf-8")
        self._digest = digest
        self._digest.update(path.name.encode("utf-8"))

    def write(self, value: Any) -> None:
        line = value.to_json()
        self._stream.write(line)
        self._stream.write("\n")
        canonical = json.dumps(
            _hashable(value.to_dict()),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._digest.update(canonical.encode("utf-8"))
        self._digest.update(b"\n")

    def close(self) -> None:
        self._stream.close()

    @property
    def closed(self) -> bool:
        return self._stream.closed


def build_snapshot(
    adapter: BenchmarkAdapter,
    *,
    output_root: Path,
    level: ValidationLevel,
    enforce_expected_counts: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[Path, DatasetSnapshotManifest]:
    """Validate and atomically write one canonical benchmark snapshot."""

    scan = scan_adapter(
        adapter,
        level,
        enforce_expected_counts=enforce_expected_counts,
        progress=progress,
    )
    if not scan.report.success:
        messages = "; ".join(issue.message for issue in scan.report.issues)
        raise BenchmarkDataError(f"cannot import invalid benchmark: {messages}")

    dataset_root = output_root / adapter.dataset.value
    dataset_root.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=dataset_root))
    digest = hashlib.sha256()
    writers = {name: _JsonlWriter(temp_path / name, digest) for name in CANONICAL_FILES}
    try:
        for asset_index, asset in enumerate(scan.assets, start=1):
            writers["assets.jsonl"].write(asset)
            if progress is not None:
                progress("write_assets", asset_index, len(scan.assets))

        for record_index, conversation in enumerate(adapter.iter_conversations(), start=1):
            for session in conversation.sessions:
                writers["sessions.jsonl"].write(session)
            for memory in conversation.memory_units:
                writers["memory_units.jsonl"].write(memory)
            for adapted in conversation.queries:
                if adapted.record.gold is None:
                    raise BenchmarkDataError(
                        f"benchmark query lacks gold record: {adapted.record.query.query_id}"
                    )
                writers["queries.jsonl"].write(
                    BenchmarkQuery(
                        query=adapted.record.query,
                        source=adapted.source,
                        eligibility=adapted.eligibility,
                    )
                )
                writers["query_gold.jsonl"].write(
                    QueryGoldRecord(
                        query_id=adapted.record.query.query_id,
                        gold=adapted.record.gold,
                    )
                )
            if progress is not None:
                progress("write_records", record_index, adapter.source_record_count)
        for writer in writers.values():
            writer.close()

        canonical_content_hash = digest.hexdigest()
        excluded_categories = sorted(EXCLUDED_SOURCE_CATEGORIES[adapter.dataset])
        snapshot_id = canonical_snapshot_id(
            adapter.dataset,
            source_content_hash=scan.source_content_hash,
            canonical_content_hash=canonical_content_hash,
            schema_version=SCHEMA_VERSION,
            adapter_name=adapter.adapter_name,
            adapter_version=adapter.adapter_version,
            dataset_version=adapter.dataset_version,
            evaluation_policy_version=DEFAULT_EVALUATION_POLICY_VERSION,
            excluded_source_categories=excluded_categories,
        )
        manifest = DatasetSnapshotManifest(
            snapshot_id=snapshot_id,
            dataset=adapter.dataset,
            dataset_version=adapter.dataset_version,
            adapter_name=adapter.adapter_name,
            adapter_version=adapter.adapter_version,
            excluded_source_categories=excluded_categories,
            source_content_hash=scan.source_content_hash,
            canonical_content_hash=canonical_content_hash,
            counts=scan.report.counts,
            category_counts=scan.category_counts,
            eligible_category_counts=scan.eligible_category_counts,
            excluded_category_counts=scan.excluded_category_counts,
            metadata={"validation_level": level.value},
        )
        (temp_path / "manifest.json").write_text(
            manifest.to_json() + "\n",
            encoding="utf-8",
        )
        (temp_path / "validation_report.json").write_text(
            scan.report.to_json() + "\n",
            encoding="utf-8",
        )

        final_path = dataset_root / snapshot_id
        if final_path.exists():
            existing = load_manifest(final_path)
            if existing.canonical_content_hash != canonical_content_hash:
                raise BenchmarkDataError(
                    f"snapshot ID collision with different content: {snapshot_id}"
                )
            shutil.rmtree(temp_path)
            return final_path, existing
        temp_path.rename(final_path)
        return final_path, manifest
    except Exception:
        for writer in writers.values():
            if not writer.closed:
                writer.close()
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def load_manifest(snapshot_path: Path) -> DatasetSnapshotManifest:
    """Read one snapshot manifest."""

    return DatasetSnapshotManifest.from_json(
        (snapshot_path / "manifest.json").read_text(encoding="utf-8")
    )


def find_snapshot(output_root: Path, snapshot_id: str) -> Path:
    """Resolve a unique snapshot ID below the benchmark artifact root."""

    if Path(snapshot_id).name != snapshot_id or "\\" in snapshot_id or snapshot_id in {".", ".."}:
        raise ValueError("snapshot_id must be a single safe path component")
    matches = sorted(output_root.glob(f"*/{snapshot_id}"))
    if not matches:
        raise FileNotFoundError(f"snapshot does not exist: {snapshot_id}")
    if len(matches) > 1:
        raise ValueError(f"snapshot ID is ambiguous: {snapshot_id}")
    return matches[0]
