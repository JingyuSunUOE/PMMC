"""Archive only redacted managed-runtime provenance to the runtime bucket."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pmmc.infrastructure.object_store import (
    ObjectStore,
    runtime_artifact_object_key,
    sha256_file,
)


def _runtime_files(runtime_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    fixed = {
        "plan": runtime_dir / "runtime-plan.json",
        "session": runtime_dir / "runtime-session.json",
        "versions": runtime_dir / "software-versions.json",
        "telemetry": runtime_dir / "gpu-telemetry.jsonl",
    }
    for artifact_type, path in fixed.items():
        if path.is_file():
            files.append((artifact_type, path))
    for path in sorted((runtime_dir / "services").glob("**/*.log.redacted")):
        files.append(("service-log", path))
    for path in sorted(runtime_dir.glob("service-states-*.json")):
        files.append(("service-state", path))
    return files


async def archive_runtime(runtime_dir: Path, store: ObjectStore) -> dict[str, object]:
    runtime_dir = runtime_dir.resolve()
    runtime_id = runtime_dir.name
    records: list[dict[str, object]] = []
    for artifact_type, path in _runtime_files(runtime_dir):
        digest = sha256_file(path)
        extension = path.name.rsplit(".", 1)[-1].lower()
        key = runtime_artifact_object_key(
            runtime_id,
            artifact_type,
            digest,
            extension=extension,
        )
        uploaded = await store.put_file(
            path,
            key,
            expected_sha256=digest,
            content_type={
                "json": "application/json",
                "jsonl": "application/x-ndjson",
                "redacted": "text/plain",
            }.get(extension, "application/octet-stream"),
        )
        records.append(
            {
                "artifact_type": artifact_type,
                "source_name": path.name,
                "key": uploaded.key,
                "sha256": digest,
                "size_bytes": uploaded.size_bytes,
                "uri": uploaded.uri,
            }
        )
    manifest_hash = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "runtime_id": runtime_id,
        "artifact_count": len(records),
        "manifest_hash": manifest_hash,
        "artifacts": records,
    }
    (runtime_dir / "archive-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


async def verify_runtime_archive(runtime_dir: Path, store: ObjectStore) -> dict[str, object]:
    manifest_path = runtime_dir / "archive-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for record in manifest.get("artifacts", []):
        remote = await store.head(str(record["key"]))
        if remote is None:
            errors.append(f"missing:{record['source_name']}")
        elif remote.sha256 != record["sha256"]:
            errors.append(f"checksum:{record['source_name']}")
    return {
        "runtime_id": manifest.get("runtime_id"),
        "valid": not errors,
        "errors": errors,
        "artifact_count": len(manifest.get("artifacts", [])),
    }
