"""Dependency-light verification for frozen experiment export directories."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

import orjson

from pmmc.infrastructure.object_store import sha256_file, validate_object_key

QUESTIONER_RUNTIME_EXPORT_VERSION = "questioner-runtime-export-v2"
QUESTIONER_V3_RUNTIME_EXPORT_VERSION = "questioner-runtime-export-v3"
PROGRAM_RUNTIME_EXPORT_VERSION = "compiler-program-runtime-export-v3"
STEP7_PROGRAM_RUNTIME_EXPORT_VERSION = "compiler-program-runtime-export-v4"
STEP7B_PROGRAM_RUNTIME_EXPORT_VERSION = "compiler-program-runtime-export-v5"
SUPPORTED_PROGRAM_RUNTIME_EXPORT_VERSIONS = frozenset(
    {
        "compiler-program-runtime-export-v1",
        "compiler-program-runtime-export-v2",
        PROGRAM_RUNTIME_EXPORT_VERSION,
        STEP7_PROGRAM_RUNTIME_EXPORT_VERSION,
        STEP7B_PROGRAM_RUNTIME_EXPORT_VERSION,
    }
)
MODEL_ARTIFACT_BUNDLE_VERSION = "model-artifact-bundle-v1"


def _json_bytes(value: Any) -> bytes:
    return orjson.dumps(
        value,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z | orjson.OPT_NON_STR_KEYS,
    )


def _validate_run_path_component(run_id: str) -> str:
    path = PurePosixPath(run_id)
    if (
        len(path.parts) != 1
        or path.name != run_id
        or run_id in {".", ".."}
        or "\\" in run_id
    ):
        raise ValueError("run_id must be a safe filesystem path component")
    return run_id


def verify_experiment_export(export_dir: Path) -> dict[str, Any]:
    """Verify a deterministic export without importing database/vector tooling."""

    manifest_path = export_dir / "export_manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("export manifest must contain a file list")
    _validate_run_path_component(str(manifest.get("run_id", "")))
    files = manifest["files"]
    if not all(isinstance(item, dict) for item in files):
        raise ValueError("export manifest files must contain objects")
    if manifest.get("file_count") != len(files):
        raise ValueError("export manifest file_count is inconsistent")
    listed_paths = [str(item.get("path", "")) for item in files]
    if len(listed_paths) != len(set(listed_paths)):
        raise ValueError("export manifest contains duplicate file paths")
    actual_paths = sorted(
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if actual_paths != sorted(listed_paths):
        raise ValueError("export directory contents differ from its manifest")
    export_root = export_dir.resolve()
    for item, relative in zip(files, listed_paths, strict=True):
        safe_relative = validate_object_key(relative)
        if safe_relative != relative:
            raise ValueError(f"export path is not canonical: {relative}")
        path = export_dir / safe_relative
        if path.is_symlink() or not path.resolve().is_relative_to(export_root):
            raise ValueError(f"export path escapes its directory: {relative}")
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"export file is missing or has the wrong size: {relative}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"export checksum mismatch: {relative}")
    content = {key: value for key, value in manifest.items() if key != "export_id"}
    expected = hashlib.sha256(_json_bytes(content)).hexdigest()
    if expected != manifest["export_id"]:
        raise ValueError("export manifest ID does not match its content")
    return manifest
