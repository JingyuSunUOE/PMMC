"""Deterministic canonical identifiers and content hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pmmc.schemas import DatasetName


def _component(value: str | int) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("canonical ID components must be non-empty")
    return quote(normalized, safe="-_.")


def canonical_id(dataset: DatasetName, kind: str, *parts: str | int) -> str:
    """Build a readable, namespaced ID without machine-specific paths."""

    encoded = [_component(dataset.value), _component(kind), *(_component(part) for part in parts)]
    return ":".join(encoded)


def stable_hash(value: Any) -> str:
    """Hash JSON-compatible data using canonical serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_snapshot_id(
    dataset: DatasetName,
    *,
    source_content_hash: str,
    canonical_content_hash: str,
    schema_version: str,
    adapter_name: str,
    adapter_version: str,
    dataset_version: str,
    evaluation_policy_version: str,
    excluded_source_categories: list[str],
) -> str:
    """Derive the immutable snapshot ID from all versioned import inputs."""

    fingerprint = stable_hash(
        {
            "source": source_content_hash,
            "canonical": canonical_content_hash,
            "schema_version": schema_version,
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "dataset_version": dataset_version,
            "evaluation_policy_version": evaluation_policy_version,
            "excluded_source_categories": excluded_source_categories,
        }
    )
    return f"{dataset.value}-{fingerprint[:16]}"


def legacy_canonical_snapshot_id(
    dataset: DatasetName,
    *,
    source_content_hash: str,
    canonical_content_hash: str,
) -> str:
    """Reproduce the pre-policy snapshot fingerprint for local 0.3.0 artifacts."""

    fingerprint = stable_hash(
        {
            "source": source_content_hash,
            "canonical": canonical_content_hash,
        }
    )
    return f"{dataset.value}-{fingerprint[:16]}"


def path_id(dataset: DatasetName, kind: str, relative_path: str) -> str:
    """Build a compact canonical ID from a dataset-relative path."""

    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return canonical_id(dataset, kind, digest)


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 without loading the whole file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
