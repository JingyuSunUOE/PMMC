"""Explicit runtime-bucket archive and verification for derived corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pmmc.baselines.corpus import validate_corpus
from pmmc.infrastructure.object_store import (
    ObjectMetadata,
    ObjectStore,
    corpus_object_key,
    sha256_file,
)

CORPUS_ARCHIVE_FILES = (
    "manifest.json",
    "windows.jsonl",
    "documents.jsonl",
    "validation_report.json",
)


@dataclass(frozen=True)
class CorpusArchiveResult:
    corpus_id: str
    objects: tuple[ObjectMetadata, ...]


async def archive_corpus(corpus_path: Path, store: ObjectStore) -> CorpusArchiveResult:
    manifest = validate_corpus(corpus_path)
    values: list[ObjectMetadata] = []
    for file_name in CORPUS_ARCHIVE_FILES:
        source = corpus_path / file_name
        digest = sha256_file(source)
        values.append(
            await store.put_file(
                source,
                corpus_object_key(
                    manifest.dataset.value,
                    manifest.snapshot_id,
                    manifest.corpus_id,
                    file_name,
                ),
                expected_sha256=digest,
                content_type=(
                    "application/x-ndjson" if file_name.endswith(".jsonl") else "application/json"
                ),
            )
        )
    return CorpusArchiveResult(manifest.corpus_id, tuple(values))


async def verify_corpus_archive(corpus_path: Path, store: ObjectStore) -> CorpusArchiveResult:
    manifest = validate_corpus(corpus_path)
    values: list[ObjectMetadata] = []
    for file_name in CORPUS_ARCHIVE_FILES:
        key = corpus_object_key(
            manifest.dataset.value,
            manifest.snapshot_id,
            manifest.corpus_id,
            file_name,
        )
        remote = await store.head(key)
        if remote is None:
            raise FileNotFoundError(f"archived corpus object is missing: {file_name}")
        local = corpus_path / file_name
        if remote.sha256 != sha256_file(local) or remote.size_bytes != local.stat().st_size:
            raise ValueError(f"archived corpus object checksum mismatch: {file_name}")
        values.append(remote)
    return CorpusArchiveResult(manifest.corpus_id, tuple(values))
