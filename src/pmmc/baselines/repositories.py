"""Local JSONL/file repositories matching the baseline persistence boundary."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

from pmmc.baselines.corpus import CorpusReader
from pmmc.schemas import (
    BaselineCorpusManifest,
    ContextAssemblyTrace,
    MemoryWindow,
    RetrievalDocument,
)


class JsonlCorpusRepository:
    """Read an immutable local corpus exchange package."""

    def __init__(self, corpus_path: Path) -> None:
        self.reader = CorpusReader(corpus_path)

    def get_manifest(self) -> BaselineCorpusManifest:
        return self.reader.manifest

    def iter_windows(self) -> Iterator[MemoryWindow]:
        yield from self.reader.iter_windows()

    def iter_documents(self) -> Iterator[RetrievalDocument]:
        yield from self.reader.iter_documents()


class JsonlContextAssemblyRepository:
    """One-file-per-ID local trace store with conflict-aware replay."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, context_assembly_id: str) -> Path:
        if Path(context_assembly_id).name != context_assembly_id or "\\" in context_assembly_id:
            raise ValueError("context_assembly_id must be a safe path component")
        return self.root / f"{context_assembly_id}.json"

    def add(self, value: ContextAssemblyTrace) -> ContextAssemblyTrace:
        target = self._path(value.context_assembly_id)
        payload = value.to_json() + "\n"
        if target.exists():
            existing = ContextAssemblyTrace.from_json(target.read_text(encoding="utf-8"))
            if existing != value:
                raise ValueError("context assembly ID already has different content")
            return existing
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.root,
            prefix=".context-",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        temporary.replace(target)
        return value

    def get(self, context_assembly_id: str) -> ContextAssemblyTrace:
        path = self._path(context_assembly_id)
        if not path.is_file():
            raise KeyError(f"context assembly does not exist: {context_assembly_id}")
        return ContextAssemblyTrace.from_json(path.read_text(encoding="utf-8"))
