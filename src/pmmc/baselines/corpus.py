"""Atomic construction and validation of snapshot-derived baseline corpora."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from pmmc.baselines.rendering import (
    RENDERER_VERSION,
    render_memory_text,
    render_memory_text_caption,
)
from pmmc.benchmarks.canonical import (
    iter_jsonl,
    load_snapshot_manifest,
    preflight_snapshot,
)
from pmmc.benchmarks.ids import stable_hash
from pmmc.models.protocols import TokenCounter
from pmmc.schemas import (
    BaselineCorpusManifest,
    BaselineCorpusSpec,
    BaseSchema,
    CorpusStatus,
    MemoryUnit,
    MemoryWindow,
    RepresentationKind,
    RetrievalDocument,
    SessionRecord,
)

CORPUS_FILES = ("windows.jsonl", "documents.jsonl")
_EVALUATION_ONLY_KEYS = frozenset(
    {
        "answer_session_ids",
        "clue",
        "gold",
        "gold_answer",
        "gold_evidence",
        "gold_image_ids",
        "gold_memory_ids",
        "gold_session_ids",
        "is_gold_evidence",
        "matched_gold_evidence_ids",
    }
)
SchemaT = TypeVar("SchemaT", bound=BaseSchema)


class CorpusIntegrityError(RuntimeError):
    """A derived corpus failed deterministic lineage or leakage checks."""


def _assert_runtime_safe_json(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in _EVALUATION_ONLY_KEYS:
                raise CorpusIntegrityError(
                    f"corpus contains evaluation-only field: {key}"
                )
            _assert_runtime_safe_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_runtime_safe_json(item)


def _json_bytes(value: BaseSchema) -> bytes:
    return (value.to_json() + "\n").encode("utf-8")


def _hash_records(values: Iterable[BaseSchema]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_json_bytes(value))
    return digest.hexdigest()


def corpus_id_for(spec: BaselineCorpusSpec) -> str:
    fingerprint = stable_hash(spec.model_dump(mode="json"))
    return f"{spec.dataset.value}-corpus-{fingerprint[:20]}"


def _window_id(corpus_id: str, session_id: str, ordinal: int, memory_ids: list[str]) -> str:
    fingerprint = stable_hash(
        {
            "corpus_id": corpus_id,
            "session_id": session_id,
            "ordinal": ordinal,
            "memory_ids": memory_ids,
        }
    )
    return f"window-{fingerprint[:24]}"


def _document_id(
    corpus_id: str,
    window_id: str,
    representation: RepresentationKind,
    image_id: str | None = None,
) -> str:
    fingerprint = stable_hash(
        {
            "corpus_id": corpus_id,
            "window_id": window_id,
            "representation": representation.value,
            "image_id": image_id,
        }
    )
    return f"document-{fingerprint[:24]}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _memory_image_ids(unit: MemoryUnit) -> list[str]:
    return _unique([*unit.image_ids, *(image.image_id for image in unit.image_refs)])


def _window_session(
    units: list[MemoryUnit],
    *,
    corpus_id: str,
    spec: BaselineCorpusSpec,
    counter: TokenCounter,
    session_orders: dict[tuple[str, str], int],
) -> list[MemoryWindow]:
    if not units:
        return []
    units = sorted(units, key=lambda item: item.timestamp_order)
    orders = [item.timestamp_order for item in units]
    if len(orders) != len(set(orders)):
        raise CorpusIntegrityError(f"duplicate timestamp order in session: {units[0].session_id}")
    windows: list[MemoryWindow] = []
    start = 0
    while start < len(units):
        selected: list[MemoryUnit] = []
        token_total = 0
        image_ids: list[str] = []
        index = start
        while index < len(units):
            unit = units[index]
            unit_tokens = counter.count_text(render_memory_text_caption(unit))
            candidate_images = _unique([*image_ids, *_memory_image_ids(unit)])
            exceeds = (
                token_total + unit_tokens > spec.max_text_tokens
                or len(candidate_images) > spec.max_images
            )
            if selected and exceeds:
                break
            selected.append(unit)
            token_total += unit_tokens
            image_ids = candidate_images
            index += 1
            if exceeds:
                break
        memory_ids = [unit.memory_id for unit in selected]
        ordinal = len(windows)
        oversized = len(selected) == 1 and (
            token_total > spec.max_text_tokens or len(image_ids) > spec.max_images
        )
        try:
            available_after_session = max(
                (unit.available_after_session for unit in selected),
                key=lambda session_id: session_orders[(units[0].conversation_id, session_id)],
            )
        except KeyError as exc:
            raise CorpusIntegrityError(
                f"memory references an unknown availability session: {exc.args[0][1]}"
            ) from exc
        windows.append(
            MemoryWindow(
                window_id=_window_id(corpus_id, units[0].session_id, ordinal, memory_ids),
                corpus_id=corpus_id,
                snapshot_id=spec.snapshot_id,
                dataset=spec.dataset,
                conversation_id=units[0].conversation_id,
                session_id=units[0].session_id,
                window_ordinal=ordinal,
                memory_ids=memory_ids,
                start_timestamp_order=selected[0].timestamp_order,
                end_timestamp_order=selected[-1].timestamp_order,
                text_caption_token_count=token_total,
                image_ids=image_ids,
                available_after_session=available_after_session,
                oversized_single_unit=oversized,
            )
        )
        if index >= len(units):
            break
        overlap = min(spec.overlap_memory_units, max(0, len(selected) - 1))
        start = index - overlap
    return windows


def _documents_for_window(
    window: MemoryWindow,
    units_by_id: dict[str, MemoryUnit],
    spec: BaselineCorpusSpec,
    counter: TokenCounter,
) -> list[RetrievalDocument]:
    units = [units_by_id[memory_id] for memory_id in window.memory_ids]
    text_caption = "\n\n".join(render_memory_text_caption(unit) for unit in units)
    text = "\n\n".join(render_memory_text(unit) for unit in units)
    values: list[RetrievalDocument] = []

    def add(
        representation: RepresentationKind,
        *,
        rendered_text: str | None,
        image_ids: list[str],
        image_key: str | None = None,
        memory_ids: list[str] | None = None,
    ) -> None:
        content_hash = stable_hash(
            {
                "window_id": window.window_id,
                "representation": representation.value,
                "text": rendered_text,
                "image_ids": image_ids,
                "renderer_version": spec.renderer_version,
            }
        )
        values.append(
            RetrievalDocument(
                document_id=_document_id(
                    window.corpus_id,
                    window.window_id,
                    representation,
                    image_key,
                ),
                corpus_id=window.corpus_id,
                snapshot_id=window.snapshot_id,
                dataset=window.dataset,
                conversation_id=window.conversation_id,
                session_id=window.session_id,
                parent_window_id=window.window_id,
                representation=representation,
                memory_ids=memory_ids or window.memory_ids,
                image_ids=image_ids,
                rendered_text=rendered_text,
                token_count=(counter.count_text(rendered_text) if rendered_text else None),
                available_after_session=window.available_after_session,
                content_hash=content_hash,
            )
        )

    if RepresentationKind.TEXT_CAPTION in spec.representations:
        add(RepresentationKind.TEXT_CAPTION, rendered_text=text_caption, image_ids=window.image_ids)
    if RepresentationKind.TEXT in spec.representations:
        add(RepresentationKind.TEXT, rendered_text=text, image_ids=[])
    if RepresentationKind.IMAGE in spec.representations:
        for image_id in window.image_ids:
            image_units = [
                unit.memory_id for unit in units if image_id in _memory_image_ids(unit)
            ]
            add(
                RepresentationKind.IMAGE,
                rendered_text=None,
                image_ids=[image_id],
                image_key=image_id,
                memory_ids=image_units,
            )
    if RepresentationKind.MULTIMODAL in spec.representations:
        add(
            RepresentationKind.MULTIMODAL,
            rendered_text=text,
            image_ids=window.image_ids,
        )
    return values


def _write_jsonl(path: Path, values: Sequence[BaseSchema]) -> None:
    with path.open("wb") as stream:
        for value in values:
            stream.write(_json_bytes(value))


def build_corpus(
    snapshot_path: Path,
    *,
    output_root: Path,
    counter: TokenCounter,
    spec: BaselineCorpusSpec | None = None,
    progress: Any | None = None,
    verify_snapshot: bool = True,
) -> tuple[Path, BaselineCorpusManifest]:
    """Build one immutable corpus using runtime-safe snapshot files only."""

    snapshot = (
        preflight_snapshot(snapshot_path).manifest
        if verify_snapshot
        else load_snapshot_manifest(snapshot_path)
    )
    resolved_spec = spec or BaselineCorpusSpec(
        snapshot_id=snapshot.snapshot_id,
        dataset=snapshot.dataset,
        snapshot_content_hash=snapshot.canonical_content_hash,
        tokenizer_alias=counter.alias,
        tokenizer_revision=counter.revision,
        renderer_version=RENDERER_VERSION,
    )
    if (
        resolved_spec.snapshot_id != snapshot.snapshot_id
        or resolved_spec.dataset is not snapshot.dataset
        or resolved_spec.snapshot_content_hash != snapshot.canonical_content_hash
    ):
        raise CorpusIntegrityError("corpus spec does not match the canonical snapshot")
    if (
        resolved_spec.tokenizer_alias != counter.alias
        or resolved_spec.tokenizer_revision != counter.revision
    ):
        raise CorpusIntegrityError("token counter identity does not match corpus spec")

    corpus_id = corpus_id_for(resolved_spec)
    final_path = output_root / snapshot.dataset.value / snapshot.snapshot_id / corpus_id
    if final_path.exists():
        return final_path, validate_corpus(final_path)

    session_orders: dict[tuple[str, str], int] = {}
    for session in iter_jsonl(snapshot_path / "sessions.jsonl", SessionRecord):
        key = (session.conversation_id, session.session_id)
        if key in session_orders:
            raise CorpusIntegrityError(f"duplicate canonical session ID: {session.session_id}")
        session_orders[key] = session.session_order

    grouped: OrderedDict[tuple[str, str], list[MemoryUnit]] = OrderedDict()
    units_by_id: dict[str, MemoryUnit] = {}
    for unit in iter_jsonl(snapshot_path / "memory_units.jsonl", MemoryUnit):
        if unit.memory_id in units_by_id:
            raise CorpusIntegrityError(f"duplicate memory ID: {unit.memory_id}")
        units_by_id[unit.memory_id] = unit
        grouped.setdefault((unit.conversation_id, unit.session_id), []).append(unit)

    windows: list[MemoryWindow] = []
    for group_index, units in enumerate(grouped.values(), start=1):
        windows.extend(
            _window_session(
                units,
                corpus_id=corpus_id,
                spec=resolved_spec,
                counter=counter,
                session_orders=session_orders,
            )
        )
        if progress is not None:
            progress("window_sessions", group_index, len(grouped))
    windows.sort(
        key=lambda item: (
            item.conversation_id,
            item.start_timestamp_order,
            item.window_id,
        )
    )
    documents: list[RetrievalDocument] = []
    for window_index, window in enumerate(windows, start=1):
        documents.extend(_documents_for_window(window, units_by_id, resolved_spec, counter))
        if progress is not None:
            progress("render_documents", window_index, len(windows))
    documents.sort(key=lambda item: item.document_id)
    windows_hash = _hash_records(windows)
    documents_hash = _hash_records(documents)
    content_hash = stable_hash(
        {
            "corpus_id": corpus_id,
            "spec": resolved_spec.model_dump(mode="json"),
            "windows_hash": windows_hash,
            "documents_hash": documents_hash,
        }
    )
    manifest = BaselineCorpusManifest(
        corpus_id=corpus_id,
        snapshot_id=snapshot.snapshot_id,
        dataset=snapshot.dataset,
        spec=resolved_spec,
        status=CorpusStatus.READY,
        content_hash=content_hash,
        windows_hash=windows_hash,
        documents_hash=documents_hash,
        window_count=len(windows),
        document_count=len(documents),
        image_document_count=sum(
            item.representation is RepresentationKind.IMAGE for item in documents
        ),
        unique_memory_count=len(units_by_id),
        unique_image_count=len(
            {image for unit in units_by_id.values() for image in _memory_image_ids(unit)}
        ),
        created_at=snapshot.created_at,
        metadata={"token_count_estimated": counter.estimated},
    )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=".corpus-", dir=final_path.parent))
    try:
        _write_jsonl(temp_path / "windows.jsonl", windows)
        _write_jsonl(temp_path / "documents.jsonl", documents)
        (temp_path / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
        report = {
            "success": True,
            "corpus_id": corpus_id,
            "window_count": len(windows),
            "document_count": len(documents),
            "unique_memory_count": len(units_by_id),
        }
        (temp_path / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temp_path.rename(final_path)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return final_path, validate_corpus(final_path)


def _iter_jsonl(path: Path, schema: type[SchemaT]) -> Iterator[SchemaT]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        yield schema.from_json(line)
                    except ValidationError as exc:
                        raise CorpusIntegrityError(
                            f"invalid {path.name} at line {line_number}"
                        ) from exc
    except OSError as exc:
        raise CorpusIntegrityError(f"cannot read {path.name}") from exc


def validate_corpus(corpus_path: Path) -> BaselineCorpusManifest:
    """Validate hashes, counts, lineage, and runtime-only content."""

    required = ("manifest.json", "validation_report.json", *CORPUS_FILES)
    if not corpus_path.is_dir() or any(
        not (corpus_path / name).is_file() for name in required
    ):
        raise CorpusIntegrityError("corpus directory is incomplete")
    try:
        manifest = BaselineCorpusManifest.from_json(
            (corpus_path / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise CorpusIntegrityError("invalid corpus manifest") from exc
    if manifest.corpus_id != corpus_path.name:
        raise CorpusIntegrityError("corpus directory name does not match manifest")
    if manifest.status is not CorpusStatus.READY:
        raise CorpusIntegrityError("local corpus manifest must be ready")
    windows = list(_iter_jsonl(corpus_path / "windows.jsonl", MemoryWindow))
    if len(windows) != manifest.window_count:
        raise CorpusIntegrityError("corpus counts do not match manifest")
    if _hash_records(windows) != manifest.windows_hash:
        raise CorpusIntegrityError("corpus JSONL hash does not match manifest")
    expected_content = stable_hash(
        {
            "corpus_id": manifest.corpus_id,
            "spec": manifest.spec.model_dump(mode="json"),
            "windows_hash": manifest.windows_hash,
            "documents_hash": manifest.documents_hash,
        }
    )
    if (
        expected_content != manifest.content_hash
        or corpus_id_for(manifest.spec) != manifest.corpus_id
    ):
        raise CorpusIntegrityError("corpus identity does not match its lineage")
    window_ids = [window.window_id for window in windows]
    if len(window_ids) != len(set(window_ids)):
        raise CorpusIntegrityError("corpus contains duplicate IDs")
    windows_by_id = {window.window_id: window for window in windows}
    memory_ids = {memory_id for window in windows for memory_id in window.memory_ids}
    image_ids = {image_id for window in windows for image_id in window.image_ids}
    if len(memory_ids) != manifest.unique_memory_count:
        raise CorpusIntegrityError("corpus unique memory count does not match its windows")
    if len(image_ids) != manifest.unique_image_count:
        raise CorpusIntegrityError("corpus unique image count does not match its windows")
    grouped_ordinals: dict[tuple[str, str], list[int]] = {}
    for window in windows:
        if (
            window.corpus_id != manifest.corpus_id
            or window.snapshot_id != manifest.snapshot_id
            or window.dataset is not manifest.dataset
        ):
            raise CorpusIntegrityError("memory window escapes corpus snapshot lineage")
        grouped_ordinals.setdefault(
            (window.conversation_id, window.session_id), []
        ).append(window.window_ordinal)
        _assert_runtime_safe_json(window.model_dump(mode="json"))
    if any(
        ordinals != list(range(len(ordinals)))
        for ordinals in grouped_ordinals.values()
    ):
        raise CorpusIntegrityError("memory window ordinals are not contiguous per session")
    document_digest = hashlib.sha256()
    document_ids: set[str] = set()
    document_count = 0
    image_document_count = 0
    for document in _iter_jsonl(
        corpus_path / "documents.jsonl", RetrievalDocument
    ):
        document_count += 1
        if document.document_id in document_ids:
            raise CorpusIntegrityError("corpus contains duplicate IDs")
        document_ids.add(document.document_id)
        document_digest.update(_json_bytes(document))
        parent_window = windows_by_id.get(document.parent_window_id)
        if (
            parent_window is None
            or document.corpus_id != manifest.corpus_id
            or document.snapshot_id != manifest.snapshot_id
            or document.dataset is not manifest.dataset
            or document.representation not in manifest.spec.representations
        ):
            raise CorpusIntegrityError("retrieval document has invalid window lineage")
        if not set(document.memory_ids).issubset(parent_window.memory_ids):
            raise CorpusIntegrityError("retrieval document memory lineage escapes its window")
        if not set(document.image_ids).issubset(parent_window.image_ids):
            raise CorpusIntegrityError("retrieval document image lineage escapes its window")
        if document.representation is RepresentationKind.IMAGE:
            image_document_count += 1
        _assert_runtime_safe_json(document.model_dump(mode="json"))
    if document_count != manifest.document_count:
        raise CorpusIntegrityError("corpus counts do not match manifest")
    if document_digest.hexdigest() != manifest.documents_hash:
        raise CorpusIntegrityError("corpus JSONL hash does not match manifest")
    if image_document_count != manifest.image_document_count:
        raise CorpusIntegrityError("image document count does not match manifest")
    try:
        validation_report = json.loads(
            (corpus_path / "validation_report.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusIntegrityError("invalid corpus validation report") from exc
    _assert_runtime_safe_json(manifest.model_dump(mode="json"))
    _assert_runtime_safe_json(validation_report)
    return manifest


class CorpusReader:
    """Local immutable reader mirroring the future Postgres repository."""

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path
        self.manifest = validate_corpus(corpus_path)

    def iter_windows(self) -> Iterator[MemoryWindow]:
        yield from _iter_jsonl(self.corpus_path / "windows.jsonl", MemoryWindow)

    def iter_documents(
        self,
        representation: RepresentationKind | None = None,
    ) -> Iterator[RetrievalDocument]:
        for document in _iter_jsonl(
            self.corpus_path / "documents.jsonl", RetrievalDocument
        ):
            if representation is None or document.representation is representation:
                yield document
