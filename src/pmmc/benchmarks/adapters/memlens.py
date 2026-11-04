"""MEMLENS 32K source adapter."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import ijson
from pydantic import BaseModel, ConfigDict, Field

from pmmc.benchmarks.base import BenchmarkAdapter, BenchmarkDataError
from pmmc.benchmarks.ids import canonical_id, path_id
from pmmc.benchmarks.models import AdaptedQuery, CanonicalConversation
from pmmc.benchmarks.paths import project_relative_path, resolve_within
from pmmc.benchmarks.taxonomy import evaluation_eligibility, map_question_type
from pmmc.schemas import (
    DatasetName,
    GoldEvidence,
    ImageRef,
    MemoryUnit,
    Modality,
    QueryGold,
    QueryInput,
    QueryRecord,
    SessionRecord,
    SourceLocator,
    SpeakerRole,
)

EMPTY_SOURCE_TURN_TEXT = "[EMPTY SOURCE TURN]"


class _RawModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class _RawImage(_RawModel):
    file: str
    image_url: str = ""
    blip_caption: str = ""


class _RawTurn(_RawModel):
    role: str
    content: str
    images: list[_RawImage] = Field(default_factory=list)


class _RawRecord(_RawModel):
    question_id: str
    question_type: str
    question_subtype: str
    question: str
    answer: str
    question_date: str
    haystack_dates: list[str]
    haystack_session_ids: list[str]
    haystack_sessions: list[list[_RawTurn]]
    answer_session_ids: list[str] = Field(default_factory=list)


def _logical_datetime(raw: str) -> str:
    match = re.fullmatch(
        r"(?P<date>\d{4}/\d{2}/\d{2}) \([A-Za-z]{3}\) (?P<time>\d{2}:\d{2})",
        raw,
    )
    if match is None:
        raise BenchmarkDataError(f"invalid MEMLENS logical datetime: {raw}")
    try:
        value = datetime.strptime(
            f"{match.group('date')} {match.group('time')}",
            "%Y/%m/%d %H:%M",
        )
    except ValueError as exc:
        raise BenchmarkDataError(f"invalid MEMLENS logical datetime: {raw}") from exc
    return value.isoformat(timespec="minutes")


def _speaker(raw: str) -> SpeakerRole:
    normalized = raw.strip().lower()
    if normalized == "user":
        return SpeakerRole.USER
    if normalized in {"assistant", "ai assistant"}:
        return SpeakerRole.ASSISTANT
    if normalized == "system":
        return SpeakerRole.SYSTEM
    raise BenchmarkDataError(f"unsupported MEMLENS speaker role: {raw}")


class MemLensAdapter(BenchmarkAdapter):
    """Normalize the retained 32K MEMLENS package."""

    dataset = DatasetName.MEMLENS
    adapter_name = "memlens_32k"
    adapter_version = "1.0.0"
    dataset_version = "32k-789-local"

    @property
    def source_record_count(self) -> int:
        return 789

    def source_files(self) -> list[Path]:
        files = [self.root / "data" / "dataset_32k.json"]
        files.extend(sorted((self.root / "prompts" / "judge").glob("*.py")))
        return files

    def _iter_raw_records(self) -> Iterator[tuple[int, _RawRecord]]:
        source_path = self.root / "data" / "dataset_32k.json"
        if not source_path.is_file():
            raise BenchmarkDataError(f"MEMLENS source file is missing: {source_path}")
        with source_path.open("rb") as stream:
            for index, payload in enumerate(ijson.items(stream, "item")):
                yield index, _RawRecord.model_validate(payload)

    def iter_conversations(self) -> Iterator[CanonicalConversation]:
        for record_index, raw in self._iter_raw_records():
            yield self._adapt_record(record_index, raw)

    def _adapt_record(self, record_index: int, raw: _RawRecord) -> CanonicalConversation:
        if not (
            len(raw.haystack_dates) == len(raw.haystack_session_ids) == len(raw.haystack_sessions)
        ):
            raise BenchmarkDataError(
                f"MEMLENS session arrays differ in length for {raw.question_id}"
            )
        if len(raw.haystack_session_ids) != len(set(raw.haystack_session_ids)):
            raise BenchmarkDataError(f"MEMLENS source session IDs repeat within {raw.question_id}")

        conversation_id = canonical_id(self.dataset, "conversation", raw.question_id)
        sessions: list[SessionRecord] = []
        memory_units: list[MemoryUnit] = []
        source_to_session: dict[str, str] = {}
        timestamp_order = 0

        for session_order, (source_session_id, source_date, turns) in enumerate(
            zip(raw.haystack_session_ids, raw.haystack_dates, raw.haystack_sessions, strict=True)
        ):
            session_id = canonical_id(
                self.dataset,
                "session",
                raw.question_id,
                source_session_id,
            )
            source_to_session[source_session_id] = session_id
            sessions.append(
                SessionRecord(
                    session_id=session_id,
                    dataset=self.dataset,
                    conversation_id=conversation_id,
                    session_order=session_order,
                    source_session_id=source_session_id,
                    source_date_raw=source_date,
                    source_date_iso=_logical_datetime(source_date),
                )
            )

            for source_turn_index, turn in enumerate(turns):
                source = SourceLocator(
                    dataset=self.dataset,
                    source_file="data/dataset_32k.json",
                    source_record_index=record_index,
                    source_query_id=raw.question_id,
                    source_session_id=source_session_id,
                    source_turn_index=source_turn_index,
                    source_category=raw.question_type,
                    source_subtype=raw.question_subtype,
                )
                image_refs = [self._image_ref(image) for image in turn.images]
                captions = [image.blip_caption.strip() for image in turn.images]
                captions = [caption for caption in captions if caption]
                text = turn.content.strip() or None
                source_content_empty = text is None and not captions and not image_refs
                if source_content_empty:
                    text = EMPTY_SOURCE_TURN_TEXT

                modalities = [Modality.TEXT] if text is not None else []
                if captions:
                    modalities.append(Modality.CAPTION)
                if image_refs:
                    modalities.append(Modality.RAW_IMAGE)

                turn_id = canonical_id(
                    self.dataset,
                    "turn",
                    raw.question_id,
                    source_session_id,
                    source_turn_index,
                )
                memory_units.append(
                    MemoryUnit(
                        memory_id=canonical_id(
                            self.dataset,
                            "memory",
                            raw.question_id,
                            source_session_id,
                            source_turn_index,
                        ),
                        dataset=self.dataset,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        timestamp_order=timestamp_order,
                        speaker=_speaker(turn.role),
                        text=text,
                        caption="\n\n".join(captions) or None,
                        image_ids=[image_ref.image_id for image_ref in image_refs],
                        image_refs=image_refs,
                        modality=modalities,
                        available_after_session=session_id,
                        metadata={
                            "unit_kind": "speaker_turn",
                            "caption_source": "benchmark_blip" if captions else None,
                            "source_content_empty": source_content_empty,
                            "source_session_date_raw": source_date,
                            "source_session_date_iso": _logical_datetime(source_date),
                            "source": source.to_dict(),
                        },
                    )
                )
                timestamp_order += 1

        unknown_answer_sessions = set(raw.answer_session_ids) - set(source_to_session)
        if unknown_answer_sessions:
            raise BenchmarkDataError(
                f"MEMLENS answer sessions do not resolve for {raw.question_id}: "
                f"{sorted(unknown_answer_sessions)}"
            )
        if not sessions:
            raise BenchmarkDataError(f"MEMLENS query has no visible sessions: {raw.question_id}")

        query_id = canonical_id(self.dataset, "query", raw.question_id)
        question_type = map_question_type(self.dataset, raw.question_type)
        query = QueryInput(
            query_id=query_id,
            dataset=self.dataset,
            conversation_id=conversation_id,
            raw_query=raw.question,
            query_type=question_type,
            query_subtype=raw.question_subtype,
            available_after_session=sessions[-1].session_id,
            metadata={
                "question_date_raw": raw.question_date,
                "question_date_iso": _logical_datetime(raw.question_date),
                "source_category": raw.question_type,
                "source_subtype": raw.question_subtype,
            },
        )
        gold = QueryGold(
            answer=raw.answer,
            evidence=GoldEvidence(
                session_ids=[source_to_session[item] for item in raw.answer_session_ids]
            ),
            metadata={"source_answer_session_ids": raw.answer_session_ids},
        )
        query_source = SourceLocator(
            dataset=self.dataset,
            source_file="data/dataset_32k.json",
            source_record_index=record_index,
            source_query_id=raw.question_id,
            source_category=raw.question_type,
            source_subtype=raw.question_subtype,
        )
        adapted_query = AdaptedQuery(
            record=QueryRecord(query=query, gold=gold),
            source=query_source,
            eligibility=evaluation_eligibility(self.dataset, raw.question_type),
        )
        return CanonicalConversation(
            dataset=self.dataset,
            conversation_id=conversation_id,
            sessions=sessions,
            memory_units=memory_units,
            queries=[adapted_query],
        )

    def _image_ref(self, raw: _RawImage) -> ImageRef:
        path = resolve_within(self.root / "images", raw.file)
        dataset_relative = path.relative_to(self.root).as_posix()
        caption = raw.blip_caption.strip() or None
        return ImageRef(
            image_id=path_id(self.dataset, "image", dataset_relative),
            asset_id=path_id(self.dataset, "asset", dataset_relative),
            local_path=project_relative_path(path, self.project_root),
            remote_url=raw.image_url.strip() or None,
            caption=caption,
            metadata={
                "dataset_relative_path": dataset_relative,
                "caption_source": "benchmark_blip" if caption else None,
            },
        )

    def expected_counts(self) -> dict[str, int]:
        return {
            "source_records": 789,
            "sessions": 10616,
            "memory_units": 97446,
            "queries": 789,
            "eligible_queries": 699,
            "excluded_queries": 90,
            "assets": 4695,
            "referenced_unique_images": 4090,
        }
