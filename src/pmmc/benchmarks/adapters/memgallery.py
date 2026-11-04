"""MemGallery source adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

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


class _RawModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class _RawProfile(_RawModel):
    name: str
    persona_summary: str
    traits: list[str] = Field(default_factory=list)
    conversation_style: str


class _RawDialogue(_RawModel):
    round: str
    user: str
    assistant: str
    image_id: list[str] = Field(default_factory=list)
    input_image: list[str] = Field(default_factory=list)
    image_caption: list[str] = Field(default_factory=list)


class _RawSession(_RawModel):
    session_id: str
    date: str
    dialogues: list[_RawDialogue]


class _RawQA(_RawModel):
    point: str
    question: str
    answer: str
    session_id: list[str] = Field(default_factory=list)
    clue: list[str] = Field(default_factory=list)
    question_image: str | None = None
    image_caption: str | None = None


class _RawScenario(_RawModel):
    character_profile: _RawProfile
    multi_session_dialogues: list[_RawSession]
    human_annotated_qas: list[_RawQA] = Field(alias="human-annotated QAs")


def _logical_date(raw: str) -> str:
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise BenchmarkDataError(f"invalid MemGallery logical date: {raw}") from exc


class MemGalleryAdapter(BenchmarkAdapter):
    """Normalize the retained 20-scenario MemGallery package."""

    dataset = DatasetName.MEMGALLERY
    adapter_name = "memgallery"
    adapter_version = "1.0.0"
    dataset_version = "20-scenario-local"

    def _dialog_files(self) -> list[Path]:
        return sorted((self.root / "data" / "dialogs").glob("*.json"))

    @property
    def source_record_count(self) -> int:
        return len(self._dialog_files())

    def source_files(self) -> list[Path]:
        files = self._dialog_files()
        files.extend(sorted((self.root / "prompts" / "source").glob("*.txt")))
        return files

    def iter_conversations(self) -> Iterator[CanonicalConversation]:
        for scenario_index, path in enumerate(self._dialog_files()):
            with path.open(encoding="utf-8") as stream:
                raw = _RawScenario.model_validate(json.load(stream))
            yield self._adapt_scenario(scenario_index, path, raw)

    def _adapt_scenario(
        self,
        scenario_index: int,
        path: Path,
        raw: _RawScenario,
    ) -> CanonicalConversation:
        scenario_id = path.stem
        source_file = f"data/dialogs/{path.name}"
        conversation_id = canonical_id(self.dataset, "conversation", scenario_id)
        profile_session_id = canonical_id(self.dataset, "session", scenario_id, "profile")
        sessions = [
            SessionRecord(
                session_id=profile_session_id,
                dataset=self.dataset,
                conversation_id=conversation_id,
                session_order=0,
                source_session_id="profile",
                is_synthetic=True,
                metadata={"unit_kind": "character_profile"},
            )
        ]
        profile_source = SourceLocator(
            dataset=self.dataset,
            source_file=source_file,
            source_record_index=scenario_index,
            scenario_id=scenario_id,
            source_session_id="profile",
        )
        memory_units = [
            MemoryUnit(
                memory_id=canonical_id(self.dataset, "memory", scenario_id, "profile"),
                dataset=self.dataset,
                conversation_id=conversation_id,
                session_id=profile_session_id,
                turn_id=canonical_id(self.dataset, "turn", scenario_id, "profile"),
                timestamp_order=0,
                speaker=SpeakerRole.SYSTEM,
                text=self._profile_text(raw.character_profile),
                modality=[Modality.TEXT],
                available_after_session=profile_session_id,
                entity_mentions=[raw.character_profile.name],
                metadata={
                    "unit_kind": "character_profile",
                    "source": profile_source.to_dict(),
                },
            )
        ]

        session_by_source: dict[str, str] = {}
        round_to_memory_ids: dict[str, list[str]] = {}
        timestamp_order = 1
        for session_order, raw_session in enumerate(raw.multi_session_dialogues, start=1):
            if raw_session.session_id in session_by_source:
                raise BenchmarkDataError(
                    f"duplicate MemGallery session {raw_session.session_id} in {scenario_id}"
                )
            session_id = canonical_id(
                self.dataset,
                "session",
                scenario_id,
                raw_session.session_id,
            )
            session_by_source[raw_session.session_id] = session_id
            sessions.append(
                SessionRecord(
                    session_id=session_id,
                    dataset=self.dataset,
                    conversation_id=conversation_id,
                    session_order=session_order,
                    source_session_id=raw_session.session_id,
                    source_date_raw=raw_session.date,
                    source_date_iso=_logical_date(raw_session.date),
                )
            )

            for source_turn_index, dialogue in enumerate(raw_session.dialogues):
                if dialogue.round in round_to_memory_ids:
                    raise BenchmarkDataError(
                        f"duplicate MemGallery round {dialogue.round} in {scenario_id}"
                    )
                image_count = len(dialogue.image_id)
                if image_count != len(dialogue.input_image) or image_count != len(
                    dialogue.image_caption
                ):
                    raise BenchmarkDataError(
                        f"image arrays differ in length for {scenario_id}/{dialogue.round}"
                    )
                image_refs = [
                    self._history_image_ref(scenario_id, image_id, image_path, caption)
                    for image_id, image_path, caption in zip(
                        dialogue.image_id,
                        dialogue.input_image,
                        dialogue.image_caption,
                        strict=True,
                    )
                ]
                source = SourceLocator(
                    dataset=self.dataset,
                    source_file=source_file,
                    source_record_index=scenario_index,
                    scenario_id=scenario_id,
                    source_session_id=raw_session.session_id,
                    source_round_id=dialogue.round,
                    source_turn_index=source_turn_index,
                )
                user_id = canonical_id(
                    self.dataset,
                    "memory",
                    scenario_id,
                    dialogue.round,
                    "user",
                )
                assistant_id = canonical_id(
                    self.dataset,
                    "memory",
                    scenario_id,
                    dialogue.round,
                    "assistant",
                )
                captions = [
                    caption.strip() for caption in dialogue.image_caption if caption.strip()
                ]
                user_modalities = [Modality.TEXT]
                if captions:
                    user_modalities.append(Modality.CAPTION)
                if image_refs:
                    user_modalities.append(Modality.RAW_IMAGE)
                memory_units.append(
                    MemoryUnit(
                        memory_id=user_id,
                        dataset=self.dataset,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        turn_id=canonical_id(
                            self.dataset,
                            "turn",
                            scenario_id,
                            dialogue.round,
                            "user",
                        ),
                        timestamp_order=timestamp_order,
                        speaker=SpeakerRole.USER,
                        text=dialogue.user,
                        caption="\n\n".join(captions) or None,
                        image_ids=[image_ref.image_id for image_ref in image_refs],
                        image_refs=image_refs,
                        modality=user_modalities,
                        available_after_session=session_id,
                        metadata={
                            "unit_kind": "speaker_turn",
                            "source_round_id": dialogue.round,
                            "caption_source": "benchmark" if captions else None,
                            "source_session_date_raw": raw_session.date,
                            "source_session_date_iso": _logical_date(raw_session.date),
                            "source": source.to_dict(),
                        },
                    )
                )
                timestamp_order += 1
                memory_units.append(
                    MemoryUnit(
                        memory_id=assistant_id,
                        dataset=self.dataset,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        turn_id=canonical_id(
                            self.dataset,
                            "turn",
                            scenario_id,
                            dialogue.round,
                            "assistant",
                        ),
                        timestamp_order=timestamp_order,
                        speaker=SpeakerRole.ASSISTANT,
                        text=dialogue.assistant,
                        modality=[Modality.TEXT],
                        available_after_session=session_id,
                        metadata={
                            "unit_kind": "speaker_turn",
                            "source_round_id": dialogue.round,
                            "source_session_date_raw": raw_session.date,
                            "source_session_date_iso": _logical_date(raw_session.date),
                            "source": source.to_dict(),
                        },
                    )
                )
                timestamp_order += 1
                round_to_memory_ids[dialogue.round] = [user_id, assistant_id]

        if len(sessions) == 1:
            raise BenchmarkDataError(f"MemGallery scenario has no dialogue sessions: {scenario_id}")
        queries = [
            self._adapt_query(
                scenario_index=scenario_index,
                scenario_id=scenario_id,
                source_file=source_file,
                qa_index=qa_index,
                qa=qa,
                conversation_id=conversation_id,
                cutoff_session_id=sessions[-1].session_id,
                session_by_source=session_by_source,
                round_to_memory_ids=round_to_memory_ids,
            )
            for qa_index, qa in enumerate(raw.human_annotated_qas)
        ]
        return CanonicalConversation(
            dataset=self.dataset,
            conversation_id=conversation_id,
            sessions=sessions,
            memory_units=memory_units,
            queries=queries,
        )

    def _adapt_query(
        self,
        *,
        scenario_index: int,
        scenario_id: str,
        source_file: str,
        qa_index: int,
        qa: _RawQA,
        conversation_id: str,
        cutoff_session_id: str,
        session_by_source: dict[str, str],
        round_to_memory_ids: dict[str, list[str]],
    ) -> AdaptedQuery:
        unknown_sessions = set(qa.session_id) - set(session_by_source)
        if unknown_sessions:
            raise BenchmarkDataError(
                f"MemGallery QA sessions do not resolve for {scenario_id}/{qa_index}: "
                f"{sorted(unknown_sessions)}"
            )
        unknown_clues = set(qa.clue) - set(round_to_memory_ids)
        if unknown_clues:
            raise BenchmarkDataError(
                f"MemGallery clues do not resolve for {scenario_id}/{qa_index}: "
                f"{sorted(unknown_clues)}"
            )

        query_id = canonical_id(self.dataset, "query", scenario_id, qa_index)
        image_refs: list[ImageRef] = []
        if qa.question_image:
            image_refs.append(
                self._query_image_ref(scenario_id, qa.question_image, qa.image_caption)
            )
        query = QueryInput(
            query_id=query_id,
            dataset=self.dataset,
            conversation_id=conversation_id,
            raw_query=qa.question,
            query_type=map_question_type(self.dataset, qa.point),
            query_subtype=qa.point,
            query_image_refs=image_refs,
            available_after_session=cutoff_session_id,
            metadata={"source_category": qa.point, "scenario_id": scenario_id},
        )
        memory_ids = [memory_id for clue in qa.clue for memory_id in round_to_memory_ids[clue]]
        evidence_ids = [canonical_id(self.dataset, "round", scenario_id, clue) for clue in qa.clue]
        gold = QueryGold(
            answer=qa.answer,
            evidence=GoldEvidence(
                evidence_ids=evidence_ids,
                memory_ids=memory_ids,
                session_ids=[session_by_source[item] for item in qa.session_id],
            ),
            metadata={
                "source_clue_ids": qa.clue,
                "source_session_ids": qa.session_id,
            },
        )
        source = SourceLocator(
            dataset=self.dataset,
            source_file=source_file,
            source_record_index=scenario_index,
            scenario_id=scenario_id,
            qa_index=qa_index,
            source_category=qa.point,
        )
        return AdaptedQuery(
            record=QueryRecord(query=query, gold=gold),
            source=source,
            eligibility=evaluation_eligibility(self.dataset, qa.point),
        )

    def _history_image_ref(
        self,
        scenario_id: str,
        source_image_id: str,
        source_path: str,
        caption: str,
    ) -> ImageRef:
        path = self._resolve_image_path(source_path)
        dataset_relative = path.relative_to(self.root).as_posix()
        clean_caption = caption.strip() or None
        return ImageRef(
            image_id=canonical_id(self.dataset, "image", scenario_id, source_image_id),
            asset_id=path_id(self.dataset, "asset", dataset_relative),
            local_path=project_relative_path(path, self.project_root),
            caption=clean_caption,
            metadata={
                "source_image_id": source_image_id,
                "dataset_relative_path": dataset_relative,
                "caption_source": "benchmark" if clean_caption else None,
            },
        )

    def _query_image_ref(
        self,
        scenario_id: str,
        source_path: str,
        caption: str | None,
    ) -> ImageRef:
        path = self._resolve_image_path(source_path)
        dataset_relative = path.relative_to(self.root).as_posix()
        clean_caption = caption.strip() if caption else None
        return ImageRef(
            image_id=path_id(self.dataset, "query-image", dataset_relative),
            asset_id=path_id(self.dataset, "asset", dataset_relative),
            local_path=project_relative_path(path, self.project_root),
            caption=clean_caption,
            metadata={
                "scenario_id": scenario_id,
                "dataset_relative_path": dataset_relative,
                "caption_source": "benchmark" if clean_caption else None,
            },
        )

    def _resolve_image_path(self, source_path: str) -> Path:
        normalized = source_path.replace("\\", "/")
        for prefix in ("../image/", "../images/", "data/image/", "data/images/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        return resolve_within(self.root / "images", normalized)

    @staticmethod
    def _profile_text(profile: _RawProfile) -> str:
        traits = ", ".join(profile.traits)
        return (
            f"Name: {profile.name}\n"
            f"Persona summary: {profile.persona_summary}\n"
            f"Traits: {traits}\n"
            f"Conversation style: {profile.conversation_style}"
        )

    def expected_counts(self) -> dict[str, int]:
        return {
            "source_records": 20,
            "sessions": 260,
            "memory_units": 7944,
            "queries": 1711,
            "eligible_queries": 1446,
            "excluded_queries": 265,
            "assets": 1490,
            "referenced_unique_images": 1490,
        }
