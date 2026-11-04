"""Deterministic session-anchored Questioner compilation units."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pmmc.compiler.ids import (
    COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1,
    canonical_content_hash,
    compilation_unit_content_id,
    compilation_unit_identity,
    compilation_unit_selection_lineage,
    stable_compiler_id,
)
from pmmc.compiler.opportunities import (
    BuiltQuestionOpportunitySet,
)
from pmmc.compiler.questioner_config import (
    ROUND_OPPORTUNITY_BUILDERS,
    SESSION_OPPORTUNITY_BUILDERS,
    QuestionerConfig,
    questioner_config_hash,
)
from pmmc.compiler.questioner_native import build_questioner_opportunities, questioner_memory_view
from pmmc.compiler.questioner_prompt import (
    questioner_request_contract_hash,
    render_questioner_request,
)
from pmmc.schemas import (
    CompilationUnit,
    CompilationUnitLink,
    CompilationUnitMemory,
    CompilationUnitMemoryRole,
    CompilationUnitRelationType,
    DatasetName,
    GenerationRequest,
    MemoryUnit,
    SessionRecord,
)


class RequestTokenCounter(Protocol):
    @property
    def revision(self) -> str: ...

    def count_request(self, request: GenerationRequest) -> int: ...


class FtsRelationHit(Protocol):
    @property
    def memory_id(self) -> str: ...

    @property
    def score(self) -> float: ...


class FtsRelationProvider(Protocol):
    selector_version: str

    def search(
        self,
        *,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
        anchor_memory_ids: Sequence[str],
        candidate_memory_ids: Sequence[str],
        lexeme_limit: int,
        top_k: int,
    ) -> Sequence[FtsRelationHit]: ...


class QuestionerInputTooLarge(ValueError):
    def __init__(self, message: str, *, error_code: str = "QuestionerInputTooLarge"):
        super().__init__(message)
        self.error_code = error_code


FULL_REQUEST_BUDGET_VERSION = "questioner-full-request-budget-v1"


@dataclass(frozen=True)
class BuiltCompilationUnit:
    unit: CompilationUnit
    links: tuple[CompilationUnitLink, ...]
    memories: tuple[MemoryUnit, ...]
    sessions: tuple[SessionRecord, ...]
    request: GenerationRequest
    input_tokens: int
    opportunities: BuiltQuestionOpportunitySet | None = None


@dataclass
class _SupportBucket:
    relation: CompilationUnitRelationType
    candidates: list[MemoryUnit]
    metadata_by_memory: dict[str, dict[str, object]]


def _memory_sort_key(
    memory: MemoryUnit,
    sessions: dict[str, SessionRecord],
) -> tuple[int, int, str]:
    return (sessions[memory.session_id].session_order, memory.timestamp_order, memory.memory_id)


def _image_ids(memory: MemoryUnit) -> set[str]:
    return set(memory.image_ids) | {item.image_id for item in memory.image_refs}


_RELATION_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{2,}")
_PREFERENCE_RELATION = re.compile(
    r"\b(?:prefer|preference|favorite|favourite|like|love|enjoy|dislike|hate|"
    r"avoid|allerg|diet|usually choose)\w*\b",
    re.IGNORECASE,
)
_RELATION_STOP_LEXEMES = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "because",
        "been",
        "before",
        "being",
        "could",
        "from",
        "have",
        "into",
        "just",
        "more",
        "that",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "very",
        "were",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "your",
    }
)


def _relation_tokens(memory: MemoryUnit) -> set[str]:
    value = f"{memory.text or ''} {memory.caption or ''}".casefold()
    return {
        match.group(0)
        for match in _RELATION_TOKEN.finditer(value)
        if match.group(0) not in _RELATION_STOP_LEXEMES
    }


def _rare_anchor_lexemes(
    anchors: list[MemoryUnit],
    corpus: list[MemoryUnit],
    *,
    limit: int,
) -> tuple[set[str], Counter[str]]:
    """Select bounded anchor lexemes by document rarity, then stable lexical order.

    This is the deterministic dev fallback. Paper-mode v3 preparation uses the canonical
    PostgreSQL ``simple`` tsvector through ``PostgresSimpleFtsRelationProvider``.
    """

    document_frequency: Counter[str] = Counter(
        token for memory in corpus for token in _relation_tokens(memory)
    )
    anchor_tokens = set().union(*(_relation_tokens(item) for item in anchors))
    selected = sorted(
        anchor_tokens,
        key=lambda token: (document_frequency[token], -len(token), token),
    )[:limit]
    return set(selected), document_frequency


def _relation_score(
    anchor_lexemes: set[str],
    memory: MemoryUnit,
    document_frequency: Counter[str],
) -> tuple[int, int]:
    shared = anchor_lexemes & _relation_tokens(memory)
    # Integer reciprocal document frequency keeps ordering deterministic across platforms.
    weighted = sum(1_000_000 // max(1, document_frequency[token]) for token in shared)
    return weighted, len(shared)


def _explicit_preference_relation(
    anchors: Sequence[MemoryUnit],
    support: MemoryUnit,
) -> bool:
    anchor_entities = set().union(*(set(item.entity_mentions) for item in anchors))
    support_text = f"{support.text or ''} {support.caption or ''}"
    return bool(
        anchor_entities
        and anchor_entities & set(support.entity_mentions)
        and _PREFERENCE_RELATION.search(support_text)
    )


class SessionAnchoredUnitBuilder:
    def __init__(
        self,
        *,
        config: QuestionerConfig,
        token_counter: RequestTokenCounter,
        resolved_builder_version: str | None = None,
        fts_relation_provider: FtsRelationProvider | None = None,
    ) -> None:
        self.config = config
        self.token_counter = token_counter
        self.resolved_builder_version = resolved_builder_version or config.unit_builder_version
        self.fts_relation_provider = fts_relation_provider

    def build(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        sessions: list[SessionRecord],
        memories: list[MemoryUnit],
    ) -> list[BuiltCompilationUnit]:
        session_map = {(item.conversation_id, item.session_id): item for item in sessions}
        if len(session_map) != len(sessions):
            raise ValueError("session-anchored units require unique conversation/session pairs")
        if any((item.conversation_id, item.session_id) not in session_map for item in memories):
            raise ValueError("all compilation memories require canonical session records")
        conversations = {item.conversation_id for item in sessions}
        if any(item.conversation_id not in conversations for item in memories):
            raise ValueError("compilation memory conversation is not represented by a session")

        sessions_by_conversation: dict[str, list[SessionRecord]] = defaultdict(list)
        memories_by_session: dict[tuple[str, str], list[MemoryUnit]] = defaultdict(list)
        for session in sessions:
            sessions_by_conversation[session.conversation_id].append(session)
        for memory in memories:
            memories_by_session[(memory.conversation_id, memory.session_id)].append(memory)
        for conversation_sessions in sessions_by_conversation.values():
            conversation_sessions.sort(key=lambda item: (item.session_order, item.session_id))
        for session_memories in memories_by_session.values():
            session_memories.sort(key=lambda item: (item.timestamp_order, item.memory_id))

        result: list[BuiltCompilationUnit] = []
        ordinal = 0
        for conversation_id in sorted(sessions_by_conversation):
            visible: list[MemoryUnit] = []
            conversation_sessions = sessions_by_conversation[conversation_id]
            conversation_session_map = {item.session_id: item for item in conversation_sessions}
            for session in conversation_sessions:
                anchors = memories_by_session.get((conversation_id, session.session_id), [])
                if not anchors:
                    continue
                for shard in self._anchor_shards(
                    snapshot_id=snapshot_id,
                    session=session,
                    anchors=anchors,
                    sessions=conversation_sessions,
                    run_id=run_id,
                    ordinal=ordinal,
                ):
                    built = self._build_one(
                        run_id=run_id,
                        snapshot_id=snapshot_id,
                        conversation_id=conversation_id,
                        session=session,
                        shard=shard,
                        prior_memories=visible,
                        sessions=conversation_sessions,
                        session_map=conversation_session_map,
                        ordinal=ordinal,
                    )
                    result.append(built)
                    ordinal += 1
                visible.extend(anchors)
        return result

    def _anchor_shards(
        self,
        *,
        snapshot_id: str,
        session: SessionRecord,
        anchors: list[MemoryUnit],
        sessions: list[SessionRecord],
        run_id: str = "questioner-budget-probe",
        ordinal: int = 0,
    ) -> list[list[MemoryUnit]]:
        shards: list[list[MemoryUnit]] = []
        current: list[MemoryUnit] = []
        for anchor in anchors:
            proposed = [*current, anchor]
            if self._anchor_fits(
                snapshot_id=snapshot_id,
                session=session,
                memories=proposed,
                sessions=sessions,
                run_id=run_id,
                ordinal=ordinal + len(shards),
            ):
                current = proposed
                continue
            if current:
                shards.append(current)
                current = []
            if not self._anchor_fits(
                snapshot_id=snapshot_id,
                session=session,
                memories=[anchor],
                sessions=sessions,
                run_id=run_id,
                ordinal=ordinal + len(shards),
            ):
                raise QuestionerInputTooLarge(
                    f"single anchor exceeds Questioner input budget: {anchor.memory_id}",
                    error_code="questioner_single_anchor_input_too_large",
                )
            current = [anchor]
        if current:
            shards.append(current)
        return shards

    def _anchor_fits(
        self, *, snapshot_id, session, memories, sessions, run_id, ordinal,
    ) -> bool:
        if not self.config.is_v36:
            return self._fits(
                snapshot_id=snapshot_id, cutoff_session_id=session.session_id,
                memories=memories, sessions=sessions,
            )
        if not self._images_fit(memories):
            return False
        built = self._materialize(
            run_id=run_id, snapshot_id=snapshot_id,
            conversation_id=session.conversation_id, session=session,
            shard=memories, selected=memories, sessions=sessions, ordinal=ordinal,
            relation_for_support={}, relation_metadata_for_support={}, audit=False,
        )
        return built.input_tokens <= self.config.max_input_tokens

    def _images_fit(self, memories: list[MemoryUnit]) -> bool:
        visible = questioner_memory_view(self.config, memories)
        count = len(set().union(*(_image_ids(m) for m in visible)))
        return count <= self.config.max_images_per_unit

    def _build_one(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        conversation_id: str,
        session: SessionRecord,
        shard: list[MemoryUnit],
        prior_memories: list[MemoryUnit],
        sessions: list[SessionRecord],
        session_map: dict[str, SessionRecord],
        ordinal: int,
    ) -> BuiltCompilationUnit:
        support_buckets = self._ordered_support_buckets(
            snapshot_id=snapshot_id,
            conversation_id=conversation_id,
            anchors=shard,
            prior_memories=prior_memories,
            anchor_session=session,
            session_map=session_map,
        )
        selected = list(shard)
        selected_ids = {item.memory_id for item in selected}
        relation_for_support: dict[str, CompilationUnitRelationType] = {}
        relation_metadata_for_support: dict[str, dict[str, object]] = {}
        while True:
            changed = False
            for support_bucket in support_buckets:
                while (
                    support_bucket.candidates
                    and support_bucket.candidates[0].memory_id in selected_ids
                ):
                    support_bucket.candidates.pop(0)
                if not support_bucket.candidates:
                    continue
                candidate = support_bucket.candidates.pop(0)
                candidate_group = [
                    item
                    for item in self._support_group(candidate, prior_memories)
                    if item.memory_id not in selected_ids
                ]
                proposed = sorted(
                    [*selected, *candidate_group],
                    key=lambda item: _memory_sort_key(item, session_map),
                )
                proposed_relations = {
                    **relation_for_support,
                    **{item.memory_id: support_bucket.relation for item in candidate_group},
                }
                proposed_metadata = {
                    **relation_metadata_for_support,
                    **{item.memory_id: {
                        **support_bucket.metadata_by_memory.get(candidate.memory_id, {}),
                        "group_member_count": len(candidate_group),
                    } for item in candidate_group},
                }
                if self.config.is_v36:
                    fits = self._images_fit(proposed) and self._materialize(
                        run_id=run_id, snapshot_id=snapshot_id, conversation_id=conversation_id,
                        session=session, shard=shard, selected=proposed, sessions=sessions,
                        ordinal=ordinal, relation_for_support=proposed_relations,
                        relation_metadata_for_support=proposed_metadata, audit=False,
                    ).input_tokens <= self.config.max_input_tokens
                else:
                    fits = self._fits(
                        snapshot_id=snapshot_id, cutoff_session_id=session.session_id,
                        memories=proposed, sessions=sessions,
                    )
                if fits:
                    selected = proposed
                    for item in candidate_group:
                        selected_ids.add(item.memory_id)
                    relation_for_support = proposed_relations
                    relation_metadata_for_support = proposed_metadata
                    changed = True
            if not changed:
                break

        return self._materialize(
            run_id=run_id, snapshot_id=snapshot_id, conversation_id=conversation_id,
            session=session, shard=shard, selected=selected, sessions=sessions,
            ordinal=ordinal, relation_for_support=relation_for_support,
            relation_metadata_for_support=relation_metadata_for_support, audit=True,
        )

    def _materialize(
        self, *, run_id, snapshot_id, conversation_id, session, shard, selected,
        sessions, ordinal, relation_for_support, relation_metadata_for_support, audit,
    ) -> BuiltCompilationUnit:
        # Budget probes and final requests must share IDs, roles, links and the
        # complete opportunity catalog. A fixed catalog allowance is not a bound.
        members = [
            CompilationUnitMemory(
                snapshot_id=snapshot_id,
                conversation_id=conversation_id,
                memory_id=memory.memory_id,
                ordinal=index,
                role=(
                    CompilationUnitMemoryRole.ANCHOR
                    if memory.memory_id in {item.memory_id for item in shard}
                    else CompilationUnitMemoryRole.SUPPORT
                ),
                metadata=(
                    {
                        "relation_type": relation_for_support[memory.memory_id].value,
                        **relation_metadata_for_support.get(memory.memory_id, {}),
                    }
                    if memory.memory_id in relation_for_support
                    else {}
                ),
            )
            for index, memory in enumerate(selected)
        ]
        builder_identity_payload = {
            "unit_builder_version": self.resolved_builder_version,
            "tokenizer_processor_revision": self.token_counter.revision,
            "max_input_tokens": self.config.max_input_tokens,
            "max_images_per_unit": self.config.max_images_per_unit,
            "questioner_config_hash": questioner_config_hash(self.config),
            **({"budget_policy_version": FULL_REQUEST_BUDGET_VERSION}
               if self.config.is_v36 else {}),
            "request_contract_hash": questioner_request_contract_hash(
                prompt_version=self.config.prompt_version,
                renderer_version=self.config.renderer_version,
                opportunity_builder_version=(
                    self.config.opportunity_builder_version if self.config.is_v3 else None
                ),
            ),
            **(
                {"relation_selector_version": self.config.relation_selector_version}
                if self.config.is_v3
                else {}
            ),
        }
        # Real processor identities contain two pinned 40-character commits. Keep
        # the queryable version column bounded while retaining the full identity
        # payload in metadata and binding its hash into the unit ID.
        builder_identity = stable_compiler_id("unit-builder", builder_identity_payload, length=64)
        selection_lineage = compilation_unit_selection_lineage(members)
        unit_id, input_hash = compilation_unit_identity(
            snapshot_id=snapshot_id,
            conversation_id=conversation_id,
            cutoff_session_id=session.session_id,
            builder_version=builder_identity,
            memories=members,
            selection_lineage=selection_lineage if self.config.is_v3 else None,
            run_id=run_id if self.config.is_v34 else None,
        )
        unit = CompilationUnit(
            compiler_unit_id=unit_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
            conversation_id=conversation_id,
            available_after_session=session.session_id,
            ordinal=ordinal,
            builder_version=builder_identity,
            input_hash=input_hash,
            memories=members,
            metadata={
                "anchor_memory_ids": [item.memory_id for item in shard],
                "builder_identity": builder_identity_payload,
                "input_image_count": len(
                    set().union(
                        *(
                            _image_ids(item)
                            for item in questioner_memory_view(self.config, selected)
                        )
                    )
                ),
                **(
                    {
                        "compiler_unit_content_id": compilation_unit_content_id(input_hash),
                        "compiler_unit_identity_protocol": (COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1),
                    }
                    if self.config.is_v34
                    else {}
                ),
                **({"relation_selection": selection_lineage} if self.config.is_v3 else {}),
                **(
                    {
                        "relation_selector_version": (
                            self.config.relation_selector_version
                            if self.fts_relation_provider is not None
                            else "deterministic-dev-fallback-v1"
                        ),
                        "fts_recall_selector_version": (
                            self.fts_relation_provider.selector_version
                            if self.fts_relation_provider is not None
                            else "deterministic-dev-fallback-v1"
                        ),
                    }
                    if self.config.is_v3
                    else {}
                ),
            },
        )
        links = self._links(
            unit=unit,
            anchors=shard,
            selected=selected,
            relations=relation_for_support,
            relation_metadata=relation_metadata_for_support,
        )
        opportunities = (
            build_questioner_opportunities(
                config=self.config,
                unit=unit,
                memories=selected,
                sessions=sessions,
                links=links,
            )
            if self.config.is_v3
            else None
        )
        request = render_questioner_request(
            request_id=f"questioner-{unit_id}",
            model_alias=self.config.compiler_model_alias,
            unit_id=unit_id,
            cutoff_session_id=session.session_id,
            memories=selected,
            sessions=sessions,
            max_output_tokens=self.config.reserved_output_tokens,
            prompt_version=self.config.prompt_version,
            renderer_version=self.config.renderer_version,
            opportunity_set=opportunities,
            memory_raw_image_policy=self.config.memory_raw_image_policy,
        )
        input_tokens = self.token_counter.count_request(request)
        if audit and input_tokens > self.config.max_input_tokens:
            raise QuestionerInputTooLarge(
                "Questioner v3 opportunity catalog exceeds the frozen input budget",
                error_code="questioner_full_request_input_too_large",
            )
        auditor = getattr(self.token_counter, "audit_request", None)
        if audit and callable(auditor):
            auditor(request, input_tokens)
        return BuiltCompilationUnit(
            unit=unit,
            links=tuple(links),
            memories=tuple(selected),
            sessions=tuple(sessions),
            request=request,
            input_tokens=input_tokens,
            opportunities=opportunities,
        )

    def _ordered_support_buckets(
        self,
        *,
        snapshot_id: str,
        conversation_id: str,
        anchors: list[MemoryUnit],
        prior_memories: list[MemoryUnit],
        anchor_session: SessionRecord,
        session_map: dict[str, SessionRecord],
    ) -> list[_SupportBucket]:
        del snapshot_id, conversation_id
        buckets = self._relation_buckets(
            anchors=anchors,
            prior_memories=prior_memories,
            anchor_session=anchor_session,
            session_map=session_map,
        )
        return [
            _SupportBucket(
                relation=relation,
                candidates=buckets[relation],
                metadata_by_memory={
                    item.memory_id: {"relation_source": "canonical_structure"}
                    for item in buckets[relation]
                },
            )
            for relation in self.config.relation_order
        ]

    def _relation_buckets(
        self,
        *,
        anchors: list[MemoryUnit],
        prior_memories: list[MemoryUnit],
        anchor_session: SessionRecord,
        session_map: dict[str, SessionRecord],
    ) -> dict[CompilationUnitRelationType, list[MemoryUnit]]:
        anchor_entities = set().union(*(set(item.entity_mentions) for item in anchors))
        anchor_images = set().union(*(_image_ids(item) for item in anchors))
        buckets: dict[CompilationUnitRelationType, list[MemoryUnit]] = {
            relation: [] for relation in self.config.relation_order
        }
        for memory in prior_memories:
            prior_session = session_map[memory.session_id]
            if prior_session.session_order == anchor_session.session_order - 1:
                buckets[CompilationUnitRelationType.ADJACENT_SESSION].append(memory)
            if anchor_entities & set(memory.entity_mentions):
                buckets[CompilationUnitRelationType.SHARED_ENTITY].append(memory)
            if anchor_images & _image_ids(memory):
                buckets[CompilationUnitRelationType.SHARED_IMAGE].append(memory)
            if prior_session.is_synthetic:
                buckets[CompilationUnitRelationType.PROFILE_PREFERENCE].append(memory)
            buckets[CompilationUnitRelationType.TEMPORAL_NEIGHBOR].append(memory)
        for relation, values in buckets.items():
            reverse = relation in {
                CompilationUnitRelationType.ADJACENT_SESSION,
                CompilationUnitRelationType.TEMPORAL_NEIGHBOR,
            }
            values.sort(
                key=lambda item: _memory_sort_key(item, session_map),
                reverse=reverse,
            )
        return buckets

    def _support_group(
        self,
        candidate: MemoryUnit,
        prior_memories: list[MemoryUnit],
    ) -> list[MemoryUnit]:
        del prior_memories
        return [candidate]

    def _links(
        self,
        *,
        unit: CompilationUnit,
        anchors: list[MemoryUnit],
        selected: list[MemoryUnit],
        relations: dict[str, CompilationUnitRelationType],
        relation_metadata: dict[str, dict[str, object]],
    ) -> list[CompilationUnitLink]:
        links: list[CompilationUnitLink] = []
        support = [item for item in selected if item.memory_id in relations]
        for ordinal, memory in enumerate(support):
            relation = relations[memory.memory_id]
            anchor = self._best_anchor(anchors, memory, relation)
            payload = [unit.compiler_unit_id, anchor.memory_id, memory.memory_id, relation.value]
            links.append(
                CompilationUnitLink(
                    link_id=stable_compiler_id("compiler-link", payload),
                    run_id=unit.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    snapshot_id=unit.snapshot_id,
                    conversation_id=unit.conversation_id,
                    anchor_memory_id=anchor.memory_id,
                    support_memory_id=memory.memory_id,
                    relation_type=relation,
                    ordinal=ordinal,
                    relation_key_hash=canonical_content_hash(payload),
                    metadata=relation_metadata.get(memory.memory_id, {}),
                )
            )
        return links

    @staticmethod
    def _best_anchor(
        anchors: list[MemoryUnit],
        support: MemoryUnit,
        relation: CompilationUnitRelationType,
    ) -> MemoryUnit:
        for anchor in anchors:
            if relation is CompilationUnitRelationType.SHARED_ENTITY and (
                set(anchor.entity_mentions) & set(support.entity_mentions)
            ):
                return anchor
            if relation is CompilationUnitRelationType.SHARED_IMAGE and (
                _image_ids(anchor) & _image_ids(support)
            ):
                return anchor
        return anchors[0]

    def _fits(
        self,
        *,
        snapshot_id: str,
        cutoff_session_id: str,
        memories: list[MemoryUnit],
        sessions: list[SessionRecord],
    ) -> bool:
        del snapshot_id
        memories = questioner_memory_view(self.config, memories)
        image_count = len(set().union(*(_image_ids(item) for item in memories)))
        if image_count > self.config.max_images_per_unit:
            return False
        # v3 reserves space for the bounded opportunity catalog. The final request
        # is still counted exactly after the unit and its links are materialized.
        if self.config.is_v3:
            from pmmc.compiler.questioner_prompt import (
                QUESTIONER_V2_PROMPT_VERSION,
                QUESTIONER_V2_RENDERER_VERSION,
            )

            prompt_version = QUESTIONER_V2_PROMPT_VERSION
            renderer_version = QUESTIONER_V2_RENDERER_VERSION
            limit = self.config.max_input_tokens - self.config.reserved_output_tokens
        else:
            prompt_version = self.config.prompt_version
            renderer_version = self.config.renderer_version
            limit = self.config.max_input_tokens
        request = render_questioner_request(
            request_id="questioner-budget-probe",
            model_alias=self.config.compiler_model_alias,
            unit_id="compiler-unit-budget-probe",
            cutoff_session_id=cutoff_session_id,
            memories=memories,
            sessions=sessions,
            max_output_tokens=self.config.reserved_output_tokens,
            prompt_version=prompt_version,
            renderer_version=renderer_version,
        )
        return self.token_counter.count_request(request) <= limit


class RoundBlockUnitBuilder(SessionAnchoredUnitBuilder):
    """Build MemGallery units from complete, non-overlapping source-round blocks."""

    def __init__(
        self,
        *,
        config: QuestionerConfig,
        token_counter: RequestTokenCounter,
        fts_relation_provider: FtsRelationProvider | None = None,
    ) -> None:
        super().__init__(
            config=config,
            token_counter=token_counter,
            resolved_builder_version=config.unit_builder_for(DatasetName.MEMGALLERY),
            fts_relation_provider=fts_relation_provider,
        )

    def build(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        sessions: list[SessionRecord],
        memories: list[MemoryUnit],
    ) -> list[BuiltCompilationUnit]:
        if any(item.dataset is not DatasetName.MEMGALLERY for item in memories):
            raise ValueError("round-block units only accept MemGallery memories")
        session_map = {(item.conversation_id, item.session_id): item for item in sessions}
        if len(session_map) != len(sessions):
            raise ValueError("round-block units require unique conversation/session pairs")
        if any((item.conversation_id, item.session_id) not in session_map for item in memories):
            raise ValueError("all round-block memories require canonical sessions")

        sessions_by_conversation: dict[str, list[SessionRecord]] = defaultdict(list)
        memories_by_session: dict[tuple[str, str], list[MemoryUnit]] = defaultdict(list)
        for session in sessions:
            sessions_by_conversation[session.conversation_id].append(session)
        for memory in memories:
            memories_by_session[(memory.conversation_id, memory.session_id)].append(memory)
        for conversation_sessions in sessions_by_conversation.values():
            conversation_sessions.sort(key=lambda item: (item.session_order, item.session_id))
        for session_memories in memories_by_session.values():
            session_memories.sort(key=lambda item: (item.timestamp_order, item.memory_id))

        result: list[BuiltCompilationUnit] = []
        ordinal = 0
        for conversation_id in sorted(sessions_by_conversation):
            conversation_sessions = sessions_by_conversation[conversation_id]
            conversation_session_map = {item.session_id: item for item in conversation_sessions}
            visible: list[MemoryUnit] = []
            for session in conversation_sessions:
                session_memories = memories_by_session.get(
                    (conversation_id, session.session_id), []
                )
                if not session_memories:
                    continue
                blocks = self._session_blocks(
                    snapshot_id=snapshot_id,
                    session=session,
                    memories=session_memories,
                    sessions=conversation_sessions,
                    run_id=run_id,
                    ordinal=ordinal,
                )
                for block_ordinal, block in enumerate(blocks):
                    built = self._build_one(
                        run_id=run_id,
                        snapshot_id=snapshot_id,
                        conversation_id=conversation_id,
                        session=session,
                        shard=block,
                        prior_memories=visible,
                        sessions=conversation_sessions,
                        session_map=conversation_session_map,
                        ordinal=ordinal,
                    )
                    source_round_ids = self._source_round_ids(block)
                    memory_by_id = {item.memory_id: item for item in built.memories}
                    unit_metadata = {
                        **built.unit.metadata,
                        "unit_builder_strategy": self.resolved_builder_version,
                        "anchor_block_ordinal": block_ordinal,
                        "anchor_source_round_ids": source_round_ids,
                    }
                    links = tuple(
                        link.model_copy(
                            update={
                                "metadata": {
                                    **link.metadata,
                                    "anchor_block_ordinal": block_ordinal,
                                    "anchor_source_round_ids": source_round_ids,
                                    "support_source_round_id": self._source_round_id(
                                        memory_by_id[link.support_memory_id],
                                        required=False,
                                    ),
                                    "support_subtype": self._support_subtype(
                                        relation=link.relation_type,
                                        support=memory_by_id[link.support_memory_id],
                                        anchor_session=session,
                                        session_map=conversation_session_map,
                                    ),
                                }
                            }
                        )
                        for link in built.links
                    )
                    result.append(
                        BuiltCompilationUnit(
                            unit=built.unit.model_copy(update={"metadata": unit_metadata}),
                            links=links,
                            memories=built.memories,
                            sessions=built.sessions,
                            request=built.request,
                            input_tokens=built.input_tokens,
                            opportunities=built.opportunities,
                        )
                    )
                    ordinal += 1
                    visible.extend(block)
        return result

    def _ordered_support_buckets(
        self,
        *,
        snapshot_id: str,
        conversation_id: str,
        anchors: list[MemoryUnit],
        prior_memories: list[MemoryUnit],
        anchor_session: SessionRecord,
        session_map: dict[str, SessionRecord],
    ) -> list[_SupportBucket]:
        """Return the frozen MemGallery support rotation without new enum values."""

        del snapshot_id, conversation_id

        anchor_entities = set().union(*(set(item.entity_mentions) for item in anchors))
        anchor_images = set().union(*(_image_ids(item) for item in anchors))
        ordered_prior = sorted(
            prior_memories,
            key=lambda item: _memory_sort_key(item, session_map),
        )
        profile = [item for item in ordered_prior if session_map[item.session_id].is_synthetic]
        same_session = [
            item for item in ordered_prior if item.session_id == anchor_session.session_id
        ]
        prior_session_orders = [
            session.session_order
            for session in session_map.values()
            if not session.is_synthetic and session.session_order < anchor_session.session_order
        ]
        previous_session: list[MemoryUnit] = []
        if prior_session_orders:
            previous_order = max(prior_session_orders)
            previous_session = [
                item
                for item in ordered_prior
                if session_map[item.session_id].session_order == previous_order
            ]
        shared_images = [item for item in ordered_prior if anchor_images & _image_ids(item)]
        shared_entities = [
            item for item in ordered_prior if anchor_entities & set(item.entity_mentions)
        ]
        values = [
            (CompilationUnitRelationType.PROFILE_PREFERENCE, profile),
            (
                CompilationUnitRelationType.TEMPORAL_NEIGHBOR,
                self._tail_round_block(same_session),
            ),
            (
                CompilationUnitRelationType.ADJACENT_SESSION,
                self._tail_round_block(previous_session),
            ),
            (CompilationUnitRelationType.SHARED_IMAGE, shared_images),
            (CompilationUnitRelationType.SHARED_ENTITY, shared_entities),
            (
                CompilationUnitRelationType.TEMPORAL_NEIGHBOR,
                list(reversed(ordered_prior)),
            ),
        ]
        return [
            _SupportBucket(
                relation=relation,
                candidates=candidates,
                metadata_by_memory={
                    item.memory_id: {"relation_source": "canonical_structure"}
                    for item in candidates
                },
            )
            for relation, candidates in values
        ]

    def _tail_round_block(self, memories: list[MemoryUnit]) -> list[MemoryUnit]:
        by_round: dict[str, list[MemoryUnit]] = defaultdict(list)
        for memory in memories:
            source_round = self._source_round_id(memory, required=False)
            if source_round:
                by_round[source_round].append(memory)
        rounds = sorted(
            by_round.values(),
            key=lambda values: (
                min(item.timestamp_order for item in values),
                self._source_round_id(values[0], required=True),
            ),
        )
        if any(len(values) != 2 for values in rounds):
            raise ValueError("MemGallery support source rounds must remain complete")
        return [item for values in rounds[-self.config.round_block_size :] for item in values]

    @staticmethod
    def _support_subtype(
        *,
        relation: CompilationUnitRelationType,
        support: MemoryUnit,
        anchor_session: SessionRecord,
        session_map: dict[str, SessionRecord],
    ) -> str:
        support_session = session_map[support.session_id]
        if support_session.is_synthetic:
            return "profile"
        if relation is CompilationUnitRelationType.SHARED_IMAGE:
            return "shared_image"
        if relation is CompilationUnitRelationType.SHARED_ENTITY:
            return "shared_entity"
        if support.session_id == anchor_session.session_id:
            return "previous_block"
        if support_session.session_order == anchor_session.session_order - 1:
            return "previous_session_tail"
        return "temporal_neighbor"

    def _session_blocks(
        self,
        *,
        snapshot_id: str,
        session: SessionRecord,
        memories: list[MemoryUnit],
        sessions: list[SessionRecord],
        run_id: str = "questioner-budget-probe",
        ordinal: int = 0,
    ) -> list[list[MemoryUnit]]:
        if session.is_synthetic:
            return self._round_boundary_shards(
                snapshot_id=snapshot_id,
                session=session,
                rounds=[memories],
                sessions=sessions,
                run_id=run_id,
                ordinal=ordinal,
            )
        by_round: dict[str, list[MemoryUnit]] = defaultdict(list)
        for memory in memories:
            by_round[self._source_round_id(memory, required=True)].append(memory)
        rounds = sorted(
            by_round.values(),
            key=lambda values: (
                min(item.timestamp_order for item in values),
                self._source_round_id(values[0], required=True),
            ),
        )
        if any(len(values) != 2 for values in rounds):
            raise ValueError("each MemGallery source round must contain two MemoryUnits")
        blocks: list[list[MemoryUnit]] = []
        size = self.config.round_block_size
        stride = self.config.round_block_stride
        for offset in range(0, len(rounds), stride):
            grouped = rounds[offset : offset + size]
            if not grouped:
                continue
            blocks.extend(
                self._round_boundary_shards(
                    snapshot_id=snapshot_id,
                    session=session,
                    rounds=grouped,
                    sessions=sessions,
                    run_id=run_id,
                    ordinal=ordinal + len(blocks),
                )
            )
        return blocks

    def _support_group(
        self,
        candidate: MemoryUnit,
        prior_memories: list[MemoryUnit],
    ) -> list[MemoryUnit]:
        source_round_id = self._source_round_id(candidate, required=False)
        if source_round_id:
            values = [
                item
                for item in prior_memories
                if item.session_id == candidate.session_id
                and self._source_round_id(item, required=False) == source_round_id
            ]
            if len(values) != 2:
                raise ValueError("MemGallery support source rounds must remain complete")
            return values
        return [item for item in prior_memories if item.session_id == candidate.session_id]

    def _round_boundary_shards(
        self,
        *,
        snapshot_id: str,
        session: SessionRecord,
        rounds: list[list[MemoryUnit]],
        sessions: list[SessionRecord],
        run_id: str = "questioner-budget-probe",
        ordinal: int = 0,
    ) -> list[list[MemoryUnit]]:
        shards: list[list[MemoryUnit]] = []
        current: list[MemoryUnit] = []
        for round_memories in rounds:
            proposed = [*current, *round_memories]
            if self._anchor_fits(
                snapshot_id=snapshot_id,
                session=session,
                memories=proposed,
                sessions=sessions,
                run_id=run_id,
                ordinal=ordinal + len(shards),
            ):
                current = proposed
                continue
            if current:
                shards.append(current)
            if not self._anchor_fits(
                snapshot_id=snapshot_id,
                session=session,
                memories=round_memories,
                sessions=sessions,
                run_id=run_id,
                ordinal=ordinal + len(shards),
            ):
                source_round = self._source_round_id(
                    round_memories[0], required=not session.is_synthetic
                )
                raise QuestionerInputTooLarge(
                    "single MemGallery source round exceeds Questioner input budget: "
                    f"{source_round or round_memories[0].memory_id}",
                    error_code="questioner_single_round_input_too_large",
                )
            current = list(round_memories)
        if current:
            shards.append(current)
        return shards

    @staticmethod
    def _source_round_id(memory: MemoryUnit, *, required: bool) -> str:
        value = memory.metadata.get("source_round_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if required:
            raise ValueError(
                f"MemGallery dialogue memory has no source_round_id: {memory.memory_id}"
            )
        return ""

    @classmethod
    def _source_round_ids(cls, memories: list[MemoryUnit]) -> list[str]:
        return list(
            dict.fromkeys(
                value
                for value in (cls._source_round_id(item, required=False) for item in memories)
                if value
            )
        )


class SessionOpportunityUnitBuilder(SessionAnchoredUnitBuilder):
    """Session builder with bounded lexical and structural support selection."""

    def _ordered_support_buckets(
        self,
        *,
        snapshot_id: str,
        conversation_id: str,
        anchors: list[MemoryUnit],
        prior_memories: list[MemoryUnit],
        anchor_session: SessionRecord,
        session_map: dict[str, SessionRecord],
    ) -> list[_SupportBucket]:
        anchor_tokens, document_frequency = _rare_anchor_lexemes(
            anchors,
            [*prior_memories, *anchors],
            limit=self.config.fts_lexeme_limit,
        )
        anchor_images = set().union(*(_image_ids(item) for item in anchors))

        def score(item: MemoryUnit) -> tuple[int, int, int, str]:
            weighted, overlap = _relation_score(anchor_tokens, item, document_frequency)
            return (
                -weighted,
                -overlap,
                -session_map[item.session_id].session_order,
                item.memory_id,
            )

        if self.fts_relation_provider is None:
            lexical = [
                item
                for item in prior_memories
                if _relation_score(anchor_tokens, item, document_frequency)[0] > 0
            ]
            lexical.sort(key=score)
            lexical = lexical[: self.config.fts_relation_top_k]
            lexical_metadata = {
                item.memory_id: {
                    "relation_source": "deterministic_dev_fallback",
                    "selector_version": "deterministic-dev-fallback-v1",
                    "score_micros": _relation_score(anchor_tokens, item, document_frequency)[0],
                    "selector_rank": rank,
                }
                for rank, item in enumerate(lexical, start=1)
            }
        else:
            by_id = {item.memory_id: item for item in prior_memories}
            hits = self.fts_relation_provider.search(
                snapshot_id=snapshot_id,
                conversation_id=conversation_id,
                cutoff_session_id=anchor_session.session_id,
                anchor_memory_ids=[item.memory_id for item in anchors],
                candidate_memory_ids=list(by_id),
                lexeme_limit=self.config.fts_lexeme_limit,
                top_k=self.config.fts_relation_top_k,
            )
            lexical = [by_id[item.memory_id] for item in hits if item.memory_id in by_id]
            lexical_metadata = {
                item.memory_id: {
                    "relation_source": "postgres_simple_fts",
                    "selector_version": self.fts_relation_provider.selector_version,
                    "score_micros": round(item.score * 1_000_000),
                    "selector_rank": rank,
                }
                for rank, item in enumerate(hits, start=1)
                if item.memory_id in by_id
            }
        adjacent = [
            item
            for item in prior_memories
            if session_map[item.session_id].session_order == anchor_session.session_order - 1
        ][-self.config.fts_relation_top_k :]
        shared_images = [item for item in prior_memories if anchor_images & _image_ids(item)]
        profile = [
            item
            for item in prior_memories
            if session_map[item.session_id].is_synthetic
            and (item in lexical or _explicit_preference_relation(anchors, item))
        ]
        profile_metadata: dict[str, dict[str, object]] = {
            item.memory_id: (
                lexical_metadata[item.memory_id]
                if item.memory_id in lexical_metadata
                else {"relation_source": "explicit_preference_relation"}
            )
            for item in profile
        }
        structural: dict[str, dict[str, object]] = {
            item.memory_id: {"relation_source": "canonical_structure"}
            for item in [*adjacent, *shared_images]
        }
        return [
            _SupportBucket(
                CompilationUnitRelationType.ADJACENT_SESSION,
                adjacent,
                structural,
            ),
            _SupportBucket(
                CompilationUnitRelationType.SHARED_ENTITY,
                lexical,
                lexical_metadata,
            ),
            _SupportBucket(
                CompilationUnitRelationType.SHARED_IMAGE,
                shared_images,
                structural,
            ),
            _SupportBucket(
                CompilationUnitRelationType.PROFILE_PREFERENCE,
                profile,
                profile_metadata,
            ),
            _SupportBucket(
                CompilationUnitRelationType.TEMPORAL_NEIGHBOR,
                lexical,
                lexical_metadata,
            ),
        ]


class RoundOpportunityUnitBuilder(RoundBlockUnitBuilder):
    """MemGallery round builder with complete, related support rounds."""

    def _ordered_support_buckets(
        self,
        *,
        snapshot_id: str,
        conversation_id: str,
        anchors: list[MemoryUnit],
        prior_memories: list[MemoryUnit],
        anchor_session: SessionRecord,
        session_map: dict[str, SessionRecord],
    ) -> list[_SupportBucket]:
        anchor_tokens, document_frequency = _rare_anchor_lexemes(
            anchors,
            [*prior_memories, *anchors],
            limit=self.config.fts_lexeme_limit,
        )
        anchor_images = set().union(*(_image_ids(item) for item in anchors))
        ordered = sorted(
            prior_memories,
            key=lambda item: _memory_sort_key(item, session_map),
        )
        if self.fts_relation_provider is None:
            lexical = [
                item
                for item in ordered
                if _relation_score(anchor_tokens, item, document_frequency)[0] > 0
            ]
            lexical.sort(
                key=lambda item: (
                    -_relation_score(anchor_tokens, item, document_frequency)[0],
                    -_relation_score(anchor_tokens, item, document_frequency)[1],
                    item.memory_id,
                )
            )
            lexical = lexical[: self.config.fts_relation_top_k]
            lexical_metadata = {
                item.memory_id: {
                    "relation_source": "deterministic_dev_fallback",
                    "selector_version": "deterministic-dev-fallback-v1",
                    "score_micros": _relation_score(anchor_tokens, item, document_frequency)[0],
                    "selector_rank": rank,
                }
                for rank, item in enumerate(lexical, start=1)
            }
        else:
            by_id = {item.memory_id: item for item in ordered}
            hits = self.fts_relation_provider.search(
                snapshot_id=snapshot_id,
                conversation_id=conversation_id,
                cutoff_session_id=anchor_session.session_id,
                anchor_memory_ids=[item.memory_id for item in anchors],
                candidate_memory_ids=list(by_id),
                lexeme_limit=self.config.fts_lexeme_limit,
                top_k=self.config.fts_relation_top_k,
            )
            lexical = [by_id[item.memory_id] for item in hits if item.memory_id in by_id]
            lexical_metadata = {
                item.memory_id: {
                    "relation_source": "postgres_simple_fts",
                    "selector_version": self.fts_relation_provider.selector_version,
                    "score_micros": round(item.score * 1_000_000),
                    "selector_rank": rank,
                }
                for rank, item in enumerate(hits, start=1)
                if item.memory_id in by_id
            }
        same_session = [item for item in ordered if item.session_id == anchor_session.session_id]
        boundary = self._tail_round_block(same_session)
        shared_images = [item for item in ordered if anchor_images & _image_ids(item)]
        profile = [
            item
            for item in ordered
            if session_map[item.session_id].is_synthetic
            and (item in lexical or _explicit_preference_relation(anchors, item))
        ]
        profile_metadata: dict[str, dict[str, object]] = {
            item.memory_id: (
                lexical_metadata[item.memory_id]
                if item.memory_id in lexical_metadata
                else {"relation_source": "explicit_preference_relation"}
            )
            for item in profile
        }
        structural: dict[str, dict[str, object]] = {
            item.memory_id: {"relation_source": "canonical_structure"}
            for item in [*boundary, *shared_images]
        }
        return [
            _SupportBucket(
                CompilationUnitRelationType.TEMPORAL_NEIGHBOR,
                boundary,
                structural,
            ),
            _SupportBucket(
                CompilationUnitRelationType.SHARED_ENTITY,
                lexical,
                lexical_metadata,
            ),
            _SupportBucket(
                CompilationUnitRelationType.SHARED_IMAGE,
                shared_images,
                structural,
            ),
            _SupportBucket(
                CompilationUnitRelationType.PROFILE_PREFERENCE,
                profile,
                profile_metadata,
            ),
        ]


def create_compilation_unit_builder(
    *,
    config: QuestionerConfig,
    token_counter: RequestTokenCounter,
    dataset: DatasetName,
    fts_relation_provider: FtsRelationProvider | None = None,
) -> SessionAnchoredUnitBuilder:
    builder = config.unit_builder_for(dataset)
    if builder == "session-anchor-candidate-set-v1":
        return SessionAnchoredUnitBuilder(
            config=config,
            token_counter=token_counter,
            resolved_builder_version=builder,
            fts_relation_provider=fts_relation_provider,
        )
    if builder == "round-block-candidate-set-v1":
        if dataset is not DatasetName.MEMGALLERY:
            raise ValueError("round-block builder is only supported for MemGallery")
        return RoundBlockUnitBuilder(
            config=config,
            token_counter=token_counter,
            fts_relation_provider=fts_relation_provider,
        )
    if builder in SESSION_OPPORTUNITY_BUILDERS:
        if dataset is not DatasetName.MEMLENS:
            raise ValueError("session opportunity builder is only supported for MEMLENS")
        return SessionOpportunityUnitBuilder(
            config=config,
            token_counter=token_counter,
            resolved_builder_version=builder,
            fts_relation_provider=fts_relation_provider,
        )
    if builder in ROUND_OPPORTUNITY_BUILDERS:
        if dataset is not DatasetName.MEMGALLERY:
            raise ValueError("round opportunity builder is only supported for MemGallery")
        return RoundOpportunityUnitBuilder(
            config=config,
            token_counter=token_counter,
            fts_relation_provider=fts_relation_provider,
        )
    raise ValueError(f"unsupported compilation unit builder: {builder}")
