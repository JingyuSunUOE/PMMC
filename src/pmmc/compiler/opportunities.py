"""Deterministic Questioner v3 opportunities and local evidence aliases."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.questioner_config import EVIDENCE_FIRST_VERSION, QuestionerConfig
from pmmc.schemas import (
    CompilationUnit,
    CompilationUnitLink,
    CompilationUnitMemoryRole,
    DatasetName,
    EvidenceType,
    MemoryUnit,
    Modality,
    OperatorEvidence,
    QuestionerV2Modality,
    QuestionerV2QuestionType,
    QuestionOpportunity,
    QuestionOpportunityEvidenceRef,
    QuestionOpportunitySet,
    QuestionOpportunityType,
    QuestionReasoningType,
    SessionRecord,
)

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{2,}")
_NUMBER_OR_DATE = re.compile(r"\b(?:\d{1,4}(?:[-/:.]\d{1,2}){0,2}|\d+(?:\.\d+)?)\b")
_TEMPORAL_CUE = re.compile(
    r"\b(?:before|after|earlier|later|first|last|next|previous|today|yesterday|"
    r"tomorrow|week|month|year|morning|afternoon|evening|recently|since|until)\b",
    re.IGNORECASE,
)
_UPDATE_CUE = re.compile(
    r"\b(?:now|currently|changed|updated|moved|new|no longer|used to|became|"
    r"started|stopped|switched|instead|latest)\b",
    re.IGNORECASE,
)
_NEGATION_CUE = re.compile(
    r"\b(?:not|never|no longer|didn't|doesn't|isn't|wasn't|cannot|can't|"
    r"but|however|instead|unlike)\b",
    re.IGNORECASE,
)
_PREFERENCE_CUE = re.compile(
    r"\b(?:prefer|preference|favorite|favourite|like|love|enjoy|dislike|hate|"
    r"avoid|allerg|diet|usually choose)\w*\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_STOP_LEXEMES = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "assistant",
        "because",
        "been",
        "before",
        "being",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "from",
        "have",
        "here",
        "into",
        "just",
        "like",
        "memory",
        "more",
        "said",
        "says",
        "session",
        "some",
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
        "will",
        "with",
        "would",
        "user",
        "your",
    }
)

# This implementation identity is intentionally separate from the scientific
# opportunity-builder version. It invalidates prepared request caches when the
# local evidence alias representation changes without altering the frozen
# Questioner protocol.
QUESTIONER_OPPORTUNITY_BINDING_VERSION = "questioner-opportunity-evidence-binding-v2"


@dataclass(frozen=True)
class BuiltQuestionOpportunitySet:
    opportunity_set: QuestionOpportunitySet
    opportunities: tuple[QuestionOpportunity, ...]
    evidence_refs: tuple[QuestionOpportunityEvidenceRef, ...]
    alias_to_evidence: dict[str, OperatorEvidence]
    alias_to_group: dict[str, str]
    memory_aliases: dict[str, str]
    session_aliases: dict[str, str]
    image_aliases: dict[tuple[str, str], str]
    caption_aliases: dict[str, str]
    sentence_span_aliases: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class _OpportunitySpec:
    opportunity_type: QuestionOpportunityType
    question_types: tuple[QuestionerV2QuestionType, ...]
    reasoning_types: tuple[QuestionReasoningType, ...]
    modalities: tuple[QuestionerV2Modality, ...]
    aliases: tuple[str, ...]
    relation_ids: tuple[str, ...] = ()
    minimum_evidence_groups: int = 1
    metadata: tuple[tuple[str, object], ...] = ()
    speaker_scope: tuple[str, ...] = ()

    def identity(self) -> dict[str, object]:
        return {
            "opportunity_type": self.opportunity_type.value,
            "question_types": [item.value for item in self.question_types],
            "reasoning_types": [item.value for item in self.reasoning_types],
            "modalities": [item.value for item in self.modalities],
            "aliases": list(self.aliases),
            "relation_ids": list(self.relation_ids),
            "minimum_evidence_groups": self.minimum_evidence_groups,
            "metadata": dict(self.metadata),
            "speaker_scope": list(self.speaker_scope),
        }


def _text(memory: MemoryUnit) -> str:
    return f"{memory.text or ''} {memory.caption or ''}".strip()


def _sentence_spans(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in _SENTENCE_BOUNDARY.split(value) if item.strip()]


def _tokens(memory: MemoryUnit) -> set[str]:
    return {
        match.group(0)
        for match in _TOKEN.finditer(_text(memory).casefold())
        if match.group(0) not in _STOP_LEXEMES
    }


def _image_ids(memory: MemoryUnit) -> list[str]:
    return list(dict.fromkeys([*memory.image_ids, *(item.image_id for item in memory.image_refs)]))


def _source_round_id(memory: MemoryUnit) -> str | None:
    value = memory.metadata.get("source_round_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int):
        return str(value)
    return None


def _memory_group(memory: MemoryUnit) -> str:
    source_round_id = _source_round_id(memory)
    if memory.dataset is DatasetName.MEMGALLERY and source_round_id is not None:
        return f"round:{source_round_id}"
    if memory.dataset is DatasetName.MEMGALLERY and memory.metadata.get("is_synthetic"):
        return f"profile:{memory.session_id}"
    if memory.dataset is DatasetName.MEMLENS:
        return f"session:{memory.session_id}"
    return f"memory:{memory.memory_id}"


def _lexical_scores(memories: list[MemoryUnit]) -> dict[tuple[str, str], float]:
    tokens = {item.memory_id: _tokens(item) for item in memories}
    frequencies = Counter(token for values in tokens.values() for token in values)
    document_count = max(1, len(memories))
    result: dict[tuple[str, str], float] = {}
    for left in memories:
        for right in memories:
            if left.memory_id >= right.memory_id:
                continue
            shared = tokens[left.memory_id] & tokens[right.memory_id]
            result[(left.memory_id, right.memory_id)] = sum(
                math.log((document_count + 1) / (frequencies[token] + 1)) + 1.0
                for token in shared
            )
    return result


def _pair_key(left: MemoryUnit, right: MemoryUnit) -> tuple[str, str]:
    return tuple(sorted((left.memory_id, right.memory_id)))  # type: ignore[return-value]


def _select_diverse(
    buckets: dict[QuestionOpportunityType, list[_OpportunitySpec]],
    *,
    limit: int,
) -> list[_OpportunitySpec]:
    order = [
        QuestionOpportunityType.LOCAL_FACT,
        QuestionOpportunityType.VISUAL_DETAIL,
        QuestionOpportunityType.IMAGE_TEXT_BINDING,
        QuestionOpportunityType.VISUAL_SEARCH,
        QuestionOpportunityType.PREFERENCE,
        QuestionOpportunityType.TEMPORAL_PAIR,
        QuestionOpportunityType.UPDATE_PAIR,
        QuestionOpportunityType.CONFLICT_PAIR,
        QuestionOpportunityType.MULTI_EVIDENCE,
    ]
    positions = {kind: 0 for kind in order}
    selected: list[_OpportunitySpec] = []
    seen: set[str] = set()
    while len(selected) < limit:
        progressed = False
        for kind in order:
            values = buckets.get(kind, [])
            position = positions[kind]
            while position < len(values):
                spec = values[position]
                position += 1
                identity = canonical_content_hash(spec.identity())
                if identity in seen:
                    continue
                seen.add(identity)
                selected.append(spec)
                progressed = True
                break
            positions[kind] = position
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


_W_RELATIONS = {
    QuestionOpportunityType.TEMPORAL_PAIR,
    QuestionOpportunityType.UPDATE_PAIR,
    QuestionOpportunityType.CONFLICT_PAIR,
    QuestionOpportunityType.MULTI_EVIDENCE,
}
_W_TYPES = (
    QuestionOpportunityType.UPDATE_PAIR, QuestionOpportunityType.PREFERENCE,
    QuestionOpportunityType.IMAGE_TEXT_BINDING, QuestionOpportunityType.CONFLICT_PAIR,
    QuestionOpportunityType.TEMPORAL_PAIR, QuestionOpportunityType.VISUAL_DETAIL,
    QuestionOpportunityType.MULTI_EVIDENCE, QuestionOpportunityType.VISUAL_SEARCH,
    QuestionOpportunityType.LOCAL_FACT,
)
_W_FILLER = re.compile(
    r"(?:ok(?:ay)?|sure|thanks?(?: you)?|thank you(?: so much)?|sounds good|great|"
    r"got it|you're welcome|happy to help)[.! ]*", re.IGNORECASE,
)
_IMAGE_REFERENCE = re.compile(
    r"\b(?:photo(?:graph)?|picture|image|pictured|shown|selfie)\b", re.IGNORECASE
)


def _w_terms(text: str) -> frozenset[str]:
    return frozenset(word for word in _TOKEN.findall(text.casefold()) if word not in _STOP_LEXEMES)


def _relation_supported(kind: QuestionOpportunityType, left: str, right: str) -> bool:
    # The group-level selector proposes a pair; verify that the actual cited
    # sentences (not some other text in those groups) carry the relation cues.
    if left.casefold() == right.casefold():
        return False
    left_values = set(_NUMBER_OR_DATE.findall(left))
    right_values = set(_NUMBER_OR_DATE.findall(right))
    if kind is QuestionOpportunityType.TEMPORAL_PAIR:
        return bool(left_values or right_values or _TEMPORAL_CUE.search(f"{left} {right}"))
    if kind is QuestionOpportunityType.UPDATE_PAIR:
        return bool(left_values and right_values and left_values != right_values) or (
            bool(_UPDATE_CUE.search(left)) != bool(_UPDATE_CUE.search(right))
        )
    if kind is QuestionOpportunityType.CONFLICT_PAIR:
        return bool(_NEGATION_CUE.search(left)) != bool(_NEGATION_CUE.search(right))
    return True


def _w_contract(kind: QuestionOpportunityType, quote: str) -> tuple[list[str], list[str]]:
    text = ["retrieve_text_by_embedding"]
    visual = ["retrieve_image_by_visual_embedding", "inspect_raw_image"]
    if kind is QuestionOpportunityType.VISUAL_SEARCH:
        return ["list"], visual
    if kind is QuestionOpportunityType.IMAGE_TEXT_BINDING:
        return ["entity", "value", "event"], [visual[0], "retrieve_image_text_pair", visual[1]]
    if kind is QuestionOpportunityType.VISUAL_DETAIL:
        return ["entity", "value", "boolean", "count"], visual
    if kind is QuestionOpportunityType.MULTI_EVIDENCE:
        return ["entity", "value", "event", "list"], text
    if kind in _W_RELATIONS:
        return ["value", "event", "boolean"], [*text, "expand_temporal_neighbors"]
    if kind is QuestionOpportunityType.PREFERENCE:
        return ["entity", "value", "boolean", "list"], text
    answers = ["entity", "value", "event"]
    if _NEGATION_CUE.search(quote):
        answers.append("boolean")
    if _NUMBER_OR_DATE.search(quote) or re.search(r"\b(?:one|two|three|four|five)\b", quote, re.I):
        answers.append("count")
    if "," in quote or " and " in quote.casefold():
        answers.append("list")
    return answers, text


def validate_evidence_first_bindings(built: BuiltQuestionOpportunitySet) -> None:
    """A malformed W catalog is a producer failure, never an empty scientific result."""
    image_sources = {alias: image for (_, image), alias in built.image_aliases.items()}
    scan_aliases: dict[str, str] = {}
    for alias, image_id in image_sources.items():
        scan_aliases.setdefault(image_id, alias)
    header = built.opportunity_set
    if header.builder_version != EVIDENCE_FIRST_VERSION:
        raise ValueError("evidence_first_catalog_version_mismatch")
    if header.opportunity_count != len(built.opportunities):
        raise ValueError("evidence_first_catalog_count_mismatch")
    for i, item in enumerate(built.opportunities):
        if (item.opportunity_set_id, item.compiler_unit_id, item.ordinal, item.run_id) != (
            header.opportunity_set_id, header.compiler_unit_id, i, header.run_id
        ):
            raise ValueError("evidence_first_catalog_identity_mismatch")
        meta = item.metadata
        kind = item.opportunity_type
        aliases = meta.get("support_aliases")
        if kind is QuestionOpportunityType.REFUSAL_BOUNDARY:
            if len(built.opportunities) != 1 or item.evidence_aliases or aliases:
                raise ValueError("evidence_first_invalid_control")
            continue
        if (
            not isinstance(aliases, list) or not aliases
            or not all(isinstance(a, str) for a in aliases) or len(set(aliases)) != len(aliases)
            or not meta.get("information_need_key")
        ):
            raise ValueError("evidence_first_unbound_opportunity")
        parents, quotes, images = [], [], []
        for alias in aliases:
            if alias in built.sentence_span_aliases:
                parent, quote = built.sentence_span_aliases[alias]
                parents.append(parent)
                quotes.append(quote)
            elif alias in image_sources:
                parents.append(alias)
                images.append(image_sources[alias])
            else:
                raise ValueError("evidence_first_unknown_support")
        if list(dict.fromkeys(parents)) != item.evidence_aliases or any(
            p not in built.alias_to_evidence for p in parents
        ):
            raise ValueError("evidence_first_support_parent_mismatch")
        groups = {built.alias_to_group.get(p) for p in parents}
        if None in groups or groups != set(item.evidence_group_ids) or (
            len(groups) < item.minimum_evidence_groups
            or item.minimum_evidence_groups != (2 if kind in _W_RELATIONS else 1)
        ):
            raise ValueError("evidence_first_support_group_mismatch")
        answers, path = _w_contract(kind, " ".join(quotes))
        if meta.get("expected_answer_types") != answers or meta.get("operator_path") != path:
            raise ValueError("evidence_first_invalid_contract")
        scan = kind is QuestionOpportunityType.VISUAL_SEARCH
        if bool(meta.get("image_set_scope")) != scan or scan and (
            quotes or not 2 <= len(images) <= 8 or aliases != list(scan_aliases.values())
            or meta.get("image_set_size") != len(images)
        ):
            raise ValueError("evidence_first_unbound_image_scan")
        if kind is QuestionOpportunityType.IMAGE_TEXT_BINDING:
            if len(images) != 1 or len(quotes) != 1:
                raise ValueError("evidence_first_invalid_image_text_binding")
            refs = [built.alias_to_evidence[p] for p in parents]
            owners = {ref.memory_id or ref.metadata.get("owning_memory_id") for ref in refs}
            if len(owners) != 1 or None in owners:
                raise ValueError("evidence_first_invalid_image_text_binding")
        elif not scan and (
            (kind is QuestionOpportunityType.VISUAL_DETAIL and (len(images) != 1 or quotes))
            or (kind is not QuestionOpportunityType.VISUAL_DETAIL and images)
        ):
            raise ValueError("evidence_first_support_modality_mismatch")


def _evidence_first_specs(
    buckets: dict[QuestionOpportunityType, list[_OpportunitySpec]], *,
    memories: dict[str, MemoryUnit], spans: dict[str, tuple[str, str]],
    evidence: dict[str, OperatorEvidence], groups: dict[str, str], limit: int,
) -> list[_OpportunitySpec]:
    """Bind viable needs before either the catalog cap or the answerable budget."""
    by_parent: dict[str, list[str]] = defaultdict(list)
    keywords: dict[str, frozenset[str]] = {}
    for alias, (parent, quote) in spans.items():
        terms = _w_terms(quote)
        if quote.endswith("?") or _W_FILLER.fullmatch(quote) or not terms:
            continue
        by_parent[parent].append(alias)
        # A shared date or update cue is not the entity/attribute being compared.
        keywords[alias] = frozenset(
            t for t in terms if not t.isdecimal()
            and not any(cue.fullmatch(t) for cue in (_UPDATE_CUE, _TEMPORAL_CUE, _NEGATION_CUE))
        )

    # One source image can have multiple owners. Count it only once in a scan.
    images: dict[str, str] = {}
    for alias, ref in evidence.items():
        if ref.image_id:
            images.setdefault(ref.image_id, alias)

    def owner(parent: str) -> MemoryUnit:
        ref = evidence[parent]
        mid = ref.memory_id or ref.metadata.get("owning_memory_id")
        if not isinstance(mid, str):
            raise ValueError("evidence-first support has no owning memory")
        return memories[mid]

    def bindings(spec: _OpportunitySpec) -> Iterator[tuple[str, ...]]:
        kind = spec.opportunity_type
        text = [alias for parent in spec.aliases for alias in by_parent.get(parent, ())]
        if kind in _W_RELATIONS:
            for parent in spec.aliases:
                for caption in captions_by_owner.get(owner(parent).memory_id, ()):
                    text.extend(by_parent.get(caption, ()))
            best = None
            best_score = 0
            # Compare only spans sharing a content word; canonical tie order
            # must not depend on Python's randomized string hashing.
            postings: dict[str, list[int]] = defaultdict(list)
            for index, b in enumerate(text):
                pb, right = spans[b]
                for prior in sorted({i for t in keywords[b] for i in postings[t]}):
                    a = text[prior]
                    pa, left = spans[a]
                    if groups[pa] == groups[pb] or _memory_group(owner(pa)) == (
                        _memory_group(owner(pb))
                    ):
                        continue
                    score = len(keywords[a] & keywords[b])
                    if score > best_score and _relation_supported(kind, left, right):
                        best_score, best = score, (a, b)
                for t in keywords[b]:
                    postings[t].append(index)
            if best:
                yield best
        elif kind is QuestionOpportunityType.VISUAL_SEARCH:
            if 2 <= len(images) <= 8:
                yield tuple(images.values())
        elif kind is QuestionOpportunityType.VISUAL_DETAIL:
            yield spec.aliases
        elif kind is QuestionOpportunityType.IMAGE_TEXT_BINDING:
            image = next(alias for alias in spec.aliases if evidence[alias].image_id)
            memory = owner(image)
            # A turn-level caption or sentence cannot identify which of
            # several images it describes. Individual images remain inspectable.
            if len(_image_ids(memory)) != 1:
                return
            captions = [
                alias for parent in captions_by_owner.get(memory.memory_id, ())
                for alias in by_parent.get(parent, ())
            ]
            caption_terms = frozenset().union(*(keywords[alias] for alias in captions))
            for alias in text:
                if _IMAGE_REFERENCE.search(spans[alias][1]) or keywords[alias] & caption_terms:
                    yield alias, image
            for alias in captions:
                yield alias, image
        else:
            for alias in text:
                parent, quote = spans[alias]
                personal = (
                    evidence[parent].evidence_type is EvidenceType.MEMORY
                    and owner(parent).speaker.value == "user"
                    and _PREFERENCE_CUE.search(quote)
                )
                if bool(personal) == (kind is QuestionOpportunityType.PREFERENCE):
                    yield (alias,)

    captions_by_owner: dict[str, list[str]] = defaultdict(list)
    for alias, ref in evidence.items():
        if ref.evidence_type is EvidenceType.CAPTION:
            captions_by_owner[owner(alias).memory_id].append(alias)

    candidates: dict[str, tuple[_OpportunitySpec, set[str], set[str], set[str]]] = {}
    for kind in _W_TYPES:
        for spec in buckets.get(kind, ()):
            for support in bindings(spec):
                parents = tuple(dict.fromkeys(spans[a][0] if a in spans else a for a in support))
                evidence_groups = {groups[p] for p in parents}
                need_parts = []
                topics: set[str] = set()
                entities: set[str] = set()
                speakers = []
                quotes = []
                for alias in support:
                    if alias in spans:
                        parent, quote = spans[alias]
                        memory = owner(parent)
                        normalized = " ".join(quote.casefold().split()).rstrip(".!")
                        quotes.append(quote)
                        topics.update(_w_terms(quote))
                        entities.update(
                            entity.casefold() for entity in memory.entity_mentions
                            if entity.casefold() in normalized
                        )
                        need_parts.append((memory.speaker.value, normalized))
                    else:
                        ref = evidence[alias]
                        memory = owner(alias)
                        assert ref.image_id is not None
                        need_parts.append(("image", ref.image_id))
                    speakers.append(memory.speaker.value)
                # A generic multi-hop label on the same pair of statements
                # doesn't buy a new need. Retain the more specific relation.
                family = "relation" if kind in _W_RELATIONS else kind.value
                key = canonical_content_hash([family, sorted(need_parts)])
                if key in candidates:
                    continue
                answers, path = _w_contract(kind, " ".join(quotes))
                metadata = dict(spec.metadata)
                metadata.update(
                    support_aliases=list(support), expected_answer_types=answers,
                    operator_path=path, information_need_key=key,
                    evidence_group_count=len(evidence_groups),
                )
                if kind is QuestionOpportunityType.VISUAL_SEARCH:
                    metadata.update(image_set_scope=True, image_set_size=len(support))
                bound = replace(
                    spec, aliases=parents, speaker_scope=tuple(dict.fromkeys(speakers)),
                    metadata=tuple(metadata.items()),
                )
                candidates[key] = (bound, evidence_groups, entities, topics)

    selected: list[_OpportunitySpec] = []
    type_counts: Counter[QuestionOpportunityType] = Counter()
    used_groups: set[str] = set()
    used_entities: set[str] = set()
    used_terms: set[str] = set()
    while candidates and len(selected) < limit:
        # Round-robin type coverage; within it prefer unseen evidence groups
        # and subject matter. Exact ties retain canonical construction order.
        def priority(key: str) -> tuple[int, bool, bool, bool, int]:
            spec, candidate_groups, subjects, words = candidates[key]
            return (
                -type_counts[spec.opportunity_type], bool(subjects - used_entities),
                bool(candidate_groups - used_groups), bool(words - used_terms),
                -_W_TYPES.index(spec.opportunity_type),
            )

        spec, evidence_groups, entities, topics = candidates.pop(max(candidates, key=priority))
        selected.append(spec)
        type_counts[spec.opportunity_type] += 1
        used_groups.update(evidence_groups)
        used_entities.update(entities)
        used_terms.update(topics)
    return selected


class QuestionOpportunityBuilder:
    """Create bounded, prospective-safe opportunities without query or gold access."""

    def __init__(self, config: QuestionerConfig) -> None:
        if not config.is_v3:
            raise ValueError("QuestionOpportunityBuilder requires Questioner v3")
        self.config = config

    def build(
        self,
        *,
        unit: CompilationUnit,
        memories: list[MemoryUnit],
        sessions: list[SessionRecord],
        links: list[CompilationUnitLink],
    ) -> BuiltQuestionOpportunitySet:
        memory_map = {item.memory_id: item for item in memories}
        session_map = {item.session_id: item for item in sessions}
        ordered_ids = [item.memory_id for item in unit.memories]
        if set(ordered_ids) != set(memory_map):
            raise ValueError("opportunity builder requires exactly the unit memories")
        if not {item.session_id for item in memories} <= set(session_map):
            raise ValueError("opportunity builder requires every unit session")
        if len({item.dataset for item in memories}) != 1:
            raise ValueError("opportunity builder cannot mix benchmark datasets")

        memory_aliases = {
            memory_id: f"M{ordinal:02d}"
            for ordinal, memory_id in enumerate(ordered_ids, start=1)
        }
        ordered_session_ids = list(
            dict.fromkeys(memory_map[memory_id].session_id for memory_id in ordered_ids)
        )
        session_aliases = {
            session_id: f"S{ordinal:02d}"
            for ordinal, session_id in enumerate(ordered_session_ids, start=1)
        }
        ordered_image_bindings = [
            (memory_id, image_id)
            for memory_id in ordered_ids
            for image_id in _image_ids(memory_map[memory_id])
        ]
        image_aliases = {
            binding: f"I{ordinal:02d}"
            for ordinal, binding in enumerate(ordered_image_bindings, start=1)
        }
        caption_memory_ids = [
            memory_id for memory_id in ordered_ids if memory_map[memory_id].caption
        ]
        caption_aliases = {
            memory_id: f"C{ordinal:02d}"
            for ordinal, memory_id in enumerate(caption_memory_ids, start=1)
        }
        sentence_span_aliases: dict[str, tuple[str, str]] = {}
        if self.config.is_v34:
            span_ordinal = 1
            for memory_id in ordered_ids:
                memory = memory_map[memory_id]
                parent_alias = memory_aliases[memory_id]
                for span in _sentence_spans(memory.text):
                    sentence_span_aliases[f"T{span_ordinal:03d}"] = (
                        parent_alias,
                        span,
                    )
                    span_ordinal += 1
                if memory.caption and memory_id in caption_aliases:
                    parent_alias = caption_aliases[memory_id]
                    for span in _sentence_spans(memory.caption):
                        sentence_span_aliases[f"T{span_ordinal:03d}"] = (
                            parent_alias,
                            span,
                        )
                        span_ordinal += 1

        alias_to_evidence: dict[str, OperatorEvidence] = {}
        alias_to_group: dict[str, str] = {}
        for memory_id in ordered_ids:
            memory = memory_map[memory_id]
            group = (
                f"memory:{memory.memory_id}"
                if self.config.is_v34 and memory.dataset is DatasetName.MEMLENS
                else _memory_group(memory)
            )
            memory_alias = memory_aliases[memory_id]
            alias_to_evidence[memory_alias] = OperatorEvidence(
                evidence_id=memory_id,
                evidence_type=EvidenceType.MEMORY,
                memory_id=memory_id,
            )
            alias_to_group[memory_alias] = group
            if memory_id in caption_aliases:
                caption_alias = caption_aliases[memory_id]
                alias_to_evidence[caption_alias] = OperatorEvidence(
                    evidence_id=memory_id,
                    evidence_type=EvidenceType.CAPTION,
                    memory_id=memory_id,
                )
                alias_to_group[caption_alias] = group
            for image_id in _image_ids(memory):
                image_alias = image_aliases[(memory_id, image_id)]
                alias_to_evidence[image_alias] = OperatorEvidence(
                    evidence_id=image_id,
                    evidence_type=EvidenceType.IMAGE,
                    image_id=image_id,
                    metadata={"owning_memory_id": memory_id},
                )
                alias_to_group[image_alias] = group
        for session_id, alias in session_aliases.items():
            alias_to_evidence[alias] = OperatorEvidence(
                evidence_id=session_id,
                evidence_type=EvidenceType.SESSION,
                session_id=session_id,
            )
            alias_to_group[alias] = f"session:{session_id}"

        member_roles = {item.memory_id: item.role for item in unit.memories}
        anchors = [
            memory_map[memory_id]
            for memory_id in ordered_ids
            if member_roles[memory_id] is CompilationUnitMemoryRole.ANCHOR
        ]
        link_by_pair = {
            (item.anchor_memory_id, item.support_memory_id): item for item in links
        }
        buckets: dict[QuestionOpportunityType, list[_OpportunitySpec]] = defaultdict(list)

        for anchor in anchors:
            memory_alias = memory_aliases[anchor.memory_id]
            text_aliases: tuple[str, ...] = (memory_alias,)
            if anchor.caption and anchor.memory_id in caption_aliases:
                text_aliases = (memory_alias, caption_aliases[anchor.memory_id])
            if _text(anchor):
                buckets[QuestionOpportunityType.LOCAL_FACT].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.LOCAL_FACT,
                        question_types=(QuestionerV2QuestionType.TEXTUAL_FACT,),
                        reasoning_types=(QuestionReasoningType.SINGLE_HOP,),
                        modalities=(QuestionerV2Modality.TEXT,),
                        aliases=text_aliases,
                        metadata=(("evidence_group_count", 1),),
                        speaker_scope=(anchor.speaker.value,),
                    )
                )
            if _PREFERENCE_CUE.search(_text(anchor)):
                buckets[QuestionOpportunityType.PREFERENCE].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.PREFERENCE,
                        question_types=(QuestionerV2QuestionType.PERSONAL_PREFERENCE,),
                        reasoning_types=(QuestionReasoningType.SINGLE_HOP,),
                        modalities=(QuestionerV2Modality.TEXT,),
                        aliases=text_aliases,
                        metadata=(("evidence_group_count", 1),),
                        speaker_scope=(anchor.speaker.value,),
                    )
                )
            for image_id in _image_ids(anchor):
                image_alias = image_aliases[(anchor.memory_id, image_id)]
                buckets[QuestionOpportunityType.VISUAL_DETAIL].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.VISUAL_DETAIL,
                        question_types=(QuestionerV2QuestionType.VISUAL_DETAIL,),
                        reasoning_types=(QuestionReasoningType.SINGLE_HOP,),
                        modalities=(QuestionerV2Modality.IMAGE,),
                        aliases=(image_alias,),
                        metadata=(("evidence_group_count", 1),),
                        speaker_scope=(anchor.speaker.value,),
                    )
                )
                buckets[QuestionOpportunityType.IMAGE_TEXT_BINDING].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.IMAGE_TEXT_BINDING,
                        question_types=(QuestionerV2QuestionType.IMAGE_TEXT_BINDING,),
                        reasoning_types=(QuestionReasoningType.SINGLE_HOP,),
                        modalities=(QuestionerV2Modality.TEXT, QuestionerV2Modality.IMAGE),
                        aliases=(memory_alias, image_alias),
                        metadata=(("evidence_group_count", 1),),
                        speaker_scope=(anchor.speaker.value,),
                    )
                )
                buckets[QuestionOpportunityType.VISUAL_SEARCH].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.VISUAL_SEARCH,
                        question_types=(QuestionerV2QuestionType.VISUAL_SEARCH,),
                        reasoning_types=(QuestionReasoningType.SINGLE_HOP,),
                        modalities=(QuestionerV2Modality.IMAGE,),
                        aliases=(image_alias,),
                        metadata=(("evidence_group_count", 1),),
                        speaker_scope=(anchor.speaker.value,),
                    )
                )

        lexical_scores = _lexical_scores(memories)
        representatives: dict[str, MemoryUnit] = {}
        group_members: dict[str, list[MemoryUnit]] = defaultdict(list)
        for memory in memories:
            group = _memory_group(memory)
            representatives.setdefault(group, memory)
            group_members[group].append(memory)
        relation_pairs: list[
            tuple[float, MemoryUnit, MemoryUnit, CompilationUnitLink | None]
        ] = []
        for anchor in anchors:
            for other in representatives.values():
                if _memory_group(anchor) == _memory_group(other):
                    continue
                anchor_group = group_members[_memory_group(anchor)]
                other_group = group_members[_memory_group(other)]
                links_for_groups = [
                    link_by_pair[(left.memory_id, right.memory_id)]
                    for left in anchor_group
                    for right in other_group
                    if (left.memory_id, right.memory_id) in link_by_pair
                ]
                link = min(links_for_groups, key=lambda item: item.link_id, default=None)
                score = max(
                    (
                        lexical_scores.get(_pair_key(left, right), 0.0)
                        for left in anchor_group
                        for right in other_group
                    ),
                    default=0.0,
                )
                if (
                    score <= 0
                    and link is None
                    and not (
                        {
                            image_id
                            for value in anchor_group
                            for image_id in _image_ids(value)
                        }
                        & {
                            image_id
                            for value in other_group
                            for image_id in _image_ids(value)
                        }
                    )
                ):
                    continue
                relation_pairs.append((score, anchor, other, link))
        relation_pairs.sort(
            key=lambda item: (
                -item[0],
                _memory_group(item[1]),
                _memory_group(item[2]),
                item[1].memory_id,
                item[2].memory_id,
            )
        )
        seen_pairs: set[tuple[str, str]] = set()
        accepted_pair_count = 0
        for score, anchor, other, link in relation_pairs:
            left_group, right_group = sorted(
                (_memory_group(anchor), _memory_group(other))
            )
            groups = (left_group, right_group)
            if groups in seen_pairs:
                continue
            seen_pairs.add(groups)
            anchor_members = group_members[_memory_group(anchor)]
            other_members = group_members[_memory_group(other)]
            aliases = tuple(
                memory_aliases[item.memory_id]
                for item in (*anchor_members, *other_members)
            )
            relation_ids = tuple(
                sorted(
                    {
                        value.link_id
                        for left in anchor_members
                        for right in other_members
                        for value in (
                            [link_by_pair[(left.memory_id, right.memory_id)]]
                            if (left.memory_id, right.memory_id) in link_by_pair
                            else []
                        )
                    }
                )
            )
            anchor_tokens = {
                token for item in anchor_members for token in _tokens(item)
            }
            other_tokens = {
                token for item in other_members for token in _tokens(item)
            }
            shared = anchor_tokens & other_tokens
            anchor_text = " ".join(_text(item) for item in anchor_members)
            other_text = " ".join(_text(item) for item in other_members)
            anchor_values = set(_NUMBER_OR_DATE.findall(anchor_text))
            other_values = set(_NUMBER_OR_DATE.findall(other_text))
            combined_text = f"{anchor_text} {other_text}"
            shared_images = {
                image_id
                for item in anchor_members
                for image_id in _image_ids(item)
            } & {
                image_id
                for item in other_members
                for image_id in _image_ids(item)
            }
            metadata = (
                ("evidence_group_count", 2),
                ("lexical_score_micros", round(score * 1_000_000)),
                ("shared_lexeme_count", len(shared)),
                ("shared_image_count", len(shared_images)),
            )
            temporal_relation = bool(
                anchor_values or other_values or _TEMPORAL_CUE.search(combined_text)
            )
            relation_is_semantic = bool(shared)
            if self.config.is_v32 or self.config.is_v33 or self.config.is_v34:
                # FTS only proposes pairs. A v3.2 relation still needs explicit
                # entity/property content or an exact image binding. Adjacency
                # and one low-information lexical hit are not relations.
                relation_is_semantic = bool(shared_images) or len(shared) >= 2 or (
                    len(shared) == 1
                    and score >= 1.5
                    and len(next(iter(shared))) >= 5
                )
            added_relation = False
            if temporal_relation and relation_is_semantic:
                buckets[QuestionOpportunityType.TEMPORAL_PAIR].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.TEMPORAL_PAIR,
                        question_types=(QuestionerV2QuestionType.TEMPORAL,),
                        reasoning_types=(QuestionReasoningType.TEMPORAL,),
                        modalities=(QuestionerV2Modality.TEXT,),
                        aliases=aliases,
                        relation_ids=relation_ids,
                        minimum_evidence_groups=2,
                        metadata=metadata,
                        speaker_scope=tuple(
                            item.speaker.value
                            for item in (*anchor_members, *other_members)
                        ),
                    )
                )
                added_relation = True
            values_changed = bool(anchor_values and other_values and anchor_values != other_values)
            if relation_is_semantic and shared and (
                values_changed
                or (
                    bool(_UPDATE_CUE.search(anchor_text))
                    != bool(_UPDATE_CUE.search(other_text))
                )
            ):
                buckets[QuestionOpportunityType.UPDATE_PAIR].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.UPDATE_PAIR,
                        question_types=(QuestionerV2QuestionType.KNOWLEDGE_UPDATE,),
                        reasoning_types=(QuestionReasoningType.UPDATE,),
                        modalities=(QuestionerV2Modality.TEXT,),
                        aliases=aliases,
                        relation_ids=relation_ids,
                        minimum_evidence_groups=2,
                        metadata=metadata,
                        speaker_scope=tuple(
                            item.speaker.value
                            for item in (*anchor_members, *other_members)
                        ),
                    )
                )
                added_relation = True
            left_negated = bool(_NEGATION_CUE.search(anchor_text))
            right_negated = bool(_NEGATION_CUE.search(other_text))
            if relation_is_semantic and shared and left_negated != right_negated:
                buckets[QuestionOpportunityType.CONFLICT_PAIR].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.CONFLICT_PAIR,
                        question_types=(QuestionerV2QuestionType.CONFLICT_DETECTION,),
                        reasoning_types=(QuestionReasoningType.CONFLICT,),
                        modalities=(QuestionerV2Modality.TEXT,),
                        aliases=aliases,
                        relation_ids=relation_ids,
                        minimum_evidence_groups=2,
                        metadata=metadata,
                        speaker_scope=tuple(
                            item.speaker.value
                            for item in (*anchor_members, *other_members)
                        ),
                    )
                )
                added_relation = True
            if relation_is_semantic or (
                link is not None
                and not (
                    self.config.is_v31
                    or self.config.is_v32
                    or self.config.is_v33
                    or self.config.is_v34
                )
            ):
                buckets[QuestionOpportunityType.MULTI_EVIDENCE].append(
                    _OpportunitySpec(
                        opportunity_type=QuestionOpportunityType.MULTI_EVIDENCE,
                        question_types=(QuestionerV2QuestionType.MULTI_HOP,),
                        reasoning_types=(QuestionReasoningType.MULTI_HOP,),
                        modalities=(QuestionerV2Modality.TEXT,),
                        aliases=aliases,
                        relation_ids=relation_ids,
                        minimum_evidence_groups=2,
                        metadata=metadata,
                        speaker_scope=tuple(
                            item.speaker.value
                            for item in (*anchor_members, *other_members)
                        ),
                    )
                )
                added_relation = True
            if added_relation:
                accepted_pair_count += 1
            if accepted_pair_count >= self.config.fts_relation_top_k:
                break

        refusal = _OpportunitySpec(
            opportunity_type=QuestionOpportunityType.REFUSAL_BOUNDARY,
            question_types=(QuestionerV2QuestionType.ANSWER_REFUSAL,),
            reasoning_types=(QuestionReasoningType.REFUSAL,),
            modalities=(QuestionerV2Modality.TEXT,),
            aliases=(),
            minimum_evidence_groups=1,
            metadata=(("evidence_group_count", 0),),
            speaker_scope=(),
        )
        if self.config.evidence_first:
            specs = _evidence_first_specs(
                buckets, memories=memory_map, spans=sentence_span_aliases,
                evidence=alias_to_evidence, groups=alias_to_group,
                limit=self.config.max_opportunities_per_unit,
            ) or [refusal]
        elif self.config.is_v33 or self.config.is_v34:
            specs = _select_diverse(
                buckets,
                limit=self.config.max_opportunities_per_unit,
            )
            # v3.3 no longer creates one refusal for every unit. A refusal
            # boundary is a control fallback only when no answerable
            # opportunity survived deterministic grounding.
            if not specs:
                specs.append(refusal)
        else:
            specs = _select_diverse(
                buckets,
                limit=max(0, self.config.max_opportunities_per_unit - 1),
            )
            specs.append(refusal)

        specs_payload = [item.identity() for item in specs]
        set_identity = {
            "run_id": unit.run_id,
            "compiler_unit_id": unit.compiler_unit_id,
            "builder_version": self.config.opportunity_builder_version,
            "specs": specs_payload,
        }
        set_hash = canonical_content_hash(set_identity)
        set_id = stable_compiler_id("question-opportunity-set", set_identity)
        opportunities: list[QuestionOpportunity] = []
        refs: list[QuestionOpportunityEvidenceRef] = []
        for ordinal, spec in enumerate(specs):
            local_id = f"O{ordinal + 1:02d}"
            stable_id = stable_compiler_id(
                "question-opportunity",
                [set_id, local_id, spec.identity()],
            )
            content_hash = canonical_content_hash(spec.identity())
            evidence_group_ids = list(
                dict.fromkeys(alias_to_group[alias] for alias in spec.aliases)
            )
            relation_content_hash = (
                canonical_content_hash(
                    {
                        "groups": evidence_group_ids,
                        "relations": list(spec.relation_ids),
                        "speaker_scope": list(dict.fromkeys(spec.speaker_scope)),
                    }
                )
                if spec.aliases
                else None
            )
            opportunities.append(
                QuestionOpportunity(
                    opportunity_id=stable_id,
                    opportunity_set_id=set_id,
                    run_id=unit.run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    snapshot_id=unit.snapshot_id,
                    conversation_id=unit.conversation_id,
                    ordinal=ordinal,
                    opportunity_type=spec.opportunity_type,
                    allowed_question_types=list(spec.question_types),
                    allowed_reasoning_types=list(spec.reasoning_types),
                    required_modalities=list(spec.modalities),
                    minimum_evidence_groups=spec.minimum_evidence_groups,
                    evidence_aliases=list(spec.aliases),
                    relation_link_ids=list(spec.relation_ids),
                    speaker_scope=list(dict.fromkeys(spec.speaker_scope)),
                    evidence_group_ids=evidence_group_ids,
                    relation_content_hash=relation_content_hash,
                    content_hash=content_hash,
                    metadata={"local_opportunity_id": local_id, **dict(spec.metadata)},
                )
            )
            for evidence_ordinal, alias in enumerate(spec.aliases):
                evidence = alias_to_evidence[alias]
                refs.append(
                    QuestionOpportunityEvidenceRef(
                        opportunity_evidence_ref_id=stable_compiler_id(
                            "question-opportunity-evidence",
                            [stable_id, evidence_ordinal, alias],
                        ),
                        run_id=unit.run_id,
                        compiler_unit_id=unit.compiler_unit_id,
                        opportunity_id=stable_id,
                        alias=alias,
                        ordinal=evidence_ordinal,
                        evidence=evidence,
                        evidence_group_id=alias_to_group[alias],
                        content_hash=canonical_content_hash(
                            [stable_id, alias, evidence.model_dump(mode="json")]
                        ),
                    )
                )
        return BuiltQuestionOpportunitySet(
            opportunity_set=QuestionOpportunitySet(
                opportunity_set_id=set_id,
                run_id=unit.run_id,
                compiler_unit_id=unit.compiler_unit_id,
                snapshot_id=unit.snapshot_id,
                conversation_id=unit.conversation_id,
                builder_version=self.config.opportunity_builder_version,
                opportunity_count=len(opportunities),
                content_hash=set_hash,
                metadata={
                    "dataset": memories[0].dataset.value if memories else None,
                    "opportunity_type_counts": dict(
                        sorted(
                            Counter(
                                item.opportunity_type.value for item in opportunities
                            ).items()
                        )
                    ),
                },
            ),
            opportunities=tuple(opportunities),
            evidence_refs=tuple(refs),
            alias_to_evidence=alias_to_evidence,
            alias_to_group=alias_to_group,
            memory_aliases=memory_aliases,
            session_aliases=session_aliases,
            image_aliases=image_aliases,
            caption_aliases=caption_aliases,
            sentence_span_aliases=sentence_span_aliases,
        )


def local_opportunity_id(value: QuestionOpportunity) -> str:
    result = value.metadata.get("local_opportunity_id")
    if not isinstance(result, str) or not result:
        raise ValueError("opportunity is missing its local alias")
    return result


def evidence_modalities(values: list[OperatorEvidence]) -> list[Modality]:
    modalities = [Modality.TEXT]
    if any(item.image_id is not None for item in values):
        modalities.append(Modality.IMAGE)
    return modalities
