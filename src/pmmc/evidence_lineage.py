"""Directional canonical evidence lineage shared by evaluation and compiler dry-runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pmmc.schemas import (
    DatasetName,
    EvidenceMatchRole,
    EvidenceMemberType,
    MemoryUnit,
    ResolvedGoldEvidenceGroup,
)


class EvidenceCandidate(Protocol):
    @property
    def memory_ids(self) -> frozenset[str]: ...

    @property
    def session_ids(self) -> frozenset[str]: ...

    @property
    def image_ids(self) -> frozenset[str]: ...

    @property
    def evidence_ids(self) -> frozenset[str]: ...


class EvidenceResolutionPolicy(StrEnum):
    """Explicit evidence domains; runtime policy never consumes benchmark gold."""

    CANDIDATE_SUPPORT_EXACT = "candidate_support_exact"
    BENCHMARK_GOLD_RESOLUTION = "benchmark_gold_resolution"
    QUERY_RUNTIME_STRUCTURAL = "query_runtime_structural"


def is_memgallery_character_profile(metadata: object) -> bool:
    """Recognize the adapter's roundless profile, not an incomplete dialogue."""
    return (
        isinstance(metadata, dict)
        and metadata.get("unit_kind") == "character_profile"
        and metadata.get("source_round_id") in (None, "")
        and isinstance(metadata.get("source"), dict)
        and metadata["source"].get("source_session_id") == "profile"
        and metadata["source"].get("source_round_id") in (None, "")
    )


def memgallery_profile_group(memory_id: str, session_id: str, metadata: dict) -> str:
    # Preserve Questioner v3.6's existing grouping without changing its producer.
    return f"profile:{session_id}" if metadata.get("is_synthetic") else f"memory:{memory_id}"


@dataclass(frozen=True)
class EvidenceClosure:
    memory_ids: frozenset[str] = frozenset()
    session_ids: frozenset[str] = frozenset()
    image_ids: frozenset[str] = frozenset()
    caption_ids: frozenset[str] = frozenset()
    evidence_group_ids: frozenset[str] = frozenset()


class EvidenceLineageResolver:
    """Expand evidence upward only; broad ancestors never imply specific children."""

    def __init__(self, memories: Iterable[MemoryUnit]) -> None:
        self._memory_session: dict[str, str] = {}
        self._image_memories: dict[str, set[str]] = {}
        self._caption_memories: dict[str, set[str]] = {}
        self._memory_group: dict[str, str] = {}
        self._session_datasets: dict[str, set[DatasetName]] = {}
        for memory in memories:
            self._memory_session[memory.memory_id] = memory.session_id
            self._session_datasets.setdefault(memory.session_id, set()).add(
                memory.dataset
            )
            for image_id in {
                *memory.image_ids,
                *(item.image_id for item in memory.image_refs),
            }:
                self._image_memories.setdefault(image_id, set()).add(memory.memory_id)
            caption_id = memory.metadata.get("caption_id")
            if isinstance(caption_id, str) and caption_id:
                self._caption_memories.setdefault(caption_id, set()).add(memory.memory_id)
            if memory.dataset is DatasetName.MEMGALLERY:
                round_id = memory.metadata.get("source_round_id")
                if isinstance(round_id, (str, int)) and str(round_id):
                    self._memory_group[memory.memory_id] = f"round:{round_id}"
                elif is_memgallery_character_profile(memory.metadata):
                    self._memory_group[memory.memory_id] = memgallery_profile_group(
                        memory.memory_id, memory.session_id, memory.metadata
                    )
            else:
                self._memory_group[memory.memory_id] = f"session:{memory.session_id}"

    def expand(
        self,
        *,
        memory_ids: Iterable[str] = (),
        session_ids: Iterable[str] = (),
        image_ids: Iterable[str] = (),
        caption_ids: Iterable[str] = (),
        evidence_ids: Iterable[str] = (),
    ) -> EvidenceClosure:
        memories = set(memory_ids)
        explicit_sessions = set(session_ids)
        sessions = set(explicit_sessions)
        images = set(image_ids)
        captions = set(caption_ids)
        for image_id in images:
            memories.update(self._image_memories.get(image_id, ()))
        for caption_id in captions:
            memories.update(self._caption_memories.get(caption_id, ()))
        sessions.update(
            session_id
            for memory_id in memories
            if (session_id := self._memory_session.get(memory_id)) is not None
        )
        groups = {
            group_id
            for memory_id in memories
            if (group_id := self._memory_group.get(memory_id)) is not None
        }
        # A memory/image may derive its owning session for lineage reporting, but
        # that broad ancestor must not collapse distinct MemGallery source-round
        # evidence groups. Explicit session evidence remains a valid group; for
        # MEMLENS, memory groups are already represented as ``session:<id>``.
        groups.update(f"session:{value}" for value in explicit_sessions)
        groups.update(evidence_ids)
        return EvidenceClosure(
            memory_ids=frozenset(memories),
            session_ids=frozenset(sessions),
            image_ids=frozenset(images),
            caption_ids=frozenset(captions),
            evidence_group_ids=frozenset(groups),
        )

    def resolve(
        self,
        *,
        policy: EvidenceResolutionPolicy,
        memory_ids: Iterable[str] = (),
        session_ids: Iterable[str] = (),
        image_ids: Iterable[str] = (),
        caption_ids: Iterable[str] = (),
        evidence_ids: Iterable[str] = (),
    ) -> EvidenceClosure:
        """Resolve a typed evidence set under an explicit non-downward policy.

        All three policies share the same safe upward closure. Their separation
        prevents evaluator-only gold semantics from being selected implicitly by
        query-time code and gives audits a stable policy identity.
        """

        if policy not in EvidenceResolutionPolicy:
            raise ValueError(f"unsupported evidence resolution policy: {policy}")
        return self.expand(
            memory_ids=memory_ids,
            session_ids=session_ids,
            image_ids=image_ids,
            caption_ids=caption_ids,
            evidence_ids=evidence_ids,
        )

    def group_ids_for(
        self,
        *,
        memory_id: str | None = None,
        session_id: str | None = None,
        image_id: str | None = None,
        caption_id: str | None = None,
        evidence_id: str | None = None,
    ) -> frozenset[str]:
        return self.expand(
            memory_ids=(() if memory_id is None else (memory_id,)),
            session_ids=(() if session_id is None else (session_id,)),
            image_ids=(() if image_id is None else (image_id,)),
            caption_ids=(() if caption_id is None else (caption_id,)),
            evidence_ids=(() if evidence_id is None else (evidence_id,)),
        ).evidence_group_ids

    def group_roles_for(
        self,
        *,
        memory_id: str | None = None,
        session_id: str | None = None,
        image_id: str | None = None,
        caption_id: str | None = None,
        evidence_id: str | None = None,
    ) -> dict[str, EvidenceMatchRole]:
        """Resolve evidence groups without promoting broad MemGallery sessions.

        Memory, image, and caption evidence always resolves upward to its
        discriminative owning group. An explicitly supplied MemGallery session
        remains contextual because it cannot identify one source round.
        """

        groups = self.group_ids_for(
            memory_id=memory_id,
            session_id=session_id,
            image_id=image_id,
            caption_id=caption_id,
            evidence_id=evidence_id,
        )
        roles = {
            group_id: EvidenceMatchRole.DISCRIMINATIVE for group_id in groups
        }
        if session_id is not None and DatasetName.MEMGALLERY in self._session_datasets.get(
            session_id, set()
        ):
            session_group = f"session:{session_id}"
            if session_group in roles:
                roles[session_group] = EvidenceMatchRole.CONTEXTUAL
        return roles

    def match_details(
        self,
        candidate: EvidenceCandidate,
        groups: tuple[ResolvedGoldEvidenceGroup, ...],
        *,
        policy: EvidenceResolutionPolicy = EvidenceResolutionPolicy.BENCHMARK_GOLD_RESOLUTION,
    ) -> dict[str, list[str]]:
        if policy is EvidenceResolutionPolicy.QUERY_RUNTIME_STRUCTURAL:
            raise ValueError("query runtime structural resolution cannot consume gold groups")
        direct: set[str] = set()
        derived: set[str] = set()
        contextual: set[str] = set()
        closure = self.resolve(
            policy=policy,
            memory_ids=candidate.memory_ids,
            session_ids=candidate.session_ids,
            image_ids=candidate.image_ids,
            evidence_ids=candidate.evidence_ids,
        )
        for group in groups:
            for member in group.members:
                direct_match = (
                    (
                        member.member_type is EvidenceMemberType.MEMORY
                        and member.member_id in candidate.memory_ids
                    )
                    or (
                        member.member_type is EvidenceMemberType.SESSION
                        and member.member_id in candidate.session_ids
                    )
                    or (
                        member.member_type is EvidenceMemberType.IMAGE
                        and member.member_id in candidate.image_ids
                    )
                    or (
                        member.member_type is EvidenceMemberType.EVIDENCE
                        and member.member_id in candidate.evidence_ids
                    )
                )
                derived_match = (
                    (
                        member.member_type is EvidenceMemberType.MEMORY
                        and member.member_id in closure.memory_ids
                    )
                    or (
                        member.member_type is EvidenceMemberType.SESSION
                        and member.member_id in closure.session_ids
                    )
                    or (
                        member.member_type is EvidenceMemberType.IMAGE
                        and member.member_id in closure.image_ids
                    )
                    or (
                        member.member_type is EvidenceMemberType.EVIDENCE
                        and (
                            member.member_id in closure.evidence_group_ids
                            or member.member_id in candidate.evidence_ids
                        )
                    )
                )
                if (
                    (direct_match or derived_match)
                    and member.match_role is EvidenceMatchRole.CONTEXTUAL
                ):
                    contextual.add(group.gold_group_id)
                elif direct_match:
                    direct.add(group.gold_group_id)
                elif derived_match:
                    derived.add(group.gold_group_id)
        return {
            "direct_gold_group_ids": sorted(direct),
            "derived_gold_group_ids": sorted(derived - direct),
            "contextual_only_gold_group_ids": sorted(
                contextual - direct - derived
            ),
            "matched_gold_group_ids": sorted(direct | derived),
        }

    def matched_gold_group_ids(
        self,
        candidate: EvidenceCandidate,
        groups: tuple[ResolvedGoldEvidenceGroup, ...],
        *,
        policy: EvidenceResolutionPolicy = EvidenceResolutionPolicy.BENCHMARK_GOLD_RESOLUTION,
    ) -> set[str]:
        if policy is EvidenceResolutionPolicy.QUERY_RUNTIME_STRUCTURAL:
            raise ValueError("query runtime structural resolution cannot consume gold groups")
        closure = self.resolve(
            policy=policy,
            memory_ids=candidate.memory_ids,
            session_ids=candidate.session_ids,
            image_ids=candidate.image_ids,
            evidence_ids=candidate.evidence_ids,
        )
        matched: set[str] = set()
        for group in groups:
            for member in group.members:
                if member.match_role is EvidenceMatchRole.CONTEXTUAL:
                    continue
                if member.member_type is EvidenceMemberType.MEMORY and (
                    member.member_id in closure.memory_ids
                ):
                    matched.add(group.gold_group_id)
                elif member.member_type is EvidenceMemberType.SESSION and (
                    member.member_id in closure.session_ids
                ):
                    matched.add(group.gold_group_id)
                elif member.member_type is EvidenceMemberType.IMAGE and (
                    member.member_id in closure.image_ids
                ):
                    matched.add(group.gold_group_id)
                elif member.member_type is EvidenceMemberType.EVIDENCE and (
                    member.member_id in closure.evidence_group_ids
                    or member.member_id in candidate.evidence_ids
                ):
                    matched.add(group.gold_group_id)
        return matched


__all__ = [
    "EvidenceCandidate",
    "EvidenceClosure",
    "EvidenceLineageResolver",
    "EvidenceResolutionPolicy",
]
