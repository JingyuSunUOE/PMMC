"""Internal source-neutral objects exchanged by benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

from pmmc.schemas import (
    DatasetName,
    EvaluationEligibility,
    MemoryUnit,
    QueryRecord,
    SessionRecord,
    SourceLocator,
)


@dataclass(slots=True)
class AdaptedQuery:
    """Import-time query record before physical runtime/gold separation."""

    record: QueryRecord
    source: SourceLocator
    eligibility: EvaluationEligibility


@dataclass(slots=True)
class CanonicalConversation:
    """One independently streamable canonical conversation."""

    dataset: DatasetName
    conversation_id: str
    sessions: list[SessionRecord] = field(default_factory=list)
    memory_units: list[MemoryUnit] = field(default_factory=list)
    queries: list[AdaptedQuery] = field(default_factory=list)
