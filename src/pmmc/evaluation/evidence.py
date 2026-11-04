"""Versioned gold resolution and evidence metrics for baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pmmc.benchmarks.ids import stable_hash
from pmmc.schemas import (
    ContextEvidenceEvaluation,
    DatasetName,
    EvaluationContextSourceKind,
    EvidenceGranularity,
    EvidenceMatchRole,
    EvidenceMemberType,
    MemoryUnit,
    MetricFamily,
    MetricScore,
    QueryGold,
    ResolvedGoldEvidenceGroup,
    ResolvedGoldEvidenceMember,
    RetrievalEvaluation,
)


@dataclass(frozen=True)
class RankedEvidenceCandidate:
    candidate_id: str
    rank: int
    memory_ids: frozenset[str] = field(default_factory=frozenset)
    session_ids: frozenset[str] = field(default_factory=frozenset)
    image_ids: frozenset[str] = field(default_factory=frozenset)
    evidence_ids: frozenset[str] = field(default_factory=frozenset)
    text_memory_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("evidence candidate rank must start at one")


@dataclass(frozen=True)
class EvidenceMetricBundle:
    groups: tuple[ResolvedGoldEvidenceGroup, ...]
    retrieval_evaluations: tuple[RetrievalEvaluation, ...]
    context_evaluations: tuple[ContextEvidenceEvaluation, ...]
    metrics: tuple[MetricScore, ...]


def _member(
    member_type: EvidenceMemberType,
    member_id: str,
    *,
    derived: bool = False,
    match_role: EvidenceMatchRole = EvidenceMatchRole.DISCRIMINATIVE,
):
    return ResolvedGoldEvidenceMember(
        member_type=member_type,
        member_id=member_id,
        derived=derived,
        match_role=match_role,
    )


def resolve_gold_evidence(
    *,
    evaluation_run_id: str,
    snapshot_id: str,
    query_id: str,
    dataset: DatasetName,
    gold: QueryGold,
    memories: list[MemoryUnit],
    policy_version: str = "evidence-resolution-v1",
    created_at: datetime | None = None,
) -> tuple[ResolvedGoldEvidenceGroup, ...]:
    """Resolve dataset-specific annotations without inventing finer gold labels."""

    if policy_version not in {"evidence-resolution-v1", "evidence-resolution-v2"}:
        raise ValueError(f"unsupported evidence resolution policy: {policy_version}")
    hardened_lineage = policy_version == "evidence-resolution-v2"
    memory_by_id = {item.memory_id: item for item in memories}
    groups: list[ResolvedGoldEvidenceGroup] = []

    if dataset is DatasetName.MEMLENS:
        for session_id in gold.evidence.session_ids:
            group_id = "gold-group-" + stable_hash(
                [evaluation_run_id, query_id, "session", session_id]
            )[:24]
            groups.append(
                ResolvedGoldEvidenceGroup(
                    gold_group_id=group_id,
                    evaluation_run_id=evaluation_run_id,
                    snapshot_id=snapshot_id,
                    query_id=query_id,
                    granularity=EvidenceGranularity.SESSION,
                    source_gold_id=session_id,
                    members=[_member(EvidenceMemberType.SESSION, session_id)],
                    resolution_policy_version=policy_version,
                    **({"created_at": created_at} if created_at is not None else {}),
                )
            )
        return tuple(groups)

    if dataset is DatasetName.MEMGALLERY and gold.evidence.evidence_ids:
        source_clues = gold.metadata.get("source_clue_ids")
        clue_values = source_clues if isinstance(source_clues, list) else []
        for index, evidence_id in enumerate(gold.evidence.evidence_ids):
            source_clue = str(clue_values[index]) if index < len(clue_values) else None
            matching = [
                memory_by_id[memory_id]
                for memory_id in gold.evidence.memory_ids
                if memory_id in memory_by_id
                and (
                    source_clue is None
                    or str(memory_by_id[memory_id].metadata.get("source_round_id"))
                    == source_clue
                )
            ]
            members = [_member(EvidenceMemberType.EVIDENCE, evidence_id)]
            derived_image_ids: set[str] = set()
            for memory in matching:
                members.append(_member(EvidenceMemberType.MEMORY, memory.memory_id))
                members.append(
                    _member(
                        EvidenceMemberType.SESSION,
                        memory.session_id,
                        derived=True,
                        match_role=(
                            EvidenceMatchRole.CONTEXTUAL
                            if hardened_lineage
                            else EvidenceMatchRole.DISCRIMINATIVE
                        ),
                    )
                )
                derived_image_ids.update(memory.image_ids)
            explicit_image_ids = derived_image_ids & set(gold.evidence.image_ids)
            if explicit_image_ids:
                for image_id in sorted(explicit_image_ids):
                    members.append(_member(EvidenceMemberType.IMAGE, image_id))
            elif len(derived_image_ids) == 1:
                members.append(
                    _member(EvidenceMemberType.IMAGE, next(iter(derived_image_ids)), derived=True)
                )
            deduplicated = {
                (member.member_type, member.member_id): member for member in members
            }
            group_id = "gold-group-" + stable_hash(
                [evaluation_run_id, query_id, "round", evidence_id]
            )[:24]
            groups.append(
                ResolvedGoldEvidenceGroup(
                    gold_group_id=group_id,
                    evaluation_run_id=evaluation_run_id,
                    snapshot_id=snapshot_id,
                    query_id=query_id,
                    granularity=EvidenceGranularity.MEMORY,
                    source_gold_id=evidence_id,
                    members=list(deduplicated.values()),
                    resolution_policy_version=policy_version,
                    metadata={"source_clue_id": source_clue} if source_clue else {},
                    **({"created_at": created_at} if created_at is not None else {}),
                )
            )
        return tuple(groups)

    # Conservative generic fallback for future adapters: use the finest explicit
    # annotation available and never derive turn/image labels from session-only gold.
    explicit: tuple[tuple[EvidenceGranularity, EvidenceMemberType, list[str]], ...] = (
        (EvidenceGranularity.MEMORY, EvidenceMemberType.MEMORY, gold.evidence.memory_ids),
        (EvidenceGranularity.IMAGE, EvidenceMemberType.IMAGE, gold.evidence.image_ids),
        (EvidenceGranularity.SESSION, EvidenceMemberType.SESSION, gold.evidence.session_ids),
        (EvidenceGranularity.GENERIC, EvidenceMemberType.EVIDENCE, gold.evidence.evidence_ids),
    )
    for granularity, member_type, identifiers in explicit:
        if identifiers:
            for identifier in identifiers:
                groups.append(
                    ResolvedGoldEvidenceGroup(
                        gold_group_id="gold-group-"
                        + stable_hash(
                            [evaluation_run_id, query_id, granularity.value, identifier]
                        )[:24],
                        evaluation_run_id=evaluation_run_id,
                        snapshot_id=snapshot_id,
                        query_id=query_id,
                        granularity=granularity,
                        source_gold_id=identifier,
                        members=[_member(member_type, identifier)],
                        resolution_policy_version=policy_version,
                        **({"created_at": created_at} if created_at is not None else {}),
                    )
                )
            break
    return tuple(groups)


def _group_member_ids(
    group: ResolvedGoldEvidenceGroup, member_type: EvidenceMemberType
) -> set[str]:
    return {
        member.member_id
        for member in group.members
        if member.member_type is member_type
        and member.match_role is EvidenceMatchRole.DISCRIMINATIVE
    }


def matched_group_ids(
    candidate: RankedEvidenceCandidate,
    groups: tuple[ResolvedGoldEvidenceGroup, ...],
    lineage_resolver: object | None = None,
) -> set[str]:
    if lineage_resolver is not None:
        from pmmc.evidence_lineage import EvidenceLineageResolver

        if not isinstance(lineage_resolver, EvidenceLineageResolver):
            raise TypeError("lineage_resolver must be an EvidenceLineageResolver")
        return lineage_resolver.matched_gold_group_ids(candidate, groups)
    matched: set[str] = set()
    for group in groups:
        if (
            candidate.memory_ids & _group_member_ids(group, EvidenceMemberType.MEMORY)
            or candidate.session_ids & _group_member_ids(group, EvidenceMemberType.SESSION)
            or candidate.image_ids & _group_member_ids(group, EvidenceMemberType.IMAGE)
            or candidate.evidence_ids & _group_member_ids(group, EvidenceMemberType.EVIDENCE)
        ):
            matched.add(group.gold_group_id)
    return matched


def matched_image_group_ids(
    candidate: RankedEvidenceCandidate,
    groups: tuple[ResolvedGoldEvidenceGroup, ...],
) -> set[str]:
    """Match only canonical image evidence, never sibling text/session members."""

    return {
        group.gold_group_id
        for group in groups
        if candidate.image_ids & _group_member_ids(group, EvidenceMemberType.IMAGE)
    }


def matched_text_group_ids(
    candidate: RankedEvidenceCandidate,
    groups: tuple[ResolvedGoldEvidenceGroup, ...],
) -> set[str]:
    """Match explicit text turns without treating captions or images as text hits."""

    return {
        group.gold_group_id
        for group in groups
        if candidate.text_memory_ids
        & _group_member_ids(group, EvidenceMemberType.MEMORY)
    }


def _metric(
    *,
    evaluation_run_id: str,
    answer_id: str,
    query_id: str,
    run_id: str,
    family: MetricFamily,
    name: str,
    version: str,
    value: float | None,
    k: int | None = None,
    unavailable_reason: str | None = None,
    created_at: datetime | None = None,
) -> MetricScore:
    return MetricScore(
        metric_score_id="metric-"
        + stable_hash([evaluation_run_id, answer_id, name, version])[:24],
        evaluation_run_id=evaluation_run_id,
        answer_id=answer_id,
        query_id=query_id,
        run_id=run_id,
        metric_family=family,
        metric_name=name,
        metric_version=version,
        k=k,
        scorer="pmmc.evaluation.evidence",
        applicable=value is not None,
        value=value,
        unavailable_reason=unavailable_reason if value is None else None,
        **({"created_at": created_at} if created_at is not None else {}),
    )


def score_evidence(
    *,
    evaluation_run_id: str,
    answer_id: str,
    query_id: str,
    run_id: str,
    groups: tuple[ResolvedGoldEvidenceGroup, ...],
    retrieval_candidates: list[RankedEvidenceCandidate],
    context_candidate: RankedEvidenceCandidate,
    context_source_kind: EvaluationContextSourceKind = (
        EvaluationContextSourceKind.BASELINE_CONTEXT
    ),
    context_assembly_id: str | None = None,
    program_execution_id: str | None = None,
    retrieval_trace_ids: list[str] | None = None,
    ks: tuple[int, ...] = (1, 5, 10, 20, 50),
    policy_version: str = "evidence-resolution-v1",
    created_at: datetime | None = None,
) -> EvidenceMetricBundle:
    """Score ranked retrieval and actual prompt coverage against resolved groups."""

    retrieval_candidates = sorted(retrieval_candidates, key=lambda item: item.rank)
    if [item.rank for item in retrieval_candidates] != list(
        range(1, len(retrieval_candidates) + 1)
    ):
        raise ValueError("retrieval candidates must have continuous 1-based ranks")
    if retrieval_trace_ids is not None and len(retrieval_trace_ids) != len(
        retrieval_candidates
    ):
        raise ValueError("retrieval trace IDs must align with ranked candidates")

    metrics: list[MetricScore] = []
    retrieval_evaluations: list[RetrievalEvaluation] = []
    relevant_ranks: list[int] = []
    for index, candidate in enumerate(retrieval_candidates):
        matched_ids = sorted(matched_group_ids(candidate, groups))
        if matched_ids:
            relevant_ranks.append(candidate.rank)
        trace_id = (
            retrieval_trace_ids[index]
            if retrieval_trace_ids is not None
            else candidate.candidate_id
        )
        retrieval_evaluations.append(
            RetrievalEvaluation(
                retrieval_evaluation_id="retrieval-eval-"
                + stable_hash([evaluation_run_id, trace_id])[:24],
                retrieval_trace_id=trace_id,
                query_id=query_id,
                run_id=run_id,
                evaluation_run_id=evaluation_run_id,
                is_gold_evidence=bool(matched_ids),
                matched_gold_evidence_ids=matched_ids,
                metadata={"policy_version": policy_version},
                **({"created_at": created_at} if created_at is not None else {}),
            )
        )

    if retrieval_candidates:
        image_groups = tuple(
            group
            for group in groups
            if _group_member_ids(group, EvidenceMemberType.IMAGE)
        )
        memory_groups = tuple(
            group
            for group in groups
            if _group_member_ids(group, EvidenceMemberType.MEMORY)
        )
        for k in ks:
            top = [item for item in retrieval_candidates if item.rank <= k]
            matched = (
                set().union(*(matched_group_ids(item, groups) for item in top))
                if top
                else set()
            )
            image_matched = (
                set().union(*(matched_image_group_ids(item, image_groups) for item in top))
                if top
                else set()
            )
            text_matched = (
                set().union(*(matched_text_group_ids(item, memory_groups) for item in top))
                if top
                else set()
            )
            reason = None if groups else "gold_evidence_unresolvable"
            recall = len(matched) / len(groups) if groups else None
            metrics.extend(
                [
                    _metric(
                        evaluation_run_id=evaluation_run_id,
                        answer_id=answer_id,
                        query_id=query_id,
                        run_id=run_id,
                        family=MetricFamily.RETRIEVAL,
                        name=f"evidence_recall_at_{k}",
                        version="evidence-recall-v1",
                        value=recall,
                        k=k,
                        unavailable_reason=reason,
                        created_at=created_at,
                    ),
                    _metric(
                        evaluation_run_id=evaluation_run_id,
                        answer_id=answer_id,
                        query_id=query_id,
                        run_id=run_id,
                        family=MetricFamily.RETRIEVAL,
                        name=f"evidence_hit_at_{k}",
                        version="evidence-hit-v1",
                        value=float(bool(matched)) if groups else None,
                        k=k,
                        unavailable_reason=reason,
                        created_at=created_at,
                    ),
                    _metric(
                        evaluation_run_id=evaluation_run_id,
                        answer_id=answer_id,
                        query_id=query_id,
                        run_id=run_id,
                        family=MetricFamily.RETRIEVAL,
                        name=f"multi_hop_complete_at_{k}",
                        version="multi-hop-complete-v1",
                        value=(float(len(matched) == len(groups)) if len(groups) > 1 else None),
                        k=k,
                        unavailable_reason=(
                            None if len(groups) > 1 else "query_has_fewer_than_two_gold_groups"
                        ),
                        created_at=created_at,
                    ),
                    _metric(
                        evaluation_run_id=evaluation_run_id,
                        answer_id=answer_id,
                        query_id=query_id,
                        run_id=run_id,
                        family=MetricFamily.RETRIEVAL,
                        name=f"image_hit_at_{k}",
                        version="image-hit-v1",
                        value=float(bool(image_matched)) if image_groups else None,
                        k=k,
                        unavailable_reason=(
                            None if image_groups else "gold_image_evidence_unavailable"
                        ),
                        created_at=created_at,
                    ),
                    _metric(
                        evaluation_run_id=evaluation_run_id,
                        answer_id=answer_id,
                        query_id=query_id,
                        run_id=run_id,
                        family=MetricFamily.RETRIEVAL,
                        name=f"text_hit_at_{k}",
                        version="text-hit-v1",
                        value=float(bool(text_matched)) if memory_groups else None,
                        k=k,
                        unavailable_reason=(
                            None if memory_groups else "gold_text_evidence_unavailable"
                        ),
                        created_at=created_at,
                    ),
                ]
            )
        metrics.append(
            _metric(
                evaluation_run_id=evaluation_run_id,
                answer_id=answer_id,
                query_id=query_id,
                run_id=run_id,
                family=MetricFamily.RETRIEVAL,
                name="mrr",
                version="evidence-mrr-v1",
                value=(
                    1.0 / min(relevant_ranks)
                    if groups and relevant_ranks
                    else (0.0 if groups else None)
                ),
                unavailable_reason=None if groups else "gold_evidence_unresolvable",
                created_at=created_at,
            )
        )

    context_matched = sorted(matched_group_ids(context_candidate, groups))
    context_evaluations = [
        ContextEvidenceEvaluation(
            context_evaluation_id="context-eval-"
            + stable_hash([evaluation_run_id, answer_id, "evidence"])[0:24],
            evaluation_run_id=evaluation_run_id,
            answer_id=answer_id,
            query_id=query_id,
            run_id=run_id,
            context_source_kind=context_source_kind,
            context_assembly_id=context_assembly_id,
            program_execution_id=program_execution_id,
            granularity=(groups[0].granularity if groups else EvidenceGranularity.GENERIC),
            applicable=bool(groups),
            matched_gold_group_ids=context_matched,
            gold_group_count=len(groups),
            policy_version=policy_version,
            unavailable_reason=None if groups else "gold_evidence_unresolvable",
            **({"created_at": created_at} if created_at is not None else {}),
        )
    ]
    metrics.append(
        _metric(
            evaluation_run_id=evaluation_run_id,
            answer_id=answer_id,
            query_id=query_id,
            run_id=run_id,
            family=MetricFamily.CONTEXT,
            name="context_evidence_recall",
            version="context-evidence-recall-v1",
            value=len(context_matched) / len(groups) if groups else None,
            unavailable_reason=None if groups else "gold_evidence_unresolvable",
            created_at=created_at,
        )
    )

    metrics.append(
        _metric(
            evaluation_run_id=evaluation_run_id,
            answer_id=answer_id,
            query_id=query_id,
            run_id=run_id,
            family=MetricFamily.CONTEXT,
            name="context_multi_hop_complete",
            version="context-multi-hop-complete-v1",
            value=(float(len(context_matched) == len(groups)) if len(groups) > 1 else None),
            unavailable_reason=(
                None if len(groups) > 1 else "query_has_fewer_than_two_gold_groups"
            ),
            created_at=created_at,
        )
    )

    image_groups = tuple(
        group
        for group in groups
        if _group_member_ids(group, EvidenceMemberType.IMAGE)
    )
    image_matched = matched_image_group_ids(context_candidate, image_groups)
    metrics.append(
        _metric(
            evaluation_run_id=evaluation_run_id,
            answer_id=answer_id,
            query_id=query_id,
            run_id=run_id,
            family=MetricFamily.CONTEXT,
            name="context_image_hit",
            version="context-image-hit-v1",
            value=float(bool(image_matched)) if image_groups else None,
            unavailable_reason=None if image_groups else "gold_image_evidence_unavailable",
            created_at=created_at,
        )
    )
    memory_groups = tuple(
        group
        for group in groups
        if _group_member_ids(group, EvidenceMemberType.MEMORY)
    )
    text_matched = matched_text_group_ids(context_candidate, memory_groups)
    metrics.append(
        _metric(
            evaluation_run_id=evaluation_run_id,
            answer_id=answer_id,
            query_id=query_id,
            run_id=run_id,
            family=MetricFamily.CONTEXT,
            name="context_text_hit",
            version="context-text-hit-v1",
            value=float(bool(text_matched)) if memory_groups else None,
            unavailable_reason=None if memory_groups else "gold_text_evidence_unavailable",
            created_at=created_at,
        )
    )
    return EvidenceMetricBundle(
        groups=groups,
        retrieval_evaluations=tuple(retrieval_evaluations),
        context_evaluations=tuple(context_evaluations),
        metrics=tuple(metrics),
    )
