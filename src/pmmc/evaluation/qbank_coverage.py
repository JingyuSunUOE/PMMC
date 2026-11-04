"""Evidence-grounded Questioner coverage metrics and failure labels."""

from __future__ import annotations

from dataclasses import dataclass

from pmmc.compiler.ids import stable_compiler_id
from pmmc.evaluation.evidence import RankedEvidenceCandidate, matched_group_ids
from pmmc.evaluation.qbank_protocols import HARDENED_QBANK_COVERAGE_PROTOCOLS
from pmmc.evidence_lineage import EvidenceLineageResolver
from pmmc.schemas import (
    QBankCoverageCandidate,
    QBankCoverageLabel,
    QBankCoverageLabelCategory,
    QBankCoverageMetric,
    ResolvedGoldEvidenceGroup,
)


@dataclass(frozen=True)
class QBankCandidateEvidence:
    qbank_id: str
    memory_ids: frozenset[str] = frozenset()
    session_ids: frozenset[str] = frozenset()
    image_ids: frozenset[str] = frozenset()
    evidence_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class QBankCoverageBundle:
    candidates: tuple[QBankCoverageCandidate, ...]
    metrics: tuple[QBankCoverageMetric, ...]
    labels: tuple[QBankCoverageLabel, ...]


def _metric(
    *,
    coverage_run_id: str,
    query_id: str,
    name: str,
    budget: int,
    k: int | None,
    value: float | None,
    unavailable_reason: str | None = None,
    metric_version: str = "qbank-coverage-v1",
) -> QBankCoverageMetric:
    return QBankCoverageMetric(
        metric_id=stable_compiler_id(
            "qbank-coverage-metric", [coverage_run_id, query_id, name, budget, k]
        ),
        coverage_run_id=coverage_run_id,
        query_id=query_id,
        metric_name=name,
        metric_version=metric_version,
        question_budget=budget,
        k=k,
        applicable=value is not None,
        value=value,
        unavailable_reason=unavailable_reason,
    )


def score_qbank_coverage(
    *,
    coverage_run_id: str,
    query_id: str,
    ranked_by_budget: dict[int, list[QBankCoverageCandidate]],
    evidence_by_qbank: dict[str, QBankCandidateEvidence],
    oracle_qbank_ids_by_budget: dict[int, list[str]] | None = None,
    gold_groups: tuple[ResolvedGoldEvidenceGroup, ...],
    match_ks: list[int],
    lineage_resolver: EvidenceLineageResolver | None = None,
    protocol_version: str = "qbank-coverage-v1",
) -> QBankCoverageBundle:
    hardened = protocol_version in HARDENED_QBANK_COVERAGE_PROTOCOLS
    metric_names = (
        (
            "qbank_evidence_recall_at_k",
            "qbank_evidence_proxy_cover_at_k",
            "qbank_evidence_proxy_match_at_k",
            "proxy_projected_fallback_at_k",
        )
        if hardened
        else (
            "qbank_evidence_recall_at_k",
            "qbank_cover_at_k",
            "qbank_match_at_k",
            "projected_fallback_at_k",
        )
    )
    if not gold_groups:
        unavailable_metrics = [
            _metric(
                coverage_run_id=coverage_run_id,
                query_id=query_id,
                name=name,
                budget=budget,
                k=k,
                value=None,
                unavailable_reason="gold_evidence_unresolvable",
                metric_version=protocol_version,
            )
            for budget in sorted(ranked_by_budget)
            for k in match_ks
            for name in metric_names
        ]
        unavailable_metrics.extend(
            _metric(
                coverage_run_id=coverage_run_id,
                query_id=query_id,
                name=name,
                budget=budget,
                k=None,
                value=None,
                unavailable_reason="gold_evidence_unresolvable",
                metric_version=protocol_version,
            )
            for budget in sorted(ranked_by_budget)
            for name in (
                (
                    "oracle_qbank_evidence_proxy_cover",
                    "oracle_qbank_evidence_proxy_match",
                )
                if hardened
                else ("oracle_qbank_cover", "oracle_qbank_match")
            )
        )
        label = QBankCoverageLabel(
            label_id=stable_compiler_id(
                "qbank-coverage-label", [coverage_run_id, query_id, "gold_unresolvable"]
            ),
            coverage_run_id=coverage_run_id,
            query_id=query_id,
            category=QBankCoverageLabelCategory.GOLD_UNRESOLVABLE,
        )
        candidates = tuple(
            item for budget in sorted(ranked_by_budget) for item in ranked_by_budget[budget]
        )
        return QBankCoverageBundle(
            candidates=candidates,
            metrics=tuple(unavailable_metrics),
            labels=(label,),
        )

    gold_ids = {item.gold_group_id for item in gold_groups}
    annotated: list[QBankCoverageCandidate] = []
    metrics: list[QBankCoverageMetric] = []
    labels: list[QBankCoverageLabel] = []
    for budget in sorted(ranked_by_budget):
        ranked = ranked_by_budget[budget]
        matched_by_candidate: dict[str, set[str]] = {}
        for candidate in ranked:
            evidence = evidence_by_qbank.get(
                candidate.qbank_id,
                QBankCandidateEvidence(candidate.qbank_id),
            )
            ranked_evidence = RankedEvidenceCandidate(
                candidate_id=candidate.qbank_id,
                rank=candidate.rank,
                memory_ids=evidence.memory_ids,
                session_ids=evidence.session_ids,
                image_ids=evidence.image_ids,
                evidence_ids=evidence.evidence_ids,
            )
            matched = matched_group_ids(
                ranked_evidence,
                gold_groups,
                lineage_resolver,
            )
            match_metadata = (
                lineage_resolver.match_details(ranked_evidence, gold_groups)
                if lineage_resolver is not None
                else {"matched_gold_group_ids": sorted(matched)}
            )
            matched_by_candidate[candidate.qbank_id] = matched
            annotated.append(
                candidate.model_copy(
                    update={
                        "matched_gold_group_ids": sorted(matched),
                        "individually_sufficient": matched == gold_ids,
                        "metadata": {
                            **candidate.metadata,
                            "evidence_match_kind": "deterministic_proxy",
                            **match_metadata,
                        },
                    }
                )
            )
        oracle_ids = (
            oracle_qbank_ids_by_budget.get(budget, [])
            if oracle_qbank_ids_by_budget is not None
            else [item.qbank_id for item in ranked]
        )
        oracle_matches: list[set[str]] = []
        for qbank_id in oracle_ids:
            evidence = evidence_by_qbank.get(qbank_id, QBankCandidateEvidence(qbank_id))
            oracle_matches.append(
                matched_group_ids(
                    RankedEvidenceCandidate(
                        candidate_id=qbank_id,
                        rank=1,
                        memory_ids=evidence.memory_ids,
                        session_ids=evidence.session_ids,
                        image_ids=evidence.image_ids,
                        evidence_ids=evidence.evidence_ids,
                    ),
                    gold_groups,
                    lineage_resolver,
                )
            )
        oracle_union = set().union(*oracle_matches) if oracle_matches else set()
        oracle_match = any(value == gold_ids for value in oracle_matches)
        metrics.extend(
            (
                _metric(
                    coverage_run_id=coverage_run_id,
                    query_id=query_id,
                    name=(
                        "oracle_qbank_evidence_proxy_cover"
                        if hardened
                        else "oracle_qbank_cover"
                    ),
                    budget=budget,
                    k=None,
                    value=float(oracle_union == gold_ids),
                    metric_version=protocol_version,
                ),
                _metric(
                    coverage_run_id=coverage_run_id,
                    query_id=query_id,
                    name=(
                        "oracle_qbank_evidence_proxy_match"
                        if hardened
                        else "oracle_qbank_match"
                    ),
                    budget=budget,
                    k=None,
                    value=float(oracle_match),
                    metric_version=protocol_version,
                ),
            )
        )
        for k in match_ks:
            top = ranked[:k]
            union = (
                set().union(*(matched_by_candidate[item.qbank_id] for item in top))
                if top
                else set()
            )
            match = any(matched_by_candidate[item.qbank_id] == gold_ids for item in top)
            values = dict(
                zip(
                    metric_names,
                    (
                        len(union) / len(gold_ids),
                        float(union == gold_ids),
                        float(match),
                        float(not match),
                    ),
                    strict=True,
                )
            )
            metrics.extend(
                _metric(
                    coverage_run_id=coverage_run_id,
                    query_id=query_id,
                    name=name,
                    budget=budget,
                    k=k,
                    value=value,
                    metric_version=protocol_version,
                )
                for name, value in values.items()
            )
            if k == max(match_ks) and budget == max(ranked_by_budget):
                category = None
                if not oracle_match:
                    category = QBankCoverageLabelCategory.QUESTIONER_COVERAGE_FAILURE
                elif not match:
                    category = QBankCoverageLabelCategory.MATCHING_FAILURE
                if category is not None:
                    labels.append(
                        QBankCoverageLabel(
                            label_id=stable_compiler_id(
                                "qbank-coverage-label",
                                [coverage_run_id, query_id, category.value, budget, k],
                            ),
                            coverage_run_id=coverage_run_id,
                            query_id=query_id,
                            category=category,
                            question_budget=budget,
                            k=k,
                        )
                    )
    return QBankCoverageBundle(
        candidates=tuple(annotated),
        metrics=tuple(metrics),
        labels=tuple(labels),
    )
