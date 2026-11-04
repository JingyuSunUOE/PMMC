"""Lossless mappings for Questioner and QBank coverage persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pmmc.schemas import (
    CompilationUnitLink,
    QBankCandidateFamily,
    QBankCandidateFamilyMember,
    QBankCoverageCandidate,
    QBankCoverageGateReport,
    QBankCoverageLabel,
    QBankCoverageMetric,
    QBankCoverageQuery,
    QBankCoverageQueryEmbedding,
    QBankCoverageRunManifest,
    QBankDuplicateLink,
    QBankIndexManifest,
    QuestionCandidateGateResult,
    QuestionerAnswerClaim,
    QuestionerCandidateSupersession,
    QuestionerCandidateVerification,
    QuestionerGeneration,
    QuestionerVerificationBatch,
    QuestionOpportunity,
    QuestionOpportunityEvidenceRef,
    QuestionOpportunitySet,
    ResolvedGoldEvidenceGroup,
)

Row = dict[str, Any]


def _metadata(value: Any) -> dict[str, Any]:
    return dict(value.metadata)


def compiler_unit_link_to_row(value: CompilationUnitLink) -> Row:
    row = value.model_dump(mode="python")
    row["relation_type"] = value.relation_type.value
    row["metadata"] = _metadata(value)
    return row


def compiler_unit_link_from_row(row: Mapping[str, Any]) -> CompilationUnitLink:
    return CompilationUnitLink.model_validate(dict(row))


def question_opportunity_set_to_row(value: QuestionOpportunitySet) -> Row:
    row = value.model_dump(mode="python")
    row["metadata"] = _metadata(value)
    return row


def question_opportunity_set_from_row(
    row: Mapping[str, Any],
) -> QuestionOpportunitySet:
    return QuestionOpportunitySet.model_validate(dict(row))


def question_opportunity_to_row(value: QuestionOpportunity) -> Row:
    row = value.model_dump(mode="python")
    row["opportunity_type"] = value.opportunity_type.value
    row["allowed_question_types"] = [item.value for item in value.allowed_question_types]
    row["allowed_reasoning_types"] = [item.value for item in value.allowed_reasoning_types]
    row["required_modalities"] = [item.value for item in value.required_modalities]
    row["metadata"] = _metadata(value)
    return row


def question_opportunity_from_row(row: Mapping[str, Any]) -> QuestionOpportunity:
    return QuestionOpportunity.model_validate(dict(row))


def question_opportunity_evidence_to_row(
    value: QuestionOpportunityEvidenceRef,
) -> Row:
    row = value.model_dump(mode="python", exclude={"evidence"})
    row.update(
        value.evidence.model_dump(
            mode="python",
            exclude={"metadata", "score"},
        )
    )
    row["evidence_type"] = value.evidence.evidence_type.value
    row["metadata"] = {
        **_metadata(value),
        "evidence_metadata": dict(value.evidence.metadata),
        "evidence_score": (
            value.evidence.score.model_dump(mode="json")
            if value.evidence.score is not None
            else None
        ),
    }
    return row


def question_opportunity_evidence_from_row(
    row: Mapping[str, Any],
) -> QuestionOpportunityEvidenceRef:
    data = dict(row)
    metadata = dict(data.get("metadata") or {})
    evidence = {
        name: data.pop(name)
        for name in (
            "evidence_id",
            "evidence_type",
            "memory_id",
            "image_id",
            "session_id",
            "caption_id",
        )
    }
    evidence["metadata"] = dict(metadata.pop("evidence_metadata", {}))
    evidence["score"] = metadata.pop("evidence_score", None)
    data["metadata"] = metadata
    data["evidence"] = evidence
    return QuestionOpportunityEvidenceRef.model_validate(data)


def questioner_generation_to_row(value: QuestionerGeneration) -> Row:
    row = value.model_dump(mode="python")
    row["parse_status"] = value.parse_status.value
    row["metadata"] = _metadata(value)
    return row


def questioner_generation_from_row(row: Mapping[str, Any]) -> QuestionerGeneration:
    return QuestionerGeneration.model_validate(dict(row))


def questioner_answer_claim_to_row(value: QuestionerAnswerClaim) -> Row:
    row = value.model_dump(mode="python")
    row["claim_kind"] = value.claim_kind.value
    row["metadata"] = _metadata(value)
    return row


def questioner_answer_claim_from_row(
    row: Mapping[str, Any],
) -> QuestionerAnswerClaim:
    return QuestionerAnswerClaim.model_validate(dict(row))


def qbank_candidate_family_to_row(value: QBankCandidateFamily) -> Row:
    row = value.model_dump(mode="python")
    row["metadata"] = _metadata(value)
    return row


def qbank_candidate_family_from_row(
    row: Mapping[str, Any],
) -> QBankCandidateFamily:
    return QBankCandidateFamily.model_validate(dict(row))


def qbank_candidate_family_member_to_row(
    value: QBankCandidateFamilyMember,
) -> Row:
    row = value.model_dump(mode="python")
    row["metadata"] = _metadata(value)
    return row


def qbank_candidate_family_member_from_row(
    row: Mapping[str, Any],
) -> QBankCandidateFamilyMember:
    return QBankCandidateFamilyMember.model_validate(dict(row))


def questioner_verification_batch_to_row(value: QuestionerVerificationBatch) -> Row:
    row = value.model_dump(mode="python")
    row["metadata"] = _metadata(value)
    return row


def questioner_verification_batch_from_row(
    row: Mapping[str, Any],
) -> QuestionerVerificationBatch:
    return QuestionerVerificationBatch.model_validate(dict(row))


def questioner_candidate_verification_to_row(
    value: QuestionerCandidateVerification,
) -> Row:
    row = value.model_dump(mode="python")
    row["verdict"] = value.verdict.value
    row["reason_codes"] = [item.value for item in value.reason_codes]
    row["metadata"] = _metadata(value)
    return row


def questioner_candidate_verification_from_row(
    row: Mapping[str, Any],
) -> QuestionerCandidateVerification:
    return QuestionerCandidateVerification.model_validate(dict(row))


def questioner_candidate_supersession_to_row(
    value: QuestionerCandidateSupersession,
) -> Row:
    row = value.model_dump(mode="python")
    row["metadata"] = _metadata(value)
    return row


def questioner_candidate_supersession_from_row(
    row: Mapping[str, Any],
) -> QuestionerCandidateSupersession:
    return QuestionerCandidateSupersession.model_validate(dict(row))


def questioner_gate_to_row(value: QuestionCandidateGateResult) -> Row:
    row = value.model_dump(mode="python")
    row["status"] = value.status.value
    row["rejection_codes"] = [item.value for item in value.rejection_codes]
    row["metadata"] = _metadata(value)
    return row


def questioner_gate_from_row(row: Mapping[str, Any]) -> QuestionCandidateGateResult:
    return QuestionCandidateGateResult.model_validate(dict(row))


def qbank_duplicate_to_row(value: QBankDuplicateLink) -> Row:
    row = value.model_dump(mode="python")
    row["score"] = value.score.model_dump(mode="json") if value.score is not None else None
    row["metadata"] = _metadata(value)
    return row


def qbank_duplicate_from_row(row: Mapping[str, Any]) -> QBankDuplicateLink:
    return QBankDuplicateLink.model_validate(dict(row))


def qbank_index_to_row(value: QBankIndexManifest) -> Row:
    row = value.model_dump(mode="python")
    row["status"] = value.status.value
    row["metadata"] = _metadata(value)
    return row


def qbank_index_from_row(row: Mapping[str, Any]) -> QBankIndexManifest:
    return QBankIndexManifest.model_validate(dict(row))


def coverage_run_to_row(value: QBankCoverageRunManifest) -> Row:
    row = value.model_dump(mode="python")
    row["gate_mode"] = value.gate_mode.value
    row["status"] = value.status.value
    row["metadata"] = _metadata(value)
    return row


def coverage_run_from_row(row: Mapping[str, Any]) -> QBankCoverageRunManifest:
    return QBankCoverageRunManifest.model_validate(dict(row))


def coverage_query_to_row(value: QBankCoverageQuery) -> Row:
    row = value.model_dump(mode="python")
    row["metadata"] = _metadata(value)
    return row


def coverage_query_from_row(row: Mapping[str, Any]) -> QBankCoverageQuery:
    return QBankCoverageQuery.model_validate(dict(row))


def coverage_embedding_to_row(value: QBankCoverageQueryEmbedding) -> Row:
    row = value.model_dump(mode="python", exclude={"vector"})
    row["channel"] = value.channel.value
    row["embedding"] = value.vector
    row["metadata"] = _metadata(value)
    return row


def coverage_embedding_from_row(row: Mapping[str, Any]) -> QBankCoverageQueryEmbedding:
    data = dict(row)
    vector = data.pop("embedding")
    data["vector"] = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    return QBankCoverageQueryEmbedding.model_validate(data)


def coverage_candidate_to_row(value: QBankCoverageCandidate) -> Row:
    row = value.model_dump(mode="python")
    for field_name in (
        "lexical_score",
        "text_semantic_score",
        "multimodal_semantic_score",
        "final_score",
    ):
        score = getattr(value, field_name)
        row[field_name] = score.model_dump(mode="json") if score is not None else None
    row["metadata"] = _metadata(value)
    return row


def coverage_candidate_from_row(row: Mapping[str, Any]) -> QBankCoverageCandidate:
    return QBankCoverageCandidate.model_validate(dict(row))


def coverage_gold_to_rows(
    value: ResolvedGoldEvidenceGroup,
    *,
    coverage_run_id: str,
) -> dict[str, list[Row]]:
    group = value.model_dump(mode="python", exclude={"members", "evaluation_run_id"})
    group["coverage_run_id"] = coverage_run_id
    group["granularity"] = value.granularity.value
    group["metadata"] = _metadata(value)
    members = []
    for member in value.members:
        members.append(
            {
                "gold_group_id": value.gold_group_id,
                "member_type": member.member_type.value,
                "member_id": member.member_id,
                "derived": member.derived,
                "match_role": member.match_role.value,
                "metadata": dict(member.metadata),
            }
        )
    return {"group": [group], "members": members}


def coverage_gold_from_rows(
    row: Mapping[str, Any],
    member_rows: Sequence[Mapping[str, Any]],
) -> ResolvedGoldEvidenceGroup:
    data = dict(row)
    data["evaluation_run_id"] = data.pop("coverage_run_id")
    members: list[Row] = []
    for member in sorted(
        member_rows,
        key=lambda item: (str(item["member_type"]), str(item["member_id"])),
    ):
        member_data = dict(member)
        member_data.pop("gold_group_id", None)
        members.append(member_data)
    data["members"] = members
    return ResolvedGoldEvidenceGroup.model_validate(data)


def coverage_metric_to_row(value: QBankCoverageMetric) -> Row:
    row = value.model_dump(mode="python")
    row["metadata"] = _metadata(value)
    return row


def coverage_metric_from_row(row: Mapping[str, Any]) -> QBankCoverageMetric:
    return QBankCoverageMetric.model_validate(dict(row))


def coverage_label_to_row(value: QBankCoverageLabel) -> Row:
    row = value.model_dump(mode="python")
    row["category"] = value.category.value
    row["metadata"] = _metadata(value)
    return row


def coverage_label_from_row(row: Mapping[str, Any]) -> QBankCoverageLabel:
    return QBankCoverageLabel.model_validate(dict(row))


def coverage_report_to_row(value: QBankCoverageGateReport) -> Row:
    row = value.model_dump(mode="python")
    row["mode"] = value.mode.value
    row["metadata"] = _metadata(value)
    return row


def coverage_report_from_row(row: Mapping[str, Any]) -> QBankCoverageGateReport:
    return QBankCoverageGateReport.model_validate(dict(row))
