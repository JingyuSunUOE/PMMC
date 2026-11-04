"""Async repositories for Questioner facts and evaluator-only QBank coverage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from sqlalchemy import func, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from pmmc.compiler.ids import canonical_content_hash, stable_compiler_id
from pmmc.compiler.questioner import GatedQuestionCandidate
from pmmc.infrastructure.database.compiler_models import (
    CompilerRunModel,
    CompilerUnitMemoryModel,
    CompilerUnitModel,
    CompilerUnitStageResultModel,
    QBankEvidenceRefModel,
    QBankQuestionModel,
)
from pmmc.infrastructure.database.compiler_repositories import (
    PostgresQBankCompilerRepository,
)
from pmmc.infrastructure.database.errors import (
    InvalidStateTransitionError,
    PersistenceConflictError,
)
from pmmc.infrastructure.database.questioner_mappers import (
    compiler_unit_link_from_row,
    compiler_unit_link_to_row,
    coverage_candidate_from_row,
    coverage_candidate_to_row,
    coverage_embedding_from_row,
    coverage_embedding_to_row,
    coverage_gold_from_rows,
    coverage_gold_to_rows,
    coverage_label_to_row,
    coverage_metric_from_row,
    coverage_metric_to_row,
    coverage_query_from_row,
    coverage_query_to_row,
    coverage_report_from_row,
    coverage_report_to_row,
    coverage_run_from_row,
    coverage_run_to_row,
    qbank_candidate_family_from_row,
    qbank_candidate_family_member_from_row,
    qbank_candidate_family_member_to_row,
    qbank_candidate_family_to_row,
    qbank_duplicate_to_row,
    qbank_index_from_row,
    qbank_index_to_row,
    question_opportunity_evidence_from_row,
    question_opportunity_evidence_to_row,
    question_opportunity_from_row,
    question_opportunity_set_from_row,
    question_opportunity_set_to_row,
    question_opportunity_to_row,
    questioner_answer_claim_from_row,
    questioner_answer_claim_to_row,
    questioner_candidate_supersession_from_row,
    questioner_candidate_supersession_to_row,
    questioner_candidate_verification_from_row,
    questioner_candidate_verification_to_row,
    questioner_gate_from_row,
    questioner_gate_to_row,
    questioner_generation_from_row,
    questioner_generation_to_row,
    questioner_verification_batch_from_row,
    questioner_verification_batch_to_row,
)
from pmmc.infrastructure.database.questioner_models import (
    CompilerUnitLinkModel,
    QBankCandidateFamilyMemberModel,
    QBankCandidateFamilyModel,
    QBankCoverageCandidateModel,
    QBankCoverageGateReportModel,
    QBankCoverageGoldGroupMemberModel,
    QBankCoverageGoldGroupModel,
    QBankCoverageLabelModel,
    QBankCoverageMatchModel,
    QBankCoverageMetricModel,
    QBankCoverageQueryEmbeddingModel,
    QBankCoverageQueryModel,
    QBankCoverageRunModel,
    QBankDuplicateLinkModel,
    QBankIndexManifestModel,
    QuestionerAnswerClaimModel,
    QuestionerCandidateGateResultModel,
    QuestionerCandidateSupersessionModel,
    QuestionerCandidateVerificationModel,
    QuestionerGenerationModel,
    QuestionerVerificationBatchModel,
    QuestionOpportunityEvidenceRefModel,
    QuestionOpportunityModel,
    QuestionOpportunitySetModel,
    RuntimeStructuredOutputItemDiagnosticModel,
)
from pmmc.schemas import (
    CandidateGateStatus,
    EmbeddingCollectionStatus,
    OperatorEvidence,
    QBankCandidateFamily,
    QBankCandidateFamilyMember,
    QBankCoverageCandidate,
    QBankCoverageGateReport,
    QBankCoverageLabel,
    QBankCoverageMetric,
    QBankCoverageQuery,
    QBankCoverageQueryEmbedding,
    QBankCoverageRunManifest,
    QBankCoverageStatus,
    QBankDuplicateLink,
    QBankIndexManifest,
    QuestionCandidate,
    QuestionCandidateGateResult,
    QuestionerAnswerClaim,
    QuestionerCandidateSupersession,
    QuestionerCandidateVerification,
    QuestionerGeneration,
    QuestionerVerificationBatch,
    QuestionOpportunity,
    QuestionOpportunityEvidenceRef,
    QuestionOpportunitySet,
    RankingScore,
    ResolvedGoldEvidenceGroup,
)
from pmmc.schemas.base import utc_now
from pmmc.schemas.questioner import QUESTIONER_STATIC_REJECTION_CODES

ModelT = TypeVar("ModelT", bound=DeclarativeBase)


def _candidate_family_dedupe_version(
    *, compiler_schema_version: str, questioner_prompt_version: str | None
) -> str:
    expected_by_identity = {
        ("0.6.0", "pmmc-questioner-v3.3"): "qbank-family-dedupe-v2",
        ("0.7.0", "pmmc-questioner-v3.4"): "qbank-family-dedupe-v3",
        ("0.8.0", "pmmc-questioner-v3.5"): "qbank-family-dedupe-v4",
        ("0.9.0", "pmmc-questioner-v3.6"): "qbank-family-dedupe-v5",
    }
    expected = expected_by_identity.get((compiler_schema_version, questioner_prompt_version or ""))
    if expected is None:
        raise ValueError("compiler candidate-family protocol identity is inconsistent")
    return expected


_V3_STATIC_REJECTION_CODES = QUESTIONER_STATIC_REJECTION_CODES


async def questioner_v3_unit_issues(
    session: AsyncSession,
    *,
    run_id: str,
    compiler_unit_id: str,
) -> list[str]:
    """Return complete v3 opportunity, verifier, revision, and terminal issues."""

    issues: list[str] = []
    sets = list(
        await session.scalars(
            select(QuestionOpportunitySetModel).where(
                QuestionOpportunitySetModel.run_id == run_id,
                QuestionOpportunitySetModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    if len(sets) != 1:
        return ["unit requires exactly one Questioner opportunity set"]
    opportunity_set = sets[0]
    opportunities = list(
        await session.scalars(
            select(QuestionOpportunityModel)
            .where(
                QuestionOpportunityModel.run_id == run_id,
                QuestionOpportunityModel.opportunity_set_id == opportunity_set.opportunity_set_id,
            )
            .order_by(QuestionOpportunityModel.ordinal)
        )
    )
    if len(opportunities) != opportunity_set.opportunity_count or [
        item.ordinal for item in opportunities
    ] != list(range(len(opportunities))):
        issues.append("opportunity-set count or ordinal lineage is incomplete")
    opportunity_ids = {item.opportunity_id for item in opportunities}
    link_ids = set(
        await session.scalars(
            select(CompilerUnitLinkModel.link_id).where(
                CompilerUnitLinkModel.run_id == run_id,
                CompilerUnitLinkModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    if any(not set(item.relation_link_ids) <= link_ids for item in opportunities):
        issues.append("opportunity relation lineage crosses its compilation unit")
    refs = list(
        await session.scalars(
            select(QuestionOpportunityEvidenceRefModel)
            .where(
                QuestionOpportunityEvidenceRefModel.run_id == run_id,
                QuestionOpportunityEvidenceRefModel.compiler_unit_id == compiler_unit_id,
            )
            .order_by(
                QuestionOpportunityEvidenceRefModel.opportunity_id,
                QuestionOpportunityEvidenceRefModel.ordinal,
            )
        )
    )
    aliases_by_opportunity: dict[str, list[str]] = {}
    for ref in refs:
        if ref.opportunity_id not in opportunity_ids:
            issues.append("opportunity evidence references a missing opportunity")
        if (
            (ref.evidence_type in {"memory", "caption"} and ref.evidence_id != ref.memory_id)
            or (ref.evidence_type == "image" and ref.evidence_id != ref.image_id)
            or (ref.evidence_type == "session" and ref.evidence_id != ref.session_id)
        ):
            issues.append("opportunity evidence ID differs from typed lineage")
        aliases_by_opportunity.setdefault(ref.opportunity_id, []).append(ref.alias)
    for opportunity in opportunities:
        if aliases_by_opportunity.get(opportunity.opportunity_id, []) != list(
            opportunity.evidence_aliases
        ):
            issues.append("opportunity evidence aliases are incomplete or unordered")

    generations = list(
        await session.scalars(
            select(QuestionerGenerationModel).where(
                QuestionerGenerationModel.run_id == run_id,
                QuestionerGenerationModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    successful_generations = [item for item in generations if item.parse_status != "failed"]
    if len(successful_generations) != 1:
        compiler_version = await session.scalar(
            select(CompilerRunModel.compiler_schema_version).where(
                CompilerRunModel.run_id == run_id
            )
        )
        terminal_generations = [
            item
            for item in generations
            if item.parse_status == "failed"
            and isinstance(item.metadata_json, dict)
            and item.metadata_json.get("terminal_reason") == "generation_structured_output_invalid"
        ]
        if (
            compiler_version in {"0.6.0", "0.7.0", "0.8.0", "0.9.0"}
            and len(generations) == 1
            and len(terminal_generations) == 1
        ):
            terminal = terminal_generations[0]
            if (
                terminal.raw_candidate_count != 0
                or terminal.valid_candidate_count != 0
                or terminal.invalid_candidate_count != 0
                or terminal.repair_attempt_count != 0
                or terminal.metadata_json.get("native_structured_output") is not False
                or terminal.metadata_json.get("response_artifact_ready") is not True
            ):
                issues.append("terminal generation rejection contract is invalid")
            downstream_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(QBankQuestionModel)
                    .where(
                        QBankQuestionModel.run_id == run_id,
                        QBankQuestionModel.compiler_unit_id == compiler_unit_id,
                    )
                )
                or 0
            )
            if downstream_count:
                issues.append("terminal generation rejection produced QBank candidates")
            diagnostics = list(
                await session.scalars(
                    select(RuntimeStructuredOutputItemDiagnosticModel).where(
                        RuntimeStructuredOutputItemDiagnosticModel.run_id == run_id,
                        RuntimeStructuredOutputItemDiagnosticModel.compiler_unit_id
                        == compiler_unit_id,
                        RuntimeStructuredOutputItemDiagnosticModel.model_call_id
                        == terminal.model_call_id,
                        RuntimeStructuredOutputItemDiagnosticModel.stage == "questioner",
                    )
                )
            )
            if len(diagnostics) != 1 or diagnostics[0].status != "conservative_reject":
                issues.append("terminal generation rejection diagnostic is incomplete")
            return issues
        return [*issues, "unit requires exactly one terminal v3 generation"]
    generation = successful_generations[0]
    questions = list(
        await session.scalars(
            select(QBankQuestionModel)
            .where(
                QBankQuestionModel.run_id == run_id,
                QBankQuestionModel.compiler_unit_id == compiler_unit_id,
            )
            .order_by(
                QBankQuestionModel.generation_rank,
                QBankQuestionModel.candidate_version,
            )
        )
    )
    generated = [item for item in questions if item.candidate_version == 0]
    if len(generated) != generation.valid_candidate_count:
        issues.append("version-0 candidate count differs from successful resolution count")
    question_by_id = {item.qbank_id: item for item in questions}
    if len(question_by_id) != len(questions):
        issues.append("QBank candidate IDs are duplicated")
    if any(
        item.generation_id != generation.generation_id or item.opportunity_id not in opportunity_ids
        for item in questions
    ):
        issues.append("candidate generation or opportunity lineage is inconsistent")

    gates = list(
        await session.scalars(
            select(QuestionerCandidateGateResultModel).where(
                QuestionerCandidateGateResultModel.run_id == run_id,
                QuestionerCandidateGateResultModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    gate_by_qbank = {item.qbank_id: item for item in gates}
    if set(gate_by_qbank) != set(question_by_id):
        issues.append("every v3 candidate must have exactly one terminal gate")

    batches = list(
        await session.scalars(
            select(QuestionerVerificationBatchModel).where(
                QuestionerVerificationBatchModel.run_id == run_id,
                QuestionerVerificationBatchModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    batch_by_id = {item.verification_batch_id: item for item in batches}
    verifications = list(
        await session.scalars(
            select(QuestionerCandidateVerificationModel).where(
                QuestionerCandidateVerificationModel.run_id == run_id,
                QuestionerCandidateVerificationModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    verification_by_key = {(item.qbank_id, item.verification_round): item for item in verifications}
    if len(verification_by_key) != len(verifications):
        issues.append("candidate verifier outcomes are duplicated")
    for batch in batches:
        values = [
            item
            for item in verifications
            if item.verification_batch_id == batch.verification_batch_id
        ]
        counts = {
            verdict: sum(item.verdict == verdict for item in values)
            for verdict in ("pass", "revise", "reject")
        }
        if (
            len(values) != batch.input_count
            or counts["pass"] != batch.pass_count
            or counts["revise"] != batch.revise_count
            or counts["reject"] != batch.reject_count
        ):
            issues.append("verification-batch accounting is incomplete")
    if any(item.verification_batch_id not in batch_by_id for item in verifications):
        issues.append("candidate verification references a missing batch")

    supersessions = list(
        await session.scalars(
            select(QuestionerCandidateSupersessionModel).where(
                QuestionerCandidateSupersessionModel.run_id == run_id,
                QuestionerCandidateSupersessionModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    supersession_by_source = {item.source_qbank_id: item for item in supersessions}
    supersession_by_replacement = {item.replacement_qbank_id: item for item in supersessions}
    if len(supersession_by_source) != len(supersessions) or len(supersession_by_replacement) != len(
        supersessions
    ):
        issues.append("candidate supersession must be one-to-one")

    by_rank: dict[int, list[QBankQuestionModel]] = {}
    for question in questions:
        by_rank.setdefault(question.generation_rank, []).append(question)
    for rank, rank_questions in by_rank.items():
        sources = [item for item in rank_questions if item.candidate_version == 0]
        replacements = [item for item in rank_questions if item.candidate_version == 1]
        if len(sources) != 1 or len(replacements) > 1:
            issues.append(f"generation rank {rank} has invalid candidate versions")
            continue
        source = sources[0]
        initial = verification_by_key.get((source.qbank_id, 0))
        source_gate = gate_by_qbank.get(source.qbank_id)
        if initial is None and (
            source_gate is None
            or source_gate.status != CandidateGateStatus.REJECTED.value
            or not (_V3_STATIC_REJECTION_CODES & set(source_gate.rejection_codes))
        ):
            issues.append(f"generation rank {rank} lacks initial semantic verification")
        if replacements:
            replacement = replacements[0]
            supersession = supersession_by_replacement.get(replacement.qbank_id)
            if (
                supersession is None
                or supersession.source_qbank_id != source.qbank_id
                or replacement.supersedes_qbank_id != source.qbank_id
                or initial is None
                or initial.verdict != "revise"
                or initial.replacement_qbank_id != replacement.qbank_id
            ):
                issues.append(f"generation rank {rank} has invalid revision lineage")
        if (
            sum(
                gate_by_qbank.get(item.qbank_id) is not None
                and gate_by_qbank[item.qbank_id].status
                == CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value
                for item in rank_questions
            )
            > 1
        ):
            issues.append(f"generation rank {rank} has multiple terminal candidates")

    revised_ids = {item.qbank_id for item in questions if item.candidate_version == 1}
    if set(supersession_by_replacement) != revised_ids:
        issues.append("revised candidates and supersession facts do not reconcile")
    for qbank_id, gate in gate_by_qbank.items():
        if gate.status != CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value:
            continue
        question = question_by_id[qbank_id]
        verification = verification_by_key.get((qbank_id, question.candidate_version))
        if verification is None or verification.verdict != "pass":
            issues.append("eligible v3 candidate lacks a terminal verifier pass")
    return issues


def _model_mapping(value: DeclarativeBase) -> dict[str, Any]:
    return {
        prop.columns[0].name: getattr(value, prop.key) for prop in value.__mapper__.column_attrs
    }


def _comparable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    return value


async def _insert_idempotent(
    session: AsyncSession,
    model: type[ModelT],
    row: dict[str, Any],
    *identity: Any,
) -> ModelT:
    key = identity[0] if len(identity) == 1 else identity
    existing = await session.get(model, key)
    if existing is not None:
        actual = _model_mapping(existing)
        if any(_comparable(actual[name]) != _comparable(value) for name, value in row.items()):
            raise PersistenceConflictError(
                f"{model.__tablename__} stable ID already exists with different content"
            )
        return existing
    try:
        async with session.begin_nested():
            await session.execute(insert(cast(Any, model.__table__)).values(row))
    except IntegrityError:
        existing = await session.get(model, key)
        if existing is None:
            raise
        actual = _model_mapping(existing)
        if any(_comparable(actual[name]) != _comparable(value) for name, value in row.items()):
            raise PersistenceConflictError(
                f"{model.__tablename__} stable ID already exists with different content"
            ) from None
        return existing
    inserted = await session.get(model, key)
    if inserted is None:
        raise RuntimeError(f"failed to read inserted {model.__tablename__}")
    return inserted


class PostgresQuestionerRepository:
    """Persist one unit's generation and candidate gate as short transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.questions = PostgresQBankCompilerRepository(session)

    async def add_links(self, values: list[Any]) -> None:
        for value in values:
            row = compiler_unit_link_to_row(value)
            await _insert_idempotent(
                self.session,
                CompilerUnitLinkModel,
                row,
                value.link_id,
            )

    async def list_links(self, compiler_unit_id: str) -> list[Any]:
        rows = list(
            (
                await self.session.execute(
                    select(CompilerUnitLinkModel)
                    .where(CompilerUnitLinkModel.compiler_unit_id == compiler_unit_id)
                    .order_by(CompilerUnitLinkModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return [compiler_unit_link_from_row(_model_mapping(item)) for item in rows]

    async def add_opportunity_set(
        self,
        value: QuestionOpportunitySet,
        opportunities: list[QuestionOpportunity],
        evidence_refs: list[QuestionOpportunityEvidenceRef],
    ) -> QuestionOpportunitySet:
        if value.opportunity_count != len(opportunities):
            raise ValueError("opportunity-set count does not match opportunity facts")
        if any(item.opportunity_set_id != value.opportunity_set_id for item in opportunities):
            raise ValueError("opportunity belongs to a different opportunity set")
        opportunity_ids = {item.opportunity_id for item in opportunities}
        if any(item.opportunity_id not in opportunity_ids for item in evidence_refs):
            raise ValueError("opportunity evidence references a different set")
        stored = await _insert_idempotent(
            self.session,
            QuestionOpportunitySetModel,
            question_opportunity_set_to_row(value),
            value.opportunity_set_id,
        )
        for opportunity in opportunities:
            await _insert_idempotent(
                self.session,
                QuestionOpportunityModel,
                question_opportunity_to_row(opportunity),
                opportunity.opportunity_id,
            )
        for evidence_ref in evidence_refs:
            await _insert_idempotent(
                self.session,
                QuestionOpportunityEvidenceRefModel,
                question_opportunity_evidence_to_row(evidence_ref),
                evidence_ref.opportunity_evidence_ref_id,
            )
        return question_opportunity_set_from_row(_model_mapping(stored))

    async def get_opportunity_set(
        self, compiler_unit_id: str
    ) -> (
        tuple[
            QuestionOpportunitySet,
            list[QuestionOpportunity],
            list[QuestionOpportunityEvidenceRef],
        ]
        | None
    ):
        stored = await self.session.scalar(
            select(QuestionOpportunitySetModel).where(
                QuestionOpportunitySetModel.compiler_unit_id == compiler_unit_id
            )
        )
        if stored is None:
            return None
        opportunities = list(
            (
                await self.session.execute(
                    select(QuestionOpportunityModel)
                    .where(QuestionOpportunityModel.opportunity_set_id == stored.opportunity_set_id)
                    .order_by(QuestionOpportunityModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        opportunity_ids = [item.opportunity_id for item in opportunities]
        evidence_refs = list(
            (
                await self.session.execute(
                    select(QuestionOpportunityEvidenceRefModel)
                    .where(QuestionOpportunityEvidenceRefModel.opportunity_id.in_(opportunity_ids))
                    .order_by(
                        QuestionOpportunityEvidenceRefModel.opportunity_id,
                        QuestionOpportunityEvidenceRefModel.ordinal,
                    )
                )
            )
            .scalars()
            .all()
        )
        return (
            question_opportunity_set_from_row(_model_mapping(stored)),
            [question_opportunity_from_row(_model_mapping(item)) for item in opportunities],
            [
                question_opportunity_evidence_from_row(_model_mapping(item))
                for item in evidence_refs
            ],
        )

    async def add_generation(self, value: QuestionerGeneration) -> QuestionerGeneration:
        stored = await _insert_idempotent(
            self.session,
            QuestionerGenerationModel,
            questioner_generation_to_row(value),
            value.generation_id,
        )
        return questioner_generation_from_row(_model_mapping(stored))

    async def add_answer_claim(
        self,
        value: QuestionerAnswerClaim,
    ) -> QuestionerAnswerClaim:
        stored = await _insert_idempotent(
            self.session,
            QuestionerAnswerClaimModel,
            questioner_answer_claim_to_row(value),
            value.answer_claim_id,
        )
        return questioner_answer_claim_from_row(_model_mapping(stored))

    async def list_answer_claims(self, qbank_id: str) -> list[QuestionerAnswerClaim]:
        rows = list(
            await self.session.scalars(
                select(QuestionerAnswerClaimModel)
                .where(QuestionerAnswerClaimModel.qbank_id == qbank_id)
                .order_by(QuestionerAnswerClaimModel.ordinal)
            )
        )
        return [questioner_answer_claim_from_row(_model_mapping(item)) for item in rows]

    async def add_candidate_family(
        self,
        family: QBankCandidateFamily,
        members: list[QBankCandidateFamilyMember],
    ) -> QBankCandidateFamily:
        if family.member_count != len(members):
            raise ValueError("candidate-family member count does not reconcile")
        if any(item.family_id != family.family_id for item in members):
            raise ValueError("candidate-family member belongs to a different family")
        if sum(item.is_primary for item in members) != 1:
            raise ValueError("candidate family requires exactly one primary")
        stored = await _insert_idempotent(
            self.session,
            QBankCandidateFamilyModel,
            qbank_candidate_family_to_row(family),
            family.family_id,
        )
        for member in members:
            await _insert_idempotent(
                self.session,
                QBankCandidateFamilyMemberModel,
                qbank_candidate_family_member_to_row(member),
                member.family_member_id,
            )
        return qbank_candidate_family_from_row(_model_mapping(stored))

    async def list_candidate_family_members(
        self,
        family_id: str,
    ) -> list[QBankCandidateFamilyMember]:
        rows = list(
            await self.session.scalars(
                select(QBankCandidateFamilyMemberModel)
                .where(QBankCandidateFamilyMemberModel.family_id == family_id)
                .order_by(QBankCandidateFamilyMemberModel.ordinal)
            )
        )
        return [qbank_candidate_family_member_from_row(_model_mapping(item)) for item in rows]

    async def get_generation(self, generation_id: str) -> QuestionerGeneration | None:
        stored = await self.session.get(QuestionerGenerationModel, generation_id)
        return (
            questioner_generation_from_row(_model_mapping(stored)) if stored is not None else None
        )

    async def get_generation_for_unit(
        self,
        compiler_unit_id: str,
        *,
        successful_only: bool = True,
    ) -> QuestionerGeneration | None:
        statement = select(QuestionerGenerationModel).where(
            QuestionerGenerationModel.compiler_unit_id == compiler_unit_id
        )
        if successful_only:
            statement = statement.where(QuestionerGenerationModel.parse_status != "failed")
        stored = await self.session.scalar(
            statement.order_by(QuestionerGenerationModel.attempt_number.desc()).limit(1)
        )
        return (
            questioner_generation_from_row(_model_mapping(stored)) if stored is not None else None
        )

    async def persist_gated_candidates(
        self,
        values: list[GatedQuestionCandidate],
    ) -> None:
        for value in values:
            await self.questions.add_question(
                value.candidate,
                expected_evidence=list(value.expected_evidence),
            )
            await _insert_idempotent(
                self.session,
                QuestionerCandidateGateResultModel,
                questioner_gate_to_row(value.gate_result),
                value.gate_result.gate_result_id,
            )
            if value.duplicate_link is not None:
                await _insert_idempotent(
                    self.session,
                    QBankDuplicateLinkModel,
                    qbank_duplicate_to_row(value.duplicate_link),
                    value.duplicate_link.duplicate_link_id,
                )

    async def add_candidate(
        self,
        value: QuestionCandidate,
        *,
        expected_evidence: list[OperatorEvidence],
    ) -> QuestionCandidate:
        return await self.questions.add_question(
            value,
            expected_evidence=expected_evidence,
        )

    async def add_gate(self, value: QuestionCandidateGateResult) -> None:
        await _insert_idempotent(
            self.session,
            QuestionerCandidateGateResultModel,
            questioner_gate_to_row(value),
            value.gate_result_id,
        )

    async def add_verification_batch(
        self,
        batch: QuestionerVerificationBatch,
        values: list[QuestionerCandidateVerification],
    ) -> QuestionerVerificationBatch:
        if batch.input_count != len(values):
            raise ValueError("verification batch count does not match item facts")
        if any(value.verification_batch_id != batch.verification_batch_id for value in values):
            raise ValueError("candidate verification belongs to a different batch")
        stored = await _insert_idempotent(
            self.session,
            QuestionerVerificationBatchModel,
            questioner_verification_batch_to_row(batch),
            batch.verification_batch_id,
        )
        for value in values:
            await _insert_idempotent(
                self.session,
                QuestionerCandidateVerificationModel,
                questioner_candidate_verification_to_row(value),
                value.candidate_verification_id,
            )
        return questioner_verification_batch_from_row(_model_mapping(stored))

    async def get_verification_batch(
        self,
        compiler_unit_id: str,
        generation_id: str,
        verification_round: int,
        *,
        scope_qbank_id: str | None = None,
    ) -> (
        tuple[
            QuestionerVerificationBatch,
            list[QuestionerCandidateVerification],
        ]
        | None
    ):
        query = select(QuestionerVerificationBatchModel).where(
            QuestionerVerificationBatchModel.compiler_unit_id == compiler_unit_id,
            QuestionerVerificationBatchModel.generation_id == generation_id,
            QuestionerVerificationBatchModel.verification_round == verification_round,
        )
        query = query.where(QuestionerVerificationBatchModel.scope_qbank_id == scope_qbank_id)
        stored = await self.session.scalar(query)
        if stored is None:
            return None
        items = list(
            (
                await self.session.execute(
                    select(QuestionerCandidateVerificationModel)
                    .where(
                        QuestionerCandidateVerificationModel.verification_batch_id
                        == stored.verification_batch_id
                    )
                    .order_by(QuestionerCandidateVerificationModel.candidate_alias)
                )
            )
            .scalars()
            .all()
        )
        return (
            questioner_verification_batch_from_row(_model_mapping(stored)),
            [questioner_candidate_verification_from_row(_model_mapping(item)) for item in items],
        )

    async def add_supersession(
        self, value: QuestionerCandidateSupersession
    ) -> QuestionerCandidateSupersession:
        stored = await _insert_idempotent(
            self.session,
            QuestionerCandidateSupersessionModel,
            questioner_candidate_supersession_to_row(value),
            value.supersession_id,
        )
        return questioner_candidate_supersession_from_row(_model_mapping(stored))

    async def get_gate(self, qbank_id: str) -> QuestionCandidateGateResult | None:
        stored = await self.session.scalar(
            select(QuestionerCandidateGateResultModel).where(
                QuestionerCandidateGateResultModel.qbank_id == qbank_id
            )
        )
        return questioner_gate_from_row(_model_mapping(stored)) if stored is not None else None

    async def list_eligible_questions(self, run_id: str) -> list[QuestionCandidate]:
        compiler_schema_version = await self.session.scalar(
            select(CompilerRunModel.compiler_schema_version).where(
                CompilerRunModel.run_id == run_id
            )
        )
        query = (
            select(QuestionerCandidateGateResultModel.qbank_id)
            .join(
                CompilerUnitModel,
                CompilerUnitModel.compiler_unit_id
                == QuestionerCandidateGateResultModel.compiler_unit_id,
            )
            .join(
                QBankQuestionModel,
                QBankQuestionModel.qbank_id == QuestionerCandidateGateResultModel.qbank_id,
            )
            .where(
                QuestionerCandidateGateResultModel.run_id == run_id,
                QuestionerCandidateGateResultModel.status
                == CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value,
            )
        )
        if compiler_schema_version in {
            "0.3.0",
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
            "0.9.0",
        }:
            query = query.join(
                QuestionerCandidateVerificationModel,
                (QuestionerCandidateVerificationModel.qbank_id == QBankQuestionModel.qbank_id)
                & (
                    QuestionerCandidateVerificationModel.verification_round
                    == QBankQuestionModel.candidate_version
                ),
            ).where(QuestionerCandidateVerificationModel.verdict == "pass")
        qbank_ids = list(
            await self.session.scalars(
                query.order_by(
                    CompilerUnitModel.ordinal,
                    QBankQuestionModel.generation_rank,
                    QBankQuestionModel.candidate_version,
                    QBankQuestionModel.qbank_id,
                )
            )
        )
        result: list[QuestionCandidate] = []
        for qbank_id in qbank_ids:
            value = await self.questions.get_question(qbank_id)
            if value is None:
                raise RuntimeError("eligible QBank gate references a missing question")
            result.append(value)
        return result

    async def list_duplicate_primaries(self, run_id: str) -> dict[str, str]:
        rows = (
            await self.session.execute(
                select(QuestionerCandidateGateResultModel).where(
                    QuestionerCandidateGateResultModel.run_id == run_id,
                    QuestionerCandidateGateResultModel.status
                    == CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value,
                )
            )
        ).scalars()
        result: dict[str, str] = {}
        for row in rows:
            key = row.metadata_json.get("duplicate_key")
            if isinstance(key, str) and key:
                result[key] = row.qbank_id
        return result

    async def rebuild_candidate_families(self, run_id: str) -> int:
        """Materialize deterministic conversation-scoped candidate families."""

        compiler = await self.session.get(CompilerRunModel, run_id)
        if compiler is None:
            raise KeyError(f"compiler run does not exist: {run_id}")
        expected_dedupe_version = _candidate_family_dedupe_version(
            compiler_schema_version=compiler.compiler_schema_version,
            questioner_prompt_version=compiler.questioner_prompt_version,
        )

        rows = list(
            (
                await self.session.execute(
                    select(
                        QBankQuestionModel,
                        QuestionerCandidateGateResultModel,
                        CompilerUnitModel.ordinal,
                    )
                    .join(
                        QuestionerCandidateGateResultModel,
                        QuestionerCandidateGateResultModel.qbank_id == QBankQuestionModel.qbank_id,
                    )
                    .join(
                        CompilerUnitModel,
                        CompilerUnitModel.compiler_unit_id == QBankQuestionModel.compiler_unit_id,
                    )
                    .where(
                        QBankQuestionModel.run_id == run_id,
                    )
                    .order_by(
                        QBankQuestionModel.conversation_id,
                        CompilerUnitModel.ordinal,
                        QBankQuestionModel.generation_rank,
                        QBankQuestionModel.candidate_version,
                        QBankQuestionModel.qbank_id,
                    )
                )
            ).all()
        )
        grouped: dict[tuple[str, str], list[tuple[Any, Any, int]]] = {}
        for question, gate, unit_ordinal in rows:
            key = gate.metadata_json.get("duplicate_key")
            if isinstance(key, str) and key:
                grouped.setdefault(
                    (str(question.conversation_id), key),
                    [],
                ).append((question, gate, int(unit_ordinal)))
        materialized_count = 0
        for (conversation_id, family_key), values in sorted(grouped.items()):
            values.sort(
                key=lambda item: (
                    item[1].status != CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value,
                    item[2],
                    item[0].generation_rank,
                    item[0].candidate_version,
                    item[0].qbank_id,
                )
            )
            if values[0][1].status != CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value:
                continue
            primary = values[0][0]
            answer_hash = values[0][1].metadata_json.get("answer_claim_hash")
            requirement_hash = values[0][1].metadata_json.get("requirement_contract_hash")
            dedupe_version = values[0][1].metadata_json.get("family_dedupe_version")
            if not isinstance(answer_hash, str) or not isinstance(requirement_hash, str):
                continue
            if not isinstance(dedupe_version, str) or not dedupe_version:
                dedupe_version = expected_dedupe_version
            elif dedupe_version != expected_dedupe_version:
                raise PersistenceConflictError(
                    "candidate family dedupe version differs from compiler identity"
                )
            family_id = stable_compiler_id(
                "qbank-family",
                [run_id, conversation_id, family_key],
            )
            members = [
                QBankCandidateFamilyMember(
                    family_member_id=stable_compiler_id(
                        "qbank-family-member",
                        [family_id, question.qbank_id],
                    ),
                    family_id=family_id,
                    run_id=run_id,
                    qbank_id=question.qbank_id,
                    ordinal=ordinal,
                    is_primary=ordinal == 0,
                    content_hash=canonical_content_hash([family_id, question.qbank_id, ordinal]),
                )
                for ordinal, (question, _gate, _unit_ordinal) in enumerate(values)
            ]
            family = QBankCandidateFamily(
                family_id=family_id,
                run_id=run_id,
                snapshot_id=primary.snapshot_id,
                conversation_id=conversation_id,
                primary_qbank_id=primary.qbank_id,
                normalized_question_hash=primary.metadata_json["normalized_question_hash"],
                answer_claim_hash=answer_hash,
                requirement_contract_hash=requirement_hash,
                member_count=len(members),
                content_hash=canonical_content_hash(
                    {
                        "family_key": family_key,
                        "members": [item.qbank_id for item in members],
                    }
                ),
                created_at=primary.created_at,
                metadata={"dedupe_version": dedupe_version},
            )
            if dedupe_version == "qbank-family-dedupe-v5":
                answer_hashes = sorted(
                    {
                        str(gate.metadata_json.get("answer_claim_hash"))
                        for _question, gate, _ordinal in values
                    }
                )
                family = family.model_copy(
                    update={
                        "metadata": {
                            "dedupe_version": dedupe_version,
                            "ranking_version": "questioner-query-blind-family-rank-v1",
                            "answer_conflict": len(answer_hashes) > 1,
                            "member_answer_claim_hashes": answer_hashes,
                            "alias_members": [
                                question.canonical_question for question, _, _ in values
                            ],
                        }
                    }
                )
            await self.add_candidate_family(family, members)
            materialized_count += 1
        return materialized_count

    async def record_semantic_nearest_neighbors(
        self,
        *,
        run_id: str,
        collection_id: str,
        method: str,
    ) -> int:
        """Persist each eligible question's nearest earlier semantic neighbor.

        These links are diagnostic only. They never change candidate gate status.
        Restricting the target to an earlier generation preserves prospective
        ordering and makes the result deterministic when scores tie.
        """

        rows = (
            await self.session.execute(
                text(
                    """WITH ordered_questions AS (
                        SELECT records.qbank_id, records.embedding,
                               questions.conversation_id,
                               questions.generation_rank,
                               questions.created_at,
                               units.ordinal AS unit_ordinal
                          FROM pmmc_runtime.embeddings records
                          JOIN pmmc_runtime.qbank_questions questions
                            ON questions.qbank_id = records.qbank_id
                          JOIN pmmc_runtime.questioner_candidate_gate_results gates
                            ON gates.qbank_id = questions.qbank_id
                           AND gates.run_id = questions.run_id
                          JOIN pmmc_runtime.compiler_units units
                            ON units.compiler_unit_id = questions.compiler_unit_id
                           AND units.run_id = questions.run_id
                         WHERE records.collection_id = :collection_id
                           AND questions.run_id = :run_id
                           AND gates.status = 'eligible_for_planning'
                    )
                    SELECT source.qbank_id AS source_qbank_id,
                           nearest.target_qbank_id,
                           nearest.score,
                           source.created_at
                      FROM ordered_questions source
                      JOIN LATERAL (
                        SELECT target.qbank_id AS target_qbank_id,
                               1.0 - (source.embedding <=> target.embedding) AS score
                          FROM ordered_questions target
                         WHERE target.conversation_id = source.conversation_id
                           AND (target.unit_ordinal, target.generation_rank,
                                target.qbank_id)
                               < (source.unit_ordinal, source.generation_rank,
                                  source.qbank_id)
                         ORDER BY source.embedding <=> target.embedding,
                                  target.qbank_id
                         LIMIT 1
                      ) nearest ON true
                     ORDER BY source.unit_ordinal, source.generation_rank,
                              source.qbank_id"""
                ),
                {"collection_id": collection_id, "run_id": run_id},
            )
        ).mappings()
        count = 0
        for row in rows:
            source_qbank_id = str(row["source_qbank_id"])
            target_qbank_id = str(row["target_qbank_id"])
            link = QBankDuplicateLink(
                duplicate_link_id=stable_compiler_id(
                    "qbank-semantic-neighbor",
                    [run_id, source_qbank_id, target_qbank_id, method],
                ),
                run_id=run_id,
                duplicate_qbank_id=source_qbank_id,
                primary_qbank_id=target_qbank_id,
                method=method,
                score=RankingScore(
                    score_type="cosine",
                    raw_value=float(row["score"]),
                    metadata={"collection_id": collection_id},
                ),
                hard_duplicate=False,
                created_at=row["created_at"],
                metadata={"diagnostic_only": True},
            )
            await _insert_idempotent(
                self.session,
                QBankDuplicateLinkModel,
                qbank_duplicate_to_row(link),
                link.duplicate_link_id,
            )
            count += 1
        return count

    async def list_expected_evidence(self, run_id: str) -> dict[str, list[OperatorEvidence]]:
        rows = list(
            (
                await self.session.execute(
                    select(QBankEvidenceRefModel)
                    .where(
                        QBankEvidenceRefModel.run_id == run_id,
                        QBankEvidenceRefModel.role == "expected",
                    )
                    .order_by(QBankEvidenceRefModel.qbank_id, QBankEvidenceRefModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        result: dict[str, list[OperatorEvidence]] = {}
        for row in rows:
            result.setdefault(row.qbank_id, []).append(
                OperatorEvidence(
                    evidence_id=row.evidence_id,
                    evidence_type=row.evidence_type,
                    memory_id=row.memory_id,
                    image_id=row.image_id,
                    session_id=row.session_id,
                    caption_id=row.caption_id,
                    metadata=dict(row.metadata_json),
                )
            )
        return result


class PostgresQBankIndexRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, value: QBankIndexManifest) -> QBankIndexManifest:
        stored = await _insert_idempotent(
            self.session,
            QBankIndexManifestModel,
            qbank_index_to_row(value),
            value.qbank_index_id,
        )
        return qbank_index_from_row(_model_mapping(stored))

    async def get(self, qbank_index_id: str) -> QBankIndexManifest | None:
        stored = await self.session.get(QBankIndexManifestModel, qbank_index_id)
        return qbank_index_from_row(_model_mapping(stored)) if stored is not None else None

    async def get_for_run(self, run_id: str) -> QBankIndexManifest | None:
        stored = await self.session.scalar(
            select(QBankIndexManifestModel).where(QBankIndexManifestModel.compiler_run_id == run_id)
        )
        return qbank_index_from_row(_model_mapping(stored)) if stored is not None else None

    async def mark_ready(self, qbank_index_id: str) -> QBankIndexManifest:
        result = await self.session.execute(
            update(QBankIndexManifestModel)
            .where(
                QBankIndexManifestModel.qbank_index_id == qbank_index_id,
                QBankIndexManifestModel.status == EmbeddingCollectionStatus.BUILDING.value,
            )
            .values(status=EmbeddingCollectionStatus.READY.value)
        )
        if cast(Any, result).rowcount != 1:
            existing = await self.get(qbank_index_id)
            if existing is None:
                raise KeyError(qbank_index_id)
            if existing.status is not EmbeddingCollectionStatus.READY:
                raise InvalidStateTransitionError("QBank index cannot transition to ready")
            return existing
        stored = await self.get(qbank_index_id)
        if stored is None:
            raise RuntimeError("ready QBank index disappeared")
        return stored

    async def lexical_search(
        self,
        *,
        compiler_run_id: str,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
        query: str,
        question_budget: int,
        top_k: int,
    ) -> list[tuple[str, float]]:
        if question_budget <= 0 or top_k <= 0:
            raise ValueError("QBank lexical search limits must be positive")
        rows = (
            await self.session.execute(
                text(
                    """WITH cutoff AS (
                        SELECT session_order FROM pmmc_runtime.benchmark_sessions
                         WHERE snapshot_id = :snapshot_id
                           AND conversation_id = :conversation_id
                           AND session_id = :cutoff_session_id
                    ), documents AS (
                        SELECT questions.qbank_id,
                            ts_rank_cd(questions.search_document,
                                plainto_tsquery('simple', :query)) AS score
                        FROM pmmc_runtime.qbank_questions questions
                        JOIN pmmc_runtime.questioner_candidate_gate_results gates
                          ON gates.qbank_id = questions.qbank_id
                         AND gates.run_id = questions.run_id
                        JOIN pmmc_runtime.benchmark_sessions available
                          ON available.snapshot_id = questions.snapshot_id
                         AND available.conversation_id = questions.conversation_id
                         AND available.session_id = questions.created_after_session
                        CROSS JOIN cutoff
                        WHERE questions.run_id = :compiler_run_id
                          AND questions.snapshot_id = :snapshot_id
                          AND questions.conversation_id = :conversation_id
                          AND CASE WHEN gates.gate_version = 'questioner-static-semantic-gate-v3.6'
                              THEN questions.generation_rank ELSE COALESCE(
                              (gates.metadata->>'family_budget_rank')::integer,
                              questions.generation_rank) END <= :question_budget
                          AND gates.status = 'eligible_for_planning'
                          AND available.session_order <= cutoff.session_order
                          AND questions.search_document @@ plainto_tsquery('simple', :query)
                        UNION ALL
                        SELECT paraphrases.qbank_id,
                            ts_rank_cd(paraphrases.search_document,
                                plainto_tsquery('simple', :query)) AS score
                        FROM pmmc_runtime.qbank_paraphrases paraphrases
                        JOIN pmmc_runtime.qbank_questions questions
                          ON questions.qbank_id = paraphrases.qbank_id
                        JOIN pmmc_runtime.questioner_candidate_gate_results gates
                          ON gates.qbank_id = questions.qbank_id
                         AND gates.run_id = questions.run_id
                        JOIN pmmc_runtime.benchmark_sessions available
                          ON available.snapshot_id = questions.snapshot_id
                         AND available.conversation_id = questions.conversation_id
                         AND available.session_id = questions.created_after_session
                        CROSS JOIN cutoff
                        WHERE questions.run_id = :compiler_run_id
                          AND questions.snapshot_id = :snapshot_id
                          AND questions.conversation_id = :conversation_id
                          AND CASE WHEN gates.gate_version = 'questioner-static-semantic-gate-v3.6'
                              THEN questions.generation_rank ELSE COALESCE(
                              (gates.metadata->>'family_budget_rank')::integer,
                              questions.generation_rank) END <= :question_budget
                          AND gates.status = 'eligible_for_planning'
                          AND available.session_order <= cutoff.session_order
                          AND paraphrases.search_document @@ plainto_tsquery('simple', :query)
                    )
                    SELECT qbank_id, max(score)::double precision AS score
                    FROM documents GROUP BY qbank_id
                    ORDER BY score DESC, qbank_id LIMIT :top_k"""
                ),
                {
                    "compiler_run_id": compiler_run_id,
                    "snapshot_id": snapshot_id,
                    "conversation_id": conversation_id,
                    "cutoff_session_id": cutoff_session_id,
                    "query": query,
                    "question_budget": question_budget,
                    "top_k": top_k,
                },
            )
        ).mappings()
        return [(str(row["qbank_id"]), float(row["score"])) for row in rows]


class PostgresQBankCoverageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_run(self, value: QBankCoverageRunManifest) -> QBankCoverageRunManifest:
        stored = await _insert_idempotent(
            self.session,
            QBankCoverageRunModel,
            coverage_run_to_row(value),
            value.coverage_run_id,
        )
        return coverage_run_from_row(_model_mapping(stored))

    async def get_run(self, coverage_run_id: str) -> QBankCoverageRunManifest | None:
        stored = await self.session.get(QBankCoverageRunModel, coverage_run_id)
        return coverage_run_from_row(_model_mapping(stored)) if stored is not None else None

    async def transition(
        self,
        coverage_run_id: str,
        status: QBankCoverageStatus,
    ) -> QBankCoverageRunManifest:
        current = await self.get_run(coverage_run_id)
        if current is None:
            raise KeyError(coverage_run_id)
        allowed = {
            QBankCoverageStatus.CREATED: {
                QBankCoverageStatus.RUNNING,
                QBankCoverageStatus.FAILED,
            },
            QBankCoverageStatus.RUNNING: {
                QBankCoverageStatus.COMPLETED,
                QBankCoverageStatus.FAILED,
            },
            QBankCoverageStatus.COMPLETED: set(),
            QBankCoverageStatus.FAILED: set(),
        }
        if status is current.status:
            return current
        if status not in allowed[current.status]:
            raise InvalidStateTransitionError(
                f"invalid QBank coverage transition: {current.status} -> {status}"
            )
        await self.session.execute(
            update(QBankCoverageRunModel)
            .where(QBankCoverageRunModel.coverage_run_id == coverage_run_id)
            .values(
                status=status.value,
                completed_at=utc_now() if status is QBankCoverageStatus.COMPLETED else None,
            )
        )
        stored = await self.get_run(coverage_run_id)
        if stored is None:
            raise RuntimeError("coverage run disappeared after transition")
        return stored

    async def add_query_bundle(
        self,
        *,
        query: QBankCoverageQuery,
        embeddings: list[QBankCoverageQueryEmbedding],
        gold_groups: list[ResolvedGoldEvidenceGroup],
        candidates: list[QBankCoverageCandidate],
        metrics: list[QBankCoverageMetric],
        labels: list[QBankCoverageLabel],
        matched_members: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        await self.add_query_facts(query=query, gold_groups=gold_groups)
        for embedding in embeddings:
            await self.add_query_embedding(embedding)
        for coverage_candidate in candidates:
            await _insert_idempotent(
                self.session,
                QBankCoverageCandidateModel,
                coverage_candidate_to_row(coverage_candidate),
                coverage_candidate.coverage_candidate_id,
            )
            for ordinal, gold_group_id in enumerate(coverage_candidate.matched_gold_group_ids):
                member_ids = (matched_members or {}).get(
                    (coverage_candidate.coverage_candidate_id, gold_group_id), []
                )
                row = {
                    "coverage_candidate_id": coverage_candidate.coverage_candidate_id,
                    "gold_group_id": gold_group_id,
                    "ordinal": ordinal,
                    "matched_member_ids": member_ids,
                }
                await _insert_idempotent(
                    self.session,
                    QBankCoverageMatchModel,
                    row,
                    coverage_candidate.coverage_candidate_id,
                    gold_group_id,
                )
        for metric in metrics:
            await _insert_idempotent(
                self.session,
                QBankCoverageMetricModel,
                coverage_metric_to_row(metric),
                metric.metric_id,
            )
        for label in labels:
            await _insert_idempotent(
                self.session,
                QBankCoverageLabelModel,
                coverage_label_to_row(label),
                label.label_id,
            )

    async def add_query_facts(
        self,
        *,
        query: QBankCoverageQuery,
        gold_groups: list[ResolvedGoldEvidenceGroup],
    ) -> None:
        """Persist the immutable denominator and resolved gold before model stages."""

        await _insert_idempotent(
            self.session,
            QBankCoverageQueryModel,
            coverage_query_to_row(query),
            query.coverage_run_id,
            query.query_id,
        )
        for gold_group in gold_groups:
            rows = coverage_gold_to_rows(
                gold_group,
                coverage_run_id=query.coverage_run_id,
            )
            await _insert_idempotent(
                self.session,
                QBankCoverageGoldGroupModel,
                rows["group"][0],
                gold_group.gold_group_id,
            )
            for row in rows["members"]:
                await _insert_idempotent(
                    self.session,
                    QBankCoverageGoldGroupMemberModel,
                    row,
                    row["gold_group_id"],
                    row["member_type"],
                    row["member_id"],
                )

    async def add_query_embedding(
        self, value: QBankCoverageQueryEmbedding
    ) -> QBankCoverageQueryEmbedding:
        stored = await _insert_idempotent(
            self.session,
            QBankCoverageQueryEmbeddingModel,
            coverage_embedding_to_row(value),
            value.query_embedding_id,
        )
        return coverage_embedding_from_row(_model_mapping(stored))

    async def get_query_embedding(
        self,
        coverage_run_id: str,
        query_id: str,
        channel: str,
    ) -> QBankCoverageQueryEmbedding | None:
        stored = await self.session.scalar(
            select(QBankCoverageQueryEmbeddingModel).where(
                QBankCoverageQueryEmbeddingModel.coverage_run_id == coverage_run_id,
                QBankCoverageQueryEmbeddingModel.query_id == query_id,
                QBankCoverageQueryEmbeddingModel.channel == channel,
            )
        )
        if stored is None:
            return None
        return coverage_embedding_from_row(_model_mapping(stored))

    async def list_query_embeddings(
        self,
        coverage_run_id: str,
    ) -> list[QBankCoverageQueryEmbedding]:
        stored = (
            await self.session.scalars(
                select(QBankCoverageQueryEmbeddingModel)
                .where(QBankCoverageQueryEmbeddingModel.coverage_run_id == coverage_run_id)
                .order_by(
                    QBankCoverageQueryEmbeddingModel.query_id,
                    QBankCoverageQueryEmbeddingModel.channel,
                )
            )
        ).all()
        return [coverage_embedding_from_row(_model_mapping(item)) for item in stored]

    async def add_report(self, value: QBankCoverageGateReport) -> QBankCoverageGateReport:
        stored = await _insert_idempotent(
            self.session,
            QBankCoverageGateReportModel,
            coverage_report_to_row(value),
            value.gate_report_id,
        )
        return coverage_report_from_row(_model_mapping(stored))

    async def get_report(self, coverage_run_id: str) -> QBankCoverageGateReport | None:
        stored = await self.session.scalar(
            select(QBankCoverageGateReportModel).where(
                QBankCoverageGateReportModel.coverage_run_id == coverage_run_id
            )
        )
        return coverage_report_from_row(_model_mapping(stored)) if stored else None

    async def get_query(self, coverage_run_id: str, query_id: str) -> QBankCoverageQuery | None:
        stored = await self.session.get(QBankCoverageQueryModel, (coverage_run_id, query_id))
        return coverage_query_from_row(_model_mapping(stored)) if stored else None

    async def list_candidates(
        self,
        coverage_run_id: str,
        query_id: str,
        question_budget: int,
    ) -> list[QBankCoverageCandidate]:
        rows = list(
            (
                await self.session.execute(
                    select(QBankCoverageCandidateModel)
                    .where(
                        QBankCoverageCandidateModel.coverage_run_id == coverage_run_id,
                        QBankCoverageCandidateModel.query_id == query_id,
                        QBankCoverageCandidateModel.question_budget == question_budget,
                    )
                    .order_by(QBankCoverageCandidateModel.rank)
                )
            )
            .scalars()
            .all()
        )
        return [coverage_candidate_from_row(_model_mapping(item)) for item in rows]

    async def list_metrics(
        self, coverage_run_id: str, query_id: str | None = None
    ) -> list[QBankCoverageMetric]:
        statement = select(QBankCoverageMetricModel).where(
            QBankCoverageMetricModel.coverage_run_id == coverage_run_id
        )
        if query_id is not None:
            statement = statement.where(QBankCoverageMetricModel.query_id == query_id)
        rows = list(
            (await self.session.execute(statement.order_by(QBankCoverageMetricModel.metric_id)))
            .scalars()
            .all()
        )
        return [coverage_metric_from_row(_model_mapping(item)) for item in rows]

    async def list_gold_groups(
        self, coverage_run_id: str, query_id: str
    ) -> list[ResolvedGoldEvidenceGroup]:
        groups = list(
            (
                await self.session.execute(
                    select(QBankCoverageGoldGroupModel)
                    .where(
                        QBankCoverageGoldGroupModel.coverage_run_id == coverage_run_id,
                        QBankCoverageGoldGroupModel.query_id == query_id,
                    )
                    .order_by(QBankCoverageGoldGroupModel.gold_group_id)
                )
            )
            .scalars()
            .all()
        )
        result: list[ResolvedGoldEvidenceGroup] = []
        for group in groups:
            members = list(
                (
                    await self.session.execute(
                        select(QBankCoverageGoldGroupMemberModel).where(
                            QBankCoverageGoldGroupMemberModel.gold_group_id == group.gold_group_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            result.append(
                coverage_gold_from_rows(
                    _model_mapping(group),
                    [_model_mapping(member) for member in members],
                )
            )
        return result


async def inspect_questioner_unit(
    session: AsyncSession,
    *,
    run_id: str,
    compiler_unit_id: str,
) -> dict[str, Any]:
    """Return runtime-safe Questioner accounting without candidate or answer text."""

    unit = await session.get(CompilerUnitModel, compiler_unit_id)
    if unit is None or unit.run_id != run_id:
        raise KeyError(f"Questioner unit does not exist in run: {compiler_unit_id}")
    generation = await session.scalar(
        select(QuestionerGenerationModel)
        .where(
            QuestionerGenerationModel.run_id == run_id,
            QuestionerGenerationModel.compiler_unit_id == compiler_unit_id,
        )
        .order_by(QuestionerGenerationModel.attempt_number.desc())
        .limit(1)
    )
    gates = (
        await session.execute(
            select(QuestionerCandidateGateResultModel.status, func.count())
            .where(
                QuestionerCandidateGateResultModel.run_id == run_id,
                QuestionerCandidateGateResultModel.compiler_unit_id == compiler_unit_id,
            )
            .group_by(QuestionerCandidateGateResultModel.status)
        )
    ).all()
    stage = await session.scalar(
        select(CompilerUnitStageResultModel)
        .where(
            CompilerUnitStageResultModel.run_id == run_id,
            CompilerUnitStageResultModel.compiler_unit_id == compiler_unit_id,
            CompilerUnitStageResultModel.stage == "questioner",
        )
        .order_by(CompilerUnitStageResultModel.attempt_number.desc())
        .limit(1)
    )
    member_count = int(
        await session.scalar(
            select(func.count())
            .select_from(CompilerUnitMemoryModel)
            .where(CompilerUnitMemoryModel.compiler_unit_id == compiler_unit_id)
        )
        or 0
    )
    link_count = int(
        await session.scalar(
            select(func.count())
            .select_from(CompilerUnitLinkModel)
            .where(CompilerUnitLinkModel.compiler_unit_id == compiler_unit_id)
        )
        or 0
    )
    question_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QBankQuestionModel)
            .where(
                QBankQuestionModel.run_id == run_id,
                QBankQuestionModel.compiler_unit_id == compiler_unit_id,
            )
        )
        or 0
    )
    return {
        "run_id": run_id,
        "compiler_unit_id": compiler_unit_id,
        "snapshot_id": unit.snapshot_id,
        "conversation_id": unit.conversation_id,
        "available_after_session": unit.available_after_session,
        "ordinal": unit.ordinal,
        "member_count": member_count,
        "link_count": link_count,
        "generation": (
            {
                "generation_id": generation.generation_id,
                "model_call_id": generation.model_call_id,
                "attempt_number": generation.attempt_number,
                "parse_status": generation.parse_status,
                "raw_candidate_count": generation.raw_candidate_count,
                "valid_candidate_count": generation.valid_candidate_count,
                "invalid_candidate_count": generation.invalid_candidate_count,
                "invalid_diagnostics": _safe_questioner_diagnostics(generation.metadata_json),
            }
            if generation is not None
            else None
        ),
        "gate_counts": {status: int(count) for status, count in gates},
        "question_count": question_count,
        "stage_result": (
            {
                "status": stage.status,
                "attempt_number": stage.attempt_number,
                "produced_count": stage.produced_count,
                "accepted_count": stage.accepted_count,
                "rejected_count": stage.rejected_count,
            }
            if stage is not None
            else None
        ),
    }


def _safe_questioner_diagnostics(metadata: object) -> list[dict[str, object]]:
    if not isinstance(metadata, dict):
        return []
    values = metadata.get("invalid_diagnostics", [])
    if not isinstance(values, list):
        return []
    result: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        errors: list[dict[str, str]] = []
        raw_errors = value.get("errors", [])
        if isinstance(raw_errors, list):
            for error in raw_errors:
                if isinstance(error, dict):
                    errors.append(
                        {
                            "location": str(error.get("location", "unknown")),
                            "type": str(error.get("type", "validation_error")),
                        }
                    )
        result.append(
            {
                "ordinal": (
                    value["ordinal"] if isinstance(value.get("ordinal"), int) else "unknown"
                ),
                "item_hash": str(value.get("item_hash", "unknown")),
                "error_type": str(value.get("error_type", "ValidationError")),
                "error_count": len(errors),
                "errors": sorted(
                    errors,
                    key=lambda item: (item["location"], item["type"]),
                ),
            }
        )
    return result


async def verify_questioner_schema_conformance(
    session: AsyncSession,
    run_id: str,
    *,
    require_managed_runtime: bool = True,
) -> dict[str, object]:
    """Verify one Questioner-only managed run without requiring QBank indexes."""

    compiler = await session.get(CompilerRunModel, run_id)
    if compiler is None:
        raise KeyError(f"compiler run does not exist: {run_id}")
    issues: list[str] = []
    if compiler.questioner_prompt_version != "pmmc-questioner-v2":
        issues.append("schema probe does not use the Questioner v2 prompt")
    if compiler.questioner_renderer_version != "questioner-multimodal-renderer-v2":
        issues.append("schema probe does not use the Questioner v2 renderer")
    if compiler.questioner_parser_version != "questioner-json-parser-v2":
        issues.append("schema probe does not use the Questioner v2 parser")
    if not compiler.questioner_model_alias or not compiler.questioner_model_revision:
        issues.append("schema probe has no frozen compiler model identity")
    from pmmc.compiler.questioner_prompt import questioner_response_schema_sha256

    expected_schema_hash = questioner_response_schema_sha256("pmmc-questioner-v2")
    if compiler.metadata_json.get("questioner_response_schema_hash") != (expected_schema_hash):
        issues.append("schema probe compiler identity has the wrong response schema hash")

    generations = list(
        await session.scalars(
            select(QuestionerGenerationModel)
            .where(QuestionerGenerationModel.run_id == run_id)
            .order_by(QuestionerGenerationModel.compiler_unit_id)
        )
    )
    if len(generations) != compiler.unit_count or not generations:
        issues.append("schema probe generation denominator is incomplete")
    invalid_count = sum(item.invalid_candidate_count for item in generations)
    raw_count = sum(item.raw_candidate_count for item in generations)
    valid_count = sum(item.valid_candidate_count for item in generations)
    if invalid_count:
        issues.append("schema probe produced a schema-invalid candidate")
    if any(item.parse_status == "failed" for item in generations):
        issues.append("schema probe contains a failed response envelope")
    if any(
        item.metadata_json.get("response_schema_hash") != expected_schema_hash
        for item in generations
    ):
        issues.append("schema probe generation has the wrong response schema hash")

    call_rows = (
        (
            await session.execute(
                text(
                    "SELECT model_call_id, model_alias, actual_model_id, status, "
                    "runtime_id, runtime_service_id, input_asset_id, output_asset_id "
                    "FROM pmmc_runtime.model_calls WHERE run_id=:run_id "
                    "AND stage='questioner' ORDER BY model_call_id"
                ),
                {"run_id": run_id},
            )
        )
        .mappings()
        .all()
    )
    if len(call_rows) != compiler.unit_count:
        issues.append("schema probe model-call denominator is incomplete")
    if any(row["status"] != "success" for row in call_rows):
        issues.append("schema probe contains a failed model call")
    if require_managed_runtime and any(
        row["runtime_id"] is None or row["runtime_service_id"] is None for row in call_rows
    ):
        issues.append("schema probe model calls lack managed-runtime lineage")
    if any(
        row["model_alias"] != compiler.questioner_model_alias or row["actual_model_id"] is None
        for row in call_rows
    ):
        issues.append("schema probe model-call identity is invalid")

    artifact_refs = [
        (str(row["input_asset_id"]), "prompt")
        for row in call_rows
        if row["input_asset_id"] is not None
    ] + [
        (str(row["output_asset_id"]), "model_response")
        for row in call_rows
        if row["output_asset_id"] is not None
    ]
    if len(artifact_refs) != 2 * len(call_rows):
        issues.append("schema probe model calls lack input/output artifacts")
    ready_artifacts = {
        (str(row["artifact_id"]), str(row["artifact_type"]))
        for row in (
            await session.execute(
                text(
                    "SELECT artifact_id, artifact_type FROM "
                    "pmmc_runtime.run_artifacts WHERE run_id=:run_id "
                    "AND state='ready'"
                ),
                {"run_id": run_id},
            )
        ).mappings()
    }
    if any(reference not in ready_artifacts for reference in artifact_refs):
        issues.append("schema probe references a missing or non-ready model artifact")
    actual_model_ids = sorted(
        {str(row["actual_model_id"]) for row in call_rows if row["actual_model_id"]}
    )
    return {
        "run_id": run_id,
        "unit_count": compiler.unit_count,
        "generation_count": len(generations),
        "raw_candidate_count": raw_count,
        "schema_valid_candidate_count": valid_count,
        "schema_invalid_candidate_count": invalid_count,
        "compiler_model_alias": compiler.questioner_model_alias,
        "compiler_model_revision": compiler.questioner_model_revision,
        "response_schema_hash": expected_schema_hash,
        "actual_model_ids": actual_model_ids,
        "managed_model_call_count": sum(
            row["runtime_id"] is not None and row["runtime_service_id"] is not None
            for row in call_rows
        ),
        "ready_model_artifact_count": len(artifact_refs),
        "require_managed_runtime": require_managed_runtime,
        "valid": not issues,
        "issues": issues,
    }


async def verify_questioner_run(
    session: AsyncSession,
    run_id: str,
    *,
    require_managed_runtime: bool = False,
    require_schema_conformance: bool = False,
    require_native_structured_output: bool = False,
) -> dict[str, object]:
    """Verify the Questioner denominator, four frozen indexes, and model lineage."""

    from pmmc.infrastructure.database.compiler_repositories import verify_compiler_run

    compiler = await session.get(CompilerRunModel, run_id)
    if compiler is None:
        raise KeyError(f"compiler run does not exist: {run_id}")
    report = await verify_compiler_run(session, run_id)
    issues = list(cast(list[str], report["issues"]))
    if compiler.compiler_schema_version not in {
        "0.2.0",
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
    }:
        issues.append("run does not use the Step 6B compiler schema")
    is_v3 = compiler.compiler_schema_version in {
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
    }
    expected_target = (
        ("verified_qbank" if compiler.doubter_config_hash else "programs")
        if compiler.compiler_schema_version == "0.9.0"
        else "verified_qbank"
        if compiler.compiler_schema_version in {"0.6.0", "0.7.0", "0.8.0"}
        else ("programs" if compiler.compiler_schema_version in {"0.4.0", "0.5.0"} else "questions")
    )
    if compiler.target_stage != expected_target:
        issues.append("run is not a Questioner target")
    generation_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QuestionerGenerationModel)
            .where(
                QuestionerGenerationModel.run_id == run_id,
                QuestionerGenerationModel.parse_status != "failed",
            )
        )
        or 0
    )
    if generation_count != compiler.unit_count:
        issues.append("Questioner generation denominator is incomplete")
    generations = list(
        await session.scalars(
            select(QuestionerGenerationModel)
            .where(QuestionerGenerationModel.run_id == run_id)
            .order_by(QuestionerGenerationModel.generation_id)
        )
    )
    raw_candidate_count = sum(item.raw_candidate_count for item in generations)
    schema_valid_candidate_count = sum(item.valid_candidate_count for item in generations)
    schema_invalid_candidate_count = sum(item.invalid_candidate_count for item in generations)
    zero_valid_candidate_unit_count = sum(item.valid_candidate_count == 0 for item in generations)
    if require_schema_conformance and schema_invalid_candidate_count:
        issues.append("Questioner provider output failed strict schema conformance")
    diagnostic_type_counts: dict[str, int] = {}
    diagnostic_location_counts: dict[str, int] = {}
    for generation in generations:
        for diagnostic in _safe_questioner_diagnostics(generation.metadata_json):
            for error in cast(list[dict[str, str]], diagnostic["errors"]):
                error_type = error["type"]
                location = error["location"]
                diagnostic_type_counts[error_type] = diagnostic_type_counts.get(error_type, 0) + 1
                diagnostic_location_counts[location] = (
                    diagnostic_location_counts.get(location, 0) + 1
                )
    question_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QBankQuestionModel)
            .where(QBankQuestionModel.run_id == run_id)
        )
        or 0
    )
    generated_question_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QBankQuestionModel)
            .where(
                QBankQuestionModel.run_id == run_id,
                QBankQuestionModel.candidate_version == 0,
            )
        )
        or 0
    )
    gate_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QuestionerCandidateGateResultModel)
            .where(QuestionerCandidateGateResultModel.run_id == run_id)
        )
        or 0
    )
    expected_generated_count = generated_question_count if is_v3 else question_count
    if expected_generated_count != schema_valid_candidate_count or gate_count != question_count:
        issues.append("Questioner candidate and gate accounting is incomplete")
    ungated = int(
        await session.scalar(
            text(
                "SELECT count(*) FROM pmmc_runtime.qbank_questions questions "
                "WHERE questions.run_id=:run_id AND NOT EXISTS ("
                "SELECT 1 FROM pmmc_runtime.questioner_candidate_gate_results gates "
                "WHERE gates.run_id=questions.run_id AND gates.qbank_id=questions.qbank_id)"
            ),
            {"run_id": run_id},
        )
        or 0
    )
    if ungated:
        issues.append("Questioner candidates are missing gate results")
    semantic_summary = (
        (
            await session.execute(
                text(
                    "SELECT count(*) AS eligible_count, "
                    "count(DISTINCT questions.conversation_id) AS conversation_count "
                    "FROM pmmc_runtime.qbank_questions questions "
                    "JOIN pmmc_runtime.questioner_candidate_gate_results gates "
                    "ON gates.run_id=questions.run_id AND gates.qbank_id=questions.qbank_id "
                    "WHERE questions.run_id=:run_id "
                    "AND gates.status='eligible_for_planning'"
                ),
                {"run_id": run_id},
            )
        )
        .mappings()
        .one()
    )
    semantic_neighbor_count = int(
        await session.scalar(
            text(
                "SELECT count(*) FROM pmmc_runtime.qbank_duplicate_links "
                "WHERE run_id=:run_id AND NOT hard_duplicate"
            ),
            {"run_id": run_id},
        )
        or 0
    )
    eligible_candidate_count = int(semantic_summary["eligible_count"])
    gate_rejected_candidate_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QuestionerCandidateGateResultModel)
            .where(
                QuestionerCandidateGateResultModel.run_id == run_id,
                QuestionerCandidateGateResultModel.status == CandidateGateStatus.REJECTED.value,
            )
        )
        or 0
    )
    duplicate_candidate_count = int(
        await session.scalar(
            text(
                "SELECT count(*) FROM "
                "pmmc_runtime.questioner_candidate_gate_results "
                "WHERE run_id=:run_id AND 'duplicate'=ANY(rejection_codes)"
            ),
            {"run_id": run_id},
        )
        or 0
    )
    expected_semantic_neighbors = max(
        0,
        int(semantic_summary["eligible_count"]) - int(semantic_summary["conversation_count"]),
    )
    if semantic_neighbor_count != expected_semantic_neighbors:
        issues.append("Questioner semantic nearest-neighbor diagnostics are incomplete")
    opportunity_set_count = 0
    opportunity_count = 0
    verification_batch_count = 0
    verification_candidate_count = 0
    revised_candidate_count = 0
    revision_rescued_count = 0
    if is_v3:
        opportunity_set_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QuestionOpportunitySetModel)
                .where(QuestionOpportunitySetModel.run_id == run_id)
            )
            or 0
        )
        opportunity_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QuestionOpportunityModel)
                .where(QuestionOpportunityModel.run_id == run_id)
            )
            or 0
        )
        verification_batch_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QuestionerVerificationBatchModel)
                .where(QuestionerVerificationBatchModel.run_id == run_id)
            )
            or 0
        )
        verification_candidate_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QuestionerCandidateVerificationModel)
                .where(QuestionerCandidateVerificationModel.run_id == run_id)
            )
            or 0
        )
        revised_candidate_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QBankQuestionModel)
                .where(
                    QBankQuestionModel.run_id == run_id,
                    QBankQuestionModel.candidate_version == 1,
                )
            )
            or 0
        )
        revision_rescued_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QBankQuestionModel)
                .join(
                    QuestionerCandidateGateResultModel,
                    QuestionerCandidateGateResultModel.qbank_id == QBankQuestionModel.qbank_id,
                )
                .where(
                    QBankQuestionModel.run_id == run_id,
                    QBankQuestionModel.candidate_version == 1,
                    QuestionerCandidateGateResultModel.status
                    == CandidateGateStatus.ELIGIBLE_FOR_PLANNING.value,
                )
            )
            or 0
        )
        if opportunity_set_count != compiler.unit_count:
            issues.append("Questioner v3 opportunity-set denominator is incomplete")
        if compiler.compiler_schema_version in {"0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.9.0"}:
            scoped_batch_issues = int(
                await session.scalar(
                    text(
                        "SELECT count(*) FROM "
                        "pmmc_runtime.questioner_verification_batches batches "
                        "WHERE batches.run_id=:run_id AND "
                        "(batches.input_count <> 1 OR "
                        "batches.scope_qbank_id IS NULL OR NOT EXISTS ("
                        "SELECT 1 FROM "
                        "pmmc_runtime.questioner_candidate_verifications items "
                        "WHERE items.verification_batch_id="
                        "batches.verification_batch_id "
                        "AND items.run_id=batches.run_id "
                        "AND items.qbank_id=batches.scope_qbank_id))"
                    ),
                    {"run_id": run_id},
                )
                or 0
            )
            cross_scope_items = int(
                await session.scalar(
                    text(
                        "SELECT count(*) FROM "
                        "pmmc_runtime.questioner_candidate_verifications items "
                        "JOIN pmmc_runtime.questioner_verification_batches batches "
                        "ON batches.verification_batch_id="
                        "items.verification_batch_id "
                        "AND batches.run_id=items.run_id "
                        "WHERE items.run_id=:run_id AND "
                        "items.qbank_id <> batches.scope_qbank_id"
                    ),
                    {"run_id": run_id},
                )
                or 0
            )
            if scoped_batch_issues or cross_scope_items:
                issues.append("Questioner v3.2 verifier candidate scope is incomplete")
    native_output_issue_count = int(
        await session.scalar(
            select(func.count())
            .select_from(RuntimeStructuredOutputItemDiagnosticModel)
            .where(
                RuntimeStructuredOutputItemDiagnosticModel.run_id == run_id,
                RuntimeStructuredOutputItemDiagnosticModel.stage.in_(
                    (
                        "questioner",
                        "questioner_verifier",
                        "questioner_revision",
                        "planner",
                    )
                ),
                RuntimeStructuredOutputItemDiagnosticModel.status.in_(
                    ("invalid", "repaired", "conservative_reject")
                ),
            )
        )
        or 0
    )
    if require_native_structured_output and native_output_issue_count:
        issues.append("compiler provider output required parser repair or conservative fallback")
    index = await session.scalar(
        select(QBankIndexManifestModel).where(QBankIndexManifestModel.compiler_run_id == run_id)
    )
    collection_count = 0
    embedding_count = 0
    collection_identities: list[dict[str, object]] = []
    if index is None or index.status != EmbeddingCollectionStatus.READY.value:
        issues.append("Questioner run has no ready QBank index")
    else:
        collection_ids = [
            index.text_question_collection_id,
            index.text_paraphrase_collection_id,
            index.multimodal_question_collection_id,
            index.multimodal_paraphrase_collection_id,
        ]
        collection_rows = (
            (
                await session.execute(
                    text(
                        "SELECT collections.collection_id, collections.status, "
                        "collections.dimension, count(records.embedding_id) AS record_count "
                        "FROM pmmc_runtime.embedding_collections collections "
                        "LEFT JOIN pmmc_runtime.embeddings records "
                        "ON records.collection_id=collections.collection_id "
                        "WHERE collections.collection_id = "
                        "ANY(CAST(:collection_ids AS varchar[])) "
                        "GROUP BY collections.collection_id, collections.status, "
                        "collections.dimension"
                    ),
                    {"collection_ids": collection_ids},
                )
            )
            .mappings()
            .all()
        )
        collection_count = len(collection_rows)
        collection_identities = [
            {
                "collection_id": str(row["collection_id"]),
                "status": str(row["status"]),
                "dimension": int(row["dimension"]),
                "record_count": int(row["record_count"]),
            }
            for row in sorted(collection_rows, key=lambda row: str(row["collection_id"]))
        ]
        embedding_count = sum(int(row["record_count"]) for row in collection_rows)
        if collection_count != 4:
            issues.append("QBank index does not reference four distinct collections")
        if any(
            row["status"] != EmbeddingCollectionStatus.READY.value or int(row["dimension"]) != 4096
            for row in collection_rows
        ):
            issues.append("QBank index collection identity is invalid")
        expected_total = 2 * (index.eligible_question_count + index.paraphrase_count)
        if embedding_count != expected_total:
            issues.append("QBank index embedding count is incomplete")
    call_summary = (
        (
            await session.execute(
                text(
                    "SELECT count(*) FILTER (WHERE status='success') AS succeeded, "
                    "count(*) FILTER (WHERE status='success' AND runtime_id IS NOT NULL "
                    "AND runtime_service_id IS NOT NULL) AS managed "
                    "FROM pmmc_runtime.model_calls WHERE run_id=:run_id "
                    "AND stage IN ('questioner','embed')"
                ),
                {"run_id": run_id},
            )
        )
        .mappings()
        .one()
    )
    succeeded_calls = int(call_summary["succeeded"])
    managed_calls = int(call_summary["managed"])
    runtime_service_usage = {
        str(row["runtime_service_id"]): int(row["call_count"])
        for row in (
            await session.execute(
                text(
                    "SELECT runtime_service_id, count(*) AS call_count "
                    "FROM pmmc_runtime.model_calls WHERE run_id=:run_id "
                    "AND stage='questioner' AND status='success' "
                    "AND runtime_service_id IS NOT NULL "
                    "GROUP BY runtime_service_id ORDER BY runtime_service_id"
                ),
                {"run_id": run_id},
            )
        ).mappings()
    }
    if succeeded_calls < compiler.unit_count:
        issues.append("Questioner model-call lineage is incomplete")
    if require_managed_runtime and managed_calls != succeeded_calls:
        issues.append("Questioner model calls are missing managed-runtime lineage")
    failed_artifacts = int(
        await session.scalar(
            text(
                "SELECT count(*) FROM pmmc_runtime.run_artifacts "
                "WHERE run_id=:run_id AND state != 'ready'"
            ),
            {"run_id": run_id},
        )
        or 0
    )
    if failed_artifacts:
        issues.append("Questioner run has non-ready artifacts")
    artifact_summary = (
        (
            await session.execute(
                text(
                    "WITH refs AS ("
                    "SELECT model_call_id, 'input' AS role, input_asset_id AS artifact_id "
                    "FROM pmmc_runtime.model_calls WHERE run_id=:run_id "
                    "AND input_asset_id IS NOT NULL UNION ALL "
                    "SELECT model_call_id, 'output' AS role, output_asset_id AS artifact_id "
                    "FROM pmmc_runtime.model_calls WHERE run_id=:run_id "
                    "AND output_asset_id IS NOT NULL) "
                    "SELECT count(*) AS referenced, "
                    "count(artifacts.artifact_id) FILTER (WHERE artifacts.state='ready') AS ready "
                    "FROM refs LEFT JOIN pmmc_runtime.run_artifacts artifacts "
                    "ON artifacts.artifact_id=refs.artifact_id AND artifacts.run_id=:run_id"
                ),
                {"run_id": run_id},
            )
        )
        .mappings()
        .one()
    )
    referenced_artifact_count = int(artifact_summary["referenced"])
    ready_referenced_artifact_count = int(artifact_summary["ready"])
    if referenced_artifact_count != ready_referenced_artifact_count:
        issues.append("Questioner model calls reference missing or non-ready artifacts")
    return {
        **report,
        "generation_count": generation_count,
        "raw_candidate_count": raw_candidate_count,
        "schema_valid_candidate_count": schema_valid_candidate_count,
        "schema_invalid_candidate_count": schema_invalid_candidate_count,
        "persisted_candidate_version_count": question_count,
        "generated_candidate_count": generated_question_count,
        "eligible_candidate_count": eligible_candidate_count,
        "gate_rejected_candidate_count": gate_rejected_candidate_count,
        "terminal_rejected_rank_count": max(0, raw_candidate_count - eligible_candidate_count),
        "total_rejected_candidate_count": (
            max(0, raw_candidate_count - eligible_candidate_count)
            if is_v3
            else schema_invalid_candidate_count + gate_rejected_candidate_count
        ),
        "duplicate_candidate_count": duplicate_candidate_count,
        "schema_valid_rate": (
            schema_valid_candidate_count / raw_candidate_count if raw_candidate_count else None
        ),
        "gate_pass_rate_among_schema_valid": (
            eligible_candidate_count / schema_valid_candidate_count
            if schema_valid_candidate_count
            else None
        ),
        "overall_eligible_rate": (
            eligible_candidate_count / raw_candidate_count if raw_candidate_count else None
        ),
        "zero_valid_candidate_unit_count": zero_valid_candidate_unit_count,
        "zero_valid_candidate_unit_rate": (
            zero_valid_candidate_unit_count / generation_count if generation_count else None
        ),
        "schema_error_type_counts": dict(sorted(diagnostic_type_counts.items())),
        "schema_error_location_counts": dict(sorted(diagnostic_location_counts.items())),
        "qbank_index_id": index.qbank_index_id if index is not None else None,
        "qbank_collection_count": collection_count,
        "qbank_collections": collection_identities,
        "qbank_embedding_count": embedding_count,
        "semantic_neighbor_count": semantic_neighbor_count,
        "opportunity_set_count": opportunity_set_count,
        "opportunity_count": opportunity_count,
        "verification_batch_count": verification_batch_count,
        "verification_candidate_count": verification_candidate_count,
        "revised_candidate_count": revised_candidate_count,
        "revision_rescued_count": revision_rescued_count,
        "successful_model_call_count": succeeded_calls,
        "managed_model_call_count": managed_calls,
        "questioner_runtime_service_usage": runtime_service_usage,
        "referenced_model_artifact_count": referenced_artifact_count,
        "ready_referenced_model_artifact_count": ready_referenced_artifact_count,
        "require_managed_runtime": require_managed_runtime,
        "require_schema_conformance": require_schema_conformance,
        "require_native_structured_output": require_native_structured_output,
        "native_structured_output_issue_count": native_output_issue_count,
        "valid": not issues,
        "issues": issues,
    }
