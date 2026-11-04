"""Async Postgres repositories for PMMC compiler facts and operator data."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

from sqlalchemy import and_, func, insert, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, aliased

from pmmc.compiler.execution import program_evidence_quotas
from pmmc.compiler.ids import (
    COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1,
    compilation_unit_content_id,
    compilation_unit_identity,
    compilation_unit_selection_lineage,
    feedback_content_hash,
    memory_program_content_hash,
    qbank_decision_content_hash,
    question_content_hash,
)
from pmmc.evidence_lineage import is_memgallery_character_profile, memgallery_profile_group
from pmmc.infrastructure.database.compiler_mappers import (
    compilation_stage_result_from_row,
    compilation_stage_result_to_row,
    compilation_unit_from_rows,
    compilation_unit_to_rows,
    compiler_run_from_row,
    compiler_run_to_row,
    decision_from_row,
    decision_to_row,
    evidence_ref_to_row,
    execution_evidence_from_row,
    execution_evidence_to_row,
    execution_from_row,
    execution_step_from_row,
    execution_step_to_row,
    execution_to_row,
    feedback_from_row,
    feedback_to_row,
    memory_program_from_rows,
    memory_program_to_rows,
    program_expected_evidence_to_row,
    question_from_rows,
    question_to_rows,
    routing_item_from_row,
)
from pmmc.infrastructure.database.compiler_models import (
    CompilerRunModel,
    CompilerUnitMemoryModel,
    CompilerUnitModel,
    CompilerUnitStageResultModel,
    DoubterFeedbackModel,
    MemoryProgramModel,
    PlannerCandidateOutcomeModel,
    ProgramExecutionEvidenceRefModel,
    ProgramExecutionModel,
    ProgramExecutionStepModel,
    ProgramExpectedEvidenceRefModel,
    ProgramStepModel,
    QBankDecisionModel,
    QBankEvidenceRefModel,
    QBankParaphraseModel,
    QBankQuestionModel,
)
from pmmc.infrastructure.database.doubter_models import (
    DoubterAssessmentModel,
    PlannerRevisionLineageModel,
)
from pmmc.infrastructure.database.errors import (
    InvalidStateTransitionError,
    PersistenceConflictError,
)
from pmmc.infrastructure.database.experiment_models import (
    EmbeddingCollectionModel,
    EmbeddingModel,
    ExperimentJobModel,
    RunQueryModel,
    RunStateModel,
    RuntimeModelCallModel,
)
from pmmc.infrastructure.database.models import (
    BenchmarkQueryModel,
    BenchmarkSessionModel,
    ImageModel,
    MemoryUnitImageModel,
    MemoryUnitModel,
    QueryImageModel,
    RunManifestModel,
)
from pmmc.memory.operators import ProgramExecutionResult
from pmmc.schemas import (
    CompilationStage,
    CompilationStageResult,
    CompilationUnit,
    CompilerRunSpec,
    DatasetName,
    DoubterFeedback,
    EmbeddingCollectionStatus,
    EmbeddingSourceType,
    EvidenceType,
    MemoryProgram,
    OperatorEvidence,
    OperatorExecutionContext,
    ProgramExecutionEvidenceRef,
    ProgramStep,
    QBankDecision,
    QBankEvidenceRole,
    QBankRoutingItem,
    QuestionCandidate,
    RankingScore,
    ScoreDirection,
    VerifiedQBankRoutingItemV2,
    VerifiedQBankRoutingItemV3,
)

ModelT = TypeVar("ModelT", bound=DeclarativeBase)

_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_FTS_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def classify_step7b_lexical_terms(
    query_text: str, *, limit: int = 12
) -> tuple[tuple[str, str], ...]:
    """Extract a bounded, injection-safe OR-FTS query without storing source text."""

    if limit <= 0:
        raise ValueError("lexical term limit must be positive")
    raw = _FTS_TOKEN_RE.findall(query_text)
    candidates: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for token in raw:
        normalized = token.casefold().replace("'", "").replace("-", "")
        if not normalized or normalized in _FTS_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        special = token[:1].isupper() or any(character.isdigit() for character in token)
        kind = (
            "date"
            if any(separator in token for separator in ("-", "/"))
            and any(character.isdigit() for character in token)
            else "number"
            if token.isdigit()
            else "entity"
            if token[:1].isupper()
            else "lexeme"
        )
        candidates.append((0 if special else 1, -len(normalized), normalized, kind))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple((item[2], item[3]) for item in candidates[:limit])


def extract_step7b_lexical_terms(query_text: str, *, limit: int = 12) -> tuple[str, ...]:
    return tuple(term for term, _kind in classify_step7b_lexical_terms(query_text, limit=limit))


def extract_step7b_exact_boost_terms(query_text: str, *, limit: int = 12) -> tuple[str, ...]:
    """Return only classified entity/number/date terms eligible for exact boost."""

    return tuple(
        term
        for term, term_kind in classify_step7b_lexical_terms(query_text, limit=limit)
        if term_kind in {"entity", "number", "date"}
    )


def build_step7b_or_tsquery(query_text: str, *, limit: int = 12) -> tuple[str, tuple[str, ...]]:
    terms = extract_step7b_lexical_terms(query_text, limit=limit)
    if not terms:
        return "", ()
    return " | ".join(terms), terms


QUESTIONER_STAGE_ACCOUNTING_SCHEMA_VERSIONS = frozenset(
    {"0.2.0", "0.3.0", "0.4.0", "0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.9.0"}
)

_COMPILER_VERIFICATION_ISSUE_MARKERS: tuple[tuple[str, str], ...] = (
    ("compiler unit count differs", "compiler_unit_denominator"),
    ("compiler unit ordinals", "compiler_unit_ordinals"),
    ("unit membership ordinals", "unit_membership_ordinals"),
    ("unit has no anchor memory", "unit_anchor_missing"),
    ("exactly one succeeded job", "compile_job_denominator"),
    ("unit is missing stages", "unit_stage_missing"),
    ("exactly one successful Questioner generation", "questioner_generation_denominator"),
    ("Questioner stage accounting", "questioner_stage_accounting"),
    ("planner stage accounting", "planner_stage_accounting"),
    ("dry_run stage accounting", "dry_run_stage_accounting"),
    ("stage produced_count differs", "stage_fact_accounting"),
    ("unit support/link counts", "unit_support_lineage"),
    ("unit has an ungated question", "questioner_gate_missing"),
    ("opportunity", "questioner_opportunity_lineage"),
    ("lacks initial semantic verification", "questioner_initial_verification_missing"),
    ("has invalid candidate versions", "questioner_candidate_versions"),
    ("has invalid revision lineage", "questioner_revision_lineage"),
    ("has multiple terminal candidates", "questioner_multiple_terminal_candidates"),
    ("generation rank", "questioner_candidate_lineage"),
    ("terminal v3 generation", "questioner_generation_terminal"),
    ("terminal generation rejection", "questioner_generation_terminal"),
    ("candidate verifier", "questioner_verifier_lineage"),
    ("semantic verification", "questioner_verifier_lineage"),
    ("eligible v3 candidate", "questioner_verifier_lineage"),
    ("candidate supersession", "questioner_revision_lineage"),
    ("revised candidates", "questioner_revision_lineage"),
    ("candidate IDs are duplicated", "questioner_candidate_identity"),
    ("candidate generation or opportunity", "questioner_candidate_lineage"),
    ("every v3 candidate", "questioner_gate_denominator"),
    ("verification-batch accounting", "questioner_verifier_accounting"),
    ("verification references a missing batch", "questioner_verifier_lineage"),
    ("no unique succeeded planner stage job", "planner_stage_job"),
    ("no unique succeeded dry_run stage job", "dry_run_stage_job"),
    ("Planner terminal outcomes", "planner_terminal_denominator"),
    ("Questioner compiler identity", "questioner_identity"),
    ("ready QBank index", "qbank_index_denominator"),
    ("QBank index references", "qbank_index_lineage"),
    ("Planner compiler identity", "planner_identity"),
    ("ready memory operator index", "memory_index_denominator"),
    ("memory operator index references", "memory_index_lineage"),
    ("query-vector denominator", "program_query_vector_denominator"),
    ("program terminal outcome", "program_terminal_outcome"),
    ("program revision lineage", "program_revision_lineage"),
    ("program validation facts", "program_validation"),
    ("lineage-rejected program facts", "program_lineage"),
    ("vector-unavailable program", "program_query_vector_lineage"),
    ("program dry-run assessment", "program_dry_run_denominator"),
    ("Doubter compiler identity", "doubter_identity"),
    ("freeze active r2", "doubter_round_identity"),
    ("immutable R0 input", "r0_input_manifest"),
    ("round projections are incomplete", "round_projection_denominator"),
    ("round projection denominator", "round_projection_denominator"),
    ("maximum-round projection", "round_projection_terminal"),
    ("QBank decision denominator", "qbank_decision_denominator"),
    ("routing projection is missing", "routing_projection_missing"),
    ("routing projection conflicts", "routing_projection_lineage"),
    ("Doubter candidate terminal outcomes", "doubter_terminal_denominator"),
    ("unfinished candidate stage jobs", "candidate_stage_jobs"),
    ("successful program execution", "program_execution_steps"),
)


def summarize_compiler_verification_issues(issues: Sequence[object]) -> dict[str, int]:
    """Return identifier-free issue categories suitable for shared runtime logs."""

    counts: Counter[str] = Counter()
    for raw_issue in issues:
        issue = str(raw_issue)
        code = next(
            (
                candidate
                for marker, candidate in _COMPILER_VERIFICATION_ISSUE_MARKERS
                if marker in issue
            ),
            "unclassified_compiler_invariant",
        )
        counts[code] += 1
    return dict(sorted(counts.items()))


def compiler_doubter_terminal_accounting(
    eligible_ids: set[str],
    terminal_ids: set[str],
    programs: Sequence[MemoryProgramModel],
    outcomes: Sequence[PlannerCandidateOutcomeModel],
) -> bool:
    """Partition the frozen R0 denominator, without fabricating Doubter assessments."""
    r0 = [item for item in programs if item.round_id == 0]
    by_candidate = {item.qbank_id: item.plan_id for item in r0}
    by_outcome = {item.qbank_id: item for item in outcomes}
    if (
        len(by_candidate) != len(r0)
        or len(by_outcome) != len(outcomes)
        or set(by_outcome) != eligible_ids
        or not set(by_candidate) <= eligible_ids
        or terminal_ids != set(by_candidate)
    ):
        return False
    for qbank_id, outcome in by_outcome.items():
        if qbank_id not in by_candidate:
            # R0 already verifies these scientific rejections and their source facts.
            if outcome.plan_id is not None or outcome.status not in {
                "parse_failed", "static_rejected", "policy_rejected",
            }:
                return False
        elif outcome.plan_id != by_candidate[qbank_id] or outcome.status not in {
            "program_ready", "program_ready_with_fallback", "program_ready_with_abstain",
            "dry_run_insufficient", "execution_failed",
        }:
            return False
    return True


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
        raise RuntimeError(f"failed to read inserted {model.__tablename__} row")
    return inserted


class PostgresCompilerRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, value: CompilerRunSpec) -> CompilerRunSpec:
        manifest = await self.session.get(RunManifestModel, value.run_id)
        if manifest is None:
            raise KeyError(f"compiler run manifest does not exist: {value.run_id}")
        payload = manifest.manifest_json
        expected = {
            "compiler_schema_version": value.compiler_schema_version,
            "operator_set_version": value.operator_set_version,
            "operator_registry_hash": value.operator_registry_hash,
            "compiler_config_hash": value.compiler_config_hash,
        }
        if (
            manifest.method != "pmmc_compiler"
            or payload.get("dataset_snapshot_ids") != [value.snapshot_id]
            or any(payload.get(name) != expected_value for name, expected_value in expected.items())
        ):
            raise ValueError("compiler run does not match its immutable run manifest")
        stored = await _insert_idempotent(
            self.session,
            CompilerRunModel,
            compiler_run_to_row(value),
            value.run_id,
        )
        return compiler_run_from_row(_model_mapping(stored))

    async def get(self, run_id: str) -> CompilerRunSpec | None:
        stored = await self.session.get(CompilerRunModel, run_id)
        return compiler_run_from_row(_model_mapping(stored)) if stored is not None else None


class PostgresCompilerUnitRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, value: CompilationUnit) -> CompilationUnit:
        compiler_run = await self.session.get(CompilerRunModel, value.run_id)
        if compiler_run is None:
            raise KeyError(f"compiler run does not exist: {value.run_id}")
        if compiler_run.snapshot_id != value.snapshot_id:
            raise ValueError("compiler unit snapshot does not match its compiler run")

        selection_lineage: list[dict[str, object]] | None = None
        if compiler_run.compiler_schema_version in {
            "0.3.0",
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
            "0.9.0",
        }:
            selection_lineage = compilation_unit_selection_lineage(value.memories)
            declared_lineage = value.metadata.get("relation_selection")
            if not isinstance(declared_lineage, list):
                raise ValueError("v3 compiler unit requires relation_selection metadata")
            if declared_lineage != selection_lineage:
                raise ValueError("v3 compiler unit relation_selection is not canonical")
        identity_protocol = value.metadata.get("compiler_unit_identity_protocol")
        if identity_protocol not in {None, COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1}:
            raise ValueError("compiler unit uses an unsupported identity protocol")
        run_scoped_identity = (
            compiler_run.compiler_schema_version in {"0.7.0", "0.8.0", "0.9.0"}
            and identity_protocol == COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1
        )
        expected_id, expected_hash = compilation_unit_identity(
            snapshot_id=value.snapshot_id,
            conversation_id=value.conversation_id,
            cutoff_session_id=value.available_after_session,
            builder_version=value.builder_version,
            memories=value.memories,
            selection_lineage=selection_lineage,
            run_id=value.run_id if run_scoped_identity else None,
        )
        if run_scoped_identity and value.metadata.get(
            "compiler_unit_content_id"
        ) != compilation_unit_content_id(expected_hash):
            raise ValueError("run-scoped compiler unit has no canonical content identity")
        if value.compiler_unit_id != expected_id or value.input_hash != expected_hash:
            raise ValueError("compiler unit ID or input hash is not canonical")
        rows = compilation_unit_to_rows(value)
        await _insert_idempotent(
            self.session,
            CompilerUnitModel,
            rows["unit"][0],
            value.compiler_unit_id,
        )
        for row in rows["memories"]:
            await _insert_idempotent(
                self.session,
                CompilerUnitMemoryModel,
                row,
                row["compiler_unit_id"],
                row["memory_id"],
            )
        stored = await self.get(value.compiler_unit_id)
        if stored is None:
            raise RuntimeError("failed to read inserted compiler unit")
        return stored

    async def get(self, compiler_unit_id: str) -> CompilationUnit | None:
        stored = await self.session.get(CompilerUnitModel, compiler_unit_id)
        if stored is None:
            return None
        members = list(
            (
                await self.session.execute(
                    select(CompilerUnitMemoryModel)
                    .where(CompilerUnitMemoryModel.compiler_unit_id == compiler_unit_id)
                    .order_by(CompilerUnitMemoryModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return compilation_unit_from_rows(
            _model_mapping(stored), [_model_mapping(value) for value in members]
        )

    async def list_for_run(self, run_id: str) -> list[CompilationUnit]:
        unit_ids = list(
            await self.session.scalars(
                select(CompilerUnitModel.compiler_unit_id)
                .where(CompilerUnitModel.run_id == run_id)
                .order_by(CompilerUnitModel.ordinal)
            )
        )
        result: list[CompilationUnit] = []
        for unit_id in unit_ids:
            value = await self.get(unit_id)
            if value is None:
                raise RuntimeError("compiler unit disappeared while listing a run")
            result.append(value)
        return result

    async def add_stage_result(self, value: CompilationStageResult) -> CompilationStageResult:
        stored = await _insert_idempotent(
            self.session,
            CompilerUnitStageResultModel,
            compilation_stage_result_to_row(value),
            value.stage_result_id,
        )
        return compilation_stage_result_from_row(_model_mapping(stored))

    async def get_stage_result(
        self,
        compiler_unit_id: str,
        stage: CompilationStage,
    ) -> CompilationStageResult | None:
        stored = await self.session.scalar(
            select(CompilerUnitStageResultModel)
            .where(
                CompilerUnitStageResultModel.compiler_unit_id == compiler_unit_id,
                CompilerUnitStageResultModel.stage == stage.value,
            )
            .order_by(CompilerUnitStageResultModel.attempt_number.desc())
            .limit(1)
        )
        return (
            compilation_stage_result_from_row(_model_mapping(stored))
            if stored is not None
            else None
        )


class _TypedEvidenceResolver:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def resolve_many(
        self,
        *,
        compiler_unit_id: str,
        evidence_ids: list[str],
    ) -> list[OperatorEvidence]:
        return [
            await self.resolve(compiler_unit_id=compiler_unit_id, evidence_id=value)
            for value in evidence_ids
        ]

    async def resolve(
        self,
        *,
        compiler_unit_id: str,
        evidence_id: str,
    ) -> OperatorEvidence:
        unit = await self.session.get(CompilerUnitModel, compiler_unit_id)
        if unit is None:
            raise KeyError(f"compiler unit does not exist: {compiler_unit_id}")
        cutoff_order = await self.session.scalar(
            select(BenchmarkSessionModel.session_order).where(
                BenchmarkSessionModel.snapshot_id == unit.snapshot_id,
                BenchmarkSessionModel.conversation_id == unit.conversation_id,
                BenchmarkSessionModel.session_id == unit.available_after_session,
            )
        )
        if cutoff_order is None:
            raise KeyError("compiler unit cutoff session does not exist")
        matches: list[OperatorEvidence] = []
        available = aliased(BenchmarkSessionModel)
        memory = await self.session.scalar(
            select(MemoryUnitModel)
            .join(
                available,
                and_(
                    available.snapshot_id == MemoryUnitModel.snapshot_id,
                    available.conversation_id == MemoryUnitModel.conversation_id,
                    available.session_id == MemoryUnitModel.available_after_session,
                ),
            )
            .where(
                MemoryUnitModel.snapshot_id == unit.snapshot_id,
                MemoryUnitModel.conversation_id == unit.conversation_id,
                MemoryUnitModel.memory_id == evidence_id,
                available.session_order <= cutoff_order,
            )
        )
        if memory is not None:
            matches.append(
                OperatorEvidence(
                    evidence_id=evidence_id,
                    evidence_type=EvidenceType.MEMORY,
                    memory_id=evidence_id,
                )
            )
        image = await self.session.scalar(
            select(ImageModel)
            .join(
                MemoryUnitImageModel,
                and_(
                    MemoryUnitImageModel.snapshot_id == ImageModel.snapshot_id,
                    MemoryUnitImageModel.image_id == ImageModel.image_id,
                ),
            )
            .join(
                MemoryUnitModel,
                and_(
                    MemoryUnitModel.snapshot_id == MemoryUnitImageModel.snapshot_id,
                    MemoryUnitModel.memory_id == MemoryUnitImageModel.memory_id,
                ),
            )
            .join(
                available,
                and_(
                    available.snapshot_id == MemoryUnitModel.snapshot_id,
                    available.conversation_id == MemoryUnitModel.conversation_id,
                    available.session_id == MemoryUnitModel.available_after_session,
                ),
            )
            .where(
                ImageModel.snapshot_id == unit.snapshot_id,
                ImageModel.image_id == evidence_id,
                MemoryUnitModel.conversation_id == unit.conversation_id,
                available.session_order <= cutoff_order,
            )
            .limit(1)
        )
        if image is not None:
            matches.append(
                OperatorEvidence(
                    evidence_id=evidence_id,
                    evidence_type=EvidenceType.IMAGE,
                    image_id=evidence_id,
                )
            )
        session = await self.session.scalar(
            select(BenchmarkSessionModel).where(
                BenchmarkSessionModel.snapshot_id == unit.snapshot_id,
                BenchmarkSessionModel.conversation_id == unit.conversation_id,
                BenchmarkSessionModel.session_id == evidence_id,
                BenchmarkSessionModel.session_order <= cutoff_order,
            )
        )
        if session is not None:
            matches.append(
                OperatorEvidence(
                    evidence_id=evidence_id,
                    evidence_type=EvidenceType.SESSION,
                    session_id=evidence_id,
                )
            )
        if not matches:
            raise ValueError(f"evidence is absent or not visible at cutoff: {evidence_id}")
        if len(matches) != 1:
            raise ValueError(f"evidence ID is ambiguous across canonical types: {evidence_id}")
        return matches[0]


class PostgresQBankCompilerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.resolver = _TypedEvidenceResolver(session)

    async def add_question(
        self,
        value: QuestionCandidate,
        *,
        expected_evidence: list[OperatorEvidence] | None = None,
    ) -> QuestionCandidate:
        if value.compiler_unit_id is None or value.snapshot_id is None:
            raise ValueError("qbank persistence requires compiler scope")
        if value.question_hash != question_content_hash(value):
            raise ValueError("qbank question_hash is not canonical")
        if (
            value.candidate_content_hash is not None
            and value.candidate_content_hash != question_content_hash(value)
        ):
            raise ValueError("qbank candidate_content_hash is not canonical")
        if expected_evidence is None:
            expected_evidence = await self.resolver.resolve_many(
                compiler_unit_id=value.compiler_unit_id,
                evidence_ids=value.evidence_ids,
            )
        if [item.evidence_id for item in expected_evidence] != value.evidence_ids:
            raise ValueError("typed expected evidence must preserve candidate evidence order")
        rows = question_to_rows(value)
        await _insert_idempotent(
            self.session,
            QBankQuestionModel,
            rows["question"][0],
            value.qbank_id,
        )
        for row in rows["paraphrases"]:
            await _insert_idempotent(self.session, QBankParaphraseModel, row, row["paraphrase_id"])
        evidence_rows = list(rows["source"])
        evidence_rows.extend(
            evidence_ref_to_row(
                run_id=value.run_id,
                snapshot_id=value.snapshot_id,
                compiler_unit_id=value.compiler_unit_id,
                owner_id=value.qbank_id,
                role=QBankEvidenceRole.EXPECTED,
                ordinal=ordinal,
                evidence=evidence,
            )
            for ordinal, evidence in enumerate(expected_evidence)
        )
        for row in evidence_rows:
            await _insert_idempotent(
                self.session, QBankEvidenceRefModel, row, row["evidence_ref_id"]
            )
        stored = await self.get_question(value.qbank_id)
        if stored is None:
            raise RuntimeError("failed to read inserted qbank question")
        return stored

    async def get_question(self, qbank_id: str) -> QuestionCandidate | None:
        question = await self.session.get(QBankQuestionModel, qbank_id)
        if question is None:
            return None
        paraphrases = list(
            (
                await self.session.execute(
                    select(QBankParaphraseModel)
                    .where(QBankParaphraseModel.qbank_id == qbank_id)
                    .order_by(QBankParaphraseModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        evidence = list(
            (
                await self.session.execute(
                    select(QBankEvidenceRefModel)
                    .where(QBankEvidenceRefModel.qbank_id == qbank_id)
                    .order_by(QBankEvidenceRefModel.role, QBankEvidenceRefModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return question_from_rows(
            _model_mapping(question),
            [_model_mapping(value) for value in paraphrases],
            [_model_mapping(value) for value in evidence],
        )

    async def add_feedback(self, value: DoubterFeedback) -> DoubterFeedback:
        if value.feedback_hash != feedback_content_hash(value):
            raise ValueError("doubter feedback_hash is not canonical")
        stored = await _insert_idempotent(
            self.session,
            DoubterFeedbackModel,
            feedback_to_row(value),
            value.feedback_id,
        )
        return feedback_from_row(_model_mapping(stored))

    async def add_decision(self, value: QBankDecision) -> QBankDecision:
        if value.decision_hash != qbank_decision_content_hash(value):
            raise ValueError("qbank decision_hash is not canonical")
        stored = await _insert_idempotent(
            self.session,
            QBankDecisionModel,
            decision_to_row(value),
            value.decision_id,
        )
        return decision_from_row(_model_mapping(stored))

    async def get_decision(self, qbank_id: str) -> QBankDecision | None:
        stored = await self.session.scalar(
            select(QBankDecisionModel).where(QBankDecisionModel.qbank_id == qbank_id)
        )
        return decision_from_row(_model_mapping(stored)) if stored is not None else None

    async def list_decisions(self, run_id: str) -> list[QBankDecision]:
        rows = (
            (
                await self.session.execute(
                    select(QBankDecisionModel)
                    .where(QBankDecisionModel.run_id == run_id)
                    .order_by(QBankDecisionModel.qbank_id)
                )
            )
            .scalars()
            .all()
        )
        return [decision_from_row(_model_mapping(item)) for item in rows]


class PostgresMemoryProgramRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.resolver = _TypedEvidenceResolver(session)

    async def add(
        self,
        value: MemoryProgram,
        *,
        expected_evidence: list[OperatorEvidence] | None = None,
    ) -> MemoryProgram:
        if value.compiler_unit_id is None:
            raise ValueError("program persistence requires compiler scope")
        if value.program_hash != memory_program_content_hash(value):
            raise ValueError("memory program_hash is not canonical")
        if expected_evidence is None:
            expected_evidence = await self.resolver.resolve_many(
                compiler_unit_id=value.compiler_unit_id,
                evidence_ids=value.expected_evidence_ids,
            )
        if [item.evidence_id for item in expected_evidence] != value.expected_evidence_ids:
            raise ValueError("typed program evidence must preserve expected evidence order")
        evidence_by_id = {item.evidence_id: item for item in expected_evidence}
        rows = memory_program_to_rows(value)
        await _insert_idempotent(
            self.session,
            MemoryProgramModel,
            rows["program"][0],
            value.plan_id,
        )
        for row in rows["steps"]:
            await _insert_idempotent(
                self.session,
                ProgramStepModel,
                row,
                row["plan_id"],
                row["step_id"],
            )
        expected_rows = [
            program_expected_evidence_to_row(program=value, evidence=item, ordinal=ordinal)
            for ordinal, item in enumerate(expected_evidence)
        ]
        for step in value.operator_sequence:
            for ordinal, evidence_id in enumerate(step.expected_evidence_ids):
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    evidence = await self.resolver.resolve(
                        compiler_unit_id=value.compiler_unit_id,
                        evidence_id=evidence_id,
                    )
                expected_rows.append(
                    program_expected_evidence_to_row(
                        program=value,
                        evidence=evidence,
                        ordinal=ordinal,
                        step_id=step.step_id,
                    )
                )
        for row in expected_rows:
            await _insert_idempotent(
                self.session,
                ProgramExpectedEvidenceRefModel,
                row,
                row["evidence_ref_id"],
            )
        stored = await self.get(value.plan_id)
        if stored is None:
            raise RuntimeError("failed to read inserted memory program")
        return stored

    async def get(self, plan_id: str) -> MemoryProgram | None:
        program = await self.session.get(MemoryProgramModel, plan_id)
        if program is None:
            return None
        steps = list(
            (
                await self.session.execute(
                    select(ProgramStepModel)
                    .where(ProgramStepModel.plan_id == plan_id)
                    .order_by(ProgramStepModel.step_id)
                )
            )
            .scalars()
            .all()
        )
        evidence = list(
            (
                await self.session.execute(
                    select(ProgramExpectedEvidenceRefModel)
                    .where(ProgramExpectedEvidenceRefModel.plan_id == plan_id)
                    .order_by(
                        ProgramExpectedEvidenceRefModel.step_id.nullsfirst(),
                        ProgramExpectedEvidenceRefModel.ordinal,
                    )
                )
            )
            .scalars()
            .all()
        )
        return memory_program_from_rows(
            _model_mapping(program),
            [_model_mapping(value) for value in steps],
            [_model_mapping(value) for value in evidence],
        )

    async def get_for_candidate_round(
        self, run_id: str, qbank_id: str, round_id: int
    ) -> MemoryProgram | None:
        plan_id = (
            await self.session.execute(
                select(MemoryProgramModel.plan_id).where(
                    MemoryProgramModel.run_id == run_id,
                    MemoryProgramModel.qbank_id == qbank_id,
                    MemoryProgramModel.round_id == round_id,
                )
            )
        ).scalar_one_or_none()
        return None if plan_id is None else await self.get(plan_id)

    async def list_for_run(self, run_id: str) -> list[MemoryProgram]:
        programs = list(
            await self.session.scalars(
                select(MemoryProgramModel)
                .where(MemoryProgramModel.run_id == run_id)
                .order_by(MemoryProgramModel.compiler_unit_id, MemoryProgramModel.qbank_id)
            )
        )
        if not programs:
            return []
        # Fetch children once per run, not three round trips for every program.
        steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
        evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for model, grouped, ordering in (
            (ProgramStepModel, steps, (ProgramStepModel.step_id,)),
            (
                ProgramExpectedEvidenceRefModel,
                evidence,
                (
                    ProgramExpectedEvidenceRefModel.step_id.nullsfirst(),
                    ProgramExpectedEvidenceRefModel.ordinal,
                ),
            ),
        ):
            for row in await self.session.scalars(
                select(model)
                .join(MemoryProgramModel, model.plan_id == MemoryProgramModel.plan_id)
                .where(MemoryProgramModel.run_id == run_id)
                .order_by(*ordering)
            ):
                values = _model_mapping(row)
                grouped[str(values["plan_id"])].append(values)
        return [
            memory_program_from_rows(
                _model_mapping(program), steps[program.plan_id], evidence[program.plan_id]
            )
            for program in programs
        ]


class PostgresProgramExecutionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, value: ProgramExecutionResult) -> ProgramExecutionResult:
        await _insert_idempotent(
            self.session,
            ProgramExecutionModel,
            execution_to_row(value.trace),
            value.trace.execution_id,
        )
        for step in value.steps:
            await _insert_idempotent(
                self.session,
                ProgramExecutionStepModel,
                execution_step_to_row(step),
                value.trace.execution_id,
                step.step_id,
            )
        for evidence in value.evidence:
            await _insert_idempotent(
                self.session,
                ProgramExecutionEvidenceRefModel,
                execution_evidence_to_row(evidence, run_id=value.trace.run_id),
                evidence.evidence_ref_id,
            )
        stored = await self.get(value.trace.execution_id)
        if stored is None:
            raise RuntimeError("failed to read inserted program execution")
        return stored

    async def get(self, execution_id: str) -> ProgramExecutionResult | None:
        trace = await self.session.get(ProgramExecutionModel, execution_id)
        if trace is None:
            return None
        steps = list(
            (
                await self.session.execute(
                    select(ProgramExecutionStepModel)
                    .where(ProgramExecutionStepModel.execution_id == execution_id)
                    .order_by(ProgramExecutionStepModel.step_id)
                )
            )
            .scalars()
            .all()
        )
        evidence = list(
            (
                await self.session.execute(
                    select(ProgramExecutionEvidenceRefModel)
                    .where(ProgramExecutionEvidenceRefModel.execution_id == execution_id)
                    .order_by(ProgramExecutionEvidenceRefModel.rank)
                )
            )
            .scalars()
            .all()
        )
        return ProgramExecutionResult(
            execution_from_row(_model_mapping(trace)),
            tuple(execution_step_from_row(_model_mapping(value)) for value in steps),
            tuple(execution_evidence_from_row(_model_mapping(value)) for value in evidence),
        )


def _versioned_routing_sql(statement: str, version: str) -> str:
    """Dispatch only code-owned view identifiers; all query scope stays parameterized."""
    if version not in {"v3", "v4"}:
        raise ValueError("unsupported verified routing SQL version")
    if version == "v4":
        return (
            statement.replace("verified_qbank_routing_v3", "verified_qbank_routing_v4")
            .replace("pmmc_runtime.qbank_questions", "pmmc_runtime.runtime_qbank_questions_v1")
            .replace("pmmc_runtime.qbank_paraphrases", "pmmc_runtime.runtime_qbank_paraphrases_v1")
            .replace(
                "verified-qbank-routing-v3",
                "verified-qbank-routing-v4",
            )
            .replace(
                "routing.program_hash, routing.operator_set_version,",
                "routing.program_hash, routing.operator_set_version, "
                "routing.projection_round, routing.program_round,",
            )
        )
    return statement


class PostgresQBankRoutingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_runtime_projection_v3(self, run_id: str) -> dict[str, Any] | None:
        row = (await self.session.execute(text(
            "SELECT routing_projection_id, run_id, snapshot_id, active_round, max_round, "
            "content_hash FROM pmmc_runtime.query_runtime_projections_v3 WHERE run_id=:run_id"
        ), {"run_id": run_id})).mappings().one_or_none()
        return dict(row) if row is not None else None

    async def list_runtime_program_vector_refs_v3(self, run_id: str) -> list[dict[str, Any]]:
        rows = (await self.session.execute(text(
            "SELECT qbank_id, channel, embedding_id "
            "FROM pmmc_runtime.query_runtime_program_vectors_v3 WHERE run_id=:run_id"
        ), {"run_id": run_id})).mappings().all()
        return [dict(row) for row in rows]

    async def list_verified_v4(self, **scope: Any) -> list[Any]:
        return await self.list_verified_v3(**scope, routing_version="v4")

    async def search_lexical_verified_v4(self, **scope: Any) -> list[tuple[str, float]]:
        return await self.search_lexical_verified_v3(**scope, routing_version="v4")

    async def search_embedding_verified_v4(self, **scope: Any) -> list[tuple[str, float]]:
        return await self.search_embedding_verified_v3(**scope, routing_version="v4")

    async def list_for_conversation(
        self,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
    ) -> list[QBankRoutingItem]:
        rows = (
            await self.session.execute(
                text(
                    """SELECT routing.*
                    FROM pmmc_runtime.qbank_routing_items routing
                    JOIN pmmc_runtime.benchmark_sessions available
                      ON available.snapshot_id = routing.snapshot_id
                     AND available.conversation_id = routing.conversation_id
                     AND available.session_id = routing.available_after_session
                    JOIN pmmc_runtime.benchmark_sessions cutoff
                      ON cutoff.snapshot_id = routing.snapshot_id
                     AND cutoff.conversation_id = routing.conversation_id
                     AND cutoff.session_id = :cutoff_session_id
                    WHERE routing.snapshot_id = :snapshot_id
                      AND routing.conversation_id = :conversation_id
                      AND available.session_order <= cutoff.session_order
                    ORDER BY routing.qbank_id"""
                ),
                {
                    "snapshot_id": snapshot_id,
                    "conversation_id": conversation_id,
                    "cutoff_session_id": cutoff_session_id,
                },
            )
        ).mappings()
        return [routing_item_from_row(dict(row)) for row in rows]

    async def list_verified_v2(
        self,
        *,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        question_budget: int,
        round_id: int,
    ) -> list[VerifiedQBankRoutingItemV2]:
        """Load the complete frozen scope before any channel ranking occurs."""

        if question_budget not in {1, 2, 4, 8} or round_id not in {0, 1, 2, 3}:
            raise ValueError("verified routing budget or round is outside the frozen grid")
        rows = (
            await self.session.execute(
                text(
                    """SELECT routing.routing_projection_id,
                              routing.compiler_run_id,
                              routing.snapshot_id,
                              routing.dataset,
                              routing.conversation_id,
                              routing.available_after_session,
                              routing.active_round,
                              routing.decision_hash,
                              routing.qbank_id,
                              routing.generation_rank,
                              routing.route_class,
                              routing.canonical_question,
                              routing.paraphrases,
                              routing.question_type,
                              routing.required_modalities,
                              routing.plan_id,
                              routing.program_hash,
                              routing.operator_set_version,
                              jsonb_build_object(
                                'requirement_contract_id',
                                  routing.requirement_contract_id,
                                'required_modalities',
                                  routing.contract_required_modalities,
                                'required_capabilities',
                                  routing.required_capabilities,
                                'allowed_capabilities',
                                  routing.allowed_capabilities,
                                'minimum_evidence_groups',
                                  routing.minimum_evidence_groups,
                                'raw_image_required',
                                  (('raw_image' = ANY(
                                      routing.contract_required_modalities
                                   )) OR
                                   ('raw_image' = ANY(
                                      routing.required_capabilities
                                   ))),
                                'fallback_required', routing.fallback_required,
                                'refusal_only', routing.refusal_only,
                                'content_hash', routing.requirement_contract_hash
                              ) AS requirement_contract,
                              routing.lexical_collection_id,
                              routing.text_collection_id,
                              routing.multimodal_collection_id,
                              jsonb_build_object(
                                'accepted_round', routing.accepted_round,
                                'projection_round', routing.projection_round,
                                'projection_ordinal', routing.ordinal
                              ) AS metadata
                       FROM pmmc_runtime.verified_qbank_routing_v2 routing
                       JOIN pmmc_runtime.benchmark_sessions available
                         ON available.snapshot_id = routing.snapshot_id
                        AND available.conversation_id = routing.conversation_id
                        AND available.session_id = routing.available_after_session
                       JOIN pmmc_runtime.benchmark_sessions cutoff
                         ON cutoff.snapshot_id = routing.snapshot_id
                        AND cutoff.conversation_id = routing.conversation_id
                        AND cutoff.session_id = :cutoff_session_id
                      WHERE routing.routing_projection_id = :routing_projection_id
                        AND routing.compiler_run_id = :compiler_run_id
                        AND routing.snapshot_id = :snapshot_id
                        AND routing.dataset = :dataset
                        AND routing.conversation_id = :conversation_id
                        AND routing.generation_rank <= :question_budget
                        AND routing.projection_round = :round_id
                        AND available.session_order <= cutoff.session_order
                      ORDER BY routing.qbank_id"""
                ),
                {
                    "routing_projection_id": routing_projection_id,
                    "compiler_run_id": compiler_run_id,
                    "snapshot_id": snapshot_id,
                    "dataset": dataset,
                    "conversation_id": conversation_id,
                    "cutoff_session_id": cutoff_session_id,
                    "question_budget": question_budget,
                    "round_id": round_id,
                },
            )
        ).mappings()
        return [VerifiedQBankRoutingItemV2.model_validate(dict(row)) for row in rows]

    async def list_verified_v3(
        self,
        *,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        question_budget: int,
        round_id: int,
        routing_version: str = "v3",
    ) -> list[VerifiedQBankRoutingItemV2]:
        """Load only Compiler 0.8 rows from the runtime-safe v3 projection."""

        if question_budget not in {1, 2, 4, 8} or round_id not in {0, 1, 2, 3}:
            raise ValueError("verified routing budget or round is outside the frozen grid")
        rows = (
            await self.session.execute(
                text(
                    _versioned_routing_sql(
                        """SELECT routing.routing_projection_id,
                              routing.compiler_run_id, routing.snapshot_id,
                              routing.dataset, routing.conversation_id,
                              routing.available_after_session, routing.active_round,
                              routing.decision_hash, routing.qbank_id,
                              routing.generation_rank, routing.route_class,
                              routing.canonical_question, routing.paraphrases,
                              routing.question_type, routing.answer_type,
                              routing.required_modalities, routing.plan_id,
                              routing.program_hash, routing.operator_set_version,
                              'verified-qbank-routing-v3' AS routing_schema_version,
                              jsonb_build_object(
                                'requirement_contract_id', routing.requirement_contract_id,
                                'required_modalities', routing.contract_required_modalities,
                                'required_capabilities', routing.required_capabilities,
                                'allowed_capabilities', routing.allowed_capabilities,
                                'minimum_evidence_groups', routing.minimum_evidence_groups,
                                'raw_image_required',
                                  (('raw_image' = ANY(routing.contract_required_modalities)) OR
                                   ('raw_image' = ANY(routing.required_capabilities))),
                                'fallback_required', routing.fallback_required,
                                'refusal_only', routing.refusal_only,
                                'content_hash', routing.requirement_contract_hash
                              ) AS requirement_contract,
                              routing.lexical_collection_id,
                              routing.text_collection_id,
                              routing.multimodal_collection_id,
                              jsonb_build_object(
                                'accepted_round', routing.accepted_round,
                                'projection_round', routing.projection_round,
                                'projection_ordinal', routing.ordinal
                              ) AS metadata
                       FROM pmmc_runtime.verified_qbank_routing_v3 routing
                       JOIN pmmc_runtime.benchmark_sessions available
                         ON available.snapshot_id = routing.snapshot_id
                        AND available.conversation_id = routing.conversation_id
                        AND available.session_id = routing.available_after_session
                       JOIN pmmc_runtime.benchmark_sessions cutoff
                         ON cutoff.snapshot_id = routing.snapshot_id
                        AND cutoff.conversation_id = routing.conversation_id
                        AND cutoff.session_id = :cutoff_session_id
                      WHERE routing.routing_projection_id = :routing_projection_id
                        AND routing.compiler_run_id = :compiler_run_id
                        AND routing.snapshot_id = :snapshot_id
                        AND routing.dataset = :dataset
                        AND routing.conversation_id = :conversation_id
                        AND routing.generation_rank <= :question_budget
                        AND routing.projection_round = :round_id
                        AND available.session_order <= cutoff.session_order
                      ORDER BY routing.qbank_id""",
                        routing_version,
                    )
                ),
                {
                    "routing_projection_id": routing_projection_id,
                    "compiler_run_id": compiler_run_id,
                    "snapshot_id": snapshot_id,
                    "dataset": dataset,
                    "conversation_id": conversation_id,
                    "cutoff_session_id": cutoff_session_id,
                    "question_budget": question_budget,
                    "round_id": round_id,
                },
            )
        ).mappings()
        if routing_version == "v4":
            from pmmc.memory.evidence_roles import EVIDENCE_ROLE_POLICY, required_evidence_roles
            from pmmc.schemas.qbank import VerifiedQBankRoutingItemV4

            values: list[VerifiedQBankRoutingItemV2] = []
            for row in rows:
                value = dict(row)
                requirement = dict(value["requirement_contract"])
                requirement["required_semantic_roles"] = required_evidence_roles(
                    modalities=requirement["required_modalities"],
                    capabilities=requirement["required_capabilities"],
                    refusal_only=requirement["refusal_only"],
                )
                requirement["semantic_role_policy"] = EVIDENCE_ROLE_POLICY
                relation_required = (
                    requirement["minimum_evidence_groups"] > 1 and not requirement["refusal_only"]
                )
                requirement["independent_group_relation_required"] = relation_required
                requirement["relation_assessment_stage"] = (
                    "answer_grounding" if relation_required else "not_applicable"
                )
                value["requirement_contract"] = requirement
                values.append(VerifiedQBankRoutingItemV4.model_validate(value))
            return values
        return [VerifiedQBankRoutingItemV3.model_validate(dict(row)) for row in rows]

    async def search_lexical_verified_v2(
        self,
        *,
        query_text: str,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Rank lexical candidates only after the frozen projection is joined."""

        if not query_text.strip() or top_k <= 0:
            raise ValueError("lexical query and top_k must be non-empty")
        params = {
            "query_text": query_text,
            "routing_projection_id": routing_projection_id,
            "compiler_run_id": compiler_run_id,
            "snapshot_id": snapshot_id,
            "dataset": dataset,
            "conversation_id": conversation_id,
            "cutoff_session_id": cutoff_session_id,
            "route_class": route_class,
            "question_budget": question_budget,
            "round_id": round_id,
            "top_k": top_k,
        }
        rows = (
            await self.session.execute(
                text(
                    """WITH scoped AS (
                         SELECT routing.qbank_id
                           FROM pmmc_runtime.verified_qbank_routing_v2 routing
                           JOIN pmmc_runtime.benchmark_sessions available
                             ON available.snapshot_id = routing.snapshot_id
                            AND available.conversation_id = routing.conversation_id
                            AND available.session_id = routing.available_after_session
                           JOIN pmmc_runtime.benchmark_sessions cutoff
                             ON cutoff.snapshot_id = routing.snapshot_id
                            AND cutoff.conversation_id = routing.conversation_id
                            AND cutoff.session_id = :cutoff_session_id
                          WHERE routing.routing_projection_id = :routing_projection_id
                            AND routing.compiler_run_id = :compiler_run_id
                            AND routing.snapshot_id = :snapshot_id
                            AND routing.dataset = :dataset
                            AND routing.conversation_id = :conversation_id
                            AND routing.route_class = :route_class
                            AND routing.generation_rank <= :question_budget
                            AND routing.projection_round = :round_id
                            AND available.session_order <= cutoff.session_order
                       ), scores AS (
                         SELECT q.qbank_id,
                                ts_rank_cd(q.search_document,
                                  plainto_tsquery('simple', :query_text)) AS score
                           FROM scoped s JOIN pmmc_runtime.qbank_questions q
                             ON q.qbank_id = s.qbank_id
                         UNION ALL
                         SELECT p.qbank_id,
                                ts_rank_cd(p.search_document,
                                  plainto_tsquery('simple', :query_text)) AS score
                           FROM scoped s JOIN pmmc_runtime.qbank_paraphrases p
                             ON p.qbank_id = s.qbank_id
                       )
                       SELECT qbank_id, max(score) AS raw_score
                         FROM scores WHERE score > 0 GROUP BY qbank_id
                        ORDER BY raw_score DESC, qbank_id LIMIT :top_k"""
                ),
                params,
            )
        ).all()
        return [(str(row.qbank_id), float(row.raw_score)) for row in rows]

    async def search_lexical_verified_v3(
        self,
        *,
        query_text: str,
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
        term_limit: int = 12,
        routing_version: str = "v3",
    ) -> list[tuple[str, float]]:
        """Use bounded OR-FTS after the v3 projection join, with exact-token boosts."""

        if top_k <= 0:
            raise ValueError("lexical top_k must be positive")
        ts_query, terms = build_step7b_or_tsquery(query_text, limit=term_limit)
        if not ts_query:
            return []
        boost_terms = extract_step7b_exact_boost_terms(query_text, limit=term_limit)
        rows = (
            await self.session.execute(
                text(
                    _versioned_routing_sql(
                        """WITH scoped AS (
                         SELECT routing.qbank_id
                           FROM pmmc_runtime.verified_qbank_routing_v3 routing
                           JOIN pmmc_runtime.benchmark_sessions available
                             ON available.snapshot_id = routing.snapshot_id
                            AND available.conversation_id = routing.conversation_id
                            AND available.session_id = routing.available_after_session
                           JOIN pmmc_runtime.benchmark_sessions cutoff
                             ON cutoff.snapshot_id = routing.snapshot_id
                            AND cutoff.conversation_id = routing.conversation_id
                            AND cutoff.session_id = :cutoff_session_id
                          WHERE routing.routing_projection_id = :routing_projection_id
                            AND routing.compiler_run_id = :compiler_run_id
                            AND routing.snapshot_id = :snapshot_id
                            AND routing.dataset = :dataset
                            AND routing.conversation_id = :conversation_id
                            AND routing.route_class = :route_class
                            AND routing.generation_rank <= :question_budget
                            AND routing.projection_round = :round_id
                            AND available.session_order <= cutoff.session_order
                       ), documents AS (
                         SELECT q.qbank_id, q.search_document, q.canonical_question AS body
                           FROM scoped s JOIN pmmc_runtime.qbank_questions q
                             ON q.qbank_id = s.qbank_id
                         UNION ALL
                         SELECT p.qbank_id, p.search_document, p.paraphrase AS body
                           FROM scoped s JOIN pmmc_runtime.qbank_paraphrases p
                             ON p.qbank_id = s.qbank_id
                       ), scores AS (
                         SELECT d.qbank_id,
                                ts_rank_cd(d.search_document,
                                  to_tsquery('simple', :ts_query)) +
                                0.05 * (
                                  SELECT count(*)
                                    FROM unnest(CAST(:boost_terms AS text[])) term
                                   WHERE to_tsvector('simple', d.body)
                                         @@ to_tsquery('simple', term)
                                ) AS score
                           FROM documents d
                          WHERE d.search_document @@ to_tsquery('simple', :ts_query)
                       )
                       SELECT qbank_id, max(score) AS raw_score
                         FROM scores GROUP BY qbank_id
                        ORDER BY raw_score DESC, qbank_id LIMIT :top_k""",
                        routing_version,
                    )
                ),
                {
                    "ts_query": ts_query,
                    "boost_terms": list(boost_terms),
                    "routing_projection_id": routing_projection_id,
                    "compiler_run_id": compiler_run_id,
                    "snapshot_id": snapshot_id,
                    "dataset": dataset,
                    "conversation_id": conversation_id,
                    "cutoff_session_id": cutoff_session_id,
                    "route_class": route_class,
                    "question_budget": question_budget,
                    "round_id": round_id,
                    "top_k": top_k,
                },
            )
        ).all()
        return [(str(row.qbank_id), float(row.raw_score)) for row in rows]

    async def search_embedding_verified_v2(
        self,
        *,
        collection_id: str,
        query_vector: Sequence[float],
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Exact cosine scan with the verified projection inside the SQL CTE."""

        if len(query_vector) != 4096 or top_k <= 0:
            raise ValueError("Step 7 embedding search requires a 4096d vector and positive top_k")
        vector_literal = (
            "[" + ",".join(format(float(value), ".17g") for value in query_vector) + "]"
        )
        rows = (
            await self.session.execute(
                text(
                    """WITH scoped AS (
                         SELECT routing.qbank_id
                           FROM pmmc_runtime.verified_qbank_routing_v2 routing
                           JOIN pmmc_runtime.benchmark_sessions available
                             ON available.snapshot_id = routing.snapshot_id
                            AND available.conversation_id = routing.conversation_id
                            AND available.session_id = routing.available_after_session
                           JOIN pmmc_runtime.benchmark_sessions cutoff
                             ON cutoff.snapshot_id = routing.snapshot_id
                            AND cutoff.conversation_id = routing.conversation_id
                            AND cutoff.session_id = :cutoff_session_id
                          WHERE routing.routing_projection_id = :routing_projection_id
                            AND routing.compiler_run_id = :compiler_run_id
                            AND routing.snapshot_id = :snapshot_id
                            AND routing.dataset = :dataset
                            AND routing.conversation_id = :conversation_id
                            AND routing.route_class = :route_class
                            AND routing.generation_rank <= :question_budget
                            AND routing.projection_round = :round_id
                            AND available.session_order <= cutoff.session_order
                       ), scored AS (
                         SELECT COALESCE(e.qbank_id, p.qbank_id) AS qbank_id,
                                1.0 - (e.embedding <=> CAST(:query_vector AS vector)) AS score
                           FROM pmmc_runtime.embeddings e
                           LEFT JOIN pmmc_runtime.qbank_paraphrases p
                             ON p.paraphrase_id = e.paraphrase_id
                           JOIN scoped s ON s.qbank_id = COALESCE(e.qbank_id, p.qbank_id)
                          WHERE e.collection_id = :collection_id
                       )
                       SELECT qbank_id, max(score) AS raw_score FROM scored
                        GROUP BY qbank_id ORDER BY raw_score DESC, qbank_id LIMIT :top_k"""
                ),
                {
                    "collection_id": collection_id,
                    "query_vector": vector_literal,
                    "routing_projection_id": routing_projection_id,
                    "compiler_run_id": compiler_run_id,
                    "snapshot_id": snapshot_id,
                    "dataset": dataset,
                    "conversation_id": conversation_id,
                    "cutoff_session_id": cutoff_session_id,
                    "route_class": route_class,
                    "question_budget": question_budget,
                    "round_id": round_id,
                    "top_k": top_k,
                },
            )
        ).all()
        return [(str(row.qbank_id), float(row.raw_score)) for row in rows]

    async def search_embedding_verified_v3(
        self,
        *,
        collection_id: str,
        query_vector: Sequence[float],
        routing_projection_id: str,
        compiler_run_id: str,
        snapshot_id: str,
        dataset: str,
        conversation_id: str,
        cutoff_session_id: str,
        route_class: str,
        question_budget: int,
        round_id: int,
        top_k: int = 50,
        routing_version: str = "v3",
    ) -> list[tuple[str, float]]:
        """Exact cosine scan with the Compiler 0.8 projection inside the SQL CTE."""

        if len(query_vector) != 4096 or top_k <= 0:
            raise ValueError("Step 7B embedding search requires 4096d and positive top_k")
        vector_literal = (
            "[" + ",".join(format(float(value), ".17g") for value in query_vector) + "]"
        )
        rows = (
            await self.session.execute(
                text(
                    _versioned_routing_sql(
                        """WITH scoped AS (
                         SELECT routing.qbank_id
                           FROM pmmc_runtime.verified_qbank_routing_v3 routing
                           JOIN pmmc_runtime.benchmark_sessions available
                             ON available.snapshot_id = routing.snapshot_id
                            AND available.conversation_id = routing.conversation_id
                            AND available.session_id = routing.available_after_session
                           JOIN pmmc_runtime.benchmark_sessions cutoff
                             ON cutoff.snapshot_id = routing.snapshot_id
                            AND cutoff.conversation_id = routing.conversation_id
                            AND cutoff.session_id = :cutoff_session_id
                          WHERE routing.routing_projection_id = :routing_projection_id
                            AND routing.compiler_run_id = :compiler_run_id
                            AND routing.snapshot_id = :snapshot_id
                            AND routing.dataset = :dataset
                            AND routing.conversation_id = :conversation_id
                            AND routing.route_class = :route_class
                            AND routing.generation_rank <= :question_budget
                            AND routing.projection_round = :round_id
                            AND available.session_order <= cutoff.session_order
                       ), scored AS (
                         SELECT COALESCE(e.qbank_id, p.qbank_id) AS qbank_id,
                                1.0 - (e.embedding <=> CAST(:query_vector AS vector)) AS score
                           FROM pmmc_runtime.embeddings e
                           LEFT JOIN pmmc_runtime.qbank_paraphrases p
                             ON p.paraphrase_id = e.paraphrase_id
                           JOIN scoped s ON s.qbank_id = COALESCE(e.qbank_id, p.qbank_id)
                          WHERE e.collection_id = :collection_id
                       )
                       SELECT qbank_id, max(score) AS raw_score FROM scored
                        GROUP BY qbank_id ORDER BY raw_score DESC, qbank_id LIMIT :top_k""",
                        routing_version,
                    )
                ),
                {
                    "collection_id": collection_id,
                    "query_vector": vector_literal,
                    "routing_projection_id": routing_projection_id,
                    "compiler_run_id": compiler_run_id,
                    "snapshot_id": snapshot_id,
                    "dataset": dataset,
                    "conversation_id": conversation_id,
                    "cutoff_session_id": cutoff_session_id,
                    "route_class": route_class,
                    "question_budget": question_budget,
                    "round_id": round_id,
                    "top_k": top_k,
                },
            )
        ).all()
        return [(str(row.qbank_id), float(row.raw_score)) for row in rows]

    async def get_program_steps(
        self, plan_id: str, *, routing_version: str = "legacy"
    ) -> list[ProgramStep]:
        if routing_version == "v4":
            view = "query_runtime_program_steps_v3"
        elif routing_version == "legacy":
            view = "qbank_routing_program_steps"
        else:
            raise ValueError("unsupported program steps routing version")
        rows = (
            await self.session.execute(
                text(
                    f"""SELECT step_id, op_name, args, expected_output_type, operator_set_version
                    FROM pmmc_runtime.{view}
                    WHERE plan_id = :plan_id
                    ORDER BY step_id"""
                ),
                {"plan_id": plan_id},
            )
        ).mappings()
        payloads = [dict(row) for row in rows]
        versions = {row.pop("operator_set_version") for row in payloads}
        if len(versions) > 1:
            raise ValueError("routing program steps mix operator protocols")
        values = [ProgramStep.model_validate(row) for row in payloads]
        if values and [item.step_id for item in values] != list(range(len(values))):
            raise ValueError("routing program steps are not contiguous")
        if versions == {"pmmc-operators-v7"}:
            # The safe view deliberately excludes arbitrary private step metadata.
            # Quotas are deterministic functions of frozen typed args, not model text.
            quotas, _ = program_evidence_quotas(values)
            values = [
                step.model_copy(update={"metadata": {"evidence_quota": quotas[step.step_id]}})
                for step in values
            ]
        return values

    async def get_runtime_program(self, plan_id: str) -> tuple[MemoryProgram, str] | None:
        """Load an expected-evidence-free executable projection of one frozen plan.

        The source program hash is returned separately and remains the identity bound by
        the Verified QBank.  The derived program receives its own canonical hash so the
        shared validator can still detect step or scope drift without reading compiler-
        private expected evidence.
        """

        stored = await self.session.get(MemoryProgramModel, plan_id)
        if stored is None:
            return None
        steps = await self.get_program_steps(plan_id)
        pending = MemoryProgram(
            plan_id=stored.plan_id,
            qbank_id=stored.qbank_id,
            run_id=stored.run_id,
            snapshot_id=stored.snapshot_id,
            compiler_unit_id=stored.compiler_unit_id,
            producer_model_call_id=None,
            operator_set_version=stored.operator_set_version,
            round_id=stored.round_id,
            operator_sequence=steps,
            expected_evidence_ids=[],
            fallback_operator=stored.fallback_operator,
            required_modalities=stored.required_modalities,
            planner_rationale_short="runtime-safe frozen strategy",
            planner_confidence=stored.planner_confidence,
            program_hash="pending",
            accepted=True,
            created_at=stored.created_at,
            metadata={"runtime_safe_projection": True},
        )
        runtime_hash = memory_program_content_hash(pending)
        return pending.model_copy(update={"program_hash": runtime_hash}), stored.program_hash

    async def get_runtime_program_v3(self, plan_id: str) -> tuple[MemoryProgram, str] | None:
        """Read only the source-locator-free ONLINE view, never the private ORM row."""
        row = (
            (
                await self.session.execute(
                    text(
                        "SELECT plan_id,qbank_id,run_id,snapshot_id,compiler_unit_id,"
                        "operator_set_version,"
                        "round_id,fallback_operator,required_modalities,planner_confidence,program_hash,"
                        "created_at FROM pmmc_runtime.query_runtime_programs_v3 programs "
                        "WHERE plan_id=:plan_id AND EXISTS ("
                        "SELECT 1 FROM pmmc_runtime.query_runtime_program_steps_v3 steps "
                        "WHERE steps.plan_id=programs.plan_id AND steps.run_id=programs.run_id "
                        "AND steps.qbank_id=programs.qbank_id "
                        "AND steps.program_hash=programs.program_hash)"
                    ),
                    {"plan_id": plan_id},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        source_hash = str(row["program_hash"])
        steps = await self.get_program_steps(plan_id, routing_version="v4")
        if not steps:
            return None
        pending = MemoryProgram(
            **{**dict(row), "program_hash": "pending"},
            operator_sequence=steps,
            expected_evidence_ids=[],
            planner_rationale_short="runtime-safe frozen strategy",
            accepted=True,
            metadata={"runtime_safe_projection": True},
        )
        return pending.model_copy(
            update={"program_hash": memory_program_content_hash(pending)}
        ), source_hash


class PostgresOperatorDataRepository:
    """Database implementation for all deterministic PMMC evidence operators."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def execution_evidence_metadata(
        self, context: OperatorExecutionContext, evidence: Sequence[ProgramExecutionEvidenceRef]
    ) -> dict[tuple[str, str], dict[str, object]]:
        """Resolve actual returned evidence groups once, without candidate support/gold.

        Do not broadcast a step's union of groups to every reference: two images
        from different owning rounds must keep their own group membership.
        """
        result: dict[tuple[str, str], dict[str, object]] = {}
        memory_ids = sorted({item.memory_id for item in evidence if item.memory_id is not None})
        if memory_ids:
            cutoff_order = await self._cutoff_order(context)
            available = aliased(BenchmarkSessionModel)
            rows = (await self.session.execute(
                select(MemoryUnitModel.memory_id, MemoryUnitModel.dataset,
                       MemoryUnitModel.session_id, MemoryUnitModel.metadata_json)
                .join(available, and_(
                    available.snapshot_id == MemoryUnitModel.snapshot_id,
                    available.conversation_id == MemoryUnitModel.conversation_id,
                    available.session_id == MemoryUnitModel.available_after_session,
                ))
                .where(
                    MemoryUnitModel.snapshot_id == context.snapshot_id,
                    MemoryUnitModel.conversation_id == context.conversation_id,
                    MemoryUnitModel.memory_id.in_(memory_ids),
                    available.session_order <= cutoff_order,
                )
            )).all()
            for row in rows:
                if row.dataset == DatasetName.MEMGALLERY.value and not (
                    isinstance(row.metadata_json, dict)
                    and isinstance(row.metadata_json.get("source_round_id"), (str, int))
                    and str(row.metadata_json["source_round_id"])
                ) and not is_memgallery_character_profile(row.metadata_json):
                    raise ValueError("MemGallery execution evidence lacks owning round")
                result[("memory", row.memory_id)] = self._evidence_group_metadata(
                    dataset=row.dataset, session_id=row.session_id,
                    memory_metadata=row.metadata_json, memory_id=row.memory_id,
                )
            if len(rows) != len(memory_ids):
                raise ValueError("execution evidence contains absent or future memory")
        image_ids = sorted({item.image_id for item in evidence if item.image_id is not None})
        if image_ids:
            bindings = await self._visible_image_memories(context, image_ids)
            if any(
                dataset == DatasetName.MEMGALLERY.value
                and not (
                    isinstance(metadata.get("source_round_id"), (str, int))
                    and str(metadata["source_round_id"])
                )
                for _, _, _, _, dataset, metadata in bindings
            ):
                raise ValueError("MemGallery execution image lacks owning round")
            result.update({
                ("image", image_id): metadata
                for image_id, metadata in self._image_evidence_group_metadata(bindings).items()
            })
        return result

    async def validate_context(self, context: OperatorExecutionContext) -> None:
        cutoff = await self.session.scalar(
            select(BenchmarkSessionModel).where(
                BenchmarkSessionModel.snapshot_id == context.snapshot_id,
                BenchmarkSessionModel.conversation_id == context.conversation_id,
                BenchmarkSessionModel.session_id == context.cutoff_session_id,
            )
        )
        if cutoff is None:
            raise ValueError("operator cutoff session does not match snapshot/conversation")
        if context.compiler_unit_id is not None:
            unit = await self.session.get(CompilerUnitModel, context.compiler_unit_id)
            if unit is None or (
                unit.run_id != context.source_compiler_run_id
                or unit.snapshot_id != context.snapshot_id
                or unit.conversation_id != context.conversation_id
            ):
                raise ValueError("operator context does not match compiler unit lineage")
            if context.query_id is None:
                if unit.available_after_session != context.cutoff_session_id:
                    raise ValueError("compiler dry-run cutoff does not match its unit")
            else:
                unit_cutoff = await self.session.scalar(
                    select(BenchmarkSessionModel.session_order).where(
                        BenchmarkSessionModel.snapshot_id == context.snapshot_id,
                        BenchmarkSessionModel.conversation_id == context.conversation_id,
                        BenchmarkSessionModel.session_id == unit.available_after_session,
                    )
                )
                query_cutoff = await self._cutoff_order(context)
                if unit_cutoff is None or int(unit_cutoff) > query_cutoff:
                    raise ValueError("compiler unit is not visible at the query cutoff")
        if context.query_id is not None:
            query = await self.session.get(
                BenchmarkQueryModel, (context.snapshot_id, context.query_id)
            )
            if (
                query is None
                or query.conversation_id != context.conversation_id
                or query.available_after_session != context.cutoff_session_id
            ):
                raise ValueError("operator context does not match query lineage")
            assignment = await self.session.get(
                RunQueryModel,
                (context.run_id, context.snapshot_id, context.query_id),
            )
            if assignment is None:
                raise ValueError("operator query is not assigned to the execution run")
            expected_images = set(
                (
                    await self.session.scalars(
                        select(QueryImageModel.image_id).where(
                            QueryImageModel.snapshot_id == context.snapshot_id,
                            QueryImageModel.query_id == context.query_id,
                        )
                    )
                ).all()
            )
            if not set(context.query_image_ids).issubset(expected_images):
                raise ValueError("operator query images do not match canonical query")
        collection_sources = {
            "text_memory": {
                EmbeddingSourceType.MEMORY_TEXT.value,
                EmbeddingSourceType.MEMORY_CAPTION.value,
                EmbeddingSourceType.MEMORY_IMAGE_TEXT_PAIR.value,
            },
            "visual_memory": {EmbeddingSourceType.MEMORY_IMAGE.value},
        }
        for role, collection_id in context.collection_bindings.items():
            if role not in collection_sources:
                raise ValueError(f"unknown operator collection binding role: {role}")
            collection = await self.session.get(EmbeddingCollectionModel, collection_id)
            if collection is None or collection.status != EmbeddingCollectionStatus.READY.value:
                raise ValueError(f"operator collection binding is not ready: {role}")
            if collection.snapshot_id != context.snapshot_id:
                raise ValueError("operator collection binding has a different snapshot")
            if collection.source_type not in collection_sources[role]:
                raise ValueError(
                    f"operator collection source type does not match binding role: {role}"
                )
        for role, vector in context.query_vectors.items():
            if not vector or any(not math.isfinite(float(value)) for value in vector):
                raise ValueError(f"operator query vector is invalid: {role}")

    async def validate_explicit_references(
        self,
        context: OperatorExecutionContext,
        *,
        memory_ids: list[str],
        image_ids: list[str],
        session_ids: list[str],
    ) -> None:
        cutoff_order = await self._cutoff_order(context)
        if memory_ids:
            available = aliased(BenchmarkSessionModel)
            visible_memories = set(
                (
                    await self.session.scalars(
                        select(MemoryUnitModel.memory_id)
                        .join(
                            available,
                            and_(
                                available.snapshot_id == MemoryUnitModel.snapshot_id,
                                available.conversation_id == MemoryUnitModel.conversation_id,
                                available.session_id == MemoryUnitModel.available_after_session,
                            ),
                        )
                        .where(
                            MemoryUnitModel.snapshot_id == context.snapshot_id,
                            MemoryUnitModel.conversation_id == context.conversation_id,
                            MemoryUnitModel.memory_id.in_(memory_ids),
                            available.session_order <= cutoff_order,
                        )
                    )
                ).all()
            )
            if visible_memories != set(memory_ids):
                raise ValueError("program contains absent or future memory IDs")
        if image_ids:
            await self._visible_image_memories(context, image_ids)
        if session_ids:
            visible_sessions = set(
                (
                    await self.session.scalars(
                        select(BenchmarkSessionModel.session_id).where(
                            BenchmarkSessionModel.snapshot_id == context.snapshot_id,
                            BenchmarkSessionModel.conversation_id == context.conversation_id,
                            BenchmarkSessionModel.session_id.in_(session_ids),
                            BenchmarkSessionModel.session_order <= cutoff_order,
                        )
                    )
                ).all()
            )
            if visible_sessions != set(session_ids):
                raise ValueError("program contains absent or future session IDs")

    async def _cutoff_order(self, context: OperatorExecutionContext) -> int:
        value = await self.session.scalar(
            select(BenchmarkSessionModel.session_order).where(
                BenchmarkSessionModel.snapshot_id == context.snapshot_id,
                BenchmarkSessionModel.conversation_id == context.conversation_id,
                BenchmarkSessionModel.session_id == context.cutoff_session_id,
            )
        )
        if value is None:
            raise KeyError("operator cutoff session does not exist")
        return int(value)

    @staticmethod
    def _evidence_group_metadata(
        *, dataset: str, session_id: str, memory_metadata: object, memory_id: str
    ) -> dict[str, object]:
        metadata = memory_metadata if isinstance(memory_metadata, dict) else {}
        source_round_id = metadata.get("source_round_id")
        if (
            dataset == DatasetName.MEMGALLERY.value
            and isinstance(source_round_id, (str, int)) and str(source_round_id)
        ):
            return {"evidence_group_id": f"round:{source_round_id}"}
        if (
            dataset == DatasetName.MEMGALLERY.value
            and is_memgallery_character_profile(metadata)
        ):
            return {"evidence_group_id": memgallery_profile_group(memory_id, session_id, metadata)}
        return {"evidence_group_id": f"session:{session_id}"}

    @classmethod
    def _image_evidence_group_metadata(
        cls,
        bindings: list[tuple[str, str, str, int, str, dict[str, object]]],
    ) -> dict[str, dict[str, object]]:
        groups_by_image: dict[str, set[str]] = {}
        for image_id, memory_id, session_id, _, dataset, memory_metadata in bindings:
            metadata = cls._evidence_group_metadata(
                dataset=dataset,
                session_id=session_id,
                memory_metadata=memory_metadata,
                memory_id=memory_id,
            )
            group_id = metadata["evidence_group_id"]
            if isinstance(group_id, str):
                groups_by_image.setdefault(image_id, set()).add(group_id)
        result: dict[str, dict[str, object]] = {}
        for image_id, groups in groups_by_image.items():
            ordered_groups = sorted(groups)
            result[image_id] = {"evidence_group_ids": ordered_groups}
            if len(ordered_groups) == 1:
                result[image_id]["evidence_group_id"] = ordered_groups[0]
        return result

    async def search_keyword(
        self,
        context: OperatorExecutionContext,
        *,
        query: str,
        top_k: int,
        entity_mentions: list[str],
    ) -> list[OperatorEvidence]:
        cutoff_order = await self._cutoff_order(context)
        available = aliased(BenchmarkSessionModel)
        tsquery = func.plainto_tsquery("simple", query)
        rank = func.ts_rank_cd(MemoryUnitModel.search_document, tsquery)
        statement = (
            select(
                MemoryUnitModel.memory_id,
                MemoryUnitModel.dataset,
                MemoryUnitModel.session_id,
                MemoryUnitModel.metadata_json,
                rank.label("score"),
            )
            .join(
                available,
                and_(
                    available.snapshot_id == MemoryUnitModel.snapshot_id,
                    available.conversation_id == MemoryUnitModel.conversation_id,
                    available.session_id == MemoryUnitModel.available_after_session,
                ),
            )
            .where(
                MemoryUnitModel.snapshot_id == context.snapshot_id,
                MemoryUnitModel.conversation_id == context.conversation_id,
                available.session_order <= cutoff_order,
                MemoryUnitModel.search_document.op("@@")(tsquery),
            )
        )
        if entity_mentions:
            statement = statement.where(MemoryUnitModel.entity_mentions.overlap(entity_mentions))
        rows = (
            await self.session.execute(
                statement.order_by(rank.desc(), MemoryUnitModel.memory_id).limit(top_k)
            )
        ).all()
        return [
            OperatorEvidence(
                evidence_id=row.memory_id,
                evidence_type=EvidenceType.MEMORY,
                memory_id=row.memory_id,
                score=RankingScore(
                    score_type="postgres_fts_rank",
                    raw_value=float(row.score),
                    direction=ScoreDirection.HIGHER_IS_BETTER,
                ),
                metadata=self._evidence_group_metadata(
                    dataset=row.dataset,
                    session_id=row.session_id,
                    memory_metadata=row.metadata_json,
                    memory_id=row.memory_id,
                ),
            )
            for row in rows
        ]

    async def search_embedding(
        self,
        context: OperatorExecutionContext,
        *,
        collection_id: str,
        vector: list[float],
        top_k: int,
        evidence_type: EvidenceType,
    ) -> list[OperatorEvidence]:
        collection = await self.session.get(EmbeddingCollectionModel, collection_id)
        if collection is None or collection.status != "ready":
            raise InvalidStateTransitionError("operator embedding collection is not ready")
        if collection.snapshot_id != context.snapshot_id:
            raise ValueError("operator embedding collection snapshot mismatch")
        if len(vector) != collection.dimension:
            raise ValueError("operator query vector dimension mismatch")
        cutoff_order = await self._cutoff_order(context)
        if collection.distance_metric == "cosine":
            distance = EmbeddingModel.embedding.cosine_distance(vector)
            raw = 1.0 - distance
            score_type = "cosine_similarity"
            direction = ScoreDirection.HIGHER_IS_BETTER
        elif collection.distance_metric == "inner_product":
            distance = EmbeddingModel.embedding.max_inner_product(vector)
            raw = -distance
            score_type = "inner_product"
            direction = ScoreDirection.HIGHER_IS_BETTER
        else:
            distance = EmbeddingModel.embedding.l2_distance(vector)
            raw = distance
            score_type = "l2_distance"
            direction = ScoreDirection.LOWER_IS_BETTER
        available = aliased(BenchmarkSessionModel)
        if evidence_type is EvidenceType.MEMORY:
            memory_source_id = func.coalesce(
                EmbeddingModel.metadata_json["memory_id"].astext,
                EmbeddingModel.source_id,
            )
            statement = (
                select(
                    memory_source_id.label("source_id"),
                    MemoryUnitModel.dataset,
                    MemoryUnitModel.session_id,
                    MemoryUnitModel.metadata_json,
                    raw.label("score"),
                )
                .join(
                    MemoryUnitModel,
                    and_(
                        MemoryUnitModel.snapshot_id == context.snapshot_id,
                        MemoryUnitModel.memory_id == memory_source_id,
                    ),
                )
                .join(
                    available,
                    and_(
                        available.snapshot_id == MemoryUnitModel.snapshot_id,
                        available.conversation_id == MemoryUnitModel.conversation_id,
                        available.session_id == MemoryUnitModel.available_after_session,
                    ),
                )
                .where(
                    EmbeddingModel.collection_id == collection_id,
                    MemoryUnitModel.conversation_id == context.conversation_id,
                    available.session_order <= cutoff_order,
                )
            )
        elif evidence_type is EvidenceType.IMAGE:
            eligible_image_ids = (
                select(MemoryUnitImageModel.image_id)
                .join(
                    MemoryUnitModel,
                    and_(
                        MemoryUnitModel.snapshot_id == MemoryUnitImageModel.snapshot_id,
                        MemoryUnitModel.memory_id == MemoryUnitImageModel.memory_id,
                    ),
                )
                .join(
                    available,
                    and_(
                        available.snapshot_id == MemoryUnitModel.snapshot_id,
                        available.conversation_id == MemoryUnitModel.conversation_id,
                        available.session_id == MemoryUnitModel.available_after_session,
                    ),
                )
                .where(
                    MemoryUnitImageModel.snapshot_id == context.snapshot_id,
                    MemoryUnitModel.conversation_id == context.conversation_id,
                    available.session_order <= cutoff_order,
                )
                .distinct()
            )
            statement = select(EmbeddingModel.source_id, raw.label("score")).where(
                EmbeddingModel.collection_id == collection_id,
                EmbeddingModel.source_id.in_(eligible_image_ids),
            )
            rows = (
                await self.session.execute(
                    statement.order_by(distance, EmbeddingModel.source_id).limit(top_k)
                )
            ).all()
            image_ids = list(dict.fromkeys(row.source_id for row in rows))
            visible = await self._visible_image_memories(context, image_ids)
            group_metadata = self._image_evidence_group_metadata(visible)
            return [
                OperatorEvidence(
                    evidence_id=row.source_id,
                    evidence_type=EvidenceType.IMAGE,
                    image_id=row.source_id,
                    score=RankingScore(
                        score_type=score_type,
                        raw_value=float(row.score),
                        direction=direction,
                    ),
                    metadata=group_metadata[row.source_id],
                )
                for row in rows
            ]
        else:
            raise ValueError("embedding operators support memory or image evidence only")
        rows = (
            await self.session.execute(
                statement.order_by(distance, EmbeddingModel.source_id).limit(top_k * 2)
            )
        ).all()
        deduplicated = []
        seen: set[str] = set()
        for row in rows:
            if row.source_id in seen:
                continue
            seen.add(row.source_id)
            deduplicated.append(row)
            if len(deduplicated) == top_k:
                break
        return [
            OperatorEvidence(
                evidence_id=row.source_id,
                evidence_type=evidence_type,
                memory_id=row.source_id,
                score=RankingScore(
                    score_type=score_type,
                    raw_value=float(row.score),
                    direction=direction,
                ),
                metadata=self._evidence_group_metadata(
                    dataset=row.dataset,
                    session_id=row.session_id,
                    memory_metadata=row.metadata_json,
                    memory_id=row.source_id,
                ),
            )
            for row in deduplicated
        ]

    async def _visible_image_memories(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[tuple[str, str, str, int, str, dict[str, object]]]:
        if not image_ids:
            return []
        cutoff_order = await self._cutoff_order(context)
        available = aliased(BenchmarkSessionModel)
        rows = (
            await self.session.execute(
                select(
                    MemoryUnitImageModel.image_id,
                    MemoryUnitModel.memory_id,
                    MemoryUnitModel.session_id,
                    MemoryUnitModel.timestamp_order,
                    MemoryUnitModel.dataset,
                    MemoryUnitModel.metadata_json,
                )
                .join(
                    MemoryUnitModel,
                    and_(
                        MemoryUnitModel.snapshot_id == MemoryUnitImageModel.snapshot_id,
                        MemoryUnitModel.memory_id == MemoryUnitImageModel.memory_id,
                    ),
                )
                .join(
                    available,
                    and_(
                        available.snapshot_id == MemoryUnitModel.snapshot_id,
                        available.conversation_id == MemoryUnitModel.conversation_id,
                        available.session_id == MemoryUnitModel.available_after_session,
                    ),
                )
                .where(
                    MemoryUnitImageModel.snapshot_id == context.snapshot_id,
                    MemoryUnitImageModel.image_id.in_(image_ids),
                    MemoryUnitModel.conversation_id == context.conversation_id,
                    available.session_order <= cutoff_order,
                )
                .order_by(
                    MemoryUnitModel.timestamp_order,
                    MemoryUnitImageModel.image_id,
                    MemoryUnitModel.memory_id,
                )
            )
        ).all()
        found = {row.image_id for row in rows}
        missing = set(image_ids) - found
        if missing:
            raise ValueError(f"image is absent or not visible at cutoff: {sorted(missing)[0]}")
        return [
            (
                row.image_id,
                row.memory_id,
                row.session_id,
                row.timestamp_order,
                row.dataset,
                dict(row.metadata_json),
            )
            for row in rows
        ]

    async def resolve_image_text_pairs(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]:
        rows = await self._visible_image_memories(context, image_ids)
        result: list[OperatorEvidence] = []
        for image_id, memory_id, session_id, _, dataset, memory_metadata in rows:
            metadata = self._evidence_group_metadata(
                dataset=dataset,
                session_id=session_id,
                memory_metadata=memory_metadata,
                memory_id=memory_id,
            )
            result.extend(
                (
                    OperatorEvidence(
                        evidence_id=image_id,
                        evidence_type=EvidenceType.IMAGE_TEXT_PAIR,
                        image_id=image_id,
                        metadata=metadata,
                    ),
                    OperatorEvidence(
                        evidence_id=memory_id,
                        evidence_type=EvidenceType.IMAGE_TEXT_PAIR,
                        memory_id=memory_id,
                        metadata=metadata,
                    ),
                )
            )
        return result

    async def context_around_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]:
        sources = await self._visible_image_memories(context, image_ids)
        identifiers: set[str] = set()
        for _, _, session_id, timestamp_order, _, _ in sources:
            rows = await self.session.scalars(
                select(MemoryUnitModel.memory_id).where(
                    MemoryUnitModel.snapshot_id == context.snapshot_id,
                    MemoryUnitModel.conversation_id == context.conversation_id,
                    MemoryUnitModel.session_id == session_id,
                    MemoryUnitModel.timestamp_order.between(
                        timestamp_order - radius, timestamp_order + radius
                    ),
                )
            )
            identifiers.update(rows.all())
        ordered = list(
            (
                await self.session.execute(
                    select(
                        MemoryUnitModel.memory_id,
                        MemoryUnitModel.dataset,
                        MemoryUnitModel.session_id,
                        MemoryUnitModel.metadata_json,
                    )
                    .where(
                        MemoryUnitModel.snapshot_id == context.snapshot_id,
                        MemoryUnitModel.conversation_id == context.conversation_id,
                        MemoryUnitModel.memory_id.in_(identifiers),
                    )
                    .order_by(MemoryUnitModel.timestamp_order, MemoryUnitModel.memory_id)
                )
            ).all()
        )
        return [
            OperatorEvidence(
                evidence_id=value.memory_id,
                evidence_type=EvidenceType.MEMORY,
                memory_id=value.memory_id,
                metadata=self._evidence_group_metadata(
                    dataset=value.dataset,
                    session_id=value.session_id,
                    memory_metadata=value.metadata_json,
                    memory_id=value.memory_id,
                ),
            )
            for value in ordered
        ]

    async def expand_temporal_neighbors(
        self,
        context: OperatorExecutionContext,
        memory_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]:
        cutoff_order = await self._cutoff_order(context)
        available = aliased(BenchmarkSessionModel)
        source_orders = list(
            (
                await self.session.scalars(
                    select(MemoryUnitModel.timestamp_order)
                    .join(
                        available,
                        and_(
                            available.snapshot_id == MemoryUnitModel.snapshot_id,
                            available.conversation_id == MemoryUnitModel.conversation_id,
                            available.session_id == MemoryUnitModel.available_after_session,
                        ),
                    )
                    .where(
                        MemoryUnitModel.snapshot_id == context.snapshot_id,
                        MemoryUnitModel.conversation_id == context.conversation_id,
                        MemoryUnitModel.memory_id.in_(memory_ids),
                        available.session_order <= cutoff_order,
                    )
                )
            ).all()
        )
        if len(source_orders) != len(set(memory_ids)):
            raise ValueError("temporal expansion received absent or future memory evidence")
        available = aliased(BenchmarkSessionModel)
        predicates = [
            MemoryUnitModel.timestamp_order.between(value - radius, value + radius)
            for value in source_orders
        ]
        if not predicates:
            return []
        rows = list(
            (
                await self.session.execute(
                    select(
                        MemoryUnitModel.memory_id,
                        MemoryUnitModel.dataset,
                        MemoryUnitModel.session_id,
                        MemoryUnitModel.metadata_json,
                    )
                    .join(
                        available,
                        and_(
                            available.snapshot_id == MemoryUnitModel.snapshot_id,
                            available.conversation_id == MemoryUnitModel.conversation_id,
                            available.session_id == MemoryUnitModel.available_after_session,
                        ),
                    )
                    .where(
                        MemoryUnitModel.snapshot_id == context.snapshot_id,
                        MemoryUnitModel.conversation_id == context.conversation_id,
                        available.session_order <= cutoff_order,
                        or_(*predicates),
                    )
                    .order_by(MemoryUnitModel.timestamp_order, MemoryUnitModel.memory_id)
                )
            ).all()
        )
        deduplicated = {row.memory_id: row for row in rows}
        return [
            OperatorEvidence(
                evidence_id=value.memory_id,
                evidence_type=EvidenceType.MEMORY,
                memory_id=value.memory_id,
                metadata=self._evidence_group_metadata(
                    dataset=value.dataset,
                    session_id=value.session_id,
                    memory_metadata=value.metadata_json,
                    memory_id=value.memory_id,
                ),
            )
            for value in deduplicated.values()
        ]

    async def inspect_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]:
        visible = await self._visible_image_memories(context, image_ids)
        group_metadata = self._image_evidence_group_metadata(visible)
        images = list(
            (
                await self.session.execute(
                    select(ImageModel)
                    .where(
                        ImageModel.snapshot_id == context.snapshot_id,
                        ImageModel.image_id.in_(image_ids),
                    )
                    .order_by(ImageModel.image_id)
                )
            )
            .scalars()
            .all()
        )
        for image in images:
            if image.storage_uri is None and image.local_path is None:
                raise ValueError(f"image cannot be materialized: {image.image_id}")
        return [
            OperatorEvidence(
                evidence_id=image.image_id,
                evidence_type=EvidenceType.RAW_IMAGE,
                image_id=image.image_id,
                metadata={
                    **group_metadata[image.image_id],
                    "sha256": image.sha256,
                    "mime_type": image.mime_type,
                    "width": image.width,
                    "height": image.height,
                },
            )
            for image in images
        ]


class SessionScopedOperatorDataRepository:
    """Run each operator read in its own short-lived database session."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def _call(self, method: str, *args: object, **kwargs: object) -> Any:
        async with self.sessions() as session:
            repository = PostgresOperatorDataRepository(session)
            operation = getattr(repository, method)
            return await operation(*args, **kwargs)

    async def validate_context(self, context: OperatorExecutionContext) -> None:
        await self._call("validate_context", context)

    async def execution_evidence_metadata(
        self, context: OperatorExecutionContext, evidence: Sequence[ProgramExecutionEvidenceRef]
    ) -> dict[tuple[str, str], dict[str, object]]:
        return cast(dict[tuple[str, str], dict[str, object]], await self._call(
            "execution_evidence_metadata", context, evidence
        ))

    async def validate_explicit_references(
        self,
        context: OperatorExecutionContext,
        *,
        memory_ids: list[str],
        image_ids: list[str],
        session_ids: list[str],
    ) -> None:
        await self._call(
            "validate_explicit_references",
            context,
            memory_ids=memory_ids,
            image_ids=image_ids,
            session_ids=session_ids,
        )

    async def search_keyword(
        self,
        context: OperatorExecutionContext,
        *,
        query: str,
        top_k: int,
        entity_mentions: list[str],
    ) -> list[OperatorEvidence]:
        return cast(
            list[OperatorEvidence],
            await self._call(
                "search_keyword",
                context,
                query=query,
                top_k=top_k,
                entity_mentions=entity_mentions,
            ),
        )

    async def search_embedding(
        self,
        context: OperatorExecutionContext,
        *,
        collection_id: str,
        vector: list[float],
        top_k: int,
        evidence_type: EvidenceType,
    ) -> list[OperatorEvidence]:
        return cast(
            list[OperatorEvidence],
            await self._call(
                "search_embedding",
                context,
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                evidence_type=evidence_type,
            ),
        )

    async def resolve_image_text_pairs(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]:
        return cast(
            list[OperatorEvidence],
            await self._call("resolve_image_text_pairs", context, image_ids),
        )

    async def context_around_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]:
        return cast(
            list[OperatorEvidence],
            await self._call("context_around_images", context, image_ids, radius),
        )

    async def expand_temporal_neighbors(
        self,
        context: OperatorExecutionContext,
        memory_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]:
        return cast(
            list[OperatorEvidence],
            await self._call("expand_temporal_neighbors", context, memory_ids, radius),
        )

    async def inspect_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]:
        return cast(
            list[OperatorEvidence],
            await self._call("inspect_images", context, image_ids),
        )


__all__ = [
    "PostgresCompilerRunRepository",
    "PostgresCompilerUnitRepository",
    "PostgresMemoryProgramRepository",
    "PostgresOperatorDataRepository",
    "PostgresProgramExecutionRepository",
    "PostgresQBankCompilerRepository",
    "PostgresQBankRoutingRepository",
    "QUESTIONER_STAGE_ACCOUNTING_SCHEMA_VERSIONS",
    "SessionScopedOperatorDataRepository",
    "compiler_questioner_stage_expected_counts",
    "compiler_unit_fact_counts",
    "inspect_compiler_run",
    "summarize_compiler_verification_issues",
    "verify_compiler_run",
]


async def inspect_compiler_run(
    session: AsyncSession,
    run_id: str,
) -> dict[str, object]:
    from pmmc.infrastructure.database.compiler_models import (
        CompilerStageJobModel,
        MemoryOperatorIndexManifestModel,
        PlannerBatchProgramModel,
        PlannerCandidateOutcomeModel,
        PlannerGenerationBatchModel,
        ProgramDryRunAssessmentModel,
        ProgramDryRunEvidenceMatchModel,
        ProgramQueryEmbeddingBatchModel,
        ProgramQueryEmbeddingModel,
        ProgramValidationIssueModel,
        ProgramValidationResultModel,
    )
    from pmmc.infrastructure.database.questioner_models import (
        CompilerUnitLinkModel,
        QBankDuplicateLinkModel,
        QBankIndexManifestModel,
        QuestionerCandidateGateResultModel,
        QuestionerCandidateSupersessionModel,
        QuestionerCandidateVerificationModel,
        QuestionerGenerationModel,
        QuestionerVerificationBatchModel,
        QuestionOpportunityEvidenceRefModel,
        QuestionOpportunityModel,
        QuestionOpportunitySetModel,
    )

    compiler = await session.get(CompilerRunModel, run_id)
    if compiler is None:
        raise KeyError(f"compiler run does not exist: {run_id}")
    state = await session.get(RunStateModel, run_id)
    count_models = (
        ("units", CompilerUnitModel),
        ("unit_memories", CompilerUnitMemoryModel),
        ("stage_results", CompilerUnitStageResultModel),
        ("questions", QBankQuestionModel),
        ("paraphrases", QBankParaphraseModel),
        ("programs", MemoryProgramModel),
        ("executions", ProgramExecutionModel),
        ("execution_steps", ProgramExecutionStepModel),
        ("feedback", DoubterFeedbackModel),
        ("decisions", QBankDecisionModel),
        ("unit_links", CompilerUnitLinkModel),
        ("questioner_generations", QuestionerGenerationModel),
        ("candidate_gate_results", QuestionerCandidateGateResultModel),
        ("duplicate_links", QBankDuplicateLinkModel),
        ("opportunity_sets", QuestionOpportunitySetModel),
        ("opportunities", QuestionOpportunityModel),
        ("opportunity_evidence_refs", QuestionOpportunityEvidenceRefModel),
        ("verification_batches", QuestionerVerificationBatchModel),
        ("candidate_verifications", QuestionerCandidateVerificationModel),
        ("candidate_supersessions", QuestionerCandidateSupersessionModel),
        ("compiler_stage_jobs", CompilerStageJobModel),
        ("planner_generation_batches", PlannerGenerationBatchModel),
        ("planner_candidate_outcomes", PlannerCandidateOutcomeModel),
        ("planner_batch_programs", PlannerBatchProgramModel),
        ("program_validation_results", ProgramValidationResultModel),
        ("program_validation_issues", ProgramValidationIssueModel),
        ("memory_operator_indexes", MemoryOperatorIndexManifestModel),
        ("program_query_embedding_batches", ProgramQueryEmbeddingBatchModel),
        ("program_query_embeddings", ProgramQueryEmbeddingModel),
        ("program_dry_run_assessments", ProgramDryRunAssessmentModel),
        ("program_dry_run_evidence_matches", ProgramDryRunEvidenceMatchModel),
    )
    counts: dict[str, int] = {}
    for name, model in count_models:
        counts[name] = int(
            await session.scalar(
                select(func.count()).select_from(model).where(model.run_id == run_id)
            )
            or 0
        )
    counts["qbank_indexes"] = int(
        await session.scalar(
            select(func.count())
            .select_from(QBankIndexManifestModel)
            .where(QBankIndexManifestModel.compiler_run_id == run_id)
        )
        or 0
    )
    job_rows = (
        await session.execute(
            select(ExperimentJobModel.status, func.count())
            .where(
                ExperimentJobModel.run_id == run_id,
                ExperimentJobModel.stage == "compile",
            )
            .group_by(ExperimentJobModel.status)
        )
    ).all()
    job_failure_rows = (
        await session.execute(
            select(
                ExperimentJobModel.error_type,
                ExperimentJobModel.error_summary,
                func.count(),
            )
            .where(
                ExperimentJobModel.run_id == run_id,
                ExperimentJobModel.stage == "compile",
                ExperimentJobModel.status == "failed",
            )
            .group_by(
                ExperimentJobModel.error_type,
                ExperimentJobModel.error_summary,
            )
            .order_by(
                ExperimentJobModel.error_type,
                ExperimentJobModel.error_summary,
            )
        )
    ).all()
    model_failure_rows = (
        await session.execute(
            select(
                RuntimeModelCallModel.stage,
                RuntimeModelCallModel.error_type,
                func.count(),
            )
            .where(
                RuntimeModelCallModel.run_id == run_id,
                RuntimeModelCallModel.status == "failed",
            )
            .group_by(
                RuntimeModelCallModel.stage,
                RuntimeModelCallModel.error_type,
            )
            .order_by(
                RuntimeModelCallModel.stage,
                RuntimeModelCallModel.error_type,
            )
        )
    ).all()
    return {
        "run_id": run_id,
        "status": state.status if state is not None else "missing",
        "target_stage": compiler.target_stage,
        "unit_count": compiler.unit_count,
        "operator_set_version": compiler.operator_set_version,
        "operator_registry_hash": compiler.operator_registry_hash,
        "compiler_config_hash": compiler.compiler_config_hash,
        "compiler_schema_version": compiler.compiler_schema_version,
        "questioner_config_hash": compiler.questioner_config_hash,
        "questioner_model_alias": compiler.questioner_model_alias,
        "qbank_matcher_version": compiler.qbank_matcher_version,
        "counts": counts,
        "compile_jobs": {status: int(count) for status, count in job_rows},
        "compile_job_failures": [
            {
                "error_type": error_type,
                "error_summary": error_summary,
                "count": int(count),
            }
            for error_type, error_summary, count in job_failure_rows
        ],
        "model_call_failures": [
            {
                "stage": stage,
                "error_type": error_type,
                "count": int(count),
            }
            for stage, error_type, count in model_failure_rows
        ],
    }


async def compiler_v04_stage_expected_counts(
    session: AsyncSession,
    *,
    run_id: str,
    compiler_unit_id: str,
    stage: str,
    fact_counts: Mapping[str, int],
) -> tuple[int, int, int] | None:
    """Return the v0.4 Planner/dry-run denominator derived from atomic facts."""

    from pmmc.infrastructure.database.compiler_models import PlannerCandidateOutcomeModel
    from pmmc.infrastructure.database.questioner_models import (
        QuestionerCandidateGateResultModel,
    )

    # Unit Planner/dry-run results describe plan0, not later Doubter revisions.
    program_count = fact_counts.get("planner_r0", fact_counts["planner"])
    if stage == "planner":
        eligible_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QuestionerCandidateGateResultModel)
                .where(
                    QuestionerCandidateGateResultModel.run_id == run_id,
                    QuestionerCandidateGateResultModel.compiler_unit_id == compiler_unit_id,
                    QuestionerCandidateGateResultModel.status == "eligible_for_planning",
                )
            )
            or 0
        )
        return eligible_count, program_count, eligible_count - program_count
    if stage == "dry_run":
        ready_statuses = {
            "program_ready",
            "program_ready_with_fallback",
            "program_ready_with_abstain",
        }
        outcome_statuses = list(
            await session.scalars(
                select(PlannerCandidateOutcomeModel.status).where(
                    PlannerCandidateOutcomeModel.run_id == run_id,
                    PlannerCandidateOutcomeModel.compiler_unit_id == compiler_unit_id,
                    PlannerCandidateOutcomeModel.plan_id.is_not(None),
                )
            )
        )
        accepted_count = sum(status in ready_statuses for status in outcome_statuses)
        return program_count, accepted_count, program_count - accepted_count
    return None


async def compiler_questioner_stage_expected_counts(
    session: AsyncSession,
    *,
    run_id: str,
    compiler_unit_id: str,
    compiler_schema_version: str,
) -> tuple[int, int, int] | None:
    """Derive Questioner stage counts from generation and terminal gate facts."""

    from pmmc.infrastructure.database.questioner_models import (
        QuestionerCandidateGateResultModel,
        QuestionerGenerationModel,
    )

    if compiler_schema_version not in QUESTIONER_STAGE_ACCOUNTING_SCHEMA_VERSIONS:
        return None
    generations = list(
        await session.scalars(
            select(QuestionerGenerationModel).where(
                QuestionerGenerationModel.run_id == run_id,
                QuestionerGenerationModel.compiler_unit_id == compiler_unit_id,
            )
        )
    )
    if len(generations) != 1:
        return None
    generation = generations[0]
    if generation.parse_status == "failed":
        metadata = generation.metadata_json
        if (
            compiler_schema_version in {"0.6.0", "0.7.0", "0.8.0", "0.9.0"}
            and generation.raw_candidate_count == 0
            and generation.valid_candidate_count == 0
            and generation.invalid_candidate_count == 0
            and isinstance(metadata, dict)
            and metadata.get("native_structured_output") is False
            and metadata.get("terminal_reason") == "generation_structured_output_invalid"
        ):
            return (0, 0, 0)
        return None
    eligible_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QuestionerCandidateGateResultModel)
            .where(
                QuestionerCandidateGateResultModel.run_id == run_id,
                QuestionerCandidateGateResultModel.compiler_unit_id == compiler_unit_id,
                QuestionerCandidateGateResultModel.status == "eligible_for_planning",
            )
        )
        or 0
    )
    rejected_count = int(
        await session.scalar(
            select(func.count())
            .select_from(QuestionerCandidateGateResultModel)
            .where(
                QuestionerCandidateGateResultModel.run_id == run_id,
                QuestionerCandidateGateResultModel.compiler_unit_id == compiler_unit_id,
                QuestionerCandidateGateResultModel.status == "rejected",
            )
        )
        or 0
    )
    return (
        generation.raw_candidate_count,
        eligible_count,
        (
            generation.invalid_candidate_count + rejected_count
            if compiler_schema_version == "0.2.0"
            else generation.raw_candidate_count - eligible_count
        ),
    )


def compiler_program_outcome_issues(
    program: MemoryProgramModel,
    *,
    outcome_status: str | None,
    revision: PlannerRevisionLineageModel | None,
    source_program: MemoryProgramModel | None,
    source_feedback: DoubterAssessmentModel | None,
) -> list[str]:
    """Separate candidate-level plan0 outcomes from append-only revision lineage."""

    if program.round_id == 0:
        return (
            [f"program terminal outcome is missing: {program.plan_id}"]
            if outcome_status is None
            else []
        )
    if (
        revision is None
        or source_program is None
        or source_feedback is None
        or outcome_status is not None
        or not 1 <= program.round_id <= 3
        or revision.run_id != program.run_id
        or revision.qbank_id != program.qbank_id
        or revision.revised_plan_id != program.plan_id
        or revision.target_round_id != program.round_id
        or revision.source_plan_id != source_program.plan_id
        or source_program.run_id != program.run_id
        or source_program.qbank_id != program.qbank_id
        or source_program.compiler_unit_id != program.compiler_unit_id
        or source_program.round_id != program.round_id - 1
        or source_feedback.run_id != program.run_id
        or source_feedback.qbank_id != program.qbank_id
        or source_feedback.compiler_unit_id != program.compiler_unit_id
        or source_feedback.plan_id != source_program.plan_id
        or source_feedback.round_id != source_program.round_id
        or source_feedback.feedback_id != revision.source_feedback_id
        or source_feedback.action != "revise_program"
    ):
        return [f"program revision lineage is incomplete: {program.plan_id}"]
    return []


async def verify_compiler_run(
    session: AsyncSession,
    run_id: str,
) -> dict[str, object]:
    from pmmc.infrastructure.database.compiler_models import (
        CompilerStageJobModel,
        MemoryOperatorIndexManifestModel,
        PlannerCandidateOutcomeModel,
        ProgramDryRunAssessmentModel,
        ProgramQueryEmbeddingModel,
        ProgramValidationResultModel,
    )
    from pmmc.infrastructure.database.doubter_models import (
        CandidateStageJobModel,
        DoubterAssessmentModel,
        QBankRoundProjectionManifestModel,
        QBankRoundProjectionMemberModel,
        R0InputManifestModel,
        RoutingProjectionManifestModel,
        RoutingProjectionMemberModel,
    )
    from pmmc.infrastructure.database.questioner_models import (
        CompilerUnitLinkModel,
        QBankIndexManifestModel,
        QuestionerCandidateGateResultModel,
    )

    summary = await inspect_compiler_run(session, run_id)
    compiler = await session.get(CompilerRunModel, run_id)
    assert compiler is not None
    issues: list[str] = []
    units = list(
        (
            await session.execute(
                select(CompilerUnitModel)
                .where(CompilerUnitModel.run_id == run_id)
                .order_by(CompilerUnitModel.ordinal)
            )
        )
        .scalars()
        .all()
    )
    if len(units) != compiler.unit_count:
        issues.append("compiler unit count differs from frozen denominator")
    if [item.ordinal for item in units] != list(range(len(units))):
        issues.append("compiler unit ordinals are not contiguous")
    required_stages = {
        "questions": {"questioner"},
        "programs": {"questioner", "planner", "dry_run"},
        "verified_qbank": {
            "questioner",
            "planner",
            "dry_run",
            "doubter",
            "decision",
        },
    }[compiler.target_stage]
    for unit in units:
        members = list(
            (
                await session.execute(
                    select(CompilerUnitMemoryModel)
                    .where(CompilerUnitMemoryModel.compiler_unit_id == unit.compiler_unit_id)
                    .order_by(CompilerUnitMemoryModel.ordinal)
                )
            )
            .scalars()
            .all()
        )
        if [item.ordinal for item in members] != list(range(len(members))):
            issues.append(f"unit membership ordinals are incomplete: {unit.compiler_unit_id}")
        if not any(item.role == "anchor" for item in members):
            issues.append(f"unit has no anchor memory: {unit.compiler_unit_id}")
        succeeded_jobs = int(
            await session.scalar(
                select(func.count())
                .select_from(ExperimentJobModel)
                .where(
                    ExperimentJobModel.run_id == run_id,
                    ExperimentJobModel.compiler_unit_id == unit.compiler_unit_id,
                    ExperimentJobModel.stage == "compile",
                    ExperimentJobModel.status == "succeeded",
                )
            )
            or 0
        )
        if succeeded_jobs != 1:
            issues.append(f"unit does not have exactly one succeeded job: {unit.compiler_unit_id}")
        succeeded_results = list(
            (
                await session.scalars(
                    select(CompilerUnitStageResultModel).where(
                        CompilerUnitStageResultModel.compiler_unit_id == unit.compiler_unit_id,
                        CompilerUnitStageResultModel.status == "success",
                    )
                )
            ).all()
        )
        succeeded_stages = {result.stage for result in succeeded_results}
        missing_stages = required_stages - succeeded_stages
        if missing_stages:
            issues.append(
                f"unit is missing stages {sorted(missing_stages)}: {unit.compiler_unit_id}"
            )
        fact_counts = await compiler_unit_fact_counts(
            session,
            run_id=run_id,
            compiler_unit_id=unit.compiler_unit_id,
        )
        for result in succeeded_results:
            if (
                result.stage == "questioner"
                and compiler.compiler_schema_version in QUESTIONER_STAGE_ACCOUNTING_SCHEMA_VERSIONS
            ):
                expected_counts = await compiler_questioner_stage_expected_counts(
                    session,
                    run_id=run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    compiler_schema_version=compiler.compiler_schema_version,
                )
                if expected_counts is None:
                    issues.append(
                        "unit does not have exactly one successful Questioner "
                        f"generation: {unit.compiler_unit_id}"
                    )
                    continue
                actual_counts = (
                    result.produced_count,
                    result.accepted_count,
                    result.rejected_count,
                )
                if actual_counts != expected_counts:
                    issues.append(
                        "Questioner stage accounting differs from generation/gate facts: "
                        f"{unit.compiler_unit_id}"
                    )
            elif result.stage in {"planner", "dry_run"} and compiler.compiler_schema_version in {
                "0.4.0",
                "0.5.0",
                "0.6.0",
                "0.7.0",
                "0.8.0",
                "0.9.0",
            }:
                expected_stage_counts = await compiler_v04_stage_expected_counts(
                    session,
                    run_id=run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                    stage=result.stage,
                    fact_counts=fact_counts,
                )
                actual_counts = (
                    result.produced_count,
                    result.accepted_count,
                    result.rejected_count,
                )
                if expected_stage_counts is None or actual_counts != expected_stage_counts:
                    issues.append(
                        f"{result.stage} stage accounting differs from atomic facts: "
                        f"{unit.compiler_unit_id}"
                    )
            elif (
                result.stage in required_stages
                and result.produced_count != fact_counts[result.stage]
            ):
                issues.append(
                    "stage produced_count differs from persisted facts: "
                    f"{unit.compiler_unit_id}/{result.stage}"
                )
        if compiler.compiler_schema_version in {
            "0.2.0",
            "0.3.0",
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
            "0.9.0",
        }:
            support_count = sum(item.role == "support" for item in members)
            link_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(CompilerUnitLinkModel)
                    .where(CompilerUnitLinkModel.compiler_unit_id == unit.compiler_unit_id)
                )
                or 0
            )
            if link_count != support_count:
                issues.append(f"unit support/link counts differ: {unit.compiler_unit_id}")
            ungated = await session.scalar(
                select(QBankQuestionModel.qbank_id)
                .where(
                    QBankQuestionModel.run_id == run_id,
                    QBankQuestionModel.compiler_unit_id == unit.compiler_unit_id,
                    ~QBankQuestionModel.qbank_id.in_(
                        select(QuestionerCandidateGateResultModel.qbank_id).where(
                            QuestionerCandidateGateResultModel.run_id == run_id
                        )
                    ),
                )
                .limit(1)
            )
            if ungated is not None:
                issues.append(f"unit has an ungated question: {unit.compiler_unit_id}")
        if compiler.compiler_schema_version in {
            "0.3.0",
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
            "0.9.0",
        }:
            from pmmc.infrastructure.database.questioner_repositories import (
                questioner_v3_unit_issues,
            )

            issues.extend(
                f"{unit.compiler_unit_id}: {issue}"
                for issue in await questioner_v3_unit_issues(
                    session,
                    run_id=run_id,
                    compiler_unit_id=unit.compiler_unit_id,
                )
            )
        if compiler.compiler_schema_version in {
            "0.4.0",
            "0.5.0",
            "0.6.0",
            "0.7.0",
            "0.8.0",
            "0.9.0",
        }:
            for stage in ("planner", "dry_run"):
                stage_job_count = int(
                    await session.scalar(
                        select(func.count())
                        .select_from(CompilerStageJobModel)
                        .where(
                            CompilerStageJobModel.run_id == run_id,
                            CompilerStageJobModel.compiler_unit_id == unit.compiler_unit_id,
                            CompilerStageJobModel.stage == stage,
                            CompilerStageJobModel.round_id == 0,
                            CompilerStageJobModel.status == "succeeded",
                        )
                    )
                    or 0
                )
                if stage_job_count != 1:
                    issues.append(
                        f"unit has no unique succeeded {stage} stage job: {unit.compiler_unit_id}"
                    )
            eligible_ids = list(
                await session.scalars(
                    select(QBankQuestionModel.qbank_id)
                    .join(
                        QuestionerCandidateGateResultModel,
                        and_(
                            QuestionerCandidateGateResultModel.qbank_id
                            == QBankQuestionModel.qbank_id,
                            QuestionerCandidateGateResultModel.run_id == QBankQuestionModel.run_id,
                        ),
                    )
                    .where(
                        QBankQuestionModel.run_id == run_id,
                        QBankQuestionModel.compiler_unit_id == unit.compiler_unit_id,
                        QuestionerCandidateGateResultModel.status == "eligible_for_planning",
                    )
                )
            )
            outcome_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(PlannerCandidateOutcomeModel)
                    .where(
                        PlannerCandidateOutcomeModel.run_id == run_id,
                        PlannerCandidateOutcomeModel.qbank_id.in_(eligible_ids),
                    )
                )
                or 0
            )
            if outcome_count != len(eligible_ids):
                issues.append(
                    f"unit Planner terminal outcomes are incomplete: {unit.compiler_unit_id}"
                )
    if compiler.compiler_schema_version in {
        "0.2.0",
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
    }:
        identity = (
            compiler.questioner_config_hash,
            compiler.questioner_model_alias,
            compiler.questioner_model_revision,
            compiler.questioner_prompt_hash,
            compiler.candidate_gate_version,
            compiler.qbank_matcher_version,
        )
        if any(value is None for value in identity):
            issues.append("Questioner compiler identity is incomplete")
        indexes = list(
            await session.scalars(
                select(QBankIndexManifestModel).where(
                    QBankIndexManifestModel.compiler_run_id == run_id,
                    QBankIndexManifestModel.status == "ready",
                )
            )
        )
        if len(indexes) != 1:
            issues.append("Questioner run requires exactly one ready QBank index")
        else:
            collection_ids = (
                indexes[0].text_question_collection_id,
                indexes[0].text_paraphrase_collection_id,
                indexes[0].multimodal_question_collection_id,
                indexes[0].multimodal_paraphrase_collection_id,
            )
            for collection_id in collection_ids:
                collection = await session.get(EmbeddingCollectionModel, collection_id)
                if collection is None:
                    issues.append("QBank index references a missing embedding collection")
                    break
                if collection.status != "ready":
                    issues.append("QBank index references a non-ready embedding collection")
                    break
    if compiler.compiler_schema_version in {
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
    }:
        planner_identity = (
            compiler.questioner_freeze_manifest_hash,
            compiler.planner_config_hash,
            compiler.planner_model_alias,
            compiler.planner_model_revision,
            compiler.planner_prompt_version,
            compiler.planner_prompt_hash,
            compiler.planner_renderer_version,
            compiler.planner_parser_version,
            compiler.planner_schema_version,
            compiler.program_policy_version,
            compiler.dry_run_policy_version,
            compiler.memory_index_representation_version,
            compiler.memory_text_model_alias,
            compiler.memory_text_model_revision,
            compiler.memory_multimodal_model_alias,
            compiler.memory_multimodal_model_revision,
        )
        if any(value is None for value in planner_identity):
            issues.append("Planner compiler identity is incomplete")
        memory_indexes = list(
            await session.scalars(
                select(MemoryOperatorIndexManifestModel).where(
                    MemoryOperatorIndexManifestModel.run_id == run_id,
                    MemoryOperatorIndexManifestModel.status == "ready",
                )
            )
        )
        if len(memory_indexes) != 1:
            issues.append("Planner run requires exactly one ready memory operator index")
        else:
            for collection_id in (
                memory_indexes[0].text_collection_id,
                memory_indexes[0].visual_collection_id,
            ):
                collection = await session.get(EmbeddingCollectionModel, collection_id)
                if (
                    collection is None
                    or collection.status != "ready"
                    or collection.dimension != 4096
                ):
                    issues.append("memory operator index references an invalid collection")
                    break
        eligible_question_count = int(
            await session.scalar(
                select(func.count())
                .select_from(QuestionerCandidateGateResultModel)
                .where(
                    QuestionerCandidateGateResultModel.run_id == run_id,
                    QuestionerCandidateGateResultModel.status == "eligible_for_planning",
                )
            )
            or 0
        )
        vector_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ProgramQueryEmbeddingModel)
                .where(ProgramQueryEmbeddingModel.run_id == run_id)
            )
            or 0
        )
        if vector_count != eligible_question_count * 2:
            issues.append("program query-vector denominator is incomplete")
        programs = list(
            await session.scalars(
                select(MemoryProgramModel).where(MemoryProgramModel.run_id == run_id)
            )
        )
        program_by_id = {item.plan_id: item for item in programs}
        revision_by_plan = (
            {
                item.revised_plan_id: item
                for item in await session.scalars(
                    select(PlannerRevisionLineageModel).where(
                        PlannerRevisionLineageModel.run_id == run_id
                    )
                )
            }
            if compiler.compiler_schema_version in {"0.6.0", "0.7.0", "0.8.0", "0.9.0"}
            else {}
        )
        feedback_by_id = (
            {
                item.feedback_id: item
                for item in await session.scalars(
                    select(DoubterAssessmentModel).where(DoubterAssessmentModel.run_id == run_id)
                )
            }
            if revision_by_plan
            else {}
        )
        for program in programs:
            outcome_status = await session.scalar(
                select(PlannerCandidateOutcomeModel.status).where(
                    PlannerCandidateOutcomeModel.run_id == run_id,
                    PlannerCandidateOutcomeModel.plan_id == program.plan_id,
                )
            )
            revision = revision_by_plan.get(program.plan_id)
            issues.extend(
                compiler_program_outcome_issues(
                    program,
                    outcome_status=outcome_status,
                    revision=revision,
                    source_program=(
                        program_by_id.get(revision.source_plan_id) if revision else None
                    ),
                    source_feedback=(
                        feedback_by_id.get(revision.source_feedback_id) if revision else None
                    ),
                )
            )
            valid_kinds = set(
                await session.scalars(
                    select(ProgramValidationResultModel.validation_kind).where(
                        ProgramValidationResultModel.plan_id == program.plan_id,
                        ProgramValidationResultModel.valid,
                    )
                )
            )
            if not {"static", "policy"} <= valid_kinds:
                issues.append(f"program validation facts are incomplete: {program.plan_id}")
            lineage_rows = list(
                await session.scalars(
                    select(ProgramValidationResultModel.valid).where(
                        ProgramValidationResultModel.plan_id == program.plan_id,
                        ProgramValidationResultModel.validation_kind == "lineage",
                    )
                )
            )
            assessment_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ProgramDryRunAssessmentModel)
                    .where(ProgramDryRunAssessmentModel.plan_id == program.plan_id)
                )
                or 0
            )
            if outcome_status == "lineage_rejected":
                if lineage_rows != [False] or assessment_count != 0:
                    issues.append(f"lineage-rejected program facts conflict: {program.plan_id}")
            elif outcome_status == "query_vector_unavailable":
                if assessment_count != 0:
                    issues.append(f"vector-unavailable program has a dry-run: {program.plan_id}")
            elif lineage_rows != [True] or assessment_count != 1:
                issues.append(f"program dry-run assessment is missing: {program.plan_id}")
    if compiler.compiler_schema_version in {"0.6.0", "0.7.0", "0.8.0", "0.9.0"}:
        doubter_identity = (
            compiler.doubter_config_hash,
            compiler.doubter_model_alias,
            compiler.doubter_model_revision,
            compiler.doubter_prompt_version,
            compiler.doubter_prompt_hash,
            compiler.doubter_renderer_version,
            compiler.doubter_parser_version,
            compiler.doubter_schema_version,
            compiler.r0_input_policy_version,
            compiler.routing_projection_version,
        )
        if any(value is None for value in doubter_identity):
            issues.append("Doubter compiler identity is incomplete")
        if compiler.active_round != 2 or compiler.max_round != 3:
            issues.append("verified QBank run must freeze active r2 and maximum r3")

        verified_eligible_ids = set(
            await session.scalars(
                select(QuestionerCandidateGateResultModel.qbank_id).where(
                    QuestionerCandidateGateResultModel.run_id == run_id,
                    QuestionerCandidateGateResultModel.status == "eligible_for_planning",
                )
            )
        )
        eligible_count = len(verified_eligible_ids)
        r0_manifests = list(
            await session.scalars(
                select(R0InputManifestModel).where(R0InputManifestModel.run_id == run_id)
            )
        )
        if (
            len(r0_manifests) != 1
            or not r0_manifests[0].hard_gate_passed
            or r0_manifests[0].candidate_count != eligible_count
        ):
            issues.append("verified QBank run has no valid immutable R0 input")

        projection_manifests = list(
            await session.scalars(
                select(QBankRoundProjectionManifestModel)
                .where(QBankRoundProjectionManifestModel.run_id == run_id)
                .order_by(QBankRoundProjectionManifestModel.round_id)
            )
        )
        if [item.round_id for item in projection_manifests] != [0, 1, 2, 3]:
            issues.append("verified QBank round projections are incomplete")
        for manifest in projection_manifests:
            projection_member_ids = set(
                await session.scalars(
                    select(QBankRoundProjectionMemberModel.qbank_id).where(
                        QBankRoundProjectionMemberModel.projection_id == manifest.projection_id
                    )
                )
            )
            member_count = len(projection_member_ids)
            if (
                projection_member_ids != verified_eligible_ids
                or manifest.member_count != member_count
            ):
                issues.append(f"round projection denominator differs at r{manifest.round_id}")
            if manifest.round_id == 3 and manifest.pending_count:
                issues.append("maximum-round projection contains pending candidates")

        decisions = list(
            await session.scalars(
                select(QBankDecisionModel).where(QBankDecisionModel.run_id == run_id)
            )
        )
        if (
            len(decisions) != eligible_count
            or {item.qbank_id for item in decisions} != verified_eligible_ids
        ):
            issues.append("active-r2 QBank decision denominator is incomplete")
        accepted_ids = {item.qbank_id for item in decisions if item.accepted}
        routing_manifests = list(
            await session.scalars(
                select(RoutingProjectionManifestModel).where(
                    RoutingProjectionManifestModel.run_id == run_id
                )
            )
        )
        if len(routing_manifests) != 1:
            issues.append("verified QBank routing projection is missing")
        else:
            routing = routing_manifests[0]
            routing_members = list(
                await session.scalars(
                    select(RoutingProjectionMemberModel).where(
                        RoutingProjectionMemberModel.routing_projection_id
                        == routing.routing_projection_id
                    )
                )
            )
            if (
                routing.active_round != 2
                or routing.max_round != 3
                or routing.member_count != len(routing_members)
                or {item.qbank_id for item in routing_members} != accepted_ids
            ):
                issues.append("runtime routing projection conflicts with active-r2 decisions")

        assessments = list(
            await session.scalars(
                select(DoubterAssessmentModel).where(DoubterAssessmentModel.run_id == run_id)
            )
        )
        terminal_ids = {
            item.qbank_id for item in assessments if item.action in {"pass", "reject_candidate"}
        }
        candidate_jobs = list(
            await session.scalars(
                select(CandidateStageJobModel).where(CandidateStageJobModel.run_id == run_id)
            )
        )
        unavailable_revision_targets = {
            (item.qbank_id, item.round_id)
            for item in candidate_jobs
            if item.stage == "planner_revision"
            and item.status == "cancelled"
            and (item.metadata_json or {}).get("terminal_reason") == "planner_revision_unavailable"
        }
        terminal_ids.update(
            item.qbank_id
            for item in assessments
            if item.action == "revise_program"
            and (item.qbank_id, item.round_id + 1) in unavailable_revision_targets
        )
        planner_outcomes = list(
            await session.scalars(
                select(PlannerCandidateOutcomeModel).where(
                    PlannerCandidateOutcomeModel.run_id == run_id
                )
            )
        )
        if not compiler_doubter_terminal_accounting(
            verified_eligible_ids, terminal_ids, programs, planner_outcomes
        ):
            issues.append("Doubter candidate terminal outcomes are incomplete")
        invalid_jobs = [
            item
            for item in candidate_jobs
            if item.status != "succeeded"
            and not (
                item.stage == "planner_revision"
                and item.status == "cancelled"
                and (item.metadata_json or {}).get("terminal_reason")
                == "planner_revision_unavailable"
                and any(
                    assessment.qbank_id == item.qbank_id
                    and assessment.round_id + 1 == item.round_id
                    and assessment.action == "revise_program"
                    for assessment in assessments
                )
            )
        ]
        if invalid_jobs:
            issues.append("verified QBank run has unfinished candidate stage jobs")
    invalid_execution = await session.scalar(
        select(ProgramExecutionModel.execution_id)
        .where(
            ProgramExecutionModel.source_compiler_run_id == run_id,
            ProgramExecutionModel.status == "success",
        )
        .where(
            ~ProgramExecutionModel.execution_id.in_(select(ProgramExecutionStepModel.execution_id))
        )
        .limit(1)
    )
    if invalid_execution is not None:
        issues.append("a successful program execution has no persisted step")
    issue_type_counts = summarize_compiler_verification_issues(issues)
    if issues:
        print(
            "COMPILER_RUN_VERIFY_FAILURE "
            + json.dumps(
                {
                    "compiler_schema_version": compiler.compiler_schema_version,
                    "issue_count": len(issues),
                    "issue_type_counts": issue_type_counts,
                    "target_stage": compiler.target_stage,
                    "stage_accounting_scope": "planner_and_dry_run_round_0",
                    "program_validation_scope": "all_persisted_rounds",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return {
        **summary,
        "valid": not issues,
        "issues": issues,
        "issue_type_counts": issue_type_counts,
    }


async def compiler_unit_fact_counts(
    session: AsyncSession,
    *,
    run_id: str,
    compiler_unit_id: str,
) -> dict[str, int]:
    """Count the immutable facts represented by each compiler stage result."""

    statements = {
        "questioner": select(func.count())
        .select_from(QBankQuestionModel)
        .where(
            QBankQuestionModel.run_id == run_id,
            QBankQuestionModel.compiler_unit_id == compiler_unit_id,
        ),
        "planner": select(func.count())
        .select_from(MemoryProgramModel)
        .where(
            MemoryProgramModel.run_id == run_id,
            MemoryProgramModel.compiler_unit_id == compiler_unit_id,
        ),
        "dry_run": select(func.count())
        .select_from(ProgramExecutionModel)
        .where(
            ProgramExecutionModel.source_compiler_run_id == run_id,
            ProgramExecutionModel.compiler_unit_id == compiler_unit_id,
            ProgramExecutionModel.purpose == "compiler_dry_run",
        ),
        "doubter": select(func.count())
        .select_from(DoubterFeedbackModel)
        .where(
            DoubterFeedbackModel.run_id == run_id,
            DoubterFeedbackModel.compiler_unit_id == compiler_unit_id,
        ),
        "decision": select(func.count())
        .select_from(QBankDecisionModel)
        .where(
            QBankDecisionModel.run_id == run_id,
            QBankDecisionModel.compiler_unit_id == compiler_unit_id,
        ),
    }
    counts = {
        stage: int(await session.scalar(statement) or 0) for stage, statement in statements.items()
    }
    counts["planner_r0"] = int(
        await session.scalar(statements["planner"].where(MemoryProgramModel.round_id == 0)) or 0
    )
    return counts
