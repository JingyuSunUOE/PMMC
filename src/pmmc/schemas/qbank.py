"""Question-bank schemas."""

from __future__ import annotations

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    DatasetName,
    JsonDict,
    Modality,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    QuestionType,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.doubter import QBankRouteClass
from pmmc.schemas.questioner import (
    QuestionerAnswerType,
    QuestionerCandidateOrigin,
    QuestionReasoningType,
)
from pmmc.schemas.score import RankingScore


class QuestionCandidate(BaseSchema):
    """Questioner output for a prospective future question.

    The expected answer is compiler-only verification context. Query-time code
    must consume ``QBankRoutingItem`` instead of this model.
    """

    qbank_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr | None = None
    compiler_unit_id: NonEmptyStr | None = None
    generation_id: NonEmptyStr | None = None
    producer_model_call_id: NonEmptyStr | None = None
    question_hash: NonEmptyStr | None = None
    conversation_id: NonEmptyStr
    created_after_session: NonEmptyStr
    canonical_question: NonEmptyStr
    paraphrases: list[NonEmptyStr] = Field(default_factory=list)
    question_type: QuestionType = QuestionType.UNKNOWN
    reasoning_type: QuestionReasoningType = QuestionReasoningType.SINGLE_HOP
    expected_answer: NonEmptyStr | None = None
    source_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    required_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    evidence_requirement: list[NonEmptyStr] = Field(default_factory=list)
    hop_count: PositiveInt = 1
    generation_rank: PositiveInt = 1
    questioner_confidence: Probability = 0.0
    candidate_content_hash: NonEmptyStr | None = None
    opportunity_id: NonEmptyStr | None = None
    candidate_origin: QuestionerCandidateOrigin = QuestionerCandidateOrigin.GENERATED
    candidate_version: int = Field(default=0, ge=0, le=1)
    supersedes_qbank_id: NonEmptyStr | None = None
    answer_type: QuestionerAnswerType | None = None
    route_class: QBankRouteClass = QBankRouteClass.ANSWERABLE
    accepted: bool | None = None
    rejection_reason: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_candidate(self) -> QuestionCandidate:
        if not self.required_modalities:
            raise ValueError("required_modalities must contain at least one item")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicate values")
        for field_name in ("paraphrases", "source_memory_ids", "evidence_ids"):
            ensure_unique_strings(getattr(self, field_name), field_name)

        if self.accepted is False and self.rejection_reason is None:
            raise ValueError("rejected candidates require rejection_reason")
        if self.accepted is not False and self.rejection_reason is not None:
            raise ValueError("rejection_reason is only valid for rejected candidates")
        if self.compiler_unit_id is not None and (
            self.snapshot_id is None or self.question_hash is None
        ):
            raise ValueError(
                "compiler-scoped question candidates require snapshot_id and question_hash"
            )
        if self.generation_id is not None and self.compiler_unit_id is None:
            raise ValueError("generation_id requires compiler scope")
        revised = self.candidate_origin is QuestionerCandidateOrigin.REVISED
        if revised != (self.candidate_version == 1):
            raise ValueError("revised candidates require candidate_version=1")
        if revised != (self.supersedes_qbank_id is not None):
            raise ValueError("revised candidates require supersession lineage")
        if self.opportunity_id is not None and self.compiler_unit_id is None:
            raise ValueError("opportunity_id requires compiler scope")
        return self


class QBankRoutingItem(BaseSchema):
    """Query-time-safe view of an accepted question-bank item."""

    qbank_id: NonEmptyStr
    snapshot_id: NonEmptyStr | None = None
    conversation_id: NonEmptyStr | None = None
    available_after_session: NonEmptyStr | None = None
    canonical_question: NonEmptyStr
    paraphrases: list[NonEmptyStr] = Field(default_factory=list)
    question_type: QuestionType = QuestionType.UNKNOWN
    required_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    memory_program_id: NonEmptyStr
    program_hash: NonEmptyStr | None = None
    operator_set_version: NonEmptyStr | None = None
    verified: bool = False
    verifier_score: Probability | None = None
    route_class: QBankRouteClass = QBankRouteClass.ANSWERABLE
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_routing_item(self) -> QBankRoutingItem:
        ensure_unique_strings(self.paraphrases, "paraphrases")
        if not self.required_modalities:
            raise ValueError("required_modalities must contain at least one item")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicate values")
        return self


class VerifiedQBankRoutingItemV2(BaseSchema):
    """Compiler 0.7 runtime projection with no answer or support locators."""

    routing_projection_id: NonEmptyStr
    compiler_run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    dataset: DatasetName
    conversation_id: NonEmptyStr
    available_after_session: NonEmptyStr
    active_round: NonNegativeInt = 2
    decision_hash: NonEmptyStr
    qbank_id: NonEmptyStr
    generation_rank: PositiveInt
    route_class: QBankRouteClass
    canonical_question: NonEmptyStr
    paraphrases: list[NonEmptyStr] = Field(default_factory=list)
    question_type: QuestionType
    required_modalities: list[Modality]
    plan_id: NonEmptyStr
    program_hash: NonEmptyStr
    operator_set_version: NonEmptyStr
    requirement_contract: JsonDict
    lexical_collection_id: NonEmptyStr
    text_collection_id: NonEmptyStr
    multimodal_collection_id: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_verified_projection(self) -> VerifiedQBankRoutingItemV2:
        if self.active_round != 2:
            raise ValueError("verified QBank routing v2 freezes active round two")
        ensure_unique_strings(self.paraphrases, "paraphrases")
        if self.canonical_question.casefold() in {item.casefold() for item in self.paraphrases}:
            raise ValueError("canonical question cannot be repeated as a paraphrase")
        if not self.required_modalities or len(self.required_modalities) != len(
            set(self.required_modalities)
        ):
            raise ValueError("verified routing requires unique modalities")
        if (
            len(
                {
                    self.lexical_collection_id,
                    self.text_collection_id,
                    self.multimodal_collection_id,
                }
            )
            != 3
        ):
            raise ValueError("verified routing collection identities must be distinct")
        forbidden = {
            "expected_answer",
            "expected_evidence",
            "evidence_ids",
            "source_memory_ids",
            "gold",
            "doubter_feedback",
        }
        if forbidden & set(self.requirement_contract):
            raise ValueError("runtime requirement contract contains private compiler facts")
        return self


class VerifiedQBankRoutingItemV3(VerifiedQBankRoutingItemV2):
    """Compiler 0.8 runtime projection consumed only by the Step 7B executor."""

    routing_schema_version: NonEmptyStr = "verified-qbank-routing-v3"
    answer_type: NonEmptyStr

    @model_validator(mode="after")
    def _validate_v3(self) -> VerifiedQBankRoutingItemV3:
        if self.operator_set_version != "pmmc-operators-v6":
            raise ValueError("verified QBank routing v3 requires operator set v6")
        return self


class VerifiedQBankRoutingItemV4(VerifiedQBankRoutingItemV2):
    """Compiler 0.9 exact-round membership, with that member's frozen program."""

    routing_schema_version: NonEmptyStr = "verified-qbank-routing-v4"
    answer_type: NonEmptyStr
    projection_round: NonNegativeInt = Field(le=3)
    program_round: NonNegativeInt = Field(le=3)

    @model_validator(mode="after")
    def _validate_v4(self) -> VerifiedQBankRoutingItemV4:
        if self.routing_schema_version != "verified-qbank-routing-v4":
            raise ValueError("verified routing v4 schema identity mismatch")
        if self.operator_set_version != "pmmc-operators-v7":
            raise ValueError("verified QBank routing v4 requires operator set v7")
        if self.program_round > self.projection_round:
            raise ValueError("routing projection cannot use a future program")
        forbidden = {
            "expected_answer",
            "expected_evidence",
            "expected_evidence_ids",
            "evidence_ids",
            "source_memory_ids",
            "source_support",
            "support_quote",
            "claim_text",
            "gold",
            "doubter_feedback",
            "doubter_assessment",
            "feedback_id",
            "effective_feedback_id",
        }

        def private(value: object) -> bool:
            if isinstance(value, dict):
                return bool(forbidden & set(value)) or any(private(v) for v in value.values())
            return isinstance(value, list) and any(private(v) for v in value)

        if private(self.requirement_contract) or private(self.metadata):
            raise ValueError("runtime routing contains private compiler facts")
        return self


class QBankItem(QuestionCandidate):
    """Persisted accepted or rejected question-bank item."""

    accepted: bool
    memory_program_id: NonEmptyStr | None = None
    verified: bool = False
    verifier_score: Probability | None = None

    def to_routing_item(self) -> QBankRoutingItem:
        """Return a runtime-safe routing view for an accepted item."""

        if not self.accepted:
            raise ValueError("rejected qbank items cannot be used for routing")
        if self.memory_program_id is None:
            raise ValueError("routing requires memory_program_id")
        return QBankRoutingItem(
            qbank_id=self.qbank_id,
            snapshot_id=self.snapshot_id,
            conversation_id=self.conversation_id,
            available_after_session=self.created_after_session,
            canonical_question=self.canonical_question,
            paraphrases=self.paraphrases,
            question_type=self.question_type,
            required_modalities=self.required_modalities,
            memory_program_id=self.memory_program_id,
            verified=self.verified,
            verifier_score=self.verifier_score,
            route_class=self.route_class,
        )


class QBankMatchCandidate(BaseSchema):
    """A query-time candidate produced from a safe qbank routing item."""

    qbank_id: NonEmptyStr
    memory_program_id: NonEmptyStr
    rank: PositiveInt
    canonical_question: NonEmptyStr | None = None
    question_type: QuestionType = QuestionType.UNKNOWN
    lexical_score: RankingScore | None = None
    semantic_score: RankingScore | None = None
    multimodal_score: RankingScore | None = None
    reranker_score: RankingScore | None = None
    final_score: RankingScore | None = None
    match_confidence: Probability | None = None
    raw_rrf: NonNegativeFloat | None = None
    generation_rank: PositiveInt | None = None
    accepted_round: NonNegativeInt | None = None
    metadata: JsonDict = Field(default_factory=dict)
