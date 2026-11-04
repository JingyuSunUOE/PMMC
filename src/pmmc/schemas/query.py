"""Runtime-safe queries and evaluation-only gold annotations."""

from __future__ import annotations

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    QuestionType,
    ensure_unique_strings,
)
from pmmc.schemas.memory import ImageRef


class GoldEvidence(BaseSchema):
    """Evaluation-only evidence metadata."""

    evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    session_ids: list[NonEmptyStr] = Field(default_factory=list)
    image_ids: list[NonEmptyStr] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ids(self) -> GoldEvidence:
        for field_name in ("evidence_ids", "memory_ids", "session_ids", "image_ids"):
            ensure_unique_strings(getattr(self, field_name), field_name)
        return self


class QueryInput(BaseSchema):
    """The only query representation visible to runtime PMMC methods."""

    query_id: NonEmptyStr
    dataset: DatasetName
    conversation_id: NonEmptyStr
    raw_query: NonEmptyStr
    query_type: QuestionType = QuestionType.UNKNOWN
    query_subtype: NonEmptyStr | None = None
    query_image_refs: list[ImageRef] = Field(default_factory=list)
    available_after_session: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_image_refs(self) -> QueryInput:
        ensure_unique_strings(
            [image_ref.image_id for image_ref in self.query_image_refs],
            "query_image_refs.image_id",
        )
        return self


class QueryGold(BaseSchema):
    """Gold answer and evidence available only to evaluation code."""

    answer: NonEmptyStr | None = None
    evidence: GoldEvidence = Field(default_factory=GoldEvidence)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_annotation(self) -> QueryGold:
        has_evidence = any(
            (
                self.evidence.evidence_ids,
                self.evidence.memory_ids,
                self.evidence.session_ids,
                self.evidence.image_ids,
            )
        )
        if self.answer is None and not has_evidence:
            raise ValueError("query gold must contain an answer or evidence annotation")
        return self


class QueryRecord(BaseSchema):
    """A complete query record with an explicit runtime/evaluation boundary."""

    query: QueryInput
    gold: QueryGold | None = None
    metadata: JsonDict = Field(default_factory=dict)
