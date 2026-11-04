"""Ranking score schemas with explicit raw and normalized semantics."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from pmmc.schemas.base import BaseSchema, FiniteFloat, JsonDict, NonEmptyStr, Probability


class ScoreDirection(StrEnum):
    """Whether larger or smaller raw values rank higher."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class RankingScore(BaseSchema):
    """A ranking score that preserves retrieval-native and normalized values."""

    score_type: NonEmptyStr
    raw_value: FiniteFloat
    normalized_value: Probability | None = None
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER
    normalization_method: NonEmptyStr | None = None
    normalization_version: NonEmptyStr | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_normalization(self) -> RankingScore:
        has_normalized_value = self.normalized_value is not None
        has_method = self.normalization_method is not None
        if has_normalized_value != has_method:
            raise ValueError("normalized_value and normalization_method must be provided together")
        if self.normalization_version is not None and not has_method:
            raise ValueError("normalization_version requires normalization_method")
        return self
