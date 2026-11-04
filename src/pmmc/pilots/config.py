"""Strict configuration for the Questioner quality pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from pmmc.schemas.base import (
    BaseSchema,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Probability,
)


class QuestionerPilotConfig(BaseSchema):
    protocol_version: NonEmptyStr = "questioner-quality-pilot-v1"
    partition_protocol: NonEmptyStr = "conversation-stratified-five-fold-v1"
    development_fold: NonNegativeInt = 0
    fold_count: PositiveInt = 5
    partition_seed: NonNegativeInt = 0
    queries_per_dataset: PositiveInt = 50
    candidate_review_count: PositiveInt = 120
    coverage_review_count: PositiveInt = 40
    secondary_review_fraction: Probability = 0.20
    question_budgets: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 4, 8])
    match_ks: list[PositiveInt] = Field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    scientific_gate: NonEmptyStr = "manual_approval"
    bootstrap_seed: NonNegativeInt = 0
    bootstrap_repetitions: PositiveInt = 2000
    underpowered_stratum_size: PositiveInt = 5
    eligible_candidate_reviews_per_dataset: PositiveInt = 40
    rejected_candidate_reviews_per_dataset: PositiveInt = 10
    duplicate_candidate_reviews_per_dataset: PositiveInt = 10
    coverage_failure_reviews_per_dataset: PositiveInt = 10
    matching_failure_reviews_per_dataset: PositiveInt = 5
    memgallery_max_queries_per_scenario: PositiveInt = 15
    memlens_queries_per_category: PositiveInt = 4
    paired_step7_queries_per_dataset: PositiveInt = 12
    candidate_audit_count: PositiveInt = 120
    coverage_audit_count: PositiveInt = 40

    @model_validator(mode="after")
    def _validate_protocol(self) -> QuestionerPilotConfig:
        supported = {
            "questioner-quality-pilot-v1": "conversation-stratified-five-fold-v1",
            "questioner-quality-pilot-v2": "scenario-balanced-five-fold-v2",
            "questioner-quality-pilot-v3": "scenario-balanced-five-fold-v2",
            "questioner-quality-pilot-v4": "scenario-balanced-five-fold-v2",
        }
        if self.protocol_version not in supported:
            raise ValueError("unsupported Questioner pilot protocol")
        if self.partition_protocol != supported[self.protocol_version] and not (
            self.protocol_version in {"questioner-quality-pilot-v3", "questioner-quality-pilot-v4"}
            and self.partition_protocol == "scenario-balanced-five-fold-v3"
        ):
            raise ValueError("Questioner pilot and partition protocols must match")
        if self.fold_count != 5 or self.development_fold != 0:
            raise ValueError("pilot v1 freezes development to fold 0 of five folds")
        expected_gate = (
            "research_lead_approval"
            if self.protocol_version in {
                "questioner-quality-pilot-v3",
                "questioner-quality-pilot-v4",
            }
            else "manual_approval"
        )
        if self.scientific_gate != expected_gate:
            raise ValueError(f"{self.protocol_version} requires {expected_gate}")
        if self.question_budgets != [1, 2, 4, 8]:
            raise ValueError("pilot v1 freezes question budgets to 1/2/4/8")
        if self.match_ks != [1, 3, 5, 10, 20, 50]:
            raise ValueError("pilot v1 freezes K to 1/3/5/10/20/50")
        expected_candidates = 2 * (
            self.eligible_candidate_reviews_per_dataset
            + self.rejected_candidate_reviews_per_dataset
            + self.duplicate_candidate_reviews_per_dataset
        )
        if self.candidate_review_count != expected_candidates:
            raise ValueError("candidate review quotas must reconcile with the total")
        if self.coverage_review_count % 2:
            raise ValueError("coverage review count must divide evenly across two datasets")
        if self.protocol_version == "questioner-quality-pilot-v4":
            if self.secondary_review_fraction != 0:
                raise ValueError("pilot v4 does not create blind human review assignments")
            if (
                self.queries_per_dataset != 16
                or self.paired_step7_queries_per_dataset != 12
                or self.memlens_queries_per_category != 4
                or self.memgallery_max_queries_per_scenario != 4
            ):
                raise ValueError(
                    "pilot v4 freezes the paired 12 plus four-category/scenario expansion"
                )
            if self.candidate_audit_count != 120 or self.coverage_audit_count != 32:
                raise ValueError("pilot v4 freezes the 120/32 audit denominator")
        elif self.protocol_version == "questioner-quality-pilot-v3":
            if self.secondary_review_fraction != 0:
                raise ValueError("pilot v3 does not create blind human review assignments")
            if self.candidate_audit_count != 120 or self.coverage_audit_count != 40:
                raise ValueError("pilot v3 freezes the 120/40 audit denominator")
        elif not 0 < self.secondary_review_fraction <= 1:
            raise ValueError("secondary review fraction must be greater than zero")
        return self

    @property
    def secondary_review_count(self) -> int:
        return round(
            (self.candidate_review_count + self.coverage_review_count)
            * self.secondary_review_fraction
        )

    @property
    def resolved_hash(self) -> str:
        value = self.model_dump(mode="json")
        if self.protocol_version == "questioner-quality-pilot-v1":
            value.pop("memgallery_max_queries_per_scenario", None)
        if self.protocol_version != "questioner-quality-pilot-v4":
            value.pop("memlens_queries_per_category", None)
            value.pop("paired_step7_queries_per_dataset", None)
        if self.protocol_version not in {
            "questioner-quality-pilot-v3",
            "questioner-quality-pilot-v4",
        }:
            value.pop("candidate_audit_count", None)
            value.pop("coverage_audit_count", None)
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_questioner_pilot_config(path: Path) -> QuestionerPilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Questioner pilot config root must be a mapping")
    return QuestionerPilotConfig.model_validate(raw)
