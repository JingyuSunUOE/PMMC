"""Strict Step 6B Questioner and coverage configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, model_serializer, model_validator

from pmmc.schemas import (
    BaseSchema,
    CompilationUnitRelationType,
    CoverageGateMode,
    DatasetName,
)
from pmmc.schemas.base import NonEmptyStr, PositiveInt

DEFAULT_RELATION_ORDER = (
    CompilationUnitRelationType.ADJACENT_SESSION,
    CompilationUnitRelationType.SHARED_ENTITY,
    CompilationUnitRelationType.SHARED_IMAGE,
    CompilationUnitRelationType.PROFILE_PREFERENCE,
    CompilationUnitRelationType.TEMPORAL_NEIGHBOR,
)

# W is an opt-in scientific policy on the native v3.6 wire contract.
EVIDENCE_FIRST_VERSION = "questioner-evidence-first-opportunities-v1"

QUESTIONER_PROTOCOL_VERSIONS = {
    "pmmc-questioner-v1": {
        "prompt_version": "pmmc-questioner-v1",
        "renderer_version": "questioner-multimodal-renderer-v1",
        "parser_version": "questioner-json-parser-v1",
    },
    "pmmc-questioner-v2": {
        "prompt_version": "pmmc-questioner-v2",
        "renderer_version": "questioner-multimodal-renderer-v2",
        "parser_version": "questioner-json-parser-v2",
    },
    "pmmc-questioner-v2.1": {
        "prompt_version": "pmmc-questioner-v2",
        "renderer_version": "questioner-multimodal-renderer-v2",
        "parser_version": "questioner-json-parser-v2",
    },
    "pmmc-questioner-v3": {
        "prompt_version": "pmmc-questioner-v3",
        "renderer_version": "questioner-multimodal-renderer-v3",
        "parser_version": "questioner-json-parser-v3",
    },
    "pmmc-questioner-v3.1": {
        "prompt_version": "pmmc-questioner-v3.1",
        "renderer_version": "questioner-safe-alias-renderer-v3.1",
        "parser_version": "questioner-json-parser-v3.1",
    },
    "pmmc-questioner-v3.2": {
        "prompt_version": "pmmc-questioner-v3.2",
        "renderer_version": "questioner-safe-alias-renderer-v3.2",
        "parser_version": "questioner-json-parser-v3.2",
    },
    "pmmc-questioner-v3.3": {
        "prompt_version": "pmmc-questioner-v3.3",
        "renderer_version": "questioner-safe-alias-renderer-v3.3",
        "parser_version": "questioner-json-parser-v3.3",
    },
    "pmmc-questioner-v3.4": {
        "prompt_version": "pmmc-questioner-v3.4",
        "renderer_version": "questioner-safe-alias-renderer-v3.4",
        "parser_version": "questioner-json-parser-v3.4",
    },
    "pmmc-questioner-v3.5": {
        "prompt_version": "pmmc-questioner-v3.5",
        "renderer_version": "questioner-safe-alias-renderer-v3.5",
        "parser_version": "questioner-json-parser-v3.5",
    },
    "pmmc-questioner-v3.6": {
        "prompt_version": "pmmc-questioner-v3.6",
        "renderer_version": "questioner-safe-alias-renderer-v3.6",
        "parser_version": "questioner-json-parser-v3.6",
    },
}

SESSION_ANCHOR_BUILDER = "session-anchor-candidate-set-v1"
ROUND_BLOCK_BUILDER = "round-block-candidate-set-v1"
SESSION_OPPORTUNITY_BUILDER = "session-opportunity-candidate-set-v2"
ROUND_OPPORTUNITY_BUILDER = "round-opportunity-candidate-set-v2"
SESSION_OPPORTUNITY_BUILDER_V3 = "session-opportunity-candidate-set-v3"
ROUND_OPPORTUNITY_BUILDER_V3 = "round-opportunity-candidate-set-v3"
SESSION_OPPORTUNITY_BUILDER_V4 = "session-opportunity-candidate-set-v4"
ROUND_OPPORTUNITY_BUILDER_V4 = "round-opportunity-candidate-set-v4"
SESSION_OPPORTUNITY_BUILDER_V5 = "session-opportunity-candidate-set-v5"
ROUND_OPPORTUNITY_BUILDER_V5 = "round-opportunity-candidate-set-v5"
SESSION_OPPORTUNITY_BUILDER_V6 = "session-opportunity-candidate-set-v6"
ROUND_OPPORTUNITY_BUILDER_V6 = "round-opportunity-candidate-set-v6"
SESSION_OPPORTUNITY_BUILDERS = frozenset(
    {
        SESSION_OPPORTUNITY_BUILDER,
        SESSION_OPPORTUNITY_BUILDER_V3,
        SESSION_OPPORTUNITY_BUILDER_V4,
        SESSION_OPPORTUNITY_BUILDER_V5,
        SESSION_OPPORTUNITY_BUILDER_V6,
        "session-opportunity-candidate-set-v7",
    }
)
ROUND_OPPORTUNITY_BUILDERS = frozenset(
    {
        ROUND_OPPORTUNITY_BUILDER,
        ROUND_OPPORTUNITY_BUILDER_V3,
        ROUND_OPPORTUNITY_BUILDER_V4,
        ROUND_OPPORTUNITY_BUILDER_V5,
        ROUND_OPPORTUNITY_BUILDER_V6,
        "round-opportunity-candidate-set-v7",
    }
)
SUPPORTED_UNIT_BUILDERS = frozenset(
    {
        SESSION_ANCHOR_BUILDER,
        ROUND_BLOCK_BUILDER,
        *SESSION_OPPORTUNITY_BUILDERS,
        *ROUND_OPPORTUNITY_BUILDERS,
    }
)


class QuestionerConfig(BaseSchema):
    protocol_version: NonEmptyStr = "pmmc-questioner-v1"
    prompt_version: NonEmptyStr = "pmmc-questioner-v1"
    renderer_version: NonEmptyStr = "questioner-multimodal-renderer-v1"
    parser_version: NonEmptyStr = "questioner-json-parser-v1"
    unit_builder_version: NonEmptyStr = "session-anchor-candidate-set-v1"
    dataset_unit_builders: dict[DatasetName, NonEmptyStr] = Field(default_factory=dict)
    round_block_size: PositiveInt = 4
    round_block_stride: PositiveInt = 4
    gate_version: NonEmptyStr = "questioner-hard-gate-v1"
    dedupe_version: NonEmptyStr = "questioner-exact-dedupe-v1"
    matcher_version: NonEmptyStr = "qbank-hybrid-rrf-v1"
    coverage_protocol_version: NonEmptyStr = "qbank-coverage-v1"
    compiler_model_alias: NonEmptyStr = "compiler-qwen3.5-9b"
    text_embedding_model_alias: NonEmptyStr = "qwen3-embedding-8b"
    multimodal_embedding_model_alias: NonEmptyStr = "qwen3-vl-embedding-8b"
    total_context_tokens: PositiveInt = 32768
    reserved_output_tokens: PositiveInt = 4096
    max_images_per_unit: PositiveInt = 8
    question_budget_max: PositiveInt = 8
    control_budget_max: PositiveInt = 1
    max_paraphrases: int = Field(default=3, ge=0, le=3)
    job_max_attempts: PositiveInt = 2
    text_embedding_batch_size: PositiveInt = 32
    multimodal_embedding_batch_size: PositiveInt = 8
    embedding_concurrency: PositiveInt = 4
    rrf_constant: PositiveInt = 60
    per_channel_top_k: PositiveInt = 50
    question_budgets: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 4, 8])
    match_ks: list[PositiveInt] = Field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    relation_order: list[CompilationUnitRelationType] = Field(
        default_factory=lambda: list(DEFAULT_RELATION_ORDER)
    )
    gate_mode: CoverageGateMode = CoverageGateMode.REPORT_ONLY
    scientific_thresholds: dict[NonEmptyStr, float] = Field(default_factory=dict)
    opportunity_builder_version: NonEmptyStr = "questioner-opportunity-builder-v1"
    verifier_prompt_version: NonEmptyStr = "pmmc-questioner-verifier-v1"
    verifier_schema_version: NonEmptyStr = "questioner-verifier-schema-v1"
    max_opportunities_per_unit: PositiveInt = 24
    fts_lexeme_limit: PositiveInt = 12
    fts_relation_top_k: PositiveInt = 8
    relation_selector_version: NonEmptyStr = "postgres-simple-fts-v1"
    semantic_revision_max: int = Field(default=1, ge=0, le=1)
    family_dedupe_version: NonEmptyStr = "qbank-family-dedupe-v1"
    memory_raw_image_policy: Literal["allow", "forbid"] = "allow"

    @model_serializer(mode="wrap")
    def _serialize_historical_contract(self, handler):
        value = handler(self)
        if not self.is_v36:
            value.pop("memory_raw_image_policy", None)
        return value

    @property
    def max_input_tokens(self) -> int:
        return self.total_context_tokens - self.reserved_output_tokens

    @property
    def is_v2(self) -> bool:
        return self.protocol_version in {"pmmc-questioner-v2", "pmmc-questioner-v2.1"}

    @property
    def is_v3(self) -> bool:
        return self.protocol_version in {
            "pmmc-questioner-v3",
            "pmmc-questioner-v3.1",
            "pmmc-questioner-v3.2",
            "pmmc-questioner-v3.3",
            "pmmc-questioner-v3.4",
            "pmmc-questioner-v3.5",
            "pmmc-questioner-v3.6",
        }

    @property
    def is_v31(self) -> bool:
        return self.protocol_version == "pmmc-questioner-v3.1"

    @property
    def is_v32(self) -> bool:
        return self.protocol_version == "pmmc-questioner-v3.2"

    @property
    def is_v33(self) -> bool:
        return self.protocol_version == "pmmc-questioner-v3.3"

    @property
    def is_v34(self) -> bool:
        # Span-capable lineage; v3.6 has its own native DTO and verifier dispatch.
        return self.protocol_version in {
            "pmmc-questioner-v3.4",
            "pmmc-questioner-v3.5",
            "pmmc-questioner-v3.6",
        }

    @property
    def is_v35(self) -> bool:
        return self.protocol_version == "pmmc-questioner-v3.5"

    @property
    def is_v36(self) -> bool:
        return self.protocol_version == "pmmc-questioner-v3.6"

    @property
    def evidence_first(self) -> bool:
        return self.opportunity_builder_version == EVIDENCE_FIRST_VERSION

    @property
    def compiler_schema_version(self) -> str:
        """Return the compiler fact contract owned by this producer protocol."""

        if self.is_v36:
            return "0.9.0"
        if self.is_v35:
            return "0.8.0"
        if self.is_v34:
            return "0.7.0"
        if self.is_v33:
            return "0.6.0"
        if self.is_v32:
            return "0.5.0"
        if self.is_v31:
            return "0.4.0"
        return "0.3.0" if self.is_v3 else "0.2.0"

    def unit_builder_for(self, dataset: DatasetName) -> str:
        return self.dataset_unit_builders.get(dataset, self.unit_builder_version)

    @model_validator(mode="after")
    def _validate_frozen_contract(self) -> QuestionerConfig:
        if self.evidence_first and not self.is_v36:
            raise ValueError("evidence-first opportunities require native Questioner v3.6")
        if not self.is_v36 and self.memory_raw_image_policy != "allow":
            raise ValueError("memory_raw_image_policy=forbid requires Questioner v3.6")
        if self.is_v36 and self.max_images_per_unit > 8:
            raise ValueError("Questioner v3.6 supports at most eight images per unit")
        expected_versions = QUESTIONER_PROTOCOL_VERSIONS.get(self.protocol_version)
        if expected_versions is None:
            raise ValueError("unsupported Questioner protocol version")
        actual_versions = {
            "prompt_version": self.prompt_version,
            "renderer_version": self.renderer_version,
            "parser_version": self.parser_version,
        }
        if actual_versions != expected_versions:
            raise ValueError("Questioner protocol, prompt, renderer, and parser must match")
        configured_builders = {
            self.unit_builder_version,
            *self.dataset_unit_builders.values(),
        }
        if not configured_builders <= SUPPORTED_UNIT_BUILDERS:
            raise ValueError("unsupported Questioner unit builder")
        if self.protocol_version == "pmmc-questioner-v2.1":
            if self.unit_builder_version != SESSION_ANCHOR_BUILDER:
                raise ValueError("Questioner v2.1 default builder must remain session-anchored")
            if self.dataset_unit_builders != {DatasetName.MEMGALLERY: ROUND_BLOCK_BUILDER}:
                raise ValueError("Questioner v2.1 must use the MemGallery round-block builder")
            if self.round_block_size != 4 or self.round_block_stride != 4:
                raise ValueError("Questioner v2.1 freezes non-overlapping four-round blocks")
        elif self.protocol_version in {
            "pmmc-questioner-v3",
            "pmmc-questioner-v3.1",
            "pmmc-questioner-v3.2",
            "pmmc-questioner-v3.3",
            "pmmc-questioner-v3.4",
            "pmmc-questioner-v3.5",
            "pmmc-questioner-v3.6",
        }:
            expected_session_builder = (
                "session-opportunity-candidate-set-v7"
                if self.is_v36
                else SESSION_OPPORTUNITY_BUILDER_V6
                if self.is_v35
                else SESSION_OPPORTUNITY_BUILDER_V5
                if self.is_v34
                else SESSION_OPPORTUNITY_BUILDER_V4
                if self.is_v33
                else SESSION_OPPORTUNITY_BUILDER_V3
                if self.is_v32
                else SESSION_OPPORTUNITY_BUILDER
            )
            expected_round_builder = (
                "round-opportunity-candidate-set-v7"
                if self.is_v36
                else ROUND_OPPORTUNITY_BUILDER_V6
                if self.is_v35
                else ROUND_OPPORTUNITY_BUILDER_V5
                if self.is_v34
                else ROUND_OPPORTUNITY_BUILDER_V4
                if self.is_v33
                else ROUND_OPPORTUNITY_BUILDER_V3
                if self.is_v32
                else ROUND_OPPORTUNITY_BUILDER
            )
            if self.unit_builder_version != expected_session_builder:
                raise ValueError("Questioner v3 requires the session opportunity builder")
            if self.dataset_unit_builders != {DatasetName.MEMGALLERY: expected_round_builder}:
                raise ValueError("Questioner v3 requires the MemGallery round opportunity builder")
            if self.round_block_size != 4 or self.round_block_stride != 4:
                raise ValueError("Questioner v3 freezes non-overlapping four-round blocks")
            expected_quality_versions = (
                (
                    EVIDENCE_FIRST_VERSION
                    if self.evidence_first else "questioner-opportunity-builder-v7",
                    "pmmc-questioner-verifier-v7",
                    "questioner-verifier-schema-v7",
                    "questioner-static-semantic-gate-v3.6",
                )
                if self.is_v36
                else (
                    "questioner-opportunity-builder-v6",
                    "pmmc-questioner-verifier-v6",
                    "questioner-verifier-schema-v6",
                    "questioner-static-semantic-gate-v3.5",
                )
                if self.is_v35
                else (
                    "questioner-opportunity-builder-v5",
                    "pmmc-questioner-verifier-v5",
                    "questioner-verifier-schema-v5",
                    "questioner-static-semantic-gate-v3.4",
                )
                if self.is_v34
                else (
                    "questioner-opportunity-builder-v4",
                    "pmmc-questioner-verifier-v4",
                    "questioner-verifier-schema-v4",
                    "questioner-static-semantic-gate-v3.3",
                )
                if self.is_v33
                else (
                    "questioner-opportunity-builder-v3",
                    "pmmc-questioner-verifier-v3",
                    "questioner-verifier-schema-v3",
                    "questioner-static-semantic-gate-v3.2",
                )
                if self.is_v32
                else (
                    "questioner-opportunity-builder-v2",
                    "pmmc-questioner-verifier-v2",
                    "questioner-verifier-schema-v2",
                    "questioner-static-semantic-gate-v3.1",
                )
                if self.is_v31
                else (
                    "questioner-opportunity-builder-v1",
                    "pmmc-questioner-verifier-v1",
                    "questioner-verifier-schema-v1",
                    "questioner-hard-gate-v3",
                )
            )
            actual_quality_versions = (
                self.opportunity_builder_version,
                self.verifier_prompt_version,
                self.verifier_schema_version,
                self.gate_version,
            )
            if actual_quality_versions != expected_quality_versions:
                raise ValueError("Questioner opportunity/verifier versions are frozen")
            if (
                self.max_opportunities_per_unit != 24
                or self.fts_lexeme_limit != 12
                or self.fts_relation_top_k != 8
                or self.relation_selector_version
                != (
                    "postgres-simple-fts-validated-v4"
                    if self.is_v34 and not (self.is_v35 or self.is_v36)
                    else "postgres-simple-fts-validated-v5"
                    if self.is_v35 or self.is_v36
                    else "postgres-simple-fts-validated-v3"
                    if self.is_v33
                    else "postgres-simple-fts-validated-v2"
                    if self.is_v32
                    else "postgres-simple-fts-v1"
                )
                or self.semantic_revision_max != (0 if self.is_v36 else 1)
            ):
                raise ValueError("Questioner v3 quality-remediation limits are frozen")
            if (self.is_v33 or self.is_v34) and (
                self.control_budget_max != 1
                or self.family_dedupe_version
                != (
                    "qbank-family-dedupe-v5"
                    if self.is_v36
                    else "qbank-family-dedupe-v4"
                    if self.is_v35
                    else "qbank-family-dedupe-v3"
                    if self.is_v34
                    else "qbank-family-dedupe-v2"
                )
                or self.dedupe_version
                != (
                    "questioner-information-need-dedupe-v3"
                    if self.is_v36
                    else "questioner-exact-dedupe-v2"
                )
            ):
                raise ValueError("Questioner v3.3 budgets and family dedupe are frozen")
        elif self.dataset_unit_builders:
            raise ValueError("dataset-specific builders require Questioner v2.1 or v3")
        if self.reserved_output_tokens >= self.total_context_tokens:
            raise ValueError("reserved output tokens must be smaller than total context")
        if (
            self.question_budget_max != 8
            or self.max_paraphrases != (0 if self.is_v36 else 3)
            or self.control_budget_max != 1
        ):
            raise ValueError("Questioner freezes budget=8 and max_paraphrases=3")
        if self.question_budgets != [1, 2, 4, 8]:
            raise ValueError("Questioner freezes budgets to 1/2/4/8")
        if self.match_ks != [1, 3, 5, 10, 20, 50]:
            raise ValueError("qbank-coverage-v1 freezes K to 1/3/5/10/20/50")
        if tuple(self.relation_order) != DEFAULT_RELATION_ORDER:
            raise ValueError("Questioner relation order is immutable")
        if self.per_channel_top_k < max(self.match_ks):
            raise ValueError("per-channel top-k must cover every reported K")
        if self.gate_mode is CoverageGateMode.REPORT_ONLY and self.scientific_thresholds:
            raise ValueError("report-only coverage gates cannot enforce scientific thresholds")
        if any(not 0.0 <= value <= 1.0 for value in self.scientific_thresholds.values()):
            raise ValueError("scientific thresholds must be probabilities")
        return self


def load_questioner_config(path: Path) -> QuestionerConfig:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("questioner config must contain a YAML mapping")
    return QuestionerConfig.model_validate(raw)


def questioner_config_hash(config: QuestionerConfig) -> str:
    value = config.model_dump(mode="json")
    if not config.is_v36:
        value.pop("memory_raw_image_policy", None)
    if config.protocol_version not in {
        "pmmc-questioner-v2.1",
        "pmmc-questioner-v3",
        "pmmc-questioner-v3.1",
        "pmmc-questioner-v3.2",
        "pmmc-questioner-v3.3",
        "pmmc-questioner-v3.4",
        "pmmc-questioner-v3.5",
        "pmmc-questioner-v3.6",
    }:
        # Preserve the identity of v1/v2 runs created before dataset-aware builders.
        for key in ("dataset_unit_builders", "round_block_size", "round_block_stride"):
            value.pop(key, None)
    if not config.is_v3:
        for key in (
            "opportunity_builder_version",
            "verifier_prompt_version",
            "verifier_schema_version",
            "max_opportunities_per_unit",
            "fts_lexeme_limit",
            "fts_relation_top_k",
            "relation_selector_version",
            "semantic_revision_max",
        ):
            value.pop(key, None)
    if not (config.is_v33 or config.is_v34):
        # v3.3 introduced separate control budgeting and family-level dedupe.
        value.pop("control_budget_max", None)
        value.pop("family_dedupe_version", None)
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
