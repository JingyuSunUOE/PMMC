"""Strict, versioned configuration for the PMMC compiler foundation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from pmmc.schemas.base import BaseSchema, NonEmptyStr, NonNegativeInt, PositiveInt

OPERATOR_SET_V1 = (
    "retrieve_text_by_keyword",
    "retrieve_text_by_embedding",
    "retrieve_image_by_visual_embedding",
    "retrieve_image_text_pair",
    "retrieve_context_around_image",
    "expand_temporal_neighbors",
    "inspect_raw_image",
    "verify_evidence_sufficiency",
    "fallback_to_multimodal_rag",
    "abstain_if_no_evidence",
)
OPERATOR_SET_V2 = OPERATOR_SET_V1
OPERATOR_SET_V3 = OPERATOR_SET_V1
OPERATOR_SET_V4 = OPERATOR_SET_V1
OPERATOR_SET_V5 = OPERATOR_SET_V1
OPERATOR_SET_V6 = OPERATOR_SET_V1
OPERATOR_SET_V7 = OPERATOR_SET_V1


class CompilerFoundationConfig(BaseSchema):
    """Frozen limits and identities shared by validation and execution."""

    protocol_version: NonEmptyStr = "pmmc-compiler-foundation-v1"
    unit_builder_version: NonEmptyStr = "compiler-unit-v1"
    operator_set_version: NonEmptyStr = "pmmc-operators-v1"
    compiler_model_alias: NonEmptyStr = "compiler-qwen3.5-9b"
    question_budget_max: PositiveInt = 8
    r_max: NonNegativeInt = 2
    program_max_steps: PositiveInt = 8
    operator_top_k_max: PositiveInt = 50
    total_evidence_max: PositiveInt = 200
    operator_timeout_seconds: PositiveInt = 60
    dry_run_policy: NonEmptyStr = "evidence_driven_v1"
    vector_search_mode: NonEmptyStr = "exact"
    fts_configuration: NonEmptyStr = "simple"
    operators: list[NonEmptyStr] = Field(default_factory=lambda: list(OPERATOR_SET_V1))

    @model_validator(mode="after")
    def _validate_frozen_contract(self) -> CompilerFoundationConfig:
        if self.vector_search_mode != "exact":
            raise ValueError("Step 6A only supports exact vector search")
        supported = {
            "pmmc-operators-v1": (OPERATOR_SET_V1, "evidence_driven_v1"),
            "pmmc-operators-v2": (OPERATOR_SET_V2, "evidence_driven_v2"),
            "pmmc-operators-v3": (OPERATOR_SET_V3, "evidence_driven_v3"),
            "pmmc-operators-v4": (OPERATOR_SET_V4, "evidence_driven_v4"),
            "pmmc-operators-v5": (OPERATOR_SET_V5, "evidence_driven_v5"),
            "pmmc-operators-v6": (OPERATOR_SET_V6, "evidence_driven_v6"),
            "pmmc-operators-v7": (OPERATOR_SET_V7, "evidence_driven_v7"),
        }
        expected = supported.get(self.operator_set_version)
        if expected is None:
            raise ValueError("unsupported compiler operator set")
        expected_operators, expected_policy = expected
        if self.dry_run_policy != expected_policy:
            raise ValueError("operator set and dry-run policy versions must match")
        if self.fts_configuration != "simple":
            raise ValueError("Step 6A freezes the Postgres simple FTS configuration")
        if tuple(self.operators) != expected_operators:
            raise ValueError("operators must exactly match the selected ordered operator set")
        if self.operator_top_k_max > self.total_evidence_max:
            raise ValueError("operator_top_k_max cannot exceed total_evidence_max")
        return self


def load_compiler_config(path: Path) -> CompilerFoundationConfig:
    """Load strict YAML, resolving OmegaConf interpolation when available."""

    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("compiler config must contain a YAML mapping")
    return CompilerFoundationConfig.model_validate(raw)


def compiler_config_hash(config: CompilerFoundationConfig) -> str:
    """Return a stable hash over every frozen compiler setting."""

    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
