"""Frozen Planner v1 configuration and identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import model_validator

from pmmc.schemas.base import BaseSchema, NonEmptyStr, PositiveInt
from pmmc.schemas.planner import PlannerStrategyDraftV12


class PlannerConfig(BaseSchema):
    protocol_version: NonEmptyStr = "pmmc-planner-v1"
    prompt_version: NonEmptyStr = "pmmc-planner-v1"
    renderer_version: NonEmptyStr = "planner-safe-alias-renderer-v1"
    parser_version: NonEmptyStr = "planner-json-parser-v1"
    schema_version: NonEmptyStr = "planner-program-schema-v1"
    policy_version: NonEmptyStr = "planner-program-policy-v1"
    operator_set_version: NonEmptyStr = "pmmc-operators-v2"
    dry_run_policy: NonEmptyStr = "evidence_driven_v2"
    planner_model_alias: NonEmptyStr = "compiler-qwen3.5-9b"
    total_context_tokens: PositiveInt = 32768
    reserved_output_tokens: PositiveInt = 4096
    candidates_per_batch: PositiveInt = 4
    program_max_steps: PositiveInt = 8
    operator_top_k_max: PositiveInt = 50
    expansion_radius_max: PositiveInt = 2
    total_evidence_max: PositiveInt = 200
    planning_mode: Literal["dynamic", "fixed"] = "dynamic"
    fixed_strategy: PlannerStrategyDraftV12 | None = None
    memory_raw_image_policy: Literal["allow", "forbid"] = "allow"

    @property
    def is_v11(self) -> bool:
        return self.protocol_version == "pmmc-planner-v1.1"

    @property
    def is_v12(self) -> bool:
        return self.protocol_version == "pmmc-planner-v1.2"

    @property
    def is_v13(self) -> bool:
        return self.protocol_version in {
            "pmmc-planner-v1.3",
            "pmmc-planner-v1.4",
            "pmmc-planner-v1.5",
        }

    @property
    def is_v15(self) -> bool:
        return self.protocol_version == "pmmc-planner-v1.5"

    @property
    def is_v14(self) -> bool:
        return self.protocol_version == "pmmc-planner-v1.4"

    @property
    def max_input_tokens(self) -> int:
        return self.total_context_tokens - self.reserved_output_tokens

    @model_validator(mode="after")
    def _validate_v1(self) -> PlannerConfig:
        expected = (
            self.protocol_version,
            self.prompt_version,
            self.renderer_version,
            self.parser_version,
            self.schema_version,
            self.policy_version,
            self.operator_set_version,
            self.dry_run_policy,
        )
        allowed = {
            (
                "pmmc-planner-v1.5",
                "pmmc-planner-v1.5",
                "planner-strategy-renderer-v1.5",
                "planner-strategy-json-parser-v1.5",
                "planner-strategy-schema-v1.5",
                "planner-program-policy-v6",
                "pmmc-operators-v7",
                "evidence_driven_v7",
            ),
            (
                "pmmc-planner-v1",
                "pmmc-planner-v1",
                "planner-safe-alias-renderer-v1",
                "planner-json-parser-v1",
                "planner-program-schema-v1",
                "planner-program-policy-v1",
                "pmmc-operators-v2",
                "evidence_driven_v2",
            ),
            (
                "pmmc-planner-v1.1",
                "pmmc-planner-v1.1",
                "planner-non-oracle-renderer-v1.1",
                "planner-json-parser-v1.1",
                "planner-program-schema-v1.1",
                "planner-program-policy-v2",
                "pmmc-operators-v3",
                "evidence_driven_v3",
            ),
            (
                "pmmc-planner-v1.2",
                "pmmc-planner-v1.2",
                "planner-strategy-renderer-v1.2",
                "planner-strategy-json-parser-v1.2",
                "planner-strategy-schema-v1.2",
                "planner-program-policy-v3",
                "pmmc-operators-v4",
                "evidence_driven_v4",
            ),
            (
                "pmmc-planner-v1.3",
                "pmmc-planner-v1.3",
                "planner-strategy-renderer-v1.3",
                "planner-strategy-json-parser-v1.3",
                "planner-strategy-schema-v1.3",
                "planner-program-policy-v4",
                "pmmc-operators-v5",
                "evidence_driven_v5",
            ),
            (
                "pmmc-planner-v1.4",
                "pmmc-planner-v1.4",
                "planner-strategy-renderer-v1.4",
                "planner-strategy-json-parser-v1.4",
                "planner-strategy-schema-v1.4",
                "planner-program-policy-v5",
                "pmmc-operators-v6",
                "evidence_driven_v6",
            ),
        }
        if expected not in allowed:
            raise ValueError("Planner protocol identities are frozen")
        if not self.is_v15 and (
            self.planning_mode != "dynamic"
            or self.fixed_strategy is not None
            or self.memory_raw_image_policy != "allow"
            or self.total_evidence_max != 200
        ):
            raise ValueError("Step8 policies require Planner v1.5")
        if (self.planning_mode == "fixed") != (self.fixed_strategy is not None):
            raise ValueError("fixed planning requires exactly one predeclared fixed_strategy")
        if self.reserved_output_tokens >= self.total_context_tokens:
            raise ValueError("Planner output reservation must be smaller than context")
        expected_batch = 1 if self.is_v12 or self.is_v13 else 4
        if (
            self.candidates_per_batch != expected_batch
            or self.program_max_steps != 8
            or self.operator_top_k_max != 50
            or self.expansion_radius_max != 2
        ):
            raise ValueError("Planner v1 limits are frozen")
        return self


def load_planner_config(path: Path) -> PlannerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("planner config must contain a YAML mapping")
    return PlannerConfig.model_validate(raw)


def load_planner_treatment_config(
    path: Path,
    *,
    treatment: Literal["main", "fixed_planner", "no_raw_image"] = "main",
) -> PlannerConfig:
    """Materialize a frozen Step8 treatment before computing the run identity."""
    config = load_planner_config(path)
    if not config.is_v15:
        raise ValueError("Step8 treatments require native Planner v1.5")
    updates: dict[str, object] = {}
    if treatment == "fixed_planner":
        updates = {
            "planning_mode": "fixed",
            "fixed_strategy": {
                "candidate_alias": "Q01",
                "retrieval_channels": ["text_dense", "visual_dense"],
                "expansion_capabilities": [],
                "top_k_tier": "standard",
                "strategy_code": "fixed_text_mm_v1",
                "confidence": 1.0,
            },
        }
    elif treatment == "no_raw_image":
        updates = {"memory_raw_image_policy": "forbid"}
    elif treatment != "main":
        raise ValueError("unknown Step8 Planner treatment")
    return PlannerConfig.model_validate({**config.model_dump(mode="json"), **updates})


def planner_config_hash(config: PlannerConfig) -> str:
    payload = json.dumps(
        config.model_dump(
            mode="json",
            exclude=(
                {"total_evidence_max", "planning_mode", "fixed_strategy", "memory_raw_image_policy"}
                if not config.is_v15
                else set()
            ),
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
