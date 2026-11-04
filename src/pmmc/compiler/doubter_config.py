"""Frozen Step 6D Doubter configuration and identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import model_validator

from pmmc.schemas.base import BaseSchema, NonEmptyStr, PositiveInt


class DoubterConfig(BaseSchema):
    protocol_version: NonEmptyStr = "pmmc-doubter-v1"
    prompt_version: NonEmptyStr = "pmmc-doubter-v1"
    renderer_version: NonEmptyStr = "doubter-evidence-renderer-v1"
    parser_version: NonEmptyStr = "doubter-json-parser-v1"
    schema_version: NonEmptyStr = "doubter-assessment-schema-v1"
    decision_policy_version: NonEmptyStr = "doubter-server-policy-v1"
    directive_policy_version: NonEmptyStr = "planner-revision-directives-v1"
    r0_input_policy_version: NonEmptyStr = "r0-input-gate-v1"
    routing_projection_version: NonEmptyStr = "verified-qbank-projection-v1"
    doubter_model_alias: NonEmptyStr = "compiler-qwen3.5-9b"
    total_context_tokens: PositiveInt = 32768
    reserved_output_tokens: PositiveInt = 4096
    active_round: PositiveInt = 2
    max_round: PositiveInt = 3
    max_evidence_items: PositiveInt = 50

    @property
    def is_v11(self) -> bool:
        return self.protocol_version in {
            "pmmc-doubter-v1.1",
            "pmmc-doubter-v1.2",
            "pmmc-doubter-v1.3",
        }

    @property
    def is_v13(self) -> bool:
        return self.protocol_version == "pmmc-doubter-v1.3"

    @property
    def is_v12(self) -> bool:
        return self.protocol_version == "pmmc-doubter-v1.2"

    @property
    def max_input_tokens(self) -> int:
        return self.total_context_tokens - self.reserved_output_tokens

    @model_validator(mode="after")
    def _validate_frozen_protocol(self) -> DoubterConfig:
        identity = (
            self.protocol_version,
            self.prompt_version,
            self.renderer_version,
            self.parser_version,
            self.schema_version,
            self.decision_policy_version,
            self.directive_policy_version,
            self.r0_input_policy_version,
            self.routing_projection_version,
        )
        allowed = {
            (
                "pmmc-doubter-v1.3",
                "pmmc-doubter-v1.3",
                "doubter-evidence-renderer-v1.3",
                "doubter-json-parser-v1.3",
                "doubter-assessment-schema-v1.3",
                "doubter-server-policy-v4",
                "planner-revision-directives-v4",
                "r0-input-gate-v4",
                "verified-qbank-projection-v4",
            ),
            (
                "pmmc-doubter-v1",
                "pmmc-doubter-v1",
                "doubter-evidence-renderer-v1",
                "doubter-json-parser-v1",
                "doubter-assessment-schema-v1",
                "doubter-server-policy-v1",
                "planner-revision-directives-v1",
                "r0-input-gate-v1",
                "verified-qbank-projection-v1",
            ),
            (
                "pmmc-doubter-v1.1",
                "pmmc-doubter-v1.1",
                "doubter-evidence-renderer-v1.1",
                "doubter-json-parser-v1.1",
                "doubter-assessment-schema-v1.1",
                "doubter-server-policy-v2",
                "planner-revision-directives-v2",
                "r0-input-gate-v2",
                "verified-qbank-projection-v2",
            ),
            (
                "pmmc-doubter-v1.2",
                "pmmc-doubter-v1.2",
                "doubter-evidence-renderer-v1.2",
                "doubter-json-parser-v1.2",
                "doubter-assessment-schema-v1.2",
                "doubter-server-policy-v3",
                "planner-revision-directives-v3",
                "r0-input-gate-v3",
                "verified-qbank-projection-v3",
            ),
        }
        if identity not in allowed:
            raise ValueError("Doubter v1 protocol identities are frozen")
        if self.reserved_output_tokens >= self.total_context_tokens:
            raise ValueError("Doubter output reservation must be smaller than context")
        if self.active_round != 2 or self.max_round != 3 or self.max_evidence_items != 50:
            raise ValueError("Step 6D requires active r2, maximum r3, and top-50 evidence")
        return self


def load_doubter_config(path: Path) -> DoubterConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Doubter config must contain a YAML mapping")
    return DoubterConfig.model_validate(raw)


def doubter_config_hash(config: DoubterConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
