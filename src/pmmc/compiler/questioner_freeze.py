"""Compatibility checks for approved Questioner protocol freezes."""

from __future__ import annotations

from typing import Any

from pmmc.compiler.ids import canonical_content_hash
from pmmc.compiler.questioner_config import QuestionerConfig, questioner_config_hash
from pmmc.compiler.questioner_prompt import (
    questioner_prompt_sha256,
    questioner_request_contract_hash,
    questioner_response_schema_sha256,
)
from pmmc.models.registry import ModelRegistry
from pmmc.schemas import QuestionerFreezeManifest


def questioner_freeze_identity(config: QuestionerConfig) -> dict[str, str]:
    """Build the immutable Questioner identity approved by a v3 freeze."""

    if not config.is_v3:
        raise ValueError("Questioner freeze identity requires a v3 protocol")
    opportunity_identity: dict[str, Any] = {
        "opportunity_builder_version": config.opportunity_builder_version,
        "unit_builder_version": config.unit_builder_version,
        "dataset_unit_builders": {
            key.value: value for key, value in config.dataset_unit_builders.items()
        },
        "relation_selector_version": config.relation_selector_version,
        "fts_lexeme_limit": config.fts_lexeme_limit,
        "fts_relation_top_k": config.fts_relation_top_k,
        "max_opportunities_per_unit": config.max_opportunities_per_unit,
        "request_contract_hash": questioner_request_contract_hash(
            prompt_version=config.prompt_version,
            renderer_version=config.renderer_version,
            opportunity_builder_version=config.opportunity_builder_version,
        ),
    }
    verifier_prompt_hash = questioner_prompt_sha256(config.verifier_prompt_version)
    verifier_schema_hash = questioner_response_schema_sha256(config.verifier_prompt_version)
    verifier_protocol_hash = canonical_content_hash(
        {
            "prompt_version": config.verifier_prompt_version,
            "prompt_hash": verifier_prompt_hash,
            "schema_version": config.verifier_schema_version,
            "schema_hash": verifier_schema_hash,
            "semantic_revision_max": config.semantic_revision_max,
            "model_alias": config.compiler_model_alias,
        }
    )
    return {
        "questioner_protocol_version": config.protocol_version,
        "questioner_config_hash": questioner_config_hash(config),
        "prompt_version": config.prompt_version,
        "prompt_hash": questioner_prompt_sha256(config.prompt_version),
        "response_schema_hash": questioner_response_schema_sha256(config.prompt_version),
        "renderer_version": config.renderer_version,
        "parser_version": config.parser_version,
        "unit_builder_version": config.unit_builder_version,
        "gate_version": config.gate_version,
        "dedupe_version": config.dedupe_version,
        "matcher_version": config.matcher_version,
        "opportunity_builder_version": config.opportunity_builder_version,
        "relation_selector_version": config.relation_selector_version,
        "opportunity_builder_hash": canonical_content_hash(opportunity_identity),
        "verifier_prompt_version": config.verifier_prompt_version,
        "verifier_prompt_hash": verifier_prompt_hash,
        "verifier_schema_hash": verifier_schema_hash,
        "verifier_protocol_hash": verifier_protocol_hash,
    }


def verify_questioner_freeze_compatibility(
    freeze: QuestionerFreezeManifest,
    config: QuestionerConfig,
    registry: ModelRegistry,
) -> None:
    """Reject an approved freeze that differs from the selected runtime identity."""

    expected = questioner_freeze_identity(config)
    for field_name, expected_value in expected.items():
        actual = getattr(freeze, field_name)
        if actual != expected_value:
            raise ValueError(
                "Questioner freeze identity mismatch for "
                f"{field_name}: expected {expected_value!r}, received {actual!r}"
            )

    required_aliases = (
        config.compiler_model_alias,
        config.text_embedding_model_alias,
        config.multimodal_embedding_model_alias,
    )
    for alias in required_aliases:
        expected_revision = registry.get(alias).model_revision
        frozen_revision = freeze.model_commits.get(alias)
        if frozen_revision != expected_revision:
            raise ValueError(
                "Questioner freeze model commit mismatch for "
                f"{alias}: expected {expected_revision!r}, received {frozen_revision!r}"
            )


__all__ = [
    "questioner_freeze_identity",
    "verify_questioner_freeze_compatibility",
]
