"""Fail closed if execution inputs drift after a campaign was planned."""

from pathlib import Path

from pmmc.experiments.step8_artifacts import sha256_file
from pmmc.experiments.step8_model_identity import (
    load_step8_model_entry_hashes,
    validate_model_alias_contract,
)
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.step8 import ANSWER_BACKBONES, ScientificProtocol, Step8BenchmarkConfig

CONFIG_FIELDS = (
    "model_registry", "questioner_config", "planner_config", "doubter_config",
    "foundation_config", "query_executor_config",
)
FIXED_CONFIGS = {
    "baseline_full_context_config": (
        "full_context_config_path", "configs/baselines/full_context.yaml",
    ),
    "baseline_rag_config": ("rag_config_path", "configs/baselines/rag.yaml"),
    "evaluation_config": ("evaluation_config_path", "configs/evaluation/baseline.yaml"),
}
POLICY_FIELDS = (
    "seed",
    "oracle_at_50",
    "match_threshold", "gap_threshold", "text_relevance_threshold",
    "multimodal_relevance_threshold", "no_reranker_match_threshold",
    "no_reranker_gap_threshold",
)


def verify_frozen_execution_inputs(
    settings: Settings, config: Step8BenchmarkConfig, protocol: ScientificProtocol,
) -> None:
    """Use original registry bytes, never the commit-resolved derivative.

    Read-only audit/export does not call this gate; every model-capable worker does.
    Settings overrides are rejected rather than silently ignored or incorporated
    into an already frozen protocol.
    """
    if protocol.answer_backbones != ANSWER_BACKBONES:
        raise ValueError("step8_retired_backbone_protocol_read_only")
    paths = {name: settings.resolve_path(getattr(config, name)) for name in CONFIG_FIELDS}
    policy = config.metadata.get("partial_import_policy")
    if policy is not None:
        paths["partial_import_policy"] = settings.resolve_path(Path(str(policy)))
    elif "partial_import_policy" in protocol.config_hashes:
        raise ValueError("step8_frozen_partial_import_policy_missing")
    for name, (setting, relative) in FIXED_CONFIGS.items():
        expected = settings.resolve_path(Path(relative))
        actual = settings.resolve_path(getattr(settings, setting))
        if actual.resolve() != expected.resolve():
            raise ValueError("step8_unfrozen_config_override_" + name)
        paths[name] = expected
    for name, path in paths.items():
        expected_hash = protocol.config_hashes.get(name)
        if expected_hash is None:
            raise ValueError("step8_frozen_config_hash_missing_" + name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError("step8_frozen_config_changed_" + name)
    lock_path = settings.resolve_path(config.model_lock)
    if not lock_path.is_file() or sha256_file(lock_path) != protocol.model_lock_hash:
        raise ValueError("step8_frozen_model_lock_changed")
    for name in POLICY_FIELDS:
        if getattr(config, name) != getattr(protocol, name):
            raise ValueError("step8_frozen_policy_changed_" + name)
    validate_model_alias_contract(config, settings.project_root)
    if load_step8_model_entry_hashes(config, settings.project_root) != protocol.model_entry_hashes:
        raise ValueError("step8_frozen_model_entry_hashes_changed")
