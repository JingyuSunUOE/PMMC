"""Validated model-lock identities and operation-owned Step 8 dependencies."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from pmmc.compiler.ids import canonical_content_hash
from pmmc.models.locking import assert_paper_lock, load_model_lock, resolve_registry_from_lock
from pmmc.models.registry import ModelRegistry, load_model_registry
from pmmc.schemas import ModelEndpointKind, ModelLockManifest
from pmmc.schemas.step8 import ANSWER_BACKBONES, ScientificProtocol, Step8BenchmarkConfig

COMPILER_ALIAS = "compiler-qwen3.5-9b"
TEXT_ALIAS = "qwen3-embedding-8b"
MULTIMODAL_ALIAS = "qwen3-vl-embedding-8b"
CONTRASTIVE_ALIAS = "siglip2-so400m-patch14-384"
RERANKER_ALIAS = "reranker-qwen3.5-9b"
GROUNDING_ALIAS = "grounding-qwen3.5-9b"
JUDGE_ALIAS = "judge-qwen3.5-9b"
COMPILER_ALIASES = (COMPILER_ALIAS, TEXT_ALIAS, MULTIMODAL_ALIAS)
REQUIRED_ALIASES = frozenset(
    (
        *COMPILER_ALIASES,
        CONTRASTIVE_ALIAS,
        RERANKER_ALIAS,
        GROUNDING_ALIAS,
        JUDGE_ALIAS,
        *ANSWER_BACKBONES,
    )
)
_METHOD_EMBEDDINGS = {
    "caption_text_rag": TEXT_ALIAS,
    "contrastive_multimodal_rag": CONTRASTIVE_ALIAS,
    "native_multimodal_rag": MULTIMODAL_ALIAS,
}


def compute_model_entry_hashes(registry: ModelRegistry, lock: ModelLockManifest) -> dict[str, str]:
    """Validate the complete original lock, then hash each resolved logical alias.

    This checks identities, not model-cache availability. Native providers have
    registry identities in lock metadata, never fabricated Hub snapshots.
    """
    registry = ModelRegistry.model_validate(registry.model_dump(mode="json"))
    lock = ModelLockManifest.model_validate(lock.model_dump(mode="json"))
    aliases = {entry.alias for entry in registry.models}
    if not REQUIRED_ALIASES <= aliases:
        raise ValueError("step8_model_identity_required_alias_missing")
    if any(entry.endpoint_kind is ModelEndpointKind.FAKE for entry in registry.models):
        raise ValueError("step8_model_identity_fake_entry")
    if lock.registry_hash != registry.resolved_hash:
        raise ValueError("step8_model_identity_original_registry_mismatch")
    assert_paper_lock(lock, registry, aliases=sorted(aliases))
    resolved = resolve_registry_from_lock(registry, lock)
    resolved.assert_paper_ready()
    native = lock.metadata.get("native_provider_entry_hashes", {})
    assert isinstance(native, dict)  # assert_paper_lock validates its shape and digests.
    if {entry.alias for entry in lock.entries} | set(native) != aliases:
        raise ValueError("step8_model_identity_lock_coverage_mismatch")
    result = {}
    for alias in sorted(aliases):
        original = registry.get(alias)
        payload: dict[str, object] = {
            "identity_version": "step8-model-entry-v1",
            "alias": alias,
            "resolved_registry_entry_hash": resolved.get(alias).resolved_hash,
        }
        if alias in native:
            payload.update(kind="native_provider", registry_entry_hash=native[alias])
        else:
            entry = lock.get(alias)
            files = [
                item.model_dump(mode="json") for item in sorted(entry.files, key=lambda f: f.path)
            ]
            if (
                entry.registry_entry_hash != original.resolved_hash
                or entry.repo_id != original.resolved_source_repo_id
                or entry.snapshot_content_hash != canonical_content_hash(files)
            ):
                raise ValueError("step8_model_identity_hub_entry_mismatch")
            payload.update(
                kind="hugging_face",
                repo_id=entry.repo_id,
                commit_sha=entry.commit_sha,
                files=files,
                snapshot_content_hash=entry.snapshot_content_hash,
            )
        result[alias] = canonical_content_hash(payload)
    return result


def load_step8_model_entry_hashes(config: Step8BenchmarkConfig, root: Path) -> dict[str, str]:
    """Read original inputs only; never feed a resolved registry into the lock gate."""
    return compute_model_entry_hashes(
        load_model_registry(root / config.model_registry),
        load_model_lock(root / config.model_lock),
    )


def validate_model_alias_contract(config: Step8BenchmarkConfig, root: Path) -> None:
    """Fail closed if configurable aliases escape the fixed Step 8 consumer map."""
    from pmmc.baselines.rag_config import load_rag_config
    from pmmc.compiler.doubter_config import load_doubter_config
    from pmmc.compiler.planner_config import load_planner_config
    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.evaluation.config import load_evaluation_config
    from pmmc.runtime.query_executor_config import load_query_executor_config

    q = load_questioner_config(root / config.questioner_config)
    p = load_planner_config(root / config.planner_config)
    d = load_doubter_config(root / config.doubter_config)
    rag = load_rag_config(root / "configs/baselines/rag.yaml")
    query = load_query_executor_config(root / config.query_executor_config)
    evaluation = load_evaluation_config(root / "configs/evaluation/baseline.yaml")
    actual = (
        q.compiler_model_alias,
        p.planner_model_alias,
        d.doubter_model_alias,
        q.text_embedding_model_alias,
        q.multimodal_embedding_model_alias,
        rag.caption_text_embedding_alias,
        rag.contrastive_embedding_alias,
        rag.native_multimodal_embedding_alias,
        query.reranker_model_alias,
        evaluation.judge_model_alias,
    )
    expected = (
        COMPILER_ALIAS,
        COMPILER_ALIAS,
        COMPILER_ALIAS,
        TEXT_ALIAS,
        MULTIMODAL_ALIAS,
        TEXT_ALIAS,
        CONTRASTIVE_ALIAS,
        MULTIMODAL_ALIAS,
        RERANKER_ALIAS,
        JUDGE_ALIAS,
    )
    if actual != expected:
        raise ValueError("step8_model_identity_config_alias_mismatch")


def selected_model_entry_hashes(
    protocol: ScientificProtocol,
    aliases: tuple[str, ...],
) -> dict[str, str]:
    if (
        protocol.compiler_alias,
        protocol.reranker_alias,
        protocol.grounding_alias,
        protocol.judge_alias,
    ) != (COMPILER_ALIAS, RERANKER_ALIAS, GROUNDING_ALIAS, JUDGE_ALIAS):
        raise ValueError("step8_model_identity_protocol_alias_mismatch")
    result = {}
    for alias in sorted(set(aliases)):
        digest = protocol.model_entry_hashes.get(alias)
        if digest is None or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError("step8_model_identity_entry_missing_or_invalid")
        result[alias] = digest
    return result


def compiler_model_entry_hashes(protocol: ScientificProtocol) -> dict[str, str]:
    return selected_model_entry_hashes(protocol, COMPILER_ALIASES)


def stage_model_entry_hashes(
    protocol: ScientificProtocol,
    *,
    stage: str,
    operation: str,
    scope: Mapping[str, object],
) -> dict[str, str]:
    """Include model identities used for calls, token budgets and saved manifests."""
    aliases: tuple[str, ...]
    if operation in {"snapshot", "suite", "query_execute"}:
        aliases = ()
    elif operation == "corpus":
        aliases = (TEXT_ALIAS,)
    elif operation == "units":
        aliases = (COMPILER_ALIAS,)
    elif operation == "adopt_questioner":
        if stage != "questioner":
            raise ValueError("step8_model_identity_invalid_questioner_adoption_stage")
        # No service starts for adoption, but the consumed producer model stays
        # part of the scientific identity.
        aliases = (COMPILER_ALIAS,)
    elif operation == "compiler":
        by_stage = {
            "questioner": (COMPILER_ALIAS,),
            "planner0": (COMPILER_ALIAS,),
            "text_index": (TEXT_ALIAS,),
            "multimodal_index": (MULTIMODAL_ALIAS,),
            "dry_run0": (),
            "r0_gate": (),
            "doubter": (COMPILER_ALIAS,),
            "projection": (),
        }
        if stage not in by_stage:
            raise ValueError("step8_model_identity_unknown_compiler_stage")
        aliases = by_stage[stage]
    elif operation == "query_embedding":
        if stage not in {"text_index", "multimodal_index"}:
            raise ValueError("step8_model_identity_unknown_embedding_stage")
        aliases = (TEXT_ALIAS if stage == "text_index" else MULTIMODAL_ALIAS,)
    elif operation in {"baseline_index", "baseline_retrieve"}:
        embedding = _METHOD_EMBEDDINGS.get(str(scope.get("method")))
        if embedding is None:
            raise ValueError("step8_model_identity_unknown_baseline_method")
        aliases = (embedding,)
        if operation == "baseline_retrieve":
            # Retrieval bootstraps the immutable child run with these model entries.
            aliases += (_answer_alias(scope), GROUNDING_ALIAS)
    elif operation == "query_match":
        aliases = (_answer_alias(scope),)
        if scope.get("treatment") != "no_reranker":
            aliases += (RERANKER_ALIAS,)
    elif operation in {"baseline_answer", "query_answer", "native_fallback_generate"}:
        aliases = (_answer_alias(scope),)
        if operation == "baseline_answer":
            # Full-context children are first registered in the answer phase.
            aliases += (GROUNDING_ALIAS,)
    elif operation in {"baseline_grounding", "query_grounding", "native_fallback_ground"}:
        aliases = (_answer_alias(scope), GROUNDING_ALIAS)
    elif operation in {"evaluate", "primary_evaluate", "analysis_evaluate", "baseline_evaluate"}:
        aliases = (JUDGE_ALIAS,)
        if (
            protocol.oracle_at_50 is not None
            and scope.get("answer_model") == protocol.oracle_at_50.answer_model
            and scope.get("treatment") == protocol.oracle_at_50.treatment
        ):
            aliases += (protocol.oracle_at_50.answer_model, GROUNDING_ALIAS)
    else:
        raise ValueError("step8_model_identity_unknown_operation")
    return selected_model_entry_hashes(protocol, aliases)


def _answer_alias(scope: Mapping[str, object]) -> str:
    value = scope.get("answer_model")
    if not isinstance(value, str) or value not in ANSWER_BACKBONES:
        raise ValueError("step8_model_identity_answer_alias_missing_or_invalid")
    return value
