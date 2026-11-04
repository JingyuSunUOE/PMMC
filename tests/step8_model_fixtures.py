"""Complete typed offline locks with computed hashes, no Hub or provider calls."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_model_identity import compute_model_entry_hashes
from pmmc.models.registry import ModelRegistry, load_model_registry
from pmmc.schemas import ModelFileRecord, ModelLockEntry, ModelLockManifest

ROOT = Path(__file__).resolve().parents[1]


def model_registry() -> ModelRegistry:
    return load_model_registry(ROOT / "configs/models/step8.yaml")


def model_lock(registry: ModelRegistry | None = None) -> ModelLockManifest:
    registry = registry or model_registry()
    entries, native = [], {}
    for model in registry.models:
        if model.provider_config is not None and model.provider_config.kind != "qwen_vllm":
            native[model.alias] = model.resolved_hash
            continue
        files = []
        for name in (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
        ):
            data = ("offline-fixture:" + model.resolved_source_repo_id + "/" + name).encode()
            files.append(
                ModelFileRecord(path=name, size_bytes=len(data), sha256=sha256(data).hexdigest())
            )
        entries.append(
            ModelLockEntry(
                alias=model.alias,
                repo_id=model.resolved_source_repo_id,
                requested_revision=model.model_revision,
                commit_sha=sha256(model.resolved_source_repo_id.encode()).hexdigest()[:40],
                files=files,
                snapshot_content_hash=canonical_content_hash(
                    [file.model_dump(mode="json") for file in files]
                ),
                registry_entry_hash=model.resolved_hash,
                metadata={
                    "fixture_only": True,
                    "requires_processor": model.processor_name is not None,
                },
            )
        )
    aliases = sorted(entry.alias for entry in entries)
    return ModelLockManifest(
        registry_hash=registry.resolved_hash,
        registry_subset_hash=registry.resolved_hash_for_aliases(aliases),
        selected_aliases=aliases,
        entries=entries,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"native_provider_entry_hashes": native, "fixture_only": True},
    )


def model_entry_hashes() -> dict[str, str]:
    registry = model_registry()
    return compute_model_entry_hashes(registry, model_lock(registry))
