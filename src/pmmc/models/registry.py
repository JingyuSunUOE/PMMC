"""Strict, secret-free model registry loading and reproducibility hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field, model_validator

from pmmc.models.provider_config import PROVIDER_MODELS, ProviderConfig
from pmmc.schemas import (
    EXPERIMENT_SCHEMA_VERSION,
    BaseSchema,
    DistanceMetric,
    JsonDict,
    ModelCapability,
    ModelEndpointKind,
    RunManifest,
)
from pmmc.schemas.base import NonEmptyStr, PositiveInt, ensure_unique_strings

_UNPINNED_REVISIONS = {
    "latest",
    "main",
    "master",
    "pinned-on-server",
    "unknown",
}
_SECRET_KEYS = {
    "access_key",
    "access_token",
    "api_key",
    "authorization",
    "credentials",
    "password",
    "secret",
    "secret_key",
}


def _assert_secret_free(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SECRET_KEYS:
                raise ValueError(f"model registry cannot contain credential field: {key}")
            _assert_secret_free(item)
    elif isinstance(value, list):
        for item in value:
            _assert_secret_free(item)


class ModelRegistryEntry(BaseSchema):
    """Immutable public configuration for one model alias."""

    alias: NonEmptyStr
    endpoint_kind: ModelEndpointKind
    provider_model: NonEmptyStr
    expected_actual_model_id: NonEmptyStr
    source_repo_id: NonEmptyStr | None = None
    model_revision: NonEmptyStr
    capabilities: list[ModelCapability]
    context_window: PositiveInt | None = None
    tokenizer_name: NonEmptyStr
    tokenizer_revision: NonEmptyStr
    processor_name: NonEmptyStr | None = None
    processor_revision: NonEmptyStr | None = None
    embedding_dimension: PositiveInt | None = None
    normalize_embeddings: bool | None = None
    distance_metric: DistanceMetric | None = None
    provider_config: ProviderConfig | None = Field(default=None, exclude_if=lambda v: v is None)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_capabilities(self) -> ModelRegistryEntry:
        if self.provider_config is not None:
            if self.endpoint_kind is not ModelEndpointKind.LITELLM_OPENAI:
                raise ValueError("provider_config requires a chat endpoint")
            legacy = (
                (self.provider_config.kind, self.provider_model) in {
                    ("openai", "gpt-5-mini-2025-08-07"),
                    ("anthropic", "claude-haiku-4-5-20251001"),
                }
            )
            if not legacy and self.provider_model not in PROVIDER_MODELS[self.provider_config.kind]:
                raise ValueError("Step 8 provider_model must be a supported concrete model ID")
            if self.provider_config.kind == "openai":
                expected_reasoning = "minimal" if legacy else "low"
                if self.provider_config.reasoning_mode != expected_reasoning:
                    raise ValueError("OpenAI model requires its frozen reasoning policy")
            if self.expected_actual_model_id != self.provider_model:
                raise ValueError("Step 8 provider and expected model identities must match")
            if self.alias.startswith(("compiler-", "reranker-", "grounding-", "judge-")) and (
                self.provider_model != "Qwen/Qwen3.5-9B"
            ):
                raise ValueError("Step 8 compiler/reranker/grounding/judge are fixed to Qwen3.5-9B")
            if (
                self.provider_config.kind != "qwen_vllm"
                and self.model_revision != self.provider_model
            ):
                raise ValueError("native provider model_revision must equal the pinned model ID")
        if not self.capabilities:
            raise ValueError("model registry entries require at least one capability")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("model capabilities cannot contain duplicates")

        chat = {ModelCapability.CHAT_TEXT, ModelCapability.CHAT_MULTIMODAL}
        embedding = {
            ModelCapability.EMBED_TEXT,
            ModelCapability.EMBED_IMAGE,
            ModelCapability.EMBED_MULTIMODAL,
        }
        capability_set = set(self.capabilities)
        if capability_set & chat and self.context_window is None:
            raise ValueError("chat models require context_window")
        if capability_set & embedding:
            if self.embedding_dimension is None or self.distance_metric is None:
                raise ValueError("embedding models require dimension and distance metric")
            if self.normalize_embeddings is None:
                raise ValueError("embedding models require an explicit normalization policy")
        elif any(
            value is not None
            for value in (
                self.embedding_dimension,
                self.normalize_embeddings,
                self.distance_metric,
            )
        ):
            raise ValueError("embedding metadata requires an embedding capability")

        if self.endpoint_kind is ModelEndpointKind.LITELLM_OPENAI and not capability_set & chat:
            raise ValueError("LiteLLM registry entries must expose a chat capability")
        if self.endpoint_kind is ModelEndpointKind.VLLM_OPENAI:
            if capability_set != {ModelCapability.EMBED_TEXT}:
                raise ValueError(
                    "vLLM OpenAI embedding entries must expose only embed_text"
                )
        if self.endpoint_kind is ModelEndpointKind.VLLM_V2_EMBED:
            if not capability_set & {ModelCapability.EMBED_IMAGE, ModelCapability.EMBED_MULTIMODAL}:
                raise ValueError("vLLM v2 embed entries require image or multimodal embedding")
        if self.endpoint_kind is ModelEndpointKind.PMMC_CONTRASTIVE:
            if capability_set != {ModelCapability.EMBED_TEXT, ModelCapability.EMBED_IMAGE}:
                raise ValueError(
                    "PMMC contrastive entries must expose embed_text and embed_image"
                )
        if (self.processor_name is None) != (self.processor_revision is None):
            raise ValueError("processor_name and processor_revision must be provided together")
        return self

    def require(self, capability: ModelCapability) -> None:
        """Fail before provider I/O when an alias lacks a required capability."""

        if capability not in self.capabilities:
            raise ValueError(f"model alias {self.alias!r} lacks capability {capability.value!r}")

    @property
    def resolved_hash(self) -> str:
        """Return a stable identity for this alias without endpoint credentials."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def resolved_source_repo_id(self) -> str:
        """Return the Hugging Face repository used by a managed local service."""

        if self.source_repo_id is not None:
            return self.source_repo_id
        if "/" in self.expected_actual_model_id:
            return self.expected_actual_model_id
        raise ValueError(f"model alias {self.alias!r} does not declare source_repo_id")


class ModelRegistry(BaseSchema):
    """Versioned registry whose hash excludes endpoints and credentials."""

    registry_version: NonEmptyStr
    models: list[ModelRegistryEntry]
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_aliases(self) -> ModelRegistry:
        if not self.models:
            raise ValueError("model registry cannot be empty")
        ensure_unique_strings([model.alias for model in self.models], "models.alias")
        _assert_secret_free(self.model_dump(mode="python"))
        return self

    @property
    def resolved_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def resolved_hash_for_aliases(self, aliases: list[str] | tuple[str, ...]) -> str:
        """Hash only the model entries actually used by one immutable run."""

        selected = sorted(set(aliases))
        if not selected:
            raise ValueError("model registry subset cannot be empty")
        payload = {
            "registry_version": self.registry_version,
            "models": [self.get(alias).model_dump(mode="json") for alias in selected],
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, alias: str) -> ModelRegistryEntry:
        try:
            return next(model for model in self.models if model.alias == alias)
        except StopIteration as exc:
            raise KeyError(f"unknown model alias: {alias}") from exc

    def verify_actual_model(self, alias: str, actual_model_id: str, *, paper_mode: bool) -> None:
        """Require exact identity in paper mode and retain diagnostics in dev mode."""

        expected = self.get(alias).expected_actual_model_id
        if paper_mode and actual_model_id != expected:
            raise ValueError(
                f"actual model mismatch for {alias!r}: expected {expected!r}, "
                f"received {actual_model_id!r}"
            )

    def assert_paper_ready(self) -> None:
        """Reject placeholder or floating model revisions before a paper-mode run."""

        problems: list[str] = []
        for model in self.models:
            revisions = {
                "model_revision": model.model_revision,
                "tokenizer_revision": model.tokenizer_revision,
            }
            if model.processor_revision is not None:
                revisions["processor_revision"] = model.processor_revision
            for field_name, revision in revisions.items():
                if revision.strip().lower() in _UNPINNED_REVISIONS:
                    problems.append(f"{model.alias}.{field_name}={revision}")
        if problems:
            raise ValueError(
                "paper mode requires immutable model revisions; replace: "
                + ", ".join(sorted(problems))
            )

    def manifest_fragment(
        self, aliases: list[str] | tuple[str, ...] | None = None
    ) -> dict[str, object]:
        """Return secret-free resolved configuration suitable for RunManifest."""

        selected = (
            sorted(self.models, key=lambda item: item.alias)
            if aliases is None
            else [self.get(alias) for alias in sorted(set(aliases))]
        )

        return {
            "registry_version": self.registry_version,
            "registry_hash": self.resolved_hash,
            "registry_subset_hash": self.resolved_hash_for_aliases(
                [model.alias for model in selected]
            ),
            "models": {
                model.alias: model.model_dump(mode="json")
                for model in selected
            },
        }


def load_model_registry(path: Path) -> ModelRegistry:
    """Load and strictly validate a UTF-8 YAML model registry."""

    import yaml  # type: ignore[import-untyped]

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid model registry YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("model registry root must be a mapping")
    return ModelRegistry.model_validate(raw)


def load_model_registry_overlay(path: Path) -> ModelRegistry:
    """Load a strict overlay, allowing the concise top-level ``models`` form."""

    import yaml  # type: ignore[import-untyped]

    content = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid model registry overlay YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("model registry overlay root must be a mapping")
    if "registry_version" not in raw:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        raw["registry_version"] = f"overlay-{digest}"
    return ModelRegistry.model_validate(raw)


def merge_model_registries(
    base: ModelRegistry,
    overlays: list[ModelRegistry],
) -> ModelRegistry:
    """Merge declarative overlays without allowing silent alias replacement."""

    by_alias = {entry.alias: entry for entry in base.models}
    order = [entry.alias for entry in base.models]
    for overlay in overlays:
        for entry in overlay.models:
            existing = by_alias.get(entry.alias)
            if existing is not None:
                if existing != entry:
                    raise ValueError(
                        f"model registry overlay conflicts with alias: {entry.alias}"
                    )
                continue
            by_alias[entry.alias] = entry
            order.append(entry.alias)
    metadata = dict(base.metadata)
    if overlays:
        metadata["overlay_registry_versions"] = [item.registry_version for item in overlays]
    return ModelRegistry(
        registry_version=base.registry_version,
        models=[by_alias[alias] for alias in order],
        metadata=metadata,
    )


def load_model_registry_with_overlays(
    base_path: Path,
    overlay_paths: list[Path] | tuple[Path, ...] = (),
) -> ModelRegistry:
    """Load a base registry followed by strict, ordered YAML overlays."""

    base = load_model_registry(base_path)
    overlays = [load_model_registry_overlay(path) for path in overlay_paths]
    return merge_model_registries(base, overlays)


def bind_registry_to_manifest(
    manifest: RunManifest,
    registry: ModelRegistry,
    *,
    aliases: list[str] | tuple[str, ...] | None = None,
    registry_identity_scope: str = "full",
) -> RunManifest:
    """Freeze the resolved, secret-free registry into a new run manifest."""

    if registry_identity_scope not in {"full", "selected_aliases"}:
        raise ValueError("unknown registry identity scope")
    if registry_identity_scope == "selected_aliases" and not aliases:
        raise ValueError("selected-alias registry identity requires explicit aliases")
    fragment = registry.manifest_fragment(aliases)
    models = fragment["models"]
    if not isinstance(models, dict):
        raise TypeError("model registry manifest fragment must contain a model mapping")
    environment = dict(manifest.environment_fingerprint)
    environment.update(
        {
            "model_registry_version": registry.registry_version,
            "model_registry_hash": (
                fragment["registry_subset_hash"]
                if registry_identity_scope == "selected_aliases"
                else registry.resolved_hash
            ),
            "model_registry_subset_hash": fragment["registry_subset_hash"],
            "model_registry_aliases": sorted(models),
        }
    )
    if registry_identity_scope == "selected_aliases":
        environment["model_registry_identity_scope"] = "selected_aliases"
    return manifest.model_copy(
        update={
            "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
            "model_configs": models,
            "environment_fingerprint": environment,
            "config_hash": None,
        }
    )


def verify_manifest_registry_compatibility(
    manifest: RunManifest,
    registry: ModelRegistry,
    required_aliases: list[str] | tuple[str, ...],
) -> None:
    """Verify only frozen aliases so unrelated registry additions do not break resume."""

    aliases = sorted(set(required_aliases))
    if not aliases:
        raise ValueError("required model aliases cannot be empty")
    for alias in aliases:
        frozen = manifest.model_configs.get(alias)
        if frozen is None:
            raise ValueError(f"run manifest does not contain model config for {alias!r}")
        current = registry.get(alias).model_dump(mode="json")
        if frozen != current:
            raise ValueError(f"current model config differs from run manifest: {alias}")
    expected_subset = manifest.environment_fingerprint.get("model_registry_subset_hash")
    if (
        expected_subset is not None
        and expected_subset != registry.resolved_hash_for_aliases(aliases)
    ):
        raise ValueError("current model registry subset differs from immutable run manifest")
