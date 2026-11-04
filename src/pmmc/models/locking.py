"""Immutable Hugging Face model locks and local-cache verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.schemas import (
    ModelEndpointKind,
    ModelFileRecord,
    ModelLockEntry,
    ModelLockManifest,
    ModelVerificationFileState,
    ModelVerificationPolicy,
    ModelVerificationReceipt,
    ModelVerificationResult,
    VerifiedModelCacheManifest,
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FLOATING_REVISIONS = {"latest", "main", "master", "pinned-on-server", "unknown"}
ModelCacheProgress = Callable[[str, str, int, int], None]
_VERIFIER_VERSION = "pmmc-model-cache-verifier-v2"
_NATIVE_PROVIDER_HASHES = "native_provider_entry_hashes"


def _native_provider_hashes(registry: ModelRegistry) -> dict[str, str]:
    """Native snapshots are registry identities, never Hub commits or files."""
    hashes: dict[str, str] = {}
    for model in registry.models:
        if model.provider_config is None or model.provider_config.kind == "qwen_vllm":
            continue
        # Revalidate copies too, before excluding anything from artifact checks.
        ModelRegistryEntry.model_validate(model.model_dump(mode="python"))
        hashes[model.alias] = model.resolved_hash
    return hashes


def _locked_native_provider_hashes(lock: ModelLockManifest) -> dict[str, str]:
    raw = lock.metadata.get(_NATIVE_PROVIDER_HASHES, {})
    if not isinstance(raw, dict) or any(
        not isinstance(alias, str) or not alias or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for alias, digest in raw.items()
    ):
        raise ValueError("invalid native provider registry-entry hashes in model lock")
    if set(raw) & {entry.alias for entry in lock.entries}:
        raise ValueError("native providers cannot have Hugging Face lock entries")
    return dict(raw)


@dataclass(frozen=True)
class ModelCachePreparation:
    paths: dict[str, Path]
    verifications: dict[str, ModelVerificationResult]


class HubApi(Protocol):
    def model_info(
        self,
        repo_id: str,
        *,
        revision: str,
        files_metadata: bool,
        token: str | None,
    ) -> Any: ...


class HubAuthApi(Protocol):
    def whoami(self, token: str) -> Any: ...


class HuggingFaceAuthenticationError(RuntimeError):
    """Raised when a configured Hub token cannot be authenticated safely."""


def validate_hf_token(
    token: str,
    *,
    api: HubAuthApi | None = None,
) -> None:
    """Validate token formatting and prove that the Hub accepts it.

    The function deliberately returns no account metadata and never includes the
    token or provider exception text in its error messages.
    """

    if not token:
        raise HuggingFaceAuthenticationError("HF_TOKEN is empty")
    if token != token.strip() or any(character in token for character in "\r\n"):
        raise HuggingFaceAuthenticationError(
            "HF_TOKEN contains leading/trailing whitespace or a line break"
        )
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        raise HuggingFaceAuthenticationError(
            "HF_TOKEN is surrounded by quotes; remove the quotes in the env file"
        )
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    try:
        identity = api.whoami(token=token)
    except Exception as exc:
        raise HuggingFaceAuthenticationError(
            "Hugging Face rejected HF_TOKEN; verify that it is active and has read access"
        ) from exc
    if not identity:
        raise HuggingFaceAuthenticationError("Hugging Face returned no identity for HF_TOKEN")


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sibling_file(sibling: object) -> ModelFileRecord:
    path = str(getattr(sibling, "rfilename", "")).strip()
    if not path or path.startswith("/") or ".." in Path(path).parts:
        raise ValueError("Hub model files must use safe relative paths")
    size = getattr(sibling, "size", None)
    lfs = getattr(sibling, "lfs", None)
    sha256 = None
    if lfs is not None:
        if isinstance(lfs, dict):
            sha256 = lfs.get("sha256") or lfs.get("oid")
            size = size if size is not None else lfs.get("size")
        else:
            sha256 = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            size = size if size is not None else getattr(lfs, "size", None)
    if isinstance(sha256, str) and sha256.startswith("sha256:"):
        sha256 = sha256.split(":", 1)[1]
    return ModelFileRecord(
        path=path,
        size_bytes=max(0, int(size or 0)),
        sha256=str(sha256) if sha256 else None,
    )


def build_model_lock(
    registry: ModelRegistry,
    *,
    aliases: Iterable[str] | None = None,
    token: str | None = None,
    api: HubApi | None = None,
) -> ModelLockManifest:
    """Lock Hub artifacts; bind native snapshots using existing manifest metadata."""

    native_hashes = _native_provider_hashes(registry)
    real_models = [
        entry for entry in registry.models if entry.endpoint_kind is not ModelEndpointKind.FAKE
    ]
    selected = set(aliases) if aliases is not None else {entry.alias for entry in real_models}
    unknown = selected - {entry.alias for entry in registry.models}
    if unknown:
        raise KeyError(f"unknown model alias: {sorted(unknown)[0]}")
    unmanageable = selected - {entry.alias for entry in real_models}
    if unmanageable:
        raise ValueError(f"fake model aliases cannot be locked: {sorted(unmanageable)[0]}")
    lockable_models = [
        entry for entry in real_models
        if entry.alias in selected and entry.alias not in native_hashes
    ]
    if not lockable_models:
        raise ValueError(
            "no Hugging Face artifacts selected; native-only models need no model lock"
        )
    if token is not None:
        validate_hf_token(token)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    entries: list[ModelLockEntry] = []
    for model in sorted(lockable_models, key=lambda item: item.alias):
        if model.alias not in selected:
            continue
        requested = model.model_revision
        hub_revision = "main" if requested.strip().lower() in _FLOATING_REVISIONS else requested
        info = api.model_info(
            model.resolved_source_repo_id,
            revision=hub_revision,
            files_metadata=True,
            token=token,
        )
        commit = str(getattr(info, "sha", "")).strip().lower()
        if not _COMMIT_RE.fullmatch(commit):
            raise ValueError(f"Hub did not return an immutable commit for {model.alias!r}")
        files = sorted(
            (_sibling_file(item) for item in (getattr(info, "siblings", None) or [])),
            key=lambda item: item.path,
        )
        if not files:
            raise ValueError(f"Hub model repository has no files: {model.resolved_source_repo_id}")
        content_hash = _stable_hash([item.to_dict() for item in files])
        entries.append(
            ModelLockEntry(
                alias=model.alias,
                repo_id=model.resolved_source_repo_id,
                requested_revision=requested,
                commit_sha=commit,
                files=files,
                snapshot_content_hash=content_hash,
                registry_entry_hash=model.resolved_hash,
                metadata={
                    "capabilities": sorted(item.value for item in model.capabilities),
                    "requires_processor": model.processor_name is not None,
                },
            )
        )
    selected_aliases = sorted(item.alias for item in entries)
    return ModelLockManifest(
        registry_hash=registry.resolved_hash,
        registry_subset_hash=registry.resolved_hash_for_aliases(selected_aliases),
        selected_aliases=selected_aliases,
        entries=entries,
        metadata={_NATIVE_PROVIDER_HASHES: {
            alias: native_hashes[alias] for alias in sorted(selected & native_hashes.keys())
        }} if selected & native_hashes.keys() else {},
    )


def write_model_lock(lock: ModelLockManifest, path: Path) -> None:
    """Atomically write a model lock as stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(lock.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_model_lock(path: Path) -> ModelLockManifest:
    return ModelLockManifest.from_json(path.read_text(encoding="utf-8"))


def assert_paper_lock(
    lock: ModelLockManifest,
    registry: ModelRegistry,
    *,
    aliases: Iterable[str] | None = None,
) -> None:
    """Check Hub artifacts and separately hashed native-provider registry identities."""

    native_hashes = _native_provider_hashes(registry)
    locked_native = _locked_native_provider_hashes(lock)
    locked_aliases = {entry.alias for entry in lock.entries}
    if locked_aliases & native_hashes.keys():
        raise ValueError("native providers cannot have Hugging Face lock entries")
    for alias, digest in locked_native.items():
        if native_hashes.get(alias) != digest:
            raise ValueError(f"native provider model lock identity differs from registry: {alias}")
    required_aliases = (
        set(aliases) if aliases is not None
        else set(lock.selected_aliases or locked_aliases) | locked_native.keys()
    )
    covered_aliases = locked_aliases | locked_native.keys()
    if not required_aliases <= covered_aliases:
        missing = sorted(required_aliases - covered_aliases)[0]
        raise ValueError(f"paper model lock is missing required alias: {missing}")
    if lock.registry_subset_hash is not None:
        if set(lock.selected_aliases) != locked_aliases:
            raise ValueError("model lock selected_aliases differ from its entries")
        expected = registry.resolved_hash_for_aliases(sorted(locked_aliases))
        if lock.registry_subset_hash != expected:
            raise ValueError("model lock registry subset differs from the active registry")
        for entry in lock.entries:
            expected_entry = registry.get(entry.alias).resolved_hash
            if entry.registry_entry_hash != expected_entry:
                raise ValueError(f"model lock entry differs from registry: {entry.alias}")
    else:
        if lock.registry_hash != registry.resolved_hash:
            raise ValueError("legacy model lock registry hash does not match the active registry")
        registry_aliases = {
            entry.alias
            for entry in registry.models
            if entry.endpoint_kind is not ModelEndpointKind.FAKE
            and entry.alias not in native_hashes
        }
        if locked_aliases != registry_aliases:
            raise ValueError("legacy paper model lock must cover every registry alias")
    for entry in lock.entries:
        if not _COMMIT_RE.fullmatch(entry.commit_sha):
            raise ValueError(f"paper model lock has a non-commit revision: {entry.alias}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_lock_hash(entry: ModelLockEntry) -> str:
    return _stable_hash(
        {
            "repo_id": entry.repo_id,
            "commit_sha": entry.commit_sha,
            "files": [item.to_dict() for item in entry.files],
            "snapshot_content_hash": entry.snapshot_content_hash,
        }
    )


def _receipt_path(cache_dir: Path, entry: ModelLockEntry) -> Path:
    repo_hash = hashlib.sha256(entry.repo_id.encode("utf-8")).hexdigest()[:24]
    return (
        cache_dir
        / ".pmmc"
        / "verification"
        / repo_hash
        / entry.commit_sha
        / _physical_lock_hash(entry)
        / f"{_VERIFIER_VERSION}.json"
    )


def _file_state(
    candidate: Path,
    *,
    relative_path: str,
    sha256: str | None = None,
) -> ModelVerificationFileState:
    lexical = candidate.lstat()
    target = candidate.resolve(strict=True)
    target_stat = target.stat()
    return ModelVerificationFileState(
        path=relative_path,
        size_bytes=target_stat.st_size,
        mtime_ns=lexical.st_mtime_ns,
        symlink_target=os.readlink(candidate) if candidate.is_symlink() else None,
        blob_size_bytes=target_stat.st_size,
        blob_mtime_ns=target_stat.st_mtime_ns,
        sha256=sha256,
    )


def _receipt_payload(receipt: ModelVerificationReceipt) -> dict[str, object]:
    return receipt.model_dump(
        mode="json",
        exclude={"content_hash", "created_at"},
    )


def _write_receipt(
    *,
    cache_dir: Path,
    entry: ModelLockEntry,
    files: list[ModelVerificationFileState],
) -> ModelVerificationReceipt:
    identity = {
        "schema_version": "0.1.0",
        "repo_id": entry.repo_id,
        "commit_sha": entry.commit_sha,
        "physical_lock_hash": _physical_lock_hash(entry),
        "verifier_version": _VERIFIER_VERSION,
        "files": [item.model_dump(mode="json") for item in files],
        "metadata": {},
    }
    verification_id = "model-verification-" + _stable_hash(identity)[:32]
    provisional = ModelVerificationReceipt(
        verification_id=verification_id,
        repo_id=entry.repo_id,
        commit_sha=entry.commit_sha,
        physical_lock_hash=_physical_lock_hash(entry),
        verifier_version=_VERIFIER_VERSION,
        files=files,
        content_hash="pending",
    )
    receipt = provisional.model_copy(
        update={"content_hash": _stable_hash(_receipt_payload(provisional))}
    )
    path = _receipt_path(cache_dir, entry)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    return receipt


def _load_valid_receipt(
    *,
    cache_dir: Path,
    entry: ModelLockEntry,
    snapshot: Path,
) -> ModelVerificationReceipt | None:
    path = _receipt_path(cache_dir, entry)
    if not path.is_file():
        return None
    try:
        receipt = ModelVerificationReceipt.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        receipt.repo_id != entry.repo_id
        or receipt.commit_sha != entry.commit_sha
        or receipt.physical_lock_hash != _physical_lock_hash(entry)
        or receipt.verifier_version != _VERIFIER_VERSION
        or receipt.content_hash != _stable_hash(_receipt_payload(receipt))
    ):
        return None
    expected_paths = [item.path for item in entry.files]
    if [item.path for item in receipt.files] != expected_paths:
        return None
    for record, frozen in zip(entry.files, receipt.files, strict=True):
        candidate = snapshot / record.path
        if not candidate.is_file():
            return None
        try:
            current = _file_state(candidate, relative_path=record.path, sha256=frozen.sha256)
        except OSError:
            return None
        if current != frozen:
            return None
    return receipt


def _validate_snapshot_structure(entry: ModelLockEntry, snapshot: Path) -> None:
    locked_paths = {record.path for record in entry.files}
    if "config.json" not in locked_paths:
        raise ValueError(f"locked model has no config.json: {entry.alias}")
    if not any(path.endswith(".safetensors") for path in locked_paths):
        raise ValueError(f"locked model has no safetensors weights: {entry.alias}")
    if not ({"tokenizer.json", "tokenizer_config.json"} & locked_paths):
        raise ValueError(f"locked model has no tokenizer metadata: {entry.alias}")
    if entry.metadata.get("requires_processor") and not (
        {"preprocessor_config.json", "processor_config.json"} & locked_paths
    ):
        raise ValueError(f"locked multimodal model has no processor metadata: {entry.alias}")
    for record in entry.files:
        relative = Path(record.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model lock file escapes the snapshot directory")
        candidate = snapshot / relative
        if not candidate.is_file():
            raise FileNotFoundError(f"locked model file is missing: {entry.alias}/{record.path}")
        if candidate.is_symlink():
            repository_cache = snapshot.parent.parent
            target = candidate.resolve(strict=True)
            if not target.is_relative_to(repository_cache):
                raise ValueError(
                    "locked model symlink escapes its Hub repository cache: "
                    f"{entry.alias}/{record.path}"
                )
        if record.size_bytes and candidate.stat().st_size != record.size_bytes:
            raise ValueError(f"locked model file size mismatch: {entry.alias}/{record.path}")


def _verified_cache_manifest_payload(
    manifest: VerifiedModelCacheManifest,
) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"content_hash"})


def write_verified_model_cache_manifest(
    *,
    destination: Path,
    lock_path: Path,
    lock: ModelLockManifest,
    preparation: ModelCachePreparation,
    verification_policy: ModelVerificationPolicy,
) -> VerifiedModelCacheManifest:
    """Freeze one launch's verified local snapshots for managed stage reuse."""

    aliases = sorted(preparation.paths)
    if aliases != sorted(preparation.verifications):
        raise ValueError("model-cache paths and verification results differ")
    locked_aliases = {entry.alias for entry in lock.entries}
    if not set(aliases) <= locked_aliases:
        raise ValueError("model-cache preparation contains an unlocked alias")
    resolved_lock = lock_path.resolve(strict=True)
    provisional = VerifiedModelCacheManifest(
        lock_path=str(resolved_lock),
        lock_sha256=_sha256_file(resolved_lock),
        registry_identity_hash=lock.registry_subset_hash or lock.registry_hash,
        selected_aliases=aliases,
        snapshot_paths={
            alias: str(preparation.paths[alias].resolve(strict=True)) for alias in aliases
        },
        verifications={alias: preparation.verifications[alias] for alias in aliases},
        verification_policy=verification_policy,
        content_hash="pending",
    )
    manifest = provisional.model_copy(
        update={"content_hash": _stable_hash(_verified_cache_manifest_payload(provisional))}
    )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(destination)
    return manifest


def load_verified_model_cache_manifest(
    manifest_path: Path,
    *,
    lock_path: Path,
    lock: ModelLockManifest,
    aliases: Iterable[str],
    verification_policy: ModelVerificationPolicy,
) -> ModelCachePreparation:
    """Validate and reuse a launch preparation without repeating receipt traversal."""

    manifest = VerifiedModelCacheManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.content_hash != _stable_hash(_verified_cache_manifest_payload(manifest)):
        raise ValueError("verified model-cache manifest content hash differs")
    resolved_lock = lock_path.resolve(strict=True)
    if Path(manifest.lock_path).resolve(strict=True) != resolved_lock:
        raise ValueError("verified model-cache manifest references another lock")
    if manifest.lock_sha256 != _sha256_file(resolved_lock):
        raise ValueError("verified model-cache manifest lock hash differs")
    if manifest.registry_identity_hash != (lock.registry_subset_hash or lock.registry_hash):
        raise ValueError("verified model-cache manifest registry identity differs")
    if manifest.verification_policy is not verification_policy:
        raise ValueError("verified model-cache manifest policy differs")
    required = set(aliases) - _locked_native_provider_hashes(lock).keys()
    if not required <= set(manifest.selected_aliases):
        missing = sorted(required - set(manifest.selected_aliases))[0]
        raise KeyError(f"verified model-cache manifest is missing alias: {missing}")

    paths: dict[str, Path] = {}
    verifications: dict[str, ModelVerificationResult] = {}
    for alias in sorted(required):
        entry = lock.get(alias)
        snapshot = Path(manifest.snapshot_paths[alias]).resolve(strict=True)
        verification = manifest.verifications[alias]
        if (
            verification.repo_id != entry.repo_id
            or verification.commit_sha != entry.commit_sha
            or Path(verification.snapshot_path).resolve(strict=True) != snapshot
        ):
            raise ValueError(f"verified model-cache identity differs: {alias}")
        _validate_snapshot_structure(entry, snapshot)
        paths[alias] = snapshot
        verifications[alias] = verification.model_copy(
            update={
                "bytes_hashed": 0,
                "metadata": {
                    **verification.metadata,
                    "launch_preparation_reused": True,
                    "preparation_manifest_hash": manifest.content_hash,
                },
            }
        )
    return ModelCachePreparation(paths=paths, verifications=verifications)


def verify_snapshot_with_policy(
    entry: ModelLockEntry,
    snapshot: Path,
    cache_dir: Path,
    *,
    policy: ModelVerificationPolicy,
    allow_full_fallback: bool = True,
) -> ModelVerificationResult:
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"model snapshot directory does not exist: {entry.alias}")
    _validate_snapshot_structure(entry, snapshot)
    if policy is ModelVerificationPolicy.IDENTITY_ONLY:
        return ModelVerificationResult(
            alias=entry.alias,
            repo_id=entry.repo_id,
            commit_sha=entry.commit_sha,
            snapshot_path=str(snapshot),
            policy=policy,
            verified_files=len(entry.files),
            metadata={"verifier_version": _VERIFIER_VERSION},
        )
    if policy is ModelVerificationPolicy.CACHED_RECEIPT:
        receipt = _load_valid_receipt(
            cache_dir=cache_dir,
            entry=entry,
            snapshot=snapshot,
        )
        if receipt is not None:
            return ModelVerificationResult(
                alias=entry.alias,
                repo_id=entry.repo_id,
                commit_sha=entry.commit_sha,
                snapshot_path=str(snapshot),
                policy=policy,
                receipt_id=receipt.verification_id,
                receipt_cache_hit=True,
                verified_files=len(entry.files),
                metadata={"verifier_version": _VERIFIER_VERSION},
            )
        if not allow_full_fallback:
            raise FileNotFoundError(f"valid model verification receipt is missing: {entry.alias}")

    files: list[ModelVerificationFileState] = []
    bytes_hashed = 0
    for record in entry.files:
        candidate = snapshot / record.path
        digest = _sha256_file(candidate)
        if record.sha256 and digest != record.sha256:
            raise ValueError(f"locked model file checksum mismatch: {entry.alias}/{record.path}")
        bytes_hashed += candidate.stat().st_size
        files.append(_file_state(candidate, relative_path=record.path, sha256=digest))
    receipt = _write_receipt(
        cache_dir=cache_dir,
        entry=entry,
        files=files,
    )
    return ModelVerificationResult(
        alias=entry.alias,
        repo_id=entry.repo_id,
        commit_sha=entry.commit_sha,
        snapshot_path=str(snapshot),
        policy=ModelVerificationPolicy.FULL,
        receipt_id=receipt.verification_id,
        receipt_cache_hit=False,
        verified_files=len(entry.files),
        bytes_hashed=bytes_hashed,
        metadata={
            "requested_policy": policy.value,
            "verifier_version": _VERIFIER_VERSION,
        },
    )


def _verify_snapshot(entry: ModelLockEntry, snapshot: Path) -> dict[str, object]:
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"model snapshot directory does not exist: {entry.alias}")
    verified = 0
    for record in entry.files:
        relative = Path(record.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model lock file escapes the snapshot directory")
        # Hub snapshots intentionally symlink into the sibling blobs directory.
        # Validate the lexical lock path while allowing that cache-managed target.
        candidate = snapshot / relative
        if not candidate.is_file():
            raise FileNotFoundError(f"locked model file is missing: {entry.alias}/{record.path}")
        if candidate.is_symlink():
            repository_cache = snapshot.parent.parent
            target = candidate.resolve(strict=True)
            if not target.is_relative_to(repository_cache):
                raise ValueError(
                    f"locked model symlink escapes its Hub repository cache: "
                    f"{entry.alias}/{record.path}"
                )
        if record.size_bytes and candidate.stat().st_size != record.size_bytes:
            raise ValueError(f"locked model file size mismatch: {entry.alias}/{record.path}")
        if record.sha256 and _sha256_file(candidate) != record.sha256:
            raise ValueError(f"locked model file checksum mismatch: {entry.alias}/{record.path}")
        verified += 1
    locked_paths = {record.path for record in entry.files}
    if "config.json" not in locked_paths:
        raise ValueError(f"locked model has no config.json: {entry.alias}")
    if not any(path.endswith(".safetensors") for path in locked_paths):
        raise ValueError(f"locked model has no safetensors weights: {entry.alias}")
    if not ({"tokenizer.json", "tokenizer_config.json"} & locked_paths):
        raise ValueError(f"locked model has no tokenizer metadata: {entry.alias}")
    if entry.metadata.get("requires_processor") and not (
        {"preprocessor_config.json", "processor_config.json"} & locked_paths
    ):
        raise ValueError(f"locked multimodal model has no processor metadata: {entry.alias}")
    return {
        "alias": entry.alias,
        "commit_sha": entry.commit_sha,
        "snapshot_path": str(snapshot),
        "verified_files": verified,
    }


def prepare_model_cache_audited(
    lock: ModelLockManifest,
    cache_dir: Path,
    *,
    token: str | None = None,
    aliases: Iterable[str] | None = None,
    local_files_only: bool = False,
    progress: ModelCacheProgress | None = None,
    verification_policy: ModelVerificationPolicy = ModelVerificationPolicy.FULL,
    allow_full_fallback: bool = True,
) -> ModelCachePreparation:
    """Materialize locked snapshots and return auditable verification results."""

    native_hashes = _locked_native_provider_hashes(lock)
    selected = (
        set(aliases) if aliases is not None else {entry.alias for entry in lock.entries}
    )
    selected -= native_hashes.keys()
    unknown = selected - {entry.alias for entry in lock.entries}
    if unknown:
        raise KeyError(f"model alias is not present in lock: {sorted(unknown)[0]}")
    if not selected:
        return ModelCachePreparation(paths={}, verifications={})

    from huggingface_hub import snapshot_download

    paths: dict[str, Path] = {}
    verifications: dict[str, ModelVerificationResult] = {}
    physical_verifications: dict[tuple[str, str, str], tuple[Path, ModelVerificationResult]] = {}
    token_validated = False
    selected_entries = [entry for entry in lock.entries if entry.alias in selected]
    for index, entry in enumerate(selected_entries, start=1):
        if progress is not None:
            progress(entry.alias, "checking", index - 1, len(selected_entries))
        physical_key = (entry.repo_id, entry.commit_sha, _physical_lock_hash(entry))
        previous = physical_verifications.get(physical_key)
        if previous is not None:
            previous_path, previous_result = previous
            _validate_snapshot_structure(entry, previous_path)
            paths[entry.alias] = previous_path
            verifications[entry.alias] = previous_result.model_copy(
                update={
                    "alias": entry.alias,
                    "bytes_hashed": 0,
                    "metadata": {
                        **previous_result.metadata,
                        "physical_verification_reused_from": previous_result.alias,
                    },
                }
            )
            if progress is not None:
                progress(entry.alias, "verified", index, len(selected_entries))
            continue
        snapshot: Path | None = None
        verification: ModelVerificationResult | None = None
        if not local_files_only:
            try:
                cached = Path(
                    snapshot_download(
                        repo_id=entry.repo_id,
                        revision=entry.commit_sha,
                        cache_dir=str(cache_dir),
                        token=token,
                        local_files_only=True,
                    )
                )
                verification = verify_snapshot_with_policy(
                    entry,
                    cached,
                    cache_dir,
                    policy=verification_policy,
                    allow_full_fallback=allow_full_fallback,
                )
                snapshot = cached
                if progress is not None:
                    progress(entry.alias, "cached", index, len(selected_entries))
            except (FileNotFoundError, OSError, ValueError):
                snapshot = None
                verification = None
        if snapshot is None:
            if token is not None and not local_files_only and not token_validated:
                validate_hf_token(token)
                token_validated = True
            if progress is not None:
                progress(
                    entry.alias,
                    "checking" if local_files_only else "downloading",
                    index - 1,
                    len(selected_entries),
                )
            snapshot = Path(
                snapshot_download(
                    repo_id=entry.repo_id,
                    revision=entry.commit_sha,
                    cache_dir=str(cache_dir),
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        if verification is None:
            verification = verify_snapshot_with_policy(
                entry,
                snapshot,
                cache_dir,
                policy=verification_policy,
                allow_full_fallback=allow_full_fallback,
            )
        resolved_snapshot = snapshot.resolve()
        paths[entry.alias] = resolved_snapshot
        verifications[entry.alias] = verification
        physical_verifications[physical_key] = (resolved_snapshot, verification)
        if progress is not None:
            progress(entry.alias, "verified", index, len(selected_entries))
    missing = selected - set(paths)
    if missing:
        raise KeyError(f"model alias is not present in lock: {sorted(missing)[0]}")
    return ModelCachePreparation(paths=paths, verifications=verifications)


def prepare_model_cache(
    lock: ModelLockManifest,
    cache_dir: Path,
    *,
    token: str | None = None,
    aliases: Iterable[str] | None = None,
    local_files_only: bool = False,
    progress: ModelCacheProgress | None = None,
    verification_policy: ModelVerificationPolicy = ModelVerificationPolicy.FULL,
    allow_full_fallback: bool = True,
) -> dict[str, Path]:
    """Compatibility wrapper returning only materialized snapshot paths."""

    return prepare_model_cache_audited(
        lock,
        cache_dir,
        token=token,
        aliases=aliases,
        local_files_only=local_files_only,
        progress=progress,
        verification_policy=verification_policy,
        allow_full_fallback=allow_full_fallback,
    ).paths


def verify_model_cache(
    lock: ModelLockManifest,
    cache_dir: Path,
    *,
    aliases: Iterable[str] | None = None,
    verification_policy: ModelVerificationPolicy = ModelVerificationPolicy.FULL,
    allow_full_fallback: bool = True,
) -> list[dict[str, object]]:
    preparation = prepare_model_cache_audited(
        lock,
        cache_dir,
        aliases=aliases,
        local_files_only=True,
        verification_policy=verification_policy,
        allow_full_fallback=allow_full_fallback,
    )
    return [
        preparation.verifications[alias].model_dump(mode="json")
        for alias in sorted(preparation.paths)
    ]


def resolve_registry_from_lock(
    registry: ModelRegistry,
    lock: ModelLockManifest,
) -> ModelRegistry:
    """Resolve Hub commits while preserving pinned native-provider entries unchanged."""

    assert_paper_lock(lock, registry)
    locked = {entry.alias: entry for entry in lock.entries}
    models = []
    for model in registry.models:
        if model.endpoint_kind is ModelEndpointKind.FAKE or (
            model.provider_config is not None and model.provider_config.kind != "qwen_vllm"
        ):
            models.append(model)
            continue
        locked_entry = locked.get(model.alias)
        if locked_entry is None:
            models.append(model)
            continue
        commit = locked_entry.commit_sha
        models.append(
            model.model_copy(
                update={
                    "model_revision": commit,
                    "tokenizer_revision": commit,
                    "processor_revision": commit if model.processor_revision else None,
                }
            )
        )
    return registry.model_copy(update={"models": models})


def write_resolved_registry(registry: ModelRegistry, path: Path) -> None:
    """Atomically write a resolved registry for child PMMC commands."""

    import yaml  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            registry.model_dump(mode="json", exclude_none=True),
            allow_unicode=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
