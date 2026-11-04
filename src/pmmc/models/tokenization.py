"""Exact and explicitly estimated token counters."""

from __future__ import annotations

import base64
import json
import math
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from pmmc.schemas import (
    GenerationRequest,
    ImageRef,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    SpeakerRole,
)


class HeuristicTokenCounter:
    """Deterministic development counter; never represented as exact."""

    def __init__(self, alias: str = "pmmc-heuristic", revision: str = "v1") -> None:
        self._alias = alias
        self._revision = revision

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def estimated(self) -> bool:
        return True

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        words = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        return max(1, math.ceil(words * 1.25))

    def count_image(self, *, width: int | None = None, height: int | None = None) -> int:
        del width, height
        return 256

    def count_request(self, request: GenerationRequest) -> int:
        total = 0
        for message in request.messages:
            total += 4
            for part in message.parts:
                if part.kind is ModelContentKind.TEXT:
                    total += self.count_text(part.text or "")
                else:
                    assert part.image is not None
                    total += self.count_image(width=part.image.width, height=part.image.height)
        return total + 2


class TransformersTokenCounter:
    """Exact text tokenizer with a deterministic processor image-token callback."""

    def __init__(
        self,
        tokenizer_name: str,
        revision: str,
        *,
        image_token_counter: Any | None = None,
        local_files_only: bool = False,
        cache_dir: Path | None = None,
        snapshot_path: Path | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        tokenizer_source = tokenizer_name
        tokenizer_options: dict[str, object] = {
            "local_files_only": local_files_only,
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }
        if snapshot_path is None:
            tokenizer_options["revision"] = revision
        else:
            tokenizer_source = str(snapshot_path.resolve(strict=True))
            tokenizer_options["local_files_only"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **tokenizer_options,
        )
        self._alias = tokenizer_name
        self._revision = revision
        self._image_token_counter = image_token_counter

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def estimated(self) -> bool:
        return False

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_image(self, *, width: int | None = None, height: int | None = None) -> int:
        if self._image_token_counter is None:
            raise RuntimeError("exact multimodal counting requires a processor image-token counter")
        return int(self._image_token_counter(width=width, height=height))

    def count_request(self, request: GenerationRequest) -> int:
        messages: list[dict[str, str]] = []
        image_tokens = 0
        for message in request.messages:
            text_parts: list[str] = []
            for part in message.parts:
                if part.kind is ModelContentKind.TEXT:
                    text_parts.append(part.text or "")
                else:
                    assert part.image is not None
                    text_parts.append("<image>")
                    image_tokens += self.count_image(
                        width=part.image.width,
                        height=part.image.height,
                    )
            messages.append({"role": message.role.value, "content": "\n".join(text_parts)})
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.count_text(str(rendered)) + image_tokens


class ProcessorTokenCounter:
    """Exact chat-template and image-processor counter for real LVLM requests."""

    def __init__(
        self,
        *,
        alias: str,
        tokenizer_name: str,
        tokenizer_revision: str,
        processor_name: str,
        processor_revision: str,
        project_root: Path,
        local_files_only: bool = False,
        require_image_metadata: bool = True,
        cache_dir: Path | None = None,
        local_asset_cache_dir: Path | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from transformers import AutoProcessor

        self.processor_kwargs = deepcopy(processor_kwargs or {})
        self.chat_template_kwargs = deepcopy(chat_template_kwargs or {})
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            revision=processor_revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        tokenizer: Any = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                revision=tokenizer_revision,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        self.tokenizer: Any = tokenizer
        self._alias = alias
        self.source_identity = {
            "alias": alias, "tokenizer_name": tokenizer_name,
            "tokenizer_revision": tokenizer_revision, "processor_name": processor_name,
            "processor_revision": processor_revision,
        }
        self._revision = f"{tokenizer_revision}+{processor_revision}"
        self.project_root = project_root.resolve()
        self.require_image_metadata = require_image_metadata
        self.local_asset_cache_dir = (
            local_asset_cache_dir.resolve()
            if local_asset_cache_dir is not None
            else None
        )
        self._request_cache: dict[str, int] = {}
        image_processor = getattr(self.processor, "image_processor", None)
        image_processor_config = (
            image_processor.to_dict()
            if image_processor is not None and hasattr(image_processor, "to_dict")
            else {}
        )
        processor_payload = {
            "processor_class": type(self.processor).__qualname__,
            "image_processor_class": (
                type(image_processor).__qualname__ if image_processor is not None else None
            ),
            "image_processor_config": image_processor_config,
        }
        if self.processor_kwargs or self.chat_template_kwargs:
            processor_payload.update(
                processor_kwargs=self.processor_kwargs,
                chat_template_kwargs=self.chat_template_kwargs,
            )
        self._processor_fingerprint = sha256(
            json.dumps(
                processor_payload,
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if self.processor_kwargs or self.chat_template_kwargs:
            self._revision += f":options-{self._processor_fingerprint}"
        self._image_cache: dict[
            tuple[str, int | None, int | None, str, str], int
        ] = {}
        self._decoded_image_cache: dict[
            tuple[str, int | None, int | None, str, str], Any
        ] = {}

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def estimated(self) -> bool:
        return False

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _image_cache_key(
        self,
        image: ImageRef,
    ) -> tuple[str, int | None, int | None, str, str]:
        return (
            image.sha256 or f"{image.image_id}:{image.local_path or ''}",
            image.width,
            image.height,
            self._revision,
            self._processor_fingerprint,
        )

    def _open_image(self, image: ImageRef):  # type: ignore[no-untyped-def]
        from PIL import Image

        cache_key = self._image_cache_key(image)
        if cache_key in self._decoded_image_cache:
            return self._decoded_image_cache[cache_key]
        if image.local_path is None:
            raise RuntimeError("exact multimodal counting requires a local image path")
        path = (self.project_root / image.local_path).resolve()
        if not path.is_relative_to(self.project_root) or not path.is_file():
            raise RuntimeError(f"exact multimodal image is unavailable: {image.image_id}")
        payload = path.read_bytes()
        digest = sha256(payload).hexdigest()
        if self.require_image_metadata and image.sha256 is None:
            raise RuntimeError(f"exact multimodal image SHA256 is missing: {image.image_id}")
        if image.sha256 is not None and digest != image.sha256.lower():
            raise RuntimeError(f"exact multimodal image SHA256 mismatch: {image.image_id}")
        image_path = path
        if self.local_asset_cache_dir is not None:
            image_path = self._cache_local_asset(payload, digest)
        opened = Image.open(image_path)
        opened.load()
        if self.require_image_metadata and (image.width is None or image.height is None):
            raise RuntimeError(
                f"exact multimodal image dimensions are missing: {image.image_id}"
            )
        if image.width is not None and opened.width != image.width:
            raise RuntimeError(f"exact multimodal image dimensions mismatch: {image.image_id}")
        if image.height is not None and opened.height != image.height:
            raise RuntimeError(f"exact multimodal image dimensions mismatch: {image.image_id}")
        decoded = opened.convert("RGB")
        self._decoded_image_cache[cache_key] = decoded
        return decoded

    def _cache_local_asset(self, payload: bytes, digest: str) -> Path:
        assert self.local_asset_cache_dir is not None
        target = self.local_asset_cache_dir / digest[:2] / f"{digest}.image"
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(target.parent, 0o700)
        if not target.is_file():
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            temporary.write_bytes(payload)
            os.chmod(temporary, 0o600)
            try:
                temporary.replace(target)
            except FileNotFoundError:
                if not target.is_file():
                    raise
        if target.stat().st_size != len(payload):
            raise RuntimeError("local benchmark asset cache size mismatch")
        return target

    def _count_parts(self, messages: list[ModelMessage]) -> int:
        rendered_messages: list[dict[str, object]] = []
        images: list[Any] = []
        for message in messages:
            content: list[dict[str, object]] = []
            for part in message.parts:
                if part.kind is ModelContentKind.TEXT:
                    content.append({"type": "text", "text": part.text or ""})
                else:
                    assert part.image is not None
                    images.append(self._open_image(part.image))
                    content.append({"type": "image"})
            rendered_messages.append({"role": message.role.value, "content": content})
        return self._encode_parts(rendered_messages, images)

    def count_transport_messages(
        self, messages: list[dict[str, Any]], *, _image_tokens: list[int] | None = None,
    ) -> int:
        """Count final inline transport bytes, not potentially stale ImageRef metadata."""
        from PIL import Image

        rendered_messages = []
        images = []
        for message in messages:
            content = []
            for part in message["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                else:
                    header, data = part["image_url"]["url"].split(",", 1)
                    if not header.startswith("data:image/") or not header.endswith(";base64"):
                        raise ValueError("exact transport counting requires inline images")
                    with Image.open(BytesIO(base64.b64decode(data, validate=True))) as opened:
                        images.append(opened.convert("RGB"))
                    content.append({"type": "image"})
            rendered_messages.append({"role": message["role"], "content": content})
        return self._encode_parts(rendered_messages, images, image_tokens=_image_tokens)

    def count_transport_profile(self, messages: list[dict[str, Any]]) -> tuple[int, list[int]]:
        """Return actual processor grids as well as the complete template token count."""
        tokens: list[int] = []
        count = self.count_transport_messages(messages, _image_tokens=tokens)
        return count, tokens

    def _encode_parts(
        self, rendered_messages: list, images: list[Any], *, image_tokens: list[int] | None = None,
    ) -> int:
        rendered = self.processor.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        if not images:
            return len(self.tokenizer.encode(str(rendered), add_special_tokens=False))
        encoded = self.processor(
            text=[str(rendered)],
            images=images,
            padding=False,
            return_tensors="pt",
            **self.processor_kwargs,
        )
        if image_tokens is not None:
            grids = encoded.get("image_grid_thw")
            merge = getattr(self.processor.image_processor, "merge_size", None)
            if grids is None or type(merge) is not int or merge <= 0:
                raise ValueError("exact processor image grids unavailable")
            rows = grids.tolist()
            if len(rows) != len(images):
                raise ValueError("exact processor image grid count differs")
            for row in rows:
                if len(row) != 3 or any(type(v) is not int or v <= 0 for v in row):
                    raise ValueError("exact processor image grid invalid")
                patches = row[0] * row[1] * row[2]
                if patches % (merge * merge):
                    raise ValueError("exact processor image grid merge invalid")
                image_tokens.append(patches // (merge * merge))
        return int(encoded["input_ids"].shape[-1])

    def count_request(self, request: GenerationRequest) -> int:
        cache_key = sha256(request.to_json().encode("utf-8")).hexdigest()
        if cache_key not in self._request_cache:
            self._request_cache[cache_key] = self._count_parts(request.messages)
        return self._request_cache[cache_key]

    def count_image_ref(self, image: ImageRef) -> int:
        key = self._image_cache_key(image)
        if key not in self._image_cache:
            text_part = ModelContentPart(kind=ModelContentKind.TEXT, text="image")
            image_part = ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
            without = [ModelMessage(role=SpeakerRole.USER, parts=[text_part])]
            with_image = [
                ModelMessage(role=SpeakerRole.USER, parts=[text_part, image_part])
            ]
            self._image_cache[key] = max(
                0,
                self._count_parts(with_image) - self._count_parts(without),
            )
        return self._image_cache[key]

    def count_image(self, *, width: int | None = None, height: int | None = None) -> int:
        del width, height
        raise RuntimeError("exact multimodal counting requires the canonical ImageRef")


class QwenVLCachedExactTokenCounter(ProcessorTokenCounter):
    """Qwen VL counter that caches exact image grids and request token counts.

    The fast path mirrors the Qwen processor's image-token expansion. Paper runs
    compare initial requests and every accepted unit against the full processor;
    a mismatch is fatal instead of falling back to an estimate.
    """

    backend_version = "qwen-vl-cached-exact-v1"

    def __init__(
        self,
        *,
        persistent_cache_dir: Path,
        strict_conformance: bool,
        conformance_samples: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.persistent_cache_dir = persistent_cache_dir.resolve()
        self.persistent_cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.persistent_cache_dir, 0o700)
        self.strict_conformance = strict_conformance
        self.conformance_samples_remaining = conformance_samples
        self.processor_calls = 0
        self.request_cache_hits = 0
        self.image_cache_hits = 0
        self.accepted_audit_count = 0
        image_processor = getattr(self.processor, "image_processor", None)
        self._image_token = getattr(self.processor, "image_token", None)
        self._merge_size = int(getattr(image_processor, "merge_size", 0) or 0)
        self.fast_supported = bool(
            image_processor is not None
            and self._image_token
            and self._merge_size > 0
            and "qwen" in type(self.processor).__qualname__.lower()
        )

    @property
    def audit(self) -> dict[str, int | str | bool]:
        return {
            "backend": self.backend_version,
            "fast_supported": self.fast_supported,
            "processor_calls": self.processor_calls,
            "request_cache_hits": self.request_cache_hits,
            "image_cache_hits": self.image_cache_hits,
        }

    @contextmanager
    def _cache_lock(self, path: Path) -> Iterator[None]:
        import fcntl

        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as stream:
            os.chmod(lock_path, 0o600)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_cache_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(path)

    def _request_cache_path(self, key: str) -> Path:
        return self.persistent_cache_dir / "requests" / key[:2] / f"{key}.json"

    def _grid_cache_path(self, key: str) -> Path:
        return self.persistent_cache_dir / "image-grids" / key[:2] / f"{key}.json"

    def _persistent_key(self, kind: str, value: object) -> str:
        payload = json.dumps(
            {
                "kind": kind,
                "revision": self._revision,
                "processor_fingerprint": self._processor_fingerprint,
                "value": value,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    def _image_grid(self, image: ImageRef) -> tuple[int, int, int]:
        source_key = self._image_cache_key(image)
        key = self._persistent_key("image-grid", source_key)
        path = self._grid_cache_path(key)
        with self._cache_lock(path):
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                grid = raw.get("grid_thw")
                if (
                    raw.get("cache_key") == key
                    and isinstance(grid, list)
                    and len(grid) == 3
                    and all(isinstance(item, int) and item > 0 for item in grid)
                ):
                    self.image_cache_hits += 1
                    return int(grid[0]), int(grid[1]), int(grid[2])
            opened = self._open_image(image)
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is None:
                raise RuntimeError("Qwen exact counter has no image processor")
            encoded = image_processor(
                images=[opened], return_tensors="pt", **self.processor_kwargs,
            )
            grid_value = encoded.get("image_grid_thw")
            if grid_value is None:
                raise RuntimeError("Qwen image processor did not return image_grid_thw")
            values = grid_value[0].tolist()
            grid = (int(values[0]), int(values[1]), int(values[2]))
            if any(item <= 0 for item in grid):
                raise RuntimeError("Qwen image processor returned an invalid image grid")
            self.processor_calls += 1
            self._atomic_cache_json(
                path, {"cache_key": key, "grid_thw": list(grid)}
            )
            return grid

    def _render_parts(
        self, messages: list[ModelMessage]
    ) -> tuple[str, list[ImageRef]]:
        rendered_messages: list[dict[str, object]] = []
        images: list[ImageRef] = []
        for message in messages:
            content: list[dict[str, object]] = []
            for part in message.parts:
                if part.kind is ModelContentKind.TEXT:
                    content.append({"type": "text", "text": part.text or ""})
                else:
                    assert part.image is not None
                    images.append(part.image)
                    content.append({"type": "image"})
            rendered_messages.append({"role": message.role.value, "content": content})
        rendered = self.processor.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        return str(rendered), images

    def _fast_count_parts(self, messages: list[ModelMessage]) -> int:
        rendered, images = self._render_parts(messages)
        if not images:
            return len(self.tokenizer.encode(rendered, add_special_tokens=False))
        if not self.fast_supported:
            return super()._count_parts(messages)
        expanded = rendered
        placeholder = "<|pmmc_image_placeholder|>"
        for image in images:
            grid = self._image_grid(image)
            sequence_length = math.prod(grid) // (self._merge_size**2)
            expanded = expanded.replace(
                str(self._image_token), placeholder * sequence_length, 1
            )
        expanded = expanded.replace(placeholder, str(self._image_token))
        return len(self.tokenizer.encode(expanded, add_special_tokens=False))

    def _reference_count(self, request: GenerationRequest) -> int:
        self.processor_calls += 1
        return super()._count_parts(request.messages)

    def _assert_conformant(self, request: GenerationRequest, actual: int) -> None:
        expected = self._reference_count(request)
        if actual != expected:
            if self.strict_conformance:
                raise RuntimeError(
                    "Qwen cached exact counter disagrees with the reference processor"
                )
            self.fast_supported = False
            if actual != expected:
                self._request_cache[
                    sha256(request.to_json().encode("utf-8")).hexdigest()
                ] = expected

    def count_request(self, request: GenerationRequest) -> int:
        request_hash = sha256(request.to_json().encode("utf-8")).hexdigest()
        key = self._persistent_key("request", request_hash)
        if request_hash in self._request_cache:
            self.request_cache_hits += 1
            return self._request_cache[request_hash]
        path = self._request_cache_path(key)
        with self._cache_lock(path):
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("cache_key") == key and isinstance(raw.get("tokens"), int):
                    value = int(raw["tokens"])
                    self._request_cache[request_hash] = value
                    self.request_cache_hits += 1
                    return value
            value = self._fast_count_parts(request.messages)
            if self.conformance_samples_remaining > 0:
                self._assert_conformant(request, value)
                value = self._request_cache.get(request_hash, value)
                self.conformance_samples_remaining -= 1
            self._request_cache[request_hash] = value
            self._atomic_cache_json(path, {"cache_key": key, "tokens": value})
            return value

    def audit_request(self, request: GenerationRequest, expected_tokens: int) -> None:
        """Reference-check a deterministic sample of accepted units."""

        request_hash = sha256(request.to_json().encode("utf-8")).hexdigest()
        should_audit = (
            self.accepted_audit_count == 0 or int(request_hash[:8], 16) % 32 == 0
        )
        self.accepted_audit_count += 1
        if not should_audit:
            return
        actual = self._reference_count(request)
        if actual != expected_tokens:
            raise RuntimeError(
                "accepted Questioner request failed exact token-counter conformance"
            )
