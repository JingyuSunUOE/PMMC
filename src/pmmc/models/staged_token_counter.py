"""Stage checksum-pinned native images for counting, never for request rewriting."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

from pmmc.models.protocols import TokenCounter
from pmmc.models.provider_images import ImageBytesResolver, verified_image_dimensions
from pmmc.schemas import GenerationRequest, ImageRef


class StagedImageTokenCounter:
    def __init__(self, counter: TokenCounter, resolver: ImageBytesResolver, root: Path):
        if counter.estimated:
            raise ValueError("staged counting requires an exact processor")
        self.counter, self.resolver, self.root = counter, resolver, root.resolve()
        self._images: dict[str, tuple[Path, int, int]] = {}
        self._lock = asyncio.Lock()

    @property
    def alias(self):
        return self.counter.alias

    @property
    def revision(self):
        return self.counter.revision

    @property
    def estimated(self):
        return False

    async def prepare_images(self, images: Sequence[ImageRef]) -> None:
        # Shared across candidate workers; avoid duplicate downloads and partial reads.
        async with self._lock:
            for image in images:
                digest = image.sha256 or ""
                if not re.fullmatch(r"[a-f0-9]{64}", digest):
                    raise ValueError("staged image requires immutable SHA256")
                if digest in self._images:
                    continue
                payload = await self.resolver.resolve(image)
                width, height = verified_image_dimensions(image, payload)
                self.root.mkdir(parents=True, exist_ok=True)
                path = self.root / digest
                path.write_bytes(payload.data)
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("staged image checksum mismatch")
                self._images[digest] = (path, width, height)

    def count_text(self, text: str) -> int:
        return self.counter.count_text(text)

    def count_image(self, *, width=None, height=None) -> int:
        return self.counter.count_image(width=width, height=height)

    def count_request(self, request: GenerationRequest) -> int:
        messages = []
        for message in request.messages:
            parts = []
            for part in message.parts:
                image = part.image
                if image is not None:
                    path, width, height = self._images[image.sha256 or ""]
                    image = image.model_copy(update={
                        "local_path": str(path),
                        "width": image.width if image.width is not None else width,
                        "height": image.height if image.height is not None else height,
                    })
                    part = part.model_copy(update={"image": image})
                parts.append(part)
            messages.append(message.model_copy(update={"parts": parts}))
        # Only the counter sees ephemeral paths. Journal/provider manifests stay native.
        return self.counter.count_request(request.model_copy(update={"messages": messages}))


def create_staged_token_counter(settings, registry, alias, store):
    from pmmc.models.factory import create_token_counter
    from pmmc.models.provider_images import ObjectStoreImageBytesResolver

    root = settings.token_counter_cache_path / "native-images"
    counter = create_token_counter(
        settings.model_copy(update={"benchmark_asset_root": root}),
        registry, alias, allow_estimated=False,
    )
    if counter.estimated:
        # Explicit FAKE endpoints remain available to offline fixtures only.
        return counter
    return StagedImageTokenCounter(counter, ObjectStoreImageBytesResolver(
        store, bucket=settings.object_store_bucket, allow_verified_mime_correction=True,
    ), root)
