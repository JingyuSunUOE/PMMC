"""Private images become ephemeral inline bytes, never private provider URLs."""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from PIL import Image

from pmmc.infrastructure.object_store import ObjectStore, validate_object_key
from pmmc.schemas import ImageRef


@dataclass(frozen=True)
class ImageBytes:
    data: bytes
    media_type: str

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.base64}"


class ImageBytesResolver(Protocol):
    async def resolve(self, image: ImageRef) -> ImageBytes: ...


def verified_image_dimensions(image: ImageRef, payload: ImageBytes) -> tuple[int, int]:
    """Read native geometry without modifying immutable references or image bytes."""
    if not image.sha256 or len(image.sha256) != 64:
        raise ValueError("image dimension recovery requires immutable SHA256")
    validate_image_bytes(image, payload, allow_verified_mime_correction=True)
    with Image.open(BytesIO(payload.data)) as decoded:
        return decoded.size


class LocalImageDimensions:
    """Resolve missing metadata using the same pinned local assets as exact counters."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._cached_read = lru_cache(maxsize=256)(self._read)

    @staticmethod
    def _read(path: Path, image_json: str, stamp: tuple[int, ...]) -> tuple[int, int]:
        del stamp
        image = ImageRef.model_validate_json(image_json)
        data = path.read_bytes()
        with Image.open(BytesIO(data)) as decoded:
            media_type = Image.MIME.get(decoded.format or "", "")
        return verified_image_dimensions(image, ImageBytes(data, media_type))

    def __call__(self, image: ImageRef) -> tuple[int, int]:
        if not image.local_path:
            raise ValueError("image dimension recovery requires a pinned local image")
        path = (self.root / image.local_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("image dimension recovery asset is unavailable")
        stat = path.stat()
        return self._cached_read(path, image.model_dump_json(), (
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def _validate_static_gif(data: bytes) -> None:
    # Pillow tolerates missing GIF block terminators/trailers and verify() is a
    # no-op. Check framing here; leave raster decoding to Pillow below.
    stream = BytesIO(data)

    def read(size: int) -> bytes:
        value = stream.read(size)
        if len(value) != size:
            raise ValueError("malformed native GIF image")
        return value

    def subblocks() -> None:
        while size := read(1)[0]:
            read(size)

    header = read(13)
    width, height, flags = struct.unpack("<HHB", header[6:11])
    if header[:6] not in {b"GIF87a", b"GIF89a"} or not width or not height:
        raise ValueError("malformed native GIF image")
    global_palette = bool(flags & 0x80)
    if global_palette:
        read(3 * (2 ** ((flags & 7) + 1)))
    seen_frame = False
    while True:
        block = read(1)
        if block == b";":
            if not seen_frame or stream.read(1):
                raise ValueError("malformed native GIF image")
            return
        if block == b",":
            if seen_frame:
                raise ValueError("animated native GIF images are unsupported; require a static GIF")
            left, top, frame_width, frame_height, flags = struct.unpack("<HHHHB", read(9))
            if (
                not frame_width
                or not frame_height
                or flags & 0x18
                or left + frame_width > width
                or top + frame_height > height
            ):
                raise ValueError("malformed native GIF image")
            if flags & 0x80:
                read(3 * (2 ** ((flags & 7) + 1)))
            elif not global_palette:
                raise ValueError("native GIF image requires a color table")
            if not 2 <= read(1)[0] <= 8:
                raise ValueError("malformed native GIF image")
            subblocks()
            seen_frame = True
        elif block == b"!":
            label = read(1)
            if label == b"\xf9":
                if read(1) != b"\x04":
                    raise ValueError("malformed native GIF image")
                control = read(4)
                if control[0] & 0xE0 or ((control[0] >> 2) & 7) > 3 or read(1) != b"\x00":
                    raise ValueError("malformed native GIF image")
            elif label == b"\xff":
                if read(1) != b"\x0b":
                    raise ValueError("malformed native GIF image")
                read(11)
                subblocks()
            elif label == b"\xfe":
                subblocks()
            else:
                # Plain-text rendering and unknown extensions are outside the
                # single-raster policy; ignoring them could discard content.
                raise ValueError("unsupported native GIF extension")
        else:
            raise ValueError("malformed native GIF image")


def validate_image_bytes(
    image: ImageRef, payload: ImageBytes, *, allow_verified_mime_correction: bool = False,
) -> ImageBytes:
    if not isinstance(payload, ImageBytes) or not payload.data:
        raise ValueError("native images require nonempty bytes")
    if image.sha256 and hashlib.sha256(payload.data).hexdigest() != image.sha256.lower():
        raise ValueError("native image checksum mismatch")
    if payload.data.startswith(b"GIF"):
        _validate_static_gif(payload.data)
    with Image.open(BytesIO(payload.data)) as decoded:
        media_type = Image.MIME.get(decoded.format or "")
        if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise ValueError("unsupported native image media type")
        declared_mismatch = bool(image.mime_type and image.mime_type != media_type)
        # Old immutable snapshots may label by extension. Correct only the transport
        # header of checksum-pinned bytes, never the snapshot, pixels or resolution.
        correction_allowed = (
            allow_verified_mime_correction and bool(image.sha256)
            and image.mime_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}
        )
        if media_type != payload.media_type or (declared_mismatch and not correction_allowed):
            raise ValueError("native image media type mismatch")
        if (image.width is not None and image.width != decoded.width) or (
            image.height is not None and image.height != decoded.height
        ):
            raise ValueError("native image dimensions mismatch")
        if decoded.format == "GIF":
            decoded.load()
        else:
            decoded.verify()
    return payload


class ObjectStoreImageBytesResolver:
    def __init__(
        self, store: ObjectStore, *, bucket: str, allow_verified_mime_correction: bool = False,
    ) -> None:
        self.store = store
        self.bucket = bucket
        self.allow_verified_mime_correction = allow_verified_mime_correction

    async def resolve(self, image: ImageRef) -> ImageBytes:
        parsed = urlparse(image.storage_uri or "")
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError("native image must belong to the runtime object-store bucket")
        data = await self.store.get_bytes(validate_object_key(parsed.path.lstrip("/")))
        with Image.open(BytesIO(data)) as decoded:
            media_type = Image.MIME.get(decoded.format or "", "")
        return validate_image_bytes(
            image, ImageBytes(data, media_type),
            allow_verified_mime_correction=self.allow_verified_mime_correction,
        )
