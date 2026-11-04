"""Ephemeral provider URL resolution for private canonical images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from pmmc.infrastructure.object_store import ObjectStore, validate_object_key
from pmmc.schemas import ImageRef


class ImageUrlResolver(Protocol):
    async def resolve(self, image: ImageRef) -> str: ...


def validate_paper_image_reference(
    image: ImageRef,
    *,
    runtime_bucket: str,
) -> None:
    """Reject a prompt image that is not reproducible and privately synchronized."""

    parsed = urlparse(image.storage_uri or "")
    if parsed.scheme != "s3" or parsed.netloc != runtime_bucket or not parsed.path.lstrip("/"):
        raise ValueError("paper mode requires a private runtime-bucket s3:// image URI")
    if (
        image.local_path is None
        or image.sha256 is None
        or image.width is None
        or image.height is None
    ):
        raise ValueError(
            "paper mode images require local path, SHA256, dimensions, and runtime storage URI"
        )


@dataclass(frozen=True)
class StaticImageUrlResolver:
    """Resolve an already provider-readable URL for tests and public assets."""

    async def resolve(self, image: ImageRef) -> str:
        value = image.storage_uri or image.remote_url
        if not value:
            raise ValueError(f"image {image.image_id!r} has no provider-readable URL")
        if value.lower().startswith("data:"):
            raise ValueError("inline base64/data image payloads are not allowed")
        return value


class ObjectStoreImageUrlResolver:
    """Resolve private S3 URIs without persisting the temporary signed URL."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        bucket: str,
        expires_seconds: int = 3600,
        paper_mode: bool = False,
    ) -> None:
        if expires_seconds <= 0:
            raise ValueError("presigned URL expiry must be positive")
        self.store = store
        self.bucket = bucket
        self.expires_seconds = expires_seconds
        self.paper_mode = paper_mode

    async def resolve(self, image: ImageRef) -> str:
        if image.storage_uri:
            parsed = urlparse(image.storage_uri)
            if parsed.scheme == "s3":
                if parsed.netloc != self.bucket:
                    raise ValueError("image storage URI points to the wrong runtime bucket")
                if self.paper_mode and (
                    image.sha256 is None or image.width is None or image.height is None
                ):
                    raise ValueError(
                        "paper mode images require SHA256, width, height, and runtime storage URI"
                    )
                return await self.store.presign_get(
                    validate_object_key(parsed.path.lstrip("/")),
                    expires_seconds=self.expires_seconds,
                )
            if self.paper_mode:
                raise ValueError("paper mode requires a private runtime-bucket s3:// image URI")
            if parsed.scheme in {"http", "https", "file"}:
                return image.storage_uri
            raise ValueError("unsupported image storage URI scheme")
        if not self.paper_mode and image.remote_url:
            if image.remote_url.lower().startswith("data:"):
                raise ValueError("inline base64/data image payloads are not allowed")
            return image.remote_url
        raise ValueError(
            f"image {image.image_id!r} is not synchronized to the runtime object store"
        )
