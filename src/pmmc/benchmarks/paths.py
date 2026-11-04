"""Portable and traversal-safe benchmark asset path handling."""

from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath

from PIL import Image

from pmmc.benchmarks.ids import path_id, sha256_file
from pmmc.schemas import AssetRef, AssetType, DatasetName, ValidationLevel

# Canonical identities must not depend on the host's optional MIME database.
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
IMAGE_SUFFIXES = set(IMAGE_MIME_TYPES)


def resolve_within(root: Path, relative_path: str) -> Path:
    """Resolve a source path and reject traversal outside ``root``."""

    normalized = relative_path.replace("\\", "/")
    source_path = PurePosixPath(normalized)
    if source_path.is_absolute():
        raise ValueError(f"absolute benchmark asset path is not allowed: {relative_path}")
    candidate = (root / Path(*source_path.parts)).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"benchmark asset path escapes dataset root: {relative_path}")
    return candidate


def project_relative_path(path: Path, project_root: Path) -> str:
    """Return a portable project-relative path, or an absolute mount path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def iter_image_files(images_root: Path) -> list[Path]:
    """Return deterministic non-hidden image paths."""

    return sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(images_root).parts)
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_asset_ref(
    *,
    path: Path,
    dataset_root: Path,
    project_root: Path,
    dataset: DatasetName,
    level: ValidationLevel,
) -> AssetRef:
    """Inspect one local image and return portable metadata."""

    if not path.is_file():
        raise ValueError(f"image file does not exist: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"image file is empty: {path}")

    dataset_relative = path.resolve().relative_to(dataset_root.resolve()).as_posix()
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    if level is ValidationLevel.FULL:
        sha256 = sha256_file(path)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()

    mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(path.name)
    return AssetRef(
        asset_id=path_id(dataset, "asset", dataset_relative),
        asset_type=AssetType.RAW_IMAGE,
        local_path=project_relative_path(path, project_root),
        sha256=sha256,
        mime_type=mime_type or "application/octet-stream",
        size_bytes=size_bytes,
        width=width,
        height=height,
        metadata={
            "dataset": dataset.value,
            "dataset_relative_path": dataset_relative,
            "validation_level": level.value,
        },
    )
