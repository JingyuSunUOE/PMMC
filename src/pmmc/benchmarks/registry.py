"""Benchmark adapter registry."""

from __future__ import annotations

from pathlib import Path

from pmmc.benchmarks.adapters import MemGalleryAdapter, MemLensAdapter
from pmmc.benchmarks.base import BenchmarkAdapter
from pmmc.schemas import DatasetName

ADAPTER_TYPES: dict[DatasetName, type[BenchmarkAdapter]] = {
    DatasetName.MEMLENS: MemLensAdapter,
    DatasetName.MEMGALLERY: MemGalleryAdapter,
}


def create_adapter(
    dataset: DatasetName,
    *,
    root: Path,
    project_root: Path,
) -> BenchmarkAdapter:
    """Instantiate a built-in source adapter."""

    if dataset is DatasetName.UNKNOWN:
        raise ValueError("unknown is not a supported benchmark adapter")
    return ADAPTER_TYPES[dataset](root=root, project_root=project_root)
