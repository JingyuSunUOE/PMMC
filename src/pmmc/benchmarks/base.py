"""Adapter interface for source-specific benchmark packages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from pmmc.benchmarks.models import CanonicalConversation
from pmmc.schemas import AssetRef, DatasetName, ValidationLevel


class BenchmarkDataError(ValueError):
    """Raised when raw benchmark data violates an ingestion invariant."""


class BenchmarkAdapter(ABC):
    """Source adapter that produces deterministic canonical records."""

    dataset: DatasetName
    adapter_name: str
    adapter_version: str
    dataset_version: str

    def __init__(self, root: Path, project_root: Path) -> None:
        self.root = root.resolve()
        self.project_root = project_root.resolve()

    @property
    @abstractmethod
    def source_record_count(self) -> int:
        """Number of top-level source records or scenarios."""

    @abstractmethod
    def source_files(self) -> list[Path]:
        """Return deterministic source data and prompt files."""

    @abstractmethod
    def iter_conversations(self) -> Iterator[CanonicalConversation]:
        """Yield canonical conversations without retaining the full corpus."""

    def iter_assets(self, level: ValidationLevel) -> Iterator[AssetRef]:
        """Yield every local image asset, including currently unreferenced files."""

        from pmmc.benchmarks.paths import build_asset_ref, iter_image_files

        for path in iter_image_files(self.root / "images"):
            yield build_asset_ref(
                path=path,
                dataset_root=self.root,
                project_root=self.project_root,
                dataset=self.dataset,
                level=level,
            )

    @abstractmethod
    def expected_counts(self) -> dict[str, int]:
        """Return regression counts for the retained local package."""
