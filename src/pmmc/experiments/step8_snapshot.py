"""Validate raw data fully without replacing the partition's frozen snapshot."""

from __future__ import annotations

import errno
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from pmmc.benchmarks import create_adapter
from pmmc.benchmarks.base import BenchmarkAdapter
from pmmc.benchmarks.canonical import iter_jsonl, preflight_snapshot
from pmmc.benchmarks.snapshot import CANONICAL_FILES, build_snapshot
from pmmc.experiments.step8_artifacts import sha256_file, write_json
from pmmc.infrastructure.settings import Settings
from pmmc.pilots.partition import verify_partition
from pmmc.schemas import AssetRef, DatasetName, DatasetSnapshotManifest, ValidationLevel
from pmmc.schemas.step8 import BenchmarkCampaignManifest, Step8BenchmarkConfig

_FILES = (*CANONICAL_FILES, "manifest.json", "validation_report.json")


class SnapshotPreparationError(ValueError):
    def __init__(self, code: str) -> None:
        self.error_code = code
        super().__init__(code)


def snapshot_event(phase: str, state: str, **fields: Any) -> None:
    print("STEP8_SNAPSHOT " + json.dumps(
        {"phase": phase, "state": state, **fields}, sort_keys=True,
    ), flush=True)


def _file_hashes(path: Path) -> dict[str, str]:
    if path.is_symlink() or any((path / name).is_symlink() for name in _FILES):
        raise SnapshotPreparationError("snapshot_symlink_forbidden")
    return {name: sha256_file(path / name) for name in _FILES}


def prepare_frozen_snapshot(
    adapter: BenchmarkAdapter,
    *,
    frozen_path: Path,
    expected_content_hash: str,
    output_root: Path,
    scratch_dir: Path | None = None,
    paper_result: bool = False,
    enforce_expected_counts: bool = True,
) -> tuple[Path, DatasetSnapshotManifest, dict[str, Any]]:
    """Check both identities, retain exact frozen bytes, and seal image checksums.

    FAST snapshots lack historical image checksums. They are accepted only for
    development, with a separate full-validation receipt, never relabelled FULL.
    """
    snapshot_event("frozen_integrity", "start", dataset=adapter.dataset.value)
    frozen_files = _file_hashes(frozen_path)
    checked = preflight_snapshot(frozen_path)
    frozen, level = checked.manifest, checked.validation_report.level
    if (frozen.dataset != adapter.dataset
            or frozen.canonical_content_hash != expected_content_hash
            or frozen.metadata.get("validation_level") != level.value):
        raise SnapshotPreparationError("snapshot_frozen_identity_mismatch")
    if paper_result and level is not ValidationLevel.FULL:
        raise SnapshotPreparationError("snapshot_paper_requires_frozen_full_validation")
    snapshot_event("frozen_integrity", "done", snapshot_id=frozen.snapshot_id,
                   identity_level=level.value, raw_validation_level="full")
    last = time.monotonic()

    def progress(phase: str, completed: int, total: int | None) -> None:
        nonlocal last
        if time.monotonic() - last >= 30:
            snapshot_event("raw_validation", "running", subphase=phase,
                           completed=completed, total=total)
            last = time.monotonic()

    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step8-snapshot-", dir=scratch_dir) as temporary:
        cold_root = Path(temporary)
        snapshot_event("raw_full", "start", image_mime_policy="fixed-supported-extensions-v1")
        full_path, full = build_snapshot(
            adapter, output_root=cold_root, level=ValidationLevel.FULL,
            enforce_expected_counts=enforce_expected_counts, progress=progress,
        )
        snapshot_event("raw_full", "done", snapshot_id=full.snapshot_id)
        projected = full
        if level is ValidationLevel.FAST:
            snapshot_event("identity_projection", "start")
            _, projected = build_snapshot(
                adapter, output_root=cold_root, level=level,
                enforce_expected_counts=enforce_expected_counts, progress=progress,
            )
        # Legacy IDs are checked by preflight_snapshot. Compare their underlying
        # raw/canonical identities too, not the newer policy-derived ID alone.
        fields = ("source_content_hash", "canonical_content_hash", "schema_version",
                  "adapter_name", "adapter_version", "dataset_version", "counts",
                  "evaluation_policy_version", "category_counts", "eligible_category_counts",
                  "excluded_category_counts")
        legacy = "excluded_source_categories" not in json.loads(
            (frozen_path / "manifest.json").read_text()
        )
        mismatches = [key for key in fields if getattr(projected, key) != getattr(frozen, key)]
        if not legacy and projected.snapshot_id != frozen.snapshot_id:
            mismatches.append("snapshot_id")
        if mismatches:
            snapshot_event("identity_projection", "failed", changed_fields=mismatches,
                           frozen_canonical_hash=frozen.canonical_content_hash,
                           rebuilt_canonical_hash=projected.canonical_content_hash)
            raise SnapshotPreparationError("snapshot_raw_differs_from_frozen_identity")
        snapshot_event("identity_projection", "done", frozen_snapshot_id=frozen.snapshot_id,
                       rebuilt_snapshot_id=projected.snapshot_id)
        checksums = {asset.asset_id: asset.sha256
                     for asset in iter_jsonl(full_path / "assets.jsonl", AssetRef)}
        if len(checksums) != full.counts["assets"] or any(
            key is None or value is None for key, value in checksums.items()
        ):
            raise SnapshotPreparationError("snapshot_full_asset_checksums_incomplete")
        proof = {
            "protocol": "step8-frozen-snapshot-validation-v1",
            "snapshot_id": frozen.snapshot_id,
            "identity_validation_level": level.value,
            "raw_validation_level": "full",
            "historical_image_checksums_available": level is ValidationLevel.FULL,
            "full_snapshot_id": full.snapshot_id,
            "full_source_content_hash": full.source_content_hash,
            "full_canonical_content_hash": full.canonical_content_hash,
            "frozen_files": frozen_files,
            "asset_sha256": checksums,
        }
        if _file_hashes(frozen_path) != frozen_files:
            raise SnapshotPreparationError("snapshot_frozen_files_changed_during_validation")
        proof_path = output_root / ".validation" / (frozen.snapshot_id + ".json")
        if proof_path.exists() and json.loads(proof_path.read_text()) != proof:
            raise SnapshotPreparationError("snapshot_full_validation_changed")
        write_json(proof_path, proof, immutable=True)
        target = output_root / adapter.dataset.value / frozen.snapshot_id
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_event("publish", "start")
        with tempfile.TemporaryDirectory(prefix=".snapshot-publish-", dir=target.parent) as tmp:
            staged = Path(tmp) / frozen.snapshot_id
            staged.mkdir()
            for name in _FILES:
                shutil.copyfile(frozen_path / name, staged / name)
            if _file_hashes(staged) != frozen_files:
                raise SnapshotPreparationError("snapshot_frozen_files_changed_during_copy")
            try:
                staged.rename(target)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
            if _file_hashes(target) != frozen_files:
                raise SnapshotPreparationError("snapshot_published_identity_conflict")
        snapshot_event("publish", "done", snapshot_id=frozen.snapshot_id,
                       raw_images_verified=len(checksums), paper_result=paper_result)
        return target, frozen, proof


def prepare_campaign_snapshot(
    settings: Settings, config: Step8BenchmarkConfig, campaign: BenchmarkCampaignManifest,
    dataset: DatasetName, snapshot_id: str,
) -> tuple[Path, DatasetSnapshotManifest, dict[str, Any]]:
    if Path(snapshot_id).name != snapshot_id or "\\" in snapshot_id or snapshot_id in {".", ".."}:
        raise SnapshotPreparationError("snapshot_identifier_invalid")
    partition = verify_partition(settings.resolve_path(config.partition_dir))
    matches = [item for item in partition.datasets
               if item.dataset == dataset and item.snapshot_id == snapshot_id]
    if (partition.content_hash != campaign.protocol.partition_hash or len(matches) != 1
            or campaign.snapshot_ids.get(dataset) != snapshot_id):
        raise SnapshotPreparationError("snapshot_partition_identity_mismatch")
    source = settings.project_root / "artifacts/benchmarks" / dataset.value / snapshot_id
    adapter = create_adapter(
        dataset, root=settings.memlens_path if dataset is DatasetName.MEMLENS
        else settings.memgallery_path, project_root=settings.benchmark_asset_path,
    )
    return prepare_frozen_snapshot(
        adapter, frozen_path=source, expected_content_hash=matches[0].snapshot_content_hash,
        output_root=settings.benchmark_artifacts_path,
        scratch_dir=settings.local_scratch_dir, paper_result=campaign.paper_result,
    )
