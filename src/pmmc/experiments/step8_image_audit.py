"""Read-only development image inventory, not capacity or scientific approval."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from pmmc.benchmarks.repositories import MemoryCorpus, RuntimeQueryStore
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_input_reconstruction import verify_reconstructed_inputs
from pmmc.experiments.step8_selection import ExposureManifest, select_benchmark_queries
from pmmc.infrastructure.settings import Settings
from pmmc.models.locking import (
    assert_paper_lock,
    load_model_lock,
    verify_snapshot_with_policy,
)
from pmmc.models.registry import load_model_registry
from pmmc.pilots.partition import load_partition, verify_partition
from pmmc.runtime.config import load_runtime_profile
from pmmc.runtime.step8_capacity import capacity_image_workload
from pmmc.schemas import DatasetName, ImageRef, ModelVerificationPolicy
from pmmc.schemas.step8 import CampaignPhase, Step8BenchmarkConfig


def inspect_image(root: Path, ref: ImageRef) -> dict[str, Any]:
    """Inspect actual bytes once; errors contain no source paths or image labels."""
    if ref.local_path is None:
        raise ValueError("image_audit_local_file_required")
    path = (root / ref.local_path).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("image_audit_file_unavailable")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if ref.sha256 is not None and ref.sha256.lower() != digest:
        raise ValueError("image_audit_checksum_mismatch")
    try:
        with Image.open(BytesIO(payload)) as decoded:
            decoded.verify()
        with Image.open(BytesIO(payload)) as decoded:
            # JPEG verify() checks headers, not whether the compressed pixels decode.
            decoded.load()
            width, height = decoded.size
            if ((ref.width is not None and ref.width != width)
                    or (ref.height is not None and ref.height != height)):
                raise ValueError("image_audit_dimensions_mismatch")
            media_type = Image.MIME.get(decoded.format or "")
            if ref.mime_type is not None and ref.mime_type != media_type:
                raise ValueError("image_audit_media_type_mismatch")
            orientation = decoded.getexif().get(274, 1)
            result = {
                "sha256": digest, "width": width, "height": height,
                "format": decoded.format, "frames": getattr(decoded, "n_frames", 1),
                "orientation": (orientation if type(orientation) is int and 0 <= orientation <= 8
                                else "invalid"),
                "declared_checksum": ref.sha256 is not None,
            }
    except (OSError, Image.DecompressionBombError):
        raise ValueError("image_audit_decode_failed") from None
    return result


def _dataset_images(
    settings: Settings, snapshot: Path, query_ids: list[str],
    conversations: dict[str, str],
) -> dict[str, Any]:
    selected = set(query_ids)
    queries = [r.query for r in RuntimeQueryStore(snapshot).iter_records()
               if r.query.query_id in selected]
    if len(queries) != len(selected) or {q.query_id for q in queries} != selected:
        raise ValueError("image_audit_query_coverage_mismatch")
    if any(conversations.get(q.query_id) != q.conversation_id for q in queries):
        raise ValueError("image_audit_conversation_mismatch")
    corpus = MemoryCorpus(snapshot)
    # The key includes declared metadata so conflicting aliases cannot hide a bad hash.
    images: dict[str, dict[str, Any]] = {}
    counts: list[int] = []
    query_counts: list[int] = []
    scope_hashes: list[str] = []
    for query in queries:
        if query.available_after_session is None:
            raise ValueError("image_audit_cutoff_required")
        memories = list(corpus.iter_visible(query.conversation_id, query.available_after_session))
        refs = [*query.query_image_refs, *(r for m in memories for r in m.image_refs)]
        # This is a pre-token-budget upper bound, not a proposed image cap.
        counts.append(len(refs))
        query_counts.append(len(query.query_image_refs))
        scope_hashes.append(canonical_content_hash([
            query.query_id, query.conversation_id, query.available_after_session,
            [m.memory_id for m in memories],
        ]))
        for ref in refs:
            key = canonical_content_hash(ref.model_dump(mode="json"))
            if key not in images:
                images[key] = inspect_image(settings.benchmark_asset_path, ref)
    unique = {row["sha256"]: row for row in images.values()}
    rows = list(unique.values())
    return {
        "query_count": len(queries),
        "scope_hash": canonical_content_hash(sorted(scope_hashes)),
        "unique_image_count": len(rows),
        "input_identity_hash": canonical_content_hash(sorted(
            images.values(), key=lambda row: canonical_content_hash(row),
        )),
        "maximum_visible_image_occurrences_before_token_budget": max(counts, default=0),
        "maximum_query_image_count": max(query_counts, default=0),
        "maximum_width": max((r["width"] for r in rows), default=0),
        "maximum_height": max((r["height"] for r in rows), default=0),
        "maximum_pixels": max((r["width"] * r["height"] for r in rows), default=0),
        "format_counts": dict(sorted(Counter(r["format"] for r in rows).items())),
        "orientation_counts": dict(sorted(Counter(str(r["orientation"]) for r in rows).items())),
        "animated_image_count": sum(r["frames"] != 1 for r in rows),
        "missing_declared_checksum_count": sum(not r["declared_checksum"] for r in images.values()),
        "capacity_format_or_orientation_rejection_count": sum(
            r["format"] not in {"PNG", "JPEG", "WEBP", "GIF"}
            or r["orientation"] not in {0, 1} or r["frames"] != 1 for r in rows
        ),
        "closed_provider_budget_format_rejection_count": sum(
            r["format"] not in {"PNG", "JPEG", "WEBP", "GIF"} or r["frames"] != 1 for r in rows
        ),
        "images_above_capacity_v2_dimension_ceiling": sum(
            r["width"] > 4096 or r["height"] > 4096 for r in rows
        ),
    }


def inspect_locked_processors(settings: Settings, config: Step8BenchmarkConfig) -> dict[str, Any]:
    """Reuse verified file receipts; never download, execute model code or hash weights."""
    aliases = ("answer-qwen3.5-9b", "answer-qwen3.5-27b")
    try:
        lock = load_model_lock(settings.resolve_path(config.model_lock))
        assert_paper_lock(lock, load_model_registry(settings.resolve_path(config.model_registry)))
    except (OSError, ValueError):
        return {"state": "unavailable", "code": "model_lock_unavailable_or_invalid"}
    result = {}
    for alias in aliases:
        try:
            entry = lock.get(alias)
            snapshot = (settings.model_cache_path / ("models--" + entry.repo_id.replace("/", "--"))
                        / "snapshots" / entry.commit_sha)
            verification = verify_snapshot_with_policy(
                entry, snapshot, settings.model_cache_path,
                policy=ModelVerificationPolicy.CACHED_RECEIPT, allow_full_fallback=False,
            )
            path = snapshot / "preprocessor_config.json"
            if not any(f.path == path.name for f in entry.files) or path.stat().st_size > 1_048_576:
                raise ValueError("processor_config_unavailable")
            payload = path.read_bytes()
            raw = json.loads(payload)
            result[alias] = {
                "state": "verified", "commit": entry.commit_sha,
                "snapshot_content_hash": entry.snapshot_content_hash,
                "processor_config_sha256": hashlib.sha256(payload).hexdigest(),
                "weight_bytes_hashed": verification.bytes_hashed,
                "native_image_processor": {k: raw[k] for k in (
                    "image_processor_type", "processor_class", "size", "min_pixels",
                    "max_pixels", "patch_size", "merge_size", "temporal_patch_size",
                    "do_resize", "resample",
                ) if k in raw},
            }
        except (OSError, ValueError, KeyError):
            result[alias] = {"state": "unavailable", "code": "cached_processor_identity_unverified"}
    return {"state": "verified" if all(r["state"] == "verified" for r in result.values())
            else "incomplete", "models": result}


def inspect_step8_images(
    settings: Settings, config: Step8BenchmarkConfig, *, selection: str = "regression",
) -> dict[str, Any]:
    """Use the real frozen development selector without creating a campaign.

    Do not call preflight_snapshot: its full evaluator audit reads query_gold.
    This command never approves missing history, modifies images, or tunes a cap.
    """
    if selection not in {"diagnostic", "regression"}:
        raise ValueError("image_audit_development_selection_required")
    verify_reconstructed_inputs(config=config, project_root=settings.project_root)
    partition_dir = settings.resolve_path(config.partition_dir)
    verify_partition(partition_dir)
    partition, assignments = load_partition(partition_dir)
    exposure = ExposureManifest.model_validate_json(
        settings.resolve_path(config.exposure_manifest).read_text(),
    )
    regression = json.loads(settings.resolve_path(config.regression_manifest).read_text())
    chosen = select_benchmark_queries(
        campaign_id="image-audit", phase=CampaignPhase.DEVELOPMENT, selection=selection,
        partition_hash=partition.content_hash, assignments=assignments, exposure=exposure,
        regression_ids={DatasetName(k): v for k, v in regression["query_ids"].items()},
        seed=config.seed,
    )
    datasets = {}
    for dataset in partition.datasets:
        print(f"STEP8_IMAGE_AUDIT dataset={dataset.dataset.value} state=start", flush=True)
        snapshot = settings.benchmark_artifacts_path / dataset.dataset.value / dataset.snapshot_id
        manifest = json.loads((snapshot / "manifest.json").read_text())
        if (manifest["snapshot_id"] != dataset.snapshot_id
                or manifest["canonical_content_hash"] != dataset.snapshot_content_hash):
            raise ValueError("image_audit_snapshot_identity_mismatch")
        datasets[dataset.dataset.value] = _dataset_images(
            settings, snapshot, chosen.query_ids[dataset.dataset],
            {r.query_id: r.conversation_id for r in assignments if r.dataset is dataset.dataset},
        )
        summary = datasets[dataset.dataset.value]
        print(f"STEP8_IMAGE_AUDIT dataset={dataset.dataset.value} state=inspected "
              f"queries={summary['query_count']} images={summary['unique_image_count']}",
              flush=True)
    profile = load_runtime_profile(settings.resolve_path(config.runtime_profile))
    contracts = []
    for service in profile.services:
        if service.metadata.get("requires_capacity_receipt") is not True:
            continue
        try:
            bounds = capacity_image_workload(service, multimodal=True)
            contracts.append({"service_id": service.service_id, "configured": True,
                              "image_workload": bounds.model_dump() if bounds else None})
        except ValueError:
            contracts.append({"service_id": service.service_id, "configured": False,
                              "error_code": "capacity_image_configuration_required"})
    print("STEP8_IMAGE_AUDIT phase=locked_processors state=start full_weight_hashing=disabled",
          flush=True)
    processors = inspect_locked_processors(settings, config)
    print(f"STEP8_IMAGE_AUDIT phase=locked_processors state={processors['state']}", flush=True)
    return {
        "protocol": "step8-development-image-audit-v1", "state": "inspected",
        "scope": "visible_images_before_context_budget_not_rendered_requests",
        "selection": selection, "selection_hash": chosen.content_hash,
        "partition_hash": partition.content_hash, "profile_hash": profile.resolved_hash,
        "datasets": datasets, "capacity_contracts": contracts,
        "locked_processors": processors,
        "gold_read": False, "provider_calls": 0, "gpu_operations": 0,
        "snapshot_file_hashes_verified": False,
        "images_modified": False, "scientific_approval": False,
        "capacity_verified": False, "gpu_benchmark_ready": False,
        "next_gate": "exact_processor_workload_and_real_capacity_measurement",
    }
