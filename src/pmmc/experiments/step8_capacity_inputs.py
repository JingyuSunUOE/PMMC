"""Materialize observed native layouts without editing frozen development inputs."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

from pmmc.baselines.config import prompt_sha256
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import durable_write, json_bytes, sha256_file
from pmmc.experiments.step8_input_reconstruction import verify_reconstructed_inputs
from pmmc.infrastructure.settings import Settings
from pmmc.models.locking import load_model_lock, verify_snapshot_with_policy
from pmmc.models.registry import load_model_registry
from pmmc.pilots.partition import load_partition
from pmmc.runtime.config import load_runtime_profile
from pmmc.runtime.grounding import load_step8_prompt, step8_prompt_scope
from pmmc.runtime.step8_capacity import capacity_image_workload
from pmmc.runtime.step8_capacity_native import (
    NativeCapacityImageWorkload,
    NativeImageLayout,
    layout_covers,
    verify_native_processor,
)
from pmmc.schemas import ModelVerificationPolicy
from pmmc.schemas.step8 import Step8BenchmarkConfig


def context_identity(settings: Settings) -> dict:
    from pmmc.experiments.step8_context_audit import context_source_fingerprint

    with step8_prompt_scope(settings.project_root):
        prompt = load_step8_prompt("pmmc_answer_contract_v3.txt")
    return {
        "context_source_fingerprint": context_source_fingerprint(),
        "full_context_config_sha256": sha256_file(settings.resolve_path(
            Path("configs/baselines/full_context.yaml"))),
        "prompt_sha256": prompt_sha256(prompt),
        "libraries": {name: version(name) for name in (
            "transformers", "tokenizers", "torch", "Pillow",
        )},
    }


def maximal_layouts(values: list[dict]) -> list[NativeImageLayout]:
    """A measured larger grid vector covers smaller ones; never combine maxima."""
    distinct = {layout.identity: layout for value in values
                if (layout := NativeImageLayout.model_validate(value))}
    kept: list[NativeImageLayout] = []
    for key in sorted(distinct):
        layout = distinct[key]
        tokens = [image.image_tokens for image in layout.images]
        if any(layout_covers(other, tokens) for other in kept):
            continue
        kept = [other for other in kept if not layout_covers(
            layout, [image.image_tokens for image in other.images],
        )]
        kept.append(layout)
    return sorted(kept, key=lambda layout: layout.identity)


def _verify_audited_inputs(settings: Settings, config: Step8BenchmarkConfig, audit: dict) -> None:
    """Recheck gold-free source bytes, not merely the unchanged config paths."""
    from pmmc.experiments.step8_context_audit import RUNTIME_FILES

    try:
        regression = settings.resolve_path(config.regression_manifest)
        if (regression.is_symlink()
                or (regression.parent / "input-reconstruction.json").is_symlink()):
            raise ValueError("reconstruction_symlink")
        reconstruction = verify_reconstructed_inputs(
            config=config, project_root=settings.project_root,
        )
        if (reconstruction is None
                or canonical_content_hash(reconstruction) != audit.get("reconstruction_hash")):
            raise ValueError("reconstruction_identity")
        partition, _ = load_partition(settings.resolve_path(config.partition_dir))
        expected = audit.get("runtime_file_sha256")
        if (partition.content_hash != audit.get("partition_hash")
                or audit.get("runtime_file_hashes_stable") is not True
                or not isinstance(expected, dict)
                or set(expected) != {item.dataset.value for item in partition.datasets}):
            raise ValueError("runtime_coverage")
        for dataset in partition.datasets:
            hashes = expected[dataset.dataset.value]
            if not isinstance(hashes, dict) or set(hashes) != set(RUNTIME_FILES):
                raise ValueError("runtime_file_coverage")
            root = (settings.benchmark_artifacts_path / dataset.dataset.value).resolve()
            snapshot = root / dataset.snapshot_id
            if (snapshot.name != dataset.snapshot_id or snapshot.is_symlink()
                    or snapshot.resolve().parent != root):
                raise ValueError("snapshot_path")
            # Iterate a source-owned allowlist, never file names supplied by a report.
            # No canonical preflight/hash here: those also read query_gold.jsonl.
            for name in RUNTIME_FILES:
                path = snapshot / name
                if path.is_symlink() or sha256_file(path) != hashes[name]:
                    raise ValueError("runtime_file_changed")
            frozen = reconstruction["inputs"]["snapshots"][dataset.dataset.value]
            if (hashes["manifest.json"] != frozen["manifest_sha256"]
                    or dataset.snapshot_id != frozen["snapshot_id"]
                    or dataset.snapshot_content_hash != frozen["canonical_content_hash"]):
                raise ValueError("snapshot_identity")
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("capacity_input_audit_sources_changed") from None


def prepare_native_capacity_inputs(
    settings: Settings, config: Step8BenchmarkConfig, *, audit_path: Path, output_dir: Path,
) -> dict:
    """Development workload proposal, NOT a passing receipt or scientific approval."""
    audit = json.loads(audit_path.read_bytes())
    content_hash = audit.get("content_hash")
    if content_hash != canonical_content_hash({
        k: v for k, v in audit.items() if k != "content_hash"
    }):
        raise ValueError("capacity_input_audit_checksum_mismatch")
    if any(audit.get(key) != value for key, value in context_identity(settings).items()):
        raise ValueError("capacity_input_audit_context_changed")
    profile = load_runtime_profile(settings.resolve_path(config.runtime_profile))
    if (audit.get("protocol") != "step8-development-context-audit-v1"
            or audit.get("state") != "inspected" or audit.get("selection") != "regression"
            or audit.get("config_hash") != canonical_content_hash(config.model_dump(mode="json"))
            or audit.get("source_profile_hash") != profile.resolved_hash
            or audit.get("model_lock_sha256") != sha256_file(
                settings.resolve_path(config.model_lock))
            or audit.get("registry_hash") != load_model_registry(
                settings.resolve_path(config.model_registry)).resolved_hash
            or audit.get("gold_read") is not False or audit.get("provider_calls") != 0
            or audit.get("gpu_operations") != 0 or audit.get("images_modified") is not False):
        raise ValueError("capacity_input_audit_identity_mismatch")
    _verify_audited_inputs(settings, config, audit)
    gated = [spec for spec in profile.services if spec.metadata.get("requires_capacity_receipt")]
    if len(gated) != 1:
        raise ValueError("capacity_input_requires_single_pool")
    source = gated[0]
    model = audit["models"][source.model_alias]
    layouts = maximal_layouts(model["layouts"])
    if not layouts:
        raise ValueError("capacity_input_has_no_native_layouts")
    raw = model["observed_raw_input_bounds"]
    bounds = NativeCapacityImageWorkload(
        protocol="native-image-layouts-v1", images_per_sequence=raw["images_per_sequence"],
        image_width=raw["image_width"], image_height=raw["image_height"],
        processor_config_sha256=model["processor_config_sha256"],
        context_audit_hash=content_hash, layouts=layouts,
    )
    entry = load_model_lock(settings.resolve_path(config.model_lock)).get(source.model_alias)
    if (model.get("commit") != entry.commit_sha
            or model.get("snapshot_content_hash") != entry.snapshot_content_hash
            or model.get("processor_kwargs") != {}
            or model.get("chat_template_kwargs") != {"enable_thinking": False}):
        raise ValueError("capacity_input_processor_identity_mismatch")
    snapshot = (settings.model_cache_path / ("models--" + entry.repo_id.replace("/", "--"))
                / "snapshots" / entry.commit_sha)
    verify_snapshot_with_policy(entry, snapshot, settings.model_cache_path,
                                policy=ModelVerificationPolicy.CACHED_RECEIPT,
                                allow_full_fallback=False)
    verify_native_processor(snapshot, bounds)
    flags = ("--limit-mm-per-prompt", "--mm-processor-kwargs",
             "--enable-prefix-caching", "--no-enable-prefix-caching")
    if any(arg.split("=", 1)[0] in flags for arg in source.extra_args):
        raise ValueError("capacity_input_refuses_existing_processor_override")
    updated = source.model_copy(update={
        "extra_args": [*source.extra_args, "--limit-mm-per-prompt",
                       json.dumps({"image": bounds.images_per_sequence}),
                       "--mm-processor-kwargs", "{}", "--no-enable-prefix-caching"],
        "metadata": {**source.metadata, "capacity_image_workload": bounds.model_dump(),
                     "capacity_receipt_filename": "step8-qwen35-27b-native-capacity.json"},
    })
    capacity_image_workload(updated, multimodal=True)
    profile = profile.model_copy(update={"services": [
        updated if spec.service_id == source.service_id else spec for spec in profile.services
    ]})
    directory = output_dir.resolve()
    if not directory.is_relative_to(settings.project_root.resolve()):
        raise ValueError("capacity_input_output_outside_project")
    profile_path = directory / "runtime-profile.yaml"
    updated_config = config.model_copy(update={
        "runtime_profile": profile_path.relative_to(settings.project_root.resolve()),
    })
    values = {
        "runtime-profile.yaml": profile.model_dump(mode="json"),
        "benchmark-development.yaml": updated_config.model_dump(mode="json"),
        "context-audit.json": audit,
    }
    receipt = {
        "protocol": "step8-native-capacity-inputs-v1", "state": "prepared_not_calibrated",
        "audit_content_hash": content_hash, "profile_hash": profile.resolved_hash,
        "source_profile_hash": audit["source_profile_hash"],
        "layout_count_before": len(model["layouts"]), "layout_count_after": len(layouts),
        "all_methods_covered": False, "paper_test_allowed": False,
        "images_modified": False, "processor_overrides": {},
        "prefix_caching": False,
        "gpu_benchmark_ready": False,
    }
    # Precheck all outputs before immutable writes; interrupted reruns are idempotent.
    for name, value in {**values, "preparation.json": receipt}.items():
        path = directory / name
        if path.exists() and path.read_bytes() != json_bytes(value):
            raise ValueError("capacity_input_refuses_overwrite")
    _verify_audited_inputs(settings, config, audit)
    for name, value in {**values, "preparation.json": receipt}.items():
        durable_write(directory / name, json_bytes(value), immutable=True)
    return receipt
