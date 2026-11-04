"""Read-only, CPU-only inventory of actual Step8 full-context requests.

``inspect_step8_contexts`` is the CLI integration boundary. It returns safe JSON
but writes nothing. Catch Step8ContextAuditError and print only its report.
The full native processor runs even for budget-search candidates: CPU RAM may
be substantial. No image cap, resize override, estimate, or capacity approval is
introduced here. Native processor resizing in RAM is not a source-image edit.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

from PIL import Image

from pmmc.baselines.config import FullContextBaselineConfig, load_full_context_config, prompt_sha256
from pmmc.baselines.context import ContextBudgeter
from pmmc.benchmarks.repositories import MemoryCorpus, RuntimeQueryStore, SessionStore
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import sha256_file
from pmmc.experiments.step8_image_audit import inspect_image
from pmmc.experiments.step8_input_reconstruction import verify_reconstructed_inputs
from pmmc.experiments.step8_jobs import _python_source_hash
from pmmc.experiments.step8_selection import ExposureManifest, select_benchmark_queries
from pmmc.infrastructure.settings import Settings
from pmmc.models.locking import assert_paper_lock, load_model_lock, verify_snapshot_with_policy
from pmmc.models.native_qwen_processor import verify_loaded_qwen_image_processor
from pmmc.models.provider_images import ImageBytes
from pmmc.models.provider_token_budget import ProviderConfiguredTokenCounter
from pmmc.models.providers import ProviderGenerationGateway
from pmmc.models.registry import load_model_registry
from pmmc.models.tokenization import ProcessorTokenCounter
from pmmc.pilots.partition import load_partition, verify_partition
from pmmc.runtime.answering import answer_format_retry_request
from pmmc.runtime.config import load_runtime_profile
from pmmc.runtime.grounding import load_step8_prompt, step8_prompt_scope
from pmmc.schemas import BaselineMethod, DatasetName, ImageRef, ModelVerificationPolicy
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION
from pmmc.schemas.step8 import CampaignPhase, Step8BenchmarkConfig

ALIASES = ("answer-qwen3.5-9b", "answer-qwen3.5-27b")
METHODS = (BaselineMethod.FULL_CONTEXT_CAPTIONS, BaselineMethod.FULL_CONTEXT_RAW_IMAGES)
RUNTIME_FILES = ("manifest.json", "queries.jsonl", "sessions.jsonl", "memory_units.jsonl")
CONTEXT_SOURCE_OWNERS = (
    "experiments/step8_context_audit.py",
    "experiments/step8_image_audit.py::inspect_image",
    "experiments/step8_input_reconstruction.py::verify_reconstructed_inputs",
    "experiments/step8_selection.py",
    "experiments/step8_jobs.py::_python_source_hash",
    "baselines/context.py", "baselines/rendering.py", "baselines/config.py",
    "baselines/typed_answer.py::adapt_baseline_typed_request",
    "benchmarks/repositories.py", "pilots/partition.py::verify_partition",
    "models/tokenization.py::ProcessorTokenCounter", "models/provider_config.py",
    "models/native_qwen_processor.py",
    "models/provider_token_budget.py", "models/provider_images.py",
    "models/providers.py::ProviderGenerationGateway._messages",
    "models/locking.py", "models/registry.py",
    "runtime/answering.py::prepare_typed_answer_request",
    "runtime/answering.py::answer_format_retry_request",
    "runtime/grounding.py::load_step8_prompt", "runtime/grounding.py::source_content_parts",
    "runtime/config.py", "compiler/ids.py", "schemas/base.py", "schemas/model.py",
    "schemas/memory.py", "schemas/query.py", "schemas/baseline.py", "schemas/answer_output.py",
)


def context_source_fingerprint() -> str:
    """Loaded package owners, using Step8's existing scientific AST hash rules.

    Materializers should recompute this before accepting a context report. This
    deliberately avoids unrelated inference, capacity, and CLI implementation.
    """
    package = Path(__file__).resolve().parents[1]
    hashes = {}
    for owner in CONTEXT_SOURCE_OWNERS:
        relative, _, selector = owner.partition("::")
        hashes[owner] = _python_source_hash((package / relative).read_text(), selector)
    return canonical_content_hash({"protocol": "step8-context-source-v1-ast", "owners": hashes})


class _AuditProgress:
    interval_seconds = 30.0

    def __init__(self):
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.phase = "selection"
        self.subphase = "selection"
        self.completed = self.total = 0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True,
                                       name="step8-context-audit-heartbeat")

    def update(self, *, phase=None, subphase=None, completed=None, total=None):
        with self.lock:
            if phase is not None:
                self.phase = phase
                self.subphase = phase
            if subphase is not None:
                self.subphase = subphase
            if completed is not None:
                self.completed = completed
            if total is not None:
                self.total = total

    def emit(self, state):
        with self.lock:
            phase, completed, total = self.phase, self.completed, self.total
            subphase = self.subphase
        print(f"STEP8_CONTEXT_AUDIT phase={phase} completed={completed} total={total} "
              f"state={state} elapsed_seconds={int(time.monotonic() - self.started)} "
              f"subphase={subphase}", flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval_seconds):
            self.emit("running")

    def start(self):
        self.emit("start")
        self.thread.start()

    def close(self, state):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.emit(state)


def _safe_error_diagnostic(error: Exception) -> dict[str, Any]:
    # Never serialize exception messages/args, frame locals, paths, or source lines.
    category = next((kind.__name__ for kind in (
        MemoryError, PermissionError, FileNotFoundError, TimeoutError, OSError,
        ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError,
    ) if isinstance(error, kind)), "Exception")
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        module = frame.f_globals.get("__name__", "")
        if module == __name__:
            frames.append({"function": frame.f_code.co_name, "line": tb.tb_lineno})
        tb = tb.tb_next
    errno = error.errno if isinstance(error, OSError) else None
    return {"exception_type": category, "audit_frames": frames[-8:],
            "errno": errno if type(errno) is int else None}


class Step8ContextAuditError(RuntimeError):
    """Underlying exceptions can contain private prompt text and source paths."""

    def __init__(self, phase: str, *, error: Exception | None = None,
                 progress: _AuditProgress | None = None):
        super().__init__("step8_context_audit_failed")
        self.report: dict[str, Any] = {
            "protocol": "step8-development-context-audit-v1", "state": "failed",
            "code": str(self), "phase": phase, "estimated_fallback": False,
            "gold_read": False, "provider_calls": 0, "gpu_operations": 0,
            "images_modified": False, "capacity_verified": False,
            "gpu_benchmark_ready": False, "scientific_approval": False,
        }
        if error is not None:
            self.report["diagnostic"] = _safe_error_diagnostic(error)
        if progress is not None:
            self.report.update(subphase=progress.subphase,
                               completed=progress.completed, total=progress.total)


def _step8_config(settings: Settings, alias: str) -> FullContextBaselineConfig:
    # Mirror Step8Backend._answer_settings, without writing its derived config.
    path = settings.resolve_path(Path("configs/baselines/full_context.yaml"))
    if settings.resolved_full_context_config_path.resolve() != path.resolve():
        raise ValueError("unfrozen_full_context_override")
    raw = load_full_context_config(path).model_dump(mode="python")
    raw.update(
        answer_model_alias=alias, answer_contract_version=TYPED_ANSWER_CONTRACT_VERSION,
        prompt_version="pmmc-answer-contract-v3",
        prompt_path=Path("prompts/runtime/pmmc_answer_contract_v3.txt"),
        reserved_output_tokens=512, max_output_tokens=512,
        max_input_tokens=32256, total_context_tokens=32768,
    )
    return FullContextBaselineConfig.model_validate(raw)


def _has_auto_map(value: Any) -> bool:
    if isinstance(value, dict):
        return "auto_map" in value or any(_has_auto_map(v) for v in value.values())
    return isinstance(value, list) and any(_has_auto_map(v) for v in value)


def _verify_processor(settings, entry, pinned) -> tuple[Path, dict[str, str]]:
    snapshot = (settings.model_cache_path / ("models--" + entry.repo_id.replace("/", "--"))
                / "snapshots" / entry.commit_sha)
    checked = verify_snapshot_with_policy(
        entry, snapshot, settings.model_cache_path,
        policy=ModelVerificationPolicy.CACHED_RECEIPT, allow_full_fallback=False,
    )
    if not checked.receipt_cache_hit or checked.bytes_hashed != 0:
        raise ValueError("cached_processor_receipt_required")
    if (pinned["state"] != "verified" or pinned["commit"] != entry.commit_sha
            or pinned["snapshot_content_hash"] != entry.snapshot_content_hash):
        raise ValueError("pinned_processor_identity_mismatch")
    locked = {f.path for f in entry.files}
    hashes = {}
    # The existing counter enables remote code. Reject all custom-code hooks
    # before calling it, and use a verified local snapshot, never a Hub name.
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        if path.suffix == ".py":
            raise ValueError("custom_processor_code_forbidden")
        if path.suffix not in {".json", ".jinja", ".txt", ".model", ".tiktoken"}:
            continue
        if relative not in locked:
            raise ValueError("unlocked_processor_file")
        hashes[relative] = sha256_file(path)
        if path.name.endswith("config.json"):
            if path.stat().st_size > 1_048_576 or _has_auto_map(json.loads(path.read_bytes())):
                raise ValueError("custom_processor_configuration_forbidden")
    if hashes.get("preprocessor_config.json") != pinned["processor_config_sha256"]:
        raise ValueError("pinned_processor_config_mismatch")
    raw = json.loads((snapshot / "preprocessor_config.json").read_bytes())
    if (raw.get("processor_class") != "Qwen3VLProcessor"
            or raw.get("image_processor_type") != "Qwen2VLImageProcessorFast"
            or {k: raw[k] for k in pinned["native_image_processor"] if k in raw}
            != pinned["native_image_processor"]):
        raise ValueError("native_qwen_processor_required")
    return snapshot, hashes


class _CpuProcessor:
    """Device-only adapter; size/patch/merge settings stay native."""

    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, **kwargs):
        encoded = self.processor(**kwargs, device="cpu")
        if any(getattr(getattr(v, "device", None), "type", "cpu") != "cpu"
               for v in encoded.values()):
            raise ValueError("non_cpu_processor_output")
        return encoded


class _ProcessorProbe:
    """Request-local observation only; never leave a last-grid attribute on the counter."""

    def __init__(self, processor):
        self.processor = processor
        self.rendered = ""
        self.encoded: dict[str, Any] = {}

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def apply_chat_template(self, *args, **kwargs):
        rendered = self.processor.apply_chat_template(*args, **kwargs)
        self.rendered = str(rendered)
        return rendered

    def __call__(self, **kwargs):
        encoded = self.processor(**kwargs)
        self.encoded = {
            "input_ids_hash": canonical_content_hash(encoded["input_ids"].tolist()),
            "grids": encoded["image_grid_thw"].tolist(),
            "pixel_values_numel": encoded["pixel_values"].numel(),
        }
        return encoded


class _LocalImageBytesResolver:
    def __init__(self, root: Path):
        self.root = root.resolve()

    async def resolve(self, image: ImageRef) -> ImageBytes:
        if image.local_path is None or image.sha256 is None or image.mime_type is None:
            raise ValueError("hydrated_local_image_required")
        path = (self.root / image.local_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("local_image_escapes_root")
        return ImageBytes(path.read_bytes(), image.mime_type)


class _LocalWireRenderer:
    """Reuse provider serialization with local original bytes and no SDK client."""

    kind = "qwen_vllm"
    # This method needs only kind/resolver; do not construct a generation client.
    render: Any = ProviderGenerationGateway._messages

    def __init__(self, root: Path):
        self.image_bytes_resolver = _LocalImageBytesResolver(root)


def _create_counter(settings, entry, snapshot):
    # Deliberately bypass create_token_counter: it may select an estimate, a
    # provider, persistent caches, or a capacity-derived resizing override.
    counter = ProcessorTokenCounter(
        alias=entry.alias, tokenizer_name=str(snapshot), tokenizer_revision=snapshot.name,
        processor_name=str(snapshot), processor_revision=snapshot.name,
        project_root=settings.benchmark_asset_path, local_files_only=True,
        require_image_metadata=True, cache_dir=settings.model_cache_path,
        local_asset_cache_dir=None, processor_kwargs={},
        chat_template_kwargs={"enable_thinking": False},
    )
    if (type(counter.processor).__name__ != "Qwen3VLProcessor"
            or type(counter.processor).__module__
            != "transformers.models.qwen3_vl.processing_qwen3_vl"):
        raise ValueError("unexpected_native_processor")
    native = json.loads((snapshot / "preprocessor_config.json").read_bytes())
    identity = verify_loaded_qwen_image_processor(counter.processor.image_processor, native)
    print(
        f"STEP8_CONTEXT_PROCESSOR alias={entry.alias} state=verified "
        f"configured_class={identity['configured_class']} loaded_class={identity['loaded_class']} "
        f"backend={identity['backend']} native_settings_verified=true device=cpu",
        flush=True,
    )
    counter.processor = _CpuProcessor(counter.processor)
    return counter


def _clear_counter_caches(counter: ProcessorTokenCounter) -> None:
    for decoded in counter._decoded_image_cache.values():
        decoded.close()
    counter._decoded_image_cache.clear()
    counter._image_cache.clear()
    counter._request_cache.clear()


def _capture_request(counter: ProcessorTokenCounter, request) -> dict[str, Any]:
    if (type(counter) is not ProcessorTokenCounter or counter.estimated
            or not isinstance(counter.processor, _CpuProcessor)
            or counter.processor_kwargs != {}
            or counter.chat_template_kwargs != {"enable_thinking": False}):
        raise ValueError("exact_native_counter_required")
    original = counter.processor
    probe = _ProcessorProbe(original)
    counter.processor = probe
    try:
        wire = asyncio.run(_LocalWireRenderer(counter.project_root).render(request))
        count, image_tokens = counter.count_transport_profile(wire)
        refs = [part.image for m in request.messages for part in m.parts if part.image is not None]
        grids = probe.encoded.get("grids", [])
        merge = probe.image_processor.merge_size
        patch = probe.image_processor.patch_size
        if len(grids) != len(refs) or len(image_tokens) != len(refs) or any(
            len(g) != 3 or any(type(n) is not int or n <= 0 for n in g)
            or math.prod(g) % (merge * merge) for g in grids
        ):
            raise ValueError("processor_grid_coverage_mismatch")
        ids_hash = probe.encoded.get("input_ids_hash")
        if not refs:
            ids = counter.tokenizer.encode(probe.rendered, add_special_tokens=False)
            if len(ids) != count:
                raise ValueError("text_tokenization_drift")
            ids_hash = canonical_content_hash([ids])
        patches = sum(math.prod(g) for g in grids)
        if image_tokens != [math.prod(g) // (merge * merge) for g in grids]:
            raise ValueError("transport_profile_grid_mismatch")
        layouts = [
            {"width": ref.width, "height": ref.height, "sha256": ref.sha256,
             "grid_thw": grid, "image_tokens": tokens}
            for ref, grid, tokens in zip(refs, grids, image_tokens, strict=True)
        ]
        return {
            "prompt_tokens": count,
            "transport_messages_hash": canonical_content_hash(wire),
            "rendered_prompt_sha256": hashlib.sha256(probe.rendered.encode()).hexdigest(),
            "input_ids_hash": ids_hash,
            "image_grid_hash": canonical_content_hash(grids),
            "image_grid_histogram": [
                {"grid_thw": list(grid), "occurrences": n}
                for grid, n in sorted(Counter(tuple(g) for g in grids).items())
            ],
            "vision_tokens": sum(image_tokens),
            "image_layout": layouts,
            "image_token_vector_desc": sorted(image_tokens, reverse=True),
            "processed_spatial_pixels": patches * patch * patch,
            "pixel_values_numel": probe.encoded.get("pixel_values_numel", 0),
        }
    finally:
        counter.processor = original


def _hydrate_image(root: Path, ref: ImageRef, files: dict[Path, str]) -> ImageRef:
    if ref.local_path is None:
        raise ValueError("local_image_required")
    stats = inspect_image(root, ref)
    if stats["frames"] != 1 or stats["orientation"] not in {0, 1}:
        raise ValueError("unsupported_image_frames_or_orientation")
    if stats["format"] not in {"PNG", "JPEG", "WEBP", "GIF"}:
        raise ValueError("unsupported_image_format")
    path = (root / ref.local_path).resolve()
    digest = stats["sha256"]
    if files.setdefault(path, digest) != digest:
        raise ValueError("image_changed_during_audit")
    # Original references and manifests remain untouched; never repair a mismatch.
    return ref.model_copy(update={
        "sha256": digest, "width": stats["width"], "height": stats["height"],
        "mime_type": Image.MIME[stats["format"]],
    })


def _image_stats(request, files: dict[Path, str], root: Path) -> dict[str, Any]:
    refs = [p.image for m in request.messages for p in m.parts if p.image is not None]
    pixels = [r.width * r.height for r in refs]
    byte_sizes = [(root / r.local_path).stat().st_size for r in refs]
    if any(files[(root / r.local_path).resolve()] != r.sha256 for r in refs):
        raise ValueError("image_identity_drift")
    return {
        "image_occurrences": len(refs), "unique_images": len({r.sha256 for r in refs}),
        "original_pixels": sum(pixels), "original_bytes": sum(byte_sizes),
        "maximum_image_pixels": max(pixels, default=0),
        "maximum_width": max((r.width for r in refs), default=0),
        "maximum_height": max((r.height for r in refs), default=0),
        "image_sequence_hash": canonical_content_hash([
            [r.sha256, r.width, r.height, r.mime_type] for r in refs
        ]),
        "original_geometry_histogram": [
            {"width": w, "height": h, "occurrences": n}
            for (w, h), n in sorted(Counter((r.width, r.height) for r in refs).items())
        ],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Each witness is a real joint workload, never max(count) * max(dimensions).
    metrics = ("prompt_tokens", "image_occurrences", "original_pixels", "original_bytes",
               "processed_spatial_pixels", "vision_tokens", "pixel_values_numel")
    return {
        "rendered_request_count": len(rows),
        "truncated_request_count": sum(r["truncated_memory_count"] > 0 for r in rows),
        "metrics": {
            key: {
                "minimum": min((r[key] for r in rows), default=0),
                "maximum": max((r[key] for r in rows), default=0),
                "sum": sum(r[key] for r in rows),
                "maximum_witness_hashes": sorted(
                    r["request_hash"] for r in rows
                    if r[key] == max(v[key] for v in rows)
                ),
            } for key in metrics
        },
    }


def _model_layouts(rows: list[dict[str, Any]], processor: dict[str, Any]) -> dict[str, Any]:
    layouts: dict[str, dict[str, Any]] = {}
    witnesses: dict[str, list[str]] = {}
    for row in rows:
        if not row["image_layout"]:
            continue
        layout = {"images": [{key: image[key] for key in ("width", "height", "image_tokens")}
                             for image in row["image_layout"]]}
        key = canonical_content_hash(layout)
        layouts[key] = layout
        witnesses.setdefault(key, []).append(row["request_hash"])
    return {
        **processor, "layouts": [layouts[key] for key in sorted(layouts)],
        "layout_witness_hashes": {key: sorted(witnesses[key]) for key in sorted(witnesses)},
        "observed_raw_input_bounds": {
            "images_per_sequence": max((r["image_occurrences"] for r in rows), default=0),
            "image_width": max((r["maximum_width"] for r in rows), default=0),
            "image_height": max((r["maximum_height"] for r in rows), default=0),
        },
        "maximum_prompt_tokens": max((r["prompt_tokens"] for r in rows), default=0),
        "text_only_request_count": sum(not r["image_layout"] for r in rows),
        "calibrated": False,
    }


def inspect_step8_contexts(
    settings: Settings, config: Step8BenchmarkConfig, *, pinned_processor_audit: Path,
    selection: str = "regression",
) -> dict[str, Any]:
    """Audit both Qwen backbones/methods using frozen development inputs only.

    Requires an existing reconstructed bundle, local lock, valid cached model
    receipts, and the pinned image-audit receipt. No preflight_snapshot (gold),
    model factory/client, service, GPU discovery, or filesystem write is used.
    Snapshot runtime hashes are observed and checked for stability, not falsely
    claimed to prove the gold-inclusive canonical snapshot digest.
    """
    phase = "selection"
    counter = None
    progress = _AuditProgress()
    state = "failed"
    try:
        progress.start()
        if selection not in {"regression", "diagnostic"}:
            raise ValueError("development_selection_required")
        reconstruction = verify_reconstructed_inputs(
            config=config, project_root=settings.project_root,
        )
        if reconstruction is None:
            raise ValueError("reconstructed_development_inputs_required")
        frozen_config = settings.resolve_path(config.regression_manifest).parent / (
            "benchmark-development.yaml"
        )
        if Step8BenchmarkConfig.model_validate_json(frozen_config.read_bytes()) != config:
            raise ValueError("reconstructed_config_changed")
        partition_path = settings.resolve_path(config.partition_dir)
        verify_partition(partition_path)
        partition, assignments = load_partition(partition_path)
        exposure = ExposureManifest.model_validate_json(
            settings.resolve_path(config.exposure_manifest).read_bytes(),
        )
        regression = json.loads(settings.resolve_path(config.regression_manifest).read_bytes())
        chosen = select_benchmark_queries(
            campaign_id="context-audit", phase=CampaignPhase.DEVELOPMENT, selection=selection,
            partition_hash=partition.content_hash, assignments=assignments, exposure=exposure,
            regression_ids={DatasetName(k): v for k, v in regression["query_ids"].items()},
            seed=config.seed,
        )
        phase = "locked_processors"
        progress.update(phase=phase)
        pinned_path = settings.resolve_path(pinned_processor_audit)
        pinned = json.loads(pinned_path.read_bytes())
        if (pinned["protocol"] != "step8-development-image-audit-v1"
                or pinned["locked_processors"]["state"] != "verified"
                or pinned["partition_hash"] != partition.content_hash):
            raise ValueError("pinned_image_audit_mismatch")
        registry_path = settings.resolve_path(config.model_registry)
        lock_path = settings.resolve_path(config.model_lock)
        registry = load_model_registry(registry_path)
        lock = load_model_lock(lock_path)
        assert_paper_lock(lock, registry, aliases=ALIASES)
        profile_path = settings.resolve_path(config.runtime_profile)
        profile = load_runtime_profile(profile_path)
        prompt_path = settings.project_root / "prompts/runtime/pmmc_answer_contract_v3.txt"
        files = {p: sha256_file(p) for p in (pinned_path, registry_path, lock_path,
                 profile_path, prompt_path,
                 settings.resolve_path(Path("configs/baselines/full_context.yaml")))}
        verified = {}
        for alias in ALIASES:
            entry = registry.get(alias)
            if (entry.provider_config is None or entry.provider_config.kind != "qwen_vllm"
                    or entry.provider_config.reasoning_mode != "off"
                    or entry.tokenizer_name != lock.get(alias).repo_id
                    or entry.processor_name != lock.get(alias).repo_id
                    or entry.context_window is None or entry.context_window < 32768):
                raise ValueError("qwen_registry_contract_mismatch")
            verified[alias] = _verify_processor(
                settings, lock.get(alias), pinned["locked_processors"]["models"][alias],
            )
        phase = "runtime_inputs"
        progress.update(phase=phase)
        scopes = []
        runtime_hashes = {}
        for dataset in partition.datasets:
            snapshot = (settings.benchmark_artifacts_path
                        / dataset.dataset.value / dataset.snapshot_id)
            hashes = {name: sha256_file(snapshot / name) for name in RUNTIME_FILES}
            files.update({snapshot / name: digest for name, digest in hashes.items()})
            runtime_hashes[dataset.dataset.value] = hashes
            manifest = json.loads((snapshot / "manifest.json").read_bytes())
            original = reconstruction["inputs"]["snapshots"][dataset.dataset.value]
            if (manifest["snapshot_id"] != dataset.snapshot_id
                    or manifest["dataset"] != dataset.dataset.value
                    or manifest["canonical_content_hash"] != dataset.snapshot_content_hash
                    or hashes["manifest.json"] != original["manifest_sha256"]):
                raise ValueError("snapshot_identity_mismatch")
            ids = chosen.query_ids[dataset.dataset]
            queries = [r.query for r in RuntimeQueryStore(snapshot).iter_records()
                       if r.query.query_id in set(ids)]
            if len(queries) != len(ids) or {q.query_id for q in queries} != set(ids):
                raise ValueError("query_coverage_mismatch")
            conversations = {r.query_id: r.conversation_id for r in assignments
                             if r.dataset is dataset.dataset}
            for query in sorted(queries, key=lambda q: q.query_id):
                if (query.available_after_session is None
                        or conversations[query.query_id] != query.conversation_id):
                    raise ValueError("query_scope_mismatch")
                memories = list(MemoryCorpus(snapshot).iter_visible(
                    query.conversation_id, query.available_after_session,
                ))
                sessions = list(SessionStore(snapshot).iter_visible(
                    query.conversation_id, query.available_after_session,
                ))
                scopes.append((dataset, query, memories, sessions))
        phase = "cpu_dependencies"
        progress.update(phase=phase)
        import torch

        libraries = {name: version(name)
                     for name in ("transformers", "tokenizers", "torch", "Pillow")}
        phase = "exact_contexts"
        progress.update(phase=phase, total=len(scopes) * len(ALIASES) * len(METHODS) * 2)
        progress.update(subphase="source_fingerprint")
        source_fingerprint = context_source_fingerprint()
        rows = []
        processor_reports = {}
        progress.update(subphase="prompt_load")
        with step8_prompt_scope(settings.project_root):
            instruction = load_step8_prompt("pmmc_answer_contract_v3.txt")
        for alias in ALIASES:
            snapshot, processor_hashes = verified[alias]
            progress.update(subphase="baseline_config")
            baseline = _step8_config(settings, alias)
            with torch.device("cpu"), step8_prompt_scope(settings.project_root):
                progress.update(subphase="processor_load")
                counter = _create_counter(settings, registry.get(alias), snapshot)
                progress.update(subphase="budgeter_init")
                budgeter = ContextBudgeter(
                    ProviderConfiguredTokenCounter(counter, registry.get(alias)),
                    total_context_tokens=baseline.total_context_tokens,
                    reserved_output_tokens=baseline.reserved_output_tokens, paper_mode=True,
                )
                for dataset, source_query, source_memories, sessions in scopes:
                    progress.update(subphase="image_hydration")
                    query = source_query.model_copy(update={"query_image_refs": [
                        _hydrate_image(settings.benchmark_asset_path, ref, files)
                        for ref in source_query.query_image_refs
                    ]})
                    memories = [m.model_copy(update={"image_refs": [
                        _hydrate_image(settings.benchmark_asset_path, ref, files)
                        for ref in m.image_refs
                    ]}) for m in source_memories]
                    scope_hash = canonical_content_hash([
                        query.model_dump(mode="json"),
                        [m.model_dump(mode="json") for m in memories],
                        [s.model_dump(mode="json") for s in sessions],
                    ])
                    for method in METHODS:
                        progress.update(subphase="context_assembly")
                        assembled = budgeter.assemble(
                            run_id="context-audit", snapshot_id=dataset.snapshot_id,
                            instruction=instruction, query=query, memories=memories,
                            sessions=sessions, method=method, answer_model_alias=alias,
                            prompt_hash=prompt_sha256(instruction),
                            renderer_version=baseline.renderer_version,
                            temperature=baseline.temperature,
                            top_p=baseline.top_p, seed=baseline.seed,
                            answer_contract_version=baseline.answer_contract_version,
                        )
                        trace = assembled.trace
                        for variant, request in (
                            ("initial", assembled.request),
                            ("format_retry", answer_format_retry_request(assembled.request)),
                        ):
                            progress.update(subphase="transport_count")
                            tokens = _capture_request(counter, request)
                            progress.update(subphase="token_consistency")
                            if (tokens["prompt_tokens"] > trace.max_input_tokens
                                    or tokens["prompt_tokens"] != counter.count_request(request)
                                    or (variant == "initial"
                                        and tokens["prompt_tokens"] != trace.input_tokens)):
                                raise ValueError("exact_context_budget_drift")
                            progress.update(subphase="request_inventory")
                            row = {
                                "dataset": dataset.dataset.value, "model_alias": alias,
                                "method": method.value, "variant": variant,
                                "query_hash": canonical_content_hash(query.query_id),
                                "scope_hash": scope_hash, "estimated": False,
                                "visible_memory_count": len(memories),
                                "included_memory_count": len(trace.included_memory_ids),
                                "truncated_memory_count": len(trace.truncated_memory_ids),
                                "included_memory_ids_hash": canonical_content_hash(
                                    trace.included_memory_ids,
                                ),
                                "max_input_tokens": trace.max_input_tokens,
                                "reserved_output_tokens": trace.reserved_output_tokens,
                                **tokens,
                                **_image_stats(request, files, settings.benchmark_asset_path),
                            }
                            row["request_hash"] = canonical_content_hash(row)
                            rows.append(row)
                            progress.update(completed=len(rows))
                    # Bound decoded-image and request caches to one visible query.
                    _clear_counter_caches(counter)
            processor_reports[alias] = {
                "commit": lock.get(alias).commit_sha,
                "snapshot_content_hash": lock.get(alias).snapshot_content_hash,
                "processor_config_sha256": processor_hashes["preprocessor_config.json"],
                "processor_files_hash": canonical_content_hash(processor_hashes),
                "processor_fingerprint": counter._processor_fingerprint,
                "loaded_image_processor": verify_loaded_qwen_image_processor(
                    counter.processor.image_processor,
                    json.loads((snapshot / "preprocessor_config.json").read_bytes()),
                ),
                "processor_kwargs": {}, "chat_template_kwargs": {"enable_thinking": False},
                "device": "cpu", "weight_bytes_hashed": 0,
            }
            progress.update(subphase="processor_recheck")
            after = _verify_processor(
                settings, lock.get(alias), pinned["locked_processors"]["models"][alias],
            )
            if after[1] != processor_hashes:
                raise ValueError("processor_files_changed")
            del budgeter
            counter = None
        phase = "input_stability"
        progress.update(phase=phase)
        if any(sha256_file(path) != digest for path, digest in files.items()):
            raise ValueError("audit_input_changed")
        after_reconstruction = verify_reconstructed_inputs(
            config=config, project_root=settings.project_root,
        )
        if after_reconstruction != reconstruction:
            raise ValueError("reconstructed_inputs_changed")
        if context_source_fingerprint() != source_fingerprint:
            raise ValueError("context_implementation_changed")
        report = {
            "protocol": "step8-development-context-audit-v1", "state": "inspected",
            "scope": "rendered_full_context_initial_and_format_retry_native_qwen",
            "bounded_scope": {
                "methods": [method.value for method in METHODS],
                "model_aliases": list(ALIASES),
                "variants": ["initial", "format_retry"],
                "all_methods_covered": False, "full_pipeline_covered": False,
                "excludes": ["rag", "pmmc", "compiler", "grounding", "judge", "generation"],
            },
            "selection": selection, "selection_hash": chosen.content_hash,
            "source_profile_hash": profile.resolved_hash,
            "config_hash": canonical_content_hash(config.model_dump(mode="json")),
            "context_source_fingerprint": source_fingerprint,
            "selected_query_ids_hash": canonical_content_hash(
                chosen.model_dump(mode="json")["query_ids"],
            ),
            "partition_hash": partition.content_hash,
            "reconstruction_hash": canonical_content_hash(reconstruction),
            "input_limitations": reconstruction["restrictions"],
            "pinned_processor_audit_sha256": files[pinned_path],
            "model_lock_sha256": files[lock_path], "registry_hash": registry.resolved_hash,
            "runtime_file_sha256": runtime_hashes, "runtime_file_hashes_stable": True,
            "snapshot_file_hashes_verified": False,
            "prompt_sha256": prompt_sha256(instruction),
            "full_context_config_sha256": files[
                settings.resolve_path(Path("configs/baselines/full_context.yaml"))
            ],
            "context_policy": {
                "total_context_tokens": 32768, "reserved_output_tokens": 512,
                "max_input_tokens": 32256,
                "truncation": "keep_recent_contiguous_whole_memory_suffix",
                "answer_contract_version": TYPED_ANSWER_CONTRACT_VERSION,
                "budget_includes_format_retry": True,
            },
            "libraries": libraries,
            "locked_processors": processor_reports,
            "models": {alias: _model_layouts(
                [row for row in rows if row["model_alias"] == alias], processor_reports[alias],
            ) for alias in ALIASES},
            "requests": rows, "aggregate": _summarize(rows),
            "gold_read": False, "provider_calls": 0, "gpu_operations": 0,
            "images_modified": False, "input_manifests_modified": False,
            "scientific_approval": False, "capacity_verified": False,
            "gpu_benchmark_ready": False,
            "next_gate": "measure_real_joint_workloads_on_managed_gpu",
        }
        report["content_hash"] = canonical_content_hash(report)
        state = "inspected"
        return report
    except Exception as error:
        raise Step8ContextAuditError(phase, error=error, progress=progress) from None
    finally:
        progress.close(state)
        if isinstance(counter, ProcessorTokenCounter):
            _clear_counter_caches(counter)
