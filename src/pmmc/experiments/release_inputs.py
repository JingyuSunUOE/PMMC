"""Local CPU preparation for standalone Step 8 exploration campaigns.

prepare_exploration returns a JSON-serializable, project-relative pointer. The
only child command is the native Typer benchmark plan. No DB, weight download,
GPU process, provider, historical approval or scientific result is needed.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import reduce
from importlib.util import module_from_spec
from operator import add
from pathlib import Path
from typing import Any

from pmmc.benchmarks.canonical import preflight_snapshot
from pmmc.benchmarks.registry import create_adapter
from pmmc.benchmarks.release_data import LOCK as SOURCE_LOCK
from pmmc.benchmarks.snapshot import CANONICAL_FILES, build_snapshot
from pmmc.benchmarks.validation import scan_adapter
from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.config import load_evaluation_config
from pmmc.experiments.step8_artifacts import sha256_file, write_json
from pmmc.experiments.step8_campaign import load_benchmark_config, make_protocol
from pmmc.experiments.step8_config_contract import CONFIG_FIELDS, FIXED_CONFIGS, POLICY_FIELDS
from pmmc.experiments.step8_exposure import recorded_exposure_query_ids
from pmmc.experiments.step8_jobs import SOURCE_OWNERS, scientific_source_fingerprint
from pmmc.experiments.step8_model_identity import load_step8_model_entry_hashes
from pmmc.experiments.step8_selection import ExposureManifest, denominator_report
from pmmc.infrastructure.settings import Settings
from pmmc.pilots import partition as native_partition
from pmmc.pilots.partition import load_partition, verify_partition
from pmmc.release_plan import plan_from_environment, plan_hash
from pmmc.runtime.config import load_runtime_profile
from pmmc.runtime.query_executor_config import load_query_executor_config
from pmmc.runtime.step8_capacity import capacity_image_workload
from pmmc.schemas import BenchmarkPartitionSpec, DatasetName, ValidationLevel
from pmmc.schemas.step8 import (
    BenchmarkCampaignManifest,
    CampaignPhase,
    SelectionExpansionManifest,
    Step8BenchmarkConfig,
)

_TEMPLATE = Path("configs/release/prepare-models.yaml")
_SPLIT = Path("configs/release/original-split-v2.json")
_SPLIT_VERSION = "memlens32k-memgallery20-original-v2"
_V2 = BenchmarkPartitionSpec(
    protocol_version="scenario-balanced-five-fold-v2", seed=0, fold_count=5, development_fold=0
)
_WRITER = Path("configs/compiler/questioner-release-w.yaml")
_READER = Path("configs/runtime/query-executor-release-r.yaml")
_DATASETS = (DatasetName.MEMLENS, DatasetName.MEMGALLERY)
_SNAPSHOT_FILES = (*CANONICAL_FILES, "manifest.json", "validation_report.json")
_VARIANTS = {"wr", "w", "r", "legacy"}


def _local(path: Path, parent: Path) -> Path:
    """Only regular project paths, never follow a pointer outside its namespace."""
    if not path.is_relative_to(parent) or path.resolve() != path:
        raise ValueError("exploration_path_outside_namespace_or_symlink")
    return path


def _run_root(settings: Settings, run_id: str, variant: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id) is None:
        raise ValueError("invalid_exploration_run_id")
    if variant not in _VARIANTS:
        raise ValueError("invalid_exploration_variant")
    root = settings.project_root.resolve()
    return _local(settings.outputs_path / "step8/exploration" / run_id, root)


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield fd


def _source_identity(root: Path) -> str:
    """A paired-method identity, not the image/CLI/operational revision."""
    base = load_benchmark_config(root / _TEMPLATE)
    paths = {key: getattr(base, key) for key in CONFIG_FIELDS}
    paths.update({key: Path(relative) for key, (_, relative) in FIXED_CONFIGS.items()})
    paths.update(
        writer_config=_WRITER,
        reader_config=_READER,
        model_lock=base.model_lock,
        original_split=_SPLIT,
    )
    # Bind resolved semantic prompt bytes as well as the config paths selecting
    # them. Stage ownership covers the compiler renderers and grounding prompt.
    paths.update(
        legacy_answer_prompt=load_query_executor_config(
            root / base.query_executor_config
        ).answer_prompt_path,
        release_answer_prompt=load_query_executor_config(root / _READER).answer_prompt_path,
        judge_prompt=load_evaluation_config(root / paths["evaluation_config"]).judge_prompt_path,
    )
    return canonical_content_hash(
        {
            "protocol": "step8-exploration-scientific-source-v1",
            "stages": {
                s: scientific_source_fingerprint(root, s) for s in SOURCE_OWNERS if s != "export"
            },
            "configs": {key: sha256_file(root / path) for key, path in paths.items()},
            "models": load_step8_model_entry_hashes(base, root),
            "policy": base.model_dump(mode="json", include=set(POLICY_FIELDS)),
        }
    )


def _known_exposure(settings: Settings, partition_dir: Path) -> ExposureManifest:
    manifest, assignments = load_partition(partition_dir)
    history = settings.outputs_path / "step8"
    ids = {dataset: set(values) for dataset, values in recorded_exposure_query_ids(history).items()}
    proofs = {canonical_content_hash({d.value: sorted(v) for d, v in ids.items()})}

    def include(values: dict[DatasetName, list[str]], *provenance: str) -> None:
        for dataset, known in values.items():
            ids[dataset].update(known)
        proofs.update(provenance)

    # Historical planned selections count conservatively; a new release plan
    # alone is not dispatch. The ledger and attempt/report evidence cover runs.
    inputs = settings.artifacts_path / "step8-inputs"
    observed_campaigns = [
        p.parent
        for p in history.glob("*/manifests/campaign.json")
        if json.loads(_local(p, history).read_bytes())["phase"] != "exploration"
        or any((p.parent.parent / "attempts").glob("attempt-*"))
        or any((p.parent.parent / "reports").glob("release-*.json"))
    ]
    exposure_files = [
        *inputs.glob("exposure.json"),
        *inputs.glob("*/exposure.json"),
        *(p / "exposure.json" for p in observed_campaigns if (p / "exposure.json").exists()),
    ]
    supplied = settings.benchmark_asset_path / "known-exposure.json"
    if os.path.lexists(supplied):
        exposure_files.append(_local(supplied, settings.benchmark_asset_path))
    for path in sorted(exposure_files):
        if path != supplied:
            _local(path, inputs if path.is_relative_to(inputs) else history)
        saved = ExposureManifest.model_validate_json(path.read_bytes())
        include(saved.query_ids, saved.content_hash)
    for path in sorted(history.glob("exploration/*/inputs/exposure.json")):
        saved = ExposureManifest.model_validate_json(_local(path, history).read_bytes())
        receipt_path = _local(path.with_name("inputs.json"), history)
        if not receipt_path.exists():  # A partial private restore is still known exposure.
            include(saved.query_ids, saved.content_hash)
            continue
        receipt = json.loads(receipt_path.read_bytes())
        if (
            saved.complete
            or saved.content_hash != receipt["exposure_hash"]
            or saved.partition_hash != receipt["partition_hash"]
        ):
            raise ValueError("exploration_retained_history_identity_mismatch")
        # Carry original provenance, not a chain of hashes of identical envelopes.
        include(saved.query_ids, *saved.provenance_hashes)
    # A retained campaign can have its private production receipt but no
    # surviving shared-ledger copy. Validate the native content-addressed event.
    for path in sorted(history.glob("*/private/exposure/*.json")):
        event = json.loads(_local(path, history).read_bytes())
        if (
            event.get("protocol") != "step8-exposure-event-v1"
            or event.get("purpose") != "production_dispatch"
            or canonical_content_hash(event) != path.stem
        ):
            raise ValueError("exploration_private_exposure_event_invalid")
        record = ExposureManifest(
            complete=False, partition_hash=event["partition_hash"], query_ids=event["query_ids"]
        )
        include(record.query_ids, path.stem)
    for path in sorted(
        p for folder in observed_campaigns for p in folder.glob("selections/*.json")
    ):
        _local(path, history)
        selection = SelectionExpansionManifest.model_validate_json(path.read_bytes())
        if path.stem != selection.content_hash:
            raise ValueError("exploration_history_selection_identity_mismatch")
        include(selection.query_ids, selection.content_hash)
    # Optional retained sample records are evidence of possible exposure only,
    # never a requirement for planning nor a source of the dev-all selection.
    for path in sorted(
        [*inputs.glob("development-regression.json"), *inputs.glob("*/development-regression.json")]
    ):
        sample = json.loads(_local(path, inputs).read_bytes())["query_ids"]
        record = ExposureManifest(
            partition_hash=manifest.content_hash, complete=False, query_ids=sample
        )
        include(record.query_ids, sha256_file(path))
    exposure = ExposureManifest(
        complete=False,
        partition_hash=manifest.content_hash,
        # The ledger always has two keys. A single-benchmark partition must
        # retain every known ID in that benchmark, not import another cohort.
        query_ids={
            d: sorted(ids[d]) if d in {item.dataset for item in manifest.datasets} else []
            for d in _DATASETS
        },
        provenance_hashes=sorted(proofs),
    )
    denominator_report(assignments, exposure)  # Don't silently discard unmappable known IDs.
    return exposure


def _build_original_partition(paths: list[Path], staging: Path) -> Path:
    # Native v2 used pre-3.12 left-to-right double summation. 3.12 changed
    # builtins.sum and can assign OLD PAPER queries to fold 0. Load the same
    # native builder with that one arithmetic primitive in an isolated module
    # namespace. Do not patch builtins or change other partition callers.
    spec = native_partition.__spec__
    if spec is None or spec.loader is None:
        raise ValueError("exploration_native_partition_loader_missing")
    native = module_from_spec(spec)
    native.__dict__["sum"] = lambda values: reduce(add, values, 0)
    spec.loader.exec_module(native)
    directory, _ = native.build_partition(snapshot_dirs=paths, output_root=staging, spec=_V2)
    return directory


def _original_membership(root: Path, partition: Path, datasets=_DATASETS) -> dict:
    """Certify FULL identities and original query-to-fold membership, not just counts."""
    reference = json.loads(_local(root / _SPLIT, root).read_bytes())
    manifest, rows = load_partition(partition)
    if (
        reference["dataset_split_version"] != _SPLIT_VERSION
        or manifest.spec != _V2
        or set(reference["datasets"]) != {d.value for d in _DATASETS}
        or {d.dataset for d in manifest.datasets} != set(datasets)
    ):
        raise ValueError("exploration_original_split_required")
    digests = {}
    for item in manifest.datasets:
        expected = reference["datasets"][item.dataset.value]
        if (
            item.snapshot_id != expected["full_snapshot_id"]
            or item.snapshot_content_hash != expected["canonical_sha256"]
        ):
            raise ValueError("exploration_unknown_data_requires_versioned_dataset_and_split")
        assignments = sorted(
            (r.query_id, r.fold, r.role.value) for r in rows if r.dataset == item.dataset
        )
        measured = {
            "assignment_sha256": canonical_content_hash(assignments),
            "development_sha256": canonical_content_hash(
                [q for q, _, role in assignments if role == "development"]
            ),
            "paper_sha256": canonical_content_hash(
                [q for q, _, role in assignments if role == "paper_test"]
            ),
        }
        if any(expected[key] != digest for key, digest in measured.items()):
            raise ValueError("exploration_membership_differs_from_original_frozen_assignments")
        digests[item.dataset.value] = measured
    return {
        "dataset_split_version": _SPLIT_VERSION,
        "reference_partition_hash": reference["reference_partition_hash"],
        "datasets": digests,
    }


def _selected_datasets(request: dict | None) -> tuple[DatasetName, ...]:
    return tuple(DatasetName(d) for d in request["benchmarks"]) if request else _DATASETS


def _source_locks(root: Path, datasets: tuple[DatasetName, ...]) -> dict[str, str]:
    lock = json.loads(_local(root / SOURCE_LOCK, root).read_bytes())
    return {
        d.value: canonical_content_hash([lock["protocol"], lock["datasets"][d.value]])
        for d in datasets
    }


def _bootstrap_inputs(settings: Settings, run: Path, request: dict | None = None) -> dict:
    datasets = _selected_datasets(request)
    locks = _source_locks(settings.project_root.resolve(), datasets) if request else None
    directory = run / "inputs"
    retained = directory.exists()
    if not retained:
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=run) as tmp:
            staging = Path(tmp) / "inputs"
            snapshots = {}
            paths = []
            # The snapshot publisher uses a shared content-addressed destination.
            # Serialize cold creation even across independent run IDs.
            with _lock(run.parent / ".snapshots.lock"):
                for dataset in datasets:
                    source = (
                        settings.memlens_path
                        if dataset is DatasetName.MEMLENS
                        else settings.memgallery_path
                    )
                    adapter = create_adapter(
                        dataset, root=source, project_root=settings.benchmark_asset_path
                    )
                    path, snapshot = build_snapshot(
                        adapter,
                        output_root=settings.benchmark_artifacts_path,
                        level=ValidationLevel.FULL,
                    )
                    checked = preflight_snapshot(path)
                    if (
                        snapshot.dataset is not dataset
                        or checked.manifest != snapshot
                        or checked.validation_report.level is not ValidationLevel.FULL
                        or snapshot.metadata.get("validation_level") != "full"
                    ):
                        raise ValueError("exploration_requires_full_snapshot")
                    paths.append(path)
                    snapshots[dataset.value] = {
                        "snapshot_id": snapshot.snapshot_id,
                        "source_hash": snapshot.source_content_hash,
                        "canonical_hash": snapshot.canonical_content_hash,
                        "files": {
                            name: sha256_file(_local(path / name, path)) for name in _SNAPSHOT_FILES
                        },
                    }
            partition = _build_original_partition(paths, staging)
            manifest = verify_partition(partition)
            membership = _original_membership(settings.project_root.resolve(), partition, datasets)
            exposure = _known_exposure(settings, partition)
            write_json(staging / "exposure.json", exposure.model_dump(mode="json"), immutable=True)
            write_json(
                staging / "inputs.json",
                {
                    **(
                        {"release_request": request, "benchmark_source_locks": locks}
                        if request is not None
                        else {}
                    ),
                    "partition_id": manifest.partition_id,
                    "partition_hash": manifest.content_hash,
                    "exposure_hash": exposure.content_hash,
                    "snapshots": snapshots,
                    "split_membership": membership,
                    "source_identity": _source_identity(settings.project_root.resolve()),
                },
                immutable=True,
            )
            staging.rename(directory)

    receipt = json.loads(_local(directory / "inputs.json", run).read_bytes())
    if request != receipt.get("release_request"):
        raise ValueError("exploration_release_request_changed_use_new_run_id")
    partition = _local(directory / receipt["partition_id"], directory)
    manifest = verify_partition(partition)
    exposure = ExposureManifest.model_validate_json(
        _local(directory / "exposure.json", run).read_bytes()
    )
    if (
        manifest.content_hash != receipt["partition_hash"]
        or locks != receipt.get("benchmark_source_locks")
        or _original_membership(settings.project_root.resolve(), partition, datasets)
        != receipt.get("split_membership")
        or exposure.complete
        or exposure.partition_hash != manifest.content_hash
        or exposure.content_hash != receipt["exposure_hash"]
        or set(receipt["snapshots"]) != {d.value for d in datasets}
    ):
        raise ValueError("exploration_frozen_inputs_changed")
    for entry in manifest.datasets:
        frozen = receipt["snapshots"][entry.dataset.value]
        if retained:
            adapter = create_adapter(
                entry.dataset,
                project_root=settings.benchmark_asset_path,
                root=settings.memlens_path
                if entry.dataset is DatasetName.MEMLENS
                else settings.memgallery_path,
            )
            raw = scan_adapter(adapter, ValidationLevel.FULL)
            if not raw.report.success or raw.source_content_hash != frozen["source_hash"]:
                raise ValueError("exploration_raw_inputs_changed_use_new_run_id")
        path = settings.benchmark_artifacts_path / entry.dataset.value / entry.snapshot_id
        if (
            frozen["snapshot_id"] != entry.snapshot_id
            or frozen["canonical_hash"] != entry.snapshot_content_hash
            or frozen["files"]
            != {name: sha256_file(_local(path / name, path)) for name in _SNAPSHOT_FILES}
        ):
            raise ValueError("exploration_frozen_snapshot_changed")
    return receipt


def _config(settings: Settings, run: Path, variant: str, inputs: dict) -> Step8BenchmarkConfig:
    root = settings.project_root.resolve()
    base = load_benchmark_config(root / _TEMPLATE)
    directory = (run / "inputs").relative_to(root)
    updates = {
        "partition_dir": directory / inputs["partition_id"],
        "exposure_manifest": directory / "exposure.json",
        "output_root": (settings.outputs_path / "step8").relative_to(root),
        "metadata": {
            **(
                {"release_request": inputs["release_request"]}
                if "release_request" in inputs
                else {}
            ),
            "exploration_run_id": run.name,
            "exploration_variant": variant,
            "source_identity": inputs["source_identity"],
            "dataset_split_version": _SPLIT_VERSION,
            "historical_approvals_allowed": False,
            "paper_result": False,
            "exposure_history_complete": False,
        },
    }
    if variant in {"w", "wr"}:
        updates["questioner_config"] = _WRITER
    if variant in {"r", "wr"}:
        updates["query_executor_config"] = _READER
    from pmmc.runtime.release_topology import materialize

    if derived := materialize(settings):
        from pmmc.experiments.execution import load_execution_config

        updates.update(derived)
        execution = load_execution_config(root / derived["execution_profile"])
        workers = execution.unit_preparation.workers
        updates["execution"] = {**base.execution.model_dump(), "cpu_workers": workers}
    if "benchmark_source_locks" in inputs:
        updates["metadata"]["benchmark_source_locks"] = inputs["benchmark_source_locks"]
    config = Step8BenchmarkConfig.model_validate({**base.model_dump(), **updates})
    profile = load_runtime_profile(root / config.runtime_profile)
    gated = [s for s in profile.services if s.metadata.get("requires_capacity_receipt")]
    if (
        profile.model_lock_path != config.model_lock
        or len(gated) != 1
        or gated[0].model_alias != "answer-qwen3.5-27b"
        or not profile.require_calibration_in_paper
    ):
        raise ValueError("exploration_capacity_gate_missing")
    capacity_image_workload(gated[0], multimodal=True)
    return config


def _plan(
    settings: Settings, config_path: Path, campaign_dir: Path | None, *, lock_fd: int
) -> Path:
    command = [
        sys.executable,
        "-m",
        "pmmc",
        "benchmark",
        "plan",
        "--config",
        str(config_path),
        "--phase",
        "exploration",
        "--selection",
        "dev-all",
    ]
    if campaign_dir is not None:
        command.extend(["--campaign-dir", str(campaign_dir)])
    root = settings.project_root.resolve()
    result = subprocess.run(
        command,
        cwd=root,
        env={
            **os.environ,
            "PMMC_PROJECT_ROOT": str(root),
            "PYTHONPATH": os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]),
        },
        capture_output=True,
        pass_fds=(lock_fd,),
        text=True,
        check=False,
    )
    if result.returncode:
        # Typer validation traces may include private benchmark fields.
        raise ValueError("exploration_benchmark_plan_failed")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if (report["state"], report["phase"], report["selection"], report["model_calls"]) != (
        "planned",
        "exploration",
        "dev-all",
        0,
    ):
        raise ValueError("exploration_unexpected_plan_result")
    return Path(report["campaign_dir"])


def _recover_campaign(settings: Settings, config: Step8BenchmarkConfig) -> Path | None:
    # A process may die after Typer publishes a plan but before pointer.json.
    # Recover by BOTH ownership keys, never by protocol alone or directory time.
    output = settings.resolve_path(config.output_root)
    matches = []
    for path in output.glob("*_exploration_*/manifests/config.json"):
        other = load_benchmark_config(_local(path, output))
        if all(
            other.metadata.get(key) == config.metadata[key]
            for key in ("exploration_run_id", "exploration_variant")
        ):
            if other != config:
                raise ValueError("exploration_owned_config_conflict")
            matches.append(path.parent.parent)
    if len(matches) > 1:
        raise ValueError("exploration_ambiguous_owned_campaign")
    return next(iter(matches), None)


def _pointer(
    settings: Settings, run: Path, variant: str, directory: Path, *, planned: bool = False
) -> dict[str, Any]:
    root = settings.project_root.resolve()
    output = settings.outputs_path / "step8"
    if _local(directory, output).parent != output:
        raise ValueError("exploration_campaign_directory_invalid")
    config_path = _local(run / variant / "benchmark.yaml", run)
    config = load_benchmark_config(config_path)
    manifests = _local(directory / "manifests", directory)
    campaign_path = _local(manifests / "campaign.json", manifests)
    campaign = BenchmarkCampaignManifest.model_validate_json(campaign_path.read_bytes())
    selections = list(_local(manifests / "selections", manifests).glob("*.json"))
    request = config.release_request
    datasets = _selected_datasets(request)
    digest = plan_hash(request) if request else None
    membership = _original_membership(
        root, _local(root / config.partition_dir, run / "inputs"), datasets
    )
    if (
        len(selections) != 1
        or config != load_benchmark_config(_local(manifests / "config.json", manifests))
        or config.metadata.get("exploration_run_id") != run.name
        or config.metadata.get("exploration_variant") != variant
        or config.metadata.get("historical_approvals_allowed") is not False
        or config.metadata.get("source_identity") != _source_identity(root)
        or config.metadata.get("dataset_split_version") != _SPLIT_VERSION
        or campaign.phase is not CampaignPhase.EXPLORATION
        or campaign.release_plan_hash != digest
        or set(campaign.snapshot_ids) != set(datasets)
        or (
            request
            and config.metadata.get("benchmark_source_locks") != _source_locks(root, datasets)
        )
        or (
            request is not None
            and (request["run_id"] != run.name or variant not in request["variants"])
        )
        or campaign.protocol != make_protocol(config, root)
        or campaign.paper_result
        or campaign.protocol_freeze_hash is not None
    ):
        raise ValueError("exploration_campaign_identity_mismatch")
    selection = SelectionExpansionManifest.model_validate_json(
        _local(selections[0], manifests).read_bytes()
    )
    if (
        selection.content_hash != selections[0].stem
        or selection.campaign_id != campaign.campaign_id
        or selection.phase is not CampaignPhase.EXPLORATION
        or selection.selection != "dev-all"
        or selection.partition_hash != campaign.protocol.partition_hash
        or selection.release_plan_hash != digest
        or set(selection.query_ids) != set(datasets)
        or any(
            canonical_content_hash(sorted(ids))
            != membership["datasets"][d.value]["development_sha256"]
            for d, ids in selection.query_ids.items()
        )
    ):
        raise ValueError("exploration_selection_identity_mismatch")
    inventory_sha = sha256_file(
        _local(manifests / f"jobs-{selection.content_hash}.json", manifests)
    )
    requirements = {}
    if request:
        from pmmc.experiments.release_models import stage_requirements

        # The native planner has just verified every private payload. Freeze
        # their alias projection once so read-only consumers need not rescan
        # tens of thousands of job files to validate an existing pointer.
        path = _local(manifests / f"stage-requirements-{selection.content_hash}.json", manifests)
        identity = dict(
            campaign_id=campaign.campaign_id,
            selection_hash=selection.content_hash,
            plan_hash=digest,
            job_inventory_sha256=inventory_sha,
        )
        if planned:
            required = stage_requirements(directory, campaign, selection.content_hash)
            write_json(path, {**identity, "requirements": required}, immutable=True)
        saved = json.loads(path.read_bytes())
        requirements = saved.pop("requirements")
        if saved != identity:
            raise ValueError("release_model_inventory_binding_mismatch")
    return {
        **(
            {
                "release_plan_hash": digest,
                **requirements,
                "gpu_stages": sorted(
                    k for k, v in requirements["local_gpu_required_by_stage"].items() if v
                ),
                "local_gpu_required": any(requirements["local_gpu_required_by_stage"].values()),
            }
            if digest is not None
            else {}
        ),
        "state": "planned",
        "phase": "exploration",
        "selection": "dev-all",
        "run_id": run.name,
        "variant": variant,
        "campaign_id": campaign.campaign_id,
        "campaign_dir": directory.relative_to(root).as_posix(),
        "config_path": config_path.relative_to(root).as_posix(),
        "pointer_path": (config_path.parent / "pointer.json").relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "protocol_hash": campaign.protocol.content_hash,
        "campaign_sha256": sha256_file(campaign_path),
        "partition_hash": selection.partition_hash,
        "split_membership": membership,
        "selection_hash": selection.content_hash,
        "exposure_hash": campaign.protocol.exposure_manifest_hash,
        "selected": {d.value: len(ids) for d, ids in selection.query_ids.items()},
        "job_inventory_sha256": inventory_sha,
        "source_identity": config.metadata["source_identity"],
        "runtime_sha256": sha256_file(root / config.runtime_profile),
        "execution_sha256": sha256_file(root / config.execution_profile),
        "capacity_calibration_required": not request
        or any(
            "answer-qwen3.5-27b" in aliases
            for aliases in requirements["calibration_aliases_by_stage"].values()
        ),
        "gpu_benchmark_ready": False,
        "paper_result": False,
        "provider_calls": 0,
        "gpu_operations": 0,
    }


def exploration_campaign_directory(settings: Settings, run_id: str, *, variant: str = "wr") -> Path:
    """Resolve and validate an existing pointer, without creating or planning anything."""
    run = _run_root(settings, run_id, variant)
    pointer = _local(run / variant / "pointer.json", run)
    saved = json.loads(pointer.read_bytes())
    directory = settings.project_root.resolve() / saved["campaign_dir"]
    if saved != _pointer(settings, run, variant, directory):
        raise ValueError("exploration_pointer_identity_mismatch")
    return directory


def prepare_exploration(settings: Settings, run_id: str, *, variant: str = "wr") -> dict[str, Any]:
    """Freeze one input cohort per ID; publish one immutable native plan per variant."""
    run = _run_root(settings, run_id, variant)
    request = plan_from_environment(os.environ)
    if request and (request["run_id"] != run_id or variant not in request["variants"]):
        raise ValueError("exploration_request_run_or_variant_mismatch")
    if request and request["stage"] == "analysis" and not request["with_deps"]:
        raise ValueError("source_analysis_cannot_plan_native_work_without_deps")
    config_path = _local(run / variant / "benchmark.yaml", run)
    pointer = config_path.parent / "pointer.json"
    with _lock(run / ".prepare.lock") as lock_fd:
        inputs = _bootstrap_inputs(settings, run, request)
        if inputs["source_identity"] != _source_identity(settings.project_root.resolve()):
            raise ValueError("exploration_source_changed_use_new_run_id")
        config = _config(settings, run, variant, inputs)
        write_json(config_path, config.model_dump(mode="json"), immutable=True)
        existing: Path | None
        if pointer.exists():
            existing = exploration_campaign_directory(settings, run_id, variant=variant)
        else:
            existing = _recover_campaign(settings, config)
        directory = _plan(settings, config_path, existing, lock_fd=lock_fd)
        if existing is not None and directory != existing:
            raise ValueError("exploration_plan_replaced_campaign")
        result = _pointer(settings, run, variant, directory, planned=True)
        write_json(pointer, result, immutable=True)
        return result
