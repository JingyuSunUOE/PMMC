"""Auditable development-only reconstruction when historical inputs are lost.

Re-running a retained sampler is not evidence of historical execution. Neither
this receipt nor its partition permits paper approval or an unseen-test claim.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from pmmc.benchmarks import canonical
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import sha256_file, write_json
from pmmc.experiments.step8_selection import ExposureManifest, select_benchmark_queries
from pmmc.pilots import config as pilot_config
from pmmc.pilots import partition
from pmmc.schemas.base import DatasetName
from pmmc.schemas.pilot import BenchmarkPartitionSpec
from pmmc.schemas.step8 import CampaignPhase, Step8BenchmarkConfig

PROTOCOL = "step8-development-input-reconstruction-v1"
SAMPLER = Path("configs/compiler/questioner-quality-step7b-pilot.yaml")
RECEIPT = "input-reconstruction.json"
DERIVATION = "configuration-derivation.json"
RESTRICTIONS = {
    "history_status": "lost_no_backup",
    "historical_partition_equality": "unverified",
    "historical_sample_equality": "unverified",
    "historical_paired_comparison_available": False,
    "exposure_complete": False,
    "unseen_confirmation_allowed": False,
    "paper_test_allowed": False,
    "scientific_approval_allowed": False,
    "paper_result": False,
    "allowed_selections": ["diagnostic", "regression"],
}


class Step8InputReconstructionError(RuntimeError):
    def __init__(self, phase: str, error_type: str):
        super().__init__("development_input_reconstruction_failed")
        self.report = {
            "state": "failed",
            "code": str(self),
            "phase": phase,
            "error_type": error_type,
            "provider_calls": 0,
            "gpu_operations": 0,
        }


@contextmanager
def _progress(phase: str) -> Iterator[None]:
    started = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(30):
            print(
                f"STEP8_INPUT_RECONSTRUCTION phase={phase} state=running "
                f"elapsed_seconds={int(time.monotonic() - started)}",
                flush=True,
            )

    print(f"STEP8_INPUT_RECONSTRUCTION phase={phase} state=start", flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        print(f"STEP8_INPUT_RECONSTRUCTION phase={phase} state=failed", flush=True)
        raise
    else:
        print(f"STEP8_INPUT_RECONSTRUCTION phase={phase} state=done", flush=True)
    finally:
        stop.set()
        thread.join()


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("reconstruction artifact must be an object")
    return value


def verify_reconstructed_inputs(
    *,
    config: Step8BenchmarkConfig,
    project_root: Path,
) -> dict[str, Any] | None:
    """Verify the bundle and carry its limitations into the campaign, without gold."""
    regression_path = _resolve(project_root, config.regression_manifest)
    regression = _read(regression_path)
    if regression.get("protocol") != PROTOCOL:
        if (regression_path.parent / RECEIPT).exists():
            raise ValueError("reconstruction regression protocol mismatch")
        return None
    directory = regression_path.parent
    receipt = _read(directory / RECEIPT)
    if (
        receipt.get("protocol") != PROTOCOL
        or receipt.get("restrictions") != RESTRICTIONS
        or receipt.get("state") != "reconstructed_development_only"
    ):
        raise ValueError("reconstruction receipt restrictions mismatch")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("reconstruction file manifest missing")
    partition_dir = _resolve(project_root, config.partition_dir)
    required = {
        "benchmark-development.yaml",
        "development-regression.json",
        "exposure.json",
        "history-loss.json",
        "sampler-config.json",
        "reconstructed-samples.json",
        *{
            f"partitions/{partition_dir.name}/{name}"
            for name in (
                "manifest.json",
                "assignments.jsonl",
                "balance_report.json",
            )
        },
    }
    if "configuration_derivation" in receipt:
        if receipt["configuration_derivation"] != DERIVATION:
            raise ValueError("unsupported configuration derivation")
        required.add(DERIVATION)
    if set(files) != required:
        raise ValueError("reconstruction file manifest coverage mismatch")
    for name, digest in files.items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("reconstruction artifact escapes bundle")
        if sha256_file(path) != digest:
            raise ValueError("reconstruction artifact checksum mismatch")
    if (
        partition_dir.resolve() != (directory / "partitions" / partition_dir.name).resolve()
        or _resolve(project_root, config.exposure_manifest).resolve()
        != (directory / "exposure.json").resolve()
    ):
        raise ValueError("reconstruction input paths mismatch")
    manifest = partition.verify_partition(partition_dir)
    exposure = ExposureManifest.model_validate(_read(directory / "exposure.json"))
    if (
        exposure.complete
        or any(exposure.query_ids.values())
        or exposure.partition_hash != manifest.content_hash
        or exposure.content_hash != receipt.get("exposure_hash")
        or manifest.content_hash != receipt.get("partition_hash")
        or canonical_content_hash(regression["query_ids"]) != receipt.get("sample_query_ids_hash")
        or exposure.provenance_hashes != [sha256_file(directory / "history-loss.json")]
        or _read(directory / "history-loss.json") != {"protocol": PROTOCOL, **RESTRICTIONS}
    ):
        raise ValueError("reconstruction scientific identity mismatch")
    if DERIVATION in files:
        from pmmc.experiments.step8_input_derivation import verify_configuration_derivation

        verify_configuration_derivation(directory, receipt)
    return receipt


def reconstruct_development_inputs(
    *,
    project_root: Path,
    config_path: Path,
    snapshots: dict[DatasetName, Path],
    output_dir: Path,
    acknowledge_history_loss: bool = False,
) -> dict[str, Any]:
    """Publish an immutable input bundle. Never write snapshots, models or old inputs.

    No flock is needed: each invocation builds privately, then atomically renames
    its complete nonempty directory. A competing publication fails closed.
    """
    phase = "configuration"
    try:
        if acknowledge_history_loss is not True:
            raise ValueError("explicit history-loss acknowledgment required")
        root = project_root.resolve()
        output = _resolve(root, output_dir)
        if output.is_symlink() or not output.resolve().is_relative_to(
            root / "artifacts/step8-inputs"
        ):
            raise ValueError("reconstruction requires a separate step8-inputs subdirectory")
        if output.resolve() == root / "artifacts/step8-inputs":
            raise ValueError("reconstruction cannot replace the input root")
        output = output.resolve()
        config_path = _resolve(root, config_path)
        config = Step8BenchmarkConfig.model_validate(yaml.safe_load(config_path.read_text()))
        sampler_path = root / SAMPLER
        sampler = pilot_config.load_questioner_pilot_config(sampler_path)
        if (
            sampler.protocol_version != "questioner-quality-pilot-v4"
            or sampler.partition_seed != 0
            or sampler.partition_protocol != "scenario-balanced-five-fold-v2"
        ):
            raise ValueError("retained Step7B sampler identity required")
        # Cold reconstruction is a new instance, not a replay of a historical
        # runtime's float arithmetic. Existing inputs must use configuration derivation.
        sampler = pilot_config.QuestionerPilotConfig.model_validate({
            **sampler.model_dump(mode="json"),
            "partition_protocol": partition.EXACT_PARTITION_PROTOCOL,
        })
        if set(snapshots) != {DatasetName.MEMLENS, DatasetName.MEMGALLERY}:
            raise ValueError("both benchmark snapshots required")
        source_fingerprints = {}
        for module in (canonical, partition, pilot_config):
            if module.__file__ is None:
                raise ValueError("source fingerprint requires a source file")
            source_fingerprints[module.__name__] = sha256_file(Path(module.__file__))
        source_fingerprints[__name__] = sha256_file(Path(__file__))
        identities: dict[str, Any] = {}
        resolved_snapshots = []
        for dataset, value in sorted(snapshots.items()):
            path = _resolve(root, value).resolve()
            phase = f"snapshot_{dataset.value}"
            with _progress(phase):
                checked = canonical.preflight_snapshot(path)
                if checked.manifest.dataset is not dataset:
                    raise ValueError("snapshot dataset mismatch")
                identities[dataset.value] = {
                    "snapshot_id": checked.manifest.snapshot_id,
                    "canonical_content_hash": checked.manifest.canonical_content_hash,
                    "source_content_hash": checked.manifest.source_content_hash,
                    "manifest_sha256": sha256_file(path / "manifest.json"),
                    "counts": checked.counts,
                    "verification": "canonical_integrity_not_raw_image_bytes",
                }
                resolved_snapshots.append(path)
        inputs = {
            "snapshots": identities,
            "sampler_hash": sampler.resolved_hash,
            "sampler_file_sha256": sha256_file(sampler_path),
            "base_config_hash": canonical_content_hash(config.model_dump(mode="json")),
            "source_fingerprints": source_fingerprints,
        }
        phase = "publication"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            with _progress("existing_bundle_verification"):
                existing_config = Step8BenchmarkConfig.model_validate(
                    yaml.safe_load((output / "benchmark-development.yaml").read_text())
                )
                receipt = verify_reconstructed_inputs(config=existing_config, project_root=root)
                if receipt is None or receipt["inputs"] != inputs:
                    raise ValueError("existing reconstruction has incompatible inputs")
                return _summary(receipt, output, root, reused=True)
        with tempfile.TemporaryDirectory(prefix=".reconstruct-", dir=output.parent) as temporary:
            staging = Path(temporary)
            phase = "partition_and_sampling"
            with _progress(phase):
                partition_path, manifest = partition.build_partition(
                    snapshot_dirs=resolved_snapshots,
                    output_root=staging / "partitions",
                    spec=BenchmarkPartitionSpec(
                        protocol_version=sampler.partition_protocol,
                        seed=sampler.partition_seed,
                        fold_count=sampler.fold_count,
                        development_fold=sampler.development_fold,
                    ),
                )
                manifest = partition.verify_partition(partition_path)
                _, assignments = partition.load_partition(partition_path)
                samples = partition.sample_development_queries(
                    manifest=manifest,
                    assignments=assignments,
                    config=sampler,
                )
                ids = {sample.dataset: sample.query_ids for sample in samples}
                write_json(staging / "history-loss.json", {"protocol": PROTOCOL, **RESTRICTIONS})
                exposure = ExposureManifest(
                    partition_hash=manifest.content_hash,
                    complete=False,
                    query_ids={dataset: [] for dataset in snapshots},
                    provenance_hashes=[sha256_file(staging / "history-loss.json")],
                )
                selection = select_benchmark_queries(
                    campaign_id="reconstructed-development",
                    phase=CampaignPhase.DEVELOPMENT,
                    selection="regression",
                    partition_hash=manifest.content_hash,
                    assignments=assignments,
                    exposure=exposure,
                    regression_ids=ids,
                )
                write_json(staging / "exposure.json", exposure.model_dump(mode="json"))
                write_json(
                    staging / "development-regression.json",
                    {
                        "protocol": PROTOCOL,
                        "query_ids": ids,
                        "historical_sample_equality": "unverified",
                    },
                )
                write_json(staging / "sampler-config.json", sampler.model_dump(mode="json"))
                write_json(
                    staging / "reconstructed-samples.json",
                    {
                        "protocol": PROTOCOL,
                        "historical_execution_verified": False,
                        "samples": [sample.model_dump(mode="json") for sample in samples],
                    },
                )
                relative = output.relative_to(root)
                generated = config.model_dump(mode="json")
                generated.update(
                    partition_dir=str(relative / "partitions" / manifest.partition_id),
                    exposure_manifest=str(relative / "exposure.json"),
                    regression_manifest=str(relative / "development-regression.json"),
                )
                Step8BenchmarkConfig.model_validate(generated)
                # JSON is a YAML subset; deterministic bytes avoid serializer drift.
                write_json(staging / "benchmark-development.yaml", generated)
                receipt = {
                    "protocol": PROTOCOL,
                    "state": "reconstructed_development_only",
                    "restrictions": RESTRICTIONS,
                    "inputs": inputs,
                    "partition_hash": manifest.content_hash,
                    "exposure_hash": exposure.content_hash,
                    "sample_query_ids_hash": canonical_content_hash(ids),
                    "sample_counts": {d.value: len(q) for d, q in ids.items()},
                    "stratum_counts": selection.model_dump(mode="json")["stratum_counts"],
                    "files": {
                        str(p.relative_to(staging)): sha256_file(p)
                        for p in sorted(staging.rglob("*"))
                        if p.is_file()
                    },
                }
                write_json(staging / RECEIPT, receipt)
            phase = "publication"
            with _progress(phase):
                if output.exists():
                    raise ValueError("competing input publication; rerun to verify")
                staging.rename(output)
        return _summary(receipt, output, root, reused=False)
    except Exception as exc:
        raise Step8InputReconstructionError(phase, type(exc).__name__) from None


def _summary(
    receipt: dict[str, Any],
    directory: Path,
    root: Path,
    *,
    reused: bool,
) -> dict[str, Any]:
    return {
        "state": receipt["state"],
        "reused": reused,
        "restrictions": RESTRICTIONS,
        "sample_counts": receipt["sample_counts"],
        "partition_hash": receipt["partition_hash"],
        "partition_protocol": _read(directory / "sampler-config.json")["partition_protocol"],
        "receipt_sha256": sha256_file(directory / RECEIPT),
        "config": str(directory.relative_to(root) / "benchmark-development.yaml"),
        "provider_calls": 0,
        "gpu_operations": 0,
    }
