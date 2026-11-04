"""Change benchmark configuration without repartitioning or resampling inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import sha256_file, write_json
from pmmc.experiments.step8_input_reconstruction import (
    DERIVATION,
    PROTOCOL,
    RECEIPT,
    RESTRICTIONS,
    _resolve,
    verify_reconstructed_inputs,
)
from pmmc.schemas.step8 import Step8BenchmarkConfig

DERIVATION_PROTOCOL = "step8-input-config-derivation-v1"
CONFIG = "benchmark-development.yaml"
MAX_LINEAGE_DEPTH = 8
MAX_PROOF_BYTES = 4 * 1024 * 1024
PATH_FIELDS = ("partition_dir", "exposure_manifest", "regression_manifest")
ALLOWED_CHANGES = ("protocol_name", "model_lock", "execution.development_budget_limit_usd")


class Step8InputDerivationError(RuntimeError):
    def __init__(self, phase: str, error_type: str):
        super().__init__("development_input_derivation_failed")
        self.report = {
            "state": "failed",
            "code": str(self),
            "phase": phase,
            "error_type": error_type,
            "provider_calls": 0,
            "gpu_operations": 0,
        }


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_PROOF_BYTES:
        raise ValueError("input identity artifact exceeds size limit")
    # Preserve exact bytes (including line endings) in the portable lineage.
    return path.read_bytes().decode("utf-8")


def _config(text: str) -> dict[str, Any]:
    return Step8BenchmarkConfig.model_validate(yaml.safe_load(text)).model_dump(mode="json")


def _changes(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    left, right = json.loads(json.dumps(before)), json.loads(json.dumps(after))
    changes = []
    for field in PATH_FIELDS:
        left.pop(field)
        right.pop(field)
    for field in ALLOWED_CHANGES:
        a, b = left, right
        parts = field.split(".")
        for part in parts[:-1]:
            a, b = a[part], b[part]
        old, new = a.pop(parts[-1]), b.pop(parts[-1])
        if old != new:
            changes.append({"field": field, "before": old, "after": new})
    if left != right:
        raise ValueError("configuration derivation changes a non-allowlisted field")
    return changes


def _check_paths(config: dict[str, Any], directory: str, partition_id: str) -> None:
    root = Path(directory)
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("lineage directory must be project relative")
    expected = (
        root / "partitions" / partition_id,
        root / "exposure.json",
        root / "development-regression.json",
    )
    if any(Path(config[key]) != value for key, value in zip(PATH_FIELDS, expected, strict=True)):
        raise ValueError("lineage configuration input paths mismatch")


def _verify_link(
    receipt: dict[str, Any],
    config_text: str,
    proof_text: str,
    *,
    depth: int = 0,
) -> None:
    if depth >= MAX_LINEAGE_DEPTH or len(proof_text.encode("utf-8")) > MAX_PROOF_BYTES:
        raise ValueError("configuration lineage limit exceeded")
    proof = json.loads(proof_text)
    if (
        proof["protocol"] != DERIVATION_PROTOCOL
        or proof["partition_rebuilt"] is not False
        or proof["samples_resampled"] is not False
        or receipt.get("configuration_derivation") != DERIVATION
        or receipt["files"][DERIVATION] != _hash_text(proof_text)
        or receipt["files"][CONFIG] != _hash_text(config_text)
    ):
        raise ValueError("configuration lineage identity mismatch")
    parent_text = proof["parent_receipt_json"]
    parent = json.loads(parent_text)
    parent_config_text = proof["parent_config_text"]
    if (
        _hash_text(parent_text) != proof["parent_receipt_sha256"]
        or parent["files"][CONFIG] != _hash_text(parent_config_text)
        or parent["protocol"] != PROTOCOL
        or parent["state"] != "reconstructed_development_only"
        or parent["restrictions"] != RESTRICTIONS
    ):
        raise ValueError("parent reconstruction identity mismatch")
    ignored = {"files", "configuration_derivation"}
    if {k: v for k, v in receipt.items() if k not in ignored} != {
        k: v for k, v in parent.items() if k not in ignored
    }:
        raise ValueError("derivation changed inherited scientific identity")
    preserved = {k: v for k, v in parent["files"].items() if k not in {CONFIG, DERIVATION}}
    current = {k: v for k, v in receipt["files"].items() if k not in {CONFIG, DERIVATION}}
    if len(preserved) != 8 or current != preserved:
        raise ValueError("derivation did not preserve all frozen input files")
    before, after = _config(parent_config_text), _config(config_text)
    partition_id = Path(before["partition_dir"]).name
    _check_paths(before, proof["source_directory"], partition_id)
    _check_paths(after, proof["output_directory"], partition_id)
    if proof["changes"] != _changes(before, after) or proof[
        "target_config_hash"
    ] != canonical_content_hash(after):
        raise ValueError("configuration change audit mismatch")
    template = proof["template_config"]
    if _config(json.dumps(template)) != template:
        raise ValueError("configuration template must be normalized")
    rebound = {**template, **{key: after[key] for key in PATH_FIELDS}}
    if rebound != after or proof["template_config_hash"] != canonical_content_hash(template):
        raise ValueError("configuration template identity mismatch")
    parent_proof = proof["parent_derivation_json"]
    if "configuration_derivation" in parent:
        if not isinstance(parent_proof, str):
            raise ValueError("parent derivation missing")
        _verify_link(parent, parent_config_text, parent_proof, depth=depth + 1)
        if json.loads(parent_proof)["output_directory"] != proof["source_directory"]:
            raise ValueError("configuration lineage directory discontinuity")
    elif parent_proof is not None or DERIVATION in parent["files"]:
        raise ValueError("unexpected parent derivation")


def verify_configuration_derivation(directory: Path, receipt: dict[str, Any]) -> None:
    """Verify portable ancestry without requiring the original directory to remain mounted."""
    _verify_link(receipt, _read_text(directory / CONFIG), _read_text(directory / DERIVATION))


def derive_development_inputs(
    *,
    project_root: Path,
    source_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    phase = "source_verification"
    try:
        root = project_root.resolve()
        source, output = _resolve(root, source_dir), _resolve(root, output_dir)
        for path in (source, output):
            if path.is_symlink() or not path.resolve().is_relative_to(
                root / "artifacts/step8-inputs"
            ):
                raise ValueError("input bundles require separate project input directories")
        source, output = source.resolve(), output.resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("source and output bundles must not overlap")
        source_config_text = _read_text(source / CONFIG)
        source_config = Step8BenchmarkConfig.model_validate(yaml.safe_load(source_config_text))
        if _resolve(root, source_config.regression_manifest).resolve().parent != source:
            raise ValueError("source configuration points outside source bundle")
        parent = verify_reconstructed_inputs(config=source_config, project_root=root)
        if parent is None:
            raise ValueError("a verified reconstruction source is required")
        parent_text = _read_text(source / RECEIPT)
        if json.loads(parent_text) != parent:
            raise ValueError("source receipt changed during verification")
        print("STEP8_INPUT_DERIVATION phase=source_verification state=passed", flush=True)
        phase = "configuration"
        template = _config(_read_text(_resolve(root, config_path)))
        relative = output.relative_to(root)
        target = {
            **template,
            "partition_dir": str(relative / "partitions" / Path(source_config.partition_dir).name),
            "exposure_manifest": str(relative / "exposure.json"),
            "regression_manifest": str(relative / "development-regression.json"),
        }
        Step8BenchmarkConfig.model_validate(target)
        changes = _changes(source_config.model_dump(mode="json"), target)
        proof = {
            "protocol": DERIVATION_PROTOCOL,
            "source_directory": str(source.relative_to(root)),
            "output_directory": str(relative),
            "parent_receipt_json": parent_text,
            "parent_receipt_sha256": _hash_text(parent_text),
            "parent_config_text": source_config_text,
            "parent_derivation_json": (
                _read_text(source / DERIVATION) if "configuration_derivation" in parent else None
            ),
            "template_config": template,
            "template_config_hash": canonical_content_hash(template),
            "target_config_hash": canonical_content_hash(target),
            "changes": changes,
            "partition_rebuilt": False,
            "samples_resampled": False,
        }
        phase = "publication"
        reused = output.exists()
        if reused:
            existing_config = Step8BenchmarkConfig.model_validate(
                yaml.safe_load(_read_text(output / CONFIG))
            )
            receipt = verify_reconstructed_inputs(config=existing_config, project_root=root)
            if (
                receipt is None
                or DERIVATION not in receipt["files"]
                or json.loads(_read_text(output / DERIVATION)) != proof
                or existing_config.model_dump(mode="json") != target
            ):
                raise ValueError("existing derived bundle has incompatible identity")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".derive-", dir=output.parent) as temporary:
                staging = Path(temporary)
                preserved = {
                    k: v for k, v in parent["files"].items() if k not in {CONFIG, DERIVATION}
                }
                for name, digest in preserved.items():
                    destination = staging / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, destination)
                    if sha256_file(destination) != digest:
                        raise ValueError("source artifact changed during copy")
                write_json(staging / CONFIG, target)
                write_json(staging / DERIVATION, proof)
                receipt = {
                    **parent,
                    "configuration_derivation": DERIVATION,
                    "files": {
                        **preserved,
                        CONFIG: sha256_file(staging / CONFIG),
                        DERIVATION: sha256_file(staging / DERIVATION),
                    },
                }
                write_json(staging / RECEIPT, receipt)
                verify_configuration_derivation(staging, receipt)
                # Recheck after copying; never publish a mixture of source revisions.
                if (
                    verify_reconstructed_inputs(config=source_config, project_root=root) != parent
                    or _read_text(source / RECEIPT) != parent_text
                ):
                    raise ValueError("source bundle changed during derivation")
                if output.exists():
                    raise ValueError("competing input publication; rerun to verify")
                staging.rename(output)
        print("STEP8_INPUT_DERIVATION phase=publication state=passed", flush=True)
        return {
            "protocol": DERIVATION_PROTOCOL,
            "state": "derived_development_only",
            "reused": reused,
            "partition_unchanged": True,
            "samples_unchanged": True,
            "preserved_file_count": 8,
            "partition_hash": receipt["partition_hash"],
            "sample_query_ids_hash": receipt["sample_query_ids_hash"],
            "sample_counts": receipt["sample_counts"],
            "restrictions": receipt["restrictions"],
            "changed_fields": [item["field"] for item in changes],
            "parent_receipt_sha256": proof["parent_receipt_sha256"],
            "receipt_sha256": sha256_file(output / RECEIPT),
            "config": str(relative / CONFIG),
            "provider_calls": 0,
            "gpu_operations": 0,
        }
    except Exception as exc:
        raise Step8InputDerivationError(phase, type(exc).__name__) from None
