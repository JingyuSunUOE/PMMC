"""Portable diagnostics for a Step 6D run blocked before Doubter."""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import zstandard

from pmmc.pilots.io import file_sha256, read_json, read_jsonl, stable_hash, write_json
from pmmc.schemas import QuestionerPilotSample

_CAPSULE_PROTOCOL = "step6d-failure-capsule-v1"
_PRIVATE_FILES = (
    "r0_input_manifests.jsonl",
    "structured_output_item_diagnostics.jsonl",
)
_FORBIDDEN = re.compile(
    r"(?:/data/users/|/workspace/|https?://|s3://|minio://|"
    r"X-Amz-(?:Signature|Credential)|GOLD_ANSWER_SENTINEL|PMMC_GOLD_SENTINEL)",
    re.IGNORECASE,
)


def _safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, str) and _FORBIDDEN.search(value):
        return "[REDACTED_PORTABLE_LOCATOR]"
    return value


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    _safe(dict(value)),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and not path.is_relative_to(root / "transfer")
    ]


def _write_transfer(root: Path) -> tuple[Path, Path]:
    transfer = root / "transfer"
    transfer.mkdir(parents=True, exist_ok=True)
    archive_path = transfer / f"aaai-pmmc-step6d-{root.name}.tar.zst"
    checksum_path = transfer / f"aaai-pmmc-step6d-{root.name}.sha256"
    manifest = read_json(root / "manifest.json")
    included = [
        root / "manifest.json",
        *(
            root / str(item["path"])
            for item in list(manifest.get("artifacts", []))
        ),
    ]
    with tempfile.NamedTemporaryFile(dir=transfer, suffix=".tar", delete=False) as handle:
        tar_path = Path(handle.name)
    try:
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(
                included, key=lambda value: value.relative_to(root).as_posix()
            ):
                relative = PurePosixPath(root.name) / path.relative_to(root).as_posix()
                payload = path.read_bytes()
                info = tarfile.TarInfo(relative.as_posix())
                info.size = len(payload)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(payload))
        archive_path.write_bytes(
            zstandard.ZstdCompressor(
                level=9,
                threads=0,
                write_checksum=True,
                write_content_size=True,
            ).compress(tar_path.read_bytes())
        )
    finally:
        tar_path.unlink(missing_ok=True)
    checksum_path.write_text(
        f"{file_sha256(archive_path)}  {archive_path.name}\n",
        encoding="ascii",
    )
    return archive_path, checksum_path


def build_step6d_failure_capsule(
    *,
    launch_id: str,
    partition_id: str,
    samples: Sequence[QuestionerPilotSample],
    work: Sequence[Mapping[str, object]],
    launch_dir: Path,
    output_root: Path,
) -> tuple[Path, dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, object]] = {}
    issue_types: list[str] = []
    for item in work:
        dataset = str(item["dataset"])
        export_dir = Path(str(item["runtime_export_dir"]))
        export_manifest = read_json(export_dir / "export_manifest.json")
        r0_value = item.get("r0_manifest")
        r0 = dict(r0_value) if isinstance(r0_value, Mapping) else {}
        raw_issues = r0.get("issue_codes")
        issues = (
            [str(value) for value in raw_issues]
            if isinstance(raw_issues, list)
            else []
        )
        issue_types.extend(f"{dataset}:{value}" for value in issues)
        sources[dataset] = {
            "run_id": str(item["compiler_run_id"]),
            "runtime_export_id": str(export_manifest["export_id"]),
            "runtime_manifest_sha256": file_sha256(
                export_dir / "export_manifest.json"
            ),
            "r0_manifest_id": r0.get("r0_manifest_id"),
            "r0_content_hash": r0.get("content_hash"),
            "r0_issue_codes": issues,
        }
    identity = {
        "protocol": _CAPSULE_PROTOCOL,
        "launch_id": launch_id,
        "partition_id": partition_id,
        "sample_ids": sorted(item.sample_id for item in samples),
        "sources": sources,
    }
    capsule_id = "step6d-remediation-" + stable_hash(identity)[:24]
    destination = output_root / capsule_id
    if destination.exists():
        verified = verify_step6d_failure_capsule(destination)
        return destination, verified
    staging = output_root / f".{capsule_id}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    root = staging / capsule_id
    (root / "public").mkdir(parents=True)
    (root / "private" / "datasets").mkdir(parents=True)
    write_json(
        root / "public" / "automatic_summary.json",
        {
            "protocol_version": _CAPSULE_PROTOCOL,
            "status": "needs_remediation",
            "automatic_integrity_passed": False,
            "integrity_issue_types": sorted(set(issue_types)),
            "paper_result": False,
        },
    )
    write_json(root / "private" / "source_exports.json", _safe(sources))
    for item in work:
        dataset = str(item["dataset"])
        export_runtime = Path(str(item["runtime_export_dir"])) / "runtime"
        target = root / "private" / "datasets" / dataset
        for name in _PRIVATE_FILES:
            source = export_runtime / name
            rows = read_jsonl(source) if source.is_file() else []
            _write_jsonl(target / name, rows)
    for name in (
        "events.jsonl",
        "stage_results.jsonl",
        "checkpoint_events.jsonl",
        "checkpoint_errors.jsonl",
        "stage_components.jsonl",
        "errors.jsonl",
    ):
        source = launch_dir / name
        _write_jsonl(
            root / "private" / "launcher" / name,
            read_jsonl(source) if source.is_file() else [],
        )
    artifacts = _artifact_inventory(root)
    manifest_payload: dict[str, object] = {
        "capsule_id": capsule_id,
        "protocol_version": _CAPSULE_PROTOCOL,
        "status": "needs_remediation",
        "launch_id": launch_id,
        "partition_id": partition_id,
        "samples": [
            item.model_dump(mode="json")
            for item in sorted(samples, key=lambda value: value.dataset.value)
        ],
        "source_runtime_export_ids": {
            key: str(value["runtime_export_id"])
            for key, value in sorted(sources.items())
        },
        "automatic_integrity_passed": False,
        "artifacts": artifacts,
    }
    manifest: dict[str, object] = {
        **manifest_payload,
        "content_hash": stable_hash(manifest_payload),
    }
    write_json(root / "manifest.json", manifest)
    for directory in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        directory.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    _write_transfer(root)
    verify_step6d_failure_capsule(root)
    root.replace(destination)
    staging.rmdir()
    return destination, manifest


def build_step6d_runtime_failure_capsule(
    *,
    launch_id: str,
    stage: str,
    error_type: str,
    completed_stage_ids: Sequence[str],
    pending_stage_ids: Sequence[str],
    latest_milestone: str | None,
    resume_command: str,
    launch_dir: Path,
    output_root: Path,
) -> tuple[Path, dict[str, object]]:
    """Build a portable capsule for an unexpected Step 6D launcher failure."""

    identity = {
        "protocol": _CAPSULE_PROTOCOL,
        "launch_id": launch_id,
        "stage": stage,
        "error_type": error_type,
        "completed_stage_ids": list(completed_stage_ids),
        "pending_stage_ids": list(pending_stage_ids),
        "latest_milestone": latest_milestone,
    }
    capsule_id = "step6d-failure-" + stable_hash(identity)[:24]
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / capsule_id
    if destination.exists():
        return destination, verify_step6d_failure_capsule(destination)
    staging = output_root / f".{capsule_id}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    root = staging / capsule_id
    (root / "public").mkdir(parents=True)
    (root / "private" / "launcher").mkdir(parents=True)
    write_json(
        root / "public" / "automatic_summary.json",
        {
            "protocol_version": _CAPSULE_PROTOCOL,
            "status": "needs_remediation",
            "automatic_integrity_passed": False,
            "integrity_issue_types": [f"{stage}:{error_type}"],
            "paper_result": False,
        },
    )
    write_json(
        root / "private" / "failure.json",
        _safe(
            {
                **identity,
                "resume_command": resume_command,
            }
        ),
    )
    for name in (
        "events.jsonl",
        "stage_results.jsonl",
        "stage_performance.jsonl",
        "checkpoint_events.jsonl",
        "checkpoint_errors.jsonl",
        "stage_components.jsonl",
        "stage_component_errors.jsonl",
        "execution_attempts.jsonl",
        "errors.jsonl",
    ):
        source = launch_dir / name
        _write_jsonl(
            root / "private" / "launcher" / name,
            read_jsonl(source) if source.is_file() else [],
        )
    conformance = launch_dir / "step6d_compiler_conformance.diagnostics.json"
    if conformance.is_file():
        write_json(
            root / "private" / "launcher" / conformance.name,
            _safe(read_json(conformance)),
        )
    checkpoint_value = read_json(launch_dir / "launch_state.json").get(
        "metadata", {}
    )
    if isinstance(checkpoint_value, Mapping):
        checkpoint_path = checkpoint_value.get("latest_checkpoint")
        if isinstance(checkpoint_path, str):
            checkpoint = Path(checkpoint_path)
            for name in ("manifest.json", "verification.json"):
                source = checkpoint / name
                if source.is_file():
                    write_json(
                        root / "private" / "launcher" / f"checkpoint_{name}",
                        _safe(read_json(source)),
                    )
    artifacts = _artifact_inventory(root)
    manifest_payload: dict[str, object] = {
        "capsule_id": capsule_id,
        "protocol_version": _CAPSULE_PROTOCOL,
        "status": "needs_remediation",
        "launch_id": launch_id,
        "partition_id": None,
        "samples": [],
        "source_runtime_export_ids": {},
        "automatic_integrity_passed": False,
        "failure_stage": stage,
        "failure_error_type": error_type,
        "latest_milestone": latest_milestone,
        "artifacts": artifacts,
    }
    manifest: dict[str, object] = {
        **manifest_payload,
        "content_hash": stable_hash(manifest_payload),
    }
    write_json(root / "manifest.json", manifest)
    for directory in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        directory.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    _write_transfer(root)
    verify_step6d_failure_capsule(root)
    root.replace(destination)
    staging.rmdir()
    return destination, manifest


def verify_step6d_failure_capsule(root: Path) -> dict[str, object]:
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("protocol_version") != _CAPSULE_PROTOCOL
        or manifest.get("status") != "needs_remediation"
        or manifest.get("automatic_integrity_passed") is not False
    ):
        raise ValueError("invalid Step 6D failure capsule identity")
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "content_hash"
    }
    if stable_hash(payload) != manifest.get("content_hash"):
        raise ValueError("Step 6D failure capsule content hash mismatch")
    expected = {str(item["path"]): item for item in manifest.get("artifacts", [])}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and not path.is_relative_to(root / "transfer")
    }
    if set(actual) != set(expected):
        raise ValueError("Step 6D failure capsule file inventory mismatch")
    for relative, path in actual.items():
        item = expected[relative]
        if (
            path.stat().st_size != int(item["size_bytes"])
            or file_sha256(path) != item["sha256"]
            or _FORBIDDEN.search(path.read_text(encoding="utf-8", errors="ignore"))
        ):
            raise ValueError(f"Step 6D failure capsule artifact is unsafe: {relative}")
    archives = list((root / "transfer").glob("*.tar.zst"))
    checksums = list((root / "transfer").glob("*.sha256"))
    if len(archives) != 1 or len(checksums) != 1:
        raise ValueError("Step 6D failure capsule transfer is incomplete")
    parts = checksums[0].read_text(encoding="ascii").split()
    if (
        len(parts) != 2
        or parts[1] != archives[0].name
        or parts[0] != file_sha256(archives[0])
    ):
        raise ValueError("Step 6D failure capsule transfer checksum mismatch")
    return manifest


__all__ = [
    "build_step6d_failure_capsule",
    "build_step6d_runtime_failure_capsule",
    "verify_step6d_failure_capsule",
]
