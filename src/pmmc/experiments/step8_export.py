"""Streaming deterministic Step 8 bundles from explicit, already-staged inputs.

No database scans, dataset discovery, service shutdown, or publication are performed.
Public review is an explicit caller attestation, not an automatic privacy detector.
Checksums establish integrity, not authenticity. Pin the receipt hash out of band.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import IO, BinaryIO, Literal, cast

import zstandard

_CHUNK = 1024 * 1024
_MANIFEST = "artifact_manifest.json"
_MAX_MANIFEST = 16 * 1024 * 1024
_VERSION = "step8-audit-bundle-v1"
_PUBLIC_KINDS = frozenset(
    {
        "aggregate",
        "configuration",
        "protocol_lock",
        "model_lock",
        "schema",
        "reproduction_script",
        "reviewed_example",
    }
)
_PRIVATE_KINDS = frozenset(
    {
        "raw_memory",
        "raw_query",
        "gold",
        "image",
        "model_payload",
        "reranker",
        "runtime_answer",
        "compiler_patch",
        "variant_cell_reuse",
        "cost_fact",
        "performance",
        "log",
        "journal",
        "failure",
        "review_payload",
    }
)
_AUDIT_KINDS = frozenset({"reranker_call", "failure", "review_sample", "attempt"})


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {".", ".."} for part in path.parts)
        or any(ord(c) < 32 or ord(c) == 127 for c in name)
        or "\\" in name
        or ":" in name
        or name == "."
    ):
        raise ValueError("noncanonical or unsafe bundle path")
    return name


def _component(name: str) -> str:
    if len(PurePosixPath(_safe_path(name)).parts) != 1:
        raise ValueError("bundle name must be a single path component")
    return name


def _check_path_conflicts(paths: set[str]) -> None:
    all_paths = paths | {_MANIFEST}
    if any(
        parent.as_posix() in all_paths
        for name in all_paths
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    ):
        raise ValueError("artifact file/directory path conflict")


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")


def _open_regular(path: Path) -> BinaryIO:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("artifact must be a regular, non-symlink file")
    return os.fdopen(fd, "rb")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_regular(path) as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PublicReview:
    """A human-approved license/privacy review reference, not a scientific approval."""

    review_id: str
    license_id: str
    privacy_approved: bool

    def __post_init__(self) -> None:
        if not self.review_id or not self.license_id or self.privacy_approved is not True:
            raise ValueError("public artifacts need explicit license and privacy review")


@dataclass(frozen=True)
class LogCutoff:
    byte_count: int
    last_event_id: str

    def __post_init__(self) -> None:
        if type(self.byte_count) is not int or self.byte_count < 0 or not self.last_event_id:
            raise ValueError("logs require an explicit byte and event cutoff")


@dataclass(frozen=True)
class ExportArtifact:
    source: Path
    archive_path: str
    kind: str
    size_bytes: int
    sha256: str
    public_review: PublicReview | None = None
    log_cutoff: LogCutoff | None = None

    def __post_init__(self) -> None:
        _safe_path(self.archive_path)
        _digest(self.sha256)
        if self.kind not in _PUBLIC_KINDS | _PRIVATE_KINDS:
            raise ValueError("unknown artifact kind")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("invalid artifact size")
        if self.kind in {"log", "journal"}:
            if self.log_cutoff is None or self.log_cutoff.byte_count != self.size_bytes:
                raise ValueError("every log/journal needs a matching explicit cutoff")
        elif self.log_cutoff is not None:
            raise ValueError("only log/journal artifacts may export a prefix")
        if (
            self.archive_path == _MANIFEST
            or "terminal_receipt" in self.archive_path
            or self.archive_path.endswith((".tar.zst", ".sha256"))
        ):
            raise ValueError("bundle manifests, receipts and checksums cannot reference themselves")


@dataclass(frozen=True)
class AuditRecord:
    kind: Literal["reranker_call", "failure", "review_sample", "attempt"]
    record_id: str
    artifact_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in _AUDIT_KINDS or not self.record_id or not self.artifact_paths:
            raise ValueError("audit record requires kind, identity and artifact references")
        if len(set(self.artifact_paths)) != len(self.artifact_paths):
            raise ValueError("duplicate audit artifact reference")
        for path in self.artifact_paths:
            _safe_path(path)


@dataclass(frozen=True)
class AuditInventory:
    """Expected IDs must come from all attempts, not only the final attempt.

    Reranker records reference input/output (or failure) payload files; review
    samples reference original payloads. This helper verifies references/hashes,
    not application-specific payload semantics or database inventory truth.
    """

    expected_records: tuple[tuple[str, str], ...] = ()
    records: tuple[AuditRecord, ...] = ()

    def __post_init__(self) -> None:
        keys = tuple((r.kind, r.record_id) for r in self.records)
        if (
            len(set(keys)) != len(keys)
            or len(set(self.expected_records)) != len(self.expected_records)
            or set(keys) != set(self.expected_records)
        ):
            raise ValueError("audit inventory is incomplete or contains duplicate records")


@dataclass(frozen=True)
class ExportPlan:
    bundle_name: str
    visibility: Literal["public", "private"]
    artifacts: tuple[ExportArtifact, ...]
    private_authorized: bool = False
    run_terminal_status: Literal["complete", "incomplete", "failed"] = "incomplete"
    audit: AuditInventory = AuditInventory()
    split_bytes: int | None = None
    compression_level: int = 9
    failure_code: Literal["artifact_export_failed"] | None = None

    def __post_init__(self) -> None:
        _component(self.bundle_name)
        if self.visibility not in {"public", "private"}:
            raise ValueError("visibility must be explicit")
        if self.visibility == "private" and self.private_authorized is not True:
            raise ValueError("private export requires explicit authorization")
        if self.run_terminal_status not in {"complete", "incomplete", "failed"}:
            raise ValueError("invalid run terminal state")
        if self.split_bytes is not None and (
            type(self.split_bytes) is not int or self.split_bytes < 1
        ):
            raise ValueError("split_bytes must be a positive integer")
        if type(self.compression_level) is not int or not 1 <= self.compression_level <= 19:
            raise ValueError("compression level must be in [1, 19]")
        paths = {a.archive_path for a in self.artifacts}
        if self.failure_code is not None and (
            self.failure_code != "artifact_export_failed" or self.run_terminal_status != "failed"
        ):
            raise ValueError("invalid failure capsule code")
        if (not paths and self.failure_code is None) or len(paths) != len(self.artifacts):
            raise ValueError("explicit nonempty artifact list with unique paths is required")
        _check_path_conflicts(paths)
        if self.visibility == "public":
            if self.audit.records:
                raise ValueError("private audit record identities cannot enter public bundles")
            if any(a.kind not in _PUBLIC_KINDS or a.public_review is None for a in self.artifacts):
                raise ValueError(
                    "public export forbids raw datasets/payloads and unreviewed artifacts"
                )
        if any(path not in paths for r in self.audit.records for path in r.artifact_paths):
            raise ValueError("audit record references an artifact absent from the manifest")


@dataclass(frozen=True)
class BundlePart:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    receipt_path: Path
    manifest_sha256: str
    archive_sha256: str
    parts: tuple[BundlePart, ...]


class ExportFailure(RuntimeError):
    def __init__(self, capsule: ExportResult):
        super().__init__("export failed; a verified, sanitized failure capsule is available")
        self.capsule = capsule


class _SplitWriter(io.RawIOBase):
    def __init__(self, root: Path, name: str, limit: int | None):
        super().__init__()
        self.root, self.name, self.limit = root, name, limit
        self.stream: BinaryIO | None = None
        self.parts: list[BundlePart] = []
        self.digest = hashlib.sha256()
        self.part_digest = hashlib.sha256()
        self.size = self.part_size = 0
        self.part_name = ""

    def _close_part(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.parts.append(
                BundlePart(self.part_name, self.part_size, self.part_digest.hexdigest())
            )
            self.stream = None

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        view = memoryview(data)
        total = len(view)
        while view:
            if self.stream is None:
                self.part_name = f"{self.name}.part{len(self.parts) + 1:05d}"
                self.stream = (self.root / self.part_name).open("xb")
                self.part_size = 0
                self.part_digest = hashlib.sha256()
            count = min(len(view), self.limit - self.part_size) if self.limit else len(view)
            piece = view[:count]
            self.stream.write(piece)
            self.digest.update(piece)
            self.part_digest.update(piece)
            self.size += count
            self.part_size += count
            view = view[count:]
            if self.limit and self.part_size == self.limit:
                self._close_part()
        return total

    def flush(self) -> None:
        if self.stream is not None:
            self.stream.flush()

    def finish(self) -> tuple[BundlePart, ...]:
        self._close_part()
        if len(self.parts) == 1:
            part = self.parts[0]
            (self.root / part.name).rename(self.root / self.name)
            self.parts[0] = BundlePart(self.name, part.size_bytes, part.sha256)
        return tuple(self.parts)

    def close(self) -> None:
        self._close_part()
        super().close()


class _HashReader:
    def __init__(self, stream: IO[bytes], size: int):
        self.stream, self.remaining = stream, size
        self.digest = hashlib.sha256()
        self.last_byte = b""

    def read(self, size: int = -1) -> bytes:
        size = min(self.remaining, _CHUNK if size < 0 else size)
        data = self.stream.read(size)
        self.remaining -= len(data)
        self.digest.update(data)
        if data:
            self.last_byte = data[-1:]
        return data


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o600, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _manifest(plan: ExportPlan) -> dict:
    files = []
    for artifact in sorted(plan.artifacts, key=lambda a: a.archive_path):
        files.append(
            {
                "path": artifact.archive_path,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
                "kind": artifact.kind,
                "public_review": asdict(artifact.public_review) if artifact.public_review else None,
                "log_cutoff": asdict(artifact.log_cutoff) if artifact.log_cutoff else None,
            }
        )
    body = {
        "version": _VERSION,
        "bundle_name": plan.bundle_name,
        "visibility": plan.visibility,
        "run_terminal_status": plan.run_terminal_status,
        "files": files,
        "audit": {
            "expected_records": sorted(plan.audit.expected_records),
            "records": [
                asdict(r) for r in sorted(plan.audit.records, key=lambda r: (r.kind, r.record_id))
            ],
        },
        "compression": {
            "level": plan.compression_level,
            "threads": 0,
            "zstandard_version": zstandard.__version__,
            "zstd_version": list(zstandard.ZSTD_VERSION),
        },
        "split_bytes": plan.split_bytes,
        "failure_code": plan.failure_code,
    }
    return {**body, "manifest_id": _hash(_json_bytes(body))}


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _checksum(root: Path, name: str, digest: str) -> None:
    _write_bytes(root / f"{name}.sha256", f"{digest}  {name}\n".encode())


def _pack(plan: ExportPlan, output_dir: Path) -> ExportResult:
    manifest = _manifest(plan)
    manifest_bytes = _json_bytes(manifest)
    if len(manifest_bytes) > _MAX_MANIFEST:
        raise ValueError("artifact manifest exceeds the bounded metadata size")
    name = f"{plan.bundle_name}.{plan.visibility}.tar.zst"
    writer = _SplitWriter(output_dir, name, plan.split_bytes)
    try:
        compressor = zstandard.ZstdCompressor(
            level=plan.compression_level, threads=0, write_checksum=True, write_content_size=False
        )
        with compressor.stream_writer(cast(IO[bytes], writer), closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                archive.addfile(
                    _tar_info(_MANIFEST, len(manifest_bytes)), io.BytesIO(manifest_bytes)
                )
                for artifact in sorted(plan.artifacts, key=lambda a: a.archive_path):
                    if artifact.source.resolve().is_relative_to(output_dir.resolve()):
                        raise ValueError("bundle cannot include its own output directory")
                    with _open_regular(artifact.source) as source:
                        size = os.fstat(source.fileno()).st_size
                        if size < artifact.size_bytes or (
                            artifact.log_cutoff is None and size != artifact.size_bytes
                        ):
                            raise ValueError("artifact size differs from manifest")
                        reader = _HashReader(source, artifact.size_bytes)
                        archive.addfile(
                            _tar_info(artifact.archive_path, artifact.size_bytes), reader
                        )
                        if reader.remaining or reader.digest.hexdigest() != artifact.sha256:
                            raise ValueError("artifact checksum differs from manifest")
                        if (
                            artifact.log_cutoff
                            and artifact.size_bytes
                            and reader.last_byte != b"\n"
                        ):
                            raise ValueError("journal cutoff must end on a complete line")
                        if (
                            artifact.log_cutoff is None
                            and os.fstat(source.fileno()).st_size != size
                        ):
                            raise ValueError("artifact changed while exporting")
        parts = writer.finish()
    finally:
        writer.close()
    for part in parts:
        _checksum(output_dir, part.name, part.sha256)
    if len(parts) > 1:
        _checksum(output_dir, name, writer.digest.hexdigest())
    receipt = {
        "version": _VERSION,
        "terminal_state": "bundle_verified",
        "scientific_approval": False,
        "visibility": plan.visibility,
        "run_terminal_status": plan.run_terminal_status,
        "archive_name": name,
        "archive_sha256": writer.digest.hexdigest(),
        "archive_size_bytes": writer.size,
        "parts": [asdict(part) for part in parts],
        "manifest_sha256": _hash(manifest_bytes),
        "manifest_id": manifest["manifest_id"],
        "log_cutoffs": [
            {"path": a.archive_path, **asdict(a.log_cutoff)}
            for a in sorted(plan.artifacts, key=lambda a: a.archive_path)
            if a.log_cutoff
        ],
    }
    # Verify all archived bytes before issuing the detached terminal receipt.
    _verify_payload(output_dir, receipt)
    receipt_path = output_dir / "terminal_receipt.json"
    payload = _json_bytes(receipt)
    _write_bytes(receipt_path, payload)
    _checksum(output_dir, receipt_path.name, _hash(payload))
    return ExportResult(
        output_dir, receipt_path, _hash(manifest_bytes), writer.digest.hexdigest(), parts
    )


def export_bundle(plan: ExportPlan, output_dir: Path) -> ExportResult:
    """Write to a new directory; failure leaves a portable sanitized capsule.

    Reassembling split parts in receipt order yields one .tar.zst stream. Input
    journal prefixes never include this export's receipt or final archive hash.
    Sources are hashed during streaming; no raw dataset is loaded into memory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        return _pack(plan, output_dir)
    except Exception as error:
        failure_dir = output_dir / "failure_capsule"
        failure_dir.mkdir(mode=0o700)
        source = output_dir / "export_failure.json"
        payload = _json_bytes(
            {
                "failure_code": "artifact_export_failed",
                "error_type": type(error).__name__,
                "scientific_approval": False,
            }
        )
        _write_bytes(source, payload)
        # No input payloads or exception messages are copied into the capsule.
        capsule_plan = ExportPlan(
            "export-failure",
            plan.visibility,
            (),
            private_authorized=plan.private_authorized,
            run_terminal_status="failed",
            failure_code="artifact_export_failed",
        )
        capsule = _pack(capsule_plan, failure_dir)
        raise ExportFailure(capsule) from error


class _JoinedReader(io.RawIOBase):
    def __init__(self, paths: tuple[Path, ...]):
        super().__init__()
        self.paths = iter(paths)
        self.stream: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        size = min(_CHUNK, _CHUNK if size < 0 else size)
        if size == 0:
            return b""
        while True:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    return b""
                self.stream = _open_regular(path)
            data = self.stream.read(size)
            if data:
                return data
            self.stream.close()
            self.stream = None

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
        super().close()


def _read_small(path: Path) -> bytes:
    with _open_regular(path) as stream:
        result = stream.read(_MAX_MANIFEST + 1)
    if len(result) > _MAX_MANIFEST:
        raise ValueError("metadata exceeds size limit")
    return result


def _check_sidecar(root: Path, name: str, digest: str) -> None:
    if _read_small(root / f"{name}.sha256") != f"{digest}  {name}\n".encode():
        raise ValueError("checksum sidecar mismatch")


def _validate_manifest(manifest: dict, receipt: dict) -> dict[str, dict]:
    if manifest.get("version") != _VERSION or manifest.get("visibility") != receipt["visibility"]:
        raise ValueError("invalid manifest version or visibility")
    body = {k: v for k, v in manifest.items() if k != "manifest_id"}
    if _hash(_json_bytes(body)) != manifest.get("manifest_id") or (
        manifest.get("manifest_id") != receipt["manifest_id"]
    ):
        raise ValueError("manifest identity mismatch")
    if manifest.get("run_terminal_status") != receipt["run_terminal_status"]:
        raise ValueError("run terminal status mismatch")
    files = {}
    for entry in manifest["files"]:
        artifact = ExportArtifact(
            Path("unused"),
            entry["path"],
            entry["kind"],
            entry["size_bytes"],
            entry["sha256"],
            PublicReview(**entry["public_review"]) if entry["public_review"] else None,
            LogCutoff(**entry["log_cutoff"]) if entry["log_cutoff"] else None,
        )
        if artifact.archive_path in files:
            raise ValueError("duplicate manifest member")
        if receipt["visibility"] == "public" and (
            artifact.kind not in _PUBLIC_KINDS or artifact.public_review is None
        ):
            raise ValueError("private/unreviewed artifact in public bundle")
        files[artifact.archive_path] = entry
    _check_path_conflicts(set(files))
    audit = manifest["audit"]
    inventory = AuditInventory(
        tuple(tuple(item) for item in audit["expected_records"]),
        tuple(
            AuditRecord(r["kind"], r["record_id"], tuple(r["artifact_paths"]))
            for r in audit["records"]
        ),
    )
    if receipt["visibility"] == "public" and inventory.records:
        raise ValueError("private audit inventory in public bundle")
    if any(path not in files for r in inventory.records for path in r.artifact_paths):
        raise ValueError("unresolved audit artifact reference")
    cutoffs = [
        {"path": name, **entry["log_cutoff"]}
        for name, entry in sorted(files.items())
        if entry["log_cutoff"]
    ]
    if cutoffs != receipt["log_cutoffs"]:
        raise ValueError("receipt journal cutoff mismatch")
    return files


def _verify_payload(root: Path, receipt: dict) -> dict:
    if (
        receipt.get("version") != _VERSION
        or receipt.get("terminal_state") != "bundle_verified"
        or receipt.get("scientific_approval") is not False
        or receipt.get("visibility") not in {"private", "public"}
    ):
        raise ValueError("invalid terminal receipt")
    name = _component(receipt["archive_name"])
    parts = tuple(BundlePart(**part) for part in receipt["parts"])
    expected_names = (
        [name] if len(parts) == 1 else [f"{name}.part{i + 1:05d}" for i in range(len(parts))]
    )
    if not parts or [part.name for part in parts] != expected_names:
        raise ValueError("missing, reordered or noncanonical archive parts")
    digest = hashlib.sha256()
    size = 0
    for part in parts:
        _component(part.name)
        _digest(part.sha256)
        part_digest = hashlib.sha256()
        part_size = 0
        with _open_regular(root / part.name) as stream:
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
                part_digest.update(chunk)
                part_size += len(chunk)
        if part_size != part.size_bytes or part_digest.hexdigest() != part.sha256:
            raise ValueError("archive part checksum/size mismatch")
        _check_sidecar(root, part.name, part.sha256)
        size += part_size
    if digest.hexdigest() != receipt["archive_sha256"] or size != receipt["archive_size_bytes"]:
        raise ValueError("archive checksum/size mismatch")
    _check_sidecar(root, name, receipt["archive_sha256"])
    joined = _JoinedReader(tuple(root / part.name for part in parts))
    try:
        with zstandard.ZstdDecompressor().stream_reader(
            cast(IO[bytes], joined), closefd=False
        ) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|", bufsize=512) as archive:
                first = archive.next()
                if (
                    first is None
                    or first.name != _MANIFEST
                    or not first.isfile()
                    or first.size > _MAX_MANIFEST
                ):
                    raise ValueError("archive must begin with bounded artifact manifest")
                manifest_stream = archive.extractfile(first)
                assert manifest_stream is not None
                payload = manifest_stream.read()
                if _hash(payload) != receipt["manifest_sha256"]:
                    raise ValueError("artifact manifest hash mismatch")
                manifest = json.loads(payload)
                files = _validate_manifest(manifest, receipt)
                seen = set()
                # archive iteration returns first again; use next() explicitly.
                while (member := archive.next()) is not None:
                    _safe_path(member.name)
                    if (
                        not member.isfile()
                        or member.name in seen
                        or member.name not in files
                        or member.uid != 0
                        or member.gid != 0
                        or member.mtime != 0
                        or member.mode != 0o600
                        or member.uname
                        or member.gname
                    ):
                        raise ValueError("unsafe, duplicate, unlisted or noncanonical tar member")
                    entry = files[member.name]
                    if member.size != entry["size_bytes"]:
                        raise ValueError("artifact size mismatch")
                    artifact_stream = archive.extractfile(member)
                    assert artifact_stream is not None
                    reader = _HashReader(artifact_stream, member.size)
                    while reader.read(_CHUNK):
                        pass
                    if reader.remaining or reader.digest.hexdigest() != entry["sha256"]:
                        raise ValueError("artifact checksum mismatch")
                    if entry["log_cutoff"] and member.size and reader.last_byte != b"\n":
                        raise ValueError("partial journal record")
                    seen.add(member.name)
                if seen != set(files):
                    raise ValueError("artifact manifest members are missing")
            while chunk := decompressed.read(_CHUNK):
                if chunk.strip(b"\0"):
                    raise ValueError("unexpected payload after tar terminator")
    finally:
        joined.close()
    return manifest


def verify_bundle(receipt_path: Path, *, expected_receipt_sha256: str | None = None) -> dict:
    """Stream-verify parts, envelope, manifest and every artifact without extraction."""
    receipt_path = Path(receipt_path)
    payload = _read_small(receipt_path)
    digest = _hash(payload)
    if expected_receipt_sha256 is not None:
        _digest(expected_receipt_sha256)
        if digest != expected_receipt_sha256:
            raise ValueError("pinned receipt checksum mismatch")
    _check_sidecar(receipt_path.parent, receipt_path.name, digest)
    return _verify_payload(receipt_path.parent, json.loads(payload))
