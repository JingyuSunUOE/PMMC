"""Durable local CAS and strictly owned temporary-file cleanup for Step 8."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def durable_write(path: Path, data: bytes, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("artifact destination cannot be a symlink")
    fd, name = tempfile.mkstemp(prefix=".step8-write-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError("immutable artifact identity conflict") from None
        else:
            os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    durable_write(path, json_bytes(value), immutable=immutable)


@dataclass(frozen=True)
class _VerifiedObject:
    signature: tuple[int, int, int, int, int, int]
    data: bytes | None


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("artifact must be a regular file")
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class SnapshotArtifactReader:
    """Bounded read-only CAS reuse for one analysis or approval invocation.

    Cached bytes are immutable; JSON is decoded afresh so callers never share
    mutable objects. Oversized objects retain only verification metadata. A
    changed stat signature or missing path invalidates either kind of entry.
    """

    def __init__(
        self,
        store: CampaignArtifactStore,
        *,
        max_cached_bytes: int = 64 * 1024 * 1024,
        max_cached_entries: int = 4096,
    ) -> None:
        if any(type(n) is not int or n < 0 for n in (max_cached_bytes, max_cached_entries)):
            raise ValueError("artifact reader cache limits must be nonnegative integers")
        self._store = store
        self._max_bytes = max_cached_bytes
        self._max_entries = max_cached_entries
        self._entries: OrderedDict[str, _VerifiedObject] = OrderedDict()
        self._cached_bytes = 0

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    @property
    def cached_entries(self) -> int:
        return len(self._entries)

    def _discard(self, digest: str) -> None:
        entry = self._entries.pop(digest, None)
        if entry is not None and entry.data is not None:
            self._cached_bytes -= len(entry.data)

    def _remember(self, digest: str, entry: _VerifiedObject) -> None:
        if not self._max_entries:
            return
        data = entry.data
        if data is not None and len(data) > self._max_bytes:
            data = None
        size = len(data) if data is not None else 0
        while self._entries and (
            len(self._entries) >= self._max_entries or self._cached_bytes + size > self._max_bytes
        ):
            self._discard(next(iter(self._entries)))
        self._entries[digest] = _VerifiedObject(entry.signature, data)
        self._cached_bytes += size

    def _read_object(
        self,
        path: Path,
        digest: str,
        signature: tuple[int, int, int, int, int, int],
        *,
        require_bytes: bool,
    ) -> _VerifiedObject:
        retain = require_bytes or signature[3] <= self._max_bytes
        chunks: list[bytes] = []
        hasher = hashlib.sha256()
        # Do not follow a replaced symlink or block on a substituted FIFO.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if _stat_signature(os.fstat(stream.fileno())) != signature:
                raise ValueError("artifact changed during verification")
            remaining = signature[3]
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                hasher.update(chunk)
                if retain:
                    chunks.append(chunk)
            if (
                remaining
                or stream.read(1)
                or hasher.hexdigest() != digest
                or _stat_signature(os.fstat(stream.fileno())) != signature
                or _stat_signature(path.lstat()) != signature
            ):
                raise ValueError("artifact checksum or presence verification failed")
        return _VerifiedObject(signature, b"".join(chunks) if retain else None)

    def _verified(self, digest: str, *, require_bytes: bool) -> tuple[Path, _VerifiedObject]:
        try:
            path = self._store.object_path(digest)
            signature = _stat_signature(path.lstat())
            entry = self._entries.get(digest)
            if entry is not None and entry.signature == signature:
                if not require_bytes or entry.data is not None:
                    self._entries.move_to_end(digest)
                    return path, entry
            self._discard(digest)
            entry = self._read_object(path, digest, signature, require_bytes=require_bytes)
            self._remember(digest, entry)
            return path, entry
        except (OSError, ValueError) as exc:
            self._discard(digest)
            if isinstance(exc, OSError):
                raise ValueError("artifact checksum or presence verification failed") from exc
            raise

    def verify(self, digest: str) -> Path:
        """Compatibility for proof validators; use read_bytes to consume bytes."""
        return self._verified(digest, require_bytes=False)[0]

    def verified_size(self, digest: str) -> int:
        return self._verified(digest, require_bytes=False)[1].signature[3]

    def read_bytes(self, digest: str) -> bytes:
        data = self._verified(digest, require_bytes=True)[1].data
        assert data is not None
        return data

    def load_json(self, digest: str) -> Any:
        return json.loads(self.read_bytes(digest))


class CampaignArtifactStore:
    def __init__(self, output_root: Path) -> None:
        self.root = output_root.resolve()
        self.store = self.root / "_store" / "sha256"
        self.cache = self.root / "_cache"

    def object_path(self, digest: str) -> Path:
        if not _SHA256.fullmatch(digest):
            raise ValueError("invalid content-addressed artifact digest")
        path = self.store / digest[:2] / digest
        if not path.resolve().is_relative_to(self.store.resolve()):
            raise ValueError("CAS object escaped store")
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("CAS cannot follow symlinks")
        return path

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        with self.collection_lock(exclusive=False):
            path = self.object_path(digest)
            durable_write(path, data, immutable=True)
            # A reused object is in flight until its immutable owner receipt is saved.
            os.utime(path, None)
        return digest

    @contextmanager
    def collection_lock(self, *, exclusive: bool):
        directory = self.root / "_store"
        directory.mkdir(parents=True, exist_ok=True)
        # NFS emulates flock with record locks: shared locks need a readable FD.
        with (directory / ".gc.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield

    def preflight_response_journal(self) -> dict[str, object]:
        """Exercise fresh journal writes; retain one tiny, reusable CAS sentinel."""
        payload = {"protocol": "step8-journal-storage-probe-v1", "provider_calls": 0}
        request_id = "storage-probe-" + uuid4().hex
        request_hash = hashlib.sha256(request_id.encode()).hexdigest()
        identity = {"request_id": request_id, "request_hash": request_hash}
        primary: BaseException | None = None
        try:
            response_hash = self.journal_response(
                **identity, response=payload, attempt_id="storage-probe-v1"
            )
            if self.restored_response(**identity) != payload:
                raise ValueError("response journal storage preflight readback mismatch")
        except BaseException as exc:
            primary = exc
            raise
        finally:
            # Only this invocation's synthetic index, never a provider journal or lock.
            try:
                (self.root / "_store/responses" / f"{request_hash}.json").unlink(missing_ok=True)
            except OSError:
                if primary is None:
                    raise
                primary.add_note("synthetic journal probe cleanup failed")
        return {
            "state": "ready", "protocol": payload["protocol"],
            "response_sha256": response_hash, "provider_calls": 0,
            "scope": "synthetic_journal_roundtrip_not_provider_recovery",
        }

    def put_json(self, value: Any) -> str:
        return self.put(json_bytes(value))

    def verify(self, digest: str) -> Path:
        path = self.object_path(digest)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("artifact checksum or presence verification failed")
        return path

    def load_json(self, digest: str) -> Any:
        return json.loads(self.verify(digest).read_bytes())

    def journal_response(
        self,
        *,
        request_id: str,
        request_hash: str,
        response: Mapping[str, Any],
        attempt_id: str,
    ) -> str:
        """Write response before DB completion; retries discover it by request hash."""
        if not _SHA256.fullmatch(request_hash) or not _SAFE_ID.fullmatch(attempt_id):
            raise ValueError("invalid response journal identity")
        journal = self.root / "_store" / "responses" / f"{request_hash}.json"
        if journal.is_file():
            existing = json.loads(journal.read_bytes())
            if existing["request_id"] != request_id or existing["request_hash"] != request_hash:
                raise ValueError("response journal request identity mismatch")
            if self.load_json(existing["response_sha256"]) != dict(response):
                raise ValueError("conflicting successful provider responses")
            return str(existing["response_sha256"])
        response_hash = self.put_json(dict(response))
        write_json(
            journal,
            {
                "protocol": "provider-response-journal-v1",
                "request_id": request_id,
                "request_hash": request_hash,
                "response_sha256": response_hash,
                "first_attempt_id": attempt_id,
            },
            immutable=True,
        )
        return response_hash

    def restored_response(self, *, request_id: str, request_hash: str) -> dict[str, Any] | None:
        if not _SHA256.fullmatch(request_hash):
            raise ValueError("invalid response journal request digest")
        journal = self.root / "_store" / "responses" / f"{request_hash}.json"
        if not journal.is_file():
            return None
        data = json.loads(journal.read_bytes())
        if data.get("request_id") != request_id or data.get("request_hash") != request_hash:
            raise ValueError("response journal request identity mismatch")
        value = self.load_json(str(data["response_sha256"]))
        if not isinstance(value, dict):
            raise ValueError("provider response journal must contain an object")
        return value

    def attempt_cache(self, attempt_id: str) -> Path:
        if not _SAFE_ID.fullmatch(attempt_id):
            raise ValueError("invalid attempt ID")
        path = self.cache / attempt_id
        if path.is_symlink() or not path.resolve().is_relative_to(self.cache.resolve()):
            raise ValueError("attempt cache escaped ownership root")
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "owner.json", {"attempt_id": attempt_id}, immutable=True)
        return path

    def clean_attempt(self, attempt_id: str, *, durable_receipt: Path) -> dict[str, Any]:
        """Never delete result, checkpoint, successful response, or model-cache trees."""
        if not _SAFE_ID.fullmatch(attempt_id):
            raise ValueError("invalid attempt ID")
        receipt = json.loads(durable_receipt.read_bytes())
        if receipt.get("attempt_id") != attempt_id or receipt.get("state") not in {
            "failed",
            "completed",
            "budget_exhausted",
            "interrupted",
        }:
            raise ValueError("cleanup requires a durable terminal attempt receipt")
        path = self.cache / attempt_id
        if not path.exists():
            return {"removed": False, "reason": "already_clean"}
        if path.is_symlink() or not path.resolve().is_relative_to(self.cache.resolve()):
            raise ValueError("cleanup cannot follow an escaped cache root")
        owner = json.loads((path / "owner.json").read_bytes())
        if owner != {"attempt_id": attempt_id}:
            raise ValueError("cleanup ownership mismatch")
        shutil.rmtree(path)
        return {"removed": True, "scope": "owned_attempt_cache", "retained_scientific_facts": True}

    def gc(
        self,
        *,
        referenced_hashes: Iterable[str],
        reference_scan_complete: bool,
        apply: bool = False,
        grace_seconds: int = 86400,
        now: float | None = None,
    ) -> dict[str, Any]:
        if not reference_scan_complete:
            raise ValueError("CAS collection requires a complete authoritative reference scan")
        if grace_seconds < 86400:
            raise ValueError("CAS collection requires at least 24 hours grace")
        retained = set(referenced_hashes)
        if any(not _SHA256.fullmatch(value) for value in retained):
            raise ValueError("invalid retained artifact reference")
        # Response and trace journals are roots before their authoritative DB commit.
        journal_roots = (("responses", "response_sha256"), ("model-traces", "trace_sha256"),
                         ("round-barriers", "barrier_sha256"))
        for directory, field in journal_roots:
            root = self.root / "_store" / directory
            if root.is_symlink():
                raise ValueError("recovery journal directory cannot be a symlink")
            for journal in sorted(root.glob("*.json")):
                if journal.is_symlink():
                    raise ValueError("recovery journal cannot be a symlink")
                receipt = json.loads(journal.read_bytes())
                digest = receipt.get(field) if isinstance(receipt, dict) else None
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise ValueError("recovery journal reference is invalid")
                retained.add(digest)
        for digest in retained:
            self.verify(digest)
        cutoff = (time.time() if now is None else now) - grace_seconds
        candidates: list[Path] = []
        for path in sorted(self.store.glob("*/*")):
            if path.is_symlink() or not _SHA256.fullmatch(path.name):
                raise ValueError("unexpected CAS entry; refusing collection")
            if path.is_file() and path.name not in retained and path.stat().st_mtime < cutoff:
                candidates.append(path)
        reclaimed_bytes = sum(path.stat().st_size for path in candidates)
        if apply:
            for path in candidates:
                path.unlink()
        return {
            "dry_run": not apply,
            "unreferenced_object_count": len(candidates),
            "reclaimable_bytes": reclaimed_bytes,
            "reference_count": len(retained),
        }
