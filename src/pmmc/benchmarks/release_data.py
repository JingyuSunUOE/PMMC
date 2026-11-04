"""Anonymous, locked source acquisition. No model, provider, DB or Settings import."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

LOCK = Path("configs/release/benchmarks.lock.json")
CHUNK = 1024 * 1024


def _safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != name
        or ".." in path.parts
        or "\\" in name
        or any(part.startswith(".") for part in path.parts)
    ):
        raise ValueError("unsafe benchmark source path")
    return name


def _local(path: Path) -> Path:
    if path.resolve() != path:
        raise ValueError("symlinks are not allowed in benchmark staging")
    return path


@dataclass(frozen=True)
class File:
    url: str
    target: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative(self.target)
        if (
            re.fullmatch(r"[a-f0-9]{64}", self.sha256) is None
            or self.size <= 0
            or not self.url.startswith(
                ("https://huggingface.co/datasets/", "https://raw.githubusercontent.com/")
            )
        ):
            raise ValueError("invalid benchmark lock entry")


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tree_digest(rows: Iterable[tuple[str, int, str]]) -> str:
    return hashlib.sha256(json.dumps(sorted(rows), separators=(",", ":")).encode()).hexdigest()


def _matches(path: Path, spec: dict) -> bool:
    if not path.is_dir():
        return False
    rows = []
    ordinary = {f["target"] for f in spec["files"] if f["target"] != "archive"}
    ordinary.update(
        p.replace("data/dialog/", "data/dialogs/", 1) for p in spec.get("regular_blob_sha256", {})
    )
    for file in path.rglob("*"):
        if file.is_symlink():
            raise ValueError("unexpected benchmark symlink")
        if file.is_dir() or file.name in {".gitkeep", ".DS_Store"}:
            continue
        if not file.is_file():
            raise ValueError("unexpected benchmark file type")
        relative = file.relative_to(path).as_posix()
        if (
            (relative not in ordinary and not relative.startswith("images/"))
            or any(part.startswith(".") for part in PurePosixPath(relative).parts)
            or relative.startswith("images/")
            and file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}
        ):
            return False  # Don't open unknown files in a manual restore.
        rows.append((relative, file.stat().st_size, _hash(file)))
    return len(rows) == spec["file_count"] and _tree_digest(rows) == spec["materialized_sha256"]


def _gallery_files(spec: dict, metadata: dict) -> list[File]:
    rev = spec["hf_revision"]
    if metadata["sha"] != rev:
        raise ValueError("benchmark metadata revision mismatch")
    url = f"https://huggingface.co/datasets/{spec['hf_dataset']}/resolve/{rev}/"
    files = [File(**row) for row in spec["files"]]
    for row in metadata["siblings"]:
        source = row["rfilename"]
        if source.startswith("data/dialog/") and source.endswith(".json"):
            target = "data/dialogs/" + source.removeprefix("data/dialog/")
        elif source.startswith("data/image/") and source.endswith((".png", ".jpg")):
            target = "images/" + source.removeprefix("data/image/")
        else:
            continue
        digest = spec["regular_blob_sha256"].get(source) or row["lfs"]["sha256"]
        files.append(File(url + quote(source), target, row["size"], digest))
    # Verify upstream LFS identities against the shipped lock BEFORE fetching any bytes.
    if (
        len(set(f.target for f in files)) != spec["file_count"]
        or _tree_digest((f.target, f.size, f.sha256) for f in files) != spec["materialized_sha256"]
    ):
        raise ValueError("benchmark remote file inventory changed")
    return files


def _sources(name: str, spec: dict, stage: Path) -> list[File]:
    if name == "memlens":
        return [File(**row) for row in spec["files"]]
    metadata = _local(stage / "metadata.json")
    if metadata.is_file():
        try:
            return _gallery_files(spec, json.loads(metadata.read_bytes()))
        except (KeyError, TypeError, ValueError):
            pass
    url = f"https://huggingface.co/api/datasets/{spec['hf_dataset']}/revision/{spec['hf_revision']}?blobs=true"
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    files = _gallery_files(spec, json.loads(raw))
    pending = _local(metadata.with_suffix(".tmp"))
    pending.write_bytes(raw)
    pending.replace(metadata)
    return files


class _Progress:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.lock = threading.Lock()
        self.downloaded = self.finished = self.total = 0
        self.last = 0.0

    def update(self, phase: str, *, size: int = 0, files: int = 0, force: bool = False):
        with self.lock:
            self.downloaded += size
            self.finished += files
            if force or time.monotonic() - self.last >= 5:
                self.last = time.monotonic()
                print(
                    json.dumps(
                        dict(
                            protocol="release-assets-v2",
                            dataset=self.dataset,
                            phase=phase,
                            files=self.finished,
                            total=self.total,
                            bytes_received=self.downloaded,
                        )
                    ),
                    file=sys.stderr,
                    flush=True,
                )


def _fetch(file: File, cache: Path, progress: _Progress) -> Path:
    path = _local(cache / file.sha256)
    if path.is_file() and path.stat().st_size == file.size and _hash(path) == file.sha256:
        return path
    part = _local(path.with_suffix(".part"))
    for attempt in range(4):
        try:
            digest = hashlib.sha256()
            count = 0
            with urlopen(file.url, timeout=90) as response, part.open("wb") as stream:
                while chunk := response.read(CHUNK):
                    count += len(chunk)
                    if count > file.size:
                        raise ValueError("benchmark download exceeds locked size")
                    stream.write(chunk)
                    digest.update(chunk)
                    progress.update("download", size=len(chunk))
            if count != file.size or digest.hexdigest() != file.sha256:
                raise ValueError("benchmark download checksum mismatch")
            part.replace(path)
            return path
        except (OSError, ValueError) as exc:
            if attempt == 3:
                raise ValueError("benchmark download failed verification") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _unpack(path: Path, root: Path) -> None:
    # Never extract code, links, devices, absolute paths or tar-supplied permissions.
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            _safe_relative(member.name)
            if member.isdir():
                continue
            if (
                not member.isfile()
                or not member.name.startswith("release_images/")
                or not member.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ):
                raise ValueError("unexpected benchmark archive member")
            dest = root / "images" / _safe_relative(member.name.removeprefix("release_images/"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as out:
                shutil.copyfileobj(archive.extractfile(member), out, CHUNK)


def _publish(candidate: Path, destination: Path) -> None:
    if not destination.exists():
        candidate.rename(destination)
        return
    if not destination.is_dir():
        raise ValueError("benchmark destination is not a directory")
    # Readers must see either complete directory, including when repairing a
    # partial manual restore. Keep the displaced data in private staging.
    lib = ctypes.CDLL(None, use_errno=True)
    a, b = os.fsencode(candidate), os.fsencode(destination)
    if sys.platform == "linux":
        status = lib.renameat2(-100, a, -100, b, 2)  # RENAME_EXCHANGE
    elif sys.platform == "darwin":
        status = lib.renamex_np(a, b, 2)  # RENAME_SWAP
    else:
        raise ValueError("atomic directory exchange unavailable")
    if status:
        raise OSError(ctypes.get_errno(), "atomic benchmark publish failed")


def prepare_data(staging_root: Path, benchmarks: Iterable[str], *, lock_path: Path = LOCK) -> dict:
    root = _local(staging_root.absolute())
    raw_lock = lock_path.read_bytes()
    lock = json.loads(raw_lock)
    names = list(benchmarks)
    if (
        lock["protocol"] != "pmmc-public-data-lock-v1"
        or not names
        or len(names) != len(set(names))
        or not set(names) <= {"memlens", "memgallery"}
    ):
        raise ValueError("invalid benchmark source selection")
    results = {}
    for name in names:
        spec = lock["datasets"][name]
        stage = _local(root / ".release-data-stage" / name)
        cache = _local(stage / "blobs")
        cache.mkdir(parents=True, exist_ok=True)
        dest = _local(root / "benchmarks" / name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        progress = _Progress(name)
        with _local(stage / "prepare.lock").open("a+b") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            progress.update("check", force=True)
            state = "verified"
            if not _matches(dest, spec):
                files = _sources(name, spec, stage)
                unique = {f.sha256: f for f in files}
                progress.total = len(unique)
                with ThreadPoolExecutor(max_workers=12) as pool:
                    tasks = [pool.submit(_fetch, f, cache, progress) for f in unique.values()]
                    for task in as_completed(tasks):
                        task.result()
                        progress.update("download", files=1)
                progress.update("materialize", force=True)
                build = Path(tempfile.mkdtemp(prefix="candidate-", dir=stage))
                try:
                    for file in files:
                        blob = cache / file.sha256
                        if file.target == "archive":
                            _unpack(blob, build)
                        else:
                            target = build / file.target
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(blob, target)
                    if not _matches(build, spec):
                        raise ValueError("benchmark materialized tree checksum mismatch")
                    state = "repaired" if dest.exists() else "installed"
                    _publish(build, dest)
                except BaseException:
                    shutil.rmtree(build)
                    raise
            progress.update(state, force=True)
            results[name] = dict(
                state=state,
                files=spec["file_count"],
                materialized_sha256=spec["materialized_sha256"],
            )
    return dict(
        protocol=lock["protocol"],
        lock_sha256=hashlib.sha256(raw_lock).hexdigest(),
        benchmarks=results,
    )
