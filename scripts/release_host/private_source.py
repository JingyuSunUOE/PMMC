"""Verify the original public tarball as data, never execute an archived verifier."""

import gzip
import io
import re
import tarfile
import zlib
from pathlib import PurePosixPath

if __package__ == "release_host":
    from release_source import public_name
else:
    from ..release_source import public_name

from .config import Error, check
from .private_io import sha256

SOURCE_LIMIT = 24 << 20
EXPANDED_LIMIT = 64 << 20
MAX_MEMBERS = 4096
IMAGE_LOCK = "deploy/release/images.lock.json"


def check_headers(raw):
    # Verify the producer's physical USTAR records before tarfile can interpret
    # GNU/PAX payloads. TarInfo subclass hooks vary between Python patch versions.
    offset = 0
    for _ in range(MAX_MEMBERS + 1):
        block = raw[offset : offset + tarfile.BLOCKSIZE]
        if block == bytes(tarfile.BLOCKSIZE):
            return
        member = tarfile.TarInfo.frombuf(block, "ascii", "strict")
        check(
            member.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE)
            and 0 <= member.size <= SOURCE_LIMIT
            and not member.linkname,
            "unsupported public archive record",
        )
        offset += tarfile.BLOCKSIZE * (
            1 + (member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
        )
    raise Error("oversized source archive")


def safe_name(name):
    parts = name.split("/")
    check(
        len(name) <= 240
        and len(parts) <= 24
        and re.fullmatch(r"[a-zA-Z0-9_./-]+", name)
        and all(
            p not in ("", ".", "..", ".git") and not p.lower().startswith(".env") for p in parts
        ),
        "unsafe public archive path",
    )
    return name


def source_files(archive, expected_sha256):
    check(
        sha256(archive) == expected_sha256 and len(archive) <= SOURCE_LIMIT,
        "public source checksum or size mismatch",
    )
    try:
        if archive.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(archive)) as packed:
                raw = packed.read(EXPANDED_LIMIT + 1)
        else:
            raw = archive
        check(len(raw) <= EXPANDED_LIMIT, "expanded source exceeds size limit")
        files, dirs, seen = {}, set(), set()
        check_headers(raw)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            for i, member in enumerate(tar):
                check(i < MAX_MEMBERS, "oversized source archive")
                if member.name in (".", "./") and member.isdir():
                    name = ""
                else:
                    name = safe_name(member.name.removeprefix("./"))
                check(name not in seen, "duplicate source archive member")
                seen.add(name)
                check(member.mode & 0o7000 == 0, "unsafe archive mode")
                check(
                    not member.linkname and member.sparse is None,
                    "unsupported public archive record",
                )
                if member.isdir():
                    check(member.size == 0, "directory with unexpected content")
                    dirs.add(name)
                    continue
                # Never let tarfile follow a link or return a special-file stream.
                check(
                    member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    and 0 <= member.size <= SOURCE_LIMIT,
                    "unsupported public archive record",
                )
                stream = tar.extractfile(member)
                check(stream is not None, "missing regular archive stream")
                with stream:
                    data = stream.read()
                check(
                    len(data) == member.size
                    and bool(member.mode & 0o100) == data.startswith(b"#!"),
                    "invalid public file content or executable bit",
                )
                files[name] = data
            # tarfile otherwise accepts an appended second archive or a corrupt final header.
            tail = raw[tar.offset :]
            check(len(tail) >= 1024 and not tail.strip(b"\0"), "invalid public archive trailer")
    except (
        tarfile.TarError,
        zlib.error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        OverflowError,
        RecursionError,
    ):
        raise Error("invalid public source archive") from None
    manifests = [n for n in files if PurePosixPath(n).name == "SOURCE.sha256"]
    check(len(manifests) == 1, "expected one public SOURCE.sha256")
    prefix = manifests[0].removesuffix("SOURCE.sha256")
    check(not prefix or prefix.count("/") == 1, "unexpected public source root")
    required_dirs = {str(p) for n in files for p in PurePosixPath(n).parents} - {"."}
    check(
        all(n.startswith(prefix) for n in files)
        and dirs - {""} <= required_dirs
        and not (required_dirs & files.keys()),
        "unexpected path outside source tree or file/directory conflict",
    )
    paths = set(files) | required_dirs
    check(len({n.casefold() for n in paths}) == len(paths), "case-colliding source paths")
    files = {n.removeprefix(prefix): data for n, data in files.items()}
    try:
        for name in files:
            public_name(name)
    except ValueError:
        raise Error("archive contains a non-public source path") from None
    try:
        manifest = files["SOURCE.sha256"].decode("ascii").splitlines()
    except UnicodeError:
        raise Error("invalid SOURCE.sha256") from None
    expected = {}
    for line in manifest:
        m = re.fullmatch(r"([0-9a-f]{64})  ([a-zA-Z0-9_./-]+)", line)
        check(m and m[2] not in expected and m[2] != "SOURCE.sha256", "invalid SOURCE.sha256")
        expected[safe_name(m[2])] = m[1]
    check(
        expected
        and expected == {n: sha256(data) for n, data in files.items() if n != "SOURCE.sha256"},
        "public archive does not match SOURCE.sha256",
    )
    return files
