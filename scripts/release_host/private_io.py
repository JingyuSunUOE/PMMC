"""Bounded, non-following POSIX I/O for the private deployment tools."""

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

from .config import Error, check


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_object(raw):
    def pairs(items):
        obj = dict(items)
        check(len(obj) == len(items), "duplicate JSON field")
        return obj

    try:
        obj = json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, UnicodeError, RecursionError):
        raise Error("invalid deployment JSON") from None
    check(isinstance(obj, dict), "invalid deployment JSON")
    return obj


def directory(path):
    # Open each directory relative to its verified parent. No symlink traversal,
    # including symlinked ancestors of otherwise regular files.
    path = Path(path).absolute()
    check(".." not in path.parts, "parent traversal is not allowed")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def input_fd(path, private):
    path = Path(path)
    fd = directory(path.parent)
    try:
        item = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)
    try:
        st = os.fstat(item)
        check(
            stat.S_ISREG(st.st_mode) and st.st_nlink == 1,
            "expected a regular input file with one link",
        )
        if private:
            check(
                st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) in (0o400, 0o600),
                "private input must be owner-only (0400 or 0600)",
            )
        return item
    except BaseException:
        os.close(item)
        raise


def read_file(path, limit, *, private=False):
    with os.fdopen(input_fd(path, private), "rb") as stream:
        check(os.fstat(stream.fileno()).st_size <= limit, "input exceeds size limit")
        data = stream.read(limit + 1)
    check(len(data) <= limit, "input exceeds size limit")
    return data


def trusted_parent(path):
    parent = directory(Path(path).parent)
    st = os.fstat(parent)
    if st.st_uid != os.geteuid() or st.st_mode & 0o022:
        os.close(parent)
        raise Error("output parent must be operator-owned and not group/world-writable")
    return parent


def write_new(path, data, mode=0o600):
    path = Path(path)
    parent = trusted_parent(path)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(path.name, flags, mode, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                os.fchmod(stream.fileno(), mode)
        except BaseException:
            os.unlink(path.name, dir_fd=parent)
            raise
    finally:
        os.close(parent)


def invoke(argv, data=None, *, kube=False, pass_fds=()):
    allowed = ("HOME", "PATH", "KUBECONFIG") if kube else ("PATH",)
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    try:
        result = subprocess.run(
            argv,
            input=data if data is not None else b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=90,
            env=env,
            pass_fds=pass_fds,
        )
    except (OSError, subprocess.SubprocessError):
        raise Error(
            "kubectl unavailable or timed out" if kube else "age unavailable or timed out"
        ) from None
    check(result.returncode == 0, "kubectl operation failed" if kube else "age operation failed")
    return result.stdout
