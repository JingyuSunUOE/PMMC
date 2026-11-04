"""Pinned one-volume bootstrap; never traverse symlinks or repair existing data."""

import json
import os
import re
import sys


def directory(root, relative):
    parts = relative.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", p) or p in (".", "..") for p in parts):
        raise ValueError("invalid layout path")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    fd = os.open(root, flags)
    try:
        for n, name in enumerate(parts):
            leaf = n == len(parts) - 1
            try:
                os.mkdir(name, 0o750 if leaf else 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            except PermissionError:
                raise PermissionError(
                    "provision the PVC parent writable by the configured UID/GID"
                ) from None
            child = os.open(name, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        if not os.access(".", os.W_OK | os.X_OK, dir_fd=fd):
            raise PermissionError("layout directory is not writable by the application UID")
    finally:
        os.close(fd)


def main(argv):
    mode, uid, gid, paths = argv
    if int(uid) <= 0 or int(gid) <= 0 or (os.geteuid(), os.getegid()) != (int(uid), int(gid)):
        raise PermissionError("storage initialization requires the configured non-root UID/GID")
    os.umask(0o022)
    if mode == "report":
        if not re.fullmatch(r"step8/exploration/[A-Za-z0-9][A-Za-z0-9._-]{0,95}/reports", paths):
            raise ValueError("invalid report destination")
        directory("/layout", paths)
        return
    if mode != "create":
        raise ValueError("invalid layout mode")
    for path in json.loads(paths):
        directory("/layout", path)


if __name__ == "__main__":
    main(sys.argv[1:])
