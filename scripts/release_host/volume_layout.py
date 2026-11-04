"""Pinned one-volume bootstrap; never traverse symlinks or repair existing data."""

import json
import os
import re
import stat
import sys


def directory(root, relative, *, create, owner=None):
    parts = relative.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", p) or p in (".", "..") for p in parts):
        raise ValueError("invalid layout path")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    fd = os.open(root, flags)
    try:
        for n, name in enumerate(parts):
            leaf = n == len(parts) - 1
            created = False
            if create:
                try:
                    os.mkdir(name, 0o750 if leaf else 0o755, dir_fd=fd)
                    created = True
                except FileExistsError:
                    pass
            if leaf and owner is not None and not created:
                existing = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(existing.st_mode)
                    or (existing.st_uid, existing.st_gid) != owner
                ):
                    raise PermissionError("existing layout directory has a different owner")
                return
            child = os.open(name, flags, dir_fd=fd)
            os.close(fd)
            fd = child
            if leaf and owner is not None:
                if created:
                    os.fchown(fd, *owner)
                existing = os.fstat(fd)
                if (existing.st_uid, existing.st_gid) != owner:
                    raise PermissionError("existing layout directory has a different owner")
        if owner is None and not os.access(".", os.W_OK | os.X_OK, dir_fd=fd):
            raise PermissionError("layout directory is not writable by the application UID")
    finally:
        os.close(fd)


def main(argv):
    os.umask(0o022)
    if argv[0] == "report":
        if not re.fullmatch(r"step8/exploration/[A-Za-z0-9][A-Za-z0-9._-]{0,95}/reports", argv[1]):
            raise ValueError("invalid report destination")
        directory("/layout", argv[1], create=True)
        return
    mode, uid, gid, paths = argv
    if mode not in ("create", "verify"):
        raise ValueError("invalid layout mode")
    for path in json.loads(paths):
        directory(
            "/layout",
            path,
            create=mode == "create",
            owner=(int(uid), int(gid)) if mode == "create" else None,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
