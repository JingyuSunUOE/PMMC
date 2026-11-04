"""Check the pristine extracted tree; optionally exercise source fingerprints offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path


def verify(root: Path, *, git_checkout: bool = False) -> int:
    manifest = root / "SOURCE.sha256"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("missing regular SOURCE.sha256")
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([a-zA-Z0-9_./-]+)", line)
        if not match or match[2] in expected or match[2] == "SOURCE.sha256":
            raise ValueError("invalid SOURCE.sha256")
        expected[match[2]] = match[1]
    if not expected:
        raise ValueError("empty SOURCE.sha256")
    actual = {}
    directories = {p for n in expected for p in Path(n).parents}

    def visit(directory: Path) -> None:
        for path in directory.iterdir():
            relative = path.relative_to(root)
            if git_checkout and relative == Path(".git") and not path.is_symlink():
                continue
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode) and relative in directories:
                visit(path)
            elif not stat.S_ISREG(mode) or path.stat().st_nlink != 1:
                raise ValueError(f"unexpected directory, link or special file: {relative}")
            elif path != manifest:
                data = path.read_bytes()
                if bool(mode & stat.S_IXUSR) != data.startswith(b"#!"):
                    raise ValueError(f"unexpected executable bit: {relative}")
                actual[relative.as_posix()] = hashlib.sha256(data).hexdigest()

    visit(root)
    if actual != expected:
        changed = sorted(
            n for n in actual.keys() | expected.keys() if actual.get(n) != expected.get(n)
        )
        raise ValueError(f"missing, changed or extra public files: {changed}")
    return len(actual)


def fingerprints(root: Path) -> dict[str, str]:
    from pmmc.experiments.step8_jobs import SOURCE_OWNERS, scientific_source_fingerprint

    return {name: scientific_source_fingerprint(root, name) for name in sorted(SOURCE_OWNERS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fingerprints", action="store_true")
    parser.add_argument(
        "--git-checkout", action="store_true", help="ignore only root .git metadata"
    )
    args = parser.parse_args()
    result = {"files_verified": verify(args.root, git_checkout=args.git_checkout)}
    if args.fingerprints:
        result["stage_fingerprints"] = fingerprints(args.root)
    print(json.dumps(result, sort_keys=True))
