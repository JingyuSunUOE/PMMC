"""Local private inventory, copy and restore drill. No deletion or network operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from collections import Counter
from pathlib import Path


def inventory(root: Path, *, hashes: bool = True) -> dict:
    entries = {}

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            name = path.relative_to(root).as_posix()
            info = path.lstat()
            item = {"mode": stat.S_IMODE(info.st_mode)}
            entries[name] = item
            if stat.S_ISLNK(info.st_mode):
                item.update(kind="link", target=os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                item.update(kind="dir")
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                item.update(kind="file", bytes=info.st_size)
                if hashes:
                    with path.open("rb") as stream:
                        item["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                    after = path.lstat()
                    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        raise ValueError(f"file changed while inventorying: {name}")
            else:
                raise ValueError(f"unsupported special file: {name}")

    visit(root)
    return {"format": "pmmc-private-tree-v1", "content_hashed": hashes, "entries": entries}


def summary(inv: dict) -> dict:
    groups = {}
    for name, entry in inv["entries"].items():
        group = groups.setdefault(name.split("/")[0], Counter())
        group[entry["kind"]] += 1
        group["bytes"] += entry.get("bytes", 0)
    return {"content_hashed": inv["content_hashed"], "groups": groups}


def require_separate(*paths: Path) -> None:
    if any(a.is_relative_to(b) for i, a in enumerate(paths) for j, b in enumerate(paths) if i != j):
        raise ValueError("source, backup and restore destinations must not overlap")


def write_new(path: Path, value: dict) -> None:
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def assert_same(expected: dict, observed: dict) -> None:
    if expected != observed:
        raise ValueError("inventory mismatch; preserve all copies and investigate privately")


def require_external(source: Path, destination_device: Path) -> None:
    if source.stat().st_dev == destination_device.stat().st_dev:
        raise ValueError("backup must be on a different filesystem; never clean up a local copy")


def check_space(inv: dict, parent: Path) -> None:
    size = sum(entry.get("bytes", 0) for entry in inv["entries"].values())
    if shutil.disk_usage(parent).free < size:
        raise ValueError("insufficient free space for a regular-file copy")


def backup(source: Path, dest: Path) -> None:
    require_separate(source, dest)
    require_external(source, dest.parent)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    inv = inventory(source)
    check_space(inv, dest.parent)
    dest.mkdir(mode=0o700, exist_ok=False)
    shutil.copytree(source, dest / "tree", symlinks=True)
    assert_same(inv, inventory(dest / "tree"))
    assert_same(inv, inventory(source))
    # A partial backup never has a completed inventory.
    write_new(dest / "inventory.json", inv)
    print("Private tree copy compared. An independent restore drill is still required.")


def restore_check(source: Path, backup_dir: Path, restore_dir: Path) -> None:
    require_separate(source, backup_dir, restore_dir)
    require_external(source, backup_dir)
    if restore_dir.exists() or restore_dir.is_symlink():
        raise FileExistsError(restore_dir)
    tree = backup_dir / "tree"
    manifest = backup_dir / "inventory.json"
    if tree.is_symlink() or manifest.is_symlink() or tree.stat().st_dev != backup_dir.stat().st_dev:
        raise ValueError("backup payload must be in the external backup directory")
    inv = json.loads(manifest.read_text())
    assert_same(inv, inventory(source))
    assert_same(inv, inventory(tree))
    check_space(inv, restore_dir.parent)
    restore_dir.mkdir(mode=0o700, exist_ok=False)
    # Copy from the verified tree, never interpret manifest paths as restore instructions.
    shutil.copytree(tree, restore_dir / "tree", symlinks=True)
    assert_same(inv, inventory(restore_dir / "tree"))
    write_new(
        restore_dir / "restore-check.json",
        {
            "tree_bytes_modes_and_link_text_match": True,
            "link_targets_not_backed_up_by_following_links": True,
            "cleanup_authorized": False,
            "inventory_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "entries": len(inv["entries"]),
        },
    )
    print("Restore compared; no cleanup authorized. Check off-tree/service coverage separately.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="action")
    check = sub.add_parser("inventory")
    check.add_argument("--report", required=True, type=Path)
    check.add_argument("--quick", action="store_true", help="metadata only, not a backup proof")
    copy = sub.add_parser("backup")
    copy.add_argument("--destination", required=True, type=Path)
    restore = sub.add_parser("restore-check")
    restore.add_argument("--backup", required=True, type=Path)
    restore.add_argument("--destination", required=True, type=Path)
    for cmd in (check, copy, restore):
        cmd.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    os.umask(0o077)
    if args.action == "inventory":
        report = args.report.resolve()
        require_separate(source, report)
        inv = inventory(source, hashes=not args.quick)
        write_new(report, inv)
        print(json.dumps(summary(inv), sort_keys=True))
    elif args.action == "backup":
        backup(source, args.destination.resolve())
    else:
        restore_check(source, args.backup.resolve(strict=True), args.destination.resolve())


if __name__ == "__main__":
    main()
