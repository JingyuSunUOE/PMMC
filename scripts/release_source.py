"""Package only reviewed public Git blobs; never traverse the working tree."""

import argparse
import gzip
import hashlib
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    "README.md",
    "RELEASE_NOTES.md",
    "SOURCE.sha256",
    "THIRD_PARTY.md",
    "alembic.ini",
    "compose.yaml",
    "pyproject.toml",
    "uv.lock",
}
TREES = {
    ".github",
    "alembic",
    "benchmarks",
    "configs",
    "deploy",
    "docker",
    "docs",
    "prompts",
    "scripts",
    "src",
    "tests",
}
DENIED = {"private", "artifacts", "runs", "outputs", "backups", "weights", "__pycache__"}
TEXT_SUFFIXES = {".py", ".mako", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".sh"}
SPECIAL_PATHS = {
    "scripts/pmmc",
    "src/pmmc/py.typed",
    "docker/minio/.dockerignore",
    "docker/minio/Dockerfile",
    "docker/minio/step8.Dockerfile",
    "docker/postgres/.dockerignore",
    "docker/postgres/Dockerfile",
}
BENCHMARK_DOCS = {
    "benchmarks/README.md",
    "benchmarks/memlens/README.md",
    "benchmarks/memgallery/README.md",
}


def public_name(name):
    p = PurePosixPath(name)
    if name in ROOT_FILES:
        return
    if (
        not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
        or name != str(p)
        or p.is_absolute()
        or ".." in p.parts
        or len(p.parts) < 2
        or p.parts[0] not in TREES
        or (name not in SPECIAL_PATHS and p.suffix not in TEXT_SUFFIXES)
        or any(s.lower() in DENIED or s.lower().startswith(".env") for s in p.parts)
        or any(s.startswith(".") and s not in {".github", ".dockerignore"} for s in p.parts)
        or name.casefold() == "deploy/release/images.lock.json"
        or (p.parts[0] == "benchmarks" and name not in BENCHMARK_DOCS)
        or (
            p.suffix in {".json", ".yaml", ".yml", ".toml", ".txt"}
            and re.search(
                r"(?:^|[._-])(?:credentials?|passwords?|secrets?)(?:$|[._-])", p.name.lower()
            )
        )
    ):
        raise ValueError(f"not a public source path: {name!r}")


def inventory(names):
    for name in names:
        public_name(name)
    dirs = {str(p) for name in names for p in PurePosixPath(name).parents if str(p) != "."}
    paths = set(names) | dirs
    if len({n.casefold() for n in paths}) != len(paths):
        raise ValueError("case-colliding source paths")
    lines = ["**"]
    for d in sorted(dirs, key=lambda s: (s.count("/"), s)):
        lines.extend(("!" + d, d + "/**"))
    lines.extend("!" + n for n in sorted(names))
    return ("\n".join(lines) + "\n").encode("ascii")


def git(*args, data=None):
    return subprocess.check_output(["git", *args], input=data)


def entries(index):
    top = Path(git("rev-parse", "--show-toplevel").decode().strip())
    if top.resolve() != Path.cwd().resolve():
        raise ValueError("run from the root of the public clone")
    rows = (git("ls-files", "-s", "-z") if index else git("ls-tree", "-rz", "HEAD")).split(b"\0")
    result = {}
    for row in rows[:-1]:
        meta, raw = row.split(b"\t", 1)
        mode, obj, last = meta.decode("ascii").split()
        name = raw.decode("ascii")
        public_name(name)
        if (
            mode not in {"100644", "100755"}
            or name in result
            or (last != "0" if index else obj != "blob")
        ):
            raise ValueError(f"links, submodules and unmerged paths are not source: {name}")
        result[name] = (int(mode[-3:], 8), obj if index else last)
    if not {".dockerignore", "SOURCE.sha256"} <= result.keys():
        raise ValueError("not a public source checkout")
    return result


def source(index=False):
    files = entries(index)
    expected = inventory(files)
    if git("cat-file", "blob", files[".dockerignore"][1]) != expected:
        raise ValueError("review the path diff and refresh the exact .dockerignore inventory")
    ids = "".join(f"{oid}\n" for _, oid in files.values()).encode()
    sizes = [
        int(row.split()[-1]) for row in git("cat-file", "--batch-check", data=ids).splitlines()
    ]
    if len(files) > 4096 or max(sizes) > (8 << 20) or sum(sizes) > (56 << 20):
        raise ValueError("oversized public source inventory")
    batch = git("cat-file", "--batch", data=ids)
    stream = io.BytesIO(batch)
    result = {}
    for name, (mode, oid) in files.items():
        actual, kind, length = stream.readline().split()
        if actual.decode() != oid or kind != b"blob" or int(length) > (8 << 20):
            raise ValueError(f"invalid source blob: {name}")
        data = stream.read(int(length))
        if stream.read(1) != b"\n" or b"\0" in data or (mode == 0o755) != data.startswith(b"#!"):
            raise ValueError(f"invalid public source content or mode: {name}")
        result[name] = (mode, data)
    return result


def manifest(files):
    return "".join(
        f"{hashlib.sha256(files[n][1]).hexdigest()}  {n}\n"
        for n in sorted(files)
        if n != "SOURCE.sha256"
    ).encode("ascii")


def archive(files, dest):
    if files["SOURCE.sha256"][1] != manifest(files):
        raise ValueError("SOURCE.sha256 does not match the committed public source")
    # Do not overwrite anything, or emit a partially verified archive.
    with dest.open("xb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
    ) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name, (mode, data) in sorted(files.items()):
                item = tarfile.TarInfo(name)
                item.mode, item.size = mode, len(data)
                tar.addfile(item, io.BytesIO(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "manifest", "archive"])
    parser.add_argument("--index", action="store_true", help="use the reviewed Git staging index")
    parser.add_argument("--output", type=Path, help="new tar.gz destination (archive only)")
    args = parser.parse_args()
    if args.action == "archive":
        if not args.output or args.index:
            parser.error("archive requires --output and an immutable HEAD")
        archive(source(), args.output)
    elif args.output:
        parser.error("redirect standard output for inventory/manifest")
    elif args.action == "manifest":
        sys.stdout.buffer.write(manifest(source(args.index)))
    else:
        sys.stdout.buffer.write(inventory(entries(args.index)))
