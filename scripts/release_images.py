"""Produce and verify the external release image lock (stdlib only)."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

LOCK = Path("deploy/release/images.lock.json")
UPSTREAM = Path("deploy/release/upstream.json")
ROLES = {"app", "iam", "postgres", "minio"}
COMPONENT = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
REF = re.compile(rf"{COMPONENT}(?::[0-9]+)?(?:/{COMPONENT})+@sha256:([0-9a-f]{{64}})")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(root, name, limit):
    path = Path(root) / name
    try:
        if path.resolve() != Path(root).resolve() / name or not path.is_file():
            raise ValueError(f"missing or linked {name}")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as exc:
        raise ValueError(f"cannot read {name}") from exc
    if len(data) > limit:
        raise ValueError(f"oversized {name}")
    return data


def image_digest(ref):
    match = REF.fullmatch(ref) if isinstance(ref, str) else None
    if not match or match[1] == "0" * 64:
        raise ValueError("expected an untagged repository@sha256 digest")
    return match[1]


def commit(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot verify source commit") from exc


def strict_json(data):
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON key")
        return result

    return json.loads(data, object_pairs_hook=unique)


def validate(root, lock, overrides=None):
    return validate_payload(
        lock,
        read(root, "SOURCE.sha256", 8 << 20),
        read(root, UPSTREAM, 16384),
        overrides,
        source_commit=commit(root) if (root / ".git").exists() else None,
    )


def validate_payload(lock, manifest, upstream_bytes, overrides=None, *, source_commit=None):
    """Validate release assets that have already been read from a verified source."""
    if (
        not isinstance(lock, dict)
        or set(lock) != {"schema_version", "source_commit", "source_manifest_sha256", "images"}
        or type(lock["schema_version"]) is not int
        or lock["schema_version"] != 1
        or not isinstance(lock["source_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", lock["source_commit"])
        or lock["source_commit"] == "0" * 40
        or lock["source_manifest_sha256"] != sha(manifest)
        or source_commit is not None
        and lock["source_commit"] != source_commit
    ):
        raise ValueError("release lock schema or source identity mismatch")
    images = lock["images"]
    if not isinstance(images, dict) or set(images) != ROLES:
        raise ValueError("release lock needs exactly four image roles")
    digests = {k: image_digest(v) for k, v in images.items()}
    upstream = strict_json(upstream_bytes)
    if not isinstance(upstream, dict) or set(upstream) != {"postgres", "minio"}:
        raise ValueError("invalid public upstream inventory")
    for role in ("postgres", "minio"):
        if image_digest(upstream[role]) != digests[role]:
            raise ValueError(f"unreviewed upstream digest: {role}")
    if not all(images[k].startswith("ghcr.io/") for k in ("app", "iam")):
        raise ValueError("published app and IAM must be GHCR references")
    if len(set(digests.values())) != len(ROLES):
        raise ValueError("duplicate image identity")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict) or overrides.keys() - ROLES:
        raise ValueError("invalid image mirrors")
    for k, v in overrides.items():
        if image_digest(v) != digests[k]:
            raise ValueError(f"mirror changes reviewed {k} digest")
    return {**images, **overrides}


def load_images(root, overrides=None):
    root = Path(root)
    data = read(root, LOCK, 16384)
    if sha(data) != os.environ.get("PMMC_IMAGE_LOCK_SHA256"):
        raise ValueError("set PMMC_IMAGE_LOCK_SHA256 to the reviewed release asset checksum")
    return validate(root, strict_json(data), overrides)


def produce(root, app, iam):
    rev = commit(root)
    # The build context came from Git objects. Never stamp bytes from a newer
    # working tree (or a changed branch) into an older candidate's lock.
    manifest = committed_blob(root, rev, "SOURCE.sha256")
    upstream = committed_blob(root, rev, UPSTREAM)
    lock = {
        "schema_version": 1,
        "source_commit": rev,
        "source_manifest_sha256": sha(manifest),
        "images": {**strict_json(upstream), "app": app, "iam": iam},
    }
    validate_payload(lock, manifest, upstream, source_commit=rev)
    return (json.dumps(lock, sort_keys=True, indent=2) + "\n").encode()


def committed_blob(root, rev, path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "blob", f"{rev}:{path}"]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"missing committed release input {path}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--iam", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path.cwd()):
        parser.error("generate the lock outside its source tree")
    output = produce(Path.cwd(), args.app, args.iam)
    with args.output.open("xb") as stream:
        stream.write(output)
