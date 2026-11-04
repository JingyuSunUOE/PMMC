"""Private age release envelope; no extraction of arbitrary members."""

import argparse
import base64
import binascii
import os
import re
import shutil
import stat
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

if __package__ == "release_host":  # ./scripts/pmmc
    from release_images import UPSTREAM, validate_payload
else:
    from ..release_images import UPSTREAM, validate_payload

from .config import Error, check, validate_config
from .private_io import (
    input_fd,
    invoke,
    json_bytes,
    json_object,
    read_file,
    sha256,
    trusted_parent,
    write_new,
)
from .private_policy import (
    ABSENT,
    CARD,
    POLICY_KEYS,
    POLICY_LIMIT,
    RATE_PATH,
    REVIEW,
    import_policy,
    validate_policy,
)
from .private_secrets import ROLE_KEYS, present_roles, read_dotenv, select_roles, validate_values
from .private_source import IMAGE_LOCK, SOURCE_LIMIT, source_files
from .secrets import import_for_config, import_secrets

FORMAT = "pmmc-private-age-v1"
POLICY_FORMAT = "pmmc-private-age-v2"
MAX_ENVELOPE = 36 << 20
CONFIG_LIMIT = 64 << 10
IMAGES_LIMIT = 16384
BENCH_LOCK = "configs/release/benchmarks.lock.json"
HEADER = b"age-encryption.org/v1\n"


def recipient_key(value):
    # Only a native X25519 public recipient. Never pass through an identity,
    # plugin, SSH recipient, passphrase, recipient file, or arbitrary age flags.
    alphabet = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    check(
        isinstance(value, str)
        and value.startswith("age1")
        and len(value) == 62
        and all(c in alphabet for c in value[4:]),
        "an explicit native age public recipient is required",
    )
    polymod = 1
    for n in [3, 3, 3, 0, 1, 7, 5, *(alphabet.index(c) for c in value[4:])]:
        top = polymod >> 25
        polymod = ((polymod & 0x1FFFFFF) << 5) ^ n
        for i, mask in enumerate((0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)):
            if (top >> i) & 1:
                polymod ^= mask
    check(polymod == 1 and alphabet.index(value[-7]) & 15 == 0, "invalid age recipient checksum")
    return value


def parse_config(raw):
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError):
        raise Error("invalid private deployment TOML") from None


def config_bytes(raw, images, *, with_policy=False):
    # Serialize only validated settings, not comments or incidental private text.
    original = parse_config(raw)
    original["images"] = images
    if with_policy:
        original.setdefault("native", {})["rate_card"] = RATE_PATH
        original.setdefault("billing", {})["policy_secret"] = True
    lines = []

    def name(key):
        return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json_bytes(key).decode()

    def literal(value):
        if isinstance(value, list):
            return "[" + ", ".join(literal(v) for v in value) + "]"
        if isinstance(value, dict):
            return (
                "{ "
                + ", ".join(name(k) + " = " + literal(v) for k, v in sorted(value.items()))
                + " }"
            )
        check(type(value) in (int, bool, str), "unsupported private configuration value")
        return json_bytes(value).decode()

    def emit(table, prefix=()):
        if prefix:
            lines.append("[" + ".".join(map(name, prefix)) + "]")
        for key, value in sorted(table.items()):
            if not isinstance(value, dict):
                lines.append(name(key) + " = " + literal(value))
        for key, value in sorted(table.items()):
            if isinstance(value, dict):
                emit(value, (*prefix, key))

    emit(original)
    return ("\n".join(lines) + "\n").encode()


def release_inputs(archive, source_sha, images, images_sha, config):
    files = source_files(archive, source_sha)
    check(
        len(images) <= IMAGES_LIMIT and sha256(images) == images_sha, "image lock checksum mismatch"
    )
    check(len(config) <= CONFIG_LIMIT, "configuration exceeds size limit")
    cfg = parse_config(config)
    try:
        cfg["images"] = validate_payload(
            json_object(images),
            files["SOURCE.sha256"],
            files[UPSTREAM.as_posix()],
            cfg.get("images"),
        )
        cfg = validate_config(cfg, cfg.get("profile"))
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OverflowError,
        RecursionError,
    ):
        raise Error("invalid image lock or deployment configuration") from None
    lock_path = cfg["native"]["model_lock"]
    check(
        lock_path.startswith("/workspace/configs/"), "model asset lock must be in the public source"
    )
    lock_path = lock_path.removeprefix("/workspace/")
    check(lock_path in files, "model asset lock is absent from the public source")
    model_lock = json_object(files[lock_path])
    entries = model_lock.get("entries")
    check(isinstance(entries, list) and entries, "empty model asset lock")
    for e in entries:
        check(
            isinstance(e, dict)
            and isinstance(e.get("commit_sha"), str)
            and re.fullmatch(r"[a-f0-9]{40}", e["commit_sha"])
            and e["commit_sha"] != "0" * 40
            and isinstance(e.get("snapshot_content_hash"), str)
            and re.fullmatch(r"[a-f0-9]{64}", e["snapshot_content_hash"]),
            "unpinned model asset",
        )
    check(BENCH_LOCK in files, "benchmark asset lock is absent from the public source")
    benchmarks = json_object(files[BENCH_LOCK])
    datasets = benchmarks.get("datasets")
    check(
        benchmarks.get("protocol") == "pmmc-public-data-lock-v1"
        and isinstance(datasets, dict)
        and datasets.keys() == {"memgallery", "memlens"},
        "invalid benchmark asset lock",
    )
    for item in datasets.values():
        check(
            isinstance(item, dict)
            and all(
                isinstance(item.get(key), str)
                and re.fullmatch(rf"[a-f0-9]{{{n}}}", item[key])
                and item[key] != "0" * n
                for key, n in (
                    ("hf_revision", 40),
                    ("github_revision", 40),
                    ("materialized_sha256", 64),
                )
            ),
            "unpinned benchmark asset",
        )
    assets = {
        "format": "pmmc-deploy-assets-v1",
        "source_sha256": source_sha,
        "source_manifest_sha256": sha256(files["SOURCE.sha256"]),
        "images_sha256": images_sha,
        "model_lock_path": lock_path,
        "model_lock_sha256": sha256(files[lock_path]),
        "benchmark_lock_path": BENCH_LOCK,
        "benchmark_lock_sha256": sha256(files[BENCH_LOCK]),
    }
    return files, cfg, assets


def blob(data):
    return {"sha256": sha256(data), "base64": base64.b64encode(data).decode("ascii")}


def unblob(obj, limit):
    check(
        isinstance(obj, dict)
        and obj.keys() == {"sha256", "base64"}
        and isinstance(obj["base64"], str)
        and len(obj["base64"]) <= 4 * ((limit + 2) // 3),
        "invalid private envelope member",
    )
    try:
        data = base64.b64decode(obj["base64"], validate=True)
    except (ValueError, binascii.Error):
        raise Error("invalid private envelope encoding") from None
    check(len(data) <= limit and sha256(data) == obj["sha256"], "private member checksum mismatch")
    return data


def check_roles(cfg, roles):
    if roles:
        select_roles(roles)
        check(
            "hub" not in roles or cfg["models"]["use_hf_token"], "Hub credential role is disabled"
        )


def pack(args):
    recipient = recipient_key(args.recipient)
    card, review = getattr(args, "rate_card", None), getattr(args, "provider_policy", None)
    check((card is None) == (review is None), "provide --rate-card and --provider-policy together")
    archive = read_file(args.source_archive, SOURCE_LIMIT)
    images = read_file(args.images_lock, IMAGES_LIMIT)
    config = read_file(args.config, CONFIG_LIMIT, private=True)
    files, cfg, assets = release_inputs(
        archive, args.source_sha256, images, args.images_sha256, config
    )
    pair = None
    if card is not None:
        check(
            cfg["native"].setdefault("rate_card", RATE_PATH) == RATE_PATH,
            "bundled rate card requires " + RATE_PATH,
        )
        cfg["billing"]["policy_secret"] = True
        pair = {
            CARD: read_file(card, POLICY_LIMIT, private=True),
            REVIEW: read_file(review, POLICY_LIMIT, private=True),
        }
        validate_policy(pair, cfg, files)
    config = config_bytes(config, cfg["images"], with_policy=pair is not None)
    check(len(config) <= CONFIG_LIMIT, "canonical configuration exceeds size limit")
    roles = args.role or []
    check_roles(cfg, roles)
    check(bool(args.dotenv) == bool(roles), "select dotenv input and credential roles together")
    values = read_dotenv(args.dotenv, roles) if roles else {}
    envelope = {
        "format": POLICY_FORMAT if pair is not None else FORMAT,
        "source_archive": blob(archive),
        "images_lock": blob(images),
        "assets_lock": blob(json_bytes(assets)),
        "private_config": blob(config),
        "credentials": values,
    }
    if pair is not None:
        envelope["pricing_policy"] = {k: blob(v) for k, v in pair.items()}
    plain = json_bytes(envelope)
    check(len(plain) <= MAX_ENVELOPE, "private envelope exceeds size limit")
    encrypted = invoke(["age", "--encrypt", "--recipient", recipient], plain)
    check(encrypted.startswith(HEADER) and len(encrypted) <= MAX_ENVELOPE, "invalid age ciphertext")
    write_new(args.output, encrypted)
    return {
        "bundle_sha256": sha256(encrypted),
        "source_sha256": args.source_sha256,
        "images_sha256": args.images_sha256,
        "pricing_policy": "bundled; explicit import required" if pair is not None else ABSENT,
    }


@dataclass(repr=False)
class Opened:
    ciphertext: bytes
    archive: bytes
    images: bytes
    files: dict
    assets: dict
    cfg: dict
    toml: bytes
    values: dict
    policy: dict | None


def decrypt(args):
    cipher = read_file(args.bundle, MAX_ENVELOPE)
    check(
        cipher.startswith(HEADER) and sha256(cipher) == args.bundle_sha256,
        "encrypted bundle checksum or header mismatch",
    )
    fd = input_fd(args.identity, private=True)
    try:
        # The identity is neither read by Python nor looked up by default.
        plain = invoke(["age", "--decrypt", "--identity", f"/dev/fd/{fd}"], cipher, pass_fds=(fd,))
    finally:
        os.close(fd)
    check(len(plain) <= MAX_ENVELOPE, "private envelope exceeds size limit")
    obj = json_object(plain)
    has_policy = obj.get("format") == POLICY_FORMAT
    check(
        obj.keys()
        == (
            {
                "format",
                "source_archive",
                "images_lock",
                "assets_lock",
                "private_config",
                "credentials",
            }
            | ({"pricing_policy"} if has_policy else set())
        )
        and obj["format"] in (FORMAT, POLICY_FORMAT),
        "unknown private envelope structure",
    )
    archive = unblob(obj["source_archive"], SOURCE_LIMIT)
    images = unblob(obj["images_lock"], IMAGES_LIMIT)
    config = unblob(obj["private_config"], CONFIG_LIMIT)
    files, cfg, assets = release_inputs(
        archive, args.source_sha256, images, args.images_sha256, config
    )
    check(
        unblob(obj["assets_lock"], IMAGES_LIMIT) == json_bytes(assets),
        "release asset lock mismatch",
    )
    values = obj["credentials"]
    check(isinstance(values, dict), "invalid credential fields")
    roles = present_roles(values)
    check_roles(cfg, roles)
    if values:
        validate_values(values, roles)
    check(config == config_bytes(config, cfg["images"]), "noncanonical private configuration")
    policy = None
    if has_policy:
        blobs = obj["pricing_policy"]
        check(
            isinstance(blobs, dict) and blobs.keys() == POLICY_KEYS,
            "unexpected pricing policy fields",
        )
        policy = {key: unblob(value, POLICY_LIMIT) for key, value in blobs.items()}
        validate_policy(policy, cfg, files)
    return Opened(cipher, archive, images, files, assets, cfg, config, values, policy)


def select_import(opened, roles):
    keys = select_roles(roles)
    check_roles(opened.cfg, roles)
    check(
        opened.cfg["profile"] == "kubernetes" and set(roles) <= set(present_roles(opened.values)),
        "requested import unavailable",
    )
    return {k: opened.values[k] for k in keys & opened.values.keys()}


def install(args):
    opened = decrypt(args)
    roles = args.role or []
    values = select_import(opened, roles) if roles else {}
    check(roles or not args.rotate, "rotation requires an explicit import role")
    dest = Path(args.destination).absolute()
    parent = trusted_parent(dest)
    try:
        os.mkdir(dest.name, mode=0o700, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        source = dest / "source"
        directories = {
            p
            for n in (*opened.files, IMAGE_LOCK)
            for p in (source / n).parents
            if p == source or p.is_relative_to(source)
        }
        for directory in sorted(directories, key=lambda p: len(p.parts)):
            directory.mkdir(mode=0o755)
            directory.chmod(0o755)  # Docker COPY must retain traversable public source dirs.
        for name, data in opened.files.items():
            path = source / name
            write_new(path, data, 0o755 if data.startswith(b"#!") else 0o644)
        image_path = source / IMAGE_LOCK
        write_new(image_path, opened.images, 0o644)
        public_name = "SOURCE.tar.gz" if opened.archive.startswith(b"\x1f\x8b") else "SOURCE.tar"
        write_new(dest / public_name, opened.archive)
        write_new(dest / "assets.lock.json", json_bytes(opened.assets))
        write_new(dest / "deployment.age", opened.ciphertext)
    except BaseException:
        shutil.rmtree(dest)
        raise
    result = {
        "installed": True,
        "public_files_verified": len(opened.files),
        "source_sha256": opened.assets["source_sha256"],
        "images_sha256": opened.assets["images_sha256"],
        "pricing_policy": "bundled; explicit import required"
        if opened.policy is not None
        else ABSENT,
    }
    if roles:
        result["secrets"] = import_for_config(opened.cfg, values, roles, rotate=args.rotate)
    return result


def add_private_commands(subparsers):
    """Register only the private group; the host CLI already owns secrets import."""
    group = subparsers.add_parser("private").add_subparsers(required=True)
    build = group.add_parser("pack")
    build.set_defaults(private_action="pack")
    for flag in (
        "source-archive",
        "source-sha256",
        "images-lock",
        "images-sha256",
        "config",
        "recipient",
        "output",
    ):
        build.add_argument("--" + flag, required=True)
    build.add_argument("--dotenv", "--env-file", help="explicit private file or - for stdin")
    build.add_argument("--role", action="append", choices=ROLE_KEYS)
    build.add_argument(
        "--rate-card", help="private reviewed pricing YAML; requires --provider-policy"
    )
    build.add_argument("--provider-policy", help="private matching pricing review JSON")
    for verb in ("install", "import", "import-policy", "config"):
        command = group.add_parser(verb)
        command.set_defaults(private_action="bundle-import" if verb == "import" else verb)
        for flag in ("bundle", "bundle-sha256", "source-sha256", "images-sha256", "identity"):
            command.add_argument("--" + flag, required=True)
        if verb in ("install", "import"):
            command.add_argument(
                "--role",
                action="append",
                choices=ROLE_KEYS,
                required=verb == "import",
                help="import this role into the configured cluster",
            )
            command.add_argument("--rotate", action="store_true")
        if verb == "install":
            command.add_argument("--destination", required=True)
        if verb == "config":
            command.add_argument(
                "--to-fd", type=int, required=True, help="explicit pipe descriptor, 3 or above"
            )


def add_secret_commands(subparsers):
    importer = subparsers.add_parser("secrets").add_subparsers(required=True).add_parser("import")
    importer.set_defaults(private_action="import")
    importer.add_argument("--config", required=True)
    importer.add_argument("--env-file", "--dotenv", required=True)
    importer.add_argument("--role", action="append", choices=ROLE_KEYS)
    importer.add_argument("--rotate", action="store_true")


def run_private_command(args):
    if args.private_action == "pack":
        return pack(args)
    if args.private_action == "install":
        return install(args)
    if args.private_action == "import-policy":
        opened = decrypt(args)
        return import_policy(opened.cfg, opened.policy)
    if args.private_action == "bundle-import":
        opened = decrypt(args)
        return import_for_config(
            opened.cfg, select_import(opened, args.role), args.role, rotate=args.rotate
        )
    if args.private_action == "config":
        check(args.to_fd >= 3, "config handoff requires an explicit pipe descriptor")
        pipe = os.fstat(args.to_fd)
        check(
            stat.S_ISFIFO(pipe.st_mode)
            and pipe.st_uid == os.geteuid()
            and (not pipe.st_nlink or not pipe.st_mode & 0o077),
            "config handoff requires an explicit pipe, never a file or terminal",
        )
        opened = decrypt(args)
        with os.fdopen(os.dup(args.to_fd), "wb") as stream:
            stream.write(opened.toml)
        return {"config_sent_to_pipe": True}
    check(args.private_action == "import", "unknown private operation")
    from .config import load_config

    fd = input_fd(args.config, private=True)
    try:
        cfg = load_config(f"/dev/fd/{fd}", "kubernetes")
    finally:
        os.close(fd)
    return import_secrets(cfg, args.env_file, roles=args.role, rotate=args.rotate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    add_private_commands(sub)
    add_secret_commands(sub)
    try:
        print(json_bytes(run_private_command(parser.parse_args(argv))).decode())
    except (Error, OSError) as exc:
        print(
            str(exc) if isinstance(exc, Error) else "private host filesystem error", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
