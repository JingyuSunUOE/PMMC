"""Strict Step8 MinIO IAM initialization; only the final report is safe to log.

Run in the dedicated image/Job, never a shared worker. mc output can contain
credentials and is deliberately captured, parsed, and never forwarded.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from botocore.exceptions import ClientError
from pydantic import SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from pmmc.experiments.step8_bootstrap import (
    _ensure_bucket,
    _object_store,
    _PreflightFailure,
    _s3_client,
)
from pmmc.infrastructure.settings import Settings

SECRET_DIRECTORY = Path("/var/run/secrets/pmmc-step8-minio")
SECRET_FIELDS = {
    "object_store_admin_access_key": "PMMC_MINIO_ROOT_USER",
    "object_store_admin_secret_key": "PMMC_MINIO_ROOT_PASSWORD",
    "object_store_access_key": "PMMC_OBJECT_STORE_ACCESS_KEY",
    "object_store_secret_key": "PMMC_OBJECT_STORE_SECRET_KEY",
    "object_store_evaluator_access_key": "PMMC_OBJECT_STORE_EVALUATOR_ACCESS_KEY",
    "object_store_evaluator_secret_key": "PMMC_OBJECT_STORE_EVALUATOR_SECRET_KEY",
}
POLICY_NAMES = {"runtime": "pmmc-step8-runtime", "evaluator": "pmmc-step8-evaluator"}
ALIAS = "step8admin"
SAFE_PROBE_ERRORS = {
    "object_store_iam_preflight_failed",
    "object_store_role_boundary_failed",
    "object_store_probe_cleanup_failed",
    "object_store_probe_cleanup_truncated",
    "object_store_probe_service_failed",
    "object_store_probe_content_mismatch",
    "object_store_bucket_creation_failed",
}


class IAMFailure(RuntimeError):
    """Contains only a fixed, credential-free error code."""


class _IAMSettings(Settings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # No .env, inherited AWS/PMMC credentials, or model-provider settings.
        return (init_settings,)


def load_settings(directory: Path, environ: Mapping[str, str]) -> Settings:
    values: dict[str, Any] = {}
    for field, filename in SECRET_FIELDS.items():
        value = (directory / filename).read_text(encoding="utf-8")
        minimum = 3 if field.endswith("access_key") else 8
        if not minimum <= len(value) <= 256 or any(
            char.isspace() or ord(char) < 32 for char in value
        ):
            raise IAMFailure("invalid_secret_file")
        if field.endswith("access_key") and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._@+=-]{2,127}", value
        ):
            raise IAMFailure("invalid_access_identifier")
        values[field] = SecretStr(value)
    for suffix in ("access_key", "secret_key"):
        if (
            len(
                {
                    values[f"{prefix}_{suffix}"].get_secret_value()
                    for prefix in ("object_store", "object_store_evaluator", "object_store_admin")
                }
            )
            != 3
        ):
            raise IAMFailure("identities_and_secrets_must_be_distinct")
    endpoint = environ.get("PMMC_OBJECT_STORE_ENDPOINT", "")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in endpoint)
    ):
        raise IAMFailure("invalid_object_store_endpoint")
    # Validate the port without including urllib's potentially sensitive error.
    _ = parsed.port
    buckets = [
        environ.get(name, "")
        for name in ("PMMC_OBJECT_STORE_BUCKET", "PMMC_OBJECT_STORE_EVALUATION_BUCKET")
    ]
    for bucket in buckets:
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket)
            or any(part in bucket for part in ("..", ".-", "-."))
            or bucket.startswith(("xn--", "sthree-", "amzn-s3-demo-"))
            or bucket.endswith(("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3"))
        ):
            raise IAMFailure("invalid_object_store_bucket")
        try:
            ipaddress.ip_address(bucket)
        except ValueError:
            pass
        else:
            raise IAMFailure("invalid_object_store_bucket")
    if len(set(buckets)) != 2:
        raise IAMFailure("buckets_must_be_distinct")
    return _IAMSettings(
        **values,
        object_store_endpoint=endpoint,
        object_store_secure=parsed.scheme == "https",
        object_store_path_style=True,
        object_store_bucket=buckets[0],
        object_store_evaluation_bucket=buckets[1],
        object_store_region=environ.get("PMMC_OBJECT_STORE_REGION", "us-east-1"),
    )


def policies(settings: Settings) -> dict[str, dict[str, Any]]:
    runtime = f"arn:aws:s3:::{settings.object_store_bucket}"
    evaluation = f"arn:aws:s3:::{settings.object_store_evaluation_bucket}"

    def statement(actions: list[str], resources: list[str]) -> dict[str, Any]:
        return {"Effect": "Allow", "Action": actions, "Resource": resources}

    return {
        "runtime": {
            "Version": "2012-10-17",
            "Statement": [
                statement(["s3:GetBucketLocation", "s3:ListBucket"], [runtime]),
                statement(["s3:GetObject", "s3:PutObject"], [runtime + "/*"]),
            ],
        },
        "evaluator": {
            "Version": "2012-10-17",
            "Statement": [
                statement(["s3:GetBucketLocation", "s3:ListBucket"], [runtime, evaluation]),
                statement(["s3:GetObject"], [runtime + "/*"]),
                statement(["s3:GetObject", "s3:PutObject"], [evaluation + "/*"]),
            ],
        },
    }


def canonical_policy(value: Any) -> str:
    """Compare exact permissions, accepting only harmless server serialization changes."""
    if not isinstance(value, dict) or value.get("Version") != "2012-10-17":
        raise IAMFailure("policy_document_invalid")
    if set(value) - {"Version", "Statement", "Id"} or value.get("Id") not in (None, ""):
        raise IAMFailure("policy_document_invalid")
    statements = value.get("Statement")
    if not isinstance(statements, list) or not statements:
        raise IAMFailure("policy_document_invalid")
    normalized = []
    for item in statements:
        if (
            not isinstance(item, dict)
            or set(item) - {"Effect", "Action", "Resource", "Sid"}
            or item.get("Sid") not in (None, "")
            or item.get("Effect") != "Allow"
        ):
            raise IAMFailure("policy_document_invalid")
        row: dict[str, Any] = {"Effect": "Allow"}
        for field in ("Action", "Resource"):
            items = item.get(field)
            items = [items] if isinstance(items, str) else items
            if (
                not isinstance(items, list)
                or not items
                or any(not isinstance(x, str) for x in items)
            ):
                raise IAMFailure("policy_document_invalid")
            row[field] = sorted(items)
        normalized.append(json.dumps(row, sort_keys=True))
    return json.dumps(sorted(normalized))


def _messages(raw: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    messages = []
    remaining = raw.strip()
    while remaining:
        value, end = decoder.raw_decode(remaining)
        if not isinstance(value, dict) or value.get("status") != "success":
            raise IAMFailure("mc_response_not_successful")
        messages.append(value)
        remaining = remaining[end:].lstrip()
    return messages


def _key(settings: Settings, role: str, suffix: str) -> str:
    prefix = "object_store" if role == "runtime" else f"object_store_{role}"
    value = getattr(settings, f"{prefix}_{suffix}")
    assert isinstance(value, SecretStr)
    return value.get_secret_value()


class MC:
    def __init__(self, settings: Settings, directory: Path):
        metadata = directory.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.getuid()
        ):
            raise IAMFailure("mc_config_directory_not_private")
        # Pinned mc's MC_HOST parser does not URL-decode credentials. Its v10
        # JSON config preserves raw strings and keeps credentials out of env/argv.
        config = {
            "version": "10",
            "aliases": {
                ALIAS: {
                    "url": settings.object_store_endpoint,
                    "accessKey": _key(settings, "admin", "access_key"),
                    "secretKey": _key(settings, "admin", "secret_key"),
                    "api": "S3v4",
                    "path": "auto",
                }
            },
        }
        descriptor = os.open(
            directory / "config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(config, stream)
        self.directory = directory
        self.environ = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(directory),
        }

    def call(
        self, *args: str, input_text: str | None = None, empty: bool = False
    ) -> list[dict[str, Any]]:
        # User-add reads a private pipe. All other commands have closed stdin.
        kwargs: dict[str, Any] = (
            {"input": input_text} if input_text is not None else {"stdin": subprocess.DEVNULL}
        )
        result = subprocess.run(
            ["mc", "--json", "--no-color", "--config-dir", str(self.directory), *args],
            env=self.environ,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            **kwargs,
        )
        if result.returncode != 0 or result.stderr.strip():
            raise IAMFailure("mc_command_failed")
        messages = _messages(result.stdout)
        if not messages and not empty:
            raise IAMFailure("mc_response_missing")
        return messages

    def inventory(self, kind: str, field: str) -> set[str]:
        records = self.call("admin", kind, "list", ALIAS, empty=True)
        result = set()
        for record in records:
            name = record.get(field)
            if not isinstance(name, str) or not name or name in result:
                raise IAMFailure("mc_inventory_invalid")
            result.add(name)
        return result

    def check_policy(self, name: str, expected: dict[str, Any]) -> None:
        destination = self.directory / "policy-readback.json"
        destination.unlink(missing_ok=True)
        self.call("admin", "policy", "info", ALIAS, name, "--policy-file", str(destination))
        if canonical_policy(json.loads(destination.read_text())) != canonical_policy(expected):
            raise IAMFailure("existing_policy_permissions_differ")

    def check_user(self, access_key: str, policy: str, *, allow_unbound: bool = False) -> bool:
        records = self.call("admin", "user", "info", ALIAS, access_key)
        if len(records) != 1:
            raise IAMFailure("mc_user_response_invalid")
        user = records[0]
        authentication = user.get("authentication", "")
        if (
            user.get("accessKey") != access_key
            or user.get("userStatus") != "enabled"
            or user.get("memberOf") not in (None, [])
            or not isinstance(authentication, str)
            or (authentication and not authentication.startswith("builtin ("))
        ):
            raise IAMFailure("existing_user_identity_or_groups_unsafe")
        if allow_unbound and user.get("policyName", "") == "":
            return False
        if user.get("policyName") != policy:
            raise IAMFailure("existing_user_policy_binding_unsafe")
        return True


def check_buckets(settings: Settings, *, create: bool) -> None:
    client = _s3_client(settings, "admin")
    try:
        for bucket in (settings.object_store_bucket, settings.object_store_evaluation_bucket):
            if create:
                _ensure_bucket(client, bucket, settings.object_store_region)
            try:
                client.get_bucket_policy(Bucket=bucket)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") != "NoSuchBucketPolicy":
                    raise IAMFailure("bucket_policy_inspection_failed") from None
            else:
                raise IAMFailure("existing_bucket_policy_requires_review")
    finally:
        client.close()


def initialize(settings: Settings, directory: Path, report: dict[str, Any]) -> None:
    mc = MC(settings, directory)
    expected = policies(settings)
    report["phase"] = "existing_iam_audit"
    existing_policies = mc.inventory("policy", "policy")
    existing_users = mc.inventory("user", "accessKey")
    for role, policy in POLICY_NAMES.items():
        if policy in existing_policies:
            mc.check_policy(policy, expected[role])
        if _key(settings, role, "access_key") in existing_users:
            mc.check_user(_key(settings, role, "access_key"), policy, allow_unbound=True)
    report["phase"] = "buckets"
    check_buckets(settings, create=True)
    report["phase"] = "provisioning"
    for role, policy in POLICY_NAMES.items():
        if policy not in existing_policies:
            destination = directory / f"{role}.json"
            destination.write_text(json.dumps(expected[role]))
            mc.call("admin", "policy", "create", ALIAS, policy, str(destination))
        access = _key(settings, role, "access_key")
        if access not in existing_users:
            mc.call(
                "admin",
                "user",
                "add",
                ALIAS,
                input_text=f"{access}\n{_key(settings, role, 'secret_key')}\n",
            )
        if not mc.check_user(access, policy, allow_unbound=True):
            mc.call("admin", "policy", "attach", ALIAS, policy, "--user", access)
        mc.check_user(access, policy)
        mc.check_policy(policy, expected[role])
    report["phase"] = "credential_and_permission_probes"
    report["object_store"] = {"ready": False}
    _object_store(settings, report["object_store"])
    report["phase"] = "final_iam_audit"
    for role, policy in POLICY_NAMES.items():
        mc.check_user(_key(settings, role, "access_key"), policy)
        mc.check_policy(policy, expected[role])
    check_buckets(settings, create=False)
    report.update(phase="completed", ready=True, policy_audit="verified")


def main() -> int:
    os.umask(0o077)
    report: dict[str, Any] = {
        "protocol": "step8-minio-iam-v1",
        "ready": False,
        "phase": "configuration",
    }
    try:
        settings = load_settings(SECRET_DIRECTORY, os.environ)
        with tempfile.TemporaryDirectory(prefix="step8-iam-") as tmp:
            initialize(settings, Path(tmp), report)
    except (Exception, KeyboardInterrupt) as exc:
        safe = isinstance(exc, IAMFailure) or (
            isinstance(exc, _PreflightFailure) and str(exc) in SAFE_PROBE_ERRORS
        )
        report.update(ready=False, error_code=str(exc) if safe else "iam_operation_failed")
        # No raw exception, traceback, mc output, Settings, or credential identifiers.
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
