"""Explicit, forward-only Step8 infrastructure bootstrap and role preflight.

No model calls, Kubernetes mutations, or MinIO IAM administration occur here.
S3 is not the MinIO admin API: missing users/policies remain an explicit blocker.
"""

from __future__ import annotations

import re
import sys
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import boto3
import psycopg
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from botocore import UNSIGNED
from botocore.config import Config as S3Config
from botocore.exceptions import ClientError
from psycopg import sql
from psycopg.rows import dict_row
from sqlalchemy.engine import URL, make_url

from pmmc.infrastructure.database.engine import normalize_async_database_url
from pmmc.infrastructure.settings import Settings
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION

TARGET_REVISION = STEP8_ALEMBIC_REVISION
ONLINE_ROLE = "pmmc_online"
ONLINE_WORKER = "pmmc_online_worker"
ROLE_FIELDS = {
    "pmmc_owner": "database_admin_url",
    "pmmc_runtime": "database_url",
    "pmmc_evaluator": "database_evaluator_url",
    ONLINE_WORKER: "database_online_url",
}
IAM_PREREQUISITE = "initialize_minio_runtime_evaluator_users_and_least_privilege_policies"
_PRIVILEGED_FLAGS = ("rolsuper", "rolcreatedb", "rolcreaterole", "rolreplication", "rolbypassrls")
_DENIED = {"AccessDenied", "AllAccessDisabled", "403"}
_IAM_ERRORS = _DENIED | {
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "InvalidToken",
    "ExpiredToken",
}
_MISSING = {"NoSuchBucket", "NoSuchKey", "NotFound", "404"}


class Step8BootstrapError(RuntimeError):
    """Only code/report are safe to print; underlying credential-bearing errors are suppressed."""

    def __init__(self, code: str, report: dict[str, Any]):
        super().__init__(code)
        self.code = code
        self.report = report


class _PreflightFailure(RuntimeError):
    pass


def _secret(settings: Settings, field: str) -> str:
    value = getattr(settings, field)
    if value is None or not value.get_secret_value().strip():
        raise _PreflightFailure(f"missing_{field}")
    return value.get_secret_value()


def validate_settings(settings: Settings) -> dict[str, URL]:
    """Validate all credentials/endpoints before the first infrastructure mutation."""
    urls: dict[str, URL] = {}
    for role, field in ROLE_FIELDS.items():
        url = make_url(normalize_async_database_url(_secret(settings, field)))
        if (
            url.username != role
            or not url.password
            or not url.host
            or not url.database
            or set(url.query) - {"sslmode", "sslrootcert", "sslcert", "sslkey"}
        ):
            raise _PreflightFailure("invalid_database_role_endpoint")
        urls[role] = url
    if (
        len({((url.host or "").lower(), url.port or 5432, url.database) for url in urls.values()})
        != 1
    ):
        raise _PreflightFailure("database_roles_target_different_endpoints")
    if len({url.password for url in urls.values()}) != len(urls):
        raise _PreflightFailure("database_role_passwords_must_be_distinct")
    endpoint = urlsplit(settings.object_store_endpoint or "")
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in {"", "/"}
        or settings.object_store_secure != (endpoint.scheme == "https")
    ):
        raise _PreflightFailure("invalid_object_store_endpoint")
    buckets = (settings.object_store_bucket, settings.object_store_evaluation_bucket)
    if len(set(buckets)) != 2 or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) or ".." in bucket
        for bucket in buckets
    ):
        raise _PreflightFailure("invalid_or_shared_object_store_buckets")
    access_keys = []
    for prefix in ("object_store", "object_store_evaluator", "object_store_admin"):
        access_keys.append(_secret(settings, f"{prefix}_access_key"))
        _secret(settings, f"{prefix}_secret_key")
    if len(set(access_keys)) != len(access_keys):
        raise _PreflightFailure("object_store_identities_must_be_distinct")
    return urls


def _connect(url: URL):
    return psycopg.connect(
        url.set(drivername="postgresql").render_as_string(hide_password=False),
        autocommit=True,
        connect_timeout=10,
        row_factory=dict_row,
    )


def _check_role(connection, role: str, *, membership: str | None = None) -> bool:
    row = connection.execute(
        "SELECT oid, rolname, rolcanlogin, rolinherit, rolsuper, rolcreatedb, rolcreaterole, "
        "rolreplication, rolbypassrls FROM pg_roles WHERE rolname=%s",
        (role,),
    ).fetchone()
    if row is None:
        return False
    if any(row[key] for key in _PRIVILEGED_FLAGS):
        raise _PreflightFailure("existing_worker_role_has_unsafe_privileges")
    memberships = connection.execute(
        "SELECT parent.rolname, membership.admin_option FROM pg_auth_members membership "
        "JOIN pg_roles parent ON parent.oid=membership.roleid "
        "WHERE membership.member=%s",
        (row["oid"],),
    ).fetchall()
    if any(item["rolname"] != membership or item["admin_option"] for item in memberships):
        raise _PreflightFailure("existing_worker_role_has_unsafe_membership")
    if role == ONLINE_WORKER:
        direct = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_shdepend WHERE refclassid='pg_authid'::regclass "
            "AND refobjid=%s AND deptype IN ('o','a')) AS unsafe",
            (row["oid"],),
        ).fetchone()
        if direct is None or direct["unsafe"]:
            raise _PreflightFailure("online_worker_has_direct_privileges_or_owned_objects")
    if role == ONLINE_ROLE and row["rolcanlogin"]:
        raise _PreflightFailure("online_group_must_not_allow_login")
    return True


def _provision_login(connection, role: str, password: str, *, membership: str | None = None):
    exists = _check_role(connection, role, membership=membership)
    # PostgreSQL utility statements do not accept a bind parameter for PASSWORD.
    # psycopg Literal performs SQL quoting; never log this composed statement.
    statement = sql.SQL(
        "{} ROLE {} LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
        "NOREPLICATION NOBYPASSRLS PASSWORD {}"
    ).format(sql.SQL("ALTER" if exists else "CREATE"), sql.Identifier(role), sql.Literal(password))
    connection.execute(statement)


def _verify_online_boundary(
    connection, role: str, source: Any, *, additional_views: tuple[str, ...] = (),
) -> None:
    read_tables = [*source._ONLINE_READ_TABLES, *source._ONLINE_VIEWS, *additional_views]
    insert_tables = list(source._ONLINE_INSERT_TABLES)
    # Check effective rights, including PUBLIC and old column grants, not only
    # the direct membership that this bootstrap creates.
    denied = connection.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname IN ('pmmc_runtime','pmmc_evaluation') "
        "AND c.relkind IN ('r','p','v','m','f') AND ("
        "((n.nspname<>'pmmc_runtime' OR NOT (c.relname=ANY(%s))) AND "
        "(has_table_privilege(%s,c.oid,'SELECT') OR has_any_column_privilege(%s,c.oid,'SELECT'))) "
        "OR ((n.nspname<>'pmmc_runtime' OR NOT (c.relname=ANY(%s))) AND "
        "(has_table_privilege(%s,c.oid,'INSERT') OR has_any_column_privilege(%s,c.oid,'INSERT'))) "
        "OR has_table_privilege(%s,c.oid,'UPDATE,DELETE,TRUNCATE') "
        "OR has_any_column_privilege(%s,c.oid,'UPDATE'))) AS unsafe",
        (read_tables, role, role, insert_tables, role, role, role, role),
    ).fetchone()
    if denied is None or denied["unsafe"]:
        raise _PreflightFailure("online_effective_privilege_boundary_failed")


def _postgres(settings: Settings, urls: dict[str, URL], report: dict[str, Any]) -> None:
    config = Config(str(settings.project_root / "alembic.ini"))
    config.set_main_option("script_location", str(settings.project_root / "alembic"))
    config.attributes["database_url"] = urls["pmmc_owner"].render_as_string(hide_password=False)
    migrations = ScriptDirectory.from_config(config)
    if migrations.get_heads() != [TARGET_REVISION]:
        raise _PreflightFailure("unexpected_migration_head")
    with _connect(urls["pmmc_owner"]) as connection:
        state = connection.execute(
            "SELECT current_user AS identity, "
            "current_setting('server_version_num')::int AS version, "
            "pg_get_userbyid(datdba) AS owner FROM pg_database WHERE datname=current_database()"
        ).fetchone()
        if (
            state is None
            or state["identity"] != "pmmc_owner"
            or state["owner"] != "pmmc_owner"
            or not 160000 <= state["version"] < 170000
        ):
            raise _PreflightFailure("postgres_owner_or_version_preflight_failed")
        lock = connection.execute(
            "SELECT pg_try_advisory_lock(1398031696, 32) AS locked"
        ).fetchone()
        if lock is None or not lock["locked"]:
            raise _PreflightFailure("bootstrap_already_running")
        for role in ("pmmc_runtime", "pmmc_evaluator", ONLINE_ROLE):
            _check_role(connection, role)
        _check_role(connection, ONLINE_WORKER, membership=ONLINE_ROLE)
        # The session lock is released when this connection closes; role DDL is
        # committed before Alembic opens its separate migration connection.
        with connection.transaction():
            for role in ("pmmc_runtime", "pmmc_evaluator"):
                _provision_login(connection, role, urls[role].password or "")
            connection.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        report["runtime_evaluator_roles"] = "provisioned"
        command.upgrade(config, TARGET_REVISION)
        revisions = connection.execute("SELECT version_num FROM public.alembic_version").fetchall()
        if revisions != [{"version_num": TARGET_REVISION}]:
            raise _PreflightFailure("migration_revision_mismatch")
        report["schema_revision"] = TARGET_REVISION
        with connection.transaction():
            from pmmc.infrastructure.database.step8_evaluation_scope import install_evaluation_scope

            report["evaluation_scope_extension"] = install_evaluation_scope(connection)
            if not _check_role(connection, ONLINE_ROLE):
                raise _PreflightFailure("online_group_missing_after_migration")
            _provision_login(
                connection,
                ONLINE_WORKER,
                urls[ONLINE_WORKER].password or "",
                membership=ONLINE_ROLE,
            )
            connection.execute(
                sql.SQL("GRANT {} TO {} WITH INHERIT TRUE").format(
                    sql.Identifier(ONLINE_ROLE),
                    sql.Identifier(ONLINE_WORKER),
                )
            )
            connection.execute(
                sql.SQL("GRANT {} TO {} WITH SET FALSE").format(
                    sql.Identifier(ONLINE_ROLE),
                    sql.Identifier(ONLINE_WORKER),
                )
            )
            grants = connection.execute(
                "SELECT parent.rolname, m.admin_option, m.inherit_option, m.set_option "
                "FROM pg_auth_members m JOIN pg_roles parent ON parent.oid=m.roleid "
                "JOIN pg_roles worker ON worker.oid=m.member WHERE worker.rolname=%s",
                (ONLINE_WORKER,),
            ).fetchall()
            if grants != [
                {
                    "rolname": ONLINE_ROLE,
                    "admin_option": False,
                    "inherit_option": True,
                    "set_option": False,
                }
            ]:
                raise _PreflightFailure("online_worker_membership_mismatch")
            source = migrations.get_revision("20260920_0030")
            if source is None:
                raise _PreflightFailure("online_boundary_migration_missing")
            additional_views = tuple(
                view
                for revision in migrations.walk_revisions(
                    base="20260920_0030", head=TARGET_REVISION,
                )
                for view in getattr(revision.module, "_ONLINE_VIEW_ADDITIONS", ())
            )
            for role in (ONLINE_ROLE, ONLINE_WORKER):
                _verify_online_boundary(
                    connection, role, source.module, additional_views=additional_views,
                )
        report["online_worker"] = "login_inherit_online_only"
        for role in ("pmmc_runtime", "pmmc_evaluator", ONLINE_WORKER):
            with _connect(urls[role]) as login:
                rights = login.execute(
                    "SELECT current_user AS identity, "
                    "has_schema_privilege(current_user,'pmmc_runtime','USAGE') AS runtime_usage, "
                    "has_schema_privilege(current_user,'pmmc_evaluation','USAGE') "
                    "AS evaluation_usage, "
                    "has_schema_privilege(current_user,'public','CREATE') AS public_create"
                ).fetchone()
                if (
                    rights is None
                    or rights["identity"] != role
                    or not rights["runtime_usage"]
                    or rights["evaluation_usage"] != (role == "pmmc_evaluator")
                    or rights["public_create"]
                ):
                    raise _PreflightFailure("database_role_login_boundary_failed")
        report["role_logins_verified"] = True
        report["ready"] = True


def _s3_client(settings: Settings, role: str):
    kwargs: dict[str, Any] = {
        "endpoint_url": settings.object_store_endpoint,
        "region_name": settings.object_store_region,
    }
    if role != "anonymous":
        prefix = {
            "admin": "object_store_admin",
            "runtime": "object_store",
            "evaluator": "object_store_evaluator",
        }[role]
        kwargs.update(
            aws_access_key_id=_secret(settings, f"{prefix}_access_key"),
            aws_secret_access_key=_secret(settings, f"{prefix}_secret_key"),
        )
    kwargs["config"] = S3Config(
        signature_version=UNSIGNED if role == "anonymous" else "s3v4",
        connect_timeout=10,
        read_timeout=30,
        retries={"max_attempts": 0},
        s3={"addressing_style": "path" if settings.object_store_path_style else "virtual"},
    )
    return boto3.client("s3", **kwargs)


def _error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", "unknown"))


def _ensure_bucket(client, bucket: str, region: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        if _error_code(exc) not in _MISSING:
            raise _PreflightFailure("object_store_admin_bucket_preflight_failed") from None
    kwargs: dict[str, Any] = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        client.create_bucket(**kwargs)
    except ClientError as exc:
        if _error_code(exc) != "BucketAlreadyOwnedByYou":
            raise _PreflightFailure("object_store_bucket_creation_failed") from None
    client.head_bucket(Bucket=bucket)


def _read_probe(client, bucket: str, key: str, payload: bytes) -> None:
    result = client.get_object(Bucket=bucket, Key=key)
    body = result["Body"]
    try:
        if body.read(len(payload) + 1) != payload:
            raise _PreflightFailure("object_store_probe_content_mismatch")
    finally:
        body.close()


def _expect_denied(operation, **kwargs) -> None:
    try:
        result = operation(**kwargs)
    except ClientError as exc:
        if _error_code(exc) in _DENIED:
            return
        code = (
            "object_store_iam_preflight_failed"
            if _error_code(exc) in _IAM_ERRORS
            else "object_store_probe_service_failed"
        )
        raise _PreflightFailure(code) from None
    if isinstance(result, dict) and "Body" in result:
        result["Body"].close()
    raise _PreflightFailure("object_store_role_boundary_failed")


def _remove_probe_versions(admin, bucket: str, key: str) -> None:
    # A random, invocation-owned key has only a bounded number of versions. Do
    # not turn cleanup into an unbounded scan or delete neighboring prefixes.
    page = admin.list_object_versions(Bucket=bucket, Prefix=key, MaxKeys=1000)
    if page.get("IsTruncated"):
        raise _PreflightFailure("object_store_probe_cleanup_truncated")
    for item in (*page.get("Versions", []), *page.get("DeleteMarkers", [])):
        if item["Key"] == key:
            admin.delete_object(Bucket=bucket, Key=key, VersionId=item["VersionId"])


def _object_store(settings: Settings, report: dict[str, Any]) -> None:
    clients = {
        role: _s3_client(settings, role) for role in ("admin", "runtime", "evaluator", "anonymous")
    }
    admin, runtime, evaluator, anonymous = (clients[role] for role in clients)
    rb, eb = settings.object_store_bucket, settings.object_store_evaluation_bucket
    keys = {bucket: f"step8-bootstrap-probes/{uuid4().hex}/probe" for bucket in (rb, eb)}
    payload = b"pmmc-step8-bootstrap-v1"
    try:
        for bucket in (rb, eb):
            _ensure_bucket(admin, bucket, settings.object_store_region)
        report["buckets_ready"] = True
        try:
            for bucket, client in ((rb, runtime), (eb, evaluator)):
                client.head_bucket(Bucket=bucket)
                client.list_objects_v2(Bucket=bucket, Prefix=keys[bucket], MaxKeys=1)
                client.put_object(Bucket=bucket, Key=keys[bucket], Body=payload)
                _read_probe(client, bucket, keys[bucket], payload)
            _read_probe(evaluator, rb, keys[rb], payload)
            evaluator.list_objects_v2(Bucket=rb, Prefix=keys[rb], MaxKeys=1)
            _expect_denied(runtime.head_bucket, Bucket=eb)
            _expect_denied(runtime.get_object, Bucket=eb, Key=keys[eb])
            _expect_denied(runtime.list_objects_v2, Bucket=eb, Prefix=keys[eb], MaxKeys=1)
            _expect_denied(evaluator.put_object, Bucket=rb, Key=keys[rb] + "-denied", Body=payload)
            _expect_denied(runtime.put_object, Bucket=eb, Key=keys[eb] + "-denied", Body=payload)
            for bucket in (rb, eb):
                _expect_denied(
                    anonymous.list_objects_v2, Bucket=bucket, Prefix=keys[bucket], MaxKeys=1
                )
                _expect_denied(anonymous.get_object, Bucket=bucket, Key=keys[bucket])
                _expect_denied(
                    anonymous.put_object, Bucket=bucket, Key=keys[bucket] + "-denied", Body=payload
                )
            for bucket, client in ((rb, runtime), (eb, evaluator)):
                _expect_denied(client.delete_object, Bucket=bucket, Key=keys[bucket])
        except ClientError as exc:
            code = (
                "object_store_iam_preflight_failed"
                if _error_code(exc) in _IAM_ERRORS
                else "object_store_probe_service_failed"
            )
            raise _PreflightFailure(code) from None
        report["credential_probe_preflight"] = "verified"
    finally:
        original_error = sys.exception()
        cleanup_failed = False
        if report.get("buckets_ready"):
            for bucket in (rb, eb):
                for key in (keys[bucket], keys[bucket] + "-denied"):
                    try:
                        _remove_probe_versions(admin, bucket, key)
                    except Exception:
                        cleanup_failed = True
            report["probe_cleanup"] = "failed" if cleanup_failed else "completed"
        for client in clients.values():
            client.close()
        if cleanup_failed and original_error is None:
            raise _PreflightFailure("object_store_probe_cleanup_failed") from None
    report["ready"] = True


def bootstrap(settings: Settings) -> dict[str, Any]:
    """Initialize owned roles/schema/buckets; return ready only after real role checks.

    The CLI should serialize the returned report, or ``Step8BootstrapError.report``
    and exit nonzero on failure. Do not serialize Settings or the original error.
    """
    report: dict[str, Any] = {
        "protocol": "step8-bootstrap-v1",
        "ready": False,
        "postgres": {"ready": False},
        "object_store": {"ready": False, "iam_management": "external_not_modified"},
        "external_prerequisites": [],
        "model_calls": 0,
    }
    phase = "configuration"
    try:
        urls = validate_settings(settings)
        phase = "postgres"
        _postgres(settings, urls, report["postgres"])
        phase = "object_store"
        _object_store(settings, report["object_store"])
    except Exception as exc:
        code = str(exc) if isinstance(exc, _PreflightFailure) else f"{phase}_operation_failed"
        report.update(failed_phase=phase, error_code=code)
        if code in {"object_store_iam_preflight_failed", "object_store_role_boundary_failed"}:
            report["external_prerequisites"] = [IAM_PREREQUISITE]
        raise Step8BootstrapError(code, report) from None
    report["ready"] = True
    return report
