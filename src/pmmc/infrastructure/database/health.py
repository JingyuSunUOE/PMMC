"""Non-sensitive Postgres connectivity and privilege diagnostics."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from pmmc.infrastructure.database.constants import (
    EVALUATION_ROLE,
    EVALUATION_SCHEMA,
    OWNER_ROLE,
    RUNTIME_ROLE,
    RUNTIME_SCHEMA,
    DatabaseRole,
)
from pmmc.infrastructure.database.errors import PersistenceError

EXPECTED_ROLE = {
    DatabaseRole.RUNTIME: RUNTIME_ROLE,
    DatabaseRole.EVALUATOR: EVALUATION_ROLE,
    DatabaseRole.ADMIN: OWNER_ROLE,
    DatabaseRole.ONLINE: "pmmc_online_worker",
}

# These relations contain compiler answers, private payloads, or artifact locators.
# Check column grants too: a table-level SELECT check alone misses those grants.
_ONLINE_PRIVATE_TABLES = (
    "run_manifests",
    "run_artifacts",
    "model_calls",
    "qbank_questions",
    "qbank_expected_evidence_ids",
    "qbank_expected_image_ids",
    "memory_programs",
    "planner_requirement_contracts",
    "doubter_feedback",
    "doubter_assessments",
    "questioner_candidate_verifications",
)


async def _check_online_boundary(connection: Any) -> dict[str, Any]:
    row = (
        (
            await connection.execute(
                text("""
        SELECT
          pg_has_role(current_user, 'pmmc_online', 'USAGE') AS online_role_inherited,
          EXISTS (SELECT 1 FROM pg_roles r WHERE r.rolname=current_user
                    AND (NOT r.rolcanlogin OR NOT r.rolinherit OR r.rolsuper
                         OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication
                         OR r.rolbypassrls))
          OR EXISTS (SELECT 1 FROM pg_roles r
                      WHERE r.rolname NOT IN (current_user, 'pmmc_online')
                        AND pg_has_role(current_user, r.oid, 'MEMBER'))
            AS unsafe_online_role,
          EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                   WHERE c.relkind IN ('r','p','v','m','f')
                     AND (n.nspname=:evaluation_schema OR
                          (n.nspname=:runtime_schema AND
                            (c.relname=ANY(:private_tables)
                             OR NOT has_table_privilege('pmmc_online',c.oid,'SELECT'))))
                     AND (has_table_privilege(current_user,c.oid,'SELECT')
                          OR has_any_column_privilege(current_user,c.oid,'SELECT')))
            AS online_private_access
        """),
                {
                    "evaluation_schema": EVALUATION_SCHEMA,
                    "runtime_schema": RUNTIME_SCHEMA,
                    "private_tables": list(_ONLINE_PRIVATE_TABLES),
                },
            )
        )
        .mappings()
        .one()
    )
    if not row["online_role_inherited"] or row["unsafe_online_role"]:
        raise PersistenceError("online login must inherit only the restricted pmmc_online role")
    if row["online_private_access"]:
        raise PersistenceError("online login unexpectedly has private compiler/evaluation access")
    return dict(row)


async def check_database_connectivity(
    engine: AsyncEngine,
    role: DatabaseRole,
) -> dict[str, Any]:
    """Check connection identity and UTC before schemas/extensions exist."""

    async with engine.connect() as connection:
        row = (
            (
                await connection.execute(
                    text(
                        "SELECT current_user AS current_user, "
                        "current_database() AS current_database, "
                        "current_setting('TimeZone') AS timezone"
                    )
                )
            )
            .mappings()
            .one()
        )
    expected = EXPECTED_ROLE[role]
    if row["current_user"] != expected:
        raise PersistenceError(
            f"database role mismatch: expected {expected}, got {row['current_user']}"
        )
    if str(row["timezone"]).upper() not in {"UTC", "ETC/UTC"}:
        raise PersistenceError("Postgres timezone must be UTC")
    return dict(row)


async def check_database(engine: AsyncEngine, role: DatabaseRole) -> dict[str, Any]:
    """Check identity, UTC, pgvector, schemas, and the gold access boundary."""

    identity = await check_database_connectivity(engine, role)
    async with engine.connect() as connection:
        row = (
            (
                await connection.execute(
                    text(
                        "SELECT EXISTS "
                        "(SELECT 1 FROM pg_extension WHERE extname = 'vector') "
                        "AS vector_enabled, "
                        "has_schema_privilege(current_user, :runtime_schema, 'USAGE') "
                        "AS runtime_schema_access, "
                        "has_schema_privilege(current_user, :evaluation_schema, 'USAGE') "
                        "AS evaluation_schema_access"
                    ),
                    {
                        "runtime_schema": RUNTIME_SCHEMA,
                        "evaluation_schema": EVALUATION_SCHEMA,
                    },
                )
            )
            .mappings()
            .one()
        )
        online = await _check_online_boundary(connection) if role is DatabaseRole.ONLINE else {}
    if not row["runtime_schema_access"]:
        raise PersistenceError("configured role cannot access the runtime schema")
    runtime_only = role in {DatabaseRole.RUNTIME, DatabaseRole.ONLINE}
    if runtime_only and row["evaluation_schema_access"]:
        raise PersistenceError("runtime/online role unexpectedly has evaluation schema access")
    if not runtime_only and not row["evaluation_schema_access"]:
        raise PersistenceError("owner/evaluator role cannot access the evaluation schema")
    if not row["vector_enabled"]:
        raise PersistenceError("pgvector extension is not enabled")
    return {**identity, **dict(row), **online}
