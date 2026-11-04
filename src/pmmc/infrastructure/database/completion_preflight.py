"""Plan the fixed Step 7B completion guard queries before GPU startup.

PostgreSQL prepares PL/pgSQL statements lazily. Function creation alone cannot
detect missing columns in branches reached only at the end of a long pilot.
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

_GUARDS = {
    "enforce_step7b_compiler_completion": 8,
    "enforce_step7b_query_completion": 4,
}
_COUNT_STATEMENT = re.compile(r"SELECT count\(\*\) INTO (?:bad|eligible) (FROM .*?);", re.DOTALL)


def completion_guard_queries(name: str, definition: str) -> tuple[str, ...]:
    """Extract only the audited count-query grammar, never arbitrary SQL.

    This is a bounded adapter for these two migration-owned functions, not a
    general SQL parser. A changed statement count fails closed for review.
    PostgreSQL itself resolves and plans each resulting query with EXPLAIN.
    """
    expected = _GUARDS[name]
    statements = _COUNT_STATEMENT.findall(definition)
    if len(statements) != expected:
        raise RuntimeError("Step 7B completion preflight statement count changed")
    result = []
    for statement in statements:
        statement = statement.replace("NEW.run_id", "CAST(NULL AS varchar)")
        statement = statement.replace("NEW.query_executor_run_id", "CAST(NULL AS varchar)")
        if "NEW." in statement:
            raise RuntimeError("Step 7B completion preflight has an unknown input")
        result.append("EXPLAIN SELECT count(*) " + statement)
    return tuple(result)


async def verify_step7b_completion_sql(connection: AsyncConnection) -> None:
    """No fact scans, writes, model calls or scientific acceptance decisions."""
    for name in _GUARDS:
        definition = await connection.scalar(
            text("SELECT pg_get_functiondef(to_regprocedure(:name))"),
            {"name": f"pmmc_runtime.{name}()"},
        )
        if not isinstance(definition, str):
            raise RuntimeError("Step 7B completion preflight function is missing")
        statements = completion_guard_queries(name, definition)
        for ordinal, statement in enumerate(statements, 1):
            try:
                await connection.exec_driver_sql(
                    statement, execution_options={"no_parameters": True}
                )
            except Exception:
                print(
                    "STEP7B_COMPLETION_SQL_PREFLIGHT "
                    f"guard={name} statement={ordinal}/{len(statements)} state=failed",
                    flush=True,
                )
                raise
        print(
            "STEP7B_COMPLETION_SQL_PREFLIGHT "
            f"guard={name} statements={len(statements)} state=ready fact_scan=false",
            flush=True,
        )
