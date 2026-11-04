"""Async Alembic environment for the PMMC owner database role."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from pmmc.infrastructure.database import (  # noqa: F401
    baseline_component_models,
    baseline_models,
    compiler_models,
    doubter_models,
    evaluation_models,
    experiment_models,
    models,
    query_executor_models,
    questioner_models,
    runtime_models,
    step8_models,
    step8_provider_control,
)
from pmmc.infrastructure.database.base import Base
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import database_url_for_role
from pmmc.infrastructure.settings import Settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _owner_url() -> str:
    configured = config.attributes.get("database_url")
    if configured is not None:
        return str(configured)
    return database_url_for_role(Settings(), DatabaseRole.ADMIN)


def run_migrations_offline() -> None:
    context.configure(
        url=_owner_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _owner_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
