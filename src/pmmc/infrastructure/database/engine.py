"""Role-aware asynchronous SQLAlchemy engine construction."""

from __future__ import annotations

from pydantic import SecretStr
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.errors import DatabaseConfigurationError
from pmmc.infrastructure.database.work_limit import CampaignQueuePool
from pmmc.infrastructure.settings import Settings


def normalize_async_database_url(value: str) -> str:
    """Normalize a PostgreSQL URL to SQLAlchemy's psycopg async dialect."""

    url = make_url(value.strip())
    if url.drivername in {"postgres", "postgresql", "postgresql+psycopg"}:
        return url.set(drivername="postgresql+psycopg").render_as_string(hide_password=False)
    raise DatabaseConfigurationError("database URL must use PostgreSQL with the psycopg driver")


def _configured_url(settings: Settings, role: DatabaseRole) -> SecretStr | None:
    if role is DatabaseRole.ONLINE:
        return settings.database_online_url
    if role is DatabaseRole.RUNTIME:
        return settings.database_url
    if role is DatabaseRole.EVALUATOR:
        return settings.database_evaluator_url
    return settings.database_admin_url


def database_url_for_role(settings: Settings, role: DatabaseRole) -> str:
    """Return one normalized DSN without exposing it through diagnostics."""

    configured = _configured_url(settings, role)
    if configured is None or not configured.get_secret_value().strip():
        raise DatabaseConfigurationError(f"database URL for role '{role.value}' is not configured")
    return normalize_async_database_url(configured.get_secret_value())


def create_engine(settings: Settings, role: DatabaseRole) -> AsyncEngine:
    """Create a role-scoped async engine with conservative pool defaults."""

    return create_async_engine(
        database_url_for_role(settings, role),
        poolclass=CampaignQueuePool,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout_seconds,
    )


def create_session_factory(
    settings: Settings,
    role: DatabaseRole,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create an engine and non-expiring async session factory."""

    engine = create_engine(settings, role)
    return engine, async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
