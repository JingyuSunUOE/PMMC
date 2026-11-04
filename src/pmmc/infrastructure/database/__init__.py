"""Postgres persistence adapters for canonical PMMC data.

Modules that require SQLAlchemy are intentionally not imported here. This keeps
schema- and benchmark-only commands usable in lightweight development installs.
"""

from pmmc.infrastructure.database.constants import (
    EVALUATION_ROLE,
    EVALUATION_SCHEMA,
    OWNER_ROLE,
    RUNTIME_ROLE,
    RUNTIME_SCHEMA,
    DatabaseRole,
    SnapshotImportState,
)
from pmmc.infrastructure.database.errors import (
    DatabaseConfigurationError,
    ExperimentPersistenceError,
    InvalidStateTransitionError,
    PersistenceConflictError,
    PersistenceError,
    SnapshotImportError,
    SnapshotIntegrityError,
)

__all__ = [
    "EVALUATION_ROLE",
    "EVALUATION_SCHEMA",
    "OWNER_ROLE",
    "RUNTIME_ROLE",
    "RUNTIME_SCHEMA",
    "DatabaseConfigurationError",
    "DatabaseRole",
    "ExperimentPersistenceError",
    "InvalidStateTransitionError",
    "PersistenceError",
    "PersistenceConflictError",
    "SnapshotImportError",
    "SnapshotImportState",
    "SnapshotIntegrityError",
]
