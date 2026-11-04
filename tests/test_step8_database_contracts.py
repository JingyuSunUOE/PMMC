"""Capture a public migration's upgrade statements without a database connection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from alembic.script import ScriptDirectory


def _migration(number: int):
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[1] / "alembic"))
    (revision,) = [r for r in scripts.walk_revisions() if r.revision.endswith(f"_{number:04d}")]
    return revision.module


def _statements(number: int) -> list[str]:
    migration = _migration(number)
    statements: list[str] = []
    with patch.object(migration, "op", SimpleNamespace(execute=statements.append)):
        migration.upgrade()
    return statements
