"""Read-only PostgreSQL ``simple`` FTS relation selection for Questioner v3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import psycopg
from pydantic import SecretStr
from sqlalchemy.engine import make_url

from pmmc.infrastructure.settings import Settings

POSTGRES_SIMPLE_FTS_SELECTOR_VERSION = "postgres-simple-fts-v1"


@dataclass(frozen=True, slots=True)
class PostgresFtsHit:
    memory_id: str
    score: float


def _runtime_sync_dsn(settings: Settings) -> str:
    configured = settings.database_url
    if not isinstance(configured, SecretStr) or not configured.get_secret_value().strip():
        raise ValueError("Questioner v3 PostgreSQL FTS requires PMMC_DATABASE_URL")
    url = make_url(configured.get_secret_value().strip())
    if url.drivername not in {
        "postgres",
        "postgresql",
        "postgresql+psycopg",
    }:
        raise ValueError("Questioner v3 PostgreSQL FTS requires a PostgreSQL DSN")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


class PostgresSimpleFtsRelationProvider:
    """Select prospective support memories with the canonical generated tsvector."""

    selector_version = POSTGRES_SIMPLE_FTS_SELECTOR_VERSION

    def __init__(self, settings: Settings) -> None:
        self._dsn = _runtime_sync_dsn(settings)
        self._connection: psycopg.Connection[tuple[object, ...]] | None = None

    def _connect(self) -> psycopg.Connection[tuple[object, ...]]:
        if self._connection is None or self._connection.closed:
            self._connection = psycopg.connect(self._dsn, autocommit=True)
            with self._connection.cursor() as cursor:
                cursor.execute("SET default_transaction_read_only = on")
        return self._connection

    def close(self) -> None:
        if self._connection is not None and not self._connection.closed:
            self._connection.close()

    def search(
        self,
        *,
        snapshot_id: str,
        conversation_id: str,
        cutoff_session_id: str,
        anchor_memory_ids: Sequence[str],
        candidate_memory_ids: Sequence[str],
        lexeme_limit: int,
        top_k: int,
    ) -> list[PostgresFtsHit]:
        """Return stable FTS hits without allowing future or foreign memories."""

        anchors = list(dict.fromkeys(anchor_memory_ids))
        candidates = list(dict.fromkeys(candidate_memory_ids))
        if not anchors or not candidates:
            return []
        visible_ids = list(dict.fromkeys([*anchors, *candidates]))
        connection = self._connect()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH cutoff AS (
                    SELECT session_order
                    FROM pmmc_runtime.benchmark_sessions
                    WHERE snapshot_id = %s
                      AND conversation_id = %s
                      AND session_id = %s
                ),
                visible AS (
                    SELECT memory.memory_id, memory.search_document
                    FROM pmmc_runtime.memory_units AS memory
                    JOIN pmmc_runtime.benchmark_sessions AS available
                      ON available.snapshot_id = memory.snapshot_id
                     AND available.conversation_id = memory.conversation_id
                     AND available.session_id = memory.available_after_session
                    CROSS JOIN cutoff
                    WHERE memory.snapshot_id = %s
                      AND memory.conversation_id = %s
                      AND memory.memory_id = ANY(%s::text[])
                      AND available.session_order <= cutoff.session_order
                ),
                lexemes AS (
                    SELECT visible.memory_id,
                           unnest(tsvector_to_array(visible.search_document)) AS lexeme
                    FROM visible
                ),
                frequencies AS (
                    SELECT lexeme, count(DISTINCT memory_id) AS document_frequency
                    FROM lexemes
                    GROUP BY lexeme
                )
                SELECT frequencies.lexeme
                FROM frequencies
                WHERE EXISTS (
                    SELECT 1
                    FROM lexemes AS anchor_lexeme
                    WHERE anchor_lexeme.lexeme = frequencies.lexeme
                      AND anchor_lexeme.memory_id = ANY(%s::text[])
                )
                ORDER BY frequencies.document_frequency,
                         length(frequencies.lexeme) DESC,
                         frequencies.lexeme
                LIMIT %s
                """,
                (
                    snapshot_id,
                    conversation_id,
                    cutoff_session_id,
                    snapshot_id,
                    conversation_id,
                    visible_ids,
                    anchors,
                    lexeme_limit,
                ),
            )
            lexemes = [str(row[0]) for row in cursor.fetchall()]
            if not lexemes:
                return []
            cursor.execute(
                """
                WITH cutoff AS (
                    SELECT session_order
                    FROM pmmc_runtime.benchmark_sessions
                    WHERE snapshot_id = %s
                      AND conversation_id = %s
                      AND session_id = %s
                ),
                query AS (
                    SELECT to_tsquery(
                        'simple',
                        string_agg(quote_literal(value), ' | ' ORDER BY ordinal)
                    ) AS value
                    FROM unnest(%s::text[]) WITH ORDINALITY AS selected(value, ordinal)
                )
                SELECT memory.memory_id,
                       ts_rank_cd(memory.search_document, query.value) AS score
                FROM pmmc_runtime.memory_units AS memory
                JOIN pmmc_runtime.benchmark_sessions AS available
                  ON available.snapshot_id = memory.snapshot_id
                 AND available.conversation_id = memory.conversation_id
                 AND available.session_id = memory.available_after_session
                CROSS JOIN cutoff
                CROSS JOIN query
                WHERE memory.snapshot_id = %s
                  AND memory.conversation_id = %s
                  AND memory.memory_id = ANY(%s::text[])
                  AND available.session_order <= cutoff.session_order
                  AND memory.search_document @@ query.value
                ORDER BY score DESC, memory.memory_id
                LIMIT %s
                """,
                (
                    snapshot_id,
                    conversation_id,
                    cutoff_session_id,
                    lexemes,
                    snapshot_id,
                    conversation_id,
                    candidates,
                    top_k,
                ),
            )
            return [
                PostgresFtsHit(memory_id=str(row[0]), score=float(str(row[1])))
                for row in cursor.fetchall()
            ]
