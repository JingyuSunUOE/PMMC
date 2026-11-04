# ruff: noqa: E501
"""Allow Step 6D program query embeddings through the lineage guard.

Revision ID: 20260726_0019
Revises: 20260725_0018
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260726_0019"
down_revision: str | None = "20260725_0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"


def _migration_event(*, state: str, protocol: str) -> None:
    print(
        "STEP6D_MIGRATION "
        f"phase=query_embedding_lineage state={state} protocol={protocol}",
        flush=True,
    )


def upgrade() -> None:
    _migration_event(state="start", protocol="v4")
    _replace_embedding_lineage_guard(enable_verified_qbank=True)
    _migration_event(state="done", protocol="v4")


def downgrade() -> None:
    _migration_event(state="start", protocol="v3")
    _replace_embedding_lineage_guard(enable_verified_qbank=False)
    _migration_event(state="done", protocol="v3")


def _replace_embedding_lineage_guard(*, enable_verified_qbank: bool) -> None:
    function_name = (
        "validate_compiler_embedding_lineage_v4"
        if enable_verified_qbank
        else "validate_compiler_embedding_lineage_v3"
    )
    step6d_clause = (
        "OR (compiler.compiler_schema_version = '0.6.0' "
        "AND compiler.target_stage = 'verified_qbank')"
        if enable_verified_qbank
        else ""
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_qbank_embedding ON {RUNTIME}.embeddings"
    )
    for candidate in (
        "validate_compiler_embedding_lineage_v3",
        "validate_compiler_embedding_lineage_v4",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{candidate}()")
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.{function_name}()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE source_kind varchar(64); collection_snapshot varchar(128);
                collection_run varchar(128); collection_metadata jsonb; valid boolean;
        BEGIN
            SELECT source_type, snapshot_id, run_id, metadata
              INTO source_kind, collection_snapshot, collection_run, collection_metadata
              FROM {RUNTIME}.embedding_collections
             WHERE collection_id = NEW.collection_id;
            IF source_kind = 'qbank_question' THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id
                   AND snapshot_id = collection_snapshot
                   AND run_id = collection_run
                   AND NEW.source_id = qbank_id
                   AND NEW.paraphrase_id IS NULL;
            ELSIF source_kind = 'qbank_paraphrase' THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_paraphrases
                 WHERE paraphrase_id = NEW.paraphrase_id
                   AND snapshot_id = collection_snapshot
                   AND run_id = collection_run
                   AND NEW.source_id = paraphrase_id
                   AND NEW.qbank_id IS NULL;
            ELSIF source_kind = 'query'
              AND collection_metadata->>'query_role' IN (
                  'question_text_query', 'question_joint_query'
              )
              AND collection_metadata->>'expected_images_allowed' = 'false'
              AND EXISTS (
                  SELECT 1 FROM {RUNTIME}.compiler_runs compiler
                   WHERE compiler.run_id = collection_run
                     AND compiler.snapshot_id = collection_snapshot
                     AND (
                       (compiler.compiler_schema_version IN ('0.4.0', '0.5.0')
                        AND compiler.target_stage = 'programs')
                       {step6d_clause}
                     )
              ) THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id
                   AND snapshot_id = collection_snapshot
                   AND run_id = collection_run
                   AND NEW.source_id = qbank_id
                   AND NEW.paraphrase_id IS NULL;
            ELSE
                valid := NEW.qbank_id IS NULL AND NEW.paraphrase_id IS NULL;
            END IF;
            IF COALESCE(valid, false) = false THEN
                RAISE EXCEPTION 'embedding typed source does not match collection source type'
                    USING ERRCODE = '23514',
                          SCHEMA = '{RUNTIME}',
                          TABLE = 'embeddings',
                          CONSTRAINT = 'ck_embeddings_compiler_source_lineage';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER validate_qbank_embedding BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.embeddings FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.{function_name}()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.{function_name}() FROM PUBLIC"
    )
