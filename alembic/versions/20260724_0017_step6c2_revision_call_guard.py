# ruff: noqa: E501
"""Repair the Questioner v3.2 revision call lineage guard.

Revision ID: 20260724_0017
Revises: 20260724_0016
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260724_0017"
down_revision: str | None = "20260724_0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
_TRIGGER_NAME = "validate_questioner_v3_candidate"
_FUNCTION_NAME = "enforce_questioner_candidate_scope_v5"
_PREVIOUS_FUNCTION_NAME = "enforce_questioner_candidate_scope_v4"


def upgrade() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {_TRIGGER_NAME} ON {RUNTIME}.qbank_questions")
    _create_candidate_scope_guard()
    op.execute(
        f"CREATE TRIGGER {_TRIGGER_NAME} BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.qbank_questions FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.{_FUNCTION_NAME}()"
    )
    op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.{_FUNCTION_NAME}() FROM PUBLIC")


def downgrade() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {_TRIGGER_NAME} ON {RUNTIME}.qbank_questions")
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{_FUNCTION_NAME}()")
    op.execute(
        f"CREATE TRIGGER {_TRIGGER_NAME} BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.qbank_questions FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.{_PREVIOUS_FUNCTION_NAME}()"
    )


def _create_candidate_scope_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.{_FUNCTION_NAME}()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE opportunity_row record; producer_call record; source_row record;
                verifier_call record; generation_row record; generation_call record;
                compiler_version text; producer_task text; expected_scope text;
        BEGIN
            IF NEW.opportunity_id IS NULL THEN
                RETURN NEW;
            END IF;

            SELECT * INTO opportunity_row
              FROM {RUNTIME}.questioner_opportunities
             WHERE opportunity_id = NEW.opportunity_id
               AND run_id = NEW.run_id;
            SELECT * INTO producer_call
              FROM {RUNTIME}.model_calls
             WHERE model_call_id = NEW.producer_model_call_id
               AND run_id = NEW.run_id;
            SELECT compiler_schema_version INTO compiler_version
              FROM {RUNTIME}.compiler_runs
             WHERE run_id = NEW.run_id;
            producer_task := COALESCE(
              producer_call.request_config->'request_metadata'->>'task',
              ''
            );
            expected_scope := 'Q' || lpad(NEW.generation_rank::text, 2, '0');

            IF opportunity_row.opportunity_id IS NULL
               OR producer_call.model_call_id IS NULL
               OR opportunity_row.compiler_unit_id != NEW.compiler_unit_id
               OR opportunity_row.snapshot_id != NEW.snapshot_id
               OR opportunity_row.conversation_id != NEW.conversation_id
               OR producer_call.compiler_unit_id != NEW.compiler_unit_id
               OR producer_call.stage != 'questioner'
               OR producer_call.status != 'success' THEN
                RAISE EXCEPTION
                  'Questioner candidate has inconsistent producer lineage'
                  USING ERRCODE = '23514',
                        SCHEMA = '{RUNTIME}',
                        TABLE = 'qbank_questions',
                        CONSTRAINT =
                          'ck_qbank_questions_questioner_producer_lineage';
            END IF;

            IF NEW.candidate_origin = 'generated' THEN
                IF producer_task != 'pmmc_questioner' THEN
                    RAISE EXCEPTION
                      'Questioner generated candidate has the wrong producer task'
                      USING ERRCODE = '23514',
                            SCHEMA = '{RUNTIME}',
                            TABLE = 'qbank_questions',
                            CONSTRAINT =
                              'ck_qbank_questions_generated_producer_task';
                END IF;
                RETURN NEW;
            END IF;

            SELECT * INTO source_row
              FROM {RUNTIME}.qbank_questions
             WHERE qbank_id = NEW.supersedes_qbank_id
               AND run_id = NEW.run_id;
            IF source_row.qbank_id IS NULL
               OR source_row.compiler_unit_id != NEW.compiler_unit_id
               OR source_row.generation_id != NEW.generation_id
               OR source_row.generation_rank != NEW.generation_rank
               OR source_row.candidate_version != 0 THEN
                RAISE EXCEPTION
                  'Questioner revised candidate source lineage is invalid'
                  USING ERRCODE = '23514',
                        SCHEMA = '{RUNTIME}',
                        TABLE = 'qbank_questions',
                        CONSTRAINT =
                          'ck_qbank_questions_revision_source_lineage';
            END IF;

            IF compiler_version != '0.5.0' THEN
                IF producer_task != 'pmmc_questioner_verifier' THEN
                    RAISE EXCEPTION
                      'Legacy Questioner revised candidate has the wrong producer task'
                      USING ERRCODE = '23514',
                            SCHEMA = '{RUNTIME}',
                            TABLE = 'qbank_questions',
                            CONSTRAINT =
                              'ck_qbank_questions_legacy_revision_task';
                END IF;
                RETURN NEW;
            END IF;

            SELECT * INTO verifier_call
              FROM {RUNTIME}.model_calls
             WHERE model_call_id = producer_call.parent_model_call_id
               AND run_id = NEW.run_id;
            SELECT * INTO generation_row
              FROM {RUNTIME}.questioner_generations
             WHERE generation_id = NEW.generation_id
               AND run_id = NEW.run_id;
            SELECT * INTO generation_call
              FROM {RUNTIME}.model_calls
             WHERE model_call_id = generation_row.model_call_id
               AND run_id = NEW.run_id;

            IF producer_task != 'pmmc_questioner_revision'
               OR COALESCE(
                    producer_call.request_config->'request_metadata'
                      ->>'candidate_scope',
                    ''
                  ) != expected_scope
               OR verifier_call.model_call_id IS NULL
               OR generation_row.generation_id IS NULL
               OR generation_call.model_call_id IS NULL
               OR verifier_call.stage != 'questioner'
               OR verifier_call.status != 'success'
               OR COALESCE(
                    verifier_call.request_config->'request_metadata'->>'task',
                    ''
                  ) != 'pmmc_questioner_verifier'
               OR COALESCE(
                    verifier_call.request_config->'request_metadata'
                      ->>'candidate_scope',
                    ''
                  ) != expected_scope
               OR verifier_call.parent_model_call_id != generation_call.model_call_id
               OR verifier_call.compiler_unit_id != NEW.compiler_unit_id
               OR generation_call.compiler_unit_id != NEW.compiler_unit_id
               OR generation_call.stage != 'questioner'
               OR generation_call.status != 'success'
               OR COALESCE(
                    generation_call.request_config->'request_metadata'->>'task',
                    ''
                  ) != 'pmmc_questioner'
               OR producer_call.job_id IS DISTINCT FROM verifier_call.job_id
               OR verifier_call.job_id IS DISTINCT FROM generation_call.job_id
               OR producer_call.model_alias != verifier_call.model_alias
               OR verifier_call.model_alias != generation_call.model_alias
               OR producer_call.actual_model_id
                    IS DISTINCT FROM verifier_call.actual_model_id
               OR verifier_call.actual_model_id
                    IS DISTINCT FROM generation_call.actual_model_id THEN
                RAISE EXCEPTION
                  'Questioner v3.2 revision call lineage is invalid'
                  USING ERRCODE = '23514',
                        SCHEMA = '{RUNTIME}',
                        TABLE = 'qbank_questions',
                        CONSTRAINT =
                          'ck_qbank_questions_v32_revision_call_lineage';
            END IF;
            RETURN NEW;
        END; $$"""
    )
