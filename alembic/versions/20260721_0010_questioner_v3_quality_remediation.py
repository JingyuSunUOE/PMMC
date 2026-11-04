# ruff: noqa: E501
"""Add Questioner v3 opportunity, verification, and revision lineage.

Revision ID: 20260721_0010
Revises: 20260720_0009
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260721_0010"
down_revision: str | None = "20260720_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"

_REJECTION_CODES = (
    "invalid_schema,unsupported_question_type,future_or_foreign_reference,"
    "missing_expected_answer,missing_expected_evidence,modality_mismatch,"
    "visual_evidence_missing,multi_hop_evidence_missing,duplicate,budget_exceeded,"
    "internal_identifier_leak,system_metadata_reference,assistant_follow_up,"
    "historical_question_replay,generic_external_question,placeholder_expected_answer,"
    "opportunity_mismatch,taxonomy_invariant_failed,reasoning_invariant_failed,"
    "evidence_not_entailed,semantic_revision_required,semantic_verification_failed,"
    "revision_exhausted"
).split(",")


def upgrade() -> None:
    _create_opportunity_tables()
    _extend_qbank_questions()
    _create_verification_tables()
    _replace_gate_constraint()
    _create_guards()
    _grant_permissions()


def _create_opportunity_tables() -> None:
    op.execute(
        f"""CREATE TABLE {RUNTIME}.questioner_opportunity_sets (
            opportunity_set_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL,
            builder_version varchar(128) NOT NULL,
            opportunity_count integer NOT NULL CHECK (opportunity_count BETWEEN 1 AND 24),
            content_hash varchar(128) NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (compiler_unit_id),
            UNIQUE (opportunity_set_id, run_id),
            FOREIGN KEY (compiler_unit_id, run_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.compiler_units(
                    compiler_unit_id, run_id, snapshot_id, conversation_id
                )
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.questioner_opportunities (
            opportunity_id varchar(128) PRIMARY KEY,
            opportunity_set_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            opportunity_type varchar(32) NOT NULL CHECK (opportunity_type IN (
                'local_fact','visual_detail','image_text_binding','visual_search',
                'temporal_pair','update_pair','conflict_pair','preference',
                'multi_evidence','refusal_boundary'
            )),
            allowed_question_types text[] NOT NULL CHECK (
                cardinality(allowed_question_types) > 0 AND allowed_question_types <@ ARRAY[
                    'textual_fact','visual_detail','image_text_binding','visual_search',
                    'multi_hop','temporal','knowledge_update','conflict_detection',
                    'answer_refusal','personal_preference']::text[]),
            allowed_reasoning_types text[] NOT NULL CHECK (
                cardinality(allowed_reasoning_types) > 0 AND allowed_reasoning_types <@ ARRAY[
                    'single_hop','multi_hop','temporal','update','conflict','refusal']::text[]),
            required_modalities text[] NOT NULL CHECK (
                cardinality(required_modalities) > 0
                AND required_modalities <@ ARRAY['text','image']::text[]),
            minimum_evidence_groups integer NOT NULL CHECK (minimum_evidence_groups > 0),
            evidence_aliases text[] NOT NULL,
            relation_link_ids text[] NOT NULL,
            content_hash varchar(128) NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (opportunity_set_id, ordinal),
            UNIQUE (opportunity_id, run_id),
            FOREIGN KEY (opportunity_set_id, run_id)
                REFERENCES {RUNTIME}.questioner_opportunity_sets(opportunity_set_id, run_id),
            FOREIGN KEY (compiler_unit_id, run_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.compiler_units(
                    compiler_unit_id, run_id, snapshot_id, conversation_id
                )
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.questioner_opportunity_evidence_refs (
            opportunity_evidence_ref_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            opportunity_id varchar(128) NOT NULL,
            alias varchar(32) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            evidence_id varchar(256) NOT NULL,
            evidence_type varchar(32) NOT NULL,
            memory_id varchar(256),
            image_id varchar(256),
            session_id varchar(256),
            caption_id varchar(256),
            evidence_group_id varchar(256) NOT NULL,
            content_hash varchar(128) NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (opportunity_id, ordinal),
            UNIQUE (opportunity_id, alias),
            FOREIGN KEY (opportunity_id, run_id)
                REFERENCES {RUNTIME}.questioner_opportunities(opportunity_id, run_id),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            CHECK (evidence_type IN ('memory','image','session','caption')),
            CHECK ((evidence_type = 'memory' AND memory_id IS NOT NULL
                    AND image_id IS NULL AND session_id IS NULL AND caption_id IS NULL)
                OR (evidence_type = 'caption' AND memory_id IS NOT NULL
                    AND image_id IS NULL AND session_id IS NULL)
                OR (evidence_type = 'image' AND image_id IS NOT NULL
                    AND memory_id IS NULL AND session_id IS NULL AND caption_id IS NULL)
                OR (evidence_type = 'session' AND session_id IS NOT NULL
                    AND memory_id IS NULL AND image_id IS NULL AND caption_id IS NULL)),
            CHECK ((evidence_type IN ('memory','caption') AND evidence_id = memory_id)
                OR (evidence_type = 'image' AND evidence_id = image_id)
                OR (evidence_type = 'session' AND evidence_id = session_id))
        )"""
    )


def _extend_qbank_questions() -> None:
    op.add_column(
        "qbank_questions",
        sa.Column("opportunity_id", sa.String(length=128), nullable=True),
        schema=RUNTIME,
    )
    op.add_column(
        "qbank_questions",
        sa.Column(
            "candidate_origin",
            sa.String(length=16),
            nullable=False,
            server_default="generated",
        ),
        schema=RUNTIME,
    )
    op.add_column(
        "qbank_questions",
        sa.Column("candidate_version", sa.Integer(), nullable=False, server_default="0"),
        schema=RUNTIME,
    )
    op.add_column(
        "qbank_questions",
        sa.Column("supersedes_qbank_id", sa.String(length=128), nullable=True),
        schema=RUNTIME,
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.qbank_questions DROP CONSTRAINT IF EXISTS qbank_questions_compiler_unit_id_generation_rank_key"
    )
    op.create_unique_constraint(
        "uq_qbank_questions_unit_rank_version",
        "qbank_questions",
        ["compiler_unit_id", "generation_rank", "candidate_version"],
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "ck_qbank_questions_candidate_origin",
        "qbank_questions",
        "candidate_origin IN ('generated','revised')",
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "ck_qbank_questions_candidate_version",
        "qbank_questions",
        "candidate_version IN (0,1) AND "
        "((candidate_origin = 'generated' AND candidate_version = 0 AND supersedes_qbank_id IS NULL) "
        "OR (candidate_origin = 'revised' AND candidate_version = 1 AND supersedes_qbank_id IS NOT NULL))",
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_qbank_questions_opportunity",
        "qbank_questions",
        "questioner_opportunities",
        ["opportunity_id", "run_id"],
        ["opportunity_id", "run_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_qbank_questions_supersedes",
        "qbank_questions",
        "qbank_questions",
        ["supersedes_qbank_id", "run_id"],
        ["qbank_id", "run_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.alter_column(
        "qbank_questions", "candidate_origin", server_default=None, schema=RUNTIME
    )
    op.alter_column(
        "qbank_questions", "candidate_version", server_default=None, schema=RUNTIME
    )


def _create_verification_tables() -> None:
    op.execute(
        f"""CREATE TABLE {RUNTIME}.questioner_verification_batches (
            verification_batch_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            generation_id varchar(128) NOT NULL,
            model_call_id varchar(128) NOT NULL UNIQUE,
            verification_round integer NOT NULL CHECK (verification_round IN (0,1)),
            prompt_version varchar(128) NOT NULL,
            schema_hash varchar(128) NOT NULL,
            input_count integer NOT NULL CHECK (input_count >= 0),
            pass_count integer NOT NULL CHECK (pass_count >= 0),
            revise_count integer NOT NULL CHECK (revise_count >= 0),
            reject_count integer NOT NULL CHECK (reject_count >= 0),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (compiler_unit_id, generation_id, verification_round),
            UNIQUE (verification_batch_id, run_id, compiler_unit_id,
                    generation_id, verification_round),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            FOREIGN KEY (generation_id, run_id)
                REFERENCES {RUNTIME}.questioner_generations(generation_id, run_id),
            FOREIGN KEY (model_call_id, run_id)
                REFERENCES {RUNTIME}.model_calls(model_call_id, run_id),
            CHECK (pass_count + revise_count + reject_count = input_count)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.questioner_candidate_verifications (
            candidate_verification_id varchar(128) PRIMARY KEY,
            verification_batch_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            generation_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            candidate_alias varchar(32) NOT NULL,
            verification_round integer NOT NULL CHECK (verification_round IN (0,1)),
            verdict varchar(16) NOT NULL CHECK (verdict IN ('pass','revise','reject')),
            reason_codes text[] NOT NULL,
            replacement_qbank_id varchar(128),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_id, verification_round),
            UNIQUE (verification_batch_id, candidate_alias),
            FOREIGN KEY (verification_batch_id, run_id, compiler_unit_id,
                         generation_id, verification_round)
                REFERENCES {RUNTIME}.questioner_verification_batches(
                    verification_batch_id, run_id, compiler_unit_id,
                    generation_id, verification_round
                ),
            FOREIGN KEY (qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            FOREIGN KEY (replacement_qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            CHECK ((verdict = 'pass' AND cardinality(reason_codes) = 0 AND replacement_qbank_id IS NULL)
                OR (verdict = 'revise' AND cardinality(reason_codes) > 0 AND replacement_qbank_id IS NOT NULL)
                OR (verdict = 'reject' AND cardinality(reason_codes) > 0 AND replacement_qbank_id IS NULL))
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.questioner_candidate_supersessions (
            supersession_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            generation_id varchar(128) NOT NULL,
            source_qbank_id varchar(128) NOT NULL,
            replacement_qbank_id varchar(128) NOT NULL,
            revision_round integer NOT NULL CHECK (revision_round = 1),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (source_qbank_id),
            UNIQUE (replacement_qbank_id),
            FOREIGN KEY (source_qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            FOREIGN KEY (replacement_qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            CHECK (source_qbank_id != replacement_qbank_id)
        )"""
    )


def _replace_gate_constraint() -> None:
    op.execute(
        f"ALTER TABLE {RUNTIME}.questioner_candidate_gate_results DROP CONSTRAINT IF EXISTS questioner_candidate_gate_results_rejection_codes_check"
    )
    values = ",".join(f"'{value}'" for value in _REJECTION_CODES)
    op.create_check_constraint(
        "ck_questioner_candidate_gate_results_rejection_codes_v3",
        "questioner_candidate_gate_results",
        f"rejection_codes <@ ARRAY[{values}]::text[]",
        schema=RUNTIME,
    )


def _create_guards() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE call_row record; generation_call record; generation_row record;
                source_row record; replacement_row record; opportunity_row record;
                question_row record; batch_row record;
        BEGIN
            IF TG_TABLE_NAME = 'questioner_opportunities' THEN
                IF EXISTS (
                    SELECT 1 FROM unnest(NEW.relation_link_ids) AS relation_link_id
                     WHERE NOT EXISTS (
                        SELECT 1 FROM {RUNTIME}.compiler_unit_links links
                         WHERE links.link_id = relation_link_id
                           AND links.run_id = NEW.run_id
                           AND links.compiler_unit_id = NEW.compiler_unit_id)) THEN
                    RAISE EXCEPTION 'Questioner opportunity relation crosses unit scope' USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_opportunity_evidence_refs' THEN
                SELECT * INTO opportunity_row FROM {RUNTIME}.questioner_opportunities
                 WHERE opportunity_id = NEW.opportunity_id AND run_id = NEW.run_id;
                IF opportunity_row.opportunity_id IS NULL
                   OR opportunity_row.compiler_unit_id != NEW.compiler_unit_id THEN
                    RAISE EXCEPTION 'Questioner opportunity evidence crosses unit scope' USING ERRCODE = '23514';
                END IF;
                IF NEW.memory_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.compiler_unit_memories members
                     WHERE members.compiler_unit_id = NEW.compiler_unit_id
                       AND members.run_id = NEW.run_id AND members.memory_id = NEW.memory_id) THEN
                    RAISE EXCEPTION 'Questioner opportunity references memory outside its unit' USING ERRCODE = '23514';
                END IF;
                IF NEW.image_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.compiler_unit_memories members
                    JOIN {RUNTIME}.memory_unit_images images
                      ON images.snapshot_id = members.snapshot_id
                     AND images.memory_id = members.memory_id
                     AND images.image_id = NEW.image_id
                     WHERE members.compiler_unit_id = NEW.compiler_unit_id
                       AND members.run_id = NEW.run_id) THEN
                    RAISE EXCEPTION 'Questioner opportunity references image outside its unit' USING ERRCODE = '23514';
                END IF;
                IF NEW.session_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.compiler_units units
                    JOIN {RUNTIME}.benchmark_sessions evidence_session
                      ON evidence_session.snapshot_id = units.snapshot_id
                     AND evidence_session.conversation_id = units.conversation_id
                     AND evidence_session.session_id = NEW.session_id
                    JOIN {RUNTIME}.benchmark_sessions cutoff_session
                      ON cutoff_session.snapshot_id = units.snapshot_id
                     AND cutoff_session.conversation_id = units.conversation_id
                     AND cutoff_session.session_id = units.available_after_session
                     WHERE units.compiler_unit_id = NEW.compiler_unit_id
                       AND units.run_id = NEW.run_id
                       AND evidence_session.session_order <= cutoff_session.session_order) THEN
                    RAISE EXCEPTION 'Questioner opportunity session is after the prospective cutoff' USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'qbank_questions' THEN
                IF NEW.opportunity_id IS NULL THEN RETURN NEW; END IF;
                SELECT * INTO opportunity_row FROM {RUNTIME}.questioner_opportunities
                 WHERE opportunity_id = NEW.opportunity_id AND run_id = NEW.run_id;
                SELECT * INTO call_row FROM {RUNTIME}.model_calls
                 WHERE model_call_id = NEW.producer_model_call_id AND run_id = NEW.run_id;
                IF opportunity_row.opportunity_id IS NULL
                   OR call_row.model_call_id IS NULL
                   OR opportunity_row.compiler_unit_id != NEW.compiler_unit_id
                   OR opportunity_row.snapshot_id != NEW.snapshot_id
                   OR opportunity_row.conversation_id != NEW.conversation_id
                   OR call_row.compiler_unit_id != NEW.compiler_unit_id
                   OR call_row.stage != 'questioner' OR call_row.status != 'success' THEN
                    RAISE EXCEPTION 'Questioner v3 candidate has inconsistent producer lineage' USING ERRCODE = '23514';
                END IF;
                IF NEW.candidate_origin = 'revised' THEN
                    SELECT * INTO source_row FROM {RUNTIME}.qbank_questions
                     WHERE qbank_id = NEW.supersedes_qbank_id AND run_id = NEW.run_id;
                    IF source_row.qbank_id IS NULL
                       OR source_row.compiler_unit_id != NEW.compiler_unit_id
                       OR source_row.generation_id != NEW.generation_id
                       OR source_row.generation_rank != NEW.generation_rank
                       OR source_row.candidate_version != 0
                       OR COALESCE(call_row.request_config->'request_metadata'->>'task', '')
                          != 'pmmc_questioner_verifier' THEN
                        RAISE EXCEPTION 'Questioner revised candidate lineage is invalid' USING ERRCODE = '23514';
                    END IF;
                ELSIF COALESCE(call_row.request_config->'request_metadata'->>'task', '')
                      != 'pmmc_questioner' THEN
                    RAISE EXCEPTION 'Questioner generated candidate has the wrong producer task' USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_verification_batches' THEN
                SELECT * INTO call_row FROM {RUNTIME}.model_calls
                 WHERE model_call_id = NEW.model_call_id AND run_id = NEW.run_id;
                SELECT * INTO generation_row FROM {RUNTIME}.questioner_generations
                 WHERE generation_id = NEW.generation_id AND run_id = NEW.run_id;
                SELECT * INTO generation_call FROM {RUNTIME}.model_calls
                 WHERE model_call_id = generation_row.model_call_id AND run_id = NEW.run_id;
                IF call_row.model_call_id IS NULL
                   OR generation_row.generation_id IS NULL
                   OR generation_call.model_call_id IS NULL
                   OR call_row.compiler_unit_id != NEW.compiler_unit_id
                   OR call_row.stage != 'questioner' OR call_row.status != 'success'
                   OR call_row.parent_model_call_id != generation_row.model_call_id
                   OR call_row.model_alias != generation_call.model_alias
                   OR call_row.actual_model_id IS DISTINCT FROM generation_call.actual_model_id
                   OR call_row.job_id IS DISTINCT FROM generation_call.job_id
                   OR NEW.prompt_version != 'pmmc-questioner-verifier-v1'
                   OR COALESCE(call_row.request_config->'request_metadata'->>'task', '') != 'pmmc_questioner_verifier' THEN
                    RAISE EXCEPTION 'Questioner verifier batch has inconsistent model-call lineage' USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_candidate_verifications' THEN
                SELECT * INTO batch_row FROM {RUNTIME}.questioner_verification_batches
                 WHERE verification_batch_id = NEW.verification_batch_id
                   AND run_id = NEW.run_id;
                SELECT * INTO question_row FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id AND run_id = NEW.run_id;
                IF batch_row.verification_batch_id IS NULL
                   OR question_row.qbank_id IS NULL
                   OR batch_row.compiler_unit_id != NEW.compiler_unit_id
                   OR batch_row.generation_id != NEW.generation_id
                   OR batch_row.verification_round != NEW.verification_round
                   OR question_row.compiler_unit_id != NEW.compiler_unit_id
                   OR question_row.generation_id != NEW.generation_id
                   OR question_row.candidate_version != NEW.verification_round THEN
                    RAISE EXCEPTION 'Questioner candidate verification crosses batch lineage' USING ERRCODE = '23514';
                END IF;
                IF NEW.replacement_qbank_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.qbank_questions replacement
                     WHERE replacement.qbank_id = NEW.replacement_qbank_id
                       AND replacement.run_id = NEW.run_id
                       AND replacement.compiler_unit_id = NEW.compiler_unit_id
                       AND replacement.generation_id = NEW.generation_id
                       AND replacement.generation_rank = question_row.generation_rank
                       AND replacement.candidate_version = 1
                       AND replacement.supersedes_qbank_id = NEW.qbank_id) THEN
                    RAISE EXCEPTION 'Questioner verifier replacement lineage is invalid' USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_candidate_supersessions' THEN
                SELECT * INTO source_row FROM {RUNTIME}.qbank_questions WHERE qbank_id = NEW.source_qbank_id;
                SELECT * INTO replacement_row FROM {RUNTIME}.qbank_questions WHERE qbank_id = NEW.replacement_qbank_id;
                IF source_row.qbank_id IS NULL OR replacement_row.qbank_id IS NULL
                   OR source_row.run_id != NEW.run_id OR replacement_row.run_id != NEW.run_id
                   OR source_row.compiler_unit_id != NEW.compiler_unit_id
                   OR replacement_row.compiler_unit_id != NEW.compiler_unit_id
                   OR source_row.generation_id != NEW.generation_id
                   OR replacement_row.generation_id != NEW.generation_id
                   OR source_row.generation_rank != replacement_row.generation_rank
                   OR source_row.candidate_version != 0 OR replacement_row.candidate_version != 1
                   OR replacement_row.supersedes_qbank_id != source_row.qbank_id THEN
                    RAISE EXCEPTION 'Questioner candidate supersession lineage is invalid' USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_questioner_v3_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler_version text; bad integer;
        BEGIN
            IF NEW.status != 'completed' THEN RETURN NEW; END IF;
            SELECT compiler_schema_version INTO compiler_version
              FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
            IF compiler_version IS DISTINCT FROM '0.3.0' THEN RETURN NEW; END IF;
            IF EXISTS (SELECT 1 FROM {RUNTIME}.compiler_runs compiler
                WHERE compiler.run_id = NEW.run_id AND (
                    compiler.questioner_config_hash IS NULL
                    OR compiler.questioner_model_alias IS NULL
                    OR compiler.questioner_model_revision IS NULL
                    OR compiler.questioner_prompt_hash IS NULL
                    OR compiler.questioner_prompt_version != 'pmmc-questioner-v3'
                    OR compiler.questioner_renderer_version != 'questioner-multimodal-renderer-v3'
                    OR compiler.questioner_parser_version != 'questioner-json-parser-v3'
                    OR compiler.candidate_gate_version != 'questioner-hard-gate-v3')) THEN
                RAISE EXCEPTION 'Questioner v3 compiler identity is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
             WHERE units.run_id = NEW.run_id AND (
                (SELECT count(*) FROM {RUNTIME}.questioner_opportunity_sets sets
                  WHERE sets.compiler_unit_id = units.compiler_unit_id) != 1
                OR EXISTS (SELECT 1 FROM {RUNTIME}.questioner_opportunity_sets sets
                  WHERE sets.compiler_unit_id = units.compiler_unit_id
                    AND sets.opportunity_count != (SELECT count(*) FROM {RUNTIME}.questioner_opportunities opportunities
                      WHERE opportunities.opportunity_set_id = sets.opportunity_set_id))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.questioner_opportunities opportunities
                  WHERE opportunities.compiler_unit_id = units.compiler_unit_id
                    AND (cardinality(opportunities.evidence_aliases) != (
                      SELECT count(*) FROM {RUNTIME}.questioner_opportunity_evidence_refs refs
                       WHERE refs.opportunity_id = opportunities.opportunity_id)
                      OR EXISTS (
                        SELECT 1 FROM unnest(opportunities.relation_link_ids) AS relation_link_id
                         WHERE NOT EXISTS (
                           SELECT 1 FROM {RUNTIME}.compiler_unit_links links
                            WHERE links.link_id = relation_link_id
                              AND links.run_id = opportunities.run_id
                              AND links.compiler_unit_id = opportunities.compiler_unit_id))))
                OR (SELECT count(*) FROM {RUNTIME}.questioner_generations generations
                  WHERE generations.compiler_unit_id = units.compiler_unit_id
                    AND generations.parse_status != 'failed') != 1
                OR EXISTS (SELECT 1 FROM {RUNTIME}.questioner_generations generations
                  WHERE generations.compiler_unit_id = units.compiler_unit_id
                    AND generations.parse_status != 'failed'
                    AND generations.valid_candidate_count != (
                      SELECT count(*) FROM {RUNTIME}.qbank_questions questions
                       WHERE questions.compiler_unit_id = units.compiler_unit_id
                         AND questions.candidate_version = 0))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
                  WHERE questions.compiler_unit_id = units.compiler_unit_id
                    AND (questions.opportunity_id IS NULL OR NOT EXISTS (
                      SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gates
                       WHERE gates.qbank_id = questions.qbank_id)))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
                  JOIN {RUNTIME}.questioner_candidate_gate_results gates USING (qbank_id, run_id)
                  WHERE questions.compiler_unit_id = units.compiler_unit_id
                    AND gates.status = 'eligible_for_planning'
                    AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verifies
                      WHERE verifies.qbank_id = questions.qbank_id AND verifies.verdict = 'pass'))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
                  WHERE questions.compiler_unit_id = units.compiler_unit_id
                    AND questions.candidate_version = 1
                    AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.questioner_candidate_supersessions revisions
                      WHERE revisions.replacement_qbank_id = questions.qbank_id))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
                  JOIN {RUNTIME}.questioner_candidate_gate_results gates USING (qbank_id, run_id)
                  WHERE questions.compiler_unit_id = units.compiler_unit_id
                    AND questions.candidate_version = 0
                    AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verifies
                      WHERE verifies.qbank_id = questions.qbank_id
                        AND verifies.verification_round = 0)
                    AND NOT (gates.rejection_codes && ARRAY[
                      'unsupported_question_type','future_or_foreign_reference',
                      'missing_expected_answer','missing_expected_evidence',
                      'modality_mismatch','visual_evidence_missing',
                      'multi_hop_evidence_missing','internal_identifier_leak',
                      'system_metadata_reference','assistant_follow_up',
                      'historical_question_replay','generic_external_question',
                      'placeholder_expected_answer','opportunity_mismatch',
                      'taxonomy_invariant_failed','reasoning_invariant_failed',
                      'evidence_not_entailed'
                    ]::text[]))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions revised
                  JOIN {RUNTIME}.questioner_candidate_gate_results gates
                    ON gates.qbank_id = revised.qbank_id AND gates.run_id = revised.run_id
                  WHERE revised.compiler_unit_id = units.compiler_unit_id
                    AND revised.candidate_version = 1
                    AND gates.status = 'eligible_for_planning'
                    AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verifies
                      WHERE verifies.qbank_id = revised.qbank_id
                        AND verifies.verification_round = 1
                        AND verifies.verdict = 'pass'))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions source
                  JOIN {RUNTIME}.questioner_candidate_supersessions revisions
                    ON revisions.source_qbank_id = source.qbank_id
                  WHERE source.compiler_unit_id = units.compiler_unit_id
                    AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verifies
                      WHERE verifies.qbank_id = source.qbank_id
                        AND verifies.verification_round = 0
                        AND verifies.verdict = 'revise'
                        AND verifies.replacement_qbank_id = revisions.replacement_qbank_id))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions source
                  WHERE source.compiler_unit_id = units.compiler_unit_id
                  GROUP BY source.generation_rank
                  HAVING count(*) FILTER (WHERE source.candidate_version = 0) != 1
                     OR count(*) FILTER (WHERE source.candidate_version = 1) > 1
                     OR count(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gates
                        WHERE gates.qbank_id = source.qbank_id
                          AND gates.run_id = source.run_id
                          AND gates.status = 'eligible_for_planning')) > 1)
             );
            IF bad != 0 THEN
                RAISE EXCEPTION 'Questioner v3 opportunity, verification, or terminal lineage is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad
              FROM {RUNTIME}.compiler_unit_stage_results results
              JOIN {RUNTIME}.compiler_units units USING (compiler_unit_id, run_id)
              JOIN {RUNTIME}.questioner_generations generations
                ON generations.compiler_unit_id = units.compiler_unit_id
               AND generations.run_id = units.run_id
               AND generations.parse_status != 'failed'
             WHERE units.run_id = NEW.run_id AND results.stage = 'questioner'
               AND results.status = 'success'
               AND (results.produced_count, results.accepted_count,
                    results.rejected_count) != (
                  generations.raw_candidate_count,
                  (SELECT count(*) FROM {RUNTIME}.questioner_candidate_gate_results gates
                    WHERE gates.compiler_unit_id = units.compiler_unit_id
                      AND gates.run_id = units.run_id
                      AND gates.status = 'eligible_for_planning'),
                  generations.raw_candidate_count -
                  (SELECT count(*) FROM {RUNTIME}.questioner_candidate_gate_results gates
                    WHERE gates.compiler_unit_id = units.compiler_unit_id
                      AND gates.run_id = units.run_id
                      AND gates.status = 'eligible_for_planning'));
            IF bad != 0 THEN
                RAISE EXCEPTION 'Questioner v3 stage accounting is incomplete' USING ERRCODE = '55000';
            END IF;
            IF (SELECT count(*) FROM {RUNTIME}.qbank_index_manifests indexes
                 WHERE indexes.compiler_run_id = NEW.run_id
                   AND indexes.status = 'ready') != 1 THEN
                RAISE EXCEPTION 'Questioner v3 requires one ready QBank index' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    for table in (
        "questioner_opportunity_sets",
        "questioner_opportunities",
        "questioner_opportunity_evidence_refs",
        "questioner_verification_batches",
        "questioner_candidate_verifications",
        "questioner_candidate_supersessions",
    ):
        op.execute(
            f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()"
        )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_verifier BEFORE INSERT OR UPDATE ON {RUNTIME}.questioner_verification_batches FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()"
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_opportunity BEFORE INSERT OR UPDATE ON {RUNTIME}.questioner_opportunities FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()"
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_opportunity_evidence BEFORE INSERT OR UPDATE ON {RUNTIME}.questioner_opportunity_evidence_refs FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()"
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_candidate BEFORE INSERT OR UPDATE ON {RUNTIME}.qbank_questions FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()"
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_verification_item BEFORE INSERT OR UPDATE ON {RUNTIME}.questioner_candidate_verifications FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()"
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_supersession BEFORE INSERT OR UPDATE ON {RUNTIME}.questioner_candidate_supersessions FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()"
    )
    op.execute(
        f"CREATE TRIGGER questioner_v3_completion_integrity BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_completion()"
    )


def _grant_permissions() -> None:
    tables = (
        "questioner_opportunity_sets",
        "questioner_opportunities",
        "questioner_opportunity_evidence_refs",
        "questioner_verification_batches",
        "questioner_candidate_verifications",
        "questioner_candidate_supersessions",
    )
    qualified = ", ".join(f"{RUNTIME}.{table}" for table in tables)
    op.execute(f"GRANT SELECT ON {qualified} TO pmmc_runtime, pmmc_evaluator")
    op.execute(f"GRANT INSERT ON {qualified} TO pmmc_runtime")
    for function in (
        "enforce_questioner_v3_scope()",
        "enforce_questioner_v3_completion()",
    ):
        op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.{function} FROM PUBLIC")


def downgrade() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS questioner_v3_completion_integrity ON {RUNTIME}.run_states"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_supersession ON {RUNTIME}.questioner_candidate_supersessions"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_verifier ON {RUNTIME}.questioner_verification_batches"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_verification_item ON {RUNTIME}.questioner_candidate_verifications"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_opportunity ON {RUNTIME}.questioner_opportunities"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_candidate ON {RUNTIME}.qbank_questions"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_opportunity_evidence ON {RUNTIME}.questioner_opportunity_evidence_refs"
    )
    for table in (
        "questioner_candidate_supersessions",
        "questioner_candidate_verifications",
        "questioner_verification_batches",
        "questioner_opportunity_evidence_refs",
        "questioner_opportunities",
        "questioner_opportunity_sets",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {RUNTIME}.{table}")
    for function in (
        "enforce_questioner_v3_completion()",
        "enforce_questioner_v3_scope()",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{function}")

    op.drop_constraint(
        "ck_questioner_candidate_gate_results_rejection_codes_v3",
        "questioner_candidate_gate_results",
        schema=RUNTIME,
        type_="check",
    )
    old_codes = ",".join(f"'{value}'" for value in _REJECTION_CODES[:10])
    op.create_check_constraint(
        "questioner_candidate_gate_results_rejection_codes_check",
        "questioner_candidate_gate_results",
        f"rejection_codes <@ ARRAY[{old_codes}]::text[]",
        schema=RUNTIME,
    )
    op.drop_constraint(
        "fk_qbank_questions_supersedes",
        "qbank_questions",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_qbank_questions_opportunity",
        "qbank_questions",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_constraint(
        "ck_qbank_questions_candidate_version",
        "qbank_questions",
        schema=RUNTIME,
        type_="check",
    )
    op.drop_constraint(
        "ck_qbank_questions_candidate_origin",
        "qbank_questions",
        schema=RUNTIME,
        type_="check",
    )
    op.drop_constraint(
        "uq_qbank_questions_unit_rank_version",
        "qbank_questions",
        schema=RUNTIME,
        type_="unique",
    )
    for column in (
        "supersedes_qbank_id",
        "candidate_version",
        "candidate_origin",
        "opportunity_id",
    ):
        op.drop_column("qbank_questions", column, schema=RUNTIME)
    op.create_unique_constraint(
        "qbank_questions_compiler_unit_id_generation_rank_key",
        "qbank_questions",
        ["compiler_unit_id", "generation_rank"],
        schema=RUNTIME,
    )
    for table in (
        "questioner_candidate_supersessions",
        "questioner_candidate_verifications",
        "questioner_verification_batches",
        "questioner_opportunity_evidence_refs",
        "questioner_opportunities",
        "questioner_opportunity_sets",
    ):
        op.execute(f"DROP TABLE {RUNTIME}.{table}")
