"""Allow an empty answer inventory for a completed PMMC no-answer evaluation.

No historical facts or final answers are changed. Baseline denominators stay positive.
"""

from alembic import op

revision: str = "20260907_0028"
down_revision: str | None = "20260907_0027"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    for table in ("evaluation_run_states", "evaluation_targets"):
        op.execute(
            f"ALTER TABLE pmmc_evaluation.{table} DROP CONSTRAINT {table}_selected_count_check"
        )
        op.execute(
            f"ALTER TABLE pmmc_evaluation.{table} ADD CONSTRAINT "
            f"{table}_selected_count_check CHECK (selected_count >= 0)"
        )
    op.execute("""CREATE FUNCTION pmmc_evaluation.enforce_no_answer_evaluation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.selected_count=0 AND NOT EXISTS (
            SELECT 1 FROM pmmc_evaluation.evaluation_runs r
             WHERE r.evaluation_run_id=NEW.evaluation_run_id
               AND r.source_kind='pmmc_query_run'
               AND r.metadata->>'no_answer_policy'='independent-answer-abstention-v1'
               AND (r.metadata->>'query_count')::integer > 0
          ) THEN
            RAISE EXCEPTION 'empty evaluation requires a PMMC no-answer denominator'
              USING ERRCODE='23514';
          END IF;
          RETURN NEW;
        END $$""")
    for table in ("evaluation_run_states", "evaluation_targets"):
        op.execute(
            f"CREATE TRIGGER no_answer_evaluation BEFORE INSERT OR UPDATE ON "
            f"pmmc_evaluation.{table} FOR EACH ROW EXECUTE FUNCTION "
            "pmmc_evaluation.enforce_no_answer_evaluation()"
        )
    op.execute("""CREATE FUNCTION pmmc_runtime.enforce_answer_abstention()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP != 'INSERT' THEN
            IF OLD.metadata->>'terminal_policy'='independent-answer-abstention-v1' THEN
              RAISE EXCEPTION 'answer abstention is append-only' USING ERRCODE='55000';
            END IF;
            IF TG_OP='DELETE' THEN RETURN OLD; END IF;
            IF NEW.metadata->>'terminal_policy'='independent-answer-abstention-v1' THEN
              RAISE EXCEPTION 'answer abstention must be inserted, not reinterpreted'
                USING ERRCODE='55000';
            END IF;
          END IF;
          IF NEW.metadata->>'terminal_policy'='independent-answer-abstention-v1' THEN
            IF NEW.supported OR NEW.answer IS NOT NULL OR NEW.answer_id IS NOT NULL
               OR NEW.abstain_reason IS NULL OR NEW.parse_status!='accepted'
               OR NOT NEW.provider_schema_conformant OR NEW.finish_reason='length'
               OR NOT EXISTS (
                 SELECT 1 FROM pmmc_runtime.model_calls c
                  WHERE c.model_call_id=NEW.model_call_id AND c.query_id=NEW.query_id
                    AND c.stage='answer' AND c.status='success'
                    AND c.run_id=COALESCE(NEW.metadata->>'source_run_id', NEW.run_id)
                    AND (NEW.metadata->>'method'!='native_multimodal_rag'
                         OR c.context_assembly_id IS NOT NULL)
               ) THEN
              RAISE EXCEPTION 'answer abstention contract or model lineage invalid'
                USING ERRCODE='23514';
            END IF;
            IF EXISTS (SELECT 1 FROM pmmc_runtime.run_states s
                        WHERE s.run_id=NEW.run_id
                          AND s.status IN ('completed','failed','cancelled'))
               OR EXISTS (SELECT 1 FROM pmmc_runtime.query_executor_runs s
                        WHERE s.query_executor_run_id=NEW.run_id
                          AND s.status IN ('completed','failed','cancelled')) THEN
              RAISE EXCEPTION 'answer abstention parent is terminal' USING ERRCODE='55000';
            END IF;
            IF NEW.metadata ? 'source_run_id' AND NOT EXISTS (
              SELECT 1 FROM pmmc_runtime.structured_answer_outputs s
                JOIN pmmc_runtime.pmmc_fallback_run_links l ON l.fallback_run_id=s.run_id
               WHERE l.query_executor_run_id=NEW.run_id AND s.query_id=NEW.query_id
                 AND s.run_id=NEW.metadata->>'source_run_id'
                 AND s.structured_answer_output_id=NEW.metadata->>'source_abstention_id'
                 AND s.content_hash=NEW.metadata->>'source_abstention_hash'
                 AND NOT s.supported AND s.parse_status='accepted'
            ) THEN
              RAISE EXCEPTION 'answer abstention projection lineage invalid' USING ERRCODE='23514';
            END IF;
          END IF;
          RETURN NEW;
        END $$""")
    op.execute("""CREATE TRIGGER answer_abstention_integrity BEFORE INSERT OR UPDATE OR DELETE
        ON pmmc_runtime.structured_answer_outputs FOR EACH ROW
        EXECUTE FUNCTION pmmc_runtime.enforce_answer_abstention()""")
    for function in (
        "pmmc_runtime.enforce_answer_abstention()",
        "pmmc_evaluation.enforce_no_answer_evaluation()",
    ):
        op.execute(f"ALTER FUNCTION {function} OWNER TO pmmc_owner")
        op.execute(f"REVOKE ALL ON FUNCTION {function} FROM PUBLIC")
    print("ANSWER_TERMINAL_CONTRACT_MIGRATION state=ready fact_rows_modified=0", flush=True)


def downgrade() -> None:
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM pmmc_runtime.structured_answer_outputs
                    WHERE metadata->>'terminal_policy'='independent-answer-abstention-v1') THEN
          RAISE EXCEPTION 'cannot downgrade with independent answer abstention facts';
        END IF;
        END $$""")
    # Adding the old positive constraint rejects downgrade if any zero-answer
    # evaluation exists; never remove or rewrite those scientific facts.
    for table in ("evaluation_run_states", "evaluation_targets"):
        op.execute(
            f"ALTER TABLE pmmc_evaluation.{table} DROP CONSTRAINT {table}_selected_count_check"
        )
        op.execute(
            f"ALTER TABLE pmmc_evaluation.{table} ADD CONSTRAINT "
            f"{table}_selected_count_check CHECK (selected_count > 0)"
        )
        op.execute(f"DROP TRIGGER no_answer_evaluation ON pmmc_evaluation.{table}")
    op.execute("DROP FUNCTION pmmc_evaluation.enforce_no_answer_evaluation()")
    op.execute("DROP TRIGGER answer_abstention_integrity ON pmmc_runtime.structured_answer_outputs")
    op.execute("DROP FUNCTION pmmc_runtime.enforce_answer_abstention()")
