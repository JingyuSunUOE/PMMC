"""Typed answers and exact declared-cell completion; historical guards stay intact."""

from alembic import op

revision = "20260920_0032"
down_revision = "20260920_0031"
branch_labels = None
depends_on = None

_DISPATCH = "IF NEW.schema_version = '0.3.0' THEN RETURN NEW; END IF; -- step8 dispatch\n"


def _dispatch(enable: bool) -> None:
    marker = "IF NEW.status != 'completed' OR OLD.status = 'completed' THEN RETURN NEW; END IF;"
    old, new = (marker, _DISPATCH + marker) if enable else (_DISPATCH + marker, marker)
    old, new = ("'" + value.replace("'", "''") + "'" for value in (old, new))
    op.execute(f"""DO $rewrite$
    DECLARE definition text;
    BEGIN
      definition := pg_get_functiondef(
          'pmmc_runtime.enforce_query_executor_completion()'::regprocedure);
      IF (length(definition)-length(replace(definition,{old},''))) / length({old})
           IS DISTINCT FROM 1 THEN
        RAISE EXCEPTION 'step8_query_completion_dispatch_contract_mismatch';
      END IF;
      EXECUTE replace(definition,{old},{new});
    END $rewrite$""")


def _create_evaluator_completion() -> None:
    # The report is an evaluator attestation of object-store bytes. SQL verifies
    # its source inventory and immutable artifact identity, not remote bytes or scores.
    op.execute("""CREATE FUNCTION pmmc_evaluation.enforce_step8_evaluation_completion()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path=pg_catalog,pmmc_evaluation,pmmc_runtime AS $$
    DECLARE evaluation pmmc_evaluation.evaluation_runs%ROWTYPE;
            report jsonb; artifact jsonb; score jsonb; source jsonb; expected integer;
            annotation jsonb; annotation_payload text; reference_kind text;
            gold_kind text; gold_exists boolean; gold_annotation_exists boolean;
            text_reference_exists boolean;
            raw_answer_exists boolean; candidate_kind text; judge_required boolean;
            null_hash text := encode(sha256(convert_to('null','UTF8')),'hex');
    BEGIN
      SELECT * INTO evaluation FROM pmmc_evaluation.evaluation_runs
       WHERE evaluation_run_id=NEW.evaluation_run_id;
      IF evaluation.protocol_version IS DISTINCT FROM 'step8-query-evaluation-v1'
         OR NEW.status!='completed' THEN RETURN NEW; END IF;
      report := NEW.metadata->'completion_report';
      artifact := NEW.metadata->'completion_artifact';
      IF jsonb_typeof(report) IS DISTINCT FROM 'object'
         OR report->>'schema_version' IS DISTINCT FROM evaluation.protocol_version
         OR report->>'evaluation_run_id' IS DISTINCT FROM NEW.evaluation_run_id
         OR report->>'identity_hash' IS DISTINCT FROM evaluation.config_hash
         OR report->>'status' IS DISTINCT FROM 'complete'
         OR report->'scientific_approval' IS DISTINCT FROM 'false'::jsonb
         OR report->>'query_executor_run_id' IS DISTINCT FROM evaluation.legacy_source_run_id
         OR report->>'query_id' IS DISTINCT FROM evaluation.metadata->>'query_id'
         OR report->>'campaign_id' IS DISTINCT FROM evaluation.metadata->>'campaign_id'
         OR report->>'suite_id' IS DISTINCT FROM evaluation.metadata->>'suite_id'
         OR NEW.selected_count!=1 OR NEW.failed_count!=0 OR NEW.completed_at IS NULL
         OR NEW.succeeded_count!=NEW.target_count
         OR jsonb_typeof(report->'scores') IS DISTINCT FROM 'array'
         OR jsonb_typeof(evaluation.metadata->'sources') IS DISTINCT FROM 'array'
         OR NOT EXISTS (
             SELECT 1 FROM pmmc_evaluation.run_artifacts a
              WHERE a.artifact_id=artifact->>'artifact_id' AND a.sha256=artifact->>'sha256'
                AND a.artifact_id='step8-scores-' || left(evaluation.config_hash,32)
                AND a.evaluation_run_id=NEW.evaluation_run_id AND a.semantic_run_id IS NULL
                AND a.run_id=evaluation.legacy_source_run_id
                AND a.query_id=evaluation.metadata->>'query_id'
                AND a.snapshot_id=evaluation.metadata->>'snapshot_id'
                AND a.artifact_type='trace_export' AND a.mime_type='application/json'
                AND a.state='ready' AND a.ready_at IS NOT NULL AND a.size_bytes>0
                AND a.sha256 ~ '^[0-9a-f]{64}$' AND a.storage_uri IS NOT NULL)
         OR NOT EXISTS (
             SELECT 1 FROM pmmc_runtime.query_executor_runs r
             JOIN pmmc_runtime.run_queries q ON q.run_id=r.query_executor_run_id
             JOIN pmmc_runtime.benchmark_campaigns c
               ON c.campaign_id=evaluation.metadata->>'campaign_id'
              WHERE r.query_executor_run_id=evaluation.legacy_source_run_id
                AND r.schema_version='0.3.0' AND r.status='completed'
                AND r.snapshot_id=evaluation.metadata->>'snapshot_id'
                AND q.query_id=evaluation.metadata->>'query_id'
                AND c.protocol_hash=evaluation.metadata->>'protocol_hash') THEN
        RAISE EXCEPTION 'step8_evaluation_artifact_or_identity_invalid' USING ERRCODE='55000';
      END IF;
      SELECT count(*) INTO expected FROM pmmc_runtime.benchmark_cell_assignments a
      JOIN pmmc_runtime.benchmark_experiment_cells c ON c.cell_id=a.cell_id
      JOIN pmmc_runtime.query_executor_runs r
        ON r.query_executor_run_id=evaluation.legacy_source_run_id
      WHERE a.campaign_id=evaluation.metadata->>'campaign_id'
        AND a.query_id=evaluation.metadata->>'query_id'
        AND c.protocol_hash=evaluation.metadata->>'protocol_hash' AND c.dataset=r.dataset
        AND c.answer_model=r.metadata->>'answer_model_alias'
        AND ((c.method='pmmc' AND c.payload->>'treatment'=r.metadata->>'treatment')
             OR (c.method!='pmmc' AND r.metadata->>'treatment'='main'));
      IF expected=0 OR expected!=NEW.target_count
         OR (report->>'selected_cell_count')::integer IS DISTINCT FROM expected
         OR jsonb_array_length(report->'scores')!=expected
         OR jsonb_array_length(evaluation.metadata->'sources')!=expected
         OR (SELECT count(DISTINCT s->>'cell_id') FROM jsonb_array_elements(report->'scores') s)
              !=expected
         OR (SELECT count(DISTINCT s->>'cell_id')
               FROM jsonb_array_elements(evaluation.metadata->'sources') s)!=expected THEN
        RAISE EXCEPTION 'step8_evaluation_cell_denominator_invalid' USING ERRCODE='55000';
      END IF;
      -- The immutable evaluator attestation binds lossless source resolution to
      -- this query/snapshot/gold and implementation. It never changes query_gold.
      annotation := evaluation.metadata->'reference_annotation';
      annotation_payload := evaluation.metadata->>'reference_annotation_payload';
      IF jsonb_typeof(annotation) IS DISTINCT FROM 'object'
         OR jsonb_typeof(evaluation.metadata->'reference_annotation_payload')
              IS DISTINCT FROM 'string'
         OR report->'reference_annotation' IS DISTINCT FROM annotation
         OR annotation_payload::jsonb IS DISTINCT FROM annotation-'hash'
         OR annotation->>'hash' IS DISTINCT FROM
              encode(sha256(convert_to(annotation_payload,'UTF8')),'hex')
         OR NOT (annotation ?& ARRAY['schema_version','resolver_version','snapshot_id',
              'query_id','query_hash','gold_hash','resolver_source_hash','source_refs_hash',
              'status','kind','expected_kind','derivation','unavailable_reason','detail',
              'reference_hash','text_judge_reference_hash','hash'])
         OR annotation - ARRAY['schema_version','resolver_version','snapshot_id',
              'query_id','query_hash','gold_hash','resolver_source_hash','source_refs_hash',
              'status','kind','expected_kind','derivation','unavailable_reason','detail',
              'reference_hash','text_judge_reference_hash','hash'] != '{}'::jsonb
         OR annotation->>'schema_version' IS DISTINCT FROM 'step8-reference-annotation-v1'
         OR annotation->>'resolver_version' IS DISTINCT FROM 'step8-lossless-typed-gold-v1'
         OR annotation->>'snapshot_id' IS DISTINCT FROM evaluation.metadata->>'snapshot_id'
         OR annotation->>'query_id' IS DISTINCT FROM evaluation.metadata->>'query_id'
         OR annotation->>'query_hash' IS DISTINCT FROM evaluation.metadata->>'query_hash'
         OR annotation->>'gold_hash' IS DISTINCT FROM evaluation.metadata->>'gold_hash'
         OR EXISTS (SELECT 1 FROM unnest(ARRAY['query_hash','gold_hash','source_refs_hash',
              'resolver_source_hash','reference_hash','text_judge_reference_hash','hash']) key
              WHERE COALESCE(annotation->>key,'') !~ '^[0-9a-f]{64}$')
         -- No frozen expected-kind protocol is currently declared by this adapter.
         OR annotation->'expected_kind' IS DISTINCT FROM 'null'::jsonb
         OR annotation->'detail' IS DISTINCT FROM 'null'::jsonb
         OR COALESCE(annotation->>'status','') NOT IN ('resolved','text_only','missing') THEN
        RAISE EXCEPTION 'step8_evaluation_reference_annotation_invalid' USING ERRCODE='55000';
      END IF;
      SELECT g.gold_metadata #>> '{typed_answer,kind}', NULLIF(g.answer,'') IS NOT NULL,
             g.gold_metadata #>> '{typed_answer,kind}'='text' OR NULLIF(g.answer,'') IS NOT NULL,
             COALESCE(g.gold_metadata->'typed_answer','null'::jsonb)!='null'::jsonb
        INTO gold_kind, raw_answer_exists, text_reference_exists, gold_annotation_exists
        FROM pmmc_evaluation.query_gold g
       WHERE g.snapshot_id=evaluation.metadata->>'snapshot_id'
         AND g.query_id=evaluation.metadata->>'query_id';
      gold_exists := FOUND;
      reference_kind := annotation->>'kind';
      IF (annotation->>'status'='missing') IS DISTINCT FROM (NOT gold_exists)
         OR (annotation->>'reference_hash'=null_hash) IS DISTINCT FROM (NOT gold_exists)
         OR (annotation->>'text_judge_reference_hash'!=null_hash)
              IS DISTINCT FROM COALESCE(text_reference_exists,false)
         OR (NOT gold_exists AND (reference_kind IS NOT NULL
              OR annotation->>'derivation' IS DISTINCT FROM 'none'
              OR annotation->>'unavailable_reason' IS DISTINCT FROM 'gold_missing'))
         OR (gold_exists AND (COALESCE(reference_kind,'') NOT IN
              ('text','number','boolean','date','image','image_set','ordered_image_list')
              OR annotation->'unavailable_reason' IS DISTINCT FROM 'null'::jsonb))
         OR (gold_annotation_exists AND (
              annotation->>'status' IS DISTINCT FROM 'resolved'
              OR annotation->>'derivation' IS DISTINCT FROM 'explicit_typed_annotation'
              OR reference_kind IS DISTINCT FROM gold_kind))
         OR (gold_exists AND NOT gold_annotation_exists AND (
              NOT raw_answer_exists
              OR ((annotation->>'status'='text_only' AND reference_kind='text'
                         AND annotation->>'derivation'='source_text')
                   OR (annotation->>'status'='resolved' AND (
                       (reference_kind IN ('number','boolean','date')
                         AND annotation->>'derivation'='source_scalar')
                       OR (reference_kind IN ('image','image_set','ordered_image_list')
                         AND annotation->>'derivation'='source_image_refs')))) IS NOT TRUE)) THEN
        RAISE EXCEPTION 'step8_evaluation_reference_contract_invalid' USING ERRCODE='55000';
      END IF;
      IF report #> '{selected_query,applicable}'
           IS DISTINCT FROM to_jsonb(reference_kind IS NOT NULL)
         OR (report->>'applicable_cell_count')::integer IS DISTINCT FROM
              (CASE WHEN reference_kind IS NULL THEN 0 ELSE expected END) THEN
        RAISE EXCEPTION 'step8_evaluation_applicability_invalid' USING ERRCODE='55000';
      END IF;
      FOR score IN SELECT value FROM jsonb_array_elements(report->'scores') LOOP
        SELECT value INTO source FROM jsonb_array_elements(evaluation.metadata->'sources')
         WHERE value->>'cell_id'=score->>'cell_id';
        IF NOT FOUND OR score->>'source_fact_id' IS DISTINCT FROM source->>'source_id'
           OR score->>'source_fact_hash' IS DISTINCT FROM source->>'source_hash'
           OR score->>'terminal_state' IS DISTINCT FROM source->>'terminal_state'
           OR COALESCE(score->'engineering_error','null'::jsonb)!='null'::jsonb
           OR score->>'status' IS NULL OR score->>'status' NOT IN ('answered','abstention')
           OR (score->>'status'='answered') IS DISTINCT FROM
                (score->>'terminal_state'='answered')
           OR score->>'terminal_state' IS NULL
           OR score->>'terminal_state' NOT IN ('answered','abstained','no_answer')
           OR NOT EXISTS (
               SELECT 1 FROM pmmc_runtime.benchmark_cell_assignments a
               JOIN pmmc_runtime.benchmark_experiment_cells c ON c.cell_id=a.cell_id
               JOIN pmmc_runtime.query_executor_runs r
                 ON r.query_executor_run_id=evaluation.legacy_source_run_id
                WHERE a.campaign_id=evaluation.metadata->>'campaign_id'
                  AND a.query_id=evaluation.metadata->>'query_id'
                  AND a.cell_id=score->>'cell_id' AND c.method=score->>'method'
                  AND c.protocol_hash=evaluation.metadata->>'protocol_hash'
                  AND c.dataset=r.dataset AND c.answer_model=r.metadata->>'answer_model_alias'
                  AND ((c.method='pmmc' AND c.payload->>'treatment'=r.metadata->>'treatment')
                       OR (c.method!='pmmc' AND r.metadata->>'treatment'='main')))
           OR NOT (EXISTS (
               SELECT 1 FROM pmmc_runtime.query_experiment_cell_results c
               JOIN pmmc_runtime.benchmark_experiment_cells definition
                 ON definition.cell_id=score->>'cell_id'
                WHERE score->>'method'='pmmc' AND c.cell_result_id=score->>'source_fact_id'
                  AND c.content_hash=score->>'source_fact_hash'
                  AND c.query_executor_run_id=evaluation.legacy_source_run_id
                  AND c.query_id=evaluation.metadata->>'query_id'
                  AND c.question_budget=(definition.payload->>'question_budget')::integer
                  AND c.revision_depth=(definition.payload->>'round_id')::integer
                  AND c.treatment=CASE definition.payload->>'source_variant'
                      WHEN 'no_fallback' THEN 'no_fallback'
                      WHEN 'forced_top1' THEN 'forced_top1_counterfactual'
                      WHEN 'active' THEN CASE definition.payload->>'treatment'
                          WHEN 'main' THEN 'pmmc' ELSE definition.payload->>'treatment' END END
                  AND c.terminal_state=score->>'terminal_state'
                  AND c.selected_answer_id IS NOT DISTINCT FROM score->>'typed_answer_id')
             OR EXISTS (
               SELECT 1 FROM pmmc_runtime.query_runtime_typed_answers a
               JOIN pmmc_runtime.baseline_suite_runs b ON b.run_id=a.query_executor_run_id
               JOIN pmmc_runtime.run_states s ON s.run_id=b.run_id AND s.status='completed'
                WHERE score->>'method'!='pmmc' AND a.typed_answer_id=score->>'source_fact_id'
                  AND a.typed_answer_id=score->>'typed_answer_id'
                  AND a.content_hash=score->>'source_fact_hash'
                  AND b.suite_id=evaluation.metadata->>'suite_id' AND b.method=score->>'method'
                  AND a.query_id=evaluation.metadata->>'query_id'
                  AND a.terminal_state=score->>'terminal_state'
                  AND a.payload #>> '{metadata,typed_result,answer_variant}'='primary')) THEN
          RAISE EXCEPTION 'step8_evaluation_source_lineage_invalid' USING ERRCODE='55000';
        END IF;
        SELECT a.payload #>> '{metadata,typed_result,output,answer_kind}' INTO candidate_kind
          FROM pmmc_runtime.query_runtime_typed_answers a
         WHERE a.typed_answer_id=score->>'typed_answer_id';
        IF score->'applicable' IS DISTINCT FROM to_jsonb(reference_kind IS NOT NULL)
           OR score->>'metric' IS DISTINCT FROM 'typed_exact_match'
           OR (reference_kind IS NULL AND (score->'value' IS DISTINCT FROM 'null'::jsonb
                OR score->>'unavailable_reason' IS DISTINCT FROM 'gold_missing'))
           OR (reference_kind IS NOT NULL AND (
                jsonb_typeof(score->'value') IS DISTINCT FROM 'number'
                OR (score->>'value')::numeric NOT BETWEEN 0 AND 1
                OR score->'unavailable_reason' IS DISTINCT FROM 'null'::jsonb))
           OR (annotation->>'status'='text_only' AND score->>'terminal_state'='answered'
                AND candidate_kind IS DISTINCT FROM 'text') THEN
          RAISE EXCEPTION 'step8_evaluation_cell_applicability_invalid' USING ERRCODE='55000';
        END IF;
        -- A derived image/number reference can still retain a real source text
        -- reference for text candidates. Image outputs are never sent to Judge.
        judge_required := COALESCE(text_reference_exists,false)
            AND score->>'terminal_state'='answered' AND candidate_kind='text';
        IF (COALESCE(judge_required,false) AND score->>'judge_model_call_id' IS NULL)
           OR (NOT COALESCE(judge_required,false) AND (
                score->'judge_model_call_id' IS DISTINCT FROM 'null'::jsonb
                OR score->'judge_value' IS DISTINCT FROM 'null'::jsonb
                OR score->'judge_method_blind' IS DISTINCT FROM 'null'::jsonb))
           OR (score->>'judge_model_call_id' IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM pmmc_evaluation.model_calls c
            JOIN pmmc_runtime.query_runtime_typed_answers answer
              ON answer.typed_answer_id=score->>'typed_answer_id'
             WHERE c.model_call_id=score->>'judge_model_call_id'
               AND c.evaluation_run_id=NEW.evaluation_run_id
               AND c.run_id=answer.query_executor_run_id
               AND c.query_id=evaluation.metadata->>'query_id'
               AND c.snapshot_id=evaluation.metadata->>'snapshot_id'
               AND c.stage='judge' AND c.status='success'
               AND c.model_alias=evaluation.judge_model_alias
               AND c.request_config #>> '{request_metadata,task}'='baseline_judge'
               AND c.request_config #>> '{request_metadata,rubric_version}'
                    =evaluation.judge_rubric_version
               AND c.parsed_output->'score'=score->'judge_value'
               AND score->'judge_method_blind'='true'::jsonb)) THEN
          RAISE EXCEPTION 'step8_evaluation_judge_lineage_invalid' USING ERRCODE='55000';
        END IF;
      END LOOP;
      RETURN NEW;
    END $$""")
    op.execute("""CREATE TRIGGER step8_evaluation_completion BEFORE INSERT OR UPDATE
      ON pmmc_evaluation.evaluation_run_states FOR EACH ROW
      EXECUTE FUNCTION pmmc_evaluation.enforce_step8_evaluation_completion()""")
    op.execute("""CREATE FUNCTION pmmc_evaluation.freeze_step8_ready_artifact()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path=pg_catalog,pmmc_evaluation AS $$
    BEGIN
      IF OLD.state='ready' AND EXISTS (SELECT 1 FROM pmmc_evaluation.evaluation_runs e
          WHERE e.evaluation_run_id=OLD.evaluation_run_id
            AND e.protocol_version='step8-query-evaluation-v1') THEN
        RAISE EXCEPTION 'step8_ready_evaluation_artifact_immutable' USING ERRCODE='55000';
      END IF;
      IF TG_OP='DELETE' THEN RETURN OLD; END IF;
      RETURN NEW;
    END $$""")
    op.execute("""CREATE TRIGGER step8_ready_artifact_immutable BEFORE UPDATE OR DELETE
      ON pmmc_evaluation.run_artifacts FOR EACH ROW
      EXECUTE FUNCTION pmmc_evaluation.freeze_step8_ready_artifact()""")
    for name in ('enforce_step8_evaluation_completion', 'freeze_step8_ready_artifact'):
        op.execute(f"ALTER FUNCTION pmmc_evaluation.{name}() OWNER TO pmmc_owner")
        op.execute(f"REVOKE ALL ON FUNCTION pmmc_evaluation.{name}() FROM PUBLIC")


def upgrade() -> None:
    _dispatch(True)
    _create_evaluator_completion()
    op.execute("""CREATE FUNCTION pmmc_runtime.enforce_step8_runtime_fact_scope()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path=pg_catalog,pmmc_runtime AS $$
    DECLARE parent pmmc_runtime.query_executor_runs%ROWTYPE;
    BEGIN
      SELECT * INTO parent FROM pmmc_runtime.query_executor_runs
       WHERE query_executor_run_id=NEW.query_executor_run_id FOR SHARE;
      IF NOT FOUND OR parent.schema_version!='0.3.0'
         OR parent.status IN ('completed','failed','cancelled')
         OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.run_queries q
              WHERE q.run_id=NEW.query_executor_run_id AND q.query_id=NEW.query_id
                AND q.snapshot_id=parent.snapshot_id) THEN
        RAISE EXCEPTION 'step8_runtime_fact_scope_invalid' USING ERRCODE='55000';
      END IF;
      IF TG_TABLE_NAME='query_experiment_cell_results' THEN
        IF NEW.payload->>'query_executor_run_id' IS DISTINCT FROM NEW.query_executor_run_id
           OR NEW.payload->>'query_id' IS DISTINCT FROM NEW.query_id
           OR NEW.payload->>'cell_id' IS DISTINCT FROM NEW.cell_id
           OR NEW.payload->>'cell_result_id' IS DISTINCT FROM NEW.cell_result_id
           OR NEW.payload->>'terminal_state' IS DISTINCT FROM NEW.terminal_state
           OR NEW.payload->>'selected_answer_id' IS DISTINCT FROM NEW.selected_answer_id
           OR NEW.payload #>> '{cell,treatment}' IS DISTINCT FROM NEW.treatment
           OR (NEW.payload #>> '{cell,question_budget}')::integer
                IS DISTINCT FROM NEW.question_budget
           OR (NEW.payload #>> '{cell,revision_depth}')::integer
                IS DISTINCT FROM NEW.revision_depth
           OR NEW.payload #>> '{cell,answer_contract}' IS DISTINCT FROM 'typed-answer-contract-v3'
           OR NEW.payload->>'content_hash' IS DISTINCT FROM NEW.content_hash
           OR jsonb_typeof(NEW.payload->'active_fallback') IS DISTINCT FROM 'boolean' THEN
          RAISE EXCEPTION 'step8_cell_payload_columns_disagree' USING ERRCODE='23514';
        END IF;
      END IF;
      RETURN NEW;
    END $$""")
    for table in ('query_experiment_cell_results', 'query_runtime_phase_states'):
        op.execute(f"CREATE TRIGGER step8_runtime_fact_scope BEFORE INSERT ON pmmc_runtime.{table} "
                   "FOR EACH ROW EXECUTE FUNCTION pmmc_runtime.enforce_step8_runtime_fact_scope()")
    op.execute("""CREATE FUNCTION pmmc_runtime.enforce_step8_typed_answer()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path = pg_catalog, pmmc_runtime AS $$
    DECLARE result jsonb; call jsonb;
    BEGIN
      PERFORM 1 FROM pmmc_runtime.run_states
       WHERE run_id=NEW.query_executor_run_id FOR SHARE;
      PERFORM 1 FROM pmmc_runtime.query_executor_runs
       WHERE query_executor_run_id=NEW.query_executor_run_id FOR SHARE;
      result := NEW.payload #> '{metadata,typed_result}';
      IF jsonb_typeof(result) IS DISTINCT FROM 'object'
         OR result->>'contract_version' IS DISTINCT FROM 'typed-answer-contract-v3'
         OR result->>'typed_answer_id' IS DISTINCT FROM NEW.typed_answer_id
         OR result->>'run_id' IS DISTINCT FROM NEW.query_executor_run_id
         OR result->>'query_id' IS DISTINCT FROM NEW.query_id
         OR result->>'terminal_state' IS DISTINCT FROM NEW.terminal_state
         OR result->>'answer_model_call_id' IS DISTINCT FROM NEW.answer_model_call_id
         OR NEW.payload->>'typed_answer_id' IS DISTINCT FROM NEW.typed_answer_id
         OR NEW.payload->>'query_executor_run_id' IS DISTINCT FROM NEW.query_executor_run_id
         OR NEW.payload->>'query_id' IS DISTINCT FROM NEW.query_id
         OR NEW.payload->>'answer_identity_hash' IS DISTINCT FROM NEW.answer_identity_hash
         OR NEW.payload->>'answer_model_call_id' IS DISTINCT FROM NEW.answer_model_call_id
         OR NEW.payload->>'terminal_state' IS DISTINCT FROM NEW.terminal_state
         OR NEW.payload->>'content_hash' IS DISTINCT FROM NEW.content_hash
         OR NEW.payload->'output' IS DISTINCT FROM result->'output'
         OR NEW.payload->'ungated_output' IS DISTINCT FROM result->'ungated_output'
         OR NEW.payload->'image_items' IS DISTINCT FROM result->'image_items'
         OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.run_queries q
             WHERE q.run_id=NEW.query_executor_run_id AND q.query_id=NEW.query_id
               AND q.snapshot_id=result->>'snapshot_id')
         OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.run_manifests r
             WHERE r.run_id=NEW.query_executor_run_id AND r.method=result->>'method')
         OR (NEW.terminal_state='fallback_required' AND result->>'method'!='pmmc') THEN
        RAISE EXCEPTION 'typed_answer_scope_invalid' USING ERRCODE='23514';
      END IF;
      IF EXISTS (SELECT 1 FROM pmmc_runtime.run_states s
                  WHERE s.run_id=NEW.query_executor_run_id
                    AND s.status IN ('completed','failed','cancelled'))
         OR EXISTS (SELECT 1 FROM pmmc_runtime.query_executor_runs s
                  WHERE s.query_executor_run_id=NEW.query_executor_run_id
                    AND s.status IN ('completed','failed','cancelled')) THEN
        RAISE EXCEPTION 'typed_answer_parent_terminal' USING ERRCODE='55000';
      END IF;
      IF NOT EXISTS (SELECT 1 FROM pmmc_runtime.model_calls c
          WHERE c.model_call_id=NEW.answer_model_call_id
            AND c.run_id=NEW.query_executor_run_id AND c.query_id=NEW.query_id
            AND c.snapshot_id=result->>'snapshot_id'
            AND c.actual_model_id=result->>'actual_model_id'
            AND c.request_config #>> '{request_metadata,model_role}'
                IN ('answer','answer_format_retry')
            AND c.stage='answer' AND c.status='success') THEN
        RAISE EXCEPTION 'typed_answer_model_lineage_missing' USING ERRCODE='23514';
      END IF;
      IF jsonb_typeof(result->'calls') IS DISTINCT FROM 'array'
         OR jsonb_array_length(result->'calls')=0 THEN
        RAISE EXCEPTION 'typed_answer_call_inventory_missing' USING ERRCODE='23514';
      END IF;
      IF (SELECT count(DISTINCT value->>'model_call_id')
            FROM jsonb_array_elements(result->'calls')) != jsonb_array_length(result->'calls')
         OR NOT EXISTS (SELECT 1 FROM jsonb_array_elements(result->'calls') c
             WHERE c->>'model_call_id'=NEW.answer_model_call_id
               AND c->>'role' IN ('answer','answer_format_retry')) THEN
        RAISE EXCEPTION 'typed_answer_call_inventory_invalid' USING ERRCODE='23514';
      END IF;
      FOR call IN SELECT value FROM jsonb_array_elements(result->'calls') LOOP
        IF NOT EXISTS (SELECT 1 FROM pmmc_runtime.model_calls c
            WHERE c.model_call_id=call->>'model_call_id' AND c.status='success'
              AND c.stage='answer' AND c.snapshot_id=result->>'snapshot_id'
              AND c.request_config #>> '{request_metadata,model_role}'=call->>'role'
              AND call->>'role' IN
                  ('answer','answer_format_retry','answer_grounding','grounding_format_retry')
              AND c.run_id=NEW.query_executor_run_id AND c.query_id=NEW.query_id) THEN
          RAISE EXCEPTION 'typed_answer_call_inventory_incomplete' USING ERRCODE='23514';
        END IF;
      END LOOP;
      IF jsonb_typeof(result #> '{output,supported}') IS DISTINCT FROM 'boolean'
         OR jsonb_typeof(result #> '{ungated_output,supported}') IS DISTINCT FROM 'boolean'
         OR (NEW.terminal_state='answered') IS DISTINCT FROM
           COALESCE((result #>> '{output,supported}')::boolean,false)
         OR (COALESCE((result #>> '{ungated_output,supported}')::boolean,false)
             AND (result->'grounding' IS NULL OR result->'grounding'='null'::jsonb
                  OR result #>> '{grounding,policy_version}'
                       IS DISTINCT FROM 'source-only-answer-grounding-v1'
                  OR result #>> '{grounding,model_alias}' IS DISTINCT FROM 'grounding-qwen3.5-9b'
                  OR (NEW.terminal_state='answered') IS DISTINCT FROM
                     (result #>> '{grounding,assessment,supported}')::boolean)) THEN
        RAISE EXCEPTION 'typed_answer_grounding_contract_invalid' USING ERRCODE='23514';
      END IF;
      IF NEW.terminal_state!='answered' AND (
          result #>> '{output,answer_kind}' IS DISTINCT FROM 'abstention'
          OR result #> '{output,answer_text}' IS DISTINCT FROM 'null'::jsonb
          OR result #> '{output,selected_image_aliases}' IS DISTINCT FROM '[]'::jsonb
          OR result->'image_items' IS DISTINCT FROM '[]'::jsonb) THEN
        RAISE EXCEPTION 'typed_answer_unsupported_output_invalid' USING ERRCODE='23514';
      END IF;
      IF (result #>> '{ungated_output,supported}')::boolean THEN
        IF result #>> '{grounding,run_id}' IS DISTINCT FROM NEW.query_executor_run_id
           OR result #>> '{grounding,query_id}' IS DISTINCT FROM NEW.query_id
           OR result #>> '{grounding,answer_model_call_id}'
                IS DISTINCT FROM NEW.answer_model_call_id
           OR result #>> '{grounding,assessment_source}' IS DISTINCT FROM 'model_assessed'
           OR jsonb_typeof(result #> '{grounding,calls}') IS DISTINCT FROM 'array'
           OR jsonb_array_length(result #> '{grounding,calls}') NOT BETWEEN 1 AND 2 THEN
          RAISE EXCEPTION 'typed_answer_grounding_lineage_invalid' USING ERRCODE='23514';
        END IF;
        IF (result #>> '{output,supported}')::boolean
              AND result->'output' IS DISTINCT FROM result->'ungated_output'
           OR EXISTS (SELECT 1 FROM unnest(ARRAY['query_closure','entity_consistency',
                 'time_consistency','set_consistency','source_support']) dimension
               WHERE result #>> ARRAY['grounding','assessment',dimension] IS NULL
                  OR result #>> ARRAY['grounding','assessment',dimension]
                       NOT IN ('pass','fail','not_applicable'))
           OR result #>> '{grounding,assessment,query_closure}'='not_applicable'
           OR result #>> '{grounding,assessment,source_support}'='not_applicable'
           OR (result #>> '{grounding,assessment,supported}')::boolean IS DISTINCT FROM
                NOT EXISTS (SELECT 1 FROM unnest(ARRAY['query_closure','entity_consistency',
                    'time_consistency','set_consistency','source_support']) dimension
                  WHERE result #>> ARRAY['grounding','assessment',dimension]='fail') THEN
          RAISE EXCEPTION 'typed_answer_grounding_assessment_invalid' USING ERRCODE='23514';
        END IF;
        FOR call IN SELECT value FROM jsonb_array_elements(result #> '{grounding,calls}') LOOP
          IF call->>'model_call_id'=NEW.answer_model_call_id
             OR NOT (result->'calls' @> jsonb_build_array(call))
             OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.model_calls c
                 WHERE c.model_call_id=call->>'model_call_id'
                   AND c.model_alias='grounding-qwen3.5-9b'
                   AND c.actual_model_id=result #>> '{grounding,actual_model_id}'
                   AND call->>'role' IN ('answer_grounding','grounding_format_retry')) THEN
            RAISE EXCEPTION 'typed_answer_grounding_call_missing' USING ERRCODE='23514';
          END IF;
        END LOOP;
      ELSIF result->'output' IS DISTINCT FROM result->'ungated_output'
         OR COALESCE(result->'grounding','null'::jsonb) != 'null'::jsonb THEN
        RAISE EXCEPTION 'typed_answer_abstention_changed' USING ERRCODE='23514';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute("""CREATE TRIGGER step8_typed_answer_lineage BEFORE INSERT
      ON pmmc_runtime.query_runtime_typed_answers FOR EACH ROW
      EXECUTE FUNCTION pmmc_runtime.enforce_step8_typed_answer()""")
    op.execute("""CREATE FUNCTION pmmc_runtime.enforce_step8_baseline_completion()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path=pg_catalog,pmmc_runtime AS $$
    BEGIN
      IF NEW.status!='completed' OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.run_manifests r
          WHERE r.run_id=NEW.run_id AND r.method!='pmmc'
            AND r.manifest_json #>> '{metadata,answer_contract_version}'=
            'typed-answer-contract-v3') THEN RETURN NEW; END IF;
      IF NEW.selected_count<=0 OR NEW.failed_count!=0
         OR NEW.succeeded_count!=NEW.selected_count
         OR (SELECT count(*) FROM pmmc_runtime.run_queries WHERE run_id=NEW.run_id)
              !=NEW.selected_count
         OR EXISTS (SELECT 1 FROM pmmc_runtime.run_queries q WHERE q.run_id=NEW.run_id
          AND (SELECT count(*) FROM pmmc_runtime.query_runtime_typed_answers a
              WHERE a.query_executor_run_id=q.run_id AND a.query_id=q.query_id
                AND a.terminal_state IN ('answered','abstained')
                AND a.payload #>> '{metadata,typed_result,answer_variant}'='primary') != 1)
         OR EXISTS (SELECT 1 FROM pmmc_runtime.final_answers WHERE run_id=NEW.run_id) THEN
        RAISE EXCEPTION 'typed_baseline_terminal_denominator_incomplete' USING ERRCODE='55000';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute("""CREATE TRIGGER step8_typed_baseline_completion BEFORE INSERT OR UPDATE
      ON pmmc_runtime.run_states FOR EACH ROW
      EXECUTE FUNCTION pmmc_runtime.enforce_step8_baseline_completion()""")
    op.execute("""CREATE FUNCTION pmmc_runtime.enforce_step8_query_completion()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path=pg_catalog,pmmc_runtime AS $$
    DECLARE expected_cells integer; match_source text; item record; active_treatment text;
    BEGIN
      IF TG_OP='DELETE' THEN
        IF OLD.schema_version='0.3.0' THEN
          RAISE EXCEPTION 'step8_query_identity_immutable' USING ERRCODE='55000';
        END IF;
        RETURN OLD;
      END IF;
      IF TG_OP='UPDATE' AND (OLD.schema_version='0.3.0' OR NEW.schema_version='0.3.0') THEN
        IF OLD.status IN ('completed','cancelled')
           OR (to_jsonb(NEW)-ARRAY['status','matched_count','executed_count','pmmc_answer_count',
                'fallback_answer_count','selected_answer_count','failed_count','started_at',
                'heartbeat_at','completed_at','error_type','error_summary'])
              IS DISTINCT FROM
              (to_jsonb(OLD)-ARRAY['status','matched_count','executed_count','pmmc_answer_count',
                'fallback_answer_count','selected_answer_count','failed_count','started_at',
                'heartbeat_at','completed_at','error_type','error_summary']) THEN
          RAISE EXCEPTION 'step8_query_identity_or_terminal_immutable' USING ERRCODE='55000';
        END IF;
      END IF;
      IF NEW.schema_version!='0.3.0' OR NEW.status!='completed' THEN RETURN NEW; END IF;
      expected_cells := CASE WHEN NEW.metadata->>'execute_full_grid'='true' THEN 48 ELSE 3 END;
      active_treatment := CASE WHEN NEW.metadata->>'treatment'='main' THEN 'pmmc'
                              ELSE NEW.metadata->>'treatment' END;
      match_source := COALESCE(NEW.metadata->>'source_match_run_id', NEW.query_executor_run_id);
      IF NEW.selected_query_count<=0 OR NEW.failed_count!=0 OR NEW.completed_at IS NULL
         OR NEW.selected_answer_count!=NEW.selected_query_count
         OR NEW.metadata->>'answer_contract_version' IS DISTINCT FROM 'typed-answer-contract-v3'
         OR jsonb_typeof(NEW.metadata->'execute_full_grid') IS DISTINCT FROM 'boolean'
         OR active_treatment IS NULL
         OR active_treatment NOT IN ('pmmc','fixed_planner','no_raw_image','no_reranker')
         OR jsonb_typeof(NEW.metadata->'declared_cells') IS DISTINCT FROM 'array'
         OR (NEW.metadata->>'expected_cell_count')::integer IS DISTINCT FROM expected_cells
         OR jsonb_array_length(NEW.metadata->'declared_cells') IS DISTINCT FROM expected_cells/3
         OR (SELECT count(DISTINCT grid) FROM jsonb_array_elements(NEW.metadata->'declared_cells')
               grid) != expected_cells/3
         OR (expected_cells=3 AND NEW.metadata->'declared_cells'
               IS DISTINCT FROM jsonb_build_array(
                 jsonb_build_array(NEW.question_budget,NEW.active_round)))
         OR (SELECT count(*) FROM pmmc_runtime.run_queries q
             WHERE q.run_id=NEW.query_executor_run_id AND q.snapshot_id=NEW.snapshot_id)
              != NEW.selected_query_count
         OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.compiler_runs c
             JOIN pmmc_runtime.run_states s ON s.run_id=c.run_id AND s.status='completed'
             JOIN pmmc_runtime.routing_projection_manifests r ON r.run_id=c.run_id
             WHERE c.run_id=NEW.source_compiler_run_id AND c.compiler_schema_version='0.9.0'
               AND r.routing_projection_id=NEW.routing_projection_id
               AND r.snapshot_id=NEW.snapshot_id
               AND c.operator_set_version='pmmc-operators-v7') THEN
        RAISE EXCEPTION 'step8_query_parent_or_grid_identity_invalid' USING ERRCODE='55000';
      END IF;
      IF match_source!=NEW.query_executor_run_id AND NOT EXISTS (
          SELECT 1 FROM pmmc_runtime.query_executor_runs source
           WHERE source.query_executor_run_id=match_source AND source.schema_version='0.3.0'
             AND source.snapshot_id=NEW.snapshot_id AND source.dataset=NEW.dataset
             AND source.source_compiler_run_id=NEW.source_compiler_run_id
             AND source.routing_projection_id=NEW.routing_projection_id
             AND source.config_hash=NEW.config_hash
             AND source.metadata->>'treatment'=NEW.metadata->>'treatment') THEN
        RAISE EXCEPTION 'step8_match_reuse_scope_invalid' USING ERRCODE='55000';
      END IF;
      FOR item IN SELECT q.* FROM pmmc_runtime.run_queries q
          WHERE q.run_id=NEW.query_executor_run_id LOOP
        IF (SELECT count(*) FROM pmmc_runtime.query_experiment_cell_results c
             WHERE c.query_executor_run_id=NEW.query_executor_run_id AND c.query_id=item.query_id)
             != expected_cells
           OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.query_runtime_phase_states s
              WHERE s.query_executor_run_id=NEW.query_executor_run_id
                AND s.query_id=item.query_id AND s.phase='answer')
           OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.query_route_decisions r
              WHERE r.query_executor_run_id=match_source AND r.query_id=item.query_id)
           OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.query_runtime_phase_states s
              WHERE s.query_executor_run_id=match_source AND s.query_id=item.query_id
                AND s.phase='match')
           OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.query_runtime_typed_answers a
              JOIN pmmc_runtime.pmmc_fallback_run_links f
                ON f.fallback_run_id=a.query_executor_run_id
              WHERE f.query_executor_run_id=NEW.query_executor_run_id
                AND f.snapshot_id=NEW.snapshot_id AND a.query_id=item.query_id
                AND a.terminal_state IN ('answered','abstained')
                AND a.payload #>> '{metadata,typed_result,answer_variant}'='primary'
                AND a.payload #>> '{metadata,typed_result,method}'='native_multimodal_rag') THEN
          RAISE EXCEPTION 'step8_query_terminal_denominator_incomplete' USING ERRCODE='55000';
        END IF;
        IF EXISTS (SELECT 1 FROM jsonb_array_elements(NEW.metadata->'declared_cells') grid
            WHERE (SELECT count(*) FROM pmmc_runtime.query_experiment_cell_results c
               WHERE c.query_executor_run_id=NEW.query_executor_run_id AND c.query_id=item.query_id
                 AND c.question_budget=(grid->>0)::integer
                 AND c.revision_depth=(grid->>1)::integer) != 3)
           OR EXISTS (
             SELECT 1 FROM jsonb_array_elements(NEW.metadata->'declared_cells') grid
             CROSS JOIN unnest(ARRAY[active_treatment,'no_fallback','forced_top1_counterfactual']) t
             WHERE (SELECT count(*) FROM pmmc_runtime.query_experiment_cell_results c
                     WHERE c.query_executor_run_id=NEW.query_executor_run_id
                       AND c.query_id=item.query_id AND c.question_budget=(grid->>0)::integer
                       AND c.revision_depth=(grid->>1)::integer AND c.treatment=t) != 1) THEN
          RAISE EXCEPTION 'step8_declared_budget_round_cell_missing' USING ERRCODE='55000';
        END IF;
      END LOOP;
      IF EXISTS (SELECT 1 FROM pmmc_runtime.query_experiment_cell_results c
          LEFT JOIN pmmc_runtime.query_runtime_typed_answers a
            ON a.typed_answer_id=c.selected_answer_id
          WHERE c.query_executor_run_id=NEW.query_executor_run_id AND (
            c.terminal_state='technical_failure'
            OR NOT EXISTS (SELECT 1 FROM pmmc_runtime.run_queries q
                WHERE q.run_id=NEW.query_executor_run_id AND q.query_id=c.query_id)
            OR (c.terminal_state='no_answer' AND c.selected_answer_id IS NOT NULL)
            OR (c.terminal_state IN ('answered','abstained') AND (
                a.typed_answer_id IS NULL OR a.query_id!=c.query_id
                OR a.terminal_state!=c.terminal_state
                OR (a.query_executor_run_id!=NEW.query_executor_run_id AND NOT EXISTS (
                   SELECT 1 FROM pmmc_runtime.pmmc_fallback_run_links f
                    WHERE f.query_executor_run_id=NEW.query_executor_run_id
                      AND f.fallback_run_id=a.query_executor_run_id))))
            OR (c.treatment IN ('no_fallback','forced_top1_counterfactual')
                AND c.selected_answer_id IS NOT NULL
                AND a.query_executor_run_id IS DISTINCT FROM NEW.query_executor_run_id)
            OR (c.treatment='no_fallback' AND c.selected_answer_id IS NOT NULL
                AND c.payload #>> '{route,outcome}' IS DISTINCT FROM 'program')
            OR (c.treatment IN ('no_fallback','forced_top1_counterfactual')
                AND (c.payload->>'active_fallback')::boolean))) THEN
        RAISE EXCEPTION 'step8_typed_selection_or_no_fallback_invalid' USING ERRCODE='55000';
      END IF;
      IF EXISTS (SELECT 1 FROM pmmc_runtime.query_experiment_cell_results c
          WHERE c.query_executor_run_id=NEW.query_executor_run_id
            AND c.payload->'binding' IS NOT NULL AND c.payload->'binding'!='null'::jsonb
            AND NOT EXISTS (SELECT 1 FROM pmmc_runtime.verified_qbank_routing_v4 r
               JOIN pmmc_runtime.benchmark_queries q
                 ON q.snapshot_id=NEW.snapshot_id AND q.query_id=c.query_id
               JOIN pmmc_runtime.benchmark_sessions available
                 ON available.snapshot_id=r.snapshot_id
                   AND available.conversation_id=r.conversation_id
                   AND available.session_id=r.available_after_session
               JOIN pmmc_runtime.benchmark_sessions cutoff
                 ON cutoff.snapshot_id=q.snapshot_id AND cutoff.conversation_id=q.conversation_id
                   AND cutoff.session_id=q.available_after_session
               WHERE r.routing_projection_id=NEW.routing_projection_id
                 AND r.compiler_run_id=NEW.source_compiler_run_id
                 AND r.snapshot_id=NEW.snapshot_id AND r.dataset=NEW.dataset
                 AND r.conversation_id=q.conversation_id
                 AND r.projection_round=c.revision_depth
                 AND r.generation_rank<=c.question_budget
                 AND r.plan_id=c.payload #>> '{binding,plan_id}'
                 AND r.qbank_id=c.payload #>> '{binding,qbank_id}'
                 AND r.program_hash=c.payload #>> '{binding,program_hash}'
                 AND available.session_order<=cutoff.session_order)) THEN
        RAISE EXCEPTION 'step8_binding_escaped_exact_projection' USING ERRCODE='55000';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute("""CREATE TRIGGER step8_query_completion BEFORE INSERT OR UPDATE OR DELETE
      ON pmmc_runtime.query_executor_runs FOR EACH ROW
      EXECUTE FUNCTION pmmc_runtime.enforce_step8_query_completion()""")
    for name in ("typed_answer", "baseline_completion", "query_completion", "runtime_fact_scope"):
        function = f"pmmc_runtime.enforce_step8_{name}()"
        op.execute(f"ALTER FUNCTION {function} OWNER TO pmmc_owner")
        op.execute(f"REVOKE ALL ON FUNCTION {function} FROM PUBLIC")


def downgrade() -> None:
    op.execute("""DO $$ BEGIN
      IF EXISTS (SELECT 1 FROM pmmc_runtime.query_runtime_typed_answers)
         OR EXISTS (SELECT 1 FROM pmmc_runtime.query_executor_runs WHERE schema_version='0.3.0')
         OR EXISTS (SELECT 1 FROM pmmc_evaluation.evaluation_runs
                    WHERE protocol_version='step8-query-evaluation-v1')
      THEN RAISE EXCEPTION 'cannot_downgrade_step8_with_typed_facts'; END IF;
    END $$""")
    op.execute("DROP TRIGGER step8_evaluation_completion ON pmmc_evaluation.evaluation_run_states")
    op.execute("DROP TRIGGER step8_ready_artifact_immutable ON pmmc_evaluation.run_artifacts")
    op.execute("DROP FUNCTION pmmc_evaluation.enforce_step8_evaluation_completion()")
    op.execute("DROP FUNCTION pmmc_evaluation.freeze_step8_ready_artifact()")
    for trigger, table in (
        ("step8_typed_answer_lineage", "query_runtime_typed_answers"),
        ("step8_typed_baseline_completion", "run_states"),
        ("step8_query_completion", "query_executor_runs"),
        ("step8_runtime_fact_scope", "query_experiment_cell_results"),
        ("step8_runtime_fact_scope", "query_runtime_phase_states"),
    ):
        op.execute(f"DROP TRIGGER {trigger} ON pmmc_runtime.{table}")
    for name in ("typed_answer", "baseline_completion", "query_completion", "runtime_fact_scope"):
        op.execute(f"DROP FUNCTION pmmc_runtime.enforce_step8_{name}()")
    _dispatch(False)
