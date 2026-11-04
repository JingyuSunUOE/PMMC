"""Native Step8 compiler persistence and completion contracts; no fact rewrites."""

from alembic import op

revision: str = "20260920_0031"
down_revision: str | None = "20260920_0030"
branch_labels: str | None = None
depends_on: str | None = None

RUNTIME = "pmmc_runtime"

# Each source is the post-0024 definition. Match counts fail closed on schema drift.
_REPLACEMENTS = (
    (
        "validate_questioner_fact_scope",
        "compiler_version IN ('0.6.0', '0.7.0', '0.8.0')",
        "compiler_version IN ('0.6.0', '0.7.0', '0.8.0', '0.9.0')",
    ),
    (
        "enforce_questioner_verifier_scope_v3",
        "WHEN '0.8.0' THEN 'pmmc-questioner-verifier-v6'",
        "WHEN '0.8.0' THEN 'pmmc-questioner-verifier-v6'\n"
        "                WHEN '0.9.0' THEN 'pmmc-questioner-verifier-v7'",
    ),
    (
        "enforce_questioner_verifier_scope_v3",
        "compiler_version IN ('0.5.0', '0.6.0', '0.7.0', '0.8.0')",
        "compiler_version IN ('0.5.0', '0.6.0', '0.7.0', '0.8.0', '0.9.0')",
    ),
    (
        "enforce_questioner_verifier_scope_v3",
        "compiler_version IN ('0.6.0', '0.7.0', '0.8.0')",
        "compiler_version IN ('0.6.0', '0.7.0', '0.8.0', '0.9.0')",
    ),
    (
        "validate_compiler_embedding_lineage_v4",
        "compiler.compiler_schema_version IN ('0.6.0', '0.7.0', '0.8.0')",
        "compiler.compiler_schema_version IN ('0.6.0', '0.7.0', '0.8.0', '0.9.0')",
    ),
    (
        "enforce_compiler_completion_v3",
        "compiler.compiler_schema_version NOT IN ('0.4.0', '0.5.0', '0.6.0', '0.7.0', '0.8.0')",
        "compiler.compiler_schema_version NOT IN "
        "('0.4.0', '0.5.0', '0.6.0', '0.7.0', '0.8.0', '0.9.0')",
    ),
)


def _rewrite_common_guards(*, forward: bool) -> None:
    # Resolve definitions on the server so --sql works without a live database.
    # Alembic's transaction rolls back all replacements if any source has drifted.
    for function, old, new in _REPLACEMENTS:
        name = f"{RUNTIME}.{function}()"
        source, target = (old, new) if forward else (new, old)
        source = "'" + source.replace("'", "''") + "'"
        target = "'" + target.replace("'", "''") + "'"
        op.execute(f"""DO $rewrite$
        DECLARE definition text;
        BEGIN
          definition := pg_get_functiondef(to_regprocedure('{name}'));
          IF definition IS NULL OR
             (length(definition)-length(replace(definition,{source},'')))
               / length({source}) IS DISTINCT FROM 1 THEN
            RAISE EXCEPTION 'step8_guard_rewrite_contract_mismatch: {name}';
          END IF;
          EXECUTE replace(definition,{source},{target});
        END $rewrite$""")


def _create_completion_guard() -> None:
    # Spaces after JSON colons prevent op.execute's TextClause from creating binds.
    op.execute(f"""CREATE FUNCTION {RUNTIME}.enforce_step8_compiler_completion()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path = pg_catalog, {RUNTIME} AS $$
    DECLARE compiler {RUNTIME}.compiler_runs%ROWTYPE;
            r0 {RUNTIME}.r0_input_manifests%ROWTYPE;
            routing {RUNTIME}.routing_projection_manifests%ROWTYPE;
            eligible_ids text[]; eligible integer; bad integer;
    BEGIN
      IF NEW.status != 'completed' THEN RETURN NEW; END IF;
      SELECT * INTO compiler FROM {RUNTIME}.compiler_runs WHERE run_id=NEW.run_id;
      IF NOT FOUND OR compiler.compiler_schema_version != '0.9.0' THEN RETURN NEW; END IF;
      IF compiler.target_stage IS DISTINCT FROM 'verified_qbank'
         OR compiler.operator_set_version IS DISTINCT FROM 'pmmc-operators-v7'
         OR compiler.questioner_prompt_version IS DISTINCT FROM 'pmmc-questioner-v3.6'
         OR compiler.planner_prompt_version IS DISTINCT FROM 'pmmc-planner-v1.5'
         OR compiler.doubter_prompt_version IS DISTINCT FROM 'pmmc-doubter-v1.3'
         OR compiler.program_policy_version IS DISTINCT FROM 'planner-program-policy-v6'
         OR compiler.r0_input_policy_version IS DISTINCT FROM 'r0-input-gate-v4'
         OR compiler.routing_projection_version IS DISTINCT FROM 'verified-qbank-projection-v4'
         OR compiler.active_round IS DISTINCT FROM 2 OR compiler.max_round IS DISTINCT FROM 3 THEN
        RAISE EXCEPTION 'Step8 compiler identity is incomplete' USING ERRCODE='55000';
      END IF;

      SELECT COALESCE(array_agg(q.qbank_id ORDER BY q.qbank_id), ARRAY[]::text[])
        INTO eligible_ids FROM {RUNTIME}.qbank_questions q
       WHERE q.run_id=NEW.run_id AND EXISTS (
         SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results g
          WHERE g.run_id=q.run_id AND g.qbank_id=q.qbank_id
            AND g.status='eligible_for_planning');
      eligible := cardinality(eligible_ids);
      SELECT count(*) INTO bad FROM {RUNTIME}.qbank_questions q
       WHERE q.run_id=NEW.run_id AND q.qbank_id=ANY(eligible_ids) AND (
         q.answer_type IS NULL OR NOT EXISTS (
           SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications v
           JOIN {RUNTIME}.questioner_verification_batches vb
             ON vb.verification_batch_id=v.verification_batch_id AND vb.run_id=v.run_id
            AND vb.compiler_unit_id=v.compiler_unit_id AND vb.generation_id=v.generation_id
            AND vb.verification_round=v.verification_round
           JOIN {RUNTIME}.model_calls vc
             ON vc.model_call_id=vb.model_call_id AND vc.run_id=vb.run_id
           WHERE v.run_id=q.run_id AND v.qbank_id=q.qbank_id
             AND v.verification_round=q.candidate_version AND v.verdict='pass'
             AND vb.prompt_version='pmmc-questioner-verifier-v7'
             AND vb.scope_qbank_id=v.qbank_id AND vb.input_count=1
             AND vb.metadata @> '{{"single_candidate_scope": true,
                 "provider_output_fallback": false, "repair_attempt_count": 0}}'::jsonb
             AND NOT (v.metadata @> '{{"provider_output_fallback": true}}'::jsonb)
             AND vc.status='success' AND vc.repair_attempt_count=0 AND vc.parse_error IS NULL));
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 eligible candidate lacks native verifier v7' USING ERRCODE='55000';
      END IF;

      SELECT * INTO r0 FROM {RUNTIME}.r0_input_manifests WHERE run_id=NEW.run_id;
      IF NOT FOUND OR NOT r0.hard_gate_passed
         OR cardinality(r0.protocol_integrity_issues) != 0
         OR r0.candidate_count != eligible
         OR r0.answerable_candidate_count != (
             SELECT count(*) FROM {RUNTIME}.qbank_questions q WHERE q.run_id=NEW.run_id
               AND q.qbank_id=ANY(eligible_ids) AND q.route_class='answerable')
         OR r0.control_candidate_count != (
             SELECT count(*) FROM {RUNTIME}.qbank_questions q WHERE q.run_id=NEW.run_id
               AND q.qbank_id=ANY(eligible_ids) AND q.route_class='control')
         OR r0.terminal_outcome_count != eligible
         OR r0.terminal_outcome_count != (
             SELECT count(*) FROM {RUNTIME}.planner_candidate_outcomes o WHERE o.run_id=NEW.run_id)
         OR r0.program_count != (
             SELECT count(*) FROM {RUNTIME}.memory_programs p
              WHERE p.run_id=NEW.run_id AND p.round_id=0)
         OR r0.execution_count != r0.program_count
         OR r0.execution_count != (
             SELECT count(*) FROM {RUNTIME}.program_dry_run_assessments a
             JOIN {RUNTIME}.memory_programs p ON p.plan_id=a.plan_id AND p.run_id=a.run_id
              WHERE a.run_id=NEW.run_id AND p.round_id=0) THEN
        RAISE EXCEPTION 'Step8 R0 denominator or input integrity mismatch' USING ERRCODE='55000';
      END IF;
      SELECT count(*) INTO bad FROM {RUNTIME}.memory_programs p
       WHERE p.run_id=NEW.run_id AND (
         NOT p.qbank_id=ANY(eligible_ids) OR p.operator_set_version!='pmmc-operators-v7'
         OR NOT EXISTS (SELECT 1 FROM {RUNTIME}.program_dry_run_assessments a
             WHERE a.run_id=p.run_id AND a.plan_id=p.plan_id AND a.qbank_id=p.qbank_id));
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 program scope or dry-run lineage mismatch' USING ERRCODE='55000';
      END IF;

      SELECT count(*) INTO bad FROM {RUNTIME}.doubter_assessments d
       LEFT JOIN {RUNTIME}.program_dry_run_assessments a
         ON a.run_id=d.run_id AND a.plan_id=d.plan_id AND a.execution_id=d.execution_id
        AND a.qbank_id=d.qbank_id
       LEFT JOIN {RUNTIME}.memory_programs p
         ON p.run_id=d.run_id AND p.plan_id=d.plan_id AND p.round_id=d.round_id
       WHERE d.run_id=NEW.run_id AND (
         d.prompt_version!='pmmc-doubter-v1.3' OR a.plan_id IS NULL OR p.plan_id IS NULL
         OR (d.action='pass' AND (
           NOT d.provider_schema_conformant OR NOT a.execution_success
           OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions q
                WHERE q.run_id=d.run_id AND q.qbank_id=d.qbank_id AND q.route_class='answerable'
                  AND (NOT a.nonempty_evidence OR NOT a.expected_evidence_complete
                       OR NOT a.required_modality_sufficient OR NOT a.pre_control_sufficient))
           OR EXISTS (SELECT 1 FROM {RUNTIME}.planner_requirement_contracts c
                WHERE c.run_id=d.run_id AND c.qbank_id=d.qbank_id AND (
                  ('raw_image'=ANY(c.required_capabilities) AND NOT a.raw_image_reached)
                  OR ('context'=ANY(c.required_capabilities) AND NOT a.context_evidence_contributed)
                  OR ('temporal'=ANY(c.required_capabilities)
                      AND NOT a.temporal_evidence_contributed)))
           OR (SELECT count(*) FROM {RUNTIME}.doubter_assessment_dimensions dim
                WHERE dim.assessment_id=d.assessment_id AND dim.run_id=d.run_id
                  AND dim.passed) != 10)));
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 Doubter pass violates source/path evidence contract'
          USING ERRCODE='55000';
      END IF;
      SELECT count(*) INTO bad FROM {RUNTIME}.planner_revision_lineage l
       WHERE l.run_id=NEW.run_id AND NOT EXISTS (
         SELECT 1 FROM {RUNTIME}.planner_revision_patches patch
          WHERE patch.run_id=l.run_id AND patch.qbank_id=l.qbank_id
            AND patch.target_round_id=l.target_round_id AND patch.applied
            AND patch.before_program_hash<>patch.after_program_hash
            AND jsonb_typeof(patch.metadata->'directive_effects')='array'
            AND jsonb_array_length(patch.metadata->'directive_effects')>0);
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 revision lacks typed directive effects' USING ERRCODE='55000';
      END IF;

      SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_manifests pm
       WHERE pm.run_id=NEW.run_id AND pm.round_id BETWEEN 0 AND 3;
      IF bad != 4 THEN
        RAISE EXCEPTION 'Step8 requires all r0-r3 projections' USING ERRCODE='55000';
      END IF;
      SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_manifests pm
       WHERE pm.run_id=NEW.run_id AND (
         pm.source_r0_manifest_id!=r0.r0_manifest_id OR pm.member_count!=eligible
         OR pm.pending_count!=0 OR pm.schema_version!='0.4.0'
         OR pm.member_count!=(SELECT count(*) FROM {RUNTIME}.qbank_round_projection_members m
                              WHERE m.projection_id=pm.projection_id AND m.run_id=pm.run_id)
         OR pm.accepted_count!=(SELECT count(*) FROM {RUNTIME}.qbank_round_projection_members m
                              WHERE m.projection_id=pm.projection_id AND m.run_id=pm.run_id
                                AND m.outcome IN ('accepted','carried_forward'))
         OR pm.rejected_count!=(SELECT count(*) FROM {RUNTIME}.qbank_round_projection_members m
                              WHERE m.projection_id=pm.projection_id AND m.run_id=pm.run_id
                                AND m.outcome IN ('rejected','max_rounds_exhausted')));
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 projection counts do not close the denominator'
          USING ERRCODE='55000';
      END IF;
      SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_members m
       JOIN {RUNTIME}.qbank_round_projection_manifests pm
         ON pm.projection_id=m.projection_id AND pm.run_id=m.run_id
       LEFT JOIN {RUNTIME}.memory_programs p
         ON p.plan_id=m.effective_plan_id AND p.run_id=m.run_id AND p.qbank_id=m.qbank_id
       WHERE m.run_id=NEW.run_id AND (
         NOT m.qbank_id=ANY(eligible_ids) OR m.outcome='pending'
         OR (m.outcome IN ('accepted','carried_forward') AND (
             p.plan_id IS NULL OR p.round_id>pm.round_id
             OR m.passed_at_round IS DISTINCT FROM p.round_id))
         OR (pm.round_id>0 AND m.outcome IN ('accepted','carried_forward') AND NOT EXISTS (
             SELECT 1 FROM {RUNTIME}.doubter_assessments d
              WHERE d.run_id=m.run_id AND d.qbank_id=m.qbank_id
                AND d.feedback_id=m.effective_feedback_id AND d.plan_id=p.plan_id
                AND d.round_id=p.round_id AND d.action='pass' AND d.provider_schema_conformant))
         OR (pm.round_id=0 AND m.effective_feedback_id IS NOT NULL));
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 projection scope/round mismatch or Doubter-contaminated R0'
          USING ERRCODE='55000';
      END IF;

      -- R0 readiness depends exclusively on P0 and its actual dry-run, not on Doubter.
      SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_members m
       JOIN {RUNTIME}.qbank_round_projection_manifests pm
         ON pm.projection_id=m.projection_id AND pm.run_id=m.run_id AND pm.round_id=0
       JOIN {RUNTIME}.qbank_questions q ON q.qbank_id=m.qbank_id AND q.run_id=m.run_id
       LEFT JOIN {RUNTIME}.memory_programs p
         ON p.qbank_id=m.qbank_id AND p.run_id=m.run_id AND p.round_id=0
       LEFT JOIN {RUNTIME}.program_dry_run_assessments a
         ON a.plan_id=p.plan_id AND a.run_id=p.run_id AND a.qbank_id=p.qbank_id
       WHERE m.run_id=NEW.run_id AND (
         (m.outcome='accepted') IS DISTINCT FROM COALESCE(
           a.execution_success AND CASE WHEN q.route_class='control' THEN a.abstain_triggered
           ELSE a.nonempty_evidence AND a.required_modality_sufficient
                AND a.pre_control_sufficient END, false)
         OR (m.outcome='accepted' AND m.effective_plan_id IS DISTINCT FROM p.plan_id)
         OR (m.outcome!='accepted' AND (m.effective_plan_id IS NOT NULL
                                       OR m.passed_at_round IS NOT NULL)));
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 R0 accepted membership differs from isolated dry-run readiness'
          USING ERRCODE='55000';
      END IF;

      SELECT * INTO routing FROM {RUNTIME}.routing_projection_manifests WHERE run_id=NEW.run_id;
      IF NOT FOUND OR routing.active_round!=2 OR routing.max_round!=3
         OR routing.schema_version!='0.4.0'
         OR routing.member_count!=(SELECT count(*) FROM {RUNTIME}.routing_projection_members rm
                                   WHERE rm.routing_projection_id=routing.routing_projection_id)
         OR routing.member_count IS DISTINCT FROM (
             SELECT pm.accepted_count FROM {RUNTIME}.qbank_round_projection_manifests pm
              WHERE pm.projection_id=routing.source_projection_id AND pm.run_id=NEW.run_id
                AND pm.round_id=2)
         OR eligible!=(SELECT count(*) FROM {RUNTIME}.qbank_decisions d WHERE d.run_id=NEW.run_id)
         OR routing.member_count!=(SELECT count(*) FROM {RUNTIME}.qbank_decisions d
                                    WHERE d.run_id=NEW.run_id AND d.accepted) THEN
        RAISE EXCEPTION 'Step8 active-r2 routing or decisions are incomplete' USING ERRCODE='55000';
      END IF;
      SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_members m
       JOIN {RUNTIME}.qbank_round_projection_manifests pm
         ON pm.projection_id=m.projection_id AND pm.run_id=m.run_id
       JOIN {RUNTIME}.memory_programs p ON p.plan_id=m.effective_plan_id AND p.run_id=m.run_id
       WHERE m.run_id=NEW.run_id AND m.outcome IN ('accepted','carried_forward')
         AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.verified_qbank_routing_v4 v
           WHERE v.compiler_run_id=m.run_id AND v.qbank_id=m.qbank_id
             AND v.plan_id=m.effective_plan_id AND v.projection_round=pm.round_id
             AND v.program_round=p.round_id);
      IF bad != 0 THEN
        RAISE EXCEPTION 'Step8 exact-round routing is unsafe or incomplete' USING ERRCODE='55000';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute(f"""CREATE TRIGGER step8_compiler_completion_integrity
        BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW
        EXECUTE FUNCTION {RUNTIME}.enforce_step8_compiler_completion()""")
    op.execute(f"ALTER FUNCTION {RUNTIME}.enforce_step8_compiler_completion() OWNER TO pmmc_owner")
    op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_step8_compiler_completion() FROM PUBLIC")


def upgrade() -> None:
    _rewrite_common_guards(forward=True)
    _create_completion_guard()


def downgrade() -> None:
    op.execute(f"""DO $$ BEGIN
      IF EXISTS (SELECT 1 FROM {RUNTIME}.compiler_runs WHERE compiler_schema_version='0.9.0')
      THEN RAISE EXCEPTION 'cannot_downgrade_step8_with_compiler_facts'; END IF;
    END $$""")
    op.execute(f"DROP TRIGGER step8_compiler_completion_integrity ON {RUNTIME}.run_states")
    op.execute(f"DROP FUNCTION {RUNTIME}.enforce_step8_compiler_completion()")
    _rewrite_common_guards(forward=False)
