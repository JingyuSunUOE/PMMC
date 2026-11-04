"""Fail fast when a Docker build context contains incompatible PMMC source files."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

_MODEL_PROTOCOLS = {
    "EmbeddingGateway",
    "GenerationGateway",
    "ModelCallWriter",
    "TokenCounter",
    "TraceSink",
}
_COMPILER_ID_FUNCTIONS = {
    "canonical_content_hash",
    "compilation_unit_identity",
    "compilation_unit_selection_lineage",
    "stable_compiler_id",
}
_STEP6C_CONTRACTS = {
    "pmmc/evidence_lineage.py": {"EvidenceLineageResolver"},
    "pmmc/compiler/planner.py": {
        "PlannerProgramPolicy",
        "materialize_planner_program",
    },
    "pmmc/compiler/program_workflow.py": {
        "ProgramWorkflowPhase",
        "run_program_workflow",
    },
    "pmmc/compiler/program_runner.py": {"ProgramDryRunRunner"},
    "pmmc/compiler/memory_index_runner.py": {
        "MemoryOperatorIndexBuilder",
        "ProgramQueryEmbeddingBuilder",
    },
    "pmmc/compiler/program_pilot.py": {
        "build_program_pilot_bundle",
        "verify_program_pilot_bundle",
        "decide_program_pilot",
    },
    "pmmc/compiler/questioner_freeze.py": {
        "questioner_freeze_identity",
        "verify_questioner_freeze_compatibility",
    },
    "pmmc/smoke/step6c.py": {
        "Step6CPlannerConformanceError",
        "Step6CPlannerConformanceResult",
        "execute_step6c_planner_conformance",
        "verify_step6c_planner_conformance_payload",
    },
}
_STEP6C_GPU_VERIFIER_VERSION = "step6c-gpu-verifier-v2"
_STEP6C2_CONTRACTS = {
    "pmmc/evidence_lineage.py": {"EvidenceLineageResolver"},
    "pmmc/compiler/planner.py": {
        "PlannerProgramPolicy",
        "build_planner_requirement_contract",
        "materialize_planner_program",
    },
    "pmmc/compiler/program_pilot.py": {
        "build_program_pilot_bundle",
        "verify_program_pilot_bundle",
        "verify_program_pilot_archive",
        "decide_program_pilot",
    },
    "pmmc/evaluation/qbank_semantic.py": {
        "QBankSemanticJudgeService",
        "build_qbank_semantic_request",
        "derive_qbank_semantic_metrics",
        "select_semantic_candidate_union",
    },
    "pmmc/evaluation/qbank_semantic_workflow.py": {
        "run_qbank_semantic_match",
        "verify_qbank_semantic_match_run",
    },
    "pmmc/smoke/step6c2.py": {
        "Step6C2SemanticConformanceError",
        "Step6C2SemanticConformanceResult",
        "run_step6c2_semantic_conformance",
        "verify_step6c2_semantic_conformance_payload",
    },
}
_STEP6C2_GPU_VERIFIER_VERSION = "step6c2-gpu-verifier-v1"
_STEP6D_CONTRACTS = {
    "pmmc/compiler/doubter.py": {
        "derive_doubter_action",
        "derive_doubter_directives",
        "derive_dry_run_issues",
    },
    "pmmc/compiler/doubter_runner.py": {
        "DoubterRunner",
        "PostgresDoubterInputProvider",
        "verify_doubter_run",
    },
    "pmmc/compiler/r0_gate.py": {
        "build_r0_input_manifest",
        "verify_r0_input",
    },
    "pmmc/compiler/verified_qbank.py": {
        "freeze_verified_qbank",
        "verify_routing_projection",
    },
    "pmmc/compiler/program_pilot.py": {
        "decide_step6d_pilot",
        "verify_verified_qbank_freeze",
    },
    "pmmc/compiler/step6d_failure.py": {
        "build_step6d_failure_capsule",
        "verify_step6d_failure_capsule",
    },
    "pmmc/smoke/step6d.py": {
        "run_step6d_compiler_conformance",
        "verify_step6d_compiler_conformance_payload",
    },
}
_STEP6D_GPU_VERIFIER_VERSION = "step6d-gpu-pilot-verifier-v1"
_STEP7_CONTRACTS = {
    "pmmc/evaluation/export.py": {
        "_validate_export_answer_scope",
        "EvaluationExporter",
        "verify_evaluation_export",
    },
    "pmmc/evaluation/facts.py": {
        "PostgresEvaluationCaseLoader",
        "RuntimePmmcContext",
    },
    "pmmc/evaluation/workflow.py": {
        "_evaluation_failure_is_retryable",
        "_preflight_pmmc_evaluation_context_lineage",
        "run_evaluation_workflow",
    },
    "pmmc/infrastructure/database/query_executor_repositories.py": {
        "PostgresQueryExecutorRepository",
        "_idempotent_insert_statement",
        "_idempotent_identity_conflicts",
        "_query_stage_job_identity_conflicts",
    },
    "pmmc/runtime/qbank_matcher.py": {
        "counterfactual_program_candidate",
        "decide_runtime_route",
        "rank_runtime_match_grid",
    },
    "pmmc/runtime/evidence_sufficiency.py": {
        "assess_query_evidence_sufficiency",
    },
    "pmmc/runtime/query_executor.py": {
        "FrozenProgramBinding",
        "OnlineQueryExecutor",
        "OnlineQueryExecutorBackend",
        "_diagnostic_node",
    },
    "pmmc/runtime/qbank_reranker.py": {"BoundedQBankReranker", "reranker_model_call_id"},
    "pmmc/runtime/query_runner.py": {
        "QueryExecutorBatchRunner",
        "QueryExecutorBatchResult",
        "_active_query_worker_count",
        "_query_job_retryable",
    },
    "pmmc/runtime/query_embeddings.py": {
        "CoverageQueryEmbeddingImporter",
        "required_query_embedding_channels",
    },
    "pmmc/runtime/query_executor_config.py": {
        "apply_approved_routing_thresholds",
        "query_executor_protocol_hash",
        "query_executor_routing_policy_hash",
        "routing_policy_hash_from_protocol",
    },
    "pmmc/runtime/query_workflow.py": {
        "QueryExecutorParentContractError",
        "QueryExecutorItemFailures",
        "create_query_executor_manifest",
        "required_coverage_query_vector_pairs",
        "preflight_query_coverage_vectors",
        "validate_coverage_query_vectors",
        "run_query_executor_workflow",
        "safe_query_executor_error_diagnostics",
        "verify_query_executor_run",
    },
    "pmmc/runtime/step7_pilot.py": {
        "_approved_step7b_threshold_projection",
        "_step7_pilot_reuse_identity",
        "build_step7_pilot_bundle",
        "decide_step7b_pilot",
        "decide_step7_pilot",
        "verify_step7b_freeze",
        "verify_step7_pilot_archive",
        "verify_step7_pilot_bundle",
    },
}
_STEP7_GPU_VERIFIER_VERSION = "step7-gpu-pilot-verifier-v1"


def _parse_module(path: Path) -> ast.Module:
    if not path.is_file():
        raise RuntimeError(f"missing source contract: {path}")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _validate_compiler_ids(source_root: Path) -> None:
    path = source_root / "pmmc" / "compiler" / "ids.py"
    module = _parse_module(path)
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module == "pmmc.compiler.ids":
            raise RuntimeError("pmmc.compiler.ids imports itself")
        if isinstance(node, ast.Import) and any(
            alias.name == "pmmc.compiler.ids" for alias in node.names
        ):
            raise RuntimeError("pmmc.compiler.ids imports itself")
    functions = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(_COMPILER_ID_FUNCTIONS - functions)
    if missing:
        raise RuntimeError(
            "incomplete pmmc.compiler.ids build context; missing: "
            + ", ".join(missing)
        )


def _validate_step6c_contracts(source_root: Path) -> None:
    for relative_path, expected_symbols in _STEP6C_CONTRACTS.items():
        module = _parse_module(source_root / relative_path)
        symbols = {
            node.name
            for node in module.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        }
        missing = sorted(expected_symbols - symbols)
        if missing:
            raise RuntimeError(
                f"incomplete Step 6C source contract in {relative_path}; missing: "
                + ", ".join(missing)
            )


def _validate_step6c2_contracts(source_root: Path) -> None:
    for relative_path, expected_symbols in _STEP6C2_CONTRACTS.items():
        module = _parse_module(source_root / relative_path)
        symbols = {
            node.name
            for node in module.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        }
        missing = sorted(expected_symbols - symbols)
        if missing:
            raise RuntimeError(
                f"incomplete Step 6C.2 source contract in {relative_path}; "
                f"missing: {', '.join(missing)}"
            )


def _validate_step6d_contracts(source_root: Path) -> None:
    for relative_path, expected_symbols in _STEP6D_CONTRACTS.items():
        module = _parse_module(source_root / relative_path)
        symbols = {
            node.name
            for node in module.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        }
        missing = sorted(expected_symbols - symbols)
        if missing:
            raise RuntimeError(
                f"incomplete Step 6D source contract in {relative_path}; "
                f"missing: {', '.join(missing)}"
            )


def _validate_step7_contracts(source_root: Path) -> None:
    for relative_path, expected_symbols in _STEP7_CONTRACTS.items():
        module = _parse_module(source_root / relative_path)
        symbols = {
            node.name
            for node in module.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        }
        missing = sorted(expected_symbols - symbols)
        if missing:
            raise RuntimeError(
                f"incomplete Step 7 source contract in {relative_path}; "
                f"missing: {', '.join(missing)}"
            )


def validate_step6c_gpu_verifier_contract(project_root: Path) -> None:
    """Reject a partially synchronized Step 6C GPU verifier script."""

    relative_path = Path("scripts/verify_step6c_gpu_smoke.py")
    path = project_root / relative_path
    module = _parse_module(path)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pmmc.smoke"
        and any(
            alias.name == "verify_step6c_planner_conformance_payload"
            for alias in node.names
        )
        for node in module.body
    )
    call_count = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "verify_step6c_planner_conformance_payload"
        for node in ast.walk(module)
    )
    source = path.read_text(encoding="utf-8")
    if not imported or call_count < 2:
        raise RuntimeError(
            "incomplete Step 6C GPU verifier contract; conformance payload validation "
            "must be imported and applied to first/resumed summaries"
        )
    if _STEP6C_GPU_VERIFIER_VERSION not in source:
        raise RuntimeError(
            "incomplete Step 6C GPU verifier contract; missing version "
            + _STEP6C_GPU_VERIFIER_VERSION
        )
    if '"benchmark_zero_program_sample"' not in source:
        raise RuntimeError(
            "incomplete Step 6C GPU verifier contract; missing benchmark/conformance split"
        )


def validate_step6c2_gpu_verifier_contract(project_root: Path) -> None:
    """Reject a partially synchronized Step 6C.2 GPU verifier."""

    path = project_root / "scripts" / "verify_step6c2_gpu_smoke.py"
    module = _parse_module(path)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pmmc.smoke"
        and {
            alias.name
            for alias in node.names
        }
        >= {
            "verify_step6c_planner_conformance_payload",
            "verify_step6c2_semantic_conformance_payload",
        }
        for node in module.body
    )
    semantic_calls = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "verify_step6c2_semantic_conformance_payload"
        for node in ast.walk(module)
    )
    source = path.read_text(encoding="utf-8")
    required_fragments = (
        _STEP6C2_GPU_VERIFIER_VERSION,
        "qbank_semantic:model-stage",
        "planner_oracle_input_violation_count",
        "broad_session_round_match_count",
        "semantic_model_calls.jsonl",
    )
    if not imported or semantic_calls < 2:
        raise RuntimeError(
            "incomplete Step 6C.2 GPU verifier contract; semantic conformance "
            "must be imported and checked before and after resume"
        )
    missing = [value for value in required_fragments if value not in source]
    if missing:
        raise RuntimeError(
            "incomplete Step 6C.2 GPU verifier contract; missing: "
            + ", ".join(missing)
        )


def validate_step6d_gpu_verifier_contract(project_root: Path) -> None:
    """Reject a partially synchronized combined Step 6D GPU/pilot verifier."""

    path = project_root / "scripts" / "verify_step6d_gpu_pilot.py"
    module = _parse_module(path)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pmmc.smoke"
        and {
            alias.name
            for alias in node.names
        }
        >= {
            "verify_step6c2_semantic_conformance_payload",
            "verify_step6d_compiler_conformance_payload",
        }
        for node in module.body
    )
    source = path.read_text(encoding="utf-8")
    required_fragments = (
        _STEP6D_GPU_VERIFIER_VERSION,
        "verify_step6d_failure_capsule",
        "verify_program_pilot_archive",
        "compiler-program-quality-pilot-v3",
        "active_round",
        "max_round",
        "resume_repeated_model_calls",
        "managed_gpu_memory_released",
    )
    missing = [value for value in required_fragments if value not in source]
    if not imported or missing:
        raise RuntimeError(
            "incomplete Step 6D GPU verifier contract; missing: "
            + ", ".join(missing or ["required smoke verifier imports"])
        )


def validate_step7_gpu_verifier_contract(project_root: Path) -> None:
    """Reject a partial Step 7 prefix/executor/transfer verifier."""

    path = project_root / "scripts" / "verify_step7_gpu_pilot.py"
    module = _parse_module(path)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pmmc.smoke"
        and {alias.name for alias in node.names}
        >= {
            "verify_step6c2_semantic_conformance_payload",
            "verify_step6d_compiler_conformance_payload",
        }
        for node in module.body
    )
    source = path.read_text(encoding="utf-8")
    required_fragments = (
        _STEP7_GPU_VERIFIER_VERSION,
        "step7-independent-real-prefix-v1",
        "verify_step7_pilot_archive",
        "verify_step7_pilot_bundle",
        "query_count_per_dataset",
        "resume_repeated_model_calls",
        "cross_benchmark_isolation_verified",
        "managed_gpu_memory_released",
        "deferred_to_shared_model_stage",
        "dataset_program_export_verified",
        "failed_program_checks",
        "managed_runtime_lineage_verified",
    )
    missing = [value for value in required_fragments if value not in source]
    if not imported or missing:
        raise RuntimeError(
            "incomplete Step 7 GPU verifier contract; missing: "
            + ", ".join(missing or ["required smoke verifier imports"])
        )


def _validate_launcher_progress_contract(source_root: Path) -> None:
    path = source_root / "pmmc/experiments/progress.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reporter = next((
        item for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == "HierarchicalProgressReporter"
    ), None)
    operation = next((
        item for item in reporter.body
        if isinstance(item, ast.FunctionDef) and item.name == "report_operation"
    ), None) if reporter is not None else None
    if operation is None or [arg.arg for arg in operation.args.args] != ["self", "phase"]:
        raise RuntimeError(
            "incompatible launcher progress API: "
            "HierarchicalProgressReporter.report_operation(self, phase) is missing; "
            "sync launcher.py and progress.py together"
        )


def _literal_assignment(module: ast.Module, name: str) -> object:
    for node in module.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            assert isinstance(node, (ast.Assign, ast.AnnAssign))
            if node.value is not None:
                return ast.literal_eval(node.value)
    raise RuntimeError(f"missing source assignment: {name}")


def _validate_launcher_schema_contract(project_root: Path) -> None:
    # Standard-library only: this also runs on the host before the image is built.
    launcher = _parse_module(project_root / "src/pmmc/experiments/launcher.py")
    historical = _literal_assignment(launcher, "_STEP7B_ALEMBIC_REVISION")
    step8 = project_root / "src/pmmc/schemas/step8.py"
    expected = (
        _literal_assignment(_parse_module(step8), "STEP8_ALEMBIC_REVISION")
        if step8.is_file() else historical
    )
    revisions: set[str] = set()
    parents: set[str] = set()
    lineage: dict[str, str | None] = {}
    for path in sorted((project_root / "alembic/versions").glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = _parse_module(path)
        revision = _literal_assignment(module, "revision")
        parent = _literal_assignment(module, "down_revision")
        if not isinstance(revision, str) or revision in revisions:
            raise RuntimeError("invalid or duplicate migration revision")
        revisions.add(revision)
        lineage[revision] = parent
        if parent is not None:
            if not isinstance(parent, str):
                raise RuntimeError("Step 7B requires a single linear migration chain")
            parents.add(parent)
    if (
        not isinstance(expected, str) or not isinstance(historical, str)
        or historical not in revisions or revisions - parents != {expected}
        or parents - revisions
    ):
        raise RuntimeError(
            "incompatible launcher Alembic revision; "
            "sync historical launcher, current schema and all migrations together"
        )
    seen: set[str] = set()
    current: str | None = expected
    while current is not None and current not in seen:
        seen.add(current)
        current = lineage[current]
    if current is not None or seen != revisions:
        raise RuntimeError("incompatible launcher Alembic revision; migration chain is not linear")


def validate_source_contracts(source_root: Path) -> None:
    protocol_path = source_root / "pmmc" / "models" / "protocols.py"
    module = _parse_module(protocol_path)
    classes = {
        node.name for node in module.body if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef))
    }
    missing = sorted(_MODEL_PROTOCOLS - classes)
    if missing:
        raise RuntimeError(
            "incomplete pmmc.models.protocols build context; missing: " + ", ".join(missing)
        )
    _validate_compiler_ids(source_root)
    _validate_step6c_contracts(source_root)
    _validate_step6c2_contracts(source_root)
    _validate_step6d_contracts(source_root)
    _validate_step7_contracts(source_root)
    _validate_launcher_progress_contract(source_root)
    _validate_upstream_import_contract(source_root)
    _validate_fallback_recovery_contract(source_root)
    _validate_semantic_run_contract(source_root)
    _validate_model_artifact_export_contract(source_root)
    project_root = source_root.parent if source_root.name == "src" else None
    if project_root is not None:
        _validate_answer_policy_contract(project_root)
        _validate_launcher_schema_contract(project_root)
        validate_step6c_gpu_verifier_contract(project_root)
        validate_step6c2_gpu_verifier_contract(project_root)
        validate_step6d_gpu_verifier_contract(project_root)
        validate_step7_gpu_verifier_contract(project_root)


def _validate_model_artifact_export_contract(source_root: Path) -> None:
    module = _parse_module(source_root / "pmmc/infrastructure/experiment_export.py")
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
               and n.name == "ExperimentExporter")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)}
    inventory = methods.get("_load_referenced_model_artifacts")
    archive = methods.get("_archive_referenced_model_artifacts")
    valid_inventory = inventory is not None and any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and ast.unparse(n.func) == "RuntimeArtifactModel.artifact_id.in_"
        for n in ast.walk(inventory)
    ) and not any(
        isinstance(n, ast.Attribute) and ast.unparse(n) == "RuntimeArtifactModel.run_id"
        for n in ast.walk(inventory)
    )
    valid_owner = archive is not None and any(
        isinstance(n, ast.Compare)
        and ast.unparse(n) == "artifact.run_id != reference.source_run_id"
        for n in ast.walk(archive)
    )
    if not valid_inventory or not valid_owner:
        raise RuntimeError("incompatible model artifact export lineage; "
                           "sync experiment_export.py with its regression tests")
    print("STEP7B_MODEL_ARTIFACT_EXPORT_CONTRACT state=ready "
          "ownership=exact_model_call_owner_v1")


def _validate_semantic_run_contract(source_root: Path) -> None:
    schema = _parse_module(source_root / "pmmc/schemas/semantic.py")
    base = _parse_module(source_root / "pmmc/schemas/base.py")
    versions = _literal_assignment(schema, "SEMANTIC_RUN_SCHEMA_VERSIONS")
    expected = {f"qbank-semantic-match-v{v}": f"0.{v}.0" for v in range(1, 5)}
    if versions != expected or expected["qbank-semantic-match-v4"] != _literal_assignment(
        base, "QBANK_SEMANTIC_SCHEMA_VERSION",
    ):
        raise RuntimeError(
            "incompatible semantic run schema versions; sync semantic.py and base.py"
        )
    config = _parse_module(source_root / "pmmc/evaluation/qbank_semantic.py")
    config_class = next((n for n in config.body if isinstance(n, ast.ClassDef)
                         and n.name == "QBankSemanticMatchConfig"), None)
    method = next((n for n in config_class.body if isinstance(n, ast.FunctionDef)
                   and n.name == "run_schema_version"), None) if config_class else None
    if method is None or not any(
        isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "semantic_run_schema_version"
        and len(n.value.args) == 1
        and isinstance(n.value.args[0], ast.Attribute)
        and n.value.args[0].attr == "protocol_version" for n in ast.walk(method)
    ):
        raise RuntimeError("incompatible semantic config schema dispatch")
    workflow = _parse_module(source_root / "pmmc/evaluation/qbank_semantic_workflow.py")
    calls = [n for n in ast.walk(workflow) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "QBankSemanticMatchRun"]
    if not calls or any(not any(
        k.arg == "schema_version" and isinstance(k.value, ast.Attribute)
        and k.value.attr == "run_schema_version" for k in call.keywords
    ) for call in calls):
        raise RuntimeError("incompatible semantic workflow schema dispatch; "
                           "sync schema, config and workflow together")
    print("STEP7B_SEMANTIC_RUN_CONTRACT state=ready protocol=v4 run_schema=0.4.0 "
          "legacy_versions=1,2,3 checked_before_model_startup=true")


def _validate_answer_policy_contract(project_root: Path) -> None:
    policy = _parse_module(project_root / "src/pmmc/runtime/answer_policy.py")
    prompt = (project_root / "prompts/runtime/pmmc_answer_contract_v2_1.txt").read_text().strip()
    retry = _literal_assignment(policy, "CURRENT_RETRY_INSTRUCTION")
    if (
        _literal_assignment(policy, "CURRENT_PROMPT_VERSION") != "pmmc-answer-contract-v2.1"
        or _literal_assignment(policy, "CURRENT_RETRY_POLICY") != "answer-format-feedback-v2"
        or hashlib.sha256(prompt.encode()).hexdigest()
        != _literal_assignment(policy, "CURRENT_PROMPT_HASH")
        or not isinstance(retry, str)
        or hashlib.sha256(retry.encode()).hexdigest()
        != _literal_assignment(policy, "CURRENT_RETRY_SHA256")
    ):
        raise RuntimeError("answer prompt/retry hash mismatch; sync answer_policy.py and prompt")
    transitions = _literal_assignment(
        _parse_module(project_root / "src/pmmc/experiments/upstream_import.py"),
        "ANSWER_INPUT_TRANSITION",
    )
    assert isinstance(transitions, dict)
    for key, relative in {
        "query_executor_config": "configs/runtime/query-executor-v2.yaml",
        "query_executor_answer_prompt": "prompts/runtime/pmmc_answer_contract_v2_1.txt",
    }.items():
        data = (project_root / relative).read_bytes()
        if (hashlib.sha256(data).hexdigest(), len(data)) != transitions[key][1]:
            raise RuntimeError("answer upgrade source contract is not the audited byte identity")
    upgrade = _parse_module(project_root / "src/pmmc/baselines/answer_upgrade.py")
    names = {node.name for node in upgrade.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    launcher = _parse_module(project_root / "src/pmmc/experiments/launcher.py")
    prepare = next(node for node in ast.walk(launcher)
                   if isinstance(node, ast.FunctionDef) and node.name == "_prepare_step7_fallback")
    bindings = {
        node.targets[0].id for node in ast.walk(prepare)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "resolved_answer_upgrade_settings"
        and len(node.value.args) == 2
        and ast.unparse(node.value.args[1]) == "self.preparation.resolved_registry_path"
    }
    rebound = [node for node in ast.walk(prepare)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "rebind_native_fallback"]
    if (
        not {"AnswerUpgradeError", "resolved_answer_upgrade_settings",
             "_expected_retrieval_reuse_job", "verify_rebound_retrieval_job"} <= names
        or not rebound or not all(
            any(arg.arg == "settings" and isinstance(arg.value, ast.Name)
                and arg.value.id in bindings for arg in call.keywords) for call in rebound
        )
    ):
        raise RuntimeError("incompatible Step 7B answer upgrade registry; "
                           "sync launcher.py and answer_upgrade.py together")
    print("STEP7B_ANSWER_POLICY_CONTRACT state=ready prompt=v2.1 retry=v2 "
          "old_answers_reusable=false upgrade_registry=prepared_commit_resolved")
    runner = _parse_module(project_root / "src/pmmc/baselines/rag_runner.py")
    runner_calls = {node.func.id for node in ast.walk(runner)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    immutable = next(node for node in runner.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_immutable_job")
    if "verify_rebound_retrieval_job" not in runner_calls or not any(
        isinstance(node, ast.Set) and any(
            isinstance(value, ast.Constant) and value.value == "created_at" for value in node.elts
        ) for node in ast.walk(immutable)
    ):
        raise RuntimeError("incompatible retrieval receipt replay; "
                           "sync answer_upgrade.py and rag_runner.py together")
    print("STEP7B_RETRIEVAL_JOB_REUSE_CONTRACT state=ready source_attempt_required=true "
          "target_execution_attempts=0")


def _validate_upstream_import_contract(source_root: Path) -> None:
    module = _parse_module(source_root / "pmmc/experiments/upstream_import.py")
    names = {node.name for node in module.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not {
        "load_source", "restore_upstream", "verify_receipt", "select_components",
        "verify_model_lock_equivalence", "_receipt_inventory", "_receipt_component_configs",
        "_inherited_component_configs", "imported_program_audit_hash",
        "preflight_imported_program_audits",
    } <= names:
        raise RuntimeError("incomplete Step 7B upstream import source contract")
    if _literal_assignment(module, "MODEL_LOCK_EQUIVALENCE_POLICY") != (
        "lock-envelope-provenance-only-v2"
    ):
        raise RuntimeError("stale Step 7B model-lock equivalence policy; "
                           "sync upstream_import.py and verify_source_contracts.py together")
    if _literal_assignment(module, "COMPONENT_PROVENANCE_POLICY") != "sealed-component-origins-v1":
        raise RuntimeError("stale Step 7B chained component provenance contract")
    checkpoint = _parse_module(source_root / "pmmc/experiments/checkpoint.py")
    service = next(node for node in checkpoint.body
                   if isinstance(node, ast.ClassDef) and node.name == "ExperimentCheckpointService")
    restore = next(node for node in service.body
                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "restore")
    if "require_empty" not in {arg.arg for arg in restore.args.kwonlyargs}:
        raise RuntimeError("incompatible Step 7B checkpoint empty-database contract")
    launcher = _parse_module(source_root / "pmmc/experiments/launcher.py")
    owner = next(node for node in launcher.body
                 if isinstance(node, ast.ClassDef) and node.name == "ResearchExperimentLauncher")
    for method, helper in (
        ("_program_work_item", "imported_program_audit_hash"),
        ("_run_calibration", "preflight_imported_program_audits"),
    ):
        function = next(node for node in owner.body
                        if isinstance(node, ast.FunctionDef) and node.name == method)
        if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == helper for node in ast.walk(function)):
            raise RuntimeError("incompatible Step 7B compiler audit provenance; "
                               "sync launcher.py and upstream_import.py together")
    print("STEP7B_UPSTREAM_IMPORT_CONTRACT state=ready protocol=step7b-pre-online-import-v1 "
          "model_lock_policy=lock-envelope-provenance-only-v2 "
          "component_provenance=sealed-component-origins-v1 "
          "program_audit=sealed_compiler_origin preflight=before_model_calibration")


def _validate_fallback_recovery_contract(source_root: Path) -> None:
    suite = _parse_module(source_root / "pmmc/baselines/suite.py")
    create = next(
        node for node in suite.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "create_baseline_suite"
    )
    if "require_existing" not in {arg.arg for arg in create.args.kwonlyargs}:
        raise RuntimeError("missing fallback suite require_existing contract; sync suite.py")
    recovery = _parse_module(source_root / "pmmc/baselines/suite_recovery.py")
    if not any(isinstance(node, ast.AsyncFunctionDef)
               and node.name == "preflight_native_fallback_retrieval" for node in recovery.body):
        raise RuntimeError("missing fallback retrieval preflight; sync suite_recovery.py")
    print("STEP7B_FALLBACK_RECOVERY_CONTRACT state=ready identity_scope=source_launch "
          "retrieval_source=postgres")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_source_contracts.py <src-root>", file=sys.stderr)
        return 2
    try:
        validate_source_contracts(Path(argv[1]))
    except (OSError, SyntaxError, RuntimeError) as exc:
        print(f"source contract verification failed: {exc}", file=sys.stderr)
        return 1
    print("validated PMMC source contracts")
    print("STEP7B_PROGRESS_API_CONTRACT state=ready method=report_operation")
    print("SCHEMA_API_CONTRACT state=ready historical_revision_preserved=true "
          "current_revision_matches_migration_head=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
