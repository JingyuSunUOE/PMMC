"""Step8 source/API preflight; no historical scientific digest is reused.

The default AST pass needs only Python's standard library, including on a build
host without project dependencies. --runtime additionally loads the real configs
and provider schemas, loads runtime prompts through the installed package,
builds the operator registry, and constructs the dry-run runner in the freshly
built image. No database or model is invoked.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, cast

MIGRATIONS = {
    "20260920_0030": "20260911_0029",
    "20260920_0031": "20260920_0030",
    "20260920_0032": "20260920_0031",
    "20260922_0033": "20260920_0032",
    "20260922_0034": "20260922_0033",
    "20260927_0035": "20260922_0034",
    "20260927_0036": "20260927_0035",
    "20261104_0037": "20260927_0036",
    "20261104_0038": "20261104_0037",
}
SYMBOLS = {
    "runtime/device_accounting.py": {
        "AllocatedDeviceAccounting", "DeviceMemorySample", "DeviceLifecycleEvidence",
        "validate_device_lifecycle",
    },
    "infrastructure/database/image_asset_view.py": {"RuntimeImageAssetView"},
    "runtime/step8_capacity_requests.py": {
        "CapacityRequestGuard", "preflight_capacity_request_contract",
    },
    "runtime/step8_capacity_runtime.py": {"load_capacity_runtime"},
    "runtime/step8_capacity_tokenization.py": {
        "LocalCapacityTokenizer", "ServiceCapacityTokenizer",
    },
    "models/step8_accounting.py": {
        "preflight_provider_rate_card",
        "reconcile_unknown_provider_call",
        "Step8AccountingGateway",
        "SelectionApproval",
    },
    "models/step8_rate_limits.py": {"ProviderRateAdmission", "ProviderRateLimit"},
    "infrastructure/database/step8_provider_control.py": {"PostgresProviderControlRepository"},
    "compiler/questioner_native.py": {
        "native_support_catalog",
        "resolve_v36_draft",
        "allocate_opportunities",
        "questioner_verifier_v7_output_parser",
    },
    "compiler/workflow.py": {"ProgramWorkflowIdentity", "questioner_workflow_manifest_versions"},
    "compiler/program_workflow.py": {"run_program_workflow"},
    "compiler/program_runner.py": {"ProgramDryRunRunner"},
    "memory/operators.py": {"OperatorRegistry"},
    "retrieval/operators.py": {"build_operator_registry"},
    "runtime/answering.py": {"prepare_typed_answer_request", "parse_typed_answer"},
    "runtime/grounding.py": {
        "AuditedGroundingService",
        "build_grounding_request",
        "load_step8_prompt",
    },
    "models/providers.py": {
        "OpenAIResponsesGenerationGateway",
        "AnthropicGenerationGateway",
        "QwenGenerationGateway",
    },
    "experiments/step8_backend.py": {"ProductionComponentBackend", "compiler_instance_id"},
    "experiments/step8_runner.py": {"CampaignWorker"},
    "experiments/step8_supervisor.py": {"supervise_campaign"},
    "experiments/step8_import_scheduler.py": {
        "importable_job", "probe_import", "ImportOnlyComponentBackend", "drain_ready_imports",
    },
    "evaluation/step8_workflow.py": {"evaluate_step8_query", "bind_cell_inputs"},
    "evaluation/step8_process_metrics.py": {"load_process_facts", "evaluate_process_metrics"},
    "evaluation/step8_baseline_process.py": {"evaluate_baseline_retrieval_metrics"},
    "evaluation/step8_qbank_semantic.py": {
        "build_step8_semantic_plan",
        "evaluate_step8_semantic_plan",
        "Step8SemanticReport",
    },
    "evaluation/step8_semantic_workflow.py": {
        "prepare_query_semantics",
        "register_semantic_owner",
        "evaluate_query_semantics",
    },
    "experiments/step8_semantic_analysis.py": {"summarize_campaign_semantics"},
    "experiments/step8_compiler_analysis.py": {"summarize_campaign_compiler"},
    "experiments/step8_cost_analysis.py": {"summarize_campaign_costs"},
    "experiments/step8_cost_dependencies.py": {
        "collect_rag_index_cost_dependencies", "validate_rag_index_dependencies",
    },
    "experiments/step8_questioner_cost_dependencies.py": {
        "collect_questioner_cost_dependencies", "validate_questioner_cost_dependencies",
    },
    "experiments/step8_r0_cost_dependencies.py": {
        "collect_r0_prefix_cost_dependencies", "validate_r0_cost_dependencies",
    },
    "experiments/step8_pmmc_cost_dependencies.py": {
        "collect_pmmc_cost_dependencies", "validate_write_graph", "validate_execution_dependencies",
    },
    "experiments/step8_performance_analysis.py": {"summarize_campaign_performance"},
    "experiments/step8_relevance_proofs.py": {
        "relevance_point", "project_relevance_score", "verify_relevance_projection",
    },
    "infrastructure/database/step8_repositories.py": {"PostgresBenchmarkRepository"},
}
FIELDS = {
    "runtime/step8_capacity.py": {
        "Step8PoolCapacityReceipt": {"measurement_source", "device_lifecycle"},
    },
    "infrastructure/settings.py": {"Settings": {"gpu_memory_accounting"}},
    "schemas/questioner.py": {
        "QuestionerClaimDraftV36": {"support_alias", "claim_text", "subject", "assertion"},
        "QuestionerNativeSupport": {
            "owning_memory_id",
            "speaker",
            "source_field",
            "start_char",
            "end_char",
            "exact_text",
        },
        "QuestionerVerificationItemV7": {"claims", "answer_question_closed"},
    },
    "compiler/questioner_config.py": {"QuestionerConfig": {"memory_raw_image_policy"}},
    "compiler/planner_config.py": {"PlannerConfig": {"memory_raw_image_policy"}},
    "schemas/answer_output.py": {
        "AnswerContractV3": {
            "answer_kind",
            "answer_text",
            "selected_image_aliases",
            "supported",
            "used_evidence_aliases",
            "abstain_reason",
        },
        "TypedAnswerResult": {"ungated_output", "grounding", "calls", "terminal_state"},
    },
    "schemas/step8.py": {
        "ScientificProtocol": {"source_fingerprints", "config_hashes", "model_lock_hash"},
        "MethodProtocolFreeze": {"protocol", "decision", "protocol_integrity_passed"},
        "CompiledQBankInstanceManifest": {"protocol_hash", "compiler_run_id", "scope_hash"},
    },
    "infrastructure/database/step8_models.py": {
        "BenchmarkCampaignModel": {"campaign_id", "protocol_hash", "freeze_hash", "phase"},
        "BenchmarkStageJobModel": {"job_id", "campaign_id", "status"},
    },
    "infrastructure/database/step8_provider_control.py": {
        "ProviderRateScopeModel": {"scope_hash", "policy", "blocked_until"},
        "ProviderRateAdmissionModel": {"dispatch_id", "request_hash", "scope_hash", "token_units"},
        "ProviderReconciliationModel": {
            "request_id",
            "request_hash",
            "reconciliation_hash",
            "payload",
        },
    },
    "infrastructure/database/query_executor_models.py": {
        "QueryRuntimeTypedAnswerModel": {
            "typed_answer_id",
            "query_executor_run_id",
            "terminal_state",
            "answer_identity_hash",
        },
        "QueryExperimentCellResultModel": {
            "cell_id",
            "treatment",
            "question_budget",
            "revision_depth",
            "selected_answer_id",
        },
    },
}
CONFIGS = {
    "configs/compiler/questioner-v3.6.yaml": {
        "protocol_version": "pmmc-questioner-v3.6",
        "verifier_prompt_version": "pmmc-questioner-verifier-v7",
        "memory_raw_image_policy": "allow",
    },
    "configs/compiler/planner-v1.5.yaml": {"protocol_version": "pmmc-planner-v1.5"},
    "configs/compiler/doubter-v1.3.yaml": {"protocol_version": "pmmc-doubter-v1.3"},
    "configs/compiler/pmmc-v7.yaml": {"operator_set_version": "pmmc-operators-v7"},
    "configs/runtime/query-executor-v3.yaml": {
        "protocol_version": "pmmc-query-executor-v3",
        "answer_contract_version": "typed-answer-contract-v3",
    },
}
RUNTIME_PROMPTS = ("pmmc_answer_contract_v3.txt", "answer_grounding_v1.txt")
ASSETS = (
    *(f"prompts/runtime/{name}" for name in RUNTIME_PROMPTS),
    "configs/models/step8.yaml",
    "configs/models/step8-providers.yaml",
    "configs/models/step8-provider-rate-card.template.yaml",
    "configs/execution/h100-replicated-8gpu-step8.yaml",
    "configs/runtime/h100-step8.yaml",
)


def module_contract(path: Path, symbols: set[str], fields: dict[str, set[str]]) -> ast.Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if missing := (symbols | fields.keys()) - definitions.keys():
        raise ValueError(f"missing Step8 API in {path.name}: {sorted(missing)}")
    for name, expected in fields.items():
        definition = definitions[name]
        if not isinstance(definition, ast.ClassDef):
            raise ValueError(f"Step8 API is not a class: {name}")
        actual = {
            node.target.id
            for node in definition.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        if missing_fields := expected - actual:
            raise ValueError(f"missing Step8 fields in {name}: {sorted(missing_fields)}")
    return tree


def migration_contract(root: Path) -> list[str]:
    found: dict[str, tuple[str | None, Path]] = {}
    for path in sorted((root / "alembic/versions").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        values: dict[str, Any] = {}
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else ([node.target] if isinstance(node, ast.AnnAssign) else [])
            )
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"revision", "down_revision"}:
                    values[target.id] = ast.literal_eval(node.value)  # type: ignore[attr-defined]
        if revision := values.get("revision"):
            if revision in found:
                raise ValueError("duplicate Alembic revision")
            found[revision] = (values.get("down_revision"), path)
    for revision, parent in MIGRATIONS.items():
        if revision not in found or found[revision][0] != parent:
            raise ValueError(f"Step8 migration chain mismatch: {revision}")
        module_contract(found[revision][1], {"upgrade", "downgrade"}, {})
    parents = {parent for parent, _path in found.values()}
    expected_heads = set(MIGRATIONS) - set(MIGRATIONS.values())
    if len(expected_heads) != 1 or set(found) - parents != expected_heads:
        raise ValueError("Step8 integration requires the sole declared additive migration head")
    return [str(found[key][1].relative_to(root)) for key in MIGRATIONS]


class _NoRuntimeIO:
    """Fail closed if constructor preflight starts using runtime dependencies."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("Step8 source preflight must not access runtime dependencies")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Step8 source preflight must not access runtime dependencies")


def runtime_contract(root: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from pmmc.compiler.config import OPERATOR_SET_V7, load_compiler_config
    from pmmc.compiler.doubter_config import load_doubter_config
    from pmmc.compiler.planner_config import load_planner_config
    from pmmc.compiler.program_runner import ProgramDryRunRunner
    from pmmc.compiler.questioner_config import load_questioner_config
    from pmmc.compiler.questioner_prompt import questioner_prompt_sha256, questioner_response_schema
    from pmmc.compiler.workflow import questioner_workflow_manifest_versions
    from pmmc.memory.operators import OperatorDataRepository
    from pmmc.retrieval.operators import build_operator_registry
    from pmmc.runtime.grounding import load_step8_prompt
    from pmmc.runtime.query_executor_config import load_query_executor_config
    from pmmc.schemas.answer_output import AnswerContractV3
    from pmmc.schemas.base import (
        SUPPORTED_COMPILER_SCHEMA_VERSIONS,
        SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS,
    )

    for name, expected in CONFIGS.items():
        config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Step8 config identity mismatch: {name}")
    questioner = load_questioner_config(root / "configs/compiler/questioner-v3.6.yaml")
    planner = load_planner_config(root / "configs/compiler/planner-v1.5.yaml")
    load_doubter_config(root / "configs/compiler/doubter-v1.3.yaml")
    foundation = load_compiler_config(root / "configs/compiler/pmmc-v7.yaml")
    load_query_executor_config(root / "configs/runtime/query-executor-v3.yaml")
    registry = build_operator_registry(cast(OperatorDataRepository, _NoRuntimeIO()), foundation)
    if registry.version != "pmmc-operators-v7" or registry.names() != OPERATOR_SET_V7:
        raise ValueError("Step8 runtime operator registry identity mismatch")
    capabilities = {
        "input_capabilities",
        "output_capabilities",
        "allowed_source_step_types",
        "modality_contribution",
        "server_derived_parameters",
    }
    for spec in registry.specs():
        if (
            spec.version != registry.version
            or not capabilities <= spec.metadata.keys()
            or not spec.metadata["output_capabilities"]
        ):
            raise ValueError("Step8 runtime operator capability contract missing")
    dry_run = ProgramDryRunRunner(
        sessions=cast(async_sessionmaker[AsyncSession], _NoRuntimeIO()),
        config=foundation,
        lease_seconds=900,
        concurrency=1,
    )
    if dry_run.config.operator_set_version != registry.version:
        raise ValueError("Step8 dry-run constructor changed operator identity")
    if (
        not questioner.is_v36
        or questioner.memory_raw_image_policy != planner.memory_raw_image_policy
    ):
        raise ValueError("Step8 compiler treatment dispatch mismatch")
    versions = questioner_workflow_manifest_versions(questioner)
    if versions != {"compiler_schema_version": "0.9.0", "schema_version": "0.15.0"}:
        raise ValueError("Step8 workflow manifest versions are not explicit")
    if "0.9.0" not in SUPPORTED_COMPILER_SCHEMA_VERSIONS:
        raise ValueError("Step8 compiler schema unsupported")
    if "0.15.0" not in SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS:
        raise ValueError("Step8 experiment schema unsupported")
    questioner_response_schema("pmmc-questioner-v3.6")
    answer_fields = set(AnswerContractV3.model_json_schema()["properties"])
    if answer_fields != FIELDS["schemas/answer_output.py"]["AnswerContractV3"]:
        raise ValueError("typed answer provider field contract drift")
    runtime_prompt_hashes = {}
    for filename in RUNTIME_PROMPTS:
        source_bytes = (root / "prompts/runtime" / filename).read_bytes()
        # The runtime API strips surrounding whitespace. Do not alter settings
        # or import paths to make a misplaced installed-package asset pass.
        expected_bytes = source_bytes.decode("utf-8").strip().encode("utf-8")
        loaded = load_step8_prompt(filename)
        if not isinstance(loaded, str) or loaded.encode("utf-8") != expected_bytes:
            raise ValueError(f"Step8 runtime prompt content mismatch: {filename}")
        runtime_prompt_hashes[filename] = {
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "loaded_utf8_sha256": hashlib.sha256(loaded.encode("utf-8")).hexdigest(),
        }
    return {
        "manifest_versions": versions,
        "runtime_prompt_hashes": runtime_prompt_hashes,
        "operator_runtime_preflight": {
            "operator_set_version": registry.version,
            "operator_count": len(registry.names()),
            "registry_hash": registry.resolved_hash,
            "capabilities_verified": True,
            "dry_run_runner_constructed": True,
        },
        "inline_prompt_hashes": {
            version: questioner_prompt_sha256(version)
            for version in ("pmmc-questioner-v3.6", "pmmc-questioner-verifier-v7")
        },
    }


def verify_source_contracts(root: Path, *, runtime: bool = False) -> dict[str, Any]:
    root = root.resolve()
    files = []
    for name in sorted(SYMBOLS.keys() | FIELDS.keys()):
        path = root / "src/pmmc" / name
        module_contract(path, SYMBOLS.get(name, set()), FIELDS.get(name, {}))
        files.append(str(path.relative_to(root)))
    files.extend(migration_contract(root))
    files.extend((*CONFIGS, *ASSETS))
    for name in files:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or unsafe Step8 contract asset: {name}")
    return {
        "protocol": "step8-source-contracts-v1",
        "success": True,
        "runtime_checked": runtime,
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
        **(runtime_contract(root) if runtime else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_source_contracts(args.root, runtime=args.runtime), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
