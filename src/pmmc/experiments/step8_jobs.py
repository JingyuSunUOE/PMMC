"""Content-addressed stage DAG and scientific compatibility for benchmark workers."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import sha256_file
from pmmc.schemas.base import BaseSchema, JsonDict, NonEmptyStr
from pmmc.schemas.step8 import ResourceClass

# Paths are package-relative; @/ paths are project-relative data inputs. A ::
# selector owns a Python symbol plus its same-module helpers/constants/imports.
# Cross-module dependencies remain explicit: do not traverse imports into export,
# observability, or every operation of a multipurpose workflow/backend.
# Config YAML bytes are bound by operation-owned config_hashes in the campaign
# recipe, not again as broad stage sources shared by unrelated operations.
_MODEL_COMMON = (
    "models/execution.py",
    "models/campaign_audit.py",
    "models/registry.py",
    "models/locking.py",
    "models/protocols.py",
    "models/images.py",
    "models/routing.py",
    "models/tokenization.py",
    "schemas/model.py",
    "schemas/trace.py",
    "schemas/persistence.py",
    "schemas/experiment.py",
    "models/factory.py::create_model_execution_service",
    "models/factory.py::create_image_url_resolver",
    "models/factory.py::create_token_counter",
)
_GENERATION = _MODEL_COMMON + (
    "runtime/step8_capacity.py",
    "runtime/step8_capacity_native.py",
    "runtime/step8_capacity_requests.py",
    "runtime/step8_capacity_runtime.py",
    "runtime/step8_capacity_tokenization.py",
    "models/providers.py",
    "models/provider_config.py",
    "models/provider_schema.py",
    "models/provider_images.py",
    "models/provider_token_budget.py",
    "models/step8_accounting.py",
    "models/step8_rate_limits.py",
    "infrastructure/database/step8_provider_control.py",
    "models/factory.py::create_generation_gateway",
    "models/gateways.py::OpenAICompatibleGenerationGateway",
    "models/gateways.py::RoutedGenerationGateway",
)
_EMBEDDING = _MODEL_COMMON + (
    "models/factory.py::create_embedding_gateway",
    "models/gateways.py::OpenAICompatibleEmbeddingGateway",
    "models/gateways.py::VLLMMultimodalEmbeddingGateway",
    "models/gateways.py::ContrastiveEmbeddingGateway",
    "models/gateways.py::RoutedEmbeddingGateway",
)
_COMPILER = (
    "experiments/step8_partial_import.py",
    "experiments/step8_partial_cost_dependencies.py",
    "infrastructure/database/step8_partial_import.py",
    "experiments/step8_upstream_import.py",
    "infrastructure/database/step8_upstream_import.py",
    "compiler/ids.py",
    "compiler/config.py",
    "compiler/workflow.py",
    "compiler/program_workflow.py",
    "schemas/compiler.py",
    "schemas/qbank.py",
    "schemas/program.py",
    "schemas/feedback.py",
    "infrastructure/database/compiler_mappers.py",
    "infrastructure/database/compiler_models.py",
    "infrastructure/database/compiler_repositories.py::PostgresCompilerRunRepository",
    "infrastructure/database/compiler_repositories.py::PostgresCompilerUnitRepository",
    "infrastructure/database/repositories.py::PostgresMemoryRepository",
    "infrastructure/database/repositories.py::PostgresImageRepository",
)
_OPERATORS = (
    "memory/operators.py",
    "memory/evidence_roles.py",
    "retrieval/operators.py",
    "evidence_lineage.py",
    "compiler/execution.py",
    "compiler/program_runner.py",
    "compiler/dry_run.py",
    "schemas/program.py",
    "infrastructure/database/compiler_repositories.py::PostgresOperatorDataRepository",
    "infrastructure/database/compiler_repositories.py::PostgresMemoryProgramRepository",
    "infrastructure/database/planner_repositories.py::PostgresProgramDryRunRepository",
)
_ONLINE = (
    "runtime/query_workflow.py",
    "runtime/query_embeddings.py",
    "runtime/query_runner.py",
    "runtime/query_executor_config.py",
    "runtime/relevance_grid.py",
    "runtime/relation_grounding.py",
    "schemas/query_executor.py",
    "experiments/step8_instances.py",
    "infrastructure/database/query_executor_repositories.py",
    "infrastructure/database/query_executor_models.py",
    "infrastructure/database/repositories.py::PostgresRuntimeQueryRepository",
)
_ORACLE_EXECUTION = (
    "evaluation/step8_oracle.py",
    "evaluation/step8_oracle_execution.py",
    "evaluation/step8_oracle_hook.py",
    "evaluation/step8_oracle_production.py",
    "compiler/ids.py",
    "compiler/config.py",
    "memory/operators.py",
    "memory/evidence_roles.py",
    "retrieval/operators.py",
    "runtime/answer_context.py",
    "runtime/answering.py",
    "runtime/grounding.py",
    "runtime/relation_grounding.py",
    "baselines/context.py",
    "baselines/rendering.py",
    "schemas/program.py",
    "schemas/compiler.py",
    "schemas/query_executor.py",
    "infrastructure/database/compiler_repositories.py::PostgresQBankRoutingRepository",
    "infrastructure/database/compiler_repositories.py::PostgresOperatorDataRepository",
    "infrastructure/database/compiler_repositories.py::SessionScopedOperatorDataRepository",
    "infrastructure/database/planner_repositories.py::PostgresMemoryOperatorIndexRepository",
    "infrastructure/database/experiment_repositories.py::PostgresEmbeddingRepository",
    "infrastructure/database/repositories.py::PostgresMemoryRepository",
    "infrastructure/database/repositories.py::PostgresImageRepository",
    "@/prompts/runtime/pmmc_answer_contract_v3.txt",
    "@/prompts/runtime/answer_grounding_v1.txt",
)
_BASELINE_RUNTIME = (
    "baselines/suite.py",
    "baselines/suite_contract.py",
    "baselines/components.py",
    "baselines/rag_workflow.py",
    "baselines/rag_runner.py",
    "baselines/rag_config.py",
    "baselines/config.py",
    "infrastructure/database/baseline_repositories.py",
    "infrastructure/database/baseline_component_repositories.py",
    "schemas/baseline.py",
    "schemas/baseline_component.py",
    "schemas/baseline_suite.py",
)
_INDEX = (
    _EMBEDDING
    + _BASELINE_RUNTIME
    + (
        "baselines/rag_index.py",
        "baselines/rag_embedding.py",
        "baselines/rag_retrieval.py",
        "infrastructure/database/experiment_repositories.py::PostgresEmbeddingRepository",
        "infrastructure/database/experiment_repositories.py::PostgresRetrievalRepository",
    )
)
_COMPILER_INDEX = _COMPILER + (
    "compiler/memory_index.py",
    "compiler/memory_index_runner.py",
    "qbank/index.py",
    "qbank/matching.py",
    "schemas/questioner.py",
    "schemas/planner.py",
    "runtime/query_workflow.py::prepare_query_executor_runtime_embeddings",
    "runtime/query_embeddings.py",
    "infrastructure/database/questioner_mappers.py",
    "infrastructure/database/questioner_models.py",
    "infrastructure/database/questioner_repositories.py::PostgresQBankIndexRepository",
    "infrastructure/database/planner_repositories.py::PostgresMemoryOperatorIndexRepository",
    "infrastructure/database/planner_repositories.py::PostgresProgramQueryEmbeddingRepository",
)
_STAGE_INPUTS = {
    "snapshot": (
        "benchmarks",
        "experiments/step8_snapshot.py",
        "infrastructure/storage_sync.py",
        "infrastructure/database/importer.py",
        "infrastructure/database/mappers.py",
        "schemas/benchmark.py",
        "schemas/memory.py",
        "schemas/query.py",
    ),
    "partition": ("pilots/partition.py", "experiments/step8_selection.py", "schemas/pilot.py"),
    "unit_prepare": (
        "experiments/step8_partial_import.py",
        "experiments/step8_partial_cost_dependencies.py",
        "infrastructure/database/step8_partial_import.py",
        "compiler/units.py",
        "compiler/preparation.py",
        "compiler/questioner_config.py",
        "compiler/questioner_prompt.py",
        "compiler/questioner_native.py",
        "compiler/opportunities.py",
        "schemas/questioner.py",
        "infrastructure/database/questioner_fts.py",
        "infrastructure/database/repositories.py::PostgresMemoryRepository",
        "infrastructure/database/repositories.py::PostgresSessionRepository",
        "models/tokenization.py",
        "models/registry.py",
        "models/factory.py::create_token_counter",
        "baselines/corpus.py",
        "baselines/rendering.py",
        "baselines/suite.py",
        "baselines/suite_contract.py",
        "baselines/config.py",
        "baselines/rag_config.py",
        "baselines/workflow.py::_full_context_prompt",
        "baselines/rag_workflow.py::_answer_prompt_for_metadata",
        "baselines/typed_answer.py::typed_baseline_metadata",
        "runtime/grounding.py::load_step8_prompt",
        "schemas/answer_output.py",
        "@/prompts/runtime/pmmc_answer_contract_v3.txt",
        "@/prompts/runtime/answer_grounding_v1.txt",
        "schemas/compiler.py",
        "schemas/baseline.py",
        "schemas/baseline_suite.py",
        "infrastructure/database/baseline_repositories.py",
    ),
    "questioner": (
        "compiler/questioner.py",
        "compiler/questioner_prompt.py",
        "compiler/questioner_runner.py",
        "compiler/questioner_config.py",
        "compiler/questioner_native.py",
        "compiler/opportunities.py",
        "schemas/questioner.py",
        "infrastructure/database/questioner_repositories.py::PostgresQuestionerRepository",
        "infrastructure/database/questioner_mappers.py",
    )
    + _GENERATION
    + _COMPILER,
    "planner0": (
        "compiler/planner.py",
        "compiler/planner_prompt.py",
        "compiler/planner_runner.py",
        "compiler/planner_config.py",
        "schemas/planner.py",
        "memory/operators.py",
        "infrastructure/database/planner_repositories.py::PostgresPlannerRepository",
        "infrastructure/database/planner_mappers.py",
    )
    + _GENERATION
    + _COMPILER,
    "text_index": _INDEX + _COMPILER_INDEX,
    "contrastive_index": _INDEX + ("models/contrastive_worker.py",),
    "multimodal_index": _INDEX + _COMPILER_INDEX + (
        "runtime/joint_embeddings.py",
        "runtime/query_executor_config.py::QueryExecutorConfig.joint_relevance_enabled",
    ),
    "dry_run0": _COMPILER + _OPERATORS,
    "r0_gate": _COMPILER
    + (
        "compiler/r0_gate.py",
        "compiler/planner.py",
        "compiler/planner_config.py",
        "schemas/planner.py",
        "schemas/doubter.py",
        "infrastructure/database/compiler_repositories.py::PostgresQBankCompilerRepository",
        "infrastructure/database/compiler_repositories.py::PostgresMemoryProgramRepository",
        "infrastructure/database/planner_repositories.py",
        "infrastructure/database/planner_mappers.py",
        "infrastructure/database/doubter_repositories.py",
    ),
    "doubter": (
        "compiler/doubter.py",
        "compiler/doubter_prompt.py",
        "compiler/doubter_runner.py",
        "schemas/doubter.py",
        "compiler/planner.py",
        "compiler/planner_revision.py",
        "compiler/doubter_config.py",
        "compiler/doubter_graph.py",
        "compiler/planner_prompt.py",
        "compiler/planner_config.py",
        "infrastructure/database/doubter_repositories.py",
        "infrastructure/database/doubter_mappers.py",
    )
    + _GENERATION
    + _COMPILER
    + _OPERATORS,
    "projection": _COMPILER
    + (
        "compiler/verified_qbank.py",
        "schemas/doubter.py",
        "experiments/step8_instances.py",
        "infrastructure/database/doubter_repositories.py",
        "infrastructure/database/doubter_mappers.py",
        "infrastructure/database/doubter_models.py",
        "infrastructure/database/compiler_repositories.py::verify_compiler_run",
    ),
    "matching": (
        "runtime/joint_embeddings.py::scoped_joint_pairs",
        "runtime/joint_embeddings.py::joint_shards",
        "runtime/joint_embeddings.py::load_joint_records",
        "infrastructure/database/repositories.py::PostgresMemoryRepository.iter_visible",
        "runtime/qbank_matcher.py",
        "runtime/qbank_reranker.py",
        "runtime/query_backend.py",
        "runtime/query_executor.py",
        "runtime/evidence_sufficiency.py",
        "infrastructure/database/compiler_repositories.py::PostgresQBankRoutingRepository",
        "infrastructure/database/questioner_fts.py",
    )
    + _ONLINE
    + _GENERATION
    + _OPERATORS,
    "execution": _ONLINE
    + _OPERATORS
    + ("runtime/query_executor.py", "runtime/query_backend.py", "runtime/evidence_sufficiency.py"),
    "answers": (
        "runtime/answering.py",
        "runtime/answer_context.py",
        "schemas/answer_output.py",
        "baselines/context.py",
        "baselines/rag_context.py",
        "baselines/answering.py",
        "baselines/typed_answer.py",
        "baselines/runner.py",
        "baselines/workflow.py",
        "runtime/answer_diagnostics.py",
        "runtime/answer_policy.py",
        "runtime/query_backend.py",
        "runtime/query_executor.py",
        "@/prompts/runtime/pmmc_answer_contract_v3.txt",
    )
    + _ONLINE
    + _GENERATION
    + _BASELINE_RUNTIME,
    "grounding": _ONLINE
    + _GENERATION
    + _BASELINE_RUNTIME
    + (
        "runtime/grounding.py",
        "runtime/answering.py",
        "schemas/answer_output.py",
        "baselines/typed_answer.py",
        "baselines/runner.py",
        "runtime/query_executor.py",
        "runtime/query_backend.py",
        "@/prompts/runtime/answer_grounding_v1.txt",
    ),
    "evaluation": _GENERATION
    + _ORACLE_EXECUTION
    + (
        "evaluation/step8_workflow.py",
        "evaluation/step8_relevance_grid.py",
        "runtime/relevance_grid.py",
        "runtime/query_executor_config.py::JOINT_POLICY",
        "evaluation/step8_typed_gold.py",
        "evaluation/step8_process_metrics.py",
        "evaluation/step8_baseline_process.py",
        "evaluation/step8_semantic_workflow.py",
        "evaluation/step8_qbank_semantic.py",
        "evaluation/qbank_semantic.py",
        "schemas/semantic.py",
        "schemas/step8_process.py",
        "evaluation/qbank_coverage.py",
        "evidence_lineage.py",
        "evaluation/config.py",
        "evaluation/judge.py",
        "evaluation/evidence.py",
        "evaluation/facts.py",
        "evaluation/metrics.py",
        "evaluation/paper_statistics.py",
        "evaluation/repositories.py",
        "schemas/evaluation.py",
        "schemas/answer_output.py",
        "infrastructure/database/evaluation_repositories.py",
        "infrastructure/database/evaluation_mappers.py",
        "@/prompts/evaluation/baseline_judge_v1.txt",
    ),
    "export": ("experiments/step8_export.py",),
}

# These are entry points, not whole-file ownership. Calls to self._helpers and
# module-level identity functions are followed by the AST selector below.
_BACKEND_METHODS = {
    "snapshot": ("_snapshot",),
    "unit_prepare": ("_corpus", "_units", "_suite"),
    "questioner": ("_compiler", "_adopt_questioner"),
    "planner0": ("_compiler",),
    "text_index": ("_compiler", "_baseline_index", "_baseline_retrieve", "_query_embedding"),
    "multimodal_index": ("_compiler", "_baseline_index", "_baseline_retrieve", "_query_embedding"),
    "contrastive_index": ("_baseline_index", "_baseline_retrieve"),
    "dry_run0": ("_compiler",),
    "r0_gate": ("_compiler",),
    "doubter": ("_compiler",),
    "projection": ("_compiler",),
    "matching": ("_query_match",),
    "execution": ("_query_execute",),
    "answers": ("_baseline_answer", "_query_answer", "_native_fallback_generate"),
    "grounding": ("_baseline_grounding", "_query_grounding", "_native_fallback_ground"),
    "evaluation": ("_evaluate", "_primary_evaluate", "_analysis_evaluate", "_baseline_evaluate"),
}
_DATABASE_COMMON = (
    "infrastructure/database/engine.py",
    "infrastructure/database/work_limit.py",
)
SOURCE_OWNERS = {
    stage: tuple(
        sorted(
            set(
                owners
                + ("schemas/base.py", "compiler/ids.py")
                + ("experiments/step8_model_identity.py::stage_model_entry_hashes",)
                + (
                    ("experiments/step8_model_identity.py::compiler_model_entry_hashes",)
                    if stage
                    in {
                        "unit_prepare",
                        "questioner",
                        "planner0",
                        "text_index",
                        "multimodal_index",
                        "dry_run0",
                        "r0_gate",
                        "doubter",
                        "projection",
                        "matching",
                        "answers",
                        "grounding",
                    }
                    else ()
                )
                + _DATABASE_COMMON
                + (
                    ("experiments/release_compiler_reuse.py",
                     "experiments/step8_backend.py::ProductionComponentBackend.execute",
                     "experiments/step8_backend.py::ProductionComponentBackend.verify_recovery")
                    if "_compiler" in _BACKEND_METHODS[stage] else ()
                )
                + tuple(
                    "experiments/step8_backend.py::ProductionComponentBackend." + method
                    for method in ("__init__", *_BACKEND_METHODS[stage])
                )
                if stage in _BACKEND_METHODS
                else owners + (("schemas/base.py", "compiler/ids.py") if stage != "export" else ())
            )
        )
    )
    for stage, owners in _STAGE_INPUTS.items()
}


class _ScientificAst(ast.NodeTransformer):
    """Drop presentation-only statements, never control flow or returned values."""

    def visit_Import(self, node: ast.Import):  # type: ignore[no-untyped-def]
        node.names = [alias for alias in node.names if alias.name not in {"logging", "structlog"}]
        return node if node.names else None

    def visit_ImportFrom(self, node: ast.ImportFrom):  # type: ignore[no-untyped-def]
        return None if node.module in {"logging", "structlog"} else node

    def visit_Assign(self, node: ast.Assign):  # type: ignore[no-untyped-def]
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id in {"logging", "structlog"}
            and value.func.attr in {"getLogger", "get_logger"}
            and all(
                isinstance(target, ast.Name) and target.id in {"logger", "log"}
                for target in node.targets
            )
        ):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):  # type: ignore[no-untyped-def]
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        if isinstance(node.value, ast.Call):
            # print(await generate()) is a producer, not pure logging. Only erase
            # logging whose arguments have no opaque calls or execution effects.
            for argument in [*node.value.args, *(item.value for item in node.value.keywords)]:
                for child in ast.walk(argument):
                    if isinstance(child, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
                        return self.generic_visit(node)
                    if isinstance(child, ast.Call) and not (
                        isinstance(child.func, ast.Name) and child.func.id in {"str", "repr", "len"}
                    ):
                        return self.generic_visit(node)
            call = node.value.func
            if isinstance(call, ast.Name) and call.id in {
                "print",
                "_parent_preflight_event",
                "_workflow_phase_event",
            }:
                return None
            if (
                isinstance(call, ast.Attribute)
                and call.attr
                in {
                    "debug",
                    "info",
                    "warning",
                    "warn",
                    "error",
                    "exception",
                    "critical",
                    "log",
                }
                and isinstance(call.value, ast.Name)
                and call.value.id in {"logger", "logging", "log"}
            ):
                return None
        return self.generic_visit(node)


@lru_cache(maxsize=256)
def _parsed_scientific_source(source: str) -> ast.Module:
    return _ScientificAst().visit(ast.parse(source))


@lru_cache(maxsize=1024)
def _python_source_hash(source: str, selector: str) -> str:
    tree = _parsed_scientific_source(source)
    if not selector:
        return canonical_content_hash(ast.dump(tree, include_attributes=False))
    symbols: dict[str, ast.AST] = {}
    classes: dict[str, ast.ClassDef] = {}

    def assignments(node: ast.Assign | ast.AnnAssign, prefix: str = "") -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for binding_node in ast.walk(target):
                if isinstance(binding_node, ast.Name):
                    symbols[prefix + binding_node.id] = node

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name] = node
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols[node.name + "." + member.name] = member
                    elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                        assignments(member, node.name + ".")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                binding = alias.asname or alias.name.split(".")[0]
                symbols[binding] = (
                    ast.Import(names=[alias])
                    if isinstance(node, ast.Import)
                    else ast.ImportFrom(module=node.module, names=[alias], level=node.level)
                )
    if selector not in symbols:
        raise ValueError("declared scientific symbol is missing")
    pending = [selector]
    selected = {}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected_node = symbols[name]
        selected[name] = ast.dump(selected_node, include_attributes=False)
        class_name = name.split(".")[0] if "." in name else None
        dependencies = [selected_node]
        if class_name in classes:
            header = classes[class_name]
            class_header = ast.ClassDef(
                name=class_name,
                bases=header.bases,
                keywords=header.keywords,
                decorator_list=header.decorator_list,
                body=[],
            )
            dependencies.append(class_header)
            selected[class_name + ".<header>"] = ast.dump(
                class_header,
                include_attributes=False,
            )
        for child in (item for dependency in dependencies for item in ast.walk(dependency)):
            reference = (
                child.id
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                else None
            )
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                if child.value.id in {"self", "cls"} and class_name:
                    reference = class_name + "." + child.attr
            if reference in symbols and reference not in selected:
                pending.append(reference)
    return canonical_content_hash(selected)


def scientific_source_fingerprint(project_root: Path, stage: str) -> str:
    if stage not in SOURCE_OWNERS:
        raise ValueError("unknown stage source owner")
    package = project_root / "src" / "pmmc"
    files: set[tuple[Path, str]] = set()
    for owner in SOURCE_OWNERS[stage]:
        relative, _, selector = owner.partition("::")
        path = project_root / relative[2:] if relative.startswith("@/") else package / relative
        if path.is_dir() and not selector:
            files.update((item, "") for item in path.rglob("*.py"))
        elif path.is_file():
            files.add((path, selector))
        else:
            raise ValueError(f"stage source contract is missing: {stage}:{owner}")
    if not files:
        raise ValueError("stage has no available source files")
    entries = {}
    for path, selector in sorted(files):
        key = str(path.relative_to(project_root)) + ("::" + selector if selector else "")
        try:
            entries[key] = (
                _python_source_hash(path.read_text(encoding="utf-8"), selector)
                if path.suffix == ".py"
                else sha256_file(path)
            )
        except (SyntaxError, ValueError):
            raise ValueError(f"stage source contract is invalid: {stage}:{key}") from None
    return canonical_content_hash(
        {
            "protocol": "step8-stage-source-v2-ast",
            "stage": stage,
            "owners": SOURCE_OWNERS[stage],
            "files": entries,
        }
    )


class BenchmarkStageJob(BaseSchema):
    campaign_id: NonEmptyStr
    stage: NonEmptyStr
    component_key: NonEmptyStr
    resource_class: ResourceClass
    source_fingerprint: NonEmptyStr
    scientific_inputs: JsonDict
    dependency_ids: list[NonEmptyStr] = Field(default_factory=list)
    operation: NonEmptyStr
    parameters: JsonDict = Field(default_factory=dict)
    release_stages: tuple[Literal["main", "analysis", "baselines"], ...] = Field(
        default=(), exclude_if=lambda stages: not stages
    )

    @model_validator(mode="after")
    def _dependencies(self) -> BenchmarkStageJob:
        if self.stage not in SOURCE_OWNERS:
            raise ValueError("stage must have a declared scientific source owner")
        if len(set(self.dependency_ids)) != len(self.dependency_ids):
            raise ValueError("dependency IDs must be unique")
        if not self.scientific_inputs:
            raise ValueError("stage scientific inputs cannot be empty")
        return self

    @property
    def scientific_input_hash(self) -> str:
        return canonical_content_hash(
            {
                "stage": self.stage,
                "source": self.source_fingerprint,
                "inputs": self.scientific_inputs,
            }
        )

    @property
    def job_id(self) -> str:
        return canonical_content_hash(
            {
                "campaign": self.campaign_id,
                "component": self.component_key,
                "identity": self.scientific_input_hash,
            }
        )


class ComponentReceipt(BaseSchema):
    protocol: Literal["step8-component-receipt-v1"] = "step8-component-receipt-v1"
    job_id: NonEmptyStr
    scientific_input_hash: NonEmptyStr
    source_fingerprint: NonEmptyStr
    schema_revision: NonEmptyStr
    upstream_receipt_hashes: dict[NonEmptyStr, NonEmptyStr]
    output: JsonDict
    artifact_hashes: list[NonEmptyStr]
    row_counts: dict[NonEmptyStr, int]

    @model_validator(mode="after")
    def _counts(self) -> ComponentReceipt:
        if any(value < 0 for value in self.row_counts.values()):
            raise ValueError("component row counts cannot be negative")
        if len(set(self.artifact_hashes)) != len(self.artifact_hashes):
            raise ValueError("artifact references cannot repeat")
        return self

    @property
    def component_identity(self) -> str:
        return canonical_content_hash(
            {
                "scientific_input": self.scientific_input_hash,
                "upstream": self.upstream_receipt_hashes,
                "schema": self.schema_revision,
            }
        )

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


def validate_job_dag(
    jobs: Sequence[BenchmarkStageJob], *, existing_ids: set[str] | None = None
) -> None:
    by_id = {job.job_id: job for job in jobs}
    if len(by_id) != len(jobs) or len({job.component_key for job in jobs}) != len(jobs):
        raise ValueError("duplicate stage job or component key")
    existing = existing_ids or set()
    visited: set[str] = set()
    pending: set[str] = set()

    def visit(job_id: str) -> None:
        if job_id in pending:
            raise ValueError("stage DAG has a dependency cycle")
        if job_id in visited or job_id in existing:
            return
        if job_id not in by_id:
            raise ValueError("stage DAG references an unknown dependency")
        pending.add(job_id)
        for dependency in by_id[job_id].dependency_ids:
            visit(dependency)
        pending.remove(job_id)
        visited.add(job_id)

    for job_id in by_id:
        visit(job_id)


def invalidated_downstream(
    jobs: Sequence[BenchmarkStageJob],
    fingerprints: Mapping[str, str],
) -> list[str]:
    invalid = {job.job_id for job in jobs if fingerprints.get(job.stage) != job.source_fingerprint}
    while True:
        expanded = invalid | {job.job_id for job in jobs if set(job.dependency_ids) & invalid}
        if expanded == invalid:
            return sorted(job.component_key for job in jobs if job.job_id in invalid)
        invalid = expanded
