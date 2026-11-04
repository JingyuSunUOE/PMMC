"""Paired-denominator orchestration for the five foundational baselines."""

from __future__ import annotations

import json
from pathlib import Path

from pmmc.baselines.config import (
    load_full_context_config,
    prompt_sha256,
)
from pmmc.baselines.rag_config import load_rag_config
from pmmc.baselines.rag_embedding import RAG_METHODS
from pmmc.baselines.rag_workflow import (
    RagRunPhase,
    _answer_prompt_for_metadata,
    run_rag_workflow,
)
from pmmc.baselines.suite_contract import (
    STRICT_ANSWER_CONTRACT,
    STRICT_PROMPT_HASH,
    STRICT_SUITE_CONTRACT,
    TYPED_SUITE_CONTRACT,
    BaselineSuiteContractError,
    suite_answer_contract,
    validate_suite_run_answer_contract,
)
from pmmc.baselines.typed_answer import typed_baseline_metadata
from pmmc.baselines.workflow import _full_context_prompt, run_full_context_workflow
from pmmc.benchmarks.ids import stable_hash
from pmmc.benchmarks.selection import selection_hash
from pmmc.infrastructure.database.baseline_repositories import PostgresCorpusRepository
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.evaluation_repositories import (
    PostgresBaselineSuiteRepository,
)
from pmmc.infrastructure.database.experiment_repositories import PostgresRunRepository
from pmmc.infrastructure.database.repositories import (
    PostgresRuntimeQueryRepository,
    PostgresSnapshotRepository,
)
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.schemas import (
    FOUNDATIONAL_BASELINE_METHODS,
    BaselineMethod,
    BaselineSuiteManifest,
    BaselineSuiteQuery,
    BaselineSuiteRunLink,
    BaselineSuiteStatus,
    BenchmarkQuery,
    BenchmarkSelection,
    RunManifest,
)
from pmmc.schemas.answer_output import TYPED_ANSWER_CONTRACT_VERSION
from pmmc.schemas.baseline_suite import (
    is_internal_pmmc_fallback_suite,
    is_step8_caption_fallback_suite,
)

_FULL_CONTEXT_METHODS = {
    BaselineMethod.FULL_CONTEXT_CAPTIONS,
    BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_suite_files(
    root: Path,
    manifest: BaselineSuiteManifest,
    selection: BenchmarkSelection,
    selected: list[BenchmarkQuery],
) -> Path:
    directory = root / "baseline-suites" / manifest.suite_id
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
    _write_json(
        directory / "selection.json",
        {
            "selection": selection.model_dump(mode="json"),
            "selection_hash": manifest.selection_hash,
            "query_ids": [item.query.query_id for item in selected],
            "source_locators": [item.source.model_dump(mode="json") for item in selected],
        },
    )
    selected_path = directory / "selected_queries.jsonl"
    temporary = selected_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(item.to_json() + "\n")
    temporary.replace(selected_path)
    return directory


def _suite_config_payload(
    *,
    snapshot_id: str,
    selection_fingerprint: str,
    methods: list[BaselineMethod],
    answer_model_alias: str,
    answer_prompt_hash: str,
    decoding_config: dict[str, object],
    total_context_tokens: int,
    reserved_output_tokens: int,
    evaluation_policy_version: str,
    seed: int,
    paper_mode: bool,
    corpus_id: str | None,
    suite_metadata: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_version": "foundational-baseline-suite-v1",
        "snapshot_id": snapshot_id,
        "selection_hash": selection_fingerprint,
        "methods": [item.value for item in methods],
        "answer_model_alias": answer_model_alias,
        "answer_prompt_hash": answer_prompt_hash,
        "decoding_config": decoding_config,
        "total_context_tokens": total_context_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "query_image_policy": "preserve",
        "evaluation_policy_version": evaluation_policy_version,
        "seed": seed,
        "paper_mode": paper_mode,
        "corpus_id": corpus_id,
    }
    if suite_metadata:
        payload["suite_metadata"] = suite_metadata
    return payload


def baseline_suite_child_run_id(suite_id: str, method: BaselineMethod) -> str:
    return "baseline-run-" + stable_hash([suite_id, method.value, "primary"])[0:24]


async def create_baseline_suite(
    *,
    settings: Settings,
    snapshot_id: str,
    selection: BenchmarkSelection,
    corpus_id: str | None,
    methods: list[BaselineMethod] | None = None,
    answer_model_alias: str | None = None,
    metadata: dict[str, object] | None = None,
    require_existing: bool = False,
) -> tuple[BaselineSuiteManifest, Path]:
    """Select once, freeze the ordered denominator, and register a suite."""

    selected_methods = list(methods or FOUNDATIONAL_BASELINE_METHODS)
    suite_metadata = dict(metadata or {})
    reserved_metadata = {
        "protocol_version",
        "selection",
        "corpus_id",
        "answer_suite_contract_version",
    }
    if reserved_metadata.intersection(suite_metadata):
        raise ValueError("suite metadata cannot override protocol-owned fields")
    if suite_metadata.get("smoke_fixture") is True:
        if suite_metadata.get("paper_result") is not False:
            raise ValueError("smoke suites must declare paper_result=false")
        if settings.mode is AppMode.PAPER:
            raise ValueError("smoke suites must be bootstrapped with dev-mode settings")
    paper_component = (
        suite_metadata.get("paper_component") is True
        and suite_metadata.get("paper_result") is False
        and len(selected_methods) == 1
        and isinstance(suite_metadata.get("baseline_family_id"), str)
        and isinstance(suite_metadata.get("baseline_component_id"), str)
    )
    internal_pmmc_fallback = is_internal_pmmc_fallback_suite(
        selected_methods,
        {
            **suite_metadata,
            "selection": selection.model_dump(mode="json"),
        },
    )
    if (
        settings.mode is AppMode.PAPER
        and set(selected_methods) != set(FOUNDATIONAL_BASELINE_METHODS)
        and not paper_component
        and not internal_pmmc_fallback
        and not is_step8_caption_fallback_suite(selected_methods, suite_metadata)
    ):
        raise ValueError("paper baseline suites require five methods or a bound component")
    if any(item in RAG_METHODS for item in selected_methods) and not corpus_id:
        raise ValueError("suites containing RAG methods require corpus_id")
    if settings.mode is AppMode.PAPER and selection.include_excluded:
        raise ValueError("paper mode cannot include excluded benchmark queries")

    full_config = load_full_context_config(
        settings.resolved_full_context_config_path,
        answer_model_alias=answer_model_alias,
    )
    rag_config = load_rag_config(
        settings.resolved_rag_config_path,
        answer_model_alias=answer_model_alias,
    )
    suite_metadata = typed_baseline_metadata(full_config, suite_metadata)
    suite_metadata = typed_baseline_metadata(rag_config, suite_metadata)
    typed = suite_metadata.get("answer_contract_version") == TYPED_ANSWER_CONTRACT_VERSION
    if suite_metadata.get("allow_memory_raw_images") is False and (
        not typed or selected_methods != [BaselineMethod.NATIVE_MULTIMODAL_RAG]
    ):
        raise ValueError("no-raw fallback suites require a typed native-MM-only child")
    prompt = _full_context_prompt(full_config, settings.project_root, suite_metadata)
    prompt_hash = prompt_sha256(prompt)
    rag_prompt = _answer_prompt_for_metadata(rag_config, settings.project_root, suite_metadata)
    if prompt_hash != prompt_sha256(rag_prompt):
        raise ValueError("full-context and RAG baselines must share one answer prompt")
    shared_fields = (
        "answer_model_alias",
        "total_context_tokens",
        "reserved_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "max_output_tokens",
    )
    if any(
        getattr(full_config, name) != getattr(rag_config, name)
        for name in shared_fields
        if not typed or name not in {"reserved_output_tokens", "max_output_tokens"}
    ):
        raise ValueError("full-context and RAG configs disagree on a controlled variable")
    strict = suite_metadata.get("answer_contract_version") == STRICT_ANSWER_CONTRACT
    if strict and not internal_pmmc_fallback:
        raise BaselineSuiteContractError("strict_suite_requires_internal_development_fallback")
    strict_prompt_hash = STRICT_PROMPT_HASH
    if strict:
        # The loader verifies the immutable hash of the selected prompt version.
        strict_prompt_hash = prompt_sha256(
            _answer_prompt_for_metadata(
                rag_config,
                settings.project_root,
                suite_metadata,
            )
        )

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        snapshots = PostgresSnapshotRepository(sessions)
        queries = PostgresRuntimeQueryRepository(sessions)
        snapshot = await snapshots.get_manifest(snapshot_id)
        if selection.datasets != [snapshot.dataset]:
            raise ValueError("suite selection dataset must match snapshot")
        selected = await queries.select(snapshot_id, selection)
        if not selected:
            raise ValueError("suite selection produced no queries")
        if corpus_id is not None:
            async with sessions() as session:
                corpus = await PostgresCorpusRepository(session).get_manifest(corpus_id)
            if corpus.snapshot_id != snapshot_id:
                raise ValueError("suite corpus and snapshot differ")
        fingerprint = selection_hash(selection, selected)
        decoding: dict[str, object] = {
            "temperature": full_config.temperature,
            "top_p": full_config.top_p,
            "seed": full_config.seed,
            "max_output_tokens": full_config.max_output_tokens,
        }
        payload = _suite_config_payload(
            snapshot_id=snapshot_id,
            selection_fingerprint=fingerprint,
            methods=selected_methods,
            answer_model_alias=full_config.answer_model_alias,
            answer_prompt_hash=prompt_hash,
            decoding_config=decoding,
            total_context_tokens=full_config.total_context_tokens,
            reserved_output_tokens=full_config.reserved_output_tokens,
            evaluation_policy_version=snapshot.evaluation_policy_version,
            seed=selection.seed,
            paper_mode=settings.mode is AppMode.PAPER,
            corpus_id=corpus_id,
            suite_metadata=suite_metadata,
        )
        reserved_output_tokens = full_config.reserved_output_tokens
        if typed:
            reserved_output_tokens = 512
            decoding = {
                **decoding,
                "max_output_tokens": 512,
                "thinking_mode": "disabled",
                "format_retries": 1,
            }
            suite_metadata = {
                **suite_metadata,
                "answer_suite_contract_version": TYPED_SUITE_CONTRACT,
            }
            payload = {
                **payload,
                "decoding_config": decoding,
                "reserved_output_tokens": 512,
                "suite_metadata": suite_metadata,
            }
        if strict:
            # Look up the exact legacy configuration before minting a corrected
            # suite. Keeping an existing ID preserves its indexed child lineage.
            legacy_id = "baseline-suite-" + stable_hash(payload)[:24]
            async with sessions() as session:
                repository = PostgresBaselineSuiteRepository(session)
                try:
                    legacy = await repository.get_manifest(legacy_id)
                except KeyError:
                    legacy = None
                if legacy is not None:
                    if legacy.config_hash != stable_hash(payload):
                        raise BaselineSuiteContractError("legacy_suite_config_hash_mismatch")
                    contract = suite_answer_contract(legacy)
                    try:
                        child = await PostgresRunRepository(session).get_manifest(
                            baseline_suite_child_run_id(legacy_id, selected_methods[0])
                        )
                    except KeyError:
                        child = None
                    if child is not None:
                        validate_suite_run_answer_contract(legacy, child)
                    print(
                        "BASELINE_SUITE_ANSWER_PREFLIGHT "
                        + json.dumps(
                            {
                                "state": "ready",
                                "mode": contract.mode,
                                "child_manifest_verified": child is not None,
                                "max_output_tokens": contract.decoding["max_output_tokens"],
                                "manifest_rewritten": False,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    return legacy, _write_suite_files(
                        settings.runs_path,
                        legacy,
                        selection,
                        selected,
                    )
            prompt_hash = strict_prompt_hash
            decoding = {**decoding, "max_output_tokens": 512}
            reserved_output_tokens = 512
            suite_metadata = {
                **suite_metadata,
                "answer_suite_contract_version": STRICT_SUITE_CONTRACT,
            }
            payload = {
                **payload,
                "answer_prompt_hash": prompt_hash,
                "decoding_config": decoding,
                "reserved_output_tokens": reserved_output_tokens,
                "suite_metadata": suite_metadata,
            }
        config_hash = stable_hash(payload)
        suite_id = "baseline-suite-" + config_hash[:24]
        async with sessions() as session, session.begin():
            repository = PostgresBaselineSuiteRepository(session)
            try:
                manifest = await repository.get_manifest(suite_id)
            except KeyError:
                if require_existing:
                    raise BaselineSuiteContractError("restored_suite_identity_not_found") from None
                manifest = BaselineSuiteManifest(
                    suite_id=suite_id,
                    dataset=snapshot.dataset,
                    snapshot_id=snapshot_id,
                    selection_hash=fingerprint,
                    methods=selected_methods,
                    answer_model_alias=full_config.answer_model_alias,
                    answer_prompt_hash=prompt_hash,
                    decoding_config=decoding,
                    total_context_tokens=full_config.total_context_tokens,
                    reserved_output_tokens=reserved_output_tokens,
                    query_image_policy="preserve",
                    evaluation_policy_version=snapshot.evaluation_policy_version,
                    seed=selection.seed,
                    paper_mode=settings.mode is AppMode.PAPER,
                    config_hash=config_hash,
                    metadata={
                        "protocol_version": "foundational-baseline-suite-v1",
                        "selection": selection.model_dump(mode="json"),
                        "corpus_id": corpus_id,
                        **suite_metadata,
                    },
                )
                suite_answer_contract(manifest)
                suite_queries = [
                    BaselineSuiteQuery(
                        suite_id=suite_id,
                        snapshot_id=snapshot_id,
                        query_id=item.query.query_id,
                        ordinal=ordinal,
                        metadata={"source": item.source.model_dump(mode="json")},
                    )
                    for ordinal, item in enumerate(selected)
                ]
                await repository.register(manifest, suite_queries)
            if manifest.config_hash != config_hash:
                raise ValueError("suite ID already belongs to another configuration")
        return manifest, _write_suite_files(settings.runs_path, manifest, selection, selected)
    finally:
        await engine.dispose()


async def run_baseline_suite(
    *,
    settings: Settings,
    suite_id: str,
    phase: RagRunPhase = RagRunPhase.ALL,
    methods: list[BaselineMethod] | None = None,
) -> Path:
    """Run or resume suite children while preserving one frozen denominator."""
    phase = RagRunPhase(phase)

    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session, session.begin():
            repository = PostgresBaselineSuiteRepository(session)
            manifest = await repository.get_manifest(suite_id)
            state = await repository.get_state(suite_id)
            links = {item.method: item for item in await repository.get_run_links(suite_id)}
            if state.status is BaselineSuiteStatus.READY:
                return settings.runs_path / "baseline-suites" / suite_id
            if state.status is BaselineSuiteStatus.CREATED:
                await repository.transition(suite_id, BaselineSuiteStatus.RUNNING)
            elif state.status is not BaselineSuiteStatus.RUNNING:
                raise ValueError(f"suite cannot resume from {state.status.value}")
        selection = BenchmarkSelection.model_validate(manifest.metadata.get("selection"))
        corpus_id = manifest.metadata.get("corpus_id")
        run_metadata: dict[str, object] = {
            key: manifest.metadata[key]
            for key in (
                "smoke_fixture",
                "paper_result",
                "internal_pmmc_fallback",
                "partition_id",
                "partition_role",
                "partition_dataset_assignment_hash",
                "answer_contract_version",
                "answer_prompt_version",
                "answer_output_schema_version",
                "grounding_policy_version",
                "grounding_model_alias",
                "grounding_prompt_hash",
                "memory_raw_image_policy",
                "allow_memory_raw_images",
                "method_protocol_hash",
            )
            if key in manifest.metadata
        }
        selected_methods = list(methods or manifest.methods)
        if len(selected_methods) != len(set(selected_methods)):
            raise ValueError("suite method filter cannot contain duplicates")
        if not set(selected_methods).issubset(set(manifest.methods)):
            raise ValueError("suite method filter contains an unconfigured method")
        for method in selected_methods:
            if method in links:
                continue
            run_id = baseline_suite_child_run_id(suite_id, method)
            run_exists = (settings.runs_path / run_id / "run_manifest.json").is_file()
            contract = suite_answer_contract(manifest)
            if run_exists:
                child = RunManifest.model_validate_json(
                    (settings.runs_path / run_id / "run_manifest.json").read_text()
                )
                validate_suite_run_answer_contract(manifest, child)
            print(
                "BASELINE_SUITE_ANSWER_PREFLIGHT "
                + json.dumps(
                    {
                        "state": "ready",
                        "mode": contract.mode,
                        "phase": phase.value,
                        "child_manifest_verified": run_exists,
                        "max_output_tokens": contract.decoding.get("max_output_tokens"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if method in _FULL_CONTEXT_METHODS:
                if phase is RagRunPhase.RETRIEVE:
                    continue
                await run_full_context_workflow(
                    settings=settings,
                    method=None if run_exists else method,
                    snapshot_id=None if run_exists else manifest.snapshot_id,
                    selection=None if run_exists else selection,
                    answer_model_alias=manifest.answer_model_alias,
                    resume_run_id=run_id if run_exists else None,
                    run_id=None if run_exists else run_id,
                    run_metadata=None if run_exists else run_metadata,
                    phase=phase.value,
                )
            else:
                if (
                    phase in {RagRunPhase.ANSWER, RagRunPhase.GENERATE, RagRunPhase.GROUND}
                    and not run_exists
                ):
                    raise ValueError(
                        f"RAG suite answer phase requires completed retrieval: {method.value}"
                    )
                await run_rag_workflow(
                    settings=settings,
                    method=None if run_exists else method,
                    snapshot_id=None if run_exists else manifest.snapshot_id,
                    corpus_id=None if run_exists else str(corpus_id),
                    selection=None if run_exists else selection,
                    answer_model_alias=manifest.answer_model_alias,
                    phase=phase,
                    resume_run_id=run_id if run_exists else None,
                    run_id=None if run_exists else run_id,
                    run_metadata=None if run_exists else run_metadata,
                )
                if phase is RagRunPhase.RETRIEVE:
                    continue
            if phase is RagRunPhase.GENERATE:
                continue
            async with sessions() as session, session.begin():
                repository = PostgresBaselineSuiteRepository(session)
                linked = await repository.link_run(
                    BaselineSuiteRunLink(
                        suite_id=suite_id,
                        method=method,
                        run_id=run_id,
                    )
                )
            print(
                "BASELINE_SUITE_ANSWER_LINK "
                + json.dumps(
                    {
                        "state": "ready",
                        "method": method.value,
                        "mode": contract.mode,
                        "explicit_binding": "answer_contract_binding" in linked.metadata,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        async with sessions() as session, session.begin():
            repository = PostgresBaselineSuiteRepository(session)
            final_links = await repository.get_run_links(suite_id)
            if {item.method for item in final_links} == set(manifest.methods):
                await repository.transition(suite_id, BaselineSuiteStatus.READY)
        print(
            "BASELINE_SUITE_PROGRESS "
            + json.dumps(
                {
                    "phase": phase.value,
                    "linked_methods": len(final_links),
                    "total_methods": len(manifest.methods),
                    "state": "ready" if len(final_links) == len(manifest.methods) else "running",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return settings.runs_path / "baseline-suites" / suite_id
    finally:
        await engine.dispose()


async def verify_baseline_suite(settings: Settings, suite_id: str) -> dict[str, object]:
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    try:
        async with sessions() as session:
            repository = PostgresBaselineSuiteRepository(session)
            manifest = await repository.get_manifest(suite_id)
            state = await repository.get_state(suite_id)
            queries = await repository.get_queries(suite_id)
            links = await repository.get_run_links(suite_id)
            contract_issues: list[str] = []
            for link in links:
                run = await PostgresRunRepository(session).get_manifest(link.run_id)
                try:
                    binding = validate_suite_run_answer_contract(manifest, run)
                except BaselineSuiteContractError as exc:
                    contract_issues.append(exc.diagnostics["contract_error_code"])
                    continue
                if (
                    binding["mode"] != "manifest"
                    and link.metadata.get("answer_contract_binding") != binding
                ):
                    contract_issues.append("suite_answer_binding_missing_or_invalid")
        issues: list[str] = []
        issues.extend(contract_issues)
        if len(queries) != state.selected_count:
            issues.append("suite denominator count differs from state")
        if [item.ordinal for item in queries] != list(range(len(queries))):
            issues.append("suite query ordinals are not contiguous")
        if state.status is BaselineSuiteStatus.READY and {item.method for item in links} != set(
            manifest.methods
        ):
            issues.append("ready suite does not link every configured method")
        return {
            "suite_id": suite_id,
            "status": state.status.value,
            "selected_count": len(queries),
            "methods": [item.value for item in manifest.methods],
            "run_ids": {item.method.value: item.run_id for item in links},
            "valid": not issues,
            "issues": issues,
        }
    finally:
        await engine.dispose()
