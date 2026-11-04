"""Ephemeral Step 5D fixture bootstrap and managed-judge acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image
from sqlalchemy import select

from pmmc.baselines.corpus import build_corpus
from pmmc.baselines.suite import (
    create_baseline_suite,
    run_baseline_suite,
    verify_baseline_suite,
)
from pmmc.benchmarks import build_snapshot, create_adapter
from pmmc.infrastructure.database.baseline_repositories import BaselineCorpusImporter
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_engine, create_session_factory
from pmmc.infrastructure.database.evaluation_repositories import (
    EvaluationUnitOfWork,
    PostgresBaselineSuiteRepository,
)
from pmmc.infrastructure.database.experiment_models import (
    EvaluationArtifactModel,
    EvaluationModelCallModel,
    FinalAnswerModel,
    JudgeScoreModel,
)
from pmmc.infrastructure.database.importer import CanonicalSnapshotImporter
from pmmc.infrastructure.database.models import RunManifestModel
from pmmc.infrastructure.database.runtime_models import (
    ManagedRuntimeServiceModel,
    ManagedRuntimeSessionModel,
)
from pmmc.infrastructure.database.storage_updates import persist_asset_sync
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.infrastructure.storage_sync import sync_assets
from pmmc.models.locking import load_model_lock
from pmmc.models.tokenization import HeuristicTokenCounter
from pmmc.schemas import (
    FOUNDATIONAL_BASELINE_METHODS,
    AppStage,
    BaselineSuiteStatus,
    BenchmarkSelection,
    DatasetName,
    EvaluationRunStatus,
    EvaluationStage,
    JobStatus,
    JudgeVerdict,
    ValidationLevel,
)

STEP5D_SMOKE_GOLD_SENTINEL = "PMMC_STEP5D_GPU_GOLD_SENTINEL_7A4E2C19"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_SMOKE_MARKERS: dict[str, object] = {
    "smoke_fixture": True,
    "paper_result": False,
}


def _secret_value(value: object | None) -> str:
    if value is None:
        return ""
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if getter else value)


def _require_loopback(value: str, label: str) -> None:
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(f"{label} must target loopback for an ephemeral smoke fixture")


def assert_ephemeral_endpoints(settings: Settings, *, confirmed: bool) -> None:
    """Refuse fixture writes unless every stateful dependency is Pod-local."""

    if not confirmed:
        raise ValueError("ephemeral smoke bootstrap requires --confirm-ephemeral")
    endpoints = {
        "admin database": _secret_value(settings.database_admin_url),
        "runtime database": _secret_value(settings.database_url),
        "evaluator database": _secret_value(settings.database_evaluator_url),
        "object store": settings.object_store_endpoint or "",
    }
    missing = [label for label, value in endpoints.items() if not value.strip()]
    if missing:
        raise ValueError("ephemeral smoke endpoints are missing: " + ", ".join(missing))
    for label, value in endpoints.items():
        _require_loopback(value, label)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color=color).save(path, format="PNG")


def build_step5d_source_fixture(settings: Settings) -> Path:
    """Create a deterministic raw MemGallery package under the project root."""

    artifacts = settings.artifacts_path
    if not artifacts.is_relative_to(settings.project_root):
        artifacts = settings.project_root / ".smoke" / "artifacts"
    root = artifacts / "smoke" / "step5d" / "source" / "memgallery"
    dialogs = root / "data" / "dialogs"
    dialogs.mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "source").mkdir(parents=True, exist_ok=True)
    _write_image(root / "images" / "scenario_smoke" / "history.png", (0, 255, 0))
    _write_image(root / "images" / "scenario_smoke" / "question.png", (0, 0, 255))
    payload = {
        "character_profile": {
            "name": "Avery",
            "persona_summary": "A careful smoke-test researcher.",
            "traits": ["careful", "concise"],
            "conversation_style": "Concise.",
        },
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2025-01-01",
                "dialogues": [
                    {
                        "round": "D1:1",
                        "user": "I photographed the green sample.",
                        "assistant": "I will remember it.",
                        "image_id": ["D1_IMG_001"],
                        "input_image": ["../image/scenario_smoke/history.png"],
                        "image_caption": ["a green sample"],
                    }
                ],
            }
        ],
        "human-annotated QAs": [
            {
                "point": "FR",
                "question": "What color was the sample?",
                "question_image": "../image/scenario_smoke/question.png",
                "image_caption": "a blue query card",
                "answer": STEP5D_SMOKE_GOLD_SENTINEL,
                "session_id": ["D1"],
                "clue": ["D1:1"],
            }
        ],
    }
    (dialogs / "scenario_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return root


def _fixture_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "mode": AppMode.DEV,
            "model_registry_path": Path("configs/models/smoke.yaml"),
            "model_dev_max_attempts": 1,
            "model_max_concurrency": 1,
            "langfuse_enabled": False,
        }
    )


async def bootstrap_step5d_fixture(
    settings: Settings,
    *,
    confirm_ephemeral: bool,
) -> dict[str, Any]:
    """Build and execute one five-method fake suite through production adapters."""

    assert_ephemeral_endpoints(settings, confirmed=confirm_ephemeral)
    smoke = _fixture_settings(settings)
    source_root = build_step5d_source_fixture(smoke)
    adapter = create_adapter(
        DatasetName.MEMGALLERY,
        root=source_root,
        project_root=smoke.project_root,
    )
    snapshot_path, snapshot = build_snapshot(
        adapter,
        output_root=smoke.benchmark_artifacts_path,
        # The real judge receives the fixture query image in paper mode. FULL
        # validation materializes its checksum and dimensions, which are part
        # of the provider-image reproducibility contract.
        level=ValidationLevel.FULL,
        enforce_expected_counts=False,
    )
    corpus_path, corpus = build_corpus(
        snapshot_path,
        output_root=smoke.baseline_corpora_path,
        counter=HeuristicTokenCounter("pmmc/smoke-corpus-tokenizer", "v1"),
    )

    admin_engine = create_engine(smoke, DatabaseRole.ADMIN)
    try:
        await CanonicalSnapshotImporter(
            admin_engine,
            batch_size=smoke.database_import_batch_size,
        ).import_snapshot(snapshot_path)
        await BaselineCorpusImporter(
            admin_engine,
            batch_size=smoke.database_import_batch_size,
        ).import_corpus(corpus_path)
        runtime_store = create_runtime_object_store(smoke)
        synchronized = await sync_assets(
            snapshot_path,
            project_root=smoke.project_root,
            runtime_store=runtime_store,
        )
        await persist_asset_sync(admin_engine, synchronized)
    finally:
        await admin_engine.dispose()

    selection = BenchmarkSelection(
        datasets=[DatasetName.MEMGALLERY],
        limit=1,
        seed=0,
        metadata=dict(_SMOKE_MARKERS),
    )
    suite, suite_directory = await create_baseline_suite(
        settings=smoke,
        snapshot_id=snapshot.snapshot_id,
        selection=selection,
        corpus_id=corpus.corpus_id,
        methods=list(FOUNDATIONAL_BASELINE_METHODS),
        answer_model_alias="answer-qwen3.5-9b",
        metadata=dict(_SMOKE_MARKERS),
    )
    await run_baseline_suite(settings=smoke, suite_id=suite.suite_id)
    verification = await verify_baseline_suite(smoke, suite.suite_id)
    if (
        not verification["valid"]
        or verification["status"] != BaselineSuiteStatus.READY.value
    ):
        raise RuntimeError("ephemeral five-method fixture suite failed verification")
    return {
        "mode": "fixture",
        "suite_id": suite.suite_id,
        "snapshot_id": snapshot.snapshot_id,
        "corpus_id": corpus.corpus_id,
        "selected_count": verification["selected_count"],
        "methods": verification["methods"],
        "smoke_fixture": True,
        "paper_result": False,
        "suite_artifacts": str(suite_directory),
    }


def _model_commit(settings: Settings, alias: str) -> str | None:
    lock_path = settings.resolved_runtime_model_lock_path or settings.resolved_model_lock_path
    if not lock_path.is_file():
        return None
    return load_model_lock(lock_path).get(alias).commit_sha


async def verify_step5d_fixture(
    settings: Settings,
    evaluation_run_id: str,
    *,
    require_managed_runtime: bool,
) -> dict[str, Any]:
    """Apply strict one-query/five-judge acceptance checks without returning gold."""

    from pmmc.evaluation.workflow import verify_evaluation_run

    engine, sessions = create_session_factory(settings, DatabaseRole.EVALUATOR)
    try:
        async with EvaluationUnitOfWork(sessions) as uow:
            manifest = await uow.runs.get_manifest(evaluation_run_id)
            state = await uow.runs.get_state(evaluation_run_id)
            jobs = await uow.runs.get_jobs(evaluation_run_id)
            targets = await uow.runs.get_targets(evaluation_run_id)
        suite_id = manifest.suite_id
        if suite_id is None:
            raise RuntimeError("Step 5D fixture evaluation is not bound to a baseline suite")
        async with sessions() as session:
            suites = PostgresBaselineSuiteRepository(session)
            suite = await suites.get_manifest(suite_id)
            suite_state = await suites.get_state(suite_id)
            suite_queries = await suites.get_queries(suite_id)
            links = await suites.get_run_links(suite_id)
            run_ids = [item.run_id for item in links]
            answers = list(
                (
                    await session.scalars(
                        select(FinalAnswerModel).where(
                            FinalAnswerModel.run_id.in_(run_ids),
                            FinalAnswerModel.answer_variant == "primary",
                        )
                    )
                ).all()
            )
            child_manifests = list(
                (
                    await session.scalars(
                        select(RunManifestModel).where(
                            RunManifestModel.run_id.in_(run_ids)
                        )
                    )
                ).all()
            )
            calls = list(
                (
                    await session.scalars(
                        select(EvaluationModelCallModel).where(
                            EvaluationModelCallModel.evaluation_run_id
                            == evaluation_run_id
                        )
                    )
                ).all()
            )
            scores = list(
                (
                    await session.scalars(
                        select(JudgeScoreModel).where(
                            JudgeScoreModel.evaluation_run_id == evaluation_run_id
                        )
                    )
                ).all()
            )
            artifacts = list(
                (
                    await session.scalars(
                        select(EvaluationArtifactModel).where(
                            EvaluationArtifactModel.evaluation_run_id
                            == evaluation_run_id
                        )
                    )
                ).all()
            )
            runtime_ids = sorted(
                {item.runtime_id for item in calls if item.runtime_id is not None}
            )
            runtime_sessions = list(
                (
                    await session.scalars(
                        select(ManagedRuntimeSessionModel).where(
                            ManagedRuntimeSessionModel.runtime_id.in_(runtime_ids)
                        )
                    )
                ).all()
            ) if runtime_ids else []
            runtime_services = list(
                (
                    await session.scalars(
                        select(ManagedRuntimeServiceModel).where(
                            ManagedRuntimeServiceModel.runtime_id.in_(runtime_ids)
                        )
                    )
                ).all()
            ) if runtime_ids else []
    finally:
        await engine.dispose()

    base = await verify_evaluation_run(
        settings,
        evaluation_run_id,
        require_managed_runtime=require_managed_runtime,
    )
    suite_report = await verify_baseline_suite(settings, suite_id)
    issues = list(base["issues"])
    if suite.metadata.get("smoke_fixture") is not True:
        issues.append("suite is not marked as a smoke fixture")
    if suite.metadata.get("paper_result") is not False:
        issues.append("smoke suite is not marked paper_result=false")
    if manifest.metadata.get("smoke_fixture") is not True:
        issues.append("evaluation run did not inherit the smoke marker")
    if manifest.metadata.get("paper_result") is not False:
        issues.append("evaluation run is not marked paper_result=false")
    if suite_state.status is not BaselineSuiteStatus.READY or not suite_report["valid"]:
        issues.append("fixture suite is not ready and valid")
    if state.status is not EvaluationRunStatus.COMPLETED:
        issues.append("fixture evaluation is not completed")
    if len(suite_queries) != 1 or len(targets) != 5 or len(answers) != 5:
        issues.append("fixture denominator must be one query across five methods")
    if len(child_manifests) != 5 or any(
        item.manifest_json.get("metadata", {}).get("smoke_fixture") is not True
        or item.manifest_json.get("metadata", {}).get("paper_result") is not False
        for item in child_manifests
    ):
        issues.append("fixture child runs are missing non-paper smoke markers")
    if set(suite.methods) != set(FOUNDATIONAL_BASELINE_METHODS):
        issues.append("fixture suite does not contain all five foundational methods")

    stage_counts = {
        stage.value: sum(item.stage is stage for item in jobs) for stage in EvaluationStage
    }
    if len(jobs) != 15 or any(value != 5 for value in stage_counts.values()):
        issues.append("fixture evaluation must contain five jobs in each of three stages")
    if any(item.status is not JobStatus.SUCCEEDED for item in jobs):
        issues.append("fixture evaluation contains a non-succeeded job")

    judge_config = manifest.metadata.get("judge_model_config")
    expected_actual_model = (
        str(judge_config.get("expected_actual_model_id"))
        if isinstance(judge_config, dict)
        else ""
    )
    if not expected_actual_model or expected_actual_model == "deterministic-fake@v1":
        issues.append("fixture judge is not bound to the real paper model identity")
    if len(calls) != 5:
        issues.append("fixture evaluation must contain exactly five judge model calls")
    if require_managed_runtime and (
        not runtime_services
        or not any(item.process_id is not None for item in runtime_services)
    ):
        issues.append("managed judge service process lineage is missing")
    for call in calls:
        if (
            call.status != "success"
            or call.stage != AppStage.JUDGE.value
            or call.model_alias != manifest.judge_model_alias
            or call.actual_model_id != expected_actual_model
        ):
            issues.append("judge call identity or status differs from the frozen manifest")
            break
        if require_managed_runtime and (
            call.runtime_id is None or call.runtime_service_id is None
        ):
            issues.append("judge call is missing managed-runtime lineage")
            break

    valid_verdicts = {item.value for item in JudgeVerdict}
    if len(scores) != 5:
        issues.append("fixture evaluation must contain exactly five judge scores")
    for score in scores:
        if (
            score.verdict not in valid_verdicts
            or score.llm_judge_score not in {0.0, 0.5, 1.0}
            or not (score.judge_reason or "").strip()
        ):
            issues.append("judge score failed structured verdict validation")
            break
    if not artifacts or any(item.state != "ready" for item in artifacts):
        issues.append("fixture evaluation artifacts are missing or not ready")

    return {
        "mode": "fixture",
        "suite_id": manifest.suite_id,
        "evaluation_run_id": evaluation_run_id,
        "smoke_fixture": True,
        "paper_result": False,
        "selected_count": len(suite_queries),
        "method_count": len(targets),
        "final_answer_count": len(answers),
        "evaluation_job_count": len(jobs),
        "evaluation_stage_counts": stage_counts,
        "judge_call_count": len(calls),
        "managed_judge_call_count": base["managed_judge_call_count"],
        "judge_score_count": len(scores),
        "evaluation_artifact_count": len(artifacts),
        "runtime_artifact_namespace_leak_count": base[
            "runtime_artifact_namespace_leak_count"
        ],
        "model_alias": manifest.judge_model_alias,
        "actual_model_id": expected_actual_model,
        "model_commit": _model_commit(settings, manifest.judge_model_alias),
        "runtime_ids": runtime_ids,
        "runtime_service_ids": sorted(
            {
                item.runtime_service_id
                for item in calls
                if item.runtime_service_id is not None
            }
        ),
        "runtime_process_ids": sorted(
            {item.process_id for item in runtime_services if item.process_id is not None}
        ),
        "gpu_fingerprints": sorted(
            {item.hardware_fingerprint for item in runtime_sessions}
        ),
        "calibration_required": settings.runtime_require_calibration_paper,
        "valid": not issues,
        "issues": issues,
    }


def _assert_safe_result(value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if STEP5D_SMOKE_GOLD_SENTINEL in encoded:
        raise ValueError("smoke result must not contain gold data")

    def inspect(item: object) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                normalized = str(key).lower().replace("-", "_")
                if any(
                    marker in normalized
                    for marker in (
                        "password",
                        "secret",
                        "token",
                        "prompt",
                        "query",
                        "gold",
                        "presigned",
                    )
                ):
                    raise ValueError("smoke result contains a forbidden sensitive field")
                inspect(nested)
        elif isinstance(item, list):
            for nested in item:
                inspect(nested)

    inspect(value)


def write_smoke_result(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist a deliberately narrow, gold-free smoke report."""

    _assert_safe_result(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
