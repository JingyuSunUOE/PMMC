"""Deterministic public/private exports for one frozen baseline evaluation run."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import select, text

from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.evaluation_models import (
    AnswerNormalizationModel,
    ContextEvaluationMatchModel,
    ContextEvaluationModel,
    EvaluationJobModel,
    EvaluationRunModel,
    EvaluationRunStateModel,
    EvaluationTargetModel,
    GoldEvidenceGroupMemberModel,
    GoldEvidenceGroupModel,
)
from pmmc.infrastructure.database.experiment_models import (
    BadCaseLabelModel,
    FinalAnswerModel,
    JudgeScoreModel,
    MetricScoreModel,
    RetrievalEvaluationMatchModel,
    RetrievalEvaluationModel,
)
from pmmc.infrastructure.database.models import BenchmarkQueryModel, QueryGoldModel
from pmmc.infrastructure.object_store import sha256_file, validate_object_key
from pmmc.infrastructure.settings import Settings
from pmmc.schemas import (
    EXPERIMENT_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    SCHEMA_VERSION,
    EvaluationRunStatus,
)


@dataclass(frozen=True)
class EvaluationExportResult:
    evaluation_run_id: str
    export_id: str
    export_dir: Path


_EVALUATION_EXPORT_ANSWER_SCOPE = "evaluation-job-answer-ids-v1"


def _validate_export_answer_scope(
    *,
    source_kind: str,
    targets: Sequence[EvaluationTargetModel],
    answers: Sequence[FinalAnswerModel],
) -> dict[str, Any]:
    """Validate the method-neutral answer denominator without exposing content."""

    expected_count = sum(item.selected_count for item in targets)
    answer_ids = [item.answer_id for item in answers]
    answer_keys = [
        (item.run_id, item.query_id, item.answer_variant) for item in answers
    ]
    target_run_ids = {item.run_id for item in targets}
    if len(answers) != expected_count:
        raise ValueError(
            "evaluation export answer denominator differs from evaluation jobs "
            f"(expected={expected_count}, actual={len(answers)})"
        )
    if len(set(answer_ids)) != len(answer_ids):
        raise ValueError("evaluation export selected duplicate answer identities")
    if len(set(answer_keys)) != len(answer_keys):
        raise ValueError(
            "evaluation export selected multiple answers for one query variant"
        )
    if any(item.run_id not in target_run_ids for item in answers):
        raise ValueError("evaluation export answer escapes its target runs")
    return {
        "answer_count": len(answers),
        "answer_scope": _EVALUATION_EXPORT_ANSWER_SCOPE,
        "answer_variants": sorted({item.answer_variant for item in answers}),
        "source_kind": source_kind,
        "target_count": len(targets),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=True,
            indent=2 if pretty else None,
            separators=separators,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value, pretty=True))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_json_bytes(row))


def _model_row(value: Any) -> dict[str, Any]:
    return {
        prop.columns[0].name: _jsonable(getattr(value, prop.key))
        for prop in value.__mapper__.column_attrs
    }


def _summaries(
    metrics: list[dict[str, Any]],
    method_by_answer: dict[str, str],
    category_by_query: dict[str, str],
    variant_by_answer: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    category_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for metric in metrics:
        method = method_by_answer[str(metric["answer_id"])]
        variant = variant_by_answer[str(metric["answer_id"])]
        family = str(metric["metric_family"])
        name = str(metric["metric_name"])
        category = category_by_query.get(str(metric["query_id"]), "unknown")
        method_groups.setdefault((method, variant, family, name), []).append(metric)
        category_groups.setdefault(
            (method, variant, category, family, name), []
        ).append(metric)

    def reduce_group(key: tuple[str, ...], values: list[dict[str, Any]]) -> dict[str, Any]:
        applicable = [float(item["value"]) for item in values if item["applicable"]]
        result: dict[str, Any] = {
            "method": key[0],
            "answer_variant": key[1],
            "metric_family": key[-2],
            "metric_name": key[-1],
            "total_count": len(values),
            "applicable_count": len(applicable),
            "unavailable_count": len(values) - len(applicable),
            "mean": sum(applicable) / len(applicable) if applicable else None,
        }
        if len(key) == 5:
            result["source_category"] = key[2]
        return result

    method_rows = [
        reduce_group(key, values)
        for key, values in sorted(method_groups.items())
    ]
    category_rows = [
        reduce_group(key, values)
        for key, values in sorted(category_groups.items())
    ]
    return method_rows, category_rows


def _judge_summaries(
    judges: list[dict[str, Any]],
    method_by_answer: dict[str, str],
    category_by_query: dict[str, str],
    variant_by_answer: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    category_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for judge in judges:
        method = method_by_answer[str(judge["answer_id"])]
        variant = variant_by_answer[str(judge["answer_id"])]
        category = category_by_query.get(str(judge["query_id"]), "unknown")
        method_groups.setdefault((method, variant), []).append(judge)
        category_groups.setdefault((method, variant, category), []).append(judge)

    def reduce_group(key: tuple[str, ...], values: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(item["llm_judge_score"]) for item in values]
        row: dict[str, Any] = {
            "method": key[0],
            "answer_variant": key[1],
            "count": len(scores),
            "mean_score": sum(scores) / len(scores),
            "incorrect_count": sum(item["verdict"] == "incorrect" for item in values),
            "partially_correct_count": sum(
                item["verdict"] == "partially_correct" for item in values
            ),
            "correct_count": sum(item["verdict"] == "correct" for item in values),
        }
        if len(key) == 3:
            row["source_category"] = key[2]
        return row

    method_rows = [
        reduce_group(key, values)
        for key, values in sorted(method_groups.items())
    ]
    category_rows = [
        reduce_group(key, values)
        for key, values in sorted(category_groups.items())
    ]
    return method_rows, category_rows


def _bad_case_summaries(
    labels: list[dict[str, Any]],
    method_by_answer: dict[str, str],
    category_by_query: dict[str, str],
    variant_by_answer: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_groups: dict[tuple[str, str, str], int] = {}
    category_groups: dict[tuple[str, str, str, str], int] = {}
    for label in labels:
        if not label["is_primary"]:
            continue
        method = method_by_answer[str(label["answer_id"])]
        variant = variant_by_answer[str(label["answer_id"])]
        bad_case = str(label["category"])
        source_category = category_by_query.get(str(label["query_id"]), "unknown")
        method_groups[(method, variant, bad_case)] = (
            method_groups.get((method, variant, bad_case), 0) + 1
        )
        key = (method, variant, source_category, bad_case)
        category_groups[key] = category_groups.get(key, 0) + 1
    method_rows = [
        {
            "method": method,
            "answer_variant": variant,
            "bad_case_category": bad_case,
            "count": count,
        }
        for (method, variant, bad_case), count in sorted(method_groups.items())
    ]
    category_rows = [
        {
            "method": method,
            "answer_variant": variant,
            "source_category": source_category,
            "bad_case_category": bad_case,
            "count": count,
        }
        for (method, variant, source_category, bad_case), count in sorted(
            category_groups.items()
        )
    ]
    return method_rows, category_rows


class EvaluationExporter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def export(self, evaluation_run_id: str) -> EvaluationExportResult:
        engine, sessions = create_session_factory(self.settings, DatabaseRole.EVALUATOR)
        temporary = (
            self.settings.outputs_path
            / "evaluations"
            / ".tmp"
            / f"{evaluation_run_id}-{hashlib.sha256(evaluation_run_id.encode()).hexdigest()[:8]}"
        )
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            async with sessions() as session:
                manifest = await session.get(EvaluationRunModel, evaluation_run_id)
                state = await session.get(EvaluationRunStateModel, evaluation_run_id)
                if manifest is None or state is None:
                    raise KeyError(f"evaluation run does not exist: {evaluation_run_id}")
                if state.status != EvaluationRunStatus.COMPLETED.value:
                    raise ValueError("only completed evaluation runs can be exported")
                targets = (
                    (
                        await session.execute(
                            select(EvaluationTargetModel)
                            .where(
                                EvaluationTargetModel.evaluation_run_id
                                == evaluation_run_id
                            )
                            .order_by(EvaluationTargetModel.method)
                        )
                    )
                    .scalars()
                    .all()
                )
                method_by_run = {item.run_id: item.method for item in targets}
                snapshot_id = targets[0].snapshot_id
                query_rows = (
                    (
                        await session.execute(
                            select(BenchmarkQueryModel).where(
                                BenchmarkQueryModel.snapshot_id == snapshot_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                category_by_query = {
                    item.query_id: item.source_category or "unknown" for item in query_rows
                }
                query_by_id = {item.query_id: item for item in query_rows}
                evaluated_answer_ids = (
                    select(EvaluationJobModel.answer_id)
                    .where(
                        EvaluationJobModel.evaluation_run_id
                        == evaluation_run_id
                    )
                    .distinct()
                )
                answers = (
                    (
                        await session.execute(
                            select(FinalAnswerModel)
                            .where(
                                FinalAnswerModel.answer_id.in_(
                                    evaluated_answer_ids
                                ),
                            )
                            .order_by(FinalAnswerModel.run_id, FinalAnswerModel.query_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                metrics_models = (
                    (
                        await session.execute(
                            select(MetricScoreModel)
                            .where(
                                MetricScoreModel.evaluation_run_id == evaluation_run_id
                            )
                            .order_by(
                                MetricScoreModel.run_id,
                                MetricScoreModel.query_id,
                                MetricScoreModel.metric_family,
                                MetricScoreModel.metric_name,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                judges = (
                    (
                        await session.execute(
                            select(JudgeScoreModel)
                            .where(
                                JudgeScoreModel.evaluation_run_id == evaluation_run_id
                            )
                            .order_by(JudgeScoreModel.run_id, JudgeScoreModel.query_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                bad_cases = (
                    (
                        await session.execute(
                            select(BadCaseLabelModel)
                            .where(
                                BadCaseLabelModel.evaluation_run_id == evaluation_run_id
                            )
                            .order_by(
                                BadCaseLabelModel.run_id,
                                BadCaseLabelModel.query_id,
                                BadCaseLabelModel.is_primary.desc(),
                                BadCaseLabelModel.category,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                gold_rows = (
                    (
                        await session.execute(
                            select(QueryGoldModel).where(
                                QueryGoldModel.snapshot_id == snapshot_id,
                                QueryGoldModel.query_id.in_(
                                    sorted({item.query_id for item in answers})
                                ),
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                normalizations = (
                    (
                        await session.execute(
                            select(AnswerNormalizationModel)
                            .where(
                                AnswerNormalizationModel.evaluation_run_id
                                == evaluation_run_id
                            )
                            .order_by(
                                AnswerNormalizationModel.run_id,
                                AnswerNormalizationModel.query_id,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                groups = (
                    (
                        await session.execute(
                            select(GoldEvidenceGroupModel)
                            .where(
                                GoldEvidenceGroupModel.evaluation_run_id
                                == evaluation_run_id
                            )
                            .order_by(
                                GoldEvidenceGroupModel.query_id,
                                GoldEvidenceGroupModel.gold_group_id,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                group_members = (
                    (
                        await session.execute(
                            select(GoldEvidenceGroupMemberModel)
                            .where(
                                GoldEvidenceGroupMemberModel.gold_group_id.in_(
                                    [item.gold_group_id for item in groups]
                                )
                            )
                            .order_by(
                                GoldEvidenceGroupMemberModel.gold_group_id,
                                GoldEvidenceGroupMemberModel.member_type,
                                GoldEvidenceGroupMemberModel.member_id,
                            )
                        )
                    )
                    .scalars()
                    .all()
                ) if groups else []
                context_evaluations = (
                    (
                        await session.execute(
                            select(ContextEvaluationModel)
                            .where(
                                ContextEvaluationModel.evaluation_run_id
                                == evaluation_run_id
                            )
                            .order_by(
                                ContextEvaluationModel.run_id,
                                ContextEvaluationModel.query_id,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                context_matches = (
                    (
                        await session.execute(
                            select(ContextEvaluationMatchModel)
                            .where(
                                ContextEvaluationMatchModel.context_evaluation_id.in_(
                                    [item.context_evaluation_id for item in context_evaluations]
                                )
                            )
                            .order_by(
                                ContextEvaluationMatchModel.context_evaluation_id,
                                ContextEvaluationMatchModel.ordinal,
                            )
                        )
                    )
                    .scalars()
                    .all()
                ) if context_evaluations else []
                retrieval_evaluations = (
                    (
                        await session.execute(
                            select(RetrievalEvaluationModel)
                            .where(
                                RetrievalEvaluationModel.evaluation_run_id
                                == evaluation_run_id
                            )
                            .order_by(
                                RetrievalEvaluationModel.run_id,
                                RetrievalEvaluationModel.query_id,
                                RetrievalEvaluationModel.retrieval_trace_id,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                retrieval_matches = (
                    (
                        await session.execute(
                            select(RetrievalEvaluationMatchModel)
                            .where(
                                RetrievalEvaluationMatchModel.retrieval_evaluation_id.in_(
                                    [item.retrieval_evaluation_id for item in retrieval_evaluations]
                                )
                            )
                            .order_by(
                                RetrievalEvaluationMatchModel.retrieval_evaluation_id,
                                RetrievalEvaluationMatchModel.ordinal,
                            )
                        )
                    )
                    .scalars()
                    .all()
                ) if retrieval_evaluations else []
                production_cost = (
                    await session.execute(
                        text(
                            "SELECT * FROM pmmc_evaluation.baseline_production_cost_summary "
                            "WHERE run_id = ANY(:run_ids) ORDER BY run_id"
                        ),
                        {"run_ids": list(method_by_run)},
                    )
                ).mappings().all()
                evaluation_cost = (
                    await session.execute(
                        text(
                            "SELECT * FROM pmmc_evaluation.evaluation_cost_summary "
                            "WHERE evaluation_run_id = :evaluation_run_id"
                        ),
                        {"evaluation_run_id": evaluation_run_id},
                    )
                ).mappings().all()
                combined_cost = (
                    await session.execute(
                        text(
                            "SELECT * FROM pmmc_evaluation.combined_research_cost_summary "
                            "WHERE evaluation_run_id = :evaluation_run_id"
                        ),
                        {"evaluation_run_id": evaluation_run_id},
                    )
                ).mappings().all()
                revision = await session.scalar(text("SELECT version_num FROM alembic_version"))

            try:
                answer_scope = _validate_export_answer_scope(
                    source_kind=manifest.source_kind,
                    targets=targets,
                    answers=answers,
                )
            except ValueError:
                print(
                    "EVALUATION_EXPORT_ANSWER_SCOPE "
                    + json.dumps(
                        {
                            "actual_answer_count": len(answers),
                            "expected_answer_count": sum(
                                item.selected_count for item in targets
                            ),
                            "source_kind": manifest.source_kind,
                            "state": "failed",
                            "target_count": len(targets),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                raise
            print(
                "EVALUATION_EXPORT_ANSWER_SCOPE "
                + json.dumps(
                    {**answer_scope, "state": "ready"},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            metrics = [_model_row(item) for item in metrics_models]
            method_by_answer = {
                item.answer_id: item.method for item in answers
            }
            variant_by_answer = {
                item.answer_id: item.answer_variant for item in answers
            }
            method_summary, category_summary = _summaries(
                metrics,
                method_by_answer,
                category_by_query,
                variant_by_answer,
            )
            judge_rows = [_model_row(item) for item in judges]
            judge_method_summary, judge_category_summary = _judge_summaries(
                judge_rows,
                method_by_answer,
                category_by_query,
                variant_by_answer,
            )
            bad_case_rows = [_model_row(item) for item in bad_cases]
            bad_case_method_summary, bad_case_category_summary = _bad_case_summaries(
                bad_case_rows,
                method_by_answer,
                category_by_query,
                variant_by_answer,
            )
            answer_variant_counts = {
                variant: sum(item.answer_variant == variant for item in answers)
                for variant in sorted({item.answer_variant for item in answers})
            }
            query_count = int(
                manifest.metadata_json.get("query_count", state.selected_count)
            )
            missing_variants = manifest.metadata_json.get("answer_variant_missing_counts", {})
            if manifest.metadata_json.get("no_answer_policy") == "independent-answer-abstention-v1":
                for summary in method_summary:
                    missing = int(missing_variants.get(summary["answer_variant"], 0))
                    summary["mean_scope"] = "answered_applicable_only"
                    summary["no_answer_count"] = missing
                    denominator = summary["applicable_count"] + missing
                    summary["all_query_score_missing_as_zero"] = (
                        (summary["mean"] or 0) * summary["applicable_count"] / denominator
                        if denominator else None
                    )
                for summary in judge_method_summary:
                    summary["mean_scope"] = "generated_answers_only"
                    summary["no_answer_count"] = int(
                        missing_variants.get(summary["answer_variant"], 0)
                    )
            _write_json(
                temporary / "public" / "evaluation_summary.json",
                {
                    "evaluation_run_id": evaluation_run_id,
                    "suite_id": manifest.suite_id,
                    "protocol_version": manifest.protocol_version,
                    "selected_count": state.selected_count,
                    "query_count": query_count,
                    "answer_variant_counts": answer_variant_counts,
                    "answer_variant_missing_counts": missing_variants,
                    "native_mm_abstention_count": manifest.metadata_json.get(
                        "native_mm_abstention_count", 0
                    ),
                    "no_answer_policy": manifest.metadata_json.get("no_answer_policy"),
                    "answer_delivery_rates": {
                        variant: count / query_count if query_count else None
                        for variant, count in manifest.metadata_json.get(
                            "answer_variant_counts", answer_variant_counts
                        ).items()
                    },
                    "target_count": state.target_count,
                    "methods": sorted(set(method_by_answer.values())),
                    "judge_model_alias": manifest.judge_model_alias,
                },
            )
            _write_jsonl(temporary / "public" / "method_summary.jsonl", method_summary)
            _write_jsonl(temporary / "public" / "category_summary.jsonl", category_summary)
            _write_jsonl(
                temporary / "public" / "judge_method_summary.jsonl",
                judge_method_summary,
            )
            _write_jsonl(
                temporary / "public" / "judge_category_summary.jsonl",
                judge_category_summary,
            )
            _write_jsonl(
                temporary / "public" / "primary_bad_case_method_summary.jsonl",
                bad_case_method_summary,
            )
            _write_jsonl(
                temporary / "public" / "primary_bad_case_category_summary.jsonl",
                bad_case_category_summary,
            )
            _write_jsonl(
                temporary / "public" / "production_cost_summary.jsonl",
                [dict(_jsonable(dict(item))) for item in production_cost],
            )
            _write_jsonl(
                temporary / "public" / "evaluation_cost_summary.jsonl",
                [dict(_jsonable(dict(item))) for item in evaluation_cost],
            )
            _write_jsonl(
                temporary / "public" / "combined_research_cost_summary.jsonl",
                [dict(_jsonable(dict(item))) for item in combined_cost],
            )

            gold_by_query = {item.query_id: item for item in gold_rows}
            judge_by_answer = {item.answer_id: item for item in judges}
            private_results: list[dict[str, Any]] = []
            for answer in answers:
                query = query_by_id[answer.query_id]
                gold = gold_by_query[answer.query_id]
                judge = judge_by_answer[answer.answer_id]
                private_results.append(
                    {
                        "answer_id": answer.answer_id,
                        "run_id": answer.run_id,
                        "method": answer.method,
                        "answer_variant": answer.answer_variant,
                        "query_id": answer.query_id,
                        "question_type": query.question_type,
                        "source_category": query.source_category,
                        "scenario_id": query.scenario_id,
                        "source_file": query.source_file,
                        "source_record_index": query.source_record_index,
                        "qa_index": query.qa_index,
                        "candidate_answer": answer.final_answer,
                        "reference_answer": gold.answer,
                        "judge_verdict": judge.verdict,
                        "judge_score": judge.llm_judge_score,
                        "judge_reason": judge.judge_reason,
                    }
                )
            _write_jsonl(
                temporary / "private" / "query_results.jsonl", private_results
            )
            private_tables = {
                "answer_normalizations.jsonl": normalizations,
                "metric_scores.jsonl": metrics_models,
                "judge_scores.jsonl": judges,
                "bad_case_labels.jsonl": bad_cases,
                "gold_evidence_groups.jsonl": groups,
                "gold_evidence_group_members.jsonl": group_members,
                "context_evaluations.jsonl": context_evaluations,
                "context_evaluation_matches.jsonl": context_matches,
                "retrieval_evaluations.jsonl": retrieval_evaluations,
                "retrieval_evaluation_matches.jsonl": retrieval_matches,
            }
            for file_name, values in private_tables.items():
                _write_jsonl(
                    temporary / "private" / file_name,
                    [_model_row(item) for item in values],
                )

            files = [
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(item for item in temporary.rglob("*") if item.is_file())
            ]
            content = {
                "evaluation_run_id": evaluation_run_id,
                "suite_id": manifest.suite_id,
                "evaluation_config_hash": manifest.config_hash,
                "protocol_version": manifest.protocol_version,
                "canonical_schema_version": SCHEMA_VERSION,
                "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
                "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
                "alembic_revision": revision,
                **answer_scope,
                "file_count": len(files),
                "files": files,
            }
            for marker in ("smoke_fixture", "paper_result"):
                if marker in manifest.metadata_json:
                    content[marker] = manifest.metadata_json[marker]
            export_id = hashlib.sha256(_json_bytes(content)).hexdigest()
            _write_json(temporary / "export_manifest.json", {"export_id": export_id, **content})
            destination = (
                self.settings.outputs_path
                / "evaluations"
                / evaluation_run_id
                / export_id
            )
            if destination.exists():
                verify_evaluation_export(destination)
                shutil.rmtree(temporary)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
            return EvaluationExportResult(evaluation_run_id, export_id, destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        finally:
            await engine.dispose()


def verify_evaluation_export(export_dir: Path) -> dict[str, Any]:
    manifest_path = export_dir / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count") != len(files):
        raise ValueError("evaluation export manifest has an invalid file inventory")
    expected_paths = sorted(str(item["path"]) for item in files)
    actual_paths = sorted(
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if expected_paths != actual_paths:
        raise ValueError("evaluation export files differ from the manifest")
    root = export_dir.resolve()
    for item in files:
        relative = validate_object_key(str(item["path"]))
        path = export_dir / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"evaluation export path escapes its root: {relative}")
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"evaluation export checksum mismatch: {relative}")
    answer_scope = manifest.get("answer_scope")
    if answer_scope is not None:
        if answer_scope != _EVALUATION_EXPORT_ANSWER_SCOPE:
            raise ValueError("evaluation export uses an unsupported answer scope")
        answer_count = manifest.get("answer_count")
        answer_variants = manifest.get("answer_variants")
        if not isinstance(answer_count, int) or answer_count < 0:
            raise ValueError("evaluation export has an invalid answer count")
        if not isinstance(answer_variants, list) or not all(
            isinstance(item, str) and item for item in answer_variants
        ):
            raise ValueError("evaluation export has invalid answer variants")
        query_results_path = export_dir / "private" / "query_results.jsonl"
        with query_results_path.open(encoding="utf-8") as stream:
            query_result_count = sum(bool(line.strip()) for line in stream)
        if query_result_count != answer_count:
            raise ValueError(
                "evaluation export query-result denominator differs from its manifest"
            )
    content = {key: value for key, value in manifest.items() if key != "export_id"}
    if hashlib.sha256(_json_bytes(content)).hexdigest() != manifest["export_id"]:
        raise ValueError("evaluation export ID does not match its content")
    return manifest
