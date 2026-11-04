"""Model-free DTOs for offline aggregation of evaluator-owned process metrics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MATCH_KS = (1, 5, 10, 50)
TECHNICAL = (
    "technical_success",
    "nonempty_evidence",
    "structural_success",
    "relevance_success",
    "raw_image_reached",
    "fallback_triggered",
)
RETRIEVED = (
    "retrieved_evidence_recall",
    "retrieved_image_hit",
    "retrieved_group_complete",
)
QBANK = tuple(f"qbank_proxy_{kind}_at_{k}" for k in MATCH_KS for kind in ("cover", "match")) + (
    "qbank_full_bank_proxy_cover",
    "qbank_full_bank_proxy_match",
)
PROCESS_METRIC_NAMES = TECHNICAL + RETRIEVED + QBANK


class ProcessMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["evaluated", "incomplete", "not_evaluated", "not_applicable"]
    value: float | None = Field(default=None, ge=0, le=1)
    numerator: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)
    denominator_kind: Literal["query", "gold_groups"] = "query"
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_observation(self) -> ProcessMetric:
        if self.status == "evaluated":
            if (
                self.value is None
                or self.numerator is None
                or not self.denominator
                or self.numerator > self.denominator
                or self.value != self.numerator / self.denominator
                or self.reason is not None
            ):
                raise ValueError("invalid evaluated process metric")
        elif self.value is not None or self.numerator is not None or not self.reason:
            raise ValueError("unavailable process metrics require a reason and no value")
        return self


class ProcessCellBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cell_id: str
    method: str
    source_run_id: str
    source_fact_id: str | None
    source_fact_hash: str | None
    question_budget: int | None = None
    round_id: int | None = None


class ProcessCellMetrics(ProcessCellBinding):
    query_id: str
    selected_query_denominator: Literal[1] = 1
    metrics: dict[str, ProcessMetric]
    provenance: dict[str, Any] = Field(default_factory=dict)


class ProcessMetricsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["step8-process-metrics-v1"] = "step8-process-metrics-v1"
    status: Literal["complete", "incomplete"]
    cells: list[ProcessCellMetrics]
    unavailable_metrics: list[str] = Field(
        default_factory=lambda: ["semantic_cover", "semantic_match", "oracle_answer_accuracy"]
    )
