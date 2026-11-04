"""Hierarchical, secret-safe progress reporting for one research launch."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pmmc.schemas.base import utc_now

Clock = Callable[[], float]
_PROGRESS_PHASE_FILE_ENV = "PMMC_PROGRESS_PHASE_FILE"
_SAFE_METADATA = {
    "alias",
    "dataset",
    "denominator_kind",
    "method",
    "phase",
    "progress_unit",
    "query_id",
    "source_category",
    "state",
}


class HierarchicalProgressReporter:
    """Display one overall task and one active major-stage task."""

    def __init__(
        self,
        *,
        launch_id: str,
        events_path: Path,
        overall_total: int,
        stage_weights: dict[str, float] | None = None,
        console: Console | None = None,
        is_tty: bool | None = None,
        report_interval_seconds: float = 30.0,
        clock: Clock = time.monotonic,
    ) -> None:
        if overall_total <= 0:
            raise ValueError("overall progress total must be positive")
        if report_interval_seconds <= 0:
            raise ValueError("progress report interval must be positive")
        self.launch_id = launch_id
        self.events_path = events_path
        self.console = console or Console()
        self.is_tty = sys.stdout.isatty() if is_tty is None else is_tty
        self.report_interval_seconds = report_interval_seconds
        self.clock = clock
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            disable=not self.is_tty,
            expand=False,
        )
        self.stage_weights = dict(stage_weights or {})
        self.weighted = bool(self.stage_weights)
        self.overall_total = (
            sum(self.stage_weights.values()) if self.weighted else float(overall_total)
        )
        if self.overall_total <= 0:
            raise ValueError("overall progress weight must be positive")
        self.overall_completed = 0.0
        self.overall_task: TaskID | None = None
        self.stage_task: TaskID | None = None
        self.subtask_task: TaskID | None = None
        self.subtask_name: str | None = None
        self.subtask_total = 0
        self.subtask_completed = 0
        self.stage_name: str | None = None
        self.stage_total = 0
        self.stage_completed = 0
        self.stage_weight = 0.0
        self._stage_progress_samples = 0
        self._stage_eta_group: tuple[object, object] | None = None
        self._stage_eta_mixed = False
        self._last_printed = 0.0
        self._started_at = 0.0
        self._stage_started_at = 0.0
        self._stream: TextIO | None = None
        self._phase_sink_error_reported = False

    def _publish_phase(self, phase: str) -> None:
        raw_path = os.environ.get(_PROGRESS_PHASE_FILE_ENV)
        if not raw_path:
            return
        path = Path(raw_path)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(f"{phase}\n", encoding="ascii")
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            if not self._phase_sink_error_reported:
                self.console.print(
                    "PROGRESS_PHASE_SINK state=unavailable "
                    f"error_type={type(exc).__name__}"
                )
                self._phase_sink_error_reported = True

    def __enter__(self) -> HierarchicalProgressReporter:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.events_path.open("a", encoding="utf-8")
        self._started_at = self.clock()
        self.progress.start()
        self._publish_phase("launch")
        self.overall_task = self.progress.add_task(
            "overall", total=self.overall_total
        )
        self._emit("launch_started", stage="launch")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        if self.subtask_task is not None:
            self.progress.remove_task(self.subtask_task)
        if self.stage_task is not None:
            self.progress.remove_task(self.stage_task)
        self._emit(
            "launch_completed" if exc is None else "launch_failed",
            stage="launch",
            error_type=type(exc).__name__ if exc is not None else None,
        )
        self._publish_phase("launcher_completed" if exc is None else "launcher_failed")
        self.progress.stop()
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def start_stage(
        self,
        name: str,
        *,
        total: int,
        description: str | None = None,
        stage_weight: float | None = None,
    ) -> None:
        if self.stage_task is not None:
            raise RuntimeError("hierarchical progress stages cannot overlap")
        if total <= 0:
            raise ValueError("stage progress total must be positive")
        self.stage_name = name
        self._stage_started_at = self.clock()
        self.stage_total = total
        self.stage_completed = 0
        self.stage_weight = float(
            stage_weight
            if stage_weight is not None
            else self.stage_weights.get(name, 1.0)
        )
        if self.stage_weight <= 0:
            raise ValueError("stage progress weight must be positive")
        self._stage_progress_samples = 0
        self._stage_eta_group = None
        self._stage_eta_mixed = False
        self._publish_phase(name)
        self.stage_task = self.progress.add_task(description or name, total=total)
        self._emit("stage_started", stage=name, current=0, total=total)
        self.console.print(f"START stage={name} total={total}")

    def advance(self, amount: int = 1, **metadata: object) -> None:
        if self.stage_task is None or self.stage_name is None:
            raise RuntimeError("no active launcher stage")
        if amount <= 0:
            raise ValueError("progress amount must be positive")
        remaining = self.stage_total - self.stage_completed
        applied = min(amount, remaining)
        if applied <= 0:
            return
        previous_fraction = self.stage_completed / self.stage_total
        self.stage_completed += applied
        current_fraction = self.stage_completed / self.stage_total
        overall_advance = (
            self.stage_weight * (current_fraction - previous_fraction)
            if self.weighted
            else float(applied)
        )
        self.overall_completed = min(
            self.overall_total, self.overall_completed + overall_advance
        )
        self.progress.update(self.stage_task, advance=applied)
        assert self.overall_task is not None
        self.progress.update(self.overall_task, advance=overall_advance)
        eta_group = (metadata.get("dataset"), metadata.get("phase"))
        if self._stage_progress_samples == 0:
            self._stage_eta_group = eta_group
        elif eta_group != self._stage_eta_group:
            self._stage_eta_mixed = True
        if metadata.get("state") in ("reused", "terminal_fact_reuse"):
            self._stage_eta_mixed = True
        self._stage_progress_samples += 1
        now = self.clock()
        should_print = (
            self.stage_completed == self.stage_total
            or now - self._last_printed >= self.report_interval_seconds
        )
        if should_print:
            self._last_printed = now
            safe = {key: value for key, value in metadata.items() if key in _SAFE_METADATA}
            self._emit(
                "stage_progress",
                stage=self.stage_name,
                current=self.stage_completed,
                total=self.stage_total,
                metadata=safe,
            )
            if not self.is_tty:
                overall = 100 * self.overall_completed / self.overall_total
                elapsed = max(0.0, now - self._started_at)
                stage_elapsed = max(0.0, now - self._stage_started_at)
                stage_rate = (
                    self.stage_completed / stage_elapsed
                    if stage_elapsed >= 5 and self._stage_progress_samples >= 2
                    and not self._stage_eta_mixed
                    and metadata.get("progress_unit") != "milestone"
                    else 0.0
                )
                eta = (
                    (self.stage_total - self.stage_completed) / stage_rate
                    if stage_rate > 0
                    else None
                )
                suffix = " ".join(f"{key}={value}" for key, value in sorted(safe.items()))
                self.console.print(
                    f"PROGRESS overall={overall:.1f}% stage={self.stage_name} "
                    f"stage_progress={self.stage_completed}/{self.stage_total} "
                    f"elapsed={_duration(elapsed)} stage_elapsed={_duration(stage_elapsed)} "
                    f"eta={_duration(eta)} eta_scope=stage "
                    f"overall_basis={'stage_weights' if self.weighted else 'work_units'} "
                    "overall_scope=attempt_traversal "
                    f"{suffix}".rstrip(),
                    soft_wrap=True,
                )

    def report_operation(self, phase: str) -> None:
        """Label work outside a weighted stage without inventing completed units."""
        self._publish_phase(phase)
        self._emit("operation_started", stage=phase, current=0, total=1)

    def set_stage_total(self, total: int) -> None:
        """Replace an estimated stage denominator with a verified actual count."""

        if self.stage_task is None or self.stage_name is None:
            raise RuntimeError("no active launcher stage")
        if total <= 0 or total < self.stage_completed:
            raise ValueError("actual stage total cannot be below completed work")
        previous_total = self.stage_total
        self.stage_total = total
        self.progress.update(self.stage_task, total=total)
        self._emit(
            "stage_total_updated",
            stage=self.stage_name,
            current=self.stage_completed,
            total=total,
        )
        if not self.is_tty and previous_total != total:
            self.console.print(
                f"STAGE_TOTAL stage={self.stage_name} previous={previous_total} "
                f"actual={total} completed={self.stage_completed}",
                soft_wrap=True,
            )

    def finish_stage(self, *, reused: bool = False) -> None:
        if self.stage_task is None or self.stage_name is None:
            raise RuntimeError("no active launcher stage")
        self.finish_subtask()
        if self.stage_completed < self.stage_total:
            self.advance(
                self.stage_total - self.stage_completed,
                state="reused" if reused else "done",
            )
        name = self.stage_name
        self._emit(
            "stage_completed",
            stage=name,
            current=self.stage_completed,
            total=self.stage_total,
            metadata={"reused": reused},
        )
        self.progress.remove_task(self.stage_task)
        self.stage_task = None
        self.stage_name = None
        self.stage_weight = 0.0
        self._publish_phase("launcher_idle")
        self.console.print(f"DONE stage={name} reused={str(reused).lower()}")

    def update_subtask(self, name: str, *, current: int, total: int) -> None:
        """Show the current model, dataset phase, method, or job group."""

        if self.stage_task is None:
            raise RuntimeError("subtasks require an active launcher stage")
        if total <= 0 or current < 0:
            raise ValueError("subtask progress requires non-negative current and positive total")
        bounded = min(current, total)
        if self.subtask_task is None or self.subtask_name != name:
            self.finish_subtask()
            self.subtask_name = name
            self.subtask_total = total
            self.subtask_completed = bounded
            self.subtask_task = self.progress.add_task(
                f"  {name}", total=total, completed=bounded
            )
            return
        self.subtask_total = total
        self.subtask_completed = bounded
        self.progress.update(self.subtask_task, total=total, completed=bounded)

    def finish_subtask(self) -> None:
        if self.subtask_task is None:
            return
        self.progress.remove_task(self.subtask_task)
        self.subtask_task = None
        self.subtask_name = None
        self.subtask_total = 0
        self.subtask_completed = 0

    def fail_stage(self, error: BaseException) -> None:
        if self.stage_name is None:
            return
        self._emit(
            "stage_failed",
            stage=self.stage_name,
            current=self.stage_completed,
            total=self.stage_total,
            error_type=type(error).__name__,
        )

    def model_progress(self, alias: str, state: str, current: int, total: int) -> None:
        """Adapter for model cache preparation without exposing Hub credentials."""

        self.update_subtask(
            f"model:{alias}",
            current=1 if state in {"cached", "verified"} else 0,
            total=1,
        )
        if state in {"cached", "verified"} and current > self.stage_completed:
            self.advance(current - self.stage_completed, alias=alias, state=state)
        elif not self.is_tty:
            self.console.print(
                f"MODEL alias={alias} state={state} progress={current}/{total}"
            )

    def _emit(
        self,
        event: str,
        *,
        stage: str,
        current: int | None = None,
        total: int | None = None,
        metadata: dict[str, object] | None = None,
        error_type: str | None = None,
    ) -> None:
        if self._stream is None:
            return
        payload = {
            "created_at": utc_now().isoformat(),
            "launch_id": self.launch_id,
            "event": event,
            "stage": stage,
            "current": current,
            "total": total,
            "overall_current": round(self.overall_completed, 8),
            "overall_total": self.overall_total,
            "error_type": error_type,
            "metadata": metadata or {},
        }
        self._stream.write(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        )
        self._stream.flush()


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"
