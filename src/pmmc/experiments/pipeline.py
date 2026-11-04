"""Reusable local-run reporting, progress, and artifact persistence."""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Iterable, MutableMapping
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TextIO, cast
from uuid import uuid4

import structlog
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from pmmc.benchmarks.selection import selection_hash
from pmmc.experiments.manifest import run_config_hash
from pmmc.schemas import (
    BenchmarkQuery,
    BenchmarkSelection,
    DatasetName,
    PipelineEvent,
    PipelineEventType,
    PipelineRunSummary,
    PipelineStageSummary,
    RunManifest,
    SourceLocator,
)
from pmmc.schemas.base import utc_now

Clock = Callable[[], float]
NowFactory = Callable[[], datetime]


def make_run_id(command: str) -> str:
    """Create a sortable local run ID without encoding user data."""

    command_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", command.strip()).strip(".-")
    if not command_slug:
        raise ValueError("run command must contain at least one safe character")
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{command_slug}-{timestamp}-{uuid4().hex[:8]}"


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_message(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for detail in error.errors(include_input=False, include_url=False):
            location = ".".join(str(item) for item in detail["loc"]) or "record"
            parts.append(f"{location}: {detail['msg']}")
        value = "; ".join(parts)
    else:
        value = str(error)
    return " ".join(value.split())[:1000] or "unspecified error"


def _drop_structlog_event(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    event_dict.pop("event", None)
    return event_dict


class PipelineStage:
    """Context-managed stage with deterministic counters and progress events."""

    def __init__(
        self,
        context: PipelineRunContext,
        name: str,
        *,
        dataset: DatasetName | None,
        total: int | None,
    ) -> None:
        self.context = context
        self.name = name
        self.dataset = dataset
        self.total = total
        self.processed = 0
        self.succeeded = 0
        self.failed = 0
        self._last_reported = 0
        self._started_clock = 0.0
        self._started_at = context.now()
        self._task_id: TaskID | None = None

    def __enter__(self) -> PipelineStage:
        if self.context.active_stage is not None:
            raise RuntimeError("pipeline stages cannot be nested")
        self.context.active_stage = self
        self._started_clock = self.context.clock()
        self._started_at = self.context.now()
        self._task_id = self.context.progress.add_task(self.name, total=self.total)
        self.context.console.print(
            f"START stage={self.name}"
            + (f" dataset={self.dataset.value}" if self.dataset is not None else "")
        )
        self.context.emit(
            PipelineEvent(
                run_id=self.context.run_id,
                event_type=PipelineEventType.STAGE_STARTED,
                stage=self.name,
                dataset=self.dataset,
                current=0,
                total=self.total,
            )
        )
        return self

    def advance(
        self,
        amount: int = 1,
        *,
        succeeded: bool = True,
        query_id: str | None = None,
        memory_id: str | None = None,
        source: SourceLocator | None = None,
        phase: str | None = None,
    ) -> None:
        """Advance one stage and periodically emit machine- and human-readable progress."""

        if amount <= 0:
            raise ValueError("stage progress amount must be positive")
        self.processed += amount
        if self.total is not None and self.processed > self.total:
            raise ValueError("stage progress cannot exceed total")
        if succeeded:
            self.succeeded += amount
        else:
            self.failed += amount
        if self._task_id is not None:
            self.context.progress.update(self._task_id, advance=amount)

        should_report = (
            self.processed == self.total
            or self.processed - self._last_reported >= self.context.progress_every
        )
        if should_report:
            self._last_reported = self.processed
            elapsed_ms = (self.context.clock() - self._started_clock) * 1000
            metadata = {"phase": phase} if phase is not None else {}
            self.context.emit(
                PipelineEvent(
                    run_id=self.context.run_id,
                    event_type=PipelineEventType.STAGE_PROGRESS,
                    stage=self.name,
                    dataset=self.dataset,
                    current=self.processed,
                    total=self.total,
                    elapsed_ms=elapsed_ms,
                    query_id=query_id,
                    memory_id=memory_id,
                    source=source,
                    metadata=metadata,
                )
            )
            if not self.context.is_tty:
                total = self.total if self.total is not None else "?"
                phase_text = f" phase={phase}" if phase is not None else ""
                self.context.console.print(
                    f"PROGRESS stage={self.name}{phase_text} current={self.processed} total={total}"
                )

    def fail_item(
        self,
        error: BaseException,
        *,
        query_id: str | None = None,
        memory_id: str | None = None,
        source: SourceLocator | None = None,
    ) -> None:
        """Record one failed item without hiding the failure from run artifacts."""

        self.advance(
            succeeded=False,
            query_id=query_id,
            memory_id=memory_id,
            source=source,
        )
        self.context.record_error(
            stage=self.name,
            error=error,
            dataset=self.dataset,
            query_id=query_id,
            memory_id=memory_id,
            source=source,
            event_type=PipelineEventType.ITEM_FAILED,
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        completed_at = self.context.now()
        elapsed_ms = max(0.0, (self.context.clock() - self._started_clock) * 1000)
        success = exc is None and self.failed == 0
        event_type = (
            PipelineEventType.STAGE_COMPLETED if success else PipelineEventType.STAGE_FAILED
        )
        if exc is not None:
            self.context.record_error(
                stage=self.name,
                error=exc,
                dataset=self.dataset,
                event_type=event_type,
            )
        elif self.failed:
            self.context.emit(
                PipelineEvent(
                    run_id=self.context.run_id,
                    event_type=event_type,
                    stage=self.name,
                    dataset=self.dataset,
                    current=self.processed,
                    total=self.total,
                    elapsed_ms=elapsed_ms,
                    message=f"{self.failed} item(s) failed",
                    error_type="ItemFailure",
                )
            )
        else:
            self.context.emit(
                PipelineEvent(
                    run_id=self.context.run_id,
                    event_type=event_type,
                    stage=self.name,
                    dataset=self.dataset,
                    current=self.processed,
                    total=self.total,
                    elapsed_ms=elapsed_ms,
                )
            )
        self.context.stage_summaries.append(
            PipelineStageSummary(
                stage=self.name,
                success=success,
                started_at=self._started_at,
                completed_at=completed_at,
                elapsed_ms=elapsed_ms,
                processed=self.processed,
                total=self.total,
                succeeded=self.succeeded,
                failed=self.failed,
            )
        )
        if self._task_id is not None:
            self.context.progress.remove_task(self._task_id)
        self.context.active_stage = None
        status = "DONE" if success else "FAILED"
        self.context.console.print(
            f"{status} stage={self.name} elapsed={elapsed_ms / 1000:.2f}s "
            f"processed={self.processed} failed={self.failed}"
        )
        return False


class PipelineRunContext:
    """Own one local run directory and its non-sensitive structured artifacts."""

    def __init__(
        self,
        *,
        runs_root: Path,
        manifest: RunManifest,
        datasets: list[DatasetName],
        console: Console | None = None,
        is_tty: bool | None = None,
        clock: Clock = time.perf_counter,
        now: NowFactory = utc_now,
        progress_every: int = 100,
        resume: bool = False,
    ) -> None:
        if progress_every <= 0:
            raise ValueError("progress_every must be positive")
        self.manifest = manifest
        self.run_id = manifest.run_id
        if (
            Path(self.run_id).name != self.run_id
            or "\\" in self.run_id
            or self.run_id in {".", ".."}
        ):
            raise ValueError("run_id must be a single safe path component")
        self.datasets = datasets
        self.clock = clock
        self.now = now
        self.progress_every = progress_every
        self.resume = resume
        self.console = console or Console()
        self.is_tty = sys.stdout.isatty() if is_tty is None else is_tty
        self.run_dir = runs_root / self.run_id
        if self.run_dir.exists() and not resume:
            raise FileExistsError(
                f"run directory already exists: {self.run_dir}"
            )
        if resume:
            manifest_path = self.run_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError("resumed run is missing run_manifest.json")
            existing_manifest = RunManifest.model_validate(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if existing_manifest != manifest:
                raise ValueError("resumed run manifest differs from immutable local manifest")
        else:
            self.run_dir.mkdir(parents=True)
        stream_mode = "a" if resume else "w"
        self._event_stream = cast(
            TextIO,
            (self.run_dir / "events.jsonl").open(stream_mode, encoding="utf-8"),
        )
        self._error_stream = cast(
            TextIO,
            (self.run_dir / "errors.jsonl").open(stream_mode, encoding="utf-8"),
        )
        self._event_logger = structlog.wrap_logger(
            structlog.PrintLogger(file=self._event_stream),
            processors=[_drop_structlog_event, structlog.processors.JSONRenderer(sort_keys=True)],
        )
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            disable=not self.is_tty,
        )
        previous_summary_path = self.run_dir / "stage_summary.json"
        previous_summary = (
            PipelineRunSummary.model_validate(
                json.loads(previous_summary_path.read_text(encoding="utf-8"))
            )
            if resume and previous_summary_path.is_file()
            else None
        )
        # Previous attempts remain in append-only events; the new summary describes this resume.
        self.stage_summaries: list[PipelineStageSummary] = []
        self.active_stage: PipelineStage | None = None
        self.selected_count = previous_summary.selected_count if previous_summary else 0
        self.succeeded_count = previous_summary.succeeded_count if previous_summary else 0
        self.failed_count = previous_summary.failed_count if previous_summary else 0
        self._started_at = self.now()
        self._started_clock = 0.0
        self._closed = False
        if not resume:
            self._write_manifest()

    def _write_manifest(self) -> None:
        _atomic_write_json(
            self.run_dir / "run_manifest.json",
            self.manifest.model_dump(mode="json"),
        )

    def __enter__(self) -> PipelineRunContext:
        self._started_at = self.now()
        self._started_clock = self.clock()
        self.progress.start()
        self.emit(
            PipelineEvent(
                run_id=self.run_id,
                event_type=PipelineEventType.RUN_STARTED,
                stage="run",
            )
        )
        self.console.print(f"{'RESUME' if self.resume else 'START'} run={self.run_id}")
        return self

    def stage(
        self,
        name: str,
        *,
        dataset: DatasetName | None = None,
        total: int | None = None,
    ) -> PipelineStage:
        return PipelineStage(self, name, dataset=dataset, total=total)

    def emit(self, event: PipelineEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError("pipeline event run_id does not match context")
        self._event_logger.info("pipeline_event", **event.model_dump(mode="json"))
        self._event_stream.flush()

    def record_error(
        self,
        *,
        stage: str,
        error: BaseException,
        event_type: PipelineEventType,
        dataset: DatasetName | None = None,
        query_id: str | None = None,
        memory_id: str | None = None,
        source: SourceLocator | None = None,
    ) -> None:
        message = _safe_message(error)
        event = PipelineEvent(
            run_id=self.run_id,
            event_type=event_type,
            stage=stage,
            dataset=dataset,
            query_id=query_id,
            memory_id=memory_id,
            source=source,
            message=message,
            error_type=type(error).__name__,
        )
        self.emit(event)
        self._error_stream.write(event.to_json() + "\n")
        self._error_stream.flush()

    def record_selection(
        self,
        selection: BenchmarkSelection,
        selected: Iterable[BenchmarkQuery],
    ) -> list[BenchmarkQuery]:
        """Persist selector, resolved IDs, and source locators without gold fields."""

        records = list(selected)
        fingerprint = selection_hash(selection, records)
        if self.manifest.selection_hash not in {None, fingerprint}:
            raise ValueError("manifest selection_hash does not match resolved selection")
        if self.manifest.seed not in {None, selection.seed}:
            raise ValueError("manifest seed does not match benchmark selection")
        updated = self.manifest.model_copy(
            update={
                "selection_hash": fingerprint,
                "seed": selection.seed,
                "config_hash": None,
            }
        )
        self.manifest = updated.model_copy(update={"config_hash": run_config_hash(updated)})
        self._write_manifest()
        _atomic_write_json(
            self.run_dir / "selection.json",
            {
                "selection_hash": fingerprint,
                "selection": selection.model_dump(mode="json"),
                "query_ids": [record.query.query_id for record in records],
                "source_locators": [record.source.model_dump(mode="json") for record in records],
            },
        )
        selected_path = self.run_dir / "selected_queries.jsonl"
        with selected_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(record.to_json() + "\n")
        self.selected_count = len(records)
        return records

    def add_snapshot_id(self, snapshot_id: str) -> None:
        if snapshot_id not in self.manifest.dataset_snapshot_ids:
            updated = self.manifest.model_copy(
                update={
                    "dataset_snapshot_ids": [
                        *self.manifest.dataset_snapshot_ids,
                        snapshot_id,
                    ],
                    "config_hash": None,
                }
            )
            config_hash = run_config_hash(updated) if updated.selection_hash else None
            self.manifest = updated.model_copy(update={"config_hash": config_hash})
            self._write_manifest()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        if self.active_stage is not None:
            raise RuntimeError("active stage was not closed before pipeline exit")
        success = exc is None and all(stage.success for stage in self.stage_summaries)
        if exc is not None:
            self.record_error(
                stage="run",
                error=exc,
                event_type=PipelineEventType.RUN_FAILED,
            )
        elif not success:
            self.record_error(
                stage="run",
                error=RuntimeError("one or more pipeline stages failed"),
                event_type=PipelineEventType.RUN_FAILED,
            )
        else:
            self.emit(
                PipelineEvent(
                    run_id=self.run_id,
                    event_type=PipelineEventType.RUN_COMPLETED,
                    stage="run",
                    elapsed_ms=max(0.0, (self.clock() - self._started_clock) * 1000),
                )
            )
        completed_at = self.now()
        summary = PipelineRunSummary(
            run_id=self.run_id,
            datasets=self.datasets,
            success=success,
            started_at=self._started_at,
            completed_at=completed_at,
            elapsed_ms=max(0.0, (self.clock() - self._started_clock) * 1000),
            stages=self.stage_summaries,
            selected_count=self.selected_count,
            succeeded_count=self.succeeded_count,
            failed_count=self.failed_count,
        )
        _atomic_write_json(
            self.run_dir / "stage_summary.json",
            summary.model_dump(mode="json"),
        )
        self.progress.stop()
        self._event_stream.close()
        self._error_stream.close()
        self._closed = True
        status = "DONE" if success else "FAILED"
        self.console.print(f"{status} run={self.run_id} elapsed={summary.elapsed_ms / 1000:.2f}s")
        return False
