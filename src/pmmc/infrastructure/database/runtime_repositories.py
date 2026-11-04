"""Async repositories for managed local runtime provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import Table, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from pmmc.infrastructure.database.errors import (
    InvalidStateTransitionError,
    PersistenceConflictError,
)
from pmmc.infrastructure.database.experiment_models import RuntimeModelCallModel
from pmmc.infrastructure.database.runtime_mappers import (
    managed_service_from_row,
    managed_service_to_row,
    runtime_calibration_from_row,
    runtime_calibration_to_row,
    runtime_session_from_row,
    runtime_session_to_row,
)
from pmmc.infrastructure.database.runtime_models import (
    ManagedRuntimeServiceModel,
    ManagedRuntimeSessionModel,
    RunRuntimeLinkModel,
    RuntimeCalibrationModel,
    RuntimeGpuAllocationModel,
)
from pmmc.schemas import (
    GpuAllocation,
    GpuDeviceInfo,
    ManagedRuntimeSession,
    ManagedServiceState,
    ManagedServiceStatus,
    RuntimeCalibration,
    RuntimeStatus,
)
from pmmc.schemas.base import utc_now


def _model_row(value: Any) -> dict[str, Any]:
    return {
        prop.columns[0].name: getattr(value, prop.key)
        for prop in value.__mapper__.column_attrs
    }


def _table(model: Any) -> Table:
    """Return the Core table so database column names remain unambiguous."""

    return model.__table__


def _comparable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _assert_equal(existing: Any, row: dict[str, Any], *, ignored: set[str]) -> None:
    stored = _model_row(existing)
    for key, value in row.items():
        if key in ignored:
            continue
        if _comparable(stored[key]) != _comparable(value):
            raise PersistenceConflictError(
                f"{existing.__tablename__} stable ID exists with different content"
            )


_RUNTIME_TRANSITIONS = {
    RuntimeStatus.PLANNED: {RuntimeStatus.STARTING, RuntimeStatus.CANCELLED},
    RuntimeStatus.STARTING: {
        RuntimeStatus.RUNNING,
        RuntimeStatus.FAILED,
        RuntimeStatus.CANCELLED,
    },
    RuntimeStatus.RUNNING: {
        RuntimeStatus.COMPLETED,
        RuntimeStatus.FAILED,
        RuntimeStatus.CANCELLED,
    },
    RuntimeStatus.COMPLETED: set(),
    RuntimeStatus.FAILED: set(),
    RuntimeStatus.CANCELLED: set(),
}

_SERVICE_TRANSITIONS = {
    ManagedServiceStatus.PLANNED: {
        ManagedServiceStatus.STARTING,
        ManagedServiceStatus.FAILED,
    },
    ManagedServiceStatus.STARTING: {
        ManagedServiceStatus.READY,
        ManagedServiceStatus.FAILED,
    },
    ManagedServiceStatus.READY: {
        ManagedServiceStatus.STOPPING,
        ManagedServiceStatus.STOPPED,
        ManagedServiceStatus.FAILED,
    },
    ManagedServiceStatus.STOPPING: {
        ManagedServiceStatus.STOPPED,
        ManagedServiceStatus.FAILED,
    },
    ManagedServiceStatus.STOPPED: set(),
    ManagedServiceStatus.FAILED: set(),
}


class ManagedRuntimeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def put_session(self, value: ManagedRuntimeSession) -> ManagedRuntimeSession:
        row = runtime_session_to_row(value)
        existing = await self.session.get(ManagedRuntimeSessionModel, value.runtime_id)
        if existing is None:
            await self.session.execute(insert(_table(ManagedRuntimeSessionModel)).values(row))
            return value
        _assert_equal(
            existing,
            row,
            ignored={
                "status",
                "completed_at",
                "error_type",
                "summary_artifact_id",
                "metadata",
            },
        )
        current = RuntimeStatus(existing.status)
        if current != value.status:
            if value.status not in _RUNTIME_TRANSITIONS[current]:
                raise InvalidStateTransitionError(
                    f"runtime cannot transition from {current.value} to {value.status.value}"
                )
            existing.status = value.status.value
            existing.completed_at = value.completed_at
            existing.error_type = value.error_type
            existing.summary_artifact_id = value.summary_artifact_id
            existing.metadata_json = dict(value.metadata)
        return runtime_session_from_row(_model_row(existing))

    async def put_service(self, value: ManagedServiceState) -> ManagedServiceState:
        row = managed_service_to_row(value)
        existing = await self.session.get(
            ManagedRuntimeServiceModel,
            value.runtime_service_id,
        )
        if existing is None:
            await self.session.execute(insert(_table(ManagedRuntimeServiceModel)).values(row))
            return value
        _assert_equal(
            existing,
            row,
            ignored={
                "status",
                "process_id",
                "observed_peak_memory_bytes",
                "started_at",
                "ready_at",
                "stopped_at",
                "exit_code",
                "error_type",
                "metadata",
            },
        )
        current = ManagedServiceStatus(existing.status)
        if current != value.status:
            if value.status not in _SERVICE_TRANSITIONS[current]:
                raise InvalidStateTransitionError(
                    f"service cannot transition from {current.value} to {value.status.value}"
                )
            existing.status = value.status.value
        existing.process_id = value.process_id
        existing.observed_peak_memory_bytes = value.observed_peak_memory_bytes
        existing.started_at = value.started_at
        existing.ready_at = value.ready_at
        existing.stopped_at = value.stopped_at
        existing.exit_code = value.exit_code
        existing.error_type = value.error_type
        existing.metadata_json = dict(value.metadata)
        return managed_service_from_row(_model_row(existing))

    async def put_allocations(
        self,
        *,
        runtime_id: str,
        runtime_service_id: str,
        stage_id: str,
        allocation: GpuAllocation,
        devices: list[GpuDeviceInfo],
    ) -> None:
        by_id = {item.device_id: item for item in devices}
        for ordinal, (device_id, visible_index) in enumerate(
            zip(allocation.device_ids, allocation.visible_indices, strict=True)
        ):
            device = by_id[device_id]
            row = {
                "runtime_service_id": runtime_service_id,
                "runtime_id": runtime_id,
                "stage_id": stage_id,
                "device_ordinal": ordinal,
                "anonymized_device_id": device_id,
                "visible_index": visible_index,
                "gpu_name": device.name,
                "total_memory_bytes": device.total_memory_bytes,
                "tensor_parallel_size": allocation.tensor_parallel_size,
                "memory_budget_bytes": allocation.memory_budget_bytes_per_gpu,
                "topology_score": allocation.topology_score,
                "metadata": {},
            }
            existing = await self.session.get(
                RuntimeGpuAllocationModel,
                (runtime_service_id, ordinal),
            )
            if existing is None:
                await self.session.execute(insert(_table(RuntimeGpuAllocationModel)).values(row))
            else:
                _assert_equal(existing, row, ignored=set())

    async def put_calibration(self, value: RuntimeCalibration) -> RuntimeCalibration:
        row = runtime_calibration_to_row(value)
        existing = await self.session.get(RuntimeCalibrationModel, value.calibration_id)
        if existing is None:
            await self.session.execute(insert(_table(RuntimeCalibrationModel)).values(row))
            return value
        _assert_equal(existing, row, ignored=set())
        return value

    async def get_calibration(self, calibration_id: str) -> RuntimeCalibration | None:
        existing = await self.session.get(RuntimeCalibrationModel, calibration_id)
        if existing is None:
            return None
        return runtime_calibration_from_row(_model_row(existing))

    async def link_runs_from_model_calls(self, runtime_id: str) -> list[str]:
        run_ids = list(
            (
                await self.session.execute(
                    select(RuntimeModelCallModel.run_id)
                    .where(RuntimeModelCallModel.runtime_id == runtime_id)
                    .distinct()
                )
            ).scalars()
        )
        if run_ids:
            statement = pg_insert(_table(RunRuntimeLinkModel)).values(
                [
                    {
                        "run_id": run_id,
                        "runtime_id": runtime_id,
                        "linked_at": utc_now(),
                        "metadata": {},
                    }
                    for run_id in run_ids
                ]
            )
            await self.session.execute(
                statement.on_conflict_do_nothing(index_elements=["run_id", "runtime_id"])
            )
        return sorted(run_ids)
