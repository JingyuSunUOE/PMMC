"""Schema/row mappings for managed runtime persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pmmc.schemas import (
    ManagedRuntimeSession,
    ManagedServiceState,
    RuntimeCalibration,
)

Row = dict[str, Any]


def runtime_session_to_row(value: ManagedRuntimeSession) -> Row:
    row = value.model_dump(mode="python")
    row["status"] = value.status.value
    row["metadata"] = dict(value.metadata)
    return row


def runtime_session_from_row(row: Mapping[str, Any]) -> ManagedRuntimeSession:
    return ManagedRuntimeSession.model_validate(dict(row))


def managed_service_to_row(value: ManagedServiceState) -> Row:
    row = value.model_dump(mode="python")
    row["role"] = value.role.value
    row["backend"] = value.backend.value
    row["status"] = value.status.value
    row["allocation_json"] = value.allocation.model_dump(mode="json")
    row.pop("allocation")
    row["metadata"] = dict(value.metadata)
    return row


def managed_service_from_row(row: Mapping[str, Any]) -> ManagedServiceState:
    data = dict(row)
    data["allocation"] = data.pop("allocation_json")
    return ManagedServiceState.model_validate(data)


def runtime_calibration_to_row(value: RuntimeCalibration) -> Row:
    row = value.model_dump(mode="python")
    row["status"] = value.status.value
    row["allocation_json"] = value.allocation.model_dump(mode="json")
    row.pop("allocation")
    row["metadata"] = dict(value.metadata)
    return row


def runtime_calibration_from_row(row: Mapping[str, Any]) -> RuntimeCalibration:
    data = dict(row)
    data["allocation"] = data.pop("allocation_json")
    return RuntimeCalibration.model_validate(data)
