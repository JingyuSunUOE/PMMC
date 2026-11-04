"""Persistence for portable baseline family components."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from pmmc.infrastructure.database.baseline_component_models import (
    BaselineComponentImportModel,
    BaselineSuiteFamilyComponentModel,
    BaselineSuiteFamilyModel,
)
from pmmc.infrastructure.database.errors import (
    InvalidStateTransitionError,
    PersistenceConflictError,
)
from pmmc.infrastructure.database.experiment_repositories import (
    _insert_idempotent,
    _model_mapping,
)
from pmmc.schemas import (
    BaselineComponentImportRecord,
    BaselineComponentManifest,
    BaselineComponentState,
    BaselineComponentStatus,
    BaselineFamilyStatus,
    BaselineSuiteFamilyManifest,
    BaselineSuiteFamilyState,
    baseline_component_transition_allowed,
)
from pmmc.schemas.base import utc_now


def _family_row(manifest: BaselineSuiteFamilyManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="python")
    return {
        **{key: value for key, value in payload.items() if key not in {"metadata"}},
        "datasets": [item.value for item in manifest.datasets],
        "methods": [item.value for item in manifest.methods],
        "snapshot_ids": {item.value: manifest.snapshot_ids[item] for item in manifest.datasets},
        "selection_hashes": {
            item.value: manifest.selection_hashes[item] for item in manifest.datasets
        },
        "selected_counts": {
            item.value: manifest.selected_counts[item] for item in manifest.datasets
        },
        "metadata": manifest.metadata,
        "status": BaselineFamilyStatus.REGISTERED.value,
        "expected_component_count": manifest.expected_component_count,
        "verified_component_count": 0,
        "updated_at": manifest.created_at,
        "completed_at": None,
        "error_type": None,
        "error_summary": None,
    }


def _family_manifest(row: dict[str, Any]) -> BaselineSuiteFamilyManifest:
    return BaselineSuiteFamilyManifest.model_validate(
        {
            key: value
            for key, value in row.items()
            if key in BaselineSuiteFamilyManifest.model_fields
        }
    )


def _family_state(row: dict[str, Any]) -> BaselineSuiteFamilyState:
    return BaselineSuiteFamilyState.model_validate(
        {
            "family_id": row["family_id"],
            "status": row["status"],
            "expected_component_count": row["expected_component_count"],
            "verified_component_count": row["verified_component_count"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "error_type": row["error_type"],
            "error_summary": row["error_summary"],
        }
    )


def _component_row(manifest: BaselineComponentManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="python")
    return {
        **{key: value for key, value in payload.items() if key != "metadata"},
        "dataset": manifest.dataset.value,
        "method": manifest.method.value,
        "status": BaselineComponentStatus.REGISTERED.value,
        "transition_ordinal": 0,
        "bundle_export_id": None,
        "bundle_sha256": None,
        "updated_at": manifest.created_at,
        "completed_at": None,
        "error_type": None,
        "error_summary": None,
        "metadata": manifest.metadata,
    }


def _component_manifest(row: dict[str, Any]) -> BaselineComponentManifest:
    return BaselineComponentManifest.model_validate(
        {key: value for key, value in row.items() if key in BaselineComponentManifest.model_fields}
    )


def _component_state(row: dict[str, Any]) -> BaselineComponentState:
    return BaselineComponentState.model_validate(
        {
            "component_id": row["component_id"],
            "status": row["status"],
            "transition_ordinal": row["transition_ordinal"],
            "bundle_export_id": row["bundle_export_id"],
            "bundle_sha256": row["bundle_sha256"],
            "registered_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "error_type": row["error_type"],
            "error_summary": row["error_summary"],
            "metadata": row["metadata"],
        }
    )


def _import_row(record: BaselineComponentImportRecord) -> dict[str, Any]:
    payload = record.model_dump(mode="python")
    return {**payload, "metadata": record.metadata}


class PostgresBaselineComponentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register_family(
        self,
        manifest: BaselineSuiteFamilyManifest,
        components: list[BaselineComponentManifest],
    ) -> BaselineSuiteFamilyState:
        expected_pairs = {
            (dataset.value, method.value)
            for dataset in manifest.datasets
            for method in manifest.methods
        }
        actual_pairs = {(item.dataset.value, item.method.value) for item in components}
        if actual_pairs != expected_pairs or len(components) != len(expected_pairs):
            raise ValueError("baseline family components must form the declared matrix")
        for component in components:
            if component.family_id != manifest.family_id:
                raise ValueError("component belongs to another baseline family")
            if component.snapshot_id != manifest.snapshot_ids[component.dataset]:
                raise ValueError("component snapshot differs from family")
            if component.selection_hash != manifest.selection_hashes[component.dataset]:
                raise ValueError("component denominator differs from family")
            if component.selected_count != manifest.selected_counts[component.dataset]:
                raise ValueError("component selected count differs from family")
            if component.scientific_config_hash != manifest.config_hash:
                raise ValueError("component scientific identity differs from family")
        existing_family = await self.session.get(BaselineSuiteFamilyModel, manifest.family_id)
        if existing_family is None:
            await _insert_idempotent(
                self.session,
                BaselineSuiteFamilyModel,
                _family_row(manifest),
                manifest.family_id,
            )
        elif _family_manifest(_model_mapping(existing_family)) != manifest:
            raise PersistenceConflictError(
                "baseline family ID already belongs to another scientific identity"
            )
        for component in sorted(
            components, key=lambda item: (item.dataset.value, item.method.value)
        ):
            existing_component = await self.session.get(
                BaselineSuiteFamilyComponentModel, component.component_id
            )
            if existing_component is None:
                await _insert_idempotent(
                    self.session,
                    BaselineSuiteFamilyComponentModel,
                    _component_row(component),
                    component.component_id,
                )
            elif _component_manifest(_model_mapping(existing_component)) != component:
                raise PersistenceConflictError(
                    "baseline component ID already belongs to another identity"
                )
        await self.session.execute(
            update(BaselineSuiteFamilyModel)
            .where(
                BaselineSuiteFamilyModel.family_id == manifest.family_id,
                BaselineSuiteFamilyModel.status == BaselineFamilyStatus.REGISTERED.value,
            )
            .values(status=BaselineFamilyStatus.ASSEMBLING.value, updated_at=utc_now())
        )
        await self.session.flush()
        return await self.get_family_state(manifest.family_id)

    async def get_family_manifest(self, family_id: str) -> BaselineSuiteFamilyManifest:
        row = await self.session.get(BaselineSuiteFamilyModel, family_id)
        if row is None:
            raise KeyError(f"baseline family does not exist: {family_id}")
        return _family_manifest(_model_mapping(row))

    async def get_family_state(self, family_id: str) -> BaselineSuiteFamilyState:
        row = await self.session.get(BaselineSuiteFamilyModel, family_id)
        if row is None:
            raise KeyError(f"baseline family does not exist: {family_id}")
        return _family_state(_model_mapping(row))

    async def get_component(
        self, component_id: str
    ) -> tuple[BaselineComponentManifest, BaselineComponentState]:
        row = await self.session.get(BaselineSuiteFamilyComponentModel, component_id)
        if row is None:
            raise KeyError(f"baseline component does not exist: {component_id}")
        mapping = _model_mapping(row)
        return _component_manifest(mapping), _component_state(mapping)

    async def list_components(
        self, family_id: str
    ) -> list[tuple[BaselineComponentManifest, BaselineComponentState]]:
        rows = (
            (
                await self.session.execute(
                    select(BaselineSuiteFamilyComponentModel)
                    .where(BaselineSuiteFamilyComponentModel.family_id == family_id)
                    .order_by(
                        BaselineSuiteFamilyComponentModel.dataset,
                        BaselineSuiteFamilyComponentModel.method,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [
            (_component_manifest(mapping), _component_state(mapping))
            for mapping in (_model_mapping(row) for row in rows)
        ]

    async def bind_run(
        self,
        component_id: str,
        *,
        suite_id: str,
        run_id: str,
        corpus_id: str | None,
    ) -> None:
        result = await self.session.execute(
            update(BaselineSuiteFamilyComponentModel)
            .where(
                BaselineSuiteFamilyComponentModel.component_id == component_id,
                BaselineSuiteFamilyComponentModel.status.in_(
                    {
                        BaselineComponentStatus.REGISTERED.value,
                        BaselineComponentStatus.RETRIEVAL_READY.value,
                    }
                ),
            )
            .values(suite_id=suite_id, run_id=run_id, corpus_id=corpus_id, updated_at=utc_now())
        )
        cursor_result = result  # AsyncSession returns a synchronous cursor result here.
        assert isinstance(cursor_result, CursorResult)
        if cursor_result.rowcount != 1:
            raise InvalidStateTransitionError("component run binding is no longer writable")

    async def transition_component(
        self,
        component_id: str,
        target: BaselineComponentStatus,
        *,
        bundle_export_id: str | None = None,
        bundle_sha256: str | None = None,
        error_type: str | None = None,
        error_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineComponentState:
        row = await self.session.scalar(
            select(BaselineSuiteFamilyComponentModel)
            .where(BaselineSuiteFamilyComponentModel.component_id == component_id)
            .with_for_update()
        )
        if row is None:
            raise KeyError(f"baseline component does not exist: {component_id}")
        current = BaselineComponentStatus(row.status)
        if not baseline_component_transition_allowed(current, target):
            raise InvalidStateTransitionError(
                f"cannot transition component from {current.value} to {target.value}"
            )
        if target is BaselineComponentStatus.FAILED and error_type is None:
            raise ValueError("failed baseline component requires error_type")
        now = utc_now()
        values: dict[str, Any] = {
            "status": target.value,
            "transition_ordinal": row.transition_ordinal + 1,
            "updated_at": now,
            "error_type": error_type,
            "error_summary": error_summary,
        }
        if bundle_export_id is not None:
            values["bundle_export_id"] = bundle_export_id
        if bundle_sha256 is not None:
            values["bundle_sha256"] = bundle_sha256
        if metadata is not None:
            values["metadata_json"] = metadata
        if target in {BaselineComponentStatus.VERIFIED, BaselineComponentStatus.FAILED}:
            values["completed_at"] = now
        await self.session.execute(
            update(BaselineSuiteFamilyComponentModel)
            .where(BaselineSuiteFamilyComponentModel.component_id == component_id)
            .values(**values)
        )
        await self.session.flush()
        if target is BaselineComponentStatus.VERIFIED:
            await self._refresh_family_count(row.family_id)
        return (await self.get_component(component_id))[1]

    async def record_import(self, record: BaselineComponentImportRecord) -> BaselineComponentState:
        manifest, state = await self.get_component(record.component_id)
        if manifest.family_id != record.family_id:
            raise ValueError("component import belongs to another family")
        if state.status is not BaselineComponentStatus.EXPORTED:
            raise InvalidStateTransitionError("only exported components can be imported")
        if (
            state.bundle_export_id != record.bundle_export_id
            or state.bundle_sha256 != record.bundle_sha256
        ):
            raise PersistenceConflictError("component import bundle identity differs")
        await _insert_idempotent(
            self.session,
            BaselineComponentImportModel,
            _import_row(record),
            record.import_id,
        )
        return await self.transition_component(
            record.component_id,
            BaselineComponentStatus.IMPORTED,
        )

    async def complete_family(self, family_id: str) -> BaselineSuiteFamilyState:
        await self._refresh_family_count(family_id)
        state = await self.get_family_state(family_id)
        if state.status is BaselineFamilyStatus.READY:
            return state
        if state.status is not BaselineFamilyStatus.ASSEMBLING:
            raise InvalidStateTransitionError("baseline family is not assembling")
        if state.verified_component_count != state.expected_component_count:
            raise InvalidStateTransitionError("baseline family components are incomplete")
        now = utc_now()
        await self.session.execute(
            update(BaselineSuiteFamilyModel)
            .where(BaselineSuiteFamilyModel.family_id == family_id)
            .values(status=BaselineFamilyStatus.READY.value, updated_at=now, completed_at=now)
        )
        await self.session.flush()
        return await self.get_family_state(family_id)

    async def _refresh_family_count(self, family_id: str) -> None:
        count = int(
            await self.session.scalar(
                select(func.count())
                .select_from(BaselineSuiteFamilyComponentModel)
                .where(
                    BaselineSuiteFamilyComponentModel.family_id == family_id,
                    BaselineSuiteFamilyComponentModel.status
                    == BaselineComponentStatus.VERIFIED.value,
                )
            )
            or 0
        )
        await self.session.execute(
            update(BaselineSuiteFamilyModel)
            .where(BaselineSuiteFamilyModel.family_id == family_id)
            .values(verified_component_count=count, updated_at=utc_now())
        )
