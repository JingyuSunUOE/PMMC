"""Ephemeral canonical fixture for real Questioner schema conformance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from pmmc.benchmarks import build_snapshot, create_adapter
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_engine
from pmmc.infrastructure.database.importer import CanonicalSnapshotImporter
from pmmc.infrastructure.database.storage_updates import persist_asset_sync
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import Settings
from pmmc.infrastructure.storage_sync import sync_assets
from pmmc.schemas import DatasetName, ValidationLevel
from pmmc.smoke.step5d import assert_ephemeral_endpoints

STEP6B_SCHEMA_GOLD_SENTINEL = "PMMC_STEP6B_SCHEMA_GOLD_SENTINEL_9E41C7A2"


def build_step6b_schema_source_fixture(settings: Settings) -> Path:
    """Create one-session MemLens input; compiler code cannot read its gold row."""

    artifacts = settings.artifacts_path
    if not artifacts.is_relative_to(settings.project_root):
        artifacts = settings.project_root / ".smoke" / "artifacts"
    root = artifacts / "smoke" / "step6b-schema" / "source" / "memlens"
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "judge").mkdir(parents=True, exist_ok=True)
    image_path = root / "images" / "needle_images" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color=(0, 255, 0)).save(image_path, format="PNG")
    payload = [
        {
            "question_id": "step6b_schema_query",
            "question_type": "answer_refusal",
            "question_subtype": "answer_refusal",
            "question": "What was not stated about the photographed sample?",
            "answer": STEP6B_SCHEMA_GOLD_SENTINEL,
            "question_date": "2025/01/02 (Thu) 12:00",
            "haystack_dates": ["2025/01/01 (Wed) 10:00"],
            "haystack_session_ids": ["schema_session_1"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "I photographed the green sample.",
                        "images": [
                            {
                                "file": "needle_images/sample.png",
                                "image_url": "https://example.invalid/sample.png",
                                "blip_caption": "a green sample",
                            }
                        ],
                    },
                    {
                        "role": "AI Assistant",
                        "content": "I will remember it.",
                        "images": [],
                    },
                ]
            ],
            "answer_session_ids": ["schema_session_1"],
        }
    ]
    (root / "data" / "dataset_32k.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return root


async def bootstrap_step6b_schema_fixture(
    settings: Settings,
    *,
    confirm_ephemeral: bool,
) -> dict[str, Any]:
    """Import one deterministic multimodal conversation into Pod-local services."""

    assert_ephemeral_endpoints(settings, confirmed=confirm_ephemeral)
    source_root = build_step6b_schema_source_fixture(settings)
    snapshot_path, snapshot = build_snapshot(
        create_adapter(
            DatasetName.MEMLENS,
            root=source_root,
            project_root=settings.project_root,
        ),
        output_root=settings.benchmark_artifacts_path,
        level=ValidationLevel.FULL,
        enforce_expected_counts=False,
    )
    admin_engine = create_engine(settings, DatabaseRole.ADMIN)
    try:
        await CanonicalSnapshotImporter(
            admin_engine,
            batch_size=settings.database_import_batch_size,
        ).import_snapshot(snapshot_path)
        synchronized = await sync_assets(
            snapshot_path,
            project_root=settings.project_root,
            runtime_store=create_runtime_object_store(settings),
        )
        await persist_asset_sync(admin_engine, synchronized)
    finally:
        await admin_engine.dispose()
    return {
        "dataset": DatasetName.MEMLENS.value,
        "snapshot_id": snapshot.snapshot_id,
        "conversation_count": snapshot.counts.get("conversations", 0),
        "smoke_fixture": True,
        "paper_result": False,
    }
