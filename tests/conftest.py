"""Pytest path setup for source-layout smoke tests."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def host_release_source(tmp_path_factory):
    """Isolated installation for host/CLI tests; never add a fixture lock to the checkout."""
    from scripts import release_images

    root = tmp_path_factory.mktemp("public-host")
    names = {
        "scripts/pmmc",
        "scripts/release_source.py",
        "scripts/release_images.py",
        "src/pmmc/__init__.py",
        "src/pmmc/release_plan.py",
        "deploy/release/kubernetes.toml",
        "deploy/release/upstream.json",
        "SOURCE.sha256",
    }
    names.update(
        n.removeprefix("!")
        for n in (ROOT / ".dockerignore").read_text().splitlines()
        if n.startswith("!scripts/release_host/") and n.endswith(".py")
    )
    for n in names:
        (root / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / n, root / n)
    lock = {
        "schema_version": 1,
        "source_commit": "1" * 40,
        "source_manifest_sha256": release_images.sha((root / "SOURCE.sha256").read_bytes()),
        "images": {
            **json.loads((root / release_images.UPSTREAM).read_bytes()),
            "app": "ghcr.io/fixture/app@sha256:" + "a" * 64,
            "iam": "ghcr.io/fixture/iam@sha256:" + "b" * 64,
        },
    }
    data = json.dumps(lock).encode()
    (root / release_images.LOCK).write_bytes(data)
    return root, release_images.sha(data)


@pytest.fixture
def installed_host_lock(host_release_source, monkeypatch, request):
    from scripts.release_host import config

    root, checksum = host_release_source
    monkeypatch.setenv("PMMC_IMAGE_LOCK_SHA256", checksum)
    monkeypatch.setattr(config, "ROOT", root)
    monkeypatch.setattr(request.module, "ROOT", root)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color=color).save(path)


@pytest.fixture
def memlens_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "benchmarks" / "memlens"
    (root / "data").mkdir(parents=True)
    (root / "prompts" / "judge").mkdir(parents=True)
    _write_image(root / "images" / "needle_images" / "needle.png", (255, 0, 0))
    payload = [
        {
            "question_id": "source_q1",
            "question_type": "answer_refusal",
            "question_subtype": "answer_refusal",
            "question": "What color was the missing item?",
            "answer": "GOLD_ANSWER_SENTINEL",
            "question_date": "2025/01/02 (Thu) 12:00",
            "haystack_dates": ["2025/01/01 (Wed) 10:00"],
            "haystack_session_ids": ["session_1"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "Look at this image.",
                        "images": [
                            {
                                "file": "needle_images/needle.png",
                                "image_url": "https://example.invalid/needle.png",
                                "blip_caption": "a red square",
                            }
                        ],
                    },
                    {"role": "AI Assistant", "content": "I can see it.", "images": []},
                    {"role": "user", "content": "", "images": []},
                ]
            ],
            "answer_session_ids": ["session_1"],
        }
    ]
    (root / "data" / "dataset_32k.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def memgallery_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "benchmarks" / "memgallery"
    dialogs = root / "data" / "dialogs"
    dialogs.mkdir(parents=True)
    (root / "prompts" / "source").mkdir(parents=True)
    _write_image(root / "images" / "scenario_one" / "history.png", (0, 255, 0))
    _write_image(root / "images" / "scenario_one" / "question.png", (0, 0, 255))
    payload = {
        "character_profile": {
            "name": "Avery",
            "persona_summary": "A careful researcher.",
            "traits": ["careful", "curious"],
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
                        "input_image": ["../image/scenario_one/history.png"],
                        "image_caption": ["a green sample"],
                    }
                ],
            }
        ],
        "human-annotated QAs": [
            {
                "point": "FR",
                "question": "What color was the sample?",
                "question_image": "../image/scenario_one/question.png",
                "image_caption": "a blue query card",
                "answer": "GOLD_ANSWER_SENTINEL",
                "session_id": ["D1"],
                "clue": ["D1:1"],
            },
            {
                "point": "AR",
                "question": "What was never discussed?",
                "answer": "Not mentioned.",
                "session_id": [],
                "clue": [],
            },
        ],
    }
    (dialogs / "scenario_one.json").write_text(json.dumps(payload), encoding="utf-8")
    return root
