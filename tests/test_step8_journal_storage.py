import asyncio
import errno
import fcntl
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.experiments.step8_cli import _preflight_response_journal
from pmmc.schemas.step8 import ResourceClass
from test_step8_accounting import request, setup


@pytest.fixture
def nfs_record_locks(monkeypatch):
    # Linux NFS implements flock via record locks, which enforce FD access mode.
    monkeypatch.setattr(fcntl, "flock", fcntl.lockf)


def test_record_lock_reproduces_write_only_shared_lock_ebadf(tmp_path):
    with (tmp_path / "old.lock").open("a") as stream:
        with pytest.raises(OSError) as failure:
            fcntl.lockf(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert failure.value.errno == errno.EBADF


@pytest.mark.parametrize("exclusive", [False, True])
def test_nfs_collection_lock_supports_both_modes(tmp_path, nfs_record_locks, exclusive):
    store = CampaignArtifactStore(tmp_path)
    with store.collection_lock(exclusive=exclusive):
        pass
    assert store.load_json(store.put_json({"fixture": True})) == {"fixture": True}


@pytest.mark.parametrize("exclusive", [False, True])
def test_native_lock_still_excludes_gc_and_releases_after_error(tmp_path, exclusive):
    store = CampaignArtifactStore(tmp_path)
    code = """
import fcntl, sys
with open(sys.argv[1], 'a+') as stream:
    try:
        fcntl.flock(stream, int(sys.argv[2]) | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(10)
"""

    def contend(mode):
        return subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "_store/.gc.lock"), str(mode)],
            check=False, timeout=10,
        ).returncode

    with pytest.raises(RuntimeError, match="fixture"):
        with store.collection_lock(exclusive=exclusive):
            assert contend(fcntl.LOCK_EX) == 10
            assert contend(fcntl.LOCK_SH) == (10 if exclusive else 0)
            raise RuntimeError("fixture")
    assert contend(fcntl.LOCK_EX) == 0


def test_nfs_paid_response_journal_settles_and_replays_without_second_call(
    tmp_path, nfs_record_locks,
):
    wrapped, repo, gateway, store = setup(tmp_path)
    store.preflight_response_journal()
    result = asyncio.run(wrapped.generate(request()))
    assert repo.rows[request().request_id]["state"] == "settled"
    replay = asyncio.run(wrapped.generate(request()))
    assert replay == result and gateway.calls == 1


def test_journal_preflight_repeats_writes_without_accumulating_indexes(
    tmp_path, nfs_record_locks, monkeypatch,
):
    store = CampaignArtifactStore(tmp_path)
    original = store.journal_response
    seen = []

    def capture(**kwargs):
        seen.append(kwargs["request_hash"])
        return original(**kwargs)

    monkeypatch.setattr(store, "journal_response", capture)
    assert store.preflight_response_journal() == store.preflight_response_journal()
    assert len(set(seen)) == 2
    assert list((tmp_path / "_store/responses").iterdir()) == []
    assert len(list((tmp_path / "_store/sha256").glob("*/*"))) == 1


def test_preflight_does_not_delete_real_journals_or_rewrite_lock(tmp_path):
    store = CampaignArtifactStore(tmp_path)
    store.journal_response(
        request_id="paid", request_hash="a" * 64, response={"value": 1}, attempt_id="attempt"
    )
    journal = tmp_path / "_store/responses" / ("a" * 64 + ".json")
    before = journal.read_bytes()
    lock = tmp_path / "_store/.gc.lock"
    lock.write_text("existing lock identity")
    inode = lock.stat().st_ino
    report = _preflight_response_journal(tmp_path)
    assert report["state"] == "ready" and report["provider_calls"] == 0
    assert lock.stat().st_ino == inode and lock.read_text() == "existing lock identity"
    assert journal.read_bytes() == before


def test_preflight_rejects_readback_mismatch_and_cleans_own_index(tmp_path, monkeypatch):
    store = CampaignArtifactStore(tmp_path)
    monkeypatch.setattr(store, "restored_response", lambda **_: None)
    with pytest.raises(ValueError, match="readback mismatch"):
        store.preflight_response_journal()
    assert list((tmp_path / "_store/responses").iterdir()) == []


def test_repeated_preflight_checks_new_index_writes(tmp_path, monkeypatch):
    from pmmc.experiments import step8_artifacts

    store = CampaignArtifactStore(tmp_path)
    store.preflight_response_journal()
    original = step8_artifacts.write_json

    def unavailable(path, *args, **kwargs):
        if path.parent.name == "responses":
            raise OSError(errno.ENOSPC, "fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(step8_artifacts, "write_json", unavailable)
    with pytest.raises(OSError) as failure:
        store.preflight_response_journal()
    assert failure.value.errno == errno.ENOSPC


def test_cleanup_error_does_not_mask_storage_root_cause(tmp_path, monkeypatch):
    store = CampaignArtifactStore(tmp_path)

    def fail_write(**kwargs):
        raise OSError(errno.EBADF, "fixture root cause")

    def fail_cleanup(*args, **kwargs):
        raise OSError(errno.EACCES, "fixture cleanup")

    monkeypatch.setattr(store, "journal_response", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError) as failure:
        store.preflight_response_journal()
    assert failure.value.errno == errno.EBADF
    assert "synthetic journal probe cleanup failed" in failure.value.__notes__


def test_cli_blocks_before_prefix_or_provider_on_journal_failure(tmp_path, monkeypatch, capsys):
    from pmmc.experiments import step8_cli, step8_prefix

    monkeypatch.setattr(step8_cli, "_load", lambda _: (
        SimpleNamespace(phase=step8_cli.CampaignPhase.DEVELOPMENT),
        SimpleNamespace(output_root=tmp_path)
    ))
    monkeypatch.setattr(step8_cli, "_register", AsyncMock(return_value=("approval", {})))
    monkeypatch.setattr(step8_cli, "_preflight_rate_card", lambda *args: None)
    monkeypatch.setattr(
        step8_cli, "get_settings", lambda: SimpleNamespace(resolve_path=lambda p: p)
    )
    prefix = AsyncMock()
    monkeypatch.setattr(step8_prefix, "require_real_prefix", prefix)

    def broken(*args):
        raise OSError(errno.EBADF, "sensitive provider payload must not be printed")

    monkeypatch.setattr(fcntl, "flock", broken)
    with pytest.raises(OSError):
        step8_cli._run_campaign(
            campaign_dir=tmp_path, resource=ResourceClass.API, worker_pool="all", stage=None,
            budget_usd="20", rate_card=tmp_path / "card.json", approver_id="operator",
            retry_failed=False, export_on_complete=False,
        )
    prefix.assert_not_called()
    output = capsys.readouterr().out
    assert "sensitive" not in output
    lines = [json.loads(line) for line in output.splitlines()]
    assert [v["state"] for v in lines] == ["start", "failed"]
    assert lines[-1]["diagnostics"]["os_error_name"] == "EBADF"
