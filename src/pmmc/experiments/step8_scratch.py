"""Attempt-owned scratch for worker assets and supervisor model staging."""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pmmc.experiments.step8_artifacts import write_json
from pmmc.infrastructure.settings import Settings

_DEFERRED: ContextVar[set[str] | None] = ContextVar("step8_scratch_cleanup_deferred", default=None)


def defer_scratch_cleanup(reason: str) -> None:
    reasons = _DEFERRED.get()
    if reasons is not None:
        reasons.add(reason)


def _unlock_owned_directories(root: Path) -> int:
    # Staged snapshots use 0555 directories. Never chmod files (possibly hardlinks)
    # or follow a symlink into the persistent model cache; operate on directory FDs.
    changed = 0
    for _, _, _, fd in os.fwalk(root, follow_symlinks=False):
        mode = stat.S_IMODE(os.fstat(fd).st_mode)
        if mode & 0o700 != 0o700:
            os.fchmod(fd, mode | 0o700)
            changed += 1
    return changed


def _cleanup_event(attempt_id: str, state: str, **fields: object) -> None:
    print(
        "STEP8_SCRATCH_CLEANUP "
        + json.dumps({"attempt_id": attempt_id, "state": state, **fields}, sort_keys=True),
        flush=True,
    )


@contextmanager
def attempt_scratch(
    settings: Settings, *, attempt_id: str, campaign_dir: Path
) -> Iterator[Settings]:
    if not attempt_id.startswith("attempt-") or not attempt_id[8:].isalnum():
        raise ValueError("scratch_attempt_id_invalid")
    base = (
        settings.resolve_path(settings.local_scratch_dir)
        if settings.local_scratch_dir
        else campaign_dir / "private/scratch"
    ).resolve()
    # Children get their own directory, never ownership of the supervisor's staging.
    root = base / "step8-attempts" / attempt_id
    if root.exists() or root.is_symlink():
        raise ValueError("scratch_attempt_directory_already_exists")
    model_cache = settings.model_cache_path.resolve()
    if base.is_relative_to(model_cache) or model_cache.is_relative_to(root):
        raise ValueError("scratch_cannot_own_persistent_model_cache")
    root.mkdir(parents=True)
    owner = {"attempt_id": attempt_id, "protocol": "step8-owned-scratch-v1"}
    write_json(root / "owner.json", owner, immutable=True)
    receipt = campaign_dir / "attempts" / f"{attempt_id}-scratch.json"
    write_json(receipt, {**owner, "path": str(root), "state": "allocated"})
    primary: BaseException | None = None
    deferred: set[str] = set()
    token = _DEFERRED.set(deferred)
    try:
        yield settings.model_copy(
            update={"local_scratch_dir": root, "token_counter_cache_dir": root / "token-counters"}
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        _DEFERRED.reset(token)
        try:
            # Runs after child/model cleanup unwinds. Scientific facts use other roots.
            if root.is_symlink() or json.loads((root / "owner.json").read_bytes()) != owner:
                raise ValueError("scratch_cleanup_ownership_mismatch")
            if deferred:
                write_json(
                    receipt,
                    {
                        **owner,
                        "path": str(root),
                        "state": "cleanup_deferred",
                        "reasons": sorted(deferred),
                        "model_weights_deleted": False,
                        "requires_verified_process_shutdown": True,
                    },
                )
            else:
                unlocked = _unlock_owned_directories(root)
                shutil.rmtree(root)
                write_json(
                    receipt,
                    {
                        **owner,
                        "path": str(root),
                        "state": "removed",
                        "model_weights_deleted": False,
                        "readonly_directories_unlocked": unlocked,
                    },
                )
                _cleanup_event(attempt_id, "removed", readonly_directories_unlocked=unlocked)
        except BaseException as error:
            failure = {
                **owner,
                "path": str(root),
                "state": "cleanup_failed",
                "error_type": type(error).__name__,
                "os_errno": error.errno if isinstance(error, OSError) else None,
                "model_weights_deleted": False,
            }
            try:
                write_json(receipt, failure)
                _cleanup_event(
                    attempt_id,
                    "cleanup_failed",
                    error_type=type(error).__name__,
                    os_errno=failure["os_errno"],
                )
            except BaseException:
                error.add_note("scratch_cleanup_failure_receipt_unavailable")
            if primary is None:
                raise
            primary.add_note("scratch_cleanup_failed: " + type(error).__name__)
