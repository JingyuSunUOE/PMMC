"""Pod-local advisory lock for exclusive managed GPU lifecycle ownership."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from types import TracebackType
from typing import TextIO


class RuntimeLockError(RuntimeError):
    """Another managed runtime currently owns this Pod's GPU lifecycle."""


class RuntimeProcessLock:
    """Use kernel flock state; stale lock-file contents never block a new owner."""

    def __init__(self, path: Path, runtime_id: str) -> None:
        self.path = path
        self.runtime_id = runtime_id
        self._stream: TextIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeLockError(
                "another managed runtime is already active in this Pod"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {"pid": os.getpid(), "runtime_id": self.runtime_id},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None

    def set_runtime_id(self, runtime_id: str) -> None:
        """Update diagnostics after planning assigns the immutable runtime ID."""

        if self._stream is None:
            raise RuntimeLockError("runtime lock must be held before updating its owner")
        self.runtime_id = runtime_id
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(
            json.dumps(
                {"pid": os.getpid(), "runtime_id": runtime_id},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        self._stream.flush()

    def __enter__(self) -> RuntimeProcessLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
