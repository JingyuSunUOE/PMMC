"""Signal conversion that guarantees managed context-manager cleanup."""

from __future__ import annotations

import signal
import threading
from types import FrameType, TracebackType
from typing import Any


class ManagedSignalGuard:
    """Convert SIGINT/SIGTERM into KeyboardInterrupt while services are active."""

    def __init__(self) -> None:
        self._previous: dict[signal.Signals, Any] = {}

    @staticmethod
    def _interrupt(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        raise KeyboardInterrupt

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for name in (signal.SIGINT, signal.SIGTERM):
            self._previous[name] = signal.getsignal(name)
            signal.signal(name, self._interrupt)

    def restore(self) -> None:
        for name, handler in self._previous.items():
            signal.signal(name, handler)
        self._previous.clear()

    def __enter__(self) -> ManagedSignalGuard:
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.restore()
