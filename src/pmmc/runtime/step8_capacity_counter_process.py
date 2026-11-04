"""Killable, CPU-only native counting over a private authenticated pipe.

No prompts, counter options, or exception details are written to files or logs.
The child retains one counter; no model code is imported by the parent module.
"""

from __future__ import annotations

import asyncio
import json
import math
import multiprocessing
import os
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
from pathlib import Path
from typing import Any

_AUTH_ENV = "_PMMC_NATIVE_COUNTER_AUTH"
_CHILD_BOOTSTRAP = (
    "from pmmc.runtime.step8_capacity_counter_process import _child_main; _child_main()"
)
_PATH_OPTIONS = {"project_root", "cache_dir", "local_asset_cache_dir"}
_MAX_REPLY_BYTES = 1_048_576
_HEARTBEAT_SECONDS = 30.0


class NativeCounterProcessError(RuntimeError):
    """Opaque failure code only, never a child exception or transport payload."""


def _profile(value: object) -> tuple[int, list[int]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError
    total, tokens = value
    if (
        type(total) is not int
        or total <= 0
        or not isinstance(tokens, list)
        or any(type(token) is not int or token <= 0 for token in tokens)
        or sum(tokens) > total
    ):
        raise ValueError
    return total, tokens


def _send(connection: Connection, value: object) -> None:
    connection.send_bytes(json.dumps(value, allow_nan=False).encode("utf-8"))


def _receive(connection: Connection, *, max_bytes: int | None = None) -> Any:
    return json.loads(connection.recv_bytes(maxlength=max_bytes))


def _load_counter(kwargs: dict[str, Any]) -> Any:
    # The subprocess launch environment already masks CUDA, before Python starts.
    import torch

    from pmmc.models.tokenization import ProcessorTokenCounter

    torch.set_default_device("cpu")
    return ProcessorTokenCounter(**kwargs)


def _serve_counter(
    connection: Connection,
    authkey: bytes,
    factory: Callable[[dict[str, Any]], Any],
) -> None:
    request_id = 0
    authenticated = False
    try:
        answer_challenge(connection, authkey)
        deliver_challenge(connection, authkey)
        authenticated = True
        initialization = _receive(connection)
        if initialization["id"] != 0 or initialization["op"] != "initialize":
            raise ValueError
        kwargs = initialization["counter_kwargs"]
        if kwargs.get("local_files_only") is not True or kwargs.get("processor_kwargs", {}) != {}:
            raise ValueError
        for name in _PATH_OPTIONS:
            if kwargs.get(name) is not None:
                kwargs[name] = Path(kwargs[name])
        counter = factory(kwargs)
        _send(connection, {"id": 0, "ok": True})
        while True:
            request = _receive(connection)
            if (
                type(request["id"]) is not int
                or request["id"] != request_id + 1
                or request["op"] not in {"count", "validate"}
            ):
                raise ValueError
            request_id = request["id"]
            if request["op"] == "validate" or request["native_layout"] is not None:
                from pmmc.runtime.step8_capacity_images import validate_native_messages
                from pmmc.runtime.step8_capacity_native import NativeImageLayout

                layout = NativeImageLayout.model_validate(request["native_layout"])
                if request["op"] == "validate":
                    count = request["count"]
                    if type(count) is not int or count <= 0:
                        raise ValueError
                    validate_native_messages(layout, request["prompts"], count=count)
                    _send(connection, {"id": request_id, "ok": True})
                    continue
                validate_native_messages(layout, [request["messages"]], count=1)
            total, tokens = _profile(counter.count_transport_profile(request["messages"]))
            _send(connection, {"id": request_id, "ok": True, "profile": [total, tokens]})
    except BaseException:
        if authenticated:
            try:
                _send(connection, {"id": request_id, "ok": False})
            except BaseException:
                pass
    finally:
        connection.close()


def _child_main() -> None:
    # No application main module is re-imported, unlike multiprocessing's spawn
    # bootstrap. CUDA is hidden even if that application imports torch globally.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        authkey = bytes.fromhex(os.environ.pop(_AUTH_ENV))
        connection = Connection(int(sys.argv[1]))
        _serve_counter(connection, authkey, _load_counter)
    except BaseException:
        # stdout/stderr are also discarded at the process boundary, including
        # native library diagnostics and interpreter-level child failures.
        return


@dataclass
class _Operation:
    request_id: int
    phase: str
    started: float = field(default_factory=time.monotonic)
    completed: threading.Event = field(default_factory=threading.Event)
    completed_at: float | None = None
    heartbeat_at: float = 0.0
    reply: Any = None
    failed: bool = False


class NativeCounterProcess:
    """One persistent child and at most one in-flight operation.

    ``counter_kwargs`` are ProcessorTokenCounter constructor options. Require
    ``local_files_only=True`` and native ``processor_kwargs={}`` (or omitted).
    Initialization, counting, and pool validation each have a timeout deadline.
    Optional ``native_layout`` is a NativeImageLayout or its JSON-compatible dict;
    its messages are validated in the child before counting. Returned grid tokens
    are actual processor results, not substituted declared layout counts.

    Use as a context manager, or call close() in finally BEFORE publishing any
    receipt. Cleanup failure raises capacity_native_counter_cleanup_unconfirmed.
    Cancellation, timeout, or child failure permanently closes this instance.
    Concurrent counts are rejected; a replacement child is never silently started.
    """

    def __init__(
        self,
        counter_kwargs: dict[str, Any],
        timeout_seconds: float,
        *,
        shutdown_timeout_seconds: float = 1.0,
    ) -> None:
        try:
            if any(
                type(value) not in {int, float} or not math.isfinite(value) or value <= 0
                for value in (timeout_seconds, shutdown_timeout_seconds)
            ):
                raise ValueError
            if timeout_seconds > 3600 or shutdown_timeout_seconds > 10:
                raise ValueError
            kwargs = {
                key: os.fspath(value) if key in _PATH_OPTIONS and value is not None else value
                for key, value in counter_kwargs.items()
            }
            kwargs = json.loads(json.dumps(kwargs, allow_nan=False))
            if (
                kwargs.get("local_files_only") is not True
                or kwargs.get("processor_kwargs", {}) != {}
            ):
                raise ValueError
        except Exception:
            raise NativeCounterProcessError(
                "capacity_native_counter_configuration_invalid"
            ) from None
        self.timeout_seconds = float(timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._busy = False
        self._next_id = 0
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._connection, child_connection = multiprocessing.Pipe(duplex=True)
        self._authkey = secrets.token_bytes(32)
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                _AUTH_ENV: self._authkey.hex(),
                "PYTHONPATH": os.pathsep.join(path for path in sys.path if isinstance(path, str)),
            }
        )
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-c", _CHILD_BOOTSTRAP, str(child_connection.fileno())],
                env=env,
                pass_fds=(child_connection.fileno(),),
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self._connection.close()
            self._closed = True
            raise NativeCounterProcessError("capacity_native_counter_start_failed") from None
        finally:
            child_connection.close()
        self._run_sync({"op": "initialize", "counter_kwargs": kwargs}, initialize=True)

    def _begin(self, request: dict[str, Any], *, initialize: bool = False) -> _Operation:
        with self._state_lock:
            if self._closed:
                raise NativeCounterProcessError("capacity_native_counter_closed")
            if self._busy:
                raise NativeCounterProcessError("capacity_native_counter_busy")
            self._busy = True
            operation = _Operation(self._next_id, "init" if initialize else request["op"])
            operation.heartbeat_at = operation.started + _HEARTBEAT_SECONDS
            self._next_id += 1

            def exchange() -> None:
                try:
                    if initialize:
                        deliver_challenge(self._connection, self._authkey)
                        answer_challenge(self._connection, self._authkey)
                    value = dict(request, id=operation.request_id)
                    layout = value.get("native_layout")
                    if layout is not None and not isinstance(layout, dict):
                        value["native_layout"] = layout.model_dump(mode="json")
                    _send(self._connection, value)
                    operation.reply = _receive(self._connection, max_bytes=_MAX_REPLY_BYTES)
                except BaseException:
                    operation.failed = True
                finally:
                    operation.completed_at = time.monotonic()
                    operation.completed.set()

            self._thread = threading.Thread(target=exchange, daemon=True, name="native-counter-ipc")
            try:
                self._thread.start()
            except Exception:
                pass
            else:
                return operation
        self.close()
        raise NativeCounterProcessError("capacity_native_counter_start_failed") from None

    def _finish(self, operation: _Operation) -> Any:
        try:
            reply = operation.reply
            if (
                operation.failed
                or not isinstance(reply, dict)
                or type(reply.get("id")) is not int
                or reply["id"] != operation.request_id
                or reply.get("ok") is not True
            ):
                raise ValueError
            result = _profile(reply["profile"]) if operation.phase == "count" else None
        except Exception:
            raise NativeCounterProcessError("capacity_native_counter_failed") from None
        with self._state_lock:
            if self._closed:
                raise NativeCounterProcessError("capacity_native_counter_closed")
            self._busy = False
        return result

    def _waiting(self, operation: _Operation) -> bool:
        now = time.monotonic()
        complete = operation.completed.is_set()
        finished = operation.completed_at
        elapsed = (finished if complete and finished is not None else now) - operation.started
        if elapsed > self.timeout_seconds:
            raise NativeCounterProcessError("capacity_native_counter_timeout")
        if not complete and now >= operation.heartbeat_at:
            print(
                f"STEP8_CAPACITY_COUNTER phase={operation.phase} state=running "
                f"elapsed_seconds={int(now - operation.started)}",
                flush=True,
            )
            operation.heartbeat_at = now + _HEARTBEAT_SECONDS
        return not complete

    def _run_sync(self, request: dict[str, Any], *, initialize: bool = False) -> Any:
        operation = self._begin(request, initialize=initialize)
        try:
            while self._waiting(operation):
                remaining = operation.started + self.timeout_seconds - time.monotonic()
                operation.completed.wait(max(0, min(_HEARTBEAT_SECONDS, remaining)))
            return self._finish(operation)
        except BaseException:
            self.close()
            raise

    def count_transport_profile(
        self,
        messages: list[dict[str, Any]],
        native_layout: Any = None,
    ) -> tuple[int, list[int]]:
        return self._run_sync({"op": "count", "messages": messages, "native_layout": native_layout})

    async def count_async(
        self,
        messages: list[dict[str, Any]],
        native_layout: Any = None,
    ) -> tuple[int, list[int]]:
        return await self._run_async(
            {"op": "count", "messages": messages, "native_layout": native_layout}
        )

    async def validate_async(self, layout: Any, prompts: list, count: int) -> None:
        """Validate the entire pool, including duplicate pixels across sequences."""
        await self._run_async(
            {"op": "validate", "native_layout": layout, "prompts": prompts, "count": count}
        )

    async def _run_async(self, request: dict[str, Any]) -> Any:
        operation = self._begin(request)
        try:
            # A private daemon thread handles blocking pipe I/O. It is explicitly
            # joined by close(), never registered with asyncio's default executor.
            while self._waiting(operation):
                await asyncio.sleep(0.01)
            return self._finish(operation)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Terminate, escalate to kill, and confirm both child and IPC thread exit."""
        deadline = time.monotonic() + self.shutdown_timeout_seconds
        if not self._close_lock.acquire(timeout=self.shutdown_timeout_seconds):
            raise NativeCounterProcessError("capacity_native_counter_cleanup_unconfirmed") from None
        try:
            with self._state_lock:
                self._closed = True
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=min(0.1, max(0, deadline - time.monotonic()) / 3))
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if process.poll() is None:
                    try:
                        process.kill()
                        process.wait(timeout=max(0, deadline - time.monotonic()) * 0.75)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            # Killing the peer releases a blocked send/recv. Do not close/reuse
            # the parent's descriptor while its IPC thread might still access it.
            thread = self._thread
            if thread is not None and thread.ident is not None:
                thread.join(timeout=max(0, deadline - time.monotonic()))
            if (process is not None and process.poll() is None) or (
                thread is not None and thread.is_alive()
            ):
                raise NativeCounterProcessError("capacity_native_counter_cleanup_unconfirmed")
            self._connection.close()
        except NativeCounterProcessError:
            raise
        except Exception:
            raise NativeCounterProcessError("capacity_native_counter_cleanup_unconfirmed") from None
        finally:
            self._close_lock.release()

    def __enter__(self) -> NativeCounterProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
