"""One bounded checkout budget across all engines in a campaign worker."""

from __future__ import annotations

import asyncio
import fcntl
import os
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.pool import AsyncAdaptedQueuePool, ConnectionPoolEntry
from sqlalchemy.util.concurrency import await_only

_LIMIT: ContextVar[DatabaseWorkLimit | None] = ContextVar("database_work_limit", default=None)
_GETTING: ContextVar[object | None] = ContextVar("database_pool_getting", default=None)


class DatabaseWorkLimit:
    def __init__(self, maximum: int, *, shared_root: Path | None = None) -> None:
        if maximum < 1:
            raise ValueError("database_work_limit_must_be_positive")
        self.maximum = maximum
        self.semaphore = asyncio.Semaphore(maximum)
        self.inflight = 0
        self.peak = 0
        self.wait_seconds = 0.0
        self.shared_root = shared_root
        self._permits: deque[int] = deque()
        if shared_root is not None:
            shared_root.mkdir(parents=True, exist_ok=True)
            if shared_root.is_symlink():
                raise ValueError("database_permit_root_symlink_forbidden")
            from pmmc.experiments.step8_artifacts import write_json

            write_json(shared_root / "budget.json", {"maximum": maximum}, immutable=True)

    async def acquire(self) -> None:
        started = time.monotonic()
        await self.semaphore.acquire()
        try:
            if self.shared_root is not None:
                while True:
                    descriptor = self._try_shared_permit()
                    if descriptor is not None:
                        self._permits.append(descriptor)
                        break
                    await asyncio.sleep(0.05)
        except BaseException:
            self.semaphore.release()
            raise
        self.wait_seconds += time.monotonic() - started
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)

    def release(self) -> None:
        if self.inflight <= 0:
            raise RuntimeError("database_work_limit_unbalanced_release")
        if self.shared_root is not None:
            os.close(self._permits.popleft())
        self.inflight -= 1
        self.semaphore.release()

    def _try_shared_permit(self) -> int | None:
        assert self.shared_root is not None
        for slot in range(self.maximum):
            path = self.shared_root / f"slot-{slot}"
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
            else:
                return descriptor
        return None


@dataclass
class _CheckoutBudget:
    limit: DatabaseWorkLimit
    terminate: Callable[[DBAPIConnection], None]
    checking_in: bool = False


@contextmanager
def database_work_scope(limit: DatabaseWorkLimit) -> Iterator[None]:
    token = _LIMIT.set(limit)
    try:
        yield
    finally:
        _LIMIT.reset(token)


class CampaignQueuePool(AsyncAdaptedQueuePool):
    """Release permits on checkin, including rollback, disconnect and disposal."""

    def __init__(self, *args, **kwargs) -> None:
        self._work_limit = _LIMIT.get()
        super().__init__(*args, **kwargs)
        if _mark_checkin not in self.dispatch.checkin:
            event.listen(self, "checkin", _mark_checkin)
        if _deny_budgeted_detach not in self.dispatch.detach:
            event.listen(self, "detach", _deny_budgeted_detach)

    def _do_get(self) -> ConnectionPoolEntry:
        # Engines constructed before the worker scope bind each checkout, not the pool.
        limit = self._work_limit or _LIMIT.get()
        if limit is None or _GETTING.get() is self:
            return super()._do_get()
        try:
            await_only(asyncio.wait_for(limit.acquire(), timeout=self._timeout))
        except TimeoutError as exc:
            raise PoolTimeoutError("campaign_database_work_limit_timeout") from exc
        token = _GETTING.set(self)
        try:
            record = super()._do_get()
            info = record.record_info
            if info is None:
                raise RuntimeError("campaign_database_pool_record_missing")
            info["step8_work_limit"] = _CheckoutBudget(limit, self._dialect.do_terminate)
            return record
        except BaseException:
            limit.release()
            raise
        finally:
            _GETTING.reset(token)

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        info = record.record_info
        budget = info.get("step8_work_limit") if info is not None else None
        returning = isinstance(budget, _CheckoutBudget) and budget.checking_in
        if returning and info is not None:
            info.pop("step8_work_limit")
        try:
            super()._do_return_conn(record)
        finally:
            # detach returns the empty record before its event closes the live connection.
            if returning and isinstance(budget, _CheckoutBudget):
                budget.limit.release()

    def recreate(self) -> CampaignQueuePool:
        pool = super().recreate()
        assert isinstance(pool, CampaignQueuePool)
        pool._work_limit = self._work_limit
        return pool


def _mark_checkin(connection: DBAPIConnection | None, record: ConnectionPoolEntry) -> None:
    info = record.record_info
    budget = info.get("step8_work_limit") if info is not None else None
    if isinstance(budget, _CheckoutBudget):
        budget.checking_in = True


def _deny_budgeted_detach(connection: DBAPIConnection, record: ConnectionPoolEntry) -> None:
    info = record.record_info
    budget = info.pop("step8_work_limit", None) if info is not None else None
    if isinstance(budget, _CheckoutBudget):
        # A failed termination must retain the permit: the detached handle may be usable.
        budget.terminate(connection)
        budget.limit.release()
        raise InvalidRequestError("campaign_database_detach_forbidden")
