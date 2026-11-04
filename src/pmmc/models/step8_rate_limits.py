"""Shared provider admission, separate from spend and request concurrency limits."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal, Protocol

from pydantic import Field

from pmmc.compiler.ids import canonical_content_hash
from pmmc.schemas.base import BaseSchema


class ProviderRateLimit(BaseSchema):
    provider: Literal["openai", "anthropic"]
    account_scope: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    model_scope: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.*-]+$")
    requests_per_minute: int = Field(gt=0, strict=True)
    tokens_per_minute: int = Field(gt=0, strict=True)

    @property
    def scope_hash(self) -> str:
        # Limits can change only by explicit operator reconfiguration. They are
        # not in the scope key, so changing a value cannot bypass an active window.
        return canonical_content_hash([self.provider, self.account_scope, self.model_scope])


@dataclass(frozen=True)
class RateWindowEntry:
    admitted_at: datetime
    token_units: int


def admission_delay(
    limit: ProviderRateLimit,
    *,
    token_units: int,
    now: datetime,
    entries: Sequence[RateWindowEntry],
    blocked_until: datetime | None = None,
) -> float:
    """Sliding 60-second window using DB time and conservative token units."""
    if type(token_units) is not int or not 0 < token_units <= limit.tokens_per_minute:
        raise ValueError("request token reservation exceeds provider TPM capacity")
    active = sorted(
        (item for item in entries if (now - item.admitted_at).total_seconds() < 60),
        key=lambda item: item.admitted_at,
    )
    delay = max(0.0, (blocked_until - now).total_seconds()) if blocked_until else 0.0
    request_count = len(active)
    tokens = sum(item.token_units for item in active)
    for item in active:
        if (
            request_count < limit.requests_per_minute
            and tokens + token_units <= limit.tokens_per_minute
        ):
            break
        delay = max(delay, 60 - (now - item.admitted_at).total_seconds())
        request_count -= 1
        tokens -= item.token_units
    return delay


class ProviderRateRepository(Protocol):
    async def try_admit(
        self,
        limit: ProviderRateLimit,
        *,
        dispatch_id: str,
        token_units: int,
        request_hash: str | None = None,
    ) -> float: ...

    async def defer(self, limit: ProviderRateLimit, *, seconds: float) -> None: ...


class ProviderRateAdmission:
    def __init__(
        self,
        repository: ProviderRateRepository,
        limit: ProviderRateLimit,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.repository, self.limit, self.sleep = repository, limit, sleep

    async def acquire(
        self, *, dispatch_id: str, token_units: int, request_hash: str | None = None
    ) -> float:
        waited = 0.0
        while True:
            delay = await self.repository.try_admit(
                self.limit,
                dispatch_id=dispatch_id,
                token_units=token_units,
                request_hash=request_hash,
            )
            if delay <= 0:
                return waited
            # No transaction is held during this wait. Short wakeups preserve
            # cancellation and allow a completed window to be re-evaluated.
            duration = min(30.0, max(0.01, delay))
            await self.sleep(duration)
            waited += duration

    async def retry_after(self, error: BaseException) -> float | None:
        seconds = explicit_retry_after(error)
        if seconds is not None:
            await self.repository.defer(self.limit, seconds=seconds)
        return seconds


def explicit_retry_after(error: BaseException, *, now: datetime | None = None) -> float | None:
    """Read only an explicit 429 header; never infer nonexecution from timeouts."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(current, "status_code", None) or getattr(response, "status_code", None)
        if status == 429 and response is not None:
            value = response.headers.get("retry-after")
            if not isinstance(value, str):
                return None
            try:
                seconds = float(value)
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        return None
                    seconds = (date - (now or datetime.now(UTC))).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    return None
            return max(0.0, seconds) if math.isfinite(seconds) else None
        current = current.__cause__
    return None
