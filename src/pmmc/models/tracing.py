"""Optional, non-authoritative model observability sinks."""

from __future__ import annotations

import inspect
from typing import Any

import structlog

from pmmc.infrastructure.settings import Settings
from pmmc.schemas import ModelCallTrace

LOGGER = structlog.get_logger(__name__)


class NoopTraceSink:
    async def emit(self, trace: ModelCallTrace) -> None:
        del trace


class BestEffortTraceSink:
    """Catch observability failures after Postgres has accepted the trace."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    async def emit(self, trace: ModelCallTrace) -> None:
        try:
            result = self.delegate.emit(trace)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            LOGGER.warning(
                "model_observability_failed",
                model_call_id=trace.model_call_id,
                error_type=type(exc).__name__,
            )


class LangfuseTraceSink:
    """Small compatibility facade; Langfuse remains non-authoritative."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def emit(self, trace: ModelCallTrace) -> None:
        metadata = {
            key: str(value)
            for key, value in {
                "run_id": trace.run_id,
                "query_id": trace.query_id,
                "model_call_id": trace.model_call_id,
                "model_alias": trace.model_alias,
                "actual_model_id": trace.actual_model_id,
                "provider_request_id": trace.provider_request_id,
                "status": trace.status.value,
            }.items()
            if value is not None
        }
        trace_id = trace.langfuse_trace_id or trace.model_call_id
        if hasattr(self.client, "start_observation"):
            observation = self.client.start_observation(
                name=f"pmmc.{trace.stage.value}",
                as_type="event",
                trace_context={"trace_id": trace_id},
                metadata=metadata,
            )
            if hasattr(observation, "end"):
                result = observation.end()
                if inspect.isawaitable(result):
                    await result
            return
        payload = {
            "name": f"pmmc.{trace.stage.value}",
            "trace_id": trace_id,
            "metadata": metadata,
        }
        if hasattr(self.client, "create_event"):
            result = self.client.create_event(**payload)
        elif hasattr(self.client, "event"):
            result = self.client.event(**payload)
        else:
            raise RuntimeError("unsupported Langfuse client API")
        if inspect.isawaitable(result):
            await result


def create_trace_sink(settings: Settings) -> NoopTraceSink | BestEffortTraceSink:
    """Create optional Langfuse observability without exposing credentials."""

    if not settings.langfuse_enabled:
        return NoopTraceSink()
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise ValueError("Langfuse is enabled but public/secret keys are not configured")
    from langfuse import Langfuse

    kwargs: dict[str, Any] = {
        "public_key": settings.langfuse_public_key.get_secret_value(),
        "secret_key": settings.langfuse_secret_key.get_secret_value(),
    }
    base_url = settings.langfuse_base_url or settings.langfuse_host
    if base_url is not None:
        kwargs["base_url"] = base_url
    return BestEffortTraceSink(LangfuseTraceSink(Langfuse(**kwargs)))
