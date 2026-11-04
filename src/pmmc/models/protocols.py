"""Narrow model protocols consumed by PMMC domain code."""

from __future__ import annotations

from typing import Protocol

from pmmc.schemas import (
    EmbeddingRequest,
    EmbeddingResult,
    GenerationRequest,
    GenerationResult,
    ModelCallTrace,
)


class GenerationGateway(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


class EmbeddingGateway(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...


class TokenCounter(Protocol):
    """Tokenizer/processor contract used by corpus and context assembly."""

    @property
    def alias(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def estimated(self) -> bool: ...

    def count_text(self, text: str) -> int: ...

    def count_image(self, *, width: int | None = None, height: int | None = None) -> int: ...

    def count_request(self, request: GenerationRequest) -> int: ...


class TraceSink(Protocol):
    async def emit(self, trace: ModelCallTrace) -> None: ...


class ModelCallWriter(Protocol):
    async def add(self, trace: ModelCallTrace) -> ModelCallTrace: ...
