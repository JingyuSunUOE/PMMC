"""Deterministic model gateway used by tests and offline development."""

from __future__ import annotations

import hashlib
import json
import math
import time

from pmmc.models.tokenization import HeuristicTokenCounter
from pmmc.schemas import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingVector,
    GenerationRequest,
    GenerationResult,
    ModelContentKind,
    TokenUsage,
)


class DeterministicFakeGateway:
    """Implements generation and embedding without network or random state."""

    def __init__(self, dimension: int = 16) -> None:
        if dimension <= 0:
            raise ValueError("fake embedding dimension must be positive")
        self.dimension = dimension
        self.counter = HeuristicTokenCounter("pmmc/fake-tokenizer", "v1")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        text = "\n".join(
            part.text or ""
            for message in request.messages
            for part in message.parts
            if part.kind is ModelContentKind.TEXT
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if request.metadata.get("task") == "baseline_judge":
            output = json.dumps(
                {
                    "verdict": "incorrect",
                    "score": 0,
                    "reason": "Deterministic fake judge verdict.",
                },
                sort_keys=True,
            )
        elif request.metadata.get("task") == "pmmc_questioner":
            output = json.dumps({"candidates": []}, sort_keys=True)
        else:
            output = f"fake:{digest}"
        input_tokens = self.counter.count_request(request)
        output_tokens = self.counter.count_text(output)
        return GenerationResult(
            request_id=request.request_id,
            model_alias=request.model_alias,
            actual_model_id="deterministic-fake@v1",
            text=output,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            provider_request_id=f"fake-{digest}",
            finish_reason="stop",
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response={"id": f"fake-{digest}", "model": "deterministic-fake@v1"},
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        started = time.perf_counter()
        vectors: list[EmbeddingVector] = []
        total_tokens = 0
        for item in request.inputs:
            payload = (item.text or "") + "|" + "|".join(
                image.image_id for image in item.image_refs
            )
            digest = hashlib.sha256(payload.encode("utf-8")).digest()
            values = [
                ((digest[index % len(digest)] / 255.0) * 2.0) - 1.0
                for index in range(self.dimension)
            ]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append(
                EmbeddingVector(
                    input_id=item.input_id,
                    vector=[value / norm for value in values],
                )
            )
            total_tokens += self.counter.count_text(item.text or "")
        return EmbeddingResult(
            request_id=request.request_id,
            model_alias=request.model_alias,
            actual_model_id="deterministic-fake@v1",
            vectors=vectors,
            usage=TokenUsage(input_tokens=total_tokens, total_tokens=total_tokens),
            provider_request_id=f"fake-{request.request_id}",
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response={"model": "deterministic-fake@v1", "count": len(vectors)},
        )
