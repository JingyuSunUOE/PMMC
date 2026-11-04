"""Real, content-minimal probes for services started by the managed harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

from PIL import Image

from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import create_embedding_gateway, create_generation_gateway
from pmmc.models.registry import ModelRegistry, load_model_registry
from pmmc.schemas import (
    EmbeddingInput,
    EmbeddingRequest,
    GenerationRequest,
    ImageRef,
    ModelCapability,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    SpeakerRole,
)


def _build_probe_png() -> bytes:
    """Build a deterministic, processor-safe RGB image without external assets."""

    image = Image.new("RGB", (64, 64), (35, 91, 155))
    image.paste((230, 159, 39), (32, 0, 64, 64))
    image.paste((240, 240, 240), (16, 16, 48, 48))
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


_PROBE_PNG = _build_probe_png()


@contextmanager
def _probe_image_url() -> Iterator[str]:
    """Serve one deterministic image on Pod loopback for an end-to-end probe."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/probe.png":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_PROBE_PNG)))
            self.end_headers()
            self.wfile.write(_PROBE_PNG)

        def log_message(self, _format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        yield f"http://127.0.0.1:{port}/probe.png"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def probe_managed_service(
    settings: Settings,
    registry: ModelRegistry,
    alias: str,
) -> dict[str, object]:
    """Execute one real request and validate model identity and vector dimensions."""

    entry = registry.get(alias)
    if ModelCapability.CHAT_TEXT in entry.capabilities:
        result = await create_generation_gateway(settings, registry, alias).generate(
            GenerationRequest(
                request_id=f"managed-runtime-probe-{alias}",
                model_alias=alias,
                messages=[
                    ModelMessage(
                        role=SpeakerRole.USER,
                        parts=[
                            ModelContentPart(
                                kind=ModelContentKind.TEXT,
                                text="Reply with OK.",
                            )
                        ],
                    )
                ],
                max_output_tokens=8,
                temperature=0,
                top_p=1,
                seed=0,
                timeout_seconds=settings.model_request_timeout_seconds,
            )
        )
        if not result.text.strip():
            raise RuntimeError("managed chat probe returned an empty response")
        registry.verify_actual_model(
            alias,
            result.actual_model_id,
            paper_mode=settings.mode.value == "paper",
        )
        return {
            "alias": alias,
            "kind": "generation",
            "actual_model_id": result.actual_model_id,
            "usage": result.usage.to_dict(),
        }

    gateway = create_embedding_gateway(settings, registry, alias)
    expected_vector_count = 1
    if (
        ModelCapability.EMBED_MULTIMODAL in entry.capabilities
        and entry.endpoint_kind.value != "fake"
    ):
        with _probe_image_url() as image_url:
            probe_input = EmbeddingInput(
                input_id="probe",
                text="PMMC managed multimodal runtime probe",
                image_refs=[
                    ImageRef(
                        image_id="managed-runtime-probe-image",
                        remote_url=image_url,
                        mime_type="image/png",
                        width=64,
                        height=64,
                    )
                ],
            )
            result = await gateway.embed(
                EmbeddingRequest(
                    request_id=f"managed-runtime-probe-{alias}",
                    model_alias=alias,
                    inputs=[probe_input],
                    timeout_seconds=settings.model_request_timeout_seconds,
                )
            )
        probe_modality = "multimodal"
    elif (
        ModelCapability.EMBED_TEXT in entry.capabilities
        and ModelCapability.EMBED_IMAGE in entry.capabilities
        and entry.endpoint_kind.value != "fake"
    ):
        with _probe_image_url() as image_url:
            result = await gateway.embed(
                EmbeddingRequest(
                    request_id=f"managed-runtime-probe-{alias}",
                    model_alias=alias,
                    inputs=[
                        EmbeddingInput(
                            input_id="probe-text",
                            text="PMMC managed contrastive text probe",
                        ),
                        EmbeddingInput(
                            input_id="probe-image",
                            image_refs=[
                                ImageRef(
                                    image_id="managed-runtime-probe-image",
                                    remote_url=image_url,
                                    mime_type="image/png",
                                    width=64,
                                    height=64,
                                )
                            ],
                        ),
                    ],
                    timeout_seconds=settings.model_request_timeout_seconds,
                )
            )
        expected_vector_count = 2
        probe_modality = "text_and_image"
    else:
        probe_input = EmbeddingInput(
            input_id="probe",
            text="PMMC managed runtime probe",
        )
        probe_modality = "text"
        result = await gateway.embed(
            EmbeddingRequest(
                request_id=f"managed-runtime-probe-{alias}",
                model_alias=alias,
                inputs=[probe_input],
                timeout_seconds=settings.model_request_timeout_seconds,
            )
        )
    if len(result.vectors) != expected_vector_count:
        raise RuntimeError("managed embedding probe returned an unexpected vector count")
    dimensions = {len(vector.vector) for vector in result.vectors}
    if len(dimensions) != 1:
        raise RuntimeError("managed embedding probe returned inconsistent dimensions")
    dimension = dimensions.pop()
    if entry.embedding_dimension is not None and dimension != entry.embedding_dimension:
        raise RuntimeError(
            f"managed embedding dimension mismatch: expected {entry.embedding_dimension}, "
            f"received {dimension}"
        )
    registry.verify_actual_model(
        alias,
        result.actual_model_id,
        paper_mode=settings.mode.value == "paper",
    )
    return {
        "alias": alias,
        "kind": "embedding",
        "actual_model_id": result.actual_model_id,
        "dimension": dimension,
        "modality": probe_modality,
        "vector_count": expected_vector_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe one PMMC managed model service")
    parser.add_argument("--alias", required=True)
    arguments = parser.parse_args()
    settings = Settings()
    registry = load_model_registry(settings.resolved_model_registry_path)
    result = asyncio.run(probe_managed_service(settings, registry, arguments.alias))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
