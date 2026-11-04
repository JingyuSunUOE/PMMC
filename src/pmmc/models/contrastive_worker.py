"""Loopback-only SigLIP2 embedding worker used by the managed runtime.

This module is launched in a child process so the runtime parent never imports
Torch or initializes CUDA before GPU placement is finalized.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from io import BytesIO
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PMMC SigLIP2 loopback worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--api-key-env", default="PMMC_CONTRASTIVE_API_KEY")
    parser.add_argument("--text-max-length", default=64, type=int)
    return parser.parse_args()


def _pooled_feature_tensor(
    output: Any,
    torch_module: Any,
    *,
    expected_batch_size: int,
) -> Any:
    """Normalize Transformers v4/v5 feature helper return contracts.

    Transformers v4 returned a tensor from ``get_*_features``. Version 5
    returns ``BaseModelOutputWithPooling`` and exposes the same pooled feature
    tensor through ``pooler_output``.
    """

    feature = output if torch_module.is_tensor(output) else getattr(output, "pooler_output", None)
    if not torch_module.is_tensor(feature):
        raise TypeError("contrastive model did not return a pooled feature tensor")
    assert feature is not None
    if feature.ndim != 2:
        raise ValueError("contrastive pooled features must be a rank-2 tensor")
    if int(feature.shape[0]) != expected_batch_size:
        raise ValueError("contrastive pooled feature batch size mismatch")
    return feature


class ContrastiveWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        if args.dtype not in dtype_by_name:
            raise ValueError(f"unsupported contrastive dtype: {args.dtype}")
        self.torch = torch
        self.model_name = str(args.served_model_name)
        self.text_max_length = int(args.text_max_length)
        self.api_key = os.environ.get(str(args.api_key_env), "")
        if not self.api_key:
            raise ValueError("contrastive worker API key is missing")
        self.processor = AutoProcessor.from_pretrained(
            args.model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=dtype_by_name[args.dtype],
            local_files_only=True,
        ).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lock = asyncio.Lock()

    async def embed(self, payload: dict[str, Any]) -> list[list[float]]:
        model = str(payload.get("model", ""))
        if model != self.model_name:
            raise ValueError("requested contrastive model identity does not match")
        raw_inputs = payload.get("input")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise ValueError("contrastive input must be a non-empty list")

        outputs: list[list[float] | None] = [None] * len(raw_inputs)
        text_items: list[tuple[int, str]] = []
        image_items: list[tuple[int, str]] = []
        for index, item in enumerate(raw_inputs):
            if not isinstance(item, dict):
                raise ValueError("contrastive input items must be objects")
            kind = item.get("type")
            if kind == "text":
                text = str(item.get("text", "")).strip().lower()
                if not text:
                    raise ValueError("contrastive text cannot be empty")
                text_items.append((index, text))
            elif kind == "image_url":
                image_url = str(item.get("image_url", "")).strip()
                if not image_url.startswith(("http://", "https://")):
                    raise ValueError("contrastive images require HTTP(S) URLs")
                image_items.append((index, image_url))
            else:
                raise ValueError("unsupported contrastive input type")

        async with self.lock:
            if text_items:
                vectors = await asyncio.to_thread(
                    self._embed_texts, [text for _index, text in text_items]
                )
                for (index, _text), vector in zip(text_items, vectors, strict=True):
                    outputs[index] = vector
            if image_items:
                images = await self._load_images([url for _index, url in image_items])
                vectors = await asyncio.to_thread(self._embed_images, images)
                for (index, _url), vector in zip(image_items, vectors, strict=True):
                    outputs[index] = vector
        if any(item is None for item in outputs):
            raise RuntimeError("contrastive worker did not produce every embedding")
        return [item for item in outputs if item is not None]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        encoded = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.get_text_features(**encoded)
            vectors = _pooled_feature_tensor(
                output,
                self.torch,
                expected_batch_size=len(texts),
            )
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors.float().cpu().tolist()

    def _embed_images(self, images: list[Any]) -> list[list[float]]:
        encoded = self.processor(images=images, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.get_image_features(**encoded)
            vectors = _pooled_feature_tensor(
                output,
                self.torch,
                expected_batch_size=len(images),
            )
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors.float().cpu().tolist()

    async def _load_images(self, urls: list[str]) -> list[Any]:
        from aiohttp import ClientSession, ClientTimeout
        from PIL import Image

        timeout = ClientTimeout(total=120)
        images: list[Any] = []
        async with ClientSession(timeout=timeout) as client:
            for url in urls:
                async with client.get(url, allow_redirects=True) as response:
                    response.raise_for_status()
                    content_length = response.content_length
                    if content_length is not None and content_length > 64 * 1024 * 1024:
                        raise ValueError("contrastive image exceeds 64 MiB")
                    payload = await response.read()
                    if len(payload) > 64 * 1024 * 1024:
                        raise ValueError("contrastive image exceeds 64 MiB")
                    image = Image.open(BytesIO(payload)).convert("RGB")
                    image.load()
                    images.append(image)
        return images


def _application(worker: ContrastiveWorker) -> Any:
    from aiohttp import web

    @web.middleware
    async def authenticate(
        request: web.Request, handler: Any
    ) -> web.StreamResponse:
        if request.headers.get("Authorization") != f"Bearer {worker.api_key}":
            raise web.HTTPUnauthorized()
        return await handler(request)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": worker.model_name, "object": "model"}]}
        )

    async def version(_request: web.Request) -> web.Response:
        import transformers

        return web.json_response({"version": transformers.__version__})

    async def embeddings(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            vectors = await worker.embed(payload)
        except (ValueError, TypeError) as exc:
            return web.json_response(
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
                status=400,
            )
        return web.json_response(
            {
                "object": "list",
                "model": worker.model_name,
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
        )

    app = web.Application(middlewares=[authenticate], client_max_size=1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/version", version)
    app.router.add_post("/v1/embeddings", embeddings)
    return app


def main() -> None:
    from aiohttp import web

    args = _arguments()
    if args.host != "127.0.0.1":
        raise ValueError("contrastive worker must listen on 127.0.0.1")
    worker = ContrastiveWorker(args)
    web.run_app(
        _application(worker),
        host=args.host,
        port=args.port,
        print=None,
        access_log=None,
    )


if __name__ == "__main__":
    main()
