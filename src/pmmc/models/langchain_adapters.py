"""Thin LangChain adapters over PMMC's provider-neutral gateway contracts."""

from __future__ import annotations

from uuid import uuid4

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from pmmc.models.protocols import EmbeddingGateway, GenerationGateway
from pmmc.schemas import (
    EmbeddingInput,
    EmbeddingRequest,
    GenerationRequest,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    SpeakerRole,
)


def _role(message: BaseMessage) -> SpeakerRole:
    return {
        "human": SpeakerRole.USER,
        "user": SpeakerRole.USER,
        "ai": SpeakerRole.ASSISTANT,
        "assistant": SpeakerRole.ASSISTANT,
        "system": SpeakerRole.SYSTEM,
    }.get(message.type, SpeakerRole.USER)


def as_langchain_chat_runnable(
    gateway: GenerationGateway,
    *,
    model_alias: str,
    max_output_tokens: int = 1024,
) -> RunnableLambda:
    """Expose a gateway as an async LangChain runnable without changing PMMC contracts."""

    async def invoke(messages: list[BaseMessage]) -> str:
        request = GenerationRequest(
            request_id=f"langchain-{uuid4().hex}",
            model_alias=model_alias,
            max_output_tokens=max_output_tokens,
            messages=[
                ModelMessage(
                    role=_role(message),
                    parts=[
                        ModelContentPart(
                            kind=ModelContentKind.TEXT,
                            text=str(message.content),
                        )
                    ],
                )
                for message in messages
            ],
        )
        return (await gateway.generate(request)).text

    return RunnableLambda(invoke)


class PMMCLangChainEmbeddings(Embeddings):
    """Async-first LangChain embedding adapter for later LangGraph nodes."""

    def __init__(self, gateway: EmbeddingGateway, model_alias: str) -> None:
        self.gateway = gateway
        self.model_alias = model_alias

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        request = EmbeddingRequest(
            request_id=f"langchain-{uuid4().hex}",
            model_alias=self.model_alias,
            inputs=[
                EmbeddingInput(input_id=str(index), text=text)
                for index, text in enumerate(texts)
            ],
        )
        result = await self.gateway.embed(request)
        return [item.vector for item in result.vectors]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("PMMCLangChainEmbeddings is async-only; use aembed_documents")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("PMMCLangChainEmbeddings is async-only; use aembed_query")
