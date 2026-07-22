"""Reusable test doubles: a fake LLM client and a deterministic embedder."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np


class _Function:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content: str | None, tool_calls: list[_ToolCall] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message) -> None:
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class FakeResponse:
    """Duck-typed stand-in for an OpenAI ChatCompletion."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[_ToolCall] | None = None,
        model: str = "fake/model",
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.choices = [_Choice(_Message(content, tool_calls))]
        self.model = model
        self._usage = usage or {}

    def model_dump(self) -> dict[str, Any]:
        return {"usage": self._usage}


def content_response(text: str, **kwargs: Any) -> FakeResponse:
    return FakeResponse(content=text, **kwargs)


def tool_response(name: str, arguments: str, **kwargs: Any) -> FakeResponse:
    return FakeResponse(tool_calls=[_ToolCall("call_1", name, arguments)], **kwargs)


class FakeLLM:
    """Duck-typed LLMClient returning a fixed sequence of responses/exceptions.

    The last response is repeated once the queue is exhausted.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for integration tests."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            vector[hash(token) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return [0.0] * self.dim
        return [float(value) for value in vector / norm]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        from kb.embeddings import apply_query_prefix

        return self._vector(apply_query_prefix(text))


def extraction_payload(
    doc_type: str = "generic",
    *,
    confidence: float = 0.95,
    attributes: dict[str, Any] | None = None,
    entities: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    links: list[str] | None = None,
) -> dict[str, Any]:
    """Build an extraction envelope for the fake LLM."""
    return {
        "doc_type": doc_type,
        "confidence": confidence,
        "attributes": attributes or {},
        "entities": entities or [],
        "relations": relations or [],
        "links": links or [],
    }


def extraction_llm(payload: dict[str, Any]) -> FakeLLM:
    """Return a fake LLM whose single response is the given extraction envelope."""
    return FakeLLM([content_response(json.dumps(payload))])
