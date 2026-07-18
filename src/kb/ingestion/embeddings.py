"""Embedding backends behind a single :class:`Embedder` interface (spec section 3).

Two backends produce 1024-dim unit vectors:

* ``local`` — voyage-4-nano ONNX (int8) on CPU. Native 2048 dims are truncated to
  ``EMBED_DIM`` (Matryoshka) and renormalised. Loaded lazily and only imported
  when actually used, so the heavy ``onnxruntime`` dependency stays optional.
* ``openrouter`` — ``POST /embeddings`` with an explicit ``dimensions``.

Documents are embedded as-is; queries are embedded with a mandatory retrieval
prefix. Mixing the two up breaks retrieval, so the asymmetry lives here.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from kb.config import Settings, get_settings

if TYPE_CHECKING:
    from openai import AsyncOpenAI

QUERY_PREFIX = "Represent the query for retrieving supporting documents: "

_LOCAL_BATCH = 32
_OPENROUTER_BATCH = 100
_VOYAGE_NATIVE_DIM = 2048


def apply_query_prefix(text: str) -> str:
    """Prepend the mandatory retrieval prefix to a query string."""
    return f"{QUERY_PREFIX}{text}"


def truncate_and_renormalize(vector: Sequence[float], dim: int) -> list[float]:
    """Matryoshka truncation to ``dim`` dims followed by L2 renormalisation.

    A zero vector is returned unchanged (its norm is zero).
    """
    arr = np.asarray(vector, dtype=np.float32)[:dim]
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return [float(x) for x in arr]
    return [float(x) for x in arr / norm]


@runtime_checkable
class Embedder(Protocol):
    """Async embedding backend producing ``dim``-length unit vectors."""

    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents as-is (no query prefix)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query, applying the retrieval prefix."""
        ...


def _batched(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


class OpenRouterEmbedder:
    """Cloud embeddings via OpenRouter's ``/embeddings`` endpoint.

    The requested ``dimensions`` must match the schema. Because different cloud
    models support different dimension sets, a deployment smoke-test of the
    dimension is mandatory (see spec section 3).
    """

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self.dim = settings.embed_dim
        self._model = settings.embed_model
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openrouter_base_url,
            )
        return self._client

    async def _embed(self, inputs: Sequence[str]) -> list[list[float]]:
        client = self._get_client()
        vectors: list[list[float]] = []
        for batch in _batched(inputs, _OPENROUTER_BATCH):
            response = await client.embeddings.create(
                model=self._model, input=batch, dimensions=self.dim
            )
            vectors.extend([list(item.embedding) for item in response.data])
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([apply_query_prefix(text)])
        return vectors[0]


class LocalVoyageEmbedder:
    """voyage-4-nano ONNX (int8) on CPU, lazily loaded.

    The model outputs 2048 dims; we keep the first ``EMBED_DIM`` (Matryoshka) and
    renormalise. ``onnxruntime`` and ``tokenizers`` are imported only on first use
    so they can stay optional dependencies.
    """

    def __init__(self, settings: Settings) -> None:
        self.dim = settings.embed_dim
        self._model_dir = Path(settings.embed_model_path)
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Local embedding backend requires the 'embed-local' extra "
                    "(onnxruntime, tokenizers). Install it or set EMBED_BACKEND=openrouter."
                ) from exc

            model_file = self._model_dir / "model.onnx"
            tokenizer_file = self._model_dir / "tokenizer.json"
            if not model_file.exists() or not tokenizer_file.exists():
                raise RuntimeError(
                    f"voyage-4-nano ONNX assets not found under {self._model_dir} "
                    "(expected model.onnx and tokenizer.json)."
                )
            self._session = ort.InferenceSession(
                str(model_file), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(str(tokenizer_file))

    def _forward(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        self._ensure_loaded()
        assert self._session is not None
        assert self._tokenizer is not None
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        outputs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        hidden = np.asarray(outputs[0], dtype=np.float32)
        if hidden.ndim == 3:
            mask = attention_mask[:, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        else:
            pooled = hidden
        return [truncate_and_renormalize(vec, self.dim) for vec in pooled]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in _batched(texts, _LOCAL_BATCH):
            vectors.extend(self._forward(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return self._forward([apply_query_prefix(text)])[0]


def make_embedder(settings: Settings | None = None) -> Embedder:
    """Return the configured embedding backend."""
    settings = settings or get_settings()
    if settings.embed_backend == "openrouter":
        return OpenRouterEmbedder(settings)
    return LocalVoyageEmbedder(settings)
