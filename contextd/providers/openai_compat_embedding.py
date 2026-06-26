"""OpenAI-compatible embedding provider for local model servers.

Targets the OpenAI ``/embeddings`` endpoint shape exposed by llama.cpp's
server, Ollama (``/v1/`` mode), LM Studio, vLLM, and LocalAI. Selecting this
provider together with an ``openai_compat`` inference backend lets the entire
indexing pipeline run with no cloud API calls — the fully-offline path.

Returned vectors are validated against the configured ``dimensions`` (which
must match the vector-index width in the baseline migrations, 1024) so a model
whose output width differs from the index fails fast with a clear error rather
than writing mismatched vectors that the index would silently reject or
mis-rank.
"""

from __future__ import annotations

import datetime as dt
import random
import time
from collections.abc import Iterator

import httpx

from contextd.config import OpenAICompatEmbeddingConfig
from contextd.providers.base import EmbeddingProvider, UsageRecord

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class OpenAICompatEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        config: OpenAICompatEmbeddingConfig,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        backoff_initial: float = 1.0,
        backoff_max: float = 60.0,
    ) -> None:
        self._cfg = config
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=config.request_timeout_seconds)
        self._last_usage: UsageRecord | None = None
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

    @property
    def dimensions(self) -> int:
        return self._cfg.dimensions

    def last_usage(self) -> UsageRecord | None:
        return self._last_usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        total_tokens = 0
        for batch in _batches(texts, self._cfg.max_batch_size):
            vectors, tokens = self._embed_batch(batch)
            all_vectors.extend(vectors)
            total_tokens += tokens
        if texts:
            self._last_usage = UsageRecord(
                provider="openai_compat",
                model=self._cfg.model,
                call_site="embedding",
                input_tokens=total_tokens,
                output_tokens=0,
                timestamp=dt.datetime.now(dt.UTC).isoformat(),
            )
        return all_vectors

    def _embed_batch(self, batch: list[str]) -> tuple[list[list[float]], int]:
        # Some servers reject empty strings; substitute a single space so the
        # one-vector-per-input alignment callers depend on is preserved.
        safe_batch = [text if text.strip() else " " for text in batch]
        url = f"{self._cfg.base_url.rstrip('/')}/embeddings"
        body: dict[str, object] = {"model": self._cfg.model, "input": safe_batch}
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempt = 0
        while True:
            try:
                response = self._client.post(url, json=body, headers=headers)
                if response.status_code in _RETRYABLE_CODES:
                    attempt += 1
                    if attempt >= self._cfg.max_retries:
                        response.raise_for_status()
                    self._sleep_backoff(attempt)
                    continue
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.TimeoutException):
                attempt += 1
                if attempt >= self._cfg.max_retries:
                    raise
                self._sleep_backoff(attempt)

        payload = response.json()
        data = sorted(payload["data"], key=lambda row: row.get("index", 0))
        vectors = [[float(x) for x in row["embedding"]] for row in data]
        for vector in vectors:
            if len(vector) != self._cfg.dimensions:
                raise ValueError(
                    f"Embedding model {self._cfg.model!r} returned a "
                    f"{len(vector)}-dimension vector but the configured / indexed "
                    f"dimension is {self._cfg.dimensions}. Either choose a model "
                    f"that emits {self._cfg.dimensions}-dim vectors (e.g. "
                    "mxbai-embed-large), or update providers.openai_compat_embedding."
                    "dimensions together with the vector-index DDL in the migrations."
                )
        usage = payload.get("usage") or {}
        tokens = int(usage.get("prompt_tokens", 0) or 0)
        return vectors, tokens

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self._backoff_initial * (2 ** (attempt - 1)), self._backoff_max)
        delay *= 1.0 + random.uniform(-0.2, 0.2)
        time.sleep(delay)


def _batches(texts: list[str], max_items: int) -> Iterator[list[str]]:
    """Yield successive ``max_items``-sized slices of ``texts``."""
    for start in range(0, len(texts), max_items):
        yield texts[start : start + max_items]
