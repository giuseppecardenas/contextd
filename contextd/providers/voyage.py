"""Voyage AI embedding provider.

Voyage was acquired by Anthropic in 2024; it is the natural companion
to the Gemini inference stack in Contextd's v1 configuration. The
default ``voyage-4-large`` model produces 1024-dim vectors with a
32k-token context per input (``chunk_tokens`` default matches). The
older ``voyage-3`` / ``voyage-3-large`` / ``voyage-code-3`` models
(all 8k-token context, 1024-dim) remain registered for users who
override via ``[providers.voyage] model``.
"""

from __future__ import annotations

import datetime as dt
import random
import time
from collections.abc import Iterator
from typing import Any

import voyageai
from voyageai.error import RateLimitError

from contextd.config import VoyageConfig
from contextd.providers.base import EmbeddingProvider, UsageRecord

_MODEL_DIMENSIONS = {
    "voyage-4-large": 1024,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
}


class VoyageProvider(EmbeddingProvider):
    def __init__(
        self,
        config: VoyageConfig,
        *,
        api_key: str,
        backoff_initial: float = 1.0,
        backoff_max: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        self._cfg = config
        self._client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        self._last_usage: UsageRecord | None = None
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._max_retries = max_retries

    @property
    def dimensions(self) -> int:
        return _MODEL_DIMENSIONS.get(self._cfg.model, 1024)

    def last_usage(self) -> UsageRecord | None:
        return self._last_usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        total_tokens = 0
        for batch in _chunks(texts, self._cfg.max_batch_size):
            result = self._embed_batch(batch)
            all_vectors.extend(result.embeddings)
            total_tokens += result.total_tokens
        self._last_usage = UsageRecord(
            provider="voyage",
            model=self._cfg.model,
            call_site="embedding",
            input_tokens=total_tokens,
            output_tokens=0,
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
        )
        return all_vectors

    def _embed_batch(self, batch: list[str]) -> Any:
        attempt = 0
        while True:
            try:
                return self._client.embed(batch, model=self._cfg.model, input_type="document")
            except RateLimitError:
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                delay = min(self._backoff_initial * (2 ** (attempt - 1)), self._backoff_max)
                delay *= 1.0 + random.uniform(-0.2, 0.2)
                time.sleep(delay)


def _chunks(seq: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
