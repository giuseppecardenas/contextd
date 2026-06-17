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
from collections.abc import Callable, Iterator
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

# Voyage rejects any single embedding request whose inputs sum to more than
# 120,000 tokens. The budget is held below that hard ceiling to leave margin
# for tokenizer/server truncation discrepancies.
_BATCH_TOKEN_BUDGET = 100_000


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

    def _count_batch_tokens(self, texts: list[str]) -> int:
        return int(self._client.count_tokens(texts, model=self._cfg.model))

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        total_tokens = 0
        for batch in _token_aware_batches(
            texts,
            max_items=self._cfg.max_batch_size,
            token_budget=_BATCH_TOKEN_BUDGET,
            count_tokens=self._count_batch_tokens,
        ):
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
        # Voyage rejects empty strings; substitute a single space so callers
        # relying on one-vector-per-input alignment (the strict zip in
        # phase_enumerate) still receive a vector for empty or blank files.
        safe_batch = [text if text.strip() else " " for text in batch]
        attempt = 0
        while True:
            try:
                return self._client.embed(safe_batch, model=self._cfg.model, input_type="document")
            except RateLimitError:
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                delay = min(self._backoff_initial * (2 ** (attempt - 1)), self._backoff_max)
                delay *= 1.0 + random.uniform(-0.2, 0.2)
                time.sleep(delay)


def _token_aware_batches(
    texts: list[str],
    *,
    max_items: int,
    token_budget: int,
    count_tokens: Callable[[list[str]], int],
) -> Iterator[list[str]]:
    """Yield batches bounded by both item count and a per-batch token budget.

    Batching by item count alone overflows Voyage's 120,000-token-per-request
    ceiling on corpora containing large files. Each text's token count is
    accumulated and a new batch is started before ``token_budget`` would be
    exceeded, while ``max_items`` still caps the number of texts per batch. A
    single text larger than the whole budget is emitted as its own batch and
    left to Voyage's server-side per-input truncation.
    """
    batch: list[str] = []
    batch_tokens = 0
    for text in texts:
        text_tokens = count_tokens([text])
        if batch and (len(batch) >= max_items or batch_tokens + text_tokens > token_budget):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += text_tokens
    if batch:
        yield batch
