"""Provider abstractions for inference and embeddings.

v1 ships a single concrete of each:
- GeminiProvider (google-genai SDK)
- VoyageProvider (voyageai SDK)

``TokenEmbedder`` extends ``EmbeddingProvider`` with per-span pooling for the
``late`` chunking strategy (implemented by ``LocalHFEmbedder``).

The abstraction stays in place so a second provider can be added later
without touching the indexer, MCP, or CLI layers (spec §4.1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

CallSite = Literal["summary", "inference", "translation"]


@dataclass
class PromptRequest:
    """A single inference call bundled with its call-site metadata.

    The call_site drives model selection per spec §4.2 (summary →
    gemma-4-31b-it; inference → gemma-4-31b-it; translation →
    gemma-4-31b-it or user-overridden to gemini-pro-latest for higher
    translation quality).
    """

    system: str
    prompt: str
    call_site: CallSite


@dataclass(frozen=True)
class UsageRecord:
    """Per-call usage metrics for cost accounting (§4.2)."""

    provider: str
    model: str
    call_site: str
    input_tokens: int
    output_tokens: int
    timestamp: str


class InferenceProvider(ABC):
    @abstractmethod
    def generate(self, request: PromptRequest) -> str:
        """Return the model's text response for ``request``."""

    @abstractmethod
    def last_usage(self) -> UsageRecord | None:
        """The UsageRecord for the most recent generate() call, or None if none yet."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    @abstractmethod
    def last_usage(self) -> UsageRecord | None: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...


class TokenEmbedder(EmbeddingProvider):
    """An embedding provider that also exposes token-level vectors for
    *late chunking* (Jina): one forward pass over a whole document, then
    each chunk's vector is the mean of the token vectors whose character
    offsets fall inside the chunk's span. Every chunk vector is therefore
    conditioned on the full document context rather than on its own text
    alone.

    ``embed`` keeps working (each text embedded independently), so a
    TokenEmbedder is a drop-in :class:`EmbeddingProvider` for the query side
    and for every non-late strategy.
    """

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Largest token window one forward pass covers. Longer texts are
        processed in overlapping windows and spans are pooled across them."""

    @abstractmethod
    def embed_spans(self, text: str, spans: list[tuple[int, int]]) -> list[list[float]]:
        """One vector per ``(start, end)`` character span of ``text``.

        Vectors are mean-pooled over the tokens intersecting each span. A span
        that receives no token (whitespace-only, or narrower than a token)
        falls back to the vector of the nearest token — never a zero vector.
        """
