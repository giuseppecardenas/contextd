"""``propositions`` — Dense-X style atomic statements, one chunk each.

One ``summary``-call-site LLM call per parent unit yields up to
``max_propositions`` self-contained statements. Each becomes a chunk of
``kind="proposition"`` whose span is the whole parent (a proposition has no
exact source line). When the provider fails or returns nothing, the parent
falls back to the ``structural`` split so it is still indexed.
"""

from __future__ import annotations

import logging

from contextd.chunking.llm import propositions as _propositions
from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.prefix import breadcrumb_text
from contextd.chunking.strategies.structural import StructuralStrategy
from contextd.chunking.tokenizer import Tokenizer
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import InferenceProvider

_log = logging.getLogger(__name__)


class PropositionsStrategy:
    name = "propositions"

    def __init__(
        self, tokenizer: Tokenizer, provider: InferenceProvider, renderer: PromptRenderer
    ) -> None:
        self._tok = tokenizer
        self._provider = provider
        self._renderer = renderer
        self._fallback = StructuralStrategy(tokenizer)

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        if not req.text.strip():
            return []
        statements = _propositions(
            self._provider,
            self._renderer,
            req.text,
            breadcrumb=breadcrumb_text(req.breadcrumb, ""),
            max_propositions=req.profile.max_propositions,
        )
        if statements is None:
            _log.warning("chunking: propositions unavailable; falling back to structural")
            return self._fallback.chunk(req)
        span = LineIndex(req.text).span(0, len(req.text), base_line=req.base_line)
        return [
            Chunk(
                ordinal=i,
                text=s,
                span=span,
                token_count=self._tok.count(s),
                kind="proposition",
                part=i + 1,
            )
            for i, s in enumerate(statements)
        ]
