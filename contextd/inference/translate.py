"""Natural-language → Cypher translation via Gemini."""

from __future__ import annotations

import re

from contextd.inference.prompts import PromptRenderer
from contextd.mcp.readonly_guard import assert_read_only
from contextd.ontology.schema import Ontology
from contextd.providers.base import InferenceProvider, PromptRequest

# Accept fences with any language tag (```cypher, ```sql, ```gremlin, ```)
# — some LLMs mis-identify the output but the body is still Cypher. The
# tag capture is non-greedy and up to the first newline so we don't swallow
# the Cypher body.
_CYPHER_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", re.DOTALL)


class QueryTranslator:
    def __init__(
        self,
        provider: InferenceProvider,
        renderer: PromptRenderer,
        ontology: Ontology,
    ) -> None:
        self._provider = provider
        self._renderer = renderer
        self._onto = ontology

    def translate(self, question: str, *, corpus: str | None = None) -> str:
        prompt = self._renderer.render(
            "translate",
            question=question,
            node_types=", ".join(sorted(self._onto.node_types)),
            edge_types=", ".join(sorted(self._onto.edge_types)),
        )
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="translation")
        )
        cypher = self._extract_cypher(response)
        if corpus:
            # TODO(M9/M10): inject corpus filter into first MATCH clause.
            # Plan-prescribed cypher.replace("MATCH", "MATCH ", 1) is a no-op that
            # just adds a trailing space — not a real filter. Deferred until the
            # cross-corpus routing design lands.
            pass
        assert_read_only(cypher)
        return cypher

    def _extract_cypher(self, text: str) -> str:
        """Extract Cypher from an LLM response.

        Strategy: try the ``` fenced block first (any language tag). If no
        fence, slice from the first Cypher keyword to end-of-text — this
        preserves multi-line continuation lines (``-[:REFERENCES]->`` etc.)
        that a line-by-line keyword filter would drop.
        """
        match = _CYPHER_FENCE.search(text)
        if match:
            return match.group(1).strip()
        # Fallback: find the first line starting with a Cypher read-only
        # keyword and keep everything from there. This handles both
        # single-line Cypher and multi-line blocks with continuation.
        keywords = {
            "MATCH",
            "OPTIONAL",
            "WITH",
            "WHERE",
            "UNWIND",
            "RETURN",
            "ORDER",
            "LIMIT",
            "SKIP",
            "CALL",
        }
        lines = text.splitlines()
        start: int | None = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped.split()[0].upper() in keywords:
                start = i
                break
        if start is None:
            raise ValueError(
                "Translator returned no Cypher-like content; "
                "provider response was empty or contained no recognised keywords"
            )
        cypher = "\n".join(lines[start:]).strip()
        return cypher
