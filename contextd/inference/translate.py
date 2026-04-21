"""Natural-language → Cypher translation via Gemini."""

from __future__ import annotations

import re

from contextd.inference.prompts import PromptRenderer
from contextd.mcp.readonly_guard import assert_read_only
from contextd.ontology.schema import Ontology
from contextd.providers.base import InferenceProvider, PromptRequest

_CYPHER_FENCE = re.compile(r"```(?:cypher)?\s*(.*?)\s*```", re.DOTALL)


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
        match = _CYPHER_FENCE.search(text)
        if match:
            return match.group(1).strip()
        # Strip prose lines — keep only lines that start with a Cypher keyword.
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
        lines = [
            line
            for line in text.splitlines()
            if line.strip() and line.strip().split()[0].upper() in keywords
        ]
        cypher = " ".join(lines)
        if not cypher:
            raise ValueError(
                "Translator returned no Cypher-like content; "
                "provider response was empty or contained no recognised keywords"
            )
        return cypher
