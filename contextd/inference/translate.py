"""Natural-language → Cypher translation via Gemini."""

from __future__ import annotations

import logging
import re

from contextd.inference.prompts import PromptRenderer
from contextd.mcp.readonly_guard import assert_read_only
from contextd.ontology.schema import Ontology
from contextd.providers.base import InferenceProvider, PromptRequest

logger = logging.getLogger(__name__)

# Accept fences with any language tag (```cypher, ```sql, ```gremlin, ```)
# — some LLMs mis-identify the output but the body is still Cypher. The
# tag capture is non-greedy and up to the first newline so we don't swallow
# the Cypher body.
_CYPHER_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", re.DOTALL)

# Allowlist for corpus names: alnum + underscore + dot + hyphen. Matches
# what `add-corpus --name` accepts in practice (directory basenames). This
# is the primary defence against Cypher-injection via corpus names — the
# value is embedded as a literal in the injected property map.
_CORPUS_NAME_RE = re.compile(r"^[\w.-]+$")

# Match the first labelled node pattern: (var:Label) or (var:Label {props}).
# Greedy property-map capture requires no nested braces (Cypher pattern
# property maps don't nest in practice).
_FIRST_LABELLED_NODE_RE = re.compile(
    r"\((?P<var>\w+)\s*:\s*(?P<label>[\w:]+)(?P<propmap>\s*\{[^{}]*\})?\s*\)"
)


def _inject_corpus_filter(cypher: str, corpus: str) -> str:
    """Inject ``{corpus: "<name>"}`` into the first labelled node pattern.

    Scope: first labelled node in the first MATCH only. Subsequent node
    patterns (e.g. the ``m`` in ``MATCH (n:File)-[:REFERENCES]->(m:File)``)
    are NOT filtered — callers querying cross-node results must anchor on a
    corpus-bearing node type for the filter to bite. When no labelled
    pattern is present (e.g. an unlabelled ``MATCH (n)`` or a bare ``CALL``
    procedure), the Cypher passes through unchanged and a warning is logged.
    """
    match = _FIRST_LABELLED_NODE_RE.search(cypher)
    if match is None:
        logger.warning(
            "corpus filter %r not applied: Cypher has no labelled node pattern",
            corpus,
        )
        return cypher
    propmap = match.group("propmap")
    literal = f'corpus: "{corpus}"'
    if propmap:
        # Strip surrounding braces + whitespace, append ", corpus: \"x\"".
        inner = propmap.strip()[1:-1].strip()
        new_map = "{" + inner + ", " + literal + "}"
    else:
        new_map = " {" + literal + "}"
    start, end = match.span()
    var = match.group("var")
    label = match.group("label")
    replacement = f"({var}:{label}{new_map})"
    return cypher[:start] + replacement + cypher[end:]


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
            if not _CORPUS_NAME_RE.fullmatch(corpus):
                raise ValueError(f"invalid corpus name: {corpus!r}")
            cypher = _inject_corpus_filter(cypher, corpus)
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
