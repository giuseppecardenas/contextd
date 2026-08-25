"""LLM-backed chunk helpers: contextual prefixes, questions, propositions.

All three follow the same shape — one ``summary``-call-site request per
*parent unit* that returns a JSON array aligned with the chunks — because
one call per parent is roughly an order of magnitude cheaper than one per
chunk and lets the model see the chunks in context. Every helper degrades
to a documented fallback on provider failure or a malformed response; a
provider miss must never abort a bootstrap (CLAUDE.md error-boundary rule).

Templates are resolved through the caller's renderer (the user's
``~/.contextd/prompts/``) with a fallback to the packaged copy, so an
install whose prompt directory predates these templates keeps working
until ``contextd init --refresh-prompts`` is run.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Any

from contextd.chunking.model import Chunk
from contextd.inference._json_body import loads_json_body
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import InferenceProvider, PromptRequest

_log = logging.getLogger(__name__)


def render_template(renderer: PromptRenderer, template: str, **vars: str) -> str:
    """Render from the user's prompt dir, falling back to the packaged template."""
    try:
        return renderer.render(template, **vars)
    except FileNotFoundError:
        packaged = Path(str(resources.files("contextd.prompts")))
        return PromptRenderer(packaged).render(template, **vars)


def _numbered(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}]\n{c.text.strip()}" for i, c in enumerate(chunks))


def _as_str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if isinstance(x, str | int | float) and str(x).strip()]


def _aligned_list(data: dict[str, Any], key: str, n: int) -> list[Any] | None:
    """The ``key`` array when it has exactly ``n`` entries, else ``None``."""
    raw = data.get(key)
    if isinstance(raw, list) and len(raw) == n:
        return raw
    return None


def contextualise(
    provider: InferenceProvider,
    renderer: PromptRenderer,
    chunks: list[Chunk],
    *,
    breadcrumb: str,
    document_summary: str,
) -> list[str]:
    """Anthropic-style situating context per chunk (50-100 tokens each).

    Returns one string per chunk; entries fall back to ``breadcrumb`` when
    the call fails or the response is not aligned with the chunks.
    """
    if not chunks:
        return []
    fallback = [breadcrumb] * len(chunks)
    prompt = render_template(
        renderer,
        "contextualise",
        breadcrumb=breadcrumb,
        document_summary=document_summary or "(no summary available)",
        chunks=_numbered(chunks),
        count=str(len(chunks)),
    )
    try:
        response = provider.generate(PromptRequest(system="", prompt=prompt, call_site="summary"))
        data = loads_json_body(response)
    except Exception as exc:
        _log.warning(
            "chunking: contextualise failed (%s: %s); using breadcrumb", type(exc).__name__, exc
        )
        return fallback
    contexts = _aligned_list(data, "contexts", len(chunks))
    if contexts is None:
        _log.warning(
            "chunking: contextualise returned %s contexts for %d chunks; using breadcrumb",
            len(data.get("contexts", [])) if isinstance(data.get("contexts"), list) else "no",
            len(chunks),
        )
        return fallback
    out: list[str] = []
    for ctx in contexts:
        text = str(ctx).strip() if isinstance(ctx, str | int | float) else ""
        out.append(f"{breadcrumb}\n{text}" if text else breadcrumb)
    return out


def generate_questions(
    provider: InferenceProvider,
    renderer: PromptRenderer,
    chunks: list[Chunk],
    *,
    breadcrumb: str,
    per_chunk: int = 2,
) -> list[list[str]]:
    """RAGFlow-style auto-questions per chunk for the full-text field.

    Returns one list per chunk (empty lists on failure).
    """
    if not chunks:
        return []
    empty: list[list[str]] = [[] for _ in chunks]
    prompt = render_template(
        renderer,
        "chunk_questions",
        breadcrumb=breadcrumb,
        chunks=_numbered(chunks),
        count=str(len(chunks)),
        per_chunk=str(per_chunk),
    )
    try:
        response = provider.generate(PromptRequest(system="", prompt=prompt, call_site="summary"))
        data = loads_json_body(response)
    except Exception as exc:
        _log.warning("chunking: question generation failed (%s: %s)", type(exc).__name__, exc)
        return empty
    questions = _aligned_list(data, "questions", len(chunks))
    if questions is None:
        _log.warning("chunking: question list not aligned with %d chunks; skipping", len(chunks))
        return empty
    return [_as_str_list(q)[:per_chunk] for q in questions]


def propositions(
    provider: InferenceProvider,
    renderer: PromptRenderer,
    text: str,
    *,
    breadcrumb: str,
    max_propositions: int,
) -> list[str] | None:
    """Dense-X style self-contained statements for one parent unit.

    Returns ``None`` when the provider fails or emits no usable list so the
    caller can fall back to a structural split instead of indexing nothing.
    """
    if not text.strip():
        return []
    prompt = render_template(
        renderer,
        "propositions",
        breadcrumb=breadcrumb,
        content=text,
        max_propositions=str(max_propositions),
    )
    try:
        response = provider.generate(PromptRequest(system="", prompt=prompt, call_site="summary"))
        data = loads_json_body(response)
    except Exception as exc:
        _log.warning("chunking: propositions failed (%s: %s)", type(exc).__name__, exc)
        return None
    props = _as_str_list(data.get("propositions"))
    if not props:
        _log.warning("chunking: propositions response carried no statements")
        return None
    return props[:max_propositions]
