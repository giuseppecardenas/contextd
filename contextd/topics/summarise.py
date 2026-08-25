"""LLM title + summary for one topic cluster (``summary`` call-site)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from contextd.chunking.llm import render_template
from contextd.inference._json_body import loads_json_body
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import InferenceProvider, PromptRequest

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicText:
    title: str
    summary: str


def summarise_topic(
    provider: InferenceProvider,
    renderer: PromptRenderer,
    member_summaries: list[str],
    *,
    corpus: str,
    max_words: int = 100,
) -> TopicText | None:
    """``None`` on provider failure or a malformed response (the caller skips the topic)."""
    if not member_summaries:
        return None
    prompt = render_template(
        renderer,
        "topic",
        corpus_name=corpus,
        member_count=str(len(member_summaries)),
        member_summaries="\n".join(f"- {s}" for s in member_summaries),
        max_words=str(max_words),
    )
    try:
        response = provider.generate(PromptRequest(system="", prompt=prompt, call_site="summary"))
        data = loads_json_body(response)
    except Exception as exc:
        _log.warning("topics: summarise failed (%s: %s)", type(exc).__name__, exc)
        return None
    title = data.get("title")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        _log.warning("topics: response carried no summary")
        return None
    return TopicText(
        title=title.strip() if isinstance(title, str) and title.strip() else summary[:60],
        summary=summary.strip(),
    )
