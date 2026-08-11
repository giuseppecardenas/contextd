"""Merge accumulated entity-description fragments into one synthesis.

The Microsoft-GraphRAG / LightRAG wheel: entity content gets rich by
*accumulating* description fragments across mentions and summarising the
accumulation — cheaper and more robust than demanding perfect properties at
first mention. Fragments are gathered at edge-write time (no LLM); this
merger runs in batch at phase level for entities whose fragment list crossed
the threshold.

The prompt is an inline constant, deliberately NOT a ``~/.contextd/prompts``
template: the copy-once template dir is exactly the silent-drift trap this
overhaul started from, and this prompt has no per-corpus customisation story.
"""

from __future__ import annotations

from contextd.inference._json_body import loads_json_body
from contextd.providers.base import InferenceProvider, PromptRequest

_MERGE_PROMPT = """You are consolidating knowledge-graph entity descriptions.

Entity: {name} (type: {label})

The following description fragments were collected from different documents
mentioning this entity. Synthesise them into ONE description of at most 80
words that captures all distinct information; drop repetition and phrasing
variance. Output valid JSON: {{"description": string}}

Fragments:
{fragments}
"""


class DescriptionMerger:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    def merge(self, name: str, label: str, fragments: list[str]) -> str:
        prompt = _MERGE_PROMPT.format(
            name=name,
            label=label,
            fragments="\n".join(f"- {f}" for f in fragments),
        )
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="inference")
        )
        data = loads_json_body(response)
        merged = data.get("description")
        if not isinstance(merged, str) or not merged:
            raise ValueError(f"merge response missing 'description'; got keys {list(data.keys())}")
        return merged
