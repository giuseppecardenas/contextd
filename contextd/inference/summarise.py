"""Per-file (or per-section) summariser that ties provider + prompt + parser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import InferenceProvider, PromptRequest


def _extract_json_body(response: str) -> str:
    """Return the substring from the first '{' to the last '}' inclusive.

    Tolerates language-tagged code fences (```json, ```yaml, ...) and
    prose around the JSON block, which Gemini occasionally emits despite
    the prompt instructing JSON-only output.
    """
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Provider response contains no JSON object; got {response!r}")
    return response[start : end + 1]


def _as_str_list(raw: object) -> list[str]:
    """Return ``raw`` as a list of strings, or empty list if shape is wrong.

    Silent empty-list-on-bad-shape mirrors the plan's tolerant approach for
    optional fields (``key_points``, ``entities_mentioned``). The required
    ``summary`` field still raises KeyError on absence.
    """
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


@dataclass
class FileSummary:
    summary: str
    key_points: list[str]
    entities_mentioned: list[str]


class Summariser:
    def __init__(
        self,
        provider: InferenceProvider,
        renderer: PromptRenderer,
        *,
        max_words: int = 100,
    ) -> None:
        self._provider = provider
        self._renderer = renderer
        self._max_words = max_words

    def summarise(self, content: str) -> FileSummary:
        prompt = self._renderer.render(
            "summarise",
            content=content,
            max_words=str(self._max_words),
        )
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="summary")
        )
        data = cast(dict[str, Any], json.loads(_extract_json_body(response)))
        return FileSummary(
            summary=cast(str, data["summary"]),
            key_points=_as_str_list(data.get("key_points")),
            entities_mentioned=_as_str_list(data.get("entities_mentioned")),
        )
