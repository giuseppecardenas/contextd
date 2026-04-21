"""Per-file (or per-section) summariser that ties provider + prompt + parser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from contextd.inference._json_body import extract_json_body
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import InferenceProvider, PromptRequest


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
        prompt_path: Path | None = None,
    ) -> None:
        self._provider = provider
        self._renderer = renderer
        self._max_words = max_words
        self._prompt_path = prompt_path

    def summarise(self, content: str) -> FileSummary:
        if self._prompt_path is not None:
            prompt = self._renderer.render_path(
                self._prompt_path,
                content=content,
                max_words=str(self._max_words),
            )
        else:
            prompt = self._renderer.render(
                "summarise",
                content=content,
                max_words=str(self._max_words),
            )
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="summary")
        )
        data = cast(dict[str, Any], json.loads(extract_json_body(response)))
        if "summary" not in data:
            raise KeyError(f"Provider response missing 'summary'; got keys {list(data.keys())}")
        summary = data["summary"]
        if not isinstance(summary, str):
            raise TypeError(
                f"Provider response 'summary' must be a string; got {type(summary).__name__}"
            )
        return FileSummary(
            summary=summary,
            key_points=_as_str_list(data.get("key_points")),
            entities_mentioned=_as_str_list(data.get("entities_mentioned")),
        )
