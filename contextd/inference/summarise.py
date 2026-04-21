"""Per-file (or per-section) summariser that ties provider + prompt + parser."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import InferenceProvider, PromptRequest

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


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
        cleaned = _FENCE.sub("", response).strip()
        data = cast(dict[str, Any], json.loads(cleaned))
        return FileSummary(
            summary=cast(str, data["summary"]),
            key_points=list(data.get("key_points", [])),
            entities_mentioned=list(data.get("entities_mentioned", [])),
        )
