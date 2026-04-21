"""Lightweight prompt-template renderer.

The renderer takes a single ``template_dir`` — the packaged ``prompts/``
directory by default. User-overridable lookup (``~/.contextd/prompts/``
with fallback to packaged) is a later milestone concern; the module
docstring deliberately does not claim what it does not do. Uses
double-brace mustache-style placeholders — no Jinja dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class PromptRenderer:
    def __init__(self, template_dir: Path) -> None:
        self._dir = template_dir

    def render(self, template: str, **kwargs: str) -> str:
        template_text = (self._dir / f"{template}.md").read_text()

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in kwargs:
                raise KeyError(f"Template {template!r}: missing variable {key!r}")
            return str(kwargs[key])

        return _PLACEHOLDER.sub(_sub, template_text)
