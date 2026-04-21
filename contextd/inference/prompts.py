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
        # Path-traversal guard: template="../../etc/passwd" would escape
        # the configured template_dir. Single-user threat model keeps this
        # low-risk, but a one-line `is_relative_to` check closes the footgun.
        template_path = (self._dir / f"{template}.md").resolve()
        if not template_path.is_relative_to(self._dir.resolve()):
            raise ValueError(
                f"Template name {template!r} escapes template_dir {self._dir}; "
                "templates must resolve inside the configured directory."
            )
        template_text = template_path.read_text()

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in kwargs:
                raise KeyError(f"Template {template!r}: missing variable {key!r}")
            return str(kwargs[key])

        return _PLACEHOLDER.sub(_sub, template_text)
