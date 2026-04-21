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


def _substitute(text: str, label: str, **kwargs: str) -> str:
    """Replace ``{{key}}`` placeholders in *text* using *kwargs*.

    Unknown placeholders (i.e. ``{{key}}`` present in *text* but absent from
    *kwargs*) raise :exc:`KeyError`.  Extra *kwargs* that have no placeholder
    in *text* are silently ignored — a template that doesn't reference
    ``max_words`` is valid.

    *label* is used only in the KeyError message (typically the template name
    or file path) to make failures actionable.
    """

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in kwargs:
            raise KeyError(f"Template {label!r}: missing variable {key!r}")
        return str(kwargs[key])

    return _PLACEHOLDER.sub(_sub, text)


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
        template_text = template_path.read_text(encoding="utf-8")
        return _substitute(template_text, template, **kwargs)

    def render_path(self, path: Path, **kwargs: str) -> str:
        """Render a template from an explicit *path* (e.g. a per-corpus override).

        *path* must be absolute — relative-path resolution is the caller's
        responsibility (it needs the corpus TOML directory).  No path-traversal
        guard is applied; callers that resolve paths from user-supplied config
        should validate existence before calling this method.

        Raises :exc:`OSError` if *path* is not readable, and
        :exc:`UnicodeDecodeError` if the file is not valid UTF-8.  Both bubble
        up to the CLI layer for wrapping into :exc:`click.ClickException`.
        """
        template_text = path.read_text(encoding="utf-8")
        return _substitute(template_text, str(path), **kwargs)
