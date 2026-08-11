"""Per-file (or per-section) summariser that ties provider + prompt + parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from contextd.inference._json_body import loads_json_body
from contextd.inference.context import UnitIdentity, identity_vars
from contextd.inference.prompts import PromptRenderer
from contextd.inference.routing import SummaryPromptRouter
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
        router: SummaryPromptRouter | None = None,
    ) -> None:
        """Wire provider + renderer + config for per-file summarisation.

        ``prompt_path``: absolute path to an override template. If None,
        Summariser uses the default 'summarise' template from the renderer's
        search directory. The override receives the same variable set as the
        default (content, max_words, and the identity keys) — templates that
        reference only a subset are valid.
        """
        self._provider = provider
        self._renderer = renderer
        self._max_words = max_words
        self._prompt_path = prompt_path
        self._router = router

    def summarise(self, content: str, *, context: UnitIdentity | None = None) -> FileSummary:
        """Summarise one unit of content.

        ``context`` carries the unit's identity (path, section title, parent
        chain) into the prompt's Source block so the model knows what it is
        reading; ``None`` renders empty identity fields. Template resolution
        order: per-suffix router (keyed by the unit's relative path) →
        corpus-wide ``prompt_path`` override → packaged default.
        """
        template_vars = {
            "content": content,
            "max_words": str(self._max_words),
            **identity_vars(context),
        }
        routed = (
            self._router.resolve(context.rel_path)
            if self._router is not None and context is not None
            else None
        )
        if routed is not None:
            prompt = self._renderer.render_path(routed, **template_vars)
        elif self._prompt_path is not None:
            prompt = self._renderer.render_path(self._prompt_path, **template_vars)
        else:
            prompt = self._renderer.render("summarise", **template_vars)
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="summary")
        )
        data = loads_json_body(response)
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

    def roll_up(
        self,
        *,
        child_summaries: list[str],
        own_prose: str,
        context: UnitIdentity | None = None,
    ) -> str:
        """Synthesise a parent-level summary from child summaries + own prose.

        Used for parent sections (whose bodies are exclusive and may be
        prose-less) and for file-level summaries over top-level section
        summaries. Always renders the packaged ``rollup`` template — never the
        per-corpus override or the router: the input is already-normalised
        summaries, format-neutral by design.
        """
        bullets = "\n".join(f"- {s}" for s in child_summaries) or "(none)"
        prompt = self._renderer.render(
            "rollup",
            child_summaries=bullets,
            own_prose=own_prose,
            max_words=str(self._max_words),
            **identity_vars(context),
        )
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="summary")
        )
        data = loads_json_body(response)
        if "summary" not in data:
            raise KeyError(f"Provider response missing 'summary'; got keys {list(data.keys())}")
        summary = data["summary"]
        if not isinstance(summary, str):
            raise TypeError(
                f"Provider response 'summary' must be a string; got {type(summary).__name__}"
            )
        return summary
