"""Per-suffix summary prompt routing.

A corpus-wide ``prompt_override`` cannot serve a mixed corpus: runeledger's
PRD-section prompt was applied verbatim to 210 Lua source files ("You are
summarising a Runeledger PRD section..."). ``[summarization.overrides]`` maps
glob patterns (matched against the unit's corpus-root-relative posix path,
first match in declaration order wins) to a resolved template path; the
router rides inside the ``Summariser`` keyed off ``UnitIdentity.rel_path``,
so no pipeline or daemon plumbing changes.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptRoute:
    pattern: str  # glob matched against corpus-root-relative posix path
    path: Path  # resolved, validated template path


class SummaryPromptRouter:
    def __init__(self, routes: Sequence[PromptRoute]) -> None:
        self._routes = tuple(routes)

    def resolve(self, rel_path: str | None) -> Path | None:
        """First-match-wins template path for a unit, or None (no route)."""
        if not rel_path:
            return None
        for route in self._routes:
            if fnmatch.fnmatch(rel_path, route.pattern):
                return route.path
        return None
