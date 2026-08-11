"""Format-plural unit extraction: parser interface, parsed-file carrier, cache.

contextd is markdown-first but not markdown-only by design. This module owns
the seam between "a file on disk" and "the list of indexable units inside it":

* :class:`UnitExtractor` — the per-format parser protocol.
  :class:`~contextd.indexer.heading_parser.HeadingParser` satisfies it
  structurally and is today's only implementation; a code-unit parser
  (module/function/class granularity) would plug in beside it without any
  storage or phase code branching on format.
* :func:`extractor_for` — the single selection point from corpus config +
  file suffix to an extractor (or ``None`` → file-granular treatment).
* :class:`ParsedFile` — one file's parse result with the derived lookups the
  phases need (anchor → section, parent → children, ancestor title chains).
* :class:`ParseCache` — parse-on-miss cache scoped to one pipeline
  invocation. A bootstrap previously parsed every file 4+ times (enumerate,
  gc, summarise, relate each re-parsed from disk); one shared cache collapses
  that to a single read+parse per file.

Thread-safety: the phases pre-populate the cache serially before dispatching
``_parallel_map`` workers; :class:`ParsedFile`'s derived maps are built at
construction, so worker threads only ever read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from contextd._paths import canonical_path
from contextd.corpus_config import CorpusConfig
from contextd.indexer.heading_parser import HeadingParser, ParsedSection

_log = logging.getLogger(__name__)


class UnitExtractor(Protocol):
    """A per-format parser producing the file's indexable units in order."""

    def parse(self, text: str) -> list[ParsedSection]: ...


def extractor_for(corpus_cfg: CorpusConfig, suffix: str) -> UnitExtractor | None:
    """Return the unit extractor for a file suffix, or ``None`` (file-granular).

    The one place format routing happens. ``corpus_cfg.corpus.content_profile``
    is consulted here as the future forcing knob; this pass routes on suffix
    only — ``.md`` gets the heading parser, everything else is file-granular.
    """
    if suffix == ".md":
        return HeadingParser(
            min_level=corpus_cfg.corpus.heading_min_level,
            max_level=corpus_cfg.corpus.heading_max_level,
        )
    return None


def own_prose(section: ParsedSection) -> str:
    """A section's body minus its heading line.

    With exclusive bodies this is exactly the prose the section itself owns.
    A parent whose heading is immediately followed by a child heading yields
    ``""`` — the definition of "prose-less" used by the roll-up phase:
    whitespace-only own prose, deterministically, with no length threshold to
    tune.
    """
    _, _, rest = section.body.partition("\n")
    return rest


@dataclass
class ParsedFile:
    """One file's parse result plus the derived lookups phases need."""

    path: Path
    canonical: str
    sections: list[ParsedSection]
    _by_anchor: dict[str, ParsedSection] = field(init=False, repr=False)
    _children: dict[str | None, list[ParsedSection]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_anchor = {sec.anchor: sec for sec in self.sections}
        self._children = {}
        for sec in self.sections:
            self._children.setdefault(sec.parent_anchor, []).append(sec)

    def by_anchor(self, anchor: str) -> ParsedSection | None:
        return self._by_anchor.get(anchor)

    def children_of(self, anchor: str) -> list[ParsedSection]:
        """Direct children of the section with this anchor, document order."""
        return self._children.get(anchor, [])

    def parent_chain(self, anchor: str) -> tuple[str, ...]:
        """Ancestor titles for a section, outermost first. Empty at top level."""
        titles: list[str] = []
        sec = self._by_anchor.get(anchor)
        while sec is not None and sec.parent_anchor is not None:
            sec = self._by_anchor.get(sec.parent_anchor)
            if sec is None:
                break
            titles.append(sec.title)
        return tuple(reversed(titles))


class ParseCache:
    """Parse-on-miss cache of :class:`ParsedFile`, one per pipeline invocation.

    Lifetime equals one bootstrap / one incremental-file call, so there is no
    invalidation: within an invocation the on-disk content is treated as
    fixed. The daemon sweep constructs a fresh extractor per check instead of
    using this cache because it must observe fresh disk state.
    """

    def __init__(self, corpus_cfg: CorpusConfig) -> None:
        self._corpus_cfg = corpus_cfg
        self._files: dict[str, ParsedFile] = {}

    def get(self, path: Path) -> ParsedFile:
        key = str(path)
        cached = self._files.get(key)
        if cached is not None:
            return cached
        extractor = extractor_for(self._corpus_cfg, path.suffix)
        if extractor is None:
            _log.debug("parse cache: no unit extractor for %s; empty unit list", path)
            sections: list[ParsedSection] = []
        else:
            sections = extractor.parse(path.read_text(encoding="utf-8", errors="replace"))
        parsed = ParsedFile(path=path, canonical=canonical_path(path), sections=sections)
        self._files[key] = parsed
        return parsed
