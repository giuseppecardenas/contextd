"""Deterministic lexical reference extraction — ground-truth edges, no LLM.

Markdown links, ``§``-anchors, and code imports are unambiguous references;
they should become edges directly instead of riding the failure-prone LLM
path. This layer runs inside the relate workers before the model call, emits
:class:`LexicalReference` rows at confidence 1.0, and routes them through the
same ``_apply_inferred_edge`` gates (ontology, triple constraints, resolution
cascade) as model output — deterministic in *detection*, uniformly validated
in *writing*.

Extractors are per-format (Contract: they receive one unit's body text plus
its :class:`~contextd.inference.context.UnitIdentity`; they never parse
document structure themselves). Per-corpus ``[[lexical.patterns]]`` rows add
domain vocabulary — e.g. runeledger's ``FR-XXX-NNN`` rows → ``FRRow`` (an
ontology alias for Ticket) — with alias resolution applied at emit time so
downstream only sees canonical types.
"""

from __future__ import annotations

import logging
import posixpath
import re
from dataclasses import dataclass
from typing import Protocol

from contextd.corpus_config import LexicalPattern
from contextd.inference.context import UnitIdentity
from contextd.ontology.schema import Ontology

_log = logging.getLogger(__name__)

# Bound the text scanned per unit so a pathological file can't stall a worker.
_MAX_SCAN_CHARS = 200_000


@dataclass(frozen=True)
class LexicalReference:
    edge_type: str
    target_type: str
    target_name: str
    rule: str  # "md-link" | "md-anchor" | "md-basename" | "lua-require" | "pattern:<regex>"


class LexicalExtractor(Protocol):
    suffixes: frozenset[str]

    def extract(self, content: str, *, identity: UnitIdentity) -> list[LexicalReference]: ...


_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_SECTION_REF = re.compile(r"§\s*(\d+(?:\.\d+)+)")
_BASENAME = re.compile(r"\b[\w][\w/.-]*\.(?:md|lua|toml)\b")


def _join_relative(identity: UnitIdentity, target: str) -> str:
    """Resolve a link target relative to the source file's directory."""
    if posixpath.isabs(target) or re.match(r"^[A-Za-z]:", target):
        return target.replace("\\", "/")
    base_dir = identity.file_path.rsplit("/", 1)[0]
    return posixpath.normpath(posixpath.join(base_dir, target.replace("\\", "/")))


class MarkdownLexicalExtractor:
    suffixes: frozenset[str] = frozenset({".md"})

    def extract(self, content: str, *, identity: UnitIdentity) -> list[LexicalReference]:
        text = content[:_MAX_SCAN_CHARS]
        refs: list[LexicalReference] = []
        linked_targets: set[str] = set()
        for m in _MD_LINK.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            linked_targets.add(target.rsplit("#", 1)[0].rsplit("/", 1)[-1])
            if "#" in target:
                path_part, anchor = target.rsplit("#", 1)
                if not anchor:
                    continue
                if path_part:
                    resolved = f"{_join_relative(identity, path_part)}#{anchor}"
                else:
                    resolved = f"{identity.file_path}#{anchor}"
                refs.append(
                    LexicalReference(
                        edge_type="REFERENCES",
                        target_type="Section",
                        target_name=resolved,
                        rule="md-link",
                    )
                )
            else:
                refs.append(
                    LexicalReference(
                        edge_type="REFERENCES",
                        target_type="File",
                        target_name=_join_relative(identity, target),
                        rule="md-link",
                    )
                )
        for m in _SECTION_REF.finditer(text):
            refs.append(
                LexicalReference(
                    edge_type="REFERENCES",
                    target_type="Section",
                    target_name=f"§{m.group(1)}",
                    rule="md-anchor",
                )
            )
        own_name = identity.file_path.rsplit("/", 1)[-1]
        for m in _BASENAME.finditer(text):
            token = m.group(0)
            basename = token.rsplit("/", 1)[-1]
            if basename == own_name or basename in linked_targets:
                continue
            refs.append(
                LexicalReference(
                    edge_type="REFERENCES",
                    target_type="File",
                    target_name=token,
                    rule="md-basename",
                )
            )
        return refs


_LUA_REQUIRE = re.compile(r"""\brequire\s*\(?\s*['"]([\w./-]+)['"]""")
_LUA_DOFILE = re.compile(r"""\bdofile\s*\(\s*['"]([\w./-]+)['"]""")


class LuaLexicalExtractor:
    suffixes: frozenset[str] = frozenset({".lua"})

    def extract(self, content: str, *, identity: UnitIdentity) -> list[LexicalReference]:
        text = content[:_MAX_SCAN_CHARS]
        refs: list[LexicalReference] = []
        for m in _LUA_REQUIRE.finditer(text):
            module = m.group(1)
            path = module if module.endswith(".lua") else module.replace(".", "/") + ".lua"
            refs.append(
                LexicalReference(
                    edge_type="DEPENDS_ON",
                    target_type="File",
                    target_name=path,
                    rule="lua-require",
                )
            )
        for m in _LUA_DOFILE.finditer(text):
            refs.append(
                LexicalReference(
                    edge_type="DEPENDS_ON",
                    target_type="File",
                    target_name=m.group(1),
                    rule="lua-dofile",
                )
            )
        return refs


class LexicalRegistry:
    """Routes a unit to its format extractor + per-corpus custom patterns.

    Edge/node aliases are resolved at emit time (this path bypasses the LLM
    parse gate where alias resolution normally happens), so ``_apply_inferred_edge``
    only ever sees canonical types. Extraction is pure and exception-guarded
    at the call site — it degrades to zero refs, never blocks the LLM call.
    """

    _BUILTINS: tuple[LexicalExtractor, ...] = (
        MarkdownLexicalExtractor(),
        LuaLexicalExtractor(),
    )

    def __init__(self, ontology: Ontology, patterns: list[LexicalPattern] | None = None) -> None:
        self._onto = ontology
        self._patterns: list[tuple[LexicalPattern, re.Pattern[str]]] = [
            (p, re.compile(p.regex)) for p in (patterns or [])
        ]

    def extract(self, content: str, *, identity: UnitIdentity) -> list[LexicalReference]:
        suffix = identity.suffix
        refs: list[LexicalReference] = []
        for extractor in self._BUILTINS:
            if suffix in extractor.suffixes:
                refs.extend(extractor.extract(content, identity=identity))
        fmt = suffix.lstrip(".")
        for pattern, compiled in self._patterns:
            if pattern.formats and fmt not in pattern.formats:
                continue
            for m in compiled.finditer(content[:_MAX_SCAN_CHARS]):
                try:
                    value = m.group(pattern.capture)
                except IndexError:
                    _log.warning(
                        "lexical: capture group %d missing for pattern %r",
                        pattern.capture,
                        pattern.regex,
                    )
                    break
                if value:
                    refs.append(
                        LexicalReference(
                            edge_type=pattern.edge_type,
                            target_type=pattern.target_type,
                            target_name=value,
                            rule=f"pattern:{pattern.regex}",
                        )
                    )
        # Alias-resolve to canon + dedupe, preserving first-seen order.
        seen: set[tuple[str, str, str]] = set()
        out: list[LexicalReference] = []
        for ref in refs:
            edge = self._onto.resolve_edge_alias(ref.edge_type)
            target_type = self._onto.resolve_alias(ref.target_type)
            key = (edge, target_type, ref.target_name)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                LexicalReference(
                    edge_type=edge,
                    target_type=target_type,
                    target_name=ref.target_name,
                    rule=ref.rule,
                )
            )
        return out
