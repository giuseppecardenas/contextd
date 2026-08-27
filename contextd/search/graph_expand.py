"""Graph-first expansion of search hits ("documents that cite what this passage cites").

A hybrid ranker finds the passage that *describes* a requirement; it rarely
finds the code that *implements* it, because a PRD section and a Lua module
share almost no vocabulary. They do share **entities**: the section mentions
``register_settlement_type`` / ``FR-STL-001``, the module carries lexical
``REFERENCES`` edges to the same ``Pattern`` / ``Ticket`` nodes. Walking two
hops from the top hits through those shared entities — the HippoRAG /
LightRAG "entity neighbourhood" move — recovers the implementing units with
no extra provider call. On the runeledger corpus it lifted recall on
relational questions from 0.21 to 0.66 (``docs/investigations``).

This module is the walk: one read query, seeds in, unit rows out. It is
pure graph traversal — it never embeds, never ranks by text — and it stays
behind :class:`~contextd.storage.base.GraphStore` (``exec_read`` only).
Fusing its rows with the direct hits is the caller's job
(:func:`contextd.mcp.tools.search`), through the same reciprocal-rank
fusion that already merges the vector and full-text rankers.

Three details of the query matter:

* **Hub damping.** Entities like the product name or the language touch
  half the corpus; without ``1 / log(2 + degree)`` they would dominate every
  expansion (the reason HippoRAG weights by node specificity). It is
  computed inline, which is cheap at the scale contextd targets.
* **Seed rank decay** (``1 / (1 + rank)``): the neighbourhood of the top hit
  counts more than the third's.
* **Structural edges are excluded** from the direct unit-to-unit branch:
  ``CONTAINS`` / ``NEXT_SIBLING`` would return every sibling of a seed,
  which ``window`` and ``section_tree`` already do on purpose. Only
  ``inferred`` / ``manual`` edges count as a relation between two units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final, Literal

from contextd.storage.base import GraphStore

_log = logging.getLogger(__name__)

ExpandMode = Literal["none", "units"]

MAX_SEEDS: Final[int] = 10
MAX_VIA: Final[int] = 5
"""Entity names reported per expanded row (the strongest connectors)."""

# Labels that are never an "entity" for the shared-entity branch: the
# document units themselves, the derived retrieval nodes, and bookkeeping.
_NOT_ENTITY: Final[str] = (
    "NOT (e:File OR e:Section OR e:Chunk OR e:Topic OR e:Corpus OR e:Meta OR e:WorkSession)"
)

_WALK: Final[str] = f"""
UNWIND $seeds AS seed
CALL {{
  WITH seed
  MATCH (s:Section {{id: seed.id}}) RETURN s
  UNION
  WITH seed
  MATCH (s:File {{path: seed.id}}) RETURN s
}}
WITH seed, s
WHERE $corpus IS NULL OR s.corpus = $corpus
CALL {{
  WITH s
  MATCH (s)-[]-(e)-[]-(u)
  WHERE {_NOT_ENTITY}
    AND (u:File OR (u:Section AND NOT $to_file)) AND u.path <> s.path
    AND ($corpus IS NULL OR u.corpus = $corpus)
  RETURN u, coalesce(e.name, e.id, e.title, e.description) AS via,
         1.0 / log(2 + COUNT {{ (e)--() }}) AS w
  UNION ALL
  WITH s
  MATCH (s)-[r]-(u)
  WHERE (u:File OR (u:Section AND NOT $to_file)) AND u.path <> s.path
    AND r.origin IN ['inferred', 'manual']
    AND ($corpus IS NULL OR u.corpus = $corpus)
  RETURN u, null AS via, 1.0 AS w
}}
WITH u, seed, via, w
WITH u, sum(w / (1.0 + seed.rank)) AS score,
     collect(DISTINCT via) AS vias, collect(DISTINCT seed.id) AS seeds
RETURN coalesce(u.id, u.path) AS id, u.path AS path, labels(u) AS labels,
       u.title AS title, u.name AS name, u.summary AS summary, u.corpus AS corpus,
       score, [v IN vias WHERE v IS NOT NULL][..$max_via] AS via, seeds
ORDER BY score DESC, path ASC, id ASC
LIMIT $limit
"""


@dataclass(frozen=True)
class Seed:
    """One unit the walk starts from, with its 0-based rank in the direct hits."""

    id: str
    rank: int


def seeds_from_rows(rows: list[dict[str, Any]], *, n: int) -> list[Seed]:
    """The first ``n`` distinct units among best-first search rows.

    A row seeds with the *finest* unit that matched — a Section id or a File
    path, the two shapes the walk matches by index. Fused chunk rows (and
    ``chunk`` unit rows) carry it as ``parent_id``; collapsed ``section`` /
    ``file`` rows carry their best chunk's parent as ``evidence.parent_id``,
    so a File row of a section-granular file seeds from the Section that
    actually contains the hit rather than from the whole file, whose entity
    edges are a different, coarser set; a unit row without evidence seeds
    with its own ``id``. Rows of no recognisable shape are skipped.

    ``search`` seeds from the fused *chunk* list rather than from the
    collapsed rows on purpose: collapse ranks a parent by the mean of its
    chunks, so its top rows are not the units with the best-matching
    passages, and the walk wants the latter.
    """
    seeds: list[Seed] = []
    seen: set[str] = set()
    for row in rows:
        raw: Any = row.get("parent_id")
        if raw is None:
            ev = row.get("evidence")
            raw = ev.get("parent_id") if isinstance(ev, dict) else None
        if raw is None and row.get("unit") in ("section", "file"):
            raw = row.get("id")
        if not isinstance(raw, str) or raw in seen:
            continue
        seen.add(raw)
        seeds.append(Seed(raw, len(seeds)))
        if len(seeds) >= n:
            break
    return seeds


def _shape(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels") or []
    unit = "section" if "Section" in labels else "file"
    out: dict[str, Any] = {
        "unit": unit,
        "id": row.get("id"),
        "path": row.get("path"),
        "summary": row.get("summary"),
        "corpus": row.get("corpus"),
        "score": float(row.get("score") or 0.0),
        "via": {
            "entities": [str(v) for v in (row.get("via") or [])],
            "seeds": [str(s) for s in (row.get("seeds") or [])],
        },
    }
    if unit == "section":
        out["title"] = row.get("title")
    else:
        out["name"] = row.get("name")
    return out


def expand_units(
    store: GraphStore,
    seeds: list[Seed],
    *,
    corpus: str | None,
    limit: int,
    to_file: bool = False,
) -> list[dict[str, Any]]:
    """Section/File units within two hops of ``seeds``, best-connected first.

    Each row has the collapsed-unit shape (``unit``, ``id``, ``path``,
    ``summary``, ``corpus``, ``score``, plus ``title`` or ``name``) and a
    ``via`` block naming the shared entities and the seeds that reached it —
    the explanation of *why* a unit that matched no query term is in the
    result. ``score`` is the damped path weight, comparable only within one
    call; callers rank by position, not by value.

    ``to_file`` (``return_unit = "file"``) restricts targets to File nodes —
    units whose entity edges hang on the file itself (file-granular code
    modules, whole documents). It deliberately does *not* roll Sections up
    to their File: summing a long document's many sections lets big
    documents swamp the modules a relational question is after (measured on
    runeledger: recall 0.17 rolled-up vs 0.65 file-only), and a
    section-granular document is still reachable through any file-level
    inferred edge.

    Never raises on a backend error: the walk is an enrichment of a search
    that already has an answer, so a failure is logged and yields no rows.
    """
    if not seeds or limit <= 0:
        return []
    params = {
        "seeds": [{"id": s.id, "rank": int(s.rank)} for s in seeds[:MAX_SEEDS]],
        "corpus": corpus,
        "limit": int(limit),
        "max_via": MAX_VIA,
        "to_file": bool(to_file),
    }
    try:
        rows = store.exec_read(_WALK, params)
    except Exception as exc:
        _log.warning("search: graph expansion failed (%s: %s)", type(exc).__name__, exc)
        return []
    return [_shape(dict(r)) for r in rows]
