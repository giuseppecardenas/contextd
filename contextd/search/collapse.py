"""Small-to-big collapse of chunk hits to their enclosing unit.

Chunks are what the rankers score; Sections and Files are what an
assistant wants back. Given fused chunk rows this module groups them by
parent and decides, per parent, whether to return the parent (with the
best chunk attached as ``evidence``) or the best chunk itself:

* ``return_unit = "chunk"`` — every hit is returned as-is.
* ``"section"`` / ``"file"`` — always collapse to that unit (Section parents
  roll up to their File for ``"file"``).
* ``"auto"`` — LlamaIndex / Haystack auto-merging: collapse to the parent
  when at least ``auto_merge_threshold`` of the parent's chunks (in the
  best-covered profile) were retrieved, otherwise return the best chunk.

The parent's score is the mean of its member chunks' fused scores, the
LlamaIndex ``AutoMergingRetriever`` convention. Rows carry ``unit`` so a
caller can tell the shapes apart.

Known limitation: "retrieved" means "returned by a ranker", and the vector
leg returns its ``fetch_k`` nearest neighbours regardless of similarity, so
on a corpus with fewer chunks than ``fetch_k`` nearly every chunk counts as
retrieved and ``auto`` collapses eagerly. It is the same behaviour as the
LlamaIndex/Haystack retrievers it mirrors; on realistic corpora ``fetch_k``
is a small fraction of the chunk count and the ratio is meaningful.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from contextd.search.expand import neighbours
from contextd.storage.base import GraphStore

ReturnUnit = Literal["chunk", "section", "file", "auto"]

_ELLIPSIS = " [...]"


def _clip(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[: max(0, max_chars - len(_ELLIPSIS))].rstrip() + _ELLIPSIS
    return text


def _evidence(
    store: GraphStore, row: dict[str, Any], max_chars: int, window: int
) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "chunk_id": row.get("id"),
        # The chunk's own parent, which for a File row of a section-granular
        # file is the Section that matched — the finest unit graph expansion
        # can seed from.
        "parent_id": row.get("parent_id"),
        "parent_label": row.get("parent_label"),
        "profile": row.get("profile"),
        "kind": row.get("kind"),
        "start_line": row.get("start_line"),
        "end_line": row.get("end_line"),
        "text": _clip(str(row.get("text") or ""), max_chars),
    }
    # Neighbour context is fetched only for rows that are actually returned
    # (bounded by ``limit``), never for every fused candidate.
    if window > 0 and row.get("parent_id") is not None and row.get("ordinal") is not None:
        try:
            before, after = neighbours(
                store,
                parent_id=str(row["parent_id"]),
                profile=str(row.get("profile")),
                ordinal=int(row["ordinal"]),
                window=window,
            )
        except Exception:
            before, after = [], []
        ev["context_before"] = _clip("\n".join(before), max_chars)
        ev["context_after"] = _clip("\n".join(after), max_chars)
    return ev


def chunk_row(
    store: GraphStore, row: dict[str, Any], max_chars: int, window: int
) -> dict[str, Any]:
    """Wire shape for a chunk returned as its own unit."""
    return {
        "unit": "chunk",
        "id": row.get("id"),
        "path": row.get("path"),
        "parent_id": row.get("parent_id"),
        "parent_label": row.get("parent_label"),
        "profile": row.get("profile"),
        "score": row.get("score"),
        "evidence": _evidence(store, row, max_chars, window),
    }


def _profile_totals(store: GraphStore, parent_ids: list[str]) -> dict[tuple[str, str], int]:
    if not parent_ids:
        return {}
    rows = store.exec_read(
        "MATCH (c:Chunk) WHERE c.parent_id IN $ids "
        "RETURN c.parent_id AS pid, c.profile AS profile, count(c) AS n",
        {"ids": parent_ids},
    )
    return {(str(r["pid"]), str(r["profile"])): int(r["n"]) for r in rows}


def _fetch_parents(
    store: GraphStore, section_ids: list[str], file_paths: list[str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if section_ids:
        for r in store.exec_read(
            "MATCH (s:Section) WHERE s.id IN $ids "
            "RETURN s.id AS id, s.path AS path, s.title AS title, s.level AS level, "
            "s.summary AS summary, s.corpus AS corpus",
            {"ids": section_ids},
        ):
            out[str(r["id"])] = {"unit": "section", **r}
    if file_paths:
        for r in store.exec_read(
            "MATCH (f:File) WHERE f.path IN $paths "
            "RETURN f.path AS id, f.path AS path, f.name AS name, "
            "f.summary AS summary, f.corpus AS corpus",
            {"paths": file_paths},
        ):
            out[str(r["id"])] = {"unit": "file", **r}
    return out


def collapse(
    store: GraphStore,
    rows: list[dict[str, Any]],
    *,
    return_unit: ReturnUnit,
    auto_merge_threshold: float,
    limit: int,
    max_evidence_chars: int,
    window: int = 0,
) -> list[dict[str, Any]]:
    """Collapse fused chunk rows (best-first) into unit rows, best-first."""
    if return_unit == "chunk" or not rows:
        return [chunk_row(store, r, max_evidence_chars, window) for r in rows[:limit]]

    # Target parent per chunk: the Section for section-parented chunks unless
    # the caller asked for files, in which case everything rolls up to the path.
    def target(r: dict[str, Any]) -> tuple[str, str]:
        parent_label = str(r.get("parent_label") or "File")
        if return_unit == "file" or parent_label != "Section":
            return ("File", str(r.get("path")))
        return ("Section", str(r.get("parent_id")))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for r in rows:
        key = target(r)
        if key not in groups:
            order.append(key)
        groups[key].append(r)

    # Auto-merge needs the ratio of retrieved chunks to the parent's total,
    # per profile, taking the best-covered profile. Only "auto" consults it.
    totals: dict[tuple[str, str], int] = {}
    if return_unit == "auto":
        totals = _profile_totals(
            store, sorted({str(r.get("parent_id")) for r in rows if r.get("parent_id")})
        )

    def merge_ratio(members: list[dict[str, Any]]) -> float:
        hits: dict[tuple[str, str], set[str]] = defaultdict(set)
        for m in members:
            hits[(str(m.get("parent_id")), str(m.get("profile")))].add(str(m.get("id")))
        best = 0.0
        for key, ids in hits.items():
            total = totals.get(key, 0)
            if total > 0:
                best = max(best, len(ids) / total)
        return best

    decided: list[tuple[tuple[str, str] | None, list[dict[str, Any]]]] = []
    for key in order:
        members = groups[key]
        if return_unit == "auto" and merge_ratio(members) < auto_merge_threshold:
            decided.append((None, members))
        else:
            decided.append((key, members))

    parents = _fetch_parents(
        store,
        [k[1] for k, _ in decided if k is not None and k[0] == "Section"],
        [k[1] for k, _ in decided if k is not None and k[0] == "File"],
    )

    # Rank first, then build evidence (with its neighbour queries) only for
    # the rows that survive ``limit``.
    ranked: list[tuple[float, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]] = []
    for target_key, members in decided:
        best = max(members, key=lambda m: float(m.get("score") or 0.0))
        parent = parents.get(target_key[1]) if target_key is not None else None
        if parent is not None:
            score = sum(float(m.get("score") or 0.0) for m in members) / len(members)
        else:
            # No collapse (auto-merge below threshold), or the parent vanished
            # between ranking and collapse: the chunk itself is the answer.
            score = float(best.get("score") or 0.0)
        ranked.append((score, parent, members, best))
    ranked.sort(key=lambda t: -t[0])

    out: list[dict[str, Any]] = []
    for score, parent, members, best in ranked[:limit]:
        if parent is None:
            out.append(chunk_row(store, best, max_evidence_chars, window))
            continue
        out.append(
            {
                **parent,
                "score": score,
                "matched_chunks": len({str(m.get("id")) for m in members}),
                "profile": best.get("profile"),
                "evidence": _evidence(store, best, max_evidence_chars, window),
            }
        )
    return out
