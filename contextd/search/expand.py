"""Neighbour expansion for chunk hits (sentence-window style).

A chunk hit is often too narrow on its own; the ``window`` neighbours on
each side (same parent, same profile, by ordinal) are attached as
``context_before`` / ``context_after``. Neighbours are looked up by ordinal
range rather than by walking ``NEXT_SIBLING`` so the result order is
deterministic and the query is one index-backed match per hit.
"""

from __future__ import annotations

from typing import Any

from contextd.storage.base import GraphStore


def neighbours(
    store: GraphStore, *, parent_id: str, profile: str, ordinal: int, window: int
) -> tuple[list[str], list[str]]:
    if window <= 0:
        return [], []
    rows = store.exec_read(
        "MATCH (n:Chunk {parent_id: $pid, profile: $profile}) "
        "WHERE n.ordinal >= $lo AND n.ordinal <= $hi AND n.ordinal <> $ord "
        "RETURN n.ordinal AS ordinal, n.text AS text ORDER BY n.ordinal",
        {
            "pid": parent_id,
            "profile": profile,
            "lo": ordinal - window,
            "hi": ordinal + window,
            "ord": ordinal,
        },
    )
    before = [str(r["text"]) for r in rows if int(r["ordinal"]) < ordinal]
    after = [str(r["text"]) for r in rows if int(r["ordinal"]) > ordinal]
    return before, after


def attach_context(store: GraphStore, rows: list[dict[str, Any]], *, window: int) -> None:
    """Add ``context_before`` / ``context_after`` to raw chunk rows in place."""
    if window <= 0:
        return
    for r in rows:
        pid, profile, ordinal = r.get("parent_id"), r.get("profile"), r.get("ordinal")
        if pid is None or profile is None or ordinal is None:
            continue
        try:
            before, after = neighbours(
                store, parent_id=str(pid), profile=str(profile), ordinal=int(ordinal), window=window
            )
        except Exception:
            continue
        r["context_before"] = "\n".join(before)
        r["context_after"] = "\n".join(after)


_ELLIPSIS = " [...]"


def _clip(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[: max(0, max_chars - len(_ELLIPSIS))].rstrip() + _ELLIPSIS
    return text


def attach_evidence_context(
    store: GraphStore, rows: list[dict[str, Any]], *, window: int, max_chars: int
) -> None:
    """Add neighbour context to the ``evidence`` block of collapsed unit rows.

    Used when collapse ran with ``window = 0`` because the rows were still
    candidates (graph expansion fuses a deeper direct list than the caller's
    ``limit``); once the final rows are known the neighbours are fetched for
    just those, by ``evidence.chunk_id`` — one index-backed query per row,
    exactly what :func:`contextd.search.collapse.collapse` would have issued,
    clipped to the same ``max_chars``. Rows without evidence (graph-only
    rows) or already carrying context are left alone.
    """
    if window <= 0:
        return
    for r in rows:
        ev = r.get("evidence")
        if not isinstance(ev, dict) or "context_before" in ev or ev.get("chunk_id") is None:
            continue
        try:
            found = store.exec_read(
                "MATCH (c:Chunk {id: $id}) "
                "MATCH (n:Chunk {parent_id: c.parent_id, profile: c.profile}) "
                "WHERE n.ordinal >= c.ordinal - $w AND n.ordinal <= c.ordinal + $w "
                "AND n.ordinal <> c.ordinal "
                "RETURN n.ordinal AS ordinal, n.text AS text, c.ordinal AS pivot "
                "ORDER BY n.ordinal",
                {"id": str(ev["chunk_id"]), "w": int(window)},
            )
        except Exception:
            continue
        before = [str(x["text"]) for x in found if int(x["ordinal"]) < int(x["pivot"])]
        after = [str(x["text"]) for x in found if int(x["ordinal"]) > int(x["pivot"])]
        ev["context_before"] = _clip("\n".join(before), max_chars)
        ev["context_after"] = _clip("\n".join(after), max_chars)


def expand_chunk(store: GraphStore, chunk_id: str, *, window: int = 2) -> dict[str, Any] | None:
    """One chunk with its neighbours and the parent's summary — the
    assistant's "show me more around this hit"."""
    rows = store.exec_read(
        "MATCH (c:Chunk {id: $id}) "
        "OPTIONAL MATCH (p)-[:CONTAINS]->(c) "
        "RETURN c.id AS id, c.path AS path, c.parent_id AS parent_id, "
        "c.parent_label AS parent_label, c.profile AS profile, c.ordinal AS ordinal, "
        "c.kind AS kind, c.text AS text, c.prefix AS prefix, c.start_line AS start_line, "
        "c.end_line AS end_line, p.summary AS parent_summary, p.title AS parent_title",
        {"id": chunk_id},
    )
    if not rows:
        return None
    row = dict(rows[0])
    before, after = neighbours(
        store,
        parent_id=str(row["parent_id"]),
        profile=str(row["profile"]),
        ordinal=int(row["ordinal"]),
        window=window,
    )
    row["context_before"] = before
    row["context_after"] = after
    return row
