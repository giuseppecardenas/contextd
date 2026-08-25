from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from contextd.search.collapse import collapse


def _chunk(
    cid: str,
    parent: str,
    path: str,
    score: float,
    *,
    profile: str = "fine",
    ordinal: int = 0,
    label: str = "Section",
) -> dict[str, Any]:
    return {
        "id": cid,
        "parent_id": parent,
        "parent_label": label,
        "path": path,
        "profile": profile,
        "ordinal": ordinal,
        "kind": "prose",
        "text": f"text of {cid}",
        "start_line": ordinal * 10,
        "end_line": ordinal * 10 + 5,
        "score": score,
    }


def _store(
    totals: dict[tuple[str, str], int],
    sections: dict[str, dict[str, Any]],
    files: dict[str, dict[str, Any]] | None = None,
) -> MagicMock:
    store = MagicMock()

    def _read(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        if "count(c) AS n" in cypher:
            return [
                {"pid": pid, "profile": prof, "n": n}
                for (pid, prof), n in totals.items()
                if pid in params["ids"]
            ]
        if "MATCH (s:Section)" in cypher:
            return [{"id": sid, **props} for sid, props in sections.items() if sid in params["ids"]]
        if "MATCH (f:File)" in cypher:
            return [
                {"id": p, "path": p, **props}
                for p, props in (files or {}).items()
                if p in params["paths"]
            ]
        if "n.ordinal" in cypher:
            return [{"ordinal": params["ord"] + 1, "text": "after text"}]
        return []

    store.exec_read.side_effect = _read
    return store


def test_return_unit_chunk_passes_hits_through_with_evidence() -> None:
    store = _store({}, {})
    rows = [_chunk("c1", "a.md#s", "a.md", 0.5), _chunk("c2", "a.md#s", "a.md", 0.4, ordinal=1)]
    out = collapse(
        store, rows, return_unit="chunk", auto_merge_threshold=0.5, limit=5, max_evidence_chars=1200
    )
    assert [r["unit"] for r in out] == ["chunk", "chunk"]
    assert out[0]["evidence"]["text"] == "text of c1"
    assert out[0]["evidence"]["start_line"] == 0 and out[0]["evidence"]["end_line"] == 5
    assert "context_after" not in out[0]["evidence"]  # window=0
    store.exec_read.assert_not_called()


def test_auto_merges_when_ratio_meets_threshold() -> None:
    sections = {
        "a.md#s": {"path": "a.md", "title": "S", "level": 2, "summary": "sum", "corpus": "c"}
    }
    store = _store({("a.md#s", "fine"): 4}, sections)
    rows = [
        _chunk("c1", "a.md#s", "a.md", 0.6, ordinal=0),
        _chunk("c2", "a.md#s", "a.md", 0.2, ordinal=1),
    ]
    out = collapse(
        store, rows, return_unit="auto", auto_merge_threshold=0.5, limit=5, max_evidence_chars=1200
    )
    assert len(out) == 1
    row = out[0]
    assert row["unit"] == "section" and row["id"] == "a.md#s" and row["title"] == "S"
    assert row["score"] == 0.4  # mean of members
    assert row["matched_chunks"] == 2
    assert row["evidence"]["chunk_id"] == "c1"


def test_auto_keeps_chunk_below_threshold() -> None:
    sections = {
        "a.md#s": {"path": "a.md", "title": "S", "level": 2, "summary": "sum", "corpus": "c"}
    }
    store = _store({("a.md#s", "fine"): 10}, sections)
    rows = [_chunk("c1", "a.md#s", "a.md", 0.6)]
    out = collapse(
        store, rows, return_unit="auto", auto_merge_threshold=0.5, limit=5, max_evidence_chars=1200
    )
    assert out[0]["unit"] == "chunk" and out[0]["id"] == "c1"


def test_explicit_section_always_collapses_and_file_rolls_up() -> None:
    sections = {
        "a.md#s": {"path": "a.md", "title": "S", "level": 2, "summary": "sum", "corpus": "c"}
    }
    files = {"a.md": {"name": "a.md", "summary": "file sum", "corpus": "c"}}
    store = _store({("a.md#s", "fine"): 100}, sections, files)
    rows = [_chunk("c1", "a.md#s", "a.md", 0.6), _chunk("c9", "a.md#t", "a.md", 0.1)]
    sec = collapse(
        store,
        rows,
        return_unit="section",
        auto_merge_threshold=0.5,
        limit=5,
        max_evidence_chars=1200,
    )
    assert sec[0]["unit"] == "section" and sec[0]["id"] == "a.md#s"
    # a.md#t is not in the parent lookup → its best chunk is returned instead.
    assert sec[1]["unit"] == "chunk" and sec[1]["id"] == "c9"
    fil = collapse(
        store, rows, return_unit="file", auto_merge_threshold=0.5, limit=5, max_evidence_chars=1200
    )
    assert len(fil) == 1 and fil[0]["unit"] == "file" and fil[0]["id"] == "a.md"
    assert fil[0]["matched_chunks"] == 2


def test_file_parented_chunks_collapse_to_file_in_auto() -> None:
    files = {"n.txt": {"name": "n.txt", "summary": None, "corpus": "c"}}
    store = _store({("n.txt", "fine"): 2}, {}, files)
    rows = [
        _chunk("c1", "n.txt", "n.txt", 0.5, label="File", ordinal=0),
        _chunk("c2", "n.txt", "n.txt", 0.3, label="File", ordinal=1),
    ]
    out = collapse(
        store, rows, return_unit="auto", auto_merge_threshold=0.5, limit=5, max_evidence_chars=1200
    )
    assert out[0]["unit"] == "file" and out[0]["id"] == "n.txt"


def test_window_attaches_neighbour_context_only_for_returned_rows() -> None:
    store = _store({}, {})
    rows = [_chunk(f"c{i}", "a.md#s", "a.md", 1.0 - i / 10, ordinal=i) for i in range(5)]
    out = collapse(
        store,
        rows,
        return_unit="chunk",
        auto_merge_threshold=0.5,
        limit=2,
        max_evidence_chars=1200,
        window=1,
    )
    assert len(out) == 2
    assert out[0]["evidence"]["context_after"] == "after text"
    # One neighbour query per returned row, not per candidate.
    assert store.exec_read.call_count == 2


def test_evidence_text_is_clipped() -> None:
    store = _store({}, {})
    row = _chunk("c1", "a.md#s", "a.md", 0.5)
    row["text"] = "x" * 500
    out = collapse(
        store, [row], return_unit="chunk", auto_merge_threshold=0.5, limit=1, max_evidence_chars=100
    )
    assert len(out[0]["evidence"]["text"]) <= 100 and out[0]["evidence"]["text"].endswith("[...]")


def test_limit_applies_after_collapse_and_ordering_is_by_score() -> None:
    sections = {
        "a.md#s": {"path": "a.md", "title": "S", "level": 2, "summary": "", "corpus": "c"},
        "b.md#t": {"path": "b.md", "title": "T", "level": 2, "summary": "", "corpus": "c"},
    }
    store = _store({("a.md#s", "fine"): 1, ("b.md#t", "fine"): 1}, sections)
    rows = [_chunk("c1", "a.md#s", "a.md", 0.2), _chunk("c2", "b.md#t", "b.md", 0.9)]
    out = collapse(
        store, rows, return_unit="auto", auto_merge_threshold=0.5, limit=1, max_evidence_chars=1200
    )
    assert len(out) == 1 and out[0]["id"] == "b.md#t"


def test_empty_rows() -> None:
    assert (
        collapse(
            MagicMock(),
            [],
            return_unit="auto",
            auto_merge_threshold=0.5,
            limit=5,
            max_evidence_chars=10,
        )
        == []
    )
