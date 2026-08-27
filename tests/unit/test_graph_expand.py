"""Graph-first expansion of search hits (contextd.search.graph_expand + tools.search)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from contextd.mcp import tools
from contextd.search.expand import attach_evidence_context
from contextd.search.fusion import fuse_rankers
from contextd.search.graph_expand import (
    MAX_SEEDS,
    Seed,
    expand_units,
    seeds_from_rows,
)

# --- seeds -------------------------------------------------------------------


def test_seeds_from_rows_uses_unit_id_and_chunk_parent_deduped() -> None:
    rows: list[dict[str, Any]] = [
        {"unit": "section", "id": "a.md#s1"},
        {"unit": "chunk", "id": "a.md#s1~fine~0", "parent_id": "a.md#s1"},  # dup of seed 0
        # A File row seeds from the Section its best chunk lives in, not the file.
        {"unit": "file", "id": "b.md", "evidence": {"parent_id": "b.md#intro"}},
        {"unit": "file", "id": "b.lua", "evidence": {"chunk_id": "x"}},
        {"unit": "chunk", "id": "c.md~fine~2", "parent_id": "c.md"},
        {"score": 1.0},  # flat row: no unit → skipped
        # A fused chunk row (no unit key) seeds with its parent like a chunk row.
        {"id": "e.md#z~fine~1", "parent_id": "e.md#z", "path": "e.md"},
        {"unit": "section", "id": "d.md#x"},
    ]
    assert seeds_from_rows(rows, n=3) == [
        Seed("a.md#s1", 0),
        Seed("b.md#intro", 1),
        Seed("b.lua", 2),
    ]
    assert seeds_from_rows(rows, n=10)[3:] == [
        Seed("c.md", 3),
        Seed("e.md#z", 4),
        Seed("d.md#x", 5),
    ]
    assert seeds_from_rows([], n=3) == []


# --- expand_units -------------------------------------------------------------


def _walk_row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "b.lua",
        "path": "b.lua",
        "labels": ["File"],
        "title": None,
        "name": "b.lua",
        "summary": "registers settlements",
        "corpus": "c",
        "score": 0.7,
        "via": ["register_settlement_type", "FR-STL-001"],
        "seeds": ["a.md#s1"],
    }
    base.update(over)
    return base


def test_expand_units_binds_seeds_corpus_limit_and_shapes_rows() -> None:
    store = MagicMock()
    store.exec_read.return_value = [
        _walk_row(),
        _walk_row(id="a.md#s2", path="a.md", labels=["Section"], title="S2", name=None, via=[]),
    ]
    rows = expand_units(
        store, [Seed("a.md#s1", 0), Seed("z.md", 1)], corpus="c", limit=7, to_file=True
    )

    cypher, params = store.exec_read.call_args[0]
    assert "UNWIND $seeds AS seed" in cypher
    assert params["seeds"] == [{"id": "a.md#s1", "rank": 0}, {"id": "z.md", "rank": 1}]
    assert params["corpus"] == "c" and params["limit"] == 7 and params["to_file"] is True
    # Only inferred/manual unit-unit edges count; structural ones are excluded.
    assert "r.origin IN ['inferred', 'manual']" in cypher
    # Hub damping and same-path exclusion are part of the query text.
    assert "1.0 / log(2 + COUNT { (e)--() })" in cypher
    assert "u.path <> s.path" in cypher

    assert rows[0] == {
        "unit": "file",
        "id": "b.lua",
        "path": "b.lua",
        "name": "b.lua",
        "summary": "registers settlements",
        "corpus": "c",
        "score": 0.7,
        "via": {"entities": ["register_settlement_type", "FR-STL-001"], "seeds": ["a.md#s1"]},
    }
    assert rows[1]["unit"] == "section" and rows[1]["title"] == "S2"
    assert rows[1]["via"] == {"entities": [], "seeds": ["a.md#s1"]}
    assert "name" not in rows[1]


def test_expand_units_no_seeds_or_zero_limit_skips_the_query() -> None:
    store = MagicMock()
    assert expand_units(store, [], corpus=None, limit=5) == []
    assert expand_units(store, [Seed("a", 0)], corpus=None, limit=0) == []
    store.exec_read.assert_not_called()


def test_expand_units_caps_seeds_and_passes_null_corpus() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    seeds = [Seed(f"s{i}", i) for i in range(MAX_SEEDS + 5)]
    expand_units(store, seeds, corpus=None, limit=5)
    _, params = store.exec_read.call_args[0]
    assert len(params["seeds"]) == MAX_SEEDS
    assert params["corpus"] is None and params["to_file"] is False


def test_expand_units_swallows_backend_errors() -> None:
    store = MagicMock()
    store.exec_read.side_effect = RuntimeError("bolt down")
    assert expand_units(store, [Seed("a", 0)], corpus="c", limit=5) == []


# --- fusion key override -------------------------------------------------------


def test_fuse_rankers_key_prop_merges_mixed_unit_rows() -> None:
    direct = [{"node": {"unit": "section", "id": "a#1", "path": "a"}, "score": 0.9}]
    graph = [
        {"node": {"unit": "file", "id": "b", "path": "b"}, "score": 0.5},
        {"node": {"unit": "section", "id": "a#1", "path": "a"}, "score": 0.4},
    ]
    out = fuse_rankers([(direct, 1.0), (graph, 1.0)], label="Section", limit=5, key_prop="id")
    # a#1 is in both lists → summed contribution beats b's single rank-1 entry.
    assert [r["id"] for r in out] == ["a#1", "b"]
    assert out[0]["unit"] == "section"  # first-seen (direct) node shape kept


def test_graph_weight_scale_trail_interleave_lead() -> None:
    direct = [{"node": {"id": f"d{i}"}, "score": 1.0} for i in range(5)]
    graph = [{"node": {"id": f"g{i}"}, "score": 1.0} for i in range(5)]

    def order(weight: float) -> list[str]:
        return [
            str(r["id"])
            for r in fuse_rankers(
                [(direct, 1.0), (graph, weight)], label="Section", limit=10, key_prop="id"
            )
        ]

    assert order(0.5) == ["d0", "d1", "d2", "d3", "d4", "g0", "g1", "g2", "g3", "g4"]
    assert order(1.0) == ["d0", "g0", "d1", "g1", "d2", "g2", "d3", "g3", "d4", "g4"]
    assert order(2.0) == ["g0", "g1", "g2", "g3", "g4", "d0", "d1", "d2", "d3", "d4"]


# --- attach_evidence_context ---------------------------------------------------


def test_attach_evidence_context_fills_only_rows_with_chunk_evidence() -> None:
    store = MagicMock()
    store.exec_read.return_value = [
        {"ordinal": 1, "text": "before " * 10, "pivot": 2},
        {"ordinal": 3, "text": "after", "pivot": 2},
    ]
    rows: list[dict[str, Any]] = [
        {"id": "a#1", "evidence": {"chunk_id": "a#1~fine~2", "text": "hit"}},
        {"id": "g", "via": {"entities": [], "seeds": []}},  # graph-only: no evidence
        {"id": "done", "evidence": {"chunk_id": "x", "context_before": "kept"}},
    ]
    attach_evidence_context(store, rows, window=1, max_chars=20)
    store.exec_read.assert_called_once()
    _, params = store.exec_read.call_args[0]
    assert params == {"id": "a#1~fine~2", "w": 1}
    ev = rows[0]["evidence"]
    assert ev["context_after"] == "after"
    assert ev["context_before"].endswith(" [...]") and len(ev["context_before"]) <= 20
    assert "evidence" not in rows[1]
    assert rows[2]["evidence"]["context_before"] == "kept"
    attach_evidence_context(store, rows, window=0, max_chars=20)
    store.exec_read.assert_called_once()  # window 0 → nothing more issued


# --- search pipeline ------------------------------------------------------------


def _chunk_hit(parent: str, path: str, ordinal: int, score: float) -> dict[str, Any]:
    return {
        "node": {
            "id": f"{parent}~fine~{ordinal}",
            "path": path,
            "parent_id": parent,
            "parent_label": "Section" if "#" in parent else "File",
            "profile": "fine",
            "ordinal": ordinal,
            "text": f"text {ordinal}",
            "start_line": ordinal,
            "end_line": ordinal + 1,
            "corpus": "c",
            "embedding": [0.0],
        },
        "score": score,
    }


def _search_store(graph_rows: list[dict[str, Any]] | Exception) -> MagicMock:
    """A store whose rankers return two section-parented chunks and whose walk
    returns ``graph_rows`` (or raises)."""
    store = MagicMock()
    hits = [_chunk_hit("a.md#s1", "a.md", 0, 0.9), _chunk_hit("a.md#s2", "a.md", 0, 0.8)]
    store.full_text_search.return_value = hits
    store.vector_search.return_value = []

    def exec_read(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if "UNWIND $seeds" in cypher:
            if isinstance(graph_rows, Exception):
                raise graph_rows
            return graph_rows
        if "MATCH (s:Section) WHERE s.id IN $ids" in cypher:
            assert params is not None
            return [
                {"id": i, "path": "a.md", "title": i, "level": 2, "summary": "sum", "corpus": "c"}
                for i in params["ids"]
            ]
        if "MATCH (c:Chunk {id: $id})" in cypher:
            return [{"ordinal": 1, "text": "nbr", "pivot": 0}]
        return []

    store.exec_read.side_effect = exec_read
    return store


def test_search_expand_none_never_walks() -> None:
    store = _search_store([])
    rows = tools.search(store, "q", mode="fulltext", return_unit="section", window=0)
    assert [r["id"] for r in rows] == ["a.md#s1", "a.md#s2"]
    assert not any("UNWIND $seeds" in c.args[0] for c in store.exec_read.call_args_list)


def test_search_expand_units_fuses_walk_rows_with_via_and_keeps_direct_evidence() -> None:
    walk = [_walk_row(), _walk_row(id="a.md#s2", path="a.md", labels=["Section"], via=["FR-1"])]
    store = _search_store(walk)
    rows = tools.search(
        store,
        "q",
        mode="fulltext",
        return_unit="section",
        window=1,
        corpus="c",
        limit=5,
        expand="units",
        expand_seeds=2,
    )
    walk_call = next(c for c in store.exec_read.call_args_list if "UNWIND $seeds" in c.args[0])
    assert walk_call.args[1]["seeds"] == [
        {"id": "a.md#s1", "rank": 0},
        {"id": "a.md#s2", "rank": 1},
    ]
    assert walk_call.args[1]["corpus"] == "c" and walk_call.args[1]["to_file"] is False

    by_id = {str(r["id"]): r for r in rows}
    # The seeds head the graph ranker, so s1 keeps rank 1 even though the walk
    # returned s2 (its neighbour) and not s1; the graph-only file trails.
    assert [r["id"] for r in rows] == ["a.md#s1", "a.md#s2", "b.lua"]
    # s2 keeps its direct evidence (with neighbour context attached last) and
    # gains the via block; the graph-only file has via but no evidence.
    assert by_id["a.md#s2"]["evidence"]["text"] == "text 0"
    assert by_id["a.md#s2"]["evidence"]["context_after"] == "nbr"
    assert by_id["a.md#s2"]["via"] == {"entities": ["FR-1"], "seeds": ["a.md#s1"]}
    assert "via" not in by_id["a.md#s1"]
    assert by_id["b.lua"]["unit"] == "file" and "evidence" not in by_id["b.lua"]
    assert by_id["b.lua"]["via"]["entities"] == ["register_settlement_type", "FR-STL-001"]


def test_search_expand_units_chunk_rows_seed_from_parent_and_truncate_to_limit() -> None:
    store = _search_store([_walk_row()])
    rows = tools.search(
        store, "q", mode="fulltext", return_unit="chunk", window=0, limit=1, expand="units"
    )
    walk_call = next(c for c in store.exec_read.call_args_list if "UNWIND $seeds" in c.args[0])
    assert walk_call.args[1]["seeds"][0] == {"id": "a.md#s1", "rank": 0}
    assert len(rows) == 1 and rows[0]["unit"] == "chunk"


def test_search_expand_units_file_rows_seed_from_matched_section_and_roll_up() -> None:
    store = _search_store([_walk_row()])
    tools.search(store, "q", mode="fulltext", return_unit="file", window=0, limit=5, expand="units")
    walk_call = next(c for c in store.exec_read.call_args_list if "UNWIND $seeds" in c.args[0])
    # Both chunks collapse to one File row (a.md), but seeding follows the
    # fused chunk order, so both Sections seed; the walk targets Files only.
    assert walk_call.args[1]["seeds"] == [
        {"id": "a.md#s1", "rank": 0},
        {"id": "a.md#s2", "rank": 1},
    ]
    assert walk_call.args[1]["to_file"] is True


def test_search_expand_units_walk_failure_degrades_to_direct_rows() -> None:
    store = _search_store(RuntimeError("bolt down"))
    rows = tools.search(
        store, "q", mode="fulltext", return_unit="section", window=0, limit=1, expand="units"
    )
    assert [r["id"] for r in rows] == ["a.md#s1"]


def test_search_expand_ignored_for_non_chunk_kinds() -> None:
    store = MagicMock()
    store.full_text_search.return_value = [{"node": {"path": "f.md"}, "score": 1.0}]
    rows = tools.search(store, "q", kind="File", mode="fulltext", expand="units")
    assert rows == [{"path": "f.md", "score": 1.0}]
    store.exec_read.assert_not_called()
