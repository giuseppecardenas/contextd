from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from contextd.search.expand import attach_context, expand_chunk, neighbours


def _store(rows: list[dict[str, Any]]) -> MagicMock:
    store = MagicMock()
    store.exec_read.return_value = rows
    return store


def test_neighbours_splits_before_and_after_by_ordinal() -> None:
    store = _store([{"ordinal": 1, "text": "b"}, {"ordinal": 3, "text": "a"}])
    before, after = neighbours(store, parent_id="p", profile="fine", ordinal=2, window=1)
    assert before == ["b"] and after == ["a"]
    params = store.exec_read.call_args.args[1]
    assert (params["lo"], params["hi"], params["ord"]) == (1, 3, 2)


def test_neighbours_window_zero_is_free() -> None:
    store = MagicMock()
    assert neighbours(store, parent_id="p", profile="fine", ordinal=0, window=0) == ([], [])
    store.exec_read.assert_not_called()


def test_attach_context_in_place_and_skips_incomplete_rows() -> None:
    store = _store([{"ordinal": 0, "text": "prev"}])
    rows = [{"parent_id": "p", "profile": "fine", "ordinal": 1}, {"id": "no-parent"}]
    attach_context(store, rows, window=1)
    assert rows[0]["context_before"] == "prev" and rows[0]["context_after"] == ""
    assert "context_before" not in rows[1]


def test_expand_chunk_returns_row_with_neighbours() -> None:
    store = MagicMock()
    store.exec_read.side_effect = [
        [
            {
                "id": "c",
                "path": "a.md",
                "parent_id": "a.md#s",
                "parent_label": "Section",
                "profile": "fine",
                "ordinal": 2,
                "kind": "prose",
                "text": "t",
                "prefix": "Doc > S",
                "start_line": 3,
                "end_line": 4,
                "parent_summary": "sum",
                "parent_title": "S",
            }
        ],
        [{"ordinal": 1, "text": "b"}, {"ordinal": 3, "text": "a"}],
    ]
    row = expand_chunk(store, "c", window=1)
    assert row is not None
    assert row["context_before"] == ["b"] and row["context_after"] == ["a"]
    assert row["parent_summary"] == "sum"


def test_expand_chunk_missing() -> None:
    assert expand_chunk(_store([]), "nope") is None
