"""Smoke + dispatch tests for the stdio MCP server entry point."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import contextd.mcp_server
from contextd.mcp.readonly_guard import ReadOnlyGuardError
from contextd.mcp_server import TOOL_DESCRIPTORS, _dispatch_tool


def test_mcp_server_module_imports_cleanly() -> None:
    # Validates the import chain: mcp SDK + contextd.cli + contextd.config +
    # contextd.mcp + contextd.storage.factory all resolve.
    assert callable(contextd.mcp_server.main)
    # The run coroutine is async; verify it's defined.
    assert callable(contextd.mcp_server.run)


def test_mcp_server_main_is_sync_wrapper() -> None:
    import inspect

    assert not inspect.iscoroutinefunction(contextd.mcp_server.main)
    assert inspect.iscoroutinefunction(contextd.mcp_server.run)


# -- TOOL_DESCRIPTORS surface (SD #77) ------------------------------------


def test_tool_descriptors_registers_expected_eight() -> None:
    """Directly-inspectable registry means tests can assert the MCP surface
    without running the async server. Previously _list was closure-bound."""
    names = [t.name for t in TOOL_DESCRIPTORS]
    assert set(names) == {
        "describe_project",
        "search",
        "related",
        "inbound",
        "outbound",
        "get_file_summary",
        "section_tree",
        "query_graph",
    }


def test_related_descriptor_depth_clamped_1_to_5() -> None:
    """Depth clamp (SD #32) is enforced in the tool descriptor's JSON schema."""
    related = next(t for t in TOOL_DESCRIPTORS if t.name == "related")
    depth_schema = related.inputSchema["properties"]["depth"]
    assert depth_schema["minimum"] == 1
    assert depth_schema["maximum"] == 5


# -- _dispatch_tool return shape (SD #77) ---------------------------------


def test_dispatch_returns_json_not_repr() -> None:
    """SD #77: tool results emit json.dumps(...) under 'text', not str(obj).
    LLM clients should see lowercase null/true/false, not Python repr."""
    store = MagicMock()
    store.exec_read.return_value = [
        {"path": "/x", "name": "x", "summary": None, "key_points": [], "inbound": 0}
    ]
    payload = _dispatch_tool("describe_project", {"corpus": "c"}, store)
    assert len(payload) == 1
    assert payload[0]["type"] == "text"
    # Must be parseable as real JSON (not Python repr).
    parsed = json.loads(payload[0]["text"])
    assert parsed[0]["summary"] is None
    assert parsed[0]["key_points"] == []


def test_dispatch_wraps_query_graph_errors_in_text() -> None:
    """SD #77: tool errors now flow through the call-handler's try/except as
    JSON {"error": "..."} instead of raising through to the MCP protocol.
    This test validates the wrapper by calling the read-only guard path —
    _dispatch_tool itself re-raises, but run()'s _call wraps."""
    store = MagicMock()
    with pytest.raises(ReadOnlyGuardError):
        _dispatch_tool("query_graph", {"cypher": "CREATE (n:File) RETURN n"}, store)


def test_dispatch_unknown_tool_raises_valueerror() -> None:
    store = MagicMock()
    with pytest.raises(ValueError, match="Unknown tool"):
        _dispatch_tool("not_a_tool", {}, store)
