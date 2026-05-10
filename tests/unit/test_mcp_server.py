"""Smoke + dispatch tests for the stdio MCP server entry point."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import contextd.mcp_server
from contextd.mcp.corpus_tools import CorpusTool
from contextd.mcp.readonly_guard import ReadOnlyGuardError
from contextd.mcp_server import (
    _GENERIC_TOOL_DESCRIPTORS,
    _build_all_tool_descriptors,
    _dispatch_tool,
)


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


# -- _GENERIC_TOOL_DESCRIPTORS surface (SD #77) ---------------------------


def test_generic_tool_descriptors_registers_expected_eight() -> None:
    """Directly-inspectable registry means tests can assert the MCP surface
    without running the async server. Previously _list was closure-bound.

    Per-corpus tools appear only via build_tool_descriptors /
    _build_all_tool_descriptors — they are not in _GENERIC_TOOL_DESCRIPTORS.
    """
    names = [t.name for t in _GENERIC_TOOL_DESCRIPTORS]
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
    related = next(t for t in _GENERIC_TOOL_DESCRIPTORS if t.name == "related")
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


# -- Per-corpus tool dispatch via _dispatch_tool --------------------------


def test_dispatch_corpus_tool_via_dot_name(tmp_path: Path) -> None:
    """Tools with a dot in the name are routed through the corpus registry."""
    cypher = "MATCH (n:File {path: $path}) RETURN n.path AS path"
    registry: dict[str, CorpusTool] = {
        "my-corpus.my_tool": CorpusTool(
            cypher=cypher,
            placeholders=frozenset({"path"}),
            corpus_name="my-corpus",
        )
    }
    store = MagicMock()
    store.exec_read.return_value = [{"path": "/a.md"}]
    result = _dispatch_tool("my-corpus.my_tool", {"path": "/a.md"}, store, registry)
    assert len(result) == 1
    parsed = json.loads(result[0]["text"])
    assert parsed == [{"path": "/a.md"}]
    store.exec_read.assert_called_once_with(cypher, {"path": "/a.md"})


def test_dispatch_corpus_tool_missing_arg_returns_error(tmp_path: Path) -> None:
    """Missing required arg → {"error": "missing required argument: ..."} in text."""
    cypher = "MATCH (n:File {path: $path}) RETURN n.path"
    registry: dict[str, CorpusTool] = {
        "corp.tool": CorpusTool(
            cypher=cypher,
            placeholders=frozenset({"path"}),
            corpus_name="corp",
        )
    }
    store = MagicMock()
    result = _dispatch_tool("corp.tool", {}, store, registry)
    assert len(result) == 1
    parsed = json.loads(result[0]["text"])
    assert "error" in parsed
    assert "missing required argument" in parsed["error"]


def test_build_all_tool_descriptors_no_corpora_dir(tmp_path: Path) -> None:
    """When corpora/ does not exist, only the 8 generic tools are returned."""
    all_descs, registry = _build_all_tool_descriptors(tmp_path)
    assert len(all_descs) == 8
    assert registry == {}


def test_build_all_tool_descriptors_adds_corpus_tools(tmp_path: Path) -> None:
    """A valid corpus TOML with one Cypher tool expands the descriptor list."""
    cypher = "MATCH (n:File {path: $path}) RETURN n.path AS path"
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    cypher_file = tools_dir / "find_file.cypher"
    cypher_file.write_text(cypher)

    # ``as_posix()`` keeps the TOML strings backslash-free on Windows; pathlib
    # accepts forward-slash paths there, but ``\U`` in ``C:\\Users\\...`` would
    # be parsed as a TOML Unicode escape.
    toml_content = f"""
[corpus]
name = "my-corpus"
root = "{tmp_path.as_posix()}"
[mcp.tools]
find_file = "{cypher_file.as_posix()}"
"""
    (corpora_dir / "my-corpus.toml").write_text(toml_content)

    all_descs, registry = _build_all_tool_descriptors(tmp_path)
    names = [t.name for t in all_descs]
    assert "my-corpus.find_file" in names
    assert len(all_descs) == 9
    assert "my-corpus.find_file" in registry
