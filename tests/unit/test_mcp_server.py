"""Smoke tests for the stdio MCP server entry point."""

from __future__ import annotations


def test_mcp_server_module_imports_cleanly() -> None:
    # Validates the import chain: mcp SDK + contextd.cli + contextd.config +
    # contextd.mcp + contextd.storage.factory all resolve.
    import contextd.mcp_server

    assert callable(contextd.mcp_server.main)
    # The run coroutine is async; verify it's defined.
    assert callable(contextd.mcp_server.run)


def test_mcp_server_main_is_sync_wrapper() -> None:
    import inspect

    import contextd.mcp_server

    assert not inspect.iscoroutinefunction(contextd.mcp_server.main)
    assert inspect.iscoroutinefunction(contextd.mcp_server.run)
