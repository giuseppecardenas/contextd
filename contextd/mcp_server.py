"""stdio MCP server for Contextd.

Spawned by MCP clients (Claude Desktop, Cursor) over stdio. Each tool
is registered with a JSON schema so clients can introspect the
surface. Tool bodies delegate to contextd.mcp.tools.

The home-directory accessor ``contextd_home()`` is imported from
``contextd._paths`` rather than ``contextd.cli`` so the MCP process
doesn't pull in click/rich — SD #69 fixed the Delta-C import coupling
that existed in the initial M7.3 implementation.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from contextd._paths import contextd_home
from contextd.config import Config
from contextd.mcp import tools
from contextd.storage.base import GraphStore
from contextd.storage.factory import build_graph_store

TOOL_DESCRIPTORS: list[Tool] = [
    Tool(
        name="describe_project",
        description="Compact project primer — top-N most-cited File nodes with summaries.",
        inputSchema={
            "type": "object",
            "properties": {
                "corpus": {"type": "string"},
                "n": {"type": "integer", "default": 40},
            },
        },
    ),
    Tool(
        name="search",
        description="Full-text search over summaries for ``kind`` (defaults to File).",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    Tool(
        name="related",
        description="Outbound+inbound traversal within N hops (1-5).",
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "depth": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["node_id"],
        },
    ),
    Tool(
        name="inbound",
        description="What cites this node?",
        inputSchema={
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    ),
    Tool(
        name="outbound",
        description="What does this node cite?",
        inputSchema={
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    ),
    Tool(
        name="get_file_summary",
        description="Summary + key points for a single file.",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    Tool(
        name="section_tree",
        description="Outline of a file (section-granular corpora).",
        inputSchema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    ),
    Tool(
        name="query_graph",
        description="Raw read-only Cypher escape hatch.",
        inputSchema={
            "type": "object",
            "properties": {"cypher": {"type": "string"}},
            "required": ["cypher"],
        },
    ),
]


def _text(obj: Any) -> list[dict[str, str]]:
    """MCP tool result shape — JSON-serialised payload under 'text'.

    Previously we used ``str(obj)`` which emits Python repr (single-quoted,
    Python True/None, etc.) — LLM clients couldn't parse it as structured
    data. ``json.dumps(..., default=str)`` renders real JSON and falls back
    to ``str()`` for non-serialisable objects (datetimes, Path).
    """
    return [{"type": "text", "text": json.dumps(obj, default=str)}]


def _dispatch_tool(name: str, arguments: dict[str, Any], store: GraphStore) -> Any:
    """Route a tool-call to the right tools.X body.

    Extracted from _call so tests can assert dispatch behaviour without
    spinning up the full async stdio loop.
    """
    match name:
        case "describe_project":
            ov = tools.describe_project(
                store,
                corpus=arguments.get("corpus"),
                n=arguments.get("n", 40),
            )
            return _text(ov.nodes)
        case "search":
            return _text(tools.search(store, **arguments))
        case "related":
            return _text(tools.related(store, **arguments))
        case "inbound":
            return _text(tools.inbound(store, **arguments))
        case "outbound":
            return _text(tools.outbound(store, **arguments))
        case "get_file_summary":
            return _text(tools.get_file_summary(store, **arguments))
        case "section_tree":
            return _text(tools.section_tree(store, **arguments))
        case "query_graph":
            return _text(tools.query_graph(store, arguments["cypher"]))
        case _:
            raise ValueError(f"Unknown tool: {name}")


async def run() -> None:
    home = contextd_home()
    cfg = (
        Config.load(home / "config.toml")
        if (home / "config.toml").exists()
        else Config.load_default()
    )
    store = build_graph_store(cfg)
    store.connect()
    try:
        server: Server[Any] = Server("contextd")

        @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
        async def _list() -> list[Tool]:
            return TOOL_DESCRIPTORS

        @server.call_tool()  # type: ignore[untyped-decorator]
        async def _call(name: str, arguments: dict[str, Any]) -> Any:
            try:
                return _dispatch_tool(name, arguments, store)
            except Exception as exc:
                # Render the error as the tool's text payload so the MCP
                # client sees a structured response instead of a protocol
                # exception. Read-only-guard rejections, malformed args,
                # and backend errors all flow through here.
                return _text({"error": f"{type(exc).__name__}: {exc}"})

        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        store.close()


def main() -> None:
    import asyncio

    asyncio.run(run())


if __name__ == "__main__":
    main()
