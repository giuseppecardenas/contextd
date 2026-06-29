"""stdio MCP server for Contextd.

Spawned by MCP clients (Claude Desktop, Cursor) over stdio. Each tool
is registered with a JSON schema so clients can introspect the
surface. Tool bodies delegate to contextd.mcp.tools.

The home-directory accessor ``contextd_home()`` is imported from
``contextd._paths`` rather than ``contextd.cli`` so the MCP process
doesn't pull in click/rich — SD #69 fixed the Delta-C import coupling
that existed in the initial M7.3 implementation.

Per-corpus tools are registered at startup by scanning
``~/.contextd/corpora/*.toml`` for ``[mcp.tools]`` entries.  Each
entry maps a tool name to a Cypher file; the resulting tools are
namespaced ``<corpus>.<tool>`` to avoid collisions with the 8 generic
tools (which never contain a dot in their names).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from contextd._paths import contextd_home
from contextd.config import Config, SearchConfig
from contextd.mcp import tools
from contextd.mcp.corpus_tools import (
    CorpusTool,
    build_tool_descriptors,
    dispatch_corpus_tool,
)
from contextd.providers.base import EmbeddingProvider
from contextd.providers.factory import ProviderFactoryError, build_embedding_provider
from contextd.storage.base import GraphStore
from contextd.storage.factory import build_graph_store

_GENERIC_TOOL_DESCRIPTORS: list[Tool] = [
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
        description=(
            "Hybrid search (vector + full-text, RRF-fused) over summaries for the "
            "given node kind (default: File). Falls back to full-text when no "
            "embedder is configured or the kind has no vector index."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "description": "Node label to search (default: File).",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum rows to return.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "fulltext", "vector"],
                    "description": "Override ranking mode (default from server config: hybrid).",
                },
            },
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


def _dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    store: GraphStore,
    corpus_registry: dict[str, CorpusTool] | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    search_cfg: SearchConfig | None = None,
) -> Any:
    """Route a tool-call to the right tools.X body.

    Extracted from _call so tests can assert dispatch behaviour without
    spinning up the full async stdio loop.

    Per-corpus tools are dispatched when ``name`` contains a dot (the
    ``<corpus>.<tool>`` namespace separator).  The ``corpus_registry``
    must be provided for those calls; if it is None or the name is not
    registered, a ValueError is raised.

    ``embedder`` and ``search_cfg`` are threaded through to the ``search``
    tool for hybrid ranking; both are optional so tests can dispatch the
    other tools without constructing them (``search`` then runs full-text
    only against ``SearchConfig`` defaults).
    """
    if "." in name:
        reg = corpus_registry or {}
        result = dispatch_corpus_tool(name, arguments, reg, store.exec_read)
        return _text(result)

    match name:
        case "describe_project":
            ov = tools.describe_project(
                store,
                corpus=arguments.get("corpus"),
                n=arguments.get("n", 40),
            )
            return _text(ov.nodes)
        case "search":
            cfg_s = search_cfg or SearchConfig()
            # Server-side knobs (rrf_k/fetch_k/weights) come from config, never
            # from client arguments; only `mode` is a client-facing override.
            return _text(
                tools.search(
                    store,
                    arguments["query"],
                    kind=arguments.get("kind"),
                    limit=arguments.get("limit", 20),
                    embedder=embedder,
                    mode=arguments.get("mode", cfg_s.mode),
                    rrf_k=cfg_s.rrf_k,
                    fetch_k=cfg_s.fetch_k,
                    vector_weight=cfg_s.vector_weight,
                    fulltext_weight=cfg_s.fulltext_weight,
                )
            )
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
        # Build the query-time embedder for hybrid search. A missing API key
        # (or unset api_key_env for a local server) raises ProviderFactoryError;
        # we swallow it and leave embedder=None so the server still starts and
        # `search` degrades to full-text. A *down* local server does not raise
        # here — it raises at first embed(), which tools.search catches.
        embedder: EmbeddingProvider | None
        try:
            embedder = build_embedding_provider(cfg)
        except ProviderFactoryError:
            embedder = None

        server: Server[Any] = Server("contextd")

        # Build the full tool list (generic 8 + per-corpus) and the
        # corpus-tool dispatch registry.  Done after store.connect() so
        # that the home-directory is accessible and any per-corpus TOML
        # parse failures are surfaced before the server loop starts.
        corpus_descriptors, corpus_registry = build_tool_descriptors(home)
        all_descriptors: list[Tool] = _GENERIC_TOOL_DESCRIPTORS + corpus_descriptors

        @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
        async def _list() -> list[Tool]:
            return all_descriptors

        @server.call_tool()  # type: ignore[untyped-decorator]
        async def _call(name: str, arguments: dict[str, Any]) -> Any:
            try:
                return _dispatch_tool(
                    name,
                    arguments,
                    store,
                    corpus_registry,
                    embedder=embedder,
                    search_cfg=cfg.search,
                )
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


def _build_all_tool_descriptors(home: Path) -> tuple[list[Tool], dict[str, CorpusTool]]:
    """Public helper: generic tools + per-corpus tools from *home*.

    Intended for tests and tooling that need the full surface without
    running the async server.
    """
    corpus_descriptors, corpus_registry = build_tool_descriptors(home)
    all_descriptors = _GENERIC_TOOL_DESCRIPTORS + corpus_descriptors
    return all_descriptors, corpus_registry
