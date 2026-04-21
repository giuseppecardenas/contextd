"""Unit tests for contextd.mcp.corpus_tools — loader, placeholder parser,
descriptor builder, and dispatch helper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextd.mcp.corpus_tools import (
    CorpusTool,
    build_tool_descriptors,
    dispatch_corpus_tool,
    extract_placeholders,
)
from contextd.mcp.readonly_guard import ReadOnlyGuardError

# ---------------------------------------------------------------------------
# extract_placeholders
# ---------------------------------------------------------------------------


def test_extract_placeholders_single_var() -> None:
    cypher = "MATCH (n:File {path: $path}) RETURN n.path"
    assert extract_placeholders(cypher) == frozenset({"path"})


def test_extract_placeholders_multiple_vars() -> None:
    cypher = "MATCH (n:File {path: $path, corpus: $corpus}) RETURN n.summary"
    result = extract_placeholders(cypher)
    assert result == frozenset({"path", "corpus"})


def test_extract_placeholders_deduplicated() -> None:
    """The same placeholder referenced multiple times yields one entry."""
    cypher = "MATCH (r:Pattern {name: $registry_name}) RETURN $registry_name AS name"
    result = extract_placeholders(cypher)
    assert result == frozenset({"registry_name"})


def test_extract_placeholders_no_vars_returns_empty_set() -> None:
    cypher = "MATCH (n:File) RETURN n.path AS path"
    assert extract_placeholders(cypher) == frozenset()


def test_extract_placeholders_false_positive_on_dollar_in_string() -> None:
    """Known limitation: $identifier-shaped tokens inside string literals are
    matched as placeholders.  This is documented in the module docstring.
    For example, 'pending$status' would register 'status' as a placeholder."""
    # A plain trailing $ not followed by an identifier is NOT matched.
    cypher = 'WHERE n.description CONTAINS "SHA=pending$"'
    assert extract_placeholders(cypher) == frozenset()

    # But $identifier inside a string IS (falsely) matched — documented limit.
    cypher_fp = 'WHERE n.description CONTAINS "pending$status"'
    result = extract_placeholders(cypher_fp)
    assert "status" in result  # false positive — expected per spec


# ---------------------------------------------------------------------------
# build_tool_descriptors — loading and descriptor shape
# ---------------------------------------------------------------------------


def _write_corpus_toml(
    corpora_dir: Path,
    corpus_name: str,
    cypher_entries: dict[str, Path],
    extra: str = "",
) -> Path:
    """Helper: write a minimal corpus TOML pointing at cypher_entries."""
    lines = [
        f'[corpus]\nname = "{corpus_name}"\nroot = "/tmp"',
        "[mcp.tools]",
    ]
    for tool_name, cypher_path in cypher_entries.items():
        lines.append(f'{tool_name} = "{cypher_path}"')
    if extra:
        lines.append(extra)
    toml_path = corpora_dir / f"{corpus_name}.toml"
    toml_path.write_text("\n".join(lines) + "\n")
    return toml_path


def test_build_tool_descriptors_loads_cypher_from_absolute_path(
    tmp_path: Path,
) -> None:
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    cypher_file = tmp_path / "find_file.cypher"
    cypher_file.write_text("MATCH (n:File {path: $path}) RETURN n.path AS path")
    _write_corpus_toml(corpora_dir, "corp", {"find_file": cypher_file})

    descriptors, registry = build_tool_descriptors(tmp_path)

    assert len(descriptors) == 1
    assert descriptors[0].name == "corp.find_file"
    assert "corp.find_file" in registry
    assert registry["corp.find_file"].cypher.startswith("MATCH")


def test_build_tool_descriptors_loads_cypher_from_relative_path_resolves_against_toml_dir(
    tmp_path: Path,
) -> None:
    """A relative cypher_path in the TOML is resolved against the TOML's directory."""
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    # Place the cypher file next to the TOML (relative path "find.cypher").
    cypher_file = corpora_dir / "find.cypher"
    cypher_file.write_text("MATCH (n:File) RETURN n.path AS path")

    # Write TOML with relative path.
    toml_content = (
        '[corpus]\nname = "corp"\nroot = "/tmp"\n[mcp.tools]\nfind_file = "find.cypher"\n'
    )
    (corpora_dir / "corp.toml").write_text(toml_content)

    descriptors, registry = build_tool_descriptors(tmp_path)

    assert len(descriptors) == 1
    assert "corp.find_file" in registry


def test_build_tool_descriptors_skips_missing_cypher_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    missing = tmp_path / "nonexistent.cypher"  # does NOT exist
    _write_corpus_toml(corpora_dir, "corp", {"ghost": missing})

    with caplog.at_level(logging.WARNING):
        descriptors, registry = build_tool_descriptors(tmp_path)

    assert descriptors == []
    assert registry == {}
    assert any("ghost" in rec.message for rec in caplog.records)


def test_build_tool_descriptors_rejects_write_cypher(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    bad_cypher = tmp_path / "bad.cypher"
    bad_cypher.write_text("CREATE (n:File {path: $path}) RETURN n")
    _write_corpus_toml(corpora_dir, "corp", {"dangerous": bad_cypher})

    with caplog.at_level(logging.WARNING):
        descriptors, registry = build_tool_descriptors(tmp_path)

    assert descriptors == []
    assert registry == {}
    assert any(
        "SECURITY" in rec.message and "dangerous" in rec.message and "corp" in rec.message
        for rec in caplog.records
    )


def test_build_tool_descriptors_namespaces_tool_names(tmp_path: Path) -> None:
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    cypher_file = tmp_path / "t.cypher"
    cypher_file.write_text("MATCH (n:File) RETURN n.path")
    _write_corpus_toml(corpora_dir, "my-corpus", {"my_tool": cypher_file})

    descriptors, _ = build_tool_descriptors(tmp_path)

    assert len(descriptors) == 1
    assert descriptors[0].name == "my-corpus.my_tool"


def test_build_tool_descriptors_schema_has_required_placeholders(
    tmp_path: Path,
) -> None:
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    cypher_file = tmp_path / "q.cypher"
    cypher_file.write_text("MATCH (n:File {path: $path, corpus: $corpus}) RETURN n.path")
    _write_corpus_toml(corpora_dir, "corp", {"q": cypher_file})

    descriptors, _ = build_tool_descriptors(tmp_path)

    schema = descriptors[0].inputSchema
    assert schema["required"] == sorted(["path", "corpus"])
    for name in ["path", "corpus"]:
        assert name in schema["properties"]
        assert schema["properties"][name] == {"type": "string"}


def test_build_tool_descriptors_skips_malformed_corpus_toml(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A structurally-malformed TOML (valid syntax, missing required fields)
    does not crash the loader; valid corpora still load."""
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()

    # Write a broken TOML (missing required [corpus] section).
    (corpora_dir / "bad.toml").write_text("[notcorpus]\nfoo = 1\n")

    # Also write a valid corpus with one tool.
    cypher_file = tmp_path / "ok.cypher"
    cypher_file.write_text("MATCH (n:File) RETURN n.path")
    _write_corpus_toml(corpora_dir, "good-corpus", {"ok": cypher_file})

    with caplog.at_level(logging.WARNING):
        descriptors, registry = build_tool_descriptors(tmp_path)

    # Only the valid corpus's tool is registered.
    assert len(descriptors) == 1
    assert "good-corpus.ok" in registry
    # Bad corpus produced a warning.
    assert any("bad.toml" in rec.message for rec in caplog.records)


def test_build_tool_descriptors_skips_syntactically_invalid_toml(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A syntactically invalid TOML (tokenizer error) does not crash the
    loader; valid corpora still load.  This exercises the Fix 1 path where
    tomllib.TOMLDecodeError is wrapped as CorpusConfigError inside
    CorpusConfig.load, so build_tool_descriptors' except CorpusConfigError
    catches it correctly."""
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()

    # Guaranteed tokenizer error — unclosed bracket.
    (corpora_dir / "syntax-error.toml").write_text("[unclosed bracket\n")

    # Also write a valid corpus with one tool.
    cypher_file = tmp_path / "ok.cypher"
    cypher_file.write_text("MATCH (n:File) RETURN n.path")
    _write_corpus_toml(corpora_dir, "good-corpus", {"ok": cypher_file})

    with caplog.at_level(logging.WARNING):
        descriptors, registry = build_tool_descriptors(tmp_path)

    # Only the valid corpus's tool is registered.
    assert len(descriptors) == 1
    assert "good-corpus.ok" in registry
    # The bad corpus produced a warning mentioning the file.
    assert any("syntax-error.toml" in rec.message for rec in caplog.records)


def test_build_tool_descriptors_empty_corpora_dir(tmp_path: Path) -> None:
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()

    descriptors, registry = build_tool_descriptors(tmp_path)

    assert descriptors == []
    assert registry == {}


def test_build_tool_descriptors_no_corpora_dir(tmp_path: Path) -> None:
    """When the corpora directory doesn't exist, return empty lists gracefully."""
    descriptors, registry = build_tool_descriptors(tmp_path)
    assert descriptors == []
    assert registry == {}


# ---------------------------------------------------------------------------
# dispatch_corpus_tool
# ---------------------------------------------------------------------------


def _make_registry(name: str, cypher: str, placeholders: frozenset[str]) -> dict[str, CorpusTool]:
    return {
        name: CorpusTool(
            cypher=cypher,
            placeholders=placeholders,
            corpus_name=name.split(".")[0],
        )
    }


def test_dispatch_tool_calls_exec_read_with_params() -> None:
    cypher = "MATCH (n:File {path: $path}) RETURN n.path AS path"
    registry = _make_registry("corp.find_file", cypher, frozenset({"path"}))
    exec_read = MagicMock(return_value=[{"path": "/a.md"}])

    result = dispatch_corpus_tool("corp.find_file", {"path": "/a.md"}, registry, exec_read)

    exec_read.assert_called_once_with(cypher, {"path": "/a.md"})
    assert result == [{"path": "/a.md"}]


def test_dispatch_tool_missing_required_arg_returns_error() -> None:
    cypher = "MATCH (n:File {path: $path}) RETURN n.path"
    registry = _make_registry("corp.t", cypher, frozenset({"path"}))
    exec_read = MagicMock()

    result = dispatch_corpus_tool("corp.t", {}, registry, exec_read)

    exec_read.assert_not_called()
    assert isinstance(result, dict)
    assert "error" in result
    assert "missing required argument" in result["error"]
    assert "path" in result["error"]


def test_dispatch_tool_unknown_namespaced_tool_raises() -> None:
    registry: dict[str, CorpusTool] = {}
    exec_read = MagicMock()

    with pytest.raises(KeyError, match="Unknown corpus tool"):
        dispatch_corpus_tool("corp.no_such_tool", {}, registry, exec_read)


def test_dispatch_tool_defence_in_depth_rejects_write_in_registry(
    tmp_path: Path,
) -> None:
    """Even if a write Cypher somehow made it into the registry (in-memory
    mutation), dispatch_corpus_tool catches it via assert_read_only."""
    # Bypass the loader by injecting directly.
    bad_cypher = "CREATE (n:File {path: $path}) RETURN n"
    registry: dict[str, CorpusTool] = {
        "corp.bad": CorpusTool(
            cypher=bad_cypher,
            placeholders=frozenset({"path"}),
            corpus_name="corp",
        )
    }
    exec_read = MagicMock()

    with pytest.raises(ReadOnlyGuardError):
        dispatch_corpus_tool("corp.bad", {"path": "/x"}, registry, exec_read)

    exec_read.assert_not_called()


def test_dispatch_tool_multiple_params_all_bound() -> None:
    cypher = "MATCH (n:File {path: $path, corpus: $corpus}) RETURN n.path"
    registry = _make_registry("corp.q", cypher, frozenset({"path", "corpus"}))
    exec_read = MagicMock(return_value=[])

    dispatch_corpus_tool("corp.q", {"path": "/a", "corpus": "c"}, registry, exec_read)

    exec_read.assert_called_once_with(cypher, {"path": "/a", "corpus": "c"})
