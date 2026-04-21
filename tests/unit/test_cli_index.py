"""Tests for the `index` CLI command."""

from __future__ import annotations

from importlib import reload
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomli_w
from click.testing import CliRunner

import contextd.cli


def _setup_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    (home / "config.toml").write_text(
        f'[storage]\nbackend = "kuzu"\n\n[storage.kuzu]\ndb_path = "{home}/graph"\n'
    )
    (home / "corpora").mkdir()
    (home / "state").mkdir()
    (home / "prompts").mkdir()
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    return home


def _register_corpus(home: Path, name: str, root: Path) -> None:
    corpus_toml = home / "corpora" / f"{name}.toml"
    data = {
        "corpus": {
            "name": name,
            "root": str(root),
            "include": ["**/*.md"],
            "granularity": "file",
        }
    }
    corpus_toml.write_bytes(tomli_w.dumps(data).encode())


def test_index_estimates_token_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hello world " * 100)
    (corpus_root / "b.md").write_text("foo bar " * 50)
    _register_corpus(home, "docs", corpus_root)
    result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--estimate-only"])
    assert result.exit_code == 0, result.output
    assert "found 2 files" in result.output
    assert "input tokens projected" in result.output


def test_index_errors_when_corpus_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_home(tmp_path, monkeypatch)
    result = CliRunner().invoke(contextd.cli.cli, ["index", "nonexistent"])
    assert result.exit_code == 1
    assert "not registered" in result.output


def test_index_errors_when_no_mode_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")
    _register_corpus(home, "docs", corpus_root)
    result = CliRunner().invoke(contextd.cli.cli, ["index", "docs"])
    assert result.exit_code == 1
    assert "--bootstrap" in result.output or "--incremental" in result.output


def test_index_incremental_prints_not_implemented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")
    _register_corpus(home, "docs", corpus_root)
    # Patch the provider / store factories so we don't hit live APIs.
    # Using contextd.cli.* targets because the imports are inside the function body;
    # after reload, contextd.cli's local scope resolves via the module-level imports
    # made at function call time — patching the factory module directly works.
    with (
        patch("contextd.providers.factory.build_inference_provider") as mock_infer,
        patch("contextd.providers.factory.build_embedding_provider") as mock_embed,
        patch("contextd.storage.factory.build_graph_store") as mock_store,
    ):
        fake_store = MagicMock()
        mock_store.return_value = fake_store
        mock_infer.return_value = MagicMock()
        mock_embed.return_value = MagicMock()
        result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--incremental"])
    assert result.exit_code == 0, result.output
    assert "not yet implemented" in result.output
