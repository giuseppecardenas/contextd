"""Tests for add-corpus and list-corpora CLI commands."""

from __future__ import annotations

import tomllib
from importlib import reload
from pathlib import Path

import pytest
from click.testing import CliRunner

import contextd.cli


def _setup_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    (home / "config.toml").write_text(
        '[storage]\nbackend = "kuzu"\n\n[storage.kuzu]\ndb_path = "' + str(home) + '/graph"\n'
    )
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    return home


def test_add_corpus_writes_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "my-corpus"
    corpus_dir.mkdir()
    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "notes", "--granularity", "file"],
    )
    assert result.exit_code == 0, result.output
    toml_path = home / "corpora" / "notes.toml"
    assert toml_path.exists()
    data = tomllib.loads(toml_path.read_text())
    assert data["corpus"]["name"] == "notes"
    assert data["corpus"]["root"] == str(corpus_dir.resolve())
    assert data["corpus"]["include"] == ["**/*.md"]
    assert data["corpus"]["granularity"] == "file"
    # heading_min_level / heading_max_level only present on section granularity.
    assert "heading_min_level" not in data["corpus"]
    assert "notes" in result.output


def test_add_corpus_section_granularity_adds_heading_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "big-doc"
    corpus_dir.mkdir()
    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "big", "--granularity", "section"],
    )
    assert result.exit_code == 0
    toml_path = home / "corpora" / "big.toml"
    data = tomllib.loads(toml_path.read_text())
    assert data["corpus"]["granularity"] == "section"
    assert data["corpus"]["heading_min_level"] == 2
    assert data["corpus"]["heading_max_level"] == 4


def test_add_corpus_default_name_is_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "auto-named"
    corpus_dir.mkdir()
    result = CliRunner().invoke(contextd.cli.cli, ["add-corpus", str(corpus_dir)])
    assert result.exit_code == 0
    assert (home / "corpora" / "auto-named.toml").exists()


def test_add_corpus_refuses_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "dup"
    corpus_dir.mkdir()
    runner = CliRunner()
    first = runner.invoke(contextd.cli.cli, ["add-corpus", str(corpus_dir)])
    assert first.exit_code == 0
    second = runner.invoke(contextd.cli.cli, ["add-corpus", str(corpus_dir)])
    assert second.exit_code == 0  # warns but doesn't error
    assert "already registered" in second.output


def test_list_corpora_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    result = CliRunner().invoke(contextd.cli.cli, ["list-corpora"])
    assert result.exit_code == 0
    assert "no corpora" in result.output.lower()


def test_list_corpora_shows_registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "first"
    corpus_dir.mkdir()
    CliRunner().invoke(contextd.cli.cli, ["add-corpus", str(corpus_dir)])
    result = CliRunner().invoke(contextd.cli.cli, ["list-corpora"])
    assert result.exit_code == 0
    assert "first" in result.output
