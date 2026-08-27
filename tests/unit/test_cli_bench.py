"""Tests for the ``contextd bench`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import tomli_w
from click.testing import CliRunner

import contextd.cli
from contextd.bench.metrics import QueryScore
from contextd.bench.report import save_report
from contextd.bench.run import BenchReport

_SPEC = """
[[queries]]
q = "note 3"
expect = [{ path = "note-3.md", anchor = "note-3" }]

[[queries]]
q = "note 7"
expect = [{ path = "note-7.md" }]
k = 2
"""


def _setup_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    (home / "config.toml").write_text(
        f'[storage]\nbackend = "neo4j"\n\n[storage.neo4j]\n'
        f'docker_compose_file = "{home.as_posix()}/docker-compose.yml"\n',
        encoding="utf-8",
    )
    (home / "corpora").mkdir()
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    # Rich wraps at 80 columns under CliRunner, which would split paths and
    # table cells across lines and break substring assertions.
    monkeypatch.setenv("COLUMNS", "250")
    return home


def _register_corpus(home: Path, name: str, root: Path, *, spec: str | None = _SPEC) -> Path:
    data: dict[str, Any] = {
        "corpus": {
            "name": name,
            "root": root.as_posix(),
            "include": ["**/*.md"],
            "granularity": "section",
        },
        "chunking": {
            "profiles": [
                {"name": "fine", "max_tokens": 256, "min_tokens": 48, "weight": 1.0},
                {"name": "coarse", "max_tokens": 1024, "min_tokens": 200, "weight": 0.5},
            ]
        },
    }
    (home / "corpora" / f"{name}.toml").write_bytes(tomli_w.dumps(data).encode())
    root.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (root / ".contextd").mkdir(exist_ok=True)
        (root / ".contextd" / "bench.toml").write_text(spec, encoding="utf-8")
    return root


def _fake_report(store: Any = None, spec: Any = None, **kwargs: Any) -> BenchReport:
    """Stand-in for ``run_bench`` with its positional ``(store, spec)`` shape."""
    return BenchReport(
        config={
            "corpus": kwargs.get("corpus"),
            "profiles": kwargs.get("profiles"),
            "return_unit": "auto",
            "k": 5,
        },
        scores=[QueryScore("note 3", 1.0, 0.5, 1.0, None)],
        summary={
            "recall": 1.0,
            "precision": 0.5,
            "mrr": 1.0,
            "iou": None,
            "queries": 1,
            "latency_ms": 2.0,
        },
        latencies_ms=[2.0],
    )


def test_bench_missing_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    result = CliRunner().invoke(contextd.cli.cli, ["bench", "ghost"])
    assert result.exit_code == 1
    assert "not registered" in result.output
    assert "Traceback" not in result.output


def test_bench_requires_corpus_without_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_home(tmp_path, monkeypatch)
    result = CliRunner().invoke(contextd.cli.cli, ["bench"])
    assert result.exit_code == 1
    assert "CORPUS is required" in result.output


def test_bench_missing_spec_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    _register_corpus(home, "notes", tmp_path / "notes", spec=None)
    result = CliRunner().invoke(contextd.cli.cli, ["bench", "notes"])
    assert result.exit_code == 1
    assert "bench spec not found" in result.output
    assert "bench.toml" in result.output


def test_bench_unknown_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    _register_corpus(home, "notes", tmp_path / "notes")
    result = CliRunner().invoke(contextd.cli.cli, ["bench", "notes", "--profiles", "fine,huge"])
    assert result.exit_code == 1
    assert "unknown chunk profile(s) huge" in result.output
    assert "fine, coarse" in result.output


def test_bench_run_with_patched_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    _register_corpus(home, "notes", tmp_path / "notes")
    out = tmp_path / "out" / "bench.json"
    with (
        patch("contextd.storage.factory.build_graph_store") as mock_store_factory,
        patch(
            "contextd.providers.factory.build_embedding_provider",
            side_effect=RuntimeError("no key"),
        ),
        patch("contextd.bench.run.run_bench", side_effect=_fake_report) as mock_run,
    ):
        fake_store = mock_store_factory.return_value
        result = CliRunner().invoke(
            contextd.cli.cli,
            [
                "bench",
                "notes",
                "--profiles",
                "fine",
                "--profiles",
                "fine, coarse",
                "--return-unit",
                "section",
                "--k",
                "7",
                "--json",
                str(out),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "embedding provider unavailable" in result.output
    assert "full-text only" in result.output
    assert "2 queries" in result.output
    assert "fine/auto@5" in result.output and "fine,coarse/auto@5" in result.output
    assert "saved 2 report(s)" in result.output

    assert fake_store.connect.called and fake_store.close.called
    assert mock_run.call_count == 2
    first, second = mock_run.call_args_list
    assert first.kwargs["profiles"] == ["fine"]
    assert second.kwargs["profiles"] == ["fine", "coarse"]
    for call in (first, second):
        assert call.args[0] is fake_store
        assert call.kwargs["embedder"] is None
        assert call.kwargs["corpus"] == "notes"
        assert call.kwargs["return_unit"] == "section"
        assert call.kwargs["k"] == 7
        assert call.kwargs["profile_weights"] == {"fine": 1.0, "coarse": 0.5}
        assert [q.q for q in call.args[1].queries] == ["note 3", "note 7"]

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved["reports"]) == 2
    assert saved["reports"][1]["config"]["profiles"] == ["fine", "coarse"]


def test_bench_default_profiles_and_queries_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    _register_corpus(home, "notes", tmp_path / "notes", spec=None)
    custom = tmp_path / "custom.toml"
    custom.write_text(_SPEC, encoding="utf-8")
    with (
        patch("contextd.storage.factory.build_graph_store"),
        patch("contextd.providers.factory.build_embedding_provider") as mock_embed,
        patch("contextd.bench.run.run_bench", side_effect=_fake_report) as mock_run,
    ):
        result = CliRunner().invoke(contextd.cli.cli, ["bench", "notes", "--queries", str(custom)])
    assert result.exit_code == 0, result.output
    assert mock_run.call_count == 1
    assert mock_run.call_args.kwargs["profiles"] is None
    assert mock_run.call_args.kwargs["embedder"] is mock_embed.return_value
    assert mock_run.call_args.kwargs["return_unit"] == "auto"  # config default
    assert str(custom) in result.output


def test_bench_run_failure_is_click_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    _register_corpus(home, "notes", tmp_path / "notes")
    with (
        patch("contextd.storage.factory.build_graph_store") as mock_store_factory,
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.bench.run.run_bench", side_effect=RuntimeError("bolt down")),
    ):
        result = CliRunner().invoke(contextd.cli.cli, ["bench", "notes"])
    assert result.exit_code == 1
    assert "bench failed: bolt down" in result.output
    assert "Traceback" not in result.output
    assert mock_store_factory.return_value.close.called


def test_bench_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    a = _fake_report(profiles=["fine"], corpus="notes")
    b = _fake_report(profiles=["coarse"], corpus="notes")
    b.summary["recall"] = 0.5
    a_path, b_path = tmp_path / "a.json", tmp_path / "b.json"
    save_report(a, a_path)
    save_report(b, b_path)
    with patch("contextd.storage.factory.build_graph_store") as mock_store_factory:
        result = CliRunner().invoke(
            contextd.cli.cli, ["bench", "--compare", str(a_path), str(b_path)]
        )
    assert result.exit_code == 0, result.output
    assert not mock_store_factory.called  # compare never touches the backend
    assert "-0.500" in result.output
    assert "A: fine/auto@5" in result.output


def test_bench_compare_bad_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    good = tmp_path / "a.json"
    save_report(_fake_report(profiles=["fine"]), good)
    bad = tmp_path / "b.json"
    bad.write_text("{}", encoding="utf-8")
    result = CliRunner().invoke(contextd.cli.cli, ["bench", "--compare", str(good), str(bad)])
    assert result.exit_code == 1
    assert "b.json" in result.output
    assert "Traceback" not in result.output


def test_bench_help(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    result = CliRunner().invoke(contextd.cli.cli, ["bench", "--help"])
    assert result.exit_code == 0
    for flag in ("--queries", "--profiles", "--return-unit", "--k", "--json", "--compare"):
        assert flag in result.output


def test_bench_expand_flags_thread_to_run_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    _register_corpus(home, "notes", tmp_path / "notes")
    with (
        patch("contextd.storage.factory.build_graph_store"),
        patch(
            "contextd.providers.factory.build_embedding_provider",
            side_effect=RuntimeError("no key"),
        ),
        patch("contextd.bench.run.run_bench", side_effect=_fake_report) as mock_run,
    ):
        result = CliRunner().invoke(
            contextd.cli.cli,
            ["bench", "notes", "--expand", "units", "--graph-weight", "2"],
        )
        assert result.exit_code == 0, result.output
        assert mock_run.call_args.kwargs["expand"] == "units"
        assert mock_run.call_args.kwargs["graph_weight"] == 2.0

        result = CliRunner().invoke(contextd.cli.cli, ["bench", "notes"])
        assert result.exit_code == 0, result.output
        # Unset flags defer to [search] config inside run_bench.
        assert mock_run.call_args.kwargs["expand"] is None
        assert mock_run.call_args.kwargs["graph_weight"] is None

    result = CliRunner().invoke(contextd.cli.cli, ["bench", "notes", "--expand", "paths"])
    assert result.exit_code != 0
