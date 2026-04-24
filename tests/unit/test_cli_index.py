"""Tests for the `index` CLI command."""

from __future__ import annotations

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
        f'[storage]\nbackend = "memgraph"\n\n[storage.memgraph]\n'
        f'docker_compose_file = "{home}/docker-compose.yml"\n'
    )
    (home / "corpora").mkdir()
    (home / "state").mkdir()
    (home / "prompts").mkdir()
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
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


def test_index_incremental_flag_is_recognised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 'not yet implemented' stub must be gone after this task."""
    _setup_home(tmp_path, monkeypatch)
    result = CliRunner().invoke(contextd.cli.cli, ["index", "--help"])
    assert "--incremental" in result.output
    # Once implemented the stub message must not appear.


def test_index_incremental_runs_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextd.indexer.pipeline import IncrementalResult

    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")
    (corpus_root / "b.md").write_text("there")
    _register_corpus(home, "docs", corpus_root)

    with (
        patch("contextd.cli.corpora._build_pipeline_deps") as mock_deps,
        patch(
            "contextd.cli.corpora.enumerate_corpus_files",
            return_value=[tmp_path / "a.md", tmp_path / "b.md"],
        ),
        patch(
            "contextd.cli.corpora.run_incremental_file",
            return_value=IncrementalResult("indexed", "a.md"),
        ) as mock_rif,
        patch("contextd.cli.corpora.branch_is_allowed", return_value=True),
    ):
        mock_deps.return_value = MagicMock()
        result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--incremental"])

    assert result.exit_code == 0, result.output
    assert mock_rif.call_count == 2


def test_index_incremental_reports_zero_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    _register_corpus(home, "docs", corpus_root)

    with (
        patch("contextd.cli.corpora._build_pipeline_deps") as mock_deps,
        patch("contextd.cli.corpora.enumerate_corpus_files", return_value=[]),
        patch("contextd.cli.corpora.branch_is_allowed", return_value=True),
    ):
        mock_deps.return_value = MagicMock()
        result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--incremental"])

    assert result.exit_code == 0
    assert "incremental scan complete" in result.output


def test_index_bootstrap_prints_per_phase_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`contextd index <corpus> --bootstrap` must invoke run_bootstrap and
    render a green-check per phase. Covers the else-branch that the existing
    three tests (estimate-only, error paths, incremental stub) don't touch."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hello")
    _register_corpus(home, "docs", corpus_root)

    # Canned BootstrapResult — 5 phases matching the file-mode pipeline.
    from contextd.indexer.phases import PhaseResult
    from contextd.indexer.pipeline import BootstrapResult

    canned = BootstrapResult(
        phases=[
            PhaseResult(name="enumerate", processed=3, skipped=0),
            PhaseResult(name="embed", processed=3, skipped=0),
            PhaseResult(name="summarise", processed=2, skipped=1),
            PhaseResult(name="relate", processed=3, skipped=0),
            PhaseResult(name="close", processed=1, skipped=0),
        ]
    )

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store") as mock_store,
        patch("contextd.indexer.pipeline.run_bootstrap", return_value=canned) as mock_run,
    ):
        mock_store.return_value = MagicMock()
        result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--bootstrap"])
    assert result.exit_code == 0, result.output
    assert mock_run.called
    # One green-check line per phase.
    for phase_name in ("enumerate", "embed", "summarise", "relate", "close"):
        assert phase_name in result.output
    # skipped count for summarise surfaces in output.
    assert "skipped=1" in result.output


# ---------------------------------------------------------------------------
# _build_pipeline_deps: ontology overrides wiring
# ---------------------------------------------------------------------------


def _register_corpus_with_overrides(
    home: Path,
    name: str,
    root: Path,
    overrides_path: str,
) -> Path:
    """Write a corpus TOML that references an overrides JSON file."""
    import tomli_w

    corpus_toml = home / "corpora" / f"{name}.toml"
    data = {
        "corpus": {
            "name": name,
            "root": str(root),
            "include": ["**/*.md"],
            "granularity": "file",
        },
        "ontology": {
            "overrides": overrides_path,
        },
    }
    corpus_toml.write_bytes(tomli_w.dumps(data).encode())
    return corpus_toml


def test_build_pipeline_deps_no_overrides_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corpus TOML with no [ontology] overrides key → ontology has only node aliases."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")
    _register_corpus(home, "docs", corpus_root)

    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig

    corpus_toml = home / "corpora" / "docs.toml"
    cfg = Config.load_default()
    corpus_cfg = CorpusConfig.load(corpus_toml)

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        deps = _build_pipeline_deps(cfg, corpus_cfg, "docs", corpus_toml)

    # No overrides → edge_aliases stays empty.
    from contextd.inference.relate import RelationshipInferrer

    assert isinstance(deps.inferrer, RelationshipInferrer)
    # Access the ontology stored on the inferrer's attribute.
    ontology = deps.inferrer._onto  # type: ignore[attr-defined]
    assert dict(ontology.edge_aliases) == {}


def test_build_pipeline_deps_resolves_relative_overrides_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative overrides path resolves relative to the corpus TOML's directory."""
    import json

    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")

    # Write the overrides JSON next to the TOML (inside corpora/).
    overrides_file = home / "corpora" / "edge_aliases.json"
    overrides_file.write_text(
        json.dumps({"edge_label_aliases": {"CITES": "REFERENCES"}}), encoding="utf-8"
    )
    corpus_toml = _register_corpus_with_overrides(
        home,
        "docs",
        corpus_root,
        "edge_aliases.json",  # relative!
    )

    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig

    cfg = Config.load_default()
    corpus_cfg = CorpusConfig.load(corpus_toml)

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        deps = _build_pipeline_deps(cfg, corpus_cfg, "docs", corpus_toml)

    ontology = deps.inferrer._onto  # type: ignore[attr-defined]
    assert dict(ontology.edge_aliases) == {"CITES": "REFERENCES"}


def test_build_pipeline_deps_absolute_overrides_path_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute overrides path in the TOML is used verbatim (no prefix added)."""
    import json

    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")

    # Place the overrides JSON somewhere else entirely (not next to the TOML).
    overrides_dir = tmp_path / "shared"
    overrides_dir.mkdir()
    overrides_file = overrides_dir / "overrides.json"
    overrides_file.write_text(
        json.dumps({"edge_label_aliases": {"CONSUMES": "USES"}}), encoding="utf-8"
    )
    corpus_toml = _register_corpus_with_overrides(
        home,
        "docs",
        corpus_root,
        str(overrides_file),  # absolute
    )

    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig

    cfg = Config.load_default()
    corpus_cfg = CorpusConfig.load(corpus_toml)

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        deps = _build_pipeline_deps(cfg, corpus_cfg, "docs", corpus_toml)

    ontology = deps.inferrer._onto  # type: ignore[attr-defined]
    assert dict(ontology.edge_aliases) == {"CONSUMES": "USES"}


def test_build_pipeline_deps_stacks_node_and_edge_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both [ontology] aliases (node) and overrides (edge) are applied and preserved."""
    import json

    import tomli_w

    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")

    # Write an overrides JSON with an edge alias.
    overrides_file = home / "corpora" / "edge_aliases.json"
    overrides_file.write_text(
        json.dumps({"edge_label_aliases": {"CITES": "REFERENCES"}}), encoding="utf-8"
    )

    # Write a TOML that declares a node alias AND references the overrides file.
    corpus_toml = home / "corpora" / "docs.toml"
    data = {
        "corpus": {
            "name": "docs",
            "root": str(corpus_root),
            "include": ["**/*.md"],
            "granularity": "file",
        },
        "ontology": {
            "aliases": {"Registry": "Pattern"},
            "overrides": "edge_aliases.json",
        },
    }
    corpus_toml.write_bytes(tomli_w.dumps(data).encode())

    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig

    cfg = Config.load_default()
    corpus_cfg = CorpusConfig.load(corpus_toml)

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        deps = _build_pipeline_deps(cfg, corpus_cfg, "docs", corpus_toml)

    ontology = deps.inferrer._onto  # type: ignore[attr-defined]
    # Node alias preserved.
    assert ontology.resolve_alias("Registry") == "Pattern"
    # Edge alias from overrides applied.
    assert ontology.resolve_edge_alias("CITES") == "REFERENCES"


def test_index_surfaces_overrides_error_as_click_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the overrides file does not exist, `contextd index --bootstrap`
    surfaces the error as a ClickException (exit code 1), not a Python traceback."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")
    # Register with an overrides path pointing at a non-existent file.
    _register_corpus_with_overrides(home, "docs", corpus_root, "does_not_exist.json")

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--bootstrap"])

    assert result.exit_code == 1
    assert "overrides file not readable" in result.output


def test_index_surfaces_bad_edge_target_as_click_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the overrides JSON has an alias pointing at an unknown edge type,
    the OntologyError surfaces as a ClickException with the original message."""
    import json

    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")

    # Create an overrides file referencing a nonexistent edge type.
    overrides_file = home / "corpora" / "bad_aliases.json"
    overrides_file.write_text(
        json.dumps({"edge_label_aliases": {"CITES": "NONEXISTENT_EDGE_TYPE"}}),
        encoding="utf-8",
    )
    _register_corpus_with_overrides(home, "docs", corpus_root, "bad_aliases.json")

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--bootstrap"])

    assert result.exit_code == 1
    assert "NONEXISTENT_EDGE_TYPE" in result.output


# ---------------------------------------------------------------------------
# _build_pipeline_deps: summarization.prompt_override wiring
# ---------------------------------------------------------------------------


def _register_corpus_with_prompt_override(
    home: Path,
    name: str,
    root: Path,
    prompt_override: str,
) -> Path:
    """Write a corpus TOML that declares a [summarization] prompt_override."""
    import tomli_w

    corpus_toml = home / "corpora" / f"{name}.toml"
    data = {
        "corpus": {
            "name": name,
            "root": str(root),
            "include": ["**/*.md"],
            "granularity": "file",
        },
        "summarization": {
            "prompt_override": prompt_override,
        },
    }
    corpus_toml.write_bytes(tomli_w.dumps(data).encode())
    return corpus_toml


def test_build_pipeline_deps_no_prompt_override_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corpus TOML with no [summarization] prompt_override → Summariser built with no
    prompt_path (default template used at summarise time)."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")
    _register_corpus(home, "docs", corpus_root)

    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig

    corpus_toml = home / "corpora" / "docs.toml"
    cfg = Config.load_default()
    corpus_cfg = CorpusConfig.load(corpus_toml)

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        deps = _build_pipeline_deps(cfg, corpus_cfg, "docs", corpus_toml)

    from contextd.inference.summarise import Summariser

    assert isinstance(deps.summariser, Summariser)
    assert deps.summariser._prompt_path is None  # type: ignore[attr-defined]


def test_build_pipeline_deps_resolves_relative_prompt_override_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative prompt_override path resolves relative to the corpus TOML's directory."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")

    # Place the override template next to the TOML (inside corpora/).
    override_file = home / "corpora" / "summary.md"
    override_file.write_text("Custom: {{content}}", encoding="utf-8")
    corpus_toml = _register_corpus_with_prompt_override(
        home,
        "docs",
        corpus_root,
        "summary.md",  # relative path
    )

    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig

    cfg = Config.load_default()
    corpus_cfg = CorpusConfig.load(corpus_toml)

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        deps = _build_pipeline_deps(cfg, corpus_cfg, "docs", corpus_toml)

    prompt_path = deps.summariser._prompt_path  # type: ignore[attr-defined]
    assert prompt_path is not None
    assert prompt_path.is_absolute()
    assert prompt_path == override_file.resolve()


def test_build_pipeline_deps_absolute_prompt_override_path_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute prompt_override path in the TOML is used verbatim (no prefix added)."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")

    # Place the override template somewhere else entirely.
    override_dir = tmp_path / "shared"
    override_dir.mkdir()
    override_file = override_dir / "summary.md"
    override_file.write_text("Absolute: {{content}}", encoding="utf-8")
    corpus_toml = _register_corpus_with_prompt_override(
        home,
        "docs",
        corpus_root,
        str(override_file),  # absolute path
    )

    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig

    cfg = Config.load_default()
    corpus_cfg = CorpusConfig.load(corpus_toml)

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        deps = _build_pipeline_deps(cfg, corpus_cfg, "docs", corpus_toml)

    prompt_path = deps.summariser._prompt_path  # type: ignore[attr-defined]
    assert prompt_path is not None
    assert prompt_path == override_file.resolve()


def test_build_pipeline_deps_missing_prompt_override_raises_clickexception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When prompt_override points at a nonexistent file, `contextd index --bootstrap`
    surfaces a ClickException (exit code 1), not a Python traceback."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("hi")
    _register_corpus_with_prompt_override(home, "docs", corpus_root, "no_such_template.md")

    with (
        patch("contextd.providers.factory.build_inference_provider"),
        patch("contextd.providers.factory.build_embedding_provider"),
        patch("contextd.storage.factory.build_graph_store"),
    ):
        result = CliRunner().invoke(contextd.cli.cli, ["index", "docs", "--bootstrap"])

    assert result.exit_code == 1
    assert "summarization.prompt_override file not found" in result.output
