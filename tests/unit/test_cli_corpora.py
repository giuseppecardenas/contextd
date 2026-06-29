"""Tests for add-corpus and list-corpora CLI commands."""

from __future__ import annotations

import tomllib
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

import contextd.cli
from contextd.cli.corpora import _rewrite_template_path


def _setup_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    (home / "config.toml").write_text(
        '[storage]\nbackend = "neo4j"\n\n[storage.neo4j]\n'
        f'docker_compose_file = "{home}/docker-compose.yml"\n'
    )
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
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
    home = _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "dup"
    corpus_dir.mkdir()
    runner = CliRunner()
    first = runner.invoke(contextd.cli.cli, ["add-corpus", str(corpus_dir)])
    assert first.exit_code == 0
    # Capture the on-disk file's mtime + bytes before the duplicate attempt.
    toml_path = home / "corpora" / "dup.toml"
    before_bytes = toml_path.read_bytes()
    before_mtime = toml_path.stat().st_mtime_ns
    second = runner.invoke(contextd.cli.cli, ["add-corpus", str(corpus_dir)])
    assert second.exit_code == 0  # warns but doesn't error
    assert "already registered" in second.output
    # Duplicate attempt must NOT have rewritten the TOML — guards against
    # an accidental early-return reorder that lets write_bytes run first.
    assert toml_path.read_bytes() == before_bytes
    assert toml_path.stat().st_mtime_ns == before_mtime


def test_list_corpora_when_corpora_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`contextd list-corpora` before `contextd init` gave a specific
    'run `contextd init` first' message. Covers the absent branch."""
    home = tmp_path / ".contextd"
    home.mkdir()
    # Deliberately do NOT mkdir corpora/.
    (home / "config.toml").write_text(
        f'[storage]\nbackend = "neo4j"\n\n[storage.neo4j]\n'
        f'docker_compose_file = "{home}/docker-compose.yml"\n'
    )
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    result = CliRunner().invoke(contextd.cli.cli, ["list-corpora"])
    assert result.exit_code == 0
    assert "run `contextd init` first" in result.output


def test_list_corpora_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """corpora/ present but empty — exercises the second-branch message
    separately from the missing-dir branch above."""
    home = _setup_home(tmp_path, monkeypatch)
    (home / "corpora").mkdir()  # present-but-empty, unlike the "missing" case
    result = CliRunner().invoke(contextd.cli.cli, ["list-corpora"])
    assert result.exit_code == 0
    assert "no corpora registered yet" in result.output


def test_list_corpora_shows_registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "first"
    corpus_dir.mkdir()
    CliRunner().invoke(contextd.cli.cli, ["add-corpus", str(corpus_dir)])
    result = CliRunner().invoke(contextd.cli.cli, ["list-corpora"])
    assert result.exit_code == 0
    assert "first" in result.output


# ---------------------------------------------------------------------------
# --from TEMPLATE tests
# ---------------------------------------------------------------------------


def _write_template(
    template_dir: Path,
    *,
    overrides: str | None = "ontology.json",
    prompt_override: str | None = "prompts/summary.md",
    mcp_tools: dict[str, str] | None = None,
    granularity: str = "section",
    name: str = "template-corpus",
    root: str = "/some/root",
) -> Path:
    """Write a synthetic corpus TOML template to *template_dir* and return its path."""
    if mcp_tools is None:
        mcp_tools = {"my_tool": "tools/query.cypher"}

    lines = [
        "[corpus]",
        f'name = "{name}"',
        f'root = "{root}"',
        'include = ["**/*.md"]',
        f'granularity = "{granularity}"',
        "",
        "[embedding]",
        'model = "voyage-4-large"',
        "",
        "[ontology]",
        'base = "default"',
    ]
    if overrides is not None:
        lines.append(f'overrides = "{overrides}"')
    lines += ["", "[ontology.aliases]", 'Foo = "Pattern"', ""]

    lines += ["[mcp.tools]"]
    for k, v in mcp_tools.items():
        lines.append(f'{k} = "{v}"')
    lines.append("")

    lines += ["[summarization]"]
    if prompt_override is not None:
        lines.append(f'prompt_override = "{prompt_override}"')

    template_path = template_dir / "corpus.toml"
    template_path.write_text("\n".join(lines), encoding="utf-8")
    return template_path


def test_add_corpus_from_template_copies_aliases_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aliases and section-structure config are preserved; root/name are overridden."""
    home = _setup_home(tmp_path, monkeypatch)
    template_dir = tmp_path / "tpl"
    template_dir.mkdir()
    template = _write_template(template_dir)

    corpus_dir = tmp_path / "my-data"
    corpus_dir.mkdir()

    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "foo", "--from", str(template)],
    )
    assert result.exit_code == 0, result.output

    toml_path = home / "corpora" / "foo.toml"
    assert toml_path.exists()
    data = tomllib.loads(toml_path.read_text())
    assert data["corpus"]["name"] == "foo"
    assert data["corpus"]["root"] == str(corpus_dir.resolve())
    # Aliases preserved.
    assert data["ontology"]["aliases"]["Foo"] == "Pattern"
    # Relative overrides path was rewritten to absolute.
    abs_overrides = str((template_dir / "ontology.json").resolve())
    assert data["ontology"]["overrides"] == abs_overrides
    # Granularity preserved from template.
    assert data["corpus"]["granularity"] == "section"
    # Success message includes template path.
    assert "foo" in result.output
    assert "from template" in result.output


def test_add_corpus_from_template_rewrites_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three relative-path fields are rewritten to absolute paths."""
    home = _setup_home(tmp_path, monkeypatch)
    template_dir = tmp_path / "tpl2"
    template_dir.mkdir()
    template = _write_template(
        template_dir,
        overrides="ontology.json",
        prompt_override="prompts/summary.md",
        mcp_tools={"t": "tools/q.cypher"},
    )

    corpus_dir = tmp_path / "data2"
    corpus_dir.mkdir()

    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "bar", "--from", str(template)],
    )
    assert result.exit_code == 0, result.output
    data = tomllib.loads((home / "corpora" / "bar.toml").read_text())

    assert Path(data["ontology"]["overrides"]).is_absolute()
    assert Path(data["ontology"]["overrides"]) == (template_dir / "ontology.json").resolve()

    assert Path(data["summarization"]["prompt_override"]).is_absolute()
    assert (
        Path(data["summarization"]["prompt_override"])
        == (template_dir / "prompts" / "summary.md").resolve()
    )

    assert Path(data["mcp"]["tools"]["t"]).is_absolute()
    assert Path(data["mcp"]["tools"]["t"]) == (template_dir / "tools" / "q.cypher").resolve()


def test_add_corpus_from_template_preserves_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute paths in the template are left unchanged."""
    home = _setup_home(tmp_path, monkeypatch)
    template_dir = tmp_path / "tpl3"
    template_dir.mkdir()
    # Use a tmp_path-rooted absolute so the path is genuinely absolute on
    # every platform (a leading "/" alone is not absolute on Windows).
    # ``as_posix()`` keeps the value backslash-free in the hand-written TOML
    # template; the resolved path round-trips through pathlib, so compare by
    # ``Path`` equality (which normalises separators on Windows).
    abs_ontology_path = tmp_path / "shared" / "ontology.json"
    abs_ontology = abs_ontology_path.as_posix()
    template = _write_template(template_dir, overrides=abs_ontology, prompt_override=None)

    corpus_dir = tmp_path / "data3"
    corpus_dir.mkdir()

    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "baz", "--from", str(template)],
    )
    assert result.exit_code == 0, result.output
    data = tomllib.loads((home / "corpora" / "baz.toml").read_text())
    assert Path(data["ontology"]["overrides"]) == abs_ontology_path


def test_add_corpus_from_template_granularity_inherited_from_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Template granularity wins over (or is unaffected by) --granularity flag."""
    home = _setup_home(tmp_path, monkeypatch)
    template_dir = tmp_path / "tpl4"
    template_dir.mkdir()
    # Template explicitly says section.
    template = _write_template(template_dir, granularity="section")

    corpus_dir = tmp_path / "data4"
    corpus_dir.mkdir()

    # Pass --granularity file (the default), but the template says section.
    result = CliRunner().invoke(
        contextd.cli.cli,
        [
            "add-corpus",
            str(corpus_dir),
            "--name",
            "qux",
            "--granularity",
            "file",
            "--from",
            str(template),
        ],
    )
    assert result.exit_code == 0, result.output
    data = tomllib.loads((home / "corpora" / "qux.toml").read_text())
    # Template granularity is preserved; the --granularity flag is ignored.
    assert data["corpus"]["granularity"] == "section"


def test_add_corpus_from_template_invalid_toml_raises_clickexception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed TOML template → exit code 1 with readable error."""
    _setup_home(tmp_path, monkeypatch)
    template_dir = tmp_path / "bad_toml"
    template_dir.mkdir()
    bad_template = template_dir / "corpus.toml"
    bad_template.write_text("this is [not valid\nTOML", encoding="utf-8")

    corpus_dir = tmp_path / "data5"
    corpus_dir.mkdir()

    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "err1", "--from", str(bad_template)],
    )
    assert result.exit_code == 1
    assert "template invalid" in result.output


def test_add_corpus_from_template_invalid_schema_raises_clickexception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structurally invalid template (e.g. corpus.name is an integer) → exit code 1."""
    _setup_home(tmp_path, monkeypatch)
    template_dir = tmp_path / "bad_schema"
    template_dir.mkdir()
    bad_template = template_dir / "corpus.toml"
    # corpus.name must be a string; 42 is invalid.
    bad_template.write_text(
        dedent("""\
        [corpus]
        name = 42
        root = "/some/path"
        """),
        encoding="utf-8",
    )

    corpus_dir = tmp_path / "data6"
    corpus_dir.mkdir()

    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "err2", "--from", str(bad_template)],
    )
    assert result.exit_code == 1
    assert "template invalid" in result.output


def test_add_corpus_without_from_preserves_existing_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: omitting --from leaves the original scratch-TOML path intact."""
    home = _setup_home(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "smoke"
    corpus_dir.mkdir()
    result = CliRunner().invoke(
        contextd.cli.cli,
        ["add-corpus", str(corpus_dir), "--name", "smoke", "--granularity", "file"],
    )
    assert result.exit_code == 0, result.output
    toml_path = home / "corpora" / "smoke.toml"
    assert toml_path.exists()
    data = tomllib.loads(toml_path.read_text())
    assert data["corpus"]["name"] == "smoke"
    assert data["corpus"]["include"] == ["**/*.md"]
    assert data["corpus"]["granularity"] == "file"


def test_add_corpus_from_acme_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip test with a copy of the realistic acme-prd corpus.toml.

    Validates:
    - Template paths are template-parent-relative (not repo-root-relative).
    - The rewritten paths resolve to real files (existence check).
    - The resulting TOML is loadable via CorpusConfig.load (the downstream
      invariant: written TOML must be usable by the indexer).
    """
    from contextd.corpus_config import CorpusConfig

    home = _setup_home(tmp_path, monkeypatch)

    # Build a directory that mimics examples/acme-prd/ structure.
    examples_dir = tmp_path / "examples" / "acme-prd"
    examples_dir.mkdir(parents=True)
    tools_dir = examples_dir / "tools"
    tools_dir.mkdir()
    prompts_dir = examples_dir / "prompts"
    prompts_dir.mkdir()

    # Create placeholder files for the artifacts referenced by the template.
    # These must exist so the existence assertions below are meaningful; without
    # them a path-doubling bug (e.g. "acme-prd/acme-prd/ontology.json")
    # would silently produce a non-existent path that the test would catch.
    (examples_dir / "ontology.json").write_text("{}", encoding="utf-8")
    (prompts_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")
    (tools_dir / "four_surface.cypher").write_text("MATCH (n) RETURN n", encoding="utf-8")
    (tools_dir / "dangling.cypher").write_text("MATCH (n) RETURN n", encoding="utf-8")
    (tools_dir / "stale_shas.cypher").write_text("MATCH (n) RETURN n", encoding="utf-8")

    # Template uses template-parent-relative paths (the correct form).
    template = examples_dir / "corpus.toml"
    template.write_text(
        dedent("""\
        [corpus]
        name = "acme-prd"
        root = "/home/you/src/acme"
        include = ["docs/prd/**/*.md", "mods/base/**/*.lua", "prd.md", "CLAUDE.md"]
        exclude = ["docs/prd/_audit-methodology.md"]
        granularity = "section"
        heading_min_level = 2
        heading_max_level = 4

        [embedding]
        model = "voyage-4-large"

        [ontology]
        base = "default"
        overrides = "ontology.json"

        [ontology.aliases]
        Registry = "Pattern"
        FRRow = "Ticket"
        LuaFile = "File"
        GapEntry = "Risk"

        [mcp.tools]
        four_surface = "tools/four_surface.cypher"
        find_dangling_registrations = "tools/dangling.cypher"
        audit_stale_shas = "tools/stale_shas.cypher"

        [summarization]
        prompt_override = "prompts/summary.md"
        """),
        encoding="utf-8",
    )

    corpus_dir = tmp_path / "acme"
    corpus_dir.mkdir()

    result = CliRunner().invoke(
        contextd.cli.cli,
        [
            "add-corpus",
            str(corpus_dir),
            "--name",
            "acme-prd",
            "--from",
            str(template),
        ],
    )
    assert result.exit_code == 0, result.output

    corpus_toml_path = home / "corpora" / "acme-prd.toml"
    data = tomllib.loads(corpus_toml_path.read_text())

    # Name and root overridden.
    assert data["corpus"]["name"] == "acme-prd"
    assert data["corpus"]["root"] == str(corpus_dir.resolve())

    # Aliases preserved.
    assert data["ontology"]["aliases"]["Registry"] == "Pattern"
    assert data["ontology"]["aliases"]["GapEntry"] == "Risk"

    # Relative paths rewritten to absolute (resolved relative to template parent).
    template_parent = examples_dir
    assert Path(data["ontology"]["overrides"]) == (template_parent / "ontology.json").resolve()
    assert (
        Path(data["summarization"]["prompt_override"])
        == (template_parent / "prompts" / "summary.md").resolve()
    )
    expected_tool_paths = {
        "four_surface": (template_parent / "tools" / "four_surface.cypher").resolve(),
        "find_dangling_registrations": (template_parent / "tools" / "dangling.cypher").resolve(),
        "audit_stale_shas": (template_parent / "tools" / "stale_shas.cypher").resolve(),
    }
    for tool_name, expected in expected_tool_paths.items():
        assert Path(data["mcp"]["tools"][tool_name]) == expected

    # Each rewritten path must resolve to a real file on disk.
    assert Path(data["ontology"]["overrides"]).exists(), (
        f"ontology.overrides does not exist: {data['ontology']['overrides']}"
    )
    assert Path(data["summarization"]["prompt_override"]).exists(), (
        f"summarization.prompt_override does not exist: {data['summarization']['prompt_override']}"
    )
    for tool_name, tool_path in data["mcp"]["tools"].items():
        assert Path(tool_path).exists(), f"mcp.tools.{tool_name} does not exist: {tool_path}"

    # The resulting TOML must be loadable via CorpusConfig.load — this is the
    # downstream invariant that actually matters: a written TOML must be usable
    # by the indexer without errors.
    loaded_cfg = CorpusConfig.load(corpus_toml_path)
    assert loaded_cfg.corpus.name == "acme-prd"
    assert loaded_cfg.corpus.granularity == "section"
    assert loaded_cfg.ontology.aliases["Registry"] == "Pattern"

    # Include/exclude preserved.
    assert "docs/prd/**/*.md" in data["corpus"]["include"]
    assert data["corpus"]["exclude"] == ["docs/prd/_audit-methodology.md"]


# ---------------------------------------------------------------------------
# _rewrite_template_path unit tests
# ---------------------------------------------------------------------------


def test_rewrite_template_path_absolute_passes_through(tmp_path: Path) -> None:
    """Absolute paths are returned unchanged regardless of the anchor.

    Uses a tmp_path-rooted absolute so the path is genuinely absolute on
    every platform (a leading "/" alone is not absolute on Windows).
    """
    abs_path = tmp_path / "etc" / "foo"
    assert Path(_rewrite_template_path(str(abs_path), Path("/tmp"))) == abs_path


def test_rewrite_template_path_relative_resolves_against_anchor(tmp_path: Path) -> None:
    """Relative paths are resolved against the supplied anchor directory."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.json").write_text("", encoding="utf-8")
    result = _rewrite_template_path("sub/x.json", tmp_path)
    assert result == str((tmp_path / "sub" / "x.json").resolve())
