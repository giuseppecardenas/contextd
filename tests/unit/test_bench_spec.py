"""Tests for the bench spec parser (contextd.bench.spec)."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from contextd.bench.metrics import Target
from contextd.bench.spec import BenchSpec, BenchSpecError, load_spec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_BENCH = _REPO_ROOT / "examples" / "minimal-notes" / ".contextd" / "bench.toml"


def _write(tmp_path: Path, text: str, name: str = "bench.toml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_toml_full_shape(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
[[queries]]
q = "what do the notes say about sourdough hydration"
expect = [
    { path = "note-3.md", anchor = "hydration", lines = [12, 30] },
    { path = "note-7.md" },
]

[[queries]]
q = "cooking"
expect = [{ path = "note-3.md" }]
k = 3
""",
    )
    spec = load_spec(p)
    assert len(spec.queries) == 2
    first = spec.queries[0]
    assert first.q == "what do the notes say about sourdough hydration"
    assert first.k is None
    assert first.expect == [
        Target("note-3.md", anchor="hydration", lines=(12, 30)),
        Target("note-7.md"),
    ]
    assert spec.queries[1].k == 3
    assert spec.queries[1].expect[0].anchor is None
    assert spec.queries[1].expect[0].lines is None
    assert BenchSpec.load(p) == spec


def test_missing_file_names_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(BenchSpecError, match=r"nope\.toml"):
        load_spec(missing)


def test_invalid_toml_syntax(tmp_path: Path) -> None:
    p = _write(tmp_path, "[[queries]\nq = ")
    with pytest.raises(BenchSpecError, match="invalid TOML syntax"):
        load_spec(p)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ('[[queries]]\nq = "x"\nexpect = [{ path = "a.md", bogus = 1 }]\n', "bogus"),
        ('[[queries]]\nq = "x"\nexpect = []\n', "expect"),
        ('[[queries]]\nq = ""\nexpect = [{ path = "a.md" }]\n', "q"),
        ('[[queries]]\nq = "x"\nexpect = [{ path = "a.md" }]\nk = 0\n', "k"),
        ('[[queries]]\nq = "x"\nexpect = [{ path = "a.md", lines = [5, 5] }]\n', "lines"),
        ('[[queries]]\nq = "x"\nexpect = [{ path = "a.md", lines = [-1, 5] }]\n', "lines"),
        ('[[queries]]\nq = "x"\nexpect = [{ path = "a.md", lines = [1, 2, 3] }]\n', "lines"),
        ('[[queries]]\nq = "x"\nexpect = [{ path = "a.md" }]\nextra = 1\n', "extra"),
        ("queries = []\n", "queries"),
        ("[other]\nx = 1\n", "queries"),
    ],
)
def test_validation_errors_carry_path(tmp_path: Path, body: str, fragment: str) -> None:
    p = _write(tmp_path, body)
    with pytest.raises(BenchSpecError) as excinfo:
        load_spec(p)
    message = str(excinfo.value)
    assert str(p) in message
    assert fragment in message


def test_unsupported_suffix(tmp_path: Path) -> None:
    p = _write(tmp_path, "queries: []", name="bench.json")
    with pytest.raises(BenchSpecError, match=r"\.toml"):
        load_spec(p)


def test_yaml_without_pyyaml_names_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _write(tmp_path, "queries:\n  - q: x\n    expect: [{path: a.md}]\n", name="bench.yaml")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "yaml", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BenchSpecError, match=r"pyyaml.*\.toml"):
        load_spec(p)


def test_yaml_parses_when_pyyaml_available(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    p = _write(
        tmp_path,
        "queries:\n  - q: x\n    expect:\n      - {path: a.md, anchor: h, lines: [1, 4]}\n",
        name="bench.yaml",
    )
    spec = load_spec(p)
    assert spec.queries[0].expect == [Target("a.md", anchor="h", lines=(1, 4))]


def test_from_mapping_direct() -> None:
    spec = BenchSpec.from_mapping({"queries": [{"q": "x", "expect": [{"path": "a.md"}]}]})
    assert spec.queries[0].expect == [Target("a.md")]
    with pytest.raises(BenchSpecError, match="<mapping>"):
        BenchSpec.from_mapping({"queries": []})


def test_example_bench_parses_and_matches_notes() -> None:
    """The shipped example spec must parse and only name files that exist."""
    spec = load_spec(_EXAMPLE_BENCH)
    assert 4 <= len(spec.queries) <= 6
    corpus_root = _EXAMPLE_BENCH.parent.parent
    for query in spec.queries:
        assert query.expect
        for target in query.expect:
            path = corpus_root / target.path
            assert path.is_file(), f"{query.q!r} expects missing file {target.path}"
            if target.lines is not None:
                n_lines = len(path.read_text(encoding="utf-8").splitlines())
                assert target.lines[1] <= n_lines
