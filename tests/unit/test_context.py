"""UnitIdentity/CandidateBundle contract — the prompt depends on render()."""

from __future__ import annotations

from contextd.inference.context import (
    CandidateBundle,
    EmptyRetriever,
    FileCandidate,
    SectionCandidate,
    UnitIdentity,
    identity_vars,
)


def _identity(**overrides: object) -> UnitIdentity:
    kwargs: dict[str, object] = {
        "corpus": "c",
        "file_path": "C:/x/docs/a.md",
        "rel_path": "docs/a.md",
        "suffix": ".md",
        "src_label": "Section",
        "src_id": "C:/x/docs/a.md#intro",
        "title": "Intro",
        "anchor": "intro",
        "parent_titles": ("Top", "Mid"),
    }
    kwargs.update(overrides)
    return UnitIdentity(**kwargs)  # type: ignore[arg-type]


def test_identity_vars_returns_every_key_always() -> None:
    keys = {"source_path", "section_title", "section_anchor", "parent_chain", "corpus_name"}
    assert set(identity_vars(None)) == keys
    assert all(v == "" for v in identity_vars(None).values())
    got = identity_vars(_identity())
    assert set(got) == keys
    assert got["source_path"] == "docs/a.md"
    assert got["parent_chain"] == "Top > Mid"


def test_render_is_deterministic_and_capped() -> None:
    bundle = CandidateBundle(
        entities_by_label={
            "Technology": tuple(f"tech-{i}" for i in range(30)),
            "Pattern": ("spatial hash",),
        },
        sections=(SectionCandidate(id="C:/x/a.md#s1", title="S1"),),
        files=(FileCandidate(path="C:/x/b.md", name="b.md"),),
    )
    text = bundle.render(per_group_cap=3)
    # Labels sorted alphabetically; caps applied per group.
    assert text.index("Pattern:") < text.index("Technology:")
    assert "tech-2" in text and "tech-3" not in text
    assert "C:/x/a.md#s1 — S1" in text
    assert "C:/x/b.md" in text
    assert bundle.render(per_group_cap=3) == text


def test_empty_bundle_renders_placeholder() -> None:
    assert CandidateBundle.empty().render() == "(none known yet)"


def test_empty_retriever_returns_empty_bundle() -> None:
    from unittest.mock import MagicMock

    bundle = EmptyRetriever().for_unit(MagicMock(), identity=_identity())
    assert bundle == CandidateBundle.empty()
