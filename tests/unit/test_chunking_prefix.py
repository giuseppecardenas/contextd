from __future__ import annotations

from contextd.chunking.model import Chunk, ChunkSpan
from contextd.chunking.prefix import apply_prefix, breadcrumb_text, static_prefix


def test_breadcrumb_joins_titles_or_falls_back_to_path() -> None:
    assert breadcrumb_text(("Doc", "Sec", "Sub"), "a/b.md") == "Doc > Sec > Sub"
    assert breadcrumb_text((" Doc ", "", "Sub"), "a/b.md") == "Doc > Sub"
    assert breadcrumb_text((), "a/b.md") == "a/b.md"


def test_static_prefix_modes() -> None:
    crumb = ("Doc", "Sec")
    assert static_prefix("none", breadcrumb=crumb, rel_path="p.md", parent_summary="s") == ""
    assert static_prefix("breadcrumb", breadcrumb=crumb, rel_path="p.md", parent_summary="s") == (
        "Doc > Sec"
    )
    assert (
        static_prefix(
            "section_summary", breadcrumb=crumb, rel_path="p.md", parent_summary=" A summary. "
        )
        == "Doc > Sec\nA summary."
    )
    # Missing summary degrades to the breadcrumb; llm is applied by the phase.
    assert static_prefix(
        "section_summary", breadcrumb=crumb, rel_path="p.md", parent_summary=None
    ) == ("Doc > Sec")
    assert (
        static_prefix("llm", breadcrumb=crumb, rel_path="p.md", parent_summary=None) == "Doc > Sec"
    )


def test_apply_prefix_sets_every_chunk() -> None:
    chunks = [Chunk(i, f"t{i}", ChunkSpan(0, 1), 1) for i in range(3)]
    out = apply_prefix(chunks, "X")
    assert out is chunks and all(c.prefix == "X" for c in chunks)
    assert chunks[0].embed_text == "X\n\nt0"
