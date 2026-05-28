"""Inference must never mint File/Section nodes.

``File`` and ``Section`` are enumeration-owned labels: they mirror real
on-disk content and are created only by the enumerate phases. When the LLM
emits an inferred relationship targeting one, ``phase_relate`` /
``phase_relate_sections`` resolve it to an *existing* node (so the edge
attaches to the real record) or drop the edge — they never create a phantom
stub. Stub-eligible labels (``Pattern``/``Risk``/...) are still upserted.

Regression guard for the phantom-stub pollution that left thousands of
``path=null`` Section nodes and bare-name File nodes in section/file queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from contextd.corpus_config import CorpusConfig
from contextd.indexer.phases import phase_relate, phase_relate_sections
from contextd.inference.relate import InferredRelationship


def _rel(target_type: str, target_name: str) -> InferredRelationship:
    return InferredRelationship(
        edge_type="REFERENCES",
        target_type=target_type,
        target_name=target_name,
        confidence=0.9,
        reason="r",
    )


def _file(tmp_path: Path) -> Path:
    f = tmp_path / "a.md"
    f.write_text("content")
    return f


# --- file-granular: File targets ---------------------------------------------


def test_phase_relate_drops_unresolvable_file_target(tmp_path: Path) -> None:
    """An LLM reference to a File that does not exist creates neither a stub
    node nor an edge; it is counted as skipped."""
    inferrer = MagicMock()
    inferrer.infer.return_value = [_rel("File", "ghost.md")]
    store = MagicMock()
    store.exec_read.return_value = []  # resume: none done; resolution: no match

    result = phase_relate(
        [_file(tmp_path)], inferrer, store, entity_sampler=lambda _s: [], corpus="c"
    )

    assert not any(c.args[0] == "File" for c in store.upsert_node.call_args_list)
    store.upsert_edge.assert_not_called()
    assert result.skipped == 1


def test_phase_relate_links_file_target_by_exact_path(tmp_path: Path) -> None:
    """An exact primary-key match resolves the edge onto the existing node."""
    inferrer = MagicMock()
    inferrer.infer.return_value = [_rel("File", "/corpus/other.md")]
    store = MagicMock()

    def _read(cypher: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        if "inferred_at IS NOT NULL" in cypher:
            return []
        if "n.path = $v" in cypher:  # exact-PK resolution hit
            return [{"v": "/corpus/other.md"}]
        return []

    store.exec_read.side_effect = _read

    phase_relate([_file(tmp_path)], inferrer, store, entity_sampler=lambda _s: [], corpus="c")

    assert not any(c.args[0] == "File" for c in store.upsert_node.call_args_list)
    store.upsert_edge.assert_called_once()
    assert store.upsert_edge.call_args.args[1] == "/corpus/other.md"


def test_phase_relate_links_file_target_by_unique_basename(tmp_path: Path) -> None:
    """A bare-name reference resolves to the one real File with that name."""
    inferrer = MagicMock()
    inferrer.infer.return_value = [_rel("File", "other.md")]
    store = MagicMock()
    real_path = "/corpus/sub/other.md"

    def _read(cypher: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        if "inferred_at IS NOT NULL" in cypher:
            return []
        if "n.name = $b" in cypher:  # basename resolution hit
            return [{"v": real_path}]
        return []  # exact-PK miss

    store.exec_read.side_effect = _read

    phase_relate([_file(tmp_path)], inferrer, store, entity_sampler=lambda _s: [], corpus="c")

    assert not any(c.args[0] == "File" for c in store.upsert_node.call_args_list)
    store.upsert_edge.assert_called_once()
    assert store.upsert_edge.call_args.args[1] == real_path


def test_phase_relate_drops_ambiguous_basename(tmp_path: Path) -> None:
    """A basename shared by two real Files is left unresolved, not mis-linked."""
    inferrer = MagicMock()
    inferrer.infer.return_value = [_rel("File", "README.md")]
    store = MagicMock()

    def _read(cypher: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        if "inferred_at IS NOT NULL" in cypher:
            return []
        if "n.name = $b" in cypher:
            return [{"v": "/corpus/x/README.md"}, {"v": "/corpus/y/README.md"}]
        return []

    store.exec_read.side_effect = _read

    result = phase_relate(
        [_file(tmp_path)], inferrer, store, entity_sampler=lambda _s: [], corpus="c"
    )

    store.upsert_edge.assert_not_called()
    assert result.skipped == 1


def test_phase_relate_still_stubs_pattern_target(tmp_path: Path) -> None:
    """Non-enumeration-owned labels (Pattern) are still upserted as stubs."""
    inferrer = MagicMock()
    inferrer.infer.return_value = [_rel("Pattern", "Singleton")]
    store = MagicMock()
    store.exec_read.return_value = []

    phase_relate([_file(tmp_path)], inferrer, store, entity_sampler=lambda _s: [], corpus="c")

    pattern_upserts = [c for c in store.upsert_node.call_args_list if c.args[0] == "Pattern"]
    assert len(pattern_upserts) == 1
    store.upsert_edge.assert_called_once()


# --- section-granular: Section targets ---------------------------------------


def _section_corpus(tmp_path: Path) -> tuple[CorpusConfig, list[dict[str, Any]]]:
    f = tmp_path / "doc.md"
    f.write_text("## Heading 0\n\nbody-0\n")
    corpus = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "c",
                "root": str(tmp_path),
                "include": ["*.md"],
                "granularity": "section",
                "heading_min_level": 2,
                "heading_max_level": 4,
            }
        }
    )
    rows = [{"id": f"{f.as_posix()}#heading-0", "path": f.as_posix()}]
    return corpus, rows


def test_phase_relate_sections_drops_unresolvable_section_target(tmp_path: Path) -> None:
    """A malformed section cross-reference (no matching id) is dropped, not
    stubbed as a phantom path-less Section."""
    corpus, rows = _section_corpus(tmp_path)
    inferrer = MagicMock()
    inferrer.infer.return_value = [_rel("Section", "�12.2.5")]
    store = MagicMock()

    def _read(cypher: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        return rows if "s.inferred_at IS NULL" in cypher else []

    store.exec_read.side_effect = _read

    result = phase_relate_sections(corpus, inferrer, store, entity_sampler=lambda _s: [])

    assert not any(c.args[0] == "Section" for c in store.upsert_node.call_args_list)
    store.upsert_edge.assert_not_called()
    assert result.skipped == 1


def test_phase_relate_sections_links_resolved_section_target(tmp_path: Path) -> None:
    """An exact section-id reference attaches the edge to the existing node."""
    corpus, rows = _section_corpus(tmp_path)
    target_id = "/corpus/doc.md#other"
    inferrer = MagicMock()
    inferrer.infer.return_value = [_rel("Section", target_id)]
    store = MagicMock()

    def _read(cypher: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        if "s.inferred_at IS NULL" in cypher:
            return rows
        if "n.id = $v" in cypher:  # exact-id resolution hit
            return [{"v": target_id}]
        return []

    store.exec_read.side_effect = _read

    phase_relate_sections(corpus, inferrer, store, entity_sampler=lambda _s: [])

    assert not any(c.args[0] == "Section" for c in store.upsert_node.call_args_list)
    store.upsert_edge.assert_called_once()
    assert store.upsert_edge.call_args.args[1] == target_id
