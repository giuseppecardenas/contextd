"""``search(expand="units")`` over a bootstrapped graph with lexical entity edges.

Two sections in different files cite the same requirement id; only one of
them contains the query term. The direct search finds that one; the graph
walk reaches the other through the shared ``Ticket`` node and explains the
hop in ``via``. A third file that cites nothing stays out, and so do the
seed's own siblings (structural edges are not a relation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from contextd._paths import canonical_path
from contextd.corpus_config import CorpusConfig
from contextd.indexer.lexical import LexicalRegistry
from contextd.indexer.phases import RelateDeps
from contextd.inference.context import EmptyRetriever
from contextd.mcp import tools
from contextd.ontology.schema import Ontology
from contextd.storage.base import GraphStore
from tests.integration._chunk_corpus import HashEmbedder, bootstrap

pytestmark = pytest.mark.integration

TOKEN = "zorblatt"
SPEC_MD = f"""# Spec

Requirements for the rollout.

## Rollout

The rollout controller enables the {TOKEN} flag per region. Tracked as REQ-42.

## Notes

Nothing else to see in this file.
"""

IMPL_MD = """# Implementation

Module notes.

## Region controller

Applies the per-region flag described in REQ-42 using the staged controller.
"""

OTHER_MD = """# Other

Unrelated.

## Glossary

Terms used across the documents, none of them tracked.
"""


def _config(root: Path) -> CorpusConfig:
    return CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "expand",
                "root": str(root),
                "include": ["*.md"],
                "granularity": "section",
            },
            "chunking": {"tokenizer": "words"},
            "lexical": {
                "patterns": [
                    {"regex": r"\bREQ-\d+\b", "edge_type": "REFERENCES", "target_type": "Ticket"}
                ]
            },
        }
    )


def _relate(cfg: CorpusConfig) -> RelateDeps:
    """No LLM: an inferrer that infers nothing plus the corpus's lexical registry,
    wired the way ``cli/corpora.py`` builds production ``RelateDeps``."""
    inferrer = MagicMock()
    inferrer.infer.return_value = []
    return RelateDeps(
        inferrer=inferrer,
        retriever=EmptyRetriever(),
        lexical=LexicalRegistry(Ontology.load_base(), cfg.lexical.patterns),
    )


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(r["id"]) for r in rows]


def test_expand_units_reaches_the_citing_section_through_the_shared_ticket(
    backend: GraphStore, tmp_path: Path
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "spec.md").write_text(SPEC_MD, encoding="utf-8")
    (root / "impl.md").write_text(IMPL_MD, encoding="utf-8")
    (root / "other.md").write_text(OTHER_MD, encoding="utf-8")
    cfg = _config(root)
    bootstrap(backend, cfg, embedder=HashEmbedder(), relate=_relate(cfg))
    spec, impl = canonical_path(root / "spec.md"), canonical_path(root / "impl.md")

    # Precondition: the lexical pass linked both sections to one Ticket node.
    citing = backend.exec_read(
        "MATCH (s:Section)-[r:REFERENCES]->(t:Ticket {id: 'REQ-42'}) RETURN s.id AS id ORDER BY id"
    )
    assert {str(r["id"]) for r in citing} == {f"{spec}#rollout", f"{impl}#region-controller"}

    common: dict[str, Any] = {
        "mode": "fulltext",
        "corpus": "expand",
        "return_unit": "section",
        "window": 1,
        "limit": 10,
    }
    direct = tools.search(backend, TOKEN, **common)
    assert _ids(direct) == [f"{spec}#rollout"]

    rows = tools.search(backend, TOKEN, expand="units", expand_seeds=1, **common)
    by_id = {str(r["id"]): r for r in rows}
    assert _ids(rows)[0] == f"{spec}#rollout"
    # The direct hit keeps its chunk evidence (with neighbour context) and
    # carries no via block — nothing reached it through the graph.
    seed = by_id[f"{spec}#rollout"]
    assert TOKEN in seed["evidence"]["text"] and "context_before" in seed["evidence"]
    assert "via" not in seed

    reached = by_id[f"{impl}#region-controller"]
    assert reached["unit"] == "section" and reached["path"] == impl
    assert reached["via"] == {"entities": ["REQ-42"], "seeds": [f"{spec}#rollout"]}
    assert "evidence" not in reached

    # Neither the untracked file nor the seed's structural sibling appears.
    assert not any(p.endswith("other.md") for p in (str(r["path"]) for r in rows))
    assert f"{spec}#notes" not in by_id
