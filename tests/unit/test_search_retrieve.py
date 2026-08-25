from __future__ import annotations

from unittest.mock import MagicMock

from contextd.search.fusion import fuse_rankers, reciprocal_rank_fusion
from contextd.search.retrieve import ProfileSpec, run_rankers


def _row(key: str, **extra: object) -> dict[str, object]:
    return {"node": {"id": key, **extra}, "score": 1.0}


def test_run_rankers_one_pair_per_profile_with_filters_and_weights() -> None:
    store = MagicMock()
    store.vector_search.return_value = [_row("v")]
    store.full_text_search.return_value = [_row("f")]
    emb = MagicMock()
    emb.embed.return_value = [[0.1, 0.2]]
    run = run_rankers(
        store,
        "q",
        label="Chunk",
        search_prop="text",
        mode="hybrid",
        fetch_k=7,
        embedder=emb,
        vector_capable=True,
        filters={"corpus": "c"},
        profiles=[ProfileSpec("fine", 1.0), ProfileSpec("coarse", 0.5)],
        vector_weight=2.0,
        fulltext_weight=1.0,
    )
    emb.embed.assert_called_once_with(["q"])
    assert run.used_vector
    assert [w for _, w in run.rankers] == [2.0, 1.0, 1.0, 0.5]
    vkw = store.vector_search.call_args_list[1].kwargs
    assert vkw["filters"] == {"corpus": "c", "profile": "coarse"} and vkw["k"] == 7
    fkw = store.full_text_search.call_args_list[0].kwargs
    assert fkw["filters"] == {"corpus": "c", "profile": "fine"}


def test_run_rankers_without_filters_omits_kwarg() -> None:
    store = MagicMock()
    store.full_text_search.return_value = []
    run_rankers(
        store,
        "q",
        label="File",
        search_prop="summary",
        mode="fulltext",
        fetch_k=5,
        embedder=None,
        vector_capable=True,
    )
    store.full_text_search.assert_called_once_with("File", "summary", "q", k=5)


def test_run_rankers_embed_failure_degrades_to_fulltext() -> None:
    store = MagicMock()
    store.full_text_search.return_value = [_row("f")]
    emb = MagicMock()
    emb.embed.side_effect = RuntimeError("down")
    run = run_rankers(
        store,
        "q",
        label="Chunk",
        search_prop="text",
        mode="hybrid",
        fetch_k=5,
        embedder=emb,
        vector_capable=True,
    )
    assert not run.used_vector and len(run.rankers) == 1
    store.vector_search.assert_not_called()


def test_run_rankers_vector_leg_failure_keeps_fulltext() -> None:
    store = MagicMock()
    store.vector_search.side_effect = RuntimeError("index missing")
    store.full_text_search.return_value = [_row("f")]
    emb = MagicMock()
    emb.embed.return_value = [[0.1]]
    run = run_rankers(
        store,
        "q",
        label="Chunk",
        search_prop="text",
        mode="hybrid",
        fetch_k=5,
        embedder=emb,
        vector_capable=True,
    )
    assert not run.used_vector and len(run.rankers) == 1


def test_run_rankers_vector_mode_skips_fulltext() -> None:
    store = MagicMock()
    store.vector_search.return_value = [_row("v")]
    emb = MagicMock()
    emb.embed.return_value = [[0.1]]
    run = run_rankers(
        store,
        "q",
        label="Chunk",
        search_prop="text",
        mode="vector",
        fetch_k=5,
        embedder=emb,
        vector_capable=True,
    )
    assert run.used_vector and len(run.rankers) == 1
    store.full_text_search.assert_not_called()


def test_fuse_rankers_matches_two_ranker_form_and_sums_across_profiles() -> None:
    v = [_row("a"), _row("b")]
    f = [_row("b"), _row("c")]
    assert fuse_rankers([(v, 1.0), (f, 1.0)], label="Section", limit=5) == reciprocal_rank_fusion(
        v, f, label="Section", limit=5
    )
    fused = fuse_rankers([(v, 1.0), (f, 1.0), ([_row("c")], 3.0)], label="Section", limit=5)
    assert fused[0]["id"] == "c"  # heavily weighted third ranker wins
