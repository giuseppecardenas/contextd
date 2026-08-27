"""Entity-resolution cascade: normalization policy + exact-norm rung.

`Tailor`/`tailor` (a prose Pattern) must collapse into one node while
case-sensitive kinds (Repo) never casefold-merge; intra-batch duplicates
collapse through the mint-updating cache; ambiguity never auto-matches.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from contextd.indexer.phases import _apply_inferred_edge
from contextd.indexer.resolution import (
    EntityCascadeResolver,
    ResolutionSettings,
    normalize_name,
)
from contextd.inference.relate import InferredRelationship


def _resolver(
    rows: list[dict[str, object]] | None = None,
    settings: ResolutionSettings | None = None,
) -> tuple[EntityCascadeResolver, MagicMock]:
    store = MagicMock()
    store.exec_read.return_value = rows or []
    return EntityCascadeResolver(store, settings or ResolutionSettings()), store


def test_normalize_name_collapses_whitespace_and_case_on_request() -> None:
    assert normalize_name("  Spatial \t Hash ", casefold=True) == "spatial hash"
    assert normalize_name("  Spatial \t Hash ", casefold=False) == "Spatial Hash"


def test_case_variant_matches_existing_for_prose_kind() -> None:
    resolver, _ = _resolver(rows=[{"name": "tailor", "name_norm": "tailor"}])
    r = resolver.resolve("Pattern", "Tailor", "c")
    assert r.action == "matched"
    assert r.pk_value == "tailor"
    assert r.rule == "exact-norm"


def test_case_variant_mints_for_case_sensitive_kind() -> None:
    resolver, _ = _resolver(rows=[{"name": "rl_core", "name_norm": "rl_core"}])
    r = resolver.resolve("Repo", "RL_Core", "c")
    assert r.action == "minted"
    assert r.pk_value == "RL_Core"


def test_whitespace_variant_matches() -> None:
    resolver, _ = _resolver(
        rows=[{"name": "Steam Workshop integration", "name_norm": "steam workshop integration"}]
    )
    r = resolver.resolve("Integration", "Steam  Workshop\tIntegration", "c")
    assert r.action == "matched"
    assert r.pk_value == "Steam Workshop integration"


def test_intra_batch_duplicates_collapse_via_mint_cache() -> None:
    resolver, store = _resolver(rows=[])
    first = resolver.resolve("Pattern", "Three-Phase Model", "c")
    second = resolver.resolve("Pattern", "three-phase model", "c")
    assert first.action == "minted"
    assert second.action == "matched"
    assert second.pk_value == "Three-Phase Model"
    # Cache loaded once, not per resolve.
    assert store.exec_read.call_count == 1


def test_legacy_rows_without_name_norm_are_normalized_on_load() -> None:
    resolver, _ = _resolver(rows=[{"name": "Spatial Hash Grid", "name_norm": None}])
    r = resolver.resolve("Pattern", "spatial hash grid", "c")
    assert r.action == "matched"
    assert r.pk_value == "Spatial Hash Grid"


def test_ambiguous_normalized_name_never_auto_matches() -> None:
    resolver, _ = _resolver(
        rows=[
            {"name": "Tailor", "name_norm": "tailor"},
            {"name": "tailor", "name_norm": "tailor"},
        ]
    )
    r = resolver.resolve("Pattern", "TAILOR", "c")
    assert r.action == "minted"  # falls through; guessing between the two is worse


def test_cache_load_failure_degrades_to_mint() -> None:
    store = MagicMock()
    store.exec_read.side_effect = RuntimeError("db down")
    resolver = EntityCascadeResolver(store, ResolutionSettings())
    r = resolver.resolve("Pattern", "spatial hash", "c")
    assert r.action == "minted"


# --- fuzzy rung ---------------------------------------------------------------


def test_near_duplicate_matches_by_fuzzy() -> None:
    resolver, _ = _resolver(
        rows=[{"name": "Steam Workshop Integrations", "name_norm": "steam workshop integrations"}]
    )
    r = resolver.resolve("Integration", "Steam Workshop Integration", "c")
    assert r.action == "matched"
    assert r.pk_value == "Steam Workshop Integrations"
    assert r.rule.startswith("fuzzy:")


def test_short_names_skip_fuzzy() -> None:
    resolver, _ = _resolver(
        rows=[{"name": "orc", "name_norm": "orc"}],
        settings=ResolutionSettings(fuzzy_min_length=6),
    )
    r = resolver.resolve("Pattern", "orcs", "c")
    assert r.action == "minted"  # 4 chars < gate — collision-prone, never fuzzed


def test_below_threshold_logs_ambiguous_and_mints() -> None:
    import logging

    import pytest  # noqa: F401  (caplog fixture import hint for readers)

    resolver, _ = _resolver(
        rows=[{"name": "economy simulation", "name_norm": "economy simulation"}],
        settings=ResolutionSettings(fuzzy_threshold=99.0),
    )
    logger = logging.getLogger("contextd.indexer.resolution")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        r = resolver.resolve("Service", "economy simulator", "c")
    finally:
        logger.removeHandler(handler)
    assert r.action == "minted"
    assert any("ambiguous-fuzzy" in rec.getMessage() for rec in records)


def test_fuzzy_never_matches_across_dissimilar_names() -> None:
    resolver, _ = _resolver(rows=[{"name": "spatial hash grid", "name_norm": "spatial hash grid"}])
    r = resolver.resolve("Pattern", "temporal amortisation", "c")
    assert r.action == "minted"


# --- embedding rung -----------------------------------------------------------


def test_embedding_match_reuses_supplied_vector_and_corpus_filters() -> None:
    store = MagicMock()
    store.exec_read.return_value = []  # empty norm cache
    store.vector_search.return_value = [
        {"node": {"name": "spatial hashing", "corpus": "c"}, "score": 0.95},
    ]
    embed = MagicMock(return_value=[[0.1] * 4])
    resolver = EntityCascadeResolver(store, ResolutionSettings(), embed=embed)
    r = resolver.resolve("Pattern", "spatial hash grid layout", "c")
    assert r.action == "matched"
    assert r.pk_value == "spatial hashing"
    assert r.rule.startswith("embedding:")
    embed.assert_called_once_with(["spatial hash grid layout"])
    assert store.vector_search.call_args.kwargs["query"] == [0.1] * 4
    # The corpus scope is pushed down to the backend as a server-side filter
    # (an other-corpus twin never reaches the resolver).
    assert store.vector_search.call_args.kwargs["filters"] == {"corpus": "c"}


def test_embedding_miss_mints_with_vector() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    store.vector_search.return_value = []
    embed = MagicMock(return_value=[[0.2] * 4])
    resolver = EntityCascadeResolver(store, ResolutionSettings(), embed=embed)
    r = resolver.resolve("Pattern", "temporal amortisation", "c")
    assert r.action == "minted"
    assert r.vector == [0.2] * 4  # one embed call serves check AND mint


def test_embedding_disabled_skips_embed_call() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    embed = MagicMock()
    resolver = EntityCascadeResolver(
        store, ResolutionSettings(embedding_enabled=False), embed=embed
    )
    r = resolver.resolve("Pattern", "spatial hash", "c")
    assert r.action == "minted"
    assert r.vector is None
    embed.assert_not_called()


def test_embed_failure_degrades_to_vectorless_mint() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    embed = MagicMock(side_effect=RuntimeError("provider down"))
    resolver = EntityCascadeResolver(store, ResolutionSettings(), embed=embed)
    r = resolver.resolve("Pattern", "spatial hash", "c")
    assert r.action == "minted"
    assert r.vector is None


def test_mint_with_vector_writes_embedding_property() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    store.vector_search.return_value = []
    embed = MagicMock(return_value=[[0.3] * 4])
    resolver = EntityCascadeResolver(store, ResolutionSettings(), embed=embed)
    written = _apply_inferred_edge(
        store, "src.md", "File", _rel(), "c", resolver=resolver, settings=resolver.settings
    )
    assert written is True
    _, props = store.upsert_node.call_args.args
    assert props["embedding"] == [0.3] * 4
    assert props["name_norm"] == "spatial hash"


# --- _apply_inferred_edge integration ----------------------------------------


def _rel(**overrides: object) -> InferredRelationship:
    kwargs: dict[str, object] = {
        "edge_type": "REFERENCES",
        "target_type": "Pattern",
        "target_name": "Spatial Hash",
        "confidence": 0.9,
        "reason": "r",
    }
    kwargs.update(overrides)
    return InferredRelationship(**kwargs)  # type: ignore[arg-type]


def test_mint_writes_name_norm() -> None:
    store = MagicMock()
    resolver, _ = _resolver(rows=[])
    written = _apply_inferred_edge(
        store, "src.md", "File", _rel(), "c", resolver=resolver, settings=resolver.settings
    )
    assert written is True
    label, props = store.upsert_node.call_args.args
    assert label == "Pattern"
    assert props["name"] == "Spatial Hash"
    assert props["name_norm"] == "spatial hash"


def test_matched_entity_absorbs_edge_without_new_stub() -> None:
    store = MagicMock()
    resolver, _ = _resolver(rows=[{"name": "spatial hash", "name_norm": "spatial hash"}])
    written = _apply_inferred_edge(
        store, "src.md", "File", _rel(), "c", resolver=resolver, settings=resolver.settings
    )
    assert written is True
    # Edge lands on the existing node's PK, and no name_norm rewrite occurs.
    _, props = store.upsert_node.call_args.args
    assert props["name"] == "spatial hash"
    assert "name_norm" not in props
    edge_args = store.upsert_edge.call_args
    assert edge_args.args[1] == "spatial hash"


def test_id_like_name_skips_fuzzy_rung() -> None:
    # Runeledger regression: WRatio("FR-SUP-026", "FR-SUP") is exactly 90.0,
    # so the fuzzy rung folded every requirement id into its LLM-minted
    # family node. An id is exact-match-only.
    resolver, _ = _resolver(rows=[{"name": "FR-SUP", "name_norm": "FR-SUP"}])
    r = resolver.resolve("Ticket", "FR-SUP-026", "c")
    assert r.action == "minted"
    assert r.pk_value == "FR-SUP-026"


def test_id_like_name_skips_embedding_rung() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    store.vector_search.return_value = [
        {"node": {"name": "FR-STL-001..023", "corpus": "c"}, "score": 0.97},
    ]
    embed = MagicMock(return_value=[[0.1] * 4])
    resolver = EntityCascadeResolver(store, ResolutionSettings(), embed=embed)
    r = resolver.resolve("Ticket", "FR-STL-001", "c")
    assert r.action == "minted"
    assert r.vector is None
    embed.assert_not_called()
    store.vector_search.assert_not_called()


def test_id_like_name_still_matches_exact_norm() -> None:
    resolver, _ = _resolver(rows=[{"name": "FR-SUP-026", "name_norm": "FR-SUP-026"}])
    r = resolver.resolve("Ticket", "FR-SUP-026", "c")
    assert r.action == "matched"
    assert r.rule == "exact-norm"


def test_code_symbols_skip_fuzzy_rung() -> None:
    # Second runeledger regression: WRatio scores a substring 100, so
    # register_material_class folded into register_material and
    # register_price_tier into register_action_tier. A snake_case symbol is
    # an identifier even without a digit.
    resolver, _ = _resolver(
        rows=[
            {"name": "register_material", "name_norm": "register_material"},
            {"name": "register_action_tier", "name_norm": "register_action_tier"},
        ]
    )
    assert resolver.resolve("Pattern", "register_material_class", "c").action == "minted"
    assert resolver.resolve("Pattern", "register_price_tier", "c").action == "minted"
    # Exact (case-folded, Pattern is case-insensitive) still matches.
    assert resolver.resolve("Pattern", "Register_Material", "c").pk_value == "register_material"


def test_prose_names_still_fuzzy_match() -> None:
    # The guard must not swallow the case the fuzzy rung exists for.
    resolver, _ = _resolver(
        rows=[{"name": "Steam Workshop Integrations", "name_norm": "steam workshop integrations"}]
    )
    assert resolver.resolve("Integration", "Steam Workshop Integration", "c").action == "matched"


def test_exact_only_pattern_is_configurable() -> None:
    # Empty pattern disables the guard: the id fuzzes into the family node
    # again, which is the pre-fix behaviour a corpus may explicitly want.
    resolver, _ = _resolver(
        rows=[{"name": "FR-SUP", "name_norm": "FR-SUP"}],
        settings=ResolutionSettings(exact_only_pattern=""),
    )
    assert resolver.resolve("Ticket", "FR-SUP-026", "c").action == "matched"
    # A narrower pattern leaves digit-carrying prose names to the fuzzy rung.
    resolver, _ = _resolver(
        rows=[{"name": "Phase 9 closure waves", "name_norm": "phase 9 closure waves"}],
        settings=ResolutionSettings(exact_only_pattern=r"^[A-Z]{2,}-"),
    )
    assert resolver.resolve("Pattern", "Phase 9 closure wave", "c").action == "matched"


def test_settings_confidence_floor_is_honoured() -> None:
    store = MagicMock()
    resolver, _ = _resolver(settings=ResolutionSettings(confidence_floor=0.95))
    written = _apply_inferred_edge(
        store,
        "src.md",
        "File",
        _rel(confidence=0.9),
        "c",
        resolver=resolver,
        settings=resolver.settings,
    )
    assert written is False
    store.upsert_edge.assert_not_called()
