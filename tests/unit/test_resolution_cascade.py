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
