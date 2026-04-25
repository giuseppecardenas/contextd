"""Unit tests for RoutingInferenceProvider."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

from contextd.providers.base import InferenceProvider, PromptRequest, UsageRecord
from contextd.providers.router import RoutingInferenceProvider


def _provider(name: str, response: str = "out") -> MagicMock:
    p = MagicMock(spec=InferenceProvider)
    p.generate.return_value = response
    p.last_usage.return_value = None
    p._name = name
    return p


def test_router_dispatches_summary_call_to_summary_provider() -> None:
    s, i, t = _provider("s"), _provider("i"), _provider("t")
    router = RoutingInferenceProvider(summary=s, inference=i, translation=t)
    router.generate(PromptRequest(system="x", prompt="y", call_site="summary"))
    assert s.generate.call_count == 1
    assert i.generate.call_count == 0
    assert t.generate.call_count == 0


def test_router_dispatches_translation_call_to_translation_provider() -> None:
    s, i, t = _provider("s"), _provider("i"), _provider("t")
    router = RoutingInferenceProvider(summary=s, inference=i, translation=t)
    router.generate(PromptRequest(system="x", prompt="y", call_site="translation"))
    assert t.generate.call_count == 1
    assert s.generate.call_count == 0
    assert i.generate.call_count == 0


def test_router_dispatches_inference_call_to_inference_provider() -> None:
    s, i, t = _provider("s"), _provider("i"), _provider("t")
    router = RoutingInferenceProvider(summary=s, inference=i, translation=t)
    router.generate(PromptRequest(system="x", prompt="y", call_site="inference"))
    assert i.generate.call_count == 1


def test_router_returns_provider_response() -> None:
    s, i, t = _provider("s", "summary-out"), _provider("i"), _provider("t")
    router = RoutingInferenceProvider(summary=s, inference=i, translation=t)
    out = router.generate(PromptRequest(system="x", prompt="y", call_site="summary"))
    assert out == "summary-out"


def test_router_last_usage_returns_most_recent_across_providers() -> None:
    s, i, t = _provider("s"), _provider("i"), _provider("t")
    older = UsageRecord(
        provider="gemini",
        model="m",
        call_site="summary",
        input_tokens=1,
        output_tokens=1,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    newer = UsageRecord(
        provider="openai_compat",
        model="m",
        call_site="translation",
        input_tokens=2,
        output_tokens=2,
        timestamp="2026-04-24T12:00:00+00:00",
    )
    s.last_usage.return_value = older
    i.last_usage.return_value = None
    t.last_usage.return_value = newer
    router = RoutingInferenceProvider(summary=s, inference=i, translation=t)
    assert router.last_usage() == newer


def test_router_last_usage_returns_none_when_no_provider_has_usage() -> None:
    s, i, t = _provider("s"), _provider("i"), _provider("t")
    router = RoutingInferenceProvider(summary=s, inference=i, translation=t)
    assert router.last_usage() is None


def test_router_shares_instance_when_same_provider_in_multiple_slots() -> None:
    """Factory may pass the same provider into multiple slots when call-sites
    map to the same backend. last_usage must not double-count it."""
    shared = _provider("shared")
    rec = UsageRecord(
        provider="openai_compat",
        model="m",
        call_site="summary",
        input_tokens=5,
        output_tokens=5,
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
    )
    shared.last_usage.return_value = rec
    other = _provider("other")
    router = RoutingInferenceProvider(summary=shared, inference=shared, translation=other)
    # generate() against any shared-backed call_site uses the same instance.
    router.generate(PromptRequest(system="x", prompt="y", call_site="inference"))
    assert shared.generate.call_count == 1
    # last_usage should consult shared once, not twice.
    assert router.last_usage() == rec
    assert shared.last_usage.call_count == 1
