"""LLM-backed chunk helpers: alignment, fallbacks, template fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from contextd.chunking.augment import apply_keywords, static_keywords
from contextd.chunking.llm import contextualise, generate_questions, propositions, render_template
from contextd.chunking.model import Chunk, ChunkSpan
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import InferenceProvider, PromptRequest, UsageRecord


class FakeProvider(InferenceProvider):
    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.requests: list[PromptRequest] = []

    def generate(self, request: PromptRequest) -> str:
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    def last_usage(self) -> UsageRecord | None:
        return None


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(i, f"chunk text {i}", ChunkSpan(i, i + 1), 3) for i in range(n)]


@pytest.fixture
def renderer(tmp_path: Path) -> PromptRenderer:
    # An empty user prompt dir: every template resolves via the packaged fallback.
    return PromptRenderer(tmp_path)


def test_render_template_falls_back_to_packaged(renderer: PromptRenderer) -> None:
    out = render_template(
        renderer, "contextualise", breadcrumb="B", document_summary="S", chunks="C", count="1"
    )
    assert "B" in out and "S" in out and "[" not in out.split("Chunks:")[0][:0]
    assert "{{" not in out


def test_render_template_prefers_user_copy(tmp_path: Path) -> None:
    (tmp_path / "propositions.md").write_text("custom {{content}}", encoding="utf-8")
    out = render_template(
        PromptRenderer(tmp_path), "propositions", breadcrumb="", content="X", max_propositions="3"
    )
    assert out == "custom X"


def test_contextualise_aligned_response(renderer: PromptRenderer) -> None:
    provider = FakeProvider(json.dumps({"contexts": ["ctx a", "ctx b"]}))
    out = contextualise(
        provider, renderer, _chunks(2), breadcrumb="Doc > Sec", document_summary="sum"
    )
    assert out == ["Doc > Sec\nctx a", "Doc > Sec\nctx b"]
    assert len(provider.requests) == 1 and provider.requests[0].call_site == "summary"
    assert "[0]\nchunk text 0" in provider.requests[0].prompt


def test_contextualise_misaligned_falls_back(
    renderer: PromptRenderer, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeProvider(json.dumps({"contexts": ["only one"]}))
    with caplog.at_level(logging.WARNING):
        out = contextualise(provider, renderer, _chunks(2), breadcrumb="B", document_summary="")
    assert out == ["B", "B"]
    assert "using breadcrumb" in caplog.text


def test_contextualise_provider_failure_falls_back(renderer: PromptRenderer) -> None:
    provider = FakeProvider(RuntimeError("boom"))
    assert contextualise(provider, renderer, _chunks(3), breadcrumb="B", document_summary="") == [
        "B",
        "B",
        "B",
    ]
    assert contextualise(provider, renderer, [], breadcrumb="B", document_summary="") == []


def test_contextualise_empty_entry_uses_breadcrumb(renderer: PromptRenderer) -> None:
    provider = FakeProvider(json.dumps({"contexts": ["", None]}))
    assert contextualise(provider, renderer, _chunks(2), breadcrumb="B", document_summary="") == [
        "B",
        "B",
    ]


def test_generate_questions(renderer: PromptRenderer) -> None:
    provider = FakeProvider(json.dumps({"questions": [["q1", "q2", "q3"], []]}))
    out = generate_questions(provider, renderer, _chunks(2), breadcrumb="B", per_chunk=2)
    assert out == [["q1", "q2"], []]


def test_generate_questions_failure_yields_empty_lists(renderer: PromptRenderer) -> None:
    assert generate_questions(
        FakeProvider(ValueError("x")), renderer, _chunks(2), breadcrumb="B"
    ) == [
        [],
        [],
    ]
    assert generate_questions(
        FakeProvider(json.dumps({"questions": [["a"]]})), renderer, _chunks(2), breadcrumb="B"
    ) == [[], []]


def test_propositions_success_and_cap(renderer: PromptRenderer) -> None:
    provider = FakeProvider(json.dumps({"propositions": ["p1", " p2 ", "", "p3"]}))
    assert propositions(provider, renderer, "text", breadcrumb="B", max_propositions=2) == [
        "p1",
        "p2",
    ]


def test_propositions_failure_returns_none(renderer: PromptRenderer) -> None:
    assert (
        propositions(
            FakeProvider(RuntimeError()), renderer, "t", breadcrumb="B", max_propositions=5
        )
        is None
    )
    assert (
        propositions(
            FakeProvider(json.dumps({"propositions": []})),
            renderer,
            "t",
            breadcrumb="B",
            max_propositions=5,
        )
        is None
    )
    assert (
        propositions(FakeProvider("{}"), renderer, "   ", breadcrumb="B", max_propositions=5) == []
    )


def test_static_keywords_and_apply() -> None:
    kws = static_keywords(
        ["key_points", "entities_mentioned"],
        key_points=["a", " b ", "", "a"],
        entities_mentioned=["b", "c"],
    )
    assert kws == ["a", "b", "c"]
    assert static_keywords(["questions"], key_points=["a"], entities_mentioned=["b"]) == []
    chunks = _chunks(2)
    apply_keywords(chunks, ["shared"], [["own0"], []])
    assert chunks[0].keywords == ["shared", "own0"] and chunks[1].keywords == ["shared"]
    apply_keywords(chunks, ["s"], None)
    assert chunks[1].keywords == ["s"]
