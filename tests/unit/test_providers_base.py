import inspect

import pytest

from contextd.providers.base import (
    EmbeddingProvider,
    InferenceProvider,
    PromptRequest,
    UsageRecord,
)


def test_inference_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        InferenceProvider()  # type: ignore[abstract]


def test_embedding_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore[abstract]


def test_inference_provider_has_generate() -> None:
    methods = {
        name for name, _ in inspect.getmembers(InferenceProvider, predicate=inspect.isfunction)
    }
    assert "generate" in methods


def test_embedding_provider_has_embed() -> None:
    methods = {
        name for name, _ in inspect.getmembers(EmbeddingProvider, predicate=inspect.isfunction)
    }
    assert "embed" in methods


def test_prompt_request_fields() -> None:
    req = PromptRequest(system="sys", prompt="hi", call_site="summary")
    assert req.system == "sys"
    assert req.prompt == "hi"
    assert req.call_site == "summary"


def test_usage_record_is_frozen() -> None:
    record = UsageRecord(
        provider="gemini",
        model="gemma-4-31b-it",
        call_site="summary",
        input_tokens=100,
        output_tokens=20,
        timestamp="2026-04-20T12:00:00Z",
    )
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.input_tokens = 999  # type: ignore[misc]
