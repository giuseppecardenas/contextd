import logging
from unittest.mock import MagicMock, patch

import pytest

from contextd.config import VoyageConfig
from contextd.providers.voyage import VoyageProvider, _estimate_tokens


def test_embed_returns_vectors() -> None:
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 10
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_result.total_tokens = 50
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        result = provider.embed(["hello", "world"])
    assert len(result) == 2
    assert all(len(v) == 1024 for v in result)
    assert provider.dimensions == 1024


def test_batches_respect_max_batch_size() -> None:
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=2)
    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 10
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_result.total_tokens = 50
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["a", "b", "c", "d"])
    # 4 inputs, batch size 2 → 2 calls.
    assert mock_client.embed.call_count == 2


def test_batches_respect_token_budget() -> None:
    # Each text counts as 60k tokens; with the 100k per-batch budget no two
    # texts fit together, so three texts force three separate embed calls even
    # though max_batch_size is far higher.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 60_000
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 60_000
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["a", "b", "c"])
    assert mock_client.embed.call_count == 3


def test_empty_strings_coerced_to_space() -> None:
    # Voyage rejects empty inputs; an empty/blank file must still be sent as a
    # non-empty token so one-vector-per-input alignment is preserved.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 0
    mock_result = MagicMock()
    mock_result.embeddings = [[0.0] * 1024, [0.0] * 1024]
    mock_result.total_tokens = 0
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["", "   "])
    sent_batch = mock_client.embed.call_args.args[0]
    assert sent_batch == [" ", " "]


def test_retries_on_rate_limit() -> None:
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    from voyageai.error import RateLimitError

    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 1
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 1
    mock_client.embed.side_effect = [RateLimitError("limit"), mock_result]
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key", backoff_initial=0.01)
        result = provider.embed(["x"])
    assert len(result) == 1
    assert mock_client.embed.call_count == 2


def test_retries_on_timeout() -> None:
    # A stalled Voyage socket surfaces as voyageai.error.Timeout once the client
    # request timeout fires; it must be retried, not propagated, so a transient
    # stall does not fail the enumerate phase on the first hiccup.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    from voyageai.error import Timeout

    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 1
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 1
    mock_client.embed.side_effect = [Timeout("read timed out"), mock_result]
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key", backoff_initial=0.01)
        result = provider.embed(["x"])
    assert len(result) == 1
    assert mock_client.embed.call_count == 2


def test_retries_on_connection_error() -> None:
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    from voyageai.error import APIConnectionError

    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 1
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 1
    mock_client.embed.side_effect = [APIConnectionError("connection reset"), mock_result]
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key", backoff_initial=0.01)
        result = provider.embed(["x"])
    assert len(result) == 1
    assert mock_client.embed.call_count == 2


def test_reraises_transport_error_after_max_retries() -> None:
    # A persistently unreachable endpoint must fail the phase (raise) rather than
    # retry forever; call_count equals max_retries.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    from voyageai.error import Timeout

    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 1
    mock_client.embed.side_effect = Timeout("perma-stall")
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key", backoff_initial=0.01, max_retries=3)
        with pytest.raises(Timeout):
            provider.embed(["x"])
    assert mock_client.embed.call_count == 3


def test_embeds_when_tokenizer_unavailable() -> None:
    # voyageai raises ImportError from count_tokens when the optional
    # `tokenizers` extra is absent. That must not fail the embed: batch sizing
    # falls back to a character estimate and the request still goes out.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    mock_client = MagicMock()
    mock_client.count_tokens.side_effect = ImportError("The package `tokenizers` is not found.")
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_result.total_tokens = 7
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        result = provider.embed(["hello", "world"])
    assert len(result) == 2
    assert mock_client.embed.call_count == 1


def test_tokenizer_failure_is_latched_after_first_attempt() -> None:
    # _token_aware_batches counts every text individually, so an unlatched
    # failure would retry the broken tokenizer once per file in the corpus —
    # a failing import or a Hub round-trip per file. It must be attempted once.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=1)
    mock_client = MagicMock()
    mock_client.count_tokens.side_effect = OSError("hub unreachable")
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 1
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["a", "b", "c", "d", "e"])
    assert mock_client.count_tokens.call_count == 1
    assert provider._tokenizer_available is False


def test_tokenizer_failure_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    mock_client = MagicMock()
    mock_client.count_tokens.side_effect = ImportError("nope")
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 1
    mock_client.embed.return_value = mock_result
    with (
        patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client),
        caplog.at_level(logging.WARNING, logger="contextd.providers.voyage"),
    ):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["a", "b", "c"])
    warnings = [r for r in caplog.records if "tokenizer unavailable" in r.getMessage()]
    assert len(warnings) == 1


def test_char_estimate_still_bounds_batches_by_token_budget() -> None:
    # The degraded path must keep honouring the per-request token ceiling.
    # 200k chars at 3 chars/token estimates ~66.7k tokens, so two texts cannot
    # share a batch under the 100k budget even though max_batch_size allows it.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    mock_client = MagicMock()
    mock_client.count_tokens.side_effect = ImportError("nope")
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 0
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["a" * 200_000, "b" * 200_000])
    assert mock_client.embed.call_count == 2


def test_working_tokenizer_is_not_bypassed() -> None:
    # The fallback must not shadow a healthy tokenizer: with count_tokens
    # working, its counts drive batching, not the character estimate.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 60_000
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 60_000
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["x", "y"])
    # Two short strings would share a batch under the char estimate; the real
    # counts split them.
    assert mock_client.embed.call_count == 2
    assert provider._tokenizer_available is True


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        ([], 0),
        ([""], 0),
        (["a"], 1),  # rounds up: a non-empty text is never zero tokens
        (["abc"], 1),
        (["abcd"], 2),
        (["abc", "abc"], 2),
    ],
)
def test_estimate_tokens(texts: list[str], expected: int) -> None:
    assert _estimate_tokens(texts) == expected


def test_client_constructed_with_request_timeout() -> None:
    # The bootstrap-hang regression guard: the voyageai client must be built with
    # an explicit timeout so a stalled embed cannot block phase_enumerate forever.
    cfg = VoyageConfig(model="voyage-4-large", max_batch_size=128)
    with patch("contextd.providers.voyage.voyageai.Client") as mock_ctor:
        VoyageProvider(cfg, api_key="test-key", request_timeout_s=42.0)
    assert mock_ctor.call_args.kwargs["timeout"] == 42.0
