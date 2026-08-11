"""OpenAI-compatible inference provider for local model servers.

Targets any HTTP server that speaks the OpenAI ``chat/completions`` shape:
Ollama (``/v1/`` mode), LM Studio, LocalAI, vLLM, llama.cpp's server.
Used to offload high-volume summary + relate traffic away from the
Gemini free-tier quota while letting `contextd ask` (translation) stay
on a stronger cloud model.

Model selection per call-site is driven by config (mirrors GeminiProvider).
``response_format={"type": "json_object"}`` is sent for the ``summary``
and ``inference`` call-sites (their prompts emit JSON); translation
prompts emit Cypher prose so JSON mode is suppressed there.

A 200 response carrying an empty completion is treated as a retryable
provider-side miss rather than a valid answer — see ``generate``.
"""

from __future__ import annotations

import datetime as dt
import random
import time

import httpx

from contextd.config import OpenAICompatConfig
from contextd.providers.base import CallSite, InferenceProvider, PromptRequest, UsageRecord

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class EmptyResponseError(RuntimeError):
    """Raised when a provider returns 200 with no completion text.

    Distinguished from a parse failure so the cause is legible in the indexer
    log: the model produced nothing, rather than producing something malformed.
    """


class OpenAICompatProvider(InferenceProvider):
    def __init__(
        self,
        config: OpenAICompatConfig,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        backoff_initial: float = 1.0,
        backoff_max: float = 60.0,
    ) -> None:
        self._cfg = config
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=config.request_timeout_seconds)
        self._last_usage: UsageRecord | None = None
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

    def generate(self, request: PromptRequest) -> str:
        model = self._model_for(request.call_site)
        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        body: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
        }
        # Cap output tokens when configured (prevents runaway generation on
        # local models that otherwise produce thousands of tokens).
        if self._cfg.max_output_tokens is not None:
            body["max_tokens"] = self._cfg.max_output_tokens

        # JSON mode for prompts that expect well-formed JSON output.
        # Translation emits Cypher prose, so leave response_format unset there.
        if self._cfg.json_mode and request.call_site != "translation":
            body["response_format"] = {"type": "json_object"}

        # Low temperature suits the JSON-extraction call sites; translation
        # keeps the server default (mirrors the json_mode carve-out above).
        if self._cfg.temperature is not None and request.call_site != "translation":
            body["temperature"] = self._cfg.temperature

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempt = 0
        # Accumulated across attempts, not overwritten: a discarded empty
        # completion still burned prompt tokens, and `contextd costs` must not
        # under-report spend just because the answer was unusable.
        input_tokens = 0
        output_tokens = 0

        while True:
            try:
                response = self._client.post(url, json=body, headers=headers)
                if response.status_code in _RETRYABLE_CODES:
                    attempt += 1
                    if attempt >= self._cfg.max_retries:
                        response.raise_for_status()
                    self._sleep_backoff(attempt)
                    continue
                response.raise_for_status()
            except (httpx.TransportError, httpx.TimeoutException):
                attempt += 1
                if attempt >= self._cfg.max_retries:
                    raise
                self._sleep_backoff(attempt)
                continue

            payload = response.json()
            text = str(payload["choices"][0]["message"]["content"] or "")
            usage = payload.get("usage") or {}
            input_tokens += int(usage.get("prompt_tokens", 0) or 0)
            output_tokens += int(usage.get("completion_tokens", 0) or 0)

            if text.strip():
                break

            # An empty completion under a 200 is a provider-side miss, not an
            # answer: every call-site prompt requires content, so an empty
            # string can only fail downstream (JSON parse for summary and
            # inference, Cypher execution for translation). Retrying on the
            # same bounded backoff as a 503 costs one more call and usually
            # succeeds; returning "" costs a whole section its summary.
            attempt += 1
            if attempt >= self._cfg.max_retries:
                self._record_usage(model, request, input_tokens, output_tokens)
                raise EmptyResponseError(
                    f"{model} returned an empty completion for call-site "
                    f"{request.call_site!r} after {attempt} attempts"
                )
            self._sleep_backoff(attempt)

        self._record_usage(model, request, input_tokens, output_tokens)
        return text

    def _record_usage(
        self, model: str, request: PromptRequest, input_tokens: int, output_tokens: int
    ) -> None:
        """Store usage for the completed call.

        Called before raising on exhaustion too, so spend on discarded attempts
        is still visible to whatever reads ``last_usage``.
        """
        self._last_usage = UsageRecord(
            provider="openai_compat",
            model=model,
            call_site=request.call_site,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
        )

    def last_usage(self) -> UsageRecord | None:
        return self._last_usage

    def _model_for(self, call_site: CallSite) -> str:
        match call_site:
            case "summary":
                return self._cfg.model_summary
            case "inference":
                return self._cfg.model_inference
            case "translation":
                return self._cfg.model_translation

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self._backoff_initial * (2 ** (attempt - 1)), self._backoff_max)
        delay *= 1.0 + random.uniform(-0.2, 0.2)
        time.sleep(delay)
