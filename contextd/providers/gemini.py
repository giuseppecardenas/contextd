"""Google Gemini inference provider using the google-genai SDK.

Model selection per call site is driven by config (spec §4.2); there
is no single global override. Safety settings are pinned to
``BLOCK_NONE`` for all four harm categories because Contextd indexes
user-owned files that regularly trip default safety thresholds
(technical / security writing) — silent blocks would drop legitimate
indexing work.
"""

from __future__ import annotations

import datetime as dt
import random
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from contextd.config import GeminiConfig
from contextd.providers.base import CallSite, InferenceProvider, PromptRequest, UsageRecord

_RETRYABLE_CODES = {429, 500, 503}


class GeminiProvider(InferenceProvider):
    def __init__(
        self,
        config: GeminiConfig,
        *,
        api_key: str,
        backoff_initial: float = 1.0,
        backoff_max: float = 60.0,
    ) -> None:
        self._cfg = config
        self._client = genai.Client(api_key=api_key)
        self._last_usage: UsageRecord | None = None
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

    def generate(self, request: PromptRequest) -> str:
        model = self._model_for(request.call_site)
        safety = self._safety_settings()
        config = genai_types.GenerateContentConfig(
            system_instruction=request.system,
            safety_settings=safety,
        )

        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=request.prompt,
                    config=config,
                )
                break
            except genai_errors.APIError as exc:
                attempt += 1
                if exc.code not in _RETRYABLE_CODES or attempt >= self._cfg.max_retries:
                    raise
                delay = min(self._backoff_initial * (2 ** (attempt - 1)), self._backoff_max)
                delay *= 1.0 + random.uniform(-0.2, 0.2)
                time.sleep(delay)

        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count if usage else None) or 0
        output_tokens = (usage.candidates_token_count if usage else None) or 0
        self._last_usage = UsageRecord(
            provider="gemini",
            model=model,
            call_site=request.call_site,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
        )
        return response.text or ""

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

    def _safety_settings(self) -> list[genai_types.SafetySetting]:
        threshold = genai_types.HarmBlockThreshold(self._cfg.safety_block)
        categories = [
            genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        return [genai_types.SafetySetting(category=c, threshold=threshold) for c in categories]
