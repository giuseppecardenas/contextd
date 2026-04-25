"""RoutingInferenceProvider — dispatches generate() by call_site.

Wraps three concrete providers (one per call-site: summary, inference,
translation) and forwards each ``PromptRequest`` to the appropriate
slot. Lets users mix cloud + local backends — e.g., summary+inference
on Ollama, translation on Gemini.

When two call-sites resolve to the same backend, the factory passes the
*same* InferenceProvider instance into multiple slots, so retry state
and the underlying HTTP client are shared.
"""

from __future__ import annotations

from contextd.providers.base import CallSite, InferenceProvider, PromptRequest, UsageRecord


class RoutingInferenceProvider(InferenceProvider):
    def __init__(
        self,
        *,
        summary: InferenceProvider,
        inference: InferenceProvider,
        translation: InferenceProvider,
    ) -> None:
        self._slots: dict[CallSite, InferenceProvider] = {
            "summary": summary,
            "inference": inference,
            "translation": translation,
        }

    def generate(self, request: PromptRequest) -> str:
        return self._slots[request.call_site].generate(request)

    def last_usage(self) -> UsageRecord | None:
        # Most-recent across the three slots, by ISO-8601 timestamp
        # (lexicographic == chronological for ISO-8601). Slots may
        # share an instance; dict.values() de-dupe via id() avoids
        # double-considering the same provider.
        seen: set[int] = set()
        records: list[UsageRecord] = []
        for provider in self._slots.values():
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            usage = provider.last_usage()
            if usage is not None:
                records.append(usage)
        if not records:
            return None
        return max(records, key=lambda r: r.timestamp)
