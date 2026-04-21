"""Shared JSON-body extractor for provider responses.

LLMs occasionally wrap JSON output in prose preambles ("Here is the JSON:")
or non-``json`` fence language tags (```yaml, ```text) despite
prompt instructions to emit JSON only. A first-``{``-to-last-``}`` slice
handles all three cases (bare JSON, any fence tag, surrounding prose) in one
line without regex fragility.
"""

from __future__ import annotations


def extract_json_body(response: str) -> str:
    """Return the substring from the first ``{`` to the last ``}`` inclusive.

    Tolerates language-tagged code fences and prose around the JSON block.
    Raises ``ValueError`` if the response contains no balanced JSON object.
    """
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Provider response contains no JSON object; got {response!r}")
    return response[start : end + 1]
