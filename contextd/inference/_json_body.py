"""Shared JSON-body extractor for provider responses.

LLMs occasionally wrap JSON output in prose preambles ("Here is the JSON:")
or non-``json`` fence language tags (```yaml, ```text) despite
prompt instructions to emit JSON only. A first-``{``-to-last-``}`` slice
handles all three cases (bare JSON, any fence tag, surrounding prose) in one
line without regex fragility.

Models also emit trailing commas before a closing ``}`` or ``]`` — JavaScript
and Python both accept them, ``json.loads`` does not, and enabling a provider's
JSON mode does not reliably suppress them. :func:`loads_json_body` repairs that
one deviation before parsing so a stray comma does not cost a whole summary or
relate call.
"""

from __future__ import annotations

import json
from typing import Any, cast


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


def _strip_trailing_commas(body: str) -> str:
    """Drop commas that directly precede a closing ``}`` or ``]``.

    Scans with string-literal awareness rather than substituting on a regex:
    summaries and ``key_points`` are natural-language prose that can legitimately
    contain a ``, }`` or ``, ]`` sequence inside a quoted value, and a blind
    substitution would silently rewrite the model's content. Only commas found
    outside a string literal are considered, and backslash escapes are tracked so
    an escaped quote does not appear to end the string.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for i, char in enumerate(body):
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            out.append(char)
            continue
        if char == ",":
            nxt = i + 1
            while nxt < len(body) and body[nxt].isspace():
                nxt += 1
            if nxt < len(body) and body[nxt] in "}]":
                continue
        out.append(char)
    return "".join(out)


def loads_json_body(response: str) -> dict[str, Any]:
    """Extract and parse the JSON object embedded in ``response``.

    Parses the extracted body as-is first, so well-formed output takes an
    untouched path and the repair can never alter valid JSON. Only when strict
    parsing fails is :func:`_strip_trailing_commas` applied and the parse
    retried; a failure on the retry propagates, and reports its position within
    the repaired body.

    :raises ValueError: if the response contains no JSON object, or if the
        parsed value is not a JSON object.
    """
    body = extract_json_body(response)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = json.loads(_strip_trailing_commas(body))
    if not isinstance(parsed, dict):
        raise ValueError(f"Provider response JSON is not an object; got {type(parsed).__name__}")
    return cast(dict[str, Any], parsed)
