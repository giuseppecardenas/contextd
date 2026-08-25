"""Regex sentence splitter returning character spans.

Shared by ``sentence_window``, ``semantic`` and the recursive fallback. It is
deliberately small: sentence-final punctuation followed by whitespace, with a
guard for common abbreviations and decimals so ``e.g. 1.5x`` is not cut. No
NLTK download, no model — it must run on an offline install.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "etc",
        "vs",
        "cf",
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "no",
        "fig",
        "approx",
    }
)

# A break candidate: terminal punctuation (optionally closed by quotes or
# brackets) followed by whitespace. Newlines inside a paragraph count as
# whitespace; a blank line is always a break (handled separately).
# ASCII + CJK (U+3002 / U+FF01 / U+FF1F) full stop, exclamation, question.
_TERMINALS = ".!?\u3002\uff01\uff1f"
# Quotes (incl. curly U+201D / U+2019) and brackets that may follow the terminal.
_CLOSERS = "\"'\u201d\u2019)\\]"
_BREAK = re.compile(rf"([{_TERMINALS}](?:[{_CLOSERS}]+)?)(\s+)")
_BLANK = re.compile(r"\n[ \t]*\n")


def _is_abbreviation(text: str, end: int) -> bool:
    """``end`` is the index just past the terminal punctuation."""
    start = end - 1
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    raw = text[start:end].rstrip(".!?")
    if raw.lower() in _ABBREVIATIONS:
        return True
    # A lone capital letter is an initial ("J. Smith"); a lone digit or
    # lowercase letter at the end of a sentence ("version 2.", "option a.")
    # is a real terminator.
    return len(raw) == 1 and raw.isupper()


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` spans of sentences; whitespace-only gaps are
    excluded, so joining ``text[s:e]`` for all spans loses only separators."""
    spans: list[tuple[int, int]] = []
    if not text.strip():
        return spans
    breaks: list[int] = []
    for m in _BREAK.finditer(text):
        punct_end = m.end(1)
        # A decimal like "1.5" has no whitespace after the dot, so it never
        # matches; an abbreviation does, and is skipped here.
        if text[punct_end - 1] == "." and _is_abbreviation(text, punct_end):
            continue
        breaks.append(m.end())
    for m in _BLANK.finditer(text):
        breaks.append(m.end())
    breaks = sorted(set(breaks))
    cursor = 0
    for b in breaks:
        seg = text[cursor:b]
        s = cursor + (len(seg) - len(seg.lstrip()))
        e = cursor + len(seg.rstrip())
        if e > s:
            spans.append((s, e))
        cursor = b
    tail = text[cursor:]
    s = cursor + (len(tail) - len(tail.lstrip()))
    e = cursor + len(tail.rstrip())
    if e > s:
        spans.append((s, e))
    return spans
