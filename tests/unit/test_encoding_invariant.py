"""Every text-mode file I/O in contextd must pin its encoding.

``Path.read_text`` / ``Path.write_text`` / ``open`` default to
``locale.getpreferredencoding(False)``. That is UTF-8 on a typical Linux CI
runner and cp1252 on a stock Windows install, so a missing ``encoding=``
silently mangles any non-ASCII corpus content — em dashes, curly quotes,
accented characters — into mojibake that then gets embedded, summarised and
hashed. Nothing raises, because cp1252 maps most bytes to wrong-but-valid
characters, and the same file hashes differently per platform.

This is asserted over the source rather than at runtime because the defect
cannot reproduce on a UTF-8 CI runner: a behavioural test would pass on CI
while the bug was live on Windows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import contextd

_IO_FUNCS = {"read_text", "write_text", "open"}

# Binary modes take no encoding; a mode argument containing "b" exempts a call.
_BINARY_HINT = "b"


def _is_binary_call(node: ast.Call) -> bool:
    for arg in node.args:
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and _BINARY_HINT in arg.value
        ):
            return True
    for kw in node.keywords:
        if (
            kw.arg == "mode"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
            and _BINARY_HINT in kw.value.value
        ):
            return True
    return False


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def test_all_text_io_pins_encoding() -> None:
    package_root = Path(contextd.__file__).parent
    offenders: list[str] = []

    for py in sorted(package_root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in _IO_FUNCS:
                continue
            if _is_binary_call(node):
                continue
            if any(kw.arg == "encoding" for kw in node.keywords):
                continue
            rel = py.relative_to(package_root.parent).as_posix()
            offenders.append(f"{rel}:{node.lineno}: {name}() without encoding=")

    assert not offenders, "text I/O without an explicit encoding:\n" + "\n".join(offenders)
