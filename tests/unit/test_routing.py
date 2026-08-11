"""SummaryPromptRouter + Summariser routing order."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from contextd.inference.context import UnitIdentity
from contextd.inference.routing import PromptRoute, SummaryPromptRouter
from contextd.inference.summarise import Summariser


def _identity(rel_path: str) -> UnitIdentity:
    return UnitIdentity(
        corpus="c",
        file_path=f"C:/x/{rel_path}",
        rel_path=rel_path,
        suffix="." + rel_path.rsplit(".", 1)[-1],
        src_label="File",
        src_id=f"C:/x/{rel_path}",
    )


def test_first_match_wins_and_no_match_returns_none() -> None:
    router = SummaryPromptRouter(
        [
            PromptRoute(pattern="**/*.lua", path=Path("C:/t/code.md")),
            PromptRoute(pattern="mods/**", path=Path("C:/t/mods.md")),
        ]
    )
    assert router.resolve("mods/base/actions/feudal.lua") == Path("C:/t/code.md")
    assert router.resolve("docs/prd/03-economy.md") is None
    assert router.resolve(None) is None
    assert router.resolve("") is None


def test_summariser_resolution_order_router_then_override_then_default(
    tmp_path: Path,
) -> None:
    routed_tpl = tmp_path / "code.md"
    routed_tpl.write_text("code {{content}}", encoding="utf-8")
    override_tpl = tmp_path / "corpus.md"
    override_tpl.write_text("corpus {{content}}", encoding="utf-8")

    provider = MagicMock()
    provider.generate.return_value = json.dumps(
        {"summary": "s", "key_points": [], "entities_mentioned": []}
    )
    renderer = MagicMock()
    renderer.render.return_value = "p"
    renderer.render_path.return_value = "p"
    router = SummaryPromptRouter([PromptRoute(pattern="**/*.lua", path=routed_tpl)])
    summariser = Summariser(
        provider=provider, renderer=renderer, prompt_path=override_tpl, router=router
    )

    # .lua routes to the code template despite the corpus-wide override.
    summariser.summarise("body", context=_identity("mods/base/feudal.lua"))
    assert renderer.render_path.call_args.args == (routed_tpl,)
    # .md falls back to the corpus-wide override.
    summariser.summarise("body", context=_identity("docs/prd/03.md"))
    assert renderer.render_path.call_args.args == (override_tpl,)
    # No context at all → corpus-wide override still applies.
    summariser.summarise("body")
    assert renderer.render_path.call_args.args == (override_tpl,)


def test_summariser_router_without_override_falls_back_to_packaged(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.generate.return_value = json.dumps(
        {"summary": "s", "key_points": [], "entities_mentioned": []}
    )
    renderer = MagicMock()
    renderer.render.return_value = "p"
    router = SummaryPromptRouter([PromptRoute(pattern="**/*.lua", path=tmp_path / "c.md")])
    summariser = Summariser(provider=provider, renderer=renderer, router=router)
    summariser.summarise("body", context=_identity("docs/a.md"))
    renderer.render.assert_called_once()
    assert renderer.render.call_args.args == ("summarise",)
