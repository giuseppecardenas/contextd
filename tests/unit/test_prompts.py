from pathlib import Path

import pytest

from contextd.inference.prompts import PromptRenderer


def test_renders_known_template(tmp_path: Path) -> None:
    template_dir = tmp_path / "prompts"
    template_dir.mkdir()
    (template_dir / "summarise.md").write_text("hello {{name}}")
    renderer = PromptRenderer(template_dir)
    result = renderer.render("summarise", name="world")
    assert result == "hello world"


def test_missing_variable_raises(tmp_path: Path) -> None:
    template_dir = tmp_path / "prompts"
    template_dir.mkdir()
    (template_dir / "greet.md").write_text("hello {{name}}")
    renderer = PromptRenderer(template_dir)
    with pytest.raises(KeyError):
        renderer.render("greet")
