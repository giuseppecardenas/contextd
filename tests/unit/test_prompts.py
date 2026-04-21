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


def test_template_name_cannot_escape_template_dir(tmp_path: Path) -> None:
    """template='../../etc/passwd' must fail before read_text() touches it."""
    template_dir = tmp_path / "prompts"
    template_dir.mkdir()
    (template_dir / "ok.md").write_text("hello")
    # Plant a file outside template_dir to prove we would have read it if the
    # guard were absent.
    (tmp_path / "secret.md").write_text("SECRET")
    renderer = PromptRenderer(template_dir)
    with pytest.raises(ValueError, match="escapes template_dir"):
        renderer.render("../secret")
