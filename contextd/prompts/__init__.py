"""Packaged prompt templates for Contextd inference.

Templates ship with the wheel (see ``[tool.hatch.build.targets.wheel]
include`` in ``pyproject.toml``). The first-run ``contextd init`` copies
them into ``~/.contextd/prompts/`` where the user can override. The
runtime ``PromptRenderer`` reads from the user's directory by default.
"""
