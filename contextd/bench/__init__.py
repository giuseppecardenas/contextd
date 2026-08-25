"""Retrieval benchmark: a labelled query set scored against ``search``.

Chunk-size and strategy decisions are config diffs plus a bench run rather
than folklore. :mod:`.metrics` is pure; :mod:`.spec` parses the YAML/TOML
query file; :mod:`.run` drives the ``search`` tool per configuration.
"""
