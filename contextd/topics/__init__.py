"""Cross-document topic clustering (RAPTOR-style summary tree across files).

:mod:`contextd.topics.cluster` is pure numerics (numpy only); the indexer
phase in ``contextd/indexer/phases_topics.py`` owns the LLM summaries,
embeddings and graph writes.
"""
