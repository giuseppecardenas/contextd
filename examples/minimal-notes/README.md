# Minimal notes example

A 10-file personal-notes corpus for smoke-testing Contextd against a small, heterogeneous tree.

Add with:

```bash
contextd add-corpus examples/minimal-notes --name notes \
    --from examples/minimal-notes/.contextd/corpus.toml
contextd index notes --bootstrap
```

Then score retrieval against the labelled queries in `.contextd/bench.toml`:

```bash
contextd bench notes --profiles fine --profiles coarse --profiles fine,coarse
```
