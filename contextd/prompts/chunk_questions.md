You are improving the findability of retrieval chunks. For each numbered chunk below, write up to {{per_chunk}} short questions that a reader would plausibly ask and that this chunk directly answers. Use the vocabulary a searcher would use, including exact identifiers, names and numbers from the chunk.

Document location: {{breadcrumb}}

There are {{count}} chunks. Output valid JSON with exactly {{count}} entries, in the same order, matching this schema:
{
  "questions": string[][]
}

Each inner array holds the questions for the chunk at that index (an empty array if nothing sensible applies). Do not include anything outside the JSON object.

Chunks:
---
{{chunks}}
---
