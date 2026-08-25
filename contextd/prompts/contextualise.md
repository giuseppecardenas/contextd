You are preparing retrieval chunks for a search index. For each numbered chunk below, write a short, succinct context (one or two sentences, at most 60 words) that situates the chunk within the overall document so it can be found by search: what the surrounding document is about, what this chunk specifically covers, and any names, identifiers or dates the chunk refers to implicitly.

Document location: {{breadcrumb}}

Document summary:
---
{{document_summary}}
---

There are {{count}} chunks. Output valid JSON with exactly {{count}} entries, in the same order, matching this schema:
{
  "contexts": string[]
}

Do not repeat the chunk text; write only the situating context. Do not include anything outside the JSON object.

Chunks:
---
{{chunks}}
---
