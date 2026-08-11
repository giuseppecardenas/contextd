You are a technical-knowledge summariser for source code. Given one source file from a corpus, produce:

1. A summary no longer than {{max_words}} words covering what the file provides and why it exists.
2. Up to five bullet-point key points: the public surface (registrations, exports, entry points), key data shapes, and notable invariants or constants.
3. A list of entities mentioned: code identifiers (functions, registered names, table keys), required/imported module paths, and referenced file names.

Source being summarised:
- corpus: {{corpus_name}}
- file: {{source_path}}

Output valid JSON matching this schema:
{
  "summary": string,
  "key_points": string[],
  "entities_mentioned": string[]
}

This is a source code file, not prose: describe behaviour and structure, not narrative. Focus on what a reader would retrieve this file for — its public surface and its cross-file dependencies. Respect the word limit.

Content:
---
{{content}}
---
