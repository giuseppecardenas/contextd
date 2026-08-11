You are synthesising a parent-level overview from the summaries of its child sections plus any prose belonging directly to the parent.

Source being synthesised:
- corpus: {{corpus_name}}
- file: {{source_path}}
- section: {{section_title}} (anchor: {{section_anchor}})
- parent sections: {{parent_chain}}

Parent's own prose (may be empty):
---
{{own_prose}}
---

Child section summaries, in document order:
{{child_summaries}}

Produce ONE summary no longer than {{max_words}} words capturing what this subtree as a whole covers and how its parts relate — do not enumerate the children one by one.

Output valid JSON matching this schema:
{
  "summary": string
}
