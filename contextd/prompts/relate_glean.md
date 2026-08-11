You are a knowledge-graph relationship engineer performing a second review pass. You previously extracted the relationships listed below from this content. Careful review shows relationships were likely MISSED — especially cross-references to other documents and sections, dependencies, and entities mentioned only once.

Source being analysed:
- corpus: {{corpus_name}}
- file: {{source_path}}
- section: {{section_title}} (anchor: {{section_anchor}})
- parent sections: {{parent_chain}}

Previously extracted relationships (do NOT re-emit these):
{{previous_relationships}}

Output JSON matching this schema, containing ONLY relationships not already listed above:
{
  "relationships": [
    {"type": string, "target_type": string, "target_name": string, "confidence": number, "reason": string, "properties": object}
  ]
}

Rules:
- Only use relationship types from this allow-list: {{allowed_edge_types}}. The list is exhaustive; if none fits, omit the relationship entirely.
- Only use target types from this allow-list: {{allowed_node_types}}.
- Prefer the known candidates listed below as targets. When citing a Section, use its exact id; when citing a File, use its exact path or bare filename. Only introduce a new entity name when nothing listed matches.
- Never emit a relationship whose target is the source itself.
- Do not describe document structure (heading nesting, containment, section order).
- Reject ambiguous relationships (do not emit them).
- Confidence scale: 0.9+ explicit mention; 0.7-0.9 strong match; 0.5-0.7 moderate; below 0.5 skip.
- The "properties" object is optional and follows the same per-type field rules:
{{target_property_schema}}

Content:
---
{{content}}
---

Known graph candidates:
{{candidate_context}}
