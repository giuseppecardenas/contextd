You are naming and summarising one topic of a document corpus. The topic groups the units below because their content is semantically similar; produce a short title (at most 8 words) and ONE summary no longer than {{max_words}} words capturing what the group as a whole is about and how its members relate — do not enumerate the members one by one.

Corpus: {{corpus_name}}
Members: {{member_count}}

Member summaries, most representative first:
{{member_summaries}}

Output valid JSON matching this schema:
{
  "title": string,
  "summary": string
}

Do not include anything outside the JSON object.
