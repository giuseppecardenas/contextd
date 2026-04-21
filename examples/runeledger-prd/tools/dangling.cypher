MATCH (r:Pattern)
OPTIONAL MATCH (r)<-[:USES]-(consumer:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(schema:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(lua:File)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(fr:Ticket)
WITH r, consumer, schema, lua, fr
WHERE consumer IS NULL OR schema IS NULL OR lua IS NULL OR fr IS NULL
RETURN r.name AS registry,
       consumer IS NOT NULL AS has_consumer,
       schema IS NOT NULL AS has_schema,
       lua IS NOT NULL AS has_lua,
       fr IS NOT NULL AS has_fr;
