MATCH (r:Pattern {name: $registry_name})
OPTIONAL MATCH (r)<-[:USES]-(consumer:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(schema:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(lua:File)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(fr:Ticket)
RETURN r.name AS registry, consumer.id AS consumer_section,
       schema.id AS schema_section, lua.path AS lua_file,
       fr.id AS fr_row;
