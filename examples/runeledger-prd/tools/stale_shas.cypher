MATCH (g:Risk)
WHERE g.description CONTAINS "SHA=pending"
RETURN g.description AS entry, g.severity AS severity
ORDER BY g.severity;
