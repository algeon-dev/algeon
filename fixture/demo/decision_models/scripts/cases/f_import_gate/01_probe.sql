-- One row per DLL that has an import node in V1 (symbols imported by 3..100 Win32 challenge samples):
-- its edges, distinct samples, distinct symbols and its 8 most widely imported symbols.
WITH import_nodes AS (
  SELECT node_id, name, dll_lower, doc_freq
  FROM ember_f_nodes
  WHERE kind = 'import' AND doc_freq >= 3 AND doc_freq <= 100
), dll_stats AS (
  SELECT n.dll_lower, COUNT(*) AS edges, COUNT(DISTINCT sn.id) AS samples, COUNT(DISTINCT sn.node_id) AS symbols
  FROM ember_f_sample_nodes sn
  JOIN import_nodes n ON n.node_id = sn.node_id
  GROUP BY n.dll_lower
  HAVING COUNT(DISTINCT sn.id) >= 3
), ranked AS (
  SELECT dll_lower, name, doc_freq,
         ROW_NUMBER() OVER (PARTITION BY dll_lower ORDER BY doc_freq DESC, name) AS rk
  FROM import_nodes
)
SELECT d.dll_lower, d.edges, d.samples, d.symbols,
       MAX(CASE WHEN r.rk = 1 THEN r.name END) AS symbol_1,
       MAX(CASE WHEN r.rk = 2 THEN r.name END) AS symbol_2,
       MAX(CASE WHEN r.rk = 3 THEN r.name END) AS symbol_3,
       MAX(CASE WHEN r.rk = 4 THEN r.name END) AS symbol_4,
       MAX(CASE WHEN r.rk = 5 THEN r.name END) AS symbol_5,
       MAX(CASE WHEN r.rk = 6 THEN r.name END) AS symbol_6,
       MAX(CASE WHEN r.rk = 7 THEN r.name END) AS symbol_7,
       MAX(CASE WHEN r.rk = 8 THEN r.name END) AS symbol_8
FROM dll_stats d
LEFT JOIN ranked r ON r.dll_lower = d.dll_lower AND r.rk <= 8
GROUP BY d.dll_lower, d.edges, d.samples, d.symbols
ORDER BY d.samples DESC, d.dll_lower
LIMIT {max_entities}
