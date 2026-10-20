-- Edge list of one graph variant: Win32 challenge sample to retained node.
-- Imports keep doc frequency 3..100 and behaviors 3..300 among the 3,225 Win32 challenge samples.
-- Nodes whose DLL is in the variant's exclusion table are dropped; behaviors carry no DLL and stay.
-- Placeholders: {kinds} (quoted node kinds), {exclusion_table}, {weight_column} (empty, or the IDF weight).
variant_edges AS (
  SELECT sn.id AS src, sn.node_id AS dst{weight_column}
  FROM ember_f_sample_nodes sn
  JOIN ember_f_nodes n ON n.node_id = sn.node_id
  LEFT JOIN {exclusion_table} x ON x.dll_lower = n.dll_lower
  WHERE x.dll_lower IS NULL
    AND n.kind IN ({kinds})
    AND n.doc_freq >= 3
    AND n.doc_freq <= CASE WHEN n.kind = 'import' THEN 100 ELSE 300 END
)
