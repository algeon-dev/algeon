-- Similarity of each fixed seed to every labeled Win32 sample with at least 5 retained nodes in the variant.
-- A seed with no retained node is absent from the graph and returns no rows.
WITH {variant_edges}, labeled_active AS (
  SELECT e.src AS id, s.family
  FROM variant_edges e
  JOIN ember_f_samples s ON s.id = e.src
  WHERE s.labeled
  GROUP BY e.src, s.family
  HAVING COUNT(*) >= 5
), seeds AS (
  SELECT src AS id FROM variant_edges WHERE src IN ({seed_ids}) GROUP BY src
)
SELECT j.first AS seed_id, j.second AS neighbor_id, j.similarity, a.family AS neighbor_family
FROM cugraph_jaccard(
  edges => (SELECT {edge_columns} FROM variant_edges),
  vertex_pairs => (
    SELECT s.id AS first, a.id AS second
    FROM seeds s CROSS JOIN labeled_active a
    WHERE a.id <> s.id
  ),
  directed => false
) j
JOIN labeled_active a ON a.id = j.second
ORDER BY j.first, j.similarity DESC, j.second
