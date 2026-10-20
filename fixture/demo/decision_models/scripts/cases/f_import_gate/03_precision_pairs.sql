-- The 300 most similar pairs among labeled Win32 samples with at least 5 retained nodes in the variant.
-- Pairs tied on similarity are ordered by the number of retained nodes the two samples share, counted
-- by joining the variant edges to themselves on dst as 07_challenge_import_review.sql counts shared imports.
WITH {variant_edges}, labeled_active AS (
  SELECT e.src AS id, s.family
  FROM variant_edges e
  JOIN ember_f_samples s ON s.id = e.src
  WHERE s.labeled
  GROUP BY e.src, s.family
  HAVING COUNT(*) >= 5
)
SELECT j.first, j.second, j.similarity, COUNT(*) AS shared_nodes,
       a.family AS family_first, b.family AS family_second
FROM cugraph_jaccard(
  edges => (SELECT {edge_columns} FROM variant_edges),
  vertex_pairs => (
    SELECT a.id AS first, b.id AS second
    FROM labeled_active a JOIN labeled_active b ON a.id < b.id
  ),
  directed => false
) j
JOIN variant_edges first_edges ON first_edges.src = j.first
JOIN variant_edges second_edges ON second_edges.src = j.second AND second_edges.dst = first_edges.dst
JOIN labeled_active a ON a.id = j.first
JOIN labeled_active b ON b.id = j.second
GROUP BY j.first, j.second, j.similarity, a.family, b.family
ORDER BY j.similarity DESC, shared_nodes DESC, j.first, j.second
LIMIT 300
