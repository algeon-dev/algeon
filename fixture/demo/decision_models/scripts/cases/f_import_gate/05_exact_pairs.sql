-- Labeled pairs with at least 5 retained nodes each whose Jaccard similarity is 1.0, and how many of them
-- share a family. IDF-weighted sums of identical node sets can land one ulp below 1.0, so the bound
-- allows 1e-9; with `= 1.0` V2 keeps 738 of the 793 identical-set pairs that V1 counts.
WITH {variant_edges}, labeled_active AS (
  SELECT e.src AS id, s.family
  FROM variant_edges e
  JOIN ember_f_samples s ON s.id = e.src
  WHERE s.labeled
  GROUP BY e.src, s.family
  HAVING COUNT(*) >= 5
)
SELECT COUNT(*) AS pairs, COALESCE(SUM(CASE WHEN a.family = b.family THEN 1 ELSE 0 END), 0) AS same_family
FROM cugraph_jaccard(
  edges => (SELECT {edge_columns} FROM variant_edges),
  vertex_pairs => (
    SELECT a.id AS first, b.id AS second
    FROM labeled_active a JOIN labeled_active b ON a.id < b.id
  ),
  directed => false
) j
JOIN labeled_active a ON a.id = j.first
JOIN labeled_active b ON b.id = j.second
WHERE j.similarity >= 0.999999999
