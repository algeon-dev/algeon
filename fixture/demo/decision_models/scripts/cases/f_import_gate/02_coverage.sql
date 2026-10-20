-- Retained nodes per Win32 sample in one variant, with its label and the labeled size of its family,
-- plus the variant's distinct node count on every row.
WITH {variant_edges}, per_sample AS (
  SELECT src AS id, COUNT(*) AS retained_nodes
  FROM variant_edges
  GROUP BY src
), family_sizes AS (
  SELECT family, COUNT(*) AS family_members
  FROM ember_f_samples
  WHERE labeled AND file_type = 'Win32'
  GROUP BY family
), node_count AS (
  SELECT COUNT(DISTINCT dst) AS variant_nodes FROM variant_edges
)
SELECT p.id, p.retained_nodes, s.labeled, s.family, COALESCE(f.family_members, 0) AS family_members,
       c.variant_nodes
FROM per_sample p
JOIN ember_f_samples s ON s.id = p.id
LEFT JOIN family_sizes f ON f.family = s.family
CROSS JOIN node_count c
ORDER BY p.id
