WITH labeled AS (
  -- Per-review labels escalated from mixed clusters take precedence over the cluster label.
  SELECT s.id, s.product_id, EXTRACT(YEAR FROM s.review_time) AS review_year,
         COALESCE(e.defect_actionable, l.defect_actionable) AS defect_actionable
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
  JOIN cluster_labels l ON l.cluster_id = c.cluster_id
  LEFT JOIN review_causes e ON e.id = c.id AND e.label_source = 'escalated'
), per_product AS (
  SELECT product_id,
         SUM(CASE WHEN review_year = 2021 AND defect_actionable THEN 1 ELSE 0 END) AS n2021,
         SUM(CASE WHEN review_year = 2021 THEN 1 ELSE 0 END) AS total2021,
         SUM(CASE WHEN review_year = 2022 AND defect_actionable THEN 1 ELSE 0 END) AS n2022,
         SUM(CASE WHEN review_year = 2022 THEN 1 ELSE 0 END) AS total2022
  FROM labeled
  GROUP BY product_id
)
SELECT t.product_id, p.title, t.n2021, t.total2021, t.n2022, t.total2022,
       100.0 * t.n2021 / t.total2021 AS share2021_pct,
       100.0 * t.n2022 / t.total2022 AS share2022_pct,
       100.0 * t.n2022 / t.total2022 - 100.0 * t.n2021 / t.total2021 AS change_pp
FROM per_product t
JOIN products p ON p.product_id = t.product_id
WHERE t.total2021 >= 30 AND t.total2022 >= 30 AND t.n2022 >= 10
ORDER BY change_pp DESC, t.product_id
