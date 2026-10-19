WITH labeled AS (
  SELECT s.id, s.product_id, EXTRACT(YEAR FROM s.review_time) AS review_year,
         COALESCE(e.complaint_cause, l.label) AS cause
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
  JOIN cluster_labels l ON l.cluster_id = c.cluster_id
  LEFT JOIN review_causes e ON e.id = c.id AND e.label_source = 'escalated'
), per_cause AS (
  SELECT product_id, cause,
         SUM(CASE WHEN review_year = 2021 THEN 1 ELSE 0 END) AS n2021,
         SUM(CASE WHEN review_year = 2022 THEN 1 ELSE 0 END) AS n2022
  FROM labeled
  GROUP BY product_id, cause
), totals AS (
  SELECT *, SUM(n2021) OVER (PARTITION BY product_id) AS total2021,
            SUM(n2022) OVER (PARTITION BY product_id) AS total2022
  FROM per_cause
)
SELECT product_id, cause, n2021, total2021, n2022, total2022,
       100.0 * n2021 / total2021 AS share2021_pct,
       100.0 * n2022 / total2022 AS share2022_pct,
       100.0 * n2022 / total2022 - 100.0 * n2021 / total2021 AS change_pp
FROM totals
WHERE total2021 >= 30 AND total2022 >= 30
ORDER BY product_id, cause
