WITH labeled AS (
  SELECT s.id, EXTRACT(YEAR FROM s.review_time) AS review_year,
         COALESCE(e.complaint_cause, l.label) AS cause,
         CASE WHEN e.id IS NULL THEN 'cluster' ELSE 'escalated' END AS label_source
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
  JOIN cluster_labels l ON l.cluster_id = c.cluster_id
  LEFT JOIN review_causes e ON e.id = c.id AND e.label_source = 'escalated'
), per_cause AS (
  SELECT cause,
         SUM(CASE WHEN review_year = 2021 THEN 1 ELSE 0 END) AS n2021,
         SUM(CASE WHEN review_year = 2022 THEN 1 ELSE 0 END) AS n2022,
         SUM(CASE WHEN label_source = 'escalated' THEN 1 ELSE 0 END) AS escalated_reviews
  FROM labeled
  GROUP BY cause
)
SELECT cause, n2021, 100.0 * n2021 / SUM(n2021) OVER () AS share2021_pct,
       n2022, 100.0 * n2022 / SUM(n2022) OVER () AS share2022_pct,
       100.0 * n2022 / SUM(n2022) OVER () - 100.0 * n2021 / SUM(n2021) OVER () AS change_pp,
       escalated_reviews
FROM per_cause
ORDER BY cause
