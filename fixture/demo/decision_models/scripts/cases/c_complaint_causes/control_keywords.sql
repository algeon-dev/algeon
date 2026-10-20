WITH texts AS (
  -- Keyword dimensions of 04_inline_kmeans_topk.sql. A review takes the dimension with the most
  -- matching patterns, ties go to the earlier dimension, and no match gives 'unclassified'.
  SELECT c.id, LOWER(s.text) AS body
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
), hits AS (
  SELECT id, 1 AS position, 'battery' AS dimension,
         CASE WHEN body LIKE '%battery%' THEN 1 ELSE 0 END AS hits
  FROM texts
  UNION ALL
  SELECT id, 2 AS position, 'connection' AS dimension,
         CASE WHEN body LIKE '%connect%' THEN 1 ELSE 0 END
           + CASE WHEN body LIKE '%bluetooth%' THEN 1 ELSE 0 END AS hits
  FROM texts
  UNION ALL
  SELECT id, 3 AS position, 'sound' AS dimension,
         CASE WHEN body LIKE '%sound%' THEN 1 ELSE 0 END
           + CASE WHEN body LIKE '%volume%' THEN 1 ELSE 0 END AS hits
  FROM texts
  UNION ALL
  SELECT id, 4 AS position, 'comfort' AS dimension,
         CASE WHEN body LIKE '%comfort%' THEN 1 ELSE 0 END
           + CASE WHEN body LIKE '%fit%' THEN 1 ELSE 0 END AS hits
  FROM texts
  UNION ALL
  SELECT id, 5 AS position, 'durability' AS dimension,
         CASE WHEN body LIKE '%broke%' THEN 1 ELSE 0 END
           + CASE WHEN body LIKE '%stopped working%' THEN 1 ELSE 0 END AS hits
  FROM texts
  UNION ALL
  SELECT id, 6 AS position, 'charging' AS dimension,
         CASE WHEN body LIKE '%charg%' THEN 1 ELSE 0 END AS hits
  FROM texts
  UNION ALL
  SELECT id, 7 AS position, 'microphone' AS dimension,
         CASE WHEN body LIKE '%microphone%' THEN 1 ELSE 0 END
           + CASE WHEN body LIKE '%mic %' THEN 1 ELSE 0 END AS hits
  FROM texts
  UNION ALL
  SELECT id, 8 AS position, 'returns' AS dimension,
         CASE WHEN body LIKE '%return%' THEN 1 ELSE 0 END
           + CASE WHEN body LIKE '%refund%' THEN 1 ELSE 0 END AS hits
  FROM texts
), ranked AS (
  SELECT id, dimension, hits, ROW_NUMBER() OVER (PARTITION BY id ORDER BY hits DESC, position) AS place
  FROM hits
), keyword AS (
  SELECT id, CASE WHEN hits = 0 THEN 'unclassified' ELSE dimension END AS keyword_cause
  FROM ranked
  WHERE place = 1
), model AS (
  SELECT c.id, COALESCE(e.complaint_cause, l.label) AS model_cause
  FROM review_clusters c
  JOIN cluster_labels l ON l.cluster_id = c.cluster_id
  LEFT JOIN review_causes e ON e.id = c.id AND e.label_source = 'escalated'
)
SELECT k.keyword_cause, m.model_cause, COUNT(*) AS reviews
FROM keyword k
JOIN model m ON m.id = k.id
GROUP BY k.keyword_cause, m.model_cause
ORDER BY k.keyword_cause, m.model_cause
