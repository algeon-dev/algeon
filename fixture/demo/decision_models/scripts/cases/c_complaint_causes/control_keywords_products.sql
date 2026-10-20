WITH qualifying AS (
  -- Dominant 2022 keyword dimension (rule of control_keywords.sql, classified reviews only) and
  -- dominant model cause for each product that passes the 2021 and 2022 volume thresholds.
  SELECT s.product_id
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
  GROUP BY s.product_id
  HAVING SUM(CASE WHEN EXTRACT(YEAR FROM s.review_time) = 2021 THEN 1 ELSE 0 END) >= 30
     AND SUM(CASE WHEN EXTRACT(YEAR FROM s.review_time) = 2022 THEN 1 ELSE 0 END) >= 30
), texts AS (
  SELECT c.id, LOWER(s.text) AS body
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
  JOIN qualifying q ON q.product_id = s.product_id
  WHERE EXTRACT(YEAR FROM s.review_time) = 2022
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
), labeled AS (
  SELECT s.product_id, k.keyword_cause, COALESCE(e.complaint_cause, l.label) AS model_cause
  FROM keyword k
  JOIN review_clusters c ON c.id = k.id
  JOIN semantic_reviews_text s ON s.id = k.id
  JOIN cluster_labels l ON l.cluster_id = c.cluster_id
  LEFT JOIN review_causes e ON e.id = k.id AND e.label_source = 'escalated'
), product_totals AS (
  SELECT product_id, COUNT(*) AS reviews2022,
         SUM(CASE WHEN keyword_cause = 'unclassified' THEN 1 ELSE 0 END) AS keyword_unclassified
  FROM labeled
  GROUP BY product_id
), keyword_top AS (
  SELECT product_id, keyword_cause, reviews,
         ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY reviews DESC, keyword_cause) AS place
  FROM (SELECT product_id, keyword_cause, COUNT(*) AS reviews
        FROM labeled WHERE keyword_cause <> 'unclassified'
        GROUP BY product_id, keyword_cause) keyword_counts
), model_top AS (
  SELECT product_id, model_cause, reviews,
         ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY reviews DESC, model_cause) AS place
  FROM (SELECT product_id, model_cause, COUNT(*) AS reviews
        FROM labeled GROUP BY product_id, model_cause) model_counts
)
SELECT t.product_id, t.reviews2022, t.keyword_unclassified,
       k.keyword_cause, k.reviews AS keyword_reviews, m.model_cause, m.reviews AS model_reviews
FROM product_totals t
JOIN model_top m ON m.product_id = t.product_id AND m.place = 1
LEFT JOIN keyword_top k ON k.product_id = t.product_id AND k.place = 1
ORDER BY t.product_id
