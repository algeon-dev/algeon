WITH assignments AS (
  SELECT id, cluster_id
  FROM cuvs_kmeans(
    input => (SELECT v.id, v.vector
      FROM semantic_reviews_vectors v
      JOIN semantic_reviews_text r ON r.id = v.id
      WHERE r.category_l3 = 'Headphones & Earbuds'
      ORDER BY v.id),
    n_clusters => 24, n_init => 5, max_iter => 100, tol => 0.0001)
), issues AS (
  SELECT r.product_id, a.cluster_id,
         SUM(CASE WHEN EXTRACT(YEAR FROM r.review_time) = 2021 THEN 1 ELSE 0 END) AS n2021,
         SUM(CASE WHEN EXTRACT(YEAR FROM r.review_time) = 2022 THEN 1 ELSE 0 END) AS n2022,
         MIN(CASE WHEN EXTRACT(YEAR FROM r.review_time) = 2022 THEN r.id END) AS example_id
  FROM assignments a JOIN semantic_reviews_text r ON r.id = a.id
  GROUP BY r.product_id, a.cluster_id
), totals AS (
  SELECT *, SUM(n2021) OVER (PARTITION BY product_id) AS total2021,
            SUM(n2022) OVER (PARTITION BY product_id) AS total2022
  FROM issues
)
SELECT p.product_id, p.title, t.cluster_id,
       t.n2021, t.total2021, t.n2022, t.total2022,
       100.0 * t.n2021 / t.total2021 AS share2021_pct,
       100.0 * t.n2022 / t.total2022 AS share2022_pct,
       100.0 * t.n2022 / t.total2022 - 100.0 * t.n2021 / t.total2021 AS change_pp,
       t.example_id, r.embedding_text AS example_review
FROM totals t
JOIN products p ON p.product_id = t.product_id
JOIN semantic_reviews_text r ON r.id = t.example_id
WHERE t.total2021 >= 30 AND t.total2022 >= 30 AND t.n2022 >= 10
  AND 1.0 * t.n2022 / t.total2022 - 1.0 * t.n2021 / t.total2021 > 0.10
ORDER BY change_pp DESC, p.product_id, t.cluster_id
LIMIT 10;
