WITH qualifying AS (
  -- Star ratings only: share of rating <= 2 among all reviews of each qualifying product.
  SELECT s.product_id
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
  GROUP BY s.product_id
  HAVING SUM(CASE WHEN EXTRACT(YEAR FROM s.review_time) = 2021 THEN 1 ELSE 0 END) >= 30
     AND SUM(CASE WHEN EXTRACT(YEAR FROM s.review_time) = 2022 THEN 1 ELSE 0 END) >= 30
), yearly AS (
  SELECT r.product_id,
         SUM(CASE WHEN r.review_time < TIMESTAMP '2022-01-01 00:00:00' AND r.rating <= 2 THEN 1 ELSE 0 END) AS low2021,
         SUM(CASE WHEN r.review_time < TIMESTAMP '2022-01-01 00:00:00' THEN 1 ELSE 0 END) AS reviews2021,
         SUM(CASE WHEN r.review_time >= TIMESTAMP '2022-01-01 00:00:00' AND r.rating <= 2 THEN 1 ELSE 0 END) AS low2022,
         SUM(CASE WHEN r.review_time >= TIMESTAMP '2022-01-01 00:00:00' THEN 1 ELSE 0 END) AS reviews2022
  FROM reviews r
  JOIN qualifying q ON q.product_id = r.product_id
  WHERE r.review_time >= TIMESTAMP '2021-01-01 00:00:00'
    AND r.review_time < TIMESTAMP '2023-01-01 00:00:00'
  GROUP BY r.product_id
)
SELECT y.product_id, p.title, y.low2021, y.reviews2021, y.low2022, y.reviews2022,
       100.0 * y.low2021 / y.reviews2021 AS share2021_pct,
       100.0 * y.low2022 / y.reviews2022 AS share2022_pct,
       100.0 * y.low2022 / y.reviews2022 - 100.0 * y.low2021 / y.reviews2021 AS change_pp
FROM yearly y
JOIN products p ON p.product_id = y.product_id
ORDER BY change_pp DESC, y.product_id
