WITH members AS (
  SELECT c.cluster_id, s.product_id
  FROM review_clusters c
  JOIN semantic_reviews_text s ON s.id = c.id
  JOIN cluster_labels l ON l.cluster_id = c.cluster_id
  WHERE l.mean_safety_hazard >= 0.50
), per_product AS (
  SELECT cluster_id, product_id, COUNT(*) AS reviews
  FROM members
  GROUP BY cluster_id, product_id
), ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY reviews DESC, product_id) AS product_rank
  FROM per_product
)
SELECT l.cluster_id, l.label, l.consistency, l.size, l.mean_safety_hazard, l.mean_severity,
       r.product_rank, r.product_id, p.title, r.reviews
FROM ranked r
JOIN cluster_labels l ON l.cluster_id = r.cluster_id
JOIN products p ON p.product_id = r.product_id
WHERE r.product_rank <= 3
ORDER BY l.mean_safety_hazard DESC, l.cluster_id, r.product_rank
