WITH cluster_stats AS (
  SELECT c.cluster_id, COUNT(*) AS cluster_size, COUNT(DISTINCT r.product_id) AS cluster_products
  FROM review_clusters c
  JOIN semantic_reviews_text r ON r.id = c.id
  GROUP BY c.cluster_id
)
SELECT r.id, c.cluster_id, p.title AS product_title, r.rating, r.verified_purchase,
       r.title AS review_title, SUBSTR(r.text, 1, 1500) AS review_text,
       CHARACTER_LENGTH(r.text) AS text_chars, s.cluster_size, s.cluster_products
FROM semantic_reviews_text r
JOIN review_clusters c ON c.id = r.id
JOIN cluster_stats s ON s.cluster_id = c.cluster_id
JOIN products p ON p.product_id = r.product_id
WHERE r.id IN ({review_ids})
ORDER BY r.id
