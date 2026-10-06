WITH review_history AS (
  SELECT product_id, COUNT(*) AS reviews_2022,
         AVG(CASE WHEN rating <= 2 THEN 1.0 ELSE 0.0 END) AS low_rating_share
  FROM reviews
  WHERE review_time >= TIMESTAMP '2022-01-01'
    AND review_time < TIMESTAMP '2023-01-01' AND verified_purchase
  GROUP BY product_id
), eligible AS (
  SELECT v.id, v.vector
  FROM semantic_products_vectors v
  JOIN products p ON p.product_id = v.id
  JOIN review_history h ON h.product_id = v.id
  WHERE v.sample_rank < 1000000
    AND p.category_l3 = 'Headphones & Earbuds'
    AND p.price BETWEEN 20 AND 150
    AND h.reviews_2022 >= 100 AND h.low_rating_share <= 0.20
)
SELECT n.rank, p.product_id, p.title, p.price,
       h.reviews_2022, h.low_rating_share,
       1.0 - n.distance / 2.0 AS cosine_similarity
FROM cuvs_brute_force_knn(
  dataset => (SELECT id, vector FROM eligible),
  queries => (SELECT id, vector FROM intent_queries WHERE id = 0),
  k => 10, metric => 'l2_expanded') n
JOIN products p ON p.product_id = n.neighbor_id
JOIN review_history h ON h.product_id = p.product_id
ORDER BY n.rank;
