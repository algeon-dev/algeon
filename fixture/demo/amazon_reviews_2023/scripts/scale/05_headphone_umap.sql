SELECT u.id, u.dim_0, u.dim_1, r.product_id,
       p.title AS product_title, EXTRACT(YEAR FROM r.review_time) AS review_year,
       r.rating, r.title, r.embedding_text
FROM cuml_umap(
  input => (SELECT v.id, v.vector
    FROM semantic_reviews_vectors v
    JOIN semantic_reviews_text r ON r.id = v.id
    WHERE r.category_l3 = 'Headphones & Earbuds' AND v.sample_rank < 100000
    ORDER BY v.id),
  n_components => 2, n_neighbors => 30, min_dist => 0.1, random_state => 42) u
JOIN semantic_reviews_text r ON r.id = u.id
JOIN products p ON p.product_id = r.product_id
ORDER BY u.id;
