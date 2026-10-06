SELECT n.query_id, q.title AS source_title, n.rank,
       n.neighbor_id, p.title AS candidate_title,
       1.0 - n.distance / 2.0 AS cosine_similarity
FROM cuvs_brute_force_knn(
  dataset => (SELECT * FROM product_corpus_1000000),
  queries => (SELECT * FROM product_queries_10000),
  k => 10, metric => 'l2_expanded') n
JOIN semantic_products_text q ON q.id = n.query_id
JOIN semantic_products_text p ON p.id = n.neighbor_id
ORDER BY n.query_id, n.rank;
