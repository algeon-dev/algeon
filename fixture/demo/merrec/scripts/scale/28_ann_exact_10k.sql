SELECT COUNT(*) AS neighbor_rows, COUNT(DISTINCT query_id) AS shoppers, AVG(distance) AS mean_distance
FROM cuvs_brute_force_knn(
  dataset => (SELECT * FROM mr_taste64 ORDER BY id),
  queries => (SELECT * FROM mr_taste64 ORDER BY id LIMIT 10000),
  k => 10, metric => 'l2_expanded'
);
