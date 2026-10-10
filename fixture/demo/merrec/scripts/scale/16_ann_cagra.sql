SELECT COUNT(*) AS neighbor_rows, COUNT(DISTINCT query_id) AS shoppers, AVG(distance) AS mean_distance
FROM cuvs_cagra(
  dataset => (SELECT * FROM mr_taste64 ORDER BY id),
  queries => (SELECT * FROM mr_taste64 ORDER BY id),
  k => 10, graph_degree => 32, intermediate_graph_degree => 64, build_algo => 'nn_descent'
);
