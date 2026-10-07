WITH exact AS (
  SELECT query_id, neighbor_id, distance FROM cuvs_brute_force_knn(
    dataset => (SELECT * FROM mr_taste64 ORDER BY id),
    queries => (SELECT * FROM mr_taste64 ORDER BY id),
    k => 10, metric => 'l2_expanded')
), approx AS (
  SELECT query_id, neighbor_id, distance FROM cuvs_cagra(
    dataset => (SELECT * FROM mr_taste64 ORDER BY id),
    queries => (SELECT * FROM mr_taste64 ORDER BY id),
    k => 10, graph_degree => 32, intermediate_graph_degree => 64, build_algo => 'nn_descent')
), kth AS (
  SELECT query_id, MAX(distance) AS kth_distance FROM exact GROUP BY query_id
)
SELECT COUNT(*) AS approx_rows,
       CAST(COUNT(e.neighbor_id) AS DOUBLE) / COUNT(*) AS id_recall,
       AVG(CASE WHEN a.distance <= k.kth_distance + 1e-5 THEN 1.0 ELSE 0.0 END) AS within_exact_kth
FROM approx a
JOIN kth k ON k.query_id = a.query_id
LEFT JOIN exact e ON e.query_id = a.query_id AND e.neighbor_id = a.neighbor_id;
