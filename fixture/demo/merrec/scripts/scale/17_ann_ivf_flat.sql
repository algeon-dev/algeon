SELECT COUNT(*) AS neighbor_rows, COUNT(DISTINCT query_id) AS shoppers, AVG(distance) AS mean_distance
FROM cuvs_ivf_flat(
  dataset => (SELECT * FROM mr_taste64 ORDER BY id),
  queries => (SELECT * FROM mr_taste64 ORDER BY id),
  k => 10, n_lists => 2048, n_probes => 32
);
