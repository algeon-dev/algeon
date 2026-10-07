WITH exact AS (
  SELECT query_id, neighbor_id, distance FROM cuvs_brute_force_knn(
    dataset => (SELECT id, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, d49, d50, d51, d52, d53, d54, d55, d56, d57, d58, d59, d60, d61, d62, d63 FROM mr_taste64 ORDER BY id),
    queries => (SELECT id, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, d49, d50, d51, d52, d53, d54, d55, d56, d57, d58, d59, d60, d61, d62, d63 FROM mr_taste64 ORDER BY id),
    k => 10, metric => 'l2_expanded')
), approx AS (
  SELECT query_id, neighbor_id, distance FROM cuvs_ivf_pq(
    dataset => (SELECT id, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, d49, d50, d51, d52, d53, d54, d55, d56, d57, d58, d59, d60, d61, d62, d63 FROM mr_taste64 ORDER BY id),
    queries => (SELECT id, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, d49, d50, d51, d52, d53, d54, d55, d56, d57, d58, d59, d60, d61, d62, d63 FROM mr_taste64 ORDER BY id),
    k => 10, n_lists => 2048, n_probes => 64, pq_dim => 32, pq_bits => 8)
), kth AS (
  SELECT query_id, MAX(distance) AS kth_distance FROM exact GROUP BY query_id
)
SELECT COUNT(*) AS approx_rows,
       CAST(COUNT(e.neighbor_id) AS DOUBLE) / COUNT(*) AS id_recall,
       AVG(CASE WHEN a.distance <= k.kth_distance + 1e-5 THEN 1.0 ELSE 0.0 END) AS within_exact_kth
FROM approx a
JOIN kth k ON k.query_id = a.query_id
LEFT JOIN exact e ON e.query_id = a.query_id AND e.neighbor_id = a.neighbor_id;
