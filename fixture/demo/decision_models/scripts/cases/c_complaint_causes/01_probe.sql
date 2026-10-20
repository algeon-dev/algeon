WITH assignments AS (
  -- The cluster id rides in the dataset id (id * 32 + cluster_id) so the kmeans CTE is read once;
  -- a second reference would be inlined and refit, and refits can assign rows differently.
  SELECT id, cluster_id
  FROM cuvs_kmeans(
    input => (SELECT v.id, v.vector
      FROM semantic_reviews_vectors v
      JOIN semantic_reviews_text r ON r.id = v.id
      WHERE r.category_l3 = 'Headphones & Earbuds'
      ORDER BY v.id),
    n_clusters => 24, n_init => 5, max_iter => 100, tol => 0.0001)
), neighbors AS (
  SELECT query_id, neighbor_id / 32 AS neighbor_review, neighbor_id % 32 AS neighbor_cluster, distance
  FROM cuvs_brute_force_knn(
    dataset => (SELECT a.id * 32 + a.cluster_id AS id, v.vector
      FROM assignments a
      JOIN semantic_reviews_vectors v ON v.id = a.id),
    queries => (SELECT v.id, v.vector
      FROM semantic_reviews_vectors v
      JOIN semantic_reviews_text r ON r.id = v.id
      WHERE r.category_l3 = 'Headphones & Earbuds'
      ORDER BY v.id),
    k => 64, metric => 'l2_expanded')
), labeled AS (
  SELECT query_id, neighbor_review, neighbor_cluster, distance,
         MAX(CASE WHEN neighbor_review = query_id THEN neighbor_cluster END)
           OVER (PARTITION BY query_id) AS own_cluster
  FROM neighbors
), density AS (
  SELECT query_id AS id, CAST(own_cluster AS INT) AS cluster_id,
         SUM(CASE WHEN neighbor_cluster = own_cluster AND neighbor_review <> query_id THEN 1 ELSE 0 END)
           AS same_cluster_neighbors,
         AVG(CASE WHEN neighbor_cluster = own_cluster AND neighbor_review <> query_id THEN distance END)
           AS mean_distance
  FROM labeled
  GROUP BY query_id, own_cluster
), ranked AS (
  -- Centroids are not returned by cuvs_kmeans, so the densest same-cluster region stands in for the
  -- center; a review needs half of its 64 neighbors in its own cluster to count as interior.
  SELECT id, cluster_id, same_cluster_neighbors, mean_distance,
         CASE WHEN same_cluster_neighbors >= 32 THEN 0 ELSE 1 END AS border
  FROM density
)
SELECT id, cluster_id, same_cluster_neighbors, mean_distance,
       ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY border, mean_distance, id) AS density_rank
FROM ranked
ORDER BY cluster_id, density_rank
