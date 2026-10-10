WITH fans AS (
  SELECT -(e.user_id + 1) AS src, CAST(i.brand_id AS BIGINT) AS dst
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id <> 0
  GROUP BY e.user_id, i.brand_id
), top_brands AS (
  SELECT dst AS brand_id, COUNT(*) AS fans FROM fans GROUP BY dst ORDER BY fans DESC, dst LIMIT 5000
), similar AS (
  SELECT first, second, similarity
  FROM cugraph_jaccard(
    edges => (SELECT src, dst FROM fans),
    vertex_pairs => (SELECT a.brand_id AS first, b.brand_id AS second
                     FROM top_brands a JOIN top_brands b ON a.brand_id < b.brand_id),
    directed => false)
  WHERE similarity >= 0.02
), worlds AS (
  SELECT vertex AS brand_id, community AS world
  FROM cugraph_leiden(
    edges => (SELECT first AS src, second AS dst, CAST(similarity AS DOUBLE) AS weight FROM similar),
    directed => false, resolution => 3, seed => 42)
), bridges AS (
  SELECT vertex AS brand_id, value AS betweenness
  FROM cugraph_betweenness_centrality(
    edges => (SELECT first AS src, second AS dst FROM similar),
    directed => false, normalized => true)
), neighbors AS (
  SELECT first AS brand_id, second AS neighbor FROM similar
  UNION ALL SELECT second, first FROM similar
), reach AS (
  SELECT n.brand_id, COUNT(*) AS neighbors, COUNT(DISTINCT w.world) AS worlds_reached
  FROM neighbors n JOIN worlds w ON w.brand_id = n.neighbor
  GROUP BY n.brand_id
), brands AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
  FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id
)
SELECT b.brand_name, t.fans, w.world, r.neighbors, r.worlds_reached, br.betweenness
FROM bridges br
JOIN worlds w ON w.brand_id = br.brand_id
JOIN reach r ON r.brand_id = br.brand_id
JOIN top_brands t ON t.brand_id = br.brand_id
JOIN brands b ON b.brand_id = br.brand_id
ORDER BY br.betweenness DESC, b.brand_name
LIMIT 15;
