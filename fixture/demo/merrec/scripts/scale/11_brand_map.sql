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
), nearest AS (
  SELECT a, b, similarity,
         ROW_NUMBER() OVER (PARTITION BY a ORDER BY similarity DESC, b) AS rk
  FROM (SELECT first AS a, second AS b, similarity FROM similar
        UNION ALL SELECT second AS a, first AS b, similarity FROM similar) d
), worlds AS (
  SELECT vertex AS brand_id, community AS world
  FROM cugraph_leiden(
    edges => (SELECT first AS src, second AS dst, CAST(similarity AS DOUBLE) AS weight FROM similar),
    directed => false, resolution => 3, seed => 42)
), layout AS (
  SELECT vertex AS brand_id, x, y
  FROM cugraph_force_atlas2(
    edges => (SELECT a AS src, b AS dst, CAST(similarity AS DOUBLE) AS weight FROM nearest WHERE rk <= 5),
    directed => false, max_iter => 2000, barnes_hut_optimize => true,
    outbound_attraction_distribution => true, lin_log_mode => true,
    scaling_ratio => 5, gravity => 5, seed => 42)
), brands AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
  FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id
)
SELECT l.brand_id, b.brand_name, t.fans, w.world, l.x, l.y
FROM layout l
JOIN worlds w ON w.brand_id = l.brand_id
JOIN top_brands t ON t.brand_id = l.brand_id
JOIN brands b ON b.brand_id = l.brand_id
ORDER BY l.brand_id;
