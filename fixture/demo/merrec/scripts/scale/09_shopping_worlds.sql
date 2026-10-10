WITH fans AS (
  SELECT -(e.user_id + 1) AS src, CAST(i.brand_id AS BIGINT) AS dst
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id <> 0
  GROUP BY e.user_id, i.brand_id
), top_brands AS (
  SELECT dst AS brand_id, COUNT(*) AS fans FROM fans GROUP BY dst ORDER BY fans DESC, dst LIMIT 5000
), worlds AS (
  SELECT vertex AS brand_id, community AS world
  FROM cugraph_leiden(
    edges => (
      SELECT first AS src, second AS dst, CAST(similarity AS DOUBLE) AS weight
      FROM cugraph_jaccard(
        edges => (SELECT src, dst FROM fans),
        vertex_pairs => (SELECT a.brand_id AS first, b.brand_id AS second
                         FROM top_brands a JOIN top_brands b ON a.brand_id < b.brand_id),
        directed => false)
      WHERE similarity >= 0.02),
    directed => false, resolution => 3, seed => 42)
), members AS (
  SELECT w.world, n.brand_name, t.fans
  FROM worlds w
  JOIN top_brands t ON t.brand_id = w.brand_id
  JOIN (SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
        FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id) n ON n.brand_id = w.brand_id
), ranked AS (
  SELECT world, brand_name, fans,
         ROW_NUMBER() OVER (PARTITION BY world ORDER BY fans DESC, brand_name) AS rk,
         COUNT(*) OVER (PARTITION BY world) AS brands,
         SUM(fans) OVER (PARTITION BY world) AS fan_links
  FROM members
)
SELECT world, brands, fan_links, rk, brand_name, fans
FROM ranked
WHERE rk <= 5 AND brands >= 15
ORDER BY fan_links DESC, rk;
