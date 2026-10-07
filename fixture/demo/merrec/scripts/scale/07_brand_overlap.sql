WITH fans AS (
  SELECT -(e.user_id + 1) AS src, CAST(i.brand_id AS BIGINT) AS dst
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id <> 0
  GROUP BY e.user_id, i.brand_id
), brand_fans AS (
  SELECT dst AS brand_id, COUNT(*) AS fans FROM fans GROUP BY dst
), brands AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
  FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id
)
SELECT b.brand_name, f.fans, ROUND(r.similarity * f.fans) AS shared_fans, r.similarity AS overlap
FROM cugraph_overlap(
  edges => (SELECT src, dst FROM fans),
  vertex_pairs => (SELECT 12630 AS first, brand_id AS second
                   FROM brand_fans WHERE fans >= 2000 AND brand_id <> 12630),
  directed => false
) r
JOIN brand_fans f ON f.brand_id = r.second
JOIN brands b ON b.brand_id = r.second
ORDER BY r.similarity DESC, b.brand_name
LIMIT 25;
