WITH fans AS (
  SELECT -(e.user_id + 1) AS src, CAST(i.brand_id AS BIGINT) AS dst
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id <> 0
  GROUP BY e.user_id, i.brand_id
), top_brands AS (
  SELECT dst AS brand_id, COUNT(*) AS fans FROM fans GROUP BY dst ORDER BY fans DESC, dst LIMIT 2000
), brands AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
  FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id
)
SELECT ba.brand_name AS brand_a, bb.brand_name AS brand_b,
       a.fans AS fans_a, b.fans AS fans_b, r.similarity AS jaccard
FROM cugraph_jaccard(
  edges => (SELECT src, dst FROM fans),
  vertex_pairs => (SELECT a.brand_id AS first, b.brand_id AS second
                   FROM top_brands a JOIN top_brands b ON a.brand_id < b.brand_id),
  directed => false
) r
JOIN top_brands a ON a.brand_id = r.first
JOIN top_brands b ON b.brand_id = r.second
JOIN brands ba ON ba.brand_id = r.first
JOIN brands bb ON bb.brand_id = r.second
ORDER BY r.similarity DESC, brand_a
LIMIT 30;
