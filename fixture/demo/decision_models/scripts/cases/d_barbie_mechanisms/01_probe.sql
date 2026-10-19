-- Copy of fixture/demo/merrec/scripts/scale/10_barbie_tie_ins.sql; run.py checks that the statements match.
WITH likes AS (
  SELECT DISTINCT -(e.user_id + 1) AS src, CAST(i.brand_id AS BIGINT) AS dst,
         e.event_time >= TIMESTAMP '2023-07-01 00:00:00' AS after_release
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id <> 0
    AND e.event_time >= TIMESTAMP '2023-05-01 00:00:00'
    AND e.event_time < TIMESTAMP '2023-09-01 00:00:00'
), lift_before AS (
  SELECT p.vertex, p.value / r.value AS lift
  FROM cugraph_personalized_pagerank(
    edges => (SELECT src, dst FROM likes WHERE NOT after_release),
    personalization => (SELECT 2595 AS vertex, 1.0 AS value),
    directed => false) p
  JOIN cugraph_pagerank(
    edges => (SELECT src, dst FROM likes WHERE NOT after_release), directed => false) r
    ON r.vertex = p.vertex
), lift_after AS (
  SELECT p.vertex, p.value / r.value AS lift
  FROM cugraph_personalized_pagerank(
    edges => (SELECT src, dst FROM likes WHERE after_release),
    personalization => (SELECT 2595 AS vertex, 1.0 AS value),
    directed => false) p
  JOIN cugraph_pagerank(
    edges => (SELECT src, dst FROM likes WHERE after_release), directed => false) r
    ON r.vertex = p.vertex
), fans AS (
  SELECT dst AS brand_id,
         SUM(CASE WHEN after_release THEN 0 ELSE 1 END) AS fans_before,
         SUM(CASE WHEN after_release THEN 1 ELSE 0 END) AS fans_after
  FROM likes GROUP BY dst
), brands AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
  FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id
)
SELECT b.brand_name, f.fans_before, f.fans_after,
       lb.lift AS lift_before, la.lift AS lift_after, la.lift / lb.lift AS lift_change
FROM lift_after la
JOIN lift_before lb ON lb.vertex = la.vertex
JOIN fans f ON f.brand_id = la.vertex
JOIN brands b ON b.brand_id = la.vertex
WHERE la.vertex > 0 AND la.vertex <> 2595
  AND f.fans_before >= 1000 AND f.fans_after >= 1000
ORDER BY lift_change DESC, b.brand_name
LIMIT 25;
