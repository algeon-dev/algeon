WITH likes AS (
  SELECT e.user_id, i.brand_id,
         CASE WHEN e.event_time < TIMESTAMP '2023-07-01 00:00:00' THEN 0 ELSE 1 END AS after_release
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id <> 0
    AND e.event_time >= TIMESTAMP '2023-05-01 00:00:00'
    AND e.event_time < TIMESTAMP '2023-09-01 00:00:00'
  GROUP BY e.user_id, i.brand_id,
           CASE WHEN e.event_time < TIMESTAMP '2023-07-01 00:00:00' THEN 0 ELSE 1 END
), before_graph AS (
  SELECT -(user_id + 1) AS src, CAST(brand_id AS BIGINT) AS dst FROM likes WHERE after_release = 0
), after_graph AS (
  SELECT -(user_id + 1) AS src, CAST(brand_id AS BIGINT) AS dst FROM likes WHERE after_release = 1
), fans AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id,
         SUM(1 - after_release) AS fans_before, SUM(after_release) AS fans_after
  FROM likes GROUP BY brand_id
), ppr_before AS (
  SELECT vertex, value FROM cugraph_personalized_pagerank(
    edges => (SELECT src, dst FROM before_graph),
    personalization => (SELECT CAST(2595 AS BIGINT) AS vertex, CAST(1.0 AS DOUBLE) AS value),
    directed => false)
), ppr_after AS (
  SELECT vertex, value FROM cugraph_personalized_pagerank(
    edges => (SELECT src, dst FROM after_graph),
    personalization => (SELECT CAST(2595 AS BIGINT) AS vertex, CAST(1.0 AS DOUBLE) AS value),
    directed => false)
), pr_before AS (
  SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM before_graph), directed => false)
), pr_after AS (
  SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM after_graph), directed => false)
), brands AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
  FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id
)
SELECT b.brand_name, f.fans_before, f.fans_after,
       pb.value / rb.value AS lift_before,
       pa.value / ra.value AS lift_after,
       (pa.value / ra.value) / (pb.value / rb.value) AS lift_change
FROM ppr_after pa
JOIN pr_after ra ON ra.vertex = pa.vertex
JOIN ppr_before pb ON pb.vertex = pa.vertex
JOIN pr_before rb ON rb.vertex = pa.vertex
JOIN fans f ON f.brand_id = pa.vertex
JOIN brands b ON b.brand_id = pa.vertex
WHERE pa.vertex > 0 AND pa.vertex <> 2595
  AND f.fans_before >= 1000 AND f.fans_after >= 1000
ORDER BY lift_change DESC, b.brand_name
LIMIT 25;
