WITH pokemon AS (
  SELECT e.user_id,
         MAX(CASE WHEN e.event_type = 'buy_comp' AND e.event_time >= TIMESTAMP '2023-07-01 00:00:00'
                   AND e.event_time < TIMESTAMP '2023-09-01 00:00:00' THEN 1 ELSE 0 END) AS seed,
         MAX(CASE WHEN e.event_type = 'item_view' AND e.event_time < TIMESTAMP '2023-09-01 00:00:00' THEN 1 ELSE 0 END) AS viewer,
         MAX(CASE WHEN e.event_type = 'buy_comp' AND e.event_time >= TIMESTAMP '2023-09-01 00:00:00' THEN 1 ELSE 0 END) AS later_buyer
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE i.brand_id = 17470
  GROUP BY e.user_id
), seed AS (
  SELECT t.id FROM mr_taste64 t JOIN pokemon p ON p.user_id = t.id WHERE p.seed = 1
), candidates AS (
  SELECT t.* FROM mr_taste64 t LEFT JOIN seed s ON s.id = t.id WHERE s.id IS NULL
), likes AS (
  SELECT DISTINCT -(e.user_id + 1) AS src, CAST(i.brand_id AS BIGINT) AS dst
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id <> 0
    AND e.event_time < TIMESTAMP '2023-09-01 00:00:00'
), affinity AS (
  SELECT -(vertex + 1) AS id, value AS ppr
  FROM cugraph_personalized_pagerank(
    edges => (SELECT src, dst FROM likes),
    personalization => (SELECT 17470 AS vertex, 1.0 AS value),
    directed => false)
  WHERE vertex < 0
), graph_audience AS (
  SELECT id FROM (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY a.ppr DESC, c.id) AS rk
    FROM affinity a JOIN candidates c ON c.id = a.id) ranked
  WHERE rk <= 20000
), taste_audience AS (
  SELECT DISTINCT neighbor_id AS id
  FROM cuvs_brute_force_knn(
    dataset => (SELECT * FROM candidates ORDER BY id),
    queries => (SELECT t.* FROM mr_taste64 t JOIN seed s ON s.id = t.id ORDER BY t.id),
    k => 20, metric => 'l2_expanded')
), features AS (
  SELECT e.user_id AS id,
    CAST(SUM(CASE WHEN e.event_type = 'item_view' THEN 1 ELSE 0 END) AS REAL) AS views,
    CAST(SUM(CASE WHEN e.event_type = 'item_like' THEN 1 ELSE 0 END) AS REAL) AS likes,
    CAST(SUM(CASE WHEN e.event_type IN ('item_add_to_cart_tap', 'offer_make', 'buy_start') THEN 1 ELSE 0 END) AS REAL) AS intents,
    CAST(SUM(CASE WHEN e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS REAL) AS buys,
    CAST(SUM(CASE WHEN e.event_type = 'buy_comp' THEN i.price ELSE 0 END) AS REAL) AS gmv,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) AS REAL) AS views_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_type = 'item_like' THEN 1 ELSE 0 END) AS REAL) AS likes_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_type IN ('item_add_to_cart_tap', 'offer_make', 'buy_start') THEN 1 ELSE 0 END) AS REAL) AS intents_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS REAL) AS buys_30d,
    CAST(SUM(i.price) / COUNT(*) AS REAL) AS mean_price
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_time >= TIMESTAMP '2023-06-01 00:00:00' AND e.event_time < TIMESTAMP '2023-09-01 00:00:00'
  GROUP BY e.user_id
), scored AS (
  SELECT id, prediction AS score
  FROM cuml_forest_predict(
    input => (SELECT id, LN(1 + views) AS views, LN(1 + likes) AS likes, LN(1 + intents) AS intents,
                     LN(1 + buys) AS buys, LN(1 + gmv) AS gmv, LN(1 + views_30d) AS views_30d,
                     LN(1 + likes_30d) AS likes_30d, LN(1 + intents_30d) AS intents_30d,
                     LN(1 + buys_30d) AS buys_30d, LN(1 + mean_price) AS mean_price
              FROM features ORDER BY id),
    model => 'propensity_xgb.json', model_format => 'xgboost_json')
), shoppers AS (
  SELECT c.id,
         COALESCE(p.viewer, 0) AS viewer,
         COALESCE(p.later_buyer, 0) AS later_buyer,
         CASE WHEN g.id IS NULL THEN 0 ELSE 1 END AS in_graph,
         CASE WHEN t.id IS NULL THEN 0 ELSE 1 END AS in_taste,
         CASE WHEN g.id IS NULL AND t.id IS NULL THEN 0 ELSE 1 END AS in_pool,
         COALESCE(s.score, 0) AS score
  FROM candidates c
  LEFT JOIN pokemon p ON p.user_id = c.id
  LEFT JOIN graph_audience g ON g.id = c.id
  LEFT JOIN taste_audience t ON t.id = c.id
  LEFT JOIN scored s ON s.id = c.id
), ranked AS (
  SELECT id, viewer, later_buyer, in_graph, in_taste, in_pool,
         ROW_NUMBER() OVER (ORDER BY score DESC, id) AS rank_all,
         ROW_NUMBER() OVER (PARTITION BY viewer ORDER BY score DESC, id) AS rank_viewers,
         ROW_NUMBER() OVER (PARTITION BY in_pool ORDER BY score DESC, id) AS rank_pool
  FROM shoppers
), stages AS (
  SELECT 1 AS stage, 'all eligible shoppers' AS audience
  UNION ALL SELECT 2, 'viewed a Pokemon listing before September'
  UNION ALL SELECT 3, 'graph affinity: top 20,000 by personalized PageRank'
  UNION ALL SELECT 4, 'taste lookalikes: 20 nearest neighbors of each seed buyer'
  UNION ALL SELECT 5, 'graph affinity or taste lookalike'
  UNION ALL SELECT 6, 'model-ranked top 5,000 of all eligible shoppers'
  UNION ALL SELECT 7, 'model-ranked top 5,000 of previous viewers'
  UNION ALL SELECT 8, 'model-ranked top 5,000 of graph or taste'
)
SELECT s.stage, s.audience, COUNT(*) AS shoppers,
       SUM(r.later_buyer) AS pokemon_buyers_sep_oct,
       CAST(SUM(r.later_buyer) AS DOUBLE) / COUNT(*) AS purchase_rate
FROM ranked r CROSS JOIN stages s
WHERE (s.stage = 1)
   OR (s.stage = 2 AND r.viewer = 1)
   OR (s.stage = 3 AND r.in_graph = 1)
   OR (s.stage = 4 AND r.in_taste = 1)
   OR (s.stage = 5 AND r.in_pool = 1)
   OR (s.stage = 6 AND r.rank_all <= 5000)
   OR (s.stage = 7 AND r.viewer = 1 AND r.rank_viewers <= 5000)
   OR (s.stage = 8 AND r.in_pool = 1 AND r.rank_pool <= 5000)
GROUP BY s.stage, s.audience
ORDER BY s.stage;
