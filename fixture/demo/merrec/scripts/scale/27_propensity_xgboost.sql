WITH test AS (
  SELECT e.user_id AS id,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) AS REAL) AS views,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'item_like' THEN 1 ELSE 0 END) AS REAL) AS likes,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type IN ('item_add_to_cart_tap', 'offer_make', 'buy_start') THEN 1 ELSE 0 END) AS REAL) AS intents,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS REAL) AS buys,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'buy_comp' THEN i.price ELSE 0 END) AS REAL) AS gmv,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) AS REAL) AS views_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'item_like' THEN 1 ELSE 0 END) AS REAL) AS likes_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type IN ('item_add_to_cart_tap', 'offer_make', 'buy_start') THEN 1 ELSE 0 END) AS REAL) AS intents_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-02 00:00:00' AND e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS REAL) AS buys_30d,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' THEN i.price ELSE 0 END) / SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' THEN 1 ELSE 0 END) AS REAL) AS mean_price,
    CAST(MAX(CASE WHEN e.event_time >= TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS INT) AS label
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_time >= TIMESTAMP '2023-06-01 00:00:00' AND e.event_time < TIMESTAMP '2023-10-01 00:00:00'
  GROUP BY e.user_id
  HAVING SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) >= 20
),
scores AS (
  SELECT id, prediction AS score FROM cuml_forest_predict(
    input => (SELECT id, LN(1 + views) AS views, LN(1 + likes) AS likes, LN(1 + intents) AS intents, LN(1 + buys) AS buys, LN(1 + gmv) AS gmv, LN(1 + views_30d) AS views_30d, LN(1 + likes_30d) AS likes_30d, LN(1 + intents_30d) AS intents_30d, LN(1 + buys_30d) AS buys_30d, LN(1 + mean_price) AS mean_price FROM test ORDER BY id),
    model => 'propensity_xgb.json', model_format => 'xgboost_json')
), ranked AS (
  SELECT t.label, t.buys_30d,
         ROW_NUMBER() OVER (ORDER BY s.score DESC, t.id) AS rk,
         COUNT(*) OVER () AS shoppers,
         SUM(CASE WHEN t.buys_30d > 0 THEN 1 ELSE 0 END) OVER () AS rule_size
  FROM scores s JOIN test t ON t.id = s.id
)
SELECT MAX(shoppers) AS shoppers, MAX(shoppers) / 100 AS top_1pct, MAX(rule_size) AS rule_size,
       SUM(label) AS buyers,
       SUM(CASE WHEN rk <= shoppers / 100 THEN label ELSE 0 END) AS model_top_1pct_buyers,
       SUM(CASE WHEN rk <= rule_size THEN label ELSE 0 END) AS model_rule_size_buyers,
       SUM(CASE WHEN buys_30d > 0 THEN label ELSE 0 END) AS rule_buyers
FROM ranked;
