WITH train AS (
  SELECT e.user_id AS id,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) AS REAL) AS views,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'item_like' THEN 1 ELSE 0 END) AS REAL) AS likes,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type IN ('item_add_to_cart_tap', 'offer_make', 'buy_start') THEN 1 ELSE 0 END) AS REAL) AS intents,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS REAL) AS buys,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'buy_comp' THEN i.price ELSE 0 END) AS REAL) AS gmv,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-07-02 00:00:00' AND e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) AS REAL) AS views_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-07-02 00:00:00' AND e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'item_like' THEN 1 ELSE 0 END) AS REAL) AS likes_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-07-02 00:00:00' AND e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type IN ('item_add_to_cart_tap', 'offer_make', 'buy_start') THEN 1 ELSE 0 END) AS REAL) AS intents_30d,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-07-02 00:00:00' AND e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS REAL) AS buys_30d,
    CAST(SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' THEN i.price ELSE 0 END) / SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' THEN 1 ELSE 0 END) AS REAL) AS mean_price,
    MAX(CASE WHEN e.event_time >= TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'buy_comp' THEN CAST(1 AS INT) ELSE CAST(0 AS INT) END) AS label,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'buy_comp' THEN i.price ELSE 0 END) AS REAL) AS next_gmv
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_time >= TIMESTAMP '2023-05-01 00:00:00' AND e.event_time < TIMESTAMP '2023-08-31 00:00:00'
  GROUP BY e.user_id
  HAVING SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) >= 20
),
test AS (
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
    MAX(CASE WHEN e.event_time >= TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'buy_comp' THEN CAST(1 AS INT) ELSE CAST(0 AS INT) END) AS label,
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'buy_comp' THEN i.price ELSE 0 END) AS REAL) AS next_gmv
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_time >= TIMESTAMP '2023-06-01 00:00:00' AND e.event_time < TIMESTAMP '2023-10-01 00:00:00'
  GROUP BY e.user_id
  HAVING SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) >= 20
),
pred AS (
  SELECT id, prediction FROM cuml_random_forest_classifier(
    training => (SELECT id, COALESCE(LN(1 + views), CAST(0 AS REAL)) AS views, COALESCE(LN(1 + likes), CAST(0 AS REAL)) AS likes, COALESCE(LN(1 + intents), CAST(0 AS REAL)) AS intents, COALESCE(LN(1 + buys), CAST(0 AS REAL)) AS buys, COALESCE(LN(1 + gmv), CAST(0 AS REAL)) AS gmv, COALESCE(LN(1 + views_30d), CAST(0 AS REAL)) AS views_30d, COALESCE(LN(1 + likes_30d), CAST(0 AS REAL)) AS likes_30d, COALESCE(LN(1 + intents_30d), CAST(0 AS REAL)) AS intents_30d, COALESCE(LN(1 + buys_30d), CAST(0 AS REAL)) AS buys_30d, COALESCE(LN(1 + mean_price), CAST(0 AS REAL)) AS mean_price, COALESCE(label, CAST(0 AS INT)) AS label FROM train WHERE label = 1 OR ROUND(CAST(id AS DOUBLE) / 20.0) = CAST(id AS DOUBLE) / 20.0 ORDER BY id),
    predict => (SELECT id, COALESCE(LN(1 + views), CAST(0 AS REAL)) AS views, COALESCE(LN(1 + likes), CAST(0 AS REAL)) AS likes, COALESCE(LN(1 + intents), CAST(0 AS REAL)) AS intents, COALESCE(LN(1 + buys), CAST(0 AS REAL)) AS buys, COALESCE(LN(1 + gmv), CAST(0 AS REAL)) AS gmv, COALESCE(LN(1 + views_30d), CAST(0 AS REAL)) AS views_30d, COALESCE(LN(1 + likes_30d), CAST(0 AS REAL)) AS likes_30d, COALESCE(LN(1 + intents_30d), CAST(0 AS REAL)) AS intents_30d, COALESCE(LN(1 + buys_30d), CAST(0 AS REAL)) AS buys_30d, COALESCE(LN(1 + mean_price), CAST(0 AS REAL)) AS mean_price FROM test ORDER BY id),
    n_trees => 100, max_depth => 12, seed => 42)
)
SELECT 'model' AS audience, SUM(p.prediction) AS flagged, SUM(p.prediction * t.label) AS buyers_flagged,
       SUM(t.label) AS buyers, CAST(SUM(p.prediction * t.label) AS DOUBLE) / SUM(p.prediction) AS precision,
       CAST(SUM(p.prediction * t.label) AS DOUBLE) / SUM(t.label) AS recall
FROM pred p JOIN test t ON t.id = p.id
UNION ALL
SELECT 'bought in last 30 days', SUM(CASE WHEN buys_30d > 0 THEN 1 ELSE 0 END),
       SUM(CASE WHEN buys_30d > 0 THEN label ELSE 0 END), SUM(label),
       CAST(SUM(CASE WHEN buys_30d > 0 THEN label ELSE 0 END) AS DOUBLE) / SUM(CASE WHEN buys_30d > 0 THEN 1 ELSE 0 END),
       CAST(SUM(CASE WHEN buys_30d > 0 THEN label ELSE 0 END) AS DOUBLE) / SUM(label)
FROM test;
