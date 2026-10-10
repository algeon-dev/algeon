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
    CAST(MAX(CASE WHEN e.event_time >= TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS INT) AS label
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_time >= TIMESTAMP '2023-05-01 00:00:00' AND e.event_time < TIMESTAMP '2023-08-31 00:00:00'
  GROUP BY e.user_id
  HAVING SUM(CASE WHEN e.event_time < TIMESTAMP '2023-08-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) >= 20
)
SELECT id, LN(1 + views) AS views, LN(1 + likes) AS likes, LN(1 + intents) AS intents, LN(1 + buys) AS buys, LN(1 + gmv) AS gmv, LN(1 + views_30d) AS views_30d, LN(1 + likes_30d) AS likes_30d, LN(1 + intents_30d) AS intents_30d, LN(1 + buys_30d) AS buys_30d, LN(1 + mean_price) AS mean_price, label
FROM train
ORDER BY id;
