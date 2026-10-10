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
    CAST(SUM(CASE WHEN e.event_time >= TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'buy_comp' THEN i.price ELSE 0 END) AS REAL) AS next_gmv
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_time >= TIMESTAMP '2023-06-01 00:00:00' AND e.event_time < TIMESTAMP '2023-10-01 00:00:00'
  GROUP BY e.user_id
  HAVING SUM(CASE WHEN e.event_time < TIMESTAMP '2023-09-01 00:00:00' AND e.event_type = 'item_view' THEN 1 ELSE 0 END) >= 20
),
pred AS (
  SELECT id, prediction FROM cuml_random_forest_regressor(
    training => (SELECT id, LN(1 + views) AS views, LN(1 + likes) AS likes, LN(1 + intents) AS intents, LN(1 + buys) AS buys, LN(1 + gmv) AS gmv, LN(1 + views_30d) AS views_30d, LN(1 + likes_30d) AS likes_30d, LN(1 + intents_30d) AS intents_30d, LN(1 + buys_30d) AS buys_30d, LN(1 + mean_price) AS mean_price, LN(1 + next_gmv) AS label FROM train ORDER BY id),
    predict => (SELECT id, LN(1 + views) AS views, LN(1 + likes) AS likes, LN(1 + intents) AS intents, LN(1 + buys) AS buys, LN(1 + gmv) AS gmv, LN(1 + views_30d) AS views_30d, LN(1 + likes_30d) AS likes_30d, LN(1 + intents_30d) AS intents_30d, LN(1 + buys_30d) AS buys_30d, LN(1 + mean_price) AS mean_price FROM test ORDER BY id),
    n_trees => 100, max_depth => 12, seed => 42)
), ranked AS (
  SELECT t.next_gmv, COUNT(*) OVER () AS cohort,
         ROW_NUMBER() OVER (ORDER BY p.prediction DESC, t.id) AS by_model,
         ROW_NUMBER() OVER (ORDER BY t.gmv DESC, t.id) AS by_past_spend,
         ROW_NUMBER() OVER (ORDER BY t.intents_30d DESC, t.id) AS by_recent_intent
  FROM pred p JOIN test t ON t.id = p.id
)
SELECT COUNT(*) AS shoppers, SUM(next_gmv) AS next_month_gmv,
       SUM(CASE WHEN by_model <= cohort / 100 THEN next_gmv ELSE 0 END) / SUM(next_gmv) AS top1pct_model,
       SUM(CASE WHEN by_past_spend <= cohort / 100 THEN next_gmv ELSE 0 END) / SUM(next_gmv) AS top1pct_past_spend,
       SUM(CASE WHEN by_recent_intent <= cohort / 100 THEN next_gmv ELSE 0 END) / SUM(next_gmv) AS top1pct_recent_intent,
       SUM(CASE WHEN by_model <= cohort / 20 THEN next_gmv ELSE 0 END) / SUM(next_gmv) AS top5pct_model,
       SUM(CASE WHEN by_past_spend <= cohort / 20 THEN next_gmv ELSE 0 END) / SUM(next_gmv) AS top5pct_past_spend
FROM ranked;
