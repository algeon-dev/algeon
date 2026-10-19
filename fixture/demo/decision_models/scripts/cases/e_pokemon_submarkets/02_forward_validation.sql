WITH labels AS (
  SELECT d.item_id, {label} AS label
  FROM {label_table} d
), early AS (
  SELECT l.label, e.user_id, e.event_type
  FROM merrec_events e
  JOIN labels l ON l.item_id = e.item_id
  WHERE e.event_time < TIMESTAMP '2023-09-01 00:00:00'
), catalog AS (
  SELECT l.label, COUNT(*) AS items, ROUND(AVG(i.price), 2) AS mean_price
  FROM labels l
  JOIN merrec_items i ON i.item_id = l.item_id
  GROUP BY l.label
), engagement AS (
  SELECT label,
         SUM(CASE WHEN event_type = 'item_view' THEN 1 ELSE 0 END) AS views,
         SUM(CASE WHEN event_type = 'item_like' THEN 1 ELSE 0 END) AS likes,
         SUM(CASE WHEN event_type = 'buy_comp' THEN 1 ELSE 0 END) AS buys
  FROM early
  GROUP BY label
), viewers AS (
  SELECT DISTINCT label, user_id FROM early WHERE event_type = 'item_view'
), pokemon_buyers AS (
  SELECT e.user_id,
         MAX(CASE WHEN e.event_time >= TIMESTAMP '2023-07-01 00:00:00'
                   AND e.event_time < TIMESTAMP '2023-09-01 00:00:00' THEN 1 ELSE 0 END) AS prior_buyer,
         MAX(CASE WHEN e.event_time >= TIMESTAMP '2023-09-01 00:00:00' THEN 1 ELSE 0 END) AS later_buyer
  FROM merrec_events e
  JOIN merrec_items i ON i.item_id = e.item_id
  WHERE i.brand_id = 17470 AND e.event_type = 'buy_comp'
  GROUP BY e.user_id
), outcome AS (
  SELECT v.label, COUNT(*) AS shoppers, SUM(COALESCE(b.later_buyer, 0)) AS buyers
  FROM viewers v
  LEFT JOIN pokemon_buyers b ON b.user_id = v.user_id
  WHERE COALESCE(b.prior_buyer, 0) = 0
  GROUP BY v.label
)
SELECT c.label, c.items, c.mean_price, g.views, g.likes, g.buys,
       COALESCE(o.shoppers, 0) AS shoppers, COALESCE(o.buyers, 0) AS buyers
FROM catalog c
LEFT JOIN engagement g ON g.label = c.label
LEFT JOIN outcome o ON o.label = c.label
ORDER BY c.label
