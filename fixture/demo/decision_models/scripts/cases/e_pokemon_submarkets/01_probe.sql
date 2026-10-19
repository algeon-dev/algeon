WITH pokemon_catalog AS (
  SELECT item_id, name, category_path, item_condition_name, color, price
  FROM merrec_items
  WHERE brand_id = 17470
), engagement AS (
  SELECT e.item_id,
         SUM(CASE WHEN e.event_type = 'item_view' THEN 1 ELSE 0 END) AS views,
         SUM(CASE WHEN e.event_type = 'item_like' THEN 1 ELSE 0 END) AS likes,
         SUM(CASE WHEN e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS buys,
         COUNT(DISTINCT e.user_id) AS users
  FROM merrec_events e
  JOIN pokemon_catalog p ON p.item_id = e.item_id
  WHERE e.event_time < TIMESTAMP '2023-09-01 00:00:00'
  GROUP BY e.item_id
)
SELECT p.item_id, p.name, p.category_path, p.item_condition_name, p.color, p.price,
       g.views, g.likes, g.buys, g.users
FROM engagement g
JOIN pokemon_catalog p ON p.item_id = g.item_id
WHERE g.views >= 50
ORDER BY g.views DESC, p.item_id
LIMIT {max_entities}
