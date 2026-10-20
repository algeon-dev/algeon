-- Bridge items: for each brand in {brand_ids}, the 25 items most liked by Barbie fans in July and August 2023.
-- A Barbie fan liked a Barbie (brand_id 2595) item between 2023-05-01 and 2023-09-01. Likes count events;
-- fan users count distinct fans. An item qualifies with at least 2 Barbie-fan likes, or with 'barbie' in its
-- title and at least 3 likes in total, so that keyword items Barbie fans ignored stay visible to the control.
WITH catalog AS (
  SELECT item_id, CAST(brand_id AS BIGINT) AS brand_id, brand_name, name, category_path, color, price
  FROM merrec_items WHERE brand_id IN ({brand_ids})
), barbie_fans AS (
  SELECT DISTINCT e.user_id
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  WHERE e.event_type = 'item_like' AND i.brand_id = 2595
    AND e.event_time >= TIMESTAMP '2023-05-01 00:00:00'
    AND e.event_time < TIMESTAMP '2023-09-01 00:00:00'
), after_likes AS (
  SELECT e.user_id, e.item_id, c.brand_id
  FROM merrec_events e JOIN catalog c ON c.item_id = e.item_id
  WHERE e.event_type = 'item_like'
    AND e.event_time >= TIMESTAMP '2023-07-01 00:00:00'
    AND e.event_time < TIMESTAMP '2023-09-01 00:00:00'
), item_likes AS (
  SELECT a.item_id, a.brand_id, COUNT(*) AS likes_after,
         SUM(CASE WHEN f.user_id IS NULL THEN 0 ELSE 1 END) AS fan_likes_after,
         COUNT(DISTINCT f.user_id) AS fan_users_after
  FROM after_likes a LEFT JOIN barbie_fans f ON f.user_id = a.user_id
  GROUP BY a.item_id, a.brand_id
), ranked AS (
  SELECT l.brand_id, c.brand_name, l.item_id, c.name AS item_name, c.category_path, c.color, c.price,
         LOWER(c.name) LIKE '%barbie%' AS keyword_barbie,
         l.likes_after, l.fan_likes_after, l.fan_users_after,
         ROW_NUMBER() OVER (PARTITION BY l.brand_id
                            ORDER BY l.fan_likes_after DESC, l.likes_after DESC, l.item_id) AS item_rank
  FROM item_likes l JOIN catalog c ON c.item_id = l.item_id
  WHERE l.fan_likes_after >= 2 OR (LOWER(c.name) LIKE '%barbie%' AND l.likes_after >= 3)
)
SELECT brand_id, brand_name, item_id, item_name, category_path, color, price, keyword_barbie,
       likes_after, fan_likes_after, fan_users_after, item_rank
FROM ranked
WHERE item_rank <= 25
ORDER BY brand_id, item_rank;
