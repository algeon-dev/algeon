SELECT user_id,
       COUNT(*) AS events,
       SUM(CASE WHEN event_type = 'item_view' THEN 1 ELSE 0 END) AS views,
       SUM(CASE WHEN event_type = 'item_like' THEN 1 ELSE 0 END) AS likes,
       SUM(CASE WHEN event_type = 'buy_comp' THEN 1 ELSE 0 END) AS completed_purchases
FROM merrec_events
GROUP BY user_id
ORDER BY events DESC, user_id
LIMIT 100;
