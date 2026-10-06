SELECT i.c0_name AS category,
       COUNT(*) AS events,
       SUM(CASE WHEN e.event_type = 'item_view' THEN 1 ELSE 0 END) AS views,
       SUM(CASE WHEN e.event_type = 'item_like' THEN 1 ELSE 0 END) AS likes,
       SUM(CASE WHEN e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS completed_purchases
FROM merrec_events e
JOIN merrec_items i ON i.item_id = e.item_id
GROUP BY i.c0_name
ORDER BY category;
