SELECT vertex, value
FROM cugraph_pagerank(
  edges => (
    SELECT item_id AS src, -(user_id + 1) AS dst,
           CAST(COUNT(*) AS DOUBLE) AS weight
    FROM merrec_events
    WHERE event_type = 'item_view'
    GROUP BY item_id, user_id
  ),
  directed => false
)
WHERE vertex >= 0
ORDER BY value DESC, vertex
LIMIT 20;
