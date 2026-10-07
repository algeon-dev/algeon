WITH october AS (
  SELECT DISTINCT user_id, session_id
  FROM merrec_events
  WHERE event_time >= TIMESTAMP '2023-10-01 00:00:00'
), components AS (
  SELECT vertex, component FROM cugraph_weakly_connected_components(
    edges => (SELECT CONCAT('user:', CAST(user_id AS VARCHAR)) AS src, session_id AS dst FROM october),
    directed => false)
), user_components AS (
  SELECT component, COUNT(*) AS users FROM components WHERE STARTS_WITH(vertex, 'user:') GROUP BY component
)
SELECT CASE WHEN users = 1 THEN '1 user'
            WHEN users <= 10 THEN '2-10 users'
            WHEN users <= 1000 THEN '11-1,000 users'
            ELSE 'over 1,000 users' END AS component_size,
       COUNT(*) AS components, SUM(users) AS users, MAX(users) AS largest
FROM user_components
GROUP BY 1
ORDER BY largest;
