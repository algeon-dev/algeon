WITH october AS (
  SELECT user_id, session_id, MIN(event_id) AS first_event
  FROM merrec_events
  WHERE event_time >= TIMESTAMP '2023-10-01 00:00:00'
  GROUP BY user_id, session_id
), sessions AS (
  SELECT session_id, MIN(first_event) AS session_vertex FROM october GROUP BY session_id
), components AS (
  SELECT vertex, component FROM cugraph_weakly_connected_components(
    edges => (SELECT -(o.user_id + 1) AS src, s.session_vertex AS dst
              FROM october o JOIN sessions s ON s.session_id = o.session_id),
    directed => false)
), user_components AS (
  SELECT component, COUNT(*) AS users FROM components WHERE vertex < 0 GROUP BY component
)
SELECT CASE WHEN users = 1 THEN '1 user'
            WHEN users <= 10 THEN '2-10 users'
            WHEN users <= 1000 THEN '11-1,000 users'
            ELSE 'over 1,000 users' END AS component_size,
       COUNT(*) AS components, SUM(users) AS users, MAX(users) AS largest
FROM user_components
GROUP BY 1
ORDER BY largest;
