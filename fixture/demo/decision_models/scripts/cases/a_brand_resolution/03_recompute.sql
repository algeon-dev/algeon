WITH likes AS (
  SELECT c.canonical_id, c.brand_id, e.user_id
  FROM merrec_events e
  JOIN merrec_items i ON i.item_id = e.item_id
  JOIN brand_canonical c ON c.brand_id = CAST(i.brand_id AS BIGINT)
  WHERE e.event_type = 'item_like'
), member_fans AS (
  SELECT canonical_id, brand_id, COUNT(DISTINCT user_id) AS fans
  FROM likes GROUP BY canonical_id, brand_id
), union_fans AS (
  SELECT canonical_id, COUNT(DISTINCT user_id) AS exact_union_fans
  FROM likes GROUP BY canonical_id
), brands AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, MIN(brand_name) AS brand_name
  FROM merrec_items WHERE brand_id <> 0 GROUP BY brand_id
)
SELECT m.canonical_id,
       MIN(cb.brand_name) AS canonical_name,
       COUNT(*) AS members,
       MAX(m.fans) AS primary_fans,
       SUM(m.fans) AS naive_sum_fans,
       MAX(u.exact_union_fans) AS exact_union_fans,
       ROUND(100.0 * (MAX(u.exact_union_fans) - MAX(m.fans)) / MAX(m.fans), 1) AS gain_pct,
       ROUND(100.0 * (SUM(m.fans) - MAX(u.exact_union_fans)) / SUM(m.fans), 1) AS double_count_pct
FROM member_fans m
JOIN union_fans u ON u.canonical_id = m.canonical_id
JOIN brands cb ON cb.brand_id = m.canonical_id
GROUP BY m.canonical_id
ORDER BY exact_union_fans DESC, m.canonical_id
