WITH community_fos AS (
  SELECT c.community, p.primary_fos, COUNT(*) AS n
  FROM cugraph_louvain(edges => (SELECT src, dst FROM ai_edges)) c
  JOIN papers p ON p.paper_id = c.vertex
  GROUP BY c.community, p.primary_fos),
ranked AS (
  SELECT SUM(n) OVER (PARTITION BY community) AS members,
         ROW_NUMBER() OVER (PARTITION BY community ORDER BY n DESC) AS rn,
         primary_fos,
         n
  FROM community_fos)
SELECT members, rn, primary_fos, n
FROM ranked
WHERE members > 2500 AND rn <= 3
ORDER BY members DESC, rn
