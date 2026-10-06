WITH ranked AS (
  SELECT vertex, value, ROW_NUMBER() OVER (ORDER BY value DESC) AS rank
  FROM cugraph_pagerank(edges => (SELECT src, dst FROM citation_edges)))
SELECT r.rank, p.title, p.year, r.value
FROM ranked r JOIN papers p ON p.paper_id = r.vertex
WHERE p.title = 'The anatomy of a large-scale hypertextual Web search engine'
  AND p.year = 1998 AND p.venue = 'The Web Conference'
