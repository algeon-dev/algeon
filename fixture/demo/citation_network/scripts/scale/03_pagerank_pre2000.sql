SELECT p.year, p.title
FROM cugraph_pagerank(edges => (SELECT src, dst FROM edges_pre2000)) r
JOIN papers p ON p.paper_id = r.vertex
ORDER BY r.value DESC
LIMIT 6
