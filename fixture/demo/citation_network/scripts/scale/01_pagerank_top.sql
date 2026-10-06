SELECT p.title, p.year, p.n_citation, r.value AS pagerank
FROM cugraph_pagerank(edges => (SELECT src, dst FROM citation_edges)) r
JOIN papers p ON p.paper_id = r.vertex
ORDER BY r.value DESC
LIMIT 5
