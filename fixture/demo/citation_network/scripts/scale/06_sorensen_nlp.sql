SELECT a.title AS paper_a, b.title AS paper_b,
       ROUND(r.similarity, 3) AS sorensen
FROM cugraph_sorensen(
  edges => (SELECT src, dst FROM bibliography_edges),
  vertex_pairs => (
    SELECT a.paper_id AS first, b.paper_id AS second
    FROM similarity_candidates a
    JOIN similarity_candidates b ON a.paper_id < b.paper_id
    WHERE LOWER(a.title) <> LOWER(b.title)),
  directed => false) r
JOIN papers a ON a.paper_id = r.first
JOIN papers b ON b.paper_id = r.second
ORDER BY r.similarity DESC, r.first, r.second
LIMIT 5
