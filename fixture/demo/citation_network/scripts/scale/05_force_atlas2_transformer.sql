WITH layout AS (
  SELECT vertex, x, y
  FROM cugraph_force_atlas2(
    edges => (SELECT src, dst FROM attention_ego_edges), max_iter => 500, seed => 42)),
community AS (
  SELECT vertex, "partition"
  FROM cugraph_louvain(edges => (SELECT src, dst FROM attention_ego_edges)))
SELECT l.vertex, l.x, l.y, c."partition", p.title, p.year
FROM layout l
JOIN community c ON c.vertex = l.vertex
JOIN papers p ON p.paper_id = l.vertex
