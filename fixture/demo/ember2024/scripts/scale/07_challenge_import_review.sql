-- Challenge import graph review queue combining Personalized PageRank and Jaccard similarity.
-- Challenge cohort: 6,315 files (3,225 Win32). Retained graph: 926 files, 4,988 symbols, 80,673 edges.
-- Seed: unlabeled Win32 challenge sample 20005740.
WITH candidates AS (
  SELECT p.vertex
  FROM cugraph_personalized_pagerank(
    edges => (
      SELECT src_id AS src, dst_id AS dst
      FROM ember_48_challenge_import_edges
    ),
    personalization => (
      SELECT id AS vertex, CAST(1.0 AS DOUBLE) AS value
      FROM ember_49_challenge_graph_seed
    ),
    directed => false
  ) p
  JOIN ember_48_challenge_import_sample_counts c ON c.id = p.vertex
  CROSS JOIN ember_49_challenge_graph_seed seed
  WHERE p.vertex <> seed.id AND c.retained_import_count >= 5
  ORDER BY p.value DESC, p.vertex
  LIMIT 25
), similarities AS (
  SELECT * FROM cugraph_jaccard(
    edges => (
      SELECT src_id AS src, dst_id AS dst
      FROM ember_48_challenge_import_edges
    ),
    vertex_pairs => (
      SELECT seed.id AS first, c.vertex AS second
      FROM candidates c CROSS JOIN ember_49_challenge_graph_seed seed
    ),
    directed => false
  )
)
SELECT j.second AS candidate_id, m.sha256, m.family,
       counts.retained_import_count, j.similarity,
       COUNT(*) AS shared_rare_imports,
       MIN(symbols.symbol) AS example_shared_import
FROM similarities j
JOIN ember_48_challenge_import_edges seed_edges ON seed_edges.src_id = j.first
JOIN ember_48_challenge_import_edges candidate_edges
  ON candidate_edges.src_id = j.second AND candidate_edges.dst_id = seed_edges.dst_id
JOIN ember_48_challenge_import_symbols symbols ON symbols.symbol_id = seed_edges.dst_id
JOIN ember_metadata_challenge m ON m.id = j.second
JOIN ember_48_challenge_import_sample_counts counts ON counts.id = j.second
GROUP BY j.second, m.sha256, m.family, counts.retained_import_count, j.similarity
ORDER BY j.similarity DESC, shared_rare_imports DESC, j.second
LIMIT 15;
