-- CAGRA approximate reference search over the same full Win32 train dataset and test queries.
-- Index is built query-locally over the complete Win32 train dataset (~1.56M files).
-- Queries: 10,000 held-out Win32 test queries from weeks 52-63.
WITH train_index AS (
    SELECT v.id, v.histogram AS vector
    FROM ember_research_vectors_train v
    JOIN ember_metadata_train m ON m.id = v.id
    WHERE m.file_type = 'Win32'
    ORDER BY v.id
), test_queries AS (
    SELECT v.id, v.histogram AS vector
    FROM ember_research_vectors_test v
    JOIN ember_metadata_test m ON m.id = v.id
    WHERE m.file_type = 'Win32'
    ORDER BY v.id
    LIMIT 10000
)
SELECT n.query_ordinal,
       n.query_id,
       qm.week_id AS query_week,
       qm.label AS query_label,
       n.rank,
       n.neighbor_id,
       tm.week_id AS neighbor_week,
       tm.label AS neighbor_label,
       tm.sha256 AS neighbor_sha256,
       tm.family AS neighbor_family,
       n.distance
FROM cuvs_cagra(
    dataset => (SELECT id, vector FROM train_index ORDER BY id),
    queries => (SELECT id, vector FROM test_queries ORDER BY id),
    k => 10,
    metric => 'l2_expanded',
    graph_degree => 32,
    intermediate_graph_degree => 64
) n
JOIN ember_metadata_test qm ON qm.id = n.query_id
JOIN ember_metadata_train tm ON tm.id = n.neighbor_id
ORDER BY n.query_ordinal, n.rank;
