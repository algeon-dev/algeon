-- Exact squared-L2 reference search over full Win32 train dataset and test queries.
-- Corpus: full deduplicated Win32 training dataset (~1.56M files) across weeks 0-51.
-- Queries: 10,000 held-out Win32 test queries from weeks 52-63.
-- Post-search enrichment joins query and neighbor metadata only after vector search.
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
FROM cuvs_brute_force_knn(
    dataset => (SELECT id, vector FROM train_index ORDER BY id),
    queries => (SELECT id, vector FROM test_queries ORDER BY id),
    k => 10,
    metric => 'l2_expanded'
) n
JOIN ember_metadata_test qm ON qm.id = n.query_id
JOIN ember_metadata_train tm ON tm.id = n.neighbor_id
ORDER BY n.query_ordinal, n.rank;
