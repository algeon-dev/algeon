-- Logistic regression classifier trained on full deduplicated Win32 training dataset (~1.56M files)
-- and evaluated on the full held-out Win32 test dataset (359,994 files).
-- Models train query-locally and labels are joined after prediction for confusion metrics.
WITH logistic AS (
SELECT p.id, p.prediction, m.label, m.week_id, m.sha256, m.family
FROM cuml_logistic_regression(
  training => (
    SELECT v.id, v.histogram AS vector, v.label
    FROM ember_research_vectors_train v
    JOIN ember_metadata_train m ON m.id = v.id
    WHERE m.file_type = 'Win32'
    ORDER BY v.id
  ),
  predict => (
    SELECT v.id, v.histogram AS vector
    FROM ember_research_vectors_test v
    JOIN ember_metadata_test m ON m.id = v.id
    WHERE m.file_type = 'Win32'
    ORDER BY v.id
  ),
  max_iter => 1000,
  penalty_l2 => 1.0
) p
JOIN ember_metadata_test m ON m.id = p.id
)
SELECT 'logistic' AS model, COUNT(*) AS files,
       SUM(CASE WHEN prediction = 1 AND label = 1 THEN 1 ELSE 0 END) AS true_positive,
       SUM(CASE WHEN prediction = 1 AND label = 0 THEN 1 ELSE 0 END) AS false_positive,
       SUM(CASE WHEN prediction = 0 AND label = 1 THEN 1 ELSE 0 END) AS false_negative,
       SUM(CASE WHEN prediction = 0 AND label = 0 THEN 1 ELSE 0 END) AS true_negative
FROM logistic;
