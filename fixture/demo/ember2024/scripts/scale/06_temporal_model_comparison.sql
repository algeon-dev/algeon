-- Unified temporal comparison training both logistic regression and random forest
-- on the full deduplicated Win32 training dataset (~1.56M files) and predicting on
-- the full held-out Win32 test dataset (359,994 files).
-- Evaluates both query-local classifiers side-by-side in one GPU statement.
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
),
random_forest AS (
SELECT p.id, p.prediction, m.label, m.week_id, m.sha256, m.family
FROM cuml_random_forest_classifier(
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
  n_trees => 100,
  max_depth => 12,
  seed => 42
) p
JOIN ember_metadata_test m ON m.id = p.id
),
classified AS (
  SELECT 'logistic' AS model, id, prediction, label FROM logistic
  UNION ALL
  SELECT 'random_forest' AS model, id, prediction, label FROM random_forest
)
SELECT model, COUNT(*) AS files,
       SUM(CASE WHEN prediction = 1 AND label = 1 THEN 1 ELSE 0 END) AS true_positive,
       SUM(CASE WHEN prediction = 1 AND label = 0 THEN 1 ELSE 0 END) AS false_positive,
       SUM(CASE WHEN prediction = 0 AND label = 1 THEN 1 ELSE 0 END) AS false_negative,
       SUM(CASE WHEN prediction = 0 AND label = 0 THEN 1 ELSE 0 END) AS true_negative
FROM classified
GROUP BY model
ORDER BY model;
