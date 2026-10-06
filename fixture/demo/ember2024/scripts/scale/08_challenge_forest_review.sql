-- Ranking high-scoring challenge files with LightGBM forest inference and attribution review.
-- Model: EMBER2024_all.model (LightGBM text format, 2568 features).
-- Precomputed CPU SHAP values are aggregated into 12 feature blocks and ranked with cuVS Top-K.
SELECT p.id, m.sha256, m.file_type, m.week_id, p.prediction,
       t.rank, g.group_name, t.value AS positive_log_odds_contribution
FROM cuml_forest_predict(
  input => (
    SELECT id, vector
    FROM ember_research_vectors_challenge
    ORDER BY id
  ),
  model => 'EMBER2024_all.model',
  model_format => 'lightgbm'
) p
JOIN cuvs_top_k(
  input => (SELECT id, vector FROM ember_attributions ORDER BY id),
  k => 3,
  select => 'max'
) t ON t.id = p.id
JOIN ember_feature_groups g ON g.group_index = t.position
JOIN ember_metadata_challenge m ON m.id = p.id
WHERE p.prediction >= 0.99 AND t.value > 0
ORDER BY p.prediction DESC, p.id, t.rank
LIMIT 30;
