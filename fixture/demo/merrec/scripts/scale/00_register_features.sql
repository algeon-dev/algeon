CREATE EXTERNAL TABLE IF NOT EXISTS mr_scale_tastes STORED AS PARQUET LOCATION 'target/merrec-exploration/results/03_user_features.parquet';
CREATE EXTERNAL TABLE IF NOT EXISTS mr_taste64 STORED AS PARQUET LOCATION 'target/merrec-exploration/results/14_taste64.parquet';
