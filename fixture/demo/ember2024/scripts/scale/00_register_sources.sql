CREATE OR REPLACE EXTERNAL TABLE ember_metadata_train STORED AS PARQUET LOCATION 'target/ember2024-research/input_metadata_train.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_metadata_test STORED AS PARQUET LOCATION 'target/ember2024-research/input_metadata_test.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_metadata_challenge STORED AS PARQUET LOCATION 'target/ember2024-research/input_metadata_challenge.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_metadata STORED AS PARQUET LOCATION 'target/ember2024-research/input_metadata.parquet';

CREATE OR REPLACE EXTERNAL TABLE ember_research_vectors_train STORED AS PARQUET LOCATION 'target/ember2024-research/input_research_vectors_train.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_research_vectors_test STORED AS PARQUET LOCATION 'target/ember2024-research/input_research_vectors_test.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_research_vectors_challenge STORED AS PARQUET LOCATION 'target/ember2024-research/input_research_vectors_challenge.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_research_vectors STORED AS PARQUET LOCATION 'target/ember2024-research/input_research_vectors.parquet';

CREATE OR REPLACE EXTERNAL TABLE ember_48_challenge_import_edges STORED AS PARQUET LOCATION 'target/ember2024-research/input_48_challenge_import_edges.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_48_challenge_import_symbols STORED AS PARQUET LOCATION 'target/ember2024-research/input_48_challenge_import_symbols.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_48_challenge_import_sample_counts STORED AS PARQUET LOCATION 'target/ember2024-research/input_48_challenge_import_sample_counts.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_49_challenge_graph_seed STORED AS PARQUET LOCATION 'target/ember2024-research/input_49_challenge_graph_seed.parquet';

CREATE OR REPLACE EXTERNAL TABLE ember_attributions STORED AS PARQUET LOCATION 'target/ember2024-research/input_attributions.parquet';
CREATE OR REPLACE EXTERNAL TABLE ember_feature_groups STORED AS PARQUET LOCATION 'target/ember2024-research/input_feature_groups.parquet';
