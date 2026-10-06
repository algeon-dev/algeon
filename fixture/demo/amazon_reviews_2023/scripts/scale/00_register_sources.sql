CREATE OR REPLACE EXTERNAL TABLE products STORED AS PARQUET LOCATION 'fixture/demo/amazon_reviews_2023/parquet/products.parquet';
CREATE OR REPLACE EXTERNAL TABLE reviews STORED AS PARQUET LOCATION 'fixture/demo/amazon_reviews_2023/parquet/reviews.parquet';

CREATE OR REPLACE EXTERNAL TABLE semantic_products_text STORED AS PARQUET LOCATION 'target/amazon-reviews-bge-m3/products_text.parquet';
CREATE OR REPLACE EXTERNAL TABLE semantic_products_vectors STORED AS PARQUET LOCATION 'target/amazon-reviews-bge-m3/products_vectors_compact.parquet';
CREATE OR REPLACE EXTERNAL TABLE semantic_reviews_text STORED AS PARQUET LOCATION 'target/amazon-reviews-bge-m3/reviews_text.parquet';
CREATE OR REPLACE EXTERNAL TABLE semantic_reviews_vectors STORED AS PARQUET LOCATION 'target/amazon-reviews-bge-m3/reviews_vectors_compact.parquet';
CREATE OR REPLACE EXTERNAL TABLE semantic_intents_text STORED AS PARQUET LOCATION 'target/amazon-reviews-bge-m3/intents_text.parquet';
CREATE OR REPLACE EXTERNAL TABLE semantic_intents_vectors STORED AS PARQUET LOCATION 'target/amazon-reviews-bge-m3/intents_vectors_compact.parquet';

CREATE OR REPLACE VIEW product_corpus_1000000 AS SELECT id, vector FROM semantic_products_vectors WHERE sample_rank < 1000000;
CREATE OR REPLACE VIEW product_queries_10000 AS SELECT id, vector FROM semantic_products_vectors WHERE sample_rank >= 1000000 AND sample_rank < 1010000;
CREATE OR REPLACE VIEW intent_queries AS SELECT id, vector FROM semantic_intents_vectors;
