CREATE OR REPLACE EXTERNAL TABLE citation_edges STORED AS PARQUET LOCATION 'fixture/demo/citation_network/v1/parquet/edges_by_src.parquet';
CREATE OR REPLACE EXTERNAL TABLE citation_edges_by_dst STORED AS PARQUET LOCATION 'fixture/demo/citation_network/v1/parquet/edges_by_dst.parquet';
CREATE OR REPLACE EXTERNAL TABLE papers STORED AS PARQUET LOCATION 'fixture/demo/citation_network/v1/parquet/profiles_by_paper_id.parquet';
CREATE OR REPLACE EXTERNAL TABLE paper_authors STORED AS PARQUET LOCATION 'fixture/demo/citation_network/v1/parquet/paper_authors.parquet';
CREATE OR REPLACE EXTERNAL TABLE paper_fos STORED AS PARQUET LOCATION 'fixture/demo/citation_network/v1/parquet/paper_fos.parquet';
