CREATE EXTERNAL TABLE customer STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/customer.parquet';
CREATE EXTERNAL TABLE lineitem STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/lineitem.parquet';
CREATE EXTERNAL TABLE nation STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/nation.parquet';
CREATE EXTERNAL TABLE orders STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/orders.parquet';
CREATE EXTERNAL TABLE part STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/part.parquet';
CREATE EXTERNAL TABLE partsupp STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/partsupp.parquet';
CREATE EXTERNAL TABLE region STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/region.parquet';
CREATE EXTERNAL TABLE supplier STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/supplier.parquet';
