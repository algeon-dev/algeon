"""Upload a TPC-DS Parquet fixture as Iceberg v2 tables via Glue.

Run via run_upload.sh which sets AWS_PROFILE and the venv. Reads
SF + SOURCE_DIR from env so the same script handles sf1 / sf10.
"""

import os
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession

SF = os.environ["TPCDS_SF"]
SOURCE_DIR = Path(os.environ["TPCDS_SOURCE_DIR"]).resolve()
GLUE_DB = os.environ.get("TPCDS_GLUE_DB", "morrist_tpcds")
WAREHOUSE = os.environ.get(
    "TPCDS_WAREHOUSE",
    f"s3://example-warehouse/tpcds_{SF}/",
)
CATALOG = "glue_tpcds"
ONLY = {t.strip() for t in os.environ.get("TPCDS_ONLY", "").split(",") if t.strip()}

TABLES = [
    "call_center", "catalog_page", "catalog_returns", "catalog_sales",
    "customer", "customer_address", "customer_demographics", "date_dim",
    "household_demographics", "income_band", "inventory", "item",
    "promotion", "reason", "ship_mode", "store", "store_returns",
    "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
    "web_sales", "web_site",
]

ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1"
ICEBERG_AWS = "org.apache.iceberg:iceberg-aws-bundle:1.6.1"

spark = (
    SparkSession.builder
    .appName(f"tpcds-{SF}-iceberg-upload")
    .config("spark.jars.packages", f"{ICEBERG_RUNTIME},{ICEBERG_AWS}")
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
    .config(f"spark.sql.catalog.{CATALOG}.catalog-impl",
            "org.apache.iceberg.aws.glue.GlueCatalog")
    .config(f"spark.sql.catalog.{CATALOG}.io-impl",
            "org.apache.iceberg.aws.s3.S3FileIO")
    .config(f"spark.sql.catalog.{CATALOG}.warehouse", WAREHOUSE)
    .config(f"spark.sql.catalog.{CATALOG}.glue.id", "123456789012")
    .config("spark.sql.defaultCatalog", CATALOG)
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"[upload] sf={SF} source={SOURCE_DIR} warehouse={WAREHOUSE} db={GLUE_DB}", flush=True)

failures = []
for table in TABLES:
    if ONLY and table not in ONLY:
        continue
    parquet_path = SOURCE_DIR / f"{table}.parquet"
    if not parquet_path.exists():
        print(f"[skip] {table}: {parquet_path} missing", flush=True)
        failures.append((table, "source missing"))
        continue
    target = f"{CATALOG}.{GLUE_DB}.{table}_{SF}"
    location = f"{WAREHOUSE.rstrip('/')}/{table}"
    t0 = time.monotonic()
    try:
        print(f"[start] {target} <- {parquet_path}", flush=True)
        spark.sql(f"DROP TABLE IF EXISTS {target} PURGE")
        # CTAS uses the parquet file's own schema (preserves decimal/date/varchar types).
        spark.sql(
            f"CREATE TABLE {target} "
            f"USING iceberg "
            f"LOCATION '{location}' "
            f"TBLPROPERTIES ('format-version'='2') "
            f"AS SELECT * FROM parquet.`{parquet_path}`"
        )
        rows = spark.sql(f"SELECT COUNT(*) AS c FROM {target}").collect()[0]["c"]
        elapsed = time.monotonic() - t0
        print(f"[done]  {target} rows={rows} elapsed={elapsed:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[fail]  {target}: {exc}", flush=True)
        failures.append((table, str(exc)))

spark.stop()

if failures:
    print(f"[summary] {len(failures)} failures:", flush=True)
    for t, e in failures:
        print(f"  - {t}: {e[:200]}", flush=True)
    sys.exit(1)
print("[summary] all tables uploaded", flush=True)
