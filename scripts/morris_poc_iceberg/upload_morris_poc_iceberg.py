"""Upload TPC-H + TPC-DS Parquet fixtures into the `demo_db` Glue database
as Iceberg v2 tables, with folder-style S3 layout:

    s3://example-warehouse/demo_db/<workload>/<sf>/<table>/

Table naming inside demo_db:

    demo_db.<workload>_<sf>_<table>   e.g. demo_db.tpch_sf10_lineitem

Driven entirely by env vars; run via run_upload.sh.

Env:
  MORRIS_POC_GLUE_DB       default demo_db
  MORRIS_POC_WAREHOUSE     default s3://example-warehouse/demo_db/
  MORRIS_POC_FIXTURE_ROOT  default <repo>/tests/data
  MORRIS_POC_SCOPE         comma list of <workload>:<sf>, e.g. "tpch:sf1,tpcds:sf10"
                           (default: all combinations that have a fixture dir)
  MORRIS_POC_ONLY          optional comma list of bare table names to filter
"""

import os
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession

GLUE_DB = os.environ.get("MORRIS_POC_GLUE_DB", "demo_db")
WAREHOUSE = os.environ.get(
    "MORRIS_POC_WAREHOUSE",
    "s3://example-warehouse/demo_db/",
).rstrip("/") + "/"
FIXTURE_ROOT = Path(os.environ.get("MORRIS_POC_FIXTURE_ROOT", "tests/data")).resolve()
ONLY = {t.strip() for t in os.environ.get("MORRIS_POC_ONLY", "").split(",") if t.strip()}
CATALOG = "glue_demo_db"

TPCH_TABLES = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier",
]
TPCDS_TABLES = [
    "call_center", "catalog_page", "catalog_returns", "catalog_sales",
    "customer", "customer_address", "customer_demographics", "date_dim",
    "household_demographics", "income_band", "inventory", "item",
    "promotion", "reason", "ship_mode", "store", "store_returns",
    "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
    "web_sales", "web_site",
]

# (workload, sf) -> (fixture_subdir, tables)
# tpch uses *_optimized fixtures (with sibling-split lineitem at sf100).
WORKLOADS = {
    ("tpch", "sf1"):    ("tpch/sf1_optimized/parquet",   TPCH_TABLES),
    ("tpch", "sf10"):   ("tpch/sf10_optimized/parquet",  TPCH_TABLES),
    ("tpch", "sf100"):  ("tpch/sf100_optimized/parquet", TPCH_TABLES),
    ("tpcds", "sf1"):   ("tpcds/sf1/parquet",            TPCDS_TABLES),
    ("tpcds", "sf10"):  ("tpcds/sf10/parquet",           TPCDS_TABLES),
    ("tpcds", "sf100"): ("tpcds/sf100/parquet",          TPCDS_TABLES),
}


def parse_scope() -> list[tuple[str, str]]:
    raw = os.environ.get("MORRIS_POC_SCOPE", "").strip()
    if not raw:
        return list(WORKLOADS.keys())
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            sys.exit(f"bad MORRIS_POC_SCOPE entry {item!r}, expected <workload>:<sf>")
        wl, sf = item.split(":", 1)
        key = (wl.strip(), sf.strip())
        if key not in WORKLOADS:
            sys.exit(f"unknown MORRIS_POC_SCOPE entry {item!r}; valid: {sorted(WORKLOADS)}")
        out.append(key)
    return out


def resolve_parquet_paths(fixture_dir: Path, table: str) -> list[Path]:
    """Return the parquet file(s) for `table`. Handles sibling-split layouts
    like lineitem_0000.parquet + lineitem_0001.parquet in tpch sf100_optimized."""
    single = fixture_dir / f"{table}.parquet"
    if single.exists():
        return [single]
    siblings = sorted(fixture_dir.glob(f"{table}_*.parquet"))
    return siblings


ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1"
ICEBERG_AWS = "org.apache.iceberg:iceberg-aws-bundle:1.6.1"

spark = (
    SparkSession.builder
    .appName("morris-poc-iceberg-upload")
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

scope = parse_scope()
print(f"[plan] db={GLUE_DB} warehouse={WAREHOUSE} fixture_root={FIXTURE_ROOT}", flush=True)
print(f"[plan] scope={scope}", flush=True)

failures: list[tuple[str, str]] = []
for workload, sf in scope:
    rel_subdir, tables = WORKLOADS[(workload, sf)]
    fixture_dir = FIXTURE_ROOT / rel_subdir
    if not fixture_dir.is_dir():
        print(f"[skip] {workload}:{sf}: fixture dir missing: {fixture_dir}", flush=True)
        failures.append((f"{workload}:{sf}", f"fixture dir missing: {fixture_dir}"))
        continue
    print(f"[scope] {workload}:{sf} <- {fixture_dir}", flush=True)
    for table in tables:
        if ONLY and table not in ONLY:
            continue
        parquet_paths = resolve_parquet_paths(fixture_dir, table)
        if not parquet_paths:
            print(f"[skip] {workload}:{sf}:{table}: no parquet file", flush=True)
            failures.append((f"{workload}:{sf}:{table}", "parquet missing"))
            continue
        target = f"{CATALOG}.{GLUE_DB}.{workload}_{sf}_{table}"
        location = f"{WAREHOUSE.rstrip('/')}/{workload}/{sf}/{table}"
        # CTAS expects a single SELECT — use parquet.`...` with a glob for sibling-split,
        # else point at the single file directly so Spark inherits the parquet schema.
        if len(parquet_paths) == 1:
            select_source = f"parquet.`{parquet_paths[0]}`"
        else:
            glob_path = fixture_dir / f"{table}_*.parquet"
            select_source = f"parquet.`{glob_path}`"
        t0 = time.monotonic()
        try:
            print(f"[start] {target} <- {select_source} files={len(parquet_paths)}", flush=True)
            spark.sql(f"DROP TABLE IF EXISTS {target} PURGE")
            spark.sql(
                f"CREATE TABLE {target} "
                f"USING iceberg "
                f"LOCATION '{location}' "
                f"TBLPROPERTIES ('format-version'='2') "
                f"AS SELECT * FROM {select_source}"
            )
            rows = spark.sql(f"SELECT COUNT(*) AS c FROM {target}").collect()[0]["c"]
            elapsed = time.monotonic() - t0
            print(f"[done]  {target} rows={rows} elapsed={elapsed:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail]  {target}: {exc}", flush=True)
            failures.append((target, str(exc)))

spark.stop()

if failures:
    print(f"[summary] {len(failures)} failures:", flush=True)
    for t, e in failures:
        print(f"  - {t}: {e[:200]}", flush=True)
    sys.exit(1)
print("[summary] all tables uploaded", flush=True)
