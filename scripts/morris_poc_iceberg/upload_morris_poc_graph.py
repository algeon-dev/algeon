"""Upload BloodHound-style graph Parquet fixtures into the `demo_db` Glue
database as Iceberg v2 tables.

Source layout (under MORRIS_POC_GRAPH_FIXTURE_ROOT, default <repo>/fixture/data):

    <workload>/bh_sf1/parquet/<table>.parquet

with `<workload>` in {github, ad_entra}.

Target Glue tables (folder-style S3 layout):

    demo_db.<workload>_sf1_<table>
    s3://example-warehouse/demo_db/<workload>/sf1/<table>/

Patterned after upload_demo_db_iceberg.py so the Spark/Glue/Iceberg setup
stays consistent across uploaders.

Env:
  MORRIS_POC_GLUE_DB              default demo_db
  MORRIS_POC_WAREHOUSE            default s3://example-warehouse/demo_db/
  MORRIS_POC_GRAPH_FIXTURE_ROOT   default <repo>/fixture/data
  MORRIS_POC_GRAPH_SCOPE          comma list of <workload>:<sf>, e.g. "github:sf1"
                                  (default: all combinations that have a fixture dir)
  MORRIS_POC_GRAPH_ONLY           optional comma list of bare table names to filter
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
FIXTURE_ROOT = Path(os.environ.get("MORRIS_POC_GRAPH_FIXTURE_ROOT", "fixture/data")).resolve()
ONLY = {t.strip() for t in os.environ.get("MORRIS_POC_GRAPH_ONLY", "").split(",") if t.strip()}
CATALOG = "glue_demo_db"

GITHUB_TABLES = [
    "bh_nodes", "bh_edges",
    "github_repositories", "github_branch_rules", "github_workflows",
    "github_secrets", "github_role_assignments",
]
AD_ENTRA_TABLES = [
    "bh_nodes", "bh_edges",
    "ad_users", "ad_groups", "ad_computers", "ad_sessions",
    "ad_aces", "ad_cert_templates",
    "azure_users", "azure_groups", "azure_apps",
    "azure_role_assignments", "azure_service_principals",
]

# (workload, sf) -> (fixture_subdir, tables)
WORKLOADS = {
    ("github",   "sf1"): ("github/bh_sf1/parquet",   GITHUB_TABLES),
    ("ad_entra", "sf1"): ("ad_entra/bh_sf1/parquet", AD_ENTRA_TABLES),
}


def parse_scope() -> list[tuple[str, str]]:
    raw = os.environ.get("MORRIS_POC_GRAPH_SCOPE", "").strip()
    if not raw:
        return list(WORKLOADS.keys())
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            sys.exit(f"bad MORRIS_POC_GRAPH_SCOPE entry {item!r}, expected <workload>:<sf>")
        wl, sf = item.split(":", 1)
        key = (wl.strip(), sf.strip())
        if key not in WORKLOADS:
            sys.exit(f"unknown MORRIS_POC_GRAPH_SCOPE entry {item!r}; valid: {sorted(WORKLOADS)}")
        out.append(key)
    return out


ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1"
ICEBERG_AWS = "org.apache.iceberg:iceberg-aws-bundle:1.6.1"

spark = (
    SparkSession.builder
    .appName("morris-poc-graph-iceberg-upload")
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
        parquet_path = fixture_dir / f"{table}.parquet"
        if not parquet_path.exists():
            print(f"[skip] {workload}:{sf}:{table}: no parquet file at {parquet_path}", flush=True)
            failures.append((f"{workload}:{sf}:{table}", "parquet missing"))
            continue
        target = f"{CATALOG}.{GLUE_DB}.{workload}_{sf}_{table}"
        location = f"{WAREHOUSE.rstrip('/')}/{workload}/{sf}/{table}"
        select_source = f"parquet.`{parquet_path}`"
        t0 = time.monotonic()
        try:
            print(f"[start] {target} <- {select_source}", flush=True)
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
