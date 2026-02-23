"""Upload TPC-H Parquet fixtures into the `demo_db_delta` Glue database
as Delta Lake tables, with folder-style S3 layout:

    s3://example-warehouse/demo_db_delta/<workload>/<sf>/<table>/

Table naming inside demo_db_delta:

    demo_db_delta.<workload>_<sf>_<table>   e.g. demo_db_delta.tpch_sf10_lineitem

Source fixtures are the tpchgen-rs output produced by
`external-engine/test/tpch_performance/generate_tpch_data.sh`, laid out as
`<fixture_root>/tpch/sf<n>_optimized/parquet/<table>/part.N.parquet`.

Two-step process:
  1. Spark writes Delta files (parquet + _delta_log/) to s3a://...
  2. boto3 registers a Glue table pointing at the location, with the
     parameters Athena/Glue expect for Delta Lake tables.

Env:
  MORRIS_POC_DELTA_GLUE_DB     default demo_db_delta
  MORRIS_POC_DELTA_WAREHOUSE   default s3://example-warehouse/demo_db_delta/
  MORRIS_POC_DELTA_FIXTURE_ROOT  default <repo>/tests/data
  MORRIS_POC_DELTA_SCOPE       comma list of <workload>:<sf>, e.g. "tpch:sf1,tpch:sf10"
                               (default: tpch sf1 + sf10)
  MORRIS_POC_DELTA_ONLY        optional comma list of bare table names to filter
  MORRIS_POC_DELTA_CATALOG_ID  AWS account id (default 123456789012)
  MORRIS_POC_DELTA_REGION      default us-east-1
"""

import os
import sys
import time
from pathlib import Path

import boto3
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType, ByteType, DateType, DecimalType, DoubleType, FloatType,
    IntegerType, LongType, ShortType, StringType, TimestampType,
)

GLUE_DB = os.environ.get("MORRIS_POC_DELTA_GLUE_DB", "demo_db_delta")
WAREHOUSE = os.environ.get(
    "MORRIS_POC_DELTA_WAREHOUSE",
    "s3://example-warehouse/demo_db_delta/",
).rstrip("/") + "/"
FIXTURE_ROOT = Path(os.environ.get("MORRIS_POC_DELTA_FIXTURE_ROOT", "tests/data")).resolve()
ONLY = {t.strip() for t in os.environ.get("MORRIS_POC_DELTA_ONLY", "").split(",") if t.strip()}
CATALOG_ID = os.environ.get("MORRIS_POC_DELTA_CATALOG_ID", "123456789012")
REGION = os.environ.get("MORRIS_POC_DELTA_REGION", "us-east-1")

TPCH_TABLES = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier",
]

# (workload, sf) -> (fixture_subdir, tables)
WORKLOADS = {
    ("tpch", "sf1"):  ("tpch/sf1_optimized/parquet",  TPCH_TABLES),
    ("tpch", "sf10"): ("tpch/sf10_optimized/parquet", TPCH_TABLES),
}


def parse_scope() -> list[tuple[str, str]]:
    raw = os.environ.get("MORRIS_POC_DELTA_SCOPE", "").strip()
    if not raw:
        return list(WORKLOADS.keys())
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            sys.exit(f"bad MORRIS_POC_DELTA_SCOPE entry {item!r}, expected <workload>:<sf>")
        wl, sf = item.split(":", 1)
        key = (wl.strip(), sf.strip())
        if key not in WORKLOADS:
            sys.exit(f"unknown MORRIS_POC_DELTA_SCOPE entry {item!r}; valid: {sorted(WORKLOADS)}")
        out.append(key)
    return out


def resolve_table_dir(fixture_dir: Path, table: str) -> Path | None:
    candidate = fixture_dir / table
    if candidate.is_dir() and any(candidate.glob("*.parquet")):
        return candidate
    return None


def spark_to_hive_type(dtype) -> str:
    """Convert a Spark DataType to a Hive/Glue column type string."""
    if isinstance(dtype, DecimalType):
        return f"decimal({dtype.precision},{dtype.scale})"
    mapping = {
        BooleanType: "boolean",
        ByteType: "tinyint",
        ShortType: "smallint",
        IntegerType: "int",
        LongType: "bigint",
        FloatType: "float",
        DoubleType: "double",
        StringType: "string",
        DateType: "date",
        TimestampType: "timestamp",
    }
    for cls, name in mapping.items():
        if isinstance(dtype, cls):
            return name
    return dtype.simpleString()


DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.2.0"
HADOOP_AWS = "org.apache.hadoop:hadoop-aws:3.3.4"

spark = (
    SparkSession.builder
    .appName("morris-poc-delta-upload")
    .config("spark.jars.packages", f"{DELTA_PACKAGE},{HADOOP_AWS}")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    .config("spark.driver.memory", "8g")
    # tpchgen-rs writes parquet v2 with DELTA_LENGTH_BYTE_ARRAY encoding,
    # which Spark 3.5's vectorized reader cannot decode for string columns.
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

glue = boto3.client("glue", region_name=REGION)

# Ensure Glue DB exists.
try:
    glue.get_database(CatalogId=CATALOG_ID, Name=GLUE_DB)
    print(f"[glue] database {GLUE_DB} already exists", flush=True)
except glue.exceptions.EntityNotFoundException:
    glue.create_database(
        CatalogId=CATALOG_ID,
        DatabaseInput={
            "Name": GLUE_DB,
            "Description": "PoC database mirroring demo_db TPC-H content in Delta Lake format",
        },
    )
    print(f"[glue] created database {GLUE_DB}", flush=True)

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
        table_dir = resolve_table_dir(fixture_dir, table)
        if table_dir is None:
            print(f"[skip] {workload}:{sf}:{table}: no parquet dir under {fixture_dir}", flush=True)
            failures.append((f"{workload}:{sf}:{table}", "parquet missing"))
            continue

        target_table = f"{workload}_{sf}_{table}"
        location_s3 = f"{WAREHOUSE.rstrip('/')}/{workload}/{sf}/{table}"
        location_s3a = location_s3.replace("s3://", "s3a://", 1)

        t0 = time.monotonic()
        try:
            print(f"[start] {GLUE_DB}.{target_table} <- {table_dir} -> {location_s3}", flush=True)

            # 1. Read source parquet, write Delta to s3a.
            df = spark.read.parquet(str(table_dir))
            df.write.format("delta").mode("overwrite").save(location_s3a)
            rows_written = df.count()

            # 2. Register/replace Glue table pointing at the Delta location.
            columns = [
                {"Name": field.name, "Type": spark_to_hive_type(field.dataType)}
                for field in df.schema.fields
            ]
            table_input = {
                "Name": target_table,
                "TableType": "EXTERNAL_TABLE",
                "Parameters": {
                    "EXTERNAL": "TRUE",
                    "spark.sql.sources.provider": "delta",
                    "table_type": "delta",
                },
                "StorageDescriptor": {
                    "Columns": columns,
                    "Location": location_s3,
                    "InputFormat": "org.apache.hadoop.hive.ql.io.SymlinkTextInputFormat",
                    "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                    "SerdeInfo": {
                        "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                    },
                },
            }
            try:
                glue.get_table(CatalogId=CATALOG_ID, DatabaseName=GLUE_DB, Name=target_table)
                glue.update_table(CatalogId=CATALOG_ID, DatabaseName=GLUE_DB, TableInput=table_input)
            except glue.exceptions.EntityNotFoundException:
                glue.create_table(CatalogId=CATALOG_ID, DatabaseName=GLUE_DB, TableInput=table_input)

            elapsed = time.monotonic() - t0
            print(f"[done]  {GLUE_DB}.{target_table} rows={rows_written} elapsed={elapsed:.1f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail]  {GLUE_DB}.{target_table}: {exc}", flush=True)
            failures.append((f"{GLUE_DB}.{target_table}", str(exc)))

spark.stop()

if failures:
    print(f"[summary] {len(failures)} failures:", flush=True)
    for t, e in failures:
        print(f"  - {t}: {e[:200]}", flush=True)
    sys.exit(1)
print("[summary] all tables uploaded", flush=True)
