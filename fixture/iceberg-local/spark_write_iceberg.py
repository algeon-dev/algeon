#!/usr/bin/env python3
"""PySpark writer for the local Iceberg REST + RustFS stack.

Reads pre-generated Parquet from a directory and writes each as an Iceberg
table through the REST catalog, so data files land in RustFS. Each source may
be either a flat `<source>.parquet` file or a `<source>/` directory of part files
(both layouts are read transparently by Spark). Table specs are either
`table` or `table=source`, where the latter writes Iceberg table `table` from
Parquet file/directory `source`.
This is the spec-correct write path (Iceberg reference implementation): DuckDB's
manifest map-type encoding is rejected by iceberg-rust 0.9.0's schema
deserializer, so the writer is PySpark while data generation stays on
tpchgen-cli / DuckDB tpcds.

Invoked inside the pinned Spark image via spark-submit (see load.sh). Catalog
config is passed as spark.sql.catalog.* conf; this script only takes the
    namespace, the input parquet dir, and the table spec list.

Naming contract (must match datafusion-nexus registration):
    catalog   : lake
    namespace : tpch_sf<SF> / tpcds_sf<SF>
    tables    : bare names
    warehouse : s3://lakehouse/warehouse
"""

import os
import sys

from pyspark.sql import SparkSession


def resolve_source(parquet_dir: str, table: str) -> str:
    """Pick the flat-file or directory-per-table layout, whichever exists.

    Spark reads a directory of part files as a single DataFrame, so the
    directory case just hands the table dir to spark.read.parquet.
    """
    flat = f"{parquet_dir}/{table}.parquet"
    nested = f"{parquet_dir}/{table}"
    if os.path.isfile(flat):
        return flat
    if os.path.isdir(nested):
        return nested
    raise FileNotFoundError(
        f"no parquet for table '{table}': tried {flat} and {nested}/"
    )


def parse_table_specs(raw: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for entry in (entry.strip() for entry in raw.split(",")):
        if not entry:
            continue
        if "=" in entry:
            table, source = (part.strip() for part in entry.split("=", 1))
        else:
            table = source = entry
        if not table or not source:
            raise ValueError("table specs must be table or table=source")
        specs.append((table, source))
    if not specs:
        raise ValueError("at least one table spec is required")
    return specs


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: spark_write_iceberg.py <namespace> <parquet_dir> <table>[=<source>][,<table>[=<source>]...]",
            file=sys.stderr,
        )
        return 2

    namespace = sys.argv[1]
    parquet_dir = sys.argv[2].rstrip("/")
    try:
        table_specs = parse_table_specs(sys.argv[3])
    except ValueError as exc:
        print(f"invalid table specs: {exc}", file=sys.stderr)
        return 2

    spark = SparkSession.builder.appName(f"iceberg-load-{namespace}").getOrCreate()
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS lake.{namespace}")

    for table, source in table_specs:
        src = resolve_source(parquet_dir, source)
        df = spark.read.parquet(src)
        fq = f"lake.{namespace}.{table}"
        # Recreate from scratch each load (ephemeral fixture).
        spark.sql(f"DROP TABLE IF EXISTS {fq}")
        df.writeTo(fq).using("iceberg").create()
        n = spark.table(fq).count()
        print(f"[spark] wrote {fq} from {source}: {n} rows", flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
