#!/usr/bin/env python3
"""PySpark writer/registrar for the local Iceberg REST + RustFS stack.

Reads pre-generated Parquet from a directory and writes each as an Iceberg
table through the REST catalog, so data files land in RustFS. Each source may
be either a flat `<source>.parquet` file or a `<source>/` directory of part files
(both layouts are read transparently by Spark). Table specs are either
`table` or `table=source`, where the latter writes Iceberg table `table` from
Parquet file/directory `source`.
The default `create` mode is the spec-correct write path (Iceberg reference
implementation): DuckDB's manifest map-type encoding is rejected by
iceberg-rust 0.9.0's schema deserializer, so the writer is PySpark while data
generation stays on tpchgen-cli / DuckDB tpcds.

The `add-files` mode expects each source to be an S3 URI for Parquet files
already staged in the object store. It creates an empty Iceberg table from the
Parquet schema, then registers those staged files with Iceberg's
`system.add_files` procedure.

Invoked inside the pinned Spark image via spark-submit (see load.sh). Catalog
config is passed as spark.sql.catalog.* conf; this script only takes the
    namespace, the input parquet dir, and the table spec list.

Naming contract (must match datafusion-nexus registration):
    catalog   : lake
    namespace : tpch_sf<SF> / tpcds_sf<SF>
    tables    : bare names
    warehouse : s3://lakehouse/warehouse
"""

import argparse
import os
import sys


def resolve_source(parquet_dir: str, table: str) -> str:
    """Pick the flat-file or directory-per-table layout, whichever exists.

    Spark reads a directory of part files as a single DataFrame, so the
    directory case just hands the table dir to spark.read.parquet.
    """
    if table.startswith("s3://"):
        return table.rstrip("/")
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("create", "add-files"),
        default="create",
        help="create rewrites Iceberg data files; add-files registers staged Parquet files in place.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="In add-files mode, drop existing tables before registering staged files.",
    )
    parser.add_argument("namespace")
    parser.add_argument("parquet_dir")
    parser.add_argument("tables", help="Comma-separated table or table=source specs.")
    return parser.parse_args(argv)


def quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


def quote_path_identifier(value: str) -> str:
    return quote_identifier(value)


def qualified_namespace(namespace: str) -> str:
    return f"{quote_identifier('lake')}.{quote_identifier(namespace)}"


def qualified_table(namespace: str, table: str) -> str:
    return f"{qualified_namespace(namespace)}.{quote_identifier(table)}"


def qualified_files_table(namespace: str, table: str) -> str:
    return f"{qualified_table(namespace, table)}.files"


def procedure_table_identifier(namespace: str, table: str) -> str:
    return f"{quote_identifier(namespace)}.{quote_identifier(table)}"


def parquet_relation_for_select(source: str) -> str:
    return f"parquet.{quote_path_identifier(source)}"


def parquet_relation_for_add_files(source: str) -> str:
    return f"`parquet`.{quote_path_identifier(source)}"


def table_exists(spark, namespace: str, table: str) -> bool:
    rows = spark.sql(f"SHOW TABLES IN {qualified_namespace(namespace)} LIKE {quote_sql_string(table)}").collect()
    return any(row.tableName == table for row in rows)


def table_data_file_count(spark, namespace: str, table: str) -> int:
    rows = spark.sql(f"SELECT COUNT(*) AS file_count FROM {qualified_files_table(namespace, table)}").collect()
    if not rows:
        return 0
    row = rows[0]
    if hasattr(row, "file_count"):
        return int(row.file_count)
    return int(row[0])


def table_row_count(spark, namespace: str, table: str) -> int:
    rows = spark.sql(f"SELECT COUNT(*) AS row_count FROM {qualified_table(namespace, table)}").collect()
    if not rows:
        return 0
    row = rows[0]
    if hasattr(row, "row_count"):
        return int(row.row_count)
    return int(row[0])


def create_mode(spark, namespace: str, table_specs: list[tuple[str, str]], parquet_dir: str) -> None:
    for table, source in table_specs:
        src = resolve_source(parquet_dir, source)
        df = spark.read.parquet(src)
        fq = qualified_table(namespace, table)
        # Recreate from scratch each load (ephemeral fixture).
        spark.sql(f"DROP TABLE IF EXISTS {fq}")
        df.writeTo(fq).using("iceberg").create()
        n = table_row_count(spark, namespace, table)
        print(f"[spark] wrote {fq} from {source}: {n} rows", flush=True)


def add_files_mode(
    spark,
    namespace: str,
    table_specs: list[tuple[str, str]],
    parquet_dir: str,
    reload: bool,
) -> None:
    for table, source in table_specs:
        src = resolve_source(parquet_dir, source)
        fq = qualified_table(namespace, table)
        if reload:
            spark.sql(f"DROP TABLE IF EXISTS {fq}")
        elif table_exists(spark, namespace, table):
            file_count = table_data_file_count(spark, namespace, table)
            if file_count > 0:
                print(f"[spark] skipped existing {fq}; pass --reload to re-register staged files", flush=True)
                continue
            print(f"[spark] replacing empty existing {fq}; previous add_files likely did not complete", flush=True)
            spark.sql(f"DROP TABLE IF EXISTS {fq}")

        select_relation = parquet_relation_for_select(src)
        add_files_relation = parquet_relation_for_add_files(src)
        created_table = False
        try:
            spark.sql(f"CREATE TABLE {fq} USING iceberg AS SELECT * FROM {select_relation} LIMIT 0")
            created_table = True
            result = spark.sql(
                "CALL lake.system.add_files("
                f"table => {quote_sql_string(procedure_table_identifier(namespace, table))}, "
                f"source_table => {quote_sql_string(add_files_relation)})"
            )
            for row in result.collect():
                print(f"[spark] add_files {fq}: {row}", flush=True)
            n = table_row_count(spark, namespace, table)
        except Exception:
            if created_table:
                spark.sql(f"DROP TABLE IF EXISTS {fq}")
            raise
        print(f"[spark] registered {fq} from {src}: {n} rows", flush=True)


def main() -> int:
    args = parse_args(sys.argv[1:])

    try:
        table_specs = parse_table_specs(args.tables)
    except ValueError as exc:
        print(f"invalid table specs: {exc}", file=sys.stderr)
        return 2

    from pyspark.sql import SparkSession

    namespace = args.namespace
    parquet_dir = args.parquet_dir.rstrip("/")
    spark = SparkSession.builder.appName(f"iceberg-load-{namespace}").getOrCreate()
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {qualified_namespace(namespace)}")

    if args.mode == "create":
        create_mode(spark, namespace, table_specs, parquet_dir)
    else:
        add_files_mode(spark, namespace, table_specs, parquet_dir, args.reload)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
