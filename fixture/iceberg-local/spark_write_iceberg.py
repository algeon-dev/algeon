#!/usr/bin/env python3
"""PySpark writer/registrar for the local Iceberg REST + RustFS stack.

Reads pre-generated Parquet from a directory and writes each as an Iceberg
table through the REST catalog, so data files land in RustFS. Each source may
be either a flat `<source>.parquet` file or a `<source>/` directory of part files
(both layouts are read transparently by Spark). Table specs are either
`table` or `table=source`, where the latter writes Iceberg table `table` from
Parquet file/directory `source`. In add-files mode a spec may also be
`table=schema_src=files_src`: the empty table is created from the local
`schema_src` Parquet schema while the staged `files_src` S3 objects are
registered with `system.add_files` (sources therefore cannot contain `=`).
The default `create` mode is the spec-correct write path (Iceberg reference
implementation): DuckDB's manifest map-type encoding is rejected by
iceberg-rust 0.9.0's schema deserializer, so the writer is PySpark while data
generation stays on tpchgen-cli / DuckDB tpcds.

The `add-files` mode expects each source to be an S3 URI for Parquet files
already staged in the object store. It creates an empty Iceberg table from the
Parquet schema, then registers those staged files with Iceberg's
`system.add_files` procedure.

The `equality-delete` mode creates tiny deterministic Iceberg v2 tables and
commits real equality-delete files through the Iceberg Java API. It is used by
the local REST E2E fixture rather than by benchmark loaders.

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
import subprocess
import sys
import tempfile
import textwrap


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


def parse_table_specs(raw: str) -> list[tuple[str, str, str | None]]:
    """Parse table specs into (table, source, files_source) triples.

    Accepted forms:
      table                     -> (table, table, None)
      table=source              -> (table, source, None)
      table=schema_src=files_src -> (table, schema_src, files_src)

    The three-part form is add-files-only: the schema is read from
    ``schema_src`` (a local path under the /data mount) while ``files_src`` (an
    S3 URI of staged Parquet) is what ``system.add_files`` registers.
    ``create_mode`` rejects it. Sources therefore cannot contain ``=``.
    """
    specs: list[tuple[str, str, str | None]] = []
    for entry in (entry.strip() for entry in raw.split(",")):
        if not entry:
            continue
        parts = [part.strip() for part in entry.split("=", 2)]
        if len(parts) == 1:
            table = source = parts[0]
            files_source = None
        elif len(parts) == 2:
            table, source = parts
            files_source = None
        elif len(parts) == 3:
            table, source, files_source = parts
            if not files_source:
                raise ValueError("table specs must be table, table=source, or table=schema_src=files_src")
        else:
            raise ValueError("table specs must be table, table=source, or table=schema_src=files_src")
        if not table or not source:
            raise ValueError("table specs must be table, table=source, or table=schema_src=files_src")
        specs.append((table, source, files_source))
    if not specs:
        raise ValueError("at least one table spec is required")
    return specs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("create", "add-files", "equality-delete"),
        default="create",
        help=(
            "create rewrites Iceberg data files; add-files registers staged Parquet files in place; "
            "equality-delete creates tiny v2 equality-delete fixture tables."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="In add-files mode, drop existing tables before registering staged files.",
    )
    parser.add_argument("namespace")
    parser.add_argument("parquet_dir")
    parser.add_argument(
        "tables",
        help=(
            "Comma-separated table or table=source specs; add-files mode also "
            "accepts table=schema_src=files_src (schema from the local /data "
            "mount, staged S3 URI registered via system.add_files)."
        ),
    )
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


def qualified_delete_files_table(namespace: str, table: str) -> str:
    return f"{qualified_table(namespace, table)}.delete_files"


def procedure_table_identifier(namespace: str, table: str) -> str:
    return f"{quote_identifier(namespace)}.{quote_identifier(table)}"


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


def table_delete_file_count(spark, namespace: str, table: str) -> int:
    rows = spark.sql(f"SELECT COUNT(*) AS delete_count FROM {qualified_delete_files_table(namespace, table)}").collect()
    if not rows:
        return 0
    row = rows[0]
    if hasattr(row, "delete_count"):
        return int(row.delete_count)
    return int(row[0])


def create_mode(
    spark,
    namespace: str,
    table_specs: list[tuple[str, str, str | None]],
    parquet_dir: str,
) -> None:
    for table, source, files_source in table_specs:
        if files_source is not None:
            raise ValueError(
                f"three-part spec '{table}={source}={files_source}' is add-files-only; "
                "create mode takes table or table=source"
            )
        src = resolve_source(parquet_dir, source)
        df = spark.read.parquet(src)
        fq = qualified_table(namespace, table)
        # Recreate from scratch each load (ephemeral fixture).
        spark.sql(f"DROP TABLE IF EXISTS {fq}")
        df.writeTo(fq).using("iceberg").tableProperty("format-version", "2").create()
        n = table_row_count(spark, namespace, table)
        print(f"[spark] wrote {fq} from {source}: {n} rows", flush=True)


def add_files_mode(
    spark,
    namespace: str,
    table_specs: list[tuple[str, str, str | None]],
    parquet_dir: str,
    reload: bool,
) -> None:
    for table, source, files_source in table_specs:
        schema_src = resolve_source(parquet_dir, source)
        src = files_source if files_source is not None else schema_src
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

        add_files_relation = parquet_relation_for_add_files(src)
        created_table = False
        try:
            # Schema-only create through the DataFrame API: a SQL
            # `SELECT * FROM parquet.`<path>`` direct file query is rejected
            # (UNSUPPORTED_DATASOURCE_FOR_DIRECT_QUERY) when
            # spark.sql.defaultCatalog is the Iceberg REST catalog. The schema
            # is read from schema_src (the local /data mount in three-part
            # specs) so the create never depends on S3A directory semantics.
            # Schema-only create; format-version 2 is the fixture minimum.
            spark.read.parquet(schema_src).limit(0).writeTo(fq).using("iceberg") \
                .tableProperty("format-version", "2").create()
            created_table = True
            result = spark.sql(
                "CALL lake.system.add_files("
                f"table => {quote_sql_string(procedure_table_identifier(namespace, table))}, "
                f"source_table => {quote_sql_string(add_files_relation)})"
            )
            for row in result.collect():
                print(f"[spark] add_files {fq}: {row}", flush=True)
            # Spark's add_files writes v1-format manifest files even into
            # format-version-2 tables. The fixture contract is Iceberg v2
            # minimum, so rewrite the manifests through the table's format
            # version (metadata-only, data files stay in place). A bare
            # rewrite_manifests no-ops on a single small manifest, so the
            # target size is pinned tiny for the call to force the rewrite.
            # The identity schema.name-mapping.default that add_files
            # generates is kept: datafusion-nexus admits identity mappings.
            spark.sql(
                f"ALTER TABLE {fq} SET TBLPROPERTIES ('commit.manifest.target-size-bytes'='1024')"
            )
            spark.sql(
                "CALL lake.system.rewrite_manifests("
                f"table => {quote_sql_string(procedure_table_identifier(namespace, table))})"
            )
            spark.sql(
                f"ALTER TABLE {fq} UNSET TBLPROPERTIES ('commit.manifest.target-size-bytes')"
            )
            n = table_row_count(spark, namespace, table)
        except Exception:
            if created_table:
                spark.sql(f"DROP TABLE IF EXISTS {fq}")
            raise
        print(f"[spark] registered {fq} from {src}: {n} rows", flush=True)


EQUALITY_DELETE_HELPER_SOURCE = r"""
import java.util.List;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.deletes.EqualityDeleteWriter;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.spark.Spark3Util;
import org.apache.spark.sql.SparkSession;

public final class NexusEqualityDeleteFixture {
  private NexusEqualityDeleteFixture() {}

  public static void commitIdEqualityDeletes(
      SparkSession spark, String tableIdentifier, String fileName, List<?> ids) throws Exception {
    Table table = Spark3Util.loadIcebergTable(spark, tableIdentifier);
    table.refresh();
    Schema deleteSchema = table.schema().select("id");
    String location = table.locationProvider().newDataLocation(fileName);
    OutputFile outputFile = table.io().newOutputFile(location);
    EqualityDeleteWriter<Record> writer = Parquet.writeDeletes(outputFile)
        .forTable(table)
        .rowSchema(deleteSchema)
        .createWriterFunc(GenericParquetWriter::create)
        .equalityFieldIds(1)
        .overwrite()
        .buildEqualityWriter();
    try {
      for (Object id : ids) {
        GenericRecord record = GenericRecord.create(deleteSchema);
        record.setField("id", Long.valueOf(((Number) id).longValue()));
        writer.write(record);
      }
    } finally {
      writer.close();
    }
    DeleteFile deleteFile = writer.toDeleteFile();
    if (!deleteFile.content().name().equals("EQUALITY_DELETES")) {
      throw new IllegalStateException("expected equality delete file, got " + deleteFile.content());
    }
    if (!deleteFile.equalityFieldIds().contains(Integer.valueOf(1))) {
      throw new IllegalStateException("expected equality field id 1, got " + deleteFile.equalityFieldIds());
    }
    table.newRowDelta().addDeletes(deleteFile).commit();
  }
}
"""


def compile_equality_delete_helper(spark):
    """Compile and load the tiny Java helper after Spark has resolved Iceberg jars."""
    helper_root = tempfile.mkdtemp(prefix="nexus-equality-delete-helper.")
    source_path = os.path.join(helper_root, "NexusEqualityDeleteFixture.java")
    classes_dir = os.path.join(helper_root, "classes")
    os.makedirs(classes_dir, exist_ok=True)
    with open(source_path, "w", encoding="utf-8") as source:
        source.write(textwrap.dedent(EQUALITY_DELETE_HELPER_SOURCE).strip())
        source.write("\n")

    classpath = f"{spark._jvm.java.lang.System.getProperty('java.class.path')}:/opt/spark/.ivy2/jars/*"
    subprocess.run(["javac", "-classpath", classpath, "-d", classes_dir, source_path], check=True)

    url = spark._jvm.java.io.File(classes_dir).toURI().toURL()
    urls = spark.sparkContext._gateway.new_array(spark._jvm.java.net.URL, 1)
    urls[0] = url
    parent = spark._jvm.java.lang.Thread.currentThread().getContextClassLoader()
    loader = spark._jvm.java.net.URLClassLoader.newInstance(urls, parent)
    spark._jvm.java.lang.Thread.currentThread().setContextClassLoader(loader)
    return spark._jvm.java.lang.Class.forName("NexusEqualityDeleteFixture", True, loader)


def invoke_static_helper(spark, helper_class, method_name: str, *args) -> None:
    method = None
    for candidate in helper_class.getMethods():
        if candidate.getName() == method_name:
            method = candidate
            break
    if method is None:
        raise RuntimeError(f"compiled helper is missing method {method_name}")

    gateway = spark.sparkContext._gateway
    java_args = gateway.new_array(spark._jvm.java.lang.Object, len(args))
    for index, value in enumerate(args):
        java_args[index] = value
    method.invoke(None, java_args)


def java_long_list(spark, values: list[int]):
    output = spark._jvm.java.util.ArrayList()
    for value in values:
        output.add(spark._jvm.java.lang.Long.valueOf(int(value)))
    return output


def equality_delete_mode(
    spark,
    namespace: str,
    table_specs: list[tuple[str, str, str | None]],
    parquet_dir: str,
) -> None:
    del parquet_dir
    helper_class = compile_equality_delete_helper(spark)
    for table, source, files_source in table_specs:
        if files_source is not None or source != table:
            raise ValueError("equality-delete mode takes plain table specs only")
        fq = qualified_table(namespace, table)
        spark.sql(f"DROP TABLE IF EXISTS {fq}")
        spark.sql(
            f"CREATE TABLE {fq} (id BIGINT, grp INT, payload STRING) "
            "USING iceberg TBLPROPERTIES ('format-version'='2')"
        )
        initial_rows = [
            (1, 10, "delete-id-1"),
            (2, 20, "delete-id-2"),
            (3, 10, "keep-a"),
            (4, 20, "keep-b"),
            (5, 30, "keep-c"),
        ]
        spark.createDataFrame(initial_rows, "id LONG, grp INT, payload STRING").writeTo(fq).append()

        table_identifier = f"lake.{namespace}.{table}"
        invoke_static_helper(
            spark,
            helper_class,
            "commitIdEqualityDeletes",
            spark._jsparkSession,
            table_identifier,
            f"eq-delete-{table}.parquet",
            java_long_list(spark, [1, 2]),
        )

        # Same equality key as a deleted row, but a newer data sequence number.
        # It must survive if sequence-number scoping is implemented correctly.
        spark.createDataFrame([(1, 40, "post-delete-keep")], "id LONG, grp INT, payload STRING").writeTo(fq).append()

        n = table_row_count(spark, namespace, table)
        delete_count = table_delete_file_count(spark, namespace, table)
        if n != 4:
            raise AssertionError(f"expected 4 visible rows after equality delete fixture load, got {n}")
        if delete_count != 1:
            raise AssertionError(f"expected exactly one live equality delete file, got {delete_count}")
        print(f"[spark] wrote equality-delete fixture {fq}: {n} visible rows, {delete_count} delete file", flush=True)


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
    elif args.mode == "add-files":
        add_files_mode(spark, namespace, table_specs, parquet_dir, args.reload)
    else:
        equality_delete_mode(spark, namespace, table_specs, parquet_dir)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
