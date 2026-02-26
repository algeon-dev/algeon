#!/usr/bin/env bash
# Load TPC-H and/or TPC-DS data into the local Iceberg REST + RustFS stack.
#
# Two stages, behind a stable "tables exist with the expected schema/names"
# contract:
#   1. Generate Parquet (one file per table) on the host:
#        TPC-H  -> tpchgen-cli
#        TPC-DS -> DuckDB tpcds extension (dsdgen + COPY ... TO parquet)
#   2. Write Iceberg tables via PySpark + iceberg-spark-runtime through the REST
#      catalog, so data files land in RustFS.
#
# The writer is PySpark (the Iceberg reference write path): DuckDB's manifest
# map-type encoding is rejected by iceberg-rust 0.9.0's schema deserializer, so
# DuckDB is used for TPC-DS *generation* only, not for the Iceberg write.
#
# Naming contract (must match datafusion-nexus registration):
#   catalog   : lake
#   namespace : tpch_sf<SF> / tpcds_sf<SF>   (scale carried by namespace)
#   tables    : bare names (lineitem, store_sales, ...)
#   warehouse : s3://lakehouse/warehouse
#
# Usage:
#   docker/iceberg-local/load.sh tpch 1
#   docker/iceberg-local/load.sh tpcds 1
#   WORKLOAD=tpch SF=10 docker/iceberg-local/load.sh
#
# Requirements on the host: tpchgen-cli (TPC-H gen), nerdctl/docker, network on
# first run (Spark downloads iceberg jars; DuckDB downloads the tpcds extension).
set -euo pipefail

WORKLOAD="${1:-${WORKLOAD:-tpch}}"
SF="${2:-${SF:-1}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- stack / credential parameters (override via env) ----------------------
DUCKDB_IMAGE="${DUCKDB_IMAGE:-datacatering/duckdb:v1.4.4}"
SPARK_IMAGE="${SPARK_IMAGE:-apache/spark:3.5.3-scala2.12-java17-python3-ubuntu}"
ICEBERG_VERSION="${ICEBERG_VERSION:-1.9.2}"
DOCKER="${DOCKER:-sudo nerdctl}"
# In-network names: Spark/DuckDB run with --net host, so use localhost.
REST_ENDPOINT="${REST_ENDPOINT:-http://localhost:8181}"
S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:9000}"
S3_ENDPOINT_HOSTPORT="${S3_ENDPOINT_HOSTPORT:-localhost:9000}"
WAREHOUSE="${WAREHOUSE:-s3://lakehouse/warehouse}"
S3_ACCESS_KEY="${RUSTFS_ACCESS_KEY:-cudfadmin}"
S3_SECRET_KEY="${RUSTFS_SECRET_KEY:-cudfadminsecret}"
S3_REGION="${S3_REGION:-us-east-1}"

# Persist generated parquet + spark ivy cache outside the temp dir so repeat
# runs do not re-download the iceberg jars.
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/.loader-data}"
IVY_CACHE="${IVY_CACHE:-${SCRIPT_DIR}/.ivy-cache}"
mkdir -p "${DATA_ROOT}" "${IVY_CACHE}"
# The Spark image runs as uid 185; the bind-mounted ivy cache must be writable
# by it so `--packages` can resolve the iceberg jars into the shared cache.
chmod 0777 "${IVY_CACHE}"

TPCH_TABLES="customer lineitem nation orders part partsupp region supplier"
TPCDS_TABLES="call_center catalog_page catalog_returns catalog_sales customer \
customer_address customer_demographics date_dim household_demographics \
income_band inventory item promotion reason ship_mode store store_returns \
store_sales time_dim warehouse web_page web_returns web_sales web_site"

# Runs a SQL script in the pinned DuckDB container on the host network.
# $1 = path to the .sql file; remaining args mount data dirs.
run_duckdb() {
  local sql_file="$1"; shift
  # shellcheck disable=SC2086
  ${DOCKER} run --rm --net host \
    -v "${sql_file}:/load.sql:ro" "$@" \
    "${DUCKDB_IMAGE}" -init /load.sql -no-stdin
}

# Writes the given comma-separated tables from a parquet dir into Iceberg via
# PySpark. $1 = namespace, $2 = host parquet dir, $3 = comma-separated tables.
spark_write() {
  local ns="$1" parquet_dir="$2" tables_csv="$3"
  local pkgs="org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:${ICEBERG_VERSION},org.apache.iceberg:iceberg-aws-bundle:${ICEBERG_VERSION}"
  echo "[load] PySpark writing ${ns} tables into Iceberg (REST -> RustFS)"
  # shellcheck disable=SC2086
  ${DOCKER} run --rm --net host \
    -v "${SCRIPT_DIR}/spark_write_iceberg.py:/app/spark_write_iceberg.py:ro" \
    -v "${parquet_dir}:/data:ro" \
    -v "${IVY_CACHE}:/opt/spark/.ivy2" \
    -e HOME=/opt/spark \
    --entrypoint /opt/spark/bin/spark-submit \
    "${SPARK_IMAGE}" \
    --packages "${pkgs}" \
    --driver-memory "${SPARK_DRIVER_MEMORY:-6g}" \
    --conf spark.jars.ivy=/opt/spark/.ivy2 \
    --conf spark.sql.files.maxPartitionBytes=67108864 \
    --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
    --conf spark.sql.catalog.lake=org.apache.iceberg.spark.SparkCatalog \
    --conf spark.sql.catalog.lake.type=rest \
    --conf spark.sql.catalog.lake.uri="${REST_ENDPOINT}" \
    --conf spark.sql.catalog.lake.warehouse="${WAREHOUSE}" \
    --conf spark.sql.catalog.lake.io-impl=org.apache.iceberg.aws.s3.S3FileIO \
    --conf spark.sql.catalog.lake.s3.endpoint="${S3_ENDPOINT}" \
    --conf spark.sql.catalog.lake.s3.path-style-access=true \
    --conf spark.sql.catalog.lake.s3.access-key-id="${S3_ACCESS_KEY}" \
    --conf spark.sql.catalog.lake.s3.secret-access-key="${S3_SECRET_KEY}" \
    --conf spark.sql.catalog.lake.client.region="${S3_REGION}" \
    /app/spark_write_iceberg.py "${ns}" /data "${tables_csv}"
}

load_tpch() {
  local ns="tpch_sf${SF}"
  local data_dir="${DATA_ROOT}/tpch_sf${SF}"
  rm -rf "${data_dir}"; mkdir -p "${data_dir}"
  echo "[load] generating TPC-H sf${SF} parquet via tpchgen-cli -> ${data_dir}"
  tpchgen-cli -s "${SF}" --format=parquet --output-dir="${data_dir}"

  local csv="${TPCH_TABLES// /,}"
  spark_write "${ns}" "${data_dir}" "${csv}"
  echo "[load] TPC-H sf${SF} done"
}

load_tpcds() {
  local ns="tpcds_sf${SF}"
  local data_dir="${DATA_ROOT}/tpcds_sf${SF}"
  rm -rf "${data_dir}"; mkdir -p "${data_dir}"

  # DuckDB generates TPC-DS in-process, then exports one parquet file per table.
  local sql="${data_dir}/export.sql"
  {
    echo "INSTALL tpcds; LOAD tpcds;"
    echo "CALL dsdgen(sf=${SF});"
    for t in ${TPCDS_TABLES}; do
      echo "COPY ${t} TO '/data/${t}.parquet' (FORMAT parquet);"
    done
  } > "${sql}"
  echo "[load] generating TPC-DS sf${SF} parquet via DuckDB tpcds -> ${data_dir}"
  run_duckdb "${sql}" -v "${data_dir}:/data"

  local csv="${TPCDS_TABLES// /,}"
  spark_write "${ns}" "${data_dir}" "${csv}"
  echo "[load] TPC-DS sf${SF} done"
}

case "${WORKLOAD}" in
  tpch)  load_tpch ;;
  tpcds) load_tpcds ;;
  *) echo "unknown workload '${WORKLOAD}' (expected tpch|tpcds)" >&2; exit 2 ;;
esac
