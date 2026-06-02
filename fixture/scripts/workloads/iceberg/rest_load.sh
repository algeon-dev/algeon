#!/usr/bin/env bash
# Load generated benchmark Parquet into the local Iceberg REST fixture + RustFS.

set -euo pipefail

WORKLOAD="${WORKLOAD:-tpch}"
SF="${SF:-1}"
REUSE_PARQUET=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_SCRIPT="$(cd "${SCRIPT_DIR}/../.." && pwd)/fixture.sh"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
LOCAL_ICEBERG_DIR="${REPO_ROOT}/fixture/iceberg-local"

SPARK_IMAGE="${SPARK_IMAGE:-apache/spark:3.5.3-scala2.12-java17-python3-ubuntu}"
ICEBERG_VERSION="${ICEBERG_VERSION:-1.9.2}"
DOCKER="${DOCKER:-docker}"
REST_ENDPOINT="${REST_ENDPOINT:-http://localhost:8181}"
S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:9000}"
WAREHOUSE="${WAREHOUSE:-s3://lakehouse/warehouse}"
S3_ACCESS_KEY="${RUSTFS_ACCESS_KEY:-cudfadmin}"
S3_SECRET_KEY="${RUSTFS_SECRET_KEY:-cudfadminsecret}"
S3_REGION="${S3_REGION:-us-east-1}"
DATA_ROOT="${DATA_ROOT:-${LOCAL_ICEBERG_DIR}/.loader-data}"
IVY_CACHE="${IVY_CACHE:-${LOCAL_ICEBERG_DIR}/.ivy-cache}"

TPCH_TABLES="customer,lineitem,nation,orders,part,partsupp,region,supplier"
TPCDS_TABLES="call_center,catalog_page,catalog_returns,catalog_sales,customer,customer_address,customer_demographics,date_dim,household_demographics,income_band,inventory,item,promotion,reason,ship_mode,store,store_returns,store_sales,time_dim,warehouse,web_page,web_returns,web_sales,web_site"

usage() {
  cat <<'USAGE'
Usage: rest_load.sh [options]

Options:
  --workload tpch|tpcds  Workload to load (default: tpch)
  --scale N|sfN          Scale factor (default: sf1)
  --reuse-parquet        Reuse existing generated Parquet under DATA_ROOT
  -h, --help             Show this help

Environment:
  DATA_ROOT, IVY_CACHE, DOCKER, SPARK_IMAGE, ICEBERG_VERSION, REST_ENDPOINT,
  S3_ENDPOINT, WAREHOUSE, RUSTFS_ACCESS_KEY, RUSTFS_SECRET_KEY, S3_REGION.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload)
      WORKLOAD="${2:?missing value for --workload}"
      shift 2
      ;;
    --scale|-s)
      SF="${2:?missing value for $1}"
      shift 2
      ;;
    --reuse-parquet)
      REUSE_PARQUET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SF="${SF#sf}"
SF="${SF#SF}"
case "${WORKLOAD}" in
  tpch) tables="${TPCH_TABLES}" ;;
  tpcds) tables="${TPCDS_TABLES}" ;;
  *) echo "error: unsupported --workload=${WORKLOAD}; expected tpch or tpcds" >&2; exit 2 ;;
esac

data_dir="${DATA_ROOT}/${WORKLOAD}_sf${SF}"
namespace="${WORKLOAD}_sf${SF}"
mkdir -p "${DATA_ROOT}" "${IVY_CACHE}"
chmod 0777 "${IVY_CACHE}"

if [[ "${REUSE_PARQUET}" == "1" && -d "${data_dir}" ]] && compgen -G "${data_dir}/*.parquet" >/dev/null; then
  echo "[iceberg-rest-load] reusing parquet -> ${data_dir}"
else
  echo "[iceberg-rest-load] generating ${WORKLOAD} sf${SF} parquet -> ${data_dir}"
  "${FIXTURE_SCRIPT}" "${WORKLOAD}" generate --scale "${SF}" --out "${data_dir}" --force
fi

pkgs="org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:${ICEBERG_VERSION},org.apache.iceberg:iceberg-aws-bundle:${ICEBERG_VERSION}"
echo "[iceberg-rest-load] PySpark writing ${namespace} tables into REST catalog"
# shellcheck disable=SC2086
${DOCKER} run --rm --net host \
  -v "${LOCAL_ICEBERG_DIR}/spark_write_iceberg.py:/app/spark_write_iceberg.py:ro" \
  -v "${data_dir}:/data:ro" \
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
  /app/spark_write_iceberg.py "${namespace}" /data "${tables}"

echo "[iceberg-rest-load] ${WORKLOAD} sf${SF} done"
