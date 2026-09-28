#!/usr/bin/env bash
# Load generated benchmark or demo Parquet into the local Iceberg REST fixture + RustFS.

set -euo pipefail

WORKLOAD="${WORKLOAD:-tpch}"
SF="${SF:-1}"
REUSE_PARQUET=0
LOAD_MODE="${LOAD_MODE:-create}"
RELOAD=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
FIXTURE_SCRIPT="${REPO_ROOT}/fixture/fixture.sh"
LOCAL_ICEBERG_DIR="${REPO_ROOT}/fixture/iceberg-local"

SPARK_IMAGE="${SPARK_IMAGE:-apache/spark:3.5.3-scala2.12-java17-python3-ubuntu}"
ICEBERG_VERSION="${ICEBERG_VERSION:-1.9.2}"
HADOOP_AWS_VERSION="${HADOOP_AWS_VERSION:-3.3.4}"
DOCKER="${DOCKER:-docker}"
REST_ENDPOINT="${REST_ENDPOINT:-http://localhost:8181}"
S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:9000}"
WAREHOUSE="${WAREHOUSE:-s3://lakehouse/warehouse}"
STAGING_PREFIX="${STAGING_PREFIX:-}"
S3_ACCESS_KEY="${RUSTFS_ACCESS_KEY:-algeonadmin}"
S3_SECRET_KEY="${RUSTFS_SECRET_KEY:-algeonadmin}"
S3_REGION="${S3_REGION:-us-east-1}"
if [[ "${S3_ENDPOINT}" == https://* ]]; then
  S3_SSL_ENABLED="${S3_SSL_ENABLED:-true}"
else
  S3_SSL_ENABLED="${S3_SSL_ENABLED:-false}"
fi
DATA_ROOT="${DATA_ROOT:-${LOCAL_ICEBERG_DIR}/.loader-data}"
IVY_CACHE="${IVY_CACHE:-${LOCAL_ICEBERG_DIR}/.ivy-cache}"
CITATION_NETWORK_DATA_DIR="${CITATION_NETWORK_DATA_DIR:-${REPO_ROOT}/fixture/demo/citation_network/v1/parquet}"

TPCH_TABLES="customer,lineitem,nation,orders,part,partsupp,region,supplier"
TPCDS_TABLES="call_center,catalog_page,catalog_returns,catalog_sales,customer,customer_address,customer_demographics,date_dim,household_demographics,income_band,inventory,item,promotion,reason,ship_mode,store,store_returns,store_sales,time_dim,warehouse,web_page,web_returns,web_sales,web_site"
CITATION_NETWORK_TABLES="citation_edges=edges_by_src,citation_edges_by_dst=edges_by_dst,papers=profiles_by_paper_id,paper_authors,paper_fos"

usage() {
  cat <<'USAGE'
Usage: iceberg_rest_load.sh [options]

Options:
  --workload tpch|tpcds|citation_network|equality_delete
                         Workload to load (default: tpch)
  --scale N|sfN          Scale factor for tpch/tpcds (default: sf1)
  --load-mode create|add-files|equality-delete
                         create rewrites tables through Spark; add-files
                         registers staged Parquet files in place (default: create)
  --reload               In add-files mode, drop existing tables before re-adding
  --reuse-parquet        Reuse existing generated Parquet under DATA_ROOT
  -h, --help             Show this help

Environment:
  DATA_ROOT, IVY_CACHE, DOCKER, SPARK_IMAGE, ICEBERG_VERSION, REST_ENDPOINT,
  S3_ENDPOINT, WAREHOUSE, RUSTFS_ACCESS_KEY, RUSTFS_SECRET_KEY, S3_REGION,
  CITATION_NETWORK_DATA_DIR, STAGING_PREFIX, HADOOP_AWS_VERSION.
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
    --load-mode)
      LOAD_MODE="${2:?missing value for --load-mode}"
      shift 2
      ;;
    --reload)
      RELOAD=1
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
  citation_network|citation-network) WORKLOAD="citation_network"; tables="${CITATION_NETWORK_TABLES}" ;;
  equality_delete|equality-delete) WORKLOAD="equality_delete"; tables="eq_deletes" ;;
  *) echo "error: unsupported --workload=${WORKLOAD}; expected tpch, tpcds, citation_network, or equality_delete" >&2; exit 2 ;;
esac

case "${LOAD_MODE}" in
  create|add-files|equality-delete) ;;
  *) echo "error: unsupported --load-mode=${LOAD_MODE}; expected create, add-files, or equality-delete" >&2; exit 2 ;;
esac

if [[ "${WORKLOAD}" == "equality_delete" ]]; then
  if [[ "${LOAD_MODE}" == "create" ]]; then
    LOAD_MODE="equality-delete"
  elif [[ "${LOAD_MODE}" != "equality-delete" ]]; then
    echo "error: equality_delete workload requires --load-mode equality-delete" >&2
    exit 2
  fi
fi

if [[ "${WORKLOAD}" == "citation_network" ]]; then
  data_dir="${CITATION_NETWORK_DATA_DIR}"
  namespace="citation_network"
elif [[ "${WORKLOAD}" == "equality_delete" ]]; then
  dataset_root="${DATA_ROOT}/${WORKLOAD}"
  data_dir="${dataset_root}/sf${SF}/parquet"
  namespace="${WORKLOAD}_sf${SF}"
else
  dataset_root="${DATA_ROOT}/${WORKLOAD}"
  data_dir="${dataset_root}/sf${SF}/parquet"
  namespace="${WORKLOAD}_sf${SF}"
fi
mkdir -p "${DATA_ROOT}" "${IVY_CACHE}"
chmod 0777 "${IVY_CACHE}"

if [[ "${WORKLOAD}" == "citation_network" ]]; then
  if [[ ! -d "${data_dir}" ]] || ! compgen -G "${data_dir}/*.parquet" >/dev/null; then
    echo "error: missing citation network parquet under ${data_dir}" >&2
    echo "       Download dblp.v12.json from the DBLP/AMiner Citation Network V12 distribution," >&2
    echo "       then run:" >&2
    echo "         python3 fixture/demo/citation_network/scripts/prepare.py --source-json /path/to/dblp.v12.json" >&2
    echo "       or set CITATION_NETWORK_DATA_DIR to an existing parquet directory." >&2
    exit 2
  fi
  echo "[iceberg-rest-load] using citation network parquet -> ${data_dir}"
elif [[ "${WORKLOAD}" == "equality_delete" ]]; then
  mkdir -p "${data_dir}"
  echo "[iceberg-rest-load] using generated equality-delete fixture"
elif [[ "${REUSE_PARQUET}" == "1" && -d "${data_dir}" ]] && compgen -G "${data_dir}/*.parquet" >/dev/null; then
  echo "[iceberg-rest-load] reusing parquet -> ${data_dir}"
else
  echo "[iceberg-rest-load] generating ${WORKLOAD} sf${SF} parquet -> ${data_dir}"
  "${FIXTURE_SCRIPT}" "${WORKLOAD}" generate --scale "${SF}" --variant raw --out-root "${dataset_root}" --force
fi

print_citation_network_acquisition_instructions() {
  echo "       Download dblp.v12.json from the DBLP/AMiner Citation Network V12 distribution," >&2
  echo "       then run:" >&2
  echo "         python3 fixture/demo/citation_network/scripts/prepare.py --source-json /path/to/dblp.v12.json" >&2
  echo "       or set CITATION_NETWORK_DATA_DIR to an existing parquet directory." >&2
}

aws_cli_for_loader() {
  ${DOCKER} run --rm --net host \
    -v "${data_dir}:/data:ro" \
    -e AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY}" \
    -e AWS_SECRET_ACCESS_KEY="${S3_SECRET_KEY}" \
    -e AWS_REGION="${S3_REGION}" \
    -e AWS_EC2_METADATA_DISABLED=true \
    --entrypoint aws \
    amazon/aws-cli:2.34.0 \
    --endpoint-url "${S3_ENDPOINT}" \
    "$@"
}

s3_uri_has_contents() {
  local uri="$1"
  if [[ "${uri}" == */ ]]; then
    aws_cli_for_loader s3 ls "${uri}" --recursive 2>/dev/null | grep -q .
  else
    aws_cli_for_loader s3 ls "${uri}" >/dev/null 2>&1
  fi
}

s3_uri_bucket() {
  local without_scheme="${1#s3://}"
  printf '%s\n' "${without_scheme%%/*}"
}

s3_uri_key() {
  local without_scheme="${1#s3://}"
  printf '%s\n' "${without_scheme#*/}"
}

s3_object_content_length() {
  local uri="$1"
  aws_cli_for_loader s3api head-object \
    --bucket "$(s3_uri_bucket "${uri}")" \
    --key "$(s3_uri_key "${uri}")" \
    --query ContentLength \
    --output text 2>/dev/null
}

put_empty_s3_object() {
  local uri="$1"
  aws_cli_for_loader s3api put-object \
    --bucket "$(s3_uri_bucket "${uri}")" \
    --key "$(s3_uri_key "${uri}")" \
    --body /dev/null >/dev/null
}

purge_rustfs_key_dir() {
  # RustFS 1.0.0-beta leaves an orphaned internal data directory behind when an
  # existing object is overwritten or deleted; the orphan then surfaces as a
  # phantom CommonPrefix that breaks S3A directory probing (Spark add_files /
  # spark.read.parquet fail with FileNotFoundException on a UUID child). Purge
  # the key directory inside the fixture container before re-uploading. Only
  # applies to the local compose stack; skip silently elsewhere.
  local uri="$1"
  local bucket key
  bucket="$(s3_uri_bucket "${uri}")"
  key="$(s3_uri_key "${uri}")"
  if ${DOCKER} ps --format '{{.Names}}' 2>/dev/null | grep -qx "cudf-iceberg-rustfs"; then
    ${DOCKER} exec cudf-iceberg-rustfs sh -c "rm -rf '/data/rustfs/${bucket}/${key}'" >/dev/null 2>&1 || true
  fi
}

stage_table_specs_for_add_files() {
  local specs="$1"
  local warehouse_without_scheme="${WAREHOUSE#s3://}"
  local warehouse_bucket="${warehouse_without_scheme%%/*}"
  local staging_prefix="${STAGING_PREFIX:-s3://${warehouse_bucket}/staging/${namespace}/v1}"
  local -a copy_args=()
  local -a missing_specs=()
  local -a staged_specs=()
  local -a table_entries=()
  local entry table source host_source container_source dest local_size marker remote_size

  if [[ -z "${warehouse_bucket}" || "${warehouse_bucket}" == "${WAREHOUSE}" ]]; then
    echo "error: WAREHOUSE must be an s3:// URI when --load-mode add-files is used (got ${WAREHOUSE})" >&2
    return 2
  fi

  IFS=',' read -ra table_entries <<<"${specs}"
  for entry in "${table_entries[@]}"; do
    table="${entry%%=*}"
    if [[ "${entry}" == *"="* ]]; then
      source="${entry#*=}"
    else
      source="${table}"
    fi
    table="${table//[[:space:]]/}"
    source="${source//[[:space:]]/}"
    if [[ -z "${table}" || -z "${source}" ]]; then
      echo "error: invalid table spec '${entry}'" >&2
      return 2
    fi
    if [[ ! -f "${data_dir}/${source}.parquet" && ! -d "${data_dir}/${source}" ]]; then
      missing_specs+=("${table}: expected ${data_dir}/${source}.parquet or ${data_dir}/${source}/")
    fi
  done
  if [[ "${#missing_specs[@]}" -gt 0 ]]; then
    echo "error: missing parquet inputs for ${WORKLOAD} under ${data_dir}" >&2
    printf '       %s\n' "${missing_specs[@]}" >&2
    if [[ "${WORKLOAD}" == "citation_network" ]]; then
      print_citation_network_acquisition_instructions
    fi
    return 2
  fi

  for entry in "${table_entries[@]}"; do
    table="${entry%%=*}"
    if [[ "${entry}" == *"="* ]]; then
      source="${entry#*=}"
    else
      source="${table}"
    fi
    table="${table//[[:space:]]/}"
    source="${source//[[:space:]]/}"
    if [[ -z "${table}" || -z "${source}" ]]; then
      echo "error: invalid table spec '${entry}'" >&2
      return 2
    fi

    if [[ -f "${data_dir}/${source}.parquet" ]]; then
      host_source="${data_dir}/${source}.parquet"
      container_source="/data/${source}.parquet"
      dest="${staging_prefix%/}/${table}.parquet"
      copy_args=("${container_source}" "${dest}")
      local_size="$(stat -c%s "${host_source}")"
      if remote_size="$(s3_object_content_length "${dest}")"; then
        if [[ "${RELOAD}" == "1" ]]; then
          echo "[iceberg-rest-load] refreshing staged parquet -> ${dest}"
          purge_rustfs_key_dir "${dest}"
          aws_cli_for_loader s3 cp "${copy_args[@]}"
        elif [[ "${remote_size}" == "${local_size}" ]]; then
          echo "[iceberg-rest-load] staging exists, leaving referenced file unchanged -> ${dest}"
        else
          echo "error: staged file exists with different size for ${table}: ${dest}" >&2
          echo "       existing=${remote_size} bytes local=${local_size} bytes; pass --reload to replace staged files and re-register" >&2
          return 2
        fi
      else
        echo "[iceberg-rest-load] staging ${host_source} -> ${dest}"
        aws_cli_for_loader s3 cp "${copy_args[@]}"
      fi
    elif [[ -d "${data_dir}/${source}" ]]; then
      host_source="${data_dir}/${source}"
      container_source="/data/${source}"
      dest="${staging_prefix%/}/${table}/"
      copy_args=("${container_source}/" "${dest}" "--recursive")
      marker="${dest%/}/_ALGEON_LOAD_COMPLETE"
      if [[ "${RELOAD}" == "1" ]]; then
        echo "[iceberg-rest-load] refreshing staged parquet prefix -> ${dest}"
        aws_cli_for_loader s3 rm "${dest}" --recursive >/dev/null
        purge_rustfs_key_dir "${dest}"
        aws_cli_for_loader s3 cp "${copy_args[@]}"
        put_empty_s3_object "${marker}"
      elif s3_object_content_length "${marker}" >/dev/null; then
        echo "[iceberg-rest-load] staging exists, leaving referenced prefix unchanged -> ${dest}"
      elif s3_uri_has_contents "${dest}"; then
        echo "error: staged prefix exists without completion marker for ${table}: ${dest}" >&2
        echo "       pass --reload to replace the partial prefix and re-register" >&2
        return 2
      else
        echo "[iceberg-rest-load] staging ${host_source} -> ${dest}"
        aws_cli_for_loader s3 cp "${copy_args[@]}"
        put_empty_s3_object "${marker}"
      fi
    else
      echo "error: no parquet for table '${table}': tried ${data_dir}/${source}.parquet and ${data_dir}/${source}/" >&2
      return 2
    fi

    # Three-part spec: schema comes from the local mount (no S3A round trip),
    # add_files registers the staged S3 objects.
    staged_specs+=("${table}=${source}=${dest}")
  done

  STAGED_TABLE_SPECS="$(IFS=','; echo "${staged_specs[*]}")"
}

spark_mode_args=("--mode" "${LOAD_MODE}")
if [[ "${RELOAD}" == "1" ]]; then
  spark_mode_args+=("--reload")
fi
spark_tables="${tables}"
if [[ "${LOAD_MODE}" == "add-files" ]]; then
  STAGED_TABLE_SPECS=""
  stage_table_specs_for_add_files "${tables}"
  spark_tables="${STAGED_TABLE_SPECS}"
fi

pkgs="org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:${ICEBERG_VERSION},org.apache.iceberg:iceberg-aws-bundle:${ICEBERG_VERSION},org.apache.hadoop:hadoop-aws:${HADOOP_AWS_VERSION}"
echo "[iceberg-rest-load] PySpark loading ${namespace} tables into REST catalog (${LOAD_MODE})"
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
  --conf spark.hadoop.fs.s3.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.hadoop.fs.s3a.endpoint="${S3_ENDPOINT}" \
  --conf spark.hadoop.fs.s3a.endpoint.region="${S3_REGION}" \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.connection.ssl.enabled="${S3_SSL_ENABLED}" \
  --conf spark.hadoop.fs.s3a.access.key="${S3_ACCESS_KEY}" \
  --conf spark.hadoop.fs.s3a.secret.key="${S3_SECRET_KEY}" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
  /app/spark_write_iceberg.py "${spark_mode_args[@]}" "${namespace}" /data "${spark_tables}"

if [[ "${WORKLOAD}" == "citation_network" || "${WORKLOAD}" == "equality_delete" ]]; then
  echo "[iceberg-rest-load] ${WORKLOAD} done"
else
  echo "[iceberg-rest-load] ${WORKLOAD} sf${SF} done"
fi
