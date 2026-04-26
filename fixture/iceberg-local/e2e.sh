#!/usr/bin/env bash
# One-shot developer entrypoint for explicit local E2E suites.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
COMPOSE_FILE="${REPO_ROOT}/fixture/iceberg-local/docker-compose.yml"
FIXTURE_SCRIPT="${REPO_ROOT}/fixture/fixture.sh"

DOCKER="${DOCKER:-docker}"
DEFAULT_REST_URI="http://localhost:8181"
DEFAULT_GLUE_URI="http://localhost:5001"
DEFAULT_S3_ENDPOINT="http://localhost:9000"
WORKLOAD="tpch"
SCALE="1"
REUSE_PARQUET=0
NO_LOAD=0
SKIP_STACK=0
LOAD_GLUE=0
RUN_REST=1
RUN_GLUE=1
RUN_SERVER=1
RUN_REMOTE_DIRECT=0
RUN_SERVER_REST_MULTIGPU=0
GPU_PARITY=0

REST_URI="${NEXUS_ICEBERG_REST_URI:-${DEFAULT_REST_URI}}"
GLUE_URI="${NEXUS_ICEBERG_GLUE_URI:-${DEFAULT_GLUE_URI}}"
WAREHOUSE="${NEXUS_ICEBERG_WAREHOUSE:-s3://lakehouse/warehouse}"
S3_ENDPOINT="${NEXUS_ICEBERG_S3_ENDPOINT:-${DEFAULT_S3_ENDPOINT}}"
S3_REGION="${NEXUS_ICEBERG_S3_REGION:-us-east-1}"
S3_ACCESS_KEY="${NEXUS_ICEBERG_S3_ACCESS_KEY_ID:-${RUSTFS_ACCESS_KEY:-nexusadmin}}"
S3_SECRET_KEY="${NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY:-${RUSTFS_SECRET_KEY:-nexusadmin}}"
MOTO_ACCESS_KEY="${MOTO_ACCESS_KEY:-moto-access}"
MOTO_SECRET_KEY="${MOTO_SECRET_KEY:-moto-secret}"
WAREHOUSE_BUCKET=""

REST_TABLE="${NEXUS_ICEBERG_REST_E2E_TABLE:-nation}"
REST_GROUP_COL="${NEXUS_ICEBERG_REST_E2E_GROUP_COL:-n_regionkey}"
REST_EXPECTED_ROWS="${NEXUS_ICEBERG_REST_E2E_EXPECTED_ROWS:-}"

usage() {
  cat <<'USAGE'
Usage:
  fixture/iceberg-local/e2e.sh iceberg-local [options]

Runs the local Iceberg REST + Glue/Moto E2E suite through
nexus-tools. The suite is explicit Cargo run tooling; it is not part
of cargo test --all-features.

Options:
  --workload tpch|equality_delete
                         Workload loaded for REST E2E (default: tpch)
  --scale N|sfN          Scale factor loaded for REST E2E (default: sf1)
  --reuse-parquet        Reuse generated Parquet under fixture/iceberg-local/.loader-data
  --no-load              Do not run the REST loader before the E2E runner
  --load-glue            Also load benchmark tables through Moto Glue
  --skip-stack           Do not run docker compose up -d; still wait for endpoints
  --rest-only            Run only the REST catalog E2E runner
  --glue-only            Run only the Glue/Moto backend E2E runner
  --server-only          Run only the server + Glue/Moto E2E runner
  --no-server            Run REST and Glue/Moto backend runners, but not server E2E
  --remote-direct        Also run the credentialless Glue/S3 remote-direct server
                         E2E (GPU required; serialized via flock /tmp/cudf-gpu.lock;
                         loads Glue tables unless --no-load)
  --remote-direct-only   Run only the credentialless remote-direct server E2E
  --server-rest-multigpu Run the production release Flight SQL acceptance with
                         one Tokio worker and GPUs 0,1,2,3 against REST/RustFS
  --server-rest-multigpu-only
                         Run only that production REST/RustFS multi-GPU acceptance
  --gpu-parity           Enable NEXUS_ICEBERG_REST_E2E_GPU_PARITY=1 for REST E2E
  --rest-table NAME      REST table used for scan/planning checks (default: nation)
  --rest-group-col NAME  REST group column for planning checks (default: n_regionkey)
  --expected-rows N      Expected REST table row count (default: 25 for tpch nation)
  -h, --help             Show this help

Environment overrides:
  DOCKER, NEXUS_ICEBERG_REST_URI,
  NEXUS_ICEBERG_GLUE_URI, NEXUS_ICEBERG_WAREHOUSE,
  NEXUS_ICEBERG_S3_ENDPOINT,
  NEXUS_ICEBERG_S3_REGION,
  NEXUS_ICEBERG_S3_ACCESS_KEY_ID,
  NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY.

Endpoint overrides must be loopback http:// URLs. Custom endpoint ports require
--skip-stack because fixture/iceberg-local/docker-compose.yml publishes fixed
host ports.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

step() {
  echo
  echo "==> $*"
}

wait_url() {
  local name="$1"
  local url="$2"
  local fail_on_http_error="${3:-1}"

  step "waiting for ${name} (${url})"
  for _ in $(seq 1 60); do
    if [[ "${fail_on_http_error}" == "1" ]]; then
      if curl --fail --silent --max-time 2 "${url}" >/dev/null; then
        return 0
      fi
    else
      if curl --silent --max-time 2 "${url}" >/dev/null; then
        return 0
      fi
    fi
    sleep 1
  done

  echo "docker compose status:" >&2
  "${DOCKER}" compose -f "${COMPOSE_FILE}" ps >&2 || true
  die "${name} did not become ready at ${url}"
}

wait_s3_endpoint() {
  local name="$1"
  local url="$2"
  local response

  step "waiting for ${name} (${url})"
  for _ in $(seq 1 60); do
    if response="$(curl --include --silent --max-time 2 "${url}" 2>/dev/null)" &&
      (grep -q '<Code>AccessDenied</Code>' <<<"${response}" ||
        grep -q '<ListAllMyBucketsResult' <<<"${response}"); then
      return 0
    fi
    sleep 1
  done

  echo "docker compose status:" >&2
  "${DOCKER}" compose -f "${COMPOSE_FILE}" ps >&2 || true
  die "${name} did not look like an S3-compatible endpoint at ${url}"
}

validate_s3_bucket() {
  step "validating RustFS credentials and ${WAREHOUSE_BUCKET} bucket"
  for _ in $(seq 1 12); do
    if env \
      AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY}" \
      AWS_SECRET_ACCESS_KEY="${S3_SECRET_KEY}" \
      AWS_REGION="${S3_REGION}" \
      "${DOCKER}" run --rm --net host \
        -e AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY \
        -e AWS_REGION \
        amazon/aws-cli:2.34.0 \
        --endpoint-url "${S3_ENDPOINT}" \
        s3 ls "s3://${WAREHOUSE_BUCKET}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  die "could not access s3://${WAREHOUSE_BUCKET} at ${S3_ENDPOINT} with the configured local RustFS credentials"
}

require_loopback_http_url() {
  local name="$1"
  local value="$2"
  local lower
  lower="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"

  if [[ "${lower}" =~ ^http://localhost(:[0-9]+)?(/.*)?$ ]] ||
    [[ "${lower}" =~ ^http://127(\.[0-9]{1,3}){3}(:[0-9]+)?(/.*)?$ ]] ||
    [[ "${lower}" =~ ^http://\[::1\](:[0-9]+)?(/.*)?$ ]]; then
    return 0
  fi

  die "${name} must be a loopback http:// URL for iceberg-local, got '${value}'"
}

has_custom_endpoint_override() {
  [[ "${REST_URI}" != "${DEFAULT_REST_URI}" ]] ||
    [[ "${GLUE_URI}" != "${DEFAULT_GLUE_URI}" ]] ||
    [[ "${S3_ENDPOINT}" != "${DEFAULT_S3_ENDPOINT}" ]]
}

warehouse_bucket() {
  local warehouse="$1"
  local rest

  if [[ "${warehouse}" != s3://* ]]; then
    die "NEXUS_ICEBERG_WAREHOUSE must be an s3:// URI for iceberg-local, got '${warehouse}'"
  fi
  rest="${warehouse#s3://}"
  rest="${rest%%/*}"
  if [[ -z "${rest}" ]]; then
    die "NEXUS_ICEBERG_WAREHOUSE must include a bucket, got '${warehouse}'"
  fi
  printf '%s\n' "${rest}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

suite="${1:-}"
case "${suite}" in
  iceberg-local) ;;
  embedded-backend)
    shift
    echo "warning: fixture/iceberg-local/e2e.sh embedded-backend moved to crates/nexus-examples/scripts/e2e.sh" >&2
    exec "${REPO_ROOT}/crates/nexus-examples/scripts/e2e.sh" "$@"
    ;;
  *)
    usage >&2
    die "unsupported e2e suite '${suite:-}' (expected iceberg-local)"
    ;;
esac
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload)
      WORKLOAD="${2:?missing value for --workload}"
      shift 2
      ;;
    --scale|-s)
      SCALE="${2:?missing value for $1}"
      shift 2
      ;;
    --reuse-parquet)
      REUSE_PARQUET=1
      shift
      ;;
    --no-load)
      NO_LOAD=1
      shift
      ;;
    --load-glue)
      LOAD_GLUE=1
      shift
      ;;
    --skip-stack)
      SKIP_STACK=1
      shift
      ;;
    --rest-only)
      RUN_REST=1
      RUN_GLUE=0
      RUN_SERVER=0
      RUN_REMOTE_DIRECT=0
      RUN_SERVER_REST_MULTIGPU=0
      shift
      ;;
    --glue-only)
      RUN_REST=0
      RUN_GLUE=1
      RUN_SERVER=0
      RUN_REMOTE_DIRECT=0
      RUN_SERVER_REST_MULTIGPU=0
      shift
      ;;
    --server-only)
      RUN_REST=0
      RUN_GLUE=0
      RUN_SERVER=1
      RUN_REMOTE_DIRECT=0
      RUN_SERVER_REST_MULTIGPU=0
      shift
      ;;
    --no-server)
      RUN_SERVER=0
      shift
      ;;
    --remote-direct)
      RUN_REMOTE_DIRECT=1
      shift
      ;;
    --remote-direct-only)
      RUN_REST=0
      RUN_GLUE=0
      RUN_SERVER=0
      RUN_REMOTE_DIRECT=1
      RUN_SERVER_REST_MULTIGPU=0
      shift
      ;;
    --server-rest-multigpu)
      RUN_SERVER_REST_MULTIGPU=1
      shift
      ;;
    --server-rest-multigpu-only)
      RUN_REST=0
      RUN_GLUE=0
      RUN_SERVER=0
      RUN_REMOTE_DIRECT=0
      RUN_SERVER_REST_MULTIGPU=1
      shift
      ;;
    --gpu-parity)
      GPU_PARITY=1
      shift
      ;;
    --rest-table)
      REST_TABLE="${2:?missing value for --rest-table}"
      shift 2
      ;;
    --rest-group-col)
      REST_GROUP_COL="${2:?missing value for --rest-group-col}"
      shift 2
      ;;
    --expected-rows)
      REST_EXPECTED_ROWS="${2:?missing value for --expected-rows}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

SCALE="${SCALE#sf}"
SCALE="${SCALE#SF}"
if [[ ! "${SCALE}" =~ ^[0-9]+$ ]]; then
  die "--scale must be an integer or sfN value, got '${SCALE}'"
fi
case "${WORKLOAD}" in
  tpch) ;;
  equality_delete|equality-delete) WORKLOAD="equality_delete" ;;
  *)
    if [[ "${RUN_REST}" == "1" ]]; then
      die "iceberg-local REST E2E currently supports --workload tpch or equality_delete"
    fi
    ;;
esac

if [[ "${WORKLOAD}" == "equality_delete" ]]; then
  if [[ "${RUN_REST}" != "1" ]]; then
    die "equality_delete workload requires the REST E2E runner"
  fi
  RUN_GLUE=0
  RUN_SERVER=0
  RUN_REMOTE_DIRECT=0
  RUN_SERVER_REST_MULTIGPU=0
  LOAD_GLUE=0
  if [[ "${REST_TABLE}" == "nation" && -z "${NEXUS_ICEBERG_REST_E2E_TABLE:-}" ]]; then
    REST_TABLE="eq_deletes"
  fi
  if [[ "${REST_GROUP_COL}" == "n_regionkey" && -z "${NEXUS_ICEBERG_REST_E2E_GROUP_COL:-}" ]]; then
    REST_GROUP_COL="grp"
  fi
  if [[ -z "${REST_EXPECTED_ROWS}" ]]; then
    REST_EXPECTED_ROWS="4"
  fi
fi

if [[ "${RUN_REST}" != "1" && "${RUN_GLUE}" != "1" && "${RUN_SERVER}" != "1" && "${RUN_REMOTE_DIRECT}" != "1" && "${RUN_SERVER_REST_MULTIGPU}" != "1" ]]; then
  die "no E2E runners selected"
fi

if [[ "${RUN_SERVER_REST_MULTIGPU}" == "1" && ("${WORKLOAD}" != "tpch" || "${SCALE}" != "1") ]]; then
  die "--server-rest-multigpu requires --workload tpch --scale 1"
fi

# The remote-direct runner queries a loaded data-bearing Glue namespace.
if [[ "${RUN_REMOTE_DIRECT}" == "1" && "${NO_LOAD}" != "1" ]]; then
  LOAD_GLUE=1
fi

require_loopback_http_url "NEXUS_ICEBERG_REST_URI" "${REST_URI}"
require_loopback_http_url "NEXUS_ICEBERG_GLUE_URI" "${GLUE_URI}"
require_loopback_http_url "NEXUS_ICEBERG_S3_ENDPOINT" "${S3_ENDPOINT}"
WAREHOUSE_BUCKET="$(warehouse_bucket "${WAREHOUSE}")"

if [[ "${SKIP_STACK}" != "1" ]] && has_custom_endpoint_override; then
  die "custom local endpoint overrides require --skip-stack because the compose file publishes fixed host ports"
fi

if [[ "${RUN_REST}" == "1" && "${REST_TABLE}" == "nation" && -z "${REST_EXPECTED_ROWS}" ]]; then
  REST_EXPECTED_ROWS="25"
fi
if [[ -n "${REST_EXPECTED_ROWS}" && ! "${REST_EXPECTED_ROWS}" =~ ^[0-9]+$ ]]; then
  die "--expected-rows must be an integer, got '${REST_EXPECTED_ROWS}'"
fi

if [[ "${SKIP_STACK}" != "1" ]]; then
  step "starting local Iceberg stack"
  if ! env RUSTFS_ACCESS_KEY="${S3_ACCESS_KEY}" RUSTFS_SECRET_KEY="${S3_SECRET_KEY}" \
    "${DOCKER}" compose -f "${COMPOSE_FILE}" up -d; then
    echo "warning: docker compose up failed; checking whether the local endpoints are already available" >&2
  fi
fi

wait_s3_endpoint "RustFS/S3" "${S3_ENDPOINT}/"
validate_s3_bucket
if [[ "${RUN_REST}" == "1" || "${RUN_SERVER_REST_MULTIGPU}" == "1" ]]; then
  wait_url "Iceberg REST" "${REST_URI}/v1/config" 1
fi
if [[ "${RUN_GLUE}" == "1" || "${RUN_SERVER}" == "1" || "${RUN_REMOTE_DIRECT}" == "1" || "${LOAD_GLUE}" == "1" ]]; then
  wait_url "Moto Glue" "${GLUE_URI}/moto-api/" 1
fi

load_args=(--workload "${WORKLOAD}" --scale "${SCALE}")
if [[ "${REUSE_PARQUET}" == "1" ]]; then
  load_args+=(--reuse-parquet)
fi

if [[ ("${RUN_REST}" == "1" || "${RUN_SERVER_REST_MULTIGPU}" == "1") && "${NO_LOAD}" != "1" ]]; then
  step "loading ${WORKLOAD} sf${SCALE} into local Iceberg REST catalog"
  env \
    REST_ENDPOINT="${REST_URI}" \
    S3_ENDPOINT="${S3_ENDPOINT}" \
    WAREHOUSE="${WAREHOUSE}" \
    S3_REGION="${S3_REGION}" \
    RUSTFS_ACCESS_KEY="${S3_ACCESS_KEY}" \
    RUSTFS_SECRET_KEY="${S3_SECRET_KEY}" \
    "${FIXTURE_SCRIPT}" iceberg rest load "${load_args[@]}"
fi

if [[ "${LOAD_GLUE}" == "1" && "${NO_LOAD}" != "1" ]]; then
  step "loading ${WORKLOAD} sf${SCALE} into local Moto Glue catalog"
  env \
    GLUE_ENDPOINT="${GLUE_URI}" \
    S3_ENDPOINT="${S3_ENDPOINT}" \
    WAREHOUSE="${WAREHOUSE}" \
    S3_REGION="${S3_REGION}" \
    RUSTFS_ACCESS_KEY="${S3_ACCESS_KEY}" \
    RUSTFS_SECRET_KEY="${S3_SECRET_KEY}" \
    MOTO_ACCESS_KEY="${MOTO_ACCESS_KEY}" \
    MOTO_SECRET_KEY="${MOTO_SECRET_KEY}" \
    "${FIXTURE_SCRIPT}" iceberg glue-moto load "${load_args[@]}"
fi

namespace="${WORKLOAD}_sf${SCALE}"

if [[ "${RUN_REST}" == "1" ]]; then
  step "running REST catalog E2E"
  rest_env=(
    "NEXUS_ICEBERG_REST_E2E=1"
    "NEXUS_ICEBERG_REST_URI=${REST_URI}"
    "NEXUS_ICEBERG_WAREHOUSE=${WAREHOUSE}"
    "NEXUS_ICEBERG_S3_ENDPOINT=${S3_ENDPOINT}"
    "NEXUS_ICEBERG_S3_PATH_STYLE=true"
    "NEXUS_ICEBERG_S3_REGION=${S3_REGION}"
    "NEXUS_ICEBERG_S3_ACCESS_KEY_ID=${S3_ACCESS_KEY}"
    "NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY=${S3_SECRET_KEY}"
    "NEXUS_ICEBERG_NAMESPACE=${namespace}"
    "NEXUS_ICEBERG_REST_E2E_TABLE=${REST_TABLE}"
    "NEXUS_ICEBERG_REST_E2E_GROUP_COL=${REST_GROUP_COL}"
  )
  if [[ -n "${REST_EXPECTED_ROWS}" ]]; then
    rest_env+=("NEXUS_ICEBERG_REST_E2E_EXPECTED_ROWS=${REST_EXPECTED_ROWS}")
  fi
  if [[ "${WORKLOAD}" == "equality_delete" ]]; then
    rest_env+=(
      "NEXUS_ICEBERG_REST_E2E_EXPECTED_EQ_DELETE_FILES=1"
      "NEXUS_ICEBERG_REST_E2E_EXPECTED_IDS=1,3,4,5"
    )
  fi
  if [[ "${GPU_PARITY}" == "1" ]]; then
    rest_env+=("NEXUS_ICEBERG_REST_E2E_GPU_PARITY=1")
    env "${rest_env[@]}" \
      NEXUS_ICEBERG_S3_SESSION_TOKEN= \
      AWS_SESSION_TOKEN= \
      AWS_SECURITY_TOKEN= \
      flock /tmp/cudf-gpu.lock \
      cargo run -p nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e
  else
    env "${rest_env[@]}" \
      NEXUS_ICEBERG_S3_SESSION_TOKEN= \
      AWS_SESSION_TOKEN= \
      AWS_SECURITY_TOKEN= \
      cargo run -p nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e
  fi
fi

glue_env=(
  "NEXUS_ICEBERG_GLUE_LOCAL_E2E=1"
  "NEXUS_ICEBERG_GLUE_URI=${GLUE_URI}"
  "NEXUS_ICEBERG_WAREHOUSE=${WAREHOUSE}"
  "NEXUS_ICEBERG_S3_ENDPOINT=${S3_ENDPOINT}"
  "NEXUS_ICEBERG_S3_PATH_STYLE=true"
  "NEXUS_ICEBERG_S3_REGION=${S3_REGION}"
  "NEXUS_ICEBERG_S3_ACCESS_KEY_ID=${S3_ACCESS_KEY}"
  "NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY=${S3_SECRET_KEY}"
  "MOTO_ACCESS_KEY=${MOTO_ACCESS_KEY}"
  "MOTO_SECRET_KEY=${MOTO_SECRET_KEY}"
)

if [[ "${RUN_GLUE}" == "1" ]]; then
  step "running Glue/Moto backend E2E"
  env "${glue_env[@]}" \
    cargo run -p nexus-tools --features iceberg-e2e -- iceberg-glue-moto-e2e
fi

if [[ "${RUN_SERVER}" == "1" ]]; then
  step "running server + Glue/Moto E2E"
  env "${glue_env[@]}" \
    cargo run -p nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-moto-e2e
fi

if [[ "${RUN_REMOTE_DIRECT}" == "1" ]]; then
  step "running credentialless Glue/S3 remote-direct server E2E (GPU, flock /tmp/cudf-gpu.lock)"
  # RustFS credentials flow ONLY through the provider-path env pair
  # (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY) so the runner exercises the
  # default credential chain; static NEXUS_ICEBERG_S3_* credentials and
  # ambient session state are scrubbed.
  env \
    NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E=1 \
    NEXUS_ICEBERG_GLUE_URI="${GLUE_URI}" \
    NEXUS_ICEBERG_WAREHOUSE="${WAREHOUSE}" \
    NEXUS_ICEBERG_S3_ENDPOINT="${S3_ENDPOINT}" \
    NEXUS_ICEBERG_S3_REGION="${S3_REGION}" \
    NEXUS_ICEBERG_NAMESPACE="${namespace}" \
    AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY}" \
    AWS_SECRET_ACCESS_KEY="${S3_SECRET_KEY}" \
    AWS_SESSION_TOKEN= \
    AWS_SECURITY_TOKEN= \
    AWS_PROFILE= \
    NEXUS_ICEBERG_S3_ACCESS_KEY_ID= \
    NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY= \
    NEXUS_ICEBERG_S3_SESSION_TOKEN= \
    flock /tmp/cudf-gpu.lock \
    cargo run -p nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-remote-direct-e2e
fi

if [[ "${RUN_SERVER_REST_MULTIGPU}" == "1" ]]; then
  step "running production Flight SQL + REST/RustFS four-GPU acceptance (flock /tmp/cudf-gpu.lock)"
  (
    flock 9
    cargo build --release --bin datafusion_nexus_server --features server,iceberg,nvml
    env \
      NEXUS_ICEBERG_SERVER_REST_MULTIGPU_E2E=1 \
      NEXUS_SERVER_BINARY="${REPO_ROOT}/target/release/datafusion_nexus_server" \
      NEXUS_ICEBERG_REST_URI="${REST_URI}" \
      NEXUS_ICEBERG_WAREHOUSE="${WAREHOUSE}" \
      NEXUS_ICEBERG_S3_ENDPOINT="${S3_ENDPOINT}" \
      NEXUS_ICEBERG_S3_PATH_STYLE=true \
      NEXUS_ICEBERG_S3_REGION="${S3_REGION}" \
      NEXUS_ICEBERG_S3_ACCESS_KEY_ID="${S3_ACCESS_KEY}" \
      NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY="${S3_SECRET_KEY}" \
      NEXUS_ICEBERG_S3_SESSION_TOKEN= \
      AWS_SESSION_TOKEN= \
      AWS_SECURITY_TOKEN= \
      NEXUS_ICEBERG_NAMESPACE="${namespace}" \
      cargo run -p nexus-tools --features server-iceberg-e2e -- server-iceberg-rest-multigpu-e2e
  ) 9>/tmp/cudf-gpu.lock
fi

step "iceberg-local E2E suite passed"
