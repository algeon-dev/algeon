#!/usr/bin/env bash
# Black-box E2E entrypoint for the embedded backend example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
COMPOSE_FILE="${REPO_ROOT}/fixture/iceberg-local/docker-compose.yml"
FIXTURE_SCRIPT="${REPO_ROOT}/fixture/fixture.sh"

DOCKER="${DOCKER:-docker}"
DEFAULT_REST_URI="http://localhost:8181"
DEFAULT_S3_ENDPOINT="http://localhost:9000"
REST_URI="${ALGEON_ICEBERG_REST_URI:-${DEFAULT_REST_URI}}"
WAREHOUSE="${ALGEON_ICEBERG_WAREHOUSE:-s3://lakehouse/warehouse}"
S3_ENDPOINT="${ALGEON_ICEBERG_S3_ENDPOINT:-${DEFAULT_S3_ENDPOINT}}"
S3_REGION="${ALGEON_ICEBERG_S3_REGION:-us-east-1}"
S3_ACCESS_KEY="${ALGEON_ICEBERG_S3_ACCESS_KEY_ID:-${RUSTFS_ACCESS_KEY:-algeonadmin}}"
S3_SECRET_KEY="${ALGEON_ICEBERG_S3_SECRET_ACCESS_KEY:-${RUSTFS_SECRET_KEY:-algeonadmin}}"
WAREHOUSE_BUCKET=""

NO_LOAD=0
RELOAD=0
SKIP_STACK=0
GPU_PARITY=0

usage() {
  cat <<'USAGE'
Usage:
  crates/examples/scripts/e2e.sh [options]

Runs the embedded_backend example black-box E2E suite. The script starts the
local Iceberg REST + RustFS stack, loads citation_network with Iceberg
add-files, then runs the REST + Flight SQL checks under the GPU flock.

Options:
  --no-load      Do not load citation_network before running the checks
  --reload       Drop existing Iceberg tables before re-adding them
  --skip-stack   Do not run docker compose up -d; still wait for endpoints
  --gpu-parity   Reserved for this suite; currently not implemented
  -h, --help     Show this help

Environment overrides:
  DOCKER, ALGEON_ICEBERG_REST_URI, ALGEON_ICEBERG_WAREHOUSE,
  ALGEON_ICEBERG_S3_ENDPOINT, ALGEON_ICEBERG_S3_REGION,
  ALGEON_ICEBERG_S3_ACCESS_KEY_ID, ALGEON_ICEBERG_S3_SECRET_ACCESS_KEY,
  CITATION_NETWORK_DATA_DIR.

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

  step "waiting for ${name} (${url})"
  for _ in $(seq 1 60); do
    if curl --fail --silent --max-time 2 "${url}" >/dev/null; then
      return 0
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

  die "${name} must be a loopback http:// URL for embedded backend E2E, got '${value}'"
}

has_custom_endpoint_override() {
  [[ "${REST_URI}" != "${DEFAULT_REST_URI}" ]] ||
    [[ "${S3_ENDPOINT}" != "${DEFAULT_S3_ENDPOINT}" ]]
}

warehouse_bucket() {
  local warehouse="$1"
  local rest

  if [[ "${warehouse}" != s3://* ]]; then
    die "ALGEON_ICEBERG_WAREHOUSE must be an s3:// URI for embedded backend E2E, got '${warehouse}'"
  fi
  rest="${warehouse#s3://}"
  rest="${rest%%/*}"
  if [[ -z "${rest}" ]]; then
    die "ALGEON_ICEBERG_WAREHOUSE must include a bucket, got '${warehouse}'"
  fi
  printf '%s\n' "${rest}"
}

if [[ "${1:-}" == "embedded-backend" ]]; then
  shift
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-load)
      NO_LOAD=1
      shift
      ;;
    --reload)
      RELOAD=1
      shift
      ;;
    --skip-stack)
      SKIP_STACK=1
      shift
      ;;
    --gpu-parity)
      GPU_PARITY=1
      shift
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

if [[ "${GPU_PARITY}" == "1" ]]; then
  die "--gpu-parity is not implemented for embedded backend E2E yet; omit the flag for the supported black-box suite"
fi

require_loopback_http_url "ALGEON_ICEBERG_REST_URI" "${REST_URI}"
require_loopback_http_url "ALGEON_ICEBERG_S3_ENDPOINT" "${S3_ENDPOINT}"
WAREHOUSE_BUCKET="$(warehouse_bucket "${WAREHOUSE}")"

if [[ "${SKIP_STACK}" != "1" ]] && has_custom_endpoint_override; then
  die "custom local endpoint overrides require --skip-stack because the compose file publishes fixed host ports"
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
wait_url "Iceberg REST" "${REST_URI}/v1/config"

if [[ "${NO_LOAD}" != "1" ]]; then
  step "loading citation_network into local Iceberg REST catalog with add-files"
  load_args=(--workload citation_network --load-mode add-files)
  if [[ "${RELOAD}" == "1" ]]; then
    load_args+=(--reload)
  fi
  env \
    REST_ENDPOINT="${REST_URI}" \
    S3_ENDPOINT="${S3_ENDPOINT}" \
    WAREHOUSE="${WAREHOUSE}" \
    S3_REGION="${S3_REGION}" \
    RUSTFS_ACCESS_KEY="${S3_ACCESS_KEY}" \
    RUSTFS_SECRET_KEY="${S3_SECRET_KEY}" \
    "${FIXTURE_SCRIPT}" iceberg rest load "${load_args[@]}"
fi

step "running embedded backend E2E (GPU, flock /tmp/cudf-gpu.lock)"
env \
  ALGEON_EMBEDDED_BACKEND_E2E=1 \
  ALGEON_ICEBERG_REST_URI="${REST_URI}" \
  ALGEON_ICEBERG_WAREHOUSE="${WAREHOUSE}" \
  ALGEON_ICEBERG_S3_ENDPOINT="${S3_ENDPOINT}" \
  ALGEON_ICEBERG_S3_PATH_STYLE=true \
  ALGEON_ICEBERG_S3_REGION="${S3_REGION}" \
  ALGEON_ICEBERG_S3_ACCESS_KEY_ID="${S3_ACCESS_KEY}" \
  ALGEON_ICEBERG_S3_SECRET_ACCESS_KEY="${S3_SECRET_KEY}" \
  ALGEON_ICEBERG_S3_SESSION_TOKEN= \
  AWS_SESSION_TOKEN= \
  AWS_SECURITY_TOKEN= \
  flock /tmp/cudf-gpu.lock \
  cargo run -p tools --features embedded-backend-e2e -- embedded-backend-e2e

step "embedded backend E2E suite passed"
