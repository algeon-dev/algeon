#!/usr/bin/env bash
# Run TPC-DS Iceberg upload for a given scale factor.
#   AWS_PROFILE=xdr-int ./run_upload.sh sf1
#   TPCDS_ONLY=call_center,reason AWS_PROFILE=xdr-int ./run_upload.sh sf1
set -euo pipefail

SF="${1:?usage: $0 <sf1|sf10>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

case "${SF}" in
  sf1)  DEFAULT_DIR="${REPO_ROOT}/fixture/data/tpcds/sf1/parquet" ;;
  sf10) DEFAULT_DIR="${REPO_ROOT}/fixture/data/tpcds/sf10/parquet" ;;
  *)    echo "Unsupported scale factor: ${SF}" >&2; exit 2 ;;
esac

export AWS_PROFILE="${AWS_PROFILE:-xdr-int}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export TPCDS_SF="${SF}"
export TPCDS_SOURCE_DIR="${TPCDS_SOURCE_DIR:-${DEFAULT_DIR}}"
export TPCDS_GLUE_DB="${TPCDS_GLUE_DB:-morrist_tpcds}"

exec "${HERE}/.venv/bin/python" "${HERE}/upload_tpcds_iceberg.py"
