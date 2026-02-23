#!/usr/bin/env bash
# Run the demo_db TPC-H + TPC-DS Iceberg upload via pyspark.
#
# Usage:
#   ./run_upload.sh                          # uploads everything defined in WORKLOADS
#   MORRIS_POC_SCOPE=tpch:sf1,tpcds:sf1 ./run_upload.sh
#   MORRIS_POC_ONLY=customer,nation ./run_upload.sh
#
# Auth: expects AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN in env,
#       OR a working AWS_PROFILE that the boto3 / aws-java-sdk chain can resolve.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

export AWS_REGION="${AWS_REGION:-us-east-1}"
export MORRIS_POC_FIXTURE_ROOT="${MORRIS_POC_FIXTURE_ROOT:-${REPO_ROOT}/tests/data}"

exec "${HERE}/.venv/bin/python" "${HERE}/upload_demo_db_iceberg.py"
