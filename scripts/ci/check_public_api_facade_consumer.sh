#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"
MANIFEST="${ROOT}/scripts/ci/fixtures/datafusion-facade-consumer/Cargo.toml"

if rg -n '^query-engine[[:space:]]*=' "${MANIFEST}"; then
  echo "ERROR: facade consumer fixture declares query-engine" >&2
  exit 1
fi

cargo check -p datafusion-facade-consumer --all-features
echo "PASS: facade consumer compiles with algeon-datafusion as its only Algeon execution dependency"
