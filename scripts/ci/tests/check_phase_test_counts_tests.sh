#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"
exec python3 scripts/ci/tests/phase_test_counts_tests.py
