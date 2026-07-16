#!/usr/bin/env bash
# Isolated doctest output parser and status-preservation tests.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"
exec python3 scripts/ci/tests/doctest_report_tests.py
