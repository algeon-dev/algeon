#!/usr/bin/env bash
# Thin entry point for std-only RAPIDS build-policy and inclusion smoke tests.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"
exec python3 scripts/build/build_support_tests.py
