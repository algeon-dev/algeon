#!/usr/bin/env bash
# Policy test for the vendored sloplint tool.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SLOPLINT="${ROOT}/scripts/ci/sloplint.sh"

if [[ ! -x "${SLOPLINT}" ]]; then
  echo "FAIL: sloplint wrapper not found or not executable at ${SLOPLINT}" >&2
  exit 1
fi

scratch=$(mktemp -d)
trap 'rm -rf "${scratch}"' EXIT
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

# 1. Clean technical prose scores low.
cat <<'EOF' >"${scratch}/clean.md"
# Query Plan Lowering

The optimizer lowers logical plans into physical candidates.
Each candidate operator specifies memory requirements and execution boundaries.
DataFusion physical plans are mapped directly to corresponding native kernels.
EOF

score_output=$("${SLOPLINT}" score "${scratch}/clean.md")
score=$(echo "${score_output}" | grep -o 'score: [0-9]*/100' | cut -d' ' -f2 | cut -d'/' -f1)
if [[ -z "${score}" || "${score}" -gt 25 ]]; then
  fail "clean file scored unexpectedly high: ${score_output}"
fi

# 2. Slop prose scores high and triggers --fail-above.
cat <<'EOF' >"${scratch}/slop.md"
# Overview

In today's fast-paced digital age, it is not just about speed; it is about excellence.
This revolutionary breakthrough plays a crucial role in today's rapidly evolving landscape.
Furthermore, it serves as a testament to our ongoing commitment to excellence and innovation.
The data tells us that we must embark on a transformative journey toward digital excellence.
EOF

if "${SLOPLINT}" score "${scratch}/slop.md" --fail-above 40 >/dev/null 2>&1; then
  fail "sloplint --fail-above 40 unexpectedly passed on slop content"
fi

# 3. Analyze command identifies specific rules.
analyze_output=$("${SLOPLINT}" analyze "${scratch}/slop.md")
if ! echo "${analyze_output}" | grep -q "in today's fast-paced"; then
  fail "analyze failed to detect 'in today\'s fast-paced': ${analyze_output}"
fi

# 4. Scan command ranks files and fails when threshold exceeded.
if "${SLOPLINT}" scan "${scratch}" --fail-above 40 >/dev/null 2>&1; then
  fail "sloplint scan --fail-above 40 did not exit with failure on slop file"
fi

# 5. Fix command replaces curly quotes and em-dashes.
cat <<'EOF' >"${scratch}/needs_fix.md"
Here is a “test” with em—dash and stray en – dash.
EOF

"${SLOPLINT}" fix "${scratch}/needs_fix.md" -w >/dev/null
fixed_content=$(cat "${scratch}/needs_fix.md")
if [[ "${fixed_content}" =~ [“”—] ]]; then
  fail "fix command failed to replace quotes or em-dash: ${fixed_content}"
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "check_sloplint_tests: ${failures} check(s) failed" >&2
  exit 1
fi

echo "check_sloplint_tests: all checks passed"
