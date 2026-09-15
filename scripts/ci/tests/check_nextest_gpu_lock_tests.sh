#!/usr/bin/env bash
# Lock behavior of scripts/ci/nextest_gpu_lock.sh against a scratch lock file.
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
wrapper="${root}/scripts/ci/nextest_gpu_lock.sh"
scratch=$(mktemp -d)
trap 'rm -rf "${scratch}"' EXIT
lock="${scratch}/gpu.lock"
: >"${lock}"
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

# A CPU test must not wait for a GPU user holding the lock exclusively. --close
# keeps the lock off the sleep child, so killing the holder releases it.
flock --close --exclusive "${lock}" sleep 30 &
holder=$!
until ! flock --nonblock --shared "${lock}" true; do :; done
if ! NEXTEST_TEST_GROUP=@global timeout 5 "${wrapper}" "${lock}" true; then
  fail "@global test waited for the GPU lock"
fi
kill "${holder}"
wait "${holder}" 2>/dev/null || true

# The wrapped test runs while holding the lock in its group's mode.
if NEXTEST_TEST_GROUP=gpu timeout 5 "${wrapper}" "${lock}" flock --nonblock --exclusive "${lock}" true; then
  fail "gpu test does not hold a shared lock"
fi
if ! NEXTEST_TEST_GROUP=gpu timeout 5 "${wrapper}" "${lock}" flock --nonblock --shared "${lock}" true; then
  fail "gpu test excludes other shared GPU users"
fi
if NEXTEST_TEST_GROUP=gpu-exclusive timeout 5 "${wrapper}" "${lock}" flock --nonblock --shared "${lock}" true; then
  fail "gpu-exclusive test does not hold an exclusive lock"
fi

# An ancestor holding the lock passes its descriptor down instead of deadlocking.
if ! timeout 5 flock --exclusive "${lock}" env NEXTEST_TEST_GROUP=gpu "${wrapper}" "${lock}" true; then
  fail "wrapper deadlocked under an ancestor holding the lock"
fi

if [[ "${failures}" -ne 0 ]]; then
  exit 1
fi
echo "nextest GPU lock wrapper: all checks passed"
