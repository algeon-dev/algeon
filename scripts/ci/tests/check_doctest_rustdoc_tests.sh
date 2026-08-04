#!/usr/bin/env bash
# Isolated check_all doctest-lock/accounting and strict-rustdoc orchestration.
# All Cargo, flock, and policy preflight commands are faked; no GPU is opened.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-doctest-rustdoc.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin" "${tmp_dir}/phase-logs"

cat >"${tmp_dir}/bin/bash" <<'SH'
#!/usr/bin/bash
case "${1:-}" in
  scripts/build/check_cpu_packaging_policy.sh | scripts/build/check_build_support.sh | scripts/ci/tests/check_doctest_report_tests.sh | scripts/ci/tests/check_phase_test_counts_tests.sh | scripts/ci/tests/check_gpu_test_lane_tests.sh | scripts/ci/check_gpu_test_lane.sh | scripts/ci/tests/check_doctest_rustdoc_tests.sh | scripts/ci/tests/check_gpu_scheduling_tests.sh)
    exit 0
    ;;
esac
exec /usr/bin/bash "$@"
SH

cat >"${tmp_dir}/bin/cargo" <<'SH'
#!/usr/bin/bash
set -u
{
  printf 'cargo rustdocflags=%s' "${RUSTDOCFLAGS-unset}"
  printf ' <%s>' "$@"
  printf '\n'
} >>"${COMMAND_LOG}"

if [[ "${1:-}" == "nextest" && "${2:-}" == "--version" ]]; then
  echo "cargo-nextest 0.9.140"
  exit 0
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "show-config" ]]; then
  echo "group: gpu (max threads = num-cpus)"
  echo "group: gpu-exclusive (max threads = 1)"
  exit 0
fi
if [[ "${1:-}" == "metadata" ]]; then
  cat <<'JSON'
{"packages":[{"id":"path+file:///fake#nexus-iceberg@0.1.0","name":"nexus-iceberg"}],"resolve":{"nodes":[{"id":"path+file:///fake#nexus-iceberg@0.1.0","deps":[]}]}}
JSON
  exit 0
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "run" ]]; then
  if [[ " $* " == *" not group(gpu) & not group(gpu-exclusive) "* ]]; then
    echo "Summary [ 0.100s] 3 tests run: 3 passed, 5 skipped"
  elif [[ " $* " == *" group(gpu-exclusive) "* ]]; then
    echo "Summary [ 0.100s] 1 test run: 1 passed, 7 skipped"
  else
    echo "Summary [ 0.100s] 2 tests run: 2 passed, 6 skipped"
  fi
  exit 0
fi
if [[ "${1:-}" == "test" && " $* " == *" --doc "* ]]; then
  case "${FAKE_DOCTEST_CASE:-multi}" in
    multi)
      echo "RAW_DOCTEST_STDOUT"
      echo "Doc-tests first_crate"
      echo "running 3 tests"
      echo "test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 4 filtered out; finished in 0.01s"
      echo "Doc-tests second_crate"
      echo "running 1 test"
      echo "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.02s"
      echo "RAW_DOCTEST_STDERR" >&2
      exit 0
      ;;
    zero)
      echo "Doc-tests zero_crate"
      echo "running 0 tests"
      echo "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
      exit 0
      ;;
    no_harness)
      echo "Finished test profile"
      exit 0
      ;;
    failure)
      echo "RAW_FAILURE_STDOUT"
      echo "Doc-tests failing_crate"
      echo "running 1 test"
      echo "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
      echo "RAW_FAILURE_STDERR" >&2
      exit 37
      ;;
    malformed)
      echo "Doc-tests malformed_crate"
      echo "test result: maybe"
      exit 0
      ;;
  esac
fi
if [[ "${1:-}" == "test" ]]; then
  echo "running 2 tests"
  echo "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
  exit 0
fi
if [[ "${1:-}" == "doc" ]]; then
  exit "${FAKE_RUSTDOC_STATUS:-0}"
fi
exit 0
SH

cat >"${tmp_dir}/bin/date" <<'SH'
#!/usr/bin/bash
set -euo pipefail
value=0
[[ ! -s "${FAKE_DATE_COUNTER}" ]] || value=$(<"${FAKE_DATE_COUNTER}")
value=$((value + 100))
printf '%s\n' "${value}" >"${FAKE_DATE_COUNTER}"
printf '%s\n' "${value}"
SH

cat >"${tmp_dir}/bin/nproc" <<'SH'
#!/usr/bin/bash
printf '16\n'
SH

cat >"${tmp_dir}/bin/flock" <<'SH'
#!/usr/bin/bash
set -u
{
  printf 'flock'
  printf ' <%s>' "$@"
  printf '\n'
} >>"${COMMAND_LOG}"
shift
"$@"
SH
chmod +x "${tmp_dir}/bin/bash" "${tmp_dir}/bin/cargo" "${tmp_dir}/bin/date" \
  "${tmp_dir}/bin/flock" "${tmp_dir}/bin/nproc"

run_check() {
  local name=$1
  shift
  local output="${tmp_dir}/${name}.output"
  local log="${tmp_dir}/${name}.log"
  local status=0
  : >"${log}"
  : >"${tmp_dir}/${name}.date"
  env -u FEATURES -u TEST_RUNNER -u GPU_LOCK -u STRICT -u RUSTDOCFLAGS \
    PATH="${tmp_dir}/bin:${PATH}" COMMAND_LOG="${log}" \
    CHECK_ALL_PHASE_LOG_DIR="${tmp_dir}/phase-logs" FAKE_DATE_COUNTER="${tmp_dir}/${name}.date" \
    "$@" >"${output}" 2>&1 || status=$?
  printf '%s\n' "${status}" >"${tmp_dir}/${name}.status"
}

assert_status() {
  local name=$1
  local expected=$2
  local actual
  actual=$(<"${tmp_dir}/${name}.status")
  if [[ "${actual}" -ne "${expected}" ]]; then
    echo "FAIL: ${name}: expected exit ${expected}, got ${actual}" >&2
    sed 's/^/  /' "${tmp_dir}/${name}.output" >&2
    exit 1
  fi
}

assert_contains() {
  local file=$1
  local expected=$2
  if ! grep -Fq -- "${expected}" "${file}"; then
    echo "FAIL: ${file} does not contain: ${expected}" >&2
    sed 's/^/  /' "${file}" >&2
    exit 1
  fi
}

assert_phase_contains() {
  local file=$1
  local phase=$2
  local expected=$3
  local row
  # Anchor at the start of the phase column so "rustdoc" does not also match
  # the later "bench-rustdoc" row introduced by the explicit bench phases.
  row=$(grep -E "^${phase} \\| " "${file}" | tail -n 1) || {
    echo "FAIL: ${file} has no phase-table row for ${phase}" >&2
    sed 's/^/  /' "${file}" >&2
    exit 1
  }
  if [[ "${row}" != *"${expected}"* ]]; then
    echo "FAIL: ${phase} row does not contain: ${expected}" >&2
    echo "  ${row}" >&2
    exit 1
  fi
}

run_check multi env STRICT=1 RUSTDOCFLAGS='-A warnings -Cdebuginfo=0' /usr/bin/bash scripts/check_all.sh
assert_status multi 0
assert_contains "${tmp_dir}/multi.output" "RAW_DOCTEST_STDOUT"
assert_contains "${tmp_dir}/multi.output" "RAW_DOCTEST_STDERR"
assert_contains "${tmp_dir}/multi.output" "DOCTEST HARNESS: name=first_crate"
assert_contains "${tmp_dir}/multi.output" "Doctest accounting preflight for final aggregation: command_status=0 parse_status=ok harnesses=2 executed=3 passed=3 failed=0 ignored=1 measured=0 filtered=6"
assert_contains "${tmp_dir}/multi.output" "Doctest accounting final: command_status=0 parse_status=ok harnesses=2 executed=3 passed=3 failed=0 ignored=1 measured=0 filtered=6"
assert_contains "${tmp_dir}/multi.log" "flock </tmp/cudf-gpu.lock> <cargo> <test> <--all-features> <--doc> <--no-fail-fast>"
assert_contains "${tmp_dir}/multi.log" "cargo rustdocflags=-A warnings -Cdebuginfo=0 -D warnings <doc> <--all-features> <--no-deps> <--keep-going>"
assert_phase_contains "${tmp_dir}/multi.output" doctests "0.100s | PASS | harnesses=2 executed=3 passed=3 failed=0 ignored=1 measured=0 filtered=6 | - | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/doctests.log"
assert_phase_contains "${tmp_dir}/multi.output" doctests "doctests | flock /tmp/cudf-gpu.lock cargo test --all-features --doc --no-fail-fast | 0.100s"
assert_phase_contains "${tmp_dir}/multi.output" rustdoc "0.100s | PASS | - | - | - | all | ${tmp_dir}/phase-logs/rustdoc.log"
assert_phase_contains "${tmp_dir}/multi.output" rustdoc "rustdoc | RUSTDOCFLAGS='-A warnings -Cdebuginfo=0 -D warnings' cargo doc --all-features --no-deps --keep-going | 0.100s"
assert_contains "${tmp_dir}/phase-logs/doctests.log" "RAW_DOCTEST_STDERR"
if [[ $(grep -c '^flock ' "${tmp_dir}/multi.log") -ne 3 ]]; then
  echo "FAIL: expected shared, exclusive, and doctest flock invocations" >&2
  sed 's/^/  /' "${tmp_dir}/multi.log" >&2
  exit 1
fi
echo "PASS: locked multi-harness accounting and final strict rustdoc flags"

run_check zero env FAKE_DOCTEST_CASE=zero /usr/bin/bash scripts/check_all.sh
assert_status zero 0
assert_contains "${tmp_dir}/zero.output" "command_status=0 parse_status=ok harnesses=1 executed=0 passed=0 failed=0 ignored=0 measured=0 filtered=0"
assert_phase_contains "${tmp_dir}/zero.output" doctests "0.100s | PASS | harnesses=1 executed=0 passed=0 failed=0 ignored=0 measured=0 filtered=0"
echo "PASS: actual zero doctests"

run_check no_harness env FAKE_DOCTEST_CASE=no_harness /usr/bin/bash scripts/check_all.sh
assert_status no_harness 1
assert_contains "${tmp_dir}/no_harness.output" "no Doc-tests harness results found"
assert_contains "${tmp_dir}/no_harness.output" "command_status=0 parse_status=error harnesses=0 executed=0"
assert_phase_contains "${tmp_dir}/no_harness.output" doctests "0.100s | FAIL(accounting) | harnesses=0 executed=0 passed=0 failed=0 ignored=0 measured=0 filtered=0"
echo "PASS: missing doctest harness fails closed"

run_check failure env FAKE_DOCTEST_CASE=failure /usr/bin/bash scripts/check_all.sh
assert_status failure 1
assert_contains "${tmp_dir}/failure.output" "RAW_FAILURE_STDOUT"
assert_contains "${tmp_dir}/failure.output" "RAW_FAILURE_STDERR"
assert_contains "${tmp_dir}/failure.output" "command_status=37 parse_status=ok harnesses=1 executed=1 passed=0 failed=1 ignored=0 measured=0 filtered=0"
assert_phase_contains "${tmp_dir}/failure.output" doctests "0.100s | FAIL(37) | harnesses=1 executed=1 passed=0 failed=1 ignored=0 measured=0 filtered=0"
assert_contains "${tmp_dir}/phase-logs/doctests.log" "RAW_FAILURE_STDERR"
echo "PASS: raw doctest failure status and complete output preserved"

run_check malformed env FAKE_DOCTEST_CASE=malformed /usr/bin/bash scripts/check_all.sh
assert_status malformed 1
assert_contains "${tmp_dir}/malformed.output" "malformed doctest result"
assert_contains "${tmp_dir}/malformed.output" "command_status=0 parse_status=error harnesses=0 executed=0"
assert_phase_contains "${tmp_dir}/malformed.output" doctests "0.100s | FAIL(accounting) | harnesses=0 executed=0 passed=0 failed=0 ignored=0 measured=0 filtered=0"
echo "PASS: malformed doctest output fails closed"

run_check rustdoc_failure env FAKE_RUSTDOC_STATUS=42 /usr/bin/bash scripts/check_all.sh
assert_status rustdoc_failure 1
assert_contains "${tmp_dir}/rustdoc_failure.output" "Rustdoc final: command_status=42 warnings=denied"
assert_phase_contains "${tmp_dir}/rustdoc_failure.output" rustdoc "0.100s | FAIL(42) | - | - | - | all | ${tmp_dir}/phase-logs/rustdoc.log"
echo "PASS: rustdoc failure status preserved"

run_check invalid_strict env STRICT=0 /usr/bin/bash scripts/check_all.sh
assert_status invalid_strict 2
assert_contains "${tmp_dir}/invalid_strict.output" "STRICT is a compatibility flag for strict rustdoc only"
if [[ -s "${tmp_dir}/invalid_strict.log" ]]; then
  echo "FAIL: invalid STRICT reached Cargo/flock commands" >&2
  sed 's/^/  /' "${tmp_dir}/invalid_strict.log" >&2
  exit 1
fi
echo "PASS: invalid STRICT compatibility flag fails before work"

echo "OK: doctest lock/accounting and strict-rustdoc orchestration tests passed without GPU execution."
