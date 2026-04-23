#!/usr/bin/env bash
# Isolated scheduling-policy tests. All Cargo/flock/sub-check invocations are
# faked: this script never executes a test binary and never opens a GPU context.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-gpu-scheduling.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin" "${tmp_dir}/phase-logs"

cat >"${tmp_dir}/bin/bash" <<'SH'
#!/usr/bin/bash
case "${1:-}" in
  */scripts/check_all.sh) ;;
  *)
    {
      printf 'bash'
      printf ' <%s>' "$@"
      printf '\n'
    } >>"${COMMAND_LOG}"
    ;;
esac
case "${1:-}" in
  scripts/build/check_build_support.sh)
    if [[ "${FAKE_POLICY_STATUS:-0}" -ne 0 ]]; then
      echo "RAW_POLICY_FAILURE" >&2
    fi
    exit "${FAKE_POLICY_STATUS:-0}"
    ;;
  scripts/ci/check_gpu_test_lane.sh)
    if [[ "${FAKE_LIVE_INVENTORY_STATUS:-0}" -ne 0 ]]; then
      echo "RAW_LIVE_INVENTORY_FAILURE" >&2
    fi
    exit "${FAKE_LIVE_INVENTORY_STATUS:-0}"
    ;;
  scripts/build/check_cpu_packaging_policy.sh | scripts/ci/tests/check_doctest_report_tests.sh | scripts/ci/tests/check_phase_test_counts_tests.sh | scripts/ci/tests/check_gpu_test_lane_tests.sh | scripts/ci/tests/check_doctest_rustdoc_tests.sh | scripts/ci/tests/check_gpu_scheduling_tests.sh)
    exit 0
    ;;
esac
exec /usr/bin/bash "$@"
SH

cat >"${tmp_dir}/bin/cargo" <<'SH'
#!/usr/bin/bash
set -u
{
  printf 'cargo'
  printf ' <%s>' "$@"
  printf ' {rustdocflags=%s}' "${RUSTDOCFLAGS-unset}"
  printf '\n'
} >>"${COMMAND_LOG}"

if [[ "${1:-}" == "nextest" && "${2:-}" == "--version" ]]; then
  [[ -z "${FAKE_NEXTEST_VERSION_STDERR:-}" ]] || echo "${FAKE_NEXTEST_VERSION_STDERR}" >&2
  echo "cargo-nextest ${FAKE_NEXTEST_VERSION:-0.9.140}"
  exit "${FAKE_NEXTEST_VERSION_STATUS:-0}"
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "show-config" ]]; then
  echo "group: gpu (max threads = ${FAKE_GPU_CAP:-num-cpus})"
  echo "group: gpu-exclusive (max threads = ${FAKE_GPU_EXCLUSIVE_CAP:-1})"
  exit "${FAKE_SHOW_CONFIG_STATUS:-0}"
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "run" ]]; then
  if [[ " $* " == *" not group(gpu) & not group(gpu-exclusive) "* ]]; then
    if [[ "${FAKE_CPU_SUMMARY_MISSING:-0}" -eq 1 ]]; then
      echo "CPU_OUTPUT_WITHOUT_SUMMARY"
      exit 0
    fi
    if [[ "${FAKE_CPU_TEST_STATUS:-0}" -ne 0 ]]; then
      echo "RAW_CPU_FAILURE" >&2
      echo "Summary [ 0.200s] 3 tests run: 2 passed, 1 failed, 5 skipped"
      exit "${FAKE_CPU_TEST_STATUS}"
    fi
    echo "Summary [ 0.100s] 3 tests run: 3 passed, 5 skipped"
    exit 0
  fi
  if [[ " $* " == *" group(gpu-exclusive) "* ]]; then
    echo "Summary [ 0.100s] 1 test run: 1 passed, 7 skipped"
    exit 0
  fi
  echo "Summary [ 0.100s] 2 tests run: 2 passed, 6 skipped"
  exit 0
fi
if [[ "${1:-}" == "test" && " $* " == *" --doc "* ]]; then
  echo "Doc-tests fake_zero"
  echo "running 0 tests"
  echo "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
  exit 0
fi
if [[ "${1:-}" == "test" ]]; then
  if [[ "${FAKE_CARGO_TEST_STATUS:-0}" -ne 0 ]]; then
    echo "RAW_CARGO_DIAGNOSTIC_FAILURE" >&2
    echo "test result: FAILED. 3 passed; 1 failed; 1 ignored; 0 measured; 2 filtered out; finished in 0.01s"
    exit "${FAKE_CARGO_TEST_STATUS}"
  fi
  echo "test result: ok. 4 passed; 0 failed; 1 ignored; 0 measured; 2 filtered out; finished in 0.01s"
  exit 0
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
printf '%s\n' "${FAKE_NPROC_OUTPUT:-16}"
exit "${FAKE_NPROC_STATUS:-0}"
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

assert_not_contains() {
  local file=$1
  local unexpected=$2
  if grep -Fq -- "${unexpected}" "${file}"; then
    echo "FAIL: ${file} unexpectedly contains: ${unexpected}" >&2
    sed 's/^/  /' "${file}" >&2
    exit 1
  fi
}

assert_empty() {
  local file=$1
  if [[ -s "${file}" ]]; then
    echo "FAIL: ${file} was expected to be empty" >&2
    sed 's/^/  /' "${file}" >&2
    exit 1
  fi
}

assert_before() {
  local file=$1
  local first=$2
  local second=$3
  local first_line second_line
  assert_contains "${file}" "${first}"
  assert_contains "${file}" "${second}"
  first_line=$(grep -nF -- "${first}" "${file}" | head -n 1 | cut -d: -f1)
  second_line=$(grep -nF -- "${second}" "${file}" | head -n 1 | cut -d: -f1)
  if [[ "${first_line}" -ge "${second_line}" ]]; then
    echo "FAIL: ${file} does not place '${first}' before '${second}'" >&2
    sed 's/^/  /' "${file}" >&2
    exit 1
  fi
}

assert_phase_contains() {
  local file=$1
  local phase=$2
  local expected=$3
  local row
  row=$(grep -F "${phase} | " "${file}" | tail -n 1) || {
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

# Shared-GPU scheduling follows the CPU count available to the environment.
python3 - <<'PY'
import tomllib

with open(".config/nextest.toml", "rb") as config_file:
    config = tomllib.load(config_file)
assert config["nextest-version"]["required"] == "0.9.140"
assert config["test-groups"]["gpu"]["max-threads"] == "num-cpus"
assert config["test-groups"]["gpu-exclusive"]["max-threads"] == 1
assert "gpu-parallel" not in config["test-groups"]
assert "gpu-parallel" not in config["profile"]
PY
echo "PASS: CPU-derived shared cap and fixed exclusive cap"

# Help and all argument/configuration errors finish before any sub-check, Cargo,
# inventory, or flock work begins.
run_check help /usr/bin/bash scripts/check_all.sh --help
assert_status help 0
assert_contains "${tmp_dir}/help.output" "Usage: bash scripts/check_all.sh [OPTIONS]"
assert_contains "${tmp_dir}/help.output" "cargo-diagnostic"
assert_contains "${tmp_dir}/help.output" "CHECK_ALL_PHASE_LOG_DIR optionally selects the persistent phase-log directory"
assert_empty "${tmp_dir}/help.log"
echo "PASS: help is side-effect free"

run_check unknown /usr/bin/bash scripts/check_all.sh --wat
assert_status unknown 2
assert_contains "${tmp_dir}/unknown.output" "unknown option: --wat"
assert_empty "${tmp_dir}/unknown.log"

run_check duplicate /usr/bin/bash scripts/check_all.sh --fast --fast
assert_status duplicate 2
assert_contains "${tmp_dir}/duplicate.output" "duplicate option: --fast"
assert_empty "${tmp_dir}/duplicate.log"

run_check feature_conflict /usr/bin/bash scripts/check_all.sh --all-features --features server
assert_status feature_conflict 2
assert_contains "${tmp_dir}/feature_conflict.output" "--all-features conflicts"
assert_empty "${tmp_dir}/feature_conflict.log"

run_check cargo_cli_spelling /usr/bin/bash scripts/check_all.sh --runner cargo
assert_status cargo_cli_spelling 2
assert_contains "${tmp_dir}/cargo_cli_spelling.output" "--runner must be nextest or cargo-diagnostic"
assert_empty "${tmp_dir}/cargo_cli_spelling.log"
echo "PASS: unknown, duplicate, and conflicting CLI options fail before work"

run_check invalid_features_env env FEATURES='server,,iceberg' /usr/bin/bash scripts/check_all.sh
assert_status invalid_features_env 2
assert_contains "${tmp_dir}/invalid_features_env.output" "FEATURES must be empty or a comma-separated feature list"
assert_empty "${tmp_dir}/invalid_features_env.log"

run_check invalid_runner_env env TEST_RUNNER=surprise /usr/bin/bash scripts/check_all.sh
assert_status invalid_runner_env 2
assert_contains "${tmp_dir}/invalid_runner_env.output" "TEST_RUNNER must be nextest, cargo, or cargo-diagnostic"
assert_empty "${tmp_dir}/invalid_runner_env.log"

run_check invalid_lock_env env GPU_LOCK= /usr/bin/bash scripts/check_all.sh
assert_status invalid_lock_env 2
assert_contains "${tmp_dir}/invalid_lock_env.output" "GPU_LOCK must be a non-empty single-line path that does not begin with '-'"
assert_empty "${tmp_dir}/invalid_lock_env.log"

run_check option_like_lock_env env GPU_LOCK=--exclusive /usr/bin/bash scripts/check_all.sh
assert_status option_like_lock_env 2
assert_contains "${tmp_dir}/option_like_lock_env.output" "GPU_LOCK must be a non-empty single-line path that does not begin with '-'"
assert_empty "${tmp_dir}/option_like_lock_env.log"

run_check invalid_phase_log_env env 'CHECK_ALL_PHASE_LOG_DIR=bad|path' /usr/bin/bash scripts/check_all.sh
assert_status invalid_phase_log_env 2
assert_contains "${tmp_dir}/invalid_phase_log_env.output" "CHECK_ALL_PHASE_LOG_DIR must be a non-empty single-line path without '|'"
assert_empty "${tmp_dir}/invalid_phase_log_env.log"

echo "PASS: invalid compatibility environment values fail before work"

# Core-count detection fails closed before any build, inventory, or test work.
run_check nproc_failure env FAKE_NPROC_STATUS=7 /usr/bin/bash scripts/check_all.sh
assert_status nproc_failure 2
assert_contains "${tmp_dir}/nproc_failure.output" "nproc failed with status 7"
assert_empty "${tmp_dir}/nproc_failure.log"

run_check invalid_nproc env FAKE_NPROC_OUTPUT=zero /usr/bin/bash scripts/check_all.sh
assert_status invalid_nproc 2
assert_contains "${tmp_dir}/invalid_nproc.output" "nproc must report one positive integer"
assert_empty "${tmp_dir}/invalid_nproc.log"
echo "PASS: available CPU detection fails closed"

run_check wrong_nextest env FAKE_NEXTEST_VERSION=0.9.139 /usr/bin/bash scripts/check_all.sh
assert_status wrong_nextest 2
assert_contains "${tmp_dir}/wrong_nextest.output" "requires cargo-nextest 0.9.140"
assert_not_contains "${tmp_dir}/wrong_nextest.log" "bash <scripts/"

run_check missing_nextest env FAKE_NEXTEST_VERSION_STATUS=127 FAKE_NEXTEST_VERSION_STDERR=missing-nextest /usr/bin/bash scripts/check_all.sh
assert_status missing_nextest 2
assert_contains "${tmp_dir}/missing_nextest.output" "missing-nextest"
assert_contains "${tmp_dir}/missing_nextest.output" "cargo-nextest 0.9.140 is required"
assert_not_contains "${tmp_dir}/missing_nextest.log" "bash <scripts/"
echo "PASS: canonical runner requires exact cargo-nextest 0.9.140"

# Default mode must report and pass all 16 available cores. The fake commands
# make this a no-GPU orchestration test.
run_check default /usr/bin/bash scripts/check_all.sh
assert_status default 0
assert_contains "${tmp_dir}/default.output" "gpu_profile=default gpu_threads=16(nproc)"
assert_contains "${tmp_dir}/default.output" "GPU scheduling preflight: shared: profile=default group=gpu configured_cap=num-cpus explicit_test_threads=16"
assert_contains "${tmp_dir}/default.output" "exclusive: group=gpu-exclusive configured_cap=1 explicit_test_threads=1"
assert_contains "${tmp_dir}/default.output" "GPU scheduling final: shared: profile=default group=gpu configured_cap=num-cpus explicit_test_threads=16"
assert_contains "${tmp_dir}/default.log" "cargo <nextest> <show-config> <--color> <never> <-P> <default> <test-groups> <--all-features>"
assert_contains "${tmp_dir}/default.log" "cargo <nextest> <run> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)> <--no-fail-fast>"
assert_contains "${tmp_dir}/default.log" "cargo <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu)> <--no-fail-fast> <--test-threads> <16>"
assert_contains "${tmp_dir}/default.log" "cargo <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu-exclusive)> <--no-fail-fast> <--test-threads> <1>"
assert_contains "${tmp_dir}/default.log" "cargo <test> <--all-features> <--doc> <--no-fail-fast>"
assert_contains "${tmp_dir}/default.log" "cargo <+nightly> <fmt> <--all> <--check>"
assert_contains "${tmp_dir}/default.log" "cargo <check> <--all-features>"
assert_contains "${tmp_dir}/default.log" "cargo <clippy> <--all-targets> <--all-features>"
assert_contains "${tmp_dir}/default.log" "cargo <check> <-p> <datafusion-nexus-tools> <--all-targets> <--all-features>"
assert_contains "${tmp_dir}/default.log" "cargo <check> <-p> <datafusion-nexus-examples> <--all-targets>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/build/check_cpu_packaging_policy.sh>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_gpu_test_lane_tests.sh>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_doctest_rustdoc_tests.sh>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_gpu_scheduling_tests.sh>"
assert_contains "${tmp_dir}/default.output" "Phase | Command | Duration | Status | Test counts | GPU cap | Lock | Features | Log"
assert_phase_contains "${tmp_dir}/default.output" policy-selftests "0.100s | PASS | checks=7 | - | - | all | ${tmp_dir}/phase-logs/policy-selftests.log"
assert_phase_contains "${tmp_dir}/default.output" cpu-tests "cpu-tests | cargo nextest run --all-features -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast | 0.100s"
assert_phase_contains "${tmp_dir}/default.output" cpu-tests "0.100s | PASS | run=3 passed=3 failed=0 skipped=5 | - | - | all | ${tmp_dir}/phase-logs/cpu-tests.log"
assert_phase_contains "${tmp_dir}/default.output" gpu-shared-tests "gpu-shared-tests | flock /tmp/cudf-gpu.lock cargo nextest run -P default --all-features -E group(gpu) --no-fail-fast --test-threads 16 | 0.100s"
assert_phase_contains "${tmp_dir}/default.output" gpu-shared-tests "0.100s | PASS | run=2 passed=2 failed=0 skipped=6 | 16 | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/gpu-shared-tests.log"
assert_phase_contains "${tmp_dir}/default.output" gpu-exclusive-tests "gpu-exclusive-tests | flock /tmp/cudf-gpu.lock cargo nextest run -P default --all-features -E group(gpu-exclusive) --no-fail-fast --test-threads 1 | 0.100s"
assert_phase_contains "${tmp_dir}/default.output" gpu-exclusive-tests "0.100s | PASS | run=1 passed=1 failed=0 skipped=7 | 1 | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/gpu-exclusive-tests.log"
assert_phase_contains "${tmp_dir}/default.output" doctests "0.100s | PASS | harnesses=1 executed=0 passed=0 failed=0 ignored=0 measured=0 filtered=0 | - | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/doctests.log"
assert_phase_contains "${tmp_dir}/default.output" cargo-diagnostic "- | SKIPPED | - | - | - | all | -"
for phase_log in policy-selftests cpu-tests gpu-shared-tests gpu-exclusive-tests doctests rustdoc clippy tools-check examples-check; do
  [[ -f "${tmp_dir}/phase-logs/${phase_log}.log" ]] || {
    echo "FAIL: expected persistent phase log ${phase_log}.log" >&2
    exit 1
  }
done
default_shared_flock="flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu)> <--no-fail-fast> <--test-threads> <16>"
default_exclusive_flock="flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu-exclusive)> <--no-fail-fast> <--test-threads> <1>"
assert_before "${tmp_dir}/default.log" "${default_shared_flock}" "${default_exclusive_flock}"
default_doctest_flock="flock </tmp/cudf-gpu.lock> <cargo> <test> <--all-features> <--doc> <--no-fail-fast>"
assert_before "${tmp_dir}/default.log" "${default_exclusive_flock}" "${default_doctest_flock}"
if [[ $(grep -c '^flock ' "${tmp_dir}/default.log") -ne 3 ]]; then
  echo "FAIL: default scheduling must use one flock invocation per GPU resource class plus doctests" >&2
  sed 's/^/  /' "${tmp_dir}/default.log" >&2
  exit 1
fi
echo "PASS: CPU exclusion and sequential default GPU resource classes"
echo "PASS: full surface includes pinned fmt, default compile, clippy, tools, examples, and isolated policies"

run_check feature_env env FEATURES=server /usr/bin/bash scripts/check_all.sh --lane cpu
assert_status feature_env 0
assert_contains "${tmp_dir}/feature_env.output" "features=features:server(env)"
assert_contains "${tmp_dir}/feature_env.log" "cargo <nextest> <run> <--features> <server>"

run_check lock_env env GPU_LOCK=/tmp/custom-cudf.lock /usr/bin/bash scripts/check_all.sh --lane gpu-shared
assert_status lock_env 0
assert_contains "${tmp_dir}/lock_env.log" "flock </tmp/custom-cudf.lock> <cargo> <nextest>"
echo "PASS: FEATURES and GPU_LOCK compatibility values remain effective"

run_check cargo_fallback env TEST_RUNNER=cargo /usr/bin/bash scripts/check_all.sh
assert_status cargo_fallback 0
assert_contains "${tmp_dir}/cargo_fallback.log" "cargo <test> <--all-features> <--no-fail-fast> <--lib> <--bins> <--tests> <--examples> <--> <--test-threads=1>"
echo "PASS: cargo fallback remains serialized under the GPU lock"

run_check cargo_diagnostic_cli /usr/bin/bash scripts/check_all.sh --runner cargo-diagnostic
assert_status cargo_diagnostic_cli 0
assert_contains "${tmp_dir}/cargo_diagnostic_cli.output" "NONCANONICAL RUNNER"
assert_contains "${tmp_dir}/cargo_diagnostic_cli.output" "NONCANONICAL cargo-diagnostic surface passed"
assert_contains "${tmp_dir}/cargo_diagnostic_cli.log" "flock </tmp/cudf-gpu.lock> <cargo> <test> <--all-features> <--no-fail-fast> <--lib> <--bins> <--tests> <--examples> <--> <--test-threads=1>"
assert_not_contains "${tmp_dir}/cargo_diagnostic_cli.log" "cargo <nextest>"
assert_phase_contains "${tmp_dir}/cargo_diagnostic_cli.output" cargo-diagnostic "0.100s | PASS | run=4 passed=4 failed=0 ignored=1 measured=0 filtered=2 harnesses=1 | 1 | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/cargo-diagnostic.log"
assert_phase_contains "${tmp_dir}/cargo_diagnostic_cli.output" cargo-diagnostic "cargo-diagnostic | RUST_TEST_THREADS=1 flock /tmp/cudf-gpu.lock cargo test --all-features --no-fail-fast --lib --bins --tests --examples -- --test-threads=1 | 0.100s"
assert_phase_contains "${tmp_dir}/cargo_diagnostic_cli.output" cpu-tests "- | SKIPPED | - | - | - | all | -"
echo "PASS: Cargo runner is explicit, serialized, and noncanonical"

run_check cargo_diagnostic_failure env FAKE_CARGO_TEST_STATUS=33 /usr/bin/bash scripts/check_all.sh --runner cargo-diagnostic
assert_status cargo_diagnostic_failure 1
assert_contains "${tmp_dir}/cargo_diagnostic_failure.output" "RAW_CARGO_DIAGNOSTIC_FAILURE"
assert_phase_contains "${tmp_dir}/cargo_diagnostic_failure.output" cargo-diagnostic "0.100s | FAIL(33) | run=4 passed=3 failed=1 ignored=1 measured=0 filtered=2 harnesses=1 | 1 | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/cargo-diagnostic.log"
assert_contains "${tmp_dir}/phase-logs/cargo-diagnostic.log" "RAW_CARGO_DIAGNOSTIC_FAILURE"
echo "PASS: Cargo diagnostic failure output and counts are preserved"

run_check twelve_cores env FAKE_NPROC_OUTPUT=12 /usr/bin/bash scripts/check_all.sh --lane gpu-shared
assert_status twelve_cores 0
assert_contains "${tmp_dir}/twelve_cores.output" "configured_cap=num-cpus explicit_test_threads=12"
assert_contains "${tmp_dir}/twelve_cores.log" "cargo <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu)> <--no-fail-fast> <--test-threads> <12>"
assert_phase_contains "${tmp_dir}/twelve_cores.output" gpu-shared-tests "0.100s | PASS | run=2 passed=2 failed=0 skipped=6 | 12 | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/gpu-shared-tests.log"
echo "PASS: shared-GPU concurrency follows the available CPU count"

run_check cli_precedence env FEATURES=server TEST_RUNNER=cargo STRICT=1 \
  /usr/bin/bash scripts/check_all.sh --all-features --runner nextest --strict
assert_status cli_precedence 0
assert_contains "${tmp_dir}/cli_precedence.output" "runner=nextest(cli) features=all(cli) gpu_profile=default gpu_threads=16(nproc) strict=always(cli)"
assert_contains "${tmp_dir}/cli_precedence.log" "cargo <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu)> <--no-fail-fast> <--test-threads> <16>"
echo "PASS: validated CLI values take precedence over compatibility environment values"

run_check fast /usr/bin/bash scripts/check_all.sh --fast --features server
assert_status fast 0
assert_contains "${tmp_dir}/fast.output" "PARTIAL SURFACE: mode=fast lane=cpu"
assert_contains "${tmp_dir}/fast.output" "fast edit-loop surface passed"
assert_contains "${tmp_dir}/fast.log" "cargo <+nightly> <fmt> <--all> <--check>"
assert_contains "${tmp_dir}/fast.log" "cargo <nextest> <run> <--features> <server> <-E> <not group(gpu) & not group(gpu-exclusive)> <--no-fail-fast>"
assert_not_contains "${tmp_dir}/fast.log" "flock"
assert_not_contains "${tmp_dir}/fast.log" "<doc>"
assert_not_contains "${tmp_dir}/fast.log" "<clippy>"
assert_not_contains "${tmp_dir}/fast.log" "datafusion-nexus-tools"
assert_not_contains "${tmp_dir}/fast.log" "bash <scripts/ci/check_gpu_test_lane.sh>"
assert_phase_contains "${tmp_dir}/fast.output" cpu-tests "0.100s | PASS | run=3 passed=3 failed=0 skipped=5 | - | - | features:server | ${tmp_dir}/phase-logs/cpu-tests.log"
assert_phase_contains "${tmp_dir}/fast.output" live-inventory "- | SKIPPED | - | - | - | features:server | -"
assert_phase_contains "${tmp_dir}/fast.output" gpu-shared-tests "- | SKIPPED | - | - | - | features:server | -"
assert_phase_contains "${tmp_dir}/fast.output" doctests "- | SKIPPED | - | - | - | features:server | -"
echo "PASS: fast surface runs policies/fmt/structure/CPU without live inventory or GPU/doc/lint/tools"

run_check no_default_features /usr/bin/bash scripts/check_all.sh --lane cpu --no-default-features --features server
assert_status no_default_features 0
assert_contains "${tmp_dir}/no_default_features.log" "cargo <nextest> <run> <--no-default-features> <--features> <server>"

run_check full_narrow_features /usr/bin/bash scripts/check_all.sh --no-default-features --features server
assert_status full_narrow_features 0
assert_contains "${tmp_dir}/full_narrow_features.log" "cargo <check> <--no-default-features> <--features> <server>"
assert_contains "${tmp_dir}/full_narrow_features.log" "cargo <test> <--no-default-features> <--features> <server> <--doc> <--no-fail-fast>"
assert_contains "${tmp_dir}/full_narrow_features.log" "cargo <doc> <--no-default-features> <--features> <server> <--no-deps> <--keep-going>"
assert_contains "${tmp_dir}/full_narrow_features.log" "cargo <clippy> <--all-targets> <--no-default-features> <--features> <server>"
assert_phase_contains "${tmp_dir}/full_narrow_features.output" default-members "no-default+features:server | ${tmp_dir}/phase-logs/default-members.log"
assert_phase_contains "${tmp_dir}/full_narrow_features.output" clippy "no-default+features:server | ${tmp_dir}/phase-logs/clippy.log"
echo "PASS: feature selectors propagate consistently"

run_check lane_cpu /usr/bin/bash scripts/check_all.sh --lane cpu
assert_status lane_cpu 0
assert_contains "${tmp_dir}/lane_cpu.output" "selected lane 'cpu' passed. PARTIAL"
assert_contains "${tmp_dir}/lane_cpu.log" "cargo <nextest> <run>"
assert_not_contains "${tmp_dir}/lane_cpu.log" "flock"
assert_not_contains "${tmp_dir}/lane_cpu.log" "<clippy>"
assert_phase_contains "${tmp_dir}/lane_cpu.output" cpu-tests "0.100s | PASS | run=3 passed=3 failed=0 skipped=5 | - | - | all | ${tmp_dir}/phase-logs/cpu-tests.log"
assert_phase_contains "${tmp_dir}/lane_cpu.output" gpu-shared-tests "- | SKIPPED | - | - | - | all | -"

run_check lane_shared /usr/bin/bash scripts/check_all.sh --lane gpu-shared
assert_status lane_shared 0
assert_contains "${tmp_dir}/lane_shared.log" "flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu)> <--no-fail-fast> <--test-threads> <16>"
assert_not_contains "${tmp_dir}/lane_shared.log" "cargo <nextest> <run> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)>"
assert_not_contains "${tmp_dir}/lane_shared.log" "group(gpu-exclusive)"
assert_phase_contains "${tmp_dir}/lane_shared.output" gpu-shared-tests "0.100s | PASS | run=2 passed=2 failed=0 skipped=6 | 16 | /tmp/cudf-gpu.lock | all | ${tmp_dir}/phase-logs/gpu-shared-tests.log"

run_check lane_exclusive /usr/bin/bash scripts/check_all.sh --lane gpu-exclusive
assert_status lane_exclusive 0
assert_contains "${tmp_dir}/lane_exclusive.log" "flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--all-features> <-E> <group(gpu-exclusive)>"
assert_not_contains "${tmp_dir}/lane_exclusive.log" "cargo <nextest> <run> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)>"
assert_not_contains "${tmp_dir}/lane_exclusive.log" "<group(gpu)>"

run_check lane_gpu /usr/bin/bash scripts/check_all.sh --lane gpu
assert_status lane_gpu 0
if [[ $(grep -c '^flock ' "${tmp_dir}/lane_gpu.log") -ne 2 ]]; then
  echo "FAIL: --lane gpu must run exactly shared and exclusive locked lanes" >&2
  sed 's/^/  /' "${tmp_dir}/lane_gpu.log" >&2
  exit 1
fi
assert_not_contains "${tmp_dir}/lane_gpu.log" "cargo <nextest> <run> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)>"

run_check lane_all /usr/bin/bash scripts/check_all.sh --lane all
assert_status lane_all 0
assert_contains "${tmp_dir}/lane_all.output" "selected lane 'all' passed. PARTIAL"
assert_contains "${tmp_dir}/lane_all.log" "cargo <nextest> <run>"
if [[ $(grep -c '^flock ' "${tmp_dir}/lane_all.log") -ne 2 ]]; then
  echo "FAIL: --lane all must run exactly two locked GPU test lanes" >&2
  sed 's/^/  /' "${tmp_dir}/lane_all.log" >&2
  exit 1
fi
assert_not_contains "${tmp_dir}/lane_all.log" "<doc>"
assert_not_contains "${tmp_dir}/lane_all.log" "<clippy>"

run_check lane_policy /usr/bin/bash scripts/check_all.sh --lane policy
assert_status lane_policy 0
assert_contains "${tmp_dir}/lane_policy.output" "selected lane 'policy' passed. PARTIAL"
assert_contains "${tmp_dir}/lane_policy.log" "cargo <+nightly> <fmt> <--all> <--check>"
assert_not_contains "${tmp_dir}/lane_policy.log" "cargo <nextest>"
assert_not_contains "${tmp_dir}/lane_policy.log" "flock"
assert_phase_contains "${tmp_dir}/lane_policy.output" policy-selftests "0.100s | PASS | checks=7 | - | - | all | ${tmp_dir}/phase-logs/policy-selftests.log"
assert_phase_contains "${tmp_dir}/lane_policy.output" cpu-tests "- | SKIPPED | - | - | - | all | -"
echo "PASS: lane selection runs only the selected lane and necessary preflight"

run_check policy_failure env FAKE_POLICY_STATUS=9 /usr/bin/bash scripts/check_all.sh --lane policy
assert_status policy_failure 1
assert_contains "${tmp_dir}/policy_failure.output" "RAW_POLICY_FAILURE"
assert_phase_contains "${tmp_dir}/policy_failure.output" policy-selftests "0.100s | FAIL(1) | checks=7 | - | - | all | ${tmp_dir}/phase-logs/policy-selftests.log"
assert_contains "${tmp_dir}/phase-logs/policy-selftests.log" "RAW_POLICY_FAILURE"
assert_phase_contains "${tmp_dir}/policy_failure.output" rustfmt "0.100s | PASS"
echo "PASS: grouped policy failure is logged while later safe phases continue"

run_check lane_cpu_failure env FAKE_CPU_TEST_STATUS=42 /usr/bin/bash scripts/check_all.sh --lane cpu
assert_status lane_cpu_failure 1
assert_contains "${tmp_dir}/lane_cpu_failure.output" "RAW_CPU_FAILURE"
assert_phase_contains "${tmp_dir}/lane_cpu_failure.output" cpu-tests "0.100s | FAIL(42) | run=3 passed=2 failed=1 skipped=5 | - | - | all | ${tmp_dir}/phase-logs/cpu-tests.log"
assert_contains "${tmp_dir}/phase-logs/cpu-tests.log" "RAW_CPU_FAILURE"

run_check missing_cpu_summary env FAKE_CPU_SUMMARY_MISSING=1 /usr/bin/bash scripts/check_all.sh --lane cpu
assert_status missing_cpu_summary 1
assert_contains "${tmp_dir}/missing_cpu_summary.output" "expected exactly one nextest Summary line, found 0"
assert_phase_contains "${tmp_dir}/missing_cpu_summary.output" cpu-tests "0.100s | FAIL(accounting) | unavailable | - | - | all | ${tmp_dir}/phase-logs/cpu-tests.log"
echo "PASS: phase failures preserve output and test-count accounting fails closed"

run_check live_inventory_failure env FAKE_LIVE_INVENTORY_STATUS=42 /usr/bin/bash scripts/check_all.sh --lane all
assert_status live_inventory_failure 1
assert_contains "${tmp_dir}/live_inventory_failure.output" "RAW_LIVE_INVENTORY_FAILURE"
assert_phase_contains "${tmp_dir}/live_inventory_failure.output" live-inventory "0.100s | FAIL(42) | - | - | - | all | ${tmp_dir}/phase-logs/live-inventory.log"
assert_phase_contains "${tmp_dir}/live_inventory_failure.output" cpu-tests "- | SKIPPED(blocked) | - | - | - | all | -"
assert_phase_contains "${tmp_dir}/live_inventory_failure.output" gpu-shared-tests "- | SKIPPED(blocked) | - | - | - | all | -"
assert_contains "${tmp_dir}/phase-logs/live-inventory.log" "RAW_LIVE_INVENTORY_FAILURE"
echo "PASS: failed live inventory is logged and blocks selected lanes"

run_check cargo_lane_conflict /usr/bin/bash scripts/check_all.sh --runner cargo-diagnostic --lane cpu
assert_status cargo_lane_conflict 2
assert_contains "${tmp_dir}/cargo_lane_conflict.output" "cargo-diagnostic cannot select individual test lanes"
assert_empty "${tmp_dir}/cargo_lane_conflict.log"
echo "PASS: diagnostic Cargo runner rejects unsupported lane selection"

run_check show_config_drift env FAKE_GPU_CAP=5 /usr/bin/bash scripts/check_all.sh
assert_status show_config_drift 2
assert_contains "${tmp_dir}/show_config_drift.output" "show-config reports cap '5'"
assert_contains "${tmp_dir}/show_config_drift.output" "expected dynamic cap 'num-cpus'"
assert_phase_contains "${tmp_dir}/show_config_drift.output" gpu-preflight "0.100s | FAIL(2) | - | 16,1 | - | all | ${tmp_dir}/phase-logs/gpu-preflight.log"
assert_phase_contains "${tmp_dir}/show_config_drift.output" gpu-shared-tests "- | SKIPPED(blocked) | - | - | - | all | -"
echo "PASS: shared show-config drift fails closed"

run_check exclusive_show_config_drift env FAKE_GPU_EXCLUSIVE_CAP=2 /usr/bin/bash scripts/check_all.sh
assert_status exclusive_show_config_drift 2
assert_contains "${tmp_dir}/exclusive_show_config_drift.output" "show-config reports cap 2 for group 'gpu-exclusive'"
echo "PASS: exclusive show-config drift fails closed"

# test_all.sh must not inject different defaults or otherwise change semantics.
run_check alias /usr/bin/bash scripts/ci/test_all.sh
assert_status alias 0
diff -u "${tmp_dir}/default.output" "${tmp_dir}/alias.output"
diff -u "${tmp_dir}/default.log" "${tmp_dir}/alias.log"
echo "PASS: test_all is a semantic alias"

echo "OK: GPU scheduling policy tests passed without running GPU tests."
