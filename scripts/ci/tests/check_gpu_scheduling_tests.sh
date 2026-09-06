#!/usr/bin/env bash
# Isolated scheduling-policy tests. All Cargo/flock/sub-check invocations are
# faked: this script never executes a test binary and never opens a GPU context.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-gpu-scheduling.XXXXXX")
mutation_fixture="${ROOT}/.source_mutation_fixture.$$"
trap 'rm -f -- "${mutation_fixture}"; rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin" "${tmp_dir}/phase-logs"

cat >"${tmp_dir}/bin/bash" <<'SH'
#!/usr/bin/bash
case "${1:-}" in
  */scripts/check_all.sh) ;;
  *)
    # Single append: background gate phases share this log.
    bash_log_line="bash"
    for bash_log_arg in "$@"; do
      bash_log_line="${bash_log_line} <${bash_log_arg}>"
    done
    printf '%s\n' "${bash_log_line}" >>"${COMMAND_LOG}"
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
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --inventory-out | --cargo-metadata-out) printf '{}\n' >"$2"; shift 2 ;;
        --lanes-out)
          [[ "${FAKE_SKIP_LANES_FILE:-0}" -eq 1 ]] || printf '%s\n' "${FAKE_SELECTED_LANES-cpu}" >"$2"
          shift 2
          ;;
        *) shift ;;
      esac
    done
    exit "${FAKE_LIVE_INVENTORY_STATUS:-0}"
    ;;
  scripts/ci/tests/check_cpu_packaging_tests.sh | scripts/ci/check_native_sources.sh | scripts/ci/check_component_packages.sh | scripts/dev/bindings.sh | scripts/ci/tests/check_dependency_direction_tests.sh | scripts/ci/tests/check_doctest_report_tests.sh | scripts/ci/tests/check_phase_test_counts_tests.sh | scripts/ci/tests/check_phase_failed_tests_tests.sh | scripts/ci/tests/check_gpu_test_lane_tests.sh | scripts/ci/tests/check_native_sources_tests.sh | scripts/ci/tests/check_doctest_rustdoc_tests.sh | scripts/ci/tests/check_gpu_scheduling_tests.sh | scripts/ci/tests/check_clean_target_artifacts_tests.sh | scripts/ci/tests/check_pass_as_skip_policy_tests.sh | scripts/ci/tests/check_public_api_surface_tests.sh)
    exit 0
    ;;
  scripts/ci/check_public_api_surface.sh)
    if [[ "${FAKE_PUBLIC_API_STATUS:-0}" -ne 0 ]]; then
      echo "RAW_PUBLIC_API_FAILURE" >&2
    fi
    exit "${FAKE_PUBLIC_API_STATUS:-0}"
    ;;
esac
exec /usr/bin/bash "$@"
SH

cat >"${tmp_dir}/bin/cargo" <<'SH'
#!/usr/bin/bash
set -u
# Single append: background gate phases share this log, so one write keeps
# each command line whole.
cargo_log_line="cargo"
for cargo_log_arg in "$@"; do
  cargo_log_line="${cargo_log_line} <${cargo_log_arg}>"
done
cargo_log_line="${cargo_log_line} {rustdocflags=${RUSTDOCFLAGS-unset}}"
printf '%s\n' "${cargo_log_line}" >>"${COMMAND_LOG}"

if [[ "${1:-}" == "nextest" && "${2:-}" == "--version" ]]; then
  [[ -z "${FAKE_NEXTEST_VERSION_STDERR:-}" ]] || echo "${FAKE_NEXTEST_VERSION_STDERR}" >&2
  echo "cargo-nextest ${FAKE_NEXTEST_VERSION:-0.9.140}"
  exit "${FAKE_NEXTEST_VERSION_STATUS:-0}"
fi
if [[ "${1:-}" == "hack" && "${2:-}" == "--version" ]]; then
  [[ -z "${FAKE_HACK_VERSION_STDERR:-}" ]] || echo "${FAKE_HACK_VERSION_STDERR}" >&2
  echo "${FAKE_HACK_VERSION_OUTPUT:-cargo-hack 0.6.45}"
  exit "${FAKE_HACK_VERSION_STATUS:-0}"
fi
if [[ "${FAKE_MUTATE_WORKTREE:-0}" -eq 1 ]]; then
  printf "mutation\n" >>"${FAKE_MUTATE_TARGET}"
fi
if [[ "${1:-}" == "hack" && "${2:-}" == "check" ]]; then
  if [[ " $* " == *" --print-command-list "* ]]; then
    # Union enumeration: replay the -p sets recorded by the shard runs, then
    # consume the record so the next scenario starts clean.
    while IFS= read -r hack_seen_pkg; do
      [[ -n "${hack_seen_pkg}" ]] || continue
      printf 'cargo check --locked --manifest-path fake-pkgs/%s/Cargo.toml\n' "${hack_seen_pkg}"
    done <"${FAKE_HACK_SEEN}" | LC_ALL=C sort
    if [[ "${FAKE_HACK_CORRUPT_UNION:-0}" -eq 1 ]]; then
      printf 'cargo check --locked --manifest-path fake-pkgs/BOGUS/Cargo.toml\n'
    fi
    : >"${FAKE_HACK_SEEN}"
    exit 0
  fi
  if [[ " $* " == *" --each-feature "* ]]; then
    # One shard run: record its packages and emit the matching check lines so
    # the union verification has the same input shape as production.
    hack_shard_pkgs=()
    hack_shard_prev=""
    for hack_shard_arg in "$@"; do
      if [[ "${hack_shard_prev}" == "-p" ]]; then
        hack_shard_pkgs+=("${hack_shard_arg}")
        printf '%s\n' "${hack_shard_arg}" >>"${FAKE_HACK_SEEN}"
      fi
      hack_shard_prev="${hack_shard_arg}"
    done
    hack_shard_total=${#hack_shard_pkgs[@]}
    hack_shard_index=0
    for hack_shard_pkg in "${hack_shard_pkgs[@]}"; do
      hack_shard_index=$((hack_shard_index + 1))
      printf 'info: running `cargo check --locked --manifest-path fake-pkgs/%s/Cargo.toml` on %s (%d/%d)\n' \
        "${hack_shard_pkg}" "${hack_shard_pkg}" "${hack_shard_index}" "${hack_shard_total}"
    done
    if [[ "${FAKE_FEATURE_ISOLATION_STATUS:-0}" -ne 0 ]]; then
      echo "RAW_FEATURE_ISOLATION_FAILURE" >&2
    fi
    exit "${FAKE_FEATURE_ISOLATION_STATUS:-0}"
  fi
  if [[ "${FAKE_MSRV_STATUS:-0}" -ne 0 ]]; then
    echo "RAW_MSRV_FAILURE" >&2
  fi
  exit "${FAKE_MSRV_STATUS:-0}"
fi
if [[ "${1:-}" == "clippy" ]]; then
  [[ "${FAKE_CLIPPY_STATUS:-0}" -eq 0 ]] || echo "RAW_CLIPPY_FAILURE" >&2
  exit "${FAKE_CLIPPY_STATUS:-0}"
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "show-config" ]]; then
  echo "group: gpu (max threads = ${FAKE_GPU_CAP:-num-cpus})"
  echo "group: gpu-exclusive (max threads = ${FAKE_GPU_EXCLUSIVE_CAP:-1})"
  exit "${FAKE_SHOW_CONFIG_STATUS:-0}"
fi
if [[ "${1:-}" == "metadata" ]]; then
  cat <<'JSON'
{"packages":[{"id":"path+file:///fake#nexus-iceberg@0.1.0","name":"nexus-iceberg"}],"resolve":{"nodes":[{"id":"path+file:///fake#nexus-iceberg@0.1.0","deps":[]}]}}
JSON
  exit 0
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "run" ]]; then
  if [[ " $* " == *" not group(gpu) & not group(gpu-exclusive) "* ]]; then
    if [[ "${FAKE_CPU_SUMMARY_MISSING:-0}" -eq 1 ]]; then
      echo "CPU_OUTPUT_WITHOUT_SUMMARY"
      exit 0
    fi
    if [[ "${FAKE_CPU_TEST_STATUS:-0}" -ne 0 ]]; then
      echo "RAW_CPU_FAILURE" >&2
      echo "FAIL [   0.001s] fake_cpu::broke"
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
  echo "running 2 tests"
  echo "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
  exit 0
fi
exit 0
SH

cat >"${tmp_dir}/bin/date" <<'SH'
#!/usr/bin/bash
# Constant output: parallel gate phases share the clock, so an incrementing
# fake would make phase durations schedule-dependent instead of deterministic.
printf '100\n'
SH

cat >"${tmp_dir}/bin/nproc" <<'SH'
#!/usr/bin/bash
printf '%s\n' "${FAKE_NPROC_OUTPUT:-16}"
exit "${FAKE_NPROC_STATUS:-0}"
SH

cat >"${tmp_dir}/bin/git" <<'SH'
#!/usr/bin/bash
set -u
if [[ "${FAKE_GITLINK_STATE_ACTIVE:-0}" -eq 1 && "${1:-}" == "-C" && \
  "${2:-}" == "${FAKE_GIT_ROOT:-}" && "${3:-}" == "submodule" && \
  "${4:-}" == "status" && "${5:-}" == "--recursive" ]]; then
  printf '%s\n' "${FAKE_GITLINK_STATUS}"
  exit 0
fi
exec /usr/bin/git "$@"
SH

cat >"${tmp_dir}/bin/flock" <<'SH'
#!/usr/bin/bash
set -u
flock_log_line="flock"
for flock_log_arg in "$@"; do
  flock_log_line="${flock_log_line} <${flock_log_arg}>"
done
printf '%s\n' "${flock_log_line}" >>"${COMMAND_LOG}"
shift
"$@"
SH
chmod +x "${tmp_dir}/bin/bash" "${tmp_dir}/bin/cargo" "${tmp_dir}/bin/date" \
  "${tmp_dir}/bin/flock" "${tmp_dir}/bin/git" "${tmp_dir}/bin/nproc"

run_check_with_phase_log_dir() {
  local name=$1
  local phase_log_dir=$2
  shift 2
  local output="${tmp_dir}/${name}.output"
  local log="${tmp_dir}/${name}.log"
  local status=0
  : >"${log}"
  : >"${tmp_dir}/${name}.date"
  : >"${tmp_dir}/hack-seen"
  env -u FEATURES -u TEST_RUNNER -u GPU_LOCK -u STRICT -u RUSTDOCFLAGS \
    PATH="${tmp_dir}/bin:${PATH}" COMMAND_LOG="${log}" FAKE_GIT_ROOT="${ROOT}" \
    CHECK_ALL_PHASE_LOG_DIR="${phase_log_dir}" FAKE_DATE_COUNTER="${tmp_dir}/${name}.date" \
    FAKE_HACK_SEEN="${tmp_dir}/hack-seen" \
    "$@" >"${output}" 2>&1 || status=$?
  printf '%s\n' "${status}" >"${tmp_dir}/${name}.status"
}

run_check() {
  local name=$1
  shift
  run_check_with_phase_log_dir "${name}" "${tmp_dir}/phase-logs" "$@"
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
  # Anchor on the leading phase-id cell so a shorter phase id cannot match a
  # longer one that merely starts with it.
  row=$(grep -E "^${phase} \| " "${file}" | tail -n 1) || {
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
assert config["profile"]["default"]["final-status-level"] == "skip"
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

echo "PASS: unknown and duplicate CLI options fail before work"

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

run_check wrong_cargo_hack env FAKE_HACK_VERSION_OUTPUT='cargo-hack 0.6.44' /usr/bin/bash scripts/check_all.sh
assert_status wrong_cargo_hack 2
assert_contains "${tmp_dir}/wrong_cargo_hack.output" "require exactly 'cargo-hack 0.6.45'"
assert_contains "${tmp_dir}/wrong_cargo_hack.output" "cargo install --locked --version 0.6.45 cargo-hack"
assert_not_contains "${tmp_dir}/wrong_cargo_hack.log" "bash <scripts/"

run_check malformed_cargo_hack env FAKE_HACK_VERSION_OUTPUT='cargo hack 0.6.45' /usr/bin/bash scripts/check_all.sh
assert_status malformed_cargo_hack 2
assert_contains "${tmp_dir}/malformed_cargo_hack.output" "require exactly 'cargo-hack 0.6.45'"
assert_not_contains "${tmp_dir}/malformed_cargo_hack.log" "bash <scripts/"

run_check missing_cargo_hack env FAKE_HACK_VERSION_STATUS=127 FAKE_HACK_VERSION_STDERR=missing-cargo-hack /usr/bin/bash scripts/check_all.sh
assert_status missing_cargo_hack 2
assert_contains "${tmp_dir}/missing_cargo_hack.output" "missing-cargo-hack"
assert_contains "${tmp_dir}/missing_cargo_hack.output" "cargo install --locked --version 0.6.45 cargo-hack"
assert_not_contains "${tmp_dir}/missing_cargo_hack.log" "bash <scripts/"
echo "PASS: full compatibility checks require exact cargo-hack 0.6.45 before work"

for partial_args in \
  '--fast' \
  '--lane all' \
  '--lane cpu' \
  '--lane gpu' \
  '--lane gpu-shared' \
  '--lane gpu-exclusive' \
  '--lane policy'; do
  partial_name="cargo_hack_optional_${partial_args// /_}"
  read -r -a partial_argv <<<"${partial_args}"
  run_check "${partial_name}" env FAKE_HACK_VERSION_STATUS=127 /usr/bin/bash scripts/check_all.sh "${partial_argv[@]}"
  assert_status "${partial_name}" 0
  assert_not_contains "${tmp_dir}/${partial_name}.log" "cargo <hack>"
done
echo "PASS: fast and every explicit lane remain independent of cargo-hack"

# Default mode must report and pass all 16 available cores. The fake commands
# make this a no-GPU orchestration test.
run_check default /usr/bin/bash scripts/check_all.sh
assert_status default 0
assert_contains "${tmp_dir}/default.output" "gpu_profile=default gpu_threads=16(nproc)"
assert_contains "${tmp_dir}/default.output" "CHECK SOURCE START: root="
assert_contains "${tmp_dir}/default.output" "gitlinks="
assert_not_contains "${tmp_dir}/default.output" "cudf_nexus="
assert_not_contains "${tmp_dir}/default.output" "cugraph_nexus="
assert_contains "${tmp_dir}/default.output" "CHECK SOURCE END: root="
assert_contains "${tmp_dir}/default.output" "GPU scheduling preflight: shared: profile=default group=gpu configured_cap=num-cpus explicit_test_threads=16"
assert_contains "${tmp_dir}/default.output" "exclusive: group=gpu-exclusive configured_cap=1 explicit_test_threads=1"
assert_contains "${tmp_dir}/default.output" "GPU scheduling final: shared: profile=default group=gpu configured_cap=num-cpus explicit_test_threads=16"
assert_contains "${tmp_dir}/default.log" "cargo <nextest> <show-config> <--color> <never> <-P> <default> <test-groups> <--cargo-metadata>"
assert_contains "${tmp_dir}/default.log" "cargo <hack> <check> <--workspace> <--all-features> <--locked> <--rust-version> <--workspace-behavior=cargo>"
if [[ $(grep -cF 'cargo <hack> <check> <--workspace> <--all-features> <--locked> <--rust-version> <--workspace-behavior=cargo>' "${tmp_dir}/default.log") -ne 1 ]]; then
  echo "FAIL: full mode must issue exactly one consolidated MSRV command" >&2
  sed 's/^/  /' "${tmp_dir}/default.log" >&2
  exit 1
fi
# Feature-isolation runs as six package shards with isolated target dirs; the
# workspace-wide each-feature form appears only as the union-verification
# enumeration.
for feature_shard in \
  "cargo <hack> <check> <-p> <nexus-tools> <--each-feature> <--exclude-features> <default> <--locked> <--keep-going>" \
  "cargo <hack> <check> <-p> <datafusion-nexus> <-p> <nexus-workloads> <-p> <datafusion-facade-consumer> <-p> <rapids-interop> <--each-feature> <--exclude-features> <default> <--locked> <--keep-going>" \
  "cargo <hack> <check> <-p> <nexus-query-runtime> <-p> <nexus-query-planning> <--each-feature> <--exclude-features> <default> <--locked> <--keep-going>" \
  "cargo <hack> <check> <-p> <nexus-query-engine> <-p> <cudf-nexus> <-p> <cudf-nexus-sys> <-p> <nexus-build-support> <--each-feature> <--exclude-features> <default> <--locked> <--keep-going>" \
  "cargo <hack> <check> <-p> <nexus-iceberg> <-p> <nexus-server> <-p> <cugraph-nexus-sys> <--each-feature> <--exclude-features> <default> <--locked> <--keep-going>" \
  "cargo <hack> <check> <-p> <nexus-examples> <-p> <nexus-bench> <-p> <cugraph-nexus> <-p> <cuvs-nexus> <-p> <cuvs-nexus-sys> <--each-feature> <--exclude-features> <default> <--locked> <--keep-going>"; do
  assert_contains "${tmp_dir}/default.log" "${feature_shard}"
done
if [[ $(grep -c 'cargo <hack> <check> <-p>.*--each-feature' "${tmp_dir}/default.log") -ne 6 ]]; then
  echo "FAIL: full mode must issue exactly six feature-isolation shard commands" >&2
  sed 's/^/  /' "${tmp_dir}/default.log" >&2
  exit 1
fi
feature_isolation_command=$(grep -F 'cargo <hack> <check> <--workspace> <--each-feature>' "${tmp_dir}/default.log")
if [[ "${feature_isolation_command}" != 'cargo <hack> <check> <--workspace> <--each-feature> <--exclude-features> <default> <--locked> <--keep-going> <--print-command-list> {rustdocflags=unset}' ]]; then
  echo "FAIL: workspace-wide each-feature must run only as the union-verification enumeration: ${feature_isolation_command}" >&2
  exit 1
fi
assert_contains "${tmp_dir}/default.log" "cargo <nextest> <run> <--cargo-metadata> <${tmp_dir}/phase-logs/cargo-metadata.json> <--binaries-metadata> <${tmp_dir}/phase-logs/binaries-metadata.json> <-E> <not group(gpu) & not group(gpu-exclusive)> <--no-fail-fast>"
assert_contains "${tmp_dir}/default.log" "cargo <nextest> <run> <-P> <default> <--cargo-metadata> <${tmp_dir}/phase-logs/cargo-metadata.json> <--binaries-metadata> <${tmp_dir}/phase-logs/binaries-metadata.json> <-E> <group(gpu)>"
assert_contains "${tmp_dir}/default.log" "cargo <test> <--workspace> <--all-features> <--doc> <--no-fail-fast>"
assert_contains "${tmp_dir}/default.log" "cargo <+nightly> <fmt> <--all> <--check>"
assert_contains "${tmp_dir}/default.log" "cargo <clippy> <--workspace> <--all-targets> <--all-features>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/check_public_api_surface.sh> <check> <--all-features>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_cpu_packaging_tests.sh>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_dependency_direction_tests.sh>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_gpu_test_lane_tests.sh>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_doctest_rustdoc_tests.sh>"
assert_contains "${tmp_dir}/default.log" "bash <scripts/ci/tests/check_gpu_scheduling_tests.sh>"
assert_contains "${tmp_dir}/default.output" "PHASE SUMMARY (18 phases: 18 PASS, 0 FAIL, 0 SKIPPED)"
assert_contains "${tmp_dir}/default.output" "Phase | Status | Duration | Counts"
assert_not_contains "${tmp_dir}/default.output" "FAILURES"
assert_contains "${tmp_dir}/default.output" "==> [gpu-shared-tests] flock /tmp/cudf-gpu.lock cargo nextest run -P default --cargo-metadata"
assert_contains "${tmp_dir}/default.output" "==> [gpu-exclusive-tests] flock /tmp/cudf-gpu.lock cargo nextest run -P default --cargo-metadata"
assert_contains "${tmp_dir}/default.output" "==> [public-api-surface] bash scripts/ci/check_public_api_surface.sh check --all-features"
assert_phase_contains "${tmp_dir}/default.output" policy-selftests "policy-selftests | PASS | 0.000s | checks=14"
assert_phase_contains "${tmp_dir}/default.output" msrv-check "msrv-check | PASS | 0.000s | -"
assert_phase_contains "${tmp_dir}/default.output" feature-isolation-check "feature-isolation-check | PASS | 0.000s | -"
assert_phase_contains "${tmp_dir}/default.output" cpu-tests "cpu-tests | PASS | 0.000s | run=3 passed=3 failed=0 skipped=5"
assert_phase_contains "${tmp_dir}/default.output" gpu-shared-tests "gpu-shared-tests | PASS | 0.000s | run=2 passed=2 failed=0 skipped=6"
assert_phase_contains "${tmp_dir}/default.output" gpu-exclusive-tests "gpu-exclusive-tests | PASS | 0.000s | run=1 passed=1 failed=0 skipped=7"
assert_phase_contains "${tmp_dir}/default.output" doctests "doctests | PASS | 0.000s | harnesses=1 executed=0 passed=0 failed=0 ignored=0 measured=0 filtered=0"
assert_phase_contains "${tmp_dir}/default.output" public-api-surface "public-api-surface | PASS | 0.000s | crates=8"
for phase_log in policy-selftests native-sources checked-bindings component-packages native-free-check msrv-check feature-isolation-check cpu-tests gpu-shared-tests gpu-exclusive-tests doctests rustdoc public-api-surface clippy; do
  [[ -f "${tmp_dir}/phase-logs/${phase_log}.log" ]] || {
    echo "FAIL: expected persistent phase log ${phase_log}.log" >&2
    exit 1
  }
done
default_shared_flock="flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--cargo-metadata> <${tmp_dir}/phase-logs/cargo-metadata.json> <--binaries-metadata> <${tmp_dir}/phase-logs/binaries-metadata.json> <-E> <group(gpu)>"
default_exclusive_flock="flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--cargo-metadata> <${tmp_dir}/phase-logs/cargo-metadata.json> <--binaries-metadata> <${tmp_dir}/phase-logs/binaries-metadata.json> <-E> <group(gpu-exclusive)>"
assert_before "${tmp_dir}/default.log" "${default_shared_flock}" "${default_exclusive_flock}"
default_doctest_flock="flock </tmp/cudf-gpu.lock> <cargo> <test> <--workspace> <--all-features> <--doc> <--no-fail-fast>"
assert_before "${tmp_dir}/default.log" "${default_exclusive_flock}" "${default_doctest_flock}"
if [[ $(grep -c '^flock ' "${tmp_dir}/default.log") -ne 3 ]]; then
  echo "FAIL: default scheduling must lock shared, exclusive, and doctest phases" >&2
  sed 's/^/  /' "${tmp_dir}/default.log" >&2
  exit 1
fi
# Parallel barriers, not launch order: the lane gate closes before inventory,
# the isolated-target bag is collected before the test lanes run, and the
# static chain is collected before doctests take the main target directory.
assert_before "${tmp_dir}/default.output" \
  "==> [lane-structure] PASS" \
  "==> [live-inventory]"
assert_before "${tmp_dir}/default.output" \
  "==> [feature-isolation-check] PASS" \
  "==> [cpu-tests]"
assert_before "${tmp_dir}/default.output" \
  "==> [clippy] PASS" \
  "==> [doctests]"
echo "PASS: unified nextest CPU/GPU component scheduling"
echo "PASS: full surface includes pinned fmt, one workspace-wide test/doc/clippy unification, and isolated policies"


run_check twelve_cores env FAKE_NPROC_OUTPUT=12 /usr/bin/bash scripts/check_all.sh --lane gpu-shared
assert_status twelve_cores 0
assert_contains "${tmp_dir}/twelve_cores.output" "configured_cap=num-cpus explicit_test_threads=12"
assert_contains "${tmp_dir}/twelve_cores.log" "cargo <nextest> <run> <-P> <default> <--cargo-metadata> <${tmp_dir}/phase-logs/cargo-metadata.json> <--binaries-metadata> <${tmp_dir}/phase-logs/binaries-metadata.json> <-E> <group(gpu)>"
assert_phase_contains "${tmp_dir}/twelve_cores.output" gpu-shared-tests "gpu-shared-tests | PASS | 0.000s | run=2 passed=2 failed=0 skipped=6"
echo "PASS: shared-GPU concurrency follows the available CPU count"


run_check fast /usr/bin/bash scripts/check_all.sh --fast
assert_status fast 0
assert_contains "${tmp_dir}/fast.output" "PARTIAL SURFACE: mode=fast lane=cpu"
assert_contains "${tmp_dir}/fast.output" "fast edit-loop surface passed"
assert_contains "${tmp_dir}/fast.log" "cargo <+nightly> <fmt> <--all> <--check>"
assert_contains "${tmp_dir}/fast.log" "cargo <nextest> <run> <--workspace> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)> <--no-fail-fast>"
assert_not_contains "${tmp_dir}/fast.log" "flock"
assert_not_contains "${tmp_dir}/fast.log" "<doc>"
assert_not_contains "${tmp_dir}/fast.log" "<clippy>"
assert_not_contains "${tmp_dir}/fast.log" "nexus-tools"
assert_not_contains "${tmp_dir}/fast.log" "nexus-bench"
assert_not_contains "${tmp_dir}/fast.log" "bash <scripts/ci/check_gpu_test_lane.sh>"
assert_phase_contains "${tmp_dir}/fast.output" cpu-tests "cpu-tests | PASS | 0.000s | run=3 passed=3 failed=0 skipped=5"
assert_phase_contains "${tmp_dir}/fast.output" live-inventory "live-inventory | SKIPPED | - | -"
assert_phase_contains "${tmp_dir}/fast.output" gpu-shared-tests "gpu-shared-tests | SKIPPED | - | -"
assert_phase_contains "${tmp_dir}/fast.output" doctests "doctests | SKIPPED | - | -"
echo "PASS: fast surface runs policies/fmt/structure/CPU without live inventory or GPU/doc/lint/tools/bench"


run_check lane_cpu /usr/bin/bash scripts/check_all.sh --lane cpu
assert_status lane_cpu 0
assert_contains "${tmp_dir}/lane_cpu.output" "selected lane 'cpu' passed. PARTIAL"
assert_contains "${tmp_dir}/lane_cpu.log" "cargo <nextest> <run>"
assert_not_contains "${tmp_dir}/lane_cpu.log" "flock"
assert_not_contains "${tmp_dir}/lane_cpu.log" "<clippy>"
assert_phase_contains "${tmp_dir}/lane_cpu.output" cpu-tests "cpu-tests | PASS | 0.000s | run=3 passed=3 failed=0 skipped=5"
assert_phase_contains "${tmp_dir}/lane_cpu.output" gpu-shared-tests "gpu-shared-tests | SKIPPED | - | -"

run_check lane_shared /usr/bin/bash scripts/check_all.sh --lane gpu-shared
assert_status lane_shared 0
assert_contains "${tmp_dir}/lane_shared.log" "flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--cargo-metadata>"
assert_not_contains "${tmp_dir}/lane_shared.log" "cargo <nextest> <run> <--workspace> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)>"
assert_not_contains "${tmp_dir}/lane_shared.log" "group(gpu-exclusive)"
assert_not_contains "${tmp_dir}/lane_shared.log" "nexus-bench"
assert_phase_contains "${tmp_dir}/lane_shared.output" gpu-shared-tests "gpu-shared-tests | PASS | 0.000s | run=2 passed=2 failed=0 skipped=6"

run_check lane_exclusive /usr/bin/bash scripts/check_all.sh --lane gpu-exclusive
assert_status lane_exclusive 0
assert_contains "${tmp_dir}/lane_exclusive.log" "flock </tmp/cudf-gpu.lock> <cargo> <nextest> <run> <-P> <default> <--cargo-metadata>"
assert_not_contains "${tmp_dir}/lane_exclusive.log" "cargo <nextest> <run> <--workspace> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)>"
assert_not_contains "${tmp_dir}/lane_exclusive.log" "<group(gpu)>"

run_check lane_gpu /usr/bin/bash scripts/check_all.sh --lane gpu
assert_status lane_gpu 0
if [[ $(grep -c '^flock ' "${tmp_dir}/lane_gpu.log") -ne 2 ]]; then
  echo "FAIL: --lane gpu must run shared and exclusive nextest phases" >&2
  sed 's/^/  /' "${tmp_dir}/lane_gpu.log" >&2
  exit 1
fi
assert_not_contains "${tmp_dir}/lane_gpu.log" "cargo <nextest> <run> <--workspace> <--all-features> <-E> <not group(gpu) & not group(gpu-exclusive)>"

run_check lane_all /usr/bin/bash scripts/check_all.sh --lane all
assert_status lane_all 0
assert_contains "${tmp_dir}/lane_all.output" "selected lane 'all' passed. PARTIAL"
assert_contains "${tmp_dir}/lane_all.log" "cargo <nextest> <run>"
if [[ $(grep -c '^flock ' "${tmp_dir}/lane_all.log") -ne 2 ]]; then
  echo "FAIL: --lane all must run shared and exclusive nextest phases" >&2
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
assert_phase_contains "${tmp_dir}/lane_policy.output" policy-selftests "policy-selftests | PASS | 0.000s | checks=14"
assert_phase_contains "${tmp_dir}/lane_policy.output" cpu-tests "cpu-tests | SKIPPED | - | -"
echo "PASS: lane selection runs only the selected lane and necessary preflight"

run_check policy_failure env FAKE_POLICY_STATUS=9 /usr/bin/bash scripts/check_all.sh --lane policy
assert_status policy_failure 1
assert_contains "${tmp_dir}/policy_failure.output" "RAW_POLICY_FAILURE"
assert_phase_contains "${tmp_dir}/policy_failure.output" policy-selftests "policy-selftests | FAIL(1) | 0.000s | checks=14"
assert_contains "${tmp_dir}/phase-logs/policy-selftests.log" "RAW_POLICY_FAILURE"
assert_phase_contains "${tmp_dir}/policy_failure.output" rustfmt "rustfmt | PASS | 0.000s | -"
assert_contains "${tmp_dir}/policy_failure.output" "--- policy-selftests [FAIL(1), 0.000s] ---"
assert_contains "${tmp_dir}/policy_failure.output" "command: 14 isolated policy/orchestration scripts"
assert_contains "${tmp_dir}/policy_failure.output" "log: ${tmp_dir}/phase-logs/policy-selftests.log"
assert_not_contains "${tmp_dir}/policy_failure.output" "failed tests"
echo "PASS: grouped policy failure is logged while later safe phases continue"

run_check lane_cpu_failure env FAKE_CPU_TEST_STATUS=42 /usr/bin/bash scripts/check_all.sh --lane cpu
assert_status lane_cpu_failure 1
assert_contains "${tmp_dir}/lane_cpu_failure.output" "RAW_CPU_FAILURE"
assert_phase_contains "${tmp_dir}/lane_cpu_failure.output" cpu-tests "cpu-tests | FAIL(42) | 0.000s | run=3 passed=2 failed=1 skipped=5"
assert_contains "${tmp_dir}/phase-logs/cpu-tests.log" "RAW_CPU_FAILURE"
assert_contains "${tmp_dir}/lane_cpu_failure.output" "--- cpu-tests [FAIL(42), 0.000s] ---"
assert_contains "${tmp_dir}/lane_cpu_failure.output" "command: cargo nextest run --cargo-metadata ${tmp_dir}/phase-logs/cargo-metadata.json --binaries-metadata ${tmp_dir}/phase-logs/binaries-metadata.json -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast --final-status-level fail"
assert_contains "${tmp_dir}/lane_cpu_failure.output" "log: ${tmp_dir}/phase-logs/cpu-tests.log"
assert_contains "${tmp_dir}/lane_cpu_failure.output" "failed tests (1):"
assert_contains "${tmp_dir}/lane_cpu_failure.output" "fake_cpu::broke"

run_check missing_cpu_summary env FAKE_CPU_SUMMARY_MISSING=1 /usr/bin/bash scripts/check_all.sh --lane cpu
assert_status missing_cpu_summary 1
assert_contains "${tmp_dir}/missing_cpu_summary.output" "expected exactly one nextest Summary line, found 0"
assert_phase_contains "${tmp_dir}/missing_cpu_summary.output" cpu-tests "cpu-tests | FAIL(accounting) | 0.000s | unavailable"
assert_contains "${tmp_dir}/missing_cpu_summary.output" "--- cpu-tests [FAIL(accounting), 0.000s] ---"
assert_contains "${tmp_dir}/missing_cpu_summary.output" "failed tests: none listed (see log)"
echo "PASS: phase failures preserve output and test-count accounting fails closed"

run_check feature_isolation_failure env FAKE_FEATURE_ISOLATION_STATUS=42 /usr/bin/bash scripts/check_all.sh
assert_status feature_isolation_failure 1
assert_contains "${tmp_dir}/phase-logs/feature-isolation-check.log" "RAW_FEATURE_ISOLATION_FAILURE"
assert_phase_contains "${tmp_dir}/feature_isolation_failure.output" feature-isolation-check "feature-isolation-check | FAIL(42) | 0.000s | -"
assert_phase_contains "${tmp_dir}/feature_isolation_failure.output" cpu-tests "cpu-tests | PASS | 0.000s | run=3"
assert_contains "${tmp_dir}/feature_isolation_failure.output" "--- feature-isolation-check [FAIL(42), 0.000s] ---"
assert_contains "${tmp_dir}/feature_isolation_failure.output" "command: cargo hack check --workspace --each-feature --exclude-features default --locked --keep-going (6 package shards, isolated target dirs)"
assert_contains "${tmp_dir}/feature_isolation_failure.output" "log: ${tmp_dir}/phase-logs/feature-isolation-check.log"
echo "PASS: feature-isolation failure is preserved and fails the aggregated full gate"

run_check feature_isolation_union_mismatch env FAKE_HACK_CORRUPT_UNION=1 /usr/bin/bash scripts/check_all.sh
assert_status feature_isolation_union_mismatch 1
assert_phase_contains "${tmp_dir}/feature_isolation_union_mismatch.output" feature-isolation-check "feature-isolation-check | FAIL(1) | 0.000s | -"
assert_contains "${tmp_dir}/phase-logs/feature-isolation-check.log" "do not match the canonical command"
echo "PASS: sharded feature-isolation union mismatch fails closed"

run_check live_inventory_failure env FAKE_LIVE_INVENTORY_STATUS=42 /usr/bin/bash scripts/check_all.sh --lane all
assert_status live_inventory_failure 1
assert_contains "${tmp_dir}/live_inventory_failure.output" "RAW_LIVE_INVENTORY_FAILURE"
assert_phase_contains "${tmp_dir}/live_inventory_failure.output" live-inventory "live-inventory | FAIL(42) | 0.000s | -"
assert_phase_contains "${tmp_dir}/live_inventory_failure.output" cpu-tests "cpu-tests | SKIPPED(blocked) | - | -"
assert_phase_contains "${tmp_dir}/live_inventory_failure.output" gpu-shared-tests "gpu-shared-tests | SKIPPED(blocked) | - | -"
assert_contains "${tmp_dir}/live_inventory_failure.output" "--- live-inventory [FAIL(42), 0.000s] ---"
assert_contains "${tmp_dir}/live_inventory_failure.output" "command: bash scripts/ci/check_gpu_test_lane.sh --all-features"
assert_contains "${tmp_dir}/phase-logs/live-inventory.log" "RAW_LIVE_INVENTORY_FAILURE"
echo "PASS: failed live inventory is logged and blocks selected lanes"

run_check dev_cpu env FAKE_SELECTED_LANES=cpu /usr/bin/bash scripts/check_all.sh \
  --dev -p datafusion-nexus --test cpu_contract_tests
assert_status dev_cpu 0
assert_contains "${tmp_dir}/dev_cpu.output" "mode=dev lane=scoped package=datafusion-nexus test=cpu_contract_tests"
assert_contains "${tmp_dir}/dev_cpu.output" "PARTIAL: full pre-merge validation was not run."
assert_contains "${tmp_dir}/dev_cpu.log" "cargo <clippy> <-p> <datafusion-nexus> <--all-targets> <--all-features>"
assert_contains "${tmp_dir}/dev_cpu.log" "bash <scripts/ci/check_gpu_test_lane.sh> <--all-features> <--package> <datafusion-nexus> <--test> <cpu_contract_tests>"
assert_contains "${tmp_dir}/dev_cpu.log" "cargo <nextest> <run> <--cargo-metadata>"
assert_not_contains "${tmp_dir}/dev_cpu.log" "flock "
assert_not_contains "${tmp_dir}/dev_cpu.log" "cargo <nextest> <show-config>"
assert_contains "${tmp_dir}/phase-logs/phases.tsv" $'cpu-tests\tPASS'
assert_contains "${tmp_dir}/phase-logs/run.tsv" $'mode\tdev'
echo "PASS: focused CPU development loop stays package-scoped and avoids GPU locks"

run_check dev_gpu env FAKE_SELECTED_LANES=gpu-shared /usr/bin/bash scripts/check_all.sh \
  --dev -p datafusion-nexus --test native_planning_tests
assert_status dev_gpu 0
assert_contains "${tmp_dir}/dev_gpu.log" "cargo <nextest> <show-config>"
if [[ $(grep -c '^flock ' "${tmp_dir}/dev_gpu.log") -ne 1 ]]; then
  echo "FAIL: focused shared-GPU loop must acquire exactly one GPU lock" >&2
  exit 1
fi
assert_not_contains "${tmp_dir}/dev_gpu.log" "<--workspace>"
echo "PASS: focused GPU development loop reuses scoped binaries and locks only execution"

run_check dev_lint_failure env FAKE_CLIPPY_STATUS=41 /usr/bin/bash scripts/check_all.sh \
  --dev -p datafusion-nexus --test cpu_contract_tests
assert_status dev_lint_failure 1
assert_contains "${tmp_dir}/dev_lint_failure.output" "ABORT: focused lint failed"
assert_not_contains "${tmp_dir}/dev_lint_failure.log" "bash <scripts/ci/check_gpu_test_lane.sh>"
assert_not_contains "${tmp_dir}/dev_lint_failure.log" "cargo <nextest> <run>"
assert_contains "${tmp_dir}/phase-logs/phases.tsv" $'clippy\tFAIL(41)'

run_check dev_cpu_failure env $'FAKE_SELECTED_LANES=cpu\ngpu-shared' FAKE_CPU_TEST_STATUS=42 \
  /usr/bin/bash scripts/check_all.sh --dev -p datafusion-nexus
assert_status dev_cpu_failure 1
assert_contains "${tmp_dir}/dev_cpu_failure.output" "ABORT: focused CPU tests failed"
assert_not_contains "${tmp_dir}/dev_cpu_failure.log" "flock "
assert_contains "${tmp_dir}/phase-logs/phases.tsv" $'gpu-shared-tests\tSKIPPED(blocked)'

for bad_lanes in empty invalid missing; do
  bad_env=(env FAKE_SELECTED_LANES=)
  [[ "${bad_lanes}" != invalid ]] || bad_env=(env FAKE_SELECTED_LANES=unexpected)
  [[ "${bad_lanes}" != missing ]] || bad_env=(env FAKE_SKIP_LANES_FILE=1)
  run_check "dev_lanes_${bad_lanes}" "${bad_env[@]}" /usr/bin/bash scripts/check_all.sh \
    --dev -p datafusion-nexus --test cpu_contract_tests
  assert_status "dev_lanes_${bad_lanes}" 1
  assert_not_contains "${tmp_dir}/dev_lanes_${bad_lanes}.log" "cargo <nextest> <run>"
done
echo "PASS: focused development failures stop before dependent test and GPU work"

run_check dev_missing_package /usr/bin/bash scripts/check_all.sh --dev
assert_status dev_missing_package 2
assert_contains "${tmp_dir}/dev_missing_package.output" "--dev requires -p PACKAGE"

run_check public_scoped_lane /usr/bin/bash scripts/check_all.sh --lane scoped
assert_status public_scoped_lane 2
assert_contains "${tmp_dir}/public_scoped_lane.output" "--lane must be all, cpu, gpu, gpu-shared, gpu-exclusive, or policy"
echo "PASS: focused development CLI fails closed on incomplete or internal-only scope"

run_check source_changed env FAKE_MUTATE_WORKTREE=1 FAKE_MUTATE_TARGET="${mutation_fixture}" /usr/bin/bash scripts/check_all.sh
assert_status source_changed 1
assert_contains "${tmp_dir}/source_changed.output" "ERROR: source identity changed during the gate."
rm -f -- "${mutation_fixture}"

printf 'initial\n' >"${mutation_fixture}"
run_check source_changed_dirty env FAKE_MUTATE_WORKTREE=1 FAKE_MUTATE_TARGET="${mutation_fixture}" /usr/bin/bash scripts/check_all.sh
assert_status source_changed_dirty 1
assert_contains "${tmp_dir}/source_changed_dirty.output" "ERROR: source identity changed during the gate."
rm -f -- "${mutation_fixture}"
echo "PASS: source identity change during gate fails closed"

# test_all.sh must not inject different defaults or otherwise change semantics.
run_check alias /usr/bin/bash scripts/ci/test_all.sh
assert_status alias 0
diff -u "${tmp_dir}/default.output" "${tmp_dir}/alias.output"
# Command-log order across parallel phases is schedule-dependent; compare the
# issued command set while console output above stays byte-identical.
diff -u <(LC_ALL=C sort "${tmp_dir}/default.log") <(LC_ALL=C sort "${tmp_dir}/alias.log")
echo "PASS: test_all is a semantic alias"

echo "OK: GPU scheduling policy tests passed without running GPU tests."
