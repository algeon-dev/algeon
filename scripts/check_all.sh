#!/usr/bin/env bash
# Canonical local validation harness. With no arguments this runs the required
# full pre-merge surface. --fast and --lane are deliberately partial surfaces.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

readonly REQUIRED_NEXTEST_VERSION=0.9.140
readonly REQUIRED_CARGO_HACK_VERSION=0.6.45
readonly GPU_EXCLUSIVE_CAP=1
readonly GPU_NEXTEST_PROFILE=default
readonly GPU_NEXTEST_GROUP=gpu
readonly GPU_LOCK=/tmp/cudf-gpu.lock

# Do not touch CARGO_INCREMENTAL. Long-term contract is to leave Cargo incremental
# policy to caller environment without process-wide export.

usage() {
  cat <<'USAGE'
Usage: bash scripts/check_all.sh [OPTIONS]

With no options, run the canonical full pre-merge surface with all features and
cargo-nextest 0.9.140. Partial modes never claim full validation.

Modes:
  --fast                         Edit loop: policies, fmt, structure, CPU tests
  --lane LANE                    Partial lane: all, cpu, gpu, gpu-shared,
                                 gpu-exclusive, or policy
  -h, --help                    Show this help and exit

Environment compatibility:
  CHECK_ALL_PHASE_LOG_DIR optionally selects the persistent phase-log directory;
  otherwise a unique directory is created below target/check-all/.

Examples:
  bash scripts/check_all.sh
  bash scripts/check_all.sh --fast
  bash scripts/check_all.sh --lane gpu-shared
USAGE
}

usage_error() {
  echo "ERROR: $*" >&2
  echo "Run 'bash scripts/check_all.sh --help' for usage." >&2
  exit 2
}

require_value() {
  local option=$1
  local remaining=$2
  local value=${3-}
  if [[ "${remaining}" -lt 2 || -z "${value}" || "${value}" == --* ]]; then
    usage_error "${option} requires a value."
  fi
}

seen_help=0
seen_fast=0
seen_lane=0
cli_lane=""

original_arg_count=$#
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      [[ "${seen_help}" -eq 0 ]] || usage_error "duplicate option: $1"
      seen_help=1
      shift
      ;;
    --fast)
      [[ "${seen_fast}" -eq 0 ]] || usage_error "duplicate option: --fast"
      seen_fast=1
      shift
      ;;
    --lane)
      [[ "${seen_lane}" -eq 0 ]] || usage_error "duplicate option: --lane"
      require_value --lane "$#" "${2-}"
      seen_lane=1
      cli_lane=$2
      shift 2
      ;;
    --lane=*)
      [[ "${seen_lane}" -eq 0 ]] || usage_error "duplicate option: --lane"
      seen_lane=1
      cli_lane=${1#*=}
      [[ -n "${cli_lane}" ]] || usage_error "--lane requires a value."
      shift
      ;;
    --*) usage_error "unknown option: $1" ;;
    *) usage_error "unexpected positional argument: $1" ;;
  esac
done

if [[ "${seen_help}" -eq 1 ]]; then
  [[ "${original_arg_count}" -eq 1 ]] || usage_error "--help cannot be combined with other options."
  usage
  exit 0
fi

[[ ! ( "${seen_fast}" -eq 1 && "${seen_lane}" -eq 1 ) ]]   || usage_error "--fast conflicts with --lane."

PHASE_LOG_DIR_OVERRIDE=""
if [[ -v CHECK_ALL_PHASE_LOG_DIR ]]; then
  [[ -n "${CHECK_ALL_PHASE_LOG_DIR}" && "${CHECK_ALL_PHASE_LOG_DIR}" != *$'\n'* && "${CHECK_ALL_PHASE_LOG_DIR}" != *'|'* ]]     || usage_error "CHECK_ALL_PHASE_LOG_DIR must be a non-empty single-line path without '|'."
  PHASE_LOG_DIR_OVERRIDE=${CHECK_ALL_PHASE_LOG_DIR}
fi

mode=full
lane=all
if [[ "${seen_fast}" -eq 1 ]]; then
  mode=fast
  lane=cpu
elif [[ "${seen_lane}" -eq 1 ]]; then
  mode=lane
  lane=${cli_lane}
fi
case "${lane}" in
  all | cpu | gpu | gpu-shared | gpu-exclusive | policy) ;;
  *) usage_error "--lane must be all, cpu, gpu, gpu-shared, gpu-exclusive, or policy, got '${lane}'." ;;
esac

gpu_threads=$(nproc 2>/dev/null)
nproc_status=$?
[[ "${nproc_status}" -eq 0 ]]   || usage_error "nproc failed with status ${nproc_status}; cannot determine available CPU cores."
[[ "${gpu_threads}" =~ ^[1-9][0-9]*$ ]]   || usage_error "nproc must report one positive integer, got '${gpu_threads}'."

# Formatting requires the installed nightly toolchain.
RUSTFMT_TOOLCHAIN="nightly"

DOC_RUSTDOCFLAGS="${RUSTDOCFLAGS:-}"
DOC_RUSTDOCFLAGS="${DOC_RUSTDOCFLAGS:+${DOC_RUSTDOCFLAGS} }-D warnings"

needs_nextest=0
if [[ "${lane}" != policy ]]; then
  needs_nextest=1
fi
if [[ "${needs_nextest}" -eq 1 ]]; then
  nextest_version_output=$(cargo nextest --version 2>&1)
  nextest_version_status=$?
  if [[ "${nextest_version_status}" -ne 0 ]]; then
    printf '%s\n' "${nextest_version_output}" >&2
    echo "ERROR: canonical runner cargo-nextest ${REQUIRED_NEXTEST_VERSION} is required (version command exited ${nextest_version_status})." >&2
    exit 2
  fi
  nextest_first_line=${nextest_version_output%%$'\n'*}
  nextest_reported_version=$(printf '%s\n' "${nextest_first_line}" | awk '$1 == "cargo-nextest" { print $2 }')
  if [[ "${nextest_reported_version}" != "${REQUIRED_NEXTEST_VERSION}" ]]; then
    echo "ERROR: canonical runner requires cargo-nextest ${REQUIRED_NEXTEST_VERSION}; got '${nextest_first_line}'." >&2
    exit 2
  fi
fi

if [[ "${mode}" == full ]]; then
  cargo_hack_version_output=$(cargo hack --version 2>&1)
  cargo_hack_version_status=$?
  if [[ "${cargo_hack_version_status}" -ne 0 ]]; then
    printf '%s\n' "${cargo_hack_version_output}" >&2
    echo "ERROR: canonical compatibility checks require cargo-hack ${REQUIRED_CARGO_HACK_VERSION} (version command exited ${cargo_hack_version_status}). Install it with: cargo install --locked --version ${REQUIRED_CARGO_HACK_VERSION} cargo-hack" >&2
    exit 2
  fi
  if [[ "${cargo_hack_version_output}" != "cargo-hack ${REQUIRED_CARGO_HACK_VERSION}" ]]; then
    echo "ERROR: canonical compatibility checks require exactly 'cargo-hack ${REQUIRED_CARGO_HACK_VERSION}'; got '${cargo_hack_version_output}'. Install it with: cargo install --locked --version ${REQUIRED_CARGO_HACK_VERSION} cargo-hack" >&2
    exit 2
  fi
fi

if [[ -n "${PHASE_LOG_DIR_OVERRIDE}" ]]; then
  PHASE_LOG_DIR=${PHASE_LOG_DIR_OVERRIDE}
  mkdir -p "${PHASE_LOG_DIR}" || usage_error "cannot create phase log directory '${PHASE_LOG_DIR}'."
else
  mkdir -p target/check-all || usage_error "cannot create target/check-all for phase logs."
  PHASE_LOG_DIR=$(mktemp -d target/check-all/run.XXXXXX)     || usage_error "cannot create a phase log directory under target/check-all."
fi

untracked_files_hash() {
  local target_dir=$1
  local path
  local -a untracked=()
  while IFS= read -r -d '' path; do
    [[ -n "${path}" ]] && untracked+=("${path}")
  done < <(git -C "${target_dir}" ls-files --others --exclude-standard -z 2>/dev/null)
  if (( ${#untracked[@]} == 0 )); then
    return 0
  fi
  (
    cd "${target_dir}" || exit 0
    sha256sum --zero -- "${untracked[@]}" 2>/dev/null
  ) | LC_ALL=C sort -z | sha256sum | awk '{print $1}'
}

root_source_fingerprint() {
  local revision diff_hash untracked_hash
  revision=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null) || revision=unavailable
  diff_hash=$(git -C "${ROOT}" diff --no-ext-diff --no-renames --binary HEAD 2>/dev/null | sha256sum | awk '{print $1}')
  untracked_hash=$(untracked_files_hash "${ROOT}")
  if [[ "${diff_hash}" == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" && -z "${untracked_hash}" ]]; then
    printf '%s:clean' "${revision}"
  else
    local dirty_hash
    dirty_hash=$(printf '%s\0%s' "${diff_hash}" "${untracked_hash}" | sha256sum | awk '{print $1}')
    printf '%s:dirty-%s' "${revision}" "${dirty_hash}"
  fi
}

submodule_source_fingerprint() {
  local submodule_root=$1
  if [[ ! -d "${submodule_root}" ]]; then
    printf 'unavailable'
    return
  fi
  local head diff_hash untracked_hash
  head=$(git -C "${submodule_root}" rev-parse HEAD 2>/dev/null) || head=unavailable
  diff_hash=$(git -C "${submodule_root}" diff --no-ext-diff --no-renames --binary HEAD 2>/dev/null | sha256sum | awk '{print $1}')
  untracked_hash=$(untracked_files_hash "${submodule_root}")
  if [[ "${diff_hash}" == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" && -z "${untracked_hash}" ]]; then
    printf '%s:clean' "${head}"
  else
    local sub_dirty_hash
    sub_dirty_hash=$(printf '%s\0%s' "${diff_hash}" "${untracked_hash}" | sha256sum | awk '{print $1}')
    printf '%s:dirty-%s' "${head}" "${sub_dirty_hash}"
  fi
}

gitlink_gate_identity() {
  local first
  local path
  local separator=""
  local status
  local status_output

  status_output=$(git -C "${ROOT}" submodule status --recursive 2>/dev/null) || return 1
  while read -r first path _; do
    [[ -n "${first}" && -n "${path}" ]] || continue
    case "${first}" in
      -* | +* | U*) status=${first:0:1}; first=${first:1} ;;
      *) status='=' ;;
    esac
    local sub_fp="unavailable"
    if [[ -d "${ROOT}/${path}" ]]; then
      sub_fp=$(submodule_source_fingerprint "${ROOT}/${path}")
    fi
    printf '%s%s%s@%s[%s]' "${separator}" "${status}" "${first}" "${path}" "${sub_fp}"
    separator=,
  done <<<"${status_output}"
  [[ -n "${separator}" ]] || return 1
}

gate_source_identity() {
  local gitlinks
  gitlinks=$(gitlink_gate_identity) || return 1
  printf 'root=%s gitlinks=%s'     "$(root_source_fingerprint)"     "${gitlinks}"
}

echo "CHECK CONFIG: mode=${mode} lane=${lane} runner=nextest(default) features=all(default) gpu_profile=${GPU_NEXTEST_PROFILE} gpu_threads=${gpu_threads}(nproc) strict=always(default)"
echo "PHASE LOG DIRECTORY: ${PHASE_LOG_DIR}"
gate_source_start=$(gate_source_identity) || usage_error "cannot resolve recursive gitlink source identity."
echo "CHECK SOURCE START: ${gate_source_start}"
if [[ "${mode}" != full ]]; then
  echo "PARTIAL SURFACE: mode=${mode} lane=${lane}; this does not satisfy full pre-merge validation."
fi

status=0
fatal_status=0
lane_safe=1
gpu_scheduling_summary=""
doctest_summary="not-run"
rustdoc_summary="not-run"

readonly POLICY_SCRIPTS=(
  scripts/ci/tests/check_cpu_packaging_tests.sh
  scripts/ci/check_iceberg_leaf_dependency.sh
  scripts/build/check_build_support.sh
  scripts/ci/tests/check_dependency_direction_tests.sh
  scripts/ci/tests/check_doctest_report_tests.sh
  scripts/ci/tests/check_phase_test_counts_tests.sh
  scripts/ci/tests/check_phase_failed_tests_tests.sh
  scripts/ci/tests/check_gpu_test_lane_tests.sh
  scripts/ci/tests/check_native_sources_tests.sh
  scripts/ci/tests/check_doctest_rustdoc_tests.sh
  scripts/ci/tests/check_gpu_scheduling_tests.sh
  scripts/ci/tests/check_clean_target_artifacts_tests.sh
  scripts/ci/tests/check_pass_as_skip_policy_tests.sh
  scripts/ci/tests/check_public_api_surface_tests.sh
)
policy_selftest_count=${#POLICY_SCRIPTS[@]}

PHASE_ORDER=(
  policy-selftests
  native-sources
  checked-bindings
  component-packages
  cuvs-native-free
  msrv-check
  feature-isolation-check
  rustfmt
  lane-structure
  live-inventory
  gpu-preflight
  cpu-tests
  gpu-shared-tests
  gpu-exclusive-tests
  doctests
  rustdoc
  public-api-surface
  clippy
)
declare -A PHASE_COMMAND PHASE_DURATION PHASE_STATUS PHASE_COUNTS PHASE_KIND
declare -A PHASE_GPU_CAP PHASE_LOCK PHASE_LOG PHASE_EXIT

for id in "${PHASE_ORDER[@]}"; do
  PHASE_COMMAND["${id}"]=-
  PHASE_DURATION["${id}"]=-
  PHASE_STATUS["${id}"]=SKIPPED
  PHASE_COUNTS["${id}"]=-
  PHASE_KIND["${id}"]=-
  PHASE_GPU_CAP["${id}"]=-
  PHASE_LOCK["${id}"]=-
  PHASE_LOG["${id}"]=-
  PHASE_EXIT["${id}"]=-
done

phase_now_ms() {
  local value
  value=$(date +%s%3N 2>/dev/null) || value=0
  [[ "${value}" =~ ^[0-9]+$ ]] || value=0
  printf '%s\n' "${value}"
}

phase_duration() {
  local start=$1
  local end=$2
  local elapsed=0
  if (( end >= start )); then
    elapsed=$(( end - start ))
  fi
  printf '%d.%03ds' "$(( elapsed / 1000 ))" "$(( elapsed % 1000 ))"
}

phase_set_accounting_failure() {
  local id=$1
  local diagnostic=$2
  if [[ "${PHASE_STATUS[${id}]}" == PASS ]]; then
    PHASE_STATUS["${id}"]='FAIL(accounting)'
  fi
  PHASE_COUNTS["${id}"]=unavailable
  status=1
  echo "ERROR: ${id} accounting failed: ${diagnostic}" >&2
}

run_phase() {
  local id=$1
  local count_kind=$2
  local gpu_cap=$3
  local lock_path=$4
  shift 4
  if [[ -z "${PHASE_COMMAND[${id}]:-}" || "${PHASE_COMMAND[${id}]}" == "-" ]]; then
    PHASE_COMMAND["${id}"]="$*"
  fi
  local phase_log="${PHASE_LOG_DIR}/${id}.log"
  local start end command_status tee_status
  local -a pipeline_status
  : >"${phase_log}" || {
    echo "ERROR: cannot create phase log '${phase_log}'." >&2
    PHASE_STATUS["${id}"]='FAIL(log)'
    PHASE_LOG["${id}"]=${phase_log}
    status=1
    return 1
  }
  PHASE_LOG["${id}"]=${phase_log}
  PHASE_KIND["${id}"]=${count_kind}
  PHASE_GPU_CAP["${id}"]=${gpu_cap}
  PHASE_LOCK["${id}"]=${lock_path}
  echo
  echo "==> [${id}] ${PHASE_COMMAND[${id}]}"
  start=$(phase_now_ms)
  "$@" 2>&1 | tee "${phase_log}"
  pipeline_status=("${PIPESTATUS[@]}")
  command_status=${pipeline_status[0]}
  tee_status=${pipeline_status[1]}
  PHASE_EXIT["${id}"]=${command_status}
  if [[ "${command_status}" -eq 0 && "${tee_status}" -eq 0 ]]; then
    PHASE_STATUS["${id}"]=PASS
  elif [[ "${tee_status}" -ne 0 ]]; then
    PHASE_STATUS["${id}"]="FAIL(tee:${tee_status})"
    status=1
  else
    PHASE_STATUS["${id}"]="FAIL(${command_status})"
    status=1
  fi

  case "${count_kind}" in
    none) ;;
    literal:*) PHASE_COUNTS["${id}"]=${count_kind#literal:} ;;
    nextest | cargo)
      local count_output count_status=0
      count_output=$(python3 scripts/ci/phase_test_counts.py --kind "${count_kind}" --input "${phase_log}" 2>&1)         || count_status=$?
      if [[ "${count_status}" -eq 0 ]]; then
        PHASE_COUNTS["${id}"]=${count_output}
        printf 'PHASE TEST COUNTS: %s\n' "${count_output}" | tee -a "${phase_log}"
      else
        printf '%s\n' "${count_output}" | tee -a "${phase_log}" >&2
        phase_set_accounting_failure "${id}" "test-count parser exited ${count_status}"
      fi
      ;;
    doctest)
      local report="${phase_log}.json" doctest_count_fields=""
      local parser_status=0 parser_tee_status=0 render_status=0
      python3 scripts/ci/doctest_report.py summarize         --input "${phase_log}"         --command-status "${command_status}"         --json-out "${report}" 2>&1 | tee -a "${phase_log}"
      pipeline_status=("${PIPESTATUS[@]}")
      parser_status=${pipeline_status[0]}
      parser_tee_status=${pipeline_status[1]}
      doctest_summary=$(python3 scripts/ci/doctest_report.py render --summary "${report}" 2>/dev/null)         || render_status=$?
      if [[ "${parser_status}" -eq 0 && "${parser_tee_status}" -eq 0 && "${render_status}" -eq 0 ]]; then
        doctest_count_fields=${doctest_summary#* }
        doctest_count_fields=${doctest_count_fields#* }
        PHASE_COUNTS["${id}"]=${doctest_count_fields}
        printf '==> Doctest accounting preflight for final aggregation: %s\n' "${doctest_summary}"           | tee -a "${phase_log}"
      else
        if [[ "${render_status}" -ne 0 ]]; then
          doctest_summary="command_status=${command_status} parse_status=unavailable harnesses=unknown executed=unknown passed=unknown failed=unknown ignored=unknown measured=unknown filtered=unknown"
        fi
        phase_set_accounting_failure "${id}" "doctest parser=${parser_status} tee=${parser_tee_status} render=${render_status}"
        if [[ "${render_status}" -eq 0 ]]; then
          doctest_count_fields=${doctest_summary#* }
          doctest_count_fields=${doctest_count_fields#* }
          PHASE_COUNTS["${id}"]=${doctest_count_fields}
        fi
        printf '==> Doctest accounting preflight for final aggregation: %s\n' "${doctest_summary}"           | tee -a "${phase_log}"
      fi
      ;;
    *)
      phase_set_accounting_failure "${id}" "unknown count kind '${count_kind}'"
      ;;
  esac

  end=$(phase_now_ms)
  PHASE_DURATION["${id}"]=$(phase_duration "${start}" "${end}")
  [[ "${PHASE_STATUS[${id}]}" == PASS ]]
}

run_isolated_policy_commands() {
  local policy_status=0
  local script
  for script in "${POLICY_SCRIPTS[@]}"; do
    echo
    echo "--> ${script}"
    bash "${script}" || policy_status=1
  done
  return "${policy_status}"
}

run_lane_structure_commands() {
  local structure_status=0
  python3 scripts/ci/test_lane_policy.py check-catalog || structure_status=1
  python3 scripts/ci/test_lane_policy.py check-nextest || structure_status=1
  return "${structure_status}"
}

needs_cpu=0
needs_shared_gpu=0
needs_exclusive_gpu=0
case "${mode}:${lane}" in
  full:all | lane:all)
    needs_cpu=1
    needs_shared_gpu=1
    needs_exclusive_gpu=1
    ;;
  fast:cpu | lane:cpu) needs_cpu=1 ;;
  lane:gpu)
    needs_shared_gpu=1
    needs_exclusive_gpu=1
    ;;
  lane:gpu-shared)
    needs_shared_gpu=1
    ;;
  lane:gpu-exclusive) needs_exclusive_gpu=1 ;;
esac

checked_bindings_command() {
  bash scripts/dev/bindings.sh cudf --check
  bash scripts/dev/bindings.sh cuvs --check
}

rustdoc_command() {
  RUSTDOCFLAGS="${DOC_RUSTDOCFLAGS}" cargo doc --workspace --all-features --no-deps --keep-going
}

if [[ "${mode}" == full || "${lane}" == policy || "${mode}" == fast ]]; then
  PHASE_COMMAND[policy-selftests]="${policy_selftest_count} isolated policy/orchestration scripts"
  run_phase policy-selftests "literal:checks=${policy_selftest_count}" - - run_isolated_policy_commands || true
  run_phase native-sources none - - bash scripts/ci/check_native_sources.sh || true
  run_phase rustfmt none - - cargo "+${RUSTFMT_TOOLCHAIN}" fmt --all --check || true
  PHASE_COMMAND[lane-structure]="test_lane_policy.py check-catalog + check-nextest"
  if ! run_phase lane-structure literal:checks=2 - - run_lane_structure_commands; then
    lane_safe=0
  fi
else
  PHASE_COMMAND[lane-structure]="test_lane_policy.py check-catalog + check-nextest"
  if ! run_phase lane-structure literal:checks=2 - - run_lane_structure_commands; then
    lane_safe=0
  fi
fi

if [[ "${mode}" == full ]]; then
  PHASE_COMMAND[checked-bindings]="scripts/dev/bindings.sh cudf --check + cuvs --check"
  run_phase checked-bindings none - - checked_bindings_command || true
  run_phase component-packages "literal:packages=3" - - bash scripts/ci/check_component_packages.sh || true
  PHASE_COMMAND[cuvs-native-free]="CUVS_NEXUS_SYS_SKIP_NATIVE=1 cargo check --locked -p cuvs-nexus-sys --all-targets --all-features"
  run_phase cuvs-native-free none - - env CUVS_NEXUS_SYS_SKIP_NATIVE=1     cargo check --locked -p cuvs-nexus-sys --all-targets --all-features || true
  run_phase msrv-check none - - cargo hack check --workspace --all-features --locked     --rust-version --workspace-behavior=cargo || true
  run_phase feature-isolation-check none - - cargo hack check --workspace --each-feature     --exclude-features default --locked --keep-going || true
fi

if [[ "${lane}" != policy ]]; then
  if [[ "${mode}" != fast ]]; then
    if ! run_phase live-inventory none - - bash scripts/ci/check_gpu_test_lane.sh --all-features; then
      lane_safe=0
      echo "ABORT: live lane inventory failed; selected nextest lanes will not run." >&2
    fi
  fi
  if [[ "${lane_safe}" -eq 1 && ( "${needs_shared_gpu}" -eq 1 || "${needs_exclusive_gpu}" -eq 1 ) ]]; then
    preflight_cap=-
    if [[ "${needs_shared_gpu}" -eq 1 && "${needs_exclusive_gpu}" -eq 1 ]]; then
      preflight_cap="${gpu_threads},${GPU_EXCLUSIVE_CAP}"
    elif [[ "${needs_shared_gpu}" -eq 1 ]]; then
      preflight_cap=${gpu_threads}
    else
      preflight_cap=${GPU_EXCLUSIVE_CAP}
    fi
    if ! run_phase gpu-preflight none "${preflight_cap}" - cargo nextest show-config --color never -P "${GPU_NEXTEST_PROFILE}" test-groups --workspace --all-features; then
      echo "ERROR: GPU scheduling preflight is invalid; refusing to run GPU lanes." >&2
      fatal_status=2
      lane_safe=0
    else
      shared_summary="shared=not-selected"
      exclusive_summary="exclusive=not-selected"
      [[ "${needs_shared_gpu}" -eq 0 ]]         || shared_summary="shared: profile=${GPU_NEXTEST_PROFILE} group=${GPU_NEXTEST_GROUP} configured_cap=num-cpus explicit_test_threads=${gpu_threads} effective_gpu_cap=${gpu_threads}"
      [[ "${needs_exclusive_gpu}" -eq 0 ]]         || exclusive_summary="exclusive: group=gpu-exclusive configured_cap=${GPU_EXCLUSIVE_CAP} explicit_test_threads=${GPU_EXCLUSIVE_CAP}"
      gpu_scheduling_summary="${shared_summary}; ${exclusive_summary}; lock=${GPU_LOCK}"
      echo "==> GPU scheduling preflight: ${gpu_scheduling_summary}"
    fi
  fi
  if [[ "${lane_safe}" -eq 1 ]]; then
    PHASE_COMMAND[cpu-tests]="cargo nextest run --workspace --all-features -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast --final-status-level fail"
    [[ "${needs_cpu}" -eq 0 ]]       || run_phase cpu-tests nextest - - cargo nextest run --workspace --all-features         -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast --final-status-level fail || true
    PHASE_COMMAND[gpu-shared-tests]="flock ${GPU_LOCK} cargo nextest run -P ${GPU_NEXTEST_PROFILE} --workspace --all-features -E group(${GPU_NEXTEST_GROUP}) --no-fail-fast --test-threads ${gpu_threads} --final-status-level fail"
    [[ "${needs_shared_gpu}" -eq 0 ]]       || run_phase gpu-shared-tests nextest "${gpu_threads}" "${GPU_LOCK}"         flock "${GPU_LOCK}" cargo nextest run -P "${GPU_NEXTEST_PROFILE}"         --workspace --all-features -E "group(${GPU_NEXTEST_GROUP})" --no-fail-fast         --test-threads "${gpu_threads}" --final-status-level fail || true
    PHASE_COMMAND[gpu-exclusive-tests]="flock ${GPU_LOCK} cargo nextest run -P ${GPU_NEXTEST_PROFILE} --workspace --all-features -E group(gpu-exclusive) --no-fail-fast --test-threads ${GPU_EXCLUSIVE_CAP} --final-status-level fail"
    [[ "${needs_exclusive_gpu}" -eq 0 ]]       || run_phase gpu-exclusive-tests nextest "${GPU_EXCLUSIVE_CAP}" "${GPU_LOCK}"         flock "${GPU_LOCK}" cargo nextest run -P "${GPU_NEXTEST_PROFILE}"         --workspace --all-features -E 'group(gpu-exclusive)' --no-fail-fast         --test-threads "${GPU_EXCLUSIVE_CAP}" --final-status-level fail || true
  else
    [[ "${needs_cpu}" -eq 0 || "${PHASE_STATUS[cpu-tests]}" != SKIPPED ]]       || PHASE_STATUS[cpu-tests]='SKIPPED(blocked)'
    [[ "${needs_shared_gpu}" -eq 0 || "${PHASE_STATUS[gpu-shared-tests]}" != SKIPPED ]]       || PHASE_STATUS[gpu-shared-tests]='SKIPPED(blocked)'
    [[ "${needs_exclusive_gpu}" -eq 0 || "${PHASE_STATUS[gpu-exclusive-tests]}" != SKIPPED ]]       || PHASE_STATUS[gpu-exclusive-tests]='SKIPPED(blocked)'
  fi
fi

if [[ "${mode}" == full ]]; then
  PHASE_COMMAND[doctests]="flock ${GPU_LOCK} cargo test --workspace --all-features --doc --no-fail-fast"
  run_phase doctests doctest - "${GPU_LOCK}"     flock "${GPU_LOCK}" cargo test --workspace --all-features --doc --no-fail-fast || true
  PHASE_COMMAND[rustdoc]="RUSTDOCFLAGS='${DOC_RUSTDOCFLAGS}' cargo doc --workspace --all-features --no-deps --keep-going"
  run_phase rustdoc none - - rustdoc_command || true
  rustdoc_summary="command_status=${PHASE_EXIT[rustdoc]} warnings=denied"
  run_phase public-api-surface literal:crates=8 - -     bash scripts/ci/check_public_api_surface.sh check --all-features || true
  PHASE_COMMAND[clippy]="cargo clippy --workspace --all-targets --all-features -- -D warnings"
  run_phase clippy none - -     cargo clippy --workspace --all-targets --all-features -- -D warnings || true
fi

sanitize_phase_cell() {
  local value=$1
  value=${value//$'\n'/ }
  value=${value//$'\t'/ }
  value=${value//|//}
  printf '%s' "${value}"
}

print_phase_table() {
  local id status pass_count=0 fail_count=0 skipped_count=0
  for id in "${PHASE_ORDER[@]}"; do
    status=${PHASE_STATUS[${id}]}
    if [[ "${status}" == PASS ]]; then
      pass_count=$((pass_count + 1))
    elif [[ "${status}" == FAIL* ]]; then
      fail_count=$((fail_count + 1))
    else
      skipped_count=$((skipped_count + 1))
    fi
  done
  echo
  echo "PHASE SUMMARY (${#PHASE_ORDER[@]} phases: ${pass_count} PASS, ${fail_count} FAIL, ${skipped_count} SKIPPED)"
  echo "Phase | Status | Duration | Counts"
  echo "--- | --- | --- | ---"
  for id in "${PHASE_ORDER[@]}"; do
    printf '%s | %s | %s | %s\n'       "$(sanitize_phase_cell "${id}")"       "$(sanitize_phase_cell "${PHASE_STATUS[${id}]}")"       "$(sanitize_phase_cell "${PHASE_DURATION[${id}]}")"       "$(sanitize_phase_cell "${PHASE_COUNTS[${id}]}")"
  done
  print_phase_failures
}

print_phase_failures() {
  local id kind parser_kind failed_output
  local -a names
  local printed=0
  for id in "${PHASE_ORDER[@]}"; do
    [[ "${PHASE_STATUS[${id}]}" == FAIL* ]] || continue
    if [[ "${printed}" -eq 0 ]]; then
      echo
      echo "FAILURES"
      printed=1
    fi
    echo "--- ${id} [${PHASE_STATUS[${id}]}, ${PHASE_DURATION[${id}]}] ---"
    printf 'command: %s\n' "${PHASE_COMMAND[${id}]}"
    printf 'log: %s\n' "${PHASE_LOG[${id}]}"
    kind=${PHASE_KIND[${id}]}
    case "${kind}" in
      nextest) parser_kind=nextest ;;
      cargo | doctest) parser_kind=cargo ;;
      *) continue ;;
    esac
    if ! failed_output=$(python3 scripts/ci/phase_failed_tests.py --kind "${parser_kind}" --input "${PHASE_LOG[${id}]}"); then
      echo "failed tests: unavailable (see log)"
    elif [[ -z "${failed_output}" ]]; then
      echo "failed tests: none listed (see log)"
    else
      mapfile -t names <<<"${failed_output}"
      printf 'failed tests (%d):\n' "${#names[@]}"
      print_phase_failure_names "${names[@]}"
    fi
  done
}

print_phase_failure_names() {
  local name shown=0 total=$#
  for name in "$@"; do
    [[ "${shown}" -lt 20 ]] || break
    printf '  %s\n' "${name}"
    shown=$((shown + 1))
  done
  if [[ "${total}" -gt 20 ]]; then
    printf '  ... and %d more (see log)\n' "$((total - 20))"
  fi
}

print_phase_table
echo
gate_source_end=$(gate_source_identity) || gate_source_end=unavailable
echo "CHECK SOURCE END: ${gate_source_end}"
if [[ "${gate_source_end}" == unavailable ]]; then
  echo "ERROR: recursive gitlink source identity became unavailable during the gate." >&2
  status=1
elif [[ "${gate_source_end}" != "${gate_source_start}" ]]; then
  echo "ERROR: source identity changed during the gate." >&2
  status=1
fi
[[ -z "${gpu_scheduling_summary}" ]] || echo "GPU scheduling final: ${gpu_scheduling_summary}"
[[ "${mode}" != full ]] || echo "Doctest accounting final: ${doctest_summary}"
[[ "${mode}" != full ]] || echo "Rustdoc final: ${rustdoc_summary}"

final_status=${status}
if [[ "${fatal_status}" -ne 0 ]]; then
  final_status=${fatal_status}
fi
if [[ "${final_status}" -ne 0 ]]; then
  if [[ "${mode}" == full ]]; then
    echo "FAIL: canonical full pre-merge surface reported problems."
  else
    echo "FAIL: selected partial validation surface reported problems."
  fi
elif [[ "${mode}" == full ]]; then
  echo "OK: canonical full pre-merge surface passed."
elif [[ "${mode}" == fast ]]; then
  echo "OK: fast edit-loop surface passed. PARTIAL: full pre-merge validation was not run."
else
  echo "OK: selected lane '${lane}' passed. PARTIAL: full pre-merge validation was not run."
fi
exit "${final_status}"
