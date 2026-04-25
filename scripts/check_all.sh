#!/usr/bin/env bash
# Canonical local validation harness. With no arguments this runs the required
# full pre-merge surface. --fast and --lane are deliberately partial surfaces.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

readonly REQUIRED_NEXTEST_VERSION=0.9.140
readonly GPU_EXCLUSIVE_CAP=1
readonly GPU_NEXTEST_PROFILE=default
readonly GPU_NEXTEST_GROUP=gpu

usage() {
  cat <<'USAGE'
Usage: bash scripts/check_all.sh [OPTIONS]

With no options, run the canonical full pre-merge surface with all features and
cargo-nextest 0.9.140. Partial modes never claim full validation.

Modes:
  --fast                         Edit loop: policies, fmt, structure, CPU tests
  --lane LANE                    Partial lane: all, cpu, gpu, gpu-shared,
                                 gpu-exclusive, or policy

Feature selection (CLI selectors replace FEATURES):
  --all-features                 Enable all default-member features (default)
  --features LIST               Enable a comma-separated feature list
  --no-default-features         Disable default features; may accompany --features

Execution:
  --runner RUNNER               nextest (canonical) or cargo-diagnostic
  --strict                      Compatibility spelling; rustdoc is always strict
  -h, --help                    Show this help and exit

Environment compatibility:
  FEATURES, TEST_RUNNER, GPU_LOCK, and STRICT remain accepted. Every present
  value is validated. Valid CLI values override their environment counterpart;
  TEST_RUNNER=cargo maps to cargo-diagnostic. Shared-GPU concurrency is always
  the number of CPU cores available to this environment, as reported by nproc.
  Any CLI feature selector replaces FEATURES rather than merging with it.
  CHECK_ALL_PHASE_LOG_DIR optionally selects the persistent phase-log directory;
  otherwise a unique directory is created below target/check-all/.

Examples:
  bash scripts/check_all.sh
  bash scripts/check_all.sh --fast --features server
  bash scripts/check_all.sh --lane gpu-shared
  bash scripts/check_all.sh --runner cargo-diagnostic
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

valid_feature_list() {
  [[ "$1" =~ ^[A-Za-z0-9_.+/-]+(,[A-Za-z0-9_.+/-]+)*$ ]]
}

parse_boolean() {
  local name=$1
  local value=$2
  case "${value}" in
    0 | false | FALSE | no | NO | off | OFF) BOOLEAN_RESULT=0 ;;
    1 | true | TRUE | yes | YES | on | ON) BOOLEAN_RESULT=1 ;;
    *) usage_error "${name} must be a boolean (0/1, false/true, no/yes, off/on), got '${value}'." ;;
  esac
}

seen_help=0
seen_fast=0
seen_lane=0
seen_features=0
seen_all_features=0
seen_no_default_features=0
seen_runner=0
seen_strict=0
cli_lane=""
cli_features=""
cli_runner=""

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
    --features)
      [[ "${seen_features}" -eq 0 ]] || usage_error "duplicate option: --features"
      require_value --features "$#" "${2-}"
      seen_features=1
      cli_features=$2
      shift 2
      ;;
    --features=*)
      [[ "${seen_features}" -eq 0 ]] || usage_error "duplicate option: --features"
      seen_features=1
      cli_features=${1#*=}
      [[ -n "${cli_features}" ]] || usage_error "--features requires a value."
      shift
      ;;
    --all-features)
      [[ "${seen_all_features}" -eq 0 ]] || usage_error "duplicate option: --all-features"
      seen_all_features=1
      shift
      ;;
    --no-default-features)
      [[ "${seen_no_default_features}" -eq 0 ]] || usage_error "duplicate option: --no-default-features"
      seen_no_default_features=1
      shift
      ;;
    --runner)
      [[ "${seen_runner}" -eq 0 ]] || usage_error "duplicate option: --runner"
      require_value --runner "$#" "${2-}"
      seen_runner=1
      cli_runner=$2
      shift 2
      ;;
    --runner=*)
      [[ "${seen_runner}" -eq 0 ]] || usage_error "duplicate option: --runner"
      seen_runner=1
      cli_runner=${1#*=}
      [[ -n "${cli_runner}" ]] || usage_error "--runner requires a value."
      shift
      ;;
    --strict)
      [[ "${seen_strict}" -eq 0 ]] || usage_error "duplicate option: --strict"
      seen_strict=1
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

[[ ! ( "${seen_fast}" -eq 1 && "${seen_lane}" -eq 1 ) ]] \
  || usage_error "--fast conflicts with --lane."
[[ ! ( "${seen_all_features}" -eq 1 && ( "${seen_features}" -eq 1 || "${seen_no_default_features}" -eq 1 ) ) ]] \
  || usage_error "--all-features conflicts with --features and --no-default-features."

if [[ "${seen_features}" -eq 1 ]] && ! valid_feature_list "${cli_features}"; then
  usage_error "--features must be a non-empty comma-separated feature list, got '${cli_features}'."
fi

# Validate every compatibility environment value before applying CLI
# precedence. An invalid ambient value is never silently hidden by a CLI flag.
env_features=""
if [[ -v FEATURES ]]; then
  env_features=${FEATURES}
  if [[ -n "${env_features}" ]] && ! valid_feature_list "${env_features}"; then
    usage_error "FEATURES must be empty or a comma-separated feature list, got '${env_features}'."
  fi
fi

runner=nextest
runner_source=default
if [[ -v TEST_RUNNER ]]; then
  case "${TEST_RUNNER}" in
    nextest) runner=nextest ;;
    cargo | cargo-diagnostic) runner=cargo-diagnostic ;;
    *) usage_error "TEST_RUNNER must be nextest, cargo, or cargo-diagnostic, got '${TEST_RUNNER}'." ;;
  esac
  runner_source=env
fi
if [[ "${seen_runner}" -eq 1 ]]; then
  case "${cli_runner}" in
    nextest | cargo-diagnostic) runner=${cli_runner} ;;
    *) usage_error "--runner must be nextest or cargo-diagnostic, got '${cli_runner}'." ;;
  esac
  runner_source=cli
fi

if [[ -v GPU_LOCK ]]; then
  [[ -n "${GPU_LOCK}" && "${GPU_LOCK}" != *$'\n'* && "${GPU_LOCK}" != -* ]] \
    || usage_error "GPU_LOCK must be a non-empty single-line path that does not begin with '-'."
else
  GPU_LOCK=/tmp/cudf-gpu.lock
fi

PHASE_LOG_DIR_OVERRIDE=""
if [[ -v CHECK_ALL_PHASE_LOG_DIR ]]; then
  [[ -n "${CHECK_ALL_PHASE_LOG_DIR}" && "${CHECK_ALL_PHASE_LOG_DIR}" != *$'\n'* && "${CHECK_ALL_PHASE_LOG_DIR}" != *'|'* ]] \
    || usage_error "CHECK_ALL_PHASE_LOG_DIR must be a non-empty single-line path without '|'."
  PHASE_LOG_DIR_OVERRIDE=${CHECK_ALL_PHASE_LOG_DIR}
fi

strict_source=default
if [[ -v STRICT ]]; then
  parse_boolean STRICT "${STRICT}"
  [[ "${BOOLEAN_RESULT}" -eq 1 ]] \
    || usage_error "STRICT is a compatibility flag for strict rustdoc only and must be true when present."
  strict_source=env
fi
if [[ "${seen_strict}" -eq 1 ]]; then
  strict_source=cli
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

if [[ "${mode}" == fast && "${runner}" != nextest ]]; then
  usage_error "--fast requires --runner nextest."
fi
if [[ "${mode}" == lane && "${runner}" == cargo-diagnostic && "${lane}" != all && "${lane}" != policy ]]; then
  usage_error "cargo-diagnostic cannot select individual test lanes; use --lane all."
fi
gpu_threads=$(nproc 2>/dev/null)
nproc_status=$?
[[ "${nproc_status}" -eq 0 ]] \
  || usage_error "nproc failed with status ${nproc_status}; cannot determine available CPU cores."
[[ "${gpu_threads}" =~ ^[1-9][0-9]*$ ]] \
  || usage_error "nproc must report one positive integer, got '${gpu_threads}'."

feature_source=default
if [[ "${seen_all_features}" -eq 1 ]]; then
  FEATURE_ARGS=(--all-features)
  feature_label=all
  feature_source=cli
elif [[ "${seen_features}" -eq 1 || "${seen_no_default_features}" -eq 1 ]]; then
  FEATURE_ARGS=()
  if [[ "${seen_no_default_features}" -eq 1 ]]; then
    FEATURE_ARGS+=(--no-default-features)
  fi
  if [[ "${seen_features}" -eq 1 ]]; then
    FEATURE_ARGS+=(--features "${cli_features}")
    if [[ "${seen_no_default_features}" -eq 1 ]]; then
      feature_label="no-default+features:${cli_features}"
    else
      feature_label="features:${cli_features}"
    fi
  else
    feature_label=no-default-features
  fi
  feature_source=cli
elif [[ -n "${env_features}" ]]; then
  FEATURE_ARGS=(--features "${env_features}")
  feature_label="features:${env_features}"
  feature_source=env
else
  FEATURE_ARGS=(--all-features)
  feature_label=all
fi

# Formatting requires the installed nightly toolchain.
RUSTFMT_TOOLCHAIN="nightly"

DOC_RUSTDOCFLAGS="${RUSTDOCFLAGS:-}"
DOC_RUSTDOCFLAGS="${DOC_RUSTDOCFLAGS:+${DOC_RUSTDOCFLAGS} }-D warnings"

needs_nextest=0
if [[ "${runner}" == nextest && "${lane}" != policy ]]; then
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

if [[ -n "${PHASE_LOG_DIR_OVERRIDE}" ]]; then
  PHASE_LOG_DIR=${PHASE_LOG_DIR_OVERRIDE}
  mkdir -p "${PHASE_LOG_DIR}" || usage_error "cannot create phase log directory '${PHASE_LOG_DIR}'."
else
  mkdir -p target/check-all || usage_error "cannot create target/check-all for phase logs."
  PHASE_LOG_DIR=$(mktemp -d target/check-all/run.XXXXXX) \
    || usage_error "cannot create a phase log directory under target/check-all."
fi

echo "CHECK CONFIG: mode=${mode} lane=${lane} runner=${runner}(${runner_source}) features=${feature_label}(${feature_source}) gpu_profile=${GPU_NEXTEST_PROFILE} gpu_threads=${gpu_threads}(nproc) strict=always(${strict_source})"
echo "PHASE LOG DIRECTORY: ${PHASE_LOG_DIR}"
if [[ "${mode}" != full ]]; then
  echo "PARTIAL SURFACE: mode=${mode} lane=${lane}; this does not satisfy full pre-merge validation."
elif [[ "${runner}" == cargo-diagnostic ]]; then
  echo "NONCANONICAL RUNNER: cargo-diagnostic serializes the test suite under the GPU lock; it does not satisfy canonical nextest validation."
fi

status=0
fatal_status=0
lane_safe=1
gpu_scheduling_summary=""
doctest_summary="not-run"
rustdoc_summary="not-run"

PHASE_ORDER=(
  policy-selftests
  rustfmt
  lane-structure
  default-members
  live-inventory
  gpu-preflight
  cpu-tests
  gpu-shared-tests
  gpu-exclusive-tests
  cargo-diagnostic
  doctests
  rustdoc
  clippy
  tools-check
  examples-check
)
declare -A PHASE_COMMAND PHASE_DURATION PHASE_STATUS PHASE_COUNTS
declare -A PHASE_GPU_CAP PHASE_LOCK PHASE_FEATURES PHASE_LOG PHASE_EXIT

define_phase() {
  local id=$1
  local command=$2
  PHASE_COMMAND["${id}"]=${command}
  PHASE_DURATION["${id}"]=-
  PHASE_STATUS["${id}"]=SKIPPED
  PHASE_COUNTS["${id}"]=-
  PHASE_GPU_CAP["${id}"]=-
  PHASE_LOCK["${id}"]=-
  PHASE_FEATURES["${id}"]=${feature_label}
  PHASE_LOG["${id}"]=-
  PHASE_EXIT["${id}"]=-
}

define_phase policy-selftests "8 isolated policy/orchestration scripts"
define_phase rustfmt "cargo +${RUSTFMT_TOOLCHAIN} fmt --all --check"
define_phase lane-structure "test_lane_policy.py check-catalog + generated override check"
define_phase default-members "cargo check ${FEATURE_ARGS[*]}"
define_phase live-inventory "bash scripts/ci/check_gpu_test_lane.sh ${FEATURE_ARGS[*]}"
define_phase gpu-preflight "cargo nextest show-config --color never -P ${GPU_NEXTEST_PROFILE} test-groups ${FEATURE_ARGS[*]}"
define_phase cpu-tests "cargo nextest run ${FEATURE_ARGS[*]} -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast"
define_phase gpu-shared-tests "flock ${GPU_LOCK} cargo nextest run -P ${GPU_NEXTEST_PROFILE} ${FEATURE_ARGS[*]} -E group(${GPU_NEXTEST_GROUP}) --no-fail-fast --test-threads ${gpu_threads}"
define_phase gpu-exclusive-tests "flock ${GPU_LOCK} cargo nextest run -P ${GPU_NEXTEST_PROFILE} ${FEATURE_ARGS[*]} -E group(gpu-exclusive) --no-fail-fast --test-threads ${GPU_EXCLUSIVE_CAP}"
define_phase cargo-diagnostic "RUST_TEST_THREADS=1 flock ${GPU_LOCK} cargo test ${FEATURE_ARGS[*]} --no-fail-fast --lib --bins --tests --examples -- --test-threads=1"
define_phase doctests "flock ${GPU_LOCK} cargo test ${FEATURE_ARGS[*]} --doc --no-fail-fast"
define_phase rustdoc "RUSTDOCFLAGS='${DOC_RUSTDOCFLAGS}' cargo doc ${FEATURE_ARGS[*]} --no-deps --keep-going"
define_phase clippy "cargo clippy --all-targets ${FEATURE_ARGS[*]}"
define_phase tools-check "cargo check -p datafusion-nexus-tools --all-targets --all-features"
define_phase examples-check "cargo check -p datafusion-nexus-examples --all-targets"

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
      count_output=$(python3 scripts/ci/phase_test_counts.py --kind "${count_kind}" --input "${phase_log}" 2>&1) \
        || count_status=$?
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
      python3 scripts/ci/doctest_report.py summarize \
        --input "${phase_log}" \
        --command-status "${command_status}" \
        --json-out "${report}" 2>&1 | tee -a "${phase_log}"
      pipeline_status=("${PIPESTATUS[@]}")
      parser_status=${pipeline_status[0]}
      parser_tee_status=${pipeline_status[1]}
      doctest_summary=$(python3 scripts/ci/doctest_report.py render --summary "${report}" 2>/dev/null) \
        || render_status=$?
      if [[ "${parser_status}" -eq 0 && "${parser_tee_status}" -eq 0 && "${render_status}" -eq 0 ]]; then
        doctest_count_fields=${doctest_summary#* }
        doctest_count_fields=${doctest_count_fields#* }
        PHASE_COUNTS["${id}"]=${doctest_count_fields}
        printf '==> Doctest accounting preflight for final aggregation: %s\n' "${doctest_summary}" \
          | tee -a "${phase_log}"
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
        printf '==> Doctest accounting preflight for final aggregation: %s\n' "${doctest_summary}" \
          | tee -a "${phase_log}"
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
  for script in \
    scripts/build/check_cpu_packaging_policy.sh \
    scripts/build/check_build_support.sh \
    scripts/ci/tests/check_doctest_report_tests.sh \
    scripts/ci/tests/check_phase_test_counts_tests.sh \
    scripts/ci/tests/check_gpu_test_lane_tests.sh \
    scripts/ci/tests/check_doctest_rustdoc_tests.sh \
    scripts/ci/tests/check_gpu_scheduling_tests.sh \
    scripts/ci/tests/check_pass_as_skip_policy_tests.sh; do
    echo
    echo "--> ${script}"
    bash "${script}" || policy_status=1
  done
  return "${policy_status}"
}

run_lane_structure_commands() {
  local structure_status=0
  python3 scripts/ci/test_lane_policy.py check-catalog || structure_status=1
  python3 scripts/ci/test_lane_policy.py generate-nextest --check || structure_status=1
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
  lane:gpu-shared) needs_shared_gpu=1 ;;
  lane:gpu-exclusive) needs_exclusive_gpu=1 ;;
esac

run_gpu_config_preflight() {
  local gpu_show_config gpu_show_config_status=0
  gpu_show_config=$(cargo nextest show-config --color never -P "${GPU_NEXTEST_PROFILE}" test-groups "${FEATURE_ARGS[@]}" 2>&1) \
    || gpu_show_config_status=$?
  if [[ "${gpu_show_config_status}" -ne 0 ]]; then
    printf '%s\n' "${gpu_show_config}" >&2
    echo "ERROR: nextest show-config failed with status ${gpu_show_config_status}; GPU scheduling cannot be verified." >&2
    return 2
  fi

  local shared_summary="shared=not-selected"
  local exclusive_summary="exclusive=not-selected"
  if [[ "${needs_shared_gpu}" -eq 1 ]]; then
    local configured_gpu_cap
    configured_gpu_cap=$(printf '%s\n' "${gpu_show_config}" \
      | sed -nE "s/^group: ${GPU_NEXTEST_GROUP} \(max threads = ([^)]+)\)$/\1/p")
    if [[ "${configured_gpu_cap}" != num-cpus ]]; then
      echo "ERROR: nextest show-config reports cap '${configured_gpu_cap}' for group '${GPU_NEXTEST_GROUP}', expected dynamic cap 'num-cpus'." >&2
      return 2
    fi
    shared_summary="shared: profile=${GPU_NEXTEST_PROFILE} group=${GPU_NEXTEST_GROUP} configured_cap=${configured_gpu_cap} explicit_test_threads=${gpu_threads} effective_gpu_cap=${gpu_threads}"
  fi
  if [[ "${needs_exclusive_gpu}" -eq 1 ]]; then
    local configured_gpu_exclusive_cap
    configured_gpu_exclusive_cap=$(printf '%s\n' "${gpu_show_config}" \
      | sed -nE 's/^group: gpu-exclusive \(max threads = ([0-9]+)\)$/\1/p')
    if ! [[ "${configured_gpu_exclusive_cap}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: nextest show-config did not report one numeric cap for group 'gpu-exclusive'." >&2
      return 2
    fi
    if [[ "${configured_gpu_exclusive_cap}" -ne "${GPU_EXCLUSIVE_CAP}" ]]; then
      echo "ERROR: nextest show-config reports cap ${configured_gpu_exclusive_cap} for group 'gpu-exclusive', expected required cap ${GPU_EXCLUSIVE_CAP}." >&2
      return 2
    fi
    exclusive_summary="exclusive: group=gpu-exclusive configured_cap=${configured_gpu_exclusive_cap} explicit_test_threads=${GPU_EXCLUSIVE_CAP}"
  fi
  echo "==> GPU scheduling preflight: ${shared_summary}; ${exclusive_summary}; lock=${GPU_LOCK}"
  return 0
}

cpu_test_command() {
  cargo nextest run "${FEATURE_ARGS[@]}" \
    -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast
}

gpu_shared_test_command() {
  flock "${GPU_LOCK}" cargo nextest run -P "${GPU_NEXTEST_PROFILE}" \
    "${FEATURE_ARGS[@]}" -E "group(${GPU_NEXTEST_GROUP})" --no-fail-fast --test-threads "${gpu_threads}"
}

gpu_exclusive_test_command() {
  flock "${GPU_LOCK}" cargo nextest run -P "${GPU_NEXTEST_PROFILE}" \
    "${FEATURE_ARGS[@]}" -E 'group(gpu-exclusive)' --no-fail-fast --test-threads "${GPU_EXCLUSIVE_CAP}"
}

cargo_diagnostic_command() {
  RUST_TEST_THREADS=1 flock "${GPU_LOCK}" cargo test \
    "${FEATURE_ARGS[@]}" --no-fail-fast --lib --bins --tests --examples -- --test-threads=1
}

doctest_command() {
  flock "${GPU_LOCK}" cargo test \
    "${FEATURE_ARGS[@]}" --doc --no-fail-fast
}

rustdoc_command() {
  RUSTDOCFLAGS="${DOC_RUSTDOCFLAGS}" cargo doc "${FEATURE_ARGS[@]}" --no-deps --keep-going
}

if [[ "${mode}" == full || "${lane}" == policy || "${mode}" == fast ]]; then
  run_phase policy-selftests literal:checks=7 - - run_isolated_policy_commands || true
  run_phase rustfmt none - - cargo "+${RUSTFMT_TOOLCHAIN}" fmt --all --check || true
  if ! run_phase lane-structure literal:checks=2 - - run_lane_structure_commands; then
    lane_safe=0
  fi
else
  if ! run_phase lane-structure literal:checks=2 - - run_lane_structure_commands; then
    lane_safe=0
  fi
fi

if [[ "${mode}" == full ]]; then
  run_phase default-members none - - cargo check "${FEATURE_ARGS[@]}" || true
fi

if [[ "${runner}" == nextest && "${lane}" != policy ]]; then
  if [[ "${mode}" != fast ]]; then
    if ! run_phase live-inventory none - - bash scripts/ci/check_gpu_test_lane.sh "${FEATURE_ARGS[@]}"; then
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
    if ! run_phase gpu-preflight none "${preflight_cap}" - run_gpu_config_preflight; then
      echo "ERROR: GPU scheduling preflight is invalid; refusing to run GPU lanes." >&2
      fatal_status=2
      lane_safe=0
    else
      shared_summary="shared=not-selected"
      exclusive_summary="exclusive=not-selected"
      [[ "${needs_shared_gpu}" -eq 0 ]] \
        || shared_summary="shared: profile=${GPU_NEXTEST_PROFILE} group=${GPU_NEXTEST_GROUP} configured_cap=num-cpus explicit_test_threads=${gpu_threads} effective_gpu_cap=${gpu_threads}"
      [[ "${needs_exclusive_gpu}" -eq 0 ]] \
        || exclusive_summary="exclusive: group=gpu-exclusive configured_cap=${GPU_EXCLUSIVE_CAP} explicit_test_threads=${GPU_EXCLUSIVE_CAP}"
      gpu_scheduling_summary="${shared_summary}; ${exclusive_summary}; lock=${GPU_LOCK}"
    fi
  fi
  if [[ "${lane_safe}" -eq 1 ]]; then
    [[ "${needs_cpu}" -eq 0 ]] \
      || run_phase cpu-tests nextest - - cpu_test_command || true
    [[ "${needs_shared_gpu}" -eq 0 ]] \
      || run_phase gpu-shared-tests nextest "${gpu_threads}" "${GPU_LOCK}" gpu_shared_test_command || true
    [[ "${needs_exclusive_gpu}" -eq 0 ]] \
      || run_phase gpu-exclusive-tests nextest "${GPU_EXCLUSIVE_CAP}" "${GPU_LOCK}" gpu_exclusive_test_command || true
  else
    [[ "${needs_cpu}" -eq 0 || "${PHASE_STATUS[cpu-tests]}" != SKIPPED ]] \
      || PHASE_STATUS[cpu-tests]='SKIPPED(blocked)'
    [[ "${needs_shared_gpu}" -eq 0 || "${PHASE_STATUS[gpu-shared-tests]}" != SKIPPED ]] \
      || PHASE_STATUS[gpu-shared-tests]='SKIPPED(blocked)'
    [[ "${needs_exclusive_gpu}" -eq 0 || "${PHASE_STATUS[gpu-exclusive-tests]}" != SKIPPED ]] \
      || PHASE_STATUS[gpu-exclusive-tests]='SKIPPED(blocked)'
  fi
elif [[ "${runner}" == cargo-diagnostic && "${lane}" == all ]]; then
  if [[ "${lane_safe}" -eq 1 ]]; then
    run_phase cargo-diagnostic cargo "${GPU_EXCLUSIVE_CAP}" "${GPU_LOCK}" cargo_diagnostic_command || true
  else
    PHASE_STATUS[cargo-diagnostic]='SKIPPED(blocked)'
  fi
fi

if [[ "${mode}" == full ]]; then
  run_phase doctests doctest - "${GPU_LOCK}" doctest_command || true
  run_phase rustdoc none - - rustdoc_command || true
  rustdoc_summary="command_status=${PHASE_EXIT[rustdoc]} warnings=denied"
  run_phase clippy none - - cargo clippy --all-targets "${FEATURE_ARGS[@]}" || true
  run_phase tools-check none - - cargo check -p datafusion-nexus-tools --all-targets --all-features || true
  run_phase examples-check none - - cargo check -p datafusion-nexus-examples --all-targets || true
fi

sanitize_phase_cell() {
  local value=$1
  value=${value//$'\n'/ }
  value=${value//$'\t'/ }
  value=${value//|//}
  printf '%s' "${value}"
}

print_phase_table() {
  local id
  echo
  echo "PHASE SUMMARY"
  echo "Phase | Command | Duration | Status | Test counts | GPU cap | Lock | Features | Log"
  echo "--- | --- | --- | --- | --- | --- | --- | --- | ---"
  for id in "${PHASE_ORDER[@]}"; do
    printf '%s | %s | %s | %s | %s | %s | %s | %s | %s\n' \
      "$(sanitize_phase_cell "${id}")" \
      "$(sanitize_phase_cell "${PHASE_COMMAND[${id}]}")" \
      "$(sanitize_phase_cell "${PHASE_DURATION[${id}]}")" \
      "$(sanitize_phase_cell "${PHASE_STATUS[${id}]}")" \
      "$(sanitize_phase_cell "${PHASE_COUNTS[${id}]}")" \
      "$(sanitize_phase_cell "${PHASE_GPU_CAP[${id}]}")" \
      "$(sanitize_phase_cell "${PHASE_LOCK[${id}]}")" \
      "$(sanitize_phase_cell "${PHASE_FEATURES[${id}]}")" \
      "$(sanitize_phase_cell "${PHASE_LOG[${id}]}")"
  done
}

print_phase_table
echo
[[ -z "${gpu_scheduling_summary}" ]] || echo "GPU scheduling final: ${gpu_scheduling_summary}"
[[ "${mode}" != full ]] || echo "Doctest accounting final: ${doctest_summary}"
[[ "${mode}" != full ]] || echo "Rustdoc final: ${rustdoc_summary}"

final_status=${status}
if [[ "${fatal_status}" -ne 0 ]]; then
  final_status=${fatal_status}
fi
if [[ "${final_status}" -ne 0 ]]; then
  if [[ "${mode}" == full && "${runner}" == nextest ]]; then
    echo "FAIL: canonical full pre-merge surface reported problems."
  else
    echo "FAIL: selected partial/noncanonical surface reported problems."
  fi
elif [[ "${mode}" == full && "${runner}" == nextest ]]; then
  echo "OK: canonical full pre-merge surface passed."
elif [[ "${mode}" == full ]]; then
  echo "OK: NONCANONICAL cargo-diagnostic surface passed; run with nextest for full pre-merge validation."
elif [[ "${mode}" == fast ]]; then
  echo "OK: fast edit-loop surface passed. PARTIAL: full pre-merge validation was not run."
else
  echo "OK: selected lane '${lane}' passed. PARTIAL: full pre-merge validation was not run."
fi
exit "${final_status}"
