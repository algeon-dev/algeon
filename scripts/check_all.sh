#!/usr/bin/env bash
# The one validation entry. With no arguments it runs the full pre-merge gate;
# --fast is a partial edit loop. Test selection, GPU locking, and timeouts live
# in .config/nextest.toml; this script runs everything else around the tests.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}" || exit 2

readonly REQUIRED_NEXTEST_VERSION=0.9.144
readonly REQUIRED_CARGO_HACK_VERSION=0.6.45
readonly GPU_LOCK=/tmp/cudf-gpu.lock

usage() {
  cat <<'USAGE'
Usage: bash scripts/check_all.sh [--fast]

  (no options)  Full pre-merge gate: policy, tests, static checks, hygiene.
  --fast        Policy checks, rustfmt, and CPU tests. Partial: never full
                validation.

Each phase logs to target/check-all/run.*/<phase>.log, or to
CHECK_ALL_PHASE_LOG_DIR when set. target/check-all/latest points at the newest
run; its summary.tsv is rewritten whenever a phase starts or finishes.
USAGE
}

mode=full
case "$*" in
  '') ;;
  --fast) mode=fast ;;
  -h | --help) usage; exit 0 ;;
  *) echo "ERROR: unknown arguments: $*" >&2; usage >&2; exit 2 ;;
esac

require_version() {
  local expected=$1 install=$2 actual
  shift 2
  actual=$("$@" 2>&1 | head -n 1)
  if [[ "${actual%% (*}" != "${expected}" ]]; then
    echo "ERROR: requires '${expected}', got '${actual}'. Install it with: ${install}" >&2
    exit 2
  fi
}
require_version "cargo-nextest ${REQUIRED_NEXTEST_VERSION}" \
  "cargo install --locked --version ${REQUIRED_NEXTEST_VERSION} cargo-nextest" cargo nextest --version
if [[ "${mode}" == full ]]; then
  require_version "cargo-hack ${REQUIRED_CARGO_HACK_VERSION}" \
    "cargo install --locked --version ${REQUIRED_CARGO_HACK_VERSION} cargo-hack" cargo hack --version
fi

mkdir -p target/check-all || exit 2
if [[ -n "${CHECK_ALL_PHASE_LOG_DIR:-}" ]]; then
  mkdir -p "${CHECK_ALL_PHASE_LOG_DIR}" || exit 2
  LOG_DIR=$(cd "${CHECK_ALL_PHASE_LOG_DIR}" && pwd)
else
  LOG_DIR="${ROOT}/$(mktemp -d target/check-all/run.XXXXXX)" || exit 2
fi
ln -sfn "${LOG_DIR}" target/check-all/latest

# Content identity of a checkout: HEAD, tracked changes, and untracked files.
tree_identity() {
  local dir=$1
  {
    git -C "${dir}" rev-parse HEAD
    git -C "${dir}" diff --no-ext-diff --no-renames --binary HEAD
    git -C "${dir}" ls-files --others --exclude-standard -z | (cd "${dir}" && xargs -0 -r sha256sum --)
  } 2>&1 | sha256sum | cut -d ' ' -f 1
}

# The gate result only describes the source it started with, so every
# submodule's content is part of the identity compared at the end.
source_identity() {
  local path identity
  identity="root=$(tree_identity "${ROOT}")"
  while read -r path; do
    identity+=" ${path}=$(tree_identity "${ROOT}/${path}")"
  done < <(git -C "${ROOT}" submodule status --recursive | awk '{ print $2 }')
  printf '%s\n' "${identity}"
}

POLICY_SCRIPTS=(
  scripts/ci/tests/check_cpu_packaging_tests.sh
  scripts/ci/check_iceberg_leaf_dependency.sh
  scripts/ci/check_single_brand.sh
  scripts/build/check_build_support.sh
  scripts/ci/tests/check_dependency_direction_tests.sh
  scripts/ci/tests/check_native_sources_tests.sh
  scripts/ci/tests/check_clean_target_artifacts_tests.sh
  scripts/ci/tests/check_pass_as_skip_policy_tests.sh
  scripts/ci/tests/check_public_api_surface_tests.sh
  scripts/ci/tests/check_nextest_gpu_lock_tests.sh
  scripts/ci/tests/check_prose_slop_tests.sh
  scripts/ci/check_prose_slop.sh
  scripts/ci/tests/check_sloplint_tests.sh
  scripts/ci/check_sloplint.sh
)

policy_scripts() {
  local script status=0
  for script in "${POLICY_SCRIPTS[@]}"; do
    echo "==> ${script}"
    bash "${script}" || { echo "FAIL: ${script}"; status=1; }
  done
  return "${status}"
}

checked_bindings() {
  local component status=0
  for component in cudf cugraph cuvs; do
    bash scripts/dev/bindings.sh "${component}" --check || status=1
  done
  return "${status}"
}

doctests() {
  # rustdoc does not pass the build scripts' rpath link arguments to doctest
  # binaries, and runs them from temporary directories, so the loader needs
  # absolute native library directories.
  local native="${ROOT}/target/native"
  NEXTEST_TEST_GROUP=gpu scripts/ci/nextest_gpu_lock.sh "${GPU_LOCK}" \
    env "LD_LIBRARY_PATH=${native}/cudf-install/lib:${native}/cuvs-install/lib:${native}/cugraph-install/lib:${native}/cugraph-build${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    cargo test --workspace --all-features --doc --no-fail-fast
}

# The canonical feature-isolation matrix
# (cargo hack check --workspace --each-feature --exclude-features default
# --locked --keep-going) split into package shards with their own target
# directories. The union check fails closed when a workspace member has no shard.
FEATURE_ISOLATION_SHARDS=(
  "tools"
  "algeon-datafusion workloads datafusion-facade-consumer algeon-interop"
  "algeon-query-runtime algeon-query-planning"
  "algeon-query-engine algeon-cudf algeon-cudf-sys algeon-build-support"
  "iceberg-catalog algeon-server algeon-cugraph-sys"
  "examples bench algeon-cugraph algeon-cuvs algeon-cuvs-sys"
)

feature_isolation() {
  local shards="${LOG_DIR}/feature-isolation" index pkg status=0 expected
  local -a pids=() args
  mkdir -p "${shards}" || return 1
  for index in "${!FEATURE_ISOLATION_SHARDS[@]}"; do
    args=()
    for pkg in ${FEATURE_ISOLATION_SHARDS[index]}; do args+=(-p "${pkg}"); done
    CARGO_TARGET_DIR="${ROOT}/target/check-all/feature-isolation-shard-${index}" CARGO_BUILD_JOBS=6 \
      cargo hack check "${args[@]}" --each-feature --exclude-features default --locked --keep-going \
      >"${shards}/shard-${index}.log" 2>&1 &
    pids+=($!)
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
      echo "FAIL: feature-isolation shard ${index} (${FEATURE_ISOLATION_SHARDS[index]}): ${shards}/shard-${index}.log"
      status=1
    fi
  done
  # Compare command lists in cargo-hack's own format: the shards' union must
  # equal what the workspace-wide command would run.
  expected=$(cargo hack check --workspace --each-feature --exclude-features default --locked --keep-going \
    --print-command-list 2>/dev/null | LC_ALL=C sort)
  if [[ -z "${expected}" ]]; then
    echo "FAIL: cannot enumerate the canonical feature-isolation checks"
    return 1
  fi
  for index in "${!FEATURE_ISOLATION_SHARDS[@]}"; do
    args=()
    for pkg in ${FEATURE_ISOLATION_SHARDS[index]}; do args+=(-p "${pkg}"); done
    cargo hack check "${args[@]}" --each-feature --exclude-features default --locked --keep-going --print-command-list
  done 2>/dev/null | LC_ALL=C sort >"${shards}/union.txt"
  if ! diff <(printf '%s\n' "${expected}") "${shards}/union.txt"; then
    echo "FAIL: sharded feature-isolation checks differ from the canonical command list"
    status=1
  fi
  return "${status}"
}

if [[ "${mode}" == full ]]; then
  PHASES=(gate rustfmt native-sources checked-bindings policy-scripts tests doctests clippy rustdoc
    public-api-surface feature-isolation msrv native-free publication-policy)
else
  PHASES=(gate rustfmt native-sources policy-scripts cpu-tests)
fi

# Phases finish concurrently, so each owns a row file and summary.tsv is
# re-rendered from all of them under a lock.
record() {
  local phase=$1 status=$2 started=$3 duration=$4 row
  printf '%s\t%s\t%s\t%s\t%s\n' "${phase}" "${status}" "${started}" "${duration}" "${LOG_DIR}/${phase}.log" \
    >"${LOG_DIR}/.${phase}.row"
  {
    flock 9
    {
      printf 'phase\tstatus\tstarted\tduration\tlog\n'
      for row in "${PHASES[@]}"; do cat "${LOG_DIR}/.${row}.row" 2>/dev/null; done
    } >"${LOG_DIR}/summary.tsv.tmp" && mv "${LOG_DIR}/summary.tsv.tmp" "${LOG_DIR}/summary.tsv"
    if [[ "${phase}" != gate && "${status}" != PENDING ]]; then
      printf 'check_all: phase=%s status=%s duration=%s log=%s\n' \
        "${phase}" "${status}" "${duration}" "${LOG_DIR}/${phase}.log"
    fi
  } 9>"${LOG_DIR}/.summary.lock"
}

run() {
  local phase=$1 started=$(date +%T) start=${SECONDS} status
  shift
  record "${phase}" RUNNING "${started}" -
  "$@" >"${LOG_DIR}/${phase}.log" 2>&1
  status=$?
  record "${phase}" "$([[ ${status} -eq 0 ]] && echo PASS || echo "FAIL(${status})")" "${started}" "$((SECONDS - start))s"
  return "${status}"
}

for phase in "${PHASES[@]}"; do record "${phase}" PENDING - -; done
gate_started=$(date +%T)
gate_start=${SECONDS}
record gate RUNNING "${gate_started}" -
echo "check_all: mode=${mode} logs=${LOG_DIR}"
source_start=$(source_identity)

run rustfmt cargo +nightly fmt --all --check &
run native-sources bash scripts/ci/check_native_sources.sh &
run policy-scripts policy_scripts &
if [[ "${mode}" == full ]]; then
  run checked-bindings checked_bindings &
  { run tests cargo nt -P gate; run doctests doctests; } &
  { run clippy cargo clippy --workspace --all-targets --all-features -- -D warnings
    RUSTDOCFLAGS="${RUSTDOCFLAGS:+${RUSTDOCFLAGS} }-D warnings" run rustdoc cargo doc --workspace --all-features --no-deps --keep-going
    run public-api-surface bash scripts/ci/check_public_api_surface.sh check --all-features; } &
  run feature-isolation feature_isolation &
  CARGO_TARGET_DIR="${ROOT}/target/check-all/msrv" \
    run msrv cargo hack check --workspace --all-features --locked --rust-version --workspace-behavior=cargo &
  DOCS_RS=1 run native-free cargo check --workspace --all-targets --all-features --locked --target-dir target/native-free-check &
  run publication-policy python3 scripts/ci/check_workspace_publication.py &
else
  run cpu-tests cargo nt -E 'not group(gpu) & not group(gpu-exclusive)' &
fi
wait

status=0
if [[ "$(source_identity)" != "${source_start}" ]]; then
  echo "ERROR: the source changed while the gate ran; its results do not describe the current source." >&2
  status=1
fi

echo
for phase in "${PHASES[@]:1}"; do
  IFS=$'\t' read -r _ result _ duration log <"${LOG_DIR}/.${phase}.row"
  printf '%-20s %-10s %6s\n' "${phase}" "${result}" "${duration}"
  [[ "${result}" == PASS ]] && continue
  status=1
  failures+="${phase}: ${log}"$'\n'
done

if [[ -n "${failures:-}" ]]; then
  echo
  echo "FAILED PHASES"
  printf '%s' "${failures}"
  for phase in tests cpu-tests; do
    [[ "${failures}" == *"${phase}: "* ]] || continue
    echo
    echo "--- ${phase}: nextest summary ---"
    summary=$(awk '/^ *Summary \[/ { found = 1 } found' "${LOG_DIR}/${phase}.log" | head -n 40)
    # A build failure ends before nextest prints a summary.
    printf '%s\n' "${summary:-$(tail -n 40 "${LOG_DIR}/${phase}.log")}"
    if [[ "${phase}" == tests ]]; then
      echo "Rerun only failed and unfinished tests: cargo nt -P gate -R latest"
    else
      echo "Rerun only failed and unfinished tests: cargo nt -E 'not group(gpu) & not group(gpu-exclusive)' -R latest"
    fi
  done
fi

if [[ ${status} -eq 0 ]]; then
  record gate PASS "${gate_started}" "$((SECONDS - gate_start))s"
  [[ "${mode}" == full ]] && echo "OK: full pre-merge gate passed." \
    || echo "OK: fast checks passed. PARTIAL: this is not full pre-merge validation."
else
  record gate FAIL "${gate_started}" "$((SECONDS - gate_start))s"
  echo "FAIL: $([[ "${mode}" == full ]] && echo "full pre-merge gate" || echo "fast checks") reported problems."
fi
exit "${status}"
