#!/usr/bin/env bash
# The one validation entry. With no arguments it runs the full pre-merge gate;
# --fast is a partial edit loop. Test selection, GPU locking, and timeouts live
# in .config/nextest.toml; this script runs everything else around the tests.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}" || exit 2

# Keep build-script fingerprints and doctest loading identical across phases.
export LD_LIBRARY_PATH="${ROOT}/target/native/cudf-install/lib:${ROOT}/target/native/cuvs-install/lib:${ROOT}/target/native/cugraph-install/lib:${ROOT}/target/native/cuml-install/lib:${ROOT}/target/native/cugraph-build"

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
if ! cargo nextest --version; then
  echo "ERROR: cargo-nextest is required. Install it with: cargo install --locked cargo-nextest" >&2
  exit 2
fi
if [[ "${mode}" == full ]]; then
  require_version "cargo-hack ${REQUIRED_CARGO_HACK_VERSION}" \
    "cargo install --locked --version ${REQUIRED_CARGO_HACK_VERSION} cargo-hack" cargo hack --version
fi

cargo +nightly -Z unstable-options config get env --format toml --show-origin \
  | python3 scripts/ci/cargo_toolchain_environment.py || exit 2
echo "check_all: LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

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

# The qualification verifier compares gitlink revisions with run artifacts.
gate_source_identity() {
  local root_revision state gitlinks="" entry marker revision path
  root_revision=$(git -C "${ROOT}" rev-parse HEAD)
  state=clean
  [[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all --ignore-submodules=none)" ]] || state=dirty
  while IFS= read -r entry; do
    marker=${entry:0:1}
    read -r revision path _ <<< "${entry:1}"
    [[ "$marker" == " " ]] && marker='='
    [[ -z "$gitlinks" ]] || gitlinks+=","
    gitlinks+="${marker}${revision}@${path}"
  done < <(git -C "${ROOT}" submodule status)
  printf 'root=%s:%s gitlinks=%s\n' "$root_revision" "$state" "$gitlinks"
}

POLICY_SCRIPTS=(
  scripts/ci/tests/check_cpu_packaging_tests.sh
  scripts/ci/check_single_brand.sh
  scripts/build/check_build_support.sh
  scripts/ci/tests/check_dependency_direction_tests.sh
  scripts/ci/tests/check_native_sources_tests.sh
  scripts/ci/tests/check_clean_target_artifacts_tests.sh
  scripts/ci/tests/check_pass_as_skip_policy_tests.sh
  scripts/ci/tests/check_public_api_surface_tests.sh
  scripts/ci/tests/check_public_api_consumers_tests.sh
  scripts/ci/tests/check_nextest_gpu_lock_tests.sh
  scripts/ci/tests/check_prose_slop_tests.sh
  scripts/ci/check_prose_slop.sh
  scripts/ci/tests/check_sloplint_tests.sh
  scripts/ci/check_sloplint.sh
  scripts/ci/tests/check_cugraph_example_ddl_tests.sh
  scripts/ci/check_cugraph_example_ddl.sh
)

policy_scripts() {
  local script status=0
  for script in "${POLICY_SCRIPTS[@]}"; do
    echo "==> ${script}"
    bash "${script}" || { echo "FAIL: ${script}"; status=1; }
  done
  python3 scripts/ci/tests/check_check_all_tests.py || status=1
  python3 scripts/ci/tests/check_single_brand_tests.py || status=1
  test_layout_policy || status=1
  return "${status}"
}

test_layout_policy() {
  python3 scripts/ci/tests/check_test_layout_tests.py && python3 scripts/ci/check_test_layout.py "${ROOT}"
}

clippy() {
  nice -n 10 cargo clippy --workspace --all-targets --all-features --message-format=json-render-diagnostics -- -D warnings \
    >"${LOG_DIR}/clippy-all-features.jsonl" || return $?
  # These owners have source files selected only when native features are off.
  nice -n 10 cargo clippy -p algeon-query-engine -p algeon-datafusion --all-targets --no-default-features \
    --message-format=json-render-diagnostics -- -D warnings >"${LOG_DIR}/clippy-feature-off.jsonl"
}

checked_bindings() {
  local component status=0
  for component in cudf cugraph cuvs cuml; do
    bash scripts/dev/bindings.sh "${component}" --check || status=1
  done
  return "${status}"
}

test_build() {
  cargo metadata --all-features --format-version 1 >"${LOG_DIR}/cargo-metadata.json" || return $?
  cargo nextest list --workspace --all-features --cargo-metadata "${LOG_DIR}/cargo-metadata.json" \
    --list-type binaries-only --message-format json >"${LOG_DIR}/binaries-metadata.json"
}

doctests() {
  NEXTEST_TEST_GROUP=gpu scripts/ci/nextest_gpu_lock.sh "${GPU_LOCK}" \
    cargo test --workspace --all-features --doc --no-fail-fast
}

# The canonical feature-isolation matrix
# (cargo hack check --workspace --each-feature --exclude-features default
# --all-targets --locked --keep-going) split into package shards with their own
# target directories. The union check fails closed when a workspace member has
# no shard.
FEATURE_ISOLATION_SHARDS=(
  "tools"
  "algeon-datafusion workloads datafusion-facade-consumer algeon-interop"
  "algeon-query-runtime algeon-query-planning"
  "algeon-query-engine algeon-cudf algeon-cudf-sys algeon-build-support"
  "algeon-cuml algeon-cuml-sys"
  "algeon-server algeon-cugraph-sys"
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
      nice -n 10 cargo hack check "${args[@]}" --each-feature --exclude-features default --all-targets --locked --keep-going \
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
  expected=$(cargo hack check --workspace --each-feature --exclude-features default --all-targets --locked \
    --keep-going --print-command-list 2>/dev/null | LC_ALL=C sort)
  if [[ -z "${expected}" ]]; then
    echo "FAIL: cannot enumerate the canonical feature-isolation checks"
    return 1
  fi
  for index in "${!FEATURE_ISOLATION_SHARDS[@]}"; do
    args=()
    for pkg in ${FEATURE_ISOLATION_SHARDS[index]}; do args+=(-p "${pkg}"); done
    cargo hack check "${args[@]}" --each-feature --exclude-features default --all-targets --locked --keep-going \
      --print-command-list
  done 2>/dev/null | LC_ALL=C sort >"${shards}/union.txt"
  if ! diff <(printf '%s\n' "${expected}") "${shards}/union.txt"; then
    echo "FAIL: sharded feature-isolation checks differ from the canonical command list"
    status=1
  fi
  return "${status}"
}

# Lints the shipped server exactly as packaged: only the algeon_server binary
# with server.features from the packaging policy. Selecting no tests, examples,
# or other members keeps dev-dependencies from enabling test-hooks, which the
# --all-features lanes always enable.
shipped_clippy() {
  local features
  features=$(python3 scripts/build/cpu_packaging_policy.py shipped-features) || return 1
  nice -n 10 cargo clippy --manifest-path crates/algeon-server/Cargo.toml --bin algeon_server --no-default-features \
    --features "${features}" --locked --target-dir "${ROOT}/target/check-all/shipped-clippy" -- -D warnings
}

if [[ "${mode}" == full ]]; then
  PHASES=(gate rustfmt native-sources checked-bindings policy-scripts test-build tests doctests clippy orphan-sources rustdoc
    public-api-surface feature-isolation shipped-clippy msrv native-free public-api-consumers publication-policy)
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
if [[ "${mode}" == full ]]; then echo "CHECK SOURCE START: $(gate_source_identity)"; fi

run rustfmt cargo +nightly fmt --all --check &
run native-sources bash scripts/ci/check_native_sources.sh &
run policy-scripts policy_scripts &
if [[ "${mode}" == full ]]; then
  run checked-bindings checked_bindings &
  if run test-build test_build; then
    { run tests cargo nextest run -P gate --cargo-metadata "${LOG_DIR}/cargo-metadata.json" \
        --binaries-metadata "${LOG_DIR}/binaries-metadata.json"; run doctests doctests; } &
  else
    printf '%s\n' 'SKIP: nextest execution requires a successful test build.' >"${LOG_DIR}/tests.log"
    record tests 'SKIP(test-build-failed)' - -
    run doctests doctests &
  fi
  { if run clippy clippy; then
      run orphan-sources python3 scripts/ci/check_test_layout.py "${ROOT}" \
        --clippy-artifacts "${LOG_DIR}/clippy-all-features.jsonl" \
        --clippy-artifacts "${LOG_DIR}/clippy-feature-off.jsonl"
    else
      printf '%s\n' 'SKIP: orphan-source verification requires successful clippy streams.' >"${LOG_DIR}/orphan-sources.log"
      record orphan-sources 'SKIP(clippy-failed)' - -
    fi
    RUSTDOCFLAGS="${RUSTDOCFLAGS:+${RUSTDOCFLAGS} }-D warnings" run rustdoc nice -n 10 cargo doc --workspace --all-features --no-deps --keep-going
    run public-api-surface nice -n 10 bash scripts/ci/check_public_api_surface.sh check --all-features; } &
  run feature-isolation feature_isolation &
  run shipped-clippy shipped_clippy &
  CARGO_TARGET_DIR="${ROOT}/target/check-all/msrv" \
    run msrv nice -n 10 cargo hack check --workspace --all-features --locked --rust-version --workspace-behavior=cargo &
  DOCS_RS=1 run native-free nice -n 10 cargo check --workspace --all-targets --all-features --locked --target-dir target/native-free-check &
  run public-api-consumers nice -n 10 python3 scripts/ci/check_public_api_consumers.py "${ROOT}" .config/policies/public-api \
    target/check-all/public-api-consumers &
  run publication-policy python3 scripts/ci/check_workspace_publication.py &
else
  run cpu-tests cargo nt -E 'not group(gpu) & not group(gpu-exclusive)' &
fi
wait

status=0
if [[ "${mode}" == full ]]; then echo "CHECK SOURCE END: $(gate_source_identity)"; fi
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
