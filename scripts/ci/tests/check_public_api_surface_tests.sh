#!/usr/bin/env bash
# Fault-injection tests for the public-API-surface snapshot check. `cargo
# public-api` itself is faked out (a `cargo-public-api` shim on PATH), so this
# never runs a real rustdoc-JSON build: it proves the comparison/reporting
# logic in scripts/ci/check_public_api_surface.sh fails closed on drift, a
# missing snapshot, and a failing underlying tool, and passes on a clean
# baseline and after `write` regenerates it.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-public-api-surface-tests.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin" "${tmp_dir}/snapshots"

# A deterministic, tiny fake "public API" per crate, independent of the real
# crates in this repository.
baseline_datafusion_nexus='pub fn datafusion_nexus::backend::NexusGpuBackend::builder() -> datafusion_nexus::backend::NexusGpuBackendBuilder
pub struct datafusion_nexus::backend::NexusGpuBackend'
baseline_nexus_query_engine='pub fn nexus_query_engine::runtime::GpuBackendSession::admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle>'
baseline_nexus_bench='pub mod nexus_bench::tpch_benchmark'
baseline_nexus_server='pub struct nexus_server::FlightSqlServer'

# A deterministic fake toolchain identity, in the same shape
# current_toolchain_identity() in the real script produces.
baseline_tool_version='cargo-public-api 0.52.0-fake'
baseline_rustc_vv='rustc 1.99.0-nightly (fake0000 2026-01-01)
binary: rustc
commit-hash: fake0000000000000000000000000000000000
commit-date: 2026-01-01
host: x86_64-unknown-linux-gnu
release: 1.99.0-nightly
LLVM version: 22.1.8'
baseline_toolchain_file=$(printf 'cargo-public-api: %s\nnightly rustc:\n%s\n' "${baseline_tool_version}" "${baseline_rustc_vv}")

cat >"${tmp_dir}/bin/rustc" <<SH
#!/usr/bin/env bash
set -u
if [[ "\$1" == "+nightly" && "\$2" == "-Vv" ]]; then
  printf '%s\n' "\${FAKE_RUSTC_NIGHTLY_VV:-${baseline_rustc_vv}}"
  exit "\${FAKE_RUSTC_STATUS:-0}"
fi
echo "unexpected fake rustc invocation: \$*" >&2
exit 97
SH
chmod +x "${tmp_dir}/bin/rustc"

cat >"${tmp_dir}/bin/cargo-public-api" <<SH
#!/usr/bin/env bash
set -u
{
  printf 'cargo-public-api'
  printf ' <%s>' "\$@"
  printf '\n'
} >>"\${COMMAND_LOG}"

if [[ "\$*" == *" --version"* || "\$*" == "--version" ]]; then
  printf '%s\n' "\${FAKE_TOOL_VERSION:-${baseline_tool_version}}"
  exit "\${FAKE_TOOL_VERSION_STATUS:-0}"
fi

crate=""
prev=""
for arg in "\$@"; do
  if [[ "\${prev}" == "-p" ]]; then
    crate="\${arg}"
  fi
  prev="\${arg}"
done

if [[ "\${crate}" == "\${FAKE_TOOL_FAILS_FOR:-}" ]]; then
  echo "FAKE_CARGO_PUBLIC_API_FAILURE for \${crate}" >&2
  exit 1
fi

case "\${crate}" in
  datafusion-nexus) printf '%s\n' "\${FAKE_DATAFUSION_NEXUS:-${baseline_datafusion_nexus}}" ;;
  nexus-query-engine) printf '%s\n' "\${FAKE_NEXUS_QUERY_ENGINE:-${baseline_nexus_query_engine}}" ;;
  nexus-bench) printf '%s\n' "\${FAKE_NEXUS_BENCH:-${baseline_nexus_bench}}" ;;
  nexus-server) printf '%s\n' "\${FAKE_NEXUS_SERVER:-${baseline_nexus_server}}" ;;
  *)
    echo "unexpected fake cargo-public-api crate: \${crate}" >&2
    exit 97
    ;;
esac
SH
chmod +x "${tmp_dir}/bin/cargo-public-api"

write_clean_snapshots() {
  local dir=$1
  mkdir -p "${dir}"
  printf '%s\n' "${baseline_toolchain_file}" >"${dir}/TOOLCHAIN.txt"
  printf '%s\n' "${baseline_datafusion_nexus}" >"${dir}/datafusion-nexus.txt"
  printf '%s\n' "${baseline_nexus_query_engine}" >"${dir}/nexus-query-engine.txt"
  printf '%s\n' "${baseline_nexus_bench}" >"${dir}/nexus-bench.txt"
  printf '%s\n' "${baseline_nexus_server}" >"${dir}/nexus-server.txt"
}

run_check() {
  local name=$1
  local snapshot_dir=$2
  shift 2
  local log="${tmp_dir}/${name}.log"
  local output="${tmp_dir}/${name}.output"
  local status=0
  : >"${log}"
  PATH="${tmp_dir}/bin:${PATH}" COMMAND_LOG="${log}" PUBLIC_API_SNAPSHOT_DIR="${snapshot_dir}" \
    PUBLIC_API_SKIP_FACADE_POLICY=1 \
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

CHECK_SCRIPT=scripts/ci/check_public_api_surface.sh

# 1. Clean baseline: check passes for all four crates.
clean_dir="${tmp_dir}/snapshots/clean"
write_clean_snapshots "${clean_dir}"
run_check clean "${clean_dir}" "${CHECK_SCRIPT}" check --all-features
assert_status clean 0
assert_contains "${tmp_dir}/clean.output" "PASS: datafusion-nexus public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: nexus-query-engine public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: nexus-bench public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: nexus-server public API matches"
echo "PASS: clean snapshot matches for all four crates"

# 2. Fault injection: a forbidden item reappears in one crate's public API.
# This is the regression this whole check exists to catch, and it must fail
# closed with a diagnostic naming the offending line.
drift_dir="${tmp_dir}/snapshots/drift"
write_clean_snapshots "${drift_dir}"
run_check drift "${drift_dir}" env \
  FAKE_NEXUS_BENCH="${baseline_nexus_bench}
pub fn nexus_bench::workload::forbidden_reintroduced_facade()" \
  bash "${CHECK_SCRIPT}" check --all-features
assert_status drift 1
assert_contains "${tmp_dir}/drift.output" "ERROR: public API of 'nexus-bench' drifted from the checked-in snapshot"
assert_contains "${tmp_dir}/drift.output" "+pub fn nexus_bench::workload::forbidden_reintroduced_facade()"
assert_contains "${tmp_dir}/drift.output" "bash scripts/ci/check_public_api_surface.sh write --all-features"
assert_contains "${tmp_dir}/drift.output" "PASS: datafusion-nexus public API matches"
echo "PASS: a reintroduced forbidden item fails closed with a useful diagnostic, without masking other crates' results"

# 3. Missing snapshot file: fails closed with instructions, not a silent skip.
missing_dir="${tmp_dir}/snapshots/missing"
write_clean_snapshots "${missing_dir}"
rm -f "${missing_dir}/nexus-query-engine.txt"
run_check missing "${missing_dir}" "${CHECK_SCRIPT}" check --all-features
assert_status missing 1
assert_contains "${tmp_dir}/missing.output" "ERROR: missing public API snapshot"
assert_contains "${tmp_dir}/missing.output" "nexus-query-engine.txt"
echo "PASS: a missing snapshot fails closed instead of being silently skipped"

# 4. The underlying tool itself failing (e.g. a build error) is reported, not
# swallowed as an empty diff.
tool_failure_dir="${tmp_dir}/snapshots/tool_failure"
write_clean_snapshots "${tool_failure_dir}"
run_check tool_failure "${tool_failure_dir}" env FAKE_TOOL_FAILS_FOR=nexus-query-engine \
  bash "${CHECK_SCRIPT}" check --all-features
assert_status tool_failure 1
assert_contains "${tmp_dir}/tool_failure.output" "ERROR: cargo public-api failed for 'nexus-query-engine'"
assert_contains "${tmp_dir}/tool_failure.output" "FAKE_CARGO_PUBLIC_API_FAILURE"
echo "PASS: a failing underlying tool is reported per-crate rather than swallowed"

# 5. `write` mode regenerates a snapshot that then passes `check`, so an
# intentional API change has a real, reviewable approval path.
write_dir="${tmp_dir}/snapshots/write_roundtrip"
mkdir -p "${write_dir}"
run_check write_mode "${write_dir}" "${CHECK_SCRIPT}" write --all-features
assert_status write_mode 0
assert_contains "${tmp_dir}/write_mode.output" "WROTE: ${write_dir}/datafusion-nexus.txt"
assert_contains "${tmp_dir}/write_mode.output" "WROTE: ${write_dir}/TOOLCHAIN.txt"
diff -u "${write_dir}/nexus-bench.txt" <(printf '%s\n' "${baseline_nexus_bench}")
diff -u "${write_dir}/TOOLCHAIN.txt" <(printf '%s\n' "${baseline_toolchain_file}")
run_check write_roundtrip_check "${write_dir}" "${CHECK_SCRIPT}" check --all-features
assert_status write_roundtrip_check 0
echo "PASS: write mode regenerates snapshots (including the toolchain identity) that then pass check"

# 5b. A toolchain drift (nightly rustc or cargo-public-api version changed
# since the snapshot was written) must fail closed with its own diagnostic
# that names both identities, and must not fall through to a raw, misleading
# item-level diff. This is the regression Problem 2 exists to catch: a
# routine nightly bump must not look like an unreviewed API change.
mismatch_rustc_dir="${tmp_dir}/snapshots/mismatch_rustc"
write_clean_snapshots "${mismatch_rustc_dir}"
run_check mismatch_rustc "${mismatch_rustc_dir}" env \
  FAKE_RUSTC_NIGHTLY_VV="rustc 1.99.0-nightly (deadbeef 2026-02-02)
binary: rustc
commit-hash: deadbeef00000000000000000000000000000000
commit-date: 2026-02-02
host: x86_64-unknown-linux-gnu
release: 1.99.0-nightly
LLVM version: 22.1.8" \
  bash "${CHECK_SCRIPT}" check --all-features
assert_status mismatch_rustc 1
assert_contains "${tmp_dir}/mismatch_rustc.output" "ERROR: public API snapshots were generated with a different toolchain than the one active now."
assert_contains "${tmp_dir}/mismatch_rustc.output" "toolchain mismatch, not necessarily a public API change"
assert_contains "${tmp_dir}/mismatch_rustc.output" "fake0000 2026-01-01"
assert_contains "${tmp_dir}/mismatch_rustc.output" "deadbeef 2026-02-02"
assert_contains "${tmp_dir}/mismatch_rustc.output" "bash scripts/ci/check_public_api_surface.sh write --all-features"
assert_not_contains "${tmp_dir}/mismatch_rustc.output" "PASS:"
assert_not_contains "${tmp_dir}/mismatch_rustc.output" "drifted from the checked-in snapshot"
echo "PASS: a nightly rustc drift fails closed with a toolchain diagnostic instead of a raw item diff"

mismatch_tool_dir="${tmp_dir}/snapshots/mismatch_tool"
write_clean_snapshots "${mismatch_tool_dir}"
run_check mismatch_tool "${mismatch_tool_dir}" env FAKE_TOOL_VERSION="cargo-public-api 0.53.0-fake" \
  bash "${CHECK_SCRIPT}" check --all-features
assert_status mismatch_tool 1
assert_contains "${tmp_dir}/mismatch_tool.output" "ERROR: public API snapshots were generated with a different toolchain than the one active now."
assert_contains "${tmp_dir}/mismatch_tool.output" "0.52.0-fake"
assert_contains "${tmp_dir}/mismatch_tool.output" "0.53.0-fake"
assert_not_contains "${tmp_dir}/mismatch_tool.output" "PASS:"
echo "PASS: a cargo-public-api tool-version drift also fails closed with a toolchain diagnostic"

# 6. cargo-public-api missing entirely fails closed with an install hint,
# rather than a confusing tool-not-found error from deep inside the script.
run_check tool_missing "${clean_dir}" env PATH=/usr/bin:/bin bash "${CHECK_SCRIPT}" check --all-features
assert_status tool_missing 2
assert_contains "${tmp_dir}/tool_missing.output" "cargo-public-api is not installed"
assert_contains "${tmp_dir}/tool_missing.output" "cargo install cargo-public-api --locked"
echo "PASS: a missing cargo-public-api installation fails closed with an install hint"

# 7. Unknown mode / missing mode argument are usage errors, not silent no-ops.
run_check unknown_mode "${clean_dir}" "${CHECK_SCRIPT}" bogus-mode
assert_status unknown_mode 2
assert_contains "${tmp_dir}/unknown_mode.output" "unknown mode 'bogus-mode'"

run_check no_args "${clean_dir}" "${CHECK_SCRIPT}"
assert_status no_args 2
assert_contains "${tmp_dir}/no_args.output" "Usage: bash scripts/ci/check_public_api_surface.sh"
echo "PASS: an unknown or missing mode argument is a usage error"

facade_dir="${tmp_dir}/facade_policy"
mkdir -p "${facade_dir}"
facade_snapshot="${facade_dir}/datafusion-nexus.txt"
facade_ledger="${facade_dir}/datafusion-nexus-engine-types.tsv"
facade_output="${facade_dir}/output"
facade_header=$'public_symbol\tengine_type\tconsumer\tdecision\tfacade_path\treason\tphase'
facade_engine_type='nexus_query_engine::observability::attempt_report::AttemptId'
printf '%s\n' \
  'pub use datafusion_nexus::attempt::AttemptId' \
  "pub fn datafusion_nexus::attempt::id() -> ${facade_engine_type}" \
  >"${facade_snapshot}"
printf '%s\n' \
  "${facade_header}" \
  $'datafusion_nexus::attempt::AttemptId\tnexus_query_engine::observability::attempt_report::AttemptId\tattempt lifecycle\tRE-EXPORT\tdatafusion_nexus::attempt::AttemptId\tstable immutable attempt identity\t3' \
  >"${facade_ledger}"
python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}"
assert_contains "${facade_output}" "PASS: 1 live engine types have unique approved facade reachability"
echo "PASS: an approved selective re-export has a reachable facade path"

printf '%s\n' \
  'pub use datafusion_nexus::attempt::AttemptId' \
  "pub fn datafusion_nexus::attempt::id() -> ${facade_engine_type}" \
  'pub fn datafusion_nexus::native::plan() -> nexus_query_engine::plan::core::QueryPlan' \
  >"${facade_snapshot}"
if python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}" 2>&1; then
  echo "FAIL: an unclassified live engine type passed the facade policy" >&2
  exit 1
fi
assert_contains "${facade_output}" "live engine type has no disposition: nexus_query_engine::plan::core::QueryPlan"
echo "PASS: an unclassified live engine type fails closed"

printf '%s\n' \
  "${facade_header}" \
  $'datafusion_nexus::attempt::AttemptId\tnexus_query_engine::observability::attempt_report::AttemptId\tattempt lifecycle\tRE-EXPORT\tdatafusion_nexus::attempt::AttemptId\tstable immutable attempt identity\t3' \
  $'datafusion_nexus::native::plan\tnexus_query_engine::plan::core::QueryPlan\tnative IR\tHIDE\tdatafusion_nexus::report::PlanningReport\traw IR stays engine-owned\t5' \
  >"${facade_ledger}"
if python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}" 2>&1; then
  echo "FAIL: a live HIDE engine type passed the facade policy" >&2
  exit 1
fi
assert_contains "${facade_output}" "nexus_query_engine::plan::core::QueryPlan is live but disposition is HIDE"
echo "PASS: a HIDE engine type cannot reappear in the public surface"

echo "OK: public API surface snapshot check fault-injection tests passed."
