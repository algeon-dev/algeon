#!/usr/bin/env bash
# Fault-injection tests for the public-API-surface snapshot check. `cargo
# public-api` itself is faked out (a `cargo-public-api` shim on PATH), so this
# never runs a real rustdoc-JSON build: it proves the comparison/reporting
# logic in scripts/ci/check_public_api_surface.sh fails closed on drift, a
# missing snapshot, and a failing underlying tool, and passes on a clean
# baseline and after `write` regenerates it.
set -euo pipefail
export PUBLIC_API_SKIP_NATIVE_COMPONENT_POLICY=1

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/algeon-datafusion-public-api-surface-tests.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin" "${tmp_dir}/snapshots"

# A deterministic, tiny fake "public API" per crate, independent of the real
# crates in this repository.
baseline_algeon_datafusion='pub fn algeon_datafusion::backend::AlgeonGpuBackend::builder() -> algeon_datafusion::backend::AlgeonGpuBackendBuilder
pub struct algeon_datafusion::backend::AlgeonGpuBackend'
baseline_query_engine='pub fn query_engine::runtime::GpuBackendSession::admit_query(&self) -> query_engine::error::Result<query_engine::runtime::QueryHandle>'
baseline_query_planning='pub struct query_planning::plan::QueryPlan'
baseline_query_runtime='pub struct query_runtime::runtime::GpuBackendSession'
baseline_interop='pub struct rapids_interop::PreparedTableExport'
baseline_bench='pub mod bench::tpch_benchmark'
baseline_cudf='pub mod cudf'
baseline_cugraph='pub mod cugraph'

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
  algeon-datafusion) printf '%s\n' "\${FAKE_DATAFUSION_ALGEON:-${baseline_algeon_datafusion}}" ;;
  algeon-query-engine) printf '%s\n' "\${FAKE_ALGEON_QUERY_ENGINE:-${baseline_query_engine}}" ;;
  algeon-query-planning) printf '%s\n' "\${FAKE_ALGEON_QUERY_PLANNING:-${baseline_query_planning}}" ;;
  algeon-query-runtime) printf '%s\n' "\${FAKE_ALGEON_QUERY_RUNTIME:-${baseline_query_runtime}}" ;;
  algeon-interop) printf '%s\n' "\${FAKE_ALGEON_INTEROP:-${baseline_interop}}" ;;
  bench) printf '%s\n' "\${FAKE_ALGEON_BENCH:-${baseline_bench}}" ;;
  algeon-cudf) printf '%s\n' "\${FAKE_CUDF_ALGEON:-${baseline_cudf}}" ;;
  algeon-cugraph) printf '%s\n' "\${FAKE_CUGRAPH_ALGEON:-${baseline_cugraph}}" ;;
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
  printf '%s\n' "${baseline_algeon_datafusion}" >"${dir}/algeon-datafusion.txt"
  printf '%s\n' "${baseline_query_engine}" >"${dir}/algeon-query-engine.txt"
  printf '%s\n' "${baseline_query_planning}" >"${dir}/algeon-query-planning.txt"
  printf '%s\n' "${baseline_query_runtime}" >"${dir}/algeon-query-runtime.txt"
  printf '%s\n' "${baseline_interop}" >"${dir}/algeon-interop.txt"
  printf '%s\n' "${baseline_bench}" >"${dir}/bench.txt"
  printf '%s\n' "${baseline_cudf}" >"${dir}/algeon-cudf.txt"
  printf '%s\n' "${baseline_cugraph}" >"${dir}/algeon-cugraph.txt"
  cp .config/policies/public-api/*-hidden.tsv "${dir}/"
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

# 1. Clean baseline: check passes for all eight independently snapshotted crates
# and every hidden ledger matches its crate's source.
clean_dir="${tmp_dir}/snapshots/clean"
write_clean_snapshots "${clean_dir}"
run_check clean "${clean_dir}" "${CHECK_SCRIPT}" check --all-features
assert_status clean 0
assert_contains "${tmp_dir}/clean.output" "PASS: algeon-datafusion public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: algeon-query-engine public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: algeon-query-planning public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: algeon-query-runtime public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: algeon-interop public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: bench public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: algeon-cudf public API matches"
assert_contains "${tmp_dir}/clean.output" "PASS: algeon-cugraph public API matches"
for hidden_ledger in algeon-cudf algeon-datafusion algeon-query-engine algeon-query-planning algeon-query-runtime; do
  assert_contains "${tmp_dir}/clean.output" "hidden symbols license-checked against ${hidden_ledger}-hidden.tsv"
done
echo "PASS: clean snapshot matches for all eight independently snapshotted crates and five hidden ledgers"

# 2. Fault injection: a forbidden item reappears in one crate's public API.
# This is the regression this whole check exists to catch, and it must fail
# closed with a diagnostic naming the offending line.
drift_dir="${tmp_dir}/snapshots/drift"
write_clean_snapshots "${drift_dir}"
run_check drift "${drift_dir}" env \
  FAKE_ALGEON_INTEROP="${baseline_interop}
pub fn rapids_interop::forbidden_reintroduced_contract()" \
  bash "${CHECK_SCRIPT}" check --all-features
assert_status drift 1
assert_contains "${tmp_dir}/drift.output" "ERROR: public API of 'algeon-interop' drifted from the checked-in snapshot"
assert_contains "${tmp_dir}/drift.output" "+pub fn rapids_interop::forbidden_reintroduced_contract()"
assert_contains "${tmp_dir}/drift.output" "bash scripts/ci/check_public_api_surface.sh write --all-features"
assert_contains "${tmp_dir}/drift.output" "PASS: algeon-datafusion public API matches"
echo "PASS: a reintroduced interop item fails closed with a useful diagnostic, without masking other crates' results"

# 3. Missing snapshot file: fails closed with instructions, not a silent skip.
missing_dir="${tmp_dir}/snapshots/missing"
write_clean_snapshots "${missing_dir}"
rm -f "${missing_dir}/algeon-query-engine.txt"
run_check missing "${missing_dir}" "${CHECK_SCRIPT}" check --all-features
assert_status missing 1
assert_contains "${tmp_dir}/missing.output" "ERROR: missing public API snapshot"
assert_contains "${tmp_dir}/missing.output" "algeon-query-engine.txt"
echo "PASS: a missing snapshot fails closed instead of being silently skipped"

# 4. The underlying tool itself failing (e.g. a build error) is reported, not
# swallowed as an empty diff.
tool_failure_dir="${tmp_dir}/snapshots/tool_failure"
write_clean_snapshots "${tool_failure_dir}"
run_check tool_failure "${tool_failure_dir}" env FAKE_TOOL_FAILS_FOR=algeon-query-engine \
  bash "${CHECK_SCRIPT}" check --all-features
assert_status tool_failure 1
assert_contains "${tmp_dir}/tool_failure.output" "ERROR: cargo public-api failed for 'algeon-query-engine'"
assert_contains "${tmp_dir}/tool_failure.output" "FAKE_CARGO_PUBLIC_API_FAILURE"
echo "PASS: a failing underlying tool is reported per-crate rather than swallowed"

# 5. `write` mode regenerates a snapshot that then passes `check`, so an
# intentional API change has a real, reviewable approval path.
write_dir="${tmp_dir}/snapshots/write_roundtrip"
mkdir -p "${write_dir}"
cp .config/policies/public-api/*-hidden.tsv "${write_dir}/"
run_check write_mode "${write_dir}" "${CHECK_SCRIPT}" write --all-features
assert_status write_mode 0
assert_contains "${tmp_dir}/write_mode.output" "WROTE: ${write_dir}/algeon-datafusion.txt"
assert_contains "${tmp_dir}/write_mode.output" "WROTE: ${write_dir}/algeon-interop.txt"
assert_contains "${tmp_dir}/write_mode.output" "WROTE: ${write_dir}/TOOLCHAIN.txt"
diff -u "${write_dir}/bench.txt" <(printf '%s\n' "${baseline_bench}")
diff -u "${write_dir}/algeon-interop.txt" <(printf '%s\n' "${baseline_interop}")
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
facade_snapshot="${facade_dir}/algeon-datafusion.txt"
facade_ledger="${facade_dir}/algeon-datafusion-engine-types.tsv"
facade_output="${facade_dir}/output"
facade_header=$'public_symbol\tengine_type\tconsumer\tdecision\tfacade_path\treason\tphase'
facade_engine_type='query_engine::observability::attempt_report::AttemptId'
printf '%s\n' \
  'pub use algeon_datafusion::attempt::AttemptId' \
  "pub fn algeon_datafusion::attempt::id() -> ${facade_engine_type}" \
  >"${facade_snapshot}"
printf '%s\n' \
  "${facade_header}" \
  $'algeon_datafusion::attempt::AttemptId\tquery_engine::observability::attempt_report::AttemptId\tattempt lifecycle\tRE-EXPORT\talgeon_datafusion::attempt::AttemptId\tstable immutable attempt identity\t3' \
  >"${facade_ledger}"
python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}"
assert_contains "${facade_output}" "PASS: 1 live engine types have unique approved facade reachability"
echo "PASS: an approved selective re-export has a reachable facade path"

printf '%s\n' \
  'pub use algeon_datafusion::attempt::RuntimeTerminalOutcome' \
  >"${facade_snapshot}"
printf '%s\n' \
  "${facade_header}" \
  $'algeon_datafusion::attempt::RuntimeTerminalOutcome\tquery_engine::observability::RuntimeTerminalOutcome\tattempt observability\tRE-EXPORT\talgeon_datafusion::attempt::RuntimeTerminalOutcome\tstable terminal outcome\t3' \
  >"${facade_ledger}"
python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}"
assert_contains "${facade_output}" "PASS: 1 live engine types have unique approved facade reachability"
echo "PASS: a pure facade re-export has a live disposition"

printf '%s\n' \
  'pub use algeon_datafusion::attempt::RuntimeTerminalOutcome' \
  >"${facade_snapshot}"
printf '%s\n' \
  "${facade_header}" \
  $'algeon_datafusion::attempt::RuntimeTerminalOutcome\tquery_engine::observability::RuntimeTerminalOutcome\tattempt observability\tHIDE\talgeon_datafusion::attempt::RuntimeTerminalOutcome\tinternal terminal outcome\t3' \
  >"${facade_ledger}"
if python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}" 2>&1; then
  echo "FAIL: a pure facade re-export with HIDE disposition passed the facade policy" >&2
  exit 1
fi
assert_contains "${facade_output}" "forbidden HIDE type reappeared in the live surface: query_engine::observability::RuntimeTerminalOutcome"
echo "PASS: a pure facade re-export cannot bypass a HIDE disposition"

printf '%s\n' \
  'pub use algeon_datafusion::attempt::AttemptId' \
  "pub fn algeon_datafusion::attempt::id() -> ${facade_engine_type}" \
  'pub fn algeon_datafusion::native::plan() -> query_engine::plan::core::QueryPlan' \
  >"${facade_snapshot}"
if python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}" 2>&1; then
  echo "FAIL: an unclassified live engine type passed the facade policy" >&2
  exit 1
fi
assert_contains "${facade_output}" "live engine type has no disposition: query_engine::plan::core::QueryPlan"
echo "PASS: an unclassified live engine type fails closed"

printf '%s\n' \
  "${facade_header}" \
  $'algeon_datafusion::attempt::AttemptId\tquery_engine::observability::attempt_report::AttemptId\tattempt lifecycle\tRE-EXPORT\talgeon_datafusion::attempt::AttemptId\tstable immutable attempt identity\t3' \
  $'algeon_datafusion::native::plan\tquery_engine::plan::core::QueryPlan\tnative IR\tHIDE\talgeon_datafusion::report::PlanningReport\traw IR stays engine-owned\t5' \
  >"${facade_ledger}"
if python3 scripts/ci/check_public_api_facade.py "${facade_snapshot}" "${facade_ledger}" >"${facade_output}" 2>&1; then
  echo "FAIL: a live HIDE engine type passed the facade policy" >&2
  exit 1
fi
assert_contains "${facade_output}" "query_engine::plan::core::QueryPlan is live but disposition is HIDE"
echo "PASS: a HIDE engine type cannot reappear in the public surface"


# Native component top-level namespace policy fault-injection tests
native_policy_dir="${tmp_dir}/native_policy"
mkdir -p "${native_policy_dir}"
native_snapshot_dir="${native_policy_dir}/snapshots"
mkdir -p "${native_snapshot_dir}"

cat >"${native_snapshot_dir}/algeon-cudf.txt" <<'EOF'
pub mod cudf
pub mod cudf::core
pub mod cudf::device
pub mod cudf::frame
pub mod cudf::hash
pub mod cudf::interop
pub mod cudf::io
pub mod cudf::ops
pub mod cudf::runtime
EOF

cat >"${native_snapshot_dir}/algeon-cugraph.txt" <<'EOF'
pub mod cugraph
pub mod cugraph::array
pub mod cugraph::centrality
pub mod cugraph::community
pub mod cugraph::cudf_interop
pub mod cugraph::error
pub mod cugraph::graph
pub mod cugraph::handle
pub mod cugraph::labeling
pub mod cugraph::layout
pub mod cugraph::params
pub mod cugraph::random
pub mod cugraph::sampling
pub mod cugraph::similarity
pub mod cugraph::structure
pub mod cugraph::traversal
pub mod cugraph::tree
pub mod cugraph::types
EOF

native_policy_file="${native_policy_dir}/native-component-surfaces.toml"
cp .config/policies/native-component-surfaces.toml "${native_policy_file}"

if ! python3 scripts/ci/check_native_component_architecture.py "${native_snapshot_dir}" "${native_policy_file}" >"${tmp_dir}/native_baseline.output" 2>&1; then
  echo "FAIL: native component architecture clean baseline failed" >&2
  cat "${tmp_dir}/native_baseline.output" >&2
  exit 1
fi
echo "PASS: native component architecture clean baseline passed"

cat >>"${native_snapshot_dir}/algeon-cudf.txt" <<'EOF'
pub mod cudf::unapproved_new_ns
EOF
if python3 scripts/ci/check_native_component_architecture.py "${native_snapshot_dir}" "${native_policy_file}" >"${tmp_dir}/native_unapproved.output" 2>&1; then
  echo "FAIL: unapproved top-level namespace passed policy" >&2
  exit 1
fi
assert_contains "${tmp_dir}/native_unapproved.output" "unapproved top-level namespaces for algeon-cudf: unapproved_new_ns"
echo "PASS: unapproved top-level namespace fails closed"

sed -i '/pub mod cudf::unapproved_new_ns/d' "${native_snapshot_dir}/algeon-cudf.txt"
sed -i '/pub mod cudf::runtime/d' "${native_snapshot_dir}/algeon-cudf.txt"
if python3 scripts/ci/check_native_component_architecture.py "${native_snapshot_dir}" "${native_policy_file}" >"${tmp_dir}/native_stale.output" 2>&1; then
  echo "FAIL: stale approved namespace passed policy" >&2
  exit 1
fi
assert_contains "${tmp_dir}/native_stale.output" "stale approved top-level namespaces for algeon-cudf: runtime"
echo "PASS: stale approved top-level namespace fails closed"

# Hidden-ledger fault injection through the surface script: one crate's ledger
# loses a row and another gains a stale row, and each failure names its symbol
# without masking the other crates' results.
hidden_drift_dir="${tmp_dir}/snapshots/hidden_drift"
write_clean_snapshots "${hidden_drift_dir}"
sed -i '0,/^query_runtime::/{//d}' "${hidden_drift_dir}/algeon-query-runtime-hidden.tsv"
printf 'query_planning::plan::core::gone\tcross-crate\towner\tconsumer\tcondition\n' \
  >>"${hidden_drift_dir}/algeon-query-planning-hidden.tsv"
run_check hidden_drift "${hidden_drift_dir}" "${CHECK_SCRIPT}" check --all-features
assert_status hidden_drift 1
assert_contains "${tmp_dir}/hidden_drift.output" "unlisted hidden symbol query_runtime::"
assert_contains "${tmp_dir}/hidden_drift.output" "stale ledger entry query_planning::plan::core::gone [cross-crate] matches no hidden site"
assert_contains "${tmp_dir}/hidden_drift.output" "hidden symbols license-checked against algeon-query-engine-hidden.tsv"
echo "PASS: a missing or stale row in one crate's hidden ledger fails closed without masking the others"

# Hidden-ledger scanner fixtures: throwaway crates scanned in isolation, so
# these never depend on the real crates' hidden surface.
HIDDEN_CHECKER=scripts/ci/check_public_api_hidden.py
hidden_dir="${tmp_dir}/hidden"

write_hidden_ledger() {
  local file=$1
  shift
  local row
  printf 'symbol\tfeature\towner\tconsumer\tremoval_condition\n' >"${file}"
  for row in "$@"; do
    printf '%s\t%s\towner\tconsumer\tcondition\n' "${row%%|*}" "${row##*|}" >>"${file}"
  done
}

write_clean_ledger_variant() {
  local file=$1
  local excluded=$2
  shift 2
  local rows=()
  local row
  for row in "${hidden_clean_rows[@]}"; do
    [[ "${row}" == "${excluded}" ]] || rows+=("${row}")
  done
  write_hidden_ledger "${file}" "${rows[@]}" "$@"
}

run_hidden() {
  local name=$1
  local src=$2
  local ledger=$3
  local status=0
  python3 "${HIDDEN_CHECKER}" "${src}" fixture "${ledger}" >"${tmp_dir}/${name}.output" 2>&1 || status=$?
  printf '%s\n' "${status}" >"${tmp_dir}/${name}.status"
}

mkdir -p "${hidden_dir}/clean/src"
cat >"${hidden_dir}/clean/src/lib.rs" <<'RS'
pub mod engine;
#[cfg(feature = "test-hooks")]
mod seam;
#[cfg(test)]
mod tests;
#[cfg(any(test, feature = "server"))]
pub mod frontend;
#[cfg(feature = "bench-support")]
pub mod bench {
    #[doc(hidden)]
    pub fn bench_probe() {}
}
#[cfg(feature = "cuml")]
#[doc(hidden)]
pub fn capability_only() {}
#[cfg(test)]
mod unit {
    #[doc(hidden)]
    pub fn test_only() {}
}
#[cfg(all(test, feature = "server"))]
#[doc(hidden)]
pub fn test_and_server() {}

#[doc(hidden)]
pub use engine::{Alpha, Beta as Gamma, nested::{Delta, Epsilon}};

pub struct Config {
    #[doc(hidden)]
    pub tuning: u8,
    pub open: u8,
}

pub enum Mode {
    A,
    #[doc(hidden)]
    Internal,
}

#[doc(hidden)]
pub struct Legacy(#[doc(hidden)] pub usize);

#[doc(hidden)]
pub(crate) fn narrowed() {}
RS
cat >"${hidden_dir}/clean/src/engine.rs" <<'RS'
use std::fmt;

pub struct Alpha;
pub struct Beta;
pub struct Wrapper<T>(pub T);

pub mod nested {
    pub struct Delta;
    pub struct Epsilon;

    impl super::Alpha {
        #[doc(hidden)]
        pub fn from_nested() -> Self {
            Self
        }
    }
}

impl Alpha {
    #[doc(hidden)]
    pub fn new() -> Self {
        Self
    }

    #[doc(hidden)]
    pub(crate) fn narrowed(&self) {}
}

impl Beta {
    #[doc(hidden)]
    pub fn new() -> Self {
        Self
    }
}

impl<T> From<T> for Wrapper<T> {
    #[doc(hidden)]
    fn from(value: T) -> Self {
        Wrapper(value)
    }
}

impl<T: Clone> fmt::Display for Wrapper<T> {
    #[doc(hidden)]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("wrapper")
    }
}

pub trait Probe {
    #[doc(hidden)]
    fn probe(&self);
}
RS
printf '#[doc(hidden)]\npub fn hook() {}\n' >"${hidden_dir}/clean/src/seam.rs"
printf '#[doc(hidden)]\npub struct Handler;\n' >"${hidden_dir}/clean/src/frontend.rs"
printf '#[doc(hidden)]\npub fn only_in_tests() {}\n' >"${hidden_dir}/clean/src/tests.rs"

hidden_clean_rows=(
  'fixture::Alpha|cross-crate'
  'fixture::Config.tuning|cross-crate'
  'fixture::Delta|cross-crate'
  'fixture::Epsilon|cross-crate'
  'fixture::Gamma|cross-crate'
  'fixture::Legacy|cross-crate'
  'fixture::Legacy.0|cross-crate'
  'fixture::Mode::Internal|cross-crate'
  'fixture::bench::bench_probe|bench-support'
  'fixture::capability_only|cross-crate'
  'fixture::engine::<Wrapper as Display>::fmt|cross-crate'
  'fixture::engine::<Wrapper as From<T>>::from|cross-crate'
  'fixture::engine::Alpha::new|cross-crate'
  'fixture::engine::Beta::new|cross-crate'
  'fixture::engine::Probe::probe|cross-crate'
  'fixture::engine::nested::Alpha::from_nested|cross-crate'
  'fixture::frontend::Handler|server'
  'fixture::seam::hook|test-hooks'
)

# Same-named methods in two impls, inherent and trait impls, `pub use` groups,
# hidden fields and variants, and cfg-derived features resolve to one row each;
# cfg(test) items and pub(crate) items need no row.
write_clean_ledger_variant "${hidden_dir}/clean.tsv" ""
run_hidden hidden_clean "${hidden_dir}/clean/src" "${hidden_dir}/clean.tsv"
assert_status hidden_clean 0
assert_contains "${tmp_dir}/hidden_clean.output" "PASS: 18 hidden symbols license-checked against clean.tsv"
echo "PASS: hidden scanner resolves impls, trait impls, use groups, members, and cfg-derived features"

write_clean_ledger_variant "${hidden_dir}/missing_beta.tsv" 'fixture::engine::Beta::new|cross-crate'
run_hidden hidden_missing_beta "${hidden_dir}/clean/src" "${hidden_dir}/missing_beta.tsv"
assert_status hidden_missing_beta 1
assert_contains "${tmp_dir}/hidden_missing_beta.output" "unlisted hidden symbol fixture::engine::Beta::new [cross-crate] has no ledger entry"
assert_not_contains "${tmp_dir}/hidden_missing_beta.output" "unlisted hidden symbol fixture::engine::Alpha::new"
echo "PASS: a hidden method without a row fails without implicating a same-named method of another type"

write_clean_ledger_variant "${hidden_dir}/module_level.tsv" 'fixture::engine::Alpha::new|cross-crate' \
  'fixture::engine::new|cross-crate'
run_hidden hidden_module_level "${hidden_dir}/clean/src" "${hidden_dir}/module_level.tsv"
assert_status hidden_module_level 1
assert_contains "${tmp_dir}/hidden_module_level.output" "unlisted hidden symbol fixture::engine::Alpha::new [cross-crate]"
assert_contains "${tmp_dir}/hidden_module_level.output" "stale ledger entry fixture::engine::new [cross-crate] matches no hidden site"
echo "PASS: a module-qualified row cannot stand in for a method of a type"

write_clean_ledger_variant "${hidden_dir}/wrong_feature.tsv" 'fixture::seam::hook|test-hooks' \
  'fixture::seam::hook|cross-crate'
run_hidden hidden_wrong_feature "${hidden_dir}/clean/src" "${hidden_dir}/wrong_feature.tsv"
assert_status hidden_wrong_feature 1
assert_contains "${tmp_dir}/hidden_wrong_feature.output" "unlisted hidden symbol fixture::seam::hook [test-hooks]"
assert_contains "${tmp_dir}/hidden_wrong_feature.output" "stale ledger entry fixture::seam::hook [cross-crate]"
echo "PASS: a row whose feature differs from the cfg-derived feature fails in both directions"

write_clean_ledger_variant "${hidden_dir}/unapproved.tsv" 'fixture::seam::hook|test-hooks' \
  'fixture::seam::hook|nightly'
run_hidden hidden_unapproved "${hidden_dir}/clean/src" "${hidden_dir}/unapproved.tsv"
assert_status hidden_unapproved 1
assert_contains "${tmp_dir}/hidden_unapproved.output" "has unapproved feature 'nightly'"
echo "PASS: a ledger feature outside the approved set is rejected"

write_clean_ledger_variant "${hidden_dir}/stale.tsv" "" 'fixture::engine::removed|cross-crate'
run_hidden hidden_stale "${hidden_dir}/clean/src" "${hidden_dir}/stale.tsv"
assert_status hidden_stale 1
assert_contains "${tmp_dir}/hidden_stale.output" "stale ledger entry fixture::engine::removed [cross-crate] matches no hidden site"
echo "PASS: a ledger row without a site fails closed"

# A hidden attribute the scanner cannot attach to an item must not pass silently.
mkdir -p "${hidden_dir}/macro/src"
cat >"${hidden_dir}/macro/src/lib.rs" <<'RS'
macro_rules! make {
    () => {
        #[doc(hidden)]
        pub fn generated() {}
    };
}
make!();
RS
write_hidden_ledger "${hidden_dir}/macro.tsv"
run_hidden hidden_macro "${hidden_dir}/macro/src" "${hidden_dir}/macro.tsv"
assert_status hidden_macro 1
assert_contains "${tmp_dir}/hidden_macro.output" "#[doc(hidden)] in a position the scanner does not classify"
echo "PASS: a hidden attribute inside a macro body fails closed"

# Two impls that differ only in generic arguments would share one symbol.
mkdir -p "${hidden_dir}/ambiguous/src"
cat >"${hidden_dir}/ambiguous/src/lib.rs" <<'RS'
pub struct Holder<T>(T);

impl Holder<u8> {
    #[doc(hidden)]
    pub fn build() {}
}

impl Holder<u16> {
    #[doc(hidden)]
    pub fn build() {}
}
RS
write_hidden_ledger "${hidden_dir}/ambiguous.tsv" 'fixture::Holder::build|cross-crate'
run_hidden hidden_ambiguous "${hidden_dir}/ambiguous/src" "${hidden_dir}/ambiguous.tsv"
assert_status hidden_ambiguous 1
assert_contains "${tmp_dir}/hidden_ambiguous.output" "fixture::Holder::build [cross-crate] is recorded at several sites"
echo "PASS: a symbol shared by two sites is reported as ambiguous instead of merged"

# A file no `mod` declaration reaches cannot be classified, so a hidden item in it fails.
mkdir -p "${hidden_dir}/orphan/src"
printf 'pub mod reachable;\n' >"${hidden_dir}/orphan/src/lib.rs"
printf 'pub fn plain() {}\n' >"${hidden_dir}/orphan/src/reachable.rs"
printf '#[doc(hidden)]\npub fn stray() {}\n' >"${hidden_dir}/orphan/src/stray.rs"
write_hidden_ledger "${hidden_dir}/orphan.tsv"
run_hidden hidden_orphan "${hidden_dir}/orphan/src" "${hidden_dir}/orphan.tsv"
assert_status hidden_orphan 1
assert_contains "${tmp_dir}/hidden_orphan.output" "no \`mod\` declaration reaches it"
echo "PASS: a hidden item in a file outside the module tree fails closed"

echo "OK: public API surface snapshot check fault-injection tests passed."
