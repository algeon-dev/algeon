#!/usr/bin/env bash
# Tripwire: every workspace test binary must be classified into exactly one test
# lane -- the GPU lane (also listed in the `gpu` test-group of .config/nextest.toml)
# or the CPU lane. Adding a new test binary, renaming one, or enabling a new bin
# target fails this check until it is classified. That makes lane membership an
# explicit, reviewed decision and prevents a GPU-executing test from silently
# landing in the unsynchronized default lane (where it would run unthrottled and
# without the /tmp/cudf-gpu.lock flock).
#
# Why a hand-maintained classification instead of detecting GPU use automatically:
# GPU use is a RUNTIME property reached indirectly. Most GPU tests never name
# `cudf::Runtime` in their own source -- they drive the device through a
# DataFusion SessionContext, the cuGraph table functions, or the Flight SQL
# server, which create the runtime inside src/. So there is no sound static
# signal: grepping for `cudf::Runtime` misses those indirect users, while coarse
# markers like `.collect()` false-match CPU-only tests. Classification is
# therefore human and explicit, keyed on the full nextest binary id; this script
# guards against silent drift and is run by scripts/check_all.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${FEATURES:-}" ]]; then
  FEATURE_ARGS=(--features "${FEATURES}")
else
  FEATURE_ARGS=(--all-features)
fi

# GPU lane -- binaries with at least one test that executes on the GPU. The
# binary NAMES here must match the binary(...) entries in the gpu test-group of
# .config/nextest.toml (consistency is checked below).
GPU_LANE="
nexus-query-engine
nexus-query-engine::native_execution_tests
nexus-query-engine::native_contract_tests
nexus-query-engine::query_service_allocator_lifecycle
datafusion-nexus::cugraph_tests
datafusion-nexus::iceberg_tests
datafusion-nexus::native_exec_tests
datafusion-nexus::native_runtime_pruning_tests
datafusion-nexus::native_tpcds_parity_tests
datafusion-nexus::server_cugraph_tests
datafusion-nexus::server_tests
"

# CPU lane -- every other test binary, reviewed NOT to execute on the GPU:
# planning/report/parser/config/classification tests, lib unit tests, and the
# bin targets' own unit tests (GPU benchmark EXECUTION runs via
# `cargo run -p datafusion-nexus-bench --bin ...`, not under nextest). If you
# make one of these execute on the GPU, MOVE it to the gpu test-group in
# .config/nextest.toml (both override blocks) and to GPU_LANE above.
CPU_LANE="
datafusion-nexus
datafusion-nexus-bench
datafusion-nexus-workloads
datafusion-nexus::bin/datafusion_nexus_server
datafusion-nexus::error_tests
datafusion-nexus::native_fallback_pressure_tests
datafusion-nexus::native_lowering_tests
datafusion-nexus::native_metrics_tests
datafusion-nexus::policy_report_tests
datafusion-nexus::public_facade_tests
datafusion-nexus::server_iceberg_tests
datafusion-nexus::session_config_tests
datafusion-nexus::table_format_scan_tests
datafusion-nexus-bench::clickbench_workload_tests
datafusion-nexus-bench::iceberg_bin_tests
datafusion-nexus-bench::native_clickbench_execution_report_tests
datafusion-nexus-bench::native_policy_run_evidence_tests
datafusion-nexus-bench::tpcds_workload_tests
datafusion-nexus-bench::tpch_workload_tests
datafusion-nexus-bench::bin/cpu_tpch_profile_sweep
datafusion-nexus-bench::bin/c1_mixed_fallback_experiment
datafusion-nexus-bench::bin/cugraph_admission_stress
datafusion-nexus-bench::bin/cugraph_sql_construction_report
datafusion-nexus-bench::bin/empty_input_agg_probe
datafusion-nexus-bench::bin/native_clickbench_benchmark
datafusion-nexus-bench::bin/native_clickbench_execution_report
datafusion-nexus-bench::bin/native_datafusion_planning_report
datafusion-nexus-bench::bin/native_fallback_pressure_report
datafusion-nexus-bench::bin/native_iceberg_smoke_report
datafusion-nexus-bench::bin/native_iceberg_tpch_benchmark
datafusion-nexus-bench::bin/native_tpcds_execution_report
datafusion-nexus-bench::bin/native_tpcds_parity_report
datafusion-nexus-bench::bin/native_tpch_benchmark
datafusion-nexus-bench::bin/native_tpch_execution_report
datafusion-nexus-bench::bin/query_service_iceberg_stress
datafusion-nexus-bench::bin/query_service_tpcds_stress
datafusion-nexus-bench::bin/query_service_tpch_stress
datafusion-nexus-bench::bin/substrait_ingress_experiment
datafusion-nexus-bench::bin/tpcds_outlier_triage
datafusion-nexus-bench::bin/tpcds_three_way_bench
"

if ! cargo nextest --version >/dev/null 2>&1; then
  echo "WARN: cargo-nextest not installed; skipping GPU test-lane classification check."
  exit 0
fi

status=0

gpu_lane_sorted=$(printf '%s\n' "${GPU_LANE}" | grep -vE '^\s*$' | sed -E 's/^[[:space:]]+//' | sort -u)
cpu_lane_sorted=$(printf '%s\n' "${CPU_LANE}" | grep -vE '^\s*$' | sed -E 's/^[[:space:]]+//' | sort -u)

# Exactly-one-lane: a binary must not appear in BOTH lanes (sort -u in the
# completeness check below would otherwise hide the overlap).
overlap=$(comm -12 <(printf '%s\n' "${gpu_lane_sorted}") <(printf '%s\n' "${cpu_lane_sorted}"))
if [[ -n "${overlap}" ]]; then
  status=1
  echo "FAIL: binaries listed in BOTH GPU_LANE and CPU_LANE (must be exactly one):" >&2
  echo "${overlap}" | sed 's/^/  - /' >&2
fi

# Consistency: EACH gpu test-group override block in .config/nextest.toml must
# contain exactly the GPU_LANE entries. Comparing package-qualified integration
# test/bin targets and package lib targets per block catches (a) the two
# override blocks (default + gpu-parallel) drifting apart and (b) a wrong
# package(...) paired with a right binary name -- both of which a name-only,
# file-wide check would miss.
override_block_pairs() { # $1 = profile name (default | gpu-parallel)
  block=$(awk -v prof="$1" '
    /^\[\[profile\./ { inblk = ($0 == "[[profile." prof ".overrides]]") }
    inblk && /package\(/ { print }
  ' .config/nextest.toml)
  {
    printf '%s\n' "${block}" \
      | grep -oE 'package\([a-z0-9_-]+\)[[:space:]]*&[[:space:]]*binary\([a-z0-9_]+\)' \
      | sed -E 's/package\(([a-z0-9_-]+)\)[[:space:]]*&[[:space:]]*binary\(([a-z0-9_]+)\)/\1::\2/'
    printf '%s\n' "${block}" \
      | grep -oE 'package\([a-z0-9_-]+\)[[:space:]]*&[[:space:]]*kind\(lib\)' \
      | sed -E 's/package\(([a-z0-9_-]+)\)[[:space:]]*&[[:space:]]*kind\(lib\)/\1/'
  } | sort -u
}
for prof in default gpu-parallel; do
  block_pairs=$(override_block_pairs "${prof}")
  if [[ "${block_pairs}" != "${gpu_lane_sorted}" ]]; then
    status=1
    echo "FAIL: gpu test-group [[profile.${prof}.overrides]] in .config/nextest.toml does not match GPU_LANE:" >&2
    diff <(printf '%s\n' "${gpu_lane_sorted}") <(printf '%s\n' "${block_pairs}") | sed 's/^/  /' >&2
  fi
done

# Completeness: every live test binary must be classified into exactly one lane.
expected=$(printf '%s\n%s\n' "${GPU_LANE}" "${CPU_LANE}" | grep -vE '^\s*$' | sed -E 's/^[[:space:]]+//' | sort -u)
live=$(cargo nextest list "${FEATURE_ARGS[@]}" --list-type binaries-only 2>/dev/null \
  | sed -E 's/^[[:space:]]+//' | grep -E '.' | sort -u)

if [[ -z "${live}" ]]; then
  echo "ERROR: could not enumerate test binaries (build failed?). Not skipping silently." >&2
  exit 2
fi

# nextest's `binary(<name>)` filter (used by the gpu test-group) and the
# GPU_LANE-vs-nextest.toml consistency check both key on the binary NAME, not the
# full `package::binary` id. Two packages sharing a test-binary name would make
# both ambiguous, so fail loudly if any name collides (strip the package prefix
# and any `bin/` segment to match nextest's binary() semantics).
dup_names=$(printf '%s\n' "${live}" | sed -E 's#.*::##; s#^bin/##' | sort | uniq -d)
if [[ -n "${dup_names}" ]]; then
  status=1
  echo "FAIL: test-binary name collides across packages (breaks binary(<name>) grouping):" >&2
  echo "${dup_names}" | sed 's/^/  - /' >&2
fi

unclassified=$(comm -13 <(printf '%s\n' "${expected}") <(printf '%s\n' "${live}"))
stale=$(comm -23 <(printf '%s\n' "${expected}") <(printf '%s\n' "${live}"))

if [[ -n "${unclassified}" ]]; then
  status=1
  echo "FAIL: unclassified test binaries (neither GPU lane nor CPU lane):" >&2
  echo "${unclassified}" | sed 's/^/  - /' >&2
  echo "Classify each in scripts/check_gpu_test_lane.sh:" >&2
  echo "  * executes on the GPU -> add to GPU_LANE here AND binary(<name>) in BOTH" >&2
  echo "    gpu override blocks of .config/nextest.toml;" >&2
  echo "  * otherwise            -> add to CPU_LANE here." >&2
fi
if [[ -n "${stale}" ]]; then
  # Under a narrower FEATURES set, feature-gated binaries legitimately disappear,
  # so a stale entry is a warning, not a failure.
  echo "WARN: classified binaries not present in this run (rename/removal, or feature-gated off):" >&2
  echo "${stale}" | sed 's/^/  - /' >&2
fi
if [[ "${status}" -eq 0 ]]; then
  echo "OK: all $(printf '%s\n' "${live}" | wc -l | tr -d ' ') test binaries are classified (GPU lane vs CPU lane)."
fi
exit "${status}"
