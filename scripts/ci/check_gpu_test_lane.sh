#!/usr/bin/env bash
# Fail-closed preflight for the declarative test-lane catalog. Admission,
# reasons, and GPU resource classes live only in .config/policies/test-lanes.toml;
# scripts/ci/test_lane_policy.py validates that catalog against nextest JSON and
# byte-checks the generated scheduling overrides in .config/nextest.toml.
#
# Live inventory is the union of:
#   1. the ordinary default-member inventory under the requested core feature args
#   2. an explicit all-feature bench inventory
#   3. the explicit all-feature DataFusion-free Iceberg leaf inventory
# Cargo, not the catalog, selects packages; the catalog only classifies what
# pre-merge admits.
set -uo pipefail
cd "$(dirname "$0")/../.."

REQUIRED_NEXTEST_VERSION=0.9.140

if [[ $# -gt 0 ]]; then
  FEATURE_ARGS=("$@")
  FEATURE_MODE=narrow
  seen_all=0
  seen_features=0
  seen_no_default=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --all-features)
        [[ "${seen_all}" -eq 0 ]] || { echo "ERROR: duplicate --all-features." >&2; exit 2; }
        seen_all=1
        FEATURE_MODE=all
        shift
        ;;
      --features)
        [[ "${seen_features}" -eq 0 && $# -ge 2 && "${2:-}" =~ ^[A-Za-z0-9_.+/-]+(,[A-Za-z0-9_.+/-]+)*$ ]] \
          || { echo "ERROR: invalid or duplicate --features." >&2; exit 2; }
        seen_features=1
        shift 2
        ;;
      --no-default-features)
        [[ "${seen_no_default}" -eq 0 ]] || { echo "ERROR: duplicate --no-default-features." >&2; exit 2; }
        seen_no_default=1
        shift
        ;;
      *) echo "ERROR: unsupported feature argument for lane inventory: $1" >&2; exit 2 ;;
    esac
  done
  [[ ! ( "${seen_all}" -eq 1 && ( "${seen_features}" -eq 1 || "${seen_no_default}" -eq 1 ) ) ]] \
    || { echo "ERROR: --all-features conflicts with narrower feature arguments." >&2; exit 2; }
elif [[ -n "${FEATURES:-}" ]]; then
  [[ "${FEATURES}" =~ ^[A-Za-z0-9_.+/-]+(,[A-Za-z0-9_.+/-]+)*$ ]] \
    || { echo "ERROR: FEATURES must be a comma-separated feature list." >&2; exit 2; }
  FEATURE_ARGS=(--features "${FEATURES}")
  FEATURE_MODE=narrow
else
  FEATURE_ARGS=(--all-features)
  FEATURE_MODE=all
fi

nextest_version_output=$(cargo nextest --version 2>&1)
nextest_version_status=$?
nextest_first_line=${nextest_version_output%%$'\n'*}
if [[ "${nextest_version_status}" -ne 0 ]]; then
  printf '%s\n' "${nextest_version_output}" >&2
  echo "ERROR: cargo-nextest ${REQUIRED_NEXTEST_VERSION} is required to enumerate test binaries; lane safety cannot be verified." >&2
  exit 2
fi
nextest_reported_version=$(printf '%s\n' "${nextest_first_line}" | awk '$1 == "cargo-nextest" { print $2 }')
if [[ "${nextest_reported_version}" != "${REQUIRED_NEXTEST_VERSION}" ]]; then
  echo "ERROR: lane inventory requires cargo-nextest ${REQUIRED_NEXTEST_VERSION}; got '${nextest_first_line}'." >&2
  exit 2
fi

inventory_tmp=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-test-inventory.XXXXXX") || {
  echo "ERROR: could not create temporary directory for nextest inventory." >&2
  exit 2
}
trap 'rm -rf "${inventory_tmp}"' EXIT
default_inventory_json="${inventory_tmp}/default-members.json"
bench_inventory_json="${inventory_tmp}/bench.json"
iceberg_leaf_inventory_json="${inventory_tmp}/iceberg-leaf.json"
inventory_stderr="${inventory_tmp}/inventory.stderr"

# A nonzero producer status always wins over partial stdout. Stream stderr so
# Cargo build progress remains visible, while teeing it to a file preserves
# native compiler/linker diagnostics for failure investigation.
list_inventory() {
  local output_json=$1
  shift
  local inventory_status=0
  cargo nextest list "$@" --list-type binaries-only --message-format json \
    >"${output_json}" 2> >(tee -a "${inventory_stderr}" >&2) || inventory_status=$?
  if [[ "${inventory_status}" -ne 0 ]]; then
    echo "ERROR: nextest inventory command failed with status ${inventory_status}; lane safety cannot be verified." >&2
    return 2
  fi
  return 0
}

: >"${inventory_stderr}"
list_inventory "${default_inventory_json}" "${FEATURE_ARGS[@]}" || exit $?
list_inventory "${bench_inventory_json}" -p nexus-bench --all-features || exit $?
list_inventory "${iceberg_leaf_inventory_json}" -p nexus-iceberg --all-features || exit $?

validation_status=0
python3 scripts/ci/test_lane_policy.py validate-inventory \
  --inventory "${default_inventory_json}" \
  --inventory "${bench_inventory_json}" \
  --inventory "${iceberg_leaf_inventory_json}" \
  --feature-mode "${FEATURE_MODE}" || validation_status=$?
exit "${validation_status}"
