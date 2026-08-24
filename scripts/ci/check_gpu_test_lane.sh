#!/usr/bin/env bash
# Fail-closed preflight for the declarative test-lane policy. Admission,
# exceptions, and GPU resource classes live in .config/policies/test-lanes.toml;
# test targets declare their lane via _cpu, _gpu, _gpu_exclusive suffixes;
# scripts/ci/test_lane_policy.py validates that policy against nextest JSON.
set -uo pipefail
cd "$(dirname "$0")/../.."

REQUIRED_NEXTEST_VERSION=0.9.140

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-features) shift ;;
    *) echo "ERROR: unsupported argument for lane inventory: $1" >&2; exit 2 ;;
  esac
done

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
trap "rm -rf "${inventory_tmp}"" EXIT
workspace_inventory_json="${inventory_tmp}/workspace.json"
inventory_stderr="${inventory_tmp}/inventory.stderr"

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
list_inventory "${workspace_inventory_json}" --workspace --all-features || exit $?

validation_status=0
python3 scripts/ci/test_lane_policy.py validate-inventory \
  --inventory "${workspace_inventory_json}" || validation_status=$?
exit "${validation_status}"