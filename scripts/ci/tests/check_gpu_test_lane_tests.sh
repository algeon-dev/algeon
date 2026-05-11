#!/usr/bin/env bash
# Fault-injection tests for the catalog-driven, fail-closed lane preflight.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-gpu-lane-tests.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin"

cat >"${tmp_dir}/bin/cargo" <<'SH'
#!/usr/bin/bash
set -u
if [[ "${1:-}" == "nextest" && "${2:-}" == "--version" ]]; then
  echo "cargo-nextest 0.9.140"
  exit 0
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "list" ]]; then
  if [[ " $* " != *" --list-type binaries-only "* || " $* " != *" --message-format json "* ]]; then
    echo "inventory must request nextest's binaries-only JSON format: $*" >&2
    exit 98
  fi
  # One workspace-wide enumeration replaces the former default-members plus
  # per-package bench/Iceberg union: the catalog classifies every live binary
  # in the workspace, so the enumeration must span the same package set.
  if [[ " $* " != *" --workspace "* ]]; then
    echo "inventory must enumerate the whole workspace: $*" >&2
    exit 95
  fi
  if [[ -n "${FAKE_REQUIRED_FEATURE_ARGS:-}" && " $* " != *" ${FAKE_REQUIRED_FEATURE_ARGS} "* ]]; then
    echo "inventory command is missing required feature args '${FAKE_REQUIRED_FEATURE_ARGS}': $*" >&2
    exit 96
  fi
  [[ -z "${FAKE_NEXTEST_STDERR:-}" ]] || printf '%s\n' "${FAKE_NEXTEST_STDERR}" >&2
  [[ -z "${FAKE_NEXTEST_INVENTORY:-}" ]] || /usr/bin/cat "${FAKE_NEXTEST_INVENTORY}"
  exit "${FAKE_NEXTEST_STATUS:-0}"
fi
echo "unexpected fake cargo invocation: $*" >&2
exit 97
SH
chmod +x "${tmp_dir}/bin/cargo"

python3 - "${tmp_dir}/expected" <<'PY'
import sys
import tomllib
from pathlib import Path

with open(".config/policies/test-lanes.toml", "rb") as catalog_file:
    catalog = tomllib.load(catalog_file)
ids = sorted(entry["id"] for entry in catalog["binary"])
Path(sys.argv[1]).write_text("\n".join(ids) + "\n", encoding="utf-8")
PY

make_inventory() {
  local ids_file=$1
  local output=$2
  python3 - "${ids_file}" "${output}" <<'PY'
import json
import sys

ids = [line.strip() for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
binaries = {}
for binary_id in ids:
    if "::" in binary_id:
        binary_name = binary_id.split("::", 1)[1].removeprefix("bin/")
    else:
        binary_name = binary_id.replace("-", "_")
    binaries[binary_id] = {
        "binary-id": binary_id,
        "binary-name": binary_name,
        "package-id": "path+file:///fault-injection#fixture",
        "kind": "test",
        "binary-path": f"/tmp/{binary_name}",
        "build-platform": "target",
    }
with open(sys.argv[2], "w", encoding="utf-8") as output_file:
    json.dump({"rust-binaries": binaries}, output_file)
PY
}

make_inventory "${tmp_dir}/expected" "${tmp_dir}/complete.json"
head -n 1 "${tmp_dir}/expected" >"${tmp_dir}/partial"
make_inventory "${tmp_dir}/partial" "${tmp_dir}/partial.json"
catalog_count=$(wc -l <"${tmp_dir}/expected")
catalog_success="all ${catalog_count} live test binaries are cataloged exactly once"

run_shell_case() {
  local name=$1
  local expected_status=$2
  local expected_diagnostic=$3
  local inventory=$4
  local command_status=${5:-0}
  local features=${6:-}
  local stderr_text=${7:-}
  local output="${tmp_dir}/${name}.output"
  local actual_status=0

  PATH="${tmp_dir}/bin:${PATH}" \
    FAKE_NEXTEST_INVENTORY="${inventory}" \
    FAKE_NEXTEST_STATUS="${command_status}" \
    FAKE_NEXTEST_STDERR="${stderr_text}" \
    FEATURES="${features}" \
    /usr/bin/bash scripts/ci/check_gpu_test_lane.sh >"${output}" 2>&1 || actual_status=$?

  [[ "${actual_status}" -eq "${expected_status}" ]] || {
    echo "FAIL: ${name}: expected exit ${expected_status}, got ${actual_status}" >&2
    sed 's/^/  /' "${output}" >&2
    exit 1
  }
  grep -Fq -- "${expected_diagnostic}" "${output}" || {
    echo "FAIL: ${name}: missing diagnostic: ${expected_diagnostic}" >&2
    sed 's/^/  /' "${output}" >&2
    exit 1
  }
  echo "PASS: ${name}"
}

run_policy_case() {
  local name=$1
  local expected_status=$2
  local expected_diagnostic=$3
  shift 3
  local output="${tmp_dir}/${name}.output"
  local actual_status=0
  python3 scripts/ci/test_lane_policy.py "$@" >"${output}" 2>&1 || actual_status=$?
  [[ "${actual_status}" -eq "${expected_status}" ]] || {
    echo "FAIL: ${name}: expected exit ${expected_status}, got ${actual_status}" >&2
    sed 's/^/  /' "${output}" >&2
    exit 1
  }
  grep -Fq -- "${expected_diagnostic}" "${output}" || {
    echo "FAIL: ${name}: missing diagnostic: ${expected_diagnostic}" >&2
    sed 's/^/  /' "${output}" >&2
    exit 1
  }
  echo "PASS: ${name}"
}

run_shell_case complete 0 "${catalog_success}" \
  "${tmp_dir}/complete.json" 0 "" "visible-successful-inventory-progress"
grep -Fq "visible-successful-inventory-progress" "${tmp_dir}/complete.output"

cli_feature_output="${tmp_dir}/cli-feature-args.output"
PATH="${tmp_dir}/bin:${PATH}" \
  FAKE_NEXTEST_INVENTORY="${tmp_dir}/complete.json" \
  FAKE_REQUIRED_FEATURE_ARGS='--no-default-features --features iceberg' \
  /usr/bin/bash scripts/ci/check_gpu_test_lane.sh --no-default-features --features iceberg \
  >"${cli_feature_output}" 2>&1
grep -Fq "${catalog_success}" "${cli_feature_output}"
echo "PASS: explicit feature arguments reach live inventory"

# Producer status wins over empty, partial, or apparently complete stdout and
# preserves the producer's native diagnostic.
run_shell_case nonzero_empty 2 "nextest inventory command failed with status 42" "" 42 "" "native-link-error-empty"
grep -Fq "native-link-error-empty" "${tmp_dir}/nonzero_empty.output"
run_shell_case nonzero_partial 2 "nextest inventory command failed with status 42" "${tmp_dir}/partial.json" 42 "" "native-link-error-partial"
grep -Fq "native-link-error-partial" "${tmp_dir}/nonzero_partial.output"
run_shell_case nonzero_apparently_complete 2 "nextest inventory command failed with status 42" "${tmp_dir}/complete.json" 42 "" "native-link-error-complete"
grep -Fq "native-link-error-complete" "${tmp_dir}/nonzero_apparently_complete.output"

printf '{not json\n' >"${tmp_dir}/malformed.json"
run_shell_case malformed_json 2 "malformed nextest JSON inventory" "${tmp_dir}/malformed.json"
printf '{"rust-binaries":{}}\n' >"${tmp_dir}/empty.json"
run_shell_case empty_inventory 2 "contained no test binaries" "${tmp_dir}/empty.json"

cp "${tmp_dir}/expected" "${tmp_dir}/duplicate-name"
printf '%s\n' 'fault-injection-package::native_exec_tests' >>"${tmp_dir}/duplicate-name"
make_inventory "${tmp_dir}/duplicate-name" "${tmp_dir}/duplicate-name.json"
run_shell_case duplicate_binary_name 1 "test-binary name collision" "${tmp_dir}/duplicate-name.json"

cp "${tmp_dir}/expected" "${tmp_dir}/unclassified"
printf '%s\n' 'datafusion-nexus::fault_injection_unclassified' >>"${tmp_dir}/unclassified"
make_inventory "${tmp_dir}/unclassified" "${tmp_dir}/unclassified.json"
run_shell_case unclassified_binary 1 "unclassified test binary" "${tmp_dir}/unclassified.json"

grep -vFx 'datafusion-nexus::cugraph_tests' "${tmp_dir}/expected" >"${tmp_dir}/feature-gated"
make_inventory "${tmp_dir}/feature-gated" "${tmp_dir}/feature-gated.json"
run_shell_case stale_all_features 1 "stale all-feature catalog entry" "${tmp_dir}/feature-gated.json"
run_shell_case stale_explicit_features 0 "absent from explicit narrower feature inventory" "${tmp_dir}/feature-gated.json" 0 iceberg

# Cross-inventory ID conflict with different binary-names fails closed.
python3 - "${tmp_dir}/bench-conflict.json" <<'PY'
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as output_file:
    json.dump(
        {
            "rust-binaries": {
                "nexus-bench": {
                    "binary-id": "nexus-bench",
                    "binary-name": "renamed_bench",
                    "package-id": "path+file:///fault-injection#fixture",
                    "kind": "test",
                    "binary-path": "/tmp/renamed_bench",
                    "build-platform": "target",
                }
            }
        },
        output_file,
    )
PY
run_policy_case conflicting_union_names 1 "conflicting binary-name for nexus-bench" \
  validate-inventory \
  --inventory "${tmp_dir}/complete.json" \
  --inventory "${tmp_dir}/bench-conflict.json" \
  --feature-mode all

# Strict catalog schema: duplicate IDs violate exact-once admission, reasons
# cannot be empty, and GPU resource classes are mandatory and enumerated.
cp .config/policies/test-lanes.toml "${tmp_dir}/duplicate-catalog.toml"
cat >>"${tmp_dir}/duplicate-catalog.toml" <<'TOML'

[[binary]]
id = "datafusion-nexus"
lane = "cpu"
reason = "duplicate fixture"
TOML
run_policy_case duplicate_catalog 1 "duplicate catalog binary id" \
  --catalog "${tmp_dir}/duplicate-catalog.toml" check-catalog

python3 - "${tmp_dir}/empty-reason.toml" "${tmp_dir}/bad-resource.toml" <<'PY'
import sys
from pathlib import Path

source = Path(".config/policies/test-lanes.toml").read_text()
Path(sys.argv[1]).write_text(source.replace(
    'reason = "Adapter library tests include coverage SQL and native execution context cases that create cuDF runtimes on bounded fixtures."',
    'reason = ""',
    1,
))
Path(sys.argv[2]).write_text(source.replace('resource_class = "shared"', 'resource_class = "whole-device"', 1))
PY
run_policy_case empty_reason 1 "reason must be a non-empty string" \
  --catalog "${tmp_dir}/empty-reason.toml" check-catalog
run_policy_case bad_resource_class 1 "requires resource_class=shared or exclusive" \
  --catalog "${tmp_dir}/bad-resource.toml" check-catalog

python3 - "${tmp_dir}/bad-id.toml" <<'PY'
import sys
from pathlib import Path

source = Path(".config/policies/test-lanes.toml").read_text()
Path(sys.argv[1]).write_text(source.replace('id = "datafusion-nexus"', 'id = "datafusion nexus"', 1))
PY
run_policy_case incomplete_binary_id 1 "not a complete nextest binary id" \
  --catalog "${tmp_dir}/bad-id.toml" check-catalog

# Generated overrides are deterministic: --check never writes, drift fails,
# and only the explicit --write command restores the canonical bytes.
cp .config/nextest.toml "${tmp_dir}/nextest-drift.toml"
python3 - "${tmp_dir}/nextest-drift.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(path.read_text().replace('binary(cugraph_tests)', 'binary(drifted_cugraph_tests)', 1))
PY
run_policy_case nextest_override_drift 1 "generated nextest overrides drifted" \
  --nextest "${tmp_dir}/nextest-drift.toml" generate-nextest --check
run_policy_case nextest_regenerate 0 "WROTE: regenerated" \
  --nextest "${tmp_dir}/nextest-drift.toml" generate-nextest --write
cmp .config/nextest.toml "${tmp_dir}/nextest-drift.toml"
run_policy_case nextest_regenerated_check 0 "generated nextest overrides match" \
  --nextest "${tmp_dir}/nextest-drift.toml" generate-nextest --check

echo "OK: catalog-driven GPU test-lane fault-injection tests passed."
