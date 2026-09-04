#!/usr/bin/env bash
# Fault-injection tests for the test-lane policy, suffix classification, and nextest config.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-gpu-lane-tests.XXXXXX")
trap "rm -rf \"${tmp_dir}\"" EXIT
mkdir -p "${tmp_dir}/bin"

cat >"${tmp_dir}/bin/cargo" <<\SH
#!/usr/bin/bash
set -u
if [[ "${1:-}" == "nextest" && "${2:-}" == "--version" ]]; then
  echo "cargo-nextest 0.9.140"
  exit 0
fi
if [[ "${1:-}" == "nextest" && "${2:-}" == "list" ]]; then
  if [[ " $* " != *" --list-type binaries-only "* || " $* " != *" --message-format json "* ]]; then
    echo "inventory must request nextest binaries-only JSON format: $*" >&2
    exit 98
  fi
  if [[ " $* " != *" --workspace "* ]]; then
    echo "inventory must enumerate the whole workspace: $*" >&2
    exit 95
  fi
  [[ -z "${FAKE_NEXTEST_STDERR:-}" ]] || printf "%s\n" "${FAKE_NEXTEST_STDERR}" >&2
  [[ -z "${FAKE_NEXTEST_INVENTORY:-}" ]] || /usr/bin/cat "${FAKE_NEXTEST_INVENTORY}"
  exit "${FAKE_NEXTEST_STATUS:-0}"
fi
echo "unexpected fake cargo invocation: $*" >&2
exit 97
SH
chmod +x "${tmp_dir}/bin/cargo"

# Capture real live inventory as base fixture
cargo nextest list --workspace --all-features --list-type binaries-only --message-format json >"${tmp_dir}/complete.json"
catalog_success="all 118 live test binaries classified"

run_shell_case() {
  local name=$1
  local expected_status=$2
  local expected_diagnostic=$3
  local inventory=$4
  local command_status=${5:-0}
  local stderr_text=${6:-}
  local output="${tmp_dir}/${name}.output"
  local actual_status=0

  PATH="${tmp_dir}/bin:${PATH}" \
    FAKE_NEXTEST_INVENTORY="${inventory}" \
    FAKE_NEXTEST_STATUS="${command_status}" \
    FAKE_NEXTEST_STDERR="${stderr_text}" \
    /usr/bin/bash scripts/ci/check_gpu_test_lane.sh --all-features >"${output}" 2>&1 || actual_status=$?

  [[ "${actual_status}" -eq "${expected_status}" ]] || {
    echo "FAIL: ${name}: expected exit ${expected_status}, got ${actual_status}" >&2
    sed "s/^/  /" "${output}" >&2
    exit 1
  }
  grep -Fq -- "${expected_diagnostic}" "${output}" || {
    echo "FAIL: ${name}: missing diagnostic: ${expected_diagnostic}" >&2
    sed "s/^/  /" "${output}" >&2
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
    sed "s/^/  /" "${output}" >&2
    exit 1
  }
  grep -Fq -- "${expected_diagnostic}" "${output}" || {
    echo "FAIL: ${name}: missing diagnostic: ${expected_diagnostic}" >&2
    sed "s/^/  /" "${output}" >&2
    exit 1
  }
  echo "PASS: ${name}"
}

run_shell_case complete 0 "${catalog_success}" "${tmp_dir}/complete.json" 0 "visible-inventory-progress"
grep -Fq "visible-inventory-progress" "${tmp_dir}/complete.output"

# Unit test for suffix inference
python3 - <<\PY
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("test_lane_policy", Path("scripts/ci/test_lane_policy.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert module.infer_lane_from_name("sample_test_cpu") == ("cpu", None)
assert module.infer_lane_from_name("sample_test_gpu") == ("gpu", "shared")
assert module.infer_lane_from_name("sample_test_gpu_exclusive") == ("gpu", "exclusive")
assert module.infer_lane_from_name("sample_test") is None
PY
echo "PASS: suffix inference rules"

# Unsupported flags fail closed
unsupported_flag_output="${tmp_dir}/unsupported-flag.output"
actual_flag_status=0
/usr/bin/bash scripts/ci/check_gpu_test_lane.sh --features foo >"${unsupported_flag_output}" 2>&1 || actual_flag_status=$?
[[ "${actual_flag_status}" -eq 2 ]]
grep -Fq "unsupported argument for lane inventory: --features" "${unsupported_flag_output}"
echo "PASS: unsupported feature flags fail closed"

# Producer errors win
run_shell_case nonzero_empty 2 "nextest inventory command failed with status 42" "" 42 "native-link-error-empty"
grep -Fq "native-link-error-empty" "${tmp_dir}/nonzero_empty.output"

printf "{not json\n" >"${tmp_dir}/malformed.json"
run_shell_case malformed_json 2 "malformed nextest JSON inventory" "${tmp_dir}/malformed.json"
printf "{\"rust-binaries\":{}}\n" >"${tmp_dir}/empty.json"
run_shell_case empty_inventory 2 "contained no test binaries" "${tmp_dir}/empty.json"

# Duplicate binary-name collision
python3 - "${tmp_dir}/duplicate-name.json" "${tmp_dir}/complete.json" <<\PY
import json
import sys

with open(sys.argv[2], encoding="utf-8") as source:
    inventory = json.load(source)
inventory["rust-binaries"].update(
    {
        "fault-injection-package::first_tests_cpu": {
            "binary-id": "fault-injection-package::first_tests_cpu",
            "binary-name": "same_target_name_cpu",
            "kind": "test",
            "package-id": "path+file:///fault-injection#fixture",
            "binary-path": "/tmp/same_target_name-first",
            "build-platform": "target",
        },
        "fault-injection-package::second_tests_cpu": {
            "binary-id": "fault-injection-package::second_tests_cpu",
            "binary-name": "same_target_name_cpu",
            "kind": "test",
            "package-id": "path+file:///fault-injection#fixture",
            "binary-path": "/tmp/same_target_name-second",
            "build-platform": "target",
        },
    }
)
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(inventory, output)
PY
run_shell_case duplicate_binary_name 1 "test-binary target collision" "${tmp_dir}/duplicate-name.json"

# Unclassified binary fails closed
python3 - "${tmp_dir}/unclassified.json" "${tmp_dir}/complete.json" <<\PY
import json
import sys

with open(sys.argv[2], encoding="utf-8") as source:
    inventory = json.load(source)
inventory["rust-binaries"]["datafusion-nexus::unclassified_test"] = {
    "binary-id": "datafusion-nexus::unclassified_test",
    "binary-name": "unclassified_test",
    "kind": "test",
    "package-id": "path+file:///fault-injection#fixture",
    "binary-path": "/tmp/unclassified_test",
    "build-platform": "target",
}
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(inventory, output)
PY
run_shell_case unclassified_binary 1 "unclassified test binary: datafusion-nexus::unclassified_test" "${tmp_dir}/unclassified.json"

# Stale exception fails closed
python3 - "${tmp_dir}/stale-exception.json" "${tmp_dir}/complete.json" <<\PY
import json
import sys

with open(sys.argv[2], encoding="utf-8") as source:
    inventory = json.load(source)
# remove an exception binary
del inventory["rust-binaries"]["datafusion-nexus::cpu_contract_tests"]
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(inventory, output)
PY
run_shell_case stale_exception 1 "stale exception entry: datafusion-nexus::cpu_contract_tests" "${tmp_dir}/stale-exception.json"

# Suffix conflict with exception fails closed
python3 - "${tmp_dir}/conflicting.json" "${tmp_dir}/complete.json" <<\PY
import json
import sys

with open(sys.argv[2], encoding="utf-8") as source:
    inventory = json.load(source)
# Change cpu_contract_tests to have suffix _gpu
inventory["rust-binaries"]["datafusion-nexus::cpu_contract_tests"]["binary-name"] = "cpu_contract_tests_gpu"
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(inventory, output)
PY
run_shell_case conflicting_lane 1 "conflicting lane for datafusion-nexus::cpu_contract_tests" "${tmp_dir}/conflicting.json"

# Strict exception policy schema tests
cp .config/policies/test-lanes.toml "${tmp_dir}/duplicate-catalog.toml"
cat >>"${tmp_dir}/duplicate-catalog.toml" <<\TOML

[[exception]]
id = "datafusion-nexus"
lane = "cpu"
TOML
run_policy_case duplicate_catalog 1 "duplicate catalog binary id"   --catalog "${tmp_dir}/duplicate-catalog.toml" check-catalog

python3 - "${tmp_dir}/bad-resource.toml" <<\PY
import sys
from pathlib import Path

source = Path(".config/policies/test-lanes.toml").read_text()
Path(sys.argv[1]).write_text(source.replace("resource_class = \"shared\"", "resource_class = \"whole-device\"", 1))
PY
run_policy_case bad_resource_class 1 "requires resource_class=shared or exclusive"   --catalog "${tmp_dir}/bad-resource.toml" check-catalog

python3 - "${tmp_dir}/bad-id.toml" <<\PY
import sys
from pathlib import Path

source = Path(".config/policies/test-lanes.toml").read_text()
Path(sys.argv[1]).write_text(source.replace("id = \"datafusion-nexus\"", "id = \"datafusion nexus\"", 1))
PY
run_policy_case incomplete_binary_id 1 "not a complete nextest binary id"   --catalog "${tmp_dir}/bad-id.toml" check-catalog

# Nextest config validation tests
run_policy_case valid_nextest 0 "OK: nextest configuration matches" check-nextest

python3 - "${tmp_dir}/bad-threads.toml" <<\PY
import sys
from pathlib import Path

source = Path(".config/nextest.toml").read_text()
Path(sys.argv[1]).write_text(source.replace("max-threads = 1", "max-threads = 2", 1))
PY
run_policy_case bad_nextest_threads 1 "test-groups.gpu-exclusive.max-threads must be 1"   --nextest "${tmp_dir}/bad-threads.toml" check-nextest

python3 - "${tmp_dir}/missing-regex.toml" <<\PY
import sys
from pathlib import Path

source = Path(".config/nextest.toml").read_text()
Path(sys.argv[1]).write_text(source.replace("binary(/_gpu_exclusive$/)", "binary(/other$/)", 1))
PY
run_policy_case missing_nextest_regex 1 "nextest gpu-exclusive filter must include binary(/_gpu_exclusive$/)"   --nextest "${tmp_dir}/missing-regex.toml" check-nextest

echo "OK: catalog-driven GPU test-lane fault-injection tests passed."
