#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly check_script="${repo_root}/scripts/ci/check_dependency_direction.py"
readonly test_root="$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-dependency-direction-tests.XXXXXX")"
trap 'rm -rf "${test_root}"' EXIT

mkdir -p "${test_root}/bin"
cat >"${test_root}/bin/cargo" <<'SH'
#!/usr/bin/env bash
set -u
expected=(metadata --locked --all-features --format-version 1)
if [[ "$*" != "${expected[*]}" ]]; then
  echo "unexpected cargo metadata arguments: $*" >&2
  exit 97
fi
if [[ -z "${FAKE_METADATA:-}" || ! -f "${FAKE_METADATA}" ]]; then
  echo "FAKE_METADATA must point to a JSON file" >&2
  exit 98
fi
/usr/bin/cat "${FAKE_METADATA}"
SH
chmod +x "${test_root}/bin/cargo"

make_case() {
  local name=$1
  local graph_source=$2
  local edge_from=$3
  local edge_to=$4
  local layer_packages=${5:-present}
  local case_root="${test_root}/${name}"
  local metadata_path="${case_root}/metadata.json"
  mkdir -p "${case_root}"
  python3 - "${case_root}" "${metadata_path}" "${graph_source}" "${edge_from}" "${edge_to}" "${layer_packages}" <<'PY'
import json
import sys
from pathlib import Path

case_root = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])
source_case = sys.argv[3]
edge_from = sys.argv[4]
edge_to = sys.argv[5]
layer_packages = sys.argv[6]

packages = []
if layer_packages == "present":
    packages.extend([
        ("nexus-query-planning", case_root / "crates/nexus-query-planning"),
        ("nexus-query-runtime", case_root / "crates/nexus-query-runtime"),
    ])
packages.extend([
    ("nexus-query-engine", case_root / "crates/nexus-query-engine"),
    ("cudf-nexus", case_root / "components/cudf/cudf-nexus"),
    ("cudf-nexus-sys", case_root / "components/cudf/cudf-nexus-sys"),
    ("cugraph-nexus", case_root / "components/cugraph/cugraph-nexus"),
    ("cugraph-nexus-sys", case_root / "components/cugraph/cugraph-nexus-sys"),
    ("cuvs-nexus", case_root / "components/cuvs/cuvs-nexus"),
    ("cuvs-nexus-sys", case_root / "components/cuvs/cuvs-nexus-sys"),
])
ids = {name: f"path+file:///{name}#0.1.0" for name, _ in packages}
for name, package_root in packages:
    (package_root / "Cargo.toml").parent.mkdir(parents=True, exist_ok=True)
    (package_root / "Cargo.toml").write_text("[package]\nname = %r\n" % name, encoding="utf-8")
    source_root = package_root / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "lib.rs").write_text("pub fn clean() {}\n", encoding="utf-8")

source_targets = {
    "planning-runtime": ("nexus-query-planning", "use nexus_query_runtime::Marker;\n"),
    "planning-runtime-module": ("nexus-query-planning", "use crate::{observability::Marker};\n"),
    "planning-engine": ("nexus-query-planning", "use nexus_query_engine::exec::Marker;\n"),
    "runtime-engine": ("nexus-query-runtime", "use nexus_query_engine::exec::Marker;\n"),
    "component-engine": ("cudf-nexus", "use nexus_query_engine::Marker;\n"),
    "component-engine-root": ("cudf-nexus", "use nexus_query_engine;\n"),
    "planning-datafusion": ("nexus-query-planning", "use datafusion::physical_plan::ExecutionPlan;\n"),
    "runtime-datafusion": ("nexus-query-runtime", "extern crate datafusion_common;\n"),
}
if source_case in source_targets:
    package_name, contents = source_targets[source_case]
    package_root = dict(packages)[package_name]
    (package_root / "src/lib.rs").write_text(contents, encoding="utf-8")

dependencies = {name: [] for name, _ in packages}
if edge_from and edge_to:
    dependencies[edge_from].append(ids[edge_to])
nodes = [{"id": ids[name], "deps": [{"pkg": target} for target in targets]} for name, targets in dependencies.items()]
metadata_path.write_text(
    json.dumps(
        {
            "packages": [
                {
                    "id": ids[name],
                    "name": name,
                    "manifest_path": str(package_root / "Cargo.toml"),
                    "dependencies": (
                        [{"name": "datafusion-expr"}]
                        if source_case == "planning-datafusion-cargo" and name == "nexus-query-planning"
                        else [{"name": "datafusion-common"}]
                        if source_case == "runtime-datafusion-cargo" and name == "nexus-query-runtime"
                        else []
                    ),
                }
                for name, package_root in packages
            ],
            "workspace_members": [ids[name] for name, _ in packages],
            "resolve": {"nodes": nodes},
        }
    ),
    encoding="utf-8",
)
PY
  printf '%s\n' "${case_root}" "${metadata_path}"
}

run_case() {
  local name=$1
  local expected_status=$2
  local expected_text=$3
  local source_case=$4
  local edge_from=$5
  local edge_to=$6
  local layer_packages=${7:-present}
  local case_root metadata_path output status=0
  mapfile -t case_paths < <(make_case "${name}" "${source_case}" "${edge_from}" "${edge_to}" "${layer_packages}")
  case_root=${case_paths[0]}
  metadata_path=${case_paths[1]}
  output="${test_root}/${name}.output"
  PATH="${test_root}/bin:${PATH}" FAKE_METADATA="${metadata_path}" \
    python3 "${check_script}" --root "${case_root}" >"${output}" 2>&1 || status=$?
  if [[ "${status}" -ne "${expected_status}" ]]; then
    echo "FAIL: ${name}: expected exit ${expected_status}, got ${status}" >&2
    sed 's/^/  /' "${output}" >&2
    exit 1
  fi
  if ! grep -Fq -- "${expected_text}" "${output}"; then
    echo "FAIL: ${name}: missing diagnostic: ${expected_text}" >&2
    sed 's/^/  /' "${output}" >&2
    exit 1
  fi
  echo "PASS: ${name}"
}

run_case clean 0 "dependency direction contracts OK" clean "" ""
run_case missing-layers 1 "required workspace packages are missing" clean "" "" missing

run_case graph-planning-runtime 1 "planning -> runtime" clean nexus-query-planning nexus-query-runtime
run_case graph-planning-engine 1 "planning -> engine-exec" clean nexus-query-planning nexus-query-engine
run_case graph-runtime-engine 1 "runtime -> engine-exec" clean nexus-query-runtime nexus-query-engine
for component in cudf-nexus cudf-nexus-sys cugraph-nexus cugraph-nexus-sys cuvs-nexus cuvs-nexus-sys; do
  run_case "graph-${component}-engine" 1 "component -> engine" clean "${component}" nexus-query-engine
done

run_case source-planning-runtime 1 "planning -> runtime" planning-runtime "" ""
run_case source-planning-runtime-module 1 "planning -> runtime" planning-runtime-module "" ""
run_case source-planning-engine 1 "planning -> engine-exec" planning-engine "" ""
run_case source-runtime-engine 1 "runtime -> engine-exec" runtime-engine "" ""
run_case source-component-engine 1 "component -> engine" component-engine "" ""
run_case source-component-engine-root 1 "component -> engine" component-engine-root "" ""
run_case source-planning-datafusion 1 "planning -> DataFusion" planning-datafusion "" ""
run_case source-runtime-datafusion 1 "runtime -> DataFusion" runtime-datafusion "" ""
run_case cargo-planning-datafusion 1 "planning -> DataFusion" planning-datafusion-cargo "" ""
run_case cargo-runtime-datafusion 1 "runtime -> DataFusion" runtime-datafusion-cargo "" ""

printf 'dependency direction fault-injection tests OK\n'
