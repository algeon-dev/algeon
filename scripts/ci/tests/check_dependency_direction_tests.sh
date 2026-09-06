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
    "planning-runtime": [("nexus-query-planning", "src/lib.rs", "use nexus_query_runtime::Marker;\n")],
    "planning-runtime-module": [("nexus-query-planning", "src/lib.rs", "use crate::{observability::Marker};\n")],
    "planning-engine": [("nexus-query-planning", "src/lib.rs", "use nexus_query_engine::exec::Marker;\n")],
    "runtime-engine": [("nexus-query-runtime", "src/lib.rs", "use nexus_query_engine::exec::Marker;\n")],
    "component-engine": [("cudf-nexus", "src/lib.rs", "use nexus_query_engine::Marker;\n")],
    "component-engine-root": [("cudf-nexus", "src/lib.rs", "use nexus_query_engine;\n")],
    "planning-datafusion": [("nexus-query-planning", "src/lib.rs", "use datafusion::physical_plan::ExecutionPlan;\n")],
    "runtime-datafusion": [("nexus-query-runtime", "src/lib.rs", "extern crate datafusion_common;\n")],
    "observability-runtime-use": [
        ("nexus-query-runtime", "src/observability/mod.rs", "use crate::runtime::ExecutionContext;\n"),
    ],
    "observability-runtime-qualified": [
        (
            "nexus-query-runtime",
            "src/observability/stage.rs",
            "pub fn stage(_context: &crate::runtime::ExecutionContext) {}\n",
        ),
    ],
    "observability-runtime-nested-use": [
        (
            "nexus-query-runtime",
            "src/observability/stage.rs",
            "use crate::{\n    observability::Marker,\n    runtime::{\n        ExecutionContext,\n    },\n};\n",
        ),
    ],
    "observability-runtime-super": [
        ("nexus-query-runtime", "src/observability/mod.rs", "use super::runtime::ExecutionContext;\n"),
    ],
    "observability-runtime-nested-super": [
        (
            "nexus-query-runtime",
            "src/observability/report/row.rs",
            "pub type Context = super::super::super::runtime::ExecutionContext;\n",
        ),
    ],
    "observability-file-runtime": [
        (
            "nexus-query-runtime",
            "src/observability.rs",
            "pub(crate) use crate::runtime::ExecutionContext as Context;\n",
        ),
    ],
    "runtime-observability-allowed": [
        ("nexus-query-runtime", "src/observability/mod.rs", "pub struct RuntimeObservationHandle;\n"),
        ("nexus-query-runtime", "src/runtime/context.rs", "use crate::observability::RuntimeObservationHandle;\n"),
    ],
    "observability-runtime-non-code": [
        (
            "nexus-query-runtime",
            "src/observability/mod.rs",
            "// use crate::runtime::ExecutionContext;\n"
            "/* crate::runtime::ExecutionContext */\n"
            'pub const PATH: &str = "crate::runtime::ExecutionContext";\n',
        ),
    ],
    "observability-inline-module": [
        ("nexus-query-runtime", "src/observability/mod.rs", "mod runtime;\nmod stage;\n"),
        ("nexus-query-runtime", "src/observability/runtime.rs", "pub struct Row;\n"),
        (
            "nexus-query-runtime",
            "src/observability/stage.rs",
            "#[cfg(test)]\nmod tests {\n    use super::super::runtime::Row;\n}\n",
        ),
    ],
    "observability-path-module": [
        ("nexus-query-runtime", "src/observability/mod.rs", "mod recorder;\nmod runtime;\n"),
        ("nexus-query-runtime", "src/observability/runtime.rs", "pub struct Row;\n"),
        (
            "nexus-query-runtime",
            "src/observability/recorder.rs",
            '#[cfg(test)]\n#[path = "recorder_tests.rs"]\nmod tests;\n',
        ),
        ("nexus-query-runtime", "src/observability/recorder_tests.rs", "use super::super::runtime::Row;\n"),
    ],
}
for package_name, relative_path, contents in source_targets.get(source_case, []):
    target = dict(packages)[package_name] / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents, encoding="utf-8")

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

run_case source-observability-runtime-use 1 "observability -> runtime" observability-runtime-use "" ""
run_case source-observability-runtime-qualified 1 "observability -> runtime" observability-runtime-qualified "" ""
run_case source-observability-runtime-nested-use 1 "stage.rs:3: runtime::{" observability-runtime-nested-use "" ""
run_case source-observability-runtime-super 1 "observability -> runtime" observability-runtime-super "" ""
run_case source-observability-runtime-nested-super 1 "observability -> runtime" observability-runtime-nested-super "" ""
run_case source-observability-file-runtime 1 "observability -> runtime" observability-file-runtime "" ""
run_case source-runtime-observability-allowed 0 "dependency direction contracts OK" runtime-observability-allowed "" ""
run_case source-observability-runtime-non-code 0 "dependency direction contracts OK" observability-runtime-non-code "" ""
run_case source-observability-inline-module 0 "dependency direction contracts OK" observability-inline-module "" ""
run_case source-observability-path-module 0 "dependency direction contracts OK" observability-path-module "" ""

printf 'dependency direction fault-injection tests OK\n'
