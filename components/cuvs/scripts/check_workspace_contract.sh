#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
metadata_file="$(mktemp)"
trap 'rm -f "${metadata_file}"' EXIT
cargo metadata --manifest-path "${repo_root}/Cargo.toml" --format-version 1 --no-deps >"${metadata_file}"

python3 - "${metadata_file}" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
packages = {package["name"]: package for package in metadata["packages"]}
assert set(packages) == {"cuvs-nexus", "cuvs-nexus-sys", "cuvs-nexus-bindings-generator"}, set(packages)

safe = packages["cuvs-nexus"]
sys_package = packages["cuvs-nexus-sys"]
generator = packages["cuvs-nexus-bindings-generator"]
safe_libs = [target["name"] for target in safe["targets"] if "lib" in target["kind"]]
sys_libs = [target["name"] for target in sys_package["targets"] if "lib" in target["kind"]]
assert safe_libs == ["cuvs"], safe_libs
assert sys_libs == ["cuvs_sys"], sys_libs

safe_dependencies = {dependency["name"] for dependency in safe["dependencies"]}
assert {"cudf-nexus", "cuvs-nexus-sys", "rapids-interop"} <= safe_dependencies, safe_dependencies
sys_dependencies = {dependency["name"] for dependency in sys_package["dependencies"]}
assert "cudf-nexus" not in sys_dependencies, sys_dependencies
assert "cudf-nexus-sys" not in sys_dependencies, sys_dependencies
assert "rapids-interop" not in sys_dependencies, sys_dependencies
assert sys_package["links"] == "cuvs_c", sys_package["links"]
assert sys_package["features"] == {}, sys_package["features"]
assert generator["publish"] == [], generator["publish"]
assert {target["name"] for target in generator["targets"]} == {"cuvs-nexus-bindings-generator"}
PY

echo "workspace package/dependency contract OK"
