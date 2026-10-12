#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
archives=""
target_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archives | --target-dir)
      if [[ $# -lt 2 || -z $2 ]]; then
        printf 'error: %s requires a directory\n' "$1" >&2
        exit 2
      fi
      if [[ $1 == --archives ]]; then archives=$2; else target_dir=$2; fi
      shift 2
      ;;
    -h | --help)
      printf 'Usage: bash scripts/ci/check_docsrs_native_free.sh [--archives DIR] [--target-dir DIR]\n'
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z ${archives} ]]; then
  archives=$(mktemp -d /tmp/algeon-docsrs.XXXXXX)
  python3 "${repo_root}/scripts/ci/release_archives.py" --output "${archives}"
fi

python3 - "${repo_root}/scripts/ci" "${archives}" "${target_dir}" <<'PY'
import json
from pathlib import Path
import sys
import tomllib

sys.path.insert(0, sys.argv[1])
from release_archives import clean_environment, outside_checkout, run_logged, validate_archives

output = Path(sys.argv[2]).resolve()
try:
    packages = validate_archives(output)
    env = clean_environment(output)
    target = outside_checkout(Path(sys.argv[3])) if sys.argv[3] else output / "docs-target"
    env.update(DOCS_RS="1", CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RUSTDOCFLAGS="-D warnings")
    unavailable = output / "unavailable-native"
    if unavailable.exists():
        raise ValueError(f"native-free sentinel must not exist: {unavailable}")
    selectors = (
        "ALGEON_NATIVE_DIR", "CUDF_INSTALL_DIR", "CUDF_CPP_BUILD_DIR", "CUGRAPH_INSTALL_DIR", "CUGRAPH_CMAKE_BUILD_DIR",
        "CUGRAPH_NATIVE_INCLUDE_DIRS", "CUGRAPH_NATIVE_LINK_DIRS", "CUGRAPH_RUNTIME_LIB_DIRS", "CUVS_INSTALL_DIR",
        "CUVS_ROOT", "CUVS_SOURCE_DIR", "CUML_INSTALL_DIR", "CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT",
        "CUDACXX", "CUDAHOSTCXX", "DLPACK_INCLUDE_DIR", "CMAKE_PREFIX_PATH", "CONDA_PREFIX", "VIRTUAL_ENV",
    )
    env.update({selector: str(unavailable / selector.lower()) for selector in selectors})
    logs = output / "logs/docsrs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"docs.rs archive checks: {output}", flush=True)
    commands = []
    for package in packages:
        path = Path(package["path"])
        with (path / "Cargo.toml").open("rb") as stream:
            docs = tomllib.load(stream)["package"]["metadata"]["docs"]["rs"]
        args = ["--manifest-path", str(path / "Cargo.toml"), "--config", str(output / "patches.toml"), "--target", docs["default-target"]]
        run_logged(["cargo", "fetch", *args], logs / f"{package['name']}-fetch.log", cwd=output, env=env)
        command = ["cargo", "doc", "--no-deps", "--locked", "--offline", "--message-format=json", *args]
        if docs.get("no-default-features", False):
            command.append("--no-default-features")
        if package["docs_features"]:
            command.extend(["--features", ",".join(package["docs_features"])])
        commands.append((package["name"], command))
    # Resolve/download every package before any build script executes.
    env["CARGO_NET_OFFLINE"] = "true"
    evidence = {"native_selectors": {selector: env[selector] for selector in selectors},
                "cargo_network": "offline", "network_sandbox": False, "sys_build_scripts": {}}
    sys_packages = {f"path+{Path(package['path']).as_uri()}#{package['name']}@{package['version']}": package["name"]
                    for package in packages if package["name"].endswith("-sys")}
    (logs / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    for name, command in commands:
        log = logs / f"{name}-doc.log"
        run_logged(command, log, cwd=output, env=env)
        for line in log.read_text().splitlines():
            if not line.startswith("{"):
                continue
            message = json.loads(line)
            if message.get("reason") != "build-script-executed" or message.get("package_id") not in sys_packages:
                continue
            package = sys_packages[message["package_id"]]
            build_output = Path(message["out_dir"]).parent / "output"
            directives = [line for line in build_output.read_text().splitlines()
                          if line.startswith(("cargo:rustc-link-", "cargo::rustc-link-"))]
            if message["linked_libs"] or message["linked_paths"] or directives:
                raise ValueError(f"{package}: DOCS_RS emitted native linker directives: {build_output}")
            evidence["sys_build_scripts"].setdefault(package, []).append(str(build_output))
    if set(evidence["sys_build_scripts"]) != set(sys_packages.values()):
        raise ValueError("rustdoc did not report build-script outputs for every packaged sys crate")
    (logs / "native-free-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: docs.rs native-free rustdoc for {len(packages)} packaged release crates; logs: {logs}")
except (ValueError, RuntimeError, OSError) as error:
    raise SystemExit(f"FAIL: docs.rs archive check: {error}")
PY
