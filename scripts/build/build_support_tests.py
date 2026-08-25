#!/usr/bin/env python3
"""Run std-only RAPIDS build-support policy tests and inclusion smokes."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "scripts/build/rapids.rs"


@dataclass(frozen=True)
class InclusionSite:
    name: str
    source: str
    support_source: str
    generic: bool
    bins: bool
    tests: bool
    cugraph_feature_gated: bool
    cuvs_feature_gated: bool
    nvtx3interop: bool


SITES = (
    InclusionSite("root", "build.rs", "scripts/build/rapids.rs", True, False, True, True, True, False),
    InclusionSite(
        "engine",
        "crates/nexus-query-engine/build.rs",
        "../../scripts/build/rapids.rs",
        True,
        False,
        True,
        True,
        True,
        True,
    ),
    InclusionSite(
        "bench",
        "crates/nexus-bench/build.rs",
        "../../scripts/build/rapids.rs",
        True,
        True,
        True,
        True,
        True,
        False,
    ),
    InclusionSite(
        "examples",
        "crates/nexus-examples/build.rs",
        "../../scripts/build/rapids.rs",
        True,
        False,
        False,
        False,
        True,
        False,
    ),
    InclusionSite(
        "server",
        "crates/nexus-server/build.rs",
        "../../scripts/build/rapids.rs",
        True,
        True,
        True,
        True,
        True,
        False,
    ),
)


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)


def require_success(result: subprocess.CompletedProcess[str], context: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{context} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def write_library(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def fake_environment(
    root: Path,
    *,
    metadata_mode: str,
    cugraph: bool,
    cuvs: bool,
    logger: bool,
    missing_rmm: bool = False,
    divergent_runtime_rmm: bool = False,
    logger_linkage: str | None = None,
    graph_logger: bool = False,
    missing_cuvs_c: bool = False,
    divergent_cuvs: bool = False,
    private_graph_cuvs: bool = False,
) -> tuple[dict[str, str], Path, Path, Path, Path]:
    cudf_dir = root / "cudf runtime"
    graph_native = root / "graph native"
    graph_runtime = root / "graph runtime"
    cuvs_runtime = root / "cuvs install" / "lib"
    cuda_target_lib = root / "cuda" / "targets" / "x86_64-linux" / "lib"
    libcudf_rust = cudf_dir / "libcudf_rust.so"
    libcudf = cudf_dir / "libcudf.so"
    rmm = cudf_dir / "librmm.so"
    logger_path = cudf_dir / "librapids_logger.so"
    write_library(libcudf_rust, b"fake libcudf Rust bridge")
    write_library(libcudf, b"fake libcudf")
    write_library(rmm, b"one rmm build")
    write_library(cuda_target_lib / "libnvtx3interop.so", b"fake nvtx interop")
    if logger:
        write_library(logger_path, b"one logger build")
    canonical_cuvs = cuvs_runtime / "libcuvs.so"
    if cugraph or cuvs:
        write_library(canonical_cuvs, b"one cuvs build")
    if cuvs and not missing_cuvs_c:
        write_library(cuvs_runtime / "libcuvs_c.so", b"one cuvs C API build")
    if cugraph:
        write_library(graph_native / "librmm.so", b"one rmm build")
        write_library(
            graph_runtime / "librmm.so",
            b"divergent rmm build" if divergent_runtime_rmm else b"one rmm build",
        )
        if logger or graph_logger:
            write_library(graph_native / "librapids_logger.so", b"one logger build")
            write_library(graph_runtime / "librapids_logger.so", b"one logger build")

    graph_cuvs = canonical_cuvs
    if cugraph and divergent_cuvs:
        graph_cuvs = root / "other cuvs" / "lib" / "libcuvs.so"
        write_library(graph_cuvs, b"different cuvs build")
    if cugraph and private_graph_cuvs:
        graph_cuvs = graph_native / "_deps" / "cuvs-build" / "libcuvs.so"
        write_library(graph_cuvs, b"private cuvs build")

    cudf_values = {
        "runtime-lib-dir": str(cudf_dir),
        "cudf-rust-runtime-lib": str(libcudf_rust),
        "libcudf-runtime-lib": str(libcudf),
        "rapids-logger-linkage": logger_linkage or ("shared" if logger else "static"),
    }
    if not missing_rmm:
        cudf_values["rmm-runtime-lib"] = str(rmm)
    if logger:
        cudf_values["rapids-logger-runtime-lib"] = str(logger_path)
    graph_values = {
        "runtime-lib-dirs": os.pathsep.join([str(graph_runtime), str(graph_native)]),
        "native-link-dirs": os.pathsep.join([str(graph_native), str(graph_runtime)]),
        "cuvs-library-path": str(graph_cuvs),
        "cuvs-runtime-lib-dir": str(graph_cuvs.parent),
    }
    cuvs_values = {
        "runtime-lib-dirs": str(cuvs_runtime),
        "native-link-dirs": str(cuvs_runtime),
        "libcuvs-runtime-lib": str(canonical_cuvs),
        "libcuvs-c-runtime-lib": str(cuvs_runtime / "libcuvs_c.so"),
    }

    # The smoke executables stand in for Cargo build scripts. Give the one
    # NVTX-enabled build script the same target and hermetic toolkit contract
    # that Cargo supplies, rather than resolving a host CUDA installation.
    environment: dict[str, str] = {
        "TARGET": "x86_64-unknown-linux-gnu",
        "CUDAToolkit_ROOT": str(root / "cuda"),
    }
    if metadata_mode == "direct":
        direct_names = {
            "runtime-lib-dir": "DEP_CUDF_RUST_RUNTIME_LIB_DIR",
            "cudf-rust-runtime-lib": "DEP_CUDF_RUST_CUDF_RUST_RUNTIME_LIB",
            "libcudf-runtime-lib": "DEP_CUDF_RUST_LIBCUDF_RUNTIME_LIB",
            "rmm-runtime-lib": "DEP_CUDF_RUST_RMM_RUNTIME_LIB",
            "rapids-logger-runtime-lib": "DEP_CUDF_RUST_RAPIDS_LOGGER_RUNTIME_LIB",
            "rapids-logger-linkage": "DEP_CUDF_RUST_RAPIDS_LOGGER_LINKAGE",
        }
        environment.update({direct_names[key]: value for key, value in cudf_values.items()})
        if cugraph:
            environment["DEP_CUGRAPH_RUNTIME_LIB_DIRS"] = graph_values["runtime-lib-dirs"]
            environment["DEP_CUGRAPH_NATIVE_LINK_DIRS"] = graph_values["native-link-dirs"]
            environment["DEP_CUGRAPH_CUVS_LIBRARY_PATH"] = graph_values["cuvs-library-path"]
            environment["DEP_CUGRAPH_CUVS_RUNTIME_LIB_DIR"] = graph_values["cuvs-runtime-lib-dir"]
        if cuvs:
            environment["DEP_CUVS_C_RUNTIME_LIB_DIRS"] = cuvs_values["runtime-lib-dirs"]
            environment["DEP_CUVS_C_NATIVE_LINK_DIRS"] = cuvs_values["native-link-dirs"]
            environment["DEP_CUVS_C_LIBCUVS_RUNTIME_LIB"] = cuvs_values["libcuvs-runtime-lib"]
            environment["DEP_CUVS_C_LIBCUVS_C_RUNTIME_LIB"] = cuvs_values["libcuvs-c-runtime-lib"]
    elif metadata_mode == "aggregate":
        environment["DEP_CUDF_RUST_METADATA"] = ";".join(f"{key}={value}" for key, value in cudf_values.items())
        if cugraph:
            environment["DEP_CUGRAPH_METADATA"] = ";".join(f"{key}={value}" for key, value in graph_values.items())
        if cuvs:
            environment["DEP_CUVS_C_METADATA"] = ";".join(f"{key}={value}" for key, value in cuvs_values.items())
    elif metadata_mode == "mixed":
        environment["DEP_CUDF_RUST_METADATA"] = ";".join(f"{key}={value}" for key, value in cudf_values.items())
        environment["DEP_CUDF_RUST_RUNTIME_LIB_DIR"] = str(cudf_dir)
        environment["DEP_CUDF_RUST_CUDF_RUST_RUNTIME_LIB"] = str(libcudf_rust)
        environment["DEP_CUDF_RUST_LIBCUDF_RUNTIME_LIB"] = str(libcudf)
        if cugraph:
            environment["DEP_CUGRAPH_METADATA"] = f"native-link-dirs={graph_values['native-link-dirs']}"
            environment["DEP_CUGRAPH_RUNTIME_LIB_DIRS"] = graph_values["runtime-lib-dirs"]
            environment["DEP_CUGRAPH_CUVS_LIBRARY_PATH"] = graph_values["cuvs-library-path"]
            environment["DEP_CUGRAPH_CUVS_RUNTIME_LIB_DIR"] = graph_values["cuvs-runtime-lib-dir"]
        if cuvs:
            environment["DEP_CUVS_C_METADATA"] = ";".join(f"{key}={value}" for key, value in cuvs_values.items())
            environment["DEP_CUVS_C_LIBCUVS_RUNTIME_LIB"] = cuvs_values["libcuvs-runtime-lib"]
    else:
        raise ValueError(f"unknown metadata mode: {metadata_mode}")

    return environment, cudf_dir, graph_native, graph_runtime, cuvs_runtime


def assert_link_target_output(site: InclusionSite, output: str, runtime_dir: Path) -> None:
    suffix = f"-Wl,-rpath,{runtime_dir}"
    expectations = {
        f"cargo:rustc-link-arg={suffix}": site.generic,
        f"cargo:rustc-link-arg-bins={suffix}": site.bins,
        f"cargo:rustc-link-arg-tests={suffix}": site.tests,
    }
    for line, expected in expectations.items():
        if (line in output) != expected:
            raise AssertionError(f"{site.name}: expected {line!r} presence={expected}\n{output}")
    tracked = f"cargo:rerun-if-changed={site.support_source}"
    if tracked not in output:
        raise AssertionError(f"{site.name}: missing support-source tracking directive {tracked!r}\n{output}")


def assert_cudf_rust_runtime_tracking(output: str, runtime_dir: Path) -> None:
    library = runtime_dir / "libcudf_rust.so"
    tracked = f"cargo:rerun-if-changed={library}"
    if tracked not in output:
        raise AssertionError(f"missing cudf Rust bridge tracking directive {tracked!r}\n{output}")


def assert_nvtx3interop_output(site: InclusionSite, output: str, fixture_root: Path) -> None:
    library = fixture_root / "cuda" / "targets" / "x86_64-linux" / "lib" / "libnvtx3interop.so"
    tracked = f"cargo:rerun-if-changed={library}"
    linked = "cargo:rustc-link-lib=dylib=nvtx3interop"
    if (tracked in output) != site.nvtx3interop or (linked in output) != site.nvtx3interop:
        raise AssertionError(f"{site.name}: unexpected NVTX interop linkage state\n{output}")


def run_site_scenario(
    site: InclusionSite,
    executable: Path,
    fixture_root: Path,
    *,
    cugraph: bool,
    cuvs: bool,
    metadata_mode: str,
    logger: bool,
    logger_linkage: str | None = None,
) -> str:
    environment, cudf_dir, _, _, cuvs_runtime = fake_environment(
        fixture_root,
        metadata_mode=metadata_mode,
        cugraph=cugraph,
        cuvs=cuvs,
        logger=logger,
        logger_linkage=logger_linkage,
    )
    if cugraph and site.cugraph_feature_gated:
        environment["CARGO_FEATURE_CUGRAPH"] = "1"
    if cuvs and site.cuvs_feature_gated:
        environment["CARGO_FEATURE_CUVS"] = "1"
    result = run([str(executable)], env=environment)
    require_success(
        result,
        f"{site.name} cugraph={cugraph} cuvs={cuvs} metadata={metadata_mode} logger={logger}",
    )
    assert_link_target_output(site, result.stdout, cudf_dir)
    assert_cudf_rust_runtime_tracking(result.stdout, cudf_dir)
    assert_nvtx3interop_output(site, result.stdout, fixture_root)
    cugraph_tracking = "cargo:rerun-if-env-changed=DEP_CUGRAPH_METADATA"
    if (cugraph_tracking in result.stdout) != cugraph:
        raise AssertionError(f"{site.name}: unexpected cugraph tracking state\n{result.stdout}")
    cuvs_tracking = "cargo:rerun-if-env-changed=DEP_CUVS_C_METADATA"
    if (cuvs_tracking in result.stdout) != cuvs:
        raise AssertionError(f"{site.name}: unexpected cuvs tracking state\n{result.stdout}")
    if cuvs:
        cuvs_rpath = f"-Wl,-rpath,{cuvs_runtime}"
        if cuvs_rpath not in result.stdout:
            raise AssertionError(f"{site.name}: missing cuVS runtime rpath\n{result.stdout}")
        for library in ("libcuvs.so", "libcuvs_c.so"):
            if f"{library},--pop-state" not in result.stdout:
                raise AssertionError(f"{site.name}: missing required {library} link directive\n{result.stdout}")
    if (cugraph or cuvs) and not logger and logger_linkage != "absorbed":
        warning = "did not expose a shared librapids_logger.so (linkage=static)"
        if warning not in result.stdout:
            raise AssertionError(f"{site.name}: missing optional logger warning\n{result.stdout}")
    if (cugraph or cuvs) and logger_linkage == "absorbed":
        if "librapids_logger" in result.stdout:
            raise AssertionError(f"{site.name}: verified absorbed logger must stay quiet\n{result.stdout}")
    if not logger and "librapids_logger.so,--pop-state" in result.stdout:
        raise AssertionError(f"{site.name}: absent optional logger was linked\n{result.stdout}")
    if logger and "librapids_logger.so,--pop-state" not in result.stdout:
        raise AssertionError(f"{site.name}: present optional logger was not linked\n{result.stdout}")
    return result.stdout


def load_manifest(path: str) -> dict[str, object]:
    with (ROOT / path).open("rb") as source:
        return tomllib.load(source)


def assert_manifest_contract() -> None:
    root = load_manifest("Cargo.toml")
    workspace_dependencies = root["workspace"]["dependencies"]
    assert workspace_dependencies["cuvs"] == {
        "package": "cuvs-nexus",
        "path": "components/cuvs/cuvs-nexus",
        "version": "=0.1.0",
    }
    assert workspace_dependencies["cuvs-sys"] == {
        "package": "cuvs-nexus-sys",
        "path": "components/cuvs/cuvs-nexus-sys",
        "version": "=0.1.0",
    }
    assert root["features"]["cuvs"] == ["dep:cuvs", "dep:cuvs-sys", "nexus-query-engine/cuvs"]
    assert "server" not in root["features"]
    assert "server-cugraph" not in root["features"]
    assert "server-cuvs" not in root["features"]
    assert root["dependencies"]["cuvs"]["optional"] is True
    assert root["dependencies"]["cuvs-sys"]["optional"] is True

    server = load_manifest("crates/nexus-server/Cargo.toml")
    assert "datafusion-nexus/cuvs" in server["features"]["cuvs"]
    assert "dep:cuvs-sys" in server["features"]["cuvs"]
    assert server["dependencies"]["cuvs-sys"]["optional"] is True

    engine = load_manifest("crates/nexus-query-engine/Cargo.toml")
    assert engine["features"]["cuvs"] == ["dep:cuvs", "dep:cuvs-sys"]
    assert engine["dependencies"]["cuvs"]["optional"] is True
    assert engine["dependencies"]["cuvs-sys"]["optional"] is True

    bench = load_manifest("crates/nexus-bench/Cargo.toml")
    assert "dep:cuvs-sys" in bench["features"]["cuvs"]
    assert bench["dependencies"]["cuvs-sys"]["optional"] is True

    examples = load_manifest("crates/nexus-examples/Cargo.toml")
    assert "dep:cuvs-sys" in examples["features"]["cuvs"]
    assert examples["dependencies"]["cuvs-sys"]["optional"] is True


def main() -> int:
    try:
        assert_manifest_contract()
        with tempfile.TemporaryDirectory(prefix="datafusion-nexus-build-support-") as temporary:
            temporary_path = Path(temporary)
            policy_binary = temporary_path / "rapids-policy-tests"
            compile_policy = run(
                ["rustc", "--edition=2024", "-D", "warnings", "--test", str(SUPPORT), "-o", str(policy_binary)]
            )
            require_success(compile_policy, "build-support policy test compilation")
            listed = run([str(policy_binary), "--list", "--format=terse"])
            require_success(listed, "build-support policy test listing")
            policy_count = sum(line.endswith(": test") for line in listed.stdout.splitlines())
            policy_tests = run([str(policy_binary), "--test-threads=1"])
            require_success(policy_tests, "build-support policy tests")
            sys.stdout.write(policy_tests.stdout)

            executables: dict[str, Path] = {}
            for site in SITES:
                executable = temporary_path / f"build-{site.name}"
                compiled = run(
                    [
                        "rustc",
                        "--edition=2024",
                        "-D",
                        "warnings",
                        "--crate-name",
                        f"build_support_smoke_{site.name}",
                        str(ROOT / site.source),
                        "-o",
                        str(executable),
                    ]
                )
                require_success(compiled, f"{site.name} inclusion compile smoke")
                executables[site.name] = executable

            scenarios = [
                ("root-neither-direct", SITES[0], False, False, "direct", True, None),
                ("root-cuvs-aggregate", SITES[0], False, True, "aggregate", True, None),
                ("root-cugraph-aggregate", SITES[0], True, False, "aggregate", True, None),
                ("root-combined-direct", SITES[0], True, True, "direct", True, None),
                ("engine-neither-aggregate", SITES[1], False, False, "aggregate", True, None),
                ("engine-cuvs-direct-no-logger", SITES[1], False, True, "direct", False, None),
                ("engine-cugraph-aggregate-absorbed", SITES[1], True, False, "aggregate", False, "absorbed"),
                ("bench-neither-mixed", SITES[2], False, False, "mixed", True, None),
                ("bench-combined-mixed", SITES[2], True, True, "mixed", True, None),
                ("examples-cugraph-direct", SITES[3], True, False, "direct", True, None),
                ("examples-combined-direct", SITES[3], True, True, "direct", True, None),
                ("server-combined-direct", SITES[4], True, True, "direct", True, None),
            ]
            for name, site, cugraph, cuvs, metadata_mode, logger, logger_linkage in scenarios:
                run_site_scenario(
                    site,
                    executables[site.name],
                    temporary_path / name,
                    cugraph=cugraph,
                    cuvs=cuvs,
                    metadata_mode=metadata_mode,
                    logger=logger,
                    logger_linkage=logger_linkage,
                )

            missing_env, _, _, _, _ = fake_environment(
                temporary_path / "missing-rmm",
                metadata_mode="direct",
                cugraph=True,
                cuvs=False,
                logger=False,
                missing_rmm=True,
            )
            missing_env["CARGO_FEATURE_CUGRAPH"] = "1"
            missing = run([str(executables["root"])], env=missing_env)
            if missing.returncode == 0 or "requires libcudf to ship a shared librmm.so" not in missing.stderr:
                raise AssertionError(f"missing required RMM did not fail stably\n{missing.stdout}\n{missing.stderr}")

            divergent_env, _, _, _, _ = fake_environment(
                temporary_path / "divergent-rmm",
                metadata_mode="aggregate",
                cugraph=True,
                cuvs=False,
                logger=False,
                divergent_runtime_rmm=True,
            )
            divergent_env["CARGO_FEATURE_CUGRAPH"] = "1"
            divergent = run([str(executables["root"])], env=divergent_env)
            if divergent.returncode == 0 or "two divergent librmm.so builds" not in divergent.stderr:
                raise AssertionError(f"divergent RMM did not hard fail stably\n{divergent.stdout}\n{divergent.stderr}")

            absorbed_env, _, _, _, _ = fake_environment(
                temporary_path / "absorbed-with-shared-candidate",
                metadata_mode="direct",
                cugraph=True,
                cuvs=False,
                logger=False,
                logger_linkage="absorbed",
                graph_logger=True,
            )
            absorbed_env["CARGO_FEATURE_CUGRAPH"] = "1"
            absorbed = run([str(executables["root"])], env=absorbed_env)
            if absorbed.returncode == 0 or "two rapids_logger builds" not in absorbed.stderr:
                raise AssertionError(
                    f"shared logger reachable in absorbed state did not hard fail stably\n{absorbed.stdout}\n{absorbed.stderr}"
                )

            missing_c_api_env, _, _, _, _ = fake_environment(
                temporary_path / "missing-cuvs-c",
                metadata_mode="direct",
                cugraph=False,
                cuvs=True,
                logger=False,
                missing_cuvs_c=True,
            )
            missing_c_api_env["CARGO_FEATURE_CUVS"] = "1"
            missing_c_api = run([str(executables["root"])], env=missing_c_api_env)
            if missing_c_api.returncode == 0 or "LIBCUVS_C_RUNTIME_LIB" not in missing_c_api.stderr:
                raise AssertionError(
                    f"missing libcuvs_c did not fail stably\n{missing_c_api.stdout}\n{missing_c_api.stderr}"
                )

            divergent_cuvs_env, _, _, _, _ = fake_environment(
                temporary_path / "divergent-cuvs",
                metadata_mode="direct",
                cugraph=True,
                cuvs=True,
                logger=False,
                divergent_cuvs=True,
            )
            divergent_cuvs_env["CARGO_FEATURE_CUGRAPH"] = "1"
            divergent_cuvs_env["CARGO_FEATURE_CUVS"] = "1"
            divergent_cuvs = run([str(executables["root"])], env=divergent_cuvs_env)
            if divergent_cuvs.returncode == 0 or "different canonical libcuvs.so files" not in divergent_cuvs.stderr:
                raise AssertionError(
                    f"divergent cuVS identities did not hard fail\n{divergent_cuvs.stdout}\n{divergent_cuvs.stderr}"
                )

            private_cuvs_env, _, _, _, _ = fake_environment(
                temporary_path / "private-graph-cuvs",
                metadata_mode="direct",
                cugraph=True,
                cuvs=False,
                logger=False,
                private_graph_cuvs=True,
            )
            private_cuvs_env["CARGO_FEATURE_CUGRAPH"] = "1"
            private_cuvs = run([str(executables["root"])], env=private_cuvs_env)
            if private_cuvs.returncode == 0 or "private `_deps` build" not in private_cuvs.stderr:
                raise AssertionError(
                    f"private cuGraph cuVS did not hard fail\n{private_cuvs.stdout}\n{private_cuvs.stderr}"
                )

            scenario_count = len(scenarios) + 6
            print(
                f"OK: manifest feature contract, {policy_count} pure RAPIDS policy tests, "
                f"{len(SITES)} inclusion compile smokes, "
                f"and {scenario_count} isolated build-script scenarios passed."
            )
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
