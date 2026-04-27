#!/usr/bin/env python3
"""Run std-only RAPIDS build-support policy tests and inclusion smokes."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
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
    feature_gated: bool


SITES = (
    InclusionSite("root", "build.rs", "scripts/build/rapids.rs", True, True, True, True),
    InclusionSite(
        "engine",
        "crates/nexus-query-engine/build.rs",
        "../../scripts/build/rapids.rs",
        True,
        False,
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
    ),
    InclusionSite(
        "examples",
        "crates/nexus-examples/build.rs",
        "../../scripts/build/rapids.rs",
        False,
        True,
        False,
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
    logger: bool,
    missing_rmm: bool = False,
    divergent_runtime_rmm: bool = False,
    logger_linkage: str | None = None,
    graph_logger: bool = False,
) -> tuple[dict[str, str], Path, Path, Path]:
    cudf_dir = root / "cudf runtime"
    graph_native = root / "graph native"
    graph_runtime = root / "graph runtime"
    libcudf = cudf_dir / "libcudf.so"
    rmm = cudf_dir / "librmm.so"
    logger_path = cudf_dir / "librapids_logger.so"
    write_library(libcudf, b"fake libcudf")
    write_library(rmm, b"one rmm build")
    if logger:
        write_library(logger_path, b"one logger build")
    if cugraph:
        write_library(graph_native / "librmm.so", b"one rmm build")
        write_library(
            graph_runtime / "librmm.so",
            b"divergent rmm build" if divergent_runtime_rmm else b"one rmm build",
        )
        if logger or graph_logger:
            write_library(graph_native / "librapids_logger.so", b"one logger build")
            write_library(graph_runtime / "librapids_logger.so", b"one logger build")

    cudf_values = {
        "runtime-lib-dir": str(cudf_dir),
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
    }

    environment: dict[str, str] = {}
    if metadata_mode == "direct":
        direct_names = {
            "runtime-lib-dir": "DEP_CUDF_RUST_RUNTIME_LIB_DIR",
            "libcudf-runtime-lib": "DEP_CUDF_RUST_LIBCUDF_RUNTIME_LIB",
            "rmm-runtime-lib": "DEP_CUDF_RUST_RMM_RUNTIME_LIB",
            "rapids-logger-runtime-lib": "DEP_CUDF_RUST_RAPIDS_LOGGER_RUNTIME_LIB",
            "rapids-logger-linkage": "DEP_CUDF_RUST_RAPIDS_LOGGER_LINKAGE",
        }
        environment.update({direct_names[key]: value for key, value in cudf_values.items()})
        if cugraph:
            environment["DEP_CUGRAPH_RUNTIME_LIB_DIRS"] = graph_values["runtime-lib-dirs"]
            environment["DEP_CUGRAPH_NATIVE_LINK_DIRS"] = graph_values["native-link-dirs"]
    elif metadata_mode == "aggregate":
        environment["DEP_CUDF_RUST_METADATA"] = ";".join(f"{key}={value}" for key, value in cudf_values.items())
        if cugraph:
            environment["DEP_CUGRAPH_METADATA"] = ";".join(f"{key}={value}" for key, value in graph_values.items())
    elif metadata_mode == "mixed":
        environment["DEP_CUDF_RUST_METADATA"] = ";".join(f"{key}={value}" for key, value in cudf_values.items())
        environment["DEP_CUDF_RUST_RUNTIME_LIB_DIR"] = str(cudf_dir)
        environment["DEP_CUDF_RUST_LIBCUDF_RUNTIME_LIB"] = str(libcudf)
        if cugraph:
            environment["DEP_CUGRAPH_METADATA"] = f"native-link-dirs={graph_values['native-link-dirs']}"
            environment["DEP_CUGRAPH_RUNTIME_LIB_DIRS"] = graph_values["runtime-lib-dirs"]
    else:
        raise ValueError(f"unknown metadata mode: {metadata_mode}")

    return environment, cudf_dir, graph_native, graph_runtime


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


def run_site_scenario(
    site: InclusionSite,
    executable: Path,
    fixture_root: Path,
    *,
    cugraph: bool,
    metadata_mode: str,
    logger: bool,
    logger_linkage: str | None = None,
) -> str:
    environment, cudf_dir, _, _ = fake_environment(
        fixture_root,
        metadata_mode=metadata_mode,
        cugraph=cugraph,
        logger=logger,
        logger_linkage=logger_linkage,
    )
    if cugraph and site.feature_gated:
        environment["CARGO_FEATURE_CUGRAPH"] = "1"
    result = run([str(executable)], env=environment)
    require_success(result, f"{site.name} cugraph={cugraph} metadata={metadata_mode} logger={logger}")
    assert_link_target_output(site, result.stdout, cudf_dir)
    cugraph_tracking = "cargo:rerun-if-env-changed=DEP_CUGRAPH_METADATA"
    if (cugraph_tracking in result.stdout) != cugraph:
        raise AssertionError(f"{site.name}: unexpected cugraph tracking state\n{result.stdout}")
    if cugraph and not logger and logger_linkage != "absorbed":
        warning = "did not expose a shared librapids_logger.so (linkage=static)"
        if warning not in result.stdout:
            raise AssertionError(f"{site.name}: missing optional logger warning\n{result.stdout}")
    if cugraph and logger_linkage == "absorbed":
        if "librapids_logger" in result.stdout:
            raise AssertionError(f"{site.name}: verified absorbed logger must stay quiet\n{result.stdout}")
    if not logger and "librapids_logger.so,--pop-state" in result.stdout:
        raise AssertionError(f"{site.name}: absent optional logger was linked\n{result.stdout}")
    if logger and "librapids_logger.so,--pop-state" not in result.stdout:
        raise AssertionError(f"{site.name}: present optional logger was not linked\n{result.stdout}")
    return result.stdout


def main() -> int:
    try:
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
                ("root-disabled-direct", SITES[0], False, "direct", True, None),
                ("root-enabled-aggregate", SITES[0], True, "aggregate", True, None),
                ("engine-disabled-aggregate", SITES[1], False, "aggregate", True, None),
                ("engine-enabled-direct-no-logger", SITES[1], True, "direct", False, None),
                ("engine-enabled-aggregate-absorbed", SITES[1], True, "aggregate", False, "absorbed"),
                ("bench-disabled-mixed", SITES[2], False, "mixed", True, None),
                ("bench-enabled-mixed", SITES[2], True, "mixed", True, None),
                ("examples-enabled-direct", SITES[3], True, "direct", True, None),
            ]
            for name, site, cugraph, metadata_mode, logger, logger_linkage in scenarios:
                run_site_scenario(
                    site,
                    executables[site.name],
                    temporary_path / name,
                    cugraph=cugraph,
                    metadata_mode=metadata_mode,
                    logger=logger,
                    logger_linkage=logger_linkage,
                )

            missing_env, _, _, _ = fake_environment(
                temporary_path / "missing-rmm",
                metadata_mode="direct",
                cugraph=True,
                logger=False,
                missing_rmm=True,
            )
            missing_env["CARGO_FEATURE_CUGRAPH"] = "1"
            missing = run([str(executables["root"])], env=missing_env)
            if missing.returncode == 0 or "requires libcudf to ship a shared librmm.so" not in missing.stderr:
                raise AssertionError(f"missing required RMM did not fail stably\n{missing.stdout}\n{missing.stderr}")

            divergent_env, _, _, _ = fake_environment(
                temporary_path / "divergent-rmm",
                metadata_mode="aggregate",
                cugraph=True,
                logger=False,
                divergent_runtime_rmm=True,
            )
            divergent_env["CARGO_FEATURE_CUGRAPH"] = "1"
            divergent = run([str(executables["root"])], env=divergent_env)
            if divergent.returncode == 0 or "two divergent librmm.so builds" not in divergent.stderr:
                raise AssertionError(f"divergent RMM did not hard fail stably\n{divergent.stdout}\n{divergent.stderr}")

            absorbed_env, _, _, _ = fake_environment(
                temporary_path / "absorbed-with-shared-candidate",
                metadata_mode="direct",
                cugraph=True,
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

            scenario_count = len(scenarios) + 3
            print(
                f"OK: {policy_count} pure RAPIDS policy tests, {len(SITES)} inclusion compile smokes, "
                f"and {scenario_count} isolated build-script scenarios passed."
            )
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
