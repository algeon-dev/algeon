#!/usr/bin/env python3
"""Prepare the publication closure outside the checkout for release checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib

from check_workspace_publication import PUBLISHED, ROOT


RELEASE_FEATURES = ("cugraph", "cuvs", "cuml", "server", "nvml")


def outside_checkout(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(ROOT):
        raise ValueError(f"release artifacts must be outside the checkout: {path}")
    return path


def run_logged(command: list[str], log: Path, *, cwd: Path = ROOT, env: dict | None = None, json_output: bool = False) -> None:
    print(f"release check: {' '.join(command)} (log: {log})", flush=True)
    with log.open("w") as stream, log.with_suffix(".stderr.log").open("w") as errors:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=errors if json_output else subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-12000:], file=sys.stderr)
        raise RuntimeError(f"command exited {result.returncode}; see {log}")


def clean_environment(output: Path) -> dict[str, str]:
    """Share downloaded crates without inheriting Cargo or native configuration."""
    output = outside_checkout(output)
    for directory in (output, *output.parents):
        for name in ("config", "config.toml"):
            if (directory / ".cargo" / name).exists():
                raise ValueError(f"isolated cwd has ancestor Cargo configuration: {directory / '.cargo' / name}")
    cargo_home = output / "cargo-home"
    cargo_home.mkdir(parents=True, exist_ok=True)
    for name in ("config", "config.toml"):
        if (cargo_home / name).exists():
            raise ValueError(f"isolated Cargo home contains configuration: {cargo_home / name}")
    cache = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))).resolve()
    for name in ("registry", "git"):
        if (cache / name).exists() and not (cargo_home / name).exists():
            (cargo_home / name).symlink_to(cache / name, target_is_directory=True)
    prefixes = ("CARGO_", "ALGEON_", "CUDF_", "CUGRAPH_", "CUVS_", "CUML_", "CUDA_", "CMAKE_", "DEP_", "LD_",
                "CC_", "CXX_", "AR_", "CFLAGS_", "CXXFLAGS_")
    selectors = {
        "DOCS_RS", "RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTDOC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
        "CUDACXX", "CUDAHOSTCXX", "CUDAToolkit_ROOT", "NVCC", "CC", "CXX", "AR", "CFLAGS", "CXXFLAGS", "LDFLAGS",
        "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR",
        "DLPACK_INCLUDE_DIR", "LIBCUVS_USE_PYTHON", "CONDA_PREFIX", "VIRTUAL_ENV", "INSTALL_PREFIX", "CPPFLAGS",
    }
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(prefixes) and key not in selectors
           and not (key.startswith("RUST") and not key.startswith("RUSTUP_"))}
    env["CARGO_HOME"] = str(cargo_home)
    return env


def docs_features(manifest: dict) -> list[str]:
    name = manifest["package"]["name"]
    docs = manifest["package"].get("metadata", {}).get("docs", {}).get("rs")
    if docs is None:
        raise ValueError(f"{name}: missing docs.rs metadata")
    if docs.get("default-target") != "x86_64-unknown-linux-gnu" or docs.get("targets") != []:
        raise ValueError(f"{name}: invalid docs.rs target contract")
    expected = set(manifest.get("features", {})) - {"default", "test-hooks", "qualification"}
    features = docs.get("features", [])
    if set(features) != expected or len(features) != len(set(features)):
        raise ValueError(f"{name}: docs.rs features {features!r}; expected {sorted(expected)!r}")
    if docs.get("all-features", False):
        raise ValueError(f"{name}: docs.rs must not enable all-features/test-hooks")
    return features


def dependency_tables(manifest: dict):
    for owner in [manifest, *manifest.get("target", {}).values()]:
        for kind in ("dependencies", "build-dependencies", "dev-dependencies"):
            yield from owner.get(kind, {}).items()


def native_build_inputs(original: Path) -> list[str]:
    required = {"build.rs", "build_support", "build_support.rs", "native", "shim", "bindings", "bindgen-stubs"}
    inputs = []
    for entry in original.iterdir():
        if entry.name not in required and not entry.name.endswith("_wrapper.h") and entry.name != "wrapper.h":
            continue
        for source in ([entry] if entry.is_file() else entry.rglob("*")):
            if source.is_file():
                inputs.append(str(source.relative_to(original)))
    if original.name.endswith("-sys"):
        inputs.append("src/bindings.rs")
    return sorted(inputs)


def validate_inputs(path: Path, inputs: list[str]) -> None:
    for relative in inputs:
        source = (path / relative).resolve()
        if not source.is_relative_to(path) or not source.is_file():
            raise ValueError(f"{path.name}: archive omitted native/build input {relative}")


def validate_manifest(path: Path, versions: dict[str, str]) -> dict:
    with (path / "Cargo.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    package = manifest["package"]
    name = package["name"]
    if name not in versions or package["version"] != versions[name]:
        raise ValueError(f"{path}: package identity differs from the release closure")
    if package.get("publish") != ["crates-io"] or "workspace" in manifest:
        raise ValueError(f"{name}: archive retains workspace/publication configuration")
    for alias, dependency in dependency_tables(manifest):
        dependency = {"version": dependency} if isinstance(dependency, str) else dependency
        if "path" in dependency or "git" in dependency or "workspace" in dependency:
            raise ValueError(f"{name}: normalized dependency {alias} retains path/git/workspace")
        target = dependency.get("package", alias)
        if target in versions and dependency.get("version") != f"={versions[target]}":
            raise ValueError(f"{name}: {target} must pin exact release version ={versions[target]}")
        if target.startswith("algeon-") and target not in PUBLISHED:
            raise ValueError(f"{name}: dependency {target} is outside the publication closure")
    build = package.get("build")
    if isinstance(build, str) and not (path / build).is_file():
        raise ValueError(f"{name}: missing packaged build script {build}")
    return {"name": name, "version": package["version"], "path": str(path), "docs_features": docs_features(manifest)}


def validate_archives(output: Path) -> list[dict]:
    output = outside_checkout(output)
    release = json.loads((output / "release.json").read_text())
    packages = release["packages"]
    names = [package["name"] for package in packages]
    if set(names) != PUBLISHED or len(names) != len(PUBLISHED):
        raise ValueError("archive package set differs from the publication allowlist")
    versions = {package["name"]: package["version"] for package in packages}
    build_inputs = release.get("build_inputs", {})
    if set(build_inputs) != PUBLISHED:
        raise ValueError("native/build input inventory differs from the publication allowlist")
    for package in packages:
        path = outside_checkout(Path(package["path"]))
        if not path.is_relative_to(output / "sources"):
            raise ValueError(f"archive package is outside {output / 'sources'}: {path}")
        if validate_manifest(path, versions) != package:
            raise ValueError(f"{package['name']}: release metadata differs from archive manifest")
        validate_inputs(path, build_inputs[package["name"]])
    with (output / "patches.toml").open("rb") as stream:
        patches = tomllib.load(stream).get("patch", {}).get("crates-io", {})
    if patches != {package["name"]: {"path": package["path"]} for package in packages}:
        raise ValueError("patches.toml must point the complete publication closure to its archives")
    return packages


def prepare(output: Path, require_clean: bool) -> None:
    output = outside_checkout(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    if require_clean:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"], cwd=ROOT, text=True
        )
        if status:
            raise ValueError("release preparation requires a clean checkout; omit --require-clean for development checks")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    run_logged(["python3", str(ROOT / "scripts/ci/check_workspace_publication.py")], logs / "publication-policy.log")
    metadata_command = ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"]
    run_logged(metadata_command, logs / "metadata.json", json_output=True)
    metadata = json.loads((logs / "metadata.json").read_text())
    members = {package["name"]: package for package in metadata["packages"] if package["id"] in metadata["workspace_members"]}
    versions = {name: members[name]["version"] for name in PUBLISHED}
    shutil.copyfile(ROOT / "Cargo.lock", output / "Cargo.lock")
    command = ["cargo", "package", "--workspace", "--locked", "--no-verify", "--no-default-features"]
    if not require_clean:
        command.append("--allow-dirty")
    for name in sorted(set(members) - PUBLISHED):
        command.extend(["--exclude", name])
    command.extend(["--features", ",".join(f"algeon-datafusion/{feature}" for feature in RELEASE_FEATURES)])
    env = dict(os.environ, CARGO_TARGET_DIR=str(output / "cargo-target"))
    run_logged(command, logs / "package.log", env=env)
    sources = output / "sources"
    sources.mkdir()
    packages = []
    build_inputs = {}
    for name in sorted(PUBLISHED):
        version = versions[name]
        archive = output / "cargo-target/package" / f"{name}-{version}.crate"
        with tarfile.open(archive) as stream:
            stream.extractall(sources, filter="data")
        path = sources / f"{name}-{version}"
        packages.append(validate_manifest(path, versions))
        original = Path(members[name]["manifest_path"]).parent
        # Local native sources and build helpers must ship even when DOCS_RS
        # skips compiling them. Compare these trees before running rustdoc.
        build_inputs[name] = native_build_inputs(original)
        validate_inputs(path, build_inputs[name])
    (output / "release.json").write_text(json.dumps({"packages": packages, "release_features": list(RELEASE_FEATURES), "build_inputs": build_inputs}, indent=2) + "\n")
    (output / "patches.toml").write_text(
        "[patch.crates-io]\n" + "".join(f"{package['name']} = {{ path = {json.dumps(package['path'])} }}\n" for package in packages)
    )
    validate_archives(output)
    print(f"PASS: prepared {len(packages)} release archives: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="empty directory outside the checkout")
    parser.add_argument("--require-clean", action="store_true", help="reject dirty release source, including submodules")
    args = parser.parse_args()
    try:
        prepare(args.output, args.require_clean)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FAIL: release archives: {error}\n")


if __name__ == "__main__":
    main()
