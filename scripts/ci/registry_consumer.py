#!/usr/bin/env python3
"""Build and execute the release application using extracted crates only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

from release_archives import ROOT, clean_environment, validate_archives


FEATURES = ("", "cuvs", "cugraph", "cuml", "cugraph,cuvs,cuml", "server", "nvml", "cugraph,cuvs,cuml,server,nvml")
FIXTURE = ROOT / "scripts/ci/fixtures/registry-consumer"
NVML_PACKAGES = ("nvml-wrapper", "nvml-wrapper-sys")


def check_quickstart() -> None:
    document = (ROOT / "website/docs/build/rust-dependency.md").read_text()
    for filename in ("Cargo.toml", "src/main.rs"):
        marker = "{/* registry-consumer:" + filename + " */}"
        match = re.search(re.escape(marker) + r"\s*```[^\n]*\n(.*?)\n```", document, re.S)
        if match is None or match[1].strip() != (FIXTURE / filename).read_text().strip():
            raise RuntimeError(f"quickstart {filename} differs from the tested registry consumer")


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path, *, failure: str | None = None) -> str:
    print(f"registry-consumer: {log.stem}", flush=True)
    with log.open("w") as output:
        output.write(f"cwd: {cwd}\ncommand: {json.dumps(command)}\n")
        output.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    text = log.read_text()
    if failure is not None:
        if result.returncode == 0 or failure not in text:
            raise RuntimeError(f"expected failure containing {failure!r}: {log}\n{text[-6000:]}")
    elif result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {log}\n{text[-6000:]}")
    return text


def native_closure(features: str) -> list[str]:
    enabled = set(features.split(","))
    components = ["cudf"]
    if enabled & {"cuvs", "cugraph", "cuml"}:
        components.append("cuvs")
    components.extend(component for component in ("cugraph", "cuml") if component in enabled)
    return components


def verify_loader(text: str, native: Path, features: str) -> None:
    if "not found" in text:
        raise RuntimeError(f"unresolved runtime dependency:\n{text}")
    libraries = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
        if match:
            libraries[match[1]] = Path(match[2]).resolve()
    required = {"cudf": ("libcudf_rust.so", "libcudf.so", "librmm.so"),
                "cuvs": ("libcuvs.so",), "cugraph": ("libcugraph.so",),
                "cuml": ("libcuml_rust.so", "libcuml.so", "libtreelite.so", "libnvforest.so")}
    for component in native_closure(features):
        for name in required[component]:
            candidates = [path for soname, path in libraries.items() if soname == name or soname.startswith(name + ".")]
            expected = (native / f"{component}-install/lib" / name).resolve(strict=True)
            if candidates != [expected]:
                raise RuntimeError(f"loader selected {candidates} for {name}; expected {expected}")


def verify_resolution(metadata: dict, packages: list[dict], app: Path, release_lock: Path | None) -> None:
    archives = {p["name"]: Path(p["path"]).resolve() for p in packages}
    for package in metadata["packages"]:
        name = package["name"]
        if name in archives:
            if Path(package["manifest_path"]).parent.resolve() != archives[name]:
                raise RuntimeError(f"{name} resolved outside the extracted release archives")
        elif package["source"] is None and Path(package["manifest_path"]).parent != app:
            raise RuntimeError(f"unexpected path dependency: {package['manifest_path']}")
    for node in metadata["resolve"]["nodes"]:
        if "algeon-" in node["id"] and {"test-hooks", "qualification"} & set(node["features"]):
            raise RuntimeError(f"non-release feature enabled: {node}")
    if release_lock:
        locked = tomllib.loads(release_lock.read_text())["package"]
        expected = {(p["name"], p["version"], p.get("source"), p.get("checksum")) for p in locked}
        actual = tomllib.loads((app / "Cargo.lock").read_text())["package"]
        changed = [p for p in actual if p.get("source") and
                   (p["name"], p["version"], p.get("source"), p.get("checksum")) not in expected]
        if changed:
            raise RuntimeError(f"release lock resolution changed third-party packages: {changed}")


def verify_nvml_selection(metadata: dict, features: str) -> None:
    """NVML enters the resolved graph only when the row selects `nvml`."""
    names = {package["id"]: package["name"] for package in metadata["packages"]}
    resolved = {names.get(node["id"]) for node in metadata["resolve"]["nodes"]}
    present = sorted(resolved & set(NVML_PACKAGES))
    selected = "nvml" in features.split(",")
    if selected and present != sorted(NVML_PACKAGES):
        raise RuntimeError(f"the nvml row resolved {present}; expected {list(NVML_PACKAGES)}")
    if not selected and present:
        raise RuntimeError(f"feature row {features or 'default'!r} resolved {present} without selecting nvml")


def discovery_regressions(output: Path, archives: Path, env: dict[str, str], native: Path, version: str) -> None:
    def check(label: str, selected: dict[str, str], *, failure: str | None = None) -> str:
        text = run(command, probe, selected, logs / f"discovery-{label}.log", failure=failure)
        if not re.search(r"Running .*algeon-cuml-sys-[^/]+/build-script-build", text):
            raise RuntimeError(f"{label}: Cargo did not rerun the cuML discovery build script")
        return text

    probe = output / "discovery"
    (probe / "src").mkdir(parents=True)
    (probe / "src/main.rs").write_text("fn main() {}\n")
    (probe / "Cargo.toml").write_text(
        '[package]\nname = "cuml-discovery-consumer"\nversion = "0.0.0"\nedition = "2024"\n'
        f'[dependencies]\nalgeon-cuml-sys = "={version}"\n[workspace]\n'
        + (archives / "patches.toml").read_text()
    )
    logs = output / "logs"
    run(["cargo", "generate-lockfile"], probe, env, logs / "discovery-resolve.log")
    command = ["cargo", "check", "--locked", "--offline", "-vv"]
    check("root-only", env)
    unchanged = run(command, probe, env, logs / "discovery-unchanged.log")
    if not re.search(r"Fresh algeon-cuml-sys ", unchanged) or re.search(
        r"Running .*algeon-cuml-sys-[^/]+/build-script-build", unchanged
    ):
        raise RuntimeError("unchanged native selectors and marker must keep cuML discovery fresh")
    for component in ("cudf", "cuvs"):
        override = output / f"override-{component}"
        override.mkdir()
        (override / "lib").symlink_to(native / f"{component}-install/lib", target_is_directory=True)
        selected = {**env, f"{component.upper()}_INSTALL_DIR": str(override)}
        check(f"{component}-override", selected)
        selector = f"{component.upper()}_INSTALL_DIR"
        missing_prefix = output / f"missing-{component}"
        selected[selector] = str(missing_prefix)
        library = "librmm.so" if component == "cudf" else "libcuvs.so"
        rejected = check(f"{component}-invalid", selected, failure=f"algeon-{component}-sys: no usable {library}")
        for expected in (f"{selector} selected prefix {missing_prefix}",
                         f"git clone --branch v{version} --recurse-submodules",
                         f"bash scripts/build/build_lib{component}.sh"):
            if expected not in rejected:
                raise RuntimeError(f"invalid {component} override diagnostic is missing {expected!r}")
        check(f"{component}-restore", env)

    cuda_selectors = ("CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH")
    cuda_lib = Path(next(env[name] for name in cuda_selectors if name in env)) / "lib64"
    for selector in cuda_selectors:
        for index in (1, 2):
            prefix = output / f"cuda-{selector.lower()}-{index}"
            prefix.mkdir()
            (prefix / "lib64").symlink_to(cuda_lib, target_is_directory=True)
            selected = {name: value for name, value in env.items() if name not in cuda_selectors}
            selected[selector] = str(prefix)
            text = check(f"cuda-{selector.lower()}-{index}", selected)
            if f"cargo:metadata=cuda-runtime={prefix}" not in text:
                raise RuntimeError(f"{selector} change did not select the requested CUDA runtime root {prefix}")
        check(f"cuda-{selector.lower()}-restore", env)

    prefix = output / "marker-cuml"
    marker = prefix / "share/algeon-cuml/crate-version"
    marker.parent.mkdir(parents=True)
    (prefix / "lib").symlink_to(native / "cuml-install/lib", target_is_directory=True)
    (prefix / "share/cuml_rust").symlink_to(native / "cuml-install/share/cuml_rust", target_is_directory=True)
    selected = {**env, "CUML_INSTALL_DIR": str(prefix)}
    marker.write_text(version + "\n")
    check("marker-match", selected)
    marker.write_text("9.9.9\n")
    mismatch = check("marker-mismatch", selected, failure="were built for crate version 9.9.9")
    marker.write_text(version + "\n")
    check("marker-before-missing", selected)
    marker.unlink()
    missing = check("marker-missing", selected, failure="have no readable share/algeon-cuml/crate-version")
    for text in (mismatch, missing):
        diagnostic = "\n".join(
            re.sub(r"^\[algeon-cuml-sys [^]]+\]\s?", "", line).strip() for line in text.splitlines()
        )
        rebuild = (
            f"cd <repo>\ngit checkout v{version}\ngit submodule update --init --recursive\n"
            "bash scripts/build/build_libcudf.sh\nbash scripts/build/build_libcuvs.sh\n"
            "bash scripts/build/build_libcuml.sh"
        )
        if f"algeon-cuml-sys {version}:" not in diagnostic or rebuild not in diagnostic or str(prefix) not in diagnostic:
            raise RuntimeError("cuML marker failure must name the prefix, expected version and ordered rebuild procedure")
    marker.write_text(version + "\n")
    check("marker-restored", selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, help="reuse release_archives.py output")
    parser.add_argument("--output", type=Path, help="new directory outside the checkout; retained after the run")
    parser.add_argument("--target-dir", type=Path, help="reuse an external Cargo target directory for focused reruns")
    parser.add_argument("--native-root", type=Path, default=Path(os.environ.get("ALGEON_NATIVE_DIR", ROOT / "target/native")))
    parser.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--jobs", type=int, default=4, help="maximum concurrent Cargo jobs (default: 4)")
    parser.add_argument("--features", action="append", choices=FEATURES, help="focused subset; repeat for several rows")
    parser.add_argument("--lane", choices=("fresh", "locked", "both"), default="both")
    parser.add_argument("--skip-discovery", action="store_true", help="focused rerun of application checks only")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--check-doc-example", action="store_true")
    args = parser.parse_args()
    check_quickstart()
    if args.check_doc_example:
        print("registry quickstart matches the consumer fixture")
        return
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="algeon-registry-consumer-"))
    if output.is_relative_to(ROOT.resolve()) or (output.exists() and any(output.iterdir())):
        raise RuntimeError("--output must be empty and outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    print(f"registry-consumer: artifacts at {output}", flush=True)
    archives = args.archives.resolve() if args.archives else output / "archives"
    if not args.archives:
        command = [sys.executable, str(ROOT / "scripts/ci/release_archives.py"), "--output", str(archives)]
        if args.require_clean:
            command.append("--require-clean")
        run(command, ROOT, dict(os.environ), logs / "archives.log")
    if args.archives and args.require_clean:
        raise RuntimeError("--require-clean applies to preparation; use a fresh --output without --archives")
    packages = validate_archives(archives)
    env = clean_environment(output)
    native, cuda = args.native_root.resolve(strict=True), args.cuda_home.resolve(strict=True)
    target = args.target_dir.resolve() if args.target_dir else output / "target"
    if target.is_relative_to(ROOT.resolve()):
        raise RuntimeError("--target-dir must be outside the checkout")
    if args.target_dir:
        # Cargo fingerprints registry sources by their Cargo-home paths too.
        env["CARGO_HOME"] = clean_environment(target.parent)["CARGO_HOME"]
    env.update(ALGEON_NATIVE_DIR=str(native), CUDA_HOME=str(cuda), CC=args.cc, CXX=args.cxx,
               CARGO_BUILD_JOBS=str(args.jobs),
               CARGO_TARGET_DIR=str(target), CARGO_PROFILE_DEV_DEBUG="0", CARGO_INCREMENTAL="0")
    versions = {name: run([name, "--version"], output, env, logs / f"{name}-version.log") for name in ("cargo", "rustc")}
    (output / "environment.json").write_text(json.dumps({
        "toolchain": versions, "native_root": str(native), "cuda_home": str(cuda), "cc": args.cc,
        "cxx": args.cxx, "features": args.features or FEATURES, "lane": args.lane,
    }, indent=2) + "\n")
    run_cwd = output / "run-from-another-cwd"
    run_cwd.mkdir()
    results = []
    for lane in (("fresh", "locked") if args.lane == "both" else (args.lane,)):
        app = output / lane
        shutil.copytree(FIXTURE, app)
        with (app / "Cargo.toml").open("a") as manifest:
            manifest.write("\n" + (archives / "patches.toml").read_text())
        release_lock = archives / "Cargo.lock" if lane == "locked" else None
        if release_lock:
            shutil.copyfile(release_lock, app / "Cargo.lock")
            # Cargo prunes workspace-only packages and adds the consumer root.
            # verify_resolution below rejects any third-party version change.
            run(["cargo", "metadata", "--offline", "--format-version", "1"], app, env, logs / f"{lane}-resolve.log")
        else:
            run(["cargo", "generate-lockfile"], app, env, logs / "fresh-resolve.log")
        for features in args.features or FEATURES:
            label = f"{lane}-{features.replace(',', '-') or 'default'}"
            selection = ["--no-default-features"] + (["--features", features] if features else [])
            run(["cargo", "fetch", "--locked"], app, env, logs / f"{label}-fetch.log")
            metadata_log = run(["cargo", "metadata", "--locked", "--offline", "--format-version", "1", *selection],
                               app, env, logs / f"{label}-metadata.log")
            metadata = json.loads(next(line for line in metadata_log.splitlines() if line.startswith('{"packages":')))
            verify_resolution(metadata, packages, app, release_lock)
            verify_nvml_selection(metadata, features)
            run(["cargo", "build", "--locked", "--offline", *selection], app, env, logs / f"{label}-build.log")
            binary = target / "debug/registry-consumer"
            elf = run(["readelf", "--wide", "--program-headers", "--dynamic", str(binary)],
                      run_cwd, env, logs / f"{label}-elf.log")
            interpreter = re.search(r"\[Requesting program interpreter: ([^]]+)\]", elf)
            if interpreter is None:
                raise RuntimeError(f"missing ELF interpreter: {binary}")
            runtime = dict(env)
            runtime["LD_LIBRARY_PATH"] = ":".join(
                [str(native / f"{component}-install/lib") for component in native_closure(features)] + [str(cuda / "lib64")]
            )
            # ldd also reports absent optional DT_AUXILIARY filters, including
            # CUDA 13.2 NVTX's visibility=hidden entry. --list uses the actual
            # loader's required-library semantics and fails on missing DT_NEEDED.
            loader = run([interpreter[1], "--list", str(binary)], run_cwd, runtime, logs / f"{label}-loader.log")
            verify_loader(loader, native, features)
            run(["flock", "--shared", "/tmp/cudf-gpu.lock", str(binary)], run_cwd, runtime, logs / f"{label}-run.log")
            results.append({"lane": lane, "features": features, "loader_path": runtime["LD_LIBRARY_PATH"], "status": "PASS"})
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if not args.skip_discovery:
        version = next(p["version"] for p in packages if p["name"] == "algeon-cuml-sys")
        discovery_regressions(output, archives, env, native, version)
    print(f"PASS: {len(results)} archive consumer build/link/run rows; evidence at {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
