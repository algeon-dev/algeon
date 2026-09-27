#!/usr/bin/env python3
"""Write a deterministic TPCH GPU query-engine baseline manifest.

The run config is JSON data only: this helper never sources env.sh or executes
run-command text. The benchmark must also provide a loaded-native-libraries
snapshot captured from its own ``/proc/self/maps`` after execution; a build
directory guess is never used as a substitute. Unknown keys and secret-looking
environment values are not copied into the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

NATIVE_SOURCES = {"cudf": "components/cudf/cudf", "cugraph": "components/cugraph/cugraph", "cuvs": "components/cuvs/cuvs"}
ENV_NAMES = tuple("""ALGEON_NATIVE_DIR ALGEON_PARQUET_DIR ALGEON_TPCH_BENCH_DATASET_MODE
ALGEON_TPCH_BENCH_ITERATIONS ALGEON_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES
ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES
ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT ALGEON_TPCH_BENCH_NATIVE_PROFILE
ALGEON_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE ALGEON_TPCH_BENCH_PARQUET_PUSHDOWN_FILTERS
ALGEON_TPCH_BENCH_QUERY_IDS ALGEON_TPCH_BENCH_WARMUPS ALGEON_TPCH_CPU_PROFILE ALGEON_TPCH_SCALE_FACTOR
CARGO_BUILD_JOBS CARGO_BUILD_TARGET CARGO_PROFILE CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES CUDAToolkit_ROOT
RAYON_NUM_THREADS RUSTUP_TOOLCHAIN TOKIO_WORKER_THREADS""".split())
CONFIG_KEYS = set("""binary build_profile cache_share_bytes configured_cpu_threads cpu_logical_threads cpu_profile
dataset_content_fingerprint dataset_metadata_fingerprint dataset_mode dataset_root dataset_scale_factor evidence_paths
dataset_table_row_group_counts dataset_total_bytes dataset_total_rows device_cache_share_bytes device_budget_bytes
max_source_reads_in_flight native_artifacts native_device_budget_bytes native_profile native_root operator_width trace_paths
loaded_libraries_path native_library_snapshot runtime_library_snapshot profile profile_source query_cap_bytes query_ids row_groups_per_chunk scale_factor source_cache_state
source_chunk_bytes source_chunk_planning_mode tpch_dataset_root tpch_scale_factor""".split())
RUN_KEYS = set("""cpu_logical cpu_profile driver_version gpu0_cuda_total_bytes gpu0_managed_capacity_bytes gpu0_model
gpu0_physical_total_bytes gpu0_uuid gpu_count max_row_groups_per_chunk max_source_chunk_bytes
max_source_reads_in_flight profile_source query_cap_bytes rust_profile source_chunk_planning_mode source_cache_state
stage operator_width""".split())
TSV_KEYS = set("""tpch_benchmark_dataset_mode tpch_benchmark_iterations tpch_benchmark_max_row_groups_per_chunk
tpch_benchmark_native_device_budget_bytes tpch_benchmark_native_max_retry_attempts
tpch_benchmark_native_max_source_chunk_bytes tpch_benchmark_native_max_source_reads_in_flight
tpch_benchmark_native_min_retry_chunk_row_groups tpch_benchmark_native_profile_name
tpch_benchmark_native_profile_source tpch_benchmark_native_source_chunk_planning_mode tpch_benchmark_query_ids
tpch_benchmark_warmups tpch_dataset_content_fingerprint tpch_dataset_file_count tpch_dataset_metadata_fingerprint
tpch_dataset_provenance tpch_dataset_root tpch_dataset_scale_factor tpch_dataset_table_count
tpch_dataset_table_byte_sizes tpch_dataset_table_file_counts tpch_dataset_table_row_counts
tpch_dataset_table_row_group_counts tpch_dataset_total_bytes tpch_dataset_total_rows""".split())
SECRET_KEY = re.compile(r"(?:pass(word)?|secret|token|api[_-]?key|authorization|cookie|credential)", re.I)
URI_CREDENTIALS = re.compile(r"^[a-z][a-z0-9+.-]*://[^/]*:[^/@]+@", re.I)
REQUIRED_LOADED_LIBRARIES = ("libcudf_rust.so", "libcudf.so")


class ManifestError(RuntimeError):
    pass


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def fact(value=None, source=None, reason=None):
    result = {"value": value}
    if source is not None:
        result["source"] = source
    if reason is not None:
        result["reason"] = reason
    return result


def safe(value, field=""):
    return "<redacted>" if isinstance(value, str) and (SECRET_KEY.search(field) or URI_CREDENTIALS.search(value)) else value


def pint(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer")
    result = int(str(value).strip())
    if result < 0:
        raise ValueError("negative values are not supported")
    return result


def plist(value):
    values = value if isinstance(value, list) else str(value).split(",")
    result = [str(item).strip() for item in values if str(item).strip()]
    if not result or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", item) for item in result):
        raise ValueError("expected identifier list")
    return result


def pmap(value):
    if not isinstance(value, str):
        raise ValueError("expected key=value pairs")
    result = {}
    for item in value.split(","):
        name, separator, number = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("expected key=value pairs")
        result[name] = pint(number)
    return dict(sorted(result.items()))


def hfile(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value, repo: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def shown(path: Path, repo: Path) -> str:
    try:
        relative = os.path.relpath(str(path), str(repo))
        if not relative.startswith(".." + os.sep) and relative != "..":
            return relative
    except ValueError:
        pass
    return str(path.absolute())


def fingerprint(path: Path | None, repo: Path, source: str, label: str):
    if path is None:
        return fact(None, reason=f"{label} was not supplied")
    if not path.exists():
        return fact(None, reason=f"{label} does not exist: {shown(path, repo)}")
    value = {"path": shown(path, repo), "kind": "directory" if path.is_dir() else "file"}
    if path.is_file():
        value.update(size_bytes=path.stat().st_size, sha256=hfile(path))
    return fact(value, source=source)


def _snapshot_string(value, field, *, required=True):
    if not isinstance(value, str) or not value.strip():
        if required:
            raise ManifestError(f"loaded library snapshot field {field} must be a non-empty string")
        return None
    return value


def _snapshot_sha256(value, field):
    value = _snapshot_string(value, field)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ManifestError(f"loaded library snapshot field {field} is not a SHA-256 digest")
    return value.lower()


def _snapshot_integer(value, field):
    if isinstance(value, bool):
        raise ManifestError(f"loaded library snapshot field {field} is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ManifestError(f"loaded library snapshot field {field} is not an integer") from error
    if result < 0:
        raise ManifestError(f"loaded library snapshot field {field} is negative")
    return result


def _verified_snapshot_file(entry, field, repo):
    if not isinstance(entry, dict):
        raise ManifestError(f"loaded library snapshot {field} must be an object")
    recorded_path = _snapshot_string(entry.get("path"), f"{field}.path")
    recorded_sha256 = _snapshot_sha256(entry.get("sha256"), f"{field}.sha256")
    recorded_size = _snapshot_integer(entry.get("size_bytes"), f"{field}.size_bytes")
    recorded_device = _snapshot_integer(entry.get("device"), f"{field}.device")
    recorded_inode = _snapshot_integer(entry.get("inode"), f"{field}.inode")
    if not Path(recorded_path).is_absolute():
        raise ManifestError(f"loaded library snapshot {field}.path is not canonical: {recorded_path}")
    resolved_path = Path(recorded_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = repo / resolved_path
    try:
        resolved_path = resolved_path.resolve(strict=True)
    except OSError as error:
        raise ManifestError(f"loaded library snapshot {field} path cannot be resolved: {recorded_path}") from error
    if Path(recorded_path) != resolved_path:
        raise ManifestError(f"loaded library snapshot {field}.path is not canonical: {recorded_path}")
    if not resolved_path.is_file():
        raise ManifestError(f"loaded library snapshot {field} is not a regular file: {recorded_path}")
    try:
        stat = resolved_path.stat()
        actual_size = stat.st_size
        actual_sha256 = hfile(resolved_path)
    except OSError as error:
        raise ManifestError(f"loaded library snapshot {field} cannot be read: {recorded_path}") from error
    if actual_size != recorded_size:
        raise ManifestError(
            f"loaded library snapshot {field} size changed: recorded {recorded_size}, observed {actual_size}: {recorded_path}"
        )
    if stat.st_dev != recorded_device or stat.st_ino != recorded_inode:
        raise ManifestError(
            f"loaded library snapshot {field} identity changed: recorded device/inode "
            f"{recorded_device}/{recorded_inode}, observed {stat.st_dev}/{stat.st_ino}: {recorded_path}"
        )
    if actual_sha256 != recorded_sha256:
        raise ManifestError(
            f"loaded library snapshot {field} hash changed: recorded {recorded_sha256}, observed {actual_sha256}: {recorded_path}"
        )
    return {
        "path": recorded_path,
        "sha256": recorded_sha256,
        "size_bytes": recorded_size,
        "device": recorded_device,
        "inode": recorded_inode,
        "resolved_path": shown(resolved_path, repo),
        "verified": True,
    }, resolved_path


def loaded_library_snapshot(path: Path | None, repo: Path, binary: Path | None, source="run_config"):
    """Validate the loader snapshot emitted by the benchmark process.

    The benchmark writes this file after native execution from its own
    ``/proc/self/maps`` view.  Re-resolving and hashing every entry here makes
    the manifest fail closed if a library was replaced after the run, while
    preserving the path and digest that the running process recorded.
    """
    if path is None:
        raise ManifestError(
            "a runtime loaded-library snapshot is required; pass --loaded-libraries "
            "or set loaded_libraries_path in the run config"
        )
    try:
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read runtime loaded-library snapshot {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ManifestError("runtime loaded-library snapshot must contain an object")
    if raw.get("source") != "/proc/self/maps":
        raise ManifestError("runtime loaded-library snapshot source must be /proc/self/maps")
    if binary is None:
        raise ManifestError("a benchmark binary is required when validating a runtime loaded-library snapshot")
    if not binary.is_file():
        raise ManifestError(f"benchmark binary does not exist: {shown(binary, repo)}")
    try:
        binary_sha256 = hfile(binary)
    except OSError as error:
        raise ManifestError(f"benchmark binary cannot be read: {binary}") from error
    runtime_binary = raw.get("binary")
    if not isinstance(runtime_binary, dict):
        raise ManifestError("runtime loaded-library snapshot binary must be an object")
    runtime_binary_path = _snapshot_string(runtime_binary.get("path"), "binary.path")
    if not Path(runtime_binary_path).is_absolute():
        raise ManifestError("runtime loaded-library snapshot binary.path must be absolute")
    runtime_binary_sha256 = _snapshot_sha256(runtime_binary.get("sha256"), "binary.sha256")
    runtime_binary_size = _snapshot_integer(runtime_binary.get("size_bytes"), "binary.size_bytes")
    runtime_binary_device = _snapshot_integer(runtime_binary.get("device"), "binary.device")
    runtime_binary_inode = _snapshot_integer(runtime_binary.get("inode"), "binary.inode")
    if binary_sha256 != runtime_binary_sha256 or binary.stat().st_size != runtime_binary_size:
        raise ManifestError("archived benchmark binary does not match the executable recorded by the benchmark process")
    libraries = raw.get("libraries")
    if not isinstance(libraries, list) or not libraries:
        raise ManifestError("runtime loaded-library snapshot libraries must be a non-empty list")
    verified_libraries = []
    seen_paths = set()
    names = set()
    for index, entry in enumerate(libraries):
        verified, resolved_path = _verified_snapshot_file(entry, f"libraries[{index}]", repo)
        if resolved_path in seen_paths:
            raise ManifestError(f"runtime loaded-library snapshot contains duplicate path: {verified['path']}")
        seen_paths.add(resolved_path)
        names.add(resolved_path.name)
        verified_libraries.append(verified)
    for required in REQUIRED_LOADED_LIBRARIES:
        if required not in names and not any(name.startswith(required + ".") for name in names):
            raise ManifestError(f"runtime loaded-library snapshot is missing required dependency {required}")
    return {
        "snapshot": fingerprint(path, repo, source, "runtime loaded-library snapshot"),
        "source": raw["source"],
        "libraries": verified_libraries,
        "library_count": len(verified_libraries),
        "binary_sha256": binary_sha256,
        "runtime_binary": {
            "path": runtime_binary_path,
            "sha256": runtime_binary_sha256,
            "size_bytes": runtime_binary_size,
            "device": runtime_binary_device,
            "inode": runtime_binary_inode,
        },
    }


def run(command: list[str], timeout=10):
    executable = shutil.which(command[0])
    if executable is None:
        return None, f"{command[0]} is unavailable"
    try:
        result = subprocess.run([executable, *command[1:]], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{command[0]} failed: {error.__class__.__name__}"
    return (result.stdout.strip(), None) if result.returncode == 0 else (None, f"{command[0]} exited with status {result.returncode}")


def flatten(value, output):
    if not isinstance(value, dict):
        return
    for raw_name, child in value.items():
        name = norm(raw_name)
        if isinstance(child, dict):
            flatten(child, output)
        elif name in CONFIG_KEYS:
            output[name] = child


def load_config(path: Path | None):
    if path is None:
        return {}, fact(None, reason="no JSON run config was supplied")
    try:
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read JSON run config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ManifestError("run config JSON must contain an object")
    selected = {}
    flatten(raw, selected)
    return selected, fact({"path": str(path), "sha256": hfile(path)}, source="run_config")


def parse_log(path: Path | None):
    if path is None:
        return {}, fact(None, reason="no run log was supplied")
    if not path.exists():
        return {}, fact(None, reason=f"run log does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return {}, fact(None, reason=f"run log could not be read: {error.__class__.__name__}")
    for line in lines:
        if "stage=run_start" not in line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        result = {}
        for token in tokens:
            name, separator, value = token.partition("=")
            if separator and norm(name) in RUN_KEYS:
                result[norm(name)] = safe(value, name)
        return result, fact(True, source="run_log")
    return {}, fact(None, reason="run log has no parseable stage=run_start record")


def parse_tsv(path: Path | None):
    if path is None:
        return {}, {"comments": fact(None, reason="no result TSV was supplied"), "summary": {}}
    if not path.exists():
        return {}, {"comments": fact(None, reason=f"result TSV does not exist: {path}"), "summary": {}}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return {}, {"comments": fact(None, reason=f"result TSV could not be read: {error.__class__.__name__}"), "summary": {}}
    comments, header, rows, malformed = {}, None, 0, 0
    counters = {name: Counter() for name in ("planning_outcome", "bounded_status", "parity_status", "runtime_outcome")}
    csv.field_size_limit(sys.maxsize)
    for line in lines:
        if line.startswith("#"):
            name, separator, value = line[1:].partition("=")
            if separator and norm(name) in TSV_KEYS:
                comments[norm(name)] = safe(value, name)
            continue
        if not line.strip():
            continue
        try:
            fields = next(csv.reader([line], delimiter="\t"))
        except csv.Error:
            malformed += 1
            continue
        if header is None:
            header = fields
            continue
        if len(fields) != len(header):
            malformed += 1
            continue
        rows += 1
        row = dict(zip(header, fields))
        for name, counter in counters.items():
            if row.get(name):
                counter[row[name]] += 1
    summary = {"row_count": fact(rows, source="result_tsv"), "malformed_rows": malformed,
               "column_count": fact(len(header or []), source="result_tsv"),
               "column_names_sha256": fact(hashlib.sha256("\t".join(header or []).encode()).hexdigest(), source="result_tsv"),
               "status_counts": {name: dict(sorted(counter.items())) for name, counter in counters.items()}}
    return comments, {"comments": fact(True, source="result_tsv"), "summary": summary}


class Inputs:
    def __init__(self, config, tsv, log, environment):
        self.sources = (("run_config", {norm(k): v for k, v in config.items()}), ("result_tsv", tsv),
                        ("run_log", log), ("environment", {norm(k): v for k, v in environment.items()}))

    def lookup(self, names):
        names = {norm(name) for name in names}
        for source, values in self.sources:
            for name, value in values.items():
                if norm(name) in names and value not in (None, ""):
                    return value, source
        return None, None

    def field(self, names, parser, reason):
        value, source = self.lookup(names)
        if source is None:
            return fact(None, reason=reason)
        try:
            value = parser(value)
        except (TypeError, ValueError):
            return fact(None, reason=f"{names[0]} from {source} is invalid")
        return fact(safe(value, names[0]), source=source)


def git(path: Path, args):
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else None


def git_snapshot(path: Path, repo: Path, label: str):
    missing = f"{label} source path is absent"
    if not path.exists():
        return {"path": shown(path, repo), "sha": fact(None, reason=missing), "dirty": fact(None, reason=missing), "diff_sha256": fact(None, reason=missing)}
    revision, status = git(path, ["rev-parse", "HEAD"]), git(path, ["status", "--porcelain=v1", "--untracked-files=all"])
    tracked, untracked = git(path, ["diff", "--binary", "HEAD", "--"]) or "", git(path, ["ls-files", "--others", "--exclude-standard", "-z"]) or ""
    parts = [tracked.encode()]
    for relative in filter(None, untracked.split("\0")):
        candidate = path / relative
        if candidate.is_file():
            parts.extend((relative.encode(), hfile(candidate).encode()))
    diff = hashlib.sha256(b"\0".join(parts)).hexdigest() if status is not None else None
    no_git = f"{label} is not a Git worktree"
    snapshot = {"path": shown(path, repo), "sha": fact(revision.strip() if revision else None, "git" if revision else None, None if revision else no_git),
            "dirty": fact(bool(status) if status is not None else None, "git" if status is not None else None, None if status is not None else no_git),
            "diff_sha256": fact(diff, "git" if diff else None, None if diff else no_git)}
    if path == repo:
        rust_tracked = git(path, ["diff", "--binary", "HEAD", "--", ":(glob)**/*.rs"])
        if rust_tracked is None:
            snapshot["rust_diff_sha256"] = fact(None, reason="Rust source diff is unavailable")
        else:
            rust_parts = [rust_tracked.encode()]
            for relative in filter(None, untracked.split("\0")):
                candidate = path / relative
                if candidate.suffix == ".rs" and candidate.is_file():
                    rust_parts.extend((relative.encode(), hfile(candidate).encode()))
            snapshot["rust_diff_sha256"] = fact(hashlib.sha256(b"\0".join(rust_parts)).hexdigest(), "git")
    return snapshot


def gpu_info():
    output, reason = run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"])
    gpus, driver = [], None
    for row in csv.reader((output or "").splitlines(), skipinitialspace=True):
        if len(row) < 5:
            continue
        try:
            memory = pint(row[4])
        except ValueError:
            memory = None
        gpus.append({"index": row[0].strip(), "model": row[1].strip(), "uuid": row[2].strip(), "driver_version": row[3].strip(), "memory_total_mib": memory})
    if gpus:
        driver = gpus[0]["driver_version"]
    cuda, cuda_reason = run(["nvcc", "--version"])
    release = re.search(r"release\s+([0-9.]+)", cuda or "", re.I)
    return {"driver_version": fact(driver, "nvidia-smi" if driver else None, None if driver else (reason or "nvidia-smi returned no GPU")),
            "cuda_version": fact(release.group(1) if release else None, "nvcc" if release else None, None if release else (cuda_reason or "nvcc output did not contain a CUDA release")), "gpus": gpus}


def rust_info():
    rustc, rustc_reason = run(["rustc", "-Vv"])
    fields = {}
    for line in (rustc or "").splitlines():
        name, separator, value = line.partition(":")
        if separator and norm(name) in {"rustc_version", "host", "commit_hash", "commit_date"}:
            fields[norm(name)] = value.strip()
    cargo, cargo_reason = run(["cargo", "-V"])
    toolchain, toolchain_reason = run(["rustup", "show", "active-toolchain"])
    return {"rustc": fact(fields or None, "rustc" if fields else None, None if fields else (rustc_reason or "rustc -Vv returned no fields")),
            "cargo": fact(cargo, "cargo" if cargo else None, None if cargo else cargo_reason),
            "active_toolchain": fact(toolchain, "rustup" if toolchain else None, None if toolchain else toolchain_reason)}


def direct(values, names, parser, reason):
    for name in names:
        value = values.get(norm(name))
        if value not in (None, ""):
            try:
                return fact(parser(value), "run_log")
            except (TypeError, ValueError):
                return fact(None, reason=f"{name} in run log is invalid")
    return fact(None, reason=reason)


def missing(value, path="", result=None):
    result = [] if result is None else result
    if isinstance(value, dict):
        if {"value", "reason"} <= set(value) and value["value"] is None:
            result.append({"field": path, "reason": str(value["reason"])})
        else:
            for name, child in value.items():
                if name != "unavailable_fields":
                    missing(child, f"{path}.{name}" if path else name, result)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            missing(child, f"{path}[{index}]", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--result-tsv", type=Path)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--native-artifact", action="append", default=[])
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--loaded-libraries",
        "--loaded-library-snapshot",
        "--native-library-snapshot",
        dest="loaded_libraries",
        type=Path,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    config_path = args.run_config.absolute() if args.run_config else None
    config, config_input = load_config(config_path)
    if config_path is not None and config_input.get("value") is not None:
        config_input["value"]["path"] = shown(config_path, repo)
    log_path = resolve(args.run_log, repo) if args.run_log else None
    tsv_path = resolve(args.result_tsv, repo) if args.result_tsv else None
    log_values, run_record = parse_log(log_path)
    tsv_values, tsv = parse_tsv(tsv_path)
    environment = {name: os.environ[name] for name in ENV_NAMES if name in os.environ}
    inputs = Inputs(config, tsv_values, log_values, environment)
    native_raw = args.native_root or config.get("native_root") or environment.get("ALGEON_NATIVE_DIR") or "target/native"
    native_source = "argument" if args.native_root else "run_config" if config.get("native_root") else "environment" if environment.get("ALGEON_NATIVE_DIR") else "default"
    native_root = resolve(native_raw, repo) if native_raw else None
    binary_raw = args.binary or config.get("binary")
    binary_source = "argument" if args.binary else "run_config" if binary_raw else "none"
    binary = resolve(binary_raw, repo) if binary_raw else None
    loaded_libraries_raw = args.loaded_libraries or config.get("loaded_libraries_path") or config.get("native_library_snapshot") or config.get("runtime_library_snapshot")
    loaded_libraries_source = "argument" if args.loaded_libraries else "run_config" if loaded_libraries_raw else "none"
    loaded_libraries_path = resolve(loaded_libraries_raw, repo) if loaded_libraries_raw else None
    loaded_libraries = loaded_library_snapshot(loaded_libraries_path, repo, binary, loaded_libraries_source)
    configured_artifacts = args.native_artifact or config.get("native_artifacts") or []
    if not isinstance(configured_artifacts, list) or any(not isinstance(item, (str, Path)) or not str(item).strip() for item in configured_artifacts):
        raise ManifestError("native_artifacts must be a non-empty list of paths")
    artifacts = list(configured_artifacts)
    if artifacts:
        artifact_paths = {str(item): resolve(item, repo) for item in artifacts}
    else:
        artifact_paths = {}
    configured_traces = config.get("trace_paths") or []
    if not isinstance(configured_traces, list) or any(not isinstance(item, (str, Path)) or not str(item).strip() for item in configured_traces):
        raise ManifestError("trace_paths must be a non-empty list of paths")
    trace_paths = [resolve(item, repo) for item in configured_traces]
    configured_evidence = config.get("evidence_paths") or []
    if not isinstance(configured_evidence, list) or any(not isinstance(item, str) or not item.strip() for item in configured_evidence):
        raise ManifestError("evidence_paths must be a list of non-empty paths")
    if len(configured_evidence) != len(set(configured_evidence)):
        raise ManifestError("evidence_paths contains duplicate paths")
    evidence_paths = [resolve(item, repo) for item in configured_evidence]
    f = inputs.field
    layout = {
        "dataset_root": f(("dataset_root", "tpch_dataset_root", "ALGEON_PARQUET_DIR"), str, "dataset root was not supplied or recorded"),
        "scale_factor": f(("scale_factor", "dataset_scale_factor", "tpch_dataset_scale_factor", "ALGEON_TPCH_SCALE_FACTOR"), str, "TPC-H scale factor was not supplied or recorded"),
        "dataset_mode": f(("dataset_mode", "tpch_benchmark_dataset_mode", "ALGEON_TPCH_BENCH_DATASET_MODE"), str, "TPC-H dataset mode was not supplied or recorded"),
        "provenance": f(("dataset_provenance", "tpch_dataset_provenance"), str, "dataset provenance was not supplied or recorded"),
        "table_count": f(("dataset_table_count", "tpch_dataset_table_count"), pint, "dataset table count was not recorded"), "file_count": f(("dataset_file_count", "tpch_dataset_file_count"), pint, "dataset file count was not recorded"),
        "total_rows": f(("dataset_total_rows", "tpch_dataset_total_rows"), pint, "dataset total rows were not recorded"), "total_bytes": f(("dataset_total_bytes", "tpch_dataset_total_bytes"), pint, "dataset total bytes were not recorded"),
        "metadata_fingerprint": f(("dataset_metadata_fingerprint", "tpch_dataset_metadata_fingerprint"), str, "dataset metadata fingerprint was not recorded"), "content_fingerprint": f(("dataset_content_fingerprint", "tpch_dataset_content_fingerprint"), str, "dataset content fingerprint was not recorded"),
        "table_file_counts": f(("dataset_table_file_counts", "tpch_dataset_table_file_counts"), pmap, "per-table file counts were not recorded"), "table_row_counts": f(("dataset_table_row_counts", "tpch_dataset_table_row_counts"), pmap, "per-table row counts were not recorded"),
        "table_byte_sizes": f(("dataset_table_byte_sizes", "tpch_dataset_table_byte_sizes"), pmap, "per-table byte sizes were not recorded"), "table_row_group_counts": f(("dataset_table_row_group_counts", "tpch_dataset_table_row_group_counts"), pmap, "per-table row-group counts were not recorded"),
    }
    if layout["dataset_root"]["value"] is not None:
        layout["dataset_root"]["value"] = shown(resolve(layout["dataset_root"]["value"], repo), repo)
    benchmark = {
        "query_ids": f(("query_ids", "tpch_benchmark_query_ids", "ALGEON_TPCH_BENCH_QUERY_IDS"), plist, "TPC-H query IDs were not supplied or recorded"),
        "warmups": f(("warmups", "tpch_benchmark_warmups", "ALGEON_TPCH_BENCH_WARMUPS"), pint, "warmup count was not supplied or recorded"), "iterations": f(("iterations", "tpch_benchmark_iterations", "ALGEON_TPCH_BENCH_ITERATIONS"), pint, "iteration count was not supplied or recorded"),
        "build_profile": f(("build_profile", "profile", "rust_profile", "CARGO_PROFILE"), str, "build profile was not supplied or recorded"), "native_profile": f(("native_profile", "tpch_benchmark_native_profile_name", "ALGEON_TPCH_BENCH_NATIVE_PROFILE"), str, "native profile was not supplied or recorded"), "native_profile_source": f(("profile_source", "tpch_benchmark_native_profile_source"), str, "native profile source was not recorded"),
        "query_cap_bytes": f(("query_cap_bytes", "native_device_budget_bytes", "device_budget_bytes", "tpch_benchmark_native_device_budget_bytes", "ALGEON_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES"), pint, "query cap bytes were not explicitly supplied or recorded"), "cache_share_bytes": f(("cache_share_bytes", "device_cache_share_bytes"), pint, "no explicit cache share was supplied; TPCH has no cache-share environment variable"),
        "source_chunk_bytes": f(("source_chunk_bytes", "max_source_chunk_bytes", "tpch_benchmark_native_max_source_chunk_bytes", "ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES"), pint, "source chunk bytes were not supplied or recorded"), "row_groups_per_chunk": f(("row_groups_per_chunk", "max_row_groups_per_chunk", "tpch_benchmark_native_max_row_groups_per_chunk", "ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK"), pint, "row-group limit was not supplied or recorded"),
        "max_source_reads_in_flight": f(("max_source_reads_in_flight", "tpch_benchmark_native_max_source_reads_in_flight", "ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT"), pint, "source-read concurrency was not supplied or recorded"), "operator_width": f(("operator_width",), pint, "existing TPCH benchmark does not expose operator width and no explicit value was supplied"),
        "cpu_threads": f(("cpu_threads", "configured_cpu_threads", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"), pint, "configured CPU thread count was not supplied or recorded"), "cpu_logical_threads": f(("cpu_logical_threads", "cpu_logical"), pint, "run did not report logical CPU count"), "cpu_profile": f(("cpu_profile", "ALGEON_TPCH_CPU_PROFILE"), str, "CPU profile was not supplied or recorded"),
        "source_cache_state": f(("source_cache_state",), str, "source cache state is not emitted by TPCH provenance and was not supplied"), "source_chunk_planning_mode": f(("source_chunk_planning_mode", "tpch_benchmark_native_source_chunk_planning_mode", "ALGEON_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE"), str, "source-chunk planning mode was not supplied or recorded"),
    }
    manifest = {
        "kind": "algeon-gpu-query-engine-tpch-baseline",
        "source": {"root": git_snapshot(repo, repo, "root"), "native": {name: git_snapshot(repo / relative, repo, name) for name, relative in NATIVE_SOURCES.items()}},
        "hardware": gpu_info(), "run_start_hardware": {"driver_version": direct(log_values, ("driver_version",), str, "run log did not report driver version"), "gpu_model": direct(log_values, ("gpu0_model",), str, "run log did not report GPU model"), "gpu_uuid": direct(log_values, ("gpu0_uuid",), str, "run log did not report GPU UUID"), "gpu_count": direct(log_values, ("gpu_count",), pint, "run log did not report GPU count")},
        "rust": rust_info(), "benchmark": benchmark, "source_layout": layout,
        "native_artifacts": {"root": fingerprint(native_root, repo, native_source, "native artifact root"), "files": {name: fingerprint(path, repo, native_source, f"native artifact {name}") for name, path in artifact_paths.items()}},
        "binary": fingerprint(binary, repo, binary_source, "benchmark binary"),
        "loaded_libraries": loaded_libraries,
        "traces": [fingerprint(path, repo, "run_config", "trace") for path in trace_paths],
        "evidence_artifacts": [fingerprint(path, repo, "run_config", "evidence artifact") for path in evidence_paths],
        "result": {"file": fingerprint(tsv_path, repo, "argument", "result TSV"), **tsv["summary"]},
        "inputs": {"run_config": config_input, "run_log": fingerprint(log_path, repo, "argument", "run log"), "run_start_record": run_record, "result_tsv": tsv["comments"], "observed_environment": {name: safe(value, name) for name, value in sorted(environment.items())}},
    }
    manifest["unavailable_fields"] = sorted(missing(manifest), key=lambda item: item["field"])
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    if output.exists() and not args.force:
        raise ManifestError(f"output exists; pass --force to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
