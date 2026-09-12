#!/usr/bin/env python3
"""Run the isolated query-service goodput controls for the local GPU baseline.

The helper intentionally drives the existing binary and frozen calibration
stores.  It does not build the binary, change source files, or infer a
successful native attempt from the process exit status alone.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "target/bench/parallel-a-2026-09-19"
GOODPUT_BINARY = ROOT / "target/debug/query_service_goodput"
GPU_LOCK = Path("/tmp/cudf-gpu.lock")
OUTPUT_ROOT = ARTIFACT_ROOT / "goodput-ten"
CAP_BYTES = 16_086_990_848
SOURCE_CHUNK_BYTES = 4_294_967_296
IDENTITY_VERSION = 1
IDENTITY_FILE = "run-identity.json"
LOADED_LIBRARIES_FILE = "loaded-native-libraries.json"
EXPECTED_WARMUP_ATTEMPTS = 1
EXPECTED_WINDOW_ATTEMPTS = 12
EXPECTED_TOTAL_ATTEMPTS = EXPECTED_WARMUP_ATTEMPTS + EXPECTED_WINDOW_ATTEMPTS
IDENTITY_DYNAMIC_SETTINGS = {
    "ALGEON_GOODPUT_GROUP",
    "ALGEON_GOODPUT_OUT_DIR",
    "ALGEON_GOODPUT_RUN_ID",
    "ALGEON_GOODPUT_SOURCE_IDENTITY",
}
INHERITED_IDENTITY_SETTINGS = {
    "ALGEON_NATIVE_DIR",
    "ALGEON_PARQUET_DIR",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "CUDAToolkit_ROOT",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "RAYON_NUM_THREADS",
    "RUST_BACKTRACE",
    "RUST_LOG",
    "TOKIO_WORKER_THREADS",
}
FAILURE_SUMMARY_COLUMNS = (
    "slo_missed",
    "output_mismatch",
    "rejected_queue_full",
    "rejected_queue_timeout",
    "failed_query_cap",
    "failed_physical_oom",
    "failed_unknown_allocation",
    "cancelled",
    "failed_unsupported",
    "failed_other",
    "non_native_dispositions",
    "unattributed_allocations",
    "harness_timeouts",
    "reports_unpublished",
)

CONTROLS = (
    ("independent_union", "adhoc.independent_union", "branch_slo.tsv", "tpch"),
    ("repeated_cte", "adhoc.repeated_cte", "branch_slo.tsv", "tpch"),
    ("variable_width_group", "adhoc.variable_width_group", "variable_width_slo.tsv", "tpch"),
    ("global_sort", "adhoc.global_sort", "goodput_slo.tsv", "tpch"),
    ("pagerank", "graph.pagerank", "graph_slo.tsv", "graph"),
    ("q39", "tpcds.q39", "tpcds_goodput_slo.tsv", "tpcds"),
    ("q78", "tpcds.q78", "tpcds_goodput_slo.tsv", "tpcds"),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"required {label} is missing: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        fail(f"required {label} is missing: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_identity(path: Path) -> dict[str, object]:
    """Hash a file or a directory's complete file set and contents."""

    if path.is_file():
        result = {
            "kind": "file",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        return result
    if not path.is_dir():
        fail(f"identity path is missing: {path}")

    entries = []
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": child.stat().st_size,
                "sha256": sha256_file(child),
            }
        )
    if not entries:
        fail(f"identity directory is empty: {path}")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = {
        "kind": "directory",
        "path": str(path),
        "file_count": len(entries),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": entries,
    }
    return result


def canonical_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def settings_identity(values: dict[str, str]) -> dict[str, str]:
    selected = {
        key: value
        for key, value in values.items()
        if key.startswith("ALGEON_GOODPUT_") and key not in IDENTITY_DYNAMIC_SETTINGS
    }
    selected.update({key: values[key] for key in INHERITED_IDENTITY_SETTINGS if key in values})
    return dict(sorted(selected.items()))


def control_identity(
    run_id: str,
    item_id: str,
    slo_path: Path,
    family: str,
    run_environment: dict[str, str],
) -> tuple[dict[str, object], str]:
    """Build the identity that authorizes reuse of one control's output."""

    if not GOODPUT_BINARY.is_file():
        fail(f"query_service_goodput binary is missing: {GOODPUT_BINARY}")
    dataset_path = {
        "tpch": ROOT / "fixture/tpc-h/data/sf1/parquet",
        "tpcds": ROOT / "fixture/tpc-ds/data/sf1/parquet",
        "graph": ARTIFACT_ROOT / "karate-edges.parquet",
    }.get(family)
    if dataset_path is None:
        fail(f"unknown control family {family}")
    document: dict[str, object] = {
        "identity_version": IDENTITY_VERSION,
        "control": {"run_id": run_id, "item_id": item_id, "family": family},
        "binary": {
            "path": canonical_path(GOODPUT_BINARY),
            "sha256": sha256_file(GOODPUT_BINARY),
        },
        "dataset": path_identity(dataset_path),
        "calibration": {
            "path": canonical_path(slo_path),
            "sha256": sha256_file(slo_path),
        },
        "settings": settings_identity(run_environment),
        "runner": {
            "path": canonical_path(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity_sha256 = hashlib.sha256(encoded).hexdigest()
    document["identity_sha256"] = identity_sha256
    return document, identity_sha256


def effective_run_id(run_id: str, identity_sha256: str) -> str:
    return f"{run_id}-{identity_sha256[:16]}"


def write_identity(run_dir: Path, document: dict[str, object]) -> None:
    identity_path = run_dir / IDENTITY_FILE
    if identity_path.exists():
        try:
            with identity_path.open(encoding="utf-8") as stream:
                existing = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            fail(f"cannot read existing run identity {identity_path}: {error}")
        if existing != document:
            fail(f"run identity collision at {run_dir}; refusing to overwrite existing evidence")
        return
    if any(path.is_file() for path in run_dir.iterdir()):
        fail(f"run output exists without {IDENTITY_FILE}: {run_dir}; refusing to overwrite existing evidence")
    temporary = run_dir / f".{IDENTITY_FILE}.tmp"
    try:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(identity_path)
    except OSError as error:
        fail(f"cannot write run identity {identity_path}: {error}")


def calibration_has_item(path: Path, item_id: str) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.reader((line for line in stream if not line.startswith("#")), delimiter="\t")
        header = next(rows, None)
        if header is None or "item_id" not in header:
            fail(f"calibration has no item_id column: {path}")
        item_column = header.index("item_id")
        return any(len(row) > item_column and row[item_column] == item_id for row in rows)


def preflight() -> None:
    require_file(GOODPUT_BINARY, "query_service_goodput binary")
    if not os.access(GOODPUT_BINARY, os.X_OK):
        fail(f"query_service_goodput binary is not executable: {GOODPUT_BINARY}")
    flock = shutil.which("flock")
    if flock is None:
        fail("required flock executable is unavailable")
    if not GPU_LOCK.parent.exists():
        fail(f"GPU lock parent is missing: {GPU_LOCK.parent}")

    require_directory(ROOT / "fixture/tpc-h/data/sf1/parquet", "TPC-H SF1 fixture")
    require_directory(ROOT / "fixture/tpc-ds/data/sf1/parquet", "TPC-DS SF1 fixture")
    require_file(ARTIFACT_ROOT / "karate-edges.parquet", "karate edge fixture")
    for _, item_id, slo_name, _ in CONTROLS:
        slo_path = ARTIFACT_ROOT / slo_name
        require_file(slo_path, f"frozen calibration for {item_id}")
        if not calibration_has_item(slo_path, item_id):
            fail(f"frozen calibration {slo_path} has no row for {item_id}")


def environment(item_id: str, slo_path: Path, run_id: str, family: str) -> dict[str, str]:
    # Do not let an inherited ALGEON_GOODPUT_* setting silently change the
    # frozen-cap, arrival, or dataset contract of an isolated control.
    result = {key: value for key, value in os.environ.items() if not key.startswith("ALGEON_GOODPUT_")}
    result.update(
        {
            "ALGEON_GOODPUT_MODE": "measure",
            "ALGEON_GOODPUT_MIX": item_id,
            "ALGEON_GOODPUT_ARRIVAL": "wave",
            "ALGEON_GOODPUT_WAVE_SIZE": "1",
            "ALGEON_GOODPUT_RATE_QPS": "1",
            "ALGEON_GOODPUT_WARMUP_SECONDS": "2",
            "ALGEON_GOODPUT_WINDOW_SECONDS": "12",
            "ALGEON_GOODPUT_SEED": "20260910",
            "ALGEON_GOODPUT_DEVICE_ORDINALS": "0",
            "ALGEON_GOODPUT_DEVICE_CAPACITY_BYTES": str(CAP_BYTES),
            "ALGEON_GOODPUT_QUERY_CAP_BYTES": str(CAP_BYTES),
            "ALGEON_GOODPUT_MAX_ACTIVE_ATTEMPTS": "1",
            "ALGEON_GOODPUT_MAX_QUEUED_ATTEMPTS": "32",
            "ALGEON_GOODPUT_MAX_QUEUE_WAIT_MS": "300000",
            "ALGEON_GOODPUT_MAX_SOURCE_CHUNK_BYTES": str(SOURCE_CHUNK_BYTES),
            "ALGEON_GOODPUT_MAX_ROW_GROUPS_PER_CHUNK": "1024",
            "ALGEON_GOODPUT_SLO_PATH": str(slo_path),
            "ALGEON_GOODPUT_SLO_MULTIPLIER": "3.0",
            "ALGEON_GOODPUT_POOL_STATE": "hot",
            "ALGEON_GOODPUT_QUERY_TIMEOUT_MS": "900000",
            "ALGEON_GOODPUT_FINGERPRINT_MAX_ROWS": "1000000",
            "ALGEON_GOODPUT_SAMPLE_INTERVAL_MS": "250",
            "ALGEON_GOODPUT_OUT_DIR": str(OUTPUT_ROOT),
            "ALGEON_GOODPUT_RUN_ID": run_id,
            "ALGEON_GOODPUT_GROUP": run_id,
            "ALGEON_GOODPUT_SOURCE_IDENTITY": run_id,
            "TOKIO_WORKER_THREADS": "32",
            "RAYON_NUM_THREADS": "32",
        }
    )
    if family == "tpch":
        result.update(
            {
                "ALGEON_GOODPUT_TPCH_DIR": str(ROOT / "fixture/tpc-h/data/sf1/parquet"),
                "ALGEON_GOODPUT_TPCH_SCALE": "sf1",
            }
        )
    elif family == "tpcds":
        result.update(
            {
                "ALGEON_GOODPUT_TPCDS_DIR": str(ROOT / "fixture/tpc-ds/data/sf1/parquet"),
                "ALGEON_GOODPUT_TPCDS_SCALE": "sf1",
            }
        )
    elif family == "graph":
        result.update(
            {
                "ALGEON_GOODPUT_GRAPH_EDGES": str(ARTIFACT_ROOT / "karate-edges.parquet"),
                "ALGEON_GOODPUT_GRAPH_DATASET": "karate",
            }
        )
    else:
        fail(f"unknown control family {family}")
    return result


def integer_field(row: dict[str, str], name: str) -> int | None:
    try:
        return int(row.get(name, ""))
    except (TypeError, ValueError):
        return None


def valid_loaded_libraries(run_dir: Path, identity_document: dict[str, object]) -> bool:
    snapshot_path = run_dir / LOADED_LIBRARIES_FILE
    try:
        with snapshot_path.open(encoding="utf-8") as stream:
            snapshot = json.load(stream)
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 2 or snapshot.get("source") != "/proc/self/maps":
            return False
        binary = snapshot.get("binary")
        expected_binary = identity_document.get("binary")
        if not isinstance(binary, dict) or not isinstance(expected_binary, dict):
            return False
        binary_path = Path(binary["path"])
        if binary_path != GOODPUT_BINARY.resolve(strict=True) or binary_path != binary_path.resolve(strict=True):
            return False
        binary_stat = binary_path.stat()
        if (
            binary["sha256"] != expected_binary["sha256"]
            or binary["sha256"] != sha256_file(binary_path)
            or binary["size_bytes"] != binary_stat.st_size
            or binary["device"] != binary_stat.st_dev
            or binary["inode"] != binary_stat.st_ino
        ):
            return False
        libraries = snapshot.get("libraries")
        if not isinstance(libraries, list) or not libraries:
            return False
        seen = set()
        names = set()
        for library in libraries:
            if not isinstance(library, dict):
                return False
            path = Path(library["path"])
            if not path.is_absolute() or path != path.resolve(strict=True) or path in seen:
                return False
            if not Path(library["mapped_path"]).is_absolute() or not isinstance(library["mapped_device"], str):
                return False
            stat = path.stat()
            if (
                library["size_bytes"] != stat.st_size
                or library["device"] != stat.st_dev
                or library["inode"] != stat.st_ino
                or library["sha256"] != sha256_file(path)
            ):
                return False
            seen.add(path)
            names.add(path.name)
        return all(name in names or any(actual.startswith(name + ".") for actual in names) for name in ("libcudf_rust.so", "libcudf.so"))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def valid_output(
    run_id: str,
    item_id: str,
    identity_document: dict[str, object],
) -> int | None:
    run_dir = OUTPUT_ROOT / run_id
    identity_path = run_dir / IDENTITY_FILE
    summary_path = run_dir / "summary.tsv"
    attempts_path = run_dir / "attempts.tsv"
    if not identity_path.is_file() or not summary_path.is_file() or not attempts_path.is_file():
        return None
    try:
        with identity_path.open(encoding="utf-8") as stream:
            if json.load(stream) != identity_document:
                return None
    except (OSError, json.JSONDecodeError):
        return None
    if not valid_loaded_libraries(run_dir, identity_document):
        return None

    try:
        with summary_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except (OSError, csv.Error):
        return None
    if len(rows) != 1:
        return None
    summary = rows[0]
    required_summary = {
        "run_id": run_id,
        "group": run_id,
        "mix": f"{item_id}@1",
        "arrival": "wave",
        "warmup_seconds": "2.000",
        "window_seconds": "12.000",
        "max_active_attempts": "1",
        "device_capacity_bytes": str(CAP_BYTES),
        "query_cap_bytes": str(CAP_BYTES),
        "source_identity": identity_document["identity_sha256"],
    }
    if any(summary.get(key) != value for key, value in required_summary.items()):
        return None
    if summary.get("memory_failure_ratio") not in {"0.000000", "0"}:
        return None
    expected_counts = {
        "offered": EXPECTED_WINDOW_ATTEMPTS,
        "completed": EXPECTED_WINDOW_ATTEMPTS,
        "succeeded": EXPECTED_WINDOW_ATTEMPTS,
        "slo_met": EXPECTED_WINDOW_ATTEMPTS,
        "attempts_begun": EXPECTED_TOTAL_ATTEMPTS,
        "reports_ready": EXPECTED_TOTAL_ATTEMPTS,
    }
    if any(integer_field(summary, key) != value for key, value in expected_counts.items()):
        return None
    if any(integer_field(summary, key) != 0 for key in FAILURE_SUMMARY_COLUMNS):
        return None

    try:
        with attempts_path.open(newline="", encoding="utf-8") as stream:
            attempts_reader = csv.DictReader(stream, delimiter="\t")
            attempts = list(attempts_reader)
    except (OSError, csv.Error):
        return None
    if len(attempts) != EXPECTED_TOTAL_ATTEMPTS:
        return None
    if any(integer_field(summary, key) != len(attempts) for key in ("attempts_begun", "reports_ready")):
        return None

    sequences = []
    for row in attempts:
        sequence = integer_field(row, "sequence")
        if sequence is None:
            return None
        sequences.append(sequence)
        if (
            row.get("item_id") != item_id
            or row.get("disposition") != "native"
            or row.get("terminal_outcome") != "succeeded"
            or row.get("outcome_class") != "succeeded"
            or row.get("fingerprint_match") != "true"
            or row.get("harness_timeout") != "false"
            or row.get("cleanup_disposition") != "completed"
            or row.get("capacity_disposition") != "released"
            or row.get("error_message") not in (None, "none", "")
            or row.get("failure_code") not in (None, "none", "")
            or row.get("failure_status") not in (None, "none", "")
            or row.get("failure_operation") not in (None, "none", "")
            or row.get("failure_reason") not in (None, "none", "")
        ):
            return None
    if sorted(sequences) != list(range(EXPECTED_TOTAL_ATTEMPTS)):
        return None
    warmups = [row for row in attempts if row.get("phase") == "warmup"]
    windows = [row for row in attempts if row.get("phase") == "window"]
    if len(warmups) != EXPECTED_WARMUP_ATTEMPTS or len(windows) != EXPECTED_WINDOW_ATTEMPTS:
        return None
    if any(row.get("slo_met") != "true" for row in windows):
        return None
    if any(row.get("output_fingerprint") in (None, "none", "") for row in attempts):
        return None
    return len(windows)


def run_control(run_id: str, item_id: str, slo_name: str, family: str) -> int:
    slo_path = ARTIFACT_ROOT / slo_name
    base_environment = environment(item_id, slo_path, run_id, family)
    identity_document, identity_sha256 = control_identity(run_id, item_id, slo_path, family, base_environment)
    actual_run_id = effective_run_id(run_id, identity_sha256)
    existing = valid_output(actual_run_id, item_id, identity_document)
    if existing is not None:
        print(f"[skip] {item_id}: {existing} valid native window attempts (identity={identity_sha256[:16]})")
        return existing

    run_dir = OUTPUT_ROOT / actual_run_id
    if run_dir.exists() and any(path.name != IDENTITY_FILE for path in run_dir.iterdir()):
        fail(f"{item_id} has incomplete or invalid evidence for identity {identity_sha256[:16]} at {run_dir}; refusing to overwrite it")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_identity(run_dir, identity_document)
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{actual_run_id}.stdout"
    stderr_path = log_dir / f"{actual_run_id}.stderr"
    if stdout_path.exists() or stderr_path.exists():
        fail(f"{item_id} has existing logs for identity {identity_sha256[:16]}; refusing to overwrite evidence")
    command = [str(shutil.which("flock") or "flock"), str(GPU_LOCK), str(GOODPUT_BINARY)]
    run_environment = environment(item_id, slo_path, actual_run_id, family)
    run_environment["ALGEON_GOODPUT_SOURCE_IDENTITY"] = identity_sha256
    print(f"[run] {item_id} -> {run_dir} (identity={identity_sha256[:16]})")
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=run_environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
    except OSError as error:
        fail(f"{item_id} could not start; see {stderr_path}: {error}")
    if completed.returncode != 0:
        fail(f"{item_id} exited with status {completed.returncode}; see {stderr_path}")
    successes = valid_output(actual_run_id, item_id, identity_document)
    if successes is None:
        fail(f"{item_id} completed without {EXPECTED_WINDOW_ATTEMPTS} clean native succeeded fingerprint-matched window attempts")
    print(f"[ok] {item_id}: {successes} valid native window attempts")
    return successes


def main() -> int:
    try:
        preflight()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for control in CONTROLS:
            run_control(*control)
    except (OSError, RuntimeError, ValueError, csv.Error) as error:
        print(f"run_gpu_query_engine_controls: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
