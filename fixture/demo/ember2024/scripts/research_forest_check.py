#!/usr/bin/env python3
"""Build a CPU LightGBM reference for the EMBER2024 challenge split."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "fixture/demo/ember2024"
REPORT = ROOT / "docs/research/ember2024"
THRESHOLDS = (0.5, 0.9, 0.99, 0.999)
ERROR_QUANTILES = (0.5, 0.9, 0.99, 0.999)
CHALLENGE_ROWS = 6315
ID_OFFSETS = {"10": 0, "11": 20_000_000, "12": 20_000_000}
INPUT_FILES = {
    "10": (Path("fixture/demo/ember2024/parquet/challenge.parquet"),),
    "11": (Path("target/ember2024-research/input_research_vectors_challenge.parquet"),),
    "12": (
        Path("target/ember2024-research/input_research_vectors_challenge.parquet"),
        Path("target/ember2024-research/input_attributions.parquet"),
        Path("target/ember2024-research/input_feature_groups.parquet"),
        Path("target/ember2024-research/input_metadata_challenge.parquet"),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_observations(stats_path: Path) -> dict:
    with gzip.open(stats_path, "rt") as source:
        lines = source.read().splitlines()
    elapsed_line = next((line for line in lines if line.startswith("# elapsed_ms=")), None)
    if elapsed_line is None:
        raise ValueError(f"missing total elapsed time in {stats_path.name}")
    elapsed_ms = float(elapsed_line.removeprefix("# elapsed_ms="))
    section = lines.index("[execution]")
    headers = lines[section + 1].split("\t")
    row = lines[section + 2].split("\t")
    if len(headers) != len(row):
        raise ValueError(f"cannot parse execution stats row in {stats_path.name}")
    values = dict(zip(headers, row, strict=True))
    task_runtimes = []
    runtime_text = values.get("task_runtime_rows", "")
    pattern = re.compile(r"compiled_task_id=([^,]+),operator_kind=([^,]+),elapsed_ns=(\d+)")
    totals_ns: dict[str, int] = {}
    for task_id, kind, elapsed_ns in pattern.findall(runtime_text):
        elapsed = int(elapsed_ns)
        totals_ns[kind] = totals_ns.get(kind, 0) + elapsed
        task_runtimes.append({"task_id": task_id, "operator_kind": kind, "elapsed_ms": elapsed / 1_000_000})
    return {
        "total_elapsed_ms": elapsed_ms,
        "host_to_device_uploaded_bytes": int(values["host_to_device_uploaded_bytes"]),
        "compiled_task_runtime_ms_by_operator_kind": {
            kind: elapsed_ns / 1_000_000 for kind, elapsed_ns in sorted(totals_ns.items())
        },
        "compiled_task_runtime_rows": task_runtimes,
    }


def result_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["id", "prediction"]).sort_by([("id", "ascending")])
    return table.column("id").to_numpy(), table.column("prediction").to_numpy()


def build_performance_shape() -> tuple[dict, bool]:
    data_dir = ROOT / "target/ember2024-research"
    result_dir = REPORT / "results"
    all_runs: dict[str, dict] = {}
    variant_checks = []
    failed = False

    for suffix in ("", "_r2", "_r3"):
        names = {
            "10": f"10_forest_challenge_wide{suffix}",
            "11": f"11_forest_challenge_vectors{suffix}",
            "12": f"12_forest_review_explanations{suffix}",
        }
        output_rows = {}
        for shape, name in names.items():
            parquet_path = data_dir / f"{name}.parquet"
            json_path = result_dir / f"{name}.json"
            stats_path = result_dir / f"{name}.stats.txt.gz"
            if not parquet_path.is_file() or not json_path.is_file() or not stats_path.is_file():
                raise FileNotFoundError(f"missing output, timing, or stats for {name}")
            record = json.loads(json_path.read_text())
            if record.get("status") != "ok":
                raise ValueError(f"{name} did not complete successfully")
            ids, predictions = result_columns(parquet_path)
            offset = ID_OFFSETS[shape]
            normalized = ids.astype(np.int64, copy=False) - offset
            if (shape != "12" and ids.size != np.unique(ids).size) or np.any(normalized < 0) or np.any(normalized >= CHALLENGE_ROWS):
                raise ValueError(f"{name} contains duplicate ids or out-of-range source ordinals")
            if shape in ("10", "11") and not np.array_equal(normalized, np.arange(CHALLENGE_ROWS, dtype=np.int64)):
                raise ValueError(f"{name} ids do not cover source ordinals 0..{CHALLENGE_ROWS - 1}")
            if not np.isfinite(predictions).all():
                raise ValueError(f"{name} contains non-finite scores")
            run_inputs = []
            for relative_path in INPUT_FILES[shape]:
                input_path = ROOT / relative_path
                if not input_path.is_file():
                    raise FileNotFoundError(f"missing serialized query input {input_path}")
                run_inputs.append({"path": relative_path.as_posix(), "size_bytes": input_path.stat().st_size})
            run = {
                "query": name,
                "rows": int(ids.size),
                "unique_source_ids": int(np.unique(normalized).size),
                "expected_id_offset": offset,
                "id_offset_valid": bool(np.all(ids >= offset)),
                "normalized_source_ids_sorted": bool(np.all(normalized[1:] > normalized[:-1])),
                "elapsed_ms_from_json": float(record["elapsed_s"]) * 1000.0,
                "serialized_input_files": run_inputs,
                "serialized_input_total_bytes": sum(item["size_bytes"] for item in run_inputs),
                "execution_observations": execution_observations(stats_path),
            }
            all_runs[name] = run
            output_rows[shape] = (normalized, predictions)

        wide_ids, wide_scores = output_rows["10"]
        vector_ids, vector_scores = output_rows["11"]
        stack_ids, stack_scores = output_rows["12"]
        wide_vector_ids_match = np.array_equal(wide_ids, vector_ids)
        wide_vector_exact = wide_vector_ids_match and np.array_equal(wide_scores, vector_scores)
        vector_map = dict(zip(vector_ids.tolist(), vector_scores.tolist(), strict=True))
        stack_expected = np.asarray([vector_map.get(int(source_id), np.nan) for source_id in stack_ids], dtype=np.float64)
        stack_ids_present = bool(np.isfinite(stack_expected).all())
        stack_exact = stack_ids_present and np.array_equal(stack_scores, stack_expected)
        wide_vector_error = np.abs(wide_scores - vector_scores) if wide_vector_ids_match else np.asarray([], dtype=np.float64)
        stack_error = np.abs(stack_scores - stack_expected) if stack_ids_present else np.asarray([], dtype=np.float64)
        check = {
            "suffix": suffix or "base",
            "wide_vs_vector": {
                "normalized_ids_match": bool(wide_vector_ids_match),
                "exact_scores_match": bool(wide_vector_exact),
                "max_absolute_score_difference": float(wide_vector_error.max()) if wide_vector_error.size else None,
            },
            "stack_vs_vector_on_returned_ids": {
                "returned_ids_are_in_vector_result": stack_ids_present,
                "exact_scores_match": bool(stack_exact),
                "max_absolute_score_difference": float(stack_error.max()) if stack_error.size else None,
            },
        }
        variant_checks.append(check)
        failed |= not wide_vector_exact or not stack_exact

    by_shape = {}
    for shape, label in (("10", "wide"), ("11", "vector"), ("12", "stack")):
        matching_runs = [run for name, run in all_runs.items() if name.startswith(shape + "_")]
        times = [run["elapsed_ms_from_json"] for run in matching_runs]
        by_shape[label] = {
            "queries": [run["query"] for run in matching_runs],
            "elapsed_ms_from_json": times,
            "median_elapsed_ms_from_json": statistics.median(times),
            "elapsed_ms_from_stats": [
                run["execution_observations"]["total_elapsed_ms"] for run in matching_runs
            ],
            "median_elapsed_ms_from_stats": statistics.median(
                run["execution_observations"]["total_elapsed_ms"] for run in matching_runs
            ),
            "serialized_input_files": matching_runs[0]["serialized_input_files"],
            "serialized_input_total_bytes": matching_runs[0]["serialized_input_total_bytes"],
        }

    report = {
        "rows_scored_by_wide_and_vector_queries": CHALLENGE_ROWS,
        "id_offsets": {"10_wide": 0, "11_vector": 20_000_000, "12_stack": 20_000_000},
        "runs": all_runs,
        "shape_summary": by_shape,
        "score_alignment_checks": variant_checks,
        "timing_scope": (
            "JSON elapsed_s is end-to-end query time covering planning, Parquet input scanning, model import, and execution. "
            "The stats expose selected compiled-task spans and uploaded-byte counts; this report does not assign the unmeasured remainder to a phase."
        ),
        "interpretation_limits": [
            "No CPU-versus-GPU speedup is estimated from these query runs.",
            "Operator timings are observations and do not establish causal attribution for the total elapsed time.",
            "The stack query emits 30 filtered review rows, so its exact score check uses only those returned ids.",
        ],
    }
    (result_dir / "performance_shape.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report, failed


def gpu_comparison(path: Path, rows: int, cpu_scores: np.ndarray) -> tuple[dict, bool]:
    table = pq.read_table(path, columns=["id", "prediction"]).sort_by([("id", "ascending")])
    ids = table.column("id").to_numpy()
    predictions = table.column("prediction").to_numpy().astype(np.float64, copy=False)
    expected_ids = np.arange(rows, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        return {"status": "error", "error": "GPU output ids do not equal source_id values 0..rows-1"}, True
    if predictions.shape != cpu_scores.shape:
        return {
            "status": "error",
            "error": f"GPU prediction count {predictions.size} does not match CPU prediction count {cpu_scores.size}",
        }, True
    if not np.isfinite(predictions).all():
        return {"status": "error", "error": "GPU predictions contain non-finite values"}, True

    absolute_error = np.abs(predictions - cpu_scores)
    quantiles = np.quantile(absolute_error, ERROR_QUANTILES)
    agreement = {}
    for threshold in THRESHOLDS:
        matches = (predictions >= threshold) == (cpu_scores >= threshold)
        agreement[str(threshold)] = {
            "matching_rows": int(matches.sum()),
            "agreement_rate": float(matches.mean()),
        }
    within_tolerance = bool(np.allclose(predictions, cpu_scores, rtol=0.0, atol=1e-6))
    return {
        "status": "passed" if within_tolerance else "mismatch",
        "atol": 1e-6,
        "rtol": 0.0,
        "max_abs_error": float(absolute_error.max()),
        "absolute_error_quantiles": {
            name: float(value)
            for name, value in zip(("p50", "p90", "p99", "p99_9"), quantiles, strict=True)
        },
        "threshold_agreement": agreement,
    }, not within_tolerance


def main() -> int:
    model_path = FIXTURE / "models/EMBER2024_all.model"
    x_path = FIXTURE / "raw/X_challenge.dat"
    y_path = FIXTURE / "raw/y_challenge.dat"
    report_dir = REPORT / "results"
    report_dir.mkdir(parents=True, exist_ok=True)

    booster = lgb.Booster(model_file=str(model_path))
    dimension = int(booster.num_feature())
    if dimension <= 0:
        raise ValueError(f"model reports invalid feature count: {dimension}")

    label_bytes = y_path.stat().st_size
    if label_bytes % np.dtype(np.int32).itemsize:
        raise ValueError(f"{y_path} size is not aligned to int32 labels")
    rows = label_bytes // np.dtype(np.int32).itemsize
    expected_x_bytes = rows * dimension * np.dtype(np.float32).itemsize
    if x_path.stat().st_size != expected_x_bytes:
        raise ValueError(
            f"{x_path} size {x_path.stat().st_size} does not match {rows} rows x {dimension} float32 features"
        )

    labels = np.memmap(y_path, dtype=np.int32, mode="r", shape=(rows,))
    if not np.all(labels == 1):
        raise ValueError("challenge split labels must all be 1")
    features = np.memmap(x_path, dtype=np.float32, mode="r", shape=(rows, dimension))
    scores = np.asarray(booster.predict(features, num_threads=8), dtype=np.float64)
    if scores.shape != (rows,) or not np.isfinite(scores).all():
        raise ValueError(f"model returned invalid scores with shape {scores.shape}")

    recall = {}
    for threshold in THRESHOLDS:
        detected = int(np.count_nonzero(scores >= threshold))
        recall[str(threshold)] = {
            "threshold": threshold,
            "detected_positive_rows": detected,
            "positive_recall": detected / rows,
        }

    gpu_path = ROOT / "target/ember2024-research/10_forest_challenge_wide.parquet"
    comparison = None
    failed = False
    if gpu_path.is_file():
        comparison, failed = gpu_comparison(gpu_path, rows, scores)

    output_table = pa.table({"source_id": np.arange(rows, dtype=np.int64), "score": scores})
    csv_path = report_dir / "forest_cpu_reference.csv.gz"
    with gzip.open(csv_path, "wb") as output:
        csv.write_csv(output_table, output)

    summary = {
        "modelSHA256": sha256(model_path),
        "rows": rows,
        "dimension": dimension,
        "model_feature_count": dimension,
        "score_summary": {
            "min": float(scores.min()),
            "median": float(np.median(scores)),
            "max": float(scores.max()),
        },
        "positive_recall": recall,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    if comparison is not None:
        summary["gpu_comparison"] = comparison
    (report_dir / "forest_cpu_check.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    performance_shape, performance_failed = build_performance_shape()

    print(
        json.dumps(
            {
                "rows": rows,
                "dimension": dimension,
                "gpu_comparison": comparison,
                "performance_shape_checks_passed": not performance_failed,
            },
            sort_keys=True,
        )
    )
    if failed or performance_failed:
        print("CPU/GPU reference or exact wide/vector/stack score alignment check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
