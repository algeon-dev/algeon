#!/usr/bin/env python3
"""Regenerate website/src/data/tpchBenchmark.ts from a relational benchmark run directory.

Usage:
  python3 website/scripts/generate_tpch_benchmark_ts.py \
    --run-dir target/relational-benchmarks/RUN \
    --out website/src/data/tpchBenchmark.ts

The run directory comes from scripts/dev/relational_benchmarks_driver.py. For each of the
datasets sf100_sorted, sf100_dbgen_order and sf500_sorted it holds tpch/ID/q01.tsv to
q22.tsv (one bench process per query), layout.json (fixture/tpc/describe_layout.py) and
page-cache.json, and it holds environment.json once for the whole run.

The generator refuses a TSV from a bench binary that predates the planning boundary, a
preamble that differs between the 22 files of a dataset, a config that differs between
datasets, a layout that disagrees with the TSV's dataset metadata, and sorted and dbgen
order outputs whose fingerprints differ.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATASETS = (
    # id, scale factor, label
    ("sf100_sorted", "sf100", "SF100 sorted"),
    ("sf100_dbgen_order", "sf100", "SF100 dbgen order"),
    ("sf500_sorted", "sf500", "SF500 sorted"),
)
QUERY_IDS = [f"q{i:02d}" for i in range(1, 23)]
WALL_TIME_BOUNDARY = "sql_planning_and_collect"
WALL_TIME_FIELDS = "per_measured_iteration_median"
NEW_COLUMNS = ("cpu_planning_ms", "native_planning_ms", "cpu_execution_ms", "native_execution_ms")
# Preamble keys that legitimately differ between the files of one dataset or between datasets.
PER_QUERY_KEYS = {"tpch_benchmark_query_ids"}
PER_DATASET_KEYS = {"tpch_benchmark_dataset_mode"}
SAMPLE_TOLERANCE_MS = 5e-4


def fail(message: str) -> None:
    raise SystemExit(message)


def load_tsv(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    meta: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in path.read_text().splitlines():
        if not raw:
            continue
        if raw.startswith("#"):
            key, sep, value = raw[1:].strip().partition("=")
            if sep:
                meta[key.strip()] = value.strip()
            continue
        cells = raw.split("\t")
        if header is None:
            header = cells
            continue
        rows.append(dict(zip(header, cells)))
    if header is None:
        fail(f"no TSV header found in {path}")
    missing = [column for column in NEW_COLUMNS if column not in header]
    if missing:
        fail(f"{path} lacks {missing}; regenerate it with a bench binary that reports planning and execution time")
    return meta, rows


def required_meta(meta: dict[str, str], key: str, path: Path) -> str:
    value = meta.get(key)
    if not value:
        fail(f"{key} missing from {path}")
    return value


def fnum(value: str | None) -> float | None:
    if value is None or value == "" or value.lower() in ("null", "none"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def required_ms(row: dict[str, str], column: str, path: Path) -> float:
    value = fnum(row.get(column))
    if value is None:
        fail(f"{path}: {column} is empty for {row.get('query_id')}")
    return value


def inum(value: str | None) -> int:
    n = fnum(value)
    return 0 if n is None else int(n)


def parse_counts(text: str) -> dict[str, int]:
    return {name: int(count) for name, _, count in (item.partition("=") for item in text.split(",") if item)}


def check_samples(row: dict[str, str], column: str, scalar: str, iterations: int, path: Path) -> None:
    samples = [float(s) for s in row[column].split(",") if s]
    if len(samples) != iterations:
        fail(f"{path}: {column} holds {len(samples)} samples for {row['query_id']}, expected {iterations}")
    median = statistics.median(samples)
    if abs(median - float(row[scalar])) > SAMPLE_TOLERANCE_MS:
        fail(f"{path}: {scalar}={row[scalar]} is not the median {median:.3f} of {column}")


def load_dataset(run_dir: Path, dataset_id: str, scale: str):
    directory = run_dir / "tpch" / dataset_id
    files = {query: directory / f"{query}.tsv" for query in QUERY_IDS}
    absent = [str(path) for path in files.values() if not path.exists()]
    if absent:
        fail(f"{dataset_id}: missing {absent}")
    rows: list[dict[str, str]] = []
    first_meta: dict[str, str] | None = None
    for query, path in files.items():
        meta, tsv_rows = load_tsv(path)
        if len(tsv_rows) != 1 or tsv_rows[0]["query_id"] != query:
            fail(f"{path}: expected one row for {query}, found {[r['query_id'] for r in tsv_rows]}")
        if meta.get("tpch_benchmark_query_ids") != query:
            fail(f"{path}: tpch_benchmark_query_ids={meta.get('tpch_benchmark_query_ids')}, expected {query}")
        if required_meta(meta, "tpch_benchmark_wall_time_boundary", path) != WALL_TIME_BOUNDARY:
            fail(f"{path}: wall time boundary is {meta['tpch_benchmark_wall_time_boundary']}, expected {WALL_TIME_BOUNDARY}")
        if required_meta(meta, "tpch_benchmark_wall_time_fields", path) != WALL_TIME_FIELDS:
            fail(f"{path}: wall time fields are {meta['tpch_benchmark_wall_time_fields']}, expected {WALL_TIME_FIELDS}")
        shared = {k: v for k, v in meta.items() if k.startswith("tpch_") and k not in PER_QUERY_KEYS}
        if first_meta is None:
            first_meta = shared
        elif shared != first_meta:
            differing = sorted(k for k in set(shared) | set(first_meta) if shared.get(k) != first_meta.get(k))
            fail(f"{path}: preamble differs from {files[QUERY_IDS[0]]} in {differing}")
        row = tsv_rows[0]
        iterations = int(meta["tpch_benchmark_iterations"])
        check_samples(row, "cpu_wall_ms_samples", "cpu_wall_ms", iterations, path)
        check_samples(row, "native_wall_ms_samples", "native_wall_ms", iterations, path)
        rows.append(row)
    assert first_meta is not None
    if first_meta["tpch_dataset_scale_factor"] != scale:
        fail(f"{directory}: tpch_dataset_scale_factor={first_meta['tpch_dataset_scale_factor']}, expected {scale}")
    layout = json.loads((directory / "layout.json").read_text())
    check_layout(directory, first_meta, layout)
    page_cache = json.loads((directory / "page-cache.json").read_text())
    return first_meta, rows, layout, page_cache


def check_layout(directory: Path, meta: dict[str, str], layout: dict) -> None:
    tables = layout["tables"]
    for key, field in (
        ("tpch_dataset_table_row_counts", "rows"),
        ("tpch_dataset_table_row_group_counts", "rowGroups"),
    ):
        reported = parse_counts(meta[key])
        measured = {name: table[field] for name, table in tables.items()}
        if reported != measured:
            fail(f"{directory}: {key} {reported} differs from layout.json {measured}")
    if parse_counts(meta["tpch_dataset_table_file_counts"]) != {name: 1 for name in tables}:
        fail(f"{directory}: the bench reports more than one file per table")
    if int(meta["tpch_dataset_total_bytes"]) != sum(table["bytes"] for table in tables.values()):
        fail(f"{directory}: tpch_dataset_total_bytes differs from the file sizes in layout.json")


def website_row(r: dict[str, str], path: Path) -> dict:
    cpu_ms, gpu_ms = required_ms(r, "cpu_wall_ms", path), required_ms(r, "native_wall_ms", path)
    return {
        "queryId": r["query_id"].upper(),
        "cpuMs": cpu_ms,
        "gpuMs": gpu_ms,
        "speedup": fnum(r.get("speedup_vs_cpu")) or 0.0,
        "cpuPlanningMs": required_ms(r, "cpu_planning_ms", path),
        "gpuPlanningMs": required_ms(r, "native_planning_ms", path),
        "cpuExecutionMs": required_ms(r, "cpu_execution_ms", path),
        "gpuExecutionMs": required_ms(r, "native_execution_ms", path),
        "planningOutcome": r.get("planning_outcome", ""),
        "runtimeOutcome": r.get("runtime_outcome", ""),
        "parityStatus": r.get("parity_status", ""),
        "rowGroupsSelected": inum(r.get("row_groups_selected")),
        "parquetFiles": inum(r.get("parquet_files")),
        "peakMemoryOperatorClass": r.get("peak_memory_operator_class", "") or "",
        "peakMemoryOperatorClassBytes": inum(r.get("peak_memory_operator_class_bytes")),
        "sourcePredicatePushdownStatus": r.get("source_predicate_pushdown_status", "") or "",
        "partitionedJoinCount": inum(r.get("partitioned_join_count")),
        "joinPartitioningStatus": r.get("join_partitioning_status", "") or "",
        "aggregateMergeStatus": r.get("aggregate_merge_status", "") or "",
        "parquetReadTimeMs": fnum(r.get("parquet_read_time_ms")),
        "sourceReadTimeMs": fnum(r.get("source_read_time_ms")),
        "operatorTimeMs": fnum(r.get("operator_time_ms")),
        "materializationTimeMs": fnum(r.get("materialization_time_ms")),
    }


def summarize(rows: list[dict[str, str]], website_rows: list[dict]) -> dict:
    cpu_total = sum(r["cpuMs"] for r in website_rows)
    gpu_total = sum(r["gpuMs"] for r in website_rows)
    cpu_planning = sum(r["cpuPlanningMs"] for r in website_rows)
    gpu_planning = sum(r["gpuPlanningMs"] for r in website_rows)
    speedups = [(r["queryId"], r["speedup"]) for r in website_rows]
    return {
        "queryCount": len(rows),
        "cpuTotalMs": round(cpu_total, 3),
        "gpuTotalMs": round(gpu_total, 3),
        "speedup": round(cpu_total / gpu_total, 3),
        "cpuPlanningTotalMs": round(cpu_planning, 3),
        "gpuPlanningTotalMs": round(gpu_planning, 3),
        "cpuPlanningShare": round(cpu_planning / cpu_total, 5),
        "gpuPlanningShare": round(gpu_planning / gpu_total, 5),
        "nativeQueryCount": sum(1 for r in rows if r["planning_outcome"] == "selected" and r["runtime_outcome"] == "succeeded"),
        "parityMatchedCount": sum(1 for r in rows if r["parity_status"] == "matched"),
        "allPlanningAccepted": all(r["planning_outcome"] == "selected" for r in rows),
        "allRuntimeSucceeded": all(r["runtime_outcome"] == "succeeded" for r in rows),
        "allParityMatched": all(r["parity_status"] == "matched" for r in rows),
        "maxPeakMemoryOperatorClassBytes": max(r["peakMemoryOperatorClassBytes"] for r in website_rows),
        "parquetFilesSummedAcrossQueries": sum(r["parquetFiles"] for r in website_rows),
        "rowGroupsSelectedSummedAcrossQueries": sum(r["rowGroupsSelected"] for r in website_rows),
        "fastestSpeedupQuery": max(speedups, key=lambda x: x[1])[0],
        "slowestSpeedupQuery": min(speedups, key=lambda x: x[1])[0],
    }


def build_config(meta: dict[str, str], path: Path) -> dict:
    def req(name: str) -> str:
        return required_meta(meta, f"tpch_benchmark_{name}", path)

    return {
        "queryIds": QUERY_IDS,
        "warmups": int(req("warmups")),
        "iterations": int(req("iterations")),
        "wallTimeFields": req("wall_time_fields"),
        "wallTimeBoundary": req("wall_time_boundary"),
        "parquetPushdownFilters": req("parquet_pushdown_filters"),
        "relationalExecution": req("relational_execution"),
        "nativeMemoryPolicy": req("native_memory_policy"),
        "nativeProfileName": req("native_profile_name"),
        "nativeProfileSource": req("native_profile_source"),
        "nativeDeviceBudgetPolicy": req("native_device_budget_policy"),
        "nativeDeviceBudgetBytes": int(req("native_device_budget_bytes")),
        "nativeMaxSourceChunkBytes": int(req("native_max_source_chunk_bytes")),
        "nativeMaxRowGroupsPerChunk": int(req("native_max_row_groups_per_chunk")),
        "nativeMaxRetryAttempts": int(req("native_max_retry_attempts")),
        "nativeMinRetryChunkRowGroups": int(req("native_min_retry_chunk_row_groups")),
        "nativeSourceChunkPlanningMode": req("native_source_chunk_planning_mode"),
        "nativeMaxSourceReadsInFlight": int(req("native_max_source_reads_in_flight")),
        "nativeRuntimePolicy": req("native_runtime_policy"),
        "parity": {
            "floatToleranceAbsolute": float(req("parity_float_tolerance_absolute")),
            "floatToleranceRelative": float(req("parity_float_tolerance_relative")),
        },
    }


def page_cache_record(record: dict) -> dict:
    return {
        "totalBytes": record["before"]["total_bytes"],
        "residentFractionBefore": round(record["before"]["resident_fraction_by_bytes"], 4),
        "residentFractionAfter": round(record["after"]["resident_fraction_by_bytes"], 4),
    }


TYPE_BLOCK = """export type TpchBenchmarkRow = {
  queryId: string;
  cpuMs: number;
  gpuMs: number;
  speedup: number;
  cpuPlanningMs: number;
  gpuPlanningMs: number;
  cpuExecutionMs: number;
  gpuExecutionMs: number;
  planningOutcome: string;
  runtimeOutcome: string;
  parityStatus: string;
  rowGroupsSelected: number;
  parquetFiles: number;
  peakMemoryOperatorClass: string;
  peakMemoryOperatorClassBytes: number;
  sourcePredicatePushdownStatus: string;
  partitionedJoinCount: number;
  joinPartitioningStatus: string;
  aggregateMergeStatus: string;
  parquetReadTimeMs: number | null;
  sourceReadTimeMs: number | null;
  operatorTimeMs: number | null;
  materializationTimeMs: number | null;
};
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    environment = json.loads((args.run_dir / "environment.json").read_text())
    datasets, configs, loaded = [], {}, {}
    for dataset_id, scale, label in DATASETS:
        meta, rows, layout, page_cache = load_dataset(args.run_dir, dataset_id, scale)
        loaded[dataset_id] = rows
        path = args.run_dir / "tpch" / dataset_id
        configs[dataset_id] = build_config({k: v for k, v in meta.items() if k not in PER_DATASET_KEYS}, path)
        website_rows = [website_row(r, path / f"{r['query_id']}.tsv") for r in rows]
        datasets.append(
            {
                "id": dataset_id,
                "label": label,
                "scaleFactor": scale,
                "generationCommand": required_meta(meta, "tpch_dataset_generation_command", path),
                "layout": layout,
                "pageCache": page_cache_record(page_cache),
                "summary": summarize(rows, website_rows),
                "rows": website_rows,
            }
        )
    first = DATASETS[0][0]
    for dataset_id, config in configs.items():
        if config != configs[first]:
            differing = sorted(k for k in config if config[k] != configs[first][k])
            fail(f"config of {dataset_id} differs from {first} in {differing}")

    mismatched = [
        q["query_id"]
        for q, d in zip(loaded["sf100_sorted"], loaded["sf100_dbgen_order"])
        if (q["output_fingerprint"], q["cpu_output_rows"], q["native_output_rows"])
        != (d["output_fingerprint"], d["cpu_output_rows"], d["native_output_rows"])
    ]
    if mismatched:
        fail(f"sf100_sorted and sf100_dbgen_order differ in output fingerprint or row count for {mismatched}")

    payload = {
        "environment": environment,
        "config": configs[first],
        "datasets": datasets,
        "dbgenOrderComparison": {"identicalOutputQueryCount": len(QUERY_IDS)},
    }
    body = json.dumps(payload, indent=2)
    text = (
        f"// Generated from {args.run_dir.name}/tpch/*/q*.tsv, layout.json and page-cache.json.\n"
        f"// Do not hand-edit benchmark values; regenerate from the run directory.\n\n"
        f"{TYPE_BLOCK}\n"
        f"export const tpchBenchmark = {body} as const satisfies "
        f"{{ datasets: readonly {{ rows: readonly TpchBenchmarkRow[]; [key: string]: unknown }}[]; [key: string]: unknown }};\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    for dataset in datasets:
        s = dataset["summary"]
        print(
            f"{dataset['id']}: queries={s['queryCount']} cpu_s={s['cpuTotalMs'] / 1000:.3f} "
            f"gpu_s={s['gpuTotalMs'] / 1000:.3f} speedup={s['speedup']:.3f} "
            f"parity_ok={s['allParityMatched']}"
        )


if __name__ == "__main__":
    main()
