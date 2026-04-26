#!/usr/bin/env python3
"""Regenerate website/src/data/tpcdsSf100Benchmark.ts from TPC-DS three-way bench TSVs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path, engine: str):
    meta = {}
    header = None
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if "=" in line:
                key, _, value = line[1:].partition("=")
                meta[key.strip()] = value.strip()
            continue
        cells = line.split("\t")
        if header is None:
            header = cells
            continue
        row = dict(zip(header, cells))
        if row.get("engine") != engine or row.get("warmup") == "true":
            continue
        rows[row["query_id"]] = row
    return meta, rows


def fnum(value):
    if value in (None, "", "not_run"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inum(value):
    number = fnum(value)
    return int(number) if number is not None else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-tsv", type=Path, required=True)
    ap.add_argument("--gpu-tsv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kvikio-compat-mode", default="ON")
    ap.add_argument("--generated-at-utc", default=None)
    args = ap.parse_args()

    cpu_meta, cpu = load(args.cpu_tsv, "cpu_datafusion")
    gpu_meta, gpu = load(args.gpu_tsv, "native_gpu")
    query_ids = sorted(set(cpu) | set(gpu), key=lambda q: (int("".join(ch for ch in q if ch.isdigit()) or "999"), q))
    rows = []
    for query in query_ids:
        cpu_row = cpu.get(query, {})
        gpu_row = gpu.get(query, {})
        cpu_ms = fnum(cpu_row.get("wall_ms")) or 0.0
        gpu_ms = fnum(gpu_row.get("wall_ms")) if gpu_row.get("outcome") == "succeeded" else None
        gpu_wall = fnum(gpu_row.get("wall_ms"))
        speedup = (cpu_ms / gpu_ms) if gpu_ms else None
        rows.append(
            {
                "queryId": query.upper(),
                "cpuMs": round(cpu_ms, 3),
                "gpuMs": round(gpu_ms, 3) if gpu_ms is not None else None,
                "gpuWallMs": round(gpu_wall, 3) if gpu_wall is not None else None,
                "speedup": round(speedup, 3) if speedup is not None else None,
                "cpuOutcome": cpu_row.get("outcome", ""),
                "gpuOutcome": gpu_row.get("outcome", ""),
                "planningOutcome": gpu_row.get("planning_outcome", ""),
                "rowsMatch": (
                    cpu_row.get("outcome") == "succeeded"
                    and gpu_row.get("outcome") == "succeeded"
                    and cpu_row.get("output_rows") == gpu_row.get("output_rows")
                ),
                "runtimeOperatorClass": gpu_row.get("runtime_operator_class", "")
                if gpu_row.get("runtime_operator_class") not in (None, "not_run")
                else "",
                "peakMemoryOperatorClass": gpu_row.get("peak_memory_operator_class", "")
                if gpu_row.get("peak_memory_operator_class") not in (None, "none")
                else "",
                "peakMemoryOperatorClassBytes": inum(gpu_row.get("peak_memory_operator_class_bytes")),
                "reservationLivePeakBytes": inum(gpu_row.get("reservation_live_peak_bytes")),
                "parquetReadTimeMs": fnum(gpu_row.get("parquet_read_time_ms")),
                "sourceReadTimeMs": fnum(gpu_row.get("source_read_time_ms")),
                "operatorTimeMs": fnum(gpu_row.get("operator_time_ms")),
                "materializationTimeMs": fnum(gpu_row.get("materialization_time_ms")),
                "gpuFailure": gpu_row.get("failure", "") or "",
            }
        )

    cpu_total = sum(row["cpuMs"] for row in rows if row["cpuOutcome"] == "succeeded")
    gpu_succ = [row for row in rows if row["gpuOutcome"] == "succeeded" and row["gpuMs"] is not None]
    gpu_total = sum(row["gpuMs"] for row in gpu_succ)
    cpu_for_gpu = sum(row["cpuMs"] for row in gpu_succ)
    ratios = [row["speedup"] for row in gpu_succ if row["speedup"] is not None]
    vals = sorted(max(row["cpuMs"], row["gpuWallMs"] or 0) for row in rows if row["cpuMs"] or row["gpuWallMs"])
    if len(vals) >= 4:
        q1, _, q3 = statistics.quantiles(vals, n=4, method="exclusive")
        threshold = q3 + 1.5 * (q3 - q1)
    else:
        threshold = float("inf")
    outliers = [row["queryId"] for row in rows if max(row["cpuMs"], row["gpuWallMs"] or 0) > threshold]

    root = Path("/opt/dlami/nvme/fixture/data/tpcds/sf100/parquet")
    files = sorted(root.glob("*.parquet")) if root.exists() else []
    dataset = {
        "fixture": "AWS EC2 g7e.4xlarge local NVMe",
        "generationScript": "fixture/fixture.sh tpcds generate",
        "generatorImplementation": "fixture/tpc-ds/generate.py",
        "scaleFactor": "sf100",
        "tableCount": len(files),
        "fileCount": len(files),
        "totalBytes": sum(path.stat().st_size for path in files),
        "tableFileCounts": ",".join(f"{path.stem}=1" for path in files),
        "tableByteSizes": ",".join(f"{path.stem}={path.stat().st_size}" for path in files),
    }
    config = {
        "queryIds": query_ids,
        "warmups": int(gpu_meta.get("tpcds_benchmark_warmups", "1")),
        "iterations": int(gpu_meta.get("tpcds_benchmark_iterations", "1")),
        "wallTimeFields": gpu_meta.get("tpcds_benchmark_wall_time_fields", "per_measured_iteration"),
        "datasetMode": "sf100_external",
        "nativePipelineExecutionMode": gpu_meta.get(
            "tpcds_benchmark_native_pipeline_execution_mode", "strict_bounded"
        ),
        "nativeProfileName": gpu_meta.get("tpcds_benchmark_native_profile_name", "sf100_gpu_throughput"),
        "nativeProfileSource": gpu_meta.get("tpcds_benchmark_native_profile_source", "env_profile"),
        "nativeDeviceBudgetBytes": int(gpu_meta.get("tpcds_benchmark_native_device_budget_bytes", "0") or 0),
        "nativeMaxSourceChunkBytes": int(gpu_meta.get("tpcds_benchmark_native_max_source_chunk_bytes", "0") or 0),
        "nativeMaxRowGroupsPerChunk": int(
            gpu_meta.get("tpcds_benchmark_native_max_row_groups_per_chunk", "0") or 0
        ),
        "nativeMaxRetryAttempts": int(gpu_meta.get("tpcds_benchmark_native_max_retry_attempts", "0") or 0),
        "nativeMinRetryChunkRowGroups": int(
            gpu_meta.get("tpcds_benchmark_native_min_retry_chunk_row_groups", "0") or 0
        ),
        "nativeSourceChunkPlanningMode": gpu_meta.get(
            "tpcds_benchmark_native_source_chunk_planning_mode", "local_parquet_throughput"
        ),
        "nativeStatisticalAggregateExecutionMode": gpu_meta.get(
            "tpcds_benchmark_native_statistical_aggregate_execution_mode", "gpu_native_tolerant"
        ),
        "kvikioCompatMode": args.kvikio_compat_mode,
    }
    summary = {
        "queryCount": len(rows),
        "cpuSuccessCount": sum(1 for row in rows if row["cpuOutcome"] == "succeeded"),
        "gpuSuccessCount": len(gpu_succ),
        "gpuFailedQueries": [row["queryId"] for row in rows if row["gpuOutcome"] != "succeeded"],
        "rowsMatchCount": sum(1 for row in rows if row["rowsMatch"]),
        "cpuTotalMs": round(cpu_total, 3),
        "gpuSuccessTotalMs": round(gpu_total, 3),
        "speedup": round(cpu_for_gpu / gpu_total, 3) if gpu_total else None,
        "gpuFasterCount": sum(1 for ratio in ratios if ratio >= 1),
        "gpuSlowerCount": sum(1 for ratio in ratios if ratio < 1),
        "maxReservationLivePeakBytes": max((row["reservationLivePeakBytes"] for row in rows), default=0),
        "maxPeakMemoryOperatorClassBytes": max((row["peakMemoryOperatorClassBytes"] for row in rows), default=0),
        "fastestSpeedupQuery": max(gpu_succ, key=lambda row: row["speedup"] or 0)["queryId"] if gpu_succ else "",
        "slowestSpeedupQuery": min(gpu_succ, key=lambda row: row["speedup"] or 1e18)["queryId"] if gpu_succ else "",
        "outlierThresholdMs": threshold if threshold != float("inf") else None,
        "outliers": outliers,
    }
    stamp = args.generated_at_utc
    if not stamp:
        for part in args.gpu_tsv.parts:
            if part.startswith("20") and "T" in part and part.endswith("Z"):
                raw = part.split("_")[-1]
                stamp = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{raw[9:11]}:{raw[11:13]}:{raw[13:15]}Z"
                break
        stamp = stamp or "unknown"
    payload = {
        "generatedAtUtc": stamp,
        "dataset": dataset,
        "config": config,
        "summary": summary,
        "rows": rows,
    }
    type_block = """export type TpcdsSf100BenchmarkRow = {
  queryId: string;
  cpuMs: number;
  gpuMs: number | null;
  gpuWallMs: number | null;
  speedup: number | null;
  cpuOutcome: string;
  gpuOutcome: string;
  planningOutcome: string;
  rowsMatch: boolean;
  runtimeOperatorClass: string;
  peakMemoryOperatorClass: string;
  peakMemoryOperatorClassBytes: number;
  reservationLivePeakBytes: number;
  parquetReadTimeMs: number | null;
  sourceReadTimeMs: number | null;
  operatorTimeMs: number | null;
  materializationTimeMs: number | null;
  gpuFailure: string;
};
"""
    label = args.gpu_tsv.parent.name
    body = json.dumps(payload, indent=2)
    text = (
        f"// Generated from target/bench/{label}/cpu_tpcds_three_way_bench.tsv and gpu_tpcds_three_way_bench.tsv.\n"
        f"// Do not hand-edit benchmark values; regenerate from the TSV artifacts.\n\n"
        f"{type_block}\n"
        f"export const tpcdsSf100Benchmark = {body} satisfies {{\n"
        f"  generatedAtUtc: string;\n"
        f"  dataset: Record<string, unknown>;\n"
        f"  config: Record<string, unknown>;\n"
        f"  summary: Record<string, unknown>;\n"
        f"  rows: readonly TpcdsSf100BenchmarkRow[];\n"
        f"}};\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    print(
        f"summary: queries={summary['queryCount']} cpu_s={cpu_total/1000:.2f} "
        f"gpu_s={gpu_total/1000:.2f} speedup={summary['speedup']} "
        f"gpu_ok={summary['gpuSuccessCount']} failed={summary['gpuFailedQueries']}"
    )


if __name__ == "__main__":
    main()
