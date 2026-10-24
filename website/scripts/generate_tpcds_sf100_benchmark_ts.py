#!/usr/bin/env python3
"""Regenerate website/src/data/tpcdsSf100Benchmark.ts from TPC-DS benchmark artifacts.

Timings come from the tpcds_gpu_cpu_bench TSVs. Result-value agreement comes from
the native_tpcds_parity_report TSV, which must cover the same dataset root and
query set as the timing runs.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def read_tsv(path: Path):
    """Returns `# key=value` metadata comments and data rows as dicts."""
    meta = {}
    header = None
    rows = []
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
        rows.append(dict(zip(header, cells)))
    return meta, rows


def load(path: Path, engine: str):
    meta, rows = read_tsv(path)
    measured = {
        row["query_id"]: row for row in rows if row.get("engine") == engine and row.get("warmup") != "true"
    }
    return meta, measured


def load_parity(path: Path):
    meta, rows = read_tsv(path)
    return meta, {row["query_id"]: row for row in rows}


def required_meta(meta, key: str, path: Path) -> str:
    value = meta.get(key)
    if not value:
        raise SystemExit(f"{key} missing from {path}")
    return value


def check_parity_inputs(args, cpu_meta, gpu_meta, cpu, gpu, parity_meta, parity) -> None:
    parity_root = required_meta(parity_meta, "dataset_root", args.parity_tsv)
    for path, meta in ((args.cpu_tsv, cpu_meta), (args.gpu_tsv, gpu_meta)):
        timing_root = required_meta(meta, "tpcds_benchmark_root", path)
        if Path(timing_root).resolve() != Path(parity_root).resolve():
            raise SystemExit(
                f"dataset root mismatch: {path} tpcds_benchmark_root={timing_root}, "
                f"{args.parity_tsv} dataset_root={parity_root}"
            )
    timing_queries = set(cpu) | set(gpu)
    parity_queries = set(parity)
    if timing_queries != parity_queries:
        raise SystemExit(
            f"query set mismatch between timing TSVs and {args.parity_tsv}: "
            f"timing only={sorted(timing_queries - parity_queries)}, "
            f"parity only={sorted(parity_queries - timing_queries)}"
        )
    unchecked = []
    for query, row in sorted(parity.items()):
        cpu_ok = row.get("cpu_planning_status") == "succeeded" and row.get("cpu_runtime_status") == "succeeded"
        timed = cpu.get(query, {}).get("outcome") == "succeeded" or gpu.get(query, {}).get("outcome") == "succeeded"
        if not cpu_ok and timed:
            unchecked.append(
                f"{query} (cpu_planning_status={row.get('cpu_planning_status')}, "
                f"cpu_runtime_status={row.get('cpu_runtime_status')})"
            )
    if unchecked:
        raise SystemExit(
            "the parity CPU oracle did not produce output for queries with a succeeded timing row: "
            + ", ".join(unchecked)
        )


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
    ap.add_argument("--parity-tsv", type=Path, required=True)
    ap.add_argument("--environment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kvikio-compat-mode", default="ON")
    ap.add_argument("--generated-at-utc", default=None)
    args = ap.parse_args()

    environment = json.loads(args.environment.read_text())
    if "decimalJoinParity" in environment:
        raise SystemExit(
            f"{args.environment} contains decimalJoinParity; remove it, result-value parity "
            "now comes from --parity-tsv"
        )
    cpu_meta, cpu = load(args.cpu_tsv, "cpu_datafusion")
    gpu_meta, gpu = load(args.gpu_tsv, "native_gpu")
    parity_meta, parity = load_parity(args.parity_tsv)
    check_parity_inputs(args, cpu_meta, gpu_meta, cpu, gpu, parity_meta, parity)
    # The environment JSON records the CPU settings of the timing run; the page
    # states that the parity CPU oracle used the same settings.
    for env_key, meta_key in (("cpuTargetPartitions", "cpu_target_partitions"), ("cpuBatchSize", "cpu_batch_size")):
        parity_value = int(required_meta(parity_meta, meta_key, args.parity_tsv))
        if env_key in environment and environment[env_key] != parity_value:
            raise SystemExit(
                f"CPU setting mismatch: {args.environment} {env_key}={environment[env_key]}, "
                f"{args.parity_tsv} {meta_key}={parity_value}"
            )
    query_ids = sorted(set(cpu) | set(gpu), key=lambda q: (int("".join(ch for ch in q if ch.isdigit()) or "999"), q))
    rows = []
    for query in query_ids:
        cpu_row = cpu.get(query, {})
        gpu_row = gpu.get(query, {})
        parity_row = parity[query]
        parity_detail = parity_row.get("parity_detail", "none")
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
                "parityStatus": parity_row["parity_status"],
                "comparisonMode": parity_row["comparison_mode"],
                "parityDetail": None if parity_detail == "none" else parity_detail,
                "runtimeOperatorClass": gpu_row.get("runtime_operator_class", "")
                if gpu_row.get("runtime_operator_class") not in (None, "not_run")
                else "",
                "peakMemoryOperatorClass": gpu_row.get("peak_memory_operator_class", "")
                if gpu_row.get("peak_memory_operator_class") not in (None, "none")
                else "",
                "peakMemoryOperatorClassBytes": inum(gpu_row.get("peak_memory_operator_class_bytes")),
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

    root_value = gpu_meta.get("tpcds_benchmark_root")
    if not root_value:
        raise SystemExit(f"tpcds_benchmark_root missing from {args.gpu_tsv}")
    root = Path(root_value)
    if not root.is_dir():
        raise SystemExit(f"tpcds_benchmark_root {root} is not a directory")
    files = sorted(root.glob("*.parquet"))
    dataset = {
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
            "tpcds_benchmark_native_pipeline_execution_mode", "bounded"
        ),
        "nativeProfileName": gpu_meta.get("tpcds_benchmark_native_profile_name", "sf100_gpu_throughput"),
        "nativeProfileSource": gpu_meta.get("tpcds_benchmark_native_profile_source", "env_profile"),
        # No device budget: the bench header prints the profile's 256 MiB placeholder,
        # and the run used an automatic limit.
        "nativeMaxSourceChunkBytes": int(gpu_meta.get("tpcds_benchmark_native_max_source_chunk_bytes", "0") or 0),
        "nativeMaxRowGroupsPerChunk": int(
            gpu_meta.get("tpcds_benchmark_native_max_row_groups_per_chunk", "0") or 0
        ),
        "nativeMaxRetryAttempts": int(gpu_meta.get("tpcds_benchmark_native_max_retry_attempts", "0") or 0),
        "nativeMinRetryChunkRowGroups": int(
            gpu_meta.get("tpcds_benchmark_native_min_retry_chunk_row_groups", "0") or 0
        ),
        "nativeSourceChunkPlanningMode": gpu_meta.get(
            "tpcds_benchmark_native_source_chunk_planning_mode", "parquet_throughput"
        ),
        "nativeStatisticalAggregateExecutionMode": gpu_meta.get(
            "tpcds_benchmark_native_statistical_aggregate_execution_mode", "gpu_native_tolerant"
        ),
        "kvikioCompatMode": args.kvikio_compat_mode,
        "parity": {
            "floatToleranceAbsolute": float(
                required_meta(parity_meta, "parity_float_tolerance_absolute", args.parity_tsv)
            ),
            "floatToleranceRelative": float(
                required_meta(parity_meta, "parity_float_tolerance_relative", args.parity_tsv)
            ),
            "statisticalAggregateExecutionMode": required_meta(
                parity_meta, "statistical_aggregate_execution_mode", args.parity_tsv
            ),
            "statisticalAggregateToleranceContract": required_meta(
                parity_meta, "statistical_aggregate_tolerance_contract", args.parity_tsv
            ),
            "cpuTargetPartitions": int(required_meta(parity_meta, "cpu_target_partitions", args.parity_tsv)),
            "cpuBatchSize": int(required_meta(parity_meta, "cpu_batch_size", args.parity_tsv)),
        },
    }
    summary = {
        "queryCount": len(rows),
        "cpuSuccessCount": sum(1 for row in rows if row["cpuOutcome"] == "succeeded"),
        "gpuSuccessCount": len(gpu_succ),
        "gpuFailedQueries": [row["queryId"] for row in rows if row["gpuOutcome"] != "succeeded"],
        "parityMatchedCount": sum(1 for row in rows if row["parityStatus"] == "matched"),
        "parityComparisonModeCounts": dict(Counter(row["comparisonMode"] for row in rows)),
        "parityMismatchedQueries": [row["queryId"] for row in rows if row["parityStatus"] != "matched"],
        "cpuTotalMs": round(cpu_total, 3),
        "gpuSuccessTotalMs": round(gpu_total, 3),
        "speedup": round(cpu_for_gpu / gpu_total, 3) if gpu_total else None,
        "gpuFasterCount": sum(1 for ratio in ratios if ratio >= 1),
        "gpuSlowerCount": sum(1 for ratio in ratios if ratio < 1),
        "maxPeakMemoryOperatorClassBytes": max((row["peakMemoryOperatorClassBytes"] for row in rows), default=0),
        "fastestSpeedupQuery": max(gpu_succ, key=lambda row: row["speedup"] or 0)["queryId"] if gpu_succ else "",
        "slowestSpeedupQuery": min(gpu_succ, key=lambda row: row["speedup"] or 1e18)["queryId"] if gpu_succ else "",
        "outlierThresholdMs": threshold if threshold != float("inf") else None,
        "outliers": outliers,
    }
    stamp = args.generated_at_utc
    if not stamp:
        m = re.search(r"(20\d{6}T\d{6}Z)", str(args.gpu_tsv))
        if m:
            raw = m.group(1)
            stamp = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{raw[9:11]}:{raw[11:13]}:{raw[13:15]}Z"
        else:
            stamp = "unknown"
    payload = {
        "generatedAtUtc": stamp,
        "environment": environment,
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
  parityStatus: string;
  comparisonMode: string;
  parityDetail: string | null;
  runtimeOperatorClass: string;
  peakMemoryOperatorClass: string;
  peakMemoryOperatorClassBytes: number;
  parquetReadTimeMs: number | null;
  sourceReadTimeMs: number | null;
  operatorTimeMs: number | null;
  materializationTimeMs: number | null;
  gpuFailure: string;
};
"""
    body = json.dumps(payload, indent=2)
    text = (
        f"// Timings generated from {args.cpu_tsv.as_posix()} and {args.gpu_tsv.as_posix()}.\n"
        f"// Result-value parity: {args.parity_tsv.as_posix()}.\n"
        f"// Environment and CPU tuning: {args.environment.as_posix()}.\n"
        f"// Do not hand-edit benchmark values; regenerate from the TSV artifacts.\n\n"
        f"{type_block}\n"
        f"export const tpcdsSf100Benchmark = {body} satisfies {{\n"
        f"  generatedAtUtc: string;\n"
        f"  environment: Record<string, unknown>;\n"
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
