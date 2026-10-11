#!/usr/bin/env python3
"""Regenerate website/src/data/tpchSf100Benchmark.ts from a native_tpch_benchmark TSV.

Usage:
  python3 website/scripts/generate_tpch_sf100_benchmark_ts.py \
    --tsv target/bench/.../native_tpch_benchmark.tsv \
    --environment target/bench/.../environment.json \
    --kvikio-compat-mode ON \
    --out website/src/data/tpchSf100Benchmark.ts
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_tsv(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    meta: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in path.read_text().splitlines():
        if not raw:
            continue
        if raw.startswith("#"):
            body = raw[1:].strip()
            if "=" in body:
                key, _, value = body.partition("=")
                meta[key.strip()] = value.strip()
            continue
        cells = raw.split("\t")
        if header is None:
            header = cells
            continue
        rows.append(dict(zip(header, cells)))
    if header is None:
        raise SystemExit(f"no TSV header found in {path}")
    return meta, rows


def fnum(value: str | None) -> float | None:
    if value is None or value == "" or value.lower() == "null":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inum(value: str | None) -> int:
    n = fnum(value)
    if n is None:
        return 0
    return int(n)


def generated_at_from_path(tsv: Path, meta: dict[str, str]) -> str:
    m = re.search(r"(20\d{6}T\d{6}Z)", str(tsv))
    if m:
        stamp = m.group(1)
        return f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z"
    return meta.get("tpch_benchmark_generated_at", "unknown")


def nullish_ms(value: str | None) -> float | None:
    n = fnum(value)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--environment", type=Path, required=True)
    ap.add_argument("--kvikio-compat-mode", default="ON")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--artifact-label", default=None, help="comment label; default = tsv parent dir name")
    args = ap.parse_args()

    meta, rows = load_tsv(args.tsv)
    environment = json.loads(args.environment.read_text())

    query_ids = [r["query_id"] for r in rows]
    cpu_total = sum(fnum(r.get("cpu_wall_ms")) or 0.0 for r in rows)
    gpu_total = sum(fnum(r.get("native_wall_ms")) or 0.0 for r in rows)
    speedup = (cpu_total / gpu_total) if gpu_total else float("nan")

    per_speedups = []
    for r in rows:
        sp = fnum(r.get("speedup_vs_cpu"))
        if sp is not None:
            per_speedups.append((r["query_id"].upper(), sp))
    fastest = max(per_speedups, key=lambda x: x[1])[0] if per_speedups else ""
    slowest = min(per_speedups, key=lambda x: x[1])[0] if per_speedups else ""

    website_rows = []
    for r in rows:
        website_rows.append(
            {
                "queryId": r["query_id"].upper(),
                "cpuMs": fnum(r.get("cpu_wall_ms")) or 0.0,
                "gpuMs": fnum(r.get("native_wall_ms")) or 0.0,
                "speedup": fnum(r.get("speedup_vs_cpu")) or 0.0,
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
                "parquetReadTimeMs": nullish_ms(r.get("parquet_read_time_ms")),
                "sourceReadTimeMs": nullish_ms(r.get("source_read_time_ms")),
                "operatorTimeMs": nullish_ms(r.get("operator_time_ms")),
                "materializationTimeMs": nullish_ms(r.get("materialization_time_ms")),
            }
        )

    summary = {
        "queryCount": len(rows),
        "cpuTotalMs": round(cpu_total, 3),
        "gpuTotalMs": round(gpu_total, 3),
        "speedup": round(speedup, 3) if speedup == speedup else None,
        "allPlanningAccepted": all(r.get("planning_outcome") == "selected" for r in rows),
        "allRuntimeSucceeded": all(r.get("runtime_outcome") == "succeeded" for r in rows),
        "allParityMatched": all(r.get("parity_status") == "matched" for r in rows),
        "maxPeakMemoryOperatorClassBytes": max(
            (inum(r.get("peak_memory_operator_class_bytes")) for r in rows), default=0
        ),
        "parquetFilesSummedAcrossQueries": sum(inum(r.get("parquet_files")) for r in rows),
        "rowGroupsSelectedSummedAcrossQueries": sum(inum(r.get("row_groups_selected")) for r in rows),
        "fastestSpeedupQuery": fastest,
        "slowestSpeedupQuery": slowest,
    }

    # Preserve known website fixture wording whenever generation command matches.
    gen_cmd = meta.get("tpch_dataset_generation_command", "fixture/fixture.sh tpch generate")
    dataset = {
        "generationScript": gen_cmd if gen_cmd.startswith("fixture/") else "fixture/fixture.sh tpch generate",
        "generatorImplementation": "fixture/tpc-h/generate.py",
        "scaleFactor": meta.get("tpch_dataset_scale_factor", "sf100"),
        "tableCount": inum(meta.get("tpch_dataset_table_count")),
        "fileCount": inum(meta.get("tpch_dataset_file_count")),
        "totalRows": inum(meta.get("tpch_dataset_total_rows")),
        "totalBytes": inum(meta.get("tpch_dataset_total_bytes")),
        "tableFileCounts": meta.get("tpch_dataset_table_file_counts", ""),
        "tableRowCounts": meta.get("tpch_dataset_table_row_counts", ""),
        "tableByteSizes": meta.get("tpch_dataset_table_byte_sizes", ""),
        "tableRowGroupCounts": meta.get("tpch_dataset_table_row_group_counts", ""),
    }

    def meta_or(key: str, default: Any) -> Any:
        return meta.get(key, default)

    config = {
        "queryIds": query_ids,
        "warmups": inum(meta_or("tpch_benchmark_warmups", "1")),
        "iterations": inum(meta_or("tpch_benchmark_iterations", "1")),
        "wallTimeFields": meta_or("tpch_benchmark_wall_time_fields", "per_measured_iteration_mean"),
        "datasetMode": meta_or("tpch_benchmark_dataset_mode", "sf100_external"),
        "parquetPushdownFilters": meta_or("tpch_benchmark_parquet_pushdown_filters", "session_default"),
        "nativePipelineExecutionMode": meta_or(
            "tpch_benchmark_native_pipeline_execution_mode", "bounded"
        ),
        "nativeMemoryPolicy": meta_or("tpch_benchmark_native_memory_policy", "bounded"),
        "nativeProfileName": meta_or("tpch_benchmark_native_profile_name", "sf100_gpu_throughput"),
        "nativeProfileSource": meta_or("tpch_benchmark_native_profile_source", "resolved_values"),
        "nativeDeviceBudgetPolicy": meta_or("tpch_benchmark_native_device_budget_policy", "automatic"),
        "nativeDeviceBudgetBytes": inum(
            meta_or("tpch_benchmark_native_device_budget_bytes", "102005473280")
        ),
        "nativeMaxSourceChunkBytes": inum(
            meta_or("tpch_benchmark_native_max_source_chunk_bytes", "34359738368")
        ),
        "nativeMaxRowGroupsPerChunk": inum(
            meta_or("tpch_benchmark_native_max_row_groups_per_chunk", "1024")
        ),
        "nativeMaxRetryAttempts": inum(meta_or("tpch_benchmark_native_max_retry_attempts", "4")),
        "nativeMinRetryChunkRowGroups": inum(
            meta_or("tpch_benchmark_native_min_retry_chunk_row_groups", "1")
        ),
        "nativeSourceChunkPlanningMode": meta_or(
            "tpch_benchmark_native_source_chunk_planning_mode", "local_parquet_throughput"
        ),
        "nativeMaxSourceReadsInFlight": inum(
            meta_or("tpch_benchmark_native_max_source_reads_in_flight", "2")
        ),
        "nativeRuntimePolicy": meta_or(
            "tpch_benchmark_native_runtime_policy", "shared_per_benchmark_run"
        ),
        "kvikioCompatMode": args.kvikio_compat_mode,
    }

    payload = {
        "generatedAtUtc": generated_at_from_path(args.tsv, meta),
        "environment": environment,
        "dataset": dataset,
        "config": config,
        "summary": summary,
        "rows": website_rows,
    }

    label = args.artifact_label or args.tsv.parent.name
    type_block = '''export type TpchSf100BenchmarkRow = {
  queryId: string;
  cpuMs: number;
  gpuMs: number;
  speedup: number;
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
'''

    # Keep numbers compact-ish like the previous file (JSON dump with default spacing).
    body = json.dumps(payload, indent=2)
    # TypeScript-friendly: leave raw nulls; JSON is valid TS object literal here.
    text = (
        f"// Generated from target/bench/{label}/native_tpch_benchmark.tsv.\n"
        f"// Do not hand-edit benchmark values; regenerate from the TSV artifact.\n\n"
        f"{type_block}\n"
        f"export const tpchSf100Benchmark = {body} as const satisfies "
        f"{{ rows: readonly TpchSf100BenchmarkRow[]; [key: string]: unknown }};\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    print(
        f"summary: queries={summary['queryCount']} cpu_s={cpu_total/1000:.3f} "
        f"gpu_s={gpu_total/1000:.3f} speedup={speedup:.3f} "
        f"planning_ok={summary['allPlanningAccepted']} runtime_ok={summary['allRuntimeSucceeded']} "
        f"parity_ok={summary['allParityMatched']}"
    )


if __name__ == "__main__":
    main()
