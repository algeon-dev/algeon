#!/usr/bin/env python3
"""Stitch datafusion-nexus native + CPU DataFusion + reference TPC-DS timings into a
single comparison TSV plus a short markdown status note.

Inputs:
  --datafusion-nexus-tsv  TSV emitted by src/bin/tpcds_three_way_bench.rs
                         (engine column carries native_gpu + cpu_datafusion).
  --reference-run-dir       reference run directory produced by
                         reference/test/tpcds_performance/benchmark_and_validate.sh,
                         containing reference/timings.csv (+ optionally duckdb/timings.csv).
  --output-dir           Where to write the comparison TSV and status note.

The script reports mean wall_ms across measured iterations (skipping warmups) for
the two datafusion-nexus engines, and parses reference timings.csv for the cold/warm
columns. It then derives speedup ratios and tags reference-wins for follow-up.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


COMPARISON_HEADER = [
    "query_id",
    "cpu_df_mean_ms",
    "cpu_df_iters",
    "native_gpu_mean_ms",
    "native_gpu_iters",
    "native_gpu_outcome",
    "native_gpu_reservation_peak_bytes",
    "reference_cold_ms",
    "reference_warm_ms",
    "duckdb_cold_ms",
    "duckdb_warm_ms",
    "speedup_native_vs_cpu_df",
    "speedup_native_vs_reference_warm",
    "speedup_reference_warm_vs_native",
    "reference_wins_vs_native",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafusion-nexus-tsv", required=True, type=Path)
    parser.add_argument("--reference-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query-ids", default="3,7,22,26,32,37,42,52,55,62,82,85,92,93,97",
                        help="Comma-separated TPC-DS query numbers (without 'q' prefix).")
    return parser.parse_args()


def parse_metadata_comment(line: str) -> tuple[str | None, str | None]:
    text = line[1:].strip()
    if "=" not in text:
        return None, None
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None, None
    return key, value.strip()


def load_datafusion_nexus(tsv_path: Path) -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    """Aggregate per-(engine, query_id) mean wall_ms over measured iterations."""
    rows = defaultdict(list)
    pending_metadata = {}
    metadata_blocks = []
    idx = None
    with tsv_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                key, value = parse_metadata_comment(line.rstrip("\n"))
                if key:
                    pending_metadata[key] = value
                continue
            cells = line.rstrip("\n").split("\t")
            if cells and cells[0] == "engine":
                idx = {column: i for i, column in enumerate(cells)}
                if pending_metadata:
                    metadata_blocks.append(pending_metadata)
                    pending_metadata = {}
                continue
            if idx is None:
                continue
            row = {column: cells[index] if index < len(cells) else "" for column, index in idx.items()}
            if row.get("warmup", "false").lower() == "true":
                continue
            engine = row["engine"]
            qid = row["query_id"]
            if row.get("outcome") != "succeeded":
                rows[(engine, qid)].append({
                    "wall_ms": None,
                    "outcome": row.get("outcome", ""),
                    "failure": row.get("failure", ""),
                    "reservation_peak": None,
                })
                continue
            try:
                wall_ms = float(row["wall_ms"])
            except (KeyError, ValueError):
                wall_ms = None
            try:
                peak = int(row["reservation_live_peak_bytes"]) if row.get("reservation_live_peak_bytes") else None
            except ValueError:
                peak = None
            rows[(engine, qid)].append({
                "wall_ms": wall_ms,
                "outcome": row.get("outcome", ""),
                "failure": row.get("failure", ""),
                "reservation_peak": peak,
            })

    summary = {}
    for (engine, qid), iters in rows.items():
        ok = [i for i in iters if i["wall_ms"] is not None]
        outcome = "succeeded" if ok else (iters[0]["outcome"] if iters else "no_data")
        summary[(engine, qid)] = {
            "mean_ms": statistics.fmean(i["wall_ms"] for i in ok) if ok else None,
            "iters": len(ok),
            "outcome": outcome,
            "failure": next((i["failure"] for i in iters if i["failure"]), ""),
            "reservation_peak": max((i["reservation_peak"] for i in ok if i["reservation_peak"] is not None),
                                    default=None),
        }
    return summary, native_gpu_metadata(metadata_blocks)


def native_gpu_metadata(metadata_blocks: list[dict[str, str]]) -> dict[str, str]:
    for metadata in metadata_blocks:
        if metadata.get("tpcds_benchmark_engine") == "native_gpu":
            return metadata
    return metadata_blocks[-1] if metadata_blocks else {}


def load_reference_timings(run_dir: Path) -> dict[str, dict]:
    timings_csv = run_dir / "reference" / "timings.csv"
    if not timings_csv.exists():
        return {}
    out = {}
    with timings_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("query")
            if not qid:
                continue
            try:
                cold = float(row["run1_time"]) * 1000.0 if row["run1_status"] == "OK" else None
                warm = float(row["run2_time"]) * 1000.0 if row["run2_status"] == "OK" else None
            except (ValueError, KeyError):
                cold = warm = None
            out[qid] = {
                "cold_ms": cold,
                "warm_ms": warm,
                "cold_status": row.get("run1_status", ""),
                "warm_status": row.get("run2_status", ""),
            }
    return out


def load_duckdb_timings(run_dir: Path) -> dict[str, dict]:
    timings_csv = run_dir / "duckdb" / "timings.csv"
    if not timings_csv.exists():
        return {}
    out = {}
    with timings_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("query")
            if not qid:
                continue
            try:
                cold = float(row["run1_time"]) * 1000.0 if row["run1_status"] == "OK" else None
                warm = float(row["run2_time"]) * 1000.0 if row["run2_status"] == "OK" else None
            except (ValueError, KeyError):
                cold = warm = None
            out[qid] = {"cold_ms": cold, "warm_ms": warm}
    return out


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cudf_summary, cudf_metadata = load_datafusion_nexus(args.datafusion_nexus_tsv)
    reference = load_reference_timings(args.reference_run_dir)
    duckdb = load_duckdb_timings(args.reference_run_dir)

    out_tsv = args.output_dir / "tpcds_sf100_three_way_comparison.tsv"
    rows_out = []
    query_ids = [q.strip() for q in args.query_ids.split(",") if q.strip()]
    for q_num in query_ids:
        qid = f"q{int(q_num):02d}"
        reference_num = q_num  # reference timings use bare numbers ("3", "7", ...)

        cpu = cudf_summary.get(("cpu_datafusion", qid), {})
        native = cudf_summary.get(("native_gpu", qid), {})
        s = reference.get(reference_num, {})
        d = duckdb.get(reference_num, {})

        speedup_native_vs_cpu = safe_ratio(cpu.get("mean_ms"), native.get("mean_ms"))
        speedup_native_vs_reference_warm = safe_ratio(s.get("warm_ms"), native.get("mean_ms"))
        speedup_reference_warm_vs_native = safe_ratio(native.get("mean_ms"), s.get("warm_ms"))
        reference_wins = False
        notes = []
        if native.get("mean_ms") is not None and s.get("warm_ms") is not None:
            if s["warm_ms"] < native["mean_ms"]:
                reference_wins = True
                notes.append(f"reference_warm_{s['warm_ms']:.0f}ms < native_{native['mean_ms']:.0f}ms")
        if native.get("outcome") and native["outcome"] != "succeeded":
            notes.append(f"native_outcome={native['outcome']}")
        if native.get("failure"):
            notes.append(f"native_failure={native['failure'][:120]}")

        rows_out.append({
            "query_id": qid,
            "cpu_df_mean_ms": fmt(cpu.get("mean_ms")),
            "cpu_df_iters": cpu.get("iters", 0),
            "native_gpu_mean_ms": fmt(native.get("mean_ms")),
            "native_gpu_iters": native.get("iters", 0),
            "native_gpu_outcome": native.get("outcome", "no_data"),
            "native_gpu_reservation_peak_bytes": fmt_int(native.get("reservation_peak")),
            "reference_cold_ms": fmt(s.get("cold_ms")),
            "reference_warm_ms": fmt(s.get("warm_ms")),
            "duckdb_cold_ms": fmt(d.get("cold_ms")),
            "duckdb_warm_ms": fmt(d.get("warm_ms")),
            "speedup_native_vs_cpu_df": fmt(speedup_native_vs_cpu, decimals=2),
            "speedup_native_vs_reference_warm": fmt(speedup_native_vs_reference_warm, decimals=2),
            "speedup_reference_warm_vs_native": fmt(speedup_reference_warm_vs_native, decimals=2),
            "reference_wins_vs_native": "true" if reference_wins else "false",
            "notes": "; ".join(notes),
        })

    with out_tsv.open("w") as f:
        f.write("\t".join(COMPARISON_HEADER) + "\n")
        for row in rows_out:
            f.write("\t".join(str(row[h]) for h in COMPARISON_HEADER) + "\n")
    print(f"wrote {out_tsv}")

    write_status_note(args.output_dir / "STATUS.md", rows_out, args.reference_run_dir, args.datafusion_nexus_tsv, cudf_metadata)
    return 0


def fmt(value, decimals: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def fmt_int(value) -> str:
    if value is None:
        return ""
    return str(int(value))


def profile_status_line(metadata: dict[str, str]) -> str:
    profile_name = metadata.get("tpcds_benchmark_native_profile_name", "unknown")
    device_budget = metadata.get("tpcds_benchmark_native_device_budget_bytes", "unknown")
    chunk_bytes = metadata.get("tpcds_benchmark_native_max_source_chunk_bytes", "unknown")
    row_groups = metadata.get("tpcds_benchmark_native_max_row_groups_per_chunk", "unknown")
    source_chunk_planning_mode = metadata.get("tpcds_benchmark_native_source_chunk_planning_mode", "unknown")
    return (
        f"- datafusion-nexus native profile: {profile_name}; "
        f"device_budget_bytes={device_budget}; "
        f"max_source_chunk_bytes={chunk_bytes}; "
        f"max_row_groups_per_chunk={row_groups}; "
        f"source_chunk_planning_mode={source_chunk_planning_mode}"
    )


def write_status_note(path: Path, rows: list[dict], reference_run_dir: Path, cudf_tsv: Path, cudf_metadata: dict[str, str]) -> None:
    reference_wins = [r for r in rows if r["reference_wins_vs_native"] == "true"]
    native_failures = [r for r in rows if r["native_gpu_outcome"] != "succeeded"]
    lines = [
        "# TPC-DS SF100 three-way bench — datafusion-nexus vs CPU DataFusion vs reference",
        "",
        f"- reference run dir: `{reference_run_dir}`",
        f"- datafusion-nexus bench TSV: `{cudf_tsv}`",
        f"- 15 reference-supported queries: q03 q07 q22 q26 q32 q37 q42 q52 q55 q62 q82 q85 q92 q93 q97",
        profile_status_line(cudf_metadata),
        f"- CPU DataFusion config: target_partitions=16, batch_size=8192, repartitioning on, dynamic filter pushdown on, parquet pushdown filters off",
        f"- reference config: 95% GPU usage limit, 32 GB host pool, 5 GiB scan/hash batch sizes",
        "",
        "## Queries where reference (warm) beats datafusion-nexus native (warm)",
        "",
    ]
    if reference_wins:
        lines.append("| query_id | reference_warm_ms | native_gpu_ms | reference / native | notes |")
        lines.append("|----------|----------------|---------------|-----------------|-------|")
        for r in reference_wins:
            lines.append(
                f"| {r['query_id']} | {r['reference_warm_ms']} | {r['native_gpu_mean_ms']} | "
                f"{r['speedup_reference_warm_vs_native']}x | {r['notes']} |"
            )
    else:
        lines.append("- _(none on this run)_")
    lines += ["", "## Queries where datafusion-nexus native did not complete", ""]
    if native_failures:
        for r in native_failures:
            lines.append(f"- **{r['query_id']}**: outcome={r['native_gpu_outcome']}; notes={r['notes']}")
    else:
        lines.append("- _(none — all 15 queries completed on native GPU)_")
    lines += ["", "## All rows", "", "See `tpcds_sf100_three_way_comparison.tsv` for the full per-query data."]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
