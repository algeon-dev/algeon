#!/usr/bin/env python3
"""Stitch datafusion-nexus native + CPU DataFusion + external-engine TPC-DS timings into a
single comparison TSV plus a short markdown status note.

Inputs:
  --datafusion-nexus-tsv  TSV emitted by src/bin/tpcds_three_way_bench.rs
                         (engine column carries native_gpu + cpu_datafusion).
  --external-engine-run-dir       external-engine run directory produced by
                         external-engine/test/tpcds_performance/benchmark_and_validate.sh,
                         containing external-engine/timings.csv (+ optionally duckdb/timings.csv).
  --output-dir           Where to write the comparison TSV and status note.

The script reports mean wall_ms across measured iterations (skipping warmups) for
the two datafusion-nexus engines, and parses external-engine timings.csv for the cold/warm
columns. It then derives speedup ratios and tags external-engine-wins for follow-up.
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
    "external-engine_cold_ms",
    "external-engine_warm_ms",
    "duckdb_cold_ms",
    "duckdb_warm_ms",
    "speedup_native_vs_cpu_df",
    "speedup_native_vs_external-engine_warm",
    "speedup_external-engine_warm_vs_native",
    "external-engine_wins_vs_native",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafusion-nexus-tsv", required=True, type=Path)
    parser.add_argument("--external-engine-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query-ids", default="3,7,22,26,32,37,42,52,55,62,82,85,92,93,97",
                        help="Comma-separated TPC-DS query numbers (without 'q' prefix).")
    return parser.parse_args()


def load_datafusion_nexus(tsv_path: Path) -> dict[tuple[str, str], dict]:
    """Aggregate per-(engine, query_id) mean wall_ms over measured iterations."""
    rows = defaultdict(list)
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
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
    return summary


def load_external-engine_timings(run_dir: Path) -> dict[str, dict]:
    timings_csv = run_dir / "external-engine" / "timings.csv"
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

    cudf_summary = load_datafusion_nexus(args.datafusion_nexus_tsv)
    external-engine = load_external-engine_timings(args.external-engine_run_dir)
    duckdb = load_duckdb_timings(args.external-engine_run_dir)

    out_tsv = args.output_dir / "tpcds_sf100_three_way_comparison.tsv"
    rows_out = []
    query_ids = [q.strip() for q in args.query_ids.split(",") if q.strip()]
    for q_num in query_ids:
        qid = f"q{int(q_num):02d}"
        external-engine_num = q_num  # external-engine timings use bare numbers ("3", "7", ...)

        cpu = cudf_summary.get(("cpu_datafusion", qid), {})
        native = cudf_summary.get(("native_gpu", qid), {})
        s = external-engine.get(external-engine_num, {})
        d = duckdb.get(external-engine_num, {})

        speedup_native_vs_cpu = safe_ratio(cpu.get("mean_ms"), native.get("mean_ms"))
        speedup_native_vs_external-engine_warm = safe_ratio(s.get("warm_ms"), native.get("mean_ms"))
        speedup_external-engine_warm_vs_native = safe_ratio(native.get("mean_ms"), s.get("warm_ms"))
        external-engine_wins = False
        notes = []
        if native.get("mean_ms") is not None and s.get("warm_ms") is not None:
            if s["warm_ms"] < native["mean_ms"]:
                external-engine_wins = True
                notes.append(f"external-engine_warm_{s['warm_ms']:.0f}ms < native_{native['mean_ms']:.0f}ms")
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
            "external-engine_cold_ms": fmt(s.get("cold_ms")),
            "external-engine_warm_ms": fmt(s.get("warm_ms")),
            "duckdb_cold_ms": fmt(d.get("cold_ms")),
            "duckdb_warm_ms": fmt(d.get("warm_ms")),
            "speedup_native_vs_cpu_df": fmt(speedup_native_vs_cpu, decimals=2),
            "speedup_native_vs_external-engine_warm": fmt(speedup_native_vs_external-engine_warm, decimals=2),
            "speedup_external-engine_warm_vs_native": fmt(speedup_external-engine_warm_vs_native, decimals=2),
            "external-engine_wins_vs_native": "true" if external-engine_wins else "false",
            "notes": "; ".join(notes),
        })

    with out_tsv.open("w") as f:
        f.write("\t".join(COMPARISON_HEADER) + "\n")
        for row in rows_out:
            f.write("\t".join(str(row[h]) for h in COMPARISON_HEADER) + "\n")
    print(f"wrote {out_tsv}")

    write_status_note(args.output_dir / "STATUS.md", rows_out, args.external-engine_run_dir, args.datafusion_nexus_tsv)
    return 0


def fmt(value, decimals: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def fmt_int(value) -> str:
    if value is None:
        return ""
    return str(int(value))


def write_status_note(path: Path, rows: list[dict], external-engine_run_dir: Path, cudf_tsv: Path) -> None:
    external-engine_wins = [r for r in rows if r["external-engine_wins_vs_native"] == "true"]
    native_failures = [r for r in rows if r["native_gpu_outcome"] != "succeeded"]
    lines = [
        "# TPC-DS SF100 three-way bench — datafusion-nexus vs CPU DataFusion vs external-engine",
        "",
        f"- external-engine run dir: `{external-engine_run_dir}`",
        f"- datafusion-nexus bench TSV: `{cudf_tsv}`",
        f"- 15 external-engine-supported queries: q03 q07 q22 q26 q32 q37 q42 q52 q55 q62 q82 q85 q92 q93 q97",
        f"- datafusion-nexus bench profile: 1 warmup + 2 measured per query, rg1024/32GiB source profile, 95 GiB device budget",
        f"- CPU DataFusion config: target_partitions=16, batch_size=8192, repartitioning on, dynamic filter pushdown on, parquet pushdown filters off",
        f"- external-engine config: 95% GPU usage limit, 32 GB host pool, 5 GiB scan/hash batch sizes",
        "",
        "## Queries where external-engine (warm) beats datafusion-nexus native (warm)",
        "",
    ]
    if external-engine_wins:
        lines.append("| query_id | external-engine_warm_ms | native_gpu_ms | external-engine / native | notes |")
        lines.append("|----------|----------------|---------------|-----------------|-------|")
        for r in external-engine_wins:
            lines.append(
                f"| {r['query_id']} | {r['external-engine_warm_ms']} | {r['native_gpu_mean_ms']} | "
                f"{r['speedup_external-engine_warm_vs_native']}x | {r['notes']} |"
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
