#!/usr/bin/env python3
"""Summarize a native_tpch_benchmark TSV into a CPU-vs-GPU review table.

Reads the TSV emitted by `native_tpch_benchmark` (schema v32+), prints:
  - per-query CPU/GPU wall, speedup, parity, final disposition, dominant runtime
    class and the GPU time breakdown that explains where the wall went;
  - overall totals (sum CPU, sum GPU, overall ratio, peak live reservation);
  - a flagged list of queries where GPU is slower or anomalous.

Usage: tpch_bench_review.py <tsv> [--slow-threshold 2.0]
"""
from __future__ import annotations
import sys, argparse

# The TSV header is frozen for downstream compatibility, but its values are
# FinalPlanDisposition labels: datafusion, native, mixed, or rejected.
FINAL_DISPOSITION_TSV_COLUMN = "post_native_execution_path"

def load(tsv):
    meta = {}
    hdr = None
    rows = []
    for line in open(tsv):
        line = line.rstrip("\n")
        if line.startswith("#"):
            body = line[1:].strip()
            if "=" in body:
                k, _, v = body.partition("=")
                meta[k.strip()] = v.strip()
            continue
        cells = line.split("\t")
        if hdr is None:
            hdr = cells
            continue
        rows.append(dict(zip(hdr, cells)))
    return meta, hdr, rows

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def fmt_ms(x):
    v = f(x)
    return f"{v:8.1f}" if v is not None else "     n/a"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--slow-threshold", type=float, default=2.0,
                    help="flag queries with speedup below this")
    args = ap.parse_args()
    meta, hdr, rows = load(args.tsv)

    print(f"dataset: scale={meta.get('workload_dataset_scale_factor', meta.get('tpch_benchmark_dataset_mode','?'))} "
          f"iterations={meta.get('tpch_benchmark_iterations','?')} warmups={meta.get('tpch_benchmark_warmups','?')} "
          f"memory_policy={meta.get('tpch_benchmark_native_memory_policy','?')} "
          f"exec_mode={meta.get('tpch_benchmark_native_pipeline_execution_mode','?')}")
    print()
    # header
    cols = ["query", "disposition", "parity", "cpu_ms", "gpu_ms", "speedup", "rt_class",
            "src_read", "operator", "fmerge", "fsort", "matzn", "rsv_peak_GiB", "retry", "denied"]
    print("  ".join(c.ljust(w) for c, w in zip(cols,
          [5,14,9,9,9,7,16,9,9,8,8,8,12,5,6])))

    total_cpu = total_gpu = 0.0
    peak_rsv = 0
    flagged = []
    per = []
    for r in rows:
        q = r["query_id"]
        cpu = f(r["cpu_wall_ms"]); gpu = f(r["native_wall_ms"]); sp = f(r["speedup_vs_cpu"])
        disposition = r.get(FINAL_DISPOSITION_TSV_COLUMN, "?")
        parity = r.get("parity_status", "?")
        rt = r.get("runtime_operator_class", "?")
        rsv = f(r.get("reservation_live_peak_bytes")) or 0
        peak_rsv = max(peak_rsv, rsv)
        retry = r.get("retry_count", "0")
        denied = r.get("reservation_denied_count", "0")
        if cpu: total_cpu += cpu
        if gpu: total_gpu += gpu
        rsv_gib = rsv / (1024**3)
        print("  ".join([
            q.ljust(5), disposition.ljust(14), parity.ljust(9),
            fmt_ms(cpu), fmt_ms(gpu),
            (f"{sp:6.2f}" if sp is not None else "   n/a").ljust(7),
            rt[:16].ljust(16),
            fmt_ms(r.get("source_read_time_ms")), fmt_ms(r.get("operator_time_ms")),
            fmt_ms(r.get("final_merge_time_ms")), fmt_ms(r.get("final_sort_time_ms")),
            fmt_ms(r.get("materialization_time_ms")),
            f"{rsv_gib:10.2f}", str(retry).rjust(5), str(denied).rjust(6),
        ]))
        per.append((q, cpu, gpu, sp, disposition, parity, rt))
        if sp is None or sp < args.slow_threshold or disposition != "native" or parity != "matched":
            flagged.append((q, cpu, gpu, sp, disposition, parity, rt))

    print()
    overall = (total_cpu / total_gpu) if total_gpu else float("nan")
    print(f"TOTAL  cpu={total_cpu/1000:7.2f}s  gpu={total_gpu/1000:7.2f}s  "
          f"overall_speedup={overall:5.2f}x  peak_live_reservation={peak_rsv/(1024**3):.2f} GiB")
    n_gpu_faster = sum(1 for p in per if p[3] is not None and p[3] > 1.0)
    print(f"queries GPU faster (speedup>1): {n_gpu_faster}/{len(per)}")
    print()
    print(f"FLAGGED (speedup<{args.slow_threshold} or final disposition is not native or parity!=matched):")
    for q, cpu, gpu, sp, disposition, parity, rt in sorted(flagged, key=lambda x: (x[3] if x[3] is not None else -1)):
        print(f"  {q}: speedup={sp if sp is not None else 'n/a'} disposition={disposition} parity={parity} "
              f"rt_class={rt} cpu={cpu}ms gpu={gpu}ms")

if __name__ == "__main__":
    main()
