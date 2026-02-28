#!/usr/bin/env python3
"""Compare TPC-DS CPU DataFusion vs native GPU from two tpcds_three_way_bench TSVs.

Each TSV has one row per (engine, query, iteration) with a `warmup` flag. We take
the mean wall_ms over measured (warmup=false) iterations per query for each engine,
then derive per-query GPU-vs-CPU speedup, parity of output rows, GPU outcome/path,
and a flagged list of queries where GPU loses or fails.

Usage: tpcds_cpu_gpu_compare.py --gpu-tsv gpu.tsv --cpu-tsv cpu.tsv [--slow-threshold 1.0]
"""
from __future__ import annotations
import argparse, statistics
import sys
from collections import defaultdict

def load(path):
    hdr = None
    metadata = {}
    by_q = defaultdict(lambda: {"measured": [], "rows": None, "outcome": None,
                                 "planning": None, "rsv": 0, "started": None,
                                 "failure": None, "rt_class": None})
    with open(path) as handle:
        lines = list(handle)
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith("#"):
            key, value = parse_metadata_comment(line)
            if key:
                metadata[key] = value
            continue
        cells = line.split("\t")
        if hdr is None:
            hdr = cells
            idx = {c: i for i, c in enumerate(hdr)}
            continue
        g = lambda c: cells[idx[c]] if c in idx and idx[c] < len(cells) else None
        q = g("query_id")
        warmup = (g("warmup") or "").lower() == "true"
        try:
            wall = float(g("wall_ms"))
        except (TypeError, ValueError):
            wall = None
        rec = by_q[q]
        if not warmup and wall is not None and (g("outcome") == "succeeded"):
            rec["measured"].append(wall)
        rec["outcome"] = g("outcome")
        rec["planning"] = g("planning_outcome")
        rec["started"] = g("native_runtime_started")
        rec["failure"] = g("failure")
        rec["rt_class"] = g("runtime_operator_class")
        try:
            rec["rows"] = int(g("output_rows"))
        except (TypeError, ValueError):
            pass
        try:
            rec["rsv"] = max(rec["rsv"], int(g("reservation_live_peak_bytes")))
        except (TypeError, ValueError):
            pass
    return by_q, metadata

def parse_metadata_comment(line):
    text = line[1:].strip()
    if "=" not in text:
        return None, None
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None, None
    return key, value.strip()

def native_profile_summary(metadata):
    name = metadata.get("tpcds_benchmark_native_profile_name", "unknown")
    source = metadata.get("tpcds_benchmark_native_profile_source", "unknown")
    device_budget = metadata.get("tpcds_benchmark_native_device_budget_bytes", "unknown")
    chunk_bytes = metadata.get("tpcds_benchmark_native_max_source_chunk_bytes", "unknown")
    row_groups = metadata.get("tpcds_benchmark_native_max_row_groups_per_chunk", "unknown")
    source_chunk_planning_mode = metadata.get("tpcds_benchmark_native_source_chunk_planning_mode", "unknown")
    return {
        "name": name,
        "source": source,
        "device_budget": device_budget,
        "chunk_bytes": chunk_bytes,
        "row_groups": row_groups,
        "source_chunk_planning_mode": source_chunk_planning_mode,
        "known": name != "unknown",
    }

def print_profile_summary(label, metadata):
    profile = native_profile_summary(metadata)
    print(
        f"{label} native profile: name={profile['name']} source={profile['source']} "
        f"device_budget_bytes={profile['device_budget']} "
        f"max_source_chunk_bytes={profile['chunk_bytes']} "
        f"max_row_groups_per_chunk={profile['row_groups']} "
        f"source_chunk_planning_mode={profile['source_chunk_planning_mode']}"
    )
    return profile

def mean(xs):
    return statistics.mean(xs) if xs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-tsv", required=True)
    ap.add_argument("--cpu-tsv", required=True)
    ap.add_argument("--slow-threshold", type=float, default=1.0)
    args = ap.parse_args()
    gpu, gpu_metadata = load(args.gpu_tsv)
    cpu, cpu_metadata = load(args.cpu_tsv)
    queries = sorted(set(gpu) | set(cpu), key=lambda q: (len(q), q))

    gpu_profile = print_profile_summary("GPU", gpu_metadata)
    print_profile_summary("CPU TSV", cpu_metadata)
    if not gpu_profile["known"]:
        print(
            "WARNING: GPU TSV does not include tpcds_benchmark_native_profile_name; "
            "CPU-vs-GPU comparison profile is unknown.",
            file=sys.stderr,
        )
    print()

    print(f"{'query':6} {'cpu_ms':>9} {'gpu_ms':>9} {'speedup':>7} {'gpu_started':>11} "
          f"{'gpu_outcome':>12} {'rows_match':>10} {'rt_class':>16} {'rsv_GiB':>8}")
    n_gpu_faster = n_gpu_ok = 0
    flagged = []
    tot_cpu = tot_gpu = 0.0
    rows_out = []
    for q in queries:
        gc = gpu.get(q, {}); cc = cpu.get(q, {})
        gm = mean(gc.get("measured", [])); cm = mean(cc.get("measured", []))
        sp = (cm / gm) if (gm and cm) else None
        started = gc.get("started")
        outcome = gc.get("outcome")
        gpu_rows = gc.get("rows"); cpu_rows = cc.get("rows")
        rows_match = "n/a"
        if gpu_rows is not None and cpu_rows is not None:
            rows_match = "yes" if gpu_rows == cpu_rows else f"NO({gpu_rows}/{cpu_rows})"
        rsv = gc.get("rsv", 0) / (1024**3)
        if outcome == "succeeded":
            n_gpu_ok += 1
        if sp and sp > 1.0:
            n_gpu_faster += 1
        if gm: tot_gpu += gm
        if cm: tot_cpu += cm
        print(f"{q:6} {cm if cm else 0:9.1f} {gm if gm else 0:9.1f} "
              f"{(sp if sp else 0):7.2f} {str(started):>11} {str(outcome):>12} "
              f"{rows_match:>10} {str(gc.get('rt_class'))[:16]:>16} {rsv:8.2f}")
        if sp is None or sp < args.slow_threshold or outcome != "succeeded" or started != "true" or (rows_match.startswith("NO")):
            flagged.append((q, cm, gm, sp, started, outcome, rows_match, gc.get("rt_class"), gc.get("failure")))
        rows_out.append((q, cm, gm, sp))

    print()
    overall = (tot_cpu / tot_gpu) if tot_gpu else float("nan")
    print(f"TOTAL cpu={tot_cpu/1000:.2f}s gpu={tot_gpu/1000:.2f}s overall={overall:.2f}x")
    print(f"queries: {len(queries)}  GPU started ok: {n_gpu_ok}  GPU faster (speedup>1): {n_gpu_faster}")
    print()
    print(f"FLAGGED (speedup<{args.slow_threshold}, gpu not started, failure, or row mismatch):")
    for q, cm, gm, sp, started, outcome, rows_match, rt, failure in sorted(
            flagged, key=lambda x: (x[3] if x[3] is not None else -1)):
        print(f"  {q}: speedup={f'{sp:.2f}' if sp else 'n/a'} started={started} outcome={outcome} "
              f"rows={rows_match} rt={rt} cpu={cm}ms gpu={gm}ms"
              + (f" failure={failure[:120]}" if failure and failure not in ('', 'none') else ""))

if __name__ == "__main__":
    main()
