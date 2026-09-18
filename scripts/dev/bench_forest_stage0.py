#!/usr/bin/env python3
"""Stage 0 benchmark and validation script for nvForest inference evaluation.

Measures:
1. Native source predictor (LightGBM / XGBoost CPU 16 threads)
2. Treelite GTIL (CPU 16 threads)
3. nvForest (Blackwell GPU)

Reports per-model:
- Model metadata: format, trees, features, classes, postprocessor, precision
- Accuracy metrics: max difference, mean absolute error against source predictor and GTIL
- Latency breakdown: import time, host-to-device transfer, GPU kernel predict, device-to-host transfer
- Throughput (rows/sec)
- GPU memory peak (nvidia-smi and device memory)
- Blackwell 96 GB grant feasibility analysis using section 4.7 formula
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp
import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
import treelite
import treelite.gtil
import xgboost as xgb
import nvforest


def get_gpu_memory_used_mib() -> int:
    """Query current GPU memory used in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human readable string."""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PiB"


def run_benchmark_model(
    model_name: str,
    model_type: str,
    model_path: str,
    X_sample: np.ndarray,
    batch_sizes: List[int],
    nthreads: int = 16,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark a single model across multiple batch sizes."""
    print(f"\n========================================================")
    print(f"Benchmarking: {model_name} (type: {model_type})")
    print(f"========================================================")

    model_results: Dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "model_path": model_path,
        "batch_benchmarks": [],
    }

    # 1. Load into Treelite and measure import time
    t0 = time.perf_counter()
    if model_type == "lightgbm":
        tl_model = treelite.frontend.load_lightgbm_model(model_path)
    elif model_type == "xgboost":
        tl_model = treelite.frontend.load_xgboost_model(model_path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    tl_load_time_sec = time.perf_counter() - t0

    num_trees = tl_model.num_tree
    num_features = tl_model.num_feature
    hdr = tl_model.get_header_accessor()
    postprocessor = hdr.get_field("postprocessor")
    class_ids = set(hdr.get_field("class_id"))
    num_classes = len(class_ids)

    model_results["num_trees"] = num_trees
    model_results["num_features"] = num_features
    model_results["postprocessor"] = postprocessor
    model_results["num_classes"] = num_classes
    model_results["treelite_load_time_sec"] = tl_load_time_sec

    # 2. Load into Source Predictor
    t0 = time.perf_counter()
    if model_type == "lightgbm":
        with open(model_path, "rb") as f:
            magic = f.read(4)
        if magic.startswith(b"\x80"):
            import pickle
            with open(model_path, "rb") as f:
                source_model = pickle.load(f)
        else:
            source_model = lgb.Booster(model_file=model_path)
    else:
        source_model = xgb.Booster()
        source_model.load_model(model_path)
        source_model.set_param({"nthread": nthreads})
    source_load_time_sec = time.perf_counter() - t0
    model_results["source_load_time_sec"] = source_load_time_sec

    # 3. Import into nvForest and measure GPU import time
    mem_before = get_gpu_memory_used_mib()
    t0 = time.perf_counter()
    nv_model = nvforest.load_from_treelite_model(tl_model)
    cp.cuda.Stream.null.synchronize()
    nv_import_time_sec = time.perf_counter() - t0
    mem_after = get_gpu_memory_used_mib()

    forest_dtype = str(nv_model.forest.get_dtype())
    is_double_forest = "float64" in forest_dtype
    model_results["nvforest_import_time_sec"] = nv_import_time_sec
    model_results["nvforest_num_outputs"] = nv_model.num_outputs
    model_results["forest_dtype"] = forest_dtype
    model_results["is_double_forest"] = is_double_forest
    model_results["import_gpu_mem_delta_mib"] = max(0, mem_after - mem_before)

    print(f"Trees: {num_trees}, Features: {num_features}, Classes: {num_classes}, Postproc: {postprocessor}")
    print(f"Treelite load: {tl_load_time_sec*1000:.2f} ms | nvForest import: {nv_import_time_sec*1000:.2f} ms | Forest dtype: {forest_dtype}")

    # 4. Evaluate each batch size
    valid_batch_sizes = [bs for bs in batch_sizes if bs <= X_sample.shape[0]]
    if not valid_batch_sizes:
        valid_batch_sizes = [X_sample.shape[0]]

    for bs in valid_batch_sizes:
        X_batch = X_sample[:bs].copy()
        print(f"\n--- Batch size: {bs:,} rows ---")

        # Prepare input for nvForest (Float32 -> Float64 upcast if double forest)
        X_nv_input = X_batch.astype(np.float64) if is_double_forest else X_batch

        # --- A. Source Predictor (CPU 16 threads) ---
        source_latencies = []
        pred_source = None
        for r in range(num_runs + 1):
            t0 = time.perf_counter()
            if model_type == "lightgbm":
                if isinstance(source_model, list):
                    p = source_model[0].predict(X_batch, num_threads=nthreads)
                else:
                    p = source_model.predict(X_batch, num_threads=nthreads)
            else:
                p = source_model.predict(xgb.DMatrix(X_batch))
            lat = time.perf_counter() - t0
            if r > 0:  # skip warmup
                source_latencies.append(lat)
            pred_source = p

        med_source_sec = float(np.median(source_latencies))

        # --- B. Treelite GTIL (CPU 16 threads) ---
        gtil_latencies = []
        pred_gtil = None
        for r in range(num_runs + 1):
            t0 = time.perf_counter()
            p = treelite.gtil.predict(tl_model, X_batch, nthread=nthreads)
            lat = time.perf_counter() - t0
            if r > 0:
                gtil_latencies.append(lat)
            pred_gtil = p

        med_gtil_sec = float(np.median(gtil_latencies))

        # --- C. nvForest GPU ---
        h2d_latencies = []
        kernel_latencies = []
        d2h_latencies = []
        total_gpu_latencies = []
        pred_nv_host = None

        for r in range(num_runs + 1):
            cp.cuda.Stream.null.synchronize()

            # H2D
            t0 = time.perf_counter()
            X_gpu = cp.asarray(X_nv_input)
            cp.cuda.Stream.null.synchronize()
            t_h2d = time.perf_counter() - t0

            # Kernel predict
            t1 = time.perf_counter()
            if hasattr(nv_model, "predict_proba") and postprocessor in ["sigmoid", "softmax"]:
                out_gpu = nv_model.predict_proba(X_gpu)
            else:
                out_gpu = nv_model.predict(X_gpu)
            cp.cuda.Stream.null.synchronize()
            t_kernel = time.perf_counter() - t1

            # D2H
            t2 = time.perf_counter()
            p_host = cp.asnumpy(out_gpu)
            t_d2h = time.perf_counter() - t2

            t_total = t_h2d + t_kernel + t_d2h

            if r > 0:
                h2d_latencies.append(t_h2d)
                kernel_latencies.append(t_kernel)
                d2h_latencies.append(t_d2h)
                total_gpu_latencies.append(t_total)
            pred_nv_host = p_host

        med_h2d_sec = float(np.median(h2d_latencies))
        med_kernel_sec = float(np.median(kernel_latencies))
        med_d2h_sec = float(np.median(d2h_latencies))
        med_total_gpu_sec = float(np.median(total_gpu_latencies))
        peak_gpu_mem = get_gpu_memory_used_mib()

        # Accuracy comparison
        p_src = np.squeeze(np.asarray(pred_source))
        p_gtil = np.squeeze(np.asarray(pred_gtil))
        p_nv = np.squeeze(np.asarray(pred_nv_host))

        # If nvForest predict_proba output shape is (N, 1), squeeze to (N,)
        if p_nv.ndim > 1 and p_nv.shape[-1] == 1:
            p_nv = np.squeeze(p_nv)
        if p_src.ndim > 1 and p_src.shape[-1] == 1:
            p_src = np.squeeze(p_src)
        if p_gtil.ndim > 1 and p_gtil.shape[-1] == 1:
            p_gtil = np.squeeze(p_gtil)

        max_diff_gtil_src = float(np.max(np.abs(p_gtil - p_src)))
        max_diff_nv_src = float(np.max(np.abs(p_nv - p_src)))
        max_diff_nv_gtil = float(np.max(np.abs(p_nv - p_gtil)))
        mae_nv_src = float(np.mean(np.abs(p_nv - p_src)))

        speedup_kernel_vs_src = med_source_sec / med_kernel_sec if med_kernel_sec > 0 else 0
        speedup_total_vs_src = med_source_sec / med_total_gpu_sec if med_total_gpu_sec > 0 else 0
        speedup_kernel_vs_gtil = med_gtil_sec / med_kernel_sec if med_kernel_sec > 0 else 0
        speedup_total_vs_gtil = med_gtil_sec / med_total_gpu_sec if med_total_gpu_sec > 0 else 0

        throughput_kernel_rps = bs / med_kernel_sec if med_kernel_sec > 0 else 0
        throughput_total_rps = bs / med_total_gpu_sec if med_total_gpu_sec > 0 else 0

        print(
            f"Source CPU (16t): {med_source_sec*1000:.2f} ms | "
            f"Treelite GTIL (16t): {med_gtil_sec*1000:.2f} ms | "
            f"nvForest GPU: {med_kernel_sec*1000:.2f} ms (kernel), {med_total_gpu_sec*1000:.2f} ms (total)"
        )
        print(
            f"H2D: {med_h2d_sec*1000:.2f} ms | D2H: {med_d2h_sec*1000:.2f} ms | "
            f"Kernel speedup: {speedup_kernel_vs_src:.2f}x (vs source), {speedup_kernel_vs_gtil:.2f}x (vs GTIL)"
        )
        print(
            f"Throughput: {throughput_kernel_rps:,.0f} rows/s (kernel), {throughput_total_rps:,.0f} rows/s (E2E) | "
            f"Max diff: nv-vs-src={max_diff_nv_src:.2e}, nv-vs-gtil={max_diff_nv_gtil:.2e}"
        )

        batch_result = {
            "batch_size": bs,
            "source_latency_sec": med_source_sec,
            "gtil_latency_sec": med_gtil_sec,
            "gpu_h2d_latency_sec": med_h2d_sec,
            "gpu_kernel_latency_sec": med_kernel_sec,
            "gpu_d2h_latency_sec": med_d2h_sec,
            "gpu_total_latency_sec": med_total_gpu_sec,
            "speedup_kernel_vs_source": speedup_kernel_vs_src,
            "speedup_total_vs_source": speedup_total_vs_src,
            "speedup_kernel_vs_gtil": speedup_kernel_vs_gtil,
            "speedup_total_vs_gtil": speedup_total_vs_gtil,
            "throughput_kernel_rows_per_sec": throughput_kernel_rps,
            "throughput_total_rows_per_sec": throughput_total_rps,
            "max_diff_nv_vs_source": max_diff_nv_src,
            "max_diff_nv_vs_gtil": max_diff_nv_gtil,
            "max_diff_gtil_vs_source": max_diff_gtil_src,
            "mae_nv_vs_source": mae_nv_src,
            "peak_gpu_mem_mib": peak_gpu_mem,
        }
        model_results["batch_benchmarks"].append(batch_result)

    return model_results


def calculate_grant_feasibility(
    num_rows: int,
    num_features: int,
    is_double_forest: bool,
    native_output_stride: int,
    model_nodes_bytes: int = 10 * 1024 * 1024,
) -> Dict[str, Any]:
    """Calculate device memory requirement using Section 4.7 formula:

    rows × features × (4 + 8 × [double forest]) + rows × native_output_stride × 4 + model_nodes
    """
    input_bytes = num_rows * num_features * 4
    staging_bytes = num_rows * num_features * 8 if is_double_forest else 0
    output_bytes = num_rows * native_output_stride * 4
    total_bytes = input_bytes + staging_bytes + output_bytes + model_nodes_bytes

    return {
        "num_rows": num_rows,
        "num_features": num_features,
        "is_double_forest": is_double_forest,
        "native_output_stride": native_output_stride,
        "input_bytes": input_bytes,
        "staging_bytes": staging_bytes,
        "output_bytes": output_bytes,
        "model_nodes_bytes": model_nodes_bytes,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/bench/forest_predict_20260925"))
    parser.add_argument("--max-rows-synthetic", type=int, default=10_000_000)
    parser.add_argument("--ember-rows", type=int, default=606_000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load real EMBER2024 features
    print("Loading EMBER2024 test features...")
    ember_dat_path = Path("fixture/ember2024/X_test.dat")
    if ember_dat_path.exists() and ember_dat_path.stat().st_size > 0:
        nrows_test = ember_dat_path.stat().st_size // (2568 * 4)
        print(f"Reading from X_test.dat ({nrows_test} rows)...")
        mmap_test = np.memmap(ember_dat_path, dtype=np.float32, mode="r", shape=(nrows_test, 2568))
        sample_size = min(args.ember_rows, nrows_test)
        X_ember = np.array(mmap_test[:sample_size], dtype=np.float32)
    else:
        # Fallback to direct extraction from JSONL
        print("Extracting features directly from test JSONL files...")
        ext = thrember.PEFeatureExtractor() if "thrember" in globals() else None
        if ext is None:
            import thrember
            ext = thrember.PEFeatureExtractor()
        samples = []
        jsonl_files = sorted(glob.glob("fixture/ember2024/*test*.jsonl"))
        for jf in jsonl_files:
            with open(jf, "r") as f:
                for line in f:
                    samples.append(ext.process_raw_features(json.loads(line)))
                    if len(samples) >= args.ember_rows:
                        break
            if len(samples) >= args.ember_rows:
                break
        X_ember = np.array(samples, dtype=np.float32)

    print(f"Loaded EMBER test features: {X_ember.shape}, dtype={X_ember.dtype}")

    # 2. Load TPC-H SF10 Lineitem features
    print("Loading TPC-H SF10 Lineitem features...")
    table = pq.read_table(
        "fixture/tpc-h/data/sf10/parquet/lineitem.parquet",
        columns=[
            "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
            "l_quantity", "l_discount", "l_tax", "l_extendedprice"
        ],
    )
    n_sample_lineitem = min(args.max_rows_synthetic, table.num_rows)
    feature_names = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_discount", "l_tax"]
    X_lineitem = np.column_stack([
        table.column(c).slice(0, n_sample_lineitem).to_numpy().astype(np.float32)
        for c in feature_names
    ])
    print(f"Loaded Lineitem features: {X_lineitem.shape}, dtype={X_lineitem.dtype}")

    all_benchmarks = []

    # 3. Benchmark Synthetic XGBoost model
    xgb_json_path = "fixture/tpc-h/models/lineitem_xgboost_500t_d8.json"
    if os.path.exists(xgb_json_path):
        res_xgb = run_benchmark_model(
            model_name="lineitem_xgboost_500t_d8",
            model_type="xgboost",
            model_path=xgb_json_path,
            X_sample=X_lineitem,
            batch_sizes=[10_000, 100_000, 1_000_000, 10_000_000],
        )
        all_benchmarks.append(res_xgb)

    # 4. Benchmark all 14 EMBER2024 models
    ember_models = sorted(glob.glob("fixture/ember2024/models/*.model"))
    for m in ember_models:
        m_name = os.path.basename(m).replace(".model", "")
        # For family model (157k trees, 2538 classes), evaluate smaller batch sizes to respect time
        if "family" in m_name:
            batch_sizes = [1_000, 5_000]
        elif m_name == "EMBER2024_all":
            batch_sizes = [10_000, 50_000, 100_000, 606_000]
        else:
            batch_sizes = [10_000, 50_000, 100_000]

        res = run_benchmark_model(
            model_name=m_name,
            model_type="lightgbm",
            model_path=m,
            X_sample=X_ember,
            batch_sizes=batch_sizes,
        )
        all_benchmarks.append(res)

    # 5. Feasibility Calculations
    feasibility_reports = []
    # EMBER test split
    feasibility_reports.append(
        calculate_grant_feasibility(
            num_rows=606_000,
            num_features=2568,
            is_double_forest=True,
            native_output_stride=1,
            model_nodes_bytes=4 * 1024 * 1024,
        )
    )
    # EMBER train split
    feasibility_reports.append(
        calculate_grant_feasibility(
            num_rows=2_626_000,
            num_features=2568,
            is_double_forest=True,
            native_output_stride=1,
            model_nodes_bytes=4 * 1024 * 1024,
        )
    )
    # EMBER full raw rows (5,252,000)
    feasibility_reports.append(
        calculate_grant_feasibility(
            num_rows=5_252_000,
            num_features=2568,
            is_double_forest=True,
            native_output_stride=1,
            model_nodes_bytes=4 * 1024 * 1024,
        )
    )
    # TPC-H SF10 lineitem full (59,986,052)
    feasibility_reports.append(
        calculate_grant_feasibility(
            num_rows=59_986_052,
            num_features=7,
            is_double_forest=False,
            native_output_stride=1,
            model_nodes_bytes=14 * 1024 * 1024,
        )
    )

    # 6. Save JSON output
    out_json = args.out_dir / "benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "benchmarks": all_benchmarks,
                "feasibility": feasibility_reports,
            },
            f,
            indent=2,
        )
    print(f"\nSaved raw benchmark results to {out_json}")

    # 7. Generate TSV summary
    out_tsv = args.out_dir / "benchmark_summary.tsv"
    with open(out_tsv, "w") as f:
        headers = [
            "model_name", "batch_size", "trees", "features", "classes",
            "source_cpu_16t_ms", "gtil_cpu_16t_ms",
            "gpu_h2d_ms", "gpu_kernel_ms", "gpu_d2h_ms", "gpu_total_ms",
            "kernel_speedup_vs_src", "total_speedup_vs_src",
            "throughput_kernel_rps", "throughput_total_rps",
            "max_diff_nv_vs_src", "max_diff_nv_vs_gtil", "mae_nv_vs_src"
        ]
        f.write("\t".join(headers) + "\n")
        for b in all_benchmarks:
            m_name = b["model_name"]
            trees = b["num_trees"]
            feats = b["num_features"]
            classes = b["num_classes"]
            for bm in b["batch_benchmarks"]:
                row = [
                    m_name,
                    str(bm["batch_size"]),
                    str(trees),
                    str(feats),
                    str(classes),
                    f"{bm['source_latency_sec']*1000:.2f}",
                    f"{bm['gtil_latency_sec']*1000:.2f}",
                    f"{bm['gpu_h2d_latency_sec']*1000:.2f}",
                    f"{bm['gpu_kernel_latency_sec']*1000:.2f}",
                    f"{bm['gpu_d2h_latency_sec']*1000:.2f}",
                    f"{bm['gpu_total_latency_sec']*1000:.2f}",
                    f"{bm['speedup_kernel_vs_source']:.2f}",
                    f"{bm['speedup_total_vs_source']:.2f}",
                    f"{bm['throughput_kernel_rows_per_sec']:.0f}",
                    f"{bm['throughput_total_rows_per_sec']:.0f}",
                    f"{bm['max_diff_nv_vs_source']:.2e}",
                    f"{bm['max_diff_nv_vs_gtil']:.2e}",
                    f"{bm['mae_nv_vs_source']:.2e}",
                ]
                f.write("\t".join(row) + "\n")
    print(f"Saved summary TSV to {out_tsv}")


if __name__ == "__main__":
    main()
