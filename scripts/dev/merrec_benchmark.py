#!/usr/bin/env python3
"""Generate MerRec scale SQL, run an exact CPU kNN baseline, and inspect captures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time


ROOT = Path(__file__).resolve().parents[2]
SCALE = ROOT / "fixture/demo/merrec/scripts/scale"
DEFAULT_BATCHES = (100, 1000, 10000, 100000)
DEFAULT_ENDS = ("2023-06-01", "2023-07-01", "2023-09-01", "2023-11-01")


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def generate(output, tastes, batches, ends, results):
    from datetime import date

    output.mkdir(parents=True, exist_ok=True)
    cases = []

    def write(name, sql, **metadata):
        path = output / f"{name}.sql"
        path.write_text(sql.rstrip() + "\n")
        cases.append({"name": name, "sql_file": str(path.resolve()), **metadata})

    write("register_tastes", "CREATE EXTERNAL TABLE mr_bench_taste64 "
          f"STORED AS PARQUET LOCATION {sql_string(tastes.resolve())};", phase="register")
    for batch in batches:
        if batch < 1:
            raise ValueError("query batches must be positive")
        for algorithm, function, parameters in (
            ("exact", "cuvs_brute_force_knn", "metric => 'l2_expanded'"),
            ("cagra", "cuvs_cagra", "graph_degree => 32, intermediate_graph_degree => 64, "
             "build_algo => 'nn_descent'"),
        ):
            write(f"knn_{algorithm}_{batch}", f"""SELECT query_id, neighbor_id, distance
FROM {function}(
  dataset => (SELECT * FROM mr_bench_taste64 ORDER BY id),
  queries => (SELECT * FROM mr_bench_taste64 ORDER BY id LIMIT {batch}),
  k => 10, {parameters}
);""", family="cuvs", phase="prepared_input",
                  algorithm=algorithm, query_limit=batch, dimensions=64, k=10)

    for end in ends:
        if date.fromisoformat(end) <= date(2023, 5, 1):
            raise ValueError("graph window end must follow 2023-05-01")
        name = f"item_user_{end.replace('-', '')}"
        edges = f"""SELECT item_id AS src, -(user_id + 1) AS dst,
       CAST(COUNT(*) AS DOUBLE) AS weight
FROM merrec_events
WHERE event_type = 'item_view'
  AND event_time >= TIMESTAMP '2023-05-01 00:00:00'
  AND event_time < TIMESTAMP '{end} 00:00:00'
GROUP BY item_id, user_id"""
        write(f"prepare_{name}", edges + ";", family="cugraph", phase="prepare",
              window_start="2023-05-01", window_end=end)
        table = f"mr_bench_{name}"
        write(f"register_{name}", f"CREATE EXTERNAL TABLE {table} "
              "STORED AS PARQUET LOCATION "
              f"{sql_string(results.resolve() / f'prepare_{name}.parquet')};", phase="register")
        for phase, relation in (("e2e", edges), ("prepared_input", f"SELECT * FROM {table}")):
            write(f"pagerank_{name}_{phase}", f"""SELECT vertex, value
FROM cugraph_pagerank(
  edges => ({relation}), directed => false,
  alpha => 0.85, epsilon => 0.00001, max_iterations => 100
)
WHERE vertex >= 0
ORDER BY value DESC, vertex
LIMIT 20;""", family="cugraph", phase=phase,
                  window_start="2023-05-01", window_end=end)

    manifest = {
        "tastes": str(tastes.resolve()), "tastes_sha256": sha256(tastes),
        "prepared_results": str(results.resolve()), "cases": cases,
        "retained_controls": [str(SCALE / file) for file in (
            "10_barbie_tie_ins.sql", "31_world_bridges.sql", "30_audience_funnel.sql")],
        "cuml_harness": "crates/bench/src/forest_predict/mod.rs",
    }
    (output / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recompute_candidate_distances(ids, matrix, query_count, neighbors):
    """Compute Float64 squared distances for supplied candidate IDs only."""
    import numpy as np

    positions = np.searchsorted(ids, neighbors)
    if (positions >= len(ids)).any() or not np.array_equal(ids[positions], neighbors):
        raise ValueError("candidate neighbor IDs are absent from the frozen input")
    distances = np.empty(neighbors.shape, dtype=np.float64)
    for start in range(0, query_count, 1024):
        stop = min(start + 1024, query_count)
        delta = matrix[positions[start:stop]].astype(np.float64) - matrix[start:stop, None].astype(np.float64)
        distances[start:stop] = np.einsum("qkd,qkd->qk", delta, delta)
    return distances


def validate_knn(table, ids, matrix, query_count, expected_distances, k, tolerance, expected_neighbors=None):
    """Compare recomputed candidate distances and independently check reported accuracy.

    expected_distances contains Float64 distances recomputed from the frozen
    input for the CPU candidates. This checks distance equivalence to those
    candidates, without proving exact Float64 top-k over the complete dataset.
    """
    import numpy as np

    columns = {name: table.column(name).to_numpy() for name in ("query_id", "neighbor_id", "distance")}
    if table.num_rows != query_count * k:
        return {"matches": False, "reason": "output_row_count", "rows": table.num_rows}
    order = np.lexsort((columns["neighbor_id"], columns["distance"], columns["query_id"]))
    queries = columns["query_id"][order].reshape(query_count, k)
    neighbors = columns["neighbor_id"][order].reshape(query_count, k)
    distances = columns["distance"][order].reshape(query_count, k)
    if not np.array_equal(queries, np.broadcast_to(ids[:query_count, None], queries.shape)):
        return {"matches": False, "reason": "query_ids"}
    positions = np.searchsorted(ids, neighbors)
    if (positions >= len(ids)).any() or not np.array_equal(ids[positions], neighbors):
        return {"matches": False, "reason": "neighbor_ids"}
    if not np.isfinite(distances).all() or (np.diff(np.sort(neighbors, axis=1), axis=1) == 0).any():
        return {"matches": False, "reason": "invalid_distance_or_duplicate_neighbor"}
    max_error = 0.0
    sorted_error = 0.0
    within = 0
    overlap = 0
    expected_sorted = np.sort(expected_distances, axis=1)
    # Bound correctness-check memory independently from the search batch size.
    for start in range(0, query_count, 1024):
        stop = min(start + 1024, query_count)
        delta = matrix[positions[start:stop]].astype(np.float64) - matrix[start:stop, None].astype(np.float64)
        actual_distance = np.einsum("qkd,qkd->qk", delta, delta)
        max_error = max(max_error, float(np.max(np.abs(actual_distance - distances[start:stop]))))
        sorted_error = max(sorted_error, float(np.max(np.abs(np.sort(actual_distance, axis=1) - expected_sorted[start:stop]))))
        within += int(np.count_nonzero(actual_distance <= expected_sorted[start:stop, -1, None] + tolerance))
        if expected_neighbors is not None:
            overlap += int(np.count_nonzero(np.any(neighbors[start:stop, :, None] == expected_neighbors[start:stop, None, :], axis=2)))
    return {"matches": max_error <= tolerance and sorted_error <= tolerance,
            "reported_distance_matches": max_error <= tolerance,
            "candidate_distance_matches": sorted_error <= tolerance,
            "squared_l2_tolerance": tolerance, "max_reported_distance_error": max_error,
            "max_exact_distance_error": sorted_error, "neighbors_within_exact_kth_distance": within,
            "neighbors": query_count * k,
            "neighbor_id_overlap": overlap if expected_neighbors is not None else None,
            "scope": "Float64 distance equivalence to FAISS returned candidates; no full-dataset Float64 top-k proof",
            "tie_policy": "Any distinct neighbor IDs with equivalent exact squared distances are valid"}


def cpu_knn(args):
    import faiss
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.blas_threshold is not None:
        faiss.cvar.distance_compute_blas_threshold = args.blas_threshold
    faiss.omp_set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for repeat in range(1, args.repeats + 1):
        started = time.perf_counter()
        table = pq.read_table(args.tastes, columns=["id"] + [f"d{i}" for i in range(64)])
        if any(not pa.types.is_float32(table.schema.field(f"d{i}").type) for i in range(64)):
            raise ValueError("CPU and GPU kNN need the same Float32 feature dtype")
        if any(column.null_count for column in table.columns):
            raise ValueError("CPU kNN requires the same non-null frozen features as GPU kNN")
        ids = table.column("id").to_numpy()
        order = np.argsort(ids)
        ids = ids[order]
        if len(ids) < args.k or len(np.unique(ids)) != len(ids):
            raise ValueError("taste input needs unique IDs and at least k rows")
        matrix = np.ascontiguousarray(np.column_stack([
            table.column(f"d{i}").to_numpy()[order] for i in range(64)
        ]), dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("taste features must be finite")
        prepared = time.perf_counter()
        index = faiss.IndexFlatL2(64)
        index.add(matrix)
        indexed = time.perf_counter()
        query_count = min(args.queries, len(ids))
        distances, indices = index.search(matrix[:query_count], args.k)
        searched = time.perf_counter()
        result = pa.table({"query_id": np.repeat(ids[:query_count], args.k),
                           "neighbor_id": ids[indices].reshape(-1), "distance": distances.reshape(-1)})
        assembled = time.perf_counter()
        path = args.output / f"knn_faiss_{args.queries}_r{repeat}.parquet"
        pq.write_table(result, path)
        expected_neighbors = ids[indices]
        expected_distances = recompute_candidate_distances(ids, matrix, query_count, expected_neighbors)
        cpu_accuracy = validate_knn(result, ids, matrix, query_count, expected_distances,
                                    args.k, args.tolerance, expected_neighbors)
        comparison = None
        if args.compare:
            comparison = validate_knn(pq.read_table(args.compare), ids, matrix, query_count,
                                      expected_distances, args.k, args.tolerance, expected_neighbors)
        run = {"repeat": repeat, "warmup": repeat <= args.warmups,
               "dataset_rows": len(ids), "query_rows": query_count,
               "requested_query_rows": args.queries, "dimensions": 64, "k": args.k,
               "load_sort_pack_s": prepared - started, "index_copy_s": indexed - prepared,
               "algorithm_s": searched - indexed, "assemble_s": assembled - searched,
               "prepared_input_total_s": assembled - started,
               "output": str(path.resolve()), "cpu_reported_accuracy": cpu_accuracy, "comparison": comparison}
        runs.append(run)
        print(json.dumps(run), flush=True)
    record = {"baseline": "faiss.IndexFlatL2", "faiss_version": faiss.__version__,
              "faiss_compile_options": faiss.get_compile_options(),
              "blas_threshold": faiss.cvar.distance_compute_blas_threshold,
              "thread_environment": {name: os.environ.get(name) for name in (
                  "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
              "numpy_version": np.__version__, "pyarrow_version": pa.__version__,
              "threads": faiss.omp_get_max_threads(),
              "input": str(args.tastes.resolve()), "input_sha256": sha256(args.tastes),
              "self_matches": "allowed", "metric": "squared_l2", "runs": runs,
              "warmups": args.warmups,
              "median_algorithm_s": statistics.median(run["algorithm_s"] for run in runs[args.warmups:]),
              "median_prepared_input_total_s": statistics.median(run["prepared_input_total_s"] for run in runs[args.warmups:])}
    (args.output / f"knn_faiss_{args.queries}.json").write_text(json.dumps(record, indent=2) + "\n")
    return 1 if any(not run["cpu_reported_accuracy"]["matches"]
                    or (run["comparison"] and not run["comparison"]["matches"]) for run in runs) else 0


def union_ns(intervals):
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def runtime_summary(path):
    started = {}
    stages = {}
    native_intervals = []
    footer = None
    unmatched_terminal = 0
    with gzip.open(path, "rt") as source:
        for line in source:
            record = json.loads(line)
            data = record["data"]
            if record["record_type"] == "footer":
                footer = data
            if record["record_type"] != "timeline_event":
                continue
            payload = data["payload"]
            if payload["kind"] == "operator_stage":
                label = payload["stage_kind"]
                strategy = payload.get("strategy_label") or ""
                if strategy.startswith("forest_import:"):
                    label += ":forest_import"
                elif label == "vector_native" and strategy:
                    label += ":" + strategy
            elif payload["kind"] == "task_lifecycle" and payload["task_kind"] == "graph_algorithm":
                label = "graph_task"
            else:
                continue
            key = (data.get("plan_execution_id"), data["span_id"])
            state = payload["lifecycle"]["state"]
            if state == "started":
                started[key] = (data["monotonic_ns"], label)
            elif state in ("finished", "failed", "cancelled"):
                beginning = started.pop(key, None)
                if beginning is None:
                    unmatched_terminal += 1
                    continue
                start, label = beginning
                stop = data["monotonic_ns"]
                stage = stages.setdefault(label, {"invocations": 0, "host_span_sum_s": 0.0, "failed": 0})
                stage["invocations"] += 1
                stage["host_span_sum_s"] += (stop - start) / 1e9
                stage["failed"] += state != "finished"
                if label.startswith("vector_native") or label in ("ml_native", "graph_task"):
                    native_intervals.append((start, stop))
    complete = (footer is not None and footer.get("complete") is True
                and not footer.get("truncated") and not footer.get("degraded")
                and not started and not unmatched_terminal)
    return {"complete": complete, "stages": stages, "unmatched_starts": len(started),
            "unmatched_terminals": unmatched_terminal,
            "native_host_span_union_s": union_ns(native_intervals) / 1e9 if complete and native_intervals else None,
            "scope": "Host spans; graph task includes input consumption and construction; native calls can enqueue asynchronous work"}


def summarize(args):
    groups = {}
    for path in sorted((args.report / "results").glob("*.json")):
        record = json.loads(path.read_text())
        if "name" not in record or "elapsed_s" not in record:
            continue
        name = record["name"]
        match = re.fullmatch(r"(.+)_r([0-9]+)", name)
        case, repeat = (match.group(1), int(match.group(2))) if match else (name, 1)
        runtime_path = path.with_name(f"{name}.runtime.jsonl.gz")
        runtime = runtime_summary(runtime_path) if runtime_path.is_file() else None
        elapsed = record["elapsed_s"]
        native = runtime["native_host_span_union_s"] if runtime else None
        groups.setdefault(case, []).append({"repeat": repeat, "status": record["status"],
                                           "elapsed_s": elapsed, "rows": record.get("rows"),
                                           "error": record.get("error"), "runtime": runtime,
                                           "native_host_span_fraction": native / elapsed if native is not None and elapsed > 0 else None})
    result = []
    for case, runs in sorted(groups.items()):
        runs.sort(key=lambda run: run["repeat"])
        measured = [run for run in runs if run["repeat"] > args.warmups and run["status"] == "ok"]
        failures = [run for run in runs if run["status"] != "ok"]
        result.append({"case": case, "runs": runs, "failed_runs": len(failures),
                       "median_elapsed_s": statistics.median(run["elapsed_s"] for run in measured) if measured else None,
                       "warmups": args.warmups})
    print(json.dumps({"report": str(args.report.resolve()), "cases": result}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Write SQL for the existing merrec_run.py client")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--tastes", type=Path, required=True)
    gen.add_argument("--results-dir", type=Path, required=True, help="Matches merrec_run.py --results-dir for graph preparation")
    gen.add_argument("--query-batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    gen.add_argument("--graph-window-ends", nargs="+", default=DEFAULT_ENDS)
    cpu = commands.add_parser("cpu-knn", help="Exact CPU search on the same frozen taste Parquet")
    cpu.add_argument("--tastes", type=Path, required=True)
    cpu.add_argument("--output", type=Path, required=True)
    cpu.add_argument("--queries", type=int, required=True)
    cpu.add_argument("--threads", type=int, default=32)
    cpu.add_argument("--blas-threshold", type=int, help="Override FAISS query-count times dimension threshold; 0 selects BLAS")
    cpu.add_argument("--repeats", type=int, default=5)
    cpu.add_argument("--warmups", type=int, default=1, help="Retain these runs but exclude them from medians")
    cpu.add_argument("--k", type=int, default=10)
    cpu.add_argument("--compare", type=Path, help="Complete GPU result Parquet from the same cohort")
    cpu.add_argument("--tolerance", type=float, default=1e-6)
    summary = commands.add_parser("summarize", help="Read existing captures without changing their format")
    summary.add_argument("--report", type=Path, required=True)
    summary.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "generate":
        print(json.dumps(generate(args.output, args.tastes, args.query_batches, args.graph_window_ends, args.results_dir), indent=2))
    elif args.command == "cpu-knn":
        if min(args.queries, args.threads, args.repeats, args.k) < 1 or args.tolerance < 0:
            parser.error("queries, threads, repeats and k must be positive; tolerance must be nonnegative")
        if not 0 <= args.warmups < args.repeats:
            parser.error("warmups must be nonnegative and smaller than repeats")
        if args.blas_threshold is not None and args.blas_threshold < 0:
            parser.error("blas-threshold must be nonnegative")
        return cpu_knn(args)
    else:
        if args.warmups < 0:
            parser.error("warmups must be nonnegative")
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
