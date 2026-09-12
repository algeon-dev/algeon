#!/usr/bin/env python3
"""Measure Sirius on the exact TPC-H SQL and Parquet files used by Algeon.

Run with Sirius's Pixi Python, which supplies its matching DuckDB package.
This is a manual, GPU-locked development entry, not a Cargo test target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import duckdb


TABLES = ("customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier")
QUERIES = ("q01", "q06", "q15", "q17", "q18", "q03")


def sql_by_id(catalog: Path) -> dict[str, str]:
    source = catalog.read_text()
    matches = re.findall(r'id: "(q\d\d)",\s*sql: r#"(.*?)"#', source, re.DOTALL)
    if len(matches) != 22 or len(dict(matches)) != 22:
        raise ValueError(f"expected 22 unique TPC-H queries in {catalog}")
    return dict(matches)


def connect(dataset: Path, extension: Path | None) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    for table in TABLES:
        path = dataset / f"{table}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        connection.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    if extension is not None:
        connection.execute(f"LOAD '{extension}'")
        connection.execute("SET enable_duckdb_fallback=false")
        connection.execute("SET gpu_execution=true")
    return connection


def run(connection: duckdb.DuckDBPyConnection, sql: str) -> tuple[float, list[tuple]]:
    started = time.perf_counter()
    rows = connection.execute(sql).fetchall()
    return (time.perf_counter() - started) * 1000, rows


def fingerprint(rows: list[tuple]) -> str:
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def git_diff_sha256(repo: Path) -> str:
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=repo, check=True, capture_output=True).stdout
    return hashlib.sha256(diff).hexdigest()


def parity(cpu_rows: list[tuple], gpu_rows: list[tuple]) -> str:
    if cpu_rows == gpu_rows:
        return "exact"
    if len(cpu_rows) != len(gpu_rows):
        return "mismatch"
    for cpu_row, gpu_row in zip(cpu_rows, gpu_rows):
        if len(cpu_row) != len(gpu_row):
            return "mismatch"
        for cpu_value, gpu_value in zip(cpu_row, gpu_row):
            if isinstance(cpu_value, float) and isinstance(gpu_value, float):
                if not math.isclose(cpu_value, gpu_value, rel_tol=1e-12, abs_tol=1e-12):
                    return "mismatch"
            elif cpu_value != gpu_value:
                return "mismatch"
    return "float_tolerance"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sirius-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--queries", default=",".join(QUERIES))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        parser.error("iterations must be positive and warmups nonnegative")
    selected_queries = args.queries.split(",")
    if not selected_queries or any(query not in QUERIES for query in selected_queries) or len(set(selected_queries)) != len(selected_queries):
        parser.error("queries must be unique IDs from " + ",".join(QUERIES))
    root = Path(__file__).resolve().parents[2]
    catalog = root / "crates/workloads/src/tpch/catalog.rs"
    extension = args.sirius_root.resolve() / "build/release/extension/sirius/sirius.duckdb_extension"
    os.environ["SIRIUS_CONFIG_FILE"] = str(args.config.resolve())
    sql = sql_by_id(catalog)
    dataset_files = {
        table: {"bytes": (args.dataset / f"{table}.parquet").stat().st_size,
                "sha256": file_sha256(args.dataset / f"{table}.parquet")}
        for table in TABLES
    }
    report: dict[str, object] = {
        "duckdb_version": duckdb.__version__,
        "sirius_sha": git_output(args.sirius_root, "rev-parse", "HEAD"),
        "sirius_dirty": bool(git_output(args.sirius_root, "status", "--porcelain")),
        "sirius_tracked_diff_sha256": git_diff_sha256(args.sirius_root),
        "sirius_untracked_paths": git_output(args.sirius_root, "ls-files", "--others", "--exclude-standard").splitlines(),
        "catalog": str(catalog),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "dataset": str(args.dataset.resolve()),
        "dataset_files": dataset_files,
        "sirius_extension": str(extension),
        "sirius_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "sirius_config": str(args.config.resolve()),
        "sirius_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "disk_spill_configured": "disk:" in args.config.read_text(),
        "duckdb_fallback": False,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "selected_queries": selected_queries,
        "queries": {},
    }
    for query_id in selected_queries:
        record: dict[str, object] = {"status": "unrunnable", "gpu_samples_ms": []}
        report["queries"][query_id] = record
        cpu = connect(args.dataset, None)
        try:
            cpu_ms, cpu_rows = run(cpu, sql[query_id])
            record.update(cpu_ms=cpu_ms, cpu_rows=len(cpu_rows), cpu_fingerprint=fingerprint(cpu_rows))
        except Exception as error:
            record.update(status="cpu_error", reason=f"{type(error).__name__}: {error}")
            print(query_id, record["status"], record["reason"], flush=True)
            continue
        finally:
            cpu.close()
        try:
            gpu = connect(args.dataset, extension)
            try:
                for _ in range(args.warmups):
                    run(gpu, sql[query_id])
                for _ in range(args.iterations):
                    elapsed, gpu_rows = run(gpu, sql[query_id])
                    record["gpu_samples_ms"].append(elapsed)
                    comparison = parity(cpu_rows, gpu_rows)
                    record["parity"] = comparison
                    if comparison == "mismatch":
                        record.update(status="parity_mismatch", gpu_rows=len(gpu_rows), gpu_fingerprint=fingerprint(gpu_rows))
                        break
            finally:
                gpu.close()
            if record["status"] != "parity_mismatch":
                samples = record["gpu_samples_ms"]
                record.update(status="native_matched", gpu_rows=len(cpu_rows), gpu_fingerprint=fingerprint(gpu_rows), median_ms=statistics.median(samples))
        except Exception as error:
            record.update(status="gpu_error", reason=f"{type(error).__name__}: {error}")
        print(query_id, record["status"], record.get("reason", ""), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
