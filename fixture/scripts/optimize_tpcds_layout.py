#!/usr/bin/env python3
"""Produce an "optimized" (date-clustered) TPC-DS parquet dataset.

The stock DuckDB ``dsdgen`` + ``EXPORT DATABASE (FORMAT PARQUET)`` layout writes
fact tables in generation order, which is effectively random with respect to the
``*_date_sk`` join keys. Every row group then carries the table's full date
range as its min/max, so join-derived dynamic-filter row-group pruning in the
native reader can never skip a row group (see
``crates/nexus-query-engine/src/dynamic_filter.rs``).

This script rewrites each fact table sorted by its date surrogate key so that
each row group covers a narrow contiguous date range. Dimension tables are
hardlinked unchanged. The rewrite changes only *row ordering*, not row content:
the same SNAPPY codec and row-group size (122880) as the DuckDB-generated source
are used. (It is a genuine Parquet re-encode, so page boundaries, per-row-group
statistics, and the exact compressed size differ from the source — but the
logical rows, schema, and codec are identical, so CPU-vs-GPU and
local-vs-Iceberg comparisons stay fair.) Row counts are verified after each
rewrite.

This is the TPC-DS analog of ``tpch/sf100_optimized`` (lineitem sorted by
``l_orderkey``), which is why TPC-H is healthy on the GPU while unsorted TPC-DS
is not.

Usage::

    optimize_tpcds_layout.py <src_parquet_dir> <dst_parquet_dir> \
        [--row-group-size 122880] [--threads N] [--memory-limit 100GB] \
        [--duckdb /path/to/duckdb]

Example::

    optimize_tpcds_layout.py ./fixture/data/tpcds/sf10/parquet \
                             ./fixture/data/tpcds/sf10_optimized/parquet
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Fact tables and the date surrogate key they are clustered on. "Sort by
# sold_date_sk only" (no composite key) — the dominant TPC-DS pattern is a
# selective date_dim filter against a huge fact table, realized as a
# join-derived dynamic filter on the fact's date_sk.
SORT_KEYS: dict[str, str] = {
    "store_sales": "ss_sold_date_sk",
    "catalog_sales": "cs_sold_date_sk",
    "web_sales": "ws_sold_date_sk",
    # Returns and inventory follow the same date-clustering pattern on their own
    # date join key; clustering them is free and helps the returns/inventory
    # queries (e.g. q81 catalog_returns, q21/q22 inventory).
    "store_returns": "sr_returned_date_sk",
    "catalog_returns": "cr_returned_date_sk",
    "web_returns": "wr_returned_date_sk",
    "inventory": "inv_date_sk",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="source TPC-DS parquet directory")
    parser.add_argument("dst", type=Path, help="destination directory for the optimized dataset")
    parser.add_argument("--row-group-size", type=int, default=122880, help="parquet row group size (default: DuckDB default)")
    parser.add_argument("--compression", default="snappy", help="parquet compression codec (default: snappy)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8, help="DuckDB threads")
    parser.add_argument("--memory-limit", default="100GB", help="DuckDB memory limit (sort spills to disk above this)")
    parser.add_argument("--duckdb", default=os.environ.get("DUCKDB", "/home/ubuntu/bin/duckdb"), help="path to the duckdb CLI")
    parser.add_argument("--temp-dir", default=None, help="DuckDB temp dir for sort spill (defaults under dst)")
    return parser.parse_args(argv)


def table_name(path: Path) -> str:
    return path.stem


def sql_str(value: str) -> str:
    """Escape a string for safe single-quoted SQL interpolation."""
    return value.replace("'", "''")


def parquet_row_count(duckdb: str, path: Path) -> int:
    proc = subprocess.run(
        [duckdb, "-noheader", "-list", "-c", f"SELECT count(*) FROM read_parquet('{sql_str(str(path))}');"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"duckdb row-count failed for {path}:\n{proc.stdout}{proc.stderr}")
    return int(proc.stdout.strip())


def run_duckdb(duckdb: str, sql: str) -> None:
    proc = subprocess.run([duckdb, "-c", sql], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"duckdb failed (exit {proc.returncode}) for SQL:\n{sql}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")
    if src.resolve() == dst.resolve():
        raise SystemExit(f"refusing to optimize in place: src and dst resolve to the same directory ({src.resolve()})")
    dst.mkdir(parents=True, exist_ok=True)
    temp_dir = args.temp_dir or str(dst / ".duckdb_tmp")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    pragmas = (
        f"SET threads={args.threads}; "
        f"SET memory_limit='{args.memory_limit}'; "
        f"SET temp_directory='{temp_dir}'; "
        "SET preserve_insertion_order=false; "
    )

    parquet_files = sorted(src.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files found under {src}")

    for f in parquet_files:
        name = table_name(f)
        out = dst / f.name
        if name in SORT_KEYS:
            key = SORT_KEYS[name]
            t0 = time.time()
            # ORDER BY key NULLS LAST clusters each row group on a contiguous
            # date range; nulls (a small fraction of fact rows) collect at the
            # tail. Single output file to mirror the source one-file-per-table
            # layout. Write to a temp file and rename so a failed COPY never
            # leaves a partial/stale output in place.
            tmp_out = out.with_suffix(".parquet.tmp")
            sql = (
                pragmas
                + f"COPY (SELECT * FROM read_parquet('{sql_str(str(f))}') ORDER BY {key} NULLS LAST) "
                + f"TO '{sql_str(str(tmp_out))}' "
                + f"(FORMAT PARQUET, COMPRESSION '{sql_str(args.compression)}', ROW_GROUP_SIZE {args.row_group_size});"
            )
            print(f"[optimize] sorting {name} by {key} ...", flush=True)
            run_duckdb(args.duckdb, sql)
            src_rows = parquet_row_count(args.duckdb, f)
            dst_rows = parquet_row_count(args.duckdb, tmp_out)
            if src_rows != dst_rows:
                tmp_out.unlink(missing_ok=True)
                raise SystemExit(f"row-count mismatch for {name}: src={src_rows} dst={dst_rows}")
            tmp_out.replace(out)
            print(
                f"[optimize] wrote {out} ({out.stat().st_size/1e9:.2f} GB, {dst_rows} rows) in {time.time()-t0:.1f}s",
                flush=True,
            )
        else:
            if out.exists():
                out.unlink()
            try:
                os.link(f, out)
            except OSError:
                shutil.copy2(f, out)
            print(f"[optimize] linked {name}", flush=True)

    # Copy any non-parquet sidecar files (schema.sql, load.sql, ...).
    for f in src.iterdir():
        if f.suffix == ".parquet" or f.is_dir():
            continue
        shutil.copy2(f, dst / f.name)

    # Best-effort cleanup of the spill dir.
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[optimize] done -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
