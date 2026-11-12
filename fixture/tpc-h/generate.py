#!/usr/bin/env python3
"""Generate optimized TPC-H Parquet fixtures with tpchgen-rs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tpc"))

from parquet_layout import (
    DEFAULT_ROW_GROUP_BYTES,
    TableLayout,
    check_spark_writer,
    optimize_layout,
    parse_scales,
    print_summary,
    replace_with_empty_dir,
    resolve_output_dir,
    run,
    scale_label,
    verify_layout,
)


TPCH_TABLES = (
    TableLayout("customer", ("c_nationkey", "c_custkey")),
    TableLayout("lineitem", ("l_shipdate", "l_receiptdate", "l_commitdate", "l_orderkey", "l_linenumber")),
    TableLayout("nation", ("n_regionkey", "n_nationkey")),
    TableLayout("orders", ("o_orderdate", "o_orderkey", "o_custkey")),
    TableLayout("part", ("p_brand", "p_type", "p_size", "p_partkey")),
    TableLayout("partsupp", ("ps_partkey", "ps_suppkey")),
    TableLayout("region", ("r_regionkey",)),
    TableLayout("supplier", ("s_nationkey", "s_suppkey")),
)


TPCH_EXPECTED_ROWS = {
    "sf1": {
        "customer": 150_000,
        "lineitem": 6_001_215,
        "nation": 25,
        "orders": 1_500_000,
        "part": 200_000,
        "partsupp": 800_000,
        "region": 5,
        "supplier": 10_000,
    },
    "sf10": {
        "customer": 1_500_000,
        "lineitem": 59_986_052,
        "nation": 25,
        "orders": 15_000_000,
        "part": 2_000_000,
        "partsupp": 8_000_000,
        "region": 5,
        "supplier": 100_000,
    },
    "sf100": {
        "customer": 15_000_000,
        "lineitem": 600_037_902,
        "nation": 25,
        "orders": 150_000_000,
        "part": 20_000_000,
        "partsupp": 80_000_000,
        "region": 5,
        "supplier": 1_000_000,
    },
    "sf200": {
        "customer": 30_000_000,
        "lineitem": 1_200_018_434,
        "nation": 25,
        "orders": 300_000_000,
        "part": 40_000_000,
        "partsupp": 160_000_000,
        "region": 5,
        "supplier": 2_000_000,
    },
    "sf300": {
        "customer": 45_000_000,
        "lineitem": 1_799_989_091,
        "nation": 25,
        "orders": 450_000_000,
        "part": 60_000_000,
        "partsupp": 240_000_000,
        "region": 5,
        "supplier": 3_000_000,
    },
    # Lineitem rows are the sum of tpchgen's 1-to-7 lines-per-order draws over the
    # 1,500,000 * SF orders, so they are not a multiple of the scale factor.
    "sf500": {
        "customer": 75_000_000,
        "lineitem": 3_000_028_242,
        "nation": 25,
        "orders": 750_000_000,
        "part": 100_000_000,
        "partsupp": 400_000_000,
        "region": 5,
        "supplier": 5_000_000,
    },
    "sf1000": {
        "customer": 150_000_000,
        "lineitem": 5_999_989_709,
        "nation": 25,
        "orders": 1_500_000_000,
        "part": 200_000_000,
        "partsupp": 800_000_000,
        "region": 5,
        "supplier": 10_000_000,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", action="append", help="Scale factor to generate; repeat for multiple scales")
    parser.add_argument("--out", type=Path, help="Output parquet directory for a single scale")
    parser.add_argument("--out-root", type=Path, help="Root containing sfN/parquet output directories")
    parser.add_argument("--variant", choices=("raw", "optimized", "both"), default="optimized")
    parser.add_argument("--force", action="store_true", help="Replace existing output directories")
    parser.add_argument("--no-verify", action="store_true", help="Skip Parquet metadata verification")
    parser.add_argument("--row-group-bytes", type=int, default=DEFAULT_ROW_GROUP_BYTES)
    parser.add_argument("--compression", default="SNAPPY")
    parser.add_argument("--tpchgen-root", type=Path, default=Path.home() / "code" / "tpchgen-rs")
    parser.add_argument("--profile", choices=("debug", "release"), default="release")
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--parts", type=int, help="Generate partitioned raw TPC-H tables before optimization")
    parser.add_argument(
        "--optimize-engine",
        choices=("spark", "duckdb", "pyarrow"),
        default="spark",
        help="spark: DuckDB sorts and Spark writes each file with parquet-java (default)",
    )
    parser.add_argument(
        "--unsorted",
        action="store_true",
        help="Keep tpchgen's row order instead of sorting by the pruning keys; needs the spark engine",
    )
    parser.add_argument(
        "--spark-python",
        default=os.environ.get("ALGEON_SPARK_PYTHON", sys.executable),
        help="Python interpreter with PySpark for the spark engine (default: $ALGEON_SPARK_PYTHON or this interpreter)",
    )
    parser.add_argument("--staging-root", type=Path, help="Root for temporary raw parquet staging")
    args = parser.parse_args()

    scales = parse_scales(args.scale)
    if args.out is not None and len(scales) != 1:
        parser.error("--out can only be used with exactly one --scale")
    if args.unsorted and args.optimize_engine != "spark":
        parser.error("--unsorted needs --optimize-engine spark")
    if args.optimize_engine == "spark" and args.variant != "raw":
        try:
            check_spark_writer(args.spark_python)
        except RuntimeError as error:
            parser.exit(2, f"{parser.prog}: error: {error}\n")

    cli = args.tpchgen_root / "target" / args.profile / "tpcgen-cli"
    if not cli.exists():
        build_cmd = ["cargo", "build", "-p", "tpcgen-cli"]
        if args.profile == "release":
            build_cmd.append("--release")
        run(build_cmd, cwd=args.tpchgen_root)
    if not cli.exists():
        raise FileNotFoundError(f"missing tpcgen-cli at {cli}")

    generated_outputs: list[Path] = []
    for scale in scales:
        output_dir = resolve_output_dir(workload="tpch", scale=scale, out=args.out, out_root=args.out_root)
        staging_root = (
            args.staging_root / "tpch" / scale_label(scale)
            if args.staging_root is not None
            else output_dir.parent / ".staging"
        )
        raw_dir = output_dir if args.variant == "raw" else staging_root / "raw"
        replace_with_empty_dir(raw_dir, force=args.force if args.variant == "raw" else True)

        cmd = [
            str(cli),
            "tpch",
            "parquet",
            "--scale-factor",
            str(scale),
            "--output-dir",
            str(raw_dir),
            "--compression",
            args.compression.upper(),
            "--row-group-bytes",
            str(args.row_group_bytes),
            "--num-threads",
            str(args.num_threads),
            "--no-progress",
        ]
        if args.parts is not None:
            cmd.extend(["--parts", str(args.parts)])
        run(cmd, cwd=args.tpchgen_root)

        if args.variant == "raw":
            if not args.no_verify:
                verify_layout(
                    output_dir=output_dir,
                    layouts=TPCH_TABLES,
                    compression=args.compression,
                    require_sorted=False,
                    expected_rows=TPCH_EXPECTED_ROWS.get(scale_label(scale)),
                )
            generated_outputs.append(output_dir)
            continue

        if args.variant == "both":
            raw_output = output_dir.parent / "raw_parquet"
            replace_with_empty_dir(raw_output, force=args.force)
            shutil.rmtree(raw_output)
            shutil.copytree(raw_dir, raw_output)

        optimize_layout(
            raw_dir=raw_dir,
            output_dir=output_dir,
            layouts=TPCH_TABLES,
            row_group_bytes=args.row_group_bytes,
            compression=args.compression,
            force=args.force,
            scale=scale,
            engine=args.optimize_engine,
            sort_rows=not args.unsorted,
            spark_python=args.spark_python,
        )
        if not args.no_verify:
            verify_layout(
                output_dir=output_dir,
                layouts=TPCH_TABLES,
                compression=args.compression,
                require_sorted=not args.unsorted,
                expected_rows=TPCH_EXPECTED_ROWS.get(scale_label(scale)),
                writer=args.optimize_engine,
            )
        if staging_root.exists():
            shutil.rmtree(staging_root)
        generated_outputs.append(output_dir)

    print_summary(generated_outputs)


if __name__ == "__main__":
    main()
