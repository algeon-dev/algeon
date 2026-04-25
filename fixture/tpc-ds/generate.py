#!/usr/bin/env python3
"""Generate optimized TPC-DS Parquet fixtures with tpchgen-rs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tpc"))

from parquet_layout import (
    DEFAULT_ROW_GROUP_BYTES,
    TableLayout,
    optimize_layout,
    parse_scales,
    print_summary,
    replace_with_empty_dir,
    resolve_output_dir,
    run,
    scale_label,
    verify_layout,
)


TPCDS_TABLES = (
    TableLayout("call_center", ("cc_call_center_sk",)),
    TableLayout("catalog_page", ("cp_catalog_page_sk",)),
    TableLayout(
        "catalog_returns",
        ("cr_returned_date_sk", "cr_item_sk", "cr_order_number"),
        (("cr_return_amount_inc_tax", "cr_return_amt_inc_tax"),),
        (("cr_return_quantity", "int64"),),
    ),
    TableLayout("catalog_sales", ("cs_sold_date_sk", "cs_item_sk", "cs_order_number"), (), (("cs_quantity", "int64"),)),
    TableLayout("customer", ("c_current_cdemo_sk", "c_customer_sk")),
    TableLayout("customer_address", ("ca_state", "ca_address_sk")),
    TableLayout("customer_demographics", ("cd_marital_status", "cd_education_status", "cd_demo_sk")),
    TableLayout("date_dim", ("d_year", "d_date_sk")),
    TableLayout("household_demographics", ("hd_dep_count", "hd_demo_sk")),
    TableLayout("income_band", ("ib_lower_bound", "ib_income_band_sk"), (("ib_income_band_id", "ib_income_band_sk"),)),
    TableLayout("inventory", ("inv_date_sk", "inv_item_sk", "inv_warehouse_sk"), (), (("inv_quantity_on_hand", "int64"),)),
    TableLayout("item", ("i_category", "i_class", "i_item_sk")),
    TableLayout("promotion", ("p_start_date_sk", "p_promo_sk")),
    TableLayout("reason", ("r_reason_sk",), (("r_reason_description", "r_reason_desc"),)),
    TableLayout("ship_mode", ("sm_ship_mode_sk",)),
    TableLayout("store", ("s_store_sk",)),
    TableLayout("store_returns", ("sr_returned_date_sk", "sr_item_sk", "sr_ticket_number"), (), (("sr_return_quantity", "int64"),)),
    TableLayout("store_sales", ("ss_sold_date_sk", "ss_item_sk", "ss_ticket_number"), (), (("ss_quantity", "int64"),)),
    TableLayout("time_dim", ("t_hour", "t_time_sk")),
    TableLayout("warehouse", ("w_warehouse_sk",)),
    TableLayout("web_page", ("wp_web_page_sk",)),
    TableLayout("web_returns", ("wr_returned_date_sk", "wr_item_sk", "wr_order_number"), (), (("wr_return_quantity", "int64"),)),
    TableLayout("web_sales", ("ws_sold_date_sk", "ws_item_sk", "ws_order_number"), (), (("ws_quantity", "int64"),)),
    TableLayout("web_site", ("web_site_sk",)),
)


TPCDS_EXPECTED_ROWS = {
    "sf1": {
        "call_center": 6,
        "catalog_page": 11_718,
        "catalog_returns": 144_067,
        "catalog_sales": 1_441_548,
        "customer": 100_000,
        "customer_address": 50_000,
        "customer_demographics": 1_920_800,
        "date_dim": 73_049,
        "household_demographics": 7_200,
        "income_band": 20,
        "inventory": 11_745_000,
        "item": 18_000,
        "promotion": 300,
        "reason": 35,
        "ship_mode": 20,
        "store": 12,
        "store_returns": 287_514,
        "store_sales": 2_880_404,
        "time_dim": 86_400,
        "warehouse": 5,
        "web_page": 60,
        "web_returns": 71_763,
        "web_sales": 719_384,
        "web_site": 30,
    },
    "sf10": {
        "call_center": 24,
        "catalog_page": 12_000,
        "catalog_returns": 1_439_749,
        "catalog_sales": 14_401_261,
        "customer": 500_000,
        "customer_address": 250_000,
        "customer_demographics": 1_920_800,
        "date_dim": 73_049,
        "household_demographics": 7_200,
        "income_band": 20,
        "inventory": 133_110_000,
        "item": 102_000,
        "promotion": 500,
        "reason": 45,
        "ship_mode": 20,
        "store": 102,
        "store_returns": 2_875_432,
        "store_sales": 28_800_991,
        "time_dim": 86_400,
        "warehouse": 10,
        "web_page": 200,
        "web_returns": 719_217,
        "web_sales": 7_197_566,
        "web_site": 42,
    },
    "sf100": {
        "call_center": 30,
        "catalog_page": 20_400,
        "catalog_returns": 14_404_374,
        "catalog_sales": 143_997_065,
        "customer": 2_000_000,
        "customer_address": 1_000_000,
        "customer_demographics": 1_920_800,
        "date_dim": 73_049,
        "household_demographics": 7_200,
        "income_band": 20,
        "inventory": 399_330_000,
        "item": 204_000,
        "promotion": 1_000,
        "reason": 55,
        "ship_mode": 20,
        "store": 402,
        "store_returns": 28_795_080,
        "store_sales": 287_997_024,
        "time_dim": 86_400,
        "warehouse": 15,
        "web_page": 2_040,
        "web_returns": 7_197_670,
        "web_sales": 72_001_237,
        "web_site": 24,
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
    parser.add_argument("--compat", default="trino")
    parser.add_argument("--optimize-engine", choices=("duckdb", "pyarrow"), default="duckdb")
    parser.add_argument("--staging-root", type=Path, help="Root for temporary raw parquet staging")
    args = parser.parse_args()

    scales = parse_scales(args.scale)
    if args.out is not None and len(scales) != 1:
        parser.error("--out can only be used with exactly one --scale")
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
        output_dir = resolve_output_dir(workload="tpcds", scale=scale, out=args.out, out_root=args.out_root)
        staging_root = (
            args.staging_root / "tpcds" / scale_label(scale)
            if args.staging_root is not None
            else output_dir.parent / ".staging"
        )
        raw_dir = output_dir if args.variant == "raw" else staging_root / "raw"
        replace_with_empty_dir(raw_dir, force=args.force if args.variant == "raw" else True)

        run(
            [
                str(cli),
                "tpcds",
                "parquet",
                "--scale-factor",
                str(scale),
                "--output-dir",
                str(raw_dir),
                "--compression",
                args.compression.upper(),
                "--row-group-bytes",
                str(args.row_group_bytes),
                "--compat",
                args.compat,
                "--no-progress",
            ],
            cwd=args.tpchgen_root,
        )

        if args.variant == "raw":
            if not args.no_verify:
                verify_layout(
                    output_dir=output_dir,
                    layouts=TPCDS_TABLES,
                    compression=args.compression,
                    require_sorted=False,
                    expected_rows=TPCDS_EXPECTED_ROWS.get(scale_label(scale)),
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
            layouts=TPCDS_TABLES,
            row_group_bytes=args.row_group_bytes,
            compression=args.compression,
            force=args.force,
            engine=args.optimize_engine,
        )
        if not args.no_verify:
            verify_layout(
                output_dir=output_dir,
                layouts=TPCDS_TABLES,
                compression=args.compression,
                require_sorted=True,
                expected_rows=TPCDS_EXPECTED_ROWS.get(scale_label(scale)),
            )
        if staging_root.exists():
            shutil.rmtree(staging_root)
        generated_outputs.append(output_dir)

    print_summary(generated_outputs)


if __name__ == "__main__":
    main()
