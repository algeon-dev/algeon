#!/usr/bin/env python3
"""Generate optimized TPC-DS Parquet fixtures with tpchgen-rs."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from tpc_parquet_layout import (
    DEFAULT_ROW_GROUP_BYTES,
    TableLayout,
    optimize_layout,
    parse_scales,
    print_summary,
    repo_root,
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
}


CARGO_TOML = r'''
[workspace]

[package]
name = "tpcds-arrow-parquet"
version = "0.1.0"
edition = "2021"

[dependencies]
arrow = "59"
parquet = "59"
tpcdsgen = { path = "__TPCGEN_ROOT__/tpcdsgen" }
tpcdsgen-arrow = { path = "__TPCGEN_ROOT__/tpcdsgen-arrow" }
'''


RUST_MAIN = r'''
use std::env;
use std::fs::{self, File};
use std::path::PathBuf;

use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tpcdsgen::config::{CompatMode, Session, SessionBuilder, Table};
use tpcdsgen_arrow::{
    CallCenterArrow, CatalogPageArrow, CatalogReturnsArrow, CatalogSalesArrow,
    CustomerAddressArrow, CustomerArrow, CustomerDemographicsArrow, DateDimArrow,
    HouseholdDemographicsArrow, IncomeBandArrow, InventoryArrow, ItemArrow, PromotionArrow,
    ReasonArrow, RecordBatchIterator, ShipModeArrow, StoreArrow, StoreReturnsArrow,
    StoreSalesArrow, TimeDimArrow, WarehouseArrow, WebPageArrow, WebReturnsArrow,
    WebSalesArrow, WebSiteArrow,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut scale = 1.0_f64;
    let mut out_dir = PathBuf::from(".");
    let mut batch_size = 65_536_usize;
    let mut compression = Compression::SNAPPY;
    let mut compat = CompatMode::Trino;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scale-factor" | "-s" => scale = args.next().ok_or("missing value for --scale-factor")?.parse()?,
            "--output-dir" | "-o" => out_dir = PathBuf::from(args.next().ok_or("missing value for --output-dir")?),
            "--batch-size" => batch_size = args.next().ok_or("missing value for --batch-size")?.parse()?,
            "--compression" => compression = parse_compression(&args.next().ok_or("missing value for --compression")?)?,
            "--compat" => compat = args.next().ok_or("missing value for --compat")?.parse()?,
            "--help" | "-h" => {
                eprintln!("usage: tpcds-arrow-parquet --scale-factor N --output-dir DIR [--batch-size ROWS] [--compression SNAPPY] [--compat trino|c]");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    fs::create_dir_all(&out_dir)?;
    let session = SessionBuilder::new()
        .with_scale_factor(scale)
        .with_compat_mode(compat)
        .with_command_line_arguments(env::args().collect::<Vec<_>>().join(" "))
        .build()?;

    for table in tables() {
        eprintln!("generating {}", table.get_name());
        write_table(*table, session.clone(), batch_size, compression, out_dir.join(format!("{}.parquet", table.get_name())))?;
    }
    Ok(())
}

fn parse_compression(value: &str) -> Result<Compression> {
    match value.to_ascii_uppercase().as_str() {
        "SNAPPY" => Ok(Compression::SNAPPY),
        "UNCOMPRESSED" => Ok(Compression::UNCOMPRESSED),
        other => Err(format!("unsupported compression for TPC-DS generator: {other}").into()),
    }
}

fn tables() -> &'static [Table] {
    &[
        Table::CallCenter,
        Table::CatalogPage,
        Table::CatalogReturns,
        Table::CatalogSales,
        Table::Customer,
        Table::CustomerAddress,
        Table::CustomerDemographics,
        Table::DateDim,
        Table::HouseholdDemographics,
        Table::IncomeBand,
        Table::Inventory,
        Table::Item,
        Table::Promotion,
        Table::Reason,
        Table::ShipMode,
        Table::Store,
        Table::StoreReturns,
        Table::StoreSales,
        Table::TimeDim,
        Table::Warehouse,
        Table::WebPage,
        Table::WebReturns,
        Table::WebSales,
        Table::WebSite,
    ]
}

fn make_iter(table: Table, session: Session, batch_size: usize) -> Box<dyn RecordBatchIterator<Item = RecordBatch>> {
    match table {
        Table::CallCenter => Box::new(CallCenterArrow::new(session).with_batch_size(batch_size)),
        Table::CatalogPage => Box::new(CatalogPageArrow::new(session).with_batch_size(batch_size)),
        Table::CatalogReturns => Box::new(CatalogReturnsArrow::new(session).with_batch_size(batch_size)),
        Table::CatalogSales => Box::new(CatalogSalesArrow::new(session).with_batch_size(batch_size)),
        Table::Customer => Box::new(CustomerArrow::new(session).with_batch_size(batch_size)),
        Table::CustomerAddress => Box::new(CustomerAddressArrow::new(session).with_batch_size(batch_size)),
        Table::CustomerDemographics => Box::new(CustomerDemographicsArrow::new(session).with_batch_size(batch_size)),
        Table::DateDim => Box::new(DateDimArrow::new(session).with_batch_size(batch_size)),
        Table::HouseholdDemographics => Box::new(HouseholdDemographicsArrow::new(session).with_batch_size(batch_size)),
        Table::IncomeBand => Box::new(IncomeBandArrow::new(session).with_batch_size(batch_size)),
        Table::Inventory => Box::new(InventoryArrow::new(session).with_batch_size(batch_size)),
        Table::Item => Box::new(ItemArrow::new(session).with_batch_size(batch_size)),
        Table::Promotion => Box::new(PromotionArrow::new(session).with_batch_size(batch_size)),
        Table::Reason => Box::new(ReasonArrow::new(session).with_batch_size(batch_size)),
        Table::ShipMode => Box::new(ShipModeArrow::new(session).with_batch_size(batch_size)),
        Table::Store => Box::new(StoreArrow::new(session).with_batch_size(batch_size)),
        Table::StoreReturns => Box::new(StoreReturnsArrow::new(session).with_batch_size(batch_size)),
        Table::StoreSales => Box::new(StoreSalesArrow::new(session).with_batch_size(batch_size)),
        Table::TimeDim => Box::new(TimeDimArrow::new(session).with_batch_size(batch_size)),
        Table::Warehouse => Box::new(WarehouseArrow::new(session).with_batch_size(batch_size)),
        Table::WebPage => Box::new(WebPageArrow::new(session).with_batch_size(batch_size)),
        Table::WebReturns => Box::new(WebReturnsArrow::new(session).with_batch_size(batch_size)),
        Table::WebSales => Box::new(WebSalesArrow::new(session).with_batch_size(batch_size)),
        Table::WebSite => Box::new(WebSiteArrow::new(session).with_batch_size(batch_size)),
        _ => unreachable!("source/update tables are not generated"),
    }
}

fn write_table(table: Table, session: Session, batch_size: usize, compression: Compression, file: PathBuf) -> Result<()> {
    let mut iter = make_iter(table, session, batch_size);
    let props = WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(batch_size))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(file)?, iter.schema().clone(), Some(props))?;
    for batch in iter.by_ref() {
        writer.write(&batch)?;
    }
    writer.close()?;
    Ok(())
}
'''


def ensure_rust_writer(tpchgen_root: Path) -> Path:
    project = repo_root() / "target" / "tpcds-arrow-parquet"
    (project / "src").mkdir(parents=True, exist_ok=True)
    (project / "Cargo.toml").write_text(CARGO_TOML.replace("__TPCGEN_ROOT__", str(tpchgen_root)))
    (project / "src" / "main.rs").write_text(RUST_MAIN)
    return project


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
    parser.add_argument("--tpchgen-root", type=Path, default=Path("~/code/tpchgen-rs"))
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--compat", default="trino")
    args = parser.parse_args()

    scales = parse_scales(args.scale)
    if args.out is not None and len(scales) != 1:
        parser.error("--out can only be used with exactly one --scale")
    if args.compression.upper() != "SNAPPY":
        parser.error("TPC-DS generation currently supports --compression SNAPPY only")

    rust_project = ensure_rust_writer(args.tpchgen_root)
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", str(args.tpchgen_root / "target"))

    generated_outputs: list[Path] = []
    for scale in scales:
        output_dir = resolve_output_dir(workload="tpcds", scale=scale, out=args.out, out_root=args.out_root)
        staging_root = repo_root() / "target" / "fixture-staging" / "tpcds" / scale_label(scale)
        raw_dir = output_dir if args.variant == "raw" else staging_root / "raw"
        replace_with_empty_dir(raw_dir, force=args.force if args.variant == "raw" else True)

        run(
            [
                "cargo",
                "run",
                "--release",
                "--manifest-path",
                str(rust_project / "Cargo.toml"),
                "--",
                "--scale-factor",
                str(scale),
                "--output-dir",
                str(raw_dir),
                "--compression",
                args.compression.upper(),
                "--batch-size",
                str(args.batch_size),
                "--compat",
                args.compat,
            ],
            env=env,
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

        optimize_layout(raw_dir=raw_dir, output_dir=output_dir, layouts=TPCDS_TABLES, row_group_bytes=args.row_group_bytes, compression=args.compression, force=args.force)
        if not args.no_verify:
            verify_layout(
                output_dir=output_dir,
                layouts=TPCDS_TABLES,
                compression=args.compression,
                require_sorted=True,
                expected_rows=TPCDS_EXPECTED_ROWS.get(scale_label(scale)),
            )
        generated_outputs.append(output_dir)

    print_summary(generated_outputs)


if __name__ == "__main__":
    main()
