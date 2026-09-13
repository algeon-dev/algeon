//! Mini TPC-H-like backend pipeline contract.
//!
//! This is intentionally not a DataFusion test and does not use the TPC-H
//! corpus. It proves that the libcudf-backed primitives needed by a query
//! backend compose across a small SF1-shaped flow.

use std::{path::Path, sync::Arc};

use arrow::{
    array::{Array, Date32Array, Decimal128Array, Int32Array, StringArray},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    AggSpec, DataFrame, DataType, GroupByAgg, SortOptions, TimestampResolution, TypeId,
    core::Table,
    io::source::{
        LocalParquetFile, LocalParquetSource, SourceColumn, SourceColumnRef, SourceOutputColumn, SourceSchema,
    },
    ops::{
        expr::{col, lit_decimal128_raw, lit_timestamp_raw},
        join::{NullEquality, inner_join_indices},
    },
    testing::prelude::*,
};
use tempfile::TempDir;

use crate::common::{run_named_cases, sample_runtime};

fn source_column(name: &str, dtype: DataType) -> SourceColumn {
    SourceColumn::new(name, dtype, false).unwrap()
}

fn output_column(name: &str, read_index: usize) -> SourceOutputColumn {
    SourceOutputColumn::new(name, SourceColumnRef::ReadColumn(read_index)).unwrap()
}

fn read_source(runtime: &cudf::Runtime, path: &Path, stable_id: &str, columns: &[(&str, DataType)]) -> DataFrame {
    let schema = SourceSchema::new(columns.iter().map(|(name, dtype)| source_column(name, *dtype))).unwrap();
    let source = LocalParquetSource::data(
        [LocalParquetFile::new(path, stable_id, []).unwrap()],
        schema,
        columns.iter().map(|(name, _)| *name),
        columns.iter().enumerate().map(|(index, (name, _))| output_column(name, index)),
        [],
        [],
    )
    .unwrap();

    DataFrame::read_parquet_source(runtime, &source).unwrap()
}

fn decimal128_array(raw_values: Vec<i128>) -> Arc<dyn Array> {
    Arc::new(Decimal128Array::from(raw_values).with_precision_and_scale(18, 2).unwrap())
}

fn write_lineitem_fixture(runtime: &cudf::Runtime, path: &Path) {
    let schema = Schema::new(vec![
        Field::new("orderkey", ArrowDataType::Int32, false),
        Field::new("shipdate", ArrowDataType::Date32, false),
        Field::new("extendedprice", ArrowDataType::Decimal128(18, 2), false),
        Field::new("discount", ArrowDataType::Decimal128(18, 2), false),
        Field::new("returnflag", ArrowDataType::Utf8, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 3, 4, 5])),
            Arc::new(Date32Array::from(vec![19_350, 19_370, 19_340, 19_320, 19_355, 19_330])),
            decimal128_array(vec![10_000, 20_000, 30_000, 40_000, 50_000, 60_000]),
            decimal128_array(vec![5, 10, 0, 25, 5, 50]),
            Arc::new(StringArray::from(vec!["R", "N", "R", "R-VOID", "A", "R"])),
        ],
    )
    .unwrap();

    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
}

fn write_orders_fixture(runtime: &cudf::Runtime, path: &Path) {
    let schema = Schema::new(vec![
        Field::new("orderkey", ArrowDataType::Int32, false),
        Field::new("custkey", ArrowDataType::Int32, false),
        Field::new("orderstatus", ArrowDataType::Utf8, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 5])),
            Arc::new(Int32Array::from(vec![10, 20, 10, 30])),
            Arc::new(StringArray::from(vec!["O", "F", "O", "O"])),
        ],
    )
    .unwrap();

    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
}

fn key_table(frame: &DataFrame, column: &str) -> Table {
    Table::from_columns(vec![frame.series(column).unwrap().as_column().deep_copy().unwrap()]).unwrap()
}

fn mini_tpch_pipeline_composes_parquet_dates_decimals_like_join_groupby_sort_and_arrow_export() {
    let dir = TempDir::new().unwrap();
    let lineitem_path = dir.path().join("lineitem.parquet");
    let orders_path = dir.path().join("orders.parquet");
    let runtime = sample_runtime();

    write_lineitem_fixture(&runtime, &lineitem_path);
    write_orders_fixture(&runtime, &orders_path);

    let lineitem = read_source(
        &runtime,
        &lineitem_path,
        "lineitem-mini",
        &[
            ("orderkey", DataType::new(TypeId::Int32)),
            ("shipdate", DataType::new(TypeId::TimestampDays)),
            ("extendedprice", DataType::fixed_point(TypeId::Decimal128, -2).unwrap()),
            ("discount", DataType::fixed_point(TypeId::Decimal128, -2).unwrap()),
            ("returnflag", DataType::new(TypeId::String)),
        ],
    );
    let orders = read_source(
        &runtime,
        &orders_path,
        "orders-mini",
        &[
            ("orderkey", DataType::new(TypeId::Int32)),
            ("custkey", DataType::new(TypeId::Int32)),
            ("orderstatus", DataType::new(TypeId::String)),
        ],
    );

    let filtered = lineitem
        .filter_expr(
            col("shipdate").lt(lit_timestamp_raw(19_360, TimestampResolution::Days))
                & col("returnflag").str().like("R%"),
        )
        .unwrap()
        .with_columns([(col("extendedprice") * (lit_decimal128_raw(100, -2) - col("discount"))).alias("revenue")])
        .unwrap();

    let join_keys = key_table(&filtered, "orderkey");
    let order_keys = key_table(&orders, "orderkey");
    let indices = inner_join_indices(
        &join_keys.testing_view().unwrap(),
        &order_keys.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    let left_indices = cudf::Series::from_column(indices.left).unwrap();
    let right_indices = cudf::Series::from_column(indices.right).unwrap();
    let lineitem_payload =
        filtered.select(["orderkey", "returnflag", "revenue"]).unwrap().gather(&left_indices).unwrap();
    let order_payload = orders.select(["custkey"]).unwrap().gather(&right_indices).unwrap();

    let joined = DataFrame::from_series([
        ("custkey", order_payload.series("custkey").unwrap()),
        ("returnflag", lineitem_payload.series("returnflag").unwrap()),
        ("revenue", lineitem_payload.series("revenue").unwrap()),
    ])
    .unwrap();

    let result = joined
        .groupby(["custkey", "returnflag"])
        .agg(&[AggSpec::new("revenue", GroupByAgg::Sum)])
        .unwrap()
        .sort_values(["custkey", "returnflag"], SortOptions::new())
        .unwrap();
    let host = result.to_record_batch().unwrap();

    assert_eq!(host.num_rows(), 4);
    assert_eq!(host.schema().field(0).name(), "custkey");
    assert_eq!(host.schema().field(1).name(), "returnflag");
    assert_eq!(host.schema().field(2).name(), "revenue_sum");
    assert_eq!(*host.schema().field(2).data_type(), ArrowDataType::Decimal128(38, 4));
    assert_eq!(host.column(0).as_any().downcast_ref::<Int32Array>().unwrap().values().as_ref(), &[10, 10, 20, 30]);
    assert_eq!(
        host.column(1).as_any().downcast_ref::<StringArray>().unwrap().iter().collect::<Vec<_>>(),
        vec![Some("R"), Some("R-VOID"), Some("R"), Some("R")]
    );
    assert_eq!(
        host.column(2).as_any().downcast_ref::<Decimal128Array>().unwrap().values().as_ref(),
        &[950_000, 3_000_000, 3_000_000, 3_000_000]
    );
}

#[test]
fn tpch_pipeline_contract_tests_cases() {
    run_named_cases(&[(
        "mini_tpch_pipeline_composes_parquet_dates_decimals_like_join_groupby_sort_and_arrow_export",
        mini_tpch_pipeline_composes_parquet_dates_decimals_like_join_groupby_sort_and_arrow_export,
    )]);
}
