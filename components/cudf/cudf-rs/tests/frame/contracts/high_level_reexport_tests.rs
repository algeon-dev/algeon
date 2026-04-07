//! Tests that verify the crate root re-exports the high-level types correctly.
//!
//! These tests mirror the crate-level doc example to ensure it compiles and
//! works as documented. All tests require a GPU.

// ---------------------------------------------------------------------------
// Crate root re-exports: DataFrame, Series, and supporting types
// ---------------------------------------------------------------------------

#[test]
fn test_crate_root_reexports_series() {
    // Series should be importable directly from the crate root.
    let s = cudf::Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    assert_eq!(s.name(), Some("ids"));
    assert_eq!(s.len(), 3);
}

#[test]
fn test_crate_root_reexports_dataframe() {
    // DataFrame should be importable directly from the crate root.
    let ids = cudf::Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = cudf::Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();

    let df = cudf::DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();
    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);
}

#[test]
fn test_doc_example_high_level_workflow() {
    // This mirrors the crate-level doc example: construct a DataFrame
    // from Series, retrieve a Series back, and inspect metadata.
    use cudf::{DataFrame, Series};

    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);

    // Retrieve a column as a Series.
    let score_series = df.series("score").unwrap();
    assert_eq!(score_series.name(), Some("score"));
    assert_eq!(score_series.len(), 3);
}

#[test]
fn test_lowlevel_types_available_via_core_module() {
    // Ensure advanced types remain available through `cudf::core::*`.
    let col = cudf::core::Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 3);

    let _dt: cudf::DataType = view.data_type().unwrap();
    let _tid: cudf::TypeId = _dt.id();
    let _: Option<cudf::core::DeviceBuffer> = None;
    let _: Option<cudf::core::DeviceSlice<'static>> = None;
    let _scalar = cudf::core::Scalar::from_value(7i32).unwrap();
}

// ---------------------------------------------------------------------------
// Crate root re-exports
// ---------------------------------------------------------------------------

#[test]
fn test_sort_options_reexported() {
    // SortOptions should be importable directly from the crate root.
    use cudf::SortOptions;

    let opts = SortOptions::new().with_descending(vec![true]).with_stable(true);
    assert!(opts.stable);
    assert_eq!(opts.descending, vec![true]);
}

#[test]
fn test_join_types_reexported() {
    // JoinType and MergeOptions should be importable directly from the crate root.
    use cudf::{JoinType, MergeOptions};

    let opts = MergeOptions::new().on(["id"]).how(JoinType::Left).suffixes("_x", "_y");
    assert_eq!(opts.join_type(), JoinType::Left);
    assert_eq!(opts.suffix_left(), "_x");
    assert_eq!(opts.suffix_right(), "_y");
}

#[test]
fn test_groupby_types_reexported() {
    // Agg, AggSpec, GroupByOptions should be importable directly from the crate root.
    use cudf::{Agg, AggSpec, GroupByOptions};

    let spec = AggSpec::new("price", Agg::Sum).with_alias("total_price");
    // Verify it constructed without error.
    let _ = spec;

    let opts = GroupByOptions::builder().include_null_keys(true).build();
    let _ = opts;
}

#[test]
fn test_high_level_workflow_sort_values() {
    // End-to-end: construct DataFrame, sort_values using crate-root imports.
    use cudf::{DataFrame, Series, SortOptions};

    let ids = Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    let df = DataFrame::from_columns([("id", ids)]).unwrap();

    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
    let data: Vec<i32> = sorted.series("id").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(data, vec![1, 2, 3]);
}

#[test]
fn test_high_level_workflow_merge() {
    // End-to-end: construct two DataFrames, merge using crate-root imports.
    use cudf::{DataFrame, JoinType, MergeOptions, Series};

    let left_id = Series::new::<i32>("id", &[1, 2, 3]).unwrap();
    let left_val = Series::new::<i32>("score", &[10, 20, 30]).unwrap();
    let left = DataFrame::from_columns([("id", left_id), ("score", left_val)]).unwrap();

    let right_id = Series::new::<i32>("id", &[2, 3, 4]).unwrap();
    let right_val = Series::new::<i32>("rank", &[200, 300, 400]).unwrap();
    let right = DataFrame::from_columns([("id", right_id), ("rank", right_val)]).unwrap();

    let opts = MergeOptions::new().on(["id"]).how(JoinType::Inner);
    let merged = left.merge(&right, opts).unwrap();

    assert_eq!(merged.num_rows(), 2);
    assert!(merged.column_names().contains(&"score"));
    assert!(merged.column_names().contains(&"rank"));
}

#[test]
fn test_high_level_workflow_groupby() {
    // End-to-end: construct DataFrame, groupby+agg using crate-root imports.
    use cudf::{Agg, AggSpec, DataFrame, GroupByOptions, Series};

    let keys = Series::new::<i32>("key", &[1, 2, 1, 2]).unwrap();
    let vals = Series::new::<i64>("value", &[10, 20, 30, 40]).unwrap();
    let df = DataFrame::from_columns([("key", keys), ("value", vals)]).unwrap();

    let result =
        df.groupby(["key"]).options(GroupByOptions::default()).agg(&[AggSpec::new("value", Agg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.column_names(), vec!["key", "value_sum"]);
}

#[test]
fn test_advanced_workflow_uses_table_view_and_low_level_ops() {
    use cudf::{
        DataFrame, Series,
        ops::copying,
    };

    let ids = Series::new::<i32>("id", &[10, 20, 30]).unwrap();
    let scores = Series::new::<f64>("score", &[1.0, 2.0, 3.0]).unwrap();
    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();
    let gather_map = cudf::core::Column::from_slice(&[2i32, 0]).unwrap();

    let table_view = df.table_view().unwrap();
    let gathered = copying::gather(table_view.table(), &gather_map.view().unwrap()).unwrap();
    let gathered = DataFrame::from_table(gathered, vec!["id".to_string(), "score".to_string()]).unwrap();

    assert_eq!(gathered.series("id").unwrap().to_vec::<i32>().unwrap(), vec![30, 10]);
    assert_eq!(gathered.series("score").unwrap().to_vec::<f64>().unwrap(), vec![3.0, 1.0]);
}

#[test]
fn test_all_high_level_types_in_single_import() {
    // Verify all high-level types can be imported in one `use` statement.
    use cudf::{Agg, AggSpec, DataFrame, GroupBy, GroupByOptions, JoinType, MergeOptions, Series, SortOptions};

    // Merely asserting that this compiles is the test. Use each type to
    // avoid unused-import warnings.
    let _ = SortOptions::new();
    let _ = MergeOptions::new().how(JoinType::Inner);
    let _ = AggSpec::new("x", Agg::Sum);
    let _ = GroupByOptions::default();

    let s = Series::new::<i32>("a", &[1]).unwrap();
    let df = DataFrame::from_columns([("a", s)]).unwrap();
    let _gb: GroupBy<'_> = df.groupby(["a"]);
}

#[test]
fn test_domain_modules_export_documented_surface() {
    use cudf::{
        frame::{
            dataframe::DataFrame as DomainDataFrame,
            series::{IntoSeries as DomainIntoSeries, Series as DomainSeries},
        },
        ops::{
            Expr as DomainExpr,
            groupby::{
                Agg as DomainAgg, AggSpec as DomainAggSpec, GroupBy as DomainGroupBy,
                GroupByOptions as DomainGroupByOptions,
            },
            join::{
                JoinResult as DomainJoinResult, JoinType as DomainJoinType, MergeOptions as DomainMergeOptions,
                NullEquality as DomainNullEquality,
            },
            sorting::SortOptions as DomainSortOptions,
        },
        runtime::{
            CudaEvent as DomainCudaEvent, CudaStream as DomainCudaStream, ExecutionContext as DomainExecutionContext,
            Runtime as DomainRuntime, RuntimeBuilder as DomainRuntimeBuilder,
        },
    };

    fn assert_into_series<T: DomainIntoSeries>() {}

    let _: Option<DomainDataFrame> = None;
    let _: Option<DomainSeries> = None;
    let _: Option<DomainExpr> = None;
    let _: Option<DomainJoinResult> = None;
    let _: Option<DomainExecutionContext<'static>> = None;
    let _: Option<DomainRuntime> = None;
    let _: Option<DomainRuntimeBuilder> = None;

    let _ = DomainSortOptions::new();
    let _ = DomainMergeOptions::new().how(DomainJoinType::Inner);
    let _ = DomainGroupByOptions::builder().include_null_keys(true).build();
    let _ = DomainAggSpec::new("value", DomainAgg::Sum);
    let _ = DomainNullEquality::Equal;
    let _ = DomainCudaEvent::new().unwrap();
    let _ = DomainCudaStream::new().unwrap();
    let _expr = cudf::ops::col("value");

    let series = DomainSeries::new::<i32>("key", &[1, 2, 1]).unwrap();
    let df = DomainDataFrame::from_columns([("key", series)]).unwrap();
    let _gb: DomainGroupBy<'_> = df.groupby(["key"]);

    assert_into_series::<DomainSeries>();
}
