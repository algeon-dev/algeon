//! Public API smoke tests.
//!
//! These intentionally keep only a small amount of import-surface coverage:
//! one crate-root workflow smoke test, one runtime-first factory smoke test,
//! one scoped control-plane smoke test, and one advanced low-level access
//! smoke test.

#[test]
fn crate_root_doc_workflow_smoke() {
    use cudf::{DataFrame, Runtime};

    let rt = Runtime::builder().build().unwrap();
    let ids = rt.series("id", &[1i32, 2, 3]).unwrap();
    let scores = rt.series("score", &[10.0f64, 20.0, 30.0]).unwrap();
    let df = DataFrame::from_series([ids, scores]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);

    let score_series = df.series("score").unwrap();
    assert_eq!(score_series.sum_value::<f64>().unwrap(), 60.0);
    assert_eq!(score_series.mean_value().unwrap(), Some(20.0));
}

#[test]
fn runtime_first_factories_smoke() {
    use cudf::{Runtime, TypeId, core::Scalar};

    let rt = Runtime::builder().build().unwrap();
    let ints = rt.series("ints", &[1i32, 2, 3]).unwrap();
    let nullable = rt.nullable_series("nullable", &[10i32, 0, 30], &[true, false, true]).unwrap();
    let strings = rt.str_series("names", &["alice", "bob"]).unwrap();
    let nulls = rt.nulls("missing", TypeId::Int32, 2).unwrap();
    let repeated = rt.series_from_scalar("seed", &Scalar::from_value(7i32).unwrap(), 3).unwrap();

    assert_eq!(ints.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(nullable.to_optional_vec::<i32>().unwrap(), vec![Some(10), None, Some(30)]);
    assert_eq!(strings.to_strings_strict().unwrap(), vec!["alice".to_string(), "bob".to_string()]);
    assert_eq!(nulls.null_count(), 2);
    assert_eq!(repeated.to_vec_strict::<i32>().unwrap(), vec![7, 7, 7]);
}

#[test]
fn scoped_control_plane_smoke() {
    use cudf::{AggSpec, DataFrame, ExecutionContext, GroupByAgg, Runtime, SortOptions, runtime::CudaStream};

    let rt = Runtime::builder().build().unwrap();
    let df = DataFrame::from_series([
        rt.series("id", &[2i32, 1, 2]).unwrap(),
        rt.series("value", &[10i64, 20, 30]).unwrap(),
    ])
    .unwrap();

    let override_stream = CudaStream::new().unwrap();
    let ctx = ExecutionContext::new().with_stream(&override_stream);

    let sorted = df.in_context(&ctx).sort_values(["id"], SortOptions::new()).unwrap();
    let grouped = sorted.groupby(["id"]).in_context(&ctx).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    assert_eq!(grouped.column_names(), vec!["id", "value_sum"]);
    assert_eq!(grouped.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(grouped.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap(), vec![20, 40]);
}

#[test]
fn advanced_core_and_low_level_ops_smoke() {
    use cudf::{DataFrame, Series, ops::copying};

    let ids = Series::new::<i32>("id", &[10, 20, 30]).unwrap();
    let scores = Series::new::<f64>("score", &[1.0, 2.0, 3.0]).unwrap();
    let df = DataFrame::from_series([ids, scores]).unwrap();
    let gather_map = cudf::core::Column::from_slice(&[2i32, 0]).unwrap();

    let table_view = df.table_view().unwrap();
    let gathered = copying::gather(table_view.table(), &gather_map.view().unwrap()).unwrap();
    let gathered = DataFrame::from_table(gathered, vec!["id".to_string(), "score".to_string()]).unwrap();

    assert_eq!(gathered.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(30), Some(10)]);
    assert_eq!(gathered.series("score").unwrap().to_optional_vec::<f64>().unwrap(), vec![Some(3.0), Some(1.0)]);
}
