//! Tests for the `DataFrame::groupby` high-level API: `GroupBy`, `GroupByAgg`, `AggSpec`,
//! `GroupByOptions`.

use cudf::{AggSpec, DataFrame, ErrorCode, ErrorKind, Exec, GroupByAgg};

use crate::common::{assert_missing_column_error, column_from_slice, dataframe_from_columns, error_context_value};

fn assert_groupby_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

fn assert_ddof_range_validation_error(err: &cudf::Error, operation: &str, request_index: &str, expected_ddof: u32) {
    assert_groupby_validation_error(
        err,
        operation,
        "ddof_exceeds_i32_range",
        "use a ddof value within libcudf's i32 range",
    );
    assert_eq!(error_context_value(err, "request_index").as_deref(), Some(request_index));
    assert_eq!(error_context_value(err, "ddof").as_deref(), Some(expected_ddof.to_string().as_str()));
}

// ---------------------------------------------------------------------------
// GroupBy::agg with AggSpec; basic aggregations
// ---------------------------------------------------------------------------

fn groupby_sum() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let sum_data: Vec<i64> = result.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

fn groupby_mean() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Mean)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let mean_data: Vec<f64> = result.series("value_mean").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

fn groupby_count() {
    let keys = column_from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("value", GroupByAgg::Count { include_nulls: false })])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let count_data: Vec<i32> = result.series("value_count").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

// ---------------------------------------------------------------------------
// Output column naming: alias support
// ---------------------------------------------------------------------------

fn groupby_alias() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("value", GroupByAgg::Sum).with_alias("total")])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "total"]);

    let sum_data: Vec<i64> = result.series("total").unwrap().to_vec_strict::<i64>().unwrap();
    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

fn groupby_deterministic_naming_without_alias() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let prices = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = dataframe_from_columns(vec!["id".into(), "price".into()], vec![keys, prices]).unwrap();

    let result = df
        .groupby(["id"])
        .on(Exec::RETAINED)
        .agg(&[
            AggSpec::new("price", GroupByAgg::Sum),
            AggSpec::new("price", GroupByAgg::Mean),
            AggSpec::new("price", GroupByAgg::Count { include_nulls: false }),
            AggSpec::new("price", GroupByAgg::Min),
            AggSpec::new("price", GroupByAgg::Max),
        ])
        .unwrap();

    let names = result.column_names();
    assert_eq!(names[0], "id");
    assert_eq!(names[1], "price_sum");
    assert_eq!(names[2], "price_mean");
    assert_eq!(names[3], "price_count");
    assert_eq!(names[4], "price_min");
    assert_eq!(names[5], "price_max");
}

// ---------------------------------------------------------------------------
// Multi-column aggregation ordering
// ---------------------------------------------------------------------------

fn groupby_multi_column_agg_ordering() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = column_from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[
            AggSpec::new("a", GroupByAgg::Sum),
            AggSpec::new("b", GroupByAgg::Sum),
            AggSpec::new("a", GroupByAgg::Min),
        ])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);

    let keys: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let a_sum: Vec<i64> = result.series("a_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let b_sum: Vec<i64> = result.series("b_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let a_min: Vec<i64> = result.series("a_min").unwrap().to_vec_strict::<i64>().unwrap();

    let mut rows: Vec<(i32, i64, i64, i64)> = keys
        .into_iter()
        .zip(a_sum.into_iter().zip(b_sum.into_iter().zip(a_min)))
        .map(|(k, (asum, (bsum, amin)))| (k, asum, bsum, amin))
        .collect();
    rows.sort_by_key(|&(k, _, _, _)| k);

    assert_eq!(rows, vec![(1, 30, 3, 10), (2, 70, 7, 30)]);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

fn groupby_missing_value_column() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("nonexistent", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent column");

    assert_missing_column_error(&err, "nonexistent");
}

fn groupby_missing_key_column() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["missing_key"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("value", GroupByAgg::Sum)])
        .expect_err("expected error for nonexistent key column");

    assert_missing_column_error(&err, "missing_key");
}

fn groupby_empty_agg_specs() {
    let keys = column_from_slice(&[1i32, 2, 1]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df.groupby(["key"]).on(Exec::RETAINED).agg(&[]).expect_err("expected error for empty aggregation specs");

    assert_groupby_validation_error(
        &err,
        "groupby::GroupBy::agg",
        "empty_agg_specs",
        "provide at least one AggSpec or use a shortcut like .sum([...])",
    );
}

// ---------------------------------------------------------------------------
// Shortcut methods: sum, mean, count
// ---------------------------------------------------------------------------

fn groupby_sum_shortcut() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).sum(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_sum"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let sum_data: Vec<i64> = result.series("value_sum").unwrap().to_vec_strict::<i64>().unwrap();

    let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 40), (2, 60)]);
}

fn groupby_mean_shortcut() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).mean(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_mean"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let mean_data: Vec<f64> = result.series("value_mean").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(mean_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 20.0), (2, 30.0)]);
}

fn groupby_count_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 2, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).count(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_count"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let count_data: Vec<i32> = result.series("value_count").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(count_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 3)]);
}

fn groupby_shortcut_handle_is_reusable() {
    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let groupby = df.groupby(["key"]);
    let summed = groupby.on(Exec::RETAINED).sum(["value"]).unwrap();
    let counted = groupby.on(Exec::RETAINED).count(["value"]).unwrap();

    assert_eq!(summed.column_names(), vec!["key", "value_sum"]);
    assert_eq!(counted.column_names(), vec!["key", "value_count"]);
}

fn scoped_groupby_handle_is_reusable() {
    use cudf::{Exec, Runtime, runtime::CudaStream};

    let rt = Runtime::builder().build().unwrap();
    let df = DataFrame::from_series([
        ("key", rt.series(&[1i32, 2, 1, 2]).unwrap()),
        ("value", rt.series(&[10i64, 20, 30, 40]).unwrap()),
    ])
    .unwrap();

    let stream = CudaStream::new().unwrap();
    let ctx = Exec::RETAINED.with_stream(&stream);
    let groupby = df.groupby(["key"]).on(ctx);
    let summed = groupby.sum(["value"]).unwrap();
    let averaged = groupby.mean(["value"]).unwrap();

    assert_eq!(summed.column_names(), vec!["key", "value_sum"]);
    assert_eq!(averaged.column_names(), vec!["key", "value_mean"]);
}

// ---------------------------------------------------------------------------
// Shortcut with multiple value columns
// ---------------------------------------------------------------------------

fn groupby_sum_shortcut_multi_columns() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = column_from_slice(&[1i64, 2, 3, 4]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "a".into(), "b".into()], vec![keys, a, b]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).sum(["a", "b"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum"]);

    let keys: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let a_sum: Vec<i64> = result.series("a_sum").unwrap().to_vec_strict::<i64>().unwrap();
    let b_sum: Vec<i64> = result.series("b_sum").unwrap().to_vec_strict::<i64>().unwrap();

    let mut rows: Vec<(i32, i64, i64)> =
        keys.into_iter().zip(a_sum.into_iter().zip(b_sum)).map(|(k, (a, b))| (k, a, b)).collect();
    rows.sort_by_key(|&(k, _, _)| k);

    assert_eq!(rows, vec![(1, 30, 3), (2, 70, 7)]);
}

// ---------------------------------------------------------------------------
// GroupByOptions: include_null_keys
// ---------------------------------------------------------------------------

fn groupby_with_options_include_null_keys() {
    use cudf::GroupByOptions;

    let keys = column_from_slice(&[1i32, 2, 1, 2]).unwrap();
    let vals = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let options = GroupByOptions::builder().include_null_keys(true).build();
    let result =
        df.groupby(["key"]).options(options).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 2);
}

// ===========================================================================
// Expanded GroupByAgg enum; Median, NUnique, Std, Variance
// ===========================================================================

// ---------------------------------------------------------------------------
// GroupByAgg::Median
// ---------------------------------------------------------------------------

fn groupby_median() {
    // keys: [1, 1, 1, 2, 2], values: [10.0, 20.0, 30.0, 100.0, 200.0]
    // key=1 median=20.0, key=2 median=150.0
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Median)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_median"]);

    let key_data: Vec<Option<i32>> = result.series("key").unwrap().to_optional_vec::<i32>().unwrap();
    let med_data: Vec<Option<f64>> = result.series("value_median").unwrap().to_optional_vec::<f64>().unwrap();

    let mut pairs: Vec<(Option<i32>, Option<f64>)> = key_data.into_iter().zip(med_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(Some(1), Some(20.0)), (Some(2), Some(150.0))]);
}

fn groupby_median_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 30.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).median(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_median"]);

    let key_data: Vec<Option<i32>> = result.series("key").unwrap().to_optional_vec::<i32>().unwrap();
    let med_data: Vec<Option<f64>> = result.series("value_median").unwrap().to_optional_vec::<f64>().unwrap();

    let mut pairs: Vec<(Option<i32>, Option<f64>)> = key_data.into_iter().zip(med_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(Some(1), Some(20.0)), (Some(2), Some(150.0))]);
}

// ---------------------------------------------------------------------------
// GroupByAgg::NUnique
// ---------------------------------------------------------------------------

fn groupby_nunique() {
    // keys: [1, 1, 1, 2, 2], values: [10, 10, 20, 30, 30]
    // key=1 nunique=2, key=2 nunique=1
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 10, 20, 30, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::NUnique)]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_nunique"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let nu_data: Vec<i32> = result.series("value_nunique").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(nu_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 1)]);
}

fn groupby_nunique_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10i32, 10, 20, 30, 30]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df.groupby(["key"]).on(Exec::RETAINED).nunique(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_nunique"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let nu_data: Vec<i32> = result.series("value_nunique").unwrap().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<(i32, i32)> = key_data.into_iter().zip(nu_data).collect();
    pairs.sort_by_key(|&(k, _)| k);
    assert_eq!(pairs, vec![(1, 2), (2, 1)]);
}

// ---------------------------------------------------------------------------
// GroupByAgg::Std
// ---------------------------------------------------------------------------

fn groupby_std_ddof1() {
    // keys: [1, 1, 2, 2], values: [10.0, 20.0, 100.0, 200.0]
    // key=1: std(ddof=1) = sqrt(((10-15)^2 + (20-15)^2) / 1) = sqrt(50) ≈ 7.0711
    // key=2: std(ddof=1) = sqrt(((100-150)^2 + (200-150)^2) / 1) = sqrt(5000) ≈ 70.7107
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result =
        df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Std { ddof: 1 })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_std"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0, 1);
    assert!((pairs[0].1 - 50.0f64.sqrt()).abs() < 1e-6, "key=1 std mismatch: {}", pairs[0].1);
    assert_eq!(pairs[1].0, 2);
    assert!((pairs[1].1 - 5000.0f64.sqrt()).abs() < 1e-6, "key=2 std mismatch: {}", pairs[1].1);
}

fn groupby_std_ddof0() {
    // key=1: std(ddof=0) = sqrt(((10-15)^2 + (20-15)^2) / 2) = sqrt(25) = 5.0
    // key=2: std(ddof=0) = sqrt(((100-150)^2 + (200-150)^2) / 2) = sqrt(2500) = 50.0
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result =
        df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Std { ddof: 0 })]).unwrap();

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 5.0).abs() < 1e-6, "key=1 std(ddof=0) mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 50.0).abs() < 1e-6, "key=2 std(ddof=0) mismatch: {}", pairs[1].1);
}

fn groupby_std_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    // Default ddof=1 for the shortcut.
    let result = df.groupby(["key"]).on(Exec::RETAINED).std(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_std"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let std_data: Vec<f64> = result.series("value_std").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(std_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0f64.sqrt()).abs() < 1e-6);
    assert!((pairs[1].1 - 5000.0f64.sqrt()).abs() < 1e-6);
}

fn groupby_rejects_ddof_above_i32_max_with_stable_contract() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let err = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("value", GroupByAgg::Std { ddof: u32::MAX })])
        .expect_err("expected error");

    assert_ddof_range_validation_error(&err, "groupby::GroupBy::agg", "0", u32::MAX);
}

// ---------------------------------------------------------------------------
// GroupByAgg::Variance
// ---------------------------------------------------------------------------

fn groupby_variance_ddof1() {
    // key=1: var(ddof=1) = 50.0
    // key=2: var(ddof=1) = 5000.0
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result =
        df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Variance { ddof: 1 })]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_var"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0).abs() < 1e-6, "key=1 var mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 5000.0).abs() < 1e-6, "key=2 var mismatch: {}", pairs[1].1);
}

fn groupby_variance_ddof0() {
    // key=1: var(ddof=0) = 25.0
    // key=2: var(ddof=0) = 2500.0
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result =
        df.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Variance { ddof: 0 })]).unwrap();

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 25.0).abs() < 1e-6, "key=1 var(ddof=0) mismatch: {}", pairs[0].1);
    assert!((pairs[1].1 - 2500.0).abs() < 1e-6, "key=2 var(ddof=0) mismatch: {}", pairs[1].1);
}

fn groupby_var_shortcut() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    // Default ddof=1 for the shortcut.
    let result = df.groupby(["key"]).on(Exec::RETAINED).var(["value"]).unwrap();

    assert_eq!(result.column_names(), vec!["key", "value_var"]);

    let key_data: Vec<i32> = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
    let var_data: Vec<f64> = result.series("value_var").unwrap().to_vec_strict::<f64>().unwrap();

    let mut pairs: Vec<(i32, f64)> = key_data.into_iter().zip(var_data).collect();
    pairs.sort_by_key(|&(k, _)| k);

    assert!((pairs[0].1 - 50.0).abs() < 1e-6);
    assert!((pairs[1].1 - 5000.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Extended naming; all GroupByAgg variants in a single groupby
// ---------------------------------------------------------------------------

fn groupby_all_agg_variants_naming() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[
            AggSpec::new("value", GroupByAgg::Sum),
            AggSpec::new("value", GroupByAgg::Mean),
            AggSpec::new("value", GroupByAgg::Min),
            AggSpec::new("value", GroupByAgg::Max),
            AggSpec::new("value", GroupByAgg::Count { include_nulls: false }),
            AggSpec::new("value", GroupByAgg::Median),
            AggSpec::new("value", GroupByAgg::NUnique),
            AggSpec::new("value", GroupByAgg::Std { ddof: 1 }),
            AggSpec::new("value", GroupByAgg::Variance { ddof: 1 }),
        ])
        .unwrap();

    assert_eq!(
        result.column_names(),
        vec![
            "key",
            "value_sum",
            "value_mean",
            "value_min",
            "value_max",
            "value_count",
            "value_median",
            "value_nunique",
            "value_std",
            "value_var",
        ]
    );
}

// ---------------------------------------------------------------------------
// Alias works with new variants
// ---------------------------------------------------------------------------

fn groupby_new_agg_alias() {
    let keys = column_from_slice(&[1i32, 1, 2, 2]).unwrap();
    let vals = column_from_slice(&[10.0f64, 20.0, 100.0, 200.0]).unwrap();
    let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals]).unwrap();

    let result = df
        .groupby(["key"])
        .on(Exec::RETAINED)
        .agg(&[
            AggSpec::new("value", GroupByAgg::Median).with_alias("mid"),
            AggSpec::new("value", GroupByAgg::Std { ddof: 1 }).with_alias("spread"),
        ])
        .unwrap();

    assert_eq!(result.column_names(), vec!["key", "mid", "spread"]);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("high_level_groupby_tests::groupby_sum", groupby_sum),
    ("high_level_groupby_tests::groupby_mean", groupby_mean),
    ("high_level_groupby_tests::groupby_count", groupby_count),
    ("high_level_groupby_tests::groupby_alias", groupby_alias),
    (
        "high_level_groupby_tests::groupby_deterministic_naming_without_alias",
        groupby_deterministic_naming_without_alias,
    ),
    ("high_level_groupby_tests::groupby_multi_column_agg_ordering", groupby_multi_column_agg_ordering),
    ("high_level_groupby_tests::groupby_missing_value_column", groupby_missing_value_column),
    ("high_level_groupby_tests::groupby_missing_key_column", groupby_missing_key_column),
    ("high_level_groupby_tests::groupby_empty_agg_specs", groupby_empty_agg_specs),
    ("high_level_groupby_tests::groupby_sum_shortcut", groupby_sum_shortcut),
    ("high_level_groupby_tests::groupby_mean_shortcut", groupby_mean_shortcut),
    ("high_level_groupby_tests::groupby_count_shortcut", groupby_count_shortcut),
    ("high_level_groupby_tests::groupby_shortcut_handle_is_reusable", groupby_shortcut_handle_is_reusable),
    ("high_level_groupby_tests::scoped_groupby_handle_is_reusable", scoped_groupby_handle_is_reusable),
    ("high_level_groupby_tests::groupby_sum_shortcut_multi_columns", groupby_sum_shortcut_multi_columns),
    ("high_level_groupby_tests::groupby_with_options_include_null_keys", groupby_with_options_include_null_keys),
    ("high_level_groupby_tests::groupby_median", groupby_median),
    ("high_level_groupby_tests::groupby_median_shortcut", groupby_median_shortcut),
    ("high_level_groupby_tests::groupby_nunique", groupby_nunique),
    ("high_level_groupby_tests::groupby_nunique_shortcut", groupby_nunique_shortcut),
    ("high_level_groupby_tests::groupby_std_ddof1", groupby_std_ddof1),
    ("high_level_groupby_tests::groupby_std_ddof0", groupby_std_ddof0),
    ("high_level_groupby_tests::groupby_std_shortcut", groupby_std_shortcut),
    (
        "high_level_groupby_tests::groupby_rejects_ddof_above_i32_max_with_stable_contract",
        groupby_rejects_ddof_above_i32_max_with_stable_contract,
    ),
    ("high_level_groupby_tests::groupby_variance_ddof1", groupby_variance_ddof1),
    ("high_level_groupby_tests::groupby_variance_ddof0", groupby_variance_ddof0),
    ("high_level_groupby_tests::groupby_var_shortcut", groupby_var_shortcut),
    ("high_level_groupby_tests::groupby_all_agg_variants_naming", groupby_all_agg_variants_naming),
    ("high_level_groupby_tests::groupby_new_agg_alias", groupby_new_agg_alias),
];
