//! Integration tests for `Series` row-selection convenience methods.
//!
//! Methods under test: `head`, `tail`, `slice`, `gather`, `filter`,
//! `dropna`, `drop_duplicates`, `value_counts`.
//!
//! All tests require a GPU.

use cudf::{ErrorCode, ErrorKind, Series};

use crate::common::{column_from_slice, error_context_value};

#[test]
fn test_head_basic() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let head = s.head(2).unwrap();
    assert_eq!(head.to_vec_strict::<i32>().unwrap(), vec![10, 20]);
}

#[test]
fn test_tail_clamps_to_len() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let tail = s.tail(99).unwrap();
    assert_eq!(tail.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_slice_preserves_unnamed_series() {
    let column = column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let s = Series::from_column(column).unwrap();
    let sliced = s.slice(1, 2).unwrap();
    assert_eq!(sliced.to_vec_strict::<i32>().unwrap(), vec![20, 30]);
}

#[test]
fn test_filter_drops_false_and_null_mask_rows() {
    let values = crate::common::series::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let mask = crate::common::nullable_series::<bool>("mask", &[true, false, false, true], &[true, true, false, true])
        .unwrap();

    let filtered = values.filter(&mask).unwrap();
    assert_eq!(filtered.to_vec_strict::<i32>().unwrap(), vec![10, 40]);
}

#[test]
fn test_gather_out_of_bounds_nullifies_rows() {
    let values = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let indices = crate::common::series::<i32>("idx", &[2, 99, 0]).unwrap();

    let gathered = values.gather(&indices).unwrap();
    assert_eq!(gathered.to_optional_vec::<i32>().unwrap(), vec![Some(30), None, Some(10)]);
}

#[test]
fn test_dropna_removes_null_rows() {
    let values = crate::common::series_opt("x", &[Some(10i32), None, Some(30), None]).unwrap();

    let cleaned = values.dropna().unwrap();
    assert_eq!(cleaned.null_count(), 0);
    assert_eq!(cleaned.to_vec_strict::<i32>().unwrap(), vec![10, 30]);
}

#[test]
fn test_drop_duplicates_returns_unique_values() {
    let values = crate::common::series::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();

    let deduped = values.drop_duplicates().unwrap();
    let mut host = deduped.to_vec_strict::<i32>().unwrap();
    host.sort();
    assert_eq!(host, vec![1, 2, 3]);
}

#[test]
fn test_value_counts_returns_schema_neutral_value_and_count_columns() {
    let values = crate::common::series_opt("x", &[Some(2i32), Some(1), None, Some(2), Some(1), Some(2)]).unwrap();

    let counts = values.value_counts().unwrap();

    assert_eq!(counts.column_names(), vec!["value", "count"]);
    let mut rows: Vec<(i32, i32)> = counts
        .series("value")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(counts.series("count").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 2), (2, 3)]);
}

#[test]
fn test_value_counts_for_unnamed_series_uses_fallback_value_name() {
    let column = column_from_slice(&[5i32, 5, 7]).unwrap();
    let values = Series::from_column(column).unwrap();

    let counts = values.value_counts().unwrap();

    assert_eq!(counts.column_names(), vec!["value", "count"]);
    let mut rows: Vec<(i32, i32)> = counts
        .series("value")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(counts.series("count").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(5, 2), (7, 1)]);
}

// -- Semantic lock-down: stable across the column-primitive refactor --------

#[test]
fn test_head_on_empty_series_returns_empty_with_name() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let head = s.head(10).unwrap();
    assert_eq!(head.len(), 0);
    assert_eq!(head.to_vec_strict::<i32>().unwrap(), Vec::<i32>::new());
}

#[test]
fn test_slice_offset_beyond_len_returns_empty() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let sliced = s.slice(10, 5).unwrap();
    assert_eq!(sliced.len(), 0);
}

#[test]
fn test_gather_empty_index_returns_empty_with_name() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let idx = crate::common::series::<i32>("idx", &[]).unwrap();

    let gathered = s.gather(&idx).unwrap();
    assert_eq!(gathered.len(), 0);
}

#[test]
fn test_gather_operation_tag_on_dtype_error() {
    let s = crate::common::series::<i32>("x", &[10, 20]).unwrap();
    // Indices must be Int32; pass Float64 to trigger dtype error.
    let bogus = crate::common::series::<f64>("idx", &[0.0, 1.0]).unwrap();

    let err = s.gather(&bogus).unwrap_err();
    assert_eq!(
        err.operation(),
        "series::gather",
        "error should surface through the series op name, not the internal dataframe path"
    );
}

#[test]
fn test_filter_operation_tag_on_dtype_error() {
    let s = crate::common::series::<i32>("x", &[10, 20]).unwrap();
    // Mask must be BOOL8; pass Int32 to trigger dtype error.
    let bogus = crate::common::series::<i32>("mask", &[1, 0]).unwrap();

    let err = s.filter(&bogus).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "series::filter");
    assert_eq!(error_context_value(&err, "left_type_id"), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id"), Some("bool8"));
}

#[test]
fn test_filter_length_mismatch_uses_canonical_length_contract() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let short_mask = crate::common::series::<bool>("mask", &[true, false]).unwrap();

    let err = s.filter(&short_mask).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "series::filter");
    assert_eq!(error_context_value(&err, "actual_rows"), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows"), Some("3"));
}

#[test]
fn test_dropna_on_non_nullable_returns_same_values() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let out = s.dropna().unwrap();
    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_drop_duplicates_preserves_unnamed_series() {
    let column = column_from_slice(&[2i32, 2, 3]).unwrap();
    let s = Series::from_column(column).unwrap();
    let deduped = s.drop_duplicates().unwrap();
    let mut host = deduped.to_vec_strict::<i32>().unwrap();
    host.sort();
    assert_eq!(host, vec![2, 3]);
}

#[test]
fn test_value_counts_with_sentinel_collision_name() {
    // value_counts now returns schema-neutral output names regardless of the
    // source series name, including names that match the old sentinel.
    let values = crate::common::series::<i32>("__cudf_series_value__", &[1, 1, 2]).unwrap();

    let counts = values.value_counts().unwrap();
    assert_eq!(counts.column_names(), vec!["value", "count"]);
}

#[test]
fn test_testing_hook_locks_value_counts_missing_key_columns_contract() {
    let err = cudf::testing::trigger_value_counts_missing_key_columns_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "series::value_counts");
    assert_eq!(error_context_value(&err, "resource"), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason"), Some("missing_key_columns"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some(
            "ensure the foreign groupby backend returns one key column and one non-empty count result for value_counts"
        )
    );
    assert_eq!(error_context_value(&err, "table_column_count"), Some("0"));
}

#[test]
fn test_testing_hook_locks_value_counts_missing_count_results_contract() {
    let err = cudf::testing::trigger_value_counts_missing_count_results_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "series::value_counts");
    assert_eq!(error_context_value(&err, "resource"), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason"), Some("missing_count_results"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some(
            "ensure the foreign groupby backend returns one key column and one non-empty count result for value_counts"
        )
    );
    assert_eq!(error_context_value(&err, "result_group_count"), Some("0"));
}

#[test]
fn test_testing_hook_locks_value_counts_empty_count_result_contract() {
    let err = cudf::testing::trigger_value_counts_empty_count_result_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "series::value_counts");
    assert_eq!(error_context_value(&err, "resource"), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason"), Some("empty_count_result"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some(
            "ensure the foreign groupby backend returns one key column and one non-empty count result for value_counts"
        )
    );
    assert_eq!(error_context_value(&err, "result_column_count"), Some("0"));
}

#[test]
fn test_testing_hook_locks_series_row_range_start_limit_contract() {
    let err = cudf::testing::trigger_series_row_range_start_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::slice");
    assert_eq!(err.message(), "row range start exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource"), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason"), Some("start_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy"), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row"), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row"), None);
}

#[test]
fn test_testing_hook_locks_series_row_range_end_limit_contract() {
    let err = cudf::testing::trigger_series_row_range_end_limit_error_for_test().unwrap_err();
    let expected_end = (i32::MAX as usize).checked_add(1).unwrap().to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::slice");
    assert_eq!(err.message(), "row range end exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource"), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason"), Some("end_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy"), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row"), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row"), Some(expected_end.as_str()));
}

#[test]
fn test_testing_hook_locks_series_row_range_overflow_contract() {
    let err = cudf::testing::trigger_series_row_range_overflow_error_for_test().unwrap_err();
    let expected_end = ((i32::MAX as u128) + (usize::MAX as u128)).to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::slice");
    assert_eq!(err.message(), "row range end overflowed usize");
    assert_eq!(error_context_value(&err, "resource"), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason"), Some("end_overflowed_usize"));
    assert_eq!(error_context_value(&err, "remedy"), Some("reduce row range bounds to fit within usize"));
    assert_eq!(error_context_value(&err, "start_row"), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row"), Some(expected_end.as_str()));
    assert_eq!(error_context_value(&err, "length"), None);
}
