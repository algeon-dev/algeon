//! Methods under test: `head`, `tail`, `slice`, `gather`, `filter`,
//! `dropna`, `drop_duplicates`, `value_counts`.

use cudf::{
    ErrorCode, ErrorKind, Exec, Series,
    core::{
        error::{FactKeyName, FactValue},
        types::TypeId,
    },
    testing::{
        groupby::{
            trigger_value_counts_empty_count_result_for_test, trigger_value_counts_missing_count_results_for_test,
            trigger_value_counts_missing_key_columns_for_test,
        },
        series::{
            trigger_series_row_range_end_limit_error_for_test, trigger_series_row_range_overflow_error_for_test,
            trigger_series_row_range_start_limit_error_for_test,
        },
    },
};

use crate::common::{column_from_slice, error_context_value, nullable_series, series, series_opt};

fn test_head_basic() {
    let s = series::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let head = s.on(Exec::RETAINED).head(2).unwrap();
    assert_eq!(head.to_vec_strict::<i32>().unwrap(), vec![10, 20]);
}

fn test_tail_clamps_to_len() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let tail = s.on(Exec::RETAINED).tail(99).unwrap();
    assert_eq!(tail.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

fn test_slice_preserves_unnamed_series() {
    let column = column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let s = Series::from_column(column).unwrap();
    let sliced = s.on(Exec::RETAINED).slice(1, 2).unwrap();
    assert_eq!(sliced.to_vec_strict::<i32>().unwrap(), vec![20, 30]);
}

fn test_filter_drops_false_and_null_mask_rows() {
    let values = series::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let mask = nullable_series::<bool>("mask", &[true, false, false, true], &[true, true, false, true]).unwrap();

    let filtered = values.on(Exec::RETAINED).filter(&mask).unwrap();
    assert_eq!(filtered.to_vec_strict::<i32>().unwrap(), vec![10, 40]);
}

fn test_gather_out_of_bounds_nullifies_rows() {
    let values = series::<i32>("x", &[10, 20, 30]).unwrap();
    let indices = series::<i32>("idx", &[2, 99, 0]).unwrap();

    let gathered = values.on(Exec::RETAINED).gather(&indices).unwrap();
    assert_eq!(gathered.to_optional_vec::<i32>().unwrap(), vec![Some(30), None, Some(10)]);
}

fn test_dropna_removes_null_rows() {
    let values = series_opt("x", &[Some(10i32), None, Some(30), None]).unwrap();

    let cleaned = values.on(Exec::RETAINED).dropna().unwrap();
    assert_eq!(cleaned.null_count(), 0);
    assert_eq!(cleaned.to_vec_strict::<i32>().unwrap(), vec![10, 30]);
}

fn test_drop_duplicates_returns_unique_values() {
    let values = series::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();

    let deduped = values.on(Exec::RETAINED).drop_duplicates().unwrap();
    let mut host = deduped.to_vec_strict::<i32>().unwrap();
    host.sort();
    assert_eq!(host, vec![1, 2, 3]);
}

fn test_value_counts_returns_schema_neutral_value_and_count_columns() {
    let values = series_opt("x", &[Some(2i32), Some(1), None, Some(2), Some(1), Some(2)]).unwrap();

    let counts = values.on(Exec::RETAINED).value_counts().unwrap();

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

fn test_value_counts_for_unnamed_series_uses_fallback_value_name() {
    let column = column_from_slice(&[5i32, 5, 7]).unwrap();
    let values = Series::from_column(column).unwrap();

    let counts = values.on(Exec::RETAINED).value_counts().unwrap();

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

// The tests below lock row-selection semantics across the column-primitive refactor.

fn test_head_on_empty_series_returns_empty_with_name() {
    let s = series::<i32>("x", &[]).unwrap();
    let head = s.on(Exec::RETAINED).head(10).unwrap();
    assert_eq!(head.len(), 0);
    assert_eq!(head.to_vec_strict::<i32>().unwrap(), Vec::<i32>::new());
}

fn test_slice_offset_beyond_len_returns_empty() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let sliced = s.on(Exec::RETAINED).slice(10, 5).unwrap();
    assert_eq!(sliced.len(), 0);
}

fn test_gather_empty_index_returns_empty_with_name() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let idx = series::<i32>("idx", &[]).unwrap();

    let gathered = s.on(Exec::RETAINED).gather(&idx).unwrap();
    assert_eq!(gathered.len(), 0);
}

fn test_gather_operation_tag_on_dtype_error() {
    let s = series::<i32>("x", &[10, 20]).unwrap();
    // Indices must be Int32; pass Float64 to trigger dtype error.
    let bogus = series::<f64>("idx", &[0.0, 1.0]).unwrap();

    let err = s.on(Exec::RETAINED).gather(&bogus).unwrap_err();
    assert_eq!(
        err.operation(),
        "series::gather",
        "error should surface through the series op name, not the internal dataframe path"
    );
}

fn test_filter_operation_tag_on_dtype_error() {
    let s = series::<i32>("x", &[10, 20]).unwrap();
    // Mask must be BOOL8; pass Int32 to trigger dtype error.
    let bogus = series::<i32>("mask", &[1, 0]).unwrap();

    let err = s.on(Exec::RETAINED).filter(&bogus).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "series::filter");
    assert_eq!(err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Bool8)));
}

fn test_filter_length_mismatch_uses_canonical_length_contract() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let short_mask = series::<bool>("mask", &[true, false]).unwrap();

    let err = s.on(Exec::RETAINED).filter(&short_mask).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "series::filter");
    assert_eq!(err.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(2)));
    assert_eq!(err.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(3)));
}

fn test_dropna_on_non_nullable_returns_same_values() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let out = s.on(Exec::RETAINED).dropna().unwrap();
    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_value_counts_with_sentinel_collision_name() {
    // value_counts returns schema-neutral output names regardless of the source series name,
    // including `__cudf_series_value__`.
    let values = series::<i32>("__cudf_series_value__", &[1, 1, 2]).unwrap();

    let counts = values.on(Exec::RETAINED).value_counts().unwrap();
    assert_eq!(counts.column_names(), vec!["value", "count"]);
}

fn test_testing_hook_locks_value_counts_missing_key_columns_contract() {
    let err = trigger_value_counts_missing_key_columns_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "series::value_counts");
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(0)));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("missing_key_columns"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "ensure the foreign groupby backend returns one key column and one non-empty count result for value_counts"
        )
    );
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("0"));
}

fn test_testing_hook_locks_value_counts_missing_count_results_contract() {
    let err = trigger_value_counts_missing_count_results_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "series::value_counts");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("missing_count_results"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "ensure the foreign groupby backend returns one key column and one non-empty count result for value_counts"
        )
    );
    assert_eq!(error_context_value(&err, "result_group_count").as_deref(), Some("0"));
}

fn test_testing_hook_locks_value_counts_empty_count_result_contract() {
    let err = trigger_value_counts_empty_count_result_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "series::value_counts");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("groupby_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("empty_count_result"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "ensure the foreign groupby backend returns one key column and one non-empty count result for value_counts"
        )
    );
    assert_eq!(error_context_value(&err, "result_column_count").as_deref(), Some("0"));
}

fn test_testing_hook_locks_series_row_range_start_limit_contract() {
    let err = trigger_series_row_range_start_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::slice");
    assert_eq!(err.message(), "row range start exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("start_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), None);
}

fn test_testing_hook_locks_series_row_range_end_limit_contract() {
    let err = trigger_series_row_range_end_limit_error_for_test().unwrap_err();
    let expected_end = (i32::MAX as usize).checked_add(1).unwrap().to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::slice");
    assert_eq!(err.message(), "row range end exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("end_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), Some(expected_end.as_str()));
}

fn test_testing_hook_locks_series_row_range_overflow_contract() {
    let err = trigger_series_row_range_overflow_error_for_test().unwrap_err();
    let expected_end = ((i32::MAX as u128) + (usize::MAX as u128)).to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::slice");
    assert_eq!(err.message(), "row range end overflowed usize");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("end_overflowed_usize"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within usize"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), Some(expected_end.as_str()));
    assert_eq!(error_context_value(&err, "length").as_deref(), None);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("row_ops_tests::test_head_basic", test_head_basic),
    ("row_ops_tests::test_tail_clamps_to_len", test_tail_clamps_to_len),
    ("row_ops_tests::test_slice_preserves_unnamed_series", test_slice_preserves_unnamed_series),
    ("row_ops_tests::test_filter_drops_false_and_null_mask_rows", test_filter_drops_false_and_null_mask_rows),
    ("row_ops_tests::test_gather_out_of_bounds_nullifies_rows", test_gather_out_of_bounds_nullifies_rows),
    ("row_ops_tests::test_dropna_removes_null_rows", test_dropna_removes_null_rows),
    ("row_ops_tests::test_drop_duplicates_returns_unique_values", test_drop_duplicates_returns_unique_values),
    (
        "row_ops_tests::test_value_counts_returns_schema_neutral_value_and_count_columns",
        test_value_counts_returns_schema_neutral_value_and_count_columns,
    ),
    (
        "row_ops_tests::test_value_counts_for_unnamed_series_uses_fallback_value_name",
        test_value_counts_for_unnamed_series_uses_fallback_value_name,
    ),
    (
        "row_ops_tests::test_head_on_empty_series_returns_empty_with_name",
        test_head_on_empty_series_returns_empty_with_name,
    ),
    ("row_ops_tests::test_slice_offset_beyond_len_returns_empty", test_slice_offset_beyond_len_returns_empty),
    ("row_ops_tests::test_gather_empty_index_returns_empty_with_name", test_gather_empty_index_returns_empty_with_name),
    ("row_ops_tests::test_gather_operation_tag_on_dtype_error", test_gather_operation_tag_on_dtype_error),
    ("row_ops_tests::test_filter_operation_tag_on_dtype_error", test_filter_operation_tag_on_dtype_error),
    (
        "row_ops_tests::test_filter_length_mismatch_uses_canonical_length_contract",
        test_filter_length_mismatch_uses_canonical_length_contract,
    ),
    ("row_ops_tests::test_dropna_on_non_nullable_returns_same_values", test_dropna_on_non_nullable_returns_same_values),
    ("row_ops_tests::test_value_counts_with_sentinel_collision_name", test_value_counts_with_sentinel_collision_name),
    (
        "row_ops_tests::test_testing_hook_locks_value_counts_missing_key_columns_contract",
        test_testing_hook_locks_value_counts_missing_key_columns_contract,
    ),
    (
        "row_ops_tests::test_testing_hook_locks_value_counts_missing_count_results_contract",
        test_testing_hook_locks_value_counts_missing_count_results_contract,
    ),
    (
        "row_ops_tests::test_testing_hook_locks_value_counts_empty_count_result_contract",
        test_testing_hook_locks_value_counts_empty_count_result_contract,
    ),
    (
        "row_ops_tests::test_testing_hook_locks_series_row_range_start_limit_contract",
        test_testing_hook_locks_series_row_range_start_limit_contract,
    ),
    (
        "row_ops_tests::test_testing_hook_locks_series_row_range_end_limit_contract",
        test_testing_hook_locks_series_row_range_end_limit_contract,
    ),
    (
        "row_ops_tests::test_testing_hook_locks_series_row_range_overflow_contract",
        test_testing_hook_locks_series_row_range_overflow_contract,
    ),
];
