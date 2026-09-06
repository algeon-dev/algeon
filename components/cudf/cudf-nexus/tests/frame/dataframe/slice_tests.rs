//! Tests for DataFrame row slicing helpers.

use cudf::{DataFrame, ErrorCode, ErrorKind, ExecutionBudgetRequest};

use crate::common::{column_from_slice, error_context_value, with_test_runtime};

/// Helper: build a 5-row, 2-column DataFrame for slicing tests.
fn sample_df() -> DataFrame {
    let ids = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let vals = column_from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap()
}

// ---------------------------------------------------------------------------
// head
// ---------------------------------------------------------------------------

fn test_head_basic() {
    let df = sample_df();
    let h = df.head(2).unwrap();

    assert_eq!(h.num_rows(), 2);
    assert_eq!(h.num_columns(), 2);
    assert_eq!(h.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20]);
    assert_eq!(h.series("val").unwrap().to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0]);
}

fn test_head_zero_rows() {
    let df = sample_df();
    let h = df.head(0).unwrap();

    assert_eq!(h.num_rows(), 0);
    assert_eq!(h.num_columns(), 2);
    assert_eq!(h.column_names(), vec!["id", "val"]);
}

fn test_head_clamps_to_num_rows() {
    let df = sample_df();
    let h = df.head(100).unwrap();

    assert_eq!(h.num_rows(), 5);
    assert_eq!(h.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40, 50]);
}

fn test_head_preserves_schema() {
    let df = sample_df();
    let h = df.head(3).unwrap();
    assert_eq!(h.column_names(), vec!["id", "val"]);
    assert_eq!(h.column_index("id"), Some(0));
    assert_eq!(h.column_index("val"), Some(1));
}

// ---------------------------------------------------------------------------
// tail
// ---------------------------------------------------------------------------

fn test_tail_basic() {
    let df = sample_df();
    let t = df.tail(2).unwrap();

    assert_eq!(t.num_rows(), 2);
    assert_eq!(t.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![40, 50]);
    assert_eq!(t.series("val").unwrap().to_vec_strict::<f64>().unwrap(), vec![4.0, 5.0]);
}

fn test_tail_clamps_to_num_rows() {
    let df = sample_df();
    let t = df.tail(100).unwrap();

    assert_eq!(t.num_rows(), 5);
    assert_eq!(t.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30, 40, 50]);
}

fn test_tail_zero_rows() {
    let df = sample_df();
    let t = df.tail(0).unwrap();

    assert_eq!(t.num_rows(), 0);
    assert_eq!(t.column_names(), vec!["id", "val"]);
}

fn test_tail_preserves_schema() {
    let df = sample_df();
    let t = df.tail(3).unwrap();
    assert_eq!(t.column_names(), vec!["id", "val"]);
}

// ---------------------------------------------------------------------------
// slice
// ---------------------------------------------------------------------------

fn test_slice_basic() {
    let df = sample_df();
    let s = df.slice(1, 2).unwrap();

    assert_eq!(s.num_rows(), 2);
    assert_eq!(s.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(s.series("val").unwrap().to_vec_strict::<f64>().unwrap(), vec![2.0, 3.0]);
}

fn test_slice_clamps_length() {
    let df = sample_df();
    // offset=3, length=100 → should clamp to rows 3..5
    let s = df.slice(3, 100).unwrap();

    assert_eq!(s.num_rows(), 2);
    assert_eq!(s.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![40, 50]);
}

fn test_slice_zero_length() {
    let df = sample_df();
    let s = df.slice(2, 0).unwrap();

    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

fn test_slice_preserves_schema() {
    let df = sample_df();
    let s = df.slice(0, 3).unwrap();
    assert_eq!(s.column_names(), vec!["id", "val"]);
    assert_eq!(s.column_index("id"), Some(0));
    assert_eq!(s.column_index("val"), Some(1));
}

fn test_slice_offset_at_end() {
    let df = sample_df();
    // offset == num_rows → empty result
    let s = df.slice(5, 10).unwrap();
    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

fn test_slice_offset_beyond_end() {
    let df = sample_df();
    // offset > num_rows → empty result
    let s = df.slice(999, 10).unwrap();
    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

fn test_slice_with_budget_reports_member_accounting_on_rejection() {
    let df = sample_df();
    let token = with_test_runtime(|runtime| {
        runtime.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("partition_slice")).unwrap()
    });

    let err = df.slice_with_budget(0, 5, &token).expect_err("one-byte budget should not cover the gathered slice copy");

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("partition_slice"));
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"));
    // The member accounting this path previously dropped by taking a raw
    // execution lease instead of the caller's budget token.
    assert!(error_context_value(&err, "budget_peak_bytes").is_some());
    assert!(error_context_value(&err, "budget_current_bytes").is_some());
    assert_eq!(error_context_value(&err, "budget_failed_allocation_count").as_deref(), Some("1"));
}

fn test_testing_hook_locks_dataframe_row_range_start_limit_contract() {
    let err = cudf::testing::dataframe::trigger_dataframe_row_range_start_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::gather_range");
    assert_eq!(err.message(), "row range start exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("start_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), None);
}

fn test_testing_hook_locks_dataframe_row_range_end_limit_contract() {
    let err = cudf::testing::dataframe::trigger_dataframe_row_range_end_limit_error_for_test().unwrap_err();
    let expected_end = (i32::MAX as usize).checked_add(1).unwrap().to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::gather_range");
    assert_eq!(err.message(), "row range end exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("end_exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), Some(expected_end.as_str()));
}

fn test_testing_hook_locks_dataframe_row_range_overflow_contract() {
    let err = cudf::testing::dataframe::trigger_dataframe_row_range_overflow_error_for_test().unwrap_err();
    let expected_end = ((i32::MAX as u128) + (usize::MAX as u128)).to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::gather_range");
    assert_eq!(err.message(), "row range end overflowed usize");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_range"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("end_overflowed_usize"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce row range bounds to fit within usize"));
    assert_eq!(error_context_value(&err, "start_row").as_deref(), Some(i32::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "end_row").as_deref(), Some(expected_end.as_str()));
    assert_eq!(error_context_value(&err, "length").as_deref(), None);
}

#[test]
fn slice_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_head_basic", test_head_basic),
        ("test_head_zero_rows", test_head_zero_rows),
        ("test_head_clamps_to_num_rows", test_head_clamps_to_num_rows),
        ("test_head_preserves_schema", test_head_preserves_schema),
        ("test_tail_basic", test_tail_basic),
        ("test_tail_clamps_to_num_rows", test_tail_clamps_to_num_rows),
        ("test_tail_zero_rows", test_tail_zero_rows),
        ("test_tail_preserves_schema", test_tail_preserves_schema),
        ("test_slice_basic", test_slice_basic),
        ("test_slice_clamps_length", test_slice_clamps_length),
        ("test_slice_zero_length", test_slice_zero_length),
        ("test_slice_preserves_schema", test_slice_preserves_schema),
        ("test_slice_offset_at_end", test_slice_offset_at_end),
        ("test_slice_offset_beyond_end", test_slice_offset_beyond_end),
        (
            "test_slice_with_budget_reports_member_accounting_on_rejection",
            test_slice_with_budget_reports_member_accounting_on_rejection,
        ),
        (
            "test_testing_hook_locks_dataframe_row_range_start_limit_contract",
            test_testing_hook_locks_dataframe_row_range_start_limit_contract,
        ),
        (
            "test_testing_hook_locks_dataframe_row_range_end_limit_contract",
            test_testing_hook_locks_dataframe_row_range_end_limit_contract,
        ),
        (
            "test_testing_hook_locks_dataframe_row_range_overflow_contract",
            test_testing_hook_locks_dataframe_row_range_overflow_contract,
        ),
    ]);
}
