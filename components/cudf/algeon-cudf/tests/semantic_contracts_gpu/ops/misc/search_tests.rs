//! Tests for CU-2: Series and ColumnView membership (isin).
use cudf::{ErrorCode, ErrorKind, Exec, testing::prelude::*};

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, dependency_wait_call_count_for_stream,
    error_context_value, explicit_i32_column, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    lock_event_observation, reset_dependency_observation, reset_exec_observation, reset_sync_observation, series,
};

// ---------------------------------------------------------------------------
// Series::isin
// ---------------------------------------------------------------------------

fn test_series_isin_with_duplicate_values() {
    let s = series::<i32>("x", &[1, 3, 4]).unwrap();
    let values = series::<i32>("vals", &[1, 1, 2, 2, 3]).unwrap();

    let mask = s.on(Exec::RETAINED).isin(&values).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

fn test_series_isin_float64() {
    let s = series::<f64>("x", &[2.0, 4.0]).unwrap();
    let values = series::<f64>("vals", &[1.0, 2.0, 3.0]).unwrap();

    let mask = s.on(Exec::RETAINED).isin(&values).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false]);
}

// ---------------------------------------------------------------------------
// ColumnView::isin
// ---------------------------------------------------------------------------

fn test_isin_i32_some_matches() {
    let col = column_from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let values = column_from_slice(&[2i32, 4]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false, true, false]);
}

fn test_isin_succeeds_with_mixed_domains() {
    let _event_observation = lock_event_observation();
    let col = explicit_i32_column(&[1, 2, 3, 4, 5], 912);
    let values = explicit_i32_column(&[2, 4], 913);
    let expected_exec_stream = col.testing_execution_domain_stream_value();
    let expected_exec_mr = col.testing_execution_domain_mr_value();
    reset_dependency_observation();
    reset_exec_observation();
    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    assert_eq!(mask.testing_execution_domain_id(), col.testing_execution_domain_id());
    assert_eq!(mask.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false, true, false]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

fn test_isin_string_column() {
    let col = column_from_strs(&["apple", "banana", "cherry"]).unwrap();
    let values = column_from_strs(&["banana", "date"]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

fn test_isin_no_matches() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let values = column_from_slice(&[10i32, 20]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

fn test_isin_all_matches() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let values = column_from_slice(&[3i32, 1, 2]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![true, true, true]);
}

fn test_isin_empty_values() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let values = column_from_slice(&[] as &[i32]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    // Empty values set → nothing can match → all false.
    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

fn test_isin_column_with_nulls() {
    let col = column_from_slice_nullable(&[1i32, 0, 3, 0], &[true, false, true, false]).unwrap();
    let values = column_from_slice(&[1i32, 3]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    // Non-null rows: 1→true, 3→true. Null rows produce null in output.
    assert_eq!(mv.len().unwrap(), 4);
    assert_eq!(mv.null_count().unwrap(), 2);

    let validity = mv.to_host_validity().unwrap().expect("mask should be nullable");
    assert_eq!(validity, vec![true, false, true, false]);

    // Valid positions should be true for 1 and 3.
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![true, false, true, false]);
}

fn test_isin_empty_column() {
    let col = column_from_slice(&[] as &[i32]).unwrap();
    let values = column_from_slice(&[1i32, 2]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.len().unwrap(), 0);
}

fn test_isin_type_mismatch_error_has_operation_context() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let values = column_from_slice(&[1.0f64, 2.0]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let err = match cv.isin(&vv) {
        Ok(_) => panic!("expected type mismatch error"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "column_view::isin");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("float64"));
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("search_tests::test_series_isin_with_duplicate_values", test_series_isin_with_duplicate_values),
    ("search_tests::test_series_isin_float64", test_series_isin_float64),
    ("search_tests::test_isin_i32_some_matches", test_isin_i32_some_matches),
    ("search_tests::test_isin_succeeds_with_mixed_domains", test_isin_succeeds_with_mixed_domains),
    ("search_tests::test_isin_string_column", test_isin_string_column),
    ("search_tests::test_isin_no_matches", test_isin_no_matches),
    ("search_tests::test_isin_all_matches", test_isin_all_matches),
    ("search_tests::test_isin_empty_values", test_isin_empty_values),
    ("search_tests::test_isin_column_with_nulls", test_isin_column_with_nulls),
    ("search_tests::test_isin_empty_column", test_isin_empty_column),
    (
        "search_tests::test_isin_type_mismatch_error_has_operation_context",
        test_isin_type_mismatch_error_has_operation_context,
    ),
];
