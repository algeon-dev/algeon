//! Tests for CU-2: contains (isin) and ColumnView::isin.
use cudf::{ErrorKind, ops::search, testing::prelude::*};

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, explicit_i32_column, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
};

#[test]
fn test_contains_basic_i32() {
    let haystack = column_from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let needles = column_from_slice(&[20i32, 40, 60, 80]).unwrap();

    let hv = haystack.testing_view().unwrap();
    let nv = needles.testing_view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false, false]);
}

#[test]
fn test_contains_all_match() {
    let haystack = column_from_slice(&[1i32, 2, 3]).unwrap();
    let needles = column_from_slice(&[3i32, 1, 2]).unwrap();

    let hv = haystack.testing_view().unwrap();
    let nv = needles.testing_view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, true]);
}

#[test]
fn test_contains_no_match() {
    let haystack = column_from_slice(&[1i32, 2, 3]).unwrap();
    let needles = column_from_slice(&[4i32, 5, 6]).unwrap();

    let hv = haystack.testing_view().unwrap();
    let nv = needles.testing_view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn test_contains_empty_needles() {
    let haystack = column_from_slice(&[1i32, 2, 3]).unwrap();
    let needles = column_from_slice(&[] as &[i32]).unwrap();

    let hv = haystack.testing_view().unwrap();
    let nv = needles.testing_view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.len().unwrap(), 0);
}

#[test]
fn test_contains_empty_haystack() {
    let haystack = column_from_slice(&[] as &[i32]).unwrap();
    let needles = column_from_slice(&[1i32, 2]).unwrap();

    let hv = haystack.testing_view().unwrap();
    let nv = needles.testing_view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false]);
}

#[test]
fn test_contains_with_duplicates_in_haystack() {
    let haystack = column_from_slice(&[1i32, 1, 2, 2, 3]).unwrap();
    let needles = column_from_slice(&[1i32, 3, 4]).unwrap();

    let hv = haystack.testing_view().unwrap();
    let nv = needles.testing_view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_contains_float64() {
    let haystack = column_from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
    let needles = column_from_slice(&[2.0f64, 4.0]).unwrap();

    let hv = haystack.testing_view().unwrap();
    let nv = needles.testing_view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.testing_view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false]);
}

#[test]
fn test_contains_succeeds_with_mixed_domains_and_uses_haystack_provenance() {
    let haystack = explicit_i32_column(&[10, 20, 30, 40], 910);
    let needles = explicit_i32_column(&[20, 50, 10], 911);
    let expected_stream = haystack.testing_execution_domain_stream_value();
    let expected_mr = haystack.testing_execution_domain_mr_value();
    let expected_sync_stream = needles.testing_execution_domain_stream_value();

    reset_sync_observation();
    reset_exec_observation();
    let result = search::contains(&haystack.testing_view().unwrap(), &needles.testing_view().unwrap()).unwrap();

    assert_eq!(result.testing_execution_domain_id(), haystack.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    assert_eq!(result.testing_view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

// ---------------------------------------------------------------------------
// ColumnView::isin
// ---------------------------------------------------------------------------

#[test]
fn test_isin_i32_some_matches() {
    let col = column_from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let values = column_from_slice(&[2i32, 4]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false, true, false]);
}

#[test]
fn test_isin_succeeds_with_mixed_domains() {
    let col = explicit_i32_column(&[1, 2, 3, 4, 5], 912);
    let values = explicit_i32_column(&[2, 4], 913);
    let expected_exec_stream = col.testing_execution_domain_stream_value();
    let expected_exec_mr = col.testing_execution_domain_mr_value();
    let expected_sync_stream = values.testing_execution_domain_stream_value();

    reset_exec_observation();
    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    assert_eq!(mask.testing_execution_domain_id(), col.testing_execution_domain_id());
    assert_eq!(mask.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false, true, false]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_isin_string_column() {
    let col = column_from_strs(&["apple", "banana", "cherry"]).unwrap();
    let values = column_from_strs(&["banana", "date"]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_isin_no_matches() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let values = column_from_slice(&[10i32, 20]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn test_isin_all_matches() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let values = column_from_slice(&[3i32, 1, 2]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![true, true, true]);
}

#[test]
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

#[test]
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

#[test]
fn test_isin_empty_column() {
    let col = column_from_slice(&[] as &[i32]).unwrap();
    let values = column_from_slice(&[1i32, 2]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.testing_view().unwrap();
    assert_eq!(mv.len().unwrap(), 0);
}

#[test]
fn test_isin_type_mismatch_error_has_operation_context() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let values = column_from_slice(&[1.0f64, 2.0]).unwrap();

    let cv = col.testing_view().unwrap();
    let vv = values.testing_view().unwrap();

    let err = match cv.isin(&vv) {
        Ok(_) => panic!("expected type mismatch error"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "column_view::isin");

    let ctx = err.context();
    assert!(ctx.iter().any(|(k, _)| *k == "input_type_id"));
    assert!(ctx.iter().any(|(k, _)| *k == "values_type_id"));
}
