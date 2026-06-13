//! Tests for filter operations.
use cudf::{ErrorCode, ErrorKind, core::Table, ops::filter, testing::prelude::*};

use crate::common::{
    column_from_slice, error_context_value, explicit_bool_column, explicit_i32_column, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
};

#[test]
fn test_boolean_mask_basic() {
    let col = column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.testing_view().unwrap();

    // Keep rows 0 and 2.
    let mask = column_from_slice(&[true, false, true, false]).unwrap();
    let mask_view = mask.testing_view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.testing_view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 2);
    let result = fv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![10, 30]);
}

#[test]
fn test_boolean_mask_all_false() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.testing_view().unwrap();

    let mask = column_from_slice(&[false, false, false]).unwrap();
    let mask_view = mask.testing_view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.testing_view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 0);
}

#[test]
fn test_boolean_mask_all_true() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.testing_view().unwrap();

    let mask = column_from_slice(&[true, true, true]).unwrap();
    let mask_view = mask.testing_view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.testing_view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 3);
    let result = fv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_boolean_mask_multi_column() {
    let c0 = column_from_slice(&[1i32, 2, 3, 4]).unwrap();
    let c1 = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let view = table.testing_view().unwrap();

    let mask = column_from_slice(&[false, true, false, true]).unwrap();
    let mask_view = mask.testing_view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.testing_view().unwrap();
    assert_eq!(fv.num_columns().unwrap(), 2);
    assert_eq!(fv.num_rows().unwrap(), 2);

    let col0 = fv.column(0).unwrap().to_vec::<i32>().unwrap();
    let col1 = fv.column(1).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(col0, vec![2, 4]);
    assert_eq!(col1, vec![20.0, 40.0]);
}

#[test]
fn test_boolean_mask_succeeds_with_mixed_domains_and_uses_input_provenance() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 920)]).unwrap();
    let mask = explicit_bool_column(&[true, false, true], 921);
    let expected_stream = table.testing_execution_domain_stream_value();
    let expected_mr = table.testing_execution_domain_mr_value();

    reset_exec_observation();
    let filtered = filter::apply_boolean_mask(&table.testing_view().unwrap(), &mask.testing_view().unwrap()).unwrap();

    assert_eq!(filtered.testing_execution_domain_id(), table.testing_execution_domain_id());
    assert_eq!(filtered.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    let host = filtered.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn test_boolean_mask_rejects_non_bool_mask_with_canonical_type_contract() {
    let table = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let mask = column_from_slice(&[1i32, 0, 1]).unwrap();

    let err = filter::apply_boolean_mask(&table.testing_view().unwrap(), &mask.testing_view().unwrap()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "filter::apply_boolean_mask");
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("bool8"));
}

#[test]
fn test_boolean_mask_rejects_wrong_length_mask_with_canonical_length_contract() {
    let table = Table::from_columns(vec![column_from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let mask = column_from_slice(&[true, false]).unwrap();

    let err = filter::apply_boolean_mask(&table.testing_view().unwrap(), &mask.testing_view().unwrap()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "filter::apply_boolean_mask");
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
}
