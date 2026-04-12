//! Tests for filter operations.

use cudf::{core::Table, ops::filter};

use crate::common::{
    column_from_slice, explicit_bool_column, explicit_i32_column, last_exec_mr_value, last_exec_stream_value,
    last_sync_stream_value, reset_exec_observation, reset_sync_observation,
};

#[test]
fn test_boolean_mask_basic() {
    let col = column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    // Keep rows 0 and 2.
    let mask = column_from_slice(&[true, false, true, false]).unwrap();
    let mask_view = mask.view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 2);
    let result = fv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![10, 30]);
}

#[test]
fn test_boolean_mask_all_false() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let mask = column_from_slice(&[false, false, false]).unwrap();
    let mask_view = mask.view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 0);
}

#[test]
fn test_boolean_mask_all_true() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let mask = column_from_slice(&[true, true, true]).unwrap();
    let mask_view = mask.view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 3);
    let result = fv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_boolean_mask_multi_column() {
    let c0 = column_from_slice(&[1i32, 2, 3, 4]).unwrap();
    let c1 = column_from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let view = table.view().unwrap();

    let mask = column_from_slice(&[false, true, false, true]).unwrap();
    let mask_view = mask.view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.view().unwrap();
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
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let filtered = filter::apply_boolean_mask(&table.view().unwrap(), &mask.view().unwrap()).unwrap();

    assert_eq!(filtered.__debug_execution_domain_id(), table.__debug_execution_domain_id());
    assert_eq!(filtered.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    let host = filtered.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(host, vec![10, 30]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}
