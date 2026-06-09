//! Tests for concatenation operations.

use cudf::{ErrorKind, core::Table, ops::concatenate};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
    sample_runtime,
};

#[test]
fn test_concatenate_columns() {
    let rt = sample_runtime();
    let a = rt.column_from_slice(&[1i32, 2, 3]).unwrap();
    let b = rt.column_from_slice(&[4i32, 5]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = concatenate::concatenate_columns(&[&av, &bv]).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 5);
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_concatenate_columns_rejects_empty_input() {
    let err = concatenate::concatenate_columns(&[]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "concatenate::concatenate_columns");
    assert!(err.message().contains("at least one input column"), "unexpected error: {err}");
}

#[test]
fn test_concatenate_tables() {
    let rt = sample_runtime();
    let c0a = rt.column_from_slice(&[1i32, 2]).unwrap();
    let c1a = rt.column_from_slice(&[10.0f64, 20.0]).unwrap();
    let t1 = Table::from_columns(vec![c0a, c1a]).unwrap();

    let c0b = rt.column_from_slice(&[3i32, 4]).unwrap();
    let c1b = rt.column_from_slice(&[30.0f64, 40.0]).unwrap();
    let t2 = Table::from_columns(vec![c0b, c1b]).unwrap();

    let v1 = t1.view().unwrap();
    let v2 = t2.view().unwrap();

    let result = concatenate::concatenate_tables(&[&v1, &v2]).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_columns().unwrap(), 2);
    assert_eq!(rv.num_rows().unwrap(), 4);

    let col0 = rv.column(0).unwrap().to_vec::<i32>().unwrap();
    let col1 = rv.column(1).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(col0, vec![1, 2, 3, 4]);
    assert_eq!(col1, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_concatenate_tables_rejects_empty_input() {
    let err = concatenate::concatenate_tables(&[]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "concatenate::concatenate_tables");
    assert!(err.message().contains("at least one input table"), "unexpected error: {err}");
}

#[test]
fn test_concatenate_nullable_columns() {
    let rt = sample_runtime();
    let a = rt.column_from_slice_nullable(&[1i32, 2], &[true, false]).unwrap();
    let b = rt.column_from_slice_nullable(&[3i32, 4], &[false, true]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = concatenate::concatenate_columns(&[&av, &bv]).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 4);

    let values = rv.to_vec::<i32>().unwrap();
    let validity = rv.validity().unwrap().unwrap();
    // Values at valid positions should match.
    assert_eq!(values[0], 1);
    assert_eq!(values[3], 4);
    assert_eq!(validity, vec![true, false, false, true]);
}

#[test]
fn test_concatenate_columns_succeeds_with_mixed_domains_and_uses_first_provenance() {
    let first_rt = sample_runtime();
    let second_rt = sample_runtime();
    let first = first_rt.column_from_slice(&[1i32, 2, 3]).unwrap();
    let second = second_rt.column_from_slice(&[4i32, 5]).unwrap();
    let expected_stream = first.__debug_execution_domain_stream_value();
    let expected_mr = first.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = concatenate::concatenate_columns(&[&first.view().unwrap(), &second.view().unwrap()]).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), first.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn test_concatenate_tables_succeeds_with_mixed_domains_and_uses_first_provenance() {
    let first_rt = sample_runtime();
    let second_rt = sample_runtime();
    let first = Table::from_columns(vec![
        first_rt.column_from_slice(&[1i32, 2]).unwrap(),
        first_rt.column_from_slice(&[10i32, 20]).unwrap(),
    ])
    .unwrap();
    let second = Table::from_columns(vec![
        second_rt.column_from_slice(&[3i32, 4]).unwrap(),
        second_rt.column_from_slice(&[30i32, 40]).unwrap(),
    ])
    .unwrap();
    let expected_stream = first.__debug_execution_domain_stream_value();
    let expected_mr = first.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = concatenate::concatenate_tables(&[&first.view().unwrap(), &second.view().unwrap()]).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), first.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    let view = result.view().unwrap();
    assert_eq!(view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(view.column(1).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}
