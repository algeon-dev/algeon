//! Tests for Table and TableView.
use cudf::{
    ErrorCode, Runtime, TypeId,
    core::Table,
    testing::{TableView, prelude::*},
};

use crate::common::{
    arm_next_table_view_negative_num_columns_fault, arm_next_table_view_negative_num_rows_fault, column_from_slice,
    column_from_slice_nullable, error_context_value,
};

#[test]
fn test_create_table_from_columns() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let view = table.testing_view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 3);
}

#[test]
fn test_table_column_access() {
    let c1 = column_from_slice(&[10i32, 20, 30]).unwrap();
    let c2 = column_from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.testing_view().unwrap();

    let col0 = view.column(0).unwrap();
    assert_eq!(col0.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![10, 20, 30]);

    let col1 = view.column(1).unwrap();
    assert_eq!(col1.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_table_deep_copy() {
    let c1 = column_from_slice(&[1i64, 2]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let copy = table.deep_copy().unwrap();

    let view = copy.testing_view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 1);
    assert_eq!(view.num_rows().unwrap(), 2);

    let col = view.column(0).unwrap();
    assert_eq!(col.to_vec::<i64>().unwrap(), vec![1, 2]);
}

#[test]
fn test_table_from_columns_into_columns_preserves_input_order() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10i32, 20, 30]).unwrap();
    let c3 = column_from_slice(&[100i32, 200, 300]).unwrap();

    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();
    let columns = table.into_columns().unwrap();

    assert_eq!(columns.len(), 3);
    assert_eq!(columns[0].testing_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(columns[1].testing_view().unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(columns[2].testing_view().unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
}

#[test]
fn test_empty_table() {
    let rt = Runtime::new().unwrap();
    let table = rt.empty_table().unwrap();
    let view = table.testing_view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 0);
    assert_eq!(view.num_rows().unwrap(), 0);
    assert_eq!(table.testing_execution_domain_kind(), "explicit");
    assert_eq!(table.testing_execution_domain_id(), rt.testing_execution_domain_id());
}

#[test]
fn test_table_from_columns_empty_requires_runtime() {
    let err = Table::from_columns(vec![]).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::ExecutionContext);
    assert_eq!(err.code(), ErrorCode::EmptyConstructionRequiresRuntime);
    assert_eq!(err.operation(), "table::from_columns");
    assert_eq!(error_context_value(&err, "remedy"), Some("use Runtime::empty_table()"));
    assert!(err.message().contains("requires an explicit runtime"), "unexpected error: {err}");
}

#[test]
fn test_table_with_nullable_columns() {
    let c1 = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let c2 = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.testing_view().unwrap();

    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 3);

    let col0 = view.column(0).unwrap();
    let null_count: usize = col0.null_count().unwrap();
    assert!(col0.nullable().unwrap());
    assert_eq!(null_count, 1);
}

#[test]
fn test_table_from_columns_mismatched_rows_rejected() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10i32, 20]).unwrap();
    let err = Table::from_columns(vec![c1, c2]).expect_err("expected mismatched row counts to fail");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "table::from_columns");
    assert_eq!(error_context_value(&err, "column_index"), Some("1"));
    assert_eq!(error_context_value(&err, "actual_rows"), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows"), Some("3"));
}

#[test]
fn test_testing_hook_locks_table_column_count_limit_contract() {
    let err = cudf::testing::trigger_table_column_count_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "table::from_columns");
    assert_eq!(err.message(), "number of columns exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource"), Some("table_column_count"));
    assert_eq!(error_context_value(&err, "reason"), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "table_column_count"), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "remedy"), Some("reduce table_column_count to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "num_columns"), None);
}

#[test]
fn test_table_view_column_rejects_out_of_bounds_index() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.testing_view().unwrap();

    let err = view.column(1).expect_err("expected out-of-bounds column access to fail");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(err.operation(), "table_view::column");
    assert_eq!(error_context_value(&err, "column_index"), Some("1"));
    assert_eq!(error_context_value(&err, "table_column_count"), Some("1"));
    assert_eq!(error_context_value(&err, "num_columns"), None);
}

#[test]
fn test_table_view_column_rejects_usize_indices_outside_i32_range() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.testing_view().unwrap();

    let err = view.column(usize::MAX).expect_err("expected oversized column index to fail");
    let expected_index = usize::MAX.to_string();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(err.operation(), "table_view::column");
    assert_eq!(error_context_value(&err, "column_index"), Some(expected_index.as_str()));
    assert_eq!(error_context_value(&err, "table_column_count"), Some("1"));
    assert_eq!(error_context_value(&err, "num_columns"), None);
}

#[test]
fn test_table_view_from_column_views() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let v1 = c1.testing_view().unwrap();
    let v2 = c2.testing_view().unwrap();

    let table_view = TableView::from_column_views(&[&v1, &v2]).unwrap();
    assert_eq!(table_view.num_columns().unwrap(), 2);
    assert_eq!(table_view.num_rows().unwrap(), 3);

    // Verify column data is accessible through the table view
    let col0 = table_view.column(0).unwrap();
    assert_eq!(col0.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let col1 = table_view.column(1).unwrap();
    assert_eq!(col1.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_table_view_from_column_views_single_column() {
    let c1 = column_from_slice(&[10i64, 20, 30, 40]).unwrap();
    let v1 = c1.testing_view().unwrap();

    let table_view = TableView::from_column_views(&[&v1]).unwrap();
    assert_eq!(table_view.num_columns().unwrap(), 1);
    assert_eq!(table_view.num_rows().unwrap(), 4);

    let col0 = table_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i64>().unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn test_table_view_from_column_views_retains_backing_after_source_views_drop() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4i32, 5, 6]).unwrap();

    let table_view = {
        let v1 = c1.testing_view().unwrap();
        let v2 = c2.testing_view().unwrap();
        TableView::from_column_views(&[&v1, &v2]).unwrap()
    };

    assert_eq!(table_view.num_columns().unwrap(), 2);
    assert_eq!(table_view.num_rows().unwrap(), 3);
    assert_eq!(table_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(table_view.column(1).unwrap().to_vec::<i32>().unwrap(), vec![4, 5, 6]);
}

#[test]
fn test_table_view_from_column_views_empty() {
    let err = TableView::from_column_views(&[]).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "table_view::from_column_views");
    assert_eq!(error_context_value(&err, "reason"), Some("empty_borrowed_table_view_requires_authoritative_source"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some("borrow at least one column view or call view() on an authoritative Table created by a Runtime")
    );
    assert!(err.message().contains("requires an authoritative source"), "unexpected error: {err}");
}

#[test]
fn test_table_view_from_column_views_rejects_mismatched_rows_with_stable_contract() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4i32, 5]).unwrap();
    let v1 = c1.testing_view().unwrap();
    let v2 = c2.testing_view().unwrap();

    let err = TableView::from_column_views(&[&v1, &v2]).expect_err("expected mismatched row counts to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "table_view::from_column_views");
    assert_eq!(error_context_value(&err, "column_index"), Some("1"));
    assert_eq!(error_context_value(&err, "actual_rows"), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows"), Some("3"));
}

#[test]
fn test_testing_hook_locks_table_view_column_count_limit_contract() {
    let err = cudf::testing::trigger_table_view_column_count_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "table_view::from_column_views");
    assert_eq!(err.message(), "number of column views exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource"), Some("table_column_count"));
    assert_eq!(error_context_value(&err, "reason"), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "table_column_count"), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "remedy"), Some("reduce table_column_count to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "num_columns"), None);
}

#[test]
fn test_table_view_select_columns() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[10i32, 20, 30]).unwrap();
    let c3 = column_from_slice(&[100i32, 200, 300]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();
    let view = table.testing_view().unwrap();

    let projected = view.select_columns(&[2, 0]).unwrap();
    assert_eq!(projected.num_columns().unwrap(), 2);
    assert_eq!(projected.num_rows().unwrap(), 3);
    assert_eq!(projected.column(0).unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
    assert_eq!(projected.column(1).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_table_view_select_columns_rejects_out_of_bounds_index() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.testing_view().unwrap();

    let err = view.select_columns(&[1]).expect_err("expected out-of-bounds error");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(err.operation(), "table_view::select_columns");
    assert_eq!(error_context_value(&err, "column_index"), Some("1"));
    assert_eq!(error_context_value(&err, "table_column_count"), Some("1"));
    assert_eq!(error_context_value(&err, "num_columns"), None);
}

#[test]
fn test_table_view_num_columns_rejects_negative_foreign_count_with_stable_contract() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.testing_view().unwrap();

    arm_next_table_view_negative_num_columns_fault();
    let err = view.num_columns().expect_err("expected malformed foreign column count to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "table_view::num_columns");
    assert_eq!(error_context_value(&err, "resource"), Some("table_column_count"));
    assert_eq!(error_context_value(&err, "reason"), Some("invalid_foreign_table_metadata"));
    assert_eq!(error_context_value(&err, "table_column_count"), Some("-1"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some("ensure the foreign table view reports a non-negative column count"),
    );
    assert_eq!(error_context_value(&err, "num_columns"), None);
}

#[test]
fn test_table_view_num_rows_rejects_negative_foreign_count_with_stable_contract() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.testing_view().unwrap();

    arm_next_table_view_negative_num_rows_fault();
    let err = view.num_rows().expect_err("expected malformed foreign row count to fail");

    assert_eq!(err.kind(), cudf::ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.operation(), "table_view::num_rows");
    assert_eq!(error_context_value(&err, "resource"), Some("table_row_count"));
    assert_eq!(error_context_value(&err, "reason"), Some("invalid_foreign_table_metadata"));
    assert_eq!(error_context_value(&err, "table_row_count"), Some("-1"));
    assert_eq!(
        error_context_value(&err, "remedy"),
        Some("ensure the foreign table view reports a non-negative row count"),
    );
    assert_eq!(error_context_value(&err, "num_rows"), None);
}
