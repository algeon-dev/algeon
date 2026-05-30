//! Tests for sorting operations.

use cudf::{
    column::Column,
    sorting::{self, NullOrder, Order},
    table::Table,
};

#[test]
fn test_sort_ascending() {
    let col = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let sorted = sorting::sort(&view, None, None).unwrap();
    let sv = sorted.view().unwrap();
    let result = sv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_sort_descending() {
    let col = Column::from_slice(&[1i32, 3, 2]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let sorted = sorting::sort(&view, Some(&[Order::Descending]), None).unwrap();
    let sv = sorted.view().unwrap();
    let result = sv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![3, 2, 1]);
}

#[test]
fn test_sort_multi_column() {
    // Sort by col0 ascending, then col1 descending.
    let c0 = Column::from_slice(&[1i32, 2, 1, 2]).unwrap();
    let c1 = Column::from_slice(&[10i32, 20, 30, 40]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let view = table.view().unwrap();

    let sorted = sorting::sort(&view, Some(&[Order::Ascending, Order::Descending]), None).unwrap();
    let sv = sorted.view().unwrap();

    let col0 = sv.column(0).unwrap().to_vec::<i32>().unwrap();
    let col1 = sv.column(1).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(col0, vec![1, 1, 2, 2]);
    assert_eq!(col1, vec![30, 10, 40, 20]);
}

#[test]
fn test_sort_with_nulls() {
    let col = Column::from_slice_nullable(&[3i32, 1, 2], &[true, false, true]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    // Nulls before: null should appear first.
    let sorted = sorting::sort(&view, Some(&[Order::Ascending]), Some(&[NullOrder::Before])).unwrap();
    let sv = sorted.view().unwrap();
    let validity = sv.column(0).unwrap().validity().unwrap().unwrap();
    // The first element should be null.
    assert!(!validity[0]);
}

#[test]
fn test_sort_by_key() {
    let keys = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let vals = Column::from_slice(&[30.0f64, 10.0, 20.0]).unwrap();

    let keys_table = Table::from_columns(vec![keys]).unwrap();
    let vals_table = Table::from_columns(vec![vals]).unwrap();
    let kv = keys_table.view().unwrap();
    let vv = vals_table.view().unwrap();

    let sorted = sorting::sort_by_key(&vv, &kv, None, None).unwrap();
    let sv = sorted.view().unwrap();
    let result = sv.column(0).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(result, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_sorted_order() {
    let col = Column::from_slice(&[30i32, 10, 20]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let indices = sorting::sorted_order(&view, None, None).unwrap();
    let iv = indices.view().unwrap();
    let result = iv.to_vec::<i32>().unwrap();
    assert_eq!(result, vec![1, 2, 0]);
}

#[test]
fn test_sort_rejects_mismatched_column_order_len() {
    let c0 = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let c1 = Column::from_slice(&[30i32, 10, 20]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let view = table.view().unwrap();

    let err = sorting::sort(&view, Some(&[Order::Ascending]), None).err().expect("expected error");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_is_sorted_rejects_mismatched_null_precedence_len() {
    let c0 = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let c1 = Column::from_slice(&[30i32, 10, 20]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let view = table.view().unwrap();

    let err = sorting::is_sorted(&view, Some(&[Order::Ascending, Order::Ascending]), Some(&[NullOrder::After]))
        .err()
        .expect("expected error");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_sort_error_contains_operation_context() {
    let col = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let err = sorting::sort(&view, Some(&[Order::Ascending, Order::Descending]), None).err().expect("expected error");

    assert_eq!(err.operation(), "sorting::sort");
    let ctx = err.context();
    assert!(ctx.iter().any(|(k, _)| *k == "num_keys"));
    assert!(ctx.iter().any(|(k, _)| *k == "column_order_len"));
}

#[test]
fn test_is_sorted() {
    let sorted_col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let sorted_table = Table::from_columns(vec![sorted_col]).unwrap();
    let sv = sorted_table.view().unwrap();
    assert!(sorting::is_sorted(&sv, None, None).unwrap());

    let unsorted_col = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let unsorted_table = Table::from_columns(vec![unsorted_col]).unwrap();
    let uv = unsorted_table.view().unwrap();
    assert!(!sorting::is_sorted(&uv, None, None).unwrap());
}
