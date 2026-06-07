//! Tests for Table and TableView.

use cudf::{Column, Table, TableView, TypeId};

#[test]
fn test_create_table_from_columns() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let view = table.view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 3);
}

#[test]
fn test_table_column_access() {
    let c1 = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let c2 = Column::from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();

    let col0 = view.column(0).unwrap();
    assert_eq!(col0.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![10, 20, 30]);

    let col1 = view.column(1).unwrap();
    assert_eq!(col1.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_table_copy() {
    let c1 = Column::from_slice(&[1i64, 2]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let copy = table.copy().unwrap();

    let view = copy.view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 1);
    assert_eq!(view.num_rows().unwrap(), 2);

    let col = view.column(0).unwrap();
    assert_eq!(col.to_vec::<i64>().unwrap(), vec![1, 2]);
}

#[test]
fn test_table_from_columns_into_columns_preserves_input_order() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let c3 = Column::from_slice(&[100i32, 200, 300]).unwrap();

    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();
    let columns = table.into_columns().unwrap();

    assert_eq!(columns.len(), 3);
    assert_eq!(columns[0].view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(columns[1].view().unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
    assert_eq!(columns[2].view().unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
}

#[test]
fn test_empty_table() {
    let table = Table::from_columns(vec![]).unwrap();
    let view = table.view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 0);
}

#[test]
fn test_table_with_nullable_columns() {
    let c1 = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();
    let view = table.view().unwrap();

    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 3);

    let col0 = view.column(0).unwrap();
    let null_count: usize = col0.null_count().unwrap();
    assert!(col0.nullable().unwrap());
    assert_eq!(null_count, 1);
}

#[test]
fn test_table_from_columns_mismatched_rows_rejected() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10i32, 20]).unwrap();
    let res = Table::from_columns(vec![c1, c2]);
    assert!(res.is_err());
    let err = res.err().unwrap();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_table_view_negative_column_index_rejected() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.view().unwrap();

    let res = view.column(-1);
    assert!(res.is_err());
    let err = res.err().unwrap();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_table_view_from_column_views() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let v1 = c1.view().unwrap();
    let v2 = c2.view().unwrap();

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
    let c1 = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let v1 = c1.view().unwrap();

    let table_view = TableView::from_column_views(&[&v1]).unwrap();
    assert_eq!(table_view.num_columns().unwrap(), 1);
    assert_eq!(table_view.num_rows().unwrap(), 4);

    let col0 = table_view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i64>().unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn test_table_view_from_column_views_empty() {
    let table_view = TableView::from_column_views(&[]).unwrap();
    assert_eq!(table_view.num_columns().unwrap(), 0);
}

#[test]
fn test_table_view_select_columns() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let c3 = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();
    let view = table.view().unwrap();

    let projected = view.select_columns(&[2, 0]).unwrap();
    assert_eq!(projected.num_columns().unwrap(), 2);
    assert_eq!(projected.num_rows().unwrap(), 3);
    assert_eq!(projected.column(0).unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);
    assert_eq!(projected.column(1).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_table_view_select_columns_rejects_out_of_bounds_index() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();
    let view = table.view().unwrap();

    let err = view.select_columns(&[1]).err().expect("expected out-of-bounds error");
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "table_view::select_columns");
}
