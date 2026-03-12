//! Tests for Table::into_columns.

use cudf::column::Column;
use cudf::table::Table;
use cudf::types::TypeId;

#[test]
fn test_into_columns_single_column() {
    let c1 = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 1);

    let view = columns[0].view().unwrap();
    assert_eq!(view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_into_columns_multiple_columns() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let c3 = Column::from_slice(&[7i64, 8, 9]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 3);

    let v0 = columns[0].view().unwrap();
    assert_eq!(v0.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(v0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let v1 = columns[1].view().unwrap();
    assert_eq!(v1.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(v1.to_vec::<f64>().unwrap(), vec![4.0, 5.0, 6.0]);

    let v2 = columns[2].view().unwrap();
    assert_eq!(v2.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(v2.to_vec::<i64>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn test_into_columns_empty_table() {
    let table = Table::from_columns(vec![]).unwrap();
    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 0);
}

#[test]
fn test_into_columns_preserves_column_order() {
    let c1 = Column::from_slice(&[100i32]).unwrap();
    let c2 = Column::from_slice(&[200i64]).unwrap();
    let c3 = Column::from_slice(&[3.14f32]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 3);

    assert_eq!(columns[0].view().unwrap().data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(columns[1].view().unwrap().data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(columns[2].view().unwrap().data_type().unwrap().id(), TypeId::Float32);
}

#[test]
fn test_into_columns_nullable_column() {
    let c1 = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 1);

    let view = columns[0].view().unwrap();
    assert_eq!(view.to_vec::<i32>().unwrap()[0], 1);
    assert_eq!(view.to_vec::<i32>().unwrap()[2], 3);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 1);

    let validity = view.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);
}

#[test]
fn test_into_columns_roundtrip() {
    // Create table, decompose into columns, re-create table, verify data.
    let c1 = Column::from_slice(&[10i32, 20]).unwrap();
    let c2 = Column::from_slice(&[1.5f64, 2.5]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 2);

    // Re-assemble into a new table.
    let table2 = Table::from_columns(columns).unwrap();
    let view = table2.view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 2);

    let col0 = view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![10, 20]);

    let col1 = view.column(1).unwrap();
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![1.5, 2.5]);
}

#[test]
fn test_into_columns_with_string_column() {
    let c1 = Column::from_strings(&["hello", "world"]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 1);

    let view = columns[0].view().unwrap();
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);
    assert_eq!(view.to_strings().unwrap(), vec!["hello", "world"]);
}

#[test]
fn test_into_columns_with_empty_rows() {
    // Table with columns that have zero rows.
    let c1 = Column::empty(TypeId::Int32).unwrap();
    let c2 = Column::empty(TypeId::Float64).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 2);

    let v0 = columns[0].view().unwrap();
    assert_eq!(v0.len().unwrap(), 0);
    assert_eq!(v0.data_type().unwrap().id(), TypeId::Int32);

    let v1 = columns[1].view().unwrap();
    assert_eq!(v1.len().unwrap(), 0);
    assert_eq!(v1.data_type().unwrap().id(), TypeId::Float64);
}
