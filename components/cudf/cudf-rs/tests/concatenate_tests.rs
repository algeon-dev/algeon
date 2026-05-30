//! Tests for concatenation operations.

use cudf::{column::Column, concatenate, table::Table};

#[test]
fn test_concatenate_columns() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[4i32, 5]).unwrap();
    let av = a.view().unwrap();
    let bv = b.view().unwrap();

    let result = concatenate::concatenate_columns(&[&av, &bv]).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 5);
    let values = rv.to_vec::<i32>().unwrap();
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_concatenate_tables() {
    let c0a = Column::from_slice(&[1i32, 2]).unwrap();
    let c1a = Column::from_slice(&[10.0f64, 20.0]).unwrap();
    let t1 = Table::from_columns(vec![c0a, c1a]).unwrap();

    let c0b = Column::from_slice(&[3i32, 4]).unwrap();
    let c1b = Column::from_slice(&[30.0f64, 40.0]).unwrap();
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
fn test_concatenate_nullable_columns() {
    let a = Column::from_slice_nullable(&[1i32, 2], &[true, false]).unwrap();
    let b = Column::from_slice_nullable(&[3i32, 4], &[false, true]).unwrap();
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
