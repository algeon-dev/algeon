//! Tests for filter operations.

use cudf::column::Column;
use cudf::filter;
use cudf::table::Table;

#[test]
fn test_boolean_mask_basic() {
    let col = Column::from_slice(&[10i32, 20, 30, 40]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    // Keep rows 0 and 2.
    let mask = Column::from_slice(&[true, false, true, false]).unwrap();
    let mask_view = mask.view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 2);
    let result = fv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![10, 30]);
}

#[test]
fn test_boolean_mask_all_false() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let mask = Column::from_slice(&[false, false, false]).unwrap();
    let mask_view = mask.view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 0);
}

#[test]
fn test_boolean_mask_all_true() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let mask = Column::from_slice(&[true, true, true]).unwrap();
    let mask_view = mask.view().unwrap();

    let filtered = filter::apply_boolean_mask(&view, &mask_view).unwrap();
    let fv = filtered.view().unwrap();
    assert_eq!(fv.num_rows().unwrap(), 3);
    let result = fv.column(0).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_boolean_mask_multi_column() {
    let c0 = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let c1 = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let view = table.view().unwrap();

    let mask = Column::from_slice(&[false, true, false, true]).unwrap();
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
