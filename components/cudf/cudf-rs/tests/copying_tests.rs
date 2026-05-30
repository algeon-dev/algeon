//! Tests for CU-7: gather (take).

use cudf::{column::Column, copying, table::Table};

#[test]
fn test_gather_basic() {
    let col = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    // Gather rows 4, 2, 0.
    let map = Column::from_slice(&[4i32, 2, 0]).unwrap();
    let map_view = map.view().unwrap();

    let result = copying::gather(&tv, &map_view, copying::OutOfBoundsPolicy::DontCheck).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![50, 30, 10]);
}

#[test]
fn test_gather_multi_column() {
    let c0 = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let c1 = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap();
    let table = Table::from_columns(vec![c0, c1]).unwrap();
    let tv = table.view().unwrap();

    let map = Column::from_slice(&[3i32, 1]).unwrap();
    let map_view = map.view().unwrap();

    let result = copying::gather(&tv, &map_view, copying::OutOfBoundsPolicy::DontCheck).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_columns().unwrap(), 2);
    assert_eq!(rv.num_rows().unwrap(), 2);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![4, 2]);
    assert_eq!(rv.column(1).unwrap().to_vec::<f64>().unwrap(), vec![40.0, 20.0]);
}

#[test]
fn test_gather_identity() {
    // Gather in order → identical copy.
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let map = Column::from_slice(&[0i32, 1, 2]).unwrap();
    let map_view = map.view().unwrap();

    let result = copying::gather(&tv, &map_view, copying::OutOfBoundsPolicy::DontCheck).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_gather_with_duplicates() {
    // Gather the same row multiple times.
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let map = Column::from_slice(&[1i32, 1, 1, 0]).unwrap();
    let map_view = map.view().unwrap();

    let result = copying::gather(&tv, &map_view, copying::OutOfBoundsPolicy::DontCheck).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![20, 20, 20, 10]);
}

#[test]
fn test_gather_empty_map() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let map = Column::from_slice(&[] as &[i32]).unwrap();
    let map_view = map.view().unwrap();

    let result = copying::gather(&tv, &map_view, copying::OutOfBoundsPolicy::DontCheck).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 0);
}

#[test]
fn test_gather_out_of_bounds_nullify() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    // Index 5 is out of bounds — should become null.
    let map = Column::from_slice(&[0i32, 5, 2]).unwrap();
    let map_view = map.view().unwrap();

    let result = copying::gather(&tv, &map_view, copying::OutOfBoundsPolicy::Nullify).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3);
    let cv = rv.column(0).unwrap();
    // Row 1 should be null.
    let validity = cv.validity().unwrap().expect("should have null mask");
    assert!(validity[0]);
    assert!(!validity[1]); // out-of-bounds → null
    assert!(validity[2]);
}
