//! Tests for CU-7: gather (take).

use cudf::{
    core::{Column, Table},
    ops::copying,
};

use crate::common::{
    explicit_i32_column, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation,
    reset_sync_observation,
};

#[test]
fn test_gather_basic() {
    let col = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    // Gather rows 4, 2, 0.
    let map = Column::from_slice(&[4i32, 2, 0]).unwrap();
    let map_view = map.view().unwrap();

    let result = copying::gather(&tv, &map_view).unwrap();
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

    let result = copying::gather(&tv, &map_view).unwrap();
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

    let result = copying::gather(&tv, &map_view).unwrap();
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

    let result = copying::gather(&tv, &map_view).unwrap();
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

    let result = copying::gather(&tv, &map_view).unwrap();
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

    let result = copying::gather(&tv, &map_view).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 3);
    let cv = rv.column(0).unwrap();
    // Row 1 should be null.
    let validity = cv.validity().unwrap().expect("should have null mask");
    assert!(validity[0]);
    assert!(!validity[1]); // out-of-bounds → null
    assert!(validity[2]);
}

#[test]
fn test_gather_unchecked_basic() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let tv = table.view().unwrap();

    let map = Column::from_slice(&[2i32, 0]).unwrap();
    let map_view = map.view().unwrap();

    // SAFETY: indices 2 and 0 are valid for a 3-row table.
    let result = unsafe { copying::gather_unchecked(&tv, &map_view) }.unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.num_rows().unwrap(), 2);
    assert_eq!(rv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![30, 10]);
}

#[test]
fn test_gather_succeeds_with_mixed_domains_and_uses_input_provenance() {
    let table = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 930)]).unwrap();
    let gather_map = explicit_i32_column(&[2, 0], 931);
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();

    reset_exec_observation();
    let result = copying::gather(&table.view().unwrap(), &gather_map.view().unwrap()).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), table.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![30, 10]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}
