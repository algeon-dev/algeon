//! Tests for ColumnView::map() — value mapping via left join.

use cudf::{TypeId, core::Column};

#[test]
fn test_map_string_to_int() {
    let input = Column::from_strs(&["a", "b", "c", "a"]).unwrap();
    let keys = Column::from_strs(&["a", "b", "c"]).unwrap();
    let values = Column::from_slice(&[1i32, 2, 3]).unwrap();

    let result = input.view().unwrap().map(&keys.view().unwrap(), &values.view().unwrap()).unwrap();

    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![1, 2, 3, 1]);
}

#[test]
fn test_map_missing_keys_become_null() {
    let input = Column::from_slice(&[1i32, 2, 3, 1]).unwrap();
    let keys = Column::from_slice(&[1i32, 2]).unwrap();
    let values = Column::from_slice(&[10i32, 20]).unwrap();

    let result = input.view().unwrap().map(&keys.view().unwrap(), &values.view().unwrap()).unwrap();

    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 4);

    let validity = rv.validity().unwrap().unwrap();
    assert!(validity[0]); // 1 → 10
    assert!(validity[1]); // 2 → 20
    assert!(!validity[2]); // 3 not in keys → null
    assert!(validity[3]); // 1 → 10

    let vals = rv.to_vec::<i32>().unwrap();
    assert_eq!(vals[0], 10);
    assert_eq!(vals[1], 20);
    assert_eq!(vals[3], 10);
}

#[test]
fn test_map_empty_input() {
    let input = Column::from_strs(&[]).unwrap();
    let keys = Column::from_strs(&["a"]).unwrap();
    let values = Column::from_slice(&[1i32]).unwrap();

    let result = input.view().unwrap().map(&keys.view().unwrap(), &values.view().unwrap()).unwrap();

    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 0);
}

#[test]
fn test_map_int_to_int() {
    let input = Column::from_slice(&[1i32, 2, 3, 1]).unwrap();
    let keys = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let values = Column::from_slice(&[100i32, 200, 300]).unwrap();

    let result = input.view().unwrap().map(&keys.view().unwrap(), &values.view().unwrap()).unwrap();

    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![100, 200, 300, 100]);
}

#[test]
fn test_map_preserves_output_type() {
    let input = Column::from_slice(&[1i32, 2]).unwrap();
    let keys = Column::from_slice(&[1i32, 2]).unwrap();
    let values = Column::from_slice(&[1.5f64, 2.5]).unwrap();

    let result = input.view().unwrap().map(&keys.view().unwrap(), &values.view().unwrap()).unwrap();

    let rv = result.view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(rv.to_vec::<f64>().unwrap(), vec![1.5, 2.5]);
}

#[test]
fn test_map_rejects_duplicate_keys() {
    let input = Column::from_slice(&[1i32, 2, 1]).unwrap();
    let keys = Column::from_slice(&[1i32, 1, 2]).unwrap();
    let values = Column::from_slice(&[10i32, 11, 20]).unwrap();

    let result = input.view().unwrap().map(&keys.view().unwrap(), &values.view().unwrap());
    let err = match result {
        Ok(_) => panic!("expected duplicate keys to be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "column_view::map");
}
