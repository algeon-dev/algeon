#![allow(deprecated)] // Tests exercise deprecated DataFrame::column()
//! Tests for DataFrame::gather().

use cudf::{column::Column, dataframe::DataFrame};

#[test]
fn test_gather_sequential_indices() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Column::from_slice(&[0i32, 1, 2]).unwrap();
    let result = df.gather(&indices.view().unwrap()).unwrap();

    assert_eq!(result.num_rows(), 3);
    let vals = result.column("x").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn test_gather_reversed() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Column::from_slice(&[2i32, 1, 0]).unwrap();
    let result = df.gather(&indices.view().unwrap()).unwrap();

    assert_eq!(result.num_rows(), 3);
    let vals = result.column("x").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![30, 20, 10]);
}

#[test]
fn test_gather_out_of_bounds_nullify() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Column::from_slice(&[0i32, 99, 2]).unwrap();
    let result = df.gather(&indices.view().unwrap()).unwrap();

    assert_eq!(result.num_rows(), 3);
    let view = result.column("x").unwrap();
    let validity = view.validity().unwrap().unwrap();
    assert!(validity[0]);
    assert!(!validity[1]); // OOB → null
    assert!(validity[2]);
}

#[test]
fn test_gather_duplicate_indices() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Column::from_slice(&[1i32, 1, 1]).unwrap();
    let result = df.gather(&indices.view().unwrap()).unwrap();

    assert_eq!(result.num_rows(), 3);
    let vals = result.column("x").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(vals, vec![20, 20, 20]);
}

#[test]
fn test_gather_preserves_schema() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let df = DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![c1, c2]).unwrap();

    let indices = Column::from_slice(&[2i32, 0]).unwrap();
    let result = df.gather(&indices.view().unwrap()).unwrap();

    assert_eq!(result.column_names(), &["id", "score"]);
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column("id").unwrap().to_vec::<i32>().unwrap(), vec![3, 1]);
    assert_eq!(result.column("score").unwrap().to_vec::<f64>().unwrap(), vec![30.0, 10.0]);
}

#[test]
fn test_gather_empty_indices() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = DataFrame::new(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Column::from_slice::<i32>(&[]).unwrap();
    let result = df.gather(&indices.view().unwrap()).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), &["x"]);
}

#[test]
fn test_gather_multi_type() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let c3 = Column::from_strings(&["a", "b", "c"]).unwrap();
    let df =
        DataFrame::new(vec!["ints".to_string(), "floats".to_string(), "strs".to_string()], vec![c1, c2, c3]).unwrap();

    let indices = Column::from_slice(&[2i32, 0]).unwrap();
    let result = df.gather(&indices.view().unwrap()).unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column("ints").unwrap().to_vec::<i32>().unwrap(), vec![3, 1]);
    assert_eq!(result.column("floats").unwrap().to_vec::<f64>().unwrap(), vec![3.5, 1.5]);
    assert_eq!(result.column("strs").unwrap().to_strings().unwrap(), vec!["c", "a"]);
}
