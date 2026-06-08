//! Tests for DataFrame::gather().

use cudf::{Series, core::Column};

#[test]
fn test_gather_sequential_indices() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Series::new::<i32>("idx", &[0, 1, 2]).unwrap();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_vec_optional.
    let vals = result.series("x").unwrap().to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), Some(20), Some(30)]);
}

#[test]
fn test_gather_reversed() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Series::new::<i32>("idx", &[2, 1, 0]).unwrap();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_vec_optional.
    let vals = result.series("x").unwrap().to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(30), Some(20), Some(10)]);
}

#[test]
fn test_gather_out_of_bounds_nullify() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Series::new::<i32>("idx", &[0, 99, 2]).unwrap();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    let view = result.series("x").unwrap();
    let validity = view.as_column_view().unwrap().validity().unwrap().unwrap();
    assert!(validity[0]);
    assert!(!validity[1]); // OOB → null
    assert!(validity[2]);
}

#[test]
fn test_gather_duplicate_indices() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Series::new::<i32>("idx", &[1, 1, 1]).unwrap();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_vec_optional.
    let vals = result.series("x").unwrap().to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(20), Some(20), Some(20)]);
}

#[test]
fn test_gather_preserves_schema() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![c1, c2]).unwrap();

    let indices = Series::new::<i32>("idx", &[2, 0]).unwrap();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.column_names(), &["id", "score"]);
    assert_eq!(result.num_rows(), 2);
    // Gather produces nullable output; use to_vec_optional.
    assert_eq!(result.series("id").unwrap().to_vec_optional::<i32>().unwrap(), vec![Some(3), Some(1)]);
    assert_eq!(result.series("score").unwrap().to_vec_optional::<f64>().unwrap(), vec![Some(30.0), Some(10.0)]);
}

#[test]
fn test_gather_empty_indices() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let df = crate::common::dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let indices = Series::new::<i32>("idx", &[]).unwrap();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), &["x"]);
}

#[test]
fn test_gather_multi_type() {
    let c1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = Column::from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let c3 = Column::from_strs(&["a", "b", "c"]).unwrap();
    let df = crate::common::dataframe_from_columns(
        vec!["ints".to_string(), "floats".to_string(), "strs".to_string()],
        vec![c1, c2, c3],
    )
    .unwrap();

    let indices = Series::new::<i32>("idx", &[2, 0]).unwrap();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.num_rows(), 2);
    // Gather produces nullable output; use to_vec_optional / to_strings_optional.
    assert_eq!(result.series("ints").unwrap().to_vec_optional::<i32>().unwrap(), vec![Some(3), Some(1)]);
    assert_eq!(result.series("floats").unwrap().to_vec_optional::<f64>().unwrap(), vec![Some(3.5), Some(1.5)]);
    assert_eq!(
        result.series("strs").unwrap().to_strings_optional().unwrap(),
        vec![Some("c".to_string()), Some("a".to_string())]
    );
}
