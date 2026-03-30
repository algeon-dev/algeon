use cudf::{Column, DataFrame};

#[allow(dead_code)]
pub fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = Column::from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}
