//! Tests for DataFrame row slicing helpers.

use cudf::{DataFrame, core::Column};

/// Helper: build a 5-row, 2-column DataFrame for slicing tests.
fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let vals = Column::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![ids, vals]).unwrap()
}

// ---------------------------------------------------------------------------
// head
// ---------------------------------------------------------------------------

#[test]
fn test_head_basic() {
    let df = sample_df();
    let h = df.head(2).unwrap();

    assert_eq!(h.num_rows(), 2);
    assert_eq!(h.num_columns(), 2);
    assert_eq!(h.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 20]);
    assert_eq!(h.series("val").unwrap().to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_head_zero_rows() {
    let df = sample_df();
    let h = df.head(0).unwrap();

    assert_eq!(h.num_rows(), 0);
    assert_eq!(h.num_columns(), 2);
    assert_eq!(h.column_names(), vec!["id", "val"]);
}

#[test]
fn test_head_clamps_to_num_rows() {
    let df = sample_df();
    let h = df.head(100).unwrap();

    assert_eq!(h.num_rows(), 5);
    assert_eq!(h.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30, 40, 50]);
}

#[test]
fn test_head_preserves_schema() {
    let df = sample_df();
    let h = df.head(3).unwrap();
    assert_eq!(h.column_names(), vec!["id", "val"]);
    assert_eq!(h.column_index("id"), Some(0));
    assert_eq!(h.column_index("val"), Some(1));
}

// ---------------------------------------------------------------------------
// tail
// ---------------------------------------------------------------------------

#[test]
fn test_tail_basic() {
    let df = sample_df();
    let t = df.tail(2).unwrap();

    assert_eq!(t.num_rows(), 2);
    assert_eq!(t.series("id").unwrap().to_vec::<i32>().unwrap(), vec![40, 50]);
    assert_eq!(t.series("val").unwrap().to_vec::<f64>().unwrap(), vec![4.0, 5.0]);
}

#[test]
fn test_tail_clamps_to_num_rows() {
    let df = sample_df();
    let t = df.tail(100).unwrap();

    assert_eq!(t.num_rows(), 5);
    assert_eq!(t.series("id").unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30, 40, 50]);
}

#[test]
fn test_tail_zero_rows() {
    let df = sample_df();
    let t = df.tail(0).unwrap();

    assert_eq!(t.num_rows(), 0);
    assert_eq!(t.column_names(), vec!["id", "val"]);
}

#[test]
fn test_tail_preserves_schema() {
    let df = sample_df();
    let t = df.tail(3).unwrap();
    assert_eq!(t.column_names(), vec!["id", "val"]);
}

// ---------------------------------------------------------------------------
// slice
// ---------------------------------------------------------------------------

#[test]
fn test_slice_basic() {
    let df = sample_df();
    let s = df.slice(1, 2).unwrap();

    assert_eq!(s.num_rows(), 2);
    assert_eq!(s.series("id").unwrap().to_vec::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(s.series("val").unwrap().to_vec::<f64>().unwrap(), vec![2.0, 3.0]);
}

#[test]
fn test_slice_clamps_length() {
    let df = sample_df();
    // offset=3, length=100 → should clamp to rows 3..5
    let s = df.slice(3, 100).unwrap();

    assert_eq!(s.num_rows(), 2);
    assert_eq!(s.series("id").unwrap().to_vec::<i32>().unwrap(), vec![40, 50]);
}

#[test]
fn test_slice_zero_length() {
    let df = sample_df();
    let s = df.slice(2, 0).unwrap();

    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

#[test]
fn test_slice_preserves_schema() {
    let df = sample_df();
    let s = df.slice(0, 3).unwrap();
    assert_eq!(s.column_names(), vec!["id", "val"]);
    assert_eq!(s.column_index("id"), Some(0));
    assert_eq!(s.column_index("val"), Some(1));
}

#[test]
fn test_slice_offset_at_end() {
    let df = sample_df();
    // offset == num_rows → empty result
    let s = df.slice(5, 10).unwrap();
    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}

#[test]
fn test_slice_offset_beyond_end() {
    let df = sample_df();
    // offset > num_rows → empty result
    let s = df.slice(999, 10).unwrap();
    assert_eq!(s.num_rows(), 0);
    assert_eq!(s.column_names(), vec!["id", "val"]);
}
