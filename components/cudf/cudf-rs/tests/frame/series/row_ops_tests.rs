//! Integration tests for `Series` row-selection convenience methods.
//!
//! Methods under test: `head`, `tail`, `slice`, `gather`, `filter`,
//! `dropna`, `drop_duplicates`, `value_counts`.
//!
//! All tests require a GPU.

use cudf::{Series, core::Column};

#[test]
fn test_head_basic() {
    let s = Series::new::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let head = s.head(2).unwrap();

    assert_eq!(head.name(), Some("x"));
    assert_eq!(head.to_vec_strict::<i32>().unwrap(), vec![10, 20]);
}

#[test]
fn test_tail_clamps_to_len() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let tail = s.tail(99).unwrap();

    assert_eq!(tail.name(), Some("x"));
    assert_eq!(tail.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_slice_preserves_unnamed_series() {
    let column = Column::from_slice(&[10i32, 20, 30, 40]).unwrap();
    let s = Series::from_column(column).unwrap();
    let sliced = s.slice(1, 2).unwrap();

    assert_eq!(sliced.name(), None);
    assert_eq!(sliced.to_vec_strict::<i32>().unwrap(), vec![20, 30]);
}

#[test]
fn test_filter_drops_false_and_null_mask_rows() {
    let values = Series::new::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let mask = Series::new_nullable::<bool>("mask", &[true, false, false, true], &[true, true, false, true]).unwrap();

    let filtered = values.filter(&mask).unwrap();

    assert_eq!(filtered.name(), Some("x"));
    assert_eq!(filtered.to_vec_strict::<i32>().unwrap(), vec![10, 40]);
}

#[test]
fn test_gather_out_of_bounds_nullifies_rows() {
    let values = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let indices = Series::new::<i32>("idx", &[2, 99, 0]).unwrap();

    let gathered = values.gather(&indices).unwrap();

    assert_eq!(gathered.name(), Some("x"));
    assert_eq!(gathered.to_optional_vec::<i32>().unwrap(), vec![Some(30), None, Some(10)]);
}

#[test]
fn test_dropna_removes_null_rows() {
    let values = Series::from_optional("x", &[Some(10i32), None, Some(30), None]).unwrap();

    let cleaned = values.dropna().unwrap();

    assert_eq!(cleaned.name(), Some("x"));
    assert_eq!(cleaned.null_count(), 0);
    assert_eq!(cleaned.to_vec_strict::<i32>().unwrap(), vec![10, 30]);
}

#[test]
fn test_drop_duplicates_returns_unique_values() {
    let values = Series::new::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();

    let deduped = values.drop_duplicates().unwrap();

    assert_eq!(deduped.name(), Some("x"));
    let mut host = deduped.to_vec_strict::<i32>().unwrap();
    host.sort();
    assert_eq!(host, vec![1, 2, 3]);
}

#[test]
fn test_value_counts_returns_value_and_count_columns() {
    let values = Series::from_optional("x", &[Some(2i32), Some(1), None, Some(2), Some(1), Some(2)]).unwrap();

    let counts = values.value_counts().unwrap();

    assert_eq!(counts.column_names(), vec!["x", "count"]);
    let mut rows: Vec<(i32, i32)> = counts
        .series("x")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(counts.series("count").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 2), (2, 3)]);
}

#[test]
fn test_value_counts_for_unnamed_series_uses_fallback_value_name() {
    let column = Column::from_slice(&[5i32, 5, 7]).unwrap();
    let values = Series::from_column(column).unwrap();

    let counts = values.value_counts().unwrap();

    assert_eq!(counts.column_names(), vec!["value", "count"]);
    let mut rows: Vec<(i32, i32)> = counts
        .series("value")
        .unwrap()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(counts.series("count").unwrap().to_vec_strict::<i32>().unwrap())
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(5, 2), (7, 1)]);
}
