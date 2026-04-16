//! Integration tests for `Series` row-selection convenience methods.
//!
//! Methods under test: `head`, `tail`, `slice`, `gather`, `filter`,
//! `dropna`, `drop_duplicates`, `value_counts`.
//!
//! All tests require a GPU.

use cudf::Series;

use crate::common::column_from_slice;

#[test]
fn test_head_basic() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let head = s.head(2).unwrap();

    assert_eq!(head.name(), Some("x"));
    assert_eq!(head.to_vec_strict::<i32>().unwrap(), vec![10, 20]);
}

#[test]
fn test_tail_clamps_to_len() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let tail = s.tail(99).unwrap();

    assert_eq!(tail.name(), Some("x"));
    assert_eq!(tail.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_slice_preserves_unnamed_series() {
    let column = column_from_slice(&[10i32, 20, 30, 40]).unwrap();
    let s = Series::from_column(column).unwrap();
    let sliced = s.slice(1, 2).unwrap();

    assert_eq!(sliced.name(), None);
    assert_eq!(sliced.to_vec_strict::<i32>().unwrap(), vec![20, 30]);
}

#[test]
fn test_filter_drops_false_and_null_mask_rows() {
    let values = crate::common::series::<i32>("x", &[10, 20, 30, 40]).unwrap();
    let mask = crate::common::nullable_series::<bool>("mask", &[true, false, false, true], &[true, true, false, true])
        .unwrap();

    let filtered = values.filter(&mask).unwrap();

    assert_eq!(filtered.name(), Some("x"));
    assert_eq!(filtered.to_vec_strict::<i32>().unwrap(), vec![10, 40]);
}

#[test]
fn test_gather_out_of_bounds_nullifies_rows() {
    let values = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let indices = crate::common::series::<i32>("idx", &[2, 99, 0]).unwrap();

    let gathered = values.gather(&indices).unwrap();

    assert_eq!(gathered.name(), Some("x"));
    assert_eq!(gathered.to_optional_vec::<i32>().unwrap(), vec![Some(30), None, Some(10)]);
}

#[test]
fn test_dropna_removes_null_rows() {
    let values = crate::common::series_opt("x", &[Some(10i32), None, Some(30), None]).unwrap();

    let cleaned = values.dropna().unwrap();

    assert_eq!(cleaned.name(), Some("x"));
    assert_eq!(cleaned.null_count(), 0);
    assert_eq!(cleaned.to_vec_strict::<i32>().unwrap(), vec![10, 30]);
}

#[test]
fn test_drop_duplicates_returns_unique_values() {
    let values = crate::common::series::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();

    let deduped = values.drop_duplicates().unwrap();

    assert_eq!(deduped.name(), Some("x"));
    let mut host = deduped.to_vec_strict::<i32>().unwrap();
    host.sort();
    assert_eq!(host, vec![1, 2, 3]);
}

#[test]
fn test_value_counts_returns_value_and_count_columns() {
    let values = crate::common::series_opt("x", &[Some(2i32), Some(1), None, Some(2), Some(1), Some(2)]).unwrap();

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
    let column = column_from_slice(&[5i32, 5, 7]).unwrap();
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

// -- Semantic lock-down: stable across the column-primitive refactor --------

#[test]
fn test_head_on_empty_series_returns_empty_with_name() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let head = s.head(10).unwrap();

    assert_eq!(head.name(), Some("x"));
    assert_eq!(head.len(), 0);
    assert_eq!(head.to_vec_strict::<i32>().unwrap(), Vec::<i32>::new());
}

#[test]
fn test_slice_offset_beyond_len_returns_empty() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let sliced = s.slice(10, 5).unwrap();

    assert_eq!(sliced.name(), Some("x"));
    assert_eq!(sliced.len(), 0);
}

#[test]
fn test_gather_empty_index_returns_empty_with_name() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let idx = crate::common::series::<i32>("idx", &[]).unwrap();

    let gathered = s.gather(&idx).unwrap();
    assert_eq!(gathered.name(), Some("x"));
    assert_eq!(gathered.len(), 0);
}

#[test]
fn test_gather_operation_tag_on_dtype_error() {
    let s = crate::common::series::<i32>("x", &[10, 20]).unwrap();
    // Indices must be Int32; pass Float64 to trigger dtype error.
    let bogus = crate::common::series::<f64>("idx", &[0.0, 1.0]).unwrap();

    let err = s.gather(&bogus).unwrap_err();
    assert_eq!(
        err.operation(),
        "series::gather",
        "error should surface through the series op name, not the internal dataframe path"
    );
}

#[test]
fn test_filter_operation_tag_on_dtype_error() {
    let s = crate::common::series::<i32>("x", &[10, 20]).unwrap();
    // Mask must be BOOL8; pass Int32 to trigger dtype error.
    let bogus = crate::common::series::<i32>("mask", &[1, 0]).unwrap();

    let err = s.filter(&bogus).unwrap_err();
    assert_eq!(err.operation(), "series::filter");
}

#[test]
fn test_dropna_on_non_nullable_returns_same_values() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let out = s.dropna().unwrap();

    assert_eq!(out.name(), Some("x"));
    assert_eq!(out.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_drop_duplicates_preserves_unnamed_series() {
    let column = column_from_slice(&[2i32, 2, 3]).unwrap();
    let s = Series::from_column(column).unwrap();
    let deduped = s.drop_duplicates().unwrap();

    assert_eq!(deduped.name(), None);
    let mut host = deduped.to_vec_strict::<i32>().unwrap();
    host.sort();
    assert_eq!(host, vec![2, 3]);
}

#[test]
fn test_value_counts_with_sentinel_collision_name() {
    // The legacy implementation used the sentinel "__cudf_series_value__"
    // as the internal column name. A series whose real name equals the
    // sentinel must still produce a clean (sentinel, count) schema.
    let values = crate::common::series::<i32>("__cudf_series_value__", &[1, 1, 2]).unwrap();

    let counts = values.value_counts().unwrap();
    assert_eq!(counts.column_names(), vec!["__cudf_series_value__", "count"]);
}
