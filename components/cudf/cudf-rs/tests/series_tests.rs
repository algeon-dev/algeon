//! Integration tests for `Series` — the named, Arc-shared GPU column wrapper.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{column::Column, series::Series, types::TypeId};

// ---------------------------------------------------------------------------
// Constructors preserve dtype and length
// ---------------------------------------------------------------------------

#[test]
fn test_new_i32_preserves_dtype_and_len() {
    let s = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len().unwrap(), 3);
    assert!(!s.is_empty().unwrap());
}

#[test]
fn test_new_f64_preserves_dtype_and_len() {
    let s = Series::new::<f64>("scores", &[1.0, 2.0]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len().unwrap(), 2);
}

#[test]
fn test_new_bool_preserves_dtype_and_len() {
    let s = Series::new::<bool>("flags", &[true, false, true]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.len().unwrap(), 3);
}

#[test]
fn test_new_empty_slice() {
    let s = Series::new::<i32>("empty", &[]).unwrap();
    assert_eq!(s.len().unwrap(), 0);
    assert!(s.is_empty().unwrap());
}

#[test]
fn test_new_nullable_preserves_null_count() {
    let s = Series::new_nullable::<i32>("x", &[10, 20, 30], &[true, false, true]).unwrap();
    assert_eq!(s.len().unwrap(), 3);
    assert_eq!(s.null_count().unwrap(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

#[test]
fn test_from_strings() {
    let s = Series::from_strings("names", &["alice", "bob"]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len().unwrap(), 2);
}

#[test]
fn test_from_strings_nullable() {
    let s = Series::from_strings_nullable("names", &["a", "b", "c"], &[true, false, true]).unwrap();
    assert_eq!(s.null_count().unwrap(), 1);
    assert_eq!(s.len().unwrap(), 3);
}

// ---------------------------------------------------------------------------
// Name accessors
// ---------------------------------------------------------------------------

#[test]
fn test_name_returns_constructor_name() {
    let s = Series::new::<i32>("my_col", &[1]).unwrap();
    assert_eq!(s.name(), Some("my_col"));
}

#[test]
fn test_from_column_has_no_name() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let s = Series::from_column(col);
    assert_eq!(s.name(), None);
}

// ---------------------------------------------------------------------------
// rename returns new Series, original unchanged
// ---------------------------------------------------------------------------

#[test]
fn test_rename_returns_new_series() {
    let s = Series::new::<i32>("old", &[1, 2, 3]).unwrap();
    let renamed = s.rename("new");

    assert_eq!(renamed.name(), Some("new"));
    assert_eq!(s.name(), Some("old")); // original unchanged
}

#[test]
fn test_rename_shares_arc() {
    let s = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let renamed = s.rename("b");

    // Both should point to the same GPU data (Arc sharing).
    let ptr_orig = s.view().unwrap().device_data_ptr().unwrap();
    let ptr_renamed = renamed.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr_orig, ptr_renamed);
}

// ---------------------------------------------------------------------------
// clone is cheap (same Arc)
// ---------------------------------------------------------------------------

#[test]
fn test_clone_shares_gpu_data() {
    let s = Series::new::<f64>("vals", &[10.0, 20.0]).unwrap();
    let cloned = s.clone();

    let ptr_s = s.view().unwrap().device_data_ptr().unwrap();
    let ptr_cloned = cloned.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr_s, ptr_cloned);

    // Both are independently usable.
    assert_eq!(cloned.name(), Some("vals"));
    assert_eq!(cloned.len().unwrap(), 2);
}

// ---------------------------------------------------------------------------
// from_column / from_column_arc round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_from_column_round_trip() {
    let col = Column::from_slice(&[5i32, 10, 15]).unwrap();
    let s = Series::from_column(col);

    assert_eq!(s.name(), None);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len().unwrap(), 3);

    // Access the underlying column.
    let view = s.column().view().unwrap();
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![5, 10, 15]);
}

#[test]
fn test_from_column_arc_shares_arc() {
    let col = Arc::new(Column::from_slice(&[1i64, 2]).unwrap());
    let arc_clone = Arc::clone(&col);
    let s = Series::from_column_arc(arc_clone);

    // The Series should share the same Arc (same device pointer).
    let ptr_orig = col.view().unwrap().device_data_ptr().unwrap();
    let ptr_series = s.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr_orig, ptr_series);

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int64);
    assert_eq!(s.len().unwrap(), 2);
}

// ---------------------------------------------------------------------------
// view() returns a valid ColumnView
// ---------------------------------------------------------------------------

#[test]
fn test_view_matches_column_data() {
    let s = Series::new::<i32>("x", &[100, 200, 300]).unwrap();
    let view = s.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![100, 200, 300]);
}
