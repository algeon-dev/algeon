//! Integration tests for `Series` — the named, Arc-shared GPU column wrapper.
//!
//! All tests require a GPU.

use std::sync::Arc;

use cudf::{Column, Series, TypeId};

// ---------------------------------------------------------------------------
// Constructors preserve dtype and length
// ---------------------------------------------------------------------------

#[test]
fn test_new_i32_preserves_dtype_and_len() {
    let s = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);
    assert!(!s.is_empty());
}

#[test]
fn test_new_f64_preserves_dtype_and_len() {
    let s = Series::new::<f64>("scores", &[1.0, 2.0]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 2);
}

#[test]
fn test_new_bool_preserves_dtype_and_len() {
    let s = Series::new::<bool>("flags", &[true, false, true]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.len(), 3);
}

#[test]
fn test_new_empty_slice() {
    let s = Series::new::<i32>("empty", &[]).unwrap();
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

#[test]
fn test_new_nullable_preserves_null_count() {
    let s = Series::new_nullable::<i32>("x", &[10, 20, 30], &[true, false, true]).unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

#[test]
fn test_from_strings() {
    let s = Series::from_strings("names", &["alice", "bob"]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 2);
}

#[test]
fn test_from_strings_nullable() {
    let s = Series::from_strings_nullable("names", &["a", "b", "c"], &[true, false, true]).unwrap();
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.len(), 3);
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
    let s = Series::from_column(col).unwrap();
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
    assert_eq!(cloned.len(), 2);
}

// ---------------------------------------------------------------------------
// from_column / from_column_arc round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_from_column_round_trip() {
    let col = Column::from_slice(&[5i32, 10, 15]).unwrap();
    let s = Series::from_column(col).unwrap();

    assert_eq!(s.name(), None);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);

    // Access the underlying column.
    let view = s.as_column().view().unwrap();
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![5, 10, 15]);
}

#[test]
fn test_from_column_arc_shares_arc() {
    let col = Arc::new(Column::from_slice(&[1i64, 2]).unwrap());
    let arc_clone = Arc::clone(&col);
    let s = Series::from_column_arc(arc_clone).unwrap();

    // The Series should share the same Arc (same device pointer).
    let ptr_orig = col.view().unwrap().device_data_ptr().unwrap();
    let ptr_series = s.view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr_orig, ptr_series);

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int64);
    assert_eq!(s.len(), 2);
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

// ===========================================================================
// Minimal scalar comparison ops
// ===========================================================================

#[test]
fn test_gt_scalar_i32() {
    let s = Series::new::<i32>("x", &[1, 5, 10]).unwrap();
    let mask = s.gt_scalar(5i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask.name(), Some("x"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, false, true]);
}

#[test]
fn test_lt_scalar_f64() {
    let s = Series::new::<f64>("y", &[1.0, 5.0, 10.0]).unwrap();
    let mask = s.lt_scalar(5.0f64).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("y"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_eq_scalar_i32() {
    let s = Series::new::<i32>("z", &[1, 2, 2, 3]).unwrap();
    let mask = s.eq_scalar(2i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("z"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, true, false]);
}

#[test]
fn test_eq_scalar_str() {
    let s = Series::from_strings("status", &["active", "inactive", "active"]).unwrap();
    let mask = s.eq_scalar_str("active").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("status"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_ne_scalar_str() {
    let s = Series::from_strings("status", &["active", "inactive", "active"]).unwrap();
    let mask = s.ne_scalar_str("active").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("status"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_gt_scalar_str() {
    // Lexicographic: "banana" > "apple", "cherry" > "apple", "apple" == "apple"
    let s = Series::from_strings("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.gt_scalar_str("apple").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("fruit"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn test_lt_scalar_str() {
    let s = Series::from_strings("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.lt_scalar_str("banana").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("fruit"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_ge_scalar_str() {
    let s = Series::from_strings("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.ge_scalar_str("banana").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("fruit"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn test_le_scalar_str() {
    let s = Series::from_strings("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.le_scalar_str("banana").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("fruit"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_gt_scalar_preserves_input() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let _mask = s.gt_scalar(2i32).unwrap();

    // Original series is unchanged.
    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// Extended scalar comparison ops
// ===========================================================================

#[test]
fn test_ne_scalar_i32() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let mask = s.ne_scalar(2i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("x"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_ge_scalar_i32() {
    let s = Series::new::<i32>("x", &[1, 5, 10]).unwrap();
    let mask = s.ge_scalar(5i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("x"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn test_le_scalar_i32() {
    let s = Series::new::<i32>("x", &[1, 5, 10]).unwrap();
    let mask = s.le_scalar(5i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), Some("x"));
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, true, false]);
}

// ===========================================================================
// Column-vs-column comparison ops
// ===========================================================================

#[test]
fn test_eq_series() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[1, 9, 3]).unwrap();
    let mask = a.eq(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.name(), None);
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_ne_series() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[1, 9, 3]).unwrap();
    let mask = a.ne(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_gt_series() {
    let a = Series::new::<i32>("a", &[1, 5, 3]).unwrap();
    let b = Series::new::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.gt(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_lt_series() {
    let a = Series::new::<i32>("a", &[1, 5, 3]).unwrap();
    let b = Series::new::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.lt(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn test_ge_series() {
    let a = Series::new::<i32>("a", &[1, 5, 3]).unwrap();
    let b = Series::new::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.ge(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn test_le_series() {
    let a = Series::new::<i32>("a", &[1, 5, 3]).unwrap();
    let b = Series::new::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.le(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_comparison_mixed_types_f64_i32() {
    // Column comparison with mixed numeric types should work (type promotion).
    let a = Series::new::<f64>("a", &[1.0, 2.5, 3.0]).unwrap();
    let b = Series::new::<i32>("b", &[1, 3, 3]).unwrap();
    let mask = a.eq(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.view().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
}

// ===========================================================================
// Arithmetic ops
// ===========================================================================

#[test]
fn test_add_series() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[10, 20, 30]).unwrap();
    let result = a.add(&b).unwrap();

    assert_eq!(result.name(), None);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_sub_series() {
    let a = Series::new::<i32>("a", &[10, 20, 30]).unwrap();
    let b = Series::new::<i32>("b", &[1, 2, 3]).unwrap();
    let result = a.sub(&b).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![9, 18, 27]);
}

#[test]
fn test_mul_series() {
    let a = Series::new::<i32>("a", &[2, 3, 4]).unwrap();
    let b = Series::new::<i32>("b", &[10, 10, 10]).unwrap();
    let result = a.mul(&b).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40]);
}

#[test]
fn test_div_series_f64() {
    let a = Series::new::<f64>("a", &[10.0, 20.0, 30.0]).unwrap();
    let b = Series::new::<f64>("b", &[2.0, 4.0, 5.0]).unwrap();
    let result = a.div(&b).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.view().unwrap().to_vec::<f64>().unwrap(), vec![5.0, 5.0, 6.0]);
}

#[test]
fn test_add_scalar_i32() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.add_scalar(10i32).unwrap();

    assert_eq!(result.name(), Some("x"));
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 12, 13]);
}

#[test]
fn test_sub_scalar_i32() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let result = s.sub_scalar(5i32).unwrap();

    assert_eq!(result.name(), Some("x"));
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![5, 15, 25]);
}

#[test]
fn test_mul_scalar_f64() {
    let s = Series::new::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let result = s.mul_scalar(2.0f64).unwrap();

    assert_eq!(result.name(), Some("x"));
    assert_eq!(result.view().unwrap().to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_div_scalar_f64() {
    let s = Series::new::<f64>("x", &[10.0, 20.0, 30.0]).unwrap();
    let result = s.div_scalar(5.0f64).unwrap();

    assert_eq!(result.name(), Some("x"));
    assert_eq!(result.view().unwrap().to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_arithmetic_mixed_types() {
    // i32 + f64 should promote to f64
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<f64>("b", &[0.5, 0.5, 0.5]).unwrap();
    let result = a.add(&b).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.view().unwrap().to_vec::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_add_operator_with_borrowed_operands() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[10, 20, 30]).unwrap();

    let result = &a + &b;

    assert_eq!(result.name(), None);
    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn test_add_operator_with_owned_and_borrowed_operands() {
    let a = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let b = Series::new::<i32>("b", &[10, 20, 30]).unwrap();

    let result = a + &b;

    assert_eq!(result.view().unwrap().to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}
