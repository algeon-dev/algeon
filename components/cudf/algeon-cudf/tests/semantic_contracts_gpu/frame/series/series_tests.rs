//! Integration tests for `Series`; the Arc-shared GPU column wrapper.
//!
//! All tests require a GPU.

use cudf::{Exec, Series, TypeId, testing::prelude::*};

use crate::common::{column_from_slice, nullable_series, series};

// ---------------------------------------------------------------------------
// Constructors preserve dtype and length
// ---------------------------------------------------------------------------

fn test_new_i32_preserves_dtype_and_len() {
    let s = series::<i32>("ids", &[1, 2, 3]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);
    assert!(!s.is_empty());
}

fn test_new_f64_preserves_dtype_and_len() {
    let s = series::<f64>("scores", &[1.0, 2.0]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(s.len(), 2);
}

fn test_new_bool_preserves_dtype_and_len() {
    let s = series::<bool>("flags", &[true, false, true]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(s.len(), 3);
}

fn test_new_empty_slice() {
    let s = series::<i32>("empty", &[]).unwrap();
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

fn test_new_nullable_preserves_null_count() {
    let s = nullable_series::<i32>("x", &[10, 20, 30], &[true, false, true]).unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

fn test_new_strs_preserves_dtype_and_len() {
    let s = series("names", &["alice", "bob"]).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::String);
    assert_eq!(s.len(), 2);
}

fn test_new_nullable_strs_preserves_null_count() {
    let s = nullable_series("names", &["a", "b", "c"], &[true, false, true]).unwrap();
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.len(), 3);
}

fn test_new_nullable_strs_rejects_mismatched_validity_length() {
    let err = nullable_series("names", &["a", "b"], &[true]).unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "column::from_strs_nullable_with_runtime");
}

// ---------------------------------------------------------------------------
// clone is cheap (same Arc)
// ---------------------------------------------------------------------------

fn test_clone_shares_gpu_data() {
    let s = series::<f64>("vals", &[10.0, 20.0]).unwrap();
    let cloned = s.clone();

    let ptr_s = s.testing_view().unwrap().device_data_ptr().unwrap();
    let ptr_cloned = cloned.testing_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr_s, ptr_cloned);

    // Both are independently usable.
    assert_eq!(cloned.len(), 2);
}

// ---------------------------------------------------------------------------
// from_column / from_column_arc round-trip
// ---------------------------------------------------------------------------

fn test_from_column_round_trip() {
    let col = column_from_slice(&[5i32, 10, 15]).unwrap();
    let s = Series::from_column(col).unwrap();
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(s.len(), 3);

    // Access the underlying column.
    let view = s.as_column().testing_view().unwrap();
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![5, 10, 15]);
}

fn test_from_column_shares_data() {
    let col = column_from_slice(&[1i64, 2]).unwrap();
    let ptr_orig = col.testing_view().unwrap().device_data_ptr().unwrap();

    let s = Series::from_column(col).unwrap();

    // The Series should share the same device pointer.
    let ptr_series = s.testing_view().unwrap().device_data_ptr().unwrap();
    assert_eq!(ptr_orig, ptr_series);

    assert_eq!(s.dtype().unwrap().id(), TypeId::Int64);
    assert_eq!(s.len(), 2);
}

// ---------------------------------------------------------------------------
// view() returns a valid ColumnView
// ---------------------------------------------------------------------------

fn test_view_matches_column_data() {
    let s = series::<i32>("x", &[100, 200, 300]).unwrap();
    let view = s.testing_view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![100, 200, 300]);
}

// ===========================================================================
// Minimal scalar comparison ops
// ===========================================================================

fn test_gt_scalar_i32() {
    let s = series::<i32>("x", &[1, 5, 10]).unwrap();
    let mask = s.on(Exec::RETAINED).gt_scalar(5i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true]);
}

fn test_lt_scalar_f64() {
    let s = series::<f64>("y", &[1.0, 5.0, 10.0]).unwrap();
    let mask = s.on(Exec::RETAINED).lt_scalar(5.0f64).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn test_ne_scalar_string() {
    let s = series("status", &["active", "inactive", "active"]).unwrap();
    let mask = s.on(Exec::RETAINED).ne_scalar("active").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_gt_scalar_string() {
    // Lexicographic: "banana" > "apple", "cherry" > "apple", "apple" == "apple"
    let s = series("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.on(Exec::RETAINED).gt_scalar("apple").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

fn test_lt_scalar_string() {
    let s = series("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.on(Exec::RETAINED).lt_scalar("banana").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn test_ge_scalar_string() {
    let s = series("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.on(Exec::RETAINED).ge_scalar("banana").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

fn test_le_scalar_string() {
    let s = series("fruit", &["apple", "banana", "cherry"]).unwrap();
    let mask = s.on(Exec::RETAINED).le_scalar("banana").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

fn test_gt_scalar_preserves_input() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let _mask = s.on(Exec::RETAINED).gt_scalar(2i32).unwrap();

    // Original series is unchanged.
    assert_eq!(s.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// Extended scalar comparison ops
// ===========================================================================

fn test_ne_scalar_i32() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let mask = s.on(Exec::RETAINED).ne_scalar(2i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn test_ge_scalar_i32() {
    let s = series::<i32>("x", &[1, 5, 10]).unwrap();
    let mask = s.on(Exec::RETAINED).ge_scalar(5i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

fn test_le_scalar_i32() {
    let s = series::<i32>("x", &[1, 5, 10]).unwrap();
    let mask = s.on(Exec::RETAINED).le_scalar(5i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

// ===========================================================================
// Column-vs-column comparison ops
// ===========================================================================

fn test_eq_series() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = series::<i32>("b", &[1, 9, 3]).unwrap();
    let mask = a.on(Exec::RETAINED).eq(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn test_ne_series() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = series::<i32>("b", &[1, 9, 3]).unwrap();
    let mask = a.on(Exec::RETAINED).ne(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_gt_series() {
    let a = series::<i32>("a", &[1, 5, 3]).unwrap();
    let b = series::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.on(Exec::RETAINED).gt(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_lt_series() {
    let a = series::<i32>("a", &[1, 5, 3]).unwrap();
    let b = series::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.on(Exec::RETAINED).lt(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn test_ge_series() {
    let a = series::<i32>("a", &[1, 5, 3]).unwrap();
    let b = series::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.on(Exec::RETAINED).ge(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

fn test_le_series() {
    let a = series::<i32>("a", &[1, 5, 3]).unwrap();
    let b = series::<i32>("b", &[2, 3, 3]).unwrap();
    let mask = a.on(Exec::RETAINED).le(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn test_comparison_mixed_types_f64_i32() {
    // Column comparison with mixed numeric types should work (type promotion).
    let a = series::<f64>("a", &[1.0, 2.5, 3.0]).unwrap();
    let b = series::<i32>("b", &[1, 3, 3]).unwrap();
    let mask = a.on(Exec::RETAINED).eq(&b).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

// ===========================================================================
// Arithmetic ops
// ===========================================================================

fn test_add_series() {
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let b = series::<i32>("b", &[10, 20, 30]).unwrap();
    let result = a.on(Exec::RETAINED).add(&b).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

fn test_sub_series() {
    let a = series::<i32>("a", &[10, 20, 30]).unwrap();
    let b = series::<i32>("b", &[1, 2, 3]).unwrap();
    let result = a.on(Exec::RETAINED).sub(&b).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![9, 18, 27]);
}

fn test_mul_series() {
    let a = series::<i32>("a", &[2, 3, 4]).unwrap();
    let b = series::<i32>("b", &[10, 10, 10]).unwrap();
    let result = a.on(Exec::RETAINED).mul(&b).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![20, 30, 40]);
}

fn test_div_series_f64() {
    let a = series::<f64>("a", &[10.0, 20.0, 30.0]).unwrap();
    let b = series::<f64>("b", &[2.0, 4.0, 5.0]).unwrap();
    let result = a.on(Exec::RETAINED).div(&b).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![5.0, 5.0, 6.0]);
}

fn test_add_scalar_i32() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.on(Exec::RETAINED).add_scalar(10i32).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![11, 12, 13]);
}

fn test_sub_scalar_i32() {
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let result = s.on(Exec::RETAINED).sub_scalar(5i32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![5, 15, 25]);
}

fn test_mul_scalar_f64() {
    let s = series::<f64>("x", &[1.0, 2.0, 3.0]).unwrap();
    let result = s.on(Exec::RETAINED).mul_scalar(2.0f64).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

fn test_div_scalar_f64() {
    let s = series::<f64>("x", &[10.0, 20.0, 30.0]).unwrap();
    let result = s.on(Exec::RETAINED).div_scalar(5.0f64).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("series_tests::test_new_i32_preserves_dtype_and_len", test_new_i32_preserves_dtype_and_len),
    ("series_tests::test_new_f64_preserves_dtype_and_len", test_new_f64_preserves_dtype_and_len),
    ("series_tests::test_new_bool_preserves_dtype_and_len", test_new_bool_preserves_dtype_and_len),
    ("series_tests::test_new_empty_slice", test_new_empty_slice),
    ("series_tests::test_new_nullable_preserves_null_count", test_new_nullable_preserves_null_count),
    ("series_tests::test_new_strs_preserves_dtype_and_len", test_new_strs_preserves_dtype_and_len),
    ("series_tests::test_new_nullable_strs_preserves_null_count", test_new_nullable_strs_preserves_null_count),
    (
        "series_tests::test_new_nullable_strs_rejects_mismatched_validity_length",
        test_new_nullable_strs_rejects_mismatched_validity_length,
    ),
    ("series_tests::test_clone_shares_gpu_data", test_clone_shares_gpu_data),
    ("series_tests::test_from_column_round_trip", test_from_column_round_trip),
    ("series_tests::test_from_column_shares_data", test_from_column_shares_data),
    ("series_tests::test_view_matches_column_data", test_view_matches_column_data),
    ("series_tests::test_gt_scalar_i32", test_gt_scalar_i32),
    ("series_tests::test_lt_scalar_f64", test_lt_scalar_f64),
    ("series_tests::test_ne_scalar_string", test_ne_scalar_string),
    ("series_tests::test_gt_scalar_string", test_gt_scalar_string),
    ("series_tests::test_lt_scalar_string", test_lt_scalar_string),
    ("series_tests::test_ge_scalar_string", test_ge_scalar_string),
    ("series_tests::test_le_scalar_string", test_le_scalar_string),
    ("series_tests::test_gt_scalar_preserves_input", test_gt_scalar_preserves_input),
    ("series_tests::test_ne_scalar_i32", test_ne_scalar_i32),
    ("series_tests::test_ge_scalar_i32", test_ge_scalar_i32),
    ("series_tests::test_le_scalar_i32", test_le_scalar_i32),
    ("series_tests::test_eq_series", test_eq_series),
    ("series_tests::test_ne_series", test_ne_series),
    ("series_tests::test_gt_series", test_gt_series),
    ("series_tests::test_lt_series", test_lt_series),
    ("series_tests::test_ge_series", test_ge_series),
    ("series_tests::test_le_series", test_le_series),
    ("series_tests::test_comparison_mixed_types_f64_i32", test_comparison_mixed_types_f64_i32),
    ("series_tests::test_add_series", test_add_series),
    ("series_tests::test_sub_series", test_sub_series),
    ("series_tests::test_mul_series", test_mul_series),
    ("series_tests::test_div_series_f64", test_div_series_f64),
    ("series_tests::test_add_scalar_i32", test_add_scalar_i32),
    ("series_tests::test_sub_scalar_i32", test_sub_scalar_i32),
    ("series_tests::test_mul_scalar_f64", test_mul_scalar_f64),
    ("series_tests::test_div_scalar_f64", test_div_scalar_f64),
];
