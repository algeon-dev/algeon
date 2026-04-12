//! Specification tests: edge cases for Series and DataFrame operations.
//!
//! These tests lock down behavior for:
//! - Empty Series through various operations
//! - Single-element Series behavior
//! - Type coercion edge cases in astype/cast
//! - Null-aware convenience ops (unique, isin, factorize, fillna with nulls)
//!
//! All tests require a GPU.

use cudf::{DataType, DropNaOptions, TypeId};

// ===========================================================================
// Empty Series through arithmetic
// ===========================================================================

#[test]
fn test_add_empty_series() {
    let a = crate::common::series::<i32>("a", &[]).unwrap();
    let b = crate::common::series::<i32>("b", &[]).unwrap();
    let result = a.add(&b).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_eq_empty_series() {
    let a = crate::common::series::<i32>("a", &[]).unwrap();
    let b = crate::common::series::<i32>("b", &[]).unwrap();
    let result = a.eq(&b).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_add_scalar_empty_series() {
    let a = crate::common::series::<i32>("a", &[]).unwrap();
    let result = a.add_scalar(5i32).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_neg_empty_series() {
    let a = crate::common::series::<i32>("a", &[]).unwrap();
    let result = (-&a).unwrap();
    assert!(result.is_empty());
}

// ===========================================================================
// Empty Series through reductions
// ===========================================================================

#[test]
fn test_min_empty_series() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.min().unwrap();
    // min of empty = null scalar (no elements to pick)
    assert!(!result.is_valid().unwrap());
}

#[test]
fn test_max_empty_series() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.max().unwrap();
    assert!(!result.is_valid().unwrap());
}

#[test]
fn test_mean_empty_series() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.mean().unwrap();
    // mean of empty = null (no elements to average)
    assert!(!result.is_valid().unwrap());
}

// ===========================================================================
// Single-element edge cases
// ===========================================================================

#[test]
fn test_sort_single_element() {
    use cudf::SortOptions;
    let s = crate::common::series::<i32>("x", &[42]).unwrap();
    let result = s.sort_values(SortOptions::default()).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42]);
}

#[test]
fn test_unique_single_element() {
    let s = crate::common::series::<i32>("x", &[42]).unwrap();
    let result = s.unique().unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42]);
}

#[test]
fn test_nunique_single_element() {
    let s = crate::common::series::<i32>("x", &[42]).unwrap();
    assert_eq!(s.nunique().unwrap(), 1);
}

#[test]
fn test_factorize_single_element() {
    let s = crate::common::series::<i32>("x", &[42]).unwrap();
    let (codes, uniques) = s.factorize().unwrap();
    assert_eq!(codes.len(), 1);
    assert_eq!(uniques.len(), 1);
}

// ===========================================================================
// Type coercion edge cases in astype
// ===========================================================================

#[test]
fn test_astype_bool_to_int() {
    // true -> 1, false -> 0
    let s = crate::common::series::<bool>("flags", &[true, false, true]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 0, 1]);
}

#[test]
fn test_astype_int_to_bool() {
    // nonzero -> true, zero -> false
    let s = crate::common::series::<i32>("x", &[0, 1, 42, -1]).unwrap();
    let result = s.astype(TypeId::Bool8).unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![false, true, true, true]);
}

#[test]
fn test_astype_i32_to_i64_widening() {
    let s = crate::common::series::<i32>("x", &[i32::MAX, i32::MIN, 0]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![i32::MAX as i64, i32::MIN as i64, 0]);
}

#[test]
fn test_astype_f64_to_i32_truncates() {
    let s = crate::common::series::<f64>("x", &[1.9, -2.7, 0.1]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    // GPU truncation: 1.9 -> 1, -2.7 -> -2, 0.1 -> 0
    assert_eq!(vals, vec![1, -2, 0]);
}

#[test]
fn test_astype_preserves_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(10.0), None, Some(30.0)]);
}

#[test]
fn test_astype_empty_series() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.astype(TypeId::Float64).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
}

#[test]
fn test_astype_same_type_noop() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_astype_unsigned_to_signed() {
    let s = crate::common::series::<u32>("x", &[0, 100, 255]).unwrap();
    let result = s.astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![0, 100, 255]);
}

#[test]
fn test_astype_i8_to_i64() {
    let s = crate::common::series::<i8>("x", &[i8::MAX, i8::MIN, 0]).unwrap();
    let result = s.astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![i8::MAX as i64, i8::MIN as i64, 0]);
}

// ===========================================================================
// Null-aware convenience operations
// ===========================================================================

#[test]
fn test_unique_with_nulls() {
    // [1, null, 2, null, 1] -> unique should keep {1, 2, null}
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 2, 0, 1], &[true, false, true, false, true]).unwrap();
    let result = s.unique().unwrap();
    // Should have 3 unique values: 1, 2, and null
    // Order is unspecified, but count should be 3
    assert_eq!(result.len(), 3);
    assert_eq!(result.null_count(), 1); // one null among the uniques
}

#[test]
fn test_nunique_with_nulls() {
    // [1, null, 2, null, 1] -> nunique excludes nulls -> 2 (just 1 and 2)
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 2, 0, 1], &[true, false, true, false, true]).unwrap();
    assert_eq!(s.nunique().unwrap(), 2);
}

#[test]
fn test_isin_with_null_in_source() {
    // source: [10, null, 30], values: [10, 30]
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let vals = crate::common::series::<i32>("vals", &[10, 30]).unwrap();
    let result = s.isin(&vals).unwrap();
    let rv = result.to_optional_vec::<bool>().unwrap();
    // 10 is in vals -> true, null isin anything -> null, 30 is in vals -> true
    assert_eq!(rv[0], Some(true));
    assert_eq!(rv[1], None);
    assert_eq!(rv[2], Some(true));
}

#[test]
fn test_fillna_with_series_both_have_nulls() {
    // source: [10, null, null], fill: [0, 20, null]
    // Expected: [10, 20, null]  (null in fill at position 2 means it stays null)
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 0], &[true, false, false]).unwrap();
    let fill = crate::common::nullable_series::<i32>("f", &[0, 20, 0], &[true, true, false]).unwrap();
    let result = s.fillna_with_series(&fill).unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(10), Some(20), None]);
}

#[test]
fn test_fillna_empty_series() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let result = s.fillna(0i32).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_factorize_with_nulls() {
    // [1, null, 2, null, 1] -> codes should map nulls consistently
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 2, 0, 1], &[true, false, true, false, true]).unwrap();
    let (codes, uniques) = s.factorize().unwrap();
    assert_eq!(codes.len(), 5);
    // The codes for valid values of 1 should be the same
    let code_vals = codes.to_optional_vec::<i32>().unwrap();
    // Positions 0 and 4 both have value 1, so their codes should match
    assert_eq!(code_vals[0], code_vals[4]);
    // Positions 1 and 3 are null
    assert_eq!(code_vals[1], code_vals[3]);
    // Codes for value 1 and value 2 should differ
    assert_ne!(code_vals[0], code_vals[2]);
    // uniques should include distinct valid values
    assert!(uniques.len() >= 2);
}

// ===========================================================================
// Sort with all nulls
// ===========================================================================

#[test]
fn test_sort_all_null() {
    use cudf::SortOptions;
    let s = crate::common::nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let result = s.sort_values(SortOptions::default()).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.null_count(), 3);
}

// ===========================================================================
// DataFrame empty edge cases
// ===========================================================================

#[test]
fn test_dataframe_filter_empty_result() {
    use cudf::DataFrame;
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series(vec![a]).unwrap();
    // Filter with all-false mask
    let mask = crate::common::series::<bool>("mask", &[false, false, false]).unwrap();
    let result = df.filter(&mask).unwrap();
    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.num_columns(), 1);
}

#[test]
fn test_dataframe_head_zero() {
    use cudf::DataFrame;
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series(vec![a]).unwrap();
    let result = df.head(0).unwrap();
    assert_eq!(result.num_rows(), 0);
}

#[test]
fn test_dataframe_slice_empty() {
    use cudf::DataFrame;
    let a = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series(vec![a]).unwrap();
    let result = df.slice(1, 0).unwrap();
    assert_eq!(result.num_rows(), 0);
}

// ===========================================================================
// Null-aware DataFrame operations
// ===========================================================================

#[test]
fn test_dataframe_dropna_with_mixed_nulls() {
    use cudf::DataFrame;
    // col a: [1, null, 3], col b: [null, 5, 6]
    let a = crate::common::nullable_series::<i32>("a", &[1, 0, 3], &[true, false, true]).unwrap();
    let b = crate::common::nullable_series::<i32>("b", &[0, 5, 6], &[false, true, true]).unwrap();
    let df = DataFrame::from_series(vec![a, b]).unwrap();
    let result = df.dropna(DropNaOptions::builder().subset(&["a", "b"]).build()).unwrap();
    // Only row 2 (index 2) has no nulls in either column
    assert_eq!(result.num_rows(), 1);
    let col_a = result.series("a").unwrap();
    assert_eq!(col_a.to_optional_vec::<i32>().unwrap(), vec![Some(3)]);
}

#[test]
fn test_dataframe_cast_column_preserves_nulls() {
    use cudf::DataFrame;
    let a = crate::common::nullable_series::<i32>("a", &[10, 0, 30], &[true, false, true]).unwrap();
    let df = DataFrame::from_series(vec![a]).unwrap();
    let result = df.cast_column("a", DataType::new(TypeId::Float64)).unwrap();
    let col = result.series("a").unwrap();
    let vals = col.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(10.0), None, Some(30.0)]);
}
