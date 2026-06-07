//! Tests for CU-4: replace_nulls (fillna), forward fill, and backward fill.

use cudf::{
    Series,
    core::{Column, Scalar},
    ops::replace::{self, FillPolicy},
};

#[test]
fn test_replace_nulls_with_scalar_i32() {
    // Column with nulls at positions 1 and 3.
    let col = Column::from_slice_nullable(&[10i32, 0, 30, 0], &[true, false, true, false]).unwrap();
    let replacement = Scalar::from_value(99i32).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls(&view, &replacement).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![10, 99, 30, 99]);
    assert_eq!(rv.null_count().unwrap(), 0);
}

#[test]
fn test_replace_nulls_no_nulls() {
    // Column without nulls — output should be identical.
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let replacement = Scalar::from_value(0i32).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls(&view, &replacement).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(rv.null_count().unwrap(), 0);
}

#[test]
fn test_replace_nulls_all_null() {
    // All-null column — all values should be replaced.
    let col = Column::from_slice_nullable(&[0i32, 0, 0], &[false, false, false]).unwrap();
    let replacement = Scalar::from_value(42i32).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls(&view, &replacement).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![42, 42, 42]);
    assert_eq!(rv.null_count().unwrap(), 0);
}

#[test]
fn test_replace_nulls_float32() {
    let col = Column::from_slice_nullable(&[1.0f32, 0.0, 3.0], &[true, false, true]).unwrap();
    let replacement = Scalar::from_value(0.0f32).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls(&view, &replacement).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<f32>().unwrap(), vec![1.0, 0.0, 3.0]);
}

#[test]
fn test_replace_nulls_empty_column() {
    let col = Column::from_slice(&[] as &[i32]).unwrap();
    let replacement = Scalar::from_value(0i32).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls(&view, &replacement).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 0);
}

// -- Column-to-column replace_nulls tests ------------------------------------

#[test]
fn test_replace_nulls_column_basic() {
    let left = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap();
    let right = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let lv = left.view().unwrap();
    let rv = right.view().unwrap();
    let result = replace::replace_nulls_column(&lv, &rv).unwrap();
    let out = result.view().unwrap();
    assert_eq!(out.to_vec::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(out.null_count().unwrap(), 0);
}

#[test]
fn test_replace_nulls_column_both_null() {
    let left = Column::from_slice_nullable(&[0i32], &[false]).unwrap();
    let right = Column::from_slice_nullable(&[0i32], &[false]).unwrap();
    let lv = left.view().unwrap();
    let rv = right.view().unwrap();
    let result = replace::replace_nulls_column(&lv, &rv).unwrap();
    let out = result.view().unwrap();
    assert_eq!(out.null_count().unwrap(), 1);
}

// -- Forward fill (ffill) tests -----------------------------------------------

#[test]
fn test_replace_nulls_policy_ffill_basic() {
    // [1, null, null, 4] → [1, 1, 1, 4]
    let col = Column::from_slice_nullable(&[1i32, 0, 0, 4], &[true, false, false, true]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Preceding).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![1, 1, 1, 4]);
    assert_eq!(rv.null_count().unwrap(), 0);
}

#[test]
fn test_replace_nulls_policy_ffill_leading_nulls_remain() {
    // [null, null, 3, null, 5] → [null, null, 3, 3, 5]
    let col = Column::from_slice_nullable(&[0i32, 0, 3, 0, 5], &[false, false, true, false, true]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Preceding).unwrap();
    let s = Series::from_column(result).unwrap();
    assert_eq!(s.null_count(), 2);
    let vals = s.to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![None, None, Some(3), Some(3), Some(5)]);
}

#[test]
fn test_replace_nulls_policy_ffill_no_nulls() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Preceding).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_replace_nulls_policy_ffill_all_null() {
    let col = Column::from_slice_nullable(&[0i32, 0, 0], &[false, false, false]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Preceding).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.null_count().unwrap(), 3);
}

#[test]
fn test_replace_nulls_policy_ffill_empty() {
    let col = Column::from_slice(&[] as &[i32]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Preceding).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 0);
}

// -- Backward fill (bfill) tests ----------------------------------------------

#[test]
fn test_replace_nulls_policy_bfill_basic() {
    // [1, null, null, 4] → [1, 4, 4, 4]
    let col = Column::from_slice_nullable(&[1i32, 0, 0, 4], &[true, false, false, true]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Following).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![1, 4, 4, 4]);
    assert_eq!(rv.null_count().unwrap(), 0);
}

#[test]
fn test_replace_nulls_policy_bfill_trailing_nulls_remain() {
    // [1, null, 3, null, null] → [1, 3, 3, null, null]
    let col = Column::from_slice_nullable(&[1i32, 0, 3, 0, 0], &[true, false, true, false, false]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Following).unwrap();
    let s = Series::from_column(result).unwrap();
    assert_eq!(s.null_count(), 2);
    let vals = s.to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), Some(3), Some(3), None, None]);
}

#[test]
fn test_replace_nulls_policy_bfill_all_null() {
    let col = Column::from_slice_nullable(&[0i32, 0], &[false, false]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Following).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.null_count().unwrap(), 2);
}

#[test]
fn test_replace_nulls_policy_float64() {
    // [1.0, null, 3.0] → ffill → [1.0, 1.0, 3.0]
    let col = Column::from_slice_nullable(&[1.0f64, 0.0, 3.0], &[true, false, true]).unwrap();
    let view = col.view().unwrap();
    let result = replace::replace_nulls_policy(&view, FillPolicy::Preceding).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 3.0]);
}
