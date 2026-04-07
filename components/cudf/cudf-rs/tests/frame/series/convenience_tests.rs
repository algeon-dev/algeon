//! Integration tests for `Series` convenience methods.
//!
//! Methods under test: `astype`, `fillna`, `fillna_with_series`, `isin`,
//! `unique`, `nunique`, `factorize`, `sort_values`, `is_null`, `is_not_null`.
//!
//! All tests require a GPU.

use cudf::{ErrorKind, Series, SortOptions, TypeId, core::Column};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
    sample_runtime,
};

// ===========================================================================
// is_null / is_not_null
// ===========================================================================

#[test]
fn test_is_null_all_valid() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let mask = s.is_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask.name(), Some("x"));
    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn test_is_null_with_nulls() {
    let s = Series::new_nullable::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let mask = s.is_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_is_null_empty() {
    let s = Series::new::<i32>("e", &[]).unwrap();
    let mask = s.is_null().unwrap();

    assert_eq!(mask.len(), 0);
    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
}

#[test]
fn test_is_not_null_all_valid() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let mask = s.is_not_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask.name(), Some("x"));
    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![true, true, true]);
}

#[test]
fn test_is_not_null_with_nulls() {
    let s = Series::new_nullable::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let mask = s.is_not_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn test_is_null_and_is_not_null_are_complementary() {
    let s = Series::new_nullable::<i32>("x", &[1, 0, 3, 0], &[true, false, true, false]).unwrap();
    let nulls = s.is_null().unwrap().to_vec::<bool>().unwrap();
    let valids = s.is_not_null().unwrap().to_vec::<bool>().unwrap();

    for (n, v) in nulls.iter().zip(valids.iter()) {
        assert_ne!(n, v, "is_null and is_not_null must be complementary");
    }
}

// ===========================================================================
// astype (cast)
// ===========================================================================

#[test]
fn test_astype_i32_to_f64() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let casted = s.astype(TypeId::Float64).unwrap();

    assert_eq!(casted.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(casted.len(), 3);
    assert_eq!(casted.name(), Some("x"));
    assert_eq!(casted.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_astype_f64_to_i32_truncates() {
    let s = Series::new::<f64>("x", &[1.9, 2.1, 3.7]).unwrap();
    let casted = s.astype(TypeId::Int32).unwrap();

    assert_eq!(casted.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(casted.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_astype_preserves_length() {
    let s = Series::new::<i32>("x", &[10, 20, 30, 40, 50]).unwrap();
    let casted = s.astype(TypeId::Int64).unwrap();

    assert_eq!(casted.len(), 5);
    assert_eq!(casted.dtype().unwrap().id(), TypeId::Int64);
}

#[test]
fn test_astype_preserves_nulls() {
    let s = Series::new_nullable::<i32>("x", &[1, 0, 3], &[true, false, true]).unwrap();
    let casted = s.astype(TypeId::Float64).unwrap();

    assert_eq!(casted.null_count(), 1);
    assert_eq!(casted.dtype().unwrap().id(), TypeId::Float64);
}

#[test]
fn test_astype_same_type_is_no_op_copy() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let casted = s.astype(TypeId::Int32).unwrap();

    assert_eq!(casted.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(casted.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// fillna (scalar)
// ===========================================================================

#[test]
fn test_fillna_replaces_nulls_with_scalar() {
    let s = Series::new_nullable::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let filled = s.fillna(99i32).unwrap();

    assert_eq!(filled.len(), 3);
    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.name(), Some("x"));
    assert_eq!(filled.to_vec::<i32>().unwrap(), vec![10, 99, 30]);
}

#[test]
fn test_fillna_no_nulls_returns_same_values() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let filled = s.fillna(0i32).unwrap();

    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_fillna_all_nulls() {
    let s = Series::new_nullable::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let filled = s.fillna(42i32).unwrap();

    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.to_vec::<i32>().unwrap(), vec![42, 42, 42]);
}

#[test]
fn test_fillna_f64() {
    let s = Series::new_nullable::<f64>("x", &[1.0, 0.0, 3.0], &[true, false, true]).unwrap();
    let filled = s.fillna(999.0f64).unwrap();

    assert_eq!(filled.to_vec::<f64>().unwrap(), vec![1.0, 999.0, 3.0]);
}

// ===========================================================================
// fillna_with_series (column fill)
// ===========================================================================

#[test]
fn test_fillna_with_series_replaces_nulls() {
    let s = Series::new_nullable::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let fill = Series::new::<i32>("fill", &[100, 200, 300]).unwrap();
    let filled = s.fillna_with_series(&fill).unwrap();

    assert_eq!(filled.len(), 3);
    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.name(), Some("x"));
    assert_eq!(filled.to_vec::<i32>().unwrap(), vec![10, 200, 30]);
}

#[test]
fn test_fillna_with_series_no_nulls() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let fill = Series::new::<i32>("fill", &[100, 200, 300]).unwrap();
    let filled = s.fillna_with_series(&fill).unwrap();

    assert_eq!(filled.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// isin
// ===========================================================================

#[test]
fn test_isin_basic() {
    let s = Series::new::<i32>("x", &[1, 2, 3, 4, 5]).unwrap();
    let values = Series::new::<i32>("vals", &[2, 4]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 5);
    assert_eq!(mask.name(), Some("x"));
    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![false, true, false, true, false]);
}

#[test]
fn test_isin_no_matches() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let values = Series::new::<i32>("vals", &[10, 20]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn test_isin_all_match() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let values = Series::new::<i32>("vals", &[3, 1, 2]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![true, true, true]);
}

#[test]
fn test_isin_strings() {
    let s = Series::from_strings("x", &["apple", "banana", "cherry"]).unwrap();
    let values = Series::from_strings("vals", &["banana", "date"]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_isin_succeeds_with_mixed_domains_and_uses_source_provenance() {
    let source_rt = sample_runtime();
    let values_rt = sample_runtime();
    let s = Series::new_with_runtime(&source_rt, "x", &[1i32, 2, 3, 4, 5]).unwrap();
    let values = Series::new_with_runtime(&values_rt, "vals", &[2i32, 4]).unwrap();
    let expected_exec_stream = s.as_column().__debug_execution_domain_stream_value();
    let expected_exec_mr = s.as_column().__debug_execution_domain_mr_value();
    let expected_sync_stream = values.as_column().__debug_execution_domain_stream_value();

    reset_exec_observation();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.as_column().__debug_execution_domain_id(), s.as_column().__debug_execution_domain_id());
    assert_eq!(mask.as_column().__debug_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_sync_stream);

    reset_sync_observation();
    assert_eq!(mask.to_vec::<bool>().unwrap(), vec![false, true, false, true, false]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ===========================================================================
// unique
// ===========================================================================

#[test]
fn test_unique_removes_duplicates() {
    let s = Series::new::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();
    let uniq = s.unique().unwrap();

    assert_eq!(uniq.name(), Some("x"));
    assert_eq!(uniq.dtype().unwrap().id(), TypeId::Int32);

    let mut vals = uniq.to_vec::<i32>().unwrap();
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_unique_already_unique() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let uniq = s.unique().unwrap();

    let mut vals = uniq.to_vec::<i32>().unwrap();
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_unique_empty() {
    let s = Series::new::<i32>("x", &[]).unwrap();
    let uniq = s.unique().unwrap();

    assert_eq!(uniq.len(), 0);
}

#[test]
fn test_unique_strings() {
    let s = Series::from_strings("x", &["a", "b", "a", "c", "b"]).unwrap();
    let uniq = s.unique().unwrap();

    assert_eq!(uniq.dtype().unwrap().id(), TypeId::String);

    let mut vals = uniq.to_strings().unwrap();
    vals.sort();
    assert_eq!(vals, vec!["a", "b", "c"]);
}

// ===========================================================================
// nunique
// ===========================================================================

#[test]
fn test_nunique_basic() {
    let s = Series::new::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();
    assert_eq!(s.nunique().unwrap(), 3);
}

#[test]
fn test_nunique_all_same() {
    let s = Series::new::<i32>("x", &[5, 5, 5]).unwrap();
    assert_eq!(s.nunique().unwrap(), 1);
}

#[test]
fn test_nunique_empty() {
    let s = Series::new::<i32>("x", &[]).unwrap();
    assert_eq!(s.nunique().unwrap(), 0);
}

#[test]
fn test_nunique_all_unique() {
    let s = Series::new::<i32>("x", &[10, 20, 30, 40]).unwrap();
    assert_eq!(s.nunique().unwrap(), 4);
}

// ===========================================================================
// factorize
// ===========================================================================

#[test]
fn test_factorize_basic() {
    let s = Series::new::<i32>("x", &[10, 20, 10, 30, 20]).unwrap();
    let (codes, uniques) = s.factorize().unwrap();

    assert_eq!(codes.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(codes.len(), 5);
    assert_eq!(codes.name(), Some("x_codes"));
    assert_eq!(uniques.name(), Some("x_uniques"));

    // Verify round-trip: same input values get same codes.
    let code_vals = codes.to_vec::<i32>().unwrap();
    // codes[0] == codes[2] (both map to 10)
    assert_eq!(code_vals[0], code_vals[2]);
    // codes[1] == codes[4] (both map to 20)
    assert_eq!(code_vals[1], code_vals[4]);
    // codes[0] != codes[1] (10 != 20)
    assert_ne!(code_vals[0], code_vals[1]);
    // codes[0] != codes[3] (10 != 30)
    assert_ne!(code_vals[0], code_vals[3]);

    // Uniques should contain exactly {10, 20, 30}.
    let mut uniq_vals = uniques.to_vec::<i32>().unwrap();
    uniq_vals.sort();
    assert_eq!(uniq_vals, vec![10, 20, 30]);
}

#[test]
fn test_factorize_strings() {
    let s = Series::from_strings("x", &["a", "b", "a", "c"]).unwrap();
    let (codes, uniques) = s.factorize().unwrap();

    assert_eq!(codes.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(codes.len(), 4);
    assert_eq!(codes.name(), Some("x_codes"));
    assert_eq!(uniques.name(), Some("x_uniques"));

    let code_vals = codes.to_vec::<i32>().unwrap();
    // codes[0] == codes[2] (both map to "a")
    assert_eq!(code_vals[0], code_vals[2]);

    let mut uniq_vals = uniques.to_strings().unwrap();
    uniq_vals.sort();
    assert_eq!(uniq_vals, vec!["a", "b", "c"]);
}

#[test]
fn test_factorize_unnamed_input_keeps_outputs_unnamed() {
    let column = Column::from_slice(&[10i32, 20, 10, 30]).unwrap();
    let s = Series::from_column(column).unwrap();
    let (codes, uniques) = s.factorize().unwrap();

    assert_eq!(codes.name(), None);
    assert_eq!(uniques.name(), None);
}

// ===========================================================================
// sort_values
// ===========================================================================

#[test]
fn test_sort_values_ascending() {
    let s = Series::new::<i32>("x", &[30, 10, 20]).unwrap();
    let sorted = s.sort_values(SortOptions::new()).unwrap();

    assert_eq!(sorted.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted.name(), Some("x"));
    assert_eq!(sorted.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_sort_values_descending() {
    let s = Series::new::<i32>("x", &[30, 10, 20]).unwrap();
    let sorted = s.sort_values(SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(sorted.to_vec::<i32>().unwrap(), vec![30, 20, 10]);
}

#[test]
fn test_sort_values_stable() {
    let s = Series::new::<i32>("x", &[3, 1, 2]).unwrap();
    let sorted = s.sort_values(SortOptions::new().with_stable(true)).unwrap();

    assert_eq!(sorted.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_sort_values_f64() {
    let s = Series::new::<f64>("x", &[3.3, 1.1, 2.2]).unwrap();
    let sorted = s.sort_values(SortOptions::new()).unwrap();

    assert_eq!(sorted.to_vec::<f64>().unwrap(), vec![1.1, 2.2, 3.3]);
}

#[test]
fn test_sort_values_empty() {
    let s = Series::new::<i32>("x", &[]).unwrap();
    let sorted = s.sort_values(SortOptions::new()).unwrap();

    assert_eq!(sorted.len(), 0);
}

#[test]
fn test_sort_values_with_nulls() {
    let s = Series::new_nullable::<i32>("x", &[30, 0, 10], &[true, false, true]).unwrap();
    // Default: ascending, nulls last
    let sorted = s.sort_values(SortOptions::new()).unwrap();
    let view = sorted.as_column_view().unwrap();

    assert_eq!(sorted.len(), 3);
    // The valid values should sort ascending, with the null row moved to the end.
    let vals = view.to_vec::<i32>().unwrap();
    assert_eq!(vals[0], 10);
    assert_eq!(vals[1], 30);
    assert_eq!(view.validity().unwrap().unwrap(), vec![true, true, false]);
    assert_eq!(sorted.null_count(), 1);
}

#[test]
fn test_sort_values_nulls_first() {
    let s = Series::new_nullable::<i32>("x", &[30, 0, 10], &[true, false, true]).unwrap();
    let sorted = s.sort_values(SortOptions::new().with_nulls_first(vec![true])).unwrap();
    let view = sorted.as_column_view().unwrap();

    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted.null_count(), 1);
    // Null should be first, then 10, 30
    let vals = view.to_vec::<i32>().unwrap();
    assert_eq!(view.validity().unwrap().unwrap(), vec![false, true, true]);
    assert_eq!(vals[1], 10);
    assert_eq!(vals[2], 30);
}

#[test]
fn test_sort_values_rejects_wrong_descending_len() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    // Single column but 2 descending flags — should fail
    let result = s.sort_values(SortOptions::new().with_descending(vec![true, false]));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidOperation);
}

// ===========================================================================
// input preservation
// ===========================================================================

#[test]
fn test_convenience_ops_preserve_input() {
    let s = Series::new_nullable::<i32>("x", &[3, 0, 1], &[true, false, true]).unwrap();

    // Run several ops and verify the original is unchanged.
    let _null_mask = s.is_null().unwrap();
    let _not_null_mask = s.is_not_null().unwrap();
    let _filled = s.fillna(0i32).unwrap();
    let _sorted = s.sort_values(SortOptions::new()).unwrap();

    assert_eq!(s.name(), Some("x"));
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}
