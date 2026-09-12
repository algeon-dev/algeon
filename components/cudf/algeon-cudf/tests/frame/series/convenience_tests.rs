//! Integration tests for `Series` convenience methods.
//!
//! Methods under test: `astype`, `fillna`, `fillna_with_series`, `isin`,
//! `unique`, `nunique`, `factorize`, `sort_values`, `is_null`, `is_not_null`,
//! `log`.
//!
//! All tests require a GPU.
use cudf::{
    DataType, ErrorCode, ErrorKind, ExecutionOverrides, Series, SortOptions, TypeId,
    core::error::{FactKeyName, FactValue},
    frame::factorization::{Codebook, DictionaryEncodingOptions, DictionaryKeyOrder},
    testing::prelude::*,
};

use crate::common::{
    column_from_slice, error_context_value, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    reset_exec_observation, reset_sync_observation, sample_runtime,
};

// ===========================================================================
// is_null / is_not_null
// ===========================================================================

fn test_is_null_all_valid() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let mask = s.is_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, false]);
}

fn test_is_null_with_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let mask = s.is_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_is_null_empty() {
    let s = crate::common::series::<i32>("e", &[]).unwrap();
    let mask = s.is_null().unwrap();

    assert_eq!(mask.len(), 0);
    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
}

fn test_is_not_null_all_valid() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let mask = s.is_not_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
}

fn test_is_not_null_with_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let mask = s.is_not_null().unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn test_is_null_and_is_not_null_are_complementary() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 3, 0], &[true, false, true, false]).unwrap();
    let nulls = s.is_null().unwrap().to_vec_strict::<bool>().unwrap();
    let valids = s.is_not_null().unwrap().to_vec_strict::<bool>().unwrap();

    for (n, v) in nulls.iter().zip(valids.iter()) {
        assert_ne!(n, v, "is_null and is_not_null must be complementary");
    }
}

// ===========================================================================
// astype (cast)
// ===========================================================================

fn test_astype_i32_to_f64() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let casted = s.astype(TypeId::Float64).unwrap();

    assert_eq!(casted.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(casted.len(), 3);
    assert_eq!(casted.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
}

fn test_astype_f64_to_i32_truncates() {
    let s = crate::common::series::<f64>("x", &[1.9, 2.1, 3.7]).unwrap();
    let casted = s.astype(TypeId::Int32).unwrap();

    assert_eq!(casted.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(casted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_astype_preserves_length() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30, 40, 50]).unwrap();
    let casted = s.astype(TypeId::Int64).unwrap();

    assert_eq!(casted.len(), 5);
    assert_eq!(casted.dtype().unwrap().id(), TypeId::Int64);
}

fn test_astype_preserves_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[1, 0, 3], &[true, false, true]).unwrap();
    let casted = s.astype(TypeId::Float64).unwrap();

    assert_eq!(casted.null_count(), 1);
    assert_eq!(casted.dtype().unwrap().id(), TypeId::Float64);
}

fn test_astype_same_type_is_no_op_copy() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let casted = s.astype(TypeId::Int32).unwrap();

    assert_eq!(casted.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(casted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_log_f64() {
    let s = crate::common::series::<f64>("x", &[1.0, std::f64::consts::E, std::f64::consts::E.powi(2)]).unwrap();
    let logged = s.log().unwrap();
    let values = logged.to_vec_strict::<f64>().unwrap();

    assert_eq!(logged.dtype().unwrap().id(), TypeId::Float64);
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[1] - 1.0).abs() < 1e-10);
    assert!((values[2] - 2.0).abs() < 1e-10);
}

// ===========================================================================
// fillna (scalar)
// ===========================================================================

fn test_fillna_replaces_nulls_with_scalar() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let filled = s.fillna(99i32).unwrap();

    assert_eq!(filled.len(), 3);
    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![10, 99, 30]);
}

fn test_fillna_no_nulls_returns_same_values() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let filled = s.fillna(0i32).unwrap();

    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_fillna_all_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[0, 0, 0], &[false, false, false]).unwrap();
    let filled = s.fillna(42i32).unwrap();

    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42]);
}

fn test_fillna_f64() {
    let s = crate::common::nullable_series::<f64>("x", &[1.0, 0.0, 3.0], &[true, false, true]).unwrap();
    let filled = s.fillna(999.0f64).unwrap();

    assert_eq!(filled.to_vec_strict::<f64>().unwrap(), vec![1.0, 999.0, 3.0]);
}

// ===========================================================================
// fillna_with_series (column fill)
// ===========================================================================

fn test_fillna_with_series_replaces_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let fill = crate::common::series::<i32>("fill", &[100, 200, 300]).unwrap();
    let filled = s.fillna_with_series(&fill).unwrap();

    assert_eq!(filled.len(), 3);
    assert_eq!(filled.null_count(), 0);
    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![10, 200, 30]);
}

fn test_fillna_with_series_no_nulls() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let fill = crate::common::series::<i32>("fill", &[100, 200, 300]).unwrap();
    let filled = s.fillna_with_series(&fill).unwrap();

    assert_eq!(filled.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

// ===========================================================================
// isin
// ===========================================================================

fn test_isin_basic() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3, 4, 5]).unwrap();
    let values = crate::common::series::<i32>("vals", &[2, 4]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.len(), 5);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false, true, false]);
}

fn test_isin_no_matches() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let values = crate::common::series::<i32>("vals", &[10, 20]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, false]);
}

fn test_isin_all_match() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let values = crate::common::series::<i32>("vals", &[3, 1, 2]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
}

fn test_isin_strings() {
    let s = crate::common::series("x", &["apple", "banana", "cherry"]).unwrap();
    let values = crate::common::series("vals", &["banana", "date"]).unwrap();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn test_isin_type_mismatch_uses_canonical_error_contract() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let values = crate::common::series::<f64>("vals", &[1.0, 2.0]).unwrap();

    let err = s.isin(&values).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "series::isin");
    assert_eq!(err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Float64)));
}

fn test_isin_succeeds_with_mixed_domains_and_uses_source_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let source_rt = sample_runtime();
    let values_rt = sample_runtime();
    let s = source_rt.series(&[1i32, 2, 3, 4, 5]).unwrap();
    let values = values_rt.series(&[2i32, 4]).unwrap();
    let expected_exec_stream = s.as_column().testing_execution_domain_stream_value();
    let expected_exec_mr = s.as_column().testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let mask = s.isin(&values).unwrap();

    assert_eq!(mask.as_column().testing_execution_domain_id(), s.as_column().testing_execution_domain_id());
    assert_eq!(mask.as_column().testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false, true, false]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ===========================================================================
// unique
// ===========================================================================

fn test_unique_removes_duplicates() {
    let s = crate::common::series::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();
    let uniq = s.unique().unwrap();
    assert_eq!(uniq.dtype().unwrap().id(), TypeId::Int32);

    let mut vals = uniq.to_vec_strict::<i32>().unwrap();
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

fn test_unique_already_unique() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    let uniq = s.unique().unwrap();

    let mut vals = uniq.to_vec_strict::<i32>().unwrap();
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

fn test_unique_empty() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let uniq = s.unique().unwrap();

    assert_eq!(uniq.len(), 0);
}

fn test_unique_strings() {
    let s = crate::common::series("x", &["a", "b", "a", "c", "b"]).unwrap();
    let uniq = s.unique().unwrap();

    assert_eq!(uniq.dtype().unwrap().id(), TypeId::String);

    let mut vals = uniq.to_strings_strict().unwrap();
    vals.sort();
    assert_eq!(vals, vec!["a", "b", "c"]);
}

// ===========================================================================
// nunique
// ===========================================================================

fn test_nunique_basic() {
    let s = crate::common::series::<i32>("x", &[3, 1, 2, 1, 3]).unwrap();
    assert_eq!(s.nunique().unwrap(), 3);
}

fn test_nunique_all_same() {
    let s = crate::common::series::<i32>("x", &[5, 5, 5]).unwrap();
    assert_eq!(s.nunique().unwrap(), 1);
}

fn test_nunique_empty() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    assert_eq!(s.nunique().unwrap(), 0);
}

fn test_nunique_all_unique() {
    let s = crate::common::series::<i32>("x", &[10, 20, 30, 40]).unwrap();
    assert_eq!(s.nunique().unwrap(), 4);
}

fn test_nunique_negative_distinct_count_uses_stable_foreign_contract() {
    let err = cudf::testing::series::trigger_nunique_negative_distinct_count_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "series::nunique");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("distinct_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("negative_count"));
    assert_eq!(error_context_value(&err, "distinct_count").as_deref(), Some("-1"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign reduction backend returns a non-negative distinct count")
    );
}

fn test_nunique_counts_nan_as_distinct_non_null_value() {
    let s = crate::common::series::<f64>("x", &[1.0, f64::NAN, 1.0, f64::NAN, 2.0]).unwrap();
    assert_eq!(s.nunique().unwrap(), 3);
}

fn test_nunique_excludes_null_but_keeps_nan_distinctness() {
    let s = crate::common::nullable_series::<f64>(
        "x",
        &[1.0, f64::NAN, 0.0, f64::NAN, 1.0],
        &[true, true, false, true, true],
    )
    .unwrap();
    assert_eq!(s.nunique().unwrap(), 2);
}

// ===========================================================================
// factorize
// ===========================================================================

fn test_factorize_basic() {
    let s = crate::common::series::<i32>("x", &[10, 20, 10, 30, 20]).unwrap();
    let factorized = s.factorize().unwrap();
    let codes = factorized.row_codes();
    let keys = factorized.codebook().keys();

    assert_eq!(codes.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(codes.len(), 5);

    // Verify round-trip: same input values get same codes.
    let code_vals = codes.to_vec_strict::<i32>().unwrap();
    // codes[0] == codes[2] (both map to 10)
    assert_eq!(code_vals[0], code_vals[2]);
    // codes[1] == codes[4] (both map to 20)
    assert_eq!(code_vals[1], code_vals[4]);
    // codes[0] != codes[1] (10 != 20)
    assert_ne!(code_vals[0], code_vals[1]);
    // codes[0] != codes[3] (10 != 30)
    assert_ne!(code_vals[0], code_vals[3]);

    // Uniques should contain exactly {10, 20, 30}.
    let mut uniq_vals = keys.to_vec_strict::<i32>().unwrap();
    uniq_vals.sort();
    assert_eq!(uniq_vals, vec![10, 20, 30]);

    let decoded = factorized.codebook().decode(factorized.row_codes()).unwrap();
    assert_eq!(decoded.to_vec_strict::<i32>().unwrap(), vec![10, 20, 10, 30, 20]);
}

fn test_factorize_strings() {
    let s = crate::common::series("x", &["a", "b", "a", "c"]).unwrap();
    let factorized = s.factorize().unwrap();
    let codes = factorized.row_codes();
    let keys = factorized.codebook().keys();

    assert_eq!(codes.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(codes.len(), 4);

    let code_vals = codes.to_vec_strict::<i32>().unwrap();
    // codes[0] == codes[2] (both map to "a")
    assert_eq!(code_vals[0], code_vals[2]);

    let mut uniq_vals = keys.to_strings_strict().unwrap();
    uniq_vals.sort();
    assert_eq!(uniq_vals, vec!["a", "b", "c"]);
}

fn test_factorize_unnamed_input_keeps_outputs_unnamed() {
    let column = column_from_slice(&[10i32, 20, 10, 30]).unwrap();
    let s = Series::from_column(column).unwrap();
    let factorized = s.factorize().unwrap();
    let codes = factorized.row_codes();
    let keys = factorized.codebook().keys();
    let codes_output = format!("{}", codes.as_series());
    let keys_output = format!("{:?}", keys);

    assert!(
        !codes_output.contains("name="),
        "factorized row codes should remain unnamed in display output: {}",
        codes_output
    );
    assert!(
        !keys_output.contains("name"),
        "factorization codebook keys should remain unnamed in debug output: {}",
        keys_output
    );
}

fn test_codebook_can_encode_and_decode_new_queries() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();
    let codebook = s.codebook().unwrap();
    let query = crate::common::series("query", &["c", "a"]).unwrap();

    let encoded = codebook.encode(&query).unwrap();
    assert_eq!(encoded.to_vec_strict::<i32>().unwrap().len(), 2);

    let decoded = codebook.decode(&encoded).unwrap();
    assert_eq!(decoded.to_strings_strict().unwrap(), vec!["c", "a"]);
}

fn test_codebook_encode_rejects_dtype_mismatch_with_codebook_operation() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();
    let codebook = s.codebook().unwrap();
    let query = crate::common::series("query", &[1i32, 2]).unwrap();

    let err = match codebook.encode(&query) {
        Ok(_) => panic!("expected codebook encode dtype mismatch to be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), cudf::ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "codebook::encode");
    assert_eq!(err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::Int32)));
}

fn test_codebook_encode_returns_codes() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();
    let codebook = s.codebook().unwrap();
    let query = crate::common::series("query", &["c", "a"]).unwrap();

    let encoded = codebook.encode(&query).unwrap();
    assert_eq!(encoded.to_vec_strict::<i32>().unwrap().len(), 2);
}

fn test_codebook_decode_returns_decoded_values() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();
    let codebook = s.codebook().unwrap();
    let query = crate::common::series("query", &["c", "a"]).unwrap();
    let encoded = codebook.encode(&query).unwrap();

    let decoded = codebook.decode(&encoded).unwrap();
    assert_eq!(decoded.to_strings_strict().unwrap(), vec!["c", "a"]);
}

fn test_codebook_decode_preserves_null_codes_as_null_values() {
    let keys = crate::common::series("key", &["alpha", "beta"]).unwrap();
    let codebook = Codebook::from_keys(keys).unwrap();
    let codes = crate::common::nullable_series::<i32>("code", &[1, 0, 0, 1], &[true, false, true, false]).unwrap();

    let decoded = codebook.decode_series(&codes).unwrap();

    assert_eq!(
        decoded.to_optional_strings().unwrap(),
        vec![Some("beta".to_string()), None, Some("alpha".to_string()), None]
    );
}

fn test_codebook_encode_and_decode_in_use_borrowed_context_domain() {
    let _event_observation = crate::common::lock_event_observation();
    let key_runtime = sample_runtime();
    let input_runtime = sample_runtime();
    let ctx_runtime = sample_runtime();
    let keys = key_runtime.series(&["alpha", "beta"]).unwrap();
    let codebook = Codebook::from_keys(keys).unwrap();
    let values = input_runtime.series(&["beta", "alpha", "missing"]).unwrap();
    let value_placement = values.placement();
    let ctx =
        ExecutionOverrides::new().with_stream(ctx_runtime.stream()).with_memory_resource(ctx_runtime.memory_resource());
    let expected_stream = ctx_runtime.stream().testing_raw_handle_value();
    let expected_mr = ctx_runtime.memory_resource().testing_raw_handle_value();

    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let encoded = codebook.encode_in(&values, &ctx).unwrap();
    let encoded_placement = encoded.placement();

    assert_eq!(encoded.to_optional_vec::<i32>().unwrap(), vec![Some(1), Some(0), None]);
    assert_eq!(encoded_placement.device, value_placement.device);
    assert_eq!(encoded_placement.workflow_id, value_placement.workflow_id);
    assert_ne!(encoded_placement.bundle_id, value_placement.bundle_id);
    assert_eq!(encoded_placement.stream_debug, expected_stream);
    assert_eq!(encoded_placement.memory_resource_debug, expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_exec_observation();
    let decoded = codebook.decode_in(&encoded, &ctx).unwrap();
    let decoded_placement = decoded.placement();

    assert_eq!(decoded.to_optional_strings().unwrap(), vec![Some("beta".to_string()), Some("alpha".to_string()), None]);
    assert_eq!(decoded_placement.stream_debug, expected_stream);
    assert_eq!(decoded_placement.memory_resource_debug, expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
}

fn test_codebook_can_decode_plain_series_from_external_code_columns() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();
    let codebook = s.codebook().unwrap();
    let encoded = codebook.encode(&crate::common::series("vertex", &["a", "c"]).unwrap()).unwrap();
    let external_codes = encoded.into_series();

    let decoded = codebook.decode_series(&external_codes).unwrap();
    assert_eq!(decoded.to_strings_strict().unwrap(), vec!["a", "c"]);
}

fn test_codebook_decode_series_works_from_plain_series() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();
    let codebook = s.codebook().unwrap();
    let encoded = codebook.encode(&crate::common::series("vertex", &["a", "c"]).unwrap()).unwrap();
    let external_codes = encoded.into_series();

    let decoded = codebook.decode_series(&external_codes).unwrap();
    assert_eq!(decoded.to_strings_strict().unwrap(), vec!["a", "c"]);
}

fn test_codebook_first_seen_ordering() {
    let s = crate::common::series("x", &["b", "a", "c", "a", "b"]).unwrap();

    let codebook = s.codebook_with(DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen)).unwrap();

    assert_eq!(codebook.keys().to_strings_strict().unwrap(), vec!["b", "a", "c"]);
}

fn test_zero_argument_codebook_preserves_sorted_default_ordering() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();

    let codebook = s.codebook().unwrap();

    assert_eq!(codebook.keys().to_strings_strict().unwrap(), vec!["a", "b", "c"]);
}

fn test_empty_decimal_codebook_preserves_scale() {
    let dtype = DataType::fixed_point(TypeId::Decimal64, -3).unwrap();
    let s = sample_runtime().empty_series(dtype).unwrap();

    let codebook = s.codebook().unwrap();

    assert_eq!(codebook.keys().dtype().unwrap(), dtype);
    assert_eq!(codebook.keys().len(), 0);
}

fn test_factorize_with_first_seen_ordering() {
    let s = crate::common::series("x", &["b", "a", "c", "a", "b"]).unwrap();

    let factorized =
        s.factorize_with(DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen)).unwrap();

    assert_eq!(factorized.codebook().keys().to_strings_strict().unwrap(), vec!["b", "a", "c"]);
    assert_eq!(factorized.row_codes().to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 1, 0]);
}

fn test_factorize_with_first_seen_ordering_preserves_row_nulls_and_excludes_null_from_codebook() {
    let s = crate::common::nullable_series("x", &["b", "", "a", "b", ""], &[true, false, true, true, false]).unwrap();

    let factorized =
        s.factorize_with(DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen)).unwrap();

    assert_eq!(factorized.codebook().keys().to_strings_strict().unwrap(), vec!["b", "a"]);
    assert_eq!(factorized.row_codes().to_optional_vec::<i32>().unwrap(), vec![Some(0), None, Some(1), Some(0), None]);
    assert_eq!(
        factorized.decode().unwrap().to_optional_strings().unwrap(),
        vec![Some("b".to_string()), None, Some("a".to_string()), Some("b".to_string()), None]
    );
}

fn test_codebook_from_keys_rejects_duplicate_keys() {
    let keys = crate::common::series("lookup_keys", &["b", "a", "b"]).unwrap();

    let err = match Codebook::from_keys(keys) {
        Ok(_) => panic!("expected duplicate key series to be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), cudf::ErrorCode::DuplicateKey);
    assert_eq!(err.operation(), "codebook::from_keys");
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(3)));
    assert_eq!(err.fact_value(FactKeyName::DistinctKeysCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.detail_value("keys_count"), None);
    assert_eq!(err.detail_value("distinct_keys_count"), None);
    assert_eq!(crate::common::error_context_value(&err, "key_count").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "distinct_key_count").as_deref(), None);
    assert!(err.message().contains("unique") || err.message().contains("duplicate"), "unexpected error: {err}");
}

fn test_factorize_with_first_seen_ordering_avoids_temp_name_collisions() {
    let s = crate::common::series("__cudf_first_seen_row_idx__", &["b", "a", "c", "a", "b"]).unwrap();

    let factorized =
        s.factorize_with(DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen)).unwrap();

    assert_eq!(factorized.codebook().keys().to_strings_strict().unwrap(), vec!["b", "a", "c"]);
    assert_eq!(factorized.row_codes().to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 1, 0]);
}

fn test_factorize_with_in_first_seen_ordering_uses_the_same_contract() {
    let s = crate::common::series("x", &["b", "a", "c", "a", "b"]).unwrap();
    let ctx = ExecutionOverrides::new();

    let factorized =
        s.factorize_with_in(DictionaryEncodingOptions::new().key_order(DictionaryKeyOrder::FirstSeen), &ctx).unwrap();

    assert_eq!(factorized.codebook().keys().to_strings_strict().unwrap(), vec!["b", "a", "c"]);
    assert_eq!(factorized.row_codes().to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 1, 0]);
}

fn test_zero_argument_factorize_preserves_sorted_default_ordering() {
    let s = crate::common::series("x", &["b", "a", "c", "a"]).unwrap();

    let factorized = s.factorize().unwrap();

    assert_eq!(factorized.codebook().keys().to_strings_strict().unwrap(), vec!["a", "b", "c"]);
    assert_eq!(factorized.row_codes().to_vec_strict::<i32>().unwrap(), vec![1, 0, 2, 0]);
}

fn test_empty_decimal_factorize_preserves_codebook_scale() {
    let dtype = DataType::fixed_point(TypeId::Decimal128, -6).unwrap();
    let s = sample_runtime().empty_series(dtype).unwrap();

    let factorized = s.factorize().unwrap();

    assert_eq!(factorized.codebook().keys().dtype().unwrap(), dtype);
    assert_eq!(factorized.codebook().keys().len(), 0);
    assert_eq!(factorized.row_codes().dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(factorized.row_codes().len(), 0);
}

// ===========================================================================
// sort_values
// ===========================================================================

fn test_sort_values_ascending() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let sorted = s.sort_values(SortOptions::new()).unwrap();

    assert_eq!(sorted.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

fn test_sort_values_descending() {
    let s = crate::common::series::<i32>("x", &[30, 10, 20]).unwrap();
    let sorted = s.sort_values(SortOptions::new().with_descending(vec![true])).unwrap();

    assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![30, 20, 10]);
}

fn test_sort_values_stable() {
    let s = crate::common::series::<i32>("x", &[3, 1, 2]).unwrap();
    let sorted = s.sort_values(SortOptions::new().with_stable(true)).unwrap();

    assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn test_sort_values_f64() {
    let s = crate::common::series::<f64>("x", &[3.3, 1.1, 2.2]).unwrap();
    let sorted = s.sort_values(SortOptions::new()).unwrap();

    assert_eq!(sorted.to_vec_strict::<f64>().unwrap(), vec![1.1, 2.2, 3.3]);
}

fn test_sort_values_empty() {
    let s = crate::common::series::<i32>("x", &[]).unwrap();
    let sorted = s.sort_values(SortOptions::new()).unwrap();

    assert_eq!(sorted.len(), 0);
}

fn test_sort_values_with_nulls() {
    let s = crate::common::nullable_series::<i32>("x", &[30, 0, 10], &[true, false, true]).unwrap();
    // Default: ascending, nulls last
    let sorted = s.sort_values(SortOptions::new()).unwrap();
    let view = sorted.testing_view().unwrap();

    assert_eq!(sorted.len(), 3);
    // The valid values should sort ascending, with the null row moved to the end.
    let vals = view.to_vec::<i32>().unwrap();
    assert_eq!(vals[0], 10);
    assert_eq!(vals[1], 30);
    assert_eq!(view.to_host_validity().unwrap().unwrap(), vec![true, true, false]);
    assert_eq!(sorted.null_count(), 1);
}

fn test_sort_values_nulls_first() {
    let s = crate::common::nullable_series::<i32>("x", &[30, 0, 10], &[true, false, true]).unwrap();
    let sorted = s.sort_values(SortOptions::new().with_nulls_first(vec![true])).unwrap();
    let view = sorted.testing_view().unwrap();

    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted.null_count(), 1);
    // Null should be first, then 10, 30
    let vals = view.to_vec::<i32>().unwrap();
    assert_eq!(view.to_host_validity().unwrap().unwrap(), vec![false, true, true]);
    assert_eq!(vals[1], 10);
    assert_eq!(vals[2], 30);
}

fn test_sort_values_rejects_wrong_descending_len() {
    let s = crate::common::series::<i32>("x", &[1, 2, 3]).unwrap();
    // Single column but 2 descending flags; should fail
    let result = s.sort_values(SortOptions::new().with_descending(vec![true, false]));
    let err = result.unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "series::sort_values");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("sort_options"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("descending_length_mismatch"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("provide either zero descending flags or exactly one flag per sort key"),
    );
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(1)));
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.detail_value("keys_count"), None);
    assert_eq!(error_context_value(&err, "num_keys").as_deref(), None);
    assert_eq!(err.detail_value("actual_count"), None);
    assert_eq!(error_context_value(&err, "descending_len").as_deref(), None);
}

// ===========================================================================
// input preservation
// ===========================================================================

fn test_convenience_ops_preserve_input() {
    let s = crate::common::nullable_series::<i32>("x", &[3, 0, 1], &[true, false, true]).unwrap();

    // Run several ops and verify the original is unchanged.
    let _null_mask = s.is_null().unwrap();
    let _not_null_mask = s.is_not_null().unwrap();
    let _filled = s.fillna(0i32).unwrap();
    let _sorted = s.sort_values(SortOptions::new()).unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(s.null_count(), 1);
    assert_eq!(s.dtype().unwrap().id(), TypeId::Int32);
}

#[test]
fn convenience_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_is_null_all_valid", test_is_null_all_valid),
        ("test_is_null_with_nulls", test_is_null_with_nulls),
        ("test_is_null_empty", test_is_null_empty),
        ("test_is_not_null_all_valid", test_is_not_null_all_valid),
        ("test_is_not_null_with_nulls", test_is_not_null_with_nulls),
        ("test_is_null_and_is_not_null_are_complementary", test_is_null_and_is_not_null_are_complementary),
        ("test_astype_i32_to_f64", test_astype_i32_to_f64),
        ("test_astype_f64_to_i32_truncates", test_astype_f64_to_i32_truncates),
        ("test_astype_preserves_length", test_astype_preserves_length),
        ("test_astype_preserves_nulls", test_astype_preserves_nulls),
        ("test_astype_same_type_is_no_op_copy", test_astype_same_type_is_no_op_copy),
        ("test_log_f64", test_log_f64),
        ("test_fillna_replaces_nulls_with_scalar", test_fillna_replaces_nulls_with_scalar),
        ("test_fillna_no_nulls_returns_same_values", test_fillna_no_nulls_returns_same_values),
        ("test_fillna_all_nulls", test_fillna_all_nulls),
        ("test_fillna_f64", test_fillna_f64),
        ("test_fillna_with_series_replaces_nulls", test_fillna_with_series_replaces_nulls),
        ("test_fillna_with_series_no_nulls", test_fillna_with_series_no_nulls),
        ("test_isin_basic", test_isin_basic),
        ("test_isin_no_matches", test_isin_no_matches),
        ("test_isin_all_match", test_isin_all_match),
        ("test_isin_strings", test_isin_strings),
        (
            "test_isin_type_mismatch_uses_canonical_error_contract",
            test_isin_type_mismatch_uses_canonical_error_contract,
        ),
        (
            "test_isin_succeeds_with_mixed_domains_and_uses_source_provenance",
            test_isin_succeeds_with_mixed_domains_and_uses_source_provenance,
        ),
        ("test_unique_removes_duplicates", test_unique_removes_duplicates),
        ("test_unique_already_unique", test_unique_already_unique),
        ("test_unique_empty", test_unique_empty),
        ("test_unique_strings", test_unique_strings),
        ("test_nunique_basic", test_nunique_basic),
        ("test_nunique_all_same", test_nunique_all_same),
        ("test_nunique_empty", test_nunique_empty),
        ("test_nunique_all_unique", test_nunique_all_unique),
        (
            "test_nunique_negative_distinct_count_uses_stable_foreign_contract",
            test_nunique_negative_distinct_count_uses_stable_foreign_contract,
        ),
        ("test_nunique_counts_nan_as_distinct_non_null_value", test_nunique_counts_nan_as_distinct_non_null_value),
        (
            "test_nunique_excludes_null_but_keeps_nan_distinctness",
            test_nunique_excludes_null_but_keeps_nan_distinctness,
        ),
        ("test_factorize_basic", test_factorize_basic),
        ("test_factorize_strings", test_factorize_strings),
        ("test_factorize_unnamed_input_keeps_outputs_unnamed", test_factorize_unnamed_input_keeps_outputs_unnamed),
        ("test_codebook_can_encode_and_decode_new_queries", test_codebook_can_encode_and_decode_new_queries),
        (
            "test_codebook_encode_rejects_dtype_mismatch_with_codebook_operation",
            test_codebook_encode_rejects_dtype_mismatch_with_codebook_operation,
        ),
        ("test_codebook_encode_returns_codes", test_codebook_encode_returns_codes),
        ("test_codebook_decode_returns_decoded_values", test_codebook_decode_returns_decoded_values),
        (
            "test_codebook_decode_preserves_null_codes_as_null_values",
            test_codebook_decode_preserves_null_codes_as_null_values,
        ),
        (
            "test_codebook_encode_and_decode_in_use_borrowed_context_domain",
            test_codebook_encode_and_decode_in_use_borrowed_context_domain,
        ),
        (
            "test_codebook_can_decode_plain_series_from_external_code_columns",
            test_codebook_can_decode_plain_series_from_external_code_columns,
        ),
        ("test_codebook_decode_series_works_from_plain_series", test_codebook_decode_series_works_from_plain_series),
        ("test_codebook_first_seen_ordering", test_codebook_first_seen_ordering),
        (
            "test_zero_argument_codebook_preserves_sorted_default_ordering",
            test_zero_argument_codebook_preserves_sorted_default_ordering,
        ),
        ("test_empty_decimal_codebook_preserves_scale", test_empty_decimal_codebook_preserves_scale),
        ("test_factorize_with_first_seen_ordering", test_factorize_with_first_seen_ordering),
        (
            "test_factorize_with_first_seen_ordering_preserves_row_nulls_and_excludes_null_from_codebook",
            test_factorize_with_first_seen_ordering_preserves_row_nulls_and_excludes_null_from_codebook,
        ),
        ("test_codebook_from_keys_rejects_duplicate_keys", test_codebook_from_keys_rejects_duplicate_keys),
        (
            "test_factorize_with_first_seen_ordering_avoids_temp_name_collisions",
            test_factorize_with_first_seen_ordering_avoids_temp_name_collisions,
        ),
        (
            "test_factorize_with_in_first_seen_ordering_uses_the_same_contract",
            test_factorize_with_in_first_seen_ordering_uses_the_same_contract,
        ),
        (
            "test_zero_argument_factorize_preserves_sorted_default_ordering",
            test_zero_argument_factorize_preserves_sorted_default_ordering,
        ),
        (
            "test_empty_decimal_factorize_preserves_codebook_scale",
            test_empty_decimal_factorize_preserves_codebook_scale,
        ),
        ("test_sort_values_ascending", test_sort_values_ascending),
        ("test_sort_values_descending", test_sort_values_descending),
        ("test_sort_values_stable", test_sort_values_stable),
        ("test_sort_values_f64", test_sort_values_f64),
        ("test_sort_values_empty", test_sort_values_empty),
        ("test_sort_values_with_nulls", test_sort_values_with_nulls),
        ("test_sort_values_nulls_first", test_sort_values_nulls_first),
        ("test_sort_values_rejects_wrong_descending_len", test_sort_values_rejects_wrong_descending_len),
        ("test_convenience_ops_preserve_input", test_convenience_ops_preserve_input),
    ]);
}
