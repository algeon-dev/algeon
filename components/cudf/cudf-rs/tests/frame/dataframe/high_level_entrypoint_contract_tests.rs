//! Tests for representative high-level DataFrame entrypoint contracts.
//!
//! These tests protect user-visible behavior for the high-level APIs that take
//! `Series` or plain Rust values, while keeping one low-level smoke path
//! through `view()` plus `cudf::ops::*`.
use cudf::{
    DataFrame, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::{copying, filter as low_level_filter, search},
    testing::prelude::*,
};

use crate::common::{
    assert_missing_column_error, column_from_slice, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value,
    reset_exec_observation, reset_sync_observation, sample_runtime, scalar,
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a 5-row DataFrame with id (i32) and score (f64) columns.
fn sample_df() -> DataFrame {
    let ids = crate::common::unnamed_series::<i32>(&[10, 20, 30, 40, 50]).unwrap();
    let scores = crate::common::unnamed_series::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    DataFrame::from_series([("id", ids), ("score", scores)]).unwrap()
}

// ===========================================================================
// 1. gather(&Series)
// ===========================================================================

#[test]
fn test_gather_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let indices = crate::common::unnamed_series::<i32>(&[0, 2, 4]).unwrap();
    let result = df.gather(&indices).unwrap();
    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_optional_vec.
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(10), Some(30), Some(50)]);
    assert_eq!(
        result.series("score").unwrap().to_optional_vec::<f64>().unwrap(),
        vec![Some(1.0), Some(3.0), Some(5.0)]
    );
}

#[test]
fn test_gather_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let df_rt = sample_runtime();
    let indices_rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", df_rt.series(&[10i32, 20, 30]).unwrap()),
        ("score", df_rt.series(&[1.0f64, 2.0, 3.0]).unwrap()),
    ])
    .unwrap();
    let indices = indices_rt.series(&[2i32, 0]).unwrap();
    let expected_exec_stream = df.testing_execution_domain_stream_value();
    let expected_exec_mr = df.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = df.gather(&indices).unwrap();

    assert_eq!(result.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    // Gather produces nullable output; use to_optional_vec.
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(30), Some(10)]);
    assert_eq!(result.series("score").unwrap().to_optional_vec::<f64>().unwrap(), vec![Some(3.0), Some(1.0)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ===========================================================================
// 2. filter_isin(&str, &Series)
// ===========================================================================

#[test]
fn test_filter_isin_with_series() {
    let df = sample_df(); // id: 10,20,30,40,50
    let values = crate::common::unnamed_series::<i32>(&[10, 30, 50]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["id", "score"]);
    let mut ids = result.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
}

#[test]
fn test_filter_isin_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let df_rt = sample_runtime();
    let values_rt = sample_runtime();
    let df = DataFrame::from_series([("id", df_rt.series(&[10i32, 20, 30]).unwrap())]).unwrap();
    let values = values_rt.series(&[20i32, 30]).unwrap();
    let expected_exec_stream = df.testing_execution_domain_stream_value();
    let expected_exec_mr = df.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = df.filter_isin("id", &values).unwrap();

    assert_eq!(result.testing_execution_domain_id(), df.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    assert_eq!(result.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_filter_isin_with_series_no_matches_preserves_schema() {
    let df = sample_df();
    let values = crate::common::unnamed_series::<i32>(&[99, 100]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

#[test]
fn test_filter_isin_with_series_rejects_missing_column() {
    let df = sample_df();
    let values = crate::common::unnamed_series::<i32>(&[10]).unwrap();
    let err = df.filter_isin("nonexistent", &values).unwrap_err();

    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// 3. fillna<T: CudfType>
// ===========================================================================

#[test]
fn test_fillna_with_plain_value() {
    let ids =
        crate::common::unnamed_nullable_series::<i32>(&[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = crate::common::unnamed_series::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();

    let filled = df.fillna::<i32>("id", 42).unwrap();
    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    // Other columns unchanged.
    assert_eq!(filled.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fillna_with_scalar_still_works() {
    let ids =
        crate::common::unnamed_nullable_series::<i32>(&[10, 0, 30, 0, 50], &[true, false, true, false, true]).unwrap();
    let scores = crate::common::unnamed_series::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();

    let fill = scalar(42i32).unwrap();
    let filled = df.fillna_with_scalar("id", &fill).unwrap();

    assert_eq!(filled.num_rows(), 5);
    assert_eq!(filled.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 42, 30, 42, 50]);
    assert_eq!(filled.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fillna_rejects_missing_column() {
    let ids = crate::common::unnamed_series::<i32>(&[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([("id", ids)]).unwrap();
    let err = df.fillna::<i32>("nonexistent", 42).unwrap_err();

    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// 4. factorize returns typed factorization results
// ===========================================================================

#[test]
fn test_factorize_returns_typed_result() {
    let df = sample_df(); // id: 10,20,30,40,50
    let factorized = df.factorize("id").unwrap();
    assert_eq!(factorized.row_codes().len(), 5);
    assert_eq!(factorized.codebook().len(), 5); // all values are unique
}

#[test]
fn test_with_lookup_from_round_trip_against_narrow_lookup_dataframe() {
    let df = DataFrame::from_series([("vertex", crate::common::unnamed_series(&["u2", "u9", "u1"]).unwrap())]).unwrap();
    let lookup_df = DataFrame::from_series([
        ("id", crate::common::unnamed_series(&["u1", "u2"]).unwrap()),
        ("label", crate::common::unnamed_series(&["alice", "bob"]).unwrap()),
    ])
    .unwrap();

    let looked_up = df.with_lookup_from("owner", "vertex", &lookup_df, "id", "label").unwrap();

    assert_eq!(
        looked_up.series("owner").unwrap().to_optional_strings().unwrap(),
        vec![Some("bob".to_string()), None, Some("alice".to_string())]
    );
}

#[test]
fn test_with_lookup_rejects_mapping_length_mismatch_with_dataframe_operation() {
    let df = DataFrame::from_series([("vertex", crate::common::unnamed_series(&["u2", "u1"]).unwrap())]).unwrap();
    let keys = crate::common::unnamed_series(&["u1", "u2"]).unwrap();
    let values = crate::common::unnamed_series(&["alice"]).unwrap();

    let err = df.with_lookup("owner", "vertex", &keys, &values).unwrap_err();

    assert_eq!(err.kind(), cudf::ErrorKind::InvalidInput);
    assert_eq!(err.code(), cudf::ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "dataframe::with_lookup");
    assert_eq!(crate::common::error_context_value(&err, "column_name").as_deref(), Some("vertex"));
    assert_eq!(err.detail_value("column_name"), None);
    assert_eq!(crate::common::error_context_value(&err, "input_column").as_deref(), None);
    assert_eq!(err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.detail_value("values_count"), Some("1"));
    assert_eq!(err.detail_value("keys_count"), None);
    assert_eq!(crate::common::error_context_value(&err, "keys_type_id").as_deref(), None);
    assert_eq!(crate::common::error_context_value(&err, "output_type_id").as_deref(), None);
}

#[test]
fn test_with_lookup_from_rejects_missing_lookup_key_column_with_canonical_context() {
    let df = DataFrame::from_series([("vertex", crate::common::unnamed_series(&["u2", "u1"]).unwrap())]).unwrap();
    let lookup_df =
        DataFrame::from_series([("label", crate::common::unnamed_series(&["alice", "bob"]).unwrap())]).unwrap();

    let err = df.with_lookup_from("owner", "vertex", &lookup_df, "id", "label").unwrap_err();

    crate::common::assert_missing_column_error(&err, "id");
    assert_eq!(err.operation(), "dataframe::with_lookup_from");
    assert_eq!(crate::common::error_context_value(&err, "key_column").as_deref(), None);
}

#[test]
fn test_with_lookup_from_rejects_missing_lookup_value_column_with_canonical_context() {
    let df = DataFrame::from_series([("vertex", crate::common::unnamed_series(&["u2", "u1"]).unwrap())]).unwrap();
    let lookup_df = DataFrame::from_series([("id", crate::common::unnamed_series(&["u1", "u2"]).unwrap())]).unwrap();

    let err = df.with_lookup_from("owner", "vertex", &lookup_df, "id", "label").unwrap_err();

    crate::common::assert_missing_column_error(&err, "label");
    assert_eq!(err.operation(), "dataframe::with_lookup_from");
    assert_eq!(crate::common::error_context_value(&err, "value_column").as_deref(), None);
}

#[test]
fn test_with_encoded_and_decoded_column_workflows() {
    let df = DataFrame::from_series([("kind", crate::common::unnamed_series(&["b", "a", "c", "a"]).unwrap())]).unwrap();
    let codebook = df.codebook("kind").unwrap();

    let encoded = df.with_encoded_column("kind_id", "kind", &codebook).unwrap();
    assert_eq!(encoded.series("kind_id").unwrap().dtype().unwrap().id(), TypeId::Int32);

    let decoded = encoded.with_decoded_column("kind_roundtrip", "kind_id", &codebook).unwrap();
    assert_eq!(decoded.series("kind_roundtrip").unwrap().to_strings_strict().unwrap(), vec!["b", "a", "c", "a"]);
}

#[test]
fn test_with_encoded_column_rejects_missing_input_column_with_canonical_context() {
    let df = DataFrame::from_series([("kind", crate::common::unnamed_series(&["b", "a", "c", "a"]).unwrap())]).unwrap();
    let codebook = df.codebook("kind").unwrap();

    let err = df.with_encoded_column("kind_id", "missing", &codebook).unwrap_err();

    crate::common::assert_missing_column_error(&err, "missing");
    assert_eq!(err.operation(), "dataframe::with_encoded_column");
    assert_eq!(crate::common::error_context_value(&err, "input_column").as_deref(), None);
}

#[test]
fn test_with_decoded_column_rejects_missing_input_column_with_canonical_context() {
    let df = DataFrame::from_series([("kind_id", crate::common::unnamed_series(&[1i32, 0, 2, 0]).unwrap())]).unwrap();
    let codebook = DataFrame::from_series([("kind", crate::common::unnamed_series(&["b", "a", "c"]).unwrap())])
        .unwrap()
        .codebook("kind")
        .unwrap();

    let err = df.with_decoded_column("kind", "missing", &codebook).unwrap_err();

    crate::common::assert_missing_column_error(&err, "missing");
    assert_eq!(err.operation(), "dataframe::with_decoded_column");
    assert_eq!(crate::common::error_context_value(&err, "input_column").as_deref(), None);
}

// ===========================================================================
// 5. Advanced gather uses view() + low-level ops::copying
// ===========================================================================

#[test]
fn test_advanced_gather_still_works_via_view_and_ops() {
    let df = sample_df(); // id: 10,20,30,40,50
    let indices = column_from_slice(&[0i32, 2, 4]).unwrap();
    let table_view = df.testing_view().unwrap();
    let result = copying::gather(&table_view, &indices.testing_view().unwrap()).unwrap();
    let result = DataFrame::from_table(result, vec!["id".to_string(), "score".to_string()]).unwrap();

    assert_eq!(result.num_rows(), 3);
    // Gather produces nullable output; use to_optional_vec.
    assert_eq!(result.series("id").unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(10), Some(30), Some(50)]);
    assert_eq!(
        result.series("score").unwrap().to_optional_vec::<f64>().unwrap(),
        vec![Some(1.0), Some(3.0), Some(5.0)]
    );
    // Schema preserved.
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

// ===========================================================================
// 6. Advanced filter_isin uses view() + low-level ops
// ===========================================================================

#[test]
fn test_advanced_filter_isin_still_works_via_view_and_ops() {
    let df = sample_df(); // id: 10,20,30,40,50
    let values = column_from_slice(&[10i32, 30, 50]).unwrap();
    let values_view = values.testing_view().unwrap();
    let ids = df.series("id").unwrap();
    let ids_view = ids.testing_view().unwrap();
    let mask = search::contains(&values_view, &ids_view).unwrap();
    let mask_view = mask.testing_view().unwrap();
    let table_view = df.testing_view().unwrap();
    let result = low_level_filter::apply_boolean_mask(&table_view, &mask_view).unwrap();
    let result = DataFrame::from_table(result, vec!["id".to_string(), "score".to_string()]).unwrap();

    assert_eq!(result.num_rows(), 3);
    let mut ids = result.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![10, 30, 50]);
    // Schema preserved.
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

// ===========================================================================
// 7. runtime.sequence(start, step, len)
// ===========================================================================

#[test]
fn test_runtime_sequence_creates_device_side_fixed_width_series() {
    let rt = sample_runtime();

    let seq = rt.sequence(0i32, 1i32, 5).unwrap();
    assert_eq!(seq.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_runtime_sequence_uses_runtime_execution_domain() {
    let rt = sample_runtime();

    let seq = rt.sequence(3i32, 2i32, 4).unwrap();

    assert_eq!(seq.as_column().testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(seq.to_vec_strict::<i32>().unwrap(), vec![3, 5, 7, 9]);
}
