//! Tests for representative high-level DataFrame entrypoint contracts.
//!
//! These tests protect user-visible behavior for the high-level APIs that take
//! `Series` or plain Rust values, while keeping one low-level smoke path
//! through `view()` plus `cudf::ops::*`.
use cudf::{
    DataFrame, ExecutionOverrides, TypeId,
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

fn test_filter_isin_with_series_no_matches_preserves_schema() {
    let df = sample_df();
    let values = crate::common::unnamed_series::<i32>(&[99, 100]).unwrap();
    let result = df.filter_isin("id", &values).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column_names(), vec!["id", "score"]);
}

fn test_filter_isin_with_series_rejects_missing_column() {
    let df = sample_df();
    let values = crate::common::unnamed_series::<i32>(&[10]).unwrap();
    let err = df.filter_isin("nonexistent", &values).unwrap_err();

    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// 3. fillna<T: CudfType>
// ===========================================================================

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

fn test_fillna_rejects_missing_column() {
    let ids = crate::common::unnamed_series::<i32>(&[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([("id", ids)]).unwrap();
    let err = df.fillna::<i32>("nonexistent", 42).unwrap_err();

    assert_missing_column_error(&err, "nonexistent");
}

// ===========================================================================
// 4. factorize returns typed factorization results
// ===========================================================================

fn test_factorize_returns_typed_result() {
    let df = sample_df(); // id: 10,20,30,40,50
    let factorized = df.factorize("id").unwrap();
    assert_eq!(factorized.row_codes().len(), 5);
    assert_eq!(factorized.codebook().len(), 5); // all values are unique
}

fn test_with_encoded_and_decoded_column_workflows() {
    let df = DataFrame::from_series([("kind", crate::common::unnamed_series(&["b", "a", "c", "a"]).unwrap())]).unwrap();
    let codebook = df.codebook("kind").unwrap();

    let encoded = df.with_encoded_column("kind_id", "kind", &codebook).unwrap();
    assert_eq!(encoded.series("kind_id").unwrap().dtype().unwrap().id(), TypeId::Int32);

    let decoded = encoded.with_decoded_column("kind_roundtrip", "kind_id", &codebook).unwrap();
    assert_eq!(decoded.series("kind_roundtrip").unwrap().to_strings_strict().unwrap(), vec!["b", "a", "c", "a"]);
}

fn test_with_decoded_column_preserves_null_codes_as_null_values() {
    let df = DataFrame::from_series([(
        "kind",
        crate::common::unnamed_nullable_series(&["b", "", "a", "b"], &[true, false, true, true]).unwrap(),
    )])
    .unwrap();
    let codebook = df.codebook("kind").unwrap();
    let encoded = df.with_encoded_column("kind_id", "kind", &codebook).unwrap();

    let decoded = encoded.with_decoded_column("kind_roundtrip", "kind_id", &codebook).unwrap();

    assert_eq!(
        decoded.series("kind_roundtrip").unwrap().to_optional_strings().unwrap(),
        vec![Some("b".to_string()), None, Some("a".to_string()), Some("b".to_string())]
    );
}

fn test_with_execution_encoded_and_decoded_columns_use_borrowed_context_domain() {
    let _event_observation = crate::common::lock_event_observation();
    let df_runtime = sample_runtime();
    let ctx_runtime = sample_runtime();
    let df = DataFrame::from_series([(
        "kind",
        df_runtime.series(&["segment-alpha-long", "segment-beta-long", "segment-alpha-long"]).unwrap(),
    )])
    .unwrap();
    let codebook = df.codebook("kind").unwrap();
    let ctx =
        ExecutionOverrides::new().with_stream(ctx_runtime.stream()).with_memory_resource(ctx_runtime.memory_resource());
    let expected_stream = ctx_runtime.stream().testing_raw_handle_value();
    let expected_mr = ctx_runtime.memory_resource().testing_raw_handle_value();

    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let encoded = df.with_execution(&ctx).with_encoded_column("kind_id", "kind", &codebook).unwrap();

    assert_eq!(encoded.series("kind_id").unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 1, 0]);
    assert_eq!(encoded.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(encoded.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_exec_observation();
    let decoded = encoded.with_execution(&ctx).with_decoded_column("kind_roundtrip", "kind_id", &codebook).unwrap();

    assert_eq!(
        decoded.series("kind_roundtrip").unwrap().to_strings_strict().unwrap(),
        vec!["segment-alpha-long", "segment-beta-long", "segment-alpha-long"]
    );
    assert_eq!(decoded.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(decoded.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
}

fn test_with_encoded_column_rejects_missing_input_column_with_canonical_context() {
    let df = DataFrame::from_series([("kind", crate::common::unnamed_series(&["b", "a", "c", "a"]).unwrap())]).unwrap();
    let codebook = df.codebook("kind").unwrap();

    let err = df.with_encoded_column("kind_id", "missing", &codebook).unwrap_err();

    crate::common::assert_missing_column_error(&err, "missing");
    assert_eq!(err.operation(), "dataframe::with_encoded_column");
    assert_eq!(crate::common::error_context_value(&err, "input_column").as_deref(), None);
}

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

fn test_runtime_sequence_creates_device_side_fixed_width_series() {
    let rt = sample_runtime();

    let seq = rt.sequence(0i32, 1i32, 5).unwrap();
    assert_eq!(seq.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2, 3, 4]);
}

fn test_runtime_sequence_uses_runtime_execution_domain() {
    let rt = sample_runtime();

    let seq = rt.sequence(3i32, 2i32, 4).unwrap();

    assert_eq!(seq.as_column().testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(seq.to_vec_strict::<i32>().unwrap(), vec![3, 5, 7, 9]);
}

#[test]
fn high_level_entrypoint_contract_tests_cases() {
    crate::common::run_named_cases(&[
        ("test_gather_with_series", test_gather_with_series),
        (
            "test_gather_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance",
            test_gather_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance,
        ),
        ("test_filter_isin_with_series", test_filter_isin_with_series),
        (
            "test_filter_isin_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance",
            test_filter_isin_with_series_succeeds_with_mixed_domains_and_uses_dataframe_provenance,
        ),
        (
            "test_filter_isin_with_series_no_matches_preserves_schema",
            test_filter_isin_with_series_no_matches_preserves_schema,
        ),
        ("test_filter_isin_with_series_rejects_missing_column", test_filter_isin_with_series_rejects_missing_column),
        ("test_fillna_with_plain_value", test_fillna_with_plain_value),
        ("test_fillna_with_scalar_still_works", test_fillna_with_scalar_still_works),
        ("test_fillna_rejects_missing_column", test_fillna_rejects_missing_column),
        ("test_factorize_returns_typed_result", test_factorize_returns_typed_result),
        ("test_with_encoded_and_decoded_column_workflows", test_with_encoded_and_decoded_column_workflows),
        (
            "test_with_decoded_column_preserves_null_codes_as_null_values",
            test_with_decoded_column_preserves_null_codes_as_null_values,
        ),
        (
            "test_with_execution_encoded_and_decoded_columns_use_borrowed_context_domain",
            test_with_execution_encoded_and_decoded_columns_use_borrowed_context_domain,
        ),
        (
            "test_with_encoded_column_rejects_missing_input_column_with_canonical_context",
            test_with_encoded_column_rejects_missing_input_column_with_canonical_context,
        ),
        (
            "test_with_decoded_column_rejects_missing_input_column_with_canonical_context",
            test_with_decoded_column_rejects_missing_input_column_with_canonical_context,
        ),
        ("test_advanced_gather_still_works_via_view_and_ops", test_advanced_gather_still_works_via_view_and_ops),
        (
            "test_advanced_filter_isin_still_works_via_view_and_ops",
            test_advanced_filter_isin_still_works_via_view_and_ops,
        ),
        (
            "test_runtime_sequence_creates_device_side_fixed_width_series",
            test_runtime_sequence_creates_device_side_fixed_width_series,
        ),
        ("test_runtime_sequence_uses_runtime_execution_domain", test_runtime_sequence_uses_runtime_execution_domain),
    ]);
}
