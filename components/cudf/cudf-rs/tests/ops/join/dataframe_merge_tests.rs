//! Tests for the high-level `DataFrame::merge` API and `MergeOptions` / `JoinType`.
use cudf::{DataFrame, ErrorCode, ErrorKind, ExecutionOverrides, JoinType, MergeOptions, testing::prelude::*};

use crate::common::{
    assert_duplicate_name_error, assert_missing_column_error, column_from_slice, column_from_slice_nullable,
    error_context_value, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation,
    reset_sync_observation, sample_runtime,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn left_df() -> DataFrame {
    // id: [1, 2, 3], score: [10, 20, 30]
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df() -> DataFrame {
    // id: [2, 3, 4], value: [200, 300, 400]
    let ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let values = column_from_slice(&[200i32, 300, 400]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn left_df_asym() -> DataFrame {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    crate::common::dataframe_from_columns(vec!["left_id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df_asym() -> DataFrame {
    let ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let values = column_from_slice(&[200i32, 300, 400]).unwrap();
    crate::common::dataframe_from_columns(vec!["right_id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn nullable_i32_column(df: &DataFrame, name: &str) -> Vec<Option<i32>> {
    df.series(name).unwrap().to_optional_vec::<i32>().unwrap()
}

fn i32_column(df: &DataFrame, name: &str) -> Vec<i32> {
    df.series(name).unwrap().to_optional_vec::<i32>().unwrap().into_iter().map(|v| v.unwrap()).collect()
}

fn assert_merge_validation_error(err: &cudf::Error, operation: &str, reason: &str, remedy: &str) {
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), operation);
    assert_eq!(error_context_value(err, "reason").as_deref(), Some(reason));
    assert_eq!(error_context_value(err, "remedy").as_deref(), Some(remedy));
}

// ---------------------------------------------------------------------------
// MergeOptions builder
// ---------------------------------------------------------------------------

#[test]
fn test_merge_options_defaults() {
    let opts = MergeOptions::new();
    assert_eq!(opts.join_type(), JoinType::Inner);
    assert_eq!(opts.suffix_left(), "_left");
    assert_eq!(opts.suffix_right(), "_right");
}

#[test]
fn test_merge_options_builder_on() {
    let opts = MergeOptions::new().on(["id"]);
    let on = opts.on_keys().unwrap();
    assert_eq!(on.len(), 1);
    assert_eq!(on[0], "id");
}

#[test]
fn test_merge_options_builder_left_right_on() {
    let opts = MergeOptions::new().left_on(["left_id"]).right_on(["right_id"]);
    let lo = opts.left_on_keys().unwrap();
    let ro = opts.right_on_keys().unwrap();
    assert_eq!(lo.len(), 1);
    assert_eq!(lo[0], "left_id");
    assert_eq!(ro.len(), 1);
    assert_eq!(ro[0], "right_id");
}

#[test]
fn test_merge_options_builder_how() {
    assert_eq!(MergeOptions::new().how(JoinType::Left).join_type(), JoinType::Left);
    assert_eq!(MergeOptions::new().how_inner().join_type(), JoinType::Inner);
    assert_eq!(MergeOptions::new().how_left().join_type(), JoinType::Left);
    assert_eq!(MergeOptions::new().how_full().join_type(), JoinType::Full);
}

#[test]
fn test_merge_options_builder_suffixes() {
    let opts = MergeOptions::new().suffixes("_x", "_y");
    assert_eq!(opts.suffix_left(), "_x");
    assert_eq!(opts.suffix_right(), "_y");
}

// ---------------------------------------------------------------------------
// Inner merge — same key name (on)
// ---------------------------------------------------------------------------

#[test]
fn test_inner_merge_same_key() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    // Inner: matches on 2 and 3 → 2 rows.
    assert_eq!(result.num_rows(), 2);
    // Schema: shared key + left payload + right payload.
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);

    let mut rows: Vec<_> = i32_column(&result, "id")
        .into_iter()
        .zip(i32_column(&result, "score"))
        .zip(i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(2, 20, 200), (3, 30, 300)]);
}

#[test]
fn test_merge_succeeds_with_mixed_domains_and_uses_left_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_columns([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_columns([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let expected_exec_stream = left.testing_execution_domain_stream_value();
    let expected_exec_mr = left.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    // DataFrame guarantees one retained provenance bundle for all columns, so
    // one dependency from the right frame's stream orders both right keys and
    // payload before the left/output stream.
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    let rows: Vec<_> = i32_column(&result, "id")
        .into_iter()
        .zip(i32_column(&result, "score"))
        .zip(i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    assert_eq!(rows, vec![(2, 20, 200), (3, 30, 300)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

// ---------------------------------------------------------------------------
// Left merge — different key names (left_on / right_on)
// ---------------------------------------------------------------------------

#[test]
fn test_left_merge_asymmetric_keys() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result =
        left.merge(&right, MergeOptions::new().how_left().left_on(["left_id"]).right_on(["right_id"])).unwrap();

    // Left: all 3 left rows preserved.
    assert_eq!(result.num_rows(), 3);
    // With left_on/right_on, both key columns are preserved in MVP.
    assert_eq!(result.column_names(), vec!["left_id", "score", "right_id", "value"]);

    let mut rows: Vec<_> = nullable_i32_column(&result, "left_id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "right_id"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|(((left_id, score), right_id), value)| (left_id, score, right_id, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None, None),
            (Some(2), Some(20), Some(2), Some(200)),
            (Some(3), Some(30), Some(3), Some(300)),
        ]
    );
}

// ---------------------------------------------------------------------------
// Suffix handling for overlapping payload names
// ---------------------------------------------------------------------------

#[test]
fn test_merge_suffix_overlapping_payload() {
    // Both sides have "score" as payload.
    let left = left_df();
    let right_ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let right_scores = column_from_slice(&[200i32, 300, 400]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string()],
        vec![right_ids, right_scores],
    )
    .unwrap();

    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    // Default suffixes: _left, _right for conflicting payloads.
    assert_eq!(result.column_names(), vec!["id", "score_left", "score_right"]);

    let mut rows: Vec<_> = i32_column(&result, "id")
        .into_iter()
        .zip(i32_column(&result, "score_left"))
        .zip(i32_column(&result, "score_right"))
        .map(|((id, left_score), right_score)| (id, left_score, right_score))
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(2, 20, 200), (3, 30, 300)]);
}

#[test]
fn test_merge_custom_suffixes() {
    let left = left_df();
    let right_ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let right_scores = column_from_slice(&[200i32, 300, 400]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string()],
        vec![right_ids, right_scores],
    )
    .unwrap();

    let result = left.merge(&right, MergeOptions::new().on(["id"]).suffixes("_x", "_y")).unwrap();

    assert_eq!(result.column_names(), vec!["id", "score_x", "score_y"]);
}

#[test]
fn test_merge_no_suffix_when_no_conflict() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().on(["id"])).unwrap();

    // No conflict: "score" and "value" are distinct.
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);
}

// ---------------------------------------------------------------------------
// Full merge
// ---------------------------------------------------------------------------

#[test]
fn test_full_merge() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().how_full().on(["id"])).unwrap();

    // Full: keys 1,2,3,4 → 4 rows.
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);

    let mut rows: Vec<_> = nullable_i32_column(&result, "id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None),
            (Some(2), Some(20), Some(200)),
            (Some(3), Some(30), Some(300)),
            (Some(4), None, Some(400)),
        ]
    );
}

#[test]
fn test_full_merge_succeeds_with_mixed_domains_and_reassembles_on_left_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let left = DataFrame::from_columns([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_columns([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let expected_exec_stream = left.testing_execution_domain_stream_value();
    let expected_exec_mr = left.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = left.merge(&right, MergeOptions::new().how_full().on(["id"])).unwrap();

    assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    // DataFrame-level merge can order the whole right frame once because all
    // columns in a DataFrame share one retained provenance bundle.
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);
    assert_eq!(result.series("id").unwrap().as_column().testing_execution_domain_stream_value(), expected_exec_stream);
    assert_eq!(
        result.series("score").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );
    assert_eq!(
        result.series("value").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );

    // Verify materialized values from the full merge result.
    reset_sync_observation();
    let mut ids = result.series("id").unwrap().to_optional_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![Some(1), Some(2), Some(3), Some(4)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);

    reset_sync_observation();
    let mut rows: Vec<_> = nullable_i32_column(&result, "id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None),
            (Some(2), Some(20), Some(200)),
            (Some(3), Some(30), Some(300)),
            (Some(4), None, Some(400)),
        ]
    );
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_full_merge_in_context_succeeds_on_mixed_domains_and_reassembles_on_context_resources() {
    let _event_observation = crate::common::lock_event_observation();
    let left_rt = sample_runtime();
    let right_rt = sample_runtime();
    let target_rt = sample_runtime();
    let left = DataFrame::from_columns([
        ("id", left_rt.series(&[1i32, 2, 3]).unwrap()),
        ("score", left_rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_columns([
        ("id", right_rt.series(&[2i32, 3, 4]).unwrap()),
        ("value", right_rt.series(&[200i32, 300, 400]).unwrap()),
    ])
    .unwrap();
    let ctx =
        ExecutionOverrides::new().with_stream(target_rt.stream()).with_memory_resource(target_rt.memory_resource());
    let expected_exec_stream = target_rt.stream().testing_raw_handle_value();
    let expected_exec_mr = target_rt.memory_resource().testing_raw_handle_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = left.with_execution(&ctx).merge(&right, MergeOptions::new().how_full().on(["id"])).unwrap();

    assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_stream_value(), expected_exec_stream);
    assert_eq!(result.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 2);
    assert_eq!(result.series("id").unwrap().as_column().testing_execution_domain_stream_value(), expected_exec_stream);
    assert_eq!(
        result.series("score").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );
    assert_eq!(
        result.series("value").unwrap().as_column().testing_execution_domain_stream_value(),
        expected_exec_stream
    );

    // Verify materialized values from the full merge result.
    reset_sync_observation();
    let mut ids = result.series("id").unwrap().to_optional_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![Some(1), Some(2), Some(3), Some(4)]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);

    reset_sync_observation();
    let mut rows: Vec<_> = nullable_i32_column(&result, "id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|((id, score), value)| (id, score, value))
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (Some(1), Some(10), None),
            (Some(2), Some(20), Some(200)),
            (Some(3), Some(30), Some(300)),
            (Some(4), None, Some(400)),
        ]
    );
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_full_merge_asymmetric_keys_preserves_separate_key_columns() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result =
        left.merge(&right, MergeOptions::new().how_full().left_on(["left_id"]).right_on(["right_id"])).unwrap();

    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["left_id", "score", "right_id", "value"]);

    let mut rows: Vec<_> = nullable_i32_column(&result, "left_id")
        .into_iter()
        .zip(nullable_i32_column(&result, "score"))
        .zip(nullable_i32_column(&result, "right_id"))
        .zip(nullable_i32_column(&result, "value"))
        .map(|(((left_id, score), right_id), value)| (left_id, score, right_id, value))
        .collect();
    rows.sort();

    assert_eq!(
        rows,
        vec![
            (None, None, Some(4), Some(400)),
            (Some(1), Some(10), None, None),
            (Some(2), Some(20), Some(2), Some(200)),
            (Some(3), Some(30), Some(3), Some(300)),
        ]
    );
}

// ---------------------------------------------------------------------------
// Invalid key errors
// ---------------------------------------------------------------------------

#[test]
fn test_merge_missing_on_key_in_left() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(["nonexistent"])).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
    assert_eq!(err.operation(), "dataframe::merge");
}

#[test]
fn test_merge_missing_on_key_in_right() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(["score"])).unwrap_err();
    assert_missing_column_error(&err, "score");
}

#[test]
fn test_merge_left_on_without_right_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().left_on(["id"])).unwrap_err();
    assert_merge_validation_error(&err, "dataframe::merge", "left_on_without_right_on", "set right_on() or use on()");
}

#[test]
fn test_merge_right_on_without_left_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().right_on(["id"])).unwrap_err();
    assert_merge_validation_error(&err, "dataframe::merge", "right_on_without_left_on", "set left_on() or use on()");
}

#[test]
fn test_merge_left_on_right_on_length_mismatch() {
    let left = left_df_asym();
    let right = right_df_asym();

    let err = left.merge(&right, MergeOptions::new().left_on(["left_id"]).right_on(["right_id", "value"])).unwrap_err();
    assert_merge_validation_error(
        &err,
        "dataframe::merge",
        "join_key_count_mismatch",
        "set left_on() and right_on() with the same number of key columns",
    );
    assert_eq!(error_context_value(&err, "left_on_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "right_on_count").as_deref(), Some("2"));
}

#[test]
fn test_merge_no_keys_specified() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new()).unwrap_err();
    assert_merge_validation_error(&err, "dataframe::merge", "missing_join_keys", "set on() or left_on() + right_on()");
}

#[test]
fn test_merge_in_context_no_keys_specified() {
    let left = left_df();
    let right = right_df();
    let rt = sample_runtime();
    let ctx = ExecutionOverrides::new().with_stream(rt.stream()).with_memory_resource(rt.memory_resource());

    let err = left.with_execution(&ctx).merge(&right, MergeOptions::new()).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::with_execution::merge",
        "missing_join_keys",
        "set on() or left_on() + right_on()",
    );
}

#[test]
fn test_merge_reassembly_key_count_invariant_uses_stable_error_contract() {
    let err = cudf::testing::join::trigger_merge_reassembly_key_count_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dataframe::merge");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("join_result_key_count_exceeds_result_columns"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some(
            "report this as a library bug; join result schema should contain at least one column per key on each side"
        )
    );
    assert_eq!(error_context_value(&err, "keys_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "left_columns").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "right_columns").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "num_keys").as_deref(), None);
}

#[test]
fn test_merge_rejects_ambiguous_join_key_configuration() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(["id"]).left_on(["id"]).right_on(["id"])).unwrap_err();

    assert_merge_validation_error(
        &err,
        "dataframe::merge",
        "ambiguous_join_key_configuration",
        "use on() or left_on() + right_on(), not both",
    );
}

#[test]
fn test_merge_rejects_duplicate_output_names_after_reassembly() {
    let left_ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let left_conflicting = column_from_slice(&[10i32, 20, 30]).unwrap();
    let left = crate::common::dataframe_from_columns(
        vec!["left_id".to_string(), "right_id".to_string()],
        vec![left_ids, left_conflicting],
    )
    .unwrap();

    let right_ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let right_values = column_from_slice(&[200i32, 300, 400]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["right_id".to_string(), "value".to_string()],
        vec![right_ids, right_values],
    )
    .unwrap();

    let err = left.merge(&right, MergeOptions::new().left_on(["left_id"]).right_on(["right_id"])).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "dataframe::merge");
    assert_duplicate_name_error(&err, "right_id");
}

// ---------------------------------------------------------------------------
// Multi-key merge
// ---------------------------------------------------------------------------

#[test]
fn test_merge_multi_key_inner() {
    let lk0 = column_from_slice(&[1i32, 1, 2]).unwrap();
    let lk1 = column_from_slice(&[10i32, 20, 10]).unwrap();
    let lp = column_from_slice(&[100i32, 200, 300]).unwrap();
    let left = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "val".to_string()],
        vec![lk0, lk1, lp],
    )
    .unwrap();

    let rk0 = column_from_slice(&[1i32, 2, 2]).unwrap();
    let rk1 = column_from_slice(&[10i32, 10, 20]).unwrap();
    let rp = column_from_slice(&[1000i32, 2000, 3000]).unwrap();
    let right = crate::common::dataframe_from_columns(
        vec!["a".to_string(), "b".to_string(), "rval".to_string()],
        vec![rk0, rk1, rp],
    )
    .unwrap();

    let result = left.merge(&right, MergeOptions::new().on(["a", "b"])).unwrap();

    // Matches: (1,10) and (2,10) → 2 rows.
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column_names(), vec!["a", "b", "val", "rval"]);

    let mut rows: Vec<_> = i32_column(&result, "a")
        .into_iter()
        .zip(i32_column(&result, "b"))
        .zip(i32_column(&result, "val"))
        .zip(i32_column(&result, "rval"))
        .map(|(((a, b), val), rval)| (a, b, val, rval))
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, 10, 100, 1000), (2, 10, 300, 2000)]);
}

// ---------------------------------------------------------------------------
// NullEquality option
// ---------------------------------------------------------------------------

#[test]
fn test_merge_null_equality_unequal() {
    // With null_equality(Unequal), null keys should NOT match.
    let left_ids = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let left_vals = column_from_slice(&[10i32, 20, 30]).unwrap();
    let left =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "val".to_string()], vec![left_ids, left_vals])
            .unwrap();

    let right_ids = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let right_vals = column_from_slice(&[100i32, 200, 300]).unwrap();
    let right =
        crate::common::dataframe_from_columns(vec!["id".to_string(), "rval".to_string()], vec![right_ids, right_vals])
            .unwrap();

    use cudf::NullEquality;

    // With Equal: null == null matches, and the joined key remains null.
    let eq_result = left.merge(&right, MergeOptions::new().on(["id"]).null_equality(NullEquality::Equal)).unwrap();
    assert_eq!(eq_result.num_rows(), 3);
    let mut eq_rows: Vec<_> = nullable_i32_column(&eq_result, "id")
        .into_iter()
        .zip(i32_column(&eq_result, "val"))
        .zip(i32_column(&eq_result, "rval"))
        .map(|((id, val), rval)| (id, val, rval))
        .collect();
    eq_rows.sort();
    assert_eq!(eq_rows, vec![(None, 20, 200), (Some(1), 10, 100), (Some(3), 30, 300)]);

    // With Unequal: null != null → only rows with id=1 and id=3 match → 2 rows.
    let ne_result = left.merge(&right, MergeOptions::new().on(["id"]).null_equality(NullEquality::Unequal)).unwrap();
    assert_eq!(ne_result.num_rows(), 2);
    let mut ne_rows: Vec<_> = nullable_i32_column(&ne_result, "id")
        .into_iter()
        .zip(i32_column(&ne_result, "val"))
        .zip(i32_column(&ne_result, "rval"))
        .map(|((id, val), rval)| (id, val, rval))
        .collect();
    ne_rows.sort();
    assert_eq!(ne_rows, vec![(Some(1), 10, 100), (Some(3), 30, 300)]);
}
