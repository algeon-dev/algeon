//! Tests for concatenation operations.
use cudf::{
    ErrorCode, ErrorKind, Exec, Series,
    testing::{
        concatenate::{
            trigger_concatenate_column_count_limit_error_for_test, trigger_concatenate_table_count_limit_error_for_test,
        },
        prelude::*,
    },
};

use crate::common::{
    error_context_value, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation,
    reset_sync_observation, run_named_cases, sample_runtime,
};

fn test_testing_hook_locks_concatenate_column_count_limit_contract() {
    let err = trigger_concatenate_column_count_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "concatenate::concatenate_columns");
    assert_eq!(err.message(), "number of input columns exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("input_columns"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "column_count").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce column_count to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "num_columns").as_deref(), None);
}

fn test_testing_hook_locks_concatenate_table_count_limit_contract() {
    let err = trigger_concatenate_table_count_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "concatenate::concatenate_tables");
    assert_eq!(err.message(), "number of input tables exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("input_tables"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "table_count").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce table_count to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "num_tables").as_deref(), None);
}

fn test_series_concat_succeeds_with_mixed_domains_and_uses_first_provenance() {
    let first_rt = sample_runtime();
    let second_rt = sample_runtime();
    let first = first_rt.series(&[1i32, 2, 3]).unwrap();
    let second = second_rt.series(&[4i32, 5]).unwrap();
    let expected_stream = first.as_column().testing_execution_domain_stream_value();
    let expected_mr = first.as_column().testing_execution_domain_mr_value();

    reset_exec_observation();
    let result = Series::concat(&[&first, &second], Exec::RETAINED).unwrap();

    assert_eq!(result.as_column().testing_execution_domain_id(), first.as_column().testing_execution_domain_id());
    assert_eq!(result.as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn concatenate_tests_cases() {
    run_named_cases(&[
        (
            "test_testing_hook_locks_concatenate_column_count_limit_contract",
            test_testing_hook_locks_concatenate_column_count_limit_contract,
        ),
        (
            "test_testing_hook_locks_concatenate_table_count_limit_contract",
            test_testing_hook_locks_concatenate_table_count_limit_contract,
        ),
        (
            "test_series_concat_succeeds_with_mixed_domains_and_uses_first_provenance",
            test_series_concat_succeeds_with_mixed_domains_and_uses_first_provenance,
        ),
    ]);
}
