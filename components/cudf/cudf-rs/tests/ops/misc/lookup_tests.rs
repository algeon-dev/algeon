//! Tests for lookup semantics through the dedicated lookup operator.
use cudf::{
    ErrorKind, TypeId,
    ops::lookup::{self, LookupTable},
    testing::prelude::*,
};

use crate::common::{
    column_from_slice, column_from_strs, explicit_i32_column, explicit_i32_series, last_exec_mr_value,
    last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
};

#[test]
fn test_lookup_probe_string_to_int() {
    let input = column_from_strs(&["a", "b", "c", "a"]).unwrap();
    let keys = column_from_strs(&["a", "b", "c"]).unwrap();
    let values = column_from_slice(&[1i32, 2, 3]).unwrap();

    let result =
        lookup::probe(&input.testing_view().unwrap(), &keys.testing_view().unwrap(), &values.testing_view().unwrap())
            .unwrap();

    let rv = result.testing_view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(rv.to_vec::<i32>().unwrap(), vec![1, 2, 3, 1]);
}

#[test]
fn test_lookup_probe_missing_keys_become_null() {
    let input = column_from_slice(&[1i32, 2, 3, 1]).unwrap();
    let keys = column_from_slice(&[1i32, 2]).unwrap();
    let values = column_from_slice(&[10i32, 20]).unwrap();

    let result =
        lookup::probe(&input.testing_view().unwrap(), &keys.testing_view().unwrap(), &values.testing_view().unwrap())
            .unwrap();

    let rv = result.testing_view().unwrap();
    assert_eq!(rv.len().unwrap(), 4);

    let validity = rv.to_host_validity().unwrap().unwrap();
    assert!(validity[0]);
    assert!(validity[1]);
    assert!(!validity[2]);
    assert!(validity[3]);

    let vals = rv.to_vec::<i32>().unwrap();
    assert_eq!(vals[0], 10);
    assert_eq!(vals[1], 20);
    assert_eq!(vals[3], 10);
}

#[test]
fn test_lookup_probe_empty_input_preserves_explicit_input_provenance() {
    let runtime = cudf::Runtime::builder().build().unwrap();
    let input = runtime.column_from_strs(&[]).unwrap();
    let keys = runtime.column_from_strs(&["a"]).unwrap();
    let values = runtime.column_from_slice(&[1i32]).unwrap();

    let result =
        lookup::probe(&input.testing_view().unwrap(), &keys.testing_view().unwrap(), &values.testing_view().unwrap())
            .unwrap();

    assert_eq!(result.testing_execution_domain_kind(), "explicit");
    assert_eq!(result.testing_execution_domain_id(), input.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_stream_value(), input.testing_execution_domain_stream_value());
    assert_eq!(result.testing_view().unwrap().len().unwrap(), 0);
}

#[test]
fn test_lookup_probe_preserves_output_type() {
    let input = column_from_slice(&[1i32, 2]).unwrap();
    let keys = column_from_slice(&[1i32, 2]).unwrap();
    let values = column_from_slice(&[1.5f64, 2.5]).unwrap();

    let result =
        lookup::probe(&input.testing_view().unwrap(), &keys.testing_view().unwrap(), &values.testing_view().unwrap())
            .unwrap();

    let rv = result.testing_view().unwrap();
    assert_eq!(rv.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(rv.to_vec::<f64>().unwrap(), vec![1.5, 2.5]);
}

#[test]
fn test_lookup_probe_rejects_duplicate_keys() {
    let input = column_from_slice(&[1i32, 2, 1]).unwrap();
    let keys = column_from_slice(&[1i32, 1, 2]).unwrap();
    let values = column_from_slice(&[10i32, 11, 20]).unwrap();

    let err =
        lookup::probe(&input.testing_view().unwrap(), &keys.testing_view().unwrap(), &values.testing_view().unwrap())
            .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "lookup::probe");
}

#[test]
fn test_lookup_probe_rejects_mapping_length_mismatch() {
    let input = column_from_slice(&[1i32, 2]).unwrap();
    let keys = column_from_slice(&[1i32, 2]).unwrap();
    let values = column_from_slice(&[10i32]).unwrap();

    let err =
        lookup::probe(&input.testing_view().unwrap(), &keys.testing_view().unwrap(), &values.testing_view().unwrap())
            .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "lookup::probe");
    assert!(err.message().contains("same length"));
}

#[test]
fn test_lookup_probe_succeeds_with_mixed_domains_and_uses_input_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let input = explicit_i32_column(&[2, 9, 1], 921);
    let keys = explicit_i32_column(&[1, 2], 920);
    let values = explicit_i32_column(&[10, 20], 920);

    let expected_stream = input.testing_execution_domain_stream_value();
    let expected_mr = input.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result =
        lookup::probe(&input.testing_view().unwrap(), &keys.testing_view().unwrap(), &values.testing_view().unwrap())
            .unwrap();

    assert_eq!(result.testing_execution_domain_id(), input.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_stream), 2);

    reset_sync_observation();
    assert_eq!(result.testing_view().unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(20), None, Some(10)]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn test_lookup_table_rejects_duplicate_keys_at_build_time() {
    let keys = crate::common::series("id", &["u1", "u1"]).unwrap();
    let values = crate::common::series("label", &["alice", "alice-2"]).unwrap();

    let err = LookupTable::new(keys, values).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "lookup_table::new");
}

#[test]
fn test_lookup_table_rejects_mapping_length_mismatch_at_build_time() {
    let keys = crate::common::series::<i32>("id", &[1, 2]).unwrap();
    let values = crate::common::series("label", &["alice"]).unwrap();

    let err = LookupTable::new(keys, values).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "lookup_table::new");
    assert!(err.message().contains("same length"));
}

#[test]
fn test_lookup_table_supports_repeated_probe() {
    let keys = crate::common::series("id", &["u1", "u2"]).unwrap();
    let values = crate::common::series("label", &["alice", "bob"]).unwrap();
    let table = LookupTable::new(keys, values).unwrap();

    let first = crate::common::series("vertex", &["u2", "u9", "u1"]).unwrap();
    let second = crate::common::series("vertex", &["u1", "u2"]).unwrap();

    let first_result = table.probe(&first).unwrap();
    let second_result = table.probe(&second).unwrap();

    assert_eq!(
        first_result.to_optional_strings().unwrap(),
        vec![Some("bob".to_string()), None, Some("alice".to_string())]
    );
    assert_eq!(second_result.to_optional_strings().unwrap(), vec![Some("alice".to_string()), Some("bob".to_string())]);
}

#[test]
fn test_lookup_table_exposes_mapping_parts() {
    let keys = crate::common::series("id", &["u1", "u2"]).unwrap();
    let values = crate::common::series("label", &["alice", "bob"]).unwrap();
    let table = LookupTable::new(keys.clone(), values.clone()).unwrap();

    assert_eq!(table.len(), 2);
    assert!(!table.is_empty());
    assert_eq!(table.keys().to_strings_strict().unwrap(), keys.to_strings_strict().unwrap());
    assert_eq!(table.values().to_strings_strict().unwrap(), values.to_strings_strict().unwrap());
}

#[test]
fn test_lookup_table_probe_returns_series() {
    let keys = crate::common::series::<i32>("id", &[1, 2]).unwrap();
    let values = crate::common::series("label", &["alice", "bob"]).unwrap();
    let table = LookupTable::new(keys, values).unwrap();
    let input = crate::common::series::<i32>("vertex", &[2, 1, 9]).unwrap();

    let result = table.probe(&input).unwrap();
    assert_eq!(result.to_optional_strings().unwrap(), vec![Some("bob".to_string()), Some("alice".to_string()), None]);
}

#[test]
fn test_lookup_table_probe_rejects_dtype_mismatch() {
    let keys = crate::common::series("id", &["u1", "u2"]).unwrap();
    let values = crate::common::series("label", &["alice", "bob"]).unwrap();
    let table = LookupTable::new(keys, values).unwrap();
    let input = crate::common::series::<i32>("vertex", &[1, 2]).unwrap();

    let err = table.probe(&input).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "lookup_table::probe");
}

#[test]
fn test_lookup_table_probe_succeeds_with_mixed_domains_and_uses_input_provenance() {
    let _event_observation = crate::common::lock_event_observation();
    let keys = explicit_i32_series("id", &[1, 2], 920);
    let values = explicit_i32_series("label", &[10, 20], 920);
    let table = LookupTable::new(keys, values).unwrap();
    let input = explicit_i32_series("vertex", &[2, 9, 1], 921);

    let expected_stream = input.as_column().testing_execution_domain_stream_value();
    let expected_mr = input.as_column().testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let result = table.probe(&input).unwrap();

    assert_eq!(result.as_column().testing_execution_domain_id(), input.as_column().testing_execution_domain_id());
    assert_eq!(result.as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_stream), 2);

    reset_sync_observation();
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(20), None, Some(10)]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}
