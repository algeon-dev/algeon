//! Tests for dictionary encode/decode.
use cudf::{
    ErrorCode, ErrorKind, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::dictionary,
    testing::{ColumnView, prelude::*},
};

use crate::common::{
    column_from_slice, column_from_slice_nullable, error_context_value, last_exec_mr_value, last_exec_stream_value,
    last_sync_stream_value, reset_exec_observation, reset_sync_observation,
};

fn dictionary_indices_to_i64(dict_view: &ColumnView<'_>) -> Vec<i64> {
    let indices_view = dictionary::indices(dict_view).unwrap();
    match indices_view.data_type().unwrap().id() {
        TypeId::Int8 => indices_view.to_vec::<i8>().unwrap().into_iter().map(i64::from).collect(),
        TypeId::Int16 => indices_view.to_vec::<i16>().unwrap().into_iter().map(i64::from).collect(),
        TypeId::Int32 => indices_view.to_vec::<i32>().unwrap().into_iter().map(i64::from).collect(),
        TypeId::Int64 => indices_view.to_vec::<i64>().unwrap(),
        other => panic!("unexpected dictionary index type: {other:?}"),
    }
}

#[test]
fn test_dictionary_encode_i32() {
    let data = vec![429i32, 111, 213, 111, 213, 429, 213];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();

    // The dictionary column should have the same number of rows.
    assert_eq!(dict_view.len().unwrap(), 7);
    // Dictionary type.
    assert_eq!(dict_view.data_type().unwrap().id(), TypeId::Dictionary32);

    // Keys: 3 unique values.
    let keys_view = dictionary::keys(&dict_view).unwrap();
    assert_eq!(keys_view.len().unwrap(), 3);
    assert_eq!(keys_view.data_type().unwrap().id(), TypeId::Int32);
    let keys = keys_view.to_vec::<i32>().unwrap();
    let mut sorted_keys = keys.clone();
    sorted_keys.sort();
    assert_eq!(sorted_keys, vec![111, 213, 429]);

    // Indices should correctly map each element back to its key.
    let indices_view = dictionary::indices(&dict_view).unwrap();
    assert_eq!(indices_view.len().unwrap(), 7);
    let indices = indices_view.to_vec::<i32>().unwrap();

    // Verify the mapping: data[i] == keys[indices[i]]
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(
            data[i], keys[idx as usize],
            "data[{}]={} should equal keys[{}]={}",
            i, data[i], idx, keys[idx as usize]
        );
    }
}

#[test]
fn test_dictionary_decode_roundtrip() {
    let data = vec![429i32, 111, 213, 111, 213, 429, 213];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    // Encode then decode.
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();
    let decoded = dictionary::decode(&dict_view).unwrap();
    let decoded_view = decoded.testing_view().unwrap();

    assert_eq!(decoded_view.len().unwrap(), 7);
    assert_eq!(decoded_view.data_type().unwrap().id(), TypeId::Int32);
    let result = decoded_view.to_vec::<i32>().unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_dictionary_encode_f64() {
    let data = vec![3.25f64, 2.75, 3.25, 1.5, 2.75];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();

    let keys_view = dictionary::keys(&dict_view).unwrap();
    assert_eq!(keys_view.len().unwrap(), 3); // 3 unique values
    let keys = keys_view.to_vec::<f64>().unwrap();

    // Verify all 3 expected values are present in keys.
    let mut sorted_keys = keys.clone();
    sorted_keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((sorted_keys[0] - 1.5).abs() < 1e-9);
    assert!((sorted_keys[1] - 2.75).abs() < 1e-9);
    assert!((sorted_keys[2] - 3.25).abs() < 1e-9);

    // Verify the mapping: data[i] == keys[indices[i]]
    let indices_view = dictionary::indices(&dict_view).unwrap();
    let indices = indices_view.to_vec::<i32>().unwrap();
    for (i, &idx) in indices.iter().enumerate() {
        assert!(
            (data[i] - keys[idx as usize]).abs() < 1e-9,
            "data[{}]={} should equal keys[{}]={}",
            i,
            data[i],
            idx,
            keys[idx as usize]
        );
    }
}

#[test]
fn test_dictionary_encode_empty() {
    let col = column_from_slice::<i32>(&[]).unwrap();
    let view = col.testing_view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();
    assert_eq!(dict_view.len().unwrap(), 0);
}

#[test]
fn test_dictionary_indices_device_ptr() {
    // Verify that the indices column's device pointer is non-null,
    // enabling zero-copy handoff to cugraph.
    let data = vec![10i32, 20, 10, 30, 20];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();

    let indices_view = dictionary::indices(&dict_view).unwrap();
    let ptr = indices_view.device_data_ptr().unwrap();
    assert!(!ptr.is_null(), "indices device pointer must be non-null");
}

#[test]
fn test_dictionary_set_keys_reindexes_and_drops_missing() {
    let data = vec![3i32, 1, 3, 2];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();

    // Key `1` is removed; key `4` is new.
    let new_keys = column_from_slice(&[2i32, 3, 4]).unwrap();
    let new_keys_view = new_keys.testing_view().unwrap();

    let updated = dictionary::set_keys(&dict_view, &new_keys_view).unwrap();
    let updated_view = updated.testing_view().unwrap();

    let decoded = dictionary::decode(&updated_view).unwrap();
    let decoded_view = decoded.testing_view().unwrap();

    let decoded_vals = decoded_view.to_vec::<i32>().unwrap();
    let validity = decoded_view.to_host_validity().unwrap().expect("removed keys should create nulls");

    assert_eq!(validity, vec![true, false, true, true]);

    let non_null_values: Vec<i32> =
        decoded_vals.iter().zip(validity.iter()).filter_map(|(v, valid)| valid.then_some(*v)).collect();
    assert_eq!(non_null_values, vec![3, 3, 2]);
}

#[test]
fn test_dictionary_set_keys_adds_new_keys_without_changing_values() {
    let data = vec![1i32, 2, 1, 2];
    let col = column_from_slice(&data).unwrap();
    let view = col.testing_view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();

    // Add a new key not present in data.
    let new_keys = column_from_slice(&[1i32, 2, 3]).unwrap();
    let new_keys_view = new_keys.testing_view().unwrap();

    let updated = dictionary::set_keys(&dict_view, &new_keys_view).unwrap();
    let updated_view = updated.testing_view().unwrap();

    let decoded = dictionary::decode(&updated_view).unwrap();
    let decoded_view = decoded.testing_view().unwrap();

    assert_eq!(decoded_view.null_count().unwrap(), 0);
    assert_eq!(decoded_view.to_vec::<i32>().unwrap(), data);
}

#[test]
fn test_dictionary_set_keys_type_mismatch_errors() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();

    // i64 keys do not match dictionary key type i32.
    let mismatched_keys = column_from_slice(&[1i64, 2, 3]).unwrap();
    let mismatched_keys_view = mismatched_keys.testing_view().unwrap();

    let err = match dictionary::set_keys(&dict_view, &mismatched_keys_view) {
        Ok(_) => panic!("expected type mismatch error"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "dictionary::set_keys");
    assert_eq!(err.fact_value(FactKeyName::LeftTypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(err.fact_value(FactKeyName::RightTypeId), Some(&FactValue::TypeId(TypeId::Int64)));
    assert_eq!(error_context_value(&err, "left_type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "right_type_id").as_deref(), Some("int64"));
}

#[test]
fn test_dictionary_set_keys_rejects_null_keys() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.testing_view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();

    let keys_with_null = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let keys_with_null_view = keys_with_null.testing_view().unwrap();

    let err = match dictionary::set_keys(&dict_view, &keys_with_null_view) {
        Ok(_) => panic!("expected invalid-operation error for null keys"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "dictionary::set_keys");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("keys"));
}

#[test]
fn test_dictionary_set_keys_rejects_non_dictionary_input() {
    let plain = column_from_slice(&[1i32, 2, 3]).unwrap();
    let new_keys = column_from_slice(&[1i32, 2, 3]).unwrap();

    let err = dictionary::set_keys(&plain.testing_view().unwrap(), &new_keys.testing_view().unwrap()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "dictionary::set_keys");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("dict_column"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
}

#[test]
fn test_match_dictionaries_requires_at_least_one_input() {
    let err = dictionary::match_dictionaries(&[]).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dictionary::match_dictionaries");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("empty_input"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("provide at least one input dictionary column"));
}

#[test]
fn test_match_dictionaries_rejects_non_dictionary_input() {
    let plain = column_from_slice(&[1i32, 2, 3]).unwrap();
    let dict =
        dictionary::encode(&column_from_slice(&[1i32, 2, 1]).unwrap().testing_view().unwrap(), TypeId::Int32).unwrap();

    let err =
        dictionary::match_dictionaries(&[&plain.testing_view().unwrap(), &dict.testing_view().unwrap()]).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "dictionary::match_dictionaries");
    assert_eq!(err.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(0)));
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(error_context_value(&err, "column_index").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
}

#[test]
fn test_match_dictionaries_testing_hook_locks_result_count_mismatch_contract() {
    let err = cudf::testing::dictionary::trigger_match_dictionaries_result_count_mismatch_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dictionary::match_dictionaries");
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(1)));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("dictionary_match_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("result_count_mismatch"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign dictionary match backend returns one output column per input dictionary column"),
    );
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "result_column_count").as_deref(), Some("1"));
    // Canonical keys are published as facts only — confirm they are not duplicated into legacy details.
    assert_eq!(err.detail_value("expected_count"), None);
    assert_eq!(err.detail_value("actual_count"), None);
}

#[test]
fn fatal_malformed_result_cleanup_preserves_primary_error_and_closes_device_admission() {
    let scope = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
    let err = cudf::testing::dictionary::trigger_match_dictionaries_fatal_malformed_cleanup_for_test()
        .expect_err("malformed dictionary result remains the primary error");

    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dictionary::match_dictionaries");
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("result_count_mismatch"));

    let device = cudf::DeviceId::DEFAULT;
    let snapshot = scope.health_snapshot(device);
    assert_eq!(snapshot.status(), cudf::runtime::DeviceHealthStatus::Unhealthy);
    assert_eq!(snapshot.fault_count(), 1);
    assert_eq!(snapshot.quarantined_teardown_count(), 1);
    let first = snapshot.first_fault().expect("fatal cleanup fault");
    assert_eq!(first.kind(), cudf::runtime::DeviceFaultKind::FatalDeviceOperation);
    assert_eq!(first.operation(), "dictionary::match_dictionaries::cleanup_malformed_result");
    assert_eq!(first.native_status(), Some(cudf_sys::CUDF_ERROR_FATAL_CUDA));

    let admission = cudf::runtime::DeviceEntryGuard::enter(device, "dictionary_cleanup::later_work")
        .expect_err("fatal malformed-result cleanup must close later device admission");
    assert_eq!(admission.code(), ErrorCode::FatalCuda);
    assert_eq!(scope.health_snapshot(device).rejected_operation_count(), 1);
}

#[test]
fn test_match_dictionaries_testing_hook_locks_null_result_array_contract() {
    let err = cudf::testing::dictionary::trigger_match_dictionaries_null_result_array_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "dictionary::match_dictionaries");
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(1)));
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(1)));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("dictionary_match_result_shape"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("null_result_array"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the foreign dictionary match backend returns one output column per input dictionary column"),
    );
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "result_column_count").as_deref(), Some("1"));
    // Canonical keys are published as facts only — confirm they are not duplicated into legacy details.
    assert_eq!(err.detail_value("expected_count"), None);
    assert_eq!(err.detail_value("actual_count"), None);
}

#[test]
fn test_match_dictionaries_testing_hook_locks_input_column_count_limit_contract() {
    let err = cudf::testing::dictionary::trigger_match_dictionaries_column_count_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "dictionary::match_dictionaries");
    assert_eq!(err.message(), "dictionary column count exceeds i32::MAX");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("table_column_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some(usize::MAX.to_string().as_str()));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("reduce table_column_count to fit within i32::MAX")
    );
    assert_eq!(error_context_value(&err, "num_columns").as_deref(), None);
}

#[test]
fn test_match_dictionaries_shared_key_space() {
    // src has keys {10, 20, 30}, dst has keys {20, 30, 40}.
    // After matching, both should share {10, 20, 30, 40} (4 unique keys).
    let src = column_from_slice(&[10i32, 20, 30, 10]).unwrap();
    let dst = column_from_slice(&[20i32, 30, 40, 20]).unwrap();

    let src_dict = dictionary::encode(&src.testing_view().unwrap(), TypeId::Int32).unwrap();
    let dst_dict = dictionary::encode(&dst.testing_view().unwrap(), TypeId::Int32).unwrap();

    let src_view = src_dict.testing_view().unwrap();
    let dst_view = dst_dict.testing_view().unwrap();

    let matched = dictionary::match_dictionaries(&[&src_view, &dst_view]).unwrap();
    assert_eq!(matched.len(), 2);

    // Both columns should share the same 4 keys.
    let matched_src_view = matched[0].testing_view().unwrap();
    let matched_dst_view = matched[1].testing_view().unwrap();

    let src_keys = dictionary::keys(&matched_src_view).unwrap();
    let dst_keys = dictionary::keys(&matched_dst_view).unwrap();

    assert_eq!(src_keys.len().unwrap(), 4);
    assert_eq!(dst_keys.len().unwrap(), 4);

    // Verify key sets are identical.
    let src_key_vals = src_keys.to_vec::<i32>().unwrap();
    let dst_key_vals = dst_keys.to_vec::<i32>().unwrap();
    assert_eq!(src_key_vals, dst_key_vals);

    // Verify that decoding produces the original values.
    let decoded_src = dictionary::decode(&matched_src_view).unwrap();
    let decoded_dst = dictionary::decode(&matched_dst_view).unwrap();

    assert_eq!(decoded_src.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30, 10]);
    assert_eq!(decoded_dst.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40, 20]);

    // Verify that the same value maps to the same index in both columns.
    let src_indices = dictionary_indices_to_i64(&matched_src_view);
    let dst_indices = dictionary_indices_to_i64(&matched_dst_view);

    // Value 20 appears in both: src[1]=20, dst[0]=20 → same index.
    assert_eq!(src_indices[1], dst_indices[0]);
    // Value 30 appears in both: src[2]=30, dst[1]=30 → same index.
    assert_eq!(src_indices[2], dst_indices[1]);
}

#[test]
fn test_match_dictionaries_no_overlap() {
    // Completely disjoint key sets.
    let a = column_from_slice(&[1i32, 2]).unwrap();
    let b = column_from_slice(&[3i32, 4]).unwrap();

    let a_dict = dictionary::encode(&a.testing_view().unwrap(), TypeId::Int32).unwrap();
    let b_dict = dictionary::encode(&b.testing_view().unwrap(), TypeId::Int32).unwrap();

    let a_view = a_dict.testing_view().unwrap();
    let b_view = b_dict.testing_view().unwrap();

    let matched = dictionary::match_dictionaries(&[&a_view, &b_view]).unwrap();

    let a_keys = dictionary::keys(&matched[0].testing_view().unwrap()).unwrap();
    let b_keys = dictionary::keys(&matched[1].testing_view().unwrap()).unwrap();
    assert_eq!(a_keys.len().unwrap(), 4);
    assert_eq!(b_keys.len().unwrap(), 4);

    // Decoding still produces the original values.
    let decoded_a = dictionary::decode(&matched[0].testing_view().unwrap()).unwrap();
    let decoded_b = dictionary::decode(&matched[1].testing_view().unwrap()).unwrap();
    assert_eq!(decoded_a.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(decoded_b.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![3, 4]);
}

#[test]
fn test_dictionary_set_keys_succeeds_with_mixed_domains_and_orders_secondary_input_before_authoritative_exec() {
    let _event_observation = crate::common::lock_event_observation();
    let dict = dictionary::encode(&column_from_slice(&[3i32, 1, 3, 2]).unwrap().testing_view().unwrap(), TypeId::Int32)
        .unwrap()
        .testing_rebind_execution_domain_explicit(601)
        .unwrap();
    let keys = column_from_slice(&[2i32, 3, 4]).unwrap().testing_rebind_execution_domain_explicit(602).unwrap();
    let expected_exec_stream = dict.testing_execution_domain_stream_value();
    let expected_exec_mr = dict.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let updated = dictionary::set_keys(&dict.testing_view().unwrap(), &keys.testing_view().unwrap()).unwrap();

    assert_eq!(updated.testing_execution_domain_id(), dict.testing_execution_domain_id());
    assert_eq!(updated.testing_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    let mut updated_keys = dictionary::keys(&updated.testing_view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    updated_keys.sort_unstable();
    assert_eq!(updated_keys, vec![2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_match_dictionaries_succeeds_with_mixed_domains_and_orders_secondary_inputs_before_first_input_exec() {
    let _event_observation = crate::common::lock_event_observation();
    let first =
        dictionary::encode(&column_from_slice(&[10i32, 20, 30]).unwrap().testing_view().unwrap(), TypeId::Int32)
            .unwrap()
            .testing_rebind_execution_domain_explicit(603)
            .unwrap();
    let second =
        dictionary::encode(&column_from_slice(&[20i32, 30, 40]).unwrap().testing_view().unwrap(), TypeId::Int32)
            .unwrap()
            .testing_rebind_execution_domain_explicit(604)
            .unwrap();
    let expected_exec_stream = first.testing_execution_domain_stream_value();
    let expected_exec_mr = first.testing_execution_domain_mr_value();
    crate::common::reset_dependency_observation();
    reset_exec_observation();
    let matched =
        dictionary::match_dictionaries(&[&first.testing_view().unwrap(), &second.testing_view().unwrap()]).unwrap();

    assert_eq!(matched.len(), 2);
    assert!(matched.iter().all(|column| column.testing_execution_domain_id() == first.testing_execution_domain_id()));
    assert!(matched.iter().all(|column| column.testing_execution_domain_mr_value() == expected_exec_mr));
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(crate::common::dependency_wait_call_count_for_stream(expected_exec_stream), 1);

    reset_sync_observation();
    let mut first_keys = dictionary::keys(&matched[0].testing_view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    let mut second_keys = dictionary::keys(&matched[1].testing_view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    first_keys.sort_unstable();
    second_keys.sort_unstable();
    assert_eq!(first_keys, second_keys);
    assert_eq!(first_keys, vec![10, 20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}
