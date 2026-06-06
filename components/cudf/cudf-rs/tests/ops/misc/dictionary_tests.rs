//! Tests for dictionary encode/decode.

use cudf::{Column, ErrorKind, TypeId, ops::dictionary};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
};

#[test]
fn test_dictionary_encode_i32() {
    let data = vec![429i32, 111, 213, 111, 213, 429, 213];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();

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
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    // Encode then decode.
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();
    let decoded = dictionary::decode(&dict_view).unwrap();
    let decoded_view = decoded.view().unwrap();

    assert_eq!(decoded_view.len().unwrap(), 7);
    assert_eq!(decoded_view.data_type().unwrap().id(), TypeId::Int32);
    let result = decoded_view.to_vec::<i32>().unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_dictionary_encode_f64() {
    let data = vec![3.25f64, 2.75, 3.25, 1.5, 2.75];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();

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
    let col = Column::from_slice::<i32>(&[]).unwrap();
    let view = col.view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();
    assert_eq!(dict_view.len().unwrap(), 0);
}

#[test]
fn test_dictionary_indices_device_ptr() {
    // Verify that the indices column's device pointer is non-null,
    // enabling zero-copy handoff to cugraph.
    let data = vec![10i32, 20, 10, 30, 20];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();

    let indices_view = dictionary::indices(&dict_view).unwrap();
    let ptr = indices_view.device_data_ptr().unwrap();
    assert!(!ptr.is_null(), "indices device pointer must be non-null");
}

#[test]
fn test_dictionary_set_keys_reindexes_and_drops_missing() {
    let data = vec![3i32, 1, 3, 2];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();

    // Key `1` is removed; key `4` is new.
    let new_keys = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let new_keys_view = new_keys.view().unwrap();

    let updated = dictionary::set_keys(&dict_view, &new_keys_view).unwrap();
    let updated_view = updated.view().unwrap();

    let decoded = dictionary::decode(&updated_view).unwrap();
    let decoded_view = decoded.view().unwrap();

    let decoded_vals = decoded_view.to_vec::<i32>().unwrap();
    let validity = decoded_view.validity().unwrap().expect("removed keys should create nulls");

    assert_eq!(validity, vec![true, false, true, true]);

    let non_null_values: Vec<i32> =
        decoded_vals.iter().zip(validity.iter()).filter_map(|(v, valid)| valid.then_some(*v)).collect();
    assert_eq!(non_null_values, vec![3, 3, 2]);
}

#[test]
fn test_dictionary_set_keys_adds_new_keys_without_changing_values() {
    let data = vec![1i32, 2, 1, 2];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();

    // Add a new key not present in data.
    let new_keys = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let new_keys_view = new_keys.view().unwrap();

    let updated = dictionary::set_keys(&dict_view, &new_keys_view).unwrap();
    let updated_view = updated.view().unwrap();

    let decoded = dictionary::decode(&updated_view).unwrap();
    let decoded_view = decoded.view().unwrap();

    assert_eq!(decoded_view.null_count().unwrap(), 0);
    assert_eq!(decoded_view.to_vec::<i32>().unwrap(), data);
}

#[test]
fn test_dictionary_set_keys_type_mismatch_errors() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();

    // i64 keys do not match dictionary key type i32.
    let mismatched_keys = Column::from_slice(&[1i64, 2, 3]).unwrap();
    let mismatched_keys_view = mismatched_keys.view().unwrap();

    let err = match dictionary::set_keys(&dict_view, &mismatched_keys_view) {
        Ok(_) => panic!("expected type mismatch error"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::TypeError);
}

#[test]
fn test_dictionary_set_keys_rejects_null_keys() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();

    let keys_with_null = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let keys_with_null_view = keys_with_null.view().unwrap();

    let err = match dictionary::set_keys(&dict_view, &keys_with_null_view) {
        Ok(_) => panic!("expected invalid-operation error for null keys"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_match_dictionaries_shared_key_space() {
    // src has keys {10, 20, 30}, dst has keys {20, 30, 40}.
    // After matching, both should share {10, 20, 30, 40} (4 unique keys).
    let src = Column::from_slice(&[10i32, 20, 30, 10]).unwrap();
    let dst = Column::from_slice(&[20i32, 30, 40, 20]).unwrap();

    let src_dict = dictionary::encode(&src.view().unwrap(), TypeId::Int32).unwrap();
    let dst_dict = dictionary::encode(&dst.view().unwrap(), TypeId::Int32).unwrap();

    let src_view = src_dict.view().unwrap();
    let dst_view = dst_dict.view().unwrap();

    let matched = dictionary::match_dictionaries(&[&src_view, &dst_view]).unwrap();
    assert_eq!(matched.len(), 2);

    // Both columns should share the same 4 keys.
    let matched_src_view = matched[0].view().unwrap();
    let matched_dst_view = matched[1].view().unwrap();

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

    assert_eq!(decoded_src.view().unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30, 10]);
    assert_eq!(decoded_dst.view().unwrap().to_vec::<i32>().unwrap(), vec![20, 30, 40, 20]);

    // Verify that the same value maps to the same index in both columns.
    let src_indices = dictionary::indices(&matched_src_view).unwrap().to_vec::<i32>().unwrap();
    let dst_indices = dictionary::indices(&matched_dst_view).unwrap().to_vec::<i32>().unwrap();

    // Value 20 appears in both: src[1]=20, dst[0]=20 → same index.
    assert_eq!(src_indices[1], dst_indices[0]);
    // Value 30 appears in both: src[2]=30, dst[1]=30 → same index.
    assert_eq!(src_indices[2], dst_indices[1]);
}

#[test]
fn test_match_dictionaries_no_overlap() {
    // Completely disjoint key sets.
    let a = Column::from_slice(&[1i32, 2]).unwrap();
    let b = Column::from_slice(&[3i32, 4]).unwrap();

    let a_dict = dictionary::encode(&a.view().unwrap(), TypeId::Int32).unwrap();
    let b_dict = dictionary::encode(&b.view().unwrap(), TypeId::Int32).unwrap();

    let a_view = a_dict.view().unwrap();
    let b_view = b_dict.view().unwrap();

    let matched = dictionary::match_dictionaries(&[&a_view, &b_view]).unwrap();

    let a_keys = dictionary::keys(&matched[0].view().unwrap()).unwrap();
    let b_keys = dictionary::keys(&matched[1].view().unwrap()).unwrap();
    assert_eq!(a_keys.len().unwrap(), 4);
    assert_eq!(b_keys.len().unwrap(), 4);

    // Decoding still produces the original values.
    let decoded_a = dictionary::decode(&matched[0].view().unwrap()).unwrap();
    let decoded_b = dictionary::decode(&matched[1].view().unwrap()).unwrap();
    assert_eq!(decoded_a.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2]);
    assert_eq!(decoded_b.view().unwrap().to_vec::<i32>().unwrap(), vec![3, 4]);
}

#[test]
fn test_dictionary_set_keys_succeeds_with_mixed_domains_and_syncs_secondary_input_before_authoritative_exec() {
    let dict = dictionary::encode(&Column::from_slice(&[3i32, 1, 3, 2]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(601)
        .unwrap();
    let keys = Column::from_slice(&[2i32, 3, 4]).unwrap().__debug_rebind_execution_domain_explicit(602).unwrap();
    let expected_exec_stream = dict.__debug_execution_domain_stream_value();
    let expected_exec_mr = dict.__debug_execution_domain_mr_value();
    let expected_presync_stream = keys.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let updated = dictionary::set_keys(&dict.view().unwrap(), &keys.view().unwrap()).unwrap();

    assert_eq!(updated.__debug_execution_domain_id(), dict.__debug_execution_domain_id());
    assert_eq!(updated.__debug_execution_domain_mr_value(), expected_exec_mr);
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_presync_stream);

    reset_sync_observation();
    let mut updated_keys = dictionary::keys(&updated.view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    updated_keys.sort_unstable();
    assert_eq!(updated_keys, vec![2, 3, 4]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}

#[test]
fn test_match_dictionaries_succeeds_with_mixed_domains_and_syncs_secondary_inputs_before_first_input_exec() {
    let first = dictionary::encode(&Column::from_slice(&[10i32, 20, 30]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(603)
        .unwrap();
    let second = dictionary::encode(&Column::from_slice(&[20i32, 30, 40]).unwrap().view().unwrap(), TypeId::Int32)
        .unwrap()
        .__debug_rebind_execution_domain_explicit(604)
        .unwrap();
    let expected_exec_stream = first.__debug_execution_domain_stream_value();
    let expected_exec_mr = first.__debug_execution_domain_mr_value();
    let expected_presync_stream = second.__debug_execution_domain_stream_value();

    reset_exec_observation();
    let matched = dictionary::match_dictionaries(&[&first.view().unwrap(), &second.view().unwrap()]).unwrap();

    assert_eq!(matched.len(), 2);
    assert!(matched.iter().all(|column| column.__debug_execution_domain_id() == first.__debug_execution_domain_id()));
    assert!(matched.iter().all(|column| column.__debug_execution_domain_mr_value() == expected_exec_mr));
    assert_eq!(last_exec_stream_value(), expected_exec_stream);
    assert_eq!(last_exec_mr_value(), expected_exec_mr);
    assert_eq!(last_sync_stream_value(), expected_presync_stream);

    reset_sync_observation();
    let mut first_keys = dictionary::keys(&matched[0].view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    let mut second_keys = dictionary::keys(&matched[1].view().unwrap()).unwrap().to_vec::<i32>().unwrap();
    first_keys.sort_unstable();
    second_keys.sort_unstable();
    assert_eq!(first_keys, second_keys);
    assert_eq!(first_keys, vec![10, 20, 30, 40]);
    assert_eq!(last_sync_stream_value(), expected_exec_stream);
}
