#[test]
fn host_nullable_mask_bytes_are_padded_to_cudf_allocation_size() {
    let mask = cudf::__debug_validity_to_host_null_mask_bytes(&[true, false, true]);

    assert_eq!(mask.len(), 64, "cuDF host null-mask buffers must use allocation-sized padding");
    assert_eq!(mask[0], 0x05, "validity bits should remain packed LSB-first");
    assert!(mask[1..].iter().all(|&byte| byte == 0), "padding bytes should be zeroed");
}

#[test]
fn host_nullable_mask_bytes_for_empty_validity_are_empty() {
    let mask = cudf::__debug_validity_to_host_null_mask_bytes(&[]);
    assert!(mask.is_empty());
}
