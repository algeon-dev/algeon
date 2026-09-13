use super::types;

/// Pack host validity bits into cuDF's allocation-sized null-mask layout.
pub fn validity_to_host_null_mask_bytes(validity: &[bool]) -> Vec<u8> {
    types::validity_to_ffi_bitmask(validity)
}
