use super::types;

pub(crate) fn validity_to_host_null_mask_bytes(validity: &[bool]) -> Vec<u8> {
    types::validity_to_ffi_bitmask(validity)
}
