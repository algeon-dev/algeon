fn main() {
    let _ = cudf::__debug_validity_to_host_null_mask_bytes(&[true, false, true]);
}
