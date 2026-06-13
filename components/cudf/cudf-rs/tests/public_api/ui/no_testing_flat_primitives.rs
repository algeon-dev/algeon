fn main() {
    let _ = cudf::testing::stats_memory_resource();
    let _ = cudf::testing::validity_to_host_null_mask_bytes(&[true, false, true]);
}
