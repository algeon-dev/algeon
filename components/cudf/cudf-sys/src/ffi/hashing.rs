use crate::{cudf_column_handle, cudf_exec_options_t, cudf_status_t, cudf_table_view_handle};

extern "C" {
    // -- Hashing -------------------------------------------------------------

    pub fn cudf_hashing_murmurhash3_x86_32(
        input: cudf_table_view_handle,
        seed: u32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hashing_murmurhash3_x86_32_ex(
        input: cudf_table_view_handle,
        seed: u32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hashing_xxhash_64(
        input: cudf_table_view_handle,
        seed: u64,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hashing_xxhash_64_ex(
        input: cudf_table_view_handle,
        seed: u64,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
