use crate::{cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle};

extern "C" {
    pub fn cudf_hash_partition(
        input: cudf_table_view_handle,
        column_indices: *const i32,
        num_columns: i32,
        num_partitions: i32,
        out: *mut cudf_table_handle,
        partition_offsets: *mut i32,
        partition_offsets_len: usize,
    ) -> cudf_status_t;

    pub fn cudf_hash_partition_ex(
        input: cudf_table_view_handle,
        column_indices: *const i32,
        num_columns: i32,
        num_partitions: i32,
        out: *mut cudf_table_handle,
        partition_offsets: *mut i32,
        partition_offsets_len: usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
