use crate::{cudf_exec_options_t, cudf_status_t, cudf_table_handle, cudf_table_view_handle};

extern "C" {
    /// Hash-partition rows with explicit execution options.
    ///
    /// `options` must be non-null, name an explicit valid device, provide a
    /// non-null stream, and provide a non-null memory resource. The memory
    /// resource must outlive the returned table.
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
