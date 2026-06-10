use std::os::raw::{c_char, c_void};

use crate::{
    cudf_column_handle, cudf_column_view_handle, cudf_exec_options_t, cudf_interop_buffer_binding_t,
    cudf_interop_semantic_column_t, cudf_scalar_handle, cudf_status_t, cudf_type_id_t,
};

extern "C" {
    // -- Column (owning) -----------------------------------------------------

    pub fn cudf_column_create_empty_ex(
        type_id: cudf_type_id_t,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a fixed-width column by copying from borrowed host buffers with
    /// explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    ///
    /// If host-source copies are queued, the selected stream is synchronized
    /// before return on success or failure so `data` and `null_mask` are no
    /// longer in use.
    pub fn cudf_column_create_numeric_ex(
        type_id: cudf_type_id_t,
        data: *const c_void,
        size: i32,
        null_mask: *const u8,
        null_count: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a fixed-width column by copying from borrowed host buffers with
    /// an explicit logical dtype and execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    ///
    /// If host-source copies are queued, the selected stream is synchronized
    /// before return on success or failure so `data` and `null_mask` are no
    /// longer in use.
    pub fn cudf_column_create_fixed_width_from_host_ex(
        type_id: cudf_type_id_t,
        scale: i32,
        data: *const c_void,
        size: i32,
        null_mask: *const u8,
        null_count: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a fixed-width sequence column with explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    pub fn cudf_column_create_sequence_ex(
        init: cudf_scalar_handle,
        step: cudf_scalar_handle,
        size: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Deep-copy a column with explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    pub fn cudf_column_copy_ex(
        src: cudf_column_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Deep-copy a column view with explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    pub fn cudf_column_copy_from_view_ex(
        src: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Deep-copy borrowed interop-view facts with explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    pub fn cudf_column_copy_from_borrowed_view_ex(
        type_id: cudf_type_id_t,
        scale: i32,
        size: i32,
        offset: i32,
        data: *const c_void,
        null_mask: *const c_void,
        null_count: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Deep-copy a prepared-device export using strict explicit execution
    /// options.
    ///
    /// `options` must name an explicit valid device, non-null stream, and
    /// non-null memory resource. The memory resource must outlive the returned
    /// column handle.
    pub fn cudf_column_copy_from_prepared_device_export_ex(
        root_column_id: u32,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Deep-copy a prepared-host export using strict explicit execution
    /// options.
    ///
    /// `options` must name an explicit valid device, non-null stream, and
    /// non-null memory resource. The memory resource must outlive the returned
    /// column handle.
    pub fn cudf_column_copy_from_prepared_host_export_ex(
        root_column_id: u32,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_create_from_owned_device_transfer(
        root_column_id: u32,
        semantic_columns: *const cudf_interop_semantic_column_t,
        num_semantic_columns: i32,
        bindings: *const cudf_interop_buffer_binding_t,
        num_bindings: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Return payload bytes with explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource.
    pub fn cudf_column_payload_size_ex(
        handle: cudf_column_handle,
        options: *const cudf_exec_options_t,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_column_device_allocated_size(handle: cudf_column_handle, out: *mut usize) -> cudf_status_t;

    pub fn cudf_column_destroy(handle: cudf_column_handle);

    // -- ColumnView (non-owning) --------------------------------------------

    pub fn cudf_column_view(handle: cudf_column_handle, out: *mut cudf_column_view_handle) -> cudf_status_t;

    pub fn cudf_column_view_destroy(handle: cudf_column_view_handle);

    pub fn cudf_column_view_type(handle: cudf_column_view_handle, out: *mut cudf_type_id_t) -> cudf_status_t;

    pub fn cudf_column_view_data_type(
        handle: cudf_column_view_handle,
        out_type_id: *mut cudf_type_id_t,
        out_scale: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_column_view_size(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_null_count(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;
    pub fn cudf_column_view_null_count_ex(
        handle: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out: *mut i32,
    ) -> cudf_status_t;

    pub fn cudf_column_view_nullable(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_num_children(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_column_view_child(
        handle: cudf_column_view_handle,
        index: i32,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    /// Return a LIST sliced-child view with explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource.
    pub fn cudf_column_view_list_sliced_child_ex(
        handle: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    pub fn cudf_column_view_data_ptr(handle: cudf_column_view_handle, out: *mut *const c_void) -> cudf_status_t;

    /// Return the STRING chars base pointer and the byte span it must cover,
    /// with explicit execution options.
    ///
    /// `*out_ptr` is the base of the chars child buffer; the view's offsets
    /// index from that base, so for sliced parent views `*out_size` ends at
    /// the last byte referenced by the logical slice (it is NOT the active
    /// byte count `last_offset - first_offset`).
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource.
    pub fn cudf_column_view_string_chars_ptr_and_size_ex(
        handle: cudf_column_view_handle,
        options: *const cudf_exec_options_t,
        out_ptr: *mut *const c_void,
        out_size: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_column_view_null_mask_ptr(handle: cudf_column_view_handle, out: *mut *const c_void) -> cudf_status_t;

    pub fn cudf_column_view_offset(handle: cudf_column_view_handle, out: *mut i32) -> cudf_status_t;

    /// Create a sliced borrowed column view with explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. Nullable slices resolve
    /// their active-range null count on the selected stream before returning.
    pub fn cudf_column_view_slice_ex(
        handle: cudf_column_view_handle,
        offset: i32,
        length: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_view_handle,
    ) -> cudf_status_t;

    /// Materialize fixed-width column data into a host buffer with explicit
    /// execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The selected stream is
    /// synchronized before return so the host buffer is populated on success.
    pub fn cudf_column_view_to_host_ex(
        handle: cudf_column_view_handle,
        host_buf: *mut c_void,
        buf_size: usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Materialize a column null mask into a host buffer with explicit
    /// execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The selected stream is
    /// synchronized before return so the host buffer is populated on success.
    /// For sliced views, the exported host bitmask is normalized to the logical
    /// slice so the view's first row maps to bit 0.
    pub fn cudf_column_view_null_mask_to_host_ex(
        handle: cudf_column_view_handle,
        host_buf: *mut u8,
        buf_size: usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Legacy non-explicit device-pointer copy wrapper.
    ///
    /// Execution-sensitive callers must use
    /// `cudf_column_create_from_device_data_ex`; this wrapper rejects calls
    /// because explicit execution options are required.
    pub fn cudf_column_create_from_device_data(
        type_id: cudf_type_id_t,
        device_data: *const c_void,
        size: i32,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a column by copying from a borrowed device pointer with
    /// explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    ///
    /// Unlike [`cudf_column_create_from_device_data`], this `_ex` variant does
    /// not synchronize before returning. The caller must keep `device_data`
    /// valid until the selected stream has completed the copy and must
    /// establish any required cross-stream ordering for downstream consumers.
    pub fn cudf_column_create_from_device_data_ex(
        type_id: cudf_type_id_t,
        device_data: *const c_void,
        size: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Create a column by repeating a scalar value with explicit execution
    /// options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    pub fn cudf_column_from_scalar_ex(
        scalar: cudf_scalar_handle,
        size: i32,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    /// Create a STRING column by copying from borrowed host buffers with
    /// explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource. The memory resource must
    /// outlive the returned column.
    ///
    /// If host-source copies are queued, the selected stream is synchronized
    /// before return on success or failure so the borrowed host buffers are no
    /// longer in use.
    pub fn cudf_column_create_strings_ex(
        strings: *const *const c_char,
        lengths: *const i32,
        num_strings: i32,
        null_mask: *const u8,
        null_count: i32,
        options: *const cudf_exec_options_t,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Materialize a STRING column into caller-provided host buffers with
    /// explicit execution options.
    ///
    /// `options` must be non-null and must name an explicit valid device,
    /// non-null stream, and non-null memory resource.
    ///
    /// Empty STRING columns still require `offsets_buf`; they write exactly
    /// one zero offset (`offsets_buf[0] = 0`) and report
    /// `*total_chars_bytes = 0`. In that case `chars_buf` is ignored because
    /// there are no chars to copy.
    ///
    /// The selected stream is synchronized before return whenever offsets or
    /// chars are materialized into the caller's host buffers.
    pub fn cudf_column_view_strings_to_host_ex(
        handle: cudf_column_view_handle,
        offsets_buf: *mut i32,
        offsets_buf_size: usize,
        chars_buf: *mut c_char,
        chars_buf_size: usize,
        total_chars_bytes: *mut i32,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
