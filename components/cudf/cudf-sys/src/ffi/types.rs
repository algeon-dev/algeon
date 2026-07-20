use crate::{cudf_data_type_handle, cudf_status_t, cudf_type_id_t};

unsafe extern "C" {
    // -- DataType ------------------------------------------------------------

    pub fn cudf_data_type_create(type_id: cudf_type_id_t, out: *mut cudf_data_type_handle) -> cudf_status_t;

    pub fn cudf_data_type_create_fixed_point(
        type_id: cudf_type_id_t,
        scale: i32,
        out: *mut cudf_data_type_handle,
    ) -> cudf_status_t;

    pub fn cudf_data_type_destroy(handle: cudf_data_type_handle) -> cudf_status_t;

    /// `out` receives a raw `cudf_type_id_t` discriminant. Declared as `i32`
    /// so an unmirrored libcudf type id never materializes an invalid Rust
    /// enum; callers must validate before converting.
    pub fn cudf_data_type_id(handle: cudf_data_type_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_data_type_scale(handle: cudf_data_type_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_type_size(type_id: cudf_type_id_t, out: *mut i32) -> cudf_status_t;
}
