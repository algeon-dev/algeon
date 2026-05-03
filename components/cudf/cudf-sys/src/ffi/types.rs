use crate::{cudf_data_type_handle, cudf_status_t, cudf_type_id_t};

extern "C" {
    // -- DataType ------------------------------------------------------------

    pub fn cudf_data_type_create(type_id: cudf_type_id_t, out: *mut cudf_data_type_handle) -> cudf_status_t;

    pub fn cudf_data_type_create_fixed_point(
        type_id: cudf_type_id_t,
        scale: i32,
        out: *mut cudf_data_type_handle,
    ) -> cudf_status_t;

    pub fn cudf_data_type_destroy(handle: cudf_data_type_handle);

    pub fn cudf_data_type_id(handle: cudf_data_type_handle, out: *mut cudf_type_id_t) -> cudf_status_t;

    pub fn cudf_data_type_scale(handle: cudf_data_type_handle, out: *mut i32) -> cudf_status_t;

    pub fn cudf_type_size(type_id: cudf_type_id_t, out: *mut i32) -> cudf_status_t;
}
