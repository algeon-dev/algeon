use std::{
    mem::{align_of, size_of},
    os::raw::c_uint,
};

use cugraph_sys::{
    bool_t, cugraph_data_mask_t, cugraph_data_type_id_t, cugraph_error_code_t, cugraph_error_t,
    cugraph_graph_properties_t, cugraph_graph_t, cugraph_resource_handle_t, cugraph_rng_state_t,
    cugraph_type_erased_device_array_t, cugraph_type_erased_device_array_view_t, cugraph_type_erased_host_array_t,
    cugraph_type_erased_host_array_view_t,
};

macro_rules! assert_opaque_i32_layout {
    ($ty:ty) => {
        assert_eq!(size_of::<$ty>(), size_of::<i32>(), "size mismatch for {}", stringify!($ty));
        assert_eq!(align_of::<$ty>(), align_of::<i32>(), "alignment mismatch for {}", stringify!($ty));
    };
}

#[test]
fn enum_newtypes_match_c_uint_layout() {
    assert_eq!(size_of::<cugraph_error_code_t>(), size_of::<c_uint>());
    assert_eq!(align_of::<cugraph_error_code_t>(), align_of::<c_uint>());

    assert_eq!(size_of::<bool_t>(), size_of::<c_uint>());
    assert_eq!(align_of::<bool_t>(), align_of::<c_uint>());

    assert_eq!(size_of::<cugraph_data_type_id_t>(), size_of::<c_uint>());
    assert_eq!(align_of::<cugraph_data_type_id_t>(), align_of::<c_uint>());
}

#[test]
fn opaque_handle_like_types_match_int32_layout() {
    assert_opaque_i32_layout!(cugraph_error_t);
    assert_opaque_i32_layout!(cugraph_resource_handle_t);
    assert_opaque_i32_layout!(cugraph_graph_t);
    assert_opaque_i32_layout!(cugraph_data_mask_t);
    assert_opaque_i32_layout!(cugraph_rng_state_t);

    assert_opaque_i32_layout!(cugraph_type_erased_device_array_t);
    assert_opaque_i32_layout!(cugraph_type_erased_device_array_view_t);
    assert_opaque_i32_layout!(cugraph_type_erased_host_array_t);
    assert_opaque_i32_layout!(cugraph_type_erased_host_array_view_t);
}

#[test]
fn graph_properties_layout_matches_two_bool_t_fields() {
    assert_eq!(size_of::<cugraph_graph_properties_t>(), 2 * size_of::<bool_t>());
    assert_eq!(align_of::<cugraph_graph_properties_t>(), align_of::<bool_t>());
}
