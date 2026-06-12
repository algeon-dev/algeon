use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{helpers::*, support::lock_shim_state};

fn export_column_arrow_device(view: *mut c_void) -> (*mut c_void, *mut c_void) {
    let mut schema = ptr::null_mut();
    let mut array = ptr::null_mut();
    let mut materialization =
        cudf_sys::cudf_arrow_device_export_materialization_t::CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;

    let status = unsafe {
        cudf_sys::cudf_arrow_device_column_export(view, ptr::null(), &mut schema, &mut array, &mut materialization)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!schema.is_null());
    assert!(!array.is_null());
    (schema, array)
}

fn export_table_arrow_device(view: *mut c_void) -> (*mut c_void, *mut c_void) {
    let mut schema = ptr::null_mut();
    let mut array = ptr::null_mut();
    let mut materialization =
        cudf_sys::cudf_arrow_device_export_materialization_t::CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;

    let status = unsafe {
        cudf_sys::cudf_arrow_device_table_export(
            view,
            ptr::null(),
            ptr::null(),
            0,
            &mut schema,
            &mut array,
            &mut materialization,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!schema.is_null());
    assert!(!array.is_null());
    (schema, array)
}

#[repr(C)]
struct ArrowSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut ArrowSchema,
    dictionary: *mut ArrowSchema,
    release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    private_data: *mut c_void,
}

#[repr(C)]
struct ArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut ArrowArray,
    dictionary: *mut ArrowArray,
    release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    private_data: *mut c_void,
}

#[repr(C)]
struct ArrowDeviceArray {
    array: ArrowArray,
    device_id: i64,
    device_type: i32,
    sync_event: *mut c_void,
    reserved: [i64; 3],
}

fn synthetic_schema(format: &'static [u8]) -> ArrowSchema {
    ArrowSchema {
        format: format.as_ptr().cast(),
        name: ptr::null(),
        metadata: ptr::null(),
        flags: 0,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: None,
        private_data: ptr::null_mut(),
    }
}

fn synthetic_device_array() -> ArrowDeviceArray {
    ArrowDeviceArray {
        array: ArrowArray {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        },
        device_id: 0,
        device_type: cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA,
        sync_event: ptr::null_mut(),
        reserved: [0; 3],
    }
}

fn assert_synthetic_column_fixed_width_ptr_rejected(schema: &mut ArrowSchema) {
    let mut array = synthetic_device_array();
    let mut out = std::ptr::dangling::<c_void>();
    let status = unsafe {
        cudf_sys::cudf_arrow_device_column_fixed_width_values_ptr(
            ptr::from_mut(schema).cast(),
            ptr::from_mut(&mut array).cast(),
            &mut out,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
}

fn assert_synthetic_table_fixed_width_ptr_rejected(child_schema: &mut ArrowSchema) {
    let mut child_array = ArrowArray {
        length: 0,
        null_count: 0,
        offset: 0,
        n_buffers: 0,
        n_children: 0,
        buffers: ptr::null_mut(),
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: None,
        private_data: ptr::null_mut(),
    };
    let mut child_schemas = [ptr::from_mut(child_schema)];
    let mut child_arrays = [ptr::from_mut(&mut child_array)];
    let mut root_schema = synthetic_schema(b"+s\0");
    root_schema.n_children = 1;
    root_schema.children = child_schemas.as_mut_ptr();

    let mut root_array = synthetic_device_array();
    root_array.array.n_children = 1;
    root_array.array.children = child_arrays.as_mut_ptr();

    let mut out = std::ptr::dangling::<c_void>();
    let status = unsafe {
        cudf_sys::cudf_arrow_device_table_fixed_width_values_ptr(
            ptr::from_mut(&mut root_schema).cast(),
            ptr::from_mut(&mut root_array).cast(),
            0,
            &mut out,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
}

#[test]
fn test_arrow_device_column_fixed_width_values_ptr_rejects_string_schema() {
    let _shim_lock = lock_shim_state();
    let column = make_string_column(&["alpha", "beta"]);
    let view = make_column_view(column);
    let (schema, array) = export_column_arrow_device(view);
    let mut out = std::ptr::dangling::<c_void>();

    let status = unsafe { cudf_sys::cudf_arrow_device_column_fixed_width_values_ptr(schema, array, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_arrow_device_schema_destroy(schema);
        cudf_sys::cudf_arrow_device_array_destroy(array);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(column);
    }
}

#[test]
fn test_arrow_device_table_fixed_width_values_ptr_rejects_string_column() {
    let _shim_lock = lock_shim_state();
    let words = make_string_column(&["alpha", "beta"]);
    let ids = make_i32_column(&[1, 2]);
    let table = make_table_from_columns(vec![words, ids]);
    let view = make_table_view(table);
    let (schema, array) = export_table_arrow_device(view);
    let mut out = std::ptr::dangling::<c_void>();

    let status = unsafe { cudf_sys::cudf_arrow_device_table_fixed_width_values_ptr(schema, array, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_arrow_device_schema_destroy(schema);
        cudf_sys::cudf_arrow_device_array_destroy(array);
        cudf_sys::cudf_table_view_destroy(view);
        cudf_sys::cudf_table_destroy(table);
    }
}

#[test]
fn test_arrow_device_column_fixed_width_values_ptr_rejects_bool_list_struct_and_dictionary_schemas() {
    let _shim_lock = lock_shim_state();

    let mut bool_schema = synthetic_schema(b"b\0");
    assert_synthetic_column_fixed_width_ptr_rejected(&mut bool_schema);

    let mut list_schema = synthetic_schema(b"+l\0");
    list_schema.n_children = 1;
    assert_synthetic_column_fixed_width_ptr_rejected(&mut list_schema);

    let mut struct_schema = synthetic_schema(b"+s\0");
    struct_schema.n_children = 1;
    assert_synthetic_column_fixed_width_ptr_rejected(&mut struct_schema);

    let mut dictionary_values = synthetic_schema(b"u\0");
    let mut dictionary_schema = synthetic_schema(b"i\0");
    dictionary_schema.dictionary = ptr::from_mut(&mut dictionary_values);
    assert_synthetic_column_fixed_width_ptr_rejected(&mut dictionary_schema);
}

#[test]
fn test_arrow_device_table_fixed_width_values_ptr_rejects_bool_list_struct_and_dictionary_columns() {
    let _shim_lock = lock_shim_state();

    let mut bool_schema = synthetic_schema(b"b\0");
    assert_synthetic_table_fixed_width_ptr_rejected(&mut bool_schema);

    let mut list_schema = synthetic_schema(b"+l\0");
    list_schema.n_children = 1;
    assert_synthetic_table_fixed_width_ptr_rejected(&mut list_schema);

    let mut struct_schema = synthetic_schema(b"+s\0");
    struct_schema.n_children = 1;
    assert_synthetic_table_fixed_width_ptr_rejected(&mut struct_schema);

    let mut dictionary_values = synthetic_schema(b"u\0");
    let mut dictionary_schema = synthetic_schema(b"i\0");
    dictionary_schema.dictionary = ptr::from_mut(&mut dictionary_values);
    assert_synthetic_table_fixed_width_ptr_rejected(&mut dictionary_schema);
}
