use std::{
    ffi::CString,
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::{make_table_from_one_column, make_table_view, read_string_column_data},
    support::{
        OwnedStream, current_device_resource_raw, last_exec_mr_raw, last_exec_stream_value,
        last_exec_sync_stream_value, make_stats_memory_resource, required_exec_options, reset_exec_observation,
        stream_value,
    },
};

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

unsafe extern "C" fn release_arrow_schema(schema: *mut ArrowSchema) {
    if !schema.is_null() {
        unsafe { (*schema).release = None };
    }
}

unsafe extern "C" fn release_arrow_array(array: *mut ArrowArray) {
    if !array.is_null() {
        unsafe { (*array).release = None };
    }
}

fn test_arrow_host_column_import_ex_imports_utf8_and_synchronizes() {
    let format = CString::new("u").unwrap();
    let mut schema = ArrowSchema {
        format: format.as_ptr(),
        name: ptr::null(),
        metadata: ptr::null(),
        flags: 0,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_schema),
        private_data: ptr::null_mut(),
    };

    let offsets = [0_i32, 5, 9, 9];
    let chars = b"alphabeta";
    let mut buffers = [ptr::null(), offsets.as_ptr().cast::<c_void>(), chars.as_ptr().cast::<c_void>()];
    let mut array = ArrowArray {
        length: 3,
        null_count: 0,
        offset: 0,
        n_buffers: buffers.len() as i64,
        n_children: 0,
        buffers: buffers.as_mut_ptr(),
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_array),
        private_data: ptr::null_mut(),
    };

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_arrow_host_column_import_ex(
            (&mut schema as *mut ArrowSchema).cast::<c_void>(),
            (&mut array as *mut ArrowArray).cast::<c_void>(),
            ptr::null(),
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());

    reset_exec_observation();
    let status = unsafe {
        cudf_sys::cudf_arrow_host_column_import_ex(
            (&mut schema as *mut ArrowSchema).cast::<c_void>(),
            (&mut array as *mut ArrowArray).cast::<c_void>(),
            &exec,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert!(schema.release.is_some(), "host import must not consume the borrowed ArrowSchema");
    assert!(array.release.is_some(), "host import must not consume the borrowed ArrowArray");
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_ne!(last_exec_mr_raw(), ptr::null_mut());
    assert_ne!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    assert_eq!(read_string_column_data(out), vec!["alpha", "beta", ""]);

    unsafe { cudf_sys::cudf_column_destroy(out) };
}

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "arrow_host_tests::test_arrow_host_column_import_ex_imports_utf8_and_synchronizes",
        test_arrow_host_column_import_ex_imports_utf8_and_synchronizes,
    ),
    (
        "hash_join_create_rejects_nested_build_keys_without_publishing_a_handle",
        hash_join_create_rejects_nested_build_keys_without_publishing_a_handle,
    ),
];

fn hash_join_create_rejects_nested_build_keys_without_publishing_a_handle() {
    let int_format = CString::new("i").unwrap();
    let list_format = CString::new("+l").unwrap();
    let mut child_schema = ArrowSchema {
        format: int_format.as_ptr(),
        name: ptr::null(),
        metadata: ptr::null(),
        flags: 0,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_schema),
        private_data: ptr::null_mut(),
    };
    let mut schema_children = [&mut child_schema as *mut ArrowSchema];
    let mut schema = ArrowSchema {
        format: list_format.as_ptr(),
        name: ptr::null(),
        metadata: ptr::null(),
        flags: 0,
        n_children: 1,
        children: schema_children.as_mut_ptr(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_schema),
        private_data: ptr::null_mut(),
    };
    let values = [1_i32, 2, 3];
    let offsets = [0_i32, 2, 3];
    let mut child_buffers = [ptr::null(), values.as_ptr().cast::<c_void>()];
    let mut child_array = ArrowArray {
        length: 3,
        null_count: 0,
        offset: 0,
        n_buffers: 2,
        n_children: 0,
        buffers: child_buffers.as_mut_ptr(),
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_array),
        private_data: ptr::null_mut(),
    };
    let mut array_children = [&mut child_array as *mut ArrowArray];
    let mut buffers = [ptr::null(), offsets.as_ptr().cast::<c_void>()];
    let mut array = ArrowArray {
        length: 2,
        null_count: 0,
        offset: 0,
        n_buffers: 2,
        n_children: 1,
        buffers: buffers.as_mut_ptr(),
        children: array_children.as_mut_ptr(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_array),
        private_data: ptr::null_mut(),
    };
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let mut column = ptr::null_mut();
    unsafe {
        assert_eq!(
            cudf_sys::cudf_arrow_host_column_import_ex(
                (&mut schema as *mut ArrowSchema).cast(),
                (&mut array as *mut ArrowArray).cast(),
                &exec,
                &mut column
            ),
            cudf_sys::CUDF_SUCCESS
        );
    }
    assert!(!column.is_null());
    let table = make_table_from_one_column(column);
    let view = make_table_view(table);
    let mut out = ptr::null_mut();
    unsafe {
        assert_eq!(
            cudf_sys::cudf_hash_join_create_ex(
                view,
                cudf_sys::CUDF_NULLABLE_JOIN_NO,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                0.5,
                &mut out,
                &exec
            ),
            cudf_sys::CUDF_ERROR_INVALID_ARGUMENT
        );
    }
    assert!(out.is_null());
    unsafe {
        assert_eq!(cudf_sys::cudf_table_view_destroy(view), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_table_destroy(table), cudf_sys::CUDF_SUCCESS);
    }
}
