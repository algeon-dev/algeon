use std::{
    ffi::CString,
    os::raw::{c_char, c_void},
    ptr,
};

use crate::{
    helpers::read_string_column_data,
    support::{
        OwnedStream, current_device_resource_raw, last_exec_mr_raw, last_exec_stream_value,
        last_exec_sync_stream_value, lock_shim_state, make_stats_memory_resource, required_exec_options,
        reset_exec_observation, stream_value,
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

#[test]
fn test_arrow_host_column_import_ex_imports_utf8_and_synchronizes() {
    let _shim_state = lock_shim_state();

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
