use super::*;

#[test]
fn test_dictionary_encode_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let col = make_i32_column(&[3, 1, 3, 2, 4, 2, 1, 5]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    crate::support::set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_dictionary_encode_ex(view, cudf_sys::CUDF_TYPE_INT32, &mut out, &exec) };
    let restored_current = current_device_resource_raw();
    unsafe { crate::support::set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary encode must restore the caller's current device resource"
    );
    // Device-domain model: libcudf temporaries resolve through the device's
    // current resource by design, so the sentinel current resource may
    // legitimately observe allocations here.
    let _ = current_before;
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary encode allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}
#[test]
fn test_dictionary_decode_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let dict = make_dictionary_i32_column(&[10, 20, 10, 30, 20, 40]);
    let dict_view = make_column_view(dict);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    crate::support::set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_dictionary_decode_ex(dict_view, &mut out, &exec) };
    let restored_current = current_device_resource_raw();
    unsafe { crate::support::set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary decode must restore the caller's current device resource"
    );
    // Device-domain model: libcudf temporaries resolve through the device's
    // current resource by design, so the sentinel current resource may
    // legitimately observe allocations here.
    let _ = current_before;
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary decode allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(dict_view);
        cudf_sys::cudf_column_destroy(dict);
    }
}

#[test]
fn test_dictionary_set_keys_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let dict = make_dictionary_i32_column(&[1, 2, 1, 3, 2, 4]);
    let dict_view = make_column_view(dict);
    let keys = make_i32_column(&[1, 2, 3, 4, 5]);
    let keys_view = make_column_view(keys);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    crate::support::set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_dictionary_set_keys_ex(dict_view, keys_view, &mut out, &exec) };
    let restored_current = current_device_resource_raw();
    unsafe { crate::support::set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary set_keys must restore the caller's current device resource"
    );
    // Device-domain model: libcudf temporaries resolve through the device's
    // current resource by design, so the sentinel current resource may
    // legitimately observe allocations here.
    let _ = current_before;
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary set_keys allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(keys_view);
        cudf_sys::cudf_column_destroy(keys);
        cudf_sys::cudf_column_view_destroy(dict_view);
        cudf_sys::cudf_column_destroy(dict);
    }
}

#[test]
fn test_dictionary_match_dictionaries_ex_uses_explicit_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left = make_dictionary_i32_column(&[10, 20, 10, 30]);
    let right = make_dictionary_i32_column(&[20, 30, 40, 20]);
    let left_view = make_column_view(left);
    let right_view = make_column_view(right);
    let dict_columns = [left_view, right_view];
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();

    let current_borrowed = current_mr.borrowed();
    crate::support::set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");
    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries_ex(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
            &exec,
        )
    };
    let restored_current = current_device_resource_raw();
    unsafe { crate::support::set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, dict_columns.len() as i32);
    assert!(!out_columns.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "dictionary match_dictionaries must restore the caller's current device resource"
    );
    // Device-domain model: libcudf temporaries resolve through the device's
    // current resource by design, so the sentinel current resource may
    // legitimately observe allocations here.
    let _ = current_before;
    assert!(
        target_mr.allocation_count() > target_before,
        "dictionary match_dictionaries allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    let matched = unsafe { std::slice::from_raw_parts(out_columns, out_count as usize) };
    unsafe {
        for &col in matched {
            cudf_sys::cudf_column_destroy(col);
        }
        cudf_sys::cudf_free_column_array(out_columns);
        cudf_sys::cudf_column_view_destroy(right_view);
        cudf_sys::cudf_column_view_destroy(left_view);
        cudf_sys::cudf_column_destroy(right);
        cudf_sys::cudf_column_destroy(left);
    }
}
